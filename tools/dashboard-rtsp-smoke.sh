#!/usr/bin/env bash
# Validate the live dashboard (`g2g-launch --observe`) against a real RTSP feed:
# the page is served, telemetry names every node with populated latency
# histograms, each edge carries its negotiated caps and live traffic counters,
# the frame journey assembles, bus events reach the socket, and the edge content
# tap yields a thumbnail on a raw edge / a codec card on a compressed one.
#
# Prerequisites:
#   - an RTSP server publishing H.264 (default rtsp://127.0.0.1:8554/pattern,
#     e.g. mediamtx with an ffmpeg publisher). Check it with:
#       ffprobe -v error -rtsp_transport tcp -i <url>
#   - python3 with the `websockets` package (pip/uv install websockets).
# Needs a live network peer, so this is validated locally, not in CI.
#
# Usage: tools/dashboard-rtsp-smoke.sh [rtsp-url] [port]
set -euo pipefail

cd "$(dirname "$0")/.."

url="${1:-rtsp://127.0.0.1:8554/pattern}"
port="${2:-8787}"
# 150 frames at 30 fps: ~5 s of stream, so the run ends on its own.
num_buffers=150
# Ceiling for the whole capture, well past the bounded run plus RTSP setup.
capture_secs=90

python3 -c "import websockets" 2>/dev/null || {
  echo "FAIL: python3 needs the 'websockets' package"; exit 1; }

echo "== building g2g-launch (rtsp + ffmpeg + observe) =="
cargo build -q -p g2g-plugins --features "rtsp ffmpeg observe" --bin g2g-launch

work="$(mktemp -d)"
# Run from a copy: a concurrent cargo build in this tree relinks
# target/debug/g2g-launch with a different feature set mid-run.
cp target/debug/g2g-launch "$work/g2g-launch"

pipeline="rtspsrc location=$url num-buffers=$num_buffers ! h264parse ! ffmpegdec ! videoconvert ! fakesink"
echo "== running: $pipeline =="
"$work/g2g-launch" --observe "$port" "$pipeline" >"$work/launch.log" 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true; rm -rf "$work"' EXIT

echo "== HTTP GET / =="
code=""
for _ in $(seq 1 60); do
  code=$(curl -s -o "$work/index.html" -w "%{http_code}" "http://127.0.0.1:$port/" || true)
  [ "$code" = "200" ] && break
  sleep 0.5
done
[ "$code" = "200" ] || { echo "FAIL: dashboard page returned '$code'"; cat "$work/launch.log"; exit 1; }
grep -q "<title>" "$work/index.html" || { echo "FAIL: page is not the dashboard"; exit 1; }
echo "  PASS 200, $(wc -c <"$work/index.html") bytes"

echo "== WebSocket telemetry / events / previews =="
set +e
python3 - "ws://127.0.0.1:$port/" "$capture_secs" <<'PY'
import asyncio, json, re, sys
import websockets

url, deadline = sys.argv[1], float(sys.argv[2])
frames, previews, events, subs = [], {}, [], {}

async def capture():
    async with websockets.connect(url, max_size=None) as ws:
        loop = asyncio.get_running_loop()
        end = loop.time() + deadline
        while loop.time() < end:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=end - loop.time()))
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                return
            if msg["type"] == "telemetry":
                frames.append(msg)
                # Tap one compressed and one raw edge as soon as caps name them.
                for index, edge in enumerate((msg.get("edges") or [])):
                    caps = edge.get("caps") or ""
                    kind = "compressed" if caps.startswith("video/x-h264") else \
                           "raw" if caps.startswith("video/x-raw") else None
                    if kind and kind not in subs:
                        subs[kind] = index
                        await ws.send(json.dumps({"type": "subscribe", "edge": index}))
            elif msg["type"] == "preview":
                previews.setdefault(msg["edge"], msg["preview"])
            elif msg["type"] == "event":
                events.append(msg)

asyncio.run(capture())

fails = []
def check(ok, label, evidence=""):
    print(("  PASS " if ok else "  FAIL ") + label + (("  " + evidence) if evidence else ""))
    if not ok:
        fails.append(label)

check(bool(frames), "telemetry frames arrived", f"n={len(frames)}")
if not frames:
    sys.exit(1)
last = frames[-1]

nodes = last["nodes"]
roles = [n["role"] for n in nodes]
check(len(nodes) == 5 and roles[0] == "source" and roles[-1] == "sink",
      "every node named with its role", ", ".join(f"{n['name']}={n['role']}" for n in nodes))
working = [n for n in nodes if n["role"] != "source"]
check(all((n["proc"] or {}).get("count", 0) > 0 for n in working),
      "per-node latency histograms counted",
      ", ".join(f"{n['name']}:n={n['proc']['count']},p50={n['proc']['p50_ns']}ns" for n in working))
check(all((n["transit"] or {}).get("count", 0) > 0 for n in working),
      "per-node transit (input-link wait) measured",
      ", ".join(f"{n['name']}:p50={n['transit']['p50_ns']}ns" for n in working))
check(all(n["push_wait"] for n in working if n["role"] == "transform"),
      "per-transform push-wait measured")

edges = last["edges"]
check(len(edges) == 4 and all(e["packets"] > 0 and e["bytes"] > 0 for e in edges),
      "every edge carried traffic",
      ", ".join(f"{e['from']}->{e['to']}:{e['packets']}pkt/{e['bytes']}B" for e in edges))
compressed = [e for e in edges if (e["caps"] or "").startswith("video/x-h264")]
raw = [e for e in edges if (e["caps"] or "").startswith("video/x-raw")]
def geometry(caps):
    m = re.search(r"width=(\d+),height=(\d+)", caps or "")
    return m.groups() if m else None
check(bool(compressed) and bool(raw), "compressed and raw edges both labelled")
if compressed and raw:
    cg, rg = geometry(compressed[0]["caps"]), geometry(raw[0]["caps"])
    check(cg is not None and cg == rg, "stream geometry survives the decode",
          f"h264 {compressed[0]['caps']} | raw {raw[0]['caps']}")

journey = last.get("journey")
check(journey is not None and len(journey["stages"]) >= 3 and journey["total_ns"] > 0,
      "frame journey assembled",
      "" if not journey else
      f"seq={journey['sequence']} total={journey['total_ns']}ns floor={journey['floor_ns']}ns "
      + "->".join(s["name"] for s in journey["stages"]))

# The runner posts no `Eos` on the bus, so the event path is asserted on the
# per-link `Buffering` levels every run reports.
kinds = sorted({e["kind"] for e in events})
check(bool(events), "bus events reached the socket", f"n={len(events)} kinds={kinds}")

card = previews.get(subs.get("compressed"))
check(card is not None and card.get("kind") == "compressed" and card.get("bytes", 0) > 0
      and card.get("frame") in ("key", "delta"),
      "compressed edge tap returns a codec card", json.dumps(card))
thumb = previews.get(subs.get("raw"))
check(thumb is not None and thumb.get("kind") == "video" and len(thumb.get("rgba") or []) >= 4,
      "raw edge tap returns a thumbnail",
      "" if not thumb else f"{thumb['w']}x{thumb['h']}, {len(thumb['rgba'])} rgba bytes")

sys.exit(1 if fails else 0)
PY
status=$?
set -e

wait "$pid" 2>/dev/null || true
tail -n 3 "$work/launch.log"

[ "$status" = 0 ] && echo "== dashboard RTSP smoke passed ==" || { echo "== dashboard RTSP smoke FAILED =="; exit 1; }
