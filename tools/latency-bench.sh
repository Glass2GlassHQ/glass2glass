#!/usr/bin/env bash
# Receive-to-present latency, g2g vs GStreamer, on the same live RTSP feed.
#
# Needs the standing feed from README "A standing RTSP feed" (mediamtx +
# ffmpeg publisher) and gst-launch-1.0.
#
# CONTAINED=1 (default) runs each side against a throwaway display server
# it spawns itself: a headless mutter for g2g's WaylandSink, Xvfb +
# ximagesink for gst. gst's waylandsink commits its first buffer before
# acking the xdg configure, which segfaults gnome-shell 49 (three desktop
# crashes on 2026-08-31), and under headless mutter the same commit stalls
# the sink after one frame, hence the X11 fallback for the gst side.
# CONTAINED=0 uses the session compositor and waylandsink on both sides:
# real glass numbers, but it can take the desktop down.
#
# The two sides do not report the same span:
#   g2g: packet arrival at RtspSrc -> compositor frame callback, per frame,
#        from the wayland_smoke harness histogram.
#   gst: per-element latency table from the latency tracer, plus the sum of
#        element means as the path total. The tracer's pipeline-level
#        records (arrival -> sink render) never make it through rtpbin on
#        GStreamer 1.26, and no sink appears in the element records, so the
#        gst total excludes sink render and present.
# The tracer needs flags=pipeline+element: flags=pipeline alone emits
# nothing on this build.
#
#   DECODER=software|nvdec FRAMES=300 tools/latency-bench.sh
set -euo pipefail

URL="${G2G_RTSP_TEST_URL:-rtsp://localhost:8554/pattern}"
FRAMES="${FRAMES:-300}"
DECODER="${DECODER:-software}"
CONTAINED="${CONTAINED:-1}"
FPS=30
G2G_WAYLAND_NAME=g2g-latency-bench
XVFB_DISPLAY=:99

cleanup() {
    [[ -n "${MUTTER_PID:-}" ]] && kill "$MUTTER_PID" 2>/dev/null || true
    [[ -n "${XVFB_PID:-}" ]] && kill "$XVFB_PID" 2>/dev/null || true
    rm -f "${GST_LOG:-}"
}
trap cleanup EXIT

if [[ "$CONTAINED" == 1 ]]; then
    mutter --headless --no-x11 --wayland-display="$G2G_WAYLAND_NAME" \
        --virtual-monitor 1280x720 >/dev/null 2>&1 &
    MUTTER_PID=$!
    Xvfb "$XVFB_DISPLAY" -screen 0 1280x720x24 >/dev/null 2>&1 &
    XVFB_PID=$!
    sleep 2
    G2G_ENV=(WAYLAND_DISPLAY="$G2G_WAYLAND_NAME")
    GST_ENV=(DISPLAY="$XVFB_DISPLAY")
    GST_SINK="ximagesink"
else
    G2G_ENV=()
    GST_ENV=()
    GST_SINK="waylandsink"
fi

echo "== g2g: RtspSrc -> FfmpegH264Dec(${DECODER}) -> WaylandSink, ${FRAMES} frames =="
env "${G2G_ENV[@]}" \
    G2G_RTSP_TEST_URL="$URL" G2G_TARGET_FRAMES="$FRAMES" G2G_DECODER="$DECODER" \
    cargo test --release -p g2g-plugins \
    --features "rtsp ffmpeg wayland-sink" \
    --test wayland_smoke -- --ignored --nocapture 2>&1 |
    grep -E "glass-to-glass latency|effective_fps"

case "$DECODER" in
software) GST_DEC="avdec_h264" ;;
nvdec) GST_DEC="nvh264dec" ;;
*)
    echo "unknown DECODER=$DECODER" >&2
    exit 1
    ;;
esac

# protocols=tcp matches g2g's RtspSrc default (TCP interleaved).
echo "== gst: rtspsrc latency=0 protocols=tcp ! rtph264depay ! h264parse ! ${GST_DEC} ! videoconvert ! ${GST_SINK} =="
GST_LOG="$(mktemp)"
# The feed is unbounded, so run for the same wall-clock span as FRAMES covers.
DURATION=$((FRAMES / FPS + 3))
env "${GST_ENV[@]}" \
    GST_TRACERS='latency(flags=pipeline+element)' GST_DEBUG=GST_TRACER:7 \
    timeout "$DURATION" gst-launch-1.0 -q \
    rtspsrc location="$URL" latency=0 protocols=tcp ! rtph264depay ! h264parse ! \
    "$GST_DEC" ! videoconvert ! "$GST_SINK" 2>"$GST_LOG" || true

python3 - "$GST_LOG" <<'EOF'
import re, sys
from collections import defaultdict

per_element = defaultdict(list)
for line in open(sys.argv[1]):
    m = re.search(
        r"element-latency, .*?element=\(string\)(\S+?), "
        r".*?time=\(guint64\)(\d+)", line)
    if m:
        per_element[m.group(1)].append(int(m.group(2)))
if not per_element:
    sys.exit("no element-latency tracer records captured")

total_mean = 0.0
for name, ns in per_element.items():
    ns.sort()
    mean = sum(ns) / len(ns) / 1e6
    p95 = ns[min(len(ns) - 1, int(len(ns) * 95 / 100))] / 1e6
    total_mean += mean
    print(f"  {name:<20} n={len(ns):<5} mean={mean:7.2f}ms p95={p95:7.2f}ms")
print(f"path total (sum of element means, no sink render): {total_mean:.1f}ms")
EOF
