#!/usr/bin/env bash
# Push-tax benchmark (M870): price g2g's push model against GStreamer's pull
# model on the same batch-demux line, `filesrc ! tsdemux ! h264parse ! fakesink`.
#
# g2g pushes every chunk across a bounded per-edge channel: each hop costs a
# send/recv wakeup plus a boxed future poll. gst-launch runs this same line in
# pull mode, where basesrc's loop task calls into the demuxer directly and no
# chunk crosses a thread boundary. Same file, same work, so the throughput gap
# is the per-chunk transport overhead, which is what this measures.
#
# Both engines are timed externally (wall clock around the process) for
# symmetry, so process startup and teardown are inside both numbers.
# gst-launch's own "Execution ended after" line is ignored for that reason.
#
# What the ratio does and does not say: it prices the whole gap, and transport
# is only one term in it. The per-element measured proc times from the g2g run
# are printed alongside the ratio so the gap can be attributed, since element
# CPU cost (the TS depacketizer, the start-code scan) can dominate the
# per-chunk transport cost by an order of magnitude.
#
# Usage: tools/pushtax-bench.sh
# Requires: ffmpeg + ffprobe (fixture + AU-count oracle), gst-launch-1.0.
# Env: G2G_PUSHTAX_DIR overrides the fixture / results directory,
#      G2G_PUSHTAX_ITERS overrides the timed iteration count (default 5).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${G2G_PUSHTAX_DIR:-$ROOT/target/pushtax}"
ITERS="${G2G_PUSHTAX_ITERS:-5}"
BIN="$ROOT/target/release/g2g-launch"
FIXTURE="$DIR/pushtax-1080p30-60s.ts"
RESULTS="$DIR/pushtax-results.tsv"

for tool in ffmpeg ffprobe gst-launch-1.0; do
  command -v "$tool" >/dev/null 2>&1 \
    || { echo "SKIP: $tool not found, cannot run the push-tax comparison"; exit 0; }
done

mkdir -p "$DIR"

if [ ! -s "$FIXTURE" ]; then
  echo "== synthesizing the fixture (60 s 1080p30 H.264 in MPEG-TS) =="
  ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i testsrc2=size=1920x1080:rate=30:duration=60 \
    -c:v libx264 -preset veryfast -b:v 12M -g 60 -pix_fmt yuv420p \
    -f mpegts "$FIXTURE"
fi
BYTES="$(stat -c %s "$FIXTURE")"
echo "fixture: $FIXTURE ($BYTES bytes)"

echo "== building g2g-launch (release; measuring a debug build is meaningless) =="
cargo build --release --manifest-path "$ROOT/Cargo.toml" \
  -p g2g-plugins --features std --bin g2g-launch

# The `.ts` extension types filesrc's byte stream, so no bytestream-format is
# needed to reach tsdemux.
G2G_LINE="filesrc location=$FIXTURE ! tsdemux ! h264parse ! fakesink"

# Shared oracle: gst-launch has no cheap per-buffer count, so the expected
# access-unit count comes from ffprobe and is asserted against g2g's own
# consumed count. A demux that quietly dropped the tail would still "pass" a
# timing run, hence the check before any timing.
echo "== correctness: the g2g line must reach EOS with the full AU count =="
EXPECT="$(ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=nb_read_frames -of csv=p=0 "$FIXTURE" | head -n1 | tr -d '\r')"
CHECK_LOG="$DIR/g2g-correctness.log"
"$BIN" "$G2G_LINE" >"$CHECK_LOG" 2>&1 \
  || { echo "FAIL: the g2g line did not run:"; cat "$CHECK_LOG"; exit 1; }
CONSUMED="$(sed -n 's/.*consumed \([0-9]*\).*/\1/p' "$CHECK_LOG" | head -n1)"
[ -n "$CONSUMED" ] && [ "$CONSUMED" -gt 0 ] \
  || { echo "FAIL: no frame count in the g2g run summary:"; cat "$CHECK_LOG"; exit 1; }
[ "$CONSUMED" = "$EXPECT" ] \
  || { echo "FAIL: h264parse pushed $CONSUMED access units, ffprobe counts $EXPECT"; exit 1; }
echo "PASS: $CONSUMED access units through h264parse to fakesink (ffprobe agrees)"

echo "results: $RESULTS (appended as each iteration lands)"
[ -s "$RESULTS" ] || printf 'utc\tengine\titer\telapsed_s\tbytes\tmbps\tload1\n' >>"$RESULTS"

G2G_MBPS=()
GST_MBPS=()

# Times one process, appends the row, echoes the MB/s. `iter` is `warmup` for
# the discarded first run of each engine. load1 records the machine's 1-minute
# load average, so a row measured against a busy host is visible afterwards.
run_one() { # engine iter cmd...
  local engine="$1" iter="$2"
  shift 2
  local log="$DIR/$engine-$iter.log" start end elapsed mbps load
  load="$(cut -d' ' -f1 /proc/loadavg)"
  start="$(date +%s%N)"
  "$@" >"$log" 2>&1 || { echo "FAIL: $engine iteration $iter exited nonzero:"; cat "$log"; exit 1; }
  end="$(date +%s%N)"
  elapsed="$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.4f", (e - s) / 1e9 }')"
  mbps="$(awk -v b="$BYTES" -v t="$elapsed" 'BEGIN { printf "%.1f", b / 1048576 / t }')"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%FT%TZ)" "$engine" "$iter" "$elapsed" "$BYTES" "$mbps" "$load" >>"$RESULTS"
  # progress to stderr, the value to stdout: callers capture the latter.
  echo "  $engine $iter: ${elapsed}s, ${mbps} MB/s (load1 $load)" >&2
  echo "$mbps"
}

g2g_run() { run_one g2g "$1" "$BIN" "$G2G_LINE"; }
gst_run() {
  run_one gst "$1" gst-launch-1.0 -q \
    filesrc "location=$FIXTURE" ! tsdemux ! h264parse ! fakesink
}

echo "== warm-up (discarded: first run pays the page-cache miss on the fixture) =="
g2g_run warmup >/dev/null
gst_run warmup >/dev/null

echo "== $ITERS timed iterations each, interleaved =="
for i in $(seq 1 "$ITERS"); do
  G2G_MBPS+=("$(g2g_run "$i" | tail -n1)")
  GST_MBPS+=("$(gst_run "$i" | tail -n1)")
done

# Same input for both, unchanged across the whole run.
[ "$(stat -c %s "$FIXTURE")" = "$BYTES" ] \
  || { echo "FAIL: the fixture changed size mid-run"; exit 1; }

median() { printf '%s\n' "$@" | sort -g | awk '{ v[NR] = $1 } END { print v[int((NR + 1) / 2)] }'; }
G2G_MED="$(median "${G2G_MBPS[@]}")"
GST_MED="$(median "${GST_MBPS[@]}")"
RATIO="$(awk -v a="$GST_MED" -v b="$G2G_MED" 'BEGIN { printf "%.2f", a / b }')"

SUMMARY="push tax: g2g (push) ${G2G_MED} MB/s vs gst-launch (pull) ${GST_MED} MB/s on ${BYTES} bytes, ratio ${RATIO}x"
echo
echo "$SUMMARY"
echo "g2g-side attribution (measured, from $CHECK_LOG):"
sed -n '/per-element/,/^  run:/p' "$CHECK_LOG"

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "### Push-tax benchmark (filesrc ! tsdemux ! h264parse ! fakesink)"
    echo
    echo '```'
    echo "$SUMMARY"
    echo '```'
  } >>"$GITHUB_STEP_SUMMARY"
fi
