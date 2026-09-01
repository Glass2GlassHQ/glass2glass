#!/usr/bin/env bash
# Offline throughput and CPU cost, g2g vs GStreamer, on one generated fixture.
#
# Both lines decode the same H.264 file with libavcodec, convert to I420, and
# throw the frames away, as fast as the machine allows. The codec work is
# byte-identical, so what differs is the framework around it: packet hand-off,
# scheduling, and per-frame copies. /usr/bin/time -v reports wall, CPU seconds,
# and peak RSS; fps is the frame count over wall.
#
# The default race is each stack as shipped, and both stacks thread the same way
# on it: nothing in a file pipeline is live, so `avdec_*` and `ffmpegdec` alike
# take frame threading, which pipelines whole pictures across the cores. Off a
# live source both fall back to slice threading, where libavcodec releases each
# picture as soon as it is done instead of holding `thread_count - 1` of them.
#
# THREAD_TYPE=frame asks g2g's decoder for frame threading whatever the pipeline,
# which on this fixture is what the default resolves to anyway. PIN_THREADS=1
# instead pins both decoders to one thread, which is the comparison that isolates
# framework overhead from codec parallelism.
#
# Encode is deliberately absent. Matching it needs one encoder library driven
# with the same settings on both sides, and this host has neither: GStreamer's
# libx264 wrapper (`x264enc`) is not packaged for Fedora, and the one encoder
# both stacks can reach, libvpx, exposes no deadline / cpu-used knob on the g2g
# side to match `vp8enc` against. A decode race is the part that can be made
# honest.
#
# THREADS=1 (default) runs the g2g side on the thread-per-arm runner, matching
# gst-launch, which puts every element on its own streaming thread. THREADS=0
# runs g2g's cooperative single-thread runner instead: the same total work on
# one core, which is the comparison to read the CPU-seconds column against.
#
# The fixture is generated once by ffmpeg and cached, so repeat runs are free.
#
#   FRAMES / geometry are the fixture's; change them together.
#   THREAD_TYPE=frame PIN_THREADS=1 THREADS=0 tools/throughput-bench.sh
set -euo pipefail

THREADS="${THREADS:-1}"
PIN_THREADS="${PIN_THREADS:-0}"
THREAD_TYPE="${THREAD_TYPE:-auto}"
FIXTURE_DIR="${FIXTURE_DIR:-/tmp/claude-1000/g2g-bench-fixtures}"
DURATION_S=60
FPS=30
WIDTH=1920
HEIGHT=1080
FIXTURE_BITRATE=8000k
FRAMES=$((DURATION_S * FPS))
FIXTURE="$FIXTURE_DIR/testsrc-${WIDTH}x${HEIGHT}p${FPS}-${DURATION_S}s.h264"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCH="$REPO/target/release/g2g-launch"
FEATURES="ffmpeg,multi-thread"

if [[ "$THREADS" == 1 ]]; then
    RUNNER=(--threads)
else
    RUNNER=()
fi

# `max-threads` is the same property name on both decoders; 0 is auto on both.
if [[ "$PIN_THREADS" == 1 ]]; then
    DECODE_THREADS=1
else
    DECODE_THREADS=0
fi

cargo build --release -p g2g-plugins --features "$FEATURES" --bin g2g-launch

if [[ ! -s "$FIXTURE" ]]; then
    echo "== generating fixture: $FIXTURE =="
    mkdir -p "$FIXTURE_DIR"
    ffmpeg -y -loglevel error \
        -f lavfi -i "testsrc=size=${WIDTH}x${HEIGHT}:rate=${FPS}:duration=${DURATION_S}" \
        -c:v libx264 -preset medium -b:v "$FIXTURE_BITRATE" -g "$((FPS * 2))" \
        -pix_fmt yuv420p -f h264 "$FIXTURE"
fi

TIME_LOG="$(mktemp)"
trap 'rm -f "$TIME_LOG"' EXIT

# `/usr/bin/time -v` writes its report on stderr; the run's own stderr goes with
# it, so the report is picked out by field name rather than by position.
report() {
    local label="$1"
    local wall user sys rss
    wall=$(grep -oP 'Elapsed \(wall clock\) time.*?: \K.*' "$TIME_LOG")
    user=$(grep -oP 'User time \(seconds\): \K.*' "$TIME_LOG")
    sys=$(grep -oP 'System time \(seconds\): \K.*' "$TIME_LOG")
    rss=$(grep -oP 'Maximum resident set size \(kbytes\): \K.*' "$TIME_LOG")
    python3 - "$label" "$wall" "$user" "$sys" "$rss" "$FRAMES" <<'EOF'
import sys
label, wall, user, sys_s, rss_kb, frames = sys.argv[1:7]
parts = [float(p) for p in wall.split(":")]
seconds = 0.0
for p in parts:
    seconds = seconds * 60 + p
cpu = float(user) + float(sys_s)
print(
    f"  {label:<4} wall={seconds:6.2f}s  fps={int(frames)/seconds:7.1f}  "
    f"cpu={cpu:6.2f}s (user {float(user):.2f} + sys {float(sys_s):.2f})  "
    f"max-rss={float(rss_kb)/1024:.0f} MiB"
)
EOF
}

run_side() {
    local label="$1"
    shift
    /usr/bin/time -v -o "$TIME_LOG" "$@" >/dev/null 2>&1
    report "$label"
}

echo "== decode + convert to I420, ${FRAMES} frames of ${WIDTH}x${HEIGHT}," \
    "libavcodec both sides, g2g runner threads=${THREADS}," \
    "decoder max-threads=${DECODE_THREADS} thread-type=${THREAD_TYPE} =="

run_side g2g "$LAUNCH" -q "${RUNNER[@]}" \
    filesrc location="$FIXTURE" \
    ! h264parse ! ffmpegdec backend=software \
    max-threads="$DECODE_THREADS" thread-type="$THREAD_TYPE" \
    ! videoconvert ! "video/x-raw,format=I420" \
    ! fakesink

run_side gst gst-launch-1.0 -q \
    filesrc location="$FIXTURE" \
    ! h264parse ! avdec_h264 max-threads="$DECODE_THREADS" \
    ! videoconvert ! video/x-raw,format=I420 \
    ! fakesink sync=false
