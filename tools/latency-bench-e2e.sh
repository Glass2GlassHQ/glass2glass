#!/usr/bin/env bash
# Same-metric receive latency, g2g vs GStreamer, on one burned-in clock.
#
# Unlike tools/latency-bench.sh (which compares a g2g arrival-to-present
# histogram against a sum of gst tracer element means), both sides here are
# measured by the same binary over the same span.
#
# One g2g publisher generates the stream. `timestampburn` writes
# CLOCK_MONOTONIC nanoseconds into the luma plane just before the encoder, so
# every frame carries the instant it left. Each consumer decodes to raw I420 and
# pipes it into `g2g-latency-reader`, which subtracts the burned value from its
# own CLOCK_MONOTONIC. Same machine, same clock, same reader code, no display
# server anywhere.
#
# The span each sample covers:
#   burn (publisher, pre-encode) -> x264 encode -> RTP over TCP-interleaved RTSP
#   on loopback -> depayload -> h264 parse -> decode -> convert to I420 -> pipe
#   read by the reader.
# Render and present are excluded. The publisher half of that span is one
# identical g2g pipeline for both sides, so the absolute number is not a
# framework figure on its own: the difference between the two lines is, and that
# difference is the consumer stack (depay, parse, decode, convert, hand-off).
#
# Each side gets its own publisher run, so neither pays for the other's
# connection. `--warmup` frames are read and discarded on both sides, covering
# decoder start-up and the frames the publisher burned while `rtspserversink`
# was still waiting for a player.
#
# PARSE=1 (default) puts an `h264parse` between depayload and decode on both
# sides, the way a pasted pipeline is normally written. PARSE=0 drops it from
# both lines. The two should read the same on both stacks: a parser fed whole
# access units passes each one straight through.
#
#   DECODER=software|nvdec FRAMES=300 WARMUP=30 PORT=8555 PARSE=0 \
#     NETEM="delay 20ms loss 1%" tools/latency-bench-e2e.sh
#
# NETEM (any tc netem argument list) reruns everything inside one unprivileged
# network namespace (`unshare -rn`) with the qdisc on `lo`, so publisher,
# consumer, and reader all see the impairment. No veth pair is needed: the whole
# bench is loopback.
#
# PORT defaults to 8555 to stay clear of the standing 8554 feed.
set -euo pipefail

FRAMES="${FRAMES:-300}"
WARMUP="${WARMUP:-30}"
DECODER="${DECODER:-software}"
PORT="${PORT:-8555}"
NETEM="${NETEM:-}"
PARSE="${PARSE:-1}"
WIDTH=1280
HEIGHT=720
FPS=30
BITRATE_BPS=4000000
URL="rtsp://127.0.0.1:${PORT}/pattern"
FEATURES="latency-bench,rtsp,rtsp-server,ffmpeg"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCH="$REPO/target/release/g2g-launch"
READER="$REPO/target/release/g2g-latency-reader"

case "$DECODER" in
software)
    G2G_DEC="ffmpegdec backend=software"
    GST_DEC="avdec_h264"
    ;;
nvdec)
    G2G_DEC="ffmpegdec backend=nvdec-cuvid"
    GST_DEC="nvh264dec"
    ;;
*)
    echo "unknown DECODER=$DECODER (software|nvdec)" >&2
    exit 1
    ;;
esac

if [[ "$PARSE" == 1 ]]; then
    PARSER=(h264parse !)
else
    PARSER=()
fi

# Build before entering the namespace: it has no network for a fetch.
if [[ "${G2G_BENCH_IN_NETNS:-0}" != 1 ]]; then
    cargo build --release -p g2g-plugins --features "$FEATURES" \
        --bin g2g-launch --bin g2g-latency-reader
fi

if [[ -n "$NETEM" && "${G2G_BENCH_IN_NETNS:-0}" != 1 ]]; then
    echo "== netem on lo: $NETEM =="
    export G2G_BENCH_IN_NETNS=1
    exec unshare -rn bash -c '
        set -euo pipefail
        ip link set lo up
        tc qdisc add dev lo root netem '"$NETEM"'
        exec "$@"
    ' bash "${BASH_SOURCE[0]}" "$@"
fi

PUB_LOG="$(mktemp)"
CONS_LOG="$(mktemp)"
PUB_PID=""
cleanup() {
    [[ -n "$PUB_PID" ]] && kill "$PUB_PID" 2>/dev/null || true
    rm -f "$PUB_LOG" "$CONS_LOG"
}
trap cleanup EXIT

# Enough frames for the warmup, the measured run, and the connect handshake.
PUBLISH_FRAMES=$((FRAMES + WARMUP + 2 * FPS))

start_publisher() {
    "$LAUNCH" -q \
        videotestsrc pattern=ball width="$WIDTH" height="$HEIGHT" \
        framerate="$FPS/1" num-buffers="$PUBLISH_FRAMES" \
        ! clocksync \
        ! videoconvert ! "video/x-raw,format=I420" \
        ! timestampburn \
        ! x264enc backend=software bitrate="$BITRATE_BPS" \
        ! rtspserversink port="$PORT" >"$PUB_LOG" 2>&1 &
    PUB_PID=$!
    for _ in $(seq 1 100); do
        if ss -ltnH "sport = :$PORT" | grep -q .; then
            return 0
        fi
        sleep 0.1
    done
    echo "publisher never listened on $PORT" >&2
    cat "$PUB_LOG" >&2
    exit 1
}

stop_publisher() {
    kill "$PUB_PID" 2>/dev/null || true
    wait "$PUB_PID" 2>/dev/null || true
    PUB_PID=""
}

# Run one consumer command (reading the shared publisher, writing raw I420 on
# stdout) into the reader. The consumer dies of SIGPIPE once the reader has its
# frames, which is expected, so only the reader's status is checked.
run_side() {
    local label="$1"
    shift
    start_publisher
    set +e
    "$@" 2>"$CONS_LOG" |
        "$READER" --width "$WIDTH" --height "$HEIGHT" \
            --frames "$FRAMES" --warmup "$WARMUP" --label "$label"
    local reader_rc=${PIPESTATUS[1]}
    set -e
    stop_publisher
    if [[ $reader_rc -ne 0 ]]; then
        echo "$label consumer produced no readable frames" >&2
        tail -20 "$CONS_LOG" >&2
        exit 1
    fi
}

set +o pipefail

echo "== ${FRAMES} frames, ${WIDTH}x${HEIGHT}@${FPS}, decoder=${DECODER}," \
    "warmup=${WARMUP}, parse=${PARSE} =="

# shellcheck disable=SC2086  # $G2G_DEC carries an element plus its property
run_side g2g "$LAUNCH" -q \
    rtspsrc location="$URL" protocols=tcp \
    ! "${PARSER[@]}" $G2G_DEC \
    ! videoconvert ! "video/x-raw,format=I420" \
    ! fdsink fd=1

run_side gst gst-launch-1.0 -q \
    rtspsrc location="$URL" latency=0 protocols=tcp \
    ! rtph264depay ! "${PARSER[@]}" "$GST_DEC" \
    ! videoconvert ! video/x-raw,format=I420 \
    ! fdsink fd=1
