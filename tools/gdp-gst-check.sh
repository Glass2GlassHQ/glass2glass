#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# GStreamer's gdpdepay reads g2g's GDP streams back to the frames g2g wrote, and g2g reads GStreamer's
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

cargo build -q --manifest-path "$repo/Cargo.toml" -p g2g-plugins --bin g2g-launch --features std,tcp
g2g="$repo/target/debug/g2g-launch"

failed=0
compare() {
    if cmp -s "$1" "$2"; then
        echo "$3: identical"
    else
        echo "$3: DIFFERS"
        failed=1
    fi
}
check() {
    local name=$1 source=$2 consumer=$3
    "$g2g" "$source ! filesink location=$work/$name.raw" > /dev/null
    "$g2g" "$source ! gdppay ! filesink location=$work/$name.g2g.gdp" > /dev/null
    gst-launch-1.0 -q filesrc location="$work/$name.g2g.gdp" ! gdpdepay ! filesink location="$work/$name.gst.raw"
    compare "$work/$name.raw" "$work/$name.gst.raw" "$name g2g to gst"
    # a converter only negotiates caps gst accepts as complete
    gst-launch-1.0 -q filesrc location="$work/$name.g2g.gdp" ! gdpdepay ! "$consumer" ! fakesink
    gst-launch-1.0 -q $source ! gdppay ! filesink location="$work/$name.gst.gdp"
    gst-launch-1.0 -q $source ! filesink location="$work/$name.gst-source.raw"
    "$g2g" "filesrc location=$work/$name.gst.gdp ! gdpdepay ! filesink location=$work/$name.g2g.raw" > /dev/null
    compare "$work/$name.gst-source.raw" "$work/$name.g2g.raw" "$name gst to g2g"
}

for format in I420 NV12 RGBA BGRx GRAY8 YUY2; do
    check "$format" "videotestsrc num-buffers=3 pattern=ball ! video/x-raw,format=$format,width=16,height=4,framerate=30/1" videoconvert
done
check S16LE "audiotestsrc num-buffers=3 ! audio/x-raw,format=S16LE,rate=48000,channels=2" audioconvert

# g2g listens, gst dials and sends
port=$(uv run --no-project python -I -c "import socket; s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])")
nv12="video/x-raw,format=NV12,width=16,height=4,framerate=30/1"
"$g2g" "tcpserversrc host=127.0.0.1 port=$port caps=application/x-gdp,original-caps=(caps)[$nv12] ! gdpdepay ! filesink location=$work/tcp.g2g.raw" > /dev/null &
receiver=$!
sleep 1
gst-launch-1.0 -q videotestsrc num-buffers=3 pattern=ball ! "$nv12" ! gdppay ! tcpclientsink host=127.0.0.1 port="$port"
wait "$receiver"
gst-launch-1.0 -q videotestsrc num-buffers=3 pattern=ball ! "$nv12" ! filesink location="$work/tcp.gst.raw"
compare "$work/tcp.gst.raw" "$work/tcp.g2g.raw" "NV12 gst to g2g over tcp"
exit $failed
