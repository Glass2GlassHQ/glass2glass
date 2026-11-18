#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# GDP streams written by GStreamer's gdppay, each next to the frames it holds
set -euo pipefail

fixtures="$(cd "$(dirname "$0")/.." && pwd)/g2g-plugins/tests/fixtures"

fixture() {
    local name=$1 source=$2 pay=$3
    # shellcheck disable=SC2086
    gst-launch-1.0 -q $source ! filesink location="$fixtures/$name.raw"
    # shellcheck disable=SC2086
    gst-launch-1.0 -q $source ! $pay ! filesink location="$fixtures/$name.gdp"
}

fixture gdp_nv12_64x48 "videotestsrc num-buffers=2 pattern=ball ! video/x-raw,format=NV12,width=64,height=48,framerate=30/1" "gdppay crc-payload=true"
# audiotestsrc also sends a tag event, which gdppay writes between the segment and the first buffer
fixture gdp_s16_48k "audiotestsrc num-buffers=2 samplesperbuffer=64 ! audio/x-raw,format=S16LE,rate=48000,channels=2" gdppay
