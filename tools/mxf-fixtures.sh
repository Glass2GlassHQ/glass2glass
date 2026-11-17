#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# Regenerate the MXF demuxer fixtures in g2g-plugins/tests/fixtures from ffmpeg.
#
# Each mxf_<essence>.mxf is ten frames at 25 fps from ffmpeg's frame-wrapping OP1a
# muxer, or its OP-Atom muxer for the sound file, and <file>.<stream>.data beside
# it is ffmpeg's stream copy of that track, every packet back to back.
# mxf_imf_pcm_clip.mxf is not written here: it is an asdcplib AS-02 stereo 24-bit
# sound file, the clip-wrapped layout IMF uses, which ffmpeg cannot write.
set -euo pipefail

fixtures="$(cd "$(dirname "$0")/.." && pwd)/g2g-plugins/tests/fixtures"
sound="sine=frequency=440:sample_rate=48000:duration=0.4"

picture() {
    echo "testsrc2=size=$1:rate=25:duration=0.4"
}

copy() {
    local file=$1 stream=$2
    ffmpeg -v error -y -i "$file" -map "0:$stream" -c copy -f data "$file.$stream.data"
}

video() {
    local name=$1 size=$2
    shift 2
    local file="$fixtures/mxf_${name}.mxf"
    ffmpeg -v error -y -f lavfi -i "$(picture "$size")" "$@" -f mxf "$file"
    copy "$file" v
}

# ffmpeg writes no temporal offsets for h.264, so its b-frames would play out of order
video h264 64x48 -c:v libx264 -g 5 -bf 0 -pix_fmt yuv420p
video prores 64x48 -c:v prores_ks -profile:v 0
video jpeg2000 64x48 -c:v jpeg2000 -q:v 20
# dnxhr takes nothing smaller than 256x120
video dnxhr 256x120 -c:v dnxhd -profile:v dnxhr_lb -pix_fmt yuv422p

mpeg2="$fixtures/mxf_mpeg2_pcm.mxf"
ffmpeg -v error -y -f lavfi -i "$(picture 64x48)" -f lavfi -i "$sound" \
    -c:v mpeg2video -g 5 -bf 2 -q:v 10 -c:a pcm_s24le -ac 2 -f mxf "$mpeg2"
copy "$mpeg2" v
copy "$mpeg2" a

opatom="$fixtures/mxf_pcm_opatom.mxf"
ffmpeg -v error -y -f lavfi -i "$sound" -c:a pcm_s24le -f mxf_opatom "$opatom"
copy "$opatom" a

clip="$fixtures/mxf_imf_pcm_clip.mxf"
ffmpeg -v error -y -i "$clip" -map 0:a -c copy -f data "$clip.a.data"
