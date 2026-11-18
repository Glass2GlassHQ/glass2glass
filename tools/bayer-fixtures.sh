#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# GStreamer bayer2rgb and rgb2bayer outputs for random input frames
# GStreamer reads Bayer rows at a stride rounded up to four pixels, so the width stays a multiple of four
set -euo pipefail

fixtures="$(cd "$(dirname "$0")/.." && pwd)/g2g-plugins/tests/fixtures"
width=12

gst() {
    local input=$1 blocksize=$2 caps=$3 element=$4 output_caps=$5 output=$6
    gst-launch-1.0 -q filesrc location="$fixtures/$input" blocksize="$blocksize" \
        ! "$caps,framerate=1/1" ! "$element" ! "$output_caps" ! filesink location="$fixtures/$output"
}

head -c $((width * 8)) /dev/urandom > "$fixtures/bayer_12x8_8bit.raw"
for format in grbg10le grbg12le grbg12be grbg16be; do
    bits=${format:4:2} endian=${format:6:2}
    uv run --no-project python -I -c "
import random, sys
order = 'little' if sys.argv[3] == 'le' else 'big'
samples = [random.getrandbits(int(sys.argv[2])) for _ in range(int(sys.argv[1]))]
sys.stdout.buffer.write(b''.join(s.to_bytes(2, order) for s in samples))
" $((width * 8)) "$bits" "$endian" > "$fixtures/bayer_12x8_$format.raw"
done
head -c $((width * 5)) /dev/urandom > "$fixtures/bayer_12x5_8bit.raw"
head -c $((width * 3)) /dev/urandom > "$fixtures/bayer_12x3_8bit.raw"
# a 6 pixel wide row is stored 8 pixels wide
head -c $((8 * 4)) /dev/urandom > "$fixtures/bayer_6x4_8bit.raw"
head -c $((width * 8 * 4)) /dev/urandom > "$fixtures/bayer_12x8_argb.raw"

for pattern in bggr rggb grbg gbrg; do
    gst bayer_12x8_8bit.raw $((width * 8)) "video/x-bayer,format=$pattern,width=$width,height=8" \
        bayer2rgb video/x-raw,format=BGRA "bayer_12x8_${pattern}_to_bgra.raw"
    gst bayer_12x8_argb.raw $((width * 8 * 4)) "video/x-raw,format=ARGB,width=$width,height=8" \
        rgb2bayer "video/x-bayer,format=$pattern" "bayer_12x8_argb_to_${pattern}.raw"
done
for format in RGBA ARGB BGRx; do
    gst bayer_12x8_8bit.raw $((width * 8)) "video/x-bayer,format=rggb,width=$width,height=8" \
        bayer2rgb "video/x-raw,format=$format" "bayer_12x8_rggb_to_${format,,}.raw"
done
for format in grbg10le grbg12le grbg12be grbg16be; do
    gst "bayer_12x8_$format.raw" $((width * 8 * 2)) "video/x-bayer,format=$format,width=$width,height=8" \
        bayer2rgb video/x-raw,format=BGRA "bayer_12x8_${format}_to_bgra.raw"
    # GStreamer's bayer2rgb sizes its output buffer at two bytes per channel for any input above eight bits
    truncate -s $((width * 8 * 4)) "$fixtures/bayer_12x8_${format}_to_bgra.raw"
done
for format in grbg12le grbg16be; do
    gst "bayer_12x8_$format.raw" $((width * 8 * 2)) "video/x-bayer,format=$format,width=$width,height=8" \
        bayer2rgb video/x-raw,format=RGBA64_LE "bayer_12x8_${format}_to_rgba64le.raw"
done
for format in rggb10le rggb16be; do
    gst bayer_12x8_argb.raw $((width * 8 * 4)) "video/x-raw,format=ARGB,width=$width,height=8" \
        rgb2bayer "video/x-bayer,format=$format" "bayer_12x8_argb_to_${format}.raw"
done
gst bayer_12x5_8bit.raw $((width * 5)) "video/x-bayer,format=gbrg,width=$width,height=5" \
    bayer2rgb video/x-raw,format=BGRA bayer_12x5_gbrg_to_bgra.raw
gst bayer_12x3_8bit.raw $((width * 3)) "video/x-bayer,format=rggb,width=$width,height=3" \
    bayer2rgb video/x-raw,format=BGRA bayer_12x3_rggb_to_bgra.raw
gst bayer_6x4_8bit.raw $((8 * 4)) "video/x-bayer,format=bggr,width=6,height=4" \
    bayer2rgb video/x-raw,format=BGRA bayer_6x4_bggr_to_bgra.raw
