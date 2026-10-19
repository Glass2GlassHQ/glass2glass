#!/usr/bin/env bash
# Regenerate the v210 fixtures in g2g-plugins/tests/fixtures from ffmpeg.
#
# Each <name>.v210 is ffmpeg's v210 encode of one testsrc2 frame, and
# <name>.v210.yuv422p10le is ffmpeg's decode of that file. ffmpeg's encoder
# refuses an odd width, so the odd-width fixture is the 38-wide encode decoded
# at 37 pixels, which shares its row stride.
set -euo pipefail

fixtures="$(cd "$(dirname "$0")/.." && pwd)/g2g-plugins/tests/fixtures"
height=4

encode() {
    local width=$1 out=$2
    shift 2
    ffmpeg -v error -y -f lavfi -i "testsrc2=size=${width}x${height}:rate=1,format=gbrp" \
        "$@" -pix_fmt yuv422p10le -c:v v210 "$out"
}

decode() {
    local width=$1 v210=$2 out=$3
    ffmpeg -v error -y -f v210 -video_size "${width}x${height}" -i "$v210" \
        -frames:v 1 -pix_fmt yuv422p10le -f rawvideo "$out"
}

for width in 38 48 52; do
    name="$fixtures/v210_${width}x${height}"
    encode "$width" "$name.v210" -frames:v 1 -f rawvideo
    decode "$width" "$name.v210" "$name.v210.yuv422p10le"
done

decode 37 "$fixtures/v210_38x${height}.v210" "$fixtures/v210_38x${height}_as_37x${height}.v210.yuv422p10le"
