#!/usr/bin/env bash
# Regenerate the v210 and r210 fixtures in g2g-plugins/tests/fixtures from ffmpeg.
#
# Each <name>.v210 is ffmpeg's v210 encode of one testsrc2 frame, and
# <name>.v210.yuv422p10le is ffmpeg's decode of that file. ffmpeg's encoder
# refuses an odd width, so the odd-width fixture is the 38-wide encode decoded
# at 37 pixels, which shares its row stride. Each v210_52x4.<container> holds two
# frames, with ffmpeg's stream copy of them as .v210 and its decode as
# .yuv422p10le beside it.
#
# Each <name>.r210 is ffmpeg's r210 encode of one testsrc2 frame with each row
# cut to its pixels, since ffmpeg pads r210 rows to 64 pixels and GStreamer does
# not, and <name>.r210.gbrp10le is ffmpeg's decode of the uncut encode.
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

for container in mov avi mkv; do
    name="$fixtures/v210_52x${height}.$container"
    encode 52 "$name" -frames:v 2
    ffmpeg -v error -y -i "$name" -c:v copy -f rawvideo "$name.v210"
    ffmpeg -v error -y -i "$name" -pix_fmt yuv422p10le -f rawvideo "$name.yuv422p10le"
done

scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
for width in 37 52 64; do
    name="$fixtures/r210_${width}x${height}"
    ffmpeg -v error -y -f lavfi -i "testsrc2=size=${width}x${height}:rate=1,format=gbrp10le" \
        -frames:v 1 -c:v r210 "$scratch/r210.mov"
    ffmpeg -v error -y -i "$scratch/r210.mov" -c:v copy -f rawvideo "$scratch/padded.r210"
    ffmpeg -v error -y -i "$scratch/r210.mov" -pix_fmt gbrp10le -f rawvideo "$name.r210.gbrp10le"
    padded_stride=$(($(stat -c %s "$scratch/padded.r210") / height))
    for ((row = 0; row < height; row++)); do
        dd if="$scratch/padded.r210" bs="$padded_stride" skip="$row" count=1 status=none \
            | head -c $((width * 4))
    done >"$name.r210"
done
