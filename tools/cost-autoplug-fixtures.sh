#!/usr/bin/env bash
# Regenerate the UHD H.265 container fixtures m1718_cost_autoplug.rs probes.
#
# Each file holds the same black frames, so it stays a few kilobytes. The
# one-frame transport stream holds a single PES unit, which nothing after it
# completes.
set -euo pipefail

fixtures="$(cd "$(dirname "$0")/.." && pwd)/g2g-plugins/tests/fixtures"

encode() {
    local frames=$1 out=$2
    ffmpeg -v error -y -f lavfi -i "color=black:size=3840x2160:rate=30" -frames:v "$frames" \
        -c:v libx265 -preset ultrafast -x265-params log-level=error -pix_fmt yuv420p "$out"
}

for container in mp4 mkv ts; do
    encode 3 "$fixtures/h265_3840x2160.$container"
done
encode 1 "$fixtures/h265_3840x2160_one_frame.ts"
