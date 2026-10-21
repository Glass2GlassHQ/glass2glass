#!/usr/bin/env bash
# Regenerate the UHD H.265 container fixtures m1718_cost_autoplug.rs probes.
#
# Each file holds the same three black frames, so it stays a few kilobytes. A
# transport stream needs more than one frame because its first PES unit only
# completes when the next one starts.
set -euo pipefail

fixtures="$(cd "$(dirname "$0")/.." && pwd)/g2g-plugins/tests/fixtures"

for container in mp4 mkv ts; do
    ffmpeg -v error -y -f lavfi -i "color=black:size=3840x2160:rate=30" -frames:v 3 \
        -c:v libx265 -preset ultrafast -x265-params log-level=error -pix_fmt yuv420p \
        "$fixtures/h265_3840x2160.$container"
done
