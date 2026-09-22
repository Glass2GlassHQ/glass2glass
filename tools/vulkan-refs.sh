#!/usr/bin/env bash
# Software reference dumps for the Vulkan decode tests (M1182): one raw planar
# ffmpeg decode per fixture, written as <fixture stem>.yuv into the output
# directory. Point G2G_VULKAN_REF_DIR at that directory and the tests compare
# their hardware output against these bit for bit instead of checking geometry
# only.
#
# Usage: tools/vulkan-refs.sh <output-dir>
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE_DIRECTORY="$REPOSITORY_ROOT/g2g-plugins/tests/fixtures"

# fixture file name -> ffmpeg pixel format. The 10-bit clips decode to 16-bit
# little-endian samples; everything else is 8-bit planar 4:2:0.
FIXTURES=(
  "av1_640x480.obu yuv420p"
  "av1_640x480_tiles2x2.obu yuv420p"
  "av1_640x480_showexisting.obu yuv420p"
  "av1_640x480_filmgrain.obu yuv420p"
  "av1_640x480_looprestore.obu yuv420p"
  "av1_640x480_10bit.obu yuv420p10le"
  "h264_640x480_bframes.h264 yuv420p"
  "h265_640x480_bframes.h265 yuv420p"
  "h265_640x480_main10.hevc yuv420p10le"
  "h265_640x480_opengop.hevc yuv420p"
)

if [ "$#" -ne 1 ]; then
  echo "usage: tools/vulkan-refs.sh <output-dir>" >&2
  exit 2
fi
OUTPUT_DIRECTORY="$1"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found; it decodes the references" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIRECTORY"
for entry in "${FIXTURES[@]}"; do
  read -r fixture pixel_format <<<"$entry"
  output="$OUTPUT_DIRECTORY/${fixture%.*}.yuv"
  ffmpeg -y -loglevel error -i "$FIXTURE_DIRECTORY/$fixture" \
    -f rawvideo -pix_fmt "$pixel_format" "$output"
  echo "$output $(stat -c '%s' "$output") bytes ($pixel_format)"
done
