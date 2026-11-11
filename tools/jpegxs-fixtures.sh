#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# SvtJpegxsEncApp fails every frame of a 64x48 input at its default wavelet decomposition
set -euo pipefail

fixtures="$(cd "$(dirname "$0")/.." && pwd)/g2g-plugins/tests/fixtures"
work="${JPEGXS_FIXTURE_WORK:-$HOME/tmp/jpegxs-fixtures}"
SVT_JPEGXS_PREFIX="${SVT_JPEGXS_PREFIX:-$HOME/opt/svt-jpegxs}"
mkdir -p "$work"

ffmpeg -loglevel error -y -f lavfi -i testsrc2=size=96x48:rate=25 -frames:v 3 \
    -pix_fmt yuv422p -f rawvideo "$work/i422.yuv"
LD_LIBRARY_PATH="$SVT_JPEGXS_PREFIX/lib64" "$SVT_JPEGXS_PREFIX/bin/SvtJpegxsEncApp" \
    -i "$work/i422.yuv" -w 96 -h 48 --colour-format yuv422 --input-depth 8 --bpp 2 -n 3 \
    -b "$fixtures/jpegxs_96x48_i422_3frames.jxs"
