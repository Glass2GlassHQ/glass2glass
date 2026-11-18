#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# ISO/IEC 23001-17 MP4 files written by GStreamer's isomp4mux, each next to the frames it holds
# GST_MAIN is a GStreamer 1.29 install prefix with isomp4mux, zlibcompress and rgb2bayer
set -euo pipefail

fixtures="$(cd "$(dirname "$0")/.." && pwd)/g2g-plugins/tests/fixtures"
gst_main=${GST_MAIN:?set GST_MAIN to a GStreamer 1.29 install prefix}
registry=$(mktemp -u)
trap 'rm -f "$registry"' EXIT
export LD_LIBRARY_PATH="$gst_main/lib64" GST_PLUGIN_SYSTEM_PATH="$gst_main/lib64/gstreamer-1.0" GST_REGISTRY="$registry"

fixture() {
    local name=$1 chain=$2
    local source="videotestsrc num-buffers=2 pattern=ball ! $chain"
    # shellcheck disable=SC2086
    "$gst_main/bin/gst-launch-1.0" -q $source ! filesink location="$fixtures/$name.raw"
    # shellcheck disable=SC2086
    "$gst_main/bin/gst-launch-1.0" -q $source ! "${3:-identity}" ! isomp4mux ! filesink location="$fixtures/$name.mp4"
}

fixture uncv_rgb_18x4 video/x-raw,format=RGB,width=18,height=4,framerate=30/1
fixture uncv_rgba_16x4 video/x-raw,format=RGBA,width=16,height=4,framerate=30/1
fixture uncv_i420_16x4 video/x-raw,format=I420,width=16,height=4,framerate=30/1
fixture uncv_yuy2_16x4 video/x-raw,format=YUY2,width=16,height=4,framerate=30/1
fixture uncv_grbg_8x4 "video/x-raw,width=8,height=4,framerate=30/1 ! rgb2bayer ! video/x-bayer,format=grbg"
fixture uncv_rggb16le_8x4 "video/x-raw,width=8,height=4,framerate=30/1 ! rgb2bayer ! video/x-bayer,format=rggb16le"
fixture uncv_nv12_16x4_zlib video/x-raw,format=NV12,width=16,height=4,framerate=30/1 zlibcompress

# GStreamer pads RGB rows to four bytes, g2g's frames hold them tight
uv run --no-project python -I -c "
import sys
padded = open(sys.argv[1], 'rb').read()
open(sys.argv[1], 'wb').write(b''.join(padded[row:row + 54] for row in range(0, len(padded), 56)))
" "$fixtures/uncv_rgb_18x4.raw"
