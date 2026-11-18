#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# GStreamer's qtdemux reads g2g's ISO/IEC 23001-17 MP4 files back to the frames g2g wrote
# GST_MAIN is a GStreamer 1.29 install prefix with qtdemux, zlibdecompress and brotlidecompress
# GStreamer strides match g2g's tight rows at these sizes, and its Bayer rows are four pixels wide in both
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
gst_main=${GST_MAIN:?set GST_MAIN to a GStreamer 1.29 install prefix}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
export LD_LIBRARY_PATH="$gst_main/lib64" GST_PLUGIN_SYSTEM_PATH="$gst_main/lib64/gstreamer-1.0" GST_REGISTRY="$work/registry.bin"

cargo build -q --manifest-path "$repo/Cargo.toml" -p g2g-plugins --bin g2g-launch --features std
g2g="$repo/target/debug/g2g-launch"

failed=0
check() {
    local name=$1 caps=$2 wrap=${3:-identity} unwrap=${4:-identity}
    local source="videotestsrc num-buffers=3 pattern=ball ! $caps"
    "$g2g" "$source ! filesink location=$work/$name.raw" > /dev/null
    "$g2g" "$source ! $wrap ! mp4mux ! filesink location=$work/$name.mp4" > /dev/null
    "$gst_main/bin/gst-launch-1.0" -q filesrc location="$work/$name.mp4" ! qtdemux ! "$unwrap" \
        ! filesink location="$work/$name.gst.raw"
    if cmp -s "$work/$name.raw" "$work/$name.gst.raw"; then
        echo "$name: identical"
    else
        echo "$name: DIFFERS"
        failed=1
    fi
}

# GStreamer's qtdemux reads no packed 4:2:2 uncv layout, not even its own YUY2 and UYVY files
for format in I420 Y42B Y444 NV12 RGB BGR RGBA ARGB BGRA BGRx GRAY8; do
    check "$format" "video/x-raw,format=$format,width=16,height=4"
done
for bayer in grbg rggb16le bggr12be; do
    check "$bayer-6x4" "video/x-raw,format=ARGB,width=6,height=4 ! rgb2bayer ! video/x-bayer,format=$bayer"
done
check NV12-zlib video/x-raw,format=NV12,width=16,height=4 zlibcompress zlibdecompress
check GRAY8-brotli video/x-raw,format=GRAY8,width=16,height=4 brotlicompress brotlidecompress
exit $failed
