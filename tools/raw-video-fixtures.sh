#!/usr/bin/env bash
# Regenerate the uncompressed AVI and QuickTime video fixtures in g2g-plugins/tests/fixtures.
#
# Each file holds two testsrc2 frames, with ffmpeg's decode of it to the same
# pixel format beside it. ffmpeg's avi muxer writes BI_RGB top-down with rows
# padded to four bytes, and GStreamer's avimux writes it bottom-up, so the 17x16
# files come from gst-launch-1.0. avimux also pads each I420 plane's rows to four
# bytes, which ffmpeg reads as pixels, so that file's reference is GStreamer's
# own avidemux output. ffmpeg's mov muxer pads `raw ` rows to two
# bytes, which ffmpeg's own decoder then reads as unpadded, so the odd-width RGB
# file's reference is the frames ffmpeg encoded rather than its decode.
set -euo pipefail

fixtures="$(cd "$(dirname "$0")/.." && pwd)/g2g-plugins/tests/fixtures"
frames=2

source_frames() {
    local width=$1 height=$2
    echo "testsrc2=size=$(((width + 1) / 2 * 2))x$(((height + 1) / 2 * 2)):rate=1,format=gbrp,crop=${width}:${height}:0:0"
}

ffmpeg_file() {
    local name=$1 width=$2 height=$3 pix_fmt=$4 muxer=$5
    shift 5
    ffmpeg -v error -y -f lavfi -i "$(source_frames "$width" "$height")" -frames:v $frames \
        -c:v rawvideo -pix_fmt "$pix_fmt" "$@" -f "$muxer" "$fixtures/$name"
}

decode() {
    local name=$1 pix_fmt=$2
    ffmpeg -v error -y -i "$fixtures/$name" -pix_fmt "$pix_fmt" -f rawvideo "$fixtures/$name.$pix_fmt"
}

ffmpeg_file avi_bgr24_7x5.avi 7 5 bgr24 avi
decode avi_bgr24_7x5.avi bgr24
ffmpeg_file avi_bgra_7x5.avi 7 5 bgra avi
decode avi_bgra_7x5.avi bgra
ffmpeg_file avi_yv12_7x5.avi 7 5 yuv420p avi -vtag YV12
decode avi_yv12_7x5.avi yuv420p
ffmpeg_file avi_i420_7x5.avi 7 5 yuv420p avi
decode avi_i420_7x5.avi yuv420p
ffmpeg_file avi_y800_7x5.avi 7 5 gray avi -vtag Y800
decode avi_y800_7x5.avi gray
ffmpeg_file avi_grey_7x5.avi 7 5 gray avi -vtag GREY
decode avi_grey_7x5.avi gray

for format in BGR BGRx GRAY8; do
    name="avi_gst_$(echo "$format" | tr '[:upper:]' '[:lower:]')_17x16.avi"
    gst-launch-1.0 -q videotestsrc num-buffers=$frames pattern=smpte \
        ! "video/x-raw,format=$format,width=17,height=16,framerate=1/1" \
        ! avimux ! filesink location="$fixtures/$name"
done
decode avi_gst_bgr_17x16.avi bgr24
decode avi_gst_bgrx_17x16.avi bgra
decode avi_gst_gray8_17x16.avi gray

gst-launch-1.0 -q videotestsrc num-buffers=$frames pattern=smpte \
    ! "video/x-raw,format=I420,width=17,height=16,framerate=1/1" \
    ! avimux ! filesink location="$fixtures/avi_gst_i420_17x16.avi"
gst_y4m="$(mktemp --suffix=.y4m)"
gst-launch-1.0 -q filesrc location="$fixtures/avi_gst_i420_17x16.avi" ! avidemux ! y4menc \
    ! filesink location="$gst_y4m"
ffmpeg -v error -y -i "$gst_y4m" -f rawvideo "$fixtures/avi_gst_i420_17x16.avi.yuv420p"
rm "$gst_y4m"

ffmpeg_file mov_rgb24_6x5.mov 6 5 rgb24 mov
decode mov_rgb24_6x5.mov rgb24
ffmpeg_file mov_argb_7x5.mov 7 5 argb mov
decode mov_argb_7x5.mov argb
ffmpeg_file mov_rgb24_7x5.mov 7 5 rgb24 mov
ffmpeg -v error -y -f lavfi -i "$(source_frames 7 5)" -frames:v $frames -pix_fmt rgb24 \
    -f rawvideo "$fixtures/mov_rgb24_7x5.mov.source.rgb24"
