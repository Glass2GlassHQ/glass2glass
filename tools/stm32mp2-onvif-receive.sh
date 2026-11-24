#!/bin/sh
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# Plays an RTSP stream with an ONVIF metadata track on an STM32MP2 board, such
# as stm32mp2-imx335-detect.sh serves under RTSP_PORT, decoding the H.264 on
# the Hantro and drawing the stream's detections over it on Weston.
# ML_LAUNCH is the ml_launch that stm32mp2-imx335-detect.sh describes, by
# default ml_launch beside this script.
#
# FRAMES stops it after that many frames, by default it runs until stopped.
#
# Usage (on the board): stm32mp2-onvif-receive.sh URL [FRAMES]
set -e
URL=$1
FRAMES=${2:--1}
HERE=$(dirname "$0")
ML_LAUNCH=${ML_LAUNCH:-$HERE/ml_launch}
XDG_RUNTIME_DIR=/run/user/1001 WAYLAND_DISPLAY=wayland-0 "$ML_LAUNCH" \
  "rtspsrcn location=$URL onvif-metadata=true num-buffers=$FRAMES name=s \
   s. ! h264parse ! v4l2slh264dec ! onvifmetadatacombiner name=c \
   ! analyticsoverlay show-label=true ! waylandsink \
   s. ! onvifmetadataparse ! c."
