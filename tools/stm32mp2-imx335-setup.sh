#!/bin/sh
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# Sets the IMX335 camera on an STM32MP2 board to WIDTH x HEIGHT on the main
# pipe (/dev/video1), the media setup in ST's launch_camera_control_mp25.sh.
# With the aux pipe linked, the main pipe fails to stream until
# stm32mp2-imx335-aux-setup.sh runs again.
#
# Usage (on the board): stm32mp2-imx335-setup.sh WIDTH HEIGHT
set -e
M=platform:48030000.dcmipp
W=$1; H=$2
S="imx335 0-001a"
C=SRGGB10_1X10
v4l2-ctl -d "$(media-ctl -d $M -e dcmipp_main_postproc)" -c gamma_correction=1
media-ctl -d $M -l "'dcmipp_input':1->'dcmipp_dump_postproc':0[0]"
media-ctl -d $M -l "'dcmipp_input':2->'dcmipp_main_isp':0[1]"
media-ctl -d $M --set-v4l2 "'$S':0[fmt:$C/2592x1944]"
media-ctl -d $M --set-v4l2 "'48020000.csi':1[fmt:$C/2592x1944]"
media-ctl -d $M --set-v4l2 "'dcmipp_input':2[fmt:$C/2592x1944]"
media-ctl -d $M --set-v4l2 "'dcmipp_main_isp':1[fmt:RGB888_1X24/2592x1944 field:none]"
media-ctl -d $M --set-v4l2 "'dcmipp_main_postproc':0[compose:(0,0)/${W}x${H}]"
media-ctl -d $M --set-v4l2 "'dcmipp_main_postproc':1[fmt:YUV8_1X24/${W}x${H}]"
