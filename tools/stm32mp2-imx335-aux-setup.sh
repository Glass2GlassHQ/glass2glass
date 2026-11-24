#!/bin/sh
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# Scales the ISP output to WIDTH x HEIGHT RGB888 on the aux pipe (dcmipp_aux_capture),
# the NPU input. Run it after stm32mp2-imx335-setup.sh.
#
# Usage (on the board): stm32mp2-imx335-aux-setup.sh WIDTH HEIGHT
set -e
M=platform:48030000.dcmipp
W=$1; H=$2
media-ctl -d $M -l "'dcmipp_main_isp':1->'dcmipp_aux_postproc':0[1]"
media-ctl -d $M --set-v4l2 "'dcmipp_aux_postproc':0[fmt:RGB888_1X24/2592x1944 field:none]"
media-ctl -d $M --set-v4l2 "'dcmipp_aux_postproc':0[compose:(0,0)/${W}x${H}]"
media-ctl -d $M --set-v4l2 "'dcmipp_aux_postproc':1[fmt:RGB888_1X24/${W}x${H}]"
