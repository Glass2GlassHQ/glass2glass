#!/bin/sh
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# Runs ST's COCO SSD MobileNet on an STM32MP2 board's NPU against the IMX335
# camera and draws the boxes on Weston. The main pipe carries 640x480 NV12 to the
# display in the camera's own buffers, the aux pipe the model's 256x256 RGB.
# Given an H.264 MP4 instead of a frame count, it runs the same detection on the
# clip, decoded on the Hantro and scaled for the model on the CPU, fullscreen.
# ML_LAUNCH is g2g-ml's ml_launch example built with the openvx,
# g2g-plugins/v4l2 and g2g-plugins/wayland-sink features, by default ml_launch
# beside this script, as are the two setup scripts.
#
# Usage (on the board): stm32mp2-imx335-detect.sh FRAMES|CLIP.mp4 [CONFIDENCE]
set -e
INPUT=$1
CONFIDENCE=${2:-0.65}
HERE=$(dirname "$0")
ML_LAUNCH=${ML_LAUNCH:-$HERE/ml_launch}
MODELS=/usr/local/x-linux-ai/object-detection/models/coco_ssd_mobilenet
MODEL_SIDE=256
MODEL_CAPS="video/x-raw,format=RGB,width=$MODEL_SIDE,height=$MODEL_SIDE"
# one frame period of waiting for the detector plus its inference and postprocessing
JOIN_LATENCY_NS=70000000
LABELS=/tmp/coco_labels.txt
# ST's label file starts with the SSD's background class, which detectionpostprocess does not count
tail -n +2 $MODELS/labels_coco_dataset_80.txt > $LABELS
if [ -f "$INPUT" ]; then
  DISPLAY_SINK="waylandsink fullscreen=true"
  SOURCES="filesrc location=$INPUT ! qtdemux ! h264parse ! v4l2slh264dec ! tee name=t \
   t. ! j.video \
   t. ! videoscale ! videoconvert ! $MODEL_CAPS"
else
  "$HERE/stm32mp2-imx335-setup.sh" 640 480
  "$HERE/stm32mp2-imx335-aux-setup.sh" $MODEL_SIDE $MODEL_SIDE
  # ST's ISP control applies exposure and white balance once per call
  while true; do /usr/local/demo/bin/dcmipp-isp-ctrl -i0 -g >/dev/null 2>&1; sleep 1; done &
  ISP=$!
  trap 'kill $ISP' EXIT
  DISPLAY_SINK=waylandsink
  SOURCES="v4l2src device=/dev/video1 io-mode=dmabuf num-buffers=$INPUT \
   ! video/x-raw,format=NV12,width=640,height=480 ! j.video \
   v4l2src device=/dev/video2 num-buffers=$INPUT ! $MODEL_CAPS"
fi
XDG_RUNTIME_DIR=/run/user/1001 WAYLAND_DISPLAY=wayland-0 "$ML_LAUNCH" \
  "analyticsjoin name=j latency=$JOIN_LATENCY_NS ! analyticsoverlay show-label=true ! $DISPLAY_SINK \
   $SOURCES \
   ! ovxinference model=$MODELS/ssd_mobilenet_v2_fpnlite_10_256_int8_per_tensor.nb attach-tensor=true tensor-name=ssd \
   ! detectionpostprocess head=ssd tensor-name=ssd conf-threshold=$CONFIDENCE iou-threshold=0.45 label-file=$LABELS \
   ! j.analytics"
