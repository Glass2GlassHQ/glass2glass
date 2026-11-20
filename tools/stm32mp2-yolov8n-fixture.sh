#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# Builds an 80-class COCO YOLOv8n for an STM32MP2 board's NPU and puts it on the
# board as DIR/yolov8n_256_coco80.nb, where stm32mp2-imx335-detect.sh MODEL=coco80
# looks for it. The host exports Ultralytics' yolov8n at 256x256 with boxes
# normalised to 0..1 and quantizes it per-tensor to int8 on coco128, then the
# board's TFLite VX delegate compiles it. Work files and the Python environment
# go to target/stm32mp2-yolov8n.
#
# Needs uv, curl, unzip, ssh, scp and network.
# Usage: tools/stm32mp2-yolov8n-fixture.sh BOARD DIR   (BOARD is an ssh destination)
set -euo pipefail

BOARD=$1
DIR=$2
HERE="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$HERE/target/stm32mp2-yolov8n"
VENV="$WORK/venv"
NETWORK=yolov8n_256_coco80.nb

mkdir -p "$WORK"
if [ ! -x "$VENV/bin/python" ]; then
  uv venv -q -p 3.12 "$VENV"
  VIRTUAL_ENV="$VENV" uv pip install -q --torch-backend cpu \
    "ultralytics==8.4.174" "torch==2.14.1" "torchvision==0.29.1" \
    "tensorflow==2.19.1" "tf-keras==2.19.0" "onnx2tf==1.28.8" "onnx==1.22.0" \
    "onnxslim==0.1.98" "onnxruntime==1.26.0" "onnx-graphsurgeon==0.6.1" \
    "sng4onnx==2.0.1" "ai-edge-litert==2.2.0" "protobuf==5.29.6"
fi
if [ ! -d "$WORK/coco128" ]; then
  curl -fsSL --retry 3 -o "$WORK/coco128.zip" \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip
  unzip -q -o "$WORK/coco128.zip" -d "$WORK"
fi
(cd "$WORK" && "$VENV/bin/python" "$HERE/tools/stm32mp2-yolov8n-export.py")

scp -q "$WORK/tflite/yolov8n_full_integer_quant.tflite" "$HERE/tools/stm32mp2-vx-compile.py" "$BOARD:/tmp/"
ssh "$BOARD" "rm -f $DIR/$NETWORK && python3 /tmp/stm32mp2-vx-compile.py /tmp/yolov8n_full_integer_quant.tflite $DIR/$NETWORK && ls -l $DIR/$NETWORK"
