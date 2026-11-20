#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# Runs g2g-ml's openvx_npu tests on an STM32MP2 board: ST's MobileNet v2 network
# binary graph through ovxinference, checked bit for bit against ST's stai_mpu
# runtime, and ST's SSD MobileNet through ovxinference and detectionpostprocess,
# checked against ST's own SSD postprocess on the same picture.
#
# Usage: tools/stm32mp2-npu-smoke.sh <ssh host>
#   CROSS_CONFIG names a cross configuration whose image can build for the board.
#   Needs uv on the host and ST's X-LINUX-AI package on the board.
set -euo pipefail

BOARD="$1"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET=aarch64-unknown-linux-gnu
AI=/usr/local/x-linux-ai
MODEL="$AI/image-classification/models/mobilenet/mobilenet_v2_1.0_224_int8_per_tensor.nb"
SIDE=224
SSD_MODEL="$AI/object-detection/models/coco_ssd_mobilenet/ssd_mobilenet_v2_fpnlite_10_256_int8_per_tensor.nb"
SSD_LABELS="$AI/object-detection/models/coco_ssd_mobilenet/labels_coco_dataset_80.txt"
SSD_SIDE=256
# ST's object detection app defaults
SSD_CONFIDENCE=0.65
SSD_IOU=0.45
INPUT_MEAN=127.5
INPUT_STD=127.5
WORK=/root/g2g/npu
cd "$REPO"

BIN="$(cross test --no-run --release --target "$TARGET" -p g2g-ml --features openvx,analytics \
    --test openvx_npu --message-format=json 2>/dev/null \
    | sed -n 's/.*"executable":"\([^"]*openvx_npu[^"]*\)".*/\1/p' | tail -1)"
BIN="target/${BIN#*/target/}"

SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
resize() {
    uv run -q --with pillow python -I - "$1" "$2" "$3" <<'PY'
import sys
from PIL import Image
side = int(sys.argv[3])
picture = Image.open(sys.argv[1]).convert("RGB").resize((side, side), Image.BILINEAR)
open(sys.argv[2], "wb").write(picture.tobytes())
PY
}
resize g2g-ml/tests/fixtures/mobilenet/sample.jpg "$SCRATCH/picture.rgb" "$SIDE"
resize g2g-ml/tests/fixtures/pose/sample.jpg "$SCRATCH/ssd_picture.rgb" "$SSD_SIDE"

ssh "$BOARD" "mkdir -p $WORK"
scp -q "$BIN" "$BOARD:$WORK/openvx_npu"
scp -q "$SCRATCH/picture.rgb" "$SCRATCH/ssd_picture.rgb" "$BOARD:$WORK/"
ssh "$BOARD" "cd $WORK && python3 - $MODEL $SIDE 2>/dev/null" <<'PY'
import sys
import numpy as np
from stai_mpu import stai_mpu_network
side = int(sys.argv[2])
network = stai_mpu_network(model_path=sys.argv[1], use_hw_acceleration=True)
network.set_input(0, np.fromfile("picture.rgb", np.uint8).reshape(1, side, side, 3))
network.run()
network.get_output(0).astype("<f4").tofile("reference.f32")
PY
ssh "$BOARD" "cd $WORK && PYTHONPATH=$AI/object-detection python3 - $SSD_MODEL $SSD_LABELS \
    $SSD_SIDE $INPUT_MEAN $INPUT_STD $SSD_CONFIDENCE $SSD_IOU >/dev/null 2>&1" <<'PY'
import sys
import numpy as np
from ssd_mobilenet_pp import NeuralNetwork
model, labels, side, mean, std, confidence, iou = sys.argv[1:]
side = int(side)
network = NeuralNetwork(model, labels, float(mean), float(std), float(confidence), float(iou))
network.launch_inference(np.fromfile("ssd_picture.rgb", np.uint8).reshape(side, side, 3))
locations, classes, scores = network.get_results()
found = len(scores[0]) if len(scores) else 0
rows = [[classes[0][index], scores[0][index], *locations[0][index]] for index in range(found)]
np.array(rows, "<f4").reshape(-1).tofile("ssd_reference.f32")
PY
ssh "$BOARD" "cd $WORK && G2G_OPENVX_MODEL=$MODEL G2G_OPENVX_INPUT=picture.rgb \
    G2G_OPENVX_REFERENCE=reference.f32 G2G_OPENVX_SSD_MODEL=$SSD_MODEL \
    G2G_OPENVX_SSD_INPUT=ssd_picture.rgb G2G_OPENVX_SSD_REFERENCE=ssd_reference.f32 \
    G2G_OPENVX_SSD_CONFIDENCE=$SSD_CONFIDENCE G2G_OPENVX_SSD_IOU=$SSD_IOU \
    timeout 120 ./openvx_npu 2>/dev/null"
