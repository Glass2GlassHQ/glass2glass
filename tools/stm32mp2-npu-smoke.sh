#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# Runs g2g-ml's openvx_mobilenet test on an STM32MP2 board: ST's MobileNet v2
# network binary graph on the NPU through ovxinference, checked bit for bit
# against ST's own stai_mpu runtime on the same picture.
#
# Usage: tools/stm32mp2-npu-smoke.sh <ssh host>
#   CROSS_CONFIG names a cross configuration whose image can build for the board.
#   Needs uv on the host and ST's X-LINUX-AI package on the board.
set -euo pipefail

BOARD="$1"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET=aarch64-unknown-linux-gnu
MODELS=/usr/local/x-linux-ai/image-classification/models/mobilenet
MODEL="$MODELS/mobilenet_v2_1.0_224_int8_per_tensor.nb"
SIDE=224
WORK=/root/g2g/npu
cd "$REPO"

BIN="$(cross test --no-run --release --target "$TARGET" -p g2g-ml --features openvx \
    --test openvx_mobilenet --message-format=json 2>/dev/null \
    | sed -n 's/.*"executable":"\([^"]*openvx_mobilenet[^"]*\)".*/\1/p' | tail -1)"
BIN="target/${BIN#*/target/}"

INPUT="$(mktemp)"
trap 'rm -f "$INPUT"' EXIT
uv run -q --with pillow python -I - g2g-ml/tests/fixtures/mobilenet/sample.jpg "$INPUT" "$SIDE" <<'PY'
import sys
from PIL import Image
side = int(sys.argv[3])
picture = Image.open(sys.argv[1]).convert("RGB").resize((side, side), Image.BILINEAR)
open(sys.argv[2], "wb").write(picture.tobytes())
PY

ssh "$BOARD" "mkdir -p $WORK"
scp -q "$BIN" "$BOARD:$WORK/openvx_mobilenet"
scp -q "$INPUT" "$BOARD:$WORK/picture.rgb"
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
ssh "$BOARD" "cd $WORK && G2G_OPENVX_MODEL=$MODEL G2G_OPENVX_INPUT=picture.rgb \
    G2G_OPENVX_REFERENCE=reference.f32 timeout 120 ./openvx_mobilenet 2>/dev/null"
