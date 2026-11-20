#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# Times ST's MobileNet v2 network binary graph per frame on an STM32MP2 board:
# g2g's ovxinference_bench, then ST's stai_mpu runtime on the same model.
#
# Usage: tools/stm32mp2-npu-bench.sh <ssh host>
#   CROSS_CONFIG names a cross configuration whose image can build for the board.
#   Needs ST's X-LINUX-AI package on the board.
set -euo pipefail

BOARD="$1"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET=aarch64-unknown-linux-gnu
MODEL=/usr/local/x-linux-ai/image-classification/models/mobilenet/mobilenet_v2_1.0_224_int8_per_tensor.nb
WARMUP_RUNS=10
TIMED_RUNS=200
WORK=/root/g2g/npu
cd "$REPO"

cross build --release --target "$TARGET" -p g2g-ml --features openvx --example ovxinference_bench
ssh "$BOARD" "mkdir -p $WORK"
scp -q "target/$TARGET/release/examples/ovxinference_bench" "$BOARD:$WORK/"
ssh "$BOARD" "cd $WORK && timeout 300 ./ovxinference_bench $MODEL 2>/dev/null"
ssh "$BOARD" "python3 - $MODEL $WARMUP_RUNS $TIMED_RUNS 2>/dev/null" <<'PY'
import sys
import time
import numpy as np
from stai_mpu import stai_mpu_network
model, warmup_runs, timed_runs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
network = stai_mpu_network(model_path=model, use_hw_acceleration=True)
shape = network.get_input_infos()[0].get_shape()
picture = np.zeros(shape, np.uint8)
def frame():
    network.set_input(0, picture)
    network.run()
    network.get_output(0)
for _ in range(warmup_runs):
    frame()
durations = []
for _ in range(timed_runs):
    start = time.perf_counter()
    frame()
    durations.append((time.perf_counter() - start) * 1000)
durations.sort()
print(f"{'stai_mpu frame':<22} {durations[len(durations) // 2]:>8.2f} "
      f"{durations[len(durations) * 99 // 100]:>8.2f} {sum(durations) / len(durations):>8.2f}")
PY
