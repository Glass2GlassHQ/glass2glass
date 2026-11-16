#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# usage: tools/esp-tflm-reference.sh, prints TFLite Micro's reference-kernel scores for the person_detection sample images, the values examples/g2g-esp32s3 pins
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BOARD="$ROOT/examples/g2g-esp32s3"
TFLM="$BOARD/third_party/esp-tflite-micro"
[ -d "$TFLM" ] || "$ROOT/tools/esp-tflm-fetch.sh"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mapfile -t SOURCES < <(sed "s|^|$TFLM/|" "$BOARD/tflm/micro_sources.txt")
# kernel_runner.cc is TFLite Micro's kernel test harness and needs its test-only mocks
mapfile -t KERNELS < <(ls "$TFLM"/tensorflow/lite/micro/kernels/*.cc | grep -v '/kernel_runner\.cc$')
g++ -O2 -std=gnu++17 -fno-rtti -fno-exceptions -w \
    -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON -DFLATBUFFERS_LOCALE_INDEPENDENT=0 \
    -I"$TFLM" -I"$TFLM/third_party/flatbuffers/include" -I"$TFLM/third_party/kissfft" \
    "${SOURCES[@]}" "${KERNELS[@]}" \
    "$BOARD/tflm/glue.cc" "$BOARD/tflm/reference_main.cc" -o "$WORK/reference"
"$WORK/reference" "$TFLM"/examples/person_detection/static_images/sample_images/image{0..9}
