#!/usr/bin/env bash
# Build the real instance-segmentation test fixtures (g2g-ml/tests/yolo_segment.rs):
# a real YOLO -seg (Ultralytics YOLO11-seg / YOLOv8-seg) ONNX + a sample image at
# the model's RGBA input geometry. The model is tens of MB and not committed, so
# this obtains it on demand into a gitignored dir (the "validated locally, not CI"
# pattern of the GPU / Android probes).
#
# Model source: $G2G_YOLO_SEG_MODEL (an existing YOLOv8/11 -seg .onnx export) if
# set, else an `ultralytics` export of yolo11n-seg. Anonymous HuggingFace
# downloads are blocked in this environment, so one of those is required.
#
# After running:
#   cargo test -p g2g-ml --features "ort analytics" --test yolo_segment -- --nocapture
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
FIX="$HERE/g2g-ml/tests/fixtures/segmentation"
VENV="${SEGMENT_VENV:-/tmp/g2g-segvenv}"

if [ ! -x "$VENV/bin/python" ]; then
  echo ">> creating venv at $VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" -q install --upgrade pip
  # torch from the CPU index keeps the ultralytics export off the CUDA wheels.
  "$VENV/bin/pip" -q install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  "$VENV/bin/pip" -q install ultralytics onnx onnxslim numpy pillow
fi

cd "$FIX"
"$VENV/bin/python" "$FIX/gen.py"

echo
echo "fixtures ready in $FIX"
echo "run: cargo test -p g2g-ml --features \"ort analytics\" --test yolo_segment -- --nocapture"
