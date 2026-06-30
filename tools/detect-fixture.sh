#!/usr/bin/env bash
# Build the real-detector test fixtures (g2g-ml/tests/yolo_detect.rs): a real YOLO
# (Ultralytics YOLO11 / YOLOv8) ONNX + a preprocessed input image. The model is
# tens of MB and not committed, so this obtains it on demand into a gitignored dir
# (the "validated locally, not CI" pattern of the GPU / Android probes).
#
# Model source: $G2G_YOLO_MODEL (an existing YOLOv8/11 .onnx export) if set, else
# an `ultralytics` export of yolo11n. Anonymous HuggingFace downloads are blocked
# in this environment, so one of those is required.
#
# After running:
#   cargo test -p g2g-ml --features "ort analytics" --test yolo_detect -- --nocapture
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
FIX="$HERE/g2g-ml/tests/fixtures/detect"
VENV="${MOBILENET_VENV:-/tmp/g2g-onnxvenv}"

if [ ! -x "$VENV/bin/python" ]; then
  echo ">> creating venv at $VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" -q install --upgrade pip
  "$VENV/bin/pip" -q install numpy pillow
fi

"$VENV/bin/python" "$FIX/gen.py"

echo
echo "fixtures ready in $FIX"
echo "run: cargo test -p g2g-ml --features \"ort analytics\" --test yolo_detect -- --nocapture"
