#!/usr/bin/env bash
# Build the pose and oriented-box fixtures (g2g-ml/tests/yolo_pose.rs,
# yolo_obb.rs): yolo11n-pose and yolo11n-obb ONNX exports, a letterboxed
# non-square photo for each, and Ultralytics' own predictions as the reference.
# Models are tens of MB, so they land in gitignored dirs. Ultralytics fetches
# its .pt weights from GitHub releases.
#
# After running:
#   cargo test -p g2g-ml --features "ort analytics" --test yolo_pose --test yolo_obb
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURES="$HERE/g2g-ml/tests/fixtures"
VENV="${G2G_POSE_VENV:-$HOME/tmp/g2g-pose-venv}"
# Ultralytics writes its settings file here rather than under ~/.config.
export YOLO_CONFIG_DIR="$VENV/ultralytics-config"

if [ ! -x "$VENV/bin/python" ]; then
  echo ">> creating venv at $VENV"
  # opencv-python has no wheel for the newest Python, so pin one that has.
  uv venv --python 3.12 "$VENV"
  uv pip install --python "$VENV/bin/python" torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python "$VENV/bin/python" ultralytics onnxruntime numpy pillow onnx onnxslim
fi

"$VENV/bin/python" "$FIXTURES/pose/gen.py"
"$VENV/bin/python" "$FIXTURES/obb/gen.py"
