#!/usr/bin/env bash
# Fetch + preprocess the real-classifier host-test fixtures
# (g2g-ml/tests/mobilenet_classify.rs): the validated int8 MobileNetV2 from the
# ONNX model zoo, a sample image, and the preprocessed f32 NCHW input. The 3.6 MB
# model is not committed (repo fixtures are KB-scale), so this fetches it on
# demand into a gitignored dir, the "validated locally, not CI" pattern of the
# GPU / Android probes.
#
# Needs python3 + network. Creates a throwaway venv with the onnx tooling
# (override its location with MOBILENET_VENV).
#
# After running:
#   cargo test -p g2g-ml --features ort --test mobilenet_classify -- --nocapture
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
FIX="$HERE/g2g-ml/tests/fixtures/mobilenet"
VENV="${MOBILENET_VENV:-/tmp/g2g-onnxvenv}"

if [ ! -x "$VENV/bin/python" ]; then
  echo "creating venv at $VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" -q install --upgrade pip
  "$VENV/bin/pip" -q install onnxruntime numpy pillow
fi

"$VENV/bin/python" "$FIX/fetch.py"

echo
echo "fixtures ready in $FIX"
echo "run: cargo test -p g2g-ml --features ort --test mobilenet_classify -- --nocapture"
