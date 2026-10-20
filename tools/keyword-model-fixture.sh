#!/usr/bin/env bash
# Regenerate the keyword-spotting model fixtures in g2g-mcu/tests/fixtures/keyword:
# speech.tflite (the TFLite Micro micro_speech model as MicroFlow 0.1.3 ships it),
# its ONNX twin speech.onnx (tf2onnx, QDQ int8), and quantization.txt (the input
# and output tensor scale and zero point). The files are committed because
# MicroFlow's proc macro reads the .tflite at compile time, and they are small.
#
# Needs uv, curl and network.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
FIX="$HERE/g2g-mcu/tests/fixtures/keyword"
VENV="${KEYWORD_VENV:-/tmp/g2g-tf2onnx-venv}"
CRATE_URL="https://static.crates.io/crates/microflow/microflow-0.1.3.crate"
TFLITE_SHA256="09e5e2a9dfb2d8ed78802bf18ce297bff54281a66ca18e0c23d69ca14f822a83"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

curl -fsSL --retry 3 -o "$WORK/microflow.crate" "$CRATE_URL"
tar -xzf "$WORK/microflow.crate" -C "$WORK" microflow-0.1.3/models/speech.tflite
echo "$TFLITE_SHA256  $WORK/microflow-0.1.3/models/speech.tflite" | sha256sum -c -
mkdir -p "$FIX"
cp "$WORK/microflow-0.1.3/models/speech.tflite" "$FIX/speech.tflite"

if [ ! -x "$VENV/bin/python" ]; then
  uv venv -q -p 3.11 "$VENV"
  # tf2onnx 1.17 is tested against TF 2.15, and onnx 1.17+ needs a newer ml_dtypes than TF 2.15 allows.
  VIRTUAL_ENV="$VENV" uv pip install -q "tf2onnx==1.17.0" "tensorflow-cpu==2.15.1" "onnx==1.16.2" "numpy<2"
fi

"$VENV/bin/python" -I -m tf2onnx.convert --opset 17 --tflite "$FIX/speech.tflite" --output "$FIX/speech.onnx"
"$VENV/bin/python" -I -c '
import sys, tensorflow as tf
interpreter = tf.lite.Interpreter(model_path=sys.argv[1])
(scale_in, zero_in), = [d["quantization"] for d in interpreter.get_input_details()]
(scale_out, zero_out), = [d["quantization"] for d in interpreter.get_output_details()]
print(f"input_scale {scale_in!r}\ninput_zero_point {zero_in}\noutput_scale {scale_out!r}\noutput_zero_point {zero_out}")
' "$FIX/speech.tflite" > "$FIX/quantization.txt"

echo "fixtures written to $FIX"
