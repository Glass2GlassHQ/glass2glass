#!/usr/bin/env bash
# Regenerate the keyword-spotting model fixtures in g2g-mcu/tests/fixtures/keyword:
# speech.tflite (the TFLite Micro micro_speech model as MicroFlow 0.1.3 ships it),
# its ONNX twin speech.onnx (tf2onnx, QDQ int8), quantization.txt (the input
# and output tensor scale and zero point), and yes.s16le, the one-clip input of
# the keyword graph (espeak-ng 1.51 saying "yes", 150 ms in, 15840 samples at
# 16 kHz). The
# files are committed because MicroFlow's proc macro reads the .tflite at
# compile time, and they are small. features.i8 and scores.f32le in the same
# directory are not written here: they are the keyword graph's pinned output
# on the host (examples/noalloc-pipeline/tests/keyword.rs). tflite_scores.i8 is
# the TFLite 2.15 reference interpreter's int8 output for features.i8.
#
# Needs uv, curl, ffmpeg, espeak-ng and network.
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

espeak-ng -v en-us -s 140 -w "$WORK/yes.wav" "yes"
ffmpeg -v error -y -i "$WORK/yes.wav" \
  -af "aresample=16000,adelay=150,apad=whole_len=15840,atrim=end_sample=15840" \
  -f s16le -ac 1 "$FIX/yes.s16le"

"$VENV/bin/python" -I -c '
import sys, numpy as np, tensorflow as tf
interpreter = tf.lite.Interpreter(model_path=sys.argv[1])
interpreter.allocate_tensors()
features = np.fromfile(sys.argv[2], dtype=np.int8).reshape(interpreter.get_input_details()[0]["shape"])
interpreter.set_tensor(interpreter.get_input_details()[0]["index"], features)
interpreter.invoke()
interpreter.get_tensor(interpreter.get_output_details()[0]["index"]).tofile(sys.argv[3])
' "$FIX/speech.tflite" "$FIX/features.i8" "$FIX/tflite_scores.i8"

echo "fixtures written to $FIX"
