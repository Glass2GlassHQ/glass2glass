#!/usr/bin/env python3
# Generate the int8 QDQ Conv->ReLU ONNX fixture used by the Android NNAPI / Edge
# TPU placement probe (`g2g-ml/tests/android_nnapi_conv_probe.rs`). Committed for
# provenance; the .onnx it writes is the checked-in fixture (the build does not run
# this). int8 QDQ is what ORT's NNAPI EP folds into a quantized conv the Edge TPU
# (DarwiNN) can actually run, vs an Identity / fp32 model NNAPI leaves on CPU/GPU.
#
# Requires onnx + onnxruntime + numpy. Run:
#   python3 -m venv /tmp/onnxvenv && /tmp/onnxvenv/bin/pip install onnx onnxruntime numpy
#   /tmp/onnxvenv/bin/python g2g-ml/tests/fixtures/gen_qconv.py
#
# The model input/output stay f32 [1,3,4,4] / [1,4,4,4] (the Q/DQ live inside), so
# it satisfies OrtInference's rank-4 f32 [N,3,H,W] contract unchanged.
import os
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FLOAT_PATH = os.path.join(HERE, "qconv_relu_float.onnx")  # intermediate, not committed
INT8_PATH = os.path.join(HERE, "qconv_relu_int8.onnx")  # the committed fixture

CIN, COUT, H, W, K = 3, 4, 4, 4, 3


def build_float_model() -> None:
    # input [1,3,4,4] -> Conv(4,3,3,3, pad 1) -> Relu -> output [1,4,4,4].
    rng = np.random.default_rng(42)
    weight = rng.standard_normal((COUT, CIN, K, K)).astype(np.float32) * 0.3
    bias = rng.standard_normal((COUT,)).astype(np.float32) * 0.1

    w_init = helper.make_tensor("W", TensorProto.FLOAT, weight.shape, weight.flatten())
    b_init = helper.make_tensor("B", TensorProto.FLOAT, bias.shape, bias.flatten())

    conv = helper.make_node(
        "Conv", ["input", "W", "B"], ["conv_out"], kernel_shape=[K, K], pads=[1, 1, 1, 1], name="conv0"
    )
    relu = helper.make_node("Relu", ["conv_out"], ["output"], name="relu0")

    graph = helper.make_graph(
        [conv, relu],
        "qconv_relu",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, CIN, H, W])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, COUT, H, W])],
        [w_init, b_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
    onnx.checker.check_model(model)
    onnx.save(model, FLOAT_PATH)


class RandomCalib(CalibrationDataReader):
    # A handful of random inputs is enough to set per-tensor scales for this tiny
    # toy conv (we are proving EP placement, not model accuracy).
    def __init__(self) -> None:
        rng = np.random.default_rng(7)
        self.samples = iter(
            [{"input": rng.random((1, CIN, H, W), dtype=np.float32)} for _ in range(8)]
        )

    def get_next(self):
        return next(self.samples, None)


def quantize() -> None:
    # QDQ format, per-tensor, uint8 activations + int8 weights: the NNAPI-friendly
    # combination ORT documents for mobile accelerators.
    quantize_static(
        FLOAT_PATH,
        INT8_PATH,
        RandomCalib(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
    )


def verify() -> None:
    # Confirm the quantized model loads and runs on the CPU EP (host sanity before
    # the fixture goes to the device).
    sess = ort.InferenceSession(INT8_PATH, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(1).random((1, CIN, H, W), dtype=np.float32)
    (out,) = sess.run(None, {"input": x})
    assert out.shape == (1, COUT, H, W), out.shape
    assert np.all(out >= 0.0), "ReLU output must be non-negative"
    has_q = any(n.op_type in ("QuantizeLinear", "QLinearConv") for n in onnx.load(INT8_PATH).graph.node)
    assert has_q, "expected a quantized op in the QDQ model"
    print(f"OK int8 QDQ model: {INT8_PATH} ({os.path.getsize(INT8_PATH)} bytes), out {out.shape}")


if __name__ == "__main__":
    build_float_model()
    quantize()
    verify()
    os.remove(FLOAT_PATH)
