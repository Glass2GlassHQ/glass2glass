#!/usr/bin/env python3
"""Build the tiny conv/BN/ReLU classifier ONNX fixture and its reference logits.

    uv run --with onnx --with onnxruntime --with numpy tools/onnx-fixture.py \
        examples/g2g-onnx-import/model/tiny_classifier.onnx

Writes the .onnx and prints the RGBA input bytes and expected logits as Rust
constants, which are pasted into the importing crate's test.
"""

import sys

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto, helper, numpy_helper

WIDTH = 4
HEIGHT = 4
IN_CHANNELS = 3
CONV_CHANNELS = 4
NUM_CLASSES = 2
KERNEL = 3
SEED = 983
OPSET = 17


def rgba_frame() -> np.ndarray:
    """Deterministic RGBA8 test frame, mirrored byte for byte in the Rust test."""
    count = WIDTH * HEIGHT * 4
    return np.array([(i * 37 + 11) % 251 for i in range(count)], dtype=np.uint8)


def nchw_input(rgba: np.ndarray) -> np.ndarray:
    """The normalization BurnInference applies: RGB planes, value / 255."""
    planes = rgba.reshape(HEIGHT * WIDTH, 4)[:, :IN_CHANNELS].T
    return (planes.astype(np.float32) / 255.0).reshape(1, IN_CHANNELS, HEIGHT, WIDTH)


def build_model() -> onnx.ModelProto:
    rng = np.random.default_rng(SEED)
    conv_w = rng.standard_normal((CONV_CHANNELS, IN_CHANNELS, KERNEL, KERNEL)).astype(np.float32)
    conv_b = rng.standard_normal(CONV_CHANNELS).astype(np.float32)
    bn_scale = (rng.random(CONV_CHANNELS) + 0.5).astype(np.float32)
    bn_bias = rng.standard_normal(CONV_CHANNELS).astype(np.float32)
    bn_mean = rng.standard_normal(CONV_CHANNELS).astype(np.float32)
    bn_var = (rng.random(CONV_CHANNELS) + 0.5).astype(np.float32)
    fc_w = rng.standard_normal((CONV_CHANNELS, NUM_CLASSES)).astype(np.float32)
    fc_b = rng.standard_normal(NUM_CLASSES).astype(np.float32)

    initializers = [
        numpy_helper.from_array(conv_w, "conv_w"),
        numpy_helper.from_array(conv_b, "conv_b"),
        numpy_helper.from_array(bn_scale, "bn_scale"),
        numpy_helper.from_array(bn_bias, "bn_bias"),
        numpy_helper.from_array(bn_mean, "bn_mean"),
        numpy_helper.from_array(bn_var, "bn_var"),
        numpy_helper.from_array(fc_w, "fc_w"),
        numpy_helper.from_array(fc_b, "fc_b"),
    ]

    nodes = [
        helper.make_node(
            "Conv",
            ["input", "conv_w", "conv_b"],
            ["conv_out"],
            kernel_shape=[KERNEL, KERNEL],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
        helper.make_node(
            "BatchNormalization",
            ["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            ["bn_out"],
            epsilon=1e-5,
        ),
        helper.make_node("Relu", ["bn_out"], ["relu_out"]),
        helper.make_node("GlobalAveragePool", ["relu_out"], ["pool_out"]),
        helper.make_node("Flatten", ["pool_out"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "fc_w", "fc_b"], ["logits"]),
    ]

    graph = helper.make_graph(
        nodes,
        "tiny_classifier",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, IN_CHANNELS, HEIGHT, WIDTH])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, NUM_CLASSES])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_path = sys.argv[1]
    model = build_model()
    onnx.save(model, out_path)

    rgba = rgba_frame()
    session = onnxruntime.InferenceSession(model.SerializeToString())
    logits = session.run(None, {"input": nchw_input(rgba)})[0].reshape(-1)

    print(f"wrote {out_path}")
    print()
    print(f"const RGBA_FRAME: [u8; {rgba.size}] = [")
    for start in range(0, rgba.size, 16):
        row = ", ".join(str(v) for v in rgba[start : start + 16])
        print(f"    {row},")
    print("];")
    print()
    print(f"const EXPECTED_LOGITS: [f32; {logits.size}] = [")
    print("    " + ", ".join(f"{v:.7}" for v in logits) + ",")
    print("];")


if __name__ == "__main__":
    main()
