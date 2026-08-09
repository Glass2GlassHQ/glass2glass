#!/usr/bin/env python3
"""Build the ONNX fixtures for examples/g2g-onnx-import and their reference logits.

    uv run --with onnx --with onnxruntime --with numpy tools/onnx-fixture.py \
        classifier examples/g2g-onnx-import/model/tiny_classifier.onnx
    uv run --with onnx --with onnxruntime --with numpy tools/onnx-fixture.py \
        attention examples/g2g-onnx-import/model/tiny_attention.onnx

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
NUM_CLASSES = 2

CONV_CHANNELS = 4
KERNEL = 3
CLASSIFIER_SEED = 983
CLASSIFIER_OPSET = 17

SEQ_LEN = WIDTH * HEIGHT
HIDDEN = 8
NUM_HEADS = 2
HEAD_SIZE = HIDDEN // NUM_HEADS
ATTENTION_SEED = 987
# The standard-domain Attention op (one node for a whole multi-head block) lands
# in opset 23, which is also the minimum onnx-ir's Attention parser accepts.
ATTENTION_OPSET = 23


def rgba_frame() -> np.ndarray:
    """Deterministic RGBA8 test frame, mirrored byte for byte in the Rust test."""
    count = WIDTH * HEIGHT * 4
    return np.array([(i * 37 + 11) % 251 for i in range(count)], dtype=np.uint8)


def nchw_input(rgba: np.ndarray) -> np.ndarray:
    """The normalization BurnInference applies: RGB planes, value / 255."""
    planes = rgba.reshape(HEIGHT * WIDTH, 4)[:, :IN_CHANNELS].T
    return (planes.astype(np.float32) / 255.0).reshape(1, IN_CHANNELS, HEIGHT, WIDTH)


def finish(nodes, initializers, name, opset) -> onnx.ModelProto:
    """Wrap a node list as a checked [1, C, H, W] -> [1, NUM_CLASSES] model."""
    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, IN_CHANNELS, HEIGHT, WIDTH])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, NUM_CLASSES])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    onnx.checker.check_model(model)
    return model


def build_classifier() -> onnx.ModelProto:
    rng = np.random.default_rng(CLASSIFIER_SEED)
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

    model = finish(nodes, initializers, "tiny_classifier", CLASSIFIER_OPSET)
    model.ir_version = 9
    return model


def build_attention() -> onnx.ModelProto:
    """Pixels as a token sequence -> multi-head self-attention -> mean pool -> linear.

    The 16 spatial positions are the sequence and the 3 colour channels the token
    features, so a 4x4 frame is a 16-token input with no extra fixture data.
    """
    rng = np.random.default_rng(ATTENTION_SEED)
    projections = {
        f"{name}_w": rng.standard_normal((IN_CHANNELS, HIDDEN)).astype(np.float32)
        for name in ("q", "k", "v")
    }
    out_w = rng.standard_normal((HIDDEN, NUM_CLASSES)).astype(np.float32)
    out_b = rng.standard_normal(NUM_CLASSES).astype(np.float32)

    initializers = [numpy_helper.from_array(w, name) for name, w in projections.items()]
    initializers += [
        numpy_helper.from_array(out_w, "out_w"),
        numpy_helper.from_array(out_b, "out_b"),
        numpy_helper.from_array(np.array([1, IN_CHANNELS, SEQ_LEN], dtype=np.int64), "shape_flat"),
        numpy_helper.from_array(
            np.array([1, SEQ_LEN, NUM_HEADS, HEAD_SIZE], dtype=np.int64), "shape_heads"
        ),
        numpy_helper.from_array(np.array([1, SEQ_LEN, HIDDEN], dtype=np.int64), "shape_merged"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "pool_axes"),
    ]

    nodes = [
        helper.make_node("Reshape", ["input", "shape_flat"], ["planes"]),
        helper.make_node("Transpose", ["planes"], ["tokens"], perm=[0, 2, 1]),
    ]
    for name in ("q", "k", "v"):
        nodes += [
            helper.make_node("MatMul", ["tokens", f"{name}_w"], [f"{name}_proj"]),
            helper.make_node("Reshape", [f"{name}_proj", "shape_heads"], [f"{name}_split"]),
            helper.make_node("Transpose", [f"{name}_split"], [name], perm=[0, 2, 1, 3]),
        ]
    nodes += [
        helper.make_node("Attention", ["q", "k", "v"], ["attn"]),
        helper.make_node("Transpose", ["attn"], ["attn_seq"], perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", ["attn_seq", "shape_merged"], ["merged"]),
        helper.make_node("ReduceMean", ["merged", "pool_axes"], ["pooled"], keepdims=0),
        helper.make_node("Gemm", ["pooled", "out_w", "out_b"], ["logits"]),
    ]

    return finish(nodes, initializers, "tiny_attention", ATTENTION_OPSET)


def attention_reference(model: onnx.ModelProto, nchw: np.ndarray) -> np.ndarray:
    """Scaled dot-product attention folded in numpy, straight from the ONNX spec.

    Second opinion on the `Attention` node itself: it is one opaque op in the
    graph, so without this the reference logits would rest entirely on ORT
    agreeing with itself.
    """
    weights = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    tokens = nchw.reshape(1, IN_CHANNELS, SEQ_LEN).transpose(0, 2, 1)
    heads = [
        (tokens @ weights[f"{name}_w"])
        .reshape(1, SEQ_LEN, NUM_HEADS, HEAD_SIZE)
        .transpose(0, 2, 1, 3)
        for name in ("q", "k", "v")
    ]
    query, key, value = heads
    scores = query @ key.transpose(0, 1, 3, 2) / np.sqrt(HEAD_SIZE)
    probabilities = np.exp(scores - scores.max(-1, keepdims=True))
    probabilities /= probabilities.sum(-1, keepdims=True)
    merged = (probabilities @ value).transpose(0, 2, 1, 3).reshape(1, SEQ_LEN, HIDDEN)
    return merged.mean(axis=1) @ weights["out_w"] + weights["out_b"]


MODELS = {
    "classifier": (build_classifier, None),
    "attention": (build_attention, attention_reference),
}


def main() -> None:
    name, out_path = sys.argv[1], sys.argv[2]
    build, reference = MODELS[name]
    model = build()
    onnx.save(model, out_path)

    rgba = rgba_frame()
    nchw = nchw_input(rgba)
    session = onnxruntime.InferenceSession(model.SerializeToString())
    logits = session.run(None, {"input": nchw})[0].reshape(-1)
    if reference is not None:
        drift = np.abs(logits - reference(model, nchw).reshape(-1)).max()
        assert drift < 1e-5, f"onnxruntime disagrees with the numpy reference by {drift}"
        print(f"onnxruntime matches the numpy reference to {drift:.3}")

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
