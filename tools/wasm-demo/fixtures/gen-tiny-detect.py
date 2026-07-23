#!/usr/bin/env python3
# Generate tiny-detect.onnx: a KB-scale ONNX detector fixture for the browser
# ort-web MVP (run_websocket_ortdetect_to_canvas). Unlike the real COCO yolov8n
# (tens of MB, fetched on demand by ../get-detection-assets.sh), this is small
# enough to commit and gives the headless test a DETERMINISTIC detection count.
#
# Same input and output layout as a real YOLOv8, so the g2g side (RGBA -> NCHW
# f32 preprocess, DetectionPostprocess channel-major decode + NMS) is exercised
# byte-for-byte as with the real model:
#   in  images  [1, 3, 640, 640] f32
#   out output0 [1, 6, 8]        f32  (4 box + 2 class channels, 8 anchors)
#
# It genuinely runs over the decoded frame: the two planted boxes' centers track
# the frame's mean pixel (ReduceMean of the input), so the output changes with
# the video. The two class scores are constant (0.9 / 0.8, above the 0.25
# threshold) and the other six anchors are zero, so decode + NMS always yields
# exactly two detections (one per class) whatever the frame is.
#
# Run: uv run --with onnx --with numpy tools/wasm-demo/fixtures/gen-tiny-detect.py
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tiny-detect.onnx")
A = 8  # anchors
C = 2  # classes -> channels = 4 + C = 6

# Planted output base [1, 6, 8]: anchor 0 is class 0 (conf 0.9) in an 80x80 box
# at (200, 200), anchor 1 is class 1 (conf 0.8) at (440, 440); the rest score 0.
# The two boxes are apart so both overlay colors (class 0 red, class 1 green)
# render distinctly.
base = np.zeros((1, 4 + C, A), dtype=np.float32)
centers = {0: 200.0, 1: 440.0}
for ai in (0, 1):
    base[0, 0, ai] = centers[ai]  # cx
    base[0, 1, ai] = centers[ai]  # cy
    base[0, 2, ai] = 80.0  # w
    base[0, 3, ai] = 80.0  # h
base[0, 4, 0] = 0.9  # anchor 0 -> class 0
base[0, 5, 1] = 0.8  # anchor 1 -> class 1

# Mask: only the box channels of the two live anchors track the frame mean, so
# the boxes shift with the video while the class scores (detection count) stay
# fixed. delta = mean(input) * mask, output = base + delta.
mask = np.zeros((1, 4 + C, A), dtype=np.float32)
for ai in (0, 1):
    mask[0, 0:4, ai] = 100.0

graph = helper.make_graph(
    nodes=[
        # opset 13: ReduceMean axes is an attribute (not an input).
        helper.make_node("ReduceMean", ["images"], ["m"], axes=[0, 1, 2, 3], keepdims=0),
        helper.make_node("Mul", ["m", "mask"], ["delta"]),
        helper.make_node("Add", ["base", "delta"], ["output0"]),
    ],
    name="tiny_detect",
    inputs=[helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
    outputs=[helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 4 + C, A])],
    initializer=[
        numpy_helper.from_array(base, name="base"),
        numpy_helper.from_array(mask, name="mask"),
    ],
)
model = helper.make_model(
    graph, opset_imports=[helper.make_operatorsetid("", 13)], producer_name="g2g-tiny-detect"
)
model.ir_version = 9  # onnxruntime-web 1.20 max supported IR
onnx.checker.check_model(model)
onnx.save(model, OUT)
print(f"wrote {OUT} ({os.path.getsize(OUT)} bytes): in=images[1,3,640,640] out=output0[1,6,8]")
