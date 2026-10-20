#!/usr/bin/env python3
# Builds the fixture for g2g-ml/tests/yolo_obb.rs: a yolo11n-obb ONNX export, a
# letterboxed boats.jpg, and Ultralytics' predictions on it as the reference.
# Run via tools/pose-obb-fixture.sh. Writes (gitignored): model.onnx,
# sample.jpg, input_f32.bin, expected.txt.
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pose"))
from yolo_fixture import build  # noqa: E402


def oriented_boxes(result, width, height):
    # One row per box: centre and size normalised by the picture's width and
    # height, the angle in radians as the model emits it, label, confidence.
    obb = result.obb
    return [
        [x / width, y / height, w / width, h / height, r, int(obb.cls[i]), float(obb.conf[i])]
        for i, (x, y, w, h, r) in enumerate(obb.xywhr.tolist())
    ]


if __name__ == "__main__":
    build(HERE, "yolo11n-obb.pt", "https://ultralytics.com/images/boats.jpg", oriented_boxes)
