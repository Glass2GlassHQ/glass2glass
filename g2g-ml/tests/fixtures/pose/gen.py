#!/usr/bin/env python3
# Builds the fixture for g2g-ml/tests/yolo_pose.rs: a yolo11n-pose ONNX export,
# a letterboxed bus.jpg, and Ultralytics' predictions on it as the reference.
# Run via tools/pose-obb-fixture.sh. Writes (gitignored): model.onnx,
# sample.jpg, input_f32.bin, expected.txt.
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from yolo_fixture import build  # noqa: E402


def poses(result, width, height):
    # One row per person: the box corners x1 y1 x2 y2 and confidence, then x y
    # visibility for each keypoint, all normalised to the picture.
    boxes = result.boxes
    keypoints = result.keypoints
    return [
        [
            *boxes.xyxyn[i].tolist(),
            float(boxes.conf[i]),
            *(v for k in range(keypoints.xyn.shape[1]) for v in (*keypoints.xyn[i][k].tolist(), float(keypoints.conf[i][k]))),
        ]
        for i in range(len(boxes))
    ]


if __name__ == "__main__":
    build(HERE, "yolo11n-pose.pt", "https://ultralytics.com/images/bus.jpg", poses)
