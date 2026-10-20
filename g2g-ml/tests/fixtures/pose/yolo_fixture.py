# Shared by pose/gen.py and obb/gen.py: export an Ultralytics model to ONNX,
# letterbox a non-square photo into its input, and run Ultralytics' own
# predictor on that photo through the same ONNX file as the reference.
#
# expected.txt is plain numbers so the Rust test needs no JSON parser: the
# first line is the picture width and height, then one row per detection as
# the head's `reference` function lays it out.
import os
import shutil
import urllib.request

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

SIZE = 640
OPSET = 12
CONFIDENCE = 0.25
IOU = 0.45


def build(fixture_dir, weights, image_url, reference):
    model = os.path.join(fixture_dir, "model.onnx")
    image = os.path.join(fixture_dir, "sample.jpg")
    if not os.path.exists(model):
        exported = YOLO(os.path.join(fixture_dir, weights)).export(format="onnx", opset=OPSET, imgsz=SIZE)
        shutil.move(exported, model)
    if not os.path.exists(image):
        urllib.request.urlretrieve(image_url, image)

    picture = cv2.imread(image)
    height, width = picture.shape[:2]
    # The predictor's own preprocessing for a fixed-size ONNX model: centred
    # letterbox, BGR to RGB, scaled to [0, 1], NCHW.
    boxed = LetterBox(new_shape=(SIZE, SIZE), auto=False)(image=picture)
    tensor = np.ascontiguousarray(boxed[..., ::-1].transpose(2, 0, 1))[None].astype(np.float32) / 255.0
    tensor.tofile(os.path.join(fixture_dir, "input_f32.bin"))

    result = YOLO(model).predict(picture, imgsz=SIZE, conf=CONFIDENCE, iou=IOU, verbose=False)[0]
    rows = reference(result, width, height)
    with open(os.path.join(fixture_dir, "expected.txt"), "w") as out:
        out.write(f"{width} {height}\n")
        for row in rows:
            out.write(" ".join(repr(float(v)) for v in row) + "\n")
    print(f"wrote {fixture_dir}: {width}x{height} picture, {len(rows)} reference detections")
