#!/usr/bin/env python3
# Provenance + on-demand fetcher for the real instance-segmentation test
# (g2g-ml/tests/yolo_segment.rs). Obtains a real YOLO -seg export (Ultralytics
# YOLO11-seg / YOLOv8-seg, which share the two-output layout OrtSegmentation
# decodes: [1, 4 + C + M, A] boxes + mask coefficients, [1, M, mh, mw] mask
# prototypes) and resizes a sample image to the model's RGBA input.
#
# The model is tens of MB, larger than this repo's KB-scale fixtures, so it is NOT
# committed; this builds it on demand into a gitignored dir (the "validated
# locally, not CI" pattern of the GPU / Android probes). The model source, in
# order: $G2G_YOLO_SEG_MODEL (copy an existing .onnx export), else an
# `ultralytics` export of yolo11n-seg (anonymous HuggingFace downloads are blocked
# in this environment, so there is no plain-curl path). Run via
# tools/segment-fixture.sh.
#
# Writes (gitignored): model.onnx, input_rgba.bin (+ the raw sample.jpg).
import os
import shutil
import urllib.request

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model.onnx")
INPUT = os.path.join(HERE, "input_rgba.bin")
IMAGE = os.path.join(HERE, "sample.jpg")
IMAGE_URL = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
SIZE = 640  # YOLO11 / YOLOv8 default input


def obtain_model():
    if os.path.exists(MODEL):
        return
    src = os.environ.get("G2G_YOLO_SEG_MODEL")
    if src and os.path.exists(src):
        print("copying model from $G2G_YOLO_SEG_MODEL:", src)
        shutil.copyfile(src, MODEL)
        return
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "no model: set G2G_YOLO_SEG_MODEL=/path/to/yolo11*-seg.onnx (a YOLOv8/11 "
            "-seg export), or `pip install ultralytics` so this can export yolo11n-seg."
        )
    print("exporting yolo11n-seg via ultralytics")
    path = YOLO("yolo11n-seg.pt").export(format="onnx", opset=12, imgsz=SIZE)
    shutil.copyfile(path, MODEL)


def main():
    obtain_model()
    if not os.path.exists(IMAGE):
        print("downloading", IMAGE_URL)
        urllib.request.urlretrieve(IMAGE_URL, IMAGE)
    # The element takes RGBA at the model's geometry and normalizes itself, so the
    # fixture is the plain resized image with an opaque alpha.
    img = Image.open(IMAGE).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    rgba = np.dstack([np.asarray(img), np.full((SIZE, SIZE, 1), 255, np.uint8)])
    rgba.tofile(INPUT)
    print(f"wrote {MODEL} + {INPUT} ({rgba.nbytes} bytes, RGBA8 {SIZE}x{SIZE})")
    print("the test expects a COCO 'dog' (class 16) segmentation in the sample image")


if __name__ == "__main__":
    main()
