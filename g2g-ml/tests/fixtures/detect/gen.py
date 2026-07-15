#!/usr/bin/env python3
# Provenance + on-demand fetcher for the real-detector test
# (g2g-ml/tests/yolo_detect.rs). Obtains a real YOLO (Ultralytics YOLO11 / YOLOv8,
# which share the [1, 4+C, A] channel-major output DetectionPostprocess decodes)
# and preprocesses a sample image to the model's input.
#
# The model is tens of MB, larger than this repo's KB-scale fixtures, so it is NOT
# committed; this builds it on demand into a gitignored dir (the "validated
# locally, not CI" pattern of the GPU / Android probes). The model source, in
# order: $G2G_YOLO_MODEL (copy an existing .onnx export), else an `ultralytics`
# export of yolo11n (anonymous HuggingFace downloads are blocked in this
# environment, so there is no plain-curl path). Run via tools/detect-fixture.sh.
#
# Writes (gitignored): model.onnx, input_f32.bin (+ the raw sample.jpg).
import os
import shutil
import urllib.request

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model.onnx")
INPUT = os.path.join(HERE, "input_f32.bin")
IMAGE = os.path.join(HERE, "sample.jpg")
IMAGE_URL = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
SIZE = 640  # YOLO11 / YOLOv8 default input


def obtain_model():
    if os.path.exists(MODEL):
        return
    src = os.environ.get("G2G_YOLO_MODEL")
    if src and os.path.exists(src):
        print("copying model from $G2G_YOLO_MODEL:", src)
        shutil.copyfile(src, MODEL)
        return
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "no model: set G2G_YOLO_MODEL=/path/to/yolo11*.onnx (a YOLOv8/11 export), "
            "or `pip install ultralytics` so this can export yolo11n."
        )
    print("exporting yolo11n via ultralytics")
    path = YOLO("yolo11n.pt").export(format="onnx", opset=12, imgsz=SIZE)
    shutil.copyfile(path, MODEL)


def main():
    obtain_model()
    if not os.path.exists(IMAGE):
        print("downloading", IMAGE_URL)
        urllib.request.urlretrieve(IMAGE_URL, IMAGE)
    # YOLO preprocessing: resize to SIZE, scale to [0,1], NCHW RGB (no mean/std).
    img = Image.open(IMAGE).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    x = np.transpose(np.asarray(img).astype(np.float32) / 255.0, (2, 0, 1))[None, ...].astype(np.float32)
    x.tofile(INPUT)
    print(f"wrote {MODEL} + {INPUT} ({x.nbytes} bytes, f32 NCHW [1,3,{SIZE},{SIZE}])")
    print("the test expects a COCO 'dog' (class 16) detection in the sample image")


if __name__ == "__main__":
    main()
