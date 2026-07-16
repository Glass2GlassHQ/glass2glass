#!/usr/bin/env python3
# Provenance + on-demand fetcher for the real-classifier host test
# (g2g-ml/tests/mobilenet_classify.rs). The committed toy fixtures (M440-M444)
# prove the on-NPU *plumbing*; this proves *utility*: the validated int8
# MobileNetV2 (ImageNet) classifying a real image through the g2g element graph.
#
# The model is 3.6 MB, far larger than this repo's KB-scale fixtures, so it is
# NOT committed. This script fetches it on demand into a gitignored dir, the same
# "validated locally, not CI" pattern as the GPU / Android probes; the build does
# not run it. Re-run via tools/mobilenet-fixture.sh.
#
# Writes (all gitignored): model.onnx, input_f32.bin, expected.txt, plus the raw
# sample.jpg / imagenet_classes.txt it derives them from.
#
# Needs: onnxruntime + numpy + pillow (tools/mobilenet-fixture.sh sets up a venv).
import os
import urllib.request

import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model.onnx")
INPUT = os.path.join(HERE, "input_f32.bin")
EXPECTED = os.path.join(HERE, "expected.txt")
IMAGE = os.path.join(HERE, "sample.jpg")
LABELS = os.path.join(HERE, "imagenet_classes.txt")

# The validated int8 MobileNetV2 (QOperator format: QLinearConv etc.), float
# input [1,3,224,224] -> float output [1,1000].
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12-int8.onnx"
# A clear single-subject sample image + the 1000 ImageNet class names.
IMAGE_URL = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

# Standard ImageNet preprocessing for this model.
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SIZE = 224


def fetch(url, path):
    if not os.path.exists(path):
        print("downloading", url)
        urllib.request.urlretrieve(url, path)


def main():
    fetch(MODEL_URL, MODEL)
    fetch(IMAGE_URL, IMAGE)
    fetch(LABELS_URL, LABELS)

    # Resize -> [0,1] -> normalize -> NCHW; write the f32 little-endian bytes the
    # Rust test reads verbatim (C order, matching a [1,3,224,224] read).
    img = Image.open(IMAGE).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    x = np.asarray(img).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)
    x.tofile(INPUT)

    # Reference top-1 from the exact bytes on disk, so the Rust assertion targets
    # the same number ONNX Runtime produces (the g2g chain must reproduce it).
    xr = np.fromfile(INPUT, dtype=np.float32).reshape(1, 3, SIZE, SIZE)
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": xr})[0][0]
    idx = int(out.argmax())
    labels = [line.strip() for line in open(LABELS)]
    with open(EXPECTED, "w") as f:
        f.write(f"{idx}\t{labels[idx]}\n")

    print(f"input {tuple(x.shape)} -> reference top-1 idx {idx} = {labels[idx]} (logit {out[idx]:.3f})")
    print("wrote", MODEL, INPUT, EXPECTED)


if __name__ == "__main__":
    main()
