#!/usr/bin/env python3
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path

import cv2
import numpy as np
import onnx2tf
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.utils.tal import make_anchors

SIDE = 256


class NormalizedDetect(Detect):
    def _get_decode_boxes(self, x):
        shape = x["feats"][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape
        # per-tensor int8 gives a multiply by a constant one scale on both sides, so boxes in grid units lose precision
        scale = self.strides / SIDE
        return self.decode_bboxes(self.dfl(x["boxes"]) * scale, (self.anchors * scale).unsqueeze(0))


model = YOLO("yolov8n.pt")
model.model.model[-1].__class__ = NormalizedDetect
onnx_path = model.export(format="onnx", imgsz=SIDE, simplify=True)

calibration = np.stack(
    [
        cv2.resize(cv2.imread(str(p))[:, :, ::-1], (SIDE, SIDE), interpolation=cv2.INTER_AREA)
        for p in sorted(Path("coco128/images/train2017").glob("*.jpg"))
    ]
).astype(np.float32)
np.save("calibration.npy", calibration)

# onnx2tf fails to load its own download of this file (onnx2tf issue 545)
attempt_download_asset("calibration_image_sample_data_20x128x128x3_float32.npy.zip", unzip=True, delete=True)
onnx2tf.convert(
    input_onnx_file_path=onnx_path,
    output_folder_path="tflite",
    output_integer_quantized_tflite=True,
    quant_type="per-tensor",
    custom_input_op_name_np_data_path=[["images", "calibration.npy", [[[[0, 0, 0]]]], [[[[255, 255, 255]]]]]],
    input_quant_dtype="uint8",
    output_quant_dtype="float32",
    not_use_onnxsim=True,
    verbosity="error",
)
