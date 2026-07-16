#!/usr/bin/env python3
# Build the uint8-input QDQ MobileNetV2 for the on-device Edge TPU phase of the
# real-classifier milestone (g2g-ml/tests/android_mobilenet_tpu_probe.rs).
#
# The zoo's int8 MobileNetV2 is QOperator format (QLinearConv etc.), which ORT's
# NNAPI EP does not place on the accelerator. So this re-quantizes the *float*
# zoo model into the NNAPI-friendly form: QDQ format, per-tensor, uint8
# activations + int8 weights, with a STATIC batch (NNAPI rejects dynamic shapes).
# It then applies the M442 uint8-input surgery (drop the boundary QuantizeLinear,
# retype the graph input to uint8) so no float op pins the boundary on the CPU.
# The caller feeds uint8 already quantized with the printed (scale, zero_point) --
# exactly what TensorConvert::quantize produces.
#
# None of the outputs are committed (the model is 3.6 MB, repo fixtures are
# KB-scale): fetched / built on demand into a gitignored dir, the "validated
# locally, not CI" pattern of the GPU / Android probes. Run via
# tools/android-mobilenet-tpu-smoke.sh. Needs onnx + onnxruntime + numpy + pillow.
#
# Writes (gitignored): mn_u8in.onnx, mn_input_f32.bin, u8in_quant.txt.
import os
import urllib.request

import numpy as np
import onnx
from onnx import TensorProto
from PIL import Image
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
FLOAT = os.path.join(HERE, "mobilenetv2-12-float.onnx")
FIXED = os.path.join(HERE, "mn_fixed.onnx")  # batch pinned to 1
QDQ = os.path.join(HERE, "mn_qdq.onnx")  # f32-input QDQ (intermediate)
U8IN = os.path.join(HERE, "mn_u8in.onnx")  # committed-shape artifact: uint8 input
INPUT = os.path.join(HERE, "mn_input_f32.bin")  # normalized f32 NCHW [1,3,224,224]
QUANT = os.path.join(HERE, "u8in_quant.txt")  # "<scale>\t<zero_point>"
IMAGE = os.path.join(HERE, "sample.jpg")
LABELS = os.path.join(HERE, "imagenet_classes.txt")

FLOAT_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
IMAGE_URL = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SIZE = 224


def fetch(url, path):
    if not os.path.exists(path):
        print("downloading", url)
        urllib.request.urlretrieve(url, path)


def prep(img):
    x = np.asarray(img.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)).astype(np.float32) / 255.0
    return np.transpose(((x - MEAN) / STD), (2, 0, 1))[None, ...].astype(np.float32)


def init_scalar(graph, name):
    for init in graph.initializer:
        if init.name == name:
            return onnx.numpy_helper.to_array(init).reshape(-1)[0]
    raise KeyError(name)


def main():
    fetch(FLOAT_URL, FLOAT)
    fetch(IMAGE_URL, IMAGE)
    fetch(LABELS_URL, LABELS)
    base = Image.open(IMAGE)

    # Pin the batch dim to 1 (NNAPI needs static shapes; a dynamic batch forces
    # the whole graph to CPU).
    m = onnx.load(FLOAT)
    for vi in list(m.graph.input) + list(m.graph.output):
        d0 = vi.type.tensor_type.shape.dim[0]
        d0.ClearField("dim_param")
        d0.dim_value = 1
    onnx.save(m, FIXED)

    # Calibrate on the sample image + simple augmentations: enough to set sane
    # per-tensor activation ranges (this proves accelerator placement, the
    # accuracy is a bonus; per-tensor avoids the opset-13 axis attr the model's
    # opset 12 cannot carry).
    augs = [
        base,
        base.transpose(Image.FLIP_LEFT_RIGHT),
        base.rotate(10),
        base.crop((50, 50, 550, 550)),
        Image.eval(base, lambda p: min(255, int(p * 1.2))),
        base.resize((300, 300)),
    ]

    class Calib(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"input": prep(a)} for a in augs])

        def get_next(self):
            return next(self.it, None)

    # uint8 activations AND uint8 weights: the most broadly supported mobile
    # combination. (int8 weights load on a recent host ORT but the older Android
    # ORT prebuilt rejects the int8 initializer with "doesn't have valid type,
    # type: 3"; uint8 weights avoid it and the Edge TPU runs them all the same.)
    quantize_static(
        FIXED, QDQ, Calib(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        per_channel=False,
    )

    # uint8-input surgery (M442): drop the boundary QuantizeLinear, rewire its
    # consumer onto the now-uint8 graph input.
    mq = onnx.load(QDQ)
    g = mq.graph
    gi = g.input[0].name
    q = next(n for n in g.node if n.op_type == "QuantizeLinear" and n.input[0] == gi)
    scale = float(init_scalar(g, q.input[1]))
    zp = int(init_scalar(g, q.input[2]))
    q_out = q.output[0]
    for n in g.node:
        n.input[:] = [gi if x == q_out else x for x in n.input]
    g.node.remove(q)
    g.input[0].type.tensor_type.elem_type = TensorProto.UINT8
    onnx.checker.check_model(mq)
    onnx.save(mq, U8IN)

    prep(base).tofile(INPUT)
    with open(QUANT, "w") as f:
        f.write(f"{scale:.8f}\t{zp}\n")

    # Host sanity: quantize the normalized input the way TensorConvert will, feed
    # uint8, confirm a sensible top-1 before the artifact goes to the device.
    xq = np.clip(np.round(prep(base) / scale) + zp, 0, 255).astype(np.uint8)
    sess = ort.InferenceSession(U8IN, providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: xq})[0][0]
    labels = [line.strip() for line in open(LABELS)]
    idx = int(out.argmax())
    print(f"uint8-input model: feed uint8 = quantize(normalized, scale={scale:.6f}, zero_point={zp})")
    print(f"   host CPU top-1 idx {idx} = {labels[idx]} (logit {out[idx]:.3f})")
    print("wrote", U8IN, INPUT, QUANT)

    for tmp in (FIXED, QDQ):
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    main()
