#!/usr/bin/env python3
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

import sys

import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

model, network = sys.argv[1], sys.argv[2]
# the VX delegate writes the network it compiles to cache_file_path on the first invoke
delegate = load_delegate("/usr/lib/libvx_delegate.so.2", {"cache_file_path": network, "allowed_cache_mode": "true"})
interpreter = Interpreter(model_path=model, experimental_delegates=[delegate])
interpreter.allocate_tensors()
tensor_in = interpreter.get_input_details()[0]
interpreter.set_tensor(tensor_in["index"], np.zeros(tensor_in["shape"], tensor_in["dtype"]))
interpreter.invoke()
