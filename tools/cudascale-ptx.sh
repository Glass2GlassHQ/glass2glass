#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# nvidia drivers load PTX only up to the ISA of their own CUDA release, and CUDA 12.8 matches driver 570, the nvenc floor
set -euo pipefail

plugins="$(cd "$(dirname "$0")/.." && pwd)/g2g-plugins/src"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

cat > "$work/compile.py" <<'PY'
import ctypes
import pathlib
import sys

import nvidia.cuda_nvrtc

source_path, output_path = sys.argv[1:]
library = pathlib.Path(nvidia.cuda_nvrtc.__path__[0]) / "lib" / "libnvrtc.so.12"
nvrtc = ctypes.CDLL(str(library))
program = ctypes.c_void_p()
source = pathlib.Path(source_path).read_bytes()
if nvrtc.nvrtcCreateProgram(ctypes.byref(program), source, b"cudascale.cu", 0, None, None):
    sys.exit("nvrtcCreateProgram failed")
options = (ctypes.c_char_p * 1)(b"--gpu-architecture=compute_52")
compiled = nvrtc.nvrtcCompileProgram(program, len(options), options)
log_size = ctypes.c_size_t()
nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(log_size))
log = ctypes.create_string_buffer(log_size.value)
nvrtc.nvrtcGetProgramLog(program, log)
if compiled:
    sys.exit(log.value.decode())
ptx_size = ctypes.c_size_t()
nvrtc.nvrtcGetPTXSize(program, ctypes.byref(ptx_size))
ptx = ctypes.create_string_buffer(ptx_size.value)
nvrtc.nvrtcGetPTX(program, ptx)
pathlib.Path(output_path).write_bytes(ptx.value)
PY
uv run --no-project --with 'nvidia-cuda-nvrtc-cu12==12.8.93' \
    python -I "$work/compile.py" "$plugins/cudascale.cu" "$plugins/cudascale.ptx"
