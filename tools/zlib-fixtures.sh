#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# a 64x48 NV12 frame and the same frame compressed by C zlib, as a zlib stream and as raw deflate
set -euo pipefail

fixtures="$(cd "$(dirname "$0")/.." && pwd)/g2g-plugins/tests/fixtures"
uv run --no-project python -I -c "
import sys, zlib
width, height = 64, 48
luma = bytes((x * 3 + y * 5) % 256 for y in range(height) for x in range(width))
chroma = bytes((x * 7 + y) % 256 for y in range(height // 2) for x in range(width))
frame = luma + chroma
fixtures = sys.argv[1]
open(f'{fixtures}/nv12_64x48.raw', 'wb').write(frame)
open(f'{fixtures}/nv12_64x48.zlib', 'wb').write(zlib.compress(frame, 6))
deflate = zlib.compressobj(6, zlib.DEFLATED, -15)
open(f'{fixtures}/nv12_64x48.deflate', 'wb').write(deflate.compress(frame) + deflate.flush())
" "$fixtures"
