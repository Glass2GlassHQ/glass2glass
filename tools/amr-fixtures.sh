#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# ffmpeg's libopencore_amrnb and libvo_amrwbenc write the storage format with its magic
set -euo pipefail

fixtures="$(cd "$(dirname "$0")/.." && pwd)/g2g-plugins/tests/fixtures"
ffmpeg -loglevel error -y -f lavfi -i sine=frequency=440:duration=0.2 -ar 8000 -ac 1 \
    -c:a libopencore_amrnb -b:a 12.2k -f amr "$fixtures/amr_nb_440hz_11frames.amr"
ffmpeg -loglevel error -y -f lavfi -i sine=frequency=440:duration=0.2 -ar 16000 -ac 1 \
    -c:a libvo_amrwbenc -b:a 23.85k -f amr "$fixtures/amr_wb_440hz_10frames.amr"
