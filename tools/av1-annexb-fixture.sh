#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

fixtures="$(cd "$(dirname "$0")/.." && pwd)/g2g-plugins/tests/fixtures"
gst-launch-1.0 -q filesrc location="$fixtures/av1_321x181.obu" ! av1parse \
    ! video/x-av1,stream-format=annexb,alignment=tu \
    ! filesink location="$fixtures/av1_321x181_annexb.obu"
