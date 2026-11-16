#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# usage: tools/esp-tflm-fetch.sh, downloads TFLite Micro (esp-tflite-micro) and the ESP-NN kernels into examples/g2g-esp32s3/third_party
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/examples/g2g-esp32s3/third_party"

fetch() {
    local name=$1 version=$2 sha256=$3
    local archive="$DEST/$name-$version.tar.gz"
    if [ -d "$DEST/$name" ]; then
        return
    fi
    mkdir -p "$DEST"
    curl -fsSL --retry 3 -o "$archive" "https://github.com/espressif/$name/archive/refs/tags/v$version.tar.gz"
    echo "$sha256  $archive" | sha256sum -c -
    tar -xzf "$archive" -C "$DEST"
    mv "$DEST/$name-$version" "$DEST/$name"
    rm "$archive"
}

fetch esp-tflite-micro 1.4.3 e912f86be3b78d196b28e9a4b35222f6e8a5c7eed629728a9cec02e326da3a18
fetch esp-nn 1.5.0 63e8432b00c354eecb7dfab7133c47bc50023337cd6ea1c4dec67ca1060fef17
