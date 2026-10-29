#!/usr/bin/env bash
# Build g2g-uniffi, generate its Swift bindings, and run g2g-uniffi/examples/swift
# against the built library. Needs swiftc on PATH.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/kotlin-tools.sh"

OUT="$TARGET/swift-example"
generate_bindings "$OUT/generated" swift
swiftc -warnings-as-errors -module-name Example \
    -Xcc -fmodule-map-file="$OUT/generated/g2g_uniffiFFI.modulemap" -I "$OUT/generated" \
    -L "$TARGET/release" -lg2g_uniffi \
    "$OUT/generated/g2g_uniffi.swift" "$REPO/g2g-uniffi/examples/swift/main.swift" -o "$OUT/example"
LD_LIBRARY_PATH="$TARGET/release" "$OUT/example"
