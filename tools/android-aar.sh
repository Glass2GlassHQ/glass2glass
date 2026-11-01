#!/usr/bin/env bash
# Build the g2g Android library (AAR) and print its path last. Needs cargo-ndk
# with ANDROID_NDK_HOME and the rustup target of each ABI, a JDK, and an SDK
# platform under ANDROID_SDK_ROOT (default ~/Android/sdk). No Gradle.
#
# Usage: tools/android-aar.sh [abi...]   (default arm64-v8a x86_64)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/kotlin-tools.sh"

ABIS=("$@")
[ ${#ABIS[@]} -gt 0 ] || ABIS=(arm64-v8a x86_64)
SDK="${ANDROID_SDK_ROOT:-$HOME/Android/sdk}"
ANDROID_JAR="$(find "$SDK/platforms" -maxdepth 2 -name android.jar | sort -V | tail -1)"
MIN_SDK=26
VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' "$REPO/Cargo.toml" | head -n 1)"
ANDROID_SOURCES="$REPO/g2g-uniffi/android"
OUT="$TARGET/android-aar"
AAR="$OUT/g2g-$VERSION.aar"

fetch_kotlin_tools
generate_bindings "$OUT/generated" kotlin
rm -rf "$OUT/stage" "$OUT/classes" "$OUT/jna" "$AAR"
mkdir -p "$OUT/stage"
TARGET_FLAGS=()
for abi in "${ABIS[@]}"; do TARGET_FLAGS+=(-t "$abi"); done
(cd "$REPO" && cargo ndk --platform "$MIN_SDK" "${TARGET_FLAGS[@]}" -o "$OUT/stage/jni" \
    build --locked --release -p g2g-uniffi --features release-aar)

unzip -q "$JNA_AAR" classes.jar -d "$OUT/jna"
"$KOTLINC" -no-jdk -no-reflect -cp "$ANDROID_JAR:$OUT/jna/classes.jar" -d "$OUT/classes" \
    "$OUT/generated" "$ANDROID_SOURCES/Surfaces.kt"
(cd "$OUT/classes" && zip -qr "$OUT/stage/classes.jar" .)
cp "$ANDROID_SOURCES/AndroidManifest.xml" "$ANDROID_SOURCES/proguard.txt" "$OUT/stage/"
# Gradle reads the resource list from R.txt, and the library has no resources
: > "$OUT/stage/R.txt"
(cd "$OUT/stage" && zip -qr "$AAR" AndroidManifest.xml classes.jar jni proguard.txt R.txt)
echo "$AAR"
