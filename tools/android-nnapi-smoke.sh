#!/usr/bin/env bash
# Build the NNAPI / XNNPACK ONNX Runtime EP probe for Android and run it on a
# connected device. NNAPI is a system API (no permission, no APK), so this runs as
# a plain native binary from /data/local/tmp like the AAudio/MediaCodec probes.
# It proves the Android ORT build carries the NNAPI + XNNPACK symbols, the EPs
# register, and a session runs on-device (output byte-exact with the CPU path).
#
# Prerequisites:
#   - Android NDK installed; cargo-ndk finds it via ANDROID_NDK_HOME (or the
#     ndk.dir in a local SDK). Install: `cargo install cargo-ndk`.
#   - The rustup target: `rustup target add aarch64-linux-android`.
#   - adb on PATH and a device with USB debugging authorised (`adb devices`).
#   - First build downloads the Android ONNX Runtime prebuilt (network needed).
#
# Usage: tools/android-nnapi-smoke.sh [abi]
#   abi defaults to arm64-v8a. Other: x86_64, armeabi-v7a.
set -euo pipefail

ABI="${1:-arm64-v8a}"
# TRIPLE is the Rust target; LIBCXX_ARCH is the NDK sysroot lib subdir (which uses
# `arm-linux-androideabi`, not the `armv7-` Rust triple, for 32-bit ARM).
case "$ABI" in
    arm64-v8a)    TRIPLE=aarch64-linux-android;   LIBCXX_ARCH=aarch64-linux-android ;;
    armeabi-v7a)  TRIPLE=armv7-linux-androideabi; LIBCXX_ARCH=arm-linux-androideabi ;;
    x86_64)       TRIPLE=x86_64-linux-android;    LIBCXX_ARCH=x86_64-linux-android ;;
    x86)          TRIPLE=i686-linux-android;      LIBCXX_ARCH=i686-linux-android ;;
    *) echo "unknown ABI '$ABI' (use arm64-v8a | x86_64 | armeabi-v7a | x86)" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# 1. Confirm a device is connected before spending time on the build.
if ! command -v adb >/dev/null; then echo "adb not found on PATH" >&2; exit 1; fi
if [ -z "$(adb devices | awk 'NR>1 && $2=="device"{print $1}')" ]; then
    echo "no authorised device (check the USB-debugging prompt on the phone); 'adb devices' shows:" >&2
    adb devices >&2
    exit 1
fi

# 2. Cross-compile just this test binary with the NDK linker (cargo-ndk).
#    --platform 27: NNAPI is API 27+ (NCHW mode would need 29; we use defaults).
echo ">> building android_nnapi_probe for $ABI ($TRIPLE)"
cargo ndk --platform 27 -t "$ABI" build --release -p g2g-ml --features "nnapi xnnpack" \
    --test android_nnapi_probe

# 3. Locate the freshly built test binary (filter by the execute bit; the deps dir
#    also holds non-executable artifacts with the same prefix).
BIN="$(find "target/$TRIPLE/release/deps" -maxdepth 1 -type f -executable \
    -name 'android_nnapi_probe-*' -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2)"
if [ -z "$BIN" ]; then echo "could not find the built test binary under target/$TRIPLE/release/deps" >&2; exit 1; fi
echo ">> built $BIN"

# 4. Push the binary and, if ORT linked dynamically, its libonnxruntime.so too
#    (set LD_LIBRARY_PATH so the loader finds it next to the binary).
DEVDIR=/data/local/tmp
DEV="$DEVDIR/g2g_nnapi_probe"
adb push "$BIN" "$DEV" >/dev/null
adb shell chmod 755 "$DEV"

# ORT is C++, so the binary needs the NDK's libc++_shared.so (not on the device);
# push it next to the binary and run with LD_LIBRARY_PATH.
NDK="${ANDROID_NDK_HOME:-}"
LIBCXX="$(find "$NDK" -name 'libc++_shared.so' -path "*/$LIBCXX_ARCH/*" 2>/dev/null | head -1)"
if [ -z "$LIBCXX" ]; then
    echo "could not find libc++_shared.so for $LIBCXX_ARCH under ANDROID_NDK_HOME ($NDK)" >&2
    exit 1
fi
adb push "$LIBCXX" "$DEVDIR/" >/dev/null

# If ORT linked dynamically (a libonnxruntime.so in the build), push that too.
SO="$(find "target/$TRIPLE/release" -name 'libonnxruntime*.so' -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2)"
if [ -n "$SO" ]; then
    echo ">> pushing dynamic ORT lib $(basename "$SO")"
    adb push "$SO" "$DEVDIR/" >/dev/null
fi

echo ">> running on device"
set +e
OUT="$(adb shell "LD_LIBRARY_PATH=$DEVDIR $DEV --nocapture --test-threads=1" 2>&1)"
CODE=$?
set -e
echo "$OUT"
adb shell rm -f "$DEV" >/dev/null 2>&1 || true

# adb shell exit-code propagation is unreliable on some devices, so also confirm
# from the libtest summary line.
if echo "$OUT" | grep -q "test result: ok"; then
    echo ">> PASS"
    exit 0
fi
echo ">> FAIL (exit $CODE)"
exit 1
