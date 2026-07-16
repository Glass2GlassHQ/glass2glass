#!/usr/bin/env bash
# Build the M305 decode -> GPU -> surface present probe for Android and run it on
# a connected device. The probe presents decoded RGBA frames through a wgpu
# surface built over an Android ANativeWindow; it stands in a headless present
# target (an ImageReader window) so it runs as a plain native binary from
# /data/local/tmp, no APK needed. A real on-screen SurfaceView is the production
# target; this validates the surface-creation + present plumbing on device.
#
# Prerequisites:
#   - Android NDK installed; cargo-ndk finds it via ANDROID_NDK_HOME (or the
#     ndk.dir in a local SDK). Install: `cargo install cargo-ndk`.
#   - The rustup target: `rustup target add aarch64-linux-android`.
#   - adb on PATH and a device with USB debugging authorised (`adb devices`).
#
# Usage: tools/android-surface-present-smoke.sh [abi]
#   abi defaults to arm64-v8a (most tablets/phones). Other: x86_64, armeabi-v7a.
set -euo pipefail

ABI="${1:-arm64-v8a}"
case "$ABI" in
    arm64-v8a)    TRIPLE=aarch64-linux-android ;;
    armeabi-v7a)  TRIPLE=armv7-linux-androideabi ;;
    x86_64)       TRIPLE=x86_64-linux-android ;;
    x86)          TRIPLE=i686-linux-android ;;
    *) echo "unknown ABI '$ABI' (use arm64-v8a | x86_64 | armeabi-v7a | x86)" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# 1. Confirm a device is connected before spending time on the build.
if ! command -v adb >/dev/null; then echo "adb not found on PATH" >&2; exit 1; fi
if [ -z "$(adb devices | awk 'NR>1 && $2=="device"{print $1}')" ]; then
    echo "no authorised device (check the USB-debugging prompt on the device); 'adb devices' shows:" >&2
    adb devices >&2
    exit 1
fi

# 2. Cross-compile just this test binary with the NDK linker (cargo-ndk).
#    --platform 26: the AHardwareBuffer NDK API and the Vulkan AHB extension the
#    GPU path imports are API 26+.
echo ">> building android_surface_present_probe for $ABI ($TRIPLE)"
cargo ndk --platform 26 -t "$ABI" build --release -p g2g-plugins --features mediacodec-wgpu \
    --test android_surface_present_probe

# 3. Locate the freshly built test binary: the newest *executable* matching
#    deps/<test>-<hash> (the dir also holds non-executable .d / .o artifacts with
#    the same prefix, so filter by the execute bit, not the extension).
BIN="$(find "target/$TRIPLE/release/deps" -maxdepth 1 -type f -executable \
    -name 'android_surface_present_probe-*' -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2)"
if [ -z "$BIN" ]; then echo "could not find the built test binary under target/$TRIPLE/release/deps" >&2; exit 1; fi
echo ">> built $BIN"

# 4. Push and run on the device. --nocapture surfaces the probe's eprintln!
#    (surface format, presented-frame count, read-back variance).
DEV=/data/local/tmp/g2g_surface_present_smoke
adb push "$BIN" "$DEV" >/dev/null
adb shell chmod 755 "$DEV"
echo ">> running on device"
set +e
OUT="$(adb shell "$DEV" --nocapture --test-threads=1 2>&1)"
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
