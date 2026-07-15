#!/usr/bin/env bash
# Build the live camera -> quantize -> Edge TPU chain probe and run it on a
# connected device, then surface the DarwiNN / Edge TPU logcat. A real camera frame
# flows Camera2Src -> TensorConvert(quantize) -> OrtInference(uint8 model), which
# runs entirely on the Edge TPU (M442). Camera capture works from an adb shell run
# (shell uid holds CAMERA); a denial falls back to a synthetic frame.
#
# Prerequisites: NDK + cargo-ndk + adb (as the other android smokes). The model
# fixture is embedded (include_bytes!), so nothing extra to push.
#
# Usage: tools/android-camera-tpu-smoke.sh [abi]   (abi defaults to arm64-v8a)
set -euo pipefail

ABI="${1:-arm64-v8a}"
case "$ABI" in
    arm64-v8a)    TRIPLE=aarch64-linux-android;   LIBCXX_ARCH=aarch64-linux-android ;;
    armeabi-v7a)  TRIPLE=armv7-linux-androideabi; LIBCXX_ARCH=arm-linux-androideabi ;;
    x86_64)       TRIPLE=x86_64-linux-android;    LIBCXX_ARCH=x86_64-linux-android ;;
    x86)          TRIPLE=i686-linux-android;      LIBCXX_ARCH=i686-linux-android ;;
    *) echo "unknown ABI '$ABI'" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

if ! command -v adb >/dev/null; then echo "adb not found on PATH" >&2; exit 1; fi
if [ -z "$(adb devices | awk 'NR>1 && $2=="device"{print $1}')" ]; then
    echo "no authorised device; 'adb devices' shows:" >&2
    adb devices >&2
    exit 1
fi

echo ">> building android_camera_tpu_probe for $ABI ($TRIPLE)"
cargo ndk --platform 27 -t "$ABI" build --release -p g2g-ml --features camera2-tpu \
    --test android_camera_tpu_probe

BIN="$(find "target/$TRIPLE/release/deps" -maxdepth 1 -type f -executable \
    -name 'android_camera_tpu_probe-*' -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2)"
if [ -z "$BIN" ]; then echo "could not find the built test binary" >&2; exit 1; fi
echo ">> built $BIN"

DEVDIR=/data/local/tmp
DEV="$DEVDIR/g2g_camera_tpu_probe"
adb push "$BIN" "$DEV" >/dev/null
adb shell chmod 755 "$DEV"

NDK="${ANDROID_NDK_HOME:-}"
LIBCXX="$(find "$NDK" -name 'libc++_shared.so' -path "*/$LIBCXX_ARCH/*" 2>/dev/null | head -1)"
if [ -z "$LIBCXX" ]; then echo "could not find libc++_shared.so for $LIBCXX_ARCH under $NDK" >&2; exit 1; fi
adb push "$LIBCXX" "$DEVDIR/" >/dev/null

adb logcat -c >/dev/null 2>&1 || true
echo ">> running on device"
set +e
OUT="$(adb shell "LD_LIBRARY_PATH=$DEVDIR $DEV --nocapture --test-threads=1" 2>&1)"
CODE=$?
set -e
echo "$OUT"
echo ">> Edge TPU / DarwiNN logcat during the run:"
adb logcat -d 2>/dev/null | grep -iE "edgetpu|darwinn|google-edgetpu" | tail -15 || echo "    (none captured)"
adb shell rm -f "$DEV" >/dev/null 2>&1 || true

if echo "$OUT" | grep -q "test result: ok"; then
    echo ">> PASS"
    exit 0
fi
echo ">> FAIL (exit $CODE)"
exit 1
