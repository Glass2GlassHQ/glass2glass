#!/usr/bin/env bash
# Build the real-MobileNetV2-on-Edge-TPU probe for Android, push the model + a
# preprocessed input, run it on a connected device, and surface the Edge TPU /
# DarwiNN logcat. The device phase of the M446 host classifier: a real ImageNet
# classifier (not the M442 toy conv) running through the g2g graph on the NPU.
#
# The model is the uint8-input QDQ MobileNetV2 from gen_u8in.py; it is 3.6 MB so
# it is not committed / embedded (CI cross-compiles this probe). This script
# builds it on demand (a throwaway venv with the onnx tooling) and pushes it
# alongside the test binary.
#
# Prerequisites: NDK + cargo-ndk + adb (as the other android smokes) + python3.
#
# Usage: tools/android-mobilenet-tpu-smoke.sh [abi]   (abi defaults to arm64-v8a)
set -euo pipefail

ABI="${1:-arm64-v8a}"
case "$ABI" in
    arm64-v8a)    TRIPLE=aarch64-linux-android;   LIBCXX_ARCH=aarch64-linux-android ;;
    armeabi-v7a)  TRIPLE=armv7-linux-androideabi; LIBCXX_ARCH=arm-linux-androideabi ;;
    x86_64)       TRIPLE=x86_64-linux-android;    LIBCXX_ARCH=x86_64-linux-android ;;
    x86)          TRIPLE=i686-linux-android;      LIBCXX_ARCH=i686-linux-android ;;
    *) echo "unknown ABI '$ABI' (use arm64-v8a | x86_64 | armeabi-v7a | x86)" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
FIX="$REPO/g2g-ml/tests/fixtures/mobilenet"
VENV="${MOBILENET_VENV:-/tmp/g2g-onnxvenv}"

if ! command -v adb >/dev/null; then echo "adb not found on PATH" >&2; exit 1; fi
if [ -z "$(adb devices | awk 'NR>1 && $2=="device"{print $1}')" ]; then
    echo "no authorised device; 'adb devices' shows:" >&2
    adb devices >&2
    exit 1
fi

# Build the uint8-input model + input fixture on demand (gen_u8in.py needs onnx).
if [ ! -f "$FIX/mn_u8in.onnx" ] || [ ! -f "$FIX/mn_input_f32.bin" ]; then
    if [ ! -x "$VENV/bin/python" ]; then
        echo ">> creating venv at $VENV"
        python3 -m venv "$VENV"
        "$VENV/bin/pip" -q install --upgrade pip
        "$VENV/bin/pip" -q install onnx onnxruntime numpy pillow
    fi
    echo ">> generating mn_u8in.onnx"
    "$VENV/bin/python" "$FIX/gen_u8in.py"
fi
read -r SCALE ZP < "$FIX/u8in_quant.txt"
echo ">> model input quant: scale=$SCALE zero_point=$ZP"

echo ">> building android_mobilenet_tpu_probe for $ABI ($TRIPLE)"
cargo ndk --platform 27 -t "$ABI" build --release -p g2g-ml --features nnapi \
    --test android_mobilenet_tpu_probe

BIN="$(find "target/$TRIPLE/release/deps" -maxdepth 1 -type f -executable \
    -name 'android_mobilenet_tpu_probe-*' -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2)"
if [ -z "$BIN" ]; then echo "could not find the built test binary under target/$TRIPLE/release/deps" >&2; exit 1; fi
echo ">> built $BIN"

DEVDIR=/data/local/tmp
DEV="$DEVDIR/g2g_mobilenet_tpu_probe"
adb push "$BIN" "$DEV" >/dev/null
adb shell chmod 755 "$DEV"
adb push "$FIX/mn_u8in.onnx" "$DEVDIR/mn_u8in.onnx" >/dev/null
adb push "$FIX/mn_input_f32.bin" "$DEVDIR/mn_input_f32.bin" >/dev/null

NDK="${ANDROID_NDK_HOME:-}"
LIBCXX="$(find "$NDK" -name 'libc++_shared.so' -path "*/$LIBCXX_ARCH/*" 2>/dev/null | head -1)"
if [ -z "$LIBCXX" ]; then
    echo "could not find libc++_shared.so for $LIBCXX_ARCH under ANDROID_NDK_HOME ($NDK)" >&2
    exit 1
fi
adb push "$LIBCXX" "$DEVDIR/" >/dev/null

adb logcat -c >/dev/null 2>&1 || true
echo ">> running on device"
set +e
OUT="$(adb shell "LD_LIBRARY_PATH=$DEVDIR G2G_MN_SCALE=$SCALE G2G_MN_ZERO_POINT=$ZP $DEV --nocapture --test-threads=1" 2>&1)"
CODE=$?
set -e
echo "$OUT"
echo ">> Edge TPU / DarwiNN / NNAPI logcat during the run:"
adb logcat -d 2>/dev/null | grep -iE "edgetpu|darwinn|neuralnetworks|nnapi" | tail -30 || echo "    (none captured)"
adb shell rm -f "$DEV" "$DEVDIR/mn_u8in.onnx" "$DEVDIR/mn_input_f32.bin" >/dev/null 2>&1 || true

if echo "$OUT" | grep -q "test result: ok"; then
    echo ">> PASS"
    exit 0
fi
echo ">> FAIL (exit $CODE)"
exit 1
