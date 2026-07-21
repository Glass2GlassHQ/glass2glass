#!/usr/bin/env bash
# Build the M742 on-screen present harness as a NativeActivity APK, install it
# on a connected device, launch it, and check logcat for presented frames. The
# true SurfaceView present the /data/local/tmp probes cannot do: the APK's
# activity owns a real on-screen window, MediaCodec decodes zero-copy onto the
# GPU, and WgpuSink presents to it (examples/g2g-android-present).
#
# Prerequisites (beyond the bare-binary smoke scripts'):
#   - Android NDK via ANDROID_NDK_HOME + cargo-ndk + the rustup target
#     (see tools/android-mediacodec-smoke.sh).
#   - Android SDK build-tools + a platform (for aapt2 / zipalign / apksigner /
#     android.jar) via ANDROID_SDK_ROOT (default: ~/Android/sdk, ~/Android/Sdk).
#     No gradle: the APK is linked, zipped, aligned, and signed by hand.
#   - keytool (JRE) for the one-time debug keystore.
#   - adb with the device authorised.
#
# Usage: tools/android-apk-present-smoke.sh [abi]   (default arm64-v8a)
set -euo pipefail

ABI="${1:-arm64-v8a}"
case "$ABI" in
    arm64-v8a)    TRIPLE=aarch64-linux-android ;;
    armeabi-v7a)  TRIPLE=armv7-linux-androideabi ;;
    x86_64)       TRIPLE=x86_64-linux-android ;;
    *) echo "unknown ABI '$ABI' (use arm64-v8a | x86_64 | armeabi-v7a)" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE="$REPO/examples/g2g-android-present"
PKG=dev.g2g.present

# 1. Tooling: SDK build-tools + platform jar.
SDK="${ANDROID_SDK_ROOT:-}"
for cand in "$HOME/Android/sdk" "$HOME/Android/Sdk"; do
    [ -z "$SDK" ] && [ -d "$cand" ] && SDK="$cand"
done
if [ -z "$SDK" ] || [ ! -d "$SDK/build-tools" ]; then
    echo "no SDK build-tools: set ANDROID_SDK_ROOT (needs build-tools/ and platforms/)" >&2
    exit 1
fi
BT="$(find "$SDK/build-tools" -maxdepth 1 -mindepth 1 -type d | sort -V | tail -1)"
JAR="$(find "$SDK/platforms" -maxdepth 2 -name android.jar 2>/dev/null | sort -V | tail -1)"
for tool in "$BT/aapt2" "$BT/zipalign" "$BT/apksigner"; do
    [ -x "$tool" ] || { echo "missing $tool" >&2; exit 1; }
done
[ -n "$JAR" ] || { echo "no platforms/android-*/android.jar under $SDK" >&2; exit 1; }

if ! command -v adb >/dev/null; then echo "adb not found on PATH" >&2; exit 1; fi
if [ -z "$(adb devices | awk 'NR>1 && $2=="device"{print $1}')" ]; then
    echo "no authorised device; 'adb devices' shows:" >&2
    adb devices >&2
    exit 1
fi

# 2. Cross-compile the cdylib (the crate is its own workspace; build in-place).
echo ">> building g2g-android-present for $ABI ($TRIPLE)"
(cd "$CRATE" && cargo ndk --platform 26 -t "$ABI" build --release)
SO="$CRATE/target/$TRIPLE/release/libg2g_android_present.so"
[ -f "$SO" ] || { echo "missing $SO" >&2; exit 1; }

# 3. Package: aapt2 links the manifest (binary AXML), the lib is zipped in,
#    then align + sign with the debug keystore.
OUT="$CRATE/target/apk"
rm -rf "$OUT" && mkdir -p "$OUT/stage/lib/$ABI"
cp "$SO" "$OUT/stage/lib/$ABI/"
"$BT/aapt2" link -o "$OUT/unsigned.apk" --manifest "$CRATE/AndroidManifest.xml" \
    -I "$JAR" --min-sdk-version 26 --target-sdk-version 35 \
    --version-code 1 --version-name m742
(cd "$OUT/stage" && zip -qr "$OUT/unsigned.apk" lib)
"$BT/zipalign" -f 4 "$OUT/unsigned.apk" "$OUT/aligned.apk"
KS="$HOME/.android/debug.keystore"
if [ ! -f "$KS" ]; then
    mkdir -p "$HOME/.android"
    keytool -genkeypair -keystore "$KS" -storepass android -keypass android \
        -alias androiddebugkey -keyalg RSA -keysize 2048 -validity 10000 \
        -dname "CN=Android Debug,O=Android,C=US"
fi
"$BT/apksigner" sign --ks "$KS" --ks-pass pass:android \
    --out "$OUT/g2g-present.apk" "$OUT/aligned.apk"
echo ">> built $OUT/g2g-present.apk"

# 4. Install, grant the capture permissions (usable by future in-APK probes;
#    best-effort), launch, and give it a few seconds to decode + present. The
#    device must be unlocked: behind the keyguard the activity is stopped and
#    its window taken away, so nothing presents.
adb shell input keyevent KEYCODE_WAKEUP
adb shell wm dismiss-keyguard 2>/dev/null || true
if adb shell dumpsys window 2>/dev/null | grep -q "isKeyguardShowing=true"; then
    echo "device is locked; unlock it (a PIN keyguard cannot be dismissed over adb)" >&2
    exit 1
fi
adb install -r "$OUT/g2g-present.apk" >/dev/null
adb shell pm grant "$PKG" android.permission.RECORD_AUDIO 2>/dev/null || true
adb shell pm grant "$PKG" android.permission.CAMERA 2>/dev/null || true
adb logcat -c
adb shell am start -n "$PKG/android.app.NativeActivity" >/dev/null
echo ">> launched; presenting for 10s"
sleep 10

# 5. Evidence: the harness logs decoded/presented counts; grab a screenshot too.
LOG="$(adb logcat -d -s 'g2g-present:*' 2>/dev/null)"
echo "$LOG" | sed -n '1,40p'
SHOT="${TMPDIR:-/tmp}/g2g-present-screen.png"
adb exec-out screencap -p > "$SHOT" 2>/dev/null || true
[ -s "$SHOT" ] && echo ">> screenshot: $SHOT"
adb shell am force-stop "$PKG"

if echo "$LOG" | grep -q "presented "; then
    echo ">> M742 on-screen present: OK"
else
    echo ">> M742 on-screen present: no presented frames in logcat" >&2
    exit 1
fi
