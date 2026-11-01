#!/usr/bin/env bash
# Build g2g-uniffi/examples/android on the g2g AAR as a debug-signed APK for one
# ABI and print its path last. Needs what tools/android-aar.sh needs plus SDK
# build-tools. No Gradle: kotlinc, d8 and aapt2 run directly.
#
# Usage: tools/android-kotlin-apk.sh [abi]   (default arm64-v8a)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/kotlin-tools.sh"

ABI="${1:-arm64-v8a}"
case "$ABI" in
    arm64-v8a | x86_64) ;;
    *) echo "unknown ABI '$ABI' (use arm64-v8a | x86_64)" >&2; exit 2 ;;
esac

SDK="${ANDROID_SDK_ROOT:-$HOME/Android/sdk}"
BUILD_TOOLS="$(find "$SDK/build-tools" -maxdepth 1 -mindepth 1 -type d | sort -V | tail -1)"
ANDROID_JAR="$(find "$SDK/platforms" -maxdepth 2 -name android.jar | sort -V | tail -1)"
MIN_SDK=26
EXAMPLE="$REPO/g2g-uniffi/examples/android"
OUT="$TARGET/android-kotlin-apk/$ABI"

AAR="$("$REPO/tools/android-aar.sh" "$ABI" | tail -1)"

rm -rf "$OUT/aar" "$OUT/jna" "$OUT/classes" "$OUT/dex" "$OUT/stage" "$OUT/assets"
mkdir -p "$OUT/dex" "$OUT/stage/lib/$ABI" "$OUT/assets"
cp "$REPO/g2g-plugins/tests/fixtures/h264_640x480.h264" "$OUT/assets/"
unzip -q "$AAR" -d "$OUT/aar"
unzip -q "$JNA_AAR" classes.jar "jni/$ABI/libjnidispatch.so" -d "$OUT/jna"
"$KOTLINC" -no-jdk -no-reflect -cp "$ANDROID_JAR:$OUT/jna/classes.jar:$OUT/aar/classes.jar" \
    -d "$OUT/classes" "$EXAMPLE/MainActivity.kt"
find "$OUT/classes" -name '*.class' -print0 \
    | xargs -0 "$BUILD_TOOLS/d8" --release --min-api "$MIN_SDK" --lib "$ANDROID_JAR" \
        --output "$OUT/dex" "$KOTLIN_STDLIB" "$OUT/jna/classes.jar" "$OUT/aar/classes.jar"

cp "$OUT/aar/jni/$ABI/libg2g_uniffi.so" "$OUT/jna/jni/$ABI/libjnidispatch.so" "$OUT/stage/lib/$ABI/"
cp "$OUT/dex/classes.dex" "$OUT/stage/"
"$BUILD_TOOLS/aapt2" link -o "$OUT/unsigned.apk" --manifest "$EXAMPLE/AndroidManifest.xml" \
    -I "$ANDROID_JAR" -A "$OUT/assets" --min-sdk-version "$MIN_SDK" --target-sdk-version 35
(cd "$OUT/stage" && zip -qr "$OUT/unsigned.apk" classes.dex lib)
"$BUILD_TOOLS/zipalign" -f 4 "$OUT/unsigned.apk" "$OUT/aligned.apk"
KEYSTORE="$HOME/.android/debug.keystore"
if [ ! -f "$KEYSTORE" ]; then
    mkdir -p "$HOME/.android"
    keytool -genkeypair -keystore "$KEYSTORE" -storepass android -keypass android \
        -alias androiddebugkey -keyalg RSA -keysize 2048 -validity 10000 \
        -dname "CN=Android Debug,O=Android,C=US"
fi
"$BUILD_TOOLS/apksigner" sign --ks "$KEYSTORE" --ks-pass pass:android \
    --out "$OUT/g2g-kotlin.apk" "$OUT/aligned.apk"
echo "$OUT/g2g-kotlin.apk"
