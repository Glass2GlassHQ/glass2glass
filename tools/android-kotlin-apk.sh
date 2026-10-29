#!/usr/bin/env bash
# Build g2g-uniffi/examples/android as a debug-signed APK for one ABI and print
# its path last. Needs cargo-ndk with ANDROID_NDK_HOME and the rustup target, a
# JDK, and SDK build-tools plus a platform under ANDROID_SDK_ROOT (default
# ~/Android/sdk). No Gradle: kotlinc, d8 and aapt2 run directly.
#
# Usage: tools/android-kotlin-apk.sh [abi]   (default arm64-v8a)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/kotlin-tools.sh"

ABI="${1:-arm64-v8a}"
case "$ABI" in
    arm64-v8a) TRIPLE=aarch64-linux-android ;;
    x86_64)    TRIPLE=x86_64-linux-android ;;
    *) echo "unknown ABI '$ABI' (use arm64-v8a | x86_64)" >&2; exit 2 ;;
esac

SDK="${ANDROID_SDK_ROOT:-$HOME/Android/sdk}"
BUILD_TOOLS="$(find "$SDK/build-tools" -maxdepth 1 -mindepth 1 -type d | sort -V | tail -1)"
ANDROID_JAR="$(find "$SDK/platforms" -maxdepth 2 -name android.jar | sort -V | tail -1)"
MIN_SDK=26
EXAMPLE="$REPO/g2g-uniffi/examples/android"
OUT="$TARGET/android-kotlin-apk/$ABI"

fetch_kotlin_tools
generate_bindings "$OUT/generated"
(cd "$REPO" && cargo ndk --platform "$MIN_SDK" -t "$ABI" build --locked --release -p g2g-uniffi)

rm -rf "$OUT/jna" "$OUT/classes" "$OUT/dex" "$OUT/stage"
mkdir -p "$OUT/dex" "$OUT/stage/lib/$ABI"
unzip -q "$JNA_AAR" classes.jar "jni/$ABI/libjnidispatch.so" -d "$OUT/jna"
"$KOTLINC" -no-jdk -no-reflect -cp "$ANDROID_JAR:$OUT/jna/classes.jar" -d "$OUT/classes" \
    "$OUT/generated" "$REPO/g2g-uniffi/android/Surfaces.kt" "$EXAMPLE/MainActivity.kt"
find "$OUT/classes" -name '*.class' -print0 \
    | xargs -0 "$BUILD_TOOLS/d8" --release --min-api "$MIN_SDK" --lib "$ANDROID_JAR" \
        --output "$OUT/dex" "$KOTLIN_STDLIB" "$OUT/jna/classes.jar"

cp "$TARGET/$TRIPLE/release/libg2g_uniffi.so" "$OUT/jna/jni/$ABI/libjnidispatch.so" "$OUT/stage/lib/$ABI/"
cp "$OUT/dex/classes.dex" "$OUT/stage/"
"$BUILD_TOOLS/aapt2" link -o "$OUT/unsigned.apk" --manifest "$EXAMPLE/AndroidManifest.xml" \
    -I "$ANDROID_JAR" --min-sdk-version "$MIN_SDK" --target-sdk-version 35
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
