#!/usr/bin/env bash
# Build g2g-uniffi, generate its Kotlin bindings, and run g2g-uniffi/examples/jvm
# on the desktop JVM. Needs a JDK on PATH.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/kotlin-tools.sh"

OUT="$TARGET/kotlin-jvm-example"
fetch_kotlin_tools
generate_bindings "$OUT/generated" kotlin
"$KOTLINC" -include-runtime -cp "$JNA_JAR" -d "$OUT/example.jar" \
    "$OUT/generated" "$REPO/g2g-uniffi/examples/jvm/Example.kt"
java --enable-native-access=ALL-UNNAMED -Djna.library.path="$TARGET/release" \
    -cp "$OUT/example.jar:$JNA_JAR" ExampleKt
