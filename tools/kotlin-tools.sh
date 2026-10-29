# Sourced by the Kotlin and Swift scripts. Downloads kotlinc and JNA once into
# ~/.cache/g2g-kotlin, checks their sha256, and generates the g2g-uniffi bindings.

KOTLIN_VERSION=2.4.21
KOTLIN_SHA256=7cc140e76daf416a0424a5557f99afd7ca89ebd463ea1101261e3e01c33e75cd
JNA_VERSION=5.19.1
JNA_JAR_SHA256=4fb141dd8ef6b0585ffceea4bc49602fbc6312fa977e2c488794ea3e6aafecae
JNA_AAR_SHA256=b57125cb7d16253f0d65a80f7d3a4c3664effa711b8bdbb7f87fb572ce1624ed

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${CARGO_TARGET_DIR:-$REPO/target}"
KOTLIN_TOOLS="${XDG_CACHE_HOME:-$HOME/.cache}/g2g-kotlin"
KOTLINC="$KOTLIN_TOOLS/kotlinc-$KOTLIN_VERSION/bin/kotlinc"
KOTLIN_STDLIB="$KOTLIN_TOOLS/kotlinc-$KOTLIN_VERSION/lib/kotlin-stdlib.jar"
JNA_JAR="$KOTLIN_TOOLS/jna-$JNA_VERSION.jar"
JNA_AAR="$KOTLIN_TOOLS/jna-$JNA_VERSION.aar"
MAVEN_JNA="https://repo1.maven.org/maven2/net/java/dev/jna/jna/$JNA_VERSION"

fetch() {
    local url="$1" file="$2" sha256="$3"
    if [ ! -f "$file" ]; then
        curl -sSfL "$url" -o "$file.part"
        echo "$sha256  $file.part" | sha256sum -c --quiet
        mv "$file.part" "$file"
    fi
}

fetch_kotlin_tools() {
    mkdir -p "$KOTLIN_TOOLS"
    local zip="$KOTLIN_TOOLS/kotlin-compiler-$KOTLIN_VERSION.zip"
    fetch "https://github.com/JetBrains/kotlin/releases/download/v$KOTLIN_VERSION/kotlin-compiler-$KOTLIN_VERSION.zip" \
        "$zip" "$KOTLIN_SHA256"
    fetch "$MAVEN_JNA/jna-$JNA_VERSION.jar" "$JNA_JAR" "$JNA_JAR_SHA256"
    fetch "$MAVEN_JNA/jna-$JNA_VERSION.aar" "$JNA_AAR" "$JNA_AAR_SHA256"
    if [ ! -x "$KOTLINC" ]; then
        rm -rf "$KOTLIN_TOOLS/kotlinc" "$KOTLIN_TOOLS/kotlinc-$KOTLIN_VERSION"
        unzip -q "$zip" -d "$KOTLIN_TOOLS"
        mv "$KOTLIN_TOOLS/kotlinc" "$KOTLIN_TOOLS/kotlinc-$KOTLIN_VERSION"
    fi
}

# Builds the host library, which the JVM and Swift examples also load, and writes the $2 bindings to $1.
generate_bindings() {
    local out="$1" language="$2"
    (cd "$REPO" && cargo build --locked --release -p g2g-uniffi)
    rm -rf "$out"
    (cd "$REPO" && cargo run --locked --release -p g2g-uniffi --features bindgen --bin uniffi-bindgen -- \
        generate --library "$TARGET/release/libg2g_uniffi.so" --language "$language" --no-format \
        --out-dir "$out")
}
