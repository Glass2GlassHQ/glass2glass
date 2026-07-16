#!/usr/bin/env bash
# Build the g2g browser cdylib into tools/wasm-demo/pkg/ with wasm-bindgen glue.
#
# Requires: rustup wasm32 target (`rustup target add wasm32-unknown-unknown`) and
# wasm-pack (`cargo install wasm-pack`). The unstable cfg is for the WebCodecs
# web-sys bindings pulled in by g2g-plugins' `web-codecs` feature.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"

# Use the rustup toolchain (Fedora's /usr/bin/rustc lacks the wasm32 std).
export PATH="$HOME/.cargo/bin:$PATH"
export RUSTFLAGS="--cfg=web_sys_unstable_apis"

echo "building g2g-web -> $here/pkg (target web)"
wasm-pack build "$repo/g2g-web" \
  --release \
  --target web \
  --out-dir "$here/pkg" \
  --out-name g2g_web \
  --no-typescript

echo
echo "done. Now, in two terminals:"
echo "  1) cargo run --release --manifest-path $here/ws-fixture-server/Cargo.toml"
echo "  2) $here/serve.sh   # then open http://127.0.0.1:8000/"
