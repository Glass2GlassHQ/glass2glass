#!/usr/bin/env bash
# Build the g2g caps-solver validator into tools/builder/public/wasm/ (target
# web), so the visual builder can run g2g's real solver client-side. Vite serves
# public/ at the root in dev and copies it into dist on build, so both the dev
# server and the static bundle load it; the single-file artifact (strict CSP)
# can't fetch it and falls back to the family heuristic. Regenerate after changing
# the core solver, parser, or registry. The output is gitignored (a built-on-
# demand binary blob).
#
# Requires: rustup wasm32 target + wasm-pack (`cargo install wasm-pack`).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"

# Use the rustup toolchain (Fedora's /usr/bin/rustc lacks the wasm32 std).
export PATH="$HOME/.cargo/bin:$PATH"
export RUSTFLAGS="--cfg=web_sys_unstable_apis"

echo "building g2g-validate-wasm -> $here/public/wasm (target web)"
wasm-pack build "$repo/g2g-validate-wasm" \
  --release \
  --target web \
  --out-dir "$here/public/wasm" \
  --out-name g2g_validate \
  --no-typescript

echo "done: $here/public/wasm"
