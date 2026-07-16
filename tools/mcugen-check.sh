#!/usr/bin/env bash
# Host graph-compiler proof (M646/M648): the `g2g-mcugen` compiler is
# trustworthy only if (1) its output is deterministic and the checked-in
# generated files are current, and (2) each generated pipeline produces the
# same bytes on the wire as the hand-written, reference-verified graph. This
# script asserts both, for both catalogs (audio + video/display):
#
#   1. Regenerate examples/mcugen-graphs/src/{audio,video}.rs from the flagship
#      graph documents and `git status`-check them: a drift means someone edited
#      a generated file by hand or changed the compiler without regenerating.
#   2. Run the mcugen-graphs equivalence tests, which drive each generated graph
#      with the reference peripherals and assert its wire checksum equals the
#      reference constant (AUDIO_EXPECTED_CHECKSUM / EXPECTED_CHECKSUM).
#
# Also builds the generated crate for a bare-metal thumb target, so "the
# compiler emits heap-free code" is checked, not assumed.
#
# Usage: tools/mcugen-check.sh
# Requires: cargo; rustup target thumbv7em-none-eabihf; git.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CRATE="$ROOT/examples/mcugen-graphs"
TARGET="thumbv7em-none-eabihf"

# Each catalog: <graph document> -> <generated module>.
declare -A GRAPHS=(
  ["$ROOT/g2g-mcugen/examples/flagship.yaml"]="$CRATE/src/audio.rs"
  ["$ROOT/g2g-mcugen/examples/display.yaml"]="$CRATE/src/video.rs"
)

for GRAPH in "${!GRAPHS[@]}"; do
  GENERATED="${GRAPHS[$GRAPH]}"
  echo "== regenerating $GENERATED from $GRAPH =="
  cargo run -q -p g2g-mcugen --manifest-path "$ROOT/Cargo.toml" -- "$GRAPH" -o "$GENERATED"

  echo "== asserting the checked-in generated file is current =="
  # Non-empty porcelain status means the regenerated file differs from the
  # committed one (` M`) or was never committed (`??`): either way, stale.
  if [ -n "$(git -C "$ROOT" status --porcelain -- "$GENERATED")" ]; then
    echo "FAIL: $GENERATED is out of date or uncommitted; commit the regenerated file:"
    git -C "$ROOT" --no-pager diff -- "$GENERATED" || true
    exit 1
  fi
done

echo "== building the generated graphs for $TARGET (heap-free) =="
rustup target add "$TARGET" >/dev/null 2>&1 || true
cargo build --manifest-path "$CRATE/Cargo.toml" --lib --target "$TARGET"

echo "== equivalence: each generated wire == its hand-written reference =="
cargo test --manifest-path "$CRATE/Cargo.toml" --quiet

echo "PASS: g2g-mcugen output is deterministic, current, heap-free for $TARGET,"
echo "      and reproduces the reference audio and display graphs byte-for-byte."
