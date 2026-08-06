#!/usr/bin/env bash
# Correctness gate for a pure-Rust Opus decode path (M916): decode identical
# packets through `opus-rs` and through libopus and compare the PCM. Fetches the
# RFC 8251 conformance vectors (~75 MB) once into tools/opus-rs-gate/vectors/.
#
# Usage: tools/opus-rs-gate.sh
# Requires: a system libopus (Fedora: opus-devel; Debian: libopus-dev), curl.
# Exits non-zero when the candidate does not reproduce libopus.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GATE="$ROOT/tools/opus-rs-gate"
VECTORS="$GATE/vectors"
TARBALL="$GATE/opus_testvectors-rfc8251.tar.gz"
URL="https://opus-codec.org/docs/opus_testvectors-rfc8251.tar.gz"

if [ ! -d "$VECTORS" ]; then
  echo "== fetching the RFC 8251 conformance vectors =="
  [ -f "$TARBALL" ] || curl -fL --retry 3 -o "$TARBALL" "$URL"
  mkdir -p "$VECTORS"
  tar xzf "$TARBALL" -C "$VECTORS" --strip-components=1
fi

echo "== running the gate =="
cargo run --manifest-path "$GATE/Cargo.toml" --release -- \
  "$VECTORS" "$ROOT/g2g-plugins/tests/fixtures"
