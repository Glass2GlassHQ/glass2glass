#!/usr/bin/env bash
# Fetch the llama chat test fixture (g2g-ml/tests/llama_chat.rs): the 1.2 MB
# stories260K GGUF llama.cpp's own CI generates with, a tiny TinyStories model
# with no chat template. Not committed, fetched on demand into a gitignored dir;
# CI caches the dir.
#
# After running:
#   cargo test -p g2g-ml --features llama --test llama_chat -- --nocapture
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
FIX="$HERE/g2g-ml/tests/fixtures/llama"
NAME=stories260K.gguf

if [ -s "$FIX/$NAME" ]; then
  echo "have $NAME"
else
  echo "fetching $NAME"
  curl -fsSL --retry 3 -o "$FIX/$NAME.part" "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/$NAME"
  mv "$FIX/$NAME.part" "$FIX/$NAME"
fi

echo
echo "fixtures ready in $FIX"
echo "run: cargo test -p g2g-ml --features llama --test llama_chat -- --nocapture"
