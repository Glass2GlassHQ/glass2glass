#!/usr/bin/env bash
# Fetch the whisper transcription test fixtures (g2g-ml/tests/whisper_transcribe.rs):
# whisper.cpp's quantized tiny.en ggml model (32 MB) and its jfk.wav sample
# (11 s of 16 kHz mono s16 speech). Not committed (repo fixtures are KB-scale),
# so this fetches them on demand into a gitignored dir; CI caches the dir.
#
# After running:
#   cargo test -p g2g-ml --features whisper --test whisper_transcribe -- --nocapture
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
FIX="$HERE/g2g-ml/tests/fixtures/whisper"

fetch() {
  local name="$1" url="$2"
  if [ -s "$FIX/$name" ]; then
    echo "have $name"
    return
  fi
  echo "fetching $name"
  curl -fsSL --retry 3 -o "$FIX/$name.part" "$url"
  mv "$FIX/$name.part" "$FIX/$name"
}

fetch ggml-tiny.en-q5_1.bin "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q5_1.bin"
fetch jfk.wav "https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav"

echo
echo "fixtures ready in $FIX"
echo "run: cargo test -p g2g-ml --features whisper --test whisper_transcribe -- --nocapture"
