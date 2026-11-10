#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# Regenerate the Matroska CodecDelay fixtures in g2g-plugins/tests/fixtures from ffmpeg.
#
# Each mkv_<codec>_codec_delay.mka is a short stereo tone whose encoder priming
# ffmpeg's muxer writes as the track's CodecDelay, and the MP3 one also ends on a
# block with DiscardPadding. <file>.s16le beside it is ffmpeg's decode of that
# file, which drops both.
set -euo pipefail

fixtures="$(cd "$(dirname "$0")/.." && pwd)/g2g-plugins/tests/fixtures"

tone() {
    local codec=$1 encoder=$2
    local file="$fixtures/mkv_${codec}_codec_delay.mka"
    ffmpeg -v error -y -f lavfi -i "sine=frequency=440:sample_rate=44100:duration=0.2" \
        -ac 2 -c:a "$encoder" -f matroska "$file"
    ffmpeg -v error -y -i "$file" -c:a pcm_s16le -f s16le "$file.s16le"
}

tone mp3 libmp3lame
tone aac aac
tone mp2 mp2
