#!/usr/bin/env bash
# Regenerate the QuickTime audio sample entry fixtures in g2g-plugins/tests/fixtures from ffmpeg.
#
# Each mov_<entry>_<layout>.mov is ffmpeg's mov muxer writing a short stereo tone
# in one PCM layout, and <file>.<format> beside it is ffmpeg's decode of that file
# to the graph format g2g carries it as. At 8 kHz the muxer writes a version 1
# sound description named for the sample type (in24, in32, fl32, fl64, with an
# enda atom for little-endian), and at 96 kHz a version 2 `lpcm` one whose flags
# carry the layout. The f64 files have no graph format, so they get no decode.
# mov_vorbis.mov is a libvorbis encode in ffmpeg's `msVo` entry, with ffmpeg's
# copy of its packets as .packets and its decode as .s16le.
set -euo pipefail

fixtures="$(cd "$(dirname "$0")/.." && pwd)/g2g-plugins/tests/fixtures"

pcm() {
    local name=$1 rate=$2 codec=$3 decode=$4
    local file="$fixtures/mov_${name}.mov"
    ffmpeg -v error -y -f lavfi -i "sine=frequency=440:sample_rate=${rate}:duration=0.01" \
        -ac 2 -c:a "$codec" -f mov "$file"
    if [ -n "$decode" ]; then
        ffmpeg -v error -y -i "$file" -c:a "pcm_${decode}" -f "$decode" "$file.$decode"
    fi
}

pcm in24_be 8000 pcm_s24be s24le
pcm in24_le 8000 pcm_s24le s24le
pcm in32_le 8000 pcm_s32le s32le
pcm fl32_be 8000 pcm_f32be f32le
pcm fl32_le 8000 pcm_f32le f32le
pcm fl64_be 8000 pcm_f64be ""
pcm lpcm_s16le 96000 pcm_s16le s16le
pcm lpcm_s24be 96000 pcm_s24be s24le
pcm lpcm_f32be 96000 pcm_f32be f32le
pcm lpcm_u8 96000 pcm_u8 u8
pcm lpcm_s8 96000 pcm_s8 u8
pcm lpcm_f64le 96000 pcm_f64le ""

vorbis="$fixtures/mov_vorbis.mov"
ffmpeg -v error -y -f lavfi -i "sine=frequency=440:sample_rate=8000:duration=0.05" \
    -ac 2 -c:a libvorbis -f mov "$vorbis"
ffmpeg -v error -y -i "$vorbis" -map 0:a -c:a copy -f data "$vorbis.packets"
ffmpeg -v error -y -i "$vorbis" -c:a pcm_s16le -f s16le "$vorbis.s16le"
