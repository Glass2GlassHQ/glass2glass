#!/usr/bin/env bash
# Regenerate the WebM Encryption fixtures in g2g-plugins/tests/fixtures.
#
# ffmpeg encodes the clear VP9 and Opus files, Shaka Packager encrypts them with
# a raw key, and ffmpeg's framemd5 of each clear file is the packets the
# decrypted blocks must match. ffmpeg's matroskadec cannot decrypt, so the
# encrypted VP9 file's framemd5 is its blocks as stored. The VP9 file is two-pass
# with alt-ref frames, so its superframes carry four partitions. The Opus file
# keeps its first 0.1 s clear, so it holds unencrypted frames in an encrypted track.
#
# PACKAGER names the Shaka Packager binary (tested with v3.9.3 packager-linux-x64).
set -euo pipefail

fixtures="$(cd "$(dirname "$0")/.." && pwd)/g2g-plugins/tests/fixtures"
packager="${PACKAGER:-$HOME/opt/shaka-packager/packager-linux-x64}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cd "$work"

key_id=0123456789abcdef0123456789abcdef
key=00112233445566778899aabbccddeeff
video="testsrc2=size=96x64:rate=10"
vp9=(-frames:v 12 -c:v libvpx-vp9 -g 12 -b:v 200k -an)

ffmpeg -hide_banner -loglevel error -f lavfi -i "$video" "${vp9[@]}" -pass 1 -f null /dev/null
ffmpeg -hide_banner -loglevel error -f lavfi -i "$video" "${vp9[@]}" -pass 2 \
    -auto-alt-ref 1 -lag-in-frames 16 vp9_clear.webm
ffmpeg -hide_banner -loglevel error -f lavfi -i sine=frequency=440:sample_rate=48000 -t 0.3 \
    -c:a libopus -b:a 32k opus_clear.webm

encrypt() {
    "$packager" "in=$1,stream=$2,output=$3" --enable_raw_key_encryption \
        --keys "label=:key_id=$key_id:key=$key" --iv "$4" "${@:5}" >/dev/null 2>&1
}
encrypt vp9_clear.webm video "$fixtures/webm_vp9_encrypted.webm" 0102030405060708 --clear_lead 0
encrypt opus_clear.webm audio "$fixtures/webm_opus_encrypted.webm" 1112131415161718 \
    --clear_lead 0.1 --segment_duration 0.1

framemd5() {
    ffmpeg -hide_banner -loglevel quiet -i "$1" -map 0 -c copy -f framemd5 "$2"
}
framemd5 vp9_clear.webm "$fixtures/webm_vp9_clear.webm.framemd5"
framemd5 opus_clear.webm "$fixtures/webm_opus_clear.webm.framemd5"
framemd5 "$fixtures/webm_vp9_encrypted.webm" "$fixtures/webm_vp9_encrypted.webm.framemd5"
printf 'key_id=%s\nkey=%s\n' "$key_id" "$key" >"$fixtures/webm_encryption.keys"
