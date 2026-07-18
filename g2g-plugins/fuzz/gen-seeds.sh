#!/usr/bin/env bash
# Regenerate the cargo-fuzz seed corpora. Demuxer / parser targets gate on a
# magic signature, so the fuzzer needs a valid sample to reach the parsing;
# rtmp_handshake needs a crypto-valid C1/S1 (see seedgen). The network targets
# (rtp_depay, flexfec, st2110_dedup) self-bootstrap and need no seed.
set -euo pipefail
cd "$(dirname "$0")"
FF="ffmpeg -nostdin -hide_banner -loglevel error -y"
SRC="testsrc2=size=64x64:rate=15:duration=1"
SINE="sine=frequency=440:duration=1"

mkdir -p corpus/{mp4_streams,flv,matroska,ogg,cea_cdp,mpegts,rtmp_handshake}
$FF -f lavfi -i "$SRC" -c:v libx264 -pix_fmt yuv420p corpus/mp4_streams/seed.mp4
$FF -f lavfi -i "$SRC" -c:v libx264 -pix_fmt yuv420p -f flv corpus/flv/seed.flv
$FF -f lavfi -i "$SRC" -c:v libx264 -f matroska corpus/matroska/seed.mkv
$FF -f lavfi -i "$SRC" -c:v libx264 -pix_fmt yuv420p -f mpegts corpus/mpegts/seed.ts
$FF -f lavfi -i "$SINE" -c:a libvorbis corpus/ogg/seed.ogg 2>/dev/null \
    || $FF -f lavfi -i "$SINE" -c:a flac -f ogg corpus/ogg/seed.ogg
# CEA-708 CDP: starts with the 0x9669 identifier so the parser gets past it
printf '\x96\x69\x10\x3f\x43\x00\x72\xf8\xfc\x94\x2c\xf8\x80\x80\x74\x00\x00\x00' \
    > corpus/cea_cdp/seed.cdp
# RTMP: valid handshake signatures (built with the crate's key schedule)
cargo run --quiet --release --manifest-path seedgen/Cargo.toml -- corpus/rtmp_handshake

echo "seed corpora ready:"
du -sh corpus/* 2>/dev/null
