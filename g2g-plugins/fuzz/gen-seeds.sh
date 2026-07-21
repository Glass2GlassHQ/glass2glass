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

# Codec bitstream parsers: real elementary streams so the fuzzer starts past the
# start-code / magic gate and mutates inside the header bit readers. Guarded with
# `|| true`: a missing encoder just skips that seed (the parser self-bootstraps).
mkdir -p corpus/{av1parse,h264parse,h265parse,vp9parse,vp8parse,aacparse,opusparse}
$FF -f lavfi -i "$SRC" -c:v libx264 -pix_fmt yuv420p -f h264 corpus/h264parse/seed.h264
$FF -f lavfi -i "$SRC" -c:v libx265 -pix_fmt yuv420p -f hevc corpus/h265parse/seed.h265 2>/dev/null || true
$FF -f lavfi -i "$SRC" -c:v libvpx-vp9 -f ivf corpus/vp9parse/seed.ivf 2>/dev/null || true
$FF -f lavfi -i "$SRC" -c:v libvpx -f ivf corpus/vp8parse/seed.ivf 2>/dev/null || true
$FF -f lavfi -i "$SINE" -c:a aac -f adts corpus/aacparse/seed.aac 2>/dev/null || true
# AV1 raw OBU stream (low-overhead, size-delimited) from whichever encoder exists
$FF -f lavfi -i "$SRC" -c:v libsvtav1 -f obu corpus/av1parse/seed.obu 2>/dev/null \
    || $FF -f lavfi -i "$SRC" -c:v libaom-av1 -f obu corpus/av1parse/seed.obu 2>/dev/null || true
# Opus identification header (OpusHead magic, version 1, 2 ch, 48kHz) so
# parse_opus_head clears the magic gate
printf 'OpusHead\x01\x02\x38\x01\x80\xbb\x00\x00\x00\x00\x00' > corpus/opusparse/seed.opushead

# parse_launch: valid gst-launch-style descriptions so the fuzzer starts from
# real element / property / caps / link / bin syntax (the surface the g2g-capi
# C entry forwards) rather than rediscovering element names.
mkdir -p corpus/parse_launch
printf 'videotestsrc num-buffers=10 ! video/x-raw,width=320,height=240 ! fakesink' > corpus/parse_launch/seed1
printf 'filesrc location=a.mp4 ! decodebin ! videoconvert ! autovideosink' > corpus/parse_launch/seed2
printf 'audiotestsrc ! tee name=t ! queue ! fakesink t. ! queue ! fakesink' > corpus/parse_launch/seed3

# ivfdemux: a real IVF (DKIF header + frame headers). fmp4: a fragmented MP4 so
# the fuzzer starts inside moof / traf / trun instead of rediscovering the box
# structure. Both gate on a magic / box layout.
mkdir -p corpus/ivfdemux corpus/fmp4
$FF -f lavfi -i "$SRC" -c:v libvpx-vp9 -f ivf corpus/ivfdemux/seed.ivf 2>/dev/null \
    || $FF -f lavfi -i "$SRC" -c:v libvpx -f ivf corpus/ivfdemux/seed.ivf 2>/dev/null || true
$FF -f lavfi -i "$SRC" -c:v libx264 -pix_fmt yuv420p \
    -movflags +frag_keyframe+empty_moov+default_base_moof corpus/fmp4/seed.mp4 2>/dev/null || true

# srtcrypto: a structurally valid KM message (AES-128) with a garbage wrapped key.
# The header magic / version / SLen / KLen clear every early gate, so the fuzzer
# reaches PBKDF2 KEK derivation + the AES-KW unwrap attempt (which fails on the
# garbage key, as intended). 16-byte header, 16-byte salt, 24-byte wrapped key.
mkdir -p corpus/srtcrypto
printf '\x12\x20\x29\x01\x00\x00\x00\x00\x02\x00\x02\x00\x00\x00\x04\x04\xAA\xAA\xAA\xAA\xAA\xAA\xAA\xAA\xAA\xAA\xAA\xAA\xAA\xAA\xAA\xAA\xBB\xBB\xBB\xBB\xBB\xBB\xBB\xBB\xBB\xBB\xBB\xBB\xBB\xBB\xBB\xBB\xBB\xBB\xBB\xBB\xBB\xBB\xBB\xBB' > corpus/srtcrypto/seed.km

echo "seed corpora ready:"
du -sh corpus/* 2>/dev/null
