#!/usr/bin/env bash
# Regenerate g2g-plugins/tests/fixtures/h264_clean_aperture.mov and ffmpeg's
# decode of it.
#
# ffmpeg's mov muxer writes no clap box, so the script muxes three testsrc2
# frames into a MOV and inserts one into the avc1 sample entry. The clap is a
# 45x33 aperture moved 3 columns right and 2 rows up from the centre of the 64x48
# picture: its edges fall on half pixels and its size is odd, so the fixture
# also covers ffmpeg's rounding down to whole 4:2:0 chroma samples.
# <name>.yuv420p is ffmpeg's default decode, which applies the clap.
set -euo pipefail

fixtures="$(cd "$(dirname "$0")/.." && pwd)/g2g-plugins/tests/fixtures"
mov="$fixtures/h264_clean_aperture.mov"

ffmpeg -v error -y -f lavfi -i "testsrc2=size=64x48:rate=10" -frames:v 3 \
    -c:v libx264 -bf 0 -pix_fmt yuv420p -f mov "$mov"

# clap fields: aperture width, aperture height, horizontal and vertical offset, each numerator then denominator
python3 - "$mov" 45 1 33 1 3 1 -2 1 <<'PY'
import struct
import sys

path = sys.argv[1]
clap_fields = [int(field) for field in sys.argv[2:10]]
data = bytearray(open(path, "rb").read())


def children(start, end):
    at = start
    while at < end:
        size, kind = struct.unpack(">I4s", data[at : at + 8])
        if size < 8:
            sys.exit(f"{path}: unsupported box size {size} at {at}")
        yield at, size, kind
        at += size


def child(parent, kind, header=8):
    start, size = parent
    for at, child_size, child_kind in children(start + header, start + size):
        if child_kind == kind:
            return at, child_size
    sys.exit(f"{path}: no {kind.decode()} box")


top = {kind: (at, size) for at, size, kind in children(0, len(data))}
moov = top[b"moov"]
if moov[0] < top[b"mdat"][0]:
    sys.exit(f"{path}: moov ahead of mdat, a longer moov would move the samples")


def video_trak():
    for at, size, kind in children(moov[0] + 8, moov[0] + moov[1]):
        if kind != b"trak":
            continue
        hdlr = child(child((at, size), b"mdia"), b"hdlr")
        # full box version and flags, pre_defined, then the handler type
        if data[hdlr[0] + 16 : hdlr[0] + 20] == b"vide":
            return at, size
    sys.exit(f"{path}: no video trak")


trak = video_trak()
mdia = child(trak, b"mdia")
minf = child(mdia, b"minf")
stbl = child(minf, b"stbl")
stsd = child(stbl, b"stsd")
# stsd is a full box with an entry count, and its first entry is the avc1
avc1 = next(children(stsd[0] + 16, stsd[0] + stsd[1]))[:2]
clap_payload = struct.pack(">8i", *clap_fields)
clap = struct.pack(">I4s", 8 + len(clap_payload), b"clap") + clap_payload
data[avc1[0] + avc1[1] : avc1[0] + avc1[1]] = clap
for start, _ in [moov, trak, mdia, minf, stbl, stsd, avc1]:
    (size,) = struct.unpack(">I", data[start : start + 4])
    data[start : start + 4] = struct.pack(">I", size + len(clap))
open(path, "wb").write(data)
PY

ffmpeg -v error -y -i "$mov" -f rawvideo -pix_fmt yuv420p "$mov.yuv420p"
ffprobe -v error -select_streams v:0 -show_entries stream=width,height:stream_side_data "$mov"
ffmpeg -v error -apply_cropping none -i "$mov" -f rawvideo -pix_fmt yuv420p - | wc -c
wc -c <"$mov.yuv420p"
