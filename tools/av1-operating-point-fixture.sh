#!/usr/bin/env bash
# Regenerate g2g-plugins/tests/fixtures/av1_320x256_l2t3_op0_base.ivf, an AV1
# stream whose operating point 0 leaves layers out. libaom always codes
# operating point 0 as every layer, so the script encodes two spatial and three
# temporal layers with libaom's svc_encoder_rtc example, then swaps
# operating_point_idc[0] with the point holding spatial layer 0 at temporal
# layers 0 and 1 (0x103). A decoder at operating point 0 outputs 160x128 base
# layer frames from a 320x256 sequence and drops temporal layer 2.
#
# Usage: tools/av1-operating-point-fixture.sh <path to libaom's svc_encoder_rtc>
# The checked-in fixture came from libaom v3.15.2.
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: tools/av1-operating-point-fixture.sh <svc_encoder_rtc>" >&2
    exit 2
fi
encoder="$1"
fixture="$(cd "$(dirname "$0")/.." && pwd)/g2g-plugins/tests/fixtures/av1_320x256_l2t3_op0_base.ivf"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

ffmpeg -v error -f lavfi -i testsrc2=size=320x256:rate=30 -frames:v 12 \
    -pix_fmt yuv420p "$work/source.yuv"
# layering mode 7 is two spatial layers with three temporal layers each
"$encoder" --width=320 --height=256 --spatial-layers=2 --temporal-layers=3 \
    --layering-mode=7 --target-bitrate=100 --bitrates=20,30,40,40,50,60 \
    --speed=9 --kf-dist=1000 --output="$work/encoded.ivf" "$work/source.yuv" >/dev/null

# svc_encoder_rtc writes one IVF frame per layer, the AV1 test vectors one per temporal unit
python3 - "$work/encoded.ivf" "$fixture" 6 0x103 <<'PY'
import struct
import sys

source, target = sys.argv[1], sys.argv[2]
temporal_unit_count, first_operating_point = int(sys.argv[3]), int(sys.argv[4], 0)
OBU_SEQUENCE_HEADER = 1
OBU_TEMPORAL_DELIMITER = 2
IVF_HEADER_BYTES = 32
IVF_FRAME_HEADER_BYTES = 12
IVF_FRAME_COUNT_OFFSET = 24
data = open(source, "rb").read()


def leb128(buffer, at):
    value = 0
    for index in range(8):
        byte = buffer[at]
        at += 1
        value |= (byte & 0x7F) << (7 * index)
        if not byte & 0x80:
            return value, at
    sys.exit("leb128 longer than 8 bytes")


def obus(unit):
    at = 0
    while at < len(unit):
        header = unit[at]
        payload = at + 1 + ((header >> 2) & 1)
        size, payload = leb128(unit, payload)
        yield (header >> 3) & 15, payload
        at = payload + size


class Bits:
    def __init__(self, buffer, byte_offset):
        self.buffer, self.position = buffer, byte_offset * 8

    def read(self, count):
        value = 0
        for _ in range(count):
            byte = self.buffer[self.position >> 3]
            value = (value << 1) | ((byte >> (7 - (self.position & 7))) & 1)
            self.position += 1
        return value

    def write(self, position, count, value):
        for index in range(count):
            at, shift = (position + index) >> 3, 7 - ((position + index) & 7)
            bit = (value >> (count - 1 - index)) & 1
            self.buffer[at] = (self.buffer[at] & ~(1 << shift)) | (bit << shift)


def move_operating_point_first(unit, payload):
    bits = Bits(unit, payload)
    bits.read(4)  # seq_profile, still_picture
    if bits.read(1):
        sys.exit("a reduced still picture header has no operating points")
    if bits.read(1):
        sys.exit("timing_info is not handled")
    initial_display_delay_present = bits.read(1)
    points = []
    for _ in range(bits.read(5) + 1):
        points.append((bits.position, bits.read(12)))
        if bits.read(5) > 7:
            bits.read(1)  # seq_tier
        if initial_display_delay_present and bits.read(1):
            bits.read(4)
    idcs = [idc for _, idc in points]
    swapped = idcs.index(first_operating_point)
    bits.write(points[0][0], 12, first_operating_point)
    bits.write(points[swapped][0], 12, idcs[0])


units = []
at = IVF_HEADER_BYTES
while at < len(data):
    (size,) = struct.unpack("<I", data[at : at + 4])
    timestamp = data[at + 4 : at + IVF_FRAME_HEADER_BYTES]
    frame = data[at + IVF_FRAME_HEADER_BYTES : at + IVF_FRAME_HEADER_BYTES + size]
    if not units or (frame[0] >> 3) & 15 == OBU_TEMPORAL_DELIMITER:
        units.append((timestamp, bytearray()))
    units[-1][1].extend(frame)
    at += IVF_FRAME_HEADER_BYTES + size
units = units[:temporal_unit_count]

for _, unit in units:
    for obu_type, payload in obus(unit):
        if obu_type == OBU_SEQUENCE_HEADER:
            move_operating_point_first(unit, payload)

header = bytearray(data[:IVF_HEADER_BYTES])
struct.pack_into("<I", header, IVF_FRAME_COUNT_OFFSET, len(units))
out = bytearray(header)
for timestamp, unit in units:
    out += struct.pack("<I", len(unit)) + timestamp + unit
open(target, "wb").write(out)
PY
echo "$fixture $(stat -c '%s' "$fixture") bytes"
