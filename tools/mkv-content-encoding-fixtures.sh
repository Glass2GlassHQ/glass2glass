#!/usr/bin/env bash
# Regenerate the Matroska ContentEncoding fixtures in g2g-plugins/tests/fixtures
# and check that ffmpeg demuxes each single-encoding file to the same packets
# and CodecPrivate as its plain file.
#
# Python's bz2 and zlib and python-lzo (liblzo2's lzo1x_1) compress the payloads,
# so the decoders under test never read their own output. The ASS files encode
# the blocks and the CodecPrivate, the text files chain a header strip under
# zlib. ffmpeg refuses chained encodings, so the chain file is not compared.
set -euo pipefail

fixtures="$(cd "$(dirname "$0")/.." && pwd)/g2g-plugins/tests/fixtures"

uv run --no-project --python 3.13 --with python-lzo==1.15 python -I - "$fixtures" <<'PY'
import bz2
import struct
import sys
import zlib
from pathlib import Path

import lzo

fixtures = Path(sys.argv[1])

ZLIB, BZIP2, LZO, HEADER_STRIP = 0, 1, 2, 3
SCOPE_BLOCK, SCOPE_CODEC_PRIVATE = 1, 2
SUBTITLE_TRACK_TYPE = 0x11

ASS_HEADER = b"""[Script Info]
ScriptType: v4.00+
PlayResX: 384
PlayResY: 288

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,16,&Hffffff,&Hffffff,&H0,&H0,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,0
Style: Sign,Arial,12,&H00ffff,&H00ffff,&H0,&H0,1,0,0,0,100,100,0,0,1,1,0,8,10,10,10,0

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
ASS_EVENTS = [
    b"0,0,Default,,0,0,0,,The first line of the subtitle track, the first line.",
    b"1,0,Default,,0,0,0,,The second line of the subtitle track, the second line.",
    b"2,0,Sign,,0,0,0,,{\\an8}A sign at the top of the picture, a sign.",
]
TEXT_PREFIX = b"- "
TEXT_EVENTS = [
    b"- Where are you going with the subtitle track?",
    b"- Home, with the subtitle track.",
    b"- Then go home, and take the subtitle track with you.",
]
EVENT_SPACING_MS = 1500
EVENT_DURATION_MS = 1200


def vint(value):
    for length in range(1, 9):
        if value < (1 << (7 * length)) - 1:
            return ((1 << (7 * length)) | value).to_bytes(length, "big")
    sys.exit(f"size {value} does not fit an EBML vint")


def element(element_id, body):
    return element_id.to_bytes((element_id.bit_length() + 7) // 8, "big") + vint(len(body)) + body


def uint(element_id, value):
    return element(element_id, value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big"))


def compress(algo, settings, data):
    if algo == ZLIB:
        return zlib.compress(data, 9)
    if algo == BZIP2:
        return bz2.compress(data, 9)
    if algo == LZO:
        return lzo.compress(data, 1, False)
    if not data.startswith(settings):
        sys.exit(f"{data!r} does not start with the stripped header {settings!r}")
    return data[len(settings) :]


# each step is (order, scope, algo, settings), and the writer applies the lowest order first
def encode(steps, scope, data):
    for _, step_scope, algo, settings in sorted(steps):
        if step_scope & scope:
            data = compress(algo, settings, data)
    return data


def content_encodings(steps):
    encodings = b""
    for order, scope, algo, settings in steps:
        compression = uint(0x4254, algo)
        if settings:
            compression += element(0x4255, settings)
        encodings += element(
            0x6240,
            uint(0x5031, order)
            + uint(0x5032, scope)
            + uint(0x5033, 0)
            + element(0x5034, compression),
        )
    return element(0x6D80, encodings)


def matroska(codec_id, codec_private, events, steps):
    header = element(
        0x1A45DFA3,
        uint(0x4286, 1)
        + uint(0x42F7, 1)
        + uint(0x42F2, 4)
        + uint(0x42F3, 8)
        + element(0x4282, b"matroska")
        + uint(0x4287, 4)
        + uint(0x4285, 2),
    )
    duration_ms = EVENT_SPACING_MS * (len(events) - 1) + EVENT_DURATION_MS
    info = element(
        0x1549A966,
        uint(0x2AD7B1, 1_000_000)
        + element(0x4D80, b"mkv-content-encoding-fixtures")
        + element(0x5741, b"mkv-content-encoding-fixtures")
        + element(0x4489, struct.pack(">d", duration_ms)),
    )
    entry = uint(0xD7, 1) + uint(0x73C5, 1) + uint(0x83, SUBTITLE_TRACK_TYPE) + uint(0x9C, 0)
    entry += element(0x86, codec_id)
    if codec_private:
        entry += element(0x63A2, encode(steps, SCOPE_CODEC_PRIVATE, codec_private))
    if steps:
        entry += content_encodings(steps)
    tracks = element(0x1654AE6B, element(0xAE, entry))
    cluster = uint(0xE7, 0)
    for index, event in enumerate(events):
        stored = encode(steps, SCOPE_BLOCK, event)
        block = vint(1) + struct.pack(">hB", index * EVENT_SPACING_MS, 0) + stored
        cluster += element(0xA0, element(0xA1, block) + uint(0x9B, EVENT_DURATION_MS))
    segment = element(0x18538067, info + tracks + element(0x1F43B675, cluster))
    return header + segment


files = {
    "mkv_ass_plain.mks": (b"S_TEXT/ASS", ASS_HEADER, ASS_EVENTS, []),
    "mkv_ass_bzip2.mks": (
        b"S_TEXT/ASS",
        ASS_HEADER,
        ASS_EVENTS,
        [(0, SCOPE_BLOCK | SCOPE_CODEC_PRIVATE, BZIP2, b"")],
    ),
    "mkv_ass_lzo.mks": (
        b"S_TEXT/ASS",
        ASS_HEADER,
        ASS_EVENTS,
        [(0, SCOPE_BLOCK | SCOPE_CODEC_PRIVATE, LZO, b"")],
    ),
    "mkv_ass_zlib_codec_private.mks": (
        b"S_TEXT/ASS",
        ASS_HEADER,
        ASS_EVENTS,
        [(0, SCOPE_CODEC_PRIVATE, ZLIB, b"")],
    ),
    "mkv_text_plain.mks": (b"S_TEXT/UTF8", b"", TEXT_EVENTS, []),
    "mkv_text_strip_zlib.mks": (
        b"S_TEXT/UTF8",
        b"",
        TEXT_EVENTS,
        [(0, SCOPE_BLOCK, HEADER_STRIP, TEXT_PREFIX), (1, SCOPE_BLOCK, ZLIB, b"")],
    ),
}
for name, (codec_id, codec_private, events, steps) in files.items():
    (fixtures / name).write_bytes(matroska(codec_id, codec_private, events, steps))

# raw payloads for the decoder unit tests, longer than the decoders' first 4 KiB output buffer,
# and a bzip2 stream one byte past the 16 MiB block bound
payload = ASS_HEADER * 16
(fixtures / "mkv_encoding_payload.txt").write_bytes(payload)
(fixtures / "mkv_encoding_payload.bz2").write_bytes(bz2.compress(payload, 9))
(fixtures / "mkv_encoding_payload.lzo").write_bytes(lzo.compress(payload, 1, False))
(fixtures / "mkv_encoding_zeros.bz2").write_bytes(bz2.compress(bytes(16 * 1024 * 1024 + 1), 9))
PY

framemd5() {
    ffmpeg -v error -i "$fixtures/$1" -map 0 -c copy -f framemd5 -
}

plain=$(framemd5 mkv_ass_plain.mks)
for encoded in mkv_ass_bzip2.mks mkv_ass_lzo.mks mkv_ass_zlib_codec_private.mks; do
    if ! diff <(printf '%s\n' "$plain") <(framemd5 "$encoded"); then
        echo "ffmpeg demuxes $encoded differently from mkv_ass_plain.mks" >&2
        exit 1
    fi
done
echo "ffmpeg demuxes every single-encoding file like its plain file"
