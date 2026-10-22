# Containers and byte streams

This file covers the container demuxers and muxers, the byte-stream caps that
feed them, adaptive streaming over HTTP, and the still-image formats that take
the same shape. Part of the design in [README.md](README.md).

## Byte stream caps

A container demuxer splits one stored or transported byte stream into the typed
elementary streams it carries. The link feeding a demuxer is
`Caps::ByteStream { encoding }`, the first byte-stream caps variant: an opaque
container stream not yet demuxed, tagged with a `ByteStreamEncoding` such as
`MpegTs` so a demuxer accepts only the format it parses. It is the
byte-stream-level analog of the codec/raw video split. A byte source declares
it, `FileSrc::new(path, Caps::ByteStream{MpegTs})`, and the demuxer's transform
constraint maps it to the elementary stream type.

A launch `filesrc` types its file from the extension, and sniffs the header when
the extension is unknown or `bytestream-format=auto`. An extension that names a
container is checked against the header too: a different container the sniff
recognizes wins, so a transport stream saved as `.mpg` reaches `tsdemux`. When the sniff recognizes
nothing, `probe_output_caps` returns an error and `decodebin` fails the parse
with `ParseError::UnrecognizedContent` instead of guessing a container. `.m4v`
has no extension mapping, since it names both an MP4 and a raw MPEG-4 Part 2
stream. The sniff types a raw H.264 or H.265 stream once it has seen two kinds
of parameter set behind well-formed NAL units, so a start-code run inside an ID3
tag or an MXF partition does not read as video.

## Encoding profiles

`encodebin` is the encode direction of `decodebin`, and it is a parse-time
macro for the same reason: bins are flattened here.
`profile="<container-caps>:<stream-caps>[:<stream-caps>...]"` expands into one
encoder per stream plus the muxer for the container, chosen through two registry
hooks the plugins crate fills in, `set_encoder_provider` and
`set_muxer_provider`. The choice is a table of launch names rather than a
search, because the auto-plug candidate set is decode-only and an encoder is
picked by codec, not found by walking caps.

Anything the profile does not name negotiates, and what it does name is applied
by an element:

- a stream part that pins a `width` or `height` splices `videoscale`
- one that pins a `framerate` splices `videorate`
- a `rate` or `channels` sets the `audioresample` and `audioconvert` the audio
  side already carries
- a pinned `bitrate` is a property on the encoder, refused when that encoder
  declares no such property

Any other field is refused rather than dropped, so a profile never claims a
setting the pipeline does not apply. The bin's `name=` moves to the muxer,
which lets a second branch reach it as `e.` through the ordinary fan-in path,
and that branch gets the encoder for its own kind of input. `transcodebin` is
rewritten to `decodebin ! encodebin` before either expands. The expansion also
splices the converter an encoder needs, asked of the encoder itself: a profile
is written against codecs, so the pixel format or sample rate a source happens
to produce is the macro's problem, not the caller's.

## splitfilesrc and dataurisrc

Two sources have no single file behind them and so derive their own type.
`splitfilesrc` joins the parts a pattern matches into one byte stream, typing it
from the first part's extension or header, since a name like `clip.ts.part003`
has no usable extension. `dataurisrc` decodes a `data:` URI's payload and types
it by sniffing the bytes, never by the URI's declared MIME type. Both expose the
resolved type through `probe_output_caps`, so `decodebin` plans from what the
source will actually produce.

## splitmuxsrc

`splitmuxsrc` plays the parts a `splitmuxsink` wrote, matched by a wildcard
`location` and played in name order. Each part is its own container and is
demuxed on its own: an MP4 part through `Mp4Src`, a Matroska or MPEG-TS part
through that container's single-stream demuxer, set by the primary-stream hook to
the part's video track (its audio track when it has none). Every part has to
carry the first part's codec.

The first part keeps its own timestamps. Each later part is shifted so its first
frame lands where the previous part ended, which holds whether its muxer
restarted at zero (MP4) or kept counting (Matroska, MPEG-TS). A part ends at its
highest PTS plus that frame's duration, and a frame with no duration (MPEG-TS)
counts as the smallest PTS gap seen between neighbouring frames. The shift is the
`shift_packet` rule `gaplesssrc` and `fallbacksrc` use. A later part's
stream-start `Segment` is dropped, a repeated `CapsChanged` is suppressed, and
sequence numbers run on across parts.

## Raw byte streams

`ByteStreamEncoding::Raw` is the one encoding with no container and no framing:
a headerless dump such as `.yuv` or `.pcm` whose shape is declared out of band.
Its readers are `rawvideoparse` and `rawaudioparse`, which cut the stream into
frames from their own `format`, geometry and rate properties. Content sniffing
never answers `Raw`, since every byte sequence matches it, so `filesrc` reaches
it only by extension or an explicit `bytestream-format=raw`, and no auto-plug
candidate claims it.

## Audio frame parsers

`aacparse`, `mpegaudioparse` and `ac3parse` share `locate_frame`, which confirms a
candidate frame by the sync header at its end, or by an ID3v2 tag there, since HLS
segments of a bare ADTS or MP3 stream each open on one. A buffered tag found while
scanning is skipped. `aacparse` holds input until the held bytes show LOAS, ADTS or a
raw AudioSpecificConfig, up to 64 KiB or EOS, before it falls back to forwarding
them as LATM.

## Uncompressed video in containers

`avidemux`, `matroskademux` and `qtdemux` emit an uncompressed video stream as
`Caps::RawVideo`, so `decodebin` plays it with no decoder in the chain. The
layout is named by the AVI `biCompression`, the `ColourSpace` of a Matroska
`V_UNCOMPRESSED` track or the fourcc of a `V_MS/VFW/FOURCC` one, and the
QuickTime sample entry: `I420` / `IYUV` is I420, `YUY2` / `YUYV` / `YUNV` /
`yuvs` is YUY2, `UYVY` / `2vuy` / `HDYC` / `UYNV` is UYVY, `v210` is v210. A
demuxer selects such a stream as `stream=raw-i420`, `raw-yuy2`, `raw-uyvy` or
`raw-v210`. The frame size comes from the `BITMAPINFOHEADER`, the track's
`PixelWidth` / `PixelHeight`, or the sample entry, never the `tkhd` display
size. A stored frame longer than its rows, v210's padded ones included, is cut
to them, which is how libavcodec reads it, and a shorter one is dropped.

## Plain-text files

`filesrc` types a `.txt` or `.text` file as `Caps::Text { Utf8 }`, and
`bytestream-format=text` does the same for any other name, such as a prompt kept
as `.md`. Content sniffing never answers plain text, since almost any small file
of ASCII bytes is valid UTF-8, so the extension and the property are the only two
ways in. GStreamer spells this as a `text/x-raw,format=utf8` caps filter after
`filesrc`, whose source pad takes any caps. A g2g source declares its caps before
negotiation, so in g2g that caps filter checks the type rather than setting it.

## AIFF and AU

Uncompressed audio files sit next to WAVE. `ByteStreamEncoding::Aiff` is EA IFF
85, a `FORM`/`AIFF` or `AIFC` file with a `COMM` descriptor and an `SSND` sample
chunk, and `ByteStreamEncoding::Au` is the Sun / NeXT `.snd` header. Both carry
multi-byte PCM big-endian. `aiffparse` and `auparse` swap to the little-endian
`AudioFormat` the rest of the graph uses, and `aiffmux` and `avmux_au` swap
back. `typefind` and `filesrc` type them by magic or extension: `aiff`, `aif`,
`aifc`, `au`, `snd`.

## MPEG-TS

`g2g-plugins::mpegts::TsDemuxer` is a pure `no_std + alloc` parser: it takes
188-byte packets, reads the PAT and then the PMT it names, and reassembles PES
per PID into access units with PTS. The `TsDemux` element wraps it. The parser
reassembles every elementary stream the PMT names. The element has one output
pad, so a `TsStream` selection picks which to emit, `H264` or `H265` video as
`CompressedVideo` and `Aac` audio as `Audio`, defaulting to H.264, and a second
`tsdemux` selecting another stream demuxes the rest of the multiplex.

Selection is by codec, not by a runtime-discovered "first video", because the
output pad's media type is fixed at negotiation before any packet is parsed.
H.264 and H.265 are distinct downstream decoders, not a refinement. Video
geometry is unknown until the bitstream parser reads the SPS, so the demuxer
advertises a fixatable placeholder `Range` refined downstream via `CapsChanged`,
the `RtspSrc` pattern in [caps.md](caps.md). AAC advertises the
sentinel
channels and rate that `aacparse` refines from the ADTS header.

### Packet framing and damaged streams

`TsPacketSplitter` cuts the elements' byte input into packets. It reads
188-byte packets, 192-byte M2TS packets with a 4-byte arrival timestamp in
front, and 204-byte DVB-ASI packets with 16 parity bytes behind. It locks a
stride once the sync byte repeats at it for 4 packets, tries 188 first, and
searches again when a packet does not start with the sync byte. At end of
stream it hands over a last packet that is shorter than its stride.

The demuxer checks each PID's continuity counter on packets that carry payload.
A packet that repeats the counter once is a duplicate and is skipped. Any other
gap is a lost packet: the PID's partial PES or table section is dropped, and
nothing more is collected on it until the next unit start. A packet whose
adaptation field sets `discontinuity_indicator` starts a new counter run and
drops nothing.

PAT, PMT, SDT and EIT sections reassemble across packets, capped at 1024 bytes
for the first three and 4096 for the EIT.

A PMT that lists one PID twice keeps the PID at its first position with the
last entry's stream type and descriptors, as ffmpeg does. One broadcast mux
lists each AC-3 PID as MPEG audio and then again as a private PES with an AC-3
descriptor.

Packets on PIDs no table names yet are held until the active program's PMT
parses, then replayed in order, so a capture that starts between two PMT
repeats keeps its first access units. At most 8192 packets are held, 0.5 s of
a 24 Mbit/s multiplex, which is the longest PAT/PMT gap ETSI TR 101 290
allows. The oldest is dropped when the limit is reached, and null packets are
never held.

PTS and DTS are 33-bit counters that wrap about every 26.5 hours. The demuxer
places each one against the last timestamp on its PID: the same cycle, one wrap
ahead, or one wrap behind once the PID has wrapped, whichever lands within an
hour of it. A larger jump keeps the raw value. The elements see timestamps that
keep rising across the wrap, and a flush or seek resets this with the rest of
the parser.

### Timestamping unstamped access units

A conforming mux need only carry a PES timestamp every 700 ms, so most access
units arrive unstamped and forwarding them that way lands them all at 0. The
demuxer times each one itself:

- a stream whose SPS proves coded order is display order advances one frame
  period per unit
- an MPEG-2 stream takes its display slot from the picture header's
  `temporal_reference`
- one that may reorder takes it from the picture order count, anchored on the
  real stamps

A single stamp carries a reordering stream when the SPS declares the frame
period, since the count step per frame follows from the codec, H.265 counting
pictures and H.264 counting fields, and the second real stamp then measures the
slope exactly. An H.264 encoder may step the count once per frame instead,
which the first odd count of a frame-coded stream proves, so that provisional
step switches to one. A field-coded stream is left alone, its bottom fields
counting odd legitimately.

### TsMuxer

`g2g-plugins::mpegts::TsMuxer` is the inverse path, wrapping access units back
into PES and 188-byte packets with a real PSI CRC. It is multi-stream:
`with_streams` builds one program carrying N elementary streams, each on its own
PID and named in one PMT. It is also multi-program: `with_programs` takes a
`(program_number, stream_type)` per stream, so the PAT names each program, each
program gets its own PMT naming only its own streams, and each gets its own PCR
on its first stream's PID.

The fan-in element exposes that layout as `prog-map`, one program number per
input pad in pad order, `prog-map=1,1,2`. GStreamer's `mpegtsmux` takes a
pad-name structure there and g2g takes a comma list because its properties are
scalar. The single-input `tsmux::TsMux` element wraps a one-stream muxer,
`! mpegtsmux !`. The multi-input `tsmuxn::TsMux` is a `MultiInputElement` that
muxes A+V, interleaving access units across inputs by decode time (the DTS when
a frame carries one, else the PTS) via the `take_earliest_by` merge, so the
multiplex is in decode order and no PES trails a PCR taken from another stream.
The `mpegtsmux` name is registered both as the single-input launch element and
as a fan-in muxer, so the text parser picks `tsmux::TsMux` for one input and
`tsmuxn::TsMux` for several by link degree, as in
`v.! m.  a.! m.  mpegtsmux name=m`.

Tables repeat. The PAT and every PMT go out together every `pat-interval`
(`pmt-interval` is the same cadence), and the SDT every `si-interval`, both in
90 kHz ticks with GStreamer's default of 9000 (100 ms). 0 writes a table once.

The PCR cadence (`pcr-interval`, default 3600 ticks) is per program. On an
access unit of any stream of a program whose PCR is due, the PCR rides the PES's
first packet when the unit is on the PCR_PID, and otherwise goes out first in a
PCR-only packet on the PCR_PID: adaptation field only, continuity counter
repeated. So the first packet of a program after its tables is a PCR. A keyframe
sets the random_access_indicator in its PES's first packet, and on the PCR_PID
also carries a PCR, as ffmpeg does, so a receiver tuning in at a keyframe gets a
clock. Teletext gets no indicator, since EN 300 472 forbids an adaptation field
beside its payload.

Every written PTS and DTS is the input value plus a decoder delay,
`TsMuxer::decoder_delay_90khz`. In variable-bitrate mode (`bitrate=0`, the
default) the delay is 100 ms and a PCR is the access unit's input clock (DTS,
else PTS), so data reaches the decoder 100 ms before its decode time. A demuxer
sees the shifted timestamps.

With `bitrate` set the multiplex is constant-bitrate. One byte clock runs for
the whole multiplex: a packet's time is the start plus the bytes before it times
8 over the bitrate, in 27 MHz, and a PCR carries the byte clock at its own byte,
extension included. The delay becomes 700 ms, ffmpeg's default mux delay, and
the byte clock starts at the first input decode time. Before each access unit
the muxer pads until the byte clock reaches the unit's input decode time, with
a PCR-only packet when a PCR is due and a null packet otherwise, and tables and
PCRs come due by the byte clock. Input that outruns the bitrate logs one warning
per stream and keeps muxing. Nothing pads at EOS.

The identity knobs are fixed once the muxer is built, like `prog-map`:
`transport-stream-id` (PAT and SDT), `original-network-id` (SDT), `start-pid`
and `pmt-start-pid` (the first elementary stream and PMT PID, the rest
following in order, 0x0010 to 0x1FFE) and `tables-version` (the PSI
version_number, 0 to 31). Building fails when the stream and PMT PIDs collide
with each other or the SDT PID. `bitrate` is fixed at build too.

### Tags over TS

Tags ride TS through its standard carriers. The muxers' `with_tags` and
`with_track_tags` write the SDT `service_descriptor`, taking the service name
from `Tag::Title` and the provider from the ffprobe-spelled `service_provider`
key, plus a per-stream ISO-639 language descriptor in the PMT. The demuxers
CRC-check and parse both and post `BusMessage::Tag` and `StreamTag` on the
`mpegts-pid-{pid}` ids. Both directions are validated against ffmpeg. Nothing
else rides TS: it has no free-form tag element.

Service text is per program. `with_program_tags(program, tags)` on the fan-in
muxer gives a `prog-map` program its own SDT entry, `with_tags` names whichever
programs do not, and a program's `Tag::Language` is the default for its streams.
The resolution order is global, then program, then track. The SDT describes the
whole multiplex, so a demuxer posts one `BusMessage::Tag` per service it names,
each carrying that service's `program_number`, whichever program the element
routes.

### AV1 over TS

AV1 rides TS on the same private PES, stream_type 0x06, that the KLV carriage
uses, told apart by its `registration_descriptor`. AV1 has no `stream_type` of
its own, so a 0x06 stream is AV1 only when its PMT entry names one. The mux
writes the 'AV1G' format_identifier and the demux accepts that and the AOM
spec's 'AV01', because only 'AV1G' has a reader: GStreamer's `mpegtsmux` and
`tsdemux` predate the spec and call the mapping custom,
`enable-custom-mappings=true`, while ffmpeg's muxer writes AV1 with no
descriptor at all, a bare 0x06 its own demuxer reports as `bin_data`.

Each PES payload is one temporal unit in the low-overhead OBU format, which
`av1parse` and the AV1 decoders read unchanged, and `TsStream::Av1`,
`tsdemux stream=av1`, selects it. The seek resume point reads the AV1 frame
header rather than Annex-B start codes. Both directions are validated against
GStreamer: a `svtav1enc ! mpegtsmux` stream demuxes to units ffmpeg's `obu`
demuxer decodes at full size, and `tsdemux ! av1parse ! dav1ddec` decodes the
g2g mux's output.

### DVB EIT

The DVB EIT on PID 0x12 adds what a service is showing. The demuxers parse the
present/following table, `table_id` 0x4E sections 0 and 1, and post each
service's `short_event_descriptor` name and text as a `BusMessage::Tag` scoped
to its `program_number`, under the `event_name` / `event_text` and
`next_event_name` / `next_event_text` keys. `Tag::Title` on that program is
already the SDT service name.

This table changes during the stream, unlike the PAT, PMT and SDT, so a section
is read when its `version_number` differs from the one last accepted for the
same `(service_id, table_id, section_number)`, and a table repeating itself
costs nothing. A section also routinely outgrows one packet, so sections
reassemble across packets behind a `table_id` filter that keeps the
other-transport-stream tables sharing the PID out of the buffer. Event text
goes through the same annex A decoder as the SDT names, which also decodes the
UTF-8 character table.

Each event's timing rides the same tags. The 5-byte `start_time`, a Modified
Julian Date and BCD hh:mm:ss per EN 300 468 Annex C, posts as `event_start` and
`next_event_start` in seconds since the Unix epoch, UTC, and the BCD `duration`
as `event_duration` and `next_event_duration` in seconds. Each is omitted when
the stream declared the field undefined or encoded it invalidly, rather than
posting a zero.

The schedule tables, `table_id` 0x50..=0x5F, parse into a separate bounded queue
the demuxers drain through `TsDemuxer::take_eit_schedule`. A schedule section
names days of events rather than the two present/following holds, so each event
posts as its own `BusMessage::Tag` under the `schedule_event_id` /
`schedule_event_name` / `schedule_event_text` / `schedule_event_start` /
`schedule_event_duration` keys, which lets a consumer building an EPG tell whose
start time is whose.

## KLV and STANAG 4609

`Caps::Klv` is the metadata elementary-stream caps, GStreamer's `meta/x-klv`,
each frame one SMPTE ST 336 key-length-value packet. STANAG 4609 is the
airborne-ISR profile of MPEG-TS that carries it.

### KLV carriage over TS

On the mux side a `Caps::Klv` input becomes a private PES, stream_type 0x06 on
PES `private_stream_1`, whose PMT entry carries the `KLVA` registration
descriptor, the MISB ST 1402 asynchronous carriage ffmpeg keys on. The demux
side accepts both that and metadata-in-PES, stream_type 0x15, the synchronous
carriage, via `TsStream::Klv`, `tsdemux stream=klv`, filtering generic 0x06 PIDs
by the registration the way Opus and DVB AC-3 selection does.

`klv-sync` on the mux elements selects the strict synchronous form instead:
stream_type 0x15 on PES `stream_id` 0xFC, each local set wrapped in one ISO
13818-1 metadata AU cell whose 5-byte header ffmpeg's demuxer skips per ST 1402,
and a `metadata_descriptor`, tag 0x26 'KLVA', in the PMT entry. Measurement
showed ffmpeg requires that descriptor to identify a 0x15 stream at all. The AU
cell layout is cross-checked against mediacommon's table 2-97 implementation.
The demux unwraps AU cells behind a validation gate, where cells must tile the
payload exactly and each must open with the ST 336 prefix, and forwards anything
else raw, so both the strict and the bare-payload sync forms decode.

Interop is validated against ffmpeg both ways: ffprobe identifies the g2g mux's
stream as `klv` and extracts its bytes bit-exact, and a TS re-authored by
ffmpeg's muxer demuxes back bit-exact.

### UAS Datalink local sets

`g2g-plugins::klv` is a pure `no_std` MISB ST 0601 UAS Datalink Local Set codec.
`UasDatalink` decodes and encodes the core telemetry tags, precision timestamp,
platform attitude, sensor position, FOV and relative angles, and frame center,
with the standard's fixed-point scalings. BER lengths and BER-OID tags are
bounds-checked and the 16-bit sum checksum, tag 1, is required on parse, so a
corrupted set is rejected whole.

The tag table covers the practical ST 0601 core: telemetry angles and positions,
the identity strings for mission id, platform designation, image source sensor
and coordinate system, slant range and target width, the four offset corner
points, target location, and the nested MISB ST 0102 security local set, tag 48,
as a typed `SecurityLocalSet` holding the classification enum preserved even for
unknown codes, the country coding methods and the classifying and object
countries. Every scale factor is cross-checked against the independent klvdata
implementation and the whole parser is validated against the published MISMMS
reference packet. That packet's declared checksum is provably wrong, 0xAA43
declared against 0x3E1E actual, and klvdata's own sum agrees, which is why
`parse`, strict and the `klvdecode` default, is paired with `parse_lenient` and
a `verify-checksum` property: real encoders get checksums wrong, and the caller
chooses whether that drops the set.

The `klvdecode` element turns each set into a timed `Text{Utf8}` `key=value`
line, so `tsdemux stream=klv ! klvdecode ! textoverlay` overlays live telemetry.
The encode direction is the `UasDatalink::encode` API through an app source.

### KLV over RTP

KLV also rides RTP directly, RFC 6597, as `rtpklv`: a sans-IO
`RtpKlvPacketizer` and `RtpKlvDepayloader` pair mirroring the H.264 `rtppay` and
`rtpdepay` shape, with 90 kHz timestamps, MTU fragmentation, the marker bit
closing each KLVunit, and whole-unit discard on any lost fragment. A fragment
carries no unit header, so resync waits for the next marker.

### VMTI

`vmti` is the MISB ST 0903 moving-target set, ST 0601 tag 74, nested with no UL
or checksum and carrying both when standalone. It decodes the VTarget series
with ST 1201 IMAPB scaling along with each target's nested VMask as a pixel
polygon or run-length mask, VObject ontology class, VTracker track id, life
cycle, velocity and acceleration packs, and VChip image chip.
`vmti_from_analytics` turns a frame's `AnalyticsMeta` detections into VTargets,
a tracked detection carrying its `object_id` as the target id, so an
in-pipeline detector emits standards-compliant VMTI.

### MIIS, misptime and cotsink

The ST 1204 MIIS core identifier, tag 94, round-trips exactly, refusing rather
than half-reading an identifier it cannot reproduce, and renders the standard
text form of grouped hex UUIDs with the Appendix B permutation check value.

`misptime` puts MISB ST 0604 microsecond timestamps in an H.264 or H.265 SEI so
video frames and KLV correlate after a remux. Extraction emits text cues rather
than restamping PTS, since an absolute epoch time on a frame would read as
decades of lateness to every sink.

`cotsink` maps decoded telemetry to Cursor-on-Target XML for TAK and ATAK, one
event per local set with the platform as the point and the frame center as a
`<sensor>` cone. With `spi=true` it also emits the ST 0805.1 Sensor Point of
Interest event, `b-m-p-s-p-i` at the target location or frame center, linked to
the platform track by `<link relation="p-p">`, following jmisb's `KlvToCot`
conventions.

### st2022fec

`st2022fec` is SMPTE 2022-1, Pro-MPEG COP3, FEC for TS over RTP. It is the wire
format only: the 2D row and column XOR algebra and the receiver bookkeeping are
implemented once in `ulpfec` and serve `flexfec` and this alike. It derives each
repair's protected set from that repair's own SNBase, offset and NA rather than
learning global L and D, so a mid-stream geometry change still decodes, and it
refuses a repair whose type field is not XOR instead of applying the wrong
algorithm to it.

### Reference implementations

Every wire format here was verified against a primary implementation rather than
prose: jmisb for ST 0903, ST 1204 and ST 0805, GStreamer's `video-sei` parser
plus a real capture vector for ST 0604, FFmpeg's `prompeg` and GStreamer's
`rtpst2022-1-fecenc`, which agree field for field, for ST 2022-1, and MITRE's
CoT schema with pytak's constants for CoT. Where a field could not be confirmed,
the codec preserves the raw bytes or declines to emit rather than guessing: the
VTarget location pack's accuracy tail is kept verbatim, and the unconfirmed CoT
detail elements are not written.

## Matroska and WebM

The same parser plus element split keyed on `Caps::ByteStream{Matroska}`.
`g2g-plugins::matroska::MatroskaDemuxer` is a pure EBML parser: variable-length
element IDs and sizes, descend into the Segment, read Tracks for the elementary
streams and `Info` TimestampScale, then parse each Cluster's SimpleBlock and
Block frames with scaled timestamps. `MkvDemux` wraps it with a per-codec
`MkvStream` selection, H.264 / H.265 / VP8 / VP9 / AV1 / MJPEG video and AAC /
Opus / AC-3 / FLAC / Vorbis / PCM audio, defaulting to VP9. An `A_PCM/*` track
takes the PCM format its byte order and `BitDepth` name, the `WAVEFORMATEX`
table WAV and AVI use, and big-endian samples are swapped at demux. A track's `ContentEncodings` chain is undone at demux, highest
`ContentEncodingOrder` first, on the blocks and on the `CodecPrivate` as each
step's scope names: zlib, header stripping, and with `matroska-compression`
bzip2 (`libbz2-rs-sys`) and LZO1X (`lzo`). A block that does not decode or
decodes past 16 MiB is dropped, and so is a `CodecPrivate` past 1 MiB. An
encryption with no supplied key, an unknown algorithm, a scope naming the next encoding, a repeated
order or a chain over eight steps flags the track `unsupported_encoding`. The
parser still returns its blocks as stored, but selecting a flagged track in
`MkvDemux` or `MkvDemuxN` fails the stream before any of its frames go out, and
`forwardable_streams` and `subtitle_streams` leave it out, so playbin never plugs
it. With `matroska-decrypt`, `with_content_keys` on
`MatroskaDemuxer`, `MkvDemux` and `MkvDemuxN` maps each 16-byte
`ContentEncKeyID` to its AES-128 key, resolved when `Tracks` parses. A WebM
Encryption step (ContentEncAlgo 5, AES-CTR, block scope only) is undone in its
chain position: the signal byte, the 8-byte IV and, on a partitioned frame, the
partition table are removed, and the encrypted sections, alternating with clear
ones from a clear first section, are deciphered with one counter run across them.
A frame with the extension bit, a partition bit without the encryption bit, a
short IV or table, or descending or out-of-frame offsets is dropped and logged
once per track. A frame with the encryption bit clear passes through. `aes_ctr`
is the AES-CTR shared with Common Encryption. MJPEG is `V_MJPEG` or a
`V_MS/VFW/FOURCC` track whose BITMAPINFOHEADER names `MJPG`. A Vorbis or
`V_THEORA` track's `CodecPrivate` Xiph-laces the three headers, which go in
band as three packets ahead of the first block. A Vorbis track's blocks go
untimed, because millisecond block timestamps cannot carry the sample-exact packet lengths `vorbisdec` trims to.
WebM, the VP8/VP9/AV1 plus Opus subset, is the browser-delivery motivator.
Block lacing, Xiph / EBML / fixed, is split, so multi-frame audio blocks demux.
A Cluster child that names a track `Tracks` did not declare, or whose size is
unknown or overflows, marks damaged data: the parser drops bytes up to the next
Cluster ID and carries on there, as ffmpeg does.

Matroska's Tracks element carries concrete geometry and audio parameters, so the
demuxer refines the output caps itself via `CapsChanged` once Tracks is parsed,
without a downstream bitstream parser. The frame rate is `1 s / DefaultDuration`,
and `Rate::Any` when the track has no `DefaultDuration` or the rate falls outside
the 1 to 240 fps placeholder the demuxer negotiated. An H.264 or H.265 track's blocks are
converted from the container-native AVCC / HVCC length-prefixed framing,
declared by the `avcC` / `hvcC` `CodecPrivate` whose `lengthSizeMinusOne` sets
the prefix width, to the Annex-B framing the pipeline assumes, with the config
record's parameter sets prepended on keyframes, ffmpeg's `h264_mp4toannexb`
discipline. The whole-block length walk is validated exactly, so a nonstandard
Annex-B block passes through unchanged instead of being mis-framed.

### Subtitle tracks

A `S_TEXT/UTF8` subtitle track maps to `MkvCodec::Subtitle(Utf8)` and fans out
of `MkvDemuxN` as a `Caps::Text { Utf8 }` port, `MkvStream::Subtitle`, with the
cue's display window carried on the frame and the `BlockGroup`'s
`BlockDuration` scaled onto `MkvFrame.duration_ns`. A `SimpleBlock` leaves that
`0`. `S_TEXT/ASS` and `S_TEXT/WEBVTT` are likewise de-framed to plain
`Text{Utf8}` cue text via the `CodecPrivate` header, and `mkv_playbin`
auto-plugs the subtitle overlay of [text.md](text.md).
`S_VOBSUB` is the bitmap case: `MkvCodec::VobSub` maps to
`Caps::SubPicture { VobSub }`, `MkvStream::VobSub`, whose blocks are forwarded
verbatim as subpicture units after the track's `.idx` `CodecPrivate` goes out in
band ahead of them.

### Seeking through Cues

The `Cues` index is parsed into a time to Cluster-byte-position map,
`cue_seek_offset`, and `MkvDemux` seeks through it in three tiers in
`poll_seek`:

- with `Cues` parsed it byte-seeks straight to the target Cluster,
  `DemuxSeek::poll_request_indexed`, keeping Tracks and TimestampScale across
  the mid-segment landing through `reset_keeping_tracks`
- with only a `SeekHead` locating an end-of-file `Cues` it prefetches them
  first, a byte-seek to `Cues`, a parse, then `begin_indexed_seek` to the target
  Cluster, with the internal prefetch flush consumed so downstream sees one only
  on the real seek
- with neither it re-scans from offset 0

`CueClusterPosition` and `SeekPosition` are relative to the Segment data start,
which the parser tracks.

### matroskamux

`matroskamux` is `MatroskaMuxer` plus the `MkvMux` element, the inverse path,
writing the EBML header, an unknown-size Segment, Tracks, and one Cluster per
frame, with the `webm` DocType for the WebM codec subset. Scope is one Segment
and one track with definite-size Clusters. Multi-track A/V muxing is the sibling
`mkvmuxn`.

Both muxers also have a `seekable` two-pass mode: the element buffers the file
and finalizes it at EOS with a front `SeekHead`, fixed-layout entries indexing
Info / Tracks / Chapters / Tags / Cues with the Cues position patched in place
once known, so the file seeks from byte 0 without reading past the Clusters. It
is mutually exclusive with `streamable`, and the default streaming output is
unchanged. The same finalize fills an `Info` `Duration` reserved beside them.
The value is the highest block end across tracks, each block's timestamp plus
its own duration rounded to a `TimestampScale` tick, which is how ffmpeg arrives
at the number it writes, so a remux reports the length ffmpeg's own file does.
Only this mode can carry a duration at all, since a streaming caller has emitted
its header long before the total is known, and a live stream has no length to
declare.

### Matroska tags

The Segment `Tags` element carries metadata in both directions, per file and per
track. A `Tag` whose `Targets` names a `TagTrackUID` scopes to that track, and a
nested `SimpleTag` flattens to a `parent/child` key. The muxers take per-track
metadata from `with_track_tags` and write each track's `TrackUID` in its
`TrackEntry`. The demuxers map a parsed UID back to its track and post the tags
as `BusMessage::StreamTag` on that stream's collection id, leaving untargeted
tags on `BusMessage::Tag`.

A track's title and language are the exception. They live in the `TrackEntry`
itself as `Name` and `Language`, with `LanguageBCP47` preferred when both are
present, which is where ffmpeg writes them and where a player reads them, so the
muxers route `Tag::Title` and `Tag::Language` there instead of writing a
`SimpleTag`, and the demuxers merge both sources into one `StreamTag` per
stream. A missing `Language` stays absent rather than becoming the spec's
implicit `eng`.

## Chapters

Chapters, the table of contents that GStreamer calls `GstToc`, travel an
out-of-band route in both Matroska and MP4. `g2g_core::Chapter` is the shared
shape: a stream-time start in nanoseconds, an optional end, a title, an optional
language, and nested sub-chapters. A demuxer posts what it parsed as
`BusMessage::Chapters`, once, like the tags, so an application builds a chapter
menu and seeks to a start without touching the data path. A muxer takes the same
list through `with_chapters`.

Matroska holds the whole shape: the `Chapters` element's `EditionEntry` and
`ChapterAtom` tree, whose times are unscaled nanoseconds rather than
`TimestampScale` ticks, with nesting and a per-chapter `ChapLanguage`. The
muxers write one default edition. The demuxer skips a hidden edition or atom,
since it is not meant to reach a menu, and bounds both the nesting depth and the
chapter count because the file supplies them.

MP4 carries less. The reader prefers the QuickTime chapter text track, a media
`trak` pointed at by `tref/chap` whose samples are the titles timed by its own
sample table, so each chapter gets an end, and falls back to the Nero
`udta/chpl` list, a flat array of starts in 100 ns ticks with no ends and no
nesting. The writers emit `chpl` only, in the version-1 shape ffmpeg's `mov`
muxer produces, so a g2g-written MP4 round-trips titles and starts but reports
the chapters open-ended.

## Ogg

The same parser plus element split on `Caps::ByteStream{Ogg}`.
`g2g-plugins::ogg::OggDemuxer` parses RFC 3533 pages: sync to "OggS", frame
packets via the segment-table lacing with cross-page reassembly, sniff the codec
from the first packet's `OpusHead`, skip the setup headers. `OggDemux` emits the
Opus audio packets as `Caps::Audio{Opus}` with the channel count refined from
`OpusHead`. The container is auto-detectable through `typefind` "OggS" and
`filesrc bytestream-format=auto`.

The VP8 mapping's `OVP80` stream-info header gives the coded size and frame
rate, the header packets stay in the container, and each frame is timed by the
count of shown frames before it, so an invisible frame takes no time. A Theora
stream's identification header gives the picture size and frame rate, and its
three header packets go in band ahead of the frames, as Vorbis's do. Each
packet is one frame period, and an empty packet, which repeats the previous
frame, takes its period without being forwarded.

### Grouped multi-stream Ogg

Grouped multi-stream Ogg is handled per serial. A file opens with one
beginning-of-stream page per logical bitstream before any other page, RFC 3533
§4, and the parser keeps an `OggLogicalStream` for each with its own codec
mapping, headers, packets and granule anchors. A serial joins only from a page
in that opening block, or as the first stream seen when the file was joined
mid-stream, which a byte-seek does. A beginning-of-stream page after the
group's end opens a chained physical stream, which continues on the same pad
behind a `Segment` and its own in-band codec headers. `flacparse` ends the
previous link's last frame at that `Segment` and forwards the new `fLaC`
header. The concurrent stream count is capped, since the serials come from the
file.

`OggDemux` forwards the first bitstream whose codec matches its `stream`
selection and drains the rest. `OggDemuxN` is the multi-output form, one port
per `OggPort` naming the bitstream it carries. Routing is positional rather than
codec-keyed because two streams of one codec in a file is ordinary. The
per-stream caps, in-band codec config and packet timing are one shared
`StreamEmitter` both elements drive, and `OggDemuxN` announces the file's
`StreamCollection` and posts each stream's VorbisComment as a `StreamTag` under
the same ids.

An embedded picture such as cover art travels as `Tag::Image`, the encoded
image with its still-image codec, the way GStreamer carries it in its `image`
tag rather than as a stream. MP4 `covr` items, FLAC `PICTURE` blocks, the
VorbisComment `METADATA_BLOCK_PICTURE` field and ID3 `APIC` frames read into it,
and a picture whose media type names no still-image codec is dropped. `flacparse`
posts a native FLAC header's comments and pictures. The tag writers skip images
through `TagList::text_tags`.

### oggmux

`oggmux` is `g2g-plugins::ogg::OggPageWriter` plus the `OggMux` element, the
inverse path on the same three mappings. The writer laces packets into pages,
255-byte segments with continuation pages past the 255-segment limit and the RFC
3533 CRC-32 with polynomial `0x04c11db7` and no reflection, holding the last
packet back so the end-of-stream flag always has a page to ride.

Codec config arrives in-band, the same convention the demuxers emit: an
`OpusHead`, the three Vorbis headers or the native `fLaC` block is held until
the first audio packet, then written as the beginning-of-stream page plus one
following page at granule 0, so audio starts on a fresh page as the mappings
require. Vorbis is remux-only, there being no encoder, and the Ogg-FLAC first
packet is rebuilt around the source STREAMINFO with the mandatory VorbisComment
appended.

Granule positions come from each mapping's own sample count, Opus TOC durations,
FLAC block sizes, and the lapped `(prev + cur) / 4` from the `VorbisTiming` mode
tables, the inverse of the demux-side durations. They are held to the total
`duration_ns` the input declared when every packet is timed. That last bound
carries a source's end-of-stream trim through a remux, so ffmpeg decodes a
remuxed Opus, Vorbis or FLAC stream to the source's samples bit for bit.

`oggmuxn` is the fan-in form: one `OggStreamMux` per input pad, each its own
logical bitstream with its own serial, packets interleaved by PTS through the
same `InputAggregator` merge the other multi-track muxers use. Grouping forces
the page order, which is why the per-stream header writing is split in two.
Every stream's beginning-of-stream page is written first, in pad order, then
each stream's remaining header pages, then the data pages. That block goes out
when the merge first releases a packet, the first moment every pad's in-band
codec config is known to have arrived. Like `mpegtsmux`, the one name `oggmux`
covers both shapes, the parser picking by link degree.

## Opus pre-skip and end trim

Opus decode applies two container trims so the PCM sample count matches ffmpeg
and gstreamer, RFC 7845. Each container spells them differently and all three
convert to the in-band convention, where the demuxer forwards the `OpusHead`
ahead of the audio as a parameter set.

`OpusDec` presents its output on the timeline without the pre-skip, as ffmpeg and
gst report Opus. It clips to the segment on the container's timeline, which counts
the pre-skip, then stamps each output frame at the container time of its first kept
sample minus the `OpusHead` pre-skip, with the kept sample count as its duration.

### Ogg

Pre-skip is codec config the decoder owns. `OggDemux` forwards the `OpusHead`
in-band and `OpusDec` reads its pre-skip at offset 10 and drops that many
leading output samples. End-of-stream padding is container knowledge only the
demuxer has: `OggDemux` tracks the running decoded sample count against the
final page's granule position and marks the closing packets short via
`duration_ns`, which `OpusDec` honors as a per-frame keep count. Fully padded
packets are dropped. Both trims are attacker-controlled inputs, folded with
saturating math, so an oversized pre-skip trims a frame to nothing and an
underflowing granule drops it. A stream with no `OpusHead` and no per-frame
duration, which is the RTP path, decodes untrimmed, matching gstreamer's
SDP-less default.

### Vorbis in Ogg

Vorbis carries the same two facts without a header field. Its playable length is
the final page's granule position, and its head trim is the first audio packet,
which primes the overlap window and decodes to nothing. `OggDemux` clips that
priming block off the front and clamps the tail to the end granule, so a decode
yields exactly the granule's worth of samples. The size of the clip comes from
the first audio page's granule, its shortfall against the natural packet
durations, which also covers a stream joined mid-file. When that page is also
the last its granule is the end of the stream and says nothing about the head,
so the clip is the priming packet's own `blocksize / 2`.

### MP4

The `dOps` OpusSpecificBox holds an `OpusHead`'s fields big-endian, so the
demuxers rebuild one from it and forward it ahead of the audio through
`opusparse::opus_head_from_dops`, validated field by field: an unknown version,
a zero channel count or a truncated channel-mapping table leaves the track
configless rather than failing the file. The end trim is the final sample's
short `stts` or `trun` duration, which arrives as `duration_ns` like the Ogg
granule trim does.

`Mp4MuxN` is the inverse. An in-band `OpusHead` is consumed as config and
becomes the `dOps` through `dops_from_opus_head`, so a remux keeps the source's
pre-skip, output gain and channel mapping byte for byte, while a freshly encoded
stream falls back to libopus' 312-sample lookahead. `OpusEnc` emits raw packets
with no header, so its RTP consumers are unaffected. The Opus `trak` also
carries the `edts`/`elst` the Opus-in-ISOBMFF binding requires, with
`media_time` = pre-skip.

AAC leaves the MP4 demuxers ADTS-framed from the track's AudioSpecificConfig.
ADTS has no profile for ER AAC, ELD or USAC and no field for 960-sample
frames, and it signals SBR only implicitly, so a decoder doubles the core rate.
Those access units, and those of a config whose SBR output rate is anything but
double the core rate (AOT 5 or 29, or a syncExtensionType 0x2B7 after the
GASpecificConfig), go out raw with the AudioSpecificConfig ahead of them as a
frame of its own. `aacparse`
forwards that stream unchanged. `ffmpegaudiodec` reopens the AAC decoder with the
config as extradata and presets the channel count and rate it declares, which
libavcodec needs to output an ER AAC frame. USAC decodes only on libavcodec 62
(ffmpeg 8) or later. `flvdemux`, `avidemux` and `matroskademux` send the
same stream from the FLV sequence header, the AVI `strf` config and the
Matroska `CodecPrivate`.

QuickTime PCM entries resolve to a layout (bits, float, signed, endian, packed),
from the fourcc (`in24`, `in32`, `fl32` and `fl64`, with `enda` inside `wave`
selecting little-endian) or from version 2 `lpcm` flags. A layout with no
`AudioFormat` skips its track and is named by `unsupported_sample_entries`.
Vorbis in `msVo` forwards its three Xiph-laced `glbl` headers in-band, as Ogg and
Matroska do.

Uncompressed tracks whose stored layout differs from the graph's tight top-down
frame carry a `RawVideoStorage`, and the demuxer fixes the layout in the copy it
already makes. For AVI, `BI_RGB` rows are 4-byte aligned when the chunk holds
them, else tight, and run bottom-up for a positive `biHeight`; 24-bit is `BGR`,
32-bit `BGRx`. `YV12` is delivered as I420 with its chroma planes swapped. An `I420` / `IYUV`
chunk whose size equals each plane's rows padded to four bytes (GStreamer's
`avimux`) is packed tight, any other size reads tight, and
`Y800` / `GREY` as `GRAY8`. QuickTime `raw ` rows (depth 24 as `RGB`, 32 as
`ARGB`) take their stride from the sample size, since writers pad them
differently.

QuickTime PCM sample entries (`twos`, `sowt`, `raw `, `alaw`, `ulaw`, version
0 or 1 sound descriptions) map through the AIFC compression table, with `twos`
read as AIFC's big-endian `NONE` and 8-bit `raw ` as unsigned. Their `stsz`
counts single frames, often at size 1, so the demuxers size a frame from the
format and channel count and emit one buffer per chunk, converted to the
graph's little-endian, unsigned 8-bit layout. A file cut short ends the track
at the last whole frame before the cut, so a `stsz` count past the file size is
clamped to it rather than refused. The rate comes from the sound
description. MP3 rides `.mp3` or an `mp4a` whose esds names MPEG-1 or MPEG-2
audio, and its frames pass unchanged. `qtdemux stream=` names the audio track
by format (`aac`, `mp3`, `pcm-s16le`, `alaw`, ...).

`Mp4MuxN`, `MkvMuxN` and both FLV muxers take an AAC stream's
AudioSpecificConfig from its first frame: synthesised from an ADTS header, or
the leading config frame of a raw stream, which becomes the `esds`,
`CodecPrivate` or FLV sequence header as is (at most 256 bytes) and is never
written as a sample. The raw units that follow are stored unchanged. ADTS frames
are stored without their header. A fan-in muxer whose AAC pad delivers frames
without either config fails the stream.

MPEG-TS carries AAC as ADTS (0x0F) unless the stream opens on a raw
AudioSpecificConfig (ER AAC, ELD, USAC, 960-sample frames). That pad goes out as
LOAS/LATM (0x11) with audioMuxVersion 1. The StreamMuxConfig repeats in every
frame, and the config sits verbatim behind its bit length, so the muxer never
parses ELDSpecificConfig or UsacConfig. The carriage is fixed by each pad's first
frame, and the PMT entry switches to 0x11 before the tables first go out. On the
read side, `TsStream::Aac` takes 0x0F and 0x11, LOAS under either, and forwards LOAS frames as the
PES carries them, and `aacparse` reads both LATM versions. `ffmpegaudiodec` opens
libavcodec's LATM decoder when the first buffer is a LOAS frame, seeds it with
the config's rate and channel count, and decodes each LOAS frame whole. ffmpeg
8.1 decodes neither USAC in LATM nor, from its CLI, ER AAC in LATM, so the USAC
leg is checked only by readback and GStreamer's `aacparse`.

### Matroska

The pre-skip is the `CodecPrivate` `OpusHead` itself, which the demuxers forward
ahead of the audio after validating it as a real header, plus a `CodecDelay` on
the TrackEntry, the ns form of the same count, which `MkvMuxN` derives from the
header it is about to write and pairs with the mapping's fixed 80 ms
`SeekPreRoll`. Block timestamps are not shifted: the first Opus block sits at 0
and `CodecDelay` tells the decoder what to discard, so a Matroska file starts at
zero where the MP4 edit list makes `start_time` negative.
A track whose `CodecDelay` differs from its pre-skip, or has none, has its blocks
moved by the difference so the decoder's pre-skip shift lands on the block
timestamps. Laced frames take the track's `DefaultDuration`. Without one, Opus
frames are spaced by their TOC duration and Vorbis frames by the lapped
(previous + current) / 4 block size read from the `CodecPrivate` setup header,
the previous size carried across blocks. `VorbisDec` stamps a packet that has a
pts but no duration with that pts, as ffmpeg does.

The end trim, with no granule to carry it, is the final block's
`DiscardPadding`, which is nanoseconds and so survives the millisecond
`TimestampScale` grid that `BlockDuration` alone would round away, where a
6.5 ms tail becomes 7. The muxer writes both, as ffmpeg does, and the demuxer
lets the ns element win. Because the packet's own length is needed to turn a
tail discard into a kept duration, the conversion reads the Opus TOC byte and
applies to Opus only.

## MP4 and fMP4

An `mp4v` entry's objectTypeIndication picks the codec in one parse shared by
every MP4 reader: 0x20 is MPEG-4 Part 2, 0x60-0x65 and 0x6A are MPEG-2 / MPEG-1.

The intra-only editing codecs come from the QuickTime sample entry: `apco`,
`apcs`, `apcn`, `apch`, `ap4h` and `ap4x` are `VideoCodec::ProRes` with the
matching `ProResVariant`, `AVdn` and `AVdh` are `VideoCodec::DnxHd`. The variant
travels in the caps, GStreamer's `variant` field, because libavcodec's ProRes
decoder takes its sample depth from the profile fourcc: 10 bits for the 4:2:2
profiles, 12 for 4444. A Matroska `V_PRORES` track names its profile with the
fourcc in its `CodecPrivate`, standard when it has none, and its blocks lack
the frame size and `icpf` tag a QuickTime ProRes frame opens with, so
`matroskademux` puts them back, as ffmpeg's demuxer does.

`mp4demux` reports a track's frame rate as `timescale / delta` when every `stts`
entry codes the same sample delta, and `Rate::Any` when the deltas vary, the
table is empty (a fragmented init) or the rate falls outside the 1 to 240 fps
placeholder it negotiated. `mp4src` applies the same rule and announces it in a
`CapsChanged` before its first sample. A fragmented file has an empty `stts`, so
`mp4src` and `fmp4demux` read the first fragment's `tfhd`
`default_sample_duration` instead. `fmp4demux` therefore announces its caps at
the first `moof` rather than at the `moov`.

`Mp4Demux` and `Mp4DemuxN` turn a video track's edit list into the `Segment`
they send ahead of its samples. It starts at the `media_time` of the first
media edit, past any leading empty edits, whose delay is not applied. When
every edit after them is a media edit with a duration, each starting at or
after the end of the one before, the segment stops where the last one ends and
the demuxers cut the media between edits out of the timeline, as ffmpeg's mov
demuxer does: a sample after a gap moves back by the length of the gaps before
it, a decode time inside a gap moves to its start, and a sample whose pts is
inside a gap moves past the segment's stop. A list of any other shape keeps
only the first media edit's start. Every sample still goes downstream, because
the cut pictures can be references, and every video decoder drops each decoded
picture whose pts `Segment::presents` rejects:
before the start, or at or past the stop, the frames ffmpeg drops for the same
edit. The intra-only decoders (`mjpegdec`, `jpegxsdec`) skip the cut units
before decoding them. An audio track's segment takes the same start and stop,
trimming the encoder's priming and end padding, and Opus gets none, its
pre-skip being trimmed by `OpusDec`. The stop is exact where ffmpeg keeps every
packet that starts before the edit's end. Raw PCM reaches no decoder, so the
demuxers cut its chunks to the segment themselves.
Every audio decoder cuts its decoded PCM to the segment by sample: a frame
that straddles the start loses the samples before it and takes the start as
its pts, a frame that straddles the stop ends there, and a frame wholly
outside is dropped. `Segment::kept_samples` does that arithmetic for all of
them.
An audio track with no edit list takes its segment from the file's iTunSMPB
tag instead, read in media timescale ticks (the core rate for implicit-SBR
HE-AAC, where ffmpeg and gst read it at twice that and start 53 ms late on the
Nero tone vector). `MpegAudioParse` derives the same kind of segment from the
LAME tag in an MP3's Xing or Info frame, with ffmpeg's rules: the start is the
first pts plus the encoder delay and libmp3lame's 529-sample decoder delay, and
the stop trims the padding only when the Xing frame count is present.

Which duration a reader reports depends on the layout, so `Mp4MuxN` writes both,
selected by the `fragmented` property, default `true`.

Fragmented is the streamable layout: `ftyp` and `moov` up front, a `moof` plus
`mdat` per fragment, empty sample tables and zero header durations. ffmpeg
derives such a file's duration by summing the `trun` sample durations and
applies an edit list only as a timestamp shift, so an Opus track reports the
media span with a negative `start_time` and `segment_duration` is written `0`,
the total being unknown when the `moov` goes out.

Progressive is the two-pass layout, the shape `matroskamux`'s `seekable` mode
has. Every sample is buffered, then `ftyp` plus a single `mdat` plus a `moov`
are emitted together at EOS, with real `stts` / `ctts` / `stss` / `stsc` /
`stsz` / `stco` tables, one sample per chunk ordered by decode timestamp, and
real `mvhd` / `tkhd` / `mdhd` durations. That is enough for ffmpeg to apply the
edit, so the reported duration is the trimmed presentation length exactly. The
cost is holding the movie in memory, so a live or long capture wants the
fragmented default. GStreamer spells this choice `fragment-duration = 0`, which
g2g already spends on "one fragment per access unit", so the layout gets its own
boolean rather than a silently redefined property.

The decode timestamp is the frame's own. `Mp4DemuxN` reads the source's `ctts`
and carries `dts_ns` beside `pts_ns`, so a reordered B-frame stream's
composition offsets survive a remux. A frame with no decode timestamp of its
own, or one past its PTS, which `ctts` version 0 cannot express, decodes when it
presents.

Compressed audio negotiates with the `0/0` "unknown until parsed" caps, so
`Mp4MuxN` adopts the concrete channel count and rate from the runtime
`CapsChanged` a demuxer emits, while the `moov` is still unwritten. Without it a
remuxed audio track would declare a zero `mdhd` timescale.

## Channel count negotiation and audioconvert

An audio decoder fixates its output caps even when a demuxer only knows the
channel count once it parses the stream. It advertises `PcmS16Le` at a concrete
rate with the `ANY_CHANNELS` placeholder, fixated to stereo for the negotiated
edge, and the real count arrives via a `CapsChanged`. `OpusDec` rebuilds libopus
for it, since the decoder is per-channel-count. A decode-to-PCM line therefore
negotiates before the count is known.

`vorbisdec` labels its PCM with the Vorbis I layout for the channel count
(`vorbis_layout`, shared with `opusdec` mapping family 1). symphonia already
permutes Vorbis order into that layout's interleave order, so no reorder happens
in g2g.

`AudioConvert` is caps-driven like `AudioResample`: a bare `audioconvert` takes
its output format and channel count from a downstream capsfilter, such as a mono
`channels=1` pin, and otherwise passes the input through. Its channel mixing is
position-aware for multichannel, meaning either side above 2. Speaker positions
come from `g2g_core::ChannelLayout::default_for`, the per-count layout
convention taken from the ffmpeg default-layout table, which is the order the
decode path interleaves. The mix matrix applies the ITU BS.775-style
coefficients: center and surrounds fold at 1/sqrt(2), back center at 0.5 into
each front, LFE is dropped, and the result is normalized against clipping. It is
verified coefficient for coefficient against ffmpeg's default rematrix. Upmix
places each input at its own speaker and leaves the rest silent. Counts past the
layout table, above 8, fall back to the layout-agnostic round-robin fold so no
channel is silently dropped.

## FLV

`g2g-plugins::flv::FlvDemuxer` parses the flat FLV tag stream on
`Caps::ByteStream{Flv}`: the "FLV" header, then `PreviousTagSize` and tag pairs,
each tag's 11-byte header framing its body. `FlvDemux` forwards the H.264 (AVC)
video and AAC audio media access units with their millisecond timestamps, PTS
from the video tag's signed composition-time offset and DTS from the tag header,
selected per `FlvStream`, `h264` or `aac`, defaulting to h264, like `TsDemux`.

The sequence-header tags are the codec-config side channel. The parser retains
the `AVCDecoderConfigurationRecord` and `AudioSpecificConfig`, and the element
uses them the way the MP4 demuxers do: re-framing the AVCC access units to
Annex-B honouring the `avcC` NAL length-prefix width, with the SPS and PPS
prepended in-band to the first access unit, ADTS-framing the raw AAC so the
audio is self-describing, and announcing the concrete channel layout and sample
rate via `CapsChanged`. Both extracted elementary streams therefore decode
standalone, validated against the ffmpeg oracle in both directions in the CI
conformance job. The `onMetaData` script tag posts as bus tags. The container is
auto-detectable through `typefind` "FLV" and `filesrc bytestream-format=auto`.

The FLV muxer is `flvmux`, `g2g-plugins::flv::FlvMuxer` plus the `FlvMux`
element. Like `FlvMuxN` it captures the decoder config in-band from the first
access unit, parameter sets from the IDR and the first ADTS header, and writes
it as the track's sequence-header tag, re-framing video Annex-B to AVCC with
keyframes flagged from the IDR NAL and audio de-ADTS'd, so a single-track
`flvmux` output is a playable FLV, which is what `RtmpSink` publishes.

With MP4 through `Mp4Src` and `Mp4Sink`, MPEG-TS, Matroska/WebM, Ogg and FLV,
the demux and mux coverage spans the major containers.

## MPEG program stream

`mpegpsdemux` on `Caps::ByteStream{MpegPs}` is the `.mpg` and `.vob` read path,
covering VCD-era MPEG-1 program streams and DVD MPEG-2 ones through one element.
`g2g-plugins::psdemux::PsDemuxer` syncs to pack headers, `00 00 01 BA`, in both
the MPEG-2 `01`-marker layout with its stuffing and the flat 8-byte MPEG-1 one,
and reads the PES packets between them. A stream that ends without a single pack
header was not a program stream, so `PsDemuxer::flush` fails it with
`CapsMismatch`. `PsDemux` and `PsDemuxN` wrap it with a
`PsStream` selection of `Mpeg2` video, `Mp2` audio, `Ac3` and `SubPicture`. Two
things make it unlike `TsDemux`.

### Stream discovery

There is no PAT or PMT. A stream is identified by its PES `stream_id`,
0xE0..=0xEF video and 0xC0..=0xDF audio, and, for the `private_stream_1` 0xBD
that DVD carries AC-3 and subpictures on, by the substream id byte opening its
payload, 0x80..=0x87 AC-3 behind a 4-byte DVD substream header and 0x20..=0x3F
subpicture. Streams are therefore discovered by observing packets, so the
`playbin` and `decodebin` probe hooks report what the probe window has actually
shown rather than reading a table, and geometry comes from the video's own
sequence header, `00 00 01 B3`, which the demuxer parses to fix the video caps
via `CapsChanged`.

A selection forwards only the first stream of its codec the parser has seen, so
a second audio language never interleaves with the first. That is the stream
ffmpeg's `-map 0:a:0` picks. A `PsDemuxN` stream selection names any one stream
by its collection id. `TsDemux` follows the same rule in PMT order.

### Access unit reframing

A PES payload is not an access unit. A program stream cuts its packets on sector
boundaries with no regard for picture boundaries, so one packet can hold the
tail of a picture and the head of the next, and feeding those to a decoder
verbatim desynchronizes it. The demuxer reframes the video on its own start
codes, a unit running from one picture header, with any sequence or GOP header
opening it, to the next. A PES timestamp names the first access unit commencing
in its packet and stamps exactly that unit. That is the job an elementary-stream
parser does for the other codecs, kept here because it is program-stream
specific and MPEG-TS needs none of it. Audio and AC-3 are self-syncing and are
grouped per timestamped packet instead, matching what `TsDemux` emits.

### Timestamp synthesis

A DVD stamps about one PES packet per GOP, so most pictures arrive with no
timestamp of their own. Carrying the last stamp forward would give a dozen
frames one shared PTS, which a pacing sink plays as a burst then a freeze, so
the demuxer synthesizes each unstamped picture's PTS as
`gop_base + temporal_reference * frame_period`. The picture header's
`temporal_reference` is its display index within the GOP, so the arithmetic
stays exact across B-frame reordering, a real PES PTS re-anchors the base so
drift never outlives a GOP, and an unstamped GOP header advances it by the span
of the GOP just closed. Unstamped DTS advances one frame period per picture in
coded order.

### Subpictures

Subpicture units span several PES packets and declare their own total size in
their first two bytes, so they are reassembled by size, bounded by the 16-bit
maximum that size field can state, and stamped with the opening packet's PTS and
the unit's own hide time as duration. A program stream carries no palette, so
the pad opens on a synthesized `.idx` holding only the video's `size:` line and
`VobSubDec`'s default palette renders the cues, see
[text.md](text.md).

Out of scope: LPCM and DTS substreams, the program stream map, a PS muxer, and
seeking.

## IV8

`iv8demux` on `Caps::ByteStream{Iv8}` reads IndigoVision 8000 files, which
often carry a `.mpg` name. The `filesrc` extension check corrects that to `Iv8`
when the header starts with the six bytes ffmpeg's probe matches. Each packet
is a 16-byte big-endian header (type, size, flags, packet number, 90 kHz PTS,
four unknown bytes) and a payload of `size - 12` bytes. Type 257 payloads join
into one MPEG-4 Part 2 frame until a packet sets flag `0x80`, and the frame
takes the PTS of its first packet. Other types are skipped. A size with no
payload fails with `CapsMismatch`, and so does a frame past 16 MiB. At end of
stream a cut video payload is emitted with the bytes that arrived, and a cut
header drops the unfinished frame, both as ffmpeg does.

## MPEG-1 and MPEG-2 video

`VideoCodec::Mpeg2` covers MPEG-1 and MPEG-2 video as one codec, libavcodec's
`MPEG2VIDEO` decoder playing both. MPEG-TS stream types 0x01 and 0x02 map to it
through `TsStream::Mpeg2`, and `au_is_keyframe` reads its sync points, an
I-picture or a sequence or GOP header.

## Start-code elementary stream parsers

`mpegvideoparse`, `mpeg4videoparse` and `vc1parse` are the standalone parsers
for the start-code elementary streams that are not NAL streams, for a launch
line that feeds a decoder from a raw `.m2v`, `.m4v` or `.vc1` file rather than
through a demuxer that already frames the video.

They share `StartCodeParse<C>` in `startcodeparse.rs`, the `NalParse`
counterpart for a `00 00 01 xx` byte stream. It accumulates input, splits at
access-unit boundaries of one coded picture plus the headers that lead it,
refines caps from the sequence header, stamps the keyframe flag, and re-inserts
cached configuration headers on a `config-interval`. A `StartCodeCodec` marker
supplies the per-codec start-code classification, geometry parse and keyframe
rule. Unlike Annex-B the prefix is exactly three bytes, so the scanner never
matches `00 00 00 01`: a header ending in a zero byte would otherwise lose it.

`mpegvideoparse` reads the same `mpeg2video::parse_sequence_header` the program-
and transport-stream demuxers do, applying the sequence extension's size and
frame-rate extensions and deriving the sample aspect ratio. `mpeg4videoparse`
reads the VOL header and carries gst's `config-interval`, the VOS / VO / VOL
prefix, re-sent before a keyframe that lacks it.

`VideoCodec::Vc1` is SMPTE 421M, gst's `video/x-wmv` with `format=WVC1`.
`vc1parse` covers advanced profile, the start-code byte stream: the sequence
header gives `MAX_CODED_WIDTH` and `MAX_CODED_HEIGHT` and, in its display
extension, the frame rate and sample aspect, read after Annex-E de-escaping.
Simple and main profile carry no start codes, so there is nothing in the byte
stream to frame or measure and the parser leaves such a stream alone. There is
no VC-1 decoder.

`Caps::CompressedVideo` has no pixel-aspect field, so the sample aspect these
three headers signal is surfaced as the read-only `pixel-aspect-ratio` property
rather than in caps.

`NalParse` and `StartCodeParse` in re-framing mode stamp the units their source
left unstamped (0 from a byte source, or none) with the synthesis `TsDemux` uses:
decode order plus the declared frame period, the picture order count for a
reordering H.264 or H.265 stream, `temporal_reference` for MPEG-2. The first such
unit opens the timeline at 0 and an IDR restarts it one period after the latest
picture. A stream with no declared rate leaves later units unset (`PTS_NONE`),
as gst's `h264parse` and `h265parse` do, and decoders pass that through.
A PES or container stamp names the first access unit starting in its packet. The
shared access-unit framer and the program-stream audio aligner give it to that
unit only, and later units in the same packet go out unstamped for the parser's
synthesis to time.

## AV1 OBU streams

A raw `.obu` file is AV1's low-overhead bitstream format: size-delimited OBUs,
each temporal unit opening on a temporal delimiter. `typefind` types it as
`CompressedVideo{Av1}` when it opens on an empty temporal delimiter followed by
a sequence-header OBU that parses whole, since the two delimiter bytes alone
would match unrelated data.

The launch `av1parse` (`Av1Parse::reframing`), which the auto-plug chain splices
ahead of every AV1 decoder, splits that byte stream into one temporal unit per
buffer, holding the last unit until the next delimiter or EOS. Input whose
buffers open on a delimiter twice in a row (IVF, MPEG-TS) counts as already
unit-aligned and is no longer held back. Input with no delimiters at all (the
MP4 and Matroska mappings strip them) passes through untouched. An OBU without
a size field, or one coding a size past 16 MiB, fails the stream, as does a held
unit growing past 16 MiB.

## deinterlace

Disc content is usually interlaced, and presenting its woven frames as-is combs
on motion. `deinterlace` is the CPU filter that undoes it: a single-rate yadif
port, an edge-directed spatial interpolation clamped to a temporal window built
from the previous and next frames, plus the cheaper `linear` and `blend`
methods, over I420 / NV12 / RGBA / BGRA at unchanged format and geometry, one
frame out per frame in. It is bit-exact against ffmpeg's `yadif=0` on the same
raw frames, including the 3-column border where ffmpeg drops the directional
search. Field order is assumed top-field-first, ffmpeg's default for a stream
that declares nothing.

### Interlace caps

Interlacing is signalled in the caps. `Caps::RawVideo` carries an `Interlace`
field of `Any`, `Progressive` or `Interleaved`, where the `Any` wildcard
intersects with anything, survives `fixate`, and reads as "progressive unless
declared", so the field never blocks a solve and nearly every caps site states
`Any`. `FfmpegVideoDec` reads libavcodec's per-picture interlaced flag and
latches `Interleaved` output caps on the first interlaced picture, sticky for
the stream so telecine content cannot flap `CapsChanged`, covering interlaced
MPEG-2 over any container and interlaced H.264 alike.

The element's `mode` property acts on that declaration:

| `mode` | behaviour |
| :--- | :--- |
| `interlaced` | the default, always weaves, the contract for hand-written lines whose upstreams declare nothing |
| `auto` | weaves only a caps-declared `Interleaved` stream in a format the kernels handle, and otherwise forwards packets untouched |
| `disabled` | pure passthrough |

In `auto` mode negotiation is kept transparent, any raw video passes, so
inserting the element never narrows a branch. Every `playbin` video branch, mkv
/ mp4 / TS / PS / HLS, plain fan-out and the subtitle, closed-caption and DVD
subpicture overlay variants, inserts `deinterlace mode=auto` after the decoder,
matching GStreamer's playbin `deinterlace` flag. A progressive stream pays only
a forwarding hop, and the interlacing verdict comes from the decoder's own
per-picture report rather than a container probe of `progressive_sequence` in
the MPEG-2 sequence extension.

## Adaptive streaming

Adaptive streaming sits one layer above the demuxers: an HTTP byte source feeds
a playlist or manifest-driven source that fetches media segments and hands them
to the matching byte-stream demuxer.

`g2g-plugins::httpsrc::HttpSrc` (the `http-src` feature, `reqwest`) GETs a URL
and streams the body as `Caps::ByteStream` chunks, the fetch layer the others
share. It owns the network-buffering story through `prebuffer-bytes` and
`with_bus`, the queue2 analog since g2g has no queue element. When set, `run`
fills a bounded byte window before pushing downstream, posting
`BusMessage::Buffering` percent on quartile transitions, streams through while
topping the window up without waiting, and re-enters buffering on a mid-stream
underrun, meaning the window is empty and the network is not ready. An
application can pause until `100` and show a buffering indicator on a stall. The
window never grows past the target, and `0`, the default, streams straight
through.

The segment loops `hlssrc` and `dashsrc` carry the duration-keyed sibling,
`prebuffer-ms` plus `with_bus`: a `segprebuf::SegmentPrebuffer` window the loop
fetches into while below its duration target, summed `#EXTINF` or MPD segment
durations, and emits from otherwise. It posts the same quartile `Buffering`
levels during the startup and post-seek fill and stays silent in steady state.
Init segments ride the window with duration 0 so an ABR re-init stays ordered
behind queued media, and a flushing seek clears the window and re-arms the fill.

A manifest or segment URL is attacker-controlled, so the shared
`fetch::get_bytes` and `get_text` never buffer an unbounded body. Each
accumulates the response chunk by chunk against a cap, `MAX_MANIFEST_BYTES` of
16 MiB for playlists, MPDs and keys, and `MAX_SEGMENT_BYTES` of 256 MiB for one
media segment, failing loud when an honest `Content-Length` or the streamed
running total exceeds it, so one oversized reply cannot exhaust memory.

### HLS

`hlssrc::HlsSrc` (`hls`) parses an RFC 8216 `.m3u8` with the pure `no_std` `hls`
parser, covering master variants for bandwidth-capped ABR and media segments,
selects a variant, and streams its segments: MPEG-TS into `tsdemux`, or fMP4 and
CMAF, signalled by `#EXT-X-MAP` and probed at negotiation, as
`ByteStream{IsoBmff}` into `fmp4demux`.

A no-ENDLIST live playlist starts near the live edge (`live_edge_start`, about
three target durations from the end per RFC 8216 §6.3.3), so playback follows
what is being published rather than replaying the stale front of the sliding
window, clamped to the window start for a short window. `with_full_replay()`
opts back into starting from the window front for a DVR replay. The source then
reloads on an interval, playing each new segment once by media sequence. An
`#EXT-X-GAP` segment is stepped over rather than fetched (RFC 8216bis §4.4.4.7),
because a live packager pads a freshly started playlist with placeholders whose
URI it never wrote, and loading one is a 404 that would fail the run before the
first frame.

`with_abr()` makes it throughput-adaptive. A shared `abr::BandwidthEstimator`
keeps an EWMA of measured download throughput, bytes over elapsed
`monotonic_ns`, and yields an effective bandwidth cap, the estimate scaled by a
safety factor and bounded by `max-bandwidth`. The run loop feeds that cap to the
existing `MasterPlaylist` selection, re-picks the best-fitting variant after each
segment, and on a change swaps the active media playlist and re-emits the init,
keeping the time-aligned segment index. It is off by default, which is a fixed
up-front variant.

Single-file CMAF is supported through `#EXT-X-BYTERANGE` and `#EXT-X-MAP`'s
`BYTERANGE`: a segment carrying one fetches only its sub-range with an HTTP
`Range` request, the offset continuing from the previous sub-range of the same
resource when the tag omits an explicit `@offset`. A server that ignores the
`Range` and replies `200` is handled by slicing the requested window from the
full body.

### Low-latency HLS

The parser reads `#EXT-X-PART` (partial segments, with `INDEPENDENT`, `GAP` and
`BYTERANGE`), `#EXT-X-PART-INF` (`PART-TARGET`) and `#EXT-X-SERVER-CONTROL`
(`CAN-BLOCK-RELOAD`, `PART-HOLD-BACK`), per RFC 8216bis. Parts precede the
`#EXTINF` of the segment they belong to, so a run of them left at the end of the
playlist becomes a `Segment` with no URI (`incomplete()`), the segment still
being produced, whose only fetchable pieces are its parts. A Part Index is
positional, since the `_HLS_part` directive names it, so a malformed
`#EXT-X-PART` fails the parse instead of silently shifting the ones after it.

`HlsSrc` follows the low-latency path whenever the playlist offers both parts and
blocking reload, and `low-latency=false` forces whole segments. The live reload
becomes a GET carrying `_HLS_msn` and `_HLS_part` for the media the run wants
next, which the server holds until that part is published, and each part is
fetched and emitted as its own `DataFrame`. A complete segment nothing was taken
from is still fetched whole, one request instead of several, so parts are used
for the segment being produced and to finish one joined part-way.

Playback starts `PART-HOLD-BACK` behind the live edge, three `PART-TARGET`s
absent the attribute, rather than three target durations, walking back to the
nearest `INDEPENDENT` part so the decoder joins on a frame it can decode, and the
startup prebuffer defaults to that same hold-back instead of
`DEFAULT_PREBUFFER_MS`. A blocking reload is bounded by a deadline of one target
duration plus a margin, and three held requests that come back with nothing new
drop the run to timed reloads for the rest of the run. Against mediamtx with 1 s
segments and 200 ms parts this puts the oldest sample in an emitted frame about
200 ms behind publication, where the whole-segment path is about 1.8 s.

### HLS encryption

`#EXT-X-KEY:METHOD=AES-128` segments are decrypted in place with AES-128-CBC via
`aes` and `cbc`, the key fetched from the key URI and cached, and the IV explicit
or derived from the media-sequence number.

`METHOD=SAMPLE-AES` encrypts only the media samples inside the container, so it
is handled after demux by the `sampleaesdecrypt::SampleAesDecrypt` transform,
`tsdemux ! sampleaesdecrypt ! h264parse`. Per the Apple TS sample-encryption
format it AES-128-CBC decrypts H.264 slice NALs (32-byte clear leader, a
16-encrypted and 144-clear pattern, emulation-prevention aware, IV reset per NAL)
and AAC ADTS frames (ADTS header plus 16 clear bytes, then whole-block CBC). The
key and IV reach it either configured directly or, in the HLS chain, auto-wired:
`HlsSrc` fetches the `#EXT-X-KEY` material and publishes it into a shared key
handle the decryptor reads, forwarding the sample-encrypted segments undecrypted
because the demuxer needs the clear framing.

For fMP4 and CMAF, SAMPLE-AES maps to the `cbcs` Common Encryption scheme
(ISO/IEC 23001-7), handled inside `fmp4demux`. The init segment's `encv`, `sinf`
and `tenc` give the crypt-to-skip pattern, 1:9 for video, and the constant IV,
each fragment's `senc` gives the per-sample clear and protected subsample ranges,
and the protected ranges are AES-128-CBC decrypted with the IV reset per
subsample and chaining over the encrypted blocks only, using the same shared key
handle `HlsSrc` fills.

The sibling schemes decrypt through the same machinery: `cenc` (whole-range
AES-CTR), `cbc1`, and `cens` (pattern AES-CTR, one IV per sample with the counter
advancing only over encrypted blocks, the pattern restarting per protected
range). Sample groups can re-key mid-fragment: a `traf` `sbgp` and `sgpd`
(`seig`) overrides the `tenc` defaults per sample run, and the movie-level `seig`
table resolves alongside it, with indices below 0x10001 resolving against the
track's `stbl` table and fragment-local ones above it, per 14496-12, strictly
scoped. A subsample map that overruns its sample is an error, never a partial
decrypt. A clear track stays a normal demux, and an encrypted track with no key
fails loud.

### hlssink

`hlssink::HlsSink` (`std`) is the publishing side. It cuts the byte stream a
muxer feeds it into media segment files and writes an `.m3u8` media playlist
beside them, rendered by the `hls` parser's `write_media` twin. The muxer stays a
separate element, `tsmux ! hlssink` or `mp4mux ! hlssink`, so one sink packages
either carrier.

A segment may only start at a keyframe and closes at the first one at or past
`target-duration`, where `0` cuts at every keyframe. For MPEG-TS one input frame
is one access unit, so `FrameTiming::keyframe` marks the candidates and the frame
PTSs give the durations. For fMP4 the stream is walked as boxes, a `moof` whose
first sample is a sync sample opens a fragment and is a candidate, and the `trun`
durations in the track timescale give the exact segment length, with `ftyp` and
`moov` split off once into the `#EXT-X-MAP` init segment. Nothing is added or
dropped, so the init segment plus the media segments concatenate back to the
muxer's own byte stream. `playlist-length` bounds the listed window, advancing
`#EXT-X-MEDIA-SEQUENCE` as segments roll off, and `max-files` deletes the files
that leave it, which is the live case. EOS appends `#EXT-X-ENDLIST` for VOD.

### DASH

`dashsrc::DashSrc` (`dash`) is the MPEG-DASH analog. It parses a static MPD with
the `mpd` parser over `roxmltree`, selects a Representation, and streams its fMP4
init and media segments into `fmp4demux`.

A Representation addresses its segments by a `SegmentSource`, one of three:

- a `SegmentTemplate`, either the `@duration` profile or a `SegmentTimeline`
  whose `<S t d r>` entries expand into per-segment times, addressed by
  `$Number$` or `$Time$`
- a `SegmentList`, an explicit ordered list of `<SegmentURL>`, each a `@media`
  URL or a `mediaRange` byte range of the `BaseURL` resource, with an
  `<Initialization>`
- a `SegmentBase`, one resource whose fragment byte ranges live in a `sidx`
  Segment Index box at `indexRange`, fetched and parsed at run time via
  `parse_sidx` and `Sidx::subsegments`, the index bytes never pushed downstream

All three resolve to one `ResolvedSegment { url, byte_range, time }` list, so a
range-carrying entry fetches just its sub-range with an HTTP `Range` request, the
DASH analog of HLS `#EXT-X-BYTERANGE`, letting a single-file CMAF DASH stream
play. A `SegmentTemplate`'s `@presentationTimeOffset` is the media instant that
lines up with the start of the Period, so `$Time$` URLs keep the media value
while every `ResolvedSegment.time`, and with it seek matching and the
Period-boundary `Segment`, is period-relative presentation time.

A dynamic live MPD is reloaded on its `minimumUpdatePeriod`, each new segment
played once tracked by start time, ending when the manifest turns static, the
same shape as the HLS live reload. Its wall-clock window comes from
`availabilityStartTime` plus `Period@start` bounded by `timeShiftBufferDepth`,
with `@availabilityTimeOffset` publishing each segment that many seconds before
its nominal completion, clamped to one segment duration so a chunked packager's
in-progress segment is reachable but nothing beyond it.

Every addressing mode but SegmentBase is held to the same availability rule. A
segment is fetched once `availabilityStartTime` plus `Period@start` plus its end,
less the clamped `@availabilityTimeOffset`, has passed on the server clock. The
`@duration` profile, each `SegmentTimeline` `<S>` entry and a `SegmentList` are
stretches of equal-length segments, so one window computation serves all three.
A negative `@r` repeats to the next entry's `@t`, else the Period end, else until
the clock stops it. The server clock comes from the first supported `UTCTiming`
element that answers. The `ntp` and `sntp` schemes send one NTP v4 client request
over UDP with a one second timeout.

A media segment or live reload that fails with a transport error, 5xx, 404 or 425
is retried three times with a 250 ms doubling backoff, gst adaptivedemux2's
`max-retries` default, in both `DashSrc` and `HlsSrc`. On a live DASH segment,
repeated 404 or 425 answers count against that budget at most once per segment
duration. Startup manifest and playlist fetches do not retry.

A multi-Period live MPD is played as gst's `dashdemux2` plays it. Playback joins
the Period holding now minus the presentation delay. A Period with an end, its
`@duration` or the next Period's `@start`, offers `ceil(span / d)` segments
(ISO/IEC 23009-1 segment count), so the window never reaches past it. The next
Period starts once the current one's last segment has been offered, since a
manifest may list a Period before the one playing has ended.

`with_abr()` makes it throughput-adaptive on the same shared
`abr::BandwidthEstimator` as `HlsSrc`. A `load_rep` helper resolves any
Representation, Template, List or `sidx`-fetched SegmentBase, into the run loop's
segment, timescale and init working set, and the estimate-derived cap drives both
the per-reload pick and a per-segment re-selection, so a static VOD adapts within
one pass, re-emitting the init on a switch.

`low-latency=true` changes how a segment is consumed, not when it is fetched. The
response body is read as a stream and each complete CMAF chunk, `styp` or `moof`
plus `mdat`, is pushed downstream as it arrives, so a segment the packager is
still writing flows at chunk latency instead of segment latency. The split is
`fmp4::CmafChunker`, an incremental box framer over the arriving bytes sharing
`mp4box::next_box_len` with `fmp4demux`, that cuts after every `mdat` and bounds
both a declared box size and its pending run by the segment cap, so a hostile
length fails the fetch instead of buffering on it. Every byte comes out exactly
once in order, so the demuxer sees the same byte stream a whole-response fetch
delivers. Byte-range segments and a set `prebuffer-ms`, which owns emission
order, stay on the whole-response path.

## Still images

A still image is the smallest case of a byte stream carrying coded frames, and it
takes the same shape as a container. A PNG or WebP file is
`CompressedVideo{Png}` or `CompressedVideo{WebP}`, one access unit per file, so
`pngdec` and `webpdec` are ordinary decoders that `decodebin` auto-plugs, and a
still is one frame of a video stream rather than a separate kind of media.
`typefind::sniff_caps` types both by magic, the PNG signature and `RIFF` plus
`WEBP`, which is what a `.png` or `.webp` extension resolves through, since
filesrc arms content sniffing on an extension it does not know. JPEG is
deliberately not typed by content: `mjpegdec` takes one whole access unit per
buffer, so a `.jpg` would plug a decoder that fails past the source's read size
until a `jpegparse` exists.

A still image is also a one-frame `CompressedVideo` stream whose decoders,
`MjpegDec`, `PngDec` and `WebPDec`, take one whole image per buffer, so a byte
source that hands over read-sized chunks needs a framer between them.
`jpegparse` and `pngparse` walk the format's own structure, JPEG markers and PNG
chunks, and emit one image per buffer with the geometry its header declares. The
auto-plug chain splices the framer ahead of the decoder, which is what lets a
`.jpg` be typed by content at all.

`mjpegdec` decodes with the pure-Rust zune-jpeg. zune-jpeg has no RGB to RGBA
mapping, so an RGB-coded JPEG decodes to RGB and `videoconvert` adds the alpha.
On Linux with `ffmpeg-long-tail`, a JPEG zune-jpeg rejects, such as one with
12-bit samples or a sampling factor of 3, is decoded by libavcodec's MJPEG
decoder and converted to RGBA, then follows the same output path.

Interlaced motion JPEG in AVI stores each frame as two JPEG images, one per
field, each half the height the AVI header declares. `jpegparse` treats an image
under three quarters of the upstream frame height as a field, the threshold
ffmpeg and GStreamer use, and sends both fields of a frame as one buffer declared
at twice the field height. `mjpegdec` decodes an access unit whose first image is
half the declared height as two fields and weaves their rows into one frame with
`Interlace::Interleaved` caps, the first field on the even rows, as GStreamer's
`jpegdec` does. ffmpeg puts the first field on the odd rows instead when the AVI
fourcc is `MJPG` and the `strf` extradata carries no AVID `PAL` token. The caps
have no field order to carry that from `avidemux`, so g2g does not follow it.

The decoders do not assume one file per buffer, because a byte source hands over
read-sized chunks (`filesrc`) or whole files (`multifilesrc`). Both cases go
through `stillimage::ImageAssembler`, which accumulates until the format's own
self-describing length says an image is complete: a PNG's chunk list walked to
the end of `IEND`, or a WebP's RIFF size field. A stream that ends mid-image
reports it at EOS rather than decoding a partial file, and the bytes held while
waiting are bounded, so a plausible signature followed by silence cannot grow the
buffer for as long as the stream flows.

Geometry is the file's word, so it is checked against a per-side and a total-byte
budget (`stillimage::rgba_byte_size`) before any buffer is sized. Both decoder
crates size their output from the header, and a 100000x100000 `IHDR` or a
20000x20000 `VP8X` canvas would otherwise ask for tens of gigabytes from a file
of a few dozen bytes.

Output is always 8-bit RGBA, with palette and sub-byte grayscale expanded, 16-bit
narrowed to its high byte, and alpha added where the file has none, announced by
a `CapsChanged` before the first frame and on any change, since a sequence of
stills can change size mid-stream. `pngenc` is the inverse: RGBA or RGB in, one
lossless PNG per frame, `compression-level` as zlib's 0 to 9. There is no WebP
encoder, because the only pure-Rust one does VP8L lossless alone, with none of
`webpenc`'s quality, speed or preset knobs.

PNM (`pnmenc` and `pnmdec`, `VideoCodec::Pnm`) is the same still-image shape with
no extra crate: a Netpbm PBM, PGM or PPM, `P1` to `P6`. `pnmenc ascii=` writes
ASCII P3 instead of binary P6. Decode always emits packed RGB8, with PGM and PBM
expanding to grey and black-white, because there is no GRAY8 raw format. The
`.pnm`, `.ppm`, `.pgm` and `.pbm` extensions type by name and `P1` to `P6` by
magic, so `filesrc location=x.ppm ! decodebin` plugs `pnmdec`.
