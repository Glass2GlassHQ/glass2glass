# Text, captions and overlay

Timed text and its renderers: subtitle parsing, the CPU and GPU overlay
elements, closed captions in and out of a video bitstream, bitmap subtitle
formats, and teletext. Part of the design in [README.md](README.md).

## textoverlay

`textoverlay::TextOverlay` is the `textoverlay` and `subtitleoverlay` analog. It
renders timed subtitle text onto a raw video frame. The path splits into two
`no_std` pieces feeding one element.

`subparse` parses SRT (SubRip) and WebVTT into a common timed `Cue` of
`{ start_ns, end_ns, text, settings }`. Both formats are blank-line-separated
blocks with a `start --> end` timing line and text on the following lines, so one
block walker covers both. The shared timestamp parser accepts the SRT comma and
the WebVTT dot fractional separators plus the WebVTT short `MM:SS.mmm` form.
Leading lines, an SRT index or a WebVTT cue id, before the `-->` line are
ignored, the `WEBVTT` header and `NOTE`, `STYLE` and `REGION` blocks are skipped,
and inline markup (`<i>`, `<c.class>`, inline cue timestamps) is stripped. BOM
and CRLF are tolerated. Malformed blocks are skipped rather than failing the
parse, the way players tolerate dirty files. The WebVTT cue settings after the
end timestamp are parsed into `CueSettings { position, line, align }`, the
placement subset the bitmap overlay honours, and `size`, `vertical` and `region`
are recognised but not applied.

`bitmapfont` is an embedded 8x8 bitmap font, MSB being the leftmost column, so
the baseline draws glyphs with no font file or rasterizer. It is an all-caps font
covering A-Z, 0-9, space and common punctuation, and lowercase folds to
uppercase.

### Placement

`TextOverlay` is an RGBA8-in, RGBA8-out identity transform on the pixels, with
`VideoConvert` upstream for other formats, except for the active cue text. By a
linear scan, since subtitle tracks are small, it draws every cue covering the
frame's `pts_ns` rather than just the first, because WebVTT and SRT allow
overlapping cues to show at once.

Each cue is placed independently from its `CueSettings`. `position`, a percentage
of width, is the horizontal anchor and `align`, one of start, center and end,
decides how the box extends from it. An explicit `line`, a percentage of height,
places the box vertically, while auto-`line` cues stack upward from the bottom in
cue order so overlapping subtitles do not collide. The WebVTT `vertical:rl` and
`lr` writing mode is parsed into `CueSettings::vertical` and carried end to end,
but the bitmap overlay lays text out horizontally. Each cue draws over its own
translucent backing box, integer-scaled to the frame height.

Cues are set programmatically through `from_srt` and `from_webvtt` or, on `std`,
through the `location=` property loading a `.srt` or `.vtt` file, typed by
extension and otherwise by content sniff. The element is registered as
`textoverlay` for the `gst-launch` text parser. This mirrors the analytics
overlay's CPU baseline in [ml.md](ml.md): the `no_std` bitmap
renderer is the portable path.

### TrueType and shaping

The `truetype-overlay` feature replaces the bitmap font with a real one.
`fontdue` parses a `.ttf` or `.ttc` and rasterizes each glyph to a coverage
bitmap, alpha-blended onto the frame in the text colour, so CJK, accented Latin
and mixed-case render, laid out horizontal or vertical from
`CueSettings::vertical` with the same `position`, `line` and `align` placement.

`fontdue` does no font fallback, so `TextOverlay` holds a fallback chain that
`add_font` appends to: each glyph is drawn from the first face whose
`lookup_glyph_index` is non-zero, so a Latin primary plus a CJK fallback covers
mixed text. `fontdue` rasterizes glyf TrueType outlines only, and CFF and CFF2
faces use the `text-shaping` feature's cosmic-text backend for horizontal cues.
The `no_std` baseline keeps the bitmap font.

### WebVTT styles

A WebVTT `STYLE` block reaches the pixels. `parse_cue_styles` resolves `::cue`,
`::cue(#id)` and `::cue(.class)` rules onto each cue's `CueSettings`, and a
span-scoped rule lands as a `SpanStyle` run over the byte range its `<c.class>`
tag covers, so nested spans resolve per property, the innermost run that sets one
winning. The presentational `<b>`, `<i>` and `<u>` tags make the same kind of run
with no stylesheet at all, and a rule matching the same span overrides them.

Beyond `color` the properties honoured are `font-size` (in `px`, or a percent of
the size the cue itself draws at), `text-shadow`, `background-color`,
`font-weight`, `font-style`, `text-decoration: underline` and `font-stretch`, and
all three render paths apply them.

On the shaped path the sized runs become cosmic-text `Metrics` overrides on the
line's `AttrsList`, so a line mixing sizes is still one shaped, bidi-reordered
run and takes the tallest span's line height, and the `ab_glyph` renderer
rasterizes each character at its own size on a shared baseline.

A shadow is one offset copy of the glyphs in the shadow colour, drawn under every
glyph of the cue so a neighbour's shadow never lands on top of one. A blur radius
is applied by zero-padding the glyph's coverage mask and running it through three
separable box passes, sized so the stack matches the gaussian CSS asks for, with
a standard deviation half the radius, and the grown mask is tinted in the shadow
colour. Vello has no filter that blurs a glyph run, so the GPU backend draws a
blurred shadow as one tinted mask image per glyph, blurred by the same code, and
falls back to a glyph run when the radius is 0.

A whole-cue `background-color` is the backing box, and a span-scoped one fills the
line box behind that span's own glyphs, over the box and under the text.

Weight, slant and width are carried as per-span cosmic-text `Attrs`, so they pick
a face out of the font database, a real bold or italic face where the family has
one, else the `wght` variation axis for weight. There is no synthetic oblique, so
an italic run with no italic face installed renders upright. They reach the Vello
backend in the glyph ids and the face each run names. That face selection is the
shaped path only: a `vertical:rl` or `lr` cue on the `ab_glyph` column renderer
keeps the element's own `font-variations=` weight.

An underline is a filled bar in the run's text colour, drawn in the glyph layer so
a neighbour's shadow stays under it, below the baseline horizontally and down the
column's right edge vertically. `-webkit-text-stroke`, in the shorthand,
longhands and unprefixed spellings, is a dilation of the same coverage mask the
shadow blits: the mask goes down in the stroke colour at every integer offset
inside the stroke radius, over the shadows and under the fills, which the Vello
backend draws as one glyph run per offset.

`font-family` reaches the shaped path as a cosmic-text `Attrs::family`, with the
CSS generic names mapping to `Family::Serif` and its siblings and any other name
queried by name. The `ab_glyph` column renderer has one loaded face and ignores
it.

Selectors also cover the cue-text element names. `::cue(b)`, `::cue(i)`,
`::cue(u)`, `::cue(c)`, `::cue(v)`, `::cue(ruby)` and `::cue(rt)` match a run by
its enclosing tag, scoring below any class per CSS specificity. A compound like
`::cue(b.loud)` is not supported and its rule is dropped.

`<ruby>base<rt>annotation</rt></ruby>` takes the `rt` text out of the line flow
into a `RubyRun` carrying the base run's byte range, which both renderers draw at
half the cue size centred over that range's horizontal extent, or beside the base
column in a vertical cue, to the right of it for `vertical:rl`, so the base keeps
the baseline it would have had alone.

Sizes and offsets are clamped at parse time, because a stylesheet is as untrusted
as the rest of the subtitle file and the size becomes a glyph raster.

### GPU backend

`vellooverlay::VelloTextOverlay` (`vello-text-overlay`) is the GPU backend for the
same cues, for a pipeline that keeps frames on the GPU: RGBA8 in,
`MemoryDomain::WgpuTexture` out, like `VelloAnalyticsOverlay` beside it. It holds
a `TextOverlay` rather than its own state, so cue selection, `CueSettings`
placement, colours, font chain and shaping are one implementation. The shared step
lays each active cue out into canvas-absolute glyph positions, which the CPU
element blits as swash rasters and this one hands to Vello as glyph runs, drawn
from the very face cosmic-text's per-codepoint fallback resolved, so a mixed Latin
and CJK cue uses the same faces on both backends. Vertical cues use the CPU
element's column renderer.

### Streamed cues

`SubParse` feeds that renderer as a stream rather than from a file. It parses a
structured subtitle document arriving on its sink pad and emits each cue as a
timed `Text{Utf8}` frame, the PTS and duration being the cue window.

Parsing is incremental for the line-based formats, SRT, WebVTT and SSA: each
`process` call drains only the blocks bounded by a blank-line or newline
separator, retains the partial trailing block, and flushes the remainder at `Eos`,
so a cue streams out as soon as it is complete instead of all cues batching at
end of stream. Chunk-boundary UTF-8 splits and a leading BOM are handled. TTML is
XML with no blank-line boundary and stays batch.

`TextOverlayN` pairs the two as a `MultiInputElement`, a video pad plus a
text-stream pad, video out. It opts into the runner's `input_pts_ordered` merge so
each cue lands just before the first video frame it covers, and because `SubParse`
streams, the merge buffers video only up to the next cue rather than to the
subtitle stream's end.

Cue placement, the `CueSettings` `position`, `line` and `align`, cannot ride the
plain `Utf8` payload, so it travels as a `TextCueMeta` frame-meta under the
`metadata` feature that `SubParse` attaches and `TextOverlayN` reads, recovering
WebVTT and SSA positioning. On the ZST baseline, with no meta, streamed cues draw
at the renderer default.

### Subtitle tracks in containers

Cue streams also go back into a container. A `Caps::Text{Utf8}` pad on either
Matroska muxer or on `Mp4MuxN` is a subtitle track, taking one cue per frame with
the window on the frame's PTS and duration. The track's init needs nothing from
the stream, so it is fixed at configure rather than at the first cue, which may be
many seconds in while every other track waits on the header.

The two containers time a cue differently. Matroska states the window per block,
so a text block is always a `BlockGroup` carrying a `BlockDuration`, since a
`SimpleBlock` has nowhere to put one. The `subtitle-format` property picks the
storage syntax, `S_TEXT/UTF8`, the default and `subrip` to ffmpeg, or
`S_TEXT/ASS`, where each cue is framed as the mapping's
`ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,Effect,Text` event with `\N`
line breaks, behind a script-header `CodecPrivate`.

MP4 has no per-sample timestamp. A `tx3g` sample, a 2-byte big-endian text length
plus UTF-8, what ffmpeg calls `mov_text`, presents where the durations before it
end, so the run before the first cue and each run between cues is filled with an
empty sample, which is also what "no subtitle on screen" means in the format.

Either muxer's per-track metadata reaches a text track unchanged, so a subtitle
track's language and title ride the Matroska `TrackEntry` the way an audio
track's do.

## Closed captions

Closed captions, CEA-608 and CEA-708, feed the same renderer, but their bytes
ride inside the compressed video bitstream rather than in a container text track,
so the path is a track rather than a `SubParse`-style drop-in. The `cea` module
(`no_std`) holds the decoders.

`extract_cc_data` mines the `(cc_type, b0, b1)` caption triples from an access
unit's SEI `user_data_registered_itu_t_t35` messages, the ATSC A/53 `GA94`
`cc_data`, for H.264 (NAL type 6) and H.265 (prefix and suffix SEI), and from
picture `user_data` blocks (`00 00 01 B2`) for MPEG-1 and MPEG-2, the same ATSC
block without the T.35 prefix. Every count, length and offset is bounds-checked
so a malformed block yields no triples.

`Cea608` decodes the legacy line-21 path, `cc_type` 0 and 1: a 15x32 character
grid with pop-on, roll-up and paint-on modes, PAC row and indent positioning, the
basic, special and extended-Western-European character sets, and channel
selection over CC1 to CC4 with the other channel's interleaved codes ignored.

`Cea708` decodes the DTVCC path, `cc_type` 2 and 3. It reassembles the DTVCC
packets from the triples, splits them into service blocks, and runs the selected
service's window command stream (DefineWindow, the DisplayWindows family,
SetPenLocation, G0 and G1 text) against an eight-window model. Both emit the same
timed `Cue` `SubParse` produces.

### CcExtract

`CcExtract` wraps the decoders as a pipeline element: a compressed H.264 or H.265
stream in, timed `Text{Utf8}` cue frames out, the same shape `SubParse` emits, so
the existing overlay consumes either. The in-band case taps
`Caps::CompressedVideo` directly, since the captions ride in the video, and a
container caption track arrives on `Caps::ClosedCaption` instead. The element
selects one service at construction through `CcSource`, defaulting to CEA-608 CC1.

In the `playbin` auto-fan-out it sits on a tee of the parsed video: one tee branch
decodes for display, the other reframes to access units, so a TS PES does not
split an SEI NAL, and runs `CcExtract` into the video's `TextOverlayN` text pad.

Captions are not discoverable up front, so they are opt-in through a
`#closed-captions=cc1` URI fragment, aliased `#cc=`, or `service-N` or `708-N`,
the file-container analog of the HLS `#subtitle-lang=` hint. The MKV, TS and MP4
file hooks honour it, and so does `hls_playbin`, which tees the variant's video
the same way for a muxed-TS variant (`build_hls_ts_cc_overlay`), an fMP4 or CMAF
variant (`build_hls_fmp4_cc_overlay`, tracks from the `#EXT-X-MAP` init), or a
variant with a separate audio rendition (`build_hls_separate_cc_overlay`, the
audio merged in as its own source). In every case the explicit caption request
wins over an auto-selected subtitle track, since there is one overlay text pad.

### HLS subtitle renditions

`HlsSrc::variant_streams` surfaces a master playlist's `SUBTITLES` renditions as
`Caps::Text`, and `MasterPlaylist::pick_rendition` selects one by the
`#audio-lang=` or `#subtitle-lang=` URI hint, which the audio fan-out honours too.
`HlsSrc::with_text` emits `Caps::Text { WebVtt }` from a raw `.vtt` rendition, and
`build_hls_subtitle_overlay` joins it through `SubParse` into the video's
`TextOverlayN` across sources, wired by `hls_playbin` for a muxed A/V TS variant
plus a `SUBTITLES` rendition. `build_hls_separate_subtitle_overlay` covers the
three-source shape: the variant's video TS, a distinct audio rendition and a
distinct WebVTT rendition in one graph.

### Caption authoring

The encode direction is the mirror image, for caption authoring and broadcast
egress. `cea::Cc608Enc` is the inverse of the `Cea608` decoder: fed cues of text
plus placement it builds the pop-on command sequence (RCL, a PAC per row, the row
text, EOC, and EDM to erase) and queues the `(cc_data_1, cc_data_2)` byte pairs,
doubling the control codes and setting odd parity.

`cea::Cc708Enc` is the 708 counterpart and the inverse of `Cea708`. It builds the
window command stream (DefineWindow for a hidden window sized to the text and
anchored from the cue's relative placement, SetPenLocation per row, the G0 text,
DisplayWindows to reveal it atomically, and HideWindows to erase), packs the
commands into DTVCC service blocks without ever splitting a command across the
31-byte `block_size`, wraps each in a DTVCC packet, and emits the `cc_type` 3 and
2 triples. Either drains one caption unit per video frame, a byte pair or a
triple, padding when idle.

`CcInsert` is the element wrapping them and the inverse of `CcExtract`: a
compressed H.264 or H.265 access-unit stream plus a timed cue stream in, a
`MultiInputElement` merging the two pads by PTS, and the same video out with a
`GA94` caption SEI (`cea::build_cc_sei`, the inverse of `extract_cc_data`) written
before each access unit's first VCL slice. It encodes CEA-608 by default or
CEA-708 via `CcInsert::cea708`. The video provides the frame clock, a cue is
queued on arrival and erased when its window ends, and a warning fires if cues
arrive against an untimed video source, where the merge would drop them.

`SubtitleSrc`, a `.srt`, `.vtt`, `.ssa` or `.ttml` file as a `Text` stream, is the
head of the authoring pipeline, so
`subtitlesrc -> subparse -> ccinsert -> tsmux`, the `examples/cc_author.rs` flow,
embeds captions from a subtitle file. The whole
`subparse -> ccinsert -> ... -> ccextract -> textoverlay` round trip is pure
in-graph.

### Caption transports

Out of band, the same triples travel in four byte layouts, and each carrier picks
a different one, so the layout is part of the media type.
`ClosedCaptionFormat` names it: `Cea608` and `Cea708` are packed ATSC `cc_data`,
`Cea708Cdp` is a SMPTE ST 334-2 caption distribution packet, `Cea608S334` is ST
334-1 Annex A triplets, and `Cea608Raw` is bare byte pairs of one line-21 field.
The `cea` module holds the parse and write pair for each behind
`parse_caption_transport` and `write_caption_transport`.

`CcConverter` (`ccconverter`) re-lays a payload from its `in-format` layout into
its `out-format` one, one output frame per input frame with the timing kept, so an
MP4 caption track feeds an ancillary-data packetizer and back. A conversion is
lossy exactly where the standards are: bare CEA-608 pairs hold one field, ST 334-1
holds no DTVCC, and a triple the target layout cannot carry is dropped rather than
mistyped.

`CcCombiner` (`cccombiner`) is where a separate caption stream rejoins the video
it belongs with: a two-pad `MultiInputElement` shaped like `SubPictureOverlay`,
video on pad 0, whose caps the merged output follows, and captions on pad 1, on
the runner's `input_pts_ordered` merge. It touches no pixels. Each video frame
leaves carrying the triples queued for it as `meta::CaptionMeta`, which is what
`CcInsert::from_meta` writes back into the bitstream's SEI, so
`cccombiner -> encode -> ccinsert` is the authoring path for a stream whose
captions arrived beside it. `max-scheduled` bounds the queue against a stalled
video pad and `input-meta-processing` says which set wins when the video frame
already carries captions of its own. The caption meta is the `metadata` feature's
typed container, so the element is gated on it.

### Subtitle writers

`SrtEnc` (`srtenc`) and `WebVttEnc` (`webvttenc`) invert `SubParse`: timed
`Text{Utf8}` cues in, one `Text{Srt}` or `Text{WebVtt}` frame per cue out holding
that cue's document block, so `... ! srtenc ! filesink location=out.srt` records a
subtitle file. The block is written by `subparse::write_cue_block`, the inverse of
the parsers and the same code the HLS WebVTT segment writer uses, so the two
halves cannot drift.

The `timestamp` and `duration` properties shift the written cue window without
moving the frame the block rides on. A WebVTT document opens with its `WEBVTT`
signature, written even for a stream that carries no cue, and a cue whose text is
blank writes nothing, since an empty block would end the preceding cue early when
the document is read back.

## Bitmap subtitles

Bitmap subtitles are the one subtitle family that is not text, so they get their
own coded media kind: `Caps::SubPicture { format: SubPictureFormat }`, a stream of
coded bitmap cues over `VobSub`, the DVD subpicture format, `DvbSub`, ETSI EN 300
743, and `Pgs`, the Blu-ray HDMV Presentation Graphic Stream.

It sits beside `Caps::Text` rather than inside it, because nothing downstream of a
`Text` link can render a palette-indexed run-length bitmap. `Caps::ClosedCaption`
is the model: a coded carriage variant whose decoder produces something the rest
of the graph already understands, which here is raw pixels.

`VobSubDec` (`vobsubdec`, gst's `dvdsubdec`), `DvbSubDec` (`dvbsubdec`, with no
gst alias since gst's `dvbsuboverlay` is a video-overlay element rather than a
bare decoder) and `PgsDec` (`pgsdec`, and gst has no PGS decoder at all) all emit
one full-frame transparent `Caps::RawVideo{Rgba8}` canvas per cue at the
subpicture display geometry, stamped with the cue's PTS and duration, so the
consumer is a pixel one: `subpictureoverlay` or the ordinary `compositor`.

A cue ends with a second, fully transparent canvas at its hide time. Either
consumer holds an overlay pad's last frame between output frames, and a zero-alpha
source-over is a no-op, so the clear canvas is exactly what makes a cue disappear
on time. One more empty canvas opens the stream, so the consumer is not waiting on
this input for however long it is until the first cue.

### SubPictureOverlay

`subpictureoverlay::SubPictureOverlay` puts those canvases on the picture. It is a
two-pad `MultiInputElement` shaped like `TextOverlayN`, video on pad 0 and the
decoder's canvases on pad 1, opting into the runner's `input_pts_ordered` merge so
a canvas lands just before the first video frame it covers.

It holds the last canvas whose PTS the video has reached and source-over blends it
onto every frame, so a cue stays up between canvases. A canvas with no drawn pixel
is dropped rather than held, which is how the clearing canvas takes the cue down.
Both pads are RGBA8 on the CPU, with `videoconvert` on either side for another
format, and a canvas whose geometry differs from the video is resampled onto it by
the `compositor`'s bilinear scaler, so a PAL-sized subpicture composites onto a
scaled picture.

`mkv_playbin` auto-plugs it: a Matroska bitmap-subtitle track decodes to canvases
and feeds this overlay where a text track feeds `subparse` and `TextOverlayN`. In
a launch line it is a fan-in muxer built by link degree like `textoverlay`, with
the video and subpicture branches on its `video` and `text` request pads. The MPEG
program-stream DVD `playbin` composites its subpicture track with `compositor` at
the video's own geometry instead.

### VobSub

The VobSub bitstream (`vobsub.rs`, `no_std`) is one subpicture unit per cue: a
packet size and a control-sequence offset, 2-bits-per-pixel run-length data in two
interlaced fields, even rows then odd, each row byte-aligned, then control
sequences carrying the display rectangle, four palette indices and four alpha
nibbles, the two field offsets, and the show and hide dates in 1024/90000 s units.

The 16-entry RGB palette and the display size are not in the bitstream. They ride
the `.idx` text a Matroska `S_VOBSUB` track carries as its `CodecPrivate`, which
`MkvDemux` forwards in band ahead of the first cue the way it forwards the FLAC
and Opus headers, and which the decoder tells apart from a cue by parsing it as
`.idx` first.

That same text is also the sidecar carriage. `VobSubSrc` (`vobsubsrc`) reads a
`.idx` and `.sub` pair off disk, emits the `.idx` as that same in-band config
frame, then reads each cue's subpicture unit out of the `.sub` at the byte offset
its `timestamp:` line names, stamped with that timestamp and with the unit's own
hide time as its duration, so
`vobsubsrc location=movie.idx ! vobsubdec ! compositor.` plays a sidecar pair the
way a muxed track plays.

A `.sub` is an MPEG-2 program stream, so a unit is reassembled from the
`private_stream_1` PES packets at that offset carrying the same subpicture
substream id, bounded by its own 16-bit packet size. An `.idx` indexing several
languages picks one by `id:` code (`language=`) or by the file's `langidx:`. An
entry pointing outside the `.sub`, or a unit the file ends in the middle of, drops
that cue and not the stream.

Every size, offset and coordinate off the wire is range-checked and the parse
returns `None` rather than allocating on a bogus rectangle. The pixel data is
bounded by the control-sequence offset, so a truncated packet fails instead of
decoding the control table as run lengths. A track that declares a Matroska
`ContentCompression` is refused outright, since its blocks are not SPU packets and
nothing here inflates them.

### DVB subtitles

DVB subtitles (`dvbsub.rs`, `no_std`) are the broadcast sibling and a different
shape: not one packet per cue but a segment stream, with decoder state carried
across display sets. Each data field holds segments sharing a `page_id`:

- a display definition, giving the display geometry, 720x576 without one
- a page composition, giving the `page_time_out`, the page state, and where each
  region sits
- a region composition, giving a region's size, 2-, 4- or 8-bit depth, CLUT and
  background, and which objects are drawn into it where
- CLUT definitions, Y, Cr and Cb plus a transparency at full or packed precision,
  converted through the same BT.601 fixed-point path a reference decoder uses so
  the rendered colours are identical
- object data, run-length coded pixels in two interlaced fields, with map tables
  lifting a shallower code into the region's depth

A page composition listing no region is how a cue ends, and a page whose timeout
expires before the next display set gets the same clear canvas at its deadline.

The composition and ancillary page ids are the out-of-band part. A Matroska
`S_DVBSUB` track's `CodecPrivate` carries them, and `TsDemux` synthesizes the
identical five-byte blob from the PMT `subtitling_descriptor` (tag 0x59) that
marks a private (0x06) stream as DVB subtitles, so both carriages reach the
decoder the same way. The decoder takes a data field with or without its PES
`data_identifier` header, since a Matroska block carries the bare segments.

Every segment length, region dimension, CLUT entry id and object position is
bounds-checked: a display set whose segment layer does not hold together is
dropped whole, and a region past `MAX_REGION_PIXELS` is never allocated.

### Blu-ray PGS

Blu-ray PGS (`pgs.rs`, `no_std`) is a segment stream too, but a flatter one. A
display set is a presentation composition, giving the video geometry, the epoch
state, which palette to read, and up to two objects with their positions, then
window definitions, palette definitions and object definitions, terminated by an
end-of-display-set segment. Objects are 8-bit run-length coded and drawn straight
onto the video, with no region layer and no interlaced fields, and a cropped
composition object shows only a sub-rectangle of its bitmap, drawn at the
composition position with the crop offset indexing the object bitmap alone.

Cropping has no reference peer, because ffmpeg parses the rectangle without
applying it. It is verified instead by anchoring it to the uncropped path that
ffmpeg does pin pixel for pixel: cropping is a pure selection, so a fixture whose
every object pixel is a different colour is presented both whole and cropped, and
the cropped canvas has to equal the window of the whole one, with the crop of the
entire object equal to no crop at all. The placement convention that oracle cannot
settle comes from libbluray's `graphics_controller.c`, which does implement
cropping. A crop rectangle running off the object is clamped to what is there
rather than trusted the way libbluray trusts a disc.

Palettes and objects persist across an epoch, keyed by id, so a later palette
segment updates only the entries it names and an object too big for one segment
arrives in fragments whose total is fixed by the first one's declared length.
Nothing rides out of band: the palette is in the stream and the geometry is in the
presentation composition, so unlike the other two codings there is no config frame
ahead of the first cue.

Palette entries are Y, Cr and Cb plus an alpha that passes through unscaled,
converted through the shared limited-range fixed-point path in `paint.rs`, whose
matrix a PGS stream picks by video height, BT.709 above 576 lines and BT.601 at or
below, since the format states no colorimetry.

PGS has no end-of-display time either: a cue stands until a later display set
replaces it, and a presentation composition listing no object is how the stream
ends one, so the clear canvas is the stream's own rather than synthesized from a
hide time. Both the `.sup` per-segment `PG` / PTS / DTS framing and the bare
Matroska `S_HDMV/PGS` block framing are accepted, told apart by the magic since no
segment type is 0x50.

Every segment length, object dimension, run length, palette index and composition
count is checked before use: a run overflowing the bitmap or codes that do not
cover it drop the object, an object larger than the video or past
`MAX_OBJECT_PIXELS` is never allocated, and a truncated segment stops the walk
with the display sets that did parse intact.

### Muxing bitmap subtitles

The write paths mirror those two carriages. A `Caps::SubPicture` input pad on
either Matroska muxer becomes an `S_VOBSUB` or `S_DVBSUB` track, and a
`Caps::SubPicture{DvbSub}` pad on either TS muxer becomes a private (0x06) stream
whose PMT entry carries the `subtitling_descriptor` naming its language, type and
pages. That descriptor replaces the `KLVA` registration a bare private stream
would otherwise get, the same substitution the teletext descriptor makes.

The out-of-band configuration each format needs is not a property but the in-band
config blob the stream already leads with, so `mkvdemux`, `tsdemux` and
`vobsubsrc` all feed a muxer without translation. A VobSub pad's `.idx` becomes
the `CodecPrivate`, normalized to the `size:` and `palette:` lines a container
holds, since the cue index is a sidecar's file offset table, and a DVB pad's
five-byte page ids become the `CodecPrivate` or the descriptor's page fields. A
stream that sends no blob is declared on the `dvbsub-page-id` property's page,
defaulting to 1 like ffmpeg.

The two carriages frame a display set differently, so the muxers convert. A
Matroska block holds the bare segments, a TS PES payload wraps them in the EN 300
743 data field, with `data_identifier` 0x20 and a subtitle stream id ahead and the
end marker behind, and both directions run through `segment_span`, which finds the
segment run by walking its headers rather than trimming bytes.

Subtitle blocks, bitmap as well as text, are written as a `BlockGroup` so a cue's
display window rides its `BlockDuration`. Both Matroska muxers take these pads,
the fan-in `MkvMuxN` beside its A/V tracks and the single-track `MkvMux` on its
one sink pad, so a sidecar subtitle file muxes over one link
(`vobsubsrc location=movie.idx ! matroskamux ! filesink`) rather than the `name=m`
shape. The mapping they share, the codec each format writes, the config-blob
recognition, the block framing and the `S_TEXT/ASS` script header, lives in
`matroska.rs` beside `MkvTrackSpec`, so the two cannot drift.

## EBU teletext

Teletext (`teletext.rs`, `no_std`) is the third TS subtitle carriage, and unlike
the two above it is characters rather than pixels, so it lands on the plain text
pad instead of a canvas: `Caps::Text { Teletext }` in, `Caps::Text { Utf8 }` cues
out of `TeletextDec` (`teletextdec`), which is the same pad a `subparse`d SRT
track produces and therefore the same `TextOverlayN` input.

DVB carries teletext in a private PES (EN 300 472): a `data_identifier` byte then
fixed 46-byte data units, each one broadcast line, holding a framing code, a
hamming 8/4 magazine and packet address, and 40 odd-parity bytes. Those address
and data bytes are transmitted LSB first, so each is bit-reversed before any code
word means anything, while the two bytes ahead of them are ordinary MSB-first
fields.

Packet X/0 is the page header, carrying the page number, the C6 subtitle bit, and
the national option subset the G0 set is read under. X/1 to X/23 are the display
rows, and the decoder holds the rows of the addressed page until the next header
for it replaces or erases the page, which is what fixes the cue's duration and
puts each cue out one page late. Spacing control codes and parity failures render
as spaces so a row keeps its columns, and a double-height row's blanked bottom
half is dropped so the line appears once. Enhancement packets (X/26, X/28, M/29)
are not read, so the national option comes from the header bits and the wider
seven-bit G0 selection is out of reach.

Which page to follow is out of band, as for DVB subtitles. `TsDemux` synthesizes
an eight-byte selection blob from the PMT `teletext_descriptor` (tag 0x56, or the
identical `VBI_teletext_descriptor` 0x46) and forwards it in band ahead of the
first line, and the `page` property overrides it. With neither, the first subtitle
page the stream carries is adopted. The blob leads with `0xFF`, which cannot begin
a teletext payload, so the decoder tells the two apart on one pad.

Every data unit length, hamming code, parity bit and page address is checked
before use, so a corrupt line or a unit length past the payload drops that line or
ends the walk rather than propagating a corrupt page.

## Pixel filters

`hsvfilter`, `hsvdetector` and `roundedcorners` work on packed RGBA and BGRA.
`hsvfilter` applies `hue-shift`, `saturation-mul`, `saturation-off`, `value-mul`
and `value-off` in HSV. `hsvdetector` writes alpha 255 inside the configured HSV
box and 0 outside. `roundedcorners border-radius-px=` punches the corners
transparent.
