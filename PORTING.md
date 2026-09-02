# Porting from GStreamer to glass2glass

A practical guide to moving pipelines, applications, and custom elements from
GStreamer to **g2g**. It assumes familiarity with GStreamer (`gst-launch-1.0`,
`GstElement`, `GstBaseTransform`) and points at the g2g equivalents.

> TL;DR: many `gst-launch` lines run unchanged through **`g2g-launch`**; element
> names mostly match (with aliases); the big conceptual shift is that g2g graphs
> are **statically typed and composed in Rust**, not built from runtime
> string-keyed factories, and plugins are **Rust crates registered at build
> time**, not scanned `.so` files.

---

## 1. Mental model: what's the same, what's different

| Concept | GStreamer | glass2glass |
| :--- | :--- | :--- |
| Pipeline text | `gst-launch-1.0 ...` | `g2g-launch ...` ([g2g-plugins/src/bin/g2g-launch.rs](g2g-plugins/src/bin/g2g-launch.rs)) |
| Introspection | `gst-inspect-1.0` | `g2g-inspect` ([g2g-plugins/src/bin/g2g-inspect.rs](g2g-plugins/src/bin/g2g-inspect.rs)) |
| Media discovery | `gst-discoverer-1.0` | `g2g-discover` ([g2g-plugins/src/bin/g2g-discover.rs](g2g-plugins/src/bin/g2g-discover.rs)) |
| Element | `GstElement` (GObject, runtime) | a Rust type implementing `AsyncElement` / `SourceLoop` / `MultiInputElement` |
| Pads / caps | `GstPad`, `GstCaps` strings | typed `Caps` enum + `intercept_caps`/negotiation |
| Pipeline object | `GstPipeline` | `Graph` + `run_graph` |
| Bus | `GstBus` messages | `Bus` / `BusMessage` |
| Plugins | scanned `.so` from a plugin path | Rust crates that call `Registry::register_*` (build time), or dynamic `.so` via `declare_plugin!` + `--plugin` / `$G2G_PLUGIN_PATH` (§7c) |
| Threading | streaming threads per element | cooperative async tasks on one runner |

The runtime core (`g2g-core`) is `no_std + alloc`; OS-coupled elements live
behind cargo features. That's invisible when porting pipelines, but it's why
"plugins" are crates, not dynamically-loaded objects (see §7).

---

## 2. Porting a `gst-launch` pipeline

Start by pasting the line into `g2g-launch`:

```sh
g2g-launch videotestsrc num-buffers=30 ! videoconvert ! fakesink
```

`g2g-launch` parses the GStreamer DSL ([g2g-core/src/runtime/launch.rs](g2g-core/src/runtime/launch.rs))
and runs it against the standard registry. Supported syntax: linear chains,
`element key=value`, `tee name=t` fan-out with `t.` branch refs, muxer fan-in
(`src ! m.  src ! m.  funnel name=m ! sink`), demuxer fan-out with named pads
(`filesrc location=movie.mkv ! matroskademux name=d  d.video_0 ! ...  d.audio_0 ! ...`),
inline caps filters (`! video/x-raw,format=NV12,width=640 !`), `queue`/`queue2`
(mapped to a per-edge backpressure policy), `decodebin` / `uridecodebin` / `playbin`.

A **demuxer fan-out** (`matroskademux` / `tsdemux` / `qtdemux` fed by a file
source) probes the file at parse time and splits it into its elementary streams,
one per output-pad reference. Pad names select by media kind the GStreamer way:
`d.video_0` is the first video stream, `d.audio_0` the first audio, `d.text_0`
(or `d.subtitle_0`) the first subtitle track, `d.src_2` the third stream overall
(a bare `d.` is positional). Each branch names its own downstream
(`d.video_0 ! h264parse ! avdec_h264 ! autovideosink`). File sources only; a
network source still uses `playbin`. Subtitle pads work for `matroskademux` and
`qtdemux` (MPEG-TS carries no demuxer subtitle track).

**`decodebin name=d` fans out and decodes** each branch for you (the gst idiom):
`filesrc location=movie.mp4 ! decodebin name=d  d.video_0 ! videoconvert !
autovideosink  d.audio_0 ! audioconvert ! autoaudiosink` probes the file and
auto-plugs a demuxer plus a decoder per requested pad, so each `d.` branch gets raw
frames without naming the parser/decoder. Same `d.video_0` / `d.audio_0` / `d.` pad
grammar as the raw demuxer fan-out above; the difference is `decodebin` decodes,
`matroskademux`/`qtdemux` hand you the elementary stream. (g2g resolves the pads by
probing at parse time rather than gst's runtime `pad-added`, so it's file sources
feeding a multi-stream container.)

**Named input / request pads** let a fan-in reference its inputs by name in any
order: `d.video_0 ! ... ! o.video   d.text_0 ! o.text   textoverlay name=o ! ...`
and `... ! m.video_0   ... ! m.audio_0   matroskamux name=m ! ...`. Bare `m.`
inputs stay positional (by reference order). The muxer defines the pad scheme, so
`textoverlay`'s video always lands on its primary pad regardless of write order.

**Subtitle overlay** uses `textoverlay` as a fan-in, the analog of GStreamer's
`textoverlay` text_sink request pad: link a video branch and a text branch into
one named `textoverlay` (video first, then text), and it paints the cues onto the
video by PTS. The text comes from a demuxed subtitle pad or a `subtitlesrc` file
run through `subparse`:

```
filesrc location=movie.mkv ! matroskademux name=d
  d.video_0 ! h264parse ! avdec_h264 ! videoconvert ! o.
  d.text_0 ! o.
  textoverlay name=o ! videoconvert ! autovideosink

subtitlesrc location=subs.srt ! subparse ! o.   # or an out-of-band .srt/.vtt
```

The write direction is `srtenc` / `webvttenc`, which take the same timed cues and
write a `.srt` / `.vtt` file, so a subtitle track is extracted with
`filesrc location=movie.mkv ! matroskademux name=d   d.text_0 ! srtenc ! filesink
location=subs.srt`.

**Typefind.** GStreamer's `filesrc` emits untyped bytes and a downstream
`typefind` sniffs the media type at runtime. g2g negotiates types statically, so a
byte source must announce its type up front, but you rarely name it by hand: a bare
`filesrc location=X` derives its type from the extension (`.mp4`/`.mkv`/`.ts`/
`.ogg`/`.flv` containers, `.vtt`/`.srt`/`.ass`/`.ttml` subtitles), so
`filesrc location=subs.vtt ! subparse` and `filesrc location=movie.mkv !
matroskademux name=d ...` run with no hint. For a mis-named or extensionless file,
`bytestream-format=auto` sniffs the header content instead (containers by magic,
subtitles by their signature) and works through `decodebin` too, so a `.mp4`
wrongly named `.ts` still decodes with `filesrc location=movie.ts
bytestream-format=auto ! decodebin`. An explicit `bytestream-format=` always
overrides; a mislabeled file *without* `auto` is trusted by its extension.
A progressive (whole-file) `.mp4` decodes through `decodebin` too
(`filesrc location=X.mp4 ! decodebin ! …`, or explicitly `! qtdemux ! h264parse
! …`); the streaming fragmented form (CMAF, from HLS / DASH) stays on `fmp4demux`.
A still image needs no hint either: `filesrc location=X.png ! decodebin` and the
`.webp` equivalent type by magic and reach `pngdec` / `webpdec`, which decode to
RGBA. `pngenc` writes one lossless PNG per frame; the single-image JPEG encoder is
`mjpegenc` (gst's `jpegenc`), and there is no WebP encoder.

There is also a `typefind` element for a byte stream that is not a file
(`srtsrc ! typefind ! …`): it sniffs the flowing bytes and re-declares the caps
mid-stream, so the source's guess is corrected before the data passes on. A
stream it cannot type fails the run instead of flowing on untyped.

**When it doesn't parse, you get a porting hint**, not just an error:

```
$ g2g-launch videotestsrc ! theoraenc ! fakesink
parse error: unknown element: theoraenc
  hint: `theoraenc` has no g2g element: no Theora encoder; use `vpxenc`
        (VP8/VP9) or `av1enc`
```

(`x264enc` itself resolves once `g2g-plugins` is built with the `ffmpeg`
feature, which provides the libx264 software encoder.)

The same guidance is available programmatically via
`g2g_plugins::gst_compat::lint_launch(&registry, line)`.

### Things you may need to change

| Symptom | Why | Fix |
| :--- | :--- | :--- |
| `x264enc` unknown | software H.264 encode is behind the `ffmpeg` feature | build `g2g-plugins` with `--features ffmpeg` (libx264); `nvh264enc`→`nvenc` (NVIDIA); `mfencode` (Windows); or AV1/VP8/VP9 via `av1enc`/`vpxenc` |
| property value has spaces or `!` | needs quoting | wrap it in double **or single** quotes: `filesrc location="/my video.ts"`, `filesrc location='/my video.ts'`, `gstwrap element="x264enc bitrate=4000"` |
| container source won't decode | `bytestream-format` isn't auto-sniffed everywhere | set it explicitly, e.g. `filesrc location=x bytestream-format=mpegts` |
| `autovideosink` etc. | resolved to an available backend | works; resolves Wayland→KMS→fake on Linux |
| `# comment` in a pasted pipeline | supported | a `#` outside quotes runs to end of line and is ignored (handy for multi-line pastes) |
| caps range `width=[1,1920]` / list `format={I420,NV12}` | supported | a range maps to a `Dim::Range` / `Rate::Range`, a list expands to alternatives; negotiation narrows it like GStreamer |

### Launch syntax g2g does not accept

The tokenizer is `gst-launch`-shaped but not identical. These forms fail (with a
hint where possible); rewrite as shown:

| gst-launch form | Status | Do this instead |
| :--- | :--- | :--- |
| caps feature `video/x-raw(memory:GLMemory)` | not supported | drop the feature; g2g picks the memory domain during negotiation (GPU vs system) automatically |
| bins `( videoconvert ! videoscale )` | not supported (no bin/grouping syntax) | flatten the group inline, or build the sub-graph in Rust and `Graph::merge` it |
| demux fan-out on a network source `udpsrc ! qtdemux name=d d.video_0 ! …` | file sources only | demux a local file, or use `uridecodebin` / `playbin` for network multi-stream |

### Equivalence cookbook

Recipes that run verbatim through `g2g-launch` on the baseline `std` registry
(no extra features). Each is exercised by the regression corpus
[g2g-plugins/tests/gst_launch_corpus.rs](g2g-plugins/tests/gst_launch_corpus.rs),
so this list stays honest as the DSL evolves. Swap `gst-launch-1.0` for
`g2g-launch` and the line is unchanged:

| What | gst-launch-1.0 / g2g-launch line |
| :--- | :--- |
| Smoke test | `videotestsrc num-buffers=30 ! videoconvert ! fakesink` |
| Inline caps (format convert) | `videotestsrc ! videoconvert ! video/x-raw,format=NV12 ! fakesink` |
| Caps-driven scale | `videotestsrc ! videoscale ! video/x-raw,width=640,height=480 ! videoconvert ! fakesink` |
| Caps-driven framerate | `videotestsrc ! videorate ! video/x-raw,framerate=15/1 ! fakesink` |
| Enum + numeric props | `videotestsrc ! videoflip method=horizontal-flip ! videobalance saturation=0.5 contrast=1.2 ! videoconvert ! fakesink` |
| Quoted path with a space | `filesrc location="/tmp/my video.ts" ! fakesink` |
| `tee` fan-out (explicit) | `videotestsrc ! tee name=t ! queue ! fakesink t. ! queue ! videoconvert ! fakesink` |
| Audio chain | `audiotestsrc ! volume volume=0.5 ! audioconvert ! audioresample ! fakesink` |

One convenience beyond GStreamer habit: a `tee` is optional. If an element's
output fans out to several branches (`... name=s ! sinkA  s. ! sinkB`) without an
explicit `tee`, g2g splices a broadcast tee in for you (GStreamer would need the
explicit `tee`). Also note `queue`/`queue2` map to a per-edge backpressure policy
rather than a distinct element node. See the negotiated caps
of any line with `g2g-launch -v`, or a Graphviz graph with `--dot`.

---

## 3. Element name mapping

Most names match GStreamer. Differences are handled two ways:

- **Aliases** resolve automatically in the registry (e.g. `autovideosink`,
  `autovideosrc`/`autoaudiosrc`, `avdec_h264` → `ffmpegdec`, `vah264dec` →
  `vaapidec`, `vp8enc`/`vp9enc` → `vpxenc`, `webmmux` → `matroskamux`,
  `adder`/`liveadder` → `audiomixer`, `videomixer` → `compositor`, `rtmp2sink` →
  `rtmpsink`, and the plain decoder names `vp8dec`/`vp9dec`/`mpeg2dec` →
  `ffmpegdec`, `mpg123audiodec`/`flacdec`/`a52dec`/`faad` → `ffmpegaudiodec`).
  See `default_registry` in [g2g-plugins/src/registry.rs](g2g-plugins/src/registry.rs).
  `autovideosrc`/`autoaudiosrc` have no test-source fallback, so a build with no
  capture element leaves the name unresolved rather than producing a test pattern.
- **Look up any gst name**: `g2g-inspect --gst <name>` tells you whether g2g has
  it, renames it, or has no equivalent (with a suggestion):

```sh
g2g-inspect --gst jpegdec        # -> g2g calls it `mjpegdec`
g2g-inspect --gst x264enc        # -> software H.264 encode behind the `ffmpeg` feature
g2g-inspect                      # list every element
g2g-inspect videoconvert         # one element's properties + pad templates
```

Common mappings: `jpegenc`/`jpegdec` → `mjpegenc`/`mjpegdec`; `souphttpsrc` →
`httpsrc`; `rtph264depay` → built into `udpsrc`/`rtspsrc`. `appsrc`/`appsink`
exist as named launch elements (`appsrc channel=<name>` / `appsink
channel=<name>`, the application registers the matching feed/sink before launch),
as programmatic graph nodes, or via the Python host (`pysrc`/`pyelement`). The
table lives in [g2g-plugins/src/gst_compat.rs](g2g-plugins/src/gst_compat.rs)
and is easy to extend.

**One gst element splits in two:** `videoconvert` changes the pixel format and
carries the colorimetry through unchanged, and `colorspace` changes what the
samples *mean* (matrix, range, transfer, primaries) at a fixed pixel format. A
line that needs both chains them. PQ and HLG are refused rather than tone mapped.

Three elements are named after the **ffmpeg filter** rather than a gst element,
having no gst counterpart of their own:

| ffmpeg filter | g2g |
| :--- | :--- |
| `areverse` | `audioreverse` (`chunk-duration=0` buffers the whole stream and reverses it at EOS, the `areverse` behavior) |
| `ebur128` | `ebur128` (momentary / short-term / gated integrated LUFS, read via getters like `level`) |
| `colorspace` | `colorspace`, the colorimetry half of gst's `videoconvert` described above |

Whole plugins answer by family rather than name:

| gst names | g2g |
| :--- | :--- |
| `rtp*pay` / `rtp*depay` | payloading is inside `udpsink` / `udpsrc` (and `rtspsrc`, `webrtcsink`) |
| `rtpbin`, `rtpsession`, `rtpjitterbuffer`, `rtprtx*`, `rtpulpfec*`, `rtpst2022-1-fec*` | `udpsrc` properties (`jitter-latency`, `jitter-depth`, `rtcp-rr-interval`, `nack`, `rtx-payload-type`, `fec-payload-type`, `flexfec-payload-type`) and `udpsink` properties (`rtcp-sr-interval`, `retransmit`, `fec-columns`, `fec-rows`) |
| `gl*` | no OpenGL elements; `wgpusink`, `wgpucompositor`, `dmabuftowgpu` / `wgputodmabuf` |
| `cuda*` | `nvdec` / `nvenc`, and `localcudasrc` / `localcudasink` for cross-process CUDA memory |
| `vulkan*` | `vulkanvideodec` decodes, `wgpusink` presents |
| `va*dec` / `vaapi*dec` | `vaapidec` |
| `nv*dec` / `nv*enc` | `nvdec` / `nvenc` |
| `ladspa*` | no LADSPA host; `volume`, `audiopanorama`, `equalizer-3bands`, `level`, `cutter` |
| effectv (`*tv`) and geometrictransform (`bulge`, `fisheye`, ...) | no effects plugins; `videoflip`, `videocrop`, `videobox`, `videoscale` are the geometry elements |
| `qml*` / `gtk*` sinks | no toolkit sinks; render with `wgpusink` or pull frames with `appsink` |
| `decklink*` | no DeckLink support |

Registered names and the exact table always win over a family rule, so `nvdec`
and `vaapidec` still answer for themselves.

### SRTP gaps

GStreamer's [`srtpenc`](https://gstreamer.freedesktop.org/documentation/srtp/srtpenc.html)
and [`srtpdec`](https://gstreamer.freedesktop.org/documentation/srtp/srtpdec.html)
are host pipeline elements backed by libsrtp. g2g's `srtp` feature is a pure
Rust, Sans-IO RFC 3711 / RFC 7714 packet layer plus the `srtpenc` / `srtpdec`
elements, and `dtls-srtp` adds `dtlssrtpenc` / `dtlssrtpdec`. It runs on
`no_std + alloc` targets. `rtp-cipher`, `rtcp-cipher`, `rtp-auth` and
`rtcp-auth` take gst's values; left unset, the cipher follows the key length
(28 or 44 bytes GCM, 30 or 46 counter mode). Every leg below is validated
against gst's libsrtp and `dtls` elements on a host that has them.

| Area | GStreamer | g2g | Status |
| :--- | :--- | :--- | :--- |
| RFC 7714 profiles | AES-128-GCM and AES-256-GCM | Both profiles, full 16-byte tags | Complete |
| RTP / SRTCP processing | RTP and RTCP element pads | Raw packet methods, including encrypted and authentication-only SRTCP | Complete |
| Stream contexts | Encoder pads share one SSRC. The decoder creates contexts per SSRC | `srtpenc` takes its SSRC from the first packet. `srtpdec` creates a context per SSRC | Complete |
| Pipeline integration | `srtpenc` and `srtpdec` elements with RTP and RTCP pads | `srtpenc` and `srtpdec` elements, one flow each (RTP or RTCP by caps) | Complete |
| Key delivery | Key property and decoder key-request signal | `key` property, or an `SrtpKeyProvider` on `srtpdec` | Complete |
| Initial ROC | Decoder caps can supply the current rollover counter | `roc` property, or the provider's keying material | Complete |
| Rekeying and limits | Soft-limit and hard-limit signals replace exhausted keys | `replace_key` or a new `key` at runtime. The soft limit posts a bus `Info`, the hard limit stops the stream | Complete |
| Replay policy | Configurable window, default 128 packets | `replay-window-size`, 64 to 32768, default 128 | Complete |
| Repeated transmission | Optional repeated transmission of an identical RTP packet | `allow-repeat-tx`, off by default | Complete |
| MKI | One send key and up to 15 receive keys selected by MKI | `mki` on `srtpenc`. `srtpdec` selects among the keys its provider returns | Complete |
| Statistics | Per-stream receive, drop, and protection statistics | `stats()` on both elements: packet counts and each stream's rollover counter | Complete |
| DTLS-SRTP | Separate DTLS-SRTP encoder and decoder elements deliver keys | `dtlssrtpenc` / `dtlssrtpdec` on the pure-Rust `dimpl` stack, paired by `connection-id`, `peer-fingerprint` pins the peer | Complete |
| Legacy profiles | AES-ICM, HMAC-SHA1, and NULL modes | The same cipher and auth values on `rtp-cipher` / `rtcp-cipher` / `rtp-auth` / `rtcp-auth` | Complete |

Not carried over: gst's `key` / `srtp-cipher` overrides that turn DTLS off on
`dtlssrtpenc` / `dtlssrtpdec` (use `srtpenc` / `srtpdec` for a fixed key),
`random-key`, and the `stats` property (`stats()` is programmatic).

### STANAG 4609 / KLV metadata: beyond parity

GStreamer carries KLV as opaque `meta/x-klv` buffers: `tsdemux` exposes the
pad, `mpegtsmux` writes the asynchronous form only (0x06 + `KLVA`
registration), `rtpklvpay`/`rtpklvdepay` move it over RTP, and everything about
the *content* (MISB ST 0601 telemetry, ST 0102 security markings, checksums) is
left to the application or to commercial third-party plugins. g2g handles the
content in-tree:

| Need | GStreamer | g2g |
| :--- | :--- | :--- |
| Demux KLV from a TS | `tsdemux` (opaque pad) | `tsdemux stream=klv` (`Caps::Klv`) |
| Mux KLV, async | `mpegtsmux` | `mpegtsmux` |
| Mux KLV, strict sync (ST 1402: 0x15, AU cells, metadata descriptor) | — | `mpegtsmux klv-sync=true` |
| KLV over RTP (RFC 6597) | `rtpklvpay` / `rtpklvdepay` | `rtpklv` packetizer / depayloader |
| Decode ST 0601 telemetry + ST 0102 security set | app code or commercial addon | `klvdecode` / `UasDatalink` |
| Build ST 0601 sets | app code or commercial addon | `UasDatalink::encode` |
| ST 0903 VMTI moving targets, ST 1204 MIIS id | — | `vmti`, `UasDatalink::miis_core_id` |
| Detector output as VMTI targets | — | `vmti_from_analytics` |
| ST 0604 MISP timestamps in SEI | — | `misptimeinsert` / `misptimeextract` |
| SMPTE 2022-1 FEC for TS over RTP | `rtpst2022-1-fecenc` / `fecdec` | `st2022fec` |
| Telemetry to a TAK / ATAK network (CoT) | — | `cotsink` |

The codec is validated against ffmpeg (bit-exact both directions), the
independent klvdata implementation, the published MISMMS reference packet, and
a real UAS capture, so a STANAG pipeline that needs a commercial GStreamer
addon ports to stock g2g.

---

## 4. Caps

GStreamer caps strings parse to the typed `Caps` enum and back:

- **string → `Caps`**: `g2g_plugins::capsfilter::parse_caps("video/x-raw,format=NV12,width=640,height=480,framerate=30/1")`
  ([g2g-plugins/src/capsfilter.rs](g2g-plugins/src/capsfilter.rs)). Media types:
  `video/x-raw`, `video/x-h264`/`h265`/`vp8`/`vp9`/`av1`, `image/jpeg`,
  `audio/x-raw`, `audio/x-opus`, `audio/mpeg` (AAC). Format names are
  case-insensitive (`NV12` or `nv12`). A `video/x-raw` with no `format` expands
  to all raw formats and is narrowed at negotiation.
- **`Caps` → string**: `caps.to_gst_string()` ([g2g-core/src/caps.rs](g2g-core/src/caps.rs))
  for logs and diagnostics. It round-trips through the parser.

In a pipeline, an inline caps filter works exactly like GStreamer:
`... ! videoscale ! video/x-raw,width=1280,height=720 ! ...`. g2g's caps-driven
transforms (`videoscale`, `videoconvert`) read their target from a downstream
capsfilter when their own properties are unset, the gst idiom.

---

## 5. Porting application code

A C/Python/Rust GStreamer app maps onto g2g's typed graph:

| GStreamer | glass2glass |
| :--- | :--- |
| `gst_parse_launch(str)` | `parse_launch(&registry, str)` → `Graph` |
| build `GstPipeline` by hand | `Graph::new()` + `add_source`/`add_transform`/`add_sink`/`add_tee`/`add_muxer` + `link` |
| `gst_element_factory_make("x", ...)` | construct the Rust element (`VideoConvert::new()`, ...) or `registry.make_element("x")` |
| `g_object_set(el, "prop", v)` | the element's `with_*` builder, or `set_property("prop", PropValue::...)` |
| `gst_element_set_state(PLAYING)` + main loop | `run_graph(graph, &clock, link_capacity).await` |
| `GstBus` watch | a `Bus` passed to the run, yielding `BusMessage` |
| pipeline clock | a `PipelineClock` (e.g. `WallClock`) passed to `run_graph` |
| `queue` for latency control | per-edge `LinkPolicy` + `link_capacity` (the latency floor is `2 * link_capacity * frame_period`) |

The programmatic path is fully typed — you hold the element values, not opaque
`GstElement*`. See `run_graph` in [g2g-core/src/runtime/graph_runner.rs](g2g-core/src/runtime/graph_runner.rs).

Worked, runnable side-by-side examples of this text-to-typed mapping (a transform
chain, an inline caps filter, a `tee` fan-out), each run both ways and asserted
equivalent, are in
[g2g-plugins/examples/gst_equivalents.rs](g2g-plugins/examples/gst_equivalents.rs):

```sh
cargo run -p g2g-plugins --features std --example gst_equivalents
```

### 5.1 Dynamic pipelines (the hardest port)

A GStreamer *application* is rarely a static line: it adds and removes branches at
runtime, blocks pads, relinks on `pad-added`, and pushes/pulls buffers from app
code. g2g reaches the same outcomes with different primitives, because Rust
ownership forbids GObject's reference-cycle + signal-callback shape. The full map
is DESIGN.md §4.9; the patterns an app developer hits most:

| GStreamer idiom | glass2glass |
| :--- | :--- |
| `appsrc` `need-data`/`push-buffer` | `appsrc channel=<name>` + `register_appsrc` → `AppSrcFeed::push`, or `g2g-bridge`'s `BridgeGraph` for a whole embedded sub-graph |
| `appsink` `new-sample`/pull | `appsink channel=<name>` + `set_appsink_callback` (callback) or `register_appsink_pull` (pull) |
| `pad-added` relink (decodebin) | bounded dynamic pads: `decodebin`/`uridecodebin` auto-plug, or `StreamDemux` / `register_demux` with N typed output ports ("dark slots" populated on discovery) |
| `gst_pad_add_probe(BUFFER)` | a `LinkInterceptor` registered on a slot (the probe analog) |
| `gst_pad_add_probe(BLOCK)` / `pad_idle`, then relink | no block to install: name the position instead (§5.2) |
| add / remove a branch at runtime | runtime fan-out via `DynamicFanoutHandle::add_branch`, fan-in via `DynamicFaninHandle`; a swappable sub-graph is a `BranchSlot` |
| enable/disable a branch, A/B switch | `Router` + `Gate` (and their `RouterHandle` / `GateHandle`) |
| element hot-swap | `ElementSlot::swap` (ArcSwap; no use-after-free with a frame in flight) |
| flushing seek | `PipelinePacket::Flush` (the runner drains and resets) |
| child→parent signal/notify | post a `BusMessage`; the parent reads it (no back-reference) |

Two ownership-driven differences to expect: relinking is **moving the receive end
of a channel under a brief gate hold** (explicit ownership transfer, not pointer
surgery), and runtime-growable pad counts beyond a fixed N use a `Slab<Slot>` in
the dynamic layer rather than unbounded GObject pads. The payoff is that the
hot-swap and pad-block choreography that is famously race-prone in GStreamer is
memory-safe here by construction. Boundary-aligned switches (bitrate / codec
change at a segment or keyframe) are part of the §4.9 design surface; check what
is wired today before relying on them.

### 5.2 Pad blocking: changing a graph that is running

The GStreamer recipe for changing a live pipeline is `gst_pad_add_probe` with
`GST_PAD_PROBE_TYPE_BLOCK`, `gst_element_unlink` / `gst_element_link` inside the
callback, then removing the probe. The block buys the quiet moment; blocking the
wrong pad, losing the buffer already in flight, or racing EOS is where dynamic
pipelines earn their reputation.

g2g has no pad to block, so there is no probe to install. The quiet moment is the
runner's, taken between packets on the producing side of the edge you name, and
which handle you want depends on what is changing:

| What you are doing | GStreamer | glass2glass |
| :--- | :--- | :--- |
| Swap one element for another in place | block, unlink, add, link, unblock | `ElementSlot::swap` (an atomic store, no block) |
| Add or drop an element on an edge | block, unlink, add, two links, unblock | `GraphMutator::insert_after` / `insert_before` / `remove` |
| Swap the source or the sink of a running pipeline | block the pad, unlink, set the old element to `NULL`, add, link, unblock | `GraphMutator::replace_source` / `replace_sink` |
| Add or drop a whole branch | block the tee pad, request / release a pad | `DynamicFanoutHandle::add_branch`, `DynamicFaninHandle::add_input` |
| Watch or drop buffers in passing | `GST_PAD_PROBE_TYPE_BUFFER` | a `LinkInterceptor` on the edge's probe slot |

[`GraphMutator`](g2g-core/src/runtime/mutate.rs) is the direct analog of the
block-and-relink dance, from `run_graph_mutable` / `run_graph_threaded_mutable`
alongside the run future: `insert_after("dec", element)` splices onto the edge
below `dec`, `insert_before("sink", element)` onto the edge above `sink`, and
`remove("videoflip0")` lifts the element back out and hands it to
you. The ordering is the runner's problem rather than yours. An insert needs no
drain, because the new element is given the very link its producer was pushing
to, so whatever is queued stays ahead of anything the new element emits; a remove
closes the element's input and lets it drain through before the bypass takes
effect. Caps consent is settled before anything moves, and a refusal leaves the
graph running unchanged, which is the case a mis-ordered `unlink` / `link` turns
into a stall or a crash. The protocol is DESIGN.md §4.8.6; a working demo that
splices a `videoflip` in and out of a live RTSP window is
[examples/g2g-mutate-demo](examples/g2g-mutate-demo).

Know the scope before you port to it: a transform position on a 1:1 edge carrying
one stream. The producing end is a source, a transform, or one output of a tee or
demux. The consuming end is a transform, a sink, or one input pad of a muxer or
terminal fan-in. The structural nodes themselves are not splice points
(`MutationError::NotMutable`), so reach for the fan-out / fan-in handles to add a
whole branch.

The two ends of the graph are not splice points either, but the element on one is
swappable: `replace_source("src", element)` and `replace_sink("sink", element)`
hand the old one back and return the name the replacement got. The old sink is
given the frames still queued for it and an end of stream, so it finalizes before
it comes back and the replacement starts only after that; a replacement source
opens with a segment that continues the running time its predecessor reached, so
it can stamp from its own zero. Neither joins the clock election or the latency
fold the run settled at startup.

An operation names whichever end of the edge is unique: `insert_after` takes the
one edge below a node, which is how a producer feeding a muxer pad is addressed,
and `insert_before` takes the one edge above a node, which is how a tee or demux
branch is addressed by its consumer. A node with several edges on the side you
asked for is `NotMutable` there and is addressed from the other side.

A removed element is drained *and* flushed, so the frames it was holding
internally reach the consumer too and a reordering position is as removable as a
stateless one. Each operation lands at the producer's next packet boundary, which
means a source that has gone quiet defers it rather than failing it.

---

## 6. Porting a custom element

This section maps GStreamer's base classes and vmethods onto g2g's traits.
[AUTHORING.md](AUTHORING.md) covers the same traits for a reader with no
GStreamer background, and goes further into the lifecycle, properties,
`no_std` and registration.

A GStreamer base-class subclass becomes a Rust trait impl:

| GStreamer base | g2g trait | File |
| :--- | :--- | :--- |
| `GstBaseTransform` / `GstBaseSink` | `AsyncElement` | [g2g-core/src/element.rs](g2g-core/src/element.rs) |
| `GstBaseSrc` / `GstPushSrc` | `SourceLoop` | [g2g-core/src/runtime/runner.rs](g2g-core/src/runtime/runner.rs) |
| `GstAggregator` (N-in/1-out) | `MultiInputElement` + the `InputAggregator` helper | [g2g-core/src/fanout.rs](g2g-core/src/fanout.rs), [g2g-core/src/aggregator.rs](g2g-core/src/aggregator.rs) |

Method mapping (transform):

| GStreamer vmethod | `AsyncElement` method |
| :--- | :--- |
| `set_caps` / caps query | `intercept_caps` (negotiate) |
| `start` / pool setup | `configure_pipeline` (fixed caps in) |
| `transform` / `transform_ip` / `render` | `process(packet, out)` (async) |
| `g_object_class_install_property` | `properties()` + `set_property`/`get_property` |
| `gst_element_class_set_metadata` | `metadata()` |
| pad templates | `PadTemplates::pad_templates()` |

Caps refinement that GStreamer pushes as a `GST_EVENT_CAPS` is a
`PipelinePacket::CapsChanged` you emit before the affected `DataFrame`. EOS is
emitted by the runner for multi-input/source ends — a transform must **not**
forward `Eos`.

A complete, runnable example (a registered third-party transform used by name in
a launch line) is at
[g2g-plugins/examples/third_party_element.rs](g2g-plugins/examples/third_party_element.rs):

```sh
cargo run -p g2g-plugins --features std --example third_party_element
```

If your element is written in **Python**, you don't port it to Rust at all — host
it via `pyelement` / `pysrc` / `pyaggregator` (see the `g2g-python` crate).

---

## 7. Adding third-party elements / plugins

g2g has **no dynamic `.so` plugin scanning** like GStreamer. There are three
regimes depending on how g2g is consumed:

### a) As a library (you build the app) — today, the primary path

Your crate depends on `g2g-core` (and `g2g-plugins`), implements the element
trait, and exposes a registration function by convention:

```rust
pub fn register(registry: &mut g2g_core::runtime::Registry) {
    registry.register_launch(LaunchFactory::of::<MyElement>("myelement", || Box::new(MyElement::new())));
    // register_source / register_muxer / register (ElementFactory, for autoplug) likewise
}
```

The app composes registries: `let mut reg = default_registry(); my_crate::register(&mut reg); other::register(&mut reg);`.
This is exactly what `g2g_python::register` does. That *is* the plugin system: a
crate + one call. (Programmatic graphs need no registry at all — just construct
and `add_transform` the element value.)

### b) Against a system-installed g2g (no recompile) — use the Python host

When g2g ships as a packaged binary you can't recompile, the supported
no-recompile extension path **today** is the **Python host**: drop a Python
module on the path and reference it by name —
`... ! pyelement module=my_mod class=MyTransform ! ...` (also `pysrc`,
`pyaggregator`). This is the gst-python analog and needs no Rust build. It
requires a g2g built with the `python` feature.

### c) Native Rust plugins in a packaged binary (dynamic `.so`, M201)

Build a plugin with plain `cargo` against the published `g2g-core` + `g2g-plugin`
(the `g2g-devel` equivalent), drop the resulting `.so` where the installed
`g2g-launch` scans, no recompile of g2g:

```toml
# my-plugin/Cargo.toml
[lib]
crate-type = ["cdylib"]
[dependencies]
g2g-core   = "0.x"   # element traits
g2g-plugin = "0.x"   # the declare_plugin! macro
```
```rust
// my-plugin/src/lib.rs: implement AsyncElement + PadTemplates for MyFilter, then:
g2g_plugin::declare_plugin! {
    elements: [ ("myfilter", MyFilter, || Box::new(MyFilter::default())) ]
}
```

`cargo build --release` produces `libmy_plugin.so`. A `g2g-launch` built with the
`plugin-loader` feature loads it via `--plugin <path>` (repeatable) or by
directory from `$G2G_PLUGIN_PATH` (`:`-separated), then resolves the element by
name: `g2g-launch --plugin libmy_plugin.so ... ! myfilter ! ...`. `g2g-inspect`
loads plugins the same way so their elements list. A complete, buildable example
is `g2g-plugins/tests/fixtures/example-plugin`.

**Signatures.** A host built with `plugin-signing` and given Ed25519 public keys,
through `$G2G_PLUGIN_TRUSTED_KEYS` (`:`-separated key files) or
`g2g-inspect --trusted-key <path>`, loads only plugins carrying a matching
`<plugin>.sig`, checked before the `dlopen`. Produce the keys and signatures with
`g2g-plugin-sign keygen | sign | verify`. With no keys configured nothing is
verified, which is the default.

**ABI lock.** Rust has no stable ABI, so a plugin and the host must share the
same `g2g-core` version, the same `rustc`, and the same layout-affecting features
(`metadata`, `multi-thread`). The plugin embeds an ABI tag
(`g2g_core::ABI_VERSION`) that folds all three together; the loader compares it
and refuses a mismatch with a clear error rather than risk UB. To cross
toolchains, declare the plugin with `g2g_plugin::declare_plugin_v2!` instead: it
emits a frozen `repr(C)` descriptor, so the host may have been built by a
different `rustc`, and the plugin may be written in C. Regime (a) — including the
**package-rebuild** path, where a vendor compiles extra element crates into the
g2g binary it ships — remains available and needs no ABI match.

### d) Hosting an *un-ported* GStreamer element (`gstwrap`)

The three regimes above register a g2g-native element. When a stage has no g2g
port yet (a proprietary GStreamer element, one you have not gotten to), you don't
have to block the migration: `gstwrap` runs the real GStreamer element *inside*
your g2g graph. This is the mirror of `g2g-bridge` (§8, which embeds g2g inside a
GStreamer app); `gstwrap` embeds GStreamer inside g2g, so you can adopt g2g as
the top-level framework now and port the remaining stages later.

```rust
// videotestsrc ! gstwrap element="videoflip method=horizontal-flip" ! autovideosink
graph.add_transform({
    let mut w = GstWrap::new();
    w.set_property("element", PropValue::Str("videoflip method=horizontal-flip".into()))?;
    w                       // a caps-preserving element declares nothing
});
// A reformatting element (encoder, scaler) declares its result:
//   w.set_property("element",     PropValue::Str("x264enc bitrate=4000".into()))?;
//   w.set_property("output-caps", PropValue::Str("video/x-h264,...".into()))?;
```

It drives `appsrc ! <element> ! appsink` in a real GStreamer pipeline internally;
system-memory frames flow in and out (a copy each way in v1). Built behind the
`gstreamer` feature (needs the gstreamer-1.0 + gstreamer-app-1.0 dev packages).
It works from `g2g-launch` too, since the launch tokenizer is quote-aware:

```sh
g2g-launch 'videotestsrc ! gstwrap element="videoflip method=horizontal-flip" ! fakesink'
```

---

## 8. Known gaps

The full outstanding list is DESIGN_TODO.md; this is what a port is most likely
to hit.

- **Platform coverage.** Linux and Windows are the primary targets. Android
  (MediaCodec decode/encode, Camera2, AAudio, Surface present, plus ML inference)
  is device-validated. macOS has VideoToolbox decode/encode, AVFoundation camera
  and mic capture, ScreenCaptureKit, Core Audio, and a Metal present sink, but
  the capture elements are only probe-validated: the CI Mac grants neither the
  camera nor the screen-recording permission. The cross-platform software path
  (parsers, container mux/demux, SW transforms, ffmpeg, `gst-launch` DSL) works
  everywhere.
- **Transport.** SRT (TSBPD/AES/key-rotation), RTP with RTCP, FEC (ULPFEC,
  FlexFEC) and RTX, RTMP ingest and egress, an RTSP server, and WebRTC
  (WHIP/WHEP) are in. Open: RTP over QUIC (waiting on the RFC and an assigned
  ALPN), several NetStreams over one RTMP connection (declined, it needs a
  runtime-arity fan-out the fixed-arity model has no room for), WebRTC FEC
  (str0m carries no FEC payload, so loss recovery is NACK/RTX only), and a run
  against LiveKit Cloud over a real TURN relay.
- `playbin uri=hls://...` probes a master playlist and fans the variant out to its
  elementary streams, but there is no DASH URI handler: point `dashsrc` at the
  MPD and wire the demuxer and decoders explicitly.
- Native dynamic-plugin loading (§7c) has two ABIs. The `declare_plugin!` path
  needs plugin and host to share a `g2g-core` version, a `rustc`, and the
  layout-affecting features; the frozen C ABI v2 loads across toolchains and from
  plain C. Still open: how a distribution supplies `g2g-core` to an offline
  plugin build.
- `g2g-bridge` (embed a g2g sub-graph inside a GStreamer pipeline for incremental
  migration, DESIGN.md §7) is in: the GObject shell (`libgstglass2glass.so`, the
  `gstreamer` feature) registers a real `glass2glass` GStreamer element, so a
  stock `gst-launch` line embeds a g2g sub-graph by name:
  `... ! glass2glass fragment="videoflip method=horizontal-flip" ! ...`. A
  caps/size-preserving fragment runs in place; a rescaling / reformatting one
  declares its result with `output-caps`, e.g.
  `glass2glass fragment=videoscale output-caps="video/x-raw,format=RGBA,width=640,height=360,framerate=30/1"`.
  Build and validate with `tools/gst-bridge-smoke.sh` (needs host GStreamer dev
  libs). A dma-buf-backed `GstBuffer` passes through zero-copy
  (`tools/gst-bridge-dmabuf-smoke.sh`); system memory is mapped and copied. The
  gap: a GPU-*compute* fragment (`dmabuftowgpu ! <compute>`) still needs a
  download or dma-buf-export element at its tail to return the GPU result to the
  shell.
- `gstwrap` (§7d), the reverse bridge that hosts an un-ported GStreamer element
  *inside* a g2g graph, is system memory only, a copy each way. dma-buf zero-copy
  through it is future work.

---

## 9. CLI quick reference

```sh
g2g-launch [-v] <pipeline>        # run a gst-launch-style line (-v: per-link negotiated caps)
g2g-inspect                       # list elements
g2g-inspect <element>             # one element's role, properties, pad templates
g2g-inspect --all                 # full catalog
g2g-inspect --gst <gst-name>      # map a GStreamer element name to g2g
g2g-inspect --gst-map             # every gst-name/g2g-name pair, tab separated
g2g-discover <file> [--json]      # container, streams, caps, duration and tags
```
