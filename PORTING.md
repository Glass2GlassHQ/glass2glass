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

**Typefind.** GStreamer's `filesrc` emits untyped bytes and a downstream
`typefind` sniffs the media type at runtime. g2g negotiates types statically, so a
byte source must announce its type up front, but you rarely name it by hand: a bare
`filesrc location=X` derives its type from the extension (`.mp4`/`.mkv`/`.ts`/
`.ogg`/`.flv` containers, `.vtt`/`.srt`/`.ass`/`.ttml` subtitles), so
`filesrc location=subs.vtt ! subparse` and `filesrc location=movie.mkv !
matroskademux name=d ...` run with no hint. For a mis-named or extensionless file,
`bytestream-format=auto` sniffs the header content instead (containers by magic,
subtitles by their signature). An explicit `bytestream-format=` always overrides.
Caveat: `filesrc ! decodebin` on a *progressive* single-file `.mp4` needs
`uridecodebin uri=file://…` instead (the `decodebin` autoplug pool has only the
fragmented MP4 demuxer); MKV/TS/FLV/Ogg decode fine through `decodebin`.

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
| property value has spaces or `!` | needs quoting | wrap it in double quotes: `filesrc location="/my video.ts"`, `gstwrap element="x264enc bitrate=4000"` |
| container source won't decode | `bytestream-format` isn't auto-sniffed everywhere | set it explicitly, e.g. `filesrc location=x bytestream-format=mpegts` |
| `autovideosink` etc. | resolved to an available backend | works; resolves Wayland→KMS→fake on Linux |

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
  `avdec_h264` → `ffmpegdec`, `vah264dec` → `vaapidec`, `vp8enc`/`vp9enc` →
  `vpxenc`). See `default_registry` in [g2g-plugins/src/registry.rs](g2g-plugins/src/registry.rs).
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
| `gst_pad_add_probe(BLOCK)` / `pad_idle` | a `LinkInterceptor` registered on a slot (the probe analog) |
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

---

## 6. Porting a custom element

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

**ABI lock.** Rust has no stable ABI, so a plugin and the host must share the
same `g2g-core` version, the same `rustc`, and the same layout-affecting features
(`metadata`, `multi-thread`). The plugin embeds an ABI tag
(`g2g_core::ABI_VERSION`) that folds all three together; the loader compares it
and refuses a mismatch with a clear error rather than risk UB. (A future
`abi_stable` facade would relax the same-toolchain requirement; a C-ABI shim was
rejected as it loses the ergonomic Rust trait.) Regime (a) — including the
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

## 8. Known gaps (as of M459)

- **Platform coverage.** Linux and Windows are the primary targets. Android
  (MediaCodec decode/encode, Camera2, AAudio, Surface present, plus ML inference)
  is device-validated. macOS (VideoToolbox / AVFoundation / Core Audio / Metal)
  is started but not yet built on Apple hardware, so treat it as unverified. The
  cross-platform software path (parsers, container mux/demux, SW transforms,
  ffmpeg, `gst-launch` DSL) works everywhere. See DESIGN_TODO.md "Gap analysis".
- Transport: SRT (TSBPD/AES/key-rotation), RTP with RTCP and FEC (ULPFEC,
  FlexFEC), and WebRTC (WHIP/WHEP) are in; still open are RTMP egress, an RTSP
  *server*, and RTP RTX. Catalogued in DESIGN_TODO.md.
- Other structural gaps (e.g. allocation re-cascade) are catalogued in
  DESIGN_TODO.md.
- No auto-plug through fan-out demuxers (chunked HLS/DASH manifests); demux/select explicitly.
- Native dynamic-plugin loading (§7c, M201) is version+toolchain-locked: a plugin
  and host must share the same `g2g-core` version, `rustc`, and layout features.
  Cross-toolchain binary plugins (an `abi_stable` facade) are future work.
- `g2g-bridge` (embed a g2g sub-graph inside a GStreamer pipeline for incremental
  migration, DESIGN.md §7): both layers exist. The impedance core (`BridgeGraph`)
  runs `appsrc ! <fragment> ! appsink` on its own thread behind a synchronous
  push/pull API; the GObject shell (`libgstglass2glass.so`, the `gstreamer`
  feature) registers a real `glass2glass` GStreamer element, so a stock
  `gst-launch` line can embed a g2g sub-graph by name:
  `... ! glass2glass fragment="videoflip method=horizontal-flip" ! ...`. A
  caps/size-preserving fragment runs in place; a rescaling / reformatting one
  declares its result with `output-caps`, e.g.
  `glass2glass fragment=videoscale output-caps="video/x-raw,format=RGBA,width=640,height=360,framerate=30/1"`.
  Build and validate with `tools/gst-bridge-smoke.sh` (needs host GStreamer dev
  libs). A dma-buf-backed `GstBuffer` is imported and handed back zero-copy (the
  shell detects `GstDmaBufMemory` on input and re-wraps a dma-buf output;
  `tools/gst-bridge-dmabuf-smoke.sh`); system-memory buffers are mapped and
  copied. A GPU-*compute* fragment (`dmabuftowgpu ! <compute>`) still needs a
  download/export element at its tail to return the GPU result to the shell.
- `gstwrap` (§7d) is the reverse bridge: it hosts an un-ported GStreamer element
  *inside* a g2g graph (`appsrc ! <element> ! appsink` on GStreamer's own
  threads), so g2g can be the top-level framework during a migration. Behind the
  `gstreamer` feature; v1 is system-memory (a copy each way), dma-buf zero-copy
  through it is future work. Usable from `g2g-launch` (the tokenizer is
  quote-aware, so `gstwrap element="x264enc bitrate=4000"` parses).

---

## 9. CLI quick reference

```sh
g2g-launch [-v] <pipeline>        # run a gst-launch-style line (-v: per-link negotiated caps)
g2g-inspect                       # list elements
g2g-inspect <element>             # one element's role, properties, pad templates
g2g-inspect --all                 # full catalog
g2g-inspect --gst <gst-name>      # map a GStreamer element name to g2g
```
