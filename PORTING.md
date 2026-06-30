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
(`src ! m.  src ! m.  funnel name=m ! sink`), inline caps filters
(`! video/x-raw,format=NV12,width=640 !`), `queue`/`queue2` (mapped to a per-edge
backpressure policy), `decodebin` / `uridecodebin` / `playbin`.

**When it doesn't parse, you get a porting hint**, not just an error:

```
$ g2g-launch videotestsrc ! x264enc ! fakesink
parse error: unknown element: x264enc
  hint: `x264enc` has no g2g element: no software H.264 encoder; use `mfencode`
        (Windows), or encode AV1/VP8/VP9 with `av1enc`/`vpxenc`
```

The same guidance is available programmatically via
`g2g_plugins::gst_compat::lint_launch(&registry, line)`.

### Things you may need to change

| Symptom | Why | Fix |
| :--- | :--- | :--- |
| `x264enc` / `nvh264enc` unknown | no SW/NVENC H.264 encoder | `mfencode` (Windows) or `av1enc` / `vpxenc`; `g2g-inspect --gst x264enc` |
| `FanOutWithoutTee` | g2g doesn't auto-insert a tee | add an explicit `tee` before the branches |
| property with spaces fails | v1 parser has no quoted-value spaces | avoid spaces in property values |
| container source won't decode | `bytestream-format` isn't auto-sniffed everywhere | set it explicitly, e.g. `filesrc location=x bytestream-format=mpegts` |
| `autovideosink` etc. | resolved to an available backend | works; resolves Wayland→KMS→fake on Linux |

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
g2g-inspect --gst appsink        # -> no g2g element; use a programmatic sink / pyelement
g2g-inspect                      # list every element
g2g-inspect videoconvert         # one element's properties + pad templates
```

Common mappings: `jpegenc`/`jpegdec` → `mjpegenc`/`mjpegdec`; `souphttpsrc` →
`httpsrc`; `rtph264depay` → built into `udpsrc`/`rtspsrc`; `appsrc`/`appsink` →
programmatic graph nodes or the Python host (`pysrc`/`pyelement`). The table
lives in [g2g-plugins/src/gst_compat.rs](g2g-plugins/src/gst_compat.rs) and is
easy to extend.

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

---

## 8. Known gaps (as of M217)

- **Platform coverage: Linux and Windows only.** macOS (VideoToolbox /
  AVFoundation / Core Audio / Metal) and Android (MediaCodec / Camera2 / AAudio /
  Surface) have no elements yet, so a pipeline that relies on platform-native
  decode / capture / present will not build on those targets. The cross-platform
  software path (parsers, container mux/demux, SW transforms, ffmpeg, `gst-launch`
  DSL) works everywhere. See DESIGN_TODO.md "Gap analysis to 80% parity".
- Other structural / transport gaps (allocation re-cascade, RTMP/SRT egress,
  RTSP server, RTP RTX/FEC) are catalogued in DESIGN_TODO.md.
- No quoted property values with spaces in the launch DSL (v1).
- No auto-plug through fan-out demuxers (chunked HLS/DASH manifests); demux/select explicitly.
- Native dynamic-plugin loading (§7c, M201) is version+toolchain-locked: a plugin
  and host must share the same `g2g-core` version, `rustc`, and layout features.
  Cross-toolchain binary plugins (an `abi_stable` facade) are future work.
- `g2g-bridge` (embed a g2g sub-graph inside a GStreamer pipeline for incremental
  migration, DESIGN.md §7): the impedance core (`BridgeGraph`, which runs
  `appsrc ! <fragment> ! appsink` on its own thread behind a synchronous
  push/pull API) is implemented and tested; the GObject `GstBaseTransform` shell
  (`libgstglass2glass.so`) that registers `glass2glass` as a loadable GStreamer
  element is still to come.
- `g2g-launch -v` reports wiring but not yet per-pad negotiated caps.

---

## 9. CLI quick reference

```sh
g2g-launch [-v] <pipeline>        # run a gst-launch-style line (-v: dump wiring)
g2g-inspect                       # list elements
g2g-inspect <element>             # one element's role, properties, pad templates
g2g-inspect --all                 # full catalog
g2g-inspect --gst <gst-name>      # map a GStreamer element name to g2g
```
