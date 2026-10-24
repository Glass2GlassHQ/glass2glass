# Writing a glass2glass element

How to write a new **g2g** element from scratch: which trait to implement, what
the runner calls and in what order, and what a launch line needs before it can
name your element. It assumes Rust and some multimedia vocabulary, not
GStreamer. If you *do* know GStreamer, [PORTING.md §6](PORTING.md#6-porting-a-custom-element)
maps its base classes and vmethods onto the traits below. This guide covers the
same ground without the mapping.

An element here is a Rust type implementing one trait. There is no base class to
subclass, no GObject registration, and no `.so` to install: an element in a
downstream crate is registered with one function call
([§10](#10-registering-it-so-a-launch-line-can-use-it)).

Commands marked *(clone)* need a checkout of this repository. Everything else
works from the published crates.

---

## 1. A complete element

A transform that inverts the colour channels of RGBA video, registered under the
name `invert` and run from a launch line. Start a binary crate
(`cargo new invert-demo`) with these dependencies:

```toml
# Cargo.toml
[dependencies]
g2g-core    = { version = "0.10", features = ["std", "runtime"] }
g2g-plugins = { version = "0.10", features = ["std"] }
tokio       = { version = "1", features = ["macros", "rt"] }
```

`g2g-core` builds the heap-free MCU subset by default. `runtime` adds
`PadTemplates` and the runners, and `std` adds the `dyn` element trait, the
`Registry`, `parse_launch` and `run_graph`. `g2g-plugins` with `std` turns both
on as well, so listing them matters for a crate that ships only elements and
depends on `g2g-core` alone.

```rust
// src/main.rs
use g2g_core::element::BoxFuture;
use g2g_core::pad_template::{PadTemplate, PadTemplates};
use g2g_core::runtime::{parse_launch, run_graph, LatencyProfile, LaunchFactory, Registry};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, Colorimetry, ConfigureOutcome, Dim, DomainSet,
    G2gError, Interlace, MemoryDomain, MemoryDomainKind, OutputSink, PipelinePacket, Rate,
    RawVideoFormat,
};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;

#[derive(Debug, Default)]
pub struct Invert;

fn rgba_any_size() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        interlace: Interlace::Any,
        colorimetry: Colorimetry::UNKNOWN,
    }
}

impl AsyncElement for Invert {
    type ProcessFuture<'a> = BoxFuture<'a, Result<(), G2gError>>;

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(CapsSet::one(rgba_any_size()))
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        rgba_any_size().intersect(upstream_caps)
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(mut frame) => {
                    let MemoryDomain::System(bytes) = &mut frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    for pixel in bytes.as_mut_slice().chunks_exact_mut(4) {
                        for channel in &mut pixel[..3] {
                            *channel = 255 - *channel;
                        }
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                // the runner sends the single end-of-stream downstream itself
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for Invert {
    fn pad_templates() -> Vec<PadTemplate> {
        let rgba = CapsSet::one(rgba_any_size());
        Vec::from([PadTemplate::sink(rgba.clone()), PadTemplate::source(rgba)])
    }
}

pub fn register(registry: &mut Registry) {
    registry.register_launch(LaunchFactory::of::<Invert>("invert", || Box::new(Invert)));
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // print warnings and errors, G2G_DEBUG=*:debug for more
    g2g_core::log::init_from_env();
    let mut registry = default_registry();
    register(&mut registry);

    let frames = 5;
    let video = format!("videotestsrc num-buffers={frames} ! invert ! fakesink");
    let graph = parse_launch(&registry, &video).expect("parse");
    let stats = run_graph(graph, &WallClock::new(), LatencyProfile::Live)
        .await
        .expect("run");
    assert_eq!(stats.frames_consumed, frames);

    let audio = "audiotestsrc num-buffers=1 ! invert ! fakesink";
    let graph = parse_launch(&registry, audio).expect("parse");
    let refused = run_graph(graph, &WallClock::new(), LatencyProfile::Live).await;
    assert!(matches!(refused, Err(G2gError::CapsMismatch)));
}
```

`cargo run` pushes five inverted frames into `fakesink`. The second pipeline
shows the caps constraint at work: `audiotestsrc ! invert` fails at startup with
`CapsMismatch` before any audio reaches `process`, and the log names the two
elements that disagreed.

What each part does:

- `caps_constraint_as_transform` is what `run_graph` negotiates with
  ([§4](#4-caps-negotiation)). `Identity` says the output caps equal the input
  caps, both drawn from the set: RGBA at any size and rate.
- `intercept_caps` answers the same question for one proposed caps. Only the
  paths in [§4.3](#43-intercept_caps) call it, so it reuses the constraint's
  caps.
- `input_domains` asks for frames in host memory, so a GPU frame upstream gets a
  download spliced in instead of reaching the `else` branch.
- `ProcessFuture` is a boxed future that borrows the element and `out`. It is
  not `Send`, because `dyn OutputSink` is not.
- `PadTemplates` is the type-level description that `g2g-inspect` and
  auto-plugging read. `LaunchFactory::of` requires it.

Two fuller examples are in the repository, each one file with an element, its
properties, metadata, `register` function and a launch line:
[third_party_element.rs](g2g-plugins/examples/third_party_element.rs) (a
transform) and [third_party_source.rs](g2g-plugins/examples/third_party_source.rs)
(a source). To run them *(clone)*:

```sh
cargo run -p g2g-plugins --features std --example third_party_element
cargo run -p g2g-plugins --features std --example third_party_source
```

## 2. Which trait

| What you are writing | Trait | Defined in | Small worked example |
| :--- | :--- | :--- | :--- |
| A transform (1 in, 1 out) or a sink (1 in, 0 out) | `AsyncElement` | [g2g-core/src/element.rs](g2g-core/src/element.rs) | [volume.rs](g2g-plugins/src/volume.rs), [level.rs](g2g-plugins/src/level.rs) |
| A source (0 in, 1 out) | `SourceLoop` | [g2g-core/src/runtime/runner.rs](g2g-core/src/runtime/runner.rs) | [audiotestsrc.rs](g2g-plugins/src/audiotestsrc.rs) |
| A fan-in (N in, 1 out): a muxer, mixer, overlay | `MultiInputElement` | [g2g-core/src/fanout.rs](g2g-core/src/fanout.rs) | [audiomixer.rs](g2g-plugins/src/audiomixer.rs) |

A sink is an `AsyncElement` that pushes nothing downstream. A transform and a
sink differ only in what `process` does with the packet, which is why they share
a trait.

`MultiInputElement` receives one packet at a time, tagged with the pad it came
in on. When the element needs a *round* (one frame from each pad before it can
produce output), buffer them in an
[`InputAggregator`](g2g-core/src/aggregator.rs), the collect-and-release helper
[mux.rs](g2g-plugins/src/mux.rs) and [compositor.rs](g2g-plugins/src/compositor.rs)
use. Simpler alternative: return `true` from `input_pts_ordered()` and the runner
delivers `DataFrame`s in global PTS order across the pads for you.

Fan-out (1 in, N out) is `MultiOutputElement` in the same file, but a plain
broadcast needs no element at all: `tee` is built in, and g2g splices one for you
when an output feeds several branches.

## 3. Start from the scaffold *(clone)*

For an element that goes into `g2g-plugins` itself:

```sh
cargo xtask new-element myfilter --kind transform    # or source | sink
```

That writes the source file with the right trait skeleton, a scaffold test, the
`pub mod` line in `lib.rs`, and prints the `registry.rs` line to paste. It
compiles as generated, with `TODO`s to fill in. In a downstream crate, start
from the element in [§1](#1-a-complete-element) instead.

## 4. Caps negotiation

`Caps` is a typed enum ([g2g-core/src/caps.rs](g2g-core/src/caps.rs)), not a
string: match on the variant and its fields.

At startup `run_graph` solves the caps of every link in the graph at once. Each
element takes part through a `CapsConstraint`, returned from the method for its
role:

| Role | Declares its caps through | Default |
| :--- | :--- | :--- |
| Transform | `AsyncElement::caps_constraint_as_transform` | `IdentityAny`: passes any caps through |
| Sink | `AsyncElement::caps_constraint_as_sink` | `AcceptsAny`: takes any caps |
| Source | `SourceLoop::caps_constraint` (async) | `Produces` the one caps its `intercept_caps` returns |
| Fan-in input pad | `MultiInputElement::caps_constraint_as_input(pad)` | `AcceptsAny` |
| Fan-in output | `MultiInputElement::caps_constraint_for_output` | `Produces` the one caps its `output_caps` returns |

The transform and sink defaults accept everything. A transform that overrides
nothing negotiates as a pass-through of whatever upstream sends, audio included,
so **override the method for your role** to state what the element handles. A
mismatch then fails the run at startup with `CapsMismatch`, and the log names the
two elements ([how to read it](DEVTOOLS.md#a-custom-element-gets-caps-it-cant-handle)).

### 4.1 The constraint variants

| Variant | Shape | Use it for |
| :--- | :--- | :--- |
| `Identity(set)` | output equals input, both from `set` | filters and meters that handle a fixed list of formats ([§1](#1-a-complete-element)) |
| `IdentityAny` | output equals input, unconstrained | a pass-through that reads no format (the transform default) |
| `Mapping(pairs)` | the solver picks one `(input set, output set)` pair | an output decided by configuration rather than by the input ([srtpenc.rs](g2g-plugins/src/srtpenc.rs)) |
| `DerivedOutput(fn)` | the output set is computed from the solved input caps | an output no table can state ([volume.rs](g2g-plugins/src/volume.rs) accepts only S16LE this way) |
| `DerivedFields(transform)` | the output is derived field by field from the input | converters that change some fields and pass the rest through ([videoconvert.rs](g2g-plugins/src/videoconvert.rs), [videoscale.rs](g2g-plugins/src/videoscale.rs)) |
| `Accepts(set)` | a sink that takes caps from `set` | a sink with a format requirement |
| `AcceptsAny` | a sink that takes anything | `fakesink` (the sink default) |
| `Produces(set)` | a source that offers the caps in `set` | a source with several formats, in order of preference |

A `CapsSet` lists alternatives in order of preference. Build one with
`CapsSet::one(caps)` or `CapsSet::from_alternatives(vec)`. `caps_preferences`
replaces that order with explicit costs when two neighbours both have a
preference. `decodebin` also reads the transform constraint of the element after
it, to pick a decoder output that element accepts.

**Every link's solved caps must fixate.** After solving, the runner calls
`Caps::fixate()` on each link, and `Dim::Any` / `Rate::Any` fixate to nothing,
which fails the run at startup. An `Identity` or `Accepts` set may hold `Any`
fields, because the solver intersects it with the upstream caps. A source's
`Produces` set and a `Mapping` output must not: a source that takes a span of
geometry offers `Dim::Range { min, max }` and `Rate::Range { .. }`, which fixate
to their minimum. (`Interlace::Any` and `Colorimetry::UNKNOWN` are the
exceptions: both survive fixation on purpose, since the bitstream refines them
later.)

A source's `intercept_caps` takes no argument and is async, so it can open a
session or probe a device while answering. One that already knows its shape
returns `core::future::ready(Ok(caps))` and pays nothing. A source that offers
alternatives (a camera's pixel formats) overrides `caps_constraint` to return
all of them as `Produces(set)`.

### 4.2 `configure_pipeline`: the solved caps

`configure_pipeline(&mut self, absolute_caps: &Caps)` receives the solved caps
for this element's input pad (for a source, its output pad), every field fixed.
Allocate here, size buffers here, cache the geometry `process` will need. A
transform then gets `configure_output` with the solved caps of its output pad.
That is how a transform that takes its output shape from a downstream capsfilter
(the `videoscale ! video/x-raw,width=1280` idiom) learns its target.

Return `ConfigureOutcome::Accepted`. `ReFixate(caps)` is a counter-proposal: at
startup the runner restarts negotiation with `caps` as the output of the nearest
upstream source, when only transforms sit between them.

The runner calls `configure_pipeline` again for every mid-stream `CapsChanged`
that reaches the element. It first re-solves the new caps against the element's
constraint (a mismatch fails the run), then calls `configure_pipeline`, then
`configure_output`, then delivers the `CapsChanged` to `process`.

### 4.3 `intercept_caps`

`intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError>` is
required, but `run_graph` never calls it on a transform or sink, neither at
startup nor on a mid-stream `CapsChanged`. Two other paths call it:

- A `GraphMutator` inserting the element into a running pipeline, which checks
  it against the caps on the link being cut and refuses the insert on an `Err`.
- A launch line that expands an encoder (`encodebin`, `transcodebin`), to ask
  whether the encoder takes the stream as it is or needs a converter in front.

Return the caps you accept, or `Err(G2gError::CapsMismatch)`, consistent with
the constraint. Intersecting the upstream caps with the caps the constraint is
built from, as [§1](#1-a-complete-element) does, keeps the two from drifting
apart.

## 5. `process` and `run`

A transform or sink implements `process`, which is async and gets the packet plus
the downstream sink. `out.push(..).await` applies backpressure: it waits for
downstream capacity rather than erroring on a full link. `PipelinePacket` is the
closed set of what crosses a link (`DataFrame`, `CapsChanged`, `Segment`,
`Flush`, `Eos`, `Tick`).

A source implements `run` instead and owns its own loop, pushing until it decides
the stream is over. It **must** push the final `Eos` itself before returning
`Ok(count)`.

### 5.1 Packet rules

**A transform must not forward `Eos`.** The runner emits the single end-of-stream
after `process(Eos)` returns, so forwarding it ends the consumer twice. Match it
and do nothing, as [§1](#1-a-complete-element) does, or use it as the flush
signal for whatever the element is holding internally (which is also what a
`GraphMutator::remove` uses to drain an element on its way out,
design/runtime.md).

**`Flush` means discard, `Eos` means release.** Drop buffered state on `Flush`
and forward it.

**Emit `CapsChanged` before the first `DataFrame` it applies to, and only when
the caps actually changed.** Negotiation settles the startup shape. A mid-stream
refinement (a decoder learning the real geometry from an SPS, a converter
retargeting) travels as a packet instead. Keep the last emitted caps and compare, the way
[volume.rs](g2g-plugins/src/volume.rs) does with its `last_caps` field, so a
steady stream emits exactly one. Re-emitting an unchanged caps forces the runner
into a needless re-solve.

Anything else (`Segment`, `Tick`, a packet a future version adds) forwards
unchanged, the `other` arm in [§1](#1-a-complete-element).

## 6. Properties

Every knob a real pipeline would tune must be settable by name, so a launch line
can reach it. Two halves, both required:

- `properties()` returns a `&'static [PropertySpec]`. `parse_launch` reads it to
  learn each name's `PropKind` before it can parse the text value.
- `set_property` / `get_property` apply and read it back.

```rust
use g2g_core::{PropKind, PropertySpec};

static VOLUME_PROPS: &[PropertySpec] = &[
    PropertySpec::new("volume", PropKind::Double, "linear gain, 0..N (1 = unchanged)"),
    PropertySpec::new("mute", PropKind::Bool, "zero the output when true"),
];
```

Rules:

- **kebab-case**, and named after the GStreamer property where an analog exists,
  with its unit and semantics (`bitrate` is `PropKind::Uint` in bits/second, not
  kbit/s).
- **Never accept a property you then ignore.** Expose only behavior the element
  applies.
- A `with_*` builder is the zero-cost construction path, not a substitute: every
  builder knob gets a property too.
- `.with_default("-1")` / `.with_range("0", "10")` / `.with_enum_values("a | b")`
  document the spec for `g2g-inspect`. For a `Str` or `Flags` property the enum
  list is the *closed* set the parser validates against, so list every nick the
  element accepts.

[third_party_element.rs](g2g-plugins/examples/third_party_element.rs) shows both
halves on a small element. In this repository,
[g2g-plugins/tests/m454_element_properties.rs](g2g-plugins/tests/m454_element_properties.rs)
is the pattern for testing them: assert the name is declared, then round-trip it
through `set_property` / `get_property` onto the field the element acts on.

## 7. Pad templates and metadata

`PadTemplates` ([g2g-core/src/pad_template.rs](g2g-core/src/pad_template.rs))
declares what the element type can link to, before any instance exists. The
auto-plug search and `g2g-inspect` read it. For an audio element:

```rust
use g2g_core::pad_template::{PadTemplate, PadTemplates};
use g2g_core::{AudioFormat, Caps, CapsSet, ChannelLayout};

struct Volume;

impl PadTemplates for Volume {
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: ChannelLayout::UNSPECIFIED,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(pcm.clone())),
            PadTemplate::source(CapsSet::one(pcm)),
        ])
    }
}
```

Templates describe the type, the constraint describes the instance: negotiation
reads only the constraint ([§4](#4-caps-negotiation)), so keep the two in
agreement. Templates may leave geometry open (`Dim::Any`).

`metadata()` is the one-line self-description a `g2g-inspect` dump prints: long
name, classification, description, author. Use the GStreamer classification
strings (`Filter/Effect/Audio`, `Source/Video`, `Filter/Converter/Video`) so the
catalog sorts alongside the rest.

## 8. The `no_std + alloc` baseline

`g2g-core` and `g2g-plugins` are `#![no_std]` with `extern crate alloc`. A pure
computational element belongs in that baseline: it then also runs on an MCU and
in the browser. In practice that means `use alloc::vec::Vec` rather than `std`,
and no `libm` (there is a `crate::mathf` for `sqrt` / trig, and elements like
`audiotestsrc` approximate deliberately to stay dependency-free).

Anything needing the OS (sockets, a GPU, COM, a device node) goes behind a named
cargo feature that implies `std`, with the dependency `optional = true`:

```toml
rtsp = ["std", "dep:retina", "dep:futures-util", "dep:url", "dep:bytes"]
```

Platform-specific dependencies go under `[target.'cfg(...)'.dependencies]`, and
the module is gated `#[cfg(all(target_os = "windows", feature = "mf-decode"))]`.

If the element reads host memory, say so:
`input_domains()` returning `DomainSet::only(MemoryDomainKind::System)` makes the
allocation cascade insert a download on a GPU producer, rather than handing the
element a VRAM frame it cannot touch.

## 9. Parsers: never trust the stream

Counts, lengths, offsets and dimensions read from a bitstream or a container are
attacker-controlled. Validate before use, and fold arithmetic with checked or
saturating operations, so malformed input **fails the parse** (returns `None` or
an error) instead of panicking, overflowing, or allocating on a bogus length.

[h264parse.rs](g2g-plugins/src/h264parse.rs) derives SPS geometry entirely in
saturating arithmetic (the crop additions and the `* 16` would otherwise
overflow), and [fmp4.rs](g2g-plugins/src/fmp4.rs) bounds its box and sample
counts before allocating. The stream parsers also have libFuzzer targets under
`g2g-plugins/fuzz`, which CI runs weekly *(clone)*:

```sh
cd g2g-plugins/fuzz && cargo +nightly fuzz run h264parse
```

## 10. Registering it so a launch line can use it

A programmatic graph needs no registry: construct the element and `add_transform`
it. Registration is what lets `parse_launch` and `g2g-launch` resolve a *name*.

| Element kind | Call |
| :--- | :--- |
| Transform / sink | `registry.register_launch(LaunchFactory::of::<T>("name", || Box::new(T::new())))` |
| Source | `registry.register_source(SourceFactory::new("name", caps, || Box::new(T::new())))` |
| Fan-in muxer | `registry.register_muxer(MuxerFactory::new("name", \|n\| Box::new(T::new(n))))` |
| Auto-plug candidate (`decodebin` may pick it) | `registry.register(ElementFactory::of::<T>("name", ..))` |

The convention for a downstream crate is one public `register` function, the
one in [§1](#1-a-complete-element), so an application composes registries:
`default_registry()`, then `my_crate::register(&mut registry)` for each crate.
In-tree, the call goes in `default_registry` in
[g2g-plugins/src/registry.rs](g2g-plugins/src/registry.rs).

For an element that must load into an installed g2g binary with no recompile, see
[PORTING.md §7](PORTING.md#7-adding-third-party-elements--plugins): the Python
host (`pyelement`), or a native `.so` built against `g2g-plugin`'s
`declare_plugin!`.

## 11. Testing

Test the feature, not the line count. A test imports and runs the real element,
mocking only external boundaries (network, COM, GPU), never the code under test.
Every test needs an assertion that fails if the feature breaks.

Unit tests go in a `#[cfg(test)] mod tests` beside the element, over the pure
function where there is one ([volume.rs](g2g-plugins/src/volume.rs) tests
`apply_gain` directly, plus `configure_pipeline`'s rejection of a wrong format).
A pipeline test runs a launch line through `run_graph` and asserts on the
result, the way `main` in [§1](#1-a-complete-element) checks that audio is
refused. In this repository those go in `g2g-plugins/tests/`, one file per
milestone *(clone)*:

```sh
cargo test --workspace
cargo test -p g2g-plugins --features rtsp     # a feature-gated element
```

**A feature-gated test file runs zero tests without its feature** and still
reports success, because `std` only reaches it through workspace feature
unification. Always pass the file's features explicitly and check the reported
test count is nonzero.

## 12. Before you call it done

- The constraint for the element's role (`caps_constraint_as_transform`,
  `caps_constraint_as_sink`, `caps_constraint`) admits only what it handles,
  and every solved link can `fixate()`.
- `intercept_caps` agrees with that constraint.
- `process` swallows `Eos`, and `CapsChanged` is emitted before the frame it
  applies to and suppressed when unchanged.
- Every `with_*` knob has a matching entry in `properties()` *and* a
  `set_property` / `get_property` arm.
- `metadata()` and `pad_templates()` are filled in, so `g2g-inspect myelement`
  reads sensibly.
- `cargo clippy --workspace --all-targets` is clean (CI treats warnings as
  errors) and `cargo fmt --all --check` passes.
- In-tree, the element appears in `g2g-inspect` and in the regenerated
  [docs/elements.html](docs/elements.html) (see
  [DEVTOOLS.md](DEVTOOLS.md#element-reference-g2g-inspect-and-the-web-page)).
