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
([§8](#8-registering-it-so-a-launch-line-can-use-it)).

Two runnable examples follow this guide end to end:

```sh
cargo run -p g2g-plugins --features std --example third_party_element   # a transform
cargo run -p g2g-plugins --features std --example third_party_source    # a source
```

---

## 1. Which trait

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

## 2. Start from the scaffold

Inside this repository:

```sh
cargo xtask new-element myfilter --kind transform    # or source | sink
```

That writes the source file with the right trait skeleton, a scaffold test, the
`pub mod` line in `lib.rs`, and prints the `registry.rs` line to paste. It
compiles as generated, with `TODO`s to fill in. In a downstream crate, copy
[third_party_element.rs](g2g-plugins/examples/third_party_element.rs) or
[third_party_source.rs](g2g-plugins/examples/third_party_source.rs) instead:
each is one file carrying an element, its pad templates, its properties, its
`register` function, and a launch line that runs it.

## 3. The lifecycle, in order

The runner calls `intercept_caps` and then `configure_pipeline` once each at
startup, and `process` (or a source's `run`) after that.

### 3.1 `intercept_caps`: negotiation

```rust
fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError>;
```

Given what upstream proposes, return what this element accepts, or
`Err(G2gError::CapsMismatch)` to refuse. `Caps` is a typed enum
([g2g-core/src/caps.rs](g2g-core/src/caps.rs)), not a string: match on the
variant and its fields.

`volume` accepts only S16LE PCM with at least one channel and passes the caps
through unchanged. Anything else is a `CapsMismatch`, which fails the link at
startup rather than at the first frame.

**Never return a caps that cannot fixate.** After negotiation the solver calls
`Caps::fixate()`, and `Dim::Any` / `Rate::Any` fixate to nothing, which fails the
run at startup. An element that genuinely accepts a span of
geometry advertises `Dim::Range { min, max }` and `Rate::Range { .. }`, which
fixate to their minimum. (`Interlace::Any` and `Colorimetry::UNKNOWN` are the
exceptions: both survive fixation on purpose, since the bitstream refines them
later.)

A source's `intercept_caps` takes no argument and is async, so it can open a
session or probe a device while answering. One that already knows its shape
returns `core::future::ready(Ok(caps))` and pays nothing.

### 3.2 `configure_pipeline`: absolute caps

```rust
fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError>;
```

Called once with the *solved* caps for this element's input pad: every field
fixed. Allocate here, size buffers here, cache the geometry `process` will need.
Return `ConfigureOutcome::Accepted`. `ReFixate(caps)` asks for a different shape,
and is only honored during the allocation re-cascade, not at startup.

A transform that takes its output shape from a downstream capsfilter (the
`videoscale ! video/x-raw,width=1280` idiom) also overrides `configure_output`,
which receives that solved *output* pad's caps.

### 3.3 `process` / `run`: the work

A transform or sink implements `process`, which is async and gets the packet plus
the downstream sink:

```rust
fn process<'a>(&'a mut self, packet: PipelinePacket, out: &'a mut dyn OutputSink)
    -> Self::ProcessFuture<'a>;
```

`out.push(..).await` applies backpressure: it waits for downstream capacity
rather than erroring on a full link. `PipelinePacket` is the closed set of what
crosses a link (`DataFrame`, `CapsChanged`, `Segment`, `Flush`, `Eos`, `Tick`).

A source implements `run` instead and owns its own loop, pushing until it decides
the stream is over. It **must** push the final `Eos` itself before returning
`Ok(count)`.

### 3.4 Packet rules

**A transform must not forward `Eos`.** The runner emits the single end-of-stream
after `process(Eos)` returns, so forwarding it ends the consumer twice. Match it
and do nothing, or use it as the flush signal for whatever the element is holding
internally (which is also what a `GraphMutator::remove` uses to drain an element
on its way out, DESIGN.md §4.8.6):

```rust
PipelinePacket::Eos => {}
```

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
unchanged: `other => out.push(other).await?`.

## 4. Properties

Every knob a real pipeline would tune must be settable by name, so a launch line
can reach it. Two halves, both required:

- `properties()` returns a `&'static [PropertySpec]`. `parse_launch` reads it to
  learn each name's `PropKind` before it can parse the text value.
- `set_property` / `get_property` apply and read it back.

```rust
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

[g2g-plugins/tests/m454_element_properties.rs](g2g-plugins/tests/m454_element_properties.rs)
is the pattern to copy for testing this: assert the name is declared, then
round-trip it through `set_property` / `get_property` onto the field the element
acts on.

## 5. Pad templates and metadata

`PadTemplates` ([g2g-core/src/pad_template.rs](g2g-core/src/pad_template.rs))
declares what the element type can link to, before any instance exists. The
auto-plug search and `g2g-inspect` read it:

```rust
impl PadTemplates for Volume {
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = Caps::Audio { format: AudioFormat::PcmS16Le, channels: 2, sample_rate: 48_000 };
        Vec::from([PadTemplate::sink(CapsSet::one(pcm.clone())), PadTemplate::source(CapsSet::one(pcm))])
    }
}
```

Templates may leave geometry open (`Dim::Any` is fine *here*, it is resolved at
instance time), unlike `intercept_caps`.

`metadata()` is the one-line self-description a `g2g-inspect` dump prints: long
name, classification, description, author. Use the GStreamer classification
strings (`Filter/Effect/Audio`, `Source/Video`, `Filter/Converter/Video`) so the
catalog sorts alongside the rest.

## 6. The `no_std + alloc` baseline

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

## 7. Parsers: never trust the stream

Counts, lengths, offsets and dimensions read from a bitstream or a container are
attacker-controlled. Validate before use, and fold arithmetic with checked or
saturating operations, so malformed input **fails the parse** (returns `None` or
an error) instead of panicking, overflowing, or allocating on a bogus length.

[h264parse.rs](g2g-plugins/src/h264parse.rs) derives SPS geometry entirely in
saturating arithmetic (the crop additions and the `* 16` would otherwise
overflow), and [fmp4.rs](g2g-plugins/src/fmp4.rs) bounds its box and sample
counts before allocating. The stream parsers also have libFuzzer targets under
`g2g-plugins/fuzz`, which CI runs weekly:

```sh
cd g2g-plugins/fuzz && cargo +nightly fuzz run h264parse
```

## 8. Registering it so a launch line can use it

A programmatic graph needs no registry: construct the element and `add_transform`
it. Registration is what lets `g2g-launch` resolve a *name*.

| Element kind | Call |
| :--- | :--- |
| Transform / sink | `registry.register_launch(LaunchFactory::of::<T>("name", || Box::new(T::new())))` |
| Source | `registry.register_source(SourceFactory::new("name", caps, || Box::new(T::new())))` |
| Fan-in muxer | `registry.register_muxer(MuxerFactory::new("name", \|n\| Box::new(T::new(n))))` |
| Auto-plug candidate (`decodebin` may pick it) | `registry.register(ElementFactory::of::<T>("name", ..))` |

The convention for a downstream crate is one public `register` function, so an
application composes registries:

```rust
pub fn register(registry: &mut g2g_core::runtime::Registry) {
    registry.register_launch(LaunchFactory::of::<MyFilter>("myfilter", || Box::new(MyFilter::new())));
}

let mut registry = default_registry();
my_crate::register(&mut registry);
let graph = parse_launch(&registry, "videotestsrc ! myfilter ! fakesink")?;
```

In-tree, that call goes in `default_registry` in
[g2g-plugins/src/registry.rs](g2g-plugins/src/registry.rs).

For an element that must load into an installed g2g binary with no recompile, see
[PORTING.md §7](PORTING.md#7-adding-third-party-elements--plugins): the Python
host (`pyelement`), or a native `.so` built against `g2g-plugin`'s
`declare_plugin!`.

## 9. Testing

Test the feature, not the line count. A test imports and runs the real element,
mocking only external boundaries (network, COM, GPU), never the code under test.
Every test needs an assertion that fails if the feature breaks.

Unit tests go in a `#[cfg(test)] mod tests` beside the element, over the pure
function where there is one ([volume.rs](g2g-plugins/src/volume.rs) tests
`apply_gain` directly, plus `configure_pipeline`'s rejection of a wrong format).
Whole-pipeline tests go in `g2g-plugins/tests/`, one file per milestone.

```sh
cargo test --workspace
cargo test -p g2g-plugins --features rtsp     # a feature-gated element
```

**A feature-gated test file runs zero tests without its feature** and still
reports success, because `std` only reaches it through workspace feature
unification. Always pass the file's features explicitly and check the reported
test count is nonzero.

## 10. Before you call it done

- `intercept_caps` returns something `fixate()` can resolve.
- `process` swallows `Eos`, and `CapsChanged` is emitted before the frame it
  applies to and suppressed when unchanged.
- Every `with_*` knob has a matching entry in `properties()` *and* a
  `set_property` / `get_property` arm.
- `metadata()` and `pad_templates()` are filled in, so `g2g-inspect myelement`
  reads sensibly.
- `cargo clippy --workspace --all-targets` is clean (CI treats warnings as
  errors) and `cargo fmt --all --check` passes.
- The element appears in `g2g-inspect` and in the regenerated
  [docs/elements.html](docs/elements.html) (see
  [DEVTOOLS.md](DEVTOOLS.md#element-reference-g2g-inspect-and-the-web-page)).
