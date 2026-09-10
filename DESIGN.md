# glass2glass design

## Summary and design philosophy

`glass2glass` (`g2g`) is a multimedia graph framework written in 100% pure Rust.
It is built around one idea: a pure-Rust core, so the same typed pipeline runs
unchanged across the whole hardware spectrum, MCU, RTOS, CPU, GPU, and WASM. A
`no_std + alloc`, sans-IO core means the graph, the element traits, the caps
negotiation, and the runner are identical on a bare-metal microcontroller, a
real-time (Embassy) target, a CPU server, a GPU-resident zero-copy pipeline, and
the web browser. Only the deployment shell changes: which executor, which
hardware elements.

The metric the project optimizes is glass-to-glass latency, the time elapsed
between physical photon or audio capture and hardware presentation.

### The four pillars

1. Asynchronous execution. Every element is a cooperative async task
   (`Future`). The framework manages no OS threads of its own and is
   runtime-agnostic.
2. Hardware-first and zero-copy. Data stays in VRAM or unified memory domains
   via hardware handles (`DMABUF`, Vulkan textures). A CPU memory copy is
   treated as a system fault.
3. Modular predictability. A `no_std + alloc` core lets the same pipelines
   execute on bare-metal microcontrollers, multi-threaded servers, or
   WebAssembly targets. Network and protocol parsers use a sans-IO design that
   keeps I/O out of the logic layer.
4. First-class machine learning. Tensor allocation, reshaping, and pipeline
   batching are part of the graph orchestration layer and execute in-flight on
   GPU memory.

### Architecture at a glance

A `g2g` pipeline is a graph of typed elements joined by bounded async channels.
A source produces packets, transforms rewrite them, and sinks consume them:

```
  Source ─────▶ Transform ─────▶ … ─────▶ Sink
 (RtspSrc,     (H264Parse,               (WaylandSink,
  V4l2Src,      decoder,                  WgpuSink,
  Mp4Src)       ML preprocess)            UdpSink)

  on each link:  CapsChanged · DataFrame(Frame) · Segment · Flush · Eos
```

Before any frame flows, the runner runs one caps-negotiation pass over the whole
graph ([DESIGN-caps.md](DESIGN-caps.md)): every link is assigned a concrete
`Caps`, every element allocates its buffers, and the memory domain each link
carries (System / DMABUF / CUDA / Vulkan / WebGPU texture) is settled, so a
zero-copy link stays zero-copy. Each element then runs as its own cooperative
async task, paced by channel backpressure rather than an internal thread.

The types you meet everywhere are all in `g2g-core`:

| Type | Role |
| :--- | :--- |
| `Frame` | one media buffer: a `MemoryDomain` payload + `FrameTiming` + a sequence number + optional metadata. Caps live on the *link*, not the frame. |
| `PipelinePacket` | what crosses a link: `CapsChanged` / `DataFrame` / `Segment` / `Flush` / `Eos`, plus the arm-local `Tick`. |
| `Caps` | the typed capability algebra (`RawVideo` / `CompressedVideo` / `Audio` / `Tensor` / `Text` / `ByteStream`), negotiated per link. |
| `AsyncElement` / `SourceLoop` | the two element traits, transform-or-sink and source. Pads are implicit in the trait shape, not a runtime object. |
| `MemoryDomain` | where a frame's bytes live: System, DMABUF, CUDA, Vulkan / WebGPU texture, the basis for zero-copy. |
| the runner (`run_graph`) | drives negotiation and then one async task per node over the channels. |

### Reading guide

This file is the architecture overview: the frame and memory model, caps and the
element traits, backpressure, and the deployment profiles. Each track has its own
document.

| Where | What |
| :--- | :--- |
| [DESIGN-caps.md](DESIGN-caps.md) | the caps CSP solver, allocation cascade, auto-plug, `decodebin` / `playbin`, bins. |
| [DESIGN-runtime.md](DESIGN-runtime.md) | dynamic graph reconfiguration, the state machine, preroll and seek, the bus, logging. |
| [DESIGN-timing.md](DESIGN-timing.md) | clock election, the audio master, PTP, presentation anchoring, latency, QoS. |
| [DESIGN-decode.md](DESIGN-decode.md) | hardware decoders and encoders, and the end-to-end RTSP pipeline. |
| [DESIGN-live.md](DESIGN-live.md) | capture, device discovery, RTP / RTMP / RTSP / SRT in both directions, fallback switching. |
| [DESIGN-containers.md](DESIGN-containers.md) | containers and byte streams, mux and demux, HLS and DASH, still images. |
| [DESIGN-text.md](DESIGN-text.md) | subtitles, closed captions, bitmap subtitles, teletext, the overlay elements. |
| [DESIGN-transports.md](DESIGN-transports.md) | WebRTC, distributed graphs, MoQ, ST 2110, local zero-copy IPC. |
| [DESIGN-launch.md](DESIGN-launch.md) | properties, introspection, the `gst-launch` DSL, plugins, hosted Python and Rhai. |
| [DESIGN-ml.md](DESIGN-ml.md) | GPU tensor preprocess, inference backends, batching, detection metadata. |
| [DESIGN-tooling.md](DESIGN-tooling.md) | DOT dumps, the negotiation explainer, `xtask`, telemetry, conformance. |
| [DESIGN-embedded.md](DESIGN-embedded.md) | the heap-free core, MCU elements, RTOS executors, footprint proofs. |

Open work is in [DESIGN_TODO.md](DESIGN_TODO.md). Shipped milestones are logged
in [CHANGELOG.md](CHANGELOG.md).

---

## Workspace structure and licensing

The project is a Cargo workspace, which enforces boundaries between interfaces,
standard elements, ML backends, and platform bindings.

| Crate | Purpose | Target profile | Licensing |
| :--- | :--- | :--- | :--- |
| `g2g-core` | Core traits, `Frame` definitions, buffer pool allocators, clock model. | `no_std + alloc` | MPL-2.0 |
| `g2g-mcu` | Heap-free MCU peripheral, codec and transport elements over `embedded-hal` seams ([DESIGN-embedded.md](DESIGN-embedded.md)). | `no_std`, no `alloc` | MPL-2.0 |
| `g2g-mcugen` | Host compiler turning a declarative graph document into a monomorphized static MCU pipeline. | `std` | MPL-2.0 |
| `g2g-plugin` | SDK for dynamically loadable plugins (the `declare_plugin!` macro + ABI tag, [DESIGN-launch.md](DESIGN-launch.md)). | `no_std + alloc` | MPL-2.0 |
| `g2g-plugins` | Standard collection of source/sink/transform elements (`rtsp`, `wgpu`, `v4l2`). | `no_std + alloc` / `std` mixed | MPL-2.0 |
| `g2g-ml` | ML inference elements built on `burn` (Wasm/embedded) and `ort` (server), plus the multi-stream tensor batcher. | `std` | MPL-2.0 |
| `g2g-bridge` | C-FFI dynamic library to embed `g2g` sub-graphs inside GStreamer pipelines. | `std` (`cdylib`) | MPL-2.0 |
| `g2g-python` | Hosts gst-python-ml elements as first-class `g2g` elements (embedded CPython via pyo3). | `std` | MPL-2.0 |
| `g2g-capi` | C ABI (cdylib/staticlib + `g2g.h`) to drive pipelines from any language: `parse_launch` + run + bus + appsrc/appsink. | `std` (`cdylib`) | MPL-2.0 |
| `g2g-pyapi` | Python (pyo3) bindings to drive pipelines: `parse_launch` + run + bus + appsrc/appsink, the inverse of `g2g-python`. | `std` | MPL-2.0 |

The `no_std + alloc` baseline is deliberate: it admits cooperative async
executors, which need a heap for futures, and `Arc` reference counting, while
still excluding the OS-dependent surface of `std`. Targets requiring strict
no-heap allocation use the static `BufferPool` and avoid the `dyn`-safe
element wrappers.



Pointers to the heap-free build of `g2g-core`, the static element model, and the
MCU element crates are in [DESIGN-embedded.md](DESIGN-embedded.md).

---

## Data representation and memory

### The Frame carrier

Media components flow through lock-free async channels as structured variants
representing data packets, lifecycle signals, or negotiation hooks, rather than
heavy C-style objects.

```rust
pub enum PipelinePacket {
    CapsChanged(Caps),
    DataFrame(Frame),
    Eos,
    /// Seek flush: discard in-flight and buffered data and reset position
    /// state. Unlike `Eos`, the stream resumes after a flush.
    Flush,
    /// Deadline tick: a fan-in element declaring `tick_interval_ns` gets one
    /// per period even while its inputs stall, so it can emit on its own
    /// cadence (the compositors' zero-order-hold). May fire spuriously, and
    /// never crosses a link: the runner's arm originates and consumes it.
    Tick,
}

pub struct Frame {
    pub domain: MemoryDomain,
    pub timing: FrameTiming,
    /// Monotonically increasing per-source sequence number assigned at
    /// capture time and preserved unchanged across the pipeline. Used
    /// for drop detection and tracing, never for AV sync.
    pub sequence: u64,
    /// Per-frame attachable metadata (the GstMeta analog). Empty on
    /// construction.
    pub meta: FrameMetaSet,
}
```

Both runners derive the ticker from the pipeline clock, `as_ticker` for the
cooperative one and `shared_ticker` for thread-per-arm, and a clock with
interior state reaches the arms via `run_graph_threaded_ticked`.

`Frame` carries a `meta` side-channel for typed blobs that travel with the
buffer: ML detection, classification and tracking results, regions of interest,
reference timestamps. It is gated behind the `metadata` cargo feature, off by
default: when off it is a zero-sized unit, so the `no_std` and RTOS baseline pays
nothing per frame, and when on it is a list of `Arc<dyn FrameMeta>`. The full
attach, propagate and demand contract is in [DESIGN-ml.md](DESIGN-ml.md).
Construct frames via `Frame::new(domain, timing, sequence)` so future field
additions do not break call sites.

Caps live on the link, not on the frame. The current caps of a link are
established by the most recent `PipelinePacket::CapsChanged(Caps)` packet to
arrive, and every subsequent `DataFrame` on that link is implicitly under those
caps until the next `CapsChanged` arrives. The runner guarantees `CapsChanged` is
ordered in the stream, sitting between the last old-caps `DataFrame` and the
first new-caps `DataFrame`, which is the load-bearing correctness property for
mid-stream format changes ([DESIGN-caps.md](DESIGN-caps.md)).

### Memory domains

g2g treats system RAM as a fallback. Buffers track hardware descriptors to allow
cross-process and cross-hardware zero-copy manipulation.

```rust
pub enum MemoryDomain {
    System(SystemSlice),
    DmaBuf(OwnedDmaBuf),
    VulkanTexture(OwnedVulkanTexture),
    WebGPUBuffer(OwnedWebGPUBuffer), // For Wasm targets
}

/// RAII wrapper that closes the underlying DMABUF on drop.
/// On `no_std` targets without libc, the owning `BufferPool` registers
/// a custom close hook via `BufferPool::with_close_fn`.
pub struct OwnedDmaBuf {
    fd: i32,
    pub stride: u32,
    pub offset: u32,
}
```

Every hardware handle is reference-counted, an `Arc`-held keep-alive owner or an
`Arc`-shared fd for DMABUF, so the underlying file descriptor or GPU allocation
is released on the last drop. Vulkan and WebGPU handles follow the same RAII
pattern, parameterised over a backend-specific allocator handle so the design
does not bake in a single binding crate.

`MemoryDomain::share()` produces a second handle for a fan-out branch: a
zero-copy refcount bump for the GPU domains and the shared-CPU `SystemView`, and
a deep copy only for owned-CPU `System` bytes. So a tee broadcasts a GPU-resident
frame to several consumers, decode-on-GPU into inference and display, with no
device-to-host copy. Branches treat the shared memory as read-only, and a
mutating branch copies first, as the per-frame metadata does copy-on-write.

### The copy plan

Because negotiation resolves the memory domain of every link before a frame
flows, whether a pipeline is zero-copy is answerable at construction time, not
only measurable after. `copyplan`, pure like `dot`, turns the negotiated per-edge
domains and fixated caps into a `CopyPlan`: the sequence of memory hops, the
domain a frame occupies on each edge, and the transfers between differing
domains.

A transfer is recorded at any node whose output domain differs from the domain it
consumed. `classify` sorts it into `None`, `Interop` (a dma-buf import or export
or a device-to-device bridge), `DeviceHost` (a GPU download or upload over the
bus), or `CrossDevice`, and it counts as a real frame copy only when a raw heavy
buffer (`Caps::is_raw_media`, so raw video, PCM audio or a tensor) crosses on both
sides. A decode from `CompressedVideo` to `RawVideo`, or an off-GPU encode, is
shown in the trace but not miscounted.

`CopyPlan::check(CopyPolicy)`, over `Allow`, `AtMost(n)` and `DenyAll`, enforces a
copy budget as a graph-level contract: a pipeline meant to stay resident on the
GPU fails the check the moment an accidental host round-trip appears, rather than
silently paying for it at runtime. `g2g-launch --copy-plan` prints the report and
`runtime::copy_plan(vg, caps, memory)` builds it from a negotiated graph. The
runner enforces it directly: `run_graph_with_copy_policy` runs the plan after
negotiation and, before any frame flows, refuses to start a graph that exceeds
the budget (`G2gError::CopyBudget`). This is what GStreamer cannot state: not
that zero-copy is possible, but that this graph is proven zero-copy or it will
not start.

The check is scoped precisely, to memory-domain transfers of a raw frame, a
device-to-host or cross-device copy. An intra-domain algorithmic copy, a
`videoconvert` allocating a new System buffer, stays within one domain and is not
a domain transfer, and the plan trusts each element's declared `output_memory`
and `input_domains`. So zero-copy here means no raw frame crosses a memory-domain
boundary, the property that governs GPU-resident and DMA pipelines.

### Buffer pools

Inside real-time or `no_std` loops, dynamic allocation during steady-state
streaming is prohibited. Elements acquire pre-allocated slots from a bounded
`BufferPool` and dropping the resulting handle returns the buffer.

```rust
let pool = BufferPool::new_byte_pool(count, bytes);
let buf = pool.acquire().await;  // awaits if exhausted; backpressure-friendly
let mut frame = SystemSlice::from_pool(buf, frame_len);  // valid payload length
```

On `no_std + alloc` and `std`, `BufferPool<T>` wraps `Arc<Mutex<Vec<T>>>` plus a
`VecDeque<Waker>` of acquire waiters, `acquire().await` resolves the moment a
`PooledBuffer` elsewhere is dropped, and `try_acquire()` is the sync fast path for
non-blocking contexts. The strict no-heap pools and the `StaticLendRing` zero-copy
lend are in [DESIGN-embedded.md](DESIGN-embedded.md).

The `SystemSlice` carrier supports three ownership models transparently:
`SystemSlice::from_boxed(Box<[u8]>)` for one-off frames,
`SystemSlice::from_pool(PooledBuffer<Box<[u8]>>, len)` for recycled frames, where
the buffer may exceed the frame so the valid length is carried, and
`SystemSlice::from_foreign(ptr, len, free, user)` for a zero-copy lend of borrowed
bytes, a `StaticLendRing` slot or an application buffer through the C ABI.
Downstream elements treat them identically.

The control plane carries the same no-allocation contract. `OutputSink` is
poll-based, its required method being `poll_push` with `push` wrapping it in a
stack `PushFuture`, so a push through `&mut dyn OutputSink` costs no heap either,
pinned at zero by `m616_dyn_push_allocates`. The element's own `ProcessFuture` is
opt-in: an element that declares a boxed one pays one box per `process` call, and
one that declares a concrete future type runs heap-free through the whole dyn
runner, with `m1000_dyn_graph_noalloc` proving a 3-stage `run_graph` steady state
at zero allocations.

---

## Graph orchestration

### Typed caps

Traditional architectures rely on runtime string lookups for stream capabilities,
such as `"video/x-raw, format=NV12"`. g2g enforces strongly typed structures.

```rust
#[derive(Clone, Debug, PartialEq)]
pub enum Caps {
    RawVideo { format: VideoFormat, width: Dim, height: Dim, framerate: Rate, .. },
    CompressedVideo { codec: VideoCodec, .. },
    Audio { format: AudioFormat, channels: u8, sample_rate: u32 },
    Tensor { dtype: TensorDType, shape: TensorShape, layout: TensorLayout },
    Text { format: TextFormat },
    ByteStream { encoding: ByteStreamEncoding },
    ..
}

/// `Fixed` after Phase 2; `Range`/`Any` only legal during Phase 1.
pub enum Dim { Any, Range { min: u32, max: u32 }, Fixed(u32) }
pub enum Rate { Any, Range { min_q16: u32, max_q16: u32 }, Fixed(u32) }
```

The `Tensor` variant is first-class because ML elements negotiate caps the same
way video elements do, rather than sitting outside the graph model. The video
kinds split into `CompressedVideo` and `RawVideo` so a codec-to-raw mismatch is
caught at `intersect`, and `ByteStream` types a not-yet-demuxed container link.

`Text` is likewise a first-class media kind, `Caps::Text { format: TextFormat }`,
rather than a bolted-on subtitle path. A `Text` link carries any text payload, a
subtitle cue, a caption, a transcription, an OCR result or an overlay string,
with `TextFormat` naming the syntax (`Utf8`, `PangoMarkup`, and the structured
`Srt`, `WebVtt`, `Ssa` and `Ttml`). Subtitle is not a separate variant: it is
timed `Text`, the cue's on-screen window carried as the frame's PTS and duration,
so one caps kind serves overlay rendering, captioning and text analytics. A
subtitle parser (`SubParse`) is the text-domain analog of a codec decoder, taking
a structured format on its sink pad and emitting plain `Utf8` cues via the same
`DerivedOutput` negotiation a decoder uses.

### Colorimetry

Both video variants carry a `colorimetry: Colorimetry` field: four enums for the
CICP vocabulary (`MatrixCoefficients`, `TransferCharacteristics`,
`ColorPrimaries`, `ColorRange`), each with an `Unknown` wildcard, plus presets
matching GStreamer's named colorimetries (`Colorimetry::BT601`, `BT709`,
`BT2020`, `BT2100_PQ`, `BT2100_HLG`, `SRGB`).

In `intersect`, `Unknown` is the per-field identity and two different concrete
values are an empty overlap, so an untagged link never blocks a tagged peer while
a wrong tag fails the link instead of silently converting with the wrong matrix.
Like `Interlace`, `Unknown` survives `fixate` and counts as fixed: negotiation
solves as before, and the concrete value arrives at runtime via `CapsChanged` when
a bitstream parser refines it, `h264parse` and `h265parse` reading the VUI colour
description and `vulkanvideodec` reading the VUI or the AV1 `color_config`.
Codepoints are validated at the parse boundary by `Colorimetry::from_cicp`, where
anything unmodeled maps to `Unknown` and never a guess.

Decoders copy the input caps' colorimetry onto their raw output caps. The encode
side is the inverse: each field's `to_cicp` gives the codepoint the caps value
names, with `Unknown` writing 2 for unspecified, and `ffmpegenc` puts those on the
libavcodec context before it opens the encoder, which is where libx264 and NVENC
both read the SPS VUI colour description from. The same colorimetry rides the
compressed output caps, so an untagged input encodes to an untagged stream and a
tagged one survives a re-encode. The caps string form is GStreamer's
`colorimetry=` value, a preset name or the numeric
`range:matrix:transfer:primaries` 4-part form, printed by `to_gst_string` and read
by the launch caps-filter parser.

Converters and sinks consume the field, matrix and range only.
`Colorimetry::yuv_conversion` resolves the pair a stage converts with, mapping an
`Unknown` matrix or range, and `Identity` which names GBR planes, to BT.601
limited, so an untagged stream converts exactly as it did before the field
existed. That is the single place the fallback lives, and a better guess would
change that function alone. The weights come from one `LumaCoefficients` table
beside `MatrixCoefficients`, the BT.601, BT.709 and BT.2020 NCL `(Kr, Kb)` pairs.
`g2g-plugins`' `yuvmatrix` derives from them both the 8-bit fixed-point tables the
CPU paths use (`videoconvert`, `videoconvertscale`, `waylandsink`, the
`compositor` background fill) and the normalized weights compiled into the GL and
WGSL convert shaders (`glsink`, `cudaglsink` and `cudakmssink` through `glnv12`,
and `wgpusink`), so no coefficient is written out twice. `g2g-ml`'s
`WgpuPreprocess` reads the same weights: they reach its compute shader in the dims
uniform rather than being spliced into the shader text, so a caps change rewrites
48 bytes instead of rebuilding the pipeline, and the host mirror the GPU tests
compare against is built from the same struct.

Converting the colorimetry itself is the `colorspace` element, the complement of
`videoconvert`: it changes what the samples mean and leaves the pixel format
alone, over 8-bit I420 and NV12 plus packed RGBA, BGRA and RGB. Matrix and range
go through an 8-bit RGB intermediate, decoding with the input's `YuvRgbMatrix` and
encoding with the output's, which makes it exactly a `videoconvert` to RGBA and
back and is how its tests derive their expected pixels. Transfer and primaries go
through linear light: the source curve linearizes, a source-to-target primaries
matrix built from the two chromaticity sets applies, every set modelled being D65
so no chromatic adaptation is needed, and the target curve re-encodes. Both curves
are folded into 256-entry tables at negotiation, and the whole element stays
`no_std` on `mathf`.

PQ and HLG convert too. Linear 1.0 is the 203 cd/m2 HDR reference white of BT.2408,
which is what the SDR curves already put at code 255, so both halves of a
conversion meet on one scale. An HDR source headed for an SDR transfer is tone
mapped with the BT.2390 EETF, applied to the pixel's brightest channel in the PQ
domain and scaled onto all three so the hue holds, from the `hdr-peak-nits` source
peak, where an HLG source peaks at the 1000 cd/m2 display its OOTF is defined
against, down to reference white. The other direction encodes the light it has and
expands no highlights, and a source that peaks at reference white needs no tone map
at all, which is what makes an SDR round trip through PQ exact. Only the two steps
whose value is per pixel rather than per code, the HLG OOTF and the tone map, cost
a `powf`.

An untagged transfer or primaries cannot be converted away from, so the output
stays untagged on those two fields instead of being relabelled, while matrix and
range do convert from an untagged input since `yuv_conversion` resolves those. The
target comes from the negotiated output caps, or from the `colorimetry` property
when set, and a pin the property contradicts fails negotiation. An RGB layout
carries no matrix and no range, so those never reach its output caps.

A mid-stream refinement reaches these stages as the runner's `configure_pipeline`
call under the new `CapsChanged`: the CPU stages re-derive their table, `wgpusink`
rebuilds its NV12 blit pipeline, and the GL sinks respawn their worker, whose
program has the weights compiled in. `VideoConvert` takes the matrix from
whichever side carries YUV, the input, or the negotiated output caps for an RGB
input, and declares on its output caps what it wrote: a YUV target carries the
matrix and range of the YUV side, an RGB target neither, and the input's transfer
and primaries ride through. The `Compositor` mixes in input 0's colorimetry, fills
its background through that same conversion, limited-range black by default where
it used to write full-range JFIF, and announces the refined output caps when they
firm up after negotiation.

### The negotiation lifecycle

Because g2g enforces a sans-IO and asynchronous execution model, capability
negotiation happens in a deterministic handshake before any data frame
processing begins. This replaces GStreamer's query and event system with a
state-machine-driven future matrix.

```
                   Phase 1: Downstream Query (Caps Filter)
           Element A ───────────────────────────────────► Element B
                     "Here is what I can produce.
                      What can you handle?"

                   Phase 2: Upstream Selection (Fixate)
           Element A ◄─────────────────────────────────── Element B
                     "I choose NV12 at 1080p.
                      Allocate your buffers."

                   Phase 3 (rare): Re-fixation
           Element A ◄─────────────────────────────────── Element B
                     "Allocation failed at 1080p;
                      counter-propose 720p."
```

In Phase 1 the runner invokes `intercept_caps()` on the source, passing initial
configuration or upstream hardware constraints. Each element returns a `Caps`
value containing ranges or `Any` where parameters are flexible, and the
downstream peer intersects against its own internal capabilities and returns a
narrowed set.

In Phase 2 the final caps are fixated, so every `Dim` and `Rate` becomes `Fixed`,
and the fixated `Caps` travel back upstream via `configure_pipeline()`. Each
element allocates exact byte arrays or VRAM texture sizes, ensuring zero dynamic
allocations during steady-state streaming.

In Phase 3, if an element's allocation fails on a VRAM budget or a driver limit,
`configure_pipeline()` returns `ConfigureOutcome::ReFixate(Caps)` with a
counter-proposal and the runner restarts Phase 2 from that element. This bounded
backtrack avoids the GStreamer pattern of failing the entire pipeline on
allocation pressure.

The solver that runs this over a whole DAG, rather than one link at a time, is in
[DESIGN-caps.md](DESIGN-caps.md).

### The element traits

Transform and sink elements implement `AsyncElement`: packet in, 0 to N packets
out. Source elements have no input pad and instead implement `SourceLoop`, which
is called once and iterates internally until EOS. The two traits share the
`intercept_caps` and `configure_pipeline` semantics.

```rust
pub trait AsyncElement: ElementBound {
    type ProcessFuture<'a>: Future<Output = Result<(), G2gError>> + 'a
    where Self: 'a;

    /// Phase 1: Intersect proposed caps with internal capabilities.
    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError>;

    /// Phase 2/3: Fixate the agreed caps and initialize hardware buffer pools.
    /// Returns `ReFixate(caps)` to trigger Phase 3 with a counter-proposal.
    fn configure_pipeline(
        &mut self,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError>;

    /// Execution: process one input packet, pushing 0..N outputs into `out`.
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a>;
}

pub trait SourceLoop: ElementBound {
    type RunFuture<'a>: Future<Output = Result<u64, G2gError>> + 'a
    where Self: 'a;

    fn intercept_caps(&self) -> Result<Caps, G2gError>;
    fn configure_pipeline(&mut self, absolute_caps: &Caps)
        -> Result<ConfigureOutcome, G2gError>;

    /// Runs until EOS or error. Implementation MUST emit a final
    /// `PipelinePacket::Eos` before returning. Returns the count of
    /// `DataFrame` packets pushed (excluding `Eos`).
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a>;
}
```

`process` takes `&mut self` to accommodate stateful codecs, demuxers and parsers,
and the sink argument accommodates fan-out (demuxers), fan-in (batchers), and
elements that emit nothing until enough input has accumulated. Push is async so
elements await downstream capacity rather than failing fast on a full bounded
link.

The `ElementBound` marker is `Send` on multi-threaded targets and empty on
single-core ones, gated by the `multi-thread` cargo feature, because Embassy and
the WebGPU main-thread wasm executor do not require `Send` and many
hardware-handle types cannot satisfy it. `Sync` is intentionally not required:
`process` takes `&mut self` so concurrent calls are statically prevented, and
cross-task sharing happens through channels rather than shared references.

The GAT-based `AsyncElement` is not `dyn`-safe, so for plugin registries on `std`
targets `g2g-core` provides `DynAsyncElement`, a boxed adapter with a blanket
impl over every `AsyncElement`.

The graph runner does not pay that box per frame. `DynAsyncElement` carries
`drive_transform_arm` and `drive_sink_arm` hooks whose blanket impls monomorphize
the arm loop over the concrete element type, so `run_graph` awaits each element's
own `ProcessFuture` unboxed, one boxed arm future per run and none per frame. The
erased `process` remains for callers that drive an element directly through the
trait object, and an element opting into the zero-alloc steady state declares a
concrete non-boxed `ProcessFuture`.

The same treatment applies to the fan-in and fan-out arms:
`DynMultiOutputElement::drive_demux_arm` and `DynMultiInputElement`'s
`drive_muxer_arm`, `drive_muxer_arm_owned_tick` and `drive_fanin_sink_arm`
monomorphize the demux, muxer (arrival-order and PTS-ordered) and terminal fan-in
arms over their element, so a demux or muxer node also runs with no per-packet
box. A graph node built from a `&mut dyn` element keeps a boxed per-packet future,
since the concrete type is already erased, and the arm drives it through a private
`AsyncElement`, `MultiInputElement` or `MultiOutputElement` face over the trait
object. `no_std` graphs use concrete element types composed via a typed graph
builder, with no boxing and no virtual dispatch.

### The pad model

Pads are not a first-class type. An element's input and output endpoints are
encoded by which trait it implements and by the `&mut dyn OutputSink` parameter
shape. There is no `pub struct Pad`, no per-pad metadata, and no runtime
introspection.

| Topology | Trait | Input pad | Output pad |
| :--- | :--- | :--- | :--- |
| Source (0 to 1) | `SourceLoop` | none | `&mut dyn OutputSink` arg to `run()` |
| Transform / sink (1 to 0..N) | `AsyncElement` | `PipelinePacket` arg to `process()` | `&mut dyn OutputSink` arg to `process()` |
| Terminal sink | `AsyncElement` whose `process()` ignores `out` | as above | `NullSink` sentinel |

This is deliberate. GStreamer's `GstPad` is a runtime object because GStreamer
composes graphs from string-keyed plugin factories loaded at runtime, while g2g
composes typed graphs at compile time, so pad metadata lives in the trait
signatures. The cost is that fan-out (tee), fan-in (muxer) and demuxer-style
dynamic pads require additional trait variants rather than runtime pad-list
mutation, which [DESIGN-runtime.md](DESIGN-runtime.md) covers.

### Backpressure and scheduling

Every link between elements has an explicit `LinkPolicy`, configured at graph
construction time. The choice is per-link because a single pipeline may have
lossy preview branches and lossless recording branches sharing an upstream
source.

```rust
pub enum LinkPolicy {
    /// Block the upstream future until the channel has capacity.
    /// Lossless; raises latency under load.
    Block,
    /// Drop the oldest queued frame on downstream stall.
    /// Default for live camera sources.
    DropOldest,
    /// Drop the newest (incoming) frame on downstream stall.
    /// Use when temporal coherence matters more than freshness.
    DropNewest,
}
```

The leaky variants are implemented in the per-edge data-plane sink: under a full
channel, `DropNewest` discards the incoming frame and `DropOldest` evicts the
oldest queued frame to make room. Only `DataFrame`s are ever dropped, and control
packets (`CapsChanged`, `Segment`, `Flush`, `Eos`) always block, so a leaky link
never corrupts the stream. If a full queue holds only control packets,
`DropOldest` falls back to blocking.

Drops are pipeline-observable, never silent: `RunStats::frames_dropped` reports
the total, and `run_graph` applies each edge's policy set via `graph.link_with`.
This per-edge policy replaces GStreamer's explicit `queue` element, since every
link is already a bounded channel and every node already its own scheduling arm.

### The `G2gError` type

Errors are a single closed enum so element authors handle the full set
exhaustively. Hardware-specific failures carry a backend-tagged payload rather
than collapsing to a `String`.

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum G2gError {
    /// Phase 1 failure: no non-empty intersection between proposed
    /// upstream caps and this element's supported caps.
    CapsMismatch,
    /// Element received a DataFrame before configure_pipeline succeeded.
    NotConfigured,
    /// Phase 2 failure: caller should retry Phase 1 with the proposal
    /// returned in `ConfigureOutcome::ReFixate`.
    FixationFailed,
    /// Buffer pool exhausted; transient, retry after upstream drain.
    PoolExhausted,
    /// Memory domain handed to an element that cannot consume it
    /// (e.g. a CPU-only filter receiving a VulkanTexture).
    UnsupportedDomain,
    /// Backend-specific hardware/driver failure.
    Hardware(HardwareError),
    /// Pipeline is shutting down; element should drain and propagate Eos.
    Shutdown,
}
```

---

## Clock and timing

All timestamps in g2g are `u64` nanoseconds relative to a single pipeline
reference clock. Source elements map their hardware capture clock onto the
reference clock during `configure_pipeline`, and downstream elements treat
presentation timestamps as monotonic.

```rust
pub struct FrameTiming {
    /// Presentation timestamp, ns relative to the pipeline reference clock.
    pub pts_ns: u64,
    /// Decode timestamp. Differs from PTS for B-frames; equals PTS otherwise.
    pub dts_ns: u64,
    /// Nominal frame duration. 0 means "until next frame arrives".
    pub duration_ns: u64,
    /// Hardware capture timestamp in the source's native clock, preserved
    /// unchanged across the pipeline for end-to-end latency measurement.
    pub capture_ns: u64,
}

pub trait PipelineClock {
    fn now_ns(&self) -> u64;
}

/// Pipeline clock with async sleep. Sync sinks, paced sources, and jitter
/// buffers take `AsyncClock` so they can both observe and schedule against
/// time. `sleep_until_ns(d)` resolves immediately if `d <= now_ns()`.
pub trait AsyncClock: PipelineClock {
    type SleepFuture<'a>: Future<Output = ()> + 'a where Self: 'a;
    fn sleep_until_ns<'a>(&'a self, deadline_ns: u64) -> Self::SleepFuture<'a>;
}
```

A `pts_ns` of `FrameTiming::PTS_NONE` (`u64::MAX`, the value GStreamer spells
`GST_CLOCK_TIME_NONE`) marks a frame with no presentation time.
`FrameTiming::pts()` reads it as `None`, and `PresentationPacer` answers
`Pace::Now` for it without latching its anchor, so a sink presents the frame as it
arrives rather than holding it to a deadline or counting it a late drop.

Sink elements compare `pts_ns` against `now_ns()` to schedule presentation, and
`capture_ns` against `now_ns()` to report true glass-to-glass latency without
ambiguity about which clock domain a timestamp lives in. Backends provide
concrete implementations: a `WallClock` over `std::time::Instant` and
`tokio::time::sleep` for std targets, `embassy-time` for RTOS, and
`performance.now()` for wasm.

A free-running source feeding a sync sink is paced automatically by upstream
backpressure: the sink only consumes after `sleep_until_ns(pts)` resolves, which
throttles the channel, which throttles the source. No explicit source-side pacing
is required for sync playback.

Clock election, the audio master clock, PTP, presentation anchoring, the latency
fold and the QoS report are in [DESIGN-timing.md](DESIGN-timing.md).

---

## Target deployment environments

Because the core processing loop requires only `core` and `alloc`, deployment
profiles vary purely based on the top-level orchestration binary.

### Enterprise server node

- Runtime driver: Tokio multi-threaded runtime.
- Inter-element channels: bounded MPMC async channels (`flume`).
- Hardware interop: `cros-codecs` bitstream parsing feeding Linux kernel VAAPI and
  V4L2 drivers, producing `OwnedDmaBuf` handles.
- Cargo features: `multi-thread`, `std`.

### Deep embedded and bare-metal RTOS

- Target hardware: RTOS targets such as FreeRTOS, Zephyr, or microkernels.
- Runtime driver: the Embassy async executor, a single-threaded cooperative
  hardware timer loop.
- Inter-element channels: zero-allocation stack channels (`embassy-sync`).
- Hardware interop: fixed-memory DMA rings mapped to microcontroller video capture
  peripherals.
- Cargo features: none, the default `no_std + alloc`, or strict no-heap via
  `StaticBufferPool<_, N>` only.

The `no_std + alloc` core runs here directly: runner futures are executor-agnostic
and `ElementBound` is empty without `multi-thread`. The embedded surface is:

- `StaticBufferPool<T, N>` in `g2g-core`, pure `core` with no feature gate, a
  compile-time-sized zero-allocation pool yielding bounded mutable references
  checked via compile-time lifetimes. This is the strict no-heap pool the
  `Arc<Mutex<Vec<T>>>` `BufferPool` cannot serve.
- `EmbassyClock` (`embassy` feature) over `embassy-time`, the `no_std` analog of
  `WallClock`. The tick rate is selected at the feature and a HAL provides the time
  driver at link.
- `PacketChannel` and `EmbassySink` (`embassy-link` feature) over `embassy-sync`, a
  zero-allocation inter-task packet link. `SinglePacketChannel` (`NoopRawMutex`) is
  the single-executor default, and `SharedPacketChannel` (`CriticalSectionRawMutex`,
  hence `Sync`) is the variant that can live in a `static`, so spawned tasks reach
  it by `&'static`, since an executor's tasks take `'static` arguments.
- Two executor models over the same runner and element futures.
  `embassy-futures::block_on` drives a whole pipeline as one joined task, the
  bare-metal `fn main` entry the host tests use, and a real `embassy-executor` runs
  each element as an independently spawned task wired by static stack channels,
  with the scheduler interleaving them. The latter is host-verified via the std
  platform's `Executor::run_until`, which polls then returns on a completion flag
  instead of the diverging `run()` an embedded app's `fn main() -> !` calls, and a
  three-task source-transform-sink pipeline runs there with no HAL time driver.

`portable-atomic` backs the `metrics::LatencyHistogram` `AtomicU64` so `thumbv7em`
(Cortex-M) and `riscv32`, which lack 64-bit atomics, compile, and
`critical-section` makes the lock-based fallback interrupt-safe. The heap-free
build and the MCU element crates are in
[DESIGN-embedded.md](DESIGN-embedded.md).

### Browser sandbox

- Runtime driver: Web Workers spawned via `wasm-bindgen-futures`.
- Hardware interop: packets ingested via WebSockets or WebRTC data channels, parsed
  by browser hardware via the native WebCodecs JS API, and injected into WebGPU
  textures.
- Cargo features: `std`, since `wasm32-unknown-unknown` provides a usable `std`
  shim.

The browser target is `cfg(target_arch = "wasm32")` elements in `g2g-plugins`
behind the `web` feature, which implies `std`. The wasm bindings
(`wasm-bindgen`, `js-sys`, `web-sys`, `wasm-bindgen-futures`) are target-gated so
native builds never resolve them. No core change is needed: the runner future is
executor-agnostic, so `wasm_bindgen_futures::spawn_local` drives it on the browser
event loop, and wasm builds without `multi-thread`, so the `!Send` JS handle types
satisfy the empty `ElementBound`.

- `WasmClock`: `performance.now()` plus `setTimeout` sleep, the wasm analog of
  `WallClock`.
- `WebSocketSrc`: ingest over a browser `WebSocket`, parallel to `FileSrc` and
  `RtspSrc`.
- `WebRtcSrc` (`web` feature): ingest over a provided `RtcDataChannel`.
- `WebCodecsDecode` (`web-codecs` feature): wraps the browser `VideoDecoder`, with
  H.264 or H.265 Annex-B access units in and a `VideoFrame` copied to `System` RGBA
  out. The codec comes from the negotiated caps and picks the WebCodecs codec
  string built from the in-band SPS (`avc1.` or `hev1.`, ISO/IEC 14496-15 Annex
  E.3), and chunks stay Annex-B, which is what a config without a `description`
  means. The build needs `--cfg=web_sys_unstable_apis`.
- `CanvasSink`: presents decoded RGBA to an HTML canvas via the 2D context.
  `WebGpuCanvasSink` (`web-gpu` feature) is the zero-copy variant, importing the
  decoded `VideoFrame` as a `GPUExternalTexture` and sampling it in a render pass
  with no readback into wasm memory.

A complete in-browser glass-to-glass pipeline is
`WebSocketSrc -> H264Parse -> WebCodecsDecode -> CanvasSink`. The local gate for
the wasm build is
`cargo check --target wasm32-unknown-unknown -p g2g-plugins --features web`.

A whole graph can run inside a dedicated module worker: one wasm instance per
worker, the same single-threaded executor, no SharedArrayBuffer and no cross-origin
isolation. The page hands the worker an `OffscreenCanvas` from
`canvas.transferControlToOffscreen()`, which the sinks take through
`CanvasSink::from_offscreen_canvas` and
`WebGpuCanvasSink::from_offscreen_canvas` instead of looking an element id up in
`document`. A worker has neither `window` nor `document`, so `WasmClock` and the
WebGPU sink resolve `performance`, `setTimeout` and `navigator.gpu` off
`js_sys::global()`, cast to `Window` or `WorkerGlobalScope`. A transferred canvas
belongs to the worker for good, so a page switches graphs by reloading.

The chain
`WebSocketSrc -> WebCodecsDecode -> WebOrtDetect -> AnalyticsOverlay -> CanvasSink`
runs a real `.onnx` model over each decoded frame in the browser with CPU tensors.
`WebOrtDetect` lives in the `g2g-web` wasm leaf crate, not `g2g-plugins` which
cannot depend on `g2g-ml`, and splits the work so the pipeline stays one typed
graph: g2g owns preprocess, RGBA to `[1,3,640,640]` NCHW f32 with a whole-to-whole
resize, and postprocess, the same `g2g-ml` `DetectionPostprocess` channel-major
YOLOv8 decode and NMS the native chain uses, while a small `ort-shim.js`, a
wasm-bindgen module bundled into the pkg, owns only `session.run` over
onnxruntime-web.

onnxruntime-web runs single-threaded (`numThreads = 1`,
`executionProviders: ['wasm']`), so it needs no SharedArrayBuffer and the demo
serves from plain static HTTP with no COOP or COEP headers, and the `.onnx` is
fetched same-origin, the same model format the native ORT path loads. The chain
runs on the single browser thread via `run_linear_chain`. It is validated headless
by `tools/wasm-demo/headless/run-ortdetect.mjs`, a WebCodecs-capable Chromium,
against a committed deterministic fixture
(`tools/wasm-demo/fixtures/tiny-detect.onnx`, generated by `gen-tiny-detect.py`)
that plants two detections per frame: the model loads, each frame yields exactly
two decoded detections, and the overlay boxes render to the canvas. Finite and
unbounded sources both run clean end to end, `WebSocketSrc` detaches its callback
on every exit, and `tools/wasm-demo/headless/repro-unbounded.mjs` checks the
unbounded case.

### QNX

- Target hardware: QNX 8 on Cortex-A and x86-64 application processors, the
  reference platform for ISO 26262 and IEC 62304 automotive and medical systems.
- Portable surface: the pure-Rust core (`g2g-core` no-alloc and `alloc` /
  `runtime`, `g2g-mcu`, the `g2g-plugins` `no_std` baseline) compiles for
  `aarch64-` and `x86_64-unknown-nto-qnx800` with zero source changes, and Linux
  hardware elements are excluded by `target_os` gating. The spike and its build
  recipe are in `PORTABILITY.md`.

---

## The GStreamer bridge

To drive early adoption without forcing full system redesigns, g2g provides the
`g2g-bridge` wrapper library, compiled as a compliant C dynamic library
(`libgstglass2glass.so`), so an isolated g2g processing sub-graph executes inside a
legacy GStreamer pipeline.

```
┌────────────────────────────────────────────────────────┐
│               Legacy C GStreamer Pipeline              │
├────────────────────────────────────────────────────────┤
│  gst-rtsp-src ──► [ gst-glass2glass-bridge ] ──► qtmux │
│                          │                             │
│                          ▼                             │
│             ┌───────────────────────────┐              │
│             │   g2g Async Safe Core     │              │
│             │  (Wgpu Filter / Burn ML)  │              │
│             └───────────────────────────┘              │
└────────────────────────────────────────────────────────┘
```

The bridge intercepts the GStreamer pipeline's internal `GstBuffer`, extracts the
underlying OS hardware file descriptor (`GstDmaBufMemory`), wraps it as a
`g2g::OwnedDmaBuf` with a no-op close hook since GStreamer retains ownership of the
fd, and forwards execution to the Rust async engine. For the sync and async
impedance it runs a dedicated Tokio current-thread runtime on its own OS thread,
communicating with the synchronous GStreamer `chain` function via bounded channels,
which isolates GStreamer's threading model from the async future matrix without
blocking either side.

The implementation splits into a transport-agnostic core and a GStreamer-facing FFI
shell, so the hard part, the sync and async match and the lifecycle, is testable on
any host without a GStreamer dependency.

`BridgeGraph` (`g2g-bridge`) is the impedance core. It embeds a g2g sub-graph by
wrapping a user launch fragment as `appsrc ! <fragment> ! appsink`, parsing it
against the standard registry, and running it on a dedicated OS thread with its own
current-thread runtime. It exposes a synchronous API: `push(bytes, pts)` feeds the
embedded `appsrc`, `try_pull()` and `pull_blocking()` drain the `appsink`, and
`end_of_stream()`, `finish()` and `Drop` tear down. The `appsrc` and `appsink`
elements are the boundary this needs, synchronous external code feeding and
draining a running async graph with bounded-channel backpressure, so the bridge
reuses them rather than reinventing the channel plumbing. Per-instance channel
names are made collision-free with an atomic counter, since the named-feed
registries are process-global. On shutdown the drain handle is released before EOS
is signalled, so an un-drained graph cannot deadlock the join. It requires the
`multi-thread` feature, since the boxed graph must be `Send` to move to the run
thread, as in `g2g-capi`.

The GObject `GstBaseTransform` shell (`libgstglass2glass.so`, the `gstreamer`
feature) is a thin C shim (`csrc/gstglass2glass.c`, built by `build.rs` via
pkg-config and `cc`) that registers `glass2glass` as a real GStreamer element and
includes the actual GStreamer headers, so the GObject struct layouts are correct by
construction rather than hand-transcribed. It delegates to the C-ABI functions in
`src/ffi.rs`, which drive one `BridgeGraph` per instance: `set_caps` builds it from
the `fragment` property and the serialized sink and src caps, normalized so the
`(type)` annotations and whitespace GStreamer emits are stripped and g2g's caps
reader and launch DSL accept them, and `stop` destroys it.

The element handles both caps-preserving and caps- or size-changing fragments. A
preserving fragment, a wgpu effect, `videoflip`, or an ML preprocessor keeping the
pixel format, runs in place via `transform_ip`, the fast path. A fragment that
rescales or reformats declares its result through an `output-caps` property, and
the shell then advertises it via `transform_caps`, sizes the output buffer via
`get_unit_size` (`gst_video_info_from_caps`), and runs the out-of-place `transform`
from `inbuf` to `outbuf`. GstBaseTransform dispatches between the two by whether the
negotiated caps differ. `BridgeGraph` pins the sub-graph's trailing inline caps
filter to the output caps, equal to the input when preserving, which both enforces
the contract and gives a caps-driven transform a fixate target.

Zero-copy DMABUF import exists at the ingest side: `appsrc` accepts a
`MemoryDomain::DmaBuf` frame through `AppSrcFeed::push_dmabuf`,
`BridgeGraph::push_dmabuf` feeds it, and the C-ABI `g2g_bridge_push_dmabuf` `dup`s
a GStreamer buffer's dma-buf fd, so GStreamer keeps the original and g2g's
`OwnedDmaBuf` closes only the dup, and no pixel bytes are copied at the boundary.

The dma-buf-consuming element is `dmabuftowgpu` (`g2g-plugins`, the `dmabuf-wgpu`
feature), which imports a `MemoryDomain::DmaBuf` frame into a GPU-resident
`wgpu::Buffer` via `VK_EXT_external_memory_dma_buf` (Vulkan `from_raw_managed` into
`create_buffer_from_hal`), so a bridge fragment like
`dmabuftowgpu ! <wgpu compute>` runs the imported buffer on the GPU with no CPU
copy. It is validated on an RTX 3060 by exporting GPU memory as a dma-buf fd and
re-importing it: a discrete GPU binds a GPU-visible dma-buf, and a CPU or
vmalloc-backed one, a USB webcam or udmabuf, it cannot, where the element returns
`UnsupportedDomain` rather than a wrong result.

The shell's dma-buf round-trip is wired on both sides. The data path is a single
`generate_output` override rather than `transform` or `transform_ip`, so the output
buffer may differ from the input in size and memory kind. On input it checks
`gst_is_dmabuf_memory` and imports the fd via `g2g_bridge_push_dmabuf`, else maps
and copies bytes, and on output the pull returns either system bytes or a dma-buf,
the FFI `G2gOut` carrying a `kind` discriminant, and the shell wraps a dma-buf frame
back into a `GstBuffer` via `gst_dmabuf_allocator_alloc` with the fd dup'ed so the
g2g frame keeps its own. A full
`dma-buf in -> glass2glass(identity) -> dma-buf out` round-trip is validated with a
memfd-backed dma-buf (`tools/gst-bridge-dmabuf-smoke.sh`), and the system-memory
path is unchanged (`tools/gst-bridge-smoke.sh`).

The plugin entry points are subtle. rustc exports only its own `#[no_mangle]`
symbols from a cdylib and localizes anything pulled from a statically-linked C
archive, so a C `GST_PLUGIN_DEFINE` descriptor is invisible to GStreamer's loader.
The `GstPluginDesc` and the `gst_plugin_<name>_get_desc` and `_register` entry
points the loader resolves, by the `libgst<name>.so` filename, are therefore
authored in Rust (`src/ffi.rs`), pointing at the C `plugin_init` that does the
actual element registration. This is the same split `gst-plugins-rs` uses. Because
the feature links the system GStreamer, the shell is built and smoke-tested locally
(`tools/gst-bridge-smoke.sh`) rather than in CI.

### gstwrap, the reverse direction

The two layers above put a g2g stage inside a GStreamer app. `gstwrap`
(`g2g-plugins`, the `gstreamer` feature) does the opposite, hosting an unported
GStreamer element inside a g2g graph. This is the incremental-migration path in the
g2g-as-top-framework direction: adopt g2g now and keep the stages you have not
ported yet running as real GStreamer elements.

It is a normal g2g `AsyncElement` whose `element` property is a GStreamer element
description, such as `x264enc bitrate=4000` or `videoflip method=horizontal-flip`.
Internally it drives `appsrc ! <element> ! appsink` in a real GStreamer pipeline on
GStreamer's own streaming threads. `process` copies each `System` input frame into a
`GstBuffer` with `gst_app_src_push_buffer`, drains ready output non-blockingly with
`gst_app_sink_try_pull_sample` at 0 timeout, and on EOS flushes the element's
buffered frames.

The C interop mirrors the shell's: a small helper (`csrc/gstwrap_host.c`, built by
the crate's `build.rs` via pkg-config and `cc`) over the gstreamer-1.0 and
gstreamer-app-1.0 C API, driven from `src/gstwrap.rs` over a C ABI. Caps translate
with `Caps::to_gst_string()`, g2g caps into the appsrc's caps, and `parse_caps()`,
an `output-caps` property into the caps a reformatting element like an encoder or
`videoscale` produces, and a caps-preserving element declares nothing and couples
input to output.

The pipeline handle is `Send` because the appsrc and appsink APIs are MT-safe, the
element driving them from one runner task at a time. The data path uses system
memory, with a copy in and a copy out, like the shell's non-dma-buf path.

It is validated locally by
`cargo test -p g2g-plugins --features gstreamer --test gstwrap`, not in CI, hosting
a real `videoflip` and asserting the pixels come back flipped, and by running
`videotestsrc ! gstwrap element="videoflip method=horizontal-flip" ! fakesink`
through `parse_launch`. A multi-word element description reaches `gstwrap` from a
`gst-launch` line because the launch tokenizer is quote-aware, treating a `"..."`
region as one token so spaces and `!` inside a value are literal.
