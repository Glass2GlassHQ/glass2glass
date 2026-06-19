# Technical Specification: `glass2glass` (g2g)
**A Next-Generation, Hardware-First, Sans-IO, Asynchronous Multimedia Framework in Rust**

---

## 1. Executive Summary & Design Philosophy
`glass2glass` (`g2g`) is an open-source, ultra-low-latency multimedia graph framework written in 100% pure Rust. It is architected from the ground up to replace GStreamer in modern AI-driven, real-time embedded (RTOS), cloud ingestion, and web browser targets.

The project prioritizes minimizing **glass-to-glass latency** — the exact time elapsed between physical photon/audio capture and hardware presentation.

### The Four Pillars of `g2g`:
1. **Asynchronous Execution:** Every element is a cooperative async task (`Future`). No internal OS thread management; the framework is runtime-agnostic.
2. **Hardware-First & Zero-Copy:** Data remains in VRAM or unified memory domains via hardware handles (`DMABUF`, Vulkan Textures). CPU memory copies are treated as system faults.
3. **Modular Predictability (`no_std + alloc` + Sans-IO):** A `no_std + alloc` core allowing the exact same pipelines to execute on bare-metal microcontrollers, heavy multi-threaded servers, or WebAssembly (Wasm) targets. Network and protocol parsers use a pure **Sans-IO** design pattern, stripping I/O operations entirely out of the logic layer.
4. **First-Class Machine Learning Integration:** Tensor allocation, reshaping, and pipeline batching are built directly into the graph orchestration layer, executing in-flight on GPU memory.

---

## 2. Core Workspace Structure & Licensing
The project is structured as a Cargo Workspace to enforce clean boundaries between interfaces, standard elements, ML backends, and proprietary or highly protected enterprise suites.

| Crate Name | Purpose | Target Profile | Licensing |
| :--- | :--- | :--- | :--- |
| `g2g-core` | Core traits, `Frame` definitions, buffer pool allocators, clock model. | `no_std + alloc` | LGPL v2.1+ |
| `g2g-plugins` | Standard collection of source/sink/transform elements (`rtsp`, `wgpu`, `v4l2`). | `no_std + alloc` / `std` mixed | LGPL v2.1+ |
| `g2g-ml` | ML inference elements built on `burn` (Wasm/embedded) and `ort` (server). | `std` | LGPL v2.1+ |
| `g2g-bridge` | C-FFI dynamic library to embed `g2g` sub-graphs inside GStreamer pipelines. | `std` (`cdylib`) | LGPL v2.1+ |
| `g2g-enterprise` | High-value multi-stream async ML batchers and tensor schedulers. | `std` | AGPL v3 |

The `no_std + alloc` baseline is deliberate: it admits cooperative async executors (which need a heap for futures) and `Arc` reference counting, while still excluding the OS-dependent surface of `std`. Targets requiring strict no-heap allocation use the static `BufferPool` (§3.3) and avoid the `dyn`-safe element wrappers (§4.3).

---

## 3. Data Representation & Memory Subsystem

### 3.1 The Universal `Frame` Carrier
To avoid heavy C-style object allocation, media components flow through lock-free async channels as structured variants representing data packets, lifecycle signals, or negotiation hooks.

```rust
pub enum PipelinePacket {
    CapsChanged(Caps),
    DataFrame(Frame),
    Eos,
    /// Seek flush: discard in-flight and buffered data and reset position
    /// state. Unlike `Eos`, the stream resumes after a flush.
    Flush,
}

pub struct Frame {
    pub domain: MemoryDomain,
    pub timing: FrameTiming,
    /// Monotonically increasing per-source sequence number assigned at
    /// capture time and preserved unchanged across the pipeline. Used
    /// for drop detection and tracing, never for AV sync.
    pub sequence: u64,
    /// Reserved per-frame attachable metadata (the GstMeta /
    /// GstAnalyticsRelationMeta analog). Empty on construction.
    pub meta: FrameMetaSet,
}
```

**Per-frame metadata (`FrameMetaSet`).** `Frame` carries a reserved `meta`
side-channel for typed blobs that travel with the buffer (ML detection /
classification / tracking results, region-of-interest, reference timestamps).
It is gated behind the `metadata` cargo feature, **off by default**: when off it
is a zero-sized unit, so the `no_std` / RTOS baseline pays nothing per frame;
when on it is a `Vec<Box<dyn FrameMeta>>` where `FrameMeta` is a
`Debug + Send + Sync` trait. The field exists unconditionally so the metadata
system can be filled in without a breaking change to the `Frame` API. The
attach / iterate / propagate contract (GstMeta's `transform_func` / `copy_func`,
plus the `AnalyticsMeta` relation-graph layer) lands with the first
metadata-producing element; until then every frame's set is empty. Construct
frames via `Frame::new(domain, timing, sequence)` so future field additions do
not break call sites. The tee fan-out path gives each clone a fresh empty set
(deep COW propagation is deferred to the full build).

**Caps live on the link, not on the frame.** A `Frame` does not carry its
`Caps`. The current caps of a link are established by the most recent
`PipelinePacket::CapsChanged(Caps)` packet to arrive; every subsequent
`DataFrame` on that link is implicitly under those caps until the next
`CapsChanged` arrives. The runner guarantees `CapsChanged` is **ordered**
in the stream — it sits between the last old-caps `DataFrame` and the
first new-caps `DataFrame`, which is the load-bearing correctness
property for mid-stream format changes (§4.13.4).

See §4.4 for the definition of `FrameTiming` and the pipeline clock model.

### 3.2 Memory Domains
`g2g` treats system RAM as a fallback. Buffers track hardware descriptors to allow cross-process and cross-hardware zero-copy manipulation. Every hardware handle is owned: dropping the `MemoryDomain` releases the underlying file descriptor or GPU allocation.

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

impl OwnedDmaBuf {
    /// # Safety
    /// `fd` must be a valid DMABUF descriptor with no other owner.
    pub unsafe fn from_raw(fd: i32, stride: u32, offset: u32) -> Self { /* … */ }
    pub fn as_raw(&self) -> i32 { self.fd }
}
```

Vulkan and WebGPU handles follow the same RAII pattern, parameterised over a backend-specific allocator handle so the spec doesn't bake in a single binding crate.

### 3.3 Zero-Alloc Buffer Pools
Inside real-time or `no_std` loops, dynamic allocation during steady-state streaming is prohibited. Elements acquire pre-allocated slots from a bounded `BufferPool` and dropping the resulting handle automatically returns the buffer.

```rust
let pool = BufferPool::new_byte_pool(count, bytes);
let buf = pool.acquire().await;  // awaits if exhausted; backpressure-friendly
let mut frame = SystemSlice::from_pool(buf, frame_len);  // valid payload length
```

- **`no_std + alloc` environments (and `std`):** `BufferPool<T>` wraps `Arc<Mutex<Vec<T>>>` plus a `VecDeque<Waker>` of acquire waiters. `acquire().await` resolves the moment a `PooledBuffer` elsewhere is dropped. `try_acquire()` is the sync fast path for non-blocking contexts.
- **Strict `no_std` (no heap) environments:** a future variant will be statically sized at construction (e.g. `StaticBufferPool::<NV12, 8>::new()`) and yield bounded mutable references checked via compile-time lifetimes. The `Arc + Mutex + Vec` variant is unsuitable here because it relies on `alloc`.

The `SystemSlice` carrier transparently supports both ownership models: `SystemSlice::from_boxed(Box<[u8]>)` for one-off frames, `SystemSlice::from_pool(PooledBuffer<Box<[u8]>>, len)` for recycled frames (the buffer may exceed the frame, so the valid length is carried). Downstream elements treat the two identically.

---

## 4. Graph Orchestration & Capability Negotiation

### 4.1 Compile-Time and Runtime Caps
Traditional architectures rely on runtime string lookups for stream capabilities (e.g. `"video/x-raw, format=NV12"`). `g2g` enforces strongly typed structures.

```rust
#[derive(Clone, Debug, PartialEq)]
pub enum Caps {
    Video {
        format: VideoFormat,
        width: Dim,
        height: Dim,
        framerate: Rate,
    },
    Audio {
        format: AudioFormat,
        channels: u8,
        sample_rate: u32,
    },
    Tensor {
        dtype: TensorDType,
        shape: TensorShape,
        layout: TensorLayout,
    },
}

/// `Fixed` after Phase 2; `Range`/`Any` only legal during Phase 1.
pub enum Dim { Any, Range { min: u32, max: u32 }, Fixed(u32) }
pub enum Rate { Any, Range { min_q16: u32, max_q16: u32 }, Fixed(u32) }
```

The `Tensor` variant is first-class because ML elements (§5) negotiate caps the same way video elements do — they don't sit outside the graph model.

### 4.2 The Capability Negotiation Lifecycle
Because `g2g` enforces a Sans-IO and asynchronous execution model, capability negotiation happens in a clear, deterministic handshake before any data frame processing begins. This replaces GStreamer's complex query/event system with a simple, state-machine-driven future matrix.

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

**Phase 1 — Downstream Query (Intersection):** The runner invokes `intercept_caps()` on the source, passing initial configuration or upstream hardware constraints. Each element returns a `Caps` value containing ranges or `Any` where parameters are flexible. The downstream peer intersects against its own internal capabilities and returns a narrowed set.

**Phase 2 — Upstream Selection (Fixation):** Once an intersection is found, the final caps are fixated (all `Dim`/`Rate` values become `Fixed`). The fixated `Caps` travel back upstream via `configure_pipeline()`. Each element allocates exact byte arrays or VRAM texture sizes, ensuring zero dynamic allocations during steady-state streaming.

**Phase 3 — Re-fixation (rare):** If an element's allocation fails (VRAM budget, driver limit), `configure_pipeline()` returns `ConfigureOutcome::ReFixate(Caps)` with a counter-proposal. The runner restarts Phase 2 from that element. This bounded backtrack avoids the GStreamer pattern of failing the entire pipeline on allocation pressure.

### 4.3 The `AsyncElement` and `SourceLoop` Traits
Transform and sink elements implement `AsyncElement` — packet in, 0..N packets out. Source elements have no input pad and instead implement `SourceLoop`, which is called once and iterates internally until EOS. The two traits share `intercept_caps` / `configure_pipeline` semantics.

```rust
use core::future::Future;

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
    /// Mutable self accommodates stateful codecs, demuxers, and parsers;
    /// the sink accommodates fan-out (demuxers), fan-in (batchers), and
    /// elements that emit nothing until enough input has accumulated.
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

pub enum ConfigureOutcome {
    Accepted,
    ReFixate(Caps),
}

/// Output sink for both transform and source elements. `push` is async
/// so elements await downstream capacity rather than failing fast on a
/// full bounded link. Dyn-safe via a boxed future.
pub trait OutputSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;
}
```

#### Thread-safety bounds
The `ElementBound` marker is `Send` on multi-threaded targets and empty on single-core ones, gated by the `multi-thread` cargo feature. Embassy and the WebGPU/main-thread Wasm executor do not require `Send`, and many hardware-handle types cannot satisfy it.

```rust
#[cfg(feature = "multi-thread")] pub trait ElementBound: Send {}
#[cfg(feature = "multi-thread")] impl<T: Send> ElementBound for T {}
#[cfg(not(feature = "multi-thread"))] pub trait ElementBound {}
#[cfg(not(feature = "multi-thread"))] impl<T> ElementBound for T {}
```

Note: `Sync` is intentionally not required. `AsyncElement::process` takes `&mut self`, so concurrent calls are statically prevented; cross-task sharing happens through channels, not shared references.

#### Dynamic dispatch
The GAT-based `AsyncElement` is not `dyn`-safe. For plugin registries on `std` targets, `g2g-core` provides a boxed adapter:

```rust
#[cfg(feature = "std")]
pub trait DynAsyncElement: ElementBound {
    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError>;
    fn configure_pipeline(&mut self, absolute_caps: &Caps)
        -> Result<ConfigureOutcome, G2gError>;
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> core::pin::Pin<alloc::boxed::Box<
        dyn Future<Output = Result<(), G2gError>> + 'a
    >>;
}

#[cfg(feature = "std")]
impl<T: AsyncElement> DynAsyncElement for T { /* blanket boxed-future impl */ }
```

`no_std` graphs use concrete element types composed via a typed graph builder (no boxing, no virtual dispatch).

### 4.4 Pipeline Clock & Timing Model
All timestamps in `g2g` are `u64` nanoseconds relative to a single **pipeline reference clock**. Source elements map their hardware capture clock onto the reference clock during `configure_pipeline`; downstream elements treat presentation timestamps as monotonic.

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
/// buffers take `AsyncClock` rather than `PipelineClock` so they can both
/// observe and schedule against time. `sleep_until_ns(d)` resolves
/// immediately if `d <= now_ns()`.
pub trait AsyncClock: PipelineClock {
    type SleepFuture<'a>: Future<Output = ()> + 'a where Self: 'a;
    fn sleep_until_ns<'a>(&'a self, deadline_ns: u64) -> Self::SleepFuture<'a>;
}
```

Sink elements compare `pts_ns` against `now_ns()` to schedule presentation, and `capture_ns` against `now_ns()` to report true glass-to-glass latency without ambiguity about which clock domain a timestamp lives in. Backends provide concrete implementations: a `WallClock` (`std::time::Instant` + `tokio::time::sleep`) for std targets, `embassy-time` for RTOS, performance.now() for Wasm.

A free-running source feeding a sync sink is paced automatically by upstream backpressure (§4.5): the sink only consumes after `sleep_until_ns(pts)` resolves, which throttles the channel, which throttles the source. No explicit source-side pacing is required for sync playback.

### 4.5 Backpressure & Scheduling
Every link between elements has an explicit `LinkPolicy`, configured at graph construction time. The choice is per-link because a single pipeline may have lossy preview branches and lossless recording branches sharing an upstream source.

```rust
pub enum LinkPolicy {
    /// Block the upstream future until the channel has capacity.
    /// Lossless; raises latency under load.
    Block,
    /// Drop the oldest queued frame on downstream stall.
    /// Default for live camera sources.
    DropOldest,
    /// Drop the newest (incoming) frame on downstream stall.
    /// Use when temporal coherence matters more than freshness
    /// (e.g. driver-assistance ML where stale-but-coherent beats torn).
    DropNewest,
}
```

The leaky variants are implemented in the per-edge data-plane sink: under a full channel, `DropNewest` discards the incoming frame and `DropOldest` evicts the oldest queued frame to make room. Only `DataFrame`s are ever dropped, control packets (`CapsChanged` / `Segment` / `Flush` / `Eos`) always block, so a leaky link never corrupts the stream; if a full queue holds only control packets, `DropOldest` falls back to blocking. Drops are pipeline-observable, never silent: `RunStats::frames_dropped` reports the total, and `run_graph` applies each edge's policy set via `graph.link_with`. This per-edge policy replaces GStreamer's explicit `queue` element, every link is already a bounded channel and every node already its own scheduling arm.

### 4.6 The `G2gError` Type
Errors are a single closed enum so element authors handle the full set exhaustively. Hardware-specific failures carry a backend-tagged payload rather than collapsing to a `String`.

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

### 4.7 Pad Model: Implicit by Trait Shape
Pads are not a first-class type. An element's input and output endpoints are encoded by which trait it implements and by the `&mut dyn OutputSink` parameter shape; there is no `pub struct Pad`, no per-pad metadata, no runtime introspection.

| Topology | Trait | Input pad | Output pad |
| :--- | :--- | :--- | :--- |
| Source (0→1) | `SourceLoop` | — | `&mut dyn OutputSink` arg to `run()` |
| Transform / sink (1→0..N) | `AsyncElement` | `PipelinePacket` arg to `process()` | `&mut dyn OutputSink` arg to `process()` |
| Terminal sink | `AsyncElement` whose `process()` ignores `out` | as above | `NullSink` sentinel |

This is deliberate. GStreamer's `GstPad` is a runtime object because GStreamer composes graphs from string-keyed plugin factories loaded at runtime; `g2g` composes typed graphs at compile time, so pad metadata lives in the trait signatures. The cost is that fan-out (tee), fan-in (muxer), and demuxer-style dynamic pads require additional trait variants rather than runtime pad-list mutation — see §4.10.

### 4.8 Dynamic Graph Reconfiguration

#### 4.8.1 Two-Layer Graph API
`g2g` exposes two graph APIs sharing the same element traits, the same negotiation lifecycle, the same `PipelinePacket` variants, and the same runner primitives. Only graph construction and slot mutation differ.

- **Static typed graph** — compile-time topology via tuple types; no `dyn`; zero-cost. Right for embedded / RTOS / static cloud pipelines.
- **Type-erased dynamic graph** — boxed elements (`Box<dyn DynAsyncElement>`) held in `ElementSlot`s and `BranchSlot`s, swappable at runtime. Right for cloud ingestion, desktop applications, and anything that needs runtime topology evolution.

#### 4.8.2 `ElementSlot` — Lock-Free Single-Element Swap
The dynamic graph holds elements in `arc_swap::ArcSwap<Box<dyn DynAsyncElement>>` cells:

```rust
let new_element = SomeTransform::new();
new_element.configure_pipeline(&caps)?;
slot.handle.store(Arc::new(Box::new(new_element)));
```

Frames mid-`process()` against the old element complete naturally; the next push observes the new element. Cost: one atomic store plus the new element's `configure_pipeline()` work. No drain, no pipeline stall.

This is the primary response to a Phase 3 `ReFixate` or a mid-stream `Reconfigure` signal: replace the affected slot's contents, do not rebuild the graph.

#### 4.8.3 `BranchSlot` — Multi-Element Sub-Graph Swap
A branch with one logical input and one logical output is structurally an element. `BranchSlot` is the multi-element analog of `ElementSlot`, with the swap trade-off made explicit at the type level:

```rust
pub struct BranchHandle<I, O> {
    input_tx: LinkSender<I>,
    output_rx: LinkReceiver<O>,
    tasks: Vec<JoinHandle<()>>,
}

pub struct BranchSlot<I, O> {
    handle: arc_swap::ArcSwap<BranchHandle<I, O>>,
    policy: SwapPolicy,
}

pub enum SwapPolicy {
    /// Flip input routing; in-flight frames inside the old branch's
    /// internal channels are discarded. Zero latency; bounded frame loss.
    /// Right for stateless filters (color grade, debug overlay).
    Immediate,

    /// Flip input routing; wait for old branch to drain its in-flight
    /// frames before exposing the new branch's output to the consumer.
    /// Zero loss; pays the old branch's pipeline depth in latency.
    DrainOld,

    /// Both branches consume in parallel for a brief overlap window;
    /// the merger cuts over at the named signal (next IDR, next segment
    /// boundary, etc.). Zero loss, zero per-frame stall; brief duplicated
    /// compute during the overlap.
    ShadowWarm { cutover: CutoverSignal },
}
```

Static-graph users at the embedded layer never instantiate `BranchSlot` and don't pay for any of this machinery.

#### 4.8.4 Router, Gate, Merger Primitives
A `Router` is a 1-to-N transform that reads an atomic discriminator per frame and pushes the frame to exactly one of its outputs. A `Gate` is a 1-to-1 transform that reads an atomic boolean and either forwards or discards each frame. A `Merger` is an N-to-1 transform that reads from one of its inputs, switching on a discriminator. Together they cover branch enable/disable, A/B switching, and the routing + cutover halves of `ShadowWarm`. These primitives form the foundation of the dynamic-graph layer.

### 4.9 GStreamer Dynamic-Feature Mapping
`g2g`'s dynamic surface is intended to be a superset of GStreamer's dynamic capabilities, achieved through a different set of primitives.

| GStreamer feature | `g2g` mechanism |
| :--- | :--- |
| Element hot-swap | `ElementSlot::swap` (ArcSwap) |
| Branch insertion / removal | `BranchSlot::swap` with `SwapPolicy::Immediate` |
| Branch enable / disable, A/B switching | `Router` + `Gate` |
| Bin nesting | `BranchSlot` is structurally a bin |
| Mid-stream caps change | `PipelinePacket::CapsChanged` + runner cascade |
| Allocation pressure backtrack | Phase 3 `ConfigureOutcome::ReFixate` |
| Bitrate switching | `BranchSlot` + `ShadowWarm { cutover: NextSegment }` |
| Codec change at keyframe | `BranchSlot` + `ShadowWarm { cutover: NextKeyframe }` |
| Demuxer dynamic-pad (bounded N) | Pre-allocated dark slots, populated on discovery |
| Live source push from app code | Direct `LinkSender::send` from external task |
| Multi-pipeline isolation | One pipeline per task tree; no shared mutable state |
| Async messages (bus) | Pipeline-level mpmc message channel |
| Latency aggregation query | Upstream-traveling query primitive |
| Allocation query | Downstream-proposed allocator handoff |
| Probes (`pad_block`, `pad_idle`) | `LinkInterceptor` trait registered on a slot |
| Seek with FLUSH | `PipelinePacket::Flush` + runner drain handling |
| Live clock distribution | `AsyncClock` provider election |
| EOS aggregation across N inputs | Fan-in / muxer |

#### 4.9.1 Differences Forced by Rust Ownership
GStreamer relies on parent ↔ child reference cycles via GObject reference counting plus signal callbacks. Rust's strict ownership doesn't allow that shape. Equivalent functionality lives in **message channels** instead of direct back-references: a child element that needs to notify its parent posts a bus message; the parent reads it. Functionally identical; structurally cleaner; no `unref` ordering hazards. Similarly, GStreamer's `gst_pad_link()` performs runtime pointer manipulation; the `g2g` equivalent — moving the receive end of a channel — requires explicit ownership transfer under a brief gate hold. Same outcome, more honest about what's happening.

#### 4.9.2 Capabilities That Fall Out For Free
- **No silent caps mismatch at runtime**: exhaustive typed `Caps` enum, `match` checked at compile time. GStreamer's string-keyed caps regularly fail at runtime with `not-negotiated`.
- **Deterministic shutdown**: Rust drop order is a topological walk; no leaked refs holding pipelines alive forever.
- **No GIL / no global state**: independent pipelines spawn on the same async runtime with zero coordination cost.
- **Memory safety across hot-swap**: ArcSwap guarantees no use-after-free when an element is replaced while a frame is in flight. GStreamer's `pad_block` / `pad_unlink` choreography is famously bug-prone here.

#### 4.9.3 The Single Architectural Trade-Off
Pre-allocated "dark slots" handle the common dynamic-pad case (a demuxer with at-most-N tracks). If an application genuinely needs runtime-growable pad count without an upper bound — e.g., a session router that accepts new RTP streams indefinitely — the dynamic layer uses a `Slab<Slot>` instead of a fixed array. Per-push slot lookup becomes one extra indirection. Since this only matters inside the already-type-erased dynamic layer, the cost is in the noise.

### 4.10 Architectural Tracks

The framework is built along five interlocking tracks. The spec sections that
follow describe each track's current architecture.

| Track | Section | Summary |
| :--- | :--- | :--- |
| Receive | §4.11, §4.12a/b | Network + capture sources and hardware decoders (RTSP, raw RTP ingest with jitter buffer + RTCP/NACK, V4L2 capture, file, fMP4, software/VAAPI/MF/NVDEC decoders). |
| Display & egress | §4.11.5, §4.12 | GPU-resident presentation sinks and outbound RTP packetizers. |
| Negotiation | §4.13 | Distributed CSP caps solver with per-link assignment and structured failure. |
| ML | §5 | Inline GPU tensor preprocess and inference (Burn / ORT). |
| Deployment | §6 | Cloud / embedded / browser orchestration over a single core. |

Open work (planned tracks, deferred items, follow-ups) lives in
[DESIGN_TODO.md](DESIGN_TODO.md).

### 4.11 Hardware Decoder Elements

The layers `RtspSrc → H264Parse` cover encoded-bitstream processing
(mux, re-stream, record). Decoded-pixel output — required for ML inference,
display, and colour-space conversion — uses a decoder `AsyncElement` that
accepts `Caps::CompressedVideo { codec: H264 | H265, .. }` and emits
`Caps::RawVideo { format: Nv12 | I420, .. }` backed by `MemoryDomain::System`,
`MemoryDomain::DmaBuf`, `MemoryDomain::Cuda`, or `MemoryDomain::D3D11Texture`
depending on backend.

#### 4.11.1 cros-codecs (Linux VAAPI)

`VaapiH264Dec` (`g2g-plugins/src/vaapidec.rs`, feature `vaapi`, `cfg(target_os = "linux")`) is built on `cros-codecs` (`vaapi` backend). The crate is maintained by the ChromeOS team and exposes a stateless decoder framework that parses H.264 bitstreams and manages the DPB; the actual decode runs on the GPU through libva.

- **Input caps:** `Caps::CompressedVideo { codec: H264, .. }` — `intercept_caps` intersects with H.264 and rejects everything else.
- **Output caps:** `Caps::RawVideo { format: Nv12, .. }` backed by `MemoryDomain::System` (CPU copy out of the GBM-allocated surface).
- **Frame allocation:** `GbmDevice::open("/dev/dri/renderD128")` (configurable via `VaapiH264Dec::with_render_node`) allocates `GenericDmaVideoFrame` surfaces; the decoder's allocator callback returns one per output picture.
- **Format negotiation:** the first `decode()` call surfaces `DecodeError::CheckEvents`; the element drains events, picks up the SPS-derived `StreamInfo` on `FormatChanged`, and re-feeds the same NAL.
- **Flush:** forwards `decoder.flush()` and propagates `PipelinePacket::Flush` downstream.
- **EOS:** flushes the decoder, drains the DPB, emits `Eos`.
- **Thread safety:** `libva::Display` is `Rc<Display>` and therefore `!Send`; `unsafe impl Send` is justified by the runner's ownership model (move-not-share).

```text
H.264 Annex-B  (MemoryDomain::System)
       │
       ▼
┌───────────────────────────────┐
│  VaapiH264Dec                 │
│   cros-codecs StatelessDecoder│
│   <H264, VaapiBackend<...>>   │
│   DPB + B-frame reorder       │
└───────────┬───────────────────┘
            │  NV12 row-copied out of GBM surface
            ▼
    downstream AsyncElement
```

#### 4.11.2 Windows Media Foundation Transform (MFT)

`MfDecode` (`g2g-plugins/src/mfdecode.rs`, feature `mf-decode`, `cfg(target_os = "windows")`) wraps `CLSID_MSH264DecoderMFT` via `windows-rs` using an MTA COM apartment.

- **Input caps:** `Caps::CompressedVideo { codec: H264, .. }` — rejects anything else at `intercept_caps`.
- **Output caps:** `Caps::RawVideo { format: Nv12, .. }` backed by `MemoryDomain::System` (CPU copy out of the MFT output buffer).
- **Flush:** forwards `MFT_MESSAGE_COMMAND_FLUSH` and propagates `PipelinePacket::Flush` downstream.
- **EOS:** sends `MFT_MESSAGE_COMMAND_DRAIN` to flush the B-frame reorder buffer before emitting `Eos`.
- **Thread safety:** `!Send` by default (COM); `unsafe impl Send` justified by MTA free-threading — the MS H.264 decoder MFT is callable from any MTA thread without marshaling.

A sibling `MfEncode` (feature `mf-encode`) wraps `CLSID_MSH264EncoderMFT` with `MF_LOW_LATENCY` set (no B-frames) and converts `Caps::RawVideo { format: Nv12 }` to `Caps::CompressedVideo { codec: H264 }`, Annex-B framed. `MfAacEncode` / `MfAacDecode` (feature `mf-aac`) cover the AAC audio path.

#### 4.11.3 ffmpeg / libavcodec

`FfmpegH264Dec` (`g2g-plugins/src/ffmpegdec.rs`, feature `ffmpeg`, `cfg(target_os = "linux")`) wraps system libavcodec via `ffmpeg-next`. Selectable backend:

| `Backend` variant | Codec opened | Output domain | Notes |
| :--- | :--- | :--- | :--- |
| `Software` | `h264` | `System` | Software decode; broadest hardware coverage. |
| `NvdecCuvid` | `h264_cuvid` | `System` | GPU decode, host copy. Pairs with CPU sinks. |
| `NvdecCuda` | `h264` + `AV_HWDEVICE_TYPE_CUDA` | `Cuda` | Zero-copy device-memory output; see §4.11.5. |

- **Input caps:** `Caps::CompressedVideo { codec: H264, .. }`.
- **Output caps:** `Caps::RawVideo { format: I420 | Nv12, .. }`. `I420` is the libavcodec native 8-bit 4:2:0 format; `Nv12` is selectable via `FfmpegH264Dec::with_output_format(OutputFormat::Nv12)`, produced by a U/V interleave with no swscale. `YUVJ420P` is accepted with the same plane layout; `YUV444P` / `YUVJ444P` are accepted with the chroma planes box-averaged down to 4:2:0. Other pixel formats are rejected with `CapsMismatch`.
- **Feed loop:** one access unit per `Packet::copy`; PTS is forwarded verbatim (libavcodec echoes it back on the decoded frame); `send_packet()` then `receive_frame()` drained until `EAGAIN`.
- **Flush / EOS:** `decoder.flush()` on `PipelinePacket::Flush`; `send_eof()` + final drain before forwarding `Eos`.
- **Thread safety:** `ffmpeg::decoder::Video` wraps a raw `*mut AVCodecContext` and is `!Send` by default; `unsafe impl Send` is justified by the same ownership-transfer argument as `MfDecode` and `VaapiH264Dec`.

#### 4.11.4 End-to-End RTSP Pipeline

The complete glass-to-glass receive pipeline is:

```
RtspSrc ──► H264Parse ──► [decoder] ──► [ML / display / encode]
(System / H264)            (System / DmaBuf / Cuda / D3D11Texture; NV12)
```

| Platform | Decoder element | Feature | Output |
| :--- | :--- | :--- | :--- |
| Linux software | `FfmpegH264Dec` (`Software`) | `ffmpeg` | `System` / I420 |
| Linux + NVIDIA | `FfmpegH264Dec` (`NvdecCuvid` / `NvdecCuda`) | `ffmpeg` + `cuda` | `System` / `Cuda` / NV12 |
| Linux + VAAPI | `VaapiH264Dec` | `vaapi` | `System` / NV12 |
| Windows | `MfDecode` | `mf-decode` | `System` / NV12 |

`RtspSrc` connects via `retina` using standard RTSP/RTP over TCP, negotiates H.264 with `FrameFormat::SIMPLE` (Annex-B) or accepts AVCC framing detected per buffer. The first SPS the parser sees provides geometry; framerate is recovered from the VUI `timing_info` (`time_scale / (2 * num_units_in_tick)`) when present, or left as `Rate::Any` when the VUI is absent.

#### 4.11.5 Zero-copy NVDEC → CUDA → GPU display

`Backend::NvdecCuvid` decodes on the GPU but copies NV12 back to system memory;
the glass-to-glass floor is then dominated by the PCIe round-trip plus the
sink's CPU NV12→XRGB convert. The CUDA-resident path keeps decoded NV12 in
device memory end-to-end so a GPU consumer (display) takes the handoff without
a host round-trip.

**Memory domain.** `MemoryDomain::Cuda(OwnedCudaBuffer)` lives in `g2g-core`,
platform-agnostic. `OwnedCudaBuffer` carries the two NV12 plane device
pointers (luma Y, interleaved chroma UV), row pitches, dims, the `CUcontext`,
and a boxed `CudaKeepAlive` owner. Core never links CUDA: the producing
element supplies the owner as a trait object, and dropping the buffer releases
the backing allocation. `AllocationParams::cuda(...)` makes
`MemoryDomainKind::Cuda` a cross-element pool domain in the allocation
negotiation (§4.13).

**Decoder.** `Backend::NvdecCuda` opens the generic `h264` codec with an
`AV_HWDEVICE_TYPE_CUDA` device and a `get_format` hook selecting
`AV_PIX_FMT_CUDA`; the resulting `AVFrame` is the keep-alive that owns the
device pointers wrapped into `OwnedCudaBuffer`.

**Consumer: CUDA↔GL interop, not dma-buf.** CUDA can only export VMM-allocated
memory (`cuMemCreate` / `cuMemMap`) to a dma-buf fd, and NVDEC decoder frames
come from libavcodec's CUDA hwframe pool (not VMM); the NVIDIA proprietary
driver also doesn't import foreign dma-bufs reliably through `nvidia-drm`.
Presentation therefore uses CUDA↔GL interop — the path GStreamer's `nvcodec`
+ `glimagesink` and NVIDIA's `FramePresenterGL` sample take:

1. Create an EGL context on the display surface.
2. Register a GL texture with `cuGraphicsGLRegisterImage` once.
3. Per frame: `cuGraphicsMapResources`, `cudaMemcpy2D` (device→device,
   honouring source pitch) the NV12 planes into the GL resource,
   `cuGraphicsUnmapResources`.
4. Sample Y + interleaved UV in a fragment shader (BT.601/709 limited range),
   present via `eglSwapBuffers`.

This is not strictly zero-copy (one device→device copy into the GL texture)
but it removes the PCIe round-trip and the CPU colour convert.

**Elements.**
- `CudaDownload` (`cuda` feature) is an `Identity(NV12)` transform that
  copies a `MemoryDomain::Cuda` frame to `MemoryDomain::System` via
  device→host `cuMemcpy2D`. Negates the latency win but lets a `NvdecCuda`
  stream reach the existing CPU sinks for correctness and bring-up.
- `CudaGlSink` (`cuda-gl` feature, Linux + NVIDIA) holds an EGL context on a
  Wayland surface (`wl_egl_window` from SCTK), a `glow` GL ES 3 program with
  the two NV12 textures, and the per-frame map/copy/unmap render loop via
  the CUDA-GL interop entry points.

**CUDA bindings: hand-rolled FFI.** `cudarc` has no CUDA-GL interop wrappers
(`cuGraphicsGLRegisterImage` and friends), and its safe API assumes it owns
the `CudaContext`, whereas the `CUcontext` is created and owned by ffmpeg's
hwdevice and carried on `OwnedCudaBuffer`. The needed surface is small:
`cuCtxPushCurrent_v2` / `_PopCurrent_v2`, `cuMemcpy2D_v2`, and the GL-interop
quartet `cuGraphicsGLRegisterImage` / `cuGraphicsMapResources` /
`cuGraphicsSubResourceGetMappedArray` / `cuGraphicsUnmapResources`. The
plugin links `libcuda` directly.

### 4.12 Live Egress

The receive path (§4.11) has an inverse: encoded video out over RTP. The
protocol logic is Sans-IO (§1): a pure packetizer produces the RTP packets
and a thin sink does the UDP I/O.

- `RtpH264Packetizer` (`rtppay.rs`) implements RFC 3550 + RFC 6184. An H.264
  access unit becomes a single-NAL RTP packet if the NAL fits the MTU, else
  FU-A fragments. The marker bit lands on the access unit's last packet;
  sequence numbers increment across packets and calls; one RTP timestamp per
  access unit. Pure `no_std` logic, host-testable.
- `UdpSink` (`udpsink.rs`, `udp-egress` feature) is an `AsyncElement` sink
  that drives the packetizer over each Annex-B access unit and sends the RTP
  packets to a destination on a tokio `UdpSocket`. The RTP timestamp is the
  90 kHz image of `FrameTiming::pts_ns`; sequence numbers and the per-AU
  marker bit come from the packetizer. `with_rtp(pt, ssrc)` and
  `with_max_payload(mtu)` configure the flow. It also keeps a bounded history of
  recently sent packets and honors receive-side RTCP NACK by retransmitting them
  (`with_retransmit(enabled, capacity)`); see the receive-side feedback loop in
  §4.12b.

### 4.12a Live Capture (V4L2)

`V4l2Src` (`v4l2src.rs`, `v4l2` feature, Linux-only) is the first real capture
source: it streams packed **YUYV** (4:2:2, the near-universal UVC output) off a
`/dev/videoN` device via V4L2 mmap streaming I/O, wrapping the pure-Rust `v4l`
crate (no libv4l C dependency). `VideoConvert` unpacks YUYV to a planar / RGB
target (§3.1 raw formats), so the canonical chain is
`V4l2Src -> VideoConvert(Yuyv -> Nv12) -> sink`.

Two design points carry the element:

- **Blocking ioctls off the async path.** V4L2 dequeue is a blocking ioctl, so
  capture runs on a dedicated `std::thread` that owns the device and the mmap
  stream (which borrows the device) and copies each frame's payload into a
  bounded channel. The `SourceLoop::run` future drains that channel and pushes
  `DataFrame`s. The channel bound (`BUFFER_COUNT`) applies backpressure: the
  capture thread blocks rather than growing memory when the pipeline falls
  behind. The source reports a live `LatencyReport` of one frame period.
- **Up-front format negotiation, re-open for capture.** `intercept_caps` opens
  the device, sets YUYV at the requested geometry, and reads back what the
  driver actually chose (it may snap to a supported mode); the probe device is
  then dropped. The capture thread re-opens the device under that exact format.
  Keeping no device handle in the struct between negotiation and `run`
  sidesteps `Send` / borrow entanglement with the stream. Errors surface as
  `G2gError::Hardware(HardwareError::V4l2(errno))`.

MJPEG-mode UVC and format-flexible negotiation (the source fixes YUYV today)
are follow-ups (DESIGN_TODO).

### 4.12b Live Ingress (UDP / RTP)

`UdpSrc` (`udpsrc.rs`, `udp-ingress` feature) is the receive-side inverse of
`UdpSink` (§4.12): it receives RTP on a tokio `UdpSocket` and depayloads H.264
into Annex-B access units pushed downstream as `CompressedVideo` H.264, so the
canonical chain is `UdpSrc -> FfmpegH264Dec -> sink`. The I/O is async, so
unlike `V4l2Src` it needs no capture thread.

The protocol logic is Sans-IO (§1), mirroring the egress split: `rtpdepay.rs`'s
`RtpH264Depayloader` is a pure, `no_std`, host-testable function that inverts
`RtpH264Packetizer`. Single-NAL and STAP-A payloads pass through; FU-A fragments
reassemble (the original NAL header is rebuilt from the FU indicator's F|NRI and
the FU header's type); the RTP marker bit closes an access unit. A sequence-
number gap drops the in-flight reassembly so loss or reorder never welds two
access units together.

**Receive-side resilience (jitter buffer + RTCP + NACK).** Between the socket and
the depayloader sits a Sans-IO jitter buffer (`rtpjitter.rs`,
`RtpJitterBuffer`): it orders packets by an *extended* sequence number (the
16-bit RTP sequence unrolled to a monotonic counter, so wraparound is handled),
releases them in order, holds a gap only until its predecessors fill or a
bounded deadline elapses (then declares loss), and drops duplicates / too-late
packets. RTCP (`rtcp.rs`, Sans-IO RFC 3550 SR/RR/BYE + RFC 4585 Generic NACK,
plus `ReceptionStats` for loss fraction / cumulative loss / interarrival jitter)
runs RTP/RTCP-muxed on the one socket (RFC 5761): `UdpSrc` sends periodic
receiver reports and emits a NACK for each detected gap, and `UdpSink` honors
those NACKs by retransmitting from its send history (§4.12). A retransmit
arriving inside the jitter hold window heals the gap before it is declared lost,
so the loop recovers packet loss end to end. RFC 4588 RTX (a distinct
retransmission payload; plain same-stream resend is used today) and FEC are the
remaining receive-side items.

This is **raw RTP** with no RTSP/SDP, so there is no out-of-band stream
description: the output geometry is a declared hint (`with_video_size` /
`with_framerate`), and since H.264 carries its real dimensions in the SPS a
downstream decoder re-derives and corrects them. SDP/SPS-driven caps discovery is
a follow-up; `RtspSrc` (via `retina`) already covers the RTSP case with its own
jitter buffer (§4.11.4).

The remaining capture/ingress breadth — `HttpSrc` (gated on a byte-stream caps
+ consumer; see DESIGN_TODO) and a `uridecodebin`-equivalent URI → source layer
over the autoplug registry — is tracked in DESIGN_TODO.

### 4.13 CSP Caps Negotiation

The handshake sketched in §4.2 is the *interface* contract. The underlying
mechanism is a **distributed constraint-satisfaction problem (CSP)**: each
element declares a constraint over `(input, output)` caps; a solver finds an
assignment over every link in the graph that satisfies all constraints,
ranked by preferences; the assignment becomes the per-link `Caps` the runner
hands each element via `configure_pipeline`.

This subsumes GStreamer's pad-by-pad negotiation: the solver runs once over
the whole graph (or over an affected subgraph on a mid-stream change),
returns structured failure when no assignment exists, and trades pad-query
round-trips for direct function calls.

#### 4.13.1 CapsSet and the constraint enum

```rust
/// A set of acceptable caps descriptions, ordered by preference.
pub struct CapsSet { alternatives: Vec<Caps> }
impl CapsSet {
    pub fn one(caps: Caps) -> Self;
    pub fn intersect(&self, other: &Self) -> Self;
    pub fn fixate(&self) -> Option<Caps>;
}

pub enum CapsConstraint {
    Accepts(CapsSet),                             // sink-shape
    AcceptsAny,                                   // wildcard sink (probes, fakes)
    Produces(CapsSet),                            // source-shape
    Identity(CapsSet),                            // pass-through transform
    IdentityAny,                                  // wildcard pass-through
    Mapping(Vec<(CapsSet, CapsSet)>),             // explicit (in, out) pairs
    DerivedOutput(Box<dyn Fn(&Caps) -> CapsSet>), // output as function of input
}
```

`Caps` is the *fixed* description used at runtime (carried by
`PipelinePacket::CapsChanged`, handed to `configure_pipeline`); `CapsSet`
is the negotiation-time vocabulary.
`Caps` is split into compressed and raw at the type level:

```rust
pub enum Caps {
    CompressedVideo { codec: VideoCodec, extras: CodecExtras },
    RawVideo { format: RawVideoFormat, width: Dim, height: Dim, framerate: Rate },
    Audio { .. },
    Tensor { .. },
}
```

so a raw-only sink simply cannot match compressed caps, and the impossibility
becomes a type-level error rather than a runtime `not-negotiated`.

#### 4.13.2 The solver

`solver::solve_linear` runs arc consistency on a chain: forward sweep
(`Produces ∩ Accepts ∩ Identity ∩ Mapping ∩ DerivedOutput`), backward sweep
to propagate narrowing, fixate each link to its highest-preference concrete
`Caps`, then call `configure_pipeline` per element with its side of the link.

```rust
pub enum NegotiationFailure {
    EmptyLink { upstream: ElementId, downstream: ElementId, missed: CapsSet },
    EndpointShapeMismatch { .. },
    Unfixable { .. },
    Cyclic { .. },
}
```

Failures name the responsible pair and what they couldn't agree on, and are
posted to the pipeline `Bus` via `BusMessage::NegotiationFailed`.

`solver::downstream_feasibility(constraints) -> Vec<Option<CapsSet>>` is a
backward fold from the sink that computes, per link, the set the downstream
tail can still fixate **ignoring the upstream**. It's source-independent and
serves as a snapshot for the mid-stream re-solve (§4.13.4).

#### 4.13.3 The DAG runner

`run_graph(Graph<GraphNodeRef>, clock, link_capacity)` is the single runner.
A `Graph` is built from `GraphNode { Source | Element | Muxer }` payloads and
edges (each carrying a `LinkPolicy`); `finish()` validates topology (topo
sort, cycle / orphan / pad-count checks) before the run. `run_graph` owns
whole-graph `solve_graph` negotiation, per-node configure, the latency /
clock / allocation folds, one data arm per node over the edge channels, the
β allocation re-cascade and the Caps-α mid-stream re-solve. It covers the
full topology space: linear, fan-out (tee), fan-in (muxer), and diamonds.

`run_linear_chain`, `run_source_transform_sink`, `run_simple_pipeline`,
`run_source_fanout`, and `run_muxer_sink` are **thin builders**: each
constructs the corresponding borrowing `Graph` and delegates to `run_graph`,
so the four historical runner shapes share one negotiation + data plane. A
node's mid-stream rejection policy is topology-derived: a node on a
single-producer chain reverse-reconfigures and keeps flowing (posting the
structured failure to the bus), while a node behind a tee fails the run loud
(a shared upstream can't honor a per-branch reconfigure).

#### 4.13.4 Mid-stream re-solve

A mid-stream `PipelinePacket::CapsChanged` triggers a re-fixation that stays
correctly downstream-aware:

1. At startup, each interior arm receives its `downstream_feasible:
   Option<CapsSet>` from the backward sweep.
2. Mid-stream, arm *i* on `CapsChanged(in)`:
   - intersect `in` with the element's input constraint; empty → loud
     `EmptyLink` and reverse `Reconfigure` upstream;
   - derive output candidates from `in` via the constraint;
   - intersect candidates with `downstream_feasible[i]`;
   - fixate; `configure_pipeline(in)`; element-local realloc; forward
     `CapsChanged(fixated_output)`.

The **runner**, not `process(CapsChanged)`, owns the forwarded output. A
format-changing element moves its derivation into the declared constraint
(`Mapping` / `DerivedOutput`) as the single source of truth; the solver
already consumes it at startup and at re-solve.

The **CapsChanged ordering invariant** is the load-bearing correctness
property. `Caps` are not stamped on each frame; they live on the link as
the most recently received `CapsChanged` packet. Correctness across a
mid-stream change therefore depends on `CapsChanged` sitting **between**
the last old-caps `DataFrame` and the first new-caps `DataFrame` in the
forward stream — not before, not after. For a format-changing element
that buffers (decoder B-frame reorder, encoder lookahead), this means
the element emits its output `CapsChanged` at the **decode/encode
boundary** in its `process` output, not at the moment it received the
input `CapsChanged`. The runner cascades that ordered event downstream;
sinks reconfigure their pools when they see it, and the next data frame
they process is unambiguously under the new caps.

#### 4.13.5 Allocation cascade

Allocation negotiation is part of the same orchestration. A coordinator task
owns refs to source / transforms / sink and orchestrates events the spawned
arms can't reach from each other:

- **Element-local realloc:** `coordinator::realloc_local` re-derives an
  element's own pool from new caps (`propose_allocation` +
  `configure_allocation`) at each mid-stream apply site.
- **N-hop re-cascade:** the `select2` combinator + per-arm control channel
  makes each transform arm interruptible at its `recv().await`, so a
  sink-side allocation proposal walks upstream one hop at a time via
  `CoordinatorEvent::ArmProposal` until it reaches the source.

This is the same machinery a future mid-stream clock change or latency
adjustment uses: cross-element mid-stream coordination becomes a coordinator
event instead of an ad-hoc back-channel.

#### 4.13.6 Fan-out and fan-in

`run_source_fanout` per-branch re-solves a mid-stream `CapsChanged` via
`re_solve_downstream_dyn_sink`. Branches run in independent arms, so the
re-solves are concurrent (max of single-branch cost, not sum). The default
failure policy is strict: a branch whose constraint rejects the new caps
fails the fan-out loud (`CapsMismatch`); a future `FanOutPolicy::AllowBranchDrop`
opt-in is anticipated for graceful degradation.

`run_muxer_sink` solves each `source ↔ muxer-input` pair at startup,
per-input re-solves on mid-stream change, and eagerly re-emits the muxer's
output `CapsChanged` downstream when the merged output caps change as a
function of an input change. `MultiInputElement` exposes
`caps_constraint_as_input(idx)` and `caps_constraint_for_output()` for the
solver to consult per-input.

Over the DAG, a node-keyed `GraphCoordinator` walks a sink's re-derived
allocation proposal upstream through tees via `in_edges` (sources and muxers
terminate the walk), and a per-edge `graph_downstream_feasibility` snapshot
steers each transform's Caps-α output on a mid-stream change. β across a
muxer (per-input-pad re-cascade) is still owed.

Two flavours of fan-in element exist. `InterleaveMux` (`mux.rs`) is a
*multiplexer*: it forwards every input's frames straight through (each frame
carries its own caps), combining encoded tracks into one stream. `Compositor`
(`compositor.rs`) is a *pixel mixer*: it overlays N raw RGBA8 inputs onto one
output canvas at configurable position, z-order, and per-pad alpha (the
`videomixer` / `compositor` analog — picture-in-picture, camera grids, sub-window
UIs). It is CPU and `no_std`-baseline like the other raw-video transforms, with
straight source-over alpha blending and left/top clipping; a wgpu GPU companion
is a follow-up. Because a mixer must combine *simultaneous* inputs rather than
interleave, it caches the latest frame per input and uses **input 0 as the
timing driver**: one composited output frame is emitted per input-0 frame,
overlaying whatever the other inputs have most recently delivered. At startup it
briefly buffers input-0 frames (bounded) until every overlay has produced once,
so a late-starting overlay (camera warm-up) still appears; on buffer overflow the
oldest is emitted overlay-less rather than dropped, so a free-running background
never stalls or latches the overlay on one stale frame. The output canvas size
and framerate are fixed at construction; per-input geometry is whatever each
input negotiates (`Accepts(RGBA8)` per pad, `Produces` the fixed canvas). A pad
may also scale its input as it composites (`CompositorPad::with_size`, integer
bilinear), so a downscaled camera inset needs no upstream `VideoScale`.

#### 4.13.7 Pad templates

Static metadata for tools that need to query pad compatibility without
constructing the element. `PadTemplate` + the `PadTemplates` trait expose
`pad_templates()` as an associated function; `pad_link` and `types_can_link`
run the solver against two element types' static templates for pre-
instantiation compatibility checks. The runtime `caps_constraint_as_*`
remains the instance-level (possibly narrower) view.

#### 4.13.8 ACCEPT_CAPS and CapsFilter

Fall out of the constraint surface:

- **ACCEPT_CAPS query** is `constraint.accepts(&caps)`, a pure check against
  the constraint's set. No runtime round-trip; the element's constraint
  already describes everything it would accept.
- **`CapsFilter`** is an `Identity(specific_set)` pass-through. Inserted
  anywhere in a pipeline to force a narrowing.

#### 4.13.9 Auto-plug and the element registry

`decodebin`-equivalent, built on the pad-template metadata (§4.13.7) and the
solver. `g2g-core::runtime::autoplug` is two layers split by what they need:

- **Search** (`runtime`, `no_std`). `ElementDesc` is a name plus an element
  type's static pad templates. `find_chain(descs, input, target, max_depth)`
  is a breadth-first search over caps states: each edge is an element whose
  sink accepts the current caps (acceptance reuses `pad_link`, so an
  `Unfixable` link counts as compatible, exactly as `types_can_link`), and the
  search advances along that element's source-pad caps until one satisfies the
  `target` shape predicate (`is_raw_video` is the canonical `decodebin`
  target). The shortest chain wins; an element is never reused on a path, so a
  same-media-type parser (H.264 → H.264) cannot loop. The result is an ordered
  `Vec<ChainLink { index, output }>`: the search picks element *types* and the
  source-pad caps each was matched to produce, leaving geometry / framerate to
  fixate later at instance negotiation.

- **Registry** (`std`). `Registry` pairs each `ElementDesc` with an
  `ElementFactory` whose constructor is `fn(&Caps) -> Box<dyn DynAsyncElement>`,
  receiving the per-hop chosen output caps so a format-flexible element (a
  converter, a multi-format decoder) configures itself correctly.
  `Registry::autoplug` runs the search and instantiates the chain;
  `Registry::decodebin(graph, from, to, input, target, max_depth)` splices it
  into a `Graph<GraphNode>` between two existing pads (an empty chain links
  `from → to` directly), returning a sub-graph onto `run_graph`. Real element
  types publish templates via the `PadTemplates` trait (`FfmpegH264Dec`:
  H.264 → NV12 / I420), so a real decoder is registered and auto-plugged, not
  just synthetic descriptors.

Source-side `typefind` is not needed: a g2g source declares its output caps via
its source pad template / `caps_constraint`, so the caps feeding `decodebin` are
known without sniffing the byte stream.

- **playbin / uridecodebin** (`std`). `Registry::build_playbin(source_name,
  sink, target, max_depth)` assembles a complete `source → chain → sink` graph
  from a *named* registered source. `build_uridecodebin(uri, sink, target,
  max_depth)` is the URI front door over it: it parses `uri` (a minimal
  `scheme://rest` split — core pulls no URL crate), dispatches on the scheme to
  a registered `UriSourceFactory` that builds the source *from the URI*
  (`udp://host:port`, `file:///clip.mp4`, `rtsp://…`, `v4l2:///dev/videoN`), and
  auto-plugs the decode chain to `target`. The scheme handlers are the analog of
  GStreamer's `GstURIHandler`; the concrete ones live in `g2g-plugins`
  (`uridecodebin.rs`), each gated to its source's feature, so an app registers
  only the schemes its build supports. A handler reports the *media type* it
  produces (geometry resolves at negotiation), which is all the chain search
  needs to pick the right decoder.

### 4.14 Pipeline Lifecycle: State Machine, Preroll, and Seek

The lifecycle spine sits on top of the DAG runner: it turns "build, run to EOS,
drop" into a controllable `NULL → READY → PAUSED → PLAYING` machine that can
preroll, pause, scrub, and resume.

**State machine + preroll.** `PipelineState` (`NULL`/`READY`/`PAUSED`/`PLAYING`)
and `StateChangeReturn` are ungated core types. A `StateController` (runtime
feature) carries the target state and a sink-side **flow gate**: below `PLAYING`
a sink parks at the gate, stops draining its edge, and backpressure stalls the
DAG upstream, the state machine reuses the existing channel backpressure rather
than a separate pause mechanism. Preroll: a non-live `PAUSED` transition admits
exactly one buffer per sink and then holds; the runner calls
`expect_prerolls(n)` and each sink's `notify_prerolled` aggregates so the async
`PAUSED` completes with a single `AsyncDone` once *all* sinks have prerolled.
Live pipelines (`set_live(true)`) take the `NoPreroll` path (no frame is held).
The lifecycle is opt-in via `run_simple_pipeline_stateful` and
`run_graph_stateful`; the plain runners are unchanged.

**Seek + SEGMENT + running time.** `g2g-core::segment` is a pure-core (ungated)
model: `Seek` / `SeekType` / `SeekFlags` describe the request, and `Segment`
carries the rate/direction-aware running-time ↔ stream-time ↔ base-time math
(`GstSegment`-equivalent), with `clip` and `for_flush_seek` (which resets `base`
so running time restarts after a flush). `PipelinePacket::Segment` is the
carrier: the runner emits an opening SEGMENT and every element forwards it
(transforms/decoders forward, sinks consume), the same way `Flush` already
flows. A `SeekController` (runtime) is a cloneable handle the application holds;
a seek-aware source's run loop polls `take_pending()` between frames and, on a
flushing seek, emits `Flush`, repositions, emits the post-flush `Segment`, and
resumes, so a seek reaches the source GStreamer-style (upstream) without a
back-reference. Non-flushing/accumulating seeks, reverse/trick-mode sink
handling, and a real repositioning source (`Mp4Src`/`FileSrc`) are open
(DESIGN_TODO).

### 4.15 Bus and Observability

The pipeline `Bus` (§4.9.1) is a many-producer / single-consumer channel for
out-of-band events, so an element notifies the application without a
back-reference. `BusMessage` covers the lifecycle and quality signals an
application reacts to:

- `Eos`, `Error`, `Warning` — stream lifecycle and faults.
- `NegotiationFailed(NegotiationFailure)` — structured caps conflict naming the
  responsible element pair (§4.13), posted by the coordinator on a startup or
  mid-stream negotiation failure.
- `StateChanged { old, new }` and `AsyncDone` — every effective lifecycle
  transition, and the completion of an async `PAUSED` once preroll aggregates
  (§4.14).
- `Qos { running_time_ns, jitter_ns, processed, dropped }` — a synchronizing
  sink (`SyncSink`) that has fallen behind the clock drops a late frame
  (`with_max_lateness_ns`) and reports it, the `GST_MESSAGE_QOS` analog.
- `Buffering { percent }` — a sink's input link fill (0 = underrun, 100 = full),
  posted by the sink arm on a quartile crossing via `run_graph_with_bus`. Since
  g2g has no `queue` element, this reports the bounded link channel's own
  occupancy (`fill_percent`), the `GST_MESSAGE_BUFFERING` analog.

Posting is non-blocking (`try_post`): a control message never stalls the data
path; a full bus drops the report rather than applying backpressure.

### 4.16 Properties, Introspection, and the `gst-launch` DSL

The typed `with_*` builders are the zero-cost construction path and the only one
the `no_std` / RTOS baseline needs, but tooling (a text-pipeline parser, an
inspector, a future GUI) needs a *runtime* face: set a property by string name,
read it back, enumerate what an element exposes. Three layers, each building on
the last (M104-M106):

- **The property bag (`g2g-core::property`, `no_std + alloc`).** `PropValue`
  (`Bool` / `Int` / `Uint` / `Double` / `Fraction` / `Str`), `PropKind`, a static
  `PropertySpec` (name + kind + blurb), and `PropError`, plus
  `PropValue::parse(kind, "text")` for the `key=value` syntax. `AsyncElement` and
  `SourceLoop` (and their dyn mirrors) gain `properties()` / `set_property()` /
  `get_property()`, all defaulting to "no properties" the same zero-cost way
  `latency()` defaults to zero, so the baseline pays nothing and an element opts in
  only by overriding them. The GObject-property analog; the builders stay the
  type-checked path, this is the string-keyed one.
- **By-name construction + introspection (`Registry`, std).** `LaunchFactory`
  registers a transform / sink under a name with a parameterless constructor and
  its pad templates (sources reuse the parameterless `SourceFactory`).
  `make_source` / `make_element` build by name; `inspect(name)` dumps an element's
  role, properties, and pad templates, the `gst-inspect` analog.
- **The text parser (`runtime::parse_launch`, std).** Turns
  `"videotestsrc num-buffers=3 ! videoflip method=rotate-180 ! fakesink"` into a
  runnable `Graph`: each `!`-separated stage is `element-name key=value ...`;
  the element is built by name, each value parsed for its property's `PropKind`
  and applied, and the stages linked source -> transforms -> sink. The result
  drops straight onto `run_graph`, so a pipeline is expressible as text without
  hand-written Rust, the `gst-launch` analog. A bare `media/type,field=value,...`
  stage is the inline caps-filter shorthand (M117): `parse_launch` rewrites it to
  a `capsfilter` whose `caps` property is parsed by `capsfilter::parse_caps` (the
  `Caps` text grammar), so `videotestsrc ! video/x-raw,format=nv12,width=320 !
  ...` pins a format / geometry as text. Branching (M118) makes this a chain
  parser: `name=t` names an element and a `t.` reference opens a branch, with
  `tee` the structural fan-out node (its width derived from the branch count)
  broadcasting to every branch; roles follow connectivity. Text muxer fan-in is
  the remaining `gst-launch` gap.

### 4.17 Containers and Byte Streams

A container demuxer splits one stored / transported byte stream into the typed
elementary streams it carries. The link feeding a demuxer is
[`Caps::ByteStream { encoding }`](crate caps), the first byte-stream caps variant:
an opaque container stream not yet demuxed, tagged with a `ByteStreamEncoding`
(e.g. `MpegTs`) so a demuxer accepts only the format it parses, the
byte-stream-level analog of the codec/raw video split. A byte source declares it
(`FileSrc::new(path, Caps::ByteStream{MpegTs})`), and the demuxer's transform
constraint maps it to the elementary stream type.

The MPEG-TS demuxer (M108) is the first: `g2g-plugins::mpegts::TsDemuxer` is a
pure `no_std + alloc` parser (sync 188-byte packets, PAT -> PMT -> elementary
streams, reassemble PES per PID into access units with PTS), and the `TsDemux`
element wraps it. The parser reassembles every elementary stream the PMT names;
the element has one output pad, so a `TsStream` selection (M109: `H264` / `H265`
video as `CompressedVideo`, `Aac` audio as `Audio`, default H.264) picks which to
emit, and a second `tsdemux` selecting another stream demuxes the rest of the
multiplex. The selection is by codec, not a runtime-discovered "first video",
because the output pad's media type is fixed at negotiation before any packet is
parsed (H.264 and H.265 are distinct downstream decoders, not a refinement). Video
geometry is unknown until the bitstream parser reads the SPS, so the demuxer
advertises a fixatable placeholder `Range` refined downstream via `CapsChanged`
(the `RtspSrc` pattern, §4.13); AAC advertises the sentinel channels/rate that
`aacparse` refines from the ADTS header. The decode-side container precedent is
`Mp4Src` / `Mp4Sink`. The TS muxer (`mpegtsmux`: `g2g-plugins::mpegts::TsMuxer` +
the `TsMux` element) is the inverse path (M114), wrapping one elementary stream
back into PES + 188-byte packets with a real PSI CRC; multi-stream muxing,
multi-program selection, and PCR-based timing are follow-ups.

The Matroska / WebM demuxer (M110) is the second, the same parser + element split
keyed on `Caps::ByteStream{Matroska}`. `g2g-plugins::matroska::MatroskaDemuxer` is
a pure EBML parser (variable-length element IDs / sizes, descend into the Segment,
read Tracks for the elementary streams and `Info` TimestampScale, parse each
Cluster's SimpleBlock / Block frames with scaled timestamps), and `MkvDemux` wraps
it with the same per-codec `MkvStream` selection (H.264 / H.265 / VP8 / VP9 / AV1
video, AAC / Opus audio, default VP9). Unlike `TsDemux`, Matroska's Tracks element
carries concrete geometry and audio parameters, so the demuxer refines the output
caps itself via `CapsChanged` once Tracks is parsed, without a downstream bitstream
parser. WebM (the VP8/VP9/AV1 + Opus subset) is the browser-delivery motivator. Block
lacing (Xiph / EBML / fixed) is split (M113), so multi-frame audio blocks demux. The
MKV muxer (`matroskamux`: `MatroskaMuxer` + the `MkvMux` element) is the inverse path
(M115), writing the EBML header, an unknown-size Segment, Tracks, and one Cluster per
frame, with the `webm` DocType for the WebM codec subset. Scope is one Segment /
one track with definite-size Clusters; unknown-size Clusters (live read), Cues
(seeking), and multi-track muxing are follow-ups.

The Ogg demuxer (M116) is the third, the same parser + element split on
`Caps::ByteStream{Ogg}`. `g2g-plugins::ogg::OggDemuxer` parses RFC 3533 pages
(sync to "OggS", frame packets via the segment-table lacing with cross-page
reassembly, sniff the codec from the first packet's `OpusHead`, skip the setup
headers), and `OggDemux` emits the Opus audio packets as `Caps::Audio{Opus}` with
the channel count refined from `OpusHead`. The container is auto-detectable
(`typefind` "OggS", `filesrc bytestream-format=auto`). Granule-position timing,
Vorbis output, and an `oggmux` are follow-ups. With MP4 (`Mp4Src`/`Mp4Sink`),
MPEG-TS, Matroska/WebM, and Ogg, the demux/mux coverage spans the major
containers.

---

## 5. First-Class Machine Learning Integration
To prevent GPU-to-CPU synchronization stalls, tensor execution happens directly inside the VRAM domain. ML elements are `AsyncElement` implementations like any other — they negotiate `Caps::RawVideo` on the input pad and `Caps::Tensor` on the output pad.

### 5.1 Inline Tensor Pre-processing via WebGPU (wgpu)
The ML element sits in the same memory domain context as the hardware decoder. When a `MemoryDomain::DmaBuf` arrives at the ML element:

1. The memory handle is bound directly as a texture inside a `wgpu` compute pipeline.
2. An inline compute shader converts color spaces (e.g. NV12 → planar RGB) and performs normalization scales directly in graphics memory.
3. The resulting tensor handle is emitted as a `Frame { domain: VulkanTexture(...), caps: Caps::Tensor { .. }, .. }`, submitted straight to the inference backend.

`WgpuPreprocess` (`g2g-ml/src/wgpupreprocess.rs`, `wgpu` feature) is the compute-shader half: an NV12 frame is converted and normalized in a wgpu compute shader to a `Caps::Tensor { F32, [1,3,H,W], Nchw }`, the same contract `OrtInference` builds on the CPU. The system-memory variant uploads NV12 to a storage buffer and reads the f32 tensor back to `MemoryDomain::System`; the surface-import + GPU-resident tensor domain variant uses `MemoryDomain::DmaBuf` / `MemoryDomain::D3D11Texture` inputs directly.

### 5.2 Unified Pure-Rust Inference Backends
`g2g` avoids bundling heavy, unsafe proprietary C++ engines. The `g2g-ml` crate provides wrapper elements targeting two execution paradigms:

- **`g2g-ml::burn`** (Embedded / Wasm / RTOS): leverages the pure-Rust Burn framework with a `wgpu` backend, compiling ONNX workflows into type-safe, compile-time Rust graphics shaders. `BurnInference` (`g2g-ml/src/burninfer.rs`, `burn` feature) is the wgpu-backend inference element over the `RawVideo` → `Tensor` contract, driving an `input · W + b` linear layer on any Vulkan / Metal / DX12 / WebGPU adapter.
- **`g2g-ml::ort`** (High-Performance Enterprise Server): wraps ONNX Runtime bindings to pass underlying memory domains to hardware-specific execution paths (CUDA / TensorRT / DirectML / Apple CoreML) natively.

### 5.3 Native Async Batching Engine
The `g2g-enterprise` layer provides a lock-free, multi-channel execution sink that groups separate asynchronous video input streams into a single hardware tensor execution array:

```
[ Camera Stream 1 ] ──► Async Channel ──┐
[ Camera Stream 2 ] ──► Async Channel ──┼─► [ Bounded Batcher ] ──► [ GPU Tensor Core ]
[ Camera Stream 3 ] ──► Async Channel ──┘     (Select / Timeout)
```

### 5.4 Per-Frame Metadata & Detection Post-processing

Inference output is only useful once it is structured and travels with the
picture. Two pieces, both `no_std`-friendly:

- **The metadata system (`g2g-core::meta`, `metadata` feature).** The `Frame`
  carries a `FrameMetaSet`: a list of typed [`FrameMeta`] trait objects (the
  GstMeta analog) with attach / typed-get / iterate and a `propagate(Transform)
  -> Propagation` survival contract (a re-encode drops pixel-derived meta; a
  scale / crop / copy keeps it). Off by default, so the RTOS baseline pays
  nothing (`FrameMetaSet` is a ZST); the field was reserved at M88 and built out
  here. The standard `AnalyticsMeta` is the `GstAnalyticsRelationMeta` analog: a
  relation graph of `ObjectDetection` / `Classification` / `Tracking` nodes plus
  directed edges, so a detector → tracker → classifier → overlay chain reads
  results by node kind and traversal instead of re-deriving joins through tensor
  offsets. Bounding boxes are normalized `[0,1]`, so they survive a downstream
  resample without a coordinate rewrite.
- **The first producer (`g2g-ml::DetectionPostprocess`, `analytics` feature).**
  Decodes a YOLOv8-style `[1, 4+C, A]` output tensor (confidence threshold +
  per-class NMS) into `ObjectDetection`s, attaches an `AnalyticsMeta`, and
  forwards the frame. A real client shaping the metadata API (rather than
  speculation) is why the system was deferred to this point.
- **Metadata through fan-out (M100).** `FrameMetaSet` holds each `FrameMeta` as
  an `Arc<dyn FrameMeta>` and is `Clone`, so a tee clone shares the analytics
  graph by refcount rather than dropping it: the graph runner's
  `try_clone_packet` carries `frame.meta.clone()`, landing the same
  `AnalyticsMeta` on both branches of a `decode -> tee -> {detect, video}`
  diamond. Mutation is copy-on-write via `FrameMeta::clone_box` (the GstMeta
  `copy_func` analog): `FrameMetaSet::get_mut` deep-copies a shared entry before
  the mutable borrow, so a branch editing its analytics never aliases the
  sibling. Still a ZST no-op when the `metadata` feature is off.
- **The overlay (M101 / M102).** The visible end of the detector chain reads the
  `AnalyticsMeta` carried onto the *display* frame (via the fan-out path) and
  draws each box, so `decode -> tee -> {detect, video} -> overlay -> display`
  works. Two backends with a shared per-class palette: the CPU
  `g2g-plugins::analyticsoverlay::AnalyticsOverlay` (`analytics` feature) paints
  box outlines onto RGBA8 with the compositor's integer source-over blend (the
  `no_std` baseline), and the GPU `vellooverlay::VelloAnalyticsOverlay`
  (`vello-overlay` feature) strokes antialiased boxes over a full-frame image
  with the Vello GPU 2D renderer, emitting the result in the new
  `MemoryDomain::WgpuTexture` domain. That domain (an `OwnedWgpuTexture` whose
  `wgpu::Texture` lives in a `WgpuKeepAlive` owner, since `g2g-core` never links
  wgpu) is the render-side analog of the decode-side CUDA / D3D11 texture
  domains: the rendered frame stays on the GPU with no readback, so a GPU sink
  presents it directly.
- **The GPU sink (M103).** `g2g-plugins::wgpusink::WgpuSink` (`wgpu-sink`) is
  that consumer: it presents a `WgpuTexture` frame by sampling it in a small
  fullscreen blit pass onto its target (an owned offscreen texture for
  render-to-texture / screenshots, or a caller-built `wgpu::Surface` for an
  on-screen window), again with no readback. Because a wgpu texture is bound to
  its device, the overlay and the sink share one device through a cloneable
  `gpu::GpuContext` (the overlay's `with_context`, the sink's constructors), and
  the producer's texture is recovered by the sink through the shared
  `gpu::WgpuTextureKeepAlive` type. This closes the analytics path end to end:
  `decode -> tee -> {detect, video} -> overlay -> WgpuSink`, detections rendered
  on the GPU reaching the display with no system-memory round-trip. Window and
  event-loop ownership stay with the application (wgpu surfaces are built from a
  window handle and must drive the app's event loop), so the sink presents to a
  surface the app supplies rather than opening its own window.

---

## 6. Target Deployment Environments
Because the core processing loop requires only `core` and `alloc`, deployment profiles vary purely based on the top-level orchestration binary.

### 6.1 Enterprise Server Node (Cloud Scaling)
- **Runtime Driver:** Tokio multi-threaded runtime.
- **Inter-Element Channels:** Bounded MPMC async channels (`flume`).
- **Hardware Interop:** `cros-codecs` bitstream parsing feeding Linux kernel VAAPI / V4L2 drivers, producing `OwnedDmaBuf` handles.
- **Cargo features:** `multi-thread`, `std`.

### 6.2 Deep Embedded / Bare-Metal RTOS (Industrial & Robotics)
- **Target Hardware:** RTOS targets such as FreeRTOS, Zephyr, or microkernels.
- **Runtime Driver:** Embassy async executor (single-threaded, cooperative multitasking hardware timer loop).
- **Inter-Element Channels:** Zero-allocation stack channels (`embassy-sync`).
- **Hardware Interop:** Fixed-memory DMA rings mapped to microcontroller video capture peripherals.
- **Cargo features:** none (default `no_std + alloc`), or strict no-heap via `StaticBufferPool<_, N>` only.

#### 6.2.1 Embedded / Embassy Element Surface

The `no_std + alloc` core runs here directly: runner futures are
executor-agnostic and `ElementBound` is empty without `multi-thread` (§4.3).
The embedded surface comprises:

- `StaticBufferPool<T, N>` in `g2g-core` (pure `core`, no feature gate) — a
  compile-time-sized zero-allocation pool yielding bounded mutable references
  checked via compile-time lifetimes. This is the strict no-heap pool the
  `Arc<Mutex<Vec<T>>>` `BufferPool` (§3.3) cannot serve.
- `EmbassyClock` (`embassy` feature) over `embassy-time`, the `no_std` analog
  of `WallClock`. The tick rate is selected at the feature; a HAL provides
  the time driver at link.
- `PacketChannel` + `EmbassySink` (`embassy-link` feature) over
  `embassy-sync`, a zero-allocation inter-task packet link — the §6.2 stack
  channel.
- `embassy-futures::block_on` drives a pipeline as a single task; the full
  `embassy-executor` multi-task integration uses the same runner futures.

`portable-atomic` backs the `metrics::LatencyHistogram` `AtomicU64` so
`thumbv7em` (Cortex-M) and `riscv32` (which lack 64-bit atomics) compile;
`critical-section` makes the lock-based fallback interrupt-safe.

### 6.3 Browser Sandbox (Web Application Scaling)
- **Runtime Driver:** Web Workers spawned via `wasm-bindgen-futures`.
- **Hardware Interop:** Packets ingested via WebSockets / WebRTC data channels, parsed by browser hardware via the native WebCodecs JS API, and injected into WebGPU textures.
- **Cargo features:** `std` (`wasm32-unknown-unknown` provides a usable `std` shim).

#### 6.3.1 Browser / Wasm Element Surface

The browser target is `cfg(target_arch = "wasm32")` elements in `g2g-plugins`
behind the `web` feature (which implies `std`). The wasm bindings
(`wasm-bindgen` / `js-sys` / `web-sys` / `wasm-bindgen-futures`) are
target-gated so native builds never resolve them. No core change is needed:
the runner future is executor-agnostic, so `wasm_bindgen_futures::spawn_local`
drives it on the browser event loop, and wasm builds without `multi-thread`,
so the `!Send` JS handle types satisfy the empty `ElementBound` (§4.3).

The browser element surface comprises:

- `WasmClock` — `performance.now()` + `setTimeout` sleep, the wasm analog
  of `WallClock`.
- `WebSocketSrc` — ingest over a browser `WebSocket`, parallel to `FileSrc`
  / `RtspSrc`.
- `WebRtcSrc` (`web` feature) — ingest over a provided `RtcDataChannel`.
- `WebCodecsDecode` (`web-codecs` feature) — wraps the browser `VideoDecoder`;
  H.264 Annex-B access units in, `VideoFrame` copied to `System` RGBA out.
  Build needs `--cfg=web_sys_unstable_apis`.
- `CanvasSink` — presents decoded RGBA to an HTML canvas via the 2D context.
  A WebGPU-texture zero-copy variant uses `MemoryDomain::WebGPUBuffer` into
  a `GPUTexture` once the async device handshake lands in the keep-alive.

A complete in-browser glass-to-glass pipeline is
`WebSocketSrc → H264Parse → WebCodecsDecode → CanvasSink`. The local gate
for the wasm build is
`cargo check --target wasm32-unknown-unknown -p g2g-plugins --features web`.

---

## 7. Ecosystem Coexistence Strategy: GStreamer Bridge
To drive early enterprise adoption without forcing full system redesigns, `g2g` provides the `g2g-bridge` wrapper library, compiled as a compliant C dynamic library (`libgstglass2glass.so`). An isolated `g2g` processing sub-graph executes inside a legacy GStreamer pipeline.

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

The bridge intercepts the GStreamer pipeline's internal `GstBuffer`, extracts the underlying OS hardware file descriptor (`GstDmaBufMemory`), wraps it as a `g2g::OwnedDmaBuf` with a no-op close hook (GStreamer retains ownership of the fd), and forwards execution to the Rust async engine.

**Sync/async impedance:** the bridge runs a dedicated Tokio current-thread runtime on its own OS thread, communicating with the synchronous GStreamer `chain` function via a bounded `flume` channel. This isolates GStreamer's threading model from the async future matrix without blocking either side.
