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
}

pub struct Frame {
    pub domain: MemoryDomain,
    pub caps: Caps,
    pub timing: FrameTiming,
    /// Monotonically increasing per-source sequence number assigned at
    /// capture time and preserved unchanged across the pipeline. Used
    /// for drop detection and tracing, never for AV sync.
    pub sequence: u64,
}
```

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

The runner reports drops via a tracing hook; drop events are pipeline-observable, never silent.

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
| Async messages (bus) | Pipeline-level mpmc message channel (M11) |
| Latency aggregation query | Upstream-traveling query primitive (M12) |
| Allocation query | Downstream-proposed allocator handoff (M12) |
| Probes (`pad_block`, `pad_idle`) | `LinkInterceptor` trait registered on a slot (M11) |
| Seek with FLUSH | `PipelinePacket::Flush` + runner drain handling (M11) |
| Live clock distribution | `AsyncClock` provider election (M12) |
| EOS aggregation across N inputs | Fan-in / muxer (M10) |

#### 4.9.1 Differences Forced by Rust Ownership
GStreamer relies on parent ↔ child reference cycles via GObject reference counting plus signal callbacks. Rust's strict ownership doesn't allow that shape. Equivalent functionality lives in **message channels** instead of direct back-references: a child element that needs to notify its parent posts a bus message; the parent reads it. Functionally identical; structurally cleaner; no `unref` ordering hazards. Similarly, GStreamer's `gst_pad_link()` performs runtime pointer manipulation; the `g2g` equivalent — moving the receive end of a channel — requires explicit ownership transfer under a brief gate hold. Same outcome, more honest about what's happening.

#### 4.9.2 Capabilities That Fall Out For Free
- **No silent caps mismatch at runtime**: exhaustive typed `Caps` enum, `match` checked at compile time. GStreamer's string-keyed caps regularly fail at runtime with `not-negotiated`.
- **Deterministic shutdown**: Rust drop order is a topological walk; no leaked refs holding pipelines alive forever.
- **No GIL / no global state**: independent pipelines spawn on the same async runtime with zero coordination cost.
- **Memory safety across hot-swap**: ArcSwap guarantees no use-after-free when an element is replaced while a frame is in flight. GStreamer's `pad_block` / `pad_unlink` choreography is famously bug-prone here.

#### 4.9.3 The Single Architectural Trade-Off
Pre-allocated "dark slots" handle the common dynamic-pad case (a demuxer with at-most-N tracks). If an application genuinely needs runtime-growable pad count without an upper bound — e.g., a session router that accepts new RTP streams indefinitely — the dynamic layer uses a `Slab<Slot>` instead of a fixed array. Per-push slot lookup becomes one extra indirection. Since this only matters inside the already-type-erased dynamic layer, the cost is in the noise.

### 4.10 Negotiation & Dynamism Milestone Roadmap

| Milestone | Scope |
| :--- | :--- |
| **M8** | Mid-stream `CapsChanged` runner cascade; upstream `Reconfigure` sideband channel; `OutputSink::push` returns `PushOutcome`; `SourceLoop::reconfigure`; Phase 3 `ReFixate` becomes a real runner path; `ElementSlot` + `ArcSwap` slot mutation. |
| **M9** | Fan-out: `Router`, `Gate`, `Merger` primitives; `BranchSlot` with all three `SwapPolicy` variants; multi-output element trait variant. |
| **M10** | Fan-in: muxer trait variant; EOS aggregation across N inputs; per-input caps negotiation. |
| **M11** | Application control surface: pipeline `Bus` for async messages; `LinkInterceptor` probes; `PipelinePacket::Flush` for seek. |
| **M12** | Live-source surface (done): latency aggregation query (`LatencyReport` → `RunStats::latency`); allocation query (`AllocationParams` / `MemoryDomainKind` → `RunStats::allocation`); live clock distribution (`ClockPriority` / `ClockCandidate` / `elect_clock` → `RunStats::{clock_priority, base_time_ns}`). |
| **M13** | Decoder elements: `MfDecode` (Windows MFT, NV12 `System`); `FfmpegH264Dec` (Linux libavcodec, I420 `System`, validated end-to-end on AMD radeonsi); `VaapiH264Dec` (Linux cros-codecs/VAAPI, Intel-only until cros-codecs grows a non-GBM surface backend); first end-to-end RTSP → decoded-pixel pipeline. |

After M12, `g2g` reaches dynamic-pipeline feature parity with GStreamer while retaining the static typed layer (§4.8.1) for embedded targets that GStreamer does not address at all. After M13, the first fully-live glass-to-glass pipeline from network source to decoded pixels is operational.

### 4.11 Hardware Decoder Elements (M13)

The layers `RtspSrc → H264Parse` are fully functional for encoded-bitstream processing (mux, re-stream, record) after M11. Decoded-pixel output — required for ML inference, display, and colour-space conversion — needs a decoder `AsyncElement` that accepts `Caps::Video { format: H264 | H265, .. }` and emits `Caps::Video { format: Nv12, .. }` backed by `MemoryDomain::DmaBuf` (Linux) or `MemoryDomain::DxgiTexture` (Windows).

#### 4.11.1 cros-codecs (Linux VAAPI) — **Implemented**

`VaapiH264Dec` (`g2g-plugins/src/vaapidec.rs`, feature `vaapi`, `cfg(target_os = "linux")`) is implemented on top of `cros-codecs` 0.0.6 (`vaapi` backend). The crate is maintained by the ChromeOS team and exposes a stateless decoder framework that parses H.264 bitstreams and manages the DPB; the actual decode runs on the GPU through libva.

- **Input caps:** `Caps::Video { format: H264, .. }` — `intercept_caps` intersects with H.264 and rejects everything else
- **Output caps:** `Caps::Video { format: Nv12, .. }` backed by `MemoryDomain::System` (CPU copy out of the GBM-allocated surface; see "Deferred" below)
- **Frame allocation:** uses `GbmDevice::open("/dev/dri/renderD128")` (configurable via `VaapiH264Dec::with_render_node`) to allocate `GenericDmaVideoFrame` surfaces; the decoder's allocator callback returns one per output picture
- **Format negotiation:** the first `decode()` call surfaces `DecodeError::CheckEvents`; the element drains events, picks up the SPS-derived `StreamInfo` on `FormatChanged`, and re-feeds the same NAL
- **Flush:** forwards `decoder.flush()` and propagates `PipelinePacket::Flush` downstream
- **EOS:** flushes the decoder, drains the DPB, emits `Eos`
- **Thread safety:** `libva::Display` is `Rc<Display>` and therefore `!Send`; `unsafe impl Send` is justified by the runner's ownership model (move-not-share), matching the `MfDecode` pattern

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

**Deferred:**
- Zero-copy `MemoryDomain::DmaBuf` output. The GBM-allocated surface is already a DMA-buf; exposing its fd via `OwnedDmaBuf` is a follow-up requiring a refcount story to keep the surface alive until downstream releases it.
- H.265 decode. The same stateless framework supports it; a sibling element keyed on `VideoFormat::H265` is straightforward.
- Mid-stream resolution change is observed (`DecoderEvent::FormatChanged` → fresh `CapsChanged` downstream), but resolution-driven `Reconfigure` upstream is not yet plumbed.

**Hardware coverage status (cros-codecs 0.0.6):** validated on a Mesa-25.3 / `radeonsi` AMD Rembrandt iGPU, the element initialises libva successfully and the cros-codecs bitstream parser ingests the SPS/PPS and reports correct stream geometry — but cros-codecs's `GbmDevice::new_frame(NV12, ...)` then fails to allocate the decoder's output surface. Two distinct cros-codecs assumptions are at fault, neither of which we can paper over from the consumer side:

1. **Hard-coded 16×16 initial VAContext.** `VaapiBackend::new` creates the VA context with `Resolution::from((16, 16))` before any bitstream is fed; AMD `radeonsi` rejects that with `VA_STATUS_ERROR_RESOLUTION_NOT_SUPPORTED` and the `.expect` panics. A larger placeholder (e.g. 1920×1088) is accepted by every libva driver in the field and is later resized by `new_sequence()` once the SPS lands.
2. **ChromeOS-specific GBM flag for NV12.** Even past (1), `gbm_bo_create(NV12, ..., GBM_BO_USE_HW_VIDEO_DECODER)` returns NULL on radeonsi — that flag is a ChromeOS GBM extension. Mesa's `radeonsi` GBM provider does not expose `NV12` contiguous allocations at all (the `GBM_BO_USE_LINEAR` fallback also fails), because there is no GBM↔VAAPI surface-import path on AMD desktop the way there is on ChromeOS hardware.

The cleanest fix is upstream: a cros-codecs surface backend that allocates VAAPI surfaces directly through libva (`vaCreateSurfaces`) instead of routing through GBM. On Intel iGPUs with the iHD VAAPI driver the current GBM path is expected to work end-to-end with only fix (1). Until cros-codecs grows a non-GBM surface backend, the Linux production path is **ffmpeg `h264_vaapi`** behind a separate `ffmpeg-vaapi` feature (not yet wired up); the `vaapi` feature here is the right abstraction and stays in place to take the upstream fix transparently.

#### 4.11.2 Windows Media Foundation Transform (MFT) — **Implemented**

`MfDecode` (`g2g-plugins/src/mfdecode.rs`, feature `mf-decode`, `cfg(target_os = "windows")`) is fully implemented. It wraps `CLSID_MSH264DecoderMFT` via `windows-rs` using an MTA COM apartment and the Microsoft software H.264 decoder path.

- **Input caps:** `Caps::Video { format: H264, .. }` — rejects anything else at `intercept_caps`
- **Output caps:** `Caps::Video { format: Nv12, .. }` backed by `MemoryDomain::System` (CPU copy out of the MFT output buffer)
- **Flush:** forwards `MFT_MESSAGE_COMMAND_FLUSH` and propagates `PipelinePacket::Flush` downstream
- **EOS:** sends `MFT_MESSAGE_COMMAND_DRAIN` to flush B-frame reorder buffer before emitting `Eos`
- **Thread safety:** `!Send` by default (COM); `unsafe impl Send` justified by MTA free-threading — the MS H.264 decoder MFT is callable from any MTA thread without marshaling

**Deferred** (documented in the module header):
- D3D11 zero-copy output via a new `MemoryDomain::DxgiTexture` variant
- DXVA hardware acceleration (`MF_SA_D3D11_AWARE`)
- Strided NV12 output (currently assumes stride == width, true for the software decoder)

#### 4.11.3 ffmpeg / libavcodec (Linux production path) — **Implemented**

`FfmpegH264Dec` (`g2g-plugins/src/ffmpegdec.rs`, feature `ffmpeg`, `cfg(target_os = "linux")`) is implemented on top of `ffmpeg-next` 8.1 against system libavcodec. This is the production-ready Linux decode path that works on every libav-equipped host (validated end-to-end on AMD radeonsi + Mesa 25.3 where `VaapiH264Dec` cannot allocate surfaces).

- **Input caps:** `Caps::Video { format: H264, .. }` — `intercept_caps` intersects with H.264 and rejects everything else
- **Output caps:** `Caps::Video { format: I420, .. }` backed by `MemoryDomain::System` (CPU copy out of libavcodec's frame buffer). I420 is what `libavcodec`'s `h264` decoder emits natively for 8-bit 4:2:0 streams; `YUVJ420P` (full-range JPEG variant) is accepted with the same plane layout. Other pixel formats are rejected loudly with `CapsMismatch`.
- **Codec construction:** `ffmpeg::init()` → `codec::decoder::find(Id::H264)` → `open_as().video()`. No hwaccel attached — software decode.
- **Feed loop:** one access unit per `Packet::copy`, PTS forwarded verbatim (libavcodec treats it opaquely and echoes it back on the decoded frame); `send_packet()` then `receive_frame()` drained until `EAGAIN`.
- **Flush:** `decoder.flush()`; `PipelinePacket::Flush` propagates downstream.
- **EOS:** `send_eof()` then final drain to flush the B-frame reorder buffer before forwarding `Eos`.
- **Thread safety:** `ffmpeg::decoder::Video` wraps a raw `*mut AVCodecContext` and is `!Send` by default; `unsafe impl Send` is justified by the same ownership-transfer argument as `MfDecode` and `VaapiH264Dec`.

**System dependencies:**

| Distro | Packages |
| :--- | :--- |
| Fedora | `ffmpeg-free-devel` (or `ffmpeg-devel` from RPM Fusion) |
| Debian / Ubuntu | `libavcodec-dev libavformat-dev libavutil-dev libswscale-dev` |
| Arch | `ffmpeg` |

**Deferred:**
- NV12 output (currently I420). Mainline g2g decoders emit NV12; adding swscale conversion behind an `nv12-output` knob is a one-element follow-up — `software-scaling` is already enabled in the `ffmpeg-next` feature set so no new deps.
- VAAPI hwaccel: open the `h264_vaapi` codec with an attached `AVHWDeviceContext(VAAPI)`, register a `get_format` callback that claims `AV_PIX_FMT_VAAPI`, and `av_hwframe_transfer_data` the decoded surface into System memory. Stays inside this module — the public `AsyncElement` shape doesn't change. Useful on Intel iGPUs and AMD desktop (radeonsi VAAPI works fine; it's only the cros-codecs GBM/NV12 assumption that breaks).
- YUV444P / 10-bit pixel formats.

#### 4.11.4 End-to-End RTSP Pipeline

After M12 + M13 the first complete glass-to-glass receive pipeline is:

```
RtspSrc ──► H264Parse ──► [decoder] ──► [ML / display / encode]
(System/H264)               (DmaBuf or System / NV12)
```

| Platform | Decoder element | Feature | Output | Status |
| :--- | :--- | :--- | :--- | :--- |
| Linux any | `FfmpegH264Dec` | `ffmpeg` | `System` / I420 | **working** (validated AMD Mesa 25.3) |
| Linux Intel (iHD) | `VaapiH264Dec` | `vaapi` | `System` / NV12 | expected-working past cros-codecs init-size patch |
| Linux AMD (radeonsi) | `VaapiH264Dec` | `vaapi` | `System` / NV12 | blocked on cros-codecs GBM/NV12 — use `ffmpeg` |
| Windows | `MfDecode` | `mf-decode` | `System` / NV12 | working |

**Why the bitstream is already Wowza-ready:** `RtspSrc` connects via `retina` using standard RTSP/RTP over TCP, negotiates H.264 with `FrameFormat::SIMPLE`, and emits Annex-B access units. It will connect to any public Wowza server on port 554 once a decoder element is wired downstream and M12's live clock is in place for proper A/V timing. Port 554 is blocked in the development sandbox; live-streaming validation must be performed by the user.

The negotiation track (M8–M12) is orthogonal to the **platform-element track**, which adds concrete OS-coupled elements behind cargo features (each implying `std`):

| Milestone | Scope |
| :--- | :--- |
| **M5** | `RtspSrc` (`rtsp` feature) wrapping `retina`. |
| **M13** | `MfDecode` (`mf-decode` feature, Windows-only): Media Foundation H.264 Decoder MFT (`IMFTransform`) → NV12 `System` frames. Target-gated `windows` 0.62 dependency. Thread-affine (COM/MTA), single-thread executor. Deferred: D3D11 zero-copy, DXVA, strided NV12. |
| **M19** | `MfEncode` (`mf-encode` feature, Windows-only): Media Foundation H.264 Encoder MFT (`CLSID_MSH264EncoderMFT`), NV12 `System` frames → Annex-B H.264, low-latency mode (`MF_LOW_LATENCY`, no B-frames). Same COM/MTA contract as `MfDecode`; verified by an in-tree encode → decode round trip. |
| **M13** | `FfmpegH264Dec` (`ffmpeg` feature, Linux-only): `ffmpeg-next` 8.1 against system libavcodec, software H.264 decode → I420 `System` frames. Production-ready baseline; validated end-to-end on AMD radeonsi. Deferred: VAAPI hwaccel via `h264_vaapi` + `AVHWDeviceContext`, NV12 conversion via swscale, 10-bit / 4:4:4. |
| **M13** | `VaapiH264Dec` (`vaapi` feature, Linux-only): cros-codecs 0.0.6 stateless H.264 decoder on a VAAPI backend, GBM-allocated NV12 surfaces row-copied into `System` frames. Target-gated `cros-codecs` 0.0.6 + libva + GBM. Blocked on AMD desktop by cros-codecs GBM/NV12 surface assumption (see §4.11.1); `FfmpegH264Dec` covers Linux until that is fixed upstream. Deferred: zero-copy `DmaBuf` export, H.265, upstream `Reconfigure`. |
| **C3** | Zero-copy NVDEC → GPU display (`MemoryDomain::Cuda`). Phase 1: CUDA memory domain in core. Phase 2: `Backend::NvdecCuda` keeps decoded NV12 in device memory via the generic `h264` decoder + CUDA hwdevice + `get_format`. Phase 3: `CudaDownload` fallback + a `CudaGlSink` (CUDA↔GL interop, not KMS/dmabuf — see `DESIGN-C3-cuda.md`). Linux + NVIDIA-GPU only; user-side e2e. |

### 4.12 Live Egress Track (M46+)

`RtspSrc` (M5) and the decoders (M13) cover the receive path; egress is the
inverse, sending encoded video out over RTP. The protocol logic is Sans-IO
(§1): a pure packetizer produces the RTP packets and a thin sink does the UDP
I/O.

| Milestone | Scope | Status |
| :--- | :--- | :--- |
| **M46** | `RtpH264Packetizer` (`rtppay.rs`): sans-IO RFC 3550 + RFC 6184, an H.264 access unit to RTP packets (single-NAL when it fits the MTU, else FU-A fragments; marker on the AU's last packet; incrementing sequence). Pure `no_std` logic, host-tested and Cortex-M-clean. | **implemented** |
| **M47** | UDP egress sink + `AsyncElement` wrapper sending the packets; RTCP sender reports; an RTSP `ANNOUNCE` / `RECORD` path for Wowza-style ingest. | planned |

The packetizer is host-verifiable (parse the RTP headers back, reassemble the
FU-A fragments); the UDP send and RTSP server are user-side (port 554 is blocked
in the sandbox, as for the receive path, §4.11.4).

---

## 5. First-Class Machine Learning Integration
To prevent GPU-to-CPU synchronization stalls, tensor execution happens directly inside the VRAM domain. ML elements are `AsyncElement` implementations like any other — they negotiate `Caps::Video` on the input pad and `Caps::Tensor` on the output pad.

### 5.1 Inline Tensor Pre-processing via WebGPU (wgpu)
The ML element sits in the same memory domain context as the hardware decoder. When a `MemoryDomain::DmaBuf` arrives at the ML element:

1. The memory handle is bound directly as a texture inside a `wgpu` compute pipeline.
2. An inline compute shader converts color spaces (e.g. NV12 → planar RGB) and performs normalization scales directly in graphics memory.
3. The resulting tensor handle is emitted as a `Frame { domain: VulkanTexture(...), caps: Caps::Tensor { .. }, .. }`, submitted straight to the inference backend.

### 5.2 Unified Pure-Rust Inference Backends
`g2g` avoids bundling heavy, unsafe proprietary C++ engines. The `g2g-ml` crate provides wrapper elements targeting two execution paradigms:

- **`g2g-ml::burn`** (Embedded / Wasm / RTOS): leverages the pure-Rust Burn framework with a `wgpu` backend, compiling ONNX workflows into type-safe, compile-time Rust graphics shaders.
- **`g2g-ml::ort`** (High-Performance Enterprise Server): wraps ONNX Runtime bindings to pass underlying memory domains to hardware-specific execution paths (CUDA/TensorRT, Apple CoreML) natively.

### 5.3 Native Async Batching Engine
The `g2g-enterprise` layer provides a lock-free, multi-channel execution sink that groups separate asynchronous video input streams into a single hardware tensor execution array:

```
[ Camera Stream 1 ] ──► Async Channel ──┐
[ Camera Stream 2 ] ──► Async Channel ──┼─► [ Bounded Batcher ] ──► [ GPU Tensor Core ]
[ Camera Stream 3 ] ──► Async Channel ──┘     (Select / Timeout)
```

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

#### 6.2.1 Embedded/Embassy Track (M43+)

Built incrementally; the `no_std + alloc` core already runs here, since the
runner futures are executor-agnostic and `ElementBound` is empty without
`multi-thread` (§4.3). `StaticBufferPool` lands in `g2g-core` (pure `core`, no
feature gate); the Embassy clock/executor glue in `g2g-plugins` behind the
no_std `embassy` feature.

| Milestone | Scope | Status |
| :--- | :--- | :--- |
| **M43** | `StaticBufferPool<T, N>` (no-alloc core pool); `EmbassyClock` (`embassy-time`); pipeline under `embassy-futures::block_on`; bare-metal compile (`aarch64-unknown-none`). | **implemented** (EmbassyClock tick owed to a HAL time driver) |
| **M44** | `portable-atomic` for the `metrics` `AtomicU64` so Cortex-M (`thumbv7em`) / RISC-V32 compile; gate the std-only `coordinator_with_recascade_n` for a warning-free no_std build. | **implemented** |
| **M45** | `embassy-sync` zero-alloc packet link (`PacketChannel` + `EmbassySink`), the §6.2 stack channel. Full `embassy-executor` multi-task integration (vs the M43 `block_on` primitive) remains a follow-up. | **implemented** |
| **M48** | Fixed DMA-ring capture `SourceLoop`; no-alloc end-to-end frame flow (a lifetime-carrying `SystemSlice` wiring `StaticBufferPool` into the zero-copy path). | planned |

M44 closed the M43 finding: `metrics::LatencyHistogram` used `AtomicU64`, which
`thumbv7em` (Cortex-M) and `riscv32` lack, so `portable-atomic` now provides it
(native where available, a lock-based fallback elsewhere; `critical-section`
makes the fallback interrupt-safe on hardware). The core and the full embedded
stack (g2g-core + g2g-plugins + `EmbassyClock`) now compile for `thumbv7em`.
Verification is largely local (unlike §6.3): the pool unit tests and the
`block_on` pipeline run on the host, and the bare-metal compile proves
no-std-ness; only `EmbassyClock`'s tick needs a HAL driver on real hardware.

### 6.3 Browser Sandbox (Web Application Scaling)
- **Runtime Driver:** Web Workers spawned via `wasm-bindgen-futures`.
- **Hardware Interop:** Packets ingested via WebSockets / WebRTC data channels, parsed by browser hardware via the native WebCodecs JS API, and injected into WebGPU textures.
- **Cargo features:** `std` (`wasm32-unknown-unknown` provides a usable `std` shim).

#### 6.3.1 Browser/Wasm Element Track (M39+)

The browser target is built incrementally as `cfg(target_arch = "wasm32")`
elements in `g2g-plugins` behind the `web` feature (which implies `std`); the
wasm bindings (`wasm-bindgen` / `js-sys` / `web-sys` / `wasm-bindgen-futures`)
are target-gated so native builds never resolve them, mirroring the
windows/linux element gating (§2). No core change is needed: the runner future
is executor-agnostic, so `wasm_bindgen_futures::spawn_local` drives it on the
browser event loop, and wasm builds without `multi-thread`, so the `!Send` JS
handle types satisfy the empty `ElementBound` (§4.3).

| Milestone | Scope | Status |
| :--- | :--- | :--- |
| **M39** | Foundation: `web` feature; `WasmClock` (`performance.now()` + `setTimeout`, the wasm analog of `WallClock`); `WebSocketSrc` ingest source (analog of `FileSrc`); `run_websocket_ingest` `spawn_local` entry. | **implemented** (browser runtime owed a `wasm-bindgen-test` run) |
| **M40** | `WebCodecsDecode`: wrap the browser `VideoDecoder` (WebCodecs), H.264 Annex-B access units in, `VideoFrame` copied out to `System` RGBA. Pairs with `H264Parse`. Needs `--cfg=web_sys_unstable_apis`. | **implemented** (H.264; HEVC + browser runtime owed) |
| **M41** | `CanvasSink`: present decoded RGBA to an HTML canvas via the 2D context, completing in-browser glass-to-glass (`WebSocketSrc → WebCodecsDecode → CanvasSink`). WebGPU-texture zero-copy (`MemoryDomain::WebGPUBuffer` into a `GPUTexture`) deferred: async device handshake + core keep-alive. | **implemented** (2D; WebGPU + browser runtime owed) |
| **M42** | `WebRtcSrc`: ingest over a provided `RtcDataChannel`, the second browser ingest path. Web Workers executor deferred (JS-bootstrap infra; `spawn_local` already drives pipelines). | **implemented** (datachannel src; Workers + runtime owed) |

The M39 foundation makes `WebSocketSrc → H264Parse → FakeSink` compile and run
in the browser; M40–M42 add hardware decode, GPU zero-copy, and the
off-main-thread executor that complete the §6.3 picture. The in-browser runtime
(live `WebSocket` receive, `performance.now()` pacing) is validated user-side,
as with the live RTSP path (§4.11.4); the local gate is
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
