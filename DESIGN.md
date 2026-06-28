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
| `g2g-plugin` | SDK for dynamically loadable plugins (the `declare_plugin!` macro + ABI tag, §4.16). | `no_std + alloc` | LGPL v2.1+ |
| `g2g-plugins` | Standard collection of source/sink/transform elements (`rtsp`, `wgpu`, `v4l2`). | `no_std + alloc` / `std` mixed | LGPL v2.1+ |
| `g2g-ml` | ML inference elements built on `burn` (Wasm/embedded) and `ort` (server). | `std` | LGPL v2.1+ |
| `g2g-bridge` | C-FFI dynamic library to embed `g2g` sub-graphs inside GStreamer pipelines. | `std` (`cdylib`) | LGPL v2.1+ |
| `g2g-enterprise` | High-value multi-stream async ML batchers and tensor schedulers. | `std` | AGPL v3 |
| `g2g-python` | Hosts gst-python-ml elements as first-class `g2g` elements (embedded CPython via pyo3). | `std` | LGPL v2.1+ |
| `g2g-capi` | C ABI (cdylib/staticlib + `g2g.h`) to drive pipelines from any language: `parse_launch` + run + bus + appsrc/appsink. | `std` (`cdylib`) | LGPL v2.1+ |
| `g2g-pyapi` | Python (pyo3) bindings to drive pipelines: `parse_launch` + run + bus + appsrc/appsink (the inverse of `g2g-python`). | `std` | LGPL v2.1+ |

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
`g2g` treats system RAM as a fallback. Buffers track hardware descriptors to allow cross-process and cross-hardware zero-copy manipulation. Every hardware handle is reference-counted (an `Arc`-held keep-alive owner, or an `Arc`-shared fd for DMABUF): the underlying file descriptor or GPU allocation is released on the *last* drop. `MemoryDomain::share()` (M213) produces a second handle for a fan-out branch, a zero-copy refcount bump for the GPU domains and the shared-CPU `SystemView`, a deep copy only for owned-CPU `System` bytes. So a tee broadcasts a GPU-resident frame to several consumers (decode-on-GPU -> {inference, display}) with no device-to-host copy; branches treat the shared memory as read-only (a mutating branch copies first, as the per-frame metadata does copy-on-write).

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
- **Strict `no_std` (no heap) environments:** two pure-`core` pools sized at construction, no `alloc`. `StaticBufferPool::<[u8; N], 8>` is the *move-out* pool: `acquire` takes an owned buffer out and the RAII handle returns it on drop, the no-heap analog of `BufferPool`. `StaticLendRing::<N, BYTES>` is the *zero-copy lend* sibling for the capture path (a DMA ring): `N` inline slots, the producer fills the next free slot and `publish`es it as a `SystemSlice` that *borrows* the slot, and a per-slot lease (an `AtomicBool`, plain store, no CAS so it builds on `thumbv6m`) is cleared when the lent frame drops, so the slot is reused only after the consumer is done, the genuine ring back-pressure (the producer stalls when every slot is in flight). The borrow is runtime-guarded, not a Rust lifetime: a `PipelinePacket` crosses the `OutputSink` / stack channel by value (`'static`), so the lend reuses the `'static` foreign-buffer carrier (`SystemSlice::from_foreign`) with the lease standing in for the borrow. This keeps `Frame` / `MemoryDomain` lifetime-free (every element signature stays clean) while still proving a heap-free capture-to-consumer path end to end (validated under `block_on` over the embassy stack channel; a real capture wires a DMA-completion ISR / HAL into the same ring).

The `SystemSlice` carrier transparently supports these ownership models: `SystemSlice::from_boxed(Box<[u8]>)` for one-off frames, `SystemSlice::from_pool(PooledBuffer<Box<[u8]>>, len)` for recycled frames (the buffer may exceed the frame, so the valid length is carried), and `SystemSlice::from_foreign(ptr, len, free, user)` for a zero-copy lend of borrowed bytes (a `StaticLendRing` slot, or an application buffer through the C ABI). Downstream elements treat them identically.

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

#### Clock distribution to sinks

A pipeline runs against one elected clock (`elect_clock`: a live source's hardware clock outranks an audio sink's clock, which outranks the system fallback). The runner samples the elected clock's `now_ns()` once at startup as the **base time** (the clock reading at running-time zero) and hands both to each sink via `set_clock_sync(ClockSync { clock, base_time_ns })`, called once after election. Both the linear runners and the DAG runner `run_graph` deliver it (the latter walks its sink nodes after election, M172), so a display sink PTS-paces in any topology. A sink that synchronises presents a frame when the elected clock reaches `base_time_ns + running_time`, where running time is the frame's `pts_ns` mapped through the active `Segment`; a sink that ignores the hook presents as fast as backpressure allows.

`WaylandSink` is the first display sink to use it: it holds each frame until its running-time deadline, tracking the `Segment` (clipping pre-target frames after an accurate seek) and re-anchoring on `Flush`. It also does **QoS late-drop** (M173, matching `SyncSink`'s M85): a frame already past its deadline by more than a configurable `max_lateness` bound is dropped instead of presented late, so the sink catches up instead of accumulating lag, posting a `BusMessage::Qos` (running time, jitter, cumulative processed/dropped) per drop.

**Playing-transition anchoring (M176).** The startup base time is sampled before the data plane and before the application presses play. For a non-live, prerolled pipeline that sits in `Paused` for a while, that is the wrong epoch: the preroll frame is consumed during `Paused`, so a sink that anchored on the startup base (or on that first frame) then rushes/drops once `Playing` finally arrives. So when a `StateController` drives the run, the runner arms a `PlayAnchor` (a shared cell) on the elected clock and hands each sink `ClockSync::with_play_anchor`; `set_state(Playing)` stamps the anchor with `clock.now_ns()` at the exact play edge (and a transition down to `Ready`/`Null` clears it, so a replay re-bases). `ClockSync::base_time()` then resolves to the play-edge stamp once armed, else the eager startup base time. `WaylandSink` reads it per frame: it first-frame-anchors a preroll frame consumed during `Paused` (presented immediately), then re-bases onto the play edge once `Playing` stamps it; a seek `Flush` forces a first-frame re-anchor so the seek target presents immediately rather than against the stale play base. The non-stateful runners keep the eager base time (no `StateController`, no play edge to anchor to).

**Upstream QoS (M174)** carries that lateness back to the producer so it sheds load too, not just the sink. It rides the same per-link reverse channel as `Reconfigure`: a sink returns a `QosMessage` from `AsyncElement::take_qos`, the runner stores it into the incoming link's reverse `QosSlot`, and the producer observes it as `PushOutcome::Qos` on its next push (reconfigure wins when both are pending; QoS is advisory and never holds the packet back). `SyncSink` originates it on a late-drop and `VideoTestSrc` reacts by skipping ~`jitter / frame_period` frames (advancing PTS without generating them). **Relay through a transform (M175)** carries the report the rest of the way to the source in a multi-element pipeline. A transform observes a downstream QoS as a `PushOutcome::Qos` inside `process`, but that outcome is discarded by a generic transform, and the runner (not the element) owns the reverse slots, so the relay is runner-mediated: the runner wires the transform's *output* `SenderSink` with a relay handle to its *input* link's `QosSlot` (`relay_qos_to`). When the output adapter then sees a downstream QoS it stores it onto the input link instead of surfacing it, so the upstream neighbour observes it on its next push, and across N transforms the report walks one hop at a time back to the source. The element's `process` is unaffected; a QoS-aware transform that wants to act on the report itself is a later refinement. This is the same shape as the reverse `Reconfigure` path. Wired in the bespoke `run_source_transform_sink` runner and in the DAG runner (`run_graph` / `run_linear_chain`, which the `WaylandSink` demo uses), so the sink's own load-shed (M173) now reaches the source through interior transforms (overlay, convert). KMS vblank reconciliation (pick-frame-for-next-flip) and slaving video to an audio device clock are the remaining steps (DESIGN_TODO).

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

This is the primary response to a Phase 3 `ReFixate` or a mid-stream `Reconfigure` signal: replace the affected slot's contents, do not rebuild the graph. The swap is validated live under load (M349): an `ElementSlot` sits as a transform in `source -> slot -> sink` driven by `run_graph`, and a `SwapHandle::swap` mid-stream reroutes the remaining frames to the replacement element while every frame still reaches the sink, no drain or rebuild.

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

The bounded-N realization landed in M205: `StreamDemux` (`g2g-plugins`) is a `MultiOutputElement` with N typed output ports, driven by `run_source_fanout`. Each port carries its own declared caps and is fed by a caller-supplied classifier (`Fn(&Frame) -> usize`); the first frame routed to a port emits that port's `CapsChanged` so the branch retypes from the demuxer's byte-stream input caps to the elementary stream's, the same announce a single-output demuxer does. The N branch links the runner pre-allocates *are* the dark slots: a port no stream ever routes to simply stays silent and takes the merged EOS at end. This is the multi-output demuxer (one element, several typed downstream branches); the prior fan-out elements (`Router`, `Gate`) only broadcast or A-B-switch a single caps. Container parsers (MPEG-TS multi-PID) wire onto it by keying the classifier on parsed stream identity.

M210 made the demux a first-class DAG node, the symmetric counterpart to the muxer fan-in. Rather than a new `NodeKind`, a demux reuses `NodeKind::Tee(n)` for the structural/solver view (it negotiates exactly like a tee at startup, per the dark-slot retyping above) and carries a `GraphNodeRef::Demux` payload that the runner dispatches to `demux_arm` (the transpose of `muxer_arm`) instead of the broadcast `tee_arm`. So the solver is unchanged and only the runtime behavior differs. `Graph::add_demux` builds the node; `DynMultiOutputElement` is the dyn-safe mirror of `MultiOutputElement`. In `gst-launch`, a name registered via `register_demux` with several outputs builds a demux (`src ! d.  d. ! …  d. ! …  <demux> name=d`) instead of erroring `FanOutWithoutTee`, the transpose of the muxer's link-degree rule. There is no content-agnostic default demux in the registry: routing is inherently stream-specific (as the muxer side ships specific muxers), so `register_demux` is the surface.

### 4.10 Architectural Tracks

The framework is built along five interlocking tracks. The spec sections that
follow describe each track's current architecture.

| Track | Section | Summary |
| :--- | :--- | :--- |
| Receive | §4.11, §4.12a/b, §4.19 | Network + capture sources and hardware decoders (RTSP, raw RTP ingest with jitter buffer + RTCP/NACK, WebRTC WHEP/sendrecv, V4L2 capture, file, fMP4, software/VAAPI/MF/NVDEC decoders). |
| Display & egress | §4.11.5, §4.12, §4.19 | GPU-resident presentation sinks and outbound RTP packetizers; WebRTC WHIP / sendrecv egress. |
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
| `Vaapi` | `h264` + `AV_HWDEVICE_TYPE_VAAPI` | `System` | GPU decode, surface downloaded to system memory (`av_hwframe_transfer_data`). The Linux AMD / Intel hardware path; works on Mesa `radeonsi` where cros-codecs `VaapiH264Dec` cannot. Pin the render node with `with_vaapi_device` (or the `device` property; launch name `ffmpegvaapidec`). |

- **Input caps:** `Caps::CompressedVideo { codec: H264, .. }`.
- **Output caps:** `Caps::RawVideo { format: I420 | Nv12, .. }`. `I420` is the libavcodec native 8-bit 4:2:0 format; `Nv12` is selectable via `FfmpegH264Dec::with_output_format(OutputFormat::Nv12)`, produced by a U/V interleave with no swscale. `YUVJ420P` is accepted with the same plane layout; `YUV444P` / `YUVJ444P` are accepted with the chroma planes box-averaged down to 4:2:0. Other pixel formats are rejected with `CapsMismatch`.
- **Feed loop:** one access unit per `Packet::copy`; PTS is forwarded verbatim (libavcodec echoes it back on the decoded frame); `send_packet()` then `receive_frame()` drained until `EAGAIN`.
- **Flush / EOS:** `decoder.flush()` on `PipelinePacket::Flush`; `send_eof()` + final drain before forwarding `Eos`.
- **Thread safety:** `ffmpeg::decoder::Video` wraps a raw `*mut AVCodecContext` and is `!Send` by default; `unsafe impl Send` is justified by the same ownership-transfer argument as `MfDecode` and `VaapiH264Dec`.

`FfmpegH264Enc` (`g2g-plugins/src/ffmpegenc.rs`, feature `ffmpeg`, `cfg(target_os = "linux")`, M266) is the encode-side mirror: `Caps::RawVideo { format: I420, .. }` in, `Caps::CompressedVideo { codec: H264, .. }` Annex-B out, via `ffmpeg-next`. It gives the Linux production path a hardware H.264 encoder, the codec `WebRtcSink` / `RtpH264Packetizer` / the RTSP server require (the other Linux encoders are AV1 / VP8/9 / MJPEG, none of which those H.264-only sinks accept). Selectable backend:

| `Backend` variant | Encoder opened | Notes |
| :--- | :--- | :--- |
| `Nvenc` (default) | `h264_nvenc` | NVIDIA NVENC; hardware, realtime. The server-side render-and-stream path wants this. Fails loud at configure if absent (no driver / libavcodec built without it). |
| `Software` | `libx264` | Portable CPU fallback (CI / no-GPU hosts), present only if libavcodec was built `--enable-libx264`. |

- **Low latency:** `max_b_frames = 0` (output in presentation order, no reorder hold), in-band SPS/PPS (the `GLOBAL_HEADER` flag is *not* set, so parameter sets ride each IDR, the Annex-B stream a network sink expects), and a per-backend low-latency preset/tune (`p4`/`ll`/CBR/`delay=0` for NVENC, `veryfast`/`zerolatency` for libx264). A downstream PLI (`Reconfigure::ForceKeyframe`) forces an IDR on the next frame via `pict_type`.
- **PTS:** the input frame's nanosecond PTS is mapped through the encoder's frame-index PTS (`time_base = 1/fps`) and recovered on the output packet, surviving any reorder.
- **Validation:** a round-trip test on the RTX 3060 encodes I420 through `Nvenc` (and `Software`) and decodes the result back through `FfmpegVideoDec`, asserting Annex-B framing and that the stream decodes to I420 at the original geometry. Like the decoder, the `ffmpeg` feature is CI-excluded (libav version-sensitivity), so this is validated on libav hosts. Deferred: runtime bitrate retarget (fixed at open, like `Av1Enc`'s rebuild), NV12 input, 10-bit.

`NvEnc` (`g2g-plugins/src/nvenc.rs`, feature `nvenc` which implies `cuda`, `cfg(target_os = "linux")`, M269) is the **zero-copy, device-resident** H.264 encoder: the moat version of the ffmpeg `Nvenc` backend. The ffmpeg encoder takes *system-memory* I420 and copies it into libavcodec; `NvEnc` ingests an NVDEC/CUDA NV12 surface (`MemoryDomain::Cuda`) **in place** and drives the NVIDIA Video Codec SDK (`nvEncodeAPI`) directly, so the pixels never leave the GPU. It closes the native `FfmpegH264Dec(NvdecCuda) -> NvEnc` loop with no PCIe download, the encode-side mirror of the §5.1 `CudaToWgpu` import bridge, and is the egress half of the server-side render-and-stream path (M267) once a wgpu->CUDA hand-off feeds it.

- **Caps:** `Caps::RawVideo { format: Nv12 | Rgba8 | Bgra8, .. }` in, `Caps::CompressedVideo { codec: H264, .. }` Annex-B out (a native `DerivedOutput`, same dims / framerate). Caps do not encode the memory domain, so negotiation is identical to a system encoder; at runtime the frame must be `MemoryDomain::Cuda` (`UnsupportedDomain` otherwise, the symmetric contract `FfmpegH264Enc` upholds for `System`). NV12 input (the NVDEC hwframe domain) must be a contiguous surface (chroma at `luma_ptr + luma_pitch * height`, one base pointer + pitch); RGBA input (the GPU-render domain, e.g. via `WgpuToCuda`, M271) is a single packed plane at `luma_ptr` with `luma_pitch = width * 4`, registered as NVENC `ABGR` (wgpu `Rgba8` byte order) / `ARGB` with NVENC doing the colour conversion to H.264 internally.
- **Bindings: hand-rolled FFI.** Like the `cuda` module (DESIGN-C3-cuda.md §6), `cudarc` is not used; the element links `libnvidia-encode` + `libcuda` directly. The SDK's giant version-tagged structs are transcribed `#[repr(C)]` with **compile-time size assertions** (`const _: () = assert!(size_of::<T>() == N)`) checked against the installed `nvEncodeAPI.h` (SDK 13.0; field offsets verified with `offsetof`), so a mismatched SDK fails the build rather than corrupting the wire layout. The one field-heavy codec-config union is left opaque (a correctly-sized `[u32; N]`): the driver fills it via `nvEncGetEncodePresetConfigEx`, and we overwrite only rate control / GOP.
- **Lifecycle:** the encode session opens lazily on the first frame, on that frame's `CUcontext` (the NVDEC source's context). Per frame: `nvEncRegisterResource` (`CUDADEVICEPTR`, NV12) -> `nvEncMapInputResource` -> `nvEncEncodePicture` -> `nvEncLockBitstream` (copy out Annex-B) -> unlock / unmap / unregister.
- **Low latency:** preset P4 + the LOW_LATENCY tuning info, CBR, no B-frames (`frameIntervalP = 1`), and an *infinite GOP* (`NVENC_INFINITE_GOPLENGTH`) so IDRs are emitted only on demand: the first frame, and on a downstream PLI (`Reconfigure::ForceKeyframe`). Each forced IDR sets `OUTPUT_SPSPPS` so in-band parameter sets ride it (the Annex-B a network sink expects). The NV12 nanosecond PTS round-trips through NVENC's `inputTimeStamp`.
- **Validation:** an on-hardware round-trip on the RTX 3060 synthesizes a CUDA-resident NV12 surface (CUDA driver alloc + upload), encodes through `NvEnc`, and decodes the Annex-B back through `FfmpegVideoDec` to the original geometry; it skips cleanly with no NVIDIA GPU. The `nvenc` feature is CI-excluded (no NVENC runtime in CI). **HEVC (H.265)** is supported alongside H.264 (M273): `with_codec(VideoCodec::H265)` / the `codec` property switches the encode GUID to `NV_ENC_CODEC_HEVC_GUID` and the output caps to `CompressedVideo{H265}`, the path otherwise identical (the round-trip test covers both). `NvEnc` declares `input_domains = {Cuda}`, so a CPU-side NV12 source feeding it gets a `CudaUpload` spliced in automatically by the converter auto-plug (M353/M354, §4.13.5); the encoder itself stays Cuda-only. Deferred: 10-bit, finite-GOP periodic IDRs (`repeatSPSPPS`). (The output-bitstream-buffer pool and runtime bitrate retarget landed in M277.) The matching native `NvDec` is the other half of the gst-`nvcodec`-style pair.
- **Thread safety:** the session is a raw NVENC handle + CUDA context driven through `&mut self` only; `unsafe impl Send` rests on the same ownership-transfer argument as `FfmpegH264Enc`.

`NvDec` (`g2g-plugins/src/nvdec.rs`, feature `nvdec` which implies `cuda`, `cfg(target_os = "linux")`, M270) is the **decode half of the gst-`nvcodec`-style pair**, the mirror of `NvEnc`. It promotes NVIDIA hardware decode from the `FfmpegH264Dec` `Backend::NvdecCuda` flag (which reaches NVDEC *through* libavcodec's cuvid hwaccel) to a first-class element driving the NVCUVID parser+decoder API directly. With `NvDec -> ... -> NvEnc` both native, the whole H.264 transcode loop stays on the GPU and out of libavcodec.

- **Caps:** `Caps::CompressedVideo { codec: H264, .. }` Annex-B in, `Caps::RawVideo { format: Nv12, .. }` out (a native `DerivedOutput`). The runtime `CapsChanged` carries the actual cropped display geometry the bitstream declares.
- **Multi-domain output (M352).** `NvDec` advertises `output_domains = {Cuda, System}` and, in `configure_allocation`, reconciles the negotiated proposal against that capability (`resolve_for_producer`, §4.13.5): a CUDA-capable consumer keeps each surface device-resident (zero-copy, the default `MemoryDomain::Cuda`); a System-only consumer makes the decoder download (reusing `cuda::download_nv12`) before emitting. The same decoder stays on the GPU or downloads, chosen by downstream demand alone, validated on the RTX 3060.
- **Callback model:** NVCUVID is callback-driven. A parser (`cuvidCreateVideoParser`) is fed the elementary stream and synchronously invokes three callbacks from inside `cuvidParseVideoData`: a *sequence* callback (creates the `CUvideodecoder` once the SPS geometry is known), a *decode* callback (`cuvidDecodePicture`), and a *display* callback (a frame is ready in display order). The display callback cannot `await`, so it maps the surface (`cuvidMapVideoFrame64`) and pushes a ready frame onto a queue that `process` drains and emits after the parse returns. The callbacks reach element state through a `*mut DecoderState` passed as the parser user-data; that pointer targets a heap `Box` so it survives the runner moving the element between worker threads.
- **Bindings: hand-rolled FFI.** Links `libnvcuvid` + `libcuda` directly (no `cudarc`). NVCUVID exports real symbols (no `CreateInstance` dispatch table, unlike NVENC), so the calls are plain `extern "C"`; the structs are transcribed `#[repr(C)]` with compile-time size assertions against the installed `cuviddec.h` / `nvcuvid.h`, and the per-picture `CUVIDPICPARAMS` is opaque (the parser fills it, we pass the pointer straight to `cuvidDecodePicture`).
- **Frame lifetime:** each output frame carries a `CudaKeepAlive` that `cuvidUnmapVideoFrame64`s on drop plus an `Arc` to the decoder, so the decoder and its CUDA context outlive any frame still in flight; the decoder, context lock, and context are destroyed (in that order) only once the last frame is released. The element owns its own CUDA context (created at configure).
- **Validation:** an on-hardware test on the RTX 3060 runs the full native loop, a synthesized CUDA NV12 surface encoded by `NvEnc` to Annex-B and decoded by `NvDec` back to CUDA NV12, asserting geometry and (via a small device->host copy) that the decoded luma holds real content; it skips with no NVIDIA GPU. The `nvdec` feature is CI-excluded. **HEVC (H.265)** is supported alongside H.264 (M273): the input caps accept `CompressedVideo{H264|H265}`, the codec is inferred and mapped to the `cudaVideoCodec` the NVCUVID parser + decoder are created for. Deferred: mid-stream resolution change (decoder reconfigure), AV1 / other codecs, 10-bit, and a configurable display delay (fixed at a low-latency 1).

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

`RtspSrc` connects via `retina` using standard RTSP/RTP over TCP, negotiates H.264 with `FrameFormat::SIMPLE` (Annex-B) or accepts AVCC framing detected per buffer. The first SPS the parser sees provides geometry; framerate is recovered from the VUI `timing_info` (`time_scale / (2 * num_units_in_tick)`) when present, or left as `Rate::Any` when the VUI is absent. `RtspSrc::with_credentials` supplies the DESCRIBE/SETUP account (threaded into retina's `SessionOptions`).

`OnvifSrc` (`onvif` feature) is the ONVIF *control plane* in front of `RtspSrc`. An ONVIF camera does not stream over ONVIF; its SOAP services tell you the RTSP URL. `discover` sends one WS-Discovery `Probe` to the `239.255.255.250:3702` multicast group and collects each camera's device-service URL from the `ProbeMatch` `XAddrs`; `resolve_stream_uri` then runs `GetCapabilities` → `GetProfiles` → `GetStreamUri`, authenticated with a WS-Security `UsernameToken` digest (`Base64(SHA1(nonce ++ created ++ password))`). The element resolves the RTSP URI lazily during negotiation (`intercept_caps`), builds an inner `RtspSrc` once (forwarding the same credentials, since cameras gate the media stream behind the device account), and delegates the rest of the `SourceLoop` to it. The SOAP layer is hand-rolled (fixed request templates + `roxmltree` response reads) to avoid the git-only `onvif`/`schema` crate tree; the footprint is reqwest + roxmltree + sha1 + base64 + getrandom. Scope is discovery + stream-URI resolution; PTZ and event subscriptions are not implemented.

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
  the CUDA-GL interop entry points. Validated on an RTX 3060 (M252/M253):
  ~10.7x lower present latency than `NvdecCuvid -> WaylandSink` at 1080p.
- `CudaKmsSink` (`cuda-kms` feature, Linux + NVIDIA, M255) is the tty /
  no-compositor counterpart: the same CUDA-GL interop + NV12->RGB shader (shared
  via the `glnv12` module), but EGL renders into a GBM surface scanned out via
  DRM page-flips instead of a Wayland surface. Needs DRM master (a bare VT or a
  DRM lease). The shared render half is the validated `CudaGlSink` path; the
  GBM/EGL/DRM present is authored + compiles but its on-tty run is owed.

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

### 4.12a Live Capture (V4L2, libcamera)

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

`LibCameraSrc` (`libcamerasrc.rs`, `libcamera` feature, Linux-only) is the
second capture source and the modern Linux camera path: it captures through the
**libcamera** stack (linking the system libcamera via the `libcamera` crate),
which drives UVC webcams through its `uvcvideo` pipeline handler (the same
devices as `V4l2Src`) plus CSI/ISP cameras that need an ISP pipeline V4L2 alone
cannot. It follows the same two design points as `V4l2Src` (blocking work off
the async path; up-front negotiation, re-configure for capture), but differs in
two ways: it asks libcamera for **NV12** and falls back to **YUYV** only when
the camera does not offer NV12 (mapping whatever survives `validate()` to
`Caps::RawVideo`), so a camera that produces planar frames needs no
`VideoConvert`; or, with `with_mjpeg(true)` / `format=mjpeg`, it negotiates
**MJPEG** and emits `CompressedVideo{Mjpeg}` for `MjpegDec` downstream (the
on-camera-compression path for resolutions / frame rates uncompressed YUYV
cannot sustain over USB). Because libcamera is callback-driven and thread-affine, the
capture thread owns the whole libcamera object graph (manager, camera, a
request-buffer ring, and the completion callback) rather than a single device
handle. Each completed request's planes are packed contiguously (Y then
interleaved UV for NV12) before being forwarded over the bounded channel. The
requested frame rate is bounded on the camera with a `FrameDurationLimits` start
control (the minimum frame duration caps the fastest rate; the maximum is left
generous so an unachievable request degrades to the camera's own ceiling instead
of collapsing). Manual exposure / gain (`with_exposure` / `with_gain`, which turn
auto-exposure off) ride the same start-control path and are the real frame-rate
lever in low light: with auto-exposure on the camera lengthens exposure until the
rate collapses (~9 fps on a dim webcam, the same rate in every format and
resolution), while a fixed short exposure restores a high rate (measured 8.8 ->
24.9 fps on the developer's webcam). `Brightness` (and `Contrast` / `Saturation`)
are post-capture image adjustments that do not touch the exposure time, so they
brighten a dim short-exposure frame without giving back the frame rate (measured
mean luma 16 -> 117 at a fixed exposure). The camera can also be selected by an
id substring (`with_camera_id`) rather than enumeration index, stable across
reboots. Start controls are applied through a support
check against the camera's `ControlInfoMap`, because libcamera aborts the process
(a C++ exception across the FFI boundary) if a control list carries an id the
pipeline handler does not advertise (a UVC webcam may expose `ExposureTime` but
not `AnalogueGain`). The `libcamera` crate requires libcamera
`>= 0.4`, newer than some distro packages, so the feature is host-validated (like
the NVIDIA stack) rather than built in CI. The camera also feeds the GPU/ML path:
the g2g-ml `libcamera-wgpu` feature chains `LibCameraSrc -> VideoConvert(NV12) ->
WgpuPreprocess` to turn live frames into a normalized f32 NCHW tensor on the GPU
(validated camera-to-tensor on an RTX 3060). A zero-copy dma-buf import of
libcamera buffers into wgpu (the Linux analog of the CUDA / AHardwareBuffer
interop) was investigated under the `libcamera-dmabuf` feature: libcamera does
export a real dma-buf fd, but on a USB camera + discrete NVIDIA GPU the driver
advertises the buffer as importable (`vkGetMemoryFdPropertiesKHR`) yet the actual
`vkAllocateMemory` import fails to bind, because the buffer is CPU/vmalloc-backed
and the dGPU cannot map it. So the CPU-upload path is correct for that
configuration; zero-copy is expected to work on an integrated GPU (shared memory)
or a CSI/ISP camera (GPU-visible buffers), and the full import-to-texture element
is gated behind the on-hardware probe rather than shipped blind.

Two more capture sources follow the same blocking-work-off-the-async-path shape:
`PipeWireSrc` (`pipewire` feature, Linux) captures interleaved PCM off the
PipeWire graph (the modern Linux media layer) by running a `pw::stream` input on
a dedicated main-loop worker thread feeding the `run` loop over a channel; it
requests a fixed PCM format the PipeWire adapter converts to, so the produced
caps are deterministic (video / screen capture is a follow-up). `MfVideoSrc`
(`mf-video-src`, Windows) is the camera sibling of `WasapiSrc`: it enumerates
video capture devices and drains NV12 / YUY2 frames via an `IMFSourceReader` on a
COM/MTA worker thread.

#### Linux audio output

The audible-output end of the audio path on Linux mirrors the Windows-only
`WasapiSink` across the three Linux audio stacks, each a `std`-gated element with
a dedicated render worker thread: `AlsaSink` (`alsa-sink`, libasound, lowest
level), `PulseSink` (`pulse-sink`, the blocking libpulse "simple" API), and
`PipeWireSink` (`pipewire`). ALSA / Pulse backpressure naturally through the
blocking write; PipeWire's `process` callback pulls on its own clock and cannot
backpressure, so that sink's hand-off queue is leaky (bounded to ~1 s, dropping
the oldest bytes, the `LinkPolicy::DropOldest` analog for an external clock). All
accept interleaved `PcmS16Le` / `PcmF32Le` and reject compressed audio
structurally. Errors surface as `HardwareError::{Alsa,PulseAudio,PipeWire}`.

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
so the loop recovers packet loss end to end. **RFC 4588 RTX** (M222, `rtx.rs`)
wraps a NACK resend in a distinct payload type / SSRC with the original sequence
prepended (`UdpSink::with_rtx` / `UdpSrc::with_rtx`), unambiguous under heavy
loss. **ULPFEC** (M225, `ulpfec.rs`, RFC 5109) adds *feedback-free* recovery: the
sender XORs each group of media packets into a repair packet (`with_fec`), and
the receiver reconstructs a single per-group loss from the repair plus the
survivors and injects it into the jitter buffer, with no round trip, the better
fit for one-way or high-RTT paths. NACK, RTX, and FEC compose; FlexFEC and
multi-level burst FEC are the remaining receive-side items.

This is **raw RTP** with no RTSP/SDP, so there is no out-of-band stream
description: the output geometry is a declared hint (`with_video_size` /
`with_framerate`), and since H.264 carries its real dimensions in the SPS a
downstream decoder re-derives and corrects them. SDP/SPS-driven caps discovery is
a follow-up; `RtspSrc` (via `retina`) already covers the RTSP case with its own
jitter buffer (§4.11.4).

**RTMP ingest.** `RtmpSrc` (`rtmpsrc.rs`, `rtmp` feature) accepts one RTMP
publisher (ffmpeg / OBS pushing `rtmp://host/app/key`) over TCP and streams the
result downstream as `Caps::ByteStream{Flv}`, so the chain is
`RtmpSrc -> flvdemux -> h264parse -> ...`. The protocol is Sans-IO (`rtmp.rs`,
`RtmpSession`): the simple (non-digest) handshake publishers fall back to, the
chunk-stream reassembly (per-chunk-stream header inheritance + `Set Chunk Size`),
and the AMF0 `connect` / `createStream` / `publish` command flow (the session
emits the Window-Ack / Set-Peer-Bandwidth / `_result` / `onStatus` replies). An
RTMP audio/video message payload is exactly an FLV tag *body*, so the session
reframes the messages into an FLV byte stream that the existing `flvdemux` (§4.17)
recovers the H.264 / AAC access units from. Scope is one publisher / one stream,
H.264 + AAC, AMF0.

**RTMP egress.** `RtmpSink` (`rtmpsink.rs`, `rtmp` feature) is the inverse:
it connects out to an RTMP server and *publishes* an incoming FLV byte stream, so
the chain is `... -> flvmux -> RtmpSink location=rtmp://host/app/key`. The
protocol is Sans-IO (`rtmp.rs`, `RtmpPublisher`), the mirror of `RtmpSession`: it
sends C0/C1, drives the `connect` / `createStream` / `publish` command ladder off
the server's `_result` / `onStatus` replies, then splits the FLV stream back into
tags and reframes each as an RTMP audio/video/data message (the tag body is the
message payload). Both directions share one `ChunkReader` (the chunk-stream
reassembly) and one fragmenting `write_message` writer, so the publisher and the
session are true inverses rather than parallel re-implementations. The element
opens the socket lazily on the first buffer (after `flvmux`'s header) and drives
the publish ladder before sending media. Validated sans-IO by pitting the
publisher against the server session (an access unit survives the RTMP round
trip); live publish to a real endpoint is operator-validated. SRT, the complex
(HMAC digest) handshake some CDNs require, and multiple streams are follow-ups
(DESIGN_TODO).

**RTSP server.** `RtspServerSink` (`rtspserversink.rs`, `rtsp-server` feature)
hosts the server side of RTSP: a player connects over TCP, runs OPTIONS /
DESCRIBE / SETUP / PLAY, and the sink streams the pipeline's H.264 to the
player's negotiated UDP port as RTP, reusing the `RtpH264Packetizer`. The
protocol is Sans-IO (`rtspserver.rs`, `RtspResponder` + `RtspRequest::parse` +
`sdp_h264`): a per-session state machine answering each method and returning an
`RtspEvent` (`Setup{client_rtp_port}` / `Play` / `Record` / `Teardown`) that the
element acts on. It also speaks the publisher path (ANNOUNCE / RECORD) for a
future receive-side source. Validated end-to-end over loopback (an in-test player
handshakes and recovers every streamed access unit). Scope is one client / one
session / unicast UDP / the PLAY direction; multi-client, TCP-interleaved
transport, and the ANNOUNCE/RECORD source element are follow-ups (DESIGN_TODO).

**SRT (Secure Reliable Transport).** `SrtSink` (caller, egress) and `SrtSrc`
(listener, ingress, `srt` feature) carry an MPEG-TS byte stream over UDP with
SRT's reliable-but-low-latency ARQ — the contribution-link transport. The
protocol is Sans-IO (`srt.rs`): the 16-byte packet header + data/control wire
codec (HSv5 HANDSHAKE with the HSREQ-latency and Stream-ID extensions,
ACK / NAK loss-report / ACKACK / KEEPALIVE / SHUTDOWN), the caller/listener
handshake driver (`SrtHandshake`, induction → conclusion with a listener cookie
challenge), and the ARQ pair `SrtSender` / `SrtReceiver` (the sender buffers and
resends on NAK with the retransmit flag; the receiver reorders by wrap-aware
sequence, NAKs gaps, and delivers in order) — the same shape as the RTP
jitter/NACK path. Validated g2g↔g2g end to end over a lossy loopback (handshake +
data + a dropped packet recovered via NAK). The wire format follows the SRT
draft so real-peer interop is the design target; encryption (AES / KMREQ), the
TSBPD timing model, congestion control, and libsrt/ffmpeg interop validation are
follow-ups (DESIGN_TODO).

The remaining capture/ingress breadth — a `uridecodebin`-equivalent URI → source
layer over the autoplug registry — is tracked in DESIGN_TODO.

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
    DerivedCoupled {                              // like DerivedOutput, plus a
        derive: Box<dyn Fn(&Caps) -> CapsSet>,    //   declared passthrough-field
        passthrough: PassthroughFields,           //   mask for bidirectional
    },                                            //   field-level coupling
}
```

`DerivedOutput` is opaque: the solver can only invert it by *dropping whole
input alternatives* whose forward image can't reach the constrained output, so
a downstream pin on a field a transform passes through (e.g. a `160x120`
geometry pin behind a format-only `videoconvert`) can't narrow a ranged input
field. `DerivedCoupled` fixes that for the caps-driven transforms (videoscale /
videoconvert / audioresample): the `passthrough` mask names the fields where
output == input, and the backward sweep (`backward_field_narrow`) intersects a
downstream pin *into* those input fields (`Range ∩ Fixed = Fixed`). The closure
stays the source of truth for the retargeted fields.

The mask and the closure are two sources of truth for one fact (which fields
couple backward), so they can drift: a mask claiming a field the closure actually
retargets is unsound (the solver would narrow the input on a field the transform
rewrites). A full *closure-free* forward-derivation descriptor would remove the
duplication, but it is a deliberate non-goal: forward derivation is genuinely
imperative (a scaler branches on format membership and enforces 4:2:0 even-dims,
the cross-field validity §4.13.10 keeps out of the declarative constraint), so it
cannot be a `Copy` descriptor without re-importing exactly what was excluded.
Instead the drift is caught directly: the solver's forward step runs a
`debug_assert!` (`verify_passthrough_sound`) that every field the mask declares
passthrough is in fact repeated unchanged across *all* of the closure's output
alternatives for the concrete input. Unlike `discover_passthrough` it stays valid
for the multi-valued closures `DerivedCoupled` exists for (it checks the declared
fields, not a single output), and it flags only the unsound direction
(declared-but-not-honoured); a field the closure passes through but the mask omits
is merely a missed coupling, which is sound.

A plain `DerivedOutput` (a decoder that declares no mask) recovers the same
backward coupling automatically (M257): `discover_passthrough` probes the closure
with two distinct concrete inputs per field and marks a field passthrough when
the single output tracks it in both, so the solver narrows those input fields via
the same `backward_field_narrow` path. A `couple_passthrough_derived` extends the
coupling across the variant change a decoder/encoder makes (`CompressedVideo <->
RawVideo`), coupling the geometry / framerate both carry (`format` is retargeted
across a codec boundary, so probing never marks it passthrough). Discovery is
conservative, a field that the closure fixes or that fails either probe stays
non-passthrough, so a genuinely non-invertible closure falls back to the
alternative-drop walk unchanged. (Discovery is gated on the closure being
single-valued on the sample: a multi-valued converter, e.g. one offering
`{passthrough, retargeted}`, has no well-defined per-field passthrough, so probing
it is unsound and yields `NONE`.) The mid-stream `backward_feasible` snapshot now
recovers the same coupling (M258 / M259): the per-edge sweep is threaded the
element's startup-fixated input caps (from the solved edge set), which supplies the
concrete probe a `DerivedOutput` needs and the input variant / scalar identity. The
passthrough fields take the downstream pin's value, but every *non-passthrough*
(re-derived) field widens to `Any` (`project_passthrough_derived`, M259): the
transform re-derives that field from whatever input it gets mid-stream, so the
input edge stays unconstrained on it. Freezing it to the startup value (M258 v1)
made the snapshot reject a legitimately re-derived mid-stream geometry, the Caps-β
forward gap; with widening, a `DerivedOutput` stacked below another
format-changing transform re-derives its output on a mid-stream input change and
the runner cascades it downstream. A decoder below a geometry pin still exposes a
constrained input edge; an empty discovered mask or a missing sample imposes none.
(A closure-free `FieldTransform` that makes forward declarative too is a planned
follow-up.)

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
structured failure to the bus), while a node behind a tee cannot (a shared
upstream can't honour a per-branch reconfigure). What a behind-a-tee rejection
then does is the tee's [`FanOutPolicy`](crate::graph::FanOutPolicy): `FailLoud`
(the `add_tee` default) fails the whole run, and `AllowBranchDrop`
(`add_tee_with_policy`) drops just that branch (its arm ends, the tee removes its
now-closed sender via `broadcast_drop_closed` and keeps broadcasting to the rest)
so an optional branch (a preview that can't follow a format switch) does not kill
the essential ones. A genuine downstream error still surfaces through that branch
arm's own result, so swallowing the closed channel at the tee is safe.

`run_graph` consumes the elements it runs (it `take()`s the boxed payloads), so a
graph runs only once. Re-running (seek-and-replay after a flushing seek, retry,
A/B benchmarking) needs *fresh* elements, because real ones carry state a rewind
cannot undo (a decoder's reference frames, a source's file offset). A
[`GraphTemplate`](crate::runtime::GraphTemplate) wraps a builder closure and hands
back a fresh `Graph<GraphNode>` per `instantiate()`, which is cleaner than making
`Graph` itself reusable: that would force every element to be `Clone` or
re-initialisable in place, a contract the element traits deliberately avoid.

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

This fixes the element-side contract for `process(CapsChanged(c))`. The arm
calls `configure_pipeline(in)` (the element's new *input*) and then
`process(CapsChanged(fixated_output))` (its pre-fixed *output*). So `c` is the
element's **output** caps, not its input: the element forwards `c` downstream
(letting a strict sink reconfigure before the first frame) and records it as
`last_caps` to suppress the duplicate emit from its data path; the input is
already set by `configure_pipeline`. A format-changing transform must **not**
re-derive its input from `c` (e.g. `videoconvert` calling `accept_input`):
when its input and output are the same `Caps` variant (raw->raw), adopting the
output as the input silently turns the next frame into an unconverted
`X->X` passthrough. This only bites when an upstream transform emits a
`CapsChanged` mid-stream (the first of two stacked auto `videoconvert`s does so
on its first frame); a lone convert right after a source never receives one,
which is why the single-convert case was always correct. A decoder, whose
input (`CompressedVideo`) and output (`RawVideo`) are distinct variants, can
safely disambiguate the two callers by inspecting `c` (see `ffmpegdec`).

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
- **Real resizable consumer:** `PoolStage` (`g2g-plugins`) is the element that
  acts on a mid-stream β proposal rather than only recording it (decoders fix
  their pool at codec open): each `configure_allocation` rebuilds its
  `BufferPool` to the proposal's `min_buffers` x `size_bytes`, and frames stage
  through it, so a mid-stream geometry change visibly resizes a live pool
  (`poolstage_recascade` asserts the rebuild end to end).

This is the same machinery a future mid-stream clock change or latency
adjustment uses: cross-element mid-stream coordination becomes a coordinator
event instead of an ad-hoc back-channel.

The startup cascade runs once in reverse topological order: each element
absorbs the proposal on its output edge and re-proposes onto its input edge.
Two fan structures have non-trivial joins:

- **Diamond join (tee).** A tee's single input must satisfy *both* branches at
  once, so the branch proposals are joined by a most-restrictive intersection
  (`AllocationParams::join`): the larger size, count, and alignment win, and the
  memory domain is the most-preferred member of the two branches' *accepted
  domain sets* intersected (`AllocationParams::accepts`, a `DomainSet` bitmask;
  the preference order favours GPU-resident domains over `System`, M351). A
  single-domain branch is just `only(domain)`, so two matching single domains
  reduce to that domain and two disjoint ones to an empty set; a branch that can
  take more than one domain (a sink that reads GPU textures *or* falls back to
  System) widens its set so the join can find a domain both branches share. An
  empty intersection (a CUDA-only branch and a D3D11-only branch) is a real
  conflict, no single producer pool serves both, and fails the whole negotiation
  loud with `G2gError::AllocationConflict` rather than silently honouring one and
  copying for the other. This is distinct from `AllocationParams::merge`, the
  asymmetric fold the linear upstream walk uses where the consumer-most proposal
  legitimately dictates the domain.
- **Producer reconciliation (M351).** Domain choice is two-sided: a producer
  advertises the set of domains it can *emit* (`output_domains`, default
  `only(output_memory())`), and at the buffer-pool origin (the source) the
  joined downstream proposal is reconciled against it
  (`AllocationParams::resolve_for_producer`): intersect the accepted set with the
  producer's capability and settle on the most-preferred survivor, so a graph
  keeps the frame copy-free when both ends can and falls back to System when the
  producer cannot reach the consumer's preferred domain. No shared domain is an
  `AllocationConflict` (a genuine case for an auto-plugged domain converter, a
  later track). Reconciliation runs at the source, not at every hop: a plain
  transform is a memory-domain pass-through (it forwards whatever domain it
  receives), so enforcing its `System` default mid-cascade would wrongly reject a
  GPU proposal merely passing through. A transform that *is* a genuine domain
  producer (a hardware decoder allocating its own output surfaces) consumes the
  same contract from inside the element: its `configure_allocation` calls
  `resolve_for_producer` against its own `output_domains` to settle its output
  domain. `NvDec` does exactly this (M352): it advertises `{Cuda, System}` and
  either keeps the decoded NV12 surface device-resident (zero-copy) or downloads
  it, chosen by the negotiated proposal alone, validated end-to-end on an RTX
  3060.
- **Converter auto-plug (M354).** When no shared domain exists (the negotiation
  would otherwise fail loud), `auto_plug_domain_converters` splices a memory-domain
  converter instead. A pre-solve graph pass: for each edge it traces the producer
  domain through structural tee/demux nodes (`output_domains`) and, if disjoint
  from the consumer's declared `input_domains` (caps-free, default
  `DomainSet::ALL`), splices a caps-`Identity` converter from a registered factory
  (`Graph::insert_on_edge`, so the caps solve is undisturbed). `g2g-plugins`
  provides the CUDA factory (`Cuda->System` = `CudaDownload`, `System->Cuda` =
  `CudaUpload`), so e.g. a System NV12 source feeding `NvEnc` (CUDA-only) gains a
  `CudaUpload` with no hand-wiring. Negotiation settles a shared domain when one
  exists; the auto-plug bridges when one does not; an unconvertible pair still
  fails loud.
- **Muxer boundary.** A muxer states its per-pad demand through
  `MultiInputElement::propose_allocation_for_input(pad, caps)` (default `None`,
  so a plain container muxer imposes nothing). At startup the runner stores it on
  each input edge so the demand crosses the boundary and re-cascades up that
  branch independently (a device-resident interleave muxer asking each video pad
  for GPU buffers). Mid-stream the same crossing holds: a `CapsChanged` on one
  pad re-derives that pad's proposal and re-cascades it up *that pad's branch
  alone* via the `Recascade::target` override (the node-keyed walk would hit
  every input), leaving the other inputs untouched. The muxer's byte output has
  no memory-domain tie to its inputs, so its output-edge proposal is not
  absorbed.

#### 4.13.6 Fan-out and fan-in

`run_source_fanout` per-branch re-solves a mid-stream `CapsChanged` via
`re_solve_downstream_dyn_sink`. Branches run in independent arms, so the
re-solves are concurrent (max of single-branch cost, not sum). The default
failure policy is strict: a branch whose constraint rejects the new caps
fails the fan-out loud (`CapsMismatch`); a future `FanOutPolicy::AllowBranchDrop`
opt-in is anticipated for graceful degradation.

A tee broadcast is **zero-copy** (M213, M250). Before fanning out, the runner
calls `MemoryDomain::make_shareable` once, which turns the frame's memory into a
refcounted handle; each branch then gets a second handle via `MemoryDomain::share`,
a refcount bump rather than a copy. The GPU domains and the shared-CPU
`SystemView` are handle-shared by construction; owned-CPU `System` bytes are made
shareable by *moving* the `Box<[u8]>` into an `Arc<Box<[u8]>>` (a move, not a
re-copy, which `Arc<[u8]>` would force), and a pooled buffer into an
`Arc<PooledBuffer>` that returns to its pool once the last branch drops. The
share is read-only: a branch that mutates pays copy-on-write
(`SystemSlice::as_mut_slice` reclaims a uniquely-held `Arc` without a copy, else
deep-copies), so siblings never alias a mutation. So a decoded frame, on CPU or
GPU, fans out to several consumers (e.g. inference + display) with no per-branch
copy, where `System` previously deep-copied per branch and a GPU frame failed loud.

`run_muxer_sink` solves each `source ↔ muxer-input` pair at startup,
per-input re-solves on mid-stream change, and eagerly re-emits the muxer's
output `CapsChanged` downstream when the merged output caps change as a
function of an input change. `MultiInputElement` exposes
`caps_constraint_as_input(idx)` and `caps_constraint_for_output()` for the
solver to consult per-input.

A fan-in muxer interleaves its inputs by **presentation timestamp** (M204), not
arrival order: `InterleaveMux` buffers frames per input in an `InputAggregator`
and releases the globally earliest-PTS frame only once every still-contributing
input has one queued (`InputAggregator::take_earliest_by`), the `GstAggregator`
collect-and-pick-earliest rule. Because each input's PTS is monotonic, holding
output until every contributor has a head guarantees the released frame is
globally earliest, so a slow input never delivers a frame that should have
preceded an already-emitted one; an input that ends drops out of the merge (its
buffered tail flushed in order). Frames carry their own caps, so reordering is
format-safe. Ordering is by PTS; a container muxer needing decode-order (DTS)
interleaving keys on that instead. This is distinct from the synchronized-*round*
collection (`take_earliest_by`'s sibling `take_round`) a compositor / audio mixer
uses, where every input contributes one item per output.

The same PTS merge is also available **at the runner level**: a
`MultiInputElement` returning `input_pts_ordered() == true` is driven by
`muxer_arm_pts` instead of the default arrival-order `muxer_arm`. That arm owns an
`InputAggregator<Frame>` and calls `process(pad, DataFrame(..))` in global PTS
order (the same collect-and-pick-earliest rule), so an element wanting time-aligned
input, a multi-camera grid or PTS-synchronized compositor, gets it without
hand-rolling its own aggregator. Per-input `Eos` (flush + the merged-EOS
aggregation) and `CapsChanged` (MX-1 / MX-2 re-solve) are handled exactly as in
`muxer_arm`; only `DataFrame`s are reordered. The default stays arrival-order
round-robin, so the existing element-level mergers (`InterleaveMux`, `tsmuxn`,
`mp4muxn`) are unchanged; the runner arm is the alternative for elements that would
rather not carry the buffering themselves.

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

**Runtime request pads.** Both fan directions can grow their pad count *while the
pipeline runs*, the GStreamer request-pad analog, without an executor `spawn`: the
no-spawn `DynamicJoin` primitive (`runtime/join.rs`) is a `join_all` that also
polls a control channel and folds newly-arrived arms into the running poll set,
completing once the channel closes and every arm resolves. On the **fan-out**
side, `run_source_router_dynamic` (M310) returns a `DynamicFanoutHandle` whose
`add_branch` attaches an output branch mid-run; the branch configures from the
fan-out's *sticky caps* (the source's fixated output caps, replayed into each
branch the moment it attaches) and then receives its share of frames.
`run_source_tee_dynamic` (M319) is the *broadcast* variant: each `DataFrame` is
shared to every branch via the M250 zero-copy path (`make_shareable` once, then a
refcount handle per branch), so an inference branch and a display branch both see
the whole stream with no byte copies; round-robin (`Router` model) and broadcast
(`tee` model) share one driver, differing only in `DataFrame` distribution.
`run_aggregator_dynamic` (M320) is the **fan-in** dual: a `DynamicFaninHandle`
whose `add_input` attaches a source to a running terminal aggregator. The
aggregator declares a fixed pad capacity (`input_count`); the handle reserves the
next pad index atomically (rejecting past capacity, the M205 dark-slot bound), the
single aggregator arm owns `&mut` and fixates + configures each new pad inline
(no aliasing), and per-pad-tagged frames merge as in `run_fanin_session`. The run
ends once the handle is dropped *and* every attached input has reached EOS (the
`DynamicJoin` completion rule). In all three the pending-pad set is drained before
each blocking select, so a pad requested before a frame is never missed.
`run_muxer_sink_dynamic` adds the trailing **sink**: the muxer's merged output is
written to a sink arm rather than discarded, with the output caps coupled without
a global re-solve. Because inputs attach one at a time, the merged output firms up
as pads configure, so the muxer arm emits a `CapsChanged` to the sink whenever the
derived `output_caps` changes (the dynamic analog of the static `run_muxer_sink`
MX-2 coupling) and the sink configures against it before the first merged frame;
when every input has ended the muxer arm closes the merged link with `Eos`, ending
the sink arm. This is the `run_muxer_sink` shape extended to runtime-added inputs
(attach a late audio track to a running `muxer ! filesink`).

#### 4.13.6a Bins and ghost pads (flattening)

GStreamer's `GstBin` is a runtime container: a node in the pipeline that holds
child elements, manages their state, and exposes interior pads as *ghost pads*.
g2g implements the same user-facing capability (reusable named subgraphs +
ghost pads) but as **construction-time flattening**, not a runtime container.
The reason is the same one in §4.9.3: g2g composes typed graphs ahead of the
run, so grouping for reuse and pad exposure can happen before validation, and
the runtime never needs a hierarchy to manage.

The whole mechanism is one primitive: `Graph::merge(inner) -> NodeIdOffset`
appends another graph's nodes and edges, re-basing the merged-in `NodeId`s by
the host's current node count. Because nodes are a flat `Vec` indexed by
`NodeId` and edges carry only pad indices, the merge is a pure index shift; the
returned `NodeIdOffset` translates the inner graph's ids (`apply` / `apply_pad`)
into the host's space. A `Bin<E>` is a `Graph<E>` plus a list of interior pads
designated as ghost inputs / outputs (1:1 with one internal pad, as in
GStreamer). `Graph::add_bin` merges the bin and returns a `BinInstance` whose
`input(i)` / `output(i)` are host-graph pad ids, linked like any other pad. A
bin is never validated alone: its ghost pads are intentionally unlinked inside
the bin and acquire their peer when the host links the `BinInstance`, so the
host's `finish()` is the single validation point.

Crucially this adds **no `NodeKind` variant**: a bin's interior nodes become
first-class host nodes on flattening, so the solver (§4.13.2) and runner
(§4.13.3) drive them with zero awareness bins ever existed, and none of the
exhaustive `NodeKind` match sites change. The decode-chain splices
(`Registry::decodebin`, the `uridecodebin` / `decodebin` launch macros) already
flatten subgraphs ad hoc at the element-vector / parse-item layer; they predate
this primitive and are left as-is rather than rerouted through it.

Out of scope (a later milestone, only if needed): a runtime `NodeKind::Bin` with
recursive solve/run, per-bin state transitions, and bus-message bubbling, i.e.
GStreamer's full hierarchical `GstBin`. None of that is required for reuse,
ghost pads, or a nestable decodebin.

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

- **playbin3 (multi-stream front door)** (`std`, M376-M382). Where `playbin`
  decodes one stream, `playbin3` splits a container into *all* its streams and
  decodes each to its own sink, built on the stream-collection model: a demuxer
  announces every track as a `BusMessage::StreamCollection` (M376), the app
  selects among them via a `StreamSelectController` (M377), and the multi-output
  `MkvDemuxN` (a `MultiOutputElement`, M378) routes N elementary streams to N
  ports in one parse. `Registry::build_playbin3_graph` (M379) assembles
  `source → demux → {decode chain → sink}` per `Playbin3Port`, with each port's
  branch statically negotiated against its codec via `port_output_caps` /
  `NodeConstraint::Demux` (M380) so a real decoder configures at startup, not
  only at runtime retype. The gst-launch front door is `playbin3 uri=X` (M382):
  `parse_launch` routes a *lone* `playbin3` to a registry `Playbin3Hook`
  (`register_playbin3`, a `Default`-friendly fn-pointer slot) that probes the
  container and auto-builds the multi-stream graph. Cross-crate by design: the
  text DSL is core, the Matroska probe (`mkv_playbin3`: read a bounded prefix,
  parse `Tracks`, one branch per `forwardable_streams` entry, video→autovideosink
  / audio→autoaudiosink) is `g2g-plugins`. The hook declines (`Ok(None)`) for a
  non-`file://` URI or non-Matroska container, falling back to single-stream
  `playbin`; it supplies a Matroska byte `FileSrc` via
  `build_playbin3_graph_with_source` rather than the `file://` handler's
  MP4-self-demuxing source.

- **Gapless playback** (`std`, M383). The playbin `about-to-finish` + next-`uri`
  analog: `GaplessSrc` (`g2g-plugins`) concatenates a playlist of sources into
  one continuous, monotonically-timed stream, reusing the downstream decode chain
  across items (the read-side analog of GStreamer reusing one decodebin across
  URIs). It wraps a current `DynSourceLoop` and a shared `GaplessController`
  (core, the `SeekController`-shaped app<->source channel: an `enqueue` playlist
  queue, an about-to-finish back-channel, a latching `finish`, and a wakeful
  `wait_event` idle). The source plays the current item, posts about-to-finish
  when nothing is queued behind it (so the app enqueues the next item *during*
  playback for a seamless swap), and on the item's EOS pulls the next, rebasing
  its PTS/DTS onto the running timeline via an interposing `ShiftSink` that also
  swallows the inner item's `Eos` — so the only terminal `Eos` is the one
  `GaplessSrc` emits when the `finish`ed playlist drains. This is the source-swap
  counterpart of the M358 segment loop (which loops *one* item via a `SEGMENT`
  seek); both are poll-based with a wakeful idle. v1 concatenates same-codec items
  (a per-item caps refinement still flows via the inner source's `CapsChanged`);
  instant (flush) URI switching and an A/V offset are follow-ups.

- **Memory-feature-aware selection** (M276). The `Caps` algebra encodes media
  type, format, and geometry but *not* the memory domain a producer emits, so a
  GPU-resident decoder (`NvDec` → NV12 in `MemoryDomain::Cuda`) is
  indistinguishable from a CPU one by caps alone. Rather than thread the domain
  through every `Caps` (446 construction sites, and it is orthogonal to the
  format algebra), it rides on the auto-plug metadata, as one field of a small
  `CapabilityDescriptor` (`ElementDesc::capabilities`): `output_memory`
  (`MemoryDomainKind`, the GStreamer `memory:CUDAMemory` caps-feature analog), an
  `Acceleration` (hardware vs software, independent of the domain: an ffmpeg
  VA-API decoder is hardware yet downloads to `System`), and a numeric `rank`.
  These are set per factory via `ElementFactory::produces(kind)` / `.hardware()` /
  `.rank(n)`, all defaulting to (software, `System`, 0).

  This is deliberately *not* GStreamer's flat global rank. A single integer can't
  express that the best element is context-dependent: a hardware decoder that
  keeps frames on the GPU beats a faster one that forces a PCIe download when the
  consumer is GPU-resident (g2g measured exactly this, the NVDEC-to-system-memory
  floor). So `CapabilityDescriptor::score(ctx)` ranks a candidate against a
  `SelectionContext { preferred_memory, prefer_hardware }`: a memory-domain match
  dominates, then a hardware preference, and `rank` is only the deterministic
  tiebreaker among otherwise-equal candidates (the explicit-override knob, the
  genuinely useful 20% of GStreamer's rank). `find_chain_with(.., ctx)` /
  `Registry::{autoplug,autoplug_names}_with(.., ctx)` score-order which candidate
  is *tried first*; it is still breadth-first (a shorter chain always wins), and a
  default `ctx` scores every candidate equally, so the visit order is registration
  order and a plain pipeline is unchanged (`NvDec` registered last never hijacks a
  CPU path). `find_chain_preferring` / `{autoplug,decodebin}_preferring(.., domain)`
  remain as the memory-only special case. Ranking matters *only* on the auto-plug
  path; an explicit typed graph names its element, so the descriptor never touches
  the core. Deriving `ctx` automatically from a downstream consumer's accepted
  input memory (so a plain `decodebin` into a CUDA sink prefers `NvDec` without an
  explicit request) is the remaining follow-up.

#### 4.13.10 Current limits

The solver is **arc consistency** (constraint propagation over per-link caps),
not a complete CSP search. That bounds exactly where it is complete and where it
is not:

- **Linear chains are complete.** A linear pipeline is a tree of binary
  (adjacent-link) constraints, and arc consistency is complete for
  tree-structured binary CSPs: if a satisfying assignment exists it is found.
  With `DerivedCoupled`'s field-level coupling (§4.13.1), a downstream pin on a
  passthrough field couples back through any number of passthrough transforms
  (`videoscale ! videoconvert ! caps`, and deeper). This family is closed.

- **Backward coupling through a format-changing (`DerivedOutput`) transform is
  partial, over its *invertible* fields.** A `DerivedOutput` is opaque, but its
  invertible fields are recovered by probing (`discover_passthrough`, M257): a
  downstream pin on a passthrough field couples back through a decoder / rescaler,
  in both the full-chain solve and the mid-stream snapshot (M258 / M259). A field
  the transform genuinely *re-derives* (a scaler's geometry) still cannot be
  inverted: a downstream pin on it does not narrow the input, and the snapshot
  leaves the upstream unconstrained on it (so a re-deriving transform mid-stream
  picks freely and the pin is enforced loud downstream if violated). This is the
  arc-consistency boundary, not a missing feature: a partial inverse over the
  invertible fields is exactly what is modelled.

- **Non-tree topologies: arc consistency plus a backtracking fixation.** Arc
  consistency is incomplete on cyclic constraint graphs, so for a true *diamond*
  (a tee whose branches diverge through format-changing transforms and re-converge
  at a fan-in) the per-link sweep can leave each edge with a locally-valid domain
  whose *greedy* per-edge fixation picks a jointly-impossible combination (two
  branches mapping the shared tee value to outputs whose alternative orders
  disagree). `solve_graph` therefore fixates by **backtracking search** over the
  arc-consistency-narrowed domains, not greedily: it assigns one fixated `Caps`
  per edge in id order, trying each edge's greedy choice first and pruning the
  moment a fully-assigned node violates its relation (a tee's branches must all
  carry its input; a transform's `(in, out)` must be a real `Mapping` pair /
  `Identity` equality / `f(in)` image). A chain or an independent fan-out has
  single-candidate domains and so fixates byte-for-byte as the greedy code did;
  only a genuinely coupled diamond explores alternatives, and one with no
  jointly-valid assignment fails loud (`NoConsistentFixation`). Diamonds are now
  solved, not a caveat. (A muxer that itself *couples* its input pads, beyond the
  per-pad accept sets the constraint vocabulary expresses today, would be the next
  step up; the search already has the shape to enforce it once such a constraint
  exists.)

- **Cross-field validity within one element is not modelled.** Constraints
  *among an element's own caps fields* (a 4:2:0 format requiring even dimensions,
  chroma siting) are non-binary and are deliberately kept out of the declarative
  constraint: caps fields stay independent within an alternative, and an element
  enumerates valid combinations as separate `CapsSet` alternatives instead. The
  hard cases were judged not worth a declarative encoding.

- **Allocation is a separate cascade.** Buffer-pool / stride / alignment
  negotiation (§4.13.5, the M12 allocation query) runs after caps fixation, not
  folded into the caps CSP. A downstream allocator whose layout requirement
  should feed back into the *caps* choice is not expressed; this is the most
  likely future pressure point as real GPU/hardware allocators land.

The fixation step is now a bounded backtracking search (above), so a diamond is
solved rather than greedily mis-fixated. Full *path consistency* during the
narrowing sweep (versus arc consistency plus search at fixation) is still not
implemented, but the search closes the practical gap: every shape that arises is
either complete or fails loud, and a coupled diamond with a satisfying assignment
now finds it instead of mis-fixating.

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
(`GstSegment`-equivalent), with `clip`, `for_flush_seek` (which resets `base`
so running time restarts after a flush), and `accumulate_seek` (M211, the
non-flushing seek: `base` advances to the running time playback has already
reached, so the running-time line stays monotonic across the seek, the gapless /
segment-seek / loop case). `PipelinePacket::Segment` is the
carrier: the runner emits an opening SEGMENT and every element forwards it
(transforms/decoders forward, sinks consume), the same way `Flush` already
flows. A `SeekController` (runtime) is a cloneable handle the application holds;
a seek-aware source's run loop polls `take_pending()` between frames and, on a
flushing seek, emits `Flush`, repositions, emits the post-flush `Segment`, and
resumes, so a seek reaches the source GStreamer-style (upstream) without a
back-reference. `Mp4Src` is the first real repositioning source (M148: flushing
seek, keyframe `SNAP_BEFORE`, re-prepended parameter sets), and `SyncSink` maps PTS
to running time through the `Segment` and clips pre-target frames so accurate seek
presents the exact requested frame (M149). A non-flushing seek (M211) emits only
the accumulating `Segment` (no `Flush`), so the source keeps producing on a
continuous running-time line. Reverse playback (M211/M212, `Seek::reverse`,
`rate < 0`) needs no sink-specific code: the source emits frames newest-PTS-first
over `[start, stop]`, and `SyncSink` schedules each by `Segment::to_running_time`
(which measures reverse from `stop`) and clips via `contains`, so descending PTS
maps to ascending running time and presents in the correct visual order, the
`Segment` abstraction generalizing the sink to negative rate transparently.
**Trick-mode KEY_UNIT** frame selection (present only keyframes for fast scrub)
is done (M226): `FrameTiming::keyframe` carries a per-frame flag (set by
`h264parse` from `h264_au_is_keyframe`, and by `mp4src` / `fmp4demux` from the
container sync-sample / `trun` keyframe flag), a `TRICKMODE` seek sets
`Segment::key_units_only` in `from_seek`, and `SyncSink` drops non-keyframe frames
under such a segment before scheduling them (counted by `trick_dropped()`).
**Segment playback / gapless looping** (M358, the `GstSeekFlags::SEGMENT` analog)
is consumed through the `SeekController`, not a new packet: g2g has no
`SEGMENT_DONE` `PipelinePacket` (it would force a new control variant through
every element's exhaustive match), so the controller carries it on the same
app<->source channel a seek already uses. A `SEGMENT`-flagged seek runs the
source to `stop`; instead of `Eos` the source calls `notify_segment_done(stop)`
and parks (polling) for the app's next move. The app observes
`segment_done_count()` / `take_segment_done()` and re-arms a *non-flushing*
`SEGMENT` seek to loop (so `accumulate_seek` advances `base` by one span per
iteration, gapless, no `Flush` downstream) or calls `shutdown()` to end the loop,
at which point the idle source emits `Eos`. The idle park is **wakeful**
(`SeekController::wait_event`, M359): the source `await`s a future that resolves
when `seek` / `shutdown` wakes the registered waker, so a looping source between
loops costs nothing (no busy-poll), the poll-free analog of GStreamer pausing the
source task. `Mp4Src` is the first real source to loop on `SEGMENT` (M359): it
clips playback to the segment `stop`, reports segment-done at the boundary, and
parks on `wait_event` for the app's loop seek (non-flushing, snapping to the
keyframe at or before the target so a decoder resumes cleanly) or `shutdown`. It
also now honours non-flushing repositioning seeks (accumulating `Segment`, no
`Flush`), not just flushing ones. **Re-preroll when paused (M360).** A paused,
prerolled pipeline backpressures its source, so a flushing seek issued now would
never take effect (the held sink never drains). `StateController::request_repreroll`
(called by the app alongside the seek) bumps a preroll generation; `flow_gate`
takes the arm's generation and reopens for a stale one, so each sink arm
re-prerolls. The arm drains the stale pre-seek frames (discarding, not presenting)
until the `Flush`, then prerolls the post-flush target and re-fires `AsyncDone`,
so scrubbing a paused pipeline updates the shown frame. **Byte-source seek
(M361).** `FileSrc` is BYTES-format seekable (`with_seek`): a flushing seek
repositions the file read to a byte offset and emits `Flush`. **Demuxer seek
(M362-M366).** A byte-stream demuxer (a transform with no random access) becomes
seekable by driving that upstream byte source. A shared `DemuxSeek` helper turns
an app time seek into an upstream byte-seek to offset 0, drops in-flight pre-seek
input until the returned `Flush`, resets the demuxer's parser, then discards
decoded units until the keyframe at/after the target and emits a resume
`Segment` (correct for any container without an index; a re-scan, with an
index-derived offset a later optimization). All five carry it
(`fmp4demux` / `tsdemux` / `mkvdemux` / `flvdemux` / `oggdemux`), each using its
own keyframe signal (the container flag, or `annexb::au_is_keyframe` for TS whose
units have none; every audio packet is a resync point, and `oggdemux` now
accumulates an Opus PTS from the TOC byte). **Adaptive segment seek (M367).** The
adaptive sources `HlsSrc` / `DashSrc` are TIME-seekable (`with_seek`): unlike the
BYTES-format `FileSrc`, an app time seek resolves to the media segment containing
the target (HLS walks cumulative `#EXTINF` durations; DASH maps the target onto
the `SegmentRef` `$Time$` line), then the source emits `Flush`, jumps to that
segment, re-emits the fMP4 init segment (the downstream demuxer reset on the flush
needs its `moov` again), emits the post-flush `Segment` at the segment start, and
resumes there. This is the CMAF / DASH segment-transition case (clamped to the
last segment; a target past the end lands there). Discontinuity / multi-period
boundary `SEGMENT` emission is a separate later concern.

### 4.15 Bus and Observability

The pipeline `Bus` (§4.9.1) is a many-producer / single-consumer channel for
out-of-band events, so an element notifies the application without a
back-reference. `BusMessage` covers the lifecycle and quality signals an
application reacts to:

- `StreamStart`, `Eos`, `Error`, `Warning`, `Info(String)` — stream lifecycle,
  faults, and non-fatal status. `StreamStart` is posted by the source arm before
  a source produces (one per source), bracketing each stream with its `Eos`
  (`GST_MESSAGE_STREAM_START`); `Info` is the third severity below `Warning`,
  element- / app-posted for status that is not a problem (`GST_MESSAGE_INFO`).
- `DurationChanged { duration_ns }` — the total stream duration became known
  (§4.15's query handle is the pull side; this is the push notification), posted
  by the source arm from `SourceLoop::query_duration` (`GST_MESSAGE_DURATION_CHANGED`).
- `Tag(TagList)` — container / stream metadata, posted out of band
  (`GST_MESSAGE_TAG`).
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

**Element-granular logging (`g2g-core::log`, M179)** is the complementary
diagnostic channel, the `GST_DEBUG` analog, for developer tracing rather than
application-facing events. A record carries a `category` (the element *type*,
e.g. `"VideoFlip"`, the filtering key) and an optional `instance` name (the
element *instance*, e.g. `"VideoFlip0"`). `LogLevel` runs `Error` (most severe)
through `Trace`, matching GStreamer's numeric levels; a per-category threshold
table (a default plus overrides) decides what is emitted, mirrored into an atomic
so a disabled `g2g_trace!` in a hot loop costs one atomic load. The macros
(`g2g_error!` .. `g2g_trace!`) take a `LogSource` (an element via `self`, or a
`Target` for logging about a named element) then a `format_args!` message,
checked against the threshold before formatting. Records route to an installed
`LogSink`; the `std` feature provides a stderr sink and `init_from_env`, which
reads `G2G_DEBUG` (a `GST_DEBUG`-style `*:warning,VideoFlip:trace` spec). The DAG
runner assigns each element a `<category>N` instance name before negotiation (the
`videotestsrc0` convention) via `set_instance_name`, logs each element's
addition, and an element that logs about itself (it implements `LogSource` with a
stored name) carries that name in its lines. Pulls no external logging crate, so
it holds on the `no_std` baseline; the sink is the RTOS plug-in point (UART /
RTT). The `tracing` feature (M202) adds a `LogSink` that forwards records to the
`tracing` crate (the `g2g` target, `category` / `instance` as fields), so a host
on `tracing-subscriber` / OTLP / tokio-console receives g2g's logs in its
existing pipeline; `log::init_tracing()` installs it and defers filtering to the
subscriber.

**Application queries: position and duration (M203).** A media-player UI needs to
poll *where* playback is and *how long* the stream is, GStreamer's `POSITION` /
`DURATION` queries. GStreamer sends a query object upstream along the pads; g2g
pushes forward and composes paths statically (as with the latency fold, §4.13's
`LatencyReport`), so instead the runner *publishes* into a shared
`runtime::PipelineProgress` handle the application holds and polls
(`position()` / `duration()`, ns). This inverts the `SeekController` idiom: there
the app writes a pending seek and the source reads it; here the runner writes and
the app reads. **Position** is published by the DAG runner's sink arm, mapping
each consumed buffer's PTS through the active segment to stream time (the sink is
the position authority, exactly as a GStreamer sink answers from its segment plus
last buffer), so it needs no element cooperation. **Duration** is the source's
answer: `SourceLoop::query_duration() -> Option<u64>` (default `None`, so a live
source stays "unknown"), polled by the source arm before producing; `Mp4Src`
reports it from the `mdhd` box. A first duration also posts
`BusMessage::DurationChanged` as a push notification. `run_graph_with_progress`
wires the handle in; the handle is plain atomics behind an `Arc`, so reading it
from the app thread while the pipeline runs needs no lock.

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
  role, properties, and pad templates, the `gst-inspect` analog. The dump is
  GStreamer-shaped (M178): a "Factory Details" header from the element type's
  `metadata()` (`ElementMetadata { long_name, klass, description, author }`, the
  `gst_element_class_set_static_metadata` analog, a zero-cost opt-in like
  `properties()`), then pad templates, then an "Element Properties" section where
  each `PropertySpec` carries its `default`, numeric `range`, enum `values`, and
  read/write `flags` alongside the blurb. `element_listing()` is the no-arg index,
  `name: Long-name` per element.
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

**Dynamic plugin loading (M201).** Beyond build-time registration (a crate that
calls `Registry::register_*`, the primary extension path), a third party can ship
a native element as a dynamically loaded `.so`, the analog of GStreamer's scanned
plugin path. They build a `cdylib` against the published `g2g-core` plus the
`g2g-plugin` SDK and use its `declare_plugin! { elements: [ (name, Type, build) ] }`
macro, which emits two C-ABI entry points: `g2g_plugin_abi` (returns the ABI tag)
and `g2g_plugin_register(&mut Registry)` (registers the elements, body in
`catch_unwind` because unwinding across `extern "C"` is UB). A host built with the
`plugin-loader` feature (`g2g_plugins::plugin_loader`, over `libloading`)
`dlopen`s the object, reads its tag, and registers it only on an exact match;
`g2g-launch` / `g2g-inspect` expose this via `--plugin <path>` and
`$G2G_PLUGIN_PATH`.

The hard constraint is that Rust has no stable ABI, so a plugin and host must
share the same `g2g-core` version, the same `rustc`, and the same
layout-affecting features. Two features change in-memory layout across the
boundary: `metadata` resizes `Frame` (the `FrameMetaSet` side-channel) and
`multi-thread` changes the `Send` bound on the boxed element trait objects.
`g2g_core::ABI_VERSION` (a `build.rs`-computed string folding version + `rustc` +
those features) is embedded in each plugin and checked by the loader, which
refuses a mismatch with a clear `AbiMismatch` error rather than risk passing a
differently-laid-out `Frame` or trait object across the boundary (undefined
behavior). Each loaded `libloading::Library` is held for the life of the process:
the registered factories are `fn` pointers into its mapped code, so dropping it
would be a use-after-free with no back-pointer to catch it. This version+toolchain
lock is the v1 design; an `abi_stable`/`stabby` facade over the element traits is
the later upgrade for cross-toolchain binary plugins, and a pure C-ABI shim was
rejected (it loses the ergonomic Rust trait). The whole path is exercised
out-of-tree by `g2g-plugins/tests/fixtures/example-plugin` +
`tests/plugin_loader_dlopen.rs`.

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
`Mp4Src` / `Mp4Sink`. The TS muxer (`g2g-plugins::mpegts::TsMuxer`) is the
inverse path (M114), wrapping access units back into PES + 188-byte packets with
a real PSI CRC. It is multi-stream (M207): `with_streams` builds one program
carrying N elementary streams, each on its own PID and named in one PMT. The
single-input `tsmux::TsMux` element wraps a one-stream muxer (`! mpegtsmux !`);
the multi-input `tsmuxn::TsMux` (a `MultiInputElement`) muxes A+V, interleaving
access units across inputs by PTS via the M204 `take_earliest_by` merge so the
multiplex is decode-ordered. The `mpegtsmux` name is registered both as the
single-input launch element and (M208) as a fan-in muxer, so the text parser
picks `tsmux::TsMux` for one input and `tsmuxn::TsMux` for several by link degree
(`v.! m.  a.! m.  mpegtsmux name=m`), mirroring gst's request sink pads. Multi-
program selection and PCR-based timing are follow-ups.

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
lacing (Xiph / EBML / fixed) is split (M113), so multi-frame audio blocks demux.
The `Cues` index is parsed into a time -> Cluster-byte-position map
(`cue_seek_offset`, M373), and `MkvDemux` seeks through it in three tiers
(`poll_seek`): with `Cues` parsed it byte-seeks straight to the target Cluster
(`DemuxSeek::poll_request_indexed`), keeping Tracks / TimestampScale across the
mid-segment landing (`reset_keeping_tracks`); with only a `SeekHead` locating an
end-of-file `Cues` it prefetches them first (M374: a byte-seek to `Cues`, parse,
then `begin_indexed_seek` to the target Cluster, the internal prefetch flush
consumed so downstream sees one only on the real seek); with neither it re-scans
from offset 0 (M364). (`CueClusterPosition` / `SeekPosition` are relative to the
Segment data start, which the parser tracks.) The MKV muxer (`matroskamux`: `MatroskaMuxer` + the
`MkvMux` element) is the inverse path (M115), writing the EBML header, an
unknown-size Segment, Tracks, and one Cluster per frame, with the `webm` DocType
for the WebM codec subset. Scope is one Segment / one track with definite-size
Clusters; unknown-size Clusters (live read), writing a `Cues` element, and
multi-track muxing are follow-ups.

The Ogg demuxer (M116) is the third, the same parser + element split on
`Caps::ByteStream{Ogg}`. `g2g-plugins::ogg::OggDemuxer` parses RFC 3533 pages
(sync to "OggS", frame packets via the segment-table lacing with cross-page
reassembly, sniff the codec from the first packet's `OpusHead`, skip the setup
headers), and `OggDemux` emits the Opus audio packets as `Caps::Audio{Opus}` with
the channel count refined from `OpusHead`. The container is auto-detectable
(`typefind` "OggS", `filesrc bytestream-format=auto`). Granule-position timing,
Vorbis output, and an `oggmux` are follow-ups.

The FLV demuxer (M119) is the fourth, on `Caps::ByteStream{Flv}`.
`g2g-plugins::flv::FlvDemuxer` parses the flat FLV tag stream (the "FLV" header,
then `PreviousTagSize` / tag pairs, each tag's 11-byte header framing its body),
and `FlvDemux` forwards the H.264 (AVC) video and AAC audio media access units
with their millisecond timestamps, selected per `FlvStream` (h264 | aac, default
h264) like `TsDemux`. The sequence-header tags (codec config) and the
`onMetaData` script tag are skipped, the codec-config / extradata side channel
being a shared demuxer follow-up. The container is auto-detectable (`typefind`
"FLV", `filesrc bytestream-format=auto`). The FLV muxer (`flvmux`:
`g2g-plugins::flv::FlvMuxer` + the `FlvMux` element, M120) is the inverse path,
wrapping one elementary stream's access units back into FLV tags (media frames
only; the sequence header / extradata and the `onMetaData` script tag are
follow-ups). With MP4 (`Mp4Src`/`Mp4Sink`), MPEG-TS, Matroska/WebM, Ogg, and FLV,
the demux/mux coverage spans the major containers.

Adaptive streaming sits one layer above these demuxers: an HTTP byte source feeds
a playlist/manifest-driven source that fetches media segments and hands them to
the matching byte-stream demuxer. `g2g-plugins::httpsrc::HttpSrc` (the `http-src`
feature, `reqwest`) GETs a URL and streams the body as `Caps::ByteStream` chunks,
the fetch layer the others share. Because a manifest/segment URL is
attacker-controlled, the shared `fetch::get_bytes`/`get_text` never buffer an
unbounded body: each accumulates the response chunk-by-chunk against a cap
(`MAX_MANIFEST_BYTES` 16 MiB for playlists/MPDs/keys, `MAX_SEGMENT_BYTES` 256 MiB
for one media segment), failing loud when an honest `Content-Length` or the
streamed running total exceeds it, so one oversized reply cannot exhaust memory.
`hlssrc::HlsSrc` (`hls`) parses an RFC 8216
`.m3u8` (the pure `no_std` `hls` parser: master variants for bandwidth-capped ABR,
media segments), selects a variant, and streams its segments, MPEG-TS into
`tsdemux` or fMP4/CMAF (signalled by `#EXT-X-MAP`, probed at negotiation) as
`ByteStream{IsoBmff}` into `fmp4demux`. It reloads a no-ENDLIST live playlist on an
interval, playing each new segment once by media sequence. With `with_abr()`
(M371) it is throughput-adaptive: a shared `abr::BandwidthEstimator` keeps an EWMA
of measured download throughput (bytes over elapsed `monotonic_ns`) and yields an
effective bandwidth cap (estimate scaled by a safety factor, bounded by
`max-bandwidth`); the run loop feeds that cap to the existing `MasterPlaylist`
selection, re-picks the best-fitting variant after each segment, and on a change
swaps the active media playlist and re-emits the init, keeping the time-aligned
segment index. Off by default (a fixed up-front variant). Single-file CMAF is
supported through `#EXT-X-BYTERANGE` (and `#EXT-X-MAP`'s `BYTERANGE`): a segment
carrying one fetches only its sub-range with an HTTP `Range` request (M368), the
offset continuing from the previous sub-range of the same resource when the tag
omits an explicit `@offset`; a server that ignores the `Range` (replies `200`)
is handled by slicing the requested window from the full body.
`#EXT-X-KEY:METHOD=AES-128` segments are decrypted in place (AES-128-CBC via
`aes`/`cbc`, key fetched from the key URI and cached, IV explicit or derived from
the media-sequence number). `METHOD=SAMPLE-AES` encrypts only the media samples
inside the container, so it is handled after demux by the
`sampleaesdecrypt::SampleAesDecrypt` transform (`tsdemux ! sampleaesdecrypt !
h264parse`): per the Apple TS sample-encryption format it AES-128-CBC decrypts
H.264 slice NALs (32-byte clear leader, 16-encrypted / 144-clear pattern,
emulation-prevention aware, IV reset per NAL) and AAC ADTS frames (ADTS header +
16 clear bytes, then whole-block CBC). The key/IV reach it either configured
directly or, in the HLS chain, auto-wired: `HlsSrc` fetches the `#EXT-X-KEY`
material and publishes it into a shared key handle the decryptor reads, forwarding
the sample-encrypted segments undecrypted (the demuxer needs the clear framing).
For fMP4/CMAF, SAMPLE-AES maps to the `cbcs` Common Encryption scheme
(ISO/IEC 23001-7), handled inside `fmp4demux`: the init segment's `encv`/`sinf`/
`tenc` give the crypt:skip pattern (1:9 for video) and constant IV, each fragment's
`senc` gives the per-sample clear/protected subsample ranges, and the protected
ranges are AES-128-CBC decrypted (IV reset per subsample, chaining over the
encrypted blocks only) using the same shared key handle `HlsSrc` fills. A clear
track stays a normal demux; an encrypted track with no key fails loud.
`dashsrc::DashSrc` (`dash`)
is the MPEG-DASH analog: it parses a static MPD (the `mpd` parser, via
`roxmltree`), selects a Representation, and streams its fMP4 init + media segments
into `fmp4demux`. A Representation addresses its segments by a `SegmentSource`, one of three: a
`SegmentTemplate` (the `@duration` profile or a `SegmentTimeline`, the `<S t d r>`
entries expanded into per-segment times, addressed by `$Number$` or `$Time$`); a
`SegmentList` (M369: an explicit ordered list of `<SegmentURL>`, each a `@media`
URL and/or a `mediaRange` byte range of the `BaseURL` resource, with an
`<Initialization>`); or a `SegmentBase` (M370: one resource whose fragment byte
ranges live in a `sidx` Segment Index box at `indexRange`, fetched and parsed at
run time via `parse_sidx` + `Sidx::subsegments`, the index bytes never pushed
downstream). All three resolve to one `ResolvedSegment { url, byte_range, time }`
list, so a range-carrying entry fetches just its sub-range with an HTTP `Range`
request, the DASH analog of HLS `#EXT-X-BYTERANGE`, letting a single-file CMAF
DASH stream play. A dynamic (live) MPD is reloaded on its `minimumUpdatePeriod`,
each new segment played once (tracked by start time), ending when the manifest
turns static, the same shape as the HLS live reload. `with_abr()` (M372) makes it
throughput-adaptive on the same shared `abr::BandwidthEstimator` as `HlsSrc`: a
`load_rep` helper resolves any Representation (Template / List / `sidx`-fetched
SegmentBase) into the run loop's segment/timescale/init working set, and the
estimate-derived cap drives both the per-reload pick and a per-segment
re-selection (so a static VOD adapts within one pass), re-emitting the init on a
switch. The wall-clock `@duration` live profile and multi-period are follow-ups
(DESIGN_TODO).

### 4.18 Subtitle Overlay (`textoverlay`)

`textoverlay::TextOverlay` (M171) is the `textoverlay` / `subtitleoverlay`
analog: it renders timed subtitle text onto a raw video frame. The path splits
into two `no_std` pieces feeding one element:

- **`subparse`** parses SRT (SubRip) and WebVTT into a common timed `Cue`
  (`{ start_ns, end_ns, text, settings }`). Both formats are blank-line-separated
  blocks with a `start --> end` timing line and text on the following lines, so
  one block walker covers both: the shared timestamp parser accepts the SRT comma
  and the WebVTT dot fractional separators plus the WebVTT short `MM:SS.mmm` form;
  leading lines (SRT index, WebVTT cue id) before the `-->` line are ignored; the
  `WEBVTT` header and `NOTE` / `STYLE` / `REGION` blocks are skipped; inline
  markup (`<i>`, `<c.class>`, inline cue timestamps) is stripped. BOM and CRLF are
  tolerated. Malformed blocks are skipped rather than failing the parse, the way
  players tolerate dirty files. The WebVTT cue settings after the end timestamp
  are parsed into `CueSettings { position, line, align }` (the placement subset
  the bitmap overlay honours; `size` / `vertical` / `region` are recognised but
  not applied).
- **`bitmapfont`** is an embedded 8x8 bitmap font (MSB = leftmost column) so the
  baseline draws glyphs with no font file or rasterizer. It is an all-caps font
  (A-Z, 0-9, space, common punctuation; lowercase folds to uppercase).

`TextOverlay` is an RGBA8-in / RGBA8-out identity transform on the pixels
(`VideoConvert` upstream for other formats) except for the active cue text. By a
linear scan (subtitle tracks are small) it draws *every* cue covering the frame's
`pts_ns`, not just the first: WebVTT (and SRT) allow overlapping cues to show at
once. Each cue is placed independently from its `CueSettings`: `position` (% of
width) is the horizontal anchor and `align` (start / center / end) decides how
the box extends from it; an explicit `line` (% of height) places the box
vertically, while auto-`line` cues stack upward from the bottom in cue order so
overlapping subtitles don't collide. Each cue draws over its own translucent
backing box, integer-scaled to the frame height. Cues are set programmatically (`from_srt` /
`from_webvtt`) or, on `std`, through the `location=` property loading a `.srt` /
`.vtt` file (format by extension, else content sniff); the element is registered
as `textoverlay` for the `gst-launch` text parser. This mirrors the analytics
overlay's CPU baseline (§5): the no_std bitmap renderer is the portable path, and
a mixed-case TrueType `vello` GPU backend (and the `clockoverlay` / `timeoverlay`
siblings) is the planned companion.

### 4.19 Native WebRTC (`str0m`)

The WebRTC elements are built on **[str0m](https://github.com/algesten/str0m)**, a
**sans-IO** WebRTC stack (ICE / DTLS / SRTP / RTP as a pure state machine): g2g
owns the `UdpSocket` and the timer and drives str0m's `poll_output` /
`handle_input` loop, exactly the contract the `srt` and `rtspserver` modules
already follow. str0m's pure-Rust **`rust-crypto`** backend is selected, so there
is no OpenSSL / libnice system dependency. Everything lives behind the opt-in
`webrtc` feature (it raises the effective MSRV above the workspace floor, so it is
off by default and the no_std baseline is unaffected). This is the native,
server-grade counterpart of the browser-only data-channel `WebRtcSrc` (§6.3).

**Element family.** One PeerConnection can carry one track per element or N tracks
in a session element; the shape is chosen by which trait the element implements,
and each maps to a terminal runner from the fan-in / fan-out family (§4.13.6):

| Element | Tracks | Direction | Trait | Runner |
| :--- | :--- | :--- | :--- | :--- |
| `WebRtcSink` | 1 | send (WHIP) | `AsyncElement` (sink) | linear |
| `WebRtcWhepSrc` | 1 | recv (WHEP) | `SourceLoop` | linear |
| `WebRtcSessionSink` | N | send (WHIP) | `MultiInputElement` | `run_fanin_session` |
| `WebRtcWhepSessionSrc` | N | recv (WHEP) | `MultiOutputSource` | `run_fanout_session` |
| `WebRtcDuplexSession` | N | sendrecv | `MultiDuplexSession` | `run_duplex_session` |

The one-track sink/source keep the `Rtc` on a spawned task and hand access units
over a bounded channel, so the element itself never touches the `Rtc` and stays
`Send`. The multi-track session sink is a terminal `MultiInputElement` (no
downstream sink — the network is the destination); `run_fanin_session` fans N
sources into it over one tagged `(input, packet)` channel. The session source is
the mirror: a terminal `MultiOutputSource` (0 inputs → N outputs) driven by
`run_fanout_session` into N sinks.

**The duplex shape.** Bidirectional sendrecv needs an element that is *at once* a
sink (for the tracks it publishes) and a source (for the tracks it receives) over
**one** connection — which neither the fan-in nor the fan-out session runner could
express. `MultiDuplexSession` is that union: N send inputs **and** M recv outputs,
driven by `run_duplex_session` (the union of the two session runners). A single
`run(inbound, out)` owns the connection and `select`s over the inbound send
packets (`DuplexInbound`) and the network, pushing received frames to `out`; the
send and recv halves therefore share `&mut self` directly with **no detached
task**, unlike the send-only session which spawns the `Rtc` onto its own task to
dodge `process` / run-loop aliasing.

**Signaling.** WHIP (egress) and WHEP (ingress) are the same wire move — an
`application/sdp` POST of the local offer that returns the remote answer (reqwest,
`webrtc_util::post_sdp`); the media server is the relay in the middle, so there is
no peer-to-peer mode for WHIP/WHEP. WHIP/WHEP are unidirectional by spec, so
sendrecv cannot use them: the duplex session instead exchanges SDP **directly**
between two peers over an `SdpChannel` (an in-process offer/answer transport for a
P2P loopback; a real SFU signaller — LiveKit, etc. — plugs into the same seam).
The two roles discover their m-line `Mid`s differently and this asymmetry is
load-bearing: the **offerer** captures its `Mid`s from `SdpApi::add_media`'s
return, while the **answerer** learns them from `Event::MediaAdded` after
`accept_offer` (str0m does not emit `MediaAdded` for media the local side added).

**ICE / NAT traversal.** `webrtc_util::add_ice_candidates` always adds the socket's
host candidate and, when a STUN server is configured, a server-reflexive candidate
discovered by a hand-rolled RFC 5389 Binding on the ICE socket; candidates ride in
the SDP, so a same-host P2P pair connects over localhost with no STUN. For the NAT
cases a reflexive candidate cannot punch through, a hand-rolled TURN client
(`turn.rs`, RFC 5766/8656: Allocate with long-term auth, Send/Data indications,
CreatePermission, periodic Refresh) provides a relay. str0m only offers
`Candidate::relayed`; the data plane is the run loop's job — a relayed pair's
transmits all carry `source == relay_addr`, which is the routing signal to wrap
the datagram in a TURN Send indication (direct host/srflx paths are untouched).

**RTCP feedback** rides the §4.13 reverse channel. A remote PLI
(`Event::KeyframeRequest`) becomes a `Reconfigure::ForceKeyframe` walked upstream
via `AsyncElement::take_reconfigure` to the encoder (`Av1Enc` forces a rav1e IDR);
ingress originates PLI on a mid-GOP join. str0m's BWE (`Event::EgressBitrateEstimate`,
TWCC/REMB) becomes `PushOutcome::Bitrate` via `take_bitrate`, and the encoder
retargets (rav1e by a hysteresis-gated context rebuild).

**Codec plumbing.** A `Track` enum unifies the per-track facts WebRTC needs to
agree on: codec (H.264 / Opus), m-line `MediaKind`, and the RTP clock (90 kHz /
48 kHz), with `media_time` mapping a nanosecond PTS onto the track's RTP
timestamp. H.264 crosses the boundary as **Annex-B** (the pipeline convention,
§4.11.4): str0m's packetizer splits NAL units and its depayloader emits start-code
framing. A receive-side video element advertises a `Dim::Range` /  `Rate::Range`
placeholder rather than `Dim::Any`, because geometry is only known from the in-band
SPS and `fixate()` (§4.13) rejects `Any` at negotiation; a downstream `H264Parse`
recovers the real dimensions.

**Validation status.** On-network validated against a local mediamtx (single-track
WHIP/WHEP and multi-track A/V) and by in-process P2P loopbacks on localhost (video
and full A/V sendrecv); the structural `webrtcbin` parity — one connection, N
tracks, BUNDLE, sendrecv, PLI, BWE — is in place. What remains is maturity rather
than architecture (browser interop, real remote-NAT / TURN / LiveKit Cloud runs,
launch-registry wiring for the session elements, renegotiation, data channels /
simulcast / FEC); `DESIGN_TODO.md`'s "WebRTC" item carries the tiered list.

### 4.20 Developer Tooling: DOT Visualization

`g2g_core::dot` renders a pipeline graph as Graphviz DOT, the
`GST_DEBUG_DUMP_DOT_DIR` analog: `Graph::to_dot` (pre-validation) and
`ValidatedGraph::to_dot` (post-`finish`) emit a `digraph { .. }` a developer
renders with `dot -Tsvg`. It is pure `no_std + alloc` string formatting (no I/O),
so it builds on every target the core does, embedded included.

Because the graph carries an opaque element payload `E`, node display names come
from a caller-supplied `Fn(NodeId) -> Option<String>`; returning `None` falls
back to the node's structural kind, the right answer for a `tee` / `mux` that
carries no element. Nodes are role-coded by shape and fill (source / sink /
transform boxes, a `tee` diamond, a muxer trapezium). Edges are annotated from a
`DotAnnotations { edge_caps, edge_memory }`, both indexed by edge id, the same
index `solve_graph` returns its `Vec<Caps>` solution under and `ValidatedGraph::edge`
uses: an edge shows its negotiated caps (`Caps::to_gst_string`), a non-`System`
memory domain (drawn bold, since a GPU / zero-copy link is the interesting one),
its non-default `LinkPolicy`, and fan-out / fan-in pad indices.

`g2g-launch --dot` is the user-facing entry: it parses a pipeline against the
registry, dumps the DOT to stdout, and exits without running, labelling each node
by its element's `log_category` (the short type name, e.g. `VideoTestSrc`) via
the new `GraphNodeRef::log_category`. To show the *chosen* caps it first calls
`negotiate_graph` (§4.20a's seam: Phase 1 source-caps probe + Phase 2 solve,
without running the pipeline), which returns the per-edge fixated caps and each
edge's memory domain (the producing node's `output_memory`, M285) the dump
renders on the edges, marking GPU / zero-copy links bold; a negotiation failure
falls back to a topology-only dump. Because negotiation probes sources, a `--dot`
of a live-ingress pipeline does that source's `intercept_caps` (typically a
connect) just as a run would. Memory domain is a per-element declaration
(`AsyncElement::output_memory` / `SourceLoop::output_memory`, default `System`,
overridden by GPU producers like `NvDec`), the runtime peer of the auto-plug
`ElementDesc::output_memory` (§4.13.9); it is not part of `Caps`.

### 4.20a Developer Tooling: Caps-Negotiation Explainer

Caps negotiation is the hardest code in the system (§4.13, with accumulating
workarounds), and a `CapsMismatch` historically gave no hint *why*. The
explainer makes the solver narrate itself. `solve_graph` emits under a reserved
`caps` log category (not an element type, so it filters independently): a setup
dump of each node's constraint, then per edge the surviving `CapsSet` and its
fixated `Caps`. On failure it narrates at ERROR, naming the two conflicting nodes
and dumping the set on every edge incident to them, so the log answers "these two
can't agree, and here is what each wanted"; an edge that survives narrowing but
can't reduce to one `Caps` logs `cannot fixate`.

Node labels come from the caller via `solve_graph_labeled`: the runner passes
each element's `log_category` (so the narration reads `h264parse -> nvdec`),
while `solve_graph` defaults to `n{id}:{kind}`. The narration is gated by the
logging framework (§4.15): all formatting is skipped unless the `caps` category
is enabled, which costs one atomic load when off, so it is free in production.
It is turned on with `G2G_CAPS_TRACE=1` (a boolean shortcut, or a level name /
number to tune verbosity) or the general `G2G_DEBUG=caps:debug`; both install the
stderr sink through `log::init_from_env`, which the launch / inspect binaries
already call at startup.

### 4.20b Developer Tooling: the `xtask` crate

`cargo xtask <command>` (a `.cargo/config.toml` alias onto the `xtask` workspace
member) is the home for the build / test invocations that were otherwise
shell-history knowledge. It is dependency-free, orchestrating only `cargo` and
toolchain tools. `ci` runs locally what the GitHub workflow runs (workspace
check / test / clippy, the Linux feature build, the embassy no-alloc tests, the
wasm core check), `--locked` like CI, so a red CI is reproducible offline.
`test --here` probes the host (`nvidia-smi`, `pkg-config` for the syslib-backed
features, `/dev/video*` and `/dev/dri` device nodes) and runs exactly the
feature-gated tests this machine supports, automating the "validate on this host"
dance; `--dry-run` prints the detected plan only. `size` builds the
`examples/g2g-size` Cortex-M harness and reports the gc-sectioned `.text`
footprint (it locates `rust-lld` in the toolchain sysroot for the final link).
`wasm` builds the wasm32 targets. The cross-compiling commands (`size`, `wasm`)
prepend `~/.cargo/bin` to `PATH` so cargo selects the rustup toolchain over a
distro `rustc` that lacks the target std, and `wasm` passes
`--cfg=web_sys_unstable_apis` for the `web-codecs` build.

`ffi-probe <header> <struct> [--field f]...` automates the hand-rolled-FFI
ritual (§4.11 / the `cuda.rs` / `nvenc.rs` convention): it generates a C program
that includes the header and prints `sizeof` of the struct plus `offsetof` of
each field, compiles and runs it, and emits the `const _: () = assert!(size_of::
<Struct>() == N)` to paste alongside the `#[repr(C)]` transcription. Layout is
locked down before it is trusted, and an SDK version bump that resizes a struct
fails the build rather than the GPU. `bench` runs the criterion benchmarks.

The criterion benchmarks live in a standalone `g2g-bench` crate, excluded from
the workspace (like `examples/g2g-size`) because criterion pulls plotters / rayon
that a `--all-targets` CI job would otherwise build on every push, and Cargo's
`required-features` does not gate a dev-dependency under `--all-targets`. They
guard the latency moat's hot paths: the caps algebra + linear / DAG solvers
(`benches/caps.rs`), the per-pixel software frame conversion
(`benches/convert.rs`), and the runner loop's bounded per-edge channel
(`benches/runner.rs`, the transport every frame crosses; the full `run_graph`
paces to PTS so it is unsuitable for a microbench). `cargo xtask bench` drives
them by manifest path, passing criterion args through (e.g. `--save-baseline`).

A dedicated `bench` workflow (separate from the main CI, so criterion never
slows the check / test / clippy jobs) runs on PRs that touch the benched crates:
it benches the PR head and its base and fails if any benchmark's mean regressed
more than 50% (a loose threshold tuned to shared-runner noise, catching a lost
fast path rather than drift).

`RunStats::report()` formats the end-of-run telemetry the runner already gathers,
frame counts + drop rate, the aggregated *declared* latency window (the
per-element `latency()` fold), the elected clock, and the head allocation, which
`g2g-launch` prints at end alongside the measured wall-clock throughput. Measured
per-element / per-link p50 / p99 + channel fill-level needs frame-level
instrumentation in the runner arms (the `LatencyHistogram` collector in
`metrics.rs` wired in) and is a follow-up.

---

## 5. First-Class Machine Learning Integration
To prevent GPU-to-CPU synchronization stalls, tensor execution happens directly inside the VRAM domain. ML elements are `AsyncElement` implementations like any other — they negotiate `Caps::RawVideo` on the input pad and `Caps::Tensor` on the output pad.

### 5.1 Inline Tensor Pre-processing via WebGPU (wgpu)
The ML element sits in the same memory domain context as the hardware decoder. When a `MemoryDomain::DmaBuf` arrives at the ML element:

1. The memory handle is bound directly as a texture inside a `wgpu` compute pipeline.
2. An inline compute shader converts color spaces (e.g. NV12 → planar RGB) and performs normalization scales directly in graphics memory.
3. The resulting tensor handle is emitted as a `Frame { domain: VulkanTexture(...), caps: Caps::Tensor { .. }, .. }`, submitted straight to the inference backend.

`WgpuPreprocess` (`g2g-ml/src/wgpupreprocess.rs`, `wgpu` feature) is the compute-shader half: an NV12 frame is converted and normalized in a wgpu compute shader to a `Caps::Tensor { F32, [1,3,H,W], Nchw }`, the same contract `OrtInference` builds on the CPU. The default system-memory variant uploads NV12 to a storage buffer and reads the f32 tensor back to `MemoryDomain::System`. **GPU-output mode (M215, `with_gpu_output`)** instead leaves the tensor in a `wgpu::Buffer` and emits `MemoryDomain::WgpuBuffer` (an on-device GPU->GPU copy into a fresh per-frame buffer, no map / read-back in the element), so a downstream GPU consumer reads it on-device; a CPU consumer pays the deferred read-back via the buffer owner. This removes the output-side GPU->CPU copy; `WgpuInference` (§5.2, M216) is the consumer that binds the resulting buffer on-device, so `preprocess -> infer` keeps the tensor on the GPU. **Surface-import input (M217)** closes the other end: when the NV12 frame arrives already GPU-resident as a `MemoryDomain::WgpuTexture` (a `WgpuNv12Texture` keep-alive wrapping an R8Uint texture of `width x height*3/2` in standard NV12 byte layout), the element adopts that texture's device and samples it with `textureLoad` straight into the compute pass, with no CPU upload, bit-identical to the storage-buffer path. With both ends GPU-resident, `surface -> WgpuPreprocess -> WgpuInference` runs with the pixels never touching the CPU. **CUDA<->wgpu interop (M220, `CudaToWgpu`, `g2g-plugins/src/cudawgpu.rs`)** joins the NVDEC decode side to this surface-import path: there is no portable "share this CUDA pointer with wgpu" call, so the bridge allocates an exportable Vulkan image (`VK_KHR_external_memory_fd`, wrapped as a `wgpu::Texture` via wgpu-hal), CUDA imports the same memory by FD (`cuImportExternalMemory`) and copies the NVDEC NV12 planes into it device->device, and the wgpu device travels on the frame's keep-alive so `WgpuPreprocess` adopts it (the M217 device-identity pattern). The whole `NVDEC -> CudaToWgpu -> WgpuPreprocess -> WgpuInference` chain is validated on an RTX 3060 (M251), matching a CPU reference with no PCIe download. Shared images are recycled from a reuse pool (M254): the Vulkan image, its CUDA import, and the `wgpu::Texture` are allocated once and returned to a free list when the downstream frame is released (a drop guard on the emitted keep-alive), so per frame only the two device->device plane copies and a sync run; a recycled entry is drained (`Device::poll`) before reuse since a wgpu submission may still sample it. The pool cut the bridge step ~2.6x at 1080p (p50 0.38 ms pooled vs 0.98 ms per-frame-allocated). **The reverse direction (M271, `WgpuToCuda`)** closes the *encode* side: a renderer writes a packed-RGBA `wgpu::Texture` on FD-exportable Vulkan memory (`export_rgba_image` / `wrap_rgba_as_texture`, the `R8G8B8A8` mirror), CUDA imports it as a 4-channel array, and `to_cuda_frame` copies it device->device into a linear `CUdeviceptr` emitted as a `MemoryDomain::Cuda` `Rgba8` frame that `NvEnc` registers as `ABGR` (§4.11.3). So a GPU render reaches the H.264 encoder with no device->host read-back, validated on an RTX 3060 (`wgpu_to_cuda` test). This is the zero-copy egress for server-side rendering / cloud-gaming (the moat version of the M267 Bevy demo, which still reads back); the bridge retains its own CUDA primary context (the GPU the interop device selects) and owns the exportable render-target texture.

### 5.2 Unified Pure-Rust Inference Backends
`g2g` avoids bundling heavy, unsafe proprietary C++ engines. The `g2g-ml` crate provides wrapper elements targeting two execution paradigms:

- **`g2g-ml::burn`** (Embedded / Wasm / RTOS): leverages the pure-Rust Burn framework with a `wgpu` backend, compiling ONNX workflows into type-safe, compile-time Rust graphics shaders. `BurnInference` (`g2g-ml/src/burninfer.rs`, `burn` feature) is the wgpu-backend inference element over the `RawVideo` → `Tensor` contract, driving an `input · W + b` linear layer on any Vulkan / Metal / DX12 / WebGPU adapter.
- **`g2g-ml::ort`** (High-Performance Enterprise Server): wraps ONNX Runtime bindings to pass underlying memory domains to hardware-specific execution paths (CUDA / TensorRT / DirectML / Apple CoreML) natively.

`WgpuInference` (`g2g-ml/src/wgpuinfer.rs`, `wgpu` feature, M216) is the GPU-resident counterpart of `BurnInference`: a raw wgpu compute pass that binds the GPU-resident tensor `WgpuPreprocess::with_gpu_output` (§5.1) produced **directly**, rather than taking `RawVideo` / `System` and uploading. It runs one of a small op zoo on that tensor, selected at construction (each its own WGSL shader behind the shared device-adopt / dispatch / read-back machinery): the original `input · W + b` linear matmul (`linear`); a same-padding stride-1 2D convolution (`conv2d`, M261) over the `[1, Cin, H, W]` NCHW tensor with `[Cout, Cin, KH, KW]` weights, leaving a `[1, Cout, H, W]` feature map; the elementwise activations `relu` / `sigmoid`; and `maxpool2d` / `avgpool2d` spatial pooling (M265). The weighted ops (linear, conv2d) bind a 5-entry group (meta, input, weights, bias, out); the weightless ops (activation, pooling) bind a 3-entry group (meta, input, out), the bind-group layout following the active shader. The conv is the keystone that lets the chain run an actual CNN layer, not just a final classifier; the activation is the nonlinearity that keeps stacked convs from collapsing to one linear map, and the pool the spatial downsampler. Chained GPU-resident (`conv2d -> relu -> maxpool`, each in `with_gpu_output` mode so the data never leaves the device between layers), they are a real small-CNN body, validated on the RTX 3060 against a CPU reference folding the same ops (`conv2d_reference` / `relu_reference` / `maxpool2d_reference`) over the exact tensor the GPU preprocess produced. **Trained weights are imported at runtime (M262)** from a `safetensors` file via a dependency-free reader (`g2g-ml::safetensors`, a focused parser for the format's `u64` length + JSON-subset header + raw tensor bytes, no `serde` / no `safetensors` crate): `conv2d_from_safetensors` reads the `[Cout, Cin, KH, KW]` weight and `[Cout]` bias by name and infers the kernel dims, so picking a different trained checkpoint is "parse a different file" while the layer topology stays this compiled element. This is the weights half; the architecture stays Rust (truly dynamic *graphs* at runtime are the `ort` backend's job, and `burn-import` build-time codegen is the Burn-side topology path). With conv / activation / pooling and GPU-resident chaining in place (M265), a full trained vision model on this path now needs only the remaining ops (batch-norm, attention) and a topology that imports the whole layer stack from one weight file, not just a single conv. It owns no device: because a `wgpu::Buffer` is bindable only on the device that created it, the element adopts the producer's device / queue (carried by the incoming `WgpuBufferOwner`) on the first frame and submits its compute on the producer's queue, which orders it after the producer's work with no fence or read-back. The logits are read back to `MemoryDomain::System` by default or left GPU-resident (`with_gpu_output`) for a downstream GPU consumer. A burn / ort consumer cannot do this zero-copy: their tensor handles are opaque (no foreign-buffer adopt) and run on their own device, so they would force the GPU->CPU->GPU round-trip M215/M216 exist to delete.

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

- **Bring-your-own-device (M263).** The same `GpuContext` sharing extends one
  step further out, to an embedding application that *already owns* a
  `wgpu::Device` (a game engine, a Bevy / Tauri app, an editor's renderer):
  `GpuContext::from_wgpu(instance, adapter, device, queue)` wraps the embedder's
  device instead of opening one, so every GPU element produces textures *on that
  device*. A decoded frame's `MemoryDomain::WgpuTexture` is then a first-class
  object in the embedder's own render graph, recovered with `gpu::texture_of` and
  bindable directly (sample it onto a 3D surface, composite it in the UI) with no
  second device, no surface hand-off, and no copy, the opposite of `for_surface`
  (where g2g opens the device). This is the integration path for the
  lightweight-app / engine use case where the application drives rendering and
  g2g is just the pipeline that hands it textures: validated on the RTX 3060 (a
  texture produced through a `from_wgpu` context reads back correctly on the
  embedder's own device handles). The frame still flows to the app through any
  sink, including the `appsink` pull channel, which carries a GPU-domain `Frame`
  unchanged.

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
  channel. `SinglePacketChannel` (`NoopRawMutex`) is the single-executor
  default; `SharedPacketChannel` (`CriticalSectionRawMutex`, hence `Sync`) is
  the variant that can live in a `static`, so spawned tasks reach it by
  `&'static` (an executor's tasks take `'static` arguments).
- Two executor models, both over the same runner / element futures:
  `embassy-futures::block_on` drives a whole pipeline as one joined task (the
  bare-metal `fn main` entry, used by the M43/M45/M260 host tests); a real
  `embassy-executor` runs each element as an independently *spawned* task wired
  by static stack channels, the scheduler interleaving them. The latter is
  host-verified via the std platform's `Executor::run_until` (polls then
  returns on a completion flag, instead of the diverging `run()` an embedded
  app's `fn main() -> !` calls); a three-task source -> transform -> sink
  pipeline runs there with no HAL time driver.

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
