# Changelog

Pre-release. Work is tracked by milestone (Mn) following the roadmap in `DESIGN.md` §4.10.
Nothing is published yet; all versions are `0.1.0`.

## Unreleased

### M12: Live-source surface (latency + allocation + clock election)
- Latency query: `LatencyReport { live, min_ns, max_ns }` + `query` module,
  GStreamer-style latency triple with `combine`/`aggregate` (min and max sum
  along the path, unbounded `max_ns = None` is infectious, liveness sticky) and
  `is_unsatisfiable`. `ZERO` contributes `max_ns = Some(0)` (non-buffering),
  distinct from `None` (unbounded buffering). `AsyncElement::latency()` /
  `SourceLoop::latency()` default methods (return `ZERO`); live sources and
  buffering transforms override.
- Allocation query: `AllocationParams { size_bytes, min_buffers, align, domain }`
  + `MemoryDomainKind` (and `MemoryDomain::kind()`). `AsyncElement::propose_allocation`
  / `configure_allocation` and `SourceLoop::configure_allocation` default
  methods let a consumer propose a buffer pool that its producer allocates into
  (zero-copy handoff). `AllocationParams::merge` folds an upstream element's own
  requirement into a downstream proposal (most-demanding size/count/alignment
  wins; consumer dictates domain).
- Clock distribution: `ClockPriority` (`SystemFallback` < `Provider` <
  `LiveSource`), `ClockCandidate` (priority + shared `Arc<dyn PipelineClock>`),
  and `elect_clock` (highest priority wins, ties resolve to the most upstream).
  `AsyncElement::provide_clock` / `SourceLoop::provide_clock` default methods
  let a live source or sink offer its clock; the runner adopts the elected clock
  over the supplied fallback.
- Linear runners (`run_simple_pipeline`, `run_source_transform_sink`) fold the
  configured chain into `RunStats::latency`, resolve the allocation query into
  `RunStats::allocation`, and elect the pipeline clock into
  `RunStats::{clock_priority, base_time_ns}` after negotiation. Fan-in /
  fan-out runners report neutral values (topology aggregation deferred).
- M12 complete: with M8–M12 done, `g2g` reaches dynamic-pipeline feature parity
  with GStreamer (per DESIGN.md §4.10) while keeping the static typed layer.

### M13: Windows hardware decode (Media Foundation)
- `MfDecode` element (`mf-decode` feature, Windows-only): wraps the Media
  Foundation H.264 Decoder MFT (`CLSID_MSH264DecoderMFT`, an `IMFTransform`).
  Consumes Annex-B H.264 `DataFrame`s and emits decoded NV12 frames as
  `MemoryDomain::System`, with a `CapsChanged(Nv12)` before the first frame and
  on each decoder stream change. Implements the canonical feed/drain MFT loop
  (`ProcessInput`/`ProcessOutput`, `NEED_MORE_INPUT`/`STREAM_CHANGE`,
  `COMMAND_DRAIN` on EOS, `COMMAND_FLUSH` on seek).
- New optional, target-gated dependency: `windows` 0.62 under
  `[target.'cfg(windows)'.dependencies]`; the `mf-decode` feature implies `std`.
- `HardwareError::MediaFoundation(i32)` carries the failing `HRESULT`.
- Constraint: COM is initialised MTA; the element is thread-affine and intended
  for a single-thread executor (asserted `Send` under a documented contract).
- Deferred: D3D11 zero-copy output (needs a new `MemoryDomain` variant), DXVA
  hardware acceleration, and strided (`MF_MT_DEFAULT_STRIDE`) NV12 copy.

### M11: Application control surface
- `Bus` + cloneable `BusHandle` + `BusMessage` (Eos/Error/Warning/Custom): mp-sc message channel so elements notify the app asynchronously without back-references. Non-blocking `try_post`; `try_recv`/`recv` on the app side.
- `LinkInterceptor` probes: a `Pass`/`Drop` interceptor installed on a link's `SenderSink` via a runtime `ProbeSlot` (GStreamer pad-probe equivalent). Empty by default, so existing links are unaffected.
- `PipelinePacket::Flush`: non-terminal seek-flush packet; elements reset position (sinks drop `last_sequence`) and forward/broadcast it, the stream resumes afterwards.
- Deferred: blocking probe action.

### M10: True muxer fan-in
- `MultiInputElement` trait + `run_muxer_sink`: combine all N inputs into one output (vs the M9 Merger selector), with per-input caps negotiation and EOS aggregation (one `Eos` after every input ends).
- `InterleaveMux` element; tagged-merge-channel runner so one `&mut` muxer task processes all inputs without a select primitive.

### M9: Dynamic fan-out / fan-in
- `Router` (1->N): routes each frame to an atomic-selected output port, broadcasts `CapsChanged` to all ports.
- `Gate` (1->1): forwards or drops data by an atomic flag; a plain `AsyncElement`, so it needs no new runner.
- `Merger` (N->1): selects one active input, drains the rest, emits one merged `Eos` after every input ends.
- Multi-output / fan-in plumbing: `MultiOutputSink`, `MultiOutputElement`, `DynSourceLoop`, `join_all`, and the `run_source_fanout` / `run_fanin_sink` runners. Each primitive has a cloneable control handle (`RouterHandle` / `GateHandle` / `MergerHandle`).
- Deferred: `BranchSlot` + `SwapPolicy` variants (need a task spawner).

### M8: Caps negotiation + dynamic element swap
- Caps algebra: `Dim`/`Rate`/`Caps` `intersect`, `is_fixed`, `fixate` (Phase 1 narrowing, Phase 2 fixation per §4.2).
- Runners fixate negotiated caps before configuring elements; bounded `ReFixate` retry; mid-stream `CapsChanged` cascade and upstream `Reconfigure` sideband.
- `ElementSlot` + `SwapHandle`: lock-free atomic hot-swap of one element mid-stream; `DynAsyncElement` blanket impl so any element boxes into a slot.

## M0-M5 (prior)
- M0/M1: Cargo workspace scaffold and minimal `no_std + alloc` pipeline runtime (bounded channel, `Join2`, linear runner).
- M2: async `OutputSink`, identity transform, 3-element runner.
- M3: `AsyncClock` + `WallClock` + `SyncSink` for PTS-paced presentation.
- M4: Arc-recycled `BufferPool` with async `acquire`; DMABUF close-on-drop fix.
- M5: `RtspSrc` element wrapping `retina`.
