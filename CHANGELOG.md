# Changelog

Pre-release. Work is tracked by milestone (Mn) following the roadmap in `DESIGN.md` §4.10.
Nothing is published yet; all versions are `0.1.0`.

## Unreleased

### M16: Caps negotiation redesign (CSP framing)
- Design doc `DESIGN-M16-caps-nego.md` (§§1-10) recasts negotiation as a
  constraint-satisfaction problem with a solver, and documents the
  6-step migration plan (§8).
- Step 1: `CapsSet` algebra in `g2g-core::caps`. Preference-ordered
  alternatives over `Caps`; `one`, `from_alternatives`, `intersect`
  (preserves self's outer order, dedupes equal results), `fixate`
  (picks the highest-preference fixable alternative). `Caps` remains
  the *fixed* runtime description; `CapsSet` is the negotiation-time
  vocabulary. Re-exported from the crate root. Unit-tested for empty
  intersection, preference preservation, dedup, and fixate fallback.
- Adjacent design debt acknowledged (no code change): `VideoFormat`
  conflates compressed codecs (H264, H265, Av1, Vp9) and raw pixel
  layouts (Nv12, I420, Rgba8, Bgra8) in a single enum, shoehorned
  into `Caps::Video { format, ... }`. GStreamer keeps these as
  separate media types (`video/x-h264` vs `video/x-raw, format=...`).
  Documented in `DESIGN-M16-caps-nego.md §11` and memory
  (`architecture_codec_vs_raw_format.md`). M17-sized refactor;
  M16 continues on the current shape.

- Step 5j (reordered): NV12 display sinks (`WaylandSink`, `KmsSink`)
  tolerate mid-stream geometry changes. Previously `configure_pipeline`
  with new dims after the worker / framebuffer pool was up returned
  `CapsMismatch`; now it tears down the existing worker/slots
  (`self.shutdown()` / `self.teardown()`) and falls through to
  fresh setup at the new geometry. Same-dims is still a no-op.
  This unblocks the next step (5k: migrate `FfmpegH264Dec` to
  `DerivedOutput`): mixed chains will land NV12 at startup with
  placeholder dims (RtspSrc workaround #1, fixated to min), and the
  real geometry will arrive via mid-stream `CapsChanged` after SPS
  parse — both transitions now succeed instead of the second
  refusing. No new tests in this commit: the rebuild path runs
  through `worker_main` (Wayland session) / `allocate_slots` (real
  DRM card), neither testable in CI. Visual verification belongs to
  the user's manual e2e run.

- Step 5h: `CapsConstraint::IdentityAny` wildcard transform variant
  for pass-through transforms (probe / metering / tee) whose
  `intercept_caps` is `Ok(upstream.clone())`. Native solver couples
  input and output links to be equal without narrowing either by a
  set; surrounding endpoints determine the actual caps.
  `IdentityTransform` (g2g-plugins) overrides
  `caps_constraint_as_transform` to return `IdentityAny`.
  Existing 3-element pipeline_smoke test
  `source_identity_sink_3_element_pipeline` now exercises the
  all-native arc-consistency path
  (`VideoTestSrc Produces → IdentityTransform IdentityAny →
  FakeSink AcceptsAny`). 3 new solver tests cover the coupling, a
  mixed-chain pass-through, and rejection of `IdentityAny` in
  endpoint positions. 99 g2g-core tests + 11 plugin lib +
  every integration suite green.

- Step 5g: first native transform. `H264Parse` overrides
  `caps_constraint_as_transform` to return
  `Identity(CapsSet::one(Caps::Video { format: H264, dims: Any }))`.
  The native solver couples its input and output links and enforces
  the H.264 format requirement during arc consistency instead of via
  the dynamic `intercept_caps` callback. New tests:
  - `h264parse::caps_constraint_is_identity_h264_any` (unit).
  - `pipeline_smoke::h264parse_identity_negotiates_in_mixed_chain`
    drives a real `run_source_transform_sink` chain
    `LegacySource(H264) → H264Parse Identity → AcceptsAny FakeSink`
    and verifies negotiation + EOS propagation through the mixed
    cascade. The source is an inline EOS-only stub (no real H.264
    bytes needed — `H264Parse::process` for `Eos` is pass-through).
  96 g2g-core tests + 6 pipeline_smoke + every integration suite
  green.
- Step 5f (revised): first native source + `SourceLoop` trait
  integration. Original 5f scope (workaround #1 placeholder dims)
  bumped — properly fixing it needs async `intercept_caps` (SDP
  DESCRIBE), and it's symbiotic with #2 so fixing alone unblocks
  nothing visible.
  - `SourceLoop` gains `caps_constraint(&self) -> Result<CapsConstraint<'_>, G2gError>`
    default method returning `LegacySource(intercept_caps()?)`.
  - `run_simple_pipeline`, `run_source_transform_sink`, and
    `run_source_fanout` call `source.caps_constraint()` instead of
    constructing `LegacySource` inline. `ReFixate` retry uses
    `LegacySource(counter)` fallback (counter-proposals are a legacy
    concept). `run_muxer_sink` stays on `intercept_caps` because
    `DynSourceLoop` doesn't yet expose `caps_constraint` — no
    migrated muxer sources exist, so adding it is deferred until
    needed.
  - `VideoTestSrc` overrides `caps_constraint` to return
    `Produces(CapsSet::one(self.caps()))`. Production chain
    `videotestsrc → FakeSink` (both native) now exercises the
    all-native arc-consistency solver path with backward
    propagation, instead of the mixed cascade. Behavior unchanged.
  - 1 new solver test (`all_native_produces_to_accepts_any_passes_through`).
    96 g2g-core tests + every integration suite green.
- Step 5e: correctness fix — `solve_legacy_cascade` reverts to
  intercept-only (bit-compatible with the pre-M16 cascade). Step 4b
  had incorrectly called the format-boundary's `propose_output_caps`
  during the legacy cascade, producing per-link caps the legacy
  workaround-#2 sinks (`WaylandSink`, `KmsSink`) can't consume —
  e.g. NV12 at placeholder dims at startup, with the deferred-setup
  branch then refusing the mid-stream real-dim `CapsChanged`.
  Restored: every all-legacy chain element receives the single
  fixated `Caps` from the cascade's final intercept, matching
  pre-M16. The CI suite missed the regression because the e2e
  `rtsp → ffmpeg → waylandsink` test is `#[ignore]`d (sandbox
  blocks port 554).

  **Revised 5d claim**: per-link configure benefits mixed chains
  (one or both endpoints native). All-legacy chains keep the
  single-fixated-caps model and workaround #2 stays load-bearing
  for them — until those sinks migrate (which requires workaround
  #1, the placeholder-dims fix, to land first so per-link NV12
  carries real dims at startup).

  Updated `legacy_cascade_with_boundary_transform` test and
  `architecture_caps_nego_debt` memory.
- Step 5d: per-link configure (deletes caps-nego workaround #2).
  Two coupled changes:
  - `solve_legacy_cascade` no longer clobbers upstream link slots
    with the final fixated caps. Each link is fixated independently
    and carries its own intercept-narrowed value, so format-changing
    boundaries (decoder: H264 in / NV12 out) keep their per-link
    identity.
  - `run_source_transform_sink` extracts both `src_caps = links[0]`
    and `sink_caps = links[1]` from the solver and passes each
    element the side it expects: `source.configure_pipeline(src_caps)`,
    `transform.configure_pipeline(src_caps)` (input side — what
    decoders like `FfmpegH264Dec` validate against), and
    `sink.configure_pipeline(sink_caps)`. M12 allocation queries
    use the downstream-facing `sink_caps`.

  **Regression caught and fixed**: in step 5c, migrating `FakeSink`
  to `AcceptsAny` routed the rtsp e2e chain through the mixed
  cascade, which correctly returned `links=[H264, NV12]`. But the
  pre-5d runner fed `links.last()=NV12` to every element, including
  the decoder, which rejects NV12 with `CapsMismatch`. The e2e test
  is `#[ignore]`d so CI missed it. New regression test
  `format_changing_transform_receives_input_side_caps` in
  `pipeline_smoke.rs` covers this shape without needing ffmpeg.

  **Workaround #2 retired**: `WaylandSink` / `KmsSink` now receive
  NV12 directly at startup (when downstream of a decoder), so their
  "Caps::Video { .. } => no-op" deferred-setup branch is no longer
  load-bearing. The branch is left in place for safety; cleanup is
  a follow-up. Updated `legacy_cascade_with_boundary_transform`
  test to assert per-link semantics.
- Step 5c: `CapsConstraint::AcceptsAny` wildcard sink variant for
  debug / probe / passthrough sinks whose `intercept_caps` is
  `Ok(upstream.clone())`. Solver treats it as no-op narrowing on
  the link (upstream's produced caps flow through unchanged) and
  enforces it at the chain's tail; in a native chain a middle
  `AcceptsAny` is silently invisible, in mixed/legacy forward
  cascade an interior `AcceptsAny` returns `EndpointShapeMismatch`.
  `FakeSink` and `syncsink` (g2g-plugins) override
  `caps_constraint_as_sink` to return `AcceptsAny`; chains
  containing them now exercise the mixed-cascade path. `identity`
  (transform-shape pass-through) stays on the legacy bridge —
  needs an `Identity`-with-wildcard variant which is a separate
  gap. 3 new solver tests; all 95 g2g-core tests + integration
  suites green.
- Step 5b: trait integration so individual elements can migrate.
  `AsyncElement` gains two default methods —
  `caps_constraint_as_sink()` and `caps_constraint_as_transform()` —
  each returning the legacy bridge (`LegacySink` / `LegacyTransform`)
  for today's `intercept_caps` + `propose_output_caps`. The runner
  (`run_simple_pipeline` and `run_source_transform_sink`) now calls
  these methods instead of constructing the bridge inline; migrated
  elements override to return a native `CapsConstraint` and chains
  containing them hit the mixed-cascade solver path. Behavior is
  identical for every existing element (all defaults). Bridge helpers
  `legacy_sink_constraint` / `legacy_transform_constraint` relaxed
  to `?Sized` so they can be called from the trait's default methods
  on `&Self`.
- Step 5a: mixed-chain support in the solver. Chains that mix
  legacy and native `CapsConstraint` variants now route to a new
  `solve_mixed_cascade` (single forward pass that handles every
  variant). Legacy variants and `DerivedOutput` require upstream to
  fixate to one concrete `Caps`, which migration chains satisfy
  because the source is typically single-alternative. Backward
  arc-consistency (Identity / Mapping filtering against downstream
  sinks) is not applied in the mixed path; once a chain becomes
  fully native, dispatch routes it back to the arc-consistency
  solver and backward propagation is restored.
  `NegotiationFailure::MixedLegacyAndNative` is no longer returned
  (kept as a variant in case future mixed shapes need it).
  4 new tests cover legacy-source/native-sink, native-source/legacy-sink,
  native-source/legacy-decoder/native-sink, and sink rejection in a
  mixed chain.
- Step 4c: roll the solver-via-legacy-bridge pattern out to the
  remaining linear runner entry points. `run_simple_pipeline`
  (source → sink) and `run_source_fanout` (source → fanout, with the
  fanout as the linear "sink" of the chain — downstream sinks
  broadcast-receive the fixated caps and don't participate in
  narrowing). `run_muxer_sink` solves each source ↔ muxer-input pair
  via the solver, wrapping the muxer's per-input `intercept_caps`
  as a `LegacySink`. `run_fanin_sink` stays direct: each source
  self-fixates with no peer narrowing, so there's no chain to solve.
  The muxer's aggregated-output → sink half stays as direct
  `fixate()` because today's runner intentionally does *not* call
  `sink.intercept_caps` for that hop (the muxer output is the
  canonical merged caps).
- Step 4b: `run_source_transform_sink` startup negotiation routes
  through `solve_linear` via the legacy bridge. The pre-M16 inline
  cascade (`source.intercept_caps` → `transform.intercept_caps` →
  `sink.intercept_caps` → `fixate`) is replaced by building
  `LegacySource` / `LegacyTransform` / `LegacySink` constraints and
  calling the solver; the cascade output is bit-compatible with what
  the inline path produced. `ReFixate` retry stays in the runner (the
  solver doesn't model counter-proposals); on each retry the
  `LegacySource` seed becomes the counter and the solver re-runs.
  Mid-stream `CapsChanged` paths are untouched. All existing tests
  pass (89 g2g-core, 14 g2g-plugins rtsp lib, all integration
  suites).
- Step 4a: legacy bridge into the solver. `CapsConstraint` gains a
  `'a` lifetime parameter and three transitional variants —
  `LegacySource(Caps)`, `LegacyTransform { intercept, propose_output }`,
  `LegacySink(intercept)` — that capture today's `AsyncElement`
  callbacks. `legacy_transform_constraint(&T)` / `legacy_sink_constraint(&T)`
  helpers wrap a borrowed element. The solver dispatches: all-native
  chains take arc consistency, all-legacy chains take
  `solve_legacy_cascade` (forward cascade that mirrors today's runner,
  then fixates the final caps and propagates to upstream link slots
  the same way `configure_pipeline` is called today). Mixed chains
  return `NegotiationFailure::MixedLegacyAndNative` until step 5
  migrates individual elements. 6 new tests cover the cascade,
  pass-through and boundary transforms, intercept failure, mixed
  chain rejection, and the AsyncElement → LegacyTransform bridge.
- Step 3: linear-pipeline caps solver in
  `g2g-core::runtime::solver` (feature `runtime`).
  `solve_linear(&[&CapsConstraint]) -> Result<Vec<Caps>, NegotiationFailure>`
  walks a source → transform* → sink chain with arc consistency:
  seed endpoint links from `Produces` / `Accepts`, forward+backward
  sweep until fixed point, fixate each link to one concrete `Caps`.
  Handles all four interior constraint shapes — `Identity` couples
  input and output, `Mapping` filters pre-enumerated (in, out) pairs
  to the surviving set, `DerivedOutput` fires once its input link
  fixates. `NegotiationFailure` reports `EmptyLink` /
  `EndpointShapeMismatch` / `Unfixable` / `Degenerate`; `Cyclic` is
  reserved for the non-linear solver. `CapsSet::union` added to
  support the `Mapping` path. 10 unit tests cover the minimal chain,
  empty/disjoint links, degenerate input, endpoint-shape errors,
  preference tie-break, identity coupling and mismatch, derived
  output, and mapping pair selection.
- Step 2: `FormatElement` trait + `CapsConstraint` enum in new
  `g2g-core::format_element` module. `CapsConstraint` variants:
  `Accepts` (sinks), `Produces` (sources), `Identity` (pass-through
  transforms), `Mapping(Vec<(in, out)>)` (pre-enumerated codecs),
  `DerivedOutput(Fn(&Caps) -> CapsSet)` (decoders reading SPS).
  `configure_link(input, output)` replaces `configure_pipeline`;
  boundary elements see distinct sides, sources/sinks see `None` on
  the unused side. `CapsPreferences` is a placeholder for the
  tie-break algebra (DESIGN-M16 §10). Coexists with `AsyncElement`;
  the legacy-bridge blanket impl lands with the solver in step 3,
  because its shape is dictated by what the solver consumes.

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

### M15: RTSP reconnect + long-running stability soak
- `RtspSrc` now supports reconnect-with-backoff. Off by default for
  backwards compatibility; opt in with `.with_reconnect(max_attempts)`
  and optionally `.with_reconnect_backoff(initial_ms, max_ms)`.
  Exponential backoff caps at `max_ms`. Network/protocol errors trigger
  retry; a graceful server-side end-of-stream (retina's demuxer returning
  `None`, typical for VOD finishing) terminates without retry.
- `run_rtsp` refactored into an outer reconnect orchestrator and inner
  `run_session`. State threaded across sessions:
  - cumulative `sequence` counter so downstream sees monotonic IDs;
  - `pts_base_ns` offset so per-session PTS continues monotonically
    across reconnects (with a deliberate 1 s gap marking the boundary).
- A `PipelinePacket::Flush` is emitted before each reconnect so the
  decoder flushes its codec state and sinks reset `last_sequence`.
- `tests/rtsp_soak.rs` (#[ignore]) is a long-running stability soak
  configurable via `G2G_SOAK_SECONDS` (default 30) and `G2G_SOAK_MIN_FRAMES`
  (default `seconds * 20`). Asserts monotonic `sequence` and `pts_ns`,
  and that the pipeline reaches the frame floor — catches PTS regressions
  and stalls that the 2-second smokes miss. Module docs cover the manual
  reconnect-exercise workflow (Ctrl-C the publisher, watch it resume).
- `tests/rtsp_smoke.rs` gains `rtspsrc_with_reconnect_retries_then_fails`
  which exercises the reconnect orchestrator against an unreachable URL,
  asserting the source retries `max_attempts` times before surfacing an
  error.

### M14: Wayland display sink (Linux, NV12, desktop-dev convenience)
- `WaylandSink` element (`wayland-sink` feature, Linux-only): opens an
  `xdg_toplevel` window on the running compositor and presents NV12 frames
  via `wl_shm` after software conversion to XRGB8888 (BT.601 limited range).
  Designed as the desktop-dev companion to `KmsSink` — same NV12 input
  contract so the upstream pipeline stays identical.
- Threading: a dedicated worker thread owns all Wayland state (Connection,
  EventQueue, SlotPool); the sink struct holds only a calloop channel and
  an `Arc<AtomicU64>` counter, both `Send + Sync`. SCTK's
  `calloop_wayland_source` multiplexes Wayland events and frame arrivals
  in a single event loop. A one-shot Mutex/Condvar handshake gates the
  sink-side `configure_pipeline` on the first compositor `configure`.
- Pulls `smithay-client-toolkit` 0.20 (transitively bringing the
  `wayland-*` family) under `[target.'cfg(target_os = "linux")'.dependencies]`.
- Constraints (v1): NV12 only, fixed input dims, no scaling (compositor
  letterboxes/clips if its configure suggests a different size), no PTS
  pacing, software conversion only (zero-copy via `zwp_linux_dmabuf_v1` is
  deferred).
- `KmsSink` is the production low-latency sink; `WaylandSink` is for
  iterating on the pipeline inside your desktop session without dropping
  to a tty.

### M14: KMS/DRM display sink (Linux, NV12)
- `KmsSink` element (`kms-sink` feature, Linux-only): primary-plane scanout
  of NV12 `DataFrame`s on the first connected connector + CRTC of the
  configured DRM device (defaults to `/dev/dri/card0`). Two-buffer dumb-
  buffer pool; first frame goes through `set_crtc`, subsequent frames page-
  flip and the next submission blocks on the prior flip's `PageFlip` event
  so the buffer being overwritten is off scanout (tearing-free).
- `FfmpegH264Dec::with_output_format(OutputFormat::Nv12)` (M14 prerequisite,
  separate commit) interleaves the U/V planes after decode without swscale;
  same total byte length as I420. I420 remains the default.
- New optional, target-gated deps: `drm` 0.15 + `drm-fourcc` 2 under
  `[target.'cfg(target_os = "linux")'.dependencies]`; `kms-sink` implies
  `std`. No `unsafe` and no GBM dependency — pure dumb-buffer path.
- Constraints (v1): NV12 only, fixed input dims (mid-stream geometry change
  not supported), no letterboxing/scaling (buffer scans out at native dims;
  smaller-than-mode video shows at origin with stale framebuffer around it),
  requires DRM master (tty or DRM lease; a running compositor will block).
- Deferred (v2): overlay-plane path with src/dst rectangles for proper
  letterboxing; async page flips for lower latency; Wayland sink as a
  desktop-dev convenience using the same NV12 input contract.

### M13: End-to-end RTSP → ffmpeg decode (Linux software path)
- `RtspSrc::intercept_caps` now advertises fixate-friendly `Dim::Range` /
  `Rate::Range` instead of `Any`. `Caps::fixate()` rejects `Any` and aborted
  Phase 2 negotiation before any network handshake; the placeholder is
  overwritten by the SDP-derived `CapsChanged` emitted from `run`.
- `RtspSrc` drops every `VideoFrame` until the first `is_random_access_point`
  IDR. retina's `FrameFormat::SIMPLE` only prepends SPS/PPS on keyframes, so
  mid-GOP tune-in (typical for live RTSP servers like MediaMTX) would feed
  parameterless slices to the decoder and stall with "non-existing PPS 0".
- New `rtsp_ffmpeg_e2e` integration test (`rtsp` + `ffmpeg` features, Linux):
  `RtspSrc → FfmpegH264Dec → FakeSink` over `run_source_transform_sink`,
  asserting decoded I420 frames reach the sink with a fixed-dim `CapsChanged`
  preceding the first `DataFrame`. Module docs include a MediaMTX + ffmpeg
  recipe for a deterministic local fixture.

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
