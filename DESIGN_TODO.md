# DESIGN_TODO

Open work tracked against the architecture in [DESIGN.md](DESIGN.md). Items
here are deferrals from the spec, follow-ups blocked on a concrete driver or
upstream fix, and forward-looking tracks that the current architecture
anticipates but hasn't yet built.

## Roadmap to 80% GStreamer parity

The two hardest tracks (negotiation, the DAG runner) and the entire Phase 1
lifecycle spine (state machine + preroll, seek + SEGMENT, auto-plug /
`decodebin` / `playbin`) are complete and meet-or-beat GStreamer on the core
runtime (DESIGN.md §4.13, §4.14). Observability (full bus coverage) and the last
Tier-1 transform (`AudioResample`) are done too (DESIGN.md §4.15, §4.5). The
remaining gap to "80% / credible replacement" is element + subsystem breadth.

### Phase 2 - Breadth + observability (mostly parallelizable).

- **Capture / URI source breadth.** `v4l2src` (DESIGN.md §4.12a), `UdpSrc`
  raw-RTP H.264 ingest (§4.12b), and the `uridecodebin` URI front door (§4.13.9:
  `build_uridecodebin` + scheme handlers for udp/file/rtsp/v4l2) are done.
  Remaining: `HttpSrc` (blocked on a byte-stream caps + a consumer; see below);
  the receive-side RTP jitter buffer is done (M94, reorder + loss/dup, bounded
  latency), leaving RTCP/NACK/RTX/FEC and SDP/SPS-driven `UdpSrc` caps
  discovery; MJPEG-mode UVC + format-flexible `v4l2src`
  negotiation (it fixes YUYV today). `uridecodebin` follow-ups: more schemes as
  sources land, and an `http(s)://` handler once `HttpSrc` exists.
- **`HttpSrc` is unblocked now (byte-stream type + consumers exist).** A
  souphttpsrc-equivalent produces a byte stream, which used to have no `Caps`
  variant and nothing to consume it. M108-M112 fixed that: `Caps::ByteStream`,
  the `tsdemux` / `matroskademux` demuxers, and the `filesrc` `bytestream-format`
  (+ `typefind`) pattern. So `HttpSrc` can be built mirroring `FileSrc` (fetch
  bytes via `reqwest` / `hyper`, declare the container by property or sniff),
  then wired into `uridecodebin` as the `http(s)://` handler. Still pairs with an
  HLS/DASH element but is no longer blocked on one.
- Leaky-link follow-up: wire leaky `LinkPolicy` for a `no_std` live-camera
  runner (the design's stated `DropOldest` use case); the leaky setters are
  `std`-gated today since only `run_graph` configures per-edge policy.

### Phase 3 - Use-case-driven breadth (pick by product target).

Compositor, HLS/DASH, RTMP/SRT, VP8/VP9/AV1/Opus, MKV/TS. Architectural cost is
low (each is "add an element"); sequence against whatever product target
matters. Detail for these lives in the sections below.

## Python-element host (M198+)

Host gst-python-ml elements as first-class g2g elements. gst-python-ml factored
its ML logic away from GStreamer behind `GSTML_BACKEND`-selected seams
(`FrameIO`, `AnalyticsBackend`, the element base classes); the `g2g-python`
crate + `python` feature is the g2g host those seams target, via embedded
CPython (pyo3). Decision taken: in-process pyo3, not out-of-process IPC.

- **M198 (done): skeleton.** Crate + `python` feature + interpreter bootstrap;
  `PyTransform` (`AsyncElement`) negotiating `Caps::RawVideo` + bridging
  properties; the `RawVideoFormat` <-> Python format-string table; the native
  `g2g` module stub + the `g2g_process(buf, w, h, fmt) -> (bytes|None, [blobs])`
  per-frame contract. Negotiation half builds on the no-libpython profile; the
  Python call is inline + a `bytes` copy.
- **Step 2 (done): zero-copy per-frame path.** `FrameBuffer` (`#[pyclass]` with
  `__getbuffer__`) hands Python a writable buffer-protocol view over the frame's
  System slice; Python reads / overwrites in place (`memoryview` /
  `np.frombuffer`), no copy either way, no Rust-side numpy crate. pyo3 bumped to
  0.26 for CPython 3.14. Verified live on the system 3.14 (stdlib fixture writes
  into the frame, mutation observed downstream). Python traceback is surfaced to
  stderr via `PyErr::print` (a richer error needs a `G2gError` string payload).
- **Step 2b (done): GIL offload.** Each `PyWorker` owns a GIL-holding OS thread;
  `PyWorker::run` hands it the owned `Frame` over a std channel and awaits the
  reply over g2g-core's Waker-based `runtime::channel`, freeing the single
  executor thread to poll other arms while Python runs. `Frame: Send` means no
  cross-thread pointer; the buffer-protocol pointer stays on the worker thread.
  GIL strategy decided: the one-thread-per-element shape IS the free-threaded
  (PEP 703, `--disable-gil`) unit, so workers parallelize unchanged on a
  free-threaded interpreter (`Python::attach` is the no-GIL API, not "acquire the
  GIL"). Per-interpreter-GIL sub-interpreters rejected: numpy / torch / cv2 are
  not reliably sub-interpreter-safe. Remaining: verify on a free-threaded build
  (none installed yet) + a `link_capacity` note for the GIL-serialized case.
- **Step 3 (done): analytics metadata.** `analytics` feature pulls
  `g2g-core/metadata`. `g2g_process` receives a `meta` arg, a native
  `g2g.MetaSink` (`#[pyclass]`, the `AnalyticsBackend` mirror); Python calls
  `add_object` / `add_classification`, the host materializes the staged results
  into `Frame::meta` as an `AnalyticsMeta` (reusing the existing
  `ObjectDetection` / relation graph the g2g-ml `detect` element built, NOT a new
  M88 build). Labels are interned `u32` ids (Python does the string->id step).
  Verified live (`m198_analytics`).
- **Step 4a (done): launch-registry factory.** `g2g_python::register(&mut
  Registry)` adds a `pyelement` `LaunchFactory`; `PyTransform: PadTemplates`
  (RGBA / any geometry). `... ! pyelement module=... class=... draw-label=... !
  ...` parses, with properties applied via M104; `configure_pipeline` errors on
  empty module/class. Verified `m198_registry`.
- **Step 4b (done): `PyAggregator`.** `MultiInputElement` on the M199
  `InputAggregator`: collects one frame per contributing input, one Python
  `g2g_process_batch(buffers, w, h, fmt, meta)` call, emits the anchor frame with
  aggregate `AnalyticsMeta`. The worker was generalized to a frame batch
  (transform = batch of 1). N-in/1-out emits one frame, so per-stream results
  need a demux; v1 assumes shared input geometry. Verified `m198_aggregator`.
- **Step 4c (done): opaque blob side-data.** `g2g-core::BlobMeta` (a `FrameMeta`)
  carries tagged opaque side-data; `g2g.MetaSink::add_blob` (the
  `FrameIO.append_blob` mirror) routes into it on the anchor frame. Verified
  `m198_analytics`. A header *registry* (decoding known headers into typed
  structures) is a later refinement; opaque carry is the foundation.
- **Step 4d (done): `PySource`.** A `SourceLoop` handing Python a blank writable
  buffer per tick via `g2g_produce(buf, w, h, fmt, meta) -> bool` (False = EOS);
  property-fixed output caps. New `Produce` `JobKind` on the worker (now
  transform / batch / produce). Registered as launch source `pysrc`. Verified
  `m198_source`.
- **Step 4e (done): `MultiInputElement` property surface.** `properties()` /
  `set_property()` / `get_property()` (defaults: none) on `MultiInputElement` +
  `DynMultiInputElement`; the launch parser applies `key=value` to muxers
  (`apply_muxer_props`). `pyaggregator` registered and launch-parsable. A general
  g2g-core gain (any muxer can now take launch properties). Verified
  `m198_registry`.
- **Step 4f: GPU zero-copy (designed, not implemented).** Hand a GPU-resident
  frame to Python without the PCIe round-trip, so torch / cupy consume device
  memory directly. Design (deferred to a GPU host with installable cupy/torch):
  - `MemoryDomain::Cuda` is concretely an **NV12 two-plane decoder output**
    (`OwnedCudaBuffer { luma_ptr, chroma_ptr, luma_pitch, chroma_pitch, width,
    height, context }`), not one contiguous tensor. So expose **two**
    `__cuda_array_interface__` (CAI v3) objects, not one:
    - luma: `shape (height, width)`, `typestr "|u1"`, `strides (luma_pitch, 1)`,
      `data (luma_ptr, read_only=false)`;
    - chroma (interleaved UV @ half res): `shape (height/2, width/2, 2)`,
      `strides (chroma_pitch, 2, 1)`.
  - A new contract method (the byte-buffer path does not apply): `g2g_process_cuda
    (luma, chroma, width, height, meta)` where the args are `g2g.CudaPlane`
    pyclasses whose only job is to carry the CAI dict. Building the dict is *safe*
    Rust (integers in a dict); the consumer (`cupy.asarray` / `torch.as_tensor`)
    does the unsafe part.
  - Caveat: CAI carries no CUDA context; the consumer must `cuCtxPushCurrent`
    `OwnedCudaBuffer.context` (or run in a matching context) before touching the
    memory. Document loudly. A `stream` field (CAI v3) should reflect the
    decoder's stream once g2g threads one through.
  - DLPack (`__dlpack__` / `from_dlpack`) is the cross-framework alternative;
    it carries device type/id but is more capsule machinery. Start with CAI
    (simpler, cupy/torch/numba all read it), add DLPack if a consumer needs it.
  - Non-zero-copy fallback: a `download` path (cuMemcpyDtoH into a System frame)
    for elements that only do CPU work; this is the [[project_nvdec_system_memory_floor]]
    tax made explicit and opt-in.
  - **Verify on the RTX 3060 host**: install cupy/torch (likely via `python3.13`,
    pointing `PYO3_PYTHON` at it — 3.14 GPU wheels may not exist yet), assert a
    `cupy` array aliases the decoder's device pointer (no copy) and a kernel sees
    the NV12 planes correctly. Until that runtime check passes, the layout
    (strides / chroma shape / typestr) is unverified and must not be presented as
    working.
- **Step 4 remainder: smaller items.** A blob header registry (decode known
  `BlobMeta` headers into typed structures).
- **Python side (gst-python-ml, separate repo):** a `backend/g2g/` package
  mirroring `backend/gst/` -- thin once the native `g2g` module exists.

## Aggregation helper adoption (M199+)

`g2g-core::InputAggregator<T>` (M199) is the shared per-input collector for
N-in-1-out `MultiInputElement`s. `PyAggregator` is its first consumer. The four
pre-existing hand-rolled collectors should adopt it incrementally to delete the
duplicated `Vec<VecDeque<_>>` + `ended` + contributor bookkeeping: the enterprise
`batcher` (closest fit, the API was modeled on its `contributors`/`drain` rule),
then `mux`, `audiomixer`, and `compositor` (compositor wants a latest-wins
snapshot policy rather than consume-one-per-round, so it needs a second
`SyncPolicy` variant first, `T: Clone` or a borrowing `peek_round`). Refactors
are behaviour-preserving and each guarded by that element's existing tests.

## GStreamer porting story (M200+)

[PORTING.md](PORTING.md) is the guide (pipelines, element mapping, app code,
custom elements, third-party registration, gaps). Infra landed in M200:
- `Caps::to_gst_string()` (g2g-core) — the inverse of the capsfilter parser;
  round-trips. Powers logs / diagnostics and a future `g2g-launch -v` caps dump.
- `g2g_plugins::gst_compat` — a gst->g2g element-name map (`gst_equivalent`) and
  a launch linter (`lint_launch`) that turns a parse failure into a porting
  suggestion. Wired into `g2g-launch` (hint on error) and `g2g-inspect --gst`.
- `examples/third_party_element.rs` — the register-a-third-party-element path,
  end to end.

### Dynamic plugin loading via cargo (DONE, M201 — see DESIGN.md §4.16)

The cargo-native dynamic-plugin path landed in M201: `g2g-core::ABI_VERSION` +
build script, the `g2g-plugin` SDK (`declare_plugin!`), the `plugin-loader`
loader in `g2g-plugins`, `g2g-launch` / `g2g-inspect` `--plugin` /
`$G2G_PLUGIN_PATH` wiring, and an out-of-tree example + dlopen test. Architecture
recorded in DESIGN.md §4.16. Remaining future work (not started):

- **`abi_stable`/`stabby` facade** over the element traits for cross-toolchain
  binary plugins (the v1 path is version+toolchain-locked).
- Whether the distro ships `g2g-core` in a local cargo registry for offline
  plugin builds.
- Plugin signing / capability gating.
- A C-FFI loader entry so non-cargo build systems can produce plugins.

## gst-launch DSL harmonization (M182+)

The `parse_launch` DSL is a GStreamer-compatibility surface; its human-facing
vocabulary is being aligned to gst so a `gst-launch` line ports with minimal
edits (the typed core is unaffected, only the string<->enum boundary).

- **M182 (done).** Format names uppercase (`NV12`/`RGBA`/`YUY2`/`S16LE`,
  case-insensitive parse + old lowercase aliases); `videoflip method` uses gst
  nicknames (`clockwise`/`counterclockwise`/`horizontal-flip`/`vertical-flip`/
  `none`) with `FlipMethod::Identity` default; `videotestsrc pattern`
  `bar`/`checkers-8`. Old g2g spellings remain aliases.

- **Remaining gst-porting gaps (uncovered by M182, both real, both > naming):**
  - **Format-less / partial geometry caps (DONE, M184).** `capsfilter` parses
    `video/x-raw,width=160,height=120` (no `format`) by expanding to a `CapsSet`
    over all raw formats at that geometry (`parse_caps_set`); the solver
    intersects down to the upstream format. `audio/x-raw` likewise.
  - **Caps-driven transform operation (videoscale M185, videoconvert M186).**
    Added `AsyncElement::configure_output(output_caps)` (default no-op,
    dyn-mirrored), delivered by the graph_runner + coordinator from the
    already-fixated output link. `videoscale` (geometry) and `videoconvert`
    (format) with unset props take their target from a downstream capsfilter
    (`videoscale ! video/x-raw,width=160`, `videoconvert ! video/x-raw,format=NV12`);
    props still override; a bare instance is a passthrough. REMAINING:
    - **`audioresample` (rate) DONE, M187.** `Caps::Audio` gained an
      `ANY_SAMPLE_RATE` (0) wildcard (a sentinel, not a type change, to avoid
      rippling the bare-`u32` `sample_rate` across ~50 audio files). `intersect`
      wildcards it and `fixate` rejects it for raw PCM only; compressed audio
      keeps `0` as its "unknown until parsed" nominal value. `audioresample`
      (auto by default) takes its rate from a downstream capsfilter
      (`audioresample ! audio/x-raw,rate=16000`); property still overrides.
    - **Stacked auto transforms (M188, partial).** Two auto transforms before
      one far capsfilter now back-propagate the pin: the solver evaluates each
      `DerivedOutput`'s forward image per input alternative and drops the inputs
      whose image can't reach the constrained output
      (`backward_filter_derived`), while the forward pass narrows the output by
      the union of `f` over the surviving inputs (`forward_derived_union`).
      `videoconvert ! videoscale ! video/x-raw,format=NV12,width=160,height=120`
      resolves; a bare `videoconvert ! videoscale` stays passthrough.
      KNOWN LIMIT: the reverse order, `videoscale ! videoconvert ! caps`, where a
      *geometry* pin sits behind a geometry-passthrough transform, does not
      resolve and fails loud at runtime (CapsMismatch, no silent mis-fixate).
      Dropping whole alternatives can't narrow a `Range` *within* an alternative;
      that needs field-level bidirectional coupling, a larger redesign deferred
      past M188. `backward_feasible()` likewise still returns `None` for
      `DerivedOutput`, so mid-stream re-solve back-prop is separate and deferred.
    - **Mid-stream re-cascade `configure_output` DONE, M189.** A caps-driven
      transform now re-resolves its output target on a mid-stream `CapsChanged`,
      not only at startup: both transform-arm re-cascade paths (the linear
      coordinator arm in `runner.rs` and the DAG `transform_arm` in
      `graph_runner.rs`) call `configure_output(&forward_caps)` after
      `configure_pipeline` accepts. (Startup already delivered it on both paths,
      M185.) The remaining runners that lacked it (`run_simple_pipeline`,
      `run_source_fanout`, `run_fanin_sink`) carry no caps-driven-transform slot
      with a downstream link, source->sink, source->tee->sinks, and
      sources->merger->sink respectively, so there is nothing to resolve there;
      the transform-bearing runners (`run_source_transform_sink`,
      `run_linear_chain`, `run_muxer_sink`) all route through the coordinator or
      the DAG runner, which deliver it.
  - Decided **pragmatic** (keep convenience props as extensions + make the caps
    route work) over strict gst-only; the items above are what "make the caps
    route work" actually requires.

- **M183 (done): videocrop / videobox property-model alignment.** `videocrop`
  now uses gst's per-edge insets `top/bottom/left/right` (was `x/y/width/height`);
  `videobox` uses gst's signed `top/bottom/left/right` (>0 crop, <0 border) +
  `fill` (added `yellow`), replacing the unsigned `border-*`. Old names replaced,
  not aliased (pre-release).

## Tensor substrate / zero-copy layout transforms (M180+)

Direction: make a strided tensor *view* the substrate for raw numeric media, so
layout-preserving transforms (flip, transpose, crop, channel reorder) are views
over the same bytes rather than copies, and raw frames feed ML inference
zero-copy. Tensor is the substrate *beneath* the semantic `Caps`, not a
replacement for them; planar/subsampled video (NV12, I420) is a *list* of views,
not one tensor. The zero-copy win is real but bounded to layout-preserving ops
(decode / colorspace / normalize / resample are arithmetic and copy regardless).

- **M180 (done).** `g2g-core::tensor::TensorView {dtype, shape, signed byte
  strides, offset}` (fixed-rank, `Copy`, heap-free) + `MemoryDomain::SystemView`
  (`Arc<[u8]>` backing + view). `VideoFlip` flips packed RGBA/BGRA zero-copy by
  composing strides on the same `Arc`; planar / owned-buffer inputs keep the copy
  path. Proven by `m180_zerocopy_flip` (source buffer pointer reaches the sink
  *through* the flip = zero copies).

- **M181: deferred orientation descriptor + sink-capability negotiation.** An
  *eagerly-applied* strided view defeats hardware flip silicon: DRM/KMS
  `plane.rotation`, Wayland `set_buffer_transform`, VAAPI VPP
  `rotation_state`/`mirror_state`, and the D3D11 VideoProcessor all rotate/mirror
  in fixed-function for free, but they want "original buffer + a rotate
  descriptor", not pre-flipped pixels in a negative-stride layout (a DRM FB has a
  single positive pitch). A hw sink handed a flipped `SystemView` would have to
  materialize (full CPU copy) first and miss the silicon, i.e. worse than not
  flipping. So a layout transform needs *two* code paths, chosen by the sink:
  - **deferred descriptor** (leave bytes + layout untouched, tag "rotate-180")
    when the sink advertises it can absorb an orientation -> `VideoFlip` is a
    pass-through that tags the frame, sink applies the transform in hardware;
  - **eager realization** (strided view, or CPU materialize) as the fallback, and
    the right path for CPU / GPU-shader-sampling consumers and for chaining
    multiple view-ops before one materialize.

  Reuse the existing capability-negotiation pattern (M12 allocation query +
  zero-copy GPU domains; GStreamer's `video-direction` / `GstVideoOrientation`
  pushed to `waylandsink` / KMS). Pieces: an orientation descriptor on the frame
  (or caps/segment) the sink can read; a sink-capability advertisement
  ("absorbs rotation X"); `VideoFlip` branching on it; one sink (KMS or Wayland)
  wired to consume it. CI-testable via a mock capability-advertising sink that
  records "applied in hardware vs materialized".

- **Queued (other modalities):** `Caps::Audio` gains a real `[frames, channels]`
  view (clean-win, no subsampling); planar video plane-LIST
  (`SmallVec<TensorView>`); text-as-embedding once an embedding element lands.

## Gap analysis to 80% parity (updated 2026-06-22, post-M217)

The honest read: the ~80% / "credible GStreamer replacement" bar is **reached**
for the Linux / Windows ingest / transcode / stream / display paths. The two
hardest tracks (CSP negotiation, the DAG runner) and the full lifecycle spine
(state machine + preroll, seek + SEGMENT, auto-plug / decodebin / playbin) were
done by M96. M107-M167 filled codec / container / streaming breadth (containers
MP4 / MPEG-TS / Matroska-WebM / FLV / Ogg / fMP4-CMAF, multi-codec ffmpeg decode,
AV1 / VP8-9 / MJPEG encode, the tag system, property system + `gst-launch` DSL +
`gst-inspect`, HLS / DASH / RTMP ingest). M168-M210 then closed the remaining
*structural* parity items: PTS-ordered muxer fan-in (M204), multi-output demuxer
/ bounded-N dynamic pads (M205), gst-launch mux fan-in (M208) + demux fan-out
(M210), flattening bins + ghost pads (M209). M211-M217 added reverse (M212) +
non-flushing accumulating (M211) seek and the zero-copy keep-on-GPU branch
(M213-M217). Element breadth is essentially done (~122 plugins).

**The structural runtime is finished** -- including the β allocation re-cascade
(M18 single-hop -> M70 N-hop `GraphCoordinator` walk; DESIGN.md §4.13.3), the
state machine, seek, and auto-plug. **What remains is platform breadth,
transports, and depth, not core runtime work.** Grouped by how much each moves a
parity claim, highest leverage first:

**1. Platforms -- the biggest remaining track.** Decode is now started on both
new platforms; the rest of each platform's elements are owed.
- macOS: `VtDecode` (VideoToolbox H.264) landed M218 and **compiles in CI** (the
  native `macos` runner). Owed: vtencode, AVFoundation capture, Core Audio, Metal
  present. See `### Platform: macOS`.
- Android: `MediaCodecDec` (NDK MediaCodec H.264) landed M219 and
  **cross-compiles in CI** (`aarch64-linux-android`; there is no native Android
  runner). Owed: encode, Camera2, AAudio, Surface present. See `### Platform:
  Android`.
- Both new decoders compile in CI but are **not yet runtime-validated on a
  device** (actual decode, and the Android output color-format packing). Linux
  and Windows are fully covered.
- **Linux audio out + modern capture -- DONE.** `AlsaSink` / `PulseSink` /
  `PipeWireSink` + `PipeWireSrc` on Linux, `MfVideoSrc` on Windows. Remaining
  depth: PipeWire video / screen capture, DMABUF zero-copy out of the sinks.

**2. Egress / transports** (ingest side done; egress track now open). RTMP egress
(`rtmpsink`) **landed M221** -- the `RtmpPublisher` client mirrors the server-side
`RtmpSession` (shared `ChunkReader`), publishing an FLV byte stream to an RTMP
endpoint (`flvmux ! rtmpsink`); validated sans-IO against the server session, live
publish is user-side. RTP **RTX (RFC 4588) landed M222** (`rtx` module +
`UdpSink`/`UdpSrc` `with_rtx`). **RTSP server started M223** (`rtspserversink` +
sans-IO `rtspserver` responder; PLAY-direction serving validated, ANNOUNCE/RECORD
spoken by the responder). **SRT both directions started M224** (`srtsink`/`srtsrc`
+ sans-IO `srt` module; HSv5 handshake + NAK ARQ validated g2g<->g2g). Remaining:
SRT encryption/TSBPD/congestion + real-peer interop, the RTSP ANNOUNCE/RECORD
*source* element, multi-client RTSP. RTP **FEC (ULPFEC) landed M225** (`ulpfec`
module + `UdpSink`/`UdpSrc` `with_fec`; jitter buffer / RTCP / NACK / RTX / FEC
all done now); FlexFEC + multi-level burst FEC remain. See the RTMP/SRT and RTP
entries under `### High`.

**3. Depth (works today, not yet future-proof).**
- Negotiation: dynamic / *unbounded* request pads (bounded-N done M205/M210),
  multi-element subgraph mid-stream re-solve (1-link done), the
  `scale_then_convert` geometry-pin known limit.
- Seek: trick-mode KEY_UNIT frame selection (needs a per-frame keyframe flag,
  deferred to its first consumer), segment seeks (CMAF / DASH), re-preroll after
  a flushing seek when paused, more seekable sources. (reverse + non-flushing
  done, M211/M212.)
- Codecs: pure-Rust / wasm decode (dav1d / rav1d, vpx) to drop the ffmpeg FFI.

**4. GPU keep-on-GPU pillar.** M213-M217 closed zero-copy GPU fan-out, the
GPU-resident tensor domain, the GPU-tensor inference consumer (`WgpuInference`),
and input-side NV12 surface-import. Remaining: CUDA<->wgpu interop to join the
NVDEC decode side to the wgpu inference / preprocess side. (ML-pillar work,
orthogonal to gst parity.)

**5. Niche / small.** Subtitle depth, controllers (animated properties),
clock/timeoverlay, generic EGL `GlSink`, compositor GPU companion + NV12 mixing,
audio-mixer rate / layout reconciliation, the last three bus messages
(segment-done / stream-status / clock-lost, each gated on a subsystem not present).

Platforms (item 1) is the only large remaining track; everything else is "add an
element / add a transport," which the codebase does cheaply and repeatably. The
core runtime, CSP negotiation (including the allocation re-cascade), and the
lifecycle spine are done. The open decision is which definition of done applies:
credible Linux/Windows replacement (reached), or full cross-platform parity
(gated on finishing macOS / Android, item 1).

## GStreamer parity gaps

Capabilities GStreamer's core runtime has that g2g doesn't. Each is sized
as the number of focused implementation sessions to reach functional
parity (not full polish), with a priority for "is g2g a credible
GStreamer replacement for this use case?"

### Critical — complete

All former Critical items are done: the DAG runner (`run_graph` over arbitrary
linear / fan-out / fan-in / diamond topologies, with the historical runners as
thin builders over it), the pipeline state machine + preroll, seek + SEGMENT,
and auto-plug / `decodebin` / `playbin`. Architecture: DESIGN.md §4.13.3
(runner), §4.14 (lifecycle + seek), §4.13.9 (auto-plug). The depth that rounds
these out is no longer blocking but still open:

- **Seek depth.** `Mp4Src` is seek-aware (M148: flushing seek, keyframe `SNAP_BEFORE`
  reposition, post-flush `Segment`), the first real repositioning source, and
  `SyncSink` clips decoded frames before the target so accurate seek presents the
  exact requested frame (M149). Remaining: non-flushing / accumulating `do_seek`
  (advance base by elapsed running time), reverse + trick-mode (rate != 1.0) handling
  at the sink, segment seeks (CMAF / DASH transitions), re-preroll after a flushing
  seek when paused, and making the other repositionable sources (`FileSrc`, the
  demuxers) seek-aware.
- **Auto-plug depth.** The `uridecodebin` URI front door is done (DESIGN.md
  §4.13.9). Remaining: richer factory construction params (geometry / device /
  file path, beyond the chosen output caps) and a hardware-backed end-to-end
  decode-through-`decodebin` run (the `ffmpeg` / `vaapi` autoplug tests read
  templates only, decode nothing; the `uridecodebin` ffmpeg test asserts the
  decoder is *spliced*, but does not run real media through it either).
  `FfmpegVideoDec` now advertises all the codecs it decodes (H.264 / H.265 / VP8
  / VP9 / AV1) on its sink template (M111), so autoplug routes them, not just
  H.264.

### High

Production-shape needs that block specific real-world use cases.

- **Per-frame metadata system (`FrameMeta` + ML relation graph).** 4–5
  sessions. `Frame` is currently `{ domain, timing, sequence }` with no
  side-channel for per-frame data; **this is now built (M98/M99), see §5.4 and
  the follow-ups below.** GStreamer's
  `GstMeta` (typed attachable per-buffer blobs: `GstVideoMeta` strides,
  `GstNetAddressMeta`, `GstReferenceTimestampMeta`,
  `GstVideoRegionOfInterestMeta`) and the newer `GstAnalyticsRelationMeta`
  (a relation graph of typed ML detection / classification / tracking /
  segmentation nodes) together cover everything from "this DMABUF's
  parent-buffer refcount" to "detection-N has classification-N has
  tracking-id-T." Without an equivalent, every ML pipeline more complex
  than "decode → classify → display" (i.e. every real one — detector →
  tracker → classifier → overlay) re-derives joins through `Caps::Tensor`
  serialization, which is type-unsafe and loses cross-frame identity.
  Two layers in one primitive:

  - **`FrameMeta` trait** — typed, per-frame, attachable. Implementation
    `HashMap<TypeId, Box<dyn FrameMeta>>` on `Frame`, with transform
    callbacks (`fn propagate(&self, kind: TransformKind) -> Propagation`)
    that say whether this meta survives an identity pass, a crop / scale,
    a re-encode, a serialization. Mirrors GstMeta's
    transform_func / copy_func / free_func contract. Keep it `no_std`-
    compatible: a small inline `arrayvec` of trait objects on
    `no_std + alloc`, gated to skip the `HashMap` overhead on RTOS.
  - **`AnalyticsMeta` — the relation graph layer.** Typed Mtd nodes
    (`ObjectDetection { bbox, label, confidence }`, `Classification
    { topk: SmallVec<(LabelId, f32)> }`, `Tracking { object_id }`,
    `Segmentation { mask_handle }`) plus directed edges between them
    encoded as `Vec<(MtdId, MtdId, RelationKind)>`. One
    `AnalyticsMeta` per frame holds the whole relation graph; downstream
    elements (overlay, recorder, alarm trigger) read by node-kind +
    relation traversal instead of by tensor offset. Built on top of the
    `FrameMeta` primitive, not a separate system.

  **Built (M98/M99), documented in DESIGN.md §5.4:** the `FrameMeta` trait
  (`as_any` + `propagate(Transform) -> Propagation`), the typed `FrameMetaSet`
  (attach / typed-get / iterate / propagate, ZST when the `metadata` feature is
  off), and the `AnalyticsMeta` relation graph (`ObjectDetection` /
  `Classification` / `Tracking` nodes + directed relations, normalized bbox), with
  the first producer `g2g-ml::DetectionPostprocess` (YOLOv8 decode + NMS).

  **Built (M100-M103), documented in DESIGN.md §5.4:** metadata through
  fan-out (`FrameMetaSet` holds `Arc<dyn FrameMeta>` + is `Clone`,
  `try_clone_packet` shares meta across a tee, `FrameMeta::clone_box` backs
  copy-on-write in `get_mut`); the detection overlay in two backends, the CPU
  `AnalyticsOverlay` (`analytics`) and the Vello GPU `VelloAnalyticsOverlay`
  (`vello-overlay`) emitting the new `MemoryDomain::WgpuTexture` (kept on GPU);
  and `WgpuSink` (`wgpu-sink`), the GPU presentation sink that blits a
  `WgpuTexture` onto an offscreen target or a caller-built `wgpu::Surface` with no
  readback, the overlay and sink sharing one device via `gpu::GpuContext`. The
  full path `decode -> tee -> {detect, video} -> overlay -> WgpuSink` is closed.

  Remaining follow-ups:
  - **A `Segmentation` node** (mask handle) to round out the GstAnalytics node
    kinds, plus more standard metas (`GstVideoMeta`-style strides, ROI).
  - **`push` vs `pull` propagation across transforms.** Today an element calls
    `FrameMetaSet::propagate(kind)` explicitly (push); whether the runner should
    intercept and apply it generically (pull) is open, deferred until more
    meta-aware transforms exist.
  - **A turnkey windowed runner for `WgpuSink`.** The sink presents to an
    app-supplied surface; a small winit/SCTK example that opens a window, builds
    the surface, and drives the overlay->sink graph would give an out-of-the-box
    on-screen demo (validated on a real display, like `wayland_smoke`).

- **Remaining bus message types.** Core bus coverage is done (DESIGN.md §4.15).
  The `tag` message landed (M137: `BusMessage::Tag(TagList)` + `oggdemux`
  VorbisComment). Still missing, each gated on a subsystem we don't have yet:
  `segment-done` (segment seeks), `stream-status` (thread pool), `clock-lost`
  (clock re-election). Smaller follow-ups: buffering on interior links; periodic
  QoS; QoS from the display sinks (the display sinks now *can* sync to the clock,
  see "Clock-synchronised presentation" below; the QoS late-drop + `Qos` post
  from them is the next step).

- **Logging framework (`g2g-core::log`) follow-ups.** The `GST_DEBUG`-analog
  facade is done (M179, DESIGN.md §4.15): levels, per-category thresholds,
  `LogSink`, the `g2g_*!` macros, a `std` stderr sink + `G2G_DEBUG` env, and
  `run_graph` instance-naming (`<category>N`) + per-element addition logs, with
  `VideoFlip` as the self-logging worked example. **Remaining:** roll instance
  naming + lifecycle logging into the bespoke linear runners
  (`run_simple_pipeline`, `run_source_transform_sink`) and the muxer path, not
  just `run_graph`; add `set_instance_name` self-logging to more elements (only
  `VideoFlip` does so far; the rest get named + runner-logged but emit nothing of
  their own); explicit names from the `gst-launch` `name=` syntax (currently
  auto-generated only); glob category matching (`*sink*:5`, currently exact + a
  single `*` default); a structured-fields / timestamped record format and a
  ring-buffer sink; and a custom (non-type-name) category override per element.

- **Clock-synchronised presentation (present each frame at its PTS).** First step
  DONE (M169, DESIGN.md §4.4): `ClockSync` (elected clock + base time) +
  `AsyncElement::set_clock_sync` (default no-op, dyn-mirrored), delivered by the
  runner after clock election; `WaylandSink` holds each frame until its
  running-time deadline (Segment-mapped, first-frame-anchored, re-anchors on
  Flush). Delivered by both the linear runners and the DAG runner `run_graph`
  (M172): after clock election the elected `ClockSync` is handed to every sink
  node, so a display sink PTS-paces in any topology. QoS late-drop +
  `BusMessage::Qos` from the display sinks DONE (M173, DESIGN.md §4.4):
  `WaylandSink` drops a frame past its deadline by more than `max_lateness` and
  posts a `Qos` report, matching `SyncSink` (M85). Upstream QoS propagation DONE
  for the source→sink runner (M174, DESIGN.md §4.4): a sink's `take_qos` rides the
  per-link reverse `QosSlot` to the producer as `PushOutcome::Qos`; `SyncSink`
  originates it and `VideoTestSrc` skips frames in response. Relay *through* a
  transform DONE (M175, DESIGN.md §4.4): the runner wires a transform's output
  `SenderSink` with a relay handle (`relay_qos_to`) to its input link's reverse
  `QosSlot`, so a downstream QoS report walks one hop at a time back to the source
  across any number of generic transforms; wired in both `run_source_transform_sink`
  and the DAG runner (`run_graph` / `run_linear_chain`, which the `WaylandSink`
  demo uses). Playing-transition anchoring DONE (M176, DESIGN.md §4.4): under a
  `StateController` the runner arms a `PlayAnchor` on the elected clock and hands
  each sink `ClockSync::with_play_anchor`; `set_state(Playing)` stamps it with
  `clock.now_ns()` at the play edge (cleared on the way down), so a sink anchors to
  when streaming began rather than to startup / the preroll frame consumed during
  `Paused`. `WaylandSink` re-bases a provisional preroll anchor onto the play edge
  and forces a first-frame re-anchor after a seek `Flush`.

  **Remaining (deferred, the two below are platform/hardware-bound and not
  CI-testable on the dev host, so parked behind testable tracks):**
  - **KMS vblank reconciliation** (pick-frame-for-next-flip) + Wayland
    frame-callback co-scheduling. Needs a DRM/KMS presentation sink: `WaylandSink`
    is SHM software present, with no flip timing to reconcile against, and no KMS
    sink exists yet. The sink would pick the frame whose deadline is closest to the
    next predicted vblank rather than sleeping to an absolute deadline. Validate on
    a real display (like `wayland_smoke`), not in CI.
  - **A/V clock slaving** (elect an audio device clock as master so video follows
    audio). Needs an audio sink that *provides* a clock tracking buffer consumption
    (`provide_clock` returning a `ClockCandidate` whose `now_ns` advances with
    samples played), plus election preferring it over a live-source clock. Drift
    correction (skew between the audio clock and the video source) is the hard part.
    Validate against a real audio device.
  - **A QoS-aware transform** that acts on a relayed report itself (a decoder
    dropping non-reference frames) rather than only forwarding it (M175). This one
    *is* CI-testable; deferred only until a decoder that can cheaply drop frames is
    the bottleneck.

- **Compositor GPU companion + depth.** The CPU `Compositor` (RGBA8 pixel
  mixer: position / z-order / per-pad alpha, input-0-driven cadence, plus per-pad
  bilinear scaling via `with_size`, M97; configurable background colour via
  `with_background`, M146) is done (DESIGN.md §4.13.6). Remaining: a wgpu compute
  variant for HD/many-input scale; NV12/I420 mixing without a round-trip through
  RGBA; and configurable output cadence.

- **Adaptive streaming (HLS, DASH).** Built and documented in DESIGN.md §4.17:
  the `HttpSrc` fetch layer, `HlsSrc` (TS + fMP4/CMAF via `#EXT-X-MAP`, live
  reload, AES-128 `#EXT-X-KEY` decryption), the `SampleAesDecrypt` transform (TS
  SAMPLE-AES H.264 + AAC, key auto-wired from `HlsSrc` via a shared handle), fMP4
  `cbcs` SAMPLE-AES decryption in `fmp4demux` (H.264/H.265, `tenc`+`senc`-driven,
  same shared key handle), and `DashSrc` (static MPD, `SegmentTemplate`
  `@duration` or `SegmentTimeline`, `$Number$`/`$Time$` addressing, dynamic
  (live) MPD reload).
  **Remaining HLS:** SAMPLE-AES key rotation mid-stream (the shared handle holds a
  single key, fine for a constant per-stream key); cbcs audio (AAC) and per-sample
  IV (cenc/cbc1), `saiz`/`saio` aux-info + `seig` sample groups (the cbcs path uses
  `senc` + `tenc` defaults); encrypted fMP4 init segments; byte-range segments;
  throughput-driven ABR (the current pick is static by declared bandwidth);
  live-edge start (skip to the last few segments); and mid-stream variant switching.
  **Remaining DASH:** the wall-clock `@duration` live profile (segment
  availability from `availabilityStartTime`; the SegmentTimeline live case is
  done), `SegmentList`/`SegmentBase` byte-range, multi-period, and
  throughput-driven ABR.

- **SRT / RTMP transports.** RTMP **ingest and egress are DONE**: `RtmpSrc`
  (`rtmp` feature) + sans-IO `rtmp::RtmpSession` accept a publisher and emit
  `ByteStream{Flv}` for `flvdemux`; `RtmpSink` (M221) + sans-IO
  `rtmp::RtmpPublisher` are the inverse, publishing an FLV byte stream out to an
  RTMP server (`flvmux ! rtmpsink`). Both share the `ChunkReader` reassembly.
  Documented in DESIGN.md §4.12b. Remaining RTMP: the complex (HMAC digest)
  handshake some CDNs require, multiple streams, server-acknowledgement back-
  pressure. **SRT ingest + egress STARTED (M224):** `SrtSink` (caller) + `SrtSrc`
  (listener) + the sans-IO `srt` module (HSv5 handshake, data/control wire layer,
  NAK-based ARQ via `SrtSender`/`SrtReceiver`) carry an MPEG-TS byte stream over
  UDP; validated g2g<->g2g end to end (handshake + loss recovery). **Remaining
  SRT:** encryption (AES / KMREQ), TSBPD timing, congestion control, and
  libsrt/ffmpeg interop validation (the wire format follows the draft, so it is
  designed to interop, but real-peer testing is operator-side).

- **RTP receive-side stack.** Largely **done**; RTX/FEC remain. The reordering
  jitter buffer (M94: `rtpjitter::RtpJitterBuffer`), the RTCP control protocol
  (M95: `rtcp` — SR/RR/BYE + RFC 4585 Generic NACK build/parse + RFC 3550
  reception stats), and the feedback loop (M96: `UdpSrc` sends RR + NACK,
  `UdpSink` honors NACK by retransmitting from a bounded history, RTP/RTCP muxed
  per RFC 5761) all shipped. Remaining:
  - **RFC 4588 RTX -- DONE (M222).** `g2g-plugins::rtx` (`build_rtx_packet` /
    `parse_rtx_packet`, `no_std`) wraps a resend in a distinct payload type / SSRC
    / sequence with the original sequence (OSN) prepended and reconstructs the
    byte-exact original (marker bit rebuilt from the RTX header, not the OSN).
    `UdpSink::with_rtx` / `UdpSrc::with_rtx` opt in; validated end-to-end over the
    lossy-proxy loopback. SSRC- and session-multiplexed RTX both supported.
  - **FEC -- ULPFEC DONE (M225).** `g2g-plugins::ulpfec` (single-level RFC 5109,
    `no_std`): `build_fec_packet` / `recover_packet` + `FecEncoder` / `FecDecoder`;
    `UdpSink::with_fec` / `UdpSrc::with_fec`. Recovers a single per-group loss with
    no round trip, validated over a one-way lossy loopback (RTCP + retransmit off).
    Remaining FEC: FlexFEC (RFC 8627), and multi-level / interleaved ULPFEC for
    burst loss (single-level recovers one loss per group).
  `RtspSrc` via `retina` covers the RTSP case (retina has its own jitterbuffer).

- **Property system + introspection + `gst-launch` DSL — DONE (M104-M106).**
  `g2g-core::property` adds a name/value bag over the `with_*` builders
  (`PropValue` / `PropKind` / `PropertySpec` / `PropError`, no_std + alloc) and
  `properties` / `set_property` / `get_property` on `AsyncElement` / `SourceLoop`
  (+ dyn mirrors), zero-cost defaults like `latency()`. The `Registry` builds
  elements by name (`LaunchFactory`, `make_source` / `make_element`) and dumps a
  `gst-inspect`-style `inspect(name)`. `runtime::parse_launch` turns a
  `"a key=v ! b ! sink"` string into a runnable `Graph`. Architecture: DESIGN.md
  §4.16. Property coverage broadened in M107 (`VideoScale` / `VideoCrop` /
  `VideoConvert` / `AudioTestSrc` / `AudioConvert` / `AudioResample` / `FileSink` /
  `FileSrc`) plus `g2g-plugins::registry::default_registry()` so `parse_launch`
  works out of the box. **Done (M178): rich `gst-inspect` dump.** Element types
  declare `ElementMetadata { long_name, klass, description, author }` via a
  zero-cost `metadata()` opt-in (like `properties()`), and `PropertySpec` gained
  `default` / `range` / enum `values` / read-write `flags`; `inspect(name)` now
  emits a GStreamer-shaped Factory Details + Element Properties dump and the no-arg
  list shows `name: Long-name`. Metadata is declared on a representative set of
  elements so far. **The feature-gated elements are now registered + carry
  metadata** (`default_registry` gains a `register_feature_gated` helper, each
  block `#[cfg]`-gated like its module: the codecs opus/av1/vpx/mjpeg, `fmp4demux`,
  the network sources/sinks rtsp/udp/http/hls/dash/rtmp, and the Linux A/V
  elements v4l2/ffmpeg/vaapi/wayland/kms/alsa/pulse), so `gst-inspect --all` and
  `parse_launch` see them when their feature is on. **Remaining depth:** carry
  metadata + properties on muxers (their inspect path does not build an instance
  today); property-set the feature-gated sources from text (`location=` / `uri=`
  on rtsp/http/hls/dash/v4l2, currently default placeholders); a value grammar for
  spaces / enums-as-named-flags; and a GUI/tooling introspection surface beyond the
  text dump. **Done (M112):** `filesrc` takes its byte-stream caps via a
  `bytestream-format` property (`mpegts` / `matroska` / `ogg` / `auto`, the last
  sniffing via `typefind`), so a text pipeline feeds a demuxer from a file.
  **Done (M117):** the `Caps` text grammar (`capsfilter::parse_caps`) + the
  property-bearing `CapsFilter`, with the inline `! video/x-raw,format=nv12,... !`
  shorthand recognized by `parse_launch`.
  **Done (M118):** `gst-launch` branching, a chain parser where `name=t` names an
  element and a `t.` reference opens a branch; `tee` is the structural fan-out
  node, its output width derived from the branch count, broadcasting each frame to
  every branch.
  **Done (M122):** text muxer fan-in. An element with several inbound links is a
  muxer, built from the registry's `MuxerFactory` with the input count derived
  from link degree; feeding chains end in a `m.` tail ref, the muxer chain last.
  `funnel` (the `InterleaveMux` N-to-1 forwarder) is registered in
  `default_registry`. Muxer `key=value` properties and a muxer-first chain
  ordering remain unsupported.

### Medium / niche

Smaller-scope items, mostly orthogonal to the architecture.

- **Tag system — core + four readers + one writer DONE (M137-M141).**
  `g2g_core::tag::{Tag, TagList}` (typed common keys + `Other` fallback) delivered
  via `BusMessage::Tag`. Readers: `oggdemux` `OpusTags` VorbisComment (M137),
  `flvdemux` FLV `onMetaData` AMF0 (M138), `matroskademux` Segment `Tags` + `Info`
  `Title` (M139), `mp4src` iTunes `udta/meta/ilst` atoms (M140). Writer:
  `matroskamux` emits a whole-stream `Tags` element via `with_tags` (M141) and
  `flvmux` writes an `onMetaData` AMF0 script tag via `with_tags` (M142), and
  `Mp4Sink` writes a `moov/udta/meta/ilst` via `with_tags` (M147), so a `TagList`
  round-trips through all three. Remaining: Matroska `Targets`-scoped (per-track)
  tags and nested SimpleTags; MP4 freeform (`----`) and integer atoms (track/disc
  number); and a per-stream tag merge policy for multi-stream containers.
- **Audio mixer — v1 DONE (M130).** `g2g-plugins::audiomixer::AudioMixer` sums
  aligned S16LE inputs (arrival-aligned, registered as the `audiomixer` muxer for
  the M122 text fan-in). Remaining: sample-rate + channel-layout reconciliation
  (pairs with `AudioConvert` / `AudioResample`) and PTS-based alignment.
- **Subtitle support.** 2 sessions. `Caps::Subtitle` variant,
  text/srt/webvtt demuxers, a text-overlay element (tied to
  compositor for the rendering half).
- **Controllers (animated properties).** 2 sessions.
  `gst-controller`-equivalent for animating properties over time
  (zoom 1.0 → 2.0 over 5 seconds). Niche but real for production
  graphics.
- **`gst-launch` text DSL — DONE (M106).** `runtime::parse_launch` takes
  `"videotestsrc num-buffers=3 ! videoflip method=rotate-180 ! fakesink"` and
  builds a runnable `Graph` (caps-filter syntax landed M117, branching M118).
  See the property-system entry above and DESIGN.md §4.16.

### What we already do better

For the full picture — capabilities where g2g beats GStreamer at parity
or better, so the gap survey above isn't read as "we're behind on
everything":

- Structured `NegotiationFailure` (names the responsible element pair
  + the caps that didn't intersect) vs GStreamer's opaque
  `not-negotiated`.
- Memory safety across hot-swap (`ArcSwap`) vs `pad_block` /
  `pad_unlink` choreography.
- `LinkPolicy` per edge (Block / DropOldest / DropNewest configured at
  graph build) replaces explicit `queue` element insertion.
- `no_std + alloc` baseline — GStreamer cannot reach RTOS / Cortex-M.
- Sans-IO protocol layer (testable without sockets, embeddable in any
  executor).
- Compile-time pad templates + the typed `Caps` split — `not-negotiated`
  becomes a type error at most boundaries instead of a runtime
  failure.

## Missing elements

Atomic element-shaped gaps. Each entry is 1–2 sessions unless noted; they
are listed because GStreamer ships them and a production multimedia stack
expects them. Grouped by category.

### Video transforms

`videoscale` (M55), `videorate` (M56), `videocrop` (M62), and
`videoflip` / `videorotate` (M66) are DONE. A `WgpuVideoScale` GPU companion
(DMABUF / D3D11Texture input, compute-shader resample) remains a follow-up.

- **`videobalance` — DONE (M124).** Brightness / contrast / saturation on packed
  RGBA / BGRA (`g2g-plugins::videobalance`). Hue remains: a faithful chroma
  rotation needs `sin`/`cos` (a `libm` dep the `no_std` baseline avoids).

### Audio transforms

- **`audioresample`.** Sample-rate conversion (48 kHz → 16 kHz for ASR, etc.).
  1–2 sessions. Mandatory for cross-rate paths; without it, every audio
  source has to negotiate the consumer's exact rate.
- **`audiomixer` — v1 DONE (M130).** Summing S16LE fan-in (see the parity-gaps
  list above); sample-rate + channel-layout reconciliation remain.

### Capture sources

Platform-coverage. Linux video capture is done (`v4l2src`, DESIGN.md §4.12a);
WASAPI covers Windows audio in/out; Linux video capture is `v4l2src`. Windows
video capture (`mfvideosrc`) and the modern PipeWire path (`pipewiresrc`) landed
(see below). Linux non-PipeWire audio capture (`alsasrc` / `pulsesrc`) and
PipeWire *video* / screen capture remain.

- **`v4l2src` follow-ups.** The element streams system-memory YUYV today. Next:
  MMAP DMABUF output (maps to `MemoryDomain::DmaBuf` for zero-copy into the GPU
  decode/display path) and format-flexible negotiation (MJPEG-mode UVC, other
  fourccs) instead of the fixed YUYV.
- **`pipewiresrc` — DONE (audio).** `g2g-plugins::pipewiresrc::PipeWireSrc`
  (`pipewire` feature, Linux) captures interleaved PCM (`PcmS16Le` / `PcmF32Le`)
  off the PipeWire graph: a `pw::stream` input on a dedicated main-loop worker
  thread feeds the async `run` loop over a channel, with a fixed requested format
  the PipeWire adapter converts to (deterministic caps). **Remaining:** video +
  screen capture (the SPA video pod + `param_changed` negotiation), and DMABUF
  output for zero-copy into the GPU path.
- **`mfvideosrc` — DONE (authored, Windows-only).**
  `g2g-plugins::mfvideosrc::MfVideoSrc` (`mf-video-src` feature) enumerates video
  capture devices and drains frames via an `IMFSourceReader` (NV12 / YUY2 to
  system memory) on a COM/MTA worker thread, the video sibling of `WasapiSrc`.
  Type-checked for the `x86_64-pc-windows-gnu` target; owes a first build + camera
  smoke test on a real Windows host (the unverified-on-Linux situation it shares
  with `mf-decode` / `wasapi-src`). **Remaining:** D3D11 zero-copy output, and a
  size/rate request beyond the device default.
- **`alsasrc` / `pulsesrc`** (Linux audio capture, non-PipeWire). 1 session
  each. Wide host coverage where PipeWire isn't installed.
- **Screen capture.** 2–3 sessions per platform. Linux: PipeWire (via the
  source above). Windows: DXGI Desktop Duplication API. macOS: ScreenCaptureKit.
  OBS / video-conferencing use case.
- **`avfvideosrc` / `avfaudiosrc`** (macOS AVFoundation). Part of the macOS
  platform gap (see below).

### Network sources

- **RTP receive-side jitter buffer.** `UdpSrc` + the `RtpH264Depayloader` are
  done (raw RTP-over-UDP ingest, DESIGN.md §4.12b) — the broadcast-contribution
  shape, distinct from `RtspSrc` (RTSP-over-TCP via retina). The basic
  depayloader assembles in order and drops on a sequence gap; a real jitter
  buffer (packet reorder, loss concealment, RTCP RR/NACK) is the remaining
  receive-side robustness work. Also SDP/SPS-driven caps discovery so `UdpSrc`
  reports real geometry instead of a declared hint.
- **`souphttpsrc` / `HttpSrc` — DONE (M155).** `g2g-plugins::httpsrc::HttpSrc`
  (`http-src` feature) GETs a URL via `reqwest` and streams the body as
  `Caps::ByteStream` chunks then `Eos`, feeding the byte-stream demuxers like
  `FileSrc` does. **Remaining:** header-sniff (`bytestream-format=auto`) and a
  `uridecodebin` `http(s)://` handler (both need a negotiation-time ranged fetch),
  HTTP range requests / seeking, and retry/reconnect for live edges.
- **`rtmpsrc`** (RTMP ingest) is **DONE** (`RtmpSrc`, `rtmp` feature; see the
  RTMP transport entry in parity gaps and DESIGN.md §4.12b).
- **`srtsrc`** (SRT ingest). Tied to the SRT transport in parity gaps.

### Sinks

- **Linux audio sinks — DONE.** `alsasink` (`g2g-plugins::alsasink::AlsaSink`,
  `alsa-sink`, libasound), `pulsesink` (`pulsesink::PulseSink`, `pulse-sink`, the
  blocking libpulse "simple" API), and `pipewiresink` (`pipewiresink::PipeWireSink`,
  `pipewire`) all play interleaved PCM (`PcmS16Le` / `PcmF32Le`) on a dedicated
  worker thread, the Linux analogs of `WasapiSink`. ALSA / Pulse backpressure via
  the blocking write; PipeWire pulls on its own clock so its queue is leaky
  (bounded ~1 s, drops oldest). **Remaining:** a host smoke test against a real
  device, channel-count / sample-format reconciliation beyond stereo S16/F32, and
  DMABUF/zero-copy paths where the stack supports it.
- **Generic `GlSink` over EGL.** 2–3 sessions. `CudaGlSink` is NVIDIA-specific;
  `WaylandSink` is software NV12. A vendor-neutral GL ES presentation sink
  over EGL with a generic NV12 / RGBA shader covers Mesa / Intel / AMD
  without CUDA, plus Android (via SurfaceFlinger EGL) once that platform
  exists.
- **`autovideosink` / `autoaudiosink`.** 1 session each, tied to the URI /
  state-machine work. Picks the right sink for the host automatically.

### Containers

- **MKV / WebM demux — DONE (M110).** Pure-Rust EBML / Matroska parser
  (`g2g-plugins::matroska::MatroskaDemuxer` + the `MkvDemux` element), fed by the
  new `Caps::ByteStream{Matroska}` link type: descends the Segment, reads Tracks
  (CodecID -> H.264 / H.265 / VP8 / VP9 / AV1 / AAC / Opus, geometry, audio
  params) and Cluster SimpleBlock / Block frames, with per-codec `MkvStream`
  selection (`matroskademux stream=vp9`, default VP9) and `CapsChanged`-refined
  output caps from Tracks. Registered as `matroskademux`. Block lacing (Xiph /
  EBML / fixed) is split (M113). The muxer `matroskamux`
  (`g2g-plugins::matroska::MatroskaMuxer` + the `MkvMux` element, single track,
  one Cluster per frame, `webm` / `matroska` DocType by codec) landed in M115. The
  Segment `Tags` element + `Info` `Title` surface as `BusMessage::Tag` via
  `MkvDemux::with_bus` (M139), and the muxer writes a `Tags` element via
  `MkvMux::with_tags` (M141, see the tag system above). Unknown-size Clusters (the
  live read shape) demux (M143), laced frames are spaced by `DefaultDuration`
  (M144), and the muxer batches frames into unknown-size Clusters (M145).
  **Remaining:** Cues / seeking, multi-track muxing, and `Targets`-scoped
  (per-track) tags.
- **MPEG-TS `tsdemux` — DONE (M108, M109).** Pure-Rust demuxer
  (`g2g-plugins::mpegts::TsDemuxer` + the `TsDemux` element): PAT/PMT/PES ->
  elementary streams, fed by the `Caps::ByteStream{MpegTs}` link type
  (`FileSrc(ByteStream) ! tsdemux ! h264parse ! ...`). M109 added per-stream
  selection (`TsStream` / the `stream` property): H.264 / H.265 video and AAC
  audio, one selected stream per output pad (`tsdemux stream=aac ! aacparse`),
  with `h265parse` / `aacparse` registered in `default_registry`. The muxer
  `mpegtsmux` (`g2g-plugins::mpegts::TsMuxer` + the `TsMux` element, single
  stream, real PSI CRC) landed in M114. **Remaining:** multi-stream / multi-program
  muxing + selection; PCR-based timing. SRT/HLS feed TS over the wire (needs the
  network byte-stream source path).
- **FLV demux + mux (`flvdemux` / `flvmux`) — DONE (M119 / M120).** Pure FLV
  parser + muxer (`g2g-plugins::flv::FlvDemuxer` / `FlvMuxer` + the `FlvDemux` /
  `FlvMux` elements) on the new `Caps::ByteStream{Flv}`: the "FLV" header then
  `PreviousTagSize` / tag pairs, the demuxer forwarding and the muxer wrapping the
  H.264 (AVC) video and AAC audio access units per `FlvStream` selection (h264 |
  aac), sniffed by `typefind`. The `onMetaData` script-tag metadata is read as tags
  (M138) and written by `flvmux` via `with_tags` (M142, see the tag system above).
  **Remaining:** codec-config / extradata plumbing, and multi-track muxing.
- **OGG demux — DONE (M116).** Pure RFC 3533 parser
  (`g2g-plugins::ogg::OggDemuxer` + the `OggDemux` element), fed by the new
  `Caps::ByteStream{Ogg}` link type: "OggS" pages, segment-table lacing with
  cross-page packet reassembly, codec sniff from the first packet (`OpusHead`),
  setup headers skipped. `ByteStream{Ogg} -> Audio{Opus}`; registered as
  `oggdemux`, sniffed by `typefind`. **Remaining:** granule-position timing,
  Vorbis output, multi-stream Ogg, and `oggmux`.
- **CMAF / fMP4 segmented.** 2 sessions. Already have `Mp4Sink` /
  `Mp4Src` (fragmented); the CMAF-specific signalling for adaptive
  streaming is a thin layer on top.

### Codecs

H.264 / H.265 / AAC are in, and **VP8 / VP9 / AV1 / H.265 decode landed in M111**
via the generalized `FfmpegVideoDec` (libavcodec; Linux / `ffmpeg` feature),
which the M108-M110 demuxers feed. The notable remaining gaps are the encoders
and the pure-Rust / browser decode paths:

- **VP8 / VP9 encode — DONE (M151), gated + unverified-on-host.**
  `g2g-plugins::vpxenc::VpxEnc` (`vpx` feature) wraps libvpx via the `vpx-encode`
  crate: `RawVideo{I420}` -> `CompressedVideo{Vp8|Vp9}`, the GStreamer
  `vp8enc`/`vp9enc` analog. Not pure Rust (links system libvpx + bindgen), so it is
  feature-gated and out of pure-Rust / no_std builds. Authored against the
  `vpx-encode` 0.6 API but compile-unverified on the dev host (no libvpx here); owes
  a real build/run on a libvpx machine, with a gated `Vp9Parse` round-trip test
  ready. **Remaining:** validate on a libvpx host, then a pure-Rust / wasm VP8/VP9
  decode path (vs the libavcodec FFI) is separate.
- **AV1 encode — DONE (M150).** `g2g-plugins::av1enc::Av1Enc` (`av1-encode`
  feature) wraps the pure-Rust `rav1e` encoder: `RawVideo{I420}` ->
  `CompressedVideo{Av1}`, the first portable / CI-verifiable video encoder (the MF
  encoders are Windows-only). `default-features = false` drops rav1e's NASM / CLI /
  threading, so it builds with no system deps (MSRV 1.74 within the workspace 1.75).
  Round-trips through `av1parse` (M136). **Remaining:** bitrate / quantizer rate
  control surface, 10-bit / 4:4:4, and an `Av1Dec` pure-Rust decode (`dav1d` /
  `rav1d`) to pair with it ffmpeg-free.
- **Opus encode + decode — DONE (M177).** `g2g-plugins::opusenc::OpusEnc` and
  `opusdec::OpusDec` (`opus` feature) wrap libopus via the `audiopus` crate, the
  WebRTC-default audio codec to pair with the existing `opusparse` (M133): `OpusEnc`
  buffers interleaved S16LE PCM and emits one Opus packet per 20 ms frame
  (zero-padding a partial tail at EOS); `OpusDec` decodes each packet back to
  S16LE. v1 is 48 kHz mono/stereo (the rate Opus always decodes at, so no
  resample). Not pure Rust (libopus FFI), so gated + std-only; the `opus` feature
  needs a system libopus dev package (pkg-config), like the other native codecs.
  **Remaining:** float (F32) PCM in/out, other frame durations, packet-loss
  concealment (decode of a missing packet), and bitrate/complexity tuning beyond
  the bitrate setter; a pure-Rust Opus path if one matures.
- **MJPEG decode — DONE (M152, I420 in M154).** `g2g-plugins::mjpegdec::MjpegDec`
  (`mjpeg` feature) decodes `CompressedVideo{Mjpeg}` to `RawVideo{Rgba8}` (default)
  or `RawVideo{I420}` (`with_output_format`) via the pure-Rust `zune-jpeg` crate (no
  system dep, CI-verified on any host). Geometry recovered from the JPEG headers per
  frame, emitted as `CapsChanged`. **Remaining:** a `mozjpeg`-backed fast path under
  a feature flag if decode CPU cost matters, and a direct YCbCr->I420 path (skip the
  RGBA intermediate).
- **JPEG decode + encode — DONE (M152 / M153 / M154).** Decode is `MjpegDec`
  (`zune-jpeg`, M152 above); encode is `g2g-plugins::mjpegenc::MjpegEnc`
  (`mjpeg-encode` feature) via the pure-Rust `jpeg-encoder` crate:
  `RawVideo{Rgba8|Bgra8|I420}` -> `CompressedVideo{Mjpeg}`, intra-only per frame
  for thumbnail / snapshot / low-latency capture. M154 added I420 in/out on both
  (planar 4:2:0, BT.601 limited range, even dims) so they pair with a video
  encoder / decoder without a `VideoConvert`. **Remaining:** a single-still image
  sink, and a direct YCbCr path to skip the RGBA intermediate.

### Parsers

For every codec we host, the bitstream parser (SPS / VPS / sequence-header
extraction, framing detection, framerate / dimension recovery) is what feeds
the negotiation. `H264Parse`, `H265Parse` (M68), and `AacParse` (M75) are in;
the codec-specific parsers below are still missing, and the two shipped
parsers have follow-ups:

- **`H265Parse` follow-ups.** Framerate from the VUI `timing_info` (past the
  PCM / ref-pic-set loops, deferred until a real-stream reference is
  available), and validation against a real H.265 elementary stream.
- **`AacParse` follow-ups.** LATM / LOAS framing (MPEG-TS / broadcast),
  AudioSpecificConfig synthesis for a downstream decoder (needs the metadata
  side channel), and validation against a real ADTS stream.
- **`OpusParse` — DONE (M133).** `g2g-plugins::opusparse` reads each packet's
  TOC byte (RFC 6716 §3.1) to recover mono/stereo channels (+ coder / bandwidth /
  frame duration), emitting `Audio{Opus, channels, 48 kHz}` caps. Multichannel
  (family 1, count in `OpusHead`) deferred.
- **`Vp8Parse` — DONE (M134).** `g2g-plugins::vp8parse` reads the VP8 frame tag
  (RFC 6386 §9.1) and, on a key frame, the start code + 14-bit width/height,
  emitting `CompressedVideo{Vp8, Fixed w/h, Rate::Any}`. Geometry only (no
  in-bitstream framerate).
- **`Vp9Parse` — DONE (M135).** `g2g-plugins::vp9parse` reads the VP9
  uncompressed header (marker, profile, `49 83 42` sync, `color_config`, 16-bit
  size fields, MSB-first) on a key frame, emitting
  `CompressedVideo{Vp9, Fixed w/h, Rate::Any}`. Intra-only resizes deferred.
- **`Av1Parse` — DONE (M136).** `g2g-plugins::av1parse` walks the OBUs
  (low-overhead format, LEB128-sized) and parses the sequence header
  (`reduced_still_picture_header` + operating-points loop) for
  `max_frame_width/height`, emitting `CompressedVideo{Av1, Fixed w/h, Rate::Any}`.
  With `OpusParse` (M133), `Vp8Parse` (M134), and `Vp9Parse` (M135) the codec
  parser gap is closed.

### Overlay / effects

- **`textoverlay` / `clockoverlay` / `timeoverlay`.** `textoverlay` is done
  (M171, DESIGN.md §4.18): a `no_std` CPU element rendering SRT / WebVTT cues by
  PTS with an embedded 8x8 bitmap font over a translucent box. Remaining: a
  mixed-case TrueType GPU backend (`cosmic-text` / `swash` / `vello`, the
  analytics-overlay CPU/GPU split) and the `clockoverlay` / `timeoverlay`
  siblings (same renderer, clock-derived text). Tied to the compositor work in
  parity gaps (overlays are one input to a compositor).

### Platform: macOS

A whole platform gap. We compile for Linux + Windows + wasm32 + bare-metal;
macOS had zero element coverage. 5–8 sessions for the baseline:

- **`vtdecode` — STARTED (M218).** VideoToolbox H.264 decode (`VtDecode`,
  `vtdecode` feature), the macOS counterpart of `MfDecode`: Annex-B H.264 in,
  NV12 `System` out, via `VTDecompressionSession`. Pulls SPS/PPS for the
  `CMVideoFormatDescription` and feeds AVCC VCL NALs per frame (the
  `annexb::{h264_parameter_sets, to_avcc}` helpers, host-tested). Establishes the
  macOS platform plumbing: `vtdecode` feature + the `objc2` /
  `objc2-video-toolbox` / `objc2-core-media` / `objc2-core-video` /
  `objc2-core-foundation` deps under `[target.'cfg(target_os = "macos")']`.
  **COMPILE-PENDING:** written against the objc2 0.3.2 API but never built (no
  macOS in CI); needs a first `cargo build` on a Mac to settle a few FFI details
  (import paths for `OSStatus`, the `CFRetained` adopt, the `CVImageBuffer` ->
  `CVPixelBuffer` cast, the `CMVideoFormatDescription` type), each marked `// NOTE`
  in `vtdecode.rs`. HEVC, a `CVPixelBuffer`/`IOSurface` zero-copy domain, and the
  registry wiring (`avdec_h264` alias) are the next steps.
- **`vtencode`** — VideoToolbox H.264 / HEVC encode.
- **`avfvideosrc` / `avfaudiosrc`** — AVFoundation camera + microphone.
- **`coreaudiosink` / `coreaudiosrc`** — Core Audio in/out.
- **`metalvideosink`** — Metal presentation.

Each individually is 1–2 sessions; the platform integration (`framework`
linking, `objc2` bindings, macOS-specific feature gates) is the bulk of
the work and only pays off once (M218 paid most of it for the decode side).

### Platform: Android

Another platform gap. Android's `MediaCodec` + `SurfaceTexture` for
hardware decode, `Camera2` for capture, `AAudio` for audio, `Surface` for
presentation. Similar 5–8 session shape to macOS.

- **`mediacodec` — STARTED (M219).** NDK MediaCodec H.264 decode (`MediaCodecDec`,
  `mediacodec` feature), the Android counterpart of `VtDecode` / `MfDecode`:
  Annex-B H.264 in, NV12 `System` out, via the `ndk` crate's `AMediaCodec`
  (no JNI). Feeds Annex-B access units directly and the SPS/PPS as `csd-0` /
  `csd-1` (reuses `annexb::h264_parameter_sets`); drives the codec synchronously
  and packs the output (semi-planar / planar) to NV12 honoring the codec's
  stride / slice-height. Establishes the Android plumbing: `mediacodec` feature +
  the `ndk` (feature `media`) dep under `[target.'cfg(target_os = "android")']`.
  **No native Android CI runner**, so the `features (android)` job
  *cross-compiles* (`cargo check --target aarch64-linux-android`) — the same
  compiler feedback the macOS job gives, but actual decode is validated on a
  device. Output color-format handling beyond semi-planar / planar (vendor /
  `COLOR_FormatYUV420Flexible`, via `AImageReader`), HEVC, a zero-copy
  `AHardwareBuffer` domain, and the `Surface` present sink are the next steps.

### Other

- **`videotestsrc` pattern coverage — DONE (M123).** `gradient` / `snow` /
  `moving-bar` plus `smpte` (75% colour bars), `checker`, `ball` (bouncing), and
  `zone-plate` (integer concentric-ring chirp). Integer-only, `no_std`-safe. A
  sinusoidal (vs square-wave) zone plate would need `libm`; deferred.
- **RTSP server (`rtsp-server`) -- STARTED (M223).** `RtspServerSink` + the
  sans-IO `rtspserver::RtspResponder` host an RTSP endpoint: a player runs
  OPTIONS / DESCRIBE / SETUP / PLAY and the sink streams the pipeline's H.264 as
  RTP/UDP (validated end-to-end over loopback). The responder also speaks the
  publisher path (ANNOUNCE / RECORD). **Remaining:** multiple concurrent clients,
  TCP-interleaved transport, RTCP/keepalive during PLAY, and the receive-side
  (ANNOUNCE/RECORD) *source* element that ingests a pushing publisher.
- **WebRTC sendrecv full-stack.** 5+ sessions. `WebRtcSrc` is data-channel
  ingest only; a complete `WebRtcBin`-equivalent with ICE, DTLS-SRTP, full
  media-engine negotiation is its own track.

### Priority summary

`videoscale`, `videorate`, `videocrop`, and `videoflip` are DONE (M55 / M56 /
M62 / M66) and dropped from this table.

| Element | Sessions | Why it matters |
| :--- | :--- | :--- |
| `pipewiresrc` / `mfvideosrc` | 2 each | live camera on PipeWire / Windows (`v4l2src` done) |
| RTP jitter buffer | 2–3 | reorder/loss/RTCP (`UdpSrc` + depay done) |
| `HttpSrc` | 2 | prereq for HLS / DASH (needs a byte-stream type + consumer first) |
| VP8 / VP9 / AV1 / Opus codecs | 3 + 3 + 3 + 2 | WebRTC + modern web |
| MJPEG decode | 1 | low-end RTSP cameras |
| Linux audio sinks | 1 each | host audio output |
| Generic EGL `GlSink` | 2–3 | vendor-neutral GPU present |
| MKV / WebM | 3 | common delivery |
| MPEG-TS | 3 | broadcast + SRT carrier |
| `textoverlay` family | 2–3 | production overlays |
| macOS platform | 5–8 | whole platform |
| Android platform | 5–8 | whole platform |
| RTSP server | 4–5 | host endpoints |
| WebRTC sendrecv | 5+ | full WebRTC media engine |

With the video transforms and `audioresample` shipped, the cheap,
highest-frequency transform gaps are closed. `v4l2src` (DESIGN.md §4.12a) and
`UdpSrc` raw-RTP ingest (§4.12b) took g2g from a "process incoming streams"
framework toward a "produce streams" one; the remaining capture tier
(`pipewiresrc` / `mfvideosrc` cameras, an RTP jitter buffer, and `HttpSrc` once
it has a byte-stream consumer) extends that across platforms and makes g2g
substantially more credible as a GStreamer replacement.

## Tier-1 element sprint — detailed plan

11–14 sessions. Closes the most embarrassing gaps so g2g passes a 10-minute
developer evaluation: resize, reframe, crop, audio-resample, and live camera
capture on Linux + Windows.

**Status (2026-06):** Phase A transforms `VideoScale` (M55), `VideoRate`
(M56), `VideoCrop` (M62), and `VideoFlip` (M66) are DONE (native + wasm32;
see CHANGELOG). Only `AudioResample` (A4) and the Phase B capture sources
remain; the shipped Phase A detail has been removed.

Goal at end of sprint: a self-contained demo with no external feed —

```
v4l2src (camera) → videoscale → videorate → wgpupreprocess →
  ortinfer (CUDA EP) → tensorpostprocess → waylandsink
```

— that exercises the full ML video stack from local hardware capture
through inference to display. Same shape on Windows with `mfvideosrc → ... →
d3d11sink`. This is the "yes you can build a real thing" baseline.

### Phase A — Software transforms (open remainder)

`VideoCrop`, `VideoRate`, `VideoScale`, and `VideoFlip` are DONE (M55 / M56 /
M62 / M66). Only the audio transform remains.

**A4 — `AudioResample` (1–2 sessions).** Sample-rate conversion using the
`rubato` crate (pure Rust, designed for SRC, supports both fixed and
arbitrary ratios). `DerivedOutput` keyed on `with_target_rate(hz)`. Caps
preserve `format` and `channels`; sample-rate replaced.

- *Choice:* `rubato` over libsamplerate FFI because pure Rust + a real
  maintainer. `rubato`'s `SincFixedIn` is the right shape for our async
  pull model.
- *Verify:* round-trip 48 kHz → 16 kHz → 48 kHz, assert THD+N within
  expected sinc-filter ripple; channel layout preserved; PTS arithmetic
  honours the rate change.
- *Why last in Phase A:* needed for ASR pipelines (Whisper wants 16 kHz)
  but no audio path is on the demo critical line yet.

**End-of-Phase-A demo:** `RtspSrc → ffmpegdec → VideoScale(720p) →
VideoRate(10) → WgpuPreprocess → OrtInference → TensorPostprocess →
FakeSink` runs end-to-end on a non-720p, non-10-fps source. The first
pipeline that doesn't require the user to hand-pick a stream that
matches the model's expected geometry.

### Phase B — Live capture (6–8 sessions)

Closes the "no live camera anywhere" gap. Ordered by platform: Linux
first (the dev-loop machine), Windows second, modern Linux PipeWire
third.

**B1 — `V4l2Src` (2 sessions).** Linux-only, `v4l2` feature. Hand-rolled
ioctl FFI (no `v4l2` crate dep — small surface, similar to the cudarc
decision). Open via `/dev/videoN` (configurable), `VIDIOC_QUERYCAP` +
`VIDIOC_S_FMT` to pin geometry and pixel format, `VIDIOC_REQBUFS` +
`VIDIOC_MMAP` to set up MMAP buffers, `VIDIOC_DQBUF` / `VIDIOC_QBUF` loop.

- *Output caps:* `Caps::RawVideo { format: Yuy2 | I420 | Nv12, .. }` for
  raw, `Caps::CompressedVideo { codec: Mjpeg, .. }` for MJPEG cameras
  (most cheap webcams). `MemoryDomain::System` for v1; DMABUF export
  via `VIDIOC_EXPBUF` is a 1-session follow-up.
- *Format negotiation:* probe via `VIDIOC_ENUM_FMT` + `_FRAMESIZES` +
  `_FRAMEINTERVALS`; the async `intercept_caps` (DESIGN.md §4.13) is
  the natural fit since the probe is sync ioctls.
- *PTS:* `v4l2_buffer.timestamp` is monotonic-clock; map to the pipeline
  reference clock at configure time.
- *Verify:* `v4l2-ctl --list-devices` for a built-in webcam on the dev
  host; manual smoke test `v4l2src → videoscale → waylandsink` with
  YUY2 output. MJPEG path needs the MJPEG decoder (Phase B follow-up
  or use a YUY2-capable webcam).
- *Skip:* the V4L2 sub-device / metadata-node surface (industrial
  cameras with separate ISP control); not needed for the consumer
  webcam baseline.

**B2 — MJPEG decode — DONE (M152).** `MjpegDec` (`mjpeg` feature) wraps the
pure-Rust `zune-jpeg` decoder: `Caps::CompressedVideo { codec: Mjpeg }` →
`Caps::RawVideo { format: Rgba8 }`. Chose `zune-jpeg` over `image` / `mozjpeg`
(pure Rust, no system dep, fast). Unblocks the half of consumer webcams that only
emit MJPEG; pairs with `V4l2Src` / `MfVideoSrc` once they negotiate MJPEG.

**B3 — `MfVideoSrc` (2 sessions).** Windows-only, `mf-video-src` feature.
Media Foundation Source Reader pattern, paralleling `WasapiSrc` /
`MfDecode`'s COM/MTA contract (`unsafe impl Send` with the documented
ownership-transfer justification). Enumerate video capture devices,
`IMFSourceReader::SetCurrentMediaType` to pin format, `ReadSample` loop
to drain frames.

- *Output caps:* `Caps::RawVideo { format: Nv12 | Yuy2, .. }` to
  `MemoryDomain::System` (CPU copy out of the IMFSample). D3D11
  zero-copy is a follow-up, same pattern as `MfDecode`'s D3D11 deferred
  item.
- *Verify:* on a Windows host, enumerate the built-in / USB camera,
  run `MfVideoSrc → VideoScale → D3d11Sink`. Manual smoke; no CI gate.

**B4 — `PipeWireSrc` (2–3 sessions).** Linux, `pipewire` feature. Single
element covers camera + microphone + screen capture, using the
`pipewire-rs` crate against the PipeWire client API. The hardest one of
the three because PipeWire's negotiation is two-step (announce node →
negotiate format → start streaming) and async-shaped.

- *Output caps:* depends on the stream type (video / audio / screen).
  Geometry negotiated via SPA pod format.
- *Verify:* PipeWire daemon is universal on modern Fedora / Ubuntu /
  Arch desktops; manual smoke `pipewiresrc → ... → sink`. Screen
  capture variant pairs with the OBS use case once a compositor
  element exists.
- *Why last:* PipeWire isn't always running in headless / server
  contexts where `V4L2Src` is more reliable, so V4L2 is the lower
  floor.

**End-of-Phase-B demo:** the goal pipeline at the top of this section
runs locally on the developer machine with no external feed. Same demo
on Windows substituting `MfVideoSrc` and `D3D11Sink`. This is the
"yes, g2g is a real framework you can build something with" baseline.

### Sequencing rationale

Phase A first because:
- All four are CI-testable, no hardware required.
- Each unblocks downstream work without depending on the others.
- They surface any negotiation / runner edge cases on simple
  derivable-output elements before the more complex capture sources
  exercise them.
- 5 sessions is two weeks of work; a visible, contained win.

Phase B after Phase A because:
- The capture sources only become demo-able once `VideoScale` /
  `VideoRate` are in (cameras emit fixed geometry / fixed framerate
  that almost never matches the consumer's needs).
- Hardware-coupled work has slower feedback loops; getting the SW work
  done first gives a longer runway of CI-green commits.

Within Phase B, V4L2 → MJPEG → MF → PipeWire because:
- V4L2 is the lowest-floor capture path (works on every Linux box,
  including headless / WSL2).
- MJPEG is a small interruption that unblocks half the world's cheap
  webcams.
- MF mirrors `MfDecode`'s already-paid COM/MTA design cost.
- PipeWire is the most modern but also the most negotiation-complex;
  doing it after V4L2 + MF means the runner's source-loop patterns
  are well-exercised by then.

### Open decisions before A1 starts

- **T1.** `VideoCrop` / `VideoScale` accept the configured rect / target
  dims at construction (`with_*` builder), not as a runtime property.
  Consistent with every other element in tree. The
  property-system parity gap (above) will eventually add runtime
  setting; this sprint pre-dates that.
- **T2.** `AudioResample` uses `rubato`, not libsamplerate or a
  hand-rolled SRC. Confirm before A4 starts. Adds ~80 KB to the binary,
  no system deps, MIT licensed. **Update (2026-06):** `rubato` 3.0.0
  (current) needs rustc 1.85, above the workspace MSRV 1.75, and is not
  `no_std`. So A4 needs one of: bump the MSRV to 1.85, pin an older
  `rubato` that builds on 1.75, or hand-roll a windowed-sinc SRC. Unlike
  the other Phase A transforms, `AudioResample` would be `std`-gated, not in
  the `no_std` baseline. Decide the MSRV/dep question before starting.
- **T3.** `V4l2Src` ships hand-rolled ioctls (no `v4l2` crate dep).
  Surface is small (≈15 ioctls + 6 ioctl-arg structs); pulling a
  binding crate brings a transitive `nix` dep and obscures the small
  Linux-specific surface for no win.
- **T4.** `MjpegDec` chooses between `image` (broad, slower) and
  `mozjpeg` (libjpeg-turbo FFI, fast, system dep). *Recommendation:*
  `image` for the SW baseline (pure Rust, no system dep); a
  `mozjpeg`-backed alternative slots in later under a feature flag if
  the JPEG-decode CPU cost matters in production.

### Sizing

| Phase | Element | Sessions | Cumulative |
| :--- | :--- | :--- | :--- |
| A1 | `VideoCrop` | 1 | 1 |
| A2 | `VideoRate` | 1 | 2 |
| A3 | `VideoScale` | 2 | 4 |
| A4 | `AudioResample` | 1–2 | 5–6 |
| B1 | `V4l2Src` | 2 | 7–8 |
| B2 | `MjpegDec` | 1 | 8–9 |
| B3 | `MfVideoSrc` | 2 | 10–11 |
| B4 | `PipeWireSrc` | 2–3 | 12–14 |

Phase A only (5–6 sessions, ~2 weeks) closes the cheapest gaps and
unblocks the existing RTSP-driven pipelines. Phase A + B1 + B2
(8–9 sessions, ~3 weeks) adds the Linux capture baseline. Full
sprint (11–14 sessions, ~5 weeks) reaches Windows + modern Linux.

## First credible product path — 18-session sequenced plan

**Why this exists.** The plans above are organized by capability area. Each
makes architectural sense in isolation. None of them, on its own, produces
something a developer can pick up and evaluate. This section is the
opposite framing: **if the project only ever ships one focused track, this
is it.** It selects the minimum cut across the other plans that produces a
working, defensible, end-to-end demo — and stops there. Everything else
waits.

**Goal at end of 18 sessions.** A working browser application — deployed
to a public URL, working in Chrome / Firefox / Safari — that exercises
the full cross-target value proposition:

```
WebSocketSrc → H264Parse → WebCodecsDecode → WebGPUPreprocess →
  WebOrtInference (segmentation) → WebGPUSink (compose + present)
```

— with every link after `WebCodecsDecode` staying on the GPU. The exact
same g2g element graph (substituting platform sources / sinks) also runs
as a native Linux desktop binary against an RTSP feed, producing the
same output to a `CudaGlSink`. Both deployments build from one
`Cargo.toml`, one set of element source files, one negotiation model.

This is the single artifact that makes the "cross-target multimedia
framework with hardware-resident graph in the browser" claim
demonstrable rather than aspirational. Nothing else in the project tells
that story until it exists.

### Sequencing principle

Three phases. Each phase produces a verifiable artifact; the project can
honestly pause after any phase boundary and still have shipped value.

| Phase | Sessions | End-of-phase artifact |
| :--- | :--- | :--- |
| **P1 — SW transforms** | 3 | Existing pipelines can resize and reframe. |
| **P2 — Browser zero-copy chain (W1)** | 8 | A browser pipeline that decodes H.264 and renders via WebGPU with no host round-trip. |
| **P3 — Reference application + cross-target proof** | 7 | A deployed, public-URL demo + the matching native binary + a written positioning piece. |

The sequencing is deliberate: P1 first because it unblocks both the
browser and native pipelines without requiring browser stack decisions;
P2 second because the browser zero-copy chain is the riskiest unproven
piece and should be derisked before the application work; P3 last
because shipping the demo requires both prior phases to be done and
validated.

### Phase 1 — Minimum SW transforms — DONE

`VideoScale` (M55), `VideoRate` (M56), and `VideoCrop` (M62) shipped, all
verified on native + wasm32. The remaining product-path work is Phases 2-3.

### Phase 2 — Browser zero-copy chain (8 sessions)

The W1 sprint, made concrete. Each step adds one element / one memory
domain primitive; each is independently testable.

**Update (2026-06): the GPU-resident chain below is not achievable from
idiomatic Rust, confirmed by an API survey.** (1) wgpu cannot import a
WebCodecs `VideoFrame` as an external texture on its WebGPU backend
(`Features::EXTERNAL_TEXTURE` is DX12-only); the import is raw
`web_sys::GpuDevice::import_external_texture` behind
`--cfg=web_sys_unstable_apis`. (2) wgpu cannot share or adopt a device on
wasm (`Device::as_hal` returns `None` for the webgpu backend, no
`create_device_from_hal`), so a wgpu compute pass cannot run on ORT's device
and a web_sys-imported external texture cannot be sampled by wgpu. (3) the
`ort-web` crate (the maintained onnxruntime-web Rust path) runs ORT in a
separate WASM module and syncs tensors through CPU, with no `fromGpuBuffer`,
and a `web_sys::GpuBuffer` cannot be extracted from a `wgpu::Buffer`. So P2.4 /
P2.5 as written (a wgpu `WebGPUPreprocess` handing GPU-buffer tensors to ORT)
cannot be built. GPU-residency in-browser would require raw `web_sys` WebGPU
(external-texture import + compute shader) plus hand-rolled onnxruntime-web JS
bindings on one ORT-owned device, all browser-unverifiable on the dev host.
The buildable alternative is a CPU-round-trip MVP via `ort-web`, reusing the
System-RGBA `WebCodecsDecode` (M40) + `CanvasSink` (M41); it still proves the
same `.onnx` runs native and in-browser, just not GPU-resident. Landed toward
the original plan: P2.1 (`WebGPUExternalTexture` carrier, M57) and P2.2
(`WebCodecsDecode::with_gpu_output`, M58) are in tree but only reachable by the
raw-web_sys path. The transform/inference groundwork (M55 `VideoScale`, M56
`VideoRate`, M59 `OrtInference` tensor-input, M61 native GPU pipeline) is
backend-agnostic and stands regardless of which browser path is chosen.

- **P2.1 — `MemoryDomain::WebGPUExternalTexture` in core** (1 session).
  Wraps a `web_sys::VideoFrame` handle behind a `WebGPUKeepAlive` trait
  object the same way `OwnedCudaBuffer` wraps a CUDA pointer behind
  `CudaKeepAlive`. Core never links `web-sys` — the producing element
  supplies the owner as `Box<dyn WebGPUKeepAlive>`. *Why first:*
  everything downstream depends on this carrier; landing it isolated
  proves the core extension is clean.
- **P2.2 — `WebCodecsDecode` GPU-resident output path** (1 session).
  Replace the existing `videoFrame.copyTo(rgba_buffer)` path with
  emitting the `VideoFrame` directly in the new memory domain. Keep
  the `copyTo` path behind a `with_system_output()` builder for the
  CPU-consumer case. *Why:* this is the load-bearing zero-copy moment —
  the decoder hands the GPU surface forward instead of pulling it back.
- **P2.3 — Async WebGPU device handshake** (1 session). The
  `request_adapter` / `request_device` calls are async and
  `configure_pipeline` is not. Resolve the device once at pipeline
  construction time (a builder-side `await`) and store the `GPUDevice`
  + `GPUQueue` on the element; element configuration uses the cached
  handles. This is the foundation P2.4 and P2.5 sit on.
- **P2.4 — `WebGPUPreprocess` element** (2 sessions). Wasm32 sibling of
  `WgpuPreprocess`: imports a `WebGPUExternalTexture` via
  `device.importExternalTexture()`, runs a compute shader that produces
  a normalized f32 NCHW tensor or downsized RGBA texture, emits in a
  GPU-resident tensor domain. *Why 2 sessions:* the
  `GPUExternalTexture` lifetime is single-render-pass and forces a
  specific shape (sample in the same pass that imports), which the
  shader and the element loop have to be designed around.
- **P2.5 — `WebOrtInference` element** (1 session). New element in
  `g2g-ml`, wasm32-only, behind a `web-ort` feature. Wraps the
  `onnxruntime-web` JS module via `wasm-bindgen`: the page loads
  `ort.min.js` (or imports the npm package), the Rust element calls
  into it through generated bindings. Same `Caps::Tensor` in /
  `Caps::Tensor` out contract as the native `OrtInference` so it slots
  into the existing tensor graph. Register the WebGPU execution
  provider ahead of the WASM CPU fallback so a usable GPU runs the
  inference; WebNN is the future optimal EP (Apple Neural Engine via
  Safari, Snapdragon Hexagon) and can be probed first with WebGPU as
  fallback once it's mainstream. **Why this over `BurnInference` for
  the browser:** runtime ONNX model loading from any URL (Burn's
  `burn-import` is build-time codegen — every model means a rebuild),
  thousands of battle-tested operators, async-first design matching
  WebGPU, and a real Microsoft-maintained runtime that's used in
  production by Bing / Office. `BurnInference` stays as the native
  pure-Rust backend; the browser path uses ORT Web. Same model file
  loads in both targets because native `OrtInference` and
  `WebOrtInference` are both ONNX consumers.
- **P2.6 — `WebGPUSink`** (1 session). Replaces `CanvasSink` for the
  GPU-resident case. Takes a render-ready GPU texture and presents to a
  canvas via `<canvas>.getContext("webgpu")` + swapchain. *Why simple:*
  it's a 50-line render pass + present loop once the device handshake
  exists.
- **P2.7 — In-browser validation harness** (1 session). A
  `wasm-bindgen-test` browser test plus a minimal manual HTML page that
  loads a known H.264 clip from a static asset, runs the full chain,
  asserts visual output via canvas pixel readback against a reference.
  *Why this needs its own session:* every element above compiles today,
  but the in-browser runtime is currently unvalidated. The harness is
  what burns down that debt before the application work starts.

**Verification gate before P3:** the harness runs the full
`WebSocketSrc → WebCodecsDecode → WebGPUPreprocess → WebOrtInference →
WebGPUSink` chain on a public CI runner with headless Chrome, asserts
no host round-trip via the `VideoFrame` lifetime trace, asserts visual
parity within tolerance against the native pipeline's output on the
same clip. **Until this gate passes, the demo cannot be claimed.**

### Phase 3 — Reference application (7 sessions)

The user-facing artifact. Picks one wedge use case, ships it
end-to-end. Recommendation: **edge ML on a streamed camera feed**
because it exercises hardware decode (vs `getUserMedia` which uses raw
camera frames and skips the decode pillar) and is the canonical
"why does this need to be in the browser" use case (zero install,
privacy, no server round-trip).

- **P3.1 — Pick the model and the scenario** (1 session). The model is
  the constraint. Three viable candidates:
  - **Selfie segmentation** (MediaPipe-Selfie-Segmentation shape, ~6MB).
    Wedge: video conferencing background blur. Wide appeal, clear
    demo value.
  - **Object detection** (a quantized YOLOv8-nano, ~6MB or
    DETR-tiny). Wedge: surveillance, monitoring. Stronger differentiation
    against `<video>` element solutions.
  - **Pose estimation** (BlazePose-lite). Wedge: fitness, accessibility,
    AR. Higher novelty.
  Decide once based on file size + WebGPU operator coverage of the
  candidate ONNX export (ORT Web's WebGPU EP runs most common ops but
  not all; check before committing). Rest of P3 doesn't depend on
  which.
- **P3.2 — Application scaffolding** (2 sessions). The static
  HTML / TypeScript glue that bootstraps the wasm module, wires
  controls (source URL, model URL, performance overlay), handles the
  cross-origin-isolation headers required for WebCodecs +
  SharedArrayBuffer, and provides the fallback path when WebGPU is
  unavailable (currently: Safari pre-17, headless contexts). The g2g
  wasm module is a single `wasm_bindgen` entry point that takes the
  source / model / canvas-id and returns a handle.
- **P3.3 — Native sibling demo** (1 session). The same pipeline shape
  as a Linux binary: `RtspSrc → H264Parse → FfmpegH264Dec (NvdecCuda)
  → WgpuPreprocess → OrtInference (CUDA EP) → CudaGlSink`. Same
  element source files, same negotiation model. **And critically, the
  same `.onnx` model file** — native `OrtInference` and browser
  `WebOrtInference` both load standard ONNX, so the cross-target
  artifact is literally "one model file, two deployments, identical
  output." That's a stronger claim than "same pipeline shape" alone.
  Running both deployments side by side on the same RTSP feed with
  the same model is the value proposition made literal.
- **P3.4 — Deployment** (1 session). Static-hosted at
  `glass2glass.dev` or a GitHub Pages route. Includes the cross-origin
  headers (deploy behind Cloudflare Worker or via the
  `Cross-Origin-Embedder-Policy: require-corp` +
  `Cross-Origin-Opener-Policy: same-origin` static-site shim). Includes
  a known-good RTSP feed (or recorded clip served over WebSocket from
  the same origin to dodge CORS).
- **P3.5 — Validation matrix** (1 session). Chrome / Firefox / Safari
  on macOS + Windows + Android + iOS. Document what works, what
  doesn't, why. WebGPU availability + WebCodecs codec coverage are the
  two main axes. This document is what an evaluator reads to decide
  whether the framework fits their target audience.
- **P3.6 — Positioning piece** (1 session). A short README +
  blog-shaped write-up: "here's what this demo does, here's how it's
  different from mediabunny / WebAV / raw WebCodecs apps, here's how
  the same pipeline runs native." Includes the side-by-side
  native + browser running on the same RTSP feed. This is the artifact
  that an external developer reads to decide whether to investigate
  the framework. Without it, the demo is technically real but
  invisible.

**Verification gate at end of P3:** the deployment is reachable from
an external network and works in at least Chrome stable + Firefox
stable on Linux + macOS + Windows desktop. The positioning piece is
linkable from the project README. The native sibling demo runs from a
`cargo run --example` against a known-good RTSP feed.

### Sizing

| Phase | Step | Sessions | Cumulative |
| :--- | :--- | :--- | :--- |
| P1 | VideoScale | 2 | 2 |
| P1 | VideoRate | 1 | 3 |
| P2 | WebGPUExternalTexture domain | 1 | 4 |
| P2 | WebCodecsDecode GPU-resident output | 1 | 5 |
| P2 | Async WebGPU device handshake | 1 | 6 |
| P2 | WebGPUPreprocess | 2 | 8 |
| P2 | WebOrtInference (onnxruntime-web) | 1 | 9 |
| P2 | WebGPUSink | 1 | 10 |
| P2 | In-browser validation harness | 1 | 11 |
| P3 | Pick model + scenario | 1 | 12 |
| P3 | Application scaffolding | 2 | 14 |
| P3 | Native sibling demo | 1 | 15 |
| P3 | Deployment | 1 | 16 |
| P3 | Validation matrix | 1 | 17 |
| P3 | Positioning piece | 1 | 18 |

### What this path deliberately excludes

To stay credible at 18 sessions, the following are explicitly out and
wait for later work, even though they appear elsewhere in this doc:

- **DAG runner.** The demo pipeline is linear; the DAG runner is needed
  for multi-source / multi-sink topologies (tee, mux). Worth doing
  next, but not blocking this artifact.
- **Auto-plug / element registry.** The demo names elements explicitly
  in Rust code. A user-facing "play this URL and figure out the chain"
  flow is the next-tier polish.
- **State machine / preroll / seek.** The demo plays once, end-to-end.
  Pause / scrub / seek would need the state machine work.
- **VideoCrop / AudioResample.** Not on the demo critical line. Add
  when an audio path or ROI-driven flow is built.
- **Live camera capture (V4l2Src / MfVideoSrc / PipeWireSrc).** The
  browser demo uses `WebSocketSrc` against a known clip / known feed;
  the native demo uses `RtspSrc`. Live camera capture is a
  general-purpose tier-1 win but not required for the cross-target
  proof.
- **RTP receive jitterbuffer, every codec beyond H.264.** Wait.
- **macOS / Android platforms.** The deployment runs on whatever
  WebGPU + WebCodecs + H.264 supports in those browsers, which covers
  more than the in-tree native macOS / Android story would.

### Why this sequence and not a different one

- **Why P1 before P2:** the two transforms are fast to land and
  unblock both targets. Doing them first lets P2's browser work
  proceed against a CSP solver that's already exercising more than
  trivial linear chains.
- **Why the W1 chain before the application:** the application is the
  user-facing surface but the chain is the technically risky surface.
  If `GPUExternalTexture` lifetime constraints or the async device
  handshake force a rethink, that risk lands early, in P2, not in P3
  where it would torpedo the deployment timeline.
- **Why a native sibling demo in P3 and not separately:** the
  cross-target story is the only defensible positioning. Showing the
  same pipeline native + browser in the same commit, side by side, is
  the only way that claim is concrete rather than rhetorical. It's
  cheap (1 session) because the wasm32-incompatible element
  substitutions are already in tree.
- **Why a positioning piece counts as a session:** an invisible demo
  is a non-demo. The artifact that takes an evaluator from "I clicked
  the link" to "I understand what's different here" is part of the
  deliverable.

### What success looks like at session 18

A linkable, runnable, public URL that an external developer can open
in 30 seconds and see hardware-accelerated H.264 decode + GPU-resident
segmentation + on-GPU composite in a browser tab, accompanied by a
linkable native binary doing the same with the same source code.
Plus a 500-word write-up that names the technical claim and shows the
artifacts that back it. This is the artifact that earns the project
the right to be evaluated against the broader plans in the rest of
this document; without it, those plans are speculation.

## Negotiation (DESIGN.md §4.13)

- **Forward coordinator re-solve walk** (Caps-β). The current mid-stream
  re-solve uses a startup downstream-feasibility snapshot
  (`solver::downstream_feasibility`). A downstream `DerivedOutput` element
  that must re-derive on a mid-stream input change isn't covered — its
  envelope was snapshotted against the startup input. The forward
  coordinator walk (request/reply through each arm, gathering each
  element's current constraint contribution) is the missing piece; design
  is settled but build is gated on a real driver where a downstream
  decoder sits below another format-changing transform.
- **Dynamic pads / request pads.** `tee::request-pad` and
  `mux::request-pad` style runtime branch/input addition. The fan-out and
  muxer are currently static.
- **Zero-copy tee.** `run_graph`'s tee deep-copies `System` frames and fails
  loud on a GPU domain because `PipelinePacket` is not `Clone` (GPU handles
  can't be). A refcounted shareable frame is the zero-copy-tee prerequisite.
- **Graceful per-branch drop on fan-out.** A rejecting branch currently fails
  the run loud (strict policy); a `FanOutPolicy::AllowBranchDrop` opt-in for
  graceful degradation is anticipated.
- **β allocation re-cascade across a muxer.** A muxer's inputs have no per-pad
  re-cascade channel, so the DAG β walk terminates at a muxer.
- **Timestamp-ordered fan-in (synchronized mixing).** The muxer arm drains its
  per-input channels round-robin (fair, no starvation) but in *arrival* order,
  not PTS order, so a `MultiInputElement` sees inputs interleaved by whoever
  produced first, not by frame time. The compositor papers over this for the
  "background + overlays" shape: input 0 drives cadence, overlays are latched
  best-effort (a startup buffer emits overlay-less on overflow so a late branch
  still appears, but a branch can still run ahead of another by buffer depth).
  The frame-accurate answer is a PTS-ordered merge in `muxer_arm`: buffer a
  little per input, release the lowest-PTS packet once every live input has a
  buffered frame at or beyond it (or a per-input deadline elapses), so inputs
  advance together by frame time regardless of arrival skew. Prerequisite for
  correct multi-camera grids, A/V interleave muxers, and PTS-synchronized
  compositing; it subsumes the compositor's ad-hoc priming. Gated on a use case
  that needs frame-accurate sync (e.g. lip-sync mux or a synchronized grid)
  rather than the current cadence-driver overlay model. See M93 compositor
  bring-up (CHANGELOG) for the failure modes the round-robin + startup buffer
  currently mitigate.
- **Allocation join policy across diamonds.** When two branches downstream of a
  tee propose different allocation params, the tee's outbound pool needs a join
  policy. Default sketch: most-restrictive intersection, loud failure on empty.
- **`Graph` re-run / clone for seek-and-replay.** `run_graph` consumes the
  elements via `take()`, leaving the `Graph` empty. A
  `GraphTemplate::instantiate() -> Graph` two-step is cleaner than making
  `Graph` reusable. Relevant to the Phase 1 seek work.
- **Hardware `tee -> {decode, mux}` integration test.** The
  `rtspsrc → parse → tee → {dec → wayland, mux → mp4}` diamond is covered by
  fake-element tests; a Linux run on the `rtsp ffmpeg wayland-sink` features is
  still owed.
- **Mid-stream element hot-swap.** `ElementSlot::swap` scaffolding exists
  (§4.8.2); the live swap of a real element under load isn't wired up.
- **Preference algebra.** `CapsPreferences` is a placeholder; the solver
  uses constraint-internal ordering for tie-breaks. A real
  competing-constraint scenario is needed to drive a concrete preference
  algebra (sum-of-indices is the placeholder).

## Receive / decode (DESIGN.md §4.11)

- **VaapiH264Dec on AMD desktop.** cros-codecs hard-codes a 16×16 initial
  `VAContext` and uses ChromeOS-specific GBM flags
  (`GBM_BO_USE_HW_VIDEO_DECODER`, `NV12` contiguous). Both fail on the
  Mesa `radeonsi` GBM provider. The clean fix is upstream: a cros-codecs
  surface backend that allocates VAAPI surfaces directly through libva
  (`vaCreateSurfaces`) instead of routing through GBM. Until then, `ffmpeg`
  is the Linux AMD path.
- **ffmpeg VAAPI hwaccel.** Open the `h264_vaapi` codec with an attached
  `AVHWDeviceContext(VAAPI)`, register a `get_format` callback claiming
  `AV_PIX_FMT_VAAPI`, and `av_hwframe_transfer_data` the decoded surface
  into `System` memory. Stays inside `FfmpegH264Dec`; the public
  `AsyncElement` shape doesn't change. Useful on Intel iGPUs and AMD
  desktop while the cros-codecs upstream fix is pending.
- **Zero-copy `MemoryDomain::DmaBuf` from `VaapiH264Dec`.** The
  GBM-allocated surface is already a DMA-buf; exposing its fd via
  `OwnedDmaBuf` needs a refcount story to keep the surface alive until
  downstream releases it.
- **H.265 in `VaapiH264Dec`.** The cros-codecs stateless framework supports
  it; a sibling element keyed on `VideoCodec::H265` is straightforward.
- **Upstream `Reconfigure` driven by `VaapiH264Dec` `FormatChanged`.**
  Resolution change is observed (`DecoderEvent::FormatChanged` → fresh
  `CapsChanged` downstream) but not yet plumbed as an upstream
  `Reconfigure`.
- **`MfDecode` zero-copy + DXVA.** D3D11 zero-copy output via the
  `MemoryDomain::D3D11Texture` variant, DXVA hardware acceleration
  (`MF_SA_D3D11_AWARE`), strided NV12 output. The software-decoder path
  currently assumes `stride == width`.
- **10-bit pixel formats in `FfmpegH264Dec`.** `YUV420P10` / `P010`.
  4:4:4 is accepted with chroma box-averaged down to 4:2:0; the 10-bit
  layout is endianness / bit-position-specific and was not added without
  a libav host to verify on.

## CUDA / display (DESIGN.md §4.11.5)

- **`CudaGlSink` first compile + e2e.** The sink draft is in tree (EGL on a
  Wayland surface via `wl_egl_window`, `glow` GL ES 3 program, NV12 shader,
  per-frame map/copy/unmap via the CUDA-GL interop FFI), but was authored
  off-Linux. The first compile pass on Linux+NVIDIA and the manual
  `wayland_smoke`-style benchmark (`rtspsrc → h264parse →
  ffmpegdec[NvdecCuda] → CudaGlSink`) versus the `NvdecCuvid → WaylandSink`
  system-memory baseline are owed.
- **GL-on-KMS variant of `CudaGlSink`.** Wayland is the dev-loop path; KMS
  / GBM is the production tty path. Re-uses the CUDA-GL interop core; only
  the windowing changes.
- **CUDA ↔ Vulkan external memory.** Importing a Vulkan image's memory
  into CUDA via `cudaImportExternalMemory` is the long-term direction; the
  GL path is the pragmatic first deliverable.
- **Real downstream consumer for the β allocation re-cascade.** The
  in-tree decoders record the sink's mid-stream proposal but the MFT and
  CUDA output pools are fixed at codec open; a pool that actually re-sizes
  on the mid-stream proposal exercises the cascade end-to-end. (The cascade
  itself is built and covered by a fake transform.)

## Egress (DESIGN.md §4.12)

- **RTCP sender reports.** RFC 3550 SR generation on the existing
  `UdpSink`.
- **RTSP `ANNOUNCE` / `RECORD` ingest.** The Wowza-style egress
  handshake. Sandbox blocks port 554, so this is bring-up + manual
  validation, not CI-testable.

## Embedded (DESIGN.md §6.2.1)

- **`EmbassyClock` HAL tick.** The tick is selected at the cargo feature;
  driving it on real hardware needs a HAL time driver. Host verification
  via the `block_on` pipeline is in place.
- **Full `embassy-executor` multi-task integration.** Pipelines run today
  under `embassy-futures::block_on`; the multi-task executor path uses
  the same runner futures.
- **Fixed DMA-ring capture `SourceLoop`.** A no-alloc end-to-end frame
  flow: a lifetime-carrying `SystemSlice` wires `StaticBufferPool` into
  the zero-copy path. This is the last piece of the strict no-heap
  embedded story.

## Browser / Wasm (DESIGN.md §6.3.1)

- **In-browser runtime validation.** `WebSocketSrc → H264Parse →
  WebCodecsDecode → CanvasSink` compiles for `wasm32-unknown-unknown` but
  the live `WebSocket` receive + `performance.now()` pacing is owed a
  `wasm-bindgen-test` (or manual) run.
- **WebGPU-texture zero-copy sink.** `MemoryDomain::WebGPUBuffer` into a
  `GPUTexture` needs the async device handshake to live in the
  `OwnedWebGPUBuffer` keep-alive (`request_adapter` / `request_device`
  are async, `configure_pipeline` is not).
- **Web Workers executor.** `spawn_local` drives pipelines on the main
  thread; off-main-thread Workers need JS bootstrap infrastructure.
- **HEVC in `WebCodecsDecode`.** The hook is parameterized; the
  per-codec setup parallels the H.264 path.
- **Browser MVP via `ort-web`.** A wasm32 inference element on the maintained
  `ort-web` crate (CPU tensors), wiring `WebSocketSrc -> WebCodecsDecode
  (system RGBA, M40) -> ort-web -> CanvasSink (M41)`, loaded from the same
  `.onnx` as native, deployed as a plain static HTTPS site (no COOP/COEP).
  Proves cross-target ONNX in-browser but is not GPU-resident. (Replaces the
  shelved GPU-resident P2.4/P2.5; `WebGPUExternalTexture` (M57) +
  `WebCodecsDecode::with_gpu_output()` (M58) landed but are reachable only by
  the raw-`web_sys` GPU path below.)
- **Raw-`web_sys` WebGPU path** (only if the GPU-resident browser claim is
  revived): external-texture import + compute + `ort.Tensor.fromGpuBuffer` on
  one ORT-owned `GPUDevice`, all outside `wgpu`. Large, browser-unverifiable.

## ML (DESIGN.md §5)

- **ONNX import via `burn-import`.** Build-time codegen; the
  `BurnInference` `AsyncElement` shape is what trained-weight modules
  slot into.
- **Trained-weight `Module` path for `BurnInference`.** Richer layers
  (conv, attention) once the codegen lands.
- **Decoder DMA-BUF / D3D11 surface import into `WgpuPreprocess`.** Today
  `WgpuPreprocess` uploads NV12 to a storage buffer; binding a decoder's
  surface directly into the compute pass needs the surface-import
  handshake and a GPU tensor `MemoryDomain` for the output.

## Documentation

- Architecture diagrams in [docs/](docs/) (the Pages site is text-only).
- Per-element rustdoc pass: ensure every public element type has an
  example block.
