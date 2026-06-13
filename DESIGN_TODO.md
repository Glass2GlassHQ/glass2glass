# DESIGN_TODO

Open work tracked against the architecture in [DESIGN.md](DESIGN.md). Items
here are deferrals from the spec, follow-ups blocked on a concrete driver or
upstream fix, and forward-looking tracks that the current architecture
anticipates but hasn't yet built.

## Status (2026-06)

Shipped this session (M55-M65, all on `master`):

- **P1 software transforms:** `VideoScale` (M55), `VideoRate` (M56),
  `VideoCrop` (M62), `VideoFlip` (M66), native + wasm32. `AudioResample`
  still open.
- **Browser decode groundwork:** the `MemoryDomain::WebGPUExternalTexture`
  carrier (M57) and `WebCodecsDecode::with_gpu_output()` (M58) landed, but are
  reachable only by the raw-`web_sys` GPU path (see the Phase 2 update below).
- **Inference composition:** `OrtInference::with_tensor_input()` (M59) lets a
  GPU preprocess feed inference directly; the native end-to-end ML pipeline
  (`VideoConvert -> VideoScale -> WgpuPreprocess -> OrtInference`) is verified
  on hardware (M61).
- **DAG runner:** D1 `Graph` + validation (M63); D2 `solve_graph` topological
  CSP, fan-out (M64) + muxer fan-in (M65); D3 `run_graph` over a
  `GraphNode { Source | Element }` payload, source/transform/sink/tee (M67).

New / reshaped follow-ups surfaced this session:

- **Browser MVP via `ort-web`** (replaces the shelved GPU-resident P2.4/P2.5;
  see the Phase 2 update): a wasm32 inference element on the maintained
  `ort-web` crate (CPU tensors), wiring `WebSocketSrc -> WebCodecsDecode
  (system RGBA, M40) -> ort-web -> CanvasSink (M41)`, loaded from the same
  `.onnx` as native, deployed as a plain static HTTPS site (no COOP/COEP). It
  proves cross-target ONNX in-browser but is not GPU-resident.
- **Muxer per-input-pad constraint API:** D2's `NodeConstraint::Muxer {
  inputs, output }` (M65) takes per-pad accept sets, but real muxer elements
  (`mux`) don't expose them. Add `caps_constraint_as_input(idx)` (or similar)
  so the D3 runner can build a muxer's constraint from the element. A
  prerequisite for wiring real muxers into `run_graph`.
- **DAG D3 done (M67):** `run_graph` over a `GraphNode { Source(Box<dyn
  DynSourceLoop>) | Element(Box<dyn DynAsyncElement>) }` payload, handling
  source/transform/sink/tee with fake-element tests. Two gaps surfaced and are
  the next DAG work: (1) a tee deep-copies `System` frames and fails loud on a
  GPU domain because `PipelinePacket` is not `Clone` (GPU handles can't be) - a
  refcounted shareable frame is the zero-copy-tee prerequisite; (2) muxer nodes
  are rejected, still owed the per-input-pad constraint API above. **DAG D4
  next:** the coordinator / mid-stream re-cascade over the DAG. The hardware
  `tee -> {decode, mux}` integration test is owed a Linux run.
- **Raw-`web_sys` WebGPU path** (only if the GPU-resident browser claim is
  revived): external-texture import + compute + `ort.Tensor.fromGpuBuffer` on
  one ORT-owned `GPUDevice`, all outside `wgpu`. Large, browser-unverifiable.

## GStreamer parity gaps

Capabilities GStreamer's core runtime has that g2g doesn't. Each is sized
as the number of focused implementation sessions to reach functional
parity (not full polish), with a priority for "is g2g a credible
GStreamer replacement for this use case?"

### Critical

These block any non-trivial multimedia application, not just specific
codecs or transports.

- **`run_graph(Graph)` over an arbitrary DAG.** 5–6 sessions. Today's
  runner has separate entry points (`run_linear_chain`,
  `run_source_fanout`, `run_muxer_sink`, `run_fanin_sink`), so a `tee`
  into both a display branch and a `mux` recording branch must be
  expressed by manually nesting `BranchSlot`s. A single `Graph` builder
  + `run_graph` entry point collapses the four runner shapes into one
  and unlocks arbitrary production topologies. Phased plan +
  load-bearing decisions: [DAG runner — detailed plan](#dag-runner--detailed-plan) below.

- **Auto-plug / element registry / `decodebin`-equivalent.** 4–5
  sessions. We have static pad templates as type-level metadata (the
  `PadTemplates` trait, §4.13.7) but no runtime registry to enumerate
  them, no search algorithm, no `decodebin`-equivalent that takes input
  caps and walks "find a chain of registered elements whose pad
  templates compose to raw video." A user has to know their stream is
  H.264 vs H.265 and pick the decoder by hand. GStreamer's `playbin
  uri=...` is the canonical "just play this" experience and we have no
  answer. `decodebin` returns a sub-graph, so this lands cleanly on top
  of the DAG runner.

- **Pipeline state machine (NULL → READY → PAUSED → PLAYING).** 3–4
  sessions. No formal state transitions, no preroll (sink rendering
  first frame before unpausing), no async state changes, no formal
  live-vs-non-live distinction beyond the `LatencyProfile::Live` hint.
  The "build, then run, then drop" lifetime is too coarse for editors,
  players, or anything user-interactive that needs to pause / scrub /
  resume. Load-bearing for correctness across seek and A/V sync at
  startup.

- **Seek + SEGMENT + rate control.** 4–5 sessions.
  `PipelinePacket::Flush` is the FLUSH half of seek only. Missing:
  `seek(rate, start, stop)` API, reverse playback, segment seeks (for
  CMAF / DASH segment transitions), trick play (rate != 1.0), and
  GStreamer's `GstSegment` running_time / stream_time / base_time
  decomposition. Without this we're a "live streaming framework," not a
  "multimedia framework." Tied to the state machine.

### High

Production-shape needs that block specific real-world use cases.

- **Per-frame metadata system (`FrameMeta` + ML relation graph).** 4–5
  sessions. `Frame` is currently `{ domain, timing, sequence }` with no
  side-channel for per-frame data that travels with the buffer. GStreamer's
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

  **Reserve the extension point now (1 session).** Independently of the
  full build, add a `meta: FrameMetaSet` field to `Frame` immediately,
  gated behind a `metadata` cargo feature so the no_std / Cortex-M path
  pays nothing when off. `FrameMetaSet` is a `()` ZST when the feature is
  off, an empty `SmallVec<[Box<dyn FrameMeta>; 0]>` when on. No methods
  yet, no trait body — just the field. Cost: one struct field added now,
  prevents an SemVer break later. Every code path that constructs a
  `Frame` (decoders, sources, transforms) gets one extra `..Default::default()`
  or explicit `meta: FrameMetaSet::new()`. Trivial to land, expensive to
  retrofit.

  **Design decision: defer the full build until a concrete ML detection
  element needs it.** No in-tree element produces detection metadata today
  (`TensorPostprocess::topk_classification` is whole-frame), so building
  the framework first is the classic over-engineering trap. Right time is
  alongside the first detection element (probably a YOLO-style ONNX
  postprocess that surfaces N bounding boxes). Documented now so the
  metadata system is shaped by a real client when it lands, and so the
  `Frame` struct's extension point is reserved (a `meta: FrameMetaSet`
  field, empty on `Default`, sized for the no_std case).

  **Open questions for when the build starts:**
  - Cost on the no_std baseline. A `Vec<Box<dyn FrameMeta>>` adds an
    allocation per frame on every push. The Cortex-M / Embassy path
    should be able to opt out at compile time (a `FrameMetaSet` newtype
    that's a `()` ZST when the `metadata` feature is off).
  - Transform callbacks: GstMeta's `transform_func` decides per-buffer
    whether a meta is copied through `videoconvert` / dropped by
    `videoscale` / serialized by `mp4mux`. The Rust analog is a
    `Propagation` enum returned by a trait method; the question is
    whether transforms call it proactively (push model) or whether the
    framework intercepts and queries (pull model). GstMeta uses push;
    pull is simpler but means every transform element has to know about
    every meta type.
  - Zero-copy of the relation graph through fan-out. If a `tee` clones a
    frame to two branches, both see the same `AnalyticsMeta`; if branch
    A's overlay mutates a label, does branch B see it? Default:
    `AnalyticsMeta` is shared via `Arc`, mutation is COW.

- **Bus message coverage.** 2 sessions. `BusMessage` carries
  `NegotiationFailed`. GStreamer's bus carries a dozen+ message types:
  state-changed, eos, error, warning, info, tag, async-done,
  segment-done, stream-status, buffering, latency, clock-lost, qos.
  Applications can't react to "buffering 60%" or "decoder dropped a
  frame" because we don't post those events. Low risk, high
  observability lever.

- **Compositor / pixel mixer (`videomixer` / `compositor`).** 3–4
  sessions. Our `mux` is a fan-in *multiplexer* (combining encoded
  tracks into a container). GStreamer's `compositor` overlays multiple
  raw video streams onto one frame at configurable positions / sizes /
  z-order — picture-in-picture, multi-camera grids, sub-window UIs.
  Common production need; we have nothing. Needs a wgpu compute
  pipeline element + a per-input position config.

- **Adaptive streaming demuxers (HLS, DASH).** 2–4 sessions each.
  Playlist parsing + ABR rate selection + per-segment fetch +
  CMAF/fMP4 handoff. The OBS / Twitch / YouTube Live / DASH player
  ecosystem isn't reachable without these. Each is its own
  non-trivial implementation; pick by use case.

- **SRT / RTMP transports.** 2–3 sessions each. RTMP for legacy
  ingest (still ubiquitous), SRT for low-latency contribution. Each
  needs a sans-IO protocol layer + a tokio I/O sink, paralleling the
  RTP packetizer + UDP sink split.

- **RTP receive-side stack.** 3 sessions. We have the egress half
  (`RtpH264Packetizer` + `UdpSink`). Receive-side jitterbuffer with
  packet reordering, RTCP RR generation, NACK-based retransmission,
  FEC, RTX are all missing. `RtspSrc` via `retina` covers the RTSP
  case (retina has its own jitterbuffer), but for raw RTP ingest (the
  broadcasting / video-contribution use case) there is no
  network-resilience story.

- **Property system + introspection.** 3 sessions. No name/value
  property bag — `with_*` builder methods only, set at construction.
  No `gst-launch foo bar=baz` runtime setting, no
  `gst-inspect`-equivalent enumeration of available elements,
  properties, and pad templates. Blocks the `gst-launch` text DSL
  beyond the basics, blocks GUI editors, blocks any tooling that wants
  to introspect a graph.

### Medium / niche

Smaller-scope items, mostly orthogonal to the architecture.

- **URI handlers (`uridecodebin`-equivalent).** 1 session. Map
  `file://` / `http://` / `rtsp://` / `srt://` to the right source
  element. Trivial layer once auto-plug + the relevant source
  elements exist.
- **Tag system.** 1 session. `GstTagList`-equivalent for stream
  metadata (title, encoder, language, artist). Container demuxers
  surface tags; applications consume them via the bus.
- **Audio mixer.** 2 sessions. Fan-in for audio with sample-rate
  conversion + channel layout reconciliation. Pairs with `AudioConvert`.
- **Subtitle support.** 2 sessions. `Caps::Subtitle` variant,
  text/srt/webvtt demuxers, a text-overlay element (tied to
  compositor for the rendering half).
- **Controllers (animated properties).** 2 sessions.
  `gst-controller`-equivalent for animating properties over time
  (zoom 1.0 → 2.0 over 5 seconds). Niche but real for production
  graphics.
- **`gst-launch` text DSL.** 2 sessions. A parser that takes
  `"rtspsrc location=... ! h264parse ! avdec_h264 ! waylandsink"` and
  builds a `Graph`. Trivial once the DAG runner + property system
  exist.

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

- **`videoscale`.** Spatial resampling (1080p → 720p, etc.). 1–2 sessions.
  Software bilinear/bicubic baseline plus a wgpu variant for the GPU. Without
  this we cannot resize video, which blocks multi-resolution output,
  thumbnails, ML preprocessing at non-native resolution, and any device
  whose input doesn't match the source.
- **`videorate`.** Temporal resampling (30 fps → 10 fps for ML, 60 fps → 30 fps
  for delivery). 1 session. Drops or duplicates frames against a target
  framerate; preserves PTS continuity. ML inference at a target rate is the
  driving use case.
- **`videocrop`.** Crop a rectangular region. 1 session. Pairs with the
  metadata system for ROI-driven cropping (detector emits boxes → cropper
  extracts patches → classifier sees each patch).
- **`videoflip` / `videorotate`.** 90° / 180° / 270° / mirror. 1 session.
  Common for portrait-mode mobile sources fed to a landscape pipeline.
- **`videobalance`.** Brightness / contrast / hue / saturation. 1 session.
  Niche; deferrable.

### Audio transforms

- **`audioresample`.** Sample-rate conversion (48 kHz → 16 kHz for ASR, etc.).
  1–2 sessions. Mandatory for cross-rate paths; without it, every audio
  source has to negotiate the consumer's exact rate.
- **`audiomixer`.** Fan-in with sample-rate + channel-layout reconciliation.
  Already in the parity-gaps list above.

### Capture sources

A whole platform-coverage gap: today we have no live camera capture on any
platform. WASAPI covers Windows audio in/out, but video capture and Linux
audio are unaddressed.

- **`v4l2src`** (Linux video capture). 2 sessions. The Linux camera baseline;
  MMAP DMABUF output already maps to our `MemoryDomain::DmaBuf`. Needed for
  any Linux-side production capture.
- **`pipewiresrc`** (Linux PipeWire video + audio, screen capture).
  2–3 sessions. PipeWire is the modern Linux media layer (replacing v4l2 +
  PulseAudio + screen-capture-via-DBUS); a single element covers camera +
  microphone + screen.
- **`mfvideosrc`** (Windows camera via Media Foundation). 2 sessions. The
  video sibling of `WasapiSrc`; Media Foundation Source Reader pattern.
- **`alsasrc` / `pulsesrc`** (Linux audio capture, non-PipeWire). 1 session
  each. Wide host coverage where PipeWire isn't installed.
- **Screen capture.** 2–3 sessions per platform. Linux: PipeWire (via the
  source above). Windows: DXGI Desktop Duplication API. macOS: ScreenCaptureKit.
  OBS / video-conferencing use case.
- **`avfvideosrc` / `avfaudiosrc`** (macOS AVFoundation). Part of the macOS
  platform gap (see below).

### Network sources

- **`UdpSrc` + RTP depayloader.** 2 sessions. We have `UdpSink` egress; the
  receive half (jitterbuffer + depayloader) is missing. Different from
  `RtspSrc` (which is RTSP-over-TCP via retina) — raw RTP-over-UDP ingest is
  the broadcast contribution shape.
- **`souphttpsrc` / `HttpSrc`.** 2 sessions. HTTP / HTTPS source.
  Blocks HLS / DASH / RTMP and any "fetch from a URL" use case. `reqwest` or
  `hyper` as the backing crate. Prerequisite for the adaptive demuxers in
  the parity-gaps list.
- **`rtmpsrc`** (RTMP ingest). Tied to the RTMP transport in parity gaps.
- **`srtsrc`** (SRT ingest). Tied to the SRT transport in parity gaps.

### Sinks

- **Linux audio sinks: `alsasink` / `pulsesink` / `pipewiresink`.** 1 session
  each. We have `WasapiSink` for Windows; Linux audio output has nothing.
- **Generic `GlSink` over EGL.** 2–3 sessions. `CudaGlSink` is NVIDIA-specific;
  `WaylandSink` is software NV12. A vendor-neutral GL ES presentation sink
  over EGL with a generic NV12 / RGBA shader covers Mesa / Intel / AMD
  without CUDA, plus Android (via SurfaceFlinger EGL) once that platform
  exists.
- **`autovideosink` / `autoaudiosink`.** 1 session each, tied to the URI /
  state-machine work. Picks the right sink for the host automatically.

### Containers

- **MKV / WebM (`matroskademux` / `matroskamux` / `webmmux`).** 3 sessions.
  Common delivery format, especially WebM for browser delivery without DRM.
- **MPEG-TS (`mpegtsmux` / `tsdemux`).** 3 sessions. Broadcast carrier; the
  payload format for SRT and a lot of professional ingest.
- **FLV (`flvmux` / `flvdemux`).** 2 sessions. RTMP carrier.
- **OGG (`oggmux` / `oggdemux`).** 1–2 sessions. Niche, mostly Opus delivery.
- **CMAF / fMP4 segmented.** 2 sessions. Already have `Mp4Sink` /
  `Mp4Src` (fragmented); the CMAF-specific signalling for adaptive
  streaming is a thin layer on top.

### Codecs

H.264 / H.265 / AAC are in. The notable gaps are everything WebRTC and the
modern web defaults to:

- **VP8 / VP9 decode + encode.** 3 sessions each. WebRTC default video.
  ffmpeg has both; the wrappers parallel `FfmpegH264Dec`.
- **AV1 decode + encode.** 3 sessions each. Modern WebRTC + streaming;
  `libaom` / `dav1d` for decode, `libaom` / `SVT-AV1` for encode.
- **Opus encode + decode.** 2 sessions. WebRTC audio default; we have AAC
  only. `opus` crate or libopus FFI.
- **MJPEG decode.** 1 session. Low-end IP cameras (a huge installed base)
  only emit MJPEG over RTSP. `mozjpeg` / `image` crates.
- **JPEG decode + encode.** 1 session. Thumbnailing, snapshot capture.

### Parsers

For every codec we host, the bitstream parser (SPS / VPS / sequence-header
extraction, framing detection, framerate / dimension recovery) is what feeds
the negotiation. We have `H264Parse`; everything else is missing.

- **`H265Parse`.** 2 sessions. VPS + SPS + PPS; we already decode and
  contain H.265 but cannot parse a raw H.265 elementary stream into framed
  caps. Means we can't restream / record raw H.265.
- **`AacParse`.** 1 session. ADTS / LATM headers, sample-rate recovery.
- **`Vp8Parse` / `Vp9Parse` / `Av1Parse` / `OpusParse`.** 1–2 sessions each,
  alongside the corresponding codec.

### Overlay / effects

- **`textoverlay` / `clockoverlay` / `timeoverlay`.** 2–3 sessions total.
  Tied to the compositor work in parity gaps (overlays are one input to a
  compositor). `textoverlay` rendering through `cosmic-text` or `swash`.

### Platform: macOS

A whole platform gap. We compile for Linux + Windows + wasm32 + bare-metal;
macOS has zero element coverage. 5–8 sessions for the baseline:

- **`vtdecode` / `vtencode`** — VideoToolbox H.264 / HEVC decode + encode.
- **`avfvideosrc` / `avfaudiosrc`** — AVFoundation camera + microphone.
- **`coreaudiosink` / `coreaudiosrc`** — Core Audio in/out.
- **`metalvideosink`** — Metal presentation.

Each individually is 1–2 sessions; the platform integration (`framework`
linking, `objc2` bindings, macOS-specific feature gates) is the bulk of
the work and only pays off once.

### Platform: Android

Another platform gap. Andoird's `MediaCodec` + `SurfaceTexture` for
hardware decode, `Camera2` for capture, `AAudio` for audio, `Surface` for
presentation. Similar 5–8 session shape to macOS.

### Other

- **`videotestsrc` pattern coverage.** 1 session. We have `VideoTestSrc`
  but with limited patterns. SMPTE bars, snow, ball, gradient, checker,
  zone plate — useful for codec testing.
- **RTSP server (`rtsp-server`).** 4–5 sessions. `RtspSrc` is the client;
  hosting RTSP endpoints (one per pipeline, dynamic client connect) is the
  OBS / surveillance / contribution-server shape.
- **WebRTC sendrecv full-stack.** 5+ sessions. `WebRtcSrc` is data-channel
  ingest only; a complete `WebRtcBin`-equivalent with ICE, DTLS-SRTP, full
  media-engine negotiation is its own track.

### Priority summary

| Element | Sessions | Why it matters |
| :--- | :--- | :--- |
| `videoscale` | 1–2 | resize anything |
| `videorate` | 1 | target-fps for ML / delivery |
| `videocrop` | 1 | ROI-driven; pairs with metadata |
| `audioresample` | 1–2 | any cross-rate audio path |
| `v4l2src` / `pipewiresrc` / `mfvideosrc` | 2 each | live camera on Linux / Windows |
| `UdpSrc` + RTP depay | 2 | raw RTP ingest |
| `HttpSrc` | 2 | prereq for HLS / DASH / random URLs |
| `H265Parse` + `AacParse` | 2 + 1 | restream codecs we already decode |
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

The first four (`videoscale`, `videorate`, `videocrop`, `audioresample`)
are the cheapest, highest-frequency gaps — almost every non-trivial
pipeline hits at least one of them. The capture sources
(`v4l2src` / `pipewiresrc` / `mfvideosrc`) are the next tier: without
live camera input we're a "process incoming streams" framework, not a
"produce streams" framework. Both tiers together are ~12 sessions and
make g2g substantially more credible as a GStreamer replacement.

## Tier-1 element sprint — detailed plan

11–14 sessions. Closes the most embarrassing gaps so g2g passes a 10-minute
developer evaluation: resize, reframe, crop, audio-resample, and live camera
capture on Linux + Windows.

**Status (2026-06):** Phase A transforms `VideoScale` (M55), `VideoRate`
(M56), `VideoCrop` (M62), and `VideoFlip` (M66) shipped (native + wasm32).
`AudioResample` (A4) and the Phase B capture sources remain.

Goal at end of sprint: a self-contained demo with no external feed —

```
v4l2src (camera) → videoscale → videorate → wgpupreprocess →
  ortinfer (CUDA EP) → tensorpostprocess → waylandsink
```

— that exercises the full ML video stack from local hardware capture
through inference to display. Same shape on Windows with `mfvideosrc → ... →
d3d11sink`. This is the "yes you can build a real thing" baseline.

### Phase A — Software transforms (5 sessions)

All four transforms are software, `System` memory in / out, fully unit-
testable against synthetic inputs (no hardware in the loop). Ordered
cheapest first so the sprint shows progress quickly.

**A1 — `VideoCrop` (1 session).** Simplest of the four — rectangular slice,
no resampling. `DerivedOutput` constraint keyed on a configured crop rect:
`with_rect(x, y, w, h)`. Per-plane copy honouring source pitch
(I420: Y full-res, U/V half-res cropped to even coords; NV12: Y + interleaved
UV). Output caps carry `(w, h)` from the rect, framerate / format
unchanged.

- *Verify:* feed a `VideoTestSrc` checker pattern, assert the cropped
  output's checker cells are at the expected positions; reject odd
  crop coords on 4:2:0 with loud `CapsMismatch`; constraint narrows
  RawVideo / rejects CompressedVideo.
- *Why first:* zero algorithmic risk, immediate ML value (ROI patches
  feed `WgpuPreprocess` at a fixed input size).

**A2 — `VideoRate` (1 session).** Temporal-only — no pixel math. Stateful
on last-emitted PTS and the target inter-frame interval. On each input
frame: compute the number of output frames whose deadlines fall on or
before the input's PTS, emit duplicates / drop excess accordingly. Caps
preserve `format`/`width`/`height`; framerate replaced with the configured
target. `with_target_fps(f64)`.

- *Verify:* feed a 30 fps `VideoTestSrc`, target 10 fps, assert 1-in-3
  pass-through with correct PTS spacing; target 60 fps, assert 1→2
  duplication with monotonic PTS; PTS-wraparound edge case.
- *Why second:* trivial state machine, the "convert 30 fps live to 10 fps
  for ML" recipe is one of the most common requests.

**A3 — `VideoScale` (2 sessions).** The substantive one. Software bilinear
baseline, separable (horizontal pass then vertical) for cache friendliness.
Per-plane resample on I420 / NV12 with chroma at correct half-resolution.
`DerivedOutput` keyed on `with_target_dims(w, h)`.

- *Why bilinear, not bicubic or Lanczos:* baseline correctness over peak
  quality; production users who care about quality pair this with a wgpu
  variant later. Bilinear is < 200 lines per plane.
- *Verify:* round-trip 1080p → 540p → 1080p, assert PSNR > 30 dB against
  the source (sanity check, not a quality bar); 16-pixel-aligned input
  and output dims, then non-aligned; reject 0/odd target dims on 4:2:0.
- *Follow-up out of scope:* a `WgpuVideoScale` companion that takes a
  GPU-resident input (DMABUF / D3D11Texture) and runs the resample in a
  compute shader. Lands later, slots in via the existing surface-import
  story.

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

**B2 — MJPEG decode (1 session).** Unblocks the half of webcams that
only emit MJPEG. Wrap `image` crate or `mozjpeg` for SW JPEG decode in a
new `MjpegDec` element; `Caps::CompressedVideo { codec: Mjpeg }` →
`Caps::RawVideo { format: I420 | Rgba8 }`. Lands here because `V4l2Src`'s
default on every consumer webcam is MJPEG, and without `MjpegDec` the
Phase B demo doesn't compose.

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
  no system deps, MIT licensed.
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

### Phase 1 — Minimum SW transforms (3 sessions)

Pulled from Tier-1 Phase A. Only the two transforms the demo critically
needs; `VideoCrop` / `AudioResample` defer until a use case forces them.

**Status: done.** `VideoScale` (M55) and `VideoRate` (M56) shipped, both
verified on native + wasm32; `VideoCrop` also landed (M62).

- **P1.1 — `VideoScale`** (2 sessions). Software bilinear, separable, per-
  plane on I420 / NV12. The ML model wants a fixed input size
  (e.g. 256×256 for a small segmentation network) and the camera /
  stream geometry won't match. *Verify:* round-trip PSNR > 30 dB, reject
  odd target dims on 4:2:0, runs in both native and wasm32 builds.
- **P1.2 — `VideoRate`** (1 session). Target-fps drop / duplicate. The ML
  network runs at a fixed rate (10–15 fps for browser ML is typical);
  source rate won't match. *Verify:* monotonic PTS, correct 1-in-3 drop
  / 1-into-2 duplicate, PTS-wraparound edge.

Both elements ship as pure `g2g-plugins` `AsyncElement` impls in the
existing CSP shape. **Verification gate before P2:** native +
`wasm32-unknown-unknown` clean compile, unit tests green on both targets.

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
- **Compositor, RTP receive jitterbuffer, bus message coverage,
  metadata system, every codec beyond H.264.** Wait.
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

## DAG runner — detailed plan

Plan, not implementation. Locks the shape of the `Graph` API, the generalized
solver, and the runner-composition strategy before code lands. Phased so each
landing is independently verifiable.

### Goal

A single entry point

```rust
run_graph(graph: Graph, clock: &Clk, profile: LatencyProfile) -> Result<RunStats, _>
```

that drives an arbitrary multimedia DAG: linear + fan-out + fan-in + nested
branches in one topology, with whole-graph CSP negotiation, per-edge
allocation cascade, mid-stream re-solve, and structured failure on bus.

**Non-goals:**

- A `gst-launch` text DSL. Orthogonal; trivially layered on top once the
  `Graph` builder exists.
- Element hot-swap under load (already a separate open item).
- Dynamic pad add/remove at runtime (`tee::request-pad` shape). Phase 5 /
  future; the static graph case is the bulk of the value.
- Cycles. Multimedia DAGs are acyclic in practice; the rare feedback case
  (ML re-entry) goes through a `LinkInterceptor` probe, not a graph edge.
  Validation rejects cycles loud.

### Proposed surface

**Builder.**

```rust
let mut g = Graph::new();

let src    = g.add_source(Box::new(RtspSrc::new(url)));
let parse  = g.add_transform(Box::new(H264Parse::new()));
let tee    = g.add_tee(2);                                   // 1 in, 2 out
let dec    = g.add_transform(Box::new(FfmpegH264Dec::new()));
let display= g.add_sink(Box::new(WaylandSink::new()));
let mux    = g.add_muxer(Box::new(Mp4Sink::open("out.mp4")?));

g.link(src,        parse)?;                                  // 1->1
g.link(parse,      tee.input())?;
g.link(tee.out(0), dec)?;
g.link(dec,        display)?;
g.link(tee.out(1), mux.input(0))?;                           // recorded encoded path

let stats = run_graph(g, &clock, LatencyProfile::Live).await?;
```

The same shape with bus integration is `run_graph_with_bus(g, &clock,
profile, &bus)`.

**Node and pad handles.**

```rust
pub struct NodeId(u32);            // opaque index into the graph
pub struct PadId  { node: NodeId, index: u8 }

impl Tee     { pub fn input(&self) -> PadId; pub fn out(&self, i: u8) -> PadId; }
impl Muxer   { pub fn input(&self, i: u8) -> PadId; pub fn output(&self) -> PadId; }
impl Source  { /* implicit single output pad; `g.link(src, ...)` */ }
impl Sink    { /* implicit single input pad */ }
impl Transform { /* implicit 1-in-1-out */ }
```

Element kind is captured at insertion time (`add_source` / `add_transform` /
`add_sink` / `add_tee` / `add_muxer`) so the builder knows the pad shape
without runtime introspection. Most call sites stay in the 1-in-1-out
shorthand `g.link(a, b)`.

**Per-edge link policy.**

```rust
g.link_with(parse, tee.input(), LinkPolicy::Block)?;
g.link_with(tee.out(1), mux.input(0), LinkPolicy::DropOldest)?;
```

Default policy is read from the `LatencyProfile`. Explicit per-edge policies
override.

**Validation.** `Graph::finish() -> Result<ValidatedGraph, GraphError>`
(called by `run_graph` internally) checks:

- exactly one source node has no incoming edge (or: every source node has
  no incoming edge, allowing N sources for a muxer);
- exactly one sink node has no outgoing edge (or: every leaf is a sink);
- every transform / mux / tee pad is linked;
- no cycles (DFS / Kahn);
- shape matches the declared pad count (e.g. `Tee(2)` has exactly two
  outgoing edges).

Returns structured `GraphError { UnlinkedPad, Cycle { nodes }, OrphanNode,
PadCountMismatch }`.

### Internal representation

Hand-rolled, no `petgraph` dep (consistent with the cudarc decision — only
pull a dep when the surface grows past a handful of operations, and
`petgraph` would force `std` while a portion of the validation logic should
stay `no_std`).

```rust
pub struct Graph {
    nodes: Vec<Node>,              // Slab of Box<dyn DynAsyncElement> / DynSourceLoop
    edges: Vec<Edge>,              // Vec<(PadId src, PadId dst, LinkPolicy)>
    kinds: Vec<NodeKind>,          // Source | Transform | Sink | Tee(n) | Muxer(n)
}

pub struct ValidatedGraph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    kinds: Vec<NodeKind>,
    topo:  Vec<NodeId>,            // topological sort, computed at validate time
    in_edges:  Vec<Vec<EdgeId>>,   // index by NodeId
    out_edges: Vec<Vec<EdgeId>>,   // index by NodeId
}
```

`Node` holds the boxed element behind an `Option<Box<dyn DynAsyncElement>>`
so the runner can `take()` it into the spawned arm and the original `Graph`
becomes empty after `run_graph`. No `Arc<Mutex<>>` per element.

### Solver generalization (`solve_graph`)

The existing linear arc-consistency solver lifts cleanly:

1. **Topological order** comes from validation (Kahn's algorithm; rejects
   cycles).
2. **Forward sweep** in topo order: for each node, intersect its
   `caps_constraint_as_*` against the inbound edge sets (multiple inbound
   edges for a muxer are independently constrained — each input pad has its
   own `caps_constraint_as_input(idx)`). Output candidate sets per outbound
   pad.
3. **Backward sweep** in reverse topo order: propagate narrowing back from
   sinks; per-output `Mapping` / `DerivedOutput` re-derived against the
   narrowed input.
4. **Iterate** to fixed point. Linear graphs converge in one round; DAGs
   with diamond shapes (tee → ... → mux) may take a second round.
5. **Fixate** each edge to its highest-preference concrete `Caps`.
6. **`configure_pipeline`** every node with its per-pad caps.

Structured failure stays the same `NegotiationFailure` enum;
`EmptyLink { upstream, downstream, missed }` already carries the two element
ids.

`downstream_feasibility` generalizes from a backward fold over a chain to a
reverse-topo fold over a DAG (a node's feasibility is the intersection over
each outbound edge's downstream feasibility). The mid-stream re-solve
mechanism (DESIGN.md §4.13.4) ports without further changes — each arm still
gets its snapshot, the only difference is the snapshot was computed over a
DAG instead of a chain.

### Runner orchestration

`run_graph` builds N tasks (one per element) joined under `join_all`:

```
For each node n in topo order:
    Build inbound channels = for each in_edge(n): the receiver end created by upstream
    Build outbound channels = for each out_edge(n): a new bounded mpsc<PipelinePacket>
    Spawn arm(n):
        match kind(n) {
            Source     => DynSourceLoop::run(out_sender)
            Transform  => for each pkt in in_recv: DynAsyncElement::process(pkt, out_sender)
            Sink       => for each pkt in in_recv: DynAsyncElement::process(pkt, null_sink)
            Tee(_)     => for each pkt in in_recv: clone to each out_sender   (LinkPolicy per out)
            Muxer(_)   => select across in_recvs; DynAsyncElement::process(pkt_with_pad_idx)
        }
```

Plus the coordinator task (already exists, DESIGN.md §4.13.5) wired to every
arm via the per-arm control channel. Allocation re-cascade and mid-stream
caps re-solve generalize from "walk up the chain" to "walk up the DAG via
`in_edges[n]`", with concurrent walks on diamonds joined at the meet point.

Tee and Muxer become first-class graph nodes rather than runner-shape
selectors. The existing `mux` and per-branch fan-out logic is reused behind
these node kinds.

### Phasing

Five focused phases. Each is independently verifiable.

| Phase | Scope | Verifiable on its own |
| :--- | :--- | :--- |
| **D1 — `Graph` data structure + validation.** DONE (M63). | The builder, `NodeId` / `PadId`, `LinkPolicy` per edge, `finish() -> ValidatedGraph` with topo sort + cycle detection + pad-count + orphan checks. No solver, no runner. Generic over the element payload `E` to stay `no_std`. | Pure data-structure tests. Reject diamond-with-cycle, accept tee→mux diamond, accept linear, accept fan-out, accept fan-in. |
| **D2 — `solve_graph` (topological CSP).** DONE (M64 fan-out, M65 muxer fan-in). | Generalize `solve_linear`'s forward + backward sweep to topo order over edges. Per-node `NodeConstraint` (single `CapsConstraint` for source/transform/sink; `Muxer { inputs, output }` per-input-pad for fan-in). `NegotiationFailure` unchanged. Reverse-topo `downstream_feasibility` fold not yet ported (only needed by D4). | Solver-only tests against fake elements: muxer fan-in narrows each input by its pad, tee fan-out couples branches, a rejecting branch strict-fails, linear regression matches `solve_linear` byte-for-byte. |
| **D3 — `run_graph` (the runner).** DONE (M67), source/transform/sink/tee. | Spawn-per-node, edge-channels, `join_all`. Element ownership is a `GraphNode { Source(Box<dyn DynSourceLoop>) | Element(Box<dyn DynAsyncElement>) }` payload (sources and transforms/sinks are different traits). The coordinator / mid-stream re-cascade is D4, so an interior arm handles a `CapsChanged` locally (no β walk). Two gaps surfaced: a tee can't share a `PipelinePacket` (not `Clone`, GPU handles), so it deep-copies `System` frames and fails loud on a GPU domain (zero-copy tee needs a refcounted shareable frame); muxer nodes are rejected, still owed the per-input-pad constraint API. | DONE: pure-fake DAG tests (linear chain, `tee(2)` fan-out to two sinks, tee with per-branch transforms, incompatible-branch solve failure). OWED: the `rtspsrc → parse → tee → {dec → wayland, mux → mp4}` hardware integration test (gated on `rtsp ffmpeg wayland-sink`), a Linux run. |
| **D4 — Mid-stream re-solve + β cascade over the DAG.** | Snapshot feasibility into each arm at startup. On mid-stream `CapsChanged`, walk the affected subgraph via topo + `in_edges`. Per-branch concurrent walks on a tee, per-input independent walks on a mux. | Fake-element regression: change source caps mid-stream, assert every downstream branch re-solves correctly, the rejecting branch fails its arm, the rest keep flowing. |
| **D5 — Convenience wrappers + deprecation path.** | `run_linear_chain` / `run_source_fanout` / `run_muxer_sink` / `run_source_transform_sink` become thin builders over `Graph` + `run_graph`. Public signatures stay the same; they construct the corresponding `Graph` internally. | Existing integration tests pass without modification. |

D1 + D2 are pure `no_std` (the `Graph` and solver are computation, no I/O).
D3 onwards is `std` because the runner is `tokio` / `flume`-coupled.

Per-phase sizing: D1 ≈ 1, D2 ≈ 1, D3 ≈ 2, D4 ≈ 1, D5 ≈ 1. Total 5–6
sessions, comparable to M16's scope.

### Load-bearing decisions for sign-off

**G1 — Element ownership: `Box<dyn DynAsyncElement>`, not `&mut`.** The
current runners take `&mut Tx` because the chain is statically typed. An
arbitrary DAG cannot be statically typed (heterogeneous element types at
every node), so element ownership moves into the graph at
`add_*(Box::new(...))` time. Cost: one allocation per element, no dynamic
dispatch beyond the existing `DynAsyncElement` trait. Worth confirming
before D1 lands because every downstream API shape depends on it.

**G2 — `NodeKind` discriminator vs trait-only.** Carry `NodeKind::{Source,
Transform, Sink, Tee(n), Muxer(n)}` alongside the boxed element so the
runner knows how to spawn each arm (Tee fans out, Muxer fans in, etc.).
Alternative: "everything is `DynAsyncElement`; the runner inspects
`caps_constraint`" — but Tee and Muxer need topology semantics
(`process(pkt)` to multiple outputs vs select-across-inputs) that the caps
constraint doesn't describe. Decision: explicit `NodeKind`.

**G3 — Cycles: reject loud at validation.** Trying to support cycles would
force a backedge concept and break topo-sort fixation. Out of scope. The
single real use case (feedback ML) goes through a `LinkInterceptor` probe,
which doesn't show up as a graph edge.

**G4 — Tee / Muxer as graph nodes vs runner shapes.** Currently the runner
shape picks (fan-out vs fan-in vs linear); under DAG they become node kinds.
The existing `mux` element and per-branch fan-out logic move behind
`NodeKind::Muxer` and `NodeKind::Tee` adapters. No new behavior; just a
refactor where the shape lives.

**G5 — Backward compatibility.** D5 reframes the existing runners as thin
wrappers that build a `Graph` + call `run_graph`. Public signatures and
behavior stay identical, including the static-typed
`run_source_transform_sink` which keeps its generic `Src, Tx, Snk`
parameters and boxes them internally. Allows incremental adoption: callers
keep their existing wiring and new pipelines use `Graph`.

**G6 — `no_std` boundary.** D1 + D2 stay `no_std` so the graph builder and
the solver are usable from the embedded / wasm sides too. D3 onwards is
`std`-gated under the existing `multi-thread` shape. Same boundary as the
rest of the runner.

### Open questions

- **`add_tee(n)` n-ary vs `add_tee()` + repeated `connect` calls.**
  GStreamer's `tee` grows pads on demand. Phase 5 lifts the `n` parameter to
  a runtime `Tee::add_branch()`; for now `n` is fixed at construction.
- **Allocation policy across diamonds.** When two branches downstream of a
  tee both propose different allocation params, the tee's outbound pool
  needs a join policy. Default: most-restrictive intersection; loud failure
  if intersection is empty. Phase 4 detail; spec it alongside `run_graph`
  mid-stream.
- **`Graph` clone / re-run.** After `run_graph` consumes the elements via
  `take()`, the `Graph` is empty. For seek-and-replay scenarios, a
  `GraphTemplate::instantiate() -> Graph` two-step is cleaner than making
  `Graph` itself reusable. Defer to a follow-up.

### Decisions needed before D1 starts

- Sign off **G1** (element ownership: `Box<dyn>`).
- Sign off **G2** (explicit `NodeKind`).
- Sign off **G5** (backward-compatibility strategy: thin wrappers in D5).

The rest can be iterated on as each phase lands.

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
