# DESIGN_TODO

Outstanding work, tracked against the architecture in [DESIGN.md](DESIGN.md).
This file is a terse catalogue of open tasks only. Completed work and the
rationale for shipped architecture live in [DESIGN.md](DESIGN.md) and
[CHANGELOG.md](CHANGELOG.md), not here.

## Roadmap (high level)

The core runtime, CSP caps negotiation (including the N-hop allocation
re-cascade), and the full lifecycle spine (state machine + preroll, seek +
SEGMENT, auto-plug / decodebin / playbin) are done. What remains, highest
leverage first:

1. **Platforms** (largest track). macOS: AVFoundation capture, Core Audio,
   Metal present, plus on-device validation of `VtDecode` / `VtEncode`.
   Android: encode, Camera2, AAudio, Surface present, plus on-device
   validation of `MediaCodecDec`.
2. **Egress / transports.** SRT TSBPD + congestion control + real-peer interop,
   AES-256 + key rotation; FlexFEC + multi-level burst FEC.
3. **Depth.** Pure-Rust / wasm codec decode (dav1d/rav1d, vpx) to drop the
   ffmpeg FFI; negotiation backward coupling through `DerivedOutput`; seek depth
   (segment seeks, re-preroll after flushing seek when paused).
4. **GPU keep-on-GPU.** CUDA <-> wgpu interop to join NVDEC decode to wgpu
   inference / preprocess.
5. **Bindings polish.** Blocking-with-timeout `appsink` pull; Python `appsink`
   zero-copy via the buffer protocol (memoryview, not a bytes copy); maturin
   wheel for `g2g-pyapi`; an in-tree C / Python example program.
6. **Browser demo (speculative product path).** Cross-target ONNX in-browser:
   CPU-round-trip MVP via `ort-web` (`WebSocketSrc -> WebCodecsDecode ->
   ort-web -> CanvasSink`), then a deployed reference app + native sibling. The
   GPU-resident in-browser chain is not achievable from idiomatic Rust (wgpu
   can't import a WebCodecs `VideoFrame` as an external texture or adopt ORT's
   device on wasm); it would need raw `web_sys` WebGPU + hand-rolled
   onnxruntime-web bindings.

## Negotiation

- **Backward coupling through a format-changing transform.**
  `backward_feasible()` returns `None` for `DerivedOutput`, so a downstream pin
  behind a decoder / convert-that-rescales fails loud (`CapsMismatch`).
  Generalize `DerivedCoupled`'s field-level inverse to the invertible fields of
  a `DerivedOutput`. (Linear passthrough coupling is done.)
- **Forward coordinator re-solve walk (Caps-β).** Mid-stream re-solve uses a
  startup downstream-feasibility snapshot; a downstream `DerivedOutput` that
  must re-derive on a mid-stream input change isn't covered. Design settled,
  gated on a real driver (a decoder below another format-changing transform).
- **Closure-free `FieldTransform` refactor.** Make forward derivation
  declarative too, for a `Debug`/`Copy` single-source-of-truth descriptor.
- **Dynamic pads / request pads.** `tee` / `mux` runtime branch/input addition;
  both are static today.
- **Zero-copy tee.** `run_graph`'s tee deep-copies `System` frames and fails on
  a GPU domain (`PipelinePacket` isn't `Clone`). Needs a refcounted shareable
  frame.
- **Graceful per-branch drop on fan-out** (`FanOutPolicy::AllowBranchDrop`); a
  rejecting branch fails the run loud today.
- **β allocation re-cascade across a muxer.** A muxer's inputs have no per-pad
  re-cascade channel, so the DAG β walk terminates at a muxer.
- **Timestamp-ordered fan-in.** `muxer_arm` drains per-input channels in arrival
  order, not PTS order. A PTS-ordered merge is the prerequisite for multi-camera
  grids, A/V interleave, and PTS-synchronized compositing. Gated on a
  frame-accurate-sync use case.
- **Allocation join policy across diamonds.** Two branches downstream of a tee
  proposing different allocation params need a join policy (sketch:
  most-restrictive intersection, loud failure on empty).
- **`Graph` re-run / clone for seek-and-replay.** `run_graph` consumes elements
  via `take()`; a `GraphTemplate::instantiate() -> Graph` two-step is cleaner
  than making `Graph` reusable.
- **Mid-stream element hot-swap.** `ElementSlot::swap` scaffolding exists; live
  swap of a real element under load isn't wired.
- **Preference algebra.** `CapsPreferences` is a placeholder (sum-of-indices);
  needs a real competing-constraint scenario to drive it.
- **Hardware `tee -> {decode, mux}` integration test** on real Linux
  (`rtsp ffmpeg wayland-sink`); only fake-element coverage today.

## Seek and auto-plug

- Non-flushing / accumulating `do_seek` (advance base by elapsed running time).
- Segment seeks (CMAF / DASH transitions).
- Re-preroll after a flushing seek when paused.
- Make `FileSrc` and the demuxers seek-aware (only `Mp4Src` is today).
- Richer auto-plug factory construction params (geometry / device / file path).
- A hardware-backed end-to-end decode-through-`decodebin` run (current tests
  read templates / assert splicing, decode no real media).

## Platform: macOS

- `VtDecode`: first `cargo build` on a Mac to settle the FFI `// NOTE` spots;
  HEVC; a `CVPixelBuffer` / `IOSurface` zero-copy domain; registry wiring
  (`avdec_h264` alias); on-device runtime validation.
- `VtEncode`: HEVC; on-device runtime validation.
- `avfvideosrc` / `avfaudiosrc` (AVFoundation camera + mic).
- `coreaudiosink` / `coreaudiosrc`.
- `metalvideosink` (Metal present).
- Screen capture (ScreenCaptureKit).

## Platform: Android

- `MediaCodecDec`: on-device runtime validation; output color-format beyond
  semi-planar / planar (`COLOR_FormatYUV420Flexible` via `AImageReader`); HEVC;
  an `AHardwareBuffer` zero-copy domain; the `Surface` present sink.
- Encode, Camera2 capture, AAudio, Surface present.

## Receive / decode

- **`VaapiH264Dec` on AMD** (cros-codecs path). Hard-codes ChromeOS GBM flags
  that fail on Mesa `radeonsi`; the clean fix is an upstream libva
  (`vaCreateSurfaces`) surface backend. The ffmpeg `Backend::Vaapi` hwaccel path
  is the working AMD / Intel decode route in the meantime (validated on a
  Rembrandt 680M); this item is only for reviving the pure cros-codecs backend.
- Zero-copy `MemoryDomain::DmaBuf` from `VaapiH264Dec` (needs a surface-keepalive
  refcount).
- H.265 in `VaapiH264Dec` (sibling element on `VideoCodec::H265`).
- Upstream `Reconfigure` driven by `VaapiH264Dec` `FormatChanged`.
- `MfDecode` zero-copy + DXVA (`D3D11Texture`, `MF_SA_D3D11_AWARE`, strided
  NV12; SW path assumes `stride == width`).
- 10-bit pixel formats in `FfmpegH264Dec` (`YUV420P10` / `P010`).

## CUDA / display

- `CudaGlSink`: first compile on Linux+NVIDIA + the `wayland_smoke`-style e2e
  benchmark vs the `NvdecCuvid -> WaylandSink` baseline (authored off-Linux).
- GL-on-KMS variant of `CudaGlSink` (production tty path).
- CUDA <-> Vulkan external memory (`cudaImportExternalMemory`).
- A real downstream consumer that re-sizes its pool on the mid-stream β
  allocation proposal (decoders record it but pools are fixed at codec open).

## Egress / transports

- **SRT:** TSBPD timing, congestion control, AES-256 + key rotation,
  libsrt/ffmpeg real-peer interop.
- **RTMP:** the HMAC-digest handshake some CDNs require, multiple streams,
  server-acknowledgement back-pressure.
- **RTP FEC:** FlexFEC (RFC 8627); multi-level / interleaved ULPFEC for burst
  loss (single-level recovers one loss per group).
- **RTCP sender reports** (RFC 3550 SR) on `UdpSink`.
- **RTSP server:** TCP-interleaved transport; RTCP / keepalive during PLAY.
- **`UdpSrc` SDP/SPS-driven caps discovery** (reports a declared hint today).
- **WebRTC.** `WebRtcSink` (WHIP egress, `webrtc` feature) publishes H.264 *or*
  Opus to a WHIP server over a `str0m` PeerConnection (ICE / DTLS / SRTP,
  pure-Rust crypto); compile-validated against str0m 0.20, with a WHEP player +
  ignored `webrtc_whip_smoke` harness for on-network validation. Remaining:
  on-network validation against a real WHIP server (mediamtx) + browser
  playback; simultaneous A/V over one PeerConnection (a `MultiInputElement`,
  not one-track-per-sink); non-stereo / non-48 kHz Opus; WHIP `DELETE` +
  graceful flush on EOS; keyframe-request (PLI) handling; a `WebRtcSrc` / WHEP
  ingest sibling (also on str0m). The browser data-channel `WebRtcSrc` stays
  wasm-only. A full `WebRtcBin`-equivalent sendrecv media engine is the larger
  track this seeds.

## Adaptive streaming (HLS / DASH)

- **HLS:** SAMPLE-AES key rotation mid-stream; cbcs audio (AAC) + per-sample IV
  (cenc/cbc1); `saiz`/`saio` aux-info + `seig` sample groups; encrypted fMP4
  init segments; byte-range segments; throughput-driven ABR; live-edge start;
  mid-stream variant switching.
- **DASH:** wall-clock `@duration` live profile; `SegmentList` / `SegmentBase`
  byte-range; multi-period; throughput-driven ABR.

## Capture sources

- `v4l2src`: MMAP DMABUF output (`MemoryDomain::DmaBuf`); format-flexible
  negotiation (MJPEG-mode UVC, other fourccs) vs fixed YUYV.
- `pipewiresrc`: video + screen capture (SPA video pod + `param_changed`);
  DMABUF output.
- `mfvideosrc`: first Windows build + camera smoke test; D3D11 zero-copy;
  size/rate request beyond device default.
- `alsasrc` / `pulsesrc` (Linux audio capture, non-PipeWire).
- Screen capture: Windows DXGI Desktop Duplication.

## Sinks

- Linux audio sinks (`alsasink` / `pulsesink` / `pipewiresink`): host smoke test
  on a real device; channel-count / sample-format reconciliation beyond stereo
  S16/F32; DMABUF / zero-copy.
- Generic `GlSink` over EGL (vendor-neutral NV12 / RGBA present, no CUDA).
- `autovideosink` / `autoaudiosink`.

## Containers

- **MKV / WebM:** Cues / seeking; multi-track muxing; `Targets`-scoped
  (per-track) tags.
- **MPEG-TS:** multi-stream / multi-program muxing + selection; PCR-based timing.
- **FLV:** codec-config / extradata plumbing; multi-track muxing.
- **OGG:** granule-position timing; Vorbis output; multi-stream; `oggmux`.
- **CMAF / fMP4:** the CMAF-specific signalling layer on `Mp4Sink` / `Mp4Src`.

## Codecs

- **VP8 / VP9 encode** (`VpxEnc`): validate on a libvpx host (compile-unverified).
- **AV1 encode** (`Av1Enc`): bitrate / quantizer rate control; 10-bit / 4:4:4.
- **Pure-Rust / wasm decode** to drop the ffmpeg FFI: `Av1Dec` (dav1d / rav1d),
  VP8 / VP9 decode, a pure-Rust Opus path.
- **Opus:** float (F32) PCM in/out; other frame durations; packet-loss
  concealment; bitrate / complexity tuning.
- **MJPEG / JPEG:** a `mozjpeg` fast path under a feature flag; a direct
  YCbCr -> I420 path (skip the RGBA intermediate); a single-still image sink.

## Parsers

- `H265Parse`: framerate from VUI `timing_info`; validate against a real H.265
  elementary stream.
- `AacParse`: LATM / LOAS framing; AudioSpecificConfig synthesis; validate
  against a real ADTS stream.
- `OpusParse`: multichannel (family 1, count in `OpusHead`).

## Transforms and effects

- **`videobalance`:** hue (faithful chroma rotation needs `sin`/`cos`, a `libm`
  dep the `no_std` baseline avoids).
- **`textoverlay`:** a mixed-case TrueType GPU backend (`cosmic-text` / `swash`
  / `vello`); the `clockoverlay` / `timeoverlay` siblings.
- **`audiomixer`:** sample-rate + channel-layout reconciliation; PTS-based
  alignment.
- **`videotestsrc`:** a sinusoidal (vs square-wave) zone plate (needs `libm`).
- **Subtitle support:** `Caps::Subtitle`; text/srt/webvtt demuxers; a
  text-overlay element (rendering tied to the compositor).
- **Controllers (animated properties):** a `gst-controller`-equivalent for
  animating properties over time.
- **Tensor substrate orientation descriptor (M181).** A deferred
  rotate/mirror descriptor the sink can absorb in hardware (DRM/KMS, Wayland
  `set_buffer_transform`, VAAPI VPP, D3D11 VideoProcessor), with eager strided /
  CPU realization as the fallback. Pieces: descriptor on the frame; sink
  capability advertisement; `VideoFlip` branching; one sink (KMS / Wayland)
  wired. (Eager strided views defeat hardware flip silicon.)

## Compositor

- A wgpu compute variant for HD / many-input scale.
- NV12 / I420 mixing without a round-trip through RGBA.
- Configurable output cadence.

## Metadata (FrameMeta / AnalyticsMeta)

- A `Segmentation` node (mask handle); more standard metas (`GstVideoMeta`-style
  strides, ROI).
- `push` vs `pull` propagation across transforms (today push-only, explicit).
- A turnkey windowed runner for `WgpuSink` (a winit/SCTK example that opens a
  window and drives the overlay -> sink graph; validate on a real display).
- A blob header registry (decode known `BlobMeta` headers into typed structures).

## Clock-synchronised presentation

- **KMS vblank reconciliation** + Wayland frame-callback co-scheduling. Needs a
  DRM/KMS presentation sink (current `WaylandSink` is SHM software). Validate on
  a real display.
- **A/V clock slaving** (elect an audio device clock as master). Needs an audio
  sink that provides a clock tracking buffer consumption; drift correction is
  the hard part. Validate against a real audio device.
- **A QoS-aware transform** that acts on a relayed report (a decoder dropping
  non-reference frames) rather than only forwarding it. CI-testable; gated on a
  decoder that can cheaply drop frames being the bottleneck.

## Bus and logging

- Remaining bus messages, each gated on a subsystem not present: `segment-done`
  (segment seeks), `stream-status` (thread pool), `clock-lost` (clock
  re-election). Plus buffering on interior links; periodic QoS; the QoS
  late-drop / `Qos` post from the display sinks.
- Logging: instance naming + lifecycle logging in the bespoke linear runners and
  the muxer path (not just `run_graph`); `set_instance_name` self-logging on more
  elements; explicit names from `gst-launch` `name=`; glob category matching
  (`*sink*:5`); a structured-fields / timestamped record format + ring-buffer
  sink; a custom (non-type-name) category override per element.

## Properties / introspection / DSL

- Carry metadata + properties on muxers (their inspect path builds no instance).
- Property-set the feature-gated sources from text (`location=` / `uri=` on
  rtsp/http/hls/dash/v4l2, default placeholders today).
- A value grammar for spaces / enums-as-named-flags.
- A GUI / tooling introspection surface beyond the text dump.

## Tag system

- Matroska `Targets`-scoped (per-track) tags + nested SimpleTags.
- MP4 freeform (`----`) and integer atoms (track / disc number).
- A per-stream tag merge policy for multi-stream containers.

## Python-element host (M198+)

- **GPU zero-copy (Step 4f, designed, not implemented).** Hand a GPU-resident
  frame to Python without the PCIe round-trip via `__cuda_array_interface__`
  (CAI v3): two CAI objects for the NV12 luma / chroma planes, a
  `g2g_process_cuda(luma, chroma, w, h, meta)` contract over `g2g.CudaPlane`
  pyclasses. Document the CUDA-context caveat (CAI carries none). DLPack is the
  cross-framework alternative. Verify on the RTX 3060 host (install cupy/torch,
  assert a `cupy` array aliases the decoder's device pointer, no copy) before
  presenting the layout as working.
- Verify GIL offload on a free-threaded (PEP 703) interpreter (none installed)
  + a `link_capacity` note for the GIL-serialized case.
- A `backend/g2g/` package in gst-python-ml mirroring `backend/gst/`.

## Aggregation helper adoption (M199+)

- Migrate the four hand-rolled per-input collectors onto
  `g2g-core::InputAggregator<T>`: enterprise `batcher` (closest fit), then `mux`,
  `audiomixer`, and `compositor` (compositor needs a second latest-wins
  `SyncPolicy` variant first). Behaviour-preserving, each guarded by existing
  tests.

## Dynamic plugin loading (M201+)

- An `abi_stable` / `stabby` facade over the element traits for cross-toolchain
  binary plugins (the v1 path is version + toolchain-locked).
- Whether the distro ships `g2g-core` in a local cargo registry for offline
  plugin builds.
- Plugin signing / capability gating.
- A C-FFI loader entry so non-cargo build systems can produce plugins.

## Embedded

- `EmbassyClock` HAL tick on real hardware (host verification via `block_on` is
  in place).
- Full `embassy-executor` multi-task integration (pipelines run under
  `block_on` today).
- A fixed DMA-ring capture `SourceLoop` (no-alloc end-to-end, a
  lifetime-carrying `SystemSlice` wiring `StaticBufferPool` into the zero-copy
  path).

## Browser / Wasm

- In-browser runtime validation of `WebSocketSrc -> WebCodecsDecode ->
  CanvasSink` (compiles; live WebSocket + `performance.now()` pacing unvalidated).
- WebGPU-texture zero-copy sink (`MemoryDomain::WebGPUBuffer` into a
  `GPUTexture`; needs the async device handshake in the keepalive).
- Web Workers executor (off-main-thread; needs JS bootstrap).
- HEVC in `WebCodecsDecode`.
- Browser MVP via `ort-web` (CPU tensors, same `.onnx` as native, plain static
  HTTPS, no COOP/COEP).
- Raw-`web_sys` WebGPU path (only if the GPU-resident browser claim is revived):
  external-texture import + compute + `ort.Tensor.fromGpuBuffer` on one
  ORT-owned `GPUDevice`. Large, browser-unverifiable on the dev host.

## ML

- ONNX import via `burn-import` (build-time codegen).
- A trained-weight `Module` path for `BurnInference` (conv, attention) once the
  codegen lands.
- Decoder DMA-BUF / D3D11 surface import into `WgpuPreprocess` (binds the
  surface directly into the compute pass; needs the surface-import handshake + a
  GPU tensor output domain).

## Documentation

- Architecture diagrams in [docs/](docs/) (the Pages site is text-only).
- Per-element rustdoc pass: every public element type gets an example block.
