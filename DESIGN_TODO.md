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
   Android: encode, Camera2, AAudio, Surface present.
2. **Egress / transports.** SRT congestion control + real-peer interop, AES-256
   + key rotation; FlexFEC + multi-level burst FEC.
3. **Depth.** Pure-Rust / wasm codec decode (dav1d/rav1d, vpx) to drop the
   ffmpeg FFI; negotiation backward coupling through `DerivedOutput`; seek depth
   (segment seeks, re-preroll after flushing seek when paused).
4. **Browser demo (speculative product path).** Cross-target ONNX in-browser:
   CPU-round-trip MVP via `ort-web` (`WebSocketSrc -> WebCodecsDecode ->
   ort-web -> CanvasSink`), then a deployed reference app + native sibling. The
   GPU-resident in-browser chain is not achievable from idiomatic Rust (wgpu
   can't import a WebCodecs `VideoFrame` as an external texture or adopt ORT's
   device on wasm); it would need raw `web_sys` WebGPU + hand-rolled
   onnxruntime-web bindings.

## Negotiation

- **Closure-free `FieldTransform` refactor.** Make forward derivation
  declarative too, for a `Debug`/`Copy` single-source-of-truth descriptor.
- **Graceful per-branch drop on fan-out** (`FanOutPolicy::AllowBranchDrop`); a
  rejecting branch fails the run loud today.
- **Merged downstream output for dynamic fan-in.** `run_aggregator_dynamic`
  (M320) drives a *terminal* aggregator; the `run_muxer_sink` shape (a trailing
  sink with output-caps coupling) for runtime-added inputs is still owed.
- **β allocation re-cascade across a muxer.** A muxer's inputs have no per-pad
  re-cascade channel, so the DAG β walk terminates at a muxer.
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

- `MediaCodecDec` zero-copy to GPU (M304): DONE, decode -> preprocess on GPU.
  `with_gpu_output()` emits decoded frames as `MemoryDomain::WgpuTexture` (RGBA)
  via the `YcbcrToRgba` converter (opaque AHB import -> immutable
  `VkSamplerYcbcrConversion` compute pass -> RGBA `wgpu::Texture`), and
  `WgpuPreprocess` consumes that texture straight into its tensor (g2g-ml
  `mediacodec-wgpu` feature). The converter pipelines via a `RING_DEPTH`-slot
  in-flight ring (no per-frame fence block), and both elements negotiate the
  RGBA-GPU path (decoder derives Rgba8 in gpu mode, WgpuPreprocess accepts it) so
  a runner can auto-plug it. Validated on the Pixel 10a end to end (9 frames ->
  NCHW tensor, no CPU frame copy). Pillar complete.
- Android `Surface` present sink (M305): DONE, validated on a Pixel. Decoded RGBA
  `WgpuTexture` (M304) presented through a `wgpu::Surface` over an `ANativeWindow`
  by the existing `WgpuSink` on the shared interop device (copy-free).
  `mediacodec_wgpu::create_android_surface` + `InteropDevice::gpu_context()` +
  `MediaCodecDec::with_gpu_device`. Remaining: a real on-screen `SurfaceView` /
  `NativeActivity` (production target; the `ImageReader`-backed surface is the
  validated headless equivalent).
- Encode (M306): `MediaCodecEnc` (NV12 -> Annex-B H.264/H.265). DONE, validated on
  a Pixel. Registered `mediacodecenc` / `mediacodecench265`.
- AAudio (M307): `AAudioSink` render + `AAudioSrc` capture. DONE, validated on a
  Pixel (render + mic capture). Registered `aaudiosink` / `aaudiosrc`.
- Camera2 capture (M308): `Camera2Src` (YUV_420_888 -> NV12 via NDK Camera2 over
  ndk-sys). DONE, validated on a Pixel (real rear-camera NV12). Registered
  `camera2src`.

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

- `CudaKmsSink` on-tty validation (M255): the GL-on-KMS present path is authored
  + compiles (render half shared with the validated `CudaGlSink`), but the
  GBM/EGL/DRM present needs a real run from a bare VT (DRM master), which the dev
  session's compositor holds. Verify the `// VERIFY:` spots there.

## Egress / transports

- **SRT:** congestion control, libsrt/ffmpeg real-peer interop. (TSBPD timing,
  AES-256 (`with_aes256`), and mid-stream key rotation (`with_key_rotation`)
  landed; KM-retransmit-until-KMRSP for lossy rekey is a refinement.)
- **RTMP:** the HMAC-digest handshake some CDNs require, multiple streams,
  server-acknowledgement back-pressure.
- **RTP FEC:** FlexFEC (RFC 8627). (Interleaved column ULPFEC for burst loss
  landed: `InterleavedFecEncoder` / `UdpSink::with_interleaved_fec`.)
- **RTSP server:** TCP-interleaved transport; RTCP / keepalive during PLAY.
- **`UdpSrc` SDP/SPS-driven caps discovery** (reports a declared hint today).
- **WebRTC.** On the sans-IO `str0m` stack (ICE / DTLS / SRTP, pure-Rust
  crypto), behind the `webrtc` feature: `WebRtcSink` (WHIP egress, H.264 *or*
  Opus) and `WebRtcWhepSrc` (WHEP ingest, H.264 *or* Opus via `media=audio`) —
  egress + ingress both exist, with shared ICE/SDP helpers (`webrtc_util`), STUN
  server-reflexive candidate gathering (`stun-server`) and a hand-rolled TURN
  relay client (`turn-server` + `turn-user` / `turn-pass`, RFC 5766/8656: Allocate
  with long-term auth, Send/Data indications, CreatePermission, Refresh) so the
  elements reach cloud SFUs through symmetric NAT, a WHEP player + ignored
  `webrtc_whip_smoke` + `webrtc_whip_to_whep_loopback` harness. Compile-validated
  against str0m 0.20. The browser data-channel `WebRtcSrc` stays wasm-only.

  Roadmap toward GStreamer (`webrtcbin` / `gst-plugins-rs` `webrtcsink`) parity,
  staying sans-IO + pure-Rust (str0m does the engine work, so no libnice /
  OpenSSL). Two enablers already exist: `MultiInputElement` / `MultiOutputElement`
  (M199) make a multi-track session expressible, and the `Reconfigure` /
  `QosMessage` reverse channel (M174/M175) already walks upstream to the source.
  str0m already emits `Event::KeyframeRequest` and `Event::EgressBitrateEstimate`;
  most feedback work is wiring those onto the reverse channel, not new engines.
  - **T0 (precondition).** On-network validation against a real WHIP/WHEP server.
    Single-track DONE (M247): WHIP egress + WHEP ingress validated end to end
    against a local mediamtx (ICE/DTLS/SRTP completes, H.264 media flows
    g2g->mediamtx->g2g, loopback receives frames); found + fixed the `Dim::Any`
    fixate-failure bug. Multi-track A/V DONE (M248): `WebRtcSessionSink` publishes
    H.264 + Opus over one PeerConnection and `WebRtcWhepSessionSrc` reads both back
    (`webrtc_av_session_loopback`, both tracks received; mediamtx logs `2 tracks`).
    Remaining: browser playback via the WHEP player, and a real LiveKit Cloud /
    TURN-relay run (genuine remote NAT).
  - **T1 (keystone): unified `WebRtcBin`-equivalent session element.** One element
    owning one `Rtc` with N tracks, on the multi-pad traits, so BUNDLE / A-V on one
    PeerConnection / sendrecv / data channels all hang off it. Fixed-arity-from-caps
    tracks declared at negotiation (NOT webrtcbin dynamic request pads), per the
    Option-A flattening decision. Egress DONE (M245): terminal fan-in runner
    `run_fanin_session` (N sources -> terminal `MultiInputElement`, no downstream
    sink) + `WebRtcSessionSink` (one `Rtc`, H.264 video + Opus audio m-lines, one
    WHIP session). Ingress DONE (M246): `MultiOutputSource` trait + terminal
    fan-out runner `run_fanout_session` (one 0-in-N-out source -> N sinks) +
    `WebRtcWhepSessionSrc` (one `Rtc`, WHEP recv H.264 video + Opus audio on two
    output pads). Bidirectional sendrecv DONE (M249): `MultiDuplexSession` trait +
    `DuplexInbound` + terminal `run_duplex_session` runner (the union of fan-in
    send and fan-out recv, expressing an element that is at once sink and source)
    + `WebRtcDuplexSession` (one `Rtc`, sendrecv m-lines; WHIP/WHEP can't carry
    sendrecv, so peers exchange SDP directly over an `SdpChannel`). Validated by
    in-process P2P loopbacks (video + full A/V, localhost, no server). Remaining:
    per-input/branch reverse-signal routing (PLI / BWE) and mid-stream re-solve
    through the multi-track runners; launch-registry wiring; STUN/TURN for the
    duplex path; mid-session transceiver add/remove (renegotiation); a pluggable
    real-SFU (LiveKit) signaller for the duplex element.
  - **T2 (mostly wiring): RTCP feedback.** PLI / keyframe-request DONE (M243):
    `Reconfigure::ForceKeyframe` + `take_reconfigure`; `WebRtcSink` maps a remote
    `Event::KeyframeRequest` to it, `Av1Enc` forces an IDR, `WebRtcWhepSrc`
    originates PLI on mid-GOP join. Adaptive bitrate / congestion control DONE
    (M244): `PushOutcome::Bitrate` + `take_bitrate`; `WebRtcSink` enables str0m
    BWE and relays `Event::EgressBitrateEstimate`, `Av1Enc` retargets (rav1e
    context rebuild, hysteresis-gated). Remaining: VP8/VP9 runtime bitrate +
    force-keyframe (needs a libvpx path `vpx-encode` does not expose); Opus
    bitrate adaptation; `ForceKeyframe`/`Bitrate` relay through an intervening
    transform; NACK / RTX (str0m-internal, enable by offering the RTX payload
    type).
  - **T3: TURN / ICE completeness.** TURN channel binding (lower overhead than
    Send/Data indications), TURN-over-TCP / -TLS, IPv6 reflexive + relay, multiple
    TURN servers, 438 stale-nonce retry, trickle ICE (WHIP/WHEP `PATCH`), ICE
    restart. Incremental on the M242 `turn.rs`.
  - **T4: signalling ecosystem.** Native LiveKit signaller (websocket + protocol),
    then Janus / Kinesis as wanted, layered over the T1 engine like
    `gst-plugins-rs` layers signallers over `webrtcbin`.
  - **T5: advanced.** Native data-channel source/sink on str0m SCTP (unifying the
    wasm-only `WebRtcSrc`); simulcast (encoder fan-out); FEC; full renegotiation.
  Smaller loose ends: non-stereo / non-48 kHz Opus; WHIP/WHEP `DELETE` + graceful
  flush on EOS. Recommended order: T0 -> T1 -> T2 (PLI first) -> T3 -> T4.

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
- **`FfmpegAacEnc`: end-to-end encode test** (needs a Linux ffmpeg build to run);
  the AAC encode core is otherwise untested.

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
- An embedder example for the bring-your-own-device path (M263): a real engine
  (Bevy / a raw `wgpu` app) that builds a `GpuContext::from_wgpu` over its own
  device, runs `decode -> (preprocess) -> appsink`, and samples the yielded
  `WgpuTexture` onto an in-app 3D surface. The mechanism is validated (M263 unit
  test); a worked example is the adoption artifact for the game-engine wedge.
  Bevy 0.19 pins the same wgpu 29 as g2g, so the device handoff type-checks
  (clone Bevy's `RenderDevice`/`Queue`/`Adapter`/`Instance` into `from_wgpu`).
- The native gst-`nvcodec`-style pair is done: `NvEnc` (zero-copy CUDA NV12 ->
  H.264, M269) and `NvDec` (H.264 -> CUDA NV12 via NVCUVID, M270). Remaining
  extensions on both:
  - `NvEnc`: system-memory NV12 input (host upload), 10-bit (P010 / Main10), and
    finite-GOP periodic IDRs with `repeatSPSPPS`. (RGBA input + the wgpu->CUDA
    `WgpuToCuda` bridge are done, M271; HEVC is done, M273; the output-bitstream
    pool + runtime bitrate retarget are done, M277. NVENC AV1 needs RTX 40-series.)
- `NvDec` depth: mid-stream resolution change (decoder reconfigure), AV1 / other
  codecs via the codec enum, 10-bit output, and a configurable display delay
  (fixed at a low-latency 1 today). (HEVC is done, M273; registry + domain-aware
  auto-plug are done, M272 / M276: `decodebin_preferring(.., Cuda)` prefers
  `NvDec`. The remaining piece is deriving that preference automatically from a
  downstream consumer's accepted input memory.)
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
- A real HAL-backed DMA capture: wire a DMA-completion ISR into the
  `StaticLendRing` (M260 proved the no-alloc lend path on the host via a fill
  stand-in; the ISR / vendor HAL plug-in is hardware-gated).

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

- Trained-weight import now exists for the hand-rolled GPU path: a dependency-free
  `safetensors` reader (M262) loads weights at runtime into `WgpuInference`
  (`conv2d_from_safetensors`); architecture stays compiled, weights are a file.
  Conv / activation (`relu`, `sigmoid`) / pooling (`maxpool2d`, `avgpool2d`) and
  GPU-resident multi-layer chaining are in place (M261/M265). Remaining: the
  remaining ops (batch-norm, attention) and a topology that imports a whole
  multi-layer stack from one weight file (not just a single conv) so a *full*
  trained model runs end to end; per-tensor dtypes beyond F32 in the loader
  (F16 / BF16 dequant).
- ONNX import via `burn-import` (build-time codegen) for the Burn backend, the
  graph-topology counterpart (safetensors carries weights, not the architecture).
- A trained-weight `Module` path for `BurnInference` (conv, attention) once the
  codegen lands.
- Decoder DMA-BUF / D3D11 surface import into `WgpuPreprocess` (binds the
  surface directly into the compute pass; needs the surface-import handshake + a
  GPU tensor output domain).

## Developer tooling

The `xtask` crate (`cargo xtask ci | test --here | size | wasm | bench |
ffi-probe`), the DOT visualizer (with negotiated caps + per-edge memory domains
via `negotiate_graph`), the caps explainer, and the criterion benches now exist;
the remaining items extend them. Highest leverage first:

- **Measured per-element latency report.** `RunStats::report()` (M287) surfaces
  the frame counts, drop rate, and *declared* latency fold at run end. Add
  frame-level instrumentation in the runner (timestamp each frame per edge) to
  report measured per-element / per-link p50 / p99 + channel fill-level, the
  glass-to-glass analyses (NVDEC floor, `link_capacity` dominance) done by hand
  today. The `LatencyHistogram` in `metrics.rs` is the collector; it needs wiring
  into the arms.
- **Element scaffolding.** `xtask new-element` (a new subcommand) stamps the
  `AsyncElement` / `SourceLoop` impl + pad templates + registry stub + milestone
  test file (the boilerplate every `Mn` repeats).
- Longer tail: a live pipeline TUI (`gst-launch -v` on steroids); a gst-parity
  differ (same launch line through real GStreamer vs g2g, diff caps / behaviour);
  a codec golden-fixture / PSNR conformance harness; an MCP server exposing
  `inspect` / `launch` / `validate` for agent-driven dev.

## Code audit follow-up

A `/code-audit-pro` pass (2026-06) fixed runtime/leak/dedup findings across the
runtime, parsers, mux/demux, RTP/network, codecs, platform codecs, the g2g-core
negotiation core, the untrusted demuxers, the g2g-ml inference path (model
shape / tensor-element / GPU-buffer arithmetic folded with checked ops), and the
g2g-python hosting boundary (zero-copy frame-buffer retention now caught by an
export counter; PyTransform worker re-spawn guarded). The audit areas are now
covered; what remains is lower-priority hardening flagged but not yet fixed (no
clear untrusted-file path, or a broader policy call):

- Bound caps geometry at `configure_pipeline` so a malformed container's huge
  width/height fails fast instead of driving a multi-GiB GPU allocation
  (`wgpupreprocess.rs`, and the weightless `wgpuinfer.rs` constructors whose
  `u32` shape products are still unchecked).
- `fetch.rs` uncapped HTTP body read (DASH/HLS DoS, network-layer).
- `mp4` `parse_progressive` allocation amplification (bounded by file size).
- Free-threaded (PEP 703) build: `host.rs` `MetaSink` uses a `RefCell` that only
  the GIL serializes today; it (and the `unsendable` pyclasses) would need a
  `Mutex` / re-audit before the "no code change" free-threaded claim holds.
- `g2g-pyapi` `Pipeline::wait` collapses the underlying `G2gError` / worker
  panic into an opaque "pipeline errored" (safe, but lossy diagnostics).

## Documentation

- Architecture diagrams in [docs/](docs/) (the Pages site is text-only).
- Per-element rustdoc pass: every public element type gets an example block.
