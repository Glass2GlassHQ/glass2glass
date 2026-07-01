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
3. **Depth.** Codec decode to cut reliance on the ffmpeg FFI: AV1 landed both as
   libdav1d (`Dav1dDec`, `dav1d` feature, C FFI) and pure Rust (`Rav1dDec`,
   `rav1d` feature, via `re_rav1d`). VP8 / VP9 decode is covered by `FfmpegVideoDec`
   (a dedicated libvpx `VpxDec` is deferred: no pure-Rust decoder exists, and a
   libvpx-FFI element would only duplicate the ffmpeg path). The headline open
   item is **`VulkanVideoDec`** (see "Receive / decode"): vendor-neutral,
   GPU-resident hardware decode straight into a `wgpu::Texture`, the cross-vendor
   generalization of the CUDA-locked `NvDec` path and the wedge that gives a
   wgpu-based consumer (game engine / visualization viewer) hardware decode on its
   own render device.
4. **Browser demo (speculative product path).** Cross-target ONNX in-browser:
   CPU-round-trip MVP via `ort-web` (`WebSocketSrc -> WebCodecsDecode ->
   ort-web -> CanvasSink`), then a deployed reference app + native sibling. The
   GPU-resident in-browser chain is not achievable from idiomatic Rust (wgpu
   can't import a WebCodecs `VideoFrame` as an external texture or adopt ORT's
   device on wasm); it would need raw `web_sys` WebGPU + hand-rolled
   onnxruntime-web bindings.

## Negotiation

- **Preference algebra.** `CapsPreferences` is a placeholder (sum-of-indices);
  needs a real competing-constraint scenario to drive it.
- **Hardware `tee -> {decode, mux}` integration test** on real Linux
  (`rtsp ffmpeg wayland-sink`); only fake-element coverage today.

## Seek and auto-plug

- Richer auto-plug factory construction params (geometry / device / file path).
- A hardware-backed end-to-end decode-through-`decodebin` run (current tests
  read templates / assert splicing, decode no real media).

## Runtime / scheduling

- **Cooperative-runner element offload (Approach C).** Opt-in cross-arm
  parallelism now exists (`run_graph_threaded`, thread-per-arm, DESIGN.md
  §4.13.3). The remaining win is for the *cooperative default* runner: offload a
  heavy synchronous element's per-frame CPU (software `ffmpegdec` decode, the
  waylandsink XRGB convert) via a runtime-guarded `spawn_blocking`, so the sink
  renders while decode runs, without opting into `--threads`. `ffmpegdec` holds a
  `!Send` `AVCodecContext`, so it needs a small `unsafe Send` offload wrapper
  consistent with the element's existing single-threaded-access contract; a
  pure-CPU transform (videoconvert) is a cleaner first target. Release-build perf
  is adequate without either (1080p HLS A/V+subs holds 30 fps); debug builds
  misrepresent it, so validate live runs with `--release`.

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
- 10-bit pixel formats in `FfmpegH264Dec` (`YUV420P10` / `P010`).

- **`VulkanVideoDec` (vendor-neutral GPU-resident hardware decode).** Decode
  H.264/H.265/AV1 with `VK_KHR_video_queue` + `VK_KHR_video_decode_*` on the same
  Vulkan device wgpu already runs, emitting a `MemoryDomain::WgpuTexture` (RGBA)
  frame with no PCIe download, the vendor-neutral analog of the CUDA-locked
  `NvDec -> CudaToWgpu` path. AMD (RADV), NVIDIA and Intel (ANV) all expose the
  extensions, so one element covers all three (validated per vendor as hardware
  is available; NVIDIA on the RTX 3060 first, AMD/Intel stay `VERIFY:` until run).
  New feature `vulkan-video = ["std", "dep:wgpu", "dep:wgpu-hal", "dep:ash"]`;
  `AsyncElement` transform, `Caps::CompressedVideo{H264|H265|AV1}` Annex-B in ->
  `Caps::RawVideo{Rgba8}` `WgpuTexture` out; `output_domains = {WgpuTexture}`
  (optionally `VulkanTexture` / multiplanar NV12). Properties: `codec`,
  `output-format`, `num-dpb-slots`, `low-latency` (bounded DPB / no reorder for
  streaming), `device-index`.
  The expensive interop half is already in-tree and reused: the VkImage -> wgpu
  import seam (`cudawgpu.rs` / `dmabufwgpu.rs` `texture_from_raw` +
  `TextureMemory::External`), custom Vulkan device creation with extra extensions
  (the `cuda-wgpu` device path), the multiplanar NV12 -> RGBA `VkSamplerYcbcrConversion`
  compute pass (`mediacodec_wgpu.rs`), the Annex-B + SPS/PPS front-end
  (`h264parse.rs` / h265parse), and domain auto-plug (M351-M354).
  DONE (M486, `vulkanvideo::probe_decode_caps`, validated on the 3060): the
  capability probe -> the decode-capable queue family + coded-extent range + DPB
  slot / active-ref budget + `DPB_AND_OUTPUT_COINCIDE` that `intercept_caps` and
  DPB sizing negotiate against (survives fixate, never advertises `Dim::Any`),
  and the driver wrinkle that the caps query needs the codec-specific output caps
  struct chained (else `ERROR_INITIALIZATION_FAILED`).
  DONE (M487): the H.264 SPS/PPS parse (`parse_h264_sps`/`_pps`) + `Std*` mapping
  (`to_std_sps`/`to_std_pps`), the tedious correctness-critical half, with GPU-free
  unit tests. DONE (M488, validated on the 3060): the decode device
  (`open_h264_decode_device`, decode queue added via wgpu-hal `open_with_callback`,
  no hand-built `VkDevice`) and `VkVideoSessionKHR` + `VkVideoSessionParametersKHR`
  (`create_h264_session`), whose parameter creation makes the driver validate the
  M487 mapping. DONE (M489, validated on the 3060): `decode_idr_luma` decodes a
  single IDR frame end to end (NV12 DPB/output image, host-visible bitstream
  buffer, `vkCmdBeginVideoCodingKHR` + `RESET` + `vkCmdDecodeVideoKHR` + end, with
  `synchronization2` image barriers) and reads back a non-uniform luma plane, the
  first real Vulkan Video decode in the tree. What is left:
  DONE (M490, validated on the 3060): `decode_idr_to_rgba_texture` lands a decoded
  frame in an `Rgba8Unorm` `wgpu::Texture` on the decode device (the wedge
  payload), via a full-NV12 readback + CPU BT.601 conversion; `m490` reads the
  texture back through wgpu and asserts real content.
  DONE (M491, validated on the 3060): zero-copy GPU-resident NV12 -> RGBA.
  `decode_idr_to_rgba_texture_gpu` decodes into a CONCURRENT NV12 image; a compute
  pass on a dedicated compute queue converts it through a `VkSamplerYcbcrConversion`
  (reusing the `mediacodec_wgpu` shader) to an RGBA `VkImage` imported into wgpu via
  `texture_from_raw`, no CPU round-trip; read-back matches the M490 CPU convert. The
  vendor-neutral hardware-decode-into-wgpu-texture path now works end to end. Left:
  1. The `VulkanVideoDec` `AsyncElement` wrapper (negotiation via the probe caps,
     properties, `output_domains = {WgpuTexture}`, driving the decode per frame),
     re-emitting parameters on mid-stream SPS/PPS change via `CapsChanged`.
  2. Full DPB reference management: per-frame slice-header parse (frame_num, POC,
     idr_pic_id, slice_type), multi-slice pictures, and P/B-frame reference slots
     (M489-M491 hardcode the lone-IDR `Std*` constants + one DPB slot, so only
     all-intra streams decode past the first frame today). The gate to a useful
     element.
  3. Pipeline the per-frame path through a `RING_DEPTH` in-flight ring (M489-M491
     fence-wait each submission; fine one-shot, not for throughput).
  4. Hardening: PSNR-vs-ffmpeg reference comparison; then H.265 and AV1 (+1 each,
     mostly their parameter mapping).
  Top risks: the `Std*` structs / DPB management (where Vulkan Video decoders
  bleed correctness) and the `create_from_hal` device adoption under wgpu 29
  (prototype in isolation first). Test `mNNN_vulkan_video_decode.rs`: decode a
  known clip on the 3060, assert the output texture matches an ffmpeg reference
  (same harness as the CUDA->wgpu validation), skips with no adapter.
  Strategic note: this is the path that turns the wgpu-resident decode story from
  NVIDIA-only into genuinely cross-vendor (a wgpu-based consumer such as a game
  engine or a visualization viewer gets hardware decode straight into its own
  render device, matching what a browser gets from WebCodecs).

## CUDA / display

- `CudaKmsSink` on-tty validation (M255): the GL-on-KMS present path is authored
  + compiles (render half shared with the validated `CudaGlSink`), but the
  GBM/EGL/DRM present needs a real run from a bare VT (DRM master), which the dev
  session's compositor holds. Verify the `// VERIFY:` spots there.

## Egress / transports

- **SRT:** libsrt/ffmpeg real-peer interop. (TSBPD timing, AES-256
  (`with_aes256`), mid-stream key rotation (`with_key_rotation`), and live-mode
  congestion control / pacing (`with_max_bandwidth`) landed; KM-retransmit-until-
  KMRSP for lossy rekey is a refinement.)
- **RTMP:** the HMAC-digest handshake some CDNs require, multiple streams,
  server-acknowledgement back-pressure.
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
  (cenc/cbc1); `saiz`/`saio` aux-info + `seig` sample groups. (Encrypted fMP4 cbcs
  *video* init segments are done, M164; `#EXT-X-BYTERANGE` single-file CMAF is
  done, M368; throughput-driven ABR with mid-stream variant switching is done,
  M371; live-edge start is done, M438.)
- **DASH:** wall-clock `@duration` live profile; multi-period. (`SegmentList`
  byte-range is done, M369; `SegmentBase` `sidx`-indexed single-file CMAF is done,
  M370; throughput-driven ABR is done, M372.)

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

## Containers

- **MKV / WebM:** a front `SeekHead` so the muxer's own output is seekable from
  byte 0 without reading past the Clusters (a two-pass / seekable-output finalize
  mode; the streaming muxer writes `Cues` at EOS, M375, but cannot place a front
  `SeekHead`); `Targets`-scoped (per-track) tags. (Multi-track A/V muxing landed:
  `mkvmuxn`, M294; `Cues` parsing + indexed demuxer seek, M373; `SeekHead`-driven
  `Cues` prefetch, M374; muxer-written `Cues`, M375.)
- **MPEG-TS:** multi-stream / multi-program muxing + selection; PCR-based timing.
- **OGG:** granule-position timing; Vorbis output; multi-stream; `oggmux`.
- **CMAF / fMP4:** the CMAF-specific signalling layer on `Mp4Sink` / `Mp4Src`.

## Codecs

- **VP8 / VP9 encode** (`VpxEnc`): validate on a libvpx host (compile-unverified).
- **AV1 encode** (`Av1Enc`): explicit quantizer rate control. (Target-bitrate
  rate control with hysteresis is done; 8/10/12-bit in 4:2:0 / 4:2:2 / 4:4:4
  all done.)
- **Pure-Rust / wasm decode** to drop the ffmpeg FFI: AV1 done (`Rav1dDec`, emits
  4:2:0 / 4:2:2 / 4:4:4 at 8/10/12-bit, round-trip tested end to end); still
  VP8 / VP9 decode and a pure-Rust Opus path.
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
- **`textoverlay` font backend:** the `truetype-overlay` feature (M409, `fontdue`)
  renders glyf-outline fonts (CJK / accented / mixed-case, horizontal + vertical)
  with an explicit Latin+CJK fallback chain. Still open: CFF / CFF2 outlines (so
  variable Noto Sans CJK works, not just glyf fonts like Droid Sans Fallback),
  real shaping + bidi, and automatic system-font discovery / fallback, all of
  which point at the `cosmic-text` upgrade; plus a `vello` GPU backend and the
  `clockoverlay` / `timeoverlay` siblings.
- **`audiomixer`:** sample-rate + channel-layout reconciliation; PTS-based
  alignment.
- **`videotestsrc`:** a sinusoidal (vs square-wave) zone plate (needs `libm`).
- **Text / subtitle pipeline depth.** The foundation is in: `Caps::Text` +
  `TextFormat` (M400), the `SubParse` element (`Text{Srt|WebVtt|Ssa|Ttml}` ->
  `Text{Utf8}`), the SRT / WebVTT / SSA-ASS / TTML parsers (M171 / M401 / M402),
  the `TextOverlay` renderer (M171), and `TextOverlayN` (M403), the two-input
  video + `Caps::Text` stream overlay, with incremental cue streaming (M405) and
  cue positioning carried as `TextCueMeta` frame-meta (M406). The `gst-launch`
  surface is complete (M477): `subparse` and `subtitlesrc` are launch elements,
  `textoverlay` doubles as a video + text-stream fan-in muxer (the text_sink
  request-pad analog, picked by link degree), and an explicit demux fan-out
  selects an embedded subtitle track by pad name (`d.text_0` / `d.subtitle_0`),
  so a subtitle-overlay line parses end to end.
  Subtitle-track extraction out of the demuxers as `Caps::Text` (feeds
  `TextOverlayN`) is started: MP4 `tx3g` timed text fans out of `Mp4DemuxN` as
  `Caps::Text{Utf8}` (M411) and `mp4_playbin` auto-plugs it through a
  `TextOverlayN` on the video branch (M412); MKV `S_TEXT/UTF8` likewise fans out of
  `MkvDemuxN` as `Caps::Text{Utf8}` with the `BlockDuration` cue window (M413), and
  `mkv_playbin` auto-plugs it through the same shared overlay builder
  (`wire_subtitle_overlay`, M415). MP4 `wvtt` / `stpp` are read too (M416: `wvtt`
  de-frames its `vttc`/`payl` boxes to `Text{Utf8}`, `stpp` passes the TTML document
  as `Text{Ttml}` through `SubParse`), as are MKV `S_TEXT/ASS` / `S_TEXT/WEBVTT`
  (M417: the block is de-framed to plain `Text{Utf8}` cue text, the source syntax
  only selecting the de-framing). Still open: the **MPEG-TS** subtitle path, which
  is a separate, larger effort, not a sibling of the MP4 / MKV text wiring: TS
  carries DVB subtitles (bitmap RLE, a `Caps::SubPicture` track, see below) and
  teletext (a page/magazine decoder), neither a text format `TextOverlayN`
  consumes, so there is no TS text stream to overlay until one of those lands.
  HLS subtitle renditions: discovery + language selection landed (M418 -
  `variant_streams` surfaces `SUBTITLES` renditions as `Caps::Text`,
  `MasterPlaylist::pick_rendition` selects by `#audio-lang=` / `#subtitle-lang=`
  URI hint, audio fan-out honours it). Subtitle *playback* fan-out landed for the
  common case (M419: `HlsSrc::with_text` emits `Caps::Text { WebVtt }` from a raw
  `.vtt` rendition, `build_hls_subtitle_overlay` joins it through `SubParse` into the
  video's `TextOverlayN` across sources, wired by `hls_playbin` for a muxed-A/V TS
  variant + `SUBTITLES` rendition). The separate-audio + subtitle three-source
  combo landed too (M420: `build_hls_separate_subtitle_overlay` pairs the variant's
  video TS with a distinct audio rendition and a distinct WebVTT rendition, three
  sources in one graph). Follow-ups: the fMP4 `wvtt` subtitle rendition (`IsoBmff` +
  `Mp4DemuxN`, reuses M416) and the `X-TIMESTAMP-MAP` offset for live (non-absolute)
  WebVTT timelines. The startup I420/NV12 gap on
  `playbin` -> `waylandsink` is closed (M414: the auto-plugged ffmpeg decoder now
  honours the chosen output layout and emits NV12 straight to a strict-NV12 sink,
  no inserted `videoconvert`). MPEG-TS / HLS H.264 now decodes cleanly on screen
  (M421: an access-unit-re-framing `h264parse` is auto-inserted before the decoder,
  validated live against Apple bipbop: 0 decode errors, matching GStreamer). Linux
  AAC decode landed too (M422: `FfmpegAudioDec` + ADTS frame splitting; the playbin
  audio branch wires `decode -> audioconvert -> audioresample -> autoaudiosink`;
  bipbop plays clean video + audio + subtitles live, audio via `PulseSink` ->
  pipewire-pulse). Mono / multichannel audio works too (M423: an `ANY_CHANNELS`
  wildcard in `Caps::Audio`, decoder advertises it instead of constant stereo, and
  `audioconvert` does general N -> M downmix/upmix), and the plain A/V fan-out routes
  audio through the convert/resample branch like the overlay path (M424:
  `build_av_fanout` / `wire_av_fanout`). HEVC TS/HLS re-frames like H.264 (M425:
  `H265Parse::reframing` auto-inserted before the decoder) and Opus auto-plugs in
  the audio branch (M425: `mkvdemux::forwardable_streams` surfaces concrete channels,
  `OpusDec` sink template relaxed to match). The overlay graph runs end to end.
  Remaining playback follow-ups:
  - **Audio breadth.** More codecs (AC-3, FLAC, etc.). The layout-agnostic downmix in
    `audioconvert` folds channels round-robin rather than applying ITU/speaker-position
    coefficients (no channel-position metadata is carried yet). Opus in MP4 / TS
    (`dOps`) is not demuxed yet (WebM / Matroska only). The audio sink needs the
    `pulse-sink` (or `alsa-sink`) feature built in, else `autoaudiosink` falls back to
    `fakesink`.
  Parsing SSA / TTML placement into `CueSettings` (only
  WebVTT populates it today, though all three now ride the frame-meta). Glyph
  rendering (incl. `vertical:rl` / `lr` layout) is the `truetype-overlay` feature
  above. WebVTT `::cue` / `::cue(#id)` `color` / `background-color` are applied
  (M410); still open: `::cue(.class)` span selectors and other CSS (font-size,
  text-shadow, etc.).
- **Closed captions: remaining carriers + authoring.** The H.264 / H.265 SEI
  decode path (`cea` decoders + `CcExtract` + file- and HLS-`playbin` auto-plug)
  and the CEA-608 encode path (`Cc608Enc` + `CcInsert`) are done (DESIGN.md
  §4.18). Still open: MPEG-2 user-data caption extraction; and the MP4 `c608` /
  `c708` *raw-caption track* (the one case justifying a `Caps::ClosedCaption
  { format }` variant).
- **Bitmap / picture subtitles (DVD / PGS / DVB).** RLE-image subtitles, not
  text: a `Caps::SubPicture { codec }` variant + RLE image decoders, mirroring the
  `CompressedVideo` / `RawVideo` split rather than folding into `Text`. Niche;
  deferred until a concrete need.
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
  - `NvEnc`: 10-bit (P010 / Main10) and finite-GOP periodic IDRs with
    `repeatSPSPPS`. (RGBA input + the wgpu->CUDA `WgpuToCuda` bridge are done,
    M271; HEVC is done, M273; the output-bitstream pool + runtime bitrate retarget
    are done, M277; system-memory NV12 input is done via the `CudaUpload`
    converter + domain auto-plug, M353/M354. NVENC AV1 needs RTX 40-series.)
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

- Property-set the remaining feature-gated sources from text (`location=` /
  `uri=` on rtsp / v4l2, default placeholders today; http / hls / dash now carry
  `location`).
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

## Aggregation helper adoption (M199+)

- Migrate the remaining hand-rolled per-input collectors onto
  `g2g-core::InputAggregator<T>` (`mux` is migrated): enterprise `batcher`
  (closest fit), `audiomixer`, and `compositor` (compositor needs a second
  latest-wins `SyncPolicy` variant first). Behaviour-preserving, each guarded by
  existing tests.

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

- Detector on the Edge TPU is blocked device-side: this Pixel's older Android ORT
  NNAPI EP rejects YOLO's op set (int8-weight initializers, SiLU `Mul` QDQ
  "unsupported quantized type", and an `AddNnapiSplit` divide bug on the C3k2
  channel split); a simple conv stack (MobileNet, M447) places fine. Needs a newer
  ORT build or a TPU-friendly detector (SSD-MobileNet-style, conv-only). The host
  detector (M448) works.
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

- **Measured per-element latency report, remaining runners + link transit.**
  M399 wired measured per-element `process()` p50/p99 + input-link fill into the
  graph runner and the two linear runners (`RunStats::per_element`,
  `ElementProbe`). Still open: the fan-in / fan-out / session / muxer runners
  (leave `per_element` empty today); per-*link* transit / queue-residency time
  (needs a wall-clock stamp carried with each packet, not just the element-side
  `process()` timing M399 collects); and source-side timing (sources run one long
  `run()` loop, so their cost only shows as the downstream element's input fill).
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
covered; the flagged hardening follow-ups are now fixed (segment-fetch body cap,
free-threaded analytics sink, descriptive `Pipeline::wait` errors).

## Documentation

- Architecture diagrams in [docs/](docs/) (the Pages site is text-only).
- Per-element rustdoc pass: every public element type gets an example block.
