# Design TODO

Outstanding work, tracked against the architecture in [README.md](README.md).
This file is a terse catalogue of open tasks only. Completed work and the
rationale for shipped architecture live in [README.md](README.md) and
[CHANGELOG.md](../CHANGELOG.md), not here.

## Roadmap (high level)

Highest leverage first:

1. **Platforms.** Run the Windows and macOS device providers on a real host,
   and persist conformance evidence for the macOS device paths.
2. **Egress / transports.** Real-peer FlexFEC interop when a peer
   implementation is available (GStreamer here lacks `rtpflexfecenc`).
3. **Depth.** Pure-Rust codec paths to cut the remaining ffmpeg FFI reliance.
   No credible pure-Rust VP8 / VP9 decoder exists (a
   libvpx-FFI `VpxDec` stays deferred: it would only duplicate the ffmpeg
   path), and the one complete pure-Rust Opus, `opus-rs` 0.1.26, fails the
   RFC 8251 vectors (re-run `tools/opus-rs-gate` to revisit on a new release).
   `VulkanVideoDec` residuals: Intel validation runs (see "Receive / decode").
4. **Browser demo (speculative product path).** A deployed reference app for the
   in-browser `ort-web` ONNX chain, plus a native sibling running the same graph.
   The GPU-resident in-browser chain is not achievable from idiomatic Rust (wgpu
   can't import a WebCodecs `VideoFrame` as an external texture or adopt ORT's
   device on wasm); it would need raw `web_sys` WebGPU + hand-rolled
   onnxruntime-web bindings.

## Architecture guarantees (validation-first)

- **Drop the vendored `cros-libva`.** The workspace patches cros-libva 0.0.12 with the libva 2.23 VP9 fields, and `[patch]` does not reach crates.io users, so the published `vaapi` feature fails on libva 2.23. Remove the patch once a cros-codecs release accepts a cros-libva with upstream commit 5938445.
- **Grow the conformance matrix.** Persist evidence from the resource-owning
  tests still uncovered as they are validated (`vaapi` decode on Intel, the Android and
  macOS device paths). Get the device-tagged `Hardware` rows into CI by wiring a
  runner that has the hardware: a `Hardware` row can only come from a run on the
  device, so a runner without a GPU or a camera will never produce one.

## Alloc-optional (heap-free) MCU core

- **On-device `Hardware` rows (ARM) on a NUCLEO-H753ZI.** The H753 has the
  H743's core, memory map and Ethernet MAC. Switch `examples/g2g-stm32h743` to
  the `stm32h753zi` embassy feature and the `STM32H753ZITx` probe-rs chip, then
  finalize the RCC, clock, RMII pin and RTP destination settings. Put the
  Ethernet DMA buffers in non-cached D2 SRAM, since the H7 Ethernet DMA cannot
  reach DTCM.
  - **Fixed input.** Run the flagship audio graph on the reference tone, with
    ffmpeg receiving the RTP on the host, and require the payload checksum to
    equal `AUDIO_EXPECTED_CHECKSUM`: the first `Hardware` conformance row. Then
    run `HwJpegDec`.
  - **Measure against the published budgets.** ROM and static RAM from the
    flashed ELF, peak stack by stack painting (budget 10572 B ROM, 0 B RAM,
    6504 B stack), per-frame worst case and jitter from the DWT cycle counter
    (QEMU figure 764 us per 10 ms frame), power-on to first packet, RTP
    sequence, timestamp and packet-spacing jitter on the receiver, and a 24 h
    soak with zero ring overruns.
  - **Live microphone.** An I2S MEMS microphone on SAI with DMA into
    `SpscCaptureSrc`: wire the DMA-completion ISR into `StaticLendRing` and
    connect `EmbassyClock` to a HAL tick. Capture-to-send latency from
    cycle-counter stamps, and a 24 h soak with zero overruns.
  - **Nightly run.** A self-hosted runner with the board attached flashes
    master nightly and writes the `Hardware` row. It takes scheduled and
    master-push jobs only, never pull requests from forks. Publish the numbers
    in the conformance matrix, README and docs site.
  - **Comparison.** The same graph with g2g on Linux and with GStreamer on the
    smallest Linux board: ROM, RAM and boot time beside the MCU numbers.
  - Add the same conformance and timing evidence for an NXP i.MX RT board.
- **TinyML inference on MCU.** Run ExecuTorch (CMSIS-NN, Ethos-U) and LiteRT
  for Microcontrollers through `CTensorModel` on real boards.
  - **Host and device check on silicon.** Reproduce the keyword graph's pinned
    features and scores (`g2g-mcu/tests/fixtures/keyword`) on a real Cortex-M
    board, and run the model's ONNX twin through `OrtInference`, which takes
    only a `[1, 3, H, W]` input today.
  - **Keyword spotting on the H753.** Live microphone -> 16 kHz -> log-mel ->
    int8 keyword model -> smoothed detection out over UART or RTP.
  - **Camera detector on an STM32N6 board.** The Rust-native NPU target:
    embassy-stm32 and probe-rs support the N657, and ST's Neural-ART runtime
    goes behind the C bridge.
  - **Camera detector on Ethos-U.** An Alif Ensemble E7 AppKit (two Ethos-U55)
    through ExecuTorch, with g2g as a static library in Alif's C or Zephyr
    project, since Alif has no Rust HAL or probe-rs target.
  - Add `thumbv8m.main-none-eabihf` (Cortex-M55 / M33) to the heap-free,
    panic-free and footprint checks.
- **ESP32-P4X board bring-up (RISC-V on-device).** Put a pipeline on the
  P4X-EYE board in two tiers. Verify these
  unknowns before committing to a toolchain: whether `esp-hal` has any
  pure-Rust MIPI-CSI / ISP / HW-H.264 support (expect C-only, so the C-seam),
  and whether bare `no_std` Rust can reach the on-board ESP32-C6 WiFi stack
  without pulling in `esp-idf`/`std` (this decides Tier 2's toolchain).
  - **Tier 1: esp-hal harness + display.** When esp-hal publishes an `esp32p4`
    release, switch `examples/g2g-esp32p4` to it. Then verify
    the GPIO map + esp-hal API calls on the board and light the ST7789. Add an
    esp-hal `I2c` adapter to reuse `Sht3xSrc` on metal, and the on-device
    evidence row (a checksum verified on the P4 plus a real-silicon timing
    sample).
  - **Tier 2: camera -> encode -> RTP flagship (needs vendor C drivers).**
    Wire the P4's HW H.264 C driver behind `CH264Encoder` on silicon. MIPI-CSI
    camera source: bridge the ESP-IDF C camera driver (`esp_cam_sensor` /
    `esp_video`) through `CFrameGrabber`, since esp-hal almost certainly lacks
    pure-Rust CSI/ISP. WiFi/RTP egress via the ESP32-C6 network stack behind
    `CPacketSender`; if bare `no_std` cannot reach the C6 stack, this forces the
    esp-idf staticlib path (FreeRTOS-on-RISC-V, the analog of
    `examples/g2g-freertos`), optionally a Zephyr `esp32p4` board target. Then
    the on-silicon flagship, `camera (MIPI-CSI) -> convert -> HW-H.264 -> RTP
    -> C6/WiFi`, wire-validated against a host RTP peer, with a tee'd branch to
    `SpiDisplaySink` for an on-panel preview.
- **QNX (safety-certified RTOS, automotive/medical).** Tier 1 (needs the free
  QNX SDP 8.0): the `std` transports; the one dependency question is `tokio`
  on QNX 8. Tier 2 (needs an SoC + partner): QNX Screen display sink + vendor
  VPU via the C-seam + GPU, as `target_os = "nto"` elements. Free to test
  (non-commercial SDP); commercial use is license-gated (confirm the
  open-source-interop clause).

## Receive / decode

- **`VulkanVideoDec` residuals.** Run the `vulkanvideo` GPU tests on Intel ANV.
- **`VulkanVideoDec` conformance residue on RADV.** `WPP_D_ericsson` (64 wide,
  wavefront parallel processing with dependent slice segments) is refused,
  because g2g decodes it wrong on RADV while ffmpeg's Vulkan decoder decodes it
  bit-exact there (md5 f710612103f386c415be3e6300693451). The Std SPS, PPS and
  picture-info fields Mesa's `get_h265_msg` reads and the slice bitstream match
  ffmpeg's on a static read, so the difference needs an experiment loop on the
  RADV device. NVIDIA Turing cannot decode it at all (minimum H.265 coded extent
  144x144).
## CUDA / display

- Run `CudaKmsSink` from a bare VT with DRM master and verify the in-tree
  `// VERIFY:` points.

## Egress / transports

- **RTP over QUIC (RoQ):** implement after the draft becomes an RFC with an
  assigned ALPN. Candidate peers: mengelbart/roq (Go), meetecho/imquic.
- **`unixfdsink`:** carry a dma-buf frame's GPU fence. GStreamer's wire has no
  fence slot, and the fd `WgpuToDmaBuf` exports is a timeline semaphore that poll
  cannot wait on, so this needs a `vkWaitSemaphores` wait in the sink or a
  pollable sync_file from the exporter.
- **RTMP:** multiple NetStreams over one connection. Deferred by design: it needs
  a dynamic-arity multi-output `RtmpSrc` (the stream count is only known once the
  client `createStream`s at runtime), which collides with g2g's fixed-arity-from-caps
  model. Niche in practice (OBS / ffmpeg / CDNs publish one stream per
  connection); revisit only with a concrete need. Egress to a real CDN stays
  user-side.
- **WebRTC.**
  - FEC is blocked upstream (str0m has no FEC payload; loss recovery is
    NACK/RTX).
  - A data channel signalled through a WHIP / WHEP session.
  - Move the bevy-g2g input backchannel from its WebSocket side channel to
    `livekitdatasink` / `livekitdatasrc`.

## Adaptive streaming (HLS / DASH)

- **Same-codec tracks in one HLS TS variant.** `build_hls_ts_fanout` and `hls_ts_uri_fanout` build `TsDemuxN::new` from `HlsStreamInfo`, which carries no PID, so a second track of the same codec in a muxed variant leaves its port with no frames. Probe the variant's PMT for PIDs and use `TsDemuxN::from_stream_infos`.
- **HLS / CENC:** the multi-key shapes (`senc` v1/v2, multi-key `seig`
  entries) stay declined fail-loud: the 23001-7:2023 syntax is paywalled and
  the two available sources (the MPEG proposal and GPAC) contradict each
  other on the flag position and field widths, so a decode would be an
  unvalidated claim. Revisit with the published spec text or a second
  independent implementation.

## Capture sources

- `mfvideosrc`: first Windows build + camera smoke test; D3D11 zero-copy;
  size/rate request beyond device default.
- Screen capture: Windows DXGI Desktop Duplication.
- Device discovery on Android (Camera2 id list) and web (enumerateDevices)
  providers.
- Camera controls (exposure, focus, white balance) as element properties on
  AVCaptureDevice and Camera2.
- ONVIF PTZ and event subscriptions.
- ONVIF metadata tracks advertised with the Streaming Specification's
  `vnd.onvif.metadata+gzip` encoding name, or with an EXI coding: retina builds
  no depacketizer for either and fails the whole session on a set-up stream it
  cannot depacketize, so those tracks are skipped today.
- Run the Windows (`mfdevice` / `wasapidevice`) and macOS (`avfdevice` /
  `coreaudiodevice`) device providers on a real host: enumeration against
  attached hardware, endpoint selection by id through each element's `device`
  property, and the `IMMNotificationClient` hotplug path. Both are
  compile-checked only (CI cross-compiles them; the runners have no camera and
  no way to replug one).
- SDI capture for Blackmagic Design DeckLink (`decklinkvideosrc`,
  `decklinkaudiosrc`).

## Sinks

- Validate `wasapisink` U8, S24, and S32 acceptance on a Windows host.
- SDI output for Blackmagic Design DeckLink (`decklinkvideosink`,
  `decklinkaudiosink`) on Windows and macOS.
- Run `decklinkvideosink` and `decklinkaudiosink` on a board: scheduled
  playback, the board clock and audio against video have only run against a
  mock device.

## Containers

- **Accurate seek in the byte-stream demuxers.** `tsdemux`, `matroskademux`, `flvdemux`, `oggdemux` and `fmp4demux` restart a seek at the first keyframe at or after the target. Keep the units from the keyframe before the target and send a segment starting at the target, as `Mp4Src` does, in the shared `DemuxSeek`.
- **FLV:** Speex decode (no Speex encoder exists anywhere to build a validated
  decode vector, and gst's header-in-tag layout is rejected by libavcodec, so
  wiring a decoder would be an unvalidated claim).
- **AV1 in MPEG-TS:** write the AOM spec's 'AV01' `registration_descriptor`
  instead of GStreamer's 'AV1G' once any demuxer reads the former (GStreamer's
  `tsdemux` activates no program for an 'AV01' stream, and ffmpeg's muxer writes
  no descriptor at all, so it identifies neither).
- **Cover art writing:** write `Tag::Image` back as an MP4 `covr` item, a FLAC
  `PICTURE` block, a VorbisComment `METADATA_BLOCK_PICTURE` field and an ID3
  `APIC` frame.
- **WebVTT track writing** (mkv, mp4 `wvtt`): blocked on a reference peer.
  ffmpeg reads only the WebM `D_WEBVTT/*` carriage (different block payload)
  and cannot write WebVTT into MP4 at all; reading both stays supported.

## Codecs

- **Pure-Rust / wasm decode** to drop the ffmpeg FFI: VP8 / VP9 decode and a
  pure-Rust Opus path (see the roadmap for why both are blocked).

## Colorimetry

- Write the caps colorimetry into the colour description the platform encoders
  emit (Media Foundation, VideoToolbox, MediaCodec), so a tagged stream
  survives a re-encode there too.
- Tag the raw output of the platform decoders that still leave colorimetry
  unknown (Media Foundation, VideoToolbox, MediaCodec, and `VaapiDec` on Intel).

## Transforms and effects

- **`textoverlay` font backend:** font-variation axes beyond `wght` on the
  shaped horizontal path (cosmic-text exposes only weight); vertical-mode
  shaping if cosmic-text ever grows writing modes.
- Add a carrier for non-default channel orders when a source needs an
  interleave order outside the per-count `ChannelLayout` convention.

## Metadata (FrameMeta / AnalyticsMeta)

- `NvEnc` AV1 encode (needs RTX 40-series hardware).

## Clock-synchronised presentation

- **KMS vblank reconciliation** + Wayland frame-callback co-scheduling. Needs a
  DRM/KMS presentation sink (current `WaylandSink` is SHM software). Validate on
  a real display.
- **PTP clock polish** (not blocking): a live multi-machine / `ptp4l`-grandmaster
  soak of `PtpClient` (host/root/reference-gear gated); a direct PHC
  (`/dev/ptpN`) read; hardware RX/TX timestamping for uncompressed ST 2110-20
  timing; BMCA/Announce, peer-delay, unicast.
- **ST 2110 media transport:** wire compliance of -20/-22/-30/-40 + multicast
  validated against reference gear (built from the RFCs, not yet
  interop-tested).

## Properties / introspection / DSL

- **Bare fan-out refs into `mp4mux`.** `p. ! queue ! m.  p. ! queue ! m.  mp4mux name=m` fails with no free stream of a kind mp4mux takes, because bare-ref matching reads the single-input `mp4mux` template (video only) and the fan-in muxer factories declare no input caps. Declare the fan-in muxers' input caps.
- A GUI / tooling introspection surface beyond the text dump.
- `g2g-discover`: read audio-only MP4 and native FLAC / MP3 files, which it
  cannot probe, so their tags show.

## GStreamer element coverage

The GStreamer names `g2g-inspect --gst` still answers with "unknown" or a hint,
grouped by what a port would need. Each line is a future element (or family)
unless it says otherwise.

- **Audio encoders:** `flacenc`, `vorbisenc`, `lamemp3enc` / `twolamemp2enc`,
  `fdkaacenc` on Linux, `webpenc`, `speexenc`, `wavpackenc`, `gsmenc`, `amrnbenc`
  / `voamrwbenc`, `sbcenc`, `lc3enc`, `ldacenc` (Bluetooth codecs pair with
  `a2dpsink` / `avdtpsink` / `avdtpsrc`).
- **Audio decoders:** `speexdec`, `wavpackdec`, `gsmdec`, `amrnbdec` / `amrwbdec`,
  `sbcdec`, `lc3dec`, `musepackdec`, `sfdec`, `gmedec` / `openmptdec` / `modplug`,
  `dvdlpcmdec`, `dsdconvert`, `sirendec` / `isacdec`.
- **Audio parsers:** `amrparse`, `dcaparse`, `sbcparse`, `wavpackparse`,
  `vorbisparse`, `theoraparse`, `icydemux` (SHOUTcast metadata in `httpsrc`).
- **Audio filters:** `audiointerleave`, `audiolatency`, `rganalysis` /
  `rgvolume` / `rglimiter`, `bs2b`, `freeverb`, `pitch` / `bpmdetect`,
  `spanplc`, `accurip`, `chromaprint`.
- **Audio visualisers:** `wavescope`, `spacescope`, `spectrascope`, `synaescope`,
  `goom` / `goom2k1`.
- **Video parsers:** `h263parse`, `h266parse`, `diracparse`, `jpeg2000parse`,
  `jifmux`, `h264timestamper` / `h265timestamper`, `codec2json`
  (`h2642json` ...).
- **Video codecs:** `openh264enc` / `openh264dec`, `svtav1enc`, `mpeg2enc`,
  `openjpegenc` / `openjpegdec`, `openexrdec`,
  `gdkpixbufdec`, `rsvgdec`, `flxdec`, `vmncdec`,
  `bayer2rgb` / `rgb2bayer`, `codecalpha` (`alphacombine`, `codecalphademux`,
  `vp8alphadecodebin`), `jp2kdecimator`.
- **Containers:** `mxfdemux` / `mxfmux`, `asfdemux` / `asfmux` / `asfparse`,
  `rmdemux` / `rademux`, `mpegpsmux`, `mplex`, `atscmux`, `tsparse`,
  `matroskaparse`, `oggparse` / `ogmaudioparse` / `ogmvideoparse` /
  `ogmtextparse` / `oggaviparse`, `3gppmux` / `ismlmux` / `mj2mux` (mp4mux
  brands), `qtmoovrecover`, `avisubtitle`, `gdppay` / `gdpdepay`, `pcapparse` /
  `irtspparse`, `bz2enc` / `bz2dec`, `midiparse`.
- **Subtitles / captions:** `ttmlparse` / `ttmlrender`, `assrender`,
  `textrender`, `dvbsubenc`, `dvdsubparse`, `cea608mux` / `cc708overlay`,
  `h264ccinserter` / `h265ccinserter` / `h264ccextractor` / `h265ccextractor`.
- **Line-21 VBI captions:** `line21encoder` / `line21decoder`. Writing the
  waveform means the biphase signal itself (7 clock run-in cycles at 32x the
  line rate, 240 ns rise/fall shaping, a 50 IRE swing) and reading it back means
  a bit slicer with threshold tracking. Both also need a raw-video convention
  that includes the VBI lines: `Caps::RawVideo` here is active picture, and no
  source produces the 720x525 interleaved frame line 21 sits in.
- **Overlays:** `cairooverlay`, `rsvgoverlay`, `gdkpixbufoverlay`, `qroverlay` /
  `debugqroverlay`, `overlaycomposition`, `faceoverlay`, `zxing`,
  `objectdetectionoverlay` (vs `analyticsoverlay`).
- **Video transforms:** `interlace`, `ivtc` / `combdetect`, `fieldanalysis`,
  `smpte` / `smptealpha`, `shapewipe`, `alphacolor`, `lcms`,
  `simplevideomark` / `simplevideomarkdetect`,
  `videoframe-audiolevel`, `timecodestamper` / `avwait`, `audiosegmentclip` /
  `videosegmentclip`, `navigationtest`.
- **Flow / bins:** `uritranscodebin` (a URI in, an encoding profile out),
  encoder presets (a named quality / speed set an encoding profile's stream part
  selects, which needs a `preset` property on the encoders first), `autoconvert`
  / `autovideoconvert` / `autodeinterlace` / `autovideoflip`, `switchbin`,
  `insertbin`, `roundrobin`, `playsink` / `streamsynchronizer` (playbin
  internals), `nlecomposition` /
  `nlesource` / `nleoperation` / `nleurisource` and `gessrc` / `gesdemux`
  (editing), `camerabin` / `viewfinderbin` / `wrappercamerabinsrc`, `bin` /
  `pipeline` as launch keywords, `msesrc`.
- **IPC:** `proxysink` / `proxysrc`, `intervideosink` / `intervideosrc` /
  `interaudiosink` / `interaudiosrc` / `intersubsink` / `intersubsrc`,
  `ipcpipelinesink` / `ipcpipelinesrc` / `ipcslavepipeline`.
- **Network:** `curlhttpsrc` and the `curl*sink` uploaders, `souphttpclientsink`, `giosrc` /
  `giosink` / `giostreamsrc` / `giostreamsink`, `shout2send`, `sctpenc` /
  `sctpdec` outside WebRTC, `multifdsink` / `multisocketsink` / `socketsrc`,
  `netsim`, `avtp*` (IEEE 1722), `rtspwms`, `asteriskh263`, `aesenc` / `aesdec`.
- **Devices:** `dvbsrc` / `dvbbasebin`, `dvdreadsrc` / `rsndvdbin`, `cdparanoiasrc`
  / `cdiocddasrc`, `dc1394src`, `rfbsrc`, `uvch264src` / `uvch264mjpgdemux`,
  `uvcsink`, `v4l2sink`, `v4l2radio`, `fbdevsink`, `osssrc` / `osssink` /
  `oss4src` / `oss4sink`, `openalsrc` / `openalsink`, `alsamidisrc`,
  `vaapisink` / `vaapipostproc` / `vapostproc` / `vacompositor` /
  `vadeinterlace` / `vaapidecodebin`.
- **Debug:** `fakevideodec`, `testsink` / `testsrcbin` / `videocodectestsink`,
  `clockselect`, `compare`, `cpureport`, `navseek`, `pushfilesrc`,
  `flitetestsrc` / `festival`, `ssdobjectdetector`.
- **gst-plugins-rs:** `gifenc` / `gifdec`, `gopbuffer`, `textwrap`,
  `jsongstenc` / `jsongstparse`, `gstregex`, `zlibcompress` / `zlibdecompress`,
  `colordetect`, `videocompare`, `uriplaylistbin`, `ndisrc` / `ndisink`,
  `originalbuffersave` / `originalbufferrestore`.

## Browser / Wasm

- Raw-`web_sys` WebGPU path (only if the GPU-resident browser claim is revived):
  external-texture import + compute + `ort.Tensor.fromGpuBuffer` on one
  ORT-owned `GPUDevice`. Large, browser-unverifiable on the dev host.

## ML

- Detector on the Edge TPU is blocked device-side: this Pixel's older Android ORT
  NNAPI EP rejects YOLO's op set (int8-weight initializers, SiLU `Mul` QDQ
  "unsupported quantized type", and an `AddNnapiSplit` divide bug on the C3k2
  channel split); a simple conv stack (MobileNet) places fine. Needs a newer
  ORT build or a TPU-friendly detector (SSD-MobileNet-style, conv-only).
- Hand-rolled GPU inference path: masked / causal attention + KV cache, if an
  autoregressive use case ever appears (unmasked full attention is in).
- D3D11 decoder surface import into `WgpuPreprocess` (bind the surface directly
  into the compute pass, the Windows counterpart of the dma-buf import).
- Run the QNN and CoreML execution providers on Qualcomm and Apple hardware.
