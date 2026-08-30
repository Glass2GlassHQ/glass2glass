# DESIGN_TODO

Outstanding work, tracked against the architecture in [DESIGN.md](DESIGN.md).
This file is a terse catalogue of open tasks only. Completed work and the
rationale for shipped architecture live in [DESIGN.md](DESIGN.md) and
[CHANGELOG.md](CHANGELOG.md), not here.

## Roadmap (high level)

Highest leverage first:

1. **Platforms.** macOS: camera / screen capture validation on a permitted
   Mac.
2. **Egress / transports.** Real-peer FlexFEC interop when a peer
   implementation is available (GStreamer here lacks `rtpflexfecenc`).
3. **Depth.** Pure-Rust codec paths to cut the remaining ffmpeg FFI reliance.
   No credible pure-Rust VP8 / VP9 decoder exists (a
   libvpx-FFI `VpxDec` stays deferred: it would only duplicate the ffmpeg
   path), and the one complete pure-Rust Opus, `opus-rs` 0.1.26, fails the
   RFC 8251 vectors (re-run `tools/opus-rs-gate` to revisit on a new release).
   `VulkanVideoDec` residuals: AMD / Intel validation runs (see "Receive /
   decode").
4. **Browser demo (speculative product path).** A deployed reference app for the
   in-browser `ort-web` ONNX chain, plus a native sibling running the same graph.
   The GPU-resident in-browser chain is not achievable from idiomatic Rust (wgpu
   can't import a WebCodecs `VideoFrame` as an external texture or adopt ORT's
   device on wasm); it would need raw `web_sys` WebGPU + hand-rolled
   onnxruntime-web bindings.

## Architecture guarantees (validation-first)

- **Grow the conformance matrix.** Persist evidence from the resource-owning
  tests still uncovered as they are validated (`vaapi` decode, the Android and
  macOS device paths). Get the device-tagged `Hardware` rows into CI by wiring a
  runner that has the hardware: a `Hardware` row can only come from a run on the
  device, so a runner without a GPU or a camera will never produce one.

## Alloc-optional (heap-free) MCU core

- **On-device `Hardware` rows (ARM).** Finalize the RCC, clock, RMII pin, and
  RTP destination settings in `examples/g2g-stm32h743`, then run the flagship
  audio graph and `HwJpegDec` on a NUCLEO-H743ZI2. Add the same conformance and
  timing evidence for an NXP i.MX RT board.
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

## Platform: macOS

- `AvfVideoSrc` / `ScreenCaptureSrc`: real capture validation on a Mac with a
  camera / screen-recording permission (the CI runner grants neither, so only
  the probe paths are validated).

## Receive / decode

- **`VulkanVideoDec` residuals.** Run the `vulkanvideo` GPU tests on AMD RADV
  and Intel ANV. Add multiplanar NV12 and `VulkanTexture` output domains.

## CUDA / display

- Run `CudaKmsSink` from a bare VT with DRM master and verify the in-tree
  `// VERIFY:` points.

## Egress / transports

- **RTP over QUIC (RoQ):** implement after the draft becomes an RFC with an
  assigned ALPN. Candidate peers: mengelbart/roq (Go), meetecho/imquic.
- **RTMP:** multiple NetStreams over one connection. Deferred by design: it needs
  a dynamic-arity multi-output `RtmpSrc` (the stream count is only known once the
  client `createStream`s at runtime), which collides with g2g's fixed-arity-from-caps
  model. Niche in practice (OBS / ffmpeg / CDNs publish one stream per
  connection); revisit only with a concrete need. Egress to a real CDN stays
  user-side.
- **WebRTC.**
  - A real LiveKit Cloud / TURN-relay run (genuine remote NAT + STUN/TURN on
    the LiveKit elements); then Janus / Kinesis as wanted.
  - FEC is blocked upstream (str0m has no FEC payload; loss recovery is
    NACK/RTX).
  - Data-channel loose ends: str0m surfaces no remote-close event, so EOS rides
    an explicit marker message; a WHIP/SFU-signalled data channel vs the P2P
    `SdpChannel` seam.
- **Remote graph carriers.** Add a native WebSocket server that pushes an
  unsolicited stream to `WsWireSrc`, a wrapper that remotes a whole `Bin`, a
  WebTransport datagram carrier, and a metadata-only response for remote
  transforms whose pixels are unchanged.

## Adaptive streaming (HLS / DASH)

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
- Run the Windows (`mfdevice` / `wasapidevice`) and macOS (`avfdevice` /
  `coreaudiodevice`) device providers on a real host: enumeration against
  attached hardware, endpoint selection by id through each element's `device`
  property, and the `IMMNotificationClient` hotplug path. Both are
  compile-checked only (CI cross-compiles them; the runners have no camera and
  no way to replug one).

## Sinks

- Add DMABUF zero-copy to `alsasink`, `pulsesink`, and `pipewiresink`.
- Validate `wasapisink` U8, S24, and S32 acceptance on a Windows host.

## Containers

- **ADPCM from a placeholder-stereo source:** `adpcmenc` takes mono only, and a
  source whose real layout arrives at runtime negotiates at the stereo
  placeholder, so `wavparse ! adpcmenc` fails the solve (and again on the runtime
  re-solve through an `audioconvert` pinned to mono). Needs either a
  channel-count-agnostic ADPCM path or a converter that renegotiates on the
  refinement.
- **FLV:** Speex decode (no Speex encoder exists anywhere to build a validated
  decode vector, and gst's header-in-tag layout is rejected by libavcodec, so
  wiring a decoder would be an unvalidated claim).
- **AV1 in MPEG-TS:** write the AOM spec's 'AV01' `registration_descriptor`
  instead of GStreamer's 'AV1G' once any demuxer reads the former (GStreamer's
  `tsdemux` activates no program for an 'AV01' stream, and ffmpeg's muxer writes
  no descriptor at all, so it identifies neither).
- **WebVTT track writing** (mkv, mp4 `wvtt`): blocked on a reference peer.
  ffmpeg reads only the WebM `D_WEBVTT/*` carriage (different block payload)
  and cannot write WebVTT into MP4 at all; reading both stays supported.
- **Matroska `ContentEncoding`:** chained encodings, bzip2 / lzo, and
  `ContentEncryption` stay refused (blocks forward as stored, flagged
  `unsupported_encoding`); zlib and header stripping are undone at demux.
- **Reordering-stream PTS from a single stamp:** an H.264 / H.265 transport
  stream that reorders stays unstamped until its second PES timestamp, since
  the picture-order-count step per frame is not declared anywhere and has to be
  measured across two stamps.

## Codecs

- **Pure-Rust / wasm decode** to drop the ffmpeg FFI: VP8 / VP9 decode and a
  pure-Rust Opus path (see the roadmap for why both are blocked).

## Transforms and effects

- **`textoverlay` font backend:** font-variation axes beyond `wght` on the
  shaped horizontal path (cosmic-text exposes only weight); vertical-mode
  shaping if cosmic-text ever grows writing modes.
- Add a carrier for non-default channel orders when a source needs an
  interleave order outside the per-count `ChannelLayout` convention.
- Add vertical cue rendering to `VelloTextOverlay`.
- Apply an `OrientationMeta` in `kmssink` (a DRM plane rotation), on the VAAPI
  VPP path and on the D3D11 VideoProcessor path, so those sinks advertise
  `Reconfigure::AbsorbOrientation` too.

## Compositor

- `wgpucompositor`: planar YUV.

## Metadata (FrameMeta / AnalyticsMeta)

- `NvEnc` AV1 encode (needs RTX 40-series hardware).

## Clock-synchronised presentation

- **KMS vblank reconciliation** + Wayland frame-callback co-scheduling. Needs a
  DRM/KMS presentation sink (current `WaylandSink` is SHM software). Validate on
  a real display.
- **A/V clock slaving** remaining pieces: extend the audio-master `DriftClock`
  discipline to `PipeWireSink` (blocked on the pinned `pipewire` 0.8 binding
  lacking `pw_stream_get_time`, plus playout accounting in its leaky realtime
  callback), and an on-display lip-sync soak on real hardware.
- **PTP clock polish** (not blocking): a live multi-machine / `ptp4l`-grandmaster
  soak of `PtpClient` (host/root/reference-gear gated); a direct PHC
  (`/dev/ptpN`) read; hardware RX/TX timestamping for uncompressed ST 2110-20
  timing; BMCA/Announce, peer-delay, unicast.
- **ST 2110 media transport:** wire compliance of -20/-22/-30/-40 + multicast
  validated against reference gear (built from the RFCs, not yet
  interop-tested).

## Properties / introspection / DSL

- A GUI / tooling introspection surface beyond the text dump.
- Text muxer fan-in in `parse_launch`.

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
  `webrtcdsp` / `webrtcechoprobe`, `spanplc` / `dtmfdetect` / `tonegeneratesrc` /
  `dtmfsrc`, `accurip`, `chromaprint`.
- **Audio visualisers:** `wavescope`, `spacescope`, `spectrascope`, `synaescope`,
  `goom` / `goom2k1`.
- **Video parsers:** `h263parse`, `h266parse`, `diracparse`, `jpeg2000parse`,
  `jifmux`, `h264timestamper` / `h265timestamper`, `codec2json`
  (`h2642json` ...).
- **Video codecs:** `openh264enc` / `openh264dec`, `svtav1enc`, `mpeg2enc`,
  `theoraparse`-side Theora decode, `openjpegenc` / `openjpegdec`, `openexrdec`,
  `pnmenc` / `pnmdec`, `gdkpixbufdec`, `rsvgdec`, `flxdec`, `vmncdec`,
  `bayer2rgb` / `rgb2bayer`, `codecalpha` (`alphacombine`, `codecalphademux`,
  `vp8alphadecodebin`), `jp2kdecimator`.
- **Containers:** `mxfdemux` / `mxfmux`, `asfdemux` / `asfmux` / `asfparse`,
  `rmdemux` / `rademux`, `mpegpsmux`, `mplex`, `atscmux`, `tsparse`,
  `matroskaparse`, `oggparse` / `ogmaudioparse` / `ogmvideoparse` /
  `ogmtextparse` / `oggaviparse`, `3gppmux` / `ismlmux` / `mj2mux` (mp4mux
  brands), `qtmoovrecover`, `avisubtitle`, `gdppay` / `gdpdepay`, `pcapparse` /
  `irtspparse`, `bz2enc` / `bz2dec`, `midiparse`.
- **Multi-file sources:** `splitmuxsrc` (each part its own container, so the
  parts have to be demuxed separately and their timestamps joined, unlike
  `splitfilesrc`'s byte join).
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
  `scenechange`,
  `videoanalyse` / `simplevideomark` / `simplevideomarkdetect`,
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
  `ipcpipelinesink` / `ipcpipelinesrc` / `ipcslavepipeline`, `unixfdsink` /
  `unixfdsrc` (fd-passing over a unix socket, the DMABUF-capable one).
- **Network:** `ristsink` / `ristsrc` and the `ristrtp*` / `ristrtx*` helpers,
  `curlhttpsrc` and the `curl*sink` uploaders, `souphttpclientsink`, `giosrc` /
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
  `clockselect`, `compare`, `debugspy`, `cpureport`, `navseek`, `pushfilesrc`,
  `flitetestsrc` / `festival`, `ssdobjectdetector`.
## Python-element host

- Add an explicit plain-text format override for files with no `.txt` extension.
- Include hosted Python class properties in inspection output without requiring
  `properties()` to return a `&'static` slice.

## Dynamic plugin loading

- Define how a distribution supplies `g2g-core` for offline plugin builds.

## Embedded

- Connect `EmbassyClock` to a HAL tick on real hardware.
- Wire a vendor HAL DMA-completion ISR into `StaticLendRing`.

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

## GStreamer bridge

- Add `WgpuBuffer` download or dma-buf export at the GStreamer bridge output.
- Add dma-buf zero-copy to `gstwrap`.

## Developer tooling

- **Per-element / per-link telemetry gaps.** Remaining `Observer` coverage:
  validate the dashboard live against an RTSP source.

## Audio decode-to-PCM QA

- calliope: AAC decode is not bit-exact across decoders, so it wants a golden /
  determinism check instead of the cross-engine differential Opus uses.
