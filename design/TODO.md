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

## Remote desktop

The plan is [remote.md](remote.md).

- `ximagesrc` for unattended X11 capture, keyboard and pointer events from the
  Wayland window sinks, and a Linux `g2g-desktop` host and client.
- `dxgisrc` (system NV12, then a CUDA surface for `nvenc`) and Windows
  `SendInput`.
- `display-index` on `screencapturesrc`, and a macOS host through `vtencode`.
- Cursor, clipboard, grab, and `force-keyframe` on `vtencode`, `mfencode`, and
  `ffmpegenc`, pulsed from the duplex PLI reverse channel.
- Gamepad injection on Linux and Windows, and a Windows service that injects
  on the secure desktop.
- A Chrome client on the same signaling and the same data channel.
- OIDC in front of the bearer token, a `g2g://` URL, and install packages.

## Architecture guarantees (validation-first)

- **Grow the conformance matrix.** Persist evidence from the resource-owning
  tests still uncovered as they are validated (`ffmpegvaapidec` on Intel and AMD, the Android and
  macOS device paths). Get the device-tagged `Hardware` rows into CI by wiring a
  runner that has the hardware: a `Hardware` row can only come from a run on the
  device, so a runner without a GPU or a camera will never produce one.

## Release binaries

The release `g2g-launch` binaries and wheels must start on a host with no media
libraries installed.

- **Bare-host check.** Before publishing, run each release binary on a GPU-less
  host on an H.264 + AAC MP4 and a camera to WebRTC pipeline, and check that
  `decodebin` picks a software decoder rather than `vulkanvideodec` on Linux.

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
- **ESP32-S3 on a XIAO ESP32S3 Sense (Xtensa on-device).** Build with
  Espressif's Rust toolchain (`espup`) for `xtensa-esp32s3-none-elf`.
  - Add the target to the footprint report (`tools/footprint.py` parses only
    ARM and RISC-V disassembly).
  - Boot the heap-free pipeline on Espressif's QEMU fork (`-machine esp32s3`),
    as `tools/qemu-check.sh` does for Cortex-M.
  - PDM microphone source on esp-hal's I2S PDM receive mode, then the keyword
    graph live on the board, reproducing the pinned scores in
    `g2g-mcu/tests/fixtures/keyword`.
  - OV2640 camera source on esp-hal's `lcd_cam` camera driver, with the
    sensor's I2C register setup ported from Espressif's `esp32-camera`. Then
    96x96 MobileNet v1 person detection through `ImageResize`, `Int8Quantize`
    and MicroFlow, with per-frame time measured.
  - Detectors MicroFlow cannot run (MobileNet v2, FOMO): LiteRT for
    Microcontrollers through `CTensorModel`, with Espressif's ESP-NN kernels
    if they build outside ESP-IDF.
  - Stream the microphone over Wi-Fi to a host graph ending in
    `whispertranscribe`.
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
  RADV device: the Vulkan API dump layer (needs root to install) or an ffmpeg
  built to print its `StdVideo*` structs, since Mesa 25.3.6's `RADV_DEBUG=dumpibs`
  prints nothing for video. NVIDIA Turing cannot decode it at all (minimum H.265
  coded extent 144x144).
- **Windows GPU decode on D3D12:** `d3d12h265dec` and `d3d12av1dec` on the D3D12
  video decode API, with their DXVA structs sized against `dxva.h`.
  - Move the H.265 and AV1 parsing and reference state out of `vulkanvideo.rs`
    with their decoders, as the H.264 code is in `h264parse` and `h264dpb`.
  - Run the JVT conformance streams through `d3d12h264dec` with a calliope
    adapter on japan.
  - Run `m2210` on japan: its access-unit split in `tests/common/annexb.rs`
    changed after its last run.
  - Pipeline `d3d12h264dec`: the CPU waits on a fence after every picture,
    which is most of its 6.7 ms per 1080p frame.
  - Rebuild the decoder and heap on a mid-stream size change instead of
    refusing the stream.
  - The Media Foundation decoder MFT with a DXGI device manager stops producing
    output after about 120 frames, so `MfDecode` decodes in software only.
- **`mfh265dec` without an HEVC decoder installed** fails with a bare
  `Hardware(MediaFoundation(0x80040154))`. Find the call that fails and log
  that the HEVC Video Extensions are missing.
- **Decoder features:**
  - H.264 field pictures (PAFF) on `VulkanVideoDec` and `d3d12h264dec`.
  - Field-coded H.265 (`field_seq_flag`): an alternate-field interlace mode in
    caps, full-frame height and frame rate from the parsers, and decoders that
    pair fields.
  - H.264 custom scaling matrices on `VulkanVideoDec` and `d3d12h264dec`.
  - AV1 and VP9 on `mediacodecdec`.
  - HEVC 4:4:4 and 12-bit on `nvdec`.
- **RTP receive:**
  - Place frames from RFC 6051 in-band NTP-64 before the first sender report,
    for faster sync.
  - VP8, VP9, AV1 and JPEG RTP payloads in `udpsrc`, and MJPEG cameras in
    `rtspsrcn` (retina has a JPEG depacketizer).
  - RED (RFC 2198) receive.
- **RTSP:**
  - retina joins an absolute `a=control:/path` onto the base URL. Report it
    upstream.
  - RTSPS in `rtspsrcn` (retina has no TLS) and `rtspclientsink`.
## CUDA / display

- Run `CudaKmsSink` from a bare VT with DRM master and verify the in-tree
  `// VERIFY:` points.

## Heterogeneous devices

- Pick the cheapest converter by `DomainTransferCost` in the domain-converter
  splice, with each converter's cost measured, once two converters compete on
  one edge (a dma-buf import into wgpu against a download and an upload, or a
  peer copy against staging through the host).
- Peer copies between two GPUs of one vendor (`hipMemcpyPeerAsync`,
  `cuMemcpyPeerAsync`), validated on a two-GPU host.
- Report `gpu_device()` from the other CUDA elements (`cudaupload`,
  `cudadownload`, `cudaglsink`, `ffmpegdec` on NVDEC) and the wgpu elements.
- Placement: time `nvdec`'s decode engine in `--profile` (a CUDA counterpart of
  `HipSync` after it), charge copies on a discrete GPU to its copy engines apart
  from its compute, and account for the host memory contention that makes a
  cross-GPU copy take 3.9 ms per 1080p frame in a running pipeline against the
  2.8 ms the model charges.
- Wait for a dma-buf frame's fence on the GPU in `hipupload` once ROCm supports
  semaphore import on Linux.

## Egress / transports

- **RTP over QUIC (RoQ):** implement after the draft becomes an RFC with an
  assigned ALPN. Candidate peers: mengelbart/roq (Go), meetecho/imquic.
- **RTMP authentication:** publish through `rtmpsink`'s Adobe and Limelight
  (llnw) auth to a real Wowza or Limelight server.
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
- **RIST bonding test flake:** `ristsender_broadcast_bonding_feeds_ristsrc`
  failed once in 50 runs on hercules, after 2.4 s. Capture its failure output
  and fix the cause.
- Pass video and audio between processes on Windows. `shmsink`, `unixfdsink`
  and the DMABUF / CUDA IPC elements are Linux only.
- **io_uring receive for ST 2110.** Profile an uncompressed 1080p60 receive
  (about 200k packets a second) on the `recvmmsg` path first. If system calls
  still show up, try multishot receive into provided buffer rings on Linux,
  keeping the `recvmmsg` path as the fallback. tokio has no io_uring backend,
  and Android apps and Docker's default seccomp profile block it.

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
- ONVIF metadata tracks with the Streaming Specification's
  `vnd.onvif.metadata+gzip` encoding name: once a retina release carries
  scottlamb/retina PR 165, map that name to gzip in `onvif_metadata_gzip`
  instead of skipping the track.
- Bump retina past 0.4.21 once a release carries its commit d163dd8: an SDP
  `m=` line with no payload type panics the client in 0.4.21.
- Bump retina once a release carries scottlamb/retina PR 167: before it, an
  RTSP camera that never sets the RTP marker bit grows rtspsrc's memory
  without bound.
- Bump retina once a release carries scottlamb/retina PR 170: before it, an
  SDP `a=rtpmap:96 H264/0` fails the whole rtspsrc session at PLAY, and
  `a=rtpmap:96 H264` with no clock rate drops the stream.
- ONVIF metadata tracks with an EXI coding (`vnd.onvif.metadata.exi.onvif`,
  `vnd.onvif.metadata.exi.ext`): g2g has no EXI decoder, so they are skipped.
- Run the Windows (`mfdevice` / `wasapidevice`) and macOS (`avfdevice` /
  `coreaudiodevice`) device providers on a real host: enumeration against
  attached hardware, endpoint selection by id through each element's `device`
  property, and the `IMMNotificationClient` hotplug path. Both are
  compile-checked only (CI cross-compiles them; the runners have no camera and
  no way to replug one).
- SDI audio capture for Blackmagic Design DeckLink (`decklinkaudiosrc`),
  sharing its input device with the `decklinkvideosrc` of the same number.
- `decklinkvideosrc` properties GStreamer has: `output-cc` and `output-afd-bar`,
  which attach captions and AFD/Bar from VANC as their own metas, and
  `timecode-format`, which needs a timecode meta. `mode=auto` should also pick
  the widescreen SD modes from AFD, as GStreamer does.
- `decklinkvideosrc` should hand downstream the SDK's frame instead of a copy.
- Run `decklinkvideosrc` on a board: capture, input format detection, VANC and
  the board clock have only run against a mock device.

## Sinks

- Validate `wasapisink` U8, S24, and S32 acceptance on a Windows host.
- Run `wasapisink` on a Windows host with a `device` set while it plays and
  with `continue-on-error` against an unplugged endpoint.
- SDI output for Blackmagic Design DeckLink (`decklinkvideosink`,
  `decklinkaudiosink`) on Windows and macOS.
- Run `decklinkvideosink` and `decklinkaudiosink` on a board: scheduled
  playback, the board clock, audio against video and VANC output have only run
  against a mock device.
- An example that shows pipeline video inside a Qt/QML window.
- Move `vulkanhdrsink` off its raw Vulkan swapchain onto a wgpu surface, using
  wgpu 30's `SurfaceConfiguration::color_space` for the HDR10 / scRGB output.
- Shorten `wgpusink`'s first-frame startup: it creates the wgpu device and
  surface on the first frame, which takes about 1.3 s on a Windows laptop, so a
  live camera shows an empty window and late-drops about 40 frames. Measure the
  split, then open the device at `configure_pipeline`.

## Containers

- **FLV:** Speex decode (no Speex encoder exists anywhere to build a validated
  decode vector, and gst's header-in-tag layout is rejected by libavcodec, so
  wiring a decoder would be an unvalidated claim).
- **Enhanced RTMP:** enhanced video in `flvdemux` / `eflvmux` (H.265, AV1 and
  VP9 fourccs, multitrack `video_N`), the Opus, FLAC and AC-3 audio fourccs, and
  `eflvmux`'s `flv-track-mode` pad property.
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
- **Corpus sweep bugs** (calliope TESTING techniques 21-23, paths under
  `~/.cache/calliope` on hercules):
  - Demuxers drop streams gst's parsebin exposes: DTS in `filter/242_4.mkv`, TTA,
    pcm_dvd LPCM in `coolitnow-partial.vob`, and the xan_dpcm audio of
    `wc4-xan` AVIs.
- **MOV-tagged DV in `matroskamux`:** a `.mov` DV stream (`dvc `) goes out as
  `V_QUICKTIME`, which gst's `matroskademux` cannot read. Write a MOV tag that
  has an AVI fourcc as `V_MS/VFW/FOURCC` with a built BITMAPINFOHEADER, as
  ffmpeg does.
- **Opus end trim in fragmented MP4:** the fragmented `mp4mux` layout writes
  its `moov` before the stream ends, so the `elst` duration is 0 and the end
  trim is only in the last sample's duration, which gst's `qtdemux` ignores (a
  48144-sample Opus stream decodes to 48648 samples in gst). Rewrite the `moov`
  at EOS with the `elst` duration when the sink can seek back. ffmpeg's `mov`
  demuxer reads no end trim from any MP4, so it cannot check this. ffmpeg also
  applies no start trim from that `elst`, so a fragmented remux of a gapless
  MP3 keeps its 1105-sample encoder delay in ffmpeg.
- **`clap` in `mp4mux`:** the MP4 reader crops to a `clap` box, and `mp4mux`
  writes none, so a remuxed file decodes uncropped.
- **ST 2038 elements:** `st2038ancdemux` should add a port when a stream no pad
  names appears, once a launch line can link a pad that appears at run time.
- **Demuxer features:** VP8 and VP9 alpha from Matroska BlockAdditions, ONVIF
  `metx` tracks and `cstb` start times in MP4, AVC-Intra fourccs in MOV, and
  AC-4 and E-AC-3 in MPEG-TS.

## Codecs

- **V4L2 mem2mem decode and encode**, stateful and stateless, the hardware
  codec path on Raspberry Pi and most ARM SoCs (gst `v4l2h264dec`,
  `v4l2h264enc`, `v4l2slh264dec`).
- Run `vaav1enc` on a GPU with VA-API AV1 encode (AMD RDNA3, Intel Arc).
- **Pure-Rust / wasm decode** to drop the ffmpeg FFI: VP8 / VP9 decode and a
  pure-Rust Opus path (see the roadmap for why both are blocked).

## Colorimetry

- Write the caps colorimetry into the colour description the platform encoders
  emit (Media Foundation, VideoToolbox, MediaCodec), so a tagged stream
  survives a re-encode there too.
- Tag the raw output of the platform decoders that still leave colorimetry
  unknown (Media Foundation, VideoToolbox, MediaCodec).

## Pixel aspect ratio

- Write the VUI sample aspect from `openh264enc` once the `openh264` crate can
  set `bAspectRatioPresent`, and carry the caps ratio through `mfencode`,
  `vtencode`, `mediacodecenc` and `webcodecsencode` into their bitstreams.
- Letterbox `kmssink` on an overlay plane with source and destination
  rectangles, and size and letterbox `glsink`, `cudaglsink`, `d3d11videosink`
  and `metalvideosink` by display aspect behind `force-aspect-ratio`.
- Keep the display aspect through `videoconvertscale` and the GPU scalers as
  `videoscale` does, and pick the output width that keeps it when downstream
  pins a ratio.
- Map the caps ratio to the widescreen SD modes of `decklinkvideosink`.

## Transforms and effects

- **`textoverlay` font backend:** font-variation axes beyond `wght` on the
  shaped horizontal path (cosmic-text exposes only weight); vertical-mode
  shaping if cosmic-text ever grows writing modes.
- Add a carrier for non-default channel orders when a source needs an
  interleave order outside the per-count `ChannelLayout` convention.

## Metadata (FrameMeta / AnalyticsMeta)

- `NvEnc` AV1 encode (needs RTX 40-series hardware).
- Return `Transform::Encode` from `meta_transform` in `MediaCodecEnc`,
  `MfEncode`, `VtEncode` and `WebCodecsEncode`, so frame metadata (captions,
  HDR10+) reaches a `ccinsert` or `seiinserter` after them. Each needs a run on
  its platform.

## Clock-synchronised presentation

- **KMS vblank reconciliation** + Wayland frame-callback co-scheduling. Needs a
  DRM/KMS presentation sink (current `WaylandSink` is SHM software). Validate on
  a real display.
- **PTP clock polish** (not blocking): a live multi-machine / `ptp4l`-grandmaster
  soak of `PtpClient` (host/root/reference-gear gated); a direct PHC
  (`/dev/ptpN`) read; hardware RX/TX timestamping for uncompressed ST 2110-20
  timing; BMCA/Announce, peer-delay, unicast.
- **Skew slaving on a real device:** run an audio sink under a `mpegtslivesrc`
  clock from a sender whose clock runs measurably off the receiver's, and
  confirm the corrections hold sync over an hour. Run `aaudiosink`'s playout
  clock on a device: CI only cross-compiles it.
- **ST 2110 media transport:** wire compliance of -20/-22/-30/-40 + multicast
  validated against reference gear (built from the RFCs, not yet
  interop-tested).
- **Pace `surfacesink`.** It offers no clock, so with no other clock provider
  in the graph nothing paces it: `videotestsrc` reaches the screen at the
  display's 60 fps instead of 30. Offering a clock fixes the bars but blanks
  `camera2src ! videoflip ! videoconvert ! surfacesink`, every frame consumed
  and none drawn. `camera2src` stamps sensor time since boot, and restamping it
  through `CaptureAnchor` with one frame of live latency did not help. Log the
  sink's per-frame lateness on the Pixel before the next attempt.

## Properties / introspection / DSL

- A GUI / tooling introspection surface beyond the text dump.
- `g2g-inspect` and `docs/elements.html` leave out the elements registered only
  as fan-out sources (`rtspsrcn`, `rtspserversrcn`, `uridecodebin`):
  `g2g-inspect rtspserversrcn` answers "No such element".
- Register URI schemes for bare `scheme://` launch stages beyond `ice+http(s)`:
  `udp`, `srt`, `rtsp`, `rtmp`, `http(s)`, and `file` (as a path for `filesrc` /
  `filesink`), each with the property its element takes the URI in.
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
- **Audio parsers:** `dcaparse`, `sbcparse`, `wavpackparse`,
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
- `ortinfer execution-provider=migraphx` on `ball11n.onnx` aborts at process exit
  with heap corruption in libgcc's frame deregistration, on both the GPU-tensor
  and the system-memory input. Find the cause.
- Choose the GPU `wgpupreprocess` opens for system-memory input, so a hybrid host
  can preprocess on its AMD GPU for MIGraphX without `VK_DRIVER_FILES`.
