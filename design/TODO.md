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

- **CI never runs m960 or m961 with `ffmpeg`.** The conformance job runs them with `rav1d` alone and linux-full enables neither rav1d nor dav1d, so a regression that only shows with both decoders built stays invisible. Add an `ffmpeg,rav1d` run of both files to a job that has the libav packages.
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
  RADV device: the Vulkan API dump layer (needs root to install) or an ffmpeg
  built to print its `StdVideo*` structs, since Mesa 25.3.6's `RADV_DEBUG=dumpibs`
  prints nothing for video. NVIDIA Turing cannot decode it at all (minimum H.265
  coded extent 144x144).
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

- **HLS init segment byte range.** `HlsPlaylists::init_segment` ignores `#EXT-X-MAP:BYTERANGE`, so a single-file CMAF variant probes the whole file, up to `MAX_SEGMENT_BYTES`. Fetch only the named range.
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
- **Named demuxer before explicit elements:** `filesrc ! matroskademux ! h264parse ! …`
  negotiates against the demuxer's default `stream=vp9`, so an H.264 file needs
  `stream=h264`. The demuxer has to read the file's tracks during negotiation.
- **TS positions with B frames:** `tsdemux` opens its segment at the first DTS, so
  positions read high by the first PTS minus the first DTS until the first seek.
  Needs a signed `Segment` time offset in g2g-core.
- **Corpus sweep bugs** (calliope TESTING techniques 21-23, paths under
  `~/.cache/calliope` on hercules):
  - TS seeks count from the video's first pts instead of the earliest stream's
    (`gst-medias/defaults/bipbop/gear2/fileSequence2.ts`), and
    `ac3/mp3ac325-4864-small.ts --seek 0.406011` finds no frame.
  - `h264/H264_might_overflow.mkv --seek 1.086` finds no frame, where ffmpeg
    lands on the keyframe at 1.086 s.
  - No seek finds a frame in `gst-medias/defaults/webm/opus_vp9-alpha.webm`.
  - `parsebin` fan-out fails on WAV, VMnc and tagged AVI, and FLV VP6 (no caps
    overlap), and on IVF, Ogg, `h264_intra_first-small.flv` and tagged-codec
    AVI (no free output pad `video_0`).
  - `flvmux` writes an AAC sequence header no decoder accepts
    (`fate-suite/aac/al04_44.mp4`).
  - `mpegtsmux` accepts USAC and writes a file nothing decodes
    (`fate-suite/aac/usac/*`).
  - LATM AAC loses samples through flvmux and mpegtsmux and fails into mp4mux
    and matroskamux (`fate-suite/aac/latm_stereo_to_51.ts`, `mpegts/loewe.ts`).
  - Remux changes samples or frame counts: bipbop TS AAC into mp4mux,
    `tron_en_ge_aac_h264.ts` AAC, `fragmented_nonseekable_sink.mp4` and
    `twofields_packet.mp4` into mkv and ts, `test7_cut.mkv` into mp4.
  - Remux fails mid-run or writes an undecodable file: `opus_vp9-alpha.webm` and
    `loewe.ts` HEVC into mp4mux and matroskamux, `test-8-7.1.opus-small.ts` into
    matroskamux and mpegtsmux, `two_first_slice.mp4` into matroskamux,
    `al04sf_48.mp4` into mpegtsmux.
  - Stream listing misses the Dolby Vision enhancement layer of
    `mpegts/dovi-p7.ts`, the PIDs a PMT change adds in `mpegts/pmtchange.ts`, and
    the second MP2 stream of `sample_mpeg_program_stream_scr.mpg`.
  - The `flvmux` duplicate-track, `avimux` second-video and `mp4mux` CMAF
    refusals still return a bare `CapsMismatch`: log what they refuse.
  - `mpeg4/packed_bframes.avi` comes out with pts going backwards: two VOPs per
    chunk need splitting. `h264/mixed-nal-coding.mp4` gives 5 frames where
    ffprobe gives 10, and `hevc/two_first_slice.mp4` 35 where gst and ffmpeg give
    34, repeating 2.000 s.
  - A TS or PS stream whose first PES carries no pts puts its first units at 0
    where ffmpeg back-fills them from the next stamp (`mpeg2/t.mpg`, 0.06 s): hold
    the leading units until the PID's first stamp, then back-fill.
  - `mpegts/loewe.ts` LATM loses two frames mid-stream (14 where ffprobe gives
    16), and `probe-format/roundup997` decodes a first MP2 frame ffmpeg rejects
    as "Header missing".
  - qdraw in MOV decodes 2 of 12 frames (`quickdraw/Airplane.mov`).
  - WavParse with a tagged codec cuts payloads at the input chunk: XMA2 packets
    are 2048 bytes but `block_align` is 6 (`xma2/xma2-multistream-tail.xma` gives
    512, 18952 and 30720 bytes of PCM at blocksize 188, 4096 and 65536). AC-3 in
    WAV (`diatonis_invisible_order_anfos_ac3-small.wav`) puts frame 1 at 0.371 s
    where ffmpeg gives 0.035 s.
  - `flvdemux` ADTS-wraps AAC whose SBR output rate equals the core rate, so it
    decodes at double rate. It needs the raw-config path the MP4 demuxers use.
  - `aac/aac-sce-in-stereo.mp4` is 23.2 ms late on every frame,
    `ac3/mp3ac325-4864-small.ts` gives 4 frames where ffmpeg gives 5,
    `mp3-conformance/sin1k0db.bit` 317 where it gives 318, and
    `mov/mov_neg_first_pts_discard_vorbis.mp4` is 2.0 s ahead.

## Codecs

- **`audioconvert ! audio/x-raw,format=S16LE` with rate and channels open fails the re-solve** with `CapsMismatch`. A capsfilter that fixes only the format should let the rate and channels pass through.
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

- **Bare refs into `textoverlay` fill its inputs in link order.** `TextOverlayN` expects video on input 0, so `d.text ! o.  d.video ! o.` with bare `o.` refs puts text on the video input. Assign a bare muxer-side ref to the input whose kind its stream has.
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
