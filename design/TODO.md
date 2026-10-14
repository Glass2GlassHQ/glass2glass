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

## Code audit: bugs, hot paths, duplication

Bugs:

- `XorFecBuffer` evicts the smallest raw `u16` seq, so ULPFEC, FlexFEC and
  ST 2022-1 stop recovering after the sequence wraps. Key by extended seq.
- `videoconvert::raw_format_to_str` panics on `Rgb8` and `raw_format_from_str`
  accepts 5 of 15 formats. Use one GStreamer format-name table from core.
- `SrtpReceiverSet::receiver_for` keeps a receiver for an SSRC before the
  packet authenticates. Insert only after unprotect succeeds, and cap the count.
- `parse_sps_reorder` returns `None` on SPS scaling lists.
- `gamma`, `videobalance` and `alpha` swallow incoming `CapsChanged`. Move them
  onto `PixelFilter`.
- The `fmp4` fragment parsers ignore the `trun` composition offset.
- `mp4audiosrc` adds `trun` durations unchecked.
- ST 2110 depacketizers read the payload at offset 12 without `RtpHeader::parse`.
- RTP timestamp rebasing in `rtprecv` and four ST 2110 receivers does not unwrap.
- `srt.rs` trims the send buffer with a plain `>=` instead of `seq_ge`, and
  `srtsrc` ACKs every datagram.
- `(pic_width_in_mbs_minus1 + 1) * 16` overflows in `streamdec` and `vulkanvideo`.
- `sink_arm_loop` drops flushed frames without `pop_transit_ns`.
- Three audio capture sources stamp `frames * (1e9 / rate)` (drifts at 44.1 kHz)
  and use unbounded queues.
- `fmp4::contains_keyframe` ignores CRA and 3-byte start codes, and `H264Parse`
  never sets `keyframe` on AVCC input.
- `ffmpegaudiodec::adts_frames` checks sync with `& 0xF0`, `ffmpegaacenc::adts_wrap`
  has no 13-bit length guard.
- `VideoTestSrc::with_shared_memory` keeps every emitted frame.

Core runtime:

- `JoinAll` and `DynamicJoin` re-poll every arm on any wake. Poll only woken arms.
- Remove the muxer forwarder arm, link `MuxerPad::Link(in_rx)` directly.
- `FlowGate` takes the shared lock per frame in Playing. Check the state
  atomically first.
- Each push takes about six spin locks. Lock only when a signal flag is set.
- `park_or_tick` builds a new boxed sleep per park.
- Mutator runs drive every element through `DynRef` (a boxed future per packet).
- `FrameMetaSet` clone allocates a Vec. `broadcast_drop_closed` shares for the last
  sender too and duplicates `broadcast`.
- `Gate::process` and `Router::process` box a future per packet.
- The transform arm samples fill twice and reads the clock about five times per frame.
- Route the sink, transform and router loops in `runner.rs` and `fanin.rs` through
  the `graph_runner` arms.
- One arm-IO builder for `run_graph_inner` and `run_graph_threaded_inner`.
- Shared helpers for `muxer_arm` and `muxer_arm_pts`, one port-ordering
  function, `inspect` formatting `describe`'s `ElementDoc`, and `launch` calling
  `maybe_prepend_parser`.

Core, bindings, ML:

- `TensorView::materialize` copies one element at a time.
- Log `enabled()` locks per call site once a category override exists, and sinks
  emit under spinlocks.
- `wire::encode_packet` starts at zero capacity and copies the payload twice.
- `ElectedClock::now_ns` locks per read, `pacing::judge` clones `ClockSync` twice.
- `BlobMeta::get` allocates a `String` per lookup.
- `g2g_bridge_push_buf` re-copies the frame on every retry.
- `g2g-python` `handoff` makes System frames writable for `Consume` jobs.
- Frames crossing the plugin ABI always arrive read-only.
- `g2g-pyapi` `pull_timeout` polls with 1 ms sleeps.
- `wgpupreprocess` copies each frame into a padded Vec before upload, and four
  wgpu readbacks are written by hand with per-frame staging buffers.
- Dedup in `g2g-ml`: RGBA to NCHW preprocessing, f32 byte conversion, dtype size,
  the `only-on` gate shared with `g2g-python`.
- Dedup the `BusMessage` projection and host-bytes frame view in capi and pyapi.
- One YUYV to I420 converter for `g2g-mcu` and the plugins.

Containers:

- `avi.rs` `apply_idx1` and `fmp4.rs` `stsc` lookups are quadratic.
- The AVI writer copies the file 4 to 5 times, `mp4muxn` progressive 2 times.
- Progressive MP4 and AVI demuxers de-frame every track to emit one.
- PS, Ogg, IVF, Matroska and fMP4 demuxers drain the buffer front per unit.
- Muxers copy each payload 3 to 5 times (mp4muxn, fmp4mux, matroska, flv, mpegts).
- One `moov`/`moof` writer for fmp4mux, mp4muxn and mp4audiosink.
- One `moof` parser for `fmp4` and `mp4audiosrc`.
- One WAV `fmt ` writer and `wFormatTag` table for wavsink, wavenc, wavparse, avi.
- Shared `MkvMuxSettings` for mkvmux and mkvmuxn.
- One table-driven MPEG CRC-32 for ogg and mpegts.
- `rtmp` reuses the FLV tag codec from `flv.rs`.

Codecs and parsers:

- Vulkan decode creates a bitstream buffer per picture and copies each AU 2 to 3
  extra times, and `ensure_decoder` parses SPS/PPS before the fingerprint check.
- Slice-header parsers de-emulate the whole slice.
- `NalParse` and `StartCodeParse` rescan the accumulator from byte 0 and scan
  each AU four or five times. Share one accumulator.
- NVENC registers its input surface every frame.
- `ffmpegenc` allocates an AVFrame per frame, `ffmpegdec` offload copies the AU twice.
- One SPS head, VUI, HRD and PTL parser for h264parse, h265parse and vulkanvideo.
- `vulkanvideo` uses av1parse's OBU walker, LEB128 and uvlc.
- One length-prefixed to Annex-B converter and one avcC/hvcC parser.
- One H.26x keyframe check for all callers.
- ffmpegaacenc and ffmpegaudiodec use the aacparse ADTS helpers.

Raw video and GPU:

- `PixelFilter` effects allocate and copy the frame. Apply in place.
- `videorate` deep-copies each frame up to three times.
- `textoverlay` reshapes and re-rasterizes every cue on every frame.
- `gaussianblur` allocates f64 scratch per frame.
- `tensorconvert` builds a permutation per frame.
- `videoconvert` hub path allocates i32 planes per frame, has no direct
  P010/NV12 arm, and computes U and V for every luma sample.
- `dmabufwgpu` and `wgpudmabuf` create Vulkan buffers per frame.
- `deinterlace` yadif copies each frame twice.
- `colorspace` calls `powf` per pixel on HDR paths.
- Compositor scaled blends divide per pixel. Share videoscale's axis tables.
- `waylandsink` has its own NV12 to XRGB converter and three frame passes.
- NV12 plane geometry is derived by hand in kmssink, waylandsink, videoflip,
  compositor, cuda, glnv12, cvnv12, and seven `w*h*3/2` sites.
- `timeoverlay` and `analyticsoverlay` copy `paint::Canvas` blending.
- `wgpusink` and `wgpucompositor` hand-write readback.

Audio:

- `audioconvert` has no passthrough, and noise shaping removes from the history front.
- `pipewiresink` moves bytes one at a time under the realtime mutex and sleeps on EOS.
- wasapisink, coreaudio and aaudio sinks block in `process`. Use `WorkerLink`.
- One capture loop for alsasrc and pulsesrc.
- audiobuffersplit, audioreverse, adpcm and audiomixer drain from the front.
- The f32 transform wrapper is copied 9 times with two allocations per buffer.
- `FirState` is direct-only, add overlap-save for long kernels.
- One IIR implementation for equalizer and audioiirfilter.
- `spectrum` recomputes the window and twiddles per FFT.
- One tone phase helper, one sample-width table, one `saturate_i16`.
- Sinks `to_vec` every buffer before the worker.

Network:

- `XorFecBuffer::try_recover` walks every repair and allocates per media packet.
- ST 2110 pacing sleeps on 1 ms tokio timers, receivers block in async `run`.
- SRTP re-keys HMAC per packet, SRT rebuilds the AES schedule per packet.
- `udpsink` uses `RetransmissionHistory` from rist.
- Remote TCP sets `TCP_NODELAY` and writes length and body once.
- `rtppay`, `rtspserversink` and `rtpdepay` copy each payload two or three times.
- ST 2110-20 allocates and copies per packet.
- Dedup the LiveKit signalling helpers, `access_token`, `MediaTime` conversion,
  and send `Transmit`s through `send_transmit`.
- One caps de-dup type for the remote and wswire elements.
- `rtcp::ntp_now` calls the core NTP converter, `rtx` uses `RtpHeader`.

Shared helpers:

- Move the plugins' ns to clock-rate tick and Q16 framerate math (about 30 sites)
  onto the `time` and `caps` helpers, `sample_bytes`/`sample_width` onto
  `AudioFormat::bytes_per_sample`, and the 7 `fixed_or_zero` copies onto `Dim::fixed`.
- Replace the `Pin::new_unchecked` polls next to the no-op wakers with `pin!`.
- `testutil::CollectSink` in src tests, a shared sink for integration tests.
- An `mfcommon` module for the Media Foundation helpers, one CUDA driver FFI module.
- One Android binder threadpool test helper.
- One hex codec and base64 encoder, one BE byte reader.
- `vtdecode`/`vtencode` `cm_time` helpers, `alertrecorder` uses `xmlutil::civil`,
  and aacparse, dvbsub, st2110anc use `annexb::BitReader`.

## Architecture guarantees (validation-first)

- **Grow the conformance matrix.** Persist evidence from the resource-owning
  tests still uncovered as they are validated (`vaapi` decode on Intel, the Android and
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

## Receive / decode

- **`VulkanVideoDec` residuals.** Run the `vulkanvideo` GPU tests on Intel ANV.
- **`VulkanVideoDec` odd-size two-plane texture.** Hand out an odd-size picture
  on the `NV12` / `P010` texture output, which Vulkan and wgpu refuse to create
  with an odd side, for example as an even-rounded texture with the picture size
  carried beside it.
- **`vulkanvideodec` behind a geometry capsfilter.** `vulkanvideodec !
  video/x-raw,width=W,height=H` fails negotiation with `NoConsistentFixation`,
  even when W x H is the stream's size.
- **10-bit H.265 through a launch line.** `filesrc ! h265parse ! vulkanvideodec
  ! fakesink` on a Main 10 stream fails with `CapsMismatch` (the decoder API
  itself decodes it).
- **`VulkanVideoDec` AV1 inter frames.** On AV1-TEST-VECTORS (2 of 242 pass)
  and the Chromium AV1 suites the keyframe decodes exactly and the first inter
  frame differs from dav1d, on both NVIDIA and RADV. 10-bit vectors fail to
  reach `I420_10LE`.
- **`VulkanVideoDec` H.264 and H.265 conformance.** 58 of 135 JVT and 104 of
  188 HEVC vectors pass, against 129 and 141 for `nvdec`. Interlaced H.264
  vectors fail with an error rather than being refused.
- **`VulkanVideoDec` with two GPUs.** On a host with an NVIDIA and an AMD GPU
  and no `VK_DRIVER_FILES`, an AV1 stream decodes to uniform gray.
- **AV1 spatial layers.** `av1-1-b8-22-svc-L2T1` and `L2T2` differ from the
  reference through `dav1ddec` and `rav1ddec`.
- **HEVC through `ffmpegdec`.** 53 of 188 HEVC vectors fail through
  `decodebin`, most of them Main 10 or RExt (4:2:2, 4:4:4, 12-bit and up).
- **`nvdec` AV1 below the hardware minimum size.** The `av1-1-b8-01-size-*`
  vectors smaller than NVDEC accepts fail the decode. Refuse them at
  negotiation so auto-plug takes a software decoder.
- **`ffmpegdec` AV1.** Every AV1 stream fails with `Hardware(Other)`, although
  the linked ffmpeg has libdav1d.
- **AV1 film grain chroma.** Match dav1d's chroma grain on libaom's
  `-aom-params film-grain-test=1` parameters: a 320x180 clip encoded with them
  differs from dav1d in 41 to 66 chroma bytes per frame, its luma is exact.
- **Cost-based decoder auto-plug (possibility).** Choose the auto-plug chain
  with the lowest total cost, summing a per-element cost and a per-frame
  memory-domain conversion cost scaled by resolution. Today the shortest chain
  always wins, so a `System` sink gets a software decoder even for 4K HEVC.

## CUDA / display

- Run `CudaKmsSink` from a bare VT with DRM master and verify the in-tree
  `// VERIFY:` points.

## Egress / transports

- **RTP over QUIC (RoQ):** implement after the draft becomes an RFC with an
  assigned ALPN. Candidate peers: mengelbart/roq (Go), meetecho/imquic.
- **`unixfdsink`:** reuse memfds from a pool for system frames, and carry a
  dma-buf frame's GPU sync fd.
- **RIST:** link bonding (`bonding-addresses`, broadcast and round-robin),
  multicast, and the Main Profile sequence-number extension (`ristrtpext` /
  `ristrtpdeext`); interop against librist's `ristsender` / `ristreceiver`.
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

## Sinks

- Validate `wasapisink` U8, S24, and S32 acceptance on a Windows host.

## Containers

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

- **Odd-size `videoflip`.** It refuses an odd width or height for the
  subsampled formats; mirror and rotate them with the chroma resampled, since
  the unpaired chroma column or row moves to the other edge.
- **Odd-size NV12 on the wgpu consumers.** `WgpuSink` and `WgpuPreprocess`
  refuse an odd NV12 width or height: the sink's packed-NV12 blit takes the
  picture width from the texture, and the preprocess storage-buffer shader reads
  the chroma rows at the luma row stride.
- **`textoverlay` font backend:** font-variation axes beyond `wght` on the
  shaped horizontal path (cosmic-text exposes only weight); vertical-mode
  shaping if cosmic-text ever grows writing modes.
- Add a carrier for non-default channel orders when a source needs an
  interleave order outside the per-count `ChannelLayout` convention.
- Apply an `OrientationMeta` in `kmssink` (a DRM plane rotation) and on the
  D3D11 VideoProcessor path, so those sinks advertise
  `Reconfigure::AbsorbOrientation` too.

## Metadata (FrameMeta / AnalyticsMeta)

- **Capture time on the other RTP paths.** Write and read the
  `abs-capture-time` and `ntp-64` header extensions on the WebRTC elements
  (str0m's `AbsoluteCaptureTime` plus an `ntp-64` mapping) and on the RTSP
  server ingest, taking the ids from the session's SDP.
- **Metadata through hardware decoders.** Carry a frame's metadata through
  `nvdec`, `vaapidec`, the Vulkan decoders and the platform decoders by PTS,
  so a capture time reaches the sink after a hardware decode.
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

- A GUI / tooling introspection surface beyond the text dump.

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
  `theoraparse`-side Theora decode, `openjpegenc` / `openjpegdec`, `openexrdec`,
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
