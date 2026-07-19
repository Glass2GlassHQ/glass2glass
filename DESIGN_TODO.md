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
   Metal present. Android: encode, Camera2, AAudio, Surface present.
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

## Architecture guarantees (validation-first)

The wedge is not feature breadth but hard, checkable guarantees on the things
GStreamer cannot easily fix (memory behavior, MCU/RTOS suitability, validation
clarity). Landed: copy plan (`copyplan`, M613, per-graph memory-domain hop trace
+ `CopyPolicy` budget, zero-copy proven at construction); conformance harness +
derived maturity (`conformance`, M614, evidence-derived `MaturityLevel` with
honesty guards, batteries in `g2g-plugins::conformance`, `g2g-inspect --maturity`).
Sequenced next:

- **Grow the conformance matrix (M615 + M619 + M621).** The persisted-evidence
  mechanism, two native-muxer oracles (`mp4mux`, `mpegtsmux` vs `ffprobe`), the
  ffmpeg-interop transports (`udpsrc` RTP / `rtmpsrc` / `srtsrc` / `srtsink` as
  peer-tagged `Oracle` rows), the Vulkan Video GPU decode tests (`vulkanvideo`
  H.264 / H.265 / AV1 as `Hardware` rows tagged with the GPU), and the CI
  `conformance` job (sets `$G2G_CONFORMANCE_LOG`, runs the oracles, publishes
  `--maturity` to the job summary) are done. The muxer oracles respect an
  externally-set log so they aggregate in CI. Remaining (optional): persist
  evidence from the other resource-owning tests as they are validated (RTSP interop,
  `wgpu`-export, native NVENC/NVDEC), plus more in-process batteries.
- **Whole-graph zero-alloc (M616 + M620).** The single-stage (M616) and multi-stage
  concrete-link (M620, source -> transform -> sink) data paths are proven zero-alloc.
  Remaining (larger, deferred): a fully zero-alloc *dyn* runner, monomorphized arms
  with unboxed `process` futures and a non-boxing `OutputSink`, so an arbitrary graph
  run through `run_graph` is heap-free, not only a hand-wired concrete chain. Low ROI
  vs. the proven data-plane claim; do it if an MCU deployment needs the full runner.
- **No-steady-state-allocation embedded mode (landed, M616).** A counting
  `#[global_allocator]` test proves the `StaticLendRing` capture -> frame -> drop
  data path is zero-alloc over 100k frames, with the `dyn OutputSink::push` per-frame
  box pinned as the honest boundary. Remaining: extend the zero-alloc proof to a
  multi-element pipeline over a concrete (non-`dyn`) link (a runner path whose
  `process` future is not boxed), so a whole graph, not just the capture edge, is
  provably heap-free in steady state.
- **Boundary-scoped time newtypes (landed, M618 + M622).** `TaiNs` / `RtpTs` in
  `g2g-core::time`; `MediaClock` takes a `TaiNs`, returns an `RtpTs`. M622 added
  `RefNs` (the monotonic reference) and typed the PTP servo's reference-vs-master
  seam: `PtpServo` / `PtpClock` `sync_exchange` take `(TaiNs, RefNs, RefNs, TaiNs)`
  and `observe_master` takes `(RefNs, TaiNs)`, so master and reference can no longer
  be swapped where the meaningless-offset mixing bug lived. No remaining work.
- **Metadata propagation contract (already in place).** The `Transform` /
  `Propagation` enums, `FrameMeta::propagate`, and `FrameMetaSet::propagate` exist,
  and `AnalyticsMeta` / `BlobMeta` declare honest drops (drop on `Encode`, keep on
  `Scale`). Remaining: framework-level *auto-application*, the runner carrying an
  input frame's meta onto a transform's output frames applying the element's declared
  transform, so meta survives a linear transform, not only a tee. This lands with the
  first non-analytics `FrameMeta` payload producer (captions / HDR / timecode still
  ride bespoke paths). See "## Metadata".

## Alloc-optional (heap-free) MCU core

The MCU/RTOS wedge's load-bearing guarantee: a build where `alloc` is not even
linked, so the framework is usable on the safety / no-heap parts that reject a
heap outright (the largest MCU market GStreamer can never reach). Scoping (done):
`g2g-core` is `no_std + alloc` with `alloc` mandatory ([lib.rs] `extern crate
alloc`). The heap splits cleanly into two layers, so this is a carve-out, not a
rip-out:

- **Data plane is nearly heap-free already.** `Frame` (heap only in a test), the
  `Caps` enum (no `Vec`/`Box` fields, so pairwise `intersect` / `fixate` between
  static elements is alloc-free), `MemoryDomain::System(SystemSlice::Foreign)`
  (the `StaticLendRing` zero-copy lend), and `staticpool.rs` (the const-generic
  ring). Pure-data modules `error` / `time` / `segment` / `link` / `mediaclock` /
  `state` are already 0-alloc; `metrics` needs only `critical-section`.
- **Heap lives in the dynamic / build-time layer**, which an MCU app does not run:
  the caps *solver* + `autoplug` + `parse_launch` + dynamic `Graph` (already behind
  `runtime`), plus `conformance` / `copyplan` / `dot` / `wire` / `tag` / `pool` /
  `property` / `stream` / `aggregator` (ungated today). MCU pipelines are static,
  known at compile time (concrete elements, const-generic capacities), matching
  g2g's "statically typed, not runtime string-keyed" identity.

Key design fork: the object-safe async traits return `BoxFuture =
Pin<Box<dyn Future>>` ([element.rs]) so the *dyn* element model is inherently
alloc. The no-alloc path needs a **generic/static element model** (async-fn-in-
trait, stable on MSRV 1.75, no `Box`) wired by direct concrete calls (the M620
pattern promoted to an API) driven by a const-generic static runner.

Phased plan:

1. **`alloc` feature seam (DONE, M623).** `g2g-core` has an `alloc` feature (`std` /
   `runtime` / `metadata` imply it; `alloc` pulls `spin`). `extern crate alloc` and
   the dynamic/build-time/tooling layer are gated behind it; `SystemSliceInner`
   keeps `Foreign` always and gates `Owned` / `Pooled` / `Shared`; `Caps::Tensor` +
   `TensorShape`, `CapsSet`, `to_gst_string`, `Frame::share`, and the GPU memory
   domains are gated. `default = []` is the no-alloc subset (also fixes the bare-build
   papercut); host consumers get `alloc` via `runtime` / `std`. Verified: `--no-default-features`
   compiles + cross-compiles clean to `thumbv7em-none-eabihf` with no allocator; the
   full build is unchanged. The `Caps::Tensor` carve-out is closed (M636:
   fixed-rank `TensorShape`), so this phase is complete.
2. **Static element model + runner (DONE, M624).** `g2g_core::staticelem`:
   `StaticSource` / `StaticTransform` / `StaticSink` using `async fn` in trait
   (unboxed futures, no `Box`, no `dyn`), the const-arity runners
   `run_source_sink` / `run_source_transform_sink`, and a `Chain` combinator for
   longer pipelines. Executor-agnostic (Embassy on an MCU, `block_on` on a host),
   part of the no-alloc subset (cross-compiles to `thumbv7em`). Runtime zero-alloc
   proof: `m624_static_pipeline_noalloc` (100k frames, 0 allocations).
3. **Link-time no-heap proof (DONE, M625).** `examples/g2g-noalloc`: a `no_std`
   staticlib on `g2g-core` `default-features = false` (no `alloc` crate) with no
   `#[global_allocator]`, building a real source -> transform -> sink pipeline. It
   links for `thumbv7em-none-eabihf` only if zero heap is used.
   `tools/noalloc-check.sh` (in CI) asserts the archive references no allocator
   symbols. Stronger than the M616 runtime counter, and the embedded analog of the
   copy-plan / conformance moat. The panic-free half is done too (M626): every
   reachable path avoids unwrap / index / overflow panics and the single-poll
   executor discharges the compiler's resumed-after-completion guard, so the
   archive has zero `core::panicking` symbols (asserted by the same script, which
   also runs the pipeline on the host via `host-harness.c` to back the symbol
   proofs with a real execution).
4. **Follow-on breadth** (own the space): the peripheral seams and the
   executor story are done (`SpiDisplaySink` M629; `FrameGrabber` +
   `GrabberSrc` M630, the proof pipeline's source; `PcmWriter` + `PcmSink`
   M631; Embassy task driving the pipeline under QEMU M632; FreeRTOS task via
   the C-ABI staticlib M633; Zephyr application via Zephyr's CMake build
   M637; fixed-point codecs, G.711 M638 + IMA ADPCM M639, both bit-exact vs
   ffmpeg; hardware-codec-peripheral seam, `JpegDecoder` + `HwJpegDec` M640,
   datasheet-tested on mocks). Still open: on-device `Hardware` conformance
   rows (platform = `STM32H747`), reusing the M621 evidence mechanism, which
   would also give the M640 JPEG seam its real-silicon tier. (The
   `forbid(unsafe)` application posture is done, M634: const ring + safe
   `drive_ready`, proven by `m634_forbid_unsafe`.) Done already: the build-time worst-case
   RAM/stack/ROM report (M627, `tools/footprint-report.sh` + `footprint.py`,
   budget-enforced in CI); the emulated Cortex-M execution proof (M628,
   `examples/g2g-qemu` + `tools/qemu-check.sh`, the shared `noalloc-pipeline`
   booted on QEMU MPS2-AN386 with the checksum verified on-target); and the
   first peripheral element (M629, `g2g-mcu::SpiDisplaySink`, ST7789/ILI9341
   over `embedded-hal`, datasheet-tested on mock peripherals and serving as
   the proof pipeline's sink: 2661 B ROM / 0 B static RAM / 1300 B stack for
   the whole pipeline).

5. **Deterministic-audio wedge track** (from the 2026-07 strategy review: one
   deterministic pipeline API across MCU vendors / RTOSes / hosts; audio
   first because vendor audio frameworks, ESP-ADF / NXP Maestro / ST
   AudioChain / SOF, prove demand and are all silicon-locked; the flagship
   demo is one graph, `capture -> convert -> resample -> mix -> encode ->
   RTP`, on STM32+FreeRTOS, NXP i.MX RT+Zephyr, and Linux, unchanged).
   The non-silicon items are closed (fault recovery, the receive path, the I2C
   sensor + UART catalog, the certification artifacts, and the RISC-V Tier-0
   port all landed), so what is left needs real hardware or is a small follow-up:
   - **On-device `Hardware` rows (ARM).** NUCLEO-H743ZI2 (Cortex-M7 =
     `thumbv7em`, the proofs' ISA; also the M640 JPEG codec's native silicon)
     and NXP i.MX RT, reusing the M621 evidence mechanism (also the home of
     real-silicon timing, the on-device complement of the M645 icount report).
     `examples/g2g-stm32h743` (M661) is the H743 harness: the flagship audio
     graph egressing RTP over on-chip Ethernet via a pure-Rust `embassy-net`
     stack (the `EmbassyNetSender: PacketSender` bridge maps the egress seam onto
     a UDP socket). It compiles for `thumbv7em` (verified); only runtime config
     (RCC/clock, RMII pins, RTP destination) needs finalizing on the board.
     Silicon rows also turn the `docs/safety` artifacts (M655) from
     emulation-backed into silicon-backed.
   - **ESP32-P4X board bring-up (RISC-V on-device).** M656 proves the no-alloc /
     panic-free / footprint guarantees for `riscv32imafc` at link time; putting a
     pipeline on the P4X-EYE board is two tiers of integration on top. Verify
     these unknowns before committing to a toolchain: whether `esp-hal` has any
     pure-Rust MIPI-CSI / ISP / HW-H.264 support (expect C-only, so the C-seam),
     and whether bare `no_std` Rust can reach the on-board ESP32-C6 WiFi stack
     without pulling in `esp-idf`/`std` (this decides Tier 2's toolchain).
     - **Tier 1: esp-hal harness + display (no camera, achievable first).**
       - Board-agnostic display runner: DONE. `noalloc_pipeline::run_display_with`
         is generic over the `embedded-hal` 1.0 `SpiDevice`/`OutputPin`/`DelayNs`
         seams, so a real HAL's peripherals drive the same proof pipeline; a host
         test drives that entry and checks the wire is bit-identical.
       - Full-panel 240x240 streaming (Tier 1.5): DONE. `SpiDisplaySink::with_stripe`
         streams a large panel in horizontal bands (the ring holds one band, not
         a 230 KB framebuffer), and `noalloc_pipeline::run_display_banded_with` is
         the 240x240 runner; host-tested (`m629_spi_display`, incl. a full refresh
         tiled from a tiny ring).
       - `examples/g2g-esp32p4` harness: DRAFTED (esp-hal `#[main]` init + SPI2 /
         GPIO panel wiring + call into `run_display_banded_with`), excluded from CI.
         Blocked on esp-hal shipping a released `esp32p4` (git `main` only today,
         so the git dep cannot enter the normal build); when released, switch the
         dep to the version and it compiles. Then verify the GPIO map + esp-hal
         API calls on the board and light the ST7789.
       - esp-hal `I2c` adapter to reuse `Sht3xSrc` (the seam is already
         `embedded-hal` 1.0 `I2c`), validating the M654 sensor catalog on metal.
       - Add the on-device evidence row (M621 mechanism): a checksum verified on
         the P4, turning the M656 footprint/exec claims from link-time into
         silicon-backed, plus a real-silicon timing sample (the M645 icount
         analog).
     - **Tier 2: camera -> encode -> RTP flagship (needs vendor C drivers).**
       - Hardware H.264 encoder seam: DONE (host side). `g2g-mcu::hwh264`
         (`H264Encoder` + `HwH264Enc`) and the `CH264Encoder` C bridge (M660) are
         built and host-tested through a mock and a real `extern "C"` callback
         (`m660_hwh264`), incl. a `camera -> encode` pipeline. What remains is
         wiring the P4's actual HW H.264 C driver behind `CH264Encoder` on silicon.
       - Color convert (camera 4:2:2 -> encoder 4:2:0): DONE. `YuyvToI420`
         (M661) is the heap-free packed-YUYV -> planar-I420 `StaticTransform`,
         host-tested including a `camera -> convert` pipeline; its output is
         exactly `HwH264Enc`'s expected I420 size.
       - MIPI-CSI camera source: bridge the ESP-IDF C camera driver
         (`esp_cam_sensor`/`esp_video`) through `CFrameGrabber` (M650 C-seam),
         since esp-hal almost certainly lacks pure-Rust CSI/ISP. C driver *is*
         the peripheral; `GrabberSrc`/`SpscCaptureSrc` stay unchanged.
       - WiFi/RTP egress via the ESP32-C6 network stack behind `CPacketSender`
         (M650). If bare `no_std` cannot reach the C6 stack, this forces the
         esp-idf staticlib path (FreeRTOS-on-RISC-V), the RISC-V analog of
         `examples/g2g-freertos`; optionally a Zephyr `esp32p4` board target
         (analog of `examples/g2g-zephyr`).
       - The on-silicon flagship: `camera (MIPI-CSI) -> convert -> HW-H.264 ->
         RTP -> C6/WiFi`, wire-validated against a host RTP peer (the M643
         ffmpeg-peer discipline), with a tee'd branch to `SpiDisplaySink` for an
         on-panel preview.
   - **QNX (safety-certified RTOS, automotive/medical).** A POSIX microkernel on
     Cortex-A / x86-64, not the MCU path; reinforces the safety-cert and PTP/
     Pro-AV wedges in their most lucrative vertical, where GStreamer (C) is the
     incumbent safety teams dislike. Tier 0 done (spike, `PORTABILITY.md`): the
     portable pure-Rust surface (`g2g-core` no-alloc + `alloc`/`runtime`,
     `g2g-mcu`, `g2g-plugins` no_std baseline) compiles for `aarch64`/`x86_64`
     `nto-qnx800` with zero changes (Linux HW is cleanly excluded via
     `target_os` gating). Tier 1 (needs the free QNX SDP 8.0): the `std`
     transports; the one dependency question is `tokio` on QNX 8. Tier 2 (needs
     an SoC + partner): QNX Screen display sink + vendor VPU via the M650 C-seam
     + GPU, as `target_os = "nto"` elements. Free to test (non-commercial SDP);
     commercial use is license-gated (confirm the open-source-interop clause).
   (Licensing was considered and decided: the whole workspace stays
   LGPL v2.1+, like ffmpeg and GStreamer.)

## Negotiation

- **Preference algebra.** `CapsPreferences` is a placeholder (sum-of-indices);
  needs a real competing-constraint scenario to drive it.
- **Closure-free `FieldTransform`** so forward derivation is declarative too
  (removing the mask/closure duplication `DerivedCoupled` carries).
- **β allocation re-cascade across a muxer** (per-input-pad re-cascade); the
  node-keyed coordinator walk terminates at muxers today.
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

## Cleanup

- **Adopt `MemoryDomain::as_system_slice()`.** ~50 sites hand-roll
  `if let MemoryDomain::System(s) = ..` to read CPU bytes; the accessor
  (returning `Option<&[u8]>`) replaces them and stays refutable under no_std
  (single-variant `MemoryDomain`), where the manual match is an irrefutable-let
  error.

## Platform: macOS

- `VtDecode`: a `CVPixelBuffer` / `IOSurface` zero-copy domain.
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
  vendor-neutral hardware-decode-into-wgpu-texture path now works end to end.
  DONE (M492, validated on the 3060): full DPB reference management. `H264DpbDecoder`
  (`create_h264_dpb_decoder` + `decode_all`) parses each slice header (`frame_num`,
  POC types 0/2, `idr_pic_id`, `slice_type`, `nal_ref_idc`), splits access units on
  `first_mb_in_slice == 0`, computes picture-order-count, and runs H.264
  sliding-window reference marking, handing the driver the active reference slots
  (with their `StdVideoDecodeH264ReferenceInfo`) per picture, so P frames after the
  IDR decode against their references. The whole 640x480 fixture (2 GOPs of IDR + 4
  P frames) decodes **bit-exact** against the ffmpeg software decoder (SAD/px 0).
  This exposed and fixed a latent one-bit bug that had been silently corrupting
  M489-M491: `parse_h264_pps` read the trailing optional block without an
  `more_rbsp_data()` check, so a baseline PPS's `rbsp_stop_one_bit` was misread as
  `transform_8x8_mode_flag = 1`, desyncing the driver's CAVLC coefficient parse
  (every decoded frame was flat mid-grey; the old tests only checked non-uniformity,
  not correctness). `BitReader::more_rbsp_data()` added; m489 now also asserts bright
  content is present.
  DONE (M493, validated on the 3060): the `VulkanVideoDec` `AsyncElement` wrapper.
  `Caps::CompressedVideo{H264}` in -> `Caps::RawVideo{Nv12}` system memory out
  (the decoder's native layout, no colour convert); `intercept_caps` +
  `caps_constraint_as_transform` (H.264 -> NV12 same geometry) so the solver gives
  each link real caps; `configure_pipeline` opens the device, the session + DPB
  build lazily on the first SPS/PPS-bearing AU (in-band, not in caps); `process`
  decodes each AU (DPB state carries across calls) and emits one NV12 frame per
  picture with a leading `CapsChanged`. Registered `vulkanvideodec` (launch-only).
  `m493_vulkan_video_element` feeds the fixture one AU per `process` call and
  asserts 10 real NV12 frames. `examples/vulkan_video_smoke` dumps PPM frames you
  can open.
  DONE (M494/M495, validated on the 3060): zero-copy GPU-resident output end to
  end. `H264DpbDecoder::decode_all_to_textures` (via `create_h264_dpb_decoder_gpu`)
  decodes every picture and converts each DPB slot in place through the ycbcr
  compute pass (shared free fn `nv12_to_wgpu_texture` with a `restore_to_dpb`
  barrier, so a slot stays a valid reference) into an RGBA `wgpu::Texture`; DPB
  slot images are `SAMPLED` + `CONCURRENT` in GPU mode. `VulkanVideoDec` is now a
  multi-domain producer (`output_memory` = `WgpuTexture` preferred,
  `output_domains = {WgpuTexture, System}`, `configure_allocation` settles it;
  falls back to NV12 if no compute queue). `VulkanVideoDevice::gpu_context` /
  `VulkanVideoDec::gpu_context` expose the decode device as a `GpuContext` so a
  `WgpuSink` shares it; the `gpu` module is built for `vulkan-video`.
  `m494_vulkan_video_dpb_texture` decodes to 10 GPU textures; `m495` runs the full
  `VulkanVideoDec -> WgpuSink::offscreen` wedge with no GPU->CPU readback.
  DONE (M496): auto-plug. `vulkanvideodec` is registered in
  `register_autoplug_candidates` tagged `produces(WgpuTexture)` + `hardware()`, so
  a WgpuTexture-preferring `decodebin`/`playbin` search picks it (domain match
  dominates), while System auto-plug is unchanged (domain mismatch, like NvDec's
  Cuda); a valid System fallback when it is the only H.264 decoder. Its source pad
  template advertises `Nv12` + `Rgba8`. `m496_vulkan_video_autoplug` (GPU-free)
  covers the selection. Left:
  3. Mid-stream resolution reconfig is done (M519, validated on the 3060): a
     keyframe whose parameter sets carry a new coded geometry rebuilds the session
     + DPB and re-emits `CapsChanged`, flushing the outgoing decoder's pipelined
     tail first so no frame is lost. Remaining: a same-geometry SPS/PPS *content*
     change (a new profile / ref config at the same dimensions), which keeps the
     session today.
  4. H.265 full-DPB decode is done (M501-M503, bit-exact vs ffmpeg on the 3060,
     see DESIGN.md 4.11.6): parse + `Std*` mapping + session + `H265DpbDecoder`
     (system NV12 + GPU texture). `VulkanVideoDec` element + auto-plug wiring is
     done for all three codecs (M504 element decode to NV12, M496 auto-plug
     selection), and the GPU-texture wedge (decode -> `WgpuTexture` -> `WgpuSink`,
     no readback) is validated for H.264 / H.265 / AV1 on the 3060 (M535).
     B-frame streams decode bit-exact and are emitted in display order by the
     whole-stream `decode_all` / `decode_all_to_textures` (M569, `index_pictures` +
     `reorder_to_display_order`, keyed by (coded-video-sequence, POC); verified for
     H.264 and closed-GOP H.265 on the 3060). Full-stream H.265 open-GOP (CRA with
     RASL leading pictures) decodes bit-exact too (M577; flush the DPB only on an
     IRAP with `NoRaslOutputFlag == 1`, else keep the RPS-listed pre-CRA refs).
     Mid-stream random-access tune-in at a CRA is done too (M587, validated on the
     3060): after a `reset` (seek) to a CRA, the CRA is the first picture, so its
     `NoRaslOutputFlag == 1` and its RASL leading pictures (which reference absent
     pre-CRA frames) are discarded (`h265_is_rasl` + a `skip_rasl` flag set per
     IRAP); the CRA's trailing pictures and following GOPs decode bit-exact vs a
     full decode. What remains for H.265: long-term references. Streaming B-frame reorder in the
     `VulkanVideoDec` element is done (M586, validated on the 3060): its system
     (NV12) path feeds retired `decode_push` frames through a `ReorderBuffer` keyed
     by (coded-video-sequence, POC), so an AU-by-AU stream with B-frames emits in
     display order (byte-exact vs the `decode_all` oracle for H.264 / H.265). The
     low-level `decode_push` / the re_video adapter stay in coding order by design
     (re_video reorders by PTS itself). Still open: a VUI-derived tighter
     reorder-depth bound (the element uses the DPB size, a safe over-approximation),
     and AV1 / GPU-texture streaming reorder (AV1's display order is
     `show_existing_frame` / `order_hint`, not a POC sort; the element's
     GPU-texture path is fed whole-stream and rides `decode_all_to_textures`).
  5. AV1: DONE (M504) the OBU framing + sequence-header parse + `StdVideoAV1SequenceHeader`
     mapping + a top-level frame-header classifier (`parse_av1_sequence_header`,
     `to_std_av1_seq_header`, `av1_frame_infos`), the parse half, with GPU-free unit
     tests over a real libaom 640x480 fixture (`av1_640x480.obu`). DONE (M505,
     validated on the 3060) the decode session: `open_av1_decode_device` + `av1_profile`
     + `create_av1_session` (`VkVideoDecodeAV1SessionParametersCreateInfoKHR` carrying
     the Std sequence header), the M488/M502 analog, which driver-validates the M504
     mapping. DONE (M506a) the full uncompressed frame-header parse
     (`parse_av1_frame_header` + all sub-parses), validated field-by-field vs ffmpeg
     `trace_headers`. DONE (M506b/M506c, on the 3060) the `Std*` picture-info mapping
     (`to_std_av1_picture_info`) and `Av1DpbDecoder` (8-slot reference model, per-tile
     offsets, `vkCmdDecodeVideoKHR`): the whole fixture (1 key + 9 inter, including the
     compound / temporal-MV frames) decodes **bit-exact** vs ffmpeg (SAD/px 0 every
     frame). M506c fix: the `setup_past_independence` loop-filter ref-deltas default
     ALTREF2 / ALTREF to -1, not 0 (else deblocking is mis-configured for compound
     blocks referencing the alt frames). The `VulkanVideoDec` element + auto-plug +
     GPU-texture wedge wiring for AV1 is done (M504 / M496 / M535, validated on the
     3060). Multi-tile decode is done too (M564, `av1_tile_layout` parses the
     OBU_FRAME tile-group size prefixes; 2x2 + 4x4 clips bit-exact on the 3060).
     Alt-ref + show_existing_frame (decode order != display order) is done too
     (M565, `scan_ops` reorder-aware path; bit-exact on the 3060). Film grain is
     synthesized on the decoded NV12 (M566, `apply_film_grain_nv12`, full spec
     7.18.3 ported from re_rav1d; bit-exact luma+chroma vs dav1d on the 3060; the
     3060 can't do driver grain, COINCIDE-only), and on the GPU-texture path too
     (M568, `grained_slot_to_texture` reads the slot back, synthesizes on the CPU,
     and uploads; grain is output-only so the DPB reference stays untouched;
     bit-exact vs dav1d on the 3060). Loop restoration is fixed (M567,
     `LoopRestorationSize` is the `1 + lr_unit_shift` encoding, not the pixel size).
  6. Colour / HDR: the YUV -> RGB conversion is colour-space aware (M570,
     `VideoColorSpace` from the H.264/H.265 VUI or AV1 color_config; BT.601/709/2020
     matrix + studio/full range, on both the CPU and GPU converters). 10-bit HEVC
     (Main 10, M571) and 10-bit AV1 (Main, M572, `av1_profile(bit_depth)` from
     `color_config.BitDepth`) decode are done on the system path (SPS/seq-derived bit
     depth -> `G10X6` format + 2-byte-per-sample readback; `Nv12Frame.bit_depth`;
     bit-exact on the 3060). The GPU-texture path carries 10-bit too (M573,
     `YcbcrConverter` formats chosen from bit depth: `G10X6` -> `R16G16B16A16_SFLOAT`
     via an `rgba16f` compute shader, imported as a `Rgba16Float` `wgpu::Texture`).
     PQ / HLG HDR -> SDR tone-mapping is done (M574, opt-in
     `create_*_dpb_decoder_gpu_tonemap`: EOTF -> BT.2390 EETF -> BT.2020->709 gamut
     -> BT.709 OETF in the shader, keyed off `VideoColorSpace::transfer` from CICP).
     HDR swapchain present is done (M575, `vulkanhdrsink`/`hdr-present`:
     `VulkanHdrSink` owns a raw `VK_KHR_swapchain` on the decode device negotiating
     `HDR10_ST2084`/scRGB/SDR + `VkHdrMetadataEXT`, presents the passthrough PQ
     texture by `vkCmdBlitImage`; on-screen validated live via the example). The HDR
     track is complete.
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

- **SRT:** real-peer interop with libsrt/ffmpeg is validated for the **full
  matrix** by `srt_ffmpeg_interop` (ignored, needs ffmpeg+libsrt): both
  directions (ffmpeg caller -> `SrtSrc` listener; `SrtSink` caller -> ffmpeg
  listener) x plaintext + AES-128 + AES-256 (M522/M525/M526). (TSBPD, AES-256,
  key rotation, congestion control landed earlier; a rekey KM is now
  retransmitted until the peer KMRSPs, M671, so it survives KM-packet loss.)
- **RTMP:** multiple NetStreams over one connection. Deferred by design: it needs
  a dynamic-arity multi-output `RtmpSrc` (the stream count is only known once the
  client `createStream`s at runtime), which collides with g2g's fixed-arity-from-caps
  model (the same call made against webrtcbin-style request pads). Niche in
  practice (OBS / ffmpeg / CDNs publish one stream per connection); revisit only
  with a concrete need. (Window-acknowledgement back-pressure is done, M533:
  `RtmpSession` emits an `Acknowledgement` every Window-Ack-Size bytes received
  (configurable via `with_window_ack_size`), and `RtmpPublisher` tracks the
  server's window + acknowledged sequence, exposing `throttled()`; `RtmpSink`
  blocks feeding media on the socket ack while throttled, so a slow server
  back-pressures the pipeline instead of bloating the socket buffer. The
  HMAC-SHA256 "genuine FMS/FP" digest handshake strict CDNs require is done,
  M521: `RtmpPublisher` sends a digest C1 + response C2 by default and
  `RtmpSession` answers / validates it, both auto-falling-back to the simple
  handshake against a non-genuine peer. Real-peer interop is validated M527:
  `rtmp_ffmpeg_interop` has ffmpeg publish into `RtmpSrc`, ffprobe decoding the
  demuxed FLV; ingest interoperates out of the box. Egress to a real CDN stays
  user-side.)
- **RTSP server:** RTCP / keepalive during PLAY; ingest multi-client (serving
  multi-client is done, `RtspServerSink`). The serving *sink*'s TCP-interleaved
  transport is done (M672: `$`-framed RTP on the control connection, RFC 2326
  §10.12, validated against `ffmpeg -rtsp_transport tcp` playing from the sink),
  as is the *ingest* source's (M532).
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
    Remaining: a real LiveKit Cloud / TURN-relay run (genuine remote NAT).
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
    mid-session transceiver ADD (a new m-line on a live session; direction
    renegotiation landed M729, and the fixed-arity pad model has no target pad
    for a genuinely new track, so this needs a design call).
  - **T2 (mostly wiring): RTCP feedback.** PLI / keyframe-request DONE (M243):
    `Reconfigure::ForceKeyframe` + `take_reconfigure`; `WebRtcSink` maps a remote
    `Event::KeyframeRequest` to it, `Av1Enc` forces an IDR, `WebRtcWhepSrc`
    originates PLI on mid-GOP join. Adaptive bitrate / congestion control DONE
    (M244): `PushOutcome::Bitrate` + `take_bitrate`; `WebRtcSink` enables str0m
    BWE and relays `Event::EgressBitrateEstimate`, `Av1Enc` retargets (rav1e
    context rebuild, hysteresis-gated). T2 is complete
    (VP8/VP9 honor both via an encoder rebuild, M730).
  - **T4: signalling ecosystem.** Drop the `[patch.crates-io]` str0m fork
    (unpadded media sends) once the LiveKit forwarder fix (livekit#4690, on
    their master) ships in a release, or str0m#1014 lands; a real LiveKit Cloud
    run (genuine remote NAT + STUN/TURN on the LiveKit elements); then Janus /
    Kinesis as wanted.
  - **T5: advanced.** Live multi-rid validation of the WHIP simulcast session
    (needs a WHIP server that ingests client simulcast: mediamtx cannot, and
    LiveKit's WHIP ingress transcodes a single layer; Janus + a WHIP front end
    is the known candidate). FEC is blocked upstream (str0m has no FEC payload;
    loss recovery is NACK/RTX). Full renegotiation; data-channel loose ends
    (str0m surfaces no remote-close event, so EOS rides an explicit marker
    message; a WHIP/SFU-signalled data channel vs the P2P `SdpChannel` seam).
  Recommended order: T1 remainders -> T2 -> T4 -> T5.

## Adaptive streaming (HLS / DASH)

- **HLS:** SAMPLE-AES key rotation mid-stream; cbcs audio (AAC) + per-sample IV
  (cenc/cbc1); `saiz`/`saio` aux-info + `seig` sample groups. (Encrypted fMP4 cbcs
  *video* init segments are done, M164; `#EXT-X-BYTERANGE` single-file CMAF is
  done, M368; throughput-driven ABR with mid-stream variant switching is done,
  M371; live-edge start is done, M438.)
- **DASH:** wall-clock `@duration` live profile; multi-period; discontinuity /
  multi-period boundary `SEGMENT` emission. (`SegmentList`
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
  done (M589, all three validated on Fedora / PipeWire playing a real tone across
  S16 + F32, stereo + mono; `m589_audio_sink_smoke`, skips with no device). Still
  open: more sample formats (S24 / S32 / U8) and multichannel speaker-position
  layouts (>2ch, position-aware down/upmix, needs channel-position metadata);
  DMABUF / zero-copy.
- Generic `GlSink` over EGL (vendor-neutral NV12 / RGBA present, no CUDA).

## Containers

- **MKV / WebM:** a front `SeekHead` so the muxer's own output is seekable from
  byte 0 without reading past the Clusters (a two-pass / seekable-output finalize
  mode; the streaming muxer writes `Cues` at EOS, M375, but cannot place a front
  `SeekHead`); `Targets`-scoped (per-track) tags. (Multi-track A/V muxing landed:
  `mkvmuxn`, M294; `Cues` parsing + indexed demuxer seek, M373; `SeekHead`-driven
  `Cues` prefetch, M374; muxer-written `Cues`, M375.)
  Single-track `MkvMux` also lacks unknown-size Clusters (live read).
- **MPEG-TS:** multi-stream / multi-program muxing + selection; PCR-based timing.
- **OGG:** granule-position timing; Vorbis output; multi-stream; `oggmux`.
- **FLV:** VP6 / H.263 / MP3 / Speex codecs (only H.264 + AAC ride the tag
  stream today); B-frame composition-time write on the mux side (the demuxer
  reads CTS, the muxers write 0).
- **CMAF / fMP4:** the CMAF-specific signalling layer on `Mp4Sink` / `Mp4Src`.

## Codecs

- **`FfmpegH264Enc`:** runtime bitrate retarget (fixed at open, like `Av1Enc`'s
  rebuild), NV12 input, 10-bit.
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

_(No open parser items.)_

## Transforms and effects

- **`textoverlay` font backend:** the `truetype-overlay` feature (M409, `ab_glyph`
  since M668) renders both glyf and CFF/CFF2 outlines (CJK / accented / mixed-case,
  horizontal + vertical) with an explicit Latin+CJK fallback chain, so OpenType-CFF
  `.otf` fonts render, not only glyf `.ttf`s. Still open: variable-font axis
  selection (a non-default instance of a variable Noto Sans CJK), real shaping +
  bidi, and automatic system-font discovery / fallback, all of which point at the
  `cosmic-text` upgrade; plus a `vello` GPU backend and the `clockoverlay` /
  `timeoverlay` siblings.
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
- Timer-driven output (emit at the output rate even when inputs stall, a
  zero-order-hold aggregator tick). Needs the runner to deadline-tick the
  compositor without an input packet; constant-rate resampling of a flowing
  output is already covered by a downstream `videorate`.

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
- **A/V clock slaving** remaining pieces. The mechanism (audio-master
  `DriftClock` disciplined from `snd_pcm_delay`, elected at `AudioProvider`) and
  the lip-sync payoff are done and CI-validated (M590/M591/M592). Still owed:
  extend the same clock discipline to `PulseSink` / `PipeWireSink` (only
  `AlsaSink` provides a clock today); a headless display sink that adopts the
  elected `ClockSync` (today `SyncSink` uses its own clock and `WaylandSink`
  needs a display, so the M592 lip-sync test uses a harness sink); an on-display
  lip-sync soak on real hardware; and optionally a tighter drift model (outlier
  rejection on a glitchy `delay()`, faster convergence).
- **PTP clock (`PtpClock`)** DONE (M593 A/B/C + M594): `PtpServo`
  (offset/delay -> `DriftClock`, lock/holdover/outlier), `PtpClock` +
  `ClockPriority::PtpGrandmaster` (elected over audio/video, slaved to sinks via
  `run_graph`), `PtpSystemClock` (OS `CLOCK_TAI` delegate, host-validated), and
  `PtpClient` (in-process software PTP SLAVE over UDP: `ptp::wire` parser +
  `ptp::slave` state machine, both CI-tested, + the `g2g-plugins` UDP transport).
  The pipeline can now be PTP-mastered by either backend. Remaining polish (not
  blocking): a live multi-machine / `ptp4l`-grandmaster soak of `PtpClient`
  (host/root/reference-gear gated); `SO_REUSEPORT` so `PtpClient` co-exists with
  `ptp4l` on one host; querying `ptp4l` state so `PtpSystemClock` confirms *true*
  grandmaster lock; a direct PHC (`/dev/ptpN`) read; hardware RX/TX timestamping
  for uncompressed ST 2110-20 timing; BMCA/Announce, peer-delay, unicast.
- **ST 2110 media transport** (the layer above the PTP clock). Started: `MediaClock`
  (-10 PTP<->RTP-timestamp mapping, M595), `st2110audio` (-30 PCM L16/L24, M595),
  `st2110anc` (-40 ancillary/captions per RFC 8331, 10-bit-word parity+checksum,
  M596), all sans-IO and CI-tested; `st2110audiortp` (-30 `St2110AudioSink` +
  `St2110AudioSrc` over UDP, PTP-clocked timestamps, `st2110` feature, end-to-end
  UDP-loopback tested, M597); `st2110ancrtp` (-40 `St2110AncSink`/`Src` over UDP
  bridging the CEA-608/708 stack via CDPs, `st2110` feature, UDP-loopback tested,
  M598); `st2110video` + `st2110videortp` (-20 uncompressed video, RFC 4175 SRD
  line runs, `St2110VideoSink`/`Src` over UDP, RGBA + YUYV 4:2:2 8-bit,
  UDP-loopback tested, M599; + 10-bit 4:2:2 from planar `I422p10`, pgroup = 5
  octets MSB-first bit-packed, M600); `st2110sdp` (RFC 4566 + SMPTE ST 2110-10/-20/
  -30/-40 SDP generator / parser, `St2110VideoSink::sdp` / `St2110VideoSrc::apply_sdp`,
  M601); L24 / F32 audio (`PcmF32Le` -> L24 wire, M602); SDP `sdp()` / `apply_sdp()`
  for the audio + ancillary elements (M603); `st2110jxs` + `st2110jxsrtp` (-22 JPEG XS
  over RTP, RFC 9134 codestream mode, `VideoCodec::JpegXs`, `jxsv` SDP, UDP-loopback
  tested, M604); `SvtJpegXsEnc` / `SvtJpegXsDec` (the -22 codec, hand-rolled
  SVT-JPEG-XS FFI, `jpegxs` feature, host-validated encode<->decode + full -22 path,
  M605); `St2110Session` (multi-section SDP bundling video + audio + anc, `a=mid`,
  M606); `AudioFormat::PcmS24Le` integer PCM riding the -30 L24 wire (M607); ST 2110-7
  seamless protection (`st2110dup::SeamlessDedup` sequence-number merge + `a=group:DUP`
  SDP, M608); ST 2110-21 sender pacing (`st2110pacing::Pacer` linear / gapped schedule
  + conformance, wired into `St2110VideoSink` over the tokio timer, M609); the -7
  dedup wired into a two-socket `St2110VideoSrc` via the reusable
  `st2110dup::RedundantRtpReceiver` (`redundant` property, M610); the `Pacer` reused
  in the -22 JPEG XS sink via the shared `st2110pacing::pace_send` (M611); the full
  per-format -21 VRX validator, `st2110pacing::VrxValidator` (the leaky-bucket
  receiver-buffer model, M612). Remaining: wire compliance of -20/-22/-30/-40 +
  multicast should be validated against reference gear (built from the RFCs, not yet
  interop-tested).
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
  Conv / activation (`relu`, `sigmoid`) / pooling (`maxpool2d`, `avgpool2d`),
  batch-norm (M524), and GPU-resident multi-layer chaining are in place
  (M261/M265). A *whole* multi-layer model now imports from one weight file and
  runs end to end (M524): `WgpuInference::stack_from_safetensors` + `StackLayer`
  build the chain, validated on a conv-BN-ReLU-pool x2 -> global-avg-pool ->
  linear classifier (3060). Skip / residual topology now imports too (M531):
  `StackLayer::SaveSkip` / `AddSkip` + `ResidualStack::run` + a two-input
  elementwise-add GPU op (`WgpuInference::add`, `add_reference`), validated
  GPU-resident on a `y = conv(relu(conv(x))) + x` block bit-matching the CPU
  reference (3060). The safetensors loader dequantizes F16 / BF16 to f32 on the
  fly (M531), so real half-precision checkpoints load. Remaining: attention (for
  transformer stacks).
- ONNX import via `burn-import` (build-time codegen) for the Burn backend, the
  graph-topology counterpart (safetensors carries weights, not the architecture).
- A trained-weight `Module` path for `BurnInference` (conv, attention) once the
  codegen lands.
- Decoder DMA-BUF / D3D11 surface import into `WgpuPreprocess` (binds the
  surface directly into the compute pass; needs the surface-import handshake + a
  GPU tensor output domain).

## Developer tooling

Outstanding developer-tooling tasks, highest leverage first.

- **Per-element / per-link telemetry gaps.** Extend the `Observer` tap
  (`g2g-core/src/runtime/observe.rs`) and the M399 `ElementProbe` coverage:
  - Per-edge packet / byte counters + drops in the live tap (drops surface only
    in end-of-run `RunStats`).
  - The standalone fan-in / fan-out / session runners (`fanin.rs` / `runner.rs`
    hand-built API, not reachable from `run_graph_observed`) leave `per_element`
    empty: give them observed entry points and wire probes if that API needs to
    be observable.
  - Source-side timing: a source runs one long `run()` loop, so its cost only
    shows as its downstream's input fill.
  - Validate the dashboard live against an RTSP source.
- **Visual builder follow-ups.** For `tools/builder/` (React Flow):
  - YAML export (the JSON export already covers the graph model; schema shared).
- **Edge preview follow-ups.** Remaining: per-edge tap on the fan-in / muxer arms
  (the slot is shared via `SenderSink`, so those arms already carry it, but they
  are not exercised).
- **Negotiation explainer follow-ups.** `validate` (MCP / `toolingjson`) returns
  per-edge negotiated caps and, on a solve conflict, the structured failure
  (kind + node indices). Remaining: carry the *both caps sets* at the point of
  failure in the structured `NegotiationFailure` (the by-default log narration
  already prints them, but the error type still hands programmatic consumers only
  the node indices), which needs the solver to surface the candidate sets.
- **Per-frame latency waterfall.** The dashboard renders an aggregate stacked
  wait+work p50 per stage. The remaining piece is a single frame's path: a
  source-stamped sequence id carried through so one frame's queue-residency +
  `process()` at each stage can be assembled end to end (the aggregate uses
  per-stage distributions, not one frame's journey), plus the measured total
  against the `2 * capacity * frame_period` floor.
- **gst-parity differ.** Same launch line through real GStreamer and g2g;
  diff the negotiated caps per edge, the element set after autoplug, and the
  output (checksum, PSNR for lossy). Calliope already does differential output
  QA in its own repo, so decide first whether this lives there (adding the
  caps / topology diff) or in-repo; don't build both.
- **MCP server follow-ups.** `g2g-mcp` exposes list_elements / inspect /
  validate / launch. Add a tool to run a declarative graph file, and stream
  `launch` telemetry (via the `Observer`) rather than only final stats.
- Longer tail: a live pipeline TUI (a ratatui consumer of the same telemetry
  tap); a codec golden-fixture / PSNR conformance harness.

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
