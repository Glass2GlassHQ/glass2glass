# glass2glass (`g2g`)

[![CI](https://github.com/Glass2GlassHQ/glass2glass/actions/workflows/ci.yml/badge.svg)](https://github.com/Glass2GlassHQ/glass2glass/actions/workflows/ci.yml)

A hardware-first, sans-IO, asynchronous multimedia graph framework in pure
Rust.

**One pipeline, five targets.** g2g has a pure-Rust `no_std` core where `alloc`
itself is optional, so the same typed graph runs unchanged across the whole
hardware spectrum: **MCU · RTOS · CPU · GPU · WASM**. On the low end that is a
bare-metal Cortex-M with **no heap at all**; on the high end a GPU-resident
server pipeline. You write the graph once; the deployment shell picks the
executor and links the hardware.

The name reflects the metric the project optimizes for: **glass-to-glass
latency**, the time between physical photon capture and hardware presentation.

See [DESIGN.md](DESIGN.md) for the architecture specification and
[DEVTOOLS.md](DEVTOOLS.md) for the developer tooling (`cargo xtask`, the pipeline
visualizer, the caps explainer, benchmarks).

## Portability: one pipeline, five targets

The core (`g2g-core`) is pure Rust, `no_std` (with `alloc` an optional feature),
and sans-IO: the graph, the element traits, `Caps` negotiation, and the runner
are identical on every target. Only the deployment shell (which executor, which
hardware elements) changes.

| Target | What runs | How |
| :--- | :--- | :--- |
| **MCU** | a heap-free static pipeline on bare-metal Cortex-M | `alloc` is optional: the no-alloc build links **no allocator at all**, is proven panic-free, and has a budgeted ~KB-scale footprint. `g2g-mcu` peripheral elements (SPI display, camera / PCM capture, G.711 / ADPCM codecs, RTP egress + ingress, jitter buffer) over `embedded-hal` seams, plus interrupt/DMA capture and a bounded fault-recovery supervisor (retry / degrade / reset / watchdog). See [the embedded wedge](#embedded-heap-free-pipelines-on-a-bare-metal-mcu). |
| **RTOS** | the same static pipeline under a real RTOS task | one graph runs bit-exact under **bare-metal, Embassy, FreeRTOS, and Zephyr**; `embassy-sync` stack channels (`embassy` / `embassy-link` features) |
| **CPU** | the full media + protocol stack | Tokio, multi-thread on servers or current-thread on edge; the whole element library |
| **GPU** | zero-copy hardware pipelines | frames stay in Vulkan / CUDA / wgpu / DMABUF domains: Vulkan Video decode → `wgpu::Texture`, NVDEC / NVENC, CUDA ↔ wgpu bridge, no PCIe round-trip |
| **WASM** | the same graph in the browser | `wasm32`, single-threaded (no cross-origin isolation): WebCodecs decode, WebGPU present, in-browser or server-offloaded ML |

Same `AsyncElement`, same `Caps`, same runner on all five. This is proven, not
asserted: **[PORTABILITY.md](PORTABILITY.md)** runs one detection-overlay pipeline
whose processing stages come from a single shared `overlay_stages()` definition,
reused verbatim by the native (CPU) runner and the browser (WASM) build, and
gives reproducible evidence for each target (Cortex-M footprint, Embassy smoke,
CPU render, GPU-resident wgpu, in-browser canvas). See also
[The four pillars](#the-four-pillars).

### Also: QNX (safety-certified RTOS)

Beyond the five, the portable core is one spike away from QNX, the POSIX
microkernel that is the reference platform for ISO 26262 / IEC 62304
automotive/medical. QNX runs on application processors (aarch64 / x86-64), so it
is the `std`-capable path, not the MCU one. This is a **Tier-0 portability spike**
(compile-checked, not yet run): verified locally with no QNX SDP
(`cargo +nightly ... -Zbuild-std`), `g2g-core` (the no-alloc subset *and* the
full `alloc` + dynamic `runtime` layer: caps solver, autoplug, dynamic `Graph`),
`g2g-mcu` (the whole peripheral catalog), and the `g2g-plugins` `no_std` baseline
all compile for `aarch64-unknown-nto-qnx800` and `x86_64-pc-nto-qnx800` with zero
code changes. It stays clean because every OS/HW element is gated by a *specific*
`target_os` (`"linux"` / `"windows"` / `"macos"` / `"android"`), never
`cfg(unix)`, so the Linux HW paths (VAAPI, DRM/KMS, dma-buf, v4l2, ALSA/PipeWire)
are excluded on `nto` rather than pulled in. The Tier 1 (`std` transports over
the free SDP; `tokio`-on-QNX is the one open dependency question) and Tier 2 (QNX
Screen display sink + vendor VPU via the C-seam) roadmap is in
[PORTABILITY.md](PORTABILITY.md#spike-qnx-safety-certified-rtos).

## Migrating an existing pipeline?

Many `gst-launch-1.0` lines run unchanged through **`g2g-launch`**. Paste one in:

```sh
cargo run -p g2g-plugins --bin g2g-launch --features std -- \
  "videotestsrc num-buffers=30 ! videoconvert ! fakesink"
```

Element names mostly match (with aliases: `avdec_h264`→`ffmpegdec`,
`qtmux`→`mp4mux`, `autovideosink`→`waylandsink`/`kmssink`, ...). Inline caps
filters, `tee name=t` fan-out, muxer fan-in, and `decodebin`/`uridecodebin`/`playbin`
all parse. When a line *doesn't* port, you get a hint, not a bare error:

```
$ g2g-launch videotestsrc ! theoraenc ! fakesink
parse error: unknown element: theoraenc
  hint: `theoraenc` has no g2g element: no Theora encoder; use `vpxenc` (VP8/VP9) or `av1enc`
```

- **`g2g-launch -v ...`** prints each link's negotiated caps + memory domain (the `gst-launch -v` analog); **`--dot`** dumps a Graphviz graph.
- **`g2g-inspect`** is `gst-inspect-1.0`: list elements, dump one's properties/pads, or map a GStreamer name with `g2g-inspect --gst x264enc`. Scan an app's source with `--gst-scan app.c`.
- Migrate incrementally in either direction: `g2g-bridge` embeds a g2g sub-graph inside a GStreamer pipeline; `gstwrap` hosts an un-ported GStreamer element inside a g2g graph.

Full guide, including the equivalence cookbook and application/element porting:
**[PORTING.md](PORTING.md)**.

## Scripting: config files and Rhai

A `gst-launch` string is the one-liner. For version-controllable, generated, or
computed pipelines there are three more surfaces, all built on the same registry
and negotiation as `parse_launch` (so any element / caps / policy works):

- **Declarative graphs (JSON / YAML), `--graph`.** `nodes` + `edges`, with a
  `{ id, caps }` capsfilter shorthand and a top-level `pipeline:` escape hatch.
  Roles follow link degree (auto source / sink / muxer / auto-tee), and property
  values are typed by the target element exactly as in a launch string.

  ```sh
  cargo run -p g2g-plugins --bin g2g-launch --features declarative-yaml -- --graph pipe.yaml
  ```
  ```yaml
  # pipe.yaml
  nodes:
    - { id: src,  element: videotestsrc, props: { num-buffers: 30 } }
    - { id: cf,   caps: "video/x-raw,format=NV12" }   # a capsfilter shorthand
    - { id: sink, element: autovideosink }
  edges:
    - { from: src, to: cf }
    - { from: cf,  to: sink }
  ```

- **Rhai builder scripts, `--script`.** Where a document is a *fixed* graph, a
  script *computes* one (loops, parameters, conditionals) via a small builder API
  (`add` / `caps` / `set` / `link` / `link_leaky`) that emits the same graph model.
  Rhai is pure Rust, so this works on the same wasm / embedded targets the core
  reaches (`--features script-rhai`).

  ```rhai
  // Fan N cameras into one funnel, sized at runtime.
  add("funnel", "mix");
  for i in 0..num_cams { let id = "cam" + i; add("rtspsrc", id); set(id, "location", cams[i]); link(id, "mix"); }
  add("autovideosink", "screen"); link("mix", "screen");
  ```

- **`scriptelement`: per-frame logic in Rhai.** A raw-video transform whose
  `process(frame)` runs on every frame, the pure-Rust cousin of `pyelement`. The
  `frame` is a **zero-copy** handle to the live buffer: index it in place and read
  its geometry.

  ```
  g2g-launch videotestsrc ! scriptelement script="fn process(f){ f.invert(); }" ! autovideosink
  ```
  ```rhai
  fn process(frame) {
      // frame.width / .height / .format / .pts / .sequence / .len
      frame[3] = 128;          // per-pixel edit in place (convenient; interpreted)
      frame.invert();          // whole-frame native ops: fill(v) / invert() / apply_lut(lut)
  }
  ```

  Performance model (the NumPy rule): the script is the control plane, native code
  is the data plane. A per-pixel Rhai loop over an HD frame is milliseconds *per
  pixel-thousand* (interpreted — inherent to any embedded scripting language), so
  use it for logic, metadata, and small regions. For whole-frame transforms call a
  native bulk op (`invert()` ~1 ms/frame vs a per-pixel loop's seconds), and build
  the general per-value transform (brightness, gamma, threshold, ...) as a 256-entry
  `apply_lut(lut)`. For heavy per-pixel math, write a compiled element instead.

- **`scriptrouter`: script-decided routing to N outputs.** A 1-to-N demux whose
  `route(frame)` picks which output port each buffer goes to: an index (negative
  drops it), or an array like `[0, 1]` to **multicast** one buffer to several
  ports at once (a shared copy per port). Put an `appsink channel=...` on each
  branch and you have per-buffer routing into *your own* code / pipelines — the
  buffers go where the script says, and each channel is `pull()`ed live from
  Python/C/Rust just like a GStreamer `appsink`. Media-agnostic (audio, video,
  byte streams); `route` reads `frame.pts` / `.sequence` / `.keyframe` / `.len`
  and can peek bytes (`frame[i]`).

  ```
  # Split an audio stream to two consumers by parity; pull each from your app.
  g2g-launch whepsrc uri=... ! opusdec ! audioconvert ! \
    scriptrouter name=r script="fn route(f){ f.sequence % 2 }" \
    r.0 ! appsink channel=even   r.1 ! appsink channel=odd
  ```
  ```python
  even, odd = g2g.AppSink("even"), g2g.AppSink("odd")   # pull() each, feed anywhere
  ```

  A runnable end-to-end demo (routes to two pull channels drained live, with real
  `pull()` timing):
  `cargo run -p g2g-plugins --features script-rhai --example scriptrouter_appsink_egress`

## Embedded: heap-free pipelines on a bare-metal MCU

The MCU end of the spectrum is not a stripped-down build, it is the same graph
with a hard guarantee the rest of the ecosystem can't make: **`alloc` is an
optional feature, and the default build links no allocator at all.** That is the
safety / no-heap MCU market (MISRA, cert) GStreamer can't reach.

- **Static element model.** A heap-free pipeline is a compile-time-static graph
  of concrete typed elements (`g2g_core::staticelem`: `StaticSource` /
  `StaticTransform` / `StaticSink` with `async fn` in trait, const-arity
  runners, a `Chain` combinator), so every stage's future is unboxed, no `dyn`,
  no `Box`, no allocation. Buffers are lent zero-copy from a const-generic
  `StaticLendRing` sized at compile time.
- **The guarantees are machine-checked in CI, not asserted.** The linked archive
  carries **zero allocator symbols** and **zero panic symbols** (`tools/noalloc-check.sh`);
  a gc-sectioned ELF is measured for **ROM / static RAM / worst-case stack** and
  budget-enforced (`tools/footprint-report.sh`); the pipeline then *executes* on
  an emulated Cortex-M (`tools/qemu-check.sh`) and a per-frame **timing / jitter**
  report runs under deterministic QEMU `-icount` (`tools/timing-report.sh`). App
  code on this surface needs zero `unsafe`.
- **One graph, four executors.** The same static pipeline runs bit-exact under a
  bare poll loop, **Embassy**, **FreeRTOS** (C-ABI staticlib), and **Zephyr**
  (a drop-in Zephyr module the app lists in its west manifest).
- **Integrates with your existing C, both directions.** A C/RTOS app can link
  the pipeline as a static library and call in, *or* your existing C drivers can
  be the peripheral: `g2g-mcu::cffi`'s `CFrameGrabber` / `CPacketSender` wrap C
  capture/send **function pointers**, and `step_source_sink` hands control back
  to your superloop after each frame. Zero Rust on the driver side; proven from a
  real C caller (`examples/g2g-cffi`), still heap-free and data-panic-free.
- **`g2g-mcu` peripheral elements.** Heap-free elements written against
  `embedded-hal` trait seams rather than chip registers, so the driver logic is
  host-tested against the datasheet with mock peripherals and a board port is
  just the vendor HAL's trait impls: `SpiDisplaySink` (ST7789 / ILI9341, whole
  frame or banded streaming for panels too large to ring-buffer), `GrabberSrc`
  (DCMI/CSI camera), `PcmSink` (I2S/SAI), the fixed-point G.711 and IMA ADPCM
  codecs (bit-exact vs ffmpeg), the hardware JPEG-decode and H.264-encode seams
  (`HwJpegDec` / `HwH264Enc`, the peripheral reached over an `embedded-hal` or a
  C-function-pointer driver), `YuyvToI420` (heap-free camera-4:2:2 → encoder-4:2:0
  convert), and `RtpSink`.
- **Interrupt/DMA-driven capture.** Real capture runs in interrupt context, so
  `SpscFrameRing` is a lock-free, heap-free single-producer/single-consumer FIFO
  a **DMA-completion ISR** fills while the pipeline drains it (`SpscCaptureSrc`,
  sleeping on `wfi` between frames), with bounded back-pressure (a full ring
  drops and counts, never stalls the interrupt). Proven on emulated Cortex-M: a
  SysTick interrupt feeds a `capture → G.711 → checksum` pipeline lossless and in
  order, bit-exact against synchronous delivery. Atomic load/store only, so it
  works on cores without CAS (`thumbv6m`).
- **Runtime fault recovery.** A supervisor (`g2g_core::supervise`) turns a
  returned peripheral fault into a **bounded, deterministic** action instead of
  aborting: a `FaultPolicy` chooses retry / skip (degraded mode) / reset /
  escalate, a `Recover` seam re-initializes the faulting stage (re-arm the
  camera, re-open the socket), and a `Watchdog` is petted only on real forward
  progress, so a wedged or escalated pipeline stops petting and a hardware
  watchdog resets the chip. A `SupervisorReport` accounts every fault for the
  safety case. Proven on emulated Cortex-M: the pipeline recovers a latched
  mid-stream capture fault (all frames still delivered) and escalates a dead
  peripheral within its bounded ladder without hanging.
- **Receive direction (RTP ingress + jitter buffer).** The inverse of the
  capture→egress flagship: `RtpSrc` receives and parses RTP (a wire-tolerant,
  bounds-checked header parser shared with the std depayloader), a heap-free
  `JitterBuffer<N, BYTES>` absorbs arrival jitter and reorders by sequence
  number, handling reorder / duplicate / late / loss explicitly and countably,
  and `G711Dec` decodes. Proven on emulated Cortex-M: a reordered RTP wire is
  reconstructed to the ordered PCM stream, verified by an order-sensitive hash
  against an independent in-order decode.
- **I2C sensors + UART transport.** Beyond the media pipeline: `Sht3xSrc` is a
  real SHT3x temperature/humidity driver over `embedded-hal` I2C (datasheet
  single-shot command, CRC-8 validation with the datasheet's `0xBEEF→0x92`
  vector, fixed-point conversion), and `UartSink` / `UartSrc` are a byte-stream
  egress / ingress over local serial seams. An I2C-sensor→UART telemetry
  pipeline is proven on emulated Cortex-M.
- **A checkable safety case.** [`docs/safety/`](docs/safety/) carries a
  requirements traceability matrix (15 requirements, each linked to the proof
  script / test / CI job that verifies it) and a safety manual (conditions of
  use, assumptions, the `unsafe` inventory). `tools/traceability-check.sh` fails
  in CI if any cited evidence goes missing, so the matrix can't drift from the
  code; `tools/qualification-kit.sh` runs the whole proof set into one report. A
  down-payment on a functional-safety case, not a certificate (pre-1.0,
  emulated not silicon).
- **ARM *and* RISC-V.** The static element model is ISA-agnostic pure Rust, so
  the no-alloc core and `g2g-mcu` build unchanged for `riscv32imafc`
  (ESP32-P4 class). The heap-free + panic-free symbol proofs and the footprint
  report run on **both** `thumbv7em` and RISC-V — the portability claim is
  machine-checked, not asserted.
- **The flagship deterministic-audio graph.** `capture -> convert -> resample ->
  mix -> encode -> RTP` composed as **one** static heap-free pipeline, fully
  fixed-point, so its RTP wire bytes are bit-exact across every target: pinned by
  a host test (DSP validated against an independent float reference) and
  re-verified on-target on all four executors.

**`g2g-mcugen`, the host graph compiler.** Develop and test on Linux, ship a
bounded static build to the MCU: a declarative graph document compiles to the
monomorphized static pipeline, with every ring sized from the graph's frame
geometry and a total ring-memory budget reported. It is a **general** MCU graph
compiler, not audio-only, spanning an audio catalog and a video / display one:

```yaml
# camera -> SPI panel, one static pipeline. `g2g-mcugen display.yaml -o graph.rs`
name: display
frame_ns: 33333333   # ~30 fps
frames: 64
nodes:
  - { id: cam,  element: grabbersrc,     props: { width-px: 4, height-px: 4, format: rgba8888 } }
  - { id: disp, element: spidisplaysink, props: { driver: st7789, width-px: 4, height-px: 4 } }
edges:
  - { from: cam, to: disp }
```

A mis-wired graph (an encoder fed the wrong sample width, a mixer whose inputs
disagree, a display fed the wrong pixel format) is rejected with a diagnostic
before a line of Rust is emitted, and the generated pipeline reproduces the
hand-written reference's wire output byte-for-byte (checked in CI for both
catalogs, `tools/mcugen-check.sh`).

## The four pillars

1. **Async execution.** Every element is a cooperative `Future`. The
   framework is runtime-agnostic (Tokio on servers, Embassy on RTOS,
   `wasm-bindgen-futures` in the browser).
2. **Hardware-first, zero-copy.** Buffers live in DMABUF / Vulkan /
   CUDA / D3D11 / WebGPU memory domains; CPU memory copies are treated
   as system faults.
3. **`no_std`, `alloc`-optional, sans-IO core.** The same pipeline shape runs on
   a bare-metal Cortex-M with no heap, an RTOS (Embassy / FreeRTOS / Zephyr), a
   multi-threaded server, a GPU-resident pipeline, or `wasm32` (see
   [Portability](#portability-one-pipeline-five-targets)).
4. **First-class ML.** Tensor allocation, reshaping, and pipeline
   batching are part of graph orchestration.

## Workspace

| Crate | Role | Profile |
| :--- | :--- | :--- |
| `g2g-core` | Traits, `Frame`/`PipelinePacket`, caps algebra, clock, runner, static element model. | `no_std`, `alloc` optional |
| `g2g-mcu` | Heap-free MCU peripheral elements (SPI display incl. banded streaming, camera / PCM capture, I2C sensor, UART, G.711 / ADPCM codecs, hardware JPEG-decode / H.264-encode seams, RTP egress + ingress, jitter buffer, fault-recovery watchdog) over `embedded-hal` / C-callback seams. | `no_std`, no alloc |
| `g2g-mcugen` | Host graph compiler: a declarative MCU graph (YAML/JSON, audio or video/display) → a monomorphized static pipeline (heap-free Rust). | `std` (host tool) |
| `g2g-plugin` | SDK for dynamically loadable plugins (`declare_plugin!` + ABI tag). | `no_std + alloc` |
| `g2g-plugins` | Sources/sinks/transforms (RTSP, RTP in/out, HTTP/HLS/DASH/RTMP ingest, V4L2 / PipeWire / MF capture, ffmpeg, VAAPI, MF, VideoToolbox (macOS), MediaCodec (Android), Wayland, KMS, WASAPI, ALSA / PulseAudio / PipeWire audio, compositor, Embassy, web), container mux/demux (MP4, MPEG-TS, Matroska/WebM, FLV, Ogg), codec parsers + encoders (AV1, VP8/9, MJPEG), the tag system, and the `gst-launch` text DSL. | mixed |
| `g2g-ml` | ORT, Burn, WgpuPreprocess, TensorPostprocess, multi-stream tensor batcher. | `std` |
| `g2g-bridge` | GStreamer C-FFI bridge. | `std` |
| `g2g-python` | Hosts gst-python-ml elements in-process (embedded CPython via pyo3). | `std` |
| `g2g-capi` | C ABI (cdylib/staticlib + `g2g.h`): launch pipelines + bus + appsrc/appsink from any language. | `std` |
| `g2g-pyapi` | Python (pyo3) bindings: drive pipelines + bus + appsrc/appsink. | `std` |

## Build

Stable Rust, MSRV 1.85, `resolver = "2"`.

```sh
cargo check --workspace          # no_std baseline
cargo test  --workspace          # default test suite (no platform features)
cargo clippy --workspace --all-targets
```

OS-coupled elements live behind cargo features:

| Element | Feature | Platform / system dep |
| :--- | :--- | :--- |
| `RtspSrc` | `rtsp` | retina |
| `H264Parse` | (default) | — |
| `FfmpegH264Dec` (sw / `NvdecCuvid` / `NvdecCuda` / `Vaapi`) | `ffmpeg` | Linux + libavcodec |
| `VaapiH264Dec` | `vaapi` | Linux + libva + GBM |
| `MfDecode` / `MfEncode` / `MfAacEncode` / `MfAacDecode` | `mf-decode`, `mf-encode`, `mf-aac` | Windows + Media Foundation |
| `VtDecode` / `VtEncode` (H.264 / H.265, validated on the CI Mac) | `vtdecode`, `vtencode` | macOS + VideoToolbox |
| `MediaCodecDec` (H.264 / H.265, on-device validated; zero-copy GPU output via `with_gpu_output`) | `mediacodec`, `mediacodec-wgpu` | Android + NDK MediaCodec (+ wgpu / Vulkan for GPU output) |
| `WaylandSink` | `wayland-sink` | Linux + Wayland |
| `KmsSink` | `kms-sink` | Linux + libdrm; needs DRM master / tty |
| `D3D11Sink` | `d3d11-sink` | Windows |
| `NvDec` (native NVDEC H.264/H.265 → CUDA NV12, NVCUVID) | `nvdec` | Linux + NVIDIA driver (libnvcuvid) |
| `NvEnc` (native NVENC CUDA NV12 → H.264/H.265) | `nvenc` | Linux + NVIDIA driver (libnvidia-encode) |
| `CudaDownload` (CUDA → System), `CudaUpload` (System → CUDA) | `cuda` | Linux + NVIDIA driver (libcuda) |
| `CudaGlSink` (CUDA-GL present), `CudaKmsSink` (CUDA-GL on KMS) | `cuda-gl`, `cuda-kms` | Linux + NVIDIA + EGL + GL (+ libdrm for KMS) |
| `CudaToWgpu` / `WgpuToCuda` (CUDA ↔ wgpu zero-copy bridge) | `cuda-wgpu` | Linux + NVIDIA + Vulkan |
| `UdpSink` + RTP packetizer | `udp-egress` | — |
| `UdpSrc` (RTP ingest + jitter buffer + RTCP/NACK) | `udp-ingress` | — |
| `RtmpSrc` (RTMP publisher ingest) | `rtmp` | — |
| `WebRtcSink` (WHIP egress, H.264 + Opus) / `WebRtcWhepSrc` (WHEP ingest, H.264), via str0m: ICE/DTLS/SRTP, trickle ICE + ICE restart, NACK/RTX | `webrtc` | str0m (rust-crypto) + reqwest |
| `WebRtcDataSrc` / `WebRtcDataSink` (P2P data channels on SCTP) | `webrtc` | str0m |
| `LiveKitSink` (publish into a LiveKit room: JWT + protobuf signalling) | `webrtc-livekit` | + tokio-tungstenite |
| `HttpSrc` (HTTP(S) byte-stream source) | `http-src` | reqwest |
| `HlsSrc` (HLS: TS + fMP4/CMAF, live, AES-128 / SAMPLE-AES) | `hls` | reqwest + aes |
| `DashSrc` (DASH: SegmentTemplate / SegmentTimeline, live) | `dash` | reqwest + roxmltree |
| `V4l2Src` | `v4l2` | Linux + V4L2 (`/dev/videoN`) |
| `WasapiSink` / `WasapiSrc` | `wasapi-sink`, `wasapi-src` | Windows |
| `AlsaSink` | `alsa-sink` | Linux + libasound |
| `PulseSink` | `pulse-sink` | Linux + libpulse |
| `PipeWireSink` / `PipeWireSrc` (audio) | `pipewire` | Linux + libpipewire |
| `MfVideoSrc` (camera) | `mf-video-src` | Windows + Media Foundation |
| `Av1Enc` (pure-Rust `rav1e`) | `av1-encode` | — |
| `VpxEnc` (VP8 / VP9 via libvpx) | `vpx` | libvpx |
| `MjpegDec` / `MjpegEnc` (pure Rust) | `mjpeg`, `mjpeg-encode` | — |
| `AnalyticsOverlay` (CPU) / `VelloAnalyticsOverlay` (GPU) / `WgpuSink` | `analytics`, `vello-overlay`, `wgpu-sink` | wgpu (GPU variants) |
| `OrtInference` (+ CUDA / DirectML EPs) | `ort`, `cuda`, `directml` (in `g2g-ml`) | onnxruntime |
| `BurnInference` | `burn` (in `g2g-ml`) | wgpu (Vulkan / Metal / DX12) |
| `WgpuPreprocess` (NV12 or Android RGBA GPU texture in, NCHW tensor out) | `wgpu`, `mediacodec-wgpu` (in `g2g-ml`) | wgpu |
| Embassy / RTOS pool + clock | `embassy`, `embassy-link` | — |
| Browser elements | `web`, `web-codecs` | `wasm32-unknown-unknown` |

The container parsers and muxers (`mp4src` / `mp4sink`, `tsdemux` / `mpegtsmux`,
`matroskademux` / `matroskamux`, `flvdemux` / `flvmux`, `oggdemux`,
`fmp4demux`), the bitstream parsers (`h264parse`, `h265parse`, `aacparse`,
`opusparse`, `vp8parse`, `vp9parse`, `av1parse`), the software video/audio
transforms (`videoscale` / `videorate` / `videocrop` / `videoflip` /
`videobalance` / `videobox` / `alpha` / `gamma` / `deinterlace` / `timeoverlay`,
`audioconvert` / `audioresample` / `audiomixer` / `volume` / `audiopanorama` /
`audioamplify` / `audioecho` / `level` / `cutter` / `equalizer-3bands` /
`spectrum`), the flow-control elements (`concat` / `input-selector` /
`output-selector` / `progressreport`), the `compositor`, the tag system, and the
`gst-launch` text DSL (`parse_launch` / `gst-inspect`) are all in the pure
`no_std + alloc` default build. The std build adds `clockoverlay`, the
`multifilesink` / `multifilesrc` image-sequence pair, and `splitmuxsink`
(segmented recording, `muxer=mp4|matroska|mpegts`).

## Sample pipelines

The graph API is `run_source_transform_sink` / `run_linear_chain` /
`run_source_fanout` / `run_muxer_sink` over typed elements (no string-keyed
factory lookup). Examples are condensed; full versions live in the integration
tests under `g2g-plugins/tests/`.

### RTSP → ffmpeg decode → Wayland window

```rust
let src  = RtspSrc::new("rtsp://localhost:8554/pattern");
let dec  = FfmpegH264Dec::new().with_output_format(OutputFormat::Nv12);
let sink = WaylandSink::new();

run_source_transform_sink(src, dec, sink, &clock, LatencyProfile::Live).await?;
```

Features: `rtsp ffmpeg wayland-sink`.

### RTSP → NVDEC (CUDA device memory) → CUDA-GL display

Zero-copy after decode: NV12 stays in CUDA device memory until the GL
fragment shader samples it.

```rust
let src  = RtspSrc::new(url);
let dec  = FfmpegH264Dec::with_backend(Backend::NvdecCuda);   // MemoryDomain::Cuda
let sink = CudaGlSink::new();                                  // EGL on Wayland, NV12 shader

run_source_transform_sink(src, dec, sink, &clock, LatencyProfile::Live).await?;
```

Features: `rtsp ffmpeg cuda cuda-gl`. Linux + NVIDIA only. See
[DESIGN.md §4.11.5](DESIGN.md).

### Native NVDEC → NVENC transcode, GPU-resident, with domain auto-plug

`NvDec` / `NvEnc` drive NVCUVID / NVENC directly (no libavcodec). The decode
stays in `MemoryDomain::Cuda` straight into the encoder. Memory-domain
negotiation settles a shared domain when one exists; where it can't (a CPU-side
NV12 source feeding the CUDA-only `NvEnc`), `auto_plug_cuda_converters` splices a
`CudaUpload` automatically — no hand-wiring.

```rust
let mut g: Graph<GraphNode> = Graph::new();
let src = g.add_source(GraphNode::source(my_nv12_source));   // System NV12
let enc = g.add_transform(GraphNode::element(NvEnc::new())); // CUDA NV12 → H.264
let snk = g.add_sink(GraphNode::element(my_h264_sink));
g.link(src, enc).unwrap();
g.link(enc, snk).unwrap();

let g = auto_plug_cuda_converters(g);   // splices CudaUpload: src → [CudaUpload] → enc → snk
run_graph(g, &clock, LatencyProfile::Live).await?;
```

Features: `nvenc` (`nvdec` for the decoder). Linux + NVIDIA only. `NvDec`
itself is multi-domain: driven by downstream demand it keeps frames on the GPU
(zero-copy) or downloads to System.

### RTSP → decode → KMS (tty / no compositor)

```rust
let src  = RtspSrc::new(url);
let dec  = FfmpegH264Dec::new().with_output_format(OutputFormat::Nv12);
let sink = KmsSink::open("/dev/dri/card0")?;

run_source_transform_sink(src, dec, sink, &clock, LatencyProfile::Live).await?;
```

Features: `rtsp ffmpeg kms-sink`. Run from a tty after stopping the
display manager (KMS sink needs DRM master).

### RTSP → decode → ML preprocess → ORT inference → postprocess

```rust
let src       = RtspSrc::new(url);
let dec       = FfmpegH264Dec::new().with_output_format(OutputFormat::Nv12);
let preproc   = WgpuPreprocess::new(w, h);                 // NV12 -> f32 NCHW on GPU
let inference = OrtInference::from_memory_with_cuda(model_bytes)?;
let post      = TensorPostprocess::topk_classification(5);

run_linear_chain(src, vec![&mut dec, &mut preproc, &mut inference, &mut post],
                 FakeSink::new(), &clock, LatencyProfile::Live).await?;
```

Features: `rtsp ffmpeg` (plugins) + `wgpu cuda` (g2g-ml). The CUDA execution
provider falls back to CPU if no CUDA runtime is present.

### Android: MediaCodec decode → GPU → ML preprocess (zero-copy)

```rust
// Decode on the NDK MediaCodec and keep the frame on the GPU as an RGBA wgpu
// texture (no CPU NV12 pack); WgpuPreprocess samples it straight into a tensor.
let dec     = MediaCodecDec::h264().with_gpu_output();   // MemoryDomain::WgpuTexture (RGBA)
let preproc = WgpuPreprocess::new();                     // samples the texture -> f32 NCHW
// dec -> preproc -> OrtInference / BurnInference, all on the GPU
```

Features: `mediacodec-wgpu` (plugins) + `mediacodec-wgpu` (g2g-ml). Android only,
validated on a Pixel 10a. The decoded `AHardwareBuffer` is imported into Vulkan
and converted to RGBA through an immutable `VkSamplerYcbcrConversion` compute
pass (the conversion wgpu's bind-group API cannot express), then handed
downstream as a `wgpu::Texture` &mdash; the frame never touches the CPU.

### File capture → H.264 parse → fMP4 record

```rust
let src   = FileSrc::open("in.h264")?;
let parse = H264Parse::new();
let sink  = Mp4Sink::open("out.mp4")?;

run_source_transform_sink(src, parse, sink, &clock, LatencyProfile::Live).await?;
```

### MPEG-TS file → demux → H.264 parse → decode → Wayland

The container demuxers (`tsdemux`, `matroskademux`, `flvdemux`, `oggdemux`,
`fmp4demux`) accept a `Caps::ByteStream` and split out elementary streams.

```rust
let src   = FileSrc::new("clip.ts", Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs });
let demux = TsDemux::new().with_stream(TsStream::H264);   // PAT/PMT/PES -> Annex-B
let parse = H264Parse::new();
let dec   = FfmpegH264Dec::new().with_output_format(OutputFormat::Nv12);
let sink  = WaylandSink::new();

run_linear_chain(src, vec![&mut demux, &mut parse, &mut dec], sink,
                 &clock, LatencyProfile::Live).await?;
```

Features: `ffmpeg wayland-sink`.

### Adaptive streaming: HLS / DASH → decode → display

```rust
let src   = HlsSrc::new("https://example.com/master.m3u8");  // or DashSrc::new(mpd_url)
let demux = TsDemux::new().with_stream(TsStream::H264);
let parse = H264Parse::new();
let dec   = FfmpegH264Dec::new().with_output_format(OutputFormat::Nv12);
let sink  = WaylandSink::new();

run_linear_chain(src, vec![&mut demux, &mut parse, &mut dec], sink,
                 &clock, LatencyProfile::Live).await?;
```

Features: `hls ffmpeg wayland-sink` (`dash` for the DASH front end). `HlsSrc`
follows live playlist reloads and decrypts AES-128 / SAMPLE-AES segments;
`DashSrc` handles `SegmentTemplate` / `SegmentTimeline` and dynamic (live) MPDs.

### `gst-launch` text pipeline

`parse_launch` builds a runnable `Graph` from a GStreamer-style string against
the `default_registry`, including caps filters, `tee` branching, and muxer
fan-in. `Registry::inspect(name)` is the `gst-inspect` analog.

```rust
let graph = parse_launch(
    "videotestsrc num-buffers=90 pattern=ball ! video/x-raw,format=nv12 \
     ! videoflip method=rotate-180 ! matroskamux ! filesink location=out.mkv",
    &default_registry(),
)?;
run_graph(graph, &clock, LatencyProfile::Live).await?;
```

Registered launch elements include `videotestsrc` / `audiotestsrc`, the SW
transforms, the demuxers (`tsdemux`, `matroskademux`, `flvdemux`, `oggdemux`)
and muxers (`mpegtsmux`, `matroskamux`, `flvmux`, `funnel`, `audiomixer`),
`filesrc` / `filesink`, and `fakesink`. Feature-gated capture / decode / display
elements still need explicit Rust construction.

### Camera → encode → RTP egress over UDP

```rust
let src  = VideoTestSrc::new(1920, 1080, 30, 0);         // RGBA test pattern, unbounded
let enc  = MfEncode::new_low_latency();                  // Windows; on Linux use the bridge
let sink = UdpSink::new("239.0.0.1:5004".parse()?)
    .with_rtp(96, 0x1234_5678);                          // payload type, SSRC

run_source_transform_sink(src, enc, sink, &clock, LatencyProfile::Live).await?;
```

Features: `udp-egress` (plus the platform encoder feature). `UdpSink` honors
receive-side NACK by retransmitting from a bounded send history
(`with_retransmit`).

### RTP ingress over UDP → ffmpeg decode → Wayland

The receive-side inverse, with a jitter buffer (reorder / bounded-latency loss
handling) and RTCP feedback (periodic receiver reports, NACK on gaps) built in.

```rust
let src  = UdpSrc::new("0.0.0.0:5004".parse()?)
    .with_jitter(50, 64)                                 // 50 ms hold, 64-packet depth
    .with_rtcp(1000, true);                              // 1 s reports, NACK enabled
let dec  = FfmpegH264Dec::new().with_output_format(OutputFormat::Nv12);
let sink = WaylandSink::new();

run_source_transform_sink(src, dec, sink, &clock, LatencyProfile::Live).await?;
```

Features: `udp-ingress ffmpeg wayland-sink`.

### Picture-in-picture: webcam over a test pattern (compositor)

```rust
let bg   = VideoTestSrc::new(1280, 720, 30, 0).with_pattern(Pattern::MovingBar);
let cam  = V4l2Src::new("/dev/video0").with_size(640, 480);   // -> VideoConvert(RGBA) -> VideoScale
let comp = Compositor::new(1280, 720, vec![
    CompositorPad::at(0, 0),                              // background, timing driver
    CompositorPad::at(940, 460).with_zorder(1),          // webcam inset
]);
// bg -> comp.input(0); cam -> rgba -> scale -> comp.input(1); comp -> sink (see tests).
```

Features: `v4l2 wayland-sink`. Full graph in
[`g2g-plugins/tests/pip_smoke.rs`](g2g-plugins/tests/pip_smoke.rs).

## Running smoke tests

Most integration tests are marked `#[ignore]` because they need a live RTSP
feed and/or a display. The pattern is the same across recipes:

```sh
cargo test -p g2g-plugins \
  --features "<comma-separated feature list>" \
  --test <test_name> -- --ignored --nocapture
```

### A standing RTSP feed

A typical loopback setup uses `mediamtx` as the relay and `ffmpeg` as the
publisher. In one terminal:

```sh
mediamtx                  # listens on 8554/tcp by default
```

In a second terminal, push a synthetic H.264 feed into it:

```sh
ffmpeg -re -f lavfi -i testsrc=size=1280x720:rate=30 \
       -c:v libx264 -preset ultrafast -tune zerolatency -g 30 \
       -f rtsp -rtsp_transport tcp rtsp://localhost:8554/pattern
```

A public RTSP feed (Wowza demo, IP camera on the LAN, etc.) also works —
the smoke tests don't care where the stream comes from.

### Software decode + Wayland

```sh
G2G_RTSP_TEST_URL=rtsp://localhost:8554/pattern \
  cargo test -p g2g-plugins \
  --features "rtsp ffmpeg wayland-sink" \
  --test wayland_smoke -- --ignored --nocapture
```

A window titled "glass2glass" appears showing the feed.

### NVIDIA NVDEC (system memory) + Wayland

```sh
G2G_DECODER=nvdec \
G2G_RTSP_TEST_URL=rtsp://localhost:8554/pattern \
G2G_TARGET_FRAMES=300 \
  cargo test -p g2g-plugins \
  --features "rtsp ffmpeg wayland-sink" \
  --test wayland_smoke -- --ignored --nocapture
```

`G2G_TARGET_FRAMES >= 300` is needed to amortize cuvid startup (libnvcuvid
load, CUDA context, surface pool alloc) for meaningful p50 / p95 latency
numbers. Compare against `G2G_DECODER=software` on the same feed.

### NVIDIA NVDEC → CUDA → CUDA-GL zero-copy display

```sh
G2G_RTSP_TEST_URL=rtsp://localhost:8554/pattern \
  cargo test -p g2g-plugins \
  --features "rtsp ffmpeg cuda cuda-gl" \
  --test cuda_gl_smoke -- --ignored --nocapture
```

### KMS scanout (no compositor)

Drop to a tty, stop the display manager, then:

```sh
G2G_RTSP_TEST_URL=rtsp://localhost:8554/pattern \
  cargo test -p g2g-plugins \
  --features "rtsp ffmpeg kms-sink" \
  --test kms_smoke -- --ignored --nocapture
```

### ML inference

```sh
# ORT with the CUDA execution provider (silently falls back to CPU):
cargo test -p g2g-ml --features cuda --test ort_inference -- --nocapture

# Pure-Rust Burn over wgpu (any Vulkan/Metal/DX12 adapter):
cargo test -p g2g-ml --features burn --test burn_inference -- --nocapture
```

### UDP egress (loopback, no network)

```sh
cargo test -p g2g-plugins --features udp-egress --test m47_udp_egress -- --nocapture
```

Binds a UDP receiver on localhost, drives the H.264 RTP packetizer, and
asserts the datagrams parse back byte-exactly, with sequence numbers,
marker bit, and FU-A reassembly all correct.

### UDP ingress + resilience (loopback, no network)

```sh
cargo test -p g2g-plugins --features "udp-ingress udp-egress" --test udp_loopback -- --nocapture
```

End-to-end over localhost: depayload round-trip, the jitter buffer reordering
out-of-order packets, and NACK-driven recovery (a lossy relay drops chosen
sequences; the receiver NACKs, the sender retransmits, every access unit is
recovered in order).

## Android on-device testing

The Android elements (`mediacodec` decode/encode, `mediacodec-wgpu` zero-copy
decode->GPU, `aaudio` audio, `camera2` capture) are cross-compiled in CI but
validated on a real device. Each ships an on-device probe + a smoke script that
builds just that test binary, `adb push`es it to `/data/local/tmp`, runs it, and
checks the libtest summary.

**Prerequisites:**

- `adb` on `PATH` with a phone attached and USB debugging authorised
  (`adb devices` lists it as `device`).
- `cargo-ndk` (`cargo install cargo-ndk`).
- The rustup target: `rustup target add aarch64-linux-android`.
- The Android NDK, with `ANDROID_NDK_HOME` pointing at it (cargo-ndk uses that to
  find the toolchain), e.g. `export ANDROID_NDK_HOME=$HOME/android-ndk-r27c`.

**Run a probe:**

```sh
export ANDROID_NDK_HOME=$HOME/android-ndk-r27c

tools/android-mediacodec-smoke.sh        # decode  (H.264 + HEVC -> NV12)
tools/android-mediacodec-enc-smoke.sh    # encode  (NV12 -> Annex-B H.264)
tools/android-aaudio-smoke.sh            # audio   (render; mic capture best-effort)
tools/android-camera2-smoke.sh           # camera  (caps + FFI; capture best-effort)
tools/android-surface-present-smoke.sh   # decode -> GPU -> on-screen present
tools/android-nnapi-smoke.sh             # ML inference (NNAPI + XNNPACK ORT EPs)
tools/android-nnapi-conv-smoke.sh        # ML on the Edge TPU (int8 conv, NNAPI placement + DarwiNN logcat)
tools/android-camera-tpu-smoke.sh        # live camera -> quantize -> Edge TPU inference, end to end
```

Each takes an optional ABI argument (default `arm64-v8a`; also `x86_64`,
`armeabi-v7a`). To drive an element by hand, build the test the same way and push
the binary yourself:

```sh
cargo ndk --platform 26 -t arm64-v8a build --release \
  -p g2g-plugins --features camera2 --test android_camera2_probe
adb push target/aarch64-linux-android/release/deps/android_camera2_probe-<hash> /data/local/tmp/probe
adb shell /data/local/tmp/probe --nocapture --test-threads=1
```

(`--platform`: 24 for `AImageReader`, 26 for `AHardwareBuffer` / AAudio.)

**Permission caveats.** A bare `/data/local/tmp` binary has no app manifest, so
the permission-gated capture paths can't run there: **mic capture** needs
`RECORD_AUDIO` and **camera capture** needs `CAMERA`. Those probes assert the
parts they can check headlessly (device open, caps, FFI linkage, encode/render)
and report the denial rather than failing; full capture and a true on-screen
`SurfaceView` present need an APK harness. If `adb` reports "insufficient
permissions", run `adb kill-server && adb start-server` and re-accept the prompt
on the phone.

## System dependencies

The cargo features pull pure Rust crates; OS-level dependencies must be
present on the host.

| Distro | Decoder (`ffmpeg`) | Wayland sink | KMS sink | VAAPI |
| :--- | :--- | :--- | :--- | :--- |
| Fedora | `ffmpeg-devel` (RPM Fusion) or `ffmpeg-free-devel` | `wayland-devel` | `libdrm-devel` | `libva-devel` |
| Debian / Ubuntu | `libavcodec-dev libavformat-dev libavutil-dev libswscale-dev` | `libwayland-dev` | `libdrm-dev` | `libva-dev` |
| Arch | `ffmpeg` | `wayland` | `libdrm` | `libva` |

For the CUDA path: install the NVIDIA driver and CUDA runtime your
distribution ships, and ensure `libnvcuvid.so` and `libcuda.so` are on the
linker path. The `ffmpeg` build must include cuvid support if you intend
to use `Backend::NvdecCuvid` / `Backend::NvdecCuda`.

`mediamtx` for the loopback RTSP server is available as a single binary
from <https://github.com/bluenviron/mediamtx/releases>; some distros also
package it.

## Layout

```
g2g-core/        # traits, runner, solver, frame, caps, clock, static element model
g2g-mcu/         # heap-free MCU peripheral elements over embedded-hal seams
g2g-mcugen/      # host graph compiler: a declarative MCU graph -> a static pipeline (heap-free Rust)
g2g-plugin/      # dynamic-plugin SDK (declare_plugin! + ABI tag)
g2g-plugins/     # all source/sink/transform elements
g2g-ml/          # ORT, Burn, WgpuPreprocess, batcher
g2g-bridge/      # GStreamer C-FFI bridge (libgstglass2glass.so)
g2g-python/      # gst-python-ml element host (embedded CPython)
g2g-capi/        # C ABI (cdylib/staticlib + include/g2g.h)
g2g-pyapi/       # Python (pyo3) bindings
xtask/           # dev-command crate (cargo xtask ci | test --here | size | wasm | bench | ffi-probe)
g2g-bench/       # criterion benchmarks (excluded from the workspace)
DESIGN.md        # architecture specification
DEVTOOLS.md      # developer tooling reference
docs/            # GitHub Pages site
```

## License

All crates are LGPL v2.1+.

See [LICENSE](LICENSE).
