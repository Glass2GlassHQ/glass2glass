# glass2glass (`g2g`)

[![CI](https://github.com/Glass2GlassHQ/glass2glass/actions/workflows/ci.yml/badge.svg)](https://github.com/Glass2GlassHQ/glass2glass/actions/workflows/ci.yml)

A pure-Rust multimedia graph framework. The core is `no_std`, `alloc`-optional,
sans-IO, and async, so one typed graph runs unchanged across
**MCU · RTOS · CPU · GPU · WASM**: a heap-free bare-metal Cortex-M at one end,
a GPU-resident server pipeline at the other. You write the graph once. The
deployment shell picks the executor and the hardware elements.

The name is the metric the project optimizes: **glass-to-glass latency**, from
photon capture to hardware presentation.

Architecture: [design/README.md](design/README.md). Developer tooling
(`cargo xtask`, the pipeline visualizer, the caps explainer, benchmarks):
[DEVTOOLS.md](DEVTOOLS.md).

- [Quick start](#quick-start)
- [Portability: one pipeline, five targets](#portability-one-pipeline-five-targets)
- [Migrating an existing pipeline?](#migrating-an-existing-pipeline)
- [Scripting: config files and Rhai](#scripting-config-files-and-rhai)
- [Embedded: heap-free pipelines on a bare-metal MCU](#embedded-heap-free-pipelines-on-a-bare-metal-mcu)
- [The four pillars](#the-four-pillars)
- [Workspace](#workspace)
- [Build](#build)
- [Sample pipelines](#sample-pipelines)
- [Running smoke tests](#running-smoke-tests)
- [Android on-device testing](#android-on-device-testing)
- [Host validation](#host-validation)
- [System dependencies](#system-dependencies)
- [Layout](#layout)
- [License](#license)

## Quick start

A complete program:

```toml
# Cargo.toml
[dependencies]
tokio       = { version = "1", features = ["macros", "rt-multi-thread"] }
g2g-core    = { git = "https://github.com/Glass2GlassHQ/glass2glass" }
g2g-plugins = { git = "https://github.com/Glass2GlassHQ/glass2glass", features = ["std"] }
```

```rust
// src/main.rs
use g2g_core::runtime::{parse_launch, run_graph, LatencyProfile};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;

#[tokio::main]
async fn main() {
    let registry = default_registry();
    let graph = parse_launch(
        &registry,
        "videotestsrc num-buffers=90 pattern=ball ! videoconvert ! autovideosink",
    )
    .expect("parse");
    run_graph(graph, &WallClock::new(), LatencyProfile::Live)
        .await
        .expect("run");
}
```

With the `wayland-sink` feature, `autovideosink` opens a window. Without it,
`autovideosink` becomes `fakesink` and the program runs headless. To run a
pipeline straight from the repo instead:

```sh
cargo run -p g2g-plugins --bin g2g-launch --features std -- \
  "videotestsrc num-buffers=90 pattern=ball ! videoconvert ! autovideosink"
```

## Portability: one pipeline, five targets

`g2g-core` is pure Rust, `no_std` with `alloc` optional, and sans-IO. The
graph, the element traits, `Caps` negotiation, and the runner are the same on
every target. Only the executor and the hardware elements change.

| Target | What runs | How |
| :--- | :--- | :--- |
| **MCU** | a heap-free static pipeline on bare-metal Cortex-M or RISC-V | no allocator linked, proven panic-free, KB-scale footprint budget. `g2g-mcu` supplies the peripheral elements. See [Embedded](#embedded-heap-free-pipelines-on-a-bare-metal-mcu). |
| **RTOS** | the same static pipeline under an RTOS task | bit-exact under bare-metal, Embassy, FreeRTOS, and Zephyr. `embassy-sync` stack channels via the `embassy` / `embassy-link` features. |
| **CPU** | the full media and protocol stack | Tokio, multi-thread on servers or current-thread on edge. |
| **GPU** | zero-copy hardware pipelines | frames stay in Vulkan / CUDA / wgpu / DMABUF domains: Vulkan Video decode to `wgpu::Texture`, NVDEC / NVENC, a CUDA to wgpu bridge. Embeds in an app's own wgpu device (`GpuContext::from_wgpu`, packaged for Bevy as `bevy-g2g`). |
| **WASM** | the same graph in the browser | `wasm32`, single-threaded, optionally in a Worker presenting to an `OffscreenCanvas`. WebCodecs H.264 / H.265 decode, WebGPU present, in-browser or server-offloaded ML. |

[PORTABILITY.md](PORTABILITY.md) runs one detection-overlay pipeline, with its
processing stages defined once in `overlay_stages()`, on the native runner and
in the browser, and records the evidence for each target: Cortex-M footprint,
Embassy smoke, CPU render, GPU-resident wgpu, in-browser canvas.

OS-, GPU-, and device-coupled elements (camera, display, NVDEC / VA-API /
Vulkan Video, VideoToolbox, MediaCodec, ML device EPs) are **experimental**.
They compile and some have host tests, but their runtime is not a CI promise.
`g2g-inspect` prints `Stability   experimental` on those factories. See
[STABILITY.md](STABILITY.md).

### QNX

QNX is the POSIX microkernel used as the reference platform for ISO 26262 and
IEC 62304 certification. It runs on application processors, so it takes the
`std` path, not the MCU one. The current state is a compile-checked spike with
no QNX SDP (`cargo +nightly -Zbuild-std`): `g2g-core` (the no-alloc subset and
the full `alloc` + `runtime` layer), `g2g-mcu`, and the `g2g-plugins` `no_std`
baseline build for `aarch64-unknown-nto-qnx800` and `x86_64-pc-nto-qnx800`
with no code changes. Every OS element is gated on a specific `target_os`,
never `cfg(unix)`, so the Linux hardware paths stay out of the `nto` build.
The roadmap for `std` transports and a QNX Screen sink is in
[PORTABILITY.md](PORTABILITY.md#spike-qnx-safety-certified-rtos).

## Migrating an existing pipeline?

Many `gst-launch-1.0` lines run unchanged through `g2g-launch`:

```sh
cargo run -p g2g-plugins --bin g2g-launch --features std -- \
  "videotestsrc num-buffers=30 ! videoconvert ! fakesink"
```

Element names mostly match, with aliases where they differ (`avdec_h264` to
`ffmpegdec`, `qtmux` to `mp4mux`, `autovideosink` to `waylandsink` / `kmssink`,
`autovideosrc` to `v4l2src` / `libcamerasrc`). Inline caps filters,
`tee name=t` fan-out, muxer fan-in, `decodebin` / `uridecodebin` / `playbin` /
`fallbacksrc`, and `encodebin` / `transcodebin` all parse. An
`encodebin profile="video/x-matroska:video/x-vp8,width=1280,height=720,bitrate=2000000:audio/x-opus"`
expands into those encoders plus `matroskamux` and splices in the scaler a
pinned geometry needs. A line that does not port gets a hint:

```
$ g2g-launch videotestsrc ! theoraenc ! fakesink
parse error: unknown element: theoraenc
  hint: `theoraenc` has no g2g element: no Theora encoder; use `vpxenc` (VP8/VP9) or `av1enc`
```

- **`g2g-launch -v`** prints each link's negotiated caps and memory domain,
  like `gst-launch -v`. `--dot` dumps a Graphviz graph.
- **`g2g-inspect`** is `gst-inspect-1.0`: list elements, dump one's properties
  and pads, or map a GStreamer name with `g2g-inspect --gst x264enc`.
  `--gst-scan app.c` scans an app's source for element names.
- **`g2g-discover clip.mkv`** is `gst-discoverer-1.0`: container, each stream's
  codec and shape, duration, and metadata, read from the demuxer's stream
  collection without decoding. `--json` for tooling. Local files only, any
  other URI scheme is refused.
- **`g2g-device-monitor`** is `gst-device-monitor-1.0`: cameras, audio
  devices, PipeWire nodes, and `Compute/GPU` devices, each with probed caps and
  the launch fragment that opens it. Filter by class
  (`g2g-device-monitor Video/Source`), `--json` for tooling, `--follow` for
  hotplug. Backends: V4L2 / ALSA / PipeWire / GPU on Linux, Media Foundation +
  WASAPI on Windows, AVFoundation + Core Audio on macOS. Each device id is
  what the element's selection property takes, so a saved launch line reopens
  the same hardware after a replug (`v4l2src device-id=`, since `/dev/videoN`
  is not stable). A V4L2 listing includes the controls and ranges
  `v4l2src extra-controls=` accepts.
- **`g2g-mcp`** (feature `mcp`) is a Model Context Protocol server over stdio,
  so an agent drives pipelines: inspect and validate a launch line, run one,
  keep one running and read its telemetry, bus events, logs and packet samples,
  set a property or splice a transform while it runs, read the records a
  `metasink` posts, snapshot a frame as PNG, cut a clip out of a file, and load
  the README's sample pipelines as prompts. Captioning a frame with a VLM and
  searching an embedding index by text stay in gst-python-ml's `pyml-mcp`, which
  reads the same files. `claude mcp add g2g -- target/release/g2g-mcp`.
- **Incremental migration.** `g2g-bridge` embeds a g2g sub-graph in a
  GStreamer pipeline. `gstwrap` hosts an un-ported GStreamer element in a g2g
  graph.

Full guide, with the equivalence cookbook and application / element porting:
[PORTING.md](PORTING.md). Writing an element from scratch:
[AUTHORING.md](AUTHORING.md).

## Scripting: config files and Rhai

A launch string is the one-liner. Three more surfaces build on the same
registry and negotiation, so any element, caps, or policy works in each:

- **Declarative graphs (JSON / YAML), `--graph`.** `nodes` and `edges`, a
  `{ id, caps }` capsfilter shorthand, and a top-level `pipeline:` escape
  hatch. Roles follow link degree (source, sink, muxer, auto-tee), and property
  values are typed by the target element as in a launch string.

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

- **Rhai builder scripts, `--script`.** A document is a fixed graph. A script
  computes one, with loops, parameters, and conditionals, through a builder
  API (`add` / `caps` / `set` / `link` / `link_leaky`) that emits the same
  graph model. Rhai is pure Rust, so this runs on the same wasm and embedded
  targets as the core (`--features script-rhai`).

  ```rhai
  // Fan N cameras into one funnel, sized at runtime.
  add("funnel", "mix");
  for i in 0..num_cams { let id = "cam" + i; add("rtspsrc", id); set(id, "location", cams[i]); link(id, "mix"); }
  add("autovideosink", "screen"); link("mix", "screen");
  ```

- **`scriptelement`: per-frame logic in Rhai.** A raw-video transform whose
  `process(frame)` runs on every frame, the pure-Rust cousin of `pyelement`.
  `frame` is a zero-copy handle to the live buffer.

  ```
  g2g-launch videotestsrc ! scriptelement script="fn process(f){ f.invert(); }" ! autovideosink
  ```
  ```rhai
  fn process(frame) {
      // frame.width / .height / .format / .pts / .sequence / .len
      frame[3] = 128;          // per-pixel edit in place (interpreted)
      frame.invert();          // whole-frame native ops: fill(v) / invert() / apply_lut(lut)
  }
  ```

  The script is the control plane and native code is the data plane, as in
  NumPy. A per-pixel Rhai loop over an HD frame takes seconds, so use the
  script for logic, metadata, and small regions. Whole-frame work goes through
  the native ops (`invert()` is about 1 ms per frame), and a per-value
  transform (brightness, gamma, threshold) is a 256-entry `apply_lut(lut)`.
  Heavy per-pixel math needs a compiled element.

- **`scriptrouter`: script-decided routing to N outputs.** A 1-to-N demux
  whose `route(frame)` returns the output port for each buffer: an index, a
  negative number to drop it, or an array like `[0, 1]` to multicast one buffer
  to several ports. With an `appsink channel=...` on each branch, every channel
  is pulled live from Python, C, or Rust like a GStreamer `appsink`. It is
  media-agnostic. `route` reads `frame.pts` / `.sequence` / `.keyframe` / `.len`
  and can peek bytes with `frame[i]`.

  ```
  # Split an audio stream to two consumers by parity; pull each from your app.
  g2g-launch whepsrc uri=... ! opusdec ! audioconvert ! \
    scriptrouter name=r script="fn route(f){ f.sequence % 2 }" \
    r.0 ! appsink channel=even   r.1 ! appsink channel=odd
  ```
  ```python
  even, odd = g2g.AppSink("even"), g2g.AppSink("odd")   # pull() each, feed anywhere
  ```

  End-to-end demo with two pull channels drained live:
  `cargo run -p g2g-plugins --features script-rhai --example scriptrouter_appsink_egress`

## Embedded: heap-free pipelines on a bare-metal MCU

The MCU build is the same graph with a hard guarantee: `alloc` is an optional
feature and the default build links no allocator. That fits safety-critical,
no-heap targets (MISRA, certification processes).

- **Static element model.** A heap-free pipeline is a compile-time-static
  graph of concrete typed elements (`g2g_core::staticelem`: `StaticSource` /
  `StaticTransform` / `StaticSink` with `async fn` in trait, const-arity
  runners, a `Chain` combinator). Every stage's future is unboxed, with no
  `dyn`, no `Box`, and no allocation. Buffers are lent zero-copy from a
  const-generic `StaticLendRing` sized at compile time. The dynamic runner
  keeps the same steady-state contract on the host: `run_graph` processes 100k
  frames without a per-frame heap allocation (counting-allocator test).
- **CI-checked guarantees.** The linked archive carries zero allocator
  symbols and zero panic symbols (`tools/noalloc-check.sh`). A gc-sectioned
  ELF is measured for ROM, static RAM, and worst-case stack against a budget
  (`tools/footprint-report.sh`). The pipeline then runs on emulated Cortex-M
  (`tools/qemu-check.sh`) with a per-frame timing and jitter report under
  deterministic QEMU `-icount` (`tools/timing-report.sh`). App code on this
  surface needs no `unsafe`.
- **One graph, four executors.** The same static pipeline runs bit-exact under
  a bare poll loop, Embassy, FreeRTOS (C-ABI staticlib), and Zephyr (a module
  the app lists in its west manifest).
- **C integration in both directions.** A C or RTOS app links the pipeline as
  a static library and calls in. Or existing C drivers are the peripheral:
  `g2g-mcu::cffi`'s `CFrameGrabber` / `CPacketSender` wrap C capture and send
  function pointers, and `step_source_sink` returns to the superloop after
  each frame. Proven from a real C caller in `examples/g2g-cffi`, heap-free
  and panic-free.
- **`g2g-mcu` peripheral elements.** Written against `embedded-hal` traits
  rather than chip registers, so the driver logic is host-tested with mock
  peripherals and a board port is the vendor HAL's trait impls:
  `SpiDisplaySink` (ST7789 / ILI9341, whole-frame or banded for panels too
  large to ring-buffer), `GrabberSrc` (DCMI/CSI camera), `PcmSink` (I2S/SAI),
  fixed-point G.711 and IMA ADPCM codecs (bit-exact vs ffmpeg), the
  `HwJpegDec` / `HwH264Enc` hardware codec seams, `YuyvToI420`, and `RtpSink`.
- **Interrupt/DMA capture.** `SpscFrameRing` is a lock-free, heap-free
  single-producer/single-consumer FIFO that a DMA-completion ISR fills while
  the pipeline drains it (`SpscCaptureSrc`, sleeping on `wfi` between frames).
  A full ring drops and counts, never stalls the interrupt. Atomic load/store
  only, so it works on cores without CAS (`thumbv6m`).
- **Fault recovery.** `g2g_core::supervise` turns a peripheral fault into a
  bounded action: a `FaultPolicy` picks retry, skip, reset, or escalate, a
  `Recover` seam re-initializes the stage, and a `Watchdog` is petted only on
  real forward progress, so an escalated pipeline lets the hardware watchdog
  reset the chip. A `SupervisorReport` accounts every fault for the safety
  case.
- **RTP ingress and jitter buffer.** `RtpSrc` parses RTP with the
  bounds-checked header parser shared with the std depayloader, a heap-free
  `JitterBuffer<N, BYTES>` reorders by sequence number and counts reorder /
  duplicate / late / loss, and `G711Dec` decodes.
- **I2C sensors and UART.** `Sht3xSrc` is an SHT3x temperature/humidity
  driver over `embedded-hal` I2C (single-shot command, CRC-8 checked against
  the datasheet vector, fixed-point conversion). `UartSink` / `UartSrc` carry
  byte streams over serial.
- **Safety case.** [`docs/safety/`](docs/safety/) holds a requirements
  traceability matrix (15 requirements, each linked to the proof script, test,
  or CI job that verifies it) and a safety manual (conditions of use,
  assumptions, the `unsafe` inventory). `tools/traceability-check.sh` fails CI
  if cited evidence goes missing. `tools/qualification-kit.sh` runs the whole
  proof set into one report. Pre-1.0 and emulated, so this is a down payment
  on a functional-safety case, not a certificate.
- **ARM and RISC-V.** The no-alloc core and `g2g-mcu` build unchanged for
  `riscv32imafc` (ESP32-P4 class). The symbol proofs and the footprint report
  run on both `thumbv7em` and RISC-V.
- **Reference deterministic-audio graph.** `capture -> convert -> resample ->
  mix -> encode -> RTP` as one static heap-free pipeline, fully fixed-point,
  so its RTP wire bytes are bit-exact on every target. Pinned by a host test
  against an independent float reference and re-verified on all four
  executors.

The capture, fault-recovery, RTP ingress, and sensor pipelines each run on
emulated Cortex-M in CI, checked bit-exact against a synchronous or in-order
reference.

**`g2g-mcugen`, the host graph compiler.** A declarative graph document
compiles to the monomorphized static pipeline, with every ring sized from the
graph's frame geometry and the total ring memory reported. It covers an audio
catalog and a video / display one:

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
disagree, a display fed the wrong pixel format) is rejected before any Rust is
emitted. The generated pipeline reproduces the hand-written reference's wire
output byte-for-byte, checked in CI for both catalogs by
`tools/mcugen-check.sh`.

## The four pillars

1. **Async execution.** Every element is a cooperative `Future`. The
   framework is runtime-agnostic: Tokio on servers, Embassy on RTOS,
   `wasm-bindgen-futures` in the browser.
2. **Hardware-first, zero-copy.** Buffers live in DMABUF / Vulkan / CUDA /
   D3D11 / WebGPU memory domains. Negotiation settles a zero-copy path per
   link where one exists and auto-plugs a converter where none does, so every
   remaining copy is explicit (`g2g-launch -v` shows each link's domain).
3. **`no_std`, `alloc`-optional, sans-IO core.** The same pipeline shape runs
   on a bare-metal Cortex-M with no heap, an RTOS (Embassy / FreeRTOS /
   Zephyr), a multi-threaded server, a GPU-resident pipeline, or `wasm32`
   (see [Portability](#portability-one-pipeline-five-targets)).
4. **First-class ML.** Tensor allocation, reshaping, and pipeline batching are
   part of graph orchestration.

## Workspace

| Crate | Role | Profile |
| :--- | :--- | :--- |
| `g2g-core` | Traits, `Frame` / `PipelinePacket`, caps algebra, clock, runner, static element model. | `no_std`, `alloc` optional |
| `g2g-mcu` | Heap-free MCU peripheral elements (SPI display, camera / PCM capture, I2C sensor, UART, G.711 / ADPCM codecs, hardware JPEG / H.264 seams, RTP egress + ingress, jitter buffer, fault-recovery watchdog) over `embedded-hal` and C-callback seams. | `no_std`, no alloc |
| `g2g-mcugen` | Host graph compiler: a declarative MCU graph (YAML / JSON) to a monomorphized heap-free static pipeline. | `std` (host tool) |
| `g2g-plugin` | SDK for dynamically loadable plugins: same-toolchain (`declare_plugin!` + ABI tag) and the frozen C ABI v2 for cross-toolchain Rust and plain-C plugins. | `no_std + alloc` |
| `g2g-plugins` | Sources, sinks, and transforms (RTSP, RTP, HTTP / HLS / DASH / RTMP, V4L2 / PipeWire / MF capture, ffmpeg, VAAPI, MF, VideoToolbox, MediaCodec, Wayland, KMS, WASAPI, ALSA / PulseAudio / PipeWire audio, compositor, Embassy, web), container mux / demux, codec parsers and encoders, the tag system, and the `gst-launch` text DSL. | mixed |
| `g2g-ml` | ORT, Burn, `WgpuPreprocess`, `TensorPostprocess`, multi-stream tensor batcher. | `std` |
| `g2g-bridge` | GStreamer C-FFI bridge. | `std` |
| `g2g-python` | Hosts gst-python-ml elements in-process (embedded CPython via pyo3). | `std` |
| `g2g-capi` | C ABI (cdylib / staticlib + `g2g.h`): pipelines, bus, appsrc / appsink from any language. | `std` |
| `g2g-pyapi` | Python (pyo3) bindings: pipelines, bus, appsrc / appsink. | `std` |

## Build

Stable Rust, `resolver = "2"`. MSRV 1.92, except `g2g-core`, `g2g-mcu`,
`g2g-mcugen`, and `g2g-plugin`, which build on 1.86 so a vendor-pinned
toolchain can consume the portable core (see `STABILITY.md`).

```sh
cargo check --workspace          # no_std baseline
cargo test  --workspace          # default test suite (no platform features)
cargo clippy --workspace --all-targets
```

### Feature-gated elements

| Element | Feature | Platform / system dep |
| :--- | :--- | :--- |
| `RtspSrc` (video) / `RtspSrcN` (video + audio, plus the ONVIF analytics metadata track under `onvif-metadata=true`) | `rtsp` | retina |
| `OnvifSrc` (camera discovery + stream-URI resolution) / `OnvifMetadataParse` / `OnvifMetadataCombiner` | `onvif` | reqwest + roxmltree |
| `H264Parse` | (default) | none |
| `FfmpegH264Dec` (sw / `NvdecCuvid` / `NvdecCuda` / `Vaapi`) | `ffmpeg` | Linux + libavcodec |
| `VaapiH264Dec` | `vaapi` | Linux + libva + GBM |
| `MfDecode` / `MfEncode` / `MfAacEncode` / `MfAacDecode` | `mf-decode`, `mf-encode`, `mf-aac` | Windows + Media Foundation |
| `VtDecode` / `VtEncode` (H.264 / H.265, zero-copy `CVPixelBuffer` output via `cv-output`) | `vtdecode`, `vtencode` | macOS + VideoToolbox |
| `MediaCodecDec` (H.264 / H.265, zero-copy GPU output via `with_gpu_output`) | `mediacodec`, `mediacodec-wgpu` | Android + NDK MediaCodec (+ wgpu / Vulkan for GPU output) |
| `WaylandSink` | `wayland-sink` | Linux + Wayland |
| `KmsSink` | `kms-sink` | Linux + libdrm, needs DRM master / tty |
| `D3D11Sink` | `d3d11-sink` | Windows |
| `MetalVideoSink` (zero-copy from `CVPixelBuffer`) | `metal-sink` | macOS + Metal |
| `WgpuPresentSink` (`wgpusink`: owns its Wayland window, presents GPU-resident frames with no upload) | `wgpu-present` | Linux + Wayland + wgpu |
| `NvDec` (native NVDEC H.264 / H.265 / AV1 to CUDA NV12 or 10-bit P010) | `nvdec` | Linux + NVIDIA driver (libnvcuvid) |
| `NvEnc` (native NVENC CUDA NV12 / P010 to H.264 / H.265, incl. HEVC Main 10) | `nvenc` | Linux + NVIDIA driver (libnvidia-encode) |
| `CudaDownload` (CUDA to System), `CudaUpload` (System to CUDA) | `cuda` | Linux + NVIDIA driver (libcuda) |
| `CudaGlSink` (CUDA-GL present), `CudaKmsSink` (CUDA-GL on KMS) | `cuda-gl`, `cuda-kms` | Linux + NVIDIA + EGL + GL (+ libdrm for KMS) |
| `CudaToWgpu` / `WgpuToCuda` (zero-copy bridge) | `cuda-wgpu` | Linux + NVIDIA + Vulkan |
| `UdpSink` + RTP packetizer, or raw datagrams (`multiudpsink` `clients=`) | `udp-egress` | none |
| `UdpSrc` (RTP ingest + jitter buffer + RTCP / NACK, or raw MPEG-TS datagrams) | `udp-ingress` | none |
| `SrtpEnc` / `SrtpDec` (RFC 3711 / RFC 7714 SRTP and SRTCP, per-SSRC receive contexts) | `srtp` | none |
| `DtlsSrtpEnc` / `DtlsSrtpDec` (DTLS-SRTP handshake over the media socket keys SRTP) | `dtls-srtp` | none |
| `TcpServerSrc` / `TcpClientSrc` / `TcpServerSink` / `TcpClientSink` | `tcp` | none |
| `ShmSink` / `ShmSrc` (GStreamer's `shm` protocol: shared-memory frames + unix control socket) | `shm` | Linux |
| `RtmpSrc` (RTMP publisher ingest) | `rtmp` | none |
| `WebRtcSink` (WHIP egress, H.264 + Opus) / `WebRtcWhepSrc` (WHEP ingest, H.264): ICE / DTLS / SRTP, trickle ICE + ICE restart, NACK / RTX | `webrtc` | str0m (rust-crypto) + reqwest |
| `WebRtcDataSrc` / `WebRtcDataSink` (P2P data channels on SCTP) | `webrtc` | str0m |
| `MoqtSink` (MoQ Transport draft-16/18 publisher: fMP4 to groups / objects over WebTransport, subgroup streams or datagrams) | `moqt` | web-transport-quinn |
| `MoqtSrc` (MoQ Transport draft-16/18 subscriber: catalog read, stream + datagram reassembly to fMP4) | `moqt` | web-transport-quinn |
| `LiveKitSink` (publish into a LiveKit room: JWT + protobuf signalling) | `webrtc-livekit` | + tokio-tungstenite |
| `HttpSrc` (HTTP(S) byte-stream source) | `http-src` | reqwest |
| `HlsSrc` (TS + fMP4 / CMAF, live, LL-HLS parts + blocking reload, AES-128 / SAMPLE-AES) | `hls` | reqwest + aes |
| `DashSrc` (`SegmentTemplate` / `SegmentTimeline`, live, CMAF chunked low latency) | `dash` | reqwest + roxmltree |
| `V4l2Src` | `v4l2` | Linux + V4L2 (`/dev/videoN`) |
| `WasapiSink` / `WasapiSrc` | `wasapi-sink`, `wasapi-src` | Windows |
| `AlsaSink` | `alsa-sink` | Linux + libasound |
| `PulseSink` | `pulse-sink` | Linux + libpulse |
| `PipeWireSink` / `PipeWireSrc` (audio) | `pipewire` | Linux + libpipewire |
| `PipeWireVideoSrc` (video capture, `io-mode=mmap` or `dmabuf`) | `pipewire` | Linux + libpipewire |
| `PipeWireVideoSrc portal=true` (screen capture via xdg-desktop-portal) | `portal` | Linux + libpipewire + a desktop portal |
| `MfVideoSrc` (camera) | `mf-video-src` | Windows + Media Foundation |
| `Av1Enc` (pure-Rust `rav1e`) | `av1-encode` | none |
| `VpxEnc` (VP8 / VP9 via libvpx) | `vpx` | libvpx |
| `MjpegDec` / `MjpegEnc` (pure Rust) | `mjpeg`, `mjpeg-encode` | none |
| `PngDec` / `PngEnc` (pure Rust) | `png` | none |
| `WebPDec` (lossy + lossless, pure Rust) | `webp` | none |
| `AnalyticsOverlay` (CPU) / `VelloAnalyticsOverlay` (GPU) (detection boxes, segmentation masks, ROIs) / `WgpuSink` | `analytics`, `vello-overlay`, `wgpu-sink` | wgpu (GPU variants) |
| `MetaSink` / `MetaReplay` (one JSON line per frame: detections, blobs, text; replay onto frames) / `AnalyticsAlert` (rules, cooldown, `alert` blob, webhook) / `AlertRecorder` (a clip around each alert) | `analytics-json` | none |
| `MqttSink` (the same record per frame, published to an MQTT topic) / `MqttSrc` (each message on a topic filter as a text frame) | `mqtt` | none |
| `EmbeddingSink` (sqlite index of embedding vectors, searchable from `pyml-mcp`) | `embedding-index` | none (sqlite bundled) |
| `VelloTextOverlay` (subtitle cues drawn on the GPU, `WgpuTexture` out) | `vello-text-overlay` | wgpu |
| `OrtInference` (+ CUDA / DirectML EPs) | `ort`, `cuda`, `directml` (in `g2g-ml`) | onnxruntime |
| `BurnInference` (linear layer, or an ONNX topology imported by `burn-onnx` codegen) | `burn` (in `g2g-ml`) | wgpu (Vulkan / Metal / DX12) |
| `WgpuPreprocess` (NV12 / YUYV system bytes, a dma-buf, or a GPU texture in, NCHW tensor out) | `wgpu`, `dmabuf-wgpu`, `mediacodec-wgpu` (in `g2g-ml`) | wgpu (Vulkan for the dma-buf import) |
| Embassy / RTOS pool + clock | `embassy`, `embassy-link` | none |
| Browser elements | `web`, `web-codecs` | `wasm32-unknown-unknown` |

### In the default `no_std + alloc` build

`g2g-inspect` lists every registered element. By group:

| Group | Elements |
| :--- | :--- |
| Container demuxers and muxers | `qtdemux` / `mp4mux`, `tsdemux` / `mpegtsmux`, `matroskademux` / `matroskamux`, `flvdemux` / `flvmux`, `oggdemux` / `oggmux`, `avidemux` / `avimux`, `fmp4demux`, `mpegpsdemux`, `multipartdemux` / `multipartmux`, `y4mdec` / `y4menc`, `aiffparse` / `aiffmux`, `auparse` / `avmux_au` |
| Bitstream parsers | `h264parse`, `h265parse`, `aacparse`, `mpegaudioparse` + `id3demux` / `apedemux`, `ac3parse`, `opusparse`, `vp8parse`, `vp9parse`, `av1parse`, `jpegparse`, `pngparse` |
| Headerless framers | `rawvideoparse` / `rawaudioparse`: a `.yuv` / `.pcm` dump cut into buffers from declared properties |
| Audio codecs | G.711 `mulawenc` / `mulawdec`, `alawenc` / `alawdec`, IMA ADPCM `adpcmenc` / `adpcmdec` |
| Video transforms | `videoscale`, `videorate`, `imagefreeze`, `videocrop`, `videoflip`, `videobalance`, `videobox`, `colorspace`, `alpha`, `gamma`, `deinterlace`, `timeoverlay`, `aspectratiocrop`, `gaussianblur`, `videomedian`, `smooth`, `coloreffects`, `chromahold`, `zebrastripe`, `videodiff`, `solarize`, `chromium`, `dilate`, `dodge`, `exclusion`, `burn` |
| Audio transforms | `audioconvert` (`dithering` and `noise-shaping` on bit-depth reduction), `audioresample` (windowed sinc, `quality` 0 to 10), `audiorate`, `audiomixer`, `interleave`, `deinterleave`, `scaletempo`, `volume`, `audiopanorama`, `audioamplify`, `audioecho`, `audiodynamic`, `audiowsinclimit`, `audiocheblimit`, `audiochannelmix`, `audiomixmatrix`, `stereo`, `audiofirfilter`, `audioiirfilter`, `removesilence`, `audiobuffersplit`, `speed`, `audioreverse`, `equalizer-3bands` |
| Audio analysis | `level`, `ebur128`, `cutter`, `spectrum` |
| Telemetry | `klvdecode` (MISB ST 0601 / STANAG 4609) |
| Subtitles and captions | bitmap decoders `vobsubdec` (alias `dvdsubdec`), `dvbsubdec`, `pgsdec` with `subpictureoverlay` to blend their cues onto video, `teletextdec`, `subparse`, `srtenc`, `webvttenc`, `ccextract` / `cccombiner`, `ccconverter` (between cc_data, CDP, S334-1A, and raw CEA-608) |
| Flow control and debug | `concat`, `input-selector`, `output-selector`, `valve`, `fakesrc`, `fdsrc`, `fdsink`, `watchdog`, `capssetter`, `taginject`, `rndbuffersize`, `errorignore`, `breakmydata`, `chopmydata`, `checksumsink`, `fakevideosink`, `fakeaudiosink`, `progressreport` |
| Composition and DSL | `compositor`, the tag system, `parse_launch` / `gst-inspect` |

### Added by the `std` build

| Element | What it does |
| :--- | :--- |
| `clockoverlay`, `fpsdisplaysink` | |
| `multifilesink` / `multifilesrc` | image sequences (`imagesequencesrc` when it stamps a framerate) |
| `splitfilesrc` | the parts of a cut recording read as one byte stream |
| `dataurisrc` | a `data:` URI's payload |
| `vobsubsrc` | a DVD subtitle `.idx` / `.sub` sidecar pair |
| `splitmuxsink` | segmented recording, `muxer=mp4\|matroska\|mpegts` |
| `togglerecord` | starts and stops several streams together on the main stream's keyframes: one element per stream joined by `group=`, `main=true` on the one that decides |
| `hlssink` | HLS packaging, segment files plus an `.m3u8` playlist, fed by `mpegtsmux` or `mp4mux` |
| `fallbackswitch` | forwards the highest-priority input still delivering, input 0 being primary |
| `fallbacksrc uri=X` | wraps `fallbackswitch` around a URI's decode chain, rebuilds the source when it dies, and reports each restart on the bus. On its own it carries every stream kind in the container through a switch and a sink of its own |
| `livesync` | keeps a stalling live input's output going by repeating the last video frame or filling audio silence |

## Sample pipelines

The graph API is `run_source_transform_sink` / `run_linear_chain` /
`run_source_fanout` / `run_muxer_sink` over typed elements. The examples are
condensed. Full versions are in `g2g-plugins/tests/`.

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
[design/decode.md](design/decode.md).

### Native NVDEC → NVENC transcode, GPU-resident, with domain auto-plug

`NvDec` / `NvEnc` drive NVCUVID / NVENC directly, without libavcodec, and the
decoded frames stay in `MemoryDomain::Cuda` into the encoder. Where no shared
domain exists (a CPU-side NV12 source feeding the CUDA-only `NvEnc`),
`auto_plug_cuda_converters` splices in a `CudaUpload`.

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
keeps frames on the GPU or downloads to System, driven by downstream demand.
It decodes H.264 / H.265 / AV1, emits P010 for 10-bit streams, reconfigures
in place on a mid-stream resolution change, and takes `max-display-delay`
(latency vs decode/display pipelining) and `cuda-device-id`. `NvEnc` encodes
P010 as HEVC Main 10 and takes `gop-size` / `repeat-sequence-header` for
periodic IDRs carrying their own SPS/PPS.

### RTSP → decode → KMS (tty / no compositor)

```rust
let src  = RtspSrc::new(url);
let dec  = FfmpegH264Dec::new().with_output_format(OutputFormat::Nv12);
let sink = KmsSink::new().with_device("/dev/dri/card0");

run_source_transform_sink(src, dec, sink, &clock, LatencyProfile::Live).await?;
```

Features: `rtsp ffmpeg kms-sink`. Run from a tty after stopping the display
manager, since the KMS sink needs DRM master.

### RTSP → decode → ML preprocess → ORT inference → postprocess

```rust
let src           = RtspSrc::new(url);
let mut dec       = FfmpegH264Dec::new().with_output_format(OutputFormat::Nv12);
let mut preproc   = WgpuPreprocess::new(w, h);             // NV12 -> f32 NCHW on GPU
let mut inference = OrtInference::from_memory_with_cuda(model_bytes)?;
let mut post      = TensorPostprocess::topk_classification(5);

run_linear_chain(src, vec![&mut dec, &mut preproc, &mut inference, &mut post],
                 FakeSink::new(), &clock, LatencyProfile::Live).await?;
```

Features: `rtsp ffmpeg` (plugins) + `wgpu cuda` (g2g-ml). The CUDA execution
provider falls back to CPU when no CUDA runtime is present.

### Android: MediaCodec decode → GPU → ML preprocess (zero-copy)

```rust
// Decode on the NDK MediaCodec and keep the frame on the GPU as an RGBA wgpu
// texture (no CPU NV12 pack); WgpuPreprocess samples it straight into a tensor.
let dec     = MediaCodecDec::h264().with_gpu_output();   // MemoryDomain::WgpuTexture (RGBA)
let preproc = WgpuPreprocess::new();                     // samples the texture -> f32 NCHW
// dec -> preproc -> OrtInference / BurnInference, all on the GPU
```

Features: `mediacodec-wgpu` (plugins) + `mediacodec-wgpu` (g2g-ml). Android
only, validated on a Pixel 10a. The decoded `AHardwareBuffer` is imported into
Vulkan and converted to RGBA through an immutable `VkSamplerYcbcrConversion`
compute pass (a conversion wgpu's bind-group API cannot express), then handed
downstream as a `wgpu::Texture`. The frame never touches the CPU.

### File → H.264 parse → fMP4 record

```rust
let graph = parse_launch(
    &default_registry(),
    "filesrc location=in.h264 ! h264parse ! mp4mux ! filesink location=out.mp4",
)?;
run_graph(graph, &clock, LatencyProfile::Live).await?;
```

### MPEG-TS file → demux → H.264 parse → decode → Wayland

The container demuxers (`tsdemux`, `matroskademux`, `flvdemux`, `oggdemux`,
`fmp4demux`, `mpegpsdemux`) accept a `Caps::ByteStream` and split out
elementary streams. `mpegpsdemux` reads `.mpg` / `.vob` program streams,
including their DVD subpicture tracks. Every `playbin` video branch carries a
`deinterlace mode=auto` (yadif): the decoder marks interlaced streams in its
output caps (`interlace-mode=interleaved`) and the filter weaves only those,
so progressive content passes through untouched.

```rust
let src       = FileSrc::new("clip.ts", Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs });
let mut demux = TsDemux::new().with_stream(TsStream::H264);   // PAT/PMT/PES -> Annex-B
let mut parse = H264Parse::new();
let mut dec   = FfmpegH264Dec::new().with_output_format(OutputFormat::Nv12);
let sink      = WaylandSink::new();

run_linear_chain(src, vec![&mut demux, &mut parse, &mut dec], sink,
                 &clock, LatencyProfile::Live).await?;
```

Features: `ffmpeg wayland-sink`.

### STANAG 4609 (drone / ISR): KLV telemetry alongside the video

`tsdemux stream=klv` splits the MISB metadata stream out of the multiplex
(private PES with the `KLVA` registration, or metadata-in-PES 0x15), and
`klvdecode` parses each ST 0601 UAS Datalink Local Set into a timed
`key=value` text line for `textoverlay` or an app sink. The tag table covers
the telemetry core, identity strings, target geometry, and the nested ST 0102
security local set, validated against the published MISMMS reference packet
with klvdata as the oracle. The mux direction takes `Caps::Klv` packets
(`UasDatalink::encode`) on a `mpegtsmux` input, and `rtpklv` carries KLV over
RTP (RFC 6597). Both directions are ffmpeg-validated bit-exact and checked
against a real UAS capture (the public "Day Flight" sample from
samples.ffmpeg.org, run locally by pointing `G2G_STANAG_SAMPLE` at it for the
`klv_stanag_sample` test).

The rest of the ISR stack sits on the same codec: `vmti` for ST 0903
moving-target reports with their nested mask / ontology / tracker / chip sets
(`vmti_from_analytics` turns an in-pipeline detector's output into VTargets),
ST 1204 MIIS identifiers, `misptimeinsert` / `misptimeextract` for ST 0604
timestamps in H.264 / H.265 SEI, `st2022fec` for SMPTE 2022-1 loss recovery
on a contribution link, and `cotsink` to put a drone track on a TAK / ATAK
network as Cursor-on-Target events, optionally with the ST 0805.1 sensor point
of interest. SRT carries it encrypted (`passphrase=` on `srtsink` / `srtsrc`).

```rust
let src   = FileSrc::new("uav.ts", Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs });
let demux = TsDemux::new().with_stream(TsStream::Klv);
let dec   = KlvDecode::new();   // -> "ts=.. lat=.. lon=.. alt=.. heading=.." lines
```

### Adaptive streaming: HLS / DASH → decode → display

```rust
let src       = HlsSrc::new("https://example.com/master.m3u8");  // or DashSrc::new(mpd_url)
let mut demux = TsDemux::new().with_stream(TsStream::H264);
let mut parse = H264Parse::new();
let mut dec   = FfmpegH264Dec::new().with_output_format(OutputFormat::Nv12);
let sink      = WaylandSink::new();

run_linear_chain(src, vec![&mut demux, &mut parse, &mut dec], sink,
                 &clock, LatencyProfile::Live).await?;
```

Features: `hls ffmpeg wayland-sink` (`dash` for the DASH front end). `HlsSrc`
follows live playlist reloads and decrypts AES-128 / SAMPLE-AES segments. On a
low-latency playlist (`#EXT-X-PART` plus `CAN-BLOCK-RELOAD`) it blocks the
reload on the next partial segment and emits each part as it is published
(`low-latency=false` forces whole segments). `DashSrc` handles
`SegmentTemplate` / `SegmentTimeline` and live MPDs, and with
`low-latency=true` consumes a CMAF segment chunk by chunk. Both prebuffer by
duration (`prebuffer-ms`) and post `Buffering` bus levels while they fill,
like `HttpSrc`'s byte window (`prebuffer-bytes`).

### `gst-launch` text pipeline

`parse_launch` builds a runnable `Graph` from a GStreamer-style string against
the `default_registry`, including caps filters, `tee` branching, and muxer
fan-in. `Registry::inspect(name)` is the `gst-inspect` analog.

```rust
let graph = parse_launch(
    &default_registry(),
    "videotestsrc num-buffers=90 pattern=ball ! video/x-raw,format=nv12 \
     ! videoflip method=rotate-180 ! matroskamux ! filesink location=out.mkv",
)?;
run_graph(graph, &clock, LatencyProfile::Live).await?;
```

Feature-gated capture, decode, and display elements register their launch
factories when their feature is enabled. `autovideosink` / `autoaudiosink`
resolve to whichever sink is built, falling back to `fakesink` so a tutorial
line runs headless.

The ML elements live in `g2g-ml`, so an app opts in after building the
registry: `g2g_ml::register(&mut reg)` (the `launch` feature) adds `ortinfer`,
`wgpupreprocess`, `detectionpostprocess`, and `ortsegment` (instance
segmentation), so
`... ! ortinfer model=yolov8n.onnx ! detectionpostprocess conf-threshold=0.3 ! ...`
parses.

### Camera → encode → RTP egress over UDP

```rust
let src  = VideoTestSrc::new(1920, 1080, 30, 0);         // RGBA test pattern, unbounded
let enc  = MfEncode::new().with_hardware();              // Windows; on Linux use NvEnc / ffmpeg
let sink = UdpSink::new("239.0.0.1:5004".parse()?)
    .with_rtp(96, 0x1234_5678);                          // payload type, SSRC

run_source_transform_sink(src, enc, sink, &clock, LatencyProfile::Live).await?;
```

Features: `udp-egress` plus the platform encoder feature. `UdpSink` answers
receive-side NACK by retransmitting from a bounded send history
(`with_retransmit`).

### RTP ingress over UDP → ffmpeg decode → Wayland

The receive side, with a jitter buffer (reorder and bounded-latency loss
handling) and RTCP feedback (periodic receiver reports, NACK on gaps).

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

As a launch line, placement goes through the flattened pad properties:

```text
videotestsrc ! c.  v4l2src device=/dev/video0 ! videoconvert ! videoscale ! c. \
  compositor name=c width=1280 height=720 sink1-xpos=940 sink1-ypos=460 sink1-zorder=1 ! waylandsink
```

Features: `v4l2 wayland-sink`. Full graph in
[`g2g-plugins/tests/pip_smoke.rs`](g2g-plugins/tests/pip_smoke.rs).
`WgpuCompositor` is the bit-exact GPU sibling and composites `WgpuTexture`
frames in place. `with_timed_output()` (`timed-output=true`) holds the output
rate over a stalled input when the pipeline clock can sleep on a deadline.

### Camera → MoQ Transport → a browser

```text
libcamerasrc width=640 height=480 framerate=30 ! videoconvert ! x264enc ! mp4mux \
  ! moqtsink location=https://127.0.0.1:4443/ namespace=live
```

[`tools/moqt-demo/`](tools/moqt-demo/) runs that end to end: `node
watch-live.mjs` starts a local `moq-relay-ietf`, publishes the camera into it,
and opens a browser that subscribes and plays. The page's MoQT client is the
third-party [MOQtail](https://github.com/moqtail/moqtail) draft-16
implementation, so the browser decodes the bytes with nothing shared from the
Rust side. `node headless/run-moqt-play.mjs` runs the same path in headless
Chromium with assertions on the decoded frames. Features:
`libcamera moqt ffmpeg`.

## Running smoke tests

Most integration tests are `#[ignore]` because they need a live RTSP feed or a
display. The pattern is the same across recipes:

```sh
cargo test -p g2g-plugins \
  --features "<comma-separated feature list>" \
  --test <test_name> -- --ignored --nocapture
```

### A standing RTSP feed

A loopback setup uses [mediamtx](https://github.com/bluenviron/mediamtx) as
the relay and `ffmpeg` as the publisher. The quickest relay is the docker
image on the host network, which listens on 8554/tcp:

```sh
docker run --rm -it --network=host bluenviron/mediamtx
```

Where host networking is unavailable (Windows, macOS), map the port and pin
RTSP to TCP, since UDP needs the real source address and docker's network
stack rewrites it:

```sh
docker run --rm -it -e MTX_RTSPTRANSPORTS=tcp -p 8554:8554 bluenviron/mediamtx
```

A native `mediamtx` binary from the
[releases page](https://github.com/bluenviron/mediamtx/releases) works the
same with no arguments. In a second terminal, push a synthetic H.264 feed
into it:

```sh
ffmpeg -re -f lavfi -i testsrc=size=1280x720:rate=30 \
       -c:v libx264 -pix_fmt yuv420p -preset ultrafast -tune zerolatency -g 30 \
       -f rtsp -rtsp_transport tcp rtsp://localhost:8554/pattern
```

Any RTSP feed works, including a public demo stream or an IP camera on the
LAN.

The multi-track tests need audio in the same stream, so add an AAC track:

```sh
ffmpeg -re -f lavfi -i testsrc=size=1280x720:rate=30 -f lavfi -i sine=frequency=440 \
       -c:v libx264 -pix_fmt yuv420p -preset ultrafast -tune zerolatency -g 30 \
       -c:a aac -ar 48000 -ac 1 \
       -f rtsp -rtsp_transport tcp rtsp://localhost:8554/avpattern
```

```sh
G2G_RTSP_AV_TEST_URL=rtsp://localhost:8554/avpattern \
  cargo test -p g2g-plugins --features "rtsp ffmpeg" \
  --test m1122_rtsp_audio_track -- --ignored --nocapture
```

### Software decode + Wayland

```sh
G2G_RTSP_TEST_URL=rtsp://localhost:8554/pattern \
  cargo test -p g2g-plugins \
  --features "rtsp ffmpeg wayland-sink" \
  --test wayland_smoke -- --ignored --nocapture
```

A window titled "glass2glass" shows the feed.

### Splicing an element into the running pipeline

```sh
cd examples/g2g-mutate-demo && cargo run --release
```

The RTSP feed plays in a window while a `videoflip` is spliced onto the
decoded-video edge every few seconds, held, and removed again. The stream
keeps running with no gap and no restart, and removing the flip first drains
the frames it still holds, so none is lost or reordered. `cargo run --release
-- tee` runs the structural variant: the decoder feeds a `tee` with a window
on each branch, and the splice lands on one branch, named by the sink at its
far end (`insert_before`), while the sibling window plays on untouched.

This is `GraphMutator` on a live graph, the counterpart to a GStreamer pad
block plus relink (see [PORTING.md](PORTING.md) §5.2). A splice point is a
transform position on a 1:1 edge, including tee / demux branches and muxer
inputs. The source and sink ends take a whole replacement element instead
(`replace_source` / `replace_sink`). The demo is a standalone crate outside
the workspace because it needs the `rtsp` + `ffmpeg` + `wayland-sink`
feature set. `G2G_RTSP_URL` picks the feed, `G2G_DEMO_SECONDS` bounds the run,
ctrl-c ends it.

### NVIDIA NVDEC (system memory) + Wayland

```sh
G2G_DECODER=nvdec \
G2G_RTSP_TEST_URL=rtsp://localhost:8554/pattern \
G2G_TARGET_FRAMES=300 \
  cargo test -p g2g-plugins \
  --features "rtsp ffmpeg wayland-sink" \
  --test wayland_smoke -- --ignored --nocapture
```

`G2G_TARGET_FRAMES >= 300` amortizes cuvid startup (libnvcuvid load, CUDA
context, surface pool) so the p50 / p95 latency numbers mean something.
Compare against `G2G_DECODER=software` on the same feed.

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

# An ONNX topology imported into that element by build-time codegen. Standalone
# (workspace-excluded): keeps burn's codegen tree out of the workspace lockfile.
cd examples/g2g-onnx-import && cargo test
```

### UDP egress (loopback, no network)

```sh
cargo test -p g2g-plugins --features udp-egress --test m47_udp_egress -- --nocapture
```

Binds a UDP receiver on localhost, drives the H.264 RTP packetizer, and
asserts the datagrams parse back byte-exactly, with sequence numbers, marker
bit, and FU-A reassembly correct.

### UDP ingress + resilience (loopback, no network)

```sh
cargo test -p g2g-plugins --features "udp-ingress udp-egress" --test udp_loopback -- --nocapture
```

End-to-end over localhost: depayload round-trip, the jitter buffer reordering
out-of-order packets, and NACK-driven recovery. A lossy relay drops chosen
sequences, the receiver NACKs, the sender retransmits, and every access unit
arrives in order.

### Benchmarks against GStreamer

`tools/latency-bench-e2e.sh` measures receive latency on both stacks with the
same code over the same span. A g2g publisher burns the current
`CLOCK_MONOTONIC` into each frame's luma plane (`timestampburn`) before the
encoder and serves it over RTSP. A g2g consumer and a `gst-launch-1.0`
consumer each decode to raw I420 and pipe it into `g2g-latency-reader`, which
subtracts the burned value from its own clock and prints n / p50 / p95 / p99 /
max. No display server is involved. `DECODER=software|nvdec` picks the decode
path, `PARSE=0` drops `h264parse` from both lines, and
`NETEM="delay 20ms loss 1%"` reruns everything inside an unprivileged network
namespace with that qdisc on `lo`. The older `tools/latency-bench.sh` covers
arrival to present, which the two stacks cannot report over one span.

`tools/throughput-bench.sh` races the two offline on a cached ffmpeg fixture:
1800 frames of 1080p H.264 decoded to I420 with libavcodec on both sides,
reporting wall, CPU seconds, peak RSS, and fps from `/usr/bin/time -v`.

## Android on-device testing

The Android elements (`mediacodec` decode/encode, `mediacodec-wgpu` zero-copy
decode to GPU, `aaudio` audio, `camera2` capture) are cross-compiled in CI and
validated on a real device. Each has an on-device probe and a smoke script
that builds that test binary, pushes it to `/data/local/tmp` with `adb`, runs
it, and checks the libtest summary.

**Prerequisites:**

- `adb` on `PATH` with a phone attached and USB debugging authorised
  (`adb devices` lists it as `device`).
- `cargo-ndk` (`cargo install cargo-ndk`).
- The rustup target: `rustup target add aarch64-linux-android`.
- The Android NDK, with `ANDROID_NDK_HOME` pointing at it, e.g.
  `export ANDROID_NDK_HOME=$HOME/android-ndk-r27c`.

**Run a probe:**

```sh
export ANDROID_NDK_HOME=$HOME/android-ndk-r27c

tools/android-mediacodec-smoke.sh        # decode  (H.264 + HEVC -> NV12)
tools/android-mediacodec-enc-smoke.sh    # encode  (NV12 -> Annex-B H.264)
tools/android-aaudio-smoke.sh            # audio   (render; mic capture best-effort)
tools/android-camera2-smoke.sh           # camera  (caps + FFI; capture best-effort)
tools/android-surface-present-smoke.sh   # decode -> GPU -> present (headless ImageReader window)
tools/android-apk-present-smoke.sh       # decode -> GPU -> on-screen present (NativeActivity APK)
tools/android-nnapi-smoke.sh             # ML inference (NNAPI + XNNPACK ORT EPs)
tools/android-nnapi-conv-smoke.sh        # ML on the Edge TPU (int8 conv, NNAPI placement + DarwiNN logcat)
tools/android-camera-tpu-smoke.sh        # live camera -> quantize -> Edge TPU inference, end to end
```

Each takes an optional ABI argument (default `arm64-v8a`, also `x86_64` and
`armeabi-v7a`). To drive an element by hand, build the test the same way and
push the binary yourself:

```sh
cargo ndk --platform 26 -t arm64-v8a build --release \
  -p g2g-plugins --features camera2 --test android_camera2_probe
adb push target/aarch64-linux-android/release/deps/android_camera2_probe-<hash> /data/local/tmp/probe
adb shell /data/local/tmp/probe --nocapture --test-threads=1
```

`--platform` is 24 for `AImageReader`, 26 for `AHardwareBuffer` / AAudio.

The APK harness (`examples/g2g-android-present`) is the one probe that is a
real app: a `NativeActivity` whose window `WgpuSink` presents to. It is built
without gradle, from aapt2 + zipalign + apksigner in the SDK build-tools plus
`keytool` for the one-time debug keystore. Point `ANDROID_SDK_ROOT` at an SDK
with `build-tools/` and `platforms/`. The device must be unlocked while it
runs. Its manifest declares `RECORD_AUDIO` / `CAMERA`, granted by the script
via `pm grant`, so permission-gated capture can run in-app.

**Permission caveats.** A bare `/data/local/tmp` binary has no app manifest,
so mic capture (`RECORD_AUDIO`) and camera capture (`CAMERA`) cannot run
there. Those probes assert what they can check headlessly (device open, caps,
FFI linkage, encode/render) and report the denial rather than failing. Full
capture and an on-screen `SurfaceView` present need the APK harness. If `adb`
reports "insufficient permissions", run `adb kill-server && adb start-server`
and re-accept the prompt on the phone.

## Host validation

The GPU and bench suites need hardware CI cannot reach, on hosts that are
powered off most of the day. There is no self-hosted runner. A systemd user
timer runs `tools/host-validation.sh`, which fast-forwards the host's checkout
to `master`, runs one suite, and posts a commit status on the SHA it tested.
Nothing on GitHub can start a process on the host, and an offline host queues
nothing.

Two suites, one per host:

| Suite | Host | Steps |
| :--- | :--- | :--- |
| `desktop-gpu` | RTX 3060 desktop | the `vulkan-video` test files, the CUDA / NVDEC / `cuda-wgpu` tests, `g2g-ml`'s `cuda_wgpu_e2e`, and the A/V lip-sync soak |
| `bench` | headless bench box | `tools/throughput-bench.sh`, with the g2g and GStreamer fps in the status description |

A step whose hardware or feed is absent reports SKIP (no Vulkan ICD, no NVIDIA
driver, no Wayland session, no HLS feed). A `cargo test` that passes with zero
tests is a FAIL, since that is what a feature-gated test file does when its
feature is missing.

**Create the token.** GitHub → Settings → Developer settings → Personal
access tokens → Fine-grained tokens → Generate new token. Resource owner: the
account that owns the repository. Repository access: Only select repositories
→ `Glass2GlassHQ/glass2glass`. Repository permissions: **Commit statuses →
Read and write**, nothing else. Generate, copy, then on the host:

```sh
install -d -m 700 ~/.config/glass2glass-validation
touch ~/.config/glass2glass-validation/token
chmod 600 ~/.config/glass2glass-validation/token
cat > ~/.config/glass2glass-validation/token   # paste the token, then Ctrl-D
```

The script refuses to run unless that file is mode 0600 and owned by the
caller. `$G2G_VALIDATION_TOKEN_FILE` overrides the path.

**Install the timer.** With the checkout at `~/src/glass2glass` (anywhere
else, override `ExecStart` with `systemctl --user edit`):

```sh
install -d -m 755 ~/.config/systemd/user
install -m 644 tools/systemd/glass2glass-validation@.service \
               tools/systemd/glass2glass-validation@.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now glass2glass-validation@desktop-gpu.timer  # 3060 desktop
systemctl --user enable --now glass2glass-validation@bench.timer        # bench box
loginctl enable-linger "$USER"   # headless host: start the user manager at boot
```

The timer fires 10 minutes after the user manager starts and once a day after
that, so a boot-and-work session gets one run.

**Read the results.** The statuses appear on the commit in the GitHub UI, or:

```sh
gh api repos/Glass2GlassHQ/glass2glass/commits/master/status \
  --jq '.statuses[] | "\(.context)\t\(.state)\t\(.description)"'
journalctl --user -u glass2glass-validation@desktop-gpu.service -n 200
```

`tools/host-validation.sh <suite> --dry-run` prints the status payload instead
of posting it, and needs no token.

The timer runs whatever is on `master`, so this is exactly as safe as push
access to `master`.

## System dependencies

The cargo features pull pure Rust crates. OS-level dependencies must be on the
host.

| Distro | Decoder (`ffmpeg`) | Wayland sink | KMS sink | VAAPI |
| :--- | :--- | :--- | :--- | :--- |
| Fedora | `ffmpeg-devel` (RPM Fusion) or `ffmpeg-free-devel` | `wayland-devel` | `libdrm-devel` | `libva-devel` |
| Debian / Ubuntu | `libavcodec-dev libavformat-dev libavutil-dev libswscale-dev` | `libwayland-dev` | `libdrm-dev` | `libva-dev` |
| Arch | `ffmpeg` | `wayland` | `libdrm` | `libva` |

For the CUDA path, install the NVIDIA driver and CUDA runtime your
distribution ships, with `libnvcuvid.so` and `libcuda.so` on the linker path.
`Backend::NvdecCuvid` / `Backend::NvdecCuda` need an `ffmpeg` build with cuvid
support.

The loopback RTSP relay is `mediamtx`, run from docker or as a single binary
(see [A standing RTSP feed](#a-standing-rtsp-feed)).

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
design/          # architecture overview, track documents and open work
DEVTOOLS.md      # developer tooling reference
docs/            # GitHub Pages site
```

## License

The whole repository is MPL-2.0: every crate, including the examples, tools,
and test fixtures. See [LICENSE](LICENSE).
