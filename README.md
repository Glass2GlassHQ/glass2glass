# glass2glass (`g2g`)

A hardware-first, sans-IO, asynchronous multimedia graph framework in pure
Rust — designed to replace GStreamer in AI-driven, real-time embedded (RTOS),
cloud ingestion, and browser targets.

The name reflects the metric the project optimizes for: **glass-to-glass
latency**, the time between physical photon capture and hardware presentation.

See [DESIGN.md](DESIGN.md) for the architecture specification.

## The four pillars

1. **Async execution.** Every element is a cooperative `Future`. The
   framework is runtime-agnostic (Tokio on servers, Embassy on RTOS,
   `wasm-bindgen-futures` in the browser).
2. **Hardware-first, zero-copy.** Buffers live in DMABUF / Vulkan /
   CUDA / D3D11 / WebGPU memory domains; CPU memory copies are treated
   as system faults.
3. **`no_std + alloc` + sans-IO core.** The same pipeline shape runs on
   a Cortex-M, a multi-threaded server, or `wasm32`.
4. **First-class ML.** Tensor allocation, reshaping, and pipeline
   batching are part of graph orchestration.

## Workspace

| Crate | Role | Profile |
| :--- | :--- | :--- |
| `g2g-core` | Traits, `Frame`/`PipelinePacket`, caps algebra, clock, runner. | `no_std + alloc` |
| `g2g-plugins` | Sources/sinks/transforms (RTSP, RTP, ffmpeg, VAAPI, MF, Wayland, KMS, WASAPI, Embassy, web). | mixed |
| `g2g-ml` | ORT, Burn, WgpuPreprocess, TensorPostprocess. | `std` |
| `g2g-bridge` | GStreamer C-FFI bridge. | `std` |
| `g2g-enterprise` | Multi-stream tensor batcher. | `std` |

## Build

Stable Rust, MSRV 1.75, `resolver = "2"`.

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
| `FfmpegH264Dec` (sw / `NvdecCuvid` / `NvdecCuda`) | `ffmpeg` | Linux + libavcodec |
| `VaapiH264Dec` | `vaapi` | Linux + libva + GBM |
| `MfDecode` / `MfEncode` / `MfAacEncode` / `MfAacDecode` | `mf-decode`, `mf-encode`, `mf-aac` | Windows + Media Foundation |
| `WaylandSink` | `wayland-sink` | Linux + Wayland |
| `KmsSink` | `kms-sink` | Linux + libdrm; needs DRM master / tty |
| `D3D11Sink` | `d3d11-sink` | Windows |
| `CudaDownload`, `CudaGlSink` | `cuda`, `cuda-gl` | Linux + NVIDIA driver (libcuda) + EGL + GL |
| `UdpSink` + RTP packetizer | `udp-egress` | — |
| `WasapiSink` / `WasapiSrc` | `wasapi-sink`, `wasapi-src` | Windows |
| `OrtInference` (+ CUDA / DirectML EPs) | `ort`, `cuda`, `directml` (in `g2g-ml`) | onnxruntime |
| `BurnInference` | `burn` (in `g2g-ml`) | wgpu (Vulkan / Metal / DX12) |
| `WgpuPreprocess` | `wgpu` (in `g2g-ml`) | wgpu |
| Embassy / RTOS pool + clock | `embassy`, `embassy-link` | — |
| Browser elements | `web`, `web-codecs` | `wasm32-unknown-unknown` |

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

### File capture → H.264 parse → fMP4 record

```rust
let src   = FileSrc::open("in.h264")?;
let parse = H264Parse::new();
let sink  = Mp4Sink::open("out.mp4")?;

run_source_transform_sink(src, parse, sink, &clock, LatencyProfile::Live).await?;
```

### Camera → encode → RTP egress over UDP

```rust
let src  = VideoTestSrc::new(RawVideoFormat::Nv12, 1920, 1080, 30.0);
let enc  = MfEncode::new_low_latency();                  // Windows; on Linux use the bridge
let sink = UdpSink::bind("0.0.0.0:0")?
    .with_remote("239.0.0.1:5004")
    .with_rtp(96, 0x1234_5678);                          // payload type, SSRC

run_source_transform_sink(src, enc, sink, &clock, LatencyProfile::Live).await?;
```

Features: `udp-egress` (plus the platform encoder feature).

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
g2g-core/        # traits, runner, solver, frame, caps, clock
g2g-plugins/     # all source/sink/transform elements
g2g-ml/          # ORT, Burn, WgpuPreprocess, batcher
g2g-bridge/      # GStreamer C-FFI bridge (libgstglass2glass.so)
g2g-enterprise/  # multi-stream tensor batcher
DESIGN.md        # architecture specification
docs/            # GitHub Pages site
```

## License

`g2g-core`, `g2g-plugins`, `g2g-ml`, `g2g-bridge`: LGPL v2.1+.
`g2g-enterprise`: AGPL v3.

See [LICENSE](LICENSE).
