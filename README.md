# glass2glass (`g2g`)

[![CI](https://github.com/Glass2GlassHQ/glass2glass/actions/workflows/ci.yml/badge.svg)](https://github.com/Glass2GlassHQ/glass2glass/actions/workflows/ci.yml)

A hardware-first, sans-IO, asynchronous multimedia graph framework in pure
Rust.

The name reflects the metric the project optimizes for: **glass-to-glass
latency**, the time between physical photon capture and hardware presentation.

See [DESIGN.md](DESIGN.md) for the architecture specification and
[DEVTOOLS.md](DEVTOOLS.md) for the developer tooling (`cargo xtask`, the pipeline
visualizer, the caps explainer, benchmarks).

## Coming from GStreamer?

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
| `g2g-plugin` | SDK for dynamically loadable plugins (`declare_plugin!` + ABI tag). | `no_std + alloc` |
| `g2g-plugins` | Sources/sinks/transforms (RTSP, RTP in/out, HTTP/HLS/DASH/RTMP ingest, V4L2 / PipeWire / MF capture, ffmpeg, VAAPI, MF, VideoToolbox (macOS), MediaCodec (Android), Wayland, KMS, WASAPI, ALSA / PulseAudio / PipeWire audio, compositor, Embassy, web), container mux/demux (MP4, MPEG-TS, Matroska/WebM, FLV, Ogg), codec parsers + encoders (AV1, VP8/9, MJPEG), the tag system, and the `gst-launch` text DSL. | mixed |
| `g2g-ml` | ORT, Burn, WgpuPreprocess, TensorPostprocess. | `std` |
| `g2g-bridge` | GStreamer C-FFI bridge. | `std` |
| `g2g-enterprise` | Multi-stream tensor batcher. | `std` |
| `g2g-python` | Hosts gst-python-ml elements in-process (embedded CPython via pyo3). | `std` |
| `g2g-capi` | C ABI (cdylib/staticlib + `g2g.h`): launch pipelines + bus + appsrc/appsink from any language. | `std` |
| `g2g-pyapi` | Python (pyo3) bindings: drive pipelines + bus + appsrc/appsink. | `std` |

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
| `FfmpegH264Dec` (sw / `NvdecCuvid` / `NvdecCuda` / `Vaapi`) | `ffmpeg` | Linux + libavcodec |
| `VaapiH264Dec` | `vaapi` | Linux + libva + GBM |
| `MfDecode` / `MfEncode` / `MfAacEncode` / `MfAacDecode` | `mf-decode`, `mf-encode`, `mf-aac` | Windows + Media Foundation |
| `VtDecode` (H.264; CI-compiled, on-device decode pending) | `vtdecode` | macOS + VideoToolbox |
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
| `WebRtcSink` (WHIP egress, H.264 + Opus) / `WebRtcWhepSrc` (WHEP ingest, H.264), via str0m: ICE/DTLS/SRTP | `webrtc` | str0m (rust-crypto) + reqwest |
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
`videobalance` / `videobox` / `alpha`, `audioconvert` / `audioresample` /
`audiomixer` / `volume` / `audiopanorama`), the `compositor`, the tag system,
and the `gst-launch` text DSL (`parse_launch` / `gst-inspect`) are all in the
pure `no_std + alloc` default build (no feature flag).

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
g2g-core/        # traits, runner, solver, frame, caps, clock
g2g-plugin/      # dynamic-plugin SDK (declare_plugin! + ABI tag)
g2g-plugins/     # all source/sink/transform elements
g2g-ml/          # ORT, Burn, WgpuPreprocess, batcher
g2g-bridge/      # GStreamer C-FFI bridge (libgstglass2glass.so)
g2g-enterprise/  # multi-stream tensor batcher
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

All crates are LGPL v2.1+ except `g2g-enterprise`, which is AGPL v3.

See [LICENSE](LICENSE).
