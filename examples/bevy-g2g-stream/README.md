# bevy-g2g-stream

A server-side **render-and-stream** demo: a [Bevy](https://bevyengine.org) app
renders a 3D scene **headless** (no window) on the GPU, and **g2g** encodes the
rendered frames to H.264 **with no GPU→CPU read-back** — the cloud-gaming /
pixel-streaming shape (render on a server, stream to a thin client).

This is the adoption artifact for g2g's zero-copy GPU-render egress: Bevy renders
on g2g's *interop* GPU device, and each rendered texture goes straight into a CUDA
surface and the native NVENC encoder. The pixels never leave the GPU until they
are compact H.264.

> Standalone crate — **not** a workspace member (excluded in the repo root
> `Cargo.toml`). It builds against g2g as path packages and keeps its own
> `Cargo.lock`, so the heavy Bevy dependency never enters the normal build / CI.

## Why it is zero-copy

The trick is making Bevy render on **g2g's interop device**. g2g creates a Vulkan
`wgpu::Device` with `VK_KHR_external_memory_fd` (`create_interop_device_full`) and
hands it to Bevy via `RenderCreation::Manual`. Bevy 0.19 and g2g both pin **wgpu
29.0.3**, so the handle types match and Bevy renders directly on the device whose
memory g2g can export to CUDA.

After Bevy renders a frame, the `WgpuToCuda` bridge (M275) copies the target
texture **device→device** into a CUDA-shared image and hands it as a
`MemoryDomain::Cuda` frame to the native `NvEnc` (M269/M271): NVENC color-converts
the RGBA surface and emits H.264, all on the GPU. Only the encoded access units
cross to the CPU. (The earlier version, M267, read the full RGBA frame back to
system memory and encoded with the ffmpeg NVENC backend; that read-back is gone.)

## Pipeline

```
Bevy headless render ─► wgpu::Texture ─► WgpuToCuda ─► NvEnc ─► H.264 AU ─► AppSrc ─► sink
   (interop device)     (same device)   (dev→dev    (CUDA RGBA  (channel)           (file / WHIP)
                                         to CUDA)    → H.264)
└──────────────────────── all on the GPU, no read-back ───────────────────────┘   └─ CPU ─┘
```

- **Render world** (`encode_via_g2g`): after Bevy renders into the target
  texture, `WgpuToCuda` copies it into a CUDA surface and `NvEnc` encodes it,
  emitting H.264 access units to the main world over a channel. No read-back.
- **Main world** (`drain_frames`): pushes the access units into the g2g sink
  pipeline's `AppSrc` feed and stops after the frame cap.
- **Sink thread** (`sink_pipeline`): `AppSrc(H.264) → FileSink` (default) or
  `→ WebRtcSink` (WHIP egress when `G2G_WHIP_URL` is set). No encoder in this
  chain — the GPU already produced H.264.

Validated on an RTX 3060: a run produces a valid `h264` 640×480 stream that
decodes back to the rendered scene (the spinning cube on a lit plane).

## Run

Needs an **NVIDIA GPU** with Vulkan + CUDA + the NVENC headers (the encode path is
the native NVIDIA Video Codec SDK, not ffmpeg). H.264 + HEVC encode on Ampere
(30-series); AV1 needs Ada (40-series).

### Capture to a file (self-contained)

```sh
cd examples/bevy-g2g-stream
cargo run --release            # writes bevy_g2g.h264 (H.264 Annex-B)
ffplay bevy_g2g.h264           # or vlc / mpv
```

### Stream live over WebRTC (WHIP)

Point it at a WHIP server and watch in a browser. With
[mediamtx](https://github.com/bluenviron/mediamtx) running locally (defaults to
`:8889`):

```sh
G2G_WHIP_URL=http://localhost:8889/g2gbevy/whip \
G2G_FRAMES=0 \
  cargo run --release
```

`G2G_FRAMES=0` runs forever (until Ctrl-C) so you can watch the stream; omit it
for the default fixed-length run. Open the bundled WHEP viewer
`../../g2g-plugins/examples/whep-player.html` (set its URL to the WHEP endpoint,
e.g. `http://localhost:8889/g2gbevy/whep`) to see the spinning cube, rendered on
the server GPU and streamed by g2g.

### Environment

| Var | Effect |
| :-- | :-- |
| `G2G_WHIP_URL` | If set, stream to this WHIP endpoint; else write to `bevy_g2g.h264`. |
| `G2G_FRAMES` | Frames to render before exit; `0` = forever. Default 240. |

## Notes

- **Pipelines are compiled synchronously** (`synchronous_pipeline_compilation:
  true`). Bevy's default async shader compilation runs Vulkan pipeline creation on
  a background thread that, on the NVIDIA driver, faults when it overlaps the CUDA
  encode on the same device (Vulkan + CUDA concurrency on the shared driver).
  Synchronous compile serialises it with the after-render encode system.
- **The process exits via `std::process::exit`** once the H.264 is flushed,
  skipping the render-world GPU teardown (the CUDA context + NVENC session on
  Bevy's device). Dropping those races Bevy's own device teardown on shutdown; the
  work is done, so the OS reclaims the GPU resources — the standard GPU-demo
  shutdown approach.
- **NVENC AV1 needs an RTX 40-series** (Ada). Ampere (e.g. an RTX 30-series) does
  H.264 + HEVC encode only, which is why this streams H.264 (also the codec
  `WebRtcSink` speaks).
- The **live WHIP/WebRTC leg** is validated against a browser / WHEP player by a
  human; it is not part of an automated test.
```
