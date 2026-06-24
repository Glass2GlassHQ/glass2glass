# bevy-g2g-stream

A server-side **render-and-stream** demo: a [Bevy](https://bevyengine.org) app
renders a 3D scene **headless** (no window) on the GPU, and **g2g** captures the
rendered frames off Bevy's *own* GPU device and encodes them — the cloud-gaming /
pixel-streaming shape (render on a server, stream to a thin client).

This is the adoption artifact for g2g's bring-your-own-device path
(`GpuContext::from_wgpu`, M263): instead of g2g opening its own GPU, it **joins
the engine's** wgpu device, so a texture Bevy renders is a first-class object g2g
can read with no second device.

> Standalone crate — **not** a workspace member (excluded in the repo root
> `Cargo.toml`). It builds against g2g as path packages and keeps its own
> `Cargo.lock`, so the heavy Bevy dependency never enters the normal build / CI.

## Why the device handoff works

Bevy 0.19 and g2g both pin **wgpu 29.0.3**. Because they resolve to one `wgpu` in
this crate's lockfile, a `wgpu::Texture` / `wgpu::Device` crosses between them
with no type mismatch. The demo clones Bevy's `RenderDevice` / `RenderQueue` /
`RenderAdapter` / `RenderInstance` (all reference-counted handles) into
`GpuContext::from_wgpu(...)` — sharing the GPU, not duplicating it.

## Pipeline

```
Bevy headless render ──► wgpu::Texture ──► g2g read-back ──► AppSrc ──► VideoConvert ──► FfmpegH264Enc ──► sink
   (server GPU)        (shared device,        (on the           (RGBA)    (RGBA→I420)      (NVENC H.264)   (file / WHIP)
                        from_wgpu)         shared device)
```

- **Phase A** (implemented): headless render of a spinning cube to an offscreen
  texture, `GpuContext::from_wgpu` over Bevy's device, and a render-world system
  that reads the rendered texture back off the shared device each frame. Asserts
  the captured frame is a real (non-blank) render.
- **Phase B** (implemented): the read-back RGBA frames are pushed into a g2g
  `AppSrc → VideoConvert → FfmpegH264Enc → FileSink` pipeline (on its own thread),
  writing an H.264 Annex-B file of the rendered scene, NVENC-encoded. Validated on
  the RTX 3060: a 240-frame run produces a valid `h264` 640×480 stream
  (`ffprobe` confirms codec/geometry/frame count).
- **Phase C** (implemented): when `G2G_WHIP_URL` is set, the sink is
  `WebRtcSink` instead of `FileSink` — the encoded H.264 is published to a WHIP
  endpoint over WebRTC (str0m: ICE/DTLS/SRTP) and viewable in a browser. The
  encode pipeline runs inside a tokio runtime (the WHIP handshake + session need
  a reactor). This is the live pixel-streaming leg, validated by a human against
  a real WHIP server / browser (not an automated test).

## Run

Needs an NVIDIA GPU (for the `h264_nvenc` encode; phase A needs any wgpu adapter)
and a libavcodec with `h264_nvenc` (or `libx264`).

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

## Honest caveats

- **Phase A/B read the texture back to CPU** because the H.264 encoder here is the
  `FfmpegH264Enc` NVENC path, which ingests CPU `I420`. The render→capture stays
  on the one GPU device (no second device, no PCIe round-trip *between* GPUs), but
  there is a device→host read-back. The fully zero-copy path — wgpu texture →
  CUDA → NVENC with no read-back — is a **native NVENC element** (NVIDIA Video
  Codec SDK), the reverse of the `CudaToWgpu` (M220) bridge, tracked as the moat
  follow-up.
- **NVENC AV1 needs an RTX 40-series** (Ada). Ampere (e.g. an RTX 30-series) does
  H.264 + HEVC encode only, which is why this streams H.264 (also the codec
  `WebRtcSink` speaks).
- The **live WHIP/WebRTC leg** (phase C) is validated against a browser / WHEP
  player by a human; it is not part of an automated test.
