# bevy-g2g

Bevy integration for **g2g**: remote rendering over WebRTC and zero-copy video
decode, with no pipeline code in the app.

> Standalone crate, **not** a workspace member (excluded in the repo root
> `Cargo.toml`). It builds against g2g as path packages and keeps its own
> `Cargo.lock`, so the heavy Bevy dependency never enters the normal build / CI.
> Bevy 0.19 pins wgpu 29.0.3, the exact version g2g uses, which is what lets a
> `wgpu::Device` / `wgpu::Texture` cross between the two with no type mismatch.

## Remote rendering (`RemoteRenderPlugins`)

The app renders headless (no window) and every frame is encoded to H.264 and
published to a WHIP endpoint over WebRTC, or written to a file. The app adds
one plugin group and spawns its scene; the camera is retargeted onto the
stream texture automatically.

```rust
use bevy::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(bevy_g2g::RemoteRenderPlugins::from_env())
        .add_systems(Startup, setup_scene);
    bevy_g2g::run(app); // runs, flushes the stream, exits the process
}
```

`StreamSettings` carries every knob (resolution, fps, bitrate, keyframe
interval, output, frame cap); `from_env` reads the demo-run convention
(`G2G_WHIP_URL`, `G2G_FRAMES`).

Two encode paths, chosen automatically:

- **Zero-copy** (`--features nvenc`, NVIDIA): Bevy renders on g2g's interop
  device (Vulkan + `VK_KHR_external_memory_fd`, handed to Bevy via
  `RenderCreation::Manual`), each rendered texture is copied device→device
  into a CUDA surface (`WgpuToCuda`) and encoded by the native NVENC. Only the
  H.264 access units leave the GPU, the cloud-gaming / pixel-streaming shape.
  If the interop device cannot be created at runtime (no NVIDIA / no CUDA),
  the plugin logs a warning and falls back to:
- **Readback** (default, any adapter): the render target is read back to
  system memory after each frame and the sink pipeline converts + encodes it
  with libx264 (`videoconvert → ffmpegenc`).

`bevy_g2g::run` exits the process after flushing rather than dropping the
`App`: on the zero-copy path the render world holds a CUDA context and an
NVENC session on Bevy's device, and their drop order races Bevy's own device
teardown in the driver.

### Try it

```sh
cargo run --release --features nvenc --example stream   # zero-copy (NVIDIA)
cargo run --release --example stream                    # readback + libx264
ffplay bevy_g2g.h264
```

Stream live over WebRTC: run [MediaMTX](https://github.com/bluenviron/mediamtx)
(`docker run --rm -it --network host bluenviron/mediamtx:1`), then

```sh
G2G_WHIP_URL=http://127.0.0.1:8889/g2gbevy/whip G2G_FRAMES=0 \
  cargo run --release --features nvenc --example stream
```

and open `../g2g-plugins/examples/whep-player.html` with the WHEP URL
`http://127.0.0.1:8889/g2gbevy/whep`. `G2G_FRAMES=0` runs until Ctrl-C.

## Video playback (`--features decode`, `VideoPlayerPlugin`)

The inverse direction: a stock windowed Bevy app keeps its own wgpu device,
g2g joins it (`GpuContext::from_wgpu`) and decodes an H.264 clip straight onto
it. Every decoded frame is a `wgpu::Texture` Bevy binds in its own render
graph: no second device, no readback, no copy. Any mesh tagged `VideoScreen`
plays the video on its `StandardMaterial`.

```rust
App::new()
    .add_plugins(DefaultPlugins)
    .add_plugins(bevy_g2g::VideoPlayerPlugin::new("my.h264"))
    .add_systems(Startup, |mut c: Commands, /* ... */| {
        // spawn a mesh with the VideoScreen component
    })
    .run();
```

```sh
cargo run --release --features decode --example decode            # bundled clip
cargo run --release --features decode --example decode -- my.h264
G2G_EXIT_AFTER_SECS=8 cargo run --release --features decode --example decode  # smoke run
```

Needs a display, a wgpu adapter, and the ffmpeg libraries (software H.264
decode).

## Notes

- **Pipelines compile synchronously** (`synchronous_pipeline_compilation`):
  Bevy's async shader compilation runs Vulkan pipeline creation on a
  background thread that, on the NVIDIA driver, faults when it overlaps the
  CUDA encode on the same device. Harmless startup latency on the readback
  path.
- **NVENC AV1 needs an RTX 40-series** (Ada). Ampere (30-series) does H.264 +
  HEVC encode only, which is why this streams H.264 (also the codec
  `WebRtcSink` speaks).
- **WebRTC playback needs periodic IDRs**: the zero-copy path forces one
  every `keyframe_interval` frames with in-band SPS/PPS; the software encoder
  keyframes on its GOP and on downstream keyframe requests.
- The live WHIP/WebRTC leg is validated against a browser / WHEP player by a
  human; it is not part of an automated test.
