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
published to a WHIP endpoint over WebRTC, to a MoQ Transport relay, or written
to a file. The app adds one plugin group and spawns its scene; the camera is
retargeted onto the stream texture automatically.

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
(`G2G_WHIP_URL`, `G2G_MOQT_URL`, `G2G_FRAMES`).

`StreamOutput` picks the egress: `Whip` ends the pipeline in `webrtcsink`,
`Moqt` in `mp4mux → moqtsink`, and `File` (the default) in `filesink`.

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

### Windowed streaming

`RemoteRenderPlugins::windowed(settings)` keeps a normal window (winit event
loop) and streams at the same time: the scene camera renders into the stream
texture and the window shows that texture through a fullscreen mirror, so the
desktop view and the stream are the same pixels. Both encode paths work; the
pacing follows the window loop rather than `fps`. In the example:
`G2G_WINDOW=1 cargo run --release --example stream`.

### Input backchannel

Set `StreamSettings::input_port` (env: `G2G_INPUT_PORT`) and the plugin group
serves a WebSocket that injects viewer input as ordinary Bevy input messages,
so `ButtonInput<KeyCode>`, `AccumulatedMouseMotion`, etc. work unchanged. A
WebRTC data channel cannot reach the publisher through a WHIP/WHEP media
server (the viewer is a separate peer connection), which is why the
backchannel is a WebSocket, the standard pixel-streaming shape.

One JSON object per text frame; `code` is the W3C `KeyboardEvent.code`
string, which is also the Bevy `KeyCode` variant name:

```json
{"type":"key","code":"KeyW","down":true}
{"type":"mouse_move","dx":3.5,"dy":-1.0}
{"type":"mouse_button","button":"left","down":true}
{"type":"wheel","dx":0.0,"dy":-1.0}
```

`examples/remote-viewer.html` is a WHEP player with input capture (click the
video, WASD/arrows move the demo cube); `examples/input_probe.rs` is a tiny
CLI client used for automated validation:

```sh
G2G_INPUT_PORT=8877 G2G_FRAMES=0 cargo run --release --example stream &
cargo run --release --example input_probe   # logs: cube moving
```

### Launching MediaMTX

The WHIP endpoint the app publishes to is a media server; viewers play the
same stream back from it over WHEP. [MediaMTX](https://github.com/bluenviron/mediamtx)
serves both with zero configuration. Container:

```sh
docker run --rm -it --network host bluenviron/mediamtx:1
# Podman: podman run --rm -it --network host docker.io/bluenviron/mediamtx:1
```

Or the standalone binary: it is a single static executable, so download the
release for your platform from
<https://github.com/bluenviron/mediamtx/releases>, unpack, and run
`./mediamtx`.

Either way it listens immediately: publish to
`http://<mediamtx-machine>:8889/<name>/whip`, watch at
`http://<mediamtx-machine>:8889/<name>/whep`, any stream name (`127.0.0.1`
when everything runs on one box). The game and MediaMTX can share a machine
or not; only MediaMTX needs to be reachable by viewers. The bundled
`mediamtx.yml` is only needed for extras like auth or recording.

### Try it

```sh
cargo run --release --features nvenc --example stream   # zero-copy (NVIDIA)
cargo run --release --example stream                    # readback + libx264
ffplay bevy_g2g.h264
```

Stream live over WebRTC: launch MediaMTX (above), then

```sh
G2G_WHIP_URL=http://127.0.0.1:8889/g2gbevy/whip G2G_FRAMES=0 \
  cargo run --release --features nvenc --example stream
```

and open `../g2g-plugins/examples/whep-player.html` with the WHEP URL
`http://127.0.0.1:8889/g2gbevy/whep`. `G2G_FRAMES=0` runs until Ctrl-C.

### MoQ Transport egress

`G2G_MOQT_URL` publishes to an IETF MoQT relay instead: the encoded H.264 goes
through `mp4mux` and `moqtsink` ships each `moof`+`mdat` as one MOQT object,
groups starting at keyframes. `G2G_MOQT_NAMESPACE` names the broadcast
(default `bevy`) and `G2G_MOQT_CERT_HASHES` takes comma-separated hex SHA-256
digests of relay certificates to accept, which a self-signed local relay needs.

```sh
G2G_MOQT_URL=https://127.0.0.1:4443/ G2G_MOQT_NAMESPACE=bevy G2G_FRAMES=0 \
  cargo run --release --example stream
```

The whole thing from one command, relay and browser included:

```sh
cd ../tools/moqt-demo && node watch-bevy.mjs
```

That mints a certificate, starts a local `moq-relay-ietf`, runs this example
against it, serves the MoQT player page and opens a browser on it.

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
