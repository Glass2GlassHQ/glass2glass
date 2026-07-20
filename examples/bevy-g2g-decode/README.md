# bevy-g2g-decode

The bring-your-own-device embedder demo (M741): a stock Bevy app owns the
window and the wgpu device; g2g joins that device via `GpuContext::from_wgpu`
and decodes video straight onto it. Every decoded frame is a `wgpu::Texture`
Bevy binds in its own render graph, no second device, no readback, no copy.
The inverse of [`bevy-g2g-stream`](../bevy-g2g-stream) (there Bevy renders and
g2g encodes; here g2g decodes and Bevy consumes).

## What it does

1. Bevy starts the ordinary way (`DefaultPlugins`, its own device).
2. After plugin setup, the app clones Bevy's `RenderDevice` / `RenderQueue` /
   `RenderAdapter` / `RenderInstance` into `GpuContext::from_wgpu`.
3. A g2g pipeline runs on its own thread:
   `filesrc -> h264parse -> ffmpegdec -> videoconvert -> vello overlay -> appsink`.
   The overlay stage renders each RGBA frame into a `wgpu::Texture` on the
   shared (= Bevy's) device; the appsink hands the frames to the app over a
   bounded pull channel (backpressure paces the decode).
4. The app registers each texture as a render-world `GpuImage` under a
   reserved image handle and binds the current frame to a spinning cube's
   `StandardMaterial` (through an sRGB view, so gamma is correct in a lit,
   tonemapped scene). Playback loops at 30 fps once the clip ends.

## Run

Needs a display, a wgpu adapter, and the ffmpeg libraries (the decode is
software H.264).

```sh
cargo run --release                 # the bundled 640x480 test clip
cargo run --release -- my.h264      # any Annex-B H.264 stream
```

Close the window or press Esc to quit. For an unattended smoke run:

```sh
G2G_EXIT_AFTER_SECS=8 cargo run --release
```

exits 0 once frames were decoded and bound to the material.

## Notes

- Standalone crate (excluded from the workspace) so Bevy stays out of the
  normal build and CI; it pins its own `Cargo.lock`. Bevy 0.19 pins wgpu 29,
  the same version g2g uses, which is what makes the handle handoff
  type-check.
- Bevy keeps the `RenderInstance` in the render world only, and `cleanup()`
  moves the render world to the pipelined-rendering thread: clone the handles
  after `app.finish()` and before `app.cleanup()` (see `main`).
