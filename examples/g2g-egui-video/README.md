# g2g-egui-video

Shows g2g pipeline video inside an egui window. g2g opens the wgpu device and
eframe renders on it (`WgpuSetup::Existing`), so every frame the pipeline
produces is a `wgpu::Texture` egui draws directly, with no readback or upload.
The pipeline runs on its own thread and hands frames over through an appsink
channel. The window keeps the newest frame and registers it with egui-wgpu as a
native texture.

```sh
cargo run --release                 # videotestsrc, a bouncing ball
cargo run --release -- clip.h264    # an H.264 Annex-B file
```

Needs a display, a wgpu adapter, and the ffmpeg libraries for the file path.

eframe 0.37 is on wgpu 30, the version g2g uses.
