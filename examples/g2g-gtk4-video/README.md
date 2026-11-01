# g2g-gtk4-video

Shows the output of any g2g launch line in a GTK4 window, the way
`gtk4paintablesink` does for GStreamer. The example appends
`! appsink channel=...` to the line, runs it on its own thread, and turns each
frame into a `GdkTexture` for a `GtkPicture`. It reads the frame's format and
size from `AppSinkPull::caps`.

- A DMABUF frame (`v4l2src io-mode=dmabuf`, `pipewirevideosrc`) becomes a
  `GdkDmabufTexture`. GTK imports the buffer and the pixels are not copied on
  the CPU. The example maps YUYV, NV12, RGBA and BGRA to their DRM fourccs and
  assumes a linear layout.
- A system-memory RGBA, BGRA or RGB frame is copied once into a
  `GdkMemoryTexture`. Any other format needs a `videoconvert` first.

```sh
cargo run --release
cargo run --release -- "v4l2src io-mode=dmabuf ! video/x-raw,format=YUY2,width=640,height=480,framerate=30/1"
cargo run --release -- "filesrc location=clip.h264 ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=RGBA ! clocksync"
```

The third line needs the `ffmpeg` feature added to `g2g-plugins` in
`Cargo.toml`. Needs GTK 4.14 or newer. `GDK_DEBUG=dmabuf` shows how GTK imports
each buffer. On an AMD GPU, GTK's Vulkan renderer cannot import linear YUYV, so
GTK imports it through GL and copies it once on the GPU.
