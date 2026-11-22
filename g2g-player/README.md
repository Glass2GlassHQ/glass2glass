# g2g-player

An embeddable media player on glass2glass, modelled on libmpv. `Player` loads a URI or a `gst-launch` line, plays, pauses and seeks it, and reports state through one event queue and named properties. Video is drawn into the host's own OpenGL framebuffer, or into a wgpu texture with the `wgpu` feature.

```rust
use g2g_player::{GlRenderContext, GlTarget, LoadOptions, Media, Player, Property};

let player = Player::new();
player.observe(Property::Duration);
player.set_update_callback(|| { /* schedule a redraw */ });
player.load(Media::Uri("file:///path/clip.mp4"), LoadOptions::default())?;

// on the GL thread, with the context current
let mut render = GlRenderContext::new(&player, |name| get_proc_address(name));
unsafe {
    render.render(GlTarget { framebuffer: fbo, width, height, flip_y: true })?;
}
```

A launch line includes `playervideosink` for video, `playeraudiosink` for host
audio, or both. Call `player.set_audio_callback(callback)` before loading to
receive host audio. URI loads select that output automatically. The playback
thread calls the callback with borrowed interleaved PCM and its negotiated
format, rate, channel layout, and timing. URI playback uses playbin's audio
conversion chain. A launch line can pin its own format and rate.
`AudioOutputEvent::Flush` tells the host to discard previously queued samples
after seeking.

## Tests

```sh
cargo test -p g2g-player --features wgpu,openh264
```

The test skips without a GPU adapter.
