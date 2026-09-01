# g2g-mutate-demo

Splicing a transform into a pipeline that is already playing, on screen.

`rtspsrc -> ffmpegdec -> waylandsink` shows an RTSP feed in a window. Every five
seconds a `videoflip` is spliced onto the decoded-video edge, held for five, and
lifted back off, while the stream keeps running: the picture turns over and back
with no gap and no restart. Each operation prints as it happens, and a removed
element is handed back to the caller alive.

That is [`GraphMutator`](../../g2g-core/src/runtime/mutate.rs) (M1115), from
`run_graph_mutable` beside the run future. The protocol is DESIGN.md §4.8.6, and
PORTING.md §5.2 maps it to the GStreamer pad-block-and-relink idiom it replaces.

## Running it

Needs a Wayland session and an RTSP feed. The feed the default URL expects is the
mediamtx + ffmpeg recipe from the repository README:

```sh
cd examples/g2g-mutate-demo
cargo run --release
```

| Variable | Effect |
| :--- | :--- |
| `G2G_RTSP_URL` | The feed to play (default `rtsp://localhost:8554/pattern`) |
| `G2G_DEMO_SECONDS` | End after N seconds instead of waiting for ctrl-c |
| `G2G_DEBUG` | The usual log spec, e.g. `G2G_DEBUG='*:info'` |

Ctrl-c closes the pipeline: the run future is dropped, which ends every arm and
takes the RTSP session and the window with it.

Standalone (workspace-excluded, own `Cargo.lock`) because the `rtsp` + `ffmpeg` +
`wayland-sink` feature set pulls libavcodec and a Wayland session, which the
workspace's default-feature build and CI stay clear of.

## What to expect

```
playing rtsp://localhost:8554/pattern in a window titled "g2g mutate demo"
splicing a 180-degree videoflip in and out every 5s; ctrl-c to stop
insert after dec: VideoFlip0 is now turning the picture
remove VideoFlip0: handed back (method=Some(Str("rotate-180"))), back to the decoder's own picture
```

The decoder emits NV12, which is what `waylandsink` takes, so the splice is
caps-preserving and needs no converter. A 180-degree turn keeps the geometry too;
a quarter turn would change the caps, which the mutator would check against what
the sink accepts before committing.
