# g2g-api-soak

Runs an RTSP decode graph forever through the g2g Rust API, for the calliope
canary's `api-rebuild` and `api-mutate` rows. Every other canary row runs
`g2g-launch`.

```sh
g2g-api-soak [-q] --observe <port> <rebuild | mutate>
```

Both scenarios build `rtspsrc ! h264parse ! avdec_h264 ! checksumsink name=video`
from `rtsp://127.0.0.1:18554/baseline` with the typed constructors, and serve the
same telemetry and bus events as `g2g-launch --observe` on `127.0.0.1:<port>`
for the whole run.

- `rebuild` runs the graph for 60 s, stops it with `set_state(Null)` on its
  `StateController`, drops it and builds a new one.
- `mutate` puts a tee in front of the `video` sink and runs one graph. Every 4 s
  it takes the next step of a cycle through `GraphMutator`: add a
  `videoscale ! checksumsink` branch to the tee, set and read back `h264parse`'s
  `config-interval`, splice an `identity` after the decoder, remove it, remove
  the branch.

Any error from the API, a stop that takes longer than 10 s, or a mutation with
no answer within 10 s ends the program with exit code 1 and the error on stderr.
`-q` drops the line printed per cycle.

Standalone (workspace-excluded, own `Cargo.lock`) because `rtsp` + `ffmpeg` +
`observe` pulls libavcodec, which the workspace's default-feature build and CI
stay clear of.

```sh
cd examples/g2g-api-soak && cargo build --release --locked
```
