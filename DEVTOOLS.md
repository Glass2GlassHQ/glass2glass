# Developer tooling

The `g2g` developer workflow is consolidated behind a few tools: a `cargo xtask`
command crate, a pipeline visualizer, a caps-negotiation explainer, criterion
benchmarks with a CI regression guard, and an end-of-run telemetry report. This
page is the reference; the architecture notes live in
[design/tooling.md](design/tooling.md).

`g2g-launch` and `g2g-inspect` install from crates.io. The launcher options
below that need a build feature say which, and that feature goes on the install
line:

```sh
cargo install g2g-plugins --features std,tooling-json,observe,tui --bin g2g-launch --bin g2g-inspect
```

A later `cargo install` replaces the binaries with the new feature set, so list
every feature you want in one install (add `wayland-sink` for a window, see the
README's system dependencies). Commands marked *(clone)* need a checkout of this
repository.

## When a pipeline fails

Start from what you see. Each case below shows the tool that explains it and
links the longer section on that tool. The output comes from debug builds of
`g2g-launch` and `g2g-inspect` with `std,wayland-sink` (plus `tooling-json` for
`--validate-json`).

### A link fails to negotiate

```sh
g2g-launch "videotestsrc ! video/x-raw,width=100000 ! fakesink"
# Setting pipeline to PLAYING ...
# ERROR caps             no caps overlap between VideoTestSrc and CapsFilter
# ERROR caps               VideoTestSrc offers: video/x-raw,format=RGBA,width=320,height=240,framerate=30/1 | ... | (+196 more, G2G_CAPS_TRACE=trace lists them)
# ERROR caps               CapsFilter wants: video/x-raw,format=NV12,width=100000 | ... | (+1 more, G2G_CAPS_TRACE=trace lists them)
# ERROR caps               width: VideoTestSrc offers 320 | [1,16384], CapsFilter wants 100000
# ERROR caps               VideoTestSrc -> CapsFilter: video/x-raw,format=RGBA,width=320,height=240,framerate=30/1 | ...
# ERROR caps               CapsFilter -> FakeSink: (not narrowed yet)
# pipeline error: CapsMismatch
```

The first line names the two elements whose link has no caps in common.
`offers` lists the caps the upstream element can send and `wants` the caps the
downstream element takes, alternatives split by `|` in order of preference. The
field line names what keeps them apart: the source's alternatives carry width
320 or the range 1 to 16384, and the capsfilter asks for 100000. The last lines
show each link's caps where the solve stopped. More in the
[caps-negotiation explainer](#caps-negotiation-explainer).

`g2g-inspect` shows the same range on the element's output caps and on its
`width` property:

```sh
g2g-inspect videotestsrc
# ...
# Output caps:
#   video/x-raw,format=RGBA,width=[1,16384],height=[1,16384],framerate=[0/1,1000/1]
#   ...
# Element Properties:
#   ...
#   width: frame width in pixels
#     flags: readable, writable
#     Unsigned Integer. Range: 1 - 16384. Default: 320
```

For a script or an editor, `--validate-json` prints the failure as JSON on
stdout and exits with status 1 without running. `upstream` and `downstream` are
indices into `nodes`, and `fields` carries the field line
([more](#pipeline-visualizer-dot--svg)):

```sh
g2g-launch --validate-json "videotestsrc ! video/x-raw,width=100000 ! fakesink"
# {"failure":{"downstream":1,"downstream_caps":["video/x-raw,format=NV12,width=100000",...],
#  "fields":[{"downstream":["100000"],"field":"width","upstream":["320","[1,16384]"]}],"kind":"empty-link",
#  "upstream":0,"upstream_caps":["video/x-raw,format=RGBA,width=320,height=240,framerate=30/1",...]},
#  "nodes":[{"index":0,"name":"VideoTestSrc"},{"index":1,"name":"CapsFilter"},{"index":2,"name":"FakeSink"}],
#  "ok":false,"stage":"negotiate"}
```

### An element name is unknown

```sh
g2g-launch "videotestsrc num-buffers=10 ! vpxenc ! fakesink"
# parse error: unknown element: vpxenc
#   hint: `vpxenc` is a g2g element but is not compiled into this build; rebuild with `--features vpx`
```

The hint names the cargo feature. Add it to the `cargo install` line together
with the features you already use. `g2g-inspect --gst <name>` gives the same
verdict for any name, and for a GStreamer element g2g lacks it names the
replacement. A plain `g2g-inspect vpxenc` prints the same hint.
`g2g-inspect` with no argument lists the elements this build has
([more](#element-reference-g2g-inspect-and-the-web-page)). The README's
[feature-gated elements](README.md#feature-gated-elements) table lists each
element's feature and the system library it needs.

### The pipeline runs but shows nothing

```sh
g2g-launch "videotestsrc num-buffers=90 pattern=ball ! videoconvert ! autovideosink"
# WARN  Registry         autovideosink found no usable sink (cannot run here: waylandsink (no Wayland compositor on wayland-0: Could not find wayland compositor); not in this build: wgpusink, kmssink, metalvideosink), using fakevideosink: the pipeline runs but its output goes nowhere
# Setting pipeline to PLAYING ...
# pipeline run summary:
# ...
```

`autovideosink` takes the first display sink that is in the build and can
present here, else `fakevideosink`. The warning lists each sink it skipped. Under
"cannot run here", `waylandsink` is compiled in but found no compositor at
`$WAYLAND_DISPLAY` (`wayland-0` when unset), so run it inside a Wayland session.
Under "not in this build" are the sinks whose feature is off: `wgpu-present`
for `wgpusink`, `kms-sink` for `kmssink` on a console with no compositor,
`metal-sink` for `metalvideosink` on macOS. Install the README's
[system dependencies](README.md#system-dependencies) before adding the feature.

### It runs but stalls, stutters or drops frames

`G2G_DEBUG='*:debug'` logs every control packet and every place that drops
data, and `G2G_DEBUG='*:log'` adds a line per buffer, so the last element to log
a buffer is where a stall is ([packet flow logging](#packet-flow-logging)). A
display sink that drops frames for arriving after their deadline logs it under
`pacing`:

```sh
G2G_DEBUG=pacing:debug g2g-launch "videotestsrc num-buffers=60 width=1920 height=1080 ! videoconvert ! waylandsink"
# Setting pipeline to PLAYING ...
# DEBUG pacing           dropping late frames from pts 0:00:00.066666666, 0:00:00.718424041 past its deadline
# DEBUG pacing           presenting again at pts 0:00:02.833333305 after dropping 18 late frames
# ...
# pipeline run summary:
#   frames:  emitted 60, consumed 60, dropped on links 0 (0.0% drop)
#   ...
#   per-element [measured]:
#     VideoConvert0    proc p50 134.22 ms / p99 134.32 ms (n=60), blocked p50 0.00 ms / p99 173.49 ms, age-out p50 536.87 ms / p99 711.76 ms, in-fill 63%/100% avg/max
#     WaylandSink0     proc p50 0.01 ms / p99 784.99 ms (n=60), in-fill 44%/100% avg/max
#   present: WaylandSink0     5 presented, 0 dropped, 55 late-dropped
#   ...
```

In this debug build `videoconvert` takes 134 ms per 1080p frame against a 33 ms
frame period. The sink drops 55 of the 60 frames as late, and logs
`presenting again at pts ...` each time it catches up. `videotestsrc` skips
ahead on the sink's QoS reports, so the 60 frames span 5 seconds instead of 2.
`dropped on links` counts only frames a leaky link discarded. The
`present:` line counts the frames the sink discarded. The
[end-of-run report](#end-of-run-report) points at the slow stage: `proc` is the
time in `process()` and `age-out` the frame's age as it leaves the element.

To watch rates while the pipeline runs, `--tui` draws them in the terminal and
`--observe <port>` serves them to a browser
([terminal UI](#terminal-ui---tui), [live dashboard](#live-dashboard---observe)).
The element whose input link stays full is the slow one. To keep the traffic
for a replay, `--record-on-error <dir>` writes the last packets on every link
when the run fails, and `recordsink` records one link of any run
([flight recorder](#flight-recorder), [record / replay](#record--replay)).

### A custom element gets caps it can't handle

An element states the caps it handles through the constraint for its role
([AUTHORING.md §4](AUTHORING.md#4-caps-negotiation)). The default accepts
anything, so an element that does not override it receives whatever upstream
sends. With the constraint in place, a mismatch fails at startup with the ERROR
block above. The `invert` element of
[AUTHORING.md §1](AUTHORING.md#1-a-complete-element) takes RGBA video only, and
`cargo run` in that crate feeds it audio as its second pipeline
(`audiotestsrc num-buffers=1 ! invert ! fakesink`):

```
ERROR caps             no caps overlap between AudioTestSrc and Invert
ERROR caps               AudioTestSrc offers: audio/x-raw,format=S16LE,channels=2,rate=48000
ERROR caps               Invert wants: video/x-raw,format=RGBA
ERROR caps               media type: AudioTestSrc offers audio/x-raw, Invert wants video/x-raw
ERROR caps               AudioTestSrc -> Invert: audio/x-raw,format=S16LE,channels=2,rate=48000
ERROR caps               Invert -> FakeSink: (not narrowed yet)
```

## `cargo xtask` *(clone)*

`xtask` is a dependency-free command crate (a `.cargo/config.toml` alias onto the
`xtask/` workspace member). It only orchestrates `cargo` and toolchain tools.

```sh
cargo xtask <command>
```

| Command | What it does |
| :--- | :--- |
| `ci` | Runs locally what CI runs (workspace check / test / clippy, the Linux feature build, the embassy no-alloc tests, the wasm core check), `--locked` like CI, stopping at the first failure. A red CI reproduced offline. |
| `test --here` | Probes this host (NVIDIA via `nvidia-smi`; VAAPI / Opus / ALSA / Pulse / Wayland / ffmpeg via `pkg-config`; `/dev/video*` cameras; `/dev/dri` GPU nodes) and runs exactly the feature-gated tests it supports. `--dry-run` prints the detected plan without running. |
| `install-launch` | The meson-style "build everything this platform supports" install. Probes the host (same probes as `test --here`, plus OpenSSL / libvpx / PipeWire / clang) and `cargo install`s `g2g-launch` with every element that compiles here, reporting each skipped element and the dev package it needs (`[ ] vpx (skipped: needs libvpx-devel + clang)`). `--enable`/`--disable f[,g]` override the detected set, `--bin N` targets another binary, `--dry-run` just prints the plan. The static `linux-full` cargo feature is the "assume all deps present" bundle counterpart. |
| `size` | Builds the `examples/g2g-size` Cortex-M footprint harness and reports the gc-sectioned `.text` size (locating `rust-lld` in the toolchain sysroot). |
| `wasm` | Builds the wasm32 targets (core `runtime`, plugins `web` / `web-codecs`). |
| `bench` | Runs the criterion benchmarks (see below). Extra args pass through, e.g. `cargo xtask bench -- --save-baseline main`. |
| `ffi-probe` | Generates a C `sizeof`/`offsetof` probe for an SDK struct and emits the `repr(C)` size assert (see below). |
| `new-element <name> --kind source\|transform\|sink` | Stamps a new `g2g-plugins` element: the source file with the right `AsyncElement`/`SourceLoop` skeleton, a scaffold test, and the `pub mod` wiring in `lib.rs`; prints the `registry.rs` line to paste. The generated element compiles as-is; fill in the TODOs. |

The cross-compiling commands (`size`, `wasm`) prepend `~/.cargo/bin` to `PATH` so
cargo uses the rustup toolchain rather than a distro `rustc` that lacks the target
std; `wasm` adds `--cfg=web_sys_unstable_apis` for the `web-codecs` build.

## Declarative and scripted pipelines

Besides a `gst-launch` string, `g2g-launch` can build a graph from a file, and a
`scriptelement` can run per-frame Rhai logic. All go through the same registry /
negotiation as the text parser (see the README "Scripting" section for the full
tour). Feature gates: `declarative` (JSON) / `declarative-yaml` (adds YAML) /
`script-rhai` (Rhai builder + `scriptelement`); the `linux-full` bundle enables
the last two.

```sh
# A declarative JSON / YAML graph (nodes + edges, a { id, caps } capsfilter
# shorthand, a top-level `pipeline:` escape hatch):
g2g-launch --graph pipe.yaml          # .json -> from_json, .yaml/.yml -> from_yaml

# A Rhai script that BUILDS a graph (loops / params / conditionals via
# add / caps / set / link / link_leaky):
g2g-launch --script pipe.rhai

# Per-frame Rhai over the live buffer (zero-copy handle):
g2g-launch videotestsrc ! scriptelement script="fn process(f){ f.invert(); }" ! fakesink

# Script-decided routing to N outputs (each branch pulled from your own app):
g2g-launch videotestsrc ! scriptrouter name=r script="fn route(f){ f.sequence % 2 }" \
  r.0 ! appsink channel=even   r.1 ! appsink channel=odd
```

`scriptelement` performance: the script is the control plane, native code the
data plane. A per-pixel Rhai loop is interpreted (seconds/frame on HD, inherent to
any embedded scripting engine), so for whole-frame work call a native bulk op
(`invert()` / `fill(v)`) or express a per-value transform as a 256-entry
`apply_lut(lut)` (~1-2 ms/frame); reserve `frame[i]` for logic and small regions.
`--graph` / `--script` compose with `--dot`, `-v`, and `--threads` below.

## Pipeline visualizer (DOT / SVG)

The `GST_DEBUG_DUMP_DOT_DIR` analog. `g2g-launch --dot` parses a pipeline,
negotiates it (without running), and prints Graphviz DOT to stdout:

```sh
g2g-launch --dot videotestsrc num-buffers=1 ! videoconvert ! fakesink | dot -Tsvg -o pipe.svg
```

Nodes are role-coded (green sources, red sinks, blue transforms, diamond tees,
trapezium muxers). Each edge is labelled with the **negotiated caps**, its memory
domain (GPU / zero-copy links are drawn bold and labelled e.g. `memory:Cuda`),
its non-default `LinkPolicy`, and fan-out / fan-in pad indices. On a negotiation
failure it falls back to a topology-only dump.

For a quick text dump without Graphviz, `g2g-launch -v` prints the same per-link
**negotiated caps** + memory domain + `LinkPolicy` to stderr before running (the
gst `-v` analog). If negotiation fails it prints the topology-only wiring list
and the negotiation ERROR block, then exits without running:

```sh
g2g-launch -v videotestsrc ! videoconvert ! video/x-raw,format=NV12 ! fakesink
#   [0] VideoTestSrc -> VideoConvert : video/x-raw,format=RGBA,...  mem=System policy=Block
#   [1] VideoConvert -> CapsFilter   : video/x-raw,format=NV12,...  mem=System policy=Block
```

For a machine-readable form of the same thing, `g2g-launch --validate-json` prints
every node and each edge's negotiated caps as JSON (or, when the pipeline cannot
negotiate, every node and the structured failure: the node indices it names and
the caps involved), and exits without running. Needs the `tooling-json` build
feature
(`cargo install g2g-plugins --features tooling-json --bin g2g-launch`):

```sh
g2g-launch --validate-json videotestsrc ! videoconvert ! fakesink
# {"edges":[{"caps":"video/x-raw,format=RGBA,...","from":0,"to":1},...],
#  "nodes":[{"index":0,"name":"VideoTestSrc"},...],"ok":true}
```

Those caps are the solver's, chosen before any data flowed, so a stream whose
geometry only arrives with the data (a demuxed file, a parsed elementary stream)
shows the startup placeholder. `--run-json` runs the line to EOS and prints the
same shape with the caps each edge actually carried, tagging every edge
`runtime` (a `CapsChanged` crossed it) or `negotiated` (none did):

```sh
g2g-launch --run-json filesrc location=clip.h264 ! h264parse ! fakesink
# {"edges":[{"caps":"video/x-h264,width=16,height=16,...","caps_source":"negotiated","from":0,"to":1},
#           {"caps":"video/x-h264,width=320,height=240,...","caps_source":"runtime","from":1,"to":2}],
#  "nodes":[{"index":0,"name":"FileSrc0"},...],"ok":true}
```

In code, `Graph::to_dot` / `ValidatedGraph::to_dot` (in `g2g_core::dot`) render
any graph; `g2g_core::runtime::negotiate_graph` runs the caps solve without
running the pipeline and returns the per-edge caps + memory domains both the DOT
dump and `-v` use.

## Caps-negotiation explainer

Caps negotiation is the hardest code in the system; the explainer makes the
solver narrate itself. Turn it on with `G2G_CAPS_TRACE=1` (or
`G2G_DEBUG=caps:debug`):

```sh
G2G_CAPS_TRACE=1 g2g-launch videotestsrc num-buffers=1 ! videoconvert ! fakesink
```

It logs each node's constraint and, per edge, the surviving caps set and its
fixated result (`VideoTestSrc -> VideoConvert: ... ✓ -> video/x-raw,...`). The
narration is free when off (one atomic load). `G2G_CAPS_TRACE` accepts a level
name / number (`debug`, `trace`, `7`) to tune verbosity.

The mismatch half is on by default: a failed negotiation logs, at ERROR, the two
conflicting elements and the caps each wanted, with no env var, so a plain
`g2g-launch` run that fails to negotiate reads as a diagnosis rather than a bare
`CapsMismatch`. When a field keeps them apart (media type, format, width, height,
framerate, channels, rate), a line names it with both sides' values:

```sh
g2g-launch videotestsrc ! video/x-raw,width=100000,height=100 ! fakesink
# ERROR caps  no caps overlap between VideoTestSrc and CapsFilter
# ERROR caps    VideoTestSrc offers: video/x-raw,format=RGBA,width=320,... | (+196 more, G2G_CAPS_TRACE=trace lists them)
# ERROR caps    CapsFilter wants: video/x-raw,format=NV12,width=100000,height=100 | ...
# ERROR caps    width: VideoTestSrc offers 320 | [1,16384], CapsFilter wants 100000
# ERROR caps    VideoTestSrc -> CapsFilter: video/x-raw,format=RGBA,width=320,... | ...
# ERROR caps    CapsFilter -> FakeSink: (not narrowed yet)
```

A set past four alternatives is cut short unless `G2G_CAPS_TRACE=trace`.
`G2G_CAPS_TRACE` only adds the successful-edge narration on top.

`G2G_DEBUG` is the general `GST_DEBUG` analog: `G2G_DEBUG=*:debug` or
`G2G_DEBUG=videoscale:trace,*:warn` set per-category (element-type) thresholds.
Without it, errors and warnings print, and `G2G_DEBUG=error` keeps only errors.

## Packet flow logging

The runner logs every packet it hands to an element, under that element's
category and instance name, the way `GST_DEBUG=GST_SCHEDULING:6,GST_EVENT:6`
does. A source's pushes log under the source.

- `*:debug` shows every control packet (`segment`, `caps`, `flush`, `eos`), plus
  the places that drop data: a sink's late frames, a demuxer's seek, a source
  skipping to its first keyframe, a jitter buffer declaring packets lost.
- `*:log` adds one line per buffer: pts, duration, size and memory domain, and
  per-buffer drops such as a decoder clipping to the segment.

```sh
G2G_DEBUG='*:log' g2g-launch videotestsrc num-buffers=2 ! videoconvert ! fakesink
# DEBUG VideoTestSrc     <VideoTestSrc0> pushing event segment: rate 1, start 0:00:00.000000000, stop 99:99:99.999999999, base 0:00:00.000000000, time 0:00:00.000000000
# LOG   VideoTestSrc     <VideoTestSrc0> pushing buffer pts 0:00:00.000000000, dur 0:00:00.033333333, size 307200, domain System
# LOG   VideoTestSrc     <VideoTestSrc0> pushing buffer pts 0:00:00.033333333, dur 0:00:00.033333333, size 307200, domain System
# DEBUG VideoTestSrc     <VideoTestSrc0> pushing event eos
# LOG   VideoConvert     <VideoConvert0> have buffer pts 0:00:00.000000000, dur 0:00:00.033333333, size 307200, domain System
# DEBUG FakeSink         <FakeSink0> have event caps: video/x-raw,format=RGBA,width=320,height=240,framerate=30/1
# LOG   FakeSink         <FakeSink0> have buffer pts 0:00:00.000000000, dur 0:00:00.033333333, size 307200, domain System
# DEBUG FakeSink         <FakeSink0> have event eos
```

Narrow it like `GST_DEBUG`: `G2G_DEBUG='FakeSink:log,*:warn'` follows one
element. With the category off a packet costs one atomic load, and the run loop
still allocates nothing.

Drops by helpers that serve many elements log under the helper's category:
`pacing` (a synchronizing sink's late drops and segment clips), `demuxseek` (a
demuxer seeking), `audiodecoder` (decoded audio clipped to the segment),
`rtpjitterbuffer` and `aggregator`.

## Benchmarks

The criterion benchmarks live in `g2g-bench`, a standalone crate **excluded from
the workspace** (criterion pulls plotters / rayon that would otherwise bloat
every `--all-targets` CI job). They guard the latency moat's hot paths:

| Bench | Covers |
| :--- | :--- |
| `caps` | The caps algebra (`intersect` / `fixate`) and the linear / DAG solvers. |
| `convert` | The per-pixel software frame conversion (RGBA ↔ NV12 / I420) at 1080p. |
| `runner` | The bounded per-edge channel, the runner loop's inner transport. |

*(clone)*

```sh
cargo xtask bench                              # all of them
cargo xtask bench -- --bench caps              # one
cargo xtask bench -- --save-baseline main      # criterion args pass through
```

**Regression guard.** A dedicated [`bench` workflow](.github/workflows/bench.yml),
separate from the main CI so criterion never slows the check / test / clippy
jobs, runs on commits that touch the benched crates. It benches the new commit
and its parent (the PR base, or the previous commit on a master push) and fails
if any benchmark's mean regressed more than 50%, a loose threshold tuned to
shared-runner noise so it catches a lost fast path or an accidental O(n²) rather
than drift. The comparison is `bench_compare.py`.

## FFI struct probe

`xtask ffi-probe` automates the hand-rolled-FFI ritual (the `repr(C)` + size
assert convention used by `cuda.rs` / `nvenc.rs`): it generates a C program that
includes a header and prints `sizeof` of a struct plus `offsetof` of each field,
compiles and runs it, then emits the assert to paste beside the transcription.

*(clone)*

```sh
cargo xtask ffi-probe --header ffnvcodec/nvEncodeAPI.h \
  --struct NV_ENC_INITIALIZE_PARAMS --field encodeGUID -I /usr/include/ffnvcodec
# ...
# sizeof(NV_ENC_INITIALIZE_PARAMS) = 1800
# const _: () = assert!(core::mem::size_of::<NV_ENC_INITIALIZE_PARAMS>() == 1800);
```

A wrong layout (e.g. after an SDK version bump that resizes a struct) then fails
the build, not the GPU.

## End-of-run report

`g2g-launch` prints a telemetry summary at the end of a run: frame counts + drop
rate, the aggregated declared latency window, the elected clock, the negotiated
head allocation, and the measured wall-clock throughput.

```
pipeline run summary:
  frames:  emitted 20, consumed 20, dropped on links 0 (0.0% drop)
  latency: 0.0 ms .. 0.0 ms (non-live) [declared]
  clock:   SystemFallback (base 136891 ns)
  run:     0.03 s wall, 601.8 fps
```

In code, `RunStats::report()` formats the same summary from any run's stats.
Live per-element telemetry is available while the run is in flight via the
dashboard below; the end-of-run summary above is the declared-latency fold plus
the measured throughput.

## Live dashboard (`--observe`)

`g2g-launch --observe <port>` runs the pipeline while serving a live dashboard on
that port. Build with the `observe` feature:

```sh
cargo install g2g-plugins --features observe --bin g2g-launch
g2g-launch --observe 8787 videotestsrc ! videoscale ! fakesink
# open http://127.0.0.1:8787 (wait for the "dashboard: ..." line first)

# or, in a clone:
cargo run -p g2g-plugins --features observe --bin g2g-launch -- \
  --observe 8787 videotestsrc ! videoscale ! fakesink
```

By default the dashboard binds loopback (`127.0.0.1`), so open it on the same
machine. To reach it from another host, bind all interfaces with
`--observe-host 0.0.0.0` and open `http://<machine-ip>:8787`. The dashboard has
no auth and its telemetry / edge previews expose frame content, so only bind
non-loopback on a trusted network.

The page (`g2g-plugins/src/dashboard.html`, self-contained, no build step) draws the
graph left-to-right, colors each node by input-link fill, shows per-element
`process()` p50 / p99 and frame counts, and logs bus events (state changes, QoS,
buffering, negotiation failures, EOS, errors) as they arrive.

Each edge is labelled with its negotiated caps (hover for the full gst-string).
Scroll to zoom, drag the background to pan, and use the fit button (or
double-click) to frame the whole graph, so a large pipeline stays navigable.

The sidebar shows a per-stage latency waterfall: each element's input-link wait
(measured queue-residency, p50) stacked with its `process()` cost (p50), so you
see where a frame's time goes. The end-of-run summary prints the same as
`wait p50/p99` beside `proc`, plus `age-out p50/p99`: the frame's total age
(source arrival to that element's output push). `age-out` is the column that
catches an element buffering frames internally, a decoder's reorder queue reads
as cheap `proc` yet an `age-out` several frame-periods high.

Click any edge to tap its content: the dashboard streams a live preview of the
packets crossing it, a downscaled thumbnail for RGBA/BGRA and planar NV12/I420
video (and MJPEG keyframes when built with the `mjpeg` feature), a codec card for
other compressed edges (codec, resolution, this packet's frame type where cheaply
parseable, and its size), a min/max waveform for PCM audio, or a bounded hexdump
otherwise. Sampling runs a few times a second on a copy and never blocks the data
path; click again to stop.

Underneath, a `g2g_core::runtime::Observer` shares the running graph's topology
and per-element probes; `run_graph_observed` registers them, so a snapshot is a
handful of relaxed atomic loads off the same probes the end-of-run report reads
(no hot-path cost). One TCP port carries both concerns: a plain `GET /` returns
the page, a WebSocket upgrade gets a JSON `telemetry` snapshot every 250 ms plus
one `event` per bus message. The JSON is built in `g2g_plugins::dashboard`
(serde_json), keeping `g2g-core` serde-free.

Today's tap covers the cooperative graph runner and the two linear runners
(per-element `process()` + input fill); the fan-in / fan-out / muxer runners,
per-link transit time, source-side timing, and the threaded runner are
follow-ups (see `design/TODO.md`).

## Terminal UI (`--tui`)

`g2g-launch --tui` draws the same live telemetry in the terminal, with no
browser and no port. Build with the `tui` feature:

```sh
cargo install g2g-plugins --features tui --bin g2g-launch
g2g-launch --tui videotestsrc ! videoscale ! fakesink

# or, in a clone:
cargo run -p g2g-plugins --features tui --bin g2g-launch -- \
  --tui videotestsrc ! videoscale ! fakesink
```

Two screens, `g` toggles: a table view (per-element `process()` p50 / p99, input
wait, push-back, input fill; per-link caps, packets, bytes, drops; the
single-frame journey as a wait / work / blocked bar per stage) and an ASCII
topology view (one box per element with its live p99, arrows following the
links, each labelled with packets crossed and `-N` frames dropped, boxes colored
by peak input fill). Arrow keys pan the topology when it is wider than the
terminal. `q`, Esc, or Ctrl-C quits, which stops the pipeline and restores the
terminal.

It reads the same `Observer` tap the dashboard does, in-process, at 250 ms; it
needs neither `observe` nor JSON. Log records would tear up the drawing, so
while the UI holds the terminal, g2g logging is diverted from stderr into a ring
buffer and shown as a log pane (`G2G_DEBUG` still selects what is recorded); the
stderr sink comes back on exit.

## Per-frame trace (`--trace`)

`g2g-launch --trace <file>` writes a span for every frame each element handles
to a Chrome trace JSON file, the `GST_TRACERS` analog. Open the file in
<https://ui.perfetto.dev>. `g2g-launch-py` takes the same flag.

```sh
g2g-launch --trace run.json videotestsrc num-buffers=100 ! videoconvert ! fakesink
```

Each element is one process in the Perfetto timeline. Its `work` track has one
span per `process()` call, and its `wait` track has the time the frame sat
queued on the element's input link. Click a span for the frame's `sequence`
and `push_wait_ns`, the part of the call spent blocked on a full downstream link.
Sources, muxers and demuxers have no spans. It works with `--threads`, and does
not combine with `--observe` or `--tui`.

## Visual pipeline builder *(clone)*

`tools/builder/` is a React Flow app (Vite + pnpm) that lets you assemble a
pipeline visually: filter the palette, click to add nodes, drag a link from
either end (React Flow only allows valid out -> in), pan / zoom, and edit each
node's properties (typed from the registry). It live-exports two formats, both
loadable back into g2g:

- a `gst-launch` line (`g2g-launch "<line>"`); linear chains use the `!` form,
  branched graphs the named-definition + `elem.` reference form.
- declarative JSON (`g2g-launch --graph <file.json>`, the `declarative.rs`
  schema).

The reverse also works: paste a `gst-launch` line or a declarative JSON document
into the import box and it loads onto the canvas (auto-detected, or forced with
the format toggle), so you can start from an existing pipeline. An element the
loaded registry doesn't know becomes a flagged placeholder rather than being
dropped.

While you wire, links are checked live. With the wasm solver built (below) the
builder runs g2g's *real* caps negotiation client-side: each edge shows its
negotiated caps, and a link that fails to negotiate is drawn red with the reason.
Without the blob it falls back to a cheap family heuristic (raw video into an
audio input, etc.); the footer says which is active. Neither blocks a link.

Run it (needs Node + pnpm):

```sh
cd tools/builder
pnpm install
./build-wasm.sh     # optional: the real caps solver (needs wasm-pack + wasm32);
                    # without it the builder uses the family heuristic
pnpm dev            # dev server with live reload, prints a localhost URL
# or a static self-contained bundle:
pnpm build && python3 -m http.server -d dist 8099   # open http://localhost:8099
```

The solver is a small dedicated crate (`g2g-validate-wasm`, not the heavy
`g2g-web`) wrapping `toolingjson::validate_json`; `build-wasm.sh` compiles it into
`src/wasm/` (gitignored, built on demand). A Vite plugin embeds the blob as
base64 and instantiates it from bytes, so `pnpm dev`, the static `pnpm build`
bundle, and the self-contained artifact all run the real solver (no fetch, so it
works under the artifact's strict CSP); build `build-wasm.sh` before `pnpm build`
to include it, else the build falls back to the heuristic. Rebuild it after
changing the core solver, parser, or registry.

A "dynamic" palette group adds `uridecodebin` (with a `uri` prop) and
`decodebin`. These are parse-time autoplug macros, not registered elements, so
they only round-trip through the `gst-launch` export, not the declarative JSON
(the builder warns when a dynamic node is present in JSON mode).

The rest of the palette comes from `g2g-inspect --json`; the checked-in
`public/registry.json` is a snapshot of the standard registry, fetched at
runtime. Regenerate it after adding or changing elements (needs the
`tooling-json` feature):

```sh
cargo run -p g2g-plugins --features tooling-json --bin g2g-inspect -- --json \
  > tools/builder/public/registry.json
```

## Record / replay

Capture a pipeline's packet stream to a file and play it back, for deterministic
repro of bugs that need a live source. Both are ordinary launch-line elements
(std build):

```sh
# record: caps + every frame crossing the sink's input go to a file
g2g-launch "rtspsrc location=rtsp://cam ! h264parse ! recordsink location=cam.g2g"

# replay: the file becomes a source (add sync=true to pace to the recorded PTS)
g2g-launch "replaysrc location=cam.g2g ! h264parse ! avdec_h264 ! autovideosink"
```

The round-trip is byte-identical (`replaysrc` re-emits the recorded caps and
frames exactly). The file is length-prefixed `g2g_core::wire` records, the same
codec the distributed-graph transports use.

## Flight recorder

`recordsink` needs to be in the pipeline before the run. When a live pipeline
fails after an hour, use the flight recorder instead: every link keeps a bounded
ring of its most recent packets, and a failed run writes each ring out as a
recording of the traffic that led there.

```sh
g2g-launch --record-on-error dumps/ "videotestsrc ! videoscale width=64 height=48 ! filesink location=/dev/full"
# on failure it prints one line per file:
#   recorded dumps/0-VideoTestSrc0-to-VideoScale0.g2grec
#   recorded dumps/1-VideoScale0-to-FileSink0.g2grec

g2g-launch "replaysrc location=dumps/1-VideoScale0-to-FileSink0.g2grec ! fakesink"
```

Files are named for the two elements the link joins, and hold the same records
`recordsink` writes, so any dump replays as a source. Retention per link is
`FLIGHT_RING_PACKETS` (60) packets or `FLIGHT_RING_BYTES` (8 MiB), whichever
binds first. Off by default and free when off. Works under `--threads`, and with
`--observe` / `--tui`. One limit: a link carrying GPU memory is skipped (device
frames do not serialize, so they cannot be replayed).

## MCP server (`g2g-mcp`)

`g2g-mcp` is a Model Context Protocol server (JSON-RPC over stdio) for
agent-driven dev. Build with `mcp` and point an MCP client at the
binary:

```sh
cargo install g2g-plugins --features mcp --bin g2g-mcp
claude mcp add g2g -- g2g-mcp

# or, in a clone:
cargo build -p g2g-plugins --features mcp --bin g2g-mcp
claude mcp add g2g -- "$PWD/target/debug/g2g-mcp"
```

Tools:

- Elements and lines: `list_elements`, `inspect`, `validate` (parse and
  negotiate, no run).
- One-shot runs with a deadline that report `RunStats`: `launch` for a launch
  line, `run_graph` for a JSON or YAML graph document.
- One managed pipeline that keeps running: `start_pipeline`, `pipeline_status`,
  `stop_pipeline`.
- Watching it: `tail_events` (bus errors, warnings, QoS, state changes),
  `tail_logs`, `set_log_level`, `sample_edge` (packet metadata on one link),
  `snapshot_frame` (the newest frame at an element, as PNG).
- Changing it while it runs: `get_property`, `set_property`,
  `validate_insertion`, `insert_transform`, `remove_transform`, `insert_chain`,
  `add_branch`, `remove_branch`, `add_input`.
- `metasink` records: `latest_metadata`, `wait_for_records`, `load_metadata`.
- Files: `clip_at` cuts the seconds around a pts out of a video.

The introspection tools use the shared `toolingjson` internals, the same code
behind `g2g-inspect --json`.

## Element reference (`g2g-inspect` and the web page)

Find the docs for a single element the way `gst-inspect` does:

```
g2g-inspect                     # list every element in this build
g2g-inspect videotestsrc        # role, caps / pad templates, and all properties
g2g-inspect --gst <name>        # what a GStreamer element name maps to in g2g
g2g-inspect --gst-map           # gst-name/g2g-runtime-name pairs, tab separated
```

Caps print as gst caps strings, one alternative per line, with ranges as
`[min,max]` (`video/x-raw,format=RGBA,width=[1,16384],...`).

`--gst-map` is for a tool comparing g2g's graph against GStreamer's: g2g names a
graph node after the Rust type (`NalParse0`) where gst names it after the factory
(`h264parse0`), so pairing the two engines' elements needs the synonyms. Only the
names that actually differ are printed.

The same facts are available as structured data (`Registry::describe(name)` /
`describe_all()` returning `ElementDoc` / `PropertyDoc`), which powers a
searchable web reference. `g2g-docgen` renders every element into a
self-contained `docs/elements.html` (one card each: role, klass, caps / pads,
properties; a client-side search box, shareable via `?q=<term>`):

```
# (clone)
cargo run -p g2g-plugins --features linux-full --bin g2g-docgen
# writes docs/elements.html (pass a path to write elsewhere)
```

The listing reflects the elements compiled into the build (a feature-gated or
platform-only element absent from the build is absent from the page), exactly
like `g2g-inspect`. The page is linked from the project landing page.

`g2g-launch-py` (crate `g2g-python`, feature `launch`) runs `g2g-inspect`,
`g2g-docgen` and `g2g-mcp` over a registry that also holds the hosted Python
elements `pyelement`, `pysrc` and `pyaggregator`: `--inspect [g2g-inspect
arguments]`, `--docgen [out.html]` and `--mcp`. Listing the hosted elements
does not start the interpreter, but the binary links libpython. The checked-in
page comes from this path, and CI regenerates it the same way *(clone)*:

```
cargo run -p g2g-python --features launch,g2g-plugins/linux-full --bin g2g-launch-py -- --docgen docs/elements.html
```
