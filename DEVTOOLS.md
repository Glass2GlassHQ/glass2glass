# Developer tooling

The `g2g` developer workflow is consolidated behind a few tools: a `cargo xtask`
command crate, a pipeline visualizer, a caps-negotiation explainer, criterion
benchmarks with a CI regression guard, and an end-of-run telemetry report. This
page is the reference; the architecture notes live in
[DESIGN.md §4.20](DESIGN.md).

## `cargo xtask`

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
gst `-v` analog), falling back to a topology-only wiring list if negotiation
fails:

```sh
g2g-launch -v videotestsrc ! videoconvert ! video/x-raw,format=NV12 ! fakesink
#   [0] VideoTestSrc -> VideoConvert : video/x-raw,format=RGBA,...  mem=System policy=Block
#   [1] VideoConvert -> CapsFilter   : video/x-raw,format=NV12,...  mem=System policy=Block
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
fixated result (`VideoTestSrc -> VideoConvert: ... ✓ -> video/x-raw,...`). On a
mismatch it logs, at ERROR, the two conflicting elements and the caps each
wanted, so a `CapsMismatch` is a readable log rather than a guess. The narration
is free when off (one atomic load). `G2G_CAPS_TRACE` accepts a level name /
number (`debug`, `trace`, `7`) to tune verbosity.

`G2G_DEBUG` is the general `GST_DEBUG` analog: `G2G_DEBUG=*:debug` or
`G2G_DEBUG=videoscale:trace,*:warn` set per-category (element-type) thresholds.

## Benchmarks

The criterion benchmarks live in `g2g-bench`, a standalone crate **excluded from
the workspace** (criterion pulls plotters / rayon that would otherwise bloat
every `--all-targets` CI job). They guard the latency moat's hot paths:

| Bench | Covers |
| :--- | :--- |
| `caps` | The caps algebra (`intersect` / `fixate`) and the linear / DAG solvers. |
| `convert` | The per-pixel software frame conversion (RGBA ↔ NV12 / I420) at 1080p. |
| `runner` | The bounded per-edge channel, the runner loop's inner transport. |

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
  frames:  emitted 20, consumed 20, dropped 0 (0.0% drop)
  latency: 0.0 ms .. 0.0 ms (non-live) [declared]
  clock:   SystemFallback (base 136891 ns)
  run:     0.03 s wall, 601.8 fps
```

In code, `RunStats::report()` formats the same summary from any run's stats.
(Measured per-element / per-link p50 / p99 + channel fill-level needs frame-level
instrumentation in the runner and is a follow-up; the latency above is the
chain's *declared* fold.)

## Element reference (`g2g-inspect` and the web page)

Find the docs for a single element the way `gst-inspect` does:

```
g2g-inspect                     # list every element in this build
g2g-inspect videotestsrc        # role, caps / pad templates, and all properties
g2g-inspect --gst <name>        # what a GStreamer element name maps to in g2g
```

The same facts are available as structured data (`Registry::describe(name)` /
`describe_all()` returning `ElementDoc` / `PropertyDoc`), which powers a
searchable web reference. `g2g-docgen` renders every element into a
self-contained `docs/elements.html` (one card each: role, klass, caps / pads,
properties; a client-side search box, shareable via `?q=<term>`):

```
cargo run -p g2g-plugins --features linux-full --bin g2g-docgen
# writes docs/elements.html (pass a path to write elsewhere)
```

The listing reflects the elements compiled into the build (a feature-gated or
platform-only element absent from the build is absent from the page), exactly
like `g2g-inspect`. The page is linked from the project landing page.
