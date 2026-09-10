# Developer tooling

Graph visualization, the negotiation explainer, `cargo xtask`, benchmarks, the
run telemetry and live dashboard, the JSON introspection surface, and the
conformance batteries that derive an element's maturity. Part of the design in
[DESIGN.md](DESIGN.md).

## DOT visualization

`g2g_core::dot` renders a pipeline graph as Graphviz DOT, the
`GST_DEBUG_DUMP_DOT_DIR` analog. `Graph::to_dot` before validation and
`ValidatedGraph::to_dot` after `finish` emit a `digraph { .. }` a developer
renders with `dot -Tsvg`. It is pure `no_std + alloc` string formatting with no
I/O, so it builds on every target the core does, embedded included.

The graph carries an opaque element payload `E`, so node display names come from
a caller-supplied `Fn(NodeId) -> Option<String>`, and returning `None` falls back
to the node's structural kind, the right answer for a `tee` or `mux` that carries
no element. Nodes are role-coded by shape and fill, boxes for source, sink and
transform, a diamond for `tee` and a trapezium for a muxer.

Edges are annotated from a `DotAnnotations { edge_caps, edge_memory }`, both
indexed by edge id, the same index `solve_graph` returns its `Vec<Caps>` solution
under and `ValidatedGraph::edge` uses. An edge shows its negotiated caps through
`Caps::to_gst_string`, a non-`System` memory domain drawn bold since a GPU or
zero-copy link is the interesting one, its non-default `LinkPolicy`, and fan-out
and fan-in pad indices.

`g2g-launch --dot` is the user-facing entry. It parses a pipeline against the
registry, dumps the DOT to stdout, and exits without running, labelling each node
by its element's `log_category`, the short type name such as `VideoTestSrc`, via
`GraphNodeRef::log_category`. To show the chosen caps it first calls
`negotiate_graph`, the explainer's seam of a Phase 1 source-caps probe plus a
Phase 2 solve without running the pipeline, which returns the per-edge fixated
caps and each edge's memory domain, the producing node's `output_memory`, that
the dump renders on the edges, marking GPU and zero-copy links bold. A
negotiation failure falls back to a topology-only dump.

It also runs the allocation cascade ([DESIGN-caps.md](DESIGN-caps.md)) before
reading those domains, since that is what settles a multi-domain producer on the
one its consumer asked for. Without it a decoder feeding a CPU sink still
reported its `Cuda` default and the dump called a downloading link a GPU link.
Because negotiation probes sources, a `--dot` of a live-ingress pipeline does
that source's `intercept_caps`, typically a connect, just as a run would.

Memory domain is a per-element declaration, `AsyncElement::output_memory` and
`SourceLoop::output_memory`, defaulting to `System` and overridden by GPU
producers like `NvDec`, and it is the runtime peer of the auto-plug
`ElementDesc::output_memory`. It is not part of `Caps`.

## Caps-negotiation explainer

Caps negotiation is the hardest code in the system, and a bare `CapsMismatch`
gives no hint why. The explainer makes the solver narrate itself. `solve_graph`
emits under a reserved `caps` log category, which is not an element type so it
filters independently: a setup dump of each node's constraint, then per edge the
surviving `CapsSet` and its fixated `Caps`. On failure it narrates at ERROR,
naming the two conflicting nodes and dumping the set on every edge incident to
them, so the log answers which two cannot agree and what each wanted. An edge
that survives narrowing but cannot reduce to one `Caps` logs `cannot fixate`.

Node labels come from the caller via `solve_graph_labeled`. The runner passes
each element's `log_category`, so the narration reads `h264parse -> nvdec`, while
`solve_graph` defaults to `n{id}:{kind}`. The narration is gated by the logging
framework: all formatting is skipped unless the `caps` category is enabled, which
costs one atomic load when off, so it is free in production. It is turned on with
`G2G_CAPS_TRACE=1`, a boolean shortcut or a level name or number to tune
verbosity, or the general `G2G_DEBUG=caps:debug`. Both install the stderr sink
through `log::init_from_env`, which the launch and inspect binaries already call
at startup.

## cargo xtask

`cargo xtask <command>`, a `.cargo/config.toml` alias onto the `xtask` workspace
member, is the home for the build and test invocations that would otherwise be
shell-history knowledge. It is dependency-free, orchestrating only `cargo` and
toolchain tools.

- `ci` runs locally what the GitHub workflow runs: workspace check, test and
  clippy, the Linux feature build, the embassy no-alloc tests and the wasm core
  check, `--locked` like CI, so a red CI is reproducible offline.
- `test --here` probes the host, `nvidia-smi`, `pkg-config` for the syslib-backed
  features, and the `/dev/video*` and `/dev/dri` device nodes, and runs exactly
  the feature-gated tests this machine supports. `--dry-run` prints the detected
  plan only.
- `size` builds the `examples/g2g-size` Cortex-M harness and reports the
  gc-sectioned `.text` footprint, locating `rust-lld` in the toolchain sysroot
  for the final link.
- `wasm` builds the wasm32 targets.
- `bench` runs the criterion benchmarks by manifest path, passing criterion args
  through, such as `--save-baseline`.
- `new-element <name> --kind source|transform|sink` stamps the boilerplate every
  new element repeats.

The cross-compiling commands, `size` and `wasm`, prepend `~/.cargo/bin` to `PATH`
so cargo selects the rustup toolchain over a distro `rustc` that lacks the target
std, and `wasm` passes `--cfg=web_sys_unstable_apis` for the `web-codecs` build.

`ffi-probe <header> <struct> [--field f]...` automates the hand-rolled-FFI ritual
the `cuda.rs` and `nvenc.rs` convention follows. It generates a C program that
includes the header and prints `sizeof` of the struct plus `offsetof` of each
field, compiles and runs it, and emits the
`const _: () = assert!(size_of::<Struct>() == N)` to paste alongside the
`#[repr(C)]` transcription. Layout is locked down before it is trusted, and an
SDK version bump that resizes a struct fails the build rather than the GPU.

`new-element` writes the `g2g-plugins` source file with the correct
`AsyncElement` or `SourceLoop` skeleton for the kind (`intercept_caps`,
`configure_pipeline`, `process` or `run`, with TODOs), a scaffold test, and the
`pub mod` wiring inserted into `lib.rs` alongside the unconditional module block.
It prints the `registry.rs` registration line to paste, since the registration
function is context-dependent. The generated element compiles as is.

## Benchmarks

The criterion benchmarks live in a standalone `g2g-bench` crate, excluded from
the workspace like `examples/g2g-size`, because criterion pulls plotters and
rayon that a `--all-targets` CI job would otherwise build on every push, and
Cargo's `required-features` does not gate a dev-dependency under `--all-targets`.

They guard the latency moat's hot paths: the caps algebra plus the linear and DAG
solvers (`benches/caps.rs`), the per-pixel software frame conversion
(`benches/convert.rs`), and the runner loop's bounded per-edge channel
(`benches/runner.rs`, the transport every frame crosses, since the full
`run_graph` paces to PTS and so is unsuitable for a microbench).

`tools/pushtax-bench.sh` prices the push model against GStreamer's pull on batch
demux: the same `filesrc ! tsdemux ! h264parse ! fakesink` line through
`g2g-launch` in release and `gst-launch-1.0` over an ffmpeg-authored 60 s 1080p30
TS, five interleaved timed runs each, results appended per iteration. Measured on
the dev host: 176 versus 1175 MB/s, a factor of 6.7. The script prints the g2g
per-element attribution under the ratio because the gap is element CPU rather
than transport. `TsDemux` at 0.26 ms p50 over 1414 chunks and `NalParse` at
0.13 ms p50 over 1800 access units account for essentially the whole wall clock
on the single-thread executor, so the per-chunk channel, wakeup and boxed-future
residual is small and a pull mode would not close the gap. Demux and parse
throughput would. `benches/runner.rs` already prices the bare channel.

A dedicated `bench` workflow, separate from the main CI so criterion never slows
the check, test and clippy jobs, runs on PRs that touch the benched crates. It
benches the PR head and its base and fails if any benchmark's mean regressed more
than 50%, a loose threshold tuned to shared-runner noise, catching a lost fast
path rather than drift.

## Run telemetry

`RunStats::report()` formats the end-of-run telemetry the runner already gathers:
frame counts and drop rate, the aggregated declared latency window from the
per-element `latency()` fold, the elected clock, and the head allocation, which
`g2g-launch` prints at end alongside the measured wall-clock throughput.

### Per-element measurement

Alongside the declared fold, the runner collects measured per-element telemetry
in `RunStats::per_element`, one `ElementLatency` row per interior element in
topological order. Each transform and sink arm holds an `Arc<ElementProbe>`
(`runtime/instrument.rs`). On every `DataFrame` it samples its input link's fill
through `LinkReceiver::fill_percent` and times the `process()` call wall-clock
with `metrics::monotonic_ns` around the `await`, recording into the lock-free
log2 `LatencyHistogram`, so the hot-path cost is a handful of relaxed atomics and
no allocation.

Once every arm has joined, the runner snapshots each probe into the report, and
`report()` prints a per-element `proc p50 / p99 (n) + in-fill avg/max` table, the
by-hand glass-to-glass analyses, the NVDEC-to-system-memory floor and
`link_capacity` dominance, turned into a number the runner emits.

The graph runner and the two linear runners, `run_simple_pipeline` and
`run_source_transform_sink`, collect it, and the dynamic fan-out, fan-in and
muxer-sink runners do too, since the `_observed` entry points register each arm's
node and edge on the observer incrementally as it attaches, so a late arm reports
like an initial one. The static session runners leave it empty, like their
declared latency.

It is `std`-gated where it needs a clock: the histogram is `no_std`, but with no
`monotonic_ns` the timing compiles out and the table is then empty, so the
`no_std` baseline pays nothing. Sources have no `process()` and so do not appear,
and their cost surfaces as the downstream element's input fill.

A paced display sink also reports what it actually put on screen. The element
overrides `presentation_stats()`, giving frames presented, frames overwritten
before paint under `DropOldest`, and frames shed by QoS late-drop, the graph
runner's sink arm stores it on the probe as the arm ends, and `report()` prints
one `present:` line per presenting sink. `frames_consumed` alone cannot
distinguish a healthy display from one silently shedding or stalling, and
`g2g-launch` divides the presented count by wall time into a presented-fps figure
next to the pipeline throughput.

The `process()` timing is the work half of a stage's latency. The wait half is
queue residency, added as measured per-link transit. When an observer is
attached, the graph runner builds `Block` edges into transform and sink arms with
a per-link transit ring (`link_with_transit`): the producer's `SenderSink` stamps
a monotonic send time as each `DataFrame` is queued, and the consuming arm pops
the stamp when it pulls the frame (`LinkReceiver::pop_transit_ns`), recording the
elapsed queue time into `ElementProbe::transit`. The ring stays aligned with the
data channel because `Block` links never drop, leaky edges being left plain so
their transit is simply not measured, and it is `Option`-gated so an
uninstrumented run carries no stamp and pays nothing. `RunStats::report()` prints
`wait p50/p99` beside `proc`, and the dashboard stacks the two per stage into a
latency waterfall.

### Edge probes

Every link also carries a per-edge content-inspection slot, `LinkSender::probe`,
a `ProbeSlot` the wrapping `SenderSink` shares, so a tool can install a
`LinkInterceptor` to sample the packets crossing any edge without touching the
arms. It is empty, pass-through and zero cost, unless a subscriber installs one.

The dashboard uses it for edge previews. Clicking an edge sends a `subscribe`
over the WebSocket, the server installs a rate-limited `PreviewTap` on that
edge's slot via `Observer::edge_probe` and `edge_caps`, and streams back a
`preview` message: a downscaled thumbnail for RGBA, BGRA and planar NV12 and I420
video, and MJPEG keyframes under the `mjpeg` feature reusing `videoconvert` and
`mjpegdec` rather than duplicating the conversion, a codec card for other
compressed edges carrying codec, resolution, header-parsed frame type and size
with no decode, PCM S16 waveform buckets, or a bounded hexdump
(`g2g-plugins::preview`). It is sampled a few times a second on a copy, never
blocking the data path.

### Live dashboard

The same probes drive a live view, not just the end-of-run table. An `Observer`
(`runtime/observe.rs`) captures the graph topology and holds clones of the arms'
probe `Arc`s, and `run_graph_observed` registers them during the prepare phase,
before any frame flows. Because the probes are the same lock-free atomics the
report reads, `Observer::snapshot` mid-run is a handful of relaxed loads and never
stalls an arm.

The transport lives in `g2g-plugins::dashboard` under the `observe` feature.
`g2g-launch --observe <port>` serves one TCP port that answers a plain `GET /`
with a self-contained dashboard page (`tools/dashboard/`) and a WebSocket upgrade
with a JSON `telemetry` snapshot every 250 ms plus one `event` per `BusMessage`,
fanned out to all clients via a broadcast channel drained off the `Bus`.

Each telemetry edge carries its negotiated caps, from the `Observer`'s per-edge
solution, and live counters, packets, CPU-payload bytes, drops, and `blocked_ns`,
the time producers spent awaiting link capacity, from a wait-free `EdgeCounters`
block the data-plane sink writes, which the page labels on the link. The page
pans and zooms so a large graph stays navigable.

Beside the aggregate per-stage waterfall the page assembles a single frame's
journey. Observed probes keep a bounded ring of `{sequence, wait, enter, exit}`
visits, joined at snapshot time along the linear prefix on the newest sequence id
consistent with one frame moving downstream, where restamping elements fail the
join rather than fabricate one and fan nodes truncate it, shown as stacked wait,
work and blocked bars with the end-to-end total against the
`2 * capacity * frame_period` floor. A journey stage's `work_ns` is compute and
`blocked_ns` is downstream backpressure, both drawn from the same push-wait bank
as the aggregate `push_wait` percentiles.

It binds loopback by default. `--observe-host <addr>`, such as `0.0.0.0`, exposes
it to other hosts, gated behind a no-auth warning since telemetry and edge
previews carry frame content. The JSON is built in the transport, so `g2g-core`
stays serde-free, consistent with the portability-core principle.

The observer rides the cooperative graph runner and, via
`run_graph_threaded_observed`, the threaded runner, and both cover the muxer and
demux fan nodes. The standalone hand-built fan-in, fan-out and session runners
(`fanin.rs`, `runner.rs`) have their own `*_observed` entry points, name and probe
their nodes like the graph runner, and fill `RunStats::per_element` even
unobserved. The dynamic runners, whose arms attach at runtime, still report no
per-element rows.

## JSON introspection and the builder

`g2g-inspect --json [element]` (the `tooling-json` feature) emits the registry as
JSON, the machine-readable sibling of the text dump: per element the identity,
role, output caps or pad templates, and each property's machine type, range,
default and read/write flags, from the same `ElementDoc` and `PropertyDoc`
introspection the text path uses. Like the dashboard it is serialized in
`g2g-plugins` with serde_json, not in `g2g-core`.

It feeds two consumers. The visual pipeline builder (`tools/builder/`) is a React
Flow app (Vite plus pnpm) that loads a `registry.json` snapshot, offers a typed
drag-drop canvas with pan and zoom and either-direction linking, and imports and
live-exports a `gst-launch` line, the `!` form for linear chains and named
definitions plus `elem.` references for branched graphs, and declarative JSON and
YAML on the `declarative.rs` schema, all of which load back into g2g. The other
consumer is the MCP server.

Links are validated live. With `g2g-validate-wasm` built, which is g2g's real
caps solver wrapping `toolingjson::validate_json`, compiled to wasm and loaded
client-side, each edge shows its negotiated caps and a failing link is flagged.
Without the blob it falls back to a coarse caps-family heuristic. A Vite plugin
embeds the wasm as base64 and instantiates it from bytes, so the solver runs in
`pnpm dev`, the static bundle, and the self-contained single-file artifact alike,
with no fetch and CSP-safe.

The builder is the one tool with a JS build step, source under `tools/builder/`
with `node_modules`, `dist` and `src/wasm` gitignored. Every other dev tool is a
Rust binary or a zero-build page.

## recordsink and replaysrc

`recordsink` and `replaysrc` (`std`-gated, in `g2g-plugins::record`) turn a live
stream into a file and back, for deterministic repro. `recordsink` writes the
negotiated caps from `configure_pipeline` then every `DataFrame` as
length-prefixed `g2g_core::wire` records. `replaysrc` reads the leading record as
its `intercept_caps` result and re-emits the caps and frames as a source,
optionally paced to the recorded PTS with `sync=true` or as fast as possible, the
default, for deterministic tests.

They are ordinary launch-line elements, `... ! recordsink location=x` and
`replaysrc location=x ! ...`, with no convenience flag, and the wire codec is
shared with the distributed-graph transports so there is one packet
serialization. A truncated trailing record, a recording cut off mid-write, is
dropped on replay rather than failing.

## g2g-mcp

`g2g-mcp` (the `tooling-json` feature) is a Model Context Protocol server so an
agent can drive g2g development. It speaks newline-delimited JSON-RPC 2.0 over
stdio with no MCP framework dependency, the envelope hand-rolled with serde_json,
and exposes five tools:

- `list_elements`
- `inspect(element)`
- `validate(pipeline)`, parse and negotiate with no run
- `launch(pipeline, duration_secs)`, run with a deadline and report `RunStats`
- `run_graph`, a declarative JSON or YAML document by path or inline, advertised
  only in `declarative` builds, with the same run conventions

Both run tools stream live telemetry while running when the client supplies a
`progressToken`, with periodic `notifications/progress` carrying the dashboard's
snapshot shape from `toolingjson::telemetry_json`, the single serializer both
consumers share. The tool bodies live in `g2g-plugins::toolingjson`, shared with
`g2g-inspect --json` so the registry-dump and run shapes have one definition, and
the async tools drive a current-thread tokio runtime via `block_on` while the
stdio loop stays synchronous.

The `validate` path returns a structured negotiation report, not just pass or
fail. `negotiate_graph` flattens a solve conflict to an opaque `CapsMismatch`,
while `negotiate_graph_explained`, its inner, returns `NegotiateError`, which
splits a setup failure (`Setup(G2gError)`, such as a source caps-probe I/O error)
from a solve conflict (`Solve(NegotiationFailure)`, the structured detail naming
the offending link).

`toolingjson::validate_json` reports, on success, the negotiated caps per edge
with the edge's endpoint node indices, and on a solve conflict the failure kind
(`empty-link`, `unfixable`, and the rest) plus those indices, so a caller can
highlight the failing link. On an `empty-link` the solver also captures both
candidate sets at the failing intersection (`CapsConflict`, upstream produce
against downstream accept, optional since some sites hold only one side), and
`validate_json` renders them as gst caps strings, `upstream_caps` and
`downstream_caps`.

`toolingjson::observed_graph_json` (`g2g-launch --run-json`) reports the same
graph after running it instead of before. Every link's `SenderSink` records the
last `CapsChanged` that entered it on the edge's `EdgeCounters`, so an `Observer`
snapshot carries both the solved caps and the observed ones
(`EdgeInfo::observed_caps`). The dump prefers the observed reading and tags each
edge `caps_source` as `runtime` or `negotiated`, which is what makes a
placeholder-then-refine stream, a demuxed file, comparable against an engine that
only reports post-run caps.

## Conformance and derived maturity

Because g2g grows fast under agent-driven development, "how validated is this
element?" has to be answerable without trusting a hand-written label, which under
fast iteration drifts into an overclaim, a maturity bumped in the same change
that adds the feature.

`conformance` (`g2g-core`, pure) makes maturity a derived value. An element's
`MaturityRecord` is a bag of `Evidence`, each tagging one `ConformanceDimension`
(`Instantiate`, `Properties`, `RoundTrip`, `LossResilience`, `ZeroCopy`,
`Latency`, `Oracle`, `Hardware`) that a check actually verified, plus the
platform, codec or peer it verified against. `MaturityRecord::level()` derives a
`MaturityLevel` (`Unverified` below `Instantiated` below `UnitTested` below
`InteropTested` below `HardwareValidated`) from that bag with no setter, and with
honesty guards: `Oracle` reaches `InteropTested` only with a named peer, and
`Hardware` reaches `HardwareValidated` only with a named platform.

So the absence of evidence is the signal. A loopback-only element carries no
`Oracle` evidence and stays `UnitTested`, which is the not-interop-validated
caveat expressed as data rather than prose.

The conformance batteries (`g2g-plugins::conformance`) exercise a real element,
never a mock, with cheap in-process checks and add evidence only on a pass, so
the level is computed from behaviour observed this run rather than asserted, and
a regression that breaks a round-trip drops the level. They cover the sans-IO
cores several transports share: the ST 2110-20 and -30 packetizer pairs including
the -7 seamless merge through per-path loss, the RFC 6184 H.264 payload core
(`rtph264`, FU-A fragmentation reassembled byte-exact, and a dropped fragment
costing only its own access unit rather than welding two together), and the RTP
jitter buffer (`rtpjitter`, reordered arrival released in sequence order, a hole
reported for NACK then skipped rather than stalling). `g2g-inspect --maturity`
runs the battery live and renders the matrix.

`Oracle` and `Hardware` evidence, which the in-process battery cannot produce
since it has no ffmpeg or GPU, comes from the resource-owning integration tests.
They append it to a tab-separated evidence log (`persist::record_evidence`, path
`$G2G_CONFORMANCE_LOG`) when a check passes, and `full_report` folds that log into
the in-process report so `--maturity` shows the `InteropTested` and
`HardwareValidated` rows too.

The native-muxer oracles mux an `Mp4MuxN` fMP4 or a `TsMux` transport stream and
have `ffprobe` demux them back, recording peer-tagged `Oracle` evidence deriving
`mp4mux` and `mpegtsmux` as `InteropTested`. The ffmpeg-interop transports carry
this further: `udpsrc` over RTP, `rtmpsrc`, `srtsrc` and `srtsink` over libsrt
including the AES variants, and both RTSP directions, `rtspserversink` played by
ffmpeg and `rtspserversrc` published into by ffmpeg over UDP and
TCP-interleaved, each derive `InteropTested` against a named reference peer.

The Vulkan Video decode tests persist GPU-tagged `Hardware` evidence via
`VulkanVideoDevice::device_name`, so `vulkanvideo` derives `HardwareValidated`
across H.264, H.265 and AV1. The rest of the GPU stack persists the same tier from
the tests that own the device: the native NVIDIA codecs, `nvenc` encoding a
CUDA-resident surface and `nvdec` decoding into one and downloading for a
System-only sink, and the `cudawgpu` bridge tag their evidence with the CUDA
device the driver names (`persist::cuda_platform_tag`, sourced from the GPU device
provider rather than hardcoded), while the dma-buf export pair, `wgputodmabuf` and
`dmabuftowgpu`, tags the subsystem, since each element opens its own
high-performance Vulkan adapter and so cannot honestly name which card ran it.

A CI `conformance` job runs the deterministic ffprobe oracles plus the
best-effort transport interop against a real ffmpeg, aggregating into one
`$G2G_CONFORMANCE_LOG`, where the muxer oracles honour an externally-set log so
they append rather than truncate, and publishing `--maturity` to the job summary.
The GPU `Hardware` rows come from a self-hosted GPU runner.

Together with the copy plan in [DESIGN.md](DESIGN.md), this is the
validation-first posture: the framework states hard, checkable properties, that
this graph is zero-copy or that this element is unit-tested but not
interop-validated, rather than leaving them to prose and trust.

## Codec goldens and PSNR

The conformance dimensions above say whether data survived an element. For a
codec that is not enough: a decoder that starts producing different pixels after
a dependency bump, or an encoder that quietly stops applying its bitrate, still
round-trips frames of the right size and shape.

`ConformanceDimension::Quality` is the dimension for the pixels and samples
themselves, and it counts as behavioural evidence, so a `Quality`-only element
derives `UnitTested`, and with a peer-tagged `Oracle` alongside it,
`InteropTested`. Its measurement helpers live next to the batteries in
`g2g-plugins::conformance`, dependency-free like the rest: `fnv1a_64`, a stable
digest for a committed golden, `i420_planes`, and `psnr_db` and `pooled_psnr_db`,
per-plane and sample-count-pooled peak signal-to-noise ratio, infinite for
identical input and `std`-gated only because `no_std` has no `log10`.

Three battery kinds produce that evidence, in `g2g-plugins/tests/m1001_*`.

The decoder goldens decode a committed fixture with the in-repo decoder and hash
the raw output against a value recorded in the test: `rav1ddec` over
`av1_640x480.obu`, `mjpegdec` over the two 16x16 JPEGs, `opusdec` and `vorbisdec`
over their Ogg fixtures, and behind the `ffmpeg` feature `ffmpegdec` over
`h264_640x480.h264`. Each of those codecs decodes bit-exactly by its
specification, so a mismatch means g2g changed rather than the reference moving.
AAC is the exception, since libavcodec decodes it in float and is not bit-exact
across versions, so its leg checks determinism and frame alignment instead of a
digest.

The encode and decode PSNR batteries encode a synthetic source generated in-test,
a gradient with a checkerboard and a walking bar so there is both smooth and
hard-edged content, and decode it back with the matching in-repo decoder,
requiring the pooled PSNR and the worst single plane to clear a per-codec floor
set a few dB under the figure observed when the battery was written. They cover
AV1 (`av1enc` and `rav1ddec`), MJPEG (`mjpegenc` and `mjpegdec`, measured in
packed RGBA because through I420 the pair converts colorspace twice and the score
stops tracking encode quality), and H.264 (`ffmpegenc` and `ffmpegdec` at a
bitrate that keeps libx264 off its quality ceiling).

The reference-decoder oracle closes the loop the goldens cannot. It has the
ffmpeg CLI decode the same fixture and measures g2g's decode against it, so a
pass is evidence about correctness rather than stability, and it persists a
peer-tagged `Oracle` row next to the `Quality` one. AV1 must agree sample for
sample there, and JPEG agrees to within each decoder's own IDCT and colorspace
rounding. It self-skips where ffmpeg is absent, like the muxer oracles.

The batteries are codec-feature-gated, so unlike the always-on ST 2110 and RTP
batteries they cannot run inside `g2g-inspect --maturity`. They persist their rows
to `$G2G_CONFORMANCE_LOG` and `full_report` folds them in, the same path the
`Hardware` rows take. CI runs the goldens and the PSNR floors in the Linux feature
job and the ffmpeg oracle in the conformance job.
