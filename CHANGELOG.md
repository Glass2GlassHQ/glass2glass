# Changelog

Pre-release. Work is tracked by milestone (Mn) following the roadmap in `DESIGN.md` §4.10.
Nothing is published yet; all versions are `0.1.0`.

## Unreleased

### M75: `AacParse` ADTS header parser

- The audio sibling of `H264Parse` / `H265Parse`: `AacParse` scans each access
  unit for an ADTS header (12-bit `0xFFF` syncword), recovers the channel count
  (from `channel_configuration`) and sample rate (from `sampling_frequency_index`),
  and emits a refining `CapsChanged` before forwarding the frame, so a raw ADTS
  AAC elementary stream can be restreamed or muxed with concrete channel/rate
  caps. We already decode AAC (`MfAacDecode`, M36); this closes the parse half.
  Pure CPU `no_std` baseline (no feature gate), native + wasm32.
- `Caps::Audio` has no open (`Any`) field, so the negotiated constraint is
  `IdentityAny` (forward whatever AAC the upstream produces) rather than the
  video parsers' `Identity(any geometry)`; a source advertising AAC before the
  first header lands uses sentinel `channels`/`sample_rate` 0, and the parser
  resolves the real values mid-stream. The AAC-only guard lives in
  `intercept_caps`. The ADTS header is plain bit fields (no exp-Golomb, no
  emulation prevention), so unlike the H.264 / H.265 parsers this needs none of
  the `annexb` machinery.
- Scope is ADTS, the common elementary-stream framing. LATM / LOAS (the
  MPEG-TS / broadcast framing) is deferred, as is synthesizing the
  AudioSpecificConfig for a downstream decoder (no per-frame side channel exists
  until the metadata system lands).
- Tests: thirteen (seven parser unit tests, incl. stereo/44100 and mono/48000
  recovery, `channel_configuration` 7 -> 8 channels, reserved sampling-index and
  channel-config-0 rejection, syncword scan past leading bytes, non-ADTS / empty
  rejection; plus six element-level tests driving `AacParse::process` through a
  recording sink: `CapsChanged` before the first frame, no re-emit on identical
  params, re-emit on a channel/rate change, non-AAC intercept rejection, the
  `IdentityAny` constraint). The synthetic-fixture approach matches the video
  parsers (no in-tree AAC source feeds it yet); validation against a real ADTS
  stream is owed. VERIFIED on the dev host: `cargo test -p g2g-plugins --lib`
  green (109, incl. the 13 new); `cargo clippy -p g2g-plugins --lib` clean;
  `cargo check -p g2g-plugins --target thumbv7em-none-eabihf` and `--target
  wasm32-unknown-unknown` green (stays in the no_std baseline); `cargo test
  --workspace` green (no regression).

### M74: `run_linear_chain` as a thin builder + topology-derived rejection policy

- `run_linear_chain` is now a thin builder: it constructs a borrowing
  `Graph<GraphNodeRef>` (source -> transform* -> sink) and delegates to
  `run_graph`, which owns negotiation, the M12 stat folds, the β allocation
  re-cascade, and the Caps-α mid-stream re-solve. Its ~290-line bespoke data
  plane (the N-hop coordinator wiring, the per-arm `select2` loops, the inline
  latency/clock/allocation folds) is deleted.
- `run_graph` gained a topology-derived mid-stream rejection policy: a node on a
  single-producer chain reverse-reconfigures and keeps flowing on a rejected
  `CapsChanged` (posting the structured failure to the bus), while a node behind
  a tee fails the run loud (the `run_source_fanout` strict default, since a
  shared upstream can't honor a per-branch reconfigure). This reconciles the
  linear runner's graceful behavior with the fan-out runner's strict one in a
  single runner, keyed on whether a tee sits upstream (`behind_tee`).
- `coordinator_with_recascade_n` (coordinator) and `downstream_feasibility`
  (solver) are now test-only: `run_linear_chain` was their last production user,
  and `run_graph`'s `GraphCoordinator` + edge-indexed `graph_downstream_feasibility`
  supersede them. Their tests still pin the linear-form behavior.
- Tests: `m18_beta_nhop` (β N-hop re-cascade), `m18_caps_resolve` (Caps-α
  mid-stream steering), `m18_dyn_latency_clock` (interior-element latency/clock
  folds), and `m18_multi_element` all pass unmodified through the rebuilt
  wrapper. VERIFIED: `cargo test --workspace` green; `cargo test -p g2g-core
  --features "std runtime"` green (149); `cargo clippy -p g2g-core --features
  "std runtime" --all-targets` clean; `cargo check -p g2g-core --target
  thumbv7em-none-eabihf` green.

### M73: legacy-bridge support in `solve_graph` + `run_muxer_sink` as a thin builder

- `solve_graph` now accepts the `Legacy*` migration-bridge constraints, not just
  the native variants, so a graph containing un-migrated elements solves: a
  `LegacySource` narrows its output like `Produces(one)`, a `LegacyTransform`
  forwards `intercept(input)` once the input fixates (the same single-caps
  forward cascade `solve_legacy_cascade` runs), and a `LegacySink` / legacy muxer
  input pad imposes no narrowing (the terminal accept the runner configures
  directly, as `run_muxer_sink` did). Previously a legacy sink hit
  `EndpointShapeMismatch`. This unblocks the D5 wrapper conversions.
- `run_muxer_sink` is now a thin builder: it constructs a borrowing
  `Graph<GraphNodeRef>` (muxer fan-in node + sources + sink) and delegates to
  `run_graph`, which owns negotiation, the per-input forwarders, the single
  merged Eos, and the MX-1 / MX-2 mid-stream re-solve. The bespoke fan-in data
  plane (`solve_mux_input` / `solve_mux_output` and the inline arms) is deleted.
  `run_graph` gained a `pub(crate) run_graph_inner` with an optional bus for the
  `_with_bus` variant (posts a startup negotiation failure).
- Tests: two solver unit tests (native muxer + `LegacySink` solves; a
  `LegacyTransform` forwards). Existing `m10_muxer` (its `CollectingSink` uses
  the default `LegacySink` bridge), `m18_mux_phase_c` (MX-1 / MX-2), and
  `m69_dag_muxer` pass unmodified through the rebuilt wrapper. VERIFIED: `cargo
  test --workspace` green; `cargo test -p g2g-core --features "std runtime"`
  green (149); `cargo clippy -p g2g-core --features "std runtime" --all-targets`
  clean; `cargo check -p g2g-core --target thumbv7em-none-eabihf` green.

### M72: lifetime-generic graph runner (borrowing graphs)

- `run_graph`'s payload is now `GraphNodeRef<'a>` (boxes hold `+ 'a` elements);
  `GraphNode` is a `'static` alias, so existing `Graph<GraphNode>` callers are
  unchanged. New `source_ref` / `element_ref` / `muxer_ref` constructors box a
  borrowed `&'a mut dyn Dyn{SourceLoop,AsyncElement,MultiInputElement}` via new
  forwarding impls (`impl Dyn… for &mut dyn Dyn…`), so a graph can borrow its
  elements instead of owning them. `run_graph` is now `run_graph<'a>` with arms
  `BoxFuture<'a>`. This is the D5 prerequisite: the convenience wrappers, which
  take `&mut` element references, can build a borrowing `Graph` and delegate to
  `run_graph` while the caller keeps its elements.
- `DynSourceLoop` gained the stat mirrors in M71; this adds the borrowed-element
  forwarding impls for all three dyn element traits.
- Test: `m72_dag_borrowed.rs` builds a `Graph<GraphNodeRef>` from `&mut` borrows
  (`src -> flip -> sink`), runs it, and reads the borrowed `FakeSink` afterward,
  proving the caller retains ownership. VERIFIED: `cargo test --workspace` green;
  `cargo clippy -p g2g-core --features "std runtime" --all-targets` clean; `cargo
  check -p g2g-core --target thumbv7em-none-eabihf` green.

### M71: DAG runner - M12 stat folds + muxer MX-1/MX-2 in `run_graph`

- `run_graph` now reports the same M12 stats the linear runners do, so the
  convenience wrappers can reduce to thin builders over it (D5 prerequisite). It
  folds latency (`LatencyReport::aggregate` over every element node), elects the
  pipeline clock (`elect_clock` over each node's `provide_clock`, with the elected
  epoch as `base_time_ns`), and runs the M12 allocation cascade in reverse topo
  order: each element absorbs the proposal on its output edge(s) and proposes from
  its output-link caps; a tee joins its branch proposals (most-demanding), a
  muxer is a boundary, and the source's absorbed proposal is the reported
  `allocation`. For a linear chain this is byte-for-byte the linear runner's
  sink->source fold.
- `DynSourceLoop` gains dyn-safe `latency` / `provide_clock` / `configure_allocation`
  mirrors (the source-side analog of `DynAsyncElement`'s), so the boxed sources in
  a `Graph` contribute to those folds.
- The graph muxer arm now runs the full `run_muxer_sink` mid-stream behavior:
  MX-1 (re-solve the changed input pad against its constraint and reconfigure it,
  consuming the input-side `CapsChanged` instead of leaking it to the output) and
  MX-2 (re-derive the merged output and emit one downstream `CapsChanged` only
  when it actually changed), replacing D4's simpler per-pad configure.
- Tests: `m70_dag_recascade.rs` updated to assert the startup allocation cascade
  now configures each branch transform (so a mid-stream change records
  `[startup, β]`, no change records `[startup]`). VERIFIED on the dev host: `cargo
  test --workspace` green; `cargo test -p g2g-core --features "std runtime"` green
  (147); `cargo clippy -p g2g-core --features "std runtime" --all-targets` clean;
  `cargo check -p g2g-core --target thumbv7em-none-eabihf` green.

### M70: DAG runner D4 - mid-stream re-solve + β allocation re-cascade

- `run_graph` now handles a mid-stream `CapsChanged` over the whole DAG, not
  just locally per arm. At startup it snapshots each edge's downstream
  feasibility (`graph_downstream_feasibility`, the graph generalization of the
  linear reverse sweep: a transform passes its output feasibility back through
  `backward_feasible`, a tee intersects its branch feasibilities, a muxer takes
  each input pad's accept set). Each transform arm steers its forwarded output
  toward a downstream-acceptable shape (Caps-α via `resolve_forward_output`); a
  sink re-solves its input against its declared constraint.
- β allocation re-cascade: a node-keyed `GraphCoordinator` walks the sink's
  re-derived allocation proposal one hop upstream per reply via `in_edges`,
  resolving through structural tee nodes; a source or muxer terminates the walk.
  Each interior transform arm selects on an `ArmDirective` control channel
  alongside its data link (so a directive applies while parked on data),
  re-derives its own proposal, and reports it onward. Tee branches re-solve and
  re-cascade independently; on EOS each transform drains a tail-end directive
  still in flight. `RunStats::coordinator_events` now reports the real count.
- Strict default (matches `run_source_fanout`): a branch whose downstream
  positively rejects the mid-stream output fails the whole graph loud
  (`CapsMismatch`). Graceful per-branch drop and a forward coordinator re-solve
  walk are follow-ups. A muxer is a β allocation boundary: its inputs carry no
  per-pad re-cascade channel, so the proposal stops there (its inputs still
  re-configure per pad on a mid-stream change).
- Tests: two solver unit tests (tee feasibility is the branch intersection,
  muxer input feasibility is per-pad) and `m70_dag_recascade.rs` (four pure-fake
  integration tests: a tee diamond re-cascades each branch independently, the
  no-change baseline records no β, a rejecting branch fails loud, a muxer
  re-configures only the changed input pad). VERIFIED on the dev host: `cargo
  test -p g2g-core --features "std runtime"` green (147); `cargo test
  --workspace` green (no regression, incl. the 4 new m70 tests); `cargo clippy
  -p g2g-core --features "std runtime" --all-targets` clean.

### M69: DAG runner fan-in - muxer nodes in `run_graph`

- Completes `run_graph`'s topology: it now drives source / transform / sink /
  tee (fan-out, M67) plus muxer (fan-in), so the canonical "split, process two
  ways, recombine" diamond runs end to end, not just fan-out. The arbitrary-DAG
  runner now covers linear + fan-out + fan-in + nested diamonds.
- Solver: `NodeConstraint::Muxer` generalizes from `{ inputs: Vec<CapsSet>,
  output: CapsSet }` to per-pad `{ inputs: Vec<CapsConstraint>, output:
  CapsConstraint }`, the shape real muxer elements expose
  (`MultiInputElement::caps_constraint_as_input` / `_for_output`). A wildcard
  (`AcceptsAny`) input pad imposes no narrowing (the interleave muxer forwards
  per-frame caps), an `Accepts(set)` pad narrows its edge, and the `Produces`
  output narrows the merged edge. This is the per-input-pad constraint API the
  DAG plan flagged as the prerequisite for wiring real muxers.
- Runner: `Graph::add_muxer` now carries the muxer element (a new
  `DynMultiInputElement`, the dyn-safe mirror of `MultiInputElement`), surfaced
  as `GraphNode::Muxer`. `run_graph` builds the muxer's `NodeConstraint` from
  the element, configures each input pad with its negotiated caps, and spawns
  the `run_muxer_sink` shape: one forwarder arm per input tags each packet with
  its pad into a single tagged channel that one muxer arm drains, combining via
  `process(pad, ..)` and emitting one merged `Eos` after every input ends. The
  per-input mid-stream re-solve (MX-1 / MX-2) stays D4.
- Tests: a new solver test (wildcard `AcceptsAny` input pads forward each
  source's caps, output takes the produced caps) and the M65 muxer test moved to
  the per-pad `CapsConstraint` API; plus `m69_dag_muxer.rs`, two pure-fake
  integration tests through the real runner (two unequal-length sources fan in
  to one sink with a single merged Eos; a `src -> tee(2) -> {flip, crop} -> mux
  -> sink` diamond recombines both branches). VERIFIED on the dev host: `cargo
  test -p g2g-core --features "std runtime"` green (145, incl. the new solver
  test); `cargo test -p g2g-plugins --features std` green (incl. the 2 new m69
  tests, no regression); `cargo clippy -p g2g-core --features "std runtime"
  --lib` + the m69 test clean; `cargo check -p g2g-core --target
  thumbv7em-none-eabihf` green (the solver/graph muxer constraint stays
  `no_std`; the runner is `std`-gated); native `cargo check --workspace` green.

### M68: `H265Parse` HEVC SPS parser

- The H.265 sibling of `H264Parse`: `H265Parse` scans each access unit for an
  SPS NAL (`nal_unit_type == 33`), recovers the coded picture dimensions, and
  emits a refining `CapsChanged` before forwarding the frame, so a raw H.265
  elementary stream (which advertises `Dim::Any` at negotiation until bytes
  flow) can be restreamed or recorded with concrete geometry. We already decode
  and contain H.265; this closes the parse half. Pure CPU `no_std` baseline (no
  feature gate), native + wasm32, mirroring `H264Parse`'s `Identity(H265 any)`
  constraint and mid-stream caps refinement.
- H.265 specifics handled: the 2-byte NAL header (type is bits `[1..7]` of the
  first byte), and `profile_tier_level` before the dimensions, a fixed 96-bit
  block for a single-layer stream (`sps_max_sub_layers_minus1 == 0`) plus the
  per-sub-layer blocks when present. Dimensions apply the conformance-window
  crop scaled by `SubWidthC` / `SubHeightC`. Framerate from the VUI is not
  recovered yet (in H.265 the VUI sits past the PCM / ref-pic-set loops, too
  deep to reach safely without a real-stream reference), so caps carry
  `Rate::Any`, a documented follow-up.
- Refactor (DRY): the RBSP de-emulation (`strip_emulation_prevention`) and the
  exp-Golomb `BitReader`, previously private to `h264parse`, move to the shared
  `annexb` module (now "NAL splitting and RBSP bitstream helpers"); both parsers
  use the single copy. `BitReader` gains `skip_bits` for the fixed-size PTL. The
  `h264parse` test suite guards the move.
- Tests: eleven (six parser unit tests, incl. dimension recovery, conformance
  cropping `1920x1088 -> 1920x1080`, the 96-bit PTL skip landing exactly on the
  next field, length-prefixed framing, non-SPS / empty rejection; plus five
  element-level tests driving `H265Parse::process` through a recording sink:
  `CapsChanged` before the first frame, no re-emit on identical SPS, re-emit on
  a resolution change, non-H.265 intercept rejection, the `Identity` constraint
  shape). The synthetic-fixture approach matches `H264Parse`'s bar; validation
  against a real H.265 elementary stream is owed (as `H264Parse` was, later
  confirmed against retina's AVCC output). VERIFIED on the dev host: `cargo test
  -p g2g-plugins --lib` green (97, incl. the 11 new and the unchanged
  h264parse/annexb after the refactor); `cargo clippy -p g2g-plugins --lib`
  clean; `cargo check -p g2g-plugins --target thumbv7em-none-eabihf` and
  `--target wasm32-unknown-unknown` green (stays in the no_std baseline); native
  `cargo check --workspace` green.

### M67: DAG runner D3 - `run_graph`

- Opens D3 of the DAG runner (DESIGN_TODO "DAG runner - detailed plan"): a
  single `run_graph(graph, clock, link_capacity)` entry point drives an
  arbitrary DAG, collapsing the linear + fan-out runner shapes. It negotiates
  the whole graph at once via `solve_graph` (D2), configures each node, then
  spawns one arm per node over per-edge channels joined with `join_all`. The
  element payload is `GraphNode { Source(Box<dyn DynSourceLoop>) |
  Element(Box<dyn DynAsyncElement>) }` (sources and transforms/sinks are
  different traits), with `GraphNode::source` / `GraphNode::element` boxing
  helpers. `ValidatedGraph` gains `element` / `element_mut` accessors so the
  runner builds each node's constraint before taking the element into its arm.
- Scope is source / transform / sink / tee (the fan-out half). A tee broadcasts
  each packet to all branches; since `PipelinePacket` is not `Clone` (a
  GPU-resident frame owns a non-copyable handle), the broadcast deep-copies
  `System` frames via a `try_clone_packet` helper and fails loud
  (`UnsupportedDomain`) on a GPU domain. The mid-stream re-cascade (coordinator)
  is D4, so an interior arm handles a `CapsChanged` locally (configure +
  forward) without the downstream-feasibility steering or β allocation walk.
- Deferred, each with a clear failure: a muxer node is rejected (real-muxer
  fan-in needs the per-input-pad constraint API, a separate follow-up); a
  GPU-resident frame in a tee fails loud (a refcounted shareable frame is the
  zero-copy-tee follow-up); the `rtspsrc -> parse -> tee -> {decode -> wayland,
  mux -> mp4}` hardware integration test is owed a Linux run. D4 (mid-stream
  re-solve over the DAG) and D5 (reframe the existing runners as thin
  `run_graph` wrappers) are the remaining phases. `run_graph` lives in
  `runtime/graph_runner.rs` under the same `std` gate as the other runners; the
  `no_std` baseline is unaffected.
- Tests: two core unit tests (`try_clone_packet` deep-copies a `System` frame's
  bytes + timing into a distinct allocation; control packets clone) plus
  `m67_dag_run_graph.rs`, four pure-fake integration tests through the real
  runner (a linear `src -> flip -> sink`; a `tee(2)` fan-out where each of two
  sinks consumes all four frames; a tee whose two branches run independent
  `flip` / `crop` transforms; and an RGBA-source-into-NV12-filter graph that
  fails the whole-graph solve loud). VERIFIED on the dev host: `cargo test -p
  g2g-core --features "std runtime"` green (144, incl. the 2 new); `cargo test
  -p g2g-plugins --features std` green (incl. the 4 new m67 tests, no
  regression); `cargo clippy -p g2g-core --features "std runtime" --lib` + the
  m67 test clean; `cargo check -p g2g-core --target thumbv7em-none-eabihf` green
  (graph_runner is `std`-gated, the baseline stays `no_std`); native `cargo
  check --workspace` green.

### M66: `VideoFlip` software flip / rotate (Tier-1 A)

- `VideoFlip::new(method)` mirrors or rotates a raw frame by a fixed
  `FlipMethod` (`HorizontalMirror`, `VerticalMirror`, `Rotate90Cw`, `Rotate180`,
  `Rotate90Ccw`), the last open transform of Tier-1 Phase A alongside
  `VideoScale` (M55), `VideoRate` (M56), and `VideoCrop` (M62). Same format set
  (`Rgba8`/`Bgra8`/`Nv12`/`I420`), a per-plane coordinate remap with no
  resampling. The two 90-degree rotations transpose the frame and swap width and
  height; the mirrors and 180 preserve geometry. 4:2:0 needs even input dims
  (chroma is 2x2 subsampled); odd dims fail loud. CPU-only `no_std` baseline, no
  feature gate, native + wasm32.
- Negotiation mirrors the sibling transforms: a native `DerivedOutput(any
  supported raw -> same format/framerate, dims swapped for the 90-degree
  rotations and preserved otherwise)`; `configure` validates the 4:2:0 even-dim
  constraint where input dims are absolute. The coordinate remap is the pure
  `transform_plane` / `flip` pair, host-testable without the runner. NV12's UV
  plane remaps with `channels = 2` so each chroma pair moves as a unit.
- Tests: seven unit tests (`transform_plane` for all five methods on a square
  plane, the 90-degree dim swap on a 3x2 plane, RGBA pixel mirror, NV12 90-degree
  geometry + byte-total preservation, `DerivedOutput` dim-swap for rotation and
  dim-preserve for mirror, configure odd-4:2:0/compressed rejection) plus
  `m66_videoflip.rs`, which rotates an 8x4 RGBA `VideoTestSrc` 90 degrees CW
  through the real runner and asserts the swapped 4x8 geometry + preserved
  framerate via `CapsChanged`. VERIFIED on the dev host: `cargo test -p
  g2g-plugins --lib videoflip` (7) and `--test m66_videoflip` (1) green; `cargo
  clippy -p g2g-plugins --lib` + the m66 test clean; `cargo check -p g2g-plugins
  --target wasm32-unknown-unknown` and `--target thumbv7em-none-eabihf` green
  (stays in the no_std baseline); native `cargo check --workspace` green.

### M65: DAG runner D2 muxer fan-in (`NodeConstraint`)

- Completes D2's fan-in half (M64 deferred it). `solve_graph` now handles
  muxer nodes via a new `NodeConstraint`: `Element(CapsConstraint)` for
  source/transform/sink (and the ignored tee slot), and `Muxer { inputs,
  output }` for fan-in, where `inputs[i]` is the accept set for input pad `i`
  and `output` is the produce set. A muxer narrows each input edge by its
  pad's accept set and its single output edge by the produce set, mapping each
  edge to its pad via the edge's `dst.index`. The M64
  `NegotiationFailure::UnsupportedNode` placeholder is removed (muxers are
  supported now), and `solve_graph`'s signature moves from `&[CapsConstraint]`
  to `&[NodeConstraint]`.
- D2 is now complete: source/transform/sink/tee fan-out plus muxer fan-in.
  The single-`CapsConstraint`-per-element model didn't fit a muxer (per-input
  pad constraints + an output), which is why the multi-input shape lives in
  `NodeConstraint`. The D3 `run_graph` runner is the remaining DAG phase.
- Tests: the M64 muxer-reject test becomes a fan-in success test (two video
  sources, input pad 0 accepts H264 and pad 1 H265, output produces a muxed
  stream; asserts each input edge is narrowed by its own pad and the output by
  the produce set); the other three D2 tests move to the `NodeConstraint` API.
  VERIFIED on the dev host: `cargo test -p g2g-core --features runtime --lib`
  green (140, incl. the rewritten muxer test); `cargo clippy -p g2g-core
  --features runtime --lib` clean; `cargo check -p g2g-core --features runtime
  --target thumbv7em-none-eabihf` green (stays `no_std`); native `cargo check
  --workspace` green.

### M64: DAG runner D2 - `solve_graph` (topological CSP)

- Generalizes `solve_linear`'s arc-consistency sweep (M16) to a topological
  sweep over a D1 `ValidatedGraph`: each edge is a link variable narrowed by
  the constraints of the nodes at both ends, swept forward in topo order and
  backward in reverse to a fixed point, then fixated to one `Caps` per edge.
  Per node kind: a source's `Produces` narrows its out-edge, a sink's
  `Accepts` / `AcceptsAny` its in-edge, a transform's `Identity` / `Mapping` /
  `DerivedOutput` / `IdentityAny` narrows its in/out edges (the edge-indexed
  analog of the linear solver's `apply_constraint`), and a tee fans its input
  caps out to every output unchanged (couples the in-edge equal to all
  out-edges). `EmptyLink` / `Unfixable` carry node ids in the DAG context.
- Scope is D2's fan-out half (source/transform/sink/tee). Fan-in muxers need
  per-input-pad constraints (a separate item), so a graph containing one
  yields the new `NegotiationFailure::UnsupportedNode`; that and the D3 runner
  are the follow-ups. Stays `no_std` (lives in `runtime/solver.rs` beside
  `solve_linear`, which the embedded path uses).
- Tests: four (a linear chain solves byte-for-byte identically to
  `solve_linear`; a tee fan-out couples every branch to the source caps; an
  incompatible branch fails the whole solve strictly; a muxer yields
  `UnsupportedNode`). VERIFIED on the dev host: `cargo test -p g2g-core
  --features runtime --lib` green (140, incl. the 4 new); `cargo clippy -p
  g2g-core --features runtime --lib` clean; `cargo check -p g2g-core --features
  runtime --target thumbv7em-none-eabihf` green (stays `no_std`); native `cargo
  check --workspace` green (the new variant breaks no match).

### M63: DAG runner D1 - `Graph` builder + validation

- Opens the DAG runner track (DESIGN_TODO "DAG runner - detailed plan", phase
  D1). `Graph<E>` builds an arbitrary multimedia DAG (linear + fan-out tee +
  fan-in muxer + nested branches) in one topology: `add_source` /
  `add_transform` / `add_sink` carry an element payload, `add_tee(n)` /
  `add_muxer(n)` are the runner shapes (no payload), and `link` / `link_with`
  connect output pads to input pads with a per-edge `LinkPolicy` (reusing
  `crate::link`). `finish()` validates and returns a `ValidatedGraph` carrying
  the topological order + per-node edge adjacency the solver (D2) and runner
  (D3) will consume.
- Generic over the element payload `E` so it stays `no_std` and carries no
  dependency on the std-gated runner: the runner will instantiate
  `Graph<Box<dyn DynAsyncElement>>`, embedded/wasm callers bring their own. It
  is a baseline module (no feature gate), usable from every target.
- Validation: every pad linked exactly once (`UnlinkedPad` /
  `PadCountMismatch`, since a pad peers with exactly one other, fan-out/in goes
  through tee/muxer), no cycles (Kahn topological sort -> `Cycle { nodes }`),
  orphan nodes rejected, and pad indices range-checked at link time
  (`PadOutOfRange` / `UnknownNode`). `NodeKind` fixes the pad counts (Source
  0/1, Transform 1/1, Sink 1/0, `Tee(n)` 1/n, `Muxer(n)` n/1). Scope is D1
  only: no solver (D2), no runner (D3).
- Tests: ten unit tests (linear-chain topo order, fan-out via tee, fan-in via
  muxer, tee->muxer diamond, cycle rejected, unlinked pad, double-linked pad,
  orphan node, pad-index out-of-range at link, `take_element` moves the payload
  once). VERIFIED on the dev host: `cargo test -p g2g-core --lib graph` (10)
  green; `cargo clippy -p g2g-core --lib` clean; `cargo check -p g2g-core
  --target thumbv7em-none-eabihf` green (stays `no_std`); native `cargo check
  --workspace` green.

### M62: `VideoCrop` software rectangular crop (Tier-1 A1)

- `VideoCrop::new(x, y, w, h)` extracts a sub-rectangle of a raw frame,
  preserving the pixel format, for ROI-driven flows (a detector emits boxes,
  the cropper extracts the patches a classifier sees). No resampling, a
  per-plane row copy. Completes the software-transform trio with `VideoScale`
  (M55) and `VideoRate` (M56), same format set (`Rgba8`/`Bgra8`/`Nv12`/`I420`).
  4:2:0 needs an even crop origin and size (chroma is 2x2 subsampled); odd
  coords fail loud, packed formats crop at any coords. CPU-only `no_std`
  baseline.
- Negotiation mirrors the sibling transforms: a native `DerivedOutput(any
  supported raw -> same format at the rect dims, framerate preserved)`, with an
  odd 4:2:0 rect collapsing to the empty set so the solve fails loud;
  `configure` additionally validates the rect lies inside the frame.
- Tests: six unit tests (`crop_plane` sub-rect copy, RGBA pixel extraction,
  NV12 per-plane sizes + luma offset, `DerivedOutput` rect mapping + odd-4:2:0
  rejection, configure fit + evenness) plus `m62_videocrop.rs`, which crops an
  8x8 RGBA `VideoTestSrc` to 4x4 at (2,2) through the real runner and asserts
  the cropped geometry + preserved framerate via `CapsChanged`. VERIFIED on the
  dev host: `cargo test -p g2g-plugins --lib videocrop` (6) and `--test
  m62_videocrop` (1) green; `cargo clippy -p g2g-plugins --lib` + the m62 test
  clean; `cargo check -p g2g-plugins --target wasm32-unknown-unknown` green.

### M61: native end-to-end ML pipeline test (cross-target proof, native half)

- An integration test runs `VideoTestSrc(RGBA) -> VideoConvert(NV12) ->
  VideoScale(NV12 4x4->2x2) -> WgpuPreprocess(GPU) -> OrtInference(tensor-input,
  identity fixture) -> FakeSink` through `run_linear_chain`, the native half of
  the cross-target story: the same element-graph shape the browser pipeline
  runs, substituting the platform source/decode/sink. It proves the software
  transforms compose with the GPU preprocess and the M59 tensor-input inference
  into one negotiated chain on real hardware: the caps solver threads
  `RGBA -> NV12 -> NV12@2x2 -> tensor[1,3,2,2] -> tensor-input` end to end, and
  three frames reach the inference output carrying the model's tensor caps.
- `g2g-plugins` is now a dev-dependency of `g2g-ml` (acyclic: g2g-plugins does
  not depend on g2g-ml), so the ML elements can be tested against the real
  source/transform/sink elements rather than fakes.
- VERIFIED on the dev host (D3D12 adapter): `cargo test -p g2g-ml --features
  "wgpu ort" --test native_ml_pipeline` green; `cargo clippy -p g2g-ml
  --features "wgpu ort" --tests` clean; the test skips gracefully when no wgpu
  adapter is present, like the other GPU-gated tests.

### M59: `OrtInference` tensor-input mode

- `OrtInference::with_tensor_input()` switches the input pad from `RawVideo`
  RGBA to `Caps::Tensor` (an already-normalized f32 NCHW `[1, 3, H, W]`), fed
  straight to the session with no CPU `/255` normalize. Closes the
  composability gap that blocked `WgpuPreprocess` / `WebGPUPreprocess` (both
  emit `Caps::Tensor`) from feeding inference directly: `OrtInference`
  previously only accepted RGBA and normalized internally, so the GPU
  preprocess output had nowhere to go. The native and browser demos both
  depend on this handoff. Default stays RGBA, so existing chains are
  unaffected.
- The session run + output extraction are factored into a shared `run_chw`,
  called by the RGBA path (`infer`, which normalizes first) and the new
  tensor path (`infer_tensor`, which reads the f32 values from the frame
  bytes). `supported_input` branches the negotiated input caps on the mode;
  `process` dispatches the two infer paths.
- Tests: a new `ort` integration test (`with_tensor_input` accepts the
  matching tensor caps, rejects RGBA, and an identity model returns the fed
  f32 NCHW tensor unchanged, proving no second normalization is applied).
  VERIFIED on the dev host: `cargo test -p g2g-ml --features ort --test
  ort_inference` green (3, incl. the new case, CPU EP); `cargo clippy -p
  g2g-ml --features ort --all-targets` clean; native `cargo check --workspace`
  green (`ortinfer` is `ort`-gated).

### M58: `WebCodecsDecode` GPU-resident output (P2.2)

- `WebCodecsDecode::with_gpu_output()` hands the decoded `VideoFrame` forward
  in the M57 `MemoryDomain::WebGPUExternalTexture` instead of copying it out to
  system RGBA, the load-bearing zero-copy step of the browser chain (the
  decoder hands the GPU surface forward rather than pulling it back). The
  default stays system RGBA, so the `CanvasSink` (M41) and `WebRtcSrc` (M42)
  paths are unaffected. The memory domain is not part of `Caps`, so negotiation
  is unchanged (`RawVideo` RGBA either way); the consumer pairs the matching
  domain at runtime (a `System`-only sink fed the GPU output fails loud with
  `UnsupportedDomain`).
- `VideoFrameOwner` wraps the frame as the `WebGPUKeepAlive`. On the GPU path
  the frame is NOT closed at hand-off: a VideoFrame-sourced external texture is
  valid only until the frame closes, so the owner closes it on drop, after
  downstream has imported and used it. `unsafe impl Send` under the
  single-threaded wasm contract (the `D3D11KeepAlive` / `MfDecode` precedent).
  The drain loop splits into `emit_system_rgba` / `emit_external_texture`
  sharing one `announce_caps`.
- VERIFIED on the dev host: `RUSTFLAGS=--cfg=web_sys_unstable_apis cargo check
  --target wasm32-unknown-unknown -p g2g-plugins --features web-codecs` green;
  wasm web-codecs clippy clean; native `cargo check --workspace` green (the
  module is wasm-gated). NOT verifiable on this Windows host: the actual
  GPU-resident decode + WebGPU import needs a browser; owed a browser run, like
  the rest of the wasm elements (M40-M42).

### M57: `WebGPUExternalTexture` memory domain + `WebGPUKeepAlive` (P2.1)

- Opens Phase 2 (the browser zero-copy chain): a new core `MemoryDomain`
  variant `WebGPUExternalTexture(OwnedWebGPUExternalTexture)` carrying a
  decoded browser `VideoFrame` to be imported into WebGPU as a
  `GPUExternalTexture` and sampled on the GPU, so a WebCodecs-decoded frame is
  preprocessed and run through inference without ever copying to CPU.
  Everything downstream in P2 (the GPU-resident decode output, the
  `WebGPUPreprocess` compute pass) depends on this carrier, so it lands first.
- Mirrors the existing `OwnedCudaBuffer`/`CudaKeepAlive` and
  `OwnedD3D11Texture`/`D3D11KeepAlive` pattern: `g2g-core` never links
  `web-sys`, so the producing element boxes the `VideoFrame` owner as
  `Box<dyn WebGPUKeepAlive>` and dropping it closes the frame. Two differences
  from the CUDA/D3D11 carriers: the payload (the `VideoFrame`) is a JS handle
  living inside the owner rather than a raw pointer in the struct, so
  `WebGPUKeepAlive` adds `as_any` for a consumer to downcast and recover the
  frame for `importExternalTexture`; and it keeps the `Send` supertrait so the
  enum stays `Send` and the carrier is native-testable, with the wasm element
  to assert `Send` under the single-threaded contract (the `MfDecode` /
  `D3D11KeepAlive` precedent). Re-exported from the crate prelude beside the
  sibling carriers.
- Tests: three core unit tests (`kind()` reports `WebGPUExternalTexture`;
  dropping the carrier closes the backing frame via the keep-alive;
  `as_any` downcasts to the concrete owner), reusing the `FlagOnDrop`
  stand-in. VERIFIED on the dev host: `cargo test -p g2g-core --lib memory`
  green (7, incl. the 3 new); `cargo check --workspace` green (the new variant
  breaks no exhaustive match); `cargo check -p g2g-core` and `-p g2g-plugins
  --target wasm32-unknown-unknown` green; `cargo clippy -p g2g-core --lib`
  clean. Design note for P2.3: research confirmed ORT-Web ignores a
  caller-supplied `env.webgpu.device` (issue #26107, open as of 1.26.x, source
  read), so the device handshake will invert (ORT creates the WebGPU device
  and the wgpu side adopts it) rather than sharing our device into ORT. The
  in-browser device adoption is owed a browser run.

### M56: `VideoRate` software temporal resampler (P1.2), completing Phase 1

- Second and last of the "first credible product path" P1 transforms
  (DESIGN_TODO.md): `VideoRate::new(target_fps)` drops or duplicates whole
  frames to hit a configured framerate (`30 -> 10` fps for ML inference, `30
  -> 60` for delivery), the temporal counterpart of `VideoScale`. It never
  touches pixels, so it is format-agnostic: preserves format and geometry,
  replaces only the framerate. `f64` target so fractional rates (29.97) work.
  CPU-only, `no_std` baseline, no feature gate, runs native + wasm32.
- Cadence follows GStreamer's `videorate`: hold the previous frame and, on
  each new frame, emit it for every output slot at least as close to the held
  frame as to the new one (nearest-neighbour, ties to the held frame so an
  on-grid input duplicates rather than drops). Output PTS is re-stamped onto
  the exact target grid; `saturating_add` survives a near-`u64::MAX` PTS; the
  held last frame is flushed once on EOS so the stream's final frame is not
  lost; `Flush` and a mid-stream geometry/format change reset the grid. The
  drop/duplicate decision is the pure `emit_slots` helper, host-testable
  without the runner. Negotiation is a native `DerivedOutput(any raw -> same
  format/geometry at the target rate)`.
- Tests: seven unit tests (downsample 1-in-3 grid spacing, upsample ~2x with
  monotonic PTS, near-`u64::MAX` PTS does not overflow, backward-PTS jump is a
  no-op, `f64` fps -> Q16 rounding incl. 29.97, configure rejects 0 fps and
  compressed input, `DerivedOutput` replaces only the framerate) plus
  `m56_videorate.rs`, which runs a 30 fps RGBA `VideoTestSrc` through the real
  runner at 10 fps (drops to 4 frames: 3 in-stream + the EOS-flushed last) and
  at 60 fps (duplicates past the 5 inputs), each announcing the new framerate
  via `CapsChanged`. VERIFIED on the dev host: `cargo test -p g2g-plugins
  --lib videorate` (7) and `--test m56_videorate` (2) green; `cargo clippy -p
  g2g-plugins --lib` + the m56 test clean; `cargo check -p g2g-plugins
  --target wasm32-unknown-unknown` green. With M55 this completes Phase 1 (the
  P1 software transforms), both green on native and wasm32.

### M55: `VideoScale` software spatial resampler (P1.1)

- First element of the "first credible product path" P1 transforms (DESIGN_TODO.md):
  `VideoScale::new(w, h)` resamples raw video to a configured output geometry,
  preserving the pixel format, so a stream's geometry can be fit to an ML model's
  fixed input size or a delivery resolution. Handles the same format set as
  `VideoConvert` (`Rgba8`, `Bgra8`, `Nv12`, `I420`): packed formats scale as one
  4-channel plane, the 4:2:0 formats scale luma and chroma independently at their
  own resolutions. CPU-only, integer-only (Q16 fixed-point bilinear, half-pixel-
  centred source mapping, no float intrinsics), `no_std`: lives in the plugin
  baseline with no feature gate, so it runs in the native and the wasm32 builds.
- Negotiation mirrors `VideoConvert`: a native `DerivedOutput(any supported raw ->
  same format at the configured target dims, framerate preserved)`. 4:2:0 needs
  even input and target dims (chroma is subsampled 2x2), rejected loud at
  negotiation; equal in/out dims short-circuit to an exact copy. Bilinear is the
  baseline-correctness choice; it undersamples on large downscale ratios, so a
  wgpu variant for GPU-resident input and higher-quality kernels is the deferred
  follow-up (DESIGN_TODO.md P1).
- Tests: eight unit tests (exact bilinear endpoint values on a 2->4 upscale,
  identity copy, per-format output byte sizes, NV12 per-plane chroma resample,
  `DerivedOutput` target mapping + odd-4:2:0-target rejection, configure even-dim
  rejection on both sides, and a smooth-gradient down/up round-trip with MSE < 65,
  i.e. PSNR > 30 dB) plus `m55_videoscale.rs`, which up- and down-scales an RGBA
  `VideoTestSrc` through the real source-transform-sink runner and asserts the
  target geometry + preserved framerate are announced via `CapsChanged`. VERIFIED
  on the dev host: `cargo test -p g2g-plugins --lib videoscale` (8) and `--test
  m55_videoscale` (2) green; `cargo clippy -p g2g-plugins --lib` + the m55 test
  clean; `cargo check -p g2g-plugins --target wasm32-unknown-unknown` green (stays
  in the no_std baseline). The planar math is unit-tested; NV12/I420 through the
  runner is not exercised end-to-end (`VideoTestSrc` emits RGBA), the runner path
  being format-agnostic.

### M54: `FfmpegH264Dec` accepts YUV444P source (chroma downsampled to 4:2:0)

- Closes the YUV444P half of the `ffmpegdec` "YUV444P / 10-bit" deferral: the
  decoder previously rejected any non-4:2:0 source with `CapsMismatch`, so a
  High 4:4:4 profile H.264 stream failed loud. `copy_yuv420` now also accepts
  `YUV444P` / `YUVJ444P` frames, box-averaging each full-resolution U/V plane
  down to 4:2:0 before packing the existing I420 or NV12 output. The reduction
  is lossy in chroma resolution but keeps the decoder's 4:2:0 output contract,
  so it needs no new `RawVideoFormat` variant or downstream change. 10-bit
  (`YUV420P10` / `P010`) stays deferred (its 16-bit sample layout is too
  endianness/bit-position-specific to add without a libav host to verify on).
- The chroma-downsample math is extracted to a pure, non-OS-gated `yuv` module
  (`downsample_chroma_420`, 2x2 box average with edge clamping for odd dims and
  source-pitch handling) so the algorithm is host-testable without libavcodec;
  the Linux-only `ffmpegdec` arms are reduced to extracting the frame planes and
  calling it. The module is gated `cfg(any(test, all(linux, ffmpeg)))`, the same
  shape as `h264util`, so it is never dead code off-Linux.
- VERIFIABILITY: the `ffmpeg` feature is Linux-only and target-gated, so the
  decoder arms were NOT compiled or run on this Windows dev host. What IS
  verified here: the `yuv` downsample math (4 host unit tests: even 2x2, 4x2,
  pitch-padded, odd-dim edge clamping), and that nothing off-Linux regressed.
  The actual 4:4:4 decode through libavcodec is owed a Linux + libav run (the
  new `ffmpegdec` plumbing mirrors the existing planar arms and uses the
  host-tested helper, but is unverified on this host). VERIFIED: `cargo test -p
  g2g-plugins --lib` green (58, incl. the 4 new `yuv` cases); `cargo check -p
  g2g-plugins --target thumbv7em-none-eabihf` green (`yuv` stays out of the
  no_std baseline); native `cargo check --workspace` + `cargo clippy --workspace
  --all-targets` + `cargo test --workspace` green.

### M53: CUDA execution provider for `OrtInference` (`cuda` feature)

- Second GPU inference EP on the `ort` path, the NVIDIA counterpart of M26's
  DirectML: a new g2g-ml `cuda` feature (implies `ort`, adds `ort/cuda`, which
  downloads a CUDA-enabled ONNX Runtime build) and
  `OrtInference::from_memory_with_cuda`, which registers the CUDA EP
  (`ep::CUDA::default().build()`) ahead of the CPU fallback. Registration is
  best-effort per ort's dispatch default: on a host without a usable CUDA device
  or runtime the session silently runs on the CPU, so the pipeline keeps flowing
  either way. The element shape is unchanged; the EP choice is a constructor
  variant. The `ep::CUDA` API + the `cuda = ["ort-sys/cuda"]` feature were
  checked against the fetched ort 2.0.0-rc.12 source, not assumed.
- This is the one deferred track buildable on the Windows dev host, so it is
  verified as far as the host allows. NOT verifiable here: that CUDA actually
  executed (this host has no NVIDIA CUDA runtime, so the EP falls back to CPU).
  The test confirms the EP wires, the CUDA-enabled runtime loads, registration
  succeeds best-effort, and inference runs byte-identically, not that a GPU ran
  it. A real CUDA run is owed to an NVIDIA box.
- Test (gated on `cuda`): the identity-model inference through
  `from_memory_with_cuda` produces byte-identical results to the CPU path,
  mirroring the DirectML test. VERIFIED on the dev host: `cargo test -p g2g-ml
  --features cuda --test ort_inference` green (3/3, incl. the CUDA case, on the
  downloaded CUDA-enabled runtime with CPU fallback); `cargo clippy -p g2g-ml
  --features cuda --all-targets` clean; `--features ort` (CPU) unregressed;
  native `cargo test --workspace` (default, cuda gated off) green.

### M52: `H264Parse` AVCC framing + VUI framerate

- Closes the AVCC + VUI items the `h264parse` module header had carried as
  "deferred to M7" since M6. `H264Parse` now accepts both Annex-B and AVCC
  (4-byte length-prefixed) access units, detected per buffer: `retina` emits
  AVCC by default, so the parser refined caps only when `RtspSrc` forced
  `FrameFormat::SIMPLE` before. The shared `annexb` module (M46) grows
  `is_annex_b`, an `AvccNals` length-prefixed iterator, and `nal_units_any`
  (picks framing, yields identical NALs); `H264Parse` routes its SPS scan
  through it and drops its own duplicate Annex-B iterator.
- The SPS parser now continues past frame cropping into the VUI and recovers the
  framerate from `timing_info` (`time_scale / (2 * num_units_in_tick)`, emitted
  as a Q16 `Rate::Fixed`), filling the `Rate::Any` placeholder the caps carried
  before. A truncated VUI leaves only the framerate unknown, never the
  dimensions; SPSes without VUI timing keep `Rate::Any`. `BitReader` gains a
  fixed-width `read_bits`.
- Tests: AVCC iteration matches Annex-B for the same NALs and stops on a
  truncated length (`annexb`); `H264Parse` parses an AVCC-framed SPS, recovers
  30 fps from a VUI fixture (with emulation-prevention bytes inserted so the
  32-bit timing fields round-trip), the direct VUI reader handles 29.97 fps Q16
  rounding and absent timing, and a process-level test drives an AVCC access
  unit to a refined `CapsChanged`. VERIFIED on the dev host: `cargo test -p
  g2g-plugins --lib` green (54, incl. the new h264parse/annexb cases); `cargo
  check -p g2g-plugins --target thumbv7em-none-eabihf` green (the parser stays
  no_std); native `cargo clippy --workspace --all-targets` + `cargo test
  --workspace` green.

### M51: pure-Rust Burn inference backend (`BurnInference`, `burn` feature)

- Stands up the §5.2 Burn backend, previously a no-op `burn` feature alias:
  `BurnInference` (`g2g-ml/src/burninfer.rs`) is the no-C++ counterpart of
  `OrtInference`, an `AsyncElement` negotiating `Caps::RawVideo` (RGBA) in and
  `Caps::Tensor` out, running the inference on burn's `wgpu` backend (any
  D3D12/Vulkan/Metal GPU, WebGPU on wasm). v1 is a single linear layer:
  `BurnInference::linear(w, h, weights, bias)` takes a row-major `[K, N]` matrix
  (`K = 3*w*h`) and `[N]` bias; each RGBA frame is normalized to a flat f32 NCHW
  RGB vector (`value / 255`, the same preprocessing `OrtInference` does) and run
  through `input . W + b`, emitting the `[1, N]` logits as an f32 tensor.
  Negotiation mirrors `OrtInference` (a native `DerivedOutput(RGBA@WxH ->
  tensor)`, geometry/format pinned at construction). Because burn drives the GPU
  runtime internally (cubecl), the element is fully synchronous, no async device
  handshake.
- Dependency: `burn` 0.21 (verified current, May 2026; API checked against the
  Burn Book + docs.rs, not assumed), `default-features = false` with `["std",
  "wgpu"]`. Module named `burninfer` so in-crate paths can't collide with the
  `burn` crate (as `ortinfer` does for `ort`). Deferred: ONNX import (burn-import
  is build-time codegen, not a runtime loader), the `Module` path with trained
  weights, and richer layers (conv); the caps/`AsyncElement` contract here is
  what they slot into.
- Tests: three unit tests (linear-layer dimension validation, intercept narrows
  RGBA / rejects NV12 and wrong geometry, configure rejects non-RGBA before
  touching the GPU) plus `burn_inference.rs`, which runs a known RGBA frame
  through the element on the real GPU and asserts the `[1, N]` logits match a CPU
  matmul of the same deterministic weights within float tolerance, the tensor
  caps emit once across two frames, and timing is inherited; it skips gracefully
  (catch_unwind probe) when burn's wgpu backend has no adapter. VERIFIED on the
  dev host (D3D12 adapter): `cargo test -p g2g-ml --features burn` green (3 unit +
  the GPU integration test); `cargo clippy -p g2g-ml --features burn
  --all-targets` clean; native `cargo check --workspace` + `cargo clippy
  --workspace --all-targets` + `cargo test --workspace` (default, burn gated off)
  green.

### M50: inline GPU tensor preprocessing (`WgpuPreprocess`), the §5.1 hardware-first pillar

- Realizes DESIGN.md §5.1 (Pillars 2 + 4: zero-copy hardware + ML): `WgpuPreprocess`
  (`g2g-ml/src/wgpupreprocess.rs`, `wgpu` feature) is an `AsyncElement` that turns
  an NV12 video frame into a normalized f32 NCHW RGB tensor
  (`Caps::RawVideo{Nv12} -> Caps::Tensor{F32,[1,3,H,W],Nchw}`), doing the BT.601
  colour conversion and the `value / 255` normalization in a wgpu compute shader
  instead of on the CPU. It produces the same tensor contract `OrtInference`
  builds on the CPU, so it slots into the existing tensor graph
  (`decoder(NV12) -> WgpuPreprocess -> TensorBatcher -> inference ->
  TensorPostprocess`). Negotiation mirrors `OrtInference`/`VideoConvert`: a native
  `DerivedOutput(NV12@even WxH -> tensor)`, non-NV12 input rejected at solve time.
- The NV12 bytes are uploaded to a storage buffer (packed `array<u32>`, unpacked
  in WGSL) and the f32 tensor is read back to `MemoryDomain::System` via a staging
  buffer + `poll(PollType::Wait)`. The GPU context (instance/adapter/device,
  pipeline, buffers sized to `W x H`) is built lazily on the first frame, since
  `request_adapter`/`request_device` are async and `configure_pipeline` is not.
  `wgpu` 29 (default backends; D3D12 on Windows). Deferred: the zero-copy path
  (binding a decoder's `DmaBuf`/`D3D11Texture` surface directly into the compute
  pass and emitting a GPU-resident tensor domain, which needs the surface-import
  handshake + a GPU tensor domain in core), RGBA input (normalize only), and
  offloading the blocking GPU round-trip to a blocking pool.
- Tests: three unit tests (the BT.601 host reference is grayscale-linear in luma,
  intercept narrows NV12 / rejects RGBA, configure rejects odd 4:2:0 geometry)
  plus `wgpu_preprocess.rs`, which runs a known NV12 frame (distinct luma, one
  neutral and one coloured chroma block) through the element on the real GPU and
  asserts the read-back tensor matches the host BT.601 reference within float
  tolerance, the tensor caps emit exactly once across two frames, and timing is
  inherited; it skips gracefully when no wgpu adapter is present. VERIFIED on the
  dev host (D3D12 adapter): `cargo test -p g2g-ml --features wgpu` green (3 unit +
  the GPU integration test); `cargo clippy -p g2g-ml --features wgpu --all-targets`
  clean; native `cargo check --workspace` + `cargo clippy --workspace
  --all-targets` + `cargo test --workspace` (default, wgpu gated off) green.

### M47: UDP egress sink (`UdpSink`), the I/O half of live egress

- Closes the encode->egress arc the M46 packetizer opened (DESIGN.md §4.12):
  `UdpSink` (`udpsink.rs`, `udp-egress` feature) is an `AsyncElement` sink that
  drives the sans-IO `RtpH264Packetizer` over each Annex-B access unit and sends
  the RTP packets to a destination on a tokio `UdpSocket`, the send-side inverse
  of `RtspSrc`'s receive path. The RTP timestamp is the 90 kHz image of
  `FrameTiming::pts_ns`; sequence numbers and the per-AU marker bit come from the
  packetizer. Accepts H.264 at any geometry (`Accepts`/intercept narrow to
  H.264, raw video rejected); `with_rtp(pt, ssrc)` and `with_max_payload(mtu)`
  configure the flow.
- The socket is bound synchronously in `configure_pipeline` (fails loud there,
  no runtime needed) and wrapped into the tokio socket lazily on the first
  `process`, where a runtime context is guaranteed (`UdpSocket::from_std`
  requires one). `Flush` does not reset the RTP sequence (a receiver tracks loss
  by gaps, so the numbering continues across a seek); `Eos` is recorded but emits
  no RTP end marker. Deferred (user-side, need the sandbox-blocked port 554):
  RTCP sender reports and the RTSP `ANNOUNCE`/`RECORD` handshake for Wowza-style
  ingest.
- Tests: three `udpsink` unit tests (intercept narrows H.264 / rejects raw,
  configure rejects non-H.264 before binding a socket, the 90 kHz pts->RTP
  timestamp conversion) plus `m47_udp_egress.rs`, which binds a loopback
  receiver, runs two access units (small NALs + an oversized NAL forcing FU-A)
  through the sink, and parses the datagrams back: the datagrams match the
  packetizer byte-for-byte, the RTP timestamp tracks `pts_ns` at 90 kHz
  independently (0 and 2999 for pts 0 and 1/30 s), sequence is contiguous, the
  marker lands on each AU's last packet, and the FU-A fragments reassemble the
  IDR NAL byte-exactly. Loopback UDP is used because RTP port 554 is
  sandbox-blocked (§4.11.4). VERIFIED on the dev host: `cargo test -p g2g-plugins
  --features udp-egress` green (71 lib incl. the new 3, plus the m47 integration
  test); `cargo clippy -p g2g-plugins --features udp-egress --all-targets` clean;
  native `cargo check --workspace` + `cargo clippy --workspace --all-targets` +
  `cargo test --workspace` (default, udp-egress gated off) green.

### M46: sans-IO H.264 RTP packetizer (`RtpH264Packetizer`)

- Opens the live-egress direction (DESIGN.md §4.12), the inverse of `RtspSrc`'s
  receive path: `rtppay.rs`'s `RtpH264Packetizer` turns an Annex-B H.264 access
  unit into RTP packets (RFC 3550 header + RFC 6184 payload), a single-NAL packet
  when the NAL fits the MTU, else FU-A fragments. The marker bit lands on the
  access unit's last packet; sequence numbers increment across packets and
  calls; one RTP timestamp per access unit. Pure Sans-IO logic (§1): no I/O, no
  dependencies, `no_std + alloc`, so an embedded device can emit RTP too.
- Refactor (DRY): the Annex-B NAL splitting `h264util` (WebCodecs) used is
  extracted to a shared `annexb` module, now used by both `h264util` and
  `rtppay`.
- Tests: four `rtppay` unit tests (single-NAL header/payload, two NALs
  incrementing sequence with the marker only on the last, an oversized NAL
  fragmenting into FU-A and reassembling byte-exactly with correct S/E/type
  bits, sequence persisting across access units) plus the moved `annexb`
  NAL-iteration test. VERIFIED on the dev host: `cargo test -p g2g-plugins
  --lib` green (47); native `cargo check --workspace` + `cargo clippy
  --workspace --all-targets` green; the WebCodecs path still compiles
  (`web-codecs` wasm32, `h264util` via `annexb`); and `rtppay` compiles for
  `thumbv7em-none-eabihf` (no_std). The UDP egress sink and RTSP server path are
  the M47 follow-up (I/O, user-side).

### M45: embassy-sync zero-alloc packet link (`PacketChannel`)

- The §6.2 "stack channels": `PacketChannel<M, N>` (`embassylink.rs`,
  `embassy-link` feature) wraps `embassy_sync::channel::Channel<PipelinePacket,
  N>`, a statically-sized, allocation-free inter-task link, the embassy-sync
  counterpart of the spin-based runtime channel. The app owns the channel (in a
  `static` / `StaticCell`) and hands its `sink` (an `OutputSink`) to a producer
  and its `receiver` to a consumer. `SinglePacketChannel<N>` is the
  single-executor (`NoopRawMutex`) default.
- Kept under a feature separate from `embassy` (the clock) so a host test links
  without embassy-time's HAL driver. The channel storage is zero-alloc; the
  `OutputSink` adapter still boxes its push future (the dyn-safe trait), so a
  fully allocation-free element model (no boxing) remains the static-graph layer
  (§4.8.1) work.
- Tests: `m45_embassy_link.rs` runs `VideoTestSrc -> EmbassySink -> channel ->
  consumer` under `embassy_futures::block_on`, asserting every frame crosses the
  embassy-sync link. VERIFIED on the dev host: `cargo test -p g2g-plugins
  --features embassy-link --test m45_embassy_link` green; `cargo check -p
  g2g-plugins --features embassy-link --target thumbv7em-none-eabihf` green
  (Cortex-M); native `cargo check --workspace` + `cargo clippy --workspace
  --all-targets` green; embassy-link clippy clean. embassy-sync references
  `critical_section::with`, so the host test pulls a `critical-section` std impl
  as a dev-dep.

### M44: Cortex-M readiness (`portable-atomic` for the metrics histogram)

- Closes the M43 finding: `metrics::LatencyHistogram` used
  `core::sync::atomic::AtomicU64`, which Cortex-M (`thumbv7em`) and RISC-V32
  lack, so the core did not compile for the canonical Embassy targets. It now
  uses `portable-atomic`'s `AtomicU64` (native where available, a lock-based
  fallback elsewhere). The new `critical-section` feature passes through to
  `portable-atomic/critical-section` so the fallback is interrupt-safe on real
  hardware (the app supplies the impl); the default `fallback` is enough to
  compile.
- Also gated the std-only `coordinator_with_recascade_n` (used only by the
  std-gated `run_linear_chain` and its tests) with `cfg(any(feature = "std",
  test))`, so the no_std `runtime` build is warning-free.
- VERIFIED on the dev host: `cargo check -p g2g-core --target
  thumbv7em-none-eabihf` (with and without `critical-section`) and `cargo check
  -p g2g-plugins --features embassy --target thumbv7em-none-eabihf` (the full
  core + plugins + EmbassyClock stack) green; the no_std `runtime` build for
  `aarch64-unknown-none` is warning-free; `cargo test -p g2g-core --lib` green
  (54, metrics on native atomics); native `cargo check --workspace` + `cargo
  clippy --workspace --all-targets` green.

### M43: Embedded/Embassy foundation (`StaticBufferPool`, `EmbassyClock`)

- Opens the third deployment target, embedded/RTOS (DESIGN.md §6.2): the no_std
  core now runs under an Embassy executor primitive. The `no_std + alloc` core
  was built for this, so M43 adds the missing pieces rather than touching the
  core.
- `StaticBufferPool<T, const N>` (g2g-core, `staticpool.rs`): the strict
  no-heap pool the §3.3 / `pool.rs` docs named as missing, the counterpart of
  the `Arc`/`Vec`-backed `BufferPool`. Fixed `[T; N]` storage, `try_acquire`
  plus an async single-waiter `acquire`, and a RAII handle that returns its slot
  on drop. Pure `core` (a `RefCell` free list, sound on the single-core
  cooperative Embassy executor), no `alloc`, no feature gate, so it works in
  strict no-heap builds.
- `EmbassyClock` (g2g-plugins, `embassy` feature): `PipelineClock` /
  `AsyncClock` over `embassy-time`, the no_std analog of `WallClock` /
  `WasmClock`. `now_ns` reads `Instant`; `sleep_until_ns` returns an
  `embassy_time::Timer` directly (no allocation). The `embassy` feature is
  no_std (does not imply std).
- `m43_embassy.rs` drives `VideoTestSrc -> FakeSink` to EOS with
  `embassy_futures::block_on` (the bare-metal future runner), proving the
  executor-agnostic runner works off Embassy, not just tokio.
- Finding: the no_std baseline compiles for bare metal on a 64-bit-atomic target
  (`aarch64-unknown-none`), but `metrics::LatencyHistogram` uses `AtomicU64`,
  which Cortex-M (`thumbv7em`) and `riscv32` lack; `portable-atomic` is the M44
  fix. A pre-existing no-std-build dead_code warning in core's `coordinator`
  (its callers are std/test-gated) is likewise an M44 cleanup.
- Tests: five `StaticBufferPool` unit tests (capacity/available, acquire +
  drop-returns, exhaustion, deref write-through, async park-then-resolve) plus
  the `block_on` integration test. VERIFIED on the dev host: `cargo test -p
  g2g-core --lib staticpool` (5) and `cargo test -p g2g-plugins --test
  m43_embassy` green; `cargo check -p g2g-core --target aarch64-unknown-none`
  and `cargo check -p g2g-plugins --features embassy --target
  aarch64-unknown-none` green (no-std bare metal); native `cargo check
  --workspace` + `cargo clippy --workspace --all-targets` green; bare-metal
  embassy clippy clean. Owed to hardware: `EmbassyClock`'s tick (a HAL time
  driver).

### M42: WebRTC data-channel ingest (`WebRtcSrc`)

- Second browser ingest path (DESIGN.md §6.3.1), alongside `WebSocketSrc`:
  `WebRtcSrc` consumes binary messages from a provided, already-open
  `RtcDataChannel` and emits each as a system-memory `DataFrame`, reusing the
  M39 `webutil::Inbox` callback-to-async bridge verbatim. Signaling
  (offer/answer/ICE) stays the application's job; the element wraps the
  negotiated channel, so it carries no signaling surface. `web` feature (stable
  web-sys). A `run_datachannel_to_canvas(channel, canvas_id)` entry wires
  `WebRtcSrc -> WebCodecsDecode -> CanvasSink`.
- Deferred: the Web Workers executor (running the pipeline off the main
  thread). That is JS-bootstrap / build infrastructure rather than element
  code, and `wasm_bindgen_futures::spawn_local` already drives pipelines on the
  event loop; it is owed a worker harness, not blocked.
- Tests: no new host-testable logic (the element is pure JS interop, like the
  display sinks). VERIFIED: base `web` and `web-codecs` (unstable cfg) wasm
  builds green; native `cargo check --workspace` + `clippy --all-targets`
  green; wasm clippy clean. NOT verifiable on this Windows host: the live
  data-channel receive needs a browser; owed a `wasm-bindgen-test` run.

### M41: in-browser presentation (`CanvasSink`), first browser glass-to-glass

- Completes the in-browser receive-to-screen path (DESIGN.md §6.3.1):
  `WebSocketSrc -> WebCodecsDecode -> CanvasSink` decodes a network H.264
  stream and draws it, the browser counterpart of the KMS / Wayland / D3D11
  display sinks. `CanvasSink` consumes decoded RGBA `System` frames and presents
  them to a `<canvas>` through the 2D context (`ImageData` + `putImageData`),
  tracking geometry from the `CapsChanged(RGBA, w, h)` the decoder emits. `web`
  feature (stable web-sys). A `run_websocket_to_canvas(url, canvas_id)` entry
  wires the full pipeline.
- The `putImageData` dx/dy argument type differs by web-sys cfg (`f64` on the
  stable bindings, `i32` under `web_sys_unstable_apis`, which the `web-codecs`
  build sets globally), so the overload is selected at compile time; the same
  `CanvasSink` source compiles under both the base `web` build and the
  unstable-cfg `web-codecs` build.
- Deferred: a WebGPU zero-copy sink (decoded `MemoryDomain::WebGPUBuffer`
  straight into a `GPUTexture`). That needs an async adapter/device handshake
  and a core keep-alive for the WebGPU domain (the §5.1 wgpu compute pillar
  builds on the same), so 2D presentation is the M41 path and WebGPU is the
  follow-up.
- Tests: no new host-testable logic (pure JS interop). VERIFIED on the dev
  host: `cargo check --target wasm32-unknown-unknown -p g2g-plugins --features
  web` (f64 path) and `RUSTFLAGS=--cfg=web_sys_unstable_apis cargo check ...
  --features web-codecs` (i32 path + canvas entries) both green; native
  `cargo check --workspace` + `cargo clippy --workspace --all-targets` green;
  both wasm clippy configs clean. NOT verifiable on this Windows host: the
  actual canvas render needs a browser; owed a `wasm-bindgen-test` run.

### M40: WebCodecs decode (`WebCodecsDecode`, `web-codecs` feature)

- Second browser/wasm element (DESIGN.md §6.3.1): the receive-to-decoded-pixels
  step. `WebCodecsDecode` wraps the browser `VideoDecoder`, consuming Annex-B
  H.264 access units and producing decoded RGBA frames in
  `MemoryDomain::System`, the browser analog of `MfDecode` (Windows) and
  `FfmpegH264Dec` (Linux). New `web-codecs` feature (implies `web`).
- Output is RGBA, not the decoder's native YUV: `VideoFrame.copyTo` is asked to
  convert via `VideoFrameCopyToOptions::format`, so negotiation fixates one
  deterministic output (`DerivedOutput(H.264 -> RGBA, same dims)`, mirroring
  `MfDecode`'s NV12) that pairs with the RGBA-consuming elements
  (`OrtInference`). Tight RGBA packing is assumed; row-stride de-pad and
  visible-rect cropping are follow-ups.
- Async shape (unlike the synchronous MFT / libav decoders): `decode()` queues
  work and the browser delivers `VideoFrame`s later through the decoder's output
  callback, bridged to the async `process` loop by the `webutil::Inbox`
  (extended with a non-blocking `try_pop`). Each `process(DataFrame)` feeds one
  chunk (tagged key/delta from in-band IDR detection) and drains the ready
  frames; `process(Eos)` awaits `flush()` then drains the reorder tail.
  Configuration is lazy: the `codec` string (`"avc1.PPCCLL"`) is derived from
  the first access unit's SPS.
- The H.264 bitstream inspection (NAL split, IDR/keyframe detection,
  codec-string from SPS) lives in a pure `h264util` module, compiled for the
  wasm build and under `cfg(test)` so it is host-testable without a browser.
- The build requires `RUSTFLAGS="--cfg=web_sys_unstable_apis"` (the WebCodecs
  web-sys bindings are unstable). A `run_websocket_decode(url)`
  `#[wasm_bindgen]` entry wires `WebSocketSrc -> WebCodecsDecode -> FakeSink`.
  H.264 only; the HEVC `codec` string is a follow-up.
- Tests: six host unit tests (`h264util`: keyframe detection, codec-string from
  SPS and its absence, NAL iteration across 3- and 4-byte start codes;
  `webutil`: `try_pop`). VERIFIED on the dev host:
  `RUSTFLAGS=--cfg=web_sys_unstable_apis cargo check --target
  wasm32-unknown-unknown -p g2g-plugins --features web-codecs` green;
  `cargo test -p g2g-plugins --lib` green (43, incl. the new 6); base `web`
  (no cfg) still builds; native `cargo check --workspace` +
  `cargo clippy --workspace --all-targets` green; wasm `web-codecs` clippy
  clean. NOT verifiable on this Windows host: the in-browser decode itself
  (live `VideoDecoder`, `copyTo` RGBA conversion) needs a browser /
  `wasm-bindgen-test` harness; the code is written to compile for wasm32 and is
  owed a browser run.

### M39: Browser/Wasm foundation (`WasmClock`, `WebSocketSrc`, `web` feature)

- Bootstraps the third deployment target (DESIGN.md §6.3): the same typed
  pipeline now compiles for `wasm32-unknown-unknown` and runs on the browser
  event loop. New `web` feature (implies std) on g2g-plugins, with the wasm
  bindings (wasm-bindgen / js-sys / web-sys / wasm-bindgen-futures) target-gated
  to `cfg(target_arch = "wasm32")` so native builds never resolve them, exactly
  like the windows/linux element gating. No core change: the runner future is
  executor-agnostic (spin-based channels), so `wasm_bindgen_futures::spawn_local`
  drives it as tokio drives it natively, and wasm builds without `multi-thread`,
  so the `!Send` JS handle types satisfy the empty `ElementBound`.
- `WasmClock` (`wasmclock.rs`): the browser `PipelineClock` / `AsyncClock`,
  `performance.now()` for `now_ns` (epoch captured at construction, like
  `WallClock`'s `Instant`) and a `setTimeout`-backed `Promise` for
  `sleep_until_ns`. The wasm analog of `WallClock`, whose tokio timer does not
  tick on `wasm32-unknown-unknown`. Degrades to a zero reading / immediate
  resolve when no `window` is present rather than panicking.
- `WebSocketSrc` (`websocketsrc.rs`): the browser ingest source, the analog of
  `FileSrc` / `RtspSrc`. Opens a `WebSocket`, receives `ArrayBuffer` messages,
  and emits each as a system-memory `DataFrame` chunk; a raw byte stream carries
  no caps, so the caller declares them at construction (`Produces`), mirroring
  `FileSrc`. The JS `onmessage` / `onclose` / `onerror` callbacks are bridged to
  the async `run` loop through a hand-rolled `Inbox` (callback-to-async queue,
  same style as the runtime's `select2`). Feed the output through `H264Parse`
  (then a `WebCodecsDecode`, M40) to recover access units.
- `run_websocket_ingest(url)` (`web.rs`): a `#[wasm_bindgen]` entry that wires
  `WebSocketSrc -> FakeSink` and `spawn_local`s `run_simple_pipeline` with
  `WasmClock`, the demonstrable browser pipeline.
- The pure logic (the `performance.now()` ms->ns conversion and the `Inbox`
  queue/waker bridge) lives in `webutil.rs`, compiled for the wasm `web` build
  and under `cfg(test)`, so it is unit-testable on the host without a browser.
- Tests: four host unit tests (`ms_to_ns` conversion + clamping; `Inbox`
  in-order drain, park-then-wake, drain-before-close). VERIFIED on the dev host:
  `cargo check --target wasm32-unknown-unknown -p g2g-plugins --features web`
  green; `cargo test -p g2g-plugins --lib webutil` green (4/4); native
  `cargo check --workspace` and `cargo clippy --workspace --all-targets` green
  (web modules are wasm32-gated, so native is unaffected); wasm `web` clippy
  clean. NOT verifiable on this Windows host: the in-browser runtime (live
  WebSocket receive, `performance.now()` pacing) needs a browser /
  `wasm-bindgen-test` harness; the code is written to compile for wasm32 and is
  owed a browser run, as the Linux display sinks are owed a Linux run.

### M38: WASAPI loopback capture in `WasapiSrc`

- `WasapiSrc::with_loopback()` captures the default render endpoint's output
  (what the system is playing) in WASAPI loopback mode, instead of a capture
  endpoint (mic / line-in). The endpoint dataflow (`eRender`) and the
  `AUDCLNT_STREAMFLAGS_LOOPBACK` init flag thread through the probe and the
  capture worker; the mix format then comes from the render endpoint.
- Tests: a unit test for the builder/`is_loopback` shape plus
  `m38_wasapi_loopback.rs`, which plays a tone through `WasapiSink` on a
  background thread while the loopback source captures, asserting the requested
  buffers of non-empty PCM arrive with `Eos`, skipping when no render endpoint
  is present. VERIFIED on the dev host: loopback captured 5 buffers of the
  played-back tone; `cargo check --workspace` green; feature clippy clean.

### M37: AAC audio in the fMP4 container (`Mp4AudioSink` / `Mp4AudioSrc`)

- An audio-only AAC fMP4 (`.m4a`) muxer/demuxer pair, the audio counterpart of
  `Mp4Sink`/`Mp4Src`: one `soun` track, `mp4a`/`esds` sample entry, one
  `moof`+`mdat` fragment per access unit, media timescale = sample rate.
  std-gated, self-contained (own box writer/reader so the video muxer is
  untouched).
- `Mp4AudioSink` writes the `esds` (ES/DecoderConfig/DecoderSpecific
  descriptors) from a supplied AudioSpecificConfig
  (`with_audio_specific_config`, from `MfAacEncode`); AAC access units are
  stored verbatim. `Mp4AudioSrc` recovers the codec/channels/rate and the ASC
  from the probe (exposed via `audio_specific_config()`) and parses the `esds`
  descriptor tree (expandable sizes) plus the `moof`/`mdat` fragments.
- Tests: unit tests on both elements (esds carries the ASC, fragment data
  offset, caps/ASC rejection, descriptor reader for single-byte and expandable
  sizes, timescale) plus `m37_audio_mp4.rs`: a cross-platform `Mp4AudioSink ->
  Mp4AudioSrc` round trip recovering every access unit byte-exactly with the
  ASC, and on Windows a full audio file loop `MfAacEncode -> Mp4AudioSink ->
  Mp4AudioSrc -> MfAacDecode` (PCM -> AAC -> .m4a -> PCM), the first complete
  file-based audio glass-to-glass loop. VERIFIED: `cargo test --workspace`
  green (0 failures), full circle green, `std`/`mf-aac` clippy clean.

### M36: `MfAacDecode` AAC audio decoder

- The decode-side mirror of `MfAacEncode`: consumes raw AAC-LC access units and
  produces interleaved 16-bit PCM via the MS AAC Decoder MFT
  (`CLSID_MSAACDecMFT`, synchronous). Windows-only, shares the `mf-aac` feature.
- Needs the stream's AudioSpecificConfig to configure its input type
  (`with_audio_specific_config`, supplied by the encoder or an MP4 `esds`); it
  is wrapped in the 12-byte HEAACWAVEINFO `MF_MT_USER_DATA` header. `configure`
  fails loud without it. A `DerivedOutput` constraint maps AAC to S16 PCM at the
  same channels/rate.
- Tests: four unit tests (intercept, derived-output mapping, user-data framing,
  missing-ASC rejection) plus `m36_aac_roundtrip.rs`, a `MfAacEncode ->
  MfAacDecode` round trip through both real MFTs that recovers the stream with a
  decoded sample-frame count within the AAC priming delay of the input.
  VERIFIED on the dev host: PCM -> AAC -> PCM round trip green; `mf-aac` clippy
  clean.

### M35: `MfAacEncode` AAC audio encoder

- Compressed-audio analog of `MfEncode`: consumes interleaved 16-bit PCM
  (`PcmS16Le`) and produces raw AAC-LC access units (`AudioFormat::Aac`, one per
  1024-sample frame, no ADTS header) via the MS AAC encoder MFT. Windows-only
  behind the `mf-aac` feature.
- The AAC encoder has no fixed CLSID, so it is enumerated by output subtype via
  `MFTEnumEx`; it is synchronous, so the same `ProcessInput`/`ProcessOutput`
  drain loop as the H.264 encoder applies. Bitrate is selectable
  (`with_bytes_per_second`, validated against the encoder's 96/128/160/192 kbps
  set). The AudioSpecificConfig is read from the negotiated output type's
  `MF_MT_USER_DATA` (past the 12-byte HEAACWAVEINFO tail) and exposed via
  `audio_specific_config()` for a decoder or MP4 `esds`.
- Negotiation: a `DerivedOutput` constraint maps S16 PCM to AAC at the same
  channels/rate; non-S16 input is rejected.
- Tests: three unit tests (intercept, derived-output mapping, bitrate
  validation) plus `m35_aac_encode.rs`, which drives the real encoder over 10
  PCM buffers and asserts non-empty AAC access units, a non-empty ASC, and AAC
  output caps. VERIFIED on the dev host: the MS AAC encoder produced AUs + ASC;
  `mf-aac` clippy clean.

### M34: `AudioConvert` PCM converter

- The audio analog of `VideoConvert`: converts interleaved PCM between sample
  formats (`PcmS16Le` <-> `PcmF32Le`) and channel counts (identity, mono
  fan-out, downmix-to-mono average) at the same sample rate, so audio chains
  compose across format boundaries (`WasapiSrc` emits f32, `WavSink` / encoders
  often want s16). CPU-only `no_std` baseline; samples pass through an f32
  intermediate, s16 rounding is half-away-from-zero without libm.
- Negotiation mirrors `VideoConvert`: a `DerivedOutput` constraint maps a
  supported PCM input to the target format/channels; unsupported channel remaps
  (e.g. 5.1 -> stereo) and non-PCM inputs yield an empty set and fail loud.
- Tests: seven unit tests (derived-output mapping, f32<->s16 round trip within a
  quantum, peak scaling, mono fan-out, stereo downmix average, ragged-input
  rejection, channel-remap rejection) plus `m34_audioconvert.rs`, which runs
  `AudioTestSrc(s16) -> AudioConvert -> WavSink` through the real
  source-transform-sink runner and asserts a float32 WAV and a downmixed mono
  track, proving the chain negotiates. VERIFIED: `cargo test --workspace` green
  (0 failures), `std` clippy clean.

### M33: Async-MFT support in `MfEncode` (hardware encoders)

- Completes the M30 deferral: `MfEncode` now drives asynchronous (event-based)
  encoder MFTs, the common shape of a hardware H.264/HEVC encoder, not just
  synchronous ones. `enumerate_encoder` includes async MFTs and unlocks them
  (`MF_TRANSFORM_ASYNC_UNLOCK`); the new `with_hardware()` builder also routes
  H.264 through enumeration (the default H.264 path keeps the fixed-CLSID MS
  software encoder).
- Async encoders are driven by an `IMFMediaEventGenerator` event loop:
  `METransformNeedInput` feeds a queued input sample (or banks a credit),
  `METransformHaveOutput` pulls an encoded frame; on `Eos` the queued input is
  flushed, then `END_OF_STREAM` + `DRAIN` run until `METransformDrainComplete`.
  The sync `ProcessInput`/`ProcessOutput` path is unchanged and selected per
  MFT. `is_async()` reports which path the live MFT uses. Flush clears pending
  input.
- Tests: unit test for the builder/`is_async` shape plus `m33_async_encode.rs`,
  which drives a real hardware encoder via `with_hardware()` and asserts every
  picture encodes out as Annex-B H.264, skipping when no hardware encoder is
  registered. VERIFIED on the dev host: it selected an asynchronous hardware
  H.264 MFT (`async_mode = true`) and encoded all 30 frames through the event
  loop; m19 (sync H.264) and m30 (HEVC) unregressed; combined `mf-encode` +
  `mf-decode` clippy clean; `cargo check --workspace` green.

### M32: `WasapiSrc` WASAPI capture source

- The input mirror of M29's `WasapiSink`: `WasapiSrc` captures interleaved PCM
  from the default audio capture endpoint (WASAPI shared mode) and emits
  `DataFrame`s, so a live mic / line-in feeds a pipeline the way `AudioTestSrc`
  feeds a synthetic tone. Windows-only behind the `wasapi-src` feature.
- Caps come from the endpoint mix format, probed during negotiation
  (`intercept_caps` on a short COM thread): `PcmF32Le` (the usual shared-mode
  format) or `PcmS16Le`, at the device's channel count and rate. A headless
  host (no capture endpoint) fails the probe loud so negotiation rejects the
  pipeline rather than hanging.
- Capture runs on a dedicated COM worker; buffers cross to the async `run` loop
  over a tokio channel where they are stamped with device-clock timing and
  pushed. Emits a fixed buffer count then `Eos` (the bounded test-source shape),
  with a wall-clock guard so a silent/stalled endpoint can't run forever.
- Tests: three unit tests (mix-format -> config mapping incl. EXTENSIBLE,
  unsupported bit-depth rejection, source-only pad template) plus
  `m32_wasapi_capture.rs`, which drives the source loop against a real endpoint
  and asserts the requested buffers of non-empty PCM arrive with `Eos`, skipping
  when no capture device is present. VERIFIED on a host with audio in: captured
  5 buffers; unit suite green; `cargo check --workspace` green; feature clippy
  clean.

### M31: HEVC in the fMP4 container (`Mp4Sink` / `Mp4Src`)

- The container is now codec-aware, matching the M30 encoders. `Mp4Sink` muxes
  H.264 (`avc1`/`avcC`, SPS+PPS) or H.265 (`hvc1`/`hvcC`, VPS+SPS+PPS); the
  codec comes from the negotiated caps. NAL parsing is codec-aware (H.265 type
  in bits 1..6, IRAP keyframes 16..=23); the `hvcC` general profile_tier_level
  is copied from the SPS, descriptive fields default to 4:2:0 8-bit (what the MS
  HEVC encoder emits), and parameter sets stay in-band so a player re-parses
  authoritative values.
- `Mp4Src` detects the sample entry (`avc1` vs `hvc1`/`hev1`), parses the
  matching config record for the parameter sets, and reports the real codec
  from the caps probe. Out-of-band parameter sets are prepended per codec when
  the first access unit carries none.
- Negotiation surfaces (`intercept_caps`, `caps_constraint_as_sink`,
  pad templates) accept H.264 or H.265; a mid-stream codec swap is rejected.
- Tests: new unit tests on both elements (H.265 NAL type + keyframe detection,
  `hvcC` build and parse, codec-aware param-set detection) plus `m31_hevc_mp4.rs`,
  a cross-platform `Mp4Sink -> Mp4Src` HEVC round trip over hand-built NALUs
  that recovers every access unit byte-exactly and probes back H.265 +
  geometry. VERIFIED: full `cargo test --workspace` green (0 failures), H.264
  m24/m28 unregressed, `std`-feature clippy clean.

### M30: HEVC support in the MF encode/decode elements

- `MfEncode` and `MfDecode` are now codec-aware: `with_codec(VideoCodec::H265)`
  selects H.265/HEVC (default stays H.264). The codec threads through caps
  negotiation (`intercept_caps`, the `DerivedOutput` constraint,
  `configure_pipeline`, mid-stream `CapsChanged`) and the MFT setup; pad
  templates advertise both codecs as the static superset, the instance narrows
  to one.
- Decode picks the MFT by fixed CLSID: `CLSID_MSH264DecoderMFT` or
  `CLSID_MSH265DecoderMFT`, with the matching input subtype
  (`MFVideoFormat_H264`/`_HEVC`). The MS HEVC decoder ships as a Store
  extension, so an absent decoder surfaces as a loud `Hardware` error.
- Encode has no fixed HEVC CLSID, so H.265 is found via `MFTEnumEx` for the
  HEVC output subtype. Only a synchronous MFT is driven by the existing
  `ProcessInput`/`ProcessOutput` loop; an enumerated asynchronous (hardware)
  MFT is rejected loud rather than mis-driven. Async-MFT support is deferred.
- Tests: new unit tests on both elements (codec default/select, CLSID/subtype
  mapping, HEVC caps derivation, codec-mismatch rejection) plus `m30_hevc.rs`,
  an `MfEncode(H265) -> MfDecode(H265)` round trip that skips gracefully when
  either MFT is unavailable. VERIFIED: on the dev host the HEVC encoder MFT is
  present and produced valid Annex-B H.265 (the test reached the decode stage
  and skipped there, the HEVC decoder extension being absent); H.264 round trip
  unregressed (`m19` green); `cargo test --workspace` green; combined
  `mf-encode`+`mf-decode` clippy clean.

### M29: `WasapiSink` WASAPI render sink

- The audible-output end of the M25 audio path (`AudioTestSrc`/`WavSink`):
  `WasapiSink` plays interleaved PCM (`PcmS16Le`/`PcmF32Le`) on the default
  render endpoint via WASAPI shared mode, so an audio pipeline makes sound
  instead of only writing a file. Windows-only behind the `wasapi-sink`
  feature; pulls the `windows` crate (`Win32_Media_Audio` +
  `Win32_System_Com_StructuredStorage`/`_Variant` for `IMMDevice::Activate`).
- Threading mirrors `D3D11Sink`: COM/WASAPI run on a dedicated worker thread
  spun up at `configure_pipeline`; the sink struct holds only `Send` handles
  (mpsc sender + a frame counter), PCM bytes cross by value. The worker tops
  up the shared-mode endpoint buffer at the device rate, queuing the source's
  faster-than-real-time output; on `Eos` it drains the queue and waits for
  playout (guarded) so the tail is not cut off.
- A headless host (no render endpoint) fails loud in `configure_pipeline`
  rather than silently dropping audio: the worker reports endpoint-open
  success over a ready channel.
- Tests: four unit tests (PCM caps mapping + compressed rejection,
  `WAVEFORMATEX` field derivation, intercept, pad template) plus
  `m29_wasapi.rs`: `AudioTestSrc -> WasapiSink` renders 200 ms of tone and
  asserts every produced sample frame reached the endpoint, skipping
  gracefully when no audio device is present. VERIFIED on a host with audio
  out: `wasapi-sink` suite green (2/2 incl. real render, 4/4 unit),
  `cargo check --workspace` green, feature clippy clean.

### M28: `Mp4Src` fragmented-MP4 demuxer source

- The read-side counterpart of `Mp4Sink`, closing the container loop:
  `Mp4Src` parses a single-video-track fMP4 and emits Annex-B H.264 access
  units with timing recovered from `tfdt`/`trun` (90 kHz -> ns), so a
  recording plays back through a decoder exactly like a live stream.
  std-gated, no new deps (hand-written box walker mirroring the sink's
  writers).
- Caps discovery rides the M18 async-source probe: `intercept_caps` reads
  `moov` before negotiation (dims from `tkhd`, timescale from `mdhd`,
  SPS/PPS from `avcC`), so downstream solves against the recording's real
  geometry. If the first sample carries no in-band SPS, the `avcC`
  parameter sets are prepended so a decoder can start (the in-tree writer
  keeps them in-band).
- Supported profile is what `Mp4Sink` writes and CMAF-style single-track
  files share: `trun` v0 with explicit sample sizes, sequential samples in
  the following `mdat`. Anything else (v1 trun, missing sizes, `mdat`
  without `moof`, truncated NALUs, non-MP4 bytes) fails loud with
  `CapsMismatch` instead of emitting a corrupt bitstream.
- Tests: four unit tests (AVCC->Annex-B round trip incl. truncation, the
  writer-profile `trun` layout, timescale inversion, SPS detection) plus
  `m28_mp4src.rs`: a write -> read round trip recovers every access unit
  byte-exactly with ~33.33 ms pts spacing, garbage/missing files fail the
  probe loud, and on Windows the full circle
  `MfEncode -> Mp4Sink -> Mp4Src -> MfDecode` returns all 10 frames as
  packed NV12 through both real MFTs. VERIFIED: `cargo test --workspace`
  green, the mf-feature suite green (3/3 incl. full circle), workspace and
  feature clippy clean.

### M27: `TensorPostprocess` classification head

- Completes the in-graph classification chain:
  `... -> OrtInference -> TensorPostprocess`. Two operations over f32
  tensors (treated as one flat vector, the conventional reading of
  `[1, N]` logits): `softmax()` (numerically stable, shift-by-max; output
  caps echo the input caps) and `argmax()` (emits a `[1, 2]` f32
  `[winning index, winning value]` tensor, first occurrence wins ties).
  Pure Rust, no dependencies, always available in `g2g-ml` (no feature
  gate). Native `DerivedOutput` constraint; non-f32 tensors rejected at
  negotiation; timing passes through so latency stays traceable.
- The hand-encoded ONNX fixture builder moved to a shared
  `tests/util/onnx_fixture.rs` (included by both g2g-ml integration test
  crates) instead of living inline in `ort_inference.rs`.
- Tests: four unit tests (softmax sums to 1 and orders correctly, stays
  finite for 1000-scale logits, argmax first-max semantics, derived-output
  shapes incl. non-f32 rejection) plus `postprocess.rs` integration:
  element-level softmax/argmax with caps emission, U8 caps rejected, and,
  under `ort`, the full real-runtime chain where identity-model inference
  into argmax finds the brightest input pixel's flat NCHW index and
  normalized value exactly. VERIFIED: `cargo test -p g2g-ml --features
  ort` green (5 unit + 7 integration across both files),
  `cargo test --workspace` green, workspace + ort-feature clippy clean.

### M26: DirectML execution provider for `OrtInference`

- First GPU inference path on Windows: a new g2g-ml `directml` feature
  (implies `ort`, adds `ort/directml`, which downloads a DirectML-enabled
  ONNX Runtime build) and `OrtInference::from_memory_with_directml`, which
  registers the DirectML EP (any D3D12 GPU) ahead of the CPU fallback.
  Registration is best-effort per ort's dispatch default: on a host
  without a usable DirectML device the session silently runs on the CPU,
  so the pipeline keeps flowing either way. The element shape is
  unchanged; the EP choice is a constructor variant.
- `ort_err` generalized over the error payload (`Error<SessionBuilder>`
  from builder-consuming calls vs the plain `Error<()>`).
- Test (gated on `directml`, runs against the real DML-enabled runtime):
  the identity-model inference through `from_memory_with_directml`
  produces byte-identical results to the CPU path. VERIFIED:
  `cargo test -p g2g-ml --features directml` green (3/3),
  `--features ort` (CPU-only) still green, matching clippy clean,
  `cargo test --workspace` green.

### M25: first audio elements (`AudioTestSrc` / `WavSink`)

- Bootstraps the audio track: `Caps::Audio` existed in core with zero
  elements using it. `AudioTestSrc` (baseline, `no_std`) is the audio
  analog of `VideoTestSrc`: deterministic interleaved S16LE PCM test tone
  (sine / square / silence) in 10 ms buffers at a configured rate and
  channel count, full `SourceLoop` with a native `Produces` constraint.
  The sine is Bhaskara I's approximation in pure f32 arithmetic (core has
  no `sin` intrinsic), accurate to ~0.2%: clean enough for a test tone
  without a libm dependency.
- `WavSink` (std) writes PCM (`PcmS16Le` / `PcmF32Le`) to a canonical
  RIFF/WAVE file, patching the running sizes in place on `Eos`. Compressed
  audio caps (`Aac` / `Opus`) are rejected at negotiation via a legacy-sink
  intercept (`Caps::Audio` has no open dims for a native set). WAV is the
  playable-anywhere convenience sink; fragmented/durable recording remains
  `Mp4Sink`'s territory.
- Tests: six unit tests (sine anchors and peak, square/silence waves, caps
  shape, canonical 44-byte header fields, PCM param mapping) plus
  `m25_audio.rs` through the real runner: 1 s of 1 kHz stereo sine lands as
  a structurally valid WAV (patched RIFF/data sizes, 48 kHz/2ch/16-bit fmt,
  byte-exact length, zero first sample, real peak amplitude); silence is
  all-zero; AAC caps fail loud. VERIFIED: `cargo test --workspace` green,
  workspace clippy clean, default (no_std) plugins build green.

### M24: `Mp4Sink` fragmented-MP4 muxer sink

- H.264 recordings become standard playable files: `Mp4Sink` wraps an
  Annex-B H.264 stream in a fragmented MP4 (`ftyp` + `moov` once, then one
  `moof`+`mdat` per access unit, CMAF-style). A truncated live recording
  stays valid up to the last complete fragment, the durability property a
  glass-to-glass recorder wants; `FileSink` remains the raw-bitstream
  alternative. std-gated, no new deps (hand-written box serializers).
- Access units convert to AVCC (4-byte length-prefixed NALUs, parameter
  sets kept in-band) for the `mdat`. The `moov` needs SPS/PPS, which arrive
  in-band with the first IDR, so the header is written on the first AU;
  dims come from the negotiated caps (`Accepts(H264 any geometry)`,
  terminal sink pad template). 90 kHz media timescale; per-fragment
  duration from `duration_ns`, else pts delta, else 1/30 s; `tfdt`
  accumulates decode time; IDR fragments are marked sync samples.
- Tests: five unit tests (Annex-B splitting across 3- and 4-byte start
  codes, AVCC length prefixing, 90 kHz conversion, box framing, and the
  `trun` data offset landing exactly on the `mdat` payload) plus
  `m24_mp4sink.rs`: a synthetic-AU recording walks back as
  `ftyp moov (moof mdat)x4` with the exact SPS/PPS bytes inside `avcC` and
  incrementing `mfhd` sequence numbers; non-H.264 caps are rejected; and
  on Windows a real `MfEncode` stream records to a structurally valid
  10-fragment file. VERIFIED: `cargo test --workspace` green,
  `cargo test -p g2g-plugins --features mf-encode --test m24_mp4sink`
  green (3/3, real-encoder case included), workspace clippy clean.

### M23: `VideoConvert` software raw-format converter

- Closes the raw-format gap between element families: chains like
  `VideoTestSrc (RGBA) -> MfEncode (NV12)` or `decoder (NV12) ->
  OrtInference (RGBA)` previously failed the whole-chain solve with no
  in-tree bridge. `VideoConvert::new(target)` converts any supported raw
  format (`Rgba8`, `Bgra8`, `Nv12`, `I420`) to the target at the same
  geometry. CPU-only, integer-only math, `no_std`: lives in the plugin
  baseline with no feature gate.
- Conversion paths: BT.601 limited-range integer transforms for
  RGB <-> 4:2:0 YUV (chroma from 2x2 block averages); lossless fast paths
  for same-family pairs (RGBA<->BGRA channel swizzle, NV12<->I420 chroma
  repack); same-format passthrough. 4:2:0 endpoints require even dims,
  rejected loud at negotiation. Caps surface is a native
  `DerivedOutput(any supported raw -> target at same dims/rate)`;
  `PadTemplates` declare the full format set on both pads.
- Tests: six unit tests (BT.601 primary colors round-trip within integer
  tolerance, grey maps to exactly neutral chroma, lossless swizzle and
  NV12<->I420 repack, odd-dim rejection, derived-output mapping) plus
  `m23_videoconvert.rs`: an RGBA test source reaches an NV12-only sink
  through the converter via the real solver, and the control chain without
  the converter fails the same negotiation. VERIFIED:
  `cargo test --workspace` green, workspace clippy clean.

### M22: `TensorBatcher` bootstraps `g2g-enterprise`; per-input Eos contract

- Bootstraps the last empty crate with the DESIGN.md §5.3 bounded batcher:
  `TensorBatcher` is a `MultiInputElement` gathering one tensor frame per
  input stream and emitting the round as one batched frame, stacked along
  the leading batch dim (`N` slots of `[1, d...]` -> `[N, d...]`; stacking
  dim 0 of a dense row-major tensor is byte concatenation in input order).
  Composes with M21's `OrtInference` family: feed a dynamic-batch model one
  execution per N camera streams. Per-input negotiation pins every input to
  the identical slot caps (`Accepts(slot)`); output is `Produces([N, d...])`.
- Liveness over completeness: an input reaching end-of-stream stops gating
  the gather (a dead camera must not stall the rest); its queued frames
  still drain into batches, then batches shrink to the survivors with a
  `CapsChanged` before the first smaller frame. Batch timing: pts is the
  newest constituent, arrival the oldest non-zero stamp (worst-case
  glass-to-glass). Owed: the §5.3 deadline-based partial flush ("Timeout"),
  gated on a runtime timer primitive.
- Core contract change (M22): `run_muxer_sink` now delivers each input's
  `Eos` to `MultiInputElement::process(input, Eos, ..)` before aggregating,
  so a stateful muxer can flush per-input state; elements must not forward
  it (the runner still owns the single merged downstream `Eos`).
  `InterleaveMux` updated to swallow per-input `Eos` accordingly; trait
  docs updated.
- Tests: three unit tests (slot validation, batch-dim stacking, non-tensor
  reject) and four integration tests: byte-exact two-round gather with
  out-of-order arrival, EOS shrink with exactly-once `CapsChanged`, an
  ended input's queue draining into full batches, and a 2-source end-to-end
  run through the real `run_muxer_sink` (3 full batches, single aggregated
  EOS). VERIFIED: `cargo test --workspace` green (m10/m18 mux suites
  unaffected by the Eos contract change), workspace clippy clean.

### M21: first `g2g-ml` element, `OrtInference` (ONNX Runtime)

- Bootstraps the `g2g-ml` crate (previously an empty stub) with its first
  inference element, per DESIGN.md §5: `OrtInference` (`ort` feature) is an
  `AsyncElement` transform negotiating `Caps::RawVideo` on the input pad
  and `Caps::Tensor` on the output pad, sitting in the graph like any other
  element. Each RGBA frame converts to a normalized f32 NCHW RGB tensor
  (`value / 255`), runs through the session (ONNX Runtime default CPU EP;
  CUDA / TensorRT / DirectML EPs are an in-module builder follow-up), and
  the model's output is emitted as a `DataFrame` of f32 LE bytes under
  `Caps::Tensor { F32, shape, Nchw }`, inheriting the source frame's timing
  so glass-to-glass latency stays traceable.
- v1 model contract validated loud at construction: one f32 tensor input
  `[N, 3, H, W]` (static H/W, batch 1 or dynamic-as-1) and one f32 tensor
  output with static dims (dynamic leading batch coerced to 1). The element
  then pins negotiation to RGBA at exactly `W x H`
  (`DerivedOutput(RGBA@WxH -> tensor caps)`), so a geometry mismatch fails
  at solve time, not at inference time.
- Dependency: `ort` 2.0.0-rc.12 (pykeio's ONNX Runtime binding, verified
  current and maintainer-recommended; API checked against the fetched crate
  source). Default ort features download the platform's ONNX Runtime
  binaries on first build (network required). The module is named
  `ortinfer` so in-crate paths can't collide with the `ort` dependency.
- Tests run against real ONNX Runtime with no checked-in blob or network
  fixture: `ort_inference.rs` hand-encodes a minimal ONNX ModelProto (one
  `Identity` node) in ~80 lines of protobuf wire format. Contract
  validation (4-channel and rank-2 models rejected), geometry-pinned
  negotiation, exactly-once `CapsChanged` emission, and byte-exact output
  (Identity model output == the normalized RGB planes) are all asserted;
  plus a unit test for the dynamic-dim coercion. VERIFIED:
  `cargo test -p g2g-ml --features ort` green (1 unit + 2 integration),
  matching clippy green, default `cargo test --workspace` green
  (the new dep is feature-gated off the default graph).

### M20: file I/O elements (`FileSink` / `FileSrc`)

- Closes the record / playback loop M19 opened: `FileSink` writes every
  `DataFrame`'s system-memory bytes to a file in arrival order (so
  `testsrc -> MfEncode -> FileSink` records a playable Annex-B `.h264`
  elementary stream), and `FileSrc` replays a file as `DataFrame` chunks
  (`FileSrc -> H264Parse -> decoder` recovers access units). Both are
  platform-agnostic, std-gated (`std::fs`, no new deps).
- `FileSink` is a wildcard sink (`AcceptsAny`; a raw byte stream can record
  anything). Caps are not representable in a raw stream, so `CapsChanged`
  packets pass without effect; `Flush` is a no-op (no seek index to reset);
  EOS flushes the writer. The file is created in `configure_pipeline`, not
  at construction. `PadTemplates` sink-any.
- `FileSrc` takes the stream's caps at construction (a raw recording
  carries none) and declares exactly that to the solver
  (`Produces(CapsSet::one(caps))`). Reads in `with_chunk_size` chunks
  (default 64 KiB, clamped to 1), short-read tolerant, no timing on chunks
  (recovered downstream by parser/decoder). No `PadTemplates`: its caps are
  instance configuration and a source pad cannot be wildcard (like
  `RtspSrc`).
- New `HardwareError::Io(i32)` core variant carries the raw OS error code
  from filesystem failures, mirroring `V4l2(i32)` / `Cuda(i32)` instead of
  flattening to `Other`. no_std-safe.
- Tests (`m20_file_io.rs`): `VideoTestSrc -> FileSink` through the real
  runner records the exact deterministic byte stream; `FileSrc` chunking
  (7-byte chunks over 20 bytes -> [7, 7, 6] + Eos, byte-exact reassembly);
  a full record -> replay round trip through `run_simple_pipeline` with a
  frame-boundary-misaligned chunk size; loud structured `Io` failures for
  an uncreatable sink path, an unconfigured sink, and a missing source
  file. VERIFIED: `cargo test --workspace` green, workspace clippy clean,
  no_std core baseline and default (no_std) plugins build green.

### M19: `MfEncode` Windows H.264 encode element

- First encoder element in the tree, completing the encode side of the
  Windows platform track. `MfEncode` (`mf-encode` feature, Windows-only)
  wraps the Media Foundation H.264 Encoder MFT (`CLSID_MSH264EncoderMFT`):
  packed NV12 `System` frames in, Annex-B H.264 access units out (one
  encoded sample per input picture, SPS/PPS attached to each IDR). Mirrors
  the `MfDecode` structure verbatim: same COM/MTA `Send` contract, same
  ProcessInput/ProcessOutput drain with the NOTACCEPTING retry, same DRAIN
  on EOS.
- Low latency by default: `MF_LOW_LATENCY` (the attribute alias of
  `CODECAPI_AVLowLatencyMode`) is set on the MFT's attribute store before
  the media types, so the encoder runs without B-frames / lookahead and
  releases one output per input. `with_bitrate(bits_per_sec)` sets
  `MF_MT_AVG_BITRATE` (default 4 Mb/s).
- Geometry contract differs from the decoder: an encoder's media types need
  concrete dims up front (no bitstream to derive them from), so
  `configure_pipeline` requires fixed, even NV12 dims and fails loud
  otherwise. A mid-stream `CapsChanged` to new dims drains the current MFT
  (buffered pictures emit under the caps they were encoded with, carried
  per-chunk) and rebuilds it at the new geometry. Caps surface is the
  inverse of the decoder: `DerivedOutput(NV12 -> H264, same dims/rate)`,
  `PadTemplates` NV12 sink pad / H.264 source pad.
- All COM (encoder CLSID, `MF_LOW_LATENCY`, the encode-side media-type
  attributes `MF_MT_AVG_BITRATE` / `MF_MT_FRAME_RATE` / `MF_MT_INTERLACE_MODE`
  / `MF_MT_MPEG2_PROFILE`, `SetSampleDuration`) verified against the fetched
  `windows-0.62.2` source per AGENTS.md.
- Tests: seven GPU-free unit tests (caps narrowing/rejection, derived
  output, pad templates, bitrate builder, Q16 framerate -> MF ratio, frame
  duration), plus `m19_mfencode.rs` against the real MS software encoder
  MFT: 10 synthetic NV12 pictures encode to 10 Annex-B access units with
  one H.264 `CapsChanged`, and the encode -> decode round trip through
  `MfDecode` recovers all 10 frames at the original geometry. VERIFIED on
  the Windows dev host: `cargo test -p g2g-plugins --features "mf-encode
  mf-decode"` green (36 lib + 2 integration), matching clippy green,
  `cargo test --workspace` (default) green.

### M18 item 4 follow-up: dyn interior elements contribute clock + latency

- `run_linear_chain` folded `RunStats.latency` and `clock_priority` from only
  the statically-typed source and sink; its `dyn` interior transforms were
  skipped because `DynAsyncElement` didn't expose `latency` / `provide_clock`.
  A buffering interior element (jitter buffer, reorder queue) under-reported
  pipeline latency, and an interior clock provider was ignored in election.
- `DynAsyncElement` gains dyn-safe `latency` / `provide_clock` mirrors
  (defaulting to zero / none, matching `AsyncElement`; the blanket impl
  delegates to the concrete element). `run_linear_chain` now folds source, every
  interior element, then sink in path order for both the latency aggregate and
  the clock election. Closes the last "owed" bullet on the `run_linear_chain`
  doc (β re-cascade and the N-hop caps re-solve already landed).
- Test: `m18_dyn_latency_clock.rs` runs `VideoTestSrc -> buffering+clock
  transform -> FakeSink` (source/sink contribute nothing), asserting the path
  latency (5 ms..10 ms) and elected `Provider` clock priority come from the
  interior element. VERIFIED: `cargo test --workspace` green, no_std + runtime +
  runtime/std builds, core clippy clean.

### M18 item 4 follow-up: Caps-α mid-stream caps re-solve over the downstream subgraph

- Closes the caps half of the multi-element re-solve gap (the allocation half
  landed as β N-hop; DESIGN-M16-caps-nego.md §13.3, §13.4 item 4). On a
  mid-stream `CapsChanged`, `run_linear_chain` now derives each interior
  element's forwarded output from its declared constraint, **steered by a
  downstream feasibility snapshot**, instead of letting the element fixate
  greedily and forward a doomed caps the sink then rejects (D3 from
  DESIGN-M16-workaround3-reconfigure.md §4; full rationale in the new
  DESIGN-M18-caps-resolve.md).
- Mechanism (Caps-α): a backward feasibility sweep computed once at startup
  (`solver::downstream_feasibility`) snapshots, per output link, the set the
  downstream subgraph can still fixate, **independent of the (mid-stream
  changing) upstream**. Each interior arm keeps its snapshot and, on a
  `CapsChanged`, calls `solver::resolve_forward_output`: it derives the
  element's output candidates from its constraint, narrows by the snapshot, and
  fixates. `Fixed` forwards the steered output; `Defer` keeps prior behavior
  (no concrete downstream set, or a `DerivedOutput`/ranged output the runner
  can't fixate, so the element's own `process` derives); `Infeasible` fails
  loud (reverse `Reconfigure` into the boundary + structured `EmptyLink` on the
  bus). This needs no central solve and no ownership move: each arm reaches only
  its own constraint, which the spawned-arm topology already allows.
- Scope: both `run_linear_chain` (N-hop) and the single-transform
  `run_source_transform_sink`. The single-transform path needs no multi-hop
  sweep (its downstream subgraph is one sink link), so it reads the sink's
  `Accepts` set inline; `resolve_forward_output` / `ForwardResolve` are no_std
  (that runner is a no_std runtime path), while `downstream_feasibility` stays
  `std`-gated with `run_linear_chain`. Steering activates only when a concrete
  downstream `Accepts` set exists, so pass-through chains (`IdentityAny`) and
  `AcceptsAny` sinks are byte-identical to before. Owed: Caps-β (a forward
  coordinator re-solve walk) for a downstream `DerivedOutput` element that must
  re-derive mid-stream, gated on a real driver.
- Tests: `m18_caps_resolve.rs` drives a `DerivedOutput` converter through a
  mid-stream RGBA -> I420 change into an NV12-only sink: Caps-α steers the
  converter to NV12 (`midstream_change_steers_converter_to_sink_acceptable_output`),
  a converter with no NV12 path fails loud to the bus
  (`..._no_acceptable_output_fails_loud_to_bus`), and the single-transform
  runner steers identically (`single_transform_runner_also_steers_to_sink_acceptable_output`).
  Plus two solver unit tests (`downstream_feasibility_is_source_independent`,
  `resolve_forward_output_steers_defers_and_rejects`). VERIFIED:
  `cargo test --workspace` green (g2g-core 125, g2g-plugins suites incl. the new
  3), β N-hop and multi-element runners unaffected, no_std / runtime / runtime+std
  builds, core clippy clean.

### M18 item 4 follow-up: β allocation re-cascade over N hops

- Extends the single-hop β (sink -> lone transform) to the full interior of
  `run_linear_chain`: a mid-stream `CapsChanged` now re-cascades the allocation
  demand through *every* interior element, sink -> t_{n-1} -> ... -> t0
  (DESIGN-M16-caps-nego.md §13.3, §13.4 item 4).
- The cascade is **reactive**, which is what keeps it deadlock-free. A direct
  upstream control chain would deadlock (the sink blocks sending the directive
  while the last transform is backpressured pushing data to the not-yet-draining
  sink). Instead the `Coordinator` is a separate task: the sink reports its
  proposal and immediately resumes draining; the coordinator forwards a
  `Recascade` to the last interior arm; that arm applies `configure_allocation`,
  re-derives its own proposal from its output caps, and replies
  `CoordinatorEvent::ArmProposal { index, .. }`; the coordinator forwards one
  hop further up (to `index - 1`), terminating at index 0 (the source is not an
  interruptible arm). No blocking walk, so no interleaving hazard.
- `Coordinator` generalized from one `transform_ctrl` to a `Vec` of per-arm
  control channels; `coordinator_with_recascade_n(capacity, n)` builds the N-arm
  variant. `run_source_transform_sink` is unchanged: it still uses the 1-arm
  `coordinator_with_recascade`, whose lone transform never replies, so its
  cascade stays exactly one hop. Interior arms became interruptible (`select2`
  over the control receiver + data link) and track their output-link caps to
  re-derive proposals; the sink arm reports `CapsChanged { proposal }`; the
  coordinator joins as the last arm and its observed count surfaces on
  `RunStats.coordinator_events`.
- Scope: the cascade applies during data flow; one triggered in the final
  frames before EOS is best-effort (interior arms exit on EOS), which is correct
  for a live stream that never reaches a final frame. The α (element-local)
  re-allocation still fires independently when an interior element forwards a
  `CapsChanged`, so a received change configures an interior element three
  times: startup (M12 fold) + α (own pool) + β (downstream neighbour's proposal).
- Tests: two coordinator unit tests (`n_hop_cascade_walks_upstream_one_hop_per_reply`,
  `..._stops_when_a_reply_has_no_proposal`) drive the reactive routing directly;
  `m18_beta_nhop.rs` proves the end-to-end walk with distinct per-element
  allocation markers (sink=100, t1=200, t0=300), so the recorded
  `[startup, α, β]` sequence pins which neighbour's proposal arrived when (t1
  ends on the sink's 100, t0 on t1's 200). Verified deterministic across 6
  runs. VERIFIED: `cargo test --workspace` green, single-hop β
  (`m18_beta_recascade`) and the data-plane (`m18_multi_element`) unaffected,
  no_std + runtime build, core clippy clean.

### M18 item 7 (complete): bus wiring across every runner

- Finishes item 7. Every runner that routes negotiation through `solve_linear`
  now posts `BusMessage::NegotiationFailed` on failure, via the opt-in
  `_with_bus` twin: `run_simple_pipeline_with_bus`,
  `run_linear_chain_with_bus`, `run_source_fanout_with_bus`,
  `run_muxer_sink_with_bus` (joining the existing
  `run_source_transform_sink_with_bus`), covering both their startup and
  mid-stream re-solve sites (the fan-out branch FO-1 strict failure and the
  muxer per-input MX-1 path included). `run_fanin_sink` is exempt: it
  self-fixates per source (`proposal.fixate()`) with no `solve_linear` chain,
  so it produces no `NegotiationFailure`.
- Shared `report_nego_failure(bus, failure)` helper (coordinator module)
  centralizes the post so every solve site reads
  `solve_linear(..).map_err(|f| { report_nego_failure(bus, f); CapsMismatch })`.
  The earlier inline posts (startup + transform-sink mid-stream) were
  refactored onto it. Each runner keeps a clean non-`bus` public signature; the
  body is an inner fn taking `Option<&BusHandle>`, and a `bus.cloned()` clone
  moves into the relevant arm / muxer task for the mid-stream sites.
- New tests: `simple_pipeline_startup_failure_posts_to_bus` (m18_bus_negotiation)
  and `incompatible_chain_posts_negotiation_failure_to_bus` (m18_multi_element)
  exercise the helper through two more runners; the fan-out / mux startup paths
  share the identical one-line call. VERIFIED: `cargo test --workspace` green,
  no_std + runtime build, core clippy clean.

### M18 item 4: arbitrary-length linear runner (`run_linear_chain`)

- Lifts the "runner caps at 3 elements" limit. The fixed-arity runners
  (`run_simple_pipeline` = 2, `run_source_transform_sink` = 3) couldn't
  express a `source -> decoder -> capsfilter -> converter -> sink` chain;
  `run_linear_chain(source, Vec<&mut dyn DynAsyncElement>, sink, clock, cap)`
  drives any length. Interior elements are `&mut dyn DynAsyncElement` (the
  same erasure the fan-out runner uses), so the chain is heterogeneous;
  source and sink stay statically typed. std-only, like the other dyn
  runners. Closes DESIGN-M16-caps-nego.md §13.4 item 4 (startup + data
  plane; the cross-element re-cascade over N hops is the documented follow-up).
- Negotiation runs `solve_linear` over all `N + 2` constraints at once and
  configures each element with its input-side caps (source with link 0,
  transform `i` with link `i`, sink with the last link). The M12 allocation
  query folds sink -> ... -> source across every hop. Data flows over `N + 1`
  bounded links across `N + 2` arms joined by `join_all`. Each interior
  element handles a mid-stream `CapsChanged` element-locally (re-configure +
  α re-allocation + forward); the sink runs the Phase-B downstream re-solve.
- `DynAsyncElement` gains `caps_constraint_as_transform` (dyn-safe mirror of
  the `AsyncElement` method) so an erased interior element declares its
  transform constraint to the solver; the blanket impl forwards it, and the
  `dyn-slot` test element implements it as `IdentityAny`.
- Scope / owed (extends the single-hop coordinator path of
  `run_source_transform_sink`): the β allocation re-cascade and full
  downstream-subgraph re-solve over N hops; clock election and latency
  aggregation across the `dyn` interior elements (only source and sink
  contribute today); the `ReFixate` startup retry (fails loud like
  `run_source_fanout`).
- New `m18_multi_element.rs` drives real plugins: a 4-element
  `VideoTestSrc -> Identity -> Identity -> FakeSink` flows every frame + EOS;
  a 5-element chain with a real `CapsFilter` mid-chain negotiates and flows;
  zero transforms degenerates to source->sink; and an NV12-only `CapsFilter`
  on an RGBA source fails the whole-chain solve loud (`CapsMismatch`).
  VERIFIED: `cargo test --workspace` green, core `runtime`+`dyn-slot` green,
  no_std baseline and no_std+runtime builds (the runner is std-gated so the
  baseline is unaffected), core clippy clean.

### M18 item 7: structured negotiation failures on the bus

- A failed startup caps negotiation no longer discards the solver's
  structured `NegotiationFailure`. Previously every solve site did
  `solve_linear(...).map_err(|_| G2gError::CapsMismatch)`, collapsing
  "which link conflicted on what" to an opaque error (DESIGN-M16-caps-nego.md
  §13.3). New `BusMessage::NegotiationFailed(NegotiationFailure)` carries the
  detail (e.g. `EmptyLink { upstream, downstream }`) to the application while
  the runner still returns `CapsMismatch` to its caller.
- New `run_source_transform_sink_with_bus(.., bus: &BusHandle)`: as
  `run_source_transform_sink` but posts the structured failure to `bus`
  (non-blocking `try_post`, so a startup failure never stalls on a full bus).
  Opt-in via a separate entry point so the existing runner signature and its
  ~dozen call sites are untouched; the shared body is an inner fn taking
  `Option<&BusHandle>`. `negotiate_source_transform_sink` threads the handle
  and posts at the solve site.
- `NegotiationFailure` gains `Eq` (all-`usize`/unit fields) so it composes
  into `BusMessage`'s derives; re-exported at `g2g_core::NegotiationFailure`.
- New `m18_bus_negotiation.rs`: an RGBA source into an NV12-only sink fails
  with `EmptyLink`, the bus carries it, and the run still errors
  `CapsMismatch`; a clean NV12->NV12 negotiation posts nothing. VERIFIED:
  `cargo test --workspace` green, `cargo test -p g2g-core --features runtime`
  (115) green, no_std + runtime build, core runtime clippy clean.
- Mid-stream completion: `run_source_transform_sink_with_bus` now also posts
  `NegotiationFailed` when the sink arm's mid-stream `re_solve_downstream_sink`
  rejects a boundary's `CapsChanged`. This is the case the bus matters most
  for, an async failure deep in an arm with no synchronous return to carry the
  detail; the run still drains to EOS (the rejected change never takes
  effect). New `mid_stream_rejected_capschange_posts_to_bus` test (an RGBA
  `CapsChanged` into an NV12-only sink posts `EmptyLink`).
- Still owed: `run_simple_pipeline`, `run_linear_chain`, and the non-linear
  runners (fan-out, fan-in/mux) discard their `NegotiationFailure` the same
  way; each needs the identical opt-in `_with_bus` wiring (mechanical, their
  startup failures already return `Err` synchronously).

### M18 item 1 (Session E): β allocation re-cascade (single hop)

- First cross-element mid-stream allocation cascade. Previously the M12
  allocation cascade ran only at startup, and mid-stream only α
  (element-local) re-derived an element's own pool. β now flows the sink's
  re-derived `propose_allocation` answer one hop upstream to the transform's
  `configure_allocation` on a mid-stream `CapsChanged`, the same step the
  startup cascade does once at setup, re-run when geometry changes. This is
  what GPU pool chains (D3D11 / CUDA / DMABUF / VAAPI) need to re-size the
  upstream pool before the first frame under the new caps
  (DESIGN-M16-workaround3-reconfigure.md §9.4 β, §9.4.1).
- New `select2` no_std combinator (`runtime::join`): polls two futures,
  resolves on the first ready, drops the loser. Sound because a channel
  `recv()` future holds no dequeued message, so dropping a pending recv
  loses nothing (unit-tested as the drop-safety proof). This is the
  interruptibility primitive §9.4.1 named as missing: a runner arm can race
  its data link against an out-of-band control channel, so a directive
  reaches it while it is parked on `recv().await`. Without it the runtime
  could only `join` (wait for all), never `select` (wait for first).
- The coordinator (R2 single task) now owns the re-cascade.
  `CoordinatorEvent::CapsChanged` carries the sink's `proposal`;
  `coordinator_with_recascade` adds a control channel to the transform arm;
  `Coordinator::run` forwards an `ArmDirective::Recascade(params)` for each
  proposal (serial, so the bounded control channel never blocks). The
  transform arm selects on that control receiver alongside its data link and
  applies `configure_allocation` on receipt. On EOS the arm drains the
  control channel until the coordinator closes it, so a tail-end directive
  still in flight at shutdown is applied deterministically (in a live stream
  these apply inline as they arrive). `realloc_local` now returns the
  element's proposal so the sink arm forwards the same value α stored.
- Scope: single hop (sink -> transform), the correct re-cascade for a
  `source -> transform -> sink` chain because a link2 `CapsChanged` (post-
  decode geometry) affects only the transform's output pool; link1
  (pre-decode) is unchanged. The source leg and the general N-hop downstream
  subgraph re-cascade belong to the multi-element runner (§13.4 item 4).
- Tests: three coordinator unit tests (forwards a proposal as a directive;
  no proposal forwards nothing; observe-only coordinator never forwards)
  driving the real `Coordinator::run` on a no_std busy-poll executor; four
  `select2` unit tests including the drop-safety proof; and
  `m18_beta_recascade.rs` end-to-end (a fake transform records
  `[startup_proposal, β_proposal]` on a mid-stream geometry change, and
  `[startup_proposal]` only without one). VERIFIED: `cargo test --workspace`
  green, `cargo test -p g2g-core --features runtime` (115) green, no_std +
  runtime build, and core runtime clippy clean.

### M16 step 5: migrate the display sinks to native `caps_constraint`

- Completes the §8 step-5 element migration. The three present sinks,
  `WaylandSink`, `KmsSink`, and `D3D11Sink`, previously rode the `LegacySink`
  bridge (pass-through `intercept_caps`, NV12 enforced only in
  `configure_pipeline`). Each now overrides
  `caps_constraint_as_sink() -> Accepts(CapsSet::one(NV12 / any geometry))`,
  so a fully-native decoder -> sink chain narrows NV12 in the solver's arc
  consistency rather than via the dynamic intercept callback. Geometry stays
  open (`Dim::Any`); the upstream decoder's `DerivedOutput` fixates it.
- Side effect: the ACCEPT_CAPS query is now truthful for these sinks. As
  `LegacySink(passthrough)` their `CapsConstraint::accepts` always returned
  `true` (the intercept clones any input), so a query wrongly reported an
  NV12-only sink would accept H.264. `Accepts(NV12)` reports it correctly.
- `InterleaveMux` was already native (`caps_constraint_as_input` /
  `caps_constraint_for_output`, M18 step 1), so no legacy `MultiInputElement`
  remains. The legacy bridge variants (`LegacySource` / `LegacyTransform` /
  `LegacySink`) stay in `CapsConstraint` as the default for any
  not-yet-overridden element, but no in-tree element relies on them now.
- One GPU-free unit test per sink (`caps_constraint_is_accepts_nv12_any`)
  locks the native shape. VERIFIED on the Windows dev host:
  `cargo test --workspace` (default) green, and
  `cargo test -p g2g-plugins --features "mf-decode d3d11-sink"` green with the
  new `D3D11Sink` test passing. `WaylandSink` / `KmsSink` are Linux-gated and
  not compiled here; their edits mirror the `D3D11Sink` pattern verbatim and
  are owed a Linux compile, same as the rest of that path.

### M16 step 6 audit: CapsFilter + ACCEPT_CAPS confirmed fully wired

- No code change. Verified §8 step 6 is complete and correct:
  `CapsConstraint::accepts` / `CapsSet::accepts` (the §7 ACCEPT_CAPS predicate)
  have real consumers (`CapsFilter::configure_pipeline` + mid-stream
  validation, `CudaDownload::configure_pipeline`), not just tests; the
  `accepts` match is exhaustive over all ten `CapsConstraint` variants
  (`DerivedOutput` checks the derived output is non-empty, the legacy bridges
  defer to their wrapped callbacks). `CapsFilter` is the native `Identity(set)`
  pass-through and is integration-tested in a real solver chain
  (`pipeline_smoke.rs`: matching filter negotiates, disjoint filter fails).

### M12: allocation query runs before `configure_pipeline` (pool sizing)

- The M12 allocation query for a `source -> transform -> sink` chain now runs
  *inside* `negotiate_source_transform_sink`, between `solve_linear` and the
  `configure_pipeline` cascade, instead of in the runner *after* negotiation.
  This lets a transform (a hardware decoder) size its buffer pool from the
  downstream consumer's `min_buffers` when it opens the codec, rather than
  after the pool is already fixed. Behavior-preserving for existing pipelines:
  sources build their pools in `run()` (after both calls either way), and the
  folded source-facing proposal still surfaces on `RunStats.allocation`.
- `LinearNegotiation` gained an `allocation` field carrying that proposal; the
  query (sink -> transform fold -> source) moved verbatim, only earlier. The
  `ReFixate` retry re-runs it each attempt, so the recorded params always match
  the final negotiated caps. `run_simple_pipeline` (source -> sink, no
  transform) is unchanged; its source allocates in `run()`, so the order
  doesn't matter there.
- `FfmpegH264Dec` (`NvdecCuda`) now sizes the CUDA hwframe pool's
  `extra_hw_frames` from the recorded `min_buffers` (`min_buffers + 4` reorder
  margin), falling back to the previous fixed `8` when no consumer proposed.
  Closes the optimization the C3 allocation-handshake notes flagged as deferred
  on the ordering. (`MfDecode`'s MFT manages its own output texture pool, so
  there `min_buffers` stays informational.)
- New runner test `transform_allocation_precedes_configure_pipeline`
  (`m12_allocation.rs`): a fake transform records, at `configure_pipeline`
  time, that its `configure_allocation` already ran, the regression guard for
  the ordering. VERIFIED: `cargo test -p g2g-core` (49) + `m12_allocation` (6),
  full workspace, workspace clippy, and the no_std core baseline all green. The
  `FfmpegH264Dec` change is `ffmpeg`-feature + Linux-only and is owed a Linux
  compile, same as the rest of that path.

### M18 item 6: pad templates for the Windows decode/display elements

- `MfDecode` and `D3D11Sink` now implement `PadTemplates`, so a tool can
  introspect their static caps and check link compatibility before either is
  constructed (`gst_element_factory_get_static_pad_templates` analog). Extends
  the existing coverage (`VideoTestSrc` / `FakeSink` / `H264Parse`) to the
  Windows GPU path. `MfDecode`: H.264 sink pad + NV12 source pad (the memory
  domain is not encoded in caps, so the templates are backend-independent).
  `D3D11Sink`: a terminal NV12 sink pad, no source pad.
- New integration test `windows_decode_to_display_chain_links_by_type`
  (gated on `mf-decode` + `d3d11-sink`) proves the whole chain is
  introspectable pre-instantiation: `H264Parse -> MfDecode -> D3D11Sink` all
  link by type, while an RGBA source is correctly rejected at the decoder.
  Plus element-local unit tests for each template. VERIFIED on the Windows dev
  host: `cargo test -p g2g-plugins --features "mf-decode d3d11-sink"` (34 lib +
  the chain test) and clippy green; default workspace unaffected.

### W1: allocation-query handshake for the D3D11 path

- Mirrors C3 step 3 on the Windows side, completing the W1 <-> C3 symmetry.
  `D3D11Sink::propose_allocation` returns `AllocationParams::d3d11(...)`: a
  `MemoryDomainKind::D3D11Texture` proposal sized to the NV12 frame
  (`w*h*3/2`, even dims guaranteed by the sink), with pool headroom
  (`min_buffers = 3`) and 256-byte alignment. The runner conveys it to the
  upstream decoder's `configure_allocation`.
- `MfDecode::configure_allocation` records the proposal (`requested_alloc()`
  accessor), like `FfmpegH264Dec`. On the `with_d3d11` path a texture request
  is honoured by construction (it already emits `D3D11Texture` frames); the
  software path can't satisfy it, so the request stays recorded for diagnostics
  rather than silently changing the output domain.
- Two GPU-free unit tests (`D3D11Sink` proposes the right size/align/headroom;
  `MfDecode` records a conveyed D3D11 proposal). VERIFIED on the Windows dev
  host: `cargo test -p g2g-plugins --features "mf-decode d3d11-sink"` (32
  passed) and clippy green; the platform-agnostic conveyance stays covered by
  `m12_allocation.rs`.

### W1 (Phase 4): `D3D11Sink` present sink

- Completes the Windows zero-copy decode -> display track. `D3D11Sink`
  (`d3d11-sink` feature, Windows-only) consumes `MemoryDomain::D3D11Texture`
  frames from `MfDecode::with_d3d11()` and presents them in a Win32 window via
  a DXGI flip-model swapchain. The NV12 -> RGB colour convert runs on the GPU
  through a D3D11 video processor (`VideoProcessorBlt`), so the decoded texture
  never leaves the GPU. The Windows analog of `CudaGlSink`.
- Same worker-thread model as `WaylandSink` / `CudaGlSink`: a dedicated thread
  owns the window + message pump + D3D11 objects (both thread-affine); the sink
  struct holds only `Send` handles (an mpsc sender + atomics). The decoded
  `OwnedD3D11Texture` is `Send`, so it crosses to the worker and the texture
  (and its owning `IMFSample`) stays pinned until presented. NV12-in-D3D11
  only (`UnsupportedDomain` otherwise); per-frame ack gives backpressure;
  mid-stream geometry change respawns the worker.
- The swapchain + video processor are created lazily on the first frame from
  that frame's `ID3D11Device` (the decoder's device), since a D3D11 resource
  and the views over it must share a device, avoiding a second device + texture
  sharing. The window is created up front (no device needed).
- All COM (D3D11 video device/context/processor/views, the input/output view
  descriptors, the DXGI swapchain, and the Win32 window + message loop) was
  verified against the fetched `windows-0.62.2` source per AGENTS.md. Adds
  `windows` features `Win32_UI_WindowsAndMessaging`, `Win32_System_LibraryLoader`,
  `Win32_Graphics_Gdi`.
- Three GPU-free unit tests (intercept pass-through, non-NV12 reject, odd-dim
  reject). VERIFIED on the Windows dev host: `cargo test -p g2g-plugins
  --features d3d11-sink` (19 passed, the full present path COMPILES) and
  `cargo clippy --features d3d11-sink` green. The actual present (real GPU
  decode into textures shown in a window) is owed as a user-side run on a GPU
  machine; the dev host can do it. Acceptance test: `rtspsrc -> h264parse ->
  mfdecode[with_d3d11] -> d3d11sink`, a visible window with decoded video.

### W1 (Phase 3): `MfDecode` zero-copy D3D11 texture output

- Completes the Windows zero-copy decode track. With `with_d3d11()`, `MfDecode`
  now emits `MemoryDomain::D3D11Texture` frames instead of reading the decoded
  pixels back to system memory: the decoded NV12 stays in the GPU texture, so a
  DXGI / D3D11 consumer (a future swapchain present sink) takes the handoff
  without a GPU->CPU copy. The Windows analog of C3's `NvdecCuda` output.
- `extract_texture` pulls the `ID3D11Texture2D` and its subresource index out of
  the DXVA output sample's `IMFDXGIBuffer` (`GetResource` +
  `GetSubresourceIndex`) and wraps them in an `OwnedD3D11Texture`. The
  keep-alive (`SampleOwner`) owns the `IMFSample`, so the texture stays valid
  until the consumer drops the frame, then the sample returns to the decoder's
  output texture pool. `Send` under the same MTA contract as the decoder.
- `DecodedPicture` gained a `DecodedPayload` enum (`System(Box<[u8]>)` vs
  `D3D11(OwnedD3D11Texture)`); `process_output` branches on the active device
  (texture extraction on the DXVA path, the Phase-2 system readback otherwise),
  and `process` maps the payload to the frame's `MemoryDomain`. The software
  path is byte-identical.
- VERIFIED on the Windows dev host: `cargo test -p g2g-plugins --features
  mf-decode` (27 passed, the full D3D11 texture path COMPILES) and
  `cargo clippy --features mf-decode` green. The DXVA decode runtime (GPU decode
  of a real H.264 stream into textures, and a D3D11 consumer to display them) is
  owed as a user-side run on a GPU machine; the dev host can do it. The present
  sink (the D3D11 consumer, analog of `CudaGlSink`) is the next phase.

### W1 (Phase 2): `MfDecode` DXVA / D3D11 hardware decode

- `MfDecode::with_d3d11()` opts the Windows decoder into DXVA hardware decode.
  `configure_pipeline` then creates a hardware D3D11 device
  (`D3D11CreateDevice`, `D3D_DRIVER_TYPE_HARDWARE`, video support, multithread
  protection on, which Media Foundation requires) and a Media Foundation DXGI
  device manager (`MFCreateDXGIDeviceManager` + `ResetDevice`), and hands it to
  the MFT via `MFT_MESSAGE_SET_D3D_MANAGER` before setting the media types. The
  decode then runs on the GPU. Default stays the MS software decoder.
- The sync `CLSID_MSH264DecoderMFT` does DXVA in-place when given a D3D
  manager, so no async-MFT / `MFTEnumEx` event loop is needed: the existing
  synchronous `ProcessInput`/`ProcessOutput` drain still drives it. With a D3D
  manager the MFT allocates its own output samples
  (`MFT_OUTPUT_STREAM_PROVIDES_SAMPLES`); `process_output` detects that flag
  and passes a null sample so the MFT fills it, instead of pre-allocating a
  system buffer. The software path is byte-identical to before.
- Phase 2 reads the (D3D11-backed) output sample back to packed system NV12 via
  the existing `copy_sample`, so every current sink keeps working with
  hardware decode. The zero-copy `MemoryDomain::D3D11Texture` output (no
  readback) is Phase 3.
- `DecoderState` holds the D3D11 device and DXGI manager for the decoder's
  lifetime; the device outlives every output sample. New GPU-free builder unit
  test (`with_d3d11` toggles the opt-in). VERIFIED on the Windows dev host:
  `cargo test -p g2g-plugins --features mf-decode` (27 passed, the D3D11 path
  COMPILES) and `cargo clippy --features mf-decode` green. The actual DXVA
  decode runtime (device creation + GPU decode of a real H.264 stream) is owed
  as a user-side run on a machine with a GPU; the Windows dev host can do it.
- Adds `windows` crate features `Win32_Graphics_Direct3D11`,
  `Win32_Graphics_Direct3D`, `Win32_Graphics_Dxgi`, `Win32_Graphics_Dxgi_Common`.

### W1 (Phase 1): Direct3D 11 memory domain foundation

- First phase of the Windows zero-copy decode -> display track, the analog of
  the C3 CUDA track for the `MfDecode` path. Goal: keep DXVA-decoded NV12
  resident in a D3D11 texture instead of copying it to system memory, so a
  DXGI / D3D11 consumer (a swapchain present sink) takes the handoff without a
  GPU->CPU copy. Phase 1 lands only the platform-agnostic core types (compiles
  and tests on the Windows dev host); the `MfDecode` DXVA output (Phase 2) and
  the present sink (Phase 3) follow.
- New `MemoryDomain::D3D11Texture(OwnedD3D11Texture)` variant and matching
  `MemoryDomainKind::D3D11Texture`, mirroring the handle-based `Cuda` variant
  (no `windows`-crate link in `no_std` core). `OwnedD3D11Texture` carries the
  `ID3D11Texture2D` pointer, the subresource index of this frame within it
  (DXVA decoders hand out a texture *array* whose subresources are the decoded
  surfaces), the visible dims, the `DXGI_FORMAT`, the `ID3D11Device`, and a
  keep-alive owner.
- `D3D11KeepAlive` trait (`Debug + Send`): the producing element boxes its
  owning handle (a Media Foundation `IMFSample` / `IMFDXGIBuffer`) as a trait
  object so core never links the `windows` crate; dropping the buffer drops
  the box and releases the sample back to the decoder.
- `AllocationParams::d3d11(size, count, align)` constructor, the Windows analog
  of `AllocationParams::cuda`, so a DXGI consumer can request texture-resident
  buffers via the M12 allocation query.
- Two unit tests in `memory.rs` (`kind()` maps the new variant; a `FlagOnDrop`
  keep-alive proves the backing texture is released exactly when the buffer
  drops). VERIFIED on the Windows dev host: full workspace tests (114 core
  passed), workspace clippy, and the no_std baseline all green.

### M13: `MfDecode` NV12 stride handling (Windows)

- `MfDecode` no longer assumes the Media Foundation decoder's NV12 output is
  tightly packed. The MFT can report an `MF_MT_DEFAULT_STRIDE` larger than the
  frame width (hardware MFTs align rows up; the MS software decoder packs
  tightly), in which case the contiguous output buffer carries per-row padding
  that would feed garbage to the packed-NV12 sinks (`WaylandSink`, `KmsSink`).
- `set_nv12_output` now also reads the output type's `MF_MT_DEFAULT_STRIDE`
  (floored at `width`, so a missing or bottom-up value degrades to the
  packed assumption) and caches it on `DecoderState`. `copy_sample` strips the
  padding via a new pure `pack_nv12(src, width, height, stride)` that copies
  `width` bytes from each of the `height + height/2` source rows (Y plane +
  half-height interleaved UV) into a tightly-packed `width*height*3/2` buffer.
  `stride == width` stays a row-wise copy; a short source buffer leaves the
  tail zeroed rather than panicking.
- Four GPU-free unit tests for `pack_nv12` (packed identity, stride
  de-padding, bad-geometry reject, short-source fail-safe). VERIFIED on the
  Windows dev host: `cargo test -p g2g-plugins --features mf-decode` (26
  passed) and `cargo clippy --features mf-decode` green.

### C3 (Phase 3, step 3): allocation-query handshake for the CUDA path

- Completes the C3 roadmap (DESIGN-C3-cuda.md §4 step 3): wires the M12
  allocation query so the GPU sink declares it wants device memory and the
  CUDA decoder records the request. The cross-element conveyance machinery
  itself already existed (and is tested generically in `m12_allocation.rs`,
  including the two CUDA-domain cases); this connects the real elements.
- `CudaGlSink::propose_allocation` returns `AllocationParams::cuda(...)`: a
  `MemoryDomainKind::Cuda` proposal sized to the NV12 frame
  (`cuda::nv12_byte_size`), with pool headroom (`min_buffers = 3`: the frame
  on the GL thread plus the one the runner link holds) and 256-byte GPU
  alignment. Returns `None` until the geometry is fixed. The runner conveys
  this to the upstream decoder's `configure_allocation`.
- `FfmpegH264Dec::configure_allocation` records the proposal
  (`requested_alloc()` accessor). On the `NvdecCuda` backend a Cuda request is
  honoured by construction (it already emits device-resident frames); the
  system-memory backends can't satisfy it, so the request stays recorded for
  diagnostics rather than silently changing the output domain. The runner
  calls `configure_allocation` *after* `configure_pipeline` (the decoder is
  already open), so the recorded `min_buffers` is not yet used to size the
  hwframe pool's `extra_hw_frames` (still the fixed `8` from Phase 2); doing so
  is a future optimization that needs the allocation query moved ahead of
  decoder open, noted in the field docs.
- New GPU-free unit tests: `cuda::nv12_byte_size` (even/odd dims),
  `CudaGlSink` proposes Cuda with the right size/align/headroom, and
  `FfmpegH264Dec` records a conveyed Cuda proposal. These run on Linux under
  `--features cuda-gl` / `--features ffmpeg` (the modules are Linux-gated);
  the platform-agnostic conveyance remains covered on the Windows host by
  `m12_allocation.rs`. Default workspace build/test/clippy green.

### C3 (Phase 3, step 2b): `CudaGlSink` (first draft, owed first compile)

- `g2g-plugins::cudaglsink::CudaGlSink` behind a new `cuda-gl` feature
  (implies `cuda`; Linux + NVIDIA). The zero-copy-ish display payoff: keeps
  `Backend::NvdecCuda` decoded NV12 on the GPU and presents it via CUDA-GL
  interop (device->texture copy + NV12->RGB fragment shader) on a Wayland EGL
  surface, removing both the device->host copy `CudaDownload` pays and the CPU
  colour convert `WaylandSink` pays (DESIGN-C3-cuda.md §3.2, §4 step 2).
- `CudaGlInterop` (in `g2g-plugins::cuda`, gated `cuda-gl`) is the CUDA side:
  registers the two GL textures once (`cuGraphicsGLRegisterImage`,
  write-discard), then per frame maps them, `cuMemcpy2D`s each NV12 plane
  device->`cudaArray`, and unmaps; unregisters on drop. This consumes the
  step-2a interop FFI. `make_context_current` pushes the ffmpeg CUDA context
  onto the sink's GL worker thread.
- Sink structure mirrors the proven `WaylandSink`: a dedicated worker thread
  owns the Wayland connection + EGL/GL context (both thread-affine); the sink
  struct holds only `Send` handles (a `calloop` channel + atomics). The
  decoded `OwnedCudaBuffer` is `Send`, so it crosses to the worker and the
  device frame stays pinned until presented. NV12-in-CUDA only (a
  system-memory frame is rejected `UnsupportedDomain`); per-frame ack gives
  compositor-paced backpressure; mid-stream geometry change respawns the
  worker (M16 5j).
- New deps under the `cuda-gl` feature (Linux-gated, verified current):
  `khronos-egl` 6 (`static`, links libEGL), `glow` 0.17 (GL ES wrapper),
  `wayland-egl` 0.32 (`wl_egl_window` from the SCTK surface). The Appendix A
  vertex + fragment shaders (from step 2a) drive a fullscreen quad with two
  NV12 textures (luma `R8`, chroma `RG8`), GL ES 3 for the single/two-channel
  formats.
- THREE GPU-free unit tests (intercept pass-through, non-NV12 reject, odd-dim
  reject) lock the negotiation surface.
- VERIFICATION: this is a FIRST DRAFT. The module is `cuda-gl` + Linux +
  NVIDIA-gated and is NOT compiled on the Windows dev host; the EGL/GL/Wayland
  worker is owed a first compile and an e2e on the Linux+GPU box. The crate-API
  spots most likely to need a small fixup carry inline `// VERIFY:` notes (the
  `wl_display` / `wl_surface` raw-pointer accessors on `wayland-client` 0.31,
  glow 0.17's `tex_image_2d` pixel-source parameter, and the
  `eglGetProcAddress` cast for glow's loader). Acceptance test: a
  `wayland_smoke`-style benchmark `rtspsrc -> h264parse ->
  ffmpegdec[NvdecCuda] -> CudaGlSink`, p50/p95 versus the `NvdecCuvid ->
  WaylandSink` system-memory baseline. Default workspace build/test/clippy and
  no_std core baseline remain green.

### C3 (Phase 3, step 2a): CUDA-GL interop foundation

- Groundwork for `CudaGlSink` (DESIGN-C3-cuda.md §3.2, §4 step 2), the real
  zero-copy-ish display path: decoded NV12 stays on the GPU and is presented
  via CUDA-GL interop (`cuGraphicsMapResources` -> mapped `cudaArray` ->
  device->array `cuMemcpy2D` -> fragment-shader YCbCr->RGB), removing the
  PCIe round-trip and the CPU colour convert. Staged ahead of the EGL/Wayland
  windowing (step 2b) so the verifiable pieces land first, mirroring how
  Phase 1's core types preceded Phase 2's FFI.
- Extends the `g2g-plugins::cuda` FFI (behind the existing `cuda` feature,
  Linux + NVIDIA) with the CUDA-GL interop entry points, verified against the
  CUDA Driver API docs (`CUDA_GL` / `CUDA_GRAPHICS` groups):
  `cuGraphicsGLRegisterImage`, `cuGraphicsUnregisterResource`,
  `cuGraphicsMapResources`, `cuGraphicsUnmapResources`,
  `cuGraphicsSubResourceGetMappedArray`, plus the `CU_MEMORYTYPE_ARRAY` /
  `WRITE_DISCARD` / `GL_TEXTURE_2D` constants and opaque handle aliases. These
  live in `libcuda` itself, so no extra link. Marked `#[allow(dead_code)]`
  until step 2b calls them.
- `nv12_gl_uploads(width, height)` computes the per-plane device->`cudaArray`
  copy extents: a full-res `R8` luma texture (1 byte/texel) and a half-res
  `RG8` interleaved CbCr texture (2 bytes/texel). Pure geometry, unit-tested
  for even and odd dims.
- The NV12->RGB fragment shader and its paired vertex shader land verbatim
  from DESIGN-C3-cuda.md Appendix A as `FRAGMENT_SHADER_NV12` / `VERTEX_SHADER`
  consts (GLSL ES 1.00, BT.601 limited range). A test locks the `y_tex` /
  `uv_tex` sampler contract the CUDA upload side depends on.
- VERIFICATION: the module is `cuda`-feature + Linux + NVIDIA-gated and does
  not compile on the Windows dev host; the GPU-free unit tests (plane extents,
  shader contract) run under `cargo test -p g2g-plugins --features cuda` on
  the Linux box. Step 2b (EGL context on a Wayland surface, GL program, the
  map/copy/unmap render loop) is the remaining lift and needs an EGL/GL
  binding-crate decision. Default workspace build/clippy green.

### C3 (Phase 3, step 1): `CudaDownload` device->host bring-up element

- New `g2g-plugins::cuda::CudaDownload` transform behind a new `cuda`
  feature (Linux + NVIDIA, implies std). Copies a `Backend::NvdecCuda`
  `MemoryDomain::Cuda` NV12 frame back to system memory (device->host
  `cuMemcpy2D`, honouring the device-side row pitch) so a CUDA-resident
  stream reaches the existing CPU sinks (`WaylandSink` / `KmsSink`). It
  negates the zero-copy latency win, but it makes the `NvdecCuda` decode
  path end-to-end usable and testable (frame counts, geometry) before the
  real `CudaGlSink` exists. This is the low-risk Phase 3 bring-up element
  (DESIGN-C3-cuda.md §3.4, §4 step 1).
- Caps surface is `Identity(NV12)` with open geometry: input and output are
  the same NV12 description (caps do not encode the memory domain), so the
  element drops into any `NvdecCuda -> sink` chain without changing
  negotiation; only the frame's domain changes (`Cuda -> System`). A frame
  already in system memory passes through untouched, so it is a safe no-op
  on the `Software` / `NvdecCuvid` backends.
- Packed NV12 destination: luma plane (`width*height`) then interleaved
  chroma (`2*ceil(w/2)*ceil(h/2)`); for even dims the standard
  `width*height*3/2`. The per-plane copy descriptors (`nv12_plane_copies`)
  are pure geometry, unit-tested for even and odd dimensions without a GPU.
- CUDA bindings: thin hand-rolled FFI linking `libcuda` directly (no crate
  dep), per DESIGN-C3-cuda.md §6 (`cudarc` has no GL-interop wrappers and
  fights the foreign-`CUcontext` ownership). Exactly the surface this
  element needs: `cuCtxPushCurrent_v2` / `cuCtxPopCurrent_v2` (push the
  ffmpeg-owned context the pointers are valid in, always pop) and
  `cuMemcpy2D_v2`. `#[repr(C)] CudaMemcpy2D` mirrors `cuda.h` field-for-field
  (verified against the CUDA Driver API docs). Every `unsafe` block carries
  a `// SAFETY:` note.
- New `HardwareError::Cuda(i32)` variant carries the raw `CUresult` on a
  driver failure, mirroring `Vulkan(i32)` / `MediaFoundation(i32)`. Core
  change; compiles in std and no_std.
- Five GPU-free unit tests (`Identity(NV12)` constraint, non-NV12 reject,
  intercept narrowing, even- and odd-dimension plane packing).
- VERIFICATION: the device-copy path is `cuda`-feature + Linux + NVIDIA-GPU
  only and does not compile on the Windows dev host; first-compile and the
  e2e (`rtspsrc -> h264parse -> ffmpegdec[NvdecCuda] -> CudaDownload ->
  WaylandSink`) are owed on the Linux+GPU box, same as the rest of C3. The
  default workspace build/test/clippy and the no_std core baseline are green.

### C3 (Phase 2): NVDEC CUDA hwframe output (`Backend::NvdecCuda`)

- New `FfmpegH264Dec` backend that keeps decoded NV12 resident in GPU
  memory. Unlike `Backend::NvdecCuvid` (standalone `h264_cuvid` codec,
  copies NV12 back to system memory), `NvdecCuda` attaches an
  `AV_HWDEVICE_TYPE_CUDA` device to the generic `h264` decoder and
  installs a `get_format` hook selecting `AV_PIX_FMT_CUDA`, the canonical
  `hw_decode.c` true-hwaccel pattern. Decoded frames are emitted as
  `MemoryDomain::Cuda` carrying the two NV12 plane device pointers, row
  pitches, dims, and the `CUcontext`; the owning `AVFrame` is boxed as the
  buffer's `CudaKeepAlive`, so the device memory is released back to the
  hwframe pool exactly when a downstream consumer drops the frame. Removes
  cuvid's device->host copy; the latency payoff lands once a CUDA-consuming
  sink (Phase 3) takes the handoff copy-free.
- Output is always NV12 (the device frame's native layout):
  `with_backend(NvdecCuda)` pins `OutputFormat::Nv12` and `configure_pipeline`
  rejects an I420 request loud (a GPU colour convert would be needed, out of
  scope). Negotiation surface is unchanged (H.264 in, NV12 out, same
  geometry) so chains compose identically to the other backends; only the
  emitted frame's memory domain differs, which caps do not encode.
- `extra_hw_frames = 8` gives the decoder pool headroom for the frames held
  in flight downstream (link capacity) plus its own reference set, so the
  pool does not starve under `LatencyProfile::Live`. `low_delay` is on (as
  for cuvid); the cuvid-private `surfaces` knob does not apply to the
  generic hwaccel.
- The decode loop allocates a fresh `AVFrame` per drained picture (the CUDA
  path moves the whole frame into the keep-alive, so it cannot be reused
  scratch); the system-memory paths (`Software`, `NvdecCuvid`) are otherwise
  byte-identical. `DecodedPicture` gained a `DecodedPayload` enum
  (`System(Box<[u8]>)` vs `Cuda(OwnedCudaBuffer)`).
- Raw FFI via `ffmpeg_next::ffi` (re-exported `ffmpeg-sys-next`):
  `av_hwdevice_ctx_create`, `av_buffer_ref`/`unref`, a `get_cuda_format`
  C callback, and an `AVCUDADeviceContextHead` `#[repr(C)]` mirror of the
  public `AVCUDADeviceContext` head (read the `CUcontext` without depending
  on ffmpeg-sys-next having bound the optional CUDA header). Every `unsafe`
  block carries a `// SAFETY:` note; `CudaFrameOwner` asserts `Send` under
  the same ownership-transfer contract as the decoder.
- Three GPU-free unit tests lock the builder surface (NvdecCuda forces NV12
  + low-delay, overrides a prior I420, and its caps constraint is NV12).
- VERIFICATION: this path is `ffmpeg`-feature + Linux + NVIDIA-GPU only and
  does not compile on the Windows dev host; first-compile and the e2e decode
  are owed on the Linux+GPU box, same as the existing rtsp/kms/wayland code.

### C3 (Phase 1): CUDA memory domain foundation

- First phase of the zero-copy NVDEC -> GPU display track. Goal: keep
  `Backend::NvdecCuvid`'s decoded NV12 resident in CUDA device memory
  instead of cuvid's default device->host copy, so a GPU consumer
  (display scanout) takes the handoff copy-free. Phase 1 lands only the
  platform-agnostic core types (compiles and tests on the Windows dev
  host); the ffmpeg hwframe output (Phase 2) and the KmsSink CUDA
  consumer (Phase 3) follow.
- New `MemoryDomain::Cuda(OwnedCudaBuffer)` variant and matching
  `MemoryDomainKind::Cuda`, mirroring the existing handle-based
  `VulkanTexture` / `WebGPUBuffer` variants (no CUDA link in `no_std`
  core). `OwnedCudaBuffer` carries the two NV12 plane device pointers
  (luma Y, interleaved chroma UV) with row pitches, visible dims, the
  `CUcontext` the pointers are valid in, and a keep-alive owner.
- `CudaKeepAlive` trait (`Debug + Send`): the producing element boxes
  its owning handle (an ffmpeg `CUDA`-hwframe `AVFrame`) as a trait
  object so core never links CUDA; dropping the buffer drops the box and
  releases the allocation back to the hwframe pool. Pointers stay valid
  for exactly the keep-alive's lifetime.
- `AllocationParams::cuda(size, count, align)` constructor so a GPU
  consumer can request device-resident buffers via the M12 allocation
  query. This makes `MemoryDomainKind::Cuda` the first cross-element
  pool domain that crosses a real producer/consumer boundary, the
  driver M18 item 1 (allocation re-cascade beta) was waiting on.
- Two unit tests in `memory.rs`: `kind()` maps the new variant, and a
  `FlagOnDrop` keep-alive proves the backing allocation is released
  exactly when the buffer drops. no_std baseline (`--no-default-features`),
  default core build, full workspace tests, and workspace clippy all
  green; the change touches only `g2g-core` (`memory.rs`, `query.rs`,
  `lib.rs`).
- Two GPU-free runner tests in `m12_allocation.rs` prove the wiring claim:
  the allocation query conveys a consumer's `MemoryDomainKind::Cuda`
  proposal end-to-end to the producer, and the CUDA domain survives a
  transform fold (most-demanding size/align win, consumer dictates the
  domain). These exercise the real linear runners with fake elements and
  run on the Windows host.

### LatencyProfile knob on runners

- New `LatencyProfile` enum + `LinkCapacity` newtype in
  `g2g_core::runtime`. Runner signatures change from
  `link_capacity: usize` to `link_capacity: impl Into<LinkCapacity>`,
  so callers express intent (`LatencyProfile::Live`,
  `LatencyProfile::Throughput`, `LatencyProfile::Custom(n)`) instead of
  remembering the steady-state floor formula `2 * cap * frame_period`.
  Test code keeps working — `4usize` still composes via
  `From<usize> for LinkCapacity`.
- `LatencyProfile::Live` -> capacity 2 (~67 ms floor at 60 fps; RTSP ->
  decode -> display). `Throughput` -> 8 (~267 ms; batch / file
  ingest). `Custom(n)` -> caller picks; clamps 0 to 1 so a misconfigured
  env var doesn't deadlock the producer.
- `wayland_smoke` defaults to `LatencyProfile::Live` (was hard-coded
  to 8 with `G2G_LINK_CAP` override). The env var is now a
  bisection-tooling override, not the only path to a low-latency run.
  The recipe in `project_wayland_smoke_recipe.md` no longer needs
  `G2G_LINK_CAP=1` to express live intent.
- 5 unit tests in `g2g-core/src/runtime/runner.rs` (`profile_tests`)
  lock the profile -> capacity mapping, the zero-clamp, and the
  `Into<LinkCapacity>` composition so a refactor can't silently
  drift the defaults.

### RtspSrc: stash post-SETUP session, skip duplicate connect

- `intercept_caps`'s probe used to DESCRIBE + SETUP, extract caps, and
  *drop* the session — `run`'s first session attempt re-paid the same
  round-trips on the same server. Now the probe stashes the post-SETUP
  `Session<Described>` in `RtspSrc::stashed_session` alongside the
  discovered caps; `run_session` takes it on the first attempt and
  goes straight to PLAY. Reconnects after a network failure rebuild
  from scratch (by definition the stashed session is gone once the
  connection drops).
- Shared `connect_describe_setup` helper extracts the DESCRIBE +
  SETUP step from `run_session`; both the probe path (`probe_session`)
  and the reconnect path call it. `probe_caps_with_reconnect` renamed
  to `probe_session_with_reconnect`, returns a `StashedSession`
  carrying session + video_idx + caps.
- `run_rtsp` and `run_session` take `&mut RtspSrc` (was `&RtspSrc`) so
  the stash can be `take`n at the boundary between probe and run.

### M17: split `VideoFormat` into codec vs. raw pixel layout

- `Caps::Video { format: VideoFormat, .. }` (where `VideoFormat`
  conflated `H264`/`H265`/`Av1`/`Vp9` with `Nv12`/`I420`/`Rgba8`/`Bgra8`)
  is gone. Replaced by two distinct `Caps` variants:
  - `Caps::CompressedVideo { codec: VideoCodec, width, height, framerate }`
  - `Caps::RawVideo { format: RawVideoFormat, width, height, framerate }`
- New enums: `VideoCodec { H264, H265, Av1, Vp9 }` and
  `RawVideoFormat { Nv12, I420, Rgba8, Bgra8 }`. Old `VideoFormat`
  removed.
- A raw sink (`waylandsink`, `kmssink`) offered compressed input now
  fails negotiation structurally via `Caps::intersect` variant
  mismatch, not a runtime format compare. Mirrors GStreamer's
  `video/x-h264` vs `video/x-raw` distinction.
- New `Caps::dims(&self) -> Option<(&Dim, &Dim, &Rate)>` helper for
  element code that needs geometry without caring whether the link is
  pre- or post-decode. Several pattern-match sites collapsed to use it.
- `Caps::intersect` / `is_fixed` / `fixate` updated for the new
  variants; `is_fixed` now delegates to `dims()`.
- Both video variants keep `width/height/framerate` for now. Honest
  answer is GStreamer drops them on `video/x-h264` because they live
  in SPS, but our solver + RtspSrc placeholder Range + Range-as-
  placeholder convention all hang off geometry on compressed caps.
  Dropping it is a deeper rework overlapping workaround #1's
  redesign; out of scope here.
- No `codec_data` / `extras` field added. Latent need (SPS+PPS in
  negotiation, profile/level) but no current consumer; add when first
  driver lands.
- `AcceptsAny` still matches any `Caps` variant — no split into
  `AcceptsAnyRaw` / `AcceptsAnyCompressed`. Sinks that need to
  discriminate already pattern-match the format.
- Migration touched ~145 `Caps::Video` constructors and ~206
  `VideoFormat::` references across the workspace (sources, decoders,
  sinks, solver, fan-in/out, ~16 test files). Most was mechanical;
  helper functions in tests gained both `video()` (raw) and
  `compressed()` variants. `OutputFormat::video_format()` renamed to
  `raw_format()` in `ffmpegdec.rs`.
- Verified: `cargo test --workspace` (175 passed, 0 failed),
  `cargo test --features rtsp` (80 passed), `cargo test --features
  "rtsp ffmpeg"` (95 passed), `cargo check --features vaapi`,
  `cargo check --features wayland-sink`, no_std baseline.

### M18 item 5: async `SourceLoop::intercept_caps`

- Closes workaround #1 (RtspSrc placeholder Range). Sources can now
  perform I/O during negotiation and return real caps instead of a wide
  placeholder. `SourceLoop` gains `type CapsFuture<'a>` and
  `fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a>`; the
  default `caps_constraint()` is also async and awaits `intercept_caps`.
  `&self` → `&mut self` so a source can stash state (the discovered
  caps, an open session) for `run` to reuse.
- `DynSourceLoop` (the erased mirror used by fan-in / muxer) becomes
  `BoxFuture<'a, Result<Caps, G2gError>>`-returning. Same shape as
  `DynAsyncElement::process`.
- Runtime call sites — `runner.rs::run_simple_chain`, `run_source_fanout`,
  `coordinator::negotiate_source_transform_sink`, `fanin.rs::run_fanin_sink`
  / `run_muxer_sink` — `.await` the new future.
  `negotiate_source_transform_sink` is now `async fn`.
- `RtspSrc` refactor: `intercept_caps` performs DESCRIBE + SETUP, parses
  H.264 `VideoParameters`, drops the session, and caches the discovered
  caps for `caps_constraint` re-fixate retries. `with_expected_dims` is
  the offline fast path (returns caller-supplied geometry without I/O,
  framerate stays a fixable Range). Failures flow through the reconnect
  policy: a transient connect drop during negotiation retries with the
  same backoff `run` uses for mid-session drops; structural failures
  (bad URL, no H.264 stream) surface immediately.
- Test sources mechanically migrated (~16 files): each gains
  `type CapsFuture<'a> = core::future::Ready<...>` and a one-line
  `intercept_caps` returning `core::future::ready(...)`.
- New `m18_async_intercept_caps` test proves the runner genuinely awaits
  the caps future: a source `tokio::time::sleep`s in `intercept_caps`,
  and the pipeline's elapsed time must be at least the sleep interval.
  A sync stub would finish in microseconds.
- RtspSrc test surface updated:
  `rtspsrc_intercept_caps_with_expected_dims_skips_probe_and_fixates`
  replaces the old placeholder-Range assertion;
  `rtspsrc_intercept_caps_probes_and_fails_on_unreachable_url` covers
  the new probe-failure pathway. `rtspsrc_with_reconnect_retries_then_fails`
  now exercises probe-side reconnect (probe failure goes through the
  same retry+backoff loop `run` uses for session drops).
- Documents the structural cost: probe and `run`'s session-open are
  distinct connections, so the server pays for two DESCRIBEs at
  startup. A future optimization is to stash the post-SETUP session
  and consume it in `run`.

### caps-nego: drop per-frame `Frame.caps`

- Removed the `caps: Caps` field from `Frame` in `g2g-core/src/frame.rs`.
  Element-level caps state (set in `configure_pipeline`, updated on
  `PipelinePacket::CapsChanged`) is the single source of truth;
  `CapsChanged` events delimit caps epochs before the first affected
  `DataFrame`. The field was write-only on every hot path (grep
  confirmed zero readers in production code), so every produced frame
  was paying a `Caps::clone` for no consumer. Matches the lesson
  modern GStreamer applied when it moved caps off `GstBuffer` onto the
  pad.
- Updated all `Frame` constructors (sources, decoders, tests) to drop
  the field. `make_frame` test helpers that took a `caps: Caps` lost
  that parameter; call sites updated.
- `PickyByWidthSink` in `m8_negotiation` now tracks the current width
  via accepted `configure_pipeline` and `CapsChanged` events through
  `process()` instead of reading `frame.caps`. The
  `mid_stream_reconfigure_round_trip` data-widths assertion changed
  from `[640, 1280, 640]` to `[640, 640, 640]`: the rejected 1280
  `CapsChanged` never reaches the sink (the runner intercepts and
  converts to a `Reconfigure`), so the sink's tracked width never
  advances to 1280. The configure_widths assertion still proves the
  reconfigure round trip.
- `FakeReorderDecoder` in `m16_workaround3_phase_a` seeds `input_caps`
  from `configure_pipeline`'s `absolute_caps` so the first
  `DataFrame` (which arrives before any runtime `CapsChanged`) still
  has the correct input caps tagged for queue reorder accounting.
- Removed now-orphan `any_h264_caps()` in `rtspsrc.rs`.

### NVDEC backend low-latency defaults + `wayland_smoke` knobs

- `FfmpegH264Dec::with_backend(Backend::NvdecCuvid)` now enables
  low-latency tuning by default: `h264_cuvid surfaces=4` (down from
  cuvid's default 25, which adds ~25 frames of in-decoder buffering)
  and the `AV_CODEC_FLAG_LOW_DELAY` codec flag (release each picture as
  soon as it's decoded, no reorder hold). Switching back to
  `Backend::Software` clears the tuning so the sw path stays at
  libavcodec defaults. Override either knob via
  `with_cuvid_surfaces(Some(n))` / `with_low_delay(bool)` *after*
  `with_backend`. Wired in `configure_pipeline` via `open_as_with` +
  `Dictionary` + `Context::set_flags`. Closes the gap where the first
  `wayland_smoke` run with `NvdecCuvid` saw p50 = 163 ms (~80 ms over
  the link-cap floor) — almost entirely cuvid's `surfaces=25` pipeline
  depth, recoverable without a CUDA memory domain.
- Four new unit tests:
  `software_backend_does_not_set_cuvid_defaults`,
  `nvdec_backend_defaults_to_low_latency_tuning`,
  `switching_back_to_software_clears_nvdec_tuning`,
  `cuvid_surfaces_override_survives_after_with_backend`. Defaults are
  policy, not implementation detail; locking them so a refactor can't
  silently revert.
- `wayland_smoke` gains `G2G_TARGET_FRAMES` (default 60) so a NVDEC
  benchmark can amortize cuvid's startup tax (libnvcuvid load + CUDA
  context + surface pool can be 1-2 s). Test timeout now scales:
  `30 + max(30, target * 100 ms / 1000)` so a 600-frame steady-state
  run fits cleanly. Steady-state p50/p95 only become meaningful for
  `target >= ~300`.

### NVDEC backend for `FfmpegH264Dec` (NVIDIA hardware H.264 decode)

- New `Backend` enum and `FfmpegH264Dec::with_backend(Backend)`. Defaults
  to `Backend::Software` (existing libavcodec built-in H.264 decoder, no
  behavior change). `Backend::NvdecCuvid` opens the `h264_cuvid` standalone
  codec via `codec::decoder::find_by_name("h264_cuvid")`; if libavcodec
  wasn't built with cuvid or `libnvcuvid.so` isn't reachable at runtime,
  `configure_pipeline` fails loud with `Hardware(Other)` so the caller's
  backend choice is honoured rather than silently downgraded.
- No new Cargo feature: cuvid is a runtime libavcodec lookup, not a
  link-time dep. No raw `ffmpeg-sys` hwaccel plumbing
  (`AVHWDeviceContext`, `get_format`, `av_hwframe_transfer_data`) is
  needed because cuvid is a standalone codec that emits NV12 directly to
  system memory.
- Caps surface unchanged: the `DerivedOutput` constraint (H.264 in →
  NV12/I420 out, same geometry, framerate forwarded) is identical
  across backends, so existing solver wiring, mixed-chain tests, and the
  M16 workaround #3 Phase A consistency check apply verbatim.
- `copy_yuv420` extended to accept `Pixel::NV12` source frames (cuvid's
  native output) in addition to `YUV420P`/`YUVJ420P` (software path).
  Four (source-layout × output-layout) branches, each honouring source
  pitch. NV12-source → I420-output de-interleaves; NV12-source →
  NV12-output is a row copy.
- New unit tests `default_backend_is_software`,
  `with_backend_overrides_default`, and
  `caps_constraint_independent_of_backend` (NVDEC and software backends
  produce identical solver constraints — chains compose the same way).
- Visual e2e (`rtspsrc → h264parse → ffmpegdec[NvdecCuvid] →
  wayland/kms`) is user-side: CI has no GPU, and the existing rtsp
  sandbox block already constrains live tests to manual verification.

### M18: GStreamer parity push (item-by-item from DESIGN-M16-caps-nego.md §13.4)

- **Item 6: pad templates (declarative, pre-instantiation metadata).**
  New `pad_template` module (g2g-core, runtime feature): `PadDirection`,
  `PadCaps` (`Fixed(CapsSet)` / `Any`), `PadTemplate`, and a `PadTemplates`
  trait whose `pad_templates()` is an associated function (no `&self`), so
  a tool inspects an element *type* without constructing it, the analog of
  GStreamer's `gst_element_factory_get_static_pad_templates`. A template is
  the static superset of what the type can do; a constructed instance's
  `caps_constraint_as_*` is a subset. `pad_link(producer, consumer)` runs
  the negotiation `solve_linear` against two templates and returns the
  fixated caps or a structured `NegotiationFailure` (`EmptyLink` =
  incompatible, `Unfixable` = compatible but geometry/framerate still
  open); `types_can_link::<A, B>()` is the convenience boolean (treating
  `Unfixable` as compatible, since static templates routinely leave
  geometry open until instance time). `PadTemplates` implemented for
  `VideoTestSrc` (RGBA source), `FakeSink` (wildcard sink), and `H264Parse`
  (H.264 sink + source). Unit tests in the module plus integration test
  `m18_pad_templates.rs` (introspection without construction;
  `VideoTestSrc -> FakeSink` compatible, `VideoTestSrc -> H264Parse`
  `EmptyLink`, direction-awareness). no_std default core build, core
  runtime suite, full workspace tests, and workspace clippy all green.

- **Item 3 (Phase C): fan-out per-branch re-solve (FO-2) with FO-1
  strict default.** A mid-stream `CapsChanged` broadcast across the
  fan-out is now re-solved per branch against that branch's declared
  `caps_constraint_as_sink()` before `configure_pipeline` (Phase B applied
  per branch), closing the gap where the fan-out runner skipped the solver
  gate. Because each branch runs in its own arm, the re-solves are
  concurrent for free. FO-1 strict default: a branch whose constraint
  rejects the new caps fails the whole fan-out loud (`CapsMismatch`),
  matching GStreamer's `tee`-with-rejecting-downstream; the
  `AllowBranchDrop` graceful-degradation policy stays a future opt-in.
  `DynAsyncElement` gains a dyn-safe `caps_constraint_as_sink` (blanket
  impl forwards to `AsyncElement`) so `Box`-erased branch sinks can be
  re-solved; `re_solve_downstream_sink` is refactored to share a
  `re_solve_against_sink_constraint` core with the new
  `re_solve_downstream_dyn_sink`. New integration test
  `m18_fanout_phase_c.rs`: FO-2 accept (a geometry change every branch
  admits reaches each branch's `process`) and FO-1 strict reject (one
  RGBA-only branch fails the fan-out on an NV12 switch, and never sees the
  rejected caps). no_std core build, core suite, and the std plugins suite
  all green.

- **Item 1 (Session D follow-up): fan-out branch α.** Completes the α
  story for non-linear topologies. `DynAsyncElement` gains dyn-safe
  `propose_allocation` / `configure_allocation` (blanket impl forwards to
  `AsyncElement`); `coordinator::realloc_local_dyn` is the `Box`-erased
  counterpart of `realloc_local`. `run_source_fanout` now re-allocates
  each branch locally after the FO-2 re-solve applies the new caps. As in
  the linear case, the fan-out runner never configures branch allocation
  at startup, so the per-branch re-allocation is solely α. Covered by an
  added assertion in `m18_fanout_phase_c.rs` (each branch records exactly
  one re-allocation sized from the new caps).

- **Item 2 (Phase C): muxer per-input re-solve (MX-1) + input-derived
  output re-emit (MX-2).** A per-input mid-stream `CapsChanged` is now
  re-solved against the muxer's per-input constraint and applied to that
  pad (`MultiInputElement::configure_pipeline(i, ..)`), then consumed,
  instead of being forwarded raw to the output as the previous
  `process`-everything path did (which leaked input-side caps as the
  merged output). After the per-input reconfigure the runner re-derives
  the merged output and, only when it changed, eagerly emits one
  downstream `CapsChanged` (MX-2). Both run inside the existing single
  muxer task, which already owns `&mut mux` and serializes all inputs, so
  no β coordinator restructure is needed (workaround3 §10.4). Strict
  failure: a per-input caps the muxer can't accept fails loud
  (`CapsMismatch`); the structured reverse-`Renegotiate`-per-input variant
  is β-gated (the muxer task can't reach an input's reconfigure slot
  pre-β). The startup per-input and output negotiation are refactored into
  shared `solve_mux_input` / `solve_mux_output` helpers that the mid-stream
  paths reuse (no duplicated solver call). New integration test
  `m18_mux_phase_c.rs`: MX-1 (input pad re-solved, input `CapsChanged` not
  leaked, static `InterleaveMux` output unchanged) and MX-2 (a
  derived-output muxer emits exactly one downstream `CapsChanged` with the
  re-derived output). Closes the runtime half of item 2 (MX-3 trait
  surface already landed); no_std core build, core suite, and the std
  plugins suite all green.

- **Item 1 (Session D): α element-local re-allocation.** First
  observable M18 behavior change. New `coordinator::realloc_local`: when
  a mid-stream `CapsChanged` is applied to an element, the runner
  re-derives that element's own allocation params from the new caps
  (`propose_allocation`) and stores them (`configure_allocation`) before
  the element processes the notification. Wired at the three
  statically-typed mid-stream apply sites: `run_simple_pipeline` (sink),
  `run_source_transform_sink` (transform and sink). No cross-element
  cascade, that is β (Session E). Element-local only, so safe under the
  per-`Frame.caps` invariant: in-flight old-caps frames keep their
  old-pool buffers. Previously M12 allocation ran solely at startup, and
  in a 3-element chain the sink's `configure_allocation` was never called
  at all (its proposal feeds the transform); now a mid-stream geometry
  change re-allocates it. Fan-out branch sinks are excluded for now:
  `DynAsyncElement` does not expose the allocation hooks, so per-branch α
  lands with the FO-2 dyn-trait extension. New integration test
  `m18_alpha_realloc.rs` (one re-allocation sized from the new caps on a
  mid-stream change; none without). no_std core build, core suite, and
  the std plugins suite all green.

- **Item 1 (Session C): startup negotiation relocated to the
  coordinator.** Pure refactor, no behavior change. The
  `source → transform → sink` startup negotiation (the `solve_linear` +
  per-link `configure_pipeline` cascade with bounded `ReFixate` retry)
  moves verbatim out of `run_source_transform_sink` into
  `coordinator::negotiate_source_transform_sink`, returning a
  `LinearNegotiation { source_link, sink_link }` that names the per-link
  caps the β re-cascade will reconfigure. `MAX_FIXATION_ATTEMPTS` moves
  with it (still used by `run_simple_pipeline` via import). The runner now
  calls the routine and uses `sink_link` exactly where the loop produced
  `negotiated_caps`. Verified by the unchanged M8/M16/M18/pipeline_smoke
  integration suites (all green) plus the no_std core build; the next
  session (D) adds α element-local re-allocation hooks, the first
  observable M18 behavior change.

- **Item 1 (Session B): coordinator control-channel scaffolding.** First
  step of the allocation re-cascade β restructure
  (DESIGN-M16-workaround3-reconfigure.md §9.4 β; R2 single-task
  coordinator, R3 out-of-band channel). New `runtime::coordinator`
  module: `CoordinatorEvent`, `CoordinatorHandle` (clonable producer over
  the in-house mpsc), `Coordinator` task, and a `coordinator(capacity)`
  constructor. `run_source_transform_sink` now spawns the coordinator as
  a fourth join arm; the sink arm reports a `CoordinatorEvent::CapsChanged`
  for each applied mid-stream `CapsChanged` and the stub only counts them.
  The channel closes when the sink arm drops its handle, so the
  coordinator terminates with the pipeline. No data-plane behavior change:
  frames and EOS flow exactly as before; this only validates the channel
  topology before Session C moves startup negotiation in and Session E
  turns each event into a real `Recascade`. New
  `RunStats.coordinator_events` field (`0` for runners without a
  coordinator). New integration test `m18_coordinator.rs` (event observed
  on applied caps change; zero events and clean termination otherwise).
  no_std core build, core suite, and the std plugins suite all green.

- **Item 3 (partial): fan-out constraint migration.** Symmetric move
  to item 2 on the `MultiOutputElement` side. Adds
  `MultiOutputElement::caps_constraint_as_input(&self) ->
  CapsConstraint<'_>` default method to the trait (wraps
  `intercept_caps(...)` as a `LegacySink` per the existing legacy
  bridge pattern). `Router` (g2g-core) overrides to return
  `AcceptsAny` — it broadcasts upstream caps verbatim with no per-
  branch format restriction. `run_source_fanout` calls the new
  method instead of constructing an inline `LegacySink`. Closes the
  structural prerequisite for Phase C FO-2 (per-branch downstream
  re-solve once a mid-stream `CapsChanged` crosses the fan-out
  boundary); the runtime execution still needs the coordinator
  restructure (workaround3 §9 β). New unit test
  `router_input_constraint_is_wildcard` in `fanout.rs`. Existing
  M9 router/gate/merger integration tests unchanged; full workspace
  + Linux feature matrix + no_std build all green.

- **Item 2 (partial): muxer constraint migration.** Adds
  `MultiInputElement::caps_constraint_as_input(idx) -> CapsConstraint<'_>`
  and `MultiInputElement::caps_constraint_for_output() ->
  Result<CapsConstraint<'_>, G2gError>` default methods to the trait.
  Default `caps_constraint_as_input` wraps `intercept_caps(idx, ...)`
  as a `LegacySink` per-pad legacy bridge; default
  `caps_constraint_for_output` eagerly evaluates `output_caps()` and
  wraps as `LegacySource`. `InterleaveMux` (g2g-plugins) overrides
  both: per-input returns `AcceptsAny` (the muxer forwards
  per-frame-tagged caps straight through), output returns
  `Produces(CapsSet::one(self.output.clone()))` (static at
  construction). `run_muxer_sink` now calls these trait methods
  instead of constructing the inline `LegacySink` for each input pair
  and using a direct `output_caps().fixate()` for the downstream sink
  hop; the muxer→sink edge goes through `solve_linear` with an
  `AcceptsAny` sink-side constraint to preserve the contract that
  the sink's `intercept_caps` is not consulted for this hop. This
  closes the structural prerequisite for Phase C MX-1 (per-input
  mid-stream re-solve) and MX-2 (eager output `CapsChanged` on
  per-input change); the runtime execution of those still requires
  the coordinator restructure (workaround3 §9 β) and lands later.
  New unit tests `mux::tests::per_input_constraint_is_wildcard` and
  `mux::tests::output_constraint_is_produces_with_configured_output`.
  Existing `m10_muxer` integration tests and full workspace +
  Linux-feature matrix unchanged.

### M16: Caps negotiation redesign (CSP framing)
- Design doc `DESIGN-M16-caps-nego.md` (§§1-10) recasts negotiation as a
  constraint-satisfaction problem with a solver, and documents the
  6-step migration plan (§8).
- Step 1: `CapsSet` algebra in `g2g-core::caps`. Preference-ordered
  alternatives over `Caps`; `one`, `from_alternatives`, `intersect`
  (preserves self's outer order, dedupes equal results), `fixate`
  (picks the highest-preference fixable alternative). `Caps` remains
  the *fixed* runtime description; `CapsSet` is the negotiation-time
  vocabulary. Re-exported from the crate root. Unit-tested for empty
  intersection, preference preservation, dedup, and fixate fallback.
- Adjacent design debt acknowledged (no code change): `VideoFormat`
  conflates compressed codecs (H264, H265, Av1, Vp9) and raw pixel
  layouts (Nv12, I420, Rgba8, Bgra8) in a single enum, shoehorned
  into `Caps::Video { format, ... }`. GStreamer keeps these as
  separate media types (`video/x-h264` vs `video/x-raw, format=...`).
  Documented in `DESIGN-M16-caps-nego.md §11` and memory
  (`architecture_codec_vs_raw_format.md`). M17-sized refactor;
  M16 continues on the current shape.

- Latency observability: `VideoTestSrc` now stamps
  `FrameTiming::arrival_ns` at frame emission (std-gated, falls back
  to 0 in no_std), matching `RtspSrc`'s convention. `FakeSink` holds
  a `LatencyHistogram` and records `monotonic_ns() - arrival_ns` per
  received `DataFrame` whose `arrival_ns` is non-zero (std-gated),
  with a `latency_snapshot()` accessor returning a `LatencySnapshot`
  (count/mean/max/p50/p95/p99 nanoseconds, log2 buckets). `KmsSink`
  gets the same treatment: per-frame recording after page-flip
  submission, with a `latency_snapshot()` accessor; the timing point
  is page-flip submission rather than vblank completion, which
  under-reports true scanout latency by up to one refresh interval
  but is good enough as a regression guard. New regression test
  `videotestsrc_to_fakesink_latency_under_25ms` in
  `pipeline_smoke.rs` asserts max + p99 latency stay under 25ms for
  the all-in-memory `videotestsrc → fakesink` chain through the M16
  solver — catches order-of-magnitude regressions (lock contention,
  blocking I/O, runner serialization) while tolerating shared-CI
  variance. WaylandSink's existing histogram is unchanged. Tested
  across base, ffmpeg, rtsp, wayland-sink, kms-sink, std, and
  no-default-features builds.
- Workaround #3 Phase B (sink-side downstream subgraph re-solve):
  `run_simple_pipeline` and `run_source_transform_sink` now route every
  mid-stream forward `CapsChanged` arriving at the sink through a new
  `re_solve_downstream_sink()` helper before calling
  `configure_pipeline`. The helper runs `solve_linear` over
  `[LegacySource(new_caps), sink.caps_constraint_as_sink()]` and
  returns the assigned sink-input caps. A `NegotiationFailure::EmptyLink`
  here means the sink's declared constraint rejects the boundary's
  output — the runner drops the forward `CapsChanged` and signals a
  reverse `Reconfigure::Renegotiate` upstream (§7 forward × reverse:
  the request hits the transform's link, not the source, so the
  failure surfaces *at the boundary*). `NegotiationFailure::Unfixable`
  (e.g. a decoder leaving `Rate::Any` because framerate isn't
  pixel-level data) is treated as success and `new_caps` is passed
  through unchanged — fixation is a startup-negotiation concern, not
  a mid-stream re-solve one. The observable change in the current
  3-element runner: a sink with a restrictive
  `Accepts(set)` constraint can short-circuit a hostile mid-stream
  `CapsChanged` via the solver even if its legacy `configure_pipeline`
  would have silently accepted. New
  `g2g-plugins/tests/m16_workaround3_phase_b.rs`: a `HostileBoundary`
  fake transform injects a non-NV12 mid-stream `CapsChanged`; a
  `PickySink` declares `Accepts(NV12)` for the solver but accepts
  every shape in `configure_pipeline`. Asserts the hostile caps never
  reach `process` while a matching NV12 geometry-change still
  propagates. Full Linux feature matrix
  (`ffmpeg rtsp wayland-sink kms-sink`) green.
- Workaround #3 Phase A (retired for linear topology): decoders
  (`FfmpegH264Dec`, `MfDecode`, `VaapiH264Dec`) stop silently swallowing
  input `PipelinePacket::CapsChanged`. On arrival they validate the
  format (loud `CapsMismatch` on a hostile mid-stream H.264 → VP9
  switch; previously dropped silently) and record the caps into a new
  `input_caps` field. The output `CapsChanged` is still emitted at the
  decode boundary from decoded-frame geometry, preserving the §3
  ordering invariant. Each decoder's `DerivedOutput` closure now
  delegates to a free `derive_output_caps()` helper so a
  `debug_assert!` before each output `CapsChanged` push verifies
  decode-time geometry agrees with the closure applied to the recorded
  input (via `Caps::intersect` non-empty — handles the
  `Fixed`-vs-`Any` framerate asymmetry without false positives). New
  test `g2g-plugins/tests/m16_workaround3_phase_a.rs`: a
  `FakeReorderDecoder` simulates B-frame reorder; asserts the output
  `CapsChanged(B)` appears strictly between the last A-tagged frame
  and the first B-tagged frame (the regression guard against any
  future "eager forward" temptation), plus the loud-reject case. All
  44 g2g-plugins lib + integration suites green across
  `ffmpeg + rtsp + wayland-sink + kms-sink + vaapi` on Fedora.
  Decisions D2 (single source of truth via the closure) and D4
  (acknowledge, don't swallow) implemented. D1 (carrier) and D3
  (runner subgraph re-solve) not load-bearing for Phase A; they
  arrive with Phase B alongside the §7 forward × reverse
  `Reconfigure` race resolution.
- Workaround #3 design (no code): `DESIGN-M16-workaround3-reconfigure.md`
  turns the deferred forward in-band reconfigure into an implementable,
  phased spec. Key finding: decoders swallow their input `CapsChanged`
  and self-detect output geometry from decoded frames, which is
  correctly ordered; the naive "re-derive and forward eagerly" fix
  corrupts the stream across a B-frame reorder boundary (old-geometry
  frames still draining). The reconfigure must stay at the decode
  boundary. Phased plan: A) validate + record the input caps instead of
  silently dropping (linear, CI-testable), B) runner re-solves the
  downstream subgraph on a boundary `CapsChanged` (allocators /
  multi-element downstream), C) non-linear topologies. Decisions D1-D4
  (reuse `CapsChanged` vs new packet; derive via the `DerivedOutput`
  closure; runner subgraph re-solve; acknowledge-don't-swallow) flagged
  for sign-off.
- Cleanup (workaround #2 fully retired): removed the non-NV12 no-op
  accept branch from `WaylandSink` / `KmsSink` `configure_pipeline`.
  Reachability audit: every chain that reaches these sinks (incl. the
  `wayland_smoke` / `kms_smoke` `rtsp → ffmpegdec → sink` tests) now goes
  through a native `DerivedOutput` decoder, so the solver assigns the
  sink link NV12 (fixated to `Fixed`) at startup; the branch only fired
  on a decoder-less H.264→display chain, which exists in no test,
  example, or real pipeline. The sinks now reject non-NV12 input loud
  (`CapsMismatch`) instead of silently accepting undisplayable caps.
  `intercept_caps` stays pass-through (the solver hands it NV12).
  Updated tests: `configure_accepts_h264_as_deferred_noop` →
  `configure_rejects_non_nv12`; `intercept_passes_through_h264_for_deferred_configure`
  → `intercept_passes_through_any_format` (both sinks). Linux-gated
  (`wayland-sink` / `kms-sink`), so not compiled on the Windows host; a
  Linux + Wayland/DRM e2e is owed alongside the other deferred visual
  checks.
- Step 6 (DESIGN §8): `CapsFilter` element + ACCEPT_CAPS surface.
  `CapsConstraint::accepts(&Caps)` and `CapsSet::accepts(&Caps)` answer
  the ACCEPT_CAPS query (§7) as a pure check against the declared
  constraint, no negotiation: set-shapes test membership, `Mapping`
  tests its input sides, `DerivedOutput` tests input validity, wildcards
  accept anything, legacy bridges defer to their wrapped callbacks.
  `g2g-plugins::capsfilter::CapsFilter` is a pass-through transform whose
  native constraint is `Identity(set)`; inserted anywhere it pins the
  link to a concrete `CapsSet` (e.g. ahead of an `AcceptsAny` sink). It
  narrows on the legacy/mixed path via `intercept_caps`, validates its
  configured caps and any mid-stream `CapsChanged` against the filter
  (loud `CapsMismatch` on violation), and forwards data unchanged. New
  tests: `accept_caps_query_checks_constraint_set` (core) and three
  `capsfilter` lib tests. Completes the §8 migration plan.
- Workaround #1 (partial, opt-in): `RtspSrc::with_expected_dims(w, h)`.
  The RTSP handshake (and the real SDP geometry) only completes inside
  `run`, so `intercept_caps` defaults to a wide placeholder `Dim::Range`
  that survives fixation, with the real dims arriving later via
  `CapsChanged`. Callers who know the camera resolution can now declare
  it up front: `intercept_caps` then advertises fixed dims, so the
  chain negotiates the real geometry at startup and a downstream sink
  sizes its surface once instead of building at the placeholder min and
  rebuilding on the first `CapsChanged`. If the SDP disagrees, the
  mid-stream `CapsChanged` still corrects it. The placeholder remains
  the default; full auto-detection still needs an async
  `SourceLoop::intercept_caps` (the "big" redesign, deferred). New unit
  tests `intercept_caps_defaults_to_placeholder_range` and
  `with_expected_dims_advertises_fixed_geometry`; verified with
  `cargo test/clippy -p g2g-plugins --features rtsp`.
- Cleanup (post-5m): removed `FfmpegH264Dec`'s now-dead
  `is_format_boundary` and `propose_output_caps` overrides (and their
  two unit tests). The decoder is native (`DerivedOutput` since 5k), so
  the runner dispatches through `caps_constraint_as_transform` and never
  consults these forward-half hooks; `is_format_boundary` has no runtime
  consumer at all (it was a forward-declaration for a redesign that
  ended up keying on the constraint surface instead). The `AsyncElement`
  trait methods themselves stay: `propose_output_caps` is still live for
  *unmigrated* legacy transforms via the solver's mixed-cascade path
  (`solver.rs` `LegacyTransform { propose_output }`). Pure deletion; not
  compiled on the Windows dev host (`ffmpeg` is Linux-gated).
  `WaylandSink`/`KmsSink` non-NV12 branches were left in place: the
  in-code note from steps 5e/5j marks them still load-bearing for
  all-legacy chains, contradicting the "no longer load-bearing"
  expectation, so they need a separate decision before removal.
- Step 5m: `VaapiH264Dec` (Linux cros-codecs VAAPI) overrides
  `caps_constraint_as_transform` to return
  `CapsConstraint::DerivedOutput(closure)`, identical in shape to steps
  5k/5l. The closure validates H.264 input and emits NV12 at the same
  dims/framerate; non-H.264 input yields an empty `CapsSet` so the
  solver rejects at negotiation time. The VAAPI backend only emits
  NV12, so the closure has no output-format choice. New unit test
  `caps_constraint_is_derived_output_h264_to_nv12`. Not compiled on the
  Windows dev host (the `vaapi` feature is Linux-only); change is
  byte-identical in shape to the verified 5l/5k pattern.
- Step 5l: `MfDecode` (Windows Media Foundation) overrides
  `caps_constraint_as_transform` to return
  `CapsConstraint::DerivedOutput(closure)`, mirroring step 5k. The
  closure validates H.264 input and emits NV12 at the same
  dims/framerate; non-H.264 input yields an empty `CapsSet` so the
  solver rejects at negotiation time. The MFT only emits NV12, so the
  closure has no output-format choice. Mixed chains get real per-link
  caps from the solver: H.264 to the decoder, NV12 to the sink. New
  unit test `caps_constraint_is_derived_output_h264_to_nv12` covers
  the H.264→NV12 derivation and the non-H.264 rejection. Verified with
  `cargo clippy/test -p g2g-plugins --features mf-decode` (Windows).
- Step 5k: `FfmpegH264Dec` overrides `caps_constraint_as_transform`
  to return `CapsConstraint::DerivedOutput(closure)`. The closure
  validates H.264 input and emits the chosen output format
  (`Nv12`/`I420`) at the same dims/framerate; non-H.264 input
  yields an empty `CapsSet` so the solver rejects at negotiation
  time. With the decoder native, the production chain
  `rtsp → ffmpegdec → sink` becomes mixed: the solver returns
  `[H264, Nv12]` per-link, the runner feeds the decoder H.264
  (what its `configure_pipeline` requires) and the sink Nv12.
  Coupled with 5j (NV12 sinks tolerate mid-stream dim changes),
  the placeholder-dim NV12 at startup → real-dim mid-stream
  `CapsChanged` transition rebuilds the surface cleanly instead of
  refusing. New unit test
  `caps_constraint_is_derived_output_h264_to_chosen_format` covers
  the H.264→NV12 derivation and the non-H.264 rejection. All 99
  g2g-core + 10 ffmpegdec lib tests + every integration suite green
  across base / rtsp / ffmpeg feature sets. Visual verification of
  the manual `rtsp → ffmpegdec → wayland/kms` chain is on the user.

- Step 5j (reordered): NV12 display sinks (`WaylandSink`, `KmsSink`)
  tolerate mid-stream geometry changes. Previously `configure_pipeline`
  with new dims after the worker / framebuffer pool was up returned
  `CapsMismatch`; now it tears down the existing worker/slots
  (`self.shutdown()` / `self.teardown()`) and falls through to
  fresh setup at the new geometry. Same-dims is still a no-op.
  This unblocks the next step (5k: migrate `FfmpegH264Dec` to
  `DerivedOutput`): mixed chains will land NV12 at startup with
  placeholder dims (RtspSrc workaround #1, fixated to min), and the
  real geometry will arrive via mid-stream `CapsChanged` after SPS
  parse — both transitions now succeed instead of the second
  refusing. No new tests in this commit: the rebuild path runs
  through `worker_main` (Wayland session) / `allocate_slots` (real
  DRM card), neither testable in CI. Visual verification belongs to
  the user's manual e2e run.

- Step 5h: `CapsConstraint::IdentityAny` wildcard transform variant
  for pass-through transforms (probe / metering / tee) whose
  `intercept_caps` is `Ok(upstream.clone())`. Native solver couples
  input and output links to be equal without narrowing either by a
  set; surrounding endpoints determine the actual caps.
  `IdentityTransform` (g2g-plugins) overrides
  `caps_constraint_as_transform` to return `IdentityAny`.
  Existing 3-element pipeline_smoke test
  `source_identity_sink_3_element_pipeline` now exercises the
  all-native arc-consistency path
  (`VideoTestSrc Produces → IdentityTransform IdentityAny →
  FakeSink AcceptsAny`). 3 new solver tests cover the coupling, a
  mixed-chain pass-through, and rejection of `IdentityAny` in
  endpoint positions. 99 g2g-core tests + 11 plugin lib +
  every integration suite green.

- Step 5g: first native transform. `H264Parse` overrides
  `caps_constraint_as_transform` to return
  `Identity(CapsSet::one(Caps::Video { format: H264, dims: Any }))`.
  The native solver couples its input and output links and enforces
  the H.264 format requirement during arc consistency instead of via
  the dynamic `intercept_caps` callback. New tests:
  - `h264parse::caps_constraint_is_identity_h264_any` (unit).
  - `pipeline_smoke::h264parse_identity_negotiates_in_mixed_chain`
    drives a real `run_source_transform_sink` chain
    `LegacySource(H264) → H264Parse Identity → AcceptsAny FakeSink`
    and verifies negotiation + EOS propagation through the mixed
    cascade. The source is an inline EOS-only stub (no real H.264
    bytes needed — `H264Parse::process` for `Eos` is pass-through).
  96 g2g-core tests + 6 pipeline_smoke + every integration suite
  green.
- Step 5f (revised): first native source + `SourceLoop` trait
  integration. Original 5f scope (workaround #1 placeholder dims)
  bumped — properly fixing it needs async `intercept_caps` (SDP
  DESCRIBE), and it's symbiotic with #2 so fixing alone unblocks
  nothing visible.
  - `SourceLoop` gains `caps_constraint(&self) -> Result<CapsConstraint<'_>, G2gError>`
    default method returning `LegacySource(intercept_caps()?)`.
  - `run_simple_pipeline`, `run_source_transform_sink`, and
    `run_source_fanout` call `source.caps_constraint()` instead of
    constructing `LegacySource` inline. `ReFixate` retry uses
    `LegacySource(counter)` fallback (counter-proposals are a legacy
    concept). `run_muxer_sink` stays on `intercept_caps` because
    `DynSourceLoop` doesn't yet expose `caps_constraint` — no
    migrated muxer sources exist, so adding it is deferred until
    needed.
  - `VideoTestSrc` overrides `caps_constraint` to return
    `Produces(CapsSet::one(self.caps()))`. Production chain
    `videotestsrc → FakeSink` (both native) now exercises the
    all-native arc-consistency solver path with backward
    propagation, instead of the mixed cascade. Behavior unchanged.
  - 1 new solver test (`all_native_produces_to_accepts_any_passes_through`).
    96 g2g-core tests + every integration suite green.
- Step 5e: correctness fix — `solve_legacy_cascade` reverts to
  intercept-only (bit-compatible with the pre-M16 cascade). Step 4b
  had incorrectly called the format-boundary's `propose_output_caps`
  during the legacy cascade, producing per-link caps the legacy
  workaround-#2 sinks (`WaylandSink`, `KmsSink`) can't consume —
  e.g. NV12 at placeholder dims at startup, with the deferred-setup
  branch then refusing the mid-stream real-dim `CapsChanged`.
  Restored: every all-legacy chain element receives the single
  fixated `Caps` from the cascade's final intercept, matching
  pre-M16. The CI suite missed the regression because the e2e
  `rtsp → ffmpeg → waylandsink` test is `#[ignore]`d (sandbox
  blocks port 554).

  **Revised 5d claim**: per-link configure benefits mixed chains
  (one or both endpoints native). All-legacy chains keep the
  single-fixated-caps model and workaround #2 stays load-bearing
  for them — until those sinks migrate (which requires workaround
  #1, the placeholder-dims fix, to land first so per-link NV12
  carries real dims at startup).

  Updated `legacy_cascade_with_boundary_transform` test and
  `architecture_caps_nego_debt` memory.
- Step 5d: per-link configure (deletes caps-nego workaround #2).
  Two coupled changes:
  - `solve_legacy_cascade` no longer clobbers upstream link slots
    with the final fixated caps. Each link is fixated independently
    and carries its own intercept-narrowed value, so format-changing
    boundaries (decoder: H264 in / NV12 out) keep their per-link
    identity.
  - `run_source_transform_sink` extracts both `src_caps = links[0]`
    and `sink_caps = links[1]` from the solver and passes each
    element the side it expects: `source.configure_pipeline(src_caps)`,
    `transform.configure_pipeline(src_caps)` (input side — what
    decoders like `FfmpegH264Dec` validate against), and
    `sink.configure_pipeline(sink_caps)`. M12 allocation queries
    use the downstream-facing `sink_caps`.

  **Regression caught and fixed**: in step 5c, migrating `FakeSink`
  to `AcceptsAny` routed the rtsp e2e chain through the mixed
  cascade, which correctly returned `links=[H264, NV12]`. But the
  pre-5d runner fed `links.last()=NV12` to every element, including
  the decoder, which rejects NV12 with `CapsMismatch`. The e2e test
  is `#[ignore]`d so CI missed it. New regression test
  `format_changing_transform_receives_input_side_caps` in
  `pipeline_smoke.rs` covers this shape without needing ffmpeg.

  **Workaround #2 retired**: `WaylandSink` / `KmsSink` now receive
  NV12 directly at startup (when downstream of a decoder), so their
  "Caps::Video { .. } => no-op" deferred-setup branch is no longer
  load-bearing. The branch is left in place for safety; cleanup is
  a follow-up. Updated `legacy_cascade_with_boundary_transform`
  test to assert per-link semantics.
- Step 5c: `CapsConstraint::AcceptsAny` wildcard sink variant for
  debug / probe / passthrough sinks whose `intercept_caps` is
  `Ok(upstream.clone())`. Solver treats it as no-op narrowing on
  the link (upstream's produced caps flow through unchanged) and
  enforces it at the chain's tail; in a native chain a middle
  `AcceptsAny` is silently invisible, in mixed/legacy forward
  cascade an interior `AcceptsAny` returns `EndpointShapeMismatch`.
  `FakeSink` and `syncsink` (g2g-plugins) override
  `caps_constraint_as_sink` to return `AcceptsAny`; chains
  containing them now exercise the mixed-cascade path. `identity`
  (transform-shape pass-through) stays on the legacy bridge —
  needs an `Identity`-with-wildcard variant which is a separate
  gap. 3 new solver tests; all 95 g2g-core tests + integration
  suites green.
- Step 5b: trait integration so individual elements can migrate.
  `AsyncElement` gains two default methods —
  `caps_constraint_as_sink()` and `caps_constraint_as_transform()` —
  each returning the legacy bridge (`LegacySink` / `LegacyTransform`)
  for today's `intercept_caps` + `propose_output_caps`. The runner
  (`run_simple_pipeline` and `run_source_transform_sink`) now calls
  these methods instead of constructing the bridge inline; migrated
  elements override to return a native `CapsConstraint` and chains
  containing them hit the mixed-cascade solver path. Behavior is
  identical for every existing element (all defaults). Bridge helpers
  `legacy_sink_constraint` / `legacy_transform_constraint` relaxed
  to `?Sized` so they can be called from the trait's default methods
  on `&Self`.
- Step 5a: mixed-chain support in the solver. Chains that mix
  legacy and native `CapsConstraint` variants now route to a new
  `solve_mixed_cascade` (single forward pass that handles every
  variant). Legacy variants and `DerivedOutput` require upstream to
  fixate to one concrete `Caps`, which migration chains satisfy
  because the source is typically single-alternative. Backward
  arc-consistency (Identity / Mapping filtering against downstream
  sinks) is not applied in the mixed path; once a chain becomes
  fully native, dispatch routes it back to the arc-consistency
  solver and backward propagation is restored.
  `NegotiationFailure::MixedLegacyAndNative` is no longer returned
  (kept as a variant in case future mixed shapes need it).
  4 new tests cover legacy-source/native-sink, native-source/legacy-sink,
  native-source/legacy-decoder/native-sink, and sink rejection in a
  mixed chain.
- Step 4c: roll the solver-via-legacy-bridge pattern out to the
  remaining linear runner entry points. `run_simple_pipeline`
  (source → sink) and `run_source_fanout` (source → fanout, with the
  fanout as the linear "sink" of the chain — downstream sinks
  broadcast-receive the fixated caps and don't participate in
  narrowing). `run_muxer_sink` solves each source ↔ muxer-input pair
  via the solver, wrapping the muxer's per-input `intercept_caps`
  as a `LegacySink`. `run_fanin_sink` stays direct: each source
  self-fixates with no peer narrowing, so there's no chain to solve.
  The muxer's aggregated-output → sink half stays as direct
  `fixate()` because today's runner intentionally does *not* call
  `sink.intercept_caps` for that hop (the muxer output is the
  canonical merged caps).
- Step 4b: `run_source_transform_sink` startup negotiation routes
  through `solve_linear` via the legacy bridge. The pre-M16 inline
  cascade (`source.intercept_caps` → `transform.intercept_caps` →
  `sink.intercept_caps` → `fixate`) is replaced by building
  `LegacySource` / `LegacyTransform` / `LegacySink` constraints and
  calling the solver; the cascade output is bit-compatible with what
  the inline path produced. `ReFixate` retry stays in the runner (the
  solver doesn't model counter-proposals); on each retry the
  `LegacySource` seed becomes the counter and the solver re-runs.
  Mid-stream `CapsChanged` paths are untouched. All existing tests
  pass (89 g2g-core, 14 g2g-plugins rtsp lib, all integration
  suites).
- Step 4a: legacy bridge into the solver. `CapsConstraint` gains a
  `'a` lifetime parameter and three transitional variants —
  `LegacySource(Caps)`, `LegacyTransform { intercept, propose_output }`,
  `LegacySink(intercept)` — that capture today's `AsyncElement`
  callbacks. `legacy_transform_constraint(&T)` / `legacy_sink_constraint(&T)`
  helpers wrap a borrowed element. The solver dispatches: all-native
  chains take arc consistency, all-legacy chains take
  `solve_legacy_cascade` (forward cascade that mirrors today's runner,
  then fixates the final caps and propagates to upstream link slots
  the same way `configure_pipeline` is called today). Mixed chains
  return `NegotiationFailure::MixedLegacyAndNative` until step 5
  migrates individual elements. 6 new tests cover the cascade,
  pass-through and boundary transforms, intercept failure, mixed
  chain rejection, and the AsyncElement → LegacyTransform bridge.
- Step 3: linear-pipeline caps solver in
  `g2g-core::runtime::solver` (feature `runtime`).
  `solve_linear(&[&CapsConstraint]) -> Result<Vec<Caps>, NegotiationFailure>`
  walks a source → transform* → sink chain with arc consistency:
  seed endpoint links from `Produces` / `Accepts`, forward+backward
  sweep until fixed point, fixate each link to one concrete `Caps`.
  Handles all four interior constraint shapes — `Identity` couples
  input and output, `Mapping` filters pre-enumerated (in, out) pairs
  to the surviving set, `DerivedOutput` fires once its input link
  fixates. `NegotiationFailure` reports `EmptyLink` /
  `EndpointShapeMismatch` / `Unfixable` / `Degenerate`; `Cyclic` is
  reserved for the non-linear solver. `CapsSet::union` added to
  support the `Mapping` path. 10 unit tests cover the minimal chain,
  empty/disjoint links, degenerate input, endpoint-shape errors,
  preference tie-break, identity coupling and mismatch, derived
  output, and mapping pair selection.
- Step 2: `FormatElement` trait + `CapsConstraint` enum in new
  `g2g-core::format_element` module. `CapsConstraint` variants:
  `Accepts` (sinks), `Produces` (sources), `Identity` (pass-through
  transforms), `Mapping(Vec<(in, out)>)` (pre-enumerated codecs),
  `DerivedOutput(Fn(&Caps) -> CapsSet)` (decoders reading SPS).
  `configure_link(input, output)` replaces `configure_pipeline`;
  boundary elements see distinct sides, sources/sinks see `None` on
  the unused side. `CapsPreferences` is a placeholder for the
  tie-break algebra (DESIGN-M16 §10). Coexists with `AsyncElement`;
  the legacy-bridge blanket impl lands with the solver in step 3,
  because its shape is dictated by what the solver consumes.

### M12: Live-source surface (latency + allocation + clock election)
- Latency query: `LatencyReport { live, min_ns, max_ns }` + `query` module,
  GStreamer-style latency triple with `combine`/`aggregate` (min and max sum
  along the path, unbounded `max_ns = None` is infectious, liveness sticky) and
  `is_unsatisfiable`. `ZERO` contributes `max_ns = Some(0)` (non-buffering),
  distinct from `None` (unbounded buffering). `AsyncElement::latency()` /
  `SourceLoop::latency()` default methods (return `ZERO`); live sources and
  buffering transforms override.
- Allocation query: `AllocationParams { size_bytes, min_buffers, align, domain }`
  + `MemoryDomainKind` (and `MemoryDomain::kind()`). `AsyncElement::propose_allocation`
  / `configure_allocation` and `SourceLoop::configure_allocation` default
  methods let a consumer propose a buffer pool that its producer allocates into
  (zero-copy handoff). `AllocationParams::merge` folds an upstream element's own
  requirement into a downstream proposal (most-demanding size/count/alignment
  wins; consumer dictates domain).
- Clock distribution: `ClockPriority` (`SystemFallback` < `Provider` <
  `LiveSource`), `ClockCandidate` (priority + shared `Arc<dyn PipelineClock>`),
  and `elect_clock` (highest priority wins, ties resolve to the most upstream).
  `AsyncElement::provide_clock` / `SourceLoop::provide_clock` default methods
  let a live source or sink offer its clock; the runner adopts the elected clock
  over the supplied fallback.
- Linear runners (`run_simple_pipeline`, `run_source_transform_sink`) fold the
  configured chain into `RunStats::latency`, resolve the allocation query into
  `RunStats::allocation`, and elect the pipeline clock into
  `RunStats::{clock_priority, base_time_ns}` after negotiation. Fan-in /
  fan-out runners report neutral values (topology aggregation deferred).
- M12 complete: with M8–M12 done, `g2g` reaches dynamic-pipeline feature parity
  with GStreamer (per DESIGN.md §4.10) while keeping the static typed layer.

### M15: RTSP reconnect + long-running stability soak
- `RtspSrc` now supports reconnect-with-backoff. Off by default for
  backwards compatibility; opt in with `.with_reconnect(max_attempts)`
  and optionally `.with_reconnect_backoff(initial_ms, max_ms)`.
  Exponential backoff caps at `max_ms`. Network/protocol errors trigger
  retry; a graceful server-side end-of-stream (retina's demuxer returning
  `None`, typical for VOD finishing) terminates without retry.
- `run_rtsp` refactored into an outer reconnect orchestrator and inner
  `run_session`. State threaded across sessions:
  - cumulative `sequence` counter so downstream sees monotonic IDs;
  - `pts_base_ns` offset so per-session PTS continues monotonically
    across reconnects (with a deliberate 1 s gap marking the boundary).
- A `PipelinePacket::Flush` is emitted before each reconnect so the
  decoder flushes its codec state and sinks reset `last_sequence`.
- `tests/rtsp_soak.rs` (#[ignore]) is a long-running stability soak
  configurable via `G2G_SOAK_SECONDS` (default 30) and `G2G_SOAK_MIN_FRAMES`
  (default `seconds * 20`). Asserts monotonic `sequence` and `pts_ns`,
  and that the pipeline reaches the frame floor — catches PTS regressions
  and stalls that the 2-second smokes miss. Module docs cover the manual
  reconnect-exercise workflow (Ctrl-C the publisher, watch it resume).
- `tests/rtsp_smoke.rs` gains `rtspsrc_with_reconnect_retries_then_fails`
  which exercises the reconnect orchestrator against an unreachable URL,
  asserting the source retries `max_attempts` times before surfacing an
  error.

### M14: Wayland display sink (Linux, NV12, desktop-dev convenience)
- `WaylandSink` element (`wayland-sink` feature, Linux-only): opens an
  `xdg_toplevel` window on the running compositor and presents NV12 frames
  via `wl_shm` after software conversion to XRGB8888 (BT.601 limited range).
  Designed as the desktop-dev companion to `KmsSink` — same NV12 input
  contract so the upstream pipeline stays identical.
- Threading: a dedicated worker thread owns all Wayland state (Connection,
  EventQueue, SlotPool); the sink struct holds only a calloop channel and
  an `Arc<AtomicU64>` counter, both `Send + Sync`. SCTK's
  `calloop_wayland_source` multiplexes Wayland events and frame arrivals
  in a single event loop. A one-shot Mutex/Condvar handshake gates the
  sink-side `configure_pipeline` on the first compositor `configure`.
- Pulls `smithay-client-toolkit` 0.20 (transitively bringing the
  `wayland-*` family) under `[target.'cfg(target_os = "linux")'.dependencies]`.
- Constraints (v1): NV12 only, fixed input dims, no scaling (compositor
  letterboxes/clips if its configure suggests a different size), no PTS
  pacing, software conversion only (zero-copy via `zwp_linux_dmabuf_v1` is
  deferred).
- `KmsSink` is the production low-latency sink; `WaylandSink` is for
  iterating on the pipeline inside your desktop session without dropping
  to a tty.

### M14: KMS/DRM display sink (Linux, NV12)
- `KmsSink` element (`kms-sink` feature, Linux-only): primary-plane scanout
  of NV12 `DataFrame`s on the first connected connector + CRTC of the
  configured DRM device (defaults to `/dev/dri/card0`). Two-buffer dumb-
  buffer pool; first frame goes through `set_crtc`, subsequent frames page-
  flip and the next submission blocks on the prior flip's `PageFlip` event
  so the buffer being overwritten is off scanout (tearing-free).
- `FfmpegH264Dec::with_output_format(OutputFormat::Nv12)` (M14 prerequisite,
  separate commit) interleaves the U/V planes after decode without swscale;
  same total byte length as I420. I420 remains the default.
- New optional, target-gated deps: `drm` 0.15 + `drm-fourcc` 2 under
  `[target.'cfg(target_os = "linux")'.dependencies]`; `kms-sink` implies
  `std`. No `unsafe` and no GBM dependency — pure dumb-buffer path.
- Constraints (v1): NV12 only, fixed input dims (mid-stream geometry change
  not supported), no letterboxing/scaling (buffer scans out at native dims;
  smaller-than-mode video shows at origin with stale framebuffer around it),
  requires DRM master (tty or DRM lease; a running compositor will block).
- Deferred (v2): overlay-plane path with src/dst rectangles for proper
  letterboxing; async page flips for lower latency; Wayland sink as a
  desktop-dev convenience using the same NV12 input contract.

### M13: End-to-end RTSP → ffmpeg decode (Linux software path)
- `RtspSrc::intercept_caps` now advertises fixate-friendly `Dim::Range` /
  `Rate::Range` instead of `Any`. `Caps::fixate()` rejects `Any` and aborted
  Phase 2 negotiation before any network handshake; the placeholder is
  overwritten by the SDP-derived `CapsChanged` emitted from `run`.
- `RtspSrc` drops every `VideoFrame` until the first `is_random_access_point`
  IDR. retina's `FrameFormat::SIMPLE` only prepends SPS/PPS on keyframes, so
  mid-GOP tune-in (typical for live RTSP servers like MediaMTX) would feed
  parameterless slices to the decoder and stall with "non-existing PPS 0".
- New `rtsp_ffmpeg_e2e` integration test (`rtsp` + `ffmpeg` features, Linux):
  `RtspSrc → FfmpegH264Dec → FakeSink` over `run_source_transform_sink`,
  asserting decoded I420 frames reach the sink with a fixed-dim `CapsChanged`
  preceding the first `DataFrame`. Module docs include a MediaMTX + ffmpeg
  recipe for a deterministic local fixture.

### M13: Windows hardware decode (Media Foundation)
- `MfDecode` element (`mf-decode` feature, Windows-only): wraps the Media
  Foundation H.264 Decoder MFT (`CLSID_MSH264DecoderMFT`, an `IMFTransform`).
  Consumes Annex-B H.264 `DataFrame`s and emits decoded NV12 frames as
  `MemoryDomain::System`, with a `CapsChanged(Nv12)` before the first frame and
  on each decoder stream change. Implements the canonical feed/drain MFT loop
  (`ProcessInput`/`ProcessOutput`, `NEED_MORE_INPUT`/`STREAM_CHANGE`,
  `COMMAND_DRAIN` on EOS, `COMMAND_FLUSH` on seek).
- New optional, target-gated dependency: `windows` 0.62 under
  `[target.'cfg(windows)'.dependencies]`; the `mf-decode` feature implies `std`.
- `HardwareError::MediaFoundation(i32)` carries the failing `HRESULT`.
- Constraint: COM is initialised MTA; the element is thread-affine and intended
  for a single-thread executor (asserted `Send` under a documented contract).
- Deferred: D3D11 zero-copy output (needs a new `MemoryDomain` variant), DXVA
  hardware acceleration, and strided (`MF_MT_DEFAULT_STRIDE`) NV12 copy.

### M11: Application control surface
- `Bus` + cloneable `BusHandle` + `BusMessage` (Eos/Error/Warning/Custom): mp-sc message channel so elements notify the app asynchronously without back-references. Non-blocking `try_post`; `try_recv`/`recv` on the app side.
- `LinkInterceptor` probes: a `Pass`/`Drop` interceptor installed on a link's `SenderSink` via a runtime `ProbeSlot` (GStreamer pad-probe equivalent). Empty by default, so existing links are unaffected.
- `PipelinePacket::Flush`: non-terminal seek-flush packet; elements reset position (sinks drop `last_sequence`) and forward/broadcast it, the stream resumes afterwards.
- Deferred: blocking probe action.

### M10: True muxer fan-in
- `MultiInputElement` trait + `run_muxer_sink`: combine all N inputs into one output (vs the M9 Merger selector), with per-input caps negotiation and EOS aggregation (one `Eos` after every input ends).
- `InterleaveMux` element; tagged-merge-channel runner so one `&mut` muxer task processes all inputs without a select primitive.

### M9: Dynamic fan-out / fan-in
- `Router` (1->N): routes each frame to an atomic-selected output port, broadcasts `CapsChanged` to all ports.
- `Gate` (1->1): forwards or drops data by an atomic flag; a plain `AsyncElement`, so it needs no new runner.
- `Merger` (N->1): selects one active input, drains the rest, emits one merged `Eos` after every input ends.
- Multi-output / fan-in plumbing: `MultiOutputSink`, `MultiOutputElement`, `DynSourceLoop`, `join_all`, and the `run_source_fanout` / `run_fanin_sink` runners. Each primitive has a cloneable control handle (`RouterHandle` / `GateHandle` / `MergerHandle`).
- Deferred: `BranchSlot` + `SwapPolicy` variants (need a task spawner).

### M8: Caps negotiation + dynamic element swap
- Caps algebra: `Dim`/`Rate`/`Caps` `intersect`, `is_fixed`, `fixate` (Phase 1 narrowing, Phase 2 fixation per §4.2).
- Runners fixate negotiated caps before configuring elements; bounded `ReFixate` retry; mid-stream `CapsChanged` cascade and upstream `Reconfigure` sideband.
- `ElementSlot` + `SwapHandle`: lock-free atomic hot-swap of one element mid-stream; `DynAsyncElement` blanket impl so any element boxes into a slot.

## M0-M5 (prior)
- M0/M1: Cargo workspace scaffold and minimal `no_std + alloc` pipeline runtime (bounded channel, `Join2`, linear runner).
- M2: async `OutputSink`, identity transform, 3-element runner.
- M3: `AsyncClock` + `WallClock` + `SyncSink` for PTS-paced presentation.
- M4: Arc-recycled `BufferPool` with async `acquire`; DMABUF close-on-drop fix.
- M5: `RtspSrc` element wrapping `retina`.
