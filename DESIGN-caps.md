# glass2glass: CSP Caps Negotiation

The capability-negotiation subsystem of `glass2glass`, extracted from
[DESIGN.md](DESIGN.md) for length (it is the largest single part of the
design). Section numbers (§4.13.x) are preserved verbatim, so the many
`§4.13.x` cross-references elsewhere in `DESIGN.md` resolve here.

---

### 4.13 CSP Caps Negotiation

The handshake sketched in §4.2 is the *interface* contract. The underlying
mechanism is a **distributed constraint-satisfaction problem (CSP)**: each
element declares a constraint over `(input, output)` caps; a solver finds an
assignment over every link in the graph that satisfies all constraints,
ranked by preferences; the assignment becomes the per-link `Caps` the runner
hands each element via `configure_pipeline`.

This subsumes GStreamer's pad-by-pad negotiation: the solver runs once over
the whole graph (or over an affected subgraph on a mid-stream change),
returns structured failure when no assignment exists, and trades pad-query
round-trips for direct function calls.

#### 4.13.1 CapsSet and the constraint enum

```rust
/// A set of acceptable caps descriptions, ordered by preference.
pub struct CapsSet { alternatives: Vec<Caps> }
impl CapsSet {
    pub fn one(caps: Caps) -> Self;
    pub fn intersect(&self, other: &Self) -> Self;
    pub fn fixate(&self) -> Option<Caps>;
}

pub enum CapsConstraint {
    Accepts(CapsSet),                             // sink-shape
    AcceptsAny,                                   // wildcard sink (probes, fakes)
    Produces(CapsSet),                            // source-shape
    Identity(CapsSet),                            // pass-through transform
    IdentityAny,                                  // wildcard pass-through
    Mapping(Vec<(CapsSet, CapsSet)>),             // explicit (in, out) pairs
    DerivedOutput(Box<dyn Fn(&Caps) -> CapsSet>), // output as function of input
    DerivedCoupled {                              // like DerivedOutput, plus a
        derive: Box<dyn Fn(&Caps) -> CapsSet>,    //   declared passthrough-field
        passthrough: PassthroughFields,           //   mask for bidirectional
    },                                            //   field-level coupling
}
```

`DerivedOutput` is opaque: the solver can only invert it by *dropping whole
input alternatives* whose forward image can't reach the constrained output, so
a downstream pin on a field a transform passes through (e.g. a `160x120`
geometry pin behind a format-only `videoconvert`) can't narrow a ranged input
field. `DerivedCoupled` fixes that for the caps-driven transforms (videoscale /
videoconvert / audioresample): the `passthrough` mask names the fields where
output == input, and the backward sweep (`backward_field_narrow`) intersects a
downstream pin *into* those input fields (`Range ∩ Fixed = Fixed`). The closure
stays the source of truth for the retargeted fields.

The mask and the closure are two sources of truth for one fact (which fields
couple backward), so they can drift: a mask claiming a field the closure actually
retargets is unsound (the solver would narrow the input on a field the transform
rewrites). A full *closure-free* forward-derivation descriptor would remove the
duplication, but it is a deliberate non-goal: forward derivation is genuinely
imperative (a scaler branches on format membership and enforces 4:2:0 even-dims,
the cross-field validity §4.13.10 keeps out of the declarative constraint), so it
cannot be a `Copy` descriptor without re-importing exactly what was excluded.
Instead the drift is caught directly: the solver's forward step runs a
`debug_assert!` (`verify_passthrough_sound`) that every field the mask declares
passthrough is in fact repeated unchanged across *all* of the closure's output
alternatives for the concrete input. Unlike `discover_passthrough` it stays valid
for the multi-valued closures `DerivedCoupled` exists for (it checks the declared
fields, not a single output), and it flags only the unsound direction
(declared-but-not-honoured); a field the closure passes through but the mask omits
is merely a missed coupling, which is sound.

A plain `DerivedOutput` (a decoder that declares no mask) recovers the same
backward coupling automatically: `discover_passthrough` probes the closure
with two distinct concrete inputs per field and marks a field passthrough when
the single output tracks it in both, so the solver narrows those input fields via
the same `backward_field_narrow` path. A `couple_passthrough_derived` extends the
coupling across the variant change a decoder/encoder makes (`CompressedVideo <->
RawVideo`), coupling the geometry / framerate both carry (`format` is retargeted
across a codec boundary, so probing never marks it passthrough). Discovery is
conservative, a field that the closure fixes or that fails either probe stays
non-passthrough, so a genuinely non-invertible closure falls back to the
alternative-drop walk unchanged. (Discovery is gated on the closure being
single-valued on the sample: a multi-valued converter, e.g. one offering
`{passthrough, retargeted}`, has no well-defined per-field passthrough, so probing
it is unsound and yields `NONE`.) The mid-stream `backward_feasible` snapshot now
recovers the same coupling: the per-edge sweep is threaded the
element's startup-fixated input caps (from the solved edge set), which supplies the
concrete probe a `DerivedOutput` needs and the input variant / scalar identity. The
passthrough fields take the downstream pin's value, but every *non-passthrough*
(re-derived) field widens to `Any` (`project_passthrough_derived`): the
transform re-derives that field from whatever input it gets mid-stream, so the
input edge stays unconstrained on it. Freezing it to the startup value
made the snapshot reject a legitimately re-derived mid-stream geometry, the Caps-β
forward gap; with widening, a `DerivedOutput` stacked below another
format-changing transform re-derives its output on a mid-stream input change and
the runner cascades it downstream. A decoder below a geometry pin still exposes a
constrained input edge; an empty discovered mask or a missing sample imposes none.

`Caps` is the *fixed* description used at runtime (carried by
`PipelinePacket::CapsChanged`, handed to `configure_pipeline`); `CapsSet`
is the negotiation-time vocabulary.
`Caps` is split into compressed and raw at the type level:

```rust
pub enum Caps {
    CompressedVideo { codec: VideoCodec, extras: CodecExtras },
    RawVideo { format: RawVideoFormat, width: Dim, height: Dim, framerate: Rate },
    Audio { .. },
    Tensor { .. },
}
```

so a raw-only sink simply cannot match compressed caps, and the impossibility
becomes a type-level error rather than a runtime `not-negotiated`.

#### 4.13.2 The solver

`solver::solve_linear` runs arc consistency on a chain: forward sweep
(`Produces ∩ Accepts ∩ Identity ∩ Mapping ∩ DerivedOutput`), backward sweep
to propagate narrowing, fixate each link to its highest-preference concrete
`Caps`, then call `configure_pipeline` per element with its side of the link.

```rust
pub enum NegotiationFailure {
    EmptyLink { upstream: ElementId, downstream: ElementId, missed: CapsSet },
    EndpointShapeMismatch { .. },
    Unfixable { .. },
    Cyclic { .. },
}
```

Failures name the responsible pair and what they couldn't agree on, and are
posted to the pipeline `Bus` via `BusMessage::NegotiationFailed`.

`solver::downstream_feasibility(constraints) -> Vec<Option<CapsSet>>` is a
backward fold from the sink that computes, per link, the set the downstream
tail can still fixate **ignoring the upstream**. It's source-independent and
serves as a snapshot for the mid-stream re-solve (§4.13.4).

#### 4.13.3 The DAG runner

`run_graph(Graph<GraphNodeRef>, clock, link_capacity)` is the single runner.
A `Graph` is built from `GraphNode { Source | Element | Muxer }` payloads and
edges (each carrying a `LinkPolicy`); `finish()` validates topology (topo
sort, cycle / orphan / pad-count checks) before the run. `run_graph` owns
whole-graph `solve_graph` negotiation, per-node configure, the latency /
clock / allocation folds, one data arm per node over the edge channels, the
β allocation re-cascade and the Caps-α mid-stream re-solve. It covers the
full topology space: linear, fan-out (tee), fan-in (muxer), and diamonds.

`run_linear_chain`, `run_source_transform_sink`, `run_simple_pipeline`,
`run_source_fanout`, and `run_muxer_sink` are **thin builders**: each
constructs the corresponding borrowing `Graph` and delegates to `run_graph`,
so the four historical runner shapes share one negotiation + data plane. A
node's mid-stream rejection policy is topology-derived: a node on a
single-producer chain forwards a feasible re-solve and keeps flowing, but a
genuinely infeasible one (the refined caps have no solution against the
downstream chain) fails the run loud (posting the structured failure to the
bus), since no runtime producer renegotiates its output caps and there is
nothing to reverse-reconfigure into. A node behind a tee likewise cannot
reverse-reconfigure (a shared upstream can't honour a per-branch reconfigure).
What a behind-a-tee rejection
then does is the tee's [`FanOutPolicy`](crate::graph::FanOutPolicy): `FailLoud`
(the `add_tee` default) fails the whole run, and `AllowBranchDrop`
(`add_tee_with_policy`) drops just that branch (its arm ends, the tee removes its
now-closed sender via `broadcast_drop_closed` and keeps broadcasting to the rest)
so an optional branch (a preview that can't follow a format switch) does not kill
the essential ones. A genuine downstream error still surfaces through that branch
arm's own result, so swallowing the closed channel at the tee is safe.

`run_graph` consumes the elements it runs (it `take()`s the boxed payloads), so a
graph runs only once. Re-running (seek-and-replay after a flushing seek, retry,
A/B benchmarking) needs *fresh* elements, because real ones carry state a rewind
cannot undo (a decoder's reference frames, a source's file offset). A
[`GraphTemplate`](crate::runtime::GraphTemplate) wraps a builder closure and hands
back a fresh `Graph<GraphNode>` per `instantiate()`, which is cleaner than making
`Graph` itself reusable: that would force every element to be `Clone` or
re-initialisable in place, a contract the element traits deliberately avoid.

**Opt-in multicore (thread-per-arm).** `run_graph` drives every arm cooperatively
on the caller's one executor thread, so a CPU-bound stage (software decode/encode)
blocks the whole graph while it runs. `run_graph_threaded(graph, clock, cap,
spawner)` is the opt-in multicore sibling: it negotiates identically (the
`prepare_graph` / `build_channels` / `fold_run_stats` helpers are shared verbatim,
so the two drivers cannot diverge in how they solve or aggregate), then hands each
arm to a [`GraphSpawner`](crate::runtime::GraphSpawner) to run on its **own OS
thread** instead of `JoinAll`-multiplexing them. This is the GStreamer
streaming-thread model, one thread per element, so CPU-bound stages overlap across
cores. Crucially it needs no `Send` *futures*: the spawner receives a `Send`
*builder closure* and constructs + drives the arm's (`!Send`) future entirely on
the worker thread, so only the element and its channels (both `Send` under the
`multi-thread` feature) cross the thread boundary. An element whose future holds a
raw hardware context (a decoder) therefore runs unchanged, exactly as under the
cooperative runner. Two spawners ship: core's dependency-free
[`ThreadSpawner`](crate::runtime::ThreadSpawner) (std threads + the park-based
`block_on`, for pure-core graphs) and `g2g-plugins`' `TokioThreadSpawner` (a
per-arm current-thread tokio runtime with `enable_all`, so network / timer
elements get a reactor). It stays opt-in (`g2g-launch --threads`) because a
per-stage thread handoff adds wakeup latency: cooperative single-thread is the
lower-latency default, and it is the only path for the `no_std` / wasm / embassy
executors, which the `run_graph_threaded` gate (`std + multi-thread`) excludes.
`run_graph_threaded` requires an owning `Graph<GraphNode>` (`'static`) so each arm
can move its element onto a worker thread.

#### 4.13.4 Mid-stream re-solve

A mid-stream `PipelinePacket::CapsChanged` triggers a re-fixation that stays
correctly downstream-aware:

1. At startup, each interior arm receives its `downstream_feasible:
   Option<CapsSet>` from the backward sweep.
2. Mid-stream, arm *i* on `CapsChanged(in)`:
   - intersect `in` with the element's input constraint; empty → fail the run
     loud (`CapsMismatch`) with a structured `EmptyLink` on the bus (M749: no
     runtime producer renegotiates its output, so the chain does not run on with
     stale caps);
   - derive output candidates from `in` via the constraint;
   - intersect candidates with `downstream_feasible[i]`;
   - fixate; `configure_pipeline(in)`; element-local realloc; forward
     `CapsChanged(fixated_output)`.

When `downstream_feasible[i]` is `None` (no backward snapshot reached this edge,
e.g. a strict `DerivedOutput` downstream whose input the backward sweep cannot
invert), the arm still forwards the element's output **when the constraint pins
it to a single producible caps** (a property-driven `videoconvert`, an identity
passthrough); only a genuinely ambiguous output (a caps-driven converter with
several producible formats and nothing to choose between them) defers to the
element's own `process`. This keeps the "`c` is the output" contract below
intact for the common `... ! avdec ! videoconvert ! textoverlay ! ...` shape:
the decoder's mid-stream `CapsChanged` (its framerate settling from a negotiated
`Range` to a fixed value) would otherwise make the converter forward its *input*
(NV12) to the strict overlay, which rejects it.

The **runner**, not `process(CapsChanged)`, owns the forwarded output. A
format-changing element moves its derivation into the declared constraint
(`Mapping` / `DerivedOutput`) as the single source of truth; the solver
already consumes it at startup and at re-solve.

This fixes the element-side contract for `process(CapsChanged(c))`. The arm
calls `configure_pipeline(in)` (the element's new *input*) and then
`process(CapsChanged(fixated_output))` (its pre-fixed *output*). So `c` is the
element's **output** caps, not its input: the element forwards `c` downstream
(letting a strict sink reconfigure before the first frame) and records it as
`last_caps` to suppress the duplicate emit from its data path; the input is
already set by `configure_pipeline`. A format-changing transform must **not**
re-derive its input from `c` (e.g. `videoconvert` calling `accept_input`):
when its input and output are the same `Caps` variant (raw->raw), adopting the
output as the input silently turns the next frame into an unconverted
`X->X` passthrough. This only bites when an upstream transform emits a
`CapsChanged` mid-stream (the first of two stacked auto `videoconvert`s does so
on its first frame); a lone convert right after a source never receives one,
which is why the single-convert case was always correct. A decoder, whose
input (`CompressedVideo`) and output (`RawVideo`) are distinct variants, can
safely disambiguate the two callers by inspecting `c` (see `ffmpegdec`).

The **CapsChanged ordering invariant** is the load-bearing correctness
property. `Caps` are not stamped on each frame; they live on the link as
the most recently received `CapsChanged` packet. Correctness across a
mid-stream change therefore depends on `CapsChanged` sitting **between**
the last old-caps `DataFrame` and the first new-caps `DataFrame` in the
forward stream — not before, not after. For a format-changing element
that buffers (decoder B-frame reorder, encoder lookahead), this means
the element emits its output `CapsChanged` at the **decode/encode
boundary** in its `process` output, not at the moment it received the
input `CapsChanged`. The runner cascades that ordered event downstream;
sinks reconfigure their pools when they see it, and the next data frame
they process is unambiguously under the new caps.

#### 4.13.5 Allocation cascade

Allocation negotiation is part of the same orchestration. A coordinator task
owns refs to source / transforms / sink and orchestrates events the spawned
arms can't reach from each other:

- **Element-local realloc:** `coordinator::realloc_local` re-derives an
  element's own pool from new caps (`propose_allocation` +
  `configure_allocation`) at each mid-stream apply site.
- **N-hop re-cascade:** the `select2` combinator + per-arm control channel
  makes each transform arm interruptible at its `recv().await`, so a
  sink-side allocation proposal walks upstream one hop at a time via
  `CoordinatorEvent::ArmProposal` until it reaches the source.
- **Real resizable consumer:** `PoolStage` (`g2g-plugins`) is the element that
  acts on a mid-stream β proposal rather than only recording it (decoders fix
  their pool at codec open): each `configure_allocation` rebuilds its
  `BufferPool` to the proposal's `min_buffers` x `size_bytes`, and frames stage
  through it, so a mid-stream geometry change visibly resizes a live pool
  (`poolstage_recascade` asserts the rebuild end to end).

This is the same machinery a future mid-stream clock change or latency
adjustment uses: cross-element mid-stream coordination becomes a coordinator
event instead of an ad-hoc back-channel.

The startup cascade runs once in reverse topological order: each element
absorbs the proposal on its output edge and re-proposes onto its input edge.
Two fan structures have non-trivial joins:

- **Diamond join (tee).** A tee's single input must satisfy *both* branches at
  once, so the branch proposals are joined by a most-restrictive intersection
  (`AllocationParams::join`): the larger size, count, and alignment win, and the
  memory domain is the most-preferred member of the two branches' *accepted
  domain sets* intersected (`AllocationParams::accepts`, a `DomainSet` bitmask;
  the preference order favours GPU-resident domains over `System`). A
  single-domain branch is just `only(domain)`, so two matching single domains
  reduce to that domain and two disjoint ones to an empty set; a branch that can
  take more than one domain (a sink that reads GPU textures *or* falls back to
  System) widens its set so the join can find a domain both branches share. An
  empty intersection (a CUDA-only branch and a D3D11-only branch) is a real
  conflict, no single producer pool serves both, and fails the whole negotiation
  loud with `G2gError::AllocationConflict` rather than silently honouring one and
  copying for the other. This is distinct from `AllocationParams::merge`, the
  asymmetric fold the linear upstream walk uses where the consumer-most proposal
  legitimately dictates the domain.
- **Producer reconciliation.** Domain choice is two-sided: a producer
  advertises the set of domains it can *emit* (`output_domains`, default
  `only(output_memory())`), and at the buffer-pool origin (the source) the
  joined downstream proposal is reconciled against it
  (`AllocationParams::resolve_for_producer`): intersect the accepted set with the
  producer's capability and settle on the most-preferred survivor, so a graph
  keeps the frame copy-free when both ends can and falls back to System when the
  producer cannot reach the consumer's preferred domain. No shared domain is an
  `AllocationConflict` (a genuine case for an auto-plugged domain converter, a
  later track). Reconciliation runs at the source, not at every hop: a plain
  transform is a memory-domain pass-through (it forwards whatever domain it
  receives), so enforcing its `System` default mid-cascade would wrongly reject a
  GPU proposal merely passing through. A transform that *is* a genuine domain
  producer (a hardware decoder allocating its own output surfaces) consumes the
  same contract from inside the element: its `configure_allocation` calls
  `resolve_for_producer` against its own `output_domains` to settle its output
  domain. `NvDec` does exactly this: it advertises `{Cuda, System}` and
  either keeps the decoded NV12 surface device-resident (zero-copy) or downloads
  it, chosen by the negotiated proposal alone, validated end-to-end on an RTX
  3060.
- **Converter auto-plug.** When no shared domain exists (the negotiation
  would otherwise fail loud), `auto_plug_domain_converters` splices a memory-domain
  converter instead. A pre-solve graph pass: for each edge it traces the producer
  domain through structural tee/demux nodes (`output_domains`) and, if disjoint
  from the consumer's declared `input_domains` (caps-free, default
  `DomainSet::ALL`), splices a caps-`Identity` converter from a registered factory
  (`Graph::insert_on_edge`, so the caps solve is undisturbed). `g2g-plugins`
  provides the CUDA factory (`Cuda->System` = `CudaDownload`, `System->Cuda` =
  `CudaUpload`), so e.g. a System NV12 source feeding `NvEnc` (CUDA-only) gains a
  `CudaUpload` with no hand-wiring. Negotiation settles a shared domain when one
  exists; the auto-plug bridges when one does not; an unconvertible pair still
  fails loud.
- **Muxer boundary.** A muxer states its per-pad demand through
  `MultiInputElement::propose_allocation_for_input(pad, caps)` (default `None`,
  so a plain container muxer imposes nothing). At startup the runner stores it on
  each input edge so the demand crosses the boundary and re-cascades up that
  branch independently (a device-resident interleave muxer asking each video pad
  for GPU buffers). Mid-stream the same crossing holds: a `CapsChanged` on one
  pad re-derives that pad's proposal and re-cascades it up *that pad's branch
  alone* via the `Recascade::target` override (the node-keyed walk would hit
  every input), leaving the other inputs untouched. The muxer's byte output has
  no memory-domain tie to its inputs, so its output-edge proposal is not
  absorbed.

#### 4.13.6 Fan-out and fan-in

`run_source_fanout` per-branch re-solves a mid-stream `CapsChanged` via
`re_solve_downstream_dyn_sink`. Branches run in independent arms, so the
re-solves are concurrent (max of single-branch cost, not sum). The default
failure policy is strict: a branch whose constraint rejects the new caps
fails the fan-out loud (`CapsMismatch`); a future `FanOutPolicy::AllowBranchDrop`
opt-in is anticipated for graceful degradation.

A tee broadcast is **zero-copy**. Before fanning out, the runner
calls `MemoryDomain::make_shareable` once, which turns the frame's memory into a
refcounted handle; each branch then gets a second handle via `MemoryDomain::share`,
a refcount bump rather than a copy. The GPU domains and the shared-CPU
`SystemView` are handle-shared by construction; owned-CPU `System` bytes are made
shareable by *moving* the `Box<[u8]>` into an `Arc<Box<[u8]>>` (a move, not a
re-copy, which `Arc<[u8]>` would force), and a pooled buffer into an
`Arc<PooledBuffer>` that returns to its pool once the last branch drops. The
share is read-only: a branch that mutates pays copy-on-write
(`SystemSlice::as_mut_slice` reclaims a uniquely-held `Arc` without a copy, else
deep-copies), so siblings never alias a mutation. So a decoded frame, on CPU or
GPU, fans out to several consumers (e.g. inference + display) with no per-branch
copy, where `System` previously deep-copied per branch and a GPU frame failed loud.

`run_muxer_sink` solves each `source ↔ muxer-input` pair at startup,
per-input re-solves on mid-stream change, and eagerly re-emits the muxer's
output `CapsChanged` downstream when the merged output caps change as a
function of an input change. `MultiInputElement` exposes
`caps_constraint_as_input(idx)` and `caps_constraint_for_output()` for the
solver to consult per-input.

A muxer whose merged output *is* one of its inputs (an identity-passthrough
mix: a `TextOverlayN` painting a subtitle stream onto video, a watermark, an
alpha mixer) returns `output_follows_input() = Some(pad)` instead of declaring
output caps. The solver then derives the output edge by coupling it to that
input pad's edge (`NodeConstraint::Muxer { follows }`), so the element negotiates
without knowing its output geometry up front; the fixpoint solve makes the
coupling order-independent. The default `None` is the independent-output case (a
container interleave, a fixed-size compositor) declared by
`caps_constraint_for_output`.

A fan-in muxer interleaves its inputs by **presentation timestamp**, not
arrival order: `InterleaveMux` buffers frames per input in an `InputAggregator`
and releases the globally earliest-PTS frame only once every still-contributing
input has one queued (`InputAggregator::take_earliest_by`), the `GstAggregator`
collect-and-pick-earliest rule. Because each input's PTS is monotonic, holding
output until every contributor has a head guarantees the released frame is
globally earliest, so a slow input never delivers a frame that should have
preceded an already-emitted one; an input that ends drops out of the merge (its
buffered tail flushed in order). Frames carry their own caps, so reordering is
format-safe. Ordering is by PTS; a container muxer needing decode-order (DTS)
interleaving keys on that instead. This is distinct from the synchronized-*round*
collection (`take_earliest_by`'s sibling `take_round`) a compositor / audio mixer
uses, where every input contributes one item per output.

The same PTS merge is also available **at the runner level**: a
`MultiInputElement` returning `input_pts_ordered() == true` is driven by
`muxer_arm_pts` instead of the default arrival-order `muxer_arm`. That arm owns an
`InputAggregator<Frame>` and calls `process(pad, DataFrame(..))` in global PTS
order (the same collect-and-pick-earliest rule), so an element wanting time-aligned
input, a multi-camera grid or PTS-synchronized compositor, gets it without
hand-rolling its own aggregator. Per-input `Eos` (flush + the merged-EOS
aggregation) and `CapsChanged` (MX-1 / MX-2 re-solve) are handled exactly as in
`muxer_arm`; only `DataFrame`s are reordered. The default stays arrival-order
round-robin, so the existing element-level mergers (`InterleaveMux`, `tsmuxn`,
`mp4muxn`) are unchanged; the runner arm is the alternative for elements that would
rather not carry the buffering themselves.

Over the DAG, a node-keyed `GraphCoordinator` walks a sink's re-derived
allocation proposal upstream through tees via `in_edges` (sources and muxers
terminate the walk), and a per-edge `graph_downstream_feasibility` snapshot
steers each transform's Caps-α output on a mid-stream change.

Two flavours of fan-in element exist. `InterleaveMux` (`mux.rs`) is a
*multiplexer*: it forwards every input's frames straight through (each frame
carries its own caps), combining encoded tracks into one stream. `Compositor`
(`compositor.rs`) is a *pixel mixer*: it overlays N raw RGBA8 inputs onto one
output canvas at configurable position, z-order, and per-pad alpha (the
`videomixer` / `compositor` analog — picture-in-picture, camera grids, sub-window
UIs). It is CPU and `no_std`-baseline like the other raw-video transforms, with
straight source-over alpha blending and left/top clipping. Because a mixer must combine *simultaneous* inputs rather than
interleave, it caches the latest frame per input and uses **input 0 as the
timing driver**: one composited output frame is emitted per input-0 frame,
overlaying whatever the other inputs have most recently delivered. At startup it
briefly buffers input-0 frames (bounded) until every overlay has produced once,
so a late-starting overlay (camera warm-up) still appears; on buffer overflow the
oldest is emitted overlay-less rather than dropped, so a free-running background
never stalls or latches the overlay on one stale frame. The output canvas size
and framerate are fixed at construction; per-input geometry is whatever each
input negotiates (`Accepts(RGBA8)` per pad, `Produces` the fixed canvas). A pad
may also scale its input as it composites (`CompositorPad::with_size`, integer
bilinear), so a downscaled camera inset needs no upstream `VideoScale`.

**Runtime request pads.** Both fan directions can grow their pad count *while the
pipeline runs*, the GStreamer request-pad analog, without an executor `spawn`: the
no-spawn `DynamicJoin` primitive (`runtime/join.rs`) is a `join_all` that also
polls a control channel and folds newly-arrived arms into the running poll set,
completing once the channel closes and every arm resolves. On the **fan-out**
side, `run_source_router_dynamic` returns a `DynamicFanoutHandle` whose
`add_branch` attaches an output branch mid-run; the branch configures from the
fan-out's *sticky caps* (the source's fixated output caps, replayed into each
branch the moment it attaches) and then receives its share of frames.
`run_source_tee_dynamic` is the *broadcast* variant: each `DataFrame` is
shared to every branch via the zero-copy path (`make_shareable` once, then a
refcount handle per branch), so an inference branch and a display branch both see
the whole stream with no byte copies; round-robin (`Router` model) and broadcast
(`tee` model) share one driver, differing only in `DataFrame` distribution.
`run_aggregator_dynamic` is the **fan-in** dual: a `DynamicFaninHandle`
whose `add_input` attaches a source to a running terminal aggregator. The
aggregator declares a fixed pad capacity (`input_count`); the handle reserves the
next pad index atomically (rejecting past capacity, the dark-slot bound), the
single aggregator arm owns `&mut` and fixates + configures each new pad inline
(no aliasing), and per-pad-tagged frames merge as in `run_fanin_session`. The run
ends once the handle is dropped *and* every attached input has reached EOS (the
`DynamicJoin` completion rule). In all three the pending-pad set is drained before
each blocking select, so a pad requested before a frame is never missed.
`run_muxer_sink_dynamic` adds the trailing **sink**: the muxer's merged output is
written to a sink arm rather than discarded, with the output caps coupled without
a global re-solve. Because inputs attach one at a time, the merged output firms up
as pads configure, so the muxer arm emits a `CapsChanged` to the sink whenever the
derived `output_caps` changes (the dynamic analog of the static `run_muxer_sink`
MX-2 coupling) and the sink configures against it before the first merged frame;
when every input has ended the muxer arm closes the merged link with `Eos`, ending
the sink arm. This is the `run_muxer_sink` shape extended to runtime-added inputs
(attach a late audio track to a running `muxer ! filesink`).

#### 4.13.6a Bins and ghost pads (flattening)

GStreamer's `GstBin` is a runtime container: a node in the pipeline that holds
child elements, manages their state, and exposes interior pads as *ghost pads*.
g2g implements the same user-facing capability (reusable named subgraphs +
ghost pads) but as **construction-time flattening**, not a runtime container.
The reason is the same one in §4.9.3: g2g composes typed graphs ahead of the
run, so grouping for reuse and pad exposure can happen before validation, and
the runtime never needs a hierarchy to manage.

The whole mechanism is one primitive: `Graph::merge(inner) -> NodeIdOffset`
appends another graph's nodes and edges, re-basing the merged-in `NodeId`s by
the host's current node count. Because nodes are a flat `Vec` indexed by
`NodeId` and edges carry only pad indices, the merge is a pure index shift; the
returned `NodeIdOffset` translates the inner graph's ids (`apply` / `apply_pad`)
into the host's space. A `Bin<E>` is a `Graph<E>` plus a list of interior pads
designated as ghost inputs / outputs (1:1 with one internal pad, as in
GStreamer). `Graph::add_bin` merges the bin and returns a `BinInstance` whose
`input(i)` / `output(i)` are host-graph pad ids, linked like any other pad. A
bin is never validated alone: its ghost pads are intentionally unlinked inside
the bin and acquire their peer when the host links the `BinInstance`, so the
host's `finish()` is the single validation point.

Crucially this adds **no `NodeKind` variant**: a bin's interior nodes become
first-class host nodes on flattening, so the solver (§4.13.2) and runner
(§4.13.3) drive them with zero awareness bins ever existed, and none of the
exhaustive `NodeKind` match sites change. The decode-chain splices
(`Registry::decodebin`, the `uridecodebin` / `decodebin` launch macros) already
flatten subgraphs ad hoc at the element-vector / parse-item layer; they predate
this primitive and are left as-is rather than rerouted through it.

Out of scope (a later milestone, only if needed): a runtime `NodeKind::Bin` with
recursive solve/run, per-bin state transitions, and bus-message bubbling, i.e.
GStreamer's full hierarchical `GstBin`. None of that is required for reuse,
ghost pads, or a nestable decodebin.

#### 4.13.7 Pad templates

Static metadata for tools that need to query pad compatibility without
constructing the element. `PadTemplate` + the `PadTemplates` trait expose
`pad_templates()` as an associated function; `pad_link` and `types_can_link`
run the solver against two element types' static templates for pre-
instantiation compatibility checks. The runtime `caps_constraint_as_*`
remains the instance-level (possibly narrower) view.

#### 4.13.8 ACCEPT_CAPS and CapsFilter

Fall out of the constraint surface:

- **ACCEPT_CAPS query** is `constraint.accepts(&caps)`, a pure check against
  the constraint's set. No runtime round-trip; the element's constraint
  already describes everything it would accept.
- **`CapsFilter`** is an `Identity(specific_set)` pass-through. Inserted
  anywhere in a pipeline to force a narrowing.

#### 4.13.9 Auto-plug and the element registry

`decodebin`-equivalent, built on the pad-template metadata (§4.13.7) and the
solver. `g2g-core::runtime::autoplug` is two layers split by what they need:

- **Search** (`runtime`, `no_std`). `ElementDesc` is a name plus an element
  type's static pad templates. `find_chain(descs, input, target, max_depth)`
  is a breadth-first search over caps states: each edge is an element whose
  sink accepts the current caps (acceptance reuses `pad_link`, so an
  `Unfixable` link counts as compatible, exactly as `types_can_link`), and the
  search advances along that element's source-pad caps until one satisfies the
  `target` shape predicate (`is_raw_video` is the canonical `decodebin`
  target). The shortest chain wins; an element is never reused on a path, so a
  same-media-type parser (H.264 → H.264) cannot loop. The result is an ordered
  `Vec<ChainLink { index, output }>`: the search picks element *types* and the
  source-pad caps each was matched to produce, leaving geometry / framerate to
  fixate later at instance negotiation.

- **Registry** (`std`). `Registry` pairs each `ElementDesc` with an
  `ElementFactory` whose constructor is `fn(&Caps) -> Box<dyn DynAsyncElement>`,
  receiving the per-hop chosen output caps so a format-flexible element (a
  converter, a multi-format decoder) configures itself correctly.
  `Registry::autoplug` runs the search and instantiates the chain;
  `Registry::decodebin(graph, from, to, input, target, max_depth)` splices it
  into a `Graph<GraphNode>` between two existing pads (an empty chain links
  `from → to` directly), returning a sub-graph onto `run_graph`. Real element
  types publish templates via the `PadTemplates` trait (`FfmpegH264Dec`:
  H.264 → NV12 / I420), so a real decoder is registered and auto-plugged, not
  just synthetic descriptors.

Source-side `typefind` is not needed: a g2g source declares its output caps via
its source pad template / `caps_constraint`, so the caps feeding `decodebin` are
known without sniffing the byte stream.

A single-stream demuxer is the one place the *demux output* is content-ambiguous:
it fixes its output pad before parsing any byte, so `TsDemux` defaults to a video
port and a bare `filesrc location=X.ts ! decodebin` on an audio-only stream would
auto-plug a video decoder and fail negotiation. `expand_decodebin` resolves this
with a `PrimaryStreamHook` (`register_primary_stream`, a `Default`-friendly
fn-pointer slot, cross-crate like the playbin hooks): for a file-backed container
it sniffs a bounded prefix (`ts_primary_stream` reads the PMT via
`forwardable_streams`) and, finding no video track, names the single-stream demux
plus its stream-selection property (`tsdemux stream=aac`) and the audio elementary
caps, so the search builds `filesrc ! tsdemux stream=aac ! <audio decoder> ! …`.
The hook declines a container with a video track (the default video port is
correct), leaving A/V behavior unchanged. A `mp4_primary_stream` sibling does the
same for MP4, sniffing the `moov` and naming `qtdemux stream=aac` so an audio-only
`.m4a` / `.mp4` plugs an audio decoder too.

- **playbin / uridecodebin** (`std`). `Registry::build_playbin(source_name,
  sink, target, max_depth)` assembles a complete `source → chain → sink` graph
  from a *named* registered source. `build_uridecodebin(uri, sink, target,
  max_depth)` is the URI front door over it: it parses `uri` (a minimal
  `scheme://rest` split — core pulls no URL crate), dispatches on the scheme to
  a registered `UriSourceFactory` that builds the source *from the URI*
  (`udp://host:port`, `file:///clip.mp4`, `rtsp://…`, `v4l2:///dev/videoN`), and
  auto-plugs the decode chain to `target`. The scheme handlers are the analog of
  GStreamer's `GstURIHandler`; the concrete ones live in `g2g-plugins`
  (`uridecodebin.rs`), each gated to its source's feature, so an app registers
  only the schemes its build supports. A handler reports the *media type* it
  produces (geometry resolves at negotiation), which is all the chain search
  needs to pick the right decoder.

- **playbin (multi-stream front door)** (`std`). Beyond the
  single-stream expansion, g2g's `playbin` can split a container into *all* its
  streams and decode each to its own sink, built on the stream-collection model:
  a demuxer announces every track as a `BusMessage::StreamCollection` (for
  `MkvDemux`/`MkvDemuxN` and `TsDemux`/`Mp4Src`), the app selects among them
  via a `StreamSelectController`, and the multi-output `MkvDemuxN` (a
  `MultiOutputElement`) routes N elementary streams to N ports in one parse. `Registry::build_playbin_graph` assembles
  `source → demux → {decode chain → sink}` per `PlaybinPort`, with each port's
  branch statically negotiated against its codec via `port_output_caps` /
  `NodeConstraint::Demux` so a real decoder configures at startup, not
  only at runtime retype. The gst-launch front door is `playbin uri=X`:
  `parse_launch` routes a *lone* `playbin` to a registry `PlaybinHook`
  (`register_playbin`, a `Default`-friendly fn-pointer slot) that probes the
  container and auto-builds the multi-stream graph. Cross-crate by design: the
  text DSL is core, the Matroska probe (`mkv_playbin`: read a bounded prefix,
  parse `Tracks`, one branch per `forwardable_streams` entry, video→autovideosink
  / audio→autoaudiosink) is `g2g-plugins`. The hook declines (`Ok(None)`) for a
  non-`file://` URI or non-Matroska container, falling back to single-stream
  `playbin`; it supplies a Matroska byte `FileSrc` via
  `build_playbin_graph_with_source` rather than the `file://` handler's
  MP4-self-demuxing source. The hook slot is a *list*: `register_playbin`
  appends and `parse_launch` tries each in turn, so one hook per container type
  coexists — `ts_playbin` is the MPEG-TS sibling (`TsDemuxN` multi-output
  demuxer), and a TS file is handled by it while an MKV file is handled by
  `mkv_playbin`, each declining the other's container. `mp4_playbin` is the
  fragmented-MP4 sibling (`Mp4DemuxN` multi-output demuxer), the multi-track
  read-side analog of the single-video-track `Mp4Src`: it parses every `moov/trak`
  (`fmp4::parse_all_tracks`) and routes each sample to the port matching its
  `track_ID`, picking the fragmented path (`parse_fragments_multi`, by `tfhd`
  `track_ID`) or the progressive `moov`+`mdat` path (`parse_progressive_multi`,
  per-track `stbl` sample tables) by the presence of a `moof`. A *fragmented*
  file is demuxed **progressively**: once the `moov` is buffered the layout
  is known (`mvex` = fragmented), and each complete `moof`+`mdat` fragment is parsed,
  emitted, and drained as it arrives, so a live / long CMAF stream flows segment by
  segment with a bounded buffer rather than stalling until EOS (the carrier that
  makes the fMP4-HLS caption overlay usable live). A *progressive*
  (non-fragmented) file has one big `mdat` its sample tables index, so it is
  accumulated whole and parsed at EOS. The per-sample emit (caps announce,
  parameter-set prepend, ADTS re-framing) is shared, with the per-port "caps
  announced" / "parameter sets owed" flags persisting across the incremental
  fragments. Compressed-audio port caps negotiate as `0/0` (AAC caps
  intersect by strict equality, so the concrete channel layout / sample rate is
  refined per port at runtime via `CapsChanged`, not advertised for the static
  branch solve). Demuxed AAC is re-framed to ADTS from the track's `esds`
  AudioSpecificConfig, so the audio elementary stream is self-describing and
  decodes without out-of-band config, symmetric with the in-band video parameter
  sets. Encrypted (cbcs / MPEG-CENC) multi-track files are supported under the
  `mp4-cenc` feature: `parse_all_tracks` reads `encv` / `enca` per-track
  `cenc` defaults, `parse_fragments_multi` decrypts each track's samples (per
  `traf` `senc`) via a callback, and `Mp4DemuxN::with_cenc_key` supplies the
  clear-key content key (the cbcs primitive lives in a shared `cenc` module, used
  by both this and the HLS fMP4 path). A timed-text `trak` (handler `text` / `sbtl`
  / `subt`) carrying a `tx3g` 3GPP-timed-text sample entry is read as a
  `TrackKind::Text` and fans out as a `Caps::Text { Utf8 }` port: the
  container supplies the per-cue timing (the sample table's PTS + duration), and a
  sample's 2-byte length prefix is stripped to the UTF-8 cue, so an embedded
  subtitle track feeds `SubParse` / `TextOverlayN` like a sidecar file would.
  `wvtt` / `stpp` sample formats are recognized-but-declined. When such a file also
  carries a video track, `mp4_playbin` routes the video branch through a
  `TextOverlayN` fed by the subtitle track
  (`uridecodebin::build_mp4_subtitle_overlay`): `Mp4DemuxN -> { video: decode ->
  videoconvert(RGBA8) -> overlay.video ; text -> overlay.text } ->
  videoconvert(NV12) -> autovideosink`. This is the one non-linear `playbin`
  branch: the decoder is auto-plugged, but the `videoconvert`s bracketing the
  overlay are wired explicitly (they are caps-driven `register_launch` elements
  outside the auto-plug search pool, and the overlay requires RGBA8 in/out while a
  display sink requires NV12), and the text port joins the overlay's second pad
  (through `SubParse` when the cue payload is a structured format, straight in for
  plain-UTF8 `tx3g`). An MP4 with no subtitle track keeps the plain per-stream
  fan-out. `hls_playbin` is the HLS sibling:
  it probes a `hls://` master playlist (the scheme maps to an `https` origin),
  discovers the selected variant's renditions (`hls` parses `#EXT-X-MEDIA`
  alternate renditions and the variant's `AUDIO` / `SUBTITLES` / `VIDEO` group
  bindings; `hlssrc::variant_streams` maps the variant's `CODECS` + alternate audio
  to streams), and for a muxed MPEG-TS variant assembles `HlsSrc → TsDemuxN →
  {decode → sink}` (the network-free assembly factored into `build_hls_ts_fanout`).
  It declines to a single-stream `hls_handler` for a media-only / fMP4 / single-
  stream variant or a probe failure. The master probe is network-coupled
  (validated live, not CI).

- **Gapless playback** (`std`). The playbin `about-to-finish` + next-`uri`
  analog: `GaplessSrc` (`g2g-plugins`) concatenates a playlist of sources into
  one continuous, monotonically-timed stream, reusing the downstream decode chain
  across items (the read-side analog of GStreamer reusing one decodebin across
  URIs). It wraps a current `DynSourceLoop` and a shared `GaplessController`
  (core, the `SeekController`-shaped app<->source channel: an `enqueue` playlist
  queue, an about-to-finish back-channel, a latching `finish`, and a wakeful
  `wait_event` idle). The source plays the current item, posts about-to-finish
  when nothing is queued behind it (so the app enqueues the next item *during*
  playback for a seamless swap), and on the item's EOS pulls the next, rebasing
  its PTS/DTS onto the running timeline via an interposing `ShiftSink` that also
  swallows the inner item's `Eos` — so the only terminal `Eos` is the one
  `GaplessSrc` emits when the `finish`ed playlist drains. This is the source-swap
  counterpart of the segment loop (which loops *one* item via a `SEGMENT`
  seek); both are poll-based with a wakeful idle. An *instant* (flushing) switch
  that preempts the current item is also supported
  (`GaplessController::switch_now`, the `instant-uri` analog): `GaplessSrc` races
  the item's `run` against `wait_instant` with `select2`, so a `switch_now` drops
  the run future (cancelling the inner source mid-stream), pushes a `Flush`, and
  resets the timeline before playing the requested source. v1 concatenates
  same-codec items (a per-item caps refinement still flows via the inner source's
  `CapsChanged`). The `gapless_playbin` helper is the one-call builder over
  this: from a playlist of URIs it wraps the first source in a `GaplessSrc`,
  pre-enqueues the rest on a shared `GaplessController`, and auto-plugs one reused
  decode chain to the sink (via the factored-out `Registry::build_source_decodebin`),
  returning the graph + the controller for the app to drive. An A/V offset is a
  separate `AvOffset` transform
  (`avoffset`): a pass-through that shifts a branch's PTS/DTS by a signed `offset`
  ns (positive delays, negative advances, clamped at 0), the `av-offset` analog,
  placed on the audio (or video) branch of a multi-stream graph to re-align it.

- **Memory-feature-aware selection**. The `Caps` algebra encodes media
  type, format, and geometry but *not* the memory domain a producer emits, so a
  GPU-resident decoder (`NvDec` → NV12 in `MemoryDomain::Cuda`) is
  indistinguishable from a CPU one by caps alone. Rather than thread the domain
  through every `Caps` (446 construction sites, and it is orthogonal to the
  format algebra), it rides on the auto-plug metadata, as one field of a small
  `CapabilityDescriptor` (`ElementDesc::capabilities`): `output_memory`
  (`MemoryDomainKind`, the GStreamer `memory:CUDAMemory` caps-feature analog), an
  `Acceleration` (hardware vs software, independent of the domain: an ffmpeg
  VA-API decoder is hardware yet downloads to `System`), and a numeric `rank`.
  These are set per factory via `ElementFactory::produces(kind)` / `.hardware()` /
  `.rank(n)`, all defaulting to (software, `System`, 0).

  This is deliberately *not* GStreamer's flat global rank. A single integer can't
  express that the best element is context-dependent: a hardware decoder that
  keeps frames on the GPU beats a faster one that forces a PCIe download when the
  consumer is GPU-resident (g2g measured exactly this, the NVDEC-to-system-memory
  floor). So `CapabilityDescriptor::score(ctx)` ranks a candidate against a
  `SelectionContext { preferred_memory, prefer_hardware }`: a memory-domain match
  dominates, then a hardware preference, and `rank` is only the deterministic
  tiebreaker among otherwise-equal candidates (the explicit-override knob, the
  genuinely useful 20% of GStreamer's rank). `find_chain_with(.., ctx)` /
  `Registry::{autoplug,autoplug_names}_with(.., ctx)` score-order which candidate
  is *tried first*; it is still breadth-first (a shorter chain always wins), and a
  default `ctx` scores every candidate equally, so the visit order is registration
  order and a plain pipeline is unchanged (`NvDec` registered last never hijacks a
  CPU path). `find_chain_preferring` / `{autoplug,decodebin}_preferring(.., domain)`
  remain as the memory-only special case. Ranking matters *only* on the auto-plug
  path; an explicit typed graph names its element, so the descriptor never touches
  the core.

#### 4.13.10 Current limits

The solver is **arc consistency** (constraint propagation over per-link caps),
not a complete CSP search. That bounds exactly where it is complete and where it
is not:

- **Linear chains are complete.** A linear pipeline is a tree of binary
  (adjacent-link) constraints, and arc consistency is complete for
  tree-structured binary CSPs: if a satisfying assignment exists it is found.
  With `DerivedCoupled`'s field-level coupling (§4.13.1), a downstream pin on a
  passthrough field couples back through any number of passthrough transforms
  (`videoscale ! videoconvert ! caps`, and deeper). This family is closed.

- **Backward coupling through a format-changing (`DerivedOutput`) transform is
  partial, over its *invertible* fields.** A `DerivedOutput` is opaque, but its
  invertible fields are recovered by probing (`discover_passthrough`): a
  downstream pin on a passthrough field couples back through a decoder / rescaler,
  in both the full-chain solve and the mid-stream snapshot. A field
  the transform genuinely *re-derives* (a scaler's geometry) still cannot be
  inverted: a downstream pin on it does not narrow the input, and the snapshot
  leaves the upstream unconstrained on it (so a re-deriving transform mid-stream
  picks freely and the pin is enforced loud downstream if violated). This is the
  arc-consistency boundary, not a missing feature: a partial inverse over the
  invertible fields is exactly what is modelled.

- **Non-tree topologies: arc consistency plus a backtracking fixation.** Arc
  consistency is incomplete on cyclic constraint graphs, so for a true *diamond*
  (a tee whose branches diverge through format-changing transforms and re-converge
  at a fan-in) the per-link sweep can leave each edge with a locally-valid domain
  whose *greedy* per-edge fixation picks a jointly-impossible combination (two
  branches mapping the shared tee value to outputs whose alternative orders
  disagree). `solve_graph` therefore fixates by **backtracking search** over the
  arc-consistency-narrowed domains, not greedily: it assigns one fixated `Caps`
  per edge in id order, trying each edge's greedy choice first and pruning the
  moment a fully-assigned node violates its relation (a tee's branches must all
  carry its input; a transform's `(in, out)` must be a real `Mapping` pair /
  `Identity` equality / `f(in)` image). A chain or an independent fan-out has
  single-candidate domains and so fixates byte-for-byte as the greedy code did;
  only a genuinely coupled diamond explores alternatives, and one with no
  jointly-valid assignment fails loud (`NoConsistentFixation`). Diamonds are now
  solved, not a caveat. (A muxer that itself *couples* its input pads, beyond the
  per-pad accept sets the constraint vocabulary expresses today, would be the next
  step up; the search already has the shape to enforce it once such a constraint
  exists.)

- **Cross-field validity within one element is not modelled.** Constraints
  *among an element's own caps fields* (a 4:2:0 format requiring even dimensions,
  chroma siting) are non-binary and are deliberately kept out of the declarative
  constraint: caps fields stay independent within an alternative, and an element
  enumerates valid combinations as separate `CapsSet` alternatives instead. The
  hard cases were judged not worth a declarative encoding.

- **Allocation is a separate cascade.** Buffer-pool / stride / alignment
  negotiation (§4.13.5, the allocation query) runs after caps fixation, not
  folded into the caps CSP. A downstream allocator whose layout requirement
  should feed back into the *caps* choice is not expressed; this is the most
  likely future pressure point as real GPU/hardware allocators land.

The fixation step is now a bounded backtracking search (above), so a diamond is
solved rather than greedily mis-fixated. Full *path consistency* during the
narrowing sweep (versus arc consistency plus search at fixation) is still not
implemented, but the search closes the practical gap: every shape that arises is
either complete or fails loud, and a coupled diamond with a satisfying assignment
now finds it instead of mis-fixating.
