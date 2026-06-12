# M16 Workaround #3 — forward in-band reconfigure

> Implementable spec for the last open M16 piece. Expands
> `DESIGN-M16-caps-nego.md` §6 from concept into a phased plan with the
> design decisions called out for sign-off. No code has landed yet.

## 1. What the workaround is

All three decoders (`FfmpegH264Dec`, `MfDecode`, `VaapiH264Dec`) treat an
incoming `PipelinePacket::CapsChanged` as a no-op:

```rust
PipelinePacket::CapsChanged(_) => {
    // Upstream H.264 caps are swallowed; we emit NV12 CapsChanged from
    // the decoder's first decoded frame and on geometry changes.
}
```

The decoder instead *self-detects* its output geometry from each decoded
frame and emits its own `CapsChanged(NV12, w, h)` downstream, de-duped
against `last_caps`. The runner re-configures the next element on that
forward `CapsChanged` (`runner.rs:218` for the sink, `:569` for a
transform) before any new-caps `DataFrame` reaches it.

So today's behavior is functionally correct for the linear
`rtsp -> decoder -> sink` chain: a mid-stream resolution change works,
just driven by decode-time self-detection rather than by the upstream
caps event.

## 2. Why it is still a workaround

1. **The constraint change bypasses the solver.** The decoder is a
   `DerivedOutput` element; an upstream caps change should re-derive its
   output through that declared constraint and re-solve the downstream
   subgraph. Instead it is recovered heuristically from pixels. A
   downstream allocator that sizes pools from negotiated caps, or a
   second format-changing transform, never participates.
2. **Silent drop.** Swallowing means the decoder cannot reject an
   illegal mid-stream format change (e.g. H.264 -> VP9) loud; it just
   ignores it.
3. **Non-linear topologies.** Fan-out / muxer branches downstream of a
   boundary can't be re-solved off a self-detected event.

## 3. The ordering constraint (why the naive fix is wrong)

The obvious "fix" is: on `CapsChanged(in)`, run the decoder's
`DerivedOutput(in)` and forward `CapsChanged(out)` immediately. **This
corrupts the stream.** Decoders buffer for B-frame reorder, so across a
resolution boundary, old-geometry frames may still drain *after* the
input `CapsChanged` arrives. Forwarding the derived output caps eagerly
reconfigures the sink to the new geometry before the buffered old frames
land, so those frames are presented under the wrong configuration.

The reconfigure event must sit in the forward stream **between the last
old-caps `DataFrame` and the first new-caps one**. That is exactly the
position the decode-time self-detected `CapsChanged` occupies today.
`Frame.caps` already tags every `DataFrame` with the caps it was produced
under, so the ordering information exists; the fix must preserve it.

## 4. Design

**Principle.** The reconfigure is an ordered forward event positioned at
the caps boundary in the data stream. `CapsChanged` already is that
event. The work is to (a) route it through the declared constraint
instead of pixel heuristics, (b) make the runner treat a boundary's
forward `CapsChanged` as a downstream *subgraph re-solve*, not just a
single-element `configure_pipeline`, and (c) stop dropping the input
event silently.

### Decisions for sign-off

- **D1 — carrier.** Reuse `CapsChanged` as the in-band reconfigure
  carrier (it is already ordered and already drives downstream
  `configure_pipeline`), vs. add a dedicated forward `Reconfigure`
  packet variant. *Recommendation: reuse.* Add semantics, not a variant;
  a new variant would duplicate `CapsChanged`'s ordering role.
- **D2 — output derivation.** The boundary's output caps come from its
  `DerivedOutput` closure (single source of truth) rather than ad-hoc
  geometry helpers. For a linear chain the decode-time geometry equals
  `DerivedOutput(input)`, so the two agree; the design keeps emitting at
  the correctly-ordered decode boundary but derives the value from the
  constraint (and asserts equality in debug).
- **D3 — runner re-solve.** On a forward `CapsChanged` crossing a format
  boundary, the runner re-solves the downstream subgraph
  (boundary -> ... -> sink) and re-issues configure to every link whose
  assignment changed, instead of configuring only the immediate next
  element. Required for chains with >1 downstream element or an
  allocator. (Today's `runner.rs:218/569` configure only the adjacent
  element.)
- **D4 — acknowledge, don't swallow.** The decoder consumes the input
  `CapsChanged`: records it as the current input caps and validates it
  (reject an incompatible format change loud), rather than dropping it.

### Phased implementation

- **Phase A (linear, low risk, CI-testable).** Decoders stop silently
  swallowing: on `CapsChanged(in)` they validate the format (reject
  non-H.264 with `CapsMismatch`), record `in` as current input caps, and
  keep emitting the output `CapsChanged` at the decode boundary (ordering
  preserved). Add a debug assertion that the decode-time output caps
  equal `DerivedOutput(recorded_input)`. Behavior is unchanged for valid
  streams; the silent drop becomes validate + record. Testable with a
  fake in-memory decoder, no hardware.
- **Phase B (subgraph re-solve).** Runner, on a forward `CapsChanged` at
  a boundary, re-solves boundary -> sink and re-configures all changed
  links, not just the next element. Unlocks downstream allocators and
  multi-element downstream segments. Testable with a 3+ element fake
  chain.
- **Phase C (non-linear).** Extend the subgraph re-solve to fan-out /
  muxer downstream of a boundary.

## 5. Frame tagging invariant

Already satisfied structurally: every `DataFrame` carries `Frame.caps`.
The spec promotes this to a stated invariant: *a `DataFrame`'s caps are
authoritative for that frame; downstream elements must honor per-frame
caps, not assume the last `CapsChanged` applies to every subsequent
frame.* Phase A should audit the display sinks to confirm they key on the
frame they render, not a cached caps, across a reconfigure.

## 6. Testing

A fake decoder driven through a resolution-change sequence, no hardware:

```
CapsChanged(in@A) -> frames@A ... -> CapsChanged(in@B)
  -> (buffered frames@A drain) -> frames@B
```

Assert downstream observes `CapsChanged(out@A)` before any A-frame, and
`CapsChanged(out@B)` only *after* the last A-frame and before the first
B-frame. This is the ordering property §3 is about and is the regression
guard for any future change to the swallow path.

## 7. Open questions for the architect

- D1 and D3 are the load-bearing choices; confirm before Phase B.
- Is **Phase A alone** enough to declare #3 retired for the linear case
  (current production topology), deferring B/C until an allocator or a
  branched topology actually needs them?
- Interaction with the existing reverse `Reconfigure` path
  (`PushOutcome::Reconfigure`, `SourceLoop::reconfigure`): the forward
  reconfigure and the downstream counter-proposal should not race. A
  forward `CapsChanged` re-solve that a downstream element rejects must
  fall back to the existing `ReFixate` -> `request_reconfigure` flow
  (`runner.rs:233`).

## 8. Out of scope

- The codec-vs-raw `Caps` split (`DESIGN-M16-caps-nego.md` §12) would
  make the decoder's input/output media types distinct types, which
  simplifies D4's validation. Independent; not required here.

## 9. Mid-stream allocation re-cascade (deferred-but-specified)

### 9.1 Why this matters for the capability target

The framework target is *at least as capable as GStreamer, ideally
lower latency*. GStreamer's allocation model (`GstQuery::Allocation`,
`gst_buffer_pool_set_config`) is part of negotiation, not an
afterthought: every src pad runs an allocation query against its peer
sink pad on each `GST_EVENT_CAPS`. GPU integrations
(`GstGLBufferPool`, VAAPI surface pools, DRM dumb-buffer pools)
*require* the pool to be re-derived from the new caps before the
first frame under those caps is allocated. A framework that re-runs
caps negotiation but not allocation negotiation cannot host these
elements correctly — it can host *only* allocators that ignore caps
and over-allocate, which gives up the latency win the redesign is
chasing.

So "defer until needed" is not the long-term answer if the framework
intends to host real GPU pools. The right answer is to land the
mechanism the first time a concrete allocator depends on it, and to
make that mechanism cleanly extensible to clock changes, latency
adjustments, and any other mid-stream cross-element coordination.

### 9.2 Today's M12 cascade and why it is startup-only

`run_source_transform_sink` (lines around 501-510 of `runner.rs`)
runs the cascade once before spawning futures:

```text
let p = sink.propose_allocation(&caps);    // sink's needs
transform.configure_allocation(&p);        // transform absorbs them
let p' = transform.propose_allocation(&caps); // transform's needs
source.configure_allocation(&p');          // source allocates pool
```

This requires `&mut` access to all three elements. Once
`Join2::new(source_fut, ...)` spawns, each element is owned by its
own future. No mid-stream caps change re-queries the cascade.

### 9.3 Failure modes the deferred state masks today

Latent until an allocator actually consults negotiated caps:

- **Pool size drift.** A sink pool sized for the negotiated H.264
  caps stays H.264-sized after the decoder advertises NV12. With
  workaround #1 placeholder dims (16×16) and the post-decode real
  dims (e.g., 1920×1088), the size delta is two orders of magnitude.
  Today nothing crashes because `propose_allocation` returns `None`
  for every element in tree.
- **Domain mismatch.** A future GPU sink that proposes `DmaBuf`
  buffers cannot retroactively flip the upstream source from
  `System` to `DmaBuf` mid-stream — the cascade never re-queries.
- **Alignment / stride changes.** Codecs that emit different strides
  for different resolutions (vaapi paths, e.g., 1280 → 1920 width
  changes the stride from 1280 to 1920 with potential
  hardware-imposed padding) need the downstream pool to know the new
  stride before the first new-resolution frame.

### 9.4 Decision: phased implementation, with the trigger condition stated

The phasing here is structurally similar to workaround #3's A → B → C:
land the cheap local mechanism first, restructure when the cheap one
can't cover the next concrete case.

| Phase | Mechanism | Covers | Trigger |
|---|---|---|---|
| **α (cheap)** | Element-local re-allocation. `configure_pipeline(new_caps)` invokes the element's own `propose_allocation` and stores the new params for the element's next-frame allocation. No cross-element cascade. | "Sink resizes its own pool." "Decoder re-derives its scratch buffer size." Common allocator-internal cases. | When the first allocator-internal pool ships. Probably alongside vaapidec NV12 surface pool or kmssink dmabuf pool. |
| **β (correct)** | Runner restructure: a single coordinator task owns refs to source/transform/sink. The futures coordinate via channels; the coordinator runs the cascade. Mid-stream caps change at the boundary triggers a coordinator `Recascade { caps }` event. | Cross-element cascades: "source must allocate sink-shaped buffers." Also: mid-stream clock changes, latency budget tightening. | When the first element advertises that it *consumes* a downstream allocator's pool proposal across a mid-stream caps change. Concrete first candidates: a zero-copy DMABUF path source → decoder → sink chain; a VAAPI surface chain. |

### 9.4.1 Implementation status (M18 sessions B–D)

The scaffolding β builds on is landed; β's own cascade is not, and is
still trigger-gated per R1.

- **Coordinator control channel (Session B).** `runtime::coordinator`:
  `CoordinatorEvent`, `CoordinatorHandle` (cloneable producer over the
  in-house mpsc), `Coordinator` task, `coordinator(capacity)`.
  `run_source_transform_sink` spawns the coordinator as a fourth join
  arm; the sink arm reports an observe-only `CoordinatorEvent::CapsChanged`
  per applied mid-stream caps change. R3 (out-of-band channel, not in-band
  packets) honored. Surfaced as `RunStats.coordinator_events`.
- **Negotiation relocated (Session C).** The startup solver +
  per-link configure cascade moved verbatim into
  `coordinator::negotiate_source_transform_sink ->
  LinearNegotiation { source_link, sink_link }`, naming the per-link caps
  the β re-cascade will reconfigure. No behavior change.
- **α (Session D).** `coordinator::realloc_local` re-derives an
  element's own pool from the new caps (`propose_allocation` then
  `configure_allocation`) at each statically-typed mid-stream apply site
  (`run_simple_pipeline` sink; `run_source_transform_sink` transform and
  sink). Element-local only. Fan-out branch sinks excluded:
  `DynAsyncElement` does not expose the allocation hooks (lands with the
  FO-2 dyn-trait extension).

- **β single hop (Session E).** *Landed.* The `select2` combinator
  (`runtime::join`) gives the interruptibility the bullet below called for;
  the transform arm now selects its data link against a coordinator control
  channel and applies `configure_allocation` on an `ArmDirective::Recascade`.
  `CoordinatorEvent::CapsChanged` carries the sink's proposal;
  `Coordinator::run` forwards it one hop to the transform.
  `coordinator_with_recascade` wires the control channel; the transform's
  EOS-drain makes a tail-end directive deterministic at shutdown. Covered by
  `m18_beta_recascade.rs` plus coordinator and `select2` unit tests. Scope is
  the single sink->transform hop, the correct re-cascade for the 3-element
  chain (a link2 `CapsChanged` affects only the transform's output pool);
  the source leg and the N-hop downstream subgraph re-cascade are the
  multi-element runner (§13.4 item 4).

**What β single-hop chose, and what is still open:** the upstream cascade
needed the upstream arm *interruptible* at its `recv().await` so a
`Recascade` directive could reach `configure_allocation` mid-stream. That is
now the `select2` + per-arm control-channel route (R2 single-task
coordinator, not the ownership move, not `Arc<Mutex>` per element). Still
open: a *real* downstream consumer that re-sizes its pool on the mid-stream
proposal (the in-tree GPU decoders record it but the MFT/CUDA pools are
fixed at open, so the cascade is exercised by a fake transform, α-style);
the source leg / N-hop subgraph cascade (multi-element runner); and the
reverse per-input/per-branch structured `Renegotiate` (still β-gated for
fan-out/mux, §10).

### 9.5 Why β is the long-term shape, not just a bigger α

The coordinator restructure is the same machinery Phase C (§10) needs
for non-linear topologies, the same machinery a future M-something
needs for sinks-as-clock-providers when their reported clock changes
mid-stream, and the same machinery audio/video sync will need. It is
not an allocation-specific cost; once it lands, every cross-element
mid-stream coordination becomes a coordinator event instead of
ad-hoc.

GStreamer effectively has this — the pipeline owns elements and
orchestrates events. Our current spawn-and-forget futures topology
is *lighter* than GStreamer's at startup but *less capable* mid-
stream. The β restructure brings us to parity.

### 9.6 Latency target for β

GStreamer's `GST_EVENT_CAPS` triggers a synchronous allocation query
on each src→sink pad link. Total latency for a 3-element chain is
roughly: 2 allocation queries × (function-call overhead +
allocator-side computation). Real-world: hundreds of microseconds to
a millisecond.

Our solver-mediated cascade can do better. The coordinator runs
`solve_linear` over the affected subgraph (already O(n²) for tiny n)
plus one allocation-cascade pass. The arithmetic is microseconds.
The latency win over GStreamer comes from *not* round-tripping
through pad query objects: our cascade is a direct function call
chain serialized by the coordinator.

Avoid double allocation across the boundary by exploiting the
per-`Frame.caps` invariant (§5): elements key their pool by the
frame's caps, not the element's currently-configured caps. In-flight
old-caps frames complete under the old pool; new-caps frames pull
from the new pool. Pre-allocate the new pool concurrently with old-
pool drain.

### 9.7 Decisions for sign-off

- **R1.** Phased landing as above (α first, β when needed) vs. β
  upfront. *Recommendation: α + β plan in design, α implemented when
  the first allocator needs it, β implemented when the first
  cross-element case needs it.* The plan stays in the design doc so
  the next implementer doesn't re-derive it.
- **R2.** β's coordinator: single task or shared `Arc<Mutex<>>` on
  each element? *Recommendation: single-task coordinator.* Cleaner
  Send/Sync story, no interior-mutability assumption forced on every
  element.
- **R3.** Mid-stream allocation event: in-band (a new
  `PipelinePacket::AllocationProposal`) or coordinator-channel?
  *Recommendation: coordinator-channel.* Backward in-band packets
  collide with the existing reverse `Reconfigure` mechanism;
  coordinator-channel keeps the data plane data-only.

## 10. Phase C — non-linear topologies

### 10.1 Scope statement

Phase C applies the §4 subgraph re-solve principle to:

- **Fan-out:** `source → fan-out → N sinks` (`run_source_fanout`).
  Boundary upstream of the fan-out emits `CapsChanged`; each branch
  needs its own subgraph re-solve.
- **Muxer:** `N sources → muxer → 1 sink` (`run_muxer_sink`). Per-
  source mid-stream caps change re-solves that source's input link.
  Muxer's own output may also change as a function of input
  changes, requiring it to emit `CapsChanged` downstream.

GStreamer's `tee` (fan-out) and demuxers/muxers handle both cases.
The capability target is parity-plus.

### 10.2 Fan-out specifics

Today's `run_source_fanout` broadcasts every mid-stream
`CapsChanged` to each branch via `MultiSenderSink`. Each branch's
sink-fut handles it with the adjacent-only `configure_pipeline`
(Phase B's `re_solve_downstream_sink` helper is NOT applied in the
fan-out runner).

**Decisions for sign-off:**

- **FO-1: failure-mode policy.** If branch B1 accepts the new caps
  and B2 rejects:
  - *Strict (recommended):* any branch reject kills the fan-out
    with a structured failure. Matches GStreamer's default behavior
    (a `tee` with a rejecting downstream errors the pipeline).
  - *Graceful degradation (opt-in, future):* rejecting branches
    drop out; remaining branches continue. Requires explicit
    "drop branch and continue" mechanism the fan-out doesn't have
    today. Add later behind a `FanOutPolicy::AllowBranchDrop`
    enum, default `Strict`.
  - *Permissive (status quo):* rejecting branches keep getting
    old-config'd `process(CapsChanged)`. **Reject this as a
    silent-corruption path.** Phase B's strictness invariant
    applies per-branch.

- **FO-2: per-branch re-solve.** Apply Phase B's
  `re_solve_downstream_sink` per branch. Trivial assuming FO-1 picks
  strict. Each branch's solver call is independent; latency win:
  run them in parallel via `join_all` instead of sequentially.

- **FO-3: branch addition / removal mid-stream.** A future
  capability: add a new sink branch to a running fan-out, or remove
  a branch cleanly. GStreamer supports this via `request-pad` on
  `tee`. Out of scope for the initial Phase C; flag in the design
  as "next after FO-1/2 land."

### 10.3 Muxer specifics

Today's `run_muxer_sink` negotiates each `source ↔ muxer-input` pair
at startup (independent per input). The muxer's *output* caps
(`mux.output_caps()`) is queried once at startup and fed to the
downstream sink. No mid-stream re-query.

**Decisions for sign-off:**

- **MX-1: per-input re-solve.** Each input is independently
  negotiated, so per-input re-solve is structurally Phase B applied
  N times. Land alongside FO-2.

- **MX-2: input-derived output propagation.** When a source mid-
  stream changes its caps, the muxer's output caps may change as a
  function of the new input. GStreamer's typical muxer behavior is
  to re-emit `GST_EVENT_CAPS` downstream when the muxed output
  format changes. We need the same: on per-input re-solve, also
  query `mux.output_caps()` and, if it changed, emit
  `CapsChanged(new_output)` downstream. The downstream
  `run_muxer_sink` sink-fut hits Phase B's re-solve path on receipt.
  - *Eager emit (recommended for latency):* the muxer emits the new
    output `CapsChanged` immediately after its per-input re-solve
    succeeds, in parallel with the in-flight old-caps frames
    draining. The frame-caps invariant (§5) keeps downstream
    correct.
  - *Lazy emit (GStreamer-equivalent):* the muxer waits until it has
    actually merged a frame under the new caps before emitting
    `CapsChanged`. Simpler but adds one frame's worth of latency.

- **MX-3: muxer's negotiation surface.** `MultiInputElement` does
  not currently expose `caps_constraint_as_input(idx)` (the
  per-input native variant). To run Phase B's solver-mediated
  re-solve per input, the trait needs this method (same shape as
  `AsyncElement::caps_constraint_as_sink` but indexed). Should be
  added when MX-1 lands.

### 10.4 The coordinator dependency

Both FO-2 and MX-1/MX-2 are most cleanly expressed when the runner
has a coordinator task (§9.4 β). Without it, per-branch and per-
input re-solves are scattered across the spawned futures and the
solver calls have to be repeated.

This is not a blocker — Phase C can land before β if FO-2 and MX-1
implementations duplicate the solver call. But β makes them clean
one-liners.

### 10.5 Latency targets vs. GStreamer

GStreamer's `tee` mid-stream caps change: each downstream branch
runs its allocation query and configure-caps sequentially. For N=4
branches: 4× the cost.

Our Phase C with the β coordinator: parallel `solve_linear` per
branch (N microseconds each, true parallel via `join_all`). Total
latency: max single-branch cost, not sum. This is the
"hopefully lower latency than GStreamer" payoff for fan-out.

Muxer mid-stream input change: GStreamer holds the muxer's output
caps for one frame after an input change (lazy emit). We can do
eager emit (MX-2 recommendation) and shave one frame of latency.

### 10.6 Trigger conditions

Phase C originally deferred until a real production chain needed it (a
`tee` to display + recording, or an adaptive-bitrate mux). The M18 parity
push implemented it ahead of that trigger because the per-input /
per-branch re-solve fits entirely inside each runner's existing
single-owner task (the muxer task already owns `&mut mux`; each fan-out
branch has its own arm), so no β coordinator restructure was needed
(§10.4). **Status: landed.**

### 10.7 Decisions for sign-off (Phase C summary)

- **FO-1.** *Implemented.* Strict failure mode default: a branch whose
  `caps_constraint_as_sink()` rejects the mid-stream caps fails the
  fan-out loud (`CapsMismatch`). `FanOutPolicy::AllowBranchDrop` remains a
  future opt-in.
- **FO-2.** *Implemented.* Per-branch re-solve via
  `re_solve_downstream_dyn_sink` (the `DynAsyncElement` mirror of Phase
  B's helper). Branches run in independent arms, so the re-solves are
  concurrent. Plus per-branch element-local α re-allocation.
- **MX-1.** *Implemented.* Per-input re-solve in the muxer task via the
  shared `solve_mux_input` helper; strict loud failure (the β-clean
  reverse-`Renegotiate`-per-input variant is still β-gated).
- **MX-2.** *Implemented.* Eager emit of the muxer output `CapsChanged`
  when the re-derived output changes (via `solve_mux_output`).
- **MX-3.** *Implemented.* `MultiInputElement::caps_constraint_as_input(idx)`
  and `caps_constraint_for_output()` landed with the item 2 trait
  migration.

What remains β-gated (not Phase C): the structured per-input/per-branch
reverse-`Renegotiate` on a rejecting peer (the runner task can't reach an
upstream reconfigure slot without the coordinator), and the cross-element
allocation cascade (β proper, §9.4.1).
