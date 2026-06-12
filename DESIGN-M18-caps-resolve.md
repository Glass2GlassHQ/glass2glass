# M18 — Mid-stream caps re-solve over the downstream subgraph

> Design + status for the last open negotiation thread
> (`DESIGN-M16-caps-nego.md` §13.3/§13.4 item 4; `DESIGN-M16-workaround3-reconfigure.md`
> §4 D3 / Phase B). The β *allocation* re-cascade is N-hop; the *caps*
> re-solve was not.
>
> **Status: Caps-α landed** for both `run_linear_chain` (N-hop) and the
> single-transform `run_source_transform_sink` (D3 signed off). The runner now
> derives each interior element's forwarded output from its constraint, steered
> by a downstream-feasibility snapshot. Caps-β (forward coordinator re-solve
> walk) remains specified and driver-gated.

## 1. The defect

In `run_linear_chain_inner` a mid-stream `PipelinePacket::CapsChanged` is
fixated **greedily and downstream-blind**:

- Each interior arm (`runner.rs:909`) runs `t.configure_pipeline(new_caps)`
  on whatever its upstream forwarded, α-reallocates, sets `out_caps =
  new_caps` (wrong for a format-changer, already flagged in the comment),
  and lets the element's own `process(CapsChanged)` derive and emit the
  downstream caps.
- The sink arm (`runner.rs:948`) runs `re_solve_downstream_sink`, which
  solves `LegacySource(new_caps)` against the sink's *own* constraint only.
  One link.

No single solve spans `boundary -> t_{i+1} -> ... -> sink`. So an interior
format-converting element picks its output with no knowledge of what
downstream accepts. A chain where the converter *could* emit a
sink-acceptable format but its `process()` chooses another fails at the
sink, even though a valid whole-subgraph assignment exists. This is the
mid-stream analog of the gap the startup whole-chain `solve_linear` already
closes.

Concrete failing case (no hardware): `src -> converter -> sink`. Source
changes mid-stream to I420. The converter accepts I420 and can emit
{I420, NV12}. The sink accepts only NV12. Startup negotiated NV12 end to
end. Mid-stream today: the converter forwards I420 (greedy), the sink
rejects it, the run drives a reverse `Reconfigure` and stalls, although
converter=I420->NV12 is a valid solution.

## 2. Why the obvious fix doesn't fit

**Central coordinator solve is not reachable.** After `join_all` spawns the
arms, each element is owned by its arm. The coordinator is a separate task
(R2: single task, no ownership move, no `Arc<Mutex>` per element), so it
cannot call `caps_constraint_as_transform()` on the interior elements to run
`solve_linear` mid-stream. The β allocation re-cascade works around this with
a reactive *backward* walk (one hop per reply); it never needs all
constraints at one site.

**Each arm can reach its own constraint.** The arm owns
`&mut dyn DynAsyncElement`, and `DynAsyncElement::caps_constraint_as_transform(&self)`
is callable from inside the arm. So a *distributed forward* re-fixation
(each arm fixates its own output link) is feasible without an ownership move.
What an arm lacks is the *downstream* feasibility it must fixate against.

## 3. Recommended approach: distributed forward re-fixation, phased α/β

Mirror the allocation re-cascade's own discipline (cheap local first, full
coordinator walk when a real driver forces it).

### Caps-α (landed): startup feasibility snapshot

Downstream **capability** is static (an element's `caps_constraint_as_*`
describes what it *can* carry, fixed at construction); only the *data* caps
change mid-stream. So the per-link feasibility envelope can be precomputed at
startup and snapshotted into each arm.

1. **Backward feasibility sweep.** `solver::downstream_feasibility(constraints)
   -> Vec<Option<CapsSet>>`: a single reverse fold from the sink that returns,
   per link, the set the *downstream* tail can still fixate, **ignoring the
   upstream**. This is the key correctness point: the startup full-chain
   `links` are narrowed by the source too, so reusing them would falsely reject
   a mid-stream change to a format the source didn't originally produce. The
   sweep is source-independent. `None` = downstream imposes no expressible
   constraint (an `AcceptsAny` sink, or a non-invertible `DerivedOutput`/legacy
   element below this link). It does **not** alter `solve_linear`.

2. **Snapshot.** At startup, hand interior arm `i` its
   `downstream_feasible: Option<CapsSet>` = the sweep's set on its output link
   (`feasibility[i + 1]`).

3. **Mid-stream, arm `i` on `CapsChanged(in)`:**
   - intersect `in` with the element's input constraint; empty -> loud
     `EmptyLink`, reverse `Reconfigure` upstream, structured failure to bus.
   - derive output candidates from `in` via the constraint
     (`Identity` / `Mapping` / `DerivedOutput`).
   - intersect candidates with `downstream_feasible[i]`.
   - fixate. `Unfixable` is pass-through (§7: ranged field, not a failure);
     `EmptyLink` is loud + reverse `Reconfigure` into this boundary (not past
     the source).
   - `configure_pipeline(in)`, α realloc, `out_caps = fixated_output`,
     forward `CapsChanged(fixated_output)`.

   The sink arm's `re_solve_downstream_sink` stays as the tail of the same
   logic (its `downstream_feasible` is just its accept set).

**Covers:** static-capability downstream chains, i.e. the converter-before-sink
case in §1 and any `Identity`/`Mapping`/`Accepts` segment. This is the
concrete capability win and is CI-testable against a fake converter chain.

**Does not cover:** a downstream element whose feasible output is a function
of the changed input (`DerivedOutput`, e.g. a decoder *downstream* of the
boundary). Its envelope was snapshotted against the startup input, so it is
stale under new caps. Caps-α detects this conservatively (the stale envelope
either still intersects -> correct, or empties -> loud fail) rather than
silently corrupting. A second decoder mid-chain is not a current topology.

### Caps-β (complete, specify now, build on a driver): forward re-solve walk

When a real chain needs a downstream `DerivedOutput` element to re-derive
mid-stream, add the forward analog of the β allocation walk: on `CapsChanged`
at the boundary, a forward request/reply walk down the arms gathers each
element's *current* constraint contribution (each arm replies with its
narrowed set given the upstream-narrowed input), converging to per-link caps,
then applies. This is GStreamer's recursive downstream caps query. It reuses
the coordinator + `select2` machinery (forward direction, request/reply
instead of backward one-shot) and adds round-trips before the first new-caps
frame. Gated per R1 on a concrete driver, exactly as β-allocation was.

## 4. Load-bearing decision (D3, signed off)

**Who derives the forwarded output caps mid-stream?** *Decided: the runner
(recommended option below), implemented for `run_linear_chain`.*

- **Status quo:** the element's `process(CapsChanged)` chooses and emits.
  Greedy, downstream-blind, no new capability. The runner only passes caps
  through.
- **Recommended (D3 from workaround3 §4):** the *runner* derives the forwarded
  output from the element's constraint + the feasibility snapshot, and hands
  the element the final caps to apply. `process(CapsChanged)` becomes "apply
  what you are given," not "choose." Pass-throughs already forward verbatim;
  format-changers move their derivation into the declared constraint
  (`Mapping`/`DerivedOutput`, D2: single source of truth), which the solver
  already consumes at startup.

This changes the data-plane contract for `CapsChanged` (the runner, not
`process`, owns the forwarded caps). It is the enabling change and is hard to
reverse once elements rely on it, so it is the one piece to confirm before
runner code lands.

## 5. Tests (landed, CI, no hardware)

`g2g-plugins/tests/m18_caps_resolve.rs` drives a `DerivedOutput` converter
(format-only, carries the input's geometry) through a mid-stream RGBA -> I420
change into an NV12-only sink:

- **Caps-α positive** (`midstream_change_steers_converter_to_sink_acceptable_output`):
  the sink records `CapsChanged(NV12)`, not the source's I420, and every frame
  flows. Fails on the prior greedy runner (the sink would reject I420).
- **Loud failure** (`..._no_acceptable_output_fails_loud_to_bus`): a converter
  with no NV12 path for I420 input drives a reverse `Reconfigure` and posts a
  structured `EmptyLink` to the bus; no NV12 reaches the sink.

Solver unit tests: `downstream_feasibility_is_source_independent` (the
snapshot ignores the source) and `resolve_forward_output_steers_defers_and_rejects`
(the three outcomes). The existing β N-hop / multi-element runner tests are
unaffected (pass-through chains hit the `Defer` path).

## 6. Scope

Landed in: `run_linear_chain` (N-hop linear) and the single-transform
`run_source_transform_sink`. `resolve_forward_output` / `ForwardResolve` are
no_std (the single-transform runner is a no_std runtime path); the multi-hop
`downstream_feasibility` sweep stays `std`-gated with `run_linear_chain`. The
single-transform path needs no sweep: its downstream subgraph is one sink link,
so the feasibility snapshot is just the sink's `Accepts` set, read inline. Out:
fan-out / muxer mid-stream caps re-solve (Phase C caps, separate), the Caps-β
build, the codec-vs-raw `Caps` split (§12, independent).
