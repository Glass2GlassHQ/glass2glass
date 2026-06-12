# M16 — Caps negotiation v2 (CSP framing)

> **Status:** design, not implementation. The runner still does the
> single-cascade Phase 1+2 negotiation described in `DESIGN.md` §4.2.
> This doc locks the shape of the replacement before code lands.

## 1. Why a redesign

`DESIGN.md` §4.2 cascades one `Caps` value through `source.intercept →
transform.intercept → sink.intercept → fixate`, then calls
`configure_pipeline(same_fixated)` on all three. Three workarounds have
accumulated to keep this working for the RTSP → ffmpeg → sink pipeline:

1. **Fixate-friendly placeholder dims** in `RtspSrc::intercept_caps` —
   returns `Dim::Range { 16, 8192 }` because `Caps::fixate()` rejects
   `Any`. Real dims arrive mid-stream via `CapsChanged`.
2. **Pass-through `intercept_caps` on sinks** — `KmsSink` and
   `WaylandSink` accept the decoder's *input* (H.264) caps at startup
   and only configure real resources when the post-decode NV12
   `CapsChanged` cascades back. A sink that wanted to advertise its
   real input domain (NV12 only) couldn't, because the trait only lets
   it narrow what upstream offered.
3. **In-band `CapsChanged` swallow on decoders** —
   `FfmpegH264Dec::process(CapsChanged(_))` is a no-op because its job
   is to *replace* the upstream H.264 caps with downstream NV12 caps.

Each is fine in isolation. Together they collapse the model: any
*second* format-changing transform, or any downstream allocator that
trusts the negotiated caps to size pools, breaks the cascade.

The right framing is **distributed constraint satisfaction**: each
element declares its constraint over (input, output) caps; the solver
finds an assignment over every link satisfying all constraints, ranked
by preferences. GStreamer's negotiation *is* this, named differently.
Naming it explicitly buys structure for the non-linear topologies (M9
fan-out, M10 muxer, future ML branches) and for non-format constraints
later (latency budget, hardware affinity, ML device placement).

## 2. CSP framing

**Variables.** One per link in the graph: the caps assigned to that
link. The forward `PipelinePacket` stream on every link carries data
under exactly that caps assignment.

**Domain.** All representable caps (a `CapsSet` — see §3).

**Constraints.** Per element. Source: only its output link constrained.
Sink: only its input link. Transform: both, with a relation between
them.

**Solution.** An assignment to every link such that every element's
constraint is satisfied. When multiple solutions exist, the
*preferences* declared by each element rank them; the solver picks the
highest-ranked.

**Failure.** No assignment exists. The solver returns *which* element's
constraint conflicted with which neighbor — a structured replacement
for today's opaque `FixationFailed`.

**Reconfiguration.** A mid-stream constraint change (sink's
`accepted_input_caps` tightens, source picks a new sub-stream, a
transform's parameters change) re-runs the solver over the affected
subgraph. Frames in flight under the old assignment continue to drain;
new frames flow under the new assignment.

## 3. CapsSet — caps with alternatives and preferences

Today's `Caps` carries one description: one `VideoFormat`, one `Dim`,
one `Rate`. It cannot express "I prefer NV12 over I420 over YV12" or
"either an audio stream of any sample rate, or a video stream up to 4K"
— both of which are routine in real pipelines.

```rust
/// A set of acceptable caps descriptions, ordered by preference.
/// The first element is highest preference; later elements are
/// fallbacks the element will accept if no peer agrees on the first.
#[derive(Clone, Debug)]
pub struct CapsSet {
    alternatives: Vec<Caps>,
}

impl CapsSet {
    /// Build from a single concrete description (equivalent to today's
    /// Caps for static call sites).
    pub fn one(caps: Caps) -> Self { ... }

    /// Intersection: the caps both sets agree on, preserving the
    /// preference order of `self`. Empty result = no assignment exists
    /// for a link between elements with these two sets.
    pub fn intersect(&self, other: &Self) -> Self { ... }

    /// Fixate the highest-preference alternative to a single concrete
    /// Caps. None if every alternative still has ranged / Any fields.
    pub fn fixate(&self) -> Option<Caps> { ... }
}
```

`Caps` itself stays as the *fixed* description used at runtime
(`DataFrame.caps`, `configure_*` callbacks). `CapsSet` is the
negotiation-time vocabulary.

This is the structural upgrade GStreamer's `GstCaps` provides via "list
of structures with priority." Without it, **preference ordering is
unrepresentable** and every element is locked into one preferred format.

## 4. Element constraint surface

```rust
trait FormatElement {
    /// Declare the constraint this element imposes on its surrounding
    /// links. See [`CapsConstraint`]. Read by the solver during
    /// negotiation.
    fn caps_constraint(&self) -> CapsConstraint;

    /// Optional preferences for tie-breaking. Defaults to None,
    /// meaning the solver uses the constraint's own preference order.
    fn caps_preferences(&self) -> Option<CapsPreferences> { None }

    /// Called by the runner once the solver has assigned caps to
    /// every link. Boundary elements receive distinct input / output
    /// values; non-boundary elements receive equal ones.
    fn configure_link(
        &mut self,
        input: Option<&Caps>,
        output: Option<&Caps>,
    ) -> Result<(), G2gError>;
}

enum CapsConstraint {
    /// Sink-shape: only the input side is constrained. Output is
    /// unused (sink has no downstream).
    Accepts(CapsSet),

    /// Source-shape: only the output side is constrained. Input is
    /// unused.
    Produces(CapsSet),

    /// Pass-through transform: input == output, both drawn from this
    /// set. Format converters with a single supported format land
    /// here, as do identity / probe / metering elements.
    Identity(CapsSet),

    /// Format-changing transform with an explicit (input, output)
    /// relation. The set enumerates all legal pairs; the solver picks
    /// one. Most decoders and encoders use this.
    Mapping(Vec<(CapsSet, CapsSet)>),

    /// Programmatic mapping: the output set is a function of the
    /// (already-narrowed) input caps. Used when output depends on the
    /// input in a way that can't be precomputed (e.g. a decoder
    /// reading SPS to fix output dims). The function is consulted by
    /// the solver during forward propagation.
    DerivedOutput(Box<dyn Fn(&Caps) -> CapsSet + Send + Sync>),
}
```

This replaces today's `intercept_caps`, `configure_pipeline`, and the
forward-half `propose_output_caps` hook that just landed. Migration is
mechanical for each element:

| Today | Tomorrow |
|---|---|
| Source: `intercept_caps()` returns its produced caps | `caps_constraint() = Produces(set)` |
| Sink: `intercept_caps(upstream)` narrows | `caps_constraint() = Accepts(set)` |
| Identity transform | `caps_constraint() = Identity(set)` |
| Decoder: `intercept_caps` accepts H.264, swallows in process | `caps_constraint() = DerivedOutput(\|h264\| nv12_from_dims(h264))` |
| `configure_pipeline(caps)` | `configure_link(Some(in), Some(out))` |

## 5. Solver

**Linear pipeline (source → transform → sink):**

1. Collect each element's constraint.
2. Walk forward from the source: compute each link's candidate set as
   the intersection of upstream-produced ∩ downstream-accepted.
   `DerivedOutput` is evaluated using upstream's fixated caps once that
   link has narrowed to a fixed value.
3. Walk backward to propagate any narrowing that downstream
   constraints imply for upstream's eventual fixation.
4. Repeat until convergence (the candidate set on each link stops
   changing — fixed point).
5. Fixate each link to its highest-preference concrete caps.
6. Call `configure_link` on every element with its assigned (input,
   output).

Steps 2–4 are vanilla arc consistency on a chain — at most O(n²)
intersections for n elements. Linear pipelines converge in one
forward+backward sweep.

**Non-linear (M9 fan-out, M10 muxer, future ML branches):**

Same algorithm; the graph structure means the propagation order needs
a topological sort. Cycles (rare; usually a configuration error)
become a known failure case.

**Preferences.** When a link's candidate set fixates with multiple
viable concrete caps, pick the one that ranks highest under the
preference function aggregated from the link's two endpoints. Default
aggregation: sum of preference indices (lower wins). Custom
aggregations can be added later.

**Failure path.** If any link's candidate set becomes empty, the
solver returns

```rust
enum NegotiationFailure {
    EmptyLink { upstream: ElementId, downstream: ElementId, missed: CapsSet },
    Cyclic { ... },
}
```

so the caller can report which pair couldn't agree on what.

## 6. Reconfiguration as constraint update

Mid-stream, an element's constraint may change:

- A source picks a new sub-stream from an IP camera → its `Produces`
  set updates.
- A downstream allocator gets reshaped (e.g. a new GPU pool) → the
  sink's `Accepts` set updates.
- A decoder learns its output dims from SPS → its `DerivedOutput`
  function reduces a previously-ranged output to a fixed value.

The element emits a `ReconfigureRequest` upstream the same way today's
N-1-lag-aware mechanism works (see commit `31339b6` for the pre-send
check). The runner re-runs the solver over the affected subgraph and
re-issues `configure_link` to every element whose link assignment
changed. Frames in flight under the old assignment continue to drain
(they're tagged with their caps so a downstream element knows which
configuration to use for each frame, regardless of what the runner has
since reconfigured).

The "in-band" part of this — flowing the reconfigure request as an
ordered packet rather than a side-channel flag — is the remaining lag
fix. The CSP framing doesn't change that mechanism; it just gives a
clean trigger semantic ("constraint X tightened") rather than the
specific "downstream rejected proposal Y."

The *forward* direction of this (the decoder swallowing its input
`CapsChanged`, workaround #3) has its own implementable spec:
`DESIGN-M16-workaround3-reconfigure.md`. Key finding: the naive eager
re-derive corrupts ordering across a B-frame reorder boundary, so the
reconfigure must stay positioned at the decode boundary; the spec
phases the fix (validate+record, then subgraph re-solve, then
non-linear).

## 7. ACCEPT_CAPS and Capsfilter

Fall out of the constraint surface for free:

- **ACCEPT_CAPS query:** `constraint.accepts(&caps)` — a pure check
  against the constraint's set. No runtime back-and-forth; the
  element's constraint already describes everything it would accept.
- **Capsfilter element:** a pass-through whose constraint is
  `Identity(specific_set)`. Inserted anywhere in a pipeline to force a
  narrowing. ~30 lines once the constraint surface exists.

## 8. Migration plan

This is M16, sized as 4–5 focused sessions:

1. **`CapsSet` type + intersect / fixate algebra.** Mechanical. Every
   element keeps using `Caps` at runtime; `CapsSet` is the
   negotiation-time wrapper. Tests for the algebra.

2. **`FormatElement` trait + `CapsConstraint` enum.** New trait,
   coexists with `AsyncElement` during migration. `AsyncElement` gets
   a default `FormatElement` impl that derives constraints from
   today's `intercept_caps` so unmigrated elements still work.

3. **Solver in `g2g-core::runtime::solver`.** Linear-pipeline arc
   consistency, returning per-link assignments. Tests for empty link,
   cyclic graph, preference tie-break.

4. **Runner refactor.** `run_source_transform_sink` etc. call the
   solver instead of the cascaded negotiation. Existing tests must
   pass.

5. **Migrate sources / sinks / decoders to `FormatElement` directly.**
   Delete the three workarounds (placeholder dims, pass-through
   intercept, deferred configure). Existing tests adapt.

6. **CapsFilter element + ACCEPT_CAPS surface.** Trivial after (5).

The order ensures each step is verifiable on its own: (1) ships an
algebra, (2) ships a trait, (3) ships a solver, (4) wires solver into
runner without changing behavior, (5) removes workarounds with the
solver actively in use, (6) is opportunistic add.

## 9. Open questions

- **`CapsSet` shape vs `GstCaps` shape exactly.** GStreamer's
  `GstCaps` is a `GArray<GstStructure>`. We could mirror that
  precisely or keep our enum-based `Caps` and just make `CapsSet`
  hold ordered alternatives. Choosing now to keep `Caps` as the enum
  — simpler, type-safe, no string-keyed structure lookup.
- **Memory-domain constraints.** A sink may accept only `DmaBuf`
  while upstream produces only `System`. Today that's a domain
  mismatch reported by `propose_allocation`. In CSP framing it's
  another constraint dimension; should it live in `CapsConstraint` or
  in a parallel `AllocationConstraint`? Defer to M17.
- **Latency / clock-budget constraints.** Same question. Out of scope
  for M16; the framing scales naturally if added later.
- **Cost of the solver at startup.** Linear pipeline = O(n²)
  intersections, n small. Real-world overhead measured in
  microseconds. Mid-stream re-solve is also cheap because only the
  affected subgraph is touched. Worth confirming with a benchmark
  once (1)–(4) land.

## 10. What this doc does NOT decide

- Whether the eventual `FormatElement` trait fully replaces
  `AsyncElement` or coexists. Bet on coexists at first, decide once
  the migration is well-understood.
- Concrete preference algebra (sum-of-indices is a placeholder).
- Whether the solver runs in `no_std` (probably yes — it's pure
  computation over heap allocations).

Code starts at step 1 of §8 in a fresh session.

## 11. Implementation status (2026-06-10)

§8 migration plan is mostly executed. Map of intent vs reality:

| §8 step | Status | Notes |
|---|---|---|
| 1. `CapsSet` algebra | ✅ done | `g2g-core::caps`. `one`, `from_alternatives`, `intersect` (preserves self's order, dedupes), `union`, `fixate` (picks first fixable alt). |
| 2. `FormatElement` + `CapsConstraint` | ✅ done | `g2g-core::format_element`. Plus three migration-bridge variants: `LegacySource`, `LegacyTransform`, `LegacySink`. Plus two wildcards added during migration: `AcceptsAny` (sink), `IdentityAny` (transform). `FormatElement::configure_link` defined but unused by the runner — runner still calls `AsyncElement::configure_pipeline` with per-link caps. |
| 3. Linear solver | ✅ done | `g2g-core::runtime::solver::solve_linear`. Three dispatch paths: all-native arc consistency (forward + backward sweep to fixed point), all-legacy intercept-only forward cascade (bit-compatible with pre-M16), mixed forward cascade. `NegotiationFailure` enum reports `EmptyLink` / `EndpointShapeMismatch` / `Unfixable` / `Degenerate` / `Cyclic` (reserved). |
| 4. Runner refactor | ✅ done | `run_simple_pipeline`, `run_source_transform_sink`, `run_source_fanout` all call `solve_linear`. `run_muxer_sink` solves each source↔mux-input pair. `run_fanin_sink` left direct (per-source self-fixation, no chain). Per-link configure: `run_source_transform_sink` extracts `src_caps = links[0]` / `sink_caps = links[1]` and passes each element its side. `ReFixate` retry preserved with `LegacySource(counter)` fallback. |
| 5. Migrate elements | ✅ done | Migrated: `FakeSink`/`syncsink` (`AcceptsAny`), `VideoTestSrc` (`Produces`), `H264Parse` (`Identity(H264)`), `IdentityTransform` (`IdentityAny`), `FfmpegH264Dec` (`DerivedOutput`), `MfDecode` (`DerivedOutput`, step 5l), `VaapiH264Dec` (`DerivedOutput`, step 5m). All three H.264 decoders are now native. `RtspSrc` migrated via async `intercept_caps` (M18 item 5, closes workaround #1). `mux` (`InterleaveMux`) migrated to `caps_constraint_as_input`/`caps_constraint_for_output` (M18 step 1). The three display sinks `WaylandSink`/`KmsSink`/`D3D11Sink` migrated to `Accepts(NV12 / any geometry)` (M16 step 5 final pass), which also makes their ACCEPT_CAPS query truthful (the old `LegacySink` passthrough always claimed acceptance). No in-tree element rides the legacy bridge now; the `Legacy*` variants stay as the default for unmigrated elements. Workaround #2 (deferred-configure on sinks) is now fully retired: the non-NV12 no-op accept branch was removed from `WaylandSink`/`KmsSink` after a reachability audit (every chain to these sinks runs through a native `DerivedOutput` decoder, so the solver lands NV12 at startup); they reject non-NV12 input loud and tolerate mid-stream dim changes (5j) by rebuilding. Workaround #1 (RtspSrc placeholder dims) now has an opt-in escape, `RtspSrc::with_expected_dims(w, h)`, which advertises fixed dims at negotiation so a sink sizes once at startup; the placeholder Range remains the default and full auto-detection still needs an async `SourceLoop::intercept_caps`. Workaround #3 (decoder swallows input CapsChanged) still in place, now with an implementable phased spec in `DESIGN-M16-workaround3-reconfigure.md` (awaiting sign-off on decisions D1-D4 before code). The dead-`FfmpegH264Dec`-hook cleanup removed its unused `is_format_boundary` / `propose_output_caps` overrides (the trait methods stay for legacy transforms). |
| 6. `CapsFilter` + `ACCEPT_CAPS` | ✅ done | `CapsConstraint::accepts` / `CapsSet::accepts` (§7 query, pure check). `g2g-plugins::capsfilter::CapsFilter` is an `Identity(set)` pass-through that narrows on the legacy path via `intercept_caps`, validates configured + mid-stream caps against the filter, and forwards data unchanged. |

**Deviations from the §8 plan:**
- `FormatElement::configure_link(input, output)` is unused. The runner instead calls `AsyncElement::configure_pipeline` with the appropriate per-link `Caps` (input side for source/transform, sink-input side for sink). This avoids forcing every migrated element to implement a new trait method — they inherit `configure_pipeline` from `AsyncElement`. If a future boundary transform needs *explicit* per-side setup (e.g. allocating input and output pools differently), `configure_link` becomes the natural escape hatch.
- The §5 design assumed every native chain hits arc consistency. In practice, partially-migrated chains hit the mixed-cascade path which is forward-only (no backward propagation). Arc consistency only activates when every element is native. This is fine for migration — the cost is that arc-consistency benefits (e.g. an `Identity` middle filtering against the downstream sink) don't apply until the whole chain migrates.
- `Cyclic` failure variant is on `NegotiationFailure` but never produced by `solve_linear`. Reserved for the eventual non-linear graph solver.
- Migration revealed two wildcards the §3-4 design didn't anticipate: `AcceptsAny` (debug/probe sinks like `FakeSink`) and `IdentityAny` (pass-through transforms like `IdentityTransform`). These accept "anything upstream produces" without a concrete `CapsSet` — they're a natural shape that `Accepts(CapsSet)` / `Identity(CapsSet)` can't express because `CapsSet` requires concrete alternatives, not a wildcard.
- §8 step 5b implies `AsyncElement` gets a default `FormatElement` impl deriving constraints from `intercept_caps`. In practice this would be a blanket impl, which then prevents elements from overriding `FormatElement` (Rust orphan rules). What landed: `AsyncElement` gains `caps_constraint_as_sink` / `caps_constraint_as_transform` default methods returning the legacy bridge; migrated elements override these directly. `FormatElement` the trait is documented as the future direction but isn't on the runner's hot path.

**Visual verification still owed.** CI can't exercise `rtsp → ffmpegdec → wayland/kms` (sandbox blocks RTSP, no Wayland session in the test env). The mixed-chain per-link path is structurally verified by `format_changing_transform_receives_input_side_caps` in `pipeline_smoke.rs`, but the production visual chain — including the new "16×16 placeholder surface → real-dim rebuild" sequence — needs a manual e2e run.

## 12. Adjacent design debt: codec vs raw format split

Acknowledged 2026-06-10 during step 5g. M16 builds on top of an
existing model smell: `VideoFormat` in `g2g-core/src/caps.rs` mixes
compressed codecs (H264, H265, Av1, Vp9) and raw pixel layouts (Nv12,
I420, Rgba8, Bgra8) in one enum, all shoehorned into
`Caps::Video { format, width, height, framerate }`.

GStreamer keeps these as separate media types:
- `video/x-h264, stream-format=byte-stream, alignment=au, profile=...`
  (no `format` field)
- `video/x-raw, format=NV12, width=1920, height=1080, framerate=30/1`

The proper shape is two variants:

```rust
pub enum Caps {
    CompressedVideo { codec: VideoCodec, extras: ... },
    RawVideo { format: RawFormat, width: Dim, height: Dim, framerate: Rate },
    Audio { ... },
    Tensor { ... },
}
```

Why M16 doesn't tackle it: M17-sized refactor. Every element's caps
logic, every test, every `CapsSet` and `AcceptsAny` site touches it.
Several M16 workarounds (#1 placeholder dims in `RtspSrc`, #2
deferred sink configure on `WaylandSink`/`KmsSink`) would dissolve
naturally with the split because the structural impossibilities
become type-level — a raw-only sink simply can't match compressed
caps. Combined refactor will be smaller than M16 + M17 separately if
ever attempted as one piece.

See `architecture_codec_vs_raw_format.md` in auto-memory for the full
analysis and `architecture_caps_nego_debt.md` for the workarounds
this would interact with.

## 13. GStreamer parity assessment (M16 is the foundation, not completion)

Assessment date 2026-06-10, after all of M16 steps 1-5m, workaround
#2 retirement, workaround #3 Phase A + B, and latency observability
landed; production visual e2e (`rtsp → ffmpegdec → wayland/kms`)
verified.

The framework target is *at least as capable as GStreamer, ideally
lower latency*. M16 covers the linear pipeline case ≥ GStreamer with
a structural latency edge (direct function calls instead of GStreamer's
pad-query round-trip), but several real GStreamer capabilities are
explicitly deferred. Honest rating: ~70-75% capability parity.

### 13.1 What M16 delivers ≥ GStreamer

- CSP solver with arc consistency on linear chains, returning per-link
  caps. GStreamer's negotiation walks pad-by-pad with no global
  validation — we get structured failure (`NegotiationFailure`) for
  free.
- ACCEPT_CAPS / `CapsFilter` covered identically.
- Forward `CapsChanged` ordering invariant tested
  (m16_workaround3_phase_a.rs) — the "decoder reorder boundary"
  scenario is a regression guard GStreamer doesn't have at the
  framework level.
- Decoder validate-record on input `CapsChanged` (Phase A) — H.264 →
  VP9 mid-stream is loud `CapsMismatch` rather than silent corruption.
- Sinks tolerate mid-stream dim changes by rebuilding the surface.
- Per-frame `Caps` tagging invariant: a `DataFrame`'s caps are
  authoritative for that frame, allowing in-flight old-caps frames to
  drain cleanly past a mid-stream reconfigure. This is the foundation
  for the no-double-allocation latency win.

### 13.2 What M16 deferred (since closed by the M18 push)

Documented in `DESIGN-M16-workaround3-reconfigure.md` §9 and §10. The M18
parity push implemented the Phase C and α items below; only β proper
remains.

- **Allocation re-cascade (§9).** M12 `propose_allocation` ran only at
  startup. Phased plan α (element-local) → β (coordinator restructure).
  *α landed* (mid-stream element-local re-allocation in the linear and
  fan-out runners). *β single-hop landed* (Session E): the no_std
  `select2` primitive plus a coordinator control channel let the transform
  arm apply the sink's re-derived proposal one hop upstream on a mid-stream
  `CapsChanged`. *Still open:* the source leg / N-hop downstream subgraph
  cascade (multi-element runner, §13.4 item 4) and a real downstream
  consumer that re-sizes its pool on the proposal (the in-tree GPU decoders
  record but don't yet act on it, so β is exercised by a fake transform).
- **Phase C fan-out (§10).** *Landed.* Per-branch re-solve (FO-2) with
  strict failure default (FO-1), branches re-solved concurrently in their
  own arms, plus per-branch α.
- **Phase C muxer (§10).** *Landed.* Per-input re-solve (MX-1) plus eager
  output `CapsChanged` emission (MX-2) inside the muxer task.

### 13.3 What M16 doesn't specify or cover

- **Multi-element subgraph re-solve.** *Runner generalized (item 4).*
  `run_linear_chain` drives an arbitrary-length `source -> t0 -> ... ->
  sink` (interior elements `&mut dyn DynAsyncElement`), so chains with 4+
  elements (boundary → capsfilter → converter → sink) are now expressible,
  with whole-chain startup negotiation + the M12 allocation fold. *Owed:*
  the mid-stream re-solve still reconfigures element-locally per interior
  element plus the Phase-B sink re-solve; the full cross-element
  downstream-subgraph re-solve and β re-cascade over N hops extend the
  single-hop coordinator path of `run_source_transform_sink`.
- **Async source caps discovery.** Workaround #1 has the opt-in
  `RtspSrc::with_expected_dims(w, h)` for callers who know the
  camera dims. The full fix needs `SourceLoop::intercept_caps` to be
  async so the source can do an SDP DESCRIBE before negotiation.
  GStreamer's `rtspsrc` does this.
- **Pad templates as declarative metadata.** *Done (M18, item 6).* The
  `pad_template` module adds `PadTemplate` + a `PadTemplates` trait whose
  `pad_templates()` is an associated function, so tools query an element
  *type*'s pads without constructing it. `pad_link` / `types_can_link` run
  the solver against two types' static templates for pre-instantiation
  compatibility checks. The runtime `caps_constraint_as_*` remains the
  instance-level (possibly narrower) view.
- **Dynamic pads / request pads.** GStreamer's `tee::request-pad` and
  `mux::request-pad` allow adding branches / inputs at runtime. Our
  fan-out and muxer are static.
- **Mid-stream element hot-swap.** M8's `ElementSlot` scaffolding
  exists but mid-stream swap of a real element isn't supported.
- **Bus events for negotiation failures.** *Partially landed (item 7).*
  `BusMessage::NegotiationFailed(NegotiationFailure)` plus
  `run_source_transform_sink_with_bus` route the structured failure to the
  bus for the linear runner's startup negotiation (the run still returns the
  opaque `G2gError::CapsMismatch`). Owed: the mid-stream re-solve path and
  the other runners (simple / fan-out / fan-in / mux), which discard their
  `NegotiationFailure` identically.
- **Preference algebra.** `CapsPreferences` is a placeholder. The
  solver uses constraint-internal order for tie-breaks. GStreamer's
  more elaborate "best fit" caps selection across competing
  structures isn't implemented.

### 13.4 The M18 parity push, ordered by value-per-work

1. **Allocation re-cascade β (coordinator restructure).** Single
   biggest lever. Unlocks GPU pools / DMABUF / VAAPI chains, AND
   cleanly enables items 3 and 4 (Phase C, multi-element re-solve)
   and a future mid-stream clock-change mechanism. The restructure
   pays off across many capabilities; it's a foundation, not a feature.
   *Status (M18 B–E):* scaffolding landed (coordinator control channel,
   negotiation relocated to the coordinator, α element-local
   re-allocation); **β single-hop landed (Session E)** via the no_std
   `select2` combinator + a coordinator->transform control channel, so the
   sink's re-derived proposal re-cascades one hop upstream on a mid-stream
   `CapsChanged` (`m18_beta_recascade.rs`). The remaining gap is the
   N-hop / source-leg downstream subgraph cascade, which folds into item 4
   (multi-element runner), plus a real downstream pool consumer that acts on
   the mid-stream proposal (the in-tree GPU decoders record but don't yet
   re-size on it; the MFT/CUDA output pools are fixed at open).
2. **`mux` migration + Phase C muxer.** *Done (M18).* Trait migration
   (MX-3) plus per-input re-solve (MX-1) and eager output re-emit (MX-2)
   inside the muxer task; no β needed (workaround3 §10.4, §10.7).
3. **Fan-out Phase C with FO-1 strict default.** *Done (M18).* Per-branch
   re-solve (FO-2) concurrent across branch arms, strict failure (FO-1),
   plus per-branch α. `tee` to display + recording is now supported.
4. **Multi-element runner.** *Landed (startup + data plane).*
   `run_linear_chain(source, Vec<&mut dyn DynAsyncElement>, sink, ..)` drives
   any-length linear chains: whole-chain `solve_linear`, per-element
   configure, the M12 allocation fold, and `N + 2` arms over `N + 1` links
   (`m18_multi_element.rs`). `DynAsyncElement` gained
   `caps_constraint_as_transform` for erased interior elements. *Owed,
   couples to item 1:* the mid-stream re-solve is element-local per interior
   element plus the Phase-B sink re-solve; covering the full downstream
   subgraph with the β re-cascade over N hops extends the single-hop
   coordinator path.
5. **Async `SourceLoop::intercept_caps`.** *Done (M18).* Trait gains
   `type CapsFuture<'a>` + `fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a>`;
   default `caps_constraint` awaits it. `DynSourceLoop` returns
   `BoxFuture`. `RtspSrc` now performs DESCRIBE + SETUP in the probe
   path, parses `VideoParameters`, caches the result; reconnect policy
   wraps the probe, so transient connect failures retry with the same
   backoff `run` uses for mid-session drops. `with_expected_dims` is
   the offline fast-path (no I/O). Closes workaround #1.
6. **Pad templates declarative metadata.** *Done (M18).* `PadTemplates`
   trait (type-level `pad_templates()`), `pad_link` / `types_can_link`
   pre-instantiation solver queries; implemented for `VideoTestSrc`,
   `FakeSink`, `H264Parse`.
7. **Bus integration for negotiation failures.** *Partially landed.*
   `BusMessage::NegotiationFailed(NegotiationFailure)` +
   `run_source_transform_sink_with_bus` carry the structured failure for the
   linear startup path (`m18_bus_negotiation.rs`). Owed: the mid-stream
   re-solve path and the simple / fan-out / fan-in / mux runners (each
   discards its `NegotiationFailure` the same way), which fold in once their
   runners take the bus.
8. **Preference algebra.** Concrete trigger required (a competing-
   constraint scenario that forces it).
9. **Dynamic pads / hot-swap.** Lowest priority. No production driver.

Items 1-4 are the bulk of the gap. Doing them gets to ~95% parity
with the latency edge intact.

### 13.5 How to think about M16 going forward

M16 is "the negotiation foundation." Its job was:
- Replace the single-cascade caps model with CSP.
- Land the solver, the legacy bridge, the mixed cascade, the per-link
  runner, native variants for the common element shapes, and the
  ordering invariants the redesign rests on.
- Retire the existing workarounds where the new architecture removes
  the need.
- Make the foundation visually verified on production hardware so
  future work doesn't rebuild it.

It is *not* "negotiation complete." The deferred items above are
real GStreamer capabilities we don't yet match. An M18-ish parity
push is the right framing for closing the gap; piecemeal additions
risk re-doing item 1's restructure work later.

When work depends on a deferred item (a real GPU pool, multi-element
chain, audio/video mux, branched topology with mid-stream caps
changes), surface the gap before building on top — don't lock the
foundation into a shape that the M18 push will need to undo.
