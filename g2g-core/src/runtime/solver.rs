//! M16 step 3 (DESIGN.md §4.13.2): linear-pipeline caps solver.
//!
//! Takes the ordered constraint list for a source → transform* → sink
//! chain and returns one fixated `Caps` per link, or a structured
//! failure describing which pair couldn't agree.
//!
//! The algorithm is arc consistency on a chain: forward pass narrows
//! every link by each constraint's contribution, backward pass
//! propagates new narrowing back upstream, repeat to fixed point.
//! `DerivedOutput` is consulted once its input link has fixated, so
//! decoders that read dims from SPS slot in naturally. Fixed-point
//! convergence is guaranteed because every iteration either shrinks at
//! least one link's candidate set or terminates.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::caps::{
    couple_passthrough_derived, discover_passthrough, Caps, CapsSet, PassthroughFields,
};
#[cfg(feature = "std")]
use crate::caps::{project_passthrough, project_passthrough_derived};
use crate::format_element::CapsConstraint;
use crate::graph::{NodeId, NodeKind, ValidatedGraph};
use crate::log::{self, LogLevel, Target, CAPS_CATEGORY};

/// Per-link assignment produced by the solver: one fixated `Caps` per
/// link between adjacent elements. For an `N`-element pipeline this is
/// length `N - 1`. The runner calls `configure_link` on element `i`
/// with `input = links[i - 1]` and `output = links[i]` (sources receive
/// `None` on input; sinks receive `None` on output).
pub type LinkSolution = Vec<Caps>;

/// Structured solver failure (DESIGN.md §4.13.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NegotiationFailure {
    /// Adjacent elements have no overlap on the link between them, or a
    /// constraint update emptied that link.
    EmptyLink {
        upstream: usize,
        downstream: usize,
    },
    /// The constraint list has fewer than two elements; nothing to
    /// negotiate.
    Degenerate,
    /// First element does not produce (or last does not accept), so the
    /// chain has no source or no sink.
    EndpointShapeMismatch {
        index: usize,
    },
    /// A link's candidate set survived narrowing but cannot be reduced
    /// to a single `Caps` (every alternative still has `Any` fields).
    Unfixable {
        upstream: usize,
        downstream: usize,
    },
    /// Reserved for the non-linear solver. The linear solver never
    /// returns this variant.
    Cyclic,
    /// Arc consistency left a non-empty domain on every edge, but no single
    /// assignment of one fixated `Caps` per edge satisfies every node at once:
    /// an over-constrained diamond (a tee whose branches re-converge at a fan-in
    /// with no jointly-valid choice). Only the DAG solver's backtracking
    /// fixation returns this.
    NoConsistentFixation,
    /// Transitional: the chain mixes `Legacy*` variants with the
    /// native `Accepts` / `Produces` / `Identity` / `Mapping` /
    /// `DerivedOutput` variants. Mixed handling is added once step 5
    /// starts migrating individual elements off the legacy bridge.
    MixedLegacyAndNative,
}

/// Solve a linear chain of caps constraints. See module docs.
///
/// `constraints[0]` must be `Produces`; `constraints[N-1]` must be
/// `Accepts`. Interior elements may be `Identity`, `Mapping`, or
/// `DerivedOutput`.
pub fn solve_linear<'a>(
    constraints: &[&CapsConstraint<'a>],
) -> Result<LinkSolution, NegotiationFailure> {
    if constraints.len() < 2 {
        return Err(NegotiationFailure::Degenerate);
    }

    // Dispatch: chains made entirely of legacy bridge variants take a
    // simple forward cascade that mirrors today's runner cascade.
    // Chains made entirely of native variants take the arc-consistency
    // path below. Mixed chains aren't handled yet (step 5 starts the
    // migration).
    let any_legacy = constraints.iter().any(|c| is_legacy(c));
    let any_native = constraints.iter().any(|c| !is_legacy(c));
    if any_legacy && any_native {
        return solve_mixed_cascade(constraints);
    }
    if any_legacy {
        return solve_legacy_cascade(constraints);
    }

    let n = constraints.len();
    let n_links = n - 1;

    // Endpoint shape check.
    match constraints[0] {
        CapsConstraint::Produces(_) => {}
        _ => return Err(NegotiationFailure::EndpointShapeMismatch { index: 0 }),
    }
    match constraints[n - 1] {
        CapsConstraint::Accepts(_) | CapsConstraint::AcceptsAny => {}
        _ => return Err(NegotiationFailure::EndpointShapeMismatch { index: n - 1 }),
    }

    // Seed each link with the broadest set we can derive: source's
    // `Produces` set on link 0, sink's `Accepts` set on link n-2,
    // everything else starts empty and gets filled by the first sweep.
    // To allow Identity / Mapping to refine before we know endpoints,
    // we use a sentinel "unconstrained" representation: an empty set
    // means "not yet constrained" *only* on the first iteration; after
    // the first forward pass empty truly means failure.
    let mut links: Vec<Option<CapsSet>> = alloc::vec![None; n_links];

    // Pre-seed endpoints.
    if let CapsConstraint::Produces(s) = constraints[0] {
        links[0] = Some(s.clone());
    }
    if let CapsConstraint::Accepts(s) = constraints[n - 1] {
        let li = n_links - 1;
        links[li] = match links[li].take() {
            Some(cur) => Some(cur.intersect(s)),
            None => Some(s.clone()),
        };
    }

    // Arc-consistency loop. Bounded by n_links * max_alternatives,
    // but practically converges in 1-2 sweeps for chains.
    let max_iters = 8 * n_links + 4;
    for _ in 0..max_iters {
        let snapshot = links.clone();

        // Forward sweep.
        for (i, c) in constraints.iter().enumerate() {
            apply_constraint(i, c, &mut links, n_links)?;
        }
        // Backward sweep — same logic in reverse promotes downstream
        // narrowing back upstream (relevant for Identity and Mapping).
        for (i, c) in constraints.iter().enumerate().rev() {
            apply_constraint(i, c, &mut links, n_links)?;
        }

        if links == snapshot {
            break;
        }
    }

    // Validate and fixate.
    let mut out = Vec::with_capacity(n_links);
    for (li, slot) in links.iter().enumerate() {
        let set = slot.as_ref().ok_or(NegotiationFailure::EmptyLink {
            upstream: li,
            downstream: li + 1,
        })?;
        if set.is_empty() {
            return Err(NegotiationFailure::EmptyLink { upstream: li, downstream: li + 1 });
        }
        let fixed = set.fixate().ok_or(NegotiationFailure::Unfixable {
            upstream: li,
            downstream: li + 1,
        })?;
        out.push(fixed);
    }
    Ok(out)
}

fn is_legacy(c: &CapsConstraint<'_>) -> bool {
    matches!(
        c,
        CapsConstraint::LegacySource(_)
            | CapsConstraint::LegacyTransform { .. }
            | CapsConstraint::LegacySink(_)
    )
}

/// Forward cascade for chains of legacy bridge variants. Mirrors
/// today's runner: source's `intercept_caps()` seeds a `Caps`; each
/// transform's `intercept_caps(upstream)` narrows; non-boundary
/// transforms forward the narrowed input as their output, boundary
/// transforms call `propose_output_caps`. The sink's `intercept_caps`
/// produces the final fixated `Caps`. Phase 2 fixate runs once at the
/// end. Mid-stream `ReFixate` retry stays in the runner.
fn solve_legacy_cascade(
    constraints: &[&CapsConstraint<'_>],
) -> Result<LinkSolution, NegotiationFailure> {
    let n = constraints.len();
    let n_links = n - 1;

    // Endpoints must be source/sink shape.
    let mut current = match constraints[0] {
        CapsConstraint::LegacySource(caps) => caps.clone(),
        _ => return Err(NegotiationFailure::EndpointShapeMismatch { index: 0 }),
    };
    match constraints[n - 1] {
        CapsConstraint::LegacySink(_) => {}
        _ => return Err(NegotiationFailure::EndpointShapeMismatch { index: n - 1 }),
    }

    let mut links: Vec<Caps> = Vec::with_capacity(n_links);
    // Interior elements: bit-compatible with the pre-M16 inline
    // cascade. Each transform contributes ONLY `intercept(upstream)`;
    // `propose_output_caps` is intentionally NOT called here. In the
    // legacy single-fixated-caps model, the same `Caps` flows through
    // every element and the decoder's output-side caps don't appear
    // until the mid-stream `CapsChanged` lands. The mixed/native
    // cascade paths use `propose_output_caps` to derive per-link caps;
    // the legacy cascade leaves that out so chains containing
    // workaround #2 sinks (waylandsink/kmssink with their
    // pass-through-then-defer pattern) keep working unchanged.
    for (i, c) in constraints.iter().enumerate().skip(1).take(n.saturating_sub(2)) {
        match c {
            CapsConstraint::LegacyTransform { intercept, propose_output: _ } => {
                current = intercept(&current).map_err(|_| NegotiationFailure::EmptyLink {
                    upstream: i - 1,
                    downstream: i,
                })?;
                links.push(current.clone());
            }
            CapsConstraint::LegacySource(_) | CapsConstraint::LegacySink(_) => {
                return Err(NegotiationFailure::EndpointShapeMismatch { index: i });
            }
            _ => return Err(NegotiationFailure::MixedLegacyAndNative),
        }
    }
    // Sink: cascade through its intercept, then fixate the result.
    if let CapsConstraint::LegacySink(intercept) = constraints[n - 1] {
        current = intercept(&current).map_err(|_| NegotiationFailure::EmptyLink {
            upstream: n - 2,
            downstream: n - 1,
        })?;
        links.push(current);
    }

    // Phase 2 fixate the final value, then propagate it to every link
    // slot. The pre-M16 cascade fed one `fixated` Caps to every
    // `configure_pipeline` call; honoring that exactly means upstream
    // slots carry the final fixated caps too. Format-changing
    // boundaries that need per-link semantics must migrate to the
    // native solver path (one endpoint at a time) — that's where the
    // mixed cascade kicks in and the runner gets real per-link caps.
    let fixed_last = links
        .last()
        .ok_or(NegotiationFailure::Degenerate)?
        .fixate()
        .map_err(|_| NegotiationFailure::Unfixable {
            upstream: n - 2,
            downstream: n - 1,
        })?;
    for slot in links.iter_mut() {
        *slot = fixed_last.clone();
    }

    Ok(links)
}

/// Unified forward cascade for chains that mix `Legacy*` and native
/// (`Produces` / `Accepts` / `Identity` / `Mapping` / `DerivedOutput`)
/// variants. Handles the migration window where elements move from
/// the legacy bridge to native constraints one at a time.
///
/// Single forward pass: each element computes its output `CapsSet`
/// from the upstream link's `CapsSet`. Legacy variants and
/// `DerivedOutput` require the upstream to fixate to a single
/// concrete `Caps` (which the typical migration chain — single-source
/// upstream — satisfies). No backward pass: arc-consistency benefits
/// (Identity / Mapping filtering against downstream sinks) are not
/// applied in the mixed path. Once a chain is fully native, dispatch
/// routes it back to the arc-consistency solver, which restores
/// backward propagation.
fn solve_mixed_cascade(
    constraints: &[&CapsConstraint<'_>],
) -> Result<LinkSolution, NegotiationFailure> {
    let n = constraints.len();
    let n_links = n - 1;

    let starts_with_source = matches!(
        constraints[0],
        CapsConstraint::Produces(_) | CapsConstraint::LegacySource(_)
    );
    let ends_with_sink = matches!(
        constraints[n - 1],
        CapsConstraint::Accepts(_) | CapsConstraint::LegacySink(_) | CapsConstraint::AcceptsAny
    );
    if !starts_with_source {
        return Err(NegotiationFailure::EndpointShapeMismatch { index: 0 });
    }
    if !ends_with_sink {
        return Err(NegotiationFailure::EndpointShapeMismatch { index: n - 1 });
    }

    // link_sets[i] is the CapsSet on the link between element i and i+1.
    let mut link_sets: Vec<CapsSet> = Vec::with_capacity(n_links);

    // Seed link 0 from the source.
    let seed = match constraints[0] {
        CapsConstraint::Produces(s) => s.clone(),
        CapsConstraint::LegacySource(c) => CapsSet::one(c.clone()),
        _ => unreachable!("checked above"),
    };
    link_sets.push(seed);

    // Forward-propagate through every middle element.
    for i in 1..(n - 1) {
        let upstream = link_sets[i - 1].clone();
        let downstream = forward_propagate(constraints[i], &upstream, i)?;
        link_sets.push(downstream);
    }

    // Narrow the final link against the sink endpoint.
    let final_idx = n_links - 1;
    let upstream = link_sets[final_idx].clone();
    let narrowed = match constraints[n - 1] {
        CapsConstraint::Accepts(s) => upstream.intersect(s),
        CapsConstraint::AcceptsAny => upstream,
        CapsConstraint::LegacySink(intercept) => {
            let fixed = upstream.fixate().ok_or(NegotiationFailure::Unfixable {
                upstream: n - 2,
                downstream: n - 1,
            })?;
            let c = intercept(&fixed).map_err(|_| NegotiationFailure::EmptyLink {
                upstream: n - 2,
                downstream: n - 1,
            })?;
            CapsSet::one(c)
        }
        _ => unreachable!("checked above"),
    };
    if narrowed.is_empty() {
        return Err(NegotiationFailure::EmptyLink { upstream: n - 2, downstream: n - 1 });
    }
    link_sets[final_idx] = narrowed;

    // Fixate every link.
    let mut out = Vec::with_capacity(n_links);
    for (li, s) in link_sets.iter().enumerate() {
        let fixed = s.fixate().ok_or(NegotiationFailure::Unfixable {
            upstream: li,
            downstream: li + 1,
        })?;
        out.push(fixed);
    }
    Ok(out)
}

fn forward_propagate(
    c: &CapsConstraint<'_>,
    upstream: &CapsSet,
    i: usize,
) -> Result<CapsSet, NegotiationFailure> {
    match c {
        CapsConstraint::Identity(s) => {
            let r = upstream.intersect(s);
            if r.is_empty() {
                return Err(NegotiationFailure::EmptyLink { upstream: i - 1, downstream: i });
            }
            Ok(r)
        }
        CapsConstraint::Mapping(pairs) => {
            let mut out = CapsSet::from_alternatives(Vec::new());
            for (in_set, out_set) in pairs {
                let in_match = upstream.intersect(in_set);
                if !in_match.is_empty() {
                    out = out.union(out_set);
                }
            }
            if out.is_empty() {
                // No input alternative matched any mapping row: the input link
                // (between elements i-1 and i) is the conflict, same as Identity.
                return Err(NegotiationFailure::EmptyLink { upstream: i - 1, downstream: i });
            }
            Ok(out)
        }
        CapsConstraint::DerivedOutput(f) | CapsConstraint::DerivedCoupled { derive: f, .. } => {
            let fixed = upstream.fixate().ok_or(NegotiationFailure::Unfixable {
                upstream: i - 1,
                downstream: i,
            })?;
            // A `DerivedCoupled`'s passthrough mask and its derive closure are two
            // sources of truth for the same fact; verify they agree on the
            // concrete input so a mask claiming a field the closure retargets is
            // caught here (debug builds), not as a silent mis-narrowing of the
            // input. `DerivedOutput` carries no mask, so nothing to check.
            if let CapsConstraint::DerivedCoupled { passthrough, .. } = c {
                debug_assert!(
                    crate::caps::verify_passthrough_sound(f.as_ref(), *passthrough, &fixed),
                    "DerivedCoupled passthrough mask claims a field its derive closure does not pass through"
                );
            }
            let r = f(&fixed);
            if r.is_empty() {
                return Err(NegotiationFailure::EmptyLink { upstream: i, downstream: i + 1 });
            }
            Ok(r)
        }
        CapsConstraint::LegacyTransform { intercept, propose_output } => {
            let fixed = upstream.fixate().ok_or(NegotiationFailure::Unfixable {
                upstream: i - 1,
                downstream: i,
            })?;
            let input = intercept(&fixed).map_err(|_| NegotiationFailure::EmptyLink {
                upstream: i - 1,
                downstream: i,
            })?;
            Ok(CapsSet::one(propose_output(&input)))
        }
        CapsConstraint::IdentityAny => {
            // Wildcard transform: pass upstream through unchanged.
            Ok(upstream.clone())
        }
        CapsConstraint::Produces(_)
        | CapsConstraint::Accepts(_)
        | CapsConstraint::AcceptsAny
        | CapsConstraint::LegacySource(_)
        | CapsConstraint::LegacySink(_) => {
            Err(NegotiationFailure::EndpointShapeMismatch { index: i })
        }
    }
}

/// Caps-α mid-stream re-fixation outcome for one interior element
/// (DESIGN.md §4.13.4). The runner derives the element's
/// forwarded output from its declared constraint, steered by the downstream
/// feasibility snapshot, instead of letting the element fixate greedily.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ForwardResolve {
    /// Runner-derived, downstream-aware output caps to forward.
    Fixed(Caps),
    /// The output can't be derived/fixated from the constraint, or there is
    /// no concrete downstream set to steer against. Fall back to the status
    /// quo: forward the incoming caps and let the element's own `process`
    /// derive its output (covers `DerivedOutput` / legacy / ranged outputs).
    Defer,
    /// The element's possible outputs positively can't satisfy what the
    /// downstream subgraph accepts. Loud: the caller drives a reverse
    /// reconfigure and posts the structured failure.
    Infeasible(NegotiationFailure),
}

/// Backward feasibility sweep: per output link, the set it can carry such
/// that the elements *downstream* of that link can still fixate to the sink,
/// ignoring the (mid-stream-changing) upstream. `None` on a link means
/// "downstream imposes no expressible constraint here" (an `AcceptsAny`
/// sink, or a non-invertible `DerivedOutput` / legacy element below it).
/// Computed once at startup and snapshotted per interior arm so the
/// mid-stream re-solve can steer an element's output without reaching the
/// downstream elements at runtime (DESIGN.md §4.13.4).
///
/// Test-only since `run_linear_chain` became a thin builder over `run_graph`
/// (which uses the edge-indexed [`graph_downstream_feasibility`]); the test
/// still pins the per-link reverse-sweep behavior.
#[cfg(all(test, feature = "std"))]
pub(crate) fn downstream_feasibility(constraints: &[&CapsConstraint<'_>]) -> Vec<Option<CapsSet>> {
    let n = constraints.len();
    if n < 2 {
        return Vec::new();
    }
    let n_links = n - 1;
    let mut feas: Vec<Option<CapsSet>> = alloc::vec![None; n_links];
    // Seed the sink link from the sink's accept set; a wildcard or legacy
    // sink leaves it unconstrained.
    feas[n_links - 1] = match constraints[n - 1] {
        CapsConstraint::Accepts(s) => Some(s.clone()),
        _ => None,
    };
    // Propagate upstream through each interior transform: the element at
    // position k+1 sits between link k and link k+1. No startup input sample is
    // threaded here (this test-only sweep pins the Identity / Mapping / coupled
    // hops); the real graph path supplies it from the solved edge sets.
    for k in (0..n_links - 1).rev() {
        feas[k] = backward_feasible(constraints[k + 1], feas[k + 1].as_ref(), None);
    }
    feas
}

/// One reverse hop of [`downstream_feasibility`]: given the feasible set on
/// an element's output link, the set its input link can carry. `in_sample` is
/// a representative input alternative the element fixated to at startup,
/// available for the closure-probing constraints (`DerivedOutput`) that need a
/// concrete input to discover their invertible fields; `None` for the others.
#[cfg(feature = "std")]
fn backward_feasible(
    c: &CapsConstraint<'_>,
    down: Option<&CapsSet>,
    in_sample: Option<&Caps>,
) -> Option<CapsSet> {
    match c {
        CapsConstraint::Identity(s) => Some(match down {
            Some(d) => s.intersect(d),
            None => s.clone(),
        }),
        CapsConstraint::IdentityAny => down.cloned(),
        CapsConstraint::Mapping(pairs) => {
            let mut acc = CapsSet::from_alternatives(Vec::new());
            for (in_set, out_set) in pairs {
                let out_ok = match down {
                    Some(d) => !out_set.intersect(d).is_empty(),
                    None => true,
                };
                if out_ok {
                    acc = acc.union(in_set);
                }
            }
            Some(acc)
        }
        // A `DerivedCoupled` transform inverts on its passthrough fields: the
        // input feasibility is the downstream set with retargeted fields widened
        // to anything the transform accepts (Dim/Rate -> Any, sample_rate ->
        // ANY). `project_passthrough` returns `None` for a retargeted scalar with
        // no wildcard (e.g. videoconvert's format), in which case the input
        // feasibility isn't expressible as a single `Caps` and we impose none.
        CapsConstraint::DerivedCoupled { passthrough, .. } => {
            let d = down?;
            let mut alts = Vec::with_capacity(d.alternatives().len());
            for o in d.alternatives() {
                alts.push(project_passthrough(o, *passthrough)?);
            }
            Some(CapsSet::from_alternatives(alts))
        }
        // A plain `DerivedOutput` (decoder / rescaler) declares no passthrough
        // mask, but M257's `discover_passthrough` recovers its invertible fields
        // by probing the closure on a concrete input. Mid-stream the snapshot has
        // only the output set; the startup-fixated `in_sample` supplies that probe
        // (and the input variant / scalar identity, which the output alone can't
        // give across a decoder's variant change). With a non-empty mask the input
        // feasibility is the downstream set's passthrough fields projected back onto
        // the sample's variant, with every non-passthrough (re-derived) field
        // *widened to `Any`* (`project_passthrough_derived`): the transform
        // re-derives that field from whatever input it gets mid-stream, so the input
        // edge stays unconstrained on it. Freezing it to the startup value (M258 v1)
        // made the snapshot reject a legitimately re-derived mid-stream geometry
        // (the Caps-β forward gap). An empty mask or no sample imposes none.
        CapsConstraint::DerivedOutput(f) => {
            let (d, sample) = (down?, in_sample?);
            let mask = discover_passthrough(f, sample);
            if mask == PassthroughFields::NONE {
                return None;
            }
            let mut alts = Vec::with_capacity(d.alternatives().len());
            for o in d.alternatives() {
                if let Some(c) = project_passthrough_derived(sample, o, mask) {
                    if !alts.contains(&c) {
                        alts.push(c);
                    }
                }
            }
            (!alts.is_empty()).then(|| CapsSet::from_alternatives(alts))
        }
        // Non-invertible (legacy) or non-transform shape: impose no constraint
        // on the input link.
        _ => None,
    }
}

/// Caps-α: derive the forwarded output for an interior element on a
/// mid-stream caps change (DESIGN.md §4.13.4). `input` is
/// the new fixated caps the element receives; `downstream_feasible` is its
/// output link's snapshot from [`downstream_feasibility`]. Steers when a
/// concrete downstream set exists; with no downstream snapshot it still
/// forwards the element's output when that output is *unambiguous* (the
/// constraint maps the input to exactly one fixated caps), and only defers to
/// the element's own `process` when the output is genuinely undetermined. A
/// format-changing transform (e.g. `videoconvert` to RGBA8) therefore announces
/// its real output to a strict downstream rather than leaking its input format.
pub(crate) fn resolve_forward_output(
    constraint: &CapsConstraint<'_>,
    input: &Caps,
    downstream_feasible: Option<&CapsSet>,
) -> ForwardResolve {
    // Index 1 is a placeholder: the link position is meaningful only inside
    // a full-chain solve. Mid-stream the failure is link-local to this arm.
    let candidates = match forward_propagate(constraint, &CapsSet::one(input.clone()), 1) {
        Ok(c) => c,
        Err(_) => return ForwardResolve::Defer,
    };
    let Some(d) = downstream_feasible else {
        // No downstream snapshot to steer by: forward the output only when the
        // constraint pins it to a single producible caps (a property-driven
        // converter, an identity passthrough). An ambiguous set (a caps-driven
        // converter with several producible formats) still defers to `process`,
        // since there is nothing to choose between them.
        return match candidates.alternatives() {
            [_one] => match candidates.fixate() {
                Some(c) => ForwardResolve::Fixed(c),
                None => ForwardResolve::Defer,
            },
            _ => ForwardResolve::Defer,
        };
    };
    let narrowed = candidates.intersect(d);
    if narrowed.is_empty() {
        return ForwardResolve::Infeasible(NegotiationFailure::EmptyLink {
            upstream: 0,
            downstream: 1,
        });
    }
    match narrowed.fixate() {
        Some(c) => ForwardResolve::Fixed(c),
        None => ForwardResolve::Defer,
    }
}

fn apply_constraint(
    i: usize,
    c: &CapsConstraint<'_>,
    links: &mut [Option<CapsSet>],
    n_links: usize,
) -> Result<(), NegotiationFailure> {
    let in_idx = if i == 0 { None } else { Some(i - 1) };
    let out_idx = if i == n_links { None } else { Some(i) };

    match c {
        CapsConstraint::Produces(s) => {
            if let Some(idx) = out_idx {
                narrow(links, idx, s, i, i + 1)?;
            }
        }
        CapsConstraint::Accepts(s) => {
            if let Some(idx) = in_idx {
                narrow(links, idx, s, i - 1, i)?;
            }
        }
        CapsConstraint::Identity(s) => {
            // Input link and output link both narrowed by S, and must
            // equal each other (pass-through).
            if let Some(idx) = in_idx {
                narrow(links, idx, s, i - 1, i)?;
            }
            if let Some(idx) = out_idx {
                narrow(links, idx, s, i, i + 1)?;
            }
            // Couple the two sides: each side ∩= the other.
            if let (Some(ii), Some(oi)) = (in_idx, out_idx) {
                let (a, b) = (links[ii].clone(), links[oi].clone());
                if let (Some(a), Some(b)) = (a, b) {
                    let coupled = a.intersect(&b);
                    if coupled.is_empty() {
                        return Err(NegotiationFailure::EmptyLink {
                            upstream: i - 1,
                            downstream: i + 1,
                        });
                    }
                    links[ii] = Some(coupled.clone());
                    links[oi] = Some(coupled);
                }
            }
        }
        CapsConstraint::Mapping(pairs) => {
            let (Some(ii), Some(oi)) = (in_idx, out_idx) else {
                return Err(NegotiationFailure::EndpointShapeMismatch { index: i });
            };
            // Filter pairs to those still consistent on both sides.
            let mut new_in = CapsSet::from_alternatives(Vec::new());
            let mut new_out = CapsSet::from_alternatives(Vec::new());
            for (in_set, out_set) in pairs {
                let in_match = match &links[ii] {
                    Some(cur) => cur.intersect(in_set),
                    None => in_set.clone(),
                };
                let out_match = match &links[oi] {
                    Some(cur) => cur.intersect(out_set),
                    None => out_set.clone(),
                };
                if !in_match.is_empty() && !out_match.is_empty() {
                    new_in = new_in.union(&in_match);
                    new_out = new_out.union(&out_match);
                }
            }
            if new_in.is_empty() || new_out.is_empty() {
                return Err(NegotiationFailure::EmptyLink {
                    upstream: i - 1,
                    downstream: i + 1,
                });
            }
            links[ii] = Some(new_in);
            links[oi] = Some(new_out);
        }
        CapsConstraint::DerivedOutput(f) => {
            let (Some(ii), Some(oi)) = (in_idx, out_idx) else {
                return Err(NegotiationFailure::EndpointShapeMismatch { index: i });
            };
            // Forward (M188): narrow the output by the union of `f` over every
            // input alternative. For a single fixated input this is just
            // `f(input)` (the M185/M186 single-transform behaviour); for a still
            // ambiguous input (a stacked auto transform whose upstream hasn't
            // fixated) it still produces an output to narrow, so the second
            // transform's output link no longer stalls at `None`.
            if let Some(input_set) = &links[ii] {
                let derived = forward_derived_union(f.as_ref(), input_set);
                if derived.is_empty() {
                    return Err(NegotiationFailure::EmptyLink {
                        upstream: i,
                        downstream: i + 1,
                    });
                }
                narrow(links, oi, &derived, i, i + 1)?;
            }
            // Backward (M188 + invertible-field coupling): probe the closure for
            // passthrough fields and narrow the input field-by-field on them
            // (a downstream geometry / framerate pin couples back through a
            // decoder); otherwise drop input alternatives that can't reach the
            // output, so stacked auto transforms resolve.
            if let (Some(in_set), Some(out_set)) = (links[ii].clone(), links[oi].clone()) {
                match derived_backward(f.as_ref(), &in_set, &out_set) {
                    Ok(Some(narrowed)) => links[ii] = Some(narrowed),
                    Ok(None) => {}
                    Err(()) => {
                        return Err(NegotiationFailure::EmptyLink {
                            upstream: i - 1,
                            downstream: i,
                        })
                    }
                }
            }
        }
        CapsConstraint::DerivedCoupled { derive, passthrough } => {
            let (Some(ii), Some(oi)) = (in_idx, out_idx) else {
                return Err(NegotiationFailure::EndpointShapeMismatch { index: i });
            };
            // Forward: identical to `DerivedOutput` (the closure is the source of
            // truth for forward derivation).
            if let Some(input_set) = &links[ii] {
                let derived = forward_derived_union(derive.as_ref(), input_set);
                if derived.is_empty() {
                    return Err(NegotiationFailure::EmptyLink { upstream: i, downstream: i + 1 });
                }
                narrow(links, oi, &derived, i, i + 1)?;
            }
            // Backward: field-level coupling, narrowing passthrough fields *within*
            // an alternative (the unblock over `DerivedOutput`'s alternative-drop).
            if let (Some(in_set), Some(out_set)) = (links[ii].clone(), links[oi].clone()) {
                match backward_field_narrow(derive.as_ref(), *passthrough, &in_set, &out_set) {
                    Ok(Some(narrowed)) => links[ii] = Some(narrowed),
                    Ok(None) => {}
                    Err(()) => {
                        return Err(NegotiationFailure::EmptyLink {
                            upstream: i - 1,
                            downstream: i,
                        })
                    }
                }
            }
        }
        CapsConstraint::AcceptsAny => {
            // Wildcard sink: no narrowing. The link feeding this sink
            // takes whatever shape upstream produces. The endpoint
            // check enforces that this only appears at the chain's
            // tail, so no further work is needed here.
        }
        CapsConstraint::IdentityAny => {
            // Wildcard transform: don't narrow either side by a set
            // (there is no set), just couple input and output to be
            // equal. Either side's current value determines both.
            if let (Some(ii), Some(oi)) = (in_idx, out_idx) {
                let (a, b) = (links[ii].clone(), links[oi].clone());
                match (a, b) {
                    (Some(a), Some(b)) => {
                        let coupled = a.intersect(&b);
                        if coupled.is_empty() {
                            return Err(NegotiationFailure::EmptyLink {
                                upstream: i - 1,
                                downstream: i + 1,
                            });
                        }
                        links[ii] = Some(coupled.clone());
                        links[oi] = Some(coupled);
                    }
                    (Some(a), None) => links[oi] = Some(a),
                    (None, Some(b)) => links[ii] = Some(b),
                    (None, None) => {}
                }
            }
        }
        CapsConstraint::LegacySource(_)
        | CapsConstraint::LegacyTransform { .. }
        | CapsConstraint::LegacySink(_) => {
            // Dispatch in `solve_linear` routes all-legacy chains to
            // `solve_legacy_cascade` and rejects mixed chains, so the
            // arc-consistency path never sees a legacy variant.
            return Err(NegotiationFailure::MixedLegacyAndNative);
        }
    }
    Ok(())
}

fn narrow(
    links: &mut [Option<CapsSet>],
    idx: usize,
    contrib: &CapsSet,
    upstream: usize,
    downstream: usize,
) -> Result<(), NegotiationFailure> {
    let next = match &links[idx] {
        Some(cur) => cur.intersect(contrib),
        None => contrib.clone(),
    };
    if next.is_empty() {
        return Err(NegotiationFailure::EmptyLink { upstream, downstream });
    }
    links[idx] = Some(next);
    Ok(())
}

/// Per-node constraint for [`solve_graph`]. Source/transform/sink carry a
/// single [`CapsConstraint`]; a fan-in muxer carries a per-input-pad constraint
/// plus its output constraint. A tee is structural, so its slot is ignored
/// (pass any `Element`).
#[derive(Debug)]
pub enum NodeConstraint<'a> {
    /// Source (`Produces`), transform (`Identity` / `Mapping` /
    /// `DerivedOutput` / `IdentityAny`), or sink (`Accepts` / `AcceptsAny`).
    Element(CapsConstraint<'a>),
    /// Fan-in muxer: `inputs[i]` is input pad `i`'s constraint (`Accepts` to
    /// narrow that pad, `AcceptsAny` for a wildcard pad that forwards
    /// per-frame caps), and `output` is the single output pad's `Produces`
    /// constraint. This is the per-pad shape real muxer elements expose
    /// (`MultiInputElement::caps_constraint_as_input` / `_for_output`), so the
    /// DAG runner builds it straight from the element.
    /// `follows` is `Some(pad)` for an identity-passthrough mux (an overlay /
    /// watermark) whose output caps are that input pad's negotiated caps: the
    /// solver derives the output edge from that input edge and `output` is unused
    /// (a placeholder). `None` is the usual independent output declared by
    /// `output` (a container interleave, a fixed compositor).
    Muxer {
        inputs: Vec<CapsConstraint<'a>>,
        output: CapsConstraint<'a>,
        follows: Option<usize>,
    },
    /// Fan-out demux (M380): `input` is the byte-stream input constraint (the
    /// container the demux consumes), and `ports[i]` is output port `i`'s
    /// `Produces` constraint (its distinct elementary stream). Unlike a broadcast
    /// tee (which couples in == every out), the ports are *decoupled*, so each
    /// branch negotiates against its own caps and a downstream decoder configures
    /// against its codec at startup. Built from a demux element that declares
    /// per-port caps (`MultiOutputElement::port_output_caps`); a broadcast fan-out
    /// (declaring none) stays a plain tee.
    Demux { input: CapsConstraint<'a>, ports: Vec<CapsConstraint<'a>> },
}

/// Solve caps for an arbitrary DAG (DESIGN_TODO "DAG runner" D2). Generalizes
/// [`solve_linear`]'s arc-consistency sweep to topological order over a
/// [`ValidatedGraph`]: each edge is a link variable, narrowed by the
/// constraints of the nodes at both ends, swept forward in topo order and
/// backward in reverse to a fixed point, then fixated. Returns one fixated
/// `Caps` per edge, indexed by edge id.
///
/// `constraints` is indexed by node id (length must equal the node count). A
/// tee fans its input caps out to every output unchanged (its slot is
/// ignored); a muxer narrows each input edge by its pad's accept set and its
/// single output edge by the produce set.
pub fn solve_graph<E>(
    graph: &ValidatedGraph<E>,
    constraints: &[NodeConstraint<'_>],
) -> Result<Vec<Caps>, NegotiationFailure> {
    // Default node labels for the caps explainer: `n{id}:{kind}`. A caller with
    // element names (the runner) uses `solve_graph_labeled` for prettier output.
    solve_graph_labeled(graph, constraints, &|n| node_label_default(graph, n))
}

/// [`solve_graph`] with caller-supplied node labels for the caps-negotiation
/// explainer (DESIGN.md 4.20a). The runner passes each node's element category
/// (e.g. `h264parse`) so the `G2G_CAPS_TRACE` narration reads in element names
/// rather than node ids; direct callers use [`solve_graph`]'s `n{id}:{kind}`
/// default. The solve itself is identical; `label` only affects the log text,
/// and all formatting is skipped unless the [`CAPS_CATEGORY`] is enabled.
pub fn solve_graph_labeled<E>(
    graph: &ValidatedGraph<E>,
    constraints: &[NodeConstraint<'_>],
    label: &dyn Fn(NodeId) -> String,
) -> Result<Vec<Caps>, NegotiationFailure> {
    let n = graph.node_count();
    if n < 2 || constraints.len() != n {
        return Err(NegotiationFailure::Degenerate);
    }
    let ne = graph.edge_count();
    let mut edges: Vec<Option<CapsSet>> = alloc::vec![None; ne];

    let t = Target::category(CAPS_CATEGORY);
    let trace = log::enabled(CAPS_CATEGORY, LogLevel::Debug);
    if trace {
        crate::g2g_debug!(t, "negotiating {n} nodes, {ne} edges:");
        for (i, c) in constraints.iter().enumerate() {
            crate::g2g_debug!(t, "  {} {}", label(NodeId(i as u32)), fmt_constraint(c));
        }
    }

    // Narrate a structured failure once, on the way out: name the conflicting
    // nodes and dump the current set on every edge incident to them, so a
    // `CapsMismatch` reads as "these two can't agree, here's what each wanted".
    // Emitted at error level (visible in any run with a sink, not just a trace),
    // but only formatted on the rare failure path.
    let report = |f: &NegotiationFailure, edges: &[Option<CapsSet>]| {
        match f {
            NegotiationFailure::EmptyLink { upstream, downstream } => {
                let (up, down) = (NodeId(*upstream as u32), NodeId(*downstream as u32));
                crate::g2g_error!(t, "no caps overlap between {} and {}", label(up), label(down));
                for (id, slot) in edges.iter().enumerate() {
                    let e = graph.edge(id);
                    if [e.src.node, e.dst.node].iter().any(|&x| x == up || x == down) {
                        crate::g2g_error!(
                            t,
                            "  {} -> {}: {}",
                            label(e.src.node),
                            label(e.dst.node),
                            fmt_set_opt(slot)
                        );
                    }
                }
            }
            other => crate::g2g_error!(t, "negotiation failed: {other:?}"),
        }
    };

    // Same convergence bound as the linear solver, generalized to edges.
    let max_iters = 8 * ne + 4;
    for _ in 0..max_iters {
        let snapshot = edges.clone();
        for &node in graph.topo() {
            if let Err(f) = apply_node(graph, node, constraints, &mut edges) {
                report(&f, &edges);
                return Err(f);
            }
        }
        for &node in graph.topo().iter().rev() {
            if let Err(f) = apply_node(graph, node, constraints, &mut edges) {
                report(&f, &edges);
                return Err(f);
            }
        }
        if edges == snapshot {
            break;
        }
    }

    // Build a per-edge candidate domain (each surviving alternative, fixated, in
    // the set's own preference order so the first candidate is exactly what the
    // old per-edge `fixate()` would have picked). Then assign one candidate per
    // edge by backtracking search so the chosen combination is *globally*
    // consistent. Arc consistency above narrows each edge against its neighbours
    // pairwise, but a diamond (a tee whose branches re-converge at a fan-in)
    // couples branch choices in a way pairwise narrowing cannot see, so per-edge
    // greedy fixation can pick a locally-valid yet jointly-impossible combination
    // (e.g. two branches that map the shared tee value to different outputs whose
    // alternative orders disagree). The search tries the greedy choice first, so a
    // chain or an independent fan-out fixates byte-for-byte as before and only a
    // genuinely coupled diamond ever explores alternatives.
    let mut domains: Vec<Vec<Caps>> = Vec::with_capacity(ne);
    for (id, slot) in edges.iter().enumerate() {
        let (up, down) = edge_endpoints(graph, id);
        let set = match slot.as_ref() {
            Some(s) if !s.is_empty() => s,
            _ => {
                let f = NegotiationFailure::EmptyLink { upstream: up, downstream: down };
                report(&f, &edges);
                return Err(f);
            }
        };
        let mut doms: Vec<Caps> = Vec::new();
        for alt in set.alternatives() {
            if let Ok(c) = alt.fixate() {
                if !doms.contains(&c) {
                    doms.push(c);
                }
            }
        }
        if doms.is_empty() {
            crate::g2g_error!(
                t,
                "{} -> {}: {} ✗ cannot fixate (still ambiguous after narrowing)",
                label(graph.edge(id).src.node),
                label(graph.edge(id).dst.node),
                fmt_set(set)
            );
            return Err(NegotiationFailure::Unfixable { upstream: up, downstream: down });
        }
        domains.push(doms);
    }

    let mut assign: Vec<Option<Caps>> = alloc::vec![None; ne];
    if !fixate_backtrack(graph, constraints, &domains, &mut assign, 0) {
        let f = NegotiationFailure::NoConsistentFixation;
        report(&f, &edges);
        return Err(f);
    }
    let out: Vec<Caps> = assign.into_iter().map(|a| a.expect("every edge assigned")).collect();
    if trace {
        for (id, c) in out.iter().enumerate() {
            let e = graph.edge(id);
            crate::g2g_debug!(
                t,
                "{} -> {}: {} ✓ -> {}",
                label(e.src.node),
                label(e.dst.node),
                fmt_set(edges[id].as_ref().expect("edge set present")),
                c.to_gst_string()
            );
        }
    }
    Ok(out)
}

/// Backtracking search for a globally-consistent edge assignment, run after arc
/// consistency has narrowed each edge's domain. Assigns edges in id order, trying
/// each candidate (greedy choice first) and pruning the moment a node whose edges
/// are all assigned violates its relation. Returns `true` with `assign` filled on
/// success. Recursion depth is the edge count and domains are tiny in practice
/// (almost always one candidate), so the worst-case product is never approached
/// for real graphs.
fn fixate_backtrack<E>(
    graph: &ValidatedGraph<E>,
    constraints: &[NodeConstraint<'_>],
    domains: &[Vec<Caps>],
    assign: &mut [Option<Caps>],
    edge: usize,
) -> bool {
    if edge == domains.len() {
        return true;
    }
    let e = graph.edge(edge);
    for cand in &domains[edge] {
        assign[edge] = Some(cand.clone());
        if node_consistent(graph, constraints, assign, e.src.node)
            && node_consistent(graph, constraints, assign, e.dst.node)
            && fixate_backtrack(graph, constraints, domains, assign, edge + 1)
        {
            return true;
        }
    }
    assign[edge] = None;
    false
}

/// Whether `node`'s relation holds for the currently-assigned values of its
/// incident edges. Returns `true` while any incident edge is still unassigned
/// (the relation cannot be violated yet), so the caller can check a node the
/// moment its last edge is assigned. Per-edge membership (a source's produce set,
/// a sink's / muxer pad's accept set) is already guaranteed by arc consistency;
/// the load-bearing checks here are the cross-edge ones a diamond needs: a tee's
/// branches all carrying its input, an `Identity`'s in == out, a `Mapping`'s
/// (in, out) being one declared pair, and a derived transform's out in f(in).
fn node_consistent<E>(
    graph: &ValidatedGraph<E>,
    constraints: &[NodeConstraint<'_>],
    assign: &[Option<Caps>],
    node: NodeId,
) -> bool {
    let in_e = graph.in_edges(node);
    let out_e = graph.out_edges(node);
    if in_e.iter().chain(out_e.iter()).any(|&e| assign[e].is_none()) {
        return true;
    }
    let get = |e: usize| assign[e].as_ref().expect("checked all assigned");
    match graph.kind(node) {
        // Membership is already ensured by arc consistency; nothing cross-edge.
        NodeKind::Source | NodeKind::Sink | NodeKind::Muxer(_) => true,
        NodeKind::Tee(_) => {
            // A demux decouples its ports (each its own produce set, ensured by arc
            // consistency), so there is no cross-edge equality; a broadcast tee
            // requires every branch to carry its input.
            if matches!(&constraints[node.0 as usize], NodeConstraint::Demux { .. }) {
                true
            } else {
                let inp = get(in_e[0]);
                out_e.iter().all(|&o| get(o) == inp)
            }
        }
        NodeKind::Transform => match &constraints[node.0 as usize] {
            NodeConstraint::Element(c) => transform_pair_consistent(c, get(in_e[0]), get(out_e[0])),
            _ => true,
        },
    }
}

/// The cross-edge relation a transform's `(input, output)` pair must satisfy, for
/// [`node_consistent`]'s backtracking check.
fn transform_pair_consistent(c: &CapsConstraint<'_>, inp: &Caps, outp: &Caps) -> bool {
    match c {
        CapsConstraint::Identity(_) | CapsConstraint::IdentityAny => inp == outp,
        CapsConstraint::Mapping(pairs) => {
            pairs.iter().any(|(i, o)| i.accepts(inp) && o.accepts(outp))
        }
        CapsConstraint::DerivedOutput(f) => f(inp).accepts(outp),
        CapsConstraint::DerivedCoupled { derive, .. } => derive(inp).accepts(outp),
        // Produce / accept shapes on a transform slot, or legacy bridges: not a
        // cross-edge relation re-checked here (arc consistency handled the forward
        // cascade; the legacy bridge stays permissive through the migration).
        _ => true,
    }
}

/// Default node label for the caps explainer: `n{id}:{kind}` (e.g. `n2:xform`),
/// used when the caller supplies no element names.
fn node_label_default<E>(graph: &ValidatedGraph<E>, node: NodeId) -> String {
    alloc::format!("n{}:{}", node.0, kind_short(graph.kind(node)))
}

fn kind_short(kind: NodeKind) -> &'static str {
    match kind {
        NodeKind::Source => "src",
        NodeKind::Transform => "xform",
        NodeKind::Sink => "sink",
        NodeKind::Tee(_) => "tee",
        NodeKind::Muxer(_) => "mux",
    }
}

/// Render a `CapsSet` for the explainer: its alternatives joined by ` | `, with
/// `∅` for empty and a `(+N more)` elision past four so a wide set stays one
/// readable line.
fn fmt_set(set: &CapsSet) -> String {
    let alts = set.alternatives();
    if alts.is_empty() {
        return String::from("∅");
    }
    let mut parts: Vec<String> = Vec::new();
    for (i, a) in alts.iter().enumerate() {
        if i == 4 {
            parts.push(alloc::format!("(+{} more)", alts.len() - 4));
            break;
        }
        parts.push(a.to_gst_string());
    }
    parts.join(" | ")
}

fn fmt_set_opt(slot: &Option<CapsSet>) -> String {
    match slot {
        Some(s) => fmt_set(s),
        None => String::from("(unconstrained)"),
    }
}

/// One-line summary of a node's constraint for the explainer's setup dump.
fn fmt_constraint(nc: &NodeConstraint<'_>) -> String {
    match nc {
        NodeConstraint::Element(c) => fmt_caps_constraint(c),
        NodeConstraint::Muxer { inputs, output, follows } => match follows {
            Some(pad) => alloc::format!("mux {} inputs -> follows input {pad}", inputs.len()),
            None => alloc::format!("mux {} inputs -> {}", inputs.len(), fmt_caps_constraint(output)),
        },
        NodeConstraint::Demux { ports, .. } => alloc::format!("demux -> {} ports", ports.len()),
    }
}

fn fmt_caps_constraint(c: &CapsConstraint<'_>) -> String {
    match c {
        CapsConstraint::Produces(s) => alloc::format!("produces {}", fmt_set(s)),
        CapsConstraint::Accepts(s) => alloc::format!("accepts {}", fmt_set(s)),
        CapsConstraint::AcceptsAny => "accepts ANY".to_string(),
        CapsConstraint::Identity(s) => alloc::format!("identity {}", fmt_set(s)),
        CapsConstraint::IdentityAny => "identity ANY".to_string(),
        CapsConstraint::Mapping(pairs) => alloc::format!("maps {} pair(s)", pairs.len()),
        CapsConstraint::DerivedOutput(_) => "derives output".to_string(),
        CapsConstraint::DerivedCoupled { .. } => "derives output (coupled)".to_string(),
        CapsConstraint::LegacySource(c) => alloc::format!("legacy source {}", c.to_gst_string()),
        CapsConstraint::LegacyTransform { .. } => "legacy transform".to_string(),
        CapsConstraint::LegacySink(_) => "legacy sink".to_string(),
    }
}

/// Per-edge downstream feasibility for the DAG runner's mid-stream re-solve
/// (D4), the graph generalization of [`downstream_feasibility`]. For each edge
/// it returns the set the edge can carry such that every node *downstream* of
/// it can still fixate, ignoring the (mid-stream-changing) upstream. `None`
/// means "downstream imposes no expressible constraint here". Indexed by edge
/// id; snapshotted into each arm so a mid-stream `CapsChanged` can steer an
/// element's output without reaching its peers at runtime.
///
/// Generalizes the linear reverse sweep to a reverse-topo fold: a transform
/// passes its output feasibility back through [`backward_feasible`]; a tee's
/// input feasibility is the intersection over its branch feasibilities (the
/// input must satisfy every branch); a muxer's input pads take their pad accept
/// sets independently (the output does not feed back to the inputs).
#[cfg(feature = "std")]
pub(crate) fn graph_downstream_feasibility<E>(
    graph: &ValidatedGraph<E>,
    constraints: &[NodeConstraint<'_>],
    solution: &[Caps],
) -> Vec<Option<CapsSet>> {
    let ne = graph.edge_count();
    let mut feas: Vec<Option<CapsSet>> = alloc::vec![None; ne];
    // Reverse topo: a node's output edges are written by its downstream
    // consumers, visited earlier in this order, before we read them here.
    for &node in graph.topo().iter().rev() {
        let idx = node.0 as usize;
        match graph.kind(node) {
            NodeKind::Source => {}
            NodeKind::Sink => {
                let ie = graph.in_edges(node)[0];
                feas[ie] = match &constraints[idx] {
                    NodeConstraint::Element(CapsConstraint::Accepts(s)) => Some(s.clone()),
                    _ => None,
                };
            }
            NodeKind::Transform => {
                let ie = graph.in_edges(node)[0];
                let oe = graph.out_edges(node)[0];
                if let NodeConstraint::Element(c) = &constraints[idx] {
                    // The element's startup-fixated input, for the closure probe a
                    // `DerivedOutput` backward hop needs (see `backward_feasible`).
                    feas[ie] = backward_feasible(c, feas[oe].as_ref(), solution.get(ie));
                }
            }
            NodeKind::Tee(_) => {
                // A demux decouples its ports, so its input feasibility is its own
                // `input` accept (not the branches'); a broadcast tee's input must
                // satisfy every branch, so it is their intersection.
                if let NodeConstraint::Demux { input, .. } = &constraints[idx] {
                    feas[graph.in_edges(node)[0]] = match input {
                        CapsConstraint::Accepts(s) => Some(s.clone()),
                        _ => None,
                    };
                } else {
                    let mut acc: Option<CapsSet> = None;
                    for &oe in graph.out_edges(node) {
                        if let Some(s) = feas[oe].as_ref() {
                            acc = Some(match acc {
                                Some(a) => a.intersect(s),
                                None => s.clone(),
                            });
                        }
                    }
                    feas[graph.in_edges(node)[0]] = acc;
                }
            }
            NodeKind::Muxer(_) => {
                if let NodeConstraint::Muxer { inputs, .. } = &constraints[idx] {
                    for &ie in graph.in_edges(node) {
                        let pad = graph.edge(ie).dst.index as usize;
                        feas[ie] = match inputs.get(pad) {
                            Some(CapsConstraint::Accepts(s)) => Some(s.clone()),
                            _ => None,
                        };
                    }
                }
            }
        }
    }
    feas
}

/// The (upstream node, downstream node) ids an edge connects, for failures.
fn edge_endpoints<E>(graph: &ValidatedGraph<E>, edge_id: usize) -> (usize, usize) {
    let e = graph.edge(edge_id);
    (e.src.node.0 as usize, e.dst.node.0 as usize)
}

fn apply_node<E>(
    graph: &ValidatedGraph<E>,
    node: NodeId,
    constraints: &[NodeConstraint<'_>],
    edges: &mut [Option<CapsSet>],
) -> Result<(), NegotiationFailure> {
    let kind = graph.kind(node);
    let in_e = graph.in_edges(node);
    let out_e = graph.out_edges(node);
    let idx = node.0 as usize;
    let nc = &constraints[idx];
    let shape_err = NegotiationFailure::EndpointShapeMismatch { index: idx };
    match kind {
        NodeKind::Source => match nc {
            NodeConstraint::Element(CapsConstraint::Produces(s)) => {
                narrow_edge(graph, edges, out_e[0], s)
            }
            // Legacy bridge: a `LegacySource` carries one fixated caps, the same
            // as `Produces(one(caps))`.
            NodeConstraint::Element(CapsConstraint::LegacySource(caps)) => {
                narrow_edge(graph, edges, out_e[0], &CapsSet::one(caps.clone()))
            }
            _ => Err(shape_err),
        },
        NodeKind::Sink => match nc {
            NodeConstraint::Element(CapsConstraint::Accepts(s)) => {
                narrow_edge(graph, edges, in_e[0], s)
            }
            // `AcceptsAny` and a `LegacySink` both leave the input edge to carry
            // whatever the upstream fixates: the legacy sink's `intercept` is the
            // terminal accept (the runner configures it with the upstream caps,
            // as `run_muxer_sink` did), so it imposes no solver narrowing.
            NodeConstraint::Element(CapsConstraint::AcceptsAny)
            | NodeConstraint::Element(CapsConstraint::LegacySink(_)) => Ok(()),
            _ => Err(shape_err),
        },
        NodeKind::Transform => match nc {
            NodeConstraint::Element(c) => {
                apply_transform_node(graph, c, in_e[0], out_e[0], edges, node)
            }
            _ => Err(shape_err),
        },
        NodeKind::Tee(_) => match nc {
            // A demux decouples its ports (each its own elementary stream); a
            // plain (broadcast) tee couples in == every out.
            NodeConstraint::Demux { input, ports } => {
                apply_demux_node(graph, in_e[0], out_e, input, ports, edges)
            }
            _ => apply_tee_node(graph, in_e[0], out_e, edges),
        },
        NodeKind::Muxer(_) => match nc {
            NodeConstraint::Muxer { inputs, output, follows } => {
                apply_muxer_node(graph, node, inputs, output, *follows, edges)
            }
            _ => Err(shape_err),
        },
    }
}

fn narrow_edge<E>(
    graph: &ValidatedGraph<E>,
    edges: &mut [Option<CapsSet>],
    edge_id: usize,
    contrib: &CapsSet,
) -> Result<(), NegotiationFailure> {
    let next = match &edges[edge_id] {
        Some(cur) => cur.intersect(contrib),
        None => contrib.clone(),
    };
    if next.is_empty() {
        let (up, down) = edge_endpoints(graph, edge_id);
        return Err(NegotiationFailure::EmptyLink { upstream: up, downstream: down });
    }
    edges[edge_id] = Some(next);
    Ok(())
}

/// Couple two edges to carry equal caps (each intersected with the other),
/// the pass-through relation a transform's input and output share.
fn couple_edges<E>(
    graph: &ValidatedGraph<E>,
    edges: &mut [Option<CapsSet>],
    a: usize,
    b: usize,
) -> Result<(), NegotiationFailure> {
    match (edges[a].clone(), edges[b].clone()) {
        (Some(sa), Some(sb)) => {
            let coupled = sa.intersect(&sb);
            if coupled.is_empty() {
                let up = edge_endpoints(graph, a).0;
                let down = edge_endpoints(graph, b).1;
                return Err(NegotiationFailure::EmptyLink { upstream: up, downstream: down });
            }
            edges[a] = Some(coupled.clone());
            edges[b] = Some(coupled);
        }
        (Some(sa), None) => edges[b] = Some(sa),
        (None, Some(sb)) => edges[a] = Some(sb),
        (None, None) => {}
    }
    Ok(())
}

/// Transform node narrowing: the edge-indexed analog of the linear solver's
/// `apply_constraint` transform arms.
fn apply_transform_node<E>(
    graph: &ValidatedGraph<E>,
    c: &CapsConstraint<'_>,
    in_e: usize,
    out_e: usize,
    edges: &mut [Option<CapsSet>],
    node: NodeId,
) -> Result<(), NegotiationFailure> {
    match c {
        CapsConstraint::Identity(s) => {
            narrow_edge(graph, edges, in_e, s)?;
            narrow_edge(graph, edges, out_e, s)?;
            couple_edges(graph, edges, in_e, out_e)
        }
        CapsConstraint::IdentityAny => couple_edges(graph, edges, in_e, out_e),
        CapsConstraint::Mapping(pairs) => {
            let mut new_in = CapsSet::from_alternatives(Vec::new());
            let mut new_out = CapsSet::from_alternatives(Vec::new());
            for (in_set, out_set) in pairs {
                let in_match = match &edges[in_e] {
                    Some(cur) => cur.intersect(in_set),
                    None => in_set.clone(),
                };
                let out_match = match &edges[out_e] {
                    Some(cur) => cur.intersect(out_set),
                    None => out_set.clone(),
                };
                if !in_match.is_empty() && !out_match.is_empty() {
                    new_in = new_in.union(&in_match);
                    new_out = new_out.union(&out_match);
                }
            }
            if new_in.is_empty() || new_out.is_empty() {
                let up = edge_endpoints(graph, in_e).0;
                let down = edge_endpoints(graph, out_e).1;
                return Err(NegotiationFailure::EmptyLink { upstream: up, downstream: down });
            }
            edges[in_e] = Some(new_in);
            edges[out_e] = Some(new_out);
            Ok(())
        }
        CapsConstraint::DerivedOutput(f) => {
            // Forward (M188): narrow the output edge by the union of `f` over
            // every input alternative, mirroring the linear solver. A single
            // fixated input gives `f(input)`; a still ambiguous input (a stacked
            // auto transform) still yields an output to narrow instead of leaving
            // the output edge at `None`.
            if let Some(in_set) = edges[in_e].clone() {
                let derived = forward_derived_union(f.as_ref(), &in_set);
                if derived.is_empty() {
                    let (up, down) = edge_endpoints(graph, out_e);
                    return Err(NegotiationFailure::EmptyLink { upstream: up, downstream: down });
                }
                narrow_edge(graph, edges, out_e, &derived)?;
            }
            // Backward (M188 + invertible-field coupling): field-level narrow on
            // the closure's probed passthrough fields, else alternative-drop.
            if let (Some(in_set), Some(out_set)) = (edges[in_e].clone(), edges[out_e].clone()) {
                match derived_backward(f.as_ref(), &in_set, &out_set) {
                    Ok(Some(narrowed)) => edges[in_e] = Some(narrowed),
                    Ok(None) => {}
                    Err(()) => {
                        let (up, down) = edge_endpoints(graph, in_e);
                        return Err(NegotiationFailure::EmptyLink { upstream: up, downstream: down });
                    }
                }
            }
            Ok(())
        }
        CapsConstraint::DerivedCoupled { derive, passthrough } => {
            // Mirror of the linear `apply_constraint` arm on graph edges:
            // forward via the closure, backward via field-level coupling.
            if let Some(in_set) = edges[in_e].clone() {
                let derived = forward_derived_union(derive.as_ref(), &in_set);
                if derived.is_empty() {
                    let (up, down) = edge_endpoints(graph, out_e);
                    return Err(NegotiationFailure::EmptyLink { upstream: up, downstream: down });
                }
                narrow_edge(graph, edges, out_e, &derived)?;
            }
            if let (Some(in_set), Some(out_set)) = (edges[in_e].clone(), edges[out_e].clone()) {
                match backward_field_narrow(derive.as_ref(), *passthrough, &in_set, &out_set) {
                    Ok(Some(narrowed)) => edges[in_e] = Some(narrowed),
                    Ok(None) => {}
                    Err(()) => {
                        let (up, down) = edge_endpoints(graph, in_e);
                        return Err(NegotiationFailure::EmptyLink { upstream: up, downstream: down });
                    }
                }
            }
            Ok(())
        }
        // Legacy bridge: forward `intercept(input)` to the output once the input
        // fixates, the same single-caps forward cascade `solve_legacy_cascade`
        // runs (no backward coupling, like the mixed-cascade path).
        CapsConstraint::LegacyTransform { intercept, .. } => {
            if let Some(fixed_input) = edges[in_e].as_ref().and_then(fixed_single) {
                let out = intercept(&fixed_input).map_err(|_| {
                    let (up, down) = edge_endpoints(graph, out_e);
                    NegotiationFailure::EmptyLink { upstream: up, downstream: down }
                })?;
                return narrow_edge(graph, edges, out_e, &CapsSet::one(out));
            }
            Ok(())
        }
        _ => Err(NegotiationFailure::EndpointShapeMismatch { index: node.0 as usize }),
    }
}

/// The single concrete caps an edge has fixated to, or `None` if it still has
/// multiple alternatives or ranged (`Any`) fields. Used by the forward-cascade
/// constraints (`DerivedOutput`, `LegacyTransform`) that need a concrete input.
fn fixed_single(set: &CapsSet) -> Option<Caps> {
    let fixed = set.fixate()?;
    (set.alternatives().len() == 1 && set.alternatives()[0] == fixed).then_some(fixed)
}

/// Forward image of a `DerivedOutput` transform over its (possibly ambiguous)
/// input set: the union of `f` over the input alternatives (M188). For a single
/// fixated input this is just `f(input)`; for a multi-alternative input it lets a
/// downstream auto transform still receive an output to narrow, instead of
/// stalling until the input fixates (which it can't, with no downstream pin).
fn forward_derived_union(f: &dyn Fn(&Caps) -> CapsSet, in_set: &CapsSet) -> CapsSet {
    in_set
        .alternatives()
        .iter()
        .fold(CapsSet::from_alternatives(Vec::new()), |acc, a| acc.union(&f(a)))
}

/// M188 backward narrowing for a `DerivedOutput` transform: given the (already
/// constrained) output set, drop input alternatives whose forward image `f(a)`
/// can no longer satisfy it. `f` is not analytically invertible, but it is
/// evaluable per candidate, so a downstream pin propagates back through a
/// not-yet-fixated transform, letting stacked auto transforms
/// (`videoconvert ! videoscale ! caps`) resolve.
///
/// Only narrows when the input is still ambiguous (more than one alternative),
/// so single-input transforms (decoders, the single-transform pipelines of
/// M185/M186) are untouched. Returns `Some(narrowed)` when it removed
/// alternatives, `None` when unchanged, `Err(())` when nothing survives.
fn backward_filter_derived(
    f: &dyn Fn(&Caps) -> CapsSet,
    in_set: &CapsSet,
    out_set: &CapsSet,
) -> Result<Option<CapsSet>, ()> {
    if in_set.alternatives().len() <= 1 {
        return Ok(None);
    }
    let kept: Vec<Caps> = in_set
        .alternatives()
        .iter()
        .filter(|a| !f(a).intersect(out_set).is_empty())
        .cloned()
        .collect();
    if kept.is_empty() {
        return Err(());
    }
    if kept.len() == in_set.alternatives().len() {
        return Ok(None);
    }
    Ok(Some(CapsSet::from_alternatives(kept)))
}

/// Backward narrowing for a `DerivedOutput` transform. The closure is not
/// declared with a passthrough mask, so [`discover_passthrough`] probes it for
/// its invertible fields; when any is found the input is narrowed field-by-field
/// exactly as a declared `DerivedCoupled` mask would
/// ([`backward_field_narrow`]), so a downstream geometry / framerate pin couples
/// back through a decoder or a rescaling convert instead of failing loud. With no
/// passthrough field discovered it falls back to the alternative-drop walk
/// ([`backward_filter_derived`]), the prior behavior, so a genuinely
/// non-invertible closure is untouched.
fn derived_backward(
    f: &dyn Fn(&Caps) -> CapsSet,
    in_set: &CapsSet,
    out_set: &CapsSet,
) -> Result<Option<CapsSet>, ()> {
    let mask = in_set
        .alternatives()
        .first()
        .map(|sample| discover_passthrough(f, sample))
        .unwrap_or(PassthroughFields::NONE);
    if mask == PassthroughFields::NONE {
        backward_filter_derived(f, in_set, out_set)
    } else {
        backward_field_narrow(f, mask, in_set, out_set)
    }
}

/// Backward field-coupling for a `DerivedCoupled` transform: the primitive the
/// alternative-dropping [`backward_filter_derived`] cannot express. For each
/// input alternative, intersect its forward image `derive(a)` with the
/// constrained output `out_set`; drop the alternative when nothing survives (the
/// same as the alternative-drop walk), otherwise narrow the alternative's
/// *passthrough* fields by intersecting each reachable output's passthrough
/// fields back in (`couple_passthrough`), e.g. a `Range(1..MAX)` width meeting a
/// `Fixed(160)` downstream pin collapses to `Fixed(160)`.
///
/// Unlike `backward_filter_derived` it runs for a single-alternative input too:
/// narrowing a `Range` field *within* that one alternative is the whole point.
/// Every step is an intersection (monotone shrink), so the arc-consistency loop
/// still converges. Returns `Some(narrowed)` when it changed the set, `None`
/// when unchanged, `Err(())` when nothing survives.
fn backward_field_narrow(
    derive: &dyn Fn(&Caps) -> CapsSet,
    passthrough: PassthroughFields,
    in_set: &CapsSet,
    out_set: &CapsSet,
) -> Result<Option<CapsSet>, ()> {
    let mut kept: Vec<Caps> = Vec::new();
    let mut changed = false;
    for a in in_set.alternatives() {
        let reach = derive(a).intersect(out_set);
        if reach.is_empty() {
            changed = true; // this input alternative can't reach the output: drop it
            continue;
        }
        // Couple each reachable output's passthrough fields back into `a`. Uses
        // the variant-tolerant coupling so a `DerivedOutput` decoder / encoder
        // (which changes variant) couples its shared geometry / rate fields;
        // a same-variant `DerivedCoupled` transform gets the exact coupling.
        let mut any = false;
        for out_alt in reach.alternatives() {
            if let Some(c) = couple_passthrough_derived(a, out_alt, passthrough) {
                if &c != a {
                    changed = true;
                }
                if !kept.contains(&c) {
                    kept.push(c);
                }
                any = true;
            }
        }
        if !any {
            // Reachable output exists but a passthrough field conflicts: drop.
            changed = true;
        }
    }
    if kept.is_empty() {
        return Err(());
    }
    if !changed {
        return Ok(None);
    }
    Ok(Some(CapsSet::from_alternatives(kept)))
}

/// A tee fans its input caps out to every output unchanged: couple the input
/// edge and all output edges to one shared set (their intersection).
fn apply_tee_node<E>(
    graph: &ValidatedGraph<E>,
    in_e: usize,
    out_e: &[usize],
    edges: &mut [Option<CapsSet>],
) -> Result<(), NegotiationFailure> {
    let mut acc: Option<CapsSet> = edges[in_e].clone();
    for &oe in out_e {
        if let Some(s) = edges[oe].clone() {
            acc = Some(match acc {
                Some(a) => a.intersect(&s),
                None => s,
            });
        }
    }
    if let Some(coupled) = acc {
        if coupled.is_empty() {
            let (up, down) = edge_endpoints(graph, in_e);
            return Err(NegotiationFailure::EmptyLink { upstream: up, downstream: down });
        }
        edges[in_e] = Some(coupled.clone());
        for &oe in out_e {
            edges[oe] = Some(coupled.clone());
        }
    }
    Ok(())
}

/// Apply a fan-out **demux** node (M380): its ports are decoupled, so the input
/// edge narrows by the demux's `input` accept (the container it consumes) and each
/// output edge narrows by its port's `Produces` caps (its elementary stream),
/// independent of one another. The port for an output edge is its source pad
/// index, so the order of `out_e` does not matter.
fn apply_demux_node<E>(
    graph: &ValidatedGraph<E>,
    in_e: usize,
    out_e: &[usize],
    input: &CapsConstraint<'_>,
    ports: &[CapsConstraint<'_>],
    edges: &mut [Option<CapsSet>],
) -> Result<(), NegotiationFailure> {
    // The byte-stream input: an `Accepts` narrows it; `AcceptsAny` / `LegacySink`
    // leave it to whatever the source fixates (the common case, the demux's
    // `intercept` being the terminal accept).
    if let CapsConstraint::Accepts(s) = input {
        narrow_edge(graph, edges, in_e, s)?;
    }
    for &oe in out_e {
        let port = graph.edge(oe).src.index as usize;
        if let Some(CapsConstraint::Produces(s)) = ports.get(port) {
            narrow_edge(graph, edges, oe, s)?;
        }
    }
    Ok(())
}

/// A muxer fans in: apply each input pad's constraint to its edge (`Accepts`
/// narrows, `AcceptsAny` leaves the edge to carry whatever per-frame caps flow
/// through), and the single output edge by the `Produces` set. `inputs[i]`
/// applies to input pad `i`; D1 validation guarantees each input pad index
/// appears exactly once.
fn apply_muxer_node<E>(
    graph: &ValidatedGraph<E>,
    node: NodeId,
    inputs: &[CapsConstraint<'_>],
    output: &CapsConstraint<'_>,
    follows: Option<usize>,
    edges: &mut [Option<CapsSet>],
) -> Result<(), NegotiationFailure> {
    let idx = node.0 as usize;
    let shape_err = NegotiationFailure::EndpointShapeMismatch { index: idx };
    for &eid in graph.in_edges(node) {
        let pad = graph.edge(eid).dst.index as usize;
        match inputs.get(pad) {
            Some(CapsConstraint::Accepts(set)) => narrow_edge(graph, edges, eid, set)?,
            // `AcceptsAny` and a legacy input pad both forward per-frame caps
            // without narrowing the edge.
            Some(CapsConstraint::AcceptsAny) | Some(CapsConstraint::LegacySink(_)) => {}
            _ => return Err(shape_err),
        }
    }
    let out_edge = graph.out_edges(node)[0];
    // Identity-passthrough mux: the output edge is the followed input pad's caps.
    // The solver iterates to a fixpoint, so if that input edge is not yet solved
    // this pass narrows nothing and a later pass (once the source has cascaded
    // forward) couples them; coupling keeps the two edges equal thereafter.
    if let Some(pad) = follows {
        let in_edge = graph
            .in_edges(node)
            .iter()
            .copied()
            .find(|&e| graph.edge(e).dst.index as usize == pad)
            .ok_or(shape_err)?;
        return couple_edges(graph, edges, in_edge, out_edge);
    }
    match output {
        CapsConstraint::Produces(set) => narrow_edge(graph, edges, out_edge, set),
        // A legacy muxer output carries one fixated merged caps.
        CapsConstraint::LegacySource(caps) => {
            narrow_edge(graph, edges, out_edge, &CapsSet::one(caps.clone()))
        }
        _ => Err(shape_err),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::couple_passthrough;
    use crate::caps::{Dim, Rate, VideoCodec, RawVideoFormat};
    use alloc::boxed::Box;
    use alloc::vec;

    fn video(fmt: RawVideoFormat, w: Dim, h: Dim, r: Rate) -> Caps {
        Caps::RawVideo { format: fmt, width: w, height: h, framerate: r }
    }

    // A multi-hop tensor chain `Produces(f32) -> quantize(f32->u8) ->
    // infer(u8->[1,N]) -> AcceptsAny` must negotiate: tensor caps have no
    // wildcard fields, so the DerivedOutput closure is the only source of truth
    // for the output, and the solver must seed the output edge from it (M451).
    #[test]
    fn solve_linear_tensor_dtype_change_chain() {
        use crate::caps::{TensorDType, TensorLayout, TensorShape};
        let t = |d: TensorDType, s: TensorShape| Caps::Tensor {
            dtype: d,
            shape: s,
            layout: TensorLayout::Nchw,
        };
        let f32_in = t(TensorDType::F32, TensorShape::new([1, 3, 4, 4]));
        let u8_mid = t(TensorDType::U8, TensorShape::new([1, 3, 4, 4]));
        let logits = t(TensorDType::F32, TensorShape::new([1, 10]));

        let src = CapsConstraint::Produces(CapsSet::one(f32_in.clone()));
        // quantize: f32 -> u8, shape/layout passthrough (the TensorConvert shape).
        let quant = CapsConstraint::DerivedOutput(Box::new(|inp: &Caps| match inp {
            Caps::Tensor { dtype: TensorDType::F32, shape, layout } => {
                CapsSet::one(Caps::Tensor { dtype: TensorDType::U8, shape: *shape, layout: *layout })
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }));
        // infer: u8 [1,3,4,4] -> f32 [1,10] (the OrtInference shape).
        let logits_c = logits.clone();
        let infer = CapsConstraint::DerivedOutput(Box::new(move |inp: &Caps| match inp {
            Caps::Tensor { dtype: TensorDType::U8, .. } => CapsSet::one(logits_c.clone()),
            _ => CapsSet::from_alternatives(Vec::new()),
        }));
        let sink = CapsConstraint::AcceptsAny;

        let links = solve_linear(&[&src, &quant, &infer, &sink]).expect("tensor chain negotiates");
        assert_eq!(links[0], f32_in, "source link f32");
        assert_eq!(links[1], u8_mid, "quantize output is u8, not the source f32");
        assert_eq!(links[2], logits, "inference output [1,10]");
    }

    // The DAG solver (the path `run_linear_chain` -> `run_graph` takes) must
    // negotiate the same tensor dtype-change chain as the linear solver.
    #[test]
    fn solve_graph_tensor_dtype_change_chain() {
        use crate::caps::{TensorDType, TensorLayout, TensorShape};
        use crate::graph::Graph;
        let t = |d: TensorDType, s: TensorShape| Caps::Tensor {
            dtype: d,
            shape: s,
            layout: TensorLayout::Nchw,
        };
        let f32_in = t(TensorDType::F32, TensorShape::new([1, 3, 4, 4]));
        let u8_mid = t(TensorDType::U8, TensorShape::new([1, 3, 4, 4]));
        let logits = t(TensorDType::F32, TensorShape::new([1, 10]));
        let logits_c = logits.clone();
        let cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(f32_in.clone()))),
            NodeConstraint::Element(CapsConstraint::DerivedOutput(Box::new(|inp: &Caps| match inp {
                Caps::Tensor { dtype: TensorDType::F32, shape, layout } => {
                    CapsSet::one(Caps::Tensor { dtype: TensorDType::U8, shape: *shape, layout: *layout })
                }
                _ => CapsSet::from_alternatives(Vec::new()),
            }))),
            NodeConstraint::Element(CapsConstraint::DerivedOutput(Box::new(move |inp: &Caps| match inp {
                Caps::Tensor { dtype: TensorDType::U8, .. } => CapsSet::one(logits_c.clone()),
                _ => CapsSet::from_alternatives(Vec::new()),
            }))),
            NodeConstraint::Element(CapsConstraint::AcceptsAny),
        ];
        let mut g: Graph<()> = Graph::new();
        let src = g.add_source(());
        let q = g.add_transform(());
        let inf = g.add_transform(());
        let sink = g.add_sink(());
        g.link(src, q).unwrap();
        g.link(q, inf).unwrap();
        g.link(inf, sink).unwrap();
        let v = g.finish().unwrap();
        let dag = solve_graph(&v, &cs).expect("tensor chain solves as a graph");
        assert_eq!(dag, vec![f32_in, u8_mid, logits]);
    }

    fn fixed_video(fmt: RawVideoFormat, w: u32, h: u32, fps: u32) -> Caps {
        video(fmt, Dim::Fixed(w), Dim::Fixed(h), Rate::Fixed(fps << 16))
    }

    fn compressed(codec: VideoCodec, w: Dim, h: Dim, r: Rate) -> Caps {
        Caps::CompressedVideo { codec, width: w, height: h, framerate: r }
    }

    fn fixed_compressed(codec: VideoCodec, w: u32, h: u32, fps: u32) -> Caps {
        compressed(codec, Dim::Fixed(w), Dim::Fixed(h), Rate::Fixed(fps << 16))
    }

    #[test]
    fn solves_source_sink_minimal_chain() {
        let src = CapsConstraint::Produces(CapsSet::one(fixed_video(RawVideoFormat::Nv12, 1280, 720, 30)));
        let sink = CapsConstraint::Accepts(CapsSet::one(video(
            RawVideoFormat::Nv12, Dim::Any, Dim::Any, Rate::Any,
        )));
        let links = solve_linear(&[&src, &sink]).unwrap();
        assert_eq!(links, vec![fixed_video(RawVideoFormat::Nv12, 1280, 720, 30)]);
    }

    #[test]
    fn empty_link_when_formats_disjoint() {
        let src = CapsConstraint::Produces(CapsSet::one(fixed_compressed(VideoCodec::H264, 1280, 720, 30)));
        let sink = CapsConstraint::Accepts(CapsSet::one(video(
            RawVideoFormat::Nv12, Dim::Any, Dim::Any, Rate::Any,
        )));
        assert_eq!(
            solve_linear(&[&src, &sink]),
            Err(NegotiationFailure::EmptyLink { upstream: 0, downstream: 1 })
        );
    }

    #[test]
    fn degenerate_when_fewer_than_two_elements() {
        let src = CapsConstraint::Produces(CapsSet::one(fixed_video(RawVideoFormat::Nv12, 1, 1, 1)));
        assert_eq!(solve_linear(&[&src]), Err(NegotiationFailure::Degenerate));
        assert_eq!(solve_linear(&[]), Err(NegotiationFailure::Degenerate));
    }

    #[test]
    fn endpoint_shape_mismatch_rejected() {
        let id = CapsConstraint::Identity(CapsSet::one(fixed_video(RawVideoFormat::Nv12, 1, 1, 1)));
        let sink = CapsConstraint::Accepts(CapsSet::one(fixed_video(RawVideoFormat::Nv12, 1, 1, 1)));
        assert_eq!(
            solve_linear(&[&id, &sink]),
            Err(NegotiationFailure::EndpointShapeMismatch { index: 0 })
        );
    }

    #[test]
    fn preference_tie_break_picks_self_first_alt() {
        // Source prefers Rgba8 then H264 (both fully fixed at the same
        // dims); sink accepts both with reversed preference.
        let rgba = fixed_video(RawVideoFormat::Rgba8, 640, 480, 30);
        let h264 = fixed_compressed(VideoCodec::H264, 640, 480, 30);
        let src = CapsConstraint::Produces(CapsSet::from_alternatives(vec![rgba.clone(), h264.clone()]));
        let sink = CapsConstraint::Accepts(CapsSet::from_alternatives(vec![h264.clone(), rgba.clone()]));
        let links = solve_linear(&[&src, &sink]).unwrap();
        // Source's outer preference wins because Produces is applied
        // first and CapsSet::intersect preserves self's order.
        assert_eq!(links, vec![rgba]);
    }

    #[test]
    fn identity_couples_input_and_output() {
        let src = CapsConstraint::Produces(CapsSet::one(fixed_video(RawVideoFormat::Nv12, 1280, 720, 30)));
        let id = CapsConstraint::Identity(CapsSet::one(video(
            RawVideoFormat::Nv12, Dim::Any, Dim::Any, Rate::Any,
        )));
        let sink = CapsConstraint::Accepts(CapsSet::one(video(
            RawVideoFormat::Nv12, Dim::Any, Dim::Any, Rate::Any,
        )));
        let links = solve_linear(&[&src, &id, &sink]).unwrap();
        assert_eq!(links, vec![
            fixed_video(RawVideoFormat::Nv12, 1280, 720, 30),
            fixed_video(RawVideoFormat::Nv12, 1280, 720, 30),
        ]);
    }

    #[test]
    fn identity_format_mismatch_returns_empty_link() {
        let src = CapsConstraint::Produces(CapsSet::one(fixed_compressed(VideoCodec::H264, 1280, 720, 30)));
        let id = CapsConstraint::Identity(CapsSet::one(video(
            RawVideoFormat::Nv12, Dim::Any, Dim::Any, Rate::Any,
        )));
        let sink = CapsConstraint::Accepts(CapsSet::one(video(
            RawVideoFormat::Nv12, Dim::Any, Dim::Any, Rate::Any,
        )));
        assert!(matches!(
            solve_linear(&[&src, &id, &sink]),
            Err(NegotiationFailure::EmptyLink { .. })
        ));
    }

    #[test]
    fn derived_output_evaluated_after_input_fixates() {
        // Decoder: H264 input → Nv12 output at the same dims.
        let src = CapsConstraint::Produces(CapsSet::one(fixed_compressed(VideoCodec::H264, 1920, 1080, 60)));
        let dec = CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::CompressedVideo { width, height, framerate, .. } => CapsSet::one(Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: width.clone(),
                height: height.clone(),
                framerate: framerate.clone(),
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        }));
        let sink = CapsConstraint::Accepts(CapsSet::one(video(
            RawVideoFormat::Nv12, Dim::Any, Dim::Any, Rate::Any,
        )));
        let links = solve_linear(&[&src, &dec, &sink]).unwrap();
        assert_eq!(links, vec![
            fixed_compressed(VideoCodec::H264, 1920, 1080, 60),
            fixed_video(RawVideoFormat::Nv12, 1920, 1080, 60),
        ]);
    }

    #[test]
    fn derived_output_couples_downstream_geometry_pin_backward() {
        // The decoder leaves the source's geometry open and the *sink* pins it
        // (1280x720). Before invertible-field discovery the open H264 input link
        // could not fixate (`backward_filter_derived` only drops whole
        // alternatives, never narrows a single one's geometry), so this failed
        // loud. Now the closure is probed: width/height/framerate are passthrough,
        // so the sink's pin couples back and the input fixates to H264 1280x720.
        let src = CapsConstraint::Produces(CapsSet::one(compressed(
            VideoCodec::H264,
            Dim::Any,
            Dim::Any,
            Rate::Fixed(30 << 16),
        )));
        let dec = CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::CompressedVideo { width, height, framerate, .. } => CapsSet::one(Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: width.clone(),
                height: height.clone(),
                framerate: framerate.clone(),
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        }));
        let sink = CapsConstraint::Accepts(CapsSet::one(fixed_video(RawVideoFormat::Nv12, 1280, 720, 30)));
        let links = solve_linear(&[&src, &dec, &sink]).unwrap();
        assert_eq!(links, vec![
            fixed_compressed(VideoCodec::H264, 1280, 720, 30),
            fixed_video(RawVideoFormat::Nv12, 1280, 720, 30),
        ]);
    }

    #[test]
    fn derived_output_fixed_output_imposes_no_backward_narrowing() {
        // A decoder whose output is fixed regardless of input (no passthrough
        // field) must not gain spurious backward coupling: discovery finds NONE,
        // so the input keeps its produced caps. The source pins its own geometry.
        let src = CapsConstraint::Produces(CapsSet::one(fixed_compressed(VideoCodec::H264, 1920, 1080, 30)));
        let dec = CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::CompressedVideo { .. } => CapsSet::one(fixed_video(RawVideoFormat::Nv12, 640, 480, 30)),
            _ => CapsSet::from_alternatives(Vec::new()),
        }));
        let sink = CapsConstraint::Accepts(CapsSet::one(video(
            RawVideoFormat::Nv12,
            Dim::Any,
            Dim::Any,
            Rate::Any,
        )));
        let links = solve_linear(&[&src, &dec, &sink]).unwrap();
        assert_eq!(links, vec![
            fixed_compressed(VideoCodec::H264, 1920, 1080, 30),
            fixed_video(RawVideoFormat::Nv12, 640, 480, 30),
        ]);
    }

    #[test]
    fn mapping_picks_compatible_pair() {
        // Codec converter declaring two pre-enumerated (in, out) pairs.
        // Source is H265 at 1280x720; the H264 pair gets filtered out.
        // Output dims come from the matching pair (mapping doesn't
        // propagate dims between paired sides — that's `DerivedOutput`'s
        // job).
        let src = CapsConstraint::Produces(CapsSet::one(fixed_compressed(VideoCodec::H265, 1280, 720, 30)));
        let map = CapsConstraint::Mapping(vec![
            (
                CapsSet::one(compressed(VideoCodec::H264, Dim::Any, Dim::Any, Rate::Any)),
                CapsSet::one(fixed_video(RawVideoFormat::Nv12, 640, 480, 30)),
            ),
            (
                CapsSet::one(compressed(VideoCodec::H265, Dim::Any, Dim::Any, Rate::Any)),
                CapsSet::one(fixed_video(RawVideoFormat::Nv12, 1280, 720, 30)),
            ),
        ]);
        let sink = CapsConstraint::Accepts(CapsSet::one(video(
            RawVideoFormat::Nv12, Dim::Any, Dim::Any, Rate::Any,
        )));
        let links = solve_linear(&[&src, &map, &sink]).unwrap();
        assert_eq!(links[0], fixed_compressed(VideoCodec::H265, 1280, 720, 30));
        assert_eq!(links[1], fixed_video(RawVideoFormat::Nv12, 1280, 720, 30));
    }

    #[test]
    fn legacy_cascade_source_to_sink() {
        // Source produces 720p NV12; sink accepts anything (returns
        // upstream unchanged from its intercept_caps).
        let src_caps = fixed_video(RawVideoFormat::Nv12, 1280, 720, 30);
        let src = CapsConstraint::LegacySource(src_caps.clone());
        let sink = CapsConstraint::LegacySink(Box::new(|upstream: &Caps| Ok(upstream.clone())));
        let links = solve_linear(&[&src, &sink]).unwrap();
        assert_eq!(links, vec![src_caps]);
    }

    #[test]
    fn legacy_cascade_with_pass_through_transform() {
        let src_caps = fixed_video(RawVideoFormat::Nv12, 1920, 1080, 60);
        let src = CapsConstraint::LegacySource(src_caps.clone());
        let id = CapsConstraint::LegacyTransform {
            intercept: Box::new(|c: &Caps| Ok(c.clone())),
            propose_output: Box::new(|c: &Caps| c.clone()),
        };
        let sink = CapsConstraint::LegacySink(Box::new(|c: &Caps| Ok(c.clone())));
        let links = solve_linear(&[&src, &id, &sink]).unwrap();
        assert_eq!(links, vec![src_caps.clone(), src_caps]);
    }

    #[test]
    fn legacy_cascade_with_boundary_transform() {
        // Decoder: input H264, output NV12 at matching dims.
        let src_caps = fixed_compressed(VideoCodec::H264, 1280, 720, 30);
        let src = CapsConstraint::LegacySource(src_caps.clone());
        let dec = CapsConstraint::LegacyTransform {
            intercept: Box::new(|c: &Caps| Ok(c.clone())),
            propose_output: Box::new(|c: &Caps| match c {
                Caps::CompressedVideo { width, height, framerate, .. } => Caps::RawVideo {
                    format: RawVideoFormat::Nv12,
                    width: width.clone(),
                    height: height.clone(),
                    framerate: framerate.clone(),
                },
                other => other.clone(),
            }),
        };
        let sink = CapsConstraint::LegacySink(Box::new(|c: &Caps| Ok(c.clone())));
        let links = solve_linear(&[&src, &dec, &sink]).unwrap();
        // M16 step 5e: the legacy cascade is intercept-only, mirroring
        // the pre-M16 single-fixated-caps model exactly. The decoder's
        // `propose_output_caps` is ignored on this path because legacy
        // sinks (e.g. waylandsink with workaround #2) depend on
        // receiving the upstream-side caps at `configure_pipeline`
        // and learning the real output dims later via mid-stream
        // `CapsChanged`. Both link slots carry the same fixated caps.
        // Format-changing semantics arrive when an element migrates to
        // a native variant and the chain becomes mixed.
        let h264 = fixed_compressed(VideoCodec::H264, 1280, 720, 30);
        assert_eq!(links, vec![h264.clone(), h264]);
    }

    #[test]
    fn legacy_cascade_intercept_failure_returns_empty_link() {
        let src = CapsConstraint::LegacySource(fixed_compressed(VideoCodec::H264, 1280, 720, 30));
        let sink = CapsConstraint::LegacySink(Box::new(|_: &Caps| Err(crate::error::G2gError::CapsMismatch)));
        assert!(matches!(
            solve_linear(&[&src, &sink]),
            Err(NegotiationFailure::EmptyLink { upstream: 0, downstream: 1 })
        ));
    }

    #[test]
    fn mixed_legacy_source_native_sink() {
        // Migration shape: source still on legacy bridge, sink moved to
        // native Accepts. The mixed cascade fixates link from the
        // source and narrows against the sink's CapsSet.
        let caps = fixed_video(RawVideoFormat::Nv12, 1280, 720, 30);
        let src = CapsConstraint::LegacySource(caps.clone());
        let sink = CapsConstraint::Accepts(CapsSet::one(video(
            RawVideoFormat::Nv12, Dim::Any, Dim::Any, Rate::Any,
        )));
        let links = solve_linear(&[&src, &sink]).unwrap();
        assert_eq!(links, vec![caps]);
    }

    #[test]
    fn mixed_native_source_legacy_sink() {
        // Reverse migration shape: native source, legacy sink.
        let caps = fixed_video(RawVideoFormat::Nv12, 640, 480, 30);
        let src = CapsConstraint::Produces(CapsSet::one(caps.clone()));
        let sink = CapsConstraint::LegacySink(Box::new(|c: &Caps| Ok(c.clone())));
        let links = solve_linear(&[&src, &sink]).unwrap();
        assert_eq!(links, vec![caps]);
    }

    #[test]
    fn mixed_native_source_legacy_transform_native_sink() {
        // Source migrated to native, decoder still on legacy bridge,
        // sink migrated. Exercises forward cascade through a legacy
        // boundary transform between two native endpoints.
        let h264 = fixed_compressed(VideoCodec::H264, 1920, 1080, 60);
        let nv12 = fixed_video(RawVideoFormat::Nv12, 1920, 1080, 60);
        let src = CapsConstraint::Produces(CapsSet::one(h264));
        let dec = CapsConstraint::LegacyTransform {
            intercept: Box::new(|c: &Caps| Ok(c.clone())),
            propose_output: Box::new(|c: &Caps| match c {
                Caps::CompressedVideo { width, height, framerate, .. } => Caps::RawVideo {
                    format: RawVideoFormat::Nv12,
                    width: width.clone(),
                    height: height.clone(),
                    framerate: framerate.clone(),
                },
                other => other.clone(),
            }),
        };
        let sink = CapsConstraint::Accepts(CapsSet::one(video(
            RawVideoFormat::Nv12, Dim::Any, Dim::Any, Rate::Any,
        )));
        let links = solve_linear(&[&src, &dec, &sink]).unwrap();
        assert_eq!(links, vec![
            fixed_compressed(VideoCodec::H264, 1920, 1080, 60),
            nv12,
        ]);
    }

    #[test]
    fn mixed_chain_empty_link_when_sink_rejects() {
        let src = CapsConstraint::LegacySource(fixed_compressed(VideoCodec::H264, 1280, 720, 30));
        let sink = CapsConstraint::Accepts(CapsSet::one(video(
            RawVideoFormat::Nv12, Dim::Any, Dim::Any, Rate::Any,
        )));
        assert!(matches!(
            solve_linear(&[&src, &sink]),
            Err(NegotiationFailure::EmptyLink { .. })
        ));
    }

    #[test]
    fn accepts_any_native_chain_passes_source_caps_through() {
        let caps = fixed_video(RawVideoFormat::Nv12, 1280, 720, 30);
        let src = CapsConstraint::Produces(CapsSet::one(caps.clone()));
        let sink = CapsConstraint::AcceptsAny;
        let links = solve_linear(&[&src, &sink]).unwrap();
        assert_eq!(links, vec![caps]);
    }

    #[test]
    fn accepts_any_mixed_chain_passes_legacy_source_through() {
        // Migration shape: legacy source still on the bridge, sink
        // migrated to AcceptsAny.
        let caps = fixed_compressed(VideoCodec::H264, 1920, 1080, 60);
        let src = CapsConstraint::LegacySource(caps.clone());
        let sink = CapsConstraint::AcceptsAny;
        let links = solve_linear(&[&src, &sink]).unwrap();
        assert_eq!(links, vec![caps]);
    }

    #[test]
    fn accepts_any_in_middle_position_is_silently_a_no_op() {
        // `AcceptsAny` in the middle of a native chain neither narrows
        // its input link nor its output link. The surrounding source
        // and sink fully determine the link assignments — the middle
        // element is invisible to the solver. Forward-cascade paths
        // (mixed/legacy) do reject this via `forward_propagate` because
        // they need an explicit output rule.
        let caps = fixed_video(RawVideoFormat::Nv12, 1, 1, 1);
        let src = CapsConstraint::Produces(CapsSet::one(caps.clone()));
        let mid = CapsConstraint::AcceptsAny;
        let sink = CapsConstraint::Accepts(CapsSet::one(caps.clone()));
        let links = solve_linear(&[&src, &mid, &sink]).unwrap();
        assert_eq!(links, vec![caps.clone(), caps]);
    }

    #[test]
    fn identity_any_couples_native_links() {
        // Fully-native: Produces → IdentityAny → AcceptsAny.
        // The wildcard transform doesn't constrain by any set; it just
        // forces input = output, so both links carry the source's
        // produced caps.
        let caps = fixed_video(RawVideoFormat::Nv12, 1280, 720, 30);
        let src = CapsConstraint::Produces(CapsSet::one(caps.clone()));
        let mid = CapsConstraint::IdentityAny;
        let sink = CapsConstraint::AcceptsAny;
        let links = solve_linear(&[&src, &mid, &sink]).unwrap();
        assert_eq!(links, vec![caps.clone(), caps]);
    }

    #[test]
    fn identity_any_in_mixed_chain_passes_legacy_source_through() {
        let caps = fixed_compressed(VideoCodec::H264, 1920, 1080, 60);
        let src = CapsConstraint::LegacySource(caps.clone());
        let mid = CapsConstraint::IdentityAny;
        let sink = CapsConstraint::AcceptsAny;
        let links = solve_linear(&[&src, &mid, &sink]).unwrap();
        assert_eq!(links, vec![caps.clone(), caps]);
    }

    #[test]
    fn identity_any_endpoint_position_rejected_in_mixed() {
        // IdentityAny is interior-only; using it as a source or sink
        // should fail the endpoint shape check.
        let caps = fixed_video(RawVideoFormat::Nv12, 1, 1, 1);
        let bad_src = CapsConstraint::IdentityAny;
        let sink = CapsConstraint::Accepts(CapsSet::one(caps));
        assert!(matches!(
            solve_linear(&[&bad_src, &sink]),
            Err(NegotiationFailure::EndpointShapeMismatch { index: 0 })
        ));
    }

    #[test]
    fn all_native_produces_to_accepts_any_passes_through() {
        // 5f-style chain: native source (Produces) → AcceptsAny.
        // Confirms the all-native arc-consistency path passes Produces's
        // caps through and the chain returns the source's fixed caps.
        let caps = fixed_video(RawVideoFormat::Rgba8, 1280, 720, 30);
        let src = CapsConstraint::Produces(CapsSet::one(caps.clone()));
        let sink = CapsConstraint::AcceptsAny;
        let links = solve_linear(&[&src, &sink]).unwrap();
        assert_eq!(links, vec![caps]);
    }

    #[test]
    fn mapping_no_surviving_pair_returns_empty_link() {
        let src = CapsConstraint::Produces(CapsSet::one(fixed_compressed(VideoCodec::Av1, 1280, 720, 30)));
        let map = CapsConstraint::Mapping(vec![(
            CapsSet::one(compressed(VideoCodec::H264, Dim::Any, Dim::Any, Rate::Any)),
            CapsSet::one(video(RawVideoFormat::Nv12, Dim::Any, Dim::Any, Rate::Any)),
        )]);
        let sink = CapsConstraint::Accepts(CapsSet::one(video(
            RawVideoFormat::Nv12, Dim::Any, Dim::Any, Rate::Any,
        )));
        assert!(matches!(
            solve_linear(&[&src, &map, &sink]),
            Err(NegotiationFailure::EmptyLink { .. })
        ));
    }

    /// Caps-α downstream feasibility ignores the source: link 0's set is the
    /// pass-through transform's own set narrowed by the sink, independent of
    /// what the source happens to produce, so a mid-stream source change can
    /// be re-fixated against the real downstream capability.
    #[cfg(feature = "std")]
    #[test]
    fn downstream_feasibility_is_source_independent() {
        let src = CapsConstraint::Produces(CapsSet::one(fixed_video(RawVideoFormat::Rgba8, 64, 64, 30)));
        let id = CapsConstraint::IdentityAny;
        let sink = CapsConstraint::Accepts(CapsSet::one(video(
            RawVideoFormat::Nv12, Dim::Any, Dim::Any, Rate::Any,
        )));
        let feas = downstream_feasibility(&[&src, &id, &sink]);
        // Two links. Both carry the sink's NV12 set (IdentityAny couples them);
        // neither is narrowed to the source's RGBA.
        assert_eq!(feas.len(), 2);
        assert!(feas[1].as_ref().unwrap().accepts(&video(
            RawVideoFormat::Nv12, Dim::Any, Dim::Any, Rate::Any,
        )));
        assert!(feas[0].as_ref().unwrap().accepts(&video(
            RawVideoFormat::Nv12, Dim::Any, Dim::Any, Rate::Any,
        )));
        assert!(!feas[0].as_ref().unwrap().accepts(&video(
            RawVideoFormat::Rgba8, Dim::Any, Dim::Any, Rate::Any,
        )));
    }

    /// `resolve_forward_output` steers a format converter toward the one
    /// output its downstream accepts, defers when there is no concrete
    /// downstream set, and rejects loud when no output can satisfy it. A
    /// format-only converter is a `DerivedOutput` so it carries the input's
    /// concrete geometry into its output (a static `Mapping` with `Any` dims
    /// can't fixate and would `Defer`).
    #[test]
    fn resolve_forward_output_steers_defers_and_rejects() {
        // Converter: any raw input -> {same format, NV12} at the input's dims.
        let conv = CapsConstraint::DerivedOutput(Box::new(|input: &Caps| {
            let Caps::RawVideo { format, width, height, framerate } = input else {
                return CapsSet::from_alternatives(vec![]);
            };
            CapsSet::from_alternatives(vec![
                video(*format, width.clone(), height.clone(), framerate.clone()),
                video(RawVideoFormat::Nv12, width.clone(), height.clone(), framerate.clone()),
            ])
        }));
        let i420 = fixed_video(RawVideoFormat::I420, 64, 64, 30);
        let nv12_set = CapsSet::one(video(RawVideoFormat::Nv12, Dim::Any, Dim::Any, Rate::Any));

        // Steered: downstream accepts only NV12, so the runner picks NV12.
        match resolve_forward_output(&conv, &i420, Some(&nv12_set)) {
            ForwardResolve::Fixed(c) => {
                assert_eq!(c, video(RawVideoFormat::Nv12, Dim::Fixed(64), Dim::Fixed(64), Rate::Fixed(30 << 16)));
            }
            other => panic!("expected Fixed(NV12), got {other:?}"),
        }

        // No concrete downstream set, but the output is ambiguous ({same, NV12}):
        // defer to the element's own process.
        assert_eq!(resolve_forward_output(&conv, &i420, None), ForwardResolve::Defer);

        // No downstream snapshot but an UNAMBIGUOUS output: a property-driven
        // converter forwards its single output (RGBA8) rather than leaking the
        // input format. This is what lets a strict downstream (a textoverlay
        // after `mp4src ! avdec ! videoconvert`) see the converted caps on a
        // mid-stream change instead of the decoder's NV12.
        let to_rgba = CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::RawVideo { width, height, framerate, .. } => CapsSet::one(video(
                RawVideoFormat::Rgba8,
                width.clone(),
                height.clone(),
                framerate.clone(),
            )),
            _ => CapsSet::from_alternatives(vec![]),
        }));
        let nv12_in = fixed_video(RawVideoFormat::Nv12, 64, 64, 30);
        match resolve_forward_output(&to_rgba, &nv12_in, None) {
            ForwardResolve::Fixed(c) => assert_eq!(
                c,
                video(RawVideoFormat::Rgba8, Dim::Fixed(64), Dim::Fixed(64), Rate::Fixed(30 << 16))
            ),
            other => panic!("expected Fixed(RGBA8), got {other:?}"),
        }

        // Downstream accepts only Bgra8, which the converter cannot emit: loud.
        let bgra_set = CapsSet::one(video(RawVideoFormat::Bgra8, Dim::Any, Dim::Any, Rate::Any));
        assert!(matches!(
            resolve_forward_output(&conv, &i420, Some(&bgra_set)),
            ForwardResolve::Infeasible(NegotiationFailure::EmptyLink { .. })
        ));
    }

    use crate::graph::Graph;

    #[test]
    fn solve_graph_matches_solve_linear_on_a_chain() {
        // source Produces fixed RGBA -> DerivedOutput RGBA->NV12 -> Accepts NV12.
        let rgba = fixed_video(RawVideoFormat::Rgba8, 64, 48, 30);
        let nv12 = fixed_video(RawVideoFormat::Nv12, 64, 48, 30);
        let lin: Vec<CapsConstraint> = vec![
            CapsConstraint::Produces(CapsSet::one(rgba.clone())),
            CapsConstraint::DerivedOutput(Box::new({
                let nv12 = nv12.clone();
                move |_input: &Caps| CapsSet::one(nv12.clone())
            })),
            CapsConstraint::Accepts(CapsSet::one(nv12.clone())),
        ];
        let refs: Vec<&CapsConstraint> = lin.iter().collect();
        let linear = solve_linear(&refs).expect("linear chain solves");

        // the same chain expressed as a graph (constraints rebuilt identically).
        let dag_cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(rgba.clone()))),
            NodeConstraint::Element(CapsConstraint::DerivedOutput(Box::new({
                let nv12 = nv12.clone();
                move |_input: &Caps| CapsSet::one(nv12.clone())
            }))),
            NodeConstraint::Element(CapsConstraint::Accepts(CapsSet::one(nv12.clone()))),
        ];
        let mut g: Graph<()> = Graph::new();
        let src = g.add_source(());
        let tx = g.add_transform(());
        let sink = g.add_sink(());
        g.link(src, tx).unwrap();
        g.link(tx, sink).unwrap();
        let v = g.finish().unwrap();
        let dag = solve_graph(&v, &dag_cs).expect("same chain as a graph solves");

        assert_eq!(dag, linear, "DAG solver matches the linear solver byte-for-byte");
        assert_eq!(dag, vec![rgba, nv12]);
    }

    #[test]
    fn solve_graph_tee_fanout_couples_branches() {
        let nv12_fixed = fixed_video(RawVideoFormat::Nv12, 64, 48, 30);
        let nv12_any = video(RawVideoFormat::Nv12, Dim::Any, Dim::Any, Rate::Any);
        // source (node 0) -> tee (1) -> two NV12 sinks (2, 3).
        let cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(nv12_fixed.clone()))),
            NodeConstraint::Element(CapsConstraint::IdentityAny), // tee slot, ignored
            NodeConstraint::Element(CapsConstraint::Accepts(CapsSet::one(nv12_any.clone()))),
            NodeConstraint::Element(CapsConstraint::Accepts(CapsSet::one(nv12_any))),
        ];
        let mut g: Graph<()> = Graph::new();
        let src = g.add_source(());
        let tee = g.add_tee(2);
        let a = g.add_sink(());
        let b = g.add_sink(());
        g.link(src, tee.input()).unwrap();
        g.link(tee.out(0), a).unwrap();
        g.link(tee.out(1), b).unwrap();
        let v = g.finish().unwrap();

        let sol = solve_graph(&v, &cs).expect("tee fan-out solves");
        assert_eq!(sol.len(), 3, "three edges");
        assert!(sol.iter().all(|c| *c == nv12_fixed), "every branch carries the source caps");
    }

    #[test]
    fn solve_graph_rejects_incompatible_branch() {
        let nv12 = fixed_video(RawVideoFormat::Nv12, 64, 48, 30);
        let nv12_any = video(RawVideoFormat::Nv12, Dim::Any, Dim::Any, Rate::Any);
        let rgba_any = video(RawVideoFormat::Rgba8, Dim::Any, Dim::Any, Rate::Any);
        // one branch accepts NV12, the other only RGBA: strict whole-graph fail.
        let cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(nv12))),
            NodeConstraint::Element(CapsConstraint::IdentityAny),
            NodeConstraint::Element(CapsConstraint::Accepts(CapsSet::one(nv12_any))),
            NodeConstraint::Element(CapsConstraint::Accepts(CapsSet::one(rgba_any))),
        ];
        let mut g: Graph<()> = Graph::new();
        let src = g.add_source(());
        let tee = g.add_tee(2);
        let a = g.add_sink(());
        let b = g.add_sink(());
        g.link(src, tee.input()).unwrap();
        g.link(tee.out(0), a).unwrap();
        g.link(tee.out(1), b).unwrap();
        let v = g.finish().unwrap();

        assert!(
            matches!(solve_graph(&v, &cs), Err(NegotiationFailure::EmptyLink { .. })),
            "an incompatible branch fails the whole solve"
        );
    }

    #[test]
    fn solve_graph_diamond_fixates_globally_consistent() {
        // True diamond: source {V,W} -> tee -> two Mapping branches -> muxer.
        // Branch 1 maps V->A, W->C; branch 2 maps W->B, V->D (orders misaligned).
        // The muxer accepts {A,C} on pad 0 and {B,D} on pad 1. Valid solutions
        // exist (source=V => b1=A, b2=D; or source=W => b1=C, b2=B), but greedy
        // per-edge fixation picks source=V, b1=A (first of {A,C}), b2=B (first of
        // {B,D}) -- and (V, B) is not a branch-2 mapping pair. Arc consistency
        // can't catch this; the backtracking fixation must.
        let v = fixed_compressed(VideoCodec::H264, 64, 48, 30);
        let w = fixed_compressed(VideoCodec::H265, 64, 48, 30);
        let a = fixed_video(RawVideoFormat::Nv12, 64, 48, 30);
        let c = fixed_video(RawVideoFormat::I420, 64, 48, 30);
        let b = fixed_video(RawVideoFormat::Rgba8, 64, 48, 30);
        let d = fixed_video(RawVideoFormat::I422, 64, 48, 30);
        let muxed = fixed_video(RawVideoFormat::I444, 64, 48, 30);

        let cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::from_alternatives(vec![
                v.clone(),
                w.clone(),
            ]))),
            NodeConstraint::Element(CapsConstraint::IdentityAny), // tee
            NodeConstraint::Element(CapsConstraint::Mapping(vec![
                (CapsSet::one(v.clone()), CapsSet::one(a.clone())),
                (CapsSet::one(w.clone()), CapsSet::one(c.clone())),
            ])),
            NodeConstraint::Element(CapsConstraint::Mapping(vec![
                (CapsSet::one(w.clone()), CapsSet::one(b.clone())),
                (CapsSet::one(v.clone()), CapsSet::one(d.clone())),
            ])),
            NodeConstraint::Muxer {
                inputs: vec![
                    CapsConstraint::Accepts(CapsSet::from_alternatives(vec![a.clone(), c.clone()])),
                    CapsConstraint::Accepts(CapsSet::from_alternatives(vec![b.clone(), d.clone()])),
                ],
                output: CapsConstraint::Produces(CapsSet::one(muxed)),
                follows: None,
            },
            NodeConstraint::Element(CapsConstraint::AcceptsAny),
        ];
        let mut g: Graph<()> = Graph::new();
        let src = g.add_source(());
        let tee = g.add_tee(2);
        let b1 = g.add_transform(());
        let b2 = g.add_transform(());
        let mux = g.add_muxer((), 2);
        let sink = g.add_sink(());
        g.link(src, tee.input()).unwrap();
        g.link(tee.out(0), b1).unwrap();
        g.link(tee.out(1), b2).unwrap();
        g.link(b1, mux.input(0)).unwrap();
        g.link(b2, mux.input(1)).unwrap();
        g.link(mux.output(), sink).unwrap();
        let vg = g.finish().unwrap();

        let sol = solve_graph(&vg, &cs).expect("diamond has a satisfying assignment");
        // Edges: 0 src->tee, 1 tee->b1, 2 tee->b2, 3 b1->mux, 4 b2->mux, 5 mux->sink.
        // The tee broadcasts one value, so both branch inputs equal the source.
        assert_eq!(sol[1], sol[0], "tee broadcasts to branch 1");
        assert_eq!(sol[2], sol[0], "tee broadcasts to branch 2");
        // Each branch's (in, out) must be one of its declared mapping pairs.
        let b1_pair = (sol[1].clone(), sol[3].clone());
        assert!(
            b1_pair == (v.clone(), a.clone()) || b1_pair == (w.clone(), c.clone()),
            "branch 1 fixated to a real mapping pair, got {b1_pair:?}"
        );
        let b2_pair = (sol[2].clone(), sol[4].clone());
        assert!(
            b2_pair == (w.clone(), b.clone()) || b2_pair == (v.clone(), d.clone()),
            "branch 2 fixated to a real mapping pair, got {b2_pair:?}"
        );
    }

    #[test]
    fn solve_graph_muxer_fan_in_narrows_each_input() {
        // two video sources combine at a muxer: input pad 0 accepts H264,
        // pad 1 accepts H265, the output produces a (token) muxed stream.
        let h264 = compressed(VideoCodec::H264, Dim::Fixed(64), Dim::Fixed(48), Rate::Fixed(30 << 16));
        let h265 = compressed(VideoCodec::H265, Dim::Fixed(64), Dim::Fixed(48), Rate::Fixed(30 << 16));
        let h264_any = compressed(VideoCodec::H264, Dim::Any, Dim::Any, Rate::Any);
        let h265_any = compressed(VideoCodec::H265, Dim::Any, Dim::Any, Rate::Any);
        let muxed = compressed(VideoCodec::H264, Dim::Fixed(64), Dim::Fixed(48), Rate::Fixed(30 << 16));

        let cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(h264.clone()))),
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(h265.clone()))),
            NodeConstraint::Muxer {
                inputs: vec![
                    CapsConstraint::Accepts(CapsSet::one(h264_any)),
                    CapsConstraint::Accepts(CapsSet::one(h265_any)),
                ],
                output: CapsConstraint::Produces(CapsSet::one(muxed.clone())),
                follows: None,
            },
            NodeConstraint::Element(CapsConstraint::AcceptsAny),
        ];
        let mut g: Graph<()> = Graph::new();
        let s0 = g.add_source(());
        let s1 = g.add_source(());
        let mux = g.add_muxer((), 2);
        let sink = g.add_sink(());
        g.link(s0, mux.input(0)).unwrap();
        g.link(s1, mux.input(1)).unwrap();
        g.link(mux.output(), sink).unwrap();
        let v = g.finish().unwrap();

        let sol = solve_graph(&v, &cs).expect("muxer fan-in solves");
        // edges in id order: s0->in0, s1->in1, mux.out->sink.
        assert_eq!(sol, vec![h264, h265, muxed], "each input narrowed by its pad, output by produce");
    }

    #[test]
    fn solve_graph_muxer_follows_input_derives_output() {
        // An identity-passthrough mux (overlay): a video pad 0, a sidecar pad 1,
        // output follows pad 0. The output edge must equal the video source's caps
        // even though no output caps were declared (`output` is a placeholder).
        let rgba = fixed_video(RawVideoFormat::Rgba8, 320, 240, 30);
        let rgba_any = video(RawVideoFormat::Rgba8, Dim::Any, Dim::Any, Rate::Any);
        let text = Caps::Text { format: crate::caps::TextFormat::Utf8 };

        let cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(rgba.clone()))),
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(text.clone()))),
            NodeConstraint::Muxer {
                inputs: vec![
                    CapsConstraint::Accepts(CapsSet::one(rgba_any)),
                    CapsConstraint::Accepts(CapsSet::one(text.clone())),
                ],
                // Placeholder: ignored because `follows` is set.
                output: CapsConstraint::AcceptsAny,
                follows: Some(0),
            },
            NodeConstraint::Element(CapsConstraint::AcceptsAny),
        ];
        let mut g: Graph<()> = Graph::new();
        let video_src = g.add_source(());
        let text_src = g.add_source(());
        let mux = g.add_muxer((), 2);
        let sink = g.add_sink(());
        g.link(video_src, mux.input(0)).unwrap();
        g.link(text_src, mux.input(1)).unwrap();
        g.link(mux.output(), sink).unwrap();
        let v = g.finish().unwrap();

        let sol = solve_graph(&v, &cs).expect("follows-input muxer solves");
        // edges: 0 video->in0, 1 text->in1, 2 mux.out->sink.
        assert_eq!(sol[2], rgba, "output edge follows the video pad's negotiated caps");
        assert_eq!(sol[0], rgba, "video pad edge unchanged");
        assert_eq!(sol[1], text, "text pad edge unchanged");
    }

    #[test]
    fn solve_graph_muxer_wildcard_inputs_forward_source_caps() {
        // The `InterleaveMux` shape: every input pad is `AcceptsAny` (frames
        // carry their own caps), the output `Produces` a fixed merged caps. The
        // wildcard inputs impose no narrowing, so each input edge keeps its
        // source's caps and the output edge takes the produced caps.
        let h264 = compressed(VideoCodec::H264, Dim::Fixed(64), Dim::Fixed(48), Rate::Fixed(30 << 16));
        let aac = Caps::Audio { format: crate::caps::AudioFormat::Aac, channels: 2, sample_rate: 48_000 };
        let merged = compressed(VideoCodec::H264, Dim::Fixed(64), Dim::Fixed(48), Rate::Fixed(30 << 16));

        let cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(h264.clone()))),
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(aac.clone()))),
            NodeConstraint::Muxer {
                inputs: vec![CapsConstraint::AcceptsAny, CapsConstraint::AcceptsAny],
                output: CapsConstraint::Produces(CapsSet::one(merged.clone())),
                follows: None,
            },
            NodeConstraint::Element(CapsConstraint::AcceptsAny),
        ];
        let mut g: Graph<()> = Graph::new();
        let s0 = g.add_source(());
        let s1 = g.add_source(());
        let mux = g.add_muxer((), 2);
        let sink = g.add_sink(());
        g.link(s0, mux.input(0)).unwrap();
        g.link(s1, mux.input(1)).unwrap();
        g.link(mux.output(), sink).unwrap();
        let v = g.finish().unwrap();

        let sol = solve_graph(&v, &cs).expect("wildcard muxer solves");
        // a wildcard input pad leaves each input edge at its own source caps.
        assert_eq!(sol, vec![h264, aac, merged]);
    }

    #[test]
    fn solve_graph_accepts_legacy_bridge_constraints() {
        // A native source/muxer feeding a `LegacySink` (the default sink bridge,
        // e.g. m10's CollectingSink). The legacy sink imposes no narrowing, so
        // the merged output flows through unchanged. Previously this hit
        // EndpointShapeMismatch; now `run_muxer_sink` can build it as a graph.
        let h264 = fixed_compressed(VideoCodec::H264, 64, 48, 30);
        let cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::LegacySource(h264.clone())),
            NodeConstraint::Element(CapsConstraint::LegacySource(h264.clone())),
            NodeConstraint::Muxer {
                inputs: vec![CapsConstraint::AcceptsAny, CapsConstraint::AcceptsAny],
                output: CapsConstraint::Produces(CapsSet::one(h264.clone())),
                follows: None,
            },
            NodeConstraint::Element(CapsConstraint::LegacySink(Box::new(|c: &Caps| Ok(c.clone())))),
        ];
        let mut g: Graph<()> = Graph::new();
        let s0 = g.add_source(());
        let s1 = g.add_source(());
        let mux = g.add_muxer((), 2);
        let sink = g.add_sink(());
        g.link(s0, mux.input(0)).unwrap();
        g.link(s1, mux.input(1)).unwrap();
        g.link(mux.output(), sink).unwrap();
        let v = g.finish().unwrap();

        let sol = solve_graph(&v, &cs).expect("native muxer + legacy sink solves");
        assert_eq!(sol, vec![h264.clone(), h264.clone(), h264]);
    }

    #[test]
    fn solve_graph_forwards_legacy_transform() {
        // src(LegacySource RGBA) -> LegacyTransform(RGBA->NV12) -> Accepts NV12.
        let rgba = fixed_video(RawVideoFormat::Rgba8, 64, 48, 30);
        let nv12 = fixed_video(RawVideoFormat::Nv12, 64, 48, 30);
        let cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::LegacySource(rgba.clone())),
            NodeConstraint::Element(CapsConstraint::LegacyTransform {
                intercept: Box::new({
                    let nv12 = nv12.clone();
                    move |_in: &Caps| Ok(nv12.clone())
                }),
                propose_output: Box::new(|c: &Caps| c.clone()),
            }),
            NodeConstraint::Element(CapsConstraint::Accepts(CapsSet::one(nv12.clone()))),
        ];
        let mut g: Graph<()> = Graph::new();
        let src = g.add_source(());
        let tx = g.add_transform(());
        let sink = g.add_sink(());
        g.link(src, tx).unwrap();
        g.link(tx, sink).unwrap();
        let v = g.finish().unwrap();

        let sol = solve_graph(&v, &cs).expect("legacy transform forwards");
        assert_eq!(sol, vec![rgba, nv12]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn graph_feasibility_intersects_tee_branches() {
        // src -> tee(2) -> {accepts NV12-any, accepts NV12 64x48}. The tee input
        // edge's feasibility is the intersection: the tighter 64x48 set.
        let nv12_any = video(RawVideoFormat::Nv12, Dim::Any, Dim::Any, Rate::Any);
        let nv12_fixed = fixed_video(RawVideoFormat::Nv12, 64, 48, 30);
        let cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(nv12_fixed.clone()))),
            NodeConstraint::Element(CapsConstraint::IdentityAny),
            NodeConstraint::Element(CapsConstraint::Accepts(CapsSet::one(nv12_any))),
            NodeConstraint::Element(CapsConstraint::Accepts(CapsSet::one(nv12_fixed.clone()))),
        ];
        let mut g: Graph<()> = Graph::new();
        let src = g.add_source(());
        let tee = g.add_tee(2);
        let a = g.add_sink(());
        let b = g.add_sink(());
        g.link(src, tee.input()).unwrap();
        g.link(tee.out(0), a).unwrap();
        g.link(tee.out(1), b).unwrap();
        let v = g.finish().unwrap();

        let sol = solve_graph(&v, &cs).expect("tee branches solve");
        let feas = graph_downstream_feasibility(&v, &cs, &sol);
        // edge 0 = src->tee, edge 1 = tee.out0->a, edge 2 = tee.out1->b.
        let tee_in = feas[0].as_ref().expect("tee input has feasibility");
        assert!(tee_in.intersect(&CapsSet::one(nv12_fixed.clone())).fixate().is_some());
        // the tee input cannot carry an off-geometry frame both branches reject.
        let off = fixed_video(RawVideoFormat::Nv12, 99, 99, 30);
        assert!(tee_in.intersect(&CapsSet::one(off)).is_empty(), "branch B pins 64x48");
    }

    #[cfg(feature = "std")]
    #[test]
    fn graph_feasibility_muxer_inputs_are_per_pad() {
        // two sources -> muxer{H264, H265} -> wildcard sink. Each input edge's
        // feasibility is its own pad accept set; the output edge is unconstrained
        // (the wildcard sink imposes nothing, and the output never feeds inputs).
        let h264_any = compressed(VideoCodec::H264, Dim::Any, Dim::Any, Rate::Any);
        let h265_any = compressed(VideoCodec::H265, Dim::Any, Dim::Any, Rate::Any);
        let cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(
                fixed_compressed(VideoCodec::H264, 64, 48, 30),
            ))),
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(
                fixed_compressed(VideoCodec::H265, 64, 48, 30),
            ))),
            NodeConstraint::Muxer {
                inputs: vec![
                    CapsConstraint::Accepts(CapsSet::one(h264_any.clone())),
                    CapsConstraint::Accepts(CapsSet::one(h265_any.clone())),
                ],
                output: CapsConstraint::Produces(CapsSet::one(fixed_compressed(
                    VideoCodec::H264, 64, 48, 30,
                ))),
                follows: None,
            },
            NodeConstraint::Element(CapsConstraint::AcceptsAny),
        ];
        let mut g: Graph<()> = Graph::new();
        let s0 = g.add_source(());
        let s1 = g.add_source(());
        let mux = g.add_muxer((), 2);
        let sink = g.add_sink(());
        g.link(s0, mux.input(0)).unwrap();
        g.link(s1, mux.input(1)).unwrap();
        g.link(mux.output(), sink).unwrap();
        let v = g.finish().unwrap();

        let sol = solve_graph(&v, &cs).expect("muxer graph solves");
        let feas = graph_downstream_feasibility(&v, &cs, &sol);
        // edges: 0 = s0->in0, 1 = s1->in1, 2 = mux.out->sink.
        assert_eq!(feas[0], Some(CapsSet::one(h264_any)), "pad 0 feasibility = its accept set");
        assert_eq!(feas[1], Some(CapsSet::one(h265_any)), "pad 1 feasibility = its accept set");
        assert_eq!(feas[2], None, "wildcard sink leaves the muxer output unconstrained");
    }

    #[cfg(feature = "std")]
    #[test]
    fn graph_feasibility_couples_pin_back_through_a_decoder() {
        // M258: src(H264, open geometry) -> decoder(DerivedOutput, geometry
        // passthrough) -> sink pinned to Nv12 1280x720. The decoder's INPUT edge
        // snapshot used to be `None` (a plain `DerivedOutput` had no input to probe
        // mid-stream), so a mid-stream re-solve couldn't steer the source back to
        // the pinned geometry. With the startup-fixated input threaded in, the
        // discovered passthrough fields couple the pin onto the H264 input edge.
        let dec_closure = |input: &Caps| match input {
            Caps::CompressedVideo { width, height, framerate, .. } => CapsSet::one(Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: width.clone(),
                height: height.clone(),
                framerate: framerate.clone(),
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        };
        let cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(compressed(
                VideoCodec::H264, Dim::Any, Dim::Any, Rate::Fixed(30 << 16),
            )))),
            NodeConstraint::Element(CapsConstraint::DerivedOutput(Box::new(dec_closure))),
            NodeConstraint::Element(CapsConstraint::Accepts(CapsSet::one(fixed_video(
                RawVideoFormat::Nv12, 1280, 720, 30,
            )))),
        ];
        let mut g: Graph<()> = Graph::new();
        let src = g.add_source(());
        let dec = g.add_transform(());
        let sink = g.add_sink(());
        g.link(src, dec).unwrap();
        g.link(dec, sink).unwrap();
        let v = g.finish().unwrap();

        let sol = solve_graph(&v, &cs).expect("decoder graph solves");
        let feas = graph_downstream_feasibility(&v, &cs, &sol);
        // edge 0 = src->dec (the decoder input), edge 1 = dec->sink.
        let dec_in = feas[0].as_ref().expect("decoder input edge is now constrained");
        assert!(
            dec_in.intersect(&CapsSet::one(fixed_compressed(VideoCodec::H264, 1280, 720, 30))).fixate().is_some(),
            "pinned 1280x720 couples back onto the H264 input edge",
        );
        let off = fixed_compressed(VideoCodec::H264, 640, 480, 30);
        assert!(dec_in.intersect(&CapsSet::one(off)).is_empty(), "off-geometry input is rejected by the snapshot");
    }

    // --- M227 field-level bidirectional caps coupling ---

    /// A scale-like `DerivedCoupled`: passthrough format + framerate, retarget
    /// geometry to [passthrough-input, Range 1..32768].
    fn scale_like<'a>() -> CapsConstraint<'a> {
        CapsConstraint::DerivedCoupled {
            derive: Box::new(|input: &Caps| match input {
                Caps::RawVideo { format, width, height, framerate } => {
                    CapsSet::from_alternatives(vec![
                        video(*format, width.clone(), height.clone(), framerate.clone()),
                        video(
                            *format,
                            Dim::Range { min: 1, max: 32768 },
                            Dim::Range { min: 1, max: 32768 },
                            framerate.clone(),
                        ),
                    ])
                }
                _ => CapsSet::from_alternatives(vec![]),
            }),
            passthrough: PassthroughFields::NONE.with_format().with_framerate(),
        }
    }

    /// A convert-like `DerivedCoupled`: passthrough geometry + framerate,
    /// retarget format to [Rgba8, Nv12] (Rgba8 preferred).
    fn convert_like<'a>() -> CapsConstraint<'a> {
        CapsConstraint::DerivedCoupled {
            derive: Box::new(|input: &Caps| match input {
                Caps::RawVideo { width, height, framerate, .. } => {
                    CapsSet::from_alternatives(vec![
                        video(RawVideoFormat::Rgba8, width.clone(), height.clone(), framerate.clone()),
                        video(RawVideoFormat::Nv12, width.clone(), height.clone(), framerate.clone()),
                    ])
                }
                _ => CapsSet::from_alternatives(vec![]),
            }),
            passthrough: PassthroughFields::NONE.with_width().with_height().with_framerate(),
        }
    }

    #[test]
    fn couple_passthrough_narrows_a_range_field_within_an_alternative() {
        // The primitive the alternative-drop walk can't express: a Range width
        // meeting a Fixed pin collapses to Fixed, format (retargeted) untouched.
        let mask = PassthroughFields::NONE.with_width().with_height().with_framerate();
        let input = video(
            RawVideoFormat::Rgba8,
            Dim::Range { min: 1, max: 32768 },
            Dim::Range { min: 1, max: 32768 },
            Rate::Fixed(30 << 16),
        );
        let pin = fixed_video(RawVideoFormat::Nv12, 160, 120, 30);
        let coupled = couple_passthrough(&input, &pin, mask).unwrap();
        assert_eq!(
            coupled,
            fixed_video(RawVideoFormat::Rgba8, 160, 120, 30),
            "passthrough width/height/framerate pinned, retargeted format kept"
        );
    }

    #[test]
    fn couple_passthrough_rejects_conflicting_passthrough_field() {
        // A passthrough format that disagrees with the pin kills the alternative.
        let mask = PassthroughFields::NONE.with_format();
        let input = fixed_video(RawVideoFormat::Rgba8, 160, 120, 30);
        let pin = fixed_video(RawVideoFormat::Nv12, 160, 120, 30);
        assert_eq!(couple_passthrough(&input, &pin, mask), None);
    }

    #[test]
    fn field_coupling_resolves_scale_then_convert() {
        // The M188 KNOWN-LIMIT, now resolved: a 160x120 geometry pin sits behind
        // the geometry-passthrough convert; coupling intersects it into the
        // scaler's output field instead of dropping whole alternatives.
        let src =
            CapsConstraint::Produces(CapsSet::one(fixed_video(RawVideoFormat::Rgba8, 320, 240, 30)));
        let scale = scale_like();
        let convert = convert_like();
        let sink =
            CapsConstraint::Accepts(CapsSet::one(fixed_video(RawVideoFormat::Nv12, 160, 120, 30)));
        let links = solve_linear(&[&src, &scale, &convert, &sink]).unwrap();
        assert_eq!(
            links,
            vec![
                fixed_video(RawVideoFormat::Rgba8, 320, 240, 30),
                fixed_video(RawVideoFormat::Rgba8, 160, 120, 30),
                fixed_video(RawVideoFormat::Nv12, 160, 120, 30),
            ],
            "scaler reads 320x240, emits 160x120; convert changes only the format"
        );
    }

    #[test]
    fn field_coupling_no_pin_stays_passthrough() {
        // No downstream pin (AcceptsAny): both transforms prefer their first
        // (passthrough) alternative, and the solve converges (no oscillation).
        let src =
            CapsConstraint::Produces(CapsSet::one(fixed_video(RawVideoFormat::Rgba8, 320, 240, 30)));
        let scale = scale_like();
        let convert = convert_like();
        let sink = CapsConstraint::AcceptsAny;
        let links = solve_linear(&[&src, &scale, &convert, &sink]).unwrap();
        assert_eq!(
            links,
            vec![
                fixed_video(RawVideoFormat::Rgba8, 320, 240, 30),
                fixed_video(RawVideoFormat::Rgba8, 320, 240, 30),
                fixed_video(RawVideoFormat::Rgba8, 320, 240, 30),
            ],
            "passthrough is preferred and stable"
        );
    }

    #[test]
    fn field_coupling_unsatisfiable_geometry_fails_loud() {
        // No scaler upstream: convert passes geometry through, so a 160x120 pin
        // against a fixed 320x240 source has no solution. Loud, never silent.
        let src =
            CapsConstraint::Produces(CapsSet::one(fixed_video(RawVideoFormat::Rgba8, 320, 240, 30)));
        let convert = convert_like();
        let sink =
            CapsConstraint::Accepts(CapsSet::one(fixed_video(RawVideoFormat::Nv12, 160, 120, 30)));
        assert!(solve_linear(&[&src, &convert, &sink]).is_err(), "geometry pin must fail loud");
    }

    // --- caps-negotiation explainer (M280) -------------------------------

    #[test]
    fn explainer_formats_sets_constraints_and_labels() {
        let rgba = fixed_video(RawVideoFormat::Rgba8, 64, 48, 30);
        let nv12 = fixed_video(RawVideoFormat::Nv12, 64, 48, 30);

        // A set renders its alternatives joined by " | "; empty is the ∅ glyph.
        let set = CapsSet::from_alternatives(vec![rgba.clone(), nv12.clone()]);
        let rendered = fmt_set(&set);
        assert!(rendered.contains("format=RGBA") && rendered.contains("format=NV12"));
        assert!(rendered.contains(" | "));
        assert_eq!(fmt_set(&CapsSet::from_alternatives(vec![])), "∅");

        // Wide sets elide past four alternatives so the line stays readable.
        let wide = CapsSet::from_alternatives(
            (1..=6).map(|w| fixed_video(RawVideoFormat::Rgba8, w * 16, 48, 30)).collect(),
        );
        assert!(fmt_set(&wide).contains("(+2 more)"), "{}", fmt_set(&wide));

        // Constraint summaries name the shape.
        assert!(fmt_caps_constraint(&CapsConstraint::Produces(CapsSet::one(rgba.clone())))
            .starts_with("produces "));
        assert_eq!(fmt_caps_constraint(&CapsConstraint::AcceptsAny), "accepts ANY");
        assert_eq!(
            fmt_caps_constraint(&CapsConstraint::DerivedOutput(Box::new(move |_: &Caps| {
                CapsSet::one(nv12.clone())
            }))),
            "derives output"
        );
    }

    #[test]
    fn solve_graph_labeled_matches_default_and_uses_labels() {
        // The labeled solver returns the identical solution; the label closure
        // only affects log text, which a label override exercises here.
        let rgba = fixed_video(RawVideoFormat::Rgba8, 64, 48, 30);
        let cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(rgba.clone()))),
            NodeConstraint::Element(CapsConstraint::AcceptsAny),
        ];
        let mut g: Graph<()> = Graph::new();
        let src = g.add_source(());
        let sink = g.add_sink(());
        g.link(src, sink).unwrap();
        let v = g.finish().unwrap();

        let default = solve_graph(&v, &cs).expect("solves");
        let labeled = solve_graph_labeled(&v, &cs, &|n| alloc::format!("node{}", n.0))
            .expect("solves with custom labels");
        assert_eq!(default, labeled);
        assert_eq!(labeled, vec![rgba]);
    }
}
