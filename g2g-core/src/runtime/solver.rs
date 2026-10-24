//! Linear-pipeline caps solver (design/caps.md).
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

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::caps::{
    batch_alternatives, conflicting_fields, Caps, CapsSet, FieldConflict, PassthroughFields,
};
#[cfg(feature = "std")]
use crate::caps_transform::CapsTransform;
use crate::format_element::{CapsConstraint, CapsPreferences};
use crate::graph::{NodeId, NodeKind, ValidatedGraph};
use crate::log::{self, LogLevel, Target, CAPS_CATEGORY};
use crate::runtime::passthrough::{couple_passthrough_derived, discover_passthrough};
#[cfg(feature = "std")]
use crate::runtime::passthrough::{project_passthrough, project_passthrough_derived};

/// Per-link assignment produced by the solver: one fixated `Caps` per
/// link between adjacent elements. For an `N`-element pipeline this is
/// length `N - 1`. The runner calls `configure_link` on element `i`
/// with `input = links[i - 1]` and `output = links[i]` (sources receive
/// `None` on input; sinks receive `None` on output).
pub type LinkSolution = Vec<Caps>;

/// The two candidate sets an [`NegotiationFailure::EmptyLink`] failed on: what
/// each end of the link still allowed when their intersection came out empty,
/// so tooling can report *what* disagreed and not just which nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct CapsConflict {
    /// What the upstream end offered.
    pub upstream: CapsSet,
    /// What the downstream end demanded.
    pub downstream: CapsSet,
}

impl CapsConflict {
    pub fn conflicting_fields(&self) -> Vec<FieldConflict> {
        conflicting_fields(&self.upstream, &self.downstream)
    }
}

/// Structured solver failure (design/caps.md).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum NegotiationFailure {
    /// Adjacent elements have no overlap on the link between them, or a
    /// constraint update emptied that link.
    EmptyLink {
        upstream: usize,
        downstream: usize,
        /// The two sides' surviving sets, when the solver had both in hand
        /// (boxed: only the cold failure path carries it).
        conflict: Option<Box<CapsConflict>>,
    },
    /// The constraint list has fewer than two elements; nothing to
    /// negotiate.
    Degenerate,
    /// First element does not produce (or last does not accept), so the
    /// chain has no source or no sink.
    EndpointShapeMismatch { index: usize },
    /// A link's candidate set survived narrowing but cannot be reduced
    /// to a single `Caps` (every alternative still has `Any` fields).
    Unfixable { upstream: usize, downstream: usize },
    /// Reserved for the non-linear solver. The linear solver never
    /// returns this variant.
    Cyclic,
    /// Arc consistency left a non-empty domain on every edge, but no single
    /// assignment of one fixated `Caps` per edge satisfies every node at once:
    /// an over-constrained diamond (a tee whose branches re-converge at a fan-in
    /// with no jointly-valid choice). Only the DAG solver's backtracking
    /// fixation returns this.
    NoConsistentFixation,
    /// `upstream` is a transform whose derived output (`DerivedOutput` /
    /// `DerivedFields`) is empty for the caps on its input link, `input`, so the
    /// link to `downstream` has nothing to carry: an encoder fed a pixel format
    /// it does not take. Reported by the DAG solver.
    NoOutputForInput {
        upstream: usize,
        downstream: usize,
        input: CapsSet,
    },
    /// The batch demux `index` would accept more than
    /// [`MAX_BATCH_ALTERNATIVES`](crate::caps::MAX_BATCH_ALTERNATIVES)
    /// combinations of its streams' caps. A capsfilter on each stream narrows
    /// them.
    TooManyBatchAlternatives { index: usize },
}

impl NegotiationFailure {
    /// `EmptyLink` naming the two nodes only, for sites that never held both
    /// sides' sets.
    pub fn empty_link(upstream: usize, downstream: usize) -> Self {
        NegotiationFailure::EmptyLink {
            upstream,
            downstream,
            conflict: None,
        }
    }

    /// `EmptyLink` carrying the sets each end still allowed.
    pub fn empty_link_conflict(
        upstream: usize,
        downstream: usize,
        up_set: CapsSet,
        down_set: CapsSet,
    ) -> Self {
        NegotiationFailure::EmptyLink {
            upstream,
            downstream,
            conflict: Some(Box::new(CapsConflict {
                upstream: up_set,
                downstream: down_set,
            })),
        }
    }

    /// The two sides' sets, when this failure captured them.
    pub fn conflict(&self) -> Option<&CapsConflict> {
        match self {
            NegotiationFailure::EmptyLink { conflict, .. } => conflict.as_deref(),
            _ => None,
        }
    }
}

/// Solve a linear chain of caps constraints. See module docs.
///
/// `constraints[0]` must be `Produces`; `constraints[N-1]` must be
/// `Accepts`. Interior elements may be `Identity`, `Mapping`, or
/// `DerivedOutput`.
pub fn solve_linear<'a>(
    constraints: &[&CapsConstraint<'a>],
) -> Result<LinkSolution, NegotiationFailure> {
    solve_linear_preferred(constraints, &[])
}

/// [`solve_linear`] with per-element [`CapsPreferences`], indexed the same way
/// as `constraints` (a shorter slice, or a `None` entry, means that element
/// declared none). When some element declares costs the chain fixates to the
/// least-total-cost consistent assignment instead of each link's own first
/// choice; when none does, this is [`solve_linear`] exactly.
pub fn solve_linear_preferred<'a>(
    constraints: &[&CapsConstraint<'a>],
    preferences: &[Option<CapsPreferences>],
) -> Result<LinkSolution, NegotiationFailure> {
    if constraints.len() < 2 {
        return Err(NegotiationFailure::Degenerate);
    }

    let n = constraints.len();
    let n_links = n - 1;

    // Endpoint shape check.
    let CapsConstraint::Produces(source_set) = constraints[0] else {
        return Err(NegotiationFailure::EndpointShapeMismatch { index: 0 });
    };
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
    links[0] = Some(source_set.clone());
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
    let mut domains = Vec::with_capacity(n_links);
    let mut target = producer_target(source_set);
    for (li, slot) in links.iter().enumerate() {
        let set = slot
            .as_ref()
            .ok_or(NegotiationFailure::empty_link(li, li + 1))?;
        if set.is_empty() {
            return Err(NegotiationFailure::empty_link(li, li + 1));
        }
        let candidates = fixated_candidates(set, target.as_ref());
        if candidates.is_empty() {
            return Err(NegotiationFailure::Unfixable {
                upstream: li,
                downstream: li + 1,
            });
        }
        if target.is_some() {
            target = transform_target(constraints[li + 1], candidates[0].clone());
        }
        domains.push(candidates);
    }

    // Each link's first candidate is its greedy pick, so a chain where nobody
    // declared costs keeps its per-link choice. With costs declared, the
    // chain-wide minimum can prefer a later candidate.
    let chain: Vec<ChainNode<'_, '_>> = constraints
        .iter()
        .enumerate()
        .map(|(i, c)| ChainNode::new(c, preference_at(preferences, i)))
        .collect();
    if let Some(pick) = min_cost_chain(&chain, &domains) {
        return Ok(pick
            .iter()
            .zip(&domains)
            .map(|(&j, d)| d[j].clone())
            .collect());
    }
    Ok(domains.into_iter().map(|mut d| d.remove(0)).collect())
}

#[cfg(feature = "std")]
fn forward_propagate(
    c: &CapsConstraint<'_>,
    upstream: &CapsSet,
    i: usize,
) -> Result<CapsSet, NegotiationFailure> {
    match c {
        CapsConstraint::Identity(s) => {
            let r = upstream.intersect(s);
            if r.is_empty() {
                return Err(NegotiationFailure::empty_link(i - 1, i));
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
                return Err(NegotiationFailure::empty_link(i - 1, i));
            }
            Ok(out)
        }
        CapsConstraint::DerivedOutput(f) => derived_forward(f.as_ref(), upstream, i),
        CapsConstraint::DerivedFields(t) => derived_forward(&|c: &Caps| t.derive(c), upstream, i),
        CapsConstraint::IdentityAny => {
            // Wildcard transform: pass upstream through unchanged.
            Ok(upstream.clone())
        }
        CapsConstraint::Produces(_) | CapsConstraint::Accepts(_) | CapsConstraint::AcceptsAny => {
            Err(NegotiationFailure::EndpointShapeMismatch { index: i })
        }
    }
}

/// One forward hop of the single-caps cascade for a derived transform
/// (`DerivedOutput` closure or `DerivedFields` declaration): the output is a
/// function of one concrete input, so the upstream link must fixate first.
#[cfg(feature = "std")]
fn derived_forward(
    f: &dyn Fn(&Caps) -> CapsSet,
    upstream: &CapsSet,
    i: usize,
) -> Result<CapsSet, NegotiationFailure> {
    let fixed = upstream.fixate().ok_or(NegotiationFailure::Unfixable {
        upstream: i - 1,
        downstream: i,
    })?;
    let r = f(&fixed);
    if r.is_empty() {
        return Err(NegotiationFailure::empty_link(i, i + 1));
    }
    Ok(r)
}

/// Caps-α mid-stream re-fixation outcome for one interior element
/// (design/caps.md). The runner derives the element's
/// forwarded output from its declared constraint, steered by the downstream
/// feasibility snapshot, instead of letting the element fixate greedily.
#[derive(Debug, Clone, PartialEq)]
#[cfg(feature = "std")]
pub(crate) enum ForwardResolve {
    /// Runner-derived, downstream-aware output caps to forward.
    Fixed(Caps),
    /// The output can't be derived/fixated from the constraint, or there is
    /// no concrete downstream set to steer against. Fall back to the status
    /// quo: forward the incoming caps and let the element's own `process`
    /// derive its output (covers `DerivedOutput` / ranged outputs).
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
/// sink, or a non-invertible `DerivedOutput` element below it).
/// Computed once at startup and snapshotted per interior arm so the
/// mid-stream re-solve can steer an element's output without reaching the
/// downstream elements at runtime (design/caps.md).
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
    // Seed the sink link from the sink's accept set; a wildcard sink leaves it
    // unconstrained.
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
        // A `DerivedFields` transform inverts on its passthrough fields: the
        // input feasibility is the downstream set with retargeted fields widened
        // to anything the transform accepts (see `derived_fields_inputs`). An
        // input set it cannot enumerate imposes none.
        CapsConstraint::DerivedFields(t) => {
            let d = down?;
            let mut alts: Vec<Caps> = Vec::new();
            for o in d.alternatives() {
                for input in derived_fields_inputs(t, o)? {
                    if !alts.contains(&input) {
                        alts.push(input);
                    }
                }
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
        // Non-invertible or non-transform shape: impose no constraint
        // on the input link.
        _ => None,
    }
}

// a retargeted format has no wildcard, so it is enumerated over the accept list
#[cfg(feature = "std")]
fn derived_fields_inputs(t: &CapsTransform, out: &Caps) -> Option<Vec<Caps>> {
    let mask = t.passthrough();
    if let Some(input) = project_passthrough(out, mask) {
        return Some(alloc::vec![input]);
    }
    let template = project_passthrough(out, mask.with_format())?;
    let inputs: Vec<Caps> = match (t, template) {
        (
            CapsTransform::RawVideo { accept, .. },
            Caps::RawVideo {
                width,
                height,
                framerate,
                interlace,
                colorimetry,
                ..
            },
        ) => accept
            .iter()
            .map(|&format| Caps::RawVideo {
                format,
                width: width.clone(),
                height: height.clone(),
                framerate: framerate.clone(),
                interlace,
                colorimetry,
            })
            .collect(),
        (
            CapsTransform::Audio { accept, .. },
            Caps::Audio {
                channels,
                sample_rate,
                channel_layout,
                ..
            },
        ) => accept
            .iter()
            .map(|&format| Caps::Audio {
                format,
                channels,
                sample_rate,
                channel_layout,
            })
            .collect(),
        _ => return None,
    };
    // an empty accept list means every format, which has no finite enumeration
    if inputs.is_empty() {
        return None;
    }
    let out_set = CapsSet::one(out.clone());
    Some(
        inputs
            .into_iter()
            .filter(|input| !t.derive(input).intersect(&out_set).is_empty())
            .collect(),
    )
}

/// Caps-α: derive the forwarded output for an interior element on a
/// mid-stream caps change (design/caps.md). `input` is
/// the new fixated caps the element receives; `downstream_feasible` is its
/// output link's snapshot from [`downstream_feasibility`]; `prev_output` is
/// the output caps the element last produced (startup-solved, then tracked by
/// the arm). Steers when a concrete downstream set exists; with no downstream
/// snapshot it still forwards the element's output when that output is
/// *unambiguous* (the constraint maps the input to exactly one fixated caps).
/// An ambiguous producible set (a caps-driven converter) keeps the shape it
/// already produces when that shape is still producible from the new input:
/// its scalar identity (format / codec / channels) carries over and only the
/// re-derived geometry / rate changes, so a downstream the snapshot could not
/// see (feasibility cannot be enumerated through some converters) is not
/// handed the element's *input* caps. The kept shape is forwarded only when
/// its re-derived fields **equal the input's** (a converter's copied
/// geometry / rate, unfixed values included); a value the input does not
/// carry (a nominal rate baked into a decoder's produce set) must not be
/// forwarded as if produced, so that case, and a previous shape no longer
/// producible, defer to the element's own `process`.
#[cfg(feature = "std")]
pub(crate) fn resolve_forward_output(
    constraint: &CapsConstraint<'_>,
    input: &Caps,
    downstream_feasible: Option<&CapsSet>,
    prev_output: Option<&Caps>,
) -> ForwardResolve {
    // Index 1 is a placeholder: the link position is meaningful only inside
    // a full-chain solve. Mid-stream the failure is link-local to this arm.
    // `forward_propagate` fixates the input first (the startup contract), which
    // rejects a mid-stream caps carrying a field the element never learned (a
    // Matroska track with no DefaultDuration flows framerate `Any`); the
    // Derived closures take the caps as-is, so probe them directly then.
    let candidates = forward_propagate(constraint, &CapsSet::one(input.clone()), 1)
        .ok()
        .or_else(|| match constraint {
            CapsConstraint::DerivedOutput(f) => Some(f(input)),
            CapsConstraint::DerivedFields(t) => Some(t.derive(input)),
            _ => None,
        })
        .filter(|c| !c.is_empty());
    let Some(candidates) = candidates else {
        return ForwardResolve::Defer;
    };
    // The previous output's scalar identity with geometry / rate widened: the
    // shape to prefer among ambiguous candidates. `project_passthrough` keeps
    // the masked scalars and widens the rest to `Any`; a variant it cannot
    // express (Text / Bytestream) yields no preference.
    let keep_shape = prev_output.and_then(|p| {
        project_passthrough(p, PassthroughFields::NONE.with_format().with_channels())
    });
    // Fields the element copies from its input. A `DerivedFields` transform
    // states which it retargets instead, so a survivor differing from the input
    // there is what the element produces (`audioconvert channels=1` fed stereo).
    // Every other constraint declares nothing, so all fields are compared.
    let copied_fields = match constraint {
        CapsConstraint::DerivedFields(t) => t.passthrough(),
        _ => PassthroughFields::ALL,
    };
    // A kept-shape survivor is forwardable only when its re-derived fields
    // EQUAL the new input's (a converter's copied geometry / rate, unfixed
    // values included: a decoder emitting no framerate stays that way through
    // the converter). A nominal fixate-fallback alternative in a decoder's
    // produce set (vorbisdec's 48 kHz against a 44.1 kHz input) is not
    // input-derived, and forwarding it would announce a value the element
    // never produces.
    fn tracks_input(survivor: &Caps, input: &Caps, copied: PassthroughFields) -> bool {
        match (survivor.dims(), input.dims()) {
            (Some((s_w, s_h, s_rate)), Some((i_w, i_h, i_rate))) => {
                (!copied.width || s_w == i_w)
                    && (!copied.height || s_h == i_h)
                    && (!copied.framerate || s_rate == i_rate)
            }
            _ => match (survivor, input) {
                (
                    Caps::Audio {
                        channels: s_ch,
                        sample_rate: s_rate,
                        ..
                    },
                    Caps::Audio {
                        channels: i_ch,
                        sample_rate: i_rate,
                        ..
                    },
                ) => {
                    (!copied.sample_rate || s_rate == i_rate) && (!copied.channels || s_ch == i_ch)
                }
                _ => false,
            },
        }
    }
    // an alternative leaving a kept scalar open would take the startup placeholder, not a value this input derives
    let fixate_kept_shape = |set: &CapsSet| -> Option<Caps> {
        let shape = keep_shape.as_ref()?;
        set.alternatives()
            .iter()
            .filter(|alternative| alternative.intersect(shape).as_ref() == Ok(*alternative))
            .find(|c| tracks_input(c, input, copied_fields))
            .cloned()
    };
    // a Derived closure computed the candidate from this input, so the element
    // really does produce it, changed geometry included
    let derives_from_input = matches!(
        constraint,
        CapsConstraint::DerivedOutput(_) | CapsConstraint::DerivedFields(_)
    );
    let Some(d) = downstream_feasible else {
        return match candidates.alternatives() {
            // Unambiguous: forward the one output, fixated, or as-is when an
            // unlearned input field (Any rate) blocks fixation but the output is
            // input-derived or otherwise tracks the input.
            [one] => match candidates.fixate() {
                Some(c) => ForwardResolve::Fixed(c),
                None if derives_from_input || tracks_input(one, input, copied_fields) => {
                    ForwardResolve::Fixed(one.clone())
                }
                None => ForwardResolve::Defer,
            },
            // Ambiguous with nothing to steer by: keep the previous output
            // shape if still producible, else the status-quo Defer (never
            // pick an arbitrary alternative here).
            _ => match fixate_kept_shape(&candidates) {
                Some(c) => ForwardResolve::Fixed(c),
                None => ForwardResolve::Defer,
            },
        };
    };
    let narrowed = candidates.intersect(d);
    if narrowed.is_empty() {
        return ForwardResolve::Infeasible(NegotiationFailure::empty_link(0, 1));
    }
    match fixate_kept_shape(&narrowed)
        .or_else(|| narrowed.fixate())
        .or_else(|| match narrowed.alternatives() {
            // Unambiguous but unfixatable: an input field the element never
            // learned (a decoder announcing its geometry before it knows the
            // framerate) leaves the derived output partly unfixed. Deferring
            // here would forward the element's INPUT, which misreports what a
            // geometry-changing element produces; downstream has already
            // accepted this shape, so send the element's own output on.
            [one] => Some(one.clone()),
            _ => None,
        }) {
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
                        return Err(NegotiationFailure::empty_link(i - 1, i + 1));
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
                return Err(NegotiationFailure::empty_link(i - 1, i + 1));
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
                    return Err(NegotiationFailure::empty_link(i, i + 1));
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
                    Err(()) => return Err(NegotiationFailure::empty_link(i - 1, i)),
                }
            }
        }
        CapsConstraint::DerivedFields(t) => {
            let (Some(ii), Some(oi)) = (in_idx, out_idx) else {
                return Err(NegotiationFailure::EndpointShapeMismatch { index: i });
            };
            let derive = |c: &Caps| t.derive(c);
            // Forward: identical to `DerivedOutput` (the declaration is the source
            // of truth for forward derivation).
            if let Some(input_set) = &links[ii] {
                let derived = forward_derived_union(&derive, input_set);
                if derived.is_empty() {
                    return Err(NegotiationFailure::empty_link(i, i + 1));
                }
                narrow(links, oi, &derived, i, i + 1)?;
            }
            // Backward: field-level coupling, narrowing passthrough fields *within*
            // an alternative (the unblock over `DerivedOutput`'s alternative-drop).
            if let (Some(in_set), Some(out_set)) = (links[ii].clone(), links[oi].clone()) {
                match backward_field_narrow(&derive, t.passthrough(), &in_set, &out_set) {
                    Ok(Some(narrowed)) => links[ii] = Some(narrowed),
                    Ok(None) => {}
                    Err(()) => return Err(NegotiationFailure::empty_link(i - 1, i)),
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
                            return Err(NegotiationFailure::empty_link(i - 1, i + 1));
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
        return Err(NegotiationFailure::empty_link(upstream, downstream));
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
    /// Fan-out demux: `input` is the byte-stream input constraint (the
    /// container the demux consumes), and `ports[i]` is output port `i`'s
    /// `Produces` constraint (its distinct elementary stream), or a transform
    /// constraint for a port whose caps follow the input. Unlike a broadcast
    /// tee (which couples in == every out), the ports are *decoupled*, so each
    /// branch negotiates against its own caps and a downstream decoder configures
    /// against its codec at startup. Built from a demux element that declares
    /// per-port caps (`MultiOutputElement::port_caps_constraint`); a broadcast fan-out
    /// (declaring none) stays a plain tee.
    Demux {
        input: CapsConstraint<'a>,
        ports: Vec<CapsConstraint<'a>>,
    },
    /// Batching mux: `inputs[i]` is input pad `i`'s constraint as in
    /// [`Muxer`](Self::Muxer), and the output edge carries a [`Caps::Batch`]
    /// whose streams are the input edges' caps in pad order.
    BatchMuxer { inputs: Vec<CapsConstraint<'a>> },
    /// Unbatching demux: `input` is the input edge's constraint as in
    /// [`Demux`](Self::Demux), and output port `i` carries stream `i` of the
    /// input's [`Caps::Batch`].
    BatchDemux { input: CapsConstraint<'a> },
}

/// Solve caps for an arbitrary DAG (design/TODO.md "DAG runner" D2). Generalizes
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
/// explainer (design/tooling.md). The runner passes each node's element category
/// (e.g. `h264parse`) so the `G2G_CAPS_TRACE` narration reads in element names
/// rather than node ids; direct callers use [`solve_graph`]'s `n{id}:{kind}`
/// default. The solve itself is identical; `label` only affects the log text,
/// and all formatting is skipped unless the [`CAPS_CATEGORY`] is enabled.
pub fn solve_graph_labeled<E>(
    graph: &ValidatedGraph<E>,
    constraints: &[NodeConstraint<'_>],
    label: &dyn Fn(NodeId) -> String,
) -> Result<Vec<Caps>, NegotiationFailure> {
    solve_graph_preferred(graph, constraints, &[], label)
}

/// [`solve_graph_labeled`] with each node's declared [`CapsPreferences`],
/// indexed by node id (a shorter slice, or a `None` entry, means that element
/// declared none). On a linear chain, and only when some element declares
/// costs, fixation picks the consistent assignment of least total cost rather
/// than the first the greedy backtrack finds; every other graph, and every
/// chain with nothing declared, fixates exactly as [`solve_graph_labeled`].
pub fn solve_graph_preferred<E>(
    graph: &ValidatedGraph<E>,
    constraints: &[NodeConstraint<'_>],
    preferences: &[Option<CapsPreferences>],
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
    let report = |f: &NegotiationFailure, edges: &[Option<CapsSet>]| match f {
        NegotiationFailure::EmptyLink {
            upstream,
            downstream,
            ..
        }
        | NegotiationFailure::NoOutputForInput {
            upstream,
            downstream,
            ..
        } => {
            let (up, down) = (NodeId(*upstream as u32), NodeId(*downstream as u32));
            if let NegotiationFailure::NoOutputForInput { input, .. } = f {
                crate::g2g_error!(t, "{} produces no output for {}", label(up), fmt_set(input));
            } else {
                crate::g2g_error!(
                    t,
                    "no caps overlap between {} and {}",
                    label(up),
                    label(down)
                );
            }
            if let Some(conflict) = f.conflict() {
                crate::g2g_error!(t, "  {} offers: {}", label(up), fmt_set(&conflict.upstream));
                crate::g2g_error!(
                    t,
                    "  {} wants: {}",
                    label(down),
                    fmt_set(&conflict.downstream)
                );
                for field in conflict.conflicting_fields() {
                    crate::g2g_error!(
                        t,
                        "  {}: {} offers {}, {} wants {}",
                        field.field,
                        label(up),
                        fmt_alternatives(field.upstream.into_iter()),
                        label(down),
                        fmt_alternatives(field.downstream.into_iter())
                    );
                }
            }
            for (id, slot) in edges.iter().enumerate() {
                let e = graph.edge(id);
                if [e.src.node, e.dst.node]
                    .iter()
                    .any(|&x| x == up || x == down)
                {
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

    // Build a per-edge candidate domain (each surviving alternative, fixated with
    // its ranges toward the edge's fixation target, in the set's own preference
    // order so the first candidate is the per-edge greedy pick). Then assign one
    // candidate per edge by backtracking search so the chosen combination is
    // *globally* consistent. Arc consistency above narrows each edge against its neighbours
    // pairwise, but a diamond (a tee whose branches re-converge at a fan-in)
    // couples branch choices in a way pairwise narrowing cannot see, so per-edge
    // greedy fixation can pick a locally-valid yet jointly-impossible combination
    // (e.g. two branches that map the shared tee value to different outputs whose
    // alternative orders disagree). The search tries the greedy choice first, so a
    // chain or an independent fan-out fixates byte-for-byte as before and only a
    // genuinely coupled diamond ever explores alternatives.
    let targets = fixation_targets(graph, constraints, &edges);
    let mut domains: Vec<Vec<Caps>> = Vec::with_capacity(ne);
    for (id, slot) in edges.iter().enumerate() {
        let (up, down) = edge_endpoints(graph, id);
        let set = match slot.as_ref() {
            Some(s) if !s.is_empty() => s,
            _ => {
                let f = NegotiationFailure::empty_link(up, down);
                report(&f, &edges);
                return Err(f);
            }
        };
        let doms = fixated_candidates(set, targets[id].as_ref());
        if doms.is_empty() {
            crate::g2g_error!(
                t,
                "{} -> {}: {} ✗ cannot fixate (still ambiguous after narrowing)",
                label(graph.edge(id).src.node),
                label(graph.edge(id).dst.node),
                fmt_set(set)
            );
            return Err(NegotiationFailure::Unfixable {
                upstream: up,
                downstream: down,
            });
        }
        domains.push(doms);
    }

    let mut assign: Vec<Option<Caps>> = alloc::vec![None; ne];
    if let Some(chosen) = preferred_chain_assignment(graph, constraints, preferences, &domains) {
        assign = chosen;
    } else if !fixate_backtrack(graph, constraints, &domains, &mut assign, 0) {
        let f = NegotiationFailure::NoConsistentFixation;
        report(&f, &edges);
        return Err(f);
    }
    let out: Vec<Caps> = assign
        .into_iter()
        .map(|a| a.expect("every edge assigned"))
        .collect();
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

/// Every alternative of `set` that collapses to a concrete `Caps`, deduped and
/// in the set's own preference order, so an empty result means the same as
/// [`CapsSet::fixate`]'s `None`. With no `preferred` caps the first entry is
/// exactly what [`CapsSet::fixate`] would return.
fn fixated_candidates(set: &CapsSet, preferred: Option<&Caps>) -> Vec<Caps> {
    let mut out: Vec<Caps> = Vec::new();
    for alt in set.alternatives() {
        let toward_preferred = preferred.and_then(|p| alt.fixate_toward(p).ok());
        // the minimum stays a candidate for when the preferred value breaks a node's relation
        for c in toward_preferred.into_iter().chain(alt.fixate().ok()) {
            if !out.contains(&c) {
                out.push(c);
            }
        }
    }
    out
}

// per edge, a producer's first alternative when fixed, else what the node passes on from a feeding edge that has one
fn fixation_targets<E>(
    graph: &ValidatedGraph<E>,
    constraints: &[NodeConstraint<'_>],
    edges: &[Option<CapsSet>],
) -> Vec<Option<Caps>> {
    let mut targets: Vec<Option<Caps>> = alloc::vec![None; edges.len()];
    for &node in graph.topo() {
        let first_fixation = |edge: usize| {
            let target = targets[edge].as_ref()?;
            fixated_candidates(edges[edge].as_ref()?, Some(target))
                .into_iter()
                .next()
        };
        let target = match (graph.kind(node), &constraints[node.0 as usize]) {
            (_, NodeConstraint::Element(CapsConstraint::Produces(set))) => producer_target(set),
            (NodeKind::Transform, NodeConstraint::Element(c)) => graph
                .in_edges(node)
                .first()
                .and_then(|&e| first_fixation(e))
                .and_then(|input| transform_target(c, input)),
            (NodeKind::Tee(_), NodeConstraint::Element(_)) => graph
                .in_edges(node)
                .first()
                .and_then(|&e| first_fixation(e)),
            (
                NodeKind::Muxer(_),
                NodeConstraint::Muxer {
                    follows: Some(pad), ..
                },
            ) => graph
                .in_edges(node)
                .iter()
                .find(|&&e| graph.edge(e).dst.index as usize == *pad)
                .and_then(|&e| first_fixation(e)),
            _ => None,
        };
        for &edge in graph.out_edges(node) {
            targets[edge] = target.clone();
        }
    }
    targets
}

fn producer_target(set: &CapsSet) -> Option<Caps> {
    set.alternatives().first().filter(|c| c.is_fixed()).cloned()
}

// what a transform outputs for a fixed `input`, `None` where its output is not a function of it
fn transform_target(c: &CapsConstraint<'_>, input: Caps) -> Option<Caps> {
    match c {
        CapsConstraint::Identity(_) | CapsConstraint::IdentityAny => Some(input),
        CapsConstraint::DerivedOutput(f) => f(&input).fixate(),
        CapsConstraint::DerivedFields(t) => t.derive(&input).fixate(),
        _ => None,
    }
}

/// `preferences[i]`, tolerating a slice shorter than the chain.
fn preference_at(
    preferences: &[Option<CapsPreferences>],
    index: usize,
) -> Option<&CapsPreferences> {
    preferences.get(index).and_then(Option::as_ref)
}

/// One element of a linear chain, paired with what it is willing to pay for
/// each of its advertised alternatives.
struct ChainNode<'c, 'a> {
    constraint: &'c CapsConstraint<'a>,
    preferences: Option<&'c CapsPreferences>,
}

impl<'c, 'a> ChainNode<'c, 'a> {
    fn new(constraint: &'c CapsConstraint<'a>, preferences: Option<&'c CapsPreferences>) -> Self {
        let preferences = preferences.filter(|p| !p.is_empty());
        Self {
            constraint,
            preferences,
        }
    }

    /// What this element pays for carrying `input` on its input link and
    /// `output` on its output link. A constraint with no alternative list to
    /// index (a wildcard or a derived transform) is free, so it
    /// neither steers the chain nor blocks a neighbour that does.
    fn cost(&self, input: Option<&Caps>, output: Option<&Caps>) -> u64 {
        match alternative_index(self.constraint, input, output) {
            Some(i) => self.preferences.map_or(i as u64, |p| p.cost(i) as u64),
            None => 0,
        }
    }
}

/// Which advertised alternative of `c` the pair `(input, output)` selects: the
/// first one compatible with the chosen caps. `None` when the constraint
/// advertises no indexable list.
fn alternative_index(
    c: &CapsConstraint<'_>,
    input: Option<&Caps>,
    output: Option<&Caps>,
) -> Option<usize> {
    let position = |set: &CapsSet, caps: &Caps| {
        set.alternatives()
            .iter()
            .position(|a| a.intersect(caps).is_ok())
    };
    match c {
        CapsConstraint::Produces(set) => position(set, output?),
        CapsConstraint::Accepts(set) => position(set, input?),
        CapsConstraint::Identity(set) => position(set, input.or(output)?),
        CapsConstraint::Mapping(pairs) => {
            let (input, output) = (input?, output?);
            pairs
                .iter()
                .position(|(i, o)| i.accepts(input) && o.accepts(output))
        }
        _ => None,
    }
}

/// Least-total-cost consistent assignment for a linear chain: one candidate
/// index per link. `nodes` are the chain's elements in order and `domains[i]`
/// the fixated candidates on the link between node `i` and node `i + 1`, in
/// that link's own preference order.
///
/// Dynamic programming over adjacent pairs: the state is the candidate chosen
/// on one link, the transition is the element between two links (its
/// input/output relation must hold, and it charges its cost for that pair).
/// Ties break toward the lexicographically first candidate sequence, which is
/// the greedy first-fixable pick, so an all-equal-cost chain resolves exactly
/// as it does with no preferences at all.
///
/// `None` when no element declared costs (the caller keeps its existing
/// fixation untouched) or when no assignment satisfies every element.
fn min_cost_chain(nodes: &[ChainNode<'_, '_>], domains: &[Vec<Caps>]) -> Option<Vec<usize>> {
    if domains.is_empty() || nodes.len() != domains.len() + 1 {
        return None;
    }
    if !nodes.iter().any(|n| n.preferences.is_some()) {
        return None;
    }

    // best[j]: the cheapest prefix ending with `domains[link][j]` on the link
    // under consideration, and the candidate indices that reached it.
    let mut best: Vec<Option<(u64, Vec<usize>)>> = domains[0]
        .iter()
        .enumerate()
        .map(|(j, caps)| Some((nodes[0].cost(None, Some(caps)), alloc::vec![j])))
        .collect();

    for link in 1..domains.len() {
        let middle = &nodes[link];
        let mut next: Vec<Option<(u64, Vec<usize>)>> = alloc::vec![None; domains[link].len()];
        for (b, output) in domains[link].iter().enumerate() {
            for (a, input) in domains[link - 1].iter().enumerate() {
                let Some((so_far, path)) = best[a].as_ref() else {
                    continue;
                };
                if !transform_pair_consistent(middle.constraint, input, output) {
                    continue;
                }
                let total = so_far.saturating_add(middle.cost(Some(input), Some(output)));
                let mut candidate = path.clone();
                candidate.push(b);
                if is_better(next[b].as_ref(), total, &candidate) {
                    next[b] = Some((total, candidate));
                }
            }
        }
        best = next;
    }

    let last = nodes.last()?;
    let mut winner: Option<(u64, Vec<usize>)> = None;
    for (j, caps) in domains[domains.len() - 1].iter().enumerate() {
        let Some((so_far, path)) = best[j].as_ref() else {
            continue;
        };
        let total = so_far.saturating_add(last.cost(Some(caps), None));
        if is_better(winner.as_ref(), total, path) {
            winner = Some((total, path.clone()));
        }
    }
    winner.map(|(_, path)| path)
}

/// Whether `(cost, path)` beats the incumbent: cheaper, or equally cheap with a
/// lexicographically earlier candidate sequence.
fn is_better(current: Option<&(u64, Vec<usize>)>, cost: u64, path: &[usize]) -> bool {
    match current {
        None => true,
        Some((c, p)) => (cost, path) < (*c, p.as_slice()),
    }
}

/// Preference-driven fixation for a graph that is a plain linear chain: the
/// same DP as [`min_cost_chain`], mapped onto edge ids. `None` (so the caller
/// keeps its greedy backtracking) when the graph is not a chain, when no
/// element declared costs, or when the DP finds no consistent assignment.
fn preferred_chain_assignment<E>(
    graph: &ValidatedGraph<E>,
    constraints: &[NodeConstraint<'_>],
    preferences: &[Option<CapsPreferences>],
    domains: &[Vec<Caps>],
) -> Option<Vec<Option<Caps>>> {
    if !preferences.iter().any(Option::is_some) {
        return None;
    }
    let (nodes, edges) = chain_order(graph, constraints)?;
    let chain: Vec<ChainNode<'_, '_>> = nodes
        .iter()
        .map(|&node| {
            let index = node.0 as usize;
            let NodeConstraint::Element(c) = &constraints[index] else {
                unreachable!("chain_order admits Element nodes only")
            };
            ChainNode::new(c, preference_at(preferences, index))
        })
        .collect();
    let chain_domains: Vec<Vec<Caps>> = edges.iter().map(|&e| domains[e].clone()).collect();
    let pick = min_cost_chain(&chain, &chain_domains)?;

    let mut assign: Vec<Option<Caps>> = alloc::vec![None; domains.len()];
    for ((&edge, candidates), &j) in edges.iter().zip(&chain_domains).zip(&pick) {
        assign[edge] = Some(candidates[j].clone());
    }
    Some(assign)
}

/// The graph as a linear chain: its nodes in source-to-sink order and the edge
/// ids between them. `None` unless every node has at most one input and one
/// output edge, the edges form a single path over all of them, and every node
/// carries a plain [`NodeConstraint::Element`] (a demux or muxer node couples
/// edges the chain DP does not model).
fn chain_order<E>(
    graph: &ValidatedGraph<E>,
    constraints: &[NodeConstraint<'_>],
) -> Option<(Vec<NodeId>, Vec<usize>)> {
    let n = graph.node_count();
    if n < 2 || graph.edge_count() + 1 != n {
        return None;
    }
    if !constraints
        .iter()
        .all(|c| matches!(c, NodeConstraint::Element(_)))
    {
        return None;
    }
    let mut head = None;
    for i in 0..n {
        let node = NodeId(i as u32);
        if graph.in_edges(node).len() > 1 || graph.out_edges(node).len() > 1 {
            return None;
        }
        if graph.in_edges(node).is_empty() {
            if head.is_some() {
                return None;
            }
            head = Some(node);
        }
    }

    let mut nodes = Vec::with_capacity(n);
    let mut edges = Vec::with_capacity(n - 1);
    let mut current = head?;
    loop {
        nodes.push(current);
        match graph.out_edges(current).first() {
            Some(&edge) => {
                edges.push(edge);
                current = graph.edge(edge).dst.node;
            }
            None => break,
        }
    }
    (nodes.len() == n).then_some((nodes, edges))
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
    if in_e
        .iter()
        .chain(out_e.iter())
        .any(|&e| assign[e].is_none())
    {
        return true;
    }
    let get = |e: usize| assign[e].as_ref().expect("checked all assigned");
    match graph.kind(node) {
        // Membership is already ensured by arc consistency; nothing cross-edge.
        NodeKind::Source | NodeKind::Sink | NodeKind::FaninSink(_) | NodeKind::FanoutSrc(_) => true,
        NodeKind::Muxer(_) => match &constraints[node.0 as usize] {
            NodeConstraint::BatchMuxer { .. } => {
                let streams = pads_in_order(graph, node)
                    .into_iter()
                    .map(|(_, e)| get(e).clone())
                    .collect();
                *get(out_e[0]) == Caps::Batch { streams }
            }
            _ => true,
        },
        NodeKind::Tee(_) => match &constraints[node.0 as usize] {
            // A demux decouples its ports (each its own produce set, ensured by arc
            // consistency), so only a port that follows the input relates to it.
            NodeConstraint::Demux { ports, .. } => out_e.iter().all(|&o| {
                ports
                    .get(graph.edge(o).src.index as usize)
                    .is_none_or(|port| transform_pair_consistent(port, get(in_e[0]), get(o)))
            }),
            NodeConstraint::BatchDemux { .. } => match get(in_e[0]) {
                Caps::Batch { streams } if streams.len() == out_e.len() => out_e
                    .iter()
                    .all(|&o| streams.get(graph.edge(o).src.index as usize) == Some(get(o))),
                _ => false,
            },
            // A broadcast tee requires every branch to carry its input.
            _ => {
                let inp = get(in_e[0]);
                out_e.iter().all(|&o| get(o) == inp)
            }
        },
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
        CapsConstraint::DerivedFields(t) => t.derive(inp).accepts(outp),
        // Produce / accept shapes on a transform slot: not a cross-edge relation
        // re-checked here (arc consistency handled the forward cascade).
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
        NodeKind::FaninSink(_) => "fanin-sink",
        NodeKind::FanoutSrc(_) => "fanout-src",
    }
}

/// Render a `CapsSet` for the explainer: its alternatives joined by ` | `, with
/// `∅` for empty and the [`fmt_alternatives`] elision.
fn fmt_set(set: &CapsSet) -> String {
    if set.is_empty() {
        return String::from("∅");
    }
    fmt_alternatives(set.alternatives().iter().map(Caps::to_gst_string))
}

// a wide set stays one readable line unless the caps category is at trace
const ALTERNATIVES_SHOWN: usize = 4;

fn fmt_alternatives(texts: impl ExactSizeIterator<Item = String>) -> String {
    let total = texts.len();
    let shown = if log::enabled(CAPS_CATEGORY, LogLevel::Trace) {
        total
    } else {
        total.min(ALTERNATIVES_SHOWN)
    };
    let mut parts: Vec<String> = texts.take(shown).collect();
    if shown < total {
        parts.push(alloc::format!(
            "(+{} more, G2G_CAPS_TRACE=trace lists them)",
            total - shown
        ));
    }
    parts.join(" | ")
}

fn fmt_set_opt(slot: &Option<CapsSet>) -> String {
    match slot {
        Some(s) => fmt_set(s),
        None => String::from("(not narrowed yet)"),
    }
}

/// One-line summary of a node's constraint for the explainer's setup dump.
fn fmt_constraint(nc: &NodeConstraint<'_>) -> String {
    match nc {
        NodeConstraint::Element(c) => fmt_caps_constraint(c),
        NodeConstraint::Muxer {
            inputs,
            output,
            follows,
        } => match follows {
            Some(pad) => alloc::format!("mux {} inputs -> follows input {pad}", inputs.len()),
            None => alloc::format!(
                "mux {} inputs -> {}",
                inputs.len(),
                fmt_caps_constraint(output)
            ),
        },
        NodeConstraint::Demux { ports, .. } => alloc::format!("demux -> {} ports", ports.len()),
        NodeConstraint::BatchMuxer { inputs } => {
            alloc::format!("batch mux {} inputs -> batch of their caps", inputs.len())
        }
        NodeConstraint::BatchDemux { .. } => "batch demux -> one port per stream".to_string(),
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
        CapsConstraint::DerivedFields(_) => "derives output (coupled)".to_string(),
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
            // Terminal ends of the DAG: no in-edges to write feasibility for.
            NodeKind::Source | NodeKind::FanoutSrc(_) => {}
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
                if let NodeConstraint::Demux { input, .. } | NodeConstraint::BatchDemux { input } =
                    &constraints[idx]
                {
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
            // A muxer and a terminal fan-in both take their input pads' accept
            // sets independently (a muxer's output does not feed back to its
            // inputs; a terminal fan-in has no output at all).
            NodeKind::Muxer(_) | NodeKind::FaninSink(_) => {
                if let NodeConstraint::Muxer { inputs, .. }
                | NodeConstraint::BatchMuxer { inputs } = &constraints[idx]
                {
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
                narrow_edge(graph, edges, out_e[0], s, node)
            }
            _ => Err(shape_err),
        },
        NodeKind::Sink => match nc {
            NodeConstraint::Element(CapsConstraint::Accepts(s)) => {
                narrow_edge(graph, edges, in_e[0], s, node)
            }
            // `AcceptsAny` leaves the input edge to carry whatever the upstream fixates.
            NodeConstraint::Element(CapsConstraint::AcceptsAny) => Ok(()),
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
                apply_demux_node(graph, node, in_e[0], out_e, input, ports, edges)
            }
            NodeConstraint::BatchDemux { input } => {
                apply_batch_demux_node(graph, node, in_e[0], out_e, input, edges)
            }
            _ => apply_tee_node(graph, in_e[0], out_e, edges),
        },
        NodeKind::Muxer(_) => match nc {
            NodeConstraint::Muxer {
                inputs,
                output,
                follows,
            } => apply_muxer_node(graph, node, inputs, output, *follows, edges),
            NodeConstraint::BatchMuxer { inputs } => {
                apply_batch_muxer_node(graph, node, inputs, edges)
            }
            _ => Err(shape_err),
        },
        // A terminal fan-in narrows only its input pads (no output edge to
        // couple); it reuses the muxer constraint shape with `output` / `follows`
        // unused.
        NodeKind::FaninSink(_) => match nc {
            NodeConstraint::Muxer { inputs, .. } => narrow_muxer_inputs(graph, node, inputs, edges),
            _ => Err(shape_err),
        },
        // A terminal fan-out source narrows each output edge by its port's
        // produce set (the demux constraint shape with the input half unused:
        // there is no input edge).
        NodeKind::FanoutSrc(_) => match nc {
            NodeConstraint::Demux { ports, .. } => {
                for &oe in out_e {
                    let port = graph.edge(oe).src.index as usize;
                    match ports.get(port) {
                        Some(CapsConstraint::Produces(set)) => {
                            narrow_edge(graph, edges, oe, set, node)?
                        }
                        _ => return Err(shape_err),
                    }
                }
                Ok(())
            }
            _ => Err(shape_err),
        },
    }
}

/// Narrow one edge by `contrib`, the set node `by` imposes on it. `by` is what
/// orients an `EmptyLink`: the contribution belongs to that end of the link, and
/// the set already on the edge stands for the other end.
fn narrow_edge<E>(
    graph: &ValidatedGraph<E>,
    edges: &mut [Option<CapsSet>],
    edge_id: usize,
    contrib: &CapsSet,
    by: NodeId,
) -> Result<(), NegotiationFailure> {
    let next = match &edges[edge_id] {
        Some(cur) => cur.intersect(contrib),
        None => contrib.clone(),
    };
    if next.is_empty() {
        let (up, down) = edge_endpoints(graph, edge_id);
        let other = edges[edge_id].clone().unwrap_or_else(empty_set);
        return Err(if graph.edge(edge_id).src.node == by {
            NegotiationFailure::empty_link_conflict(up, down, contrib.clone(), other)
        } else {
            NegotiationFailure::empty_link_conflict(up, down, other, contrib.clone())
        });
    }
    edges[edge_id] = Some(next);
    Ok(())
}

fn empty_set() -> CapsSet {
    CapsSet::from_alternatives(Vec::new())
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
                return Err(NegotiationFailure::empty_link_conflict(up, down, sa, sb));
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
            narrow_edge(graph, edges, in_e, s, node)?;
            narrow_edge(graph, edges, out_e, s, node)?;
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
                return Err(NegotiationFailure::empty_link_conflict(
                    up,
                    down,
                    edges[in_e].clone().unwrap_or_else(empty_set),
                    edges[out_e].clone().unwrap_or_else(empty_set),
                ));
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
                    let (upstream, downstream) = edge_endpoints(graph, out_e);
                    return Err(NegotiationFailure::NoOutputForInput {
                        upstream,
                        downstream,
                        input: in_set,
                    });
                }
                narrow_edge(graph, edges, out_e, &derived, node)?;
            }
            // Backward (M188 + invertible-field coupling): field-level narrow on
            // the closure's probed passthrough fields, else alternative-drop.
            if let (Some(in_set), Some(out_set)) = (edges[in_e].clone(), edges[out_e].clone()) {
                match derived_backward(f.as_ref(), &in_set, &out_set) {
                    Ok(Some(narrowed)) => edges[in_e] = Some(narrowed),
                    Ok(None) => {}
                    Err(()) => {
                        let (up, down) = edge_endpoints(graph, in_e);
                        return Err(NegotiationFailure::empty_link(up, down));
                    }
                }
            }
            Ok(())
        }
        CapsConstraint::DerivedFields(t) => {
            // Mirror of the linear `apply_constraint` arm on graph edges:
            // forward via the declaration, backward via field-level coupling.
            let derive = |c: &Caps| t.derive(c);
            if let Some(in_set) = edges[in_e].clone() {
                let derived = forward_derived_union(&derive, &in_set);
                if derived.is_empty() {
                    let (upstream, downstream) = edge_endpoints(graph, out_e);
                    return Err(NegotiationFailure::NoOutputForInput {
                        upstream,
                        downstream,
                        input: in_set,
                    });
                }
                narrow_edge(graph, edges, out_e, &derived, node)?;
            }
            if let (Some(in_set), Some(out_set)) = (edges[in_e].clone(), edges[out_e].clone()) {
                match backward_field_narrow(&derive, t.passthrough(), &in_set, &out_set) {
                    Ok(Some(narrowed)) => edges[in_e] = Some(narrowed),
                    Ok(None) => {}
                    Err(()) => {
                        let (up, down) = edge_endpoints(graph, in_e);
                        return Err(NegotiationFailure::empty_link(up, down));
                    }
                }
            }
            Ok(())
        }
        _ => Err(NegotiationFailure::EndpointShapeMismatch {
            index: node.0 as usize,
        }),
    }
}

/// Forward image of a `DerivedOutput` transform over its (possibly ambiguous)
/// input set: the union of `f` over the input alternatives. For a single
/// fixated input this is just `f(input)`; for a multi-alternative input it lets a
/// downstream auto transform still receive an output to narrow, instead of
/// stalling until the input fixates (which it can't, with no downstream pin).
fn forward_derived_union(f: &dyn Fn(&Caps) -> CapsSet, in_set: &CapsSet) -> CapsSet {
    in_set
        .alternatives()
        .iter()
        .fold(CapsSet::from_alternatives(Vec::new()), |acc, a| {
            acc.union(&f(a))
        })
}

/// Backward narrowing for a `DerivedOutput` transform: given the (already
/// constrained) output set, drop input alternatives whose forward image `f(a)`
/// can no longer satisfy it. `f` is not analytically invertible, but it is
/// evaluable per candidate, so a downstream pin propagates back through a
/// not-yet-fixated transform, letting stacked auto transforms
/// (`videoconvert ! videoscale ! caps`) resolve.
///
/// Only narrows when the input is still ambiguous (more than one alternative),
/// so single-input transforms (decoders, single-transform pipelines) are
/// untouched. Returns `Some(narrowed)` when it removed
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
/// exactly as a declared `DerivedFields` mask would
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

/// Backward field-coupling for a `DerivedFields` transform: the primitive the
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
        // a same-variant `DerivedFields` transform gets the exact coupling.
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
            return Err(NegotiationFailure::empty_link(up, down));
        }
        edges[in_e] = Some(coupled.clone());
        for &oe in out_e {
            edges[oe] = Some(coupled.clone());
        }
    }
    Ok(())
}

/// Apply a fan-out **demux** node: its ports are decoupled, so the input
/// edge narrows by the demux's `input` accept (the container it consumes) and each
/// output edge narrows by its port's `Produces` caps (its elementary stream),
/// independent of one another. A port with a transform constraint instead
/// relates its edge to the input edge as a transform would. The port for an
/// output edge is its source pad index, so the order of `out_e` does not matter.
fn apply_demux_node<E>(
    graph: &ValidatedGraph<E>,
    node: NodeId,
    in_e: usize,
    out_e: &[usize],
    input: &CapsConstraint<'_>,
    ports: &[CapsConstraint<'_>],
    edges: &mut [Option<CapsSet>],
) -> Result<(), NegotiationFailure> {
    // The byte-stream input: an `Accepts` narrows it; `AcceptsAny` leaves it to
    // whatever the source fixates.
    if let CapsConstraint::Accepts(s) = input {
        narrow_edge(graph, edges, in_e, s, node)?;
    }
    for &oe in out_e {
        let port = graph.edge(oe).src.index as usize;
        match ports.get(port) {
            Some(CapsConstraint::Produces(s)) => narrow_edge(graph, edges, oe, s, node)?,
            Some(follows_input) => {
                apply_transform_node(graph, follows_input, in_e, oe, edges, node)?
            }
            None => {}
        }
    }
    Ok(())
}

/// Narrow each of a fan-in node's input edges by its pad's accept set, shared by
/// the muxer (which also couples an output) and the terminal fan-in (which does
/// not). `Accepts` narrows; an `AcceptsAny` pad forwards per-frame caps
/// without narrowing. `inputs[i]` applies to input pad `i`; D1 validation
/// guarantees each input pad index appears exactly once.
fn narrow_muxer_inputs<E>(
    graph: &ValidatedGraph<E>,
    node: NodeId,
    inputs: &[CapsConstraint<'_>],
    edges: &mut [Option<CapsSet>],
) -> Result<(), NegotiationFailure> {
    let shape_err = NegotiationFailure::EndpointShapeMismatch {
        index: node.0 as usize,
    };
    for &eid in graph.in_edges(node) {
        let pad = graph.edge(eid).dst.index as usize;
        match inputs.get(pad) {
            Some(CapsConstraint::Accepts(set)) => narrow_edge(graph, edges, eid, set, node)?,
            Some(CapsConstraint::AcceptsAny) => {}
            _ => return Err(shape_err),
        }
    }
    Ok(())
}

/// A muxer fans in: apply each input pad's constraint to its edge, and the
/// single output edge by the `Produces` set (or couple it to the followed
/// input pad).
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
    narrow_muxer_inputs(graph, node, inputs, edges)?;
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
        CapsConstraint::Produces(set) => narrow_edge(graph, edges, out_edge, set, node),
        _ => Err(shape_err),
    }
}

fn pads_in_order<E>(graph: &ValidatedGraph<E>, node: NodeId) -> Vec<(usize, usize)> {
    let mut pads: Vec<(usize, usize)> = graph
        .in_edges(node)
        .iter()
        .map(|&e| (graph.edge(e).dst.index as usize, e))
        .collect();
    pads.sort_unstable();
    pads
}

fn batch_stream_alternatives(set: &CapsSet, streams: usize, stream: usize) -> CapsSet {
    let mut out: Vec<Caps> = Vec::new();
    for alternative in set.alternatives() {
        if let Caps::Batch { streams: batch } = alternative {
            if batch.len() == streams && !out.contains(&batch[stream]) {
                out.push(batch[stream].clone());
            }
        }
    }
    CapsSet::from_alternatives(out)
}

fn apply_batch_muxer_node<E>(
    graph: &ValidatedGraph<E>,
    node: NodeId,
    inputs: &[CapsConstraint<'_>],
    edges: &mut [Option<CapsSet>],
) -> Result<(), NegotiationFailure> {
    narrow_muxer_inputs(graph, node, inputs, edges)?;
    let out_edge = graph.out_edges(node)[0];
    let pads = pads_in_order(graph, node);
    let input_sets: Option<Vec<&[Caps]>> = pads
        .iter()
        .map(|&(_, e)| edges[e].as_ref().map(CapsSet::alternatives))
        .collect();
    if let Some(input_sets) = input_sets {
        let batches =
            batch_alternatives(&input_sets).unwrap_or_else(|| preferred_batch(&input_sets));
        narrow_edge(
            graph,
            edges,
            out_edge,
            &CapsSet::from_alternatives(batches),
            node,
        )?;
    }
    if let Some(output) = edges[out_edge].clone() {
        for (stream, &(_, e)) in pads.iter().enumerate() {
            let projected = batch_stream_alternatives(&output, pads.len(), stream);
            narrow_edge(graph, edges, e, &projected, node)?;
        }
    }
    Ok(())
}

// too many combinations to list, so each stream takes the alternative its producer prefers
fn preferred_batch(streams: &[&[Caps]]) -> Vec<Caps> {
    streams
        .iter()
        .map(|alternatives| alternatives.first().cloned())
        .collect::<Option<Vec<Caps>>>()
        .map(|streams| Caps::Batch { streams })
        .into_iter()
        .collect()
}

fn apply_batch_demux_node<E>(
    graph: &ValidatedGraph<E>,
    node: NodeId,
    in_e: usize,
    out_e: &[usize],
    input: &CapsConstraint<'_>,
    edges: &mut [Option<CapsSet>],
) -> Result<(), NegotiationFailure> {
    if let CapsConstraint::Accepts(s) = input {
        narrow_edge(graph, edges, in_e, s, node)?;
    }
    let ports = out_e.len();
    let Some(batches) = edges[in_e].clone() else {
        return Ok(());
    };
    for &oe in out_e {
        let port = graph.edge(oe).src.index as usize;
        let projected = batch_stream_alternatives(&batches, ports, port);
        narrow_edge(graph, edges, oe, &projected, node)?;
    }
    let mut port_sets: Vec<Option<CapsSet>> = alloc::vec![None; ports];
    for &oe in out_e {
        let port = graph.edge(oe).src.index as usize;
        if let Some(slot) = port_sets.get_mut(port) {
            slot.clone_from(&edges[oe]);
        }
    }
    let Some(port_sets) = port_sets.into_iter().collect::<Option<Vec<CapsSet>>>() else {
        return Ok(());
    };
    let too_many = NegotiationFailure::TooManyBatchAlternatives {
        index: node.0 as usize,
    };
    let mut narrowed: Vec<Caps> = Vec::new();
    for alternative in batches.alternatives() {
        let Caps::Batch { streams } = alternative else {
            continue;
        };
        if streams.len() != ports {
            continue;
        }
        let per_stream: Vec<Vec<Caps>> = streams
            .iter()
            .zip(&port_sets)
            .map(|(stream, port)| {
                port.alternatives()
                    .iter()
                    .filter_map(|caps| stream.intersect(caps).ok())
                    .collect()
            })
            .collect();
        let per_stream: Vec<&[Caps]> = per_stream.iter().map(Vec::as_slice).collect();
        for batch in batch_alternatives(&per_stream).ok_or(too_many.clone())? {
            if !narrowed.contains(&batch) {
                narrowed.push(batch);
            }
        }
        if narrowed.len() > crate::caps::MAX_BATCH_ALTERNATIVES {
            return Err(too_many);
        }
    }
    narrow_edge(
        graph,
        edges,
        in_e,
        &CapsSet::from_alternatives(narrowed),
        node,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "std")]
    use crate::caps::AudioFormat;
    use crate::caps::{Dim, Rate, RawVideoFormat, VideoCodec};
    #[cfg(feature = "std")]
    use crate::caps_transform::AudioShape;
    use crate::caps_transform::{CapsTransform, FieldTransform, RawVideoShape};
    use crate::runtime::passthrough::couple_passthrough;
    use alloc::boxed::Box;
    use alloc::vec;

    fn video(fmt: RawVideoFormat, w: Dim, h: Dim, r: Rate) -> Caps {
        Caps::RawVideo {
            format: fmt,
            width: w,
            height: h,
            framerate: r,
            interlace: crate::Interlace::Any,
            colorimetry: crate::Colorimetry::UNKNOWN,
        }
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
            Caps::Tensor {
                dtype: TensorDType::F32,
                shape,
                layout,
            } => CapsSet::one(Caps::Tensor {
                dtype: TensorDType::U8,
                shape: *shape,
                layout: *layout,
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        }));
        // infer: u8 [1,3,4,4] -> f32 [1,10] (the OrtInference shape).
        let logits_c = logits.clone();
        let infer = CapsConstraint::DerivedOutput(Box::new(move |inp: &Caps| match inp {
            Caps::Tensor {
                dtype: TensorDType::U8,
                ..
            } => CapsSet::one(logits_c.clone()),
            _ => CapsSet::from_alternatives(Vec::new()),
        }));
        let sink = CapsConstraint::AcceptsAny;

        let links = solve_linear(&[&src, &quant, &infer, &sink]).expect("tensor chain negotiates");
        assert_eq!(links[0], f32_in, "source link f32");
        assert_eq!(
            links[1], u8_mid,
            "quantize output is u8, not the source f32"
        );
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
            NodeConstraint::Element(CapsConstraint::DerivedOutput(Box::new(
                |inp: &Caps| match inp {
                    Caps::Tensor {
                        dtype: TensorDType::F32,
                        shape,
                        layout,
                    } => CapsSet::one(Caps::Tensor {
                        dtype: TensorDType::U8,
                        shape: *shape,
                        layout: *layout,
                    }),
                    _ => CapsSet::from_alternatives(Vec::new()),
                },
            ))),
            NodeConstraint::Element(CapsConstraint::DerivedOutput(Box::new(
                move |inp: &Caps| match inp {
                    Caps::Tensor {
                        dtype: TensorDType::U8,
                        ..
                    } => CapsSet::one(logits_c.clone()),
                    _ => CapsSet::from_alternatives(Vec::new()),
                },
            ))),
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
        Caps::CompressedVideo {
            codec,
            width: w,
            height: h,
            framerate: r,
            colorimetry: crate::Colorimetry::UNKNOWN,
            alignment: crate::BitstreamAlignment::Unspecified,
        }
    }

    fn fixed_compressed(codec: VideoCodec, w: u32, h: u32, fps: u32) -> Caps {
        compressed(codec, Dim::Fixed(w), Dim::Fixed(h), Rate::Fixed(fps << 16))
    }

    #[test]
    fn solves_source_sink_minimal_chain() {
        let src = CapsConstraint::Produces(CapsSet::one(fixed_video(
            RawVideoFormat::Nv12,
            1280,
            720,
            30,
        )));
        let sink = CapsConstraint::Accepts(CapsSet::one(video(
            RawVideoFormat::Nv12,
            Dim::Any,
            Dim::Any,
            Rate::Any,
        )));
        let links = solve_linear(&[&src, &sink]).unwrap();
        assert_eq!(
            links,
            vec![fixed_video(RawVideoFormat::Nv12, 1280, 720, 30)]
        );
    }

    #[test]
    fn empty_link_when_formats_disjoint() {
        let src = CapsConstraint::Produces(CapsSet::one(fixed_compressed(
            VideoCodec::H264,
            1280,
            720,
            30,
        )));
        let sink = CapsConstraint::Accepts(CapsSet::one(video(
            RawVideoFormat::Nv12,
            Dim::Any,
            Dim::Any,
            Rate::Any,
        )));
        assert_eq!(
            solve_linear(&[&src, &sink]),
            Err(NegotiationFailure::empty_link(0, 1))
        );
    }

    #[test]
    fn degenerate_when_fewer_than_two_elements() {
        let src =
            CapsConstraint::Produces(CapsSet::one(fixed_video(RawVideoFormat::Nv12, 1, 1, 1)));
        assert_eq!(solve_linear(&[&src]), Err(NegotiationFailure::Degenerate));
        assert_eq!(solve_linear(&[]), Err(NegotiationFailure::Degenerate));
    }

    #[test]
    fn endpoint_shape_mismatch_rejected() {
        let id = CapsConstraint::Identity(CapsSet::one(fixed_video(RawVideoFormat::Nv12, 1, 1, 1)));
        let sink =
            CapsConstraint::Accepts(CapsSet::one(fixed_video(RawVideoFormat::Nv12, 1, 1, 1)));
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
        let src =
            CapsConstraint::Produces(CapsSet::from_alternatives(vec![rgba.clone(), h264.clone()]));
        let sink =
            CapsConstraint::Accepts(CapsSet::from_alternatives(vec![h264.clone(), rgba.clone()]));
        let links = solve_linear(&[&src, &sink]).unwrap();
        // Source's outer preference wins because Produces is applied
        // first and CapsSet::intersect preserves self's order.
        assert_eq!(links, vec![rgba]);
    }

    #[test]
    fn identity_couples_input_and_output() {
        let src = CapsConstraint::Produces(CapsSet::one(fixed_video(
            RawVideoFormat::Nv12,
            1280,
            720,
            30,
        )));
        let id = CapsConstraint::Identity(CapsSet::one(video(
            RawVideoFormat::Nv12,
            Dim::Any,
            Dim::Any,
            Rate::Any,
        )));
        let sink = CapsConstraint::Accepts(CapsSet::one(video(
            RawVideoFormat::Nv12,
            Dim::Any,
            Dim::Any,
            Rate::Any,
        )));
        let links = solve_linear(&[&src, &id, &sink]).unwrap();
        assert_eq!(
            links,
            vec![
                fixed_video(RawVideoFormat::Nv12, 1280, 720, 30),
                fixed_video(RawVideoFormat::Nv12, 1280, 720, 30),
            ]
        );
    }

    #[test]
    fn identity_format_mismatch_returns_empty_link() {
        let src = CapsConstraint::Produces(CapsSet::one(fixed_compressed(
            VideoCodec::H264,
            1280,
            720,
            30,
        )));
        let id = CapsConstraint::Identity(CapsSet::one(video(
            RawVideoFormat::Nv12,
            Dim::Any,
            Dim::Any,
            Rate::Any,
        )));
        let sink = CapsConstraint::Accepts(CapsSet::one(video(
            RawVideoFormat::Nv12,
            Dim::Any,
            Dim::Any,
            Rate::Any,
        )));
        assert!(matches!(
            solve_linear(&[&src, &id, &sink]),
            Err(NegotiationFailure::EmptyLink { .. })
        ));
    }

    #[test]
    fn derived_output_evaluated_after_input_fixates() {
        // Decoder: H264 input → Nv12 output at the same dims.
        let src = CapsConstraint::Produces(CapsSet::one(fixed_compressed(
            VideoCodec::H264,
            1920,
            1080,
            60,
        )));
        let dec = CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::CompressedVideo {
                width,
                height,
                framerate,
                ..
            } => CapsSet::one(Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: width.clone(),
                height: height.clone(),
                framerate: framerate.clone(),
                interlace: crate::Interlace::Any,
                colorimetry: crate::Colorimetry::UNKNOWN,
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        }));
        let sink = CapsConstraint::Accepts(CapsSet::one(video(
            RawVideoFormat::Nv12,
            Dim::Any,
            Dim::Any,
            Rate::Any,
        )));
        let links = solve_linear(&[&src, &dec, &sink]).unwrap();
        assert_eq!(
            links,
            vec![
                fixed_compressed(VideoCodec::H264, 1920, 1080, 60),
                fixed_video(RawVideoFormat::Nv12, 1920, 1080, 60),
            ]
        );
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
            Caps::CompressedVideo {
                width,
                height,
                framerate,
                ..
            } => CapsSet::one(Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: width.clone(),
                height: height.clone(),
                framerate: framerate.clone(),
                interlace: crate::Interlace::Any,
                colorimetry: crate::Colorimetry::UNKNOWN,
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        }));
        let sink = CapsConstraint::Accepts(CapsSet::one(fixed_video(
            RawVideoFormat::Nv12,
            1280,
            720,
            30,
        )));
        let links = solve_linear(&[&src, &dec, &sink]).unwrap();
        assert_eq!(
            links,
            vec![
                fixed_compressed(VideoCodec::H264, 1280, 720, 30),
                fixed_video(RawVideoFormat::Nv12, 1280, 720, 30),
            ]
        );
    }

    #[test]
    fn derived_output_fixed_output_imposes_no_backward_narrowing() {
        // A decoder whose output is fixed regardless of input (no passthrough
        // field) must not gain spurious backward coupling: discovery finds NONE,
        // so the input keeps its produced caps. The source pins its own geometry.
        let src = CapsConstraint::Produces(CapsSet::one(fixed_compressed(
            VideoCodec::H264,
            1920,
            1080,
            30,
        )));
        let dec = CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::CompressedVideo { .. } => {
                CapsSet::one(fixed_video(RawVideoFormat::Nv12, 640, 480, 30))
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }));
        let sink = CapsConstraint::Accepts(CapsSet::one(video(
            RawVideoFormat::Nv12,
            Dim::Any,
            Dim::Any,
            Rate::Any,
        )));
        let links = solve_linear(&[&src, &dec, &sink]).unwrap();
        assert_eq!(
            links,
            vec![
                fixed_compressed(VideoCodec::H264, 1920, 1080, 30),
                fixed_video(RawVideoFormat::Nv12, 640, 480, 30),
            ]
        );
    }

    #[test]
    fn mapping_picks_compatible_pair() {
        // Codec converter declaring two pre-enumerated (in, out) pairs.
        // Source is H265 at 1280x720; the H264 pair gets filtered out.
        // Output dims come from the matching pair (mapping doesn't
        // propagate dims between paired sides — that's `DerivedOutput`'s
        // job).
        let src = CapsConstraint::Produces(CapsSet::one(fixed_compressed(
            VideoCodec::H265,
            1280,
            720,
            30,
        )));
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
            RawVideoFormat::Nv12,
            Dim::Any,
            Dim::Any,
            Rate::Any,
        )));
        let links = solve_linear(&[&src, &map, &sink]).unwrap();
        assert_eq!(links[0], fixed_compressed(VideoCodec::H265, 1280, 720, 30));
        assert_eq!(links[1], fixed_video(RawVideoFormat::Nv12, 1280, 720, 30));
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
    fn accepts_any_in_middle_position_is_silently_a_no_op() {
        // `AcceptsAny` in the middle of a native chain neither narrows
        // its input link nor its output link. The surrounding source
        // and sink fully determine the link assignments: the middle
        // element is invisible to the solver.
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
    fn identity_any_endpoint_position_rejected() {
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
        let src = CapsConstraint::Produces(CapsSet::one(fixed_compressed(
            VideoCodec::Av1,
            1280,
            720,
            30,
        )));
        let map = CapsConstraint::Mapping(vec![(
            CapsSet::one(compressed(VideoCodec::H264, Dim::Any, Dim::Any, Rate::Any)),
            CapsSet::one(video(RawVideoFormat::Nv12, Dim::Any, Dim::Any, Rate::Any)),
        )]);
        let sink = CapsConstraint::Accepts(CapsSet::one(video(
            RawVideoFormat::Nv12,
            Dim::Any,
            Dim::Any,
            Rate::Any,
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
        let src =
            CapsConstraint::Produces(CapsSet::one(fixed_video(RawVideoFormat::Rgba8, 64, 64, 30)));
        let id = CapsConstraint::IdentityAny;
        let sink = CapsConstraint::Accepts(CapsSet::one(video(
            RawVideoFormat::Nv12,
            Dim::Any,
            Dim::Any,
            Rate::Any,
        )));
        let feas = downstream_feasibility(&[&src, &id, &sink]);
        // Two links. Both carry the sink's NV12 set (IdentityAny couples them);
        // neither is narrowed to the source's RGBA.
        assert_eq!(feas.len(), 2);
        assert!(feas[1].as_ref().unwrap().accepts(&video(
            RawVideoFormat::Nv12,
            Dim::Any,
            Dim::Any,
            Rate::Any,
        )));
        assert!(feas[0].as_ref().unwrap().accepts(&video(
            RawVideoFormat::Nv12,
            Dim::Any,
            Dim::Any,
            Rate::Any,
        )));
        assert!(!feas[0].as_ref().unwrap().accepts(&video(
            RawVideoFormat::Rgba8,
            Dim::Any,
            Dim::Any,
            Rate::Any,
        )));
    }

    /// `resolve_forward_output` steers a format converter toward the one
    /// output its downstream accepts, defers when there is no concrete
    /// downstream set, and rejects loud when no output can satisfy it. A
    /// format-only converter is a `DerivedOutput` so it carries the input's
    /// concrete geometry into its output (a static `Mapping` with `Any` dims
    /// can't fixate and would `Defer`).
    #[cfg(feature = "std")]
    #[test]
    fn resolve_forward_output_steers_defers_and_rejects() {
        // Converter: any raw input -> {same format, NV12} at the input's dims.
        let conv = CapsConstraint::DerivedOutput(Box::new(|input: &Caps| {
            let Caps::RawVideo {
                format,
                width,
                height,
                framerate,
                ..
            } = input
            else {
                return CapsSet::from_alternatives(vec![]);
            };
            CapsSet::from_alternatives(vec![
                video(*format, width.clone(), height.clone(), framerate.clone()),
                video(
                    RawVideoFormat::Nv12,
                    width.clone(),
                    height.clone(),
                    framerate.clone(),
                ),
            ])
        }));
        let i420 = fixed_video(RawVideoFormat::I420, 64, 64, 30);
        let nv12_set = CapsSet::one(video(RawVideoFormat::Nv12, Dim::Any, Dim::Any, Rate::Any));

        // Steered: downstream accepts only NV12, so the runner picks NV12.
        match resolve_forward_output(&conv, &i420, Some(&nv12_set), None) {
            ForwardResolve::Fixed(c) => {
                assert_eq!(
                    c,
                    video(
                        RawVideoFormat::Nv12,
                        Dim::Fixed(64),
                        Dim::Fixed(64),
                        Rate::Fixed(30 << 16)
                    )
                );
            }
            other => panic!("expected Fixed(NV12), got {other:?}"),
        }

        // No concrete downstream set, but the output is ambiguous ({same, NV12}):
        // defer to the element's own process.
        assert_eq!(
            resolve_forward_output(&conv, &i420, None, None),
            ForwardResolve::Defer
        );

        // No downstream snapshot but an UNAMBIGUOUS output: a property-driven
        // converter forwards its single output (RGBA8) rather than leaking the
        // input format. This is what lets a strict downstream (a textoverlay
        // after `mp4src ! avdec ! videoconvert`) see the converted caps on a
        // mid-stream change instead of the decoder's NV12.
        let to_rgba = CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::RawVideo {
                width,
                height,
                framerate,
                ..
            } => CapsSet::one(video(
                RawVideoFormat::Rgba8,
                width.clone(),
                height.clone(),
                framerate.clone(),
            )),
            _ => CapsSet::from_alternatives(vec![]),
        }));
        let nv12_in = fixed_video(RawVideoFormat::Nv12, 64, 64, 30);
        match resolve_forward_output(&to_rgba, &nv12_in, None, None) {
            ForwardResolve::Fixed(c) => assert_eq!(
                c,
                video(
                    RawVideoFormat::Rgba8,
                    Dim::Fixed(64),
                    Dim::Fixed(64),
                    Rate::Fixed(30 << 16)
                )
            ),
            other => panic!("expected Fixed(RGBA8), got {other:?}"),
        }

        // Downstream accepts only Bgra8, which the converter cannot emit: loud.
        let bgra_set = CapsSet::one(video(RawVideoFormat::Bgra8, Dim::Any, Dim::Any, Rate::Any));
        assert!(matches!(
            resolve_forward_output(&conv, &i420, Some(&bgra_set), None),
            ForwardResolve::Infeasible(NegotiationFailure::EmptyLink { .. })
        ));
    }

    /// A converter that retargets the channel count keeps producing its target
    /// when the real layout arrives mid-stream. The `wavparse ! audioconvert
    /// channels=1 ! adpcmenc` regression: the WAV's stereo `fmt ` chunk lands
    /// after negotiation fixated on the mono placeholder, and deferring there
    /// forwards the converter's stereo INPUT to a mono-only encoder.
    #[cfg(feature = "std")]
    #[test]
    fn resolve_forward_output_keeps_a_retargeted_channel_count() {
        let audio = |channels: u8, sample_rate: u32| Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels,
            sample_rate,
            channel_layout: crate::ChannelLayout::UNSPECIFIED,
        };
        let to_mono = CapsConstraint::DerivedFields(CapsTransform::Audio {
            accept: vec![AudioFormat::PcmS16Le],
            produce: Vec::new(),
            shapes: vec![AudioShape::PASSTHROUGH
                .with_format(FieldTransform::Fixed(AudioFormat::PcmS16Le))
                .with_channels(FieldTransform::Fixed(1))],
        });
        // Solved output, then the source's real (stereo) layout mid-stream.
        let previous = audio(1, 48_000);
        match resolve_forward_output(&to_mono, &audio(2, 8_000), None, Some(&previous)) {
            ForwardResolve::Fixed(c) => assert_eq!(c, audio(1, 8_000)),
            other => panic!("expected Fixed(mono), got {other:?}"),
        }
    }

    /// A rescaler's single output is forwarded even when an input field it never
    /// learned leaves that output unfixatable. The `filesrc ! qtdemux ! h264parse
    /// ! ffmpegdec ! videoscale ! video/x-raw,width=640,height=640` regression:
    /// the decoder announces its geometry before it knows the framerate, and
    /// deferring on that first event forwards the scaler's 1280x720 INPUT, which
    /// the capsfilter then rightly rejects.
    #[cfg(feature = "std")]
    #[test]
    fn resolve_forward_output_forwards_an_unfixatable_single_output() {
        // Rescaler: any raw input -> the same format at a fixed 640x640, the
        // input's framerate carried through (unfixed included).
        let scale = CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::RawVideo {
                format, framerate, ..
            } => CapsSet::one(video(
                *format,
                Dim::Fixed(640),
                Dim::Fixed(640),
                framerate.clone(),
            )),
            _ => CapsSet::from_alternatives(vec![]),
        }));
        // The decoder's first mid-stream caps: real geometry, framerate not yet
        // known.
        let no_rate = video(
            RawVideoFormat::I420,
            Dim::Fixed(1280),
            Dim::Fixed(720),
            Rate::Any,
        );
        let downstream = CapsSet::one(video(
            RawVideoFormat::I420,
            Dim::Fixed(640),
            Dim::Fixed(640),
            Rate::Any,
        ));
        match resolve_forward_output(&scale, &no_rate, Some(&downstream), None) {
            ForwardResolve::Fixed(c) => assert_eq!(
                c,
                video(
                    RawVideoFormat::I420,
                    Dim::Fixed(640),
                    Dim::Fixed(640),
                    Rate::Any
                ),
                "the scaler's own output geometry, not its input's"
            ),
            other => panic!("expected Fixed(640x640), got {other:?}"),
        }
    }

    /// The same unfixatable single output with no downstream snapshot, which a
    /// wildcard sink below a chain of `DerivedOutput` elements leaves empty.
    /// Deferring forwards a letterboxing `videobox`'s INPUT, which the
    /// fixed-size consumer below it then rejects.
    #[cfg(feature = "std")]
    #[test]
    fn resolve_forward_output_forwards_a_derived_single_output_without_a_snapshot() {
        // Letterbox: any raw input -> same format, 24 rows taller, input's rate.
        let letterbox = CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::RawVideo {
                format,
                width,
                height: Dim::Fixed(h),
                framerate,
                ..
            } => CapsSet::one(video(
                *format,
                width.clone(),
                Dim::Fixed(h + 24),
                framerate.clone(),
            )),
            _ => CapsSet::from_alternatives(vec![]),
        }));
        let no_rate = video(
            RawVideoFormat::Rgba8,
            Dim::Fixed(640),
            Dim::Fixed(360),
            Rate::Any,
        );
        match resolve_forward_output(&letterbox, &no_rate, None, None) {
            ForwardResolve::Fixed(c) => assert_eq!(
                c,
                video(
                    RawVideoFormat::Rgba8,
                    Dim::Fixed(640),
                    Dim::Fixed(384),
                    Rate::Any
                ),
                "the boxer's own output geometry, not its input's"
            ),
            other => panic!("expected Fixed(640x384), got {other:?}"),
        }
    }

    /// An ambiguous producible set with no downstream snapshot keeps the shape
    /// the element already produces (its previous output with re-derived
    /// geometry), and defers only when that shape is no longer producible. The
    /// `filesrc ! decodebin ! videoconvert ! textoverlay` regression: the
    /// converter's mid-stream re-solve must forward RGBA (its negotiated
    /// output) at the new geometry, not its I420 input.
    #[cfg(feature = "std")]
    #[test]
    fn resolve_forward_output_keeps_previous_shape() {
        // Converter: any raw input -> {same format, NV12} at the input's dims.
        let conv = CapsConstraint::DerivedOutput(Box::new(|input: &Caps| {
            let Caps::RawVideo {
                format,
                width,
                height,
                framerate,
                ..
            } = input
            else {
                return CapsSet::from_alternatives(vec![]);
            };
            CapsSet::from_alternatives(vec![
                video(*format, width.clone(), height.clone(), framerate.clone()),
                video(
                    RawVideoFormat::Nv12,
                    width.clone(),
                    height.clone(),
                    framerate.clone(),
                ),
            ])
        }));
        let i420_big = fixed_video(RawVideoFormat::I420, 1920, 1080, 30);

        // Startup produced NV12 at the placeholder geometry: the re-solve keeps
        // NV12 and takes the refined dims.
        let prev = fixed_video(RawVideoFormat::Nv12, 16, 16, 1);
        match resolve_forward_output(&conv, &i420_big, None, Some(&prev)) {
            ForwardResolve::Fixed(c) => assert_eq!(
                c,
                video(
                    RawVideoFormat::Nv12,
                    Dim::Fixed(1920),
                    Dim::Fixed(1080),
                    Rate::Fixed(30 << 16)
                )
            ),
            other => panic!("expected Fixed(NV12 at new dims), got {other:?}"),
        }

        // The previous shape is no longer producible: defer to the element.
        let prev_gone = fixed_video(RawVideoFormat::Bgra8, 16, 16, 1);
        assert_eq!(
            resolve_forward_output(&conv, &i420_big, None, Some(&prev_gone)),
            ForwardResolve::Defer
        );

        // A field the candidates leave open must not be invented: a decoder
        // whose produce set carries no framerate (the vorbisdec regression, an
        // unpinned sample rate steered to a default 48000) defers so the
        // element's own `process` emits the real value.
        let open_rate = CapsConstraint::DerivedOutput(Box::new(|_input: &Caps| {
            CapsSet::from_alternatives(vec![
                video(
                    RawVideoFormat::Nv12,
                    Dim::Fixed(1920),
                    Dim::Fixed(1080),
                    Rate::Any,
                ),
                video(
                    RawVideoFormat::I420,
                    Dim::Fixed(1920),
                    Dim::Fixed(1080),
                    Rate::Any,
                ),
            ])
        }));
        assert_eq!(
            resolve_forward_output(&open_rate, &i420_big, None, Some(&prev)),
            ForwardResolve::Defer
        );

        // With a downstream snapshot admitting both, the previous shape still
        // orders the fixation.
        let both = CapsSet::from_alternatives(vec![
            video(RawVideoFormat::I420, Dim::Any, Dim::Any, Rate::Any),
            video(RawVideoFormat::Nv12, Dim::Any, Dim::Any, Rate::Any),
        ]);
        match resolve_forward_output(&conv, &i420_big, Some(&both), Some(&prev)) {
            ForwardResolve::Fixed(Caps::RawVideo { format, .. }) => {
                assert_eq!(format, RawVideoFormat::Nv12);
            }
            other => panic!("expected Fixed(NV12), got {other:?}"),
        }
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

        assert_eq!(
            dag, linear,
            "DAG solver matches the linear solver byte-for-byte"
        );
        assert_eq!(dag, vec![rgba, nv12]);
    }

    #[test]
    fn solve_graph_empty_link_carries_both_sides_sets() {
        // source produces RGBA only, sink accepts NV12 only: the failure must
        // name the two nodes *and* what each of them still allowed.
        let rgba = fixed_video(RawVideoFormat::Rgba8, 64, 48, 30);
        let nv12 = fixed_video(RawVideoFormat::Nv12, 64, 48, 30);
        let cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(rgba.clone()))),
            NodeConstraint::Element(CapsConstraint::Accepts(CapsSet::one(nv12.clone()))),
        ];
        let mut g: Graph<()> = Graph::new();
        let src = g.add_source(());
        let sink = g.add_sink(());
        g.link(src, sink).unwrap();
        let v = g.finish().unwrap();

        let err = solve_graph(&v, &cs).expect_err("RGBA source vs NV12 sink cannot solve");
        assert!(matches!(
            err,
            NegotiationFailure::EmptyLink {
                upstream: 0,
                downstream: 1,
                ..
            }
        ));
        let c = err.conflict().expect("both candidate sets captured");
        assert_eq!(c.upstream, CapsSet::one(rgba));
        assert_eq!(c.downstream, CapsSet::one(nv12));
    }

    #[test]
    fn solve_graph_names_the_transform_whose_derived_output_is_empty() {
        // An encoder that takes NV12 only, fed RGBA: its output link has
        // nothing to carry, and the wildcard sink below it is not to blame.
        let rgba = fixed_video(RawVideoFormat::Rgba8, 64, 48, 30);
        let nv12_only = CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                ..
            } => CapsSet::one(input.clone()),
            _ => CapsSet::from_alternatives(Vec::new()),
        }));
        let cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(rgba.clone()))),
            NodeConstraint::Element(nv12_only),
            NodeConstraint::Element(CapsConstraint::AcceptsAny),
        ];
        let mut g: Graph<()> = Graph::new();
        let src = g.add_source(());
        let encoder = g.add_transform(());
        let sink = g.add_sink(());
        g.link(src, encoder).unwrap();
        g.link(encoder, sink).unwrap();
        let v = g.finish().unwrap();

        let err = solve_graph(&v, &cs).expect_err("RGBA into an NV12-only encoder");
        assert_eq!(
            err,
            NegotiationFailure::NoOutputForInput {
                upstream: 1,
                downstream: 2,
                input: CapsSet::one(rgba),
            }
        );
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
        assert!(
            sol.iter().all(|c| *c == nv12_fixed),
            "every branch carries the source caps"
        );
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
            matches!(
                solve_graph(&v, &cs),
                Err(NegotiationFailure::EmptyLink { .. })
            ),
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
        let h264 = compressed(
            VideoCodec::H264,
            Dim::Fixed(64),
            Dim::Fixed(48),
            Rate::Fixed(30 << 16),
        );
        let h265 = compressed(
            VideoCodec::H265,
            Dim::Fixed(64),
            Dim::Fixed(48),
            Rate::Fixed(30 << 16),
        );
        let h264_any = compressed(VideoCodec::H264, Dim::Any, Dim::Any, Rate::Any);
        let h265_any = compressed(VideoCodec::H265, Dim::Any, Dim::Any, Rate::Any);
        let muxed = compressed(
            VideoCodec::H264,
            Dim::Fixed(64),
            Dim::Fixed(48),
            Rate::Fixed(30 << 16),
        );

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
        assert_eq!(
            sol,
            vec![h264, h265, muxed],
            "each input narrowed by its pad, output by produce"
        );
    }

    #[test]
    fn solve_graph_muxer_follows_input_derives_output() {
        // An identity-passthrough mux (overlay): a video pad 0, a sidecar pad 1,
        // output follows pad 0. The output edge must equal the video source's caps
        // even though no output caps were declared (`output` is a placeholder).
        let rgba = fixed_video(RawVideoFormat::Rgba8, 320, 240, 30);
        let rgba_any = video(RawVideoFormat::Rgba8, Dim::Any, Dim::Any, Rate::Any);
        let text = Caps::Text {
            format: crate::caps::TextFormat::Utf8,
        };

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
        assert_eq!(
            sol[2], rgba,
            "output edge follows the video pad's negotiated caps"
        );
        assert_eq!(sol[0], rgba, "video pad edge unchanged");
        assert_eq!(sol[1], text, "text pad edge unchanged");
    }

    #[test]
    fn solve_graph_muxer_wildcard_inputs_forward_source_caps() {
        // The `InterleaveMux` shape: every input pad is `AcceptsAny` (frames
        // carry their own caps), the output `Produces` a fixed merged caps. The
        // wildcard inputs impose no narrowing, so each input edge keeps its
        // source's caps and the output edge takes the produced caps.
        let h264 = compressed(
            VideoCodec::H264,
            Dim::Fixed(64),
            Dim::Fixed(48),
            Rate::Fixed(30 << 16),
        );
        let aac = Caps::Audio {
            format: crate::caps::AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: crate::ChannelLayout::UNSPECIFIED,
        };
        let merged = compressed(
            VideoCodec::H264,
            Dim::Fixed(64),
            Dim::Fixed(48),
            Rate::Fixed(30 << 16),
        );

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
        assert!(tee_in
            .intersect(&CapsSet::one(nv12_fixed.clone()))
            .fixate()
            .is_some());
        // the tee input cannot carry an off-geometry frame both branches reject.
        let off = fixed_video(RawVideoFormat::Nv12, 99, 99, 30);
        assert!(
            tee_in.intersect(&CapsSet::one(off)).is_empty(),
            "branch B pins 64x48"
        );
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
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(fixed_compressed(
                VideoCodec::H264,
                64,
                48,
                30,
            )))),
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(fixed_compressed(
                VideoCodec::H265,
                64,
                48,
                30,
            )))),
            NodeConstraint::Muxer {
                inputs: vec![
                    CapsConstraint::Accepts(CapsSet::one(h264_any.clone())),
                    CapsConstraint::Accepts(CapsSet::one(h265_any.clone())),
                ],
                output: CapsConstraint::Produces(CapsSet::one(fixed_compressed(
                    VideoCodec::H264,
                    64,
                    48,
                    30,
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
        assert_eq!(
            feas[0],
            Some(CapsSet::one(h264_any)),
            "pad 0 feasibility = its accept set"
        );
        assert_eq!(
            feas[1],
            Some(CapsSet::one(h265_any)),
            "pad 1 feasibility = its accept set"
        );
        assert_eq!(
            feas[2], None,
            "wildcard sink leaves the muxer output unconstrained"
        );
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
            Caps::CompressedVideo {
                width,
                height,
                framerate,
                ..
            } => CapsSet::one(Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: width.clone(),
                height: height.clone(),
                framerate: framerate.clone(),
                interlace: crate::Interlace::Any,
                colorimetry: crate::Colorimetry::UNKNOWN,
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        };
        let cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(compressed(
                VideoCodec::H264,
                Dim::Any,
                Dim::Any,
                Rate::Fixed(30 << 16),
            )))),
            NodeConstraint::Element(CapsConstraint::DerivedOutput(Box::new(dec_closure))),
            NodeConstraint::Element(CapsConstraint::Accepts(CapsSet::one(fixed_video(
                RawVideoFormat::Nv12,
                1280,
                720,
                30,
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
        let dec_in = feas[0]
            .as_ref()
            .expect("decoder input edge is now constrained");
        assert!(
            dec_in
                .intersect(&CapsSet::one(fixed_compressed(
                    VideoCodec::H264,
                    1280,
                    720,
                    30
                )))
                .fixate()
                .is_some(),
            "pinned 1280x720 couples back onto the H264 input edge",
        );
        let off = fixed_compressed(VideoCodec::H264, 640, 480, 30);
        assert!(
            dec_in.intersect(&CapsSet::one(off)).is_empty(),
            "off-geometry input is rejected by the snapshot"
        );
    }

    // --- M227 field-level bidirectional caps coupling ---

    /// A scale-like `DerivedFields`: passthrough format + framerate, retarget
    /// geometry to [passthrough-input, Range 1..32768].
    fn scale_like<'a>() -> CapsConstraint<'a> {
        let range = Dim::Range { min: 1, max: 32768 };
        CapsConstraint::DerivedFields(CapsTransform::RawVideo {
            accept: Vec::new(),
            produce: Vec::new(),
            shapes: vec![
                RawVideoShape::PASSTHROUGH,
                RawVideoShape::PASSTHROUGH
                    .with_width(FieldTransform::Fixed(range.clone()))
                    .with_height(FieldTransform::Fixed(range)),
            ],
        })
    }

    /// A convert-like `DerivedFields`: passthrough geometry + framerate,
    /// retarget format to [Rgba8, Nv12] (Rgba8 preferred).
    fn convert_like<'a>() -> CapsConstraint<'a> {
        CapsConstraint::DerivedFields(CapsTransform::RawVideo {
            accept: Vec::new(),
            produce: Vec::new(),
            shapes: vec![
                RawVideoShape::PASSTHROUGH
                    .with_format(FieldTransform::Fixed(RawVideoFormat::Rgba8)),
                RawVideoShape::PASSTHROUGH.with_format(FieldTransform::Fixed(RawVideoFormat::Nv12)),
            ],
        })
    }

    #[test]
    fn couple_passthrough_narrows_a_range_field_within_an_alternative() {
        // The primitive the alternative-drop walk can't express: a Range width
        // meeting a Fixed pin collapses to Fixed, format (retargeted) untouched.
        let mask = PassthroughFields::NONE
            .with_width()
            .with_height()
            .with_framerate();
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
        let src = CapsConstraint::Produces(CapsSet::one(fixed_video(
            RawVideoFormat::Rgba8,
            320,
            240,
            30,
        )));
        let scale = scale_like();
        let convert = convert_like();
        let sink = CapsConstraint::Accepts(CapsSet::one(fixed_video(
            RawVideoFormat::Nv12,
            160,
            120,
            30,
        )));
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
        let src = CapsConstraint::Produces(CapsSet::one(fixed_video(
            RawVideoFormat::Rgba8,
            320,
            240,
            30,
        )));
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
        let src = CapsConstraint::Produces(CapsSet::one(fixed_video(
            RawVideoFormat::Rgba8,
            320,
            240,
            30,
        )));
        let convert = convert_like();
        let sink = CapsConstraint::Accepts(CapsSet::one(fixed_video(
            RawVideoFormat::Nv12,
            160,
            120,
            30,
        )));
        assert!(
            solve_linear(&[&src, &convert, &sink]).is_err(),
            "geometry pin must fail loud"
        );
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
            (1..=6)
                .map(|w| fixed_video(RawVideoFormat::Rgba8, w * 16, 48, 30))
                .collect(),
        );
        assert!(fmt_set(&wide).contains("(+2 more"), "{}", fmt_set(&wide));

        // Constraint summaries name the shape.
        assert!(
            fmt_caps_constraint(&CapsConstraint::Produces(CapsSet::one(rgba.clone())))
                .starts_with("produces ")
        );
        assert_eq!(
            fmt_caps_constraint(&CapsConstraint::AcceptsAny),
            "accepts ANY"
        );
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
    #[test]
    fn solve_graph_batch_mux_and_demux_negotiate_each_stream() {
        // branch 0 only takes NV12, which source 0 offers second
        let rgba = fixed_video(RawVideoFormat::Rgba8, 320, 240, 30);
        let nv12 = fixed_video(RawVideoFormat::Nv12, 320, 240, 30);
        let i420 = fixed_video(RawVideoFormat::I420, 640, 480, 25);
        let nv12_any = video(RawVideoFormat::Nv12, Dim::Any, Dim::Any, Rate::Any);
        let cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::from_alternatives(vec![
                rgba,
                nv12.clone(),
            ]))),
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(i420.clone()))),
            NodeConstraint::BatchMuxer {
                inputs: vec![CapsConstraint::AcceptsAny, CapsConstraint::AcceptsAny],
            },
            NodeConstraint::BatchDemux {
                input: CapsConstraint::AcceptsAny,
            },
            NodeConstraint::Element(CapsConstraint::Accepts(CapsSet::one(nv12_any))),
            NodeConstraint::Element(CapsConstraint::AcceptsAny),
        ];
        let mut g: Graph<()> = Graph::new();
        let first = g.add_source(());
        let second = g.add_source(());
        let mux = g.add_muxer((), 2);
        let demux = g.add_tee(2);
        let branch0 = g.add_sink(());
        let branch1 = g.add_sink(());
        g.link(first, mux.input(0)).unwrap();
        g.link(second, mux.input(1)).unwrap();
        g.link(mux.output(), demux.input()).unwrap();
        g.link(demux.out(0), branch0).unwrap();
        g.link(demux.out(1), branch1).unwrap();
        let v = g.finish().unwrap();

        let sol = solve_graph(&v, &cs).expect("batch mux and demux solve");
        assert_eq!(sol[0], nv12, "branch 0's NV12 reached source 0");
        assert_eq!(sol[1], i420);
        assert_eq!(
            sol[2],
            Caps::Batch {
                streams: vec![nv12.clone(), i420.clone()]
            }
        );
        assert_eq!(sol[3], nv12, "port 0 carries stream 0");
        assert_eq!(sol[4], i420, "port 1 carries stream 1");
    }

    #[test]
    fn solve_graph_batch_demux_needs_one_port_per_stream() {
        let rgba = fixed_video(RawVideoFormat::Rgba8, 320, 240, 30);
        let cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(Caps::Batch {
                streams: vec![rgba.clone(), rgba],
            }))),
            NodeConstraint::BatchDemux {
                input: CapsConstraint::AcceptsAny,
            },
            NodeConstraint::Element(CapsConstraint::AcceptsAny),
            NodeConstraint::Element(CapsConstraint::AcceptsAny),
            NodeConstraint::Element(CapsConstraint::AcceptsAny),
        ];
        let mut g: Graph<()> = Graph::new();
        let src = g.add_source(());
        let demux = g.add_tee(3);
        g.link(src, demux.input()).unwrap();
        for port in 0..3 {
            let sink = g.add_sink(());
            g.link(demux.out(port), sink).unwrap();
        }
        let v = g.finish().unwrap();
        assert!(matches!(
            solve_graph(&v, &cs),
            Err(NegotiationFailure::EmptyLink { .. })
        ));
    }

    #[test]
    fn solve_graph_batch_mux_settles_each_stream_past_the_combination_cap() {
        // Three inputs of seven alternatives each is 343 batches, past the cap.
        let alternatives = |format| {
            CapsSet::from_alternatives(
                (1..=7)
                    .map(|i| fixed_video(format, 16 * i, 16, 30))
                    .collect(),
            )
        };
        let cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::Produces(alternatives(RawVideoFormat::Nv12))),
            NodeConstraint::Element(CapsConstraint::Produces(alternatives(RawVideoFormat::I420))),
            NodeConstraint::Element(CapsConstraint::Produces(alternatives(
                RawVideoFormat::Rgba8,
            ))),
            NodeConstraint::BatchMuxer {
                inputs: (0..3).map(|_| CapsConstraint::AcceptsAny).collect(),
            },
            NodeConstraint::Element(CapsConstraint::AcceptsAny),
        ];
        let mut g: Graph<()> = Graph::new();
        let sources: Vec<_> = (0..3).map(|_| g.add_source(())).collect();
        let mux = g.add_muxer((), 3);
        let sink = g.add_sink(());
        for (pad, src) in sources.into_iter().enumerate() {
            g.link(src, mux.input(pad as u8)).unwrap();
        }
        g.link(mux.output(), sink).unwrap();
        let v = g.finish().unwrap();
        let preferred = [
            RawVideoFormat::Nv12,
            RawVideoFormat::I420,
            RawVideoFormat::Rgba8,
        ]
        .map(|format| fixed_video(format, 16, 16, 30));
        let sol = solve_graph(&v, &cs).expect("the batch settles past the cap");
        assert_eq!(
            &sol[..3],
            &preferred,
            "each source keeps its first alternative"
        );
        assert_eq!(
            sol[3],
            Caps::Batch {
                streams: preferred.to_vec()
            }
        );
    }

    #[test]
    fn linear_failures_name_the_elements_that_disagree() {
        let nv12 = fixed_video(RawVideoFormat::Nv12, 64, 48, 30);
        let i420 = fixed_video(RawVideoFormat::I420, 64, 48, 30);
        let both = CapsSet::from_alternatives(vec![nv12.clone(), i420.clone()]);
        let only = |caps: &Caps| CapsSet::one(caps.clone());
        let produces = |caps: &Caps| CapsConstraint::Produces(CapsSet::one(caps.clone()));
        let accepts = |caps: &Caps| CapsConstraint::Accepts(CapsSet::one(caps.clone()));
        let derives = |caps: Option<Caps>| {
            CapsConstraint::DerivedOutput(Box::new(move |_: &Caps| {
                CapsSet::from_alternatives(caps.iter().cloned().collect())
            }))
        };
        let nv12_only_passthrough = CapsConstraint::DerivedFields(CapsTransform::RawVideo {
            accept: vec![RawVideoFormat::Nv12],
            produce: Vec::new(),
            shapes: vec![RawVideoShape::PASSTHROUGH],
        });
        let passthrough = CapsConstraint::DerivedFields(CapsTransform::RawVideo {
            accept: Vec::new(),
            produce: Vec::new(),
            shapes: vec![RawVideoShape::PASSTHROUGH],
        });
        let open_geometry = video(RawVideoFormat::Nv12, Dim::Any, Dim::Any, Rate::Any);

        let cases: Vec<(&str, Vec<CapsConstraint>, NegotiationFailure)> = vec![
            (
                "the last element produces",
                vec![
                    produces(&nv12),
                    CapsConstraint::IdentityAny,
                    produces(&nv12),
                ],
                NegotiationFailure::EndpointShapeMismatch { index: 2 },
            ),
            (
                "an open geometry cannot fixate",
                vec![produces(&open_geometry), CapsConstraint::AcceptsAny],
                NegotiationFailure::Unfixable {
                    upstream: 0,
                    downstream: 1,
                },
            ),
            (
                "an identity rejects its input",
                vec![
                    produces(&nv12),
                    CapsConstraint::Identity(only(&i420)),
                    CapsConstraint::AcceptsAny,
                ],
                NegotiationFailure::empty_link(0, 1),
            ),
            (
                "an identity cannot feed its sink",
                vec![
                    CapsConstraint::Produces(both.clone()),
                    CapsConstraint::Identity(only(&nv12)),
                    accepts(&i420),
                ],
                NegotiationFailure::empty_link(1, 2),
            ),
            (
                "an identity's two sides never meet",
                vec![
                    produces(&nv12),
                    CapsConstraint::Identity(both.clone()),
                    accepts(&i420),
                ],
                NegotiationFailure::empty_link(0, 2),
            ),
            (
                "no mapping row takes the input",
                vec![
                    produces(&nv12),
                    CapsConstraint::Mapping(vec![(only(&i420), only(&nv12))]),
                    CapsConstraint::AcceptsAny,
                ],
                NegotiationFailure::empty_link(0, 2),
            ),
            (
                "a derived output is empty",
                vec![produces(&nv12), derives(None), CapsConstraint::AcceptsAny],
                NegotiationFailure::empty_link(1, 2),
            ),
            (
                "a derived output misses the sink",
                vec![produces(&nv12), derives(Some(nv12.clone())), accepts(&i420)],
                NegotiationFailure::empty_link(1, 2),
            ),
            (
                "a declared transform refuses its input",
                vec![
                    produces(&i420),
                    nv12_only_passthrough,
                    CapsConstraint::AcceptsAny,
                ],
                NegotiationFailure::empty_link(1, 2),
            ),
            (
                "a declared transform misses the sink",
                vec![produces(&nv12), passthrough, accepts(&i420)],
                NegotiationFailure::empty_link(1, 2),
            ),
            (
                "a wildcard identity's two sides never meet",
                vec![produces(&nv12), CapsConstraint::IdentityAny, accepts(&i420)],
                NegotiationFailure::empty_link(0, 2),
            ),
        ];
        for (case, chain, expected) in cases {
            let refs: Vec<&CapsConstraint> = chain.iter().collect();
            assert_eq!(solve_linear(&refs), Err(expected), "{case}");
        }
    }

    #[test]
    fn identity_preferences_steer_the_chain() {
        let nv12 = fixed_video(RawVideoFormat::Nv12, 64, 48, 30);
        let i420 = fixed_video(RawVideoFormat::I420, 64, 48, 30);
        let both = CapsSet::from_alternatives(vec![nv12.clone(), i420.clone()]);
        let src = CapsConstraint::Produces(both.clone());
        let identity = CapsConstraint::Identity(both);
        let sink = CapsConstraint::AcceptsAny;
        // I420 costs the source one, NV12 costs the identity ten
        let identity_costs = CapsPreferences::new(vec![10, 0]);
        let links = solve_linear_preferred(
            &[&src, &identity, &sink],
            &[None, Some(identity_costs), None],
        )
        .expect("chain negotiates");
        assert_eq!(links, vec![i420.clone(), i420]);
    }

    #[test]
    fn mapping_preferences_charge_the_row_both_sides_select() {
        let nv12 = fixed_video(RawVideoFormat::Nv12, 64, 48, 30);
        let to_rgba = fixed_video(RawVideoFormat::Rgba8, 64, 48, 30);
        let to_bgra = fixed_video(RawVideoFormat::Bgra8, 64, 48, 30);
        let src = CapsConstraint::Produces(CapsSet::one(nv12.clone()));
        let map = CapsConstraint::Mapping(vec![
            (CapsSet::one(nv12.clone()), CapsSet::one(to_rgba.clone())),
            (CapsSet::one(nv12.clone()), CapsSet::one(to_bgra.clone())),
        ]);
        let sink =
            CapsConstraint::Accepts(CapsSet::from_alternatives(vec![to_rgba, to_bgra.clone()]));
        // BGRA costs the sink one, the RGBA row costs ten
        let row_costs = CapsPreferences::new(vec![10, 0]);
        let links = solve_linear_preferred(&[&src, &map, &sink], &[None, Some(row_costs), None])
            .expect("chain negotiates");
        assert_eq!(links, vec![nv12, to_bgra]);
    }

    #[test]
    fn preferences_never_pair_a_derived_input_with_an_output_it_cannot_produce() {
        let nv12 = fixed_video(RawVideoFormat::Nv12, 64, 48, 30);
        let i420 = fixed_video(RawVideoFormat::I420, 64, 48, 30);
        let both = CapsSet::from_alternatives(vec![nv12.clone(), i420.clone()]);
        // the sink's NV12 costs more than the source's I420
        let sink_costs = CapsPreferences::new(vec![10, 0]);
        let passthrough_closure =
            CapsConstraint::DerivedOutput(Box::new(|input: &Caps| CapsSet::one(input.clone())));
        let passthrough_declared = CapsConstraint::DerivedFields(CapsTransform::RawVideo {
            accept: Vec::new(),
            produce: Vec::new(),
            shapes: vec![RawVideoShape::PASSTHROUGH],
        });
        let src = CapsConstraint::Produces(both.clone());
        let sink = CapsConstraint::Accepts(both);
        for transform in [&passthrough_closure, &passthrough_declared] {
            let links = solve_linear_preferred(
                &[&src, transform, &sink],
                &[None, None, Some(sink_costs.clone())],
            )
            .expect("chain negotiates");
            assert_eq!(links, vec![i420.clone(), i420.clone()]);
        }
    }

    #[test]
    fn derived_output_drops_inputs_that_cannot_reach_the_sink() {
        let h264 = fixed_compressed(VideoCodec::H264, 64, 48, 30);
        let h265 = fixed_compressed(VideoCodec::H265, 64, 48, 30);
        let nv12 = fixed_video(RawVideoFormat::Nv12, 320, 240, 30);
        let i420 = fixed_video(RawVideoFormat::I420, 320, 240, 30);
        let (h264_out, h265_out) = (nv12, i420.clone());
        let decoder = CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            } => CapsSet::one(h264_out.clone()),
            _ => CapsSet::one(h265_out.clone()),
        }));
        let src = CapsConstraint::Produces(CapsSet::from_alternatives(vec![h264, h265.clone()]));
        let sink = CapsConstraint::Accepts(CapsSet::one(i420.clone()));
        let links = solve_linear(&[&src, &decoder, &sink]).expect("chain negotiates");
        assert_eq!(links, vec![h265, i420]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn resolve_forward_output_maps_the_input_through_its_row() {
        let nv12 = fixed_video(RawVideoFormat::Nv12, 64, 48, 30);
        let i420 = fixed_video(RawVideoFormat::I420, 64, 48, 30);
        let rgba = fixed_video(RawVideoFormat::Rgba8, 64, 48, 30);
        let bgra = fixed_video(RawVideoFormat::Bgra8, 64, 48, 30);
        let map = CapsConstraint::Mapping(vec![
            (CapsSet::one(nv12.clone()), CapsSet::one(rgba.clone())),
            (CapsSet::one(i420), CapsSet::one(bgra)),
        ]);
        assert_eq!(
            resolve_forward_output(&map, &nv12, None, None),
            ForwardResolve::Fixed(rgba)
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn resolve_forward_output_defers_an_unfixatable_output_that_does_not_track_the_input() {
        let input = fixed_video(RawVideoFormat::Nv12, 320, 240, 30);
        let open_rate = video(
            RawVideoFormat::Nv12,
            Dim::Fixed(640),
            Dim::Fixed(480),
            Rate::Any,
        );
        let map =
            CapsConstraint::Mapping(vec![(CapsSet::one(input.clone()), CapsSet::one(open_rate))]);
        assert_eq!(
            resolve_forward_output(&map, &input, None, None),
            ForwardResolve::Defer
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn resolve_forward_output_keeps_a_previous_shape_only_with_the_input_geometry() {
        let input = fixed_video(RawVideoFormat::I420, 320, 240, 30);
        let previous = fixed_video(RawVideoFormat::Nv12, 16, 16, 1);
        let resolve = |width: u32, height: u32| {
            let map = CapsConstraint::Mapping(vec![(
                CapsSet::one(input.clone()),
                CapsSet::from_alternatives(vec![
                    fixed_video(RawVideoFormat::I420, width, height, 30),
                    fixed_video(RawVideoFormat::Nv12, width, height, 30),
                ]),
            )]);
            resolve_forward_output(&map, &input, None, Some(&previous))
        };
        assert_eq!(
            resolve(320, 240),
            ForwardResolve::Fixed(fixed_video(RawVideoFormat::Nv12, 320, 240, 30))
        );
        assert_eq!(resolve(640, 240), ForwardResolve::Defer, "width differs");
        assert_eq!(resolve(320, 480), ForwardResolve::Defer, "height differs");
    }

    #[cfg(feature = "std")]
    #[test]
    fn resolve_forward_output_keeps_a_previous_audio_shape_only_with_the_input_layout() {
        let audio = |format: AudioFormat, channels: u8, sample_rate: u32| Caps::Audio {
            format,
            channels,
            sample_rate,
            channel_layout: crate::ChannelLayout::UNSPECIFIED,
        };
        let input = audio(AudioFormat::PcmS16Le, 2, 48_000);
        let resolve = |channels: u8, sample_rate: u32| {
            let map = CapsConstraint::Mapping(vec![(
                CapsSet::one(input.clone()),
                CapsSet::from_alternatives(vec![
                    audio(AudioFormat::PcmS16Le, channels, sample_rate),
                    audio(AudioFormat::PcmF32Le, channels, sample_rate),
                ]),
            )]);
            let previous = audio(AudioFormat::PcmF32Le, channels, 8_000);
            resolve_forward_output(&map, &input, None, Some(&previous))
        };
        assert_eq!(
            resolve(2, 48_000),
            ForwardResolve::Fixed(audio(AudioFormat::PcmF32Le, 2, 48_000))
        );
        assert_eq!(resolve(2, 44_100), ForwardResolve::Defer, "rate differs");
        assert_eq!(resolve(1, 48_000), ForwardResolve::Defer, "channels differ");
    }

    #[cfg(feature = "std")]
    #[test]
    fn resolve_forward_output_derives_a_declared_transform_from_an_unfixed_input() {
        let to_rgba = CapsConstraint::DerivedFields(CapsTransform::RawVideo {
            accept: Vec::new(),
            produce: Vec::new(),
            shapes: vec![RawVideoShape::PASSTHROUGH
                .with_format(FieldTransform::Fixed(RawVideoFormat::Rgba8))],
        });
        let no_rate = video(
            RawVideoFormat::Nv12,
            Dim::Fixed(640),
            Dim::Fixed(480),
            Rate::Any,
        );
        assert_eq!(
            resolve_forward_output(&to_rgba, &no_rate, None, None),
            ForwardResolve::Fixed(video(
                RawVideoFormat::Rgba8,
                Dim::Fixed(640),
                Dim::Fixed(480),
                Rate::Any
            ))
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn resolve_forward_output_keeps_a_retargeted_field_among_ambiguous_outputs() {
        let audio = |format: AudioFormat, channels: u8, sample_rate: u32| Caps::Audio {
            format,
            channels,
            sample_rate,
            channel_layout: crate::ChannelLayout::UNSPECIFIED,
        };
        let mono = |format| {
            AudioShape::PASSTHROUGH
                .with_format(FieldTransform::Fixed(format))
                .with_channels(FieldTransform::Fixed(1))
        };
        let to_mono = CapsConstraint::DerivedFields(CapsTransform::Audio {
            accept: Vec::new(),
            produce: Vec::new(),
            shapes: vec![mono(AudioFormat::PcmS16Le), mono(AudioFormat::PcmF32Le)],
        });
        let previous = audio(AudioFormat::PcmF32Le, 1, 48_000);
        assert_eq!(
            resolve_forward_output(
                &to_mono,
                &audio(AudioFormat::PcmS16Le, 2, 8_000),
                None,
                Some(&previous)
            ),
            ForwardResolve::Fixed(audio(AudioFormat::PcmF32Le, 1, 8_000))
        );
    }

    #[cfg(feature = "std")]
    fn chain_feasibility(constraints: Vec<NodeConstraint>) -> Vec<Option<CapsSet>> {
        let mut g: Graph<()> = Graph::new();
        let src = g.add_source(());
        let transform = g.add_transform(());
        let sink = g.add_sink(());
        g.link(src, transform).unwrap();
        g.link(transform, sink).unwrap();
        let v = g.finish().unwrap();
        let solution = solve_graph(&v, &constraints).expect("chain negotiates");
        graph_downstream_feasibility(&v, &constraints, &solution)
    }

    #[cfg(feature = "std")]
    #[test]
    fn graph_feasibility_inverts_identity_and_mapping() {
        let nv12 = fixed_video(RawVideoFormat::Nv12, 64, 48, 30);
        let i420 = fixed_video(RawVideoFormat::I420, 64, 48, 30);
        let rgba = fixed_video(RawVideoFormat::Rgba8, 64, 48, 30);
        let bgra = fixed_video(RawVideoFormat::Bgra8, 64, 48, 30);

        let feasibility = chain_feasibility(vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(i420.clone()))),
            NodeConstraint::Element(CapsConstraint::Identity(CapsSet::from_alternatives(vec![
                nv12.clone(),
                i420.clone(),
            ]))),
            NodeConstraint::Element(CapsConstraint::Accepts(CapsSet::from_alternatives(vec![
                i420.clone(),
                rgba.clone(),
            ]))),
        ]);
        assert_eq!(feasibility[0], Some(CapsSet::one(i420.clone())));

        let feasibility = chain_feasibility(vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(i420.clone()))),
            NodeConstraint::Element(CapsConstraint::Mapping(vec![
                (CapsSet::one(nv12), CapsSet::one(rgba)),
                (CapsSet::one(i420.clone()), CapsSet::one(bgra.clone())),
            ])),
            NodeConstraint::Element(CapsConstraint::Accepts(CapsSet::one(bgra))),
        ]);
        assert_eq!(feasibility[0], Some(CapsSet::one(i420)));
    }

    #[cfg(feature = "std")]
    #[test]
    fn graph_feasibility_enumerates_the_formats_a_converter_accepts() {
        let rgba = fixed_video(RawVideoFormat::Rgba8, 64, 48, 30);
        let to_rgba = CapsTransform::RawVideo {
            accept: vec![RawVideoFormat::Nv12, RawVideoFormat::I420],
            produce: Vec::new(),
            shapes: vec![RawVideoShape::PASSTHROUGH
                .with_format(FieldTransform::Fixed(RawVideoFormat::Rgba8))],
        };
        let feasibility = chain_feasibility(vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(fixed_video(
                RawVideoFormat::Nv12,
                64,
                48,
                30,
            )))),
            NodeConstraint::Element(CapsConstraint::DerivedFields(to_rgba.clone())),
            NodeConstraint::Element(CapsConstraint::Accepts(CapsSet::one(rgba.clone()))),
        ]);
        let CapsTransform::RawVideo { accept, .. } = &to_rgba else {
            unreachable!()
        };
        let expected: Vec<Caps> = accept
            .iter()
            .map(|&format| fixed_video(format, 64, 48, 30))
            .collect();
        assert_eq!(feasibility[0], Some(CapsSet::from_alternatives(expected)));

        let audio = |format| Caps::Audio {
            format,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: crate::ChannelLayout::UNSPECIFIED,
        };
        let accept = vec![AudioFormat::PcmS16Le, AudioFormat::PcmF32Le];
        let to_s16 = CapsTransform::Audio {
            accept: accept.clone(),
            produce: Vec::new(),
            shapes: vec![
                AudioShape::PASSTHROUGH.with_format(FieldTransform::Fixed(AudioFormat::PcmS16Le))
            ],
        };
        let feasibility = chain_feasibility(vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(audio(
                AudioFormat::PcmF32Le,
            )))),
            NodeConstraint::Element(CapsConstraint::DerivedFields(to_s16)),
            NodeConstraint::Element(CapsConstraint::Accepts(CapsSet::one(audio(
                AudioFormat::PcmS16Le,
            )))),
        ]);
        let expected: Vec<Caps> = accept.into_iter().map(audio).collect();
        assert_eq!(feasibility[0], Some(CapsSet::from_alternatives(expected)));
    }

    fn demux_graph() -> ValidatedGraph<()> {
        let mut g: Graph<()> = Graph::new();
        let src = g.add_source(());
        let demux = g.add_tee(2);
        let video_sink = g.add_sink(());
        let audio_sink = g.add_sink(());
        g.link(src, demux.input()).unwrap();
        g.link(demux.out(0), video_sink).unwrap();
        g.link(demux.out(1), audio_sink).unwrap();
        g.finish().unwrap()
    }

    #[test]
    fn solve_graph_demux_ports_carry_their_own_streams() {
        let container = fixed_compressed(VideoCodec::H264, 64, 48, 30);
        let video_stream = fixed_video(RawVideoFormat::Nv12, 64, 48, 30);
        let audio_stream = fixed_video(RawVideoFormat::I420, 32, 24, 30);
        let cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(container.clone()))),
            NodeConstraint::Demux {
                input: CapsConstraint::Accepts(CapsSet::one(container.clone())),
                ports: vec![
                    CapsConstraint::Produces(CapsSet::one(video_stream.clone())),
                    CapsConstraint::Produces(CapsSet::one(audio_stream.clone())),
                ],
            },
            NodeConstraint::Element(CapsConstraint::AcceptsAny),
            NodeConstraint::Element(CapsConstraint::AcceptsAny),
        ];
        let v = demux_graph();
        assert_eq!(
            solve_graph(&v, &cs),
            Ok(vec![container, video_stream, audio_stream])
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn graph_feasibility_of_a_demux_input_is_its_accept_set() {
        let container = fixed_compressed(VideoCodec::H264, 64, 48, 30);
        let accepted = CapsSet::from_alternatives(vec![
            container.clone(),
            fixed_compressed(VideoCodec::H265, 64, 48, 30),
        ]);
        let cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(container))),
            NodeConstraint::Demux {
                input: CapsConstraint::Accepts(accepted.clone()),
                ports: vec![
                    CapsConstraint::Produces(CapsSet::one(fixed_video(
                        RawVideoFormat::Nv12,
                        64,
                        48,
                        30,
                    ))),
                    CapsConstraint::Produces(CapsSet::one(fixed_video(
                        RawVideoFormat::I420,
                        32,
                        24,
                        30,
                    ))),
                ],
            },
            NodeConstraint::Element(CapsConstraint::AcceptsAny),
            NodeConstraint::Element(CapsConstraint::AcceptsAny),
        ];
        let v = demux_graph();
        let solution = solve_graph(&v, &cs).expect("demux graph negotiates");
        assert_eq!(
            graph_downstream_feasibility(&v, &cs, &solution)[0],
            Some(accepted)
        );
    }

    #[test]
    fn solve_graph_rejects_a_constraint_count_that_differs_from_the_node_count() {
        let nv12 = fixed_video(RawVideoFormat::Nv12, 64, 48, 30);
        let cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(nv12))),
            NodeConstraint::Element(CapsConstraint::AcceptsAny),
            NodeConstraint::Element(CapsConstraint::AcceptsAny),
        ];
        let mut g: Graph<()> = Graph::new();
        let src = g.add_source(());
        let sink = g.add_sink(());
        g.link(src, sink).unwrap();
        let v = g.finish().unwrap();
        assert_eq!(solve_graph(&v, &cs), Err(NegotiationFailure::Degenerate));
    }

    #[test]
    fn solve_graph_mapping_with_no_row_reaching_the_sink_fails_at_the_sink() {
        let nv12 = fixed_video(RawVideoFormat::Nv12, 64, 48, 30);
        let i420 = fixed_video(RawVideoFormat::I420, 64, 48, 30);
        let rgba = fixed_video(RawVideoFormat::Rgba8, 64, 48, 30);
        let bgra = fixed_video(RawVideoFormat::Bgra8, 64, 48, 30);
        let cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(nv12.clone()))),
            NodeConstraint::Element(CapsConstraint::Mapping(vec![
                (CapsSet::one(nv12), CapsSet::one(rgba)),
                (CapsSet::one(i420), CapsSet::one(bgra.clone())),
            ])),
            NodeConstraint::Element(CapsConstraint::Accepts(CapsSet::one(bgra))),
        ];
        let mut g: Graph<()> = Graph::new();
        let src = g.add_source(());
        let map = g.add_transform(());
        let sink = g.add_sink(());
        g.link(src, map).unwrap();
        g.link(map, sink).unwrap();
        let v = g.finish().unwrap();
        let err = solve_graph(&v, &cs).expect_err("the NV12 row maps to RGBA only");
        assert!(
            matches!(
                err,
                NegotiationFailure::EmptyLink {
                    upstream,
                    downstream,
                    ..
                } if (upstream, downstream) == (map.0 as usize, sink.0 as usize)
            ),
            "{err:?}"
        );
    }

    #[test]
    fn solve_graph_muxer_pad_narrows_its_source() {
        let nv12 = fixed_video(RawVideoFormat::Nv12, 64, 48, 30);
        let i420 = fixed_video(RawVideoFormat::I420, 64, 48, 30);
        let muxed = fixed_compressed(VideoCodec::H264, 64, 48, 30);
        let cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::from_alternatives(vec![
                nv12.clone(),
                i420.clone(),
            ]))),
            NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(nv12.clone()))),
            NodeConstraint::Muxer {
                inputs: vec![
                    CapsConstraint::Accepts(CapsSet::one(i420.clone())),
                    CapsConstraint::AcceptsAny,
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
        assert_eq!(solve_graph(&v, &cs), Ok(vec![i420, nv12, muxed]));
    }

    #[test]
    fn solve_graph_batch_output_matches_the_streams_it_was_fixated_with() {
        let nv12 = fixed_video(RawVideoFormat::Nv12, 64, 48, 30);
        let i420 = fixed_video(RawVideoFormat::I420, 64, 48, 30);
        let both = CapsSet::from_alternatives(vec![nv12.clone(), i420.clone()]);
        let batch = |first: &Caps, second: &Caps| Caps::Batch {
            streams: vec![first.clone(), second.clone()],
        };
        let mixed = CapsSet::from_alternatives(vec![batch(&nv12, &i420), batch(&i420, &nv12)]);
        let cs: Vec<NodeConstraint> = vec![
            NodeConstraint::Element(CapsConstraint::Produces(both.clone())),
            NodeConstraint::Element(CapsConstraint::Produces(both)),
            NodeConstraint::BatchMuxer {
                inputs: vec![CapsConstraint::AcceptsAny, CapsConstraint::AcceptsAny],
            },
            NodeConstraint::Element(CapsConstraint::Accepts(mixed)),
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
        let sol = solve_graph(&v, &cs).expect("a mixed batch exists");
        assert_eq!(sol[2], batch(&sol[0], &sol[1]));
    }

    #[test]
    fn solve_graph_batch_demux_takes_exactly_the_combination_cap() {
        let max = crate::caps::MAX_BATCH_ALTERNATIVES;
        let side = max.isqrt() as u32;
        let batches: Vec<Caps> = (1..=side + 1)
            .flat_map(|first| {
                (1..=side + 1).map(move |second| Caps::Batch {
                    streams: vec![
                        fixed_video(RawVideoFormat::Nv12, 16 * first, 16, 30),
                        fixed_video(RawVideoFormat::I420, 16 * second, 16, 30),
                    ],
                })
            })
            .collect();
        let solve = |count: usize| {
            let cs: Vec<NodeConstraint> = vec![
                NodeConstraint::Element(CapsConstraint::Produces(CapsSet::from_alternatives(
                    batches[..count].to_vec(),
                ))),
                NodeConstraint::BatchDemux {
                    input: CapsConstraint::AcceptsAny,
                },
                NodeConstraint::Element(CapsConstraint::AcceptsAny),
                NodeConstraint::Element(CapsConstraint::AcceptsAny),
            ];
            solve_graph(&demux_graph(), &cs)
        };
        let at_cap = solve(max).expect("the cap itself is allowed");
        assert_eq!(at_cap[0], batches[0]);
        assert_eq!(
            solve(max + 1),
            Err(NegotiationFailure::TooManyBatchAlternatives { index: 1 })
        );
    }

    #[test]
    fn a_range_fixates_toward_the_source_first_alternative() {
        const SOURCE_WIDTH: u32 = 320;
        const FILTER_MIN: u32 = 100;
        const FILTER_MAX: u32 = 200;
        const PADDING: u32 = 16;
        let with_width = |width: Dim| {
            video(
                RawVideoFormat::Rgba8,
                width,
                Dim::Fixed(240),
                Rate::Fixed(30 << 16),
            )
        };
        let source_range = with_width(Dim::Range {
            min: 1,
            max: u32::MAX - PADDING,
        });
        let constraints = |source: Vec<Caps>| {
            vec![
                CapsConstraint::Produces(CapsSet::from_alternatives(source)),
                CapsConstraint::Identity(CapsSet::one(with_width(Dim::Range {
                    min: FILTER_MIN,
                    max: FILTER_MAX,
                }))),
                CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| {
                    let Some((width, _, _)) = input.dims() else {
                        return CapsSet::from_alternatives(Vec::new());
                    };
                    let padded = match width {
                        Dim::Fixed(w) => Dim::Fixed(w + PADDING),
                        Dim::Range { min, max } => Dim::Range {
                            min: min + PADDING,
                            max: max + PADDING,
                        },
                        Dim::Any => Dim::Any,
                    };
                    CapsSet::one(with_width(padded))
                })),
                CapsConstraint::AcceptsAny,
            ]
        };
        let mut g: Graph<()> = Graph::new();
        let nodes = [
            g.add_source(()),
            g.add_transform(()),
            g.add_transform(()),
            g.add_sink(()),
        ];
        for pair in nodes.windows(2) {
            g.link(pair[0], pair[1]).unwrap();
        }
        let v = g.finish().unwrap();
        let solve_both = |source: Vec<Caps>| {
            let linear = constraints(source.clone());
            let linear = solve_linear(&linear.iter().collect::<Vec<_>>()).expect("chain solves");
            let graph: Vec<NodeConstraint> = constraints(source)
                .into_iter()
                .map(NodeConstraint::Element)
                .collect();
            assert_eq!(solve_graph(&v, &graph), Ok(linear.clone()));
            linear
        };
        let expected = |width: u32| {
            vec![
                with_width(Dim::Fixed(width)),
                with_width(Dim::Fixed(width)),
                with_width(Dim::Fixed(width + PADDING)),
            ]
        };

        let preferred = solve_both(vec![
            with_width(Dim::Fixed(SOURCE_WIDTH)),
            source_range.clone(),
        ]);
        assert_eq!(
            preferred,
            expected(SOURCE_WIDTH.clamp(FILTER_MIN, FILTER_MAX))
        );
        let unpreferred = solve_both(vec![source_range]);
        assert_eq!(
            unpreferred,
            expected(FILTER_MIN),
            "no fixed first alternative keeps the minimum"
        );
    }
}
