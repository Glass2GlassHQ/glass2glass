//! M16 step 3 (DESIGN-M16-caps-nego.md §5): linear-pipeline caps solver.
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

use alloc::vec::Vec;

use crate::caps::{Caps, CapsSet};
use crate::format_element::CapsConstraint;

/// Per-link assignment produced by the solver: one fixated `Caps` per
/// link between adjacent elements. For an `N`-element pipeline this is
/// length `N - 1`. The runner calls `configure_link` on element `i`
/// with `input = links[i - 1]` and `output = links[i]` (sources receive
/// `None` on input; sinks receive `None` on output).
pub type LinkSolution = Vec<Caps>;

/// Structured solver failure (DESIGN-M16 §5).
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
                return Err(NegotiationFailure::EmptyLink { upstream: i - 1, downstream: i + 1 });
            }
            Ok(out)
        }
        CapsConstraint::DerivedOutput(f) => {
            let fixed = upstream.fixate().ok_or(NegotiationFailure::Unfixable {
                upstream: i - 1,
                downstream: i,
            })?;
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
            // Only fire once the input link has fixated to a single
            // concrete Caps. Until then the derived output is unknown
            // and we leave the output link to other constraints.
            if let Some(input_set) = &links[ii] {
                if let Some(fixed_input) = input_set.fixate() {
                    if input_set.alternatives().len() == 1
                        && input_set.alternatives()[0] == fixed_input
                    {
                        let derived = f(&fixed_input);
                        if derived.is_empty() {
                            return Err(NegotiationFailure::EmptyLink {
                                upstream: i,
                                downstream: i + 1,
                            });
                        }
                        narrow(links, oi, &derived, i, i + 1)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::{Dim, Rate, VideoCodec, RawVideoFormat};
    use alloc::boxed::Box;
    use alloc::vec;

    fn video(fmt: RawVideoFormat, w: Dim, h: Dim, r: Rate) -> Caps {
        Caps::RawVideo { format: fmt, width: w, height: h, framerate: r }
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
}
