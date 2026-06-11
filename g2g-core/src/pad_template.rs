//! Pad templates: declarative, pre-instantiation metadata describing the
//! caps an element *type* can accept or produce.
//!
//! `caps_constraint_as_*` are runtime methods on a *constructed* element,
//! so they can reflect instance state (a sink already narrowed to one
//! format, a decoder's derived output). A pad template is the *static*
//! superset of what the element type can ever do, and it is queryable
//! without constructing the element. This is the analog of GStreamer's
//! static pad templates from `gst_element_factory_get_static_pad_templates`.
//!
//! Two uses:
//! - **Introspection.** A tool lists an element type's pads and the caps
//!   each supports without building a graph.
//! - **Pre-instantiation solver queries.** [`pad_link`] / [`types_can_link`]
//!   run the same [`solve_linear`] used at negotiation against two types'
//!   templates, answering "can A's output feed B's input?" before either
//!   element exists.

use alloc::vec::Vec;

use crate::caps::{Caps, CapsSet};
use crate::format_element::CapsConstraint;
use crate::runtime::solver::{solve_linear, NegotiationFailure};

/// Which side of an element a pad sits on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadDirection {
    /// Input pad: the element consumes caps here.
    Sink,
    /// Output pad: the element produces caps here.
    Source,
}

/// The static caps capability of a pad.
#[derive(Debug, Clone, PartialEq)]
pub enum PadCaps {
    /// A concrete, ordered set of supported caps (highest preference first).
    Fixed(CapsSet),
    /// Wildcard: a sink pad that accepts any caps (e.g. `FakeSink`). On a
    /// source pad this is degenerate, a producer must name concrete caps,
    /// and is treated as the empty producible set.
    Any,
}

/// Static, pre-instantiation declaration of one pad's caps capability. The
/// runtime `caps_constraint_as_*` of a constructed instance is always a
/// subset of its pad template (it may narrow to instance configuration).
#[derive(Debug, Clone, PartialEq)]
pub struct PadTemplate {
    pub direction: PadDirection,
    pub caps: PadCaps,
}

impl PadTemplate {
    /// A sink pad accepting the given concrete caps set.
    pub fn sink(caps: CapsSet) -> Self {
        Self { direction: PadDirection::Sink, caps: PadCaps::Fixed(caps) }
    }

    /// A source pad producing the given concrete caps set.
    pub fn source(caps: CapsSet) -> Self {
        Self { direction: PadDirection::Source, caps: PadCaps::Fixed(caps) }
    }

    /// A sink pad accepting any caps (wildcard).
    pub fn sink_any() -> Self {
        Self { direction: PadDirection::Sink, caps: PadCaps::Any }
    }

    /// The solver constraint this pad contributes at its end of a link: a
    /// source pad `Produces`, a sink pad `Accepts` (or `AcceptsAny`). The
    /// result owns its caps, so it borrows nothing from `self`.
    pub fn as_constraint(&self) -> CapsConstraint<'static> {
        match (self.direction, &self.caps) {
            (PadDirection::Source, PadCaps::Fixed(s)) => CapsConstraint::Produces(s.clone()),
            (PadDirection::Sink, PadCaps::Fixed(s)) => CapsConstraint::Accepts(s.clone()),
            (PadDirection::Sink, PadCaps::Any) => CapsConstraint::AcceptsAny,
            (PadDirection::Source, PadCaps::Any) => {
                CapsConstraint::Produces(CapsSet::from_alternatives(Vec::new()))
            }
        }
    }
}

/// Element types that publish static pad templates. The query is an
/// associated function (no `&self`), so a tool inspects a *type* without
/// constructing it: `<FakeSink as PadTemplates>::pad_templates()`.
pub trait PadTemplates {
    /// Every pad this element type exposes, in declaration order.
    fn pad_templates() -> Vec<PadTemplate>;

    /// The first pad template in the given direction, if any.
    fn pad_template(direction: PadDirection) -> Option<PadTemplate> {
        Self::pad_templates().into_iter().find(|t| t.direction == direction)
    }
}

/// Pre-instantiation solver query: the caps an element whose source pad is
/// `producer` would fixate to feeding an element whose sink pad is
/// `consumer`, without constructing either. Returns the fixated caps, or a
/// structured [`NegotiationFailure`]:
/// - `EmptyLink` — the pads' caps don't intersect (genuinely incompatible).
/// - `Unfixable` — the shapes *are* compatible, but a field is still open
///   on both sides (common for static templates that leave geometry or
///   framerate `Any`); a concrete value is chosen at instance time. Use
///   [`types_can_link`] when "are they compatible?" is the question.
/// - `EndpointShapeMismatch` — the directions were swapped.
pub fn pad_link(
    producer: &PadTemplate,
    consumer: &PadTemplate,
) -> Result<Caps, NegotiationFailure> {
    let producer_c = producer.as_constraint();
    let consumer_c = consumer.as_constraint();
    solve_linear(&[&producer_c, &consumer_c])?
        .into_iter()
        .last()
        .ok_or(NegotiationFailure::Degenerate)
}

/// Convenience tooling query: can a `A`-typed element's source pad feed a
/// `B`-typed element's sink pad? `false` if either lacks the needed pad or
/// the caps don't intersect. An `Unfixable` result counts as compatible:
/// static templates routinely leave geometry / framerate open, and that
/// only resolves at instance time, not at type-compatibility time. Use
/// [`pad_link`] when you need the fixated caps or the precise failure.
pub fn types_can_link<A, B>() -> bool
where
    A: PadTemplates,
    B: PadTemplates,
{
    match (A::pad_template(PadDirection::Source), B::pad_template(PadDirection::Sink)) {
        (Some(producer), Some(consumer)) => matches!(
            pad_link(&producer, &consumer),
            Ok(_) | Err(NegotiationFailure::Unfixable { .. })
        ),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::{Dim, Rate, VideoCodec, RawVideoFormat};

    /// Fully concrete caps (a real producer names every field).
    fn fixed(format: RawVideoFormat, w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(30 << 16),
        }
    }

    /// A format at any geometry / framerate (a broad static template).
    fn any_geom(format: RawVideoFormat) -> Caps {
        Caps::RawVideo { format, width: Dim::Any, height: Dim::Any, framerate: Rate::Any }
    }

    /// Produces RGBA at any geometry (static superset, like `VideoTestSrc`).
    struct RgbaSource;
    impl PadTemplates for RgbaSource {
        fn pad_templates() -> Vec<PadTemplate> {
            alloc::vec![PadTemplate::source(CapsSet::one(any_geom(RawVideoFormat::Rgba8)))]
        }
    }

    /// Accepts only NV12.
    struct Nv12Sink;
    impl PadTemplates for Nv12Sink {
        fn pad_templates() -> Vec<PadTemplate> {
            alloc::vec![PadTemplate::sink(CapsSet::one(any_geom(RawVideoFormat::Nv12)))]
        }
    }

    /// Accepts anything.
    struct AnySink;
    impl PadTemplates for AnySink {
        fn pad_templates() -> Vec<PadTemplate> {
            alloc::vec![PadTemplate::sink_any()]
        }
    }

    #[test]
    fn compatible_pads_link_to_fixated_caps() {
        // A concrete producer feeding a broad sink fixates to the producer's caps.
        let producer = PadTemplate::source(CapsSet::one(fixed(RawVideoFormat::Rgba8, 1280, 720)));
        let consumer = PadTemplate::sink(CapsSet::one(any_geom(RawVideoFormat::Rgba8)));
        let caps = pad_link(&producer, &consumer).expect("pads overlap");
        assert_eq!(caps, fixed(RawVideoFormat::Rgba8, 1280, 720));
    }

    #[test]
    fn disjoint_pads_report_empty_link() {
        let producer = PadTemplate::source(CapsSet::one(any_geom(RawVideoFormat::Rgba8)));
        let consumer = PadTemplate::sink(CapsSet::one(any_geom(RawVideoFormat::Nv12)));
        assert!(
            matches!(pad_link(&producer, &consumer), Err(NegotiationFailure::EmptyLink { .. })),
            "RGBA producer cannot feed an NV12-only sink"
        );
    }

    #[test]
    fn wildcard_sink_accepts_any_producer() {
        let producer = PadTemplate::source(CapsSet::one(fixed(RawVideoFormat::Rgba8, 640, 480)));
        let caps = pad_link(&producer, &PadTemplate::sink_any()).expect("any sink accepts");
        assert_eq!(caps, fixed(RawVideoFormat::Rgba8, 640, 480));
    }

    #[test]
    fn types_can_link_uses_each_type_template() {
        assert!(types_can_link::<RgbaSource, AnySink>(), "RGBA source -> any sink");
        assert!(!types_can_link::<RgbaSource, Nv12Sink>(), "RGBA source -/-> NV12 sink");
        // A source has no sink pad, so nothing can feed into it.
        assert!(!types_can_link::<RgbaSource, RgbaSource>(), "source has no sink pad");
    }
}
