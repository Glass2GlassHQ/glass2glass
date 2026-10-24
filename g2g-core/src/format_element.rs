//! Negotiation-time element surface (design/caps.md).
//!
//! `FormatElement` is the trait the solver consumes; `CapsConstraint` is
//! the per-element data it walks.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::caps::{Caps, CapsSet};
use crate::caps_transform::CapsTransform;
use crate::element::ElementBound;
use crate::error::G2gError;

/// Per-element constraint surface read by the solver. Element categories
/// map cleanly:
///
/// | Element                                              | Constraint                        |
/// | ---                                                  | ---                               |
/// | Source producing its caps                            | `Produces(set)`                   |
/// | Sink narrowing what it takes from upstream           | `Accepts(set)`                    |
/// | Identity transform                                   | `Identity(set)`                   |
/// | Decoder reading dims from input SPS                  | `DerivedOutput(\|in\| out_set)`   |
/// | Caps-driven transform (scale / convert / rate)       | `DerivedFields(transform)`        |
/// | Pre-enumerated input/output pairs (codecs, scalers)  | `Mapping(pairs)`                  |
pub enum CapsConstraint<'a> {
    /// Sink shape: only the input side is constrained. Output is
    /// unused (sink has no downstream).
    Accepts(CapsSet),

    /// Source shape: only the output side is constrained. Input is
    /// unused.
    Produces(CapsSet),

    /// Pass-through transform: input == output, both drawn from this
    /// set. Identity / probe / metering elements land here, as do
    /// format converters that accept only one format.
    Identity(CapsSet),

    /// Format-changing transform with an explicit (input, output)
    /// relation. The vector enumerates all legal pairs; the solver
    /// picks one. Most decoders and encoders use this when their
    /// output is determined by configuration rather than the input
    /// data itself.
    Mapping(Vec<(CapsSet, CapsSet)>),

    /// Programmatic mapping: the output set is a function of the
    /// already-narrowed input caps. Used when output depends on the
    /// input in a way that can't be precomputed, e.g. a decoder
    /// reading SPS to fix output dims. The solver calls this during
    /// forward propagation, after the input link has been narrowed
    /// but before the output link is solved.
    DerivedOutput(Box<dyn Fn(&Caps) -> CapsSet + Send + Sync + 'a>),

    /// Declarative form of [`DerivedOutput`](Self::DerivedOutput): the
    /// forward derivation stated field by field as data the solver can inspect.
    /// Because it knows which fields are passed through unchanged, it can
    /// couple those *bidirectionally and per field*, so a downstream pin on a
    /// passthrough field narrows the corresponding input field (`Range ∩ Fixed
    /// = Fixed`) rather than only dropping whole input alternatives. That is
    /// what lets a geometry pin flow back through a geometry-passthrough
    /// transform (`videoscale ! videoconvert ! caps`). Used by the caps-driven
    /// transforms (videoscale / videoconvert / videorate / audioconvert /
    /// audioresample); decoders whose output isn't a per-field function of
    /// their input stay on `DerivedOutput`.
    DerivedFields(CapsTransform),

    /// Sink-shape wildcard: accepts whatever upstream produces, of any
    /// media type, format, dims, or rate. Models debug / probe /
    /// passthrough sinks (`FakeSink`, `syncsink`, `identity`-as-sink)
    /// whose `intercept_caps` is `Ok(upstream.clone())`. The solver
    /// treats this as a no-op narrowing on the link: the upstream's
    /// produced caps flow through unchanged.
    AcceptsAny,

    /// Transform-shape wildcard: forwards whatever upstream produces.
    /// Input == output, both unconstrained. Models pass-through
    /// transforms like `IdentityTransform` (probe / tee / metering
    /// without format constraints). The solver couples the input and
    /// output links to be equal but doesn't narrow either by a set.
    IdentityAny,
}

impl core::fmt::Debug for CapsConstraint<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Accepts(s) => f.debug_tuple("Accepts").field(s).finish(),
            Self::Produces(s) => f.debug_tuple("Produces").field(s).finish(),
            Self::Identity(s) => f.debug_tuple("Identity").field(s).finish(),
            Self::Mapping(v) => f.debug_tuple("Mapping").field(v).finish(),
            Self::DerivedOutput(_) => f.debug_tuple("DerivedOutput").field(&"<fn>").finish(),
            Self::DerivedFields(t) => f.debug_tuple("DerivedFields").field(t).finish(),
            Self::AcceptsAny => f.write_str("AcceptsAny"),
            Self::IdentityAny => f.write_str("IdentityAny"),
        }
    }
}

impl CapsConstraint<'_> {
    /// ACCEPT_CAPS query (design/caps.md): would this element accept a
    /// link carrying `caps`? A pure check against the declared
    /// constraint, with no runtime negotiation or back-and-forth.
    ///
    /// For sink / transform shapes this tests the *input* side; for the
    /// source shape it tests the *produced* side. Wildcards accept
    /// anything.
    pub fn accepts(&self, caps: &Caps) -> bool {
        match self {
            Self::Accepts(set) | Self::Produces(set) | Self::Identity(set) => set.accepts(caps),
            Self::Mapping(pairs) => pairs.iter().any(|(input, _)| input.accepts(caps)),
            Self::DerivedOutput(f) => !f(caps).is_empty(),
            Self::DerivedFields(t) => !t.derive(caps).is_empty(),
            Self::AcceptsAny | Self::IdentityAny => true,
        }
    }
}

/// What one element is willing to pay for each of its advertised
/// alternatives: a cost per alternative, aligned index-for-index with the
/// [`CapsSet`] its [`CapsConstraint`] advertises (with the pair list for
/// [`Mapping`](CapsConstraint::Mapping)). Lower is more preferred. The solver
/// picks, among the consistent assignments of a linear chain, the one whose
/// per-element costs sum to the least.
///
/// An element that declares nothing keeps the implicit rule "cost =
/// alternative index", which is the set's own order. Costs buy the two things
/// that order cannot say: *equal* cost between alternatives, meaning this
/// element does not care and a neighbour's preference should decide, and a
/// large gap, meaning a fallback is much worse than the first choice so a
/// neighbour's mild preference must not pull the chain onto it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CapsPreferences {
    costs: Vec<u32>,
}

impl CapsPreferences {
    /// Declare one cost per advertised alternative, in the same order.
    pub fn new(costs: Vec<u32>) -> Self {
        Self { costs }
    }

    /// The implicit default stated explicitly: alternative `i` costs `i`.
    pub fn by_order(count: usize) -> Self {
        Self {
            costs: (0..count as u32).collect(),
        }
    }

    /// Every alternative equally acceptable, so neighbours decide.
    pub fn indifferent(count: usize) -> Self {
        Self {
            costs: alloc::vec![0; count],
        }
    }

    /// The declared costs, in alternative order.
    pub fn costs(&self) -> &[u32] {
        &self.costs
    }

    /// True when nothing is declared, which the solver reads the same as
    /// no preferences at all.
    pub fn is_empty(&self) -> bool {
        self.costs.is_empty()
    }

    /// Cost of alternative `index`. An index past the declared costs falls
    /// back to the implicit "cost = index", so a short list still describes
    /// the alternatives it covers.
    pub fn cost(&self, index: usize) -> u32 {
        self.costs.get(index).copied().unwrap_or(index as u32)
    }
}

/// Negotiation-time view of an element. The runtime side
/// (`AsyncElement::process`) is unchanged; this trait carries the
/// information the solver needs to assign caps to every link before
/// `process` runs.
pub trait FormatElement: ElementBound {
    /// Declare the constraint this element imposes on its surrounding
    /// links. Read by the solver during negotiation.
    fn caps_constraint(&self) -> CapsConstraint<'_>;

    /// Optional per-alternative costs for this element's advertised
    /// constraint. Defaults to `None`, meaning the solver uses the
    /// constraint's own preference order (cost = alternative index).
    fn caps_preferences(&self) -> Option<CapsPreferences> {
        None
    }

    /// Called by the runner once the solver has assigned caps to every
    /// link. Boundary elements (decoders, encoders) receive distinct
    /// input / output values; non-boundary elements receive equal
    /// ones. Sources see `input = None`; sinks see `output = None`.
    fn configure_link(
        &mut self,
        input: Option<&Caps>,
        output: Option<&Caps>,
    ) -> Result<(), G2gError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::{Dim, Rate, RawVideoFormat, VideoCodec};
    use alloc::vec;

    fn video(format: RawVideoFormat, w: Dim, h: Dim, r: Rate) -> Caps {
        Caps::RawVideo {
            format,
            width: w,
            height: h,
            framerate: r,
            interlace: crate::Interlace::Any,
            colorimetry: crate::Colorimetry::UNKNOWN,
        }
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

    struct FakeSource;
    impl FormatElement for FakeSource {
        fn caps_constraint(&self) -> CapsConstraint<'_> {
            CapsConstraint::Produces(CapsSet::one(compressed(
                VideoCodec::H264,
                Dim::Fixed(1920),
                Dim::Fixed(1080),
                Rate::Fixed(30 << 16),
            )))
        }
        fn configure_link(
            &mut self,
            input: Option<&Caps>,
            output: Option<&Caps>,
        ) -> Result<(), G2gError> {
            assert!(input.is_none() && output.is_some());
            Ok(())
        }
    }

    struct FakeSink;
    impl FormatElement for FakeSink {
        fn caps_constraint(&self) -> CapsConstraint<'_> {
            CapsConstraint::Accepts(CapsSet::one(video(
                RawVideoFormat::Nv12,
                Dim::Any,
                Dim::Any,
                Rate::Any,
            )))
        }
        fn configure_link(
            &mut self,
            input: Option<&Caps>,
            output: Option<&Caps>,
        ) -> Result<(), G2gError> {
            assert!(input.is_some() && output.is_none());
            Ok(())
        }
    }

    struct FakeDecoder;
    impl FormatElement for FakeDecoder {
        fn caps_constraint(&self) -> CapsConstraint<'_> {
            // Output dims are read from the input caps; framerate is preserved.
            CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
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
            }))
        }
        fn configure_link(
            &mut self,
            _input: Option<&Caps>,
            _output: Option<&Caps>,
        ) -> Result<(), G2gError> {
            Ok(())
        }
    }

    #[test]
    fn source_constraint_is_produces() {
        let s = FakeSource;
        let c = s.caps_constraint();
        match c {
            CapsConstraint::Produces(set) => assert_eq!(set.alternatives().len(), 1),
            _ => panic!("expected Produces"),
        }
    }

    #[test]
    fn sink_constraint_is_accepts() {
        let s = FakeSink;
        let c = s.caps_constraint();
        match c {
            CapsConstraint::Accepts(set) => assert!(!set.is_empty()),
            _ => panic!("expected Accepts"),
        }
    }

    #[test]
    fn derived_output_is_function_of_input() {
        let d = FakeDecoder;
        let input = compressed(
            VideoCodec::H264,
            Dim::Fixed(1280),
            Dim::Fixed(720),
            Rate::Fixed(30 << 16),
        );
        let c = d.caps_constraint();
        match c {
            CapsConstraint::DerivedOutput(f) => {
                let out = f(&input);
                assert_eq!(
                    out.alternatives(),
                    &[video(
                        RawVideoFormat::Nv12,
                        Dim::Fixed(1280),
                        Dim::Fixed(720),
                        Rate::Fixed(30 << 16)
                    )]
                );
            }
            _ => panic!("expected DerivedOutput"),
        }
    }

    #[test]
    fn mapping_constraint_holds_pairs() {
        struct FakeScaler;
        impl FormatElement for FakeScaler {
            fn caps_constraint(&self) -> CapsConstraint<'_> {
                let in_a = CapsSet::one(video(
                    RawVideoFormat::Nv12,
                    Dim::Fixed(1920),
                    Dim::Fixed(1080),
                    Rate::Any,
                ));
                let out_a = CapsSet::one(video(
                    RawVideoFormat::Nv12,
                    Dim::Fixed(1280),
                    Dim::Fixed(720),
                    Rate::Any,
                ));
                let in_b = CapsSet::one(video(
                    RawVideoFormat::I420,
                    Dim::Fixed(1920),
                    Dim::Fixed(1080),
                    Rate::Any,
                ));
                let out_b = CapsSet::one(video(
                    RawVideoFormat::I420,
                    Dim::Fixed(1280),
                    Dim::Fixed(720),
                    Rate::Any,
                ));
                CapsConstraint::Mapping(vec![(in_a, out_a), (in_b, out_b)])
            }
            fn configure_link(
                &mut self,
                _input: Option<&Caps>,
                _output: Option<&Caps>,
            ) -> Result<(), G2gError> {
                Ok(())
            }
        }
        match FakeScaler.caps_constraint() {
            CapsConstraint::Mapping(pairs) => assert_eq!(pairs.len(), 2),
            other => panic!("expected Mapping, got {other:?}"),
        }
    }

    #[test]
    fn default_preferences_is_none() {
        assert!(FakeSource.caps_preferences().is_none());
        assert!(FakeSink.caps_preferences().is_none());
    }

    #[test]
    fn accept_caps_query_checks_constraint_set() {
        // ACCEPT_CAPS (design/README.md §7): pure check against the declared set.
        let nv12_720 = video(
            RawVideoFormat::Nv12,
            Dim::Fixed(1280),
            Dim::Fixed(720),
            Rate::Any,
        );
        let h264_720 = compressed(
            VideoCodec::H264,
            Dim::Fixed(1280),
            Dim::Fixed(720),
            Rate::Any,
        );

        // Accepts(NV12/any) takes NV12, rejects H.264.
        let sink = FakeSink.caps_constraint();
        assert!(sink.accepts(&nv12_720));
        assert!(!sink.accepts(&h264_720));

        // Identity(NV12@1280x720) takes the exact caps, rejects a mismatch.
        let id = CapsConstraint::Identity(CapsSet::one(nv12_720.clone()));
        assert!(id.accepts(&nv12_720));
        assert!(!id.accepts(&video(
            RawVideoFormat::Nv12,
            Dim::Fixed(1920),
            Dim::Fixed(1080),
            Rate::Any,
        )));

        // DerivedOutput keys on input validity: H.264 in, not NV12.
        let dec = FakeDecoder.caps_constraint();
        assert!(dec.accepts(&h264_720));

        // Wildcards take anything.
        assert!(CapsConstraint::AcceptsAny.accepts(&h264_720));
        assert!(CapsConstraint::IdentityAny.accepts(&nv12_720));
    }

    #[test]
    fn configure_link_signature_works_for_all_shapes() {
        let cap = video(
            RawVideoFormat::Nv12,
            Dim::Fixed(640),
            Dim::Fixed(480),
            Rate::Fixed(30 << 16),
        );
        FakeSource.configure_link(None, Some(&cap)).unwrap();
        FakeSink.configure_link(Some(&cap), None).unwrap();
        FakeDecoder.configure_link(Some(&cap), Some(&cap)).unwrap();
    }
}
