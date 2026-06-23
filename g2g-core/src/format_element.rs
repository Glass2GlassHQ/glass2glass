//! M16 step 2 (DESIGN.md §4.13.1): negotiation-time element
//! surface.
//!
//! `FormatElement` is the trait the future solver (M16 step 3) consumes;
//! `CapsConstraint` is the per-element data it walks. Both coexist with
//! the runtime `AsyncElement` surface during migration. The
//! `AsyncElement` → `FormatElement` adapter (the "legacy" path the
//! design plan calls for) lands together with the solver in step 3,
//! because its exact shape is dictated by what the solver consumes.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::caps::{Caps, CapsSet, PassthroughFields};
use crate::element::{AsyncElement, ElementBound};
use crate::error::G2gError;

/// Boxed `intercept_caps`-style callback (input narrowing) for the legacy
/// migration bridge. Deleted with the bridge.
type InterceptFn<'a> = Box<dyn Fn(&Caps) -> Result<Caps, G2gError> + 'a>;
/// Boxed `propose_output_caps`-style callback (forward derivation) for the
/// legacy migration bridge. Deleted with the bridge.
type ProposeFn<'a> = Box<dyn Fn(&Caps) -> Caps + 'a>;

/// Per-element constraint surface read by the solver.
///
/// Replaces today's `intercept_caps` (the input-narrowing half) and the
/// forward-half `propose_output_caps` hook with a single declarative
/// shape. Element categories map cleanly:
///
/// | Today                                                | Tomorrow                          |
/// | ---                                                  | ---                               |
/// | Source `intercept_caps` returns its produced caps    | `Produces(set)`                   |
/// | Sink `intercept_caps` narrows upstream               | `Accepts(set)`                    |
/// | Identity transform                                   | `Identity(set)`                   |
/// | Decoder reading dims from input SPS                  | `DerivedOutput(\|in\| out_set)`   |
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

    /// Like [`DerivedOutput`](Self::DerivedOutput), but additionally declares
    /// which caps fields the transform passes through unchanged (output field
    /// == input field). The `derive` closure stays the source of truth for
    /// forward derivation (the retargeted fields); the `passthrough` mask lets
    /// the solver couple the passthrough fields *bidirectionally and per field*
    /// so a downstream pin on a passthrough field narrows the corresponding
    /// input field (`Range ∩ Fixed = Fixed`), not just drops whole input
    /// alternatives. This is what lets a geometry pin flow back through a
    /// geometry-passthrough transform (`videoscale ! videoconvert ! caps`).
    /// Used by the caps-driven transforms (videoscale / videoconvert /
    /// audioresample); decoders that genuinely can't invert stay on
    /// `DerivedOutput`.
    DerivedCoupled {
        derive: Box<dyn Fn(&Caps) -> CapsSet + Send + Sync + 'a>,
        passthrough: PassthroughFields,
    },

    /// **Migration bridge.** Source whose
    /// [`SourceLoop::intercept_caps`](crate::runtime::SourceLoop::intercept_caps)
    /// has been evaluated to a single concrete `Caps`. Used by the
    /// runner to wrap legacy elements until step 5 migrates them to
    /// `Produces`. Deleted at the end of the migration.
    LegacySource(Caps),

    /// **Migration bridge.** Transform with the today's
    /// `intercept_caps(upstream) -> Caps` + `propose_output_caps(input)
    /// -> Caps` callbacks. Non-boundary transforms set
    /// `propose_output` to clone the input. Deleted at the end of the
    /// migration. The boxes are not `Send + Sync` because the solver
    /// consumes them synchronously on the runner thread.
    LegacyTransform {
        intercept: InterceptFn<'a>,
        propose_output: ProposeFn<'a>,
    },

    /// **Migration bridge.** Sink with today's
    /// `intercept_caps(upstream) -> Caps` callback. Deleted at the end
    /// of the migration.
    LegacySink(InterceptFn<'a>),

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
            Self::DerivedCoupled { passthrough, .. } => f
                .debug_struct("DerivedCoupled")
                .field("derive", &"<fn>")
                .field("passthrough", passthrough)
                .finish(),
            Self::LegacySource(c) => f.debug_tuple("LegacySource").field(c).finish(),
            Self::LegacyTransform { .. } => {
                f.debug_struct("LegacyTransform").field("intercept", &"<fn>").finish_non_exhaustive()
            }
            Self::LegacySink(_) => f.debug_tuple("LegacySink").field(&"<fn>").finish(),
            Self::AcceptsAny => f.write_str("AcceptsAny"),
            Self::IdentityAny => f.write_str("IdentityAny"),
        }
    }
}

impl CapsConstraint<'_> {
    /// ACCEPT_CAPS query (DESIGN.md §4.13.1): would this element accept a
    /// link carrying `caps`? A pure check against the declared
    /// constraint, with no runtime negotiation or back-and-forth.
    ///
    /// For sink / transform shapes this tests the *input* side; for the
    /// source shape it tests the *produced* side. Wildcards accept
    /// anything; the legacy bridges defer to their wrapped callbacks.
    pub fn accepts(&self, caps: &Caps) -> bool {
        match self {
            Self::Accepts(set) | Self::Produces(set) | Self::Identity(set) => set.accepts(caps),
            Self::Mapping(pairs) => pairs.iter().any(|(input, _)| input.accepts(caps)),
            Self::DerivedOutput(f) => !f(caps).is_empty(),
            Self::DerivedCoupled { derive, .. } => !derive(caps).is_empty(),
            Self::AcceptsAny | Self::IdentityAny => true,
            Self::LegacySource(produced) => produced.intersect(caps).is_ok(),
            Self::LegacyTransform { intercept, .. } => intercept(caps).is_ok(),
            Self::LegacySink(intercept) => intercept(caps).is_ok(),
        }
    }
}

/// Optional preference data for tie-breaking. Reserved for the solver's
/// scoring pass; the concrete algebra is unspecified (DESIGN.md §4.13.2).
/// The empty value is meaningful: "use the constraint's own preference
/// order."
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CapsPreferences;

/// Negotiation-time view of an element. The runtime side
/// (`AsyncElement::process`) is unchanged; this trait carries the
/// information the solver needs to assign caps to every link before
/// `process` runs.
pub trait FormatElement: ElementBound {
    /// Declare the constraint this element imposes on its surrounding
    /// links. Read by the solver during negotiation.
    fn caps_constraint(&self) -> CapsConstraint<'_>;

    /// Optional preferences for tie-breaking. Defaults to `None`,
    /// meaning the solver uses the constraint's own preference order.
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

/// Bridge an `AsyncElement` transform into a `LegacyTransform`
/// constraint. The returned constraint borrows `transform` for `'a`.
/// Non-boundary transforms get `propose_output = clone(input)`;
/// boundary transforms route to `AsyncElement::propose_output_caps`.
/// Used by the runner during the migration window to feed legacy
/// elements to the M16 solver.
pub fn legacy_transform_constraint<'a, T: AsyncElement + ?Sized>(
    transform: &'a T,
) -> CapsConstraint<'a> {
    CapsConstraint::LegacyTransform {
        intercept: Box::new(move |upstream: &Caps| transform.intercept_caps(upstream)),
        propose_output: Box::new(move |input: &Caps| transform.propose_output_caps(input)),
    }
}

/// Bridge an `AsyncElement` sink into a `LegacySink` constraint. The
/// returned constraint borrows `sink` for `'a`.
pub fn legacy_sink_constraint<'a, S: AsyncElement + ?Sized>(sink: &'a S) -> CapsConstraint<'a> {
    CapsConstraint::LegacySink(Box::new(move |upstream: &Caps| sink.intercept_caps(upstream)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::{Dim, Rate, VideoCodec, RawVideoFormat};
    use alloc::vec;

    fn video(format: RawVideoFormat, w: Dim, h: Dim, r: Rate) -> Caps {
        Caps::RawVideo { format, width: w, height: h, framerate: r }
    }

    fn compressed(codec: VideoCodec, w: Dim, h: Dim, r: Rate) -> Caps {
        Caps::CompressedVideo { codec, width: w, height: h, framerate: r }
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
                Caps::CompressedVideo { width, height, framerate, .. } => CapsSet::one(Caps::RawVideo {
                    format: RawVideoFormat::Nv12,
                    width: width.clone(),
                    height: height.clone(),
                    framerate: framerate.clone(),
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
        let input = compressed(VideoCodec::H264, Dim::Fixed(1280), Dim::Fixed(720), Rate::Fixed(30 << 16));
        let c = d.caps_constraint();
        match c {
            CapsConstraint::DerivedOutput(f) => {
                let out = f(&input);
                assert_eq!(
                    out.alternatives(),
                    &[video(RawVideoFormat::Nv12, Dim::Fixed(1280), Dim::Fixed(720), Rate::Fixed(30 << 16))]
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
                let in_a = CapsSet::one(video(RawVideoFormat::Nv12, Dim::Fixed(1920), Dim::Fixed(1080), Rate::Any));
                let out_a = CapsSet::one(video(RawVideoFormat::Nv12, Dim::Fixed(1280), Dim::Fixed(720), Rate::Any));
                let in_b = CapsSet::one(video(RawVideoFormat::I420, Dim::Fixed(1920), Dim::Fixed(1080), Rate::Any));
                let out_b = CapsSet::one(video(RawVideoFormat::I420, Dim::Fixed(1280), Dim::Fixed(720), Rate::Any));
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
        // ACCEPT_CAPS (DESIGN §7): pure check against the declared set.
        let nv12_720 = video(RawVideoFormat::Nv12, Dim::Fixed(1280), Dim::Fixed(720), Rate::Any);
        let h264_720 = compressed(VideoCodec::H264, Dim::Fixed(1280), Dim::Fixed(720), Rate::Any);

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
    fn legacy_transform_bridge_wraps_async_element() {
        use crate::element::{AsyncElement, ConfigureOutcome, OutputSink};
        use crate::frame::PipelinePacket;
        use core::future::{ready, Ready};

        // Minimal AsyncElement impl that narrows width to 640 and
        // declares itself a boundary that re-formats to NV12.
        struct FakeXform;
        impl AsyncElement for FakeXform {
            type ProcessFuture<'a> = Ready<Result<(), G2gError>>;
            fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
                match upstream.dims() {
                    Some((_w, height, framerate)) => Ok(Caps::CompressedVideo {
                        codec: VideoCodec::H264,
                        width: Dim::Fixed(640),
                        height: height.clone(),
                        framerate: framerate.clone(),
                    }),
                    None => Err(G2gError::CapsMismatch),
                }
            }
            fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
                Ok(ConfigureOutcome::Accepted)
            }
            fn process<'a>(
                &'a mut self,
                _: PipelinePacket,
                _: &'a mut dyn OutputSink,
            ) -> Self::ProcessFuture<'a> {
                ready(Ok(()))
            }
            fn is_format_boundary(&self) -> bool {
                true
            }
            fn propose_output_caps(&self, input: &Caps) -> Caps {
                match input {
                    Caps::CompressedVideo { width, height, framerate, .. } => Caps::RawVideo {
                        format: RawVideoFormat::Nv12,
                        width: width.clone(),
                        height: height.clone(),
                        framerate: framerate.clone(),
                    },
                    other => other.clone(),
                }
            }
        }

        let xf = FakeXform;
        let c = legacy_transform_constraint(&xf);
        let upstream = compressed(VideoCodec::H264, Dim::Fixed(1920), Dim::Fixed(720), Rate::Fixed(30 << 16));
        match &c {
            CapsConstraint::LegacyTransform { intercept, propose_output } => {
                let narrowed = intercept(&upstream).unwrap();
                assert_eq!(narrowed, compressed(VideoCodec::H264, Dim::Fixed(640), Dim::Fixed(720), Rate::Fixed(30 << 16)));
                let out = propose_output(&narrowed);
                assert_eq!(out, video(RawVideoFormat::Nv12, Dim::Fixed(640), Dim::Fixed(720), Rate::Fixed(30 << 16)));
            }
            _ => panic!("expected LegacyTransform"),
        }
    }

    #[test]
    fn configure_link_signature_works_for_all_shapes() {
        let cap = video(RawVideoFormat::Nv12, Dim::Fixed(640), Dim::Fixed(480), Rate::Fixed(30 << 16));
        FakeSource.configure_link(None, Some(&cap)).unwrap();
        FakeSink.configure_link(Some(&cap), None).unwrap();
        FakeDecoder.configure_link(Some(&cap), Some(&cap)).unwrap();
    }
}
