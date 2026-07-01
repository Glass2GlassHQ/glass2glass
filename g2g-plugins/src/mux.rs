//! Interleaving muxer (M10). Combines N input streams into one output, ordering
//! every input's frames by presentation timestamp (M204). Negotiation is
//! per-input: each pad accepts and records its own caps; the merged output caps
//! are fixed at construction.
//!
//! Time-ordering (M204): frames are buffered per input in an [`InputAggregator`]
//! and released by smallest `pts_ns` only once every still-contributing input
//! has one queued, so the merged stream is globally PTS-ordered (the
//! `GstAggregator` collect-and-pick-earliest rule). A frame carries its own
//! caps, so reordering is format-safe. Earlier this forwarded frames in arrival
//! order; that interleaved two time-skewed inputs incorrectly. Ordering is by
//! PTS; container muxers that need decode-order (DTS) interleaving key on that
//! instead, a separate concern from this generic interleaver.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, Frame, G2gError,
    InputAggregator, MultiInputElement, OutputSink, PipelinePacket,
};

#[derive(Debug)]
pub struct InterleaveMux {
    inputs: usize,
    output: Caps,
    configured: Vec<Option<Caps>>,
    /// Per-input frame buffer; releases the globally earliest-PTS frame once
    /// every contributor has one queued (M204).
    agg: InputAggregator<Frame>,
}

impl InterleaveMux {
    pub fn new(inputs: usize, output: Caps) -> Self {
        assert!(inputs > 0, "InterleaveMux needs at least one input");
        Self {
            inputs,
            output,
            configured: vec![None; inputs],
            agg: InputAggregator::new(inputs),
        }
    }

    /// The caps input pad `input` was configured with, i.e. the result of
    /// that input's independent negotiation. `None` before configuration.
    pub fn input_caps(&self, input: usize) -> Option<&Caps> {
        self.configured.get(input).and_then(|c| c.as_ref())
    }
}

impl MultiInputElement for InterleaveMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.inputs
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Each input negotiates independently; the interleave accepts any
        // per-input caps (frames carry their own caps downstream).
        Ok(upstream_caps.clone())
    }

    /// M18 step 1: per-input wildcard. The interleave forwards frames
    /// straight through tagged with their own per-frame caps; it has
    /// no per-input format constraint. `AcceptsAny` is the native
    /// shape (analogous to `FakeSink`'s migration in M16 5c). Skips
    /// the dynamic intercept callback on the per-input solver path.
    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    /// M18 step 1: the merged output caps are fixed at construction
    /// (`InterleaveMux::new(_, output)`). `Produces` is the native
    /// shape for a static muxer output. Skips the legacy bridge so
    /// the downstream sink negotiation hits the all-native solver
    /// path.
    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        Ok(CapsConstraint::Produces(CapsSet::one(self.output.clone())))
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        self.configured[input] = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(self.output.clone())
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Funnel N-to-1 forwarder",
            "Generic/Muxer",
            "Interleaves N input streams into one output by presentation timestamp; the structural fan-in for text/data streams (the gst `funnel` analog).",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                // M204: buffer the frame; release any now globally-earliest
                // frames (every contributor has a head) in PTS order.
                PipelinePacket::DataFrame(frame) => self.agg.push(input, frame),
                // M22: a per-input Eos is informational (the runner aggregates
                // ends and emits the single merged Eos). M204: mark the input
                // ended so the merge can release frames held waiting on it, and
                // flush everything once the last input ends.
                PipelinePacket::Eos => self.agg.mark_ended(input),
                // Control packets (Segment, Flush, ...) carry no frame to order;
                // forward straight through.
                other => {
                    out.push(other).await?;
                    return Ok(());
                }
            }
            while let Some((_, frame)) = self.agg.take_earliest_by(|f| f.timing.pts_ns) {
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{Dim, Rate, RawVideoFormat};

    fn nv12(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(30 << 16),
        }
    }

    #[test]
    fn per_input_constraint_is_wildcard() {
        // M18 step 1: each input pad accepts any per-frame caps
        // because the interleave forwards frames straight through
        // tagged with their own caps. `AcceptsAny` is the native
        // shape.
        let mux = InterleaveMux::new(3, nv12(1920, 1080));
        for idx in 0..3 {
            let c = mux.caps_constraint_as_input(idx);
            assert!(
                matches!(c, CapsConstraint::AcceptsAny),
                "input {idx} should be AcceptsAny, got {c:?}"
            );
        }
    }

    #[test]
    fn output_constraint_is_produces_with_configured_output() {
        // M18 step 1: the merged output is fixed at construction.
        // `Produces(CapsSet::one(...))` is the native shape; the
        // solver fixates it identically to the pre-migration
        // `output_caps().fixate()` call.
        let out = nv12(1280, 720);
        let mux = InterleaveMux::new(2, out.clone());
        let c = mux.caps_constraint_for_output().unwrap();
        match c {
            CapsConstraint::Produces(set) => {
                assert_eq!(set.alternatives(), &[out]);
            }
            other => panic!("expected Produces, got {other:?}"),
        }
    }
}
