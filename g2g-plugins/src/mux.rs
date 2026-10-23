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

use g2g_core::log::Target;
use g2g_core::{
    g2g_error, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, Frame, G2gError,
    InputAggregator, MultiInputElement, OutputSink, PipelinePacket,
};

// the solver lets any caps onto a fan-in muxer pad and reports no refusal itself
pub(crate) fn refused_input(muxer: &'static str, caps: &Caps) -> G2gError {
    g2g_error!(
        Target::category(muxer),
        "cannot carry {}",
        caps.to_gst_string()
    );
    G2gError::CapsMismatch
}

// a muxer that carries the codec but not this stream beside the ones it already has
pub(crate) fn refused_input_because(muxer: &'static str, caps: &Caps, reason: &str) -> G2gError {
    g2g_error!(
        Target::category(muxer),
        "cannot carry {}: {reason}",
        caps.to_gst_string()
    );
    G2gError::CapsMismatch
}

/// # Example
///
/// ```no_run
/// use g2g_core::{AudioFormat, Caps, ChannelLayout};
/// use g2g_plugins::mux::InterleaveMux;
///
/// let format = AudioFormat::PcmS16Le;
/// let output = Caps::Audio {
///     format,
///     channels: 2,
///     sample_rate: 48_000,
///     channel_layout: ChannelLayout::UNSPECIFIED,
/// };
/// let element = InterleaveMux::new(2, output);
/// ```
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
    /// shape for a static muxer output.
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
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
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

    // the refusal log lines for `caps`, configured onto a fresh two-pad `M`
    #[cfg(feature = "std")]
    fn refusal_lines<M: MultiInputElement>(mut mux: M, caps: &Caps) -> Vec<alloc::string::String> {
        use g2g_core::log::{self, RingSink};
        // every sink sees every test's records, so two refusals logged at once mix
        static ONE_REFUSAL_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _serial = ONE_REFUSAL_AT_A_TIME
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let ring = RingSink::new(64);
        let sink = log::add_sink(alloc::boxed::Box::new(ring.clone()));
        let result = mux.configure_pipeline(1, caps);
        log::remove_sink(sink);
        assert_eq!(result.err(), Some(G2gError::CapsMismatch));
        let category = g2g_core::log::short_type_name::<M>();
        ring.drain()
            .into_iter()
            .filter(|record| record.category == category)
            .map(|record| record.message)
            .collect()
    }

    #[cfg(feature = "std")]
    #[test]
    fn every_fan_in_container_muxer_names_the_caps_it_refuses() {
        let raw = nv12(16, 16);
        let expected = alloc::vec![alloc::format!("cannot carry {}", raw.to_gst_string())];
        assert_eq!(
            refusal_lines(crate::mp4muxn::Mp4MuxN::new(2), &raw),
            expected
        );
        assert_eq!(
            refusal_lines(crate::mkvmuxn::MkvMuxN::new(2), &raw),
            expected
        );
        assert_eq!(refusal_lines(crate::tsmuxn::TsMux::new(2), &raw), expected);
        assert_eq!(
            refusal_lines(crate::flvmuxn::FlvMuxN::new(2), &raw),
            expected
        );
        assert_eq!(
            refusal_lines(crate::oggmuxn::OggMuxN::new(2), &raw),
            expected
        );
        assert_eq!(
            refusal_lines(crate::avimux::AviMuxN::new(2), &raw),
            expected
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn a_muxer_names_the_stream_it_has_no_room_for() {
        let h264 = Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::H264,
            width: Dim::Fixed(16),
            height: Dim::Fixed(16),
            framerate: Rate::Fixed(30 << 16),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        };
        let expected = |reason: &str| {
            alloc::vec![alloc::format!(
                "cannot carry {}: {reason}",
                h264.to_gst_string()
            )]
        };
        let mut flv = crate::flvmuxn::FlvMuxN::new(2);
        flv.configure_pipeline(0, &h264)
            .expect("the first video track");
        assert_eq!(
            refusal_lines(flv, &h264),
            expected("FLV holds one video and one audio track")
        );
        let mut avi = crate::avimux::AviMuxN::new(2);
        avi.configure_pipeline(0, &h264)
            .expect("the first video stream");
        assert_eq!(
            refusal_lines(avi, &h264),
            expected("AVI holds one video stream")
        );
        assert_eq!(
            refusal_lines(crate::mp4muxn::Mp4MuxN::new(2).with_cmaf(true), &h264),
            expected("a CMAF track file holds one stream, fragmented")
        );
    }
}
