//! Multi-stream MPEG-TS multiplexer element (M207): N elementary streams in
//! (e.g. H.264 video + AAC audio), one MPEG-TS byte stream out. The A+V analog
//! of the single-input [`crate::tsmux::TsMux`], the everyday live-streaming
//! container case.
//!
//! A [`MultiInputElement`]: each input pad accepts one elementary stream
//! (`Caps::CompressedVideo{H264|H265}` or `Caps::Audio{Aac}`), and the access
//! units are interleaved into one program by presentation timestamp before being
//! written to their per-stream PIDs. Time-ordering reuses the M204
//! [`InputAggregator::take_earliest_by`](g2g_core::InputAggregator::take_earliest_by)
//! merge (release the globally earliest AU once every contributing input has one
//! queued), so the muxed TS is monotonic across streams the way a decoder
//! expects. The PMT (built once all inputs are configured) names every stream.
//!
//! Scope: one program, no PCR (see [`crate::mpegts::TsMuxer`]). Reachable from
//! the `gst-launch` fan-in syntax (M208): registered as the `mpegtsmux` muxer in
//! [`default_registry`](crate::registry::default_registry), so
//! `v.! m.  a.! m.  mpegtsmux name=m ! sink` builds this element when more than
//! one input links (a single input still builds the single-stream
//! [`crate::tsmux::TsMux`]). Also runs programmatically through `run_muxer_sink`
//! / a `run_graph` muxer node.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, FrameTiming, G2gError,
    InputAggregator, MemoryDomain, MultiInputElement, OutputSink, PipelinePacket,
};

use crate::mpegts::TsMuxer;
use crate::tsmux::stream_type_for;

/// Muxes N elementary streams into one MPEG-TS byte stream, PTS-ordered.
#[derive(Debug)]
pub struct TsMux {
    inputs: usize,
    /// PMT stream type per input pad, learned at configure; the muxer is built
    /// once all are known.
    stream_types: Vec<Option<u8>>,
    /// Built lazily once every input is configured (the PMT needs all streams).
    mux: Option<TsMuxer>,
    /// Per-input AU buffer; releases the globally earliest-PTS AU (M204).
    agg: InputAggregator<Frame>,
    emitted: u64,
}

impl TsMux {
    /// A muxer with `inputs` input pads. Each pad's stream type is determined
    /// from its negotiated caps; the order of inputs is the order of streams in
    /// the PMT (and their PID assignment).
    pub fn new(inputs: usize) -> Self {
        assert!(inputs > 0, "TsMux needs at least one input");
        Self {
            inputs,
            stream_types: alloc::vec![None; inputs],
            mux: None,
            agg: InputAggregator::new(inputs),
            emitted: 0,
        }
    }

    /// Count of TS byte frames emitted.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn output_caps_value() -> Caps {
        Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs }
    }
}

impl MultiInputElement for TsMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.inputs
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if stream_type_for(upstream_caps).is_some() {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        // Each pad forwards its stream verbatim (frames carry their own caps);
        // the per-pad stream type is pinned at `configure_pipeline`, which rejects
        // an unsupported caps. `AcceptsAny` is the native muxer-input shape (as in
        // `InterleaveMux`); the legacy intercept-narrowing path is skipped.
        CapsConstraint::AcceptsAny
    }

    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        Ok(CapsConstraint::Produces(CapsSet::one(Self::output_caps_value())))
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        let stream_type = stream_type_for(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        self.stream_types[input] = Some(stream_type);
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(Self::output_caps_value())
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                // M204: buffer the AU; release the globally earliest below.
                PipelinePacket::DataFrame(frame) => self.agg.push(input, frame),
                // M22: a per-input Eos lets the merge release AUs held waiting on
                // this input, and flush its tail; the runner emits the merged Eos.
                PipelinePacket::Eos => self.agg.mark_ended(input),
                // CapsChanged is consumed by the runner's muxer arm; geometry /
                // params do not change the TS framing.
                PipelinePacket::CapsChanged(_) => return Ok(()),
                other => {
                    out.push(other).await?;
                    return Ok(());
                }
            }

            // The PMT needs every stream type, so build the muxer only once all
            // inputs are configured (always true by the first frame: the runner
            // negotiates every input pad before any process call).
            if self.mux.is_none() {
                if self.stream_types.iter().any(|s| s.is_none()) {
                    return Ok(());
                }
                let types: Vec<u8> = self.stream_types.iter().map(|s| s.expect("all set")).collect();
                self.mux = Some(TsMuxer::with_streams(&types));
            }

            // Drain every AU now safe to emit, in global PTS order, writing each
            // to its stream's PID.
            while let Some((stream, frame)) = self.agg.take_earliest_by(|f| f.timing.pts_ns) {
                let MemoryDomain::System(slice) = &frame.domain else {
                    return Err(G2gError::UnsupportedDomain);
                };
                let pts_90khz = (frame.timing.pts_ns as u128 * 90_000 / 1_000_000_000) as u64;
                let ts = self
                    .mux
                    .as_mut()
                    .expect("built above")
                    .push_au_on(stream, slice.as_slice(), Some(pts_90khz));
                let out_frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(ts.into_boxed_slice())),
                    FrameTiming { pts_ns: frame.timing.pts_ns, ..FrameTiming::default() },
                    self.emitted,
                );
                self.emitted += 1;
                out.push(PipelinePacket::DataFrame(out_frame)).await?;
            }
            Ok(())
        })
    }
}
