//! Content-based demultiplexer (M205): one input, N typed output ports.
//!
//! The `pad-added` / dynamic-pad analog for the bounded-N case (DESIGN.md
//! §4.9.3 "dark slots"). A container or multiplexed stream carries several
//! elementary streams; this element splits them onto distinct output ports, one
//! per stream, each with its own caps. It is a [`MultiOutputElement`] driven by
//! [`run_source_fanout`](g2g_core::runtime::run_source_fanout), so a single
//! demuxer feeds multiple downstream branches in one pipeline, rather than
//! instantiating one single-output demuxer per stream.
//!
//! The N output ports are the pre-allocated "dark slots": fixed at construction,
//! each carrying a declared [`Caps`]. A frame is routed to a port by a
//! classifier the caller supplies (`Fn(&Frame) -> usize`), the "content-based
//! demux" hook the fan-out layer anticipates. The first frame routed to a port
//! emits that port's [`PipelinePacket::CapsChanged`] ahead of it, so the branch
//! retypes from the demuxer's (byte-stream) input caps to the elementary
//! stream's caps, exactly as a single-output demuxer announces its output. A
//! port that no stream ever routes to simply stays dark (and receives the merged
//! EOS at end), the bounded-N realization of a demuxer whose container happens
//! not to carry that stream.
//!
//! Routing is generic: a container demuxer keys the classifier on its parsed
//! stream identity (e.g. an MPEG-TS PID / `stream_type`); the unit tests key on
//! a leading stream-id byte. Wiring this into the `gst-launch` text DSL
//! (`demux name=d  d. ! ...  d. ! ...`) and a `run_graph` demux node are the
//! follow-ups; today it runs through the fan-out runner programmatically.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::{
    Caps, ConfigureOutcome, Frame, G2gError, MultiOutputElement, MultiOutputSink, PipelinePacket,
};

/// Classifier mapping a frame to the output port it belongs on.
type Classifier = Box<dyn Fn(&Frame) -> usize + Send>;

/// A demultiplexer: routes each input frame to one of N typed output ports.
pub struct StreamDemux {
    /// The (byte-stream) caps this demuxer accepts on its single input.
    input: Caps,
    /// Per-output-port caps; port `i` announces `port_caps[i]` before its first
    /// frame. These are the "dark slots".
    port_caps: Vec<Caps>,
    /// Picks the output port for a frame (clamped to the port range).
    classify: Classifier,
    /// Whether port `i` has emitted its opening `CapsChanged` yet.
    announced: Vec<bool>,
}

impl core::fmt::Debug for StreamDemux {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StreamDemux")
            .field("input", &self.input)
            .field("port_caps", &self.port_caps)
            .finish_non_exhaustive()
    }
}

impl StreamDemux {
    /// A demuxer accepting `input` caps, with one output port per entry of
    /// `port_caps`, routing each frame via `classify`. `classify`'s result is
    /// clamped to a valid port, so an out-of-range classification lands on the
    /// last port rather than panicking.
    pub fn new(
        input: Caps,
        port_caps: Vec<Caps>,
        classify: impl Fn(&Frame) -> usize + Send + 'static,
    ) -> Self {
        assert!(
            !port_caps.is_empty(),
            "StreamDemux needs at least one output port"
        );
        let announced = alloc::vec![false; port_caps.len()];
        Self {
            input,
            port_caps,
            classify: Box::new(classify),
            announced,
        }
    }

    /// Number of output ports (the dark-slot count).
    pub fn port_count(&self) -> usize {
        self.port_caps.len()
    }
}

impl MultiOutputElement for StreamDemux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&self.input)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // Accept the negotiated input (the byte stream); per-port output caps are
        // announced from `process` as each stream first routes.
        absolute_caps
            .intersect(&self.input)
            .map(|_| ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a> {
        let ports = self.port_caps.len();
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let port = (self.classify)(&frame).min(ports - 1);
                    // Announce this port's caps before its first frame, so the
                    // branch retypes from the input (byte-stream) caps.
                    if !self.announced[port] {
                        out.push_to(
                            port,
                            PipelinePacket::CapsChanged(self.port_caps[port].clone()),
                        )
                        .await?;
                        self.announced[port] = true;
                    }
                    out.push_to(port, PipelinePacket::DataFrame(frame)).await?;
                }
                // Timing / flush apply to every branch, so broadcast them.
                PipelinePacket::Segment(seg) => {
                    for port in 0..ports {
                        out.push_to(port, PipelinePacket::Segment(seg)).await?;
                    }
                }
                PipelinePacket::Flush => {
                    for port in 0..ports {
                        out.push_to(port, PipelinePacket::Flush).await?;
                    }
                }
                // The input's own CapsChanged (the byte-stream caps) is consumed:
                // each output port defines its own caps, announced per port above.
                PipelinePacket::CapsChanged(_) => {}
                // The runner broadcasts the single merged Eos to every port after
                // this returns; the element must not forward it.
                PipelinePacket::Eos => {}
                // future PipelinePacket variants: no-op.
                _ => {}
            }
            Ok(())
        })
    }
}
