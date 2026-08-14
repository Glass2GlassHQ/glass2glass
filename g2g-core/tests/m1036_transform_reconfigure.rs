//! M1036: a transform that originates a `Reconfigure` (rather than relaying one
//! from downstream) reaches its upstream producer. The graph runner polls
//! `take_reconfigure` on the transform arm and stores the request on the
//! transform's input link, where the source sees it as
//! `PushOutcome::Reconfigure` on its next push. The VAAPI decoders use this for
//! a mid-stream resolution change.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::element::Reconfigure;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{block_on, run_graph, run_source_transform_sink, GraphNode, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, Frame, FrameTiming, G2gError, Graph, OutputSink,
    PipelineClock, PipelinePacket, PushOutcome, Rate, RawVideoFormat,
};

/// Frames pushed by the source. Enough that the bounded links force the
/// transform to run (and park its request) well before the last push.
const FRAME_COUNT: u64 = 32;
const LINK_CAPACITY: usize = 2;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    video_caps(2)
}

/// The geometry the transform asks its upstream for, distinct from the
/// negotiated caps so the payload is identifiable at the source.
fn proposed_caps() -> Caps {
    video_caps(4)
}

fn video_caps(side: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(side),
        height: Dim::Fixed(side),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

fn frame(sequence: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 16]))),
        FrameTiming {
            pts_ns: sequence * 33_000_000,
            ..FrameTiming::default()
        },
        sequence,
    ))
}

/// Pushes frames then `Eos`, recording every reverse signal a push reports.
struct RecordingSrc {
    observed: Arc<Mutex<Vec<Reconfigure>>>,
}

impl SourceLoop for RecordingSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(caps()))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for i in 0..FRAME_COUNT {
                if let PushOutcome::Reconfigure(r) = out.push(frame(i)).await? {
                    self.observed.lock().unwrap().push(r);
                }
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(FRAME_COUNT)
        })
    }
}

/// Forwards everything and, on its first frame, parks one upstream proposal:
/// the decoder-reads-a-new-resolution shape, minus the decoder.
#[derive(Default)]
struct ProposingTransform {
    parked: Option<Reconfigure>,
    seen: u64,
}

impl AsyncElement for ProposingTransform {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn take_reconfigure(&mut self) -> Option<Reconfigure> {
        self.parked.take()
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if matches!(packet, PipelinePacket::DataFrame(_)) {
                self.seen += 1;
                if self.seen == 1 {
                    self.parked = Some(Reconfigure::Propose(proposed_caps()));
                }
            }
            if !matches!(packet, PipelinePacket::Eos) {
                out.push(packet).await?;
            }
            Ok(())
        })
    }
}

/// Plain consumer for the sink position.
struct Discard;

impl AsyncElement for Discard {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(core::future::ready(Ok(())))
    }
}

#[test]
fn transform_originated_reconfigure_reaches_the_source() {
    let observed = Arc::new(Mutex::new(Vec::new()));
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(RecordingSrc {
        observed: Arc::clone(&observed),
    }));
    let mid = g.add_transform(GraphNode::element(ProposingTransform::default()));
    let sink = g.add_sink(GraphNode::element(Discard));
    g.link(src, mid).unwrap();
    g.link(mid, sink).unwrap();

    let stats = block_on(run_graph(g, &ZeroClock, LINK_CAPACITY)).expect("graph runs");
    assert_eq!(stats.frames_consumed, FRAME_COUNT);

    assert_proposal_observed(&observed);
}

#[test]
fn the_linear_runner_relays_it_too() {
    let observed = Arc::new(Mutex::new(Vec::new()));
    let mut source = RecordingSrc {
        observed: Arc::clone(&observed),
    };
    let mut transform = ProposingTransform::default();
    let mut sink = Discard;

    let stats = block_on(run_source_transform_sink(
        &mut source,
        &mut transform,
        &mut sink,
        &ZeroClock,
        LINK_CAPACITY,
    ))
    .expect("pipeline runs");
    assert_eq!(stats.frames_consumed, FRAME_COUNT);

    assert_proposal_observed(&observed);
}

fn assert_proposal_observed(observed: &Mutex<Vec<Reconfigure>>) {
    assert_eq!(
        observed.lock().unwrap().as_slice(),
        &[Reconfigure::Propose(proposed_caps())],
        "the source must see the transform's proposal exactly once"
    );
}
