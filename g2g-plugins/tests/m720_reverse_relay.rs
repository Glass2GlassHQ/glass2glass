//! M720 reverse-signal relay through intervening transforms: a downstream
//! keyframe request (`Reconfigure::ForceKeyframe`, the WebRTC PLI path) and a
//! bitrate target hop upstream past any transform that does not consume them
//! (`handles_keyframe_requests` / `handles_bitrate_requests` false), so
//! `enc ! h264parse ! webrtc-sink` shapes reach the encoder. Pure-fake
//! elements.

use core::future::{ready, Future, Ready};
use core::pin::Pin;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use g2g_core::runtime::{run_graph, run_source_transform_sink, GraphNode, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, Graph, OutputSink,
    PipelineClock, PipelinePacket, PushOutcome, Rate, RawVideoFormat, Reconfigure,
};
use g2g_plugins::identity::IdentityTransform;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn rgba(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Shared counters for the reverse signals an element observed on its pushes.
#[derive(Default)]
struct Seen {
    keyframes: AtomicU64,
    bitrate: AtomicU32,
}

/// Stand-in encoder: a pass-through transform that DOES consume keyframe /
/// bitrate signals (like a real encoder), recording what reaches it.
struct FakeEnc {
    seen: Arc<Seen>,
}

impl AsyncElement for FakeEnc {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn handles_keyframe_requests(&self) -> bool {
        true
    }
    fn handles_bitrate_requests(&self) -> bool {
        true
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match out.push(packet).await? {
                PushOutcome::Reconfigure(Reconfigure::ForceKeyframe) => {
                    self.seen.keyframes.fetch_add(1, Ordering::SeqCst);
                }
                PushOutcome::Bitrate(bps) => {
                    self.seen.bitrate.store(bps, Ordering::SeqCst);
                }
                _ => {}
            }
            Ok(())
        })
    }
}

/// Sink that raises one keyframe request and one bitrate target after its
/// second frame, via the runner-polled `take_*` hooks (the WebRTC sink shape).
struct RequestingSink {
    frames: u64,
    kf_sent: bool,
    bps_sent: bool,
}

impl RequestingSink {
    fn new() -> Self {
        Self {
            frames: 0,
            kf_sent: false,
            bps_sent: false,
        }
    }
}

impl AsyncElement for RequestingSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn take_reconfigure(&mut self) -> Option<Reconfigure> {
        (self.frames >= 2 && !core::mem::replace(&mut self.kf_sent, true))
            .then_some(Reconfigure::ForceKeyframe)
    }
    fn take_bitrate(&mut self) -> Option<u32> {
        (self.frames >= 2 && !core::mem::replace(&mut self.bps_sent, true)).then_some(500_000)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if matches!(packet, PipelinePacket::DataFrame(_)) {
                self.frames += 1;
            }
            Ok(())
        })
    }
}

/// Source that records reverse signals reaching all the way upstream.
struct ObservingSrc {
    frames: u64,
    seen: Arc<Seen>,
}

impl SourceLoop for ObservingSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = Ready<Result<Caps, G2gError>>;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        ready(Ok(rgba(8, 8)))
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::CapsChanged(rgba(8, 8))).await?;
            for seq in 0..self.frames {
                let buf = vec![0u8; 8 * 8 * 4];
                let frame = g2g_core::frame::Frame::new(
                    g2g_core::MemoryDomain::System(g2g_core::memory::SystemSlice::from_boxed(
                        buf.into_boxed_slice(),
                    )),
                    g2g_core::FrameTiming::default(),
                    seq,
                );
                match out.push(PipelinePacket::DataFrame(frame)).await? {
                    PushOutcome::Reconfigure(Reconfigure::ForceKeyframe) => {
                        self.seen.keyframes.fetch_add(1, Ordering::SeqCst);
                    }
                    PushOutcome::Bitrate(bps) => {
                        self.seen.bitrate.store(bps, Ordering::SeqCst);
                    }
                    _ => {}
                }
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.frames)
        })
    }
}

/// DAG runner: the requests cross a non-consuming `IdentityTransform` and stop
/// at the consuming stand-in encoder, never reaching the source.
#[tokio::test]
async fn relay_crosses_identity_and_stops_at_encoder() {
    let enc_seen = Arc::new(Seen::default());
    let src_seen = Arc::new(Seen::default());
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(ObservingSrc {
        frames: 12,
        seen: src_seen.clone(),
    }));
    let enc = g.add_transform(GraphNode::element(FakeEnc {
        seen: enc_seen.clone(),
    }));
    let ident = g.add_transform(GraphNode::element(IdentityTransform::new()));
    let sink = g.add_sink(GraphNode::element(RequestingSink::new()));
    g.link(src, enc).unwrap();
    g.link(enc, ident).unwrap();
    g.link(ident, sink).unwrap();

    run_graph(g, &ZeroClock, 2).await.expect("graph runs");
    assert!(
        enc_seen.keyframes.load(Ordering::SeqCst) >= 1,
        "the keyframe request crossed the identity transform to the encoder"
    );
    assert_eq!(
        enc_seen.bitrate.load(Ordering::SeqCst),
        500_000,
        "the bitrate target crossed the identity transform to the encoder"
    );
    assert_eq!(
        src_seen.keyframes.load(Ordering::SeqCst),
        0,
        "the encoder consumed the request; it did not leak to the source"
    );
    assert_eq!(src_seen.bitrate.load(Ordering::SeqCst), 0);
}

/// Linear runner: with only a non-consuming transform in the chain, the
/// requests relay all the way to the source.
#[tokio::test]
async fn linear_relay_reaches_the_source() {
    let src_seen = Arc::new(Seen::default());
    let mut src = ObservingSrc {
        frames: 12,
        seen: src_seen.clone(),
    };
    let mut ident = IdentityTransform::new();
    let mut sink = RequestingSink::new();
    run_source_transform_sink(&mut src, &mut ident, &mut sink, &ZeroClock, 2)
        .await
        .expect("linear runs");
    assert!(
        src_seen.keyframes.load(Ordering::SeqCst) >= 1,
        "the keyframe request crossed the identity transform to the source"
    );
    assert_eq!(src_seen.bitrate.load(Ordering::SeqCst), 500_000);
}
