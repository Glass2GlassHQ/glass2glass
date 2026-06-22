//! M208 `gst-launch` multi-stream MPEG-TS fan-in: a text pipeline that joins an
//! H.264 video stream and an AAC audio stream at `mpegtsmux name=m` builds the
//! multi-input `tsmuxn::TsMux` (the muxer is picked over the single-input
//! `tsmux::TsMux` by link degree), and runs end to end through `run_graph`. The
//! byte-level correctness of the multiplex is covered by `m207_ts_av_mux`; this
//! test proves the launch text path reaches the multi-input muxer and that the
//! one `mpegtsmux` name covers both the single- and multi-input shapes.

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{parse_launch, run_graph, DynSourceLoop, LaunchFactory, Registry, SourceFactory, SourceLoop};
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, ConfigureOutcome, Dim, FrameTiming, G2gError,
    MemoryDomain, NodeKind, OutputSink, PipelineClock, PipelinePacket, Rate, VideoCodec,
};
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        // Fixed (not Any): the muxer runner fixates source caps before run.
        framerate: Rate::Fixed(30 << 16),
    }
}
fn aac_caps() -> Caps {
    Caps::Audio { format: AudioFormat::Aac, channels: 2, sample_rate: 48_000 }
}

/// Emits a fixed script of (access-unit, pts_ns) for one elementary stream, then
/// EOS. A `fn()`-buildable source so it can register with the launch parser.
struct AuSrc {
    caps: Caps,
    aus: Vec<(Vec<u8>, u64)>,
    configured: bool,
}

impl SourceLoop for AuSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.caps.clone()))
    }
    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        let aus = self.aus.clone();
        let configured = self.configured;
        Box::pin(async move {
            assert!(configured, "runner configures before run");
            for (i, (au, pts)) in aus.iter().enumerate() {
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(au.clone().into_boxed_slice())),
                    FrameTiming { pts_ns: *pts, ..FrameTiming::default() },
                    i as u64,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
                tokio::task::yield_now().await;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(aus.len() as u64)
        })
    }
}

fn build_h264() -> Box<dyn DynSourceLoop> {
    // Video AUs at 0/40/80 ms (3 frames).
    Box::new(AuSrc {
        caps: h264_caps(),
        aus: vec![
            (vec![0u8, 0, 0, 1, 0x65, 0x11], 0),
            (vec![0u8, 0, 0, 1, 0x41, 0x22], 40_000_000),
            (vec![0u8, 0, 0, 1, 0x41, 0x33], 80_000_000),
        ],
        configured: false,
    })
}

fn build_aac() -> Box<dyn DynSourceLoop> {
    // Audio AUs at 20/60 ms (2 frames), interleaving the video timeline.
    Box::new(AuSrc {
        caps: aac_caps(),
        aus: vec![(vec![0xFFu8, 0xF1, 0xAA], 20_000_000), (vec![0xFFu8, 0xF1, 0xBB], 60_000_000)],
        configured: false,
    })
}

/// Accepts any packet (the muxer output is a `Caps::ByteStream`) without
/// `FakeSink`'s monotonic-sequence assertion (interleaved AUs reuse sequence 0/1).
#[derive(Default)]
struct AnySink;
impl AsyncElement for AnySink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;
    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

fn registry_with_av_sources() -> Registry {
    let mut reg = default_registry();
    reg.register_source(SourceFactory::new("h264src", h264_caps(), build_h264));
    reg.register_source(SourceFactory::new("aacsrc", aac_caps(), build_aac));
    reg.register_launch(LaunchFactory::new("anysink", Vec::new(), || Box::new(AnySink)));
    reg
}

#[tokio::test]
async fn mpegtsmux_fans_in_audio_and_video() {
    let reg = registry_with_av_sources();
    // Heterogeneous fan-in: H.264 video + AAC audio join at the muxer. Each AU
    // becomes one muxed TS byte frame (3 video + 2 audio = 5), interleaved by PTS.
    let graph = parse_launch(
        &reg,
        "h264src ! m.   aacsrc ! m.   mpegtsmux name=m ! anysink",
    )
    .expect("A+V fan-in pipeline parses");

    let stats = run_graph(graph, &ZeroClock, 4).await.expect("A+V TS mux pipeline runs");
    assert_eq!(stats.frames_consumed, 5, "all five AUs (3 video + 2 audio) muxed into TS frames");
}

#[tokio::test]
async fn multi_input_mpegtsmux_builds_a_two_input_muxer_node() {
    let reg = registry_with_av_sources();
    let graph = parse_launch(
        &reg,
        "h264src ! m.   aacsrc ! m.   mpegtsmux name=m ! anysink",
    )
    .expect("parses");
    let vg = graph.finish().expect("valid graph");
    let muxers: Vec<NodeKind> = vg
        .topo()
        .iter()
        .map(|&n| vg.kind(n))
        .filter(|k| matches!(k, NodeKind::Muxer(_)))
        .collect();
    // Two inbound links select the multi-input `tsmuxn::TsMux` over the
    // single-input launch element registered under the same name.
    assert_eq!(muxers, [NodeKind::Muxer(2)], "one muxer node with two input pads");
}

#[tokio::test]
async fn single_input_mpegtsmux_stays_a_transform() {
    let reg = registry_with_av_sources();
    // One input: `mpegtsmux` resolves to the single-stream launch element (a
    // transform node), not the fan-in muxer. It must not appear as a Muxer node.
    let graph = parse_launch(&reg, "h264src ! mpegtsmux ! anysink").expect("parses");
    let vg = graph.finish().expect("valid graph");
    let has_muxer = vg.topo().iter().any(|&n| matches!(vg.kind(n), NodeKind::Muxer(_)));
    assert!(!has_muxer, "single-input mpegtsmux is a transform, not a muxer node");
}
