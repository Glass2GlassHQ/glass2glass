//! M725 launch-registry wiring for the WebRTC session sinks: a `gst-launch`
//! line ends on `webrtcsessionsink` / `livekitsink` as a terminal fan-in node
//! (the M713 launch shape), with the track kinds read from each linked pad's
//! caps and the endpoint/room set via properties. Parse + topology only (a run
//! would dial the network).

#![cfg(feature = "webrtc")]

use core::future::{ready, Future, Ready};
use core::pin::Pin;

use g2g_core::graph::NodeKind;
use g2g_core::runtime::{parse_launch, SourceFactory, SourceLoop};
use g2g_core::{
    AudioFormat, Caps, ConfigureOutcome, Dim, G2gError, OutputSink, PipelinePacket, Rate,
    VideoCodec,
};
use g2g_plugins::registry::default_registry;

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn opus_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Opus,
        channels: 2,
        sample_rate: 48_000,
    }
}

/// Zero-frame typed source, enough for parse + negotiation shape.
struct NullSrc(Caps);

impl SourceLoop for NullSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = Ready<Result<Caps, G2gError>>;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        ready(Ok(self.0.clone()))
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::Eos).await?;
            Ok(0)
        })
    }
}

fn registry() -> g2g_core::runtime::Registry {
    let mut reg = default_registry();
    reg.register_source(SourceFactory::new("h264src", h264_caps(), || {
        Box::new(NullSrc(Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Fixed(30 << 16),
        }))
    }));
    reg.register_source(SourceFactory::new("opussrc", opus_caps(), || {
        Box::new(NullSrc(Caps::Audio {
            format: AudioFormat::Opus,
            channels: 2,
            sample_rate: 48_000,
        }))
    }));
    reg
}

#[test]
fn session_sink_launch_line_builds_terminal_fanin() {
    let reg = registry();
    let vg = parse_launch(
        &reg,
        "h264src ! s.   opussrc ! s.   webrtcsessionsink name=s location=http://sfu/whip",
    )
    .expect("parses")
    .finish()
    .expect("valid graph");
    let fanins: Vec<NodeKind> = vg
        .topo()
        .iter()
        .map(|&n| vg.kind(n))
        .filter(|k| matches!(k, NodeKind::FaninSink(_)))
        .collect();
    assert_eq!(
        fanins,
        [NodeKind::FaninSink(2)],
        "one 2-input terminal node"
    );
}

#[cfg(feature = "webrtc-livekit")]
#[test]
fn livekitsink_launch_line_builds_terminal_fanin() {
    let reg = registry();
    let vg = parse_launch(
        &reg,
        "h264src ! s.   opussrc ! s.   livekitsink name=s url=ws://h:7880 room=demo identity=pub api-key=devkey api-secret=secret",
    )
    .expect("parses")
    .finish()
    .expect("valid graph");
    let fanins: Vec<NodeKind> = vg
        .topo()
        .iter()
        .map(|&n| vg.kind(n))
        .filter(|k| matches!(k, NodeKind::FaninSink(_)))
        .collect();
    assert_eq!(fanins, [NodeKind::FaninSink(2)]);
}
