//! M106 `gst-launch` text pipeline parser, end to end: a registry, a pipeline
//! string parsed into a `Graph`, then *run* through `run_graph`. This is the
//! payoff of the properties + registry track: a pipeline expressed as text,
//! constructed and configured by name, actually moving frames.

use g2g_core::runtime::{parse_launch, run_graph, LaunchFactory, ParseError, Registry, SourceFactory};
use g2g_core::{Caps, Dim, PipelineClock, Rate, RawVideoFormat};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videoflip::{FlipMethod, VideoFlip};
use g2g_plugins::videorate::VideoRate;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn rgba_any() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

fn registry() -> Registry {
    let mut reg = Registry::new();
    reg.register_source(SourceFactory::new("videotestsrc", rgba_any(), || {
        // num-buffers / dims / framerate come from the parsed properties.
        Box::new(VideoTestSrc::new(64, 48, 30, 0))
    }));
    reg.register_launch(LaunchFactory::of::<VideoFlip>("videoflip", || {
        Box::new(VideoFlip::new(FlipMethod::HorizontalMirror))
    }));
    reg.register_launch(LaunchFactory::new("videorate", Vec::new(), || {
        Box::new(VideoRate::new(30.0))
    }));
    reg.register_launch(LaunchFactory::new("fakesink", Vec::new(), || Box::new(FakeSink::new())));
    reg
}

#[tokio::test]
async fn parse_and_run_linear_pipeline() {
    let reg = registry();
    let graph = parse_launch(
        &reg,
        "videotestsrc num-buffers=3 pattern=snow ! videoflip method=rotate-180 ! fakesink",
    )
    .expect("pipeline parses");

    let stats = run_graph(graph, &ZeroClock, 4).await.expect("parsed pipeline runs");
    // The source's num-buffers=3 property reached it and the frames flowed all
    // the way through the flip to the sink.
    assert_eq!(stats.frames_emitted, 3, "source emitted num-buffers frames");
    assert_eq!(stats.frames_consumed, 3, "all frames reached the sink");
}

#[tokio::test]
async fn source_only_sink_pipeline_runs() {
    let reg = registry();
    // Minimal two-stage pipeline (no transforms): source straight to sink.
    let graph = parse_launch(&reg, "videotestsrc num-buffers=2 ! fakesink").expect("parses");
    let stats = run_graph(graph, &ZeroClock, 4).await.expect("runs");
    assert_eq!(stats.frames_consumed, 2);
}

#[tokio::test]
async fn unknown_element_is_reported() {
    let reg = registry();
    let err = parse_launch(&reg, "videotestsrc ! nosuchelement ! fakesink").unwrap_err();
    assert_eq!(err, ParseError::UnknownElement("nosuchelement".into()));

    let err = parse_launch(&reg, "nosuchsource ! fakesink").unwrap_err();
    assert_eq!(err, ParseError::UnknownSource("nosuchsource".into()));
}

#[tokio::test]
async fn bad_property_value_is_reported() {
    let reg = registry();
    // num-buffers wants an int; "lots" does not parse.
    let err = parse_launch(&reg, "videotestsrc num-buffers=lots ! fakesink").unwrap_err();
    assert!(
        matches!(err, ParseError::BadValue { ref key, .. } if key == "num-buffers"),
        "got {err:?}"
    );

    // An unknown property name on a known element.
    let err = parse_launch(&reg, "videotestsrc bogus=1 ! fakesink").unwrap_err();
    assert!(matches!(err, ParseError::UnknownProperty { ref key, .. } if key == "bogus"));
}
