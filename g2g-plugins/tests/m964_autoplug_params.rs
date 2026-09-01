//! M964: per-element property assignments for the auto-plug search
//! (`AutoplugParams`). Auto-plug picks the elements, so the caller never names
//! them; these params are how geometry / device / file-path knobs still reach
//! them, addressed by factory name and applied through the element's
//! `set_property` right after the factory builds it.

use g2g_core::runtime::{
    is_raw_video, run_graph, AutoplugError, AutoplugParams, ElementFactory, GraphNode,
    LaunchFactory, PlaybinPort, Registry, RunStats, SelectionContext, UriError, UriSourceFactory,
};
use g2g_core::{
    Caps, CapsSet, Dim, Graph, PadTemplate, PipelineClock, PropError, PropValue, Rate,
    RawVideoFormat, VideoCodec,
};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::h264parse::H264Parse;
use g2g_plugins::identity::IdentityTransform;
use g2g_plugins::mkvdemux::{MkvDemuxN, MkvStream};
use g2g_plugins::videoconvert::VideoConvert;
use g2g_plugins::videoscale::VideoScale;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn h264() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn rgba_any() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn is_nv12(c: &Caps) -> bool {
    matches!(
        c,
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            ..
        }
    )
}

/// The real scaler, built in auto mode (0x0 = take the geometry from the
/// negotiated caps). Any geometry it ends up with therefore came from the
/// params, not from the constructor.
fn videoscale_factory() -> ElementFactory {
    ElementFactory::of::<VideoScale>("videoscale", |_| Box::new(VideoScale::new(0, 0)))
}

/// The converter, configured from the output caps the search chose for it.
fn videoconvert_factory() -> ElementFactory {
    ElementFactory::of::<VideoConvert>("videoconvert", |out| match out {
        Caps::RawVideo { format, .. } => Box::new(VideoConvert::new(*format)),
        _ => unreachable!("autoplug only routes raw caps into videoconvert"),
    })
}

/// A decoder descriptor: H.264 in, raw NV12 out, bodied by an identity stand-in
/// (only the templates matter, the chain is never run).
fn decoder_factory() -> ElementFactory {
    let templates = Vec::from([
        PadTemplate::sink(CapsSet::one(h264())),
        PadTemplate::source(CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        })),
    ]);
    ElementFactory::new("h264dec", templates, |_| Box::new(IdentityTransform::new()))
}

#[test]
fn params_set_geometry_on_the_selected_element() {
    let mut reg = Registry::new();
    reg.register(videoscale_factory());

    // Without params the scaler stays in auto mode: the factory sets no geometry.
    let plain = reg
        .autoplug(&rgba_any(), &is_nv12, 4)
        .expect("videoscale is selected");
    assert_eq!(plain[0].get_property("width"), Some(PropValue::Uint(0)));

    let params = AutoplugParams::new()
        .set("videoscale", "width", PropValue::Uint(640))
        .set("videoscale", "height", PropValue::Uint(360));
    let chain = reg
        .autoplug_with_params(
            &rgba_any(),
            &is_nv12,
            4,
            SelectionContext::default(),
            &params,
        )
        .expect("videoscale is selected");
    assert_eq!(chain.len(), 1);
    assert_eq!(chain[0].get_property("width"), Some(PropValue::Uint(640)));
    assert_eq!(chain[0].get_property("height"), Some(PropValue::Uint(360)));
}

#[test]
fn unknown_property_on_a_selected_element_is_an_error() {
    let mut reg = Registry::new();
    reg.register(videoscale_factory());

    let params = AutoplugParams::new().set("videoscale", "device-index", PropValue::Uint(1));
    let Err(err) = reg.autoplug_with_params(
        &rgba_any(),
        &is_nv12,
        4,
        SelectionContext::default(),
        &params,
    ) else {
        panic!("an unknown property must fail loud rather than being skipped");
    };
    match err {
        AutoplugError::Property {
            element,
            property,
            source,
        } => {
            assert_eq!(element, "videoscale");
            assert_eq!(property, "device-index");
            assert_eq!(source, PropError::Unknown);
        }
        other => panic!("expected a property error, got {other:?}"),
    }
}

#[test]
fn a_value_the_element_rejects_is_an_error() {
    let mut reg = Registry::new();
    reg.register(videoscale_factory());

    // `width` is a Uint property; a string value is a type mismatch.
    let params = AutoplugParams::new().set("videoscale", "width", PropValue::Str("wide".into()));
    let Err(err) = reg.autoplug_with_params(
        &rgba_any(),
        &is_nv12,
        4,
        SelectionContext::default(),
        &params,
    ) else {
        panic!("a rejected value must fail loud");
    };
    assert!(
        matches!(
            err,
            AutoplugError::Property {
                source: PropError::Type,
                ..
            }
        ),
        "got {err:?}"
    );
}

#[test]
fn an_assignment_for_an_unselected_factory_is_unused() {
    // Both factories reach NV12 in one hop, so registration order selects the
    // converter and the scaler is never built. Its assignment is not an error:
    // addressing an element the search may or may not pick is legitimate.
    let mut reg = Registry::new();
    reg.register(videoconvert_factory())
        .register(videoscale_factory());

    let params = AutoplugParams::new()
        .set("videoscale", "width", PropValue::Uint(640))
        .set(
            "nosuchelement",
            "location",
            PropValue::Str("out.mp4".into()),
        );
    let chain = reg
        .autoplug_with_params(
            &rgba_any(),
            &is_nv12,
            4,
            SelectionContext::default(),
            &params,
        )
        .expect("an assignment for an unselected factory is unused, not an error");
    assert_eq!(chain.len(), 1);
    assert!(
        chain[0].get_property("width").is_none(),
        "the converter was selected, not the scaler"
    );
}

#[test]
fn params_reach_the_parser_decodebin_injects() {
    // The parser `decodebin` prepends is auto-plugged too, so it is addressable
    // by its launch name like any chain element.
    let mut reg = Registry::new();
    reg.register(decoder_factory())
        .register_launch(LaunchFactory::of::<H264Parse>("h264parse", || {
            Box::new(H264Parse::new())
        }))
        .set_parser_provider(|caps| match caps {
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            } => Some("h264parse"),
            _ => None,
        });

    let build = |params: &AutoplugParams| {
        let mut g: Graph<GraphNode> = Graph::new();
        let src = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, 1)));
        let sink = g.add_sink(GraphNode::element(FakeSink::new()));
        reg.decodebin_with_params(&mut g, src, sink, &h264(), &is_raw_video, 4, params)
    };

    let good = AutoplugParams::new().set("h264parse", "config-interval", PropValue::Int(2));
    let inserted = build(&good).expect("parser + decoder splice");
    assert_eq!(inserted.len(), 2, "h264parse ahead of the decoder");

    let bad = AutoplugParams::new().set("h264parse", "no-such-knob", PropValue::Int(2));
    let err = build(&bad).expect_err("the injected parser validates its assignments too");
    assert!(
        matches!(err, AutoplugError::Property { ref element, .. } if element == "h264parse"),
        "got {err:?}"
    );
}

#[tokio::test]
async fn decodebin_with_params_splices_a_runnable_chain() {
    let mut reg = Registry::new();
    reg.register(videoconvert_factory());

    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, 4)));
    let sink = g.add_sink(GraphNode::element(FakeSink::new()));
    let params = AutoplugParams::new().set("videoconvert", "format", PropValue::Str("NV12".into()));
    let inserted = reg
        .decodebin_with_params(&mut g, src, sink, &rgba_any(), &is_nv12, 4, &params)
        .expect("decodebin splices an RGBA -> NV12 converter");
    assert_eq!(inserted.len(), 1);

    let stats: RunStats = run_graph(g, &NullClock, 4)
        .await
        .expect("spliced graph runs");
    assert_eq!(stats.frames_emitted, 4);
    assert_eq!(stats.frames_consumed, 4);
}

/// A `mem://` handler whose source is irrelevant: the playbin-graph test only
/// assembles, it never runs.
fn mem_uri_build(
    _uri: &g2g_core::runtime::Uri,
) -> Result<(Box<dyn g2g_core::runtime::DynSourceLoop>, Caps), UriError> {
    Ok((
        Box::new(VideoTestSrc::new(8, 8, 30, 1)),
        Caps::ByteStream {
            encoding: g2g_core::ByteStreamEncoding::Matroska,
        },
    ))
}

#[test]
fn playbin_graph_params_reach_every_branch() {
    let mut reg = Registry::new();
    reg.register_uri(UriSourceFactory::new("mem", mem_uri_build))
        .register(decoder_factory());
    let ports = || {
        vec![PlaybinPort {
            input_caps: h264(),
            target: Box::new(is_raw_video),
            sink: Box::new(FakeSink::new()),
        }]
    };

    // An assignment for a factory no branch selected leaves assembly untouched.
    let unused = AutoplugParams::new().set("videoscale", "width", PropValue::Uint(640));
    let graph = reg
        .build_playbin_graph_with_params(
            "mem://clip.mkv",
            MkvDemuxN::new(vec![MkvStream::H264]),
            ports(),
            6,
            &unused,
        )
        .expect("playbin graph assembles");
    assert_eq!(graph.node_count(), 4, "source, demux, decoder, sink");

    // One addressed at the branch's decoder is applied there, so a bad name fails.
    let bad = AutoplugParams::new().set("h264dec", "no-such-knob", PropValue::Uint(1));
    let err = reg
        .build_playbin_graph_with_params(
            "mem://clip.mkv",
            MkvDemuxN::new(vec![MkvStream::H264]),
            ports(),
            6,
            &bad,
        )
        .expect_err("a branch element validates its assignments");
    assert!(
        matches!(err, AutoplugError::Property { ref element, .. } if element == "h264dec"),
        "got {err:?}"
    );
}

#[test]
fn build_playbin_with_params_reports_an_unknown_source() {
    let reg = Registry::new();
    let err = reg
        .build_playbin_with_params(
            "no-such-source",
            FakeSink::new(),
            &is_raw_video,
            4,
            &AutoplugParams::new(),
        )
        .unwrap_err();
    assert!(matches!(err, AutoplugError::UnknownSource), "got {err:?}");
}
