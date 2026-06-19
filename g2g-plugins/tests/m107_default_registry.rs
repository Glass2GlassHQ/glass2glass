//! M107 coverage broadening: the pre-populated `default_registry` plus the newly
//! property-enabled video/audio transforms, exercised through `parse_launch` end
//! to end. A `gst-launch` string now builds and runs a real multi-stage pipeline
//! with no hand-registration.
//!
//! `default_registry` (and `filesink`) are `std`-gated, so this file is too: run
//! with `cargo test -p g2g-plugins --features std`.
#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{PipelineClock, PropValue};
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn video_chain_parses_and_runs() {
    let reg = default_registry();
    // Scale then flip, all RGBA, driven entirely from the text pipeline.
    let graph = parse_launch(
        &reg,
        "videotestsrc num-buffers=4 ! videoscale width=160 height=120 ! videoflip method=rotate-180 ! fakesink",
    )
    .expect("video pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4).await.expect("video pipeline runs");
    assert_eq!(stats.frames_consumed, 4, "all frames reached the sink");
}

#[tokio::test]
async fn videoconvert_format_property_runs() {
    let reg = default_registry();
    // 320x240 RGBA -> NV12 (even dims), exercising the format= string property.
    let graph = parse_launch(&reg, "videotestsrc num-buffers=2 ! videoconvert format=nv12 ! fakesink")
        .expect("parses");
    let stats = run_graph(graph, &ZeroClock, 4).await.expect("runs");
    assert_eq!(stats.frames_consumed, 2);
}

#[tokio::test]
async fn audio_chain_parses_and_runs() {
    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        "audiotestsrc num-buffers=3 freq=440 ! audioconvert channels=1 ! audioresample samplerate=16000 ! fakesink",
    )
    .expect("audio pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4).await.expect("audio pipeline runs");
    assert_eq!(stats.frames_consumed, 3);
}

#[tokio::test]
async fn inline_caps_filter_parses_and_runs() {
    // M117: the gst-launch caps-description shorthand becomes a capsfilter,
    // pinning videotestsrc's output format / geometry.
    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        "videotestsrc num-buffers=3 ! video/x-raw,format=rgba,width=320,height=240,framerate=30/1 ! fakesink",
    )
    .expect("inline caps pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4).await.expect("inline caps pipeline runs");
    assert_eq!(stats.frames_consumed, 3, "frames pass the caps filter to the sink");
}

#[test]
fn default_registry_inspects_new_elements() {
    let reg = default_registry();
    assert!(reg.inspect("videoscale").unwrap().contains("width"));
    assert!(reg.inspect("videocrop").unwrap().contains("height"));
    assert!(reg.inspect("videoconvert").unwrap().contains("format"));
    assert!(reg.inspect("audiotestsrc").unwrap().contains("freq"));
    assert!(reg.inspect("audioconvert").unwrap().contains("channels"));
    assert!(reg.inspect("audioresample").unwrap().contains("samplerate"));
    assert!(reg.inspect("filesink").unwrap().contains("location"));

    let names = reg.element_names();
    for expected in
        ["videotestsrc", "audiotestsrc", "videoscale", "videoconvert", "audioconvert", "fakesink"]
    {
        assert!(names.contains(&expected), "registry has {expected}");
    }
}

#[test]
fn new_elements_property_round_trip_by_name() {
    let reg = default_registry();

    let mut scale = reg.make_element("videoscale").unwrap();
    scale.set_property("width", PropValue::Uint(640)).unwrap();
    scale.set_property("height", PropValue::Uint(360)).unwrap();
    assert_eq!(scale.get_property("width"), Some(PropValue::Uint(640)));
    assert_eq!(scale.get_property("height"), Some(PropValue::Uint(360)));

    let mut conv = reg.make_element("videoconvert").unwrap();
    conv.set_property("format", PropValue::Str("i420".into())).unwrap();
    assert_eq!(conv.get_property("format"), Some(PropValue::Str("i420".into())));

    let mut ac = reg.make_element("audioconvert").unwrap();
    ac.set_property("format", PropValue::Str("f32le".into())).unwrap();
    ac.set_property("channels", PropValue::Uint(2)).unwrap();
    assert_eq!(ac.get_property("format"), Some(PropValue::Str("f32le".into())));

    let mut sink = reg.make_element("filesink").unwrap();
    sink.set_property("location", PropValue::Str("/tmp/out.bin".into())).unwrap();
    assert_eq!(sink.get_property("location"), Some(PropValue::Str("/tmp/out.bin".into())));

    let mut ats = reg.make_source("audiotestsrc").unwrap();
    ats.set_property("wave", PropValue::Str("square".into())).unwrap();
    assert_eq!(ats.get_property("wave"), Some(PropValue::Str("square".into())));
}

#[test]
fn demuxer_and_its_parsers_registered() {
    // M109: tsdemux gains a stream selector and the parsers it feeds (h265parse,
    // aacparse) join the default registry so the audio / H.265 chains build by name.
    let reg = default_registry();
    assert!(reg.inspect("h265parse").is_some());
    assert!(reg.inspect("aacparse").is_some());
    assert!(reg.inspect("mpegtsmux").is_some()); // M114: the TS muxer
    assert!(reg.inspect("oggdemux").is_some()); // M116: the Ogg demuxer
    assert!(reg.inspect("tsdemux").unwrap().contains("stream"));

    let mut demux = reg.make_element("tsdemux").unwrap();
    assert_eq!(demux.get_property("stream"), Some(PropValue::Str("h264".into())));
    demux.set_property("stream", PropValue::Str("aac".into())).unwrap();
    assert_eq!(demux.get_property("stream"), Some(PropValue::Str("aac".into())));
}

#[test]
fn filesrc_registered_with_bytestream_format() {
    // M112: filesrc joins the registry; its bytestream-format property supplies
    // the container a raw byte stream lacks, so it can feed a demuxer as text.
    let reg = default_registry();
    assert!(reg.inspect("filesrc").unwrap().contains("bytestream-format"));

    let mut src = reg.make_source("filesrc").unwrap();
    src.set_property("location", PropValue::Str("/tmp/x.webm".into())).unwrap();
    src.set_property("bytestream-format", PropValue::Str("matroska".into())).unwrap();
    assert_eq!(src.get_property("bytestream-format"), Some(PropValue::Str("matroska".into())));
    src.set_property("bytestream-format", PropValue::Str("auto".into())).unwrap();
    assert_eq!(src.get_property("bytestream-format"), Some(PropValue::Str("auto".into())));
}

#[test]
fn matroska_demuxer_registered() {
    // M110: the MKV / WebM demuxer joins the registry with its stream selector.
    let reg = default_registry();
    assert!(reg.inspect("matroskademux").unwrap().contains("stream"));
    assert!(reg.inspect("matroskamux").is_some()); // M115: the MKV / WebM muxer

    let mut mkv = reg.make_element("matroskademux").unwrap();
    assert_eq!(mkv.get_property("stream"), Some(PropValue::Str("vp9".into())));
    mkv.set_property("stream", PropValue::Str("opus".into())).unwrap();
    assert_eq!(mkv.get_property("stream"), Some(PropValue::Str("opus".into())));
}
