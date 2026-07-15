//! M197 (coverage): complex branching pipelines, end to end. A `decodebin`
//! feeding a `tee` whose branches process independently, the canonical
//! "decode once, display on one branch and store on another" shape. Each branch
//! carries its own `queue` and converters, so branches can target different
//! formats / geometries off the tee's single broadcast, exactly as a gst line
//! does. Uses a stub decoder (templates drive the auto-plug; the bodies forward
//! frames) so it runs without a real codec. Gated off when a real multi-codec
//! decoder (`ffmpegdec`, which handles VP9) is compiled in: the search would
//! correctly prefer it over the stub, and it would then try to actually decode
//! the synthetic frames. The branching behaviour under test is decoder
//! independent, so the baseline build covers it; m118 covers tee under all
//! features.
#![cfg(all(feature = "std", not(any(feature = "ffmpeg", feature = "vaapi"))))]

use g2g_core::runtime::{parse_launch, run_graph, ElementFactory, SourceFactory};
use g2g_core::{Caps, CapsSet, Dim, PadTemplate, PipelineClock, Rate, RawVideoFormat, VideoCodec};
use g2g_plugins::identity::IdentityTransform;
use g2g_plugins::registry::default_registry;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn vp9() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::Vp9,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

fn nv12() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

/// `default_registry` plus a fake VP9 source and a stub VP9 -> raw decoder, so
/// `decodebin` resolves without a real codec feature. VP9 has no in-tree decoder,
/// so the stub is never out-competed by a real one (which would choke on the fake
/// frames), keeping this deterministic across feature builds.
fn registry() -> g2g_core::runtime::Registry {
    let mut reg = default_registry();
    reg.register_source(SourceFactory::new("vp9src", vp9(), || {
        Box::new(VideoTestSrc::new(16, 16, 30, 4))
    }));
    reg.register(ElementFactory::new(
        "vp9dec",
        Vec::from([PadTemplate::sink(CapsSet::one(vp9())), PadTemplate::source(CapsSet::one(nv12()))]),
        |_| Box::new(IdentityTransform::new()),
    ));
    reg.register_launch(g2g_core::runtime::LaunchFactory::new(
        "vp9dec",
        Vec::new(),
        || Box::new(IdentityTransform::new()),
    ));
    reg
}

#[tokio::test]
async fn decode_then_tee_display_and_store() {
    // decode once, then fan out: one branch "displays" (convert + sink), the other
    // "stores" (a bare sink, standing in for filesink). Both get every frame.
    let reg = registry();
    let line = "vp9src ! decodebin ! tee name=t \
                ! queue ! videoconvert ! fakesink \
                t. ! queue ! fakesink";
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line:?} parse: {e}"));
    let stats = run_graph(graph, &ZeroClock, 4).await.expect("branching decode runs");
    assert_eq!(stats.frames_emitted, 4, "source emitted 4 frames");
    assert_eq!(stats.frames_consumed, 8, "both branches consumed all 4 frames");
}

#[tokio::test]
async fn tee_branches_target_different_formats() {
    // The tee broadcasts one format; each branch converts to its own off that,
    // just like gst (a per-branch videoconvert). NV12 on one, I420 on the other.
    let reg = registry();
    let line = "vp9src ! decodebin ! tee name=t \
                ! queue ! videoconvert ! video/x-raw,format=NV12 ! fakesink \
                t. ! queue ! videoconvert ! video/x-raw,format=I420 ! fakesink";
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line:?} parse: {e}"));
    let stats = run_graph(graph, &ZeroClock, 4).await.expect("differing-format branches run");
    assert_eq!(stats.frames_consumed, 8, "both format branches consumed all frames");
}

#[tokio::test]
async fn tee_branches_scale_independently() {
    // One branch rescales, the other passes through, off the same decoded stream.
    let reg = registry();
    let line = "vp9src ! decodebin ! tee name=t \
                ! queue ! videoscale ! video/x-raw,width=8,height=8 ! fakesink \
                t. ! queue ! fakesink";
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line:?} parse: {e}"));
    let stats = run_graph(graph, &ZeroClock, 4).await.expect("differing-scale branches run");
    assert_eq!(stats.frames_consumed, 8);
}
