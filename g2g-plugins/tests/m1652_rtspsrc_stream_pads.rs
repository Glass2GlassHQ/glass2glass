// `rtspsrc name=s  s. ! ...  s. ! ...` links a pad per stream, as gst's `rtspsrc` does: video on
// the first linked pad, audio on the second, off one session. One linked pad stays the
// video-only source.
#![cfg(all(feature = "std", feature = "rtsp", target_os = "linux"))]

use g2g_core::graph::NodeKind;
use g2g_core::runtime::parse_launch;
use g2g_plugins::registry::default_registry;

#[cfg(feature = "ffmpeg")]
#[path = "common/checksum_counts.rs"]
mod checksum_counts;
#[cfg(feature = "ffmpeg")]
#[path = "common/ffmpeg_tool.rs"]
mod ffmpeg_tool;
#[cfg(feature = "ffmpeg")]
#[path = "common/rtsp_av_relay.rs"]
mod rtsp_av_relay;

const URL: &str = "rtsp://127.0.0.1/stream";
const STREAM_PADS: u8 = 2;

fn node_kinds(line: &str) -> Vec<NodeKind> {
    let graph = parse_launch(&default_registry(), line)
        .unwrap_or_else(|e| panic!("parses `{line}`: {e}"))
        .finish()
        .expect("valid graph");
    graph.topo().iter().map(|&node| graph.kind(node)).collect()
}

#[test]
fn two_linked_pads_build_one_session_with_a_pad_each() {
    let kinds = node_kinds(&format!(
        "rtspsrc name=s location={URL}  s. ! fakesink  s. ! fakesink"
    ));
    assert!(
        kinds.contains(&NodeKind::FanoutSrc(STREAM_PADS)),
        "{kinds:?}"
    );
    assert!(!kinds.contains(&NodeKind::Tee(STREAM_PADS)), "{kinds:?}");
}

#[test]
fn one_linked_pad_is_the_video_source() {
    let kinds = node_kinds(&format!("rtspsrc location={URL} ! fakesink"));
    assert!(kinds.contains(&NodeKind::Source), "{kinds:?}");
}

#[cfg(feature = "ffmpeg")]
mod live {
    use std::time::Duration;

    use super::*;

    const AV_FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/av_h264_aac.mp4"
    );
    const SINKS: [&str; 2] = ["video", "audio"];
    // both tracks keep flowing, not just a first frame each
    const LIVE_FRAMES: usize = 10;
    const DEADLINE: Duration = Duration::from_secs(20);

    #[test]
    fn each_pad_carries_its_own_track() {
        if !ffmpeg_tool::have_ffmpeg() {
            eprintln!("ffmpeg not present; skipping the RTSP A/V pads");
            return;
        }
        let relay = rtsp_av_relay::serve_av_fixture(AV_FIXTURE);
        let line = format!(
            "rtspsrc name=s location={} \
             s. ! h264parse ! avdec_h264 ! checksumsink name=video \
             s. ! aacparse ! avdec_aac ! checksumsink name=audio",
            relay.url
        );
        let graph = parse_launch(&default_registry(), &line)
            .unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
        let counts = checksum_counts::count_checksums(graph, &SINKS, LIVE_FRAMES, DEADLINE);
        assert!(
            counts.iter().all(|&count| count >= LIVE_FRAMES),
            "decoded frames per pad {SINKS:?}: {counts:?}"
        );
    }
}
