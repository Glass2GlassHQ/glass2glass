// A fan-out stream no branch links is dropped and the linked ones play, as gst leaves the
// extra pad unlinked: one branch on a two-stream URI decodes its stream alone.
#![cfg(all(feature = "std", feature = "ffmpeg", target_os = "linux"))]

use std::time::Duration;

use g2g_core::runtime::parse_launch;
use g2g_plugins::registry::default_registry;

#[path = "common/checksum_counts.rs"]
mod checksum_counts;
use checksum_counts::count_checksums;
#[cfg(feature = "rtsp")]
#[path = "common/ffmpeg_tool.rs"]
mod ffmpeg_tool;
#[cfg(feature = "rtsp")]
#[path = "common/rtsp_av_relay.rs"]
mod rtsp_av_relay;

const AV_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/av_h264_aac.mp4"
);
const SINK: &str = "branch";
const DEADLINE: Duration = Duration::from_secs(20);
// the fixture is a fraction of a second long, so a file run checks that the branch decodes at all
const FILE_FRAMES: usize = 1;
// a live run checks that the linked track keeps flowing
#[cfg(feature = "rtsp")]
const LIVE_FRAMES: usize = 10;

fn assert_the_branch_decodes(line: &str, enough: usize) {
    let graph =
        parse_launch(&default_registry(), line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let counts = count_checksums(graph, &[SINK], enough, DEADLINE);
    assert!(counts[0] >= enough, "`{line}` decoded {} frames", counts[0]);
}

#[test]
fn uridecodebin_plays_the_one_linked_stream() {
    assert_the_branch_decodes(
        &format!(
            "uridecodebin uri=file://{AV_FIXTURE} name=d d. ! videoconvert ! checksumsink name={SINK}"
        ),
        FILE_FRAMES,
    );
}

#[test]
fn decodebin_plays_the_one_linked_stream() {
    assert_the_branch_decodes(
        &format!(
            "filesrc location={AV_FIXTURE} ! decodebin name=d d.audio_0 ! audioconvert ! checksumsink name={SINK}"
        ),
        FILE_FRAMES,
    );
}

#[cfg(feature = "rtsp")]
mod live {
    use super::*;

    // the hook DESCRIBEs the stream at parse time
    #[test]
    fn an_rtsp_uri_plays_the_one_linked_stream() {
        if !ffmpeg_tool::have_ffmpeg() {
            eprintln!("ffmpeg not present; skipping the RTSP fan-out with one branch");
            return;
        }
        let relay = rtsp_av_relay::serve_av_fixture(AV_FIXTURE);
        assert_the_branch_decodes(
            &format!(
                "uridecodebin uri={} name=d d. ! videoconvert ! checksumsink name={SINK}",
                relay.url
            ),
            LIVE_FRAMES,
        );
    }
}
