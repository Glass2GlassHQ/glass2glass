// `decodebin current-video=N` / `current-audio=N` decode the Nth stream of that kind in
// container order, the stream `ffmpeg -map 0:v:N` / `-map 0:a:N` reads.
#![cfg(all(feature = "std", feature = "ffmpeg", target_os = "linux"))]

#[path = "common/multitrack_fixture.rs"]
mod multitrack_fixture;

use g2g_core::runtime::{parse_launch, ParseError};
use g2g_core::stream::StreamType;

use multitrack_fixture::{
    assert_decoded_stream, multitrack_fixture, probed_of_kind, registry, run_recording,
    RECORDING_SINK,
};

const CONTAINERS: [&str; 4] = ["mp4", "mkv", "ts", "mpg"];

fn launch_line(container: &str, kind: StreamType, index: usize, label: &str) -> String {
    format!(
        "filesrc location={} ! decodebin current-{}={index} ! {RECORDING_SINK} label={label}",
        multitrack_fixture(container).display(),
        kind.name()
    )
}

async fn each_index_decodes_its_own_track(kind: StreamType) {
    for container in CONTAINERS {
        let probed = probed_of_kind(container, kind);
        assert!(
            probed.len() > 1,
            "{container} fixture has several {} tracks",
            kind.name()
        );
        for (index, stream) in probed.iter().enumerate() {
            let label = format!("{container}-{}-{index}", kind.name());
            let caps = run_recording(&launch_line(container, kind, index, &label), &[&label]).await;
            assert_decoded_stream(
                &caps[0],
                stream,
                &format!("{container} current-{}={index}", kind.name()),
            );
        }
    }
}

#[tokio::test]
async fn each_video_index_decodes_its_own_track() {
    each_index_decodes_its_own_track(StreamType::Video).await;
}

#[tokio::test]
async fn each_audio_index_decodes_its_own_track() {
    each_index_decodes_its_own_track(StreamType::Audio).await;
}

#[test]
fn an_index_past_the_last_stream_fails_at_parse() {
    for container in CONTAINERS {
        for kind in [StreamType::Video, StreamType::Audio] {
            let count = probed_of_kind(container, kind).len();
            let line = launch_line(container, kind, count, "out-of-range");
            let error = parse_launch(&registry(), &line)
                .err()
                .unwrap_or_else(|| panic!("`{line}` names a stream past the last one"));
            assert!(
                matches!(
                    error,
                    ParseError::StreamIndexOutOfRange { kind: k, index, count: c, .. }
                        if k == kind && index == count && c == count
                ),
                "`{line}`: {error}"
            );
        }
    }
}
