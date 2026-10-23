// A demux or decodebin fan-out over two tracks of one codec carries each track on its own
// port: `d.audio_1` is the second AAC track, not a dark port behind the first one.
#![cfg(all(feature = "std", feature = "ffmpeg", target_os = "linux"))]

#[path = "common/multitrack_fixture.rs"]
mod multitrack_fixture;

use g2g_core::stream::StreamType;

use multitrack_fixture::{
    assert_decoded_stream, multitrack_fixture, probed_of_kind, run_recording, RECORDING_SINK,
};

const DEMUXERS: [(&str, &str); 4] = [
    ("mkv", "matroskademux"),
    ("ts", "tsdemux"),
    ("mp4", "qtdemux"),
    ("mpg", "mpegpsdemux"),
];

fn decoder(container: &str, kind: StreamType) -> &'static str {
    match (container, kind) {
        ("mpg", StreamType::Video) => "mpegvideoparse ! avdec_mpeg2video !",
        ("mpg", _) => "mpegaudioparse ! mpg123audiodec !",
        (_, StreamType::Video) => "h264parse ! avdec_h264 !",
        _ => "aacparse ! avdec_aac !",
    }
}

// `branch_head` sits between each port and its sink: a parser and decoder after a demuxer, nothing after a decodebin
async fn each_port_carries_its_own_track(
    container: &str,
    fanout: &str,
    kind: StreamType,
    branch_head: &str,
) {
    let probed = probed_of_kind(container, kind);
    assert!(
        probed.len() > 1,
        "{container} fixture has several {} tracks",
        kind.name()
    );
    let labels: Vec<String> = (0..probed.len())
        .map(|index| format!("{container}-{fanout}-{}-{index}", kind.name()))
        .collect();
    let branches: String = labels
        .iter()
        .enumerate()
        .map(|(index, label)| {
            format!(
                "  d.{}_{index} ! {branch_head} {RECORDING_SINK} label={label}",
                kind.name()
            )
        })
        .collect();
    let line = format!(
        "filesrc location={} ! {fanout} name=d{branches}",
        multitrack_fixture(container).display()
    );
    let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
    let decoded = run_recording(&line, &label_refs).await;
    for (index, (caps, stream)) in decoded.iter().zip(&probed).enumerate() {
        assert_decoded_stream(caps, stream, &format!("`{line}` port {index}"));
    }
}

#[tokio::test]
async fn a_demux_fanout_carries_each_same_codec_track() {
    for (container, demuxer) in DEMUXERS {
        for kind in [StreamType::Video, StreamType::Audio] {
            each_port_carries_its_own_track(container, demuxer, kind, decoder(container, kind))
                .await;
        }
    }
}

#[tokio::test]
async fn a_decodebin_fanout_carries_each_same_codec_track() {
    for (container, _) in DEMUXERS {
        for kind in [StreamType::Video, StreamType::Audio] {
            each_port_carries_its_own_track(container, "decodebin", kind, "").await;
        }
    }
}
