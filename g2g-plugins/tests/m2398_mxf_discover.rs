// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "std")]

use g2g_core::runtime::block_on;
use g2g_core::stream::StreamType;
use g2g_plugins::discover::discover;
use g2g_plugins::registry::default_registry;

#[path = "common/g2g_launch.rs"]
mod g2g_launch;

use g2g_launch::written_bytes;

// every fixture is ten frames at 25 fps, see tools/mxf-fixtures.sh
const FRAMES: u64 = 10;
const FRAME_PERIOD_NS: u64 = 40_000_000;
const PICTURE_AND_SOUND_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/mxf_mpeg2_pcm.mxf"
);
// ffprobe names the fixture's sound pcm_s24le
const SOUND_STREAM: &str = "S24LE";

#[test]
fn discover_reads_every_track_and_the_duration_of_an_mxf_file() {
    let registry = default_registry();
    let discovery = block_on(discover(&registry, PICTURE_AND_SOUND_FIXTURE))
        .expect("discover opens an mxf file");
    assert_eq!(discovery.duration_ns, Some(FRAMES * FRAME_PERIOD_NS));
    let stream_types: Vec<StreamType> = discovery
        .streams
        .iter()
        .map(|stream| stream.stream_type)
        .collect();
    assert_eq!(stream_types, [StreamType::Video, StreamType::Audio]);
}

#[test]
fn mxfdemux_writes_the_selected_track_as_ffmpeg_copies_it() {
    let reference = |stream: &str| {
        let path = format!("{PICTURE_AND_SOUND_FIXTURE}.{stream}.data");
        std::fs::read(&path).unwrap_or_else(|error| panic!("cannot read {path}: {error}"))
    };
    let demuxed = |stream: &str, file_name: &str| {
        written_bytes(
            &format!("filesrc location={PICTURE_AND_SOUND_FIXTURE} ! mxfdemux stream={stream}"),
            file_name,
        )
    };
    assert_eq!(demuxed("mpeg2", "m2398_picture.data"), reference("v"));
    assert_eq!(demuxed(SOUND_STREAM, "m2398_sound.data"), reference("a"));
}

#[cfg(feature = "ffmpeg")]
const PICTURE_FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/mxf_h264.mxf");

// the digest of each line of a `checksumsink` run, in order
#[cfg(feature = "ffmpeg")]
fn digests(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .map(|line| {
            let (_, digest) = line
                .rsplit_once(' ')
                .expect("a checksum line is `<pts> <digest>`");
            digest.to_string()
        })
        .collect()
}

#[cfg(feature = "ffmpeg")]
#[test]
fn decodebin_decodes_every_frame_of_an_mxf_file() {
    use g2g_launch::{checksum_lines, launch};

    let pipeline = format!("filesrc location={PICTURE_FIXTURE} ! decodebin ! checksumsink");
    let lines = checksum_lines(&launch(&[], &pipeline));
    assert_eq!(lines.len(), FRAMES as usize, "{pipeline}");
}

#[cfg(feature = "ffmpeg")]
#[test]
fn two_mxf_segments_with_no_length_play_back_to_back() {
    use g2g_launch::{checksum_lines, launch};
    use g2g_plugins::checksumsink::checksum_line;

    let pipeline =
        format!("edlsrc uri=\"edl://{PICTURE_FIXTURE};{PICTURE_FIXTURE}\" ! checksumsink hash=md5");
    let lines = checksum_lines(&launch(&[], &pipeline));
    let file_digests = digests(&lines[..FRAMES as usize]);
    let expected: Vec<String> = file_digests
        .iter()
        .chain(&file_digests)
        .enumerate()
        .map(|(frame, digest)| checksum_line(frame as u64 * FRAME_PERIOD_NS, digest.clone()))
        .collect();
    assert_eq!(lines, expected, "{pipeline}");
}
