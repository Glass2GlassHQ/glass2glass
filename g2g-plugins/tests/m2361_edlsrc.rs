// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "ffmpeg")]

use g2g_core::log::ClockTime;
use g2g_core::time::ns_to_ticks;
use g2g_plugins::checksumsink::checksum_line;

#[path = "common/g2g_launch.rs"]
mod g2g_launch;

use g2g_launch::{checksum_lines, launch};

// ffmpeg -f lavfi -i testsrc=size=64x48:rate=25:duration=3 -c:v libx264 -g 10 -keyint_min 10 -sc_threshold 0 -bf 0
const MP4_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h264_64x48_25fps_gop10.mp4"
);
// ffprobe start_time is 0.4, so its stream time runs 0.4 s behind its timestamps
const OFFSET_MKV_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h264_64x48_25fps_gop10_bframes_offset.mkv"
);
const FRAME_PERIOD_NS: u64 = 40_000_000;
const NANOS_PER_SECOND: f64 = 1_000_000_000.0;

#[derive(Debug, Clone, Copy)]
struct Cut {
    file: &'static str,
    start_ns: u64,
    length_ns: Option<u64>,
}

const CUTS: [Cut; 4] = [
    Cut {
        file: MP4_FIXTURE,
        start_ns: 1_000_000_000,
        length_ns: Some(400_000_000),
    },
    Cut {
        file: OFFSET_MKV_FIXTURE,
        start_ns: 200_000_000,
        length_ns: Some(600_000_000),
    },
    // plays to the end of the file
    Cut {
        file: OFFSET_MKV_FIXTURE,
        start_ns: 2_600_000_000,
        length_ns: None,
    },
    Cut {
        file: MP4_FIXTURE,
        start_ns: 0,
        length_ns: Some(200_000_000),
    },
];

fn seconds(nanoseconds: u64) -> String {
    format!("{}", nanoseconds as f64 / NANOS_PER_SECOND)
}

fn edl_segments(cuts: &[Cut]) -> String {
    let segments: Vec<String> = cuts
        .iter()
        .map(|cut| match cut.length_ns {
            Some(length_ns) => format!(
                "{},{},{}",
                cut.file,
                seconds(cut.start_ns),
                seconds(length_ns)
            ),
            None => format!("{},{}", cut.file, seconds(cut.start_ns)),
        })
        .collect();
    segments.join(";")
}

fn edl_pipeline() -> String {
    format!(
        "edlsrc uri=\"edl://{}\" ! checksumsink hash=md5",
        edl_segments(&CUTS)
    )
}

// the digests of every frame of `file` decoded to NV12, in presentation order
fn decoded_digests(file: &str) -> Vec<String> {
    let lines = checksum_lines(&launch(
        &[],
        &format!("filesrc location={file} ! decodebin ! videoconvert ! video/x-raw,format=NV12 ! checksumsink hash=md5"),
    ));
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

// the frames of every cut in order, stamped with their place on the timeline
fn timeline_lines() -> Vec<String> {
    let mut lines = Vec::new();
    for cut in CUTS {
        let digests = decoded_digests(cut.file);
        let first = (cut.start_ns / FRAME_PERIOD_NS) as usize;
        let end = cut.length_ns.map_or(digests.len(), |length_ns| {
            first + (length_ns / FRAME_PERIOD_NS) as usize
        });
        for digest in &digests[first..end] {
            let timeline_pts_ns = lines.len() as u64 * FRAME_PERIOD_NS;
            lines.push(checksum_line(timeline_pts_ns, digest.clone()));
        }
    }
    lines
}

#[test]
fn plays_each_cut_back_to_back_on_one_timeline() {
    let expected = timeline_lines();
    assert_eq!(
        checksum_lines(&launch(&[], &edl_pipeline())),
        expected,
        "{}",
        edl_pipeline()
    );
}

#[test]
fn a_seek_into_the_second_cut_resumes_at_its_source_frame() {
    let expected = timeline_lines();
    let target_ns = CUTS[0].length_ns.expect("the first cut has a length") + 2 * FRAME_PERIOD_NS;
    let seeked = checksum_lines(&launch(&["--seek", &seconds(target_ns)], &edl_pipeline()));
    let first = (target_ns / FRAME_PERIOD_NS) as usize;
    assert_eq!(
        seeked,
        expected[first..],
        "seek to {}",
        ClockTime(target_ns)
    );
}

// ffprobe: pcm_s16le, 8000 Hz, 2 channels, 1 s
const WAV_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/sine_8k_stereo_s16le.wav"
);
const WAV_SAMPLE_RATE: u32 = 8_000;
const WAV_SAMPLE_FRAME_BYTES: usize = 4;

const AUDIO_CUTS: [Cut; 3] = [
    Cut {
        file: WAV_FIXTURE,
        start_ns: 250_000_000,
        length_ns: Some(300_000_000),
    },
    Cut {
        file: WAV_FIXTURE,
        start_ns: 0,
        length_ns: Some(125_000_000),
    },
    // plays to the end of the file
    Cut {
        file: WAV_FIXTURE,
        start_ns: 750_000_000,
        length_ns: None,
    },
];

fn written_bytes(pipeline: &str, file_name: &str) -> Vec<u8> {
    let path = format!("{}/{file_name}", env!("CARGO_TARGET_TMPDIR"));
    let output = launch(&[], &format!("{pipeline} ! filesink location={path}"));
    assert!(
        output.status.success(),
        "{pipeline} fails: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read(&path).unwrap_or_else(|error| panic!("cannot read {path}: {error}"))
}

#[test]
fn the_audio_stream_after_new_stream_plays_each_cut_back_to_back() {
    let pcm = written_bytes(
        &format!("filesrc location={WAV_FIXTURE} ! wavparse"),
        "m2361_wav.pcm",
    );
    let byte_offset = |ns: u64| ns_to_ticks(ns, WAV_SAMPLE_RATE) as usize * WAV_SAMPLE_FRAME_BYTES;
    let expected: Vec<u8> = AUDIO_CUTS
        .iter()
        .flat_map(|cut| {
            let first = byte_offset(cut.start_ns);
            let end = cut
                .length_ns
                .map_or(pcm.len(), |length_ns| byte_offset(cut.start_ns + length_ns));
            pcm[first..end].iter().copied()
        })
        .collect();
    let uri = format!(
        "edl://{};!new_stream;!no_clip;{}",
        edl_segments(&CUTS),
        edl_segments(&AUDIO_CUTS)
    );
    let played = written_bytes(
        &format!("edlsrc uri=\"{uri}\" stream=1 media=audio"),
        "m2361_edl_audio.pcm",
    );
    assert_eq!(played.len(), expected.len(), "{uri}");
    assert!(played == expected, "{uri}: the samples differ");
}
