// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "http-src")]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::{Command, Output};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

// ffmpeg's matroska muxer: 25 fps video encoded with -g 10, a SeekHead at the start locating the Cues at the end
const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/av_seek_h264_aac.mkv"
);
const VIDEO_FRAME_COUNT: usize = 50;
const FRAME_PERIOD_NS: u64 = 40_000_000;
const FRAMES_PER_KEYFRAME: usize = 10;
// between the keyframes at 0.8 s and 1.2 s
const TARGET_NS: u64 = 1_020_000_000;
const NANOS_PER_SECOND: f64 = 1_000_000_000.0;
// the body goes out in pieces this size, so the run reaches the seek position before the body ends
const PACED_WRITE_LEN: usize = 256;
const PACED_WRITE_INTERVAL: Duration = Duration::from_millis(5);
// back from the frame at 0.6 s to between the keyframes at 0.4 s and 0.8 s, so the output leaves the unseeked run
const BACKWARD_SEEK_POSITION_NS: u64 = 600_000_000;
const BACKWARD_SEEK_TARGET_NS: u64 = 420_000_000;

#[derive(Debug, Clone, Copy)]
enum RangeAnswer {
    PartialContent,
    // the whole resource with 200, as a server without range support answers
    WholeResource,
}

// the first byte each request asked for, 0 for a request without a Range
type RequestedStarts = Arc<Mutex<Vec<u64>>>;

fn range_start(request: &str) -> Option<u64> {
    let line = request
        .lines()
        .find(|line| line.to_ascii_lowercase().starts_with("range:"))?;
    let (_, spec) = line.split_once("bytes=")?;
    spec.split('-').next()?.trim().parse().ok()
}

// one connection per request for the one resource the URL names, writing the body `paced_write_len` bytes at a time
fn serve(
    resource: Vec<u8>,
    answer: RangeAnswer,
    paced_write_len: Option<usize>,
) -> (String, RequestedStarts) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let starts = RequestedStarts::default();
    let logged = starts.clone();
    thread::spawn(move || {
        for connection in listener.incoming() {
            let Ok(mut stream) = connection else {
                break;
            };
            let mut request = Vec::new();
            let mut byte = [0u8; 1];
            while stream.read(&mut byte).unwrap_or(0) == 1 {
                request.push(byte[0]);
                if request.ends_with(b"\r\n\r\n") {
                    break;
                }
            }
            let start = range_start(&String::from_utf8_lossy(&request));
            logged.lock().unwrap().push(start.unwrap_or(0));
            let total = resource.len();
            let (status, content_range, body) = match (start, answer) {
                (Some(start), RangeAnswer::PartialContent) => {
                    let start = (start as usize).min(total);
                    (
                        "206 Partial Content",
                        format!("Content-Range: bytes {start}-{}/{total}\r\n", total - 1),
                        &resource[start..],
                    )
                }
                _ => ("200 OK", String::new(), &resource[..]),
            };
            let header = format!(
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{content_range}Connection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let Some(paced_write_len) = paced_write_len else {
                let _ = stream.write_all(body);
                continue;
            };
            for piece in body.chunks(paced_write_len) {
                if stream.write_all(piece).is_err() {
                    break;
                }
                thread::sleep(PACED_WRITE_INTERVAL);
            }
        }
    });
    (format!("http://127.0.0.1:{port}/av.mkv"), starts)
}

const MKV_DEMUX: &str =
    "bytestream-format=matroska ! matroskademux name=d d.video_0 ! checksumsink d.audio_0 ! fakesink";

fn launch(url: &str, demux: &str, options: &[&str]) -> Output {
    let pipeline = format!("httpsrc location={url} {demux}");
    Command::new(env!("CARGO_BIN_EXE_g2g-launch"))
        .arg("-q")
        .args(options)
        .args(pipeline.split_whitespace())
        .output()
        .expect("g2g-launch starts")
}

fn checksum_lines(output: &Output) -> Vec<String> {
    assert!(
        output.status.success(),
        "g2g-launch fails: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::to_string)
        .collect()
}

fn seek_at(position_ns: u64, target_ns: u64) -> String {
    format!(
        "{}:{}",
        position_ns as f64 / NANOS_PER_SECOND,
        target_ns as f64 / NANOS_PER_SECOND
    )
}

// the seeked run leaves the unseeked one at the seek, then plays from the keyframe before the target to the end
fn assert_resumes_at_the_keyframe(seeked: &[String], unseeked: &[String], target_ns: u64) {
    let resumed = seeked
        .iter()
        .zip(unseeked)
        .position(|(seeked, unseeked)| seeked != unseeked)
        .unwrap_or(unseeked.len());
    let keyframe =
        (target_ns / FRAME_PERIOD_NS) as usize / FRAMES_PER_KEYFRAME * FRAMES_PER_KEYFRAME;
    let expected = &unseeked[keyframe..];
    let pts = |lines: &[String]| {
        lines
            .iter()
            .map(|line| line.split_whitespace().next().map(str::to_string))
            .collect::<Vec<_>>()
    };
    assert_eq!(pts(&seeked[resumed..]), pts(expected));
    // the first frame after the flush carries the codec config, so its digest differs
    assert_eq!(seeked[resumed + 1..], expected[1..]);
}

fn fixture() -> Vec<u8> {
    std::fs::read(FIXTURE).expect("the fixture reads")
}

#[test]
fn a_seek_during_playback_fetches_the_rest_with_a_range_request() {
    let (url, starts) = serve(
        fixture(),
        RangeAnswer::PartialContent,
        Some(PACED_WRITE_LEN),
    );
    let unseeked = checksum_lines(&launch(&url, MKV_DEMUX, &[]));
    assert_eq!(unseeked.len(), VIDEO_FRAME_COUNT);
    starts.lock().unwrap().clear();
    let seek = seek_at(BACKWARD_SEEK_POSITION_NS, BACKWARD_SEEK_TARGET_NS);
    let seeked = checksum_lines(&launch(&url, MKV_DEMUX, &["--seek-at", &seek]));
    assert!(
        starts.lock().unwrap().iter().any(|&start| start > 0),
        "the seek reads on from a byte past the start, requests {:?}",
        starts.lock().unwrap()
    );
    assert_resumes_at_the_keyframe(&seeked, &unseeked, BACKWARD_SEEK_TARGET_NS);
}

#[test]
fn a_seek_after_the_end_fetches_from_the_parked_source() {
    let (url, starts) = serve(fixture(), RangeAnswer::PartialContent, None);
    let unseeked = checksum_lines(&launch(&url, MKV_DEMUX, &[]));
    starts.lock().unwrap().clear();
    let last_frame_ns = (VIDEO_FRAME_COUNT - 1) as u64 * FRAME_PERIOD_NS;
    let seek = seek_at(last_frame_ns, TARGET_NS);
    let seeked = checksum_lines(&launch(&url, MKV_DEMUX, &["--seek-at", &seek]));
    assert_eq!(
        seeked[..unseeked.len()],
        unseeked,
        "the whole run comes before the seek"
    );
    assert!(starts.lock().unwrap().iter().any(|&start| start > 0));
    assert_resumes_at_the_keyframe(&seeked, &unseeked, TARGET_NS);
}

#[test]
fn a_server_that_ignores_the_range_fails_the_seek() {
    let (url, _) = serve(fixture(), RangeAnswer::WholeResource, None);
    let last_frame_ns = (VIDEO_FRAME_COUNT - 1) as u64 * FRAME_PERIOD_NS;
    let output = launch(
        &url,
        MKV_DEMUX,
        &["--seek-at", &seek_at(last_frame_ns, TARGET_NS)],
    );
    assert!(
        !output.status.success(),
        "the unserved seek went unreported"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("cannot seek"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ffmpeg's default mp4 layout, the moov after the mdat, the same 2 s of video
const MP4_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/av_seek_h264_aac.mp4"
);
const MP4_DEMUX: &str = "bytestream-format=mp4 ! qtdemux ! checksumsink";

#[test]
fn a_one_port_qtdemux_reads_a_moov_last_mp4_with_range_requests_and_seeks() {
    let mp4 = std::fs::read(MP4_FIXTURE).expect("the fixture reads");
    let (url, starts) = serve(mp4, RangeAnswer::PartialContent, None);
    let unseeked = checksum_lines(&launch(&url, MP4_DEMUX, &[]));
    assert_eq!(unseeked.len(), VIDEO_FRAME_COUNT);
    starts.lock().unwrap().clear();
    let last_frame_ns = (VIDEO_FRAME_COUNT - 1) as u64 * FRAME_PERIOD_NS;
    let seeked = checksum_lines(&launch(
        &url,
        MP4_DEMUX,
        &["--seek-at", &seek_at(last_frame_ns, TARGET_NS)],
    ));
    assert_eq!(seeked[..unseeked.len()], unseeked);
    assert!(
        starts
            .lock()
            .unwrap()
            .iter()
            .filter(|&&start| start > 0)
            .count()
            > 1,
        "the moov and the seek each read from past the start, requests {:?}",
        starts.lock().unwrap()
    );
    assert_resumes_at_the_keyframe(&seeked, &unseeked, TARGET_NS);
}
