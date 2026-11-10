// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! M2042: HTTP serving from the pipeline. `hlssink http-port` answers for the
//! playlist and the segments it has published, and `httpserversink` streams a
//! muxed MPEG-TS, fMP4 or live Matroska byte stream to clients that join
//! mid-stream at the next keyframe, after the fMP4 init segment or the Matroska
//! header.
#![cfg(feature = "http-server")]

#[path = "common/hls_mux_fixture.rs"]
mod hls_mux_fixture;
use hls_mux_fixture::{
    bytestream, frame, mux_fmp4, mux_live_mkv, mux_ts, synthetic_aus, NullSink, GOP,
};

use std::path::{Path, PathBuf};

use g2g_core::element::AsyncElement;
use g2g_core::frame::{FrameTiming, PipelinePacket};
use g2g_core::ByteStreamEncoding;
use g2g_plugins::hlssink::HlsSink;
use g2g_plugins::httpserver::HttpServerSink;
use g2g_plugins::matroska::{MatroskaDemuxer, MkvFrame};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

const LOOPBACK: &str = "127.0.0.1";
const HEAD_END: &[u8] = b"\r\n\r\n";
const MATROSKA_CLUSTER_ID: [u8; 4] = [0x1F, 0x43, 0xB6, 0x75];

fn work_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("g2g-m2042-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create work dir");
    dir
}

fn at(dir: &Path, name: &str) -> String {
    dir.join(name).to_string_lossy().into_owned()
}

async fn send_request(port: u16, method: &str, path: &str) -> TcpStream {
    let mut stream = TcpStream::connect((LOOPBACK, port)).await.expect("connect");
    stream
        .write_all(format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\n\r\n").as_bytes())
        .await
        .expect("send request");
    stream
}

fn split_response(response: &[u8]) -> (String, Vec<u8>) {
    let head_end = response
        .windows(HEAD_END.len())
        .position(|window| window == HEAD_END)
        .expect("response head");
    let head = String::from_utf8(response[..head_end].to_vec()).expect("ascii head");
    (head, response[head_end + HEAD_END.len()..].to_vec())
}

async fn request(port: u16, method: &str, path: &str) -> (String, Vec<u8>) {
    let mut stream = send_request(port, method, path).await;
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.expect("read");
    split_response(&response)
}

// reads up to the end of the head, so the server has handed the connection to the sink
async fn read_head(stream: &mut TcpStream) -> String {
    let mut head = Vec::new();
    while !head.ends_with(HEAD_END) {
        let mut byte = [0u8; 1];
        stream.read_exact(&mut byte).await.expect("read head");
        head.push(byte[0]);
    }
    String::from_utf8(head).expect("ascii head")
}

fn dechunk(mut body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let line_end = body
            .windows(2)
            .position(|window| window == b"\r\n")
            .expect("chunk size line");
        let size_text = std::str::from_utf8(&body[..line_end]).expect("ascii size");
        let size = usize::from_str_radix(size_text, 16).expect("hex chunk size");
        body = &body[line_end + 2..];
        if size == 0 {
            assert_eq!(body, b"\r\n", "the last chunk ends the body");
            return out;
        }
        out.extend_from_slice(&body[..size]);
        assert_eq!(&body[size..size + 2], b"\r\n", "chunk data ends in CRLF");
        body = &body[size + 2..];
    }
}

async fn push(element: &mut impl AsyncElement, frames: &[(Vec<u8>, FrameTiming)]) {
    for (bytes, timing) in frames {
        element
            .process(frame(bytes.clone(), *timing), &mut NullSink)
            .await
            .expect("push frame");
    }
}

#[tokio::test]
async fn hlssink_serves_what_it_publishes_and_nothing_else() {
    let dir = work_dir("hls");
    let mut sink = HlsSink::new(at(&dir, "seg%05d.ts"))
        .with_playlist_location(at(&dir, "live.m3u8"))
        .with_target_duration(1)
        .with_playlist_length(2)
        .with_max_files(2)
        .with_http_server(LOOPBACK, 0);
    sink.configure_pipeline(&bytestream(ByteStreamEncoding::MpegTs))
        .expect("configure");
    let port = sink.http_port().expect("bound port");
    std::fs::write(dir.join("unpublished.txt"), b"not served").expect("write stray file");

    let frames = mux_ts(&synthetic_aus(4)).await;
    push(&mut sink, &frames).await;
    sink.process(PipelinePacket::Eos, &mut NullSink)
        .await
        .expect("eos");

    let (head, playlist) = request(port, "GET", "/live.m3u8").await;
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    assert!(head.contains("Content-Type: application/vnd.apple.mpegurl"));
    assert_eq!(playlist, std::fs::read(dir.join("live.m3u8")).unwrap());

    let listed: Vec<String> = String::from_utf8(playlist)
        .unwrap()
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(String::from)
        .collect();
    assert_eq!(listed.len(), 2, "the rolling playlist lists two segments");
    for name in &listed {
        let (head, segment) = request(port, "GET", &format!("/{name}")).await;
        assert!(head.contains("Content-Type: video/mp2t"), "{head}");
        assert_eq!(segment, std::fs::read(dir.join(name)).unwrap());
    }

    let (pruned, _) = request(port, "GET", "/seg00000.ts").await;
    assert!(
        pruned.starts_with("HTTP/1.1 404"),
        "a pruned segment is gone: {pruned}"
    );
    let (stray, _) = request(port, "GET", "/unpublished.txt").await;
    assert!(
        stray.starts_with("HTTP/1.1 404"),
        "only published files: {stray}"
    );
    let (escape, _) = request(port, "GET", "/../live.m3u8").await;
    assert!(
        escape.starts_with("HTTP/1.1 200"),
        "names resolve by last segment only"
    );
    let (post, _) = request(port, "POST", "/live.m3u8").await;
    assert!(post.starts_with("HTTP/1.1 405"), "{post}");
}

// the client connects after the first GOP has gone out, so it starts at the second
async fn late_client_stream(
    encoding: ByteStreamEncoding,
    frames: &[(Vec<u8>, FrameTiming)],
    joined_before: usize,
) -> (String, Vec<u8>) {
    let mut sink = HttpServerSink::new(LOOPBACK, 0);
    sink.configure_pipeline(&bytestream(encoding))
        .expect("configure");
    let port = sink.current_port().expect("bound port");
    push(&mut sink, &frames[..joined_before]).await;

    let mut client = send_request(port, "GET", "/").await;
    let head = read_head(&mut client).await;
    push(&mut sink, &frames[joined_before..]).await;
    sink.process(PipelinePacket::Eos, &mut NullSink)
        .await
        .expect("eos");
    drop(sink);
    let mut body = Vec::new();
    client.read_to_end(&mut body).await.expect("read body");
    (head, dechunk(&body))
}

#[tokio::test]
async fn httpserversink_starts_a_late_ts_client_at_the_next_keyframe() {
    let frames = mux_ts(&synthetic_aus(3)).await;
    let keyframes: Vec<usize> = (0..frames.len())
        .filter(|&index| frames[index].1.keyframe)
        .collect();
    assert_eq!(keyframes.len(), 3, "one keyframe frame per GOP");
    let joined_before = keyframes[0] + 1;

    let (head, body) = late_client_stream(ByteStreamEncoding::MpegTs, &frames, joined_before).await;
    assert!(head.contains("Content-Type: video/mp2t"), "{head}");
    assert!(head.contains("Transfer-Encoding: chunked"), "{head}");
    let expected: Vec<u8> = frames[keyframes[1]..]
        .iter()
        .flat_map(|(bytes, _)| bytes.clone())
        .collect();
    assert_eq!(body, expected, "the stream starts at the second keyframe");
}

#[tokio::test]
async fn httpserversink_sends_a_late_fmp4_client_the_init_segment_first() {
    let frames = mux_fmp4(&synthetic_aus(3), 1000).await;
    assert_eq!(
        &frames[0].0[4..8],
        b"ftyp",
        "the stream opens on the init segment"
    );
    assert!(frames.len() >= 3, "fragments follow the init segment");

    let joined_before = 2;
    let (head, body) =
        late_client_stream(ByteStreamEncoding::IsoBmff, &frames, joined_before).await;
    assert!(head.contains("Content-Type: video/mp4"), "{head}");
    let rest: Vec<u8> = frames[joined_before..]
        .iter()
        .flat_map(|(bytes, _)| bytes.clone())
        .collect();
    assert!(
        body.ends_with(&rest),
        "the stream continues from the join point"
    );
    let init = &body[..body.len() - rest.len()];
    assert!(
        frames[0].0.starts_with(init),
        "the prefix is the stream's own init"
    );
    let has = |kind: &[u8]| init.windows(4).any(|window| window == kind);
    assert!(
        has(b"moov") && !has(b"moof"),
        "the prefix is the init segment alone"
    );
}

fn demux(stream: &[u8]) -> Vec<MkvFrame> {
    let mut demuxer = MatroskaDemuxer::new();
    demuxer.push_data(stream);
    demuxer.finish();
    demuxer.take_frames()
}

// a client joining halfway through the first GOP starts at the second keyframe
async fn assert_late_matroska_client_starts_at_the_second_keyframe(
    aus: &[(Vec<u8>, FrameTiming)],
    second_keyframe_opens_a_cluster: bool,
) {
    let frames = mux_live_mkv(aus).await;
    assert_eq!(frames.len(), aus.len(), "one muxed frame per access unit");
    assert_eq!(
        frames[GOP].0.starts_with(&MATROSKA_CLUSTER_ID),
        second_keyframe_opens_a_cluster,
        "the fixture produces the join shape under test"
    );
    let whole: Vec<u8> = frames.iter().flat_map(|(bytes, _)| bytes.clone()).collect();
    let expected = demux(&whole).split_off(GOP);

    let (head, body) = late_client_stream(ByteStreamEncoding::Matroska, &frames, GOP / 2).await;
    assert!(head.contains("Content-Type: video/x-matroska"), "{head}");
    let received = demux(&body);
    assert!(received[0].keyframe, "the client starts on a keyframe");
    assert_eq!(
        received, expected,
        "the late client decodes the same frames at the same times"
    );
}

#[tokio::test]
async fn httpserversink_opens_a_cluster_for_a_late_matroska_client_joining_mid_cluster() {
    assert_late_matroska_client_starts_at_the_second_keyframe(&synthetic_aus(3), false).await;
}

#[tokio::test]
async fn httpserversink_starts_a_late_matroska_client_at_a_cluster_that_opens_on_a_keyframe() {
    // at 41 ms a frame each keyframe is the first past the muxer's 1 s Cluster span
    let frame_period_ns = 41_000_000;
    let aus: Vec<(Vec<u8>, FrameTiming)> = synthetic_aus(3)
        .into_iter()
        .enumerate()
        .map(|(index, (bytes, timing))| {
            let pts_ns = index as u64 * frame_period_ns;
            let timing = FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                duration_ns: frame_period_ns,
                ..timing
            };
            (bytes, timing)
        })
        .collect();
    assert_late_matroska_client_starts_at_the_second_keyframe(&aus, true).await;
}
