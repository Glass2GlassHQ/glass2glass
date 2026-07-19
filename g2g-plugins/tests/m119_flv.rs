//! M119 FLV demux, end to end through `parse_launch`: `filesrc` reads a synthetic
//! FLV file (its container auto-sniffed via `typefind`), feeds `flvdemux`, and the
//! selected elementary stream's access units reach the sink. Exercises the new
//! `ByteStream{Flv}` caps, the FLV sniff, and the demuxer's stream selection.
//!
//! `default_registry` / `filesrc` are `std`-gated, so this file is too.
#![cfg(feature = "std")]

use std::fs;
use std::path::PathBuf;

use g2g_core::runtime::{parse_launch, run_graph, ParseError};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn push_u24(out: &mut Vec<u8>, v: u32) {
    out.push((v >> 16) as u8);
    out.push((v >> 8) as u8);
    out.push(v as u8);
}

/// One FLV tag (type, ms timestamp, body), without its leading `PreviousTagSize`.
fn tag(tag_type: u8, timestamp: u32, body: &[u8]) -> Vec<u8> {
    let mut t = vec![tag_type];
    push_u24(&mut t, body.len() as u32);
    push_u24(&mut t, timestamp & 0x00FF_FFFF);
    t.push((timestamp >> 24) as u8);
    push_u24(&mut t, 0); // stream id
    t.extend_from_slice(body);
    t
}

/// A video tag body: one AVCC access unit (`avc_packet_type` 1).
fn avc_nalu(au: &[u8]) -> Vec<u8> {
    let mut b = vec![0x17u8, 0x01, 0x00, 0x00, 0x00];
    b.extend_from_slice(au);
    b
}

/// An audio tag body: one raw AAC frame (`aac_packet_type` 1).
fn aac_raw(frame: &[u8]) -> Vec<u8> {
    let mut b = vec![0xAFu8, 0x01];
    b.extend_from_slice(frame);
    b
}

/// An FLV stream: header, then one audio + two video access units interleaved.
/// Video payloads are valid AVCC (4-byte length prefix per NAL).
fn synthetic_flv() -> Vec<u8> {
    let tags = [
        tag(9, 0, &avc_nalu(&[0, 0, 0, 3, 0x65, 0x11, 0x22])),
        tag(8, 0, &aac_raw(&[0x33, 0x44])),
        tag(9, 33, &avc_nalu(&[0, 0, 0, 2, 0x41, 0x55])),
    ];
    let mut s = b"FLV".to_vec();
    s.push(1); // version
    s.push(0x05); // flags: audio + video present
    s.extend_from_slice(&9u32.to_be_bytes()); // data offset
    let mut prev = 0u32;
    for t in &tags {
        s.extend_from_slice(&prev.to_be_bytes());
        s.extend_from_slice(t);
        prev = t.len() as u32;
    }
    s
}

fn write_temp(name: &str, bytes: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(name);
    fs::write(&path, bytes).expect("write temp fixture");
    path
}

async fn run_pipeline(text: &str) -> u64 {
    let reg = default_registry();
    let graph = parse_launch(&reg, text).expect("pipeline parses");
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("pipeline runs")
        .frames_consumed
}

#[tokio::test]
async fn filesrc_auto_sniffs_flv_and_demuxes_video() {
    let path = write_temp("g2g_m119_auto.flv", &synthetic_flv());
    let text = format!(
        "filesrc location={} bytestream-format=auto ! flvdemux ! fakesink",
        path.display()
    );
    assert_eq!(
        run_pipeline(&text).await,
        2,
        "two H.264 access units demuxed to the sink"
    );
    let _ = fs::remove_file(&path);
}

#[tokio::test]
async fn filesrc_explicit_flv_selects_audio() {
    let path = write_temp("g2g_m119_audio.flv", &synthetic_flv());
    let text = format!(
        "filesrc location={} bytestream-format=flv ! flvdemux stream=aac ! fakesink",
        path.display()
    );
    assert_eq!(
        run_pipeline(&text).await,
        1,
        "one AAC frame demuxed to the sink"
    );
    let _ = fs::remove_file(&path);
}

#[tokio::test]
async fn unknown_stream_name_is_rejected() {
    // An unsupported stream selection is rejected at parse time, before the file
    // is opened.
    let reg = default_registry();
    let err = parse_launch(
        &reg,
        "filesrc location=x.flv bytestream-format=flv ! flvdemux stream=vp9 ! fakesink",
    )
    .unwrap_err();
    assert!(
        matches!(err, ParseError::BadValue { ref key, .. } if key == "stream"),
        "got {err:?}"
    );
}
