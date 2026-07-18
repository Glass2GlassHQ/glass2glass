//! M112: `filesrc` declares a container via its `bytestream-format` property, so
//! a `gst-launch` text pipeline can feed a demuxer from a file. Covers the
//! explicit `mpegts` form and the `auto` form (sniff the header), both built by
//! `parse_launch` from `default_registry` and run end to end through `run_graph`.
//!
//! `default_registry` / `filesrc` are `std`-gated, so this file is too.
#![cfg(feature = "std")]

use std::fs;
use std::path::PathBuf;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

// --- synthetic container builders (mirror the demuxer tests) ---

fn ts_packet(pid: u16, pusi: bool, payload: &[u8]) -> Vec<u8> {
    const LEN: usize = 188;
    const ROOM: usize = LEN - 4;
    let mut p = vec![0u8; LEN];
    p[0] = 0x47;
    p[1] = if pusi { 0x40 } else { 0x00 } | ((pid >> 8) as u8 & 0x1F);
    p[2] = (pid & 0xFF) as u8;
    let l = payload.len();
    if l == ROOM {
        p[3] = 0x10;
        p[4..].copy_from_slice(payload);
    } else {
        p[3] = 0x30;
        let af_len = ROOM - 1 - l;
        p[4] = af_len as u8;
        if af_len >= 1 {
            p[5] = 0x00;
            for b in p.iter_mut().take(6 + (af_len - 1)).skip(6) {
                *b = 0xFF;
            }
        }
        p[5 + af_len..].copy_from_slice(payload);
    }
    p
}

fn psi(pid: u16, table_id: u8, body: &[u8]) -> Vec<u8> {
    let section_length = body.len() + 4;
    let mut s = vec![
        table_id,
        0xB0 | ((section_length >> 8) as u8 & 0x0F),
        (section_length & 0xFF) as u8,
    ];
    s.extend_from_slice(body);
    s.extend_from_slice(&[0, 0, 0, 0]);
    let mut payload = vec![0u8];
    payload.extend_from_slice(&s);
    ts_packet(pid, true, &payload)
}

fn h264_pes(es: &[u8]) -> Vec<u8> {
    let mut p = vec![0x00, 0x00, 0x01, 0xE0];
    let header = [0x80u8, 0x00, 0x00];
    let len = header.len() + es.len();
    p.push((len >> 8) as u8);
    p.push((len & 0xFF) as u8);
    p.extend_from_slice(&header);
    p.extend_from_slice(es);
    p
}

/// PAT + a single-H.264-stream PMT + three H.264 access units, each its own PES.
fn synthetic_ts() -> Vec<u8> {
    let pmt_pid = 0x1000u16;
    let es_pid = 0x0100u16;
    let mut s = Vec::new();
    s.extend_from_slice(&psi(
        0x0000,
        0x00,
        &[
            0,
            1,
            0xC1,
            0,
            0,
            0,
            1,
            0xE0 | (pmt_pid >> 8) as u8 & 0x1F,
            pmt_pid as u8,
        ],
    ));
    s.extend_from_slice(&psi(
        pmt_pid,
        0x02,
        &[
            0x00,
            0x01,
            0xC1,
            0x00,
            0x00,
            0xE0 | (es_pid >> 8) as u8 & 0x1F,
            es_pid as u8,
            0xF0,
            0x00,
            0x1B, // stream_type H.264
            0xE0 | (es_pid >> 8) as u8 & 0x1F,
            es_pid as u8,
            0xF0,
            0x00,
        ],
    ));
    for n in 0..3u8 {
        // Each PES is one IDR access unit: start code, NAL header (type 5), then a
        // slice header whose first byte has the top bit set (`first_mb_in_slice ==
        // 0`, the mark of a new picture's first slice) so the access-unit-aligning
        // h264parse counts three distinct pictures; `n` keeps them distinguishable.
        s.extend_from_slice(&ts_packet(
            es_pid,
            true,
            &h264_pes(&[0, 0, 0, 1, 0x65, 0x88, n]),
        ));
    }
    s
}

fn vint(value: u64) -> Vec<u8> {
    let mut len = 1usize;
    while len < 8 && value >= (1u64 << (7 * len)) - 1 {
        len += 1;
    }
    let mut out = vec![0u8; len];
    let mut v = value;
    for i in (0..len).rev() {
        out[i] = (v & 0xFF) as u8;
        v >>= 8;
    }
    out[0] |= 1 << (8 - len);
    out
}

fn elem(id: &[u8], body: &[u8]) -> Vec<u8> {
    let mut out = id.to_vec();
    out.extend_from_slice(&vint(body.len() as u64));
    out.extend_from_slice(body);
    out
}

/// A minimal single-track (VP9) WebM with one Cluster of two blocks.
fn synthetic_webm() -> Vec<u8> {
    let video = {
        // PixelWidth / PixelHeight as single-byte uints (value only needs to be
        // a valid positive dim for the demuxer to emit refined caps).
        let v = [elem(&[0xB0], &[64]), elem(&[0xBA], &[64])].concat();
        let body = [
            elem(&[0xD7], &[1]),
            elem(&[0x86], b"V_VP9"),
            elem(&[0xE0], &v),
        ]
        .concat();
        elem(&[0xAE], &body)
    };
    let tracks = elem(&[0x16, 0x54, 0xAE, 0x6B], &video);
    let block = |rel: i16, f: &[u8]| {
        let mut b = vint(1);
        b.extend_from_slice(&rel.to_be_bytes());
        b.push(0x80);
        b.extend_from_slice(f);
        b
    };
    let cluster = elem(
        &[0x1F, 0x43, 0xB6, 0x75],
        &[
            elem(&[0xE7], &[0]),
            elem(&[0xA3], &block(0, &[1, 2])),
            elem(&[0xA3], &block(10, &[3, 4])),
        ]
        .concat(),
    );
    let segment = elem(&[0x18, 0x53, 0x80, 0x67], &[tracks, cluster].concat());
    [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat()
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
async fn filesrc_mpegts_explicit_feeds_tsdemux() {
    let path = write_temp("g2g_m112_explicit.ts", &synthetic_ts());
    let text = format!(
        "filesrc location={} bytestream-format=mpegts ! tsdemux ! h264parse ! fakesink",
        path.display()
    );
    assert_eq!(
        run_pipeline(&text).await,
        3,
        "three demuxed H.264 AUs reached the sink"
    );
    let _ = fs::remove_file(&path);
}

#[tokio::test]
async fn filesrc_auto_sniffs_mpegts() {
    let path = write_temp("g2g_m112_auto.ts", &synthetic_ts());
    let text = format!(
        "filesrc location={} bytestream-format=auto ! tsdemux ! fakesink",
        path.display()
    );
    assert_eq!(
        run_pipeline(&text).await,
        3,
        "auto-detected TS demuxed to the sink"
    );
    let _ = fs::remove_file(&path);
}

#[tokio::test]
async fn filesrc_auto_sniffs_matroska() {
    let path = write_temp("g2g_m112_auto.webm", &synthetic_webm());
    let text = format!(
        "filesrc location={} bytestream-format=auto ! matroskademux stream=vp9 ! fakesink",
        path.display()
    );
    assert_eq!(
        run_pipeline(&text).await,
        2,
        "auto-detected WebM VP9 demuxed to the sink"
    );
    let _ = fs::remove_file(&path);
}
