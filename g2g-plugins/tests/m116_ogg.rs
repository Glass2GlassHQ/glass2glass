//! M116 Ogg / Opus demux, end to end through `parse_launch`: `filesrc` reads a
//! synthetic Ogg file (its container auto-sniffed via `typefind`), feeds
//! `oggdemux`, and the Opus packets reach the sink. Exercises the new
//! `ByteStream{Ogg}` caps, the OggS sniff, and the demuxer together.
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

/// One Ogg page carrying `packets` (each laced into 255-byte segments).
fn page(header_type: u8, serial: u32, seq: u32, packets: &[&[u8]]) -> Vec<u8> {
    let mut table = Vec::new();
    let mut body = Vec::new();
    for p in packets {
        let mut n = p.len();
        loop {
            let seg = n.min(255);
            table.push(seg as u8);
            n -= seg;
            if seg < 255 {
                break;
            }
        }
        body.extend_from_slice(p);
    }
    let mut out = b"OggS".to_vec();
    out.push(0); // version
    out.push(header_type);
    out.extend_from_slice(&0u64.to_le_bytes()); // granule
    out.extend_from_slice(&serial.to_le_bytes());
    out.extend_from_slice(&seq.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // CRC (ignored on read)
    out.push(table.len() as u8);
    out.extend_from_slice(&table);
    out.extend_from_slice(&body);
    out
}

fn opus_head(channels: u8) -> Vec<u8> {
    let mut h = b"OpusHead".to_vec();
    h.push(1); // version
    h.push(channels);
    h.extend_from_slice(&[0, 0]); // pre-skip
    h.extend_from_slice(&48_000u32.to_le_bytes());
    h.extend_from_slice(&[0, 0, 0]); // output gain + mapping family
    h
}

/// OpusHead (BOS) + OpusTags + a page of three audio packets.
fn synthetic_ogg() -> Vec<u8> {
    let serial = 0xABCD_1234;
    let mut s = Vec::new();
    s.extend_from_slice(&page(0x02, serial, 0, &[&opus_head(2)]));
    s.extend_from_slice(&page(0x00, serial, 1, &[b"OpusTags\0\0\0\0"]));
    s.extend_from_slice(&page(
        0x00,
        serial,
        2,
        &[&[0x10, 0x11], &[0x20], &[0x30, 0x31, 0x32]],
    ));
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
async fn filesrc_auto_sniffs_ogg_and_demuxes_opus() {
    let path = write_temp("g2g_m116_auto.opus", &synthetic_ogg());
    let text = format!(
        "filesrc location={} bytestream-format=auto ! oggdemux ! fakesink",
        path.display()
    );
    assert_eq!(
        run_pipeline(&text).await,
        3,
        "three Opus packets demuxed to the sink"
    );
    let _ = fs::remove_file(&path);
}

#[tokio::test]
async fn filesrc_explicit_ogg_demuxes_opus() {
    let path = write_temp("g2g_m116_explicit.opus", &synthetic_ogg());
    let text = format!(
        "filesrc location={} bytestream-format=ogg ! oggdemux ! fakesink",
        path.display()
    );
    assert_eq!(
        run_pipeline(&text).await,
        3,
        "three Opus packets demuxed to the sink"
    );
    let _ = fs::remove_file(&path);
}
