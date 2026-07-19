//! M133 Opus parser, end to end through `parse_launch`: `filesrc` reads a
//! synthetic Ogg file, `oggdemux` splits out the Opus packets, and `opusparse`
//! refines the caps from each packet's TOC before the sink. Exercises the new
//! parser inside a real registry-built pipeline downstream of a demuxer; the TOC
//! decode itself is unit-tested in `opusparse`.
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

/// OpusHead (BOS) + OpusTags + a page of three audio packets, each a valid
/// stereo Opus TOC (config 31 = CELT fullband 20 ms, stereo bit set) plus a
/// payload byte the parser ignores.
fn synthetic_ogg() -> Vec<u8> {
    let serial = 0xABCD_1234;
    const STEREO_TOC: u8 = (31 << 3) | (1 << 2);
    let mut s = Vec::new();
    s.extend_from_slice(&page(0x02, serial, 0, &[&opus_head(2)]));
    s.extend_from_slice(&page(0x00, serial, 1, &[b"OpusTags\0\0\0\0"]));
    s.extend_from_slice(&page(
        0x00,
        serial,
        2,
        &[
            &[STEREO_TOC, 0x00],
            &[STEREO_TOC, 0x01],
            &[STEREO_TOC, 0x02],
        ],
    ));
    s
}

fn write_temp(name: &str, bytes: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(name);
    fs::write(&path, bytes).expect("write temp fixture");
    path
}

#[tokio::test]
async fn oggdemux_feeds_opusparse_end_to_end() {
    let path = write_temp("g2g_m133_opusparse.opus", &synthetic_ogg());
    let reg = default_registry();
    let text = format!(
        "filesrc location={} bytestream-format=ogg ! oggdemux ! opusparse ! fakesink",
        path.display()
    );
    let graph = parse_launch(&reg, &text).expect("pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("pipeline runs");
    assert_eq!(
        stats.frames_consumed, 3,
        "three Opus packets pass through the parser to the sink"
    );
    let _ = fs::remove_file(&path);
}

#[test]
fn opusparse_registered_and_constructable() {
    let reg = default_registry();
    assert!(
        reg.inspect("opusparse").is_some(),
        "opusparse joins the default registry"
    );
    assert!(
        reg.make_element("opusparse").is_some(),
        "opusparse builds by name"
    );
}
