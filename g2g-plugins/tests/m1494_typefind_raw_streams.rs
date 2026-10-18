//! M1494: typefind takes a raw MPEG-4 Part 2 stream for one, whatever its
//! `.m4v` extension suggests, and does not take start-code runs inside another
//! format for H.264.

#![cfg(all(feature = "std", feature = "ffmpeg"))]

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_plugins::registry::default_registry;

const MPEG4_MP4: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/mpeg4part2_640x480.mp4"
);
// the clip `m563_mpeg4part2` decodes
const MPEG4_FRAMES: u64 = 10;

struct ZeroClock;
impl g2g_core::PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

async fn run_line(line: &str) -> u64 {
    let graph =
        parse_launch(&default_registry(), line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"))
        .frames_consumed
}

fn temp(tag: &str, ext: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("g2g-m1494-{tag}-{}.{ext}", std::process::id()))
}

#[tokio::test]
async fn a_raw_m4v_stream_decodes() {
    let raw = temp("raw", "m4v");
    run_line(&format!(
        "filesrc location={MPEG4_MP4} ! qtdemux ! filesink location={}",
        raw.display()
    ))
    .await;
    let frames = run_line(&format!(
        "filesrc location={} ! decodebin ! video/x-raw,format=I420 ! fakesink",
        raw.display()
    ))
    .await;
    std::fs::remove_file(&raw).ok();
    assert_eq!(frames, MPEG4_FRAMES);
}

fn adts_frame(payload_len: usize) -> Vec<u8> {
    const HEADER_LEN: usize = 7;
    const LC_PROFILE: u8 = 1;
    const SAMPLE_RATE_INDEX_44100: u8 = 4;
    const STEREO: u8 = 2;
    let frame_len = HEADER_LEN + payload_len;
    let mut frame = vec![
        0xFF,
        0xF1,
        LC_PROFILE << 6 | SAMPLE_RATE_INDEX_44100 << 2 | STEREO >> 2,
        (STEREO & 0x3) << 6 | (frame_len >> 11) as u8,
        (frame_len >> 3) as u8,
        ((frame_len & 0x7) as u8) << 5 | 0x1F,
        0xFC,
    ];
    frame.resize(frame_len, 0);
    frame
}

#[test]
fn adts_behind_an_id3_tag_holding_a_start_code_run_is_aac() {
    // the shape of fate-suite aac/id3v2.aac: a tag frame whose body holds an
    // SPS-like `00 00 01 67` run, then the ADTS frames
    let mut tag_body = b"PRIV\x00\x00\x00\x08\x00\x00".to_vec();
    tag_body.extend_from_slice(&[0x00, 0x00, 0x01, 0x67, 0x88, 0x00, 0x00, 0x04]);
    let mut bytes = b"ID3\x03\x00\x00\x00\x00\x00".to_vec();
    bytes.push(tag_body.len() as u8);
    bytes.extend_from_slice(&tag_body);
    for _ in 0..4 {
        bytes.extend_from_slice(&adts_frame(32));
    }
    let path = temp("id3", "bin");
    std::fs::write(&path, &bytes).expect("write file");
    let line = format!("filesrc location={} ! decodebin ! fakesink", path.display());
    let parsed = parse_launch(&default_registry(), &line);
    std::fs::remove_file(&path).ok();
    let graph = parsed.unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let vg = graph.finish().expect("valid graph");
    let names: Vec<String> = vg
        .topo()
        .iter()
        .filter_map(|&n| vg.element(n).map(|e| e.log_category().to_string()))
        .collect();
    assert!(
        names.iter().any(|n| n == "AacParse"),
        "AAC parsed: {names:?}"
    );
}
