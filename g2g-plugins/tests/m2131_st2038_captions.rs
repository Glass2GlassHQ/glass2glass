#![cfg(feature = "std")]

use std::path::PathBuf;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::mpegts::{TsMuxer, STREAM_TYPE_H264, STREAM_TYPE_PRIVATE_PES};
use g2g_plugins::st2038::{parse_st2038, write_st2038};
use g2g_plugins::st291::{AncPacket, CAPTION_DID, CEA608_SDID};

// gst-plugins-rs cctost2038anc wrote this CEA-708 CDP packet at its default line 9
const GST_CEA708_PACKET: &[u8] = include_bytes!("fixtures/st2038_cea708_gst.bin");
const CEA608_S334: [u8; 3] = [0x80, 0x94, 0x2C];
const PTS_90KHZ: u64 = 900_000;
const FRAME_90KHZ: u64 = 3_600;
// gst tsdemux needs about a second of frames before it activates a program
const FRAMES: usize = 25;
const IDR: [u8; 6] = [0, 0, 0, 1, 0x65, 0x11];

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn scratch(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(name);
    let _ = std::fs::remove_file(&path);
    path
}

fn cdp() -> Vec<u8> {
    parse_st2038(GST_CEA708_PACKET).unwrap()[0].user_data_bytes()
}

fn captions_ts() -> Vec<u8> {
    let cea608 = AncPacket::from_bytes(CAPTION_DID, CEA608_SDID, &CEA608_S334);
    let mut payload = GST_CEA708_PACKET.to_vec();
    payload.extend(write_st2038(&[cea608]).unwrap());
    let mut muxer = TsMuxer::with_streams(&[STREAM_TYPE_H264, STREAM_TYPE_PRIVATE_PES]);
    muxer.set_stream_st2038(1);
    let mut ts = Vec::new();
    for frame in 0..FRAMES as u64 {
        let pts = PTS_90KHZ + frame * FRAME_90KHZ;
        ts.extend(muxer.push_au_on(0, &IDR, Some(pts), None, true));
        ts.extend(muxer.push_au_on(1, &payload, Some(pts), None, false));
    }
    ts
}

async fn run_launch(line: &str) {
    let registry = g2g_plugins::registry::default_registry();
    let graph = parse_launch(&registry, line).expect("the launch line parses");
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the launch line runs");
}

#[tokio::test]
async fn st2038anctocc_and_cctost2038anc_rebuild_the_gst_packet() {
    let ts = scratch("g2g_m2131_rebuild.ts");
    let out = scratch("g2g_m2131_rebuild.bin");
    std::fs::write(&ts, captions_ts()).unwrap();
    run_launch(&format!(
        "filesrc location={} bytestream-format=mpegts ! tsdemux stream=st2038 \
         ! st2038anctocc ! cctost2038anc ! filesink location={}",
        ts.display(),
        out.display()
    ))
    .await;
    assert_eq!(std::fs::read(&out).unwrap(), GST_CEA708_PACKET.repeat(FRAMES));
}

#[tokio::test]
async fn st2038anctocc_splits_captions_to_named_pads() {
    let ts = scratch("g2g_m2131_split.ts");
    let cea708 = scratch("g2g_m2131_split_708.bin");
    let cea608 = scratch("g2g_m2131_split_608.bin");
    std::fs::write(&ts, captions_ts()).unwrap();
    run_launch(&format!(
        "filesrc location={} bytestream-format=mpegts ! tsdemux stream=st2038 \
         ! st2038anctocc name=d d.src_cea608 ! filesink location={} \
         d.src_cea708 ! filesink location={}",
        ts.display(),
        cea608.display(),
        cea708.display()
    ))
    .await;
    assert_eq!(std::fs::read(&cea708).unwrap(), cdp().repeat(FRAMES));
    assert_eq!(std::fs::read(&cea608).unwrap(), CEA608_S334.repeat(FRAMES));
}
