#![cfg(feature = "std")]

use std::path::PathBuf;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::mpegts::{TsMuxer, STREAM_TYPE_H264, STREAM_TYPE_PRIVATE_PES};
use g2g_plugins::st2038::write_st2038;
use g2g_plugins::st291::{AncPacket, CAPTION_DID, CEA608_SDID};

// gst-plugins-rs cctost2038anc wrote this CEA-708 CDP packet at line 9, offset 0
const GST_CEA708_PACKET: &[u8] = include_bytes!("fixtures/st2038_cea708_gst.bin");
const CEA608_S334: [u8; 3] = [0x80, 0x94, 0x2C];
const CEA608_LINE: u16 = 21;
const PTS_90KHZ: u64 = 900_000;
const FRAME_90KHZ: u64 = 3_600;
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

fn cea608_packet() -> Vec<u8> {
    let mut cea608 = AncPacket::from_bytes(CAPTION_DID, CEA608_SDID, &CEA608_S334);
    cea608.line_number = CEA608_LINE;
    cea608.horizontal_offset = 0;
    write_st2038(&[cea608]).unwrap()
}

fn payload() -> Vec<u8> {
    [GST_CEA708_PACKET, &cea608_packet()].concat()
}

fn captions_ts() -> Vec<u8> {
    let mut muxer = TsMuxer::with_streams(&[STREAM_TYPE_H264, STREAM_TYPE_PRIVATE_PES]);
    muxer.set_stream_st2038(1);
    let mut ts = Vec::new();
    for frame in 0..FRAMES as u64 {
        let pts = PTS_90KHZ + frame * FRAME_90KHZ;
        ts.extend(muxer.push_au_on(0, &IDR, Some(pts), None, true));
        ts.extend(muxer.push_au_on(1, &payload(), Some(pts), None, false));
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
async fn a_named_demux_pad_carries_one_stream() {
    let ts = scratch("g2g_m2133_named.ts");
    let out = scratch("g2g_m2133_named.bin");
    std::fs::write(&ts, captions_ts()).unwrap();
    run_launch(&format!(
        "filesrc location={} bytestream-format=mpegts ! tsdemux stream=st2038 \
         ! st2038ancdemux name=d d.anc_61_02_at_{CEA608_LINE}_0 ! filesink location={}",
        ts.display(),
        out.display()
    ))
    .await;
    assert_eq!(std::fs::read(&out).unwrap(), cea608_packet().repeat(FRAMES));
}

#[tokio::test]
async fn demuxed_streams_mux_back_into_each_frame() {
    for framerate in ["", ",framerate=25/1"] {
        let ts = scratch("g2g_m2133_remux.ts");
        let out = scratch("g2g_m2133_remux.bin");
        std::fs::write(&ts, captions_ts()).unwrap();
        run_launch(&format!(
            "filesrc location={} bytestream-format=mpegts ! tsdemux stream=st2038 \
             ! st2038ancdemux name=d d.anc_61_02_at_{CEA608_LINE}_0 ! m. \
             d.anc_61_01_at_9_0 ! m. st2038ancmux name=m \
             ! meta/x-st-2038,alignment=frame{framerate} ! filesink location={}",
            ts.display(),
            out.display()
        ))
        .await;
        assert_eq!(
            std::fs::read(&out).unwrap(),
            payload().repeat(FRAMES),
            "framerate field {framerate:?}"
        );
    }
}
