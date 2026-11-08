#![cfg(feature = "std")]

use std::path::{Path, PathBuf};
use std::process::Command;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::mpegts::{
    TsDemuxer, TsMuxer, STREAM_TYPE_H264, STREAM_TYPE_PRIVATE_PES, TS_PACKET_LEN,
};

// two ST 2038 packets as gst-plugins-rs cctost2038anc lays them out: a line 9 CEA-708 header and its words
const ANC: [u8; 12] = [
    0x00, 0x02, 0x40, 0x00, 0x18, 0x5A, 0x90, 0x04, 0x2A, 0x95, 0x5F, 0xFF,
];
const PTS_90KHZ: u64 = 900_000;
// a 25 fps frame on the 90 kHz clock, and a second of them
const FRAME_90KHZ: u64 = 3_600;
const FRAMES: u64 = 25;
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

fn video_and_st2038_ts() -> Vec<u8> {
    let mut muxer = TsMuxer::with_streams(&[STREAM_TYPE_H264, STREAM_TYPE_PRIVATE_PES]);
    muxer.set_stream_st2038(1);
    let mut ts = Vec::new();
    for frame in 0..FRAMES {
        let pts = PTS_90KHZ + frame * FRAME_90KHZ;
        ts.extend(muxer.push_au_on(0, &IDR, Some(pts), None, true));
        ts.extend(muxer.push_au_on(1, &ANC, Some(pts), None, false));
    }
    ts
}

fn st2038_payloads(ts: &[u8]) -> Vec<Vec<u8>> {
    let mut demuxer = TsDemuxer::new();
    for packet in ts.chunks(TS_PACKET_LEN) {
        demuxer.push_packet(packet);
    }
    demuxer.flush();
    let pids: Vec<u16> = demuxer.streams().iter().map(|s| s.pid).collect();
    let st2038_pid = pids
        .into_iter()
        .find(|&pid| demuxer.is_st2038(pid))
        .expect("a stream with the 'VANC' registration");
    demuxer
        .take_units()
        .into_iter()
        .filter(|unit| unit.pid == st2038_pid)
        .map(|unit| unit.data)
        .collect()
}

fn gst_launch(pipeline: &str) {
    let status = Command::new("gst-launch-1.0")
        .arg("-q")
        .args(pipeline.split_whitespace())
        .status()
        .expect("run gst-launch-1.0");
    assert!(status.success(), "gst-launch-1.0 {pipeline}");
}

fn path(p: &Path) -> String {
    p.display().to_string()
}

#[test]
#[ignore = "needs gst-launch-1.0 1.24 or later"]
fn gst_tsdemux_reads_the_st2038_stream() {
    let ts = scratch("g2g_m2129_g2g.ts");
    let out = scratch("g2g_m2129_gst_anc.bin");
    std::fs::write(&ts, video_and_st2038_ts()).unwrap();
    gst_launch(&format!(
        "filesrc location={} ! tsdemux name=d d. ! meta/x-st-2038 ! filesink location={}",
        path(&ts),
        path(&out)
    ));
    assert_eq!(std::fs::read(&out).unwrap(), ANC.repeat(FRAMES as usize));
}

#[test]
#[ignore = "needs gst-launch-1.0 1.24 or later"]
fn tsdemux_reads_gst_mpegtsmux_st2038() {
    let ts = scratch("g2g_m2129_source.ts");
    let remuxed = scratch("g2g_m2129_gst.ts");
    std::fs::write(&ts, video_and_st2038_ts()).unwrap();
    gst_launch(&format!(
        "filesrc location={} ! tsdemux name=d d. ! meta/x-st-2038 ! mpegtsmux ! filesink location={}",
        path(&ts),
        path(&remuxed)
    ));
    assert_eq!(
        st2038_payloads(&std::fs::read(&remuxed).unwrap()),
        vec![ANC.to_vec(); FRAMES as usize]
    );
}

#[test]
fn the_st2038_stream_round_trips_through_the_g2g_muxer() {
    assert_eq!(
        st2038_payloads(&video_and_st2038_ts()),
        vec![ANC.to_vec(); FRAMES as usize]
    );
}

#[tokio::test]
async fn mpegtsmux_carries_the_st2038_stream_tsdemux_selects() {
    let source = scratch("g2g_m2129_launch_source.ts");
    let remuxed = scratch("g2g_m2129_launch_remuxed.ts");
    std::fs::write(&source, video_and_st2038_ts()).unwrap();
    let registry = g2g_plugins::registry::default_registry();
    let graph = parse_launch(
        &registry,
        &format!(
            "filesrc location={} bytestream-format=mpegts ! tsdemux stream=st2038 \
             ! mpegtsmux ! filesink location={}",
            path(&source),
            path(&remuxed)
        ),
    )
    .expect("the remux line parses");
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the remux runs");
    assert_eq!(
        st2038_payloads(&std::fs::read(&remuxed).unwrap()),
        vec![ANC.to_vec(); FRAMES as usize]
    );
}
