// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "std")]

use std::path::Path;
use std::process::Command;

use g2g_plugins::mpegts::{TsDemuxer, TS_PACKET_LEN};

#[path = "common/st2038_ts.rs"]
mod st2038_ts;

use st2038_ts::{run_launch, scratch, video_and_st2038_ts, FRAMES};

// two ST 2038 packets as gst-plugins-rs cctost2038anc lays them out: a line 9 CEA-708 header and its words
const ANC: [u8; 12] = [
    0x00, 0x02, 0x40, 0x00, 0x18, 0x5A, 0x90, 0x04, 0x2A, 0x95, 0x5F, 0xFF,
];

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
    std::fs::write(&ts, video_and_st2038_ts(&ANC)).unwrap();
    gst_launch(&format!(
        "filesrc location={} ! tsdemux name=d d. ! meta/x-st-2038 ! filesink location={}",
        path(&ts),
        path(&out)
    ));
    assert_eq!(std::fs::read(&out).unwrap(), ANC.repeat(FRAMES));
}

#[test]
#[ignore = "needs gst-launch-1.0 1.24 or later"]
fn tsdemux_reads_gst_mpegtsmux_st2038() {
    let ts = scratch("g2g_m2129_source.ts");
    let remuxed = scratch("g2g_m2129_gst.ts");
    std::fs::write(&ts, video_and_st2038_ts(&ANC)).unwrap();
    gst_launch(&format!(
        "filesrc location={} ! tsdemux name=d d. ! meta/x-st-2038 ! mpegtsmux ! filesink location={}",
        path(&ts),
        path(&remuxed)
    ));
    assert_eq!(
        st2038_payloads(&std::fs::read(&remuxed).unwrap()),
        vec![ANC.to_vec(); FRAMES]
    );
}

#[test]
fn the_st2038_stream_round_trips_through_the_g2g_muxer() {
    assert_eq!(
        st2038_payloads(&video_and_st2038_ts(&ANC)),
        vec![ANC.to_vec(); FRAMES]
    );
}

#[tokio::test]
async fn mpegtsmux_carries_the_st2038_stream_tsdemux_selects() {
    let source = scratch("g2g_m2129_launch_source.ts");
    let remuxed = scratch("g2g_m2129_launch_remuxed.ts");
    std::fs::write(&source, video_and_st2038_ts(&ANC)).unwrap();
    run_launch(&format!(
        "filesrc location={} bytestream-format=mpegts ! tsdemux stream=st2038 \
         ! mpegtsmux ! filesink location={}",
        path(&source),
        path(&remuxed)
    ))
    .await;
    assert_eq!(
        st2038_payloads(&std::fs::read(&remuxed).unwrap()),
        vec![ANC.to_vec(); FRAMES]
    );
}
