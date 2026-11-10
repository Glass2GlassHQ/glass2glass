// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![allow(dead_code)] // no one test file uses every helper here

use std::path::PathBuf;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::mpegts::{TsMuxer, STREAM_TYPE_H264, STREAM_TYPE_PRIVATE_PES};

// gst-plugins-rs cctost2038anc wrote this CEA-708 CDP packet at its default line 9, offset 0
pub(crate) const GST_CEA708_PACKET: &[u8] = include_bytes!("../fixtures/st2038_cea708_gst.bin");
pub(crate) const CEA608_S334: [u8; 3] = [0x80, 0x94, 0x2C];
pub(crate) const PTS_90KHZ: u64 = 900_000;
// a 25 fps frame on the 90 kHz clock
pub(crate) const FRAME_90KHZ: u64 = 3_600;
// gst tsdemux needs about a second of frames before it activates a program
pub(crate) const FRAMES: usize = 25;
const IDR: [u8; 6] = [0, 0, 0, 1, 0x65, 0x11];

pub(crate) struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

pub(crate) fn scratch(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(name);
    let _ = std::fs::remove_file(&path);
    path
}

// an H.264 stream and an ST 2038 stream carrying `st2038_payload` at every frame
pub(crate) fn video_and_st2038_ts(st2038_payload: &[u8]) -> Vec<u8> {
    let mut muxer = TsMuxer::with_streams(&[STREAM_TYPE_H264, STREAM_TYPE_PRIVATE_PES]);
    muxer.set_stream_st2038(1);
    let mut ts = Vec::new();
    for frame in 0..FRAMES as u64 {
        let pts = PTS_90KHZ + frame * FRAME_90KHZ;
        ts.extend(muxer.push_au_on(0, &IDR, Some(pts), None, true));
        ts.extend(muxer.push_au_on(1, st2038_payload, Some(pts), None, false));
    }
    ts
}

pub(crate) async fn run_launch(line: &str) {
    let registry = g2g_plugins::registry::default_registry();
    let graph = parse_launch(&registry, line).expect("the launch line parses");
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the launch line runs");
}
