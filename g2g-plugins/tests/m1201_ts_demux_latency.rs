#![cfg(feature = "std")]

mod demux_latency_common;

use demux_latency_common::run_demux;
use g2g_core::{ByteStreamEncoding, Caps};
use g2g_plugins::mpegts::{TsMuxer, STREAM_TYPE_H264};
use g2g_plugins::tsdemux::{TsDemuxN, TsStream};

const ACCESS_UNIT_COUNT: u64 = 6;
const FIRST_PTS_90KHZ: u64 = 900_000;
const FRAME_INTERVAL_90KHZ: u64 = 3_600;
const PTS_CLOCK_HZ: u64 = 90_000;
const NANOS_PER_SECOND: u64 = 1_000_000_000;
const FRAME_INTERVAL_NS: u64 = FRAME_INTERVAL_90KHZ * NANOS_PER_SECOND / PTS_CLOCK_HZ;
const IDR_ACCESS_UNIT: [u8; 6] = [0, 0, 0, 1, 0x65, 0x11];
const P_ACCESS_UNIT: [u8; 6] = [0, 0, 0, 1, 0x41, 0x22];

fn transport_stream_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    }
}

// One access unit's TS packets per frame, the way a live capture delivers them.
fn transport_stream_chunks() -> Vec<Vec<u8>> {
    let mut muxer = TsMuxer::with_streams(&[STREAM_TYPE_H264]);
    (0..ACCESS_UNIT_COUNT)
        .map(|i| {
            let access_unit = if i == 0 {
                &IDR_ACCESS_UNIT
            } else {
                &P_ACCESS_UNIT
            };
            let pts = FIRST_PTS_90KHZ + i * FRAME_INTERVAL_90KHZ;
            muxer.push_au(access_unit, Some(pts), None)
        })
        .collect()
}

#[test]
fn tsdemux_paces_its_sinks_with_the_access_unit_it_holds() {
    let (log, stats) = run_demux(
        TsDemuxN::new(vec![TsStream::H264]),
        transport_stream_caps(),
        transport_stream_chunks(),
    );
    assert_eq!(
        log.per_frame_path_latency_ns.len() as u64,
        ACCESS_UNIT_COUNT,
        "every access unit reaches the sink"
    );
    assert_eq!(
        log.startup_path_latency_ns,
        Some(0),
        "nothing is parsed before the run starts"
    );
    assert_eq!(
        log.per_frame_path_latency_ns.last(),
        Some(&FRAME_INTERVAL_NS),
        "the sink ends paced with one held access unit: {:?}",
        log.per_frame_path_latency_ns
    );
    assert_eq!(stats.latency.min_ns, FRAME_INTERVAL_NS);
}
