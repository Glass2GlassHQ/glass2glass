#![cfg(feature = "std")]

#[path = "common/collect_sink.rs"]
mod collect_sink;

use collect_sink::CollectSink;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{AsyncElement, ByteStreamEncoding, Caps};
use g2g_plugins::mpegts::{TsMuxer, MUX_PMT_PID, STREAM_TYPE_H264, TS_PACKET_LEN};
use g2g_plugins::tsdemux::{TsDemux, TsStream};

const PID_PAT: u16 = 0x0000;
const PAYLOAD_UNIT_START: u8 = 0x40;
const FRAME_PERIOD_90KHZ: u64 = 3000;
const UNITS: usize = 6;
const UNITS_BEFORE_TABLES: usize = 3;
const UNIT_LEN: usize = 400;

fn pid_of(packet: &[u8]) -> u16 {
    (u16::from(packet[1] & 0x1F) << 8) | u16::from(packet[2])
}

fn is_table(packet: &[u8]) -> bool {
    matches!(pid_of(packet), PID_PAT | MUX_PMT_PID)
}

fn access_unit(index: usize) -> Vec<u8> {
    let nal_type = if index == 0 { 0x65 } else { 0x41 };
    let mut unit = vec![0, 0, 0, 1, nal_type];
    unit.extend((0..UNIT_LEN).map(|offset| (offset + index) as u8 & 0x3F));
    unit
}

// a capture that starts between two table repeats opens with unit packets
fn stream_with_late_tables(units: &[Vec<u8>], units_before_tables: usize) -> Vec<u8> {
    let mut mux = TsMuxer::new(STREAM_TYPE_H264);
    let mut packets = Vec::new();
    let mut late_unit_start = None;
    for (index, unit) in units.iter().enumerate() {
        let ts = mux.push_au(unit, Some(index as u64 * FRAME_PERIOD_90KHZ), None, false);
        if index == units_before_tables {
            let first_unit_packet = ts
                .chunks(TS_PACKET_LEN)
                .position(|packet| !is_table(packet) && packet[1] & PAYLOAD_UNIT_START != 0)
                .expect("each unit starts a PES");
            late_unit_start = Some(packets.len() + first_unit_packet);
        }
        packets.extend(ts.chunks(TS_PACKET_LEN).map(<[u8]>::to_vec));
    }
    let late_unit_start = late_unit_start.expect("the tables land inside the stream");
    let (early, late) = packets.split_at(late_unit_start);
    let (early_tables, early_units): (Vec<_>, Vec<_>) =
        early.iter().partition(|packet| is_table(packet));
    assert!(!early_tables.is_empty(), "the muxer sent its tables first");
    early_units
        .into_iter()
        .chain(early_tables)
        .chain(late)
        .flatten()
        .copied()
        .collect()
}

#[tokio::test]
async fn units_sent_before_the_pmt_are_emitted() {
    let units: Vec<Vec<u8>> = (0..UNITS).map(access_unit).collect();
    let ts = stream_with_late_tables(&units, UNITS_BEFORE_TABLES);
    let first_table = ts
        .chunks(TS_PACKET_LEN)
        .position(is_table)
        .expect("the stream carries tables");
    assert!(first_table > 0, "the stream opens with unit packets");

    let mut demux = TsDemux::new().with_stream(TsStream::H264);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        })
        .unwrap();
    let mut sink = CollectSink::default();
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(ts.into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    demux
        .process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .unwrap();
    demux.process(PipelinePacket::Eos, &mut sink).await.unwrap();

    let emitted: Vec<Vec<u8>> = sink
        .packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => frame.domain.as_system_slice().map(<[u8]>::to_vec),
            _ => None,
        })
        .collect();
    assert_eq!(emitted, units);
}
