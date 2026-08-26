//! Fixtures shared by the DVB EIT tests (`m1049_ts_av1_eit`, `m1056_eit_schedule`):
//! hand-built EIT sections with a real MPEG-2 CRC-32, the TS packetization that
//! carries them, and the collect-into-Vec sink the element legs push into. The CRC
//! is computed here rather than reused from the parser, so a broken CRC in the
//! parser cannot pass by agreeing with itself. One definition, included per test
//! binary via `mod eit_common;`.
#![allow(dead_code)] // no one test file uses every fixture here

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, G2gError, OutputSink, PushOutcome};
use g2g_plugins::mpegts::{TsDemuxer, TS_PACKET_LEN};

/// The PID the EIT rides, and the present/following table id.
pub(crate) const PID_EIT: u16 = 0x0012;
pub(crate) const TABLE_ID_EIT_PF: u8 = 0x4E;
/// The first EIT schedule table id of this transport stream, and the first of the
/// other-TS range, which describes services carried elsewhere.
pub(crate) const TABLE_ID_EIT_SCHEDULE: u8 = 0x50;
pub(crate) const TABLE_ID_EIT_SCHEDULE_OTHER_TS: u8 = 0x60;

/// EN 300 468 Annex C's worked example: MJD 45218 is 1982-09-06, so with the BCD
/// time 12:45:00 this `start_time` is 1982-09-06 12:45:00 UTC.
pub(crate) const ANNEX_C_START_TIME: [u8; 5] = [0xB0, 0xA2, 0x12, 0x45, 0x00];
/// What it decodes to, the value `date -u -d @400164300` prints as that instant.
pub(crate) const ANNEX_C_START_UNIX_SECS: u64 = 400_164_300;
/// The `start_time` EN 300 468 defines as undefined.
pub(crate) const UNDEFINED_START_TIME: [u8; 5] = [0xFF; 5];
/// A 90-minute `duration` in BCD hh:mm:ss, and the same in seconds.
pub(crate) const EVENT_DURATION: [u8; 3] = [0x01, 0x30, 0x00];
pub(crate) const EVENT_DURATION_SECS: u32 = 5400;

/// MPEG-2 CRC-32.
pub(crate) fn crc32_mpeg(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= (b as u32) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 {
                (crc << 1) ^ 0x04C1_1DB7
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// A DVB `short_event_descriptor` (tag 0x4D) for one event.
pub(crate) fn short_event_descriptor(name: &[u8], text: &[u8]) -> Vec<u8> {
    let mut body = Vec::from(*b"eng");
    body.push(name.len() as u8);
    body.extend_from_slice(name);
    body.push(text.len() as u8);
    body.extend_from_slice(text);
    let mut out = Vec::from([0x4D, body.len() as u8]);
    out.extend_from_slice(&body);
    out
}

/// One EIT event entry: the id, the 5-byte start_time and 3-byte duration, then
/// the descriptor loop.
pub(crate) fn eit_event(
    event_id: u16,
    start_time: [u8; 5],
    duration: [u8; 3],
    descriptors: &[u8],
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&event_id.to_be_bytes());
    out.extend_from_slice(&start_time);
    out.extend_from_slice(&duration);
    let len = descriptors.len();
    out.push(0x80 | ((len >> 8) as u8 & 0x0F)); // running_status 4, free_CA 0
    out.push(len as u8);
    out.extend_from_slice(descriptors);
    out
}

/// A whole EIT section with a correct trailing CRC-32.
pub(crate) fn eit_section(
    table_id: u8,
    service_id: u16,
    version: u8,
    section_number: u8,
    events: &[u8],
) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&service_id.to_be_bytes());
    body.push(0xC1 | (version << 1)); // reserved, version, current_next = 1
    body.push(section_number);
    // Present/following runs to section 1; a schedule section is the last of its
    // own segment here, which is what the sparse real tables also look like.
    let last_section_number = section_number.max(1);
    body.push(last_section_number);
    body.extend_from_slice(&[0x00, 0x01]); // transport_stream_id
    body.extend_from_slice(&[0x00, 0x01]); // original_network_id
    body.push(last_section_number); // segment_last_section_number
    body.push(table_id); // last_table_id
    body.extend_from_slice(events);

    let section_length = body.len() + 4; // the body plus the CRC
    let mut section = Vec::from([
        table_id,
        0xB0 | ((section_length >> 8) as u8 & 0x0F),
        section_length as u8,
    ]);
    section.extend_from_slice(&body);
    let crc = crc32_mpeg(&section);
    section.extend_from_slice(&crc.to_be_bytes());
    section
}

/// Carry a PSI section on `pid`, across as many 188-byte packets as it needs.
pub(crate) fn psi_packets(pid: u16, section: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut rest = section;
    let mut continuity = 0u8;
    let mut first = true;
    while first || !rest.is_empty() {
        let mut pkt = Vec::with_capacity(TS_PACKET_LEN);
        let pusi = if first { 0x40 } else { 0x00 };
        pkt.push(0x47);
        pkt.push(pusi | ((pid >> 8) as u8 & 0x1F));
        pkt.push(pid as u8);
        pkt.push(0x10 | (continuity & 0x0F));
        if first {
            pkt.push(0x00); // pointer_field
        }
        let room = TS_PACKET_LEN - pkt.len();
        let take = room.min(rest.len());
        pkt.extend_from_slice(&rest[..take]);
        rest = &rest[take..];
        pkt.resize(TS_PACKET_LEN, 0xFF);
        out.extend_from_slice(&pkt);
        continuity = continuity.wrapping_add(1);
        first = false;
    }
    out
}

/// Feed whole sections through a fresh parser.
pub(crate) fn parse_sections(sections: &[Vec<u8>]) -> TsDemuxer {
    let mut demux = TsDemuxer::new();
    feed_sections(&mut demux, sections);
    demux
}

/// Feed whole sections through a parser that has already read some.
pub(crate) fn feed_sections(demux: &mut TsDemuxer, sections: &[Vec<u8>]) {
    for section in sections {
        for pkt in psi_packets(PID_EIT, section).chunks(TS_PACKET_LEN) {
            demux.push_packet(pkt);
        }
    }
}

/// Records every forwarded access unit.
#[derive(Default)]
pub(crate) struct CaptureSink {
    pub(crate) aus: Vec<Vec<u8>>,
    pub(crate) caps: Vec<Caps>,
}

impl OutputSink for CaptureSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.aus.push(s.to_vec());
                    }
                }
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

pub(crate) fn data_frame(bytes: &[u8]) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}
