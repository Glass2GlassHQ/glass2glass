//! M1244-M1247: the MPEG-TS demuxer keeps timestamps rising across the 33-bit
//! wrap, drops what a lost packet damaged, reassembles PSI sections spanning
//! packets, and reads the 192-byte M2TS and 204-byte DVB-ASI packet strides.

#![cfg(feature = "std")]

use g2g_core::fanout::MultiOutputSink;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, G2gError, MultiOutputElement, OutputSink, PushOutcome,
};
use g2g_plugins::mpegts::{
    TsDemuxer, TsMuxer, TsPacketSplitter, STREAM_TYPE_AAC, STREAM_TYPE_H264, STREAM_TYPE_H265,
    SYNC_BYTE, TS_PACKET_LEN,
};
use g2g_plugins::tsdemux::{TsDemux, TsDemuxN, TsStream};
use g2g_plugins::typefind::{sniff, SNIFF_LEN};
use g2g_plugins::uridecodebin::ts_primary_stream;

const PES_TIMESTAMP_CYCLE_90KHZ: u64 = 1 << 33;
const FRAME_PERIOD_90KHZ: u64 = 3000;
const NANOSECONDS_PER_SECOND: u128 = 1_000_000_000;
const TICKS_PER_SECOND: u128 = 90_000;
const PAYLOAD_UNIT_START: u8 = 0x40;
const ADAPTATION_FIELD_PRESENT: u8 = 0x20;
const PAYLOAD_ONLY: u8 = 0x10;
const ADAPTATION_FIELD_CONTROL: u8 = 0x30;
const DISCONTINUITY_INDICATOR: u8 = 0x80;
const PAYLOAD_ROOM: usize = TS_PACKET_LEN - 4;
const M2TS_PREFIX_LEN: usize = 4;
const DVB_ASI_PARITY_LEN: usize = 16;
const READ_CHUNK_LEN: usize = 1000;

fn byte_stream_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    }
}

fn data_frame(bytes: Vec<u8>) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

fn frame_of(packet: PipelinePacket) -> Option<(Vec<u8>, u64)> {
    let PipelinePacket::DataFrame(frame) = packet else {
        return None;
    };
    let data = frame.domain.as_system_slice()?.to_vec();
    Some((data, frame.timing.pts_ns))
}

#[derive(Default)]
struct FrameTap {
    frames: Vec<(Vec<u8>, u64)>,
}

impl OutputSink for FrameTap {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        self.frames.extend(frame_of(packet));
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

struct PortTap {
    frames: Vec<Vec<(Vec<u8>, u64)>>,
}

impl MultiOutputSink for PortTap {
    fn poll_push_to(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        port: usize,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        self.frames[port].extend(frame_of(packet));
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }

    fn port_count(&self) -> usize {
        self.frames.len()
    }
}

async fn demux_h264(chunks: &[&[u8]]) -> Vec<(Vec<u8>, u64)> {
    let mut demux = TsDemux::new().with_stream(TsStream::H264);
    demux.configure_pipeline(&byte_stream_caps()).unwrap();
    let mut tap = FrameTap::default();
    for chunk in chunks {
        demux
            .process(data_frame(chunk.to_vec()), &mut tap)
            .await
            .unwrap();
    }
    demux.process(PipelinePacket::Eos, &mut tap).await.unwrap();
    tap.frames
}

fn video_access_unit(index: usize, len: usize) -> Vec<u8> {
    let nal_type = if index == 0 { 0x65 } else { 0x41 };
    let mut au = vec![0, 0, 0, 1, nal_type];
    au.extend((0..len).map(|offset| (offset + index) as u8 & 0x3F));
    au
}

fn pid_of(packet: &[u8]) -> u16 {
    (u16::from(packet[1] & 0x1F) << 8) | u16::from(packet[2])
}

fn video_pid(ts: &[u8]) -> u16 {
    let mut probe = TsDemuxer::new();
    for packet in ts.chunks(TS_PACKET_LEN) {
        probe.push_packet(packet);
    }
    probe.video_pid().expect("the PMT names a video stream")
}

fn demux_units(packets: &[&[u8]]) -> Vec<Vec<u8>> {
    let mut demux = TsDemuxer::new();
    for packet in packets {
        demux.push_packet(packet);
    }
    demux.flush();
    demux
        .take_units()
        .into_iter()
        .map(|unit| unit.data)
        .collect()
}

fn is_strictly_rising(stamps: &[u64]) -> bool {
    stamps.windows(2).all(|pair| pair[0] < pair[1])
}

fn pts_of(frames: &[(Vec<u8>, u64)]) -> Vec<u64> {
    frames.iter().map(|(_, pts_ns)| *pts_ns).collect()
}

#[tokio::test]
async fn timestamps_keep_rising_across_the_33_bit_wrap() {
    const FRAMES: u64 = 20;
    let first = PES_TIMESTAMP_CYCLE_90KHZ - FRAMES / 2 * FRAME_PERIOD_90KHZ;
    let mut mux = TsMuxer::with_streams(&[STREAM_TYPE_H264, STREAM_TYPE_AAC]);
    let mut ts = Vec::new();
    for index in 0..FRAMES {
        let pts = first + index * FRAME_PERIOD_90KHZ;
        let video = video_access_unit(index as usize, 32);
        ts.extend(mux.push_au_on(0, &video, Some(pts), None, false));
        ts.extend(mux.push_au_on(
            1,
            &[0xFF, 0xF1, 0x50, 0x80, index as u8],
            Some(pts),
            None,
            false,
        ));
    }
    let expected_ns: Vec<u64> = (0..FRAMES)
        .map(|index| {
            let ticks = u128::from(first + mux.decoder_delay_90khz() + index * FRAME_PERIOD_90KHZ);
            (ticks * NANOSECONDS_PER_SECOND / TICKS_PER_SECOND) as u64
        })
        .collect();

    let single = demux_h264(&[&ts]).await;
    assert!(is_strictly_rising(&pts_of(&single)), "tsdemux PTS");
    assert_eq!(pts_of(&single), expected_ns);

    let mut demux = TsDemuxN::new(vec![TsStream::H264, TsStream::Aac]);
    demux.configure_pipeline(&byte_stream_caps()).unwrap();
    let mut tap = PortTap {
        frames: vec![Vec::new(); 2],
    };
    demux.process(data_frame(ts), &mut tap).await.unwrap();
    demux.process(PipelinePacket::Eos, &mut tap).await.unwrap();
    for port in &tap.frames {
        assert!(is_strictly_rising(&pts_of(port)), "tsdemuxn port PTS");
        assert_eq!(pts_of(port), expected_ns);
    }
}

struct DamagedStream {
    access_units: Vec<Vec<u8>>,
    packets: Vec<Vec<u8>>,
    // index into `packets` of a payload-only packet in the middle of access unit `DAMAGED_UNIT`
    lost_packet: usize,
}

const DAMAGED_UNIT: usize = 2;

fn stream_with_a_mid_pes_packet() -> DamagedStream {
    const UNITS: usize = 5;
    const UNIT_LEN: usize = 400;
    let mut mux = TsMuxer::new(STREAM_TYPE_H264);
    let access_units: Vec<Vec<u8>> = (0..UNITS)
        .map(|index| video_access_unit(index, UNIT_LEN))
        .collect();
    let mut ts = Vec::new();
    for (index, au) in access_units.iter().enumerate() {
        let pts = index as u64 * FRAME_PERIOD_90KHZ;
        ts.extend(mux.push_au(au, Some(pts), None, false));
    }
    let pid = video_pid(&ts);
    let packets: Vec<Vec<u8>> = ts.chunks(TS_PACKET_LEN).map(<[u8]>::to_vec).collect();
    let unit_starts: Vec<usize> = packets
        .iter()
        .enumerate()
        .filter(|(_, packet)| pid_of(packet) == pid && packet[1] & PAYLOAD_UNIT_START != 0)
        .map(|(index, _)| index)
        .collect();
    let lost_packet = unit_starts[DAMAGED_UNIT] + 1;
    assert_eq!(pid_of(&packets[lost_packet]), pid);
    assert_eq!(packets[lost_packet][1] & PAYLOAD_UNIT_START, 0);
    assert_eq!(
        packets[lost_packet][3] & ADAPTATION_FIELD_CONTROL,
        PAYLOAD_ONLY
    );
    DamagedStream {
        access_units,
        packets,
        lost_packet,
    }
}

#[test]
fn a_lost_packet_drops_only_its_access_unit() {
    let stream = stream_with_a_mid_pes_packet();
    let kept: Vec<&[u8]> = stream
        .packets
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != stream.lost_packet)
        .map(|(_, packet)| packet.as_slice())
        .collect();
    let mut expected = stream.access_units.clone();
    expected.remove(DAMAGED_UNIT);
    assert_eq!(demux_units(&kept), expected);
}

#[test]
fn a_discontinuity_indicator_after_the_gap_keeps_every_access_unit() {
    let mut stream = stream_with_a_mid_pes_packet();
    let next = &mut stream.packets[stream.lost_packet + 1];
    assert!(
        next[3] & ADAPTATION_FIELD_PRESENT != 0 && next[4] > 0,
        "the packet after the gap has adaptation field flags to set"
    );
    next[5] |= DISCONTINUITY_INDICATOR;
    let kept: Vec<&[u8]> = stream
        .packets
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != stream.lost_packet)
        .map(|(_, packet)| packet.as_slice())
        .collect();
    let units = demux_units(&kept);
    assert_eq!(units.len(), stream.access_units.len());
    for (index, (unit, original)) in units.iter().zip(&stream.access_units).enumerate() {
        if index == DAMAGED_UNIT {
            assert_eq!(unit.len(), original.len() - PAYLOAD_ROOM);
        } else {
            assert_eq!(unit, original);
        }
    }
}

#[test]
fn a_pmt_spanning_two_packets_names_every_stream() {
    const STREAMS_PER_KIND: usize = 12;
    const LANGUAGES: [&str; 3] = ["eng", "fra", "deu"];
    let stream_types: Vec<u8> = [STREAM_TYPE_H264, STREAM_TYPE_AAC]
        .iter()
        .flat_map(|&stream_type| [stream_type; STREAMS_PER_KIND])
        .collect();
    let mut mux = TsMuxer::with_streams(&stream_types);
    for index in 0..stream_types.len() {
        mux.set_stream_language(index, LANGUAGES[index % LANGUAGES.len()]);
    }
    let ts = mux.push_au(&video_access_unit(0, 8), Some(0), None, false);

    let mut demux = TsDemuxer::new();
    for packet in ts.chunks(TS_PACKET_LEN) {
        demux.push_packet(packet);
    }
    // the muxer writes the PAT, then the PMT
    let pmt_pid = pid_of(&ts[TS_PACKET_LEN..]);
    let pmt_packets = ts
        .chunks(TS_PACKET_LEN)
        .filter(|packet| pid_of(packet) == pmt_pid)
        .count();
    assert!(pmt_packets > 1, "the PMT spans packets");

    let streams = demux.streams();
    assert_eq!(streams.len(), stream_types.len());
    for (index, stream) in streams.iter().enumerate() {
        assert_eq!(stream.stream_type, stream_types[index]);
        assert_eq!(
            stream.language_code(),
            Some(LANGUAGES[index % LANGUAGES.len()])
        );
    }
}

fn stride_test_stream() -> (Vec<Vec<u8>>, Vec<u8>) {
    const UNITS: usize = 12;
    let mut mux = TsMuxer::new(STREAM_TYPE_H264);
    let access_units: Vec<Vec<u8>> = (0..UNITS)
        .map(|index| video_access_unit(index, 150 + 97 * index))
        .collect();
    let mut ts = Vec::new();
    for (index, au) in access_units.iter().enumerate() {
        ts.extend(mux.push_au(au, Some(index as u64 * FRAME_PERIOD_90KHZ), None, false));
    }
    (access_units, ts)
}

fn m2ts_wrap(ts: &[u8]) -> Vec<u8> {
    const ARRIVAL_TICKS_PER_PACKET: u32 = 1013;
    const ARRIVAL_TIMESTAMP_MASK: u32 = 0x3FFF_FFFF;
    let mut out = Vec::new();
    for (index, packet) in ts.chunks(TS_PACKET_LEN).enumerate() {
        let arrival = (index as u32 * ARRIVAL_TICKS_PER_PACKET) & ARRIVAL_TIMESTAMP_MASK;
        out.extend_from_slice(&arrival.to_be_bytes());
        out.extend_from_slice(packet);
    }
    out
}

fn dvb_asi_wrap(ts: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for (index, packet) in ts.chunks(TS_PACKET_LEN).enumerate() {
        out.extend_from_slice(packet);
        out.extend((0..DVB_ASI_PARITY_LEN).map(|offset| (index * 31 + offset * 7) as u8));
    }
    out
}

#[tokio::test]
async fn every_packet_stride_demuxes_identical_access_units() {
    let (access_units, ts) = stride_test_stream();
    let expected: Vec<Vec<u8>> = access_units;
    for (name, wrapped) in [
        ("188", ts.clone()),
        ("192", m2ts_wrap(&ts)),
        ("204", dvb_asi_wrap(&ts)),
    ] {
        let whole: Vec<Vec<u8>> = demux_h264(&[&wrapped])
            .await
            .into_iter()
            .map(|(data, _)| data)
            .collect();
        assert_eq!(whole, expected, "stride {name} fed whole");
        let chunks: Vec<&[u8]> = wrapped.chunks(READ_CHUNK_LEN).collect();
        let chunked: Vec<Vec<u8>> = demux_h264(&chunks)
            .await
            .into_iter()
            .map(|(data, _)| data)
            .collect();
        assert_eq!(chunked, expected, "stride {name} fed in chunks");
    }
}

#[test]
fn a_short_trailing_packet_demuxes_at_end_of_stream() {
    const KEPT_PARITY: usize = 5;
    let (access_units, ts) = stride_test_stream();
    let mut truncated = dvb_asi_wrap(&ts);
    truncated.truncate(truncated.len() - (DVB_ASI_PARITY_LEN - KEPT_PARITY));
    for wrapped in [m2ts_wrap(&ts), truncated] {
        let mut splitter = TsPacketSplitter::new();
        let mut demux = TsDemuxer::new();
        splitter.push(&wrapped, &mut demux);
        splitter.finish(&mut demux);
        demux.flush();
        let units: Vec<Vec<u8>> = demux
            .take_units()
            .into_iter()
            .map(|unit| unit.data)
            .collect();
        assert_eq!(units, access_units);
    }
}

#[tokio::test]
async fn the_splitter_resyncs_after_junk_between_packets() {
    // a sync byte at offset 4 would sit where the locked stride expects the next packet
    const JUNK: [u8; 7] = [0x00, SYNC_BYTE, 0x01, 0x02, 0x03, SYNC_BYTE, 0x04];
    const JUNK_AFTER_PACKET: usize = 6;
    let (access_units, ts) = stride_test_stream();
    let wrapped = m2ts_wrap(&ts);
    let split_at = JUNK_AFTER_PACKET * (M2TS_PREFIX_LEN + TS_PACKET_LEN);
    let mut damaged = wrapped[..split_at].to_vec();
    damaged.extend_from_slice(&JUNK);
    damaged.extend_from_slice(&wrapped[split_at..]);
    let units: Vec<Vec<u8>> = demux_h264(&[&damaged])
        .await
        .into_iter()
        .map(|(data, _)| data)
        .collect();
    assert_eq!(units, access_units);
}

fn every_stride(ts: &[u8]) -> [(&'static str, Vec<u8>); 3] {
    [
        ("188", ts.to_vec()),
        ("192", m2ts_wrap(ts)),
        ("204", dvb_asi_wrap(ts)),
    ]
}

#[test]
fn every_packet_stride_sniffs_as_mpegts() {
    let (_, ts) = stride_test_stream();
    for (name, wrapped) in every_stride(&ts) {
        let header = &wrapped[..SNIFF_LEN.min(wrapped.len())];
        assert_eq!(
            sniff(header),
            Some(ByteStreamEncoding::MpegTs),
            "stride {name}"
        );
    }
}

#[test]
fn decodebin_probes_the_pmt_at_every_packet_stride() {
    const UNITS: usize = 4;
    let mut mux = TsMuxer::new(STREAM_TYPE_H265);
    let mut ts = Vec::new();
    for index in 0..UNITS {
        let au = video_access_unit(index, 300);
        ts.extend(mux.push_au(&au, Some(index as u64 * FRAME_PERIOD_90KHZ), None, false));
    }
    let caps = byte_stream_caps();
    for (name, wrapped) in every_stride(&ts) {
        let path =
            std::env::temp_dir().join(format!("g2g-m1247-stride-{name}-{}.ts", std::process::id()));
        std::fs::write(&path, &wrapped).expect("write the transport stream");
        let primary = ts_primary_stream(path.to_str().expect("utf8 path"), &caps)
            .and_then(|streams| streams.streams.into_iter().next());
        std::fs::remove_file(&path).ok();
        let selected_codec = primary.and_then(|primary| {
            primary
                .props
                .into_iter()
                .find(|(key, _)| key == "stream")
                .map(|(_, codec)| codec)
        });
        assert_eq!(
            selected_codec.as_deref(),
            Some("h265"),
            "stride {name} names the PMT's video codec"
        );
    }
}
