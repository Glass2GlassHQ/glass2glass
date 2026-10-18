#![cfg(feature = "std")]

#[path = "common/collect_sink.rs"]
mod collect_sink;

use collect_sink::CollectSink;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{AsyncElement, ByteStreamEncoding, Caps, G2gError};
use g2g_plugins::iv8demux::Iv8Demux;
use g2g_plugins::typefind::sniff;

const PACKET_TYPE_VIDEO: u16 = 257;
const PACKET_TYPE_SKIPPED: u16 = 258;
const HEADER_BYTES_AFTER_SIZE: u16 = 12;
const LAST_PACKET_OF_FRAME: u16 = 0x80;
// the flags of the first packet in the IndigoVision sample, which ffmpeg probes for
const FIRST_PACKET_FLAGS: u16 = 0x8060;
// a first payload this long gives the size 952 the probe also matches
const FIRST_PAYLOAD_LEN: usize = 940;
const IV8_CLOCK_HZ: u64 = 90_000;
const NANOSECONDS_PER_SECOND: u64 = 1_000_000_000;
// a multiple of nine, so the pts lands on a whole nanosecond
const FIRST_FRAME_PTS: u32 = 847_765_926;
const FRAME_PERIOD_90KHZ: u32 = 3600;
const CHUNK_LEN: usize = 7;

fn packet(packet_type: u16, flags: u16, pts_90khz: u32, payload: &[u8]) -> Vec<u8> {
    let size = u16::try_from(payload.len()).unwrap() + HEADER_BYTES_AFTER_SIZE;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&packet_type.to_be_bytes());
    bytes.extend_from_slice(&size.to_be_bytes());
    bytes.extend_from_slice(&flags.to_be_bytes());
    bytes.extend_from_slice(&0u16.to_be_bytes());
    bytes.extend_from_slice(&pts_90khz.to_be_bytes());
    bytes.extend_from_slice(&[0x6A, 0x13, 0xE3, 0x88]);
    bytes.extend_from_slice(payload);
    bytes
}

fn payload(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|offset| (offset as u8).wrapping_add(seed))
        .collect()
}

fn pts_ns(pts_90khz: u32) -> u64 {
    u64::from(pts_90khz) * NANOSECONDS_PER_SECOND / IV8_CLOCK_HZ
}

fn data_frame(bytes: &[u8]) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

async fn demux(stream: &[u8]) -> Result<Vec<(Vec<u8>, u64)>, G2gError> {
    let mut demux = Iv8Demux::new();
    demux.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::Iv8,
    })?;
    let mut sink = CollectSink::default();
    for chunk in stream.chunks(CHUNK_LEN) {
        demux.process(data_frame(chunk), &mut sink).await?;
    }
    demux.process(PipelinePacket::Eos, &mut sink).await?;
    Ok(sink
        .packets
        .into_iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some((
                frame.domain.as_system_slice().unwrap().to_vec(),
                frame.timing.pts_ns,
            )),
            _ => None,
        })
        .collect())
}

struct TwoFrames {
    stream: Vec<u8>,
    first_frame: Vec<u8>,
    second_frame: Vec<u8>,
    second_pts: u32,
}

// the first frame spans two packets with a skipped packet between them
fn two_frames() -> TwoFrames {
    let first_head = payload(FIRST_PAYLOAD_LEN, 1);
    let first_tail = payload(30, 2);
    let second_frame = payload(50, 3);
    let second_pts = FIRST_FRAME_PTS + FRAME_PERIOD_90KHZ;
    let mut stream = packet(
        PACKET_TYPE_VIDEO,
        FIRST_PACKET_FLAGS,
        FIRST_FRAME_PTS,
        &first_head,
    );
    stream.extend(packet(
        PACKET_TYPE_SKIPPED,
        LAST_PACKET_OF_FRAME,
        FIRST_FRAME_PTS,
        &payload(20, 9),
    ));
    stream.extend(packet(
        PACKET_TYPE_VIDEO,
        LAST_PACKET_OF_FRAME,
        second_pts,
        &first_tail,
    ));
    stream.extend(packet(
        PACKET_TYPE_VIDEO,
        LAST_PACKET_OF_FRAME,
        second_pts,
        &second_frame,
    ));
    TwoFrames {
        stream,
        first_frame: [first_head, first_tail].concat(),
        second_frame,
        second_pts,
    }
}

#[test]
fn typefind_recognizes_the_iv8_header() {
    assert_eq!(sniff(&two_frames().stream), Some(ByteStreamEncoding::Iv8));
}

#[tokio::test]
async fn packets_join_into_frames_stamped_with_their_first_pts() {
    let input = two_frames();
    assert_eq!(
        demux(&input.stream).await.unwrap(),
        vec![
            (input.first_frame, pts_ns(FIRST_FRAME_PTS)),
            (input.second_frame, pts_ns(input.second_pts)),
        ]
    );
}

#[tokio::test]
async fn a_truncated_last_payload_ends_the_stream_with_what_arrived() {
    const ARRIVED: usize = 4;
    let mut input = two_frames();
    let third_pts = input.second_pts + FRAME_PERIOD_90KHZ;
    let third_payload = payload(40, 4);
    let third = packet(
        PACKET_TYPE_VIDEO,
        LAST_PACKET_OF_FRAME,
        third_pts,
        &third_payload,
    );
    let header_len = third.len() - third_payload.len();
    input.stream.extend(&third[..header_len + ARRIVED]);
    let frames = demux(&input.stream).await.unwrap();
    assert_eq!(frames.len(), 3);
    assert_eq!(
        frames[2],
        (third_payload[..ARRIVED].to_vec(), pts_ns(third_pts))
    );
}

#[tokio::test]
async fn a_truncated_header_drops_the_unfinished_frame() {
    let mut input = two_frames();
    let pts = input.second_pts + FRAME_PERIOD_90KHZ;
    input
        .stream
        .extend(packet(PACKET_TYPE_VIDEO, 0, pts, &payload(10, 5)));
    let cut_header = packet(PACKET_TYPE_VIDEO, LAST_PACKET_OF_FRAME, pts, &[]);
    input.stream.extend(&cut_header[..cut_header.len() / 2]);
    let frames = demux(&input.stream).await.unwrap();
    assert_eq!(frames.len(), 2);
}

#[tokio::test]
async fn a_size_without_payload_fails_the_parse() {
    let mut input = two_frames();
    input.stream.extend(packet(
        PACKET_TYPE_VIDEO,
        LAST_PACKET_OF_FRAME,
        FIRST_FRAME_PTS,
        &[],
    ));
    assert_eq!(demux(&input.stream).await, Err(G2gError::CapsMismatch));
}
