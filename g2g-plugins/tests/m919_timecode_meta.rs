//! M919: the source's SMPTE 12M timecode, mined from the bitstream and burnt in.
//! `H265Parse` reads the `time_code` SEI onto the frame as a `TimecodeMeta`, and
//! `TimeOverlay` draws that count instead of the PTS.
//!
//! Unit under test = `NalParse`'s timecode mine + `TimeOverlay`'s consumption,
//! driven through the real elements.

#![cfg(all(feature = "std", feature = "metadata"))]

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::meta::TimecodeMeta;
use g2g_core::{
    AsyncElement, Caps, Dim, G2gError, MemoryDomain, OutputSink, PipelinePacket, PushOutcome, Rate,
    RawVideoFormat, VideoCodec,
};
use g2g_plugins::h265parse::H265Parse;
use g2g_plugins::sei::{build_sei_nal, PAYLOAD_TIME_CODE};
use g2g_plugins::timeoverlay::TimeOverlay;

/// One H.265 IDR_W_RADL slice NAL (type 19), the picture the SEI describes.
const VCL: [u8; 8] = [0x00, 0x00, 0x00, 0x01, 0x26, 0x01, 0xAF, 0x00];

#[derive(Default)]
struct RecordingSink {
    frames: Vec<Frame>,
}

impl OutputSink for RecordingSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            if let PipelinePacket::DataFrame(f) = packet {
                self.frames.push(f);
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Pack `(value, bit_width)` fields MSB-first, zero-padded to a byte boundary.
fn pack(fields: &[(u32, u32)]) -> Vec<u8> {
    let mut out = Vec::new();
    let (mut acc, mut nbits) = (0u32, 0u32);
    for &(v, w) in fields {
        for i in (0..w).rev() {
            acc = (acc << 1) | ((v >> i) & 1);
            nbits += 1;
            if nbits == 8 {
                out.push(acc as u8);
                acc = 0;
                nbits = 0;
            }
        }
    }
    if nbits > 0 {
        out.push((acc << (8 - nbits)) as u8);
    }
    out
}

/// An H.265 `time_code` payload for one full non-drop timestamp.
fn time_code_payload(h: u32, m: u32, s: u32, frames: u32) -> Vec<u8> {
    pack(&[
        (1, 2),      // num_clock_ts
        (1, 1),      // clock_timestamp_flag
        (0, 1),      // units_field_based_flag
        (0, 5),      // counting_type
        (1, 1),      // full_timestamp_flag
        (0, 1),      // discontinuity_flag
        (0, 1),      // cnt_dropped_flag
        (frames, 9), // n_frames
        (s, 6),      // seconds_value
        (m, 6),      // minutes_value
        (h, 5),      // hours_value
        (0, 5),      // time_offset_length
    ])
}

fn h265_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H265,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

fn au_frame(bytes: Vec<u8>) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

/// Push one `time_code`-carrying access unit through the parser and return the
/// timecode it attached.
async fn parse_timecode(h: u32, m: u32, s: u32, frames: u32) -> Option<TimecodeMeta> {
    let mut au = build_sei_nal(
        PAYLOAD_TIME_CODE,
        &time_code_payload(h, m, s, frames),
        VideoCodec::H265,
    );
    au.extend_from_slice(&VCL);

    let mut el = H265Parse::new();
    el.configure_pipeline(&h265_caps()).unwrap();
    let mut sink = RecordingSink::default();
    el.process(au_frame(au), &mut sink).await.unwrap();
    sink.frames[0].meta.get::<TimecodeMeta>().copied()
}

#[tokio::test]
async fn time_code_sei_becomes_frame_metadata() {
    let tc = parse_timecode(1, 2, 3, 14)
        .await
        .expect("timecode attached");
    assert_eq!((tc.hours, tc.minutes, tc.seconds, tc.frames), (1, 2, 3, 14));
    assert!(!tc.drop_frame);
}

#[tokio::test]
async fn a_stream_without_a_time_code_carries_no_timecode() {
    let mut el = H265Parse::new();
    el.configure_pipeline(&h265_caps()).unwrap();
    let mut sink = RecordingSink::default();
    el.process(au_frame(VCL.to_vec()), &mut sink).await.unwrap();
    assert!(sink.frames[0].meta.get::<TimecodeMeta>().is_none());
}

#[tokio::test]
async fn timeoverlay_burns_in_the_carried_timecode() {
    // The overlay must draw the source's count, not the frame's PTS: render the
    // same frame with and without the meta and require different pixels.
    let caps = Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(96),
        height: Dim::Fixed(32),
        framerate: Rate::Fixed(25 << 16),
        interlace: g2g_core::Interlace::Any,
    };
    let tc = parse_timecode(1, 2, 3, 14)
        .await
        .expect("timecode attached");

    let render = |meta: Option<TimecodeMeta>| {
        let caps = caps.clone();
        async move {
            let mut ov = TimeOverlay::new();
            ov.configure_pipeline(&caps).unwrap();
            let mut f = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(
                    vec![255u8; 96 * 32 * 4].into_boxed_slice(),
                )),
                FrameTiming::default(),
                0,
            );
            if let Some(m) = meta {
                f.meta.attach(m);
            }
            let mut sink = RecordingSink::default();
            ov.process(PipelinePacket::DataFrame(f), &mut sink)
                .await
                .unwrap();
            sink.frames[0]
                .domain
                .as_system_slice()
                .expect("rendered frame")
                .to_vec()
        }
    };

    let with_tc = render(Some(tc)).await;
    let without = render(None).await;
    assert_ne!(
        with_tc, without,
        "the burnt-in text changes when the frame carries a timecode"
    );
}
