#![cfg(feature = "std")]

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{AsyncElement, Caps, Dim, G2gError, OutputSink, PushOutcome, Rate, VideoCodec};
use g2g_plugins::h265parse::H265Parse;

const VPS: u8 = 32;
const SPS: u8 = 33;
const PPS: u8 = 34;
const TRAIL_R: u8 = 1;
const RASL_N: u8 = 8;
const RASL_R: u8 = 9;
const CRA: u8 = 21;
const END_OF_SEQUENCE: u8 = 36;
const FIRST_SLICE_IN_PICTURE: u8 = 0x80;

#[derive(Default)]
struct PictureTags {
    tags: Vec<u8>,
}

impl OutputSink for PictureTags {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        if let PipelinePacket::DataFrame(frame) = packet {
            let bytes = frame.domain.as_system_slice().expect("system memory");
            self.tags.extend(slice_tags(bytes));
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn nal(nal_type: u8, payload: &[u8]) -> Vec<u8> {
    let mut unit = vec![0, 0, 0, 1, nal_type << 1, 1];
    unit.extend_from_slice(payload);
    unit
}

// Each picture is one slice whose only payload byte after the slice flag is `tag`.
fn picture(nal_type: u8, tag: u8) -> Vec<u8> {
    nal(nal_type, &[FIRST_SLICE_IN_PICTURE, tag])
}

fn slice_tags(bytes: &[u8]) -> Vec<u8> {
    bytes
        .windows(7)
        .filter(|w| w[..3] == [0, 0, 1] && (w[3] >> 1) <= 31 && w[5] == FIRST_SLICE_IN_PICTURE)
        .map(|w| w[6])
        .collect()
}

async fn parse(stream: Vec<u8>) -> Vec<u8> {
    let caps = Caps::CompressedVideo {
        codec: VideoCodec::H265,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    };
    let mut parser = H265Parse::reframing();
    parser.configure_pipeline(&caps).unwrap();
    let mut sink = PictureTags::default();
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(stream.into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    parser
        .process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .unwrap();
    parser
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    sink.tags
}

fn parameter_sets() -> Vec<u8> {
    [nal(VPS, &[1]), nal(SPS, &[1]), nal(PPS, &[1])].concat()
}

#[tokio::test]
async fn drops_rasl_of_the_first_cra_and_keeps_rasl_of_later_cras() {
    let stream = [
        parameter_sets(),
        picture(CRA, 1),
        picture(RASL_R, 2),
        picture(RASL_N, 3),
        picture(TRAIL_R, 4),
        picture(CRA, 5),
        picture(RASL_R, 6),
        picture(RASL_N, 7),
        picture(TRAIL_R, 8),
    ]
    .concat();
    assert_eq!(parse(stream).await, [1, 4, 5, 6, 7, 8]);
}

#[tokio::test]
async fn drops_rasl_of_a_cra_after_an_end_of_sequence_nal() {
    let stream = [
        parameter_sets(),
        picture(CRA, 1),
        picture(RASL_N, 2),
        picture(TRAIL_R, 3),
        nal(END_OF_SEQUENCE, &[]),
        picture(CRA, 4),
        picture(RASL_N, 5),
        picture(TRAIL_R, 6),
    ]
    .concat();
    assert_eq!(parse(stream).await, [1, 3, 4, 6]);
}
