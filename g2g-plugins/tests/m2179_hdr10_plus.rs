//! M2179: HDR10+ dynamic metadata. `h265parse` attaches each access unit's
//! SMPTE ST 2094-40 SEI as `Hdr10PlusMeta`, and `h265seiinserter` writes it back.
//! hdr10plus_tool is the reference writer: inserting the parsed metadata into the
//! stream before injection must give the stream it injected. libaom re-serializes
//! the same metadata into AV1 metadata OBUs, which `av1parse` must read to the
//! same bytes.
//!
//! ```text
//! ffmpeg -f lavfi -i testsrc=size=160x96:rate=30 -frames:v 4 -pix_fmt yuv420p10le \
//!   -c:v libx265 -x265-params log-level=error:bframes=0:info=0 -f hevc \
//!   tests/fixtures/hevc_hdr10plus_base.hevc
//! hdr10plus_tool inject -i tests/fixtures/hevc_hdr10plus_base.hevc \
//!   -j tests/fixtures/hevc_hdr10plus.json -o tests/fixtures/hevc_hdr10plus.hevc
//! ffmpeg -i tests/fixtures/hevc_hdr10plus.hevc -c:v libaom-av1 -cpu-used 8 -crf 50 \
//!   -g 999 -f obu tests/fixtures/av1_hdr10plus.obu
//! ```
#![cfg(all(feature = "std", feature = "metadata"))]

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::meta::Hdr10PlusMeta;
use g2g_core::{AsyncElement, Caps, Dim, G2gError, OutputSink, PushOutcome, Rate, VideoCodec};
use g2g_plugins::av1parse::Av1Parse;
use g2g_plugins::h265parse::H265Parse;
use g2g_plugins::seiinsert::H265SeiInserter;

mod ffprobe;
use ffprobe::fixture_path;

const BASE_FIXTURE: &str = "hevc_hdr10plus_base.hevc";
const INJECTED_FIXTURE: &str = "hevc_hdr10plus.hevc";
const METADATA_FIXTURE: &str = "hevc_hdr10plus.json";
const AV1_FIXTURE: &str = "av1_hdr10plus.obu";

#[derive(Default)]
struct CaptureSink {
    frames: Vec<Frame>,
}

impl OutputSink for CaptureSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::DataFrame(frame)) = packet_slot.take() {
            self.frames.push(frame);
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn compressed_caps(codec: VideoCodec) -> Caps {
    Caps::CompressedVideo {
        codec,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        alignment: g2g_core::BitstreamAlignment::Unspecified,
    }
}

fn system_frame(bytes: Vec<u8>) -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming::default(),
        0,
    )
}

fn bytes(frame: &Frame) -> &[u8] {
    frame.domain.as_system_slice().expect("system memory")
}

// hdr10plus_tool rewrites the slice start code after its SEI to four bytes, emulation prevention keeps 00 00 00 out of NAL payloads
fn three_byte_start_codes(access_unit: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(access_unit.len());
    for (index, &byte) in access_unit.iter().enumerate() {
        let leading_zero = access_unit[index..].starts_with(&[0, 0, 0, 1]);
        if !leading_zero {
            out.push(byte);
        }
    }
    out
}

async fn parsed<E: AsyncElement>(mut parse: E, codec: VideoCodec, fixture: &str) -> Vec<Frame> {
    let stream = std::fs::read(fixture_path(fixture)).expect("the fixture is checked in");
    parse
        .configure_pipeline(&compressed_caps(codec))
        .expect("configure");
    let mut sink = CaptureSink::default();
    parse
        .process(PipelinePacket::DataFrame(system_frame(stream)), &mut sink)
        .await
        .expect("parse");
    parse
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");
    sink.frames
}

async fn access_units(fixture: &str) -> Vec<Frame> {
    parsed(H265Parse::reframing(), VideoCodec::H265, fixture).await
}

fn hdr10_plus(frame: &Frame) -> &Hdr10PlusMeta {
    frame.meta.get::<Hdr10PlusMeta>().expect("HDR10+ meta")
}

// hdr10plus_tool writes one SceneInfo entry per frame
fn injected_frame_count() -> usize {
    std::fs::read_to_string(fixture_path(METADATA_FIXTURE))
        .expect("the metadata is checked in")
        .matches("\"SequenceFrameIndex\"")
        .count()
}

async fn inserted(frame: Frame) -> Frame {
    let mut insert = H265SeiInserter::new();
    insert
        .configure_pipeline(&compressed_caps(VideoCodec::H265))
        .expect("configure");
    let mut sink = CaptureSink::default();
    insert
        .process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .expect("insert");
    sink.frames.pop().expect("one frame out")
}

#[tokio::test]
async fn each_injected_access_unit_carries_hdr10_plus_meta() {
    let injected = access_units(INJECTED_FIXTURE).await;
    let base = access_units(BASE_FIXTURE).await;
    assert_eq!(injected.len(), injected_frame_count());
    assert_eq!(base.len(), injected.len());
    for (index, frame) in injected.iter().enumerate() {
        assert!(
            frame.meta.get::<Hdr10PlusMeta>().is_some(),
            "frame {index} carries HDR10+"
        );
    }
    assert!(base
        .iter()
        .all(|frame| frame.meta.get::<Hdr10PlusMeta>().is_none()));
}

#[tokio::test]
async fn the_inserter_writes_the_metadata_back_as_hdr10plus_tool_injected_it() {
    let injected = access_units(INJECTED_FIXTURE).await;
    let base = access_units(BASE_FIXTURE).await;
    for (index, (base, injected)) in base.into_iter().zip(&injected).enumerate() {
        let mut frame = base;
        frame.meta.attach(hdr10_plus(injected).clone());
        let written = inserted(frame).await;
        assert_eq!(
            three_byte_start_codes(bytes(&written)),
            three_byte_start_codes(bytes(injected)),
            "frame {index}"
        );
    }
}

#[tokio::test]
async fn an_access_unit_that_already_carries_hdr10_plus_is_left_alone() {
    let injected = access_units(INJECTED_FIXTURE).await;
    let first = injected.into_iter().next().expect("a frame");
    let before = bytes(&first).to_vec();
    let written = inserted(first).await;
    assert_eq!(bytes(&written), before);
}

#[tokio::test]
async fn av1parse_reads_the_same_metadata_libaom_wrote() {
    let hevc = access_units(INJECTED_FIXTURE).await;
    let av1 = parsed(Av1Parse::reframing(), VideoCodec::Av1, AV1_FIXTURE).await;
    assert_eq!(av1.len(), hevc.len());
    for (index, (av1, hevc)) in av1.iter().zip(&hevc).enumerate() {
        assert_eq!(hdr10_plus(av1), hdr10_plus(hevc), "frame {index}");
    }
}

#[tokio::test]
async fn an_access_unit_without_the_sei_carries_no_hdr10_plus() {
    let injected = access_units(INJECTED_FIXTURE).await;
    let base = access_units(BASE_FIXTURE).await;
    let mut parse = H265Parse::new();
    parse
        .configure_pipeline(&compressed_caps(VideoCodec::H265))
        .expect("configure");
    let mut sink = CaptureSink::default();
    for frame in [&injected[0], &base[1]] {
        let unit = system_frame(bytes(frame).to_vec());
        parse
            .process(PipelinePacket::DataFrame(unit), &mut sink)
            .await
            .expect("parse");
    }
    let carried: Vec<bool> = sink
        .frames
        .iter()
        .map(|frame| frame.meta.get::<Hdr10PlusMeta>().is_some())
        .collect();
    assert_eq!(carried, [true, false]);
}
