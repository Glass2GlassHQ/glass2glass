//! M1550: an FLV VP6 tag's dimension adjustment byte rides at the front of the
//! unit `flvdemux` emits, and `flvmux` writes it back, so a remux is byte-exact.

#[path = "common/collect_sink.rs"]
mod collect_sink;
use collect_sink::CollectSink;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{AsyncElement, ByteStreamEncoding, Caps, Dim, Rate, VideoCodec};
use g2g_plugins::flvdemux::{FlvDemux, FlvStream};
use g2g_plugins::flvmux::FlvMux;

const TAG_VIDEO: u8 = 9;
const KEYFRAME: u8 = 1 << 4;
// 300x180 coded as 304x192: four columns and twelve rows of adjustment.
const DIMENSION_BYTE: u8 = 0x4C;

// A video-only FLV holding one tag with `body`, laid out the way `flvmux` writes it.
fn single_tag_flv(body: &[u8]) -> Vec<u8> {
    let mut flv = b"FLV\x01\x01\x00\x00\x00\x09".to_vec();
    flv.extend_from_slice(&0u32.to_be_bytes());
    flv.push(TAG_VIDEO);
    flv.extend_from_slice(&(body.len() as u32).to_be_bytes()[1..]);
    flv.extend_from_slice(&[0; 7]);
    flv.extend_from_slice(body);
    flv
}

fn system_frame(bytes: Vec<u8>, timing: FrameTiming) -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing,
        0,
    )
}

fn data_frames(sink: CollectSink) -> Vec<Frame> {
    sink.packets
        .into_iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(frame),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn vp6_dimension_byte_survives_demux_and_remux() {
    for (alpha, stream, codec_id, frame_data) in [
        (false, FlvStream::Vp6, 4u8, vec![0xAA, 0xBB, 0xCC]),
        // VP6-alpha's 24-bit offset to the alpha plane follows the dimension byte.
        (
            true,
            FlvStream::Vp6Alpha,
            5,
            vec![0x00, 0x00, 0x01, 0xAA, 0xBB],
        ),
    ] {
        let mut unit = vec![DIMENSION_BYTE];
        unit.extend_from_slice(&frame_data);
        let mut tag_body = vec![KEYFRAME | codec_id];
        tag_body.extend_from_slice(&unit);
        let flv = single_tag_flv(&tag_body);

        let mut demux = FlvDemux::new().with_stream(stream);
        demux
            .configure_pipeline(&Caps::ByteStream {
                encoding: ByteStreamEncoding::Flv,
            })
            .unwrap();
        let mut demuxed = CollectSink::default();
        demux
            .process(
                PipelinePacket::DataFrame(system_frame(flv.clone(), FrameTiming::default())),
                &mut demuxed,
            )
            .await
            .unwrap();
        let units = data_frames(demuxed);
        assert_eq!(units.len(), 1, "{stream:?}");
        assert_eq!(
            units[0].domain.as_system_slice().unwrap(),
            &unit[..],
            "{stream:?} unit starts with the dimension byte"
        );

        let mut mux = FlvMux::new();
        mux.configure_pipeline(&Caps::CompressedVideo {
            codec: VideoCodec::Vp6 { alpha },
            width: Dim::Fixed(300),
            height: Dim::Fixed(180),
            framerate: Rate::Fixed(25 << 16),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        })
        .unwrap();
        let mut muxed = CollectSink::default();
        let demuxed_unit = units[0].domain.as_system_slice().unwrap().to_vec();
        mux.process(
            PipelinePacket::DataFrame(system_frame(demuxed_unit, units[0].timing)),
            &mut muxed,
        )
        .await
        .unwrap();
        let remuxed: Vec<u8> = data_frames(muxed)
            .iter()
            .flat_map(|frame| frame.domain.as_system_slice().unwrap().to_vec())
            .collect();
        assert_eq!(remuxed, flv, "{stream:?} remux is byte-exact");
    }
}
