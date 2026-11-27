// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(feature = "std", feature = "metadata"))]

#[path = "common/collect_sink.rs"]
mod collect_sink;
use collect_sink::{feed_in_chunks, CollectSink};

use g2g_core::{
    AspectRatio, AsyncElement, BitstreamAlignment, Caps, Colorimetry, Dim, Frame, FrameTiming,
    Interlace, PipelinePacket, PixelAspect, PropValue, Rate, RawVideoFormat, VideoCodec,
};
use g2g_plugins::afd::{AfdBarMeta, BarData};
use g2g_plugins::h264parse::H264Parse;
use g2g_plugins::h265parse::H265Parse;
use g2g_plugins::sei::{build_sei_nal, PAYLOAD_USER_DATA_REGISTERED};
use g2g_plugins::videocrop::VideoCrop;

const AFD_4_3: &[u8] = b"\xb5\x00\x31DTG1\x41\xf9";
const AFD_WIDE: &[u8] = b"\xb5\x00\x31DTG1\x41\xf4";
// SMPTE ST 2016-1, Annex B.1. The encoded rows correspond to y=93 through 625.
const LETTERBOX: &[u8] = b"\xb5\x00\x31GA94\x06\xcf\xc0\x76\xc2\x8c";
// The active columns are 32 through 271 in a 304-pixel raster.
#[cfg(feature = "ffmpeg")]
const PILLARBOX: &[u8] = b"\xb5\x00\x31GA94\x06\x3f\xc0\x1f\xc1\x10";

const H264_FIXTURE: &[u8] = include_bytes!("fixtures/h264_304x180_crop.h264");
const H265_FIXTURE: &[u8] = include_bytes!("fixtures/h265_304x180_main10_crop.hevc");

fn coded_caps(codec: VideoCodec, alignment: BitstreamAlignment) -> Caps {
    Caps::CompressedVideo {
        codec,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        colorimetry: Colorimetry::UNKNOWN,
        pixel_aspect: PixelAspect::Any,
        alignment,
    }
}

fn sei(codec: VideoCodec, payloads: &[&[u8]]) -> Vec<u8> {
    payloads
        .iter()
        .flat_map(|payload| build_sei_nal(PAYLOAD_USER_DATA_REGISTERED, payload, codec))
        .collect()
}

async fn parsed(
    mut parser: impl AsyncElement,
    codec: VideoCodec,
    fixture: &[u8],
    payloads: &[&[u8]],
) -> Vec<PipelinePacket> {
    parser
        .configure_pipeline(&coded_caps(codec, BitstreamAlignment::Unspecified))
        .unwrap();
    let mut stream = sei(codec, payloads);
    stream.extend_from_slice(fixture);
    let mut sink = CollectSink::default();
    feed_in_chunks(&mut parser, &stream, &mut sink).await;
    sink.packets
}

#[tokio::test]
async fn both_parsers_attach_afd_and_bars_to_their_access_unit() {
    let broken_bar = b"\xb5\x00\x31GA94\x06\xcf\xc0";
    let payloads = &[AFD_WIDE, LETTERBOX, broken_bar.as_slice()];
    let h264 = parsed(
        H264Parse::reframing(),
        VideoCodec::H264,
        H264_FIXTURE,
        payloads,
    )
    .await;
    let h265 = parsed(
        H265Parse::reframing(),
        VideoCodec::H265,
        H265_FIXTURE,
        payloads,
    )
    .await;
    for packets in [h264, h265] {
        let mut frames = packets.iter().filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(frame),
            _ => None,
        });
        let first = frames.next().expect("first access unit");
        let entries = &first
            .meta
            .get::<AfdBarMeta>()
            .expect("AFD and bars")
            .entries;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].active_format, Some(4));
        assert_eq!(entries[0].widescreen, None);
        assert_eq!(
            entries[0].bars,
            Some(BarData::TopBottom {
                top_end: 118,
                bottom_start: 652
            }),
        );
        assert!(frames.next().unwrap().meta.get::<AfdBarMeta>().is_none());
    }
}

fn active_crop() -> VideoCrop {
    let mut crop = VideoCrop::new(0, 0, 0, 0);
    crop.set_property("use-afd", PropValue::Bool(true)).unwrap();
    crop
}

#[tokio::test]
async fn crop_uses_afd_or_bar_data_and_announces_each_new_geometry_once() {
    let mut parser = H264Parse::new();
    parser
        .configure_pipeline(&coded_caps(
            VideoCodec::H264,
            BitstreamAlignment::AccessUnit,
        ))
        .unwrap();
    let mut crop = active_crop();
    let pixel_aspect = PixelAspect::Exact(AspectRatio::new(3, 2).unwrap());
    let raw_caps = |width, height| Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate: Rate::Any,
        interlace: Interlace::Progressive,
        colorimetry: Colorimetry::UNKNOWN,
        pixel_aspect,
    };
    let (width, height) = (1280, 720);
    crop.configure_pipeline(&raw_caps(width, height)).unwrap();

    let mut raw = CollectSink::default();
    for (index, payloads) in [
        &[AFD_WIDE, LETTERBOX][..],
        &[AFD_WIDE, LETTERBOX],
        &[AFD_4_3],
        &[],
    ]
    .into_iter()
    .enumerate()
    {
        let unit = Frame::from_system_bytes(
            sei(VideoCodec::H264, payloads),
            FrameTiming::default(),
            index as u64,
        );
        let mut coded = CollectSink::default();
        parser
            .process(PipelinePacket::DataFrame(unit), &mut coded)
            .await
            .unwrap();
        let Some(PipelinePacket::DataFrame(parsed)) = coded.packets.pop() else {
            panic!("parser passed the access unit");
        };
        // Mark the top active row and column so a wrong crop origin is visible.
        let mut pixels = vec![0; (width * height * 4) as usize];
        pixels[(93 * width * 4) as usize] = 17;
        pixels[320 * 4] = 21;
        let mut frame = Frame::from_system_bytes(pixels, parsed.timing, parsed.sequence);
        frame.meta = parsed.meta;
        crop.process(PipelinePacket::DataFrame(frame), &mut raw)
            .await
            .unwrap();
    }
    let [PipelinePacket::CapsChanged(letterboxed), PipelinePacket::DataFrame(letterbox), PipelinePacket::DataFrame(repeated), PipelinePacket::CapsChanged(pillarboxed), PipelinePacket::DataFrame(pillarbox), PipelinePacket::CapsChanged(full_raster), PipelinePacket::DataFrame(_)] =
        raw.packets.as_slice()
    else {
        panic!("each new geometry precedes the first affected frame");
    };
    assert_eq!(*letterboxed, raw_caps(1280, 533));
    assert_eq!(*pillarboxed, raw_caps(640, 720));
    assert_eq!(*full_raster, raw_caps(1280, 720));
    assert_eq!(letterbox.domain.as_system_slice().unwrap()[0], 17);
    assert_eq!(repeated.domain.as_system_slice().unwrap()[0], 17);
    assert_eq!(pillarbox.domain.as_system_slice().unwrap()[0], 21);
}

#[cfg(feature = "ffmpeg")]
#[tokio::test]
async fn libavcodec_decodes_both_codecs_with_their_pillarbox_metadata() {
    use g2g_plugins::ffmpegdec::{FfmpegVideoDec, OutputFormat};
    let h264 = parsed(
        H264Parse::reframing(),
        VideoCodec::H264,
        H264_FIXTURE,
        &[PILLARBOX],
    )
    .await;
    let h265 = parsed(
        H265Parse::reframing(),
        VideoCodec::H265,
        H265_FIXTURE,
        &[PILLARBOX],
    )
    .await;
    for (codec, packets) in [(VideoCodec::H264, h264), (VideoCodec::H265, h265)] {
        let mut dec = FfmpegVideoDec::new().with_output_format(OutputFormat::Auto);
        dec.configure_pipeline(&coded_caps(codec, BitstreamAlignment::AccessUnit))
            .unwrap();
        let mut decoded = CollectSink::default();
        for (index, packet) in packets.into_iter().enumerate() {
            dec.process(packet, &mut decoded)
                .await
                .unwrap_or_else(|error| panic!("{codec:?} packet {index}: {error:?}"));
        }
        dec.process(PipelinePacket::Eos, &mut decoded)
            .await
            .unwrap();
        let mut crop = active_crop();
        let mut video = CollectSink::default();
        for packet in decoded.packets {
            if let PipelinePacket::CapsChanged(caps) = &packet {
                crop.configure_pipeline(caps).unwrap();
            } else {
                crop.process(packet, &mut video).await.unwrap();
            }
        }
        let geometries: Vec<_> = video
            .packets
            .iter()
            .filter_map(|packet| match packet {
                PipelinePacket::CapsChanged(Caps::RawVideo {
                    width: Dim::Fixed(w),
                    height: Dim::Fixed(h),
                    ..
                }) => Some((*w, *h)),
                _ => None,
            })
            .collect();
        assert_eq!(geometries, [(240, 180), (304, 180)], "{codec:?}");
    }
}
