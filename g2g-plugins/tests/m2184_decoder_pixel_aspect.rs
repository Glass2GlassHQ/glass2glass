// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! M2184: decoders hand the sample aspect the parser read from the H.264 VUI on
//! to their raw output caps. The oracle is a stream ffmpeg encoded with `setsar`,
//! so the ratio sits in the bitstream the parser reads. Self-skips where ffmpeg
//! or libx264 is absent.
#![cfg(any(feature = "ffmpeg", feature = "openh264"))]

use std::process::Command;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AspectRatio, AsyncElement, Caps, Dim, G2gError, OutputSink, PixelAspect, PushOutcome, Rate,
    VideoCodec,
};
use g2g_plugins::h264parse::H264Parse;

const SAMPLE_ASPECT: (u32, u32) = (16, 15);

#[derive(Default)]
struct CollectPackets {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for CollectPackets {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        self.packets.push(packet);
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: PixelAspect::Any,
        alignment: g2g_core::BitstreamAlignment::Unspecified,
    }
}

fn expected_pixel_aspect() -> PixelAspect {
    let (width, height) = SAMPLE_ASPECT;
    PixelAspect::Exact(AspectRatio::new(width, height).expect("nonzero"))
}

// an Annex-B H.264 stream whose VUI declares SAMPLE_ASPECT, `None` without ffmpeg or libx264
fn encode_with_sample_aspect() -> Option<Vec<u8>> {
    let (width, height) = SAMPLE_ASPECT;
    let output = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=0.2:size=320x240:rate=25",
            "-vf",
            &format!("setsar={width}/{height}"),
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-bf",
            "0",
            "-f",
            "h264",
            "-",
        ])
        .output()
        .ok()?;
    output.status.success().then_some(output.stdout)
}

// the parser's output for the whole stream: its CapsChanged, then one access unit per frame
async fn parsed(stream: Vec<u8>) -> Vec<PipelinePacket> {
    let mut parse = H264Parse::reframing();
    parse.configure_pipeline(&h264_caps()).unwrap();
    let mut sink = CollectPackets::default();
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(stream.into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    parse
        .process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .unwrap();
    parse.process(PipelinePacket::Eos, &mut sink).await.unwrap();
    sink.packets
}

async fn decoded_pixel_aspect<E: AsyncElement>(decoder: &mut E) -> Option<PixelAspect> {
    let stream = encode_with_sample_aspect()?;
    decoder.configure_pipeline(&h264_caps()).unwrap();
    let mut sink = CollectPackets::default();
    for packet in parsed(stream).await {
        decoder.process(packet, &mut sink).await.unwrap();
    }
    let raw_pixel_aspect = sink.packets.iter().rev().find_map(|packet| match packet {
        PipelinePacket::CapsChanged(Caps::RawVideo { pixel_aspect, .. }) => Some(*pixel_aspect),
        _ => None,
    });
    Some(raw_pixel_aspect.expect("the decoder announced raw caps"))
}

#[cfg(feature = "ffmpeg")]
#[tokio::test]
async fn ffmpegdec_carries_the_vui_sample_aspect() {
    let mut decoder = g2g_plugins::ffmpegdec::FfmpegVideoDec::new();
    let Some(pixel_aspect) = decoded_pixel_aspect(&mut decoder).await else {
        eprintln!("ffmpeg with libx264 not present; skipping");
        return;
    };
    assert_eq!(pixel_aspect, expected_pixel_aspect());
}

#[cfg(feature = "openh264")]
#[tokio::test]
async fn openh264dec_carries_the_vui_sample_aspect() {
    let mut decoder = g2g_plugins::openh264dec::OpenH264Dec::new();
    let Some(pixel_aspect) = decoded_pixel_aspect(&mut decoder).await else {
        eprintln!("ffmpeg with libx264 not present; skipping");
        return;
    };
    assert_eq!(pixel_aspect, expected_pixel_aspect());
}
