// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![allow(dead_code)] // each test file uses a different part of this

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    ByteStreamEncoding, Caps, Dim, G2gError, MultiInputElement, OutputSink, PushOutcome, Rate,
    VideoCodec,
};
use g2g_plugins::mkvmuxn::MkvMuxN;
use g2g_plugins::mp4muxn::Mp4MuxN;
use g2g_plugins::tsmuxn::TsMux;

pub(crate) const WIDTH: u32 = 320;
pub(crate) const HEIGHT: u32 = 240;
/// One frame at 25 fps, a whole number of the muxers' 90 kHz ticks, so the
/// synthetic fixture's segment durations are exact.
pub(crate) const FRAME_25FPS_NS: u64 = 40_000_000;
/// Frames per GOP in the synthetic fixture: exactly 1 s, the target duration the
/// segmenting tests use.
pub(crate) const GOP: usize = 25;

/// Captures the byte-stream frames an element pushes downstream, timing and all,
/// which is exactly what the runner would hand the next element.
#[derive(Default)]
pub(crate) struct CaptureSink {
    pub(crate) frames: Vec<(Vec<u8>, FrameTiming)>,
}

impl OutputSink for CaptureSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.frames.push((s.to_vec(), f.timing));
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}
pub(crate) struct NullSink;
impl OutputSink for NullSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        packet_slot.take();
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

pub(crate) fn frame(bytes: Vec<u8>, timing: FrameTiming) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing,
        0,
    ))
}

pub(crate) fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(25 << 16),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
        alignment: g2g_core::BitstreamAlignment::Unspecified,
    }
}

pub(crate) fn bytestream(encoding: ByteStreamEncoding) -> Caps {
    Caps::ByteStream { encoding }
}

pub(crate) fn annexb(nals: &[&[u8]]) -> Vec<u8> {
    let mut v = Vec::new();
    for n in nals {
        v.extend_from_slice(&[0, 0, 0, 1]);
        v.extend_from_slice(n);
    }
    v
}

/// `groups` GOPs of [`GOP`] access units at 25 fps, each opening on an IDR. The
/// muxers never look inside a slice, so synthetic NALs suffice for the
/// segmenting tests; the ffmpeg oracle uses a real encode instead.
pub(crate) fn synthetic_aus(groups: usize) -> Vec<(Vec<u8>, FrameTiming)> {
    let sps: &[u8] = &[0x67, 0x42, 0x00, 0x1e, 0x88];
    let pps: &[u8] = &[0x68, 0xce, 0x3c, 0x80];
    let idr: &[u8] = &[0x65, 0x88, 0x84, 0x00];
    let inter: &[u8] = &[0x41, 0x9a, 0x00];
    let mut aus = Vec::new();
    for g in 0..groups {
        for i in 0..GOP {
            let index = (g * GOP + i) as u64;
            let bytes = if i == 0 {
                annexb(&[sps, pps, idr])
            } else {
                annexb(&[inter])
            };
            aus.push((
                bytes,
                FrameTiming {
                    pts_ns: index * FRAME_25FPS_NS,
                    dts_ns: index * FRAME_25FPS_NS,
                    duration_ns: FRAME_25FPS_NS,
                    keyframe: i == 0,
                    ..FrameTiming::default()
                },
            ));
        }
    }
    aus
}

async fn mux_one_track(
    mut mux: impl MultiInputElement,
    aus: &[(Vec<u8>, FrameTiming)],
) -> Vec<(Vec<u8>, FrameTiming)> {
    mux.configure_pipeline(0, &h264_caps()).expect("configure");
    let mut sink = CaptureSink::default();
    for (au, timing) in aus {
        mux.process(0, frame(au.clone(), *timing), &mut sink)
            .await
            .expect("mux");
    }
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .expect("mux eos");
    sink.frames
}

/// Access units through a one-pad `TsMux`, as the frames it pushes downstream.
pub(crate) async fn mux_ts(aus: &[(Vec<u8>, FrameTiming)]) -> Vec<(Vec<u8>, FrameTiming)> {
    mux_one_track(TsMux::new(1), aus).await
}

/// Access units through `Mp4MuxN` batched into `fragment_ms` fragments.
pub(crate) async fn mux_fmp4(
    aus: &[(Vec<u8>, FrameTiming)],
    fragment_ms: u64,
) -> Vec<(Vec<u8>, FrameTiming)> {
    mux_one_track(Mp4MuxN::new(1).with_fragment_duration_ms(fragment_ms), aus).await
}

/// Access units through a streamable `MkvMuxN`, one output frame per access unit.
pub(crate) async fn mux_live_mkv(aus: &[(Vec<u8>, FrameTiming)]) -> Vec<(Vec<u8>, FrameTiming)> {
    mux_one_track(MkvMuxN::new(1).with_streamable(true), aus).await
}
