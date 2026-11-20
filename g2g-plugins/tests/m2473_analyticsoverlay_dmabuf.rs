// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! M2473: `analyticsoverlay` draws into a dma-buf frame in place, so a capture
//! source exporting its buffers reaches `waylandsink` without a copy. The
//! pixels must match what the same detections draw onto a system-memory frame,
//! with tight rows and with padded ones.
//!
//! The buffer is a real dma-buf, a sealed memfd exported through
//! `/dev/udmabuf`. Self-skips where that device is not available (the CI
//! runners).

#![cfg(all(target_os = "linux", feature = "analytics", feature = "std"))]

#[path = "common/udmabuf.rs"]
mod udmabuf;

use core::future::Future;
use core::pin::Pin;

use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AnalyticsMeta, AsyncElement, BBox, Caps, Dim, Frame, FrameTiming, G2gError, ObjectDetection,
    OutputSink, PipelinePacket, PushOutcome, Rate, RawVideoFormat,
};
use g2g_plugins::analyticsoverlay::AnalyticsOverlay;
use g2g_plugins::dmabufmap::DmaBufReadMap;
use udmabuf::udmabuf_of;

const WIDTH: usize = 64;
const HEIGHT: usize = 64;
const PADDED_STRIDE: usize = 128;
const PADDING_BYTE: u8 = 0xEE;
// udmabuf exports whole pages only
const PAGE: usize = 4096;

#[derive(Default)]
struct LastFrame(Option<Frame>);

impl OutputSink for LastFrame {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::DataFrame(frame)) = packet_slot.take() {
            self.0 = Some(frame);
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn nv12_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(WIDTH as u32),
        height: Dim::Fixed(HEIGHT as u32),
        framerate: Rate::Any,
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
    }
}

// no run of equal bytes, so a write at the wrong offset cannot match by accident
fn tight_nv12() -> Vec<u8> {
    (0..WIDTH * HEIGHT * 3 / 2)
        .map(|i| (i % 251) as u8)
        .collect()
}

fn round_up_to_page(len: usize) -> usize {
    len.div_ceil(PAGE) * PAGE
}

// the tight frame's rows spread `PADDED_STRIDE` apart, luma then interleaved chroma
fn padded(tight: &[u8]) -> Vec<u8> {
    let rows = HEIGHT * 3 / 2;
    let mut out = vec![PADDING_BYTE; round_up_to_page(rows * PADDED_STRIDE)];
    for (row, pixels) in tight.as_chunks::<WIDTH>().0.iter().enumerate() {
        out[row * PADDED_STRIDE..][..WIDTH].copy_from_slice(pixels);
    }
    out
}

async fn draw(frame: Frame) -> Frame {
    let mut overlay = AnalyticsOverlay::new().with_thickness(2);
    overlay.configure_pipeline(&nv12_caps()).expect("nv12");
    let mut meta = AnalyticsMeta::new();
    meta.add_detection(ObjectDetection {
        bbox: BBox {
            x: 0.25,
            y: 0.25,
            w: 0.5,
            h: 0.5,
        },
        label: 0,
        confidence: 0.9,
    });
    let mut frame = frame;
    frame.meta.attach(meta);
    let mut out = LastFrame::default();
    let process: Pin<Box<dyn Future<Output = Result<(), G2gError>>>> =
        overlay.process(PipelinePacket::DataFrame(frame), &mut out);
    process.await.expect("overlay draws");
    out.0.expect("frame forwarded")
}

async fn drawn_on_system_memory() -> Vec<u8> {
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(tight_nv12().into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    let drawn = draw(frame).await;
    let reference = drawn
        .domain
        .as_system_slice()
        .expect("system in, system out")
        .to_vec();
    assert_ne!(reference, tight_nv12(), "the box must change the frame");
    reference
}

fn dmabuf_bytes(frame: &Frame) -> Vec<u8> {
    let MemoryDomain::DmaBuf(dmabuf) = &frame.domain else {
        panic!("a dma-buf frame must leave the overlay as a dma-buf");
    };
    DmaBufReadMap::read(dmabuf)
        .expect("map")
        .from_fd_start()
        .to_vec()
}

#[tokio::test]
async fn a_tight_dmabuf_frame_gets_the_system_memory_pixels() {
    let mut bytes = tight_nv12();
    bytes.resize(round_up_to_page(bytes.len()), 0);
    let Some(dmabuf) = udmabuf_of(&bytes) else {
        eprintln!("SKIP: /dev/udmabuf is not available, so no dma-buf to draw into");
        return;
    };
    let reference = drawn_on_system_memory().await;

    let drawn = draw(Frame::new(
        MemoryDomain::DmaBuf(dmabuf),
        FrameTiming::default(),
        0,
    ))
    .await;

    assert_eq!(&dmabuf_bytes(&drawn)[..reference.len()], &reference[..]);
}

#[tokio::test]
async fn a_padded_dmabuf_frame_gets_the_system_memory_pixels_and_keeps_its_padding() {
    let Some(mut dmabuf) = udmabuf_of(&padded(&tight_nv12())) else {
        eprintln!("SKIP: /dev/udmabuf is not available, so no dma-buf to draw into");
        return;
    };
    dmabuf.stride = PADDED_STRIDE as u32;
    let reference = drawn_on_system_memory().await;

    let drawn = draw(Frame::new(
        MemoryDomain::DmaBuf(dmabuf),
        FrameTiming::default(),
        0,
    ))
    .await;

    assert_eq!(dmabuf_bytes(&drawn), padded(&reference));
}
