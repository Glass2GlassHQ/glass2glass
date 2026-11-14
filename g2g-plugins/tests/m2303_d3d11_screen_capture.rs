// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "windows", feature = "d3d11-screen-capture"))]

use g2g_core::frame::PipelinePacket;
use g2g_core::memory::MemoryDomain;
use g2g_core::runtime::SourceLoop;
use g2g_core::{Caps, Dim, G2gError, OutputSink, PushOutcome, RawVideoFormat};
use g2g_plugins::d3d11screencapturesrc::{monitor_bounds, D3D11ScreenCaptureSrc};

const FRAMES: u64 = 3;
const BGRA_BYTES_PER_PIXEL: usize = 4;
const PRIMARY_MONITOR: i32 = -1;

#[derive(Default)]
struct Collect {
    frames: Vec<MemoryDomain>,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        if let PipelinePacket::DataFrame(frame) = packet {
            self.frames.push(frame.domain);
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

// desktop duplication needs an interactive desktop, which an ssh session or a CI service lacks
#[tokio::test(flavor = "current_thread")]
async fn captures_the_primary_monitor_or_reports_no_desktop() {
    let mut source = D3D11ScreenCaptureSrc::new().with_frame_limit(FRAMES);
    let caps = match source.intercept_caps().await {
        Err(G2gError::DeviceNotFound) => {
            eprintln!("skipping screen capture: no desktop monitor");
            return;
        }
        other => other.expect("caps"),
    };
    let monitor = monitor_bounds(PRIMARY_MONITOR).expect("the probe found the monitor");
    let Caps::RawVideo {
        format: RawVideoFormat::Bgra8,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        ..
    } = caps
    else {
        panic!("monitor caps must be fixed BGRA, got {caps:?}");
    };
    assert_eq!((width, height), (monitor.width, monitor.height));
    source.configure_pipeline(&caps).expect("configure");

    let mut out = Collect::default();
    match source.run(&mut out).await {
        Ok(captured) => {
            assert_eq!(captured, FRAMES);
            assert_eq!(out.frames.len() as u64, FRAMES);
            for frame in &out.frames {
                let bytes = frame.as_system_slice().expect("system memory frames");
                assert_eq!(
                    bytes.len(),
                    width as usize * height as usize * BGRA_BYTES_PER_PIXEL
                );
            }
        }
        Err(G2gError::Hardware(e)) => {
            eprintln!("skipping screen capture: duplication refused, {e:?}");
        }
        Err(e) => panic!("screen capture failed: {e:?}"),
    }
}
