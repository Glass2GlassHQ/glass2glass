// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(feature = "std", feature = "rtsp", feature = "rtsp-server"))]

use core::task::{Context, Poll};
use core::time::Duration;
use std::net::TcpListener as StdTcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use g2g_core::clock::ClockSync;
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, MonotonicClock, MultiOutputSink,
    MultiOutputSource, PipelineClock, PipelinePacket, PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::rtspserversink::RtspServerSink;
use g2g_plugins::rtspsrcn::RtspSrcN;

#[path = "common/annexb.rs"]
mod annexb;
#[path = "common/collect_sink.rs"]
mod collect_sink;

use annexb::access_units;
use collect_sink::CollectSink;

const FIXTURE: &[u8] = include_bytes!("fixtures/h264_64x48_no_vui_timing.h264");
const FRAME_PERIOD: Duration = Duration::from_millis(33);
const FRAMES: u64 = 10;
const RUN_LIMIT: Duration = Duration::from_secs(30);
// how long the pipeline has been playing when the player connects
const PIPELINE_AGE: Duration = Duration::from_secs(5);

// a pipeline clock whose base time of zero lies PIPELINE_AGE before the process started
#[derive(Debug)]
struct AgedClock;

impl PipelineClock for AgedClock {
    fn now_ns(&self) -> u64 {
        MonotonicClock.now_ns() + PIPELINE_AGE.as_nanos() as u64
    }
}

#[derive(Default)]
struct PtsSink {
    pts_ns: Vec<u64>,
}

impl MultiOutputSink for PtsSink {
    fn port_count(&self) -> usize {
        1
    }

    fn poll_push_to(
        &mut self,
        _cx: &mut Context<'_>,
        _port: usize,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::DataFrame(frame)) = packet_slot.take() {
            self.pts_ns.push(frame.timing.pts_ns);
        }
        Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

#[tokio::test]
async fn the_first_picture_is_stamped_with_the_running_time_it_arrived_at() {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind rtsp control");
    let port = listener.local_addr().unwrap().port();
    let mut server = RtspServerSink::from_listener(listener).unwrap();
    server
        .configure_pipeline(&Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        })
        .expect("configure the server");

    let access_units = access_units(FIXTURE);
    let client_done = AtomicBool::new(false);
    let serve = async {
        let mut out = CollectSink::default();
        for (index, access_unit) in access_units.iter().cycle().enumerate() {
            if client_done.load(Ordering::Relaxed) {
                break;
            }
            let sequence = index as u64;
            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(
                    access_unit.clone().into_boxed_slice(),
                )),
                timing: FrameTiming {
                    pts_ns: sequence * FRAME_PERIOD.as_nanos() as u64,
                    ..FrameTiming::default()
                },
                sequence,
                meta: Default::default(),
            };
            if server
                .process(PipelinePacket::DataFrame(frame), &mut out)
                .await
                .is_err()
            {
                break;
            }
            tokio::time::sleep(FRAME_PERIOD).await;
        }
    };

    let mut src = RtspSrcN::new(format!("rtsp://127.0.0.1:{port}/stream")).with_frame_limit(FRAMES);
    MultiOutputSource::set_clock_sync(&mut src, ClockSync::new(Arc::new(AgedClock), 0));
    let mut sink = PtsSink::default();
    let earliest_arrival_ns = AgedClock.now_ns();
    let play = async {
        let result = tokio::time::timeout(RUN_LIMIT, src.run(&mut sink)).await;
        client_done.store(true, Ordering::Relaxed);
        result
    };
    let (played, ()) = tokio::join!(play, serve);
    played
        .expect("the session ends within the run limit")
        .expect("the session plays to its frame limit");

    let latest_arrival_ns = AgedClock.now_ns();
    let first_pts_ns = sink.pts_ns[0];
    assert!(
        (earliest_arrival_ns..=latest_arrival_ns).contains(&first_pts_ns),
        "first pts {first_pts_ns} ns is not a running time between {earliest_arrival_ns} and {latest_arrival_ns} ns"
    );
}
