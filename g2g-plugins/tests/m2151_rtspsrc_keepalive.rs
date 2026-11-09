#![cfg(all(feature = "std", feature = "rtsp", feature = "rtsp-server"))]

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use core::time::Duration;
use std::net::TcpListener as StdTcpListener;
use std::sync::atomic::{AtomicBool, Ordering};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, MultiOutputSink,
    MultiOutputSource, PipelinePacket, PushOutcome, Rate, VideoCodec,
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
const FIXTURE_WIDTH: u32 = 64;
const FIXTURE_HEIGHT: u32 = 48;
const FRAME_PERIOD: Duration = Duration::from_millis(33);
const SESSION_TIMEOUT: Duration = Duration::from_secs(2);
// well past the session timeout, so only keepalives sent during the block keep the session
const BLOCK: Duration = Duration::from_secs(5);
// more frames than the server sends before it reaps a silent client
const FRAMES: u64 = 150;
const RUN_LIMIT: Duration = Duration::from_secs(30);

#[derive(Default)]
struct BlockingSink {
    unblock: Option<Pin<Box<tokio::time::Sleep>>>,
    frames: u64,
}

impl MultiOutputSink for BlockingSink {
    fn port_count(&self) -> usize {
        1
    }

    fn poll_push_to(
        &mut self,
        cx: &mut Context<'_>,
        _port: usize,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        if matches!(packet_slot, Some(PipelinePacket::DataFrame(_))) {
            let unblock = self
                .unblock
                .get_or_insert_with(|| Box::pin(tokio::time::sleep(BLOCK)));
            if unblock.as_mut().poll(cx).is_pending() {
                return Poll::Pending;
            }
            self.frames += 1;
        }
        packet_slot.take();
        Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

#[tokio::test]
async fn a_blocked_push_keeps_the_rtsp_session_alive() {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind rtsp control");
    let port = listener.local_addr().unwrap().port();
    let mut server = RtspServerSink::from_listener(listener)
        .unwrap()
        .with_rtp(96, 0x1234_5678)
        .with_session_timeout(SESSION_TIMEOUT);
    server
        .configure_pipeline(&Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(FIXTURE_WIDTH),
            height: Dim::Fixed(FIXTURE_HEIGHT),
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        })
        .expect("configure the server");

    let aus = access_units(FIXTURE);
    let client_done = AtomicBool::new(false);
    let serve = async {
        let mut out = CollectSink::default();
        for (index, au) in aus.iter().cycle().enumerate() {
            if client_done.load(Ordering::Relaxed) {
                break;
            }
            let sequence = index as u64;
            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(
                    au.clone().into_boxed_slice(),
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
    let mut sink = BlockingSink::default();
    let play = async {
        let result = tokio::time::timeout(RUN_LIMIT, src.run(&mut sink)).await;
        client_done.store(true, Ordering::Relaxed);
        result
    };

    let (played, ()) = tokio::join!(play, serve);
    played
        .expect("the session keeps delivering after the block")
        .expect("the session plays to its frame limit");
    assert_eq!(sink.frames, FRAMES);
}
