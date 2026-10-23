#![cfg(all(
    feature = "std",
    feature = "rtsp",
    feature = "rtsp-server",
    feature = "ffmpeg",
    target_os = "linux"
))]

use core::time::Duration;
use std::net::TcpListener as StdTcpListener;
use std::sync::atomic::{AtomicBool, Ordering};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{parse_launch, run_graph, LatencyProfile};
use g2g_core::{
    AsyncElement, Caps, Dim, FrameTiming, MemoryDomain, PipelinePacket, Rate, VideoCodec,
};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;
use g2g_plugins::rtspserversink::RtspServerSink;

#[path = "common/annexb.rs"]
mod annexb;
#[path = "common/collect_sink.rs"]
mod collect_sink;

use annexb::access_units;
use collect_sink::CollectSink;

// openh264 output: its SPS has VUI but no timing info, so no frame rate
const FIXTURE: &[u8] = include_bytes!("fixtures/h264_64x48_no_vui_timing.h264");
const FIXTURE_WIDTH: u32 = 64;
const FIXTURE_HEIGHT: u32 = 48;
const FRAME_PERIOD: Duration = Duration::from_millis(33);
const DECODED_FRAMES: u64 = 10;
// the server's first frame waits for a PLAY that a failed negotiation never sends
const SERVE_LIMIT: Duration = Duration::from_secs(20);
const PAYLOAD_TYPE: u8 = 96;
const SSRC: u32 = 0x1234_5678;

#[tokio::test]
async fn rtspsrc_decodes_a_stream_whose_sps_has_no_frame_rate() {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind rtsp control");
    let port = listener.local_addr().unwrap().port();
    let mut server = RtspServerSink::from_listener(listener)
        .unwrap()
        .with_rtp(PAYLOAD_TYPE, SSRC);
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
    let serving = async {
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
    let serve = async {
        let _ = tokio::time::timeout(SERVE_LIMIT, serving).await;
    };

    let line = format!(
        "rtspsrc location=rtsp://127.0.0.1:{port}/stream num-buffers={DECODED_FRAMES} \
         ! h264parse ! avdec_h264 ! fakesink"
    );
    let registry = default_registry();
    let play = async {
        let graph =
            parse_launch(&registry, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
        let result = run_graph(graph, &WallClock::new(), LatencyProfile::Throughput).await;
        client_done.store(true, Ordering::Relaxed);
        result
    };

    let (stats, ()) = tokio::join!(play, serve);
    let stats = stats.unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    assert_eq!(
        stats.frames_emitted, DECODED_FRAMES,
        "rtspsrc delivers every frame"
    );
    assert!(
        stats.frames_consumed >= DECODED_FRAMES,
        "decoded frames reach the sink"
    );
}
