//! Real-peer RTSP serving interop: ffmpeg *plays* from the g2g `RtspServerSink`
//! over TCP-interleaved (`-rtsp_transport tcp`, RFC 2326 §10.12) and decodes the
//! stream. The sink packetizes a real H.264 fixture and frames the RTP as
//! `$`-framed binary on the control connection; ffmpeg (the reference client)
//! must handshake, demux, depayload, and decode it. Validates the serving-sink
//! interleaved path against a reference peer (the in-process loopback cannot).
//!
//! Ignored by default (needs ffmpeg, opens a local TCP socket). Run:
//!
//! ```sh
//! cargo test -p g2g-plugins --features rtsp-server --test rtsp_ffmpeg_serve_interop \
//!     -- --ignored --nocapture
//! ```
#![cfg(feature = "rtsp-server")]

use core::time::Duration;
use std::net::TcpListener as StdTcpListener;
use std::process::Command;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket,
    PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::rtspserversink::RtspServerSink;

#[path = "common/annexb.rs"]
mod annexb;

use annexb::access_units;

const FIXTURE: &[u8] = include_bytes!("fixtures/h264_640x480.h264");

struct NullOut;
impl OutputSink for NullOut {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        packet_slot.take();
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

#[tokio::test]
#[ignore = "needs ffmpeg with RTSP; opens a local TCP socket"]
async fn ffmpeg_plays_interleaved_from_rtspserversink() {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind rtsp control");
    let port = listener.local_addr().unwrap().port();
    let mut sink = RtspServerSink::from_listener(listener)
        .unwrap()
        .with_rtp(96, 0x1234_5678);
    sink.configure_pipeline(&h264_caps()).expect("configure");

    // ffmpeg connects as a player over TCP-interleaved and decodes a few frames.
    let url = format!("rtsp://127.0.0.1:{port}/stream");
    let ffmpeg = tokio::task::spawn_blocking(move || {
        Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-rtsp_transport",
                "tcp",
                "-i",
                &url,
                "-frames:v",
                "3",
                "-f",
                "null",
                "-",
            ])
            .status()
    });

    let aus = access_units(FIXTURE);
    assert!(!aus.is_empty(), "fixture split into access units");

    // Drive the sink: the first frame blocks until ffmpeg has PLAYed. Loop the
    // fixture so ffmpeg reliably decodes its 3 frames, then it exits and a send
    // fails (which just means it is done).
    let server = async move {
        let mut null = NullOut;
        for i in 0u64..200 {
            let au = aus[(i as usize) % aus.len()].clone();
            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
                timing: FrameTiming {
                    pts_ns: i * 33_000_000,
                    ..FrameTiming::default()
                },
                sequence: i,
                meta: Default::default(),
            };
            if sink
                .process(PipelinePacket::DataFrame(frame), &mut null)
                .await
                .is_err()
            {
                break; // ffmpeg left after decoding its frames
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        sink.frames_sent()
    };

    let (status, frames_sent) = tokio::join!(
        tokio::time::timeout(Duration::from_secs(25), ffmpeg),
        server
    );
    let status = status
        .expect("ffmpeg finishes within 25s")
        .expect("join")
        .expect("ffmpeg ran");
    assert!(
        status.success(),
        "ffmpeg decoded the interleaved RTSP stream (exit {status:?})"
    );
    assert!(frames_sent > 0, "sink served frames after PLAY");

    // A reference RTSP client handshook, depayloaded, and decoded the served
    // stream: persist peer-tagged `Oracle` evidence so `--maturity` derives
    // rtspserversink as InteropTested.
    use g2g_core::conformance::{ConformanceDimension, Evidence};
    g2g_plugins::conformance::persist::record_evidence(
        "rtspserversink",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("h264")
            .detail("ffmpeg plays and decodes the TCP-interleaved RTSP stream"),
    )
    .expect("record oracle evidence");
}
