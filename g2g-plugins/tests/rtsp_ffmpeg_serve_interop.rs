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

use core::future::Future;
use core::pin::Pin;
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

const FIXTURE: &[u8] = include_bytes!("fixtures/h264_640x480.h264");

struct NullOut;
impl OutputSink for NullOut {
    fn push<'a>(
        &'a mut self,
        _p: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Split an Annex-B stream into NAL units (each retaining its leading start code).
fn nals(mut data: &[u8]) -> Vec<Vec<u8>> {
    // find the first start code
    let mut out = Vec::new();
    let sc = |d: &[u8]| d.windows(3).position(|w| w == [0, 0, 1]);
    let Some(first) = sc(data) else { return out };
    data = &data[first..];
    while !data.is_empty() {
        // skip this NAL's start code, find the next one
        let next = sc(&data[3..]).map(|p| p + 3);
        let end = next.unwrap_or(data.len());
        // trim a 4-byte start code's leading zero back onto the previous NAL split
        let mut nal_end = end;
        if let Some(n) = next {
            if data[n - 1] == 0 {
                nal_end = n - 1;
            }
        }
        out.push(data[..nal_end].to_vec());
        data = &data[nal_end..];
    }
    out
}

/// Group NALs into access units: a VCL NAL (type 1..=5) closes the AU it ends, so
/// SPS/PPS attach to the following IDR and each coded picture is its own AU.
fn access_units(data: &[u8]) -> Vec<Vec<u8>> {
    let mut aus = Vec::new();
    let mut cur: Vec<u8> = Vec::new();
    let mut cur_has_vcl = false;
    for nal in nals(data) {
        let sc_len = if nal.starts_with(&[0, 0, 0, 1]) { 4 } else { 3 };
        let nal_type = nal.get(sc_len).map(|b| b & 0x1F).unwrap_or(0);
        let is_vcl = (1..=5).contains(&nal_type);
        if is_vcl && cur_has_vcl {
            aus.push(core::mem::take(&mut cur));
            cur_has_vcl = false;
        }
        cur.extend_from_slice(&nal);
        cur_has_vcl |= is_vcl;
    }
    if !cur.is_empty() {
        aus.push(cur);
    }
    aus
}

#[tokio::test]
#[ignore = "needs ffmpeg with RTSP; opens a local TCP socket"]
async fn ffmpeg_plays_interleaved_from_rtspserversink() {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind rtsp control");
    let port = listener.local_addr().unwrap().port();
    let mut sink = RtspServerSink::from_listener(listener).unwrap().with_rtp(96, 0x1234_5678);
    sink.configure_pipeline(&h264_caps()).expect("configure");

    // ffmpeg connects as a player over TCP-interleaved and decodes a few frames.
    let url = format!("rtsp://127.0.0.1:{port}/stream");
    let ffmpeg = tokio::task::spawn_blocking(move || {
        Command::new("ffmpeg")
            .args([
                "-hide_banner", "-loglevel", "error",
                "-rtsp_transport", "tcp", "-i", &url,
                "-frames:v", "3", "-f", "null", "-",
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
                timing: FrameTiming { pts_ns: i * 33_000_000, ..FrameTiming::default() },
                sequence: i,
                meta: Default::default(),
            };
            if sink.process(PipelinePacket::DataFrame(frame), &mut null).await.is_err() {
                break; // ffmpeg left after decoding its frames
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        sink.frames_sent()
    };

    let (status, frames_sent) =
        tokio::join!(tokio::time::timeout(Duration::from_secs(25), ffmpeg), server);
    let status = status.expect("ffmpeg finishes within 25s").expect("join").expect("ffmpeg ran");
    assert!(status.success(), "ffmpeg decoded the interleaved RTSP stream (exit {status:?})");
    assert!(frames_sent > 0, "sink served frames after PLAY");
}
