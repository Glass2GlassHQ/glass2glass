//! Real-peer RTSP ingest interop (M529): ffmpeg *publishes* to the g2g
//! `RtspServerSrc` over RTSP (`-f rtsp -rtsp_transport udp`, the
//! OPTIONS/ANNOUNCE/SETUP/RECORD direction), and the server depayloads the RTP it
//! then receives into Annex-B access units. This drives the g2g RTSP responder +
//! the shared RTP receive path (jitter buffer + depayload, M520) against a
//! reference RTSP client, which the in-tree `record_loopback` (a hand-rolled g2g
//! publisher) cannot.
//!
//! Ignored by default (needs ffmpeg, opens local TCP + UDP sockets). Run:
//!
//! ```sh
//! cargo test -p g2g-plugins --features rtsp-server --test rtsp_ffmpeg_record_interop \
//!     -- --ignored --nocapture
//! ```
#![cfg(feature = "rtsp-server")]

use std::net::TcpListener as StdTcpListener;
use std::process::Command;
use std::time::Duration;

use g2g_core::runtime::SourceLoop;
use g2g_core::{G2gError, OutputSink, PipelinePacket, PushOutcome};

use g2g_plugins::rtspserversrc::RtspServerSrc;

#[derive(Default)]
struct AuCollect {
    aus: Vec<(Vec<u8>, usize, bool)>,
}

impl OutputSink for AuCollect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let p = packet_slot.take().expect("poll_push without a packet");

        if let PipelinePacket::DataFrame(frame) = &p {
            if let Some(slice) = frame.domain.as_system_slice() {
                let s = slice;
                self.aus
                    .push((s[..s.len().min(5)].to_vec(), s.len(), frame.timing.keyframe));
            }
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn is_annex_b(head: &[u8]) -> bool {
    head.starts_with(&[0, 0, 0, 1]) || head.starts_with(&[0, 0, 1])
}

#[tokio::test]
#[ignore = "needs ffmpeg with RTSP; opens local TCP + UDP sockets"]
async fn ffmpeg_rtsp_publisher_records_into_rtspserversrc() {
    // Bind the RTSP control listener up front so ffmpeg's TCP connect queues in
    // the backlog before RtspServerSrc calls accept().
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind rtsp control");
    let port = listener.local_addr().unwrap().port();
    const N: u64 = 20;
    let mut src = RtspServerSrc::from_listener(listener)
        .expect("adopt listener")
        .with_video_size(320, 240)
        .with_frame_limit(N);

    // ffmpeg publishes ~4 s of H.264 over RTSP (unicast UDP RTP after RECORD).
    let url = format!("rtsp://127.0.0.1:{port}/stream");
    let ffmpeg = tokio::task::spawn_blocking(move || {
        Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-re",
                "-an",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=15:duration=4",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-f",
                "rtsp",
                "-rtsp_transport",
                "udp",
                &url,
            ])
            .status()
    });

    let mut sink = AuCollect::default();
    let received = tokio::time::timeout(Duration::from_secs(25), src.run(&mut sink))
        .await
        .expect("RtspServerSrc receives N access units within 25s")
        .expect("RtspServerSrc runs");
    let _ = ffmpeg.await;

    assert_eq!(
        received, N,
        "depayloaded the requested number of access units"
    );
    for (i, (head, len, _)) in sink.aus.iter().enumerate() {
        assert!(*len > 4, "AU {i} has content");
        assert!(
            is_annex_b(head),
            "AU {i} is Annex-B framed (got {head:02x?})"
        );
    }
    assert!(
        sink.aus.iter().any(|(_, _, kf)| *kf),
        "at least one keyframe among the RTSP-ingested access units"
    );

    record_ingest_evidence("ffmpeg RECORDs over UDP RTP, depayloaded to Annex-B");
}

/// Persist the peer-tagged `Oracle` evidence both ingest directions earn, so
/// `--maturity` derives rtspserversrc as InteropTested against ffmpeg.
fn record_ingest_evidence(detail: &str) {
    use g2g_core::conformance::{ConformanceDimension, Evidence};
    g2g_plugins::conformance::persist::record_evidence(
        "rtspserversrc",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("h264")
            .detail(detail),
    )
    .expect("record oracle evidence");
}

/// M532: the same interop over **TCP-interleaved** transport
/// (`-rtsp_transport tcp`, RFC 2326 §10.12): ffmpeg pushes RTP as `$`-framed
/// binary on the control connection, and `RtspServerSrc` demuxes + depayloads it.
/// Exercises the interleaved receive path against a reference client (the g2g
/// `interleaved_loopback` cannot).
#[tokio::test]
#[ignore = "needs ffmpeg with RTSP; opens a local TCP socket"]
async fn ffmpeg_rtsp_publisher_records_interleaved_into_rtspserversrc() {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind rtsp control");
    let port = listener.local_addr().unwrap().port();
    const N: u64 = 20;
    let mut src = RtspServerSrc::from_listener(listener)
        .expect("adopt listener")
        .with_video_size(320, 240)
        .with_frame_limit(N);

    let url = format!("rtsp://127.0.0.1:{port}/stream");
    let ffmpeg = tokio::task::spawn_blocking(move || {
        Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-re",
                "-an",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=15:duration=4",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-f",
                "rtsp",
                "-rtsp_transport",
                "tcp",
                &url,
            ])
            .status()
    });

    let mut sink = AuCollect::default();
    let received = tokio::time::timeout(Duration::from_secs(25), src.run(&mut sink))
        .await
        .expect("RtspServerSrc receives N access units within 25s")
        .expect("RtspServerSrc runs");
    let _ = ffmpeg.await;

    assert_eq!(
        received, N,
        "depayloaded the requested number of interleaved access units"
    );
    for (i, (head, len, _)) in sink.aus.iter().enumerate() {
        assert!(*len > 4, "AU {i} has content");
        assert!(
            is_annex_b(head),
            "AU {i} is Annex-B framed (got {head:02x?})"
        );
    }
    assert!(
        sink.aus.iter().any(|(_, _, kf)| *kf),
        "at least one keyframe among the interleaved RTSP-ingested access units"
    );

    record_ingest_evidence("ffmpeg RECORDs over TCP-interleaved RTP, depayloaded to Annex-B");
}
