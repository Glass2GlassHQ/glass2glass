//! Real-peer RTP interop (M528): ffmpeg's RTP H.264 payloader (`-f rtp`) streams
//! into the g2g `UdpSrc`, which depayloads RFC 6184 RTP (single-NAL / FU-A /
//! STAP-A) back into Annex-B access units. This proves the g2g depayloader +
//! jitter buffer match ffmpeg's payloader on the wire (FU-A fragmentation of a
//! large IDR is where depay bugs hide), which the g2g<->g2g `udp_loopback` (both
//! sides g2g) cannot.
//!
//! Ignored by default (needs ffmpeg, opens a local UDP socket). Run:
//!
//! ```sh
//! cargo test -p g2g-plugins --features udp-ingress --test udp_ffmpeg_rtp_interop \
//!     -- --ignored --nocapture
//! ```
#![cfg(feature = "udp-ingress")]

use core::future::Future;
use core::pin::Pin;
use std::net::UdpSocket as StdUdpSocket;
use std::process::Command;
use std::time::Duration;

use g2g_core::runtime::SourceLoop;
use g2g_core::{G2gError, OutputSink, PipelinePacket, PushOutcome};

use g2g_plugins::udpsrc::UdpSrc;

/// Records each emitted access unit's leading bytes, length, and keyframe flag.
#[derive(Default)]
struct AuCollect {
    aus: Vec<(Vec<u8>, usize, bool)>,
}

impl OutputSink for AuCollect {
    fn push<'a>(
        &'a mut self,
        p: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        if let PipelinePacket::DataFrame(frame) = &p {
            if let Some(slice) = frame.domain.as_system_slice() {
                let s = slice;
                let head = s[..s.len().min(5)].to_vec();
                self.aus.push((head, s.len(), frame.timing.keyframe));
            }
        }
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

/// An Annex-B access unit starts with a 3- or 4-byte start code.
fn is_annex_b(head: &[u8]) -> bool {
    head.starts_with(&[0, 0, 0, 1]) || head.starts_with(&[0, 0, 1])
}

#[tokio::test]
#[ignore = "needs ffmpeg with the rtp muxer; opens a local UDP socket"]
async fn ffmpeg_rtp_h264_streams_into_udpsrc() {
    // Bind the receive socket up front; ffmpeg blasts RTP with no handshake, so
    // the kernel buffers early datagrams until UdpSrc reads them.
    let sock = StdUdpSocket::bind("127.0.0.1:0").expect("bind udp");
    let port = sock.local_addr().unwrap().port();
    const N: u64 = 20;
    let mut src = UdpSrc::from_socket(sock)
        .expect("adopt socket")
        .with_frame_limit(N);
    src.configure_pipeline(&src_caps()).expect("configure");

    // ffmpeg payloads ~3 s of H.264 as RTP to our port (PT 96, its own SSRC).
    let url = format!("rtp://127.0.0.1:{port}");
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
                "testsrc=size=320x240:rate=15:duration=3",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-payload_type",
                "96",
                "-f",
                "rtp",
                &url,
            ])
            .status()
    });

    let mut sink = AuCollect::default();
    let received = tokio::time::timeout(Duration::from_secs(20), src.run(&mut sink))
        .await
        .expect("UdpSrc receives N access units within 20s")
        .expect("UdpSrc runs");
    let _ = ffmpeg.await;

    assert_eq!(
        received, N,
        "depayloaded the requested number of access units"
    );
    assert_eq!(sink.aus.len() as u64, N);
    // Every emitted AU must be Annex-B framed (the depayloader re-framed the RTP
    // NALs correctly, including reassembling FU-A fragments).
    for (i, (head, len, _)) in sink.aus.iter().enumerate() {
        assert!(*len > 4, "AU {i} has content");
        assert!(
            is_annex_b(head),
            "AU {i} is Annex-B framed (got {head:02x?})"
        );
    }
    // At least one IDR access unit must be recognized as a keyframe (proves an
    // FU-A-fragmented IDR reassembled into a valid access unit).
    assert!(
        sink.aus.iter().any(|(_, _, kf)| *kf),
        "at least one keyframe among the depayloaded access units"
    );

    // The depayloader consumed a real ffmpeg RTP payloader on the wire: persist
    // peer-tagged `Oracle` evidence so `--maturity` derives udpsrc as InteropTested.
    use g2g_core::conformance::{ConformanceDimension, Evidence};
    g2g_plugins::conformance::persist::record_evidence(
        "udpsrc",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("h264")
            .detail("ffmpeg RTP (RFC 6184) payloader depayloaded to Annex-B, FU-A IDR reassembled"),
    )
    .expect("record oracle evidence");
}

fn src_caps() -> g2g_core::Caps {
    use g2g_core::{Caps, Dim, Rate, VideoCodec};
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(15 << 16),
    }
}
