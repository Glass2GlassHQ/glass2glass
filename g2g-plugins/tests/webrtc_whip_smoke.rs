//! End-to-end smoke test for the WebRTC WHIP egress sink (`WebRtcSink`).
//!
//! Pipeline: `FileSrc(h264) -> H264Parse -> WebRtcSink(WHIP)`.
//!
//! Ignored by default because it needs:
//! - A WHIP server reachable at `G2G_WHIP_URL` (mediamtx is the easy local one,
//!   see below). The sandbox blocks the WebRTC ports, so this is a user-run
//!   harness, not a CI gate.
//! - An H.264 Annex-B fixture path in `G2G_H264_FIXTURE`.
//!
//! Recipe (local mediamtx loopback):
//!
//! ```sh
//! # 1. Start mediamtx (serves WHIP ingest on :8889, WHEP playback on the same).
//! mediamtx
//!
//! # 2. Make an H.264 Annex-B fixture (any clip works):
//! ffmpeg -f lavfi -i testsrc=size=640x480:rate=30:duration=10 \
//!        -c:v libx264 -bsf:v h264_mp4toannexb -f h264 /tmp/clip.h264
//!
//! # 3. Publish it through g2g to mediamtx's WHIP endpoint:
//! G2G_H264_FIXTURE=/tmp/clip.h264 \
//! G2G_WHIP_URL=http://localhost:8889/mystream/whip \
//!     cargo test -p g2g-plugins --features webrtc \
//!     --test webrtc_whip_smoke -- --ignored --nocapture
//!
//! # 4. Watch it: open g2g-plugins/examples/whep-player.html in a browser and
//! #    point it at http://localhost:8889/mystream/whep (mediamtx's WHEP URL).
//! ```
//!
//! A green run means the WHIP handshake (ICE/DTLS/SRTP via str0m) completed and
//! frames were published without error; visual confirmation is the WHEP player.
//!
//! NAT traversal for a cloud SFU (LiveKit, etc.): set `G2G_STUN_SERVER`
//! (`host:port`) for a server-reflexive candidate, and/or `G2G_TURN_SERVER` +
//! `G2G_TURN_USER` + `G2G_TURN_PASS` for the relay fallback. Both are applied to
//! the sink (and, in the loopback, the source) when present.

#![cfg(all(target_os = "linux", feature = "webrtc"))]

use core::future::{ready, Future, Ready};
use core::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use g2g_core::runtime::{run_simple_pipeline, run_source_transform_sink, SourceLoop};
use g2g_core::{
    Caps, ConfigureOutcome, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelineClock,
    PipelinePacket, Rate, VideoCodec,
};
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::filesrc::FileSrc;
use g2g_plugins::h264parse::H264Parse;
use g2g_plugins::webrtcsink::WebRtcSink;
use g2g_plugins::webrtcwhepsrc::WebRtcWhepSrc;

/// Split an Annex-B H.264 byte stream into NAL units, each re-prefixed with a
/// 4-byte start code. Used by the paced publisher to feed real NALs continuously.
fn split_annexb(data: &[u8]) -> std::vec::Vec<std::vec::Vec<u8>> {
    let mut starts = std::vec::Vec::new();
    let mut i = 0usize;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            starts.push(i);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut nals = std::vec::Vec::new();
    for (k, &s) in starts.iter().enumerate() {
        let payload = s + 3;
        let end = if k + 1 < starts.len() {
            let next = starts[k + 1];
            if next > 0 && data[next - 1] == 0 {
                next - 1
            } else {
                next
            }
        } else {
            data.len()
        };
        let mut nal = std::vec![0u8, 0, 0, 1];
        nal.extend_from_slice(&data[payload..end]);
        nals.push(nal);
    }
    nals
}

/// Source that loops a fixture's NALs in real time for `duration`, so a
/// `WebRtcSink` stays alive and publishing across the whole ICE/DTLS handshake +
/// subscriber window (the flat file dump in the other tests finishes in ~0.1 s,
/// before ICE completes, so no media ever flows).
struct PacedH264Src {
    nals: Arc<std::vec::Vec<std::vec::Vec<u8>>>,
    duration: Duration,
}

impl SourceLoop for PacedH264Src {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = Ready<Result<Caps, G2gError>>;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        ready(Ok(paced_caps()))
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::CapsChanged(paced_caps())).await?;
            let start = Instant::now();
            let mut seq = 0u64;
            let mut idx = 0usize;
            while Instant::now().duration_since(start) < self.duration {
                let nal = self.nals[idx % self.nals.len()].clone();
                idx += 1;
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(nal.into_boxed_slice())),
                    FrameTiming { pts_ns: seq * 5_000_000, ..FrameTiming::default() },
                    seq,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
                seq += 1;
                // ~5 ms between NALs: a NAL-paced ~200/s feed, continuous over the
                // whole window without flooding str0m's send buffer.
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(seq)
        })
    }
}

fn paced_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    }
}

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn h264_caps() -> Caps {
    // Concrete geometry matching the fixture: negotiation fixates before any data
    // flows, and `fixate()` rejects `Dim::Any` / `Rate::Any` (see the
    // intercept-caps-must-fixate note), so a source feeding the sink must
    // advertise concrete caps, not wildcards.
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Apply `G2G_STUN_SERVER` / `G2G_TURN_SERVER`+`G2G_TURN_USER`+`G2G_TURN_PASS`
/// to a sink, so a cloud-SFU run picks them up without code changes.
fn with_ice_env_sink(mut sink: WebRtcSink) -> WebRtcSink {
    if let Ok(stun) = std::env::var("G2G_STUN_SERVER") {
        sink = sink.with_stun_server(stun);
    }
    if let (Ok(server), Ok(user), Ok(pass)) = (
        std::env::var("G2G_TURN_SERVER"),
        std::env::var("G2G_TURN_USER"),
        std::env::var("G2G_TURN_PASS"),
    ) {
        eprintln!("using TURN relay {server}");
        sink = sink.with_turn_server(server, user, pass);
    }
    sink
}

/// Same for a WHEP source.
fn with_ice_env_src(mut src: WebRtcWhepSrc) -> WebRtcWhepSrc {
    if let Ok(stun) = std::env::var("G2G_STUN_SERVER") {
        src = src.with_stun_server(stun);
    }
    if let (Ok(server), Ok(user), Ok(pass)) = (
        std::env::var("G2G_TURN_SERVER"),
        std::env::var("G2G_TURN_USER"),
        std::env::var("G2G_TURN_PASS"),
    ) {
        src = src.with_turn_server(server, user, pass);
    }
    src
}

#[tokio::test]
#[ignore = "needs a WHIP server (G2G_WHIP_URL) + an H.264 fixture (G2G_H264_FIXTURE)"]
async fn webrtcsink_publishes_h264_to_whip() {
    let (Ok(whip_url), Ok(fixture)) =
        (std::env::var("G2G_WHIP_URL"), std::env::var("G2G_H264_FIXTURE"))
    else {
        eprintln!("skipping: set G2G_WHIP_URL and G2G_H264_FIXTURE to run");
        return;
    };
    eprintln!("publishing {fixture} -> {whip_url}");

    let mut src = FileSrc::new(&fixture, h264_caps());
    let mut parse = H264Parse::new();
    let mut sink = with_ice_env_sink(WebRtcSink::new(whip_url));
    let clock = ZeroClock;

    let stats = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        run_source_transform_sink(&mut src, &mut parse, &mut sink, &clock, 4),
    )
    .await
    .expect("pipeline should complete within 30s")
    .expect("WHIP publish pipeline should succeed");

    eprintln!("source emitted={} frames published={}", stats.frames_emitted, sink.frames_sent());
    assert!(sink.frames_sent() > 0, "expected at least one access unit published over WHIP");
}

/// g2g -> g2g round trip through a media server: `WebRtcSink` (WHIP publish) and
/// `WebRtcWhepSrc` (WHEP subscribe) against the same mediamtx stream. Both
/// elements are HTTP clients, so the server is the relay in the middle (there is
/// no peer-to-peer mode). Proves the full publish + subscribe path in Rust with
/// no browser: the subscriber must receive frames the publisher sent.
///
/// Recipe (extends the single-element recipe above):
///
/// ```sh
/// mediamtx
/// ffmpeg -f lavfi -i testsrc=size=640x480:rate=30:duration=30 \
///        -c:v libx264 -bsf:v h264_mp4toannexb -f h264 /tmp/clip.h264
/// G2G_H264_FIXTURE=/tmp/clip.h264 \
/// G2G_WHIP_URL=http://localhost:8889/loop/whip \
/// G2G_WHEP_URL=http://localhost:8889/loop/whep \
///     cargo test -p g2g-plugins --features webrtc \
///     --test webrtc_whip_smoke webrtc_whip_to_whep_loopback -- --ignored --nocapture
/// ```
///
/// Use a reasonably long fixture (>= ~10s): the publisher must still be live
/// when the subscriber connects a couple seconds later.
#[tokio::test]
#[ignore = "needs mediamtx (WHIP+WHEP) + an H.264 fixture (G2G_WHIP_URL/G2G_WHEP_URL/G2G_H264_FIXTURE)"]
async fn webrtc_whip_to_whep_loopback() {
    let (Ok(whip), Ok(whep), Ok(fixture)) = (
        std::env::var("G2G_WHIP_URL"),
        std::env::var("G2G_WHEP_URL"),
        std::env::var("G2G_H264_FIXTURE"),
    ) else {
        eprintln!("skipping: set G2G_WHIP_URL, G2G_WHEP_URL and G2G_H264_FIXTURE to run");
        return;
    };
    eprintln!("loopback: publish {fixture} -> {whip} ; subscribe <- {whep}");

    // Both pipelines run concurrently on this task (the runner futures are
    // !Send, so `join!` rather than `spawn`). The publisher loops the fixture's
    // NALs in real time so the session stays alive through ICE/DTLS and the
    // subscriber window (a flat file dump finishes before ICE completes, so no
    // media ever reaches the server); the subscriber connects after a moment.
    let bytes = std::fs::read(&fixture).expect("read fixture");
    let nals = Arc::new(split_annexb(&bytes));
    let publisher = async move {
        let mut src = PacedH264Src { nals, duration: Duration::from_secs(12) };
        let mut sink = with_ice_env_sink(WebRtcSink::new(whip));
        let clock = ZeroClock;
        let _ = run_simple_pipeline(&mut src, &mut sink, &clock, 8).await;
    };
    let subscriber = async {
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        let mut rsrc = with_ice_env_src(WebRtcWhepSrc::new(whep).with_frame_limit(30));
        let mut rparse = H264Parse::new();
        let mut rsink = FakeSink::new();
        let clock = ZeroClock;
        tokio::time::timeout(
            std::time::Duration::from_secs(30),
            run_source_transform_sink(&mut rsrc, &mut rparse, &mut rsink, &clock, 4),
        )
        .await
    };

    let (_, sub) = tokio::join!(publisher, subscriber);
    let stats = sub.expect("subscribe should complete within 30s").expect("WHEP subscribe ok");
    eprintln!("subscriber received {} frames over WHEP", stats.frames_emitted);
    assert!(stats.frames_emitted >= 1, "expected to receive at least one frame published by the sink");
}

/// Paced publish: loop a fixture's NALs in real time to a WHIP server for a few
/// seconds, so the WebRTC session stays alive across the full ICE/DTLS handshake
/// and real media flows (unlike the flat-dump tests). A green run + the server
/// logging the stream as published confirms the egress path end to end.
///
/// ```sh
/// docker run -d --network host bluenviron/mediamtx
/// ffmpeg -f lavfi -i testsrc=size=640x480:rate=30:duration=12 \
///        -c:v libx264 -bsf:v h264_mp4toannexb -f h264 /tmp/clip.h264
/// G2G_H264_FIXTURE=/tmp/clip.h264 G2G_WHIP_URL=http://localhost:8889/pub/whip \
///     cargo test -p g2g-plugins --features webrtc \
///     --test webrtc_whip_smoke webrtc_publish_paced -- --ignored --nocapture
/// # then: docker logs <mediamtx>  -> expect "is publishing to path 'pub'"
/// ```
#[tokio::test]
#[ignore = "needs a WHIP server (G2G_WHIP_URL) + an H.264 fixture (G2G_H264_FIXTURE)"]
async fn webrtc_publish_paced() {
    let (Ok(whip), Ok(fixture)) =
        (std::env::var("G2G_WHIP_URL"), std::env::var("G2G_H264_FIXTURE"))
    else {
        eprintln!("skipping: set G2G_WHIP_URL and G2G_H264_FIXTURE to run");
        return;
    };
    let bytes = std::fs::read(&fixture).expect("read fixture");
    let nals = Arc::new(split_annexb(&bytes));
    assert!(!nals.is_empty(), "fixture had no NAL units");
    eprintln!("paced publish: {} NALs looped -> {whip}", nals.len());

    let mut src = PacedH264Src { nals, duration: Duration::from_secs(8) };
    let mut sink = with_ice_env_sink(WebRtcSink::new(whip));
    let clock = ZeroClock;
    let stats = tokio::time::timeout(
        Duration::from_secs(20),
        run_simple_pipeline(&mut src, &mut sink, &clock, 8),
    )
    .await
    .expect("pipeline completes within 20s")
    .expect("paced WHIP publish succeeds");

    eprintln!("paced publish emitted={} handed-to-session={}", stats.frames_emitted, sink.frames_sent());
    assert!(sink.frames_sent() > 100, "expected a continuous feed, got {}", sink.frames_sent());
}
