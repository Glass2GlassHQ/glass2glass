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

#![cfg(all(target_os = "linux", feature = "webrtc"))]

use g2g_core::runtime::run_source_transform_sink;
use g2g_core::{Caps, Dim, PipelineClock, Rate, VideoCodec};
use g2g_plugins::filesrc::FileSrc;
use g2g_plugins::h264parse::H264Parse;
use g2g_plugins::webrtcsink::WebRtcSink;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
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

    let h264 = Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    };
    let mut src = FileSrc::new(&fixture, h264);
    let mut parse = H264Parse::new();
    let mut sink = WebRtcSink::new(whip_url);
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
