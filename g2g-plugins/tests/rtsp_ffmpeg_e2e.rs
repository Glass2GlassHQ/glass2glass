//! End-to-end smoke test: pull H.264 from a public RTSP server and decode
//! it to I420 through the ffmpeg software decoder.
//!
//! Pipeline:
//!
//! ```text
//! RtspSrc ─► FfmpegH264Dec ─► FakeSink
//!  System/H.264 Annex-B        System/I420
//! ```
//!
//! Why this exists: the Linux production decode path is `ffmpeg` (the
//! `vaapi` feature is blocked on cros-codecs 0.0.6 GBM/NV12 allocation on
//! AMD iGPUs — see `vaapidec.rs`). This test exercises the full chain a
//! user would actually deploy against a public camera.
//!
//! Ignored by default: needs outbound RTSP (TCP 554 / RTP). Run with:
//!
//! ```sh
//! cargo test -p g2g-plugins --features "rtsp ffmpeg" \
//!     --test rtsp_ffmpeg_e2e -- --ignored --nocapture
//! ```
//!
//! Override the URL via `G2G_RTSP_TEST_URL`; the default targets the
//! `rtsp.stream/pattern` public test feed (same one `rtsp_smoke` uses).
//! The historical Wowza demo (`wowzaec2demo.streamlock.net`) is unreliable
//! and was the cause of an earlier 60s hang in this test.
//!
//! # Recommended: local RTSP server
//!
//! Public test feeds come and go. For a deterministic loop, run MediaMTX
//! locally and publish a synthetic stream into it with ffmpeg.
//!
//! Terminal 1 (RTSP server, Docker):
//!
//! ```sh
//! docker run --rm -it --network host \
//!     -e MTX_PROTOCOLS=tcp \
//!     bluenviron/mediamtx:latest
//! ```
//!
//! Terminal 2 (publisher — note `-pix_fmt yuv420p`, x264 baseline rejects
//! testsrc's native 4:4:4):
//!
//! ```sh
//! ffmpeg -re -f lavfi -i testsrc=size=640x480:rate=30 \
//!     -pix_fmt yuv420p \
//!     -c:v libx264 -tune zerolatency -profile:v baseline \
//!     -f rtsp -rtsp_transport tcp rtsp://localhost:8554/pattern
//! ```
//!
//! Terminal 3 (this test):
//!
//! ```sh
//! G2G_RTSP_TEST_URL=rtsp://localhost:8554/pattern \
//!     cargo test -p g2g-plugins --features "rtsp ffmpeg" \
//!     --test rtsp_ffmpeg_e2e -- --ignored --nocapture
//! ```

#![cfg(all(target_os = "linux", feature = "rtsp", feature = "ffmpeg"))]

use g2g_core::runtime::run_source_transform_sink;
use g2g_core::{Caps, Dim, PipelineClock, VideoFormat};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::ffmpegdec::FfmpegH264Dec;
use g2g_plugins::rtspsrc::RtspSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
#[ignore = "requires outbound RTSP; set G2G_RTSP_TEST_URL or use the Wowza demo default"]
async fn rtsp_to_ffmpeg_decode_emits_i420_frames() {
    let url = std::env::var("G2G_RTSP_TEST_URL")
        .unwrap_or_else(|_| "rtsp://rtsp.stream/pattern".to_string());
    eprintln!("connecting to {url}");

    const TARGET: u64 = 30;

    let mut src = RtspSrc::new(url).with_frame_limit(TARGET);
    let mut dec = FfmpegH264Dec::new();
    let mut snk = FakeSink::new();
    let clock = ZeroClock;

    let stats = tokio::time::timeout(
        std::time::Duration::from_secs(120),
        run_source_transform_sink(&mut src, &mut dec, &mut snk, &clock, 8),
    )
    .await
    .expect("pipeline should complete within 120s")
    .expect("end-to-end pipeline should succeed");

    eprintln!(
        "pipeline stats: frames_consumed={}, decoded={}, sink_received={}",
        stats.frames_consumed,
        dec.decoded_count(),
        snk.received(),
    );

    // The decoder may swallow the first few access units while it primes
    // (SPS/PPS, B-frame reordering), so we don't assert decoded == TARGET.
    // What matters: bytes flowed end-to-end and at least one decoded I420
    // frame reached the sink.
    assert!(
        dec.decoded_count() > 0,
        "decoder produced no I420 frames from {} access units",
        stats.frames_consumed,
    );
    assert!(snk.received() > 0, "sink received no decoded frames");
    assert!(snk.eos_seen(), "EOS must propagate after frame limit");

    // FfmpegH264Dec emits an I420 CapsChanged before the first decoded
    // frame; the H.264 CapsChanged from RtspSrc is swallowed inside the
    // decoder, so the sink should only see the I420 one(s).
    let caps_changes = snk.caps_changes();
    assert!(
        !caps_changes.is_empty(),
        "sink must observe at least one I420 CapsChanged"
    );
    let first = &caps_changes[0];
    assert_eq!(
        first.frames_before, 0,
        "CapsChanged must precede the first decoded frame"
    );
    match &first.caps {
        Caps::Video {
            format: VideoFormat::I420,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        } => {
            eprintln!("decoded resolution: {}x{}", w, h);
            assert!(*w > 0 && *h > 0);
        }
        other => panic!("expected fixed I420 caps, got {other:?}"),
    }
}
