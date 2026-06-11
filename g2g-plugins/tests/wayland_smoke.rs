//! End-to-end smoke test for the Wayland display sink.
//!
//! Pipeline: RtspSrc -> FfmpegH264Dec(Nv12) -> WaylandSink.
//!
//! Ignored by default. Requires:
//! - A running Wayland session (`WAYLAND_DISPLAY` set in the environment).
//! - An RTSP feed at `G2G_RTSP_TEST_URL`, or the rtsp.stream default.
//!
//! Unlike `kms_smoke` this test runs *inside* your normal desktop session,
//! so the easiest setup is to leave the MediaMTX + ffmpeg recipe from
//! `rtsp_ffmpeg_e2e.rs` running in two terminals, then:
//!
//! ```sh
//! G2G_RTSP_TEST_URL=rtsp://localhost:8554/pattern \
//!     cargo test -p g2g-plugins \
//!     --features "rtsp ffmpeg wayland-sink" \
//!     --test wayland_smoke -- --ignored --nocapture
//! ```
//!
//! A window titled "glass2glass" should appear on the active output
//! showing the test pattern for ~2 seconds.
//!
//! Decoder backend is selectable via `G2G_DECODER`: `software` (default,
//! libavcodec built-in) or `nvdec` (h264_cuvid on NVIDIA, requires
//! libnvcuvid + a libavcodec build with cuvid). Example:
//!
//! ```sh
//! G2G_DECODER=nvdec \
//!     G2G_RTSP_TEST_URL=rtsp://localhost:8554/pattern \
//!     cargo test -p g2g-plugins \
//!     --features "rtsp ffmpeg wayland-sink" \
//!     --test wayland_smoke -- --ignored --nocapture
//! ```
//!
//! Default link capacity is now `LatencyProfile::Live` (cap = 2), so
//! the smoke runs under the same low-latency profile a production live
//! pipeline uses. Override via `G2G_LINK_CAP=N` to probe other depths.

#![cfg(all(
    target_os = "linux",
    feature = "rtsp",
    feature = "ffmpeg",
    feature = "wayland-sink"
))]

use g2g_core::runtime::{run_source_transform_sink, LatencyProfile, LinkCapacity};
use g2g_core::PipelineClock;
use g2g_plugins::ffmpegdec::{Backend, FfmpegH264Dec, OutputFormat};
use g2g_plugins::rtspsrc::RtspSrc;
use g2g_plugins::waylandsink::WaylandSink;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
#[ignore = "needs a Wayland session + an RTSP feed (set G2G_RTSP_TEST_URL)"]
async fn wayland_sink_shows_rtsp_h264_in_a_window() {
    if std::env::var_os("WAYLAND_DISPLAY").is_none() {
        eprintln!("skipping: no WAYLAND_DISPLAY in env (run under a Wayland session)");
        return;
    }

    let url = std::env::var("G2G_RTSP_TEST_URL")
        .unwrap_or_else(|_| "rtsp://localhost:8554/pattern".to_string());
    eprintln!("connecting to {url}");

    // Frames the source pushes before EOS. NVDEC has a noticeable
    // startup tax (libnvcuvid load, CUDA context, surface pool alloc);
    // 60 frames is fine for sw but doesn't amortize cuvid's startup, so
    // expose this as an env knob. Steady-state latency numbers (p50/p95)
    // only get meaningful with TARGET >= ~300.
    let target: u64 = std::env::var("G2G_TARGET_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    eprintln!("target frames = {target}");
    // Each of the two pipeline links holds this many in-flight packets.
    // Latency at steady state is dominated by `2 * cap * frame_period`,
    // so this is the key knob for the glass-to-glass latency hunt.
    // Default to `LatencyProfile::Live` (cap = 2) since the smoke test
    // models a camera-to-display feed. `G2G_LINK_CAP=N` overrides to
    // probe behaviour at an arbitrary depth (the live-edge bisection
    // tooling uses this).
    let link_cap: LinkCapacity = match std::env::var("G2G_LINK_CAP").ok().and_then(|s| s.parse().ok()) {
        Some(n) => LinkCapacity::new(n),
        None => LatencyProfile::Live.link_capacity(),
    };
    eprintln!("link capacity = {}", link_cap.get());

    // G2G_DECODER selects the libavcodec backend: `software` (default,
    // built-in H.264 decoder) or `nvdec` (h264_cuvid, requires NVIDIA
    // driver + libnvcuvid + a libavcodec build with cuvid).
    let backend = match std::env::var("G2G_DECODER")
        .unwrap_or_else(|_| "software".into())
        .to_ascii_lowercase()
        .as_str()
    {
        "software" | "sw" => Backend::Software,
        "nvdec" | "cuvid" | "nvidia" => Backend::NvdecCuvid,
        other => panic!("unknown G2G_DECODER={other:?} (expected software|nvdec)"),
    };
    eprintln!("decoder backend = {backend:?}");

    let mut src = RtspSrc::new(url).with_frame_limit(target);
    let mut dec = FfmpegH264Dec::new()
        .with_output_format(OutputFormat::Nv12)
        .with_backend(backend);
    let mut snk = WaylandSink::new().with_title("glass2glass smoke test");
    let clock = ZeroClock;

    let start = std::time::Instant::now();
    // Budget: 30 s of fixed setup tax (cuvid startup can be 1-2 s alone)
    // plus 100 ms per requested frame. 60 frames -> 36 s; 600 frames ->
    // 90 s; matches sw and NVDEC steady-state throughput with headroom.
    let timeout_s = 30 + (target * 100 / 1000).max(30);
    let stats = tokio::time::timeout(
        std::time::Duration::from_secs(timeout_s),
        run_source_transform_sink(&mut src, &mut dec, &mut snk, &clock, link_cap),
    )
    .await
    .unwrap_or_else(|_| panic!("pipeline should complete within {timeout_s}s"))
    .expect("end-to-end Wayland pipeline should succeed");
    let elapsed = start.elapsed();

    let fps = stats.frames_emitted as f64 / elapsed.as_secs_f64();
    let lat = snk.latency_snapshot();
    eprintln!(
        "stats: emitted={} decoded={} presented={} elapsed={:.2}s effective_fps={:.1}",
        stats.frames_emitted,
        dec.decoded_count(),
        snk.frames_presented(),
        elapsed.as_secs_f64(),
        fps,
    );
    eprintln!(
        "glass-to-glass latency: n={} mean={:.1}ms p50={:.1}ms p95={:.1}ms p99={:.1}ms max={:.1}ms",
        lat.count,
        lat.mean_ns as f64 / 1e6,
        lat.p50_ns as f64 / 1e6,
        lat.p95_ns as f64 / 1e6,
        lat.p99_ns as f64 / 1e6,
        lat.max_ns as f64 / 1e6,
    );
    assert!(lat.count > 0, "no latency samples — arrival_ns not threaded through");

    assert!(dec.decoded_count() > 0, "decoder produced no NV12 frames");
    // We don't assert `presented > 0` because the compositor's frame-
    // callback cadence and the EOS-driven shutdown can race: the last
    // few frames in flight may not paint before the worker exits.

    // Pacing assertions: with compositor frame-callback gating, the
    // producer is throttled to refresh. On a 60 Hz output, 60 frames
    // should take ~1s. If pacing regresses (process() returns without
    // waiting for the frame callback), fps will be hundreds — the
    // decoder runs faster than display refresh by an order of magnitude.
    // The lower bound catches the opposite regression (stall on a
    // never-arriving callback) that the outer timeout would otherwise
    // mask as "passed, just slow".
    assert!(
        fps < 200.0,
        "pacing regression: {fps:.1} fps suggests process() is not waiting on the frame callback"
    );
    assert!(
        fps > 10.0,
        "pacing stall: only {fps:.1} fps — frame callback may not be firing"
    );
}
