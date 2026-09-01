//! M1118: live A/V lip-sync soak on a real display and a real audio device.
//!
//! Plays an HLS A/V feed for `G2G_SOAK_SECONDS` (default 60) through
//!
//! ```text
//! HlsSrc(video rendition) -> Mp4DemuxN -> FfmpegH264Dec -> WaylandSink
//! HlsSrc(audio rendition) -> Mp4DemuxN -> FfmpegAudioDec -> AudioConvert -> AudioResample -> PipeWireSink
//! ```
//!
//! and then asserts the sync held for the whole run: the audio sink's
//! `DriftClock` won election and the video sink is slaved to it, the video sink
//! presented one frame per feed frame period with no cumulative loss, late drops
//! stayed near zero, and the video sink's per-frame presentation deadline error
//! (M1118, `WaylandSink::deadline_error_samples`) is both bounded and free of
//! drift, its median over the first quarter of the run matching the last.
//!
//! What this does *not* measure is the absolute audio-to-video offset: the two
//! renditions are independent sources, and the video pacer anchors on its own
//! first frame, so a constant startup offset is absorbed by the anchor. The
//! deadline error is video-against-the-audio-timeline, so its *drift* is the
//! lip-sync number. A camera-and-microphone cross-check belongs to a rig test.
//!
//! Ignored by default. Requires:
//! - A Wayland session (`WAYLAND_DISPLAY` set).
//! - A reachable PipeWire daemon.
//! - `curl` on PATH (used for the playlist / init-segment discovery fetches).
//! - An HLS A/V feed at `G2G_HLS_TEST_URL` (default
//!   `http://localhost:8888/avpattern/index.m3u8`), muxed or with a separate
//!   audio rendition, fMP4 or plain-TS-free CMAF segments.
//!
//! ```sh
//! G2G_SOAK_SECONDS=90 cargo test --release -p g2g-plugins \
//!     --features "hls ffmpeg wayland-sink pipewire" \
//!     --test av_lipsync_soak -- --ignored --nocapture
//! ```
//!
//! A window titled "g2g lipsync soak" appears for the length of the run and the
//! feed's audio plays out on the default PipeWire device.

#![cfg(all(
    target_os = "linux",
    feature = "hls",
    feature = "ffmpeg",
    feature = "wayland-sink",
    feature = "pipewire"
))]

use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use g2g_core::graph::Graph;
use g2g_core::runtime::{run_graph, GraphNodeRef};
use g2g_core::{MonotonicClock, PipelineClock};

use g2g_plugins::audioconvert::AudioConvert;
use g2g_plugins::audioresample::AudioResample;
use g2g_plugins::ffmpegaudiodec::FfmpegAudioDec;
use g2g_plugins::ffmpegdec::{FfmpegH264Dec, OutputFormat};
use g2g_plugins::hls::{parse, MediaType, Playlist};
use g2g_plugins::hlssrc::HlsSrc;
use g2g_plugins::mp4demuxn::{forwardable_streams, Mp4DemuxN, Mp4Port};
use g2g_plugins::pipewiresink::PipeWireSink;
use g2g_plugins::waylandsink::WaylandSink;

/// The feed the soak plays when `G2G_HLS_TEST_URL` is unset.
const DEFAULT_FEED_URL: &str = "http://localhost:8888/avpattern/index.m3u8";

/// Seconds of playback the soak asserts over, when `G2G_SOAK_SECONDS` is unset.
const DEFAULT_SOAK_SECONDS: u64 = 60;

/// Wall time allowed before the first frame is on screen: the playlist probe,
/// the source's first segment fetch, and the decoder open. Frames are only
/// expected for the run past this point.
const STARTUP_ALLOWANCE_SECONDS: f64 = 3.0;

/// How far the presented frame count may sit from the feed's nominal rate over
/// the counted window. Covers a segment fetch running long, not a pacing
/// regression (which presents at decode speed, several times over).
const FRAME_RATE_TOLERANCE: f64 = 0.05;

/// Frames that may still be between the decoder and the screen when the run is
/// cut: the two links' in-flight packets and the one being presented. Anything
/// beyond this that the decoder produced and the sink neither presented nor
/// counted as a late drop is the cumulative loss this soak is looking for.
const IN_FLIGHT_FRAME_ALLOWANCE: u64 = 2 * LINK_CAPACITY as u64 + 1;

/// Late-drop bound the video sink runs under: a frame this far past its deadline
/// is dropped rather than presented late, so a stall is visible as a drop count
/// instead of hiding in the deadline-error tail. Three frame periods at 30 fps.
const MAX_LATENESS_NS: u64 = 100_000_000;

/// Fraction of presented frames that may be late-dropped. A steady feed on an
/// idle compositor should drop none; a handful around a segment fetch stall is
/// not a sync failure.
const LATE_DROP_TOLERANCE: f64 = 0.005;

/// Bound on the 95th percentile of the absolute presentation deadline error. The
/// sink commits at the deadline and reads the clock after the compositor's frame
/// callback, so one refresh period is the floor; this allows three at 60 Hz.
const DEADLINE_ERROR_P95_BOUND_NS: i64 = 50_000_000;

/// How far the median deadline error may move between the first quarter of the
/// run and the last. This is the lip-sync drift number: a video sink losing
/// ground against the audio clock walks its error out monotonically, and 10 ms
/// over a minute is already visible as a lip-sync error.
const DEADLINE_ERROR_DRIFT_BOUND_NS: i64 = 10_000_000;

/// How far the audio clock's estimated playout rate may sit from real time.
/// Both timelines are real time, so the fit should read ~1.0.
const DRIFT_CLOCK_SLOPE_TOLERANCE: f64 = 0.01;

/// Observations the audio clock must have taken for its slope to mean anything
/// (two points are the minimum for a rate at all).
const MIN_DRIFT_CLOCK_OBSERVATIONS: usize = 2;

/// In-flight packets per link. The graph is a playback chain off a buffered
/// network source, not a glass-to-glass live path, so a little depth costs
/// nothing and keeps the decoder fed across a segment fetch.
const LINK_CAPACITY: usize = 4;

/// Fetch `url`, returning `None` when curl is missing or the server refuses. The
/// discovery fetches (master playlist, media playlist, `#EXT-X-MAP` init) are
/// one-shot and outside the graph, so they use curl rather than standing up a
/// second HTTP client.
fn http_get(url: &str) -> Option<Vec<u8>> {
    let out = Command::new("curl")
        .args(["-sSfL", "--max-time", "10", url])
        .output()
        .ok()?;
    out.status.success().then_some(out.stdout)
}

fn http_get_text(url: &str) -> Option<String> {
    http_get(url).and_then(|b| String::from_utf8(b).ok())
}

/// Resolve `reference` (absolute, or relative to `base`'s directory) the way the
/// playlists in this feed use it: a sibling file name.
fn resolve_url(base: &str, reference: &str) -> String {
    if reference.contains("://") {
        return reference.to_string();
    }
    match base.rfind('/') {
        Some(i) => format!("{}{}", &base[..=i], reference),
        None => reference.to_string(),
    }
}

/// The `#EXT-X-MAP` init segment of the media playlist at `url`, and the url
/// itself. `None` when the playlist is not fMP4 or cannot be fetched.
fn init_segment(url: &str) -> Option<Vec<u8>> {
    let text = http_get_text(url)?;
    let Ok(Playlist::Media(media)) = parse(&text) else {
        return None;
    };
    let map = media.map_uri?;
    http_get(&resolve_url(url, &map))
}

/// The feed's frame rate: `G2G_FEED_FPS` if set, else the master playlist's
/// `FRAME-RATE` attribute. The expected frame count is derived from it, so it is
/// read from the feed rather than assumed.
fn feed_fps(master_text: &str) -> Option<f64> {
    if let Ok(v) = std::env::var("G2G_FEED_FPS") {
        return v.parse().ok();
    }
    let rest = master_text.split("FRAME-RATE=").nth(1)?;
    let digits: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    digits.parse().ok()
}

/// Nearest-rank percentile of a sorted slice.
fn percentile(sorted: &[i64], p: usize) -> i64 {
    sorted[(sorted.len() - 1) * p / 100]
}

fn median(samples: &[i64]) -> i64 {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    percentile(&sorted, 50)
}

/// What the run's presentation deadline errors look like: the tails, and the
/// two quarter medians whose difference is the lip-sync drift.
#[derive(Debug)]
struct DeadlineErrorStats {
    p5: i64,
    p50: i64,
    p95: i64,
    first_median: i64,
    last_median: i64,
}

impl DeadlineErrorStats {
    /// Movement of the median error from the first quarter of the run to the
    /// last: video walking away from the audio clock.
    fn drift(&self) -> i64 {
        self.last_median - self.first_median
    }

    /// The larger of the two tails, since an error is as bad early as late.
    fn worst_tail(&self) -> i64 {
        self.p95.abs().max(self.p5.abs())
    }
}

/// Score `errors` (in presentation order), leaving them sorted. `None` for an
/// unpaced run, which records no deadlines at all.
fn deadline_error_stats(errors: &mut [i64]) -> Option<DeadlineErrorStats> {
    if errors.is_empty() {
        return None;
    }
    let quarter = (errors.len() / 4).max(1);
    let first_median = median(&errors[..quarter]);
    let last_median = median(&errors[errors.len() - quarter..]);
    errors.sort_unstable();
    Some(DeadlineErrorStats {
        p5: percentile(errors, 5),
        p50: percentile(errors, 50),
        p95: percentile(errors, 95),
        first_median,
        last_median,
    })
}

/// One demux port plus the decode-side facts the branch needs, discovered from
/// an fMP4 rendition's init segment.
fn only_port(init: &[u8], want_video: bool, what: &str) -> Mp4Port {
    let stream = forwardable_streams(init)
        .into_iter()
        .find(|s| s.video == want_video)
        .unwrap_or_else(|| panic!("{what} rendition's init segment carries no matching track"));
    Mp4Port {
        track_id: stream.track_id,
        caps: stream.caps,
    }
}

#[tokio::test]
#[ignore = "needs a Wayland session, PipeWire, and a live HLS A/V feed (G2G_HLS_TEST_URL)"]
async fn av_stays_in_lipsync_over_a_long_run() {
    g2g_core::log::init_from_env();
    if std::env::var_os("WAYLAND_DISPLAY").is_none() {
        eprintln!("skipping: no WAYLAND_DISPLAY in env (run under a Wayland session)");
        return;
    }

    let master_url =
        std::env::var("G2G_HLS_TEST_URL").unwrap_or_else(|_| DEFAULT_FEED_URL.to_string());
    let Some(master_text) = http_get_text(&master_url) else {
        eprintln!("skipping: no HLS feed at {master_url} (is the server up? is curl installed?)");
        return;
    };
    let Ok(Playlist::Master(master)) = parse(&master_text) else {
        eprintln!("skipping: {master_url} is not a master playlist");
        return;
    };
    let Some(variant) = master.select(None) else {
        eprintln!("skipping: master playlist offers no variant");
        return;
    };
    let Some(fps) = feed_fps(&master_text) else {
        eprintln!("skipping: no FRAME-RATE in the master playlist, set G2G_FEED_FPS");
        return;
    };

    // The video rides the selected variant's own segments; the audio is either
    // muxed into them or a separate AUDIO-group rendition with its own playlist.
    let video_url = resolve_url(&master_url, &variant.uri);
    let audio_url = variant
        .audio_group
        .as_deref()
        .and_then(|g| {
            master
                .renditions_in(MediaType::Audio, g)
                .into_iter()
                .find_map(|r| r.uri.as_deref())
        })
        .map(|uri| resolve_url(&master_url, uri))
        .unwrap_or_else(|| video_url.clone());

    let (Some(video_init), Some(audio_init)) = (init_segment(&video_url), init_segment(&audio_url))
    else {
        eprintln!("skipping: the feed's renditions are not fMP4 (no #EXT-X-MAP init segment)");
        return;
    };
    let video_port = only_port(&video_init, true, "video");
    let audio_port = only_port(&audio_init, false, "audio");

    let soak_seconds: u64 = std::env::var("G2G_SOAK_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SOAK_SECONDS);
    eprintln!(
        "soak: {soak_seconds}s of {fps} fps from {master_url}\n  video {video_url}\n  audio {audio_url}"
    );

    let mut video_sink = WaylandSink::new()
        .with_title("g2g lipsync soak")
        .with_max_lateness_ns(MAX_LATENESS_NS);
    let mut audio_sink = PipeWireSink::new();
    let audio_clock = audio_sink.clock();
    let mut video_dec = FfmpegH264Dec::new().with_output_format(OutputFormat::Nv12);

    let start = Instant::now();
    {
        let mut video_src = HlsSrc::new(video_url.clone());
        let mut video_demux = Mp4DemuxN::new(vec![video_port]);
        let mut audio_src = HlsSrc::new(audio_url.clone());
        let mut audio_demux = Mp4DemuxN::new(vec![audio_port]);
        let mut audio_dec = FfmpegAudioDec::new();
        let mut convert = AudioConvert::auto();
        let mut resample = AudioResample::auto();

        let mut graph: Graph<GraphNodeRef> = Graph::new();
        let vsrc = graph.add_source(GraphNodeRef::source_ref(&mut video_src));
        let vdemux = graph.add_demux(GraphNodeRef::demux_ref(&mut video_demux), 1);
        let vdec = graph.add_transform(GraphNodeRef::element_ref(&mut video_dec));
        let vsink = graph.add_sink(GraphNodeRef::element_ref(&mut video_sink));
        graph.link(vsrc, vdemux.input()).unwrap();
        graph.link(vdemux.out(0), vdec).unwrap();
        graph.link(vdec, vsink).unwrap();

        let asrc = graph.add_source(GraphNodeRef::source_ref(&mut audio_src));
        let ademux = graph.add_demux(GraphNodeRef::demux_ref(&mut audio_demux), 1);
        let adec = graph.add_transform(GraphNodeRef::element_ref(&mut audio_dec));
        let aconv = graph.add_transform(GraphNodeRef::element_ref(&mut convert));
        let ares = graph.add_transform(GraphNodeRef::element_ref(&mut resample));
        let asink = graph.add_sink(GraphNodeRef::element_ref(&mut audio_sink));
        graph.link(asrc, ademux.input()).unwrap();
        graph.link(ademux.out(0), adec).unwrap();
        graph.link(adec, aconv).unwrap();
        graph.link(aconv, ares).unwrap();
        graph.link(ares, asink).unwrap();

        // A live feed never ends, so the soak window is the timeout: it expiring
        // is the expected outcome, and the graph returning early is the feed
        // dropping out mid-run.
        let budget = Duration::from_secs_f64(soak_seconds as f64 + STARTUP_ALLOWANCE_SECONDS);
        let clock = MonotonicClock;
        match tokio::time::timeout(budget, run_graph(graph, &clock, LINK_CAPACITY)).await {
            Err(_) => {}
            Ok(Ok(stats)) => panic!(
                "the feed ended after {:.1}s, before the {soak_seconds}s soak window: {stats:?}",
                start.elapsed().as_secs_f64()
            ),
            Ok(Err(e)) => panic!(
                "pipeline failed after {:.1}s: {e:?}",
                start.elapsed().as_secs_f64()
            ),
        }
    }
    let elapsed = start.elapsed().as_secs_f64();

    // (1) The audio sink's DriftClock won election and the video sink is paced
    // against that very clock, not a fallback (its own WaylandClock, which it
    // offers at the lower Provider tier).
    let sync = video_sink
        .clock_sync()
        .expect("the runner handed the video sink an elected clock");
    let elected: &Arc<dyn PipelineClock + Send + Sync> = &sync.clock;
    let audio: Arc<dyn PipelineClock + Send + Sync> = audio_clock.clone();
    assert!(
        Arc::ptr_eq(elected, &audio),
        "video is not paced against the audio DriftClock: elected clock reads {}, \
         the audio clock {}, the monotonic wall clock {}",
        elected.now_ns(),
        audio.now_ns(),
        MonotonicClock.now_ns(),
    );

    // (2) One presented frame per feed frame period, over the run past startup,
    // and nothing lost between the decoder and the screen.
    let presented = video_sink.frames_presented();
    let decoded = video_dec.decoded_count();
    let nominal = elapsed * fps;
    let least = (elapsed - STARTUP_ALLOWANCE_SECONDS).max(0.0) * fps * (1.0 - FRAME_RATE_TOLERANCE);
    let most = nominal * (1.0 + FRAME_RATE_TOLERANCE);
    let late_dropped = video_sink.late_dropped();
    let mut errors = video_sink.deadline_error_samples();
    let slope = audio_clock.slope();
    let observations = audio_clock.observations();

    // Both summary lines print before the first assertion: a failing run is the
    // one whose numbers are worth reading, and an assert would cut them off.
    let deadline = deadline_error_stats(&mut errors);
    eprintln!(
        "soak summary: elapsed={elapsed:.1}s decoded={decoded} presented={presented} \
         nominal={nominal:.0} late_dropped={late_dropped} deadline_samples={} \
         audio_clock slope={slope:.6} observations={observations}",
        errors.len()
    );
    match &deadline {
        Some(d) => eprintln!(
            "deadline error (ms): p5={:.1} p50={:.1} p95={:.1} | \
             first-quarter median={:.1} last-quarter median={:.1} drift={:.1}",
            d.p5 as f64 / 1e6,
            d.p50 as f64 / 1e6,
            d.p95 as f64 / 1e6,
            d.first_median as f64 / 1e6,
            d.last_median as f64 / 1e6,
            d.drift() as f64 / 1e6,
        ),
        None => eprintln!("deadline error: no samples, the sink presented unpaced"),
    }

    assert!(
        presented as f64 >= least,
        "presented {presented} frames, at least {least:.0} expected over {elapsed:.1}s at {fps} fps",
    );
    assert!(
        presented as f64 <= most,
        "presented {presented} frames, more than the {fps} fps feed carries in {elapsed:.1}s",
    );
    let unaccounted = decoded.saturating_sub(presented + late_dropped);
    assert!(
        unaccounted <= IN_FLIGHT_FRAME_ALLOWANCE,
        "{unaccounted} of {decoded} decoded frames reached neither the screen nor the drop \
         count, past the {IN_FLIGHT_FRAME_ALLOWANCE} still in flight at shutdown",
    );

    // (3) Late drops stay near zero.
    let drop_budget = nominal * LATE_DROP_TOLERANCE;
    assert!(
        late_dropped as f64 <= drop_budget,
        "{late_dropped} frames dropped past the {MAX_LATENESS_NS} ns lateness bound \
         (budget {drop_budget:.1})",
    );

    // (4) The presentation deadline error is bounded and does not drift: the
    // median over the first quarter of the run must match the last quarter's.
    let deadline = deadline.expect("no deadline-error samples: the sink presented unpaced");
    let worst_tail = deadline.worst_tail();
    assert!(
        worst_tail < DEADLINE_ERROR_P95_BOUND_NS,
        "deadline error p95 is {worst_tail} ns, past the {DEADLINE_ERROR_P95_BOUND_NS} ns bound",
    );
    let drift = deadline.drift();
    assert!(
        drift.abs() < DEADLINE_ERROR_DRIFT_BOUND_NS,
        "deadline error drifted {drift} ns from the first quarter of the run to the last, \
         past the {DEADLINE_ERROR_DRIFT_BOUND_NS} ns bound: video is walking away from audio",
    );

    // (5) The audio clock tracked a real ~1.0x playout rate throughout.
    assert!(
        observations >= MIN_DRIFT_CLOCK_OBSERVATIONS,
        "audio clock took {observations} observations; discipline did not run",
    );
    assert!(
        (slope - 1.0).abs() < DRIFT_CLOCK_SLOPE_TOLERANCE,
        "audio clock slope {slope} is not the ~1.0 a real-time playout should fit",
    );
}
