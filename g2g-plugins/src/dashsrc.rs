//! DASH source (DashSrc, `dash` feature): fetches an MPD manifest, selects a
//! Representation (simple bandwidth-capped ABR), and streams its fMP4 init +
//! media segments downstream as a `Caps::ByteStream{IsoBmff}` for `fmp4demux`,
//! then `Eos`. The [`mpd`](crate::mpd) parser does the manifest work; this adds
//! the fetching (via `reqwest`, shared with [`HlsSrc`](crate::hlssrc)) and the
//! `SegmentTemplate` `$Number$` / `$Time$` addressing.
//!
//! `SegmentList` is also supported: an explicit ordered list of
//! `<SegmentURL>` entries, each a `@media` URL and/or a `mediaRange` byte range
//! of the `BaseURL` resource, with an `<Initialization>` (`sourceURL` / `range`).
//! A range-only entry fetches just its sub-range with an HTTP `Range` request,
//! the DASH analog of HLS `#EXT-X-BYTERANGE`, so a single-file CMAF DASH stream
//! plays.
//!
//! Live (`type="dynamic"`): the MPD is reloaded on its refresh period, and
//! every addressing mode but `SegmentBase` gets its available segment window
//! from the wall clock against `availabilityStartTime` + `Period@start`, bounded
//! by `timeShiftBufferDepth`: a `SegmentTimeline` or `SegmentList` entry listed
//! ahead of its availability time is held back like a `@duration` segment.
//! Playback starts `suggestedPresentationDelay` behind the live edge
//! (see [`presentation_delay_secs`]) rather than replaying the whole window. An
//! `@availabilityTimeOffset` moves each segment's availability that many seconds
//! ahead of its nominal completion, so a chunked low-latency packager's newest
//! segment is fetched while it is still being written.
//!
//! That wall clock is the server's: the local clock plus the offset measured
//! from the first `UTCTiming` source whose scheme is supported (`direct`,
//! `http-head`, `http-xsdate`, `http-iso`, `http-ntp`, `ntp`, `sntp`) and that
//! answers, the next one being tried when one fails. It is re-measured every 30
//! minutes, or every 30 seconds after a failed measurement, as gst's
//! `dashdemux` does.
//!
//! Segment times are period-relative presentation times: a template's
//! `@presentationTimeOffset` is already off them (`$Time$` URLs keep the media
//! value), so seek targets and the boundary `Segment` are on the same timeline
//! whatever media timestamps the Period uses.
//!
//! Multi-period: the Periods play through back to back, each with its own
//! `BaseURL` / template / Representation choice. Crossing a boundary emits a
//! [`PipelinePacket::Segment`] whose `base` is the media played before it and
//! whose `time` is 0: running time continues across the boundary while stream
//! time restarts with the new Period's own timeline.
//!
//! CMAF chunked consumption (`low-latency`): a segment response is read as
//! a stream and each complete chunk (`styp` / `moof`+`mdat`) is pushed downstream
//! as it arrives, so the media of a segment the packager is still writing flows at
//! chunk latency instead of segment latency. `@availabilityTimeOffset` decides
//! *when* such a segment is fetched, this decides how it is consumed.
//!
//! Scope: one `DataFrame` per segment (per chunk under `low-latency`);
//! `SegmentBase` (`sidx`-indexed single resource), `SegmentList` and both
//! `SegmentTemplate` profiles.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::log::LogSource;
use g2g_core::runtime::{SeekController, SourceLoop};
use g2g_core::time::ticks_to_ns;
use g2g_core::{
    g2g_debug, g2g_warn, BusHandle, ByteStreamEncoding, Caps, CapsConstraint, CapsSet,
    ConfigureOutcome, ElementMetadata, G2gError, OutputSink, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, Seek, Segment,
};

use crate::abr::BandwidthEstimator;
use crate::dashclock::{local_now_unix, RequestSpan, ServerClock};
use crate::fetch::{
    body_read_err, byte_frame, get_bytes_retrying, get_range_bytes_retrying, get_response_retrying,
    get_text, get_text_retrying, is_origin_failure, resolve_url, FetchRetry, MAX_MANIFEST_BYTES,
    MAX_SEGMENT_BYTES,
};
use crate::fmp4::CmafChunker;
use crate::mpd::{
    live_start_offset, parse, parse_sidx, ByteRange, LiveEdge, Mpd, Representation, ResolvedSegment,
};

/// Fallback distance behind the live edge when the MPD suggests none, in ms.
/// DASH has no equivalent of the HLS three-target-duration rule (RFC 8216
/// §6.3.3): DASH-IF leaves the choice to the client and says only that
/// `suggestedPresentationDelay` wins when present. GStreamer `dashdemux`
/// defaults its `presentation-delay` property to 10s, so match that.
const DEFAULT_PRESENTATION_DELAY_MS: u64 = 10_000;

// no retry, so a wrong URL fails the start at once
async fn fetch_manifest(
    client: &reqwest::Client,
    url: &str,
) -> Result<(Mpd, RequestSpan), G2gError> {
    let sent_unix = local_now_unix();
    let text = get_text(client, url, MAX_MANIFEST_BYTES).await?;
    parse_manifest(&text, sent_unix)
}

async fn reload_manifest(
    client: &reqwest::Client,
    url: &str,
) -> Result<(Mpd, RequestSpan), G2gError> {
    let sent_unix = local_now_unix();
    let text = get_text_retrying(client, url, MAX_MANIFEST_BYTES, FetchRetry::BUDGETED).await?;
    parse_manifest(&text, sent_unix)
}

fn parse_manifest(text: &str, sent_unix: f64) -> Result<(Mpd, RequestSpan), G2gError> {
    let request = RequestSpan {
        sent_unix,
        answered_unix: local_now_unix(),
    };
    let mpd = parse(text).map_err(|_| G2gError::CapsMismatch)?;
    Ok((mpd, request))
}

/// Convert a time in `timescale` units to nanoseconds.
fn to_ns(time: u64, timescale: u32) -> u64 {
    ticks_to_ns(time, timescale.max(1))
}

#[derive(Debug, Clone, Copy)]
struct PlayedSegment {
    time: u64,
    timescale: u32,
}

impl PlayedSegment {
    // rounded up, so the segment just played stays played in a finer timescale
    fn time_in(self, timescale: u32) -> u64 {
        let rescaled =
            (u128::from(self.time) * u128::from(timescale)).div_ceil(u128::from(self.timescale));
        u64::try_from(rescaled).unwrap_or(u64::MAX)
    }
}

/// Resolve a segment / init URL against the base. An empty URL means the piece
/// is a byte range of the `BaseURL` resource itself (a pure byte-range
/// `SegmentList` / `Initialization`), so the base is fetched directly.
fn seg_url(base: &str, url: &str) -> String {
    if url.is_empty() {
        String::from(base)
    } else {
        resolve_url(base, url)
    }
}

/// Fetch a segment, issuing a `Range` request when the segment carries a byte
/// sub-range (single-file CMAF), else fetching the whole resource.
async fn fetch_segment(
    client: &reqwest::Client,
    url: &str,
    range: Option<ByteRange>,
    retry: FetchRetry,
) -> Result<Vec<u8>, G2gError> {
    match range {
        Some(r) => {
            get_range_bytes_retrying(client, url, r.offset, r.length, MAX_SEGMENT_BYTES, retry)
                .await
        }
        None => get_bytes_retrying(client, url, MAX_SEGMENT_BYTES, retry).await,
    }
}

/// Consume a segment response chunk by chunk, pushing each complete CMAF
/// chunk (`styp` / `moof`+`mdat`) downstream as it arrives instead of waiting for
/// the whole body, which is what a low-latency packager writing an open segment
/// serves. Returns the total body length (what the ABR estimator measures). The
/// frames pushed concatenate to exactly the response body, so the demuxer sees
/// the same byte stream the whole-response path hands it.
async fn stream_segment_chunks(
    client: &reqwest::Client,
    url: &str,
    out: &mut dyn OutputSink,
    sequence: &mut u64,
    retry: FetchRetry,
) -> Result<usize, G2gError> {
    let mut resp = get_response_retrying(client, url, MAX_SEGMENT_BYTES, retry).await?;
    let mut chunker = CmafChunker::new(MAX_SEGMENT_BYTES);
    let mut total = 0usize;
    while let Some(bytes) = resp.chunk().await.map_err(|e| body_read_err(url, e))? {
        total = total.saturating_add(bytes.len());
        chunker.feed(&bytes)?;
        while let Some(chunk) = chunker.next_chunk()? {
            out.push(PipelinePacket::DataFrame(byte_frame(chunk, *sequence)))
                .await?;
            *sequence += 1;
        }
    }
    // A response that ends mid-chunk (or carries no `mdat` at all) still forwards
    // its bytes: the demuxer, not the fetch loop, judges the box structure.
    if let Some(tail) = chunker.finish() {
        out.push(PipelinePacket::DataFrame(byte_frame(tail, *sequence)))
            .await?;
        *sequence += 1;
    }
    Ok(total)
}

/// Resolve a Representation's addressing into the run loop's working set: the
/// ordered segments, the timescale, and the init descriptor. A `SegmentBase`
/// representation fetches + parses its `sidx` here (the only async case); the
/// `sidx` carries the authoritative timescale. Used for the initial pick and on
/// every ABR switch.
///
/// A live Representation instead takes its segments from the wall-clock window
/// (`live`), which is empty until the Period's first segment is available: a
/// `SegmentTimeline` or `SegmentList` entry is held back until its availability
/// time just like a `@duration` segment.
async fn load_rep(
    client: &reqwest::Client,
    base: &str,
    rep: &Representation,
    total_secs: f64,
    live: Option<&LiveEdge>,
    now_unix: f64,
) -> Result<
    (
        Vec<ResolvedSegment>,
        u32,
        Option<(String, Option<ByteRange>)>,
    ),
    G2gError,
> {
    let init = rep.init();
    if let Some(index_range) = rep.segment_base().map(|sb| sb.index_range) {
        let idx_bytes = fetch_segment(
            client,
            &seg_url(base, ""),
            Some(index_range),
            FetchRetry::BUDGETED,
        )
        .await?;
        let sidx = parse_sidx(&idx_bytes).ok_or(G2gError::CapsMismatch)?;
        return Ok((
            sidx.subsegments(index_range.offset),
            sidx.timescale.max(1),
            init,
        ));
    }
    if let Some(segs) =
        live.and_then(|edge| rep.live_segments(edge, now_unix, period_end_secs(total_secs)))
    {
        return Ok((segs, rep.timescale(), init));
    }
    Ok((rep.resolved_segments(total_secs), rep.timescale(), init))
}

/// The declared presentation span of Period `idx` in seconds: its `@duration`,
/// else the gap to the next Period's `@start`, else what remains of
/// `mediaPresentationDuration` after its `@start`. A Period's `@duration`
/// template covers only its own span, so this (not the whole presentation) is
/// what its segment count is computed from.
fn period_span_secs(mpd: &Mpd, idx: usize) -> f64 {
    let period = &mpd.periods[idx];
    if let Some(d) = period.duration_secs {
        return d.max(0.0);
    }
    if let Some(next) = mpd.periods.get(idx + 1) {
        let gap = next.start_secs - period.start_secs;
        if gap > 0.0 {
            return gap;
        }
    }
    (mpd.duration_secs - period.start_secs).max(0.0)
}

fn period_end_secs(span_secs: f64) -> Option<f64> {
    (span_secs > 0.0).then_some(span_secs)
}

/// The running-time span of Period `idx`: its declared span, else the media
/// actually played from it. The running-time offset the next Period's boundary
/// `Segment` carries.
fn period_span_ns(mpd: &Mpd, idx: usize, played_ns: u64) -> u64 {
    let declared = period_span_secs(mpd, idx);
    if declared > 0.0 {
        (declared * 1_000_000_000.0) as u64
    } else {
        played_ns
    }
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::dashsrc::DashSrc;
///
/// let source = DashSrc::new("https://example.com/live/manifest.mpd")
///     .with_low_latency()
///     .with_max_bandwidth(6_000_000);
/// ```
#[derive(Debug)]
pub struct DashSrc {
    url: String,
    /// ABR cap: select the highest-bandwidth Representation at or below this
    /// (0 = no cap, pick the highest available).
    max_bandwidth: u64,
    /// Live-MPD reload interval in ms (0 = derive from `minimumUpdatePeriod`).
    reload_interval_ms: u64,
    /// How far behind the live edge a dynamic manifest starts, in ms
    /// (0 = derive; see [`presentation_delay_secs`]).
    presentation_delay_ms: u64,
    /// Optional time-seek channel: resolves a TIME seek to the media
    /// segment whose start time precedes the target (the `SegmentRef.time` is
    /// already a stream-time in `timescale` units), flushes, re-emits the init
    /// segment, and resumes there. The CMAF/DASH segment-transition case.
    seek: Option<SeekController>,
    /// Throughput-driven ABR: when set, the run loop measures each segment
    /// download and re-selects the Representation whose bandwidth fits the
    /// estimate (under `max_bandwidth`), switching mid-stream (new init re-emitted,
    /// resuming after the last segment played). Off by default (a fixed up-front
    /// Representation).
    abr: bool,
    /// Duration-keyed prebuffer target in ms (0 = off): fetch this much media
    /// ahead before emitting, posting `Buffering` on the attached bus.
    prebuffer_ms: u64,
    /// Consume a segment chunk by chunk as the packager writes it instead
    /// of once its whole response has arrived. Off by default.
    low_latency: bool,
    bus: Option<BusHandle>,
    configured: bool,
}

impl DashSrc {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            max_bandwidth: 0,
            reload_interval_ms: 0,
            presentation_delay_ms: 0,
            seek: None,
            abr: false,
            prebuffer_ms: 0,
            low_latency: false,
            bus: None,
            configured: false,
        }
    }

    /// Consume each segment chunk by chunk as it arrives: the response
    /// body is read as a stream and every complete CMAF chunk
    /// (`styp` / `moof`+`mdat`) is pushed downstream immediately, so a segment a
    /// low-latency packager is still writing plays from its first chunk instead of
    /// its last. The byte stream downstream is unchanged, only its timing.
    ///
    /// Ignored for a byte-range segment (single-file `SegmentList` / `SegmentBase`,
    /// where a server may answer a `Range` with the whole resource) and while
    /// `prebuffer-ms` is set (prebuffering deliberately trades latency for
    /// robustness and owns emission order); those fetch whole responses.
    pub fn with_low_latency(mut self) -> Self {
        self.low_latency = true;
        self
    }

    /// Buffer this many milliseconds of media (summed segment durations)
    /// before emitting, and again after a flushing seek. `0` disables
    /// prebuffering. The duration-keyed sibling of `HttpSrc::prebuffer-bytes`.
    pub fn with_prebuffer_ms(mut self, ms: u64) -> Self {
        self.prebuffer_ms = ms;
        self
    }

    /// Attach the pipeline bus so prebuffering posts
    /// [`g2g_core::BusMessage::Buffering`] level reports.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// Enable throughput-driven ABR: measure each segment's download and
    /// re-select the Representation whose declared bandwidth fits the smoothed
    /// estimate (under any `max_bandwidth` cap), switching mid-stream and
    /// re-emitting the init segment on a change. Off by default. Shares the
    /// estimator with [`HlsSrc`](crate::hlssrc).
    pub fn with_abr(mut self) -> Self {
        self.abr = true;
        self
    }

    /// Make the source time-seekable: `run` polls `controller` before each
    /// segment fetch and, on a flushing seek, selects the segment containing the
    /// target (the last whose `$Time$` start precedes it), emits `Flush`, re-emits
    /// the init segment for a reset downstream demuxer, emits the post-flush
    /// `Segment`, and resumes there. The application keeps a clone to scrub.
    pub fn with_seek(mut self, controller: SeekController) -> Self {
        self.seek = Some(controller);
        self
    }

    /// Cap Representation selection to this bitrate (bits/sec); 0 picks the highest.
    pub fn with_max_bandwidth(mut self, bits_per_sec: u64) -> Self {
        self.max_bandwidth = bits_per_sec;
        self
    }

    /// Override the live-MPD reload interval (ms); 0 derives it from the MPD
    /// `minimumUpdatePeriod`.
    pub fn with_reload_interval_ms(mut self, ms: u64) -> Self {
        self.reload_interval_ms = ms;
        self
    }

    /// Override how far behind the live edge a dynamic manifest starts (ms);
    /// 0 derives it (see [`presentation_delay_secs`]). Larger is more robust
    /// against download stalls, smaller is lower latency.
    pub fn with_presentation_delay_ms(mut self, ms: u64) -> Self {
        self.presentation_delay_ms = ms;
        self
    }

    /// How far behind the live edge to start a dynamic manifest, in seconds:
    /// the `presentation-delay-ms` override when set, else the MPD's
    /// `suggestedPresentationDelay` (which DASH-IF says a client should prefer
    /// over any value of its own), else [`DEFAULT_PRESENTATION_DELAY_MS`]. The
    /// available window bounds it, so a delay past the start of the
    /// `timeShiftBufferDepth` window starts at the window front.
    fn presentation_delay_secs(&self, mpd: &Mpd) -> f64 {
        if self.presentation_delay_ms != 0 {
            return self.presentation_delay_ms as f64 / 1000.0;
        }
        mpd.suggested_presentation_delay_secs
            .unwrap_or(DEFAULT_PRESENTATION_DELAY_MS as f64 / 1000.0)
    }

    fn output_caps() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        }
    }
}

/// The index of the media segment containing `target_ns` (the last whose
/// `$Time$` start, converted from `timescale` units to ns, is at or before the
/// target) and that segment's start time in ns. Segments are time-ordered, so
/// the scan stops at the first start past the target. A target before the first
/// segment clamps to it; empty input returns `(0, 0)`.
fn segment_for_time(segs: &[ResolvedSegment], timescale: u32, target_ns: u64) -> (usize, u64) {
    let mut chosen = 0usize;
    let mut chosen_start = 0u64;
    for (idx, seg) in segs.iter().enumerate() {
        let start_ns = to_ns(seg.time, timescale);
        if start_ns <= target_ns {
            chosen = idx;
            chosen_start = start_ns;
        } else {
            break;
        }
    }
    (chosen, chosen_start)
}

impl LogSource for DashSrc {
    fn log_category(&self) -> &'static str {
        "dashsrc"
    }
}

impl SourceLoop for DashSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(Self::output_caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(
            Self::output_caps(),
        ))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let client = reqwest::Client::new();
            let cap = (self.max_bandwidth != 0).then_some(self.max_bandwidth);

            let (mut mpd, manifest_request) = fetch_manifest(&client, &self.url).await?;
            // every live availability is computed on the server's clock
            let mut clock = ServerClock::default();
            if mpd.dynamic {
                clock.poll(&client, &mpd, manifest_request).await;
            }

            let mut sequence = 0u64;
            // Duration-keyed prebuffer window (0 = pass-through); init segments
            // ride it with duration 0 so ordering survives an ABR re-init.
            let mut window =
                crate::segprebuf::SegmentPrebuffer::new(self.prebuffer_ms, self.bus.clone());
            // Fallback duration for the last segment of a list (no successor to
            // diff against): the previous segment's duration.
            let mut last_dur_ns = 0u64;
            let mut init_emitted = false;
            // Largest segment start time already played; on a live reload only
            // segments past it are new (SegmentTimeline times are monotonic).
            let mut last_played: Option<PlayedSegment> = None;
            let mut playing_rep_id: Option<String> = None;
            // ABR throughput estimate, persisted across live reloads. The effective
            // selection cap is the estimate (when ABR is on and a sample exists),
            // else the user `max_bandwidth` cap.
            let mut estimator = BandwidthEstimator::new();
            let sel_cap = |est: &BandwidthEstimator| {
                if self.abr {
                    est.effective_cap(self.max_bandwidth)
                } else {
                    cap
                }
            };
            // Multi-period state. `period_base_ns` is the running time already
            // played when the current Period starts; `period_played_ns` is what
            // this Period has contributed so far (the span fallback for a
            // manifest that declares neither @duration nor the next @start).
            let mut period_idx = 0usize;
            let mut period_base_ns = 0u64;
            let mut period_played_ns = 0u64;
            // A boundary was crossed: the next Period's first media is preceded
            // by a `Segment` carrying the running-time offset.
            let mut boundary_pending = false;
            // Live playback positions itself once per Period, not on every
            // manifest reload (a reload continues from `last_played`).
            let mut positioned = false;
            if mpd.dynamic {
                period_idx =
                    mpd.live_period_index(clock.now_unix(), self.presentation_delay_secs(&mpd));
            }
            'manifest: loop {
                let period = &mpd.periods[period_idx];
                // Segment URIs resolve against the Period's BaseURL, itself
                // resolved against the MPD BaseURL / the manifest URL.
                let mut base = match &mpd.base_url {
                    Some(b) => resolve_url(&self.url, b),
                    None => self.url.clone(),
                };
                if let Some(b) = &period.base_url {
                    base = resolve_url(&base, b);
                }
                let delay_secs = self.presentation_delay_secs(&mpd);
                let edge = mpd.live_edge(period, delay_secs);
                // `SegmentTemplate` ($Number$/$Time$, SegmentTimeline or @duration),
                // an explicit `SegmentList`, or a `sidx`-indexed `SegmentBase` all
                // resolve to one ordered segment list (see `load_rep`). Pick the
                // Representation fitting the current estimate (or the user cap).
                let rep = period
                    .select(sel_cap(&estimator))
                    .ok_or(G2gError::CapsMismatch)?;
                // a reload can pick another Representation than the one playing
                if playing_rep_id.as_deref() != Some(rep.id.as_str()) {
                    playing_rep_id = Some(rep.id.clone());
                    init_emitted = false;
                }
                let span_secs = period_span_secs(&mpd, period_idx);
                let loaded_unix = clock.now_unix();
                let (mut segs, mut timescale, mut init) =
                    load_rep(&client, &base, rep, span_secs, edge.as_ref(), loaded_unix).await?;
                // the next live Period can be listed before this one has ended
                let lists_period_end = |rep: &Representation, at_unix: f64| {
                    edge.as_ref().is_none_or(|e| {
                        rep.lists_period_end(e, at_unix, period_end_secs(span_secs))
                    })
                };
                let mut period_listed_whole = lists_period_end(rep, loaded_unix);

                // a restarted origin starts its timeline over
                let timeline_went_back = mpd.dynamic
                    && last_played.is_some_and(|played| {
                        segs.last()
                            .is_some_and(|newest| newest.time < played.time_in(timescale))
                    });
                if timeline_went_back {
                    g2g_warn!(
                        self,
                        "{} went back before the last segment played, restarting at its live edge",
                        self.url
                    );
                    out.push(PipelinePacket::Flush).await?;
                    period_idx = mpd.live_period_index(clock.now_unix(), delay_secs);
                    last_played = None;
                    last_dur_ns = 0;
                    period_played_ns = 0;
                    init_emitted = false;
                    positioned = false;
                    continue 'manifest;
                }

                // A new Period restarts stream time at its own first segment
                // while running time carries on where the previous one ended.
                if boundary_pending && !segs.is_empty() {
                    let start_ns = to_ns(segs[0].time, timescale);
                    out.push(PipelinePacket::Segment(Segment {
                        base: period_base_ns,
                        start: start_ns,
                        position: start_ns,
                        time: 0,
                        ..Segment::new()
                    }))
                    .await?;
                    boundary_pending = false;
                }

                // Live: start near the live edge rather than replaying the whole
                // available window (the wall-clock window, or every listed
                // segment when the manifest has no availabilityStartTime).
                let mut idx = 0usize;
                if mpd.dynamic && !positioned && !segs.is_empty() {
                    idx = live_start_offset(&segs, timescale, (delay_secs * 1e9) as u64);
                    positioned = true;
                }
                loop {
                    // Apply a pending flushing time seek before the next fetch:
                    // jump to the segment containing the target, flush, and re-emit
                    // the init segment (a downstream demuxer reset on the flush
                    // needs its moov again).
                    if let Some(seek) = self.seek.as_ref().and_then(|c| c.take_pending()) {
                        if seek.is_flush() {
                            let (target_idx, start_ns) =
                                segment_for_time(&segs, timescale, seek.start);
                            // Queued lookahead is pre-seek media: drop it (and
                            // re-arm the prebuffer fill) before flushing.
                            window.clear();
                            out.push(PipelinePacket::Flush).await?;
                            idx = target_idx;
                            // Jumped by index; clear the reload-dedup watermark so
                            // the target segment is not skipped as "already played".
                            last_played = None;
                            init_emitted = false;
                            out.push(PipelinePacket::Segment(Segment::for_flush_seek(
                                &Seek::flush_to(start_ns),
                                None,
                            )))
                            .await?;
                        }
                        continue; // re-evaluate from the repositioned index
                    }

                    if !init_emitted {
                        if let Some((init_url, init_range)) = &init {
                            let url = seg_url(&base, init_url);
                            g2g_debug!(self, "init segment {url}");
                            let bytes =
                                fetch_segment(&client, &url, *init_range, FetchRetry::BUDGETED)
                                    .await?;
                            if !bytes.is_empty() {
                                window.admit(bytes, 0);
                            }
                        }
                        init_emitted = true;
                    }

                    // Fetch phase: pull segments into the window while it is
                    // below its duration target (or empty) and segments remain.
                    if idx < segs.len() && window.wants_fetch() {
                        // Bytes + elapsed of the segment just fetched, for the ABR
                        // estimator (None when this index was skipped on a live reload).
                        let mut measured: Option<(usize, u64)> = None;
                        {
                            let seg = &segs[idx];
                            if last_played.is_none_or(|played| seg.time > played.time_in(timescale))
                            {
                                // Play time of this segment: the start-time delta to
                                // its successor; the last segment reuses the previous
                                // duration (no successor to diff against).
                                let duration_ns = match segs.get(idx + 1) {
                                    Some(n) if n.time > seg.time => {
                                        to_ns(n.time - seg.time, timescale)
                                    }
                                    _ => last_dur_ns,
                                };
                                last_dur_ns = duration_ns;
                                let url = seg_url(&base, &seg.url);
                                g2g_debug!(self, "segment {url}");
                                // a live origin answers 404 or 425 until it publishes the segment
                                let retry = FetchRetry {
                                    live_segment_duration: edge
                                        .is_some()
                                        .then(|| core::time::Duration::from_nanos(duration_ns)),
                                };
                                let t0 = g2g_core::metrics::monotonic_ns();
                                // Low latency: read the body as a stream and push
                                // each CMAF chunk as it lands. The prebuffer window
                                // is empty here whenever it is disabled, so pushing
                                // from inside the fetch keeps emission ordered.
                                let fetched = if self.low_latency
                                    && self.prebuffer_ms == 0
                                    && seg.byte_range.is_none()
                                {
                                    stream_segment_chunks(&client, &url, out, &mut sequence, retry)
                                        .await
                                } else {
                                    fetch_segment(&client, &url, seg.byte_range, retry)
                                        .await
                                        .map(|bytes| {
                                            let len = bytes.len();
                                            if !bytes.is_empty() {
                                                window.admit(bytes, duration_ns);
                                            }
                                            len
                                        })
                                };
                                last_played = Some(PlayedSegment {
                                    time: seg.time,
                                    timescale,
                                });
                                period_played_ns = period_played_ns.saturating_add(duration_ns);
                                match fetched {
                                    Ok(len) => {
                                        measured = Some((
                                            len,
                                            g2g_core::metrics::monotonic_ns().saturating_sub(t0),
                                        ))
                                    }
                                    Err(error) if mpd.dynamic && is_origin_failure(&error) => {
                                        g2g_warn!(
                                            self,
                                            "{url} failed with {error:?}, skipping it and reloading the manifest"
                                        );
                                        idx = segs.len();
                                        continue;
                                    }
                                    Err(error) => return Err(error),
                                }
                            }
                        }
                        idx += 1;

                        // ABR: feed the measured throughput and, if the best-fitting
                        // Representation changed, switch to it (re-resolve its segments
                        // / init, re-emit the init). A live list resolved now can start
                        // later than the old one, so the walk restarts at its front and
                        // `last_played` skips what was played. The `seg` borrow above has
                        // ended, so reassigning is safe.
                        if self.abr {
                            if let Some((len, elapsed)) = measured {
                                estimator.sample(len, elapsed);
                                if let Some(best) =
                                    mpd.periods[period_idx].select(sel_cap(&estimator))
                                {
                                    if playing_rep_id.as_deref() != Some(best.id.as_str()) {
                                        playing_rep_id = Some(best.id.clone());
                                        let switched_unix = clock.now_unix();
                                        let (s, ts, ini) = load_rep(
                                            &client,
                                            &base,
                                            best,
                                            span_secs,
                                            edge.as_ref(),
                                            switched_unix,
                                        )
                                        .await?;
                                        period_listed_whole = lists_period_end(best, switched_unix);
                                        segs = s;
                                        timescale = ts;
                                        init = ini;
                                        init_emitted = false;
                                        idx = 0;
                                    }
                                }
                            }
                        }
                        continue;
                    }

                    // Emit phase: push the window front downstream. An empty
                    // window with nothing left to fetch ends this pass (live
                    // reload or end of manifest).
                    if let Some(bytes) = window.pop() {
                        out.push(PipelinePacket::DataFrame(byte_frame(bytes, sequence)))
                            .await?;
                        sequence += 1;
                        continue;
                    }
                    break;
                }

                // This Period is played out. A following Period continues the
                // presentation: carry the running time over, restart the
                // per-Period state, and re-enter with the boundary pending.
                if period_idx + 1 < mpd.periods.len() && period_listed_whole {
                    period_base_ns = period_base_ns.saturating_add(period_span_ns(
                        &mpd,
                        period_idx,
                        period_played_ns,
                    ));
                    period_idx += 1;
                    period_played_ns = 0;
                    last_played = None;
                    last_dur_ns = 0;
                    init_emitted = false;
                    positioned = false;
                    boundary_pending = true;
                    continue 'manifest;
                }

                if !mpd.dynamic {
                    break; // static (VOD, or the final live update) ends the stream
                }
                // Live: wait the update period, then refetch the manifest.
                let interval_ms = if self.reload_interval_ms != 0 {
                    self.reload_interval_ms
                } else {
                    (mpd.minimum_update_period_secs.unwrap_or(1.0) * 1000.0) as u64
                };
                tokio::time::sleep(core::time::Duration::from_millis(interval_ms.max(1))).await;
                let playing_start_secs = mpd.periods[period_idx].start_secs;
                let manifest_request;
                (mpd, manifest_request) = reload_manifest(&client, &self.url).await?;
                if mpd.dynamic {
                    clock.poll(&client, &mpd, manifest_request).await;
                }
                // Periods that aged out of the window drop off the front, so the
                // one playing is found by its start, not its old index.
                period_idx = mpd
                    .periods
                    .iter()
                    .position(|p| p.start_secs == playing_start_secs)
                    .unwrap_or(0);
            }

            out.push(PipelinePacket::Eos).await?;
            Ok(sequence)
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        DASHSRC_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "DASH source",
            "Source/Network",
            "Reads a DASH MPD and streams its segments",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" => {
                self.url = String::from(value.as_str().ok_or(PropError::Type)?);
                Ok(())
            }
            "max-bandwidth" => match value {
                PropValue::Uint(v) => {
                    self.max_bandwidth = v;
                    Ok(())
                }
                _ => Err(PropError::Type),
            },
            "reload-interval-ms" => match value {
                PropValue::Uint(v) => {
                    self.reload_interval_ms = v;
                    Ok(())
                }
                _ => Err(PropError::Type),
            },
            "presentation-delay-ms" => match value {
                PropValue::Uint(v) => {
                    self.presentation_delay_ms = v;
                    Ok(())
                }
                _ => Err(PropError::Type),
            },
            "abr" => match value {
                PropValue::Bool(v) => {
                    self.abr = v;
                    Ok(())
                }
                _ => Err(PropError::Type),
            },
            "prebuffer-ms" => match value {
                PropValue::Uint(v) => {
                    self.prebuffer_ms = v;
                    Ok(())
                }
                _ => Err(PropError::Type),
            },
            "low-latency" => match value {
                PropValue::Bool(v) => {
                    self.low_latency = v;
                    Ok(())
                }
                _ => Err(PropError::Type),
            },
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.url.clone())),
            "max-bandwidth" => Some(PropValue::Uint(self.max_bandwidth)),
            "reload-interval-ms" => Some(PropValue::Uint(self.reload_interval_ms)),
            "presentation-delay-ms" => Some(PropValue::Uint(self.presentation_delay_ms)),
            "abr" => Some(PropValue::Bool(self.abr)),
            "prebuffer-ms" => Some(PropValue::Uint(self.prebuffer_ms)),
            "low-latency" => Some(PropValue::Bool(self.low_latency)),
            _ => None,
        }
    }
}

static DASHSRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new("location", PropKind::Str, "MPD manifest URL (.mpd)"),
    PropertySpec::new(
        "max-bandwidth",
        PropKind::Uint,
        "ABR cap in bits/sec; 0 selects the highest-bandwidth Representation",
    ),
    PropertySpec::new(
        "reload-interval-ms",
        PropKind::Uint,
        "live-MPD reload interval in ms; 0 derives it from minimumUpdatePeriod",
    ),
    PropertySpec::new(
        "presentation-delay-ms",
        PropKind::Uint,
        "how far behind the live edge a dynamic MPD starts, ms; 0 uses suggestedPresentationDelay",
    )
    .with_default("0"),
    PropertySpec::new(
        "abr",
        PropKind::Bool,
        "throughput-driven Representation switching (measure downloads, re-select mid-stream)",
    )
    .with_default("false"),
    PropertySpec::new(
        "prebuffer-ms",
        PropKind::Uint,
        "media to buffer ahead before emitting, ms; posts Buffering bus messages (0 = off)",
    )
    .with_default("0"),
    PropertySpec::new(
        "low-latency",
        PropKind::Bool,
        "consume a segment chunk by chunk as the packager writes it (CMAF chunked transfer)",
    )
    .with_default("false"),
];
