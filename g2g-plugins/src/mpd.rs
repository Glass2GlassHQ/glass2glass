//! DASH MPD manifest parser (a subset of ISO/IEC 23009-1), driven by
//! [`DashSrc`](crate::dashsrc). Pure (no I/O), so it is fully unit-testable:
//! the wall-clock live math takes "now" as an argument.
//!
//! Scope: `SegmentTemplate` (both the `@duration` profile and `SegmentTimeline`,
//! with `$Number$` or `$Time$` addressing), `SegmentList`, and `SegmentBase`
//! byte ranges, over one or more `Period`s. Dynamic (live) manifests carry the
//! wall-clock attributes (`availabilityStartTime`, `timeShiftBufferDepth`,
//! `suggestedPresentationDelay`) that [`LiveEdge`] turns into an available
//! segment window. Attribute inheritance (geometry / codecs / the
//! `SegmentTemplate` itself declared on the `AdaptationSet` and shared by its
//! `Representation`s) is resolved by walking ancestors.
//!
//! The two `SegmentTemplate` timing offsets are modelled: `@presentationTimeOffset`
//! is the media-timeline instant that lines up with the start of the Period, so
//! `$Time$` addressing keeps the media value while a [`ResolvedSegment`] carries
//! period-relative presentation time; `@availabilityTimeOffset` publishes a
//! segment that many seconds before its nominal completion (the low-latency
//! chunked knob), clamped to one segment duration so a manifest cannot claim
//! media that does not exist yet.

use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::time::ticks_to_ns;
use roxmltree::{Document, Node};

use crate::bereader::{take_be_u16, take_be_u32, take_be_u64};
use crate::hls::select_by_bandwidth;
pub use crate::hls::ByteRange;
use crate::xmlutil::days_from_civil;

/// One selectable Representation (a single quality rendition).
#[derive(Debug, Clone, PartialEq)]
pub struct Representation {
    pub id: String,
    pub bandwidth: u64,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub codecs: Option<String>,
    pub mime_type: Option<String>,
    /// How this Representation's segments are addressed.
    pub source: SegmentSource,
}

/// How a Representation addresses its segments.
#[derive(Debug, Clone, PartialEq)]
pub enum SegmentSource {
    /// `SegmentTemplate`: `$Number$` / `$Time$` URL synthesis.
    Template(SegmentTemplate),
    /// `SegmentList`: an explicit ordered list of segment URLs / byte ranges.
    List(SegmentList),
    /// `SegmentBase`: one resource whose subsegment byte ranges come from a
    /// `sidx` index box (`indexRange`), resolved by fetching + parsing it.
    Base(SegmentBase),
}

/// `SegmentBase`: a single-resource (single-file CMAF) Representation. The media
/// fragments are byte ranges of the `BaseURL` resource, discovered at run time by
/// fetching the `sidx` box at `index_range` and parsing it ([`parse_sidx`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentBase {
    /// Byte range of the `sidx` Segment Index box in the resource.
    pub index_range: ByteRange,
    /// `@timescale` (advisory; the `sidx` carries the authoritative one).
    pub timescale: u32,
    /// `<Initialization range>` byte range of the init segment (the `ftyp`+`moov`
    /// at the head of the resource); `None` when no `<Initialization>` is given.
    pub init_range: Option<ByteRange>,
    pub init_present: bool,
}

/// One entry parsed from a `sidx` box: a subsegment's byte size, its duration in
/// the `sidx` timescale, and whether it references a child `sidx` (hierarchical)
/// rather than media.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SidxEntry {
    pub size: u64,
    pub duration: u64,
    pub reference_type: bool,
}

/// A parsed `sidx` (Segment Index) box: the box's own byte size, the
/// `first_offset` (anchor-relative start of the first subsegment), the segment
/// timescale, and the per-subsegment entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sidx {
    pub box_size: u64,
    pub first_offset: u64,
    pub timescale: u32,
    pub entries: Vec<SidxEntry>,
}

/// `SegmentList`: an explicit ordered list of media segments, each a URL and/or
/// a `mediaRange` byte range of the `BaseURL` resource, plus an `Initialization`.
/// `@duration` / `@timescale` give per-segment timing (a nested `SegmentTimeline`
/// is a follow-up).
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentList {
    /// `<Initialization sourceURL>`, empty when the init is a byte range of the
    /// `BaseURL` itself; `init_present` distinguishes "no init element" from it.
    pub init_url: String,
    pub init_range: Option<ByteRange>,
    pub init_present: bool,
    pub duration: u64,
    pub timescale: u32,
    pub availability_time_offset: Option<f64>,
    pub segments: Vec<SegmentUrl>,
}

/// One `<SegmentURL>` in a `SegmentList`: a `@media` URL (empty = the `BaseURL`
/// resource itself) and an optional `mediaRange` byte sub-range of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentUrl {
    pub media: String,
    pub media_range: Option<ByteRange>,
}

/// A segment resolved from either addressing mode for the source loop: the URL
/// (template-expanded or list-explicit; empty means the `BaseURL` resource), an
/// optional byte range, and the segment start time in `timescale` units.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSegment {
    pub url: String,
    pub byte_range: Option<ByteRange>,
    /// Period-relative presentation start time: the segment's media time less
    /// the template's `@presentationTimeOffset`, so a Period always starts at 0
    /// however its media timeline is numbered.
    pub time: u64,
}

/// `SegmentTemplate` with `$Number$` / `$Time$` addressing.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentTemplate {
    /// Init-segment template (resolves `$RepresentationID$`).
    pub initialization: Option<String>,
    /// Media-segment template (resolves `$RepresentationID$`, `$Number$`, `$Time$`).
    pub media: String,
    pub start_number: u64,
    /// Segment duration in `timescale` units (the `@duration` profile; the
    /// `SegmentTimeline` carries its own per-entry durations instead).
    pub duration: u64,
    pub timescale: u32,
    /// `@presentationTimeOffset` in `timescale` units: the media-timeline instant
    /// that lines up with the start of the Period. Segment times (`$Time$`,
    /// `SegmentTimeline` `@t`) are media-timeline values, so this comes off them
    /// to get presentation time.
    pub presentation_time_offset: u64,
    /// `@availabilityTimeOffset` in seconds: how much earlier than its nominal
    /// completion a segment becomes available (chunked low-latency delivery).
    /// `INF` parses as infinity; [`live_available`] clamps the effective value to
    /// one segment duration.
    ///
    /// [`live_available`]: Self::live_available
    pub availability_time_offset: Option<f64>,
    /// `SegmentTimeline` `<S>` entries when present; empty for the `@duration`
    /// profile.
    pub timeline: Vec<TimelineEntry>,
}

/// Cap on segments materialized from one manifest. A real presentation has far
/// fewer (1M two-second segments is ~23 days); the bound stops an untrusted
/// `@r` repeat or a tiny `@duration` from forcing an unbounded allocation.
const MAX_SEGMENTS: u64 = 1 << 20;

/// One `SegmentTimeline` `<S>` entry: a media-timeline start time `t` (absent =
/// continue from the previous entry), a duration `d`, and its `r` repeats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimelineEntry {
    pub t: Option<u64>,
    pub d: u64,
    pub r: TimelineRepeat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimelineRepeat {
    Times(u64),
    // a negative `@r`
    UntilNextEntryOrPeriodEnd,
}

// a stretch of equal-length segments: the `@duration` profile, one `<S>` entry, or a `SegmentList`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SegmentRun {
    first_number: u64,
    start_time: u64,
    duration: u64,
    // `None` runs on until the wall clock stops it
    count: Option<u64>,
}

impl SegmentRun {
    fn segment(&self, index: u64) -> SegmentRef {
        SegmentRef {
            number: self.first_number.saturating_add(index),
            time: self
                .start_time
                .saturating_add(index.saturating_mul(self.duration)),
        }
    }
}

// the wall-clock position of a live Period and the availability rules applied to its segments
#[derive(Debug, Clone, Copy)]
struct LiveAvailability {
    elapsed_secs: f64,
    time_shift_secs: Option<f64>,
    availability_time_offset_secs: f64,
    timescale: u32,
    presentation_time_offset: u64,
}

impl LiveAvailability {
    fn new(
        edge: &LiveEdge,
        now_unix: f64,
        availability_time_offset: Option<f64>,
        timescale: u32,
        presentation_time_offset: u64,
    ) -> Self {
        Self {
            elapsed_secs: now_unix - edge.anchor_unix,
            time_shift_secs: edge.time_shift_secs,
            availability_time_offset_secs: availability_time_offset.unwrap_or(0.0),
            timescale: timescale.max(1),
            presentation_time_offset,
        }
    }

    // indices of the first segment inside `timeShiftBufferDepth` and of the newest complete one
    fn run_window(&self, run: &SegmentRun) -> Option<(u64, u64)> {
        if run.duration == 0 {
            return None;
        }
        let timescale = f64::from(self.timescale);
        let segment_secs = run.duration as f64 / timescale;
        let run_start_secs =
            run.start_time.saturating_sub(self.presentation_time_offset) as f64 / timescale;
        let elapsed = self.elapsed_secs - run_start_secs;
        // Early availability is capped at one segment duration: the in-progress
        // segment can be published chunked, but nothing beyond it exists yet, and
        // the cap also keeps the window from opening before the anchor.
        let early = self.availability_time_offset_secs.clamp(0.0, segment_secs);
        // `as u64` saturates at 0 for a negative / NaN elapsed, and the checked
        // decrement then reports "nothing available yet" instead of wrapping.
        let mut last = (((elapsed + early) / segment_secs) as u64).checked_sub(1)?;
        if let Some(count) = run.count {
            last = last.min(count.checked_sub(1)?);
        }
        let first = match self.time_shift_secs {
            // Fully inside the window: the segment starts no earlier than the
            // depth allows, so round the boundary up.
            Some(depth) => (((elapsed - depth) / segment_secs).ceil()) as u64,
            None => 0,
        };
        Some((first, last))
    }

    // at most `MAX_SEGMENTS`, so an ancient `availabilityStartTime` cannot force a huge allocation
    fn segments(&self, runs: &[SegmentRun]) -> Vec<SegmentRef> {
        let mut out = Vec::new();
        let mut newest = None;
        for run in runs {
            // a run that has not started publishing means no later one has either
            let Some((first, last)) = self.run_window(run) else {
                break;
            };
            newest = Some(run.segment(last));
            if first > last {
                continue;
            }
            let room = MAX_SEGMENTS.saturating_sub(out.len() as u64);
            let count = (last - first).saturating_add(1).min(room);
            out.extend((first..first.saturating_add(count)).map(|index| run.segment(index)));
        }
        // a depth shorter than the newest segment still offers that segment
        if out.is_empty() {
            out.extend(newest);
        }
        out
    }

    fn lists_final_segment(&self, runs: &[SegmentRun]) -> bool {
        let Some(run) = runs.last() else {
            return true;
        };
        let Some(final_index) = run.count.and_then(|count| count.checked_sub(1)) else {
            return false;
        };
        self.run_window(run)
            .is_some_and(|(_, last)| last == final_index)
    }
}

/// One resolved media segment: its `$Number$` and its `$Time$` (media-timeline
/// start time in `timescale` units, so it includes `@presentationTimeOffset`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentRef {
    pub number: u64,
    pub time: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Mpd {
    /// `BaseURL` declared directly on `MPD` (a Period declares its own).
    pub base_url: Option<String>,
    /// `mediaPresentationDuration` in seconds (for the VOD segment count).
    pub duration_secs: f64,
    /// `@type="dynamic"`: a live manifest, refetched until it turns static.
    pub dynamic: bool,
    /// `@minimumUpdatePeriod` in seconds: how often a live manifest is reloaded.
    pub minimum_update_period_secs: Option<f64>,
    /// `@availabilityStartTime` as unix seconds: wall-clock zero of a dynamic
    /// presentation, the anchor the live segment window is computed from.
    pub availability_start_unix: Option<f64>,
    /// `@timeShiftBufferDepth` in seconds: how far back of the live edge media
    /// stays available (`None` = unbounded).
    pub time_shift_buffer_depth_secs: Option<f64>,
    /// `@suggestedPresentationDelay` in seconds: how far behind the live edge
    /// the packager wants playback to sit.
    pub suggested_presentation_delay_secs: Option<f64>,
    /// `@maxSegmentDuration` in seconds.
    pub max_segment_duration_secs: Option<f64>,
    /// The presentation's Periods, in order. Always non-empty.
    pub periods: Vec<Period>,
    /// The `UTCTiming` elements, in manifest order.
    pub utc_timings: Vec<UtcTiming>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UtcTiming {
    pub scheme_id_uri: String,
    pub value: String,
}

/// One `Period`: a self-contained stretch of the presentation with its own
/// Representations, base URL and segment addressing. Consecutive Periods play
/// through back to back; each one's segment times restart at zero, so a
/// segment's presentation time is its time plus the Period's `start_secs`.
#[derive(Debug, Clone, PartialEq)]
pub struct Period {
    pub id: Option<String>,
    /// `@start` in seconds. Absent `@start`s accumulate from the previous
    /// Period's start + duration, so this is always resolved.
    pub start_secs: f64,
    /// `@duration` in seconds, when declared.
    pub duration_secs: Option<f64>,
    /// The nearest `BaseURL` inside this Period (Period / AdaptationSet /
    /// Representation level), resolved against the MPD base by the source.
    pub base_url: Option<String>,
    pub representations: Vec<Representation>,
}

/// The wall-clock parameters of a dynamic presentation, resolved for one Period.
/// Turns "now" into the window of segments a live Representation currently
/// offers, and the index playback should start at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LiveEdge {
    /// Unix seconds at which this Period's presentation time zero became
    /// available: `availabilityStartTime` + `Period@start`.
    pub anchor_unix: f64,
    /// `timeShiftBufferDepth` in seconds (`None` = the whole presentation stays
    /// available).
    pub time_shift_secs: Option<f64>,
    /// How far behind the live edge to start playback, in seconds.
    pub presentation_delay_secs: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpdError {
    /// XML did not parse, or no usable Representation was found.
    Invalid,
}

impl SegmentTemplate {
    /// Number of segments for a VOD presentation of `total_secs`.
    pub fn segment_count(&self, total_secs: f64) -> u64 {
        if self.duration == 0 || self.timescale == 0 {
            return 0;
        }
        let seg_secs = self.duration as f64 / self.timescale as f64;
        if seg_secs <= 0.0 {
            return 0;
        }
        (total_secs / seg_secs).ceil() as u64
    }

    /// The init-segment URL template expanded for `rep_id`.
    pub fn init_url(&self, rep_id: &str) -> Option<String> {
        self.initialization
            .as_ref()
            .map(|t| expand(t, rep_id, None, None))
    }

    /// The media-segment URL template expanded for `rep_id` and a segment's
    /// `$Number$` / `$Time$`.
    pub fn media_url(&self, rep_id: &str, seg: SegmentRef) -> String {
        expand(&self.media, rep_id, Some(seg.number), Some(seg.time))
    }

    /// Period-relative presentation time of a media-timeline `time`, in
    /// `timescale` units: `@presentationTimeOffset` is the media instant that
    /// lines up with the Period start, so it comes off. Saturates, so a manifest
    /// listing times before its own offset clamps to the Period start.
    pub fn presentation_time(&self, media_time: u64) -> u64 {
        media_time.saturating_sub(self.presentation_time_offset)
    }

    /// Nominal segment duration in seconds: `@duration` for that profile, else
    /// the first `SegmentTimeline` entry's `d`. `None` when neither is usable.
    pub fn nominal_segment_secs(&self) -> Option<f64> {
        let ts = self.timescale.max(1) as f64;
        let d = if self.duration != 0 {
            self.duration
        } else {
            self.timeline.first()?.d
        };
        (d != 0).then(|| d as f64 / ts)
    }

    /// The `@duration` live profile's currently available segment indices
    /// (0-based from `start_number`) at wall-clock `now_unix`: the earliest one
    /// still inside `timeShiftBufferDepth` and the newest available one. Segment
    /// `i` covers presentation time `[i*d, (i+1)*d)` and is complete `(i+1)*d`
    /// after the Period anchor, `@availabilityTimeOffset` seconds earlier when
    /// the packager publishes it chunked. A Period with an end (`period_secs`)
    /// offers nothing past its last segment. `None` before the first available
    /// segment exists, or for a `SegmentTimeline` (whose entries
    /// [`live_segments`] holds back by the same rule).
    ///
    /// [`live_segments`]: Self::live_segments
    pub fn live_available(
        &self,
        edge: &LiveEdge,
        now_unix: f64,
        period_secs: Option<f64>,
    ) -> Option<(u64, u64)> {
        if !self.timeline.is_empty() {
            return None;
        }
        let run = self.duration_run(period_secs)?;
        let (first, last) = self.live_availability(edge, now_unix).run_window(&run)?;
        Some((first.min(last), last))
    }

    fn final_index(&self, period_secs: Option<f64>) -> Option<u64> {
        self.segment_count(period_secs?).checked_sub(1)
    }

    fn live_availability(&self, edge: &LiveEdge, now_unix: f64) -> LiveAvailability {
        LiveAvailability::new(
            edge,
            now_unix,
            self.availability_time_offset,
            self.timescale,
            self.presentation_time_offset,
        )
    }

    fn duration_run(&self, period_secs: Option<f64>) -> Option<SegmentRun> {
        (self.duration != 0).then(|| SegmentRun {
            first_number: self.start_number,
            start_time: self.presentation_time_offset,
            duration: self.duration,
            count: self
                .final_index(period_secs)
                .map(|index| index.saturating_add(1)),
        })
    }

    // a negative `@r` repeats up to the next entry's `@t`, else to the Period end, else without end
    fn timeline_runs(&self, period_secs: Option<f64>) -> Vec<SegmentRun> {
        let period_end_time = period_secs.map(|secs| {
            self.presentation_time_offset
                .saturating_add((secs * f64::from(self.timescale.max(1))) as u64)
        });
        let mut runs = Vec::new();
        let mut number = self.start_number;
        let mut time = 0u64;
        for (position, entry) in self.timeline.iter().enumerate() {
            if let Some(t) = entry.t {
                time = t;
            }
            let count = match entry.r {
                TimelineRepeat::Times(repeats) => Some(repeats.saturating_add(1)),
                TimelineRepeat::UntilNextEntryOrPeriodEnd if entry.d == 0 => Some(1),
                TimelineRepeat::UntilNextEntryOrPeriodEnd => {
                    let end_time = match self.timeline.get(position + 1) {
                        Some(next) => next.t,
                        None => period_end_time,
                    };
                    end_time.map(|end| end.saturating_sub(time).div_ceil(entry.d))
                }
            };
            let run = SegmentRun {
                first_number: number,
                start_time: time,
                duration: entry.d,
                count,
            };
            let Some(count) = count else {
                // nothing after an open-ended repeat has a start time
                runs.push(run);
                break;
            };
            if count > 0 {
                runs.push(run);
            }
            number = number.saturating_add(count);
            time = time.saturating_add(count.saturating_mul(entry.d));
        }
        runs
    }

    fn live_runs(&self, period_secs: Option<f64>) -> Vec<SegmentRun> {
        if self.timeline.is_empty() {
            self.duration_run(period_secs).into_iter().collect()
        } else {
            self.timeline_runs(period_secs)
        }
    }

    pub fn live_offers_period_end(
        &self,
        edge: &LiveEdge,
        now_unix: f64,
        period_secs: Option<f64>,
    ) -> bool {
        let runs = self.live_runs(period_secs);
        if runs.is_empty() && self.timeline.is_empty() {
            return false;
        }
        self.live_availability(edge, now_unix)
            .lists_final_segment(&runs)
    }

    /// The live segments available at `now_unix`, newest window last, each
    /// carrying its `$Number$` and its media-timeline `$Time$`. The `@duration`
    /// profile's segment `i` starts at `@presentationTimeOffset + i*d`, and a
    /// `SegmentTimeline` entry is held back by the same availability rule until
    /// it is complete. Bounded by [`MAX_SEGMENTS`] so an ancient
    /// `availabilityStartTime` cannot force an unbounded allocation.
    pub fn live_segments(
        &self,
        edge: &LiveEdge,
        now_unix: f64,
        period_secs: Option<f64>,
    ) -> Vec<SegmentRef> {
        self.live_availability(edge, now_unix)
            .segments(&self.live_runs(period_secs))
    }

    /// The ordered media segments for a VOD presentation of `total_secs`. Driven
    /// by the `SegmentTimeline` when present, else by `@duration`. Each carries
    /// its `$Number$` (from `startNumber`) and `$Time$` (the media-timeline start
    /// time, so the `@duration` profile counts from `@presentationTimeOffset` and
    /// a `SegmentTimeline` uses its own `@t` values). A negative `@r` with neither
    /// a following `@t` nor a `total_secs` to stop it lists its first segment only.
    pub fn segments(&self, total_secs: f64) -> Vec<SegmentRef> {
        if self.timeline.is_empty() {
            let count = self.segment_count(total_secs).min(MAX_SEGMENTS);
            return self.duration_run(None).map_or_else(Vec::new, |run| {
                (0..count).map(|index| run.segment(index)).collect()
            });
        }
        let mut out = Vec::new();
        for run in self.timeline_runs((total_secs > 0.0).then_some(total_secs)) {
            let room = MAX_SEGMENTS.saturating_sub(out.len() as u64);
            let count = run.count.unwrap_or(1).min(room);
            out.extend((0..count).map(|index| run.segment(index)));
        }
        out
    }
}

impl SegmentList {
    fn live_run(&self) -> Option<SegmentRun> {
        (self.duration != 0 && !self.segments.is_empty()).then_some(SegmentRun {
            first_number: 0,
            start_time: 0,
            duration: self.duration,
            count: Some(self.segments.len() as u64),
        })
    }

    fn live_availability(&self, edge: &LiveEdge, now_unix: f64) -> LiveAvailability {
        LiveAvailability::new(
            edge,
            now_unix,
            self.availability_time_offset,
            self.timescale,
            0,
        )
    }
}

impl Sidx {
    /// Resolve the indexed subsegments to byte ranges + cumulative start times.
    /// `index_offset` is the byte offset of the `sidx` box in the resource (the
    /// `indexRange` start); media begins at `index_offset + box_size +
    /// first_offset`. Hierarchical references (`reference_type == 1`, a child
    /// `sidx`) are not media, so they advance the cursor but emit no segment.
    pub fn subsegments(&self, index_offset: u64) -> Vec<ResolvedSegment> {
        let mut pos = index_offset
            .saturating_add(self.box_size)
            .saturating_add(self.first_offset);
        let mut time = 0u64;
        let mut out = Vec::new();
        for e in &self.entries {
            if !e.reference_type {
                out.push(ResolvedSegment {
                    url: String::new(),
                    byte_range: Some(ByteRange {
                        offset: pos,
                        length: e.size,
                    }),
                    time,
                });
            }
            pos = pos.saturating_add(e.size);
            time = time.saturating_add(e.duration);
        }
        out
    }
}

impl Representation {
    /// Addressing-mode-agnostic segment timescale (>= 1). For `SegmentBase` this
    /// is the manifest `@timescale`; the authoritative one is in the `sidx`.
    pub fn timescale(&self) -> u32 {
        match &self.source {
            SegmentSource::Template(t) => t.timescale.max(1),
            SegmentSource::List(l) => l.timescale.max(1),
            SegmentSource::Base(b) => b.timescale.max(1),
        }
    }

    /// The init segment, if any: its URL (empty = the `BaseURL` resource) and an
    /// optional byte range. The source loop resolves the URL against the base.
    pub fn init(&self) -> Option<(String, Option<ByteRange>)> {
        match &self.source {
            SegmentSource::Template(t) => t.init_url(&self.id).map(|u| (u, None)),
            SegmentSource::List(l) => l.init_present.then(|| (l.init_url.clone(), l.init_range)),
            // SegmentBase init is a byte range of the BaseURL resource (empty URL).
            SegmentSource::Base(b) => b.init_present.then(|| (String::new(), b.init_range)),
        }
    }

    /// The ordered segments resolved for the source loop without I/O. Template
    /// synthesizes URLs by `$Number$` / `$Time$` (media time) while the resolved
    /// `time` is period-relative presentation time; List returns its explicit
    /// URLs / ranges with cumulative `@duration` start times. `SegmentBase`
    /// returns empty here: its subsegments need the fetched `sidx` (see
    /// [`segment_base`] and [`Sidx::subsegments`]).
    ///
    /// [`segment_base`]: Self::segment_base
    pub fn resolved_segments(&self, total_secs: f64) -> Vec<ResolvedSegment> {
        match &self.source {
            SegmentSource::Template(t) => t
                .segments(total_secs)
                .into_iter()
                .map(|s| ResolvedSegment {
                    url: t.media_url(&self.id, s),
                    byte_range: None,
                    time: t.presentation_time(s.time),
                })
                .collect(),
            SegmentSource::List(l) => {
                let mut out = Vec::new();
                // A SegmentList carries no media-timeline values (the entries are
                // URLs / byte ranges), so its synthesized times are already
                // period-relative and no offset applies.
                let mut time = 0u64;
                for su in &l.segments {
                    out.push(ResolvedSegment {
                        url: su.media.clone(),
                        byte_range: su.media_range,
                        time,
                    });
                    time = time.saturating_add(l.duration);
                }
                out
            }
            SegmentSource::Base(_) => Vec::new(),
        }
    }

    /// The media segments a dynamic Representation offers at wall-clock
    /// `now_unix`, empty before this Period's first segment is available: the
    /// `@duration` window, or the `SegmentTimeline` / `SegmentList` entries whose
    /// availability time has passed. `None` for `SegmentBase` and for a
    /// `SegmentList` without `@duration`, which list their own segments.
    pub fn live_segments(
        &self,
        edge: &LiveEdge,
        now_unix: f64,
        period_secs: Option<f64>,
    ) -> Option<Vec<ResolvedSegment>> {
        match &self.source {
            SegmentSource::Template(t) => Some(
                t.live_segments(edge, now_unix, period_secs)
                    .into_iter()
                    .map(|s| ResolvedSegment {
                        url: t.media_url(&self.id, s),
                        byte_range: None,
                        time: t.presentation_time(s.time),
                    })
                    .collect(),
            ),
            SegmentSource::List(l) => {
                let run = l.live_run()?;
                Some(
                    l.live_availability(edge, now_unix)
                        .segments(&[run])
                        .into_iter()
                        .map(|s| {
                            // the run numbers the list entries from 0
                            let entry = &l.segments[s.number as usize];
                            ResolvedSegment {
                                url: entry.media.clone(),
                                byte_range: entry.media_range,
                                time: s.time,
                            }
                        })
                        .collect(),
                )
            }
            SegmentSource::Base(_) => None,
        }
    }

    pub fn lists_period_end(
        &self,
        edge: &LiveEdge,
        now_unix: f64,
        period_secs: Option<f64>,
    ) -> bool {
        match &self.source {
            SegmentSource::Template(t) => t.live_offers_period_end(edge, now_unix, period_secs),
            SegmentSource::List(l) => l.live_run().is_none_or(|run| {
                l.live_availability(edge, now_unix)
                    .lists_final_segment(&[run])
            }),
            // an index names every segment it has
            SegmentSource::Base(_) => true,
        }
    }

    /// The `SegmentBase` when this Representation is `sidx`-indexed single-file;
    /// the source loop fetches `index_range`, parses the `sidx`, and builds the
    /// subsegment list. `None` for Template / List addressing.
    pub fn segment_base(&self) -> Option<&SegmentBase> {
        match &self.source {
            SegmentSource::Base(b) => Some(b),
            _ => None,
        }
    }

    /// The `SegmentTemplate` when this Representation uses template addressing
    /// (for inspection / tests); `None` otherwise.
    pub fn template(&self) -> Option<&SegmentTemplate> {
        match &self.source {
            SegmentSource::Template(t) => Some(t),
            _ => None,
        }
    }
}

/// Where a fresh player starts inside a live window: the earliest segment of
/// `segs` that still leaves `delay_ns` of media ahead of it, so playback follows
/// the live edge instead of replaying the whole window. Clamps to the window
/// front when the window is shorter than the delay. Works off segment start-time
/// deltas, so it serves both the `@duration` window and a `SegmentTimeline`.
pub fn live_start_offset(segs: &[ResolvedSegment], timescale: u32, delay_ns: u64) -> usize {
    // Duration of segment `i`: the delta to its successor, or (for the last
    // one) the delta from its predecessor.
    let dur_ns = |i: usize| -> u64 {
        let (a, b) = if i + 1 < segs.len() {
            (segs[i].time, segs[i + 1].time)
        } else if i > 0 {
            (segs[i - 1].time, segs[i].time)
        } else {
            return 0;
        };
        ticks_to_ns(b.saturating_sub(a), timescale.max(1))
    };
    let mut ahead_ns = 0u64;
    let mut start = 0usize;
    for i in (0..segs.len()).rev() {
        ahead_ns = ahead_ns.saturating_add(dur_ns(i));
        start = i;
        if ahead_ns >= delay_ns {
            break;
        }
    }
    start
}

/// Parse a `sidx` (Segment Index) box (ISO/IEC 14496-12). Untrusted input: every
/// field read is bounds-checked, so a malformed box / hostile `reference_count`
/// fails to `None` rather than over-reading or over-allocating.
pub fn parse_sidx(data: &[u8]) -> Option<Sidx> {
    // FullBox header: size(4) type(4) version(1) flags(3).
    let box_size = u32::from_be_bytes(data.get(0..4)?.try_into().ok()?) as u64;
    if data.get(4..8)? != b"sidx" {
        return None;
    }
    let version = *data.get(8)?;
    let mut p = 12usize; // skip the 3 flag bytes
    let _reference_id = take_be_u32(data, &mut p)?;
    let timescale = take_be_u32(data, &mut p)?;
    // earliest_presentation_time + first_offset: 32-bit in v0, 64-bit in v1.
    let first_offset = if version == 0 {
        let _ept = take_be_u32(data, &mut p)?;
        take_be_u32(data, &mut p)? as u64
    } else {
        let _ept = take_be_u64(data, &mut p)?;
        take_be_u64(data, &mut p)?
    };
    let _reserved = take_be_u16(data, &mut p)?;
    let reference_count = take_be_u16(data, &mut p)?;
    let mut entries = Vec::new();
    for _ in 0..reference_count {
        // reference_type(1) | referenced_size(31); subsegment_duration(32);
        // starts_with_SAP(1) | SAP_type(3) | SAP_delta_time(28).
        let w0 = take_be_u32(data, &mut p)?;
        let duration = take_be_u32(data, &mut p)? as u64;
        let _sap = take_be_u32(data, &mut p)?;
        entries.push(SidxEntry {
            reference_type: (w0 >> 31) & 1 == 1,
            size: (w0 & 0x7fff_ffff) as u64,
            duration,
        });
    }
    Some(Sidx {
        box_size,
        first_offset,
        timescale,
        entries,
    })
}

impl Period {
    /// Pick the highest-bandwidth Representation at or below `max_bandwidth`
    /// (the overall highest when `None`, the lowest when nothing fits).
    pub fn select(&self, max_bandwidth: Option<u64>) -> Option<&Representation> {
        select_by_bandwidth(&self.representations, |r| r.bandwidth, max_bandwidth)
    }
}

impl Mpd {
    pub fn live_period_index(&self, now_unix: f64, presentation_delay_secs: f64) -> usize {
        let Some(availability_start) = self.availability_start_unix else {
            return 0;
        };
        self.period_index_at(now_unix - presentation_delay_secs - availability_start)
    }

    // a time before the first Period lands in it
    pub fn period_index_at(&self, presentation_secs: f64) -> usize {
        self.periods
            .iter()
            .rposition(|p| p.start_secs <= presentation_secs)
            .unwrap_or(0)
    }

    /// The wall-clock live parameters for `period`, or `None` for a static
    /// manifest / one without an `availabilityStartTime`. `presentation_delay`
    /// is resolved by the caller (`suggestedPresentationDelay`, an override, or
    /// a segment-duration default).
    pub fn live_edge(&self, period: &Period, presentation_delay_secs: f64) -> Option<LiveEdge> {
        if !self.dynamic {
            return None;
        }
        Some(LiveEdge {
            anchor_unix: self.availability_start_unix? + period.start_secs,
            time_shift_secs: self.time_shift_buffer_depth_secs,
            presentation_delay_secs,
        })
    }
}

/// Parse an MPD manifest.
pub fn parse(xml: &str) -> Result<Mpd, MpdError> {
    let doc = Document::parse(xml).map_err(|_| MpdError::Invalid)?;
    let root = doc.root_element();

    let duration_secs = root
        .attribute("mediaPresentationDuration")
        .and_then(parse_iso_duration)
        .unwrap_or(0.0);
    let dynamic = root.attribute("type") == Some("dynamic");
    let minimum_update_period_secs = root
        .attribute("minimumUpdatePeriod")
        .and_then(parse_iso_duration);
    let base_url = root
        .children()
        .find(|n| n.is_element() && n.has_tag_name("BaseURL"))
        .and_then(|n| n.text())
        .map(|s| String::from(s.trim()));

    // Periods with no usable Representation (nothing addressable) are skipped
    // rather than failing the manifest; `@start` accumulates when absent.
    let mut periods: Vec<Period> = Vec::new();
    let mut previous: Option<PeriodSpan> = None;
    for node in root
        .children()
        .filter(|n| n.is_element() && n.has_tag_name("Period"))
    {
        let declared_start = node.attribute("start").and_then(parse_iso_duration);
        let period = parse_period(node, period_start_secs(declared_start, previous, dynamic)?);
        previous = Some(PeriodSpan {
            start_secs: period.start_secs,
            duration_secs: period.duration_secs,
        });
        if !period.representations.is_empty() {
            periods.push(period);
        }
    }

    if periods.is_empty() {
        return Err(MpdError::Invalid);
    }
    Ok(Mpd {
        base_url,
        duration_secs,
        dynamic,
        minimum_update_period_secs,
        availability_start_unix: root
            .attribute("availabilityStartTime")
            .and_then(parse_xs_datetime),
        time_shift_buffer_depth_secs: root
            .attribute("timeShiftBufferDepth")
            .and_then(parse_iso_duration),
        suggested_presentation_delay_secs: root
            .attribute("suggestedPresentationDelay")
            .and_then(parse_iso_duration),
        max_segment_duration_secs: root
            .attribute("maxSegmentDuration")
            .and_then(parse_iso_duration),
        periods,
        utc_timings: root
            .children()
            .filter(|n| n.is_element() && n.has_tag_name("UTCTiming"))
            .filter_map(|n| {
                Some(UtcTiming {
                    scheme_id_uri: String::from(n.attribute("schemeIdUri")?),
                    value: String::from(n.attribute("value")?),
                })
            })
            .collect(),
    })
}

#[derive(Debug, Clone, Copy)]
struct PeriodSpan {
    start_secs: f64,
    duration_secs: Option<f64>,
}

// gst rejects a start not after the one before, and an unknown start in a static MPD
fn period_start_secs(
    declared_start: Option<f64>,
    previous: Option<PeriodSpan>,
    dynamic: bool,
) -> Result<f64, MpdError> {
    match (declared_start, previous) {
        (Some(start), Some(previous)) if start <= previous.start_secs => Err(MpdError::Invalid),
        (Some(start), _) => Ok(start),
        (None, None) => Ok(0.0),
        (None, Some(previous)) => match previous.duration_secs {
            Some(duration) => Ok(previous.start_secs + duration),
            // a live manifest can list a Period before the one ahead of it has ended
            None if dynamic => Ok(previous.start_secs),
            None => Err(MpdError::Invalid),
        },
    }
}

/// Parse one `Period` and its Representations.
fn parse_period(node: Node, start_secs: f64) -> Period {
    let mut representations = Vec::new();
    for rep in node
        .descendants()
        .filter(|n| n.has_tag_name("Representation"))
    {
        let Some(id) = rep.attribute("id") else {
            continue;
        };
        let Some(source) = segment_source(rep) else {
            continue;
        };
        representations.push(Representation {
            id: String::from(id),
            bandwidth: inherited(rep, "bandwidth")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            width: inherited(rep, "width").and_then(|s| s.parse().ok()),
            height: inherited(rep, "height").and_then(|s| s.parse().ok()),
            codecs: inherited(rep, "codecs").map(String::from),
            mime_type: inherited(rep, "mimeType").map(String::from),
            source,
        });
    }
    Period {
        id: node.attribute("id").map(String::from),
        start_secs,
        duration_secs: node.attribute("duration").and_then(parse_iso_duration),
        base_url: node
            .descendants()
            .find(|n| n.has_tag_name("BaseURL"))
            .and_then(|n| n.text())
            .map(|s| String::from(s.trim())),
        representations,
    }
}

/// The addressing for a Representation: its nearest `SegmentList` (preferred when
/// present) or `SegmentTemplate`, searching its own children then ancestors'
/// (AdaptationSet / Period inheritance). `None` if neither is usable (e.g. a
/// `SegmentBase`-only Representation, a follow-up).
fn segment_source(rep: Node) -> Option<SegmentSource> {
    if let Some(sl) = rep.ancestors().find_map(|n| {
        n.children()
            .find(|c| c.is_element() && c.has_tag_name("SegmentList"))
    }) {
        return Some(SegmentSource::List(parse_segment_list(sl)));
    }
    if let Some(sb) = rep
        .ancestors()
        .find_map(|n| {
            n.children()
                .find(|c| c.is_element() && c.has_tag_name("SegmentBase"))
        })
        .and_then(parse_segment_base)
    {
        return Some(SegmentSource::Base(sb));
    }
    segment_template(rep).map(SegmentSource::Template)
}

/// Parse a `SegmentBase` element. Requires an `indexRange` (the `sidx` location);
/// without it there is no way to discover the subsegments, so it is not usable.
fn parse_segment_base(sb: Node) -> Option<SegmentBase> {
    let index_range = sb.attribute("indexRange").and_then(parse_dash_range)?;
    let init = sb
        .children()
        .find(|c| c.is_element() && c.has_tag_name("Initialization"));
    Some(SegmentBase {
        index_range,
        timescale: sb
            .attribute("timescale")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1),
        init_range: init
            .and_then(|n| n.attribute("range"))
            .and_then(parse_dash_range),
        init_present: init.is_some(),
    })
}

/// The nearest `SegmentTemplate` for a Representation (its own, else inherited
/// from an ancestor AdaptationSet / Period), parsed into a [`SegmentTemplate`].
/// Requires a `media` attribute (number addressing).
fn segment_template(rep: Node) -> Option<SegmentTemplate> {
    let st = rep.ancestors().find_map(|n| {
        n.children()
            .find(|c| c.is_element() && c.has_tag_name("SegmentTemplate"))
    })?;
    let timeline = st
        .children()
        .find(|c| c.is_element() && c.has_tag_name("SegmentTimeline"))
        .map(parse_timeline)
        .unwrap_or_default();
    Some(SegmentTemplate {
        initialization: st.attribute("initialization").map(String::from),
        media: String::from(st.attribute("media")?),
        start_number: st
            .attribute("startNumber")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1),
        duration: st
            .attribute("duration")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0),
        timescale: st
            .attribute("timescale")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1),
        presentation_time_offset: st
            .attribute("presentationTimeOffset")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0),
        availability_time_offset: st
            .attribute("availabilityTimeOffset")
            .and_then(parse_availability_time_offset),
        timeline,
    })
}

/// Parse `@availabilityTimeOffset`: non-negative decimal seconds, or the literal
/// `INF` (available from the availability start time). Manifest input, so a
/// negative / `NaN` / unparsable value is dropped rather than shifting the
/// availability window.
fn parse_availability_time_offset(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("INF") {
        return Some(f64::INFINITY);
    }
    let secs: f64 = s.parse().ok()?;
    (secs.is_finite() && secs >= 0.0).then_some(secs)
}

/// Parse a `SegmentList` element into its init + ordered `<SegmentURL>` entries.
fn parse_segment_list(sl: Node) -> SegmentList {
    let init = sl
        .children()
        .find(|c| c.is_element() && c.has_tag_name("Initialization"));
    let segments = sl
        .children()
        .filter(|c| c.is_element() && c.has_tag_name("SegmentURL"))
        .map(|s| SegmentUrl {
            media: s.attribute("media").map(String::from).unwrap_or_default(),
            media_range: s.attribute("mediaRange").and_then(parse_dash_range),
        })
        .collect();
    SegmentList {
        init_url: init
            .and_then(|n| n.attribute("sourceURL"))
            .map(String::from)
            .unwrap_or_default(),
        init_range: init
            .and_then(|n| n.attribute("range"))
            .and_then(parse_dash_range),
        init_present: init.is_some(),
        duration: sl
            .attribute("duration")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0),
        timescale: sl
            .attribute("timescale")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1),
        availability_time_offset: sl
            .attribute("availabilityTimeOffset")
            .and_then(parse_availability_time_offset),
        segments,
    }
}

/// A DASH byte range `"start-end"` (inclusive end) -> [`ByteRange`]. A reversed
/// or malformed range yields `None` (the segment then fetches whole).
fn parse_dash_range(s: &str) -> Option<ByteRange> {
    let (start, end) = s.trim().split_once('-')?;
    let offset: u64 = start.trim().parse().ok()?;
    let end: u64 = end.trim().parse().ok()?;
    Some(ByteRange {
        offset,
        length: end.checked_sub(offset)?.checked_add(1)?,
    })
}

/// Parse a `SegmentTimeline`'s `<S>` entries. A negative `@r` repeats until the
/// next entry's `@t` or the Period end; an unparsable one counts as no repeats.
fn parse_timeline(tl: Node) -> Vec<TimelineEntry> {
    tl.children()
        .filter(|c| c.is_element() && c.has_tag_name("S"))
        .map(|s| TimelineEntry {
            t: s.attribute("t").and_then(|v| v.parse().ok()),
            d: s.attribute("d").and_then(|v| v.parse().ok()).unwrap_or(0),
            r: s.attribute("r")
                .map_or(TimelineRepeat::Times(0), parse_timeline_repeat),
        })
        .collect()
}

fn parse_timeline_repeat(value: &str) -> TimelineRepeat {
    if let Ok(repeats) = value.parse::<u64>() {
        return TimelineRepeat::Times(repeats);
    }
    if value.parse::<i64>().is_ok_and(|repeats| repeats < 0) {
        return TimelineRepeat::UntilNextEntryOrPeriodEnd;
    }
    TimelineRepeat::Times(0)
}

/// An attribute on `node` or the nearest ancestor that carries it.
fn inherited<'a>(node: Node<'a, '_>, name: &str) -> Option<&'a str> {
    node.ancestors().find_map(|n| n.attribute(name))
}

/// Expand a `SegmentTemplate` URL: `$$` -> `$`, `$RepresentationID$` -> id,
/// `$Number$` / `$Number%0Nd$` -> the segment number, `$Time$` / `$Time%0Nd$` ->
/// the segment start time, both honoring a `%0Nd` zero-pad width.
fn expand(tmpl: &str, rep_id: &str, number: Option<u64>, time: Option<u64>) -> String {
    let mut out = String::new();
    for (i, part) in tmpl.split('$').enumerate() {
        if i % 2 == 0 {
            out.push_str(part);
        } else if part.is_empty() {
            out.push('$'); // "$$"
        } else if part == "RepresentationID" {
            out.push_str(rep_id);
        } else if let Some(fmt) = part.strip_prefix("Number") {
            out.push_str(&format_number(fmt, number.unwrap_or(0)));
        } else if let Some(fmt) = part.strip_prefix("Time") {
            out.push_str(&format_number(fmt, time.unwrap_or(0)));
        }
        // any other identifier is dropped
    }
    out
}

/// Format a `$Number...$` value, honoring a `%0Nd` zero-pad width.
fn format_number(fmt: &str, n: u64) -> String {
    if let Some(width) = fmt
        .strip_prefix("%0")
        .and_then(|s| s.strip_suffix('d'))
        .and_then(|s| s.parse::<usize>().ok())
    {
        alloc::format!("{n:0width$}")
    } else {
        alloc::format!("{n}")
    }
}

/// Parse an ISO 8601 duration's time component (`PT1H2M3.5S`) to seconds. The
/// date part (years/months/days before `T`) is not expected in media durations
/// and is ignored.
fn parse_iso_duration(s: &str) -> Option<f64> {
    let time = s.split_once('T').map(|(_, t)| t).unwrap_or("");
    let mut secs = 0.0f64;
    let mut num = String::new();
    for ch in time.chars() {
        match ch {
            '0'..='9' | '.' => num.push(ch),
            'H' => {
                secs += num.parse::<f64>().ok()? * 3600.0;
                num.clear();
            }
            'M' => {
                secs += num.parse::<f64>().ok()? * 60.0;
                num.clear();
            }
            'S' => {
                secs += num.parse::<f64>().ok()?;
                num.clear();
            }
            _ => return None,
        }
    }
    Some(secs)
}

/// Parse an `xs:dateTime` (`2026-08-03T20:45:39.541Z`, or with a `+HH:MM` /
/// `-HH:MM` zone offset, or naive = UTC) to unix seconds. Manifest input, so
/// every field is range-checked and anything malformed yields `None` rather
/// than a bogus live-edge anchor.
pub(crate) fn parse_xs_datetime(s: &str) -> Option<f64> {
    let (date, rest) = s.trim().split_once('T')?;
    let mut ymd = date.split('-');
    let year: i64 = ymd.next()?.parse().ok()?;
    let month: u32 = ymd.next()?.parse().ok()?;
    let day: u32 = ymd.next()?.parse().ok()?;
    if ymd.next().is_some() || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }

    // Split the zone designator off the time: 'Z', or a sign that is not the
    // first character (the time itself never starts with one).
    let (time, zone_secs) = match rest.strip_suffix('Z') {
        Some(t) => (t, 0.0),
        None => match rest.rfind(['+', '-']).filter(|&i| i > 0) {
            Some(i) => (&rest[..i], parse_zone_offset(&rest[i..])?),
            None => (rest, 0.0),
        },
    };
    let mut hms = time.split(':');
    let hour: u32 = hms.next()?.parse().ok()?;
    let minute: u32 = hms.next()?.parse().ok()?;
    let second: f64 = hms.next()?.parse().ok()?;
    if hms.next().is_some() || hour > 23 || minute > 59 || !(0.0..61.0).contains(&second) {
        return None;
    }

    let days = days_from_civil(year, month, day);
    Some(days as f64 * 86_400.0 + hour as f64 * 3600.0 + minute as f64 * 60.0 + second - zone_secs)
}

/// Parse a `+HH:MM` / `-HH:MM` zone designator to seconds east of UTC.
fn parse_zone_offset(s: &str) -> Option<f64> {
    let sign = if s.starts_with('-') { -1.0 } else { 1.0 };
    let (h, m) = s[1..].split_once(':')?;
    let h: u32 = h.parse().ok()?;
    let m: u32 = m.parse().ok()?;
    if h > 23 || m > 59 {
        return None;
    }
    Some(sign * (h as f64 * 3600.0 + m as f64 * 60.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    const MPD: &str = r#"<?xml version="1.0"?>
<MPD mediaPresentationDuration="PT0H0M12.0S" type="static">
  <Period>
    <AdaptationSet mimeType="video/mp4" codecs="avc1.4d401f">
      <SegmentTemplate initialization="init-$RepresentationID$.mp4"
                       media="seg-$RepresentationID$-$Number%03d$.m4s"
                       startNumber="1" duration="4000" timescale="1000"/>
      <Representation id="low" bandwidth="800000" width="640" height="360"/>
      <Representation id="high" bandwidth="2400000" width="1280" height="720"/>
    </AdaptationSet>
  </Period>
</MPD>"#;

    #[test]
    fn parses_representations_with_inherited_template_and_geometry() {
        let mpd = parse(MPD).unwrap();
        assert!((mpd.duration_secs - 12.0).abs() < 1e-6);
        assert_eq!(mpd.periods[0].representations.len(), 2);
        let high = mpd.periods[0].select(None).unwrap();
        assert_eq!(high.id, "high");
        assert_eq!(high.bandwidth, 2_400_000);
        assert_eq!(high.width, Some(1280));
        // codecs inherited from the AdaptationSet
        assert_eq!(high.codecs.as_deref(), Some("avc1.4d401f"));
        assert_eq!(high.template().unwrap().timescale, 1000);
        assert_eq!(high.template().unwrap().duration, 4000);
    }

    #[test]
    fn abr_caps_selection() {
        let mpd = parse(MPD).unwrap();
        assert_eq!(mpd.periods[0].select(Some(1_000_000)).unwrap().id, "low");
        assert_eq!(mpd.periods[0].select(Some(1)).unwrap().id, "low"); // fallback to lowest
    }

    #[test]
    fn segment_count_and_url_templating() {
        let mpd = parse(MPD).unwrap();
        let rep = mpd.periods[0].select(None).unwrap();
        let template = rep.template().unwrap();
        // 12s / 4s = 3 segments.
        assert_eq!(template.segment_count(mpd.duration_secs), 3);
        assert_eq!(template.init_url(&rep.id).as_deref(), Some("init-high.mp4"));
        assert_eq!(
            template.media_url(&rep.id, SegmentRef { number: 1, time: 0 }),
            "seg-high-001.m4s"
        );
        assert_eq!(
            template.media_url(
                &rep.id,
                SegmentRef {
                    number: 12,
                    time: 0
                }
            ),
            "seg-high-012.m4s"
        );
        // The @duration profile yields startNumber.. with cumulative $Time$.
        let segs = template.segments(mpd.duration_secs);
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[0], SegmentRef { number: 1, time: 0 });
        assert_eq!(
            segs[2],
            SegmentRef {
                number: 3,
                time: 8000
            }
        );
    }

    #[test]
    fn adversarial_segment_counts_are_capped() {
        // A crafted @r repeat must not expand to billions of segments.
        let timeline = SegmentTemplate {
            initialization: None,
            media: String::from("seg-$Number$.m4s"),
            start_number: 1,
            duration: 0,
            timescale: 1000,
            presentation_time_offset: 0,
            availability_time_offset: None,
            timeline: Vec::from([TimelineEntry {
                t: Some(0),
                d: 1000,
                r: TimelineRepeat::Times(u64::MAX),
            }]),
        };
        assert_eq!(timeline.segments(10.0).len() as u64, MAX_SEGMENTS);

        // A near-zero @duration must not expand the @duration profile either.
        let tiny = SegmentTemplate {
            initialization: None,
            media: String::from("seg-$Number$.m4s"),
            start_number: 1,
            duration: 1,
            timescale: u32::MAX,
            presentation_time_offset: 0,
            availability_time_offset: None,
            timeline: Vec::new(),
        };
        assert_eq!(tiny.segments(1.0e9).len() as u64, MAX_SEGMENTS);
    }

    const TIMELINE_MPD: &str = r#"<?xml version="1.0"?>
<MPD type="static">
  <Period>
    <AdaptationSet mimeType="video/mp4">
      <SegmentTemplate initialization="init.mp4" media="seg-$Time$.m4s"
                       startNumber="1" timescale="90000">
        <SegmentTimeline>
          <S t="0" d="180000" r="2"/>
          <S d="90000"/>
        </SegmentTimeline>
      </SegmentTemplate>
      <Representation id="v0" bandwidth="1000000" width="640" height="360"/>
    </AdaptationSet>
  </Period>
</MPD>"#;

    #[test]
    fn segment_timeline_expands_repeats_with_time_addressing() {
        let mpd = parse(TIMELINE_MPD).unwrap();
        let rep = mpd.periods[0].select(None).unwrap();
        // <S r="2"> = 3 segments of d=180000, then one of d=90000.
        let segs = rep.template().unwrap().segments(mpd.duration_secs);
        assert_eq!(segs.len(), 4);
        assert_eq!(segs[0], SegmentRef { number: 1, time: 0 });
        assert_eq!(
            segs[1],
            SegmentRef {
                number: 2,
                time: 180_000
            }
        );
        assert_eq!(
            segs[2],
            SegmentRef {
                number: 3,
                time: 360_000
            }
        );
        assert_eq!(
            segs[3],
            SegmentRef {
                number: 4,
                time: 540_000
            }
        );
        // $Time$ addressing uses each segment's start time.
        assert_eq!(
            rep.template().unwrap().media_url(&rep.id, segs[2]),
            "seg-360000.m4s"
        );
    }

    // two Periods of one `$Number$` Representation, `first` and `second` holding their timing attributes
    fn two_period_mpd(kind: &str, first: &str, second: &str) -> String {
        let period = |timing: &str| {
            format!(
                r#"<Period {timing}><AdaptationSet>
                  <SegmentTemplate media="$Number$.m4s" duration="1" timescale="1"/>
                  <Representation id="r" bandwidth="1"/>
                </AdaptationSet></Period>"#
            )
        };
        format!(
            r#"<MPD type="{kind}" mediaPresentationDuration="PT20S">{}{}</MPD>"#,
            period(first),
            period(second)
        )
    }

    #[test]
    fn a_period_without_a_start_follows_the_one_before() {
        let mpd = parse(&two_period_mpd(
            "static",
            r#"start="PT2S" duration="PT5S""#,
            "",
        ))
        .unwrap();
        assert_eq!(mpd.periods[1].start_secs, 2.0 + 5.0);
    }

    #[test]
    fn a_static_period_whose_start_is_unknown_fails_the_manifest() {
        assert_eq!(
            parse(&two_period_mpd("static", r#"start="PT2S""#, "")),
            Err(MpdError::Invalid)
        );
    }

    #[test]
    fn a_live_period_whose_start_is_unknown_keeps_the_start_before_it() {
        let mpd = parse(&two_period_mpd("dynamic", r#"start="PT2S""#, "")).unwrap();
        assert_eq!(mpd.periods[1].start_secs, 2.0);
    }

    #[test]
    fn a_period_starting_before_the_one_ahead_fails_the_manifest() {
        for kind in ["static", "dynamic"] {
            assert_eq!(
                parse(&two_period_mpd(kind, r#"start="PT5S""#, r#"start="PT5S""#)),
                Err(MpdError::Invalid),
                "{kind}"
            );
        }
    }

    #[test]
    fn segment_timeline_t_attribute_resets_the_running_time() {
        let xml = r#"<MPD type="static"><Period><AdaptationSet>
          <SegmentTemplate media="$Time$.m4s" timescale="1000">
            <SegmentTimeline><S t="0" d="1000"/><S t="5000" d="1000" r="1"/></SegmentTimeline>
          </SegmentTemplate>
          <Representation id="r" bandwidth="1"/>
        </AdaptationSet></Period></MPD>"#;
        let mpd = parse(xml).unwrap();
        let segs = mpd.periods[0].representations[0]
            .template()
            .unwrap()
            .segments(0.0);
        // A gap: the second <S t="5000"> jumps the running time past 1000.
        assert_eq!(
            segs,
            [
                SegmentRef { number: 1, time: 0 },
                SegmentRef {
                    number: 2,
                    time: 5000
                },
                SegmentRef {
                    number: 3,
                    time: 6000
                },
            ]
        );
    }

    /// A `SegmentTimeline` on a media timeline that starts at the offset rather
    /// than at zero (90kHz, 1s segments, the Period beginning at media time 1s).
    const PTO_MPD: &str = r#"<?xml version="1.0"?>
<MPD type="static">
  <Period>
    <AdaptationSet mimeType="video/mp4">
      <SegmentTemplate initialization="init.mp4" media="seg-$Time$.m4s"
                       startNumber="1" timescale="90000" presentationTimeOffset="90000">
        <SegmentTimeline>
          <S t="90000" d="90000" r="2"/>
        </SegmentTimeline>
      </SegmentTemplate>
      <Representation id="v0" bandwidth="1000000"/>
    </AdaptationSet>
  </Period>
</MPD>"#;

    #[test]
    fn presentation_time_offset_maps_media_time_onto_the_period_timeline() {
        let mpd = parse(PTO_MPD).unwrap();
        let rep = &mpd.periods[0].representations[0];
        let tmpl = rep.template().unwrap();
        assert_eq!(tmpl.presentation_time_offset, 90_000);

        // $Time$ addressing stays on the media timeline.
        let segs = tmpl.segments(0.0);
        assert_eq!(segs[0].time, 90_000);
        assert_eq!(tmpl.media_url(&rep.id, segs[0]), "seg-90000.m4s");
        assert_eq!(tmpl.media_url(&rep.id, segs[2]), "seg-270000.m4s");

        // The resolved segments are period-relative: the Period starts at 0.
        let resolved = rep.resolved_segments(0.0);
        assert_eq!(resolved.len(), 3);
        assert_eq!(resolved[0].time, 0);
        assert_eq!(resolved[1].time, 90_000);
        assert_eq!(resolved[2].time, 180_000);
        assert_eq!(resolved[0].url, "seg-90000.m4s");

        // A time listed before the offset clamps to the Period start.
        assert_eq!(tmpl.presentation_time(1000), 0);
    }

    #[test]
    fn duration_profile_counts_time_from_the_presentation_time_offset() {
        let xml = r#"<MPD type="static" mediaPresentationDuration="PT3S"><Period><AdaptationSet>
          <SegmentTemplate media="seg-$Time$.m4s" duration="1000" timescale="1000"
                           presentationTimeOffset="5000"/>
          <Representation id="r" bandwidth="1"/>
        </AdaptationSet></Period></MPD>"#;
        let mpd = parse(xml).unwrap();
        let rep = &mpd.periods[0].representations[0];
        let segs = rep.template().unwrap().segments(mpd.duration_secs);
        assert_eq!(
            segs.iter().map(|s| s.time).collect::<Vec<_>>(),
            [5000, 6000, 7000],
            "$Time$ counts from the offset"
        );
        assert_eq!(
            rep.resolved_segments(mpd.duration_secs)
                .iter()
                .map(|s| s.time)
                .collect::<Vec<_>>(),
            [0, 1000, 2000],
            "presentation time still starts at the Period"
        );
    }

    #[test]
    fn missing_timing_offsets_default_to_none() {
        let tmpl = parse(TIMELINE_MPD).unwrap().periods[0].representations[0]
            .template()
            .unwrap()
            .clone();
        assert_eq!(tmpl.presentation_time_offset, 0);
        assert_eq!(tmpl.availability_time_offset, None);
        assert_eq!(tmpl.presentation_time(180_000), 180_000);
    }

    #[test]
    fn availability_time_offset_forms() {
        assert_eq!(parse_availability_time_offset("0.967"), Some(0.967));
        assert_eq!(parse_availability_time_offset(" 2 "), Some(2.0));
        assert_eq!(parse_availability_time_offset("INF"), Some(f64::INFINITY));
        for bad in ["-1", "NaN", "inf inity", "", "1s"] {
            assert_eq!(parse_availability_time_offset(bad), None, "{bad} rejected");
        }
    }

    #[test]
    fn parses_segment_list_with_byte_ranges() {
        // Single-file CMAF: init + three fragments are byte ranges of one BaseURL
        // resource (empty @media), each <SegmentURL> a mediaRange.
        let xml = r#"<MPD type="static"><Period><AdaptationSet mimeType="video/mp4">
          <BaseURL>all.m4s</BaseURL>
          <SegmentList duration="1000" timescale="1000">
            <Initialization range="0-799"/>
            <SegmentURL mediaRange="800-999"/>
            <SegmentURL mediaRange="1000-1299"/>
            <SegmentURL mediaRange="1300-1449"/>
          </SegmentList>
          <Representation id="v0" bandwidth="1000000" width="64" height="48"/>
        </AdaptationSet></Period></MPD>"#;
        let mpd = parse(xml).unwrap();
        let rep = mpd.periods[0].select(None).unwrap();
        assert_eq!(rep.timescale(), 1000);

        // Init is a byte range of the BaseURL (empty URL, range present).
        let (init_url, init_range) = rep.init().unwrap();
        assert_eq!(init_url, "");
        assert_eq!(
            init_range,
            Some(ByteRange {
                offset: 0,
                length: 800
            })
        );

        let segs = rep.resolved_segments(mpd.duration_secs);
        assert_eq!(segs.len(), 3);
        // Range "800-999" (inclusive) -> offset 800, length 200; times accumulate
        // by @duration (0, 1000, 2000 in timescale units).
        assert_eq!(
            segs[0],
            ResolvedSegment {
                url: String::new(),
                byte_range: Some(ByteRange {
                    offset: 800,
                    length: 200
                }),
                time: 0,
            }
        );
        assert_eq!(
            segs[1].byte_range,
            Some(ByteRange {
                offset: 1000,
                length: 300
            })
        );
        assert_eq!(segs[1].time, 1000);
        assert_eq!(
            segs[2].byte_range,
            Some(ByteRange {
                offset: 1300,
                length: 150
            })
        );
        assert_eq!(segs[2].time, 2000);
        // A SegmentList Representation has no template.
        assert!(rep.template().is_none());
    }

    #[test]
    fn parses_segment_list_with_explicit_media_urls() {
        let xml = r#"<MPD type="static"><Period><AdaptationSet>
          <SegmentList duration="1000" timescale="1000">
            <Initialization sourceURL="init.mp4"/>
            <SegmentURL media="seg0.m4s"/>
            <SegmentURL media="seg1.m4s"/>
          </SegmentList>
          <Representation id="v0" bandwidth="1"/>
        </AdaptationSet></Period></MPD>"#;
        let mpd = parse(xml).unwrap();
        let rep = mpd.periods[0].select(None).unwrap();
        assert_eq!(rep.init(), Some((String::from("init.mp4"), None)));
        let segs = rep.resolved_segments(mpd.duration_secs);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].url, "seg0.m4s");
        assert_eq!(segs[0].byte_range, None);
        assert_eq!(segs[1].url, "seg1.m4s");
        assert_eq!(segs[1].time, 1000);
    }

    /// Build a version-0 `sidx` box from `(referenced_size, subsegment_duration)`
    /// entries (all media references, SAP set).
    fn build_sidx(timescale: u32, entries: &[(u32, u32)]) -> Vec<u8> {
        let mut b = Vec::new();
        let box_size = 32 + 12 * entries.len() as u32;
        b.extend_from_slice(&box_size.to_be_bytes());
        b.extend_from_slice(b"sidx");
        b.extend_from_slice(&[0, 0, 0, 0]); // version 0 + flags
        b.extend_from_slice(&1u32.to_be_bytes()); // reference_ID
        b.extend_from_slice(&timescale.to_be_bytes());
        b.extend_from_slice(&0u32.to_be_bytes()); // earliest_presentation_time
        b.extend_from_slice(&0u32.to_be_bytes()); // first_offset
        b.extend_from_slice(&0u16.to_be_bytes()); // reserved
        b.extend_from_slice(&(entries.len() as u16).to_be_bytes());
        for &(size, dur) in entries {
            b.extend_from_slice(&(size & 0x7fff_ffff).to_be_bytes()); // reference_type 0
            b.extend_from_slice(&dur.to_be_bytes());
            b.extend_from_slice(&0x9000_0000u32.to_be_bytes()); // starts_with_SAP, type 1
        }
        b
    }

    #[test]
    fn parses_sidx_and_resolves_subsegment_ranges() {
        let sidx_bytes = build_sidx(1000, &[(200, 1000), (300, 1000), (150, 1000)]);
        let sidx = parse_sidx(&sidx_bytes).unwrap();
        assert_eq!(sidx.timescale, 1000);
        assert_eq!(sidx.first_offset, 0);
        assert_eq!(sidx.box_size as usize, sidx_bytes.len());
        assert_eq!(sidx.entries.len(), 3);
        assert_eq!(
            sidx.entries[0],
            SidxEntry {
                size: 200,
                duration: 1000,
                reference_type: false
            }
        );

        // The sidx sits at byte `index_offset`; media starts right after it
        // (box_size + first_offset). Ranges accumulate by size, times by duration.
        let index_offset = 800u64;
        let media_start = index_offset + sidx.box_size; // first_offset 0
        let segs = sidx.subsegments(index_offset);
        assert_eq!(segs.len(), 3);
        assert_eq!(
            segs[0].byte_range,
            Some(ByteRange {
                offset: media_start,
                length: 200
            })
        );
        assert_eq!(segs[0].time, 0);
        assert_eq!(
            segs[1].byte_range,
            Some(ByteRange {
                offset: media_start + 200,
                length: 300
            })
        );
        assert_eq!(segs[1].time, 1000);
        assert_eq!(
            segs[2].byte_range,
            Some(ByteRange {
                offset: media_start + 500,
                length: 150
            })
        );
        assert_eq!(segs[2].time, 2000);
    }

    #[test]
    fn parse_sidx_rejects_truncated_and_wrong_box() {
        // Truncated mid-entry: a hostile reference_count must fail, not over-read.
        let mut sidx = build_sidx(1000, &[(200, 1000), (300, 1000)]);
        sidx.truncate(sidx.len() - 4);
        assert!(parse_sidx(&sidx).is_none(), "truncated sidx rejected");
        // Not a sidx box.
        let mut notsidx = build_sidx(1000, &[(1, 1)]);
        notsidx[4..8].copy_from_slice(b"moof");
        assert!(parse_sidx(&notsidx).is_none(), "non-sidx box rejected");
        assert!(parse_sidx(&[0, 0, 0, 4]).is_none(), "too short rejected");
    }

    #[test]
    fn parses_segment_base_representation() {
        let xml = r#"<MPD type="static"><Period><AdaptationSet mimeType="video/mp4">
          <BaseURL>media.mp4</BaseURL>
          <Representation id="v0" bandwidth="1000000">
            <SegmentBase indexRange="900-1199" timescale="1000">
              <Initialization range="0-899"/>
            </SegmentBase>
          </Representation>
        </AdaptationSet></Period></MPD>"#;
        let mpd = parse(xml).unwrap();
        let rep = mpd.periods[0].select(None).unwrap();
        let sb = rep.segment_base().expect("SegmentBase addressing");
        assert_eq!(
            sb.index_range,
            ByteRange {
                offset: 900,
                length: 300
            }
        );
        assert_eq!(rep.timescale(), 1000);
        assert_eq!(
            rep.init(),
            Some((
                String::new(),
                Some(ByteRange {
                    offset: 0,
                    length: 900
                })
            ))
        );
        // SegmentBase resolves segments only after fetching the sidx, so the
        // pure (no-I/O) path is empty.
        assert!(rep.resolved_segments(mpd.duration_secs).is_empty());
        assert!(rep.template().is_none());
    }

    #[test]
    fn dash_range_parse_rejects_reversed_and_malformed() {
        assert_eq!(
            parse_dash_range("0-799"),
            Some(ByteRange {
                offset: 0,
                length: 800
            })
        );
        assert_eq!(
            parse_dash_range("800-800"),
            Some(ByteRange {
                offset: 800,
                length: 1
            })
        );
        assert_eq!(parse_dash_range("999-800"), None, "reversed range rejected");
        assert_eq!(parse_dash_range("notarange"), None);
    }

    #[test]
    fn iso_duration_forms() {
        assert_eq!(parse_iso_duration("PT12.0S"), Some(12.0));
        assert_eq!(parse_iso_duration("PT1H2M3S"), Some(3723.0));
        assert_eq!(parse_iso_duration("PT0.5S"), Some(0.5));
    }

    #[test]
    fn static_manifest_is_not_dynamic() {
        let mpd = parse(MPD).unwrap();
        assert!(!mpd.dynamic);
        assert_eq!(mpd.minimum_update_period_secs, None);
    }

    #[test]
    fn dynamic_manifest_carries_update_period() {
        let xml = r#"<MPD type="dynamic" minimumUpdatePeriod="PT2S"><Period><AdaptationSet>
          <SegmentTemplate media="$Number$.m4s" startNumber="1" duration="1000" timescale="1000"/>
          <Representation id="r" bandwidth="1"/>
        </AdaptationSet></Period></MPD>"#;
        let mpd = parse(xml).unwrap();
        assert!(mpd.dynamic);
        assert_eq!(mpd.minimum_update_period_secs, Some(2.0));
    }

    #[test]
    fn rejects_non_mpd() {
        assert_eq!(parse("not xml at all <<<"), Err(MpdError::Invalid));
    }

    // --- wall-clock live profile + multi-period (M836) -------------------

    /// A dynamic `@duration` MPD as ffmpeg's dash muxer writes one mid-stream
    /// (`-f dash -streaming 1 -window_size 3 -use_template 1 -use_timeline 0`),
    /// with the availabilityStartTime pinned so the window math is exact.
    const FFMPEG_LIVE_MPD: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" profiles="urn:mpeg:dash:profile:isoff-live:2011"
     type="dynamic" minimumUpdatePeriod="PT500S" suggestedPresentationDelay="PT1S"
     availabilityStartTime="2026-08-03T00:00:00Z" publishTime="2026-08-03T00:00:07Z"
     timeShiftBufferDepth="PT3.0S" maxSegmentDuration="PT1.0S" minBufferTime="PT2.0S">
  <Period id="0" start="PT0.0S">
    <AdaptationSet id="0" contentType="video" frameRate="30/1">
      <Representation id="0" mimeType="video/mp4" codecs="avc1.f4000a" bandwidth="47600" width="64" height="48">
        <SegmentTemplate timescale="1000000" duration="1000000" availabilityTimeOffset="0.967"
                         initialization="init-stream$RepresentationID$.m4s"
                         media="chunk-stream$RepresentationID$-$Number%05d$.m4s" startNumber="1"/>
      </Representation>
    </AdaptationSet>
  </Period>
</MPD>"#;

    /// Unix seconds of `2026-08-03T00:00:00Z`.
    const AST_UNIX: f64 = 1_785_715_200.0;

    #[test]
    fn parses_dynamic_wall_clock_attributes() {
        let mpd = parse(FFMPEG_LIVE_MPD).unwrap();
        assert!(mpd.dynamic);
        assert_eq!(mpd.availability_start_unix, Some(AST_UNIX));
        assert_eq!(mpd.time_shift_buffer_depth_secs, Some(3.0));
        assert_eq!(mpd.suggested_presentation_delay_secs, Some(1.0));
        assert_eq!(mpd.max_segment_duration_secs, Some(1.0));
        assert_eq!(mpd.periods.len(), 1);
        assert_eq!(mpd.periods[0].start_secs, 0.0);
        let tmpl = mpd.periods[0].representations[0].template().unwrap();
        assert_eq!(tmpl.availability_time_offset, Some(0.967));
        assert_eq!(tmpl.presentation_time_offset, 0);
        let edge = mpd.live_edge(&mpd.periods[0], 1.0).unwrap();
        assert_eq!(edge.anchor_unix, AST_UNIX);
        assert_eq!(edge.time_shift_secs, Some(3.0));
    }

    #[test]
    fn live_window_is_the_complete_segments_inside_the_time_shift_depth() {
        let mpd = parse(FFMPEG_LIVE_MPD).unwrap();
        let rep = &mpd.periods[0].representations[0];
        let tmpl = rep.template().unwrap();
        let edge = mpd.live_edge(&mpd.periods[0], 1.0).unwrap();

        // 7s in: segments 0..=6 (0-based) are complete (segment i finishes at
        // (i+1)s); the 3s depth keeps the last three of them.
        assert_eq!(
            tmpl.live_available(&edge, AST_UNIX + 7.0, None),
            Some((4, 6))
        );
        let segs = rep.live_segments(&edge, AST_UNIX + 7.0, None).unwrap();
        let urls: Vec<&str> = segs.iter().map(|s| s.url.as_str()).collect();
        assert_eq!(
            urls,
            [
                "chunk-stream0-00005.m4s",
                "chunk-stream0-00006.m4s",
                "chunk-stream0-00007.m4s"
            ],
            "$Number$ = startNumber + index, zero-padded"
        );
        // The segment start time, in timescale units (this manifest declares no
        // presentationTimeOffset, so media time is presentation time).
        assert_eq!(segs[0].time, 4_000_000);

        // The boundaries of the nominal window, with the manifest's 0.967s
        // availabilityTimeOffset cleared (its own test covers that): nothing
        // before the first segment completes, exactly one segment at 1.5s in.
        let mut nominal = tmpl.clone();
        nominal.availability_time_offset = None;
        assert_eq!(nominal.live_available(&edge, AST_UNIX + 0.5, None), None);
        assert_eq!(
            nominal.live_available(&edge, AST_UNIX + 1.5, None),
            Some((0, 0))
        );
        // Nothing at all before the presentation starts.
        assert_eq!(tmpl.live_available(&edge, AST_UNIX - 60.0, None), None);
        assert_eq!(
            rep.live_segments(&edge, AST_UNIX - 60.0, None),
            Some(Vec::new())
        );
    }

    #[test]
    fn availability_time_offset_publishes_the_in_progress_segment_early() {
        let mpd = parse(FFMPEG_LIVE_MPD).unwrap();
        let tmpl = mpd.periods[0].representations[0].template().unwrap();
        let edge = mpd.live_edge(&mpd.periods[0], 1.0).unwrap();

        // 6.98s in, 1s segments: segment 6 (0-based) completes at 7s, but the
        // 0.967s availabilityTimeOffset publishes it from 6.033s.
        assert_eq!(
            tmpl.live_available(&edge, AST_UNIX + 6.98, None),
            Some((4, 6))
        );
        let mut nominal = tmpl.clone();
        nominal.availability_time_offset = None;
        assert_eq!(
            nominal.live_available(&edge, AST_UNIX + 6.98, None),
            Some((4, 5)),
            "without the offset only complete segments are available"
        );

        // An offset past the segment duration (or INF) clamps to one segment: the
        // in-progress one, never media that does not exist yet, and never a
        // window before the anchor.
        for early in [1e9, f64::INFINITY] {
            let mut wide = tmpl.clone();
            wide.availability_time_offset = Some(early);
            assert_eq!(
                wide.live_available(&edge, AST_UNIX + 6.98, None),
                Some((4, 6))
            );
            assert_eq!(
                wide.live_available(&edge, AST_UNIX + 0.5, None),
                Some((0, 0))
            );
            // The clamp is exactly one segment, so nothing opens before the anchor.
            assert_eq!(wide.live_available(&edge, AST_UNIX - 0.001, None), None);
            assert_eq!(wide.live_available(&edge, AST_UNIX - 60.0, None), None);
        }
    }

    #[test]
    fn live_window_without_time_shift_depth_runs_back_to_the_period_start() {
        let mpd = parse(FFMPEG_LIVE_MPD).unwrap();
        let tmpl = mpd.periods[0].representations[0].template().unwrap();
        let edge = LiveEdge {
            anchor_unix: AST_UNIX,
            time_shift_secs: None,
            presentation_delay_secs: 1.0,
        };
        assert_eq!(
            tmpl.live_available(&edge, AST_UNIX + 7.0, None),
            Some((0, 6))
        );
    }

    #[test]
    fn a_period_start_offsets_the_live_anchor() {
        let mpd = parse(FFMPEG_LIVE_MPD).unwrap();
        let tmpl = mpd.periods[0].representations[0].template().unwrap();
        // The same wall clock, but a Period starting 5s into the presentation:
        // only 2s of it exist, so one segment is complete.
        let edge = LiveEdge {
            anchor_unix: AST_UNIX + 5.0,
            time_shift_secs: Some(3.0),
            presentation_delay_secs: 1.0,
        };
        assert_eq!(
            tmpl.live_available(&edge, AST_UNIX + 7.0, None),
            Some((0, 1))
        );
    }

    #[test]
    fn a_period_end_caps_the_live_window() {
        let mpd = parse(FFMPEG_LIVE_MPD).unwrap();
        let tmpl = mpd.periods[0].representations[0].template().unwrap();
        let edge = mpd.live_edge(&mpd.periods[0], 1.0).unwrap();
        let now = AST_UNIX + 7.0;
        let (_, open_last) = tmpl.live_available(&edge, now, None).unwrap();
        let ended_secs = open_last as f64;
        let (_, ended_last) = tmpl.live_available(&edge, now, Some(ended_secs)).unwrap();
        assert_eq!(
            ended_last,
            tmpl.segment_count(ended_secs) - 1,
            "the last segment offered is the Period's last"
        );
        assert!(tmpl.live_offers_period_end(&edge, now, Some(ended_secs)));
        let running_secs = ended_secs * 4.0;
        assert_eq!(
            tmpl.live_available(&edge, now, Some(running_secs)),
            tmpl.live_available(&edge, now, None)
        );
        assert!(!tmpl.live_offers_period_end(&edge, now, Some(running_secs)));
        assert!(!tmpl.live_offers_period_end(&edge, now, None));
    }

    #[test]
    fn live_start_offset_sits_a_presentation_delay_behind_the_edge() {
        let mpd = parse(FFMPEG_LIVE_MPD).unwrap();
        let rep = &mpd.periods[0].representations[0];
        let edge = mpd.live_edge(&mpd.periods[0], 1.0).unwrap();
        let segs = rep.live_segments(&edge, AST_UNIX + 7.0, None).unwrap(); // indices 4..=6
        let ts = rep.timescale();

        // 1s of delay = the newest complete segment only.
        assert_eq!(live_start_offset(&segs, ts, 1_000_000_000), 2);
        assert_eq!(live_start_offset(&segs, ts, 2_000_000_000), 1);
        // A delay deeper than the window clamps to its front rather than
        // requesting media that has already aged out.
        assert_eq!(live_start_offset(&segs, ts, 3_000_000_000), 0);
        assert_eq!(live_start_offset(&segs, ts, 60_000_000_000), 0);
        assert_eq!(live_start_offset(&[], ts, 1_000_000_000), 0);
    }

    const LIVE_TIME_SHIFT_DEPTH_SECS: u32 = 4;

    // a dynamic MPD around `addressing`, anchored at `AST_UNIX`
    fn live_mpd_with(addressing: &str) -> Mpd {
        parse(&alloc::format!(
            r#"<MPD type="dynamic" availabilityStartTime="2026-08-03T00:00:00Z"
     timeShiftBufferDepth="PT{LIVE_TIME_SHIFT_DEPTH_SECS}S">
  <Period id="0" start="PT0S"><AdaptationSet mimeType="video/mp4">{addressing}
    <Representation id="v" bandwidth="1"/>
  </AdaptationSet></Period>
</MPD>"#
        ))
        .unwrap()
    }

    fn live_urls(mpd: &Mpd, now_unix: f64, period_secs: Option<f64>) -> Vec<String> {
        let edge = mpd.live_edge(&mpd.periods[0], 1.0).unwrap();
        mpd.periods[0].representations[0]
            .live_segments(&edge, now_unix, period_secs)
            .unwrap()
            .into_iter()
            .map(|segment| segment.url)
            .collect()
    }

    fn lists_live_period_end(mpd: &Mpd, now_unix: f64, period_secs: Option<f64>) -> bool {
        let edge = mpd.live_edge(&mpd.periods[0], 1.0).unwrap();
        mpd.periods[0].representations[0].lists_period_end(&edge, now_unix, period_secs)
    }

    // two 2s entries then three 1s entries, the whole timeline listed ahead of the clock
    const LIVE_TIMELINE: &str = r#"<SegmentTemplate media="seg-$Time$.m4s" timescale="1000">
      <SegmentTimeline><S t="0" d="2000" r="1"/><S d="1000" r="2"/></SegmentTimeline>
    </SegmentTemplate>"#;

    #[test]
    fn live_timeline_holds_back_entries_until_they_are_complete() {
        let mpd = live_mpd_with(LIVE_TIMELINE);
        assert!(live_urls(&mpd, AST_UNIX - 60.0, None).is_empty());
        assert!(
            live_urls(&mpd, AST_UNIX + 1.5, None).is_empty(),
            "the first entry ends 2s in"
        );
        assert_eq!(live_urls(&mpd, AST_UNIX + 2.0, None), ["seg-0.m4s"]);
        // 0, 2000 and 4000 are complete, the 4s depth drops the one starting at 0
        assert_eq!(
            live_urls(&mpd, AST_UNIX + 5.5, None),
            ["seg-2000.m4s", "seg-4000.m4s"]
        );
        assert!(!lists_live_period_end(&mpd, AST_UNIX + 5.5, None));
        assert!(lists_live_period_end(&mpd, AST_UNIX + 7.0, None));
        // long after the last entry ends only the newest one is offered
        assert_eq!(live_urls(&mpd, AST_UNIX + 60.0, None), ["seg-6000.m4s"]);
    }

    #[test]
    fn live_timeline_availability_time_offset_publishes_an_entry_early() {
        let mpd = live_mpd_with(&LIVE_TIMELINE.replace(
            "timescale=\"1000\"",
            "timescale=\"1000\" availabilityTimeOffset=\"0.5\"",
        ));
        // the 1s entry starting at 4000 ends at 5s and is published from 4.5s
        assert_eq!(
            live_urls(&mpd, AST_UNIX + 4.5, None),
            ["seg-2000.m4s", "seg-4000.m4s"]
        );
        // the 2s entry starting at 0 is published no earlier than 1.5s
        assert!(live_urls(&mpd, AST_UNIX + 1.4, None).is_empty());
        assert_eq!(live_urls(&mpd, AST_UNIX + 1.5, None), ["seg-0.m4s"]);
    }

    const OPEN_ENDED_TIMELINE: &str = r#"<SegmentTemplate media="seg-$Number$.m4s" startNumber="1" timescale="1000">
      <SegmentTimeline><S t="0" d="1000" r="-1"/></SegmentTimeline>
    </SegmentTemplate>"#;

    #[test]
    fn live_negative_repeat_runs_on_with_the_clock() {
        let mpd = live_mpd_with(OPEN_ENDED_TIMELINE);
        assert_eq!(
            mpd.periods[0].representations[0]
                .template()
                .unwrap()
                .timeline[0]
                .r,
            TimelineRepeat::UntilNextEntryOrPeriodEnd
        );
        assert_eq!(
            live_urls(&mpd, AST_UNIX + 3.5, None),
            ["seg-1.m4s", "seg-2.m4s", "seg-3.m4s"]
        );
        // 100 segments are complete, the 4s depth keeps those starting from 96.5s
        assert_eq!(
            live_urls(&mpd, AST_UNIX + 100.5, None),
            ["seg-98.m4s", "seg-99.m4s", "seg-100.m4s"]
        );
        assert!(!lists_live_period_end(&mpd, AST_UNIX + 100.5, None));

        // a 5s Period stops the repeat at its fifth segment
        let period_secs = Some(5.0);
        assert_eq!(
            live_urls(&mpd, AST_UNIX + 100.5, period_secs),
            ["seg-5.m4s"]
        );
        assert!(!lists_live_period_end(&mpd, AST_UNIX + 4.5, period_secs));
        assert!(lists_live_period_end(&mpd, AST_UNIX + 5.0, period_secs));
    }

    #[test]
    fn negative_repeat_stops_at_the_next_entry_start() {
        let mpd = live_mpd_with(
            r#"<SegmentTemplate media="seg-$Time$.m4s" timescale="1000">
      <SegmentTimeline><S t="0" d="1000" r="-1"/><S t="3000" d="2000"/></SegmentTimeline>
    </SegmentTemplate>"#,
        );
        assert_eq!(
            live_urls(&mpd, AST_UNIX + 5.0, None),
            ["seg-1000.m4s", "seg-2000.m4s", "seg-3000.m4s"]
        );
        assert!(lists_live_period_end(&mpd, AST_UNIX + 5.0, None));
    }

    #[test]
    fn vod_negative_repeat_fills_the_period() {
        let mpd = live_mpd_with(OPEN_ENDED_TIMELINE);
        let template = mpd.periods[0].representations[0].template().unwrap();
        let numbers: Vec<u64> = template.segments(4.0).iter().map(|s| s.number).collect();
        assert_eq!(numbers, [1, 2, 3, 4]);
        assert_eq!(
            template.segments(0.0).len(),
            1,
            "no end to stop the repeat lists its first segment"
        );
    }

    // four 2s entries, all listed ahead of the clock
    const LIVE_SEGMENT_LIST: &str = r#"<SegmentList duration="2" timescale="1">
      <SegmentURL media="s0.m4s"/><SegmentURL media="s1.m4s"/>
      <SegmentURL media="s2.m4s"/><SegmentURL media="s3.m4s"/>
    </SegmentList>"#;

    #[test]
    fn live_segment_list_holds_back_entries_until_they_are_complete() {
        let mpd = live_mpd_with(LIVE_SEGMENT_LIST);
        assert!(live_urls(&mpd, AST_UNIX + 1.0, None).is_empty());
        assert_eq!(live_urls(&mpd, AST_UNIX + 2.0, None), ["s0.m4s"]);
        // s0, s1 and s2 are complete, the 4s depth keeps those starting from 2.5s
        assert_eq!(live_urls(&mpd, AST_UNIX + 6.5, None), ["s2.m4s"]);
        assert!(!lists_live_period_end(&mpd, AST_UNIX + 6.5, None));
        let edge = mpd.live_edge(&mpd.periods[0], 1.0).unwrap();
        let segments = mpd.periods[0].representations[0]
            .live_segments(&edge, AST_UNIX + 8.0, None)
            .unwrap();
        let times: Vec<u64> = segments.iter().map(|s| s.time).collect();
        assert_eq!(times, [4, 6], "list times accumulate by @duration");
        assert!(lists_live_period_end(&mpd, AST_UNIX + 8.0, None));
    }

    #[test]
    fn live_segment_list_availability_time_offset_publishes_an_entry_early() {
        let mpd = live_mpd_with(&LIVE_SEGMENT_LIST.replace(
            "timescale=\"1\"",
            "timescale=\"1\" availabilityTimeOffset=\"1.5\"",
        ));
        assert_eq!(live_urls(&mpd, AST_UNIX + 6.5, None), ["s2.m4s", "s3.m4s"]);
    }

    #[test]
    fn live_segment_list_without_duration_lists_every_entry() {
        let mpd = live_mpd_with(&LIVE_SEGMENT_LIST.replace("duration=\"2\" ", ""));
        let edge = mpd.live_edge(&mpd.periods[0], 1.0).unwrap();
        assert_eq!(
            mpd.periods[0].representations[0].live_segments(&edge, AST_UNIX + 1.0, None),
            None
        );
    }

    #[test]
    fn xs_datetime_forms_and_rejects() {
        assert_eq!(
            parse_xs_datetime("2026-08-03T20:45:39.541Z"),
            Some(1_785_789_939.541)
        );
        assert_eq!(parse_xs_datetime("1970-01-01T00:00:00Z"), Some(0.0));
        assert_eq!(
            parse_xs_datetime("1999-12-31T23:59:59Z"),
            Some(946_684_799.0)
        );
        // A zone offset is east of UTC, so it comes off the unix value.
        assert_eq!(
            parse_xs_datetime("2026-08-03T20:45:39+02:00"),
            Some(1_785_782_739.0)
        );
        // Naive (no designator) is read as UTC.
        assert_eq!(parse_xs_datetime("2026-08-03T00:00:00"), Some(AST_UNIX));
        for bad in [
            "2026-08-03",
            "2026-13-03T00:00:00Z",
            "2026-08-32T00:00:00Z",
            "2026-08-03T24:00:00Z",
            "2026-08-03T00:60:00Z",
            "2026-08-03T00:00Z",
            "not-a-date",
        ] {
            assert_eq!(parse_xs_datetime(bad), None, "{bad} rejected");
        }
    }

    /// Two Periods stitched into one presentation, the canonical multi-period
    /// form: each Period carries its own BaseURL + template.
    const MULTI_PERIOD_MPD: &str = r#"<MPD type="static" mediaPresentationDuration="PT6S">
      <Period id="a" start="PT0S" duration="PT3S">
        <BaseURL>one/</BaseURL>
        <AdaptationSet mimeType="video/mp4">
          <SegmentTemplate initialization="init.mp4" media="seg$Number$.m4s"
                           startNumber="0" duration="1000" timescale="1000"/>
          <Representation id="v0" bandwidth="1000000"/>
        </AdaptationSet>
      </Period>
      <Period id="b">
        <BaseURL>two/</BaseURL>
        <AdaptationSet mimeType="video/mp4">
          <SegmentTemplate initialization="init.mp4" media="seg$Number$.m4s"
                           startNumber="0" duration="1000" timescale="1000"/>
          <Representation id="v0" bandwidth="1000000"/>
        </AdaptationSet>
      </Period>
    </MPD>"#;

    #[test]
    fn parses_consecutive_periods_with_their_own_base_urls() {
        let mpd = parse(MULTI_PERIOD_MPD).unwrap();
        assert_eq!(mpd.periods.len(), 2);
        assert_eq!(mpd.periods[0].id.as_deref(), Some("a"));
        assert_eq!(mpd.periods[0].base_url.as_deref(), Some("one/"));
        assert_eq!(mpd.periods[0].duration_secs, Some(3.0));
        // The second Period declares no @start, so it accumulates from the
        // first one's start + duration.
        assert_eq!(mpd.periods[1].start_secs, 3.0);
        assert_eq!(mpd.periods[1].base_url.as_deref(), Some("two/"));
        // An MPD-level BaseURL is not confused with a Period's.
        assert_eq!(mpd.base_url, None);
        // Each Period selects within its own Representations.
        assert_eq!(mpd.periods[1].select(None).unwrap().id, "v0");
    }

    #[test]
    fn periods_without_a_usable_representation_are_skipped() {
        let xml = r#"<MPD type="static"><Period id="ad" duration="PT5S"/><Period id="main"><AdaptationSet>
          <SegmentTemplate media="$Number$.m4s" duration="1000" timescale="1000"/>
          <Representation id="v0" bandwidth="1"/>
        </AdaptationSet></Period></MPD>"#;
        let mpd = parse(xml).unwrap();
        assert_eq!(mpd.periods.len(), 1);
        assert_eq!(mpd.periods[0].id.as_deref(), Some("main"));
    }
}
