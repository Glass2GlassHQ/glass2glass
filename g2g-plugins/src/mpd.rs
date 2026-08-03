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
//! Not modelled: `@presentationTimeOffset` (segment `$Time$` is period-relative
//! and starts at zero) and `@availabilityTimeOffset` (the low-latency chunked
//! early-availability knob, which only makes the window more conservative here).

use alloc::string::String;
use alloc::vec::Vec;

use roxmltree::{Document, Node};

/// One selectable Representation (a single quality rendition).
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// A byte sub-range of a resource (`mediaRange` / `range` / `indexRange`):
/// `length` bytes from `offset`. The DASH analog of HLS `#EXT-X-BYTERANGE`; the
/// MPD spells it `"start-end"` (inclusive end), parsed by [`parse_dash_range`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteRange {
    pub offset: u64,
    pub length: u64,
}

/// How a Representation addresses its segments.
#[derive(Debug, Clone, PartialEq, Eq)]
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
    pub timescale: u64,
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
    pub timescale: u64,
    pub entries: Vec<SidxEntry>,
}

/// `SegmentList`: an explicit ordered list of media segments, each a URL and/or
/// a `mediaRange` byte range of the `BaseURL` resource, plus an `Initialization`.
/// `@duration` / `@timescale` give per-segment timing (a nested `SegmentTimeline`
/// is a follow-up).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentList {
    /// `<Initialization sourceURL>`, empty when the init is a byte range of the
    /// `BaseURL` itself; `init_present` distinguishes "no init element" from it.
    pub init_url: String,
    pub init_range: Option<ByteRange>,
    pub init_present: bool,
    pub duration: u64,
    pub timescale: u64,
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
    pub time: u64,
}

/// `SegmentTemplate` with `$Number$` / `$Time$` addressing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentTemplate {
    /// Init-segment template (resolves `$RepresentationID$`).
    pub initialization: Option<String>,
    /// Media-segment template (resolves `$RepresentationID$`, `$Number$`, `$Time$`).
    pub media: String,
    pub start_number: u64,
    /// Segment duration in `timescale` units (the `@duration` profile; the
    /// `SegmentTimeline` carries its own per-entry durations instead).
    pub duration: u64,
    pub timescale: u64,
    /// `SegmentTimeline` `<S>` entries when present; empty for the `@duration`
    /// profile.
    pub timeline: Vec<TimelineEntry>,
}

/// Cap on segments materialized from one manifest. A real presentation has far
/// fewer (1M two-second segments is ~23 days); the bound stops an untrusted
/// `@r` repeat or a tiny `@duration` from forcing an unbounded allocation.
const MAX_SEGMENTS: u64 = 1 << 20;

/// One `SegmentTimeline` `<S>` entry: a start time `t` (absent = continue from
/// the previous entry), a duration `d`, and `r` additional repeats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimelineEntry {
    pub t: Option<u64>,
    pub d: u64,
    pub r: u64,
}

/// One resolved media segment: its `$Number$` and its `$Time$` (start time in
/// `timescale` units).
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
    /// The presentation's Periods, in order. Always non-empty.
    pub periods: Vec<Period>,
}

/// One `Period`: a self-contained stretch of the presentation with its own
/// Representations, base URL and segment addressing. Consecutive Periods play
/// through back to back; each one's segment times restart at zero, so the
/// running-time offset of a Period is its `start_secs` (or the media played
/// before it).
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
/// Turns "now" into the window of segments a live `@duration` `SegmentTemplate`
/// currently offers, and the index playback should start at.
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
    /// still inside `timeShiftBufferDepth` and the newest complete one. Segment
    /// `i` covers presentation time `[i*d, (i+1)*d)` and is complete `(i+1)*d`
    /// after the Period anchor. `None` before the first complete segment exists,
    /// or for a `SegmentTimeline` (which lists its own window).
    pub fn live_available(&self, edge: &LiveEdge, now_unix: f64) -> Option<(u64, u64)> {
        if !self.timeline.is_empty() {
            return None;
        }
        let d = self.nominal_segment_secs()?;
        let elapsed = now_unix - edge.anchor_unix;
        // `as u64` saturates at 0 for a negative / NaN elapsed, and the checked
        // decrement then reports "no complete segment yet" instead of wrapping.
        let last = ((elapsed / d) as u64).checked_sub(1)?;
        let first = match edge.time_shift_secs {
            // Fully inside the window: the segment starts no earlier than the
            // depth allows, so round the boundary up.
            Some(depth) => (((elapsed - depth) / d).ceil()) as u64,
            None => 0,
        };
        Some((first.min(last), last))
    }

    /// The `@duration` live profile's available segments at `now_unix`, newest
    /// window last, each carrying its `$Number$` and its period-relative
    /// `$Time$`. Bounded by [`MAX_SEGMENTS`] so an ancient
    /// `availabilityStartTime` cannot force an unbounded allocation.
    pub fn live_segments(&self, edge: &LiveEdge, now_unix: f64) -> Vec<SegmentRef> {
        let Some((first, last)) = self.live_available(edge, now_unix) else {
            return Vec::new();
        };
        let count = (last - first).saturating_add(1).min(MAX_SEGMENTS);
        (first..first.saturating_add(count))
            .map(|i| SegmentRef {
                number: self.start_number.saturating_add(i),
                time: i.saturating_mul(self.duration),
            })
            .collect()
    }

    /// The ordered media segments for a VOD presentation of `total_secs`. Driven
    /// by the `SegmentTimeline` when present, else by `@duration`. Each carries
    /// its `$Number$` (from `startNumber`) and `$Time$` (accumulated start time).
    pub fn segments(&self, total_secs: f64) -> Vec<SegmentRef> {
        let mut out = Vec::new();
        let mut number = self.start_number;
        if self.timeline.is_empty() {
            let count = self.segment_count(total_secs).min(MAX_SEGMENTS);
            let mut time = 0u64;
            for _ in 0..count {
                out.push(SegmentRef { number, time });
                number += 1;
                time = time.saturating_add(self.duration);
            }
        } else {
            let mut time = 0u64;
            'timeline: for entry in &self.timeline {
                if let Some(t) = entry.t {
                    time = t;
                }
                for _ in 0..=entry.r {
                    if out.len() as u64 >= MAX_SEGMENTS {
                        break 'timeline;
                    }
                    out.push(SegmentRef { number, time });
                    number += 1;
                    time = time.saturating_add(entry.d);
                }
            }
        }
        out
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
    pub fn timescale(&self) -> u64 {
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
    /// synthesizes URLs by `$Number$` / `$Time$`; List returns its explicit URLs
    /// / ranges with cumulative `@duration` start times. `SegmentBase` returns
    /// empty here: its subsegments need the fetched `sidx` (see [`segment_base`]
    /// and [`Sidx::subsegments`]).
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
                    time: s.time,
                })
                .collect(),
            SegmentSource::List(l) => {
                let mut out = Vec::new();
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

    /// The media segments a dynamic `@duration` Representation offers at
    /// wall-clock `now_unix` (the live profile). Empty for any other addressing,
    /// or before this Period's first segment is complete.
    pub fn live_segments(&self, edge: &LiveEdge, now_unix: f64) -> Vec<ResolvedSegment> {
        let SegmentSource::Template(t) = &self.source else {
            return Vec::new();
        };
        t.live_segments(edge, now_unix)
            .into_iter()
            .map(|s| ResolvedSegment {
                url: t.media_url(&self.id, s),
                byte_range: None,
                time: s.time,
            })
            .collect()
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
pub fn live_start_offset(segs: &[ResolvedSegment], timescale: u64, delay_ns: u64) -> usize {
    let ts = timescale.max(1) as u128;
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
        ((b.saturating_sub(a) as u128 * 1_000_000_000) / ts) as u64
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
    let _reference_id = read_u32(data, &mut p)?;
    let timescale = read_u32(data, &mut p)? as u64;
    // earliest_presentation_time + first_offset: 32-bit in v0, 64-bit in v1.
    let first_offset = if version == 0 {
        let _ept = read_u32(data, &mut p)?;
        read_u32(data, &mut p)? as u64
    } else {
        let _ept = read_u64(data, &mut p)?;
        read_u64(data, &mut p)?
    };
    let _reserved = read_u16(data, &mut p)?;
    let reference_count = read_u16(data, &mut p)?;
    let mut entries = Vec::new();
    for _ in 0..reference_count {
        // reference_type(1) | referenced_size(31); subsegment_duration(32);
        // starts_with_SAP(1) | SAP_type(3) | SAP_delta_time(28).
        let w0 = read_u32(data, &mut p)?;
        let duration = read_u32(data, &mut p)? as u64;
        let _sap = read_u32(data, &mut p)?;
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

fn read_u16(d: &[u8], p: &mut usize) -> Option<u16> {
    let v = u16::from_be_bytes(d.get(*p..*p + 2)?.try_into().ok()?);
    *p += 2;
    Some(v)
}

fn read_u32(d: &[u8], p: &mut usize) -> Option<u32> {
    let v = u32::from_be_bytes(d.get(*p..*p + 4)?.try_into().ok()?);
    *p += 4;
    Some(v)
}

fn read_u64(d: &[u8], p: &mut usize) -> Option<u64> {
    let v = u64::from_be_bytes(d.get(*p..*p + 8)?.try_into().ok()?);
    *p += 8;
    Some(v)
}

impl Period {
    /// Pick the highest-bandwidth Representation at or below `max_bandwidth`
    /// (or the overall highest when `None` / nothing fits).
    pub fn select(&self, max_bandwidth: Option<u64>) -> Option<&Representation> {
        let under = |r: &&Representation| max_bandwidth.is_none_or(|cap| r.bandwidth <= cap);
        self.representations
            .iter()
            .filter(under)
            .max_by_key(|r| r.bandwidth)
            .or_else(|| self.representations.iter().min_by_key(|r| r.bandwidth))
    }
}

impl Mpd {
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
    let mut next_start = 0.0f64;
    for node in root
        .children()
        .filter(|n| n.is_element() && n.has_tag_name("Period"))
    {
        let period = parse_period(node, next_start);
        next_start = period.start_secs + period.duration_secs.unwrap_or(0.0);
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
        periods,
    })
}

/// Parse one `Period` and its Representations. `default_start` is the previous
/// Period's end, used when `@start` is absent.
fn parse_period(node: Node, default_start: f64) -> Period {
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
        start_secs: node
            .attribute("start")
            .and_then(parse_iso_duration)
            .unwrap_or(default_start),
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
        timeline,
    })
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

/// Parse a `SegmentTimeline`'s `<S>` entries. A negative `@r` (live "repeat to
/// period end") fails to parse as `u64` and falls back to 0; live is a follow-up.
fn parse_timeline(tl: Node) -> Vec<TimelineEntry> {
    tl.children()
        .filter(|c| c.is_element() && c.has_tag_name("S"))
        .map(|s| TimelineEntry {
            t: s.attribute("t").and_then(|v| v.parse().ok()),
            d: s.attribute("d").and_then(|v| v.parse().ok()).unwrap_or(0),
            r: s.attribute("r").and_then(|v| v.parse().ok()).unwrap_or(0),
        })
        .collect()
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
fn parse_xs_datetime(s: &str) -> Option<f64> {
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

/// Days from the unix epoch to a proleptic-Gregorian civil date (Howard
/// Hinnant's `days_from_civil`). Shifts the year to start in March so the leap
/// day lands at the end of the era.
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let mp = if month > 2 { month - 3 } else { month + 9 } as i64; // March = 0
    let doy = (153 * mp + 2) / 5 + day as i64 - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;

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
            timeline: Vec::from([TimelineEntry {
                t: Some(0),
                d: 1000,
                r: u64::MAX,
            }]),
        };
        assert_eq!(timeline.segments(10.0).len() as u64, MAX_SEGMENTS);

        // A near-zero @duration must not expand the @duration profile either.
        let tiny = SegmentTemplate {
            initialization: None,
            media: String::from("seg-$Number$.m4s"),
            start_number: 1,
            duration: 1,
            timescale: u64::MAX,
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
        assert_eq!(mpd.periods.len(), 1);
        assert_eq!(mpd.periods[0].start_secs, 0.0);
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
        assert_eq!(tmpl.live_available(&edge, AST_UNIX + 7.0), Some((4, 6)));
        let segs = rep.live_segments(&edge, AST_UNIX + 7.0);
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
        // $Time$ stays period-relative in timescale units.
        assert_eq!(segs[0].time, 4_000_000);

        // Nothing is complete before one segment duration has passed.
        assert_eq!(tmpl.live_available(&edge, AST_UNIX + 0.5), None);
        assert!(rep.live_segments(&edge, AST_UNIX + 0.5).is_empty());
        // Nor before the presentation starts at all.
        assert_eq!(tmpl.live_available(&edge, AST_UNIX - 60.0), None);
        // Exactly one segment complete at 1.5s in.
        assert_eq!(tmpl.live_available(&edge, AST_UNIX + 1.5), Some((0, 0)));
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
        assert_eq!(tmpl.live_available(&edge, AST_UNIX + 7.0), Some((0, 6)));
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
        assert_eq!(tmpl.live_available(&edge, AST_UNIX + 7.0), Some((0, 1)));
    }

    #[test]
    fn live_start_offset_sits_a_presentation_delay_behind_the_edge() {
        let mpd = parse(FFMPEG_LIVE_MPD).unwrap();
        let rep = &mpd.periods[0].representations[0];
        let edge = mpd.live_edge(&mpd.periods[0], 1.0).unwrap();
        let segs = rep.live_segments(&edge, AST_UNIX + 7.0); // indices 4..=6
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
        let xml = r#"<MPD type="static"><Period id="ad"/><Period id="main"><AdaptationSet>
          <SegmentTemplate media="$Number$.m4s" duration="1000" timescale="1000"/>
          <Representation id="v0" bandwidth="1"/>
        </AdaptationSet></Period></MPD>"#;
        let mpd = parse(xml).unwrap();
        assert_eq!(mpd.periods.len(), 1);
        assert_eq!(mpd.periods[0].id.as_deref(), Some("main"));
    }
}
