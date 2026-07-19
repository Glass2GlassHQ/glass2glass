//! DASH MPD manifest parser (a subset of ISO/IEC 23009-1), driven by
//! [`DashSrc`](crate::dashsrc). Pure (no I/O), so it is fully unit-testable.
//!
//! Scope: static (VOD) manifests using `SegmentTemplate`, both the `@duration`
//! profile and `SegmentTimeline`, with `$Number$` or `$Time$` addressing.
//! `SegmentList`, `SegmentBase` byte-ranges, and dynamic (live) manifests are
//! follow-ups. Attribute inheritance (geometry / codecs /
//! the `SegmentTemplate` itself declared on the `AdaptationSet` and shared by its
//! `Representation`s) is resolved by walking ancestors.

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
    pub base_url: Option<String>,
    /// `mediaPresentationDuration` in seconds (for the VOD segment count).
    pub duration_secs: f64,
    /// `@type="dynamic"`: a live manifest, refetched until it turns static.
    pub dynamic: bool,
    /// `@minimumUpdatePeriod` in seconds: how often a live manifest is reloaded.
    pub minimum_update_period_secs: Option<f64>,
    pub representations: Vec<Representation>,
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

impl Mpd {
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
        .descendants()
        .find(|n| n.has_tag_name("BaseURL"))
        .and_then(|n| n.text())
        .map(|s| String::from(s.trim()));

    let mut representations = Vec::new();
    for rep in root
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

    if representations.is_empty() {
        return Err(MpdError::Invalid);
    }
    Ok(Mpd {
        base_url,
        duration_secs,
        dynamic,
        minimum_update_period_secs,
        representations,
    })
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
        assert_eq!(mpd.representations.len(), 2);
        let high = mpd.select(None).unwrap();
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
        assert_eq!(mpd.select(Some(1_000_000)).unwrap().id, "low");
        assert_eq!(mpd.select(Some(1)).unwrap().id, "low"); // fallback to lowest
    }

    #[test]
    fn segment_count_and_url_templating() {
        let mpd = parse(MPD).unwrap();
        let rep = mpd.select(None).unwrap();
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
        let rep = mpd.select(None).unwrap();
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
        let segs = mpd.representations[0].template().unwrap().segments(0.0);
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
        let rep = mpd.select(None).unwrap();
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
        let rep = mpd.select(None).unwrap();
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
        let rep = mpd.select(None).unwrap();
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
}
