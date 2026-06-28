//! DASH source (DashSrc, `dash` feature): fetches an MPD manifest, selects a
//! Representation (simple bandwidth-capped ABR), and streams its fMP4 init +
//! media segments downstream as a `Caps::ByteStream{IsoBmff}` for `fmp4demux`,
//! then `Eos`. The [`mpd`](crate::mpd) parser does the manifest work; this adds
//! the fetching (via `reqwest`, shared with [`HlsSrc`](crate::hlssrc)) and the
//! `SegmentTemplate` `$Number$` / `$Time$` addressing.
//!
//! `SegmentList` is also supported (M369): an explicit ordered list of
//! `<SegmentURL>` entries, each a `@media` URL and/or a `mediaRange` byte range
//! of the `BaseURL` resource, with an `<Initialization>` (`sourceURL` / `range`).
//! A range-only entry fetches just its sub-range with an HTTP `Range` request,
//! the DASH analog of HLS `#EXT-X-BYTERANGE`, so a single-file CMAF DASH stream
//! plays.
//!
//! Scope: `SegmentTemplate` (`@duration` profile or `SegmentTimeline`) and
//! `SegmentList`, one `DataFrame` per segment, a fixed Representation. Dynamic
//! (live) manifests are handled by reloading on the MPD's refresh period. The
//! wall-clock `availabilityStartTime` live profile, `SegmentBase` (`sidx`-indexed
//! single resource), multi-period, and mid-stream switching are follow-ups
//! (DESIGN_TODO).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::runtime::{SeekController, SourceLoop};
use g2g_core::{
    ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError,
    OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Seek, Segment,
};

use crate::fetch::{
    byte_frame, get_bytes, get_range_bytes, get_text, resolve_url, MAX_MANIFEST_BYTES,
    MAX_SEGMENT_BYTES,
};
use crate::mpd::{parse, parse_sidx, ByteRange, ResolvedSegment};

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
) -> Result<Vec<u8>, G2gError> {
    match range {
        Some(r) => get_range_bytes(client, url, r.offset, r.length, MAX_SEGMENT_BYTES).await,
        None => get_bytes(client, url, MAX_SEGMENT_BYTES).await,
    }
}

#[derive(Debug)]
pub struct DashSrc {
    url: String,
    /// ABR cap: select the highest-bandwidth Representation at or below this
    /// (0 = no cap, pick the highest available).
    max_bandwidth: u64,
    /// Live-MPD reload interval in ms (0 = derive from `minimumUpdatePeriod`).
    reload_interval_ms: u64,
    /// Optional time-seek channel (M367): resolves a TIME seek to the media
    /// segment whose start time precedes the target (the `SegmentRef.time` is
    /// already a stream-time in `timescale` units), flushes, re-emits the init
    /// segment, and resumes there. The CMAF/DASH segment-transition case.
    seek: Option<SeekController>,
    configured: bool,
}

impl DashSrc {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            max_bandwidth: 0,
            reload_interval_ms: 0,
            seek: None,
            configured: false,
        }
    }

    /// Make the source time-seekable (M367): `run` polls `controller` before each
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

    fn output_caps() -> Caps {
        Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff }
    }
}

/// The index of the media segment containing `target_ns` (the last whose
/// `$Time$` start, converted from `timescale` units to ns, is at or before the
/// target) and that segment's start time in ns. Segments are time-ordered, so
/// the scan stops at the first start past the target. A target before the first
/// segment clamps to it; empty input returns `(0, 0)`.
fn segment_for_time(segs: &[ResolvedSegment], timescale: u64, target_ns: u64) -> (usize, u64) {
    let timescale = timescale.max(1) as u128;
    let mut chosen = 0usize;
    let mut chosen_start = 0u64;
    for (idx, seg) in segs.iter().enumerate() {
        let start_ns = (seg.time as u128 * 1_000_000_000 / timescale) as u64;
        if start_ns <= target_ns {
            chosen = idx;
            chosen_start = start_ns;
        } else {
            break;
        }
    }
    (chosen, chosen_start)
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
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(Self::output_caps()))))
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

            let mut mpd = {
                let text = get_text(&client, &self.url, MAX_MANIFEST_BYTES).await?;
                parse(&text).map_err(|_| G2gError::CapsMismatch)?
            };

            let mut sequence = 0u64;
            let mut init_emitted = false;
            // Largest segment start time already played; on a live reload only
            // segments past it are new (SegmentTimeline times are monotonic).
            let mut last_time: Option<u64> = None;
            loop {
                // Segment URIs resolve against the MPD BaseURL (if any) resolved
                // against the manifest URL, else the manifest URL's directory.
                let base = match &mpd.base_url {
                    Some(b) => resolve_url(&self.url, b),
                    None => self.url.clone(),
                };
                let rep = mpd.select(cap).ok_or(G2gError::CapsMismatch)?;
                // `SegmentTemplate` ($Number$/$Time$, SegmentTimeline or @duration),
                // an explicit `SegmentList`, or a `sidx`-indexed `SegmentBase` all
                // resolve to one ordered segment list, each with an optional byte
                // sub-range and a start time. An empty URL is the BaseURL resource
                // itself (byte-range SegmentList / SegmentBase / Initialization).
                let mut timescale = rep.timescale();
                let init = rep.init();
                let base_index = rep.segment_base().map(|sb| sb.index_range);
                let segs = if let Some(index_range) = base_index {
                    // SegmentBase: fetch the sidx index, parse it, and turn its
                    // subsegments into byte ranges of the BaseURL resource. The
                    // sidx carries the authoritative timescale for the durations.
                    let index_url = seg_url(&base, "");
                    let idx_bytes =
                        fetch_segment(&client, &index_url, Some(index_range)).await?;
                    let sidx = parse_sidx(&idx_bytes).ok_or(G2gError::CapsMismatch)?;
                    timescale = sidx.timescale.max(1);
                    sidx.subsegments(index_range.offset)
                } else {
                    rep.resolved_segments(mpd.duration_secs)
                };

                let mut idx = 0usize;
                loop {
                    // Apply a pending flushing time seek before the next fetch:
                    // jump to the segment containing the target, flush, and re-emit
                    // the init segment (a downstream demuxer reset on the flush
                    // needs its moov again).
                    if let Some(seek) = self.seek.as_ref().and_then(|c| c.take_pending()) {
                        if seek.is_flush() {
                            let (target_idx, start_ns) =
                                segment_for_time(&segs, timescale, seek.start);
                            out.push(PipelinePacket::Flush).await?;
                            idx = target_idx;
                            // Jumped by index; clear the reload-dedup watermark so
                            // the target segment is not skipped as "already played".
                            last_time = None;
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
                            let bytes = fetch_segment(&client, &url, *init_range).await?;
                            if !bytes.is_empty() {
                                out.push(PipelinePacket::DataFrame(byte_frame(bytes, sequence)))
                                    .await?;
                                sequence += 1;
                            }
                        }
                        init_emitted = true;
                    }

                    if idx >= segs.len() {
                        break;
                    }
                    let seg = &segs[idx];
                    if !last_time.is_some_and(|lt| seg.time <= lt) {
                        let url = seg_url(&base, &seg.url);
                        let bytes = fetch_segment(&client, &url, seg.byte_range).await?;
                        if !bytes.is_empty() {
                            out.push(PipelinePacket::DataFrame(byte_frame(bytes, sequence))).await?;
                            sequence += 1;
                        }
                        last_time = Some(seg.time);
                    }
                    idx += 1;
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
                let text = get_text(&client, &self.url, MAX_MANIFEST_BYTES).await?;
                mpd = parse(&text).map_err(|_| G2gError::CapsMismatch)?;
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
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.url.clone())),
            "max-bandwidth" => Some(PropValue::Uint(self.max_bandwidth)),
            "reload-interval-ms" => Some(PropValue::Uint(self.reload_interval_ms)),
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
];
