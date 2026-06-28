//! DASH source (DashSrc, `dash` feature): fetches an MPD manifest, selects a
//! Representation (simple bandwidth-capped ABR), and streams its fMP4 init +
//! media segments downstream as a `Caps::ByteStream{IsoBmff}` for `fmp4demux`,
//! then `Eos`. The [`mpd`](crate::mpd) parser does the manifest work; this adds
//! the fetching (via `reqwest`, shared with [`HlsSrc`](crate::hlssrc)) and the
//! `SegmentTemplate` `$Number$` / `$Time$` addressing.
//!
//! Scope: `SegmentTemplate` with the `@duration` profile or `SegmentTimeline`,
//! one `DataFrame` per segment, a fixed Representation. Dynamic (live) manifests
//! are handled by reloading on the MPD's refresh period. The wall-clock
//! `availabilityStartTime` live profile, `SegmentList`/`SegmentBase`, multi-period,
//! and mid-stream switching are follow-ups (DESIGN_TODO).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;

use g2g_core::runtime::SourceLoop;
use g2g_core::{
    ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError,
    OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::fetch::{
    byte_frame, get_bytes, get_text, resolve_url, MAX_MANIFEST_BYTES, MAX_SEGMENT_BYTES,
};
use crate::mpd::parse;

#[derive(Debug)]
pub struct DashSrc {
    url: String,
    /// ABR cap: select the highest-bandwidth Representation at or below this
    /// (0 = no cap, pick the highest available).
    max_bandwidth: u64,
    /// Live-MPD reload interval in ms (0 = derive from `minimumUpdatePeriod`).
    reload_interval_ms: u64,
    configured: bool,
}

impl DashSrc {
    pub fn new(url: impl Into<String>) -> Self {
        Self { url: url.into(), max_bandwidth: 0, reload_interval_ms: 0, configured: false }
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

                if !init_emitted {
                    if let Some(init) = rep.template.init_url(&rep.id) {
                        let bytes =
                            get_bytes(&client, &resolve_url(&base, &init), MAX_SEGMENT_BYTES).await?;
                        if !bytes.is_empty() {
                            out.push(PipelinePacket::DataFrame(byte_frame(bytes, sequence))).await?;
                            sequence += 1;
                        }
                    }
                    init_emitted = true;
                }
                // SegmentTimeline (if present) or the @duration profile drives the
                // ordered segments, with $Number$ / $Time$ resolved per segment.
                for seg in rep.template.segments(mpd.duration_secs) {
                    if last_time.is_some_and(|lt| seg.time <= lt) {
                        continue; // already played on an earlier reload
                    }
                    let media = rep.template.media_url(&rep.id, seg);
                    let bytes =
                        get_bytes(&client, &resolve_url(&base, &media), MAX_SEGMENT_BYTES).await?;
                    if !bytes.is_empty() {
                        out.push(PipelinePacket::DataFrame(byte_frame(bytes, sequence))).await?;
                        sequence += 1;
                    }
                    last_time = Some(seg.time);
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
