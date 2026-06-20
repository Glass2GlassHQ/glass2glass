//! HLS source (HlsSrc, `hls` feature): fetches an `.m3u8` playlist, selects a
//! variant (simple bandwidth-capped ABR), and streams that variant's MPEG-TS
//! media segments downstream as a `Caps::ByteStream{MpegTs}` for `tsdemux`, then
//! `Eos`. The [`hls`](crate::hls) parser does the playlist work; this element
//! adds the fetching (via `reqwest`, like [`HttpSrc`](crate::httpsrc)) and URL
//! resolution.
//!
//! VOD (a playlist with `#EXT-X-ENDLIST`) plays its segments once then `Eos`.
//! Live (no ENDLIST) reloads the media playlist on an interval, plays each new
//! segment once (tracked by HLS media-sequence), and ends when ENDLIST finally
//! appears or downstream shuts down.
//!
//! Scope: in-order segment fetch, one `DataFrame` per segment, a fixed variant
//! (no mid-stream ABR switch). fMP4/CMAF segments, byte-range segments,
//! key/decryption, throughput-driven ABR, and live-edge start (skip to the last
//! few segments) are follow-ups (DESIGN_TODO).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;

use g2g_core::runtime::SourceLoop;
use g2g_core::{
    ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError, OutputSink,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::fetch::{byte_frame, get_bytes, get_text, resolve_url};
use crate::hls::{parse, MediaPlaylist, Playlist};

#[derive(Debug)]
pub struct HlsSrc {
    url: String,
    /// ABR cap: select the highest-bandwidth variant at or below this (0 = no
    /// cap, pick the highest available).
    max_bandwidth: u64,
    /// Live-playlist reload interval in ms (0 = derive from `TARGETDURATION`).
    reload_interval_ms: u64,
    /// Container discovered by the negotiation-time probe: `IsoBmff` when the
    /// media playlist has an `#EXT-X-MAP` init segment (fMP4/CMAF), else `MpegTs`.
    /// Memoized so a re-fixate retry skips the probe.
    container: Option<ByteStreamEncoding>,
    configured: bool,
}

impl HlsSrc {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            max_bandwidth: 0,
            reload_interval_ms: 0,
            container: None,
            configured: false,
        }
    }

    /// Cap variant selection to this bitrate (bits/sec); 0 picks the highest.
    pub fn with_max_bandwidth(mut self, bits_per_sec: u64) -> Self {
        self.max_bandwidth = bits_per_sec;
        self
    }

    /// Override the live-playlist reload interval (ms); 0 derives it from the
    /// playlist `TARGETDURATION`.
    pub fn with_reload_interval_ms(mut self, ms: u64) -> Self {
        self.reload_interval_ms = ms;
        self
    }

    fn cap(&self) -> Option<u64> {
        (self.max_bandwidth != 0).then_some(self.max_bandwidth)
    }

    /// Fetch the playlist (resolving master -> media) and decide the segment
    /// container: `IsoBmff` if the media playlist carries an `#EXT-X-MAP` init
    /// segment, else `MpegTs`. Memoized in `self.container`.
    async fn probe(&mut self) -> Result<ByteStreamEncoding, G2gError> {
        if let Some(enc) = self.container {
            return Ok(enc);
        }
        let client = reqwest::Client::new();
        let (media, _) = resolve_media(&client, &self.url, self.cap()).await?;
        let enc = if media.map_uri.is_some() {
            ByteStreamEncoding::IsoBmff
        } else {
            ByteStreamEncoding::MpegTs
        };
        self.container = Some(enc);
        Ok(enc)
    }
}

/// Fetch `url` and resolve a master playlist down to a media playlist, returning
/// it with the URL it came from (for segment-URI resolution and live reload).
async fn resolve_media(
    client: &reqwest::Client,
    url: &str,
    cap: Option<u64>,
) -> Result<(MediaPlaylist, String), G2gError> {
    let text = get_text(client, url).await?;
    match parse(&text).map_err(|_| G2gError::CapsMismatch)? {
        Playlist::Media(m) => Ok((m, String::from(url))),
        Playlist::Master(master) => {
            let variant = master.select(cap).ok_or(G2gError::CapsMismatch)?;
            let media_url = resolve_url(url, &variant.uri);
            let media_text = get_text(client, &media_url).await?;
            match parse(&media_text).map_err(|_| G2gError::CapsMismatch)? {
                Playlist::Media(m) => Ok((m, media_url)),
                // A master pointing at another master is malformed.
                Playlist::Master(_) => Err(G2gError::CapsMismatch),
            }
        }
    }
}

impl SourceLoop for HlsSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = Pin<Box<dyn Future<Output = Result<Caps, G2gError>> + 'a>>
    where
        Self: 'a;

    /// Probe the playlist at negotiation to discover the segment container
    /// (TS vs fMP4), the way `RtspSrc` does its DESCRIBE. The probe is memoized.
    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        Box::pin(async move {
            let encoding = self.probe().await?;
            Ok(Caps::ByteStream { encoding })
        })
    }

    async fn caps_constraint(&mut self) -> Result<CapsConstraint<'_>, G2gError> {
        let caps = self.intercept_caps().await?;
        Ok(CapsConstraint::Produces(CapsSet::one(caps)))
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
            let (mut media, media_url) = resolve_media(&client, &self.url, self.cap()).await?;

            let mut sequence = 0u64;
            // Next HLS media-sequence number to play; segments below it on a live
            // reload were already delivered.
            let mut next_seq = media.media_sequence;
            // fMP4: the EXT-X-MAP init segment (ftyp+moov) is emitted once, before
            // any media fragment, so a downstream fmp4demux sees the moov first.
            let mut init_emitted = false;
            loop {
                if let Some(map) = &media.map_uri {
                    if !init_emitted {
                        let init_url = resolve_url(&media_url, map);
                        let bytes = get_bytes(&client, &init_url).await?;
                        if !bytes.is_empty() {
                            out.push(PipelinePacket::DataFrame(byte_frame(bytes, sequence))).await?;
                            sequence += 1;
                        }
                        init_emitted = true;
                    }
                }
                for (seg_seq, segment) in (media.media_sequence..).zip(media.segments.iter()) {
                    if seg_seq >= next_seq {
                        let seg_url = resolve_url(&media_url, &segment.uri);
                        let bytes = get_bytes(&client, &seg_url).await?;
                        if !bytes.is_empty() {
                            out.push(PipelinePacket::DataFrame(byte_frame(bytes, sequence))).await?;
                            sequence += 1;
                        }
                        next_seq = seg_seq + 1;
                    }
                }

                if media.end_list {
                    break;
                }
                // Live: wait a reload interval, then refetch the media playlist.
                let interval_ms = if self.reload_interval_ms != 0 {
                    self.reload_interval_ms
                } else {
                    u64::from(media.target_duration_secs.max(1)) * 1000
                };
                tokio::time::sleep(core::time::Duration::from_millis(interval_ms)).await;
                let text = get_text(&client, &media_url).await?;
                media = match parse(&text).map_err(|_| G2gError::CapsMismatch)? {
                    Playlist::Media(m) => m,
                    Playlist::Master(_) => return Err(G2gError::CapsMismatch),
                };
            }

            out.push(PipelinePacket::Eos).await?;
            Ok(sequence)
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        HLSSRC_PROPS
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

static HLSSRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new("location", PropKind::Str, "HLS playlist URL (.m3u8)"),
    PropertySpec::new(
        "max-bandwidth",
        PropKind::Uint,
        "ABR cap in bits/sec; 0 selects the highest-bandwidth variant",
    ),
    PropertySpec::new(
        "reload-interval-ms",
        PropKind::Uint,
        "live-playlist reload interval in ms; 0 derives it from TARGETDURATION",
    ),
];
