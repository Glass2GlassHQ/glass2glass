//! HLS source (HlsSrc, `hls` feature): fetches an `.m3u8` playlist, selects a
//! variant (simple bandwidth-capped ABR), and streams that variant's MPEG-TS
//! media segments downstream as a `Caps::ByteStream{MpegTs}` for `tsdemux`, then
//! `Eos`. The [`hls`](crate::hls) parser does the playlist work; this element
//! adds the fetching (via `reqwest`, like [`HttpSrc`](crate::httpsrc)) and URL
//! resolution.
//!
//! Scope (v1): VOD, in-order segment fetch, one `DataFrame` per segment. Live
//! playlist reload, fMP4/CMAF segments, byte-range segments, key/decryption, and
//! throughput-driven ABR are follow-ups (DESIGN_TODO).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, FrameTiming, G2gError,
    HardwareError, MemoryDomain, OutputSink, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec,
};

use crate::hls::{parse, Playlist};

#[derive(Debug)]
pub struct HlsSrc {
    url: String,
    /// ABR cap: select the highest-bandwidth variant at or below this (0 = no
    /// cap, pick the highest available).
    max_bandwidth: u64,
    configured: bool,
}

impl HlsSrc {
    pub fn new(url: impl Into<String>) -> Self {
        Self { url: url.into(), max_bandwidth: 0, configured: false }
    }

    /// Cap variant selection to this bitrate (bits/sec); 0 picks the highest.
    pub fn with_max_bandwidth(mut self, bits_per_sec: u64) -> Self {
        self.max_bandwidth = bits_per_sec;
        self
    }

    fn output_caps() -> Caps {
        Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs }
    }
}

fn net_err(_e: reqwest::Error) -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

async fn get_bytes(client: &reqwest::Client, url: &str) -> Result<Vec<u8>, G2gError> {
    let resp = client.get(url).send().await.map_err(net_err)?.error_for_status().map_err(net_err)?;
    Ok(resp.bytes().await.map_err(net_err)?.to_vec())
}

async fn get_text(client: &reqwest::Client, url: &str) -> Result<String, G2gError> {
    let resp = client.get(url).send().await.map_err(net_err)?.error_for_status().map_err(net_err)?;
    resp.text().await.map_err(net_err)
}

/// Resolve a possibly-relative playlist/segment URI against the playlist URL.
/// Handles absolute URLs, absolute paths (`/a/b`), and path-relative names; the
/// HLS cases in practice. Not a full RFC 3986 resolver (no `..` collapsing).
fn resolve_url(base: &str, rel: &str) -> String {
    if rel.starts_with("http://") || rel.starts_with("https://") {
        return String::from(rel);
    }
    let scheme_end = base.find("://").map(|i| i + 3).unwrap_or(0);
    if let Some(stripped) = rel.strip_prefix('/') {
        // absolute path: keep scheme://authority, replace the path
        let authority_end =
            base[scheme_end..].find('/').map(|i| scheme_end + i).unwrap_or(base.len());
        let mut out = String::from(&base[..authority_end]);
        out.push('/');
        out.push_str(stripped);
        out
    } else {
        // relative to the playlist's directory (everything up to the last '/')
        let dir_end = base.rfind('/').map(|i| i + 1).unwrap_or(base.len());
        let mut out = String::from(&base[..dir_end]);
        out.push_str(rel);
        out
    }
}

impl SourceLoop for HlsSrc {
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

            // Resolve a master playlist down to a media playlist + its base URL.
            let text = get_text(&client, &self.url).await?;
            let (media, base) = match parse(&text).map_err(|_| G2gError::CapsMismatch)? {
                Playlist::Media(m) => (m, self.url.clone()),
                Playlist::Master(master) => {
                    let variant = master.select(cap).ok_or(G2gError::CapsMismatch)?;
                    let media_url = resolve_url(&self.url, &variant.uri);
                    let media_text = get_text(&client, &media_url).await?;
                    match parse(&media_text).map_err(|_| G2gError::CapsMismatch)? {
                        Playlist::Media(m) => (m, media_url),
                        // A master pointing at another master is malformed.
                        Playlist::Master(_) => return Err(G2gError::CapsMismatch),
                    }
                }
            };

            let mut sequence = 0u64;
            for segment in &media.segments {
                let seg_url = resolve_url(&base, &segment.uri);
                let bytes = get_bytes(&client, &seg_url).await?;
                if bytes.is_empty() {
                    continue;
                }
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                    timing: FrameTiming {
                        arrival_ns: g2g_core::metrics::monotonic_ns(),
                        ..FrameTiming::default()
                    },
                    sequence,
                    meta: Default::default(),
                };
                sequence += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
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
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.url.clone())),
            "max-bandwidth" => Some(PropValue::Uint(self.max_bandwidth)),
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
];

#[cfg(test)]
mod tests {
    use super::resolve_url;

    #[test]
    fn resolves_relative_absolute_and_full_uris() {
        let base = "http://h/v/media.m3u8";
        assert_eq!(resolve_url(base, "seg0.ts"), "http://h/v/seg0.ts");
        assert_eq!(resolve_url(base, "/x/seg0.ts"), "http://h/x/seg0.ts");
        assert_eq!(resolve_url(base, "http://o/s.ts"), "http://o/s.ts");
    }
}
