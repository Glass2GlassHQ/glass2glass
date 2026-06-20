//! Shared HTTP fetch + URL helpers for the adaptive-streaming sources
//! ([`HlsSrc`](crate::hlssrc), [`DashSrc`](crate::dashsrc)). Thin wrappers over
//! `reqwest` plus the relative-URL resolution both need.

use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{FrameTiming, G2gError, HardwareError, MemoryDomain};

/// reqwest transport / status failures map to a hardware-ish I/O error; the run
/// fails loud and the pipeline surfaces it.
pub(crate) fn net_err(_e: reqwest::Error) -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

pub(crate) async fn get_bytes(client: &reqwest::Client, url: &str) -> Result<Vec<u8>, G2gError> {
    let resp = client.get(url).send().await.map_err(net_err)?.error_for_status().map_err(net_err)?;
    Ok(resp.bytes().await.map_err(net_err)?.to_vec())
}

pub(crate) async fn get_text(client: &reqwest::Client, url: &str) -> Result<String, G2gError> {
    let resp = client.get(url).send().await.map_err(net_err)?.error_for_status().map_err(net_err)?;
    resp.text().await.map_err(net_err)
}

/// A `DataFrame`-ready system-memory frame for a fetched byte chunk.
pub(crate) fn byte_frame(bytes: Vec<u8>, sequence: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            arrival_ns: g2g_core::metrics::monotonic_ns(),
            ..FrameTiming::default()
        },
        sequence,
        meta: Default::default(),
    }
}

/// Resolve a possibly-relative URI against a base URL. Handles absolute URLs,
/// absolute paths (`/a/b`), and path-relative names; the HLS/DASH cases in
/// practice. Not a full RFC 3986 resolver (no `..` collapsing).
pub(crate) fn resolve_url(base: &str, rel: &str) -> String {
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
