//! Shared HTTP fetch + URL helpers for the adaptive-streaming sources
//! ([`HlsSrc`](crate::hlssrc), [`DashSrc`](crate::dashsrc)). Thin wrappers over
//! `reqwest` plus the relative-URL resolution both need.

use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{FrameTiming, G2gError, HardwareError, MemoryDomain};

/// Body-size cap for manifests and keys (playlists, MPDs, AES-128 keys). These
/// are small by construction; 16 MiB is generous headroom for the largest live
/// VOD playlist while still bounding a hostile/buggy server's response.
pub(crate) const MAX_MANIFEST_BYTES: usize = 16 * 1024 * 1024;

/// Body-size cap for a single media segment. A few seconds of high-bitrate
/// video, with headroom; large enough for legitimate 4K segments, small enough
/// that one bogus segment cannot exhaust memory.
pub(crate) const MAX_SEGMENT_BYTES: usize = 256 * 1024 * 1024;

/// reqwest transport / status failures map to a hardware-ish I/O error; the run
/// fails loud and the pipeline surfaces it.
pub(crate) fn net_err(_e: reqwest::Error) -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

/// A response body that exceeds its negotiated cap is treated as a (loud) I/O
/// failure: a remote URL is attacker-controlled, so an over-cap body is a
/// denial-of-service attempt, not a recoverable condition. Same error surface as
/// [`net_err`] so callers need not special-case it.
fn body_too_large() -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

/// Fetch `url`, accumulating the body but never allocating past `max` bytes.
/// The `Content-Length` header (when present) is an early-out, but it is
/// advisory: a server can omit or lie about it, so the running total over the
/// streamed chunks is the real bound.
pub(crate) async fn get_bytes(
    client: &reqwest::Client,
    url: &str,
    max: usize,
) -> Result<Vec<u8>, G2gError> {
    let mut resp =
        client.get(url).send().await.map_err(net_err)?.error_for_status().map_err(net_err)?;
    if let Some(len) = resp.content_length() {
        if len > max as u64 {
            return Err(body_too_large());
        }
    }
    let mut body = Vec::new();
    while let Some(chunk) = resp.chunk().await.map_err(net_err)? {
        if body.len().saturating_add(chunk.len()) > max {
            return Err(body_too_large());
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

/// Fetch the byte sub-range `[offset, offset + length)` of `url` via an HTTP
/// `Range` request, capped at `max` like [`get_bytes`]. A `length` of `0`
/// fetches nothing (a malformed `#EXT-X-BYTERANGE` is inert, not fatal). A
/// server that ignores the `Range` (replies `200 OK` with the whole resource)
/// is handled by slicing the requested window from the full body, so a
/// non-range-capable origin still yields the right bytes.
///
/// Used by HLS `#EXT-X-BYTERANGE` and DASH `SegmentList` byte-range addressing.
#[cfg(any(feature = "hls", feature = "dash"))]
pub(crate) async fn get_range_bytes(
    client: &reqwest::Client,
    url: &str,
    offset: u64,
    length: u64,
    max: usize,
) -> Result<Vec<u8>, G2gError> {
    if length == 0 {
        return Ok(Vec::new());
    }
    if length > max as u64 {
        return Err(body_too_large());
    }
    // Inclusive end per RFC 7233: bytes=offset-(offset+length-1).
    let end = offset.saturating_add(length).saturating_sub(1);
    let range = alloc::format!("bytes={offset}-{end}");
    let resp = client
        .get(url)
        .header(reqwest::header::RANGE, range)
        .send()
        .await
        .map_err(net_err)?
        .error_for_status()
        .map_err(net_err)?;
    // 200 means the server ignored the Range and sent the whole resource; 206 is
    // exactly the requested span. Either way bound the accumulation by `max`.
    let range_ignored = resp.status() == reqwest::StatusCode::OK;
    let mut resp = resp;
    let mut body = Vec::new();
    while let Some(chunk) = resp.chunk().await.map_err(net_err)? {
        if body.len().saturating_add(chunk.len()) > max {
            return Err(body_too_large());
        }
        body.extend_from_slice(&chunk);
    }
    if range_ignored {
        let start = (offset as usize).min(body.len());
        let stop = start.saturating_add(length as usize).min(body.len());
        body = body[start..stop].to_vec();
    }
    Ok(body)
}

/// Capped text fetch: reuses [`get_bytes`] for the size bound, then decodes as
/// UTF-8 (HLS playlists and DASH MPDs are UTF-8 by spec; a stray byte decodes
/// lossily rather than failing the fetch, matching reqwest's `text()`).
pub(crate) async fn get_text(
    client: &reqwest::Client,
    url: &str,
    max: usize,
) -> Result<String, G2gError> {
    let bytes = get_bytes(client, url, max).await?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
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
        // relative to the playlist's directory (everything up to the last '/').
        // Scan only the path, not the query/fragment, so a '/' inside a signed
        // CDN query string is not mistaken for a path separator.
        let path_end = base.find(['?', '#']).unwrap_or(base.len());
        let dir_end = base[..path_end].rfind('/').map(|i| i + 1).unwrap_or(path_end);
        let mut out = String::from(&base[..dir_end]);
        out.push_str(rel);
        out
    }
}

#[cfg(test)]
mod cap_tests {
    //! The body cap is the DASH/HLS DoS defense: a remote URL is
    //! attacker-controlled, so one oversized response must not exhaust memory.
    //! A local one-shot HTTP/1.1 server (no extra deps) serves a known body; the
    //! tests drive the real `get_bytes` over a real `reqwest` client and assert
    //! both bound mechanisms, the `Content-Length` early-out and the streamed
    //! running-total, while an under-cap fetch still returns the whole body.
    use super::{get_bytes, MAX_SEGMENT_BYTES};
    use alloc::format;
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    /// Serve `body` once. When `declare_length` is set, send an honest
    /// `Content-Length`; otherwise frame the body by connection close (no
    /// `Content-Length`), forcing the streamed-accumulator path. Returns the URL.
    fn serve_once(body: Vec<u8>, declare_length: bool) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut req = Vec::new();
            let mut byte = [0u8; 1];
            while stream.read(&mut byte).unwrap_or(0) == 1 {
                req.push(byte[0]);
                if req.ends_with(b"\r\n\r\n") {
                    break;
                }
            }
            let header = if declare_length {
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
            } else {
                String::from("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n")
            };
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(&body);
            let _ = stream.flush();
        });
        format!("http://127.0.0.1:{port}/body")
    }

    #[tokio::test]
    async fn rejects_over_cap_via_content_length() {
        // A complete, honest body over a small cap: the declared length trips the
        // early-out before any body is buffered. (Without the cap this fetch
        // succeeds and returns all 4096 bytes, so the assertion is discriminating.)
        let url = serve_once(vec![7u8; 4096], true);
        let client = reqwest::Client::new();
        assert!(get_bytes(&client, &url, 1024).await.is_err());
    }

    #[tokio::test]
    async fn rejects_over_cap_via_streamed_total() {
        // No Content-Length, so the early-out cannot fire: the running total over
        // the streamed chunks is the only thing that can stop an over-cap body.
        let url = serve_once(vec![9u8; 4096], false);
        let client = reqwest::Client::new();
        assert!(get_bytes(&client, &url, 1024).await.is_err());
    }

    #[tokio::test]
    async fn accepts_under_cap_body_whole() {
        // An under-cap fetch returns the entire body unchanged.
        let body = vec![3u8; 4096];
        let url = serve_once(body.clone(), true);
        let client = reqwest::Client::new();
        let got = get_bytes(&client, &url, MAX_SEGMENT_BYTES).await.unwrap();
        assert_eq!(got, body);
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

    #[test]
    fn signed_query_slash_is_not_a_path_separator() {
        // A '/' inside the base's signed query must not become the directory
        // boundary when resolving a relative segment name.
        let base = "http://h/v/media.m3u8?token=ab/cd";
        assert_eq!(resolve_url(base, "seg0.ts"), "http://h/v/seg0.ts");
    }
}
