//! Shared HTTP fetch + URL helpers for the adaptive-streaming sources
//! ([`HlsSrc`](crate::hlssrc), [`DashSrc`](crate::dashsrc)). Thin wrappers over
//! `reqwest`, with bounded retries for media segments and live playlist reloads,
//! plus the relative-URL resolution both need.

use core::future::Future;
use core::time::Duration;
use std::time::Instant;

use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{FrameTiming, G2gError, HardwareError, HttpTransportFailure, MemoryDomain};

/// Body-size cap for manifests and keys (playlists, MPDs, AES-128 keys). These
/// are small by construction; 16 MiB is generous headroom for the largest live
/// VOD playlist while still bounding a hostile/buggy server's response.
pub(crate) const MAX_MANIFEST_BYTES: usize = 16 * 1024 * 1024;

/// Body-size cap for a single media segment. A few seconds of high-bitrate
/// video, with headroom; large enough for legitimate 4K segments, small enough
/// that one bogus segment cannot exhaust memory.
pub(crate) const MAX_SEGMENT_BYTES: usize = 256 * 1024 * 1024;

/// `G2G_DEBUG=fetch:...` names the shared HTTP fetch path in a log spec.
struct FetchLog;

impl g2g_core::log::LogSource for FetchLog {
    fn log_category(&self) -> &'static str {
        "fetch"
    }
}

/// reqwest transport / status failures map to a hardware-ish I/O error; the run
/// fails loud and the pipeline surfaces it with the HTTP status. The URL only
/// survives in the log line: `G2gError` cannot carry it.
pub(crate) fn net_err(e: reqwest::Error) -> G2gError {
    g2g_core::g2g_warn!(&FetchLog, "http fetch failed: {e}");
    crate::httperror::http_err(e)
}

// a body that breaks off after the response head is past the retries, so it ends the run
#[cfg(feature = "dash")]
pub(crate) fn body_read_err(url: &str, e: reqwest::Error) -> G2gError {
    g2g_core::g2g_error!(&FetchLog, "{url}: {e}");
    crate::httperror::http_err(e)
}

/// A response body that exceeds its negotiated cap is treated as a (loud) I/O
/// failure: a remote URL is attacker-controlled, so an over-cap body is a
/// denial-of-service attempt, not a recoverable condition. Same error surface as
/// [`net_err`] so callers need not special-case it.
fn body_too_large() -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

// a failed fetch kept unmapped, so the retry decision reads the reqwest error itself
#[derive(Debug)]
enum FetchFailure {
    Request(reqwest::Error),
    BodyTooLarge,
}

impl FetchFailure {
    fn into_error(self) -> G2gError {
        match self {
            FetchFailure::Request(error) => net_err(error),
            FetchFailure::BodyTooLarge => body_too_large(),
        }
    }

    fn retry_reason(&self) -> Option<RetryReason> {
        let FetchFailure::Request(error) = self else {
            return None;
        };
        match error.status() {
            Some(reqwest::StatusCode::NOT_FOUND | reqwest::StatusCode::TOO_EARLY) => {
                Some(RetryReason::NotPublishedYet)
            }
            Some(status) if status.is_server_error() => Some(RetryReason::ServerOrTransport),
            Some(_) => None,
            None => {
                let transport = error.is_timeout()
                    || error.is_connect()
                    || error.is_request()
                    || error.is_body();
                transport.then_some(RetryReason::ServerOrTransport)
            }
        }
    }
}

impl core::fmt::Display for FetchFailure {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FetchFailure::Request(error) => core::fmt::Display::fmt(error, f),
            FetchFailure::BodyTooLarge => f.write_str("response body over its size cap"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RetryReason {
    // 404 or 425: a live origin whose clock runs behind ours has not published it yet
    NotPublishedYet,
    ServerOrTransport,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FetchRetry {
    // a live segment's not-yet-published answers count against the budget once per this long at most
    pub(crate) live_segment_duration: Option<Duration>,
}

impl FetchRetry {
    pub(crate) const BUDGETED: Self = Self {
        live_segment_duration: None,
    };
}

// gst adaptivedemux2's `max-retries` default
const MAX_FETCH_RETRIES: u32 = 3;
// doubles per counted retry: 250, 500 and 1000 ms outlast a one second origin restart
const FIRST_RETRY_DELAY: Duration = Duration::from_millis(250);

async fn with_retries<T, Attempt>(
    url: &str,
    retry: FetchRetry,
    mut attempt: impl FnMut() -> Attempt,
) -> Result<T, G2gError>
where
    Attempt: Future<Output = Result<T, FetchFailure>>,
{
    let mut counted_retries = 0u32;
    let mut last_counted_not_published: Option<Instant> = None;
    loop {
        let failure = match attempt().await {
            Ok(value) => return Ok(value),
            Err(failure) => failure,
        };
        let Some(reason) = failure.retry_reason() else {
            g2g_core::g2g_error!(&FetchLog, "{url}: {failure}");
            return Err(failure.into_error());
        };
        let counts = match (retry.live_segment_duration, reason) {
            (Some(segment_duration), RetryReason::NotPublishedYet) => {
                last_counted_not_published.is_none_or(|at| at.elapsed() >= segment_duration)
            }
            _ => true,
        };
        if counts {
            if counted_retries == MAX_FETCH_RETRIES {
                g2g_core::g2g_error!(
                    &FetchLog,
                    "{url}: {failure}, gave up after {MAX_FETCH_RETRIES} retries"
                );
                return Err(failure.into_error());
            }
            counted_retries += 1;
            if reason == RetryReason::NotPublishedYet {
                last_counted_not_published = Some(Instant::now());
            }
        }
        let delay = FIRST_RETRY_DELAY * 2u32.pow(counted_retries.saturating_sub(1));
        g2g_core::g2g_warn!(
            &FetchLog,
            "{url}: {failure}, retry {counted_retries} of {MAX_FETCH_RETRIES} in {} ms",
            delay.as_millis()
        );
        tokio::time::sleep(delay).await;
    }
}

// a restarting origin answers 4xx or 5xx, or drops the connection
pub(crate) fn is_origin_failure(error: &G2gError) -> bool {
    match error {
        G2gError::Hardware(HardwareError::Http(status)) => reqwest::StatusCode::from_u16(*status)
            .is_ok_and(|code| code.is_client_error() || code.is_server_error()),
        G2gError::Hardware(HardwareError::HttpTransport(failure)) => matches!(
            failure,
            HttpTransportFailure::Timeout
                | HttpTransportFailure::Connect
                | HttpTransportFailure::Request
                | HttpTransportFailure::Body
        ),
        _ => false,
    }
}

/// GET `url` and hand back the checked response, whose body the caller reads
/// chunk by chunk (what [`get_bytes`] does, and what the DASH low-latency path
/// consumes progressively). The `Content-Length` header (when present) is an
/// early-out against `max`, but it is advisory: a server can omit or lie about
/// it, so the caller still bounds what it accumulates.
async fn try_get_response(
    client: &reqwest::Client,
    url: &str,
    max: usize,
) -> Result<reqwest::Response, FetchFailure> {
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(FetchFailure::Request)?
        .error_for_status()
        .map_err(FetchFailure::Request)?;
    if let Some(len) = resp.content_length() {
        if len > max as u64 {
            return Err(FetchFailure::BodyTooLarge);
        }
    }
    Ok(resp)
}

// retries only until the response head arrives, the body is read by the caller
#[cfg(feature = "dash")]
pub(crate) async fn get_response_retrying(
    client: &reqwest::Client,
    url: &str,
    max: usize,
    retry: FetchRetry,
) -> Result<reqwest::Response, G2gError> {
    with_retries(url, retry, || try_get_response(client, url, max)).await
}

/// Accumulate a response body, never allocating past `max` bytes: the running
/// total over the streamed chunks is the real bound (see [`try_get_response`] for
/// the advisory header check).
async fn read_capped(mut resp: reqwest::Response, max: usize) -> Result<Vec<u8>, FetchFailure> {
    let mut body = Vec::new();
    while let Some(chunk) = resp.chunk().await.map_err(FetchFailure::Request)? {
        if body.len().saturating_add(chunk.len()) > max {
            return Err(FetchFailure::BodyTooLarge);
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

async fn try_get_bytes(
    client: &reqwest::Client,
    url: &str,
    max: usize,
) -> Result<Vec<u8>, FetchFailure> {
    read_capped(try_get_response(client, url, max).await?, max).await
}

/// Fetch `url`, accumulating the body but never allocating past `max` bytes.
pub(crate) async fn get_bytes(
    client: &reqwest::Client,
    url: &str,
    max: usize,
) -> Result<Vec<u8>, G2gError> {
    try_get_bytes(client, url, max)
        .await
        .map_err(FetchFailure::into_error)
}

pub(crate) async fn get_bytes_retrying(
    client: &reqwest::Client,
    url: &str,
    max: usize,
    retry: FetchRetry,
) -> Result<Vec<u8>, G2gError> {
    with_retries(url, retry, || try_get_bytes(client, url, max)).await
}

/// Fetch the byte sub-range `[offset, offset + length)` of `url` via an HTTP
/// `Range` request, capped at `max` like [`get_bytes`]. A `length` of `0`
/// fetches nothing (a malformed `#EXT-X-BYTERANGE` is inert, not fatal). A
/// server that ignores the `Range` (replies `200 OK` with the whole resource)
/// is handled by slicing the requested window from the full body, so a
/// non-range-capable origin still yields the right bytes.
///
/// Used by HLS `#EXT-X-BYTERANGE` and DASH `SegmentList` byte-range addressing.
async fn try_get_range_bytes(
    client: &reqwest::Client,
    url: &str,
    offset: u64,
    length: u64,
    max: usize,
) -> Result<Vec<u8>, FetchFailure> {
    if length == 0 {
        return Ok(Vec::new());
    }
    if length > max as u64 {
        return Err(FetchFailure::BodyTooLarge);
    }
    // Inclusive end per RFC 7233: bytes=offset-(offset+length-1).
    let end = offset.saturating_add(length).saturating_sub(1);
    let range = alloc::format!("bytes={offset}-{end}");
    let resp = client
        .get(url)
        .header(reqwest::header::RANGE, range)
        .send()
        .await
        .map_err(FetchFailure::Request)?
        .error_for_status()
        .map_err(FetchFailure::Request)?;
    // 200 means the server ignored the Range and sent the whole resource; 206 is
    // exactly the requested span. Either way bound the accumulation by `max`.
    let range_ignored = resp.status() == reqwest::StatusCode::OK;
    let mut body = read_capped(resp, max).await?;
    if range_ignored {
        let start = (offset as usize).min(body.len());
        let stop = start.saturating_add(length as usize).min(body.len());
        body = body[start..stop].to_vec();
    }
    Ok(body)
}

pub(crate) async fn get_range_bytes_retrying(
    client: &reqwest::Client,
    url: &str,
    offset: u64,
    length: u64,
    max: usize,
    retry: FetchRetry,
) -> Result<Vec<u8>, G2gError> {
    with_retries(url, retry, || {
        try_get_range_bytes(client, url, offset, length, max)
    })
    .await
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

// for a live playlist reload, which a failure must not end
pub(crate) async fn get_text_retrying(
    client: &reqwest::Client,
    url: &str,
    max: usize,
    retry: FetchRetry,
) -> Result<String, G2gError> {
    let bytes = get_bytes_retrying(client, url, max, retry).await?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Capped text fetch with `query` appended to the URL and a hard `timeout`, for
/// the LL-HLS blocking playlist reload: the `_HLS_msn` / `_HLS_part` delivery
/// directives ask the server to hold the response open until that partial
/// segment is published, so the request needs its own deadline rather than the
/// client's (absent) default. Existing query parameters on `url` are kept, so a
/// signed CDN playlist URL still works.
#[cfg(feature = "hls")]
pub(crate) async fn get_text_query(
    client: &reqwest::Client,
    url: &str,
    query: &[(&str, u64)],
    timeout: core::time::Duration,
    max: usize,
) -> Result<String, G2gError> {
    let resp = client
        .get(url)
        .query(query)
        .timeout(timeout)
        .send()
        .await
        .map_err(net_err)?
        .error_for_status()
        .map_err(net_err)?;
    let bytes = read_capped(resp, max)
        .await
        .map_err(FetchFailure::into_error)?;
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
        let authority_end = base[scheme_end..]
            .find('/')
            .map(|i| scheme_end + i)
            .unwrap_or(base.len());
        let mut out = String::from(&base[..authority_end]);
        out.push('/');
        out.push_str(stripped);
        out
    } else {
        // relative to the playlist's directory (everything up to the last '/').
        // Scan only the path, not the query/fragment, so a '/' inside a signed
        // CDN query string is not mistaken for a path separator.
        let path_end = base.find(['?', '#']).unwrap_or(base.len());
        let dir_end = base[..path_end]
            .rfind('/')
            .map(|i| i + 1)
            .unwrap_or(path_end);
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
