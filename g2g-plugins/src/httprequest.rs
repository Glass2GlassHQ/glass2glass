use alloc::string::{String, ToString};
use alloc::vec::Vec;

use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::task::JoinHandle;

pub(crate) const HEAD_TERMINATOR: &[u8] = b"\r\n\r\n";
pub(crate) const MAX_HEAD_BYTES: usize = 64 * 1024;
pub(crate) const MAX_BODY_BYTES: usize = 1024 * 1024;
pub(crate) const READ_CHUNK_BYTES: usize = 4096;

#[derive(Debug)]
pub(crate) struct AbortOnDrop(pub(crate) JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[derive(Debug, Default)]
pub(crate) struct HttpRequest {
    pub(crate) method: String,
    pub(crate) path: String,
    pub(crate) authorization: Option<String>,
    pub(crate) content_type: Option<String>,
    pub(crate) if_match: Option<String>,
    pub(crate) range: Option<String>,
    pub(crate) body: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RequestError {
    Malformed,
    TooLarge,
}

/// One complete request from `buf`, `None` until the head and body have arrived.
pub(crate) fn parse_request(buf: &[u8]) -> Result<Option<HttpRequest>, RequestError> {
    let head_end = buf
        .windows(HEAD_TERMINATOR.len())
        .position(|window| window == HEAD_TERMINATOR);
    let Some(head_end) = head_end else {
        if buf.len() > MAX_HEAD_BYTES {
            return Err(RequestError::TooLarge);
        }
        return Ok(None);
    };
    if head_end > MAX_HEAD_BYTES {
        return Err(RequestError::TooLarge);
    }
    let head = core::str::from_utf8(&buf[..head_end]).map_err(|_| RequestError::Malformed)?;
    let mut lines = head.split("\r\n");
    let mut request_line = lines.next().unwrap_or_default().split(' ');
    let (Some(method), Some(path)) = (request_line.next(), request_line.next()) else {
        return Err(RequestError::Malformed);
    };
    let mut request = HttpRequest {
        method: method.to_string(),
        path: path.to_string(),
        ..HttpRequest::default()
    };
    let mut content_length = 0usize;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        if name.eq_ignore_ascii_case("Content-Length") {
            content_length = value.parse().map_err(|_| RequestError::Malformed)?;
        } else if name.eq_ignore_ascii_case("Authorization") {
            request.authorization = Some(value.to_string());
        } else if name.eq_ignore_ascii_case("Content-Type") {
            request.content_type = Some(value.to_string());
        } else if name.eq_ignore_ascii_case("If-Match") {
            request.if_match = Some(value.to_string());
        } else if name.eq_ignore_ascii_case("Range") {
            request.range = Some(value.to_string());
        }
    }
    if content_length > MAX_BODY_BYTES {
        return Err(RequestError::TooLarge);
    }
    let body_start = head_end.saturating_add(HEAD_TERMINATOR.len());
    let body_end = body_start.saturating_add(content_length);
    let Some(body) = buf.get(body_start..body_end) else {
        return Ok(None);
    };
    request.body = core::str::from_utf8(body)
        .map_err(|_| RequestError::Malformed)?
        .to_string();
    Ok(Some(request))
}

pub(crate) async fn read_request<S: AsyncRead + Unpin>(
    stream: &mut S,
) -> Result<HttpRequest, RequestError> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; READ_CHUNK_BYTES];
    loop {
        if let Some(request) = parse_request(&buf)? {
            return Ok(request);
        }
        let n = stream
            .read(&mut chunk)
            .await
            .map_err(|_| RequestError::Malformed)?;
        if n == 0 {
            return Err(RequestError::Malformed);
        }
        buf.extend_from_slice(&chunk[..n]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn parser_reads_head_and_body() {
        let raw = b"POST /whip HTTP/1.1\r\ncontent-LENGTH: 4\r\nCONTENT-TYPE: application/sdp\r\nauthorization: Bearer t\r\n\r\nv=0\n";
        let parsed = parse_request(raw).unwrap().unwrap();
        assert_eq!(parsed.method, "POST");
        assert_eq!(parsed.path, "/whip");
        assert_eq!(parsed.content_type.as_deref(), Some("application/sdp"));
        assert_eq!(parsed.authorization.as_deref(), Some("Bearer t"));
        assert_eq!(parsed.body, "v=0\n");
        assert!(parse_request(&raw[..raw.len() - 1]).unwrap().is_none());
        assert!(parse_request(b"POST /whip HTTP/1.1\r\n").unwrap().is_none());
    }

    #[test]
    fn parser_rejects_oversize_and_garbage() {
        let huge_body = format!(
            "POST / HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            MAX_BODY_BYTES + 1
        );
        assert_eq!(
            parse_request(huge_body.as_bytes()).unwrap_err(),
            RequestError::TooLarge
        );
        let endless_head = alloc::vec![b'a'; MAX_HEAD_BYTES + 1];
        assert_eq!(
            parse_request(&endless_head).unwrap_err(),
            RequestError::TooLarge
        );
        let bad_length = b"POST / HTTP/1.1\r\nContent-Length: lots\r\n\r\n";
        assert_eq!(
            parse_request(bad_length).unwrap_err(),
            RequestError::Malformed
        );
        let overflowing = format!("POST / HTTP/1.1\r\nContent-Length: {}\r\n\r\n", usize::MAX);
        assert_eq!(
            parse_request(overflowing.as_bytes()).unwrap_err(),
            RequestError::TooLarge
        );
    }
}
