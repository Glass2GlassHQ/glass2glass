// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::collections::HashMap;
use std::io;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

const HEAD_LIMIT: usize = 16 * 1024;
const BODY_LIMIT: usize = 1024;
const HEADER_COUNT: usize = 32;
const READ_CHUNK: usize = 2048;

#[derive(Debug)]
pub(crate) struct Request {
    pub(crate) method: String,
    pub(crate) path: String,
    pub(crate) query: HashMap<String, String>,
    pub(crate) body: Vec<u8>,
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

pub(crate) async fn read_request<S: AsyncRead + Unpin>(
    stream: &mut S,
) -> io::Result<Option<Request>> {
    let mut buffer = Vec::with_capacity(READ_CHUNK);
    let mut chunk = [0u8; READ_CHUNK];
    let (method, target, head_length, body_length) = loop {
        let read = stream.read(&mut chunk).await?;
        // moonlight connects without a request to read its local interface and MTU
        if read == 0 && buffer.is_empty() {
            return Ok(None);
        }
        if read == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        buffer.extend_from_slice(&chunk[..read]);
        let mut headers = [httparse::EMPTY_HEADER; HEADER_COUNT];
        let mut parsed = httparse::Request::new(&mut headers);
        let status = parsed
            .parse(&buffer)
            .map_err(|_| invalid("malformed HTTP request"))?;
        if let httparse::Status::Complete(head_length) = status {
            let body_length = match parsed
                .headers
                .iter()
                .find(|header| header.name.eq_ignore_ascii_case("content-length"))
            {
                Some(header) => std::str::from_utf8(header.value)
                    .ok()
                    .and_then(|value| value.trim().parse::<usize>().ok())
                    .ok_or_else(|| invalid("bad Content-Length"))?,
                None => 0,
            };
            let method = parsed.method.unwrap_or_default().to_string();
            let target = parsed.path.unwrap_or_default().to_string();
            break (method, target, head_length, body_length);
        }
        if buffer.len() > HEAD_LIMIT {
            return Err(invalid("HTTP request head too long"));
        }
    };
    if body_length > BODY_LIMIT {
        return Err(invalid("HTTP request body too long"));
    }
    let mut body = buffer.split_off(head_length);
    while body.len() < body_length {
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        body.extend_from_slice(&chunk[..read]);
    }
    body.truncate(body_length);
    let (path, query) = target.split_once('?').unwrap_or((target.as_str(), ""));
    Ok(Some(Request {
        method,
        path: path.to_string(),
        query: form_urlencoded::parse(query.as_bytes())
            .into_owned()
            .collect(),
        body,
    }))
}

#[derive(Debug)]
pub(crate) struct Reply {
    code: u16,
    content_type: &'static str,
    body: String,
}

impl Reply {
    pub(crate) fn xml(inner: &str) -> Self {
        Self {
            code: 200,
            content_type: "application/xml",
            body: format!("<root status_code=\"200\">{inner}</root>"),
        }
    }

    // moonlight-qt reads the body only on HTTP 200, so the GameStream status rides in the XML
    pub(crate) fn xml_error(status: u16, message: &str) -> Self {
        Self {
            code: 200,
            content_type: "application/xml",
            body: format!(
                "<root status_code=\"{status}\" status_message=\"{}\"></root>",
                escape_xml(message)
            ),
        }
    }

    pub(crate) fn text(code: u16, message: &str) -> Self {
        Self {
            code,
            content_type: "text/plain; charset=utf-8",
            body: message.to_string(),
        }
    }

    pub(crate) fn not_found() -> Self {
        Self::text(404, "not found")
    }
}

fn reason(code: u16) -> &'static str {
    match code {
        200 => "OK",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        409 => "Conflict",
        _ => "Error",
    }
}

pub(crate) async fn write_reply<S: AsyncWrite + Unpin>(
    stream: &mut S,
    reply: Reply,
) -> io::Result<()> {
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        reply.code,
        reason(reply.code),
        reply.content_type,
        reply.body.len()
    );
    stream.write_all(head.as_bytes()).await?;
    stream.write_all(reply.body.as_bytes()).await?;
    stream.shutdown().await
}

pub(crate) fn escape_xml(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&apos;"),
            other => escaped.push(other),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_request_reads_its_query_and_body() {
        let wire = b"POST /submit-pin?uniqueid=ab%20c HTTP/1.1\r\nHost: x\r\nContent-Length: 8\r\n\r\npin=1234";
        let request = read_request(&mut &wire[..])
            .await
            .expect("parsed")
            .expect("a request");
        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/submit-pin");
        assert_eq!(
            request.query.get("uniqueid").map(String::as_str),
            Some("ab c")
        );
        assert_eq!(request.body, b"pin=1234");
    }

    #[tokio::test]
    async fn an_endless_head_or_a_large_body_is_refused() {
        let mut endless = b"GET / HTTP/1.1\r\nX-Long: ".to_vec();
        endless.extend(std::iter::repeat_n(b'a', HEAD_LIMIT + READ_CHUNK));
        assert!(read_request(&mut endless.as_slice()).await.is_err());
        let large = format!(
            "POST / HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            BODY_LIMIT + 1
        );
        assert!(read_request(&mut large.as_bytes()).await.is_err());
    }

    #[tokio::test]
    async fn a_connection_without_bytes_is_no_request_and_a_cut_head_fails() {
        assert!(read_request(&mut &b""[..]).await.expect("closed").is_none());
        assert!(read_request(&mut &b"GET / HTTP/1.1\r\n"[..]).await.is_err());
    }
}
