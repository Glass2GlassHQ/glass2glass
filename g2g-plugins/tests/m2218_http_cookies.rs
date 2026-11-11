// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "hls")]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use g2g_core::runtime::SourceLoop;
use g2g_core::{
    ByteStreamEncoding, Caps, G2gError, OutputSink, PipelinePacket, PropValue, PushOutcome,
};
use g2g_plugins::hlssrc::HlsSrc;
use g2g_plugins::httpsrc::HttpSrc;

const USER_COOKIE: &str = "token=xyz";
const SESSION_COOKIE: &str = "session=abc";
const PLAYLIST_PATH: &str = "/live/index.m3u8";
const SEGMENT_PATH: &str = "/media/seg0.ts";
const SEGMENT: &[u8] = b"segment bytes";

// segments sit outside the playlist's directory, where a cookie's default path does not reach
const PLAYLIST: &str = "#EXTM3U\n\
    #EXT-X-TARGETDURATION:4\n\
    #EXT-X-MEDIA-SEQUENCE:0\n\
    #EXTINF:4.0,\n\
    /media/seg0.ts\n\
    #EXT-X-ENDLIST\n";

fn request_cookies(request: &str) -> Vec<String> {
    request
        .lines()
        .filter_map(|line| line.split_once(':'))
        .filter(|(name, _)| name.eq_ignore_ascii_case("cookie"))
        .flat_map(|(_, value)| value.split(';').map(|pair| pair.trim().to_owned()))
        .collect()
}

// the playlist needs the user's cookie and sets a session cookie, segments need both, anything else is a 403
fn serve() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        for mut stream in listener.incoming().map_while(Result::ok) {
            let mut request = Vec::new();
            let mut byte = [0u8; 1];
            while stream.read(&mut byte).unwrap_or(0) == 1 {
                request.push(byte[0]);
                if request.ends_with(b"\r\n\r\n") {
                    break;
                }
            }
            let request = String::from_utf8_lossy(&request);
            let path = request.split_whitespace().nth(1).unwrap_or("");
            let cookies = request_cookies(&request);
            let has = |cookie: &str| cookies.iter().any(|sent| sent == cookie);
            let (extra_header, body) = match path {
                PLAYLIST_PATH if has(USER_COOKIE) => (
                    format!("Set-Cookie: {SESSION_COOKIE}; Path=/\r\n"),
                    PLAYLIST.as_bytes(),
                ),
                SEGMENT_PATH if has(USER_COOKIE) && has(SESSION_COOKIE) => (String::new(), SEGMENT),
                _ => {
                    let _ = stream.write_all(
                        b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                    continue;
                }
            };
            let header = format!(
                "HTTP/1.1 200 OK\r\n{extra_header}Content-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(body);
        }
    });
    format!("http://127.0.0.1:{port}")
}

#[derive(Default)]
struct Body(Vec<u8>);

impl OutputSink for Body {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::DataFrame(frame)) = packet_slot.take() {
            self.0
                .extend_from_slice(frame.domain.as_system_slice().expect("system memory"));
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

#[tokio::test]
async fn hlssrc_sends_its_cookies_and_the_playlist_session_cookie_with_segments() {
    let origin = serve();
    let mut src = HlsSrc::new(format!("{origin}{PLAYLIST_PATH}"));
    src.set_property("cookies", PropValue::Str(USER_COOKIE.into()))
        .unwrap();
    // the probe fetches the playlist, so the run fetches only the segment
    src.caps_constraint().await.expect("the playlist answers");
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .unwrap();
    let mut body = Body::default();
    src.run(&mut body).await.expect("the segment answers");
    assert_eq!(body.0, SEGMENT);
}

#[tokio::test]
async fn hlssrc_without_cookies_is_refused() {
    let origin = serve();
    let mut src = HlsSrc::new(format!("{origin}{PLAYLIST_PATH}"));
    assert!(src.caps_constraint().await.is_err());
}

#[tokio::test]
async fn httpsrc_sends_its_cookies() {
    let origin = serve();
    let mut src = HttpSrc::new(
        format!("{origin}{PLAYLIST_PATH}"),
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        },
    );
    src.set_property("cookies", PropValue::Str(USER_COOKIE.into()))
        .unwrap();
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .unwrap();
    let mut body = Body::default();
    src.run(&mut body).await.expect("the playlist answers");
    assert_eq!(body.0, PLAYLIST.as_bytes());
}

#[test]
fn a_cookie_without_a_name_is_refused() {
    let mut src = HlsSrc::new("http://127.0.0.1/index.m3u8");
    assert!(src
        .set_property("cookies", PropValue::Str("=value".into()))
        .is_err());
    assert!(src
        .set_property("cookies", PropValue::Str("a=1; b=2".into()))
        .is_ok());
}
