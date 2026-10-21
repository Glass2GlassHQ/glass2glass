#![cfg(feature = "http-src")]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use g2g_core::runtime::SourceLoop;
use g2g_core::{ByteStreamEncoding, Caps, G2gError, HardwareError, HttpTransportFailure};

#[path = "common/collect_sink.rs"]
mod collect_sink;
use collect_sink::CollectSink;

const TOO_MANY_REQUESTS: u16 = 429;
const NOT_FOUND: u16 = 404;

// answers every request with `status` and no body, as a rate-limiting origin does
fn serve_status(status: u16) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        for mut stream in listener.incoming().flatten() {
            let mut request = Vec::new();
            let mut byte = [0u8; 1];
            while stream.read(&mut byte).unwrap_or(0) == 1 {
                request.push(byte[0]);
                if request.ends_with(b"\r\n\r\n") {
                    break;
                }
            }
            let _ = stream.write_all(
                format!(
                    "HTTP/1.1 {status} Refused\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                )
                .as_bytes(),
            );
        }
    });
    format!("http://127.0.0.1:{port}")
}

async fn run_error(mut src: impl SourceLoop, caps: Caps) -> G2gError {
    src.configure_pipeline(&caps).unwrap();
    src.run(&mut CollectSink::default())
        .await
        .expect_err("the server refused every request")
}

fn http_status(status: u16) -> G2gError {
    G2gError::Hardware(HardwareError::Http(status))
}

// a port that was bound and released, so a connection to it is refused
fn refused_origin() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    format!("http://127.0.0.1:{port}")
}

fn refused_connection() -> G2gError {
    G2gError::Hardware(HardwareError::HttpTransport(HttpTransportFailure::Connect))
}

#[cfg(feature = "dash")]
#[tokio::test]
async fn a_refused_dash_manifest_names_its_status() {
    for status in [TOO_MANY_REQUESTS, NOT_FOUND] {
        let url = format!("{}/Manifest.mpd", serve_status(status));
        let src = g2g_plugins::dashsrc::DashSrc::new(url);
        let caps = Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        };
        assert_eq!(run_error(src, caps).await, http_status(status));
    }
}

#[cfg(feature = "hls")]
#[tokio::test]
async fn a_refused_hls_playlist_names_its_status() {
    for status in [TOO_MANY_REQUESTS, NOT_FOUND] {
        let url = format!("{}/index.m3u8", serve_status(status));
        let src = g2g_plugins::hlssrc::HlsSrc::new(url);
        let caps = Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        };
        assert_eq!(run_error(src, caps).await, http_status(status));
    }
}

#[tokio::test]
async fn a_refused_http_stream_names_its_status() {
    for status in [TOO_MANY_REQUESTS, NOT_FOUND] {
        let caps = Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        };
        let src = g2g_plugins::httpsrc::HttpSrc::new(
            format!("{}/stream.ts", serve_status(status)),
            caps.clone(),
        );
        assert_eq!(run_error(src, caps).await, http_status(status));
    }
}

#[cfg(feature = "dash")]
#[tokio::test]
async fn an_unreachable_dash_manifest_names_the_refused_connection() {
    let src = g2g_plugins::dashsrc::DashSrc::new(format!("{}/Manifest.mpd", refused_origin()));
    let caps = Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    };
    assert_eq!(run_error(src, caps).await, refused_connection());
}

#[cfg(feature = "hls")]
#[tokio::test]
async fn an_unreachable_hls_playlist_names_the_refused_connection() {
    let src = g2g_plugins::hlssrc::HlsSrc::new(format!("{}/index.m3u8", refused_origin()));
    let caps = Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    };
    assert_eq!(run_error(src, caps).await, refused_connection());
}

#[tokio::test]
async fn an_unreachable_http_stream_names_the_refused_connection() {
    let caps = Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    };
    let src = g2g_plugins::httpsrc::HttpSrc::new(
        format!("{}/stream.ts", refused_origin()),
        caps.clone(),
    );
    assert_eq!(run_error(src, caps).await, refused_connection());
}
