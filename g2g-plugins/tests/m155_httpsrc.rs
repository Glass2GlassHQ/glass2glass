//! M155 HTTP byte-stream source: `HttpSrc` GETs a URL and streams the response
//! body downstream as `Caps::ByteStream` `DataFrame` chunks, then `Eos`. A
//! local one-shot TCP server (no extra deps) serves a known payload; the test
//! asserts the reassembled bytes, the produced caps, and the EOS terminator.

#![cfg(feature = "http-src")]

use core::future::Future;
use core::pin::Pin;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use g2g_core::runtime::SourceLoop;
use g2g_core::{
    ByteStreamEncoding, Caps, CapsConstraint, CapsSet, G2gError, OutputSink, PipelinePacket,
    PushOutcome,
};
use g2g_plugins::httpsrc::HttpSrc;

#[derive(Default)]
struct CaptureSink {
    body: Vec<u8>,
    frames: usize,
    eos: bool,
}

impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.body.extend_from_slice(s);
                        self.frames += 1;
                    }
                }
                PipelinePacket::Eos => self.eos = true,
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Serve `payload` once over HTTP/1.1 on an ephemeral port; returns the URL.
fn serve_once(payload: Vec<u8>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        // Drain the request headers (up to the blank line).
        let mut req = Vec::new();
        let mut byte = [0u8; 1];
        while stream.read(&mut byte).unwrap_or(0) == 1 {
            req.push(byte[0]);
            if req.ends_with(b"\r\n\r\n") {
                break;
            }
        }
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            payload.len()
        );
        stream.write_all(header.as_bytes()).unwrap();
        stream.write_all(&payload).unwrap();
        stream.flush().unwrap();
    });
    format!("http://127.0.0.1:{port}/segment.ts")
}

#[tokio::test]
async fn fetches_and_streams_the_body_then_eos() {
    // A payload larger than a TCP segment so the body spans multiple reads.
    let payload: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
    let url = serve_once(payload.clone());

    let mut src = HttpSrc::new(
        url,
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        },
    );

    // Negotiation produces the declared byte-stream caps. Scoped so the
    // borrow held by `CapsConstraint` is released before `run`.
    {
        match src.caps_constraint().await.unwrap() {
            CapsConstraint::Produces(set) => assert_eq!(
                set,
                CapsSet::one(Caps::ByteStream {
                    encoding: ByteStreamEncoding::MpegTs
                })
            ),
            _ => panic!("HttpSrc should Produce its declared caps"),
        }
    }

    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    let count = src.run(&mut sink).await.unwrap();

    assert!(sink.eos, "EOS terminates the stream");
    assert!(sink.frames >= 1, "at least one chunk");
    assert_eq!(count, sink.frames as u64, "run returns the DataFrame count");
    assert_eq!(
        sink.body, payload,
        "reassembled body matches the served bytes"
    );
}

#[tokio::test]
async fn prebuffer_posts_buffering_levels_then_streams_intact() {
    // 200 KB payload, 64 KB prebuffer: the fill posts rising Buffering levels
    // ending at 100, and the delivered body is bit-identical.
    let payload: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
    let url = serve_once(payload.clone());

    let (bus, handle) = g2g_core::bus::Bus::new(64);
    let mut src = HttpSrc::new(
        url,
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        },
    )
    .with_prebuffer_bytes(64 * 1024)
    .with_bus(handle);
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .unwrap();

    let mut sink = CaptureSink::default();
    src.run(&mut sink).await.unwrap();
    assert!(sink.eos);
    assert_eq!(sink.body, payload, "prebuffering reorders nothing");

    let mut levels = Vec::new();
    while let Some(msg) = bus.try_recv() {
        if let g2g_core::BusMessage::Buffering { percent, .. } = msg {
            levels.push(percent);
        }
    }
    assert!(!levels.is_empty(), "the fill posted Buffering reports");
    assert!(
        levels.iter().any(|&p| p < 100),
        "a below-100 level was reported while filling: {levels:?}"
    );
    assert_eq!(*levels.last().unwrap(), 100, "buffering completes at 100");
}

/// Serve `payload` in two halves with a pause between them, so the source's
/// window underruns mid-stream; returns the URL.
fn serve_stalled(payload: Vec<u8>, stall: std::time::Duration) -> String {
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
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            payload.len()
        );
        stream.write_all(header.as_bytes()).unwrap();
        let half = payload.len() / 2;
        stream.write_all(&payload[..half]).unwrap();
        stream.flush().unwrap();
        thread::sleep(stall);
        stream.write_all(&payload[half..]).unwrap();
        stream.flush().unwrap();
    });
    format!("http://127.0.0.1:{port}/stream.ts")
}

#[tokio::test]
async fn mid_stream_stall_rebuffers_and_reports_it() {
    // A small window drains during the server's stall: the source re-enters
    // buffering (a below-100 report after a 100) and still delivers every byte.
    let payload: Vec<u8> = (0..200_000u32).map(|i| (i % 241) as u8).collect();
    let url = serve_stalled(payload.clone(), std::time::Duration::from_millis(300));

    let (bus, handle) = g2g_core::bus::Bus::new(64);
    let mut src = HttpSrc::new(
        url,
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        },
    )
    .with_prebuffer_bytes(8 * 1024)
    .with_bus(handle);
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .unwrap();

    let mut sink = CaptureSink::default();
    src.run(&mut sink).await.unwrap();
    assert!(sink.eos);
    assert_eq!(sink.body, payload, "the stall loses nothing");

    let mut levels = Vec::new();
    while let Some(msg) = bus.try_recv() {
        if let g2g_core::BusMessage::Buffering { percent, .. } = msg {
            levels.push(percent);
        }
    }
    let first_full = levels
        .iter()
        .position(|&p| p == 100)
        .expect("initial fill reached 100");
    assert!(
        levels[first_full..].iter().any(|&p| p < 100),
        "the stall re-entered buffering (below-100 after 100): {levels:?}"
    );
    assert_eq!(
        *levels.last().unwrap(),
        100,
        "re-buffering completes at 100 again"
    );
}

#[tokio::test]
async fn errors_loud_on_http_404() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut byte = [0u8; 1];
        let mut req = Vec::new();
        while stream.read(&mut byte).unwrap_or(0) == 1 {
            req.push(byte[0]);
            if req.ends_with(b"\r\n\r\n") {
                break;
            }
        }
        let _ = stream
            .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
    });

    let mut src = HttpSrc::new(
        format!("http://127.0.0.1:{port}/missing.ts"),
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        },
    );
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    assert!(
        src.run(&mut sink).await.is_err(),
        "a 4xx status fails the run"
    );
}
