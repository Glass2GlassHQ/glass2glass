//! M156 HLS source end to end: `HlsSrc` fetches a master playlist, selects a
//! variant, fetches its media playlist, then streams the TS segments in order
//! as `Caps::ByteStream{MpegTs}` `DataFrame`s ending in `Eos`. A local routing
//! HTTP server (no extra deps) serves the playlists and segments by path.

#![cfg(feature = "hls")]

use core::future::Future;
use core::pin::Pin;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use g2g_core::runtime::SourceLoop;
use g2g_core::{
    ByteStreamEncoding, Caps, G2gError, MemoryDomain, OutputSink, PipelinePacket, PushOutcome,
};
use g2g_plugins::hlssrc::HlsSrc;

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
                    if let MemoryDomain::System(s) = &f.domain {
                        self.body.extend_from_slice(s.as_slice());
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

const MASTER: &str = "#EXTM3U\n\
    #EXT-X-STREAM-INF:BANDWIDTH=800000,RESOLUTION=640x360\n\
    v/low.m3u8\n\
    #EXT-X-STREAM-INF:BANDWIDTH=2400000,RESOLUTION=1280x720\n\
    v/high.m3u8\n";

const MEDIA_HIGH: &str = "#EXTM3U\n\
    #EXT-X-TARGETDURATION:4\n\
    #EXT-X-MEDIA-SEQUENCE:0\n\
    #EXTINF:4.0,\n\
    seg0.ts\n\
    #EXTINF:4.0,\n\
    seg1.ts\n\
    #EXT-X-ENDLIST\n";

/// Route requests by path; serve playlists and two TS segments. Loops so each
/// reqwest connection (Connection: close) is handled in turn.
fn serve(seg0: Vec<u8>, seg1: Vec<u8>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        for conn in listener.incoming() {
            let mut stream = match conn {
                Ok(s) => s,
                Err(_) => break,
            };
            let mut req = Vec::new();
            let mut byte = [0u8; 1];
            while stream.read(&mut byte).unwrap_or(0) == 1 {
                req.push(byte[0]);
                if req.ends_with(b"\r\n\r\n") {
                    break;
                }
            }
            let line = String::from_utf8_lossy(&req);
            let path = line.split_whitespace().nth(1).unwrap_or("");
            let body: Vec<u8> = match path {
                "/master.m3u8" => MASTER.as_bytes().to_vec(),
                "/v/high.m3u8" => MEDIA_HIGH.as_bytes().to_vec(),
                "/v/seg0.ts" => seg0.clone(),
                "/v/seg1.ts" => seg1.clone(),
                _ => {
                    let _ = stream.write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                    continue;
                }
            };
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(&body);
        }
    });
    format!("http://127.0.0.1:{port}/master.m3u8")
}

#[tokio::test]
async fn streams_selected_variant_segments_in_order() {
    let seg0: Vec<u8> = (0..50_000u32).map(|i| (i % 251) as u8).collect();
    let seg1: Vec<u8> = (0..40_000u32).map(|i| (i % 239) as u8 ^ 0x5a).collect();
    let url = serve(seg0.clone(), seg1.clone());

    // No cap -> the 2.4 Mbps "high" variant is selected.
    let mut src = HlsSrc::new(url);
    src.configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs }).unwrap();
    let mut sink = CaptureSink::default();
    let count = src.run(&mut sink).await.unwrap();

    assert!(sink.eos, "EOS terminates the VOD playlist");
    assert_eq!(count, 2, "one DataFrame per segment");
    assert_eq!(sink.frames, 2);
    let mut expected = seg0.clone();
    expected.extend_from_slice(&seg1);
    assert_eq!(sink.body, expected, "segments delivered in playlist order, byte-exact");
}

/// The live media playlist returned on the Nth reload: a 2-segment sliding
/// window that advances each time and adds ENDLIST on the third fetch.
fn live_playlist(reload: usize) -> String {
    match reload {
        0 => "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXT-X-MEDIA-SEQUENCE:0\n\
              #EXTINF:1.0,\nseg0.ts\n#EXTINF:1.0,\nseg1.ts\n"
            .into(),
        1 => "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXT-X-MEDIA-SEQUENCE:1\n\
              #EXTINF:1.0,\nseg1.ts\n#EXTINF:1.0,\nseg2.ts\n"
            .into(),
        _ => "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXT-X-MEDIA-SEQUENCE:2\n\
              #EXTINF:1.0,\nseg2.ts\n#EXTINF:1.0,\nseg3.ts\n#EXT-X-ENDLIST\n"
            .into(),
    }
}

fn serve_live(segs: Vec<Vec<u8>>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let reloads = Arc::new(AtomicUsize::new(0));
    thread::spawn(move || {
        for conn in listener.incoming() {
            let mut stream = match conn {
                Ok(s) => s,
                Err(_) => break,
            };
            let mut req = Vec::new();
            let mut byte = [0u8; 1];
            while stream.read(&mut byte).unwrap_or(0) == 1 {
                req.push(byte[0]);
                if req.ends_with(b"\r\n\r\n") {
                    break;
                }
            }
            let line = String::from_utf8_lossy(&req);
            let path = line.split_whitespace().nth(1).unwrap_or("");
            let body: Vec<u8> = if path == "/live.m3u8" {
                let n = reloads.fetch_add(1, Ordering::SeqCst);
                live_playlist(n).into_bytes()
            } else if let Some(idx) = path
                .strip_prefix("/seg")
                .and_then(|s| s.strip_suffix(".ts"))
                .and_then(|s| s.parse::<usize>().ok())
            {
                segs.get(idx).cloned().unwrap_or_default()
            } else {
                let _ = stream.write_all(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                );
                continue;
            };
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(&body);
        }
    });
    format!("http://127.0.0.1:{port}/live.m3u8")
}

#[tokio::test]
async fn live_reloads_playlist_and_plays_each_new_segment_once() {
    let segs: Vec<Vec<u8>> =
        (0..4u8).map(|s| (0..1000u32).map(|i| (i as u8) ^ (s * 37)).collect()).collect();
    let url = serve_live(segs.clone());

    let mut src = HlsSrc::new(url).with_reload_interval_ms(20);
    src.configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs }).unwrap();
    let mut sink = CaptureSink::default();
    let count = src.run(&mut sink).await.unwrap();

    assert!(sink.eos, "ENDLIST on the final reload terminates the live stream");
    assert_eq!(count, 4, "each of the 4 segments played exactly once across reloads");
    let expected: Vec<u8> = segs.concat();
    assert_eq!(sink.body, expected, "seg0..seg3 delivered once, in order, no duplicates");
}
