//! M819: duration-keyed prebuffer on the adaptive segment loops. With
//! `prebuffer-ms` set, `HlsSrc` / `DashSrc` fetch ahead until the queued
//! segment durations reach the target before emitting anything, posting
//! `Buffering` levels on the attached bus during the fill, and the delivered
//! byte stream is unchanged by the window.

#![cfg(any(feature = "hls", feature = "dash"))]

use core::future::Future;
use core::pin::Pin;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use g2g_core::runtime::SourceLoop;
use g2g_core::{ByteStreamEncoding, Caps, G2gError, OutputSink, PipelinePacket, PushOutcome};

/// Capture sink that also snapshots how many media-segment fetches the server
/// had answered when the first DataFrame arrived (proof the window filled
/// before emission started).
struct CaptureSink {
    body: Vec<u8>,
    frames: usize,
    eos: bool,
    fetch_counter: Arc<AtomicUsize>,
    fetches_at_first_frame: Option<usize>,
}

impl CaptureSink {
    fn new(fetch_counter: Arc<AtomicUsize>) -> Self {
        Self {
            body: Vec::new(),
            frames: 0,
            eos: false,
            fetch_counter,
            fetches_at_first_frame: None,
        }
    }
}

impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if self.fetches_at_first_frame.is_none() {
                        self.fetches_at_first_frame =
                            Some(self.fetch_counter.load(Ordering::SeqCst));
                    }
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

/// Minimal routing HTTP server: serves `routes` by path, counting hits whose
/// path passes `counted`.
fn serve(routes: Vec<(String, Vec<u8>)>, counted: fn(&str) -> bool) -> (u16, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let counter = Arc::new(AtomicUsize::new(0));
    let c = counter.clone();
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
            let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
            let Some((_, body)) = routes.iter().find(|(p, _)| *p == path) else {
                let _ = stream.write_all(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                );
                continue;
            };
            if counted(&path) {
                c.fetch_add(1, Ordering::SeqCst);
            }
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(body);
        }
    });
    (port, counter)
}

fn buffering_levels(bus: &g2g_core::bus::Bus) -> Vec<u8> {
    let mut levels = Vec::new();
    while let Some(msg) = bus.try_recv() {
        if let g2g_core::BusMessage::Buffering { percent } = msg {
            levels.push(percent);
        }
    }
    levels
}

#[cfg(feature = "hls")]
#[tokio::test]
async fn hls_prebuffer_fills_two_segments_before_first_emit() {
    use g2g_plugins::hlssrc::HlsSrc;

    let media = "#EXTM3U\n\
        #EXT-X-TARGETDURATION:4\n\
        #EXT-X-MEDIA-SEQUENCE:0\n\
        #EXTINF:4.0,\nseg0.ts\n\
        #EXTINF:4.0,\nseg1.ts\n\
        #EXTINF:4.0,\nseg2.ts\n\
        #EXTINF:4.0,\nseg3.ts\n\
        #EXT-X-ENDLIST\n";
    let segs: Vec<Vec<u8>> = (0..4u8)
        .map(|i| (0..30_000u32).map(|j| (j % 251) as u8 ^ i).collect())
        .collect();
    let mut routes = vec![("/media.m3u8".to_string(), media.as_bytes().to_vec())];
    for (i, s) in segs.iter().enumerate() {
        routes.push((format!("/seg{i}.ts"), s.clone()));
    }
    let (port, counter) = serve(routes, |p| p.ends_with(".ts"));

    let (bus, handle) = g2g_core::bus::Bus::new(64);
    // 8 s target over 4 s segments: two fetches before the first emit.
    let mut src = HlsSrc::new(format!("http://127.0.0.1:{port}/media.m3u8"))
        .with_prebuffer_ms(8_000)
        .with_bus(handle);
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .unwrap();
    let mut sink = CaptureSink::new(counter);
    let count = src.run(&mut sink).await.unwrap();

    assert!(sink.eos);
    assert_eq!(count, 4, "one DataFrame per segment");
    let expected: Vec<u8> = segs.concat();
    assert_eq!(sink.body, expected, "the window reorders nothing");
    assert!(
        sink.fetches_at_first_frame.unwrap() >= 2,
        "the 8 s fill fetched two 4 s segments before emitting: {:?}",
        sink.fetches_at_first_frame
    );

    let levels = buffering_levels(&bus);
    assert!(
        levels.iter().any(|&p| p < 100),
        "a below-100 level was reported while filling: {levels:?}"
    );
    assert_eq!(*levels.last().unwrap(), 100, "buffering completes at 100");
}

#[cfg(feature = "dash")]
#[tokio::test]
async fn dash_prebuffer_fills_before_first_emit() {
    use g2g_plugins::dashsrc::DashSrc;

    // 4 segments at 1 s each (@duration profile), startNumber=0.
    let mpd = "<?xml version=\"1.0\"?>\n\
        <MPD mediaPresentationDuration=\"PT4S\" type=\"static\">\n\
          <Period>\n\
            <AdaptationSet mimeType=\"video/mp4\" codecs=\"avc1.4d401f\">\n\
              <SegmentTemplate initialization=\"init.mp4\" media=\"seg$Number$.m4s\" \
                 startNumber=\"0\" duration=\"1000\" timescale=\"1000\"/>\n\
              <Representation id=\"v0\" bandwidth=\"1000000\" width=\"64\" height=\"48\"/>\n\
            </AdaptationSet>\n\
          </Period>\n\
        </MPD>";
    let init: Vec<u8> = (0..2_000u32).map(|j| (j % 149) as u8).collect();
    let segs: Vec<Vec<u8>> = (0..4u8)
        .map(|i| (0..20_000u32).map(|j| (j % 233) as u8 ^ i).collect())
        .collect();
    let mut routes = vec![
        ("/manifest.mpd".to_string(), mpd.as_bytes().to_vec()),
        ("/init.mp4".to_string(), init.clone()),
    ];
    for (i, s) in segs.iter().enumerate() {
        routes.push((format!("/seg{i}.m4s"), s.clone()));
    }
    let (port, counter) = serve(routes, |p| p.ends_with(".m4s"));

    let (bus, handle) = g2g_core::bus::Bus::new(64);
    // 2 s target over 1 s segments: two media fetches before the first emit.
    let mut src = DashSrc::new(format!("http://127.0.0.1:{port}/manifest.mpd"))
        .with_prebuffer_ms(2_000)
        .with_bus(handle);
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .unwrap();
    let mut sink = CaptureSink::new(counter);
    let count = src.run(&mut sink).await.unwrap();

    assert!(sink.eos);
    assert_eq!(count, 5, "init plus one DataFrame per segment");
    let mut expected = init;
    expected.extend(segs.concat());
    assert_eq!(sink.body, expected, "init first, segments in order");
    assert!(
        sink.fetches_at_first_frame.unwrap() >= 2,
        "the 2 s fill fetched two 1 s segments before emitting: {:?}",
        sink.fetches_at_first_frame
    );

    let levels = buffering_levels(&bus);
    assert!(
        levels.iter().any(|&p| p < 100),
        "a below-100 level was reported while filling: {levels:?}"
    );
    assert_eq!(*levels.last().unwrap(), 100, "buffering completes at 100");
}
