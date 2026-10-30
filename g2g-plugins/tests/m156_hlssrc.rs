//! M156 HLS source end to end: `HlsSrc` fetches a master playlist, selects a
//! variant, fetches its media playlist, then streams the TS segments in order
//! as `Caps::ByteStream{MpegTs}` `DataFrame`s ending in `Eos`. A local routing
//! HTTP server (no extra deps) serves the playlists and segments by path.

#![cfg(feature = "hls")]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{Registry, SeekController, SourceLoop};
use g2g_core::time::ticks_to_ns;
use g2g_core::{
    ByteStreamEncoding, Caps, CapsConstraint, Dim, G2gError, MemoryDomain, MultiInputElement,
    OutputSink, PipelinePacket, PushOutcome, Rate, Seek, VideoCodec,
};
use g2g_plugins::fmp4demux::Fmp4Demux;
use g2g_plugins::hlssrc::HlsSrc;
use g2g_plugins::mp4muxn::Mp4MuxN;
use g2g_plugins::mpegts::{TsMuxer, STREAM_TYPE_H264};
use g2g_plugins::uridecodebin::hls_uri_fanout;

#[derive(Default)]
struct CaptureSink {
    body: Vec<u8>,
    frames: usize,
    eos: bool,
    flushes: usize,
    /// Stream-time start of each post-flush `Segment` emitted (seek observability).
    segment_starts: Vec<u64>,
    segment_times: Vec<u64>,
}

impl OutputSink for CaptureSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.body.extend_from_slice(s);
                        self.frames += 1;
                    }
                }
                PipelinePacket::Eos => self.eos = true,
                PipelinePacket::Flush => self.flushes += 1,
                PipelinePacket::Segment(seg) => {
                    self.segment_starts.push(seg.start);
                    self.segment_times.push(seg.time);
                }
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
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    let count = src.run(&mut sink).await.unwrap();

    assert!(sink.eos, "EOS terminates the VOD playlist");
    assert_eq!(count, 2, "one DataFrame per segment");
    assert_eq!(sink.frames, 2);
    let mut expected = seg0.clone();
    expected.extend_from_slice(&seg1);
    assert_eq!(
        sink.body, expected,
        "segments delivered in playlist order, byte-exact"
    );
}

/// Like `serve` but counts how many times the media playlist is fetched, to
/// prove the negotiation probe and `run()` share one fetch.
fn serve_counting(seg0: Vec<u8>, seg1: Vec<u8>) -> (String, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let media_fetches = Arc::new(AtomicUsize::new(0));
    let counter = media_fetches.clone();
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
                "/v/high.m3u8" => {
                    counter.fetch_add(1, Ordering::SeqCst);
                    MEDIA_HIGH.as_bytes().to_vec()
                }
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
    (
        format!("http://127.0.0.1:{port}/master.m3u8"),
        media_fetches,
    )
}

#[tokio::test]
async fn probe_playlist_is_reused_by_run_not_refetched() {
    let seg0: Vec<u8> = (0..1000u32).map(|i| i as u8).collect();
    let seg1: Vec<u8> = (0..1000u32).map(|i| (i as u8) ^ 0x5a).collect();
    let (url, media_fetches) = serve_counting(seg0, seg1);

    let mut src = HlsSrc::new(url);
    // Negotiation probe resolves the media playlist once.
    let _ = src.caps_constraint().await.unwrap();
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    src.run(&mut sink).await.unwrap();

    assert!(sink.eos);
    assert_eq!(
        media_fetches.load(Ordering::SeqCst),
        1,
        "media playlist fetched once at probe and reused by run, not refetched"
    );
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
    let segs: Vec<Vec<u8>> = (0..4u8)
        .map(|s| (0..1000u32).map(|i| (i as u8) ^ (s * 37)).collect())
        .collect();
    let url = serve_live(segs.clone());

    let mut src = HlsSrc::new(url).with_reload_interval_ms(20);
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    let count = src.run(&mut sink).await.unwrap();

    assert!(
        sink.eos,
        "ENDLIST on the final reload terminates the live stream"
    );
    assert_eq!(
        count, 4,
        "each of the 4 segments played exactly once across reloads"
    );
    let expected: Vec<u8> = segs.concat();
    assert_eq!(
        sink.body, expected,
        "seg0..seg3 delivered once, in order, no duplicates"
    );
}

// mediamtx's shape: the master hands out a session in the media playlist URL, and a
// restarted server answers the old session with 401 and starts its sequence again
fn serve_restarting_origin(before: Vec<u8>, after: Vec<Vec<u8>>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        let mut session = 1;
        let mut first_session_loads = 0;
        for conn in listener.incoming() {
            let Ok(mut stream) = conn else { break };
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
            let body: Vec<u8> = if path == "/master.m3u8" {
                format!(
                    "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=800000\nmedia.m3u8?session={session}\n"
                )
                .into_bytes()
            } else if path == "/media.m3u8?session=1" && session == 1 {
                first_session_loads += 1;
                if first_session_loads > 1 {
                    session = 2;
                    let _ = stream.write_all(
                        b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                    continue;
                }
                "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXT-X-MEDIA-SEQUENCE:0\n#EXTINF:1.0,\nbefore.ts\n"
                    .as_bytes()
                    .to_vec()
            } else if path == "/media.m3u8?session=2" && session == 2 {
                let mut playlist =
                    String::from("#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXT-X-MEDIA-SEQUENCE:0\n");
                for index in 0..after.len() {
                    playlist.push_str(&format!("#EXTINF:1.0,\nafter{index}.ts\n"));
                }
                playlist.push_str("#EXT-X-ENDLIST\n");
                playlist.into_bytes()
            } else if path == "/before.ts" {
                before.clone()
            } else if let Some(index) = path
                .strip_prefix("/after")
                .and_then(|s| s.strip_suffix(".ts"))
                .and_then(|s| s.parse::<usize>().ok())
            {
                after[index].clone()
            } else {
                let _ = stream.write_all(
                    b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
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
    format!("http://127.0.0.1:{port}/master.m3u8")
}

#[tokio::test]
async fn a_lost_session_resolves_the_master_again_and_flushes() {
    let segment = |seed: u8| -> Vec<u8> { (0..1000u32).map(|i| (i as u8) ^ seed).collect() };
    let before = segment(11);
    let after = vec![segment(23), segment(47)];
    let url = serve_restarting_origin(before.clone(), after.clone());

    let mut src = HlsSrc::new(url).with_reload_interval_ms(20);
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    src.run(&mut sink).await.unwrap();

    assert!(sink.eos);
    assert_eq!(sink.flushes, 1);
    assert_eq!(sink.body, [before, after.concat()].concat());
}

/// A live server whose first media-playlist load offers a *wide* window (6
/// one-second segments, no ENDLIST), then on the next reload appends ENDLIST with
/// the same segments. Used to prove live-edge start.
fn serve_wide_live(segs: Vec<Vec<u8>>) -> String {
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
                // The same six-segment window every load; ENDLIST from the 2nd
                // reload on, so the run terminates after the live-edge segments.
                let mut p =
                    String::from("#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXT-X-MEDIA-SEQUENCE:0\n");
                for i in 0..6 {
                    p.push_str(&format!("#EXTINF:1.0,\nseg{i}.ts\n"));
                }
                if n >= 1 {
                    p.push_str("#EXT-X-ENDLIST\n");
                }
                p.into_bytes()
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
async fn live_starts_near_the_edge_skipping_the_stale_window_front() {
    // Six 1s segments, target duration 1s: the live edge is 3 target durations
    // (3s) from the end, so the first segment played is seg3 (seg3,4,5 = 3s ahead),
    // not seg0. The stale front (seg0..seg2) is never fetched.
    let segs: Vec<Vec<u8>> = (0..6u8)
        .map(|s| {
            (0..500u32)
                .map(|i| (i as u8) ^ (s.wrapping_mul(53)))
                .collect()
        })
        .collect();
    let url = serve_wide_live(segs.clone());

    let mut src = HlsSrc::new(url).with_reload_interval_ms(20);
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    let count = src.run(&mut sink).await.unwrap();

    assert!(sink.eos, "ENDLIST on the reload terminates the stream");
    assert_eq!(
        count, 3,
        "only the three live-edge segments (seg3..seg5) play"
    );
    let expected: Vec<u8> = segs[3..].concat();
    assert_eq!(
        sink.body, expected,
        "playback starts at the live edge, not the window front"
    );
}

#[tokio::test]
async fn full_replay_starts_a_live_playlist_from_the_window_front() {
    // The same six-segment live window, but `with_full_replay()` opts out of the
    // live-edge skip: every segment from the front plays (DVR replay).
    let segs: Vec<Vec<u8>> = (0..6u8)
        .map(|s| {
            (0..500u32)
                .map(|i| (i as u8) ^ (s.wrapping_mul(53)))
                .collect()
        })
        .collect();
    let url = serve_wide_live(segs.clone());

    let mut src = HlsSrc::new(url)
        .with_reload_interval_ms(20)
        .with_full_replay();
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    let count = src.run(&mut sink).await.unwrap();

    assert!(sink.eos, "ENDLIST on the reload terminates the stream");
    assert_eq!(
        count, 6,
        "full replay plays the whole window from the front (seg0..seg5)"
    );
    let expected: Vec<u8> = segs.concat();
    assert_eq!(
        sink.body, expected,
        "every segment delivered, in order, from the window front"
    );
}

// --- time seek: jump to the segment containing the target (M367) ----------

/// A 3-segment VOD playlist, 4s each, so a time seek maps unambiguously to a
/// segment (seg0 = [0,4)s, seg1 = [4,8)s, seg2 = [8,12)s).
const MEDIA_SEEK: &str = "#EXTM3U\n\
    #EXT-X-TARGETDURATION:4\n\
    #EXT-X-MEDIA-SEQUENCE:0\n\
    #EXTINF:4.0,\nseg0.ts\n\
    #EXTINF:4.0,\nseg1.ts\n\
    #EXTINF:4.0,\nseg2.ts\n\
    #EXT-X-ENDLIST\n";

fn serve_seek(segs: Vec<Vec<u8>>) -> String {
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
            let body: Vec<u8> = if path == "/media.m3u8" {
                MEDIA_SEEK.as_bytes().to_vec()
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
    format!("http://127.0.0.1:{port}/media.m3u8")
}

#[tokio::test]
async fn time_seek_jumps_to_the_segment_containing_the_target() {
    let segs: Vec<Vec<u8>> = (0..3u8)
        .map(|s| (0..1000u32).map(|i| (i as u8) ^ (s * 53)).collect())
        .collect();
    let url = serve_seek(segs.clone());

    let seek = SeekController::new();
    // Pre-arm a 5s seek before run(): deterministic (no fetch/EOF race). 5s lands
    // in seg1 ([4,8)s), so playback must start there, skipping seg0.
    seek.seek(Seek::flush_to(5 * 1_000_000_000));

    let mut src = HlsSrc::new(url).with_seek(seek);
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    src.run(&mut sink).await.unwrap();

    assert!(sink.eos, "the VOD playlist still ends with Eos");
    assert_eq!(sink.flushes, 1, "the flushing seek emits exactly one Flush");
    assert_eq!(
        sink.segment_starts,
        vec![5 * 1_000_000_000],
        "with no timestamp in the segment, the post-flush Segment starts at the target on playlist time"
    );
    // Body is seg1 then seg2: the seek resumes at the target segment and plays on.
    let mut expected = segs[1].clone();
    expected.extend_from_slice(&segs[2]);
    assert_eq!(sink.body, expected, "playback starts at seg1, seg0 skipped");
}

// --- the seek Segment on the segment's own timestamps (M2008) -------------

const TS_CLOCK_HZ: u32 = 90_000;
// MEDIA_SEEK's `#EXTINF:4.0`, one frame a second
const SEEK_SEGMENT_NS: u64 = 4_000_000_000;
const SEEK_SEGMENT_FRAMES: u64 = 4;
const SEEK_FRAME_TICKS: u64 = TS_CLOCK_HZ as u64;
// an encoder that does not start its clock at zero
const SEEK_MEDIA_START_TICKS: u64 = 10 * SEEK_FRAME_TICKS;
const SEEK_TARGET_NS: u64 = 5_000_000_000;

// MEDIA_SEEK's segments as transport stream, and the pts the first frame of each carries
fn seek_ts_segments() -> (Vec<Vec<u8>>, Vec<u64>) {
    let mut mux = TsMuxer::new(STREAM_TYPE_H264);
    // each segment opens with its PAT and PMT, as a segmenter writes them
    mux.set_table_interval_90khz(SEEK_FRAME_TICKS);
    let idr = [0u8, 0, 0, 1, 0x65, 0x88, 0x84, 0x00];
    let mut segments = Vec::new();
    let mut first_pts_ns = Vec::new();
    for segment in 0..3 {
        let mut bytes = Vec::new();
        for frame in 0..SEEK_SEGMENT_FRAMES {
            let pts =
                SEEK_MEDIA_START_TICKS + (segment * SEEK_SEGMENT_FRAMES + frame) * SEEK_FRAME_TICKS;
            if frame == 0 {
                first_pts_ns.push(ticks_to_ns(pts + mux.decoder_delay_90khz(), TS_CLOCK_HZ));
            }
            bytes.extend(mux.push_au(&idr, Some(pts), None, true));
        }
        segments.push(bytes);
    }
    (segments, first_pts_ns)
}

#[tokio::test]
async fn a_seek_starts_at_the_target_on_the_segment_pts() {
    let (segments, first_pts_ns) = seek_ts_segments();
    let controller = SeekController::new();
    controller.seek(Seek::flush_to(SEEK_TARGET_NS));
    let mut src = HlsSrc::new(serve_seek(segments)).with_seek(controller);
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    src.run(&mut sink).await.unwrap();

    assert_eq!(
        sink.segment_starts,
        vec![first_pts_ns[1] + SEEK_TARGET_NS - SEEK_SEGMENT_NS]
    );
    assert_eq!(sink.segment_times, vec![SEEK_TARGET_NS]);
}

// --- throughput-driven ABR (M371) -----------------------------------------

/// Master with two far-apart variants so the bandwidth estimate selects one
/// unambiguously: high = 50 Mbit/s, low = 500 kbit/s.
const ABR_MASTER: &str = "#EXTM3U\n\
    #EXT-X-STREAM-INF:BANDWIDTH=50000000,RESOLUTION=1920x1080\n\
    v/high.m3u8\n\
    #EXT-X-STREAM-INF:BANDWIDTH=500000,RESOLUTION=320x180\n\
    v/low.m3u8\n";

/// Two time-aligned media playlists (same 1s segments), distinct URIs. Segment
/// `{kind}{n}.ts` has media sequence `n`, and the window starts at `first`.
fn abr_media(kind: &str, first: usize, segments: usize) -> String {
    let mut playlist = format!("#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXT-X-MEDIA-SEQUENCE:{first}\n");
    for index in first..segments {
        playlist += &format!("#EXTINF:1.0,\n{kind}{index}.ts\n");
    }
    playlist + "#EXT-X-ENDLIST\n"
}

/// Serve the ABR master + variants, the low playlist's window starting at
/// `low_window_start`, delaying each `.ts` response by
/// `throttle(path, body length)` so a segment's download time is measurable.
fn serve_abr(
    high: Vec<Vec<u8>>,
    low: Vec<Vec<u8>>,
    low_window_start: usize,
    throttle: impl Fn(&str, usize) -> std::time::Duration + Send + 'static,
) -> String {
    let segments = high.len();
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
            let seg = |kind: &[Vec<u8>], p: &str, pre: &str| {
                p.strip_prefix(pre)
                    .and_then(|s| s.strip_suffix(".ts"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .and_then(|i| kind.get(i).cloned())
            };
            let body: Vec<u8> = match path {
                "/master.m3u8" => ABR_MASTER.as_bytes().to_vec(),
                "/v/high.m3u8" => abr_media("high", 0, segments).into_bytes(),
                "/v/low.m3u8" => abr_media("low", low_window_start, segments).into_bytes(),
                p if p.starts_with("/v/high") => seg(&high, p, "/v/high").unwrap_or_default(),
                p if p.starts_with("/v/low") => seg(&low, p, "/v/low").unwrap_or_default(),
                _ => {
                    let _ = stream.write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                    continue;
                }
            };
            // Playlists are served immediately.
            if path.ends_with(".ts") {
                thread::sleep(throttle(path, body.len()));
            }
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
async fn abr_downswitches_when_measured_bandwidth_is_low() {
    // High segments are large (slow over the throttled server), low segments tiny.
    let high: Vec<Vec<u8>> = (0..3u8)
        .map(|s| (0..60_000u32).map(|i| (i as u8) ^ (s + 1)).collect())
        .collect();
    let low: Vec<Vec<u8>> = (0..3u8)
        .map(|s| (0..2_000u32).map(|i| (i as u8) ^ (0xF0 + s)).collect())
        .collect();
    // ~1ms per KiB, so a large (high-variant) segment yields a low measured
    // throughput, forcing a downswitch; small (low-variant) segments stay cheap.
    let url = serve_abr(high.clone(), low.clone(), 0, |_, len| {
        std::time::Duration::from_millis((len / 1024) as u64)
    });

    let mut src = HlsSrc::new(url).with_abr();
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    let count = src.run(&mut sink).await.unwrap();

    assert!(sink.eos);
    assert_eq!(
        count, 3,
        "three segments played (variants are time-aligned)"
    );
    // No estimate yet -> the first segment is the high variant. The measured
    // throughput of that large/slow segment then downswitches to low for the rest.
    let mut expected = high[0].clone();
    expected.extend_from_slice(&low[1]);
    expected.extend_from_slice(&low[2]);
    assert_eq!(
        sink.body, expected,
        "ABR starts on high, then switches to low after measuring the slow download"
    );
}

const ABR_SEGMENTS: usize = 6;

// large high segments and small low ones, distinct per segment
fn abr_variants() -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let high = (0..ABR_SEGMENTS as u8)
        .map(|s| (0..600_000u32).map(|i| (i as u8) ^ (s + 1)).collect())
        .collect();
    let low = (0..ABR_SEGMENTS as u8)
        .map(|s| (0..2_000u32).map(|i| (i as u8) ^ (0xF0 + s)).collect())
        .collect();
    (high, low)
}

// a 600 kB high segment from `slow_from` on takes 500 ms, 9.6 Mbit/s against the high variant's 50 Mbit/s,
// and every other segment arrives at loopback speed
fn high_slow_from(slow_from: usize) -> impl Fn(&str, usize) -> std::time::Duration {
    move |path, _| {
        let high_index = path
            .strip_prefix("/v/high")
            .and_then(|s| s.strip_suffix(".ts"))
            .and_then(|s| s.parse::<usize>().ok());
        match high_index {
            Some(index) if index >= slow_from => std::time::Duration::from_millis(500),
            _ => std::time::Duration::ZERO,
        }
    }
}

async fn abr_body(url: String) -> Vec<u8> {
    let mut src = HlsSrc::new(url).with_abr();
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    src.run(&mut sink).await.unwrap();
    sink.body
}

#[tokio::test]
async fn abr_downswitches_on_the_first_slow_segment_after_fast_ones() {
    const SLOW_FROM: usize = 3;
    let (high, low) = abr_variants();
    let url = serve_abr(high.clone(), low.clone(), 0, high_slow_from(SLOW_FROM));
    let expected: Vec<u8> = high[..=SLOW_FROM]
        .iter()
        .chain(&low[SLOW_FROM + 1..])
        .flatten()
        .copied()
        .collect();
    assert!(
        abr_body(url).await == expected,
        "the fast downloads keep the high variant, the first slow one switches to low"
    );
}

#[tokio::test]
async fn abr_switch_resumes_at_the_next_sequence_when_the_new_window_has_moved() {
    const SLOW_FROM: usize = 1;
    // a live playlist fetched later than the old one starts further on
    const LOW_WINDOW_START: usize = 1;
    let (high, low) = abr_variants();
    let url = serve_abr(
        high.clone(),
        low.clone(),
        LOW_WINDOW_START,
        high_slow_from(SLOW_FROM),
    );
    let expected: Vec<u8> = high[..=SLOW_FROM]
        .iter()
        .chain(&low[SLOW_FROM + 1..])
        .flatten()
        .copied()
        .collect();
    assert!(
        abr_body(url).await == expected,
        "the switch to low neither repeats nor skips a media sequence"
    );
}

// --- fMP4 / CMAF over HLS (EXT-X-MAP) -------------------------------------

fn au_frame(bytes: Vec<u8>, pts_ns: u64, seq: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            duration_ns: 33_333_333,
            ..FrameTiming::default()
        },
        sequence: seq,
        meta: Default::default(),
    }
}

fn access_units() -> Vec<Vec<u8>> {
    let sps = [0x67u8, 0x42, 0xC0, 0x1E, 0x11, 0x22];
    let pps = [0x68u8, 0xCE, 0x3C, 0x80];
    let idr: Vec<u8> = [
        &[0, 0, 0, 1][..],
        &sps,
        &[0, 0, 0, 1],
        &pps,
        &[0, 0, 0, 1],
        &[0x65, 0xAA, 0xBB],
    ]
    .concat();
    let p = |f: u8| [&[0, 0, 0, 1][..], &[0x41, f, f]].concat();
    vec![idr, p(1), p(2)]
}

/// Mux the access units to an fMP4 buffer via the `Mp4MuxN` element, returning
/// the byte stream it forwards downstream.
async fn make_fmp4(aus: &[Vec<u8>]) -> Vec<u8> {
    let mut mux = Mp4MuxN::new(1);
    mux.configure_pipeline(
        0,
        &Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(64),
            height: Dim::Fixed(48),
            framerate: Rate::Fixed(30 << 16),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        },
    )
    .unwrap();
    let mut out = CaptureSink::default();
    for (i, au) in aus.iter().enumerate() {
        mux.process(
            0,
            PipelinePacket::DataFrame(au_frame(au.clone(), i as u64 * 33_333_333, i as u64)),
            &mut out,
        )
        .await
        .unwrap();
    }
    mux.process(0, PipelinePacket::Eos, &mut out).await.unwrap();
    out.body
}

/// Split an fMP4 buffer into the init segment (ftyp+moov, everything before the
/// first `moof`) and one media segment per `moof`+`mdat` fragment, as a CMAF HLS
/// origin would serve them.
fn split_fmp4(data: &[u8]) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut spans = Vec::new(); // (kind, start, end)
    let mut i = 0;
    while i + 8 <= data.len() {
        let size = u32::from_be_bytes(data[i..i + 4].try_into().unwrap()) as usize;
        let kind: [u8; 4] = data[i + 4..i + 8].try_into().unwrap();
        spans.push((kind, i, i + size));
        i += size;
    }
    let first_moof = spans.iter().find(|(k, _, _)| k == b"moof").unwrap().1;
    let init = data[..first_moof].to_vec();
    let mut segments = Vec::new();
    let mut j = 0;
    while j < spans.len() {
        if &spans[j].0 == b"moof" {
            let (_, start, _) = spans[j];
            let (_, _, end) = spans[j + 1]; // the following mdat
            segments.push(data[start..end].to_vec());
            j += 2;
        } else {
            j += 1;
        }
    }
    (init, segments)
}

fn serve_fmp4(init: Vec<u8>, segs: Vec<Vec<u8>>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let mut playlist =
        String::from("#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXT-X-MAP:URI=\"init.mp4\"\n");
    for n in 0..segs.len() {
        playlist.push_str(&format!("#EXTINF:1.0,\nseg{n}.m4s\n"));
    }
    playlist.push_str("#EXT-X-ENDLIST\n");
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
            let body: Vec<u8> = if path == "/fmp4.m3u8" {
                playlist.clone().into_bytes()
            } else if path == "/init.mp4" {
                init.clone()
            } else if let Some(idx) = path
                .strip_prefix("/seg")
                .and_then(|s| s.strip_suffix(".m4s"))
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
    format!("http://127.0.0.1:{port}/fmp4.m3u8")
}

#[tokio::test]
async fn fmp4_hls_emits_init_then_fragments_and_demuxes() {
    let aus = access_units();
    let fmp4 = make_fmp4(&aus).await;
    let (init, segs) = split_fmp4(&fmp4);
    let url = serve_fmp4(init.clone(), segs.clone());

    let mut src = HlsSrc::new(url);

    // The negotiation probe sees EXT-X-MAP and declares the fMP4 container.
    {
        match src.caps_constraint().await.unwrap() {
            CapsConstraint::Produces(set) => assert_eq!(
                set,
                g2g_core::CapsSet::one(Caps::ByteStream {
                    encoding: ByteStreamEncoding::IsoBmff
                })
            ),
            _ => panic!("fMP4 HLS should produce IsoBmff caps"),
        }
    }

    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    src.run(&mut sink).await.unwrap();

    // The byte stream is the init segment followed by every media fragment.
    let mut expected = init.clone();
    for s in &segs {
        expected.extend_from_slice(s);
    }
    assert_eq!(
        sink.body, expected,
        "init segment emitted first, then fragments in order"
    );

    // End to end: the delivered fMP4 byte stream demuxes back to the access units.
    let mut dmx = Fmp4Demux::new();
    dmx.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .unwrap();
    let mut dsink = DemuxSink::default();
    dmx.process(
        PipelinePacket::DataFrame(au_frame(sink.body.clone(), 0, 0)),
        &mut dsink,
    )
    .await
    .unwrap();
    assert_eq!(
        dsink.aus, aus,
        "HlsSrc -> Fmp4Demux recovers the original access units"
    );
}

// --- single-file CMAF via #EXT-X-BYTERANGE (M368) -------------------------

/// Parse a `Range: bytes=a-b` request header, returning the inclusive `(a, b)`.
fn parse_range_header(req: &str) -> Option<(usize, usize)> {
    let line = req
        .lines()
        .find(|l| l.to_ascii_lowercase().starts_with("range:"))?;
    let spec = line.split_once('=')?.1.trim();
    let (a, b) = spec.split_once('-')?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
}

// the inclusive `Range` of each `all.m4s` request, `None` for a whole-file request
type RangeLog = Arc<Mutex<Vec<Option<(usize, usize)>>>>;

/// Serve one resource `all.m4s` (init + fragments concatenated) honouring HTTP
/// `Range` requests with `206 Partial Content`, plus the byte-range playlist
/// (`/cmaf.m3u8`) and a master naming it (`/master.m3u8`). Returns the server's
/// base URL and the log of `all.m4s` ranges.
fn serve_byterange(resource: Vec<u8>, playlist: String) -> (String, RangeLog) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let ranges = RangeLog::default();
    let logged = ranges.clone();
    let master = "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=100000\ncmaf.m3u8\n";
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
            let text = String::from_utf8_lossy(&req);
            let path = text.split_whitespace().nth(1).unwrap_or("");
            let served_playlist = match path {
                "/cmaf.m3u8" => Some(playlist.as_str()),
                "/master.m3u8" => Some(master),
                _ => None,
            };
            if let Some(served_playlist) = served_playlist {
                let body = served_playlist.as_bytes();
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(header.as_bytes());
                let _ = stream.write_all(body);
            } else if path == "/all.m4s" {
                let range = parse_range_header(&text);
                logged.lock().unwrap().push(range);
                // Honour a Range with 206 + the exact slice; otherwise full body.
                let (body, status, extra) = match range {
                    Some((a, b)) => {
                        let end = (b + 1).min(resource.len());
                        let start = a.min(end);
                        let total = resource.len();
                        (
                            resource[start..end].to_vec(),
                            "206 Partial Content",
                            format!("Content-Range: bytes {start}-{}/{total}\r\n", end - 1),
                        )
                    }
                    None => (resource.clone(), "200 OK", String::new()),
                };
                let header = format!(
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{extra}Connection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(header.as_bytes());
                let _ = stream.write_all(&body);
            } else {
                let _ = stream.write_all(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                );
            }
        }
    });
    (format!("http://127.0.0.1:{port}"), ranges)
}

// the playlist addressing init then `fragments` by byte range of one `all.m4s`
fn byterange_playlist(init_len: usize, fragments: &[Vec<u8>]) -> String {
    let mut playlist = format!(
        "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXT-X-MAP:URI=\"all.m4s\",BYTERANGE=\"{init_len}@0\"\n"
    );
    let mut offset = init_len;
    for fragment in fragments {
        playlist.push_str(&format!(
            "#EXTINF:1.0,\n#EXT-X-BYTERANGE:{}@{offset}\nall.m4s\n",
            fragment.len()
        ));
        offset += fragment.len();
    }
    playlist.push_str("#EXT-X-ENDLIST\n");
    playlist
}

#[tokio::test]
async fn byterange_single_file_cmaf_fetches_subranges_and_demuxes() {
    let aus = access_units();
    let fmp4 = make_fmp4(&aus).await;
    let (init, segs) = split_fmp4(&fmp4);
    assert_eq!(segs.len(), 3);

    // One resource: init then the three fragments back to back.
    let mut resource = init.clone();
    for s in &segs {
        resource.extend_from_slice(s);
    }

    // Playlist addresses each piece by byte range of `all.m4s`: the MAP carries an
    // explicit offset, seg0 an explicit offset, seg1/seg2 implicit (continuing).
    let init_len = init.len();
    let playlist = format!(
        "#EXTM3U\n#EXT-X-TARGETDURATION:1\n\
         #EXT-X-MAP:URI=\"all.m4s\",BYTERANGE=\"{}@0\"\n\
         #EXTINF:1.0,\n#EXT-X-BYTERANGE:{}@{}\nall.m4s\n\
         #EXTINF:1.0,\n#EXT-X-BYTERANGE:{}\nall.m4s\n\
         #EXTINF:1.0,\n#EXT-X-BYTERANGE:{}\nall.m4s\n\
         #EXT-X-ENDLIST\n",
        init_len,
        segs[0].len(),
        init_len,
        segs[1].len(),
        segs[2].len(),
    );
    let (base, _) = serve_byterange(resource.clone(), playlist);

    let mut src = HlsSrc::new(format!("{base}/cmaf.m3u8"));
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    src.run(&mut sink).await.unwrap();

    // The reassembled byte stream is init + the three fragments, byte-exact: each
    // came from its own Range request against the single resource.
    assert_eq!(
        sink.body, resource,
        "byte-range fetches reassemble the single-file resource"
    );

    // End to end: the reassembled fMP4 demuxes back to the original access units.
    let mut dmx = Fmp4Demux::new();
    dmx.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .unwrap();
    let mut dsink = DemuxSink::default();
    dmx.process(
        PipelinePacket::DataFrame(au_frame(sink.body.clone(), 0, 0)),
        &mut dsink,
    )
    .await
    .unwrap();
    assert_eq!(
        dsink.aus, aus,
        "byte-range CMAF -> Fmp4Demux recovers the access units"
    );
}

#[tokio::test]
async fn the_fanout_probe_fetches_only_the_init_byte_range() {
    let (init, segs) = split_fmp4(&make_fmp4(&access_units()).await);
    let mut resource = init.clone();
    for s in &segs {
        resource.extend_from_slice(s);
    }
    let (base, ranges) = serve_byterange(resource, byterange_playlist(init.len(), &segs));

    let fanout = hls_uri_fanout(&Registry::new(), &format!("{base}/master.m3u8"))
        .unwrap()
        .expect("the init's video track fans out");
    assert_eq!(fanout.ports.len(), 1);
    assert_eq!(*ranges.lock().unwrap(), [Some((0, init.len() - 1))]);
}

#[derive(Default)]
struct DemuxSink {
    aus: Vec<Vec<u8>>,
}
impl OutputSink for DemuxSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.aus.push(s.to_vec());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}
