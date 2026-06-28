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

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{SeekController, SourceLoop};
use g2g_core::{
    ByteStreamEncoding, Caps, CapsConstraint, Dim, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, PushOutcome, Rate, Seek, VideoCodec,
};
use g2g_plugins::fmp4demux::Fmp4Demux;
use g2g_plugins::hlssrc::HlsSrc;
use g2g_plugins::mp4mux::Mp4Mux;

#[derive(Default)]
struct CaptureSink {
    body: Vec<u8>,
    frames: usize,
    eos: bool,
    flushes: usize,
    /// Stream-time start of each post-flush `Segment` emitted (seek observability).
    segment_starts: Vec<u64>,
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
                PipelinePacket::Flush => self.flushes += 1,
                PipelinePacket::Segment(seg) => self.segment_starts.push(seg.start),
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
    (format!("http://127.0.0.1:{port}/master.m3u8"), media_fetches)
}

#[tokio::test]
async fn probe_playlist_is_reused_by_run_not_refetched() {
    let seg0: Vec<u8> = (0..1000u32).map(|i| i as u8).collect();
    let seg1: Vec<u8> = (0..1000u32).map(|i| (i as u8) ^ 0x5a).collect();
    let (url, media_fetches) = serve_counting(seg0, seg1);

    let mut src = HlsSrc::new(url);
    // Negotiation probe resolves the media playlist once.
    let _ = src.caps_constraint().await.unwrap();
    src.configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs }).unwrap();
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
    let segs: Vec<Vec<u8>> =
        (0..3u8).map(|s| (0..1000u32).map(|i| (i as u8) ^ (s * 53)).collect()).collect();
    let url = serve_seek(segs.clone());

    let seek = SeekController::new();
    // Pre-arm a 5s seek before run(): deterministic (no fetch/EOF race). 5s lands
    // in seg1 ([4,8)s), so playback must start there, skipping seg0.
    seek.seek(Seek::flush_to(5 * 1_000_000_000));

    let mut src = HlsSrc::new(url).with_seek(seek);
    src.configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs }).unwrap();
    let mut sink = CaptureSink::default();
    src.run(&mut sink).await.unwrap();

    assert!(sink.eos, "the VOD playlist still ends with Eos");
    assert_eq!(sink.flushes, 1, "the flushing seek emits exactly one Flush");
    assert_eq!(
        sink.segment_starts,
        vec![4 * 1_000_000_000],
        "the post-flush Segment starts at seg1's cumulative time (4s)"
    );
    // Body is seg1 then seg2: the seek resumes at the target segment and plays on.
    let mut expected = segs[1].clone();
    expected.extend_from_slice(&segs[2]);
    assert_eq!(sink.body, expected, "playback starts at seg1, seg0 skipped");
}

// --- fMP4 / CMAF over HLS (EXT-X-MAP) -------------------------------------

fn au_frame(bytes: Vec<u8>, pts_ns: u64, seq: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming { pts_ns, dts_ns: pts_ns, duration_ns: 33_333_333, ..FrameTiming::default() },
        sequence: seq,
        meta: Default::default(),
    }
}

fn access_units() -> Vec<Vec<u8>> {
    let sps = [0x67u8, 0x42, 0xC0, 0x1E, 0x11, 0x22];
    let pps = [0x68u8, 0xCE, 0x3C, 0x80];
    let idr: Vec<u8> =
        [&[0, 0, 0, 1][..], &sps, &[0, 0, 0, 1], &pps, &[0, 0, 0, 1], &[0x65, 0xAA, 0xBB]].concat();
    let p = |f: u8| [&[0, 0, 0, 1][..], &[0x41, f, f]].concat();
    vec![idr, p(1), p(2)]
}

/// Mux the access units to an fMP4 buffer via the `Mp4Mux` element, returning
/// the byte stream it forwards downstream.
async fn make_fmp4(aus: &[Vec<u8>]) -> Vec<u8> {
    let mut mux = Mp4Mux::new();
    mux.configure_pipeline(&Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(64),
        height: Dim::Fixed(48),
        framerate: Rate::Fixed(30 << 16),
    })
    .unwrap();
    let mut out = CaptureSink::default();
    for (i, au) in aus.iter().enumerate() {
        mux.process(PipelinePacket::DataFrame(au_frame(au.clone(), i as u64 * 33_333_333, i as u64)), &mut out)
            .await
            .unwrap();
    }
    mux.process(PipelinePacket::Eos, &mut out).await.unwrap();
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
    let mut playlist = String::from("#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXT-X-MAP:URI=\"init.mp4\"\n");
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
                g2g_core::CapsSet::one(Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff })
            ),
            _ => panic!("fMP4 HLS should produce IsoBmff caps"),
        }
    }

    src.configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff }).unwrap();
    let mut sink = CaptureSink::default();
    src.run(&mut sink).await.unwrap();

    // The byte stream is the init segment followed by every media fragment.
    let mut expected = init.clone();
    for s in &segs {
        expected.extend_from_slice(s);
    }
    assert_eq!(sink.body, expected, "init segment emitted first, then fragments in order");

    // End to end: the delivered fMP4 byte stream demuxes back to the access units.
    let mut dmx = Fmp4Demux::new();
    dmx.configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff }).unwrap();
    let mut dsink = DemuxSink::default();
    dmx.process(PipelinePacket::DataFrame(au_frame(sink.body.clone(), 0, 0)), &mut dsink)
        .await
        .unwrap();
    assert_eq!(dsink.aus, aus, "HlsSrc -> Fmp4Demux recovers the original access units");
}

#[derive(Default)]
struct DemuxSink {
    aus: Vec<Vec<u8>>,
}
impl OutputSink for DemuxSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let MemoryDomain::System(s) = &f.domain {
                    self.aus.push(s.as_slice().to_vec());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}
