//! M160 DASH source end to end: `DashSrc` parses an MPD, selects a
//! Representation, and streams its fMP4 init + media segments (SegmentTemplate
//! $Number$ addressing). A local routing server serves the manifest + segments
//! (real fMP4 from `Mp4Mux`); `DashSrc -> Fmp4Demux` recovers the access units.

#![cfg(feature = "dash")]

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
    ByteStreamEncoding, Caps, Dim, G2gError, MemoryDomain, OutputSink, PipelinePacket, PushOutcome,
    Rate, Seek, VideoCodec,
};
use g2g_plugins::dashsrc::DashSrc;
use g2g_plugins::fmp4demux::Fmp4Demux;
use g2g_plugins::mp4mux::Mp4Mux;

#[derive(Default)]
struct CaptureSink {
    body: Vec<u8>,
    aus: Vec<Vec<u8>>,
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
                        self.aus.push(s.as_slice().to_vec());
                    }
                }
                PipelinePacket::Flush => self.flushes += 1,
                PipelinePacket::Segment(seg) => self.segment_starts.push(seg.start),
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

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

/// Mux the access units to an fMP4 byte buffer via the `Mp4Mux` element,
/// returning the byte stream it forwards downstream.
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

/// Split fMP4 into the init segment (ftyp+moov) and one segment per moof+mdat.
fn split_fmp4(data: &[u8]) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut spans = Vec::new();
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
            segments.push(data[spans[j].1..spans[j + 1].2].to_vec());
            j += 2;
        } else {
            j += 1;
        }
    }
    (init, segments)
}

fn serve(init: Vec<u8>, segs: Vec<Vec<u8>>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    // 3 segments at 1s each, startNumber=0 -> seg0.m4s..seg2.m4s.
    let mpd = format!(
        "<?xml version=\"1.0\"?>\n\
         <MPD mediaPresentationDuration=\"PT{}S\" type=\"static\">\n\
           <Period>\n\
             <AdaptationSet mimeType=\"video/mp4\" codecs=\"avc1.4d401f\">\n\
               <SegmentTemplate initialization=\"init.mp4\" media=\"seg$Number$.m4s\" \
                  startNumber=\"0\" duration=\"1000\" timescale=\"1000\"/>\n\
               <Representation id=\"v0\" bandwidth=\"1000000\" width=\"64\" height=\"48\"/>\n\
             </AdaptationSet>\n\
           </Period>\n\
         </MPD>",
        segs.len()
    );
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
            let body: Vec<u8> = if path == "/manifest.mpd" {
                mpd.clone().into_bytes()
            } else if path == "/init.mp4" {
                init.clone()
            } else if let Some(idx) = path
                .strip_prefix("/seg")
                .and_then(|s| s.strip_suffix(".m4s"))
                .and_then(|s| s.parse::<usize>().ok())
            {
                segs.get(idx).cloned().unwrap_or_default()
            } else {
                let _ = stream
                    .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
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
    format!("http://127.0.0.1:{port}/manifest.mpd")
}

/// Serve a `SegmentTimeline` + `$Time$` manifest: one `<S>` with `r` repeats so
/// the 1s segments map to start times 0, 1000, 2000 (`seg<time>.m4s`).
fn serve_timeline(init: Vec<u8>, segs: Vec<Vec<u8>>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let repeats = segs.len() as u64 - 1;
    let mpd = format!(
        "<?xml version=\"1.0\"?>\n\
         <MPD type=\"static\">\n\
           <Period>\n\
             <AdaptationSet mimeType=\"video/mp4\" codecs=\"avc1.4d401f\">\n\
               <SegmentTemplate initialization=\"init.mp4\" media=\"seg$Time$.m4s\" \
                  startNumber=\"1\" timescale=\"1000\">\n\
                 <SegmentTimeline><S t=\"0\" d=\"1000\" r=\"{repeats}\"/></SegmentTimeline>\n\
               </SegmentTemplate>\n\
               <Representation id=\"v0\" bandwidth=\"1000000\" width=\"64\" height=\"48\"/>\n\
             </AdaptationSet>\n\
           </Period>\n\
         </MPD>"
    );
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
            let body: Vec<u8> = if path == "/manifest.mpd" {
                mpd.clone().into_bytes()
            } else if path == "/init.mp4" {
                init.clone()
            } else if let Some(time) = path
                .strip_prefix("/seg")
                .and_then(|s| s.strip_suffix(".m4s"))
                .and_then(|s| s.parse::<usize>().ok())
            {
                segs.get(time / 1000).cloned().unwrap_or_default()
            } else {
                let _ = stream
                    .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
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
    format!("http://127.0.0.1:{port}/manifest.mpd")
}

/// The dynamic MPD returned on the Nth reload: a 2-segment sliding window over
/// the SegmentTimeline that advances each time and turns `static` on the third.
fn live_mpd(reload: usize) -> String {
    let (start_t, mpd_type) = match reload {
        0 => (0, "dynamic"),
        1 => (1000, "dynamic"),
        _ => (2000, "static"),
    };
    format!(
        "<?xml version=\"1.0\"?>\n\
         <MPD type=\"{mpd_type}\" minimumUpdatePeriod=\"PT1S\">\n\
           <Period>\n\
             <AdaptationSet mimeType=\"video/mp4\" codecs=\"avc1.4d401f\">\n\
               <SegmentTemplate initialization=\"init.mp4\" media=\"seg$Time$.m4s\" \
                  startNumber=\"1\" timescale=\"1000\">\n\
                 <SegmentTimeline><S t=\"{start_t}\" d=\"1000\" r=\"1\"/></SegmentTimeline>\n\
               </SegmentTemplate>\n\
               <Representation id=\"v0\" bandwidth=\"1000000\" width=\"64\" height=\"48\"/>\n\
             </AdaptationSet>\n\
           </Period>\n\
         </MPD>"
    )
}

fn serve_live(init: Vec<u8>, segs: Vec<Vec<u8>>) -> String {
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
            let body: Vec<u8> = if path == "/manifest.mpd" {
                live_mpd(reloads.fetch_add(1, Ordering::SeqCst)).into_bytes()
            } else if path == "/init.mp4" {
                init.clone()
            } else if let Some(time) = path
                .strip_prefix("/seg")
                .and_then(|s| s.strip_suffix(".m4s"))
                .and_then(|s| s.parse::<usize>().ok())
            {
                segs.get(time / 1000).cloned().unwrap_or_default()
            } else {
                let _ = stream
                    .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
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
    format!("http://127.0.0.1:{port}/manifest.mpd")
}

#[tokio::test]
async fn dash_live_reloads_dynamic_mpd_and_plays_each_segment_once() {
    // Four access units -> four 1s fragments addressed by $Time$ (0,1000,2000,3000).
    let mut aus = access_units();
    aus.push([&[0, 0, 0, 1][..], &[0x41u8, 3, 3]].concat());
    let fmp4 = make_fmp4(&aus).await;
    let (init, segs) = split_fmp4(&fmp4);
    assert_eq!(segs.len(), 4);
    let url = serve_live(init.clone(), segs.clone());

    let mut src = DashSrc::new(url).with_reload_interval_ms(20);
    src.configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff }).unwrap();
    let mut sink = CaptureSink::default();
    let count = src.run(&mut sink).await.unwrap();

    assert_eq!(count, 5, "init + 4 segments, each played once across reloads");
    let mut expected = init.clone();
    for s in &segs {
        expected.extend_from_slice(s);
    }
    assert_eq!(sink.body, expected, "sliding-window segments delivered once, in order");

    // End to end through the demuxer.
    let mut dmx = Fmp4Demux::new();
    dmx.configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff }).unwrap();
    let mut dsink = CaptureSink::default();
    dmx.process(PipelinePacket::DataFrame(au_frame(sink.body.clone(), 0, 0)), &mut dsink)
        .await
        .unwrap();
    assert_eq!(dsink.aus, aus, "live DASH -> Fmp4Demux recovers all access units once");
}

#[tokio::test]
async fn dash_segment_timeline_time_addressing_demuxes() {
    let aus = access_units();
    let fmp4 = make_fmp4(&aus).await;
    let (init, segs) = split_fmp4(&fmp4);
    let url = serve_timeline(init.clone(), segs.clone());

    let mut src = DashSrc::new(url);
    src.configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff }).unwrap();
    let mut sink = CaptureSink::default();
    let count = src.run(&mut sink).await.unwrap();

    assert_eq!(count, 4, "init + 3 timeline segments addressed by $Time$");
    let mut expected = init.clone();
    for s in &segs {
        expected.extend_from_slice(s);
    }
    assert_eq!(sink.body, expected, "timeline segments delivered in time order");

    let mut dmx = Fmp4Demux::new();
    dmx.configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff }).unwrap();
    let mut dsink = CaptureSink::default();
    dmx.process(PipelinePacket::DataFrame(au_frame(sink.body.clone(), 0, 0)), &mut dsink)
        .await
        .unwrap();
    assert_eq!(dsink.aus, aus, "SegmentTimeline DashSrc -> Fmp4Demux recovers the access units");
}

#[tokio::test]
async fn dash_time_seek_jumps_to_the_segment_containing_the_target() {
    // 3 fragments at 1s each (SegmentTemplate @duration, timescale 1000): segment
    // start times 0, 1s, 2s. A 1.5s seek lands in seg1 ([1,2)s).
    let aus = access_units();
    let fmp4 = make_fmp4(&aus).await;
    let (init, segs) = split_fmp4(&fmp4);
    assert_eq!(segs.len(), 3);
    let url = serve(init.clone(), segs.clone());

    let seek = SeekController::new();
    // Pre-arm before run(): deterministic, no fetch/EOF race.
    seek.seek(Seek::flush_to(1_500_000_000));

    let mut src = DashSrc::new(url).with_seek(seek);
    src.configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff }).unwrap();
    let mut sink = CaptureSink::default();
    src.run(&mut sink).await.unwrap();

    assert_eq!(sink.flushes, 1, "the flushing seek emits exactly one Flush");
    assert_eq!(
        sink.segment_starts,
        vec![1_000_000_000],
        "the post-flush Segment starts at seg1's $Time$ (1s)"
    );
    // After the seek the init is re-emitted (a reset demuxer needs its moov), then
    // seg1 and seg2: playback resumes at the target segment.
    let mut expected = init.clone();
    expected.extend_from_slice(&segs[1]);
    expected.extend_from_slice(&segs[2]);
    assert_eq!(sink.body, expected, "init re-emitted, then seg1 onward; seg0 skipped");

    // The re-emitted init + the two fragments still demux to two access units
    // (the post-target tail). The first emitted AU gets the config-record
    // parameter sets prepended (fmp4demux re-arms them after the reset), so check
    // the count and that the last AU is byte-exact rather than equality on all.
    let mut dmx = Fmp4Demux::new();
    dmx.configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff }).unwrap();
    let mut dsink = CaptureSink::default();
    dmx.process(PipelinePacket::DataFrame(au_frame(sink.body.clone(), 0, 0)), &mut dsink)
        .await
        .unwrap();
    assert_eq!(dsink.aus.len(), 2, "seek output demuxes to the two post-target access units");
    assert_eq!(dsink.aus[1], aus[2], "the last demuxed AU is byte-exact (no param-set prepend)");
}

#[tokio::test]
async fn dash_streams_init_then_segments_and_demuxes() {
    let aus = access_units();
    let fmp4 = make_fmp4(&aus).await;
    let (init, segs) = split_fmp4(&fmp4);
    assert_eq!(segs.len(), 3, "one fragment per access unit");
    let url = serve(init.clone(), segs.clone());

    let mut src = DashSrc::new(url);
    src.configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff }).unwrap();
    let mut sink = CaptureSink::default();
    let count = src.run(&mut sink).await.unwrap();

    assert_eq!(count, 4, "init segment + 3 media segments");
    let mut expected = init.clone();
    for s in &segs {
        expected.extend_from_slice(s);
    }
    assert_eq!(sink.body, expected, "init first, then segments in $Number$ order");

    // End to end: the delivered byte stream demuxes back to the access units.
    let mut dmx = Fmp4Demux::new();
    dmx.configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff }).unwrap();
    let mut dsink = CaptureSink::default();
    dmx.process(PipelinePacket::DataFrame(au_frame(sink.body.clone(), 0, 0)), &mut dsink)
        .await
        .unwrap();
    assert_eq!(dsink.aus, aus, "DashSrc -> Fmp4Demux recovers the original access units");
}
