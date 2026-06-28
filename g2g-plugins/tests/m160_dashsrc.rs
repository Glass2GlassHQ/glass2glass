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

/// Parse a `Range: bytes=a-b` request header, returning the inclusive `(a, b)`.
fn parse_range_header(req: &str) -> Option<(usize, usize)> {
    let line = req.lines().find(|l| l.to_ascii_lowercase().starts_with("range:"))?;
    let spec = line.split_once('=')?.1.trim();
    let (a, b) = spec.split_once('-')?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
}

/// Serve a single-file CMAF resource `all.m4s` honouring HTTP `Range` with `206`,
/// plus a `SegmentList` MPD that addresses init + fragments by `mediaRange`.
fn serve_segment_list(resource: Vec<u8>, mpd: String) -> String {
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
            let text = String::from_utf8_lossy(&req);
            let path = text.split_whitespace().nth(1).unwrap_or("");
            if path == "/manifest.mpd" {
                let body = mpd.clone().into_bytes();
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(header.as_bytes());
                let _ = stream.write_all(&body);
            } else if path == "/all.m4s" {
                let (body, status) = match parse_range_header(&text) {
                    Some((a, b)) => {
                        let end = (b + 1).min(resource.len());
                        let start = a.min(end);
                        (resource[start..end].to_vec(), "206 Partial Content")
                    }
                    None => (resource.clone(), "200 OK"),
                };
                let header = format!(
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(header.as_bytes());
                let _ = stream.write_all(&body);
            } else {
                let _ = stream
                    .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            }
        }
    });
    format!("http://127.0.0.1:{port}/manifest.mpd")
}

#[tokio::test]
async fn dash_segment_list_byte_ranges_single_file_cmaf_demuxes() {
    let aus = access_units();
    let fmp4 = make_fmp4(&aus).await;
    let (init, segs) = split_fmp4(&fmp4);
    assert_eq!(segs.len(), 3);

    // One resource: init then the three fragments back to back; the SegmentList
    // addresses each piece by an inclusive `mediaRange`/`range` of `all.m4s`.
    let mut resource = init.clone();
    for s in &segs {
        resource.extend_from_slice(s);
    }
    let r = |start: usize, len: usize| format!("{}-{}", start, start + len - 1);
    let o0 = init.len();
    let o1 = o0 + segs[0].len();
    let o2 = o1 + segs[1].len();
    let mpd = format!(
        "<?xml version=\"1.0\"?>\n\
         <MPD type=\"static\"><Period><AdaptationSet mimeType=\"video/mp4\" codecs=\"avc1.4d401f\">\n\
           <BaseURL>all.m4s</BaseURL>\n\
           <SegmentList duration=\"1000\" timescale=\"1000\">\n\
             <Initialization range=\"{}\"/>\n\
             <SegmentURL mediaRange=\"{}\"/>\n\
             <SegmentURL mediaRange=\"{}\"/>\n\
             <SegmentURL mediaRange=\"{}\"/>\n\
           </SegmentList>\n\
           <Representation id=\"v0\" bandwidth=\"1000000\" width=\"64\" height=\"48\"/>\n\
         </AdaptationSet></Period></MPD>",
        r(0, init.len()),
        r(o0, segs[0].len()),
        r(o1, segs[1].len()),
        r(o2, segs[2].len()),
    );
    let url = serve_segment_list(resource.clone(), mpd);

    let mut src = DashSrc::new(url);
    src.configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff }).unwrap();
    let mut sink = CaptureSink::default();
    src.run(&mut sink).await.unwrap();

    // The byte-range fetches reassemble the single-file resource exactly.
    assert_eq!(sink.body, resource, "SegmentList byte ranges reassemble the single-file resource");

    // End to end: the reassembled fMP4 demuxes back to the original access units.
    let mut dmx = Fmp4Demux::new();
    dmx.configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff }).unwrap();
    let mut dsink = CaptureSink::default();
    dmx.process(PipelinePacket::DataFrame(au_frame(sink.body.clone(), 0, 0)), &mut dsink)
        .await
        .unwrap();
    assert_eq!(dsink.aus, aus, "SegmentList CMAF -> Fmp4Demux recovers the access units");
}

/// Build a version-0 `sidx` box from `(referenced_size, subsegment_duration)`.
fn build_sidx(timescale: u32, entries: &[(u32, u32)]) -> Vec<u8> {
    let mut b = Vec::new();
    let box_size = 32 + 12 * entries.len() as u32;
    b.extend_from_slice(&box_size.to_be_bytes());
    b.extend_from_slice(b"sidx");
    b.extend_from_slice(&[0, 0, 0, 0]); // version 0 + flags
    b.extend_from_slice(&1u32.to_be_bytes()); // reference_ID
    b.extend_from_slice(&timescale.to_be_bytes());
    b.extend_from_slice(&0u32.to_be_bytes()); // earliest_presentation_time
    b.extend_from_slice(&0u32.to_be_bytes()); // first_offset
    b.extend_from_slice(&0u16.to_be_bytes()); // reserved
    b.extend_from_slice(&(entries.len() as u16).to_be_bytes());
    for &(size, dur) in entries {
        b.extend_from_slice(&(size & 0x7fff_ffff).to_be_bytes());
        b.extend_from_slice(&dur.to_be_bytes());
        b.extend_from_slice(&0x9000_0000u32.to_be_bytes()); // SAP
    }
    b
}

#[tokio::test]
async fn dash_segment_base_sidx_indexed_single_file_demuxes() {
    let aus = access_units();
    let fmp4 = make_fmp4(&aus).await;
    let (init, segs) = split_fmp4(&fmp4);
    assert_eq!(segs.len(), 3);

    // Single-file layout: [init][sidx][frag0][frag1][frag2]. The sidx indexes the
    // three fragments; the source fetches init + each fragment by range (never the
    // sidx), so the demuxer sees init + fragments just like the other profiles.
    let sidx = build_sidx(1000, &segs.iter().map(|s| (s.len() as u32, 1000)).collect::<Vec<_>>());
    let mut resource = init.clone();
    resource.extend_from_slice(&sidx);
    for s in &segs {
        resource.extend_from_slice(s);
    }

    let init_end = init.len() - 1;
    let idx_start = init.len();
    let idx_end = init.len() + sidx.len() - 1;
    let mpd = format!(
        "<?xml version=\"1.0\"?>\n\
         <MPD type=\"static\"><Period><AdaptationSet mimeType=\"video/mp4\" codecs=\"avc1.4d401f\">\n\
           <BaseURL>all.m4s</BaseURL>\n\
           <Representation id=\"v0\" bandwidth=\"1000000\" width=\"64\" height=\"48\">\n\
             <SegmentBase indexRange=\"{idx_start}-{idx_end}\" timescale=\"1000\">\n\
               <Initialization range=\"0-{init_end}\"/>\n\
             </SegmentBase>\n\
           </Representation>\n\
         </AdaptationSet></Period></MPD>"
    );
    let url = serve_segment_list(resource, mpd);

    let mut src = DashSrc::new(url);
    src.configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff }).unwrap();
    let mut sink = CaptureSink::default();
    src.run(&mut sink).await.unwrap();

    // Fetched stream is init + the three fragments (the sidx is not fetched).
    let mut expected = init.clone();
    for s in &segs {
        expected.extend_from_slice(s);
    }
    assert_eq!(sink.body, expected, "sidx-indexed fetches skip the index, reassemble init+frags");

    let mut dmx = Fmp4Demux::new();
    dmx.configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff }).unwrap();
    let mut dsink = CaptureSink::default();
    dmx.process(PipelinePacket::DataFrame(au_frame(sink.body.clone(), 0, 0)), &mut dsink)
        .await
        .unwrap();
    assert_eq!(dsink.aus, aus, "SegmentBase CMAF -> Fmp4Demux recovers the access units");
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
