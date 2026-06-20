//! M160 DASH source end to end: `DashSrc` parses an MPD, selects a
//! Representation, and streams its fMP4 init + media segments (SegmentTemplate
//! $Number$ addressing). A local routing server serves the manifest + segments
//! (real fMP4 from `Mp4Sink`); `DashSrc -> Fmp4Demux` recovers the access units.

#![cfg(feature = "dash")]

use core::future::Future;
use core::pin::Pin;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::thread;

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    ByteStreamEncoding, Caps, Dim, G2gError, MemoryDomain, OutputSink, PipelinePacket, PushOutcome,
    Rate, VideoCodec,
};
use g2g_plugins::dashsrc::DashSrc;
use g2g_plugins::fmp4demux::Fmp4Demux;
use g2g_plugins::mp4sink::Mp4Sink;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g_m160_{}_{}.mp4", std::process::id(), name))
}

struct NullOut;
impl OutputSink for NullOut {
    fn push<'a>(
        &'a mut self,
        _p: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

#[derive(Default)]
struct CaptureSink {
    body: Vec<u8>,
    aus: Vec<Vec<u8>>,
}
impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let MemoryDomain::System(s) = &f.domain {
                    self.body.extend_from_slice(s.as_slice());
                    self.aus.push(s.as_slice().to_vec());
                }
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

async fn make_fmp4(aus: &[Vec<u8>]) -> Vec<u8> {
    let path = temp_path("dash");
    let mut sink = Mp4Sink::new(&path);
    sink.configure_pipeline(&Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(64),
        height: Dim::Fixed(48),
        framerate: Rate::Fixed(30 << 16),
    })
    .unwrap();
    let mut out = NullOut;
    for (i, au) in aus.iter().enumerate() {
        sink.process(PipelinePacket::DataFrame(au_frame(au.clone(), i as u64 * 33_333_333, i as u64)), &mut out)
            .await
            .unwrap();
    }
    sink.process(PipelinePacket::Eos, &mut out).await.unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    bytes
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
