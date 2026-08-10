//! M922 - an fMP4 (`wvtt`) HLS subtitle rendition plays like a raw `.vtt` one:
//! `HlsSrc(text)` fetches the rendition's `#EXT-X-MAP` init segment, de-frames the
//! ISO 14496-30 cues out of each fragment, and forwards them as the same
//! `Caps::Text{WebVtt}` stream a `.vtt` rendition produces, so the
//! `build_hls_subtitle_overlay` branch (`HlsSrc(text) -> SubParse -> overlay`)
//! works unchanged. The source and the real `SubParse` are chained here, so the
//! assertion is on the timed cues that reach the overlay's text pad.
//!
//! A local HTTP server (no extra deps) serves the playlist, the init segment and
//! the hand-built fragments.

#![cfg(all(feature = "std", feature = "hls"))]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use g2g_core::element::AsyncElement;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, ConfigureOutcome, G2gError, OutputSink, PipelinePacket, PushOutcome, TextFormat,
};
use g2g_plugins::hlssrc::HlsSrc;
use g2g_plugins::subparse::SubParse;

// ---------------------------------------------------------------- box building

fn mp4_box(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = ((body.len() + 8) as u32).to_be_bytes().to_vec();
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
    out
}

fn concat(parts: &[Vec<u8>]) -> Vec<u8> {
    parts.iter().flat_map(|p| p.iter().copied()).collect()
}

/// A `tkhd` v0 whose `track_ID` sits at payload offset 12 and whose width /
/// height 16.16 fields sit at 76 / 80, the layout the parser reads.
fn tkhd(track_id: u32) -> Vec<u8> {
    let mut p = vec![0u8; 84];
    p[12..16].copy_from_slice(&track_id.to_be_bytes());
    mp4_box(b"tkhd", &p)
}

/// An `mdhd` v0 with the media `timescale` at payload offset 12.
fn mdhd(timescale: u32) -> Vec<u8> {
    let mut p = vec![0u8; 20];
    p[12..16].copy_from_slice(&timescale.to_be_bytes());
    mp4_box(b"mdhd", &p)
}

/// An `hdlr` whose 4-byte handler type sits at payload offset 8.
fn hdlr(handler: &[u8; 4]) -> Vec<u8> {
    let mut p = vec![0u8; 8];
    p.extend_from_slice(handler);
    p.extend_from_slice(b"g2g\0");
    mp4_box(b"hdlr", &p)
}

/// A WebVTT-in-MP4 init segment: `ftyp` + a `moov` with one `subt` track whose
/// sample entry is `wvtt` (its `vttC` config is the WebVTT header).
fn init_segment(track_id: u32, timescale: u32) -> Vec<u8> {
    let wvtt = mp4_box(b"wvtt", &mp4_box(b"vttC", b"WEBVTT"));
    // stsd: version/flags + entry_count, then the sample entry.
    let mut stsd_body = vec![0u8, 0, 0, 0, 0, 0, 0, 1];
    stsd_body.extend_from_slice(&wvtt);
    let stbl = mp4_box(b"stbl", &mp4_box(b"stsd", &stsd_body));
    let minf = mp4_box(b"minf", &stbl);
    let mdia = mp4_box(b"mdia", &concat(&[mdhd(timescale), hdlr(b"subt"), minf]));
    let trak = mp4_box(b"trak", &concat(&[tkhd(track_id), mdia]));
    let moov = mp4_box(b"moov", &trak);
    concat(&[mp4_box(b"ftyp", b"iso6\0\0\0\x01iso6"), moov])
}

/// One `wvtt` sample: a `vttc` cue box carrying the UTF-8 payload in a `payl`.
fn wvtt_sample(text: &str) -> Vec<u8> {
    mp4_box(b"vttc", &mp4_box(b"payl", text.as_bytes()))
}

/// One media fragment: `moof` (`tfhd` + `tfdt` + `trun`) + `mdat`, with each
/// sample's duration and size written per sample in the `trun`.
fn fragment(track_id: u32, base_time: u32, samples: &[(Vec<u8>, u32)]) -> Vec<u8> {
    let mut tfhd_body = vec![0u8; 4]; // version/flags: no optional fields
    tfhd_body.extend_from_slice(&track_id.to_be_bytes());
    let tfhd = mp4_box(b"tfhd", &tfhd_body);

    let mut tfdt_body = vec![0u8; 4]; // version 0
    tfdt_body.extend_from_slice(&base_time.to_be_bytes());
    let tfdt = mp4_box(b"tfdt", &tfdt_body);

    // trun v0, flags 0x300: per-sample duration + per-sample size.
    let mut trun_body = vec![0u8, 0x00, 0x03, 0x00];
    trun_body.extend_from_slice(&(samples.len() as u32).to_be_bytes());
    for (bytes, duration) in samples {
        trun_body.extend_from_slice(&duration.to_be_bytes());
        trun_body.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
    }
    let trun = mp4_box(b"trun", &trun_body);

    let traf = mp4_box(b"traf", &concat(&[tfhd, tfdt, trun]));
    let moof = mp4_box(b"moof", &traf);
    let mdat_body: Vec<u8> = samples
        .iter()
        .flat_map(|(b, _)| b.iter().copied())
        .collect();
    concat(&[moof, mp4_box(b"mdat", &mdat_body)])
}

// ------------------------------------------------------------------ HTTP serve

const PLAYLIST: &str = "#EXTM3U\n\
    #EXT-X-TARGETDURATION:4\n\
    #EXT-X-MEDIA-SEQUENCE:0\n\
    #EXT-X-MAP:URI=\"init.mp4\"\n\
    #EXTINF:4.0,\n\
    seg0.m4s\n\
    #EXTINF:4.0,\n\
    seg1.m4s\n\
    #EXT-X-ENDLIST\n";

/// A raw `.vtt` rendition (no `#EXT-X-MAP`), the carriage the fMP4 one has to
/// behave like.
const VTT_PLAYLIST: &str = "#EXTM3U\n\
    #EXT-X-TARGETDURATION:4\n\
    #EXT-X-MEDIA-SEQUENCE:0\n\
    #EXTINF:4.0,\n\
    seg0.m4s\n\
    #EXTINF:4.0,\n\
    seg1.m4s\n\
    #EXT-X-ENDLIST\n";

fn serve(init: Vec<u8>, seg0: Vec<u8>, seg1: Vec<u8>) -> String {
    serve_playlist(PLAYLIST, init, seg0, seg1)
}

fn serve_playlist(playlist: &'static str, init: Vec<u8>, seg0: Vec<u8>, seg1: Vec<u8>) -> String {
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
                "/subs.m3u8" => playlist.as_bytes().to_vec(),
                "/init.mp4" => init.clone(),
                "/seg0.m4s" => seg0.clone(),
                "/seg1.m4s" => seg1.clone(),
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
    format!("http://127.0.0.1:{port}/subs.m3u8")
}

// --------------------------------------------------------------------- sinks

/// The cues that reach the overlay's text pad: one `(pts_ns, duration_ns, text)`
/// per `Caps::Text{Utf8}` frame `SubParse` emits.
#[derive(Default)]
struct CueSink {
    cues: Vec<(u64, u64, String)>,
    caps: Option<Caps>,
}

impl OutputSink for CueSink {
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
                        self.cues.push((
                            f.timing.pts_ns,
                            f.timing.duration_ns,
                            String::from_utf8_lossy(s).into_owned(),
                        ));
                    }
                }
                PipelinePacket::CapsChanged(c) => self.caps = Some(c),
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// The `HlsSrc(text) -> SubParse` link: every packet the source pushes runs
/// through the real parser element, whose cues land in `cues`.
struct SubParseSink {
    sub: SubParse,
    cues: CueSink,
}

impl OutputSink for SubParseSink {
    fn poll_push(
        &mut self,
        cx: &mut core::task::Context<'_>,
        packet: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet.take().expect("poll_push without a packet");
        let mut fut = self.sub.process(packet, &mut self.cues);
        match fut.as_mut().poll(cx) {
            core::task::Poll::Ready(r) => {
                core::task::Poll::Ready(r.map(|()| PushOutcome::Accepted))
            }
            // In-memory test elements never block: their only awaits are
            // pushes into always-ready capture sinks.
            core::task::Poll::Pending => panic!("element future did not resolve in one poll"),
        }
    }
}

// --------------------------------------------------------------------- tests

#[tokio::test]
async fn fmp4_wvtt_rendition_reaches_the_overlay_as_timed_cues() {
    // Two fragments on a 1 kHz media timescale: cues at 1s (2s long) and 5s.
    // The second sample of fragment 0 is an empty gap, which must not become a cue.
    let init = init_segment(1, 1000);
    let seg0 = fragment(
        1,
        1000,
        &[
            (wvtt_sample("Hello\nthere"), 2000),
            (mp4_box(b"vtte", b""), 1000),
        ],
    );
    let seg1 = fragment(1, 5000, &[(wvtt_sample("World"), 1500)]);
    let url = serve(init, seg0, seg1);

    let mut src = HlsSrc::new(url).with_text();
    let caps = src.intercept_caps().await.unwrap();
    assert_eq!(
        caps,
        Caps::Text {
            format: TextFormat::WebVtt
        },
        "a subtitle rendition advertises WebVTT text whatever its segment carriage"
    );
    src.configure_pipeline(&caps).unwrap();

    let mut sub = SubParse::new();
    assert!(
        matches!(
            sub.configure_pipeline(&caps).unwrap(),
            ConfigureOutcome::Accepted
        ),
        "the parser takes the source's WebVTT caps"
    );
    let mut sink = SubParseSink {
        sub,
        cues: CueSink::default(),
    };
    src.run(&mut sink).await.unwrap();

    let cues = &sink.cues.cues;
    assert_eq!(
        sink.cues.caps,
        Some(Caps::Text {
            format: TextFormat::Utf8
        })
    );
    assert_eq!(cues.len(), 2, "two cues, the empty gap sample dropped");
    assert_eq!(cues[0].0, 1_000_000_000, "tfdt-based cue onset");
    assert_eq!(
        cues[0].1, 2_000_000_000,
        "sample duration is the cue window"
    );
    assert_eq!(cues[0].2, "Hello\nthere", "both payl payloads, de-framed");
    assert_eq!(
        cues[1].0, 5_000_000_000,
        "second fragment keeps its own time"
    );
    assert_eq!(cues[1].1, 1_500_000_000);
    assert_eq!(cues[1].2, "World");
}

#[tokio::test]
async fn each_vtt_segments_timestamp_map_rebases_its_own_cues() {
    // The raw `.vtt` carriage: two segments, each with its own X-TIMESTAMP-MAP
    // (written in the two field orders that occur in the wild). Segment 0 maps
    // cue time 0 to MPEGTS 900000 (10s), segment 1 maps cue time 0 to 1800000
    // (20s), so the cues land 10s and 20s onto the variant's media timeline even
    // though both segments number their cues from zero.
    let seg0 = "WEBVTT\n\
        X-TIMESTAMP-MAP=LOCAL:00:00:00.000,MPEGTS:900000\n\n\
        00:00:01.000 --> 00:00:03.000\nfirst segment\n"
        .as_bytes()
        .to_vec();
    let seg1 = "WEBVTT\n\
        X-TIMESTAMP-MAP=MPEGTS:1800000,LOCAL:00:00:00.000\n\n\
        00:00:01.000 --> 00:00:02.000\nsecond segment\n"
        .as_bytes()
        .to_vec();
    let url = serve_playlist(VTT_PLAYLIST, Vec::new(), seg0, seg1);

    let mut src = HlsSrc::new(url).with_text();
    let caps = src.intercept_caps().await.unwrap();
    src.configure_pipeline(&caps).unwrap();
    let mut sub = SubParse::new();
    sub.configure_pipeline(&caps).unwrap();
    let mut sink = SubParseSink {
        sub,
        cues: CueSink::default(),
    };
    src.run(&mut sink).await.unwrap();

    let cues = &sink.cues.cues;
    assert_eq!(cues.len(), 2);
    assert_eq!(cues[0].2, "first segment");
    assert_eq!(cues[0].0, 11_000_000_000, "1s cue + the segment's 10s map");
    assert_eq!(cues[1].2, "second segment");
    assert_eq!(cues[1].0, 21_000_000_000, "1s cue + the segment's 20s map");
}

#[tokio::test]
async fn a_malformed_fmp4_subtitle_segment_is_dropped_not_forwarded() {
    // A segment that sniffs as ISO-BMFF but whose fragment does not parse must
    // yield no cues (and no binary in the text stream) rather than failing or
    // panicking; the following well-formed segment still plays.
    let init = init_segment(1, 1000);
    let mut bad = mp4_box(b"styp", b"iso6");
    bad.extend_from_slice(&mp4_box(b"moof", b"\0\0\0\x08junk"));
    let good = fragment(1, 4000, &[(wvtt_sample("survivor"), 1000)]);
    let url = serve(init, bad, good);

    let mut src = HlsSrc::new(url).with_text();
    let caps = src.intercept_caps().await.unwrap();
    src.configure_pipeline(&caps).unwrap();
    let mut sub = SubParse::new();
    sub.configure_pipeline(&caps).unwrap();
    let mut sink = SubParseSink {
        sub,
        cues: CueSink::default(),
    };
    src.run(&mut sink).await.unwrap();

    assert_eq!(
        sink.cues.cues.len(),
        1,
        "only the well-formed segment's cue"
    );
    assert_eq!(sink.cues.cues[0].2, "survivor");
    assert_eq!(sink.cues.cues[0].0, 4_000_000_000);
}
