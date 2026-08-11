//! M836 DASH live + multi-period: the wall-clock `@duration` live profile
//! (`type="dynamic"`, `availabilityStartTime`, `timeShiftBufferDepth`,
//! `suggestedPresentationDelay`), live-edge start for a `SegmentTimeline`
//! window, and consecutive `Period`s playing through with a boundary `Segment`.
//!
//! The live manifests are the shape ffmpeg's dash muxer writes mid-stream
//! (`-f dash -streaming 1 -window_size N -use_template 1 -use_timeline 0`), with
//! `availabilityStartTime` rewritten relative to "now" so the wall-clock math is
//! exercised against a known window. Segment durations are 2s so a test running
//! within a second of the manifest it was served cannot straddle a segment
//! boundary.

#![cfg(feature = "dash")]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    ByteStreamEncoding, Caps, Dim, G2gError, MemoryDomain, OutputSink, PipelinePacket, PropValue,
    PushOutcome, Rate, Segment, VideoCodec,
};
use g2g_plugins::dashsrc::DashSrc;
use g2g_plugins::fmp4demux::Fmp4Demux;
use g2g_plugins::mp4mux::Mp4Mux;

#[derive(Default)]
struct CaptureSink {
    body: Vec<u8>,
    aus: Vec<Vec<u8>>,
    segments: Vec<Segment>,
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
                        self.aus.push(s.to_vec());
                    }
                }
                PipelinePacket::Segment(seg) => self.segments.push(seg),
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn now_unix() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

/// Format `unix` seconds as an `xs:dateTime` in UTC, the `availabilityStartTime`
/// form ffmpeg writes, milliseconds included. The inverse of the parser's
/// `days_from_civil` (Howard Hinnant's `civil_from_days`). Keeping the
/// sub-second part matters: truncating it would move the anchor up to a second
/// earlier and could tip the window onto the next segment.
fn xs_datetime(unix: f64) -> String {
    let secs = unix.floor() as i64;
    let millis = ((unix - secs as f64) * 1000.0) as i64;
    let (days, tod) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{millis:03}Z",
        tod / 3600,
        (tod / 60) % 60,
        tod % 60
    )
}

/// A one-connection-per-request HTTP server over `route`, which maps a request
/// path to a body (`None` = 404). Returns the manifest URL.
fn serve(route: impl Fn(&str) -> Option<Vec<u8>> + Send + 'static) -> String {
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
            let Some(body) = route(path) else {
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
    format!("http://127.0.0.1:{port}/manifest.mpd")
}

/// The body served for media segment `$Number$` (distinct per number, so the
/// delivered byte stream identifies exactly which segments were fetched).
fn media_body(number: usize) -> Vec<u8> {
    format!("media-{number:05}-").repeat(4).into_bytes()
}

const INIT_BODY: &[u8] = b"init-stream0-payload";

/// A dynamic `@duration` MPD in ffmpeg's `-use_template 1 -use_timeline 0`
/// shape, with 2s segments and `availabilityStartTime` `age` seconds in the
/// past. `spd` is the optional `suggestedPresentationDelay` in seconds, `ato` the
/// template's `@availabilityTimeOffset` (seconds, or `"INF"`).
fn live_mpd(age_secs: f64, tsb_secs: f64, spd: Option<f64>, ato: Option<&str>) -> String {
    let ast = xs_datetime(now_unix() - age_secs);
    let spd = spd.map_or(String::new(), |s| {
        format!(" suggestedPresentationDelay=\"PT{s}S\"")
    });
    let ato = ato.map_or(String::new(), |v| {
        format!(" availabilityTimeOffset=\"{v}\"")
    });
    format!(
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n\
         <MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\" \
              profiles=\"urn:mpeg:dash:profile:isoff-live:2011\" type=\"dynamic\" \
              minimumUpdatePeriod=\"PT500S\"{spd} availabilityStartTime=\"{ast}\" \
              timeShiftBufferDepth=\"PT{tsb_secs}S\" maxSegmentDuration=\"PT2.0S\" \
              minBufferTime=\"PT4.0S\">\n\
           <Period id=\"0\" start=\"PT0.0S\">\n\
             <AdaptationSet id=\"0\" contentType=\"video\">\n\
               <Representation id=\"0\" mimeType=\"video/mp4\" codecs=\"avc1.f4000a\" \
                  bandwidth=\"47600\" width=\"64\" height=\"48\">\n\
                 <SegmentTemplate timescale=\"1000000\" duration=\"2000000\"{ato} \
                    initialization=\"init-stream$RepresentationID$.m4s\" \
                    media=\"chunk-stream$RepresentationID$-$Number%05d$.m4s\" startNumber=\"1\"/>\n\
               </Representation>\n\
             </AdaptationSet>\n\
           </Period>\n\
         </MPD>"
    )
}

/// The same presentation turned `static` over `total_secs`, which is how the
/// run ends: the already-played segments dedup away and the source goes to EOS.
fn ended_mpd(total_secs: f64) -> String {
    format!(
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n\
         <MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\" type=\"static\" \
              mediaPresentationDuration=\"PT{total_secs}S\">\n\
           <Period id=\"0\" start=\"PT0.0S\">\n\
             <AdaptationSet id=\"0\" contentType=\"video\">\n\
               <Representation id=\"0\" mimeType=\"video/mp4\" codecs=\"avc1.f4000a\" \
                  bandwidth=\"47600\" width=\"64\" height=\"48\">\n\
                 <SegmentTemplate timescale=\"1000000\" duration=\"2000000\" \
                    initialization=\"init-stream$RepresentationID$.m4s\" \
                    media=\"chunk-stream$RepresentationID$-$Number%05d$.m4s\" startNumber=\"1\"/>\n\
               </Representation>\n\
             </AdaptationSet>\n\
           </Period>\n\
         </MPD>"
    )
}

/// Serve the live manifests in `manifests` (one per manifest request, the last
/// repeating) plus the init / media segments, recording every requested path.
fn serve_live(manifests: Vec<String>) -> (String, Arc<Mutex<Vec<String>>>) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&requests);
    let loads = AtomicUsize::new(0);
    let url = serve(move |path| {
        seen.lock().unwrap().push(String::from(path));
        if path == "/manifest.mpd" {
            let n = loads.fetch_add(1, Ordering::SeqCst);
            return Some(manifests[n.min(manifests.len() - 1)].clone().into_bytes());
        }
        if path == "/init-stream0.m4s" {
            return Some(INIT_BODY.to_vec());
        }
        path.strip_prefix("/chunk-stream0-")
            .and_then(|s| s.strip_suffix(".m4s"))
            .and_then(|s| s.parse::<usize>().ok())
            .map(media_body)
    });
    (url, requests)
}

async fn run_dash(src: &mut DashSrc) -> CaptureSink {
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    src.run(&mut sink).await.unwrap();
    sink
}

/// The expected byte stream for a live run that fetched `numbers` in order.
fn expected_stream(numbers: &[usize]) -> Vec<u8> {
    let mut out = INIT_BODY.to_vec();
    for &n in numbers {
        out.extend_from_slice(&media_body(n));
    }
    out
}

#[tokio::test]
async fn dash_live_duration_profile_starts_at_the_live_edge_and_reloads() {
    // 15s of a 2s-segment presentation exist, so segments 1..=7 (numbers) are
    // complete; the 6s time-shift depth keeps 6 and 7. A 2s presentation delay
    // puts playback on the newest one, number 7. The reload advances the wall
    // clock by 4s (two more segments), which must deliver 8 and 9 exactly once.
    let (url, requests) = serve_live(vec![
        live_mpd(15.0, 6.0, Some(2.0), None),
        live_mpd(19.0, 6.0, Some(2.0), None),
        ended_mpd(18.0),
    ]);

    let mut src = DashSrc::new(url).with_reload_interval_ms(20);
    let sink = run_dash(&mut src).await;

    assert_eq!(
        sink.body,
        expected_stream(&[7, 8, 9]),
        "starts at the live edge (7), then the reload adds 8 and 9 once each"
    );
    let fetched = requests.lock().unwrap().clone();
    assert!(
        !fetched.iter().any(|p| p.contains("chunk-stream0-00006")),
        "the stale end of the window is not replayed: {fetched:?}"
    );
}

#[tokio::test]
async fn dash_live_start_clamps_to_the_time_shift_buffer_window() {
    // No suggestedPresentationDelay, so the default 10s delay applies: 10s back
    // from the live edge is segment 3, which has long fallen out of the 6s
    // time-shift window. The start clamps to the window front (6) instead of
    // requesting media the server no longer has.
    let (url, requests) = serve_live(vec![live_mpd(15.0, 6.0, None, None), ended_mpd(14.0)]);

    let mut src = DashSrc::new(url).with_reload_interval_ms(20);
    let sink = run_dash(&mut src).await;

    assert_eq!(
        sink.body,
        expected_stream(&[6, 7]),
        "playback clamps to the earliest segment still inside timeShiftBufferDepth"
    );
    let fetched = requests.lock().unwrap().clone();
    assert!(
        // Zero-padded numbers, so a lexical comparison is a numeric one.
        !fetched
            .iter()
            .any(|p| p.starts_with("/chunk-stream0-") && p.as_str() < "/chunk-stream0-00006"),
        "nothing outside the window is requested: {fetched:?}"
    );
}

#[tokio::test]
async fn dash_live_presentation_delay_property_overrides_the_manifest() {
    // The manifest suggests 2s (number 7 only); asking for 4s starts one
    // segment earlier, at 6.
    let (url, _) = serve_live(vec![live_mpd(15.0, 6.0, Some(2.0), None), ended_mpd(14.0)]);

    // Driven through the runtime property, the way a launch line sets it.
    let mut src = DashSrc::new(url).with_reload_interval_ms(20);
    src.set_property("presentation-delay-ms", PropValue::Uint(4000))
        .unwrap();
    assert_eq!(
        src.get_property("presentation-delay-ms"),
        Some(PropValue::Uint(4000))
    );
    assert!(src
        .properties()
        .iter()
        .any(|p| p.name == "presentation-delay-ms"));
    let sink = run_dash(&mut src).await;

    assert_eq!(sink.body, expected_stream(&[6, 7]));
}

#[tokio::test]
async fn dash_live_availability_time_offset_publishes_a_segment_before_it_completes() {
    // 15.5s into a 2s-segment presentation: segment number 8 covers [14,16)s, so
    // it is still being written, but a 1s availabilityTimeOffset publishes it
    // from 15s. It is the live edge, and a 2s presentation delay starts there.
    let (url, _) = serve_live(vec![
        live_mpd(15.5, 6.0, Some(2.0), Some("1")),
        ended_mpd(14.0),
    ]);
    let mut src = DashSrc::new(url).with_reload_interval_ms(20);
    let sink = run_dash(&mut src).await;
    assert_eq!(
        sink.body,
        expected_stream(&[8]),
        "the early-available segment is the live edge"
    );

    // The same wall clock with no offset: only complete segments are available,
    // so the edge is the previous one.
    let (url, _) = serve_live(vec![live_mpd(15.5, 6.0, Some(2.0), None), ended_mpd(14.0)]);
    let mut src = DashSrc::new(url).with_reload_interval_ms(20);
    let sink = run_dash(&mut src).await;
    assert_eq!(
        sink.body,
        expected_stream(&[7]),
        "without the offset the in-progress segment is not fetched"
    );
}

#[tokio::test]
async fn dash_live_oversized_availability_time_offset_clamps_to_one_segment() {
    // An offset far past the segment duration (or the literal INF) cannot publish
    // media that does not exist: it clamps at the in-progress segment, number 8.
    for ato in ["999", "INF"] {
        let (url, requests) = serve_live(vec![
            live_mpd(15.5, 6.0, Some(2.0), Some(ato)),
            ended_mpd(14.0),
        ]);
        let mut src = DashSrc::new(url).with_reload_interval_ms(20);
        let sink = run_dash(&mut src).await;
        assert_eq!(sink.body, expected_stream(&[8]), "ato={ato}");
        let fetched = requests.lock().unwrap().clone();
        assert!(
            !fetched.iter().any(|p| p.contains("chunk-stream0-00009")),
            "nothing past the in-progress segment is requested: {fetched:?}"
        );
    }
}

/// A dynamic `SegmentTimeline` manifest: the timeline lists the available
/// window itself, so the live-edge start walks it instead of the wall clock.
fn live_timeline_mpd(segments: u64, dynamic: bool) -> String {
    let mpd_type = if dynamic { "dynamic" } else { "static" };
    format!(
        "<?xml version=\"1.0\"?>\n\
         <MPD type=\"{mpd_type}\" minimumUpdatePeriod=\"PT500S\" \
              suggestedPresentationDelay=\"PT2S\">\n\
           <Period>\n\
             <AdaptationSet mimeType=\"video/mp4\">\n\
               <SegmentTemplate initialization=\"init-stream0.m4s\" \
                  media=\"chunk-stream0-$Number%05d$.m4s\" startNumber=\"1\" timescale=\"1000\">\n\
                 <SegmentTimeline><S t=\"0\" d=\"1000\" r=\"{}\"/></SegmentTimeline>\n\
               </SegmentTemplate>\n\
               <Representation id=\"0\" bandwidth=\"47600\" width=\"64\" height=\"48\"/>\n\
             </AdaptationSet>\n\
           </Period>\n\
         </MPD>",
        segments - 1
    )
}

#[tokio::test]
async fn dash_live_segment_timeline_starts_near_the_live_edge() {
    // Six 1s segments in the window; a 2s presentation delay starts on the
    // second to last, so only 5 and 6 play.
    let (url, _) = serve_live(vec![
        live_timeline_mpd(6, true),
        live_timeline_mpd(6, false),
    ]);

    let mut src = DashSrc::new(url).with_reload_interval_ms(20);
    let sink = run_dash(&mut src).await;

    assert_eq!(
        sink.body,
        expected_stream(&[5, 6]),
        "the timeline window is entered a presentation delay from its end"
    );
}

// --- multi-period ---------------------------------------------------------

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

/// Three access units whose payload bytes are keyed by `mark`, so the two
/// Periods carry distinguishable media.
fn access_units(mark: u8) -> Vec<Vec<u8>> {
    let sps = [0x67u8, 0x42, 0xC0, 0x1E, 0x11, 0x22];
    let pps = [0x68u8, 0xCE, 0x3C, 0x80];
    let idr: Vec<u8> = [
        &[0, 0, 0, 1][..],
        &sps,
        &[0, 0, 0, 1],
        &pps,
        &[0, 0, 0, 1],
        &[0x65, 0xAA, mark],
    ]
    .concat();
    let p = |f: u8| [&[0, 0, 0, 1][..], &[0x41, f, mark]].concat();
    vec![idr, p(1), p(2)]
}

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
        mux.process(
            PipelinePacket::DataFrame(au_frame(au.clone(), i as u64 * 33_333_333, i as u64)),
            &mut out,
        )
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

/// Two Periods of three 1s segments each, the canonical hand-stitched form:
/// two independently authored assets under their own `BaseURL`s, the second
/// Period starting where the first ends.
const MULTI_PERIOD_MPD: &str = "<?xml version=\"1.0\"?>\n\
    <MPD type=\"static\" mediaPresentationDuration=\"PT6S\">\n\
      <Period id=\"a\" start=\"PT0S\" duration=\"PT3S\">\n\
        <BaseURL>a/</BaseURL>\n\
        <AdaptationSet mimeType=\"video/mp4\" codecs=\"avc1.4d401f\">\n\
          <SegmentTemplate initialization=\"init.mp4\" media=\"seg$Number$.m4s\" \
             startNumber=\"0\" duration=\"1000\" timescale=\"1000\"/>\n\
          <Representation id=\"v0\" bandwidth=\"1000000\" width=\"64\" height=\"48\"/>\n\
        </AdaptationSet>\n\
      </Period>\n\
      <Period id=\"b\" start=\"PT3S\" duration=\"PT3S\">\n\
        <BaseURL>b/</BaseURL>\n\
        <AdaptationSet mimeType=\"video/mp4\" codecs=\"avc1.4d401f\">\n\
          <SegmentTemplate initialization=\"init.mp4\" media=\"seg$Number$.m4s\" \
             startNumber=\"0\" duration=\"1000\" timescale=\"1000\"/>\n\
          <Representation id=\"v0\" bandwidth=\"1000000\" width=\"64\" height=\"48\"/>\n\
        </AdaptationSet>\n\
      </Period>\n\
    </MPD>";

#[tokio::test]
async fn dash_multi_period_plays_through_and_emits_a_boundary_segment() {
    let aus_a = access_units(0xA0);
    let aus_b = access_units(0xB0);
    let (init_a, segs_a) = split_fmp4(&make_fmp4(&aus_a).await);
    let (init_b, segs_b) = split_fmp4(&make_fmp4(&aus_b).await);
    assert_eq!(segs_a.len(), 3);

    let (ia, sa, ib, sb) = (
        init_a.clone(),
        segs_a.clone(),
        init_b.clone(),
        segs_b.clone(),
    );
    let url = serve(move |path| {
        if path == "/manifest.mpd" {
            return Some(MULTI_PERIOD_MPD.as_bytes().to_vec());
        }
        let (rest, init, segs) = match path.strip_prefix("/a/") {
            Some(rest) => (rest, &ia, &sa),
            None => (path.strip_prefix("/b/")?, &ib, &sb),
        };
        if rest == "init.mp4" {
            return Some(init.clone());
        }
        rest.strip_prefix("seg")
            .and_then(|s| s.strip_suffix(".m4s"))
            .and_then(|s| s.parse::<usize>().ok())
            .and_then(|i| segs.get(i).cloned())
    });

    let mut src = DashSrc::new(url);
    let sink = run_dash(&mut src).await;

    // Both Periods delivered back to back, each with its own init, from its own
    // BaseURL.
    let mut expected = init_a.clone();
    for s in &segs_a {
        expected.extend_from_slice(s);
    }
    expected.extend_from_slice(&init_b);
    for s in &segs_b {
        expected.extend_from_slice(s);
    }
    assert_eq!(sink.body, expected, "period a then period b, in order");

    // One boundary Segment, before the second Period's media: running time
    // continues where period a ended (3s), stream time restarts at 0.
    assert_eq!(sink.segments.len(), 1, "one Segment, at the boundary");
    let seg = sink.segments[0];
    assert_eq!(
        seg.base, 3_000_000_000,
        "running time continues past period a"
    );
    assert_eq!(seg.time, 0, "stream time restarts with period b");
    assert_eq!(seg.start, 0, "period b's own timeline starts at 0");
    assert_eq!(
        seg.to_running_time(0),
        Some(3_000_000_000),
        "period b's first sample lands at 3s of running time"
    );
    assert_eq!(
        seg.to_stream_time(1_000_000_000),
        Some(1_000_000_000),
        "1s into period b is 1s of stream time"
    );

    // Sample-exact: the delivered stream demuxes back to both Periods' access
    // units, in order.
    let mut dmx = Fmp4Demux::new();
    dmx.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .unwrap();
    let mut dsink = CaptureSink::default();
    dmx.process(
        PipelinePacket::DataFrame(au_frame(sink.body.clone(), 0, 0)),
        &mut dsink,
    )
    .await
    .unwrap();
    let expected_aus: Vec<Vec<u8>> = aus_a.iter().chain(aus_b.iter()).cloned().collect();
    assert_eq!(
        dsink.aus, expected_aus,
        "multi-period DashSrc -> Fmp4Demux recovers every access unit of both periods"
    );
}
