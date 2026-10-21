#![cfg(all(feature = "dash", feature = "hls"))]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use g2g_core::frame::PipelinePacket;
use g2g_core::runtime::SourceLoop;
use g2g_core::{ByteStreamEncoding, Caps, G2gError, HardwareError};
use g2g_plugins::dashsrc::DashSrc;
use g2g_plugins::hlssrc::HlsSrc;

#[path = "common/collect_sink.rs"]
mod collect_sink;
#[path = "common/dash_live_server.rs"]
mod dash_live_server;
#[path = "common/http_faults.rs"]
mod http_faults;
use collect_sink::CollectSink;
use dash_live_server::{now_unix, xs_datetime};
use http_faults::{serve_answers, Answer};

const NOT_FOUND: u16 = 404;
const TOO_EARLY: u16 = 425;
const SERVICE_UNAVAILABLE: u16 = 503;
// the first attempt plus gst adaptivedemux2's three retries
const ATTEMPTS_BEFORE_FAILING: usize = 4;
const INIT_PATH: &str = "/init.mp4";
const INIT_BODY: &[u8] = b"init-segment";
// the answer segment `i` gets on its first request, before its body
const FIRST_ANSWERS: [Answer; 3] = [
    Answer::Status(TOO_EARLY),
    Answer::Status(SERVICE_UNAVAILABLE),
    Answer::Drop,
];

const DASH_MANIFEST_PATH: &str = "/manifest.mpd";
const DASH_VOD_MPD: &str = "<MPD type=\"static\" mediaPresentationDuration=\"PT3S\">\
       <Period><AdaptationSet mimeType=\"video/mp4\">\
         <SegmentTemplate initialization=\"init.mp4\" media=\"seg$Number$.m4s\" \
            startNumber=\"0\" duration=\"1\" timescale=\"1\"/>\
         <Representation id=\"v\" bandwidth=\"1\"/>\
       </AdaptationSet></Period>\
     </MPD>";

const HLS_PLAYLIST_PATH: &str = "/index.m3u8";
const HLS_VOD_PLAYLIST: &str = "#EXTM3U\n\
    #EXT-X-TARGETDURATION:1\n\
    #EXT-X-MEDIA-SEQUENCE:0\n\
    #EXTINF:1.0,\n\
    seg0.ts\n\
    #EXTINF:1.0,\n\
    seg1.ts\n\
    #EXTINF:1.0,\n\
    seg2.ts\n\
    #EXT-X-ENDLIST\n";

type RequestCounts = Arc<Mutex<HashMap<String, usize>>>;

fn segment_body(index: usize) -> Vec<u8> {
    format!("segment-{index}-body").into_bytes()
}

fn segment_path(index: usize, extension: &str) -> String {
    format!("/seg{index}.{extension}")
}

fn segment_index(path: &str, extension: &str) -> Option<usize> {
    path.strip_prefix("/seg")?
        .strip_suffix(extension)?
        .strip_suffix('.')?
        .parse()
        .ok()
}

// the number of requests `path` got before this one, counting this one
fn count_request(counts: &RequestCounts, path: &str) -> usize {
    let mut counts = counts.lock().unwrap();
    let seen = counts.entry(String::from(path)).or_default();
    *seen += 1;
    *seen - 1
}

fn requests_for(counts: &RequestCounts, path: &str) -> usize {
    counts.lock().unwrap().get(path).copied().unwrap_or(0)
}

// `manifest` at `manifest_path`, then the init and segments that `segment_answer(index, attempt)` picks
fn serve_segments(
    manifest_path: &'static str,
    manifest: &'static str,
    extension: &'static str,
    segment_answer: fn(usize, usize) -> Answer,
) -> (String, RequestCounts) {
    let counts = RequestCounts::default();
    let seen = Arc::clone(&counts);
    let origin = serve_answers(move |path| {
        let attempt = count_request(&seen, path);
        if path == manifest_path {
            return Answer::Body(manifest.as_bytes().to_vec());
        }
        if path == INIT_PATH {
            return Answer::Body(INIT_BODY.to_vec());
        }
        match segment_index(path, extension) {
            Some(index) => segment_answer(index, attempt),
            None => Answer::Status(NOT_FOUND),
        }
    });
    (format!("{origin}{manifest_path}"), counts)
}

fn fails_once_then_answers(index: usize, attempt: usize) -> Answer {
    if attempt == 0 {
        FIRST_ANSWERS[index].clone()
    } else {
        Answer::Body(segment_body(index))
    }
}

fn delivered_bytes(sink: &CollectSink) -> Vec<u8> {
    let mut bytes = Vec::new();
    for packet in &sink.packets {
        if let PipelinePacket::DataFrame(frame) = packet {
            if let Some(slice) = frame.domain.as_system_slice() {
                bytes.extend_from_slice(slice);
            }
        }
    }
    bytes
}

fn all_segment_bodies() -> Vec<u8> {
    (0..FIRST_ANSWERS.len()).flat_map(segment_body).collect()
}

fn iso_bmff() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    }
}

fn mpeg_ts() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    }
}

#[tokio::test]
async fn dash_segments_survive_too_early_unavailable_and_dropped_answers() {
    let (url, counts) = serve_segments(
        DASH_MANIFEST_PATH,
        DASH_VOD_MPD,
        "m4s",
        fails_once_then_answers,
    );
    let mut src = DashSrc::new(url);
    src.configure_pipeline(&iso_bmff()).unwrap();
    let mut sink = CollectSink::default();
    src.run(&mut sink).await.unwrap();

    let mut expected = INIT_BODY.to_vec();
    expected.extend(all_segment_bodies());
    assert_eq!(delivered_bytes(&sink), expected);
    for index in 0..FIRST_ANSWERS.len() {
        let path = segment_path(index, "m4s");
        assert_eq!(requests_for(&counts, &path), 2, "{path} is retried once");
    }
}

#[tokio::test]
async fn hls_segments_survive_too_early_unavailable_and_dropped_answers() {
    let (url, counts) = serve_segments(
        HLS_PLAYLIST_PATH,
        HLS_VOD_PLAYLIST,
        "ts",
        fails_once_then_answers,
    );
    let mut src = HlsSrc::new(url);
    src.configure_pipeline(&mpeg_ts()).unwrap();
    let mut sink = CollectSink::default();
    src.run(&mut sink).await.unwrap();

    assert_eq!(delivered_bytes(&sink), all_segment_bodies());
    for index in 0..FIRST_ANSWERS.len() {
        let path = segment_path(index, "ts");
        assert_eq!(requests_for(&counts, &path), 2, "{path} is retried once");
    }
}

#[tokio::test]
async fn a_segment_failing_past_the_retry_budget_ends_the_run_with_its_status() {
    let (url, counts) = serve_segments(DASH_MANIFEST_PATH, DASH_VOD_MPD, "m4s", |_, _| {
        Answer::Status(SERVICE_UNAVAILABLE)
    });
    let mut src = DashSrc::new(url);
    src.configure_pipeline(&iso_bmff()).unwrap();
    let error = src
        .run(&mut CollectSink::default())
        .await
        .expect_err("every segment request fails");

    assert_eq!(
        error,
        G2gError::Hardware(HardwareError::Http(SERVICE_UNAVAILABLE))
    );
    assert_eq!(
        requests_for(&counts, &segment_path(0, "m4s")),
        ATTEMPTS_BEFORE_FAILING
    );
    assert_eq!(requests_for(&counts, &segment_path(1, "m4s")), 0);
}

const HLS_LIVE_PLAYLIST: &str = "#EXTM3U\n\
    #EXT-X-TARGETDURATION:1\n\
    #EXT-X-MEDIA-SEQUENCE:0\n\
    #EXTINF:1.0,\n\
    seg0.ts\n";
const HLS_ENDED_PLAYLIST: &str = "#EXTM3U\n\
    #EXT-X-TARGETDURATION:1\n\
    #EXT-X-MEDIA-SEQUENCE:0\n\
    #EXTINF:1.0,\n\
    seg0.ts\n\
    #EXTINF:1.0,\n\
    seg1.ts\n\
    #EXT-X-ENDLIST\n";
// what the first reloads get while the origin restarts
const RESTARTING_ANSWERS: [Answer; 2] = [Answer::Status(SERVICE_UNAVAILABLE), Answer::Drop];
const RELOAD_INTERVAL_MS: u64 = 20;

#[tokio::test]
async fn an_hls_live_reload_survives_an_origin_restart() {
    let counts = RequestCounts::default();
    let seen = Arc::clone(&counts);
    let origin = serve_answers(move |path| {
        let attempt = count_request(&seen, path);
        if path == HLS_PLAYLIST_PATH {
            return match attempt {
                0 => Answer::Body(HLS_LIVE_PLAYLIST.as_bytes().to_vec()),
                reload if reload <= RESTARTING_ANSWERS.len() => {
                    RESTARTING_ANSWERS[reload - 1].clone()
                }
                _ => Answer::Body(HLS_ENDED_PLAYLIST.as_bytes().to_vec()),
            };
        }
        match segment_index(path, "ts") {
            Some(index) => Answer::Body(segment_body(index)),
            None => Answer::Status(NOT_FOUND),
        }
    });
    let mut src = HlsSrc::new(format!("{origin}{HLS_PLAYLIST_PATH}"))
        .with_reload_interval_ms(RELOAD_INTERVAL_MS);
    src.configure_pipeline(&mpeg_ts()).unwrap();
    let mut sink = CollectSink::default();
    src.run(&mut sink).await.unwrap();

    let mut expected = segment_body(0);
    expected.extend(segment_body(1));
    assert_eq!(delivered_bytes(&sink), expected);
    assert_eq!(
        requests_for(&counts, HLS_PLAYLIST_PATH),
        RESTARTING_ANSWERS.len() + 2,
        "the startup fetch, the failed reloads and the one that answered"
    );
}

const LIVE_SEGMENT_SECS: u32 = 4;
// two segments are complete when the run starts, the third appears after it ends
const LIVE_PRESENTATION_AGE_SECS: f64 = 8.2;
// the newest segment, the one an origin with a slower clock has not published yet
const NEWEST_INDEX: usize = 1;
const LIVE_RUN: Duration = Duration::from_secs(3);
// more too-early answers than the budget allows, all inside one segment duration
const TOO_EARLY_ANSWERS: usize = ATTEMPTS_BEFORE_FAILING + 1;

fn live_mpd(availability_start_unix: f64) -> String {
    format!(
        "<MPD type=\"dynamic\" availabilityStartTime=\"{}\" timeShiftBufferDepth=\"PT30S\" \
              minimumUpdatePeriod=\"PT1S\">\
           <Period id=\"p\" start=\"PT0S\"><AdaptationSet mimeType=\"video/mp4\">\
             <SegmentTemplate initialization=\"init.mp4\" media=\"seg$Number$.m4s\" \
                startNumber=\"0\" duration=\"{LIVE_SEGMENT_SECS}\" timescale=\"1\"/>\
             <Representation id=\"v\" bandwidth=\"1\"/>\
           </AdaptationSet></Period>\
         </MPD>",
        xs_datetime(availability_start_unix)
    )
}

#[tokio::test]
async fn a_live_dash_segment_answering_too_early_is_waited_for() {
    let manifest = live_mpd(now_unix() - LIVE_PRESENTATION_AGE_SECS);
    let counts = RequestCounts::default();
    let seen = Arc::clone(&counts);
    let origin = serve_answers(move |path| {
        let attempt = count_request(&seen, path);
        if path == DASH_MANIFEST_PATH {
            return Answer::Body(manifest.clone().into_bytes());
        }
        if path == INIT_PATH {
            return Answer::Body(INIT_BODY.to_vec());
        }
        match segment_index(path, "m4s") {
            Some(NEWEST_INDEX) if attempt < TOO_EARLY_ANSWERS => Answer::Status(TOO_EARLY),
            Some(index) => Answer::Body(segment_body(index)),
            None => Answer::Status(NOT_FOUND),
        }
    });
    let mut src = DashSrc::new(format!("{origin}{DASH_MANIFEST_PATH}"))
        .with_reload_interval_ms(RELOAD_INTERVAL_MS);
    src.configure_pipeline(&iso_bmff()).unwrap();
    let mut sink = CollectSink::default();
    let run = tokio::time::timeout(LIVE_RUN, src.run(&mut sink)).await;

    assert!(
        run.is_err(),
        "a live presentation plays until stopped: {run:?}"
    );
    let newest_path = segment_path(NEWEST_INDEX, "m4s");
    assert_eq!(requests_for(&counts, &newest_path), TOO_EARLY_ANSWERS + 1);
    let mut expected = INIT_BODY.to_vec();
    expected.extend((0..=NEWEST_INDEX).flat_map(segment_body));
    assert_eq!(delivered_bytes(&sink), expected);
}
