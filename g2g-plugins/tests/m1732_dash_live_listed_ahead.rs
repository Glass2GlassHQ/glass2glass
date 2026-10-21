#![cfg(feature = "dash")]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use g2g_core::runtime::SourceLoop;
use g2g_core::{ByteStreamEncoding, Caps};
use g2g_plugins::dashsrc::DashSrc;
use g2g_plugins::mpd::parse;

#[path = "common/collect_sink.rs"]
mod collect_sink;
#[path = "common/dash_live_server.rs"]
mod dash_live_server;
#[path = "common/http_routes.rs"]
mod http_routes;
use collect_sink::CollectSink;
use dash_live_server::{now_unix, xs_datetime};
use http_routes::serve;

const MANIFEST_PATH: &str = "/manifest.mpd";
const SEGMENT_MILLIS: u64 = 1000;
const LISTED_SEGMENTS: u64 = 20;
const RELOAD_INTERVAL_MS: u64 = 20;
// the presentation's age when the run starts
const PRESENTATION_AGE_SECS: f64 = 2.5;
const RUN: Duration = Duration::from_secs(3);
// the availabilityStartTime is written to the millisecond
const CLOCK_ROUNDING_TOLERANCE_SECS: f64 = 0.002;

// a `$Time$` timeline that repeats without end
fn open_ended_timeline() -> String {
    format!(
        "<SegmentTemplate initialization=\"init.mp4\" media=\"t$Time$.m4s\" timescale=\"1000\">\
           <SegmentTimeline><S t=\"0\" d=\"{SEGMENT_MILLIS}\" r=\"-1\"/></SegmentTimeline>\
         </SegmentTemplate>"
    )
}

// a list naming far more segments than exist when the run starts
fn segment_list_listed_ahead() -> String {
    let entries: String = (0..LISTED_SEGMENTS)
        .map(|index| format!("<SegmentURL media=\"l{}.m4s\"/>", index * SEGMENT_MILLIS))
        .collect();
    format!(
        "<SegmentList duration=\"{SEGMENT_MILLIS}\" timescale=\"1000\">\
           <Initialization sourceURL=\"init.mp4\"/>{entries}\
         </SegmentList>"
    )
}

fn live_mpd(availability_start_unix: f64, addressing: &str) -> String {
    format!(
        "<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\" type=\"dynamic\" \
              availabilityStartTime=\"{}\" timeShiftBufferDepth=\"PT30S\" \
              suggestedPresentationDelay=\"PT1S\" minimumUpdatePeriod=\"PT1S\">\
           <Period id=\"p\" start=\"PT0S\">\
             <AdaptationSet contentType=\"video\" mimeType=\"video/mp4\">{addressing}\
               <Representation id=\"v\" codecs=\"avc1.64001e\" bandwidth=\"1000\"/>\
             </AdaptationSet>\
           </Period>\
         </MPD>",
        xs_datetime(availability_start_unix)
    )
}

// the unix second a segment path becomes available: availabilityStartTime plus the
// segment's end, read back from the start time the path names
fn availability_unix(availability_start_unix: f64, path: &str) -> Option<f64> {
    let start_millis: u64 = path
        .strip_prefix("/t")
        .or_else(|| path.strip_prefix("/l"))?
        .strip_suffix(".m4s")?
        .parse()
        .ok()?;
    Some(availability_start_unix + (start_millis + SEGMENT_MILLIS) as f64 / 1000.0)
}

// plays a live MPD whose `addressing` lists segments before they exist and checks
// that each is requested only once available, and that new ones keep being played
async fn assert_segments_wait_for_availability(addressing: String) {
    let run_start_unix = now_unix();
    let availability_start_unix = parse(&live_mpd(
        run_start_unix - PRESENTATION_AGE_SECS,
        &addressing,
    ))
    .unwrap()
    .availability_start_unix
    .unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&requests);
    let origin = serve(move |path| {
        seen.lock().unwrap().push((String::from(path), now_unix()));
        if path == MANIFEST_PATH {
            return Some(live_mpd(availability_start_unix, &addressing).into_bytes());
        }
        Some(path.as_bytes().to_vec())
    });
    let mut src = DashSrc::new(format!("{origin}{MANIFEST_PATH}"))
        .with_reload_interval_ms(RELOAD_INTERVAL_MS);
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .unwrap();
    let mut sink = CollectSink::default();
    let run = tokio::time::timeout(RUN, src.run(&mut sink)).await;
    assert!(
        run.is_err(),
        "a live presentation plays until stopped: {run:?}"
    );

    let requests = requests.lock().unwrap().clone();
    let segments: Vec<(String, f64, f64)> = requests
        .iter()
        .filter_map(|(path, at)| {
            availability_unix(availability_start_unix, path).map(|ready| (path.clone(), *at, ready))
        })
        .collect();
    let early: Vec<_> = segments
        .iter()
        .filter(|(_, at, ready)| at + CLOCK_ROUNDING_TOLERANCE_SECS < *ready)
        .collect();
    assert!(early.is_empty(), "requested before available: {early:?}");
    let published_during_run = segments
        .iter()
        .filter(|(_, _, ready)| *ready > run_start_unix)
        .count();
    assert!(
        published_during_run > 0,
        "segments published during the run are played: {requests:?}"
    );
}

#[tokio::test]
async fn an_open_ended_timeline_is_played_as_its_segments_appear() {
    assert_segments_wait_for_availability(open_ended_timeline()).await;
}

#[tokio::test]
async fn a_segment_list_listed_ahead_is_held_back() {
    assert_segments_wait_for_availability(segment_list_listed_ahead()).await;
}
