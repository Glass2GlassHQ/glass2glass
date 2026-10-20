#![cfg(feature = "dash")]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use g2g_core::runtime::SourceLoop;
use g2g_core::{ByteStreamEncoding, Caps};
use g2g_plugins::dashsrc::DashSrc;
use g2g_plugins::mpd::parse;

#[path = "common/collect_sink.rs"]
mod collect_sink;
#[allow(dead_code)]
#[path = "common/dash_live_server.rs"]
mod dash_live_server;
#[path = "common/http_routes.rs"]
mod http_routes;
use collect_sink::CollectSink;
use dash_live_server::{now_unix, xs_datetime};
use http_routes::serve;

const MANIFEST_PATH: &str = "/manifest.mpd";
const SEGMENT_SECS: u32 = 1;
const START_NUMBER: u32 = 1;
const PRESENTATION_DELAY_SECS: u32 = 1;
const RELOAD_INTERVAL_MS: u64 = 20;
// how far the origin's clock is off the local one, far more than a reload interval
const SERVER_CLOCK_OFFSET_SECS: f64 = 2.0;
// the presentation's age on the origin's clock when the run starts
const PRESENTATION_AGE_SECS: f64 = 2.5;
const RUN: Duration = Duration::from_secs(3);
// the direct value is written to the millisecond and the request takes a few
const CLOCK_ESTIMATE_TOLERANCE_SECS: f64 = 0.05;
// a segment published during the run is requested within a few reloads
const PUBLISH_TO_REQUEST_BUDGET_SECS: f64 = 0.5;
// listed first so the test also checks that an unsupported scheme is skipped
const UNSUPPORTED_SCHEME: &str = "urn:mpeg:dash:utc:ntp:2014";
const DIRECT_SCHEME: &str = "urn:mpeg:dash:utc:direct:2014";

fn live_mpd(availability_start_unix: f64, server_now_unix: f64) -> String {
    format!(
        "<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\" type=\"dynamic\" \
              availabilityStartTime=\"{}\" timeShiftBufferDepth=\"PT30S\" \
              suggestedPresentationDelay=\"PT{PRESENTATION_DELAY_SECS}S\" \
              minimumUpdatePeriod=\"PT1S\">\
           <Period id=\"p\" start=\"PT0S\">\
             <AdaptationSet contentType=\"video\" mimeType=\"video/mp4\">\
               <SegmentTemplate duration=\"{SEGMENT_SECS}\" startNumber=\"{START_NUMBER}\" \
                  initialization=\"init.mp4\" media=\"$Number$.m4s\"/>\
               <Representation id=\"v\" codecs=\"avc1.64001e\" bandwidth=\"1000\"/>\
             </AdaptationSet>\
           </Period>\
           <UTCTiming schemeIdUri=\"{UNSUPPORTED_SCHEME}\" value=\"time.example.invalid\"/>\
           <UTCTiming schemeIdUri=\"{DIRECT_SCHEME}\" value=\"{}\"/>\
         </MPD>",
        xs_datetime(availability_start_unix),
        xs_datetime(server_now_unix)
    )
}

// the origin-clock second a segment path becomes available: availabilityStartTime
// plus the segment's end
fn availability_unix(availability_start_unix: f64, path: &str) -> Option<f64> {
    let number: u32 = path.strip_prefix('/')?.strip_suffix(".m4s")?.parse().ok()?;
    Some(availability_start_unix + f64::from((number - START_NUMBER + 1) * SEGMENT_SECS))
}

// plays a live MPD from an origin whose clock is `offset_secs` off the local one,
// with the origin time stated in a direct UTCTiming, and checks every segment
// request against the origin's clock
async fn assert_requests_follow_the_server_clock(offset_secs: f64) {
    let server_now = move || now_unix() + offset_secs;
    let run_start_server_unix = server_now();
    let availability_start_unix = parse(&live_mpd(
        run_start_server_unix - PRESENTATION_AGE_SECS,
        run_start_server_unix,
    ))
    .unwrap()
    .availability_start_unix
    .unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&requests);
    let origin = serve(move |path| {
        seen.lock()
            .unwrap()
            .push((String::from(path), server_now()));
        if path == MANIFEST_PATH {
            return Some(live_mpd(availability_start_unix, server_now()).into_bytes());
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
        .filter(|(_, at, ready)| at + CLOCK_ESTIMATE_TOLERANCE_SECS < *ready)
        .collect();
    assert!(early.is_empty(), "requested before available: {early:?}");
    let published_during_run: Vec<_> = segments
        .iter()
        .filter(|(_, _, ready)| *ready > run_start_server_unix)
        .collect();
    assert!(
        !published_during_run.is_empty(),
        "segments published during the run are played: {requests:?}"
    );
    let late: Vec<_> = published_during_run
        .iter()
        .filter(|(_, at, ready)| at - ready > PUBLISH_TO_REQUEST_BUDGET_SECS)
        .collect();
    assert!(
        late.is_empty(),
        "requested long after publication: {late:?}"
    );
}

#[tokio::test]
async fn a_server_clock_ahead_of_the_local_one_moves_requests_earlier() {
    assert_requests_follow_the_server_clock(SERVER_CLOCK_OFFSET_SECS).await;
}

#[tokio::test]
async fn a_server_clock_behind_the_local_one_moves_requests_later() {
    assert_requests_follow_the_server_clock(-SERVER_CLOCK_OFFSET_SECS).await;
}
