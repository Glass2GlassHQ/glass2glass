#![cfg(feature = "dash")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use g2g_core::runtime::SourceLoop;
use g2g_core::{ByteStreamEncoding, Caps};
use g2g_plugins::dashsrc::DashSrc;

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

const SEGMENT_SECS: u32 = 2;
const SECOND_PERIOD_START_SECS: u32 = 4;
const SECOND_PERIOD_SECS: u32 = 4;

// a livesim2-shaped Period: `@duration` segments numbered on from the previous
// Period, named after the Period so a request shows which one it addressed
fn period(id: &str, start_secs: u32, start_number: u32, duration_secs: Option<u32>) -> String {
    let duration = duration_secs.map_or(String::new(), |d| format!(" duration=\"PT{d}S\""));
    format!(
        "<Period id=\"{id}\" start=\"PT{start_secs}S\"{duration}>\
           <AdaptationSet contentType=\"video\" mimeType=\"video/mp4\">\
             <SegmentTemplate duration=\"{SEGMENT_SECS}\" startNumber=\"{start_number}\" \
                initialization=\"{id}-init.mp4\" media=\"{id}-$Number$.m4s\"/>\
             <Representation id=\"v\" codecs=\"avc1.64001e\" bandwidth=\"1000\"/>\
           </AdaptationSet>\
         </Period>"
    )
}

fn first_period(duration_secs: Option<u32>) -> String {
    period("a", 0, 1, duration_secs)
}

fn second_period(duration_secs: Option<u32>) -> String {
    period(
        "b",
        SECOND_PERIOD_START_SECS,
        1 + SECOND_PERIOD_START_SECS / SEGMENT_SECS,
        duration_secs,
    )
}

fn live_mpd(age_secs: f64, presentation_delay_secs: u32, periods: &[String]) -> String {
    format!(
        "<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\" type=\"dynamic\" \
              availabilityStartTime=\"{}\" timeShiftBufferDepth=\"PT30S\" \
              suggestedPresentationDelay=\"PT{presentation_delay_secs}S\" minimumUpdatePeriod=\"PT1S\">{}</MPD>",
        xs_datetime(now_unix() - age_secs),
        periods.concat()
    )
}

fn ended_mpd(periods: &[String]) -> String {
    format!(
        "<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\" type=\"static\" \
              mediaPresentationDuration=\"PT{}S\">{}</MPD>",
        SECOND_PERIOD_START_SECS + SECOND_PERIOD_SECS,
        periods.concat()
    )
}

// the paths `dashsrc` requested while it played `manifests` in turn, the last
// one repeating
async fn requested_paths(manifests: Vec<String>) -> Vec<String> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&requests);
    let loads = AtomicUsize::new(0);
    let origin = serve(move |path| {
        seen.lock().unwrap().push(String::from(path));
        if path == MANIFEST_PATH {
            let n = loads
                .fetch_add(1, Ordering::SeqCst)
                .min(manifests.len() - 1);
            return Some(manifests[n].clone().into_bytes());
        }
        Some(path.as_bytes().to_vec())
    });
    let mut src = DashSrc::new(format!("{origin}{MANIFEST_PATH}")).with_reload_interval_ms(20);
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .unwrap();
    src.run(&mut CollectSink::default()).await.unwrap();
    let paths = requests.lock().unwrap().clone();
    paths
}

fn first_period_number_past_its_end() -> String {
    format!("/a-{}.m4s", 1 + SECOND_PERIOD_START_SECS / SEGMENT_SECS)
}

fn second_period_first_number() -> String {
    format!("/b-{}.m4s", 1 + SECOND_PERIOD_START_SECS / SEGMENT_SECS)
}

// the live edge has passed the first Period's end, whose last segment is a-2:
// a-3 is media of the second Period, which must play under its own template
#[tokio::test]
async fn the_live_window_stops_at_the_period_end_and_moves_to_the_next_period() {
    let delay_secs = 3;
    let age_secs = 6.5;
    let fetched = requested_paths(vec![
        live_mpd(
            age_secs,
            delay_secs,
            &[first_period(None), second_period(None)],
        ),
        ended_mpd(&[
            first_period(Some(SECOND_PERIOD_START_SECS)),
            second_period(Some(SECOND_PERIOD_SECS)),
        ]),
    ])
    .await;
    assert!(
        fetched.iter().any(|p| p == "/a-2.m4s"),
        "the first Period plays to its last segment: {fetched:?}"
    );
    assert!(
        !fetched.contains(&first_period_number_past_its_end()),
        "nothing is requested past the first Period's end: {fetched:?}"
    );
    assert!(
        fetched.contains(&second_period_first_number()),
        "the second Period starts at its own startNumber: {fetched:?}"
    );
}

// a player joining after the second Period started begins there, as gst's
// dashdemux2 picks the Period holding now minus the presentation delay
#[tokio::test]
async fn a_live_join_starts_in_the_period_holding_the_presentation_delay() {
    let delay_secs = 2;
    let age_secs = 7.0;
    let fetched = requested_paths(vec![
        live_mpd(
            age_secs,
            delay_secs,
            &[first_period(None), second_period(None)],
        ),
        ended_mpd(&[
            first_period(Some(SECOND_PERIOD_START_SECS)),
            second_period(Some(SECOND_PERIOD_SECS)),
        ]),
    ])
    .await;
    assert!(
        !fetched.iter().any(|p| p.starts_with("/a-")),
        "the aged-out first Period is not played: {fetched:?}"
    );
    assert!(
        fetched.contains(&second_period_first_number()),
        "{fetched:?}"
    );
}
