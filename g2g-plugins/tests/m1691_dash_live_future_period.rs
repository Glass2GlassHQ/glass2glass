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
const PRESENTATION_DELAY_SECS: u32 = 1;
const RELOAD_INTERVAL_MS: u64 = 20;

// livesim2's shape: `@duration` segments numbered on across Periods, and the
// Period after the playing one listed before it starts
#[derive(Debug, Clone, Copy)]
struct FixturePeriod {
    id: &'static str,
    start_secs: u32,
}

const PERIODS: [FixturePeriod; 3] = [
    FixturePeriod {
        id: "a",
        start_secs: 0,
    },
    FixturePeriod {
        id: "b",
        start_secs: 2,
    },
    FixturePeriod {
        id: "c",
        start_secs: 4,
    },
];
// the first Period has ended and the second has run for less than one segment
const PRESENTATION_AGE_SECS: f64 = 2.3;
// long enough for both segments of the second Period to become available
const RUN: Duration = Duration::from_secs(3);

impl FixturePeriod {
    fn start_number(self) -> u32 {
        1 + self.start_secs / SEGMENT_SECS
    }

    fn xml(self) -> String {
        format!(
            "<Period id=\"{}\" start=\"PT{}S\">\
               <AdaptationSet contentType=\"video\" mimeType=\"video/mp4\">\
                 <SegmentTemplate duration=\"{SEGMENT_SECS}\" startNumber=\"{}\" \
                    initialization=\"{}-init.mp4\" media=\"{}-$Number$.m4s\"/>\
                 <Representation id=\"v\" codecs=\"avc1.64001e\" bandwidth=\"1000\"/>\
               </AdaptationSet>\
             </Period>",
            self.id,
            self.start_secs,
            self.start_number(),
            self.id,
            self.id
        )
    }
}

fn live_mpd(availability_start_unix: f64) -> String {
    let periods: String = PERIODS.iter().map(|p| p.xml()).collect();
    format!(
        "<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\" type=\"dynamic\" \
              availabilityStartTime=\"{}\" timeShiftBufferDepth=\"PT30S\" \
              suggestedPresentationDelay=\"PT{PRESENTATION_DELAY_SECS}S\" \
              minimumUpdatePeriod=\"PT1S\">{periods}</MPD>",
        xs_datetime(availability_start_unix)
    )
}

// the unix second a media segment's path becomes available: availabilityStartTime
// plus the Period start plus the segment's end, as gst's dashdemux waits for
fn availability_unix(availability_start_unix: f64, path: &str) -> Option<f64> {
    let (id, number) = path
        .strip_prefix('/')?
        .strip_suffix(".m4s")?
        .split_once('-')?;
    let period = PERIODS.iter().find(|p| p.id == id)?;
    let index = number.parse::<u32>().ok()? - period.start_number();
    Some(availability_start_unix + f64::from(period.start_secs + (index + 1) * SEGMENT_SECS))
}

// a live Period listed before it starts is entered at its start, and each of
// its segments is requested only once it is available
#[tokio::test]
async fn a_period_listed_ahead_is_fetched_only_as_its_segments_become_available() {
    let manifest = live_mpd(now_unix() - PRESENTATION_AGE_SECS);
    // the manifest's millisecond value, the one dashsrc reads
    let availability_start_unix = parse(&manifest).unwrap().availability_start_unix.unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&requests);
    let origin = serve(move |path| {
        seen.lock().unwrap().push((String::from(path), now_unix()));
        if path == MANIFEST_PATH {
            return Some(manifest.clone().into_bytes());
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
        .filter(|(_, at, ready)| at < ready)
        .collect();
    assert!(early.is_empty(), "requested before available: {early:?}");
    let second = PERIODS[1];
    let second_last = format!("/{}-{}.m4s", second.id, PERIODS[2].start_number() - 1);
    assert!(
        segments.iter().any(|(path, _, _)| *path == second_last),
        "the second Period plays once it starts: {requests:?}"
    );
}
