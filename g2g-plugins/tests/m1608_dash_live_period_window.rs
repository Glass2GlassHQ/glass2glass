#![cfg(feature = "dash")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use g2g_core::element::{OutputSink, PushOutcome};
use g2g_core::runtime::SourceLoop;
use g2g_core::{ByteStreamEncoding, Caps, G2gError, PipelinePacket};
use g2g_plugins::dashsrc::DashSrc;

#[path = "common/dash_live_server.rs"]
mod dash_live_server;
use dash_live_server::{now_unix, serve, xs_datetime};

struct DiscardSink;

impl OutputSink for DiscardSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        packet_slot.take();
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

// one Period of 2 s `@duration` segments, named after the Period
fn period(id: &str, start_secs: u32, duration_secs: Option<u32>) -> String {
    let duration = duration_secs.map_or(String::new(), |d| format!(" duration=\"PT{d}S\""));
    format!(
        "<Period id=\"{id}\" start=\"PT{start_secs}S\"{duration}>\
           <AdaptationSet contentType=\"video\">\
             <Representation id=\"v\" mimeType=\"video/mp4\" codecs=\"avc1.f4000a\" bandwidth=\"1000\">\
               <SegmentTemplate duration=\"2\" startNumber=\"1\" \
                  initialization=\"{id}-init.m4s\" media=\"{id}-$Number$.m4s\"/>\
             </Representation>\
           </AdaptationSet>\
         </Period>"
    )
}

// a sliding window as livesim2 serves it: `age_secs` of the presentation exist
fn live_mpd(age_secs: f64, periods: &[String]) -> String {
    format!(
        "<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\" type=\"dynamic\" \
              availabilityStartTime=\"{}\" timeShiftBufferDepth=\"PT30S\" \
              suggestedPresentationDelay=\"PT2S\" minimumUpdatePeriod=\"PT1S\">{}</MPD>",
        xs_datetime(now_unix() - age_secs),
        periods.concat()
    )
}

fn ended_mpd(periods: &[String]) -> String {
    format!(
        "<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\" type=\"static\" \
              mediaPresentationDuration=\"PT14S\">{}</MPD>",
        periods.concat()
    )
}

// the reload that ages Period a out of the window must keep playing Period b,
// which slid from index 1 to index 0, rather than jump to Period c
#[tokio::test]
async fn a_live_reload_keeps_the_period_being_played_when_older_ones_age_out() {
    let manifests = [
        live_mpd(7.0, &[period("a", 0, None), period("b", 4, None)]),
        live_mpd(11.0, &[period("b", 4, None), period("c", 12, None)]),
        ended_mpd(&[period("b", 4, Some(8)), period("c", 12, Some(2))]),
    ];
    let requests = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&requests);
    let loads = AtomicUsize::new(0);
    let url = serve(move |path| {
        seen.lock().unwrap().push(String::from(path));
        if path == "/manifest.mpd" {
            let n = loads
                .fetch_add(1, Ordering::SeqCst)
                .min(manifests.len() - 1);
            return Some(manifests[n].clone().into_bytes());
        }
        Some(path.as_bytes().to_vec())
    });

    let mut src = DashSrc::new(url).with_reload_interval_ms(20);
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .unwrap();
    src.run(&mut DiscardSink).await.unwrap();

    let fetched = requests.lock().unwrap().clone();
    for segment in ["/b-2.m4s", "/b-3.m4s", "/c-1.m4s"] {
        assert!(
            fetched.iter().any(|p| p == segment),
            "{segment} played: {fetched:?}"
        );
    }
}
