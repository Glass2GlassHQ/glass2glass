#![cfg(feature = "dash")]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use g2g_core::element::{OutputSink, PushOutcome};
use g2g_core::runtime::SourceLoop;
use g2g_core::{ByteStreamEncoding, Caps, G2gError, PipelinePacket};
use g2g_plugins::dashsrc::DashSrc;

#[path = "common/dash_live_server.rs"]
mod dash_live_server;
use dash_live_server::{now_unix, xs_datetime};
#[path = "common/http_routes.rs"]
mod http_routes;
use http_routes::serve;

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

// 1 s segments numbered from 0, with half a second of margin from every window boundary
const AGE_SECS: f64 = 20.5;
const TIME_SHIFT_SECS: u32 = 4;
// the high segment the run starts on takes this long, so the window slides two segments before the switch
const SLOW_HIGH: Duration = Duration::from_secs(2);

fn live_mpd(anchor_unix: f64) -> String {
    format!(
        "<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\" type=\"dynamic\" \
              availabilityStartTime=\"{}\" timeShiftBufferDepth=\"PT{TIME_SHIFT_SECS}S\" \
              suggestedPresentationDelay=\"PT2S\" minimumUpdatePeriod=\"PT30S\">\
           <Period id=\"p\" start=\"PT0S\">\
             <AdaptationSet contentType=\"video\" mimeType=\"video/mp4\" codecs=\"avc1.f4000a\">\
               <SegmentTemplate duration=\"1000\" timescale=\"1000\" startNumber=\"0\" \
                  initialization=\"init-$RepresentationID$.m4s\" media=\"$RepresentationID$-$Number$.m4s\"/>\
               <Representation id=\"high\" bandwidth=\"50000000\"/>\
               <Representation id=\"low\" bandwidth=\"500000\"/>\
             </AdaptationSet>\
           </Period>\
         </MPD>",
        xs_datetime(anchor_unix)
    )
}

// the segment number of a media request, `None` for the manifest and init segments
fn media_number(path: &str) -> Option<u64> {
    let name = path.strip_suffix(".m4s")?;
    let (_, number) = name.rsplit_once('-')?;
    number.parse().ok()
}

// a live window that slides on while a slow segment downloads must not make the switch skip segments
#[tokio::test]
async fn an_abr_switch_resumes_after_the_last_segment_when_the_live_window_moved() {
    let anchor_unix = now_unix() - AGE_SECS;
    let requests = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&requests);
    let url = serve(move |path| {
        if path == "/manifest.mpd" {
            return Some(live_mpd(anchor_unix).into_bytes());
        }
        seen.lock().unwrap().push(String::from(path));
        if path.starts_with("/high-") {
            std::thread::sleep(SLOW_HIGH);
            return Some(vec![0x11; 600_000]);
        }
        Some(vec![0x22; 2_000])
    });
    let mut src = DashSrc::new(format!("{url}/manifest.mpd")).with_abr();
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .unwrap();
    let mut sink = DiscardSink;
    // the stream is live and never ends, the window it lists runs out well within this
    let played = tokio::time::timeout(SLOW_HIGH * 3, src.run(&mut sink)).await;
    assert!(played.is_err(), "a live run plays until it is stopped");

    let requests = requests.lock().unwrap();
    let media: Vec<&String> = requests
        .iter()
        .filter(|path| media_number(path).is_some())
        .collect();
    assert!(
        media.first().is_some_and(|path| path.starts_with("/high-")),
        "starts on high: {requests:?}"
    );
    assert!(
        media.iter().skip(1).all(|path| path.starts_with("/low-")),
        "switches to low after the slow segment: {requests:?}"
    );
    let numbers: Vec<u64> = media.iter().filter_map(|path| media_number(path)).collect();
    assert!(numbers.len() >= 3, "{requests:?}");
    assert!(
        numbers.windows(2).all(|pair| pair[1] == pair[0] + 1),
        "each segment follows the one before: {requests:?}"
    );
}
