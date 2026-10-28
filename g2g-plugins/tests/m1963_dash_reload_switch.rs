#![cfg(feature = "dash")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use g2g_core::runtime::SourceLoop;
use g2g_core::{ByteStreamEncoding, Caps, PipelinePacket};
use g2g_plugins::dashsrc::DashSrc;

#[path = "common/collect_sink.rs"]
mod collect_sink;
use collect_sink::CollectSink;
#[path = "common/dash_live_server.rs"]
mod dash_live_server;
use dash_live_server::{now_unix, xs_datetime};
#[path = "common/http_routes.rs"]
mod http_routes;
use http_routes::serve;

// two 4 s segments are complete when the run starts, the third is not
const LIVE_PRESENTATION_AGE_SECS: f64 = 8.2;
const RELOAD_INTERVAL_MS: u64 = 20;

// one Representation in whole seconds
fn live_mpd(availability_start_unix: f64) -> String {
    format!(
        "<MPD type=\"dynamic\" availabilityStartTime=\"{}\" timeShiftBufferDepth=\"PT30S\" \
              minimumUpdatePeriod=\"PT1S\">\
           <Period id=\"p\" start=\"PT0S\"><AdaptationSet mimeType=\"video/mp4\">\
             <SegmentTemplate initialization=\"init-$RepresentationID$.mp4\" \
                media=\"$RepresentationID$-$Number$.m4s\" startNumber=\"0\" duration=\"4\" timescale=\"1\"/>\
             <Representation id=\"seconds\" bandwidth=\"1\"/>\
           </AdaptationSet></Period>\
         </MPD>",
        xs_datetime(availability_start_unix)
    )
}

// the finished presentation adds a higher Representation in milliseconds, which the reload picks
const ENDED_MPD: &str = "<MPD type=\"static\" mediaPresentationDuration=\"PT12S\">\
       <Period><AdaptationSet mimeType=\"video/mp4\">\
         <Representation id=\"seconds\" bandwidth=\"1\">\
           <SegmentTemplate initialization=\"init-$RepresentationID$.mp4\" \
              media=\"$RepresentationID$-$Number$.m4s\" startNumber=\"0\" duration=\"4\" timescale=\"1\"/>\
         </Representation>\
         <Representation id=\"millis\" bandwidth=\"2\">\
           <SegmentTemplate initialization=\"init-$RepresentationID$.mp4\" \
              media=\"$RepresentationID$-$Number$.m4s\" startNumber=\"0\" duration=\"4000\" timescale=\"1000\"/>\
         </Representation>\
       </AdaptationSet></Period>\
     </MPD>";

fn delivered_bodies(sink: &CollectSink) -> String {
    let mut bodies = String::new();
    for packet in &sink.packets {
        if let PipelinePacket::DataFrame(frame) = packet {
            let bytes = frame.domain.as_system_slice().unwrap();
            bodies.push_str(std::str::from_utf8(bytes).unwrap());
        }
    }
    bodies
}

#[tokio::test]
async fn a_reload_picking_another_representation_resumes_after_the_last_segment() {
    let manifest = live_mpd(now_unix() - LIVE_PRESENTATION_AGE_SECS);
    let manifest_requests = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&manifest_requests);
    let url = serve(move |path| {
        if path != "/manifest.mpd" {
            return Some(format!("{path};").into_bytes());
        }
        let body = if seen.fetch_add(1, Ordering::SeqCst) == 0 {
            manifest.clone()
        } else {
            String::from(ENDED_MPD)
        };
        Some(body.into_bytes())
    });
    let mut src = DashSrc::new(format!("{url}/manifest.mpd"))
        .with_reload_interval_ms(RELOAD_INTERVAL_MS);
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .unwrap();
    let mut sink = CollectSink::default();
    src.run(&mut sink).await.unwrap();

    assert_eq!(
        delivered_bodies(&sink),
        "/init-seconds.mp4;/seconds-0.m4s;/seconds-1.m4s;/init-millis.mp4;/millis-2.m4s;"
    );
}
