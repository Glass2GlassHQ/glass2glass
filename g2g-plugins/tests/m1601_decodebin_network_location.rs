#![cfg(feature = "rtsp")]

use g2g_core::log::{self, LogLevel, RingSink};
use g2g_core::runtime::parse_launch;
use g2g_plugins::registry::default_registry;

#[test]
fn an_rtsp_location_is_not_opened_as_a_file() {
    let location = "rtsp://127.0.0.1:1/g2g-m1601";
    let ring = RingSink::new(64);
    log::set_sink(Box::new(ring.clone()));
    log::set_default_level(LogLevel::Error);
    // the parse result depends on which H.264 decoders this build has
    let _ = parse_launch(
        &default_registry(),
        &format!("rtspsrc location={location} ! decodebin ! fakesink"),
    );
    let lines: Vec<String> = ring.drain().into_iter().map(|r| r.message).collect();
    assert!(
        !lines.iter().any(|line| line.contains(location)),
        "the parse opens the RTSP location as a file: {lines:?}"
    );
}
