#![no_main]
// RTSP request parsing: the request line (method / URI / version) + headers and
// content-length body framing over an attacker-controlled request buffer.
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = g2g_plugins::rtspserver::RtspRequest::parse(data);
});
