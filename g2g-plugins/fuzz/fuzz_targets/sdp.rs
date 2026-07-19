#![no_main]
// SDP text parsing (WebRTC / ST 2110 session description) over untrusted input.
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let s = String::from_utf8_lossy(data);
    let _ = g2g_plugins::st2110sdp::St2110Sdp::parse(&s);
    let _ = g2g_plugins::st2110sdp::St2110Session::parse(&s);
});
