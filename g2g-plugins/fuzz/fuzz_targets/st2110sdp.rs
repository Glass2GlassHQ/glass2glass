#![no_main]
// ST 2110 SDP text parsing: the media / rtpmap / fmtp / ptp lines a receiver
// configures from, attacker-controlled as `from_utf8_lossy` text.
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    let _ = g2g_plugins::st2110sdp::St2110Sdp::parse(&text);
    let _ = g2g_plugins::st2110sdp::St2110Session::parse(&text);
});
