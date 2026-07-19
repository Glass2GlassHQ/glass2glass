#![no_main]
// RTCP compound-packet parsing over untrusted input (the RTP control / feedback
// channel used by WebRTC and RTP sessions).
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = g2g_plugins::rtcp::parse_compound(data);
});
