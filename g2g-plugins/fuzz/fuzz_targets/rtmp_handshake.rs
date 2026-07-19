#![no_main]
// RTMP handshake parsers over the peer's raw signature bytes (C1 / S1). These
// run before any size is trusted, so an out-of-bounds read here is wire-reachable.
use libfuzzer_sys::fuzz_target;
use g2g_plugins::rtmphandshake::{build_c2, build_s2, c1_has_digest};

fuzz_target!(|data: &[u8]| {
    let _ = build_s2(data);
    let _ = build_c2(data);
    let _ = c1_has_digest(data);
});
