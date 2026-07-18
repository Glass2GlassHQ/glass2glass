#![no_main]
// H.265 SPS geometry parsing: NAL scan + Exp-Golomb bit reader over attacker-controlled access-unit bytes.
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    g2g_plugins::h265parse::fuzz_parse(data);
});
