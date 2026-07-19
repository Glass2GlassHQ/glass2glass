#![no_main]
// VP8 keyframe header parsing (start-code + dimension words).
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    g2g_plugins::vp8parse::fuzz_parse(data);
});
