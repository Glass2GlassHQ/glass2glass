#![no_main]
// Opus identification header + packet TOC / frame-count parsing.
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    g2g_plugins::opusparse::fuzz_parse(data);
});
