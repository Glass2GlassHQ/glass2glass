#![no_main]
// AAC ADTS / LOAS-LATM header + AudioSpecificConfig bit reader.
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    g2g_plugins::aacparse::fuzz_parse(data);
});
