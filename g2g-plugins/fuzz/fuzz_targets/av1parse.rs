#![no_main]
// AV1 sequence-header OBU parsing: hand-written LEB128 + MSB-first bit reader
// over attacker-controlled frame bytes (the densest untrusted-input path).
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    g2g_plugins::av1parse::fuzz_parse(data);
});
