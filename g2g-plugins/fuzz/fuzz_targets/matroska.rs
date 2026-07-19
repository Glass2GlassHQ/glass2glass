#![no_main]
// EBML / Matroska demux: feed arbitrary bytes to the parser.
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut d = g2g_plugins::matroska::MatroskaDemuxer::new();
    d.push_data(data);
});
