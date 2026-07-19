#![no_main]
// FLV tag demux over untrusted input.
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut d = g2g_plugins::flv::FlvDemuxer::new();
    d.push_data(data);
});
