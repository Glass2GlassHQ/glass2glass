#![no_main]
// Fragmented-MP4 / CMAF demux over untrusted input: the moof / traf / trun / senc
// box parsing the HLS-fMP4 path runs, distinct from the progressive mp4_streams
// box parser. Driven through the element's fragment reassembly.
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    g2g_plugins::fmp4demux::fuzz_parse(data);
});
