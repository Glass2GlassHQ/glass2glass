#![no_main]
// IV8 demux over untrusted input: the 16-byte packet headers and the video
// frame reassembly, driven through the element's buffering and end-of-stream paths.
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    g2g_plugins::iv8demux::fuzz_parse(data);
});
