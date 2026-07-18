#![no_main]
// MP4 box parsing over untrusted input: g2g-owned pure-Rust demux, prime target.
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = g2g_plugins::mp4demuxn::forwardable_streams(data);
    let _ = g2g_plugins::mp4demuxn::subtitle_streams(data);
});
