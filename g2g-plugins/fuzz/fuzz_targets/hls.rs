#![no_main]
// HLS m3u8 playlist parsing (master + media): attribute / tag / duration parsing
// over an attacker-controlled manifest.
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    let _ = g2g_plugins::hls::parse(&text);
});
