#![no_main]
// Container / codec content sniffing over untrusted leading bytes: the magic
// signature probes plus Annex-B and text detection FileSrc runs on any input.
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = g2g_plugins::typefind::sniff(data);
    let _ = g2g_plugins::typefind::sniff_caps(data);
});
