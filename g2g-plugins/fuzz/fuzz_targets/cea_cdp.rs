#![no_main]
// CEA-708 CDP closed-caption parsing (a notoriously bug-prone area).
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = g2g_plugins::cea::parse_cdp(data);
});
