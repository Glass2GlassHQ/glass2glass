#![no_main]
// gst-launch-style pipeline-description parser (g2g_core::runtime::parse_launch),
// the untrusted-text surface the g2g-capi `g2g_pipeline_launch` C entry forwards
// after its NUL / UTF-8 checks. Element / property / caps-filter / link / bin
// text parsing over an attacker-controlled description. Parse only, no execution.
use libfuzzer_sys::fuzz_target;

thread_local! {
    // Building the element registry is expensive; do it once per fuzz worker.
    static REGISTRY: g2g_core::runtime::Registry = g2g_plugins::registry::default_registry();
}

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    REGISTRY.with(|reg| {
        let _ = g2g_core::runtime::parse_launch(reg, &text);
    });
});
