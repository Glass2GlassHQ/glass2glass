//! M187: caps-driven audioresample. `Caps::Audio.sample_rate` gained an "any"
//! sentinel (`ANY_SAMPLE_RATE` = 0), so a bare `audioresample` advertises "any
//! rate" and takes its output rate from a downstream capsfilter
//! (`audioresample ! audio/x-raw,rate=16000`). The `samplerate` property still
//! wins, and a bare audioresample is a passthrough.
//!
//! Why a green run proves the resample: audiotestsrc emits 48 kHz, so the only
//! way the pipeline satisfies a 16 kHz capsfilter is audioresample taking 16 kHz
//! from the solve and resampling to it.
//!
//! `default_registry` is `std`-gated, so this file is too.
#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

async fn run_line(line: &str) -> u64 {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line:?} should parse: {e:?}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("{line:?} should run: {e:?}"))
        .frames_consumed
}

#[tokio::test]
async fn capsfilter_drives_audioresample_rate() {
    // No samplerate on audioresample: the downstream capsfilter (16 kHz) pins
    // the output, configure_output hands it over, audioresample 48k -> 16k.
    let line = "audiotestsrc num-buffers=3 freq=440 \
                ! audioresample ! audio/x-raw,format=S16LE,rate=16000 ! fakesink";
    assert_eq!(run_line(line).await, 3, "{line}");
}

#[tokio::test]
async fn rate_only_capsfilter_drives_resample() {
    // Combine with M184: a format-less, rate-only capsfilter still drives the
    // rate (format stays the source's S16LE).
    let line = "audiotestsrc num-buffers=3 ! audioresample ! audio/x-raw,rate=16000 ! fakesink";
    assert_eq!(run_line(line).await, 3, "{line}");
}

#[tokio::test]
async fn bare_audioresample_is_passthrough() {
    // No property, no downstream rate constraint: passthrough (no resampling),
    // instead of failing to negotiate.
    assert_eq!(
        run_line("audiotestsrc num-buffers=3 ! audioresample ! fakesink").await,
        3
    );
}

#[tokio::test]
async fn audioresample_samplerate_property_still_works() {
    assert_eq!(
        run_line("audiotestsrc num-buffers=3 ! audioresample samplerate=16000 ! fakesink").await,
        3
    );
}
