// with no display or audio sink in the build, the auto sinks fall back to a fake sink that still takes only their media
#![cfg(all(
    feature = "std",
    not(any(
        feature = "wgpu-present",
        feature = "wayland-sink",
        feature = "kms-sink",
        feature = "gl-sink",
        feature = "metal-sink",
        feature = "alsa-sink",
        feature = "pulse-sink",
        feature = "coreaudio",
        feature = "wasapi-sink",
    ))
))]

use g2g_core::runtime::{block_on, parse_launch, run_graph, RunStats};
use g2g_core::G2gError;
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;

const FRAMES: u64 = 10;
const LINK_CAPACITY: usize = 4;

fn run(line: &str) -> Result<RunStats, G2gError> {
    let graph = parse_launch(&default_registry(), line).expect("the line parses");
    block_on(run_graph(graph, &WallClock::new(), LINK_CAPACITY))
}

#[test]
fn audio_into_autovideosink_fails_negotiation() {
    let result = run(&format!(
        "audiotestsrc num-buffers={FRAMES} ! autovideosink"
    ));
    assert_eq!(result.err(), Some(G2gError::CapsMismatch));
}

#[test]
fn video_into_autovideosink_runs() {
    let stats = run(&format!(
        "videotestsrc num-buffers={FRAMES} ! autovideosink"
    ))
    .expect("raw video reaches the fallback");
    assert_eq!(stats.frames_consumed, FRAMES);
}

#[test]
fn video_into_autoaudiosink_fails_negotiation() {
    let result = run(&format!(
        "videotestsrc num-buffers={FRAMES} ! autoaudiosink"
    ));
    assert_eq!(result.err(), Some(G2gError::CapsMismatch));
}

#[test]
fn audio_into_autoaudiosink_runs() {
    let stats = run(&format!(
        "audiotestsrc num-buffers={FRAMES} ! autoaudiosink"
    ))
    .expect("raw audio reaches the fallback");
    assert_eq!(stats.frames_consumed, FRAMES);
}
