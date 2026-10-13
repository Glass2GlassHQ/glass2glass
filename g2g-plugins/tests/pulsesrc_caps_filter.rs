#![cfg(all(target_os = "linux", feature = "pulse-src"))]

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{G2gError, HardwareError, PipelineClock};
use g2g_plugins::appsink::{register_appsink_pull, Pull};
use g2g_plugins::registry::default_registry;

const CHANNEL: &str = "pulsesrc_caps_filter";
const FILTER_RATE: u64 = 16_000;
const FILTER_CHANNELS: u64 = 1;
const DEFAULT_FORMAT_SAMPLE_BYTES: u64 = 2;
const FRAGMENT_US: u64 = 10_000;
const FRAGMENTS: u64 = 5;
const MICROSECONDS_PER_SECOND: u64 = 1_000_000;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn records_the_rate_and_channels_a_caps_filter_asks_for() {
    let pull = register_appsink_pull(CHANNEL);
    let launch = format!(
        "pulsesrc num-buffers={FRAGMENTS} latency-time={FRAGMENT_US} \
         ! audio/x-raw,rate={FILTER_RATE},channels={FILTER_CHANNELS} \
         ! appsink channel={CHANNEL}"
    );
    let graph = parse_launch(&default_registry(), &launch).expect("parses");
    match run_graph(graph, &ZeroClock, 4).await {
        Err(G2gError::Hardware(HardwareError::PulseAudio(_))) => {
            eprintln!("skip pulsesrc_caps_filter: no reachable PulseAudio server");
            return;
        }
        other => other.expect("runs"),
    };

    let fragment_bytes = FILTER_RATE * FILTER_CHANNELS * DEFAULT_FORMAT_SAMPLE_BYTES * FRAGMENT_US
        / MICROSECONDS_PER_SECOND;
    let mut fragments = 0;
    while let Pull::Frame(frame) = pull.try_pull() {
        let bytes = frame.domain.as_system_slice().expect("system memory");
        assert_eq!(bytes.len() as u64, fragment_bytes);
        fragments += 1;
    }
    assert_eq!(fragments, FRAGMENTS);
}
