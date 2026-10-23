#![cfg(feature = "std")]

use std::path::PathBuf;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

const LINK_CAPACITY: usize = 4;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn fixture(name: &str) -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
        .display()
        .to_string()
}

async fn run_to_bytes(line_before_sink: &str, tag: &str) -> Vec<u8> {
    let out = std::env::temp_dir().join(format!("g2g-m1838-{tag}-{}.pcm", std::process::id()));
    let line = format!("{line_before_sink} ! filesink location={}", out.display());
    let graph = parse_launch(&default_registry(), &line).expect("pipeline parses");
    run_graph(graph, &ZeroClock, LINK_CAPACITY)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    let bytes = std::fs::read(&out).expect("the sink wrote a file");
    std::fs::remove_file(&out).expect("remove the output");
    bytes
}

// The fixture is mono 8 kHz S16LE, neither of which is the stereo 48 kHz the
// solve fixates an open caps to before the parser reads the header.
#[tokio::test]
async fn a_format_only_capsfilter_passes_the_stream_rate_and_channels_through() {
    let source = format!(
        "filesrc location={} ! wavparse",
        fixture("sine_8k_mono_s16le.wav")
    );
    let parsed = run_to_bytes(&source, "parsed").await;
    let converted = run_to_bytes(
        &format!("{source} ! audioconvert ! audio/x-raw,format=S16LE"),
        "converted",
    )
    .await;
    assert!(!parsed.is_empty());
    assert_eq!(converted, parsed);
}
