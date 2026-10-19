#![cfg(feature = "std")]

use std::path::{Path, PathBuf};

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

// a 440 Hz sine as pcm_u8 in Matroska, and ffmpeg's decode of it to s16le
const U8_MKA: &str = "sine_8k_mono_u8.mka";
const U8_DECODED: &str = "sine_8k_mono_u8.mka.s16le";
const AUDIO_CAPS: &str = "audio/x-raw,format=S16LE,rate=8000,channels=1";

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[tokio::test]
async fn matroska_pcm_decodes_through_audioresample() {
    let out = std::env::temp_dir().join(format!("g2g-m1598-{}.s16le", std::process::id()));
    let line = format!(
        "filesrc location={} ! decodebin ! audioconvert dithering=none ! audioresample \
         ! {AUDIO_CAPS} ! filesink location={}",
        fixture(U8_MKA).display(),
        out.display()
    );
    let graph =
        parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    let decoded = std::fs::read(&out).expect("the sink wrote its file");
    std::fs::remove_file(&out).ok();
    assert_eq!(
        decoded,
        std::fs::read(fixture(U8_DECODED)).expect("the expected decode is committed")
    );
}
