#![cfg(all(target_os = "linux", feature = "ffmpeg"))]

use std::path::{Path, PathBuf};

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

// ffmpeg's `-c:a ac3 -f wav` and `-c:a libmp3lame -f wav` of a 440 Hz sine, beside ffmpeg's s16le decode of each
const AC3_WAV: &str = "wav_ac3.wav";
const MP3_WAV: &str = "wav_mp3.wav";

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

async fn decode(name: &str) -> Vec<u8> {
    let out = std::env::temp_dir().join(format!("g2g-m1854-{name}-{}.raw", std::process::id()));
    let line = format!(
        "filesrc location={} ! decodebin ! audioconvert ! audio/x-raw,format=S16LE ! filesink location={}",
        fixture(name).display(),
        out.display()
    );
    let graph =
        parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    let bytes = std::fs::read(&out).expect("the sink wrote its file");
    std::fs::remove_file(&out).ok();
    bytes
}

fn expected(name: &str) -> Vec<u8> {
    std::fs::read(fixture(&format!("{name}.s16le"))).expect("the expected decode is committed")
}

#[tokio::test]
async fn ac3_in_wav_decodes_as_ffmpeg_does() {
    assert_eq!(decode(AC3_WAV).await, expected(AC3_WAV));
}

#[tokio::test]
async fn mp3_in_wav_decodes_as_ffmpeg_does() {
    assert_eq!(decode(MP3_WAV).await, expected(MP3_WAV));
}
