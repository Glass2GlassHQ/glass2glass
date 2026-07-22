//! M755 - element order between `audioconvert` and `audioresample` must not matter
//! for a mid-stream rate retarget. A 44.1 kHz decode reaching a `rate=48000` pin
//! flows to 48 kHz whether the resampler sits before or after the converter.
//!
//! `audioresample ! audioconvert ! rate-pin` used to fail loud at runtime: the
//! converter retargets `format` (a scalar with no wildcard), so the resampler's
//! backward feasibility snapshot was empty, and on the 44.1 kHz `CapsChanged` the
//! resampler defaulted to passthrough instead of keeping its 48 kHz target. The
//! resampler now derives its output from the caps-resolved rate (not just the
//! property), so it holds the target regardless of its position, and both orders
//! produce byte-identical output.
//!
//! Reuses the M747/M749 AAC-in-TS fixture (mono 44.1 kHz). Needs the AAC decoder in
//! the autoplug pool (ffmpeg) and `default_registry` (std).
#![cfg(all(feature = "std", feature = "ffmpeg"))]

use std::path::PathBuf;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{G2gError, PipelineClock};
use g2g_plugins::registry::default_registry;

const AAC_44100: &[u8] = include_bytes!("fixtures/aac_44100.ts");

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn temp(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("g2g-m755-{tag}-{}.ts", std::process::id()));
    std::fs::write(&path, AAC_44100).expect("write temp");
    path
}

fn out_path(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("g2g-m755-out-{tag}-{}.wav", std::process::id()));
    let _ = std::fs::remove_file(&path);
    path
}

async fn run_line(line: &str) -> Result<u64, G2gError> {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line:?} should parse: {e:?}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .map(|s| s.frames_consumed)
}

/// A WAV's sample rate (bytes 24..28) and its raw PCM data chunk (past the 44-byte
/// header), for asserting both the header and the samples themselves.
fn wav_rate_and_data(bytes: &[u8]) -> (u32, Vec<u8>) {
    assert!(
        bytes.len() > 44 && &bytes[..4] == b"RIFF",
        "a RIFF/WAVE file"
    );
    let rate = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    let len = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]) as usize;
    (rate, bytes[44..44 + len].to_vec())
}

/// Run one converter ordering to 48 kHz and return its resampled PCM data.
async fn run_order(tag: &str, chain: &str) -> Vec<u8> {
    let src = temp(tag);
    let wav = out_path(tag);
    let line = format!(
        "filesrc location={} ! tsdemux stream=aac ! ffmpegaudiodec ! {chain} \
         ! audio/x-raw,format=S16LE,rate=48000,channels=1 ! wavsink location={}",
        src.display(),
        wav.display()
    );
    assert!(run_line(&line).await.expect("chain flows") > 0, "{line}");
    let bytes = std::fs::read(&wav).expect("wav written");
    let (rate, data) = wav_rate_and_data(&bytes);
    assert_eq!(rate, 48_000, "resampled to 48 kHz: {line}");
    // 44.1 kHz upsampled to 48 kHz is more samples than the native decode; a silent
    // passthrough relabel would leave the native-rate count instead.
    assert!(
        data.len() / 2 > 6_000,
        "resampled 48 kHz sample count: {}",
        data.len() / 2
    );
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&wav);
    data
}

/// Both orderings flow to 48 kHz and produce byte-identical output: element order
/// between the two converters is irrelevant for a rate retarget.
#[tokio::test]
async fn converter_order_is_irrelevant_for_rate_retarget() {
    let convert_then_resample = run_order("ca", "audioconvert ! audioresample").await;
    let resample_then_convert = run_order("rc", "audioresample ! audioconvert").await;
    assert_eq!(
        convert_then_resample, resample_then_convert,
        "both converter orders must resample 44.1 kHz to the same 48 kHz output"
    );
}

/// The previously-broken order, asserted on its own as the direct regression: the
/// resampler placed before the converter must not fail loud.
#[tokio::test]
async fn resample_before_convert_flows() {
    assert!(!run_order("solo", "audioresample ! audioconvert")
        .await
        .is_empty());
}
