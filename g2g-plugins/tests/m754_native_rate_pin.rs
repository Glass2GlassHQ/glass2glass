//! M754 - a downstream concrete rate pin at the stream's NATIVE rate negotiates
//! and flows with no resampler. `FfmpegAudioDec` advertises its PCM output rate as
//! two alternatives (a concrete `48_000` default plus an `ANY_SAMPLE_RATE`
//! wildcard), so a `rate=44100` capsfilter intersects the wildcard and the 44.1 kHz
//! decode reaches the sink directly, while a `rate=48000` pin on a 48 kHz stream
//! still works and a `rate=48000` pin on a 44.1 kHz stream with no `audioresample`
//! still fails loud (M749 preserved: no silent relabel).
//!
//! Reuses the M747/M749 AAC-in-TS fixtures (mono 44.1 kHz and 48 kHz). Needs the
//! AAC decoder in the autoplug pool (ffmpeg) and `default_registry` (std).
#![cfg(all(feature = "std", feature = "ffmpeg"))]

use std::path::PathBuf;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{G2gError, PipelineClock};
use g2g_plugins::registry::default_registry;

const AAC_44100: &[u8] = include_bytes!("fixtures/aac_44100.ts");
const AAC_48000: &[u8] = include_bytes!("fixtures/aac_48000.ts");

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn temp(tag: &str, bytes: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!("g2g-m754-{tag}-{}.ts", std::process::id()));
    std::fs::write(&path, bytes).expect("write temp");
    path
}

fn out_path(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("g2g-m754-out-{tag}-{}.wav", std::process::id()));
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

/// A WAV header's sample rate (bytes 24..28, LE) and PCM data length (40..44, LE).
fn wav_rate_and_samples(bytes: &[u8]) -> (u32, u32) {
    assert!(
        bytes.len() > 44 && &bytes[..4] == b"RIFF",
        "a RIFF/WAVE file"
    );
    let rate = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    let data = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
    (rate, data / 2) // mono S16 = 2 bytes/sample
}

/// The headline fix: a 44.1 kHz stream to a `rate=44100` pin negotiates and flows
/// with no resampler, and the sink writes a real 44.1 kHz header.
#[tokio::test]
async fn native_44100_pin_flows_no_resampler() {
    let src = temp("n44", AAC_44100);
    let wav = out_path("n44");
    let line = format!(
        "filesrc location={} ! tsdemux stream=aac ! ffmpegaudiodec ! audioconvert \
         ! audio/x-raw,format=S16LE,rate=44100,channels=1 ! wavsink location={}",
        src.display(),
        wav.display()
    );
    assert!(
        run_line(&line).await.expect("native-rate chain flows") > 0,
        "{line}"
    );
    let bytes = std::fs::read(&wav).expect("wav written");
    let (rate, samples) = wav_rate_and_samples(&bytes);
    assert_eq!(
        rate, 44_100,
        "output header is the native 44.1 kHz, not the 48 kHz default"
    );
    assert!(samples > 0, "decoded samples present");
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&wav);
}

/// The 48 kHz stream to a `rate=48000` pin still works (regression guard: the
/// concrete first alternative keeps satisfying an explicit 48 kHz pin).
#[tokio::test]
async fn native_48000_pin_still_flows() {
    let src = temp("n48", AAC_48000);
    let wav = out_path("n48");
    let line = format!(
        "filesrc location={} ! tsdemux stream=aac ! ffmpegaudiodec ! audioconvert \
         ! audio/x-raw,format=S16LE,rate=48000,channels=1 ! wavsink location={}",
        src.display(),
        wav.display()
    );
    assert!(
        run_line(&line).await.expect("48 kHz chain flows") > 0,
        "{line}"
    );
    let bytes = std::fs::read(&wav).expect("wav written");
    let (rate, samples) = wav_rate_and_samples(&bytes);
    assert_eq!(rate, 48_000);
    assert!(samples > 0);
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&wav);
}

/// M749 preserved: a 44.1 kHz stream to a `rate=48000` pin with NO resampler still
/// fails loud (the wildcard lets the pin negotiate at startup, but the runtime
/// 44.1 kHz `CapsChanged` has no converter to cross, so it fails rather than
/// relabelling native-rate samples as 48 kHz).
#[tokio::test]
async fn cross_rate_pin_without_resampler_fails_loud() {
    let src = temp("x48", AAC_44100);
    let wav = out_path("x48");
    let line = format!(
        "filesrc location={} ! tsdemux stream=aac ! ffmpegaudiodec ! audioconvert \
         ! audio/x-raw,format=S16LE,rate=48000,channels=1 ! wavsink location={}",
        src.display(),
        wav.display()
    );
    assert_eq!(
        run_line(&line).await,
        Err(G2gError::CapsMismatch),
        "44.1 kHz to a 48 kHz pin with no resampler still fails loud: {line}"
    );
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&wav);
}
