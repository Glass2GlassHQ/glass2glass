//! M749 - a LINEAR audio chain (no tee, so the runner's `Reconfigure` branch
//! mode) with an infeasible sample-rate pin and no `audioresample` fails loud
//! instead of silently relabelling native-rate PCM. A `tsdemux stream=aac`
//! feeds one AAC decoder that only learns the real 44.1 kHz rate once it decodes
//! a frame and emits it as a runtime `CapsChanged`. When that rate cannot cross
//! a downstream `rate=48000` pin and the chain has no resampler, the refinement
//! has no solution: no runtime producer renegotiates its output caps, so the run
//! fails with `CapsMismatch` rather than writing 44.1 kHz samples under a 48 kHz
//! header. With an `audioresample` in the chain the pin is satisfiable and the
//! chain flows to 48 kHz.
//!
//! Reuses the M747 AAC-in-TS fixtures. Needs the AAC decoder in the autoplug
//! pool (ffmpeg) and `default_registry` (std).
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
    let path = std::env::temp_dir().join(format!("g2g-m749-{tag}-{}.ts", std::process::id()));
    std::fs::write(&path, AAC_44100).expect("write temp");
    path
}

fn out_path(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("g2g-m749-out-{tag}-{}.raw", std::process::id()));
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

/// The mono S16LE sample rate a WAV header advertises (bytes 24..28, LE).
fn wav_rate(bytes: &[u8]) -> u32 {
    assert!(
        bytes.len() > 44 && &bytes[..4] == b"RIFF",
        "a RIFF/WAVE file"
    );
    u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]])
}

/// The headline regression: a linear 44.1 kHz decode reaching a `rate=48000`
/// pin with no resampler fails loud with `CapsMismatch` on a `filesink` tail
/// (which accepts any caps, so the conflict must surface upstream at the pin,
/// not at the sink), rather than writing 44.1 kHz samples labelled 48 kHz.
#[tokio::test]
async fn linear_infeasible_pin_fails_loud_filesink() {
    let src = temp("nors-fs");
    let raw = out_path("nors-fs");
    let line = format!(
        "filesrc location={} ! tsdemux stream=aac ! ffmpegaudiodec ! audioconvert \
         ! audio/x-raw,format=S16LE,rate=48000,channels=1 ! filesink location={}",
        src.display(),
        raw.display()
    );
    assert_eq!(
        run_line(&line).await,
        Err(G2gError::CapsMismatch),
        "unsatisfiable linear rate pin fails loud: {line}"
    );
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&raw);
}

/// Same infeasible chain with a `wavsink` tail: a native sink that would have
/// written a 48 kHz header over 44.1 kHz samples must also fail loud.
#[tokio::test]
async fn linear_infeasible_pin_fails_loud_wavsink() {
    let src = temp("nors-wav");
    let wav = out_path("nors-wav");
    let line = format!(
        "filesrc location={} ! tsdemux stream=aac ! ffmpegaudiodec ! audioconvert \
         ! audio/x-raw,format=S16LE,rate=48000,channels=1 ! wavsink location={}",
        src.display(),
        wav.display()
    );
    assert_eq!(
        run_line(&line).await,
        Err(G2gError::CapsMismatch),
        "unsatisfiable linear rate pin fails loud on wavsink: {line}"
    );
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&wav);
}

/// The feasible twin still negotiates and flows: an `audioresample` before the
/// `rate=48000` pin absorbs the 44.1 kHz refinement, so the linear chain reaches
/// the 48 kHz sink and writes a WAV whose header and sample count are the real
/// 48 kHz output, not the native rate.
#[tokio::test]
async fn linear_with_resampler_flows_to_48k() {
    let src = temp("rs");
    let wav = out_path("rs.wav");
    let line = format!(
        "filesrc location={} ! tsdemux stream=aac ! ffmpegaudiodec ! audioconvert ! audioresample \
         ! audio/x-raw,format=S16LE,rate=48000,channels=1 ! wavsink location={}",
        src.display(),
        wav.display()
    );
    assert!(run_line(&line).await.expect("chain flows") > 0, "{line}");
    let bytes = std::fs::read(&wav).expect("wav written");
    assert_eq!(wav_rate(&bytes), 48_000, "output resampled to 48 kHz");
    // A 44.1 kHz half-second resampled to 48 kHz is > 20k mono S16 samples; a
    // silent relabel would leave the ~11.7k native-rate samples instead.
    let data_bytes = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
    assert!(
        data_bytes / 2 > 20_000,
        "resampled 48 kHz sample count, not native-rate: {}",
        data_bytes / 2
    );
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&wav);
}
