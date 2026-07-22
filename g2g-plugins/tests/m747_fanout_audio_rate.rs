//! M747 - a `decodebin` fan-out audio branch survives a runtime sample-rate
//! refinement. A demux advertises the compressed stream with placeholder
//! channels / rate; the AAC decoder only learns the real rate once it decodes a
//! frame and emits it as a runtime `CapsChanged`. When that rate differs from
//! the branch's negotiation-time pin (a 44.1 kHz stream reaching a `rate=48000`
//! capsfilter), an `audioresample` in the branch absorbs the refinement and the
//! branch flows; without one the branch is genuinely unsatisfiable and fails
//! loud, naming the conflicting caps.
//!
//! The negotiation fix lives in the caps solver: a caps-driven `audioresample`
//! after a caps-driven `audioconvert` sees the decoder's `ANY_CHANNELS`
//! placeholder, and the passthrough coupling must treat that as a wildcard (it
//! did not, collapsing the derived set to empty). This exercises the real
//! decode-through-fan-out path on checked-in AAC-in-TS fixtures.
//!
//! Needs the AAC decoder in the autoplug pool (ffmpeg) and `default_registry`
//! (std).
#![cfg(all(feature = "std", feature = "ffmpeg"))]

use std::path::PathBuf;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, G2gError, PipelineClock, ANY_CHANNELS,
};
use g2g_plugins::audioresample::AudioResample;
use g2g_plugins::registry::default_registry;

const AAC_44100: &[u8] = include_bytes!("fixtures/aac_44100.ts");
const AAC_48000: &[u8] = include_bytes!("fixtures/aac_48000.ts");
const AV_44100: &[u8] = include_bytes!("fixtures/av_h264_aac44100.ts");

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn temp(tag: &str, bytes: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!("g2g-m747-{tag}-{}.ts", std::process::id()));
    std::fs::write(&path, bytes).expect("write temp");
    path
}

fn out_path(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("g2g-m747-out-{tag}-{}.raw", std::process::id()));
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

/// A caps-driven `audioresample` must derive a non-empty PCM output for a
/// decoder's `ANY_CHANNELS` placeholder input, passing the placeholder through
/// (a downstream capsfilter pins it later). The regression: the derive rejected
/// a 0 channel count, so the solver read the branch as an empty link.
#[test]
fn audioresample_derives_output_for_any_channels_placeholder() {
    let r = AudioResample::auto();
    let CapsConstraint::DerivedCoupled { derive, .. } = r.caps_constraint_as_transform() else {
        panic!("audioresample exposes a DerivedCoupled transform constraint");
    };
    let placeholder = Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: ANY_CHANNELS,
        sample_rate: 44_100,
    };
    let out = derive(&placeholder);
    assert!(
        !out.is_empty(),
        "an ANY_CHANNELS placeholder input must still derive an output"
    );
    // Channels pass through untouched: every derived alternative keeps the
    // placeholder count (a downstream capsfilter narrows it, not the resampler).
    assert!(
        out.alternatives().iter().all(|c| matches!(
            c,
            Caps::Audio {
                channels: ANY_CHANNELS,
                ..
            }
        )),
        "channels pass through as the placeholder: {:?}",
        out.alternatives()
    );
}

/// The headline: a 44.1 kHz AAC-in-TS through the `decodebin` fan-out, with an
/// `audioresample` bridging to a `rate=48000` pin, decodes to 48 kHz mono S16LE.
/// The decoder learns the real 44.1 kHz rate only at runtime; the resampler
/// absorbs it. Deterministic (two runs are byte-identical).
#[tokio::test]
async fn fanout_44100_with_resampler_flows_to_48k() {
    let src = temp("aac44", AAC_44100);
    let wav = out_path("aac44.wav");
    let line = format!(
        "filesrc location={} ! decodebin name=d d.audio_0 ! audioconvert ! audioresample \
         ! audio/x-raw,format=S16LE,rate=48000,channels=1 ! wavsink location={}",
        src.display(),
        wav.display()
    );
    assert!(run_line(&line).await.expect("branch flows") > 0, "{line}");
    let bytes = std::fs::read(&wav).expect("wav written");
    assert_eq!(wav_rate(&bytes), 48_000, "output resampled to 48 kHz");
    // ~0.5 s at 48 kHz mono S16 (plus decoder priming) is tens of KB, never empty.
    assert!(bytes.len() > 40_000, "decoded PCM present: {}", bytes.len());

    // Determinism: a second run to a raw sink is byte-identical to the first.
    let raw1 = out_path("aac44-1");
    let raw2 = out_path("aac44-2");
    let raw_line = |p: &PathBuf| {
        format!(
            "filesrc location={} ! decodebin name=d d.audio_0 ! audioconvert ! audioresample \
             ! audio/x-raw,format=S16LE,rate=48000,channels=1 ! filesink location={}",
            src.display(),
            p.display()
        )
    };
    run_line(&raw_line(&raw1)).await.expect("run 1");
    run_line(&raw_line(&raw2)).await.expect("run 2");
    assert_eq!(
        std::fs::read(&raw1).unwrap(),
        std::fs::read(&raw2).unwrap(),
        "resampled output is deterministic"
    );

    for p in [&src, &wav, &raw1, &raw2] {
        let _ = std::fs::remove_file(p);
    }
}

/// The rate-matched case still works: a 48 kHz stream through the same fan-out +
/// resampler (which passes through at 1:1) reaches the 48 kHz sink.
#[tokio::test]
async fn fanout_48000_with_resampler_flows() {
    let src = temp("aac48", AAC_48000);
    let wav = out_path("aac48.wav");
    let line = format!(
        "filesrc location={} ! decodebin name=d d.audio_0 ! audioconvert ! audioresample \
         ! audio/x-raw,format=S16LE,rate=48000,channels=1 ! wavsink location={}",
        src.display(),
        wav.display()
    );
    assert!(run_line(&line).await.expect("branch flows") > 0, "{line}");
    let bytes = std::fs::read(&wav).expect("wav written");
    assert_eq!(wav_rate(&bytes), 48_000);
    assert!(bytes.len() > 40_000, "decoded PCM present: {}", bytes.len());
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&wav);
}

/// Without a resampler the 44.1 kHz -> 48 kHz branch is genuinely unsatisfiable
/// (`audioconvert` does not resample): the runtime rate refinement cannot cross
/// the `rate=48000` pin, so the fan-out branch fails loud with `CapsMismatch`
/// (the run also logs the conflicting caps on the caps category). A clean,
/// deterministic failure, not a silent wrong-rate output.
#[tokio::test]
async fn fanout_44100_without_resampler_fails_loud() {
    let src = temp("aac44-nors", AAC_44100);
    let raw = out_path("aac44-nors");
    let line = format!(
        "filesrc location={} ! decodebin name=d d.audio_0 ! audioconvert \
         ! audio/x-raw,format=S16LE,rate=48000,channels=1 ! filesink location={}",
        src.display(),
        raw.display()
    );
    assert_eq!(
        run_line(&line).await,
        Err(G2gError::CapsMismatch),
        "unsatisfiable rate pin fails loud: {line}"
    );
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&raw);
}

/// Both fan-out branches of an A/V TS flow in one graph: the audio branch
/// resamples 44.1 -> 48 kHz while the video branch decodes to I420. The M747
/// change does not disturb the video branch (its head is a passthrough parser).
#[tokio::test]
async fn fanout_av_both_branches_flow() {
    let src = temp("av", AV_44100);
    let a = out_path("av-a");
    let v = out_path("av-v");
    let line = format!(
        "filesrc location={} ! decodebin name=d \
         d.audio_0 ! audioconvert ! audioresample \
           ! audio/x-raw,format=S16LE,rate=48000,channels=1 ! filesink location={} \
         d.video_0 ! videoconvert ! video/x-raw,format=I420 ! filesink location={}",
        src.display(),
        a.display(),
        v.display()
    );
    assert!(
        run_line(&line).await.expect("both branches flow") > 0,
        "{line}"
    );
    let (asz, vsz) = (
        std::fs::metadata(&a).map(|m| m.len()).unwrap_or(0),
        std::fs::metadata(&v).map(|m| m.len()).unwrap_or(0),
    );
    assert!(asz > 0, "audio branch produced PCM: {asz}");
    // I420 at 160x120 is 160*120*3/2 = 28800 bytes per frame; whole frames only.
    assert!(
        vsz > 0 && vsz % 28_800 == 0,
        "video branch produced I420 frames: {vsz}"
    );
    for p in [&src, &a, &v] {
        let _ = std::fs::remove_file(p);
    }
}
