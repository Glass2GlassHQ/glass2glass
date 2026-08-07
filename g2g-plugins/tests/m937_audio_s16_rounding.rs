//! M937 - float->s16 audio decode matches ffmpeg bit-exactly. `FfmpegAudioDec`
//! converted float samples with truncation at 32767 scale where every other
//! libavcodec consumer (ffmpeg's swresample, GStreamer) rounds to nearest at
//! 32768, so an AC-3 decode differed from ffmpeg's decode of the same stream by
//! +-1 LSB on ~75% of samples. Oracle gate: ffmpeg encodes AC-3 in TS, then
//! both ffmpeg and g2g decode it to interleaved S16LE and the PCM must be
//! byte-identical (the M615 oracle discipline).
//!
//! Self-skips where the ffmpeg CLI is absent.
#![cfg(all(feature = "std", feature = "ffmpeg"))]

use std::path::PathBuf;
use std::process::Command;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::conformance::persist;
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn have(bin: &str) -> bool {
    Command::new(bin).arg("-version").output().is_ok()
}

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m937-{name}-{}", std::process::id()))
}

fn run(cmd: &mut Command) {
    let out = cmd.output().expect("the tool runs");
    assert!(
        out.status.success(),
        "{cmd:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

async fn run_line(line: &str) -> u64 {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line:?} should parse: {e:?}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("{line:?} should run: {e:?}"))
        .frames_consumed
}

fn record_oracle(detail: &str) {
    if std::env::var_os("G2G_CONFORMANCE_LOG").is_none() {
        std::env::set_var(
            "G2G_CONFORMANCE_LOG",
            std::env::temp_dir().join("g2g-conformance-m937.tsv"),
        );
    }
    persist::record_evidence(
        "ffmpegaudiodec",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("ac3")
            .detail(detail),
    )
    .expect("record oracle evidence");
}

#[tokio::test]
async fn ac3_decode_is_byte_identical_to_ffmpeg() {
    if !have("ffmpeg") {
        eprintln!("skipping: ffmpeg CLI not installed");
        return;
    }
    let ts = tmp("tone.ts");
    let reference = tmp("ref.pcm");
    let decoded = tmp("g2g.pcm");
    run(Command::new("ffmpeg")
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-y"])
        .args(["-f", "lavfi", "-i"])
        .arg("aevalsrc=exprs='0.4*sin(440*2*PI*t)|0.4*sin(660*2*PI*t)':s=48000:d=1")
        .args(["-c:a", "ac3", "-b:a", "192k", "-f", "mpegts"])
        .arg(&ts));
    run(Command::new("ffmpeg")
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&ts)
        .args(["-f", "s16le", "-ar", "48000", "-ac", "2"])
        .arg(&reference));
    let line = format!(
        "filesrc location={} ! decodebin ! audioconvert ! \
         audio/x-raw,format=S16LE,rate=48000,channels=2 ! filesink location={}",
        ts.display(),
        decoded.display()
    );
    assert!(run_line(&line).await > 0, "{line}");
    let want = std::fs::read(&reference).expect("reference pcm");
    let got = std::fs::read(&decoded).expect("g2g pcm");
    assert_eq!(
        want.len(),
        got.len(),
        "decoded pcm length differs from ffmpeg's"
    );
    assert_eq!(want, got, "decoded pcm differs from ffmpeg's");
    record_oracle("ac3-in-ts decode byte-identical to ffmpeg CLI (s16 rounding)");
    for p in [ts, reference, decoded] {
        std::fs::remove_file(p).ok();
    }
}
