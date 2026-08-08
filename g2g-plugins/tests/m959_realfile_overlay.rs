//! M959: a real demuxed file's placeholder startup caps refine mid-stream
//! into a strict RGBA-only overlay. The webm's video port fixes nominal
//! 16x16 caps before any byte is parsed (and the track carries no
//! DefaultDuration, so the refined caps flow an `Any` framerate), so the
//! whole chain re-solves at PLAYING; the regression is the caps-driven
//! converter forwarding its raw INPUT caps to the overlay instead of the
//! RGBA it already produces (CapsMismatch). ffmpeg authors the fixture; the
//! test self-skips without it. The demux + decoder are named explicitly to
//! pin this test to the re-solve; the bare-`decodebin` form of the same chain
//! is m960's.
#![cfg(feature = "rav1d")]

use std::process::Command;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl g2g_core::PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn temp_path(tag: &str, ext: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("g2g-m959-{tag}-{}.{ext}", std::process::id()))
}

/// A real 1 s 320x180 AV1 webm encoded by ffmpeg (libaom), or `None` when the
/// host cannot.
fn encode_av1_webm() -> Option<std::path::PathBuf> {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        return None;
    }
    let path = temp_path("clip", "webm");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=30:duration=1",
            "-c:v",
            "libaom-av1",
            "-cpu-used",
            "8",
            "-crf",
            "50",
            "-g",
            "15",
        ])
        .arg(&path)
        .status()
        .ok()?;
    status.success().then_some(path)
}

#[tokio::test]
async fn decodebin_file_refines_caps_into_strict_overlay() {
    let Some(clip) = encode_av1_webm() else {
        eprintln!("skipping: no ffmpeg / no AV1 encoder");
        return;
    };
    let subs = temp_path("subs", "vtt");
    std::fs::write(&subs, "WEBVTT\n\n00:00.000 --> 00:01.000\nhello\n").unwrap();
    let line = format!(
        "filesrc location={} ! matroskademux stream=av1 ! rav1ddec ! videoconvert \
         ! textoverlay location={} ! videoconvert ! fakesink",
        clip.display(),
        subs.display()
    );
    let reg = default_registry();
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    // 30 encoded frames; the decoder's flush may hold back the last one.
    assert!(
        stats.frames_consumed >= 29,
        "the clip's frames reach the sink (got {})",
        stats.frames_consumed
    );
    let _ = std::fs::remove_file(&clip);
    let _ = std::fs::remove_file(&subs);
}
