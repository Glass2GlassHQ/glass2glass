//! M776: a mid-stream `CapsChanged` whose forward re-solve DEFERS (the
//! element's derived output has several alternatives, e.g. a decoder offering
//! S16LE | F32LE) must not feed the incoming INPUT caps to `configure_output`,
//! whose contract is output caps only. A strict transform (`OpusDec` rejects
//! anything but its PCM formats) failed the whole run with `CapsMismatch` when
//! the demuxer refined its caps, so no Ogg-Opus file played through `run_graph`.
#![cfg(all(feature = "std", feature = "opus"))]

use std::process::Command;

use g2g_core::runtime::parse_launch;
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl g2g_core::PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// An Ogg-Opus tone from ffmpeg (libopus), or `None` when the host cannot.
fn encode_ogg_opus(tag: &str) -> Option<std::path::PathBuf> {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        return None;
    }
    let path = std::env::temp_dir().join(format!("g2g-m776-{tag}-{}.opus", std::process::id()));
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.5:sample_rate=48000",
            "-ac",
            "2",
            "-c:a",
            "libopus",
            "-f",
            "ogg",
        ])
        .arg(&path)
        .status()
        .ok()?;
    status.success().then_some(path)
}

/// The demuxer's mid-stream refine (`audio/x-opus` with the real channel count)
/// reaches `OpusDec` through the transform arm and the run completes; before the
/// fix the arm called `configure_output(audio/x-opus)` and the run died.
#[tokio::test]
async fn ogg_opus_chain_survives_midstream_refine() {
    let Some(src) = encode_ogg_opus("chain") else {
        eprintln!("skipping: no ffmpeg / libopus");
        return;
    };
    let reg = default_registry();
    let line = format!(
        "filesrc location={} ! oggdemux ! opusdec ! audioconvert ! \
         audio/x-raw,format=S16LE,rate=48000,channels=2 ! fakesink",
        src.display()
    );
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let stats = g2g_core::runtime::run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    let _ = std::fs::remove_file(&src);
    assert!(stats.frames_consumed > 0, "PCM reached the sink");
}

/// The same stream through a lone `playbin uri=` (the M775 audio hook + this
/// fix together make `.opus` files playable end to end).
#[tokio::test]
async fn playbin_plays_ogg_opus() {
    let Some(src) = encode_ogg_opus("playbin") else {
        eprintln!("skipping: no ffmpeg / libopus");
        return;
    };
    let reg = default_registry();
    let line = format!("playbin uri=file://{}", src.display());
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let stats = g2g_core::runtime::run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    let _ = std::fs::remove_file(&src);
    assert!(stats.frames_consumed > 0, "PCM reached the sink");
}
