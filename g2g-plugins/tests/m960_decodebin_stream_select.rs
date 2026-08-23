//! M960: a bare `decodebin` names the demuxer's `stream=` selection from the
//! file's actual primary track. The Matroska port's parameterless default is
//! VP9, so an AV1 webm with only a codec-specific decoder compiled in
//! (`rav1ddec`) used to plug that decoder against nominal VP9 caps and fail
//! startup negotiation; the primary-stream hook now selects the first video
//! track by codec, making the port's startup caps truthful. The parse-level
//! tests build a synthetic Matroska header (no ffmpeg needed); the decode run
//! uses an ffmpeg-authored clip and self-skips without it.
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
    std::env::temp_dir().join(format!("g2g-m960-{tag}-{}.{ext}", std::process::id()))
}

fn vint(value: u64) -> Vec<u8> {
    let mut len = 1usize;
    while len < 8 && value >= (1u64 << (7 * len)) - 1 {
        len += 1;
    }
    let mut out = vec![0u8; len];
    let mut v = value;
    for i in (0..len).rev() {
        out[i] = (v & 0xFF) as u8;
        v >>= 8;
    }
    out[0] |= 1 << (8 - len);
    out
}
fn elem(id: &[u8], body: &[u8]) -> Vec<u8> {
    let mut out = id.to_vec();
    out.extend_from_slice(&vint(body.len() as u64));
    out.extend_from_slice(body);
    out
}
fn uint_body(v: u64) -> Vec<u8> {
    if v == 0 {
        return vec![0];
    }
    let mut bytes = v.to_be_bytes().to_vec();
    while bytes.len() > 1 && bytes[0] == 0 {
        bytes.remove(0);
    }
    bytes
}
fn video_track(num: u64, codec: &[u8], w: u32, h: u32) -> Vec<u8> {
    let v = [
        elem(&[0xB0], &uint_body(w as u64)),
        elem(&[0xBA], &uint_body(h as u64)),
    ]
    .concat();
    let body = [
        elem(&[0xD7], &uint_body(num)),
        elem(&[0x83], &uint_body(1)),
        elem(&[0x86], codec),
        elem(&[0xE0], &v),
    ]
    .concat();
    elem(&[0xAE], &body)
}

/// An MKV header whose `Tracks` carries one video track of `codec`. Only the
/// header is needed: the primary-stream hook probes `Tracks`, not clusters.
fn mkv_with_video(codec: &[u8]) -> Vec<u8> {
    let tracks_body = video_track(1, codec, 320, 180);
    let tracks = elem(&[0x16, 0x54, 0xAE, 0x6B], &tracks_body);
    let segment = elem(&[0x18, 0x53, 0x80, 0x67], &tracks);
    [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat()
}

fn chain_names(line: &str) -> Vec<String> {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let vg = graph.finish().expect("valid graph");
    vg.topo()
        .iter()
        .filter_map(|&n| vg.element(n).map(|e| e.log_category().to_string()))
        .collect()
}

#[test]
fn bare_decodebin_selects_the_av1_track() {
    let path = temp_path("av1-header", "mkv");
    std::fs::write(&path, mkv_with_video(b"V_AV1")).unwrap();
    let line = format!("filesrc location={} ! decodebin ! fakesink", path.display());
    let names = chain_names(&line);
    // either AV1 decoder proves the track was selected; dav1ddec outranks
    // rav1ddec when both are built
    assert!(
        names.iter().any(|n| n == "Rav1dDec" || n == "Dav1dDec"),
        "an AV1 decoder is plugged from the selected track, got {names:?}"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn bare_decodebin_still_selects_an_audio_only_track() {
    // No video track: the hook keeps the M757 audio-only selection. This build
    // carries no Opus decoder, so the chain search fails, but the failure must
    // name the SELECTED Opus caps (the pre-hook decline would have searched
    // from the container caps instead).
    let path = temp_path("opus-header", "mka");
    let body = [
        elem(&[0xD7], &uint_body(1)),
        elem(&[0x83], &uint_body(2)),
        elem(&[0x86], b"A_OPUS"),
    ]
    .concat();
    let tracks = elem(&[0x16, 0x54, 0xAE, 0x6B], &elem(&[0xAE], &body));
    let segment = elem(&[0x18, 0x53, 0x80, 0x67], &tracks);
    let mkv = [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat();
    std::fs::write(&path, mkv).unwrap();
    let line = format!("filesrc location={} ! decodebin ! fakesink", path.display());
    let reg = default_registry();
    match parse_launch(&reg, &line) {
        Ok(graph) => {
            let vg = graph.finish().expect("valid graph");
            let names: Vec<String> = vg
                .topo()
                .iter()
                .filter_map(|&n| vg.element(n).map(|e| e.log_category().to_string()))
                .collect();
            assert!(
                names.iter().any(|n| n.contains("Opus")),
                "the audio track's decoder is plugged, got {names:?}"
            );
        }
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("Opus"),
                "the chain search starts from the selected Opus track: {msg}"
            );
        }
    }
    let _ = std::fs::remove_file(&path);
}

/// A real AV1 webm decodes end to end through the bare `decodebin` line with
/// only the codec-specific pure-Rust decoder compiled in.
#[tokio::test]
async fn bare_decodebin_decodes_a_real_av1_webm() {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let clip = temp_path("clip", "webm");
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
        .arg(&clip)
        .status();
    if !status.map(|s| s.success()).unwrap_or(false) {
        eprintln!("skipping: no AV1 encoder");
        return;
    }
    let line = format!(
        "filesrc location={} ! decodebin ! videoconvert ! fakesink",
        clip.display()
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
}
