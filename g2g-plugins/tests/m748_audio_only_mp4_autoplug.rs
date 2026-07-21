//! M748 - bare `decodebin` auto-plugs an audio-only MP4. The single-stream
//! [`Mp4Demux`](g2g_plugins::mp4demux) fixes its output pad before parsing any
//! byte, so it defaults to a video port; on an audio-only `.m4a`, `filesrc
//! location=X.m4a ! decodebin ! audioconvert ! ...` would plug a video decoder and
//! fail "no caps overlap". A primary-stream hook sniffs the `moov` and, finding no
//! video track, selects the demux's audio stream (`qtdemux stream=aac`), so the
//! auto-plug builds `qtdemux stream=aac ! aacparse ! <audio decoder> ! ...`, the
//! MP4 sibling of the M746 MPEG-TS hook.
//!
//! Asserts the parse-time WIRING (an audio decoder in the chain, no video decoder);
//! the end-to-end PCM output is live-validated with `g2g-launch`. Fixtures are tiny
//! ffmpeg clips (a `moov` cannot be hand-synthesized like an MPEG-TS PMT).

#![cfg(all(feature = "std", feature = "ffmpeg"))]

use std::path::PathBuf;

use g2g_core::runtime::parse_launch;
use g2g_plugins::registry::default_registry;

const AUDIO_ONLY: &[u8] = include_bytes!("fixtures/audio_aac_48000.m4a");
const AV: &[u8] = include_bytes!("fixtures/av_h264_aac.mp4");

/// Write `bytes` to a unique temp path with the given extension; caller removes it.
fn temp(tag: &str, ext: &str, bytes: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!("g2g-m748-{tag}-{}.{ext}", std::process::id()));
    std::fs::write(&path, bytes).expect("write fixture");
    path
}

/// The element type names in the built graph, in topological order.
fn chain_names(line: &str) -> Vec<String> {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let vg = graph.finish().expect("valid graph");
    vg.topo()
        .iter()
        .filter_map(|&n| vg.element(n).map(|e| e.log_category().to_string()))
        .collect()
}

/// An audio-only MP4 through a bare `decodebin` plugs an AUDIO decoder (not the
/// default video decoder): the primary-stream hook sniffs the `moov`, finds no
/// video, and selects `qtdemux`'s AAC stream.
#[test]
fn audio_only_mp4_bare_decodebin_plugs_audio_decoder() {
    let path = temp("aac", "m4a", AUDIO_ONLY);
    let names = chain_names(&format!(
        "filesrc location={} ! decodebin ! audioconvert ! \
         audio/x-raw,format=S16LE,rate=48000,channels=1 ! fakesink",
        path.display()
    ));
    std::fs::remove_file(&path).ok();

    assert!(
        names.iter().any(|n| n == "Mp4Demux"),
        "single-stream qtdemux was plugged: {names:?}"
    );
    assert!(
        names.iter().any(|n| n == "FfmpegAudioDec"),
        "an audio decoder was plugged for the audio-only stream: {names:?}"
    );
    assert!(
        !names.iter().any(|n| n == "FfmpegH264Dec"),
        "no video decoder was plugged (the audio stream was selected): {names:?}"
    );
}

/// An A/V MP4 through a bare `decodebin` still plugs the VIDEO decoder (the hook
/// declines when a video track is present, leaving the demux's default video port):
/// the M748 change does not alter existing A/V behavior.
#[test]
fn av_mp4_bare_decodebin_still_plugs_video_decoder() {
    let path = temp("av", "mp4", AV);
    let names = chain_names(&format!(
        "filesrc location={} ! decodebin ! videoconvert ! \
         video/x-raw,format=I420 ! fakesink",
        path.display()
    ));
    std::fs::remove_file(&path).ok();

    assert!(
        names.iter().any(|n| n == "FfmpegH264Dec"),
        "a video decoder is still plugged for the A/V container: {names:?}"
    );
    assert!(
        !names.iter().any(|n| n == "FfmpegAudioDec"),
        "the video path is unchanged (no audio decoder spliced): {names:?}"
    );
}
