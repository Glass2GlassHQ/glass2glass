//! M746 - bare `decodebin` auto-plugs an audio-only container. A single-stream
//! demux fixes its output pad before parsing any byte, so it defaults to a video
//! port; on an audio-only MPEG-TS `filesrc location=X.ts ! decodebin ! audioconvert
//! ! ...` would plug a video decoder and fail "no caps overlap". A primary-stream
//! hook sniffs the PMT and, finding no video track, selects the demux's audio
//! stream, so the auto-plug builds `tsdemux stream=aac ! <audio decoder> ! ...`.
//!
//! Asserts the parse-time WIRING (an audio decoder in the chain, no video decoder);
//! the end-to-end PCM output is live-validated with `g2g-launch`. Needs the audio
//! decoder in the autoplug pool (ffmpeg).

#![cfg(all(feature = "std", feature = "ffmpeg"))]

use g2g_core::runtime::parse_launch;
use g2g_plugins::mpegts::{STREAM_TYPE_AAC, STREAM_TYPE_H264};
use g2g_plugins::registry::default_registry;

const TS_SYNC: u8 = 0x47;
const TS_PACKET_LEN: usize = 188;

// --- minimal MPEG-TS section builders (mirroring the m388 tsdemux helpers) ---
fn ts_packet(pid: u16, pusi: bool, payload: &[u8]) -> Vec<u8> {
    const ROOM: usize = TS_PACKET_LEN - 4;
    let mut p = vec![0u8; TS_PACKET_LEN];
    p[0] = TS_SYNC;
    p[1] = if pusi { 0x40 } else { 0x00 } | ((pid >> 8) as u8 & 0x1F);
    p[2] = (pid & 0xFF) as u8;
    let l = payload.len();
    if l == ROOM {
        p[3] = 0x10;
        p[4..].copy_from_slice(payload);
    } else {
        p[3] = 0x30;
        let af_len = ROOM - 1 - l;
        p[4] = af_len as u8;
        if af_len >= 1 {
            p[5] = 0x00;
            for b in p.iter_mut().take(6 + (af_len - 1)).skip(6) {
                *b = 0xFF;
            }
        }
        p[5 + af_len..].copy_from_slice(payload);
    }
    p
}
fn psi(pid: u16, table_id: u8, body: &[u8]) -> Vec<u8> {
    let section_length = body.len() + 4;
    let mut s = vec![
        table_id,
        0xB0 | ((section_length >> 8) as u8 & 0x0F),
        (section_length & 0xFF) as u8,
    ];
    s.extend_from_slice(body);
    s.extend_from_slice(&[0, 0, 0, 0]);
    let mut payload = vec![0u8];
    payload.extend_from_slice(&s);
    ts_packet(pid, true, &payload)
}
fn pat(pmt_pid: u16) -> Vec<u8> {
    psi(
        0x0000,
        0x00,
        &[
            0,
            1,
            0xC1,
            0,
            0,
            0,
            1,
            0xE0 | (pmt_pid >> 8) as u8 & 0x1F,
            pmt_pid as u8,
        ],
    )
}
/// One ES entry for a PMT: `stream_type` on `pid`.
fn es(stream_type: u8, pid: u16) -> Vec<u8> {
    vec![
        stream_type,
        0xE0 | (pid >> 8) as u8 & 0x1F,
        pid as u8,
        0xF0,
        0x00,
    ]
}
/// A PMT declaring the given elementary streams; the first `pid` is the PCR PID.
fn pmt(streams: &[(u8, u16)]) -> Vec<u8> {
    let pcr = streams.first().map(|&(_, p)| p).unwrap_or(0x0100);
    let mut body = vec![
        0x00,
        0x01,
        0xC1,
        0x00,
        0x00,
        0xE0 | (pcr >> 8) as u8 & 0x1F,
        pcr as u8,
        0xF0,
        0x00,
    ];
    for &(ty, pid) in streams {
        body.extend_from_slice(&es(ty, pid));
    }
    psi(0x1000, 0x02, &body)
}
/// A transport stream whose PMT declares `streams` (PMT-only is enough: the demux
/// discovers forwardable streams from the PMT, no PES payload needed for the sniff).
fn ts_with_streams(streams: &[(u8, u16)]) -> Vec<u8> {
    let mut ts = pat(0x1000);
    ts.extend_from_slice(&pmt(streams));
    ts
}

/// Write `bytes` to a unique temp `.ts` path; caller removes it.
fn temp_ts(tag: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("g2g-m746-{tag}-{}.ts", std::process::id()));
    std::fs::write(&path, bytes).expect("write ts");
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

/// An audio-only MPEG-TS through a bare `decodebin` plugs an AUDIO decoder (not the
/// default video decoder): the primary-stream hook sniffs the PMT, finds no video,
/// and selects `tsdemux`'s AAC stream.
#[test]
fn audio_only_ts_bare_decodebin_plugs_audio_decoder() {
    let path = temp_ts("aac", &ts_with_streams(&[(STREAM_TYPE_AAC, 0x0101)]));
    let names = chain_names(&format!(
        "filesrc location={} ! decodebin ! audioconvert ! \
         audio/x-raw,format=S16LE,rate=48000,channels=1 ! fakesink",
        path.display()
    ));
    std::fs::remove_file(&path).ok();

    assert!(
        names.iter().any(|n| n == "TsDemux"),
        "single-stream tsdemux was plugged: {names:?}"
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

/// An A/V MPEG-TS through a bare `decodebin` still plugs the VIDEO decoder (the hook
/// declines when a video track is present, leaving the demux's default video port):
/// the M746 change does not alter existing A/V behavior.
#[test]
fn av_ts_bare_decodebin_still_plugs_video_decoder() {
    let path = temp_ts(
        "av",
        &ts_with_streams(&[(STREAM_TYPE_H264, 0x0100), (STREAM_TYPE_AAC, 0x0101)]),
    );
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
