//! M753 - MPEG-TS audio beyond AAC: MPEG audio (`mp2`, stream_type 0x03/0x04)
//! and Opus (private PES 0x06 with an 'Opus' registration descriptor). The demux
//! forwards mp2 frames raw and unwraps Opus control-header AUs; the primary-stream
//! hook selects them on audio-only content like it does AAC.
//!
//! Asserts the parse-time WIRING (the right decoder in the chain); the end-to-end
//! PCM output is live-validated with `g2g-launch` (mp2 bit-exact vs ffmpeg, Opus
//! bit-exact vs ffmpeg's TS decode). Needs the ffmpeg + opus decoder pool.

#![cfg(all(feature = "std", feature = "ffmpeg", feature = "opus"))]

use g2g_core::runtime::parse_launch;
use g2g_plugins::mpegts::{
    STREAM_TYPE_H264, STREAM_TYPE_MPEG1_AUDIO, STREAM_TYPE_MPEG2_AUDIO, STREAM_TYPE_PRIVATE_PES,
};
use g2g_plugins::registry::default_registry;

const TS_SYNC: u8 = 0x47;
const TS_PACKET_LEN: usize = 188;

// --- minimal MPEG-TS section builders (mirroring the m746 helpers) ---
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
/// One ES entry for a PMT: `stream_type` on `pid` with raw ES descriptors.
fn es(stream_type: u8, pid: u16, descriptors: &[u8]) -> Vec<u8> {
    let mut e = vec![
        stream_type,
        0xE0 | (pid >> 8) as u8 & 0x1F,
        pid as u8,
        0xF0 | ((descriptors.len() >> 8) as u8 & 0x0F),
        (descriptors.len() & 0xFF) as u8,
    ];
    e.extend_from_slice(descriptors);
    e
}
/// The DVB Opus carriage descriptors: registration 'Opus' + extension channel code.
fn opus_descriptors(channels: u8) -> Vec<u8> {
    vec![0x05, 4, b'O', b'p', b'u', b's', 0x7F, 2, 0x80, channels]
}
/// A PMT declaring the given elementary streams (type, pid, descriptors).
fn pmt(streams: &[(u8, u16, Vec<u8>)]) -> Vec<u8> {
    let pcr = streams.first().map(|&(_, p, _)| p).unwrap_or(0x0100);
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
    for (ty, pid, desc) in streams {
        body.extend_from_slice(&es(*ty, *pid, desc));
    }
    psi(0x1000, 0x02, &body)
}
fn ts_with_streams(streams: &[(u8, u16, Vec<u8>)]) -> Vec<u8> {
    let mut ts = pat(0x1000);
    ts.extend_from_slice(&pmt(streams));
    ts
}

/// Write `bytes` to a unique temp `.ts` path; caller removes it.
fn temp_ts(tag: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("g2g-m753-{tag}-{}.ts", std::process::id()));
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

fn bare_decodebin_line(path: &std::path::Path) -> String {
    format!(
        "filesrc location={} ! decodebin ! audioconvert ! \
         audio/x-raw,format=S16LE,rate=48000,channels=1 ! fakesink",
        path.display()
    )
}

/// An mp2-only MPEG-TS (stream_type 0x03) through a bare `decodebin` plugs the
/// ffmpeg audio decoder, not the default video decoder.
#[test]
fn mp2_only_ts_bare_decodebin_plugs_audio_decoder() {
    let path = temp_ts(
        "mp2",
        &ts_with_streams(&[(STREAM_TYPE_MPEG1_AUDIO, 0x0101, Vec::new())]),
    );
    let names = chain_names(&bare_decodebin_line(&path));
    std::fs::remove_file(&path).ok();
    assert!(
        names.iter().any(|n| n == "FfmpegAudioDec"),
        "audio decoder plugged: {names:?}"
    );
    assert!(
        !names.iter().any(|n| n.contains("H264Dec")),
        "no video decoder: {names:?}"
    );
}

/// The MPEG-2 audio stream_type (0x04) routes to the same mp2 selection.
#[test]
fn mpeg2_audio_stream_type_also_selects_mp2() {
    let path = temp_ts(
        "mp2v2",
        &ts_with_streams(&[(STREAM_TYPE_MPEG2_AUDIO, 0x0101, Vec::new())]),
    );
    let names = chain_names(&bare_decodebin_line(&path));
    std::fs::remove_file(&path).ok();
    assert!(
        names.iter().any(|n| n == "FfmpegAudioDec"),
        "audio decoder plugged: {names:?}"
    );
}

/// An Opus-only MPEG-TS (private PES + 'Opus' registration) through a bare
/// `decodebin` plugs OpusDec.
#[test]
fn opus_only_ts_bare_decodebin_plugs_opusdec() {
    let path = temp_ts(
        "opus",
        &ts_with_streams(&[(STREAM_TYPE_PRIVATE_PES, 0x0101, opus_descriptors(2))]),
    );
    let names = chain_names(&bare_decodebin_line(&path));
    std::fs::remove_file(&path).ok();
    assert!(
        names.iter().any(|n| n == "OpusDec"),
        "OpusDec plugged: {names:?}"
    );
}

/// A private PES stream WITHOUT the 'Opus' registration is not forwarded: the
/// sniff finds no forwardable stream and declines, so the default video chain is
/// built (and then fails caps negotiation at startup, live-validated). No audio
/// decoder must be plugged off unidentified private data.
#[test]
fn private_pes_without_opus_registration_is_not_selected() {
    let path = temp_ts(
        "priv",
        &ts_with_streams(&[(STREAM_TYPE_PRIVATE_PES, 0x0101, Vec::new())]),
    );
    let names = chain_names(&bare_decodebin_line(&path));
    std::fs::remove_file(&path).ok();
    assert!(
        !names
            .iter()
            .any(|n| n == "OpusDec" || n == "FfmpegAudioDec"),
        "no audio decoder off unidentified private data: {names:?}"
    );
    assert!(
        names.iter().any(|n| n.contains("H264Dec")),
        "hook declined to the default video path: {names:?}"
    );
}

/// An A/V TS (h264 + mp2) through a bare `decodebin` still plugs the video
/// decoder; the audio-only hook declines when a video track is present.
#[test]
fn av_ts_with_mp2_still_plugs_video_decoder() {
    let path = temp_ts(
        "av",
        &ts_with_streams(&[
            (STREAM_TYPE_H264, 0x0100, Vec::new()),
            (STREAM_TYPE_MPEG1_AUDIO, 0x0101, Vec::new()),
        ]),
    );
    let names = chain_names(&format!(
        "filesrc location={} ! decodebin ! fakesink",
        path.display()
    ));
    std::fs::remove_file(&path).ok();
    assert!(
        names.iter().any(|n| n.contains("H264Dec")),
        "video decoder plugged: {names:?}"
    );
    assert!(
        !names
            .iter()
            .any(|n| n == "FfmpegAudioDec" || n == "OpusDec"),
        "no audio decoder on the A/V default path: {names:?}"
    );
}
