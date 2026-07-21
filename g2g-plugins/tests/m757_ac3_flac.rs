//! M757 - AC-3 and FLAC audio decode. AC-3 rides MPEG-TS (ATSC stream_type 0x81,
//! or a DVB private PES 0x06 with an AC-3 descriptor) and Matroska (`A_AC3`);
//! FLAC rides Matroska (`A_FLAC`, its `CodecPrivate` forwarded in-band as decoder
//! extradata). Both decode via the generalized `FfmpegAudioDec`.
//!
//! Asserts the parse-time WIRING (the right decoder in the chain); the PCM output
//! is live-validated with `g2g-launch` (FLAC bit-exact vs ffmpeg, AC-3 within the
//! documented 1-LSB float rounding). Needs the ffmpeg decoder pool.

#![cfg(all(feature = "std", feature = "ffmpeg"))]

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::mpegts::{STREAM_TYPE_AC3, STREAM_TYPE_H264, STREAM_TYPE_PRIVATE_PES};
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Run a launch line to completion, returning frames consumed at the sink.
async fn run_line(line: &str) -> u64 {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line:?} should parse: {e:?}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("{line:?} should run: {e:?}"))
        .frames_consumed
}

const TS_SYNC: u8 = 0x47;
const TS_PACKET_LEN: usize = 188;

// --- minimal MPEG-TS section builders (mirroring the m753 helpers) ---
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

const FLAC_STEREO: &[u8] = include_bytes!("fixtures/flac_stereo_48k.mka");
const AC3_STEREO: &[u8] = include_bytes!("fixtures/ac3_stereo_48k.mka");

/// Write a fixture to a temp path so `filesrc` can read it; caller removes it.
fn temp_fixture(tag: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("g2g-m757-{tag}-{}.mka", std::process::id()));
    std::fs::write(&path, bytes).expect("write fixture");
    path
}

fn temp_ts(tag: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("g2g-m757-{tag}-{}.ts", std::process::id()));
    std::fs::write(&path, bytes).expect("write ts");
    path
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

fn bare_decodebin_line(path: &std::path::Path, channels: u8) -> String {
    format!(
        "filesrc location={} ! decodebin ! audioconvert ! \
         audio/x-raw,format=S16LE,rate=48000,channels={channels} ! fakesink",
        path.display()
    )
}

/// An ATSC AC-3 TS (stream_type 0x81) through a bare `decodebin` plugs the ffmpeg
/// audio decoder, not the default video decoder.
#[test]
fn atsc_ac3_ts_bare_decodebin_plugs_audio_decoder() {
    let path = temp_ts(
        "atsc",
        &ts_with_streams(&[(STREAM_TYPE_AC3, 0x0101, Vec::new())]),
    );
    let names = chain_names(&bare_decodebin_line(&path, 2));
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

/// A DVB AC-3 TS (private PES 0x06 with an AC-3 descriptor, tag 0x6A) routes to
/// the same AC-3 selection; a bare 0x06 without the descriptor does not.
#[test]
fn dvb_ac3_descriptor_selects_ac3() {
    let path = temp_ts(
        "dvb",
        &ts_with_streams(&[(STREAM_TYPE_PRIVATE_PES, 0x0101, vec![0x6A, 1, 0x00])]),
    );
    let names = chain_names(&bare_decodebin_line(&path, 2));
    std::fs::remove_file(&path).ok();
    assert!(
        names.iter().any(|n| n == "FfmpegAudioDec"),
        "audio decoder plugged for DVB AC-3: {names:?}"
    );
}

/// An A/V TS (h264 + AC-3) through a bare `decodebin` still plugs the video
/// decoder; the audio-only hook declines when a video track is present.
#[test]
fn av_ts_with_ac3_still_plugs_video_decoder() {
    let path = temp_ts(
        "av",
        &ts_with_streams(&[
            (STREAM_TYPE_H264, 0x0100, Vec::new()),
            (STREAM_TYPE_AC3, 0x0101, Vec::new()),
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
        !names.iter().any(|n| n == "FfmpegAudioDec"),
        "no audio decoder on the A/V default path: {names:?}"
    );
}

/// The explicit selections parse: `tsdemux stream=ac3`, `matroskademux
/// stream=ac3|flac` resolve their property values and build the decode chain.
#[test]
fn explicit_stream_selections_parse() {
    let path = temp_ts(
        "sel",
        &ts_with_streams(&[(STREAM_TYPE_AC3, 0x0101, Vec::new())]),
    );
    let names = chain_names(&format!(
        "filesrc location={} ! tsdemux stream=ac3 ! ffmpegaudiodec ! audioconvert ! \
         audio/x-raw,format=S16LE,rate=48000,channels=2 ! fakesink",
        path.display()
    ));
    std::fs::remove_file(&path).ok();
    assert!(
        names.iter().any(|n| n == "TsDemux") && names.iter().any(|n| n == "FfmpegAudioDec"),
        "explicit ac3 chain builds: {names:?}"
    );
}

/// Stereo FLAC decodes through the real chain to the exact PCM byte count
/// (0.25 s * 48000 * 2 ch * 2 B), with real signal in BOTH interleaved channels.
/// Regression for the packed-sample-layout bug: `plane::<i16>(0)` on a packed
/// frame holds only `samples` elements, so stereo panicked (mono hid it).
#[tokio::test]
async fn stereo_flac_decodes_bit_exact_flow() {
    let out = std::env::temp_dir().join(format!("g2g-m757-flac-{}.raw", std::process::id()));
    let _ = std::fs::remove_file(&out);
    let src = temp_fixture("flacsrc", FLAC_STEREO);
    let line = format!(
        "filesrc location={} bytestream-format=matroska ! \
         matroskademux stream=flac ! ffmpegaudiodec ! audioconvert ! \
         audio/x-raw,format=S16LE,rate=48000,channels=2 ! filesink location={}",
        src.display(),
        out.display()
    );
    assert!(run_line(&line).await > 0, "{line}");
    let pcm = std::fs::read(&out).expect("pcm written");
    assert_eq!(pcm.len(), 48_000 / 4 * 2 * 2, "0.25 s stereo S16");
    // both channels carry the tone (a channel-collapse bug would zero/garble one).
    let mut max = [0i16; 2];
    for (i, ch) in pcm.chunks_exact(2).enumerate() {
        let v = i16::from_le_bytes([ch[0], ch[1]]).unsigned_abs() as i16;
        max[i % 2] = max[i % 2].max(v);
    }
    assert!(
        max[0] > 1000 && max[1] > 1000,
        "both channels live: {max:?}"
    );
    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_file(&src);
}

/// Stereo AC-3 decodes through the real chain (planar-float path) to the frame-
/// quantized byte count with live audio in both channels.
#[tokio::test]
async fn stereo_ac3_decodes_flow() {
    let out = std::env::temp_dir().join(format!("g2g-m757-ac3-{}.raw", std::process::id()));
    let _ = std::fs::remove_file(&out);
    let src = temp_fixture("ac3src", AC3_STEREO);
    let line = format!(
        "filesrc location={} bytestream-format=matroska ! \
         matroskademux stream=ac3 ! ffmpegaudiodec ! audioconvert ! \
         audio/x-raw,format=S16LE,rate=48000,channels=2 ! filesink location={}",
        src.display(),
        out.display()
    );
    assert!(run_line(&line).await > 0, "{line}");
    let pcm = std::fs::read(&out).expect("pcm written");
    // AC-3 frames are 1536 samples; 0.25 s rounds up to 8 frames = 12288 samples.
    assert!(
        pcm.len() >= 12_000 * 4 && pcm.len() <= 13_000 * 4,
        "{}",
        pcm.len()
    );
    let mut max = [0i16; 2];
    for (i, ch) in pcm.chunks_exact(2).enumerate() {
        let v = i16::from_le_bytes([ch[0], ch[1]]).unsigned_abs() as i16;
        max[i % 2] = max[i % 2].max(v);
    }
    assert!(
        max[0] > 1000 && max[1] > 1000,
        "both channels live: {max:?}"
    );
    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_file(&src);
}
