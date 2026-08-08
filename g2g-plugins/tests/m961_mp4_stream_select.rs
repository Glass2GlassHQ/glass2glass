//! M961: the MP4 analog of M960. `qtdemux`'s port default is nominal H.264, so
//! an AV1 mp4 with only a codec-specific decoder compiled in (`rav1ddec`) used
//! to plug that decoder against nominal H.264 caps and fail startup
//! negotiation; the primary-stream hook now names the first video track's codec
//! (`stream=av1`), making the port's startup caps truthful. The parse-level
//! tests build a synthetic `moov` (no ffmpeg needed); the decode run uses an
//! ffmpeg-authored clip and self-skips without it. The hook probes a 4 MiB
//! prefix, so the tiny clip's trailing `moov` is always inside it.
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
    std::env::temp_dir().join(format!("g2g-m961-{tag}-{}.{ext}", std::process::id()))
}

fn mp4_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    out
}
fn full_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    // version 0, flags 0
    mp4_box(kind, &[&[0u8; 4], payload].concat())
}
// tkhd v0: track_ID at payload offset 12, width/height as 16.16 at 76/80.
fn tkhd(track_id: u32, w: u32, h: u32) -> Vec<u8> {
    let mut c = vec![0u8; 80];
    c[8..12].copy_from_slice(&track_id.to_be_bytes());
    c[72..76].copy_from_slice(&(w << 16).to_be_bytes());
    c[76..80].copy_from_slice(&(h << 16).to_be_bytes());
    full_box(b"tkhd", &c)
}
// mdhd v0: timescale at payload offset 12, duration at 16.
fn mdhd(timescale: u32) -> Vec<u8> {
    let mut c = vec![0u8; 16];
    c[8..12].copy_from_slice(&timescale.to_be_bytes());
    full_box(b"mdhd", &c)
}
// hdlr: handler_type at payload offset 8.
fn hdlr(handler: &[u8; 4]) -> Vec<u8> {
    let mut c = vec![0u8; 20];
    c[4..8].copy_from_slice(handler);
    full_box(b"hdlr", &c)
}
fn trak(tkhd: &[u8], mdhd: &[u8], hdlr: &[u8], sample_entry: &[u8]) -> Vec<u8> {
    let mut stsd_payload = 1u32.to_be_bytes().to_vec();
    stsd_payload.extend_from_slice(sample_entry);
    let stsd = full_box(b"stsd", &stsd_payload);
    let minf = mp4_box(b"minf", &mp4_box(b"stbl", &stsd));
    let mdia = mp4_box(b"mdia", &[mdhd, hdlr, &minf].concat());
    mp4_box(b"trak", &[tkhd, &mdia].concat())
}

/// An MP4 header whose `moov` carries one AV1 video track. Only the header is
/// needed: the primary-stream hook probes the `moov`, not samples.
fn mp4_with_av1_video() -> Vec<u8> {
    // av01 sample entry: 78 fixed bytes, then an av1C with its 4 fixed bytes.
    let av01 = {
        let mut p = vec![0u8; 78];
        p.extend_from_slice(&mp4_box(b"av1C", &[0x81, 0x00, 0x00, 0x00]));
        mp4_box(b"av01", &p)
    };
    let video = trak(&tkhd(1, 320, 180), &mdhd(90_000), &hdlr(b"vide"), &av01);
    let moov = mp4_box(b"moov", &video);
    [mp4_box(b"ftyp", b"isom\x00\x00\x02\x00isomiso2"), moov].concat()
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
    let path = temp_path("av1-header", "mp4");
    std::fs::write(&path, mp4_with_av1_video()).unwrap();
    let line = format!("filesrc location={} ! decodebin ! fakesink", path.display());
    let names = chain_names(&line);
    assert!(
        names.iter().any(|n| n == "Rav1dDec"),
        "the AV1 decoder is plugged from the selected track, got {names:?}"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn bare_decodebin_still_selects_an_audio_only_track() {
    // No video track: the hook keeps the M748 audio-only selection, so the
    // chain search starts from the selected AAC caps (with or without an AAC
    // decoder in this build), never from the container caps.
    let mp4a = {
        let mut p = vec![0u8; 28];
        p[16..18].copy_from_slice(&2u16.to_be_bytes());
        // esds: ES descriptor > decoder config (OTI 0x40 = AAC) > DSI
        let dsi = [0x05, 0x02, 0x12, 0x10];
        let mut dcd_body = vec![0u8; 13];
        dcd_body[0] = 0x40;
        dcd_body.extend_from_slice(&dsi);
        let mut dcd = vec![0x04, dcd_body.len() as u8];
        dcd.extend_from_slice(&dcd_body);
        let mut es_body = vec![0u8; 3];
        es_body.extend_from_slice(&dcd);
        let mut es = vec![0x03, es_body.len() as u8];
        es.extend_from_slice(&es_body);
        p.extend_from_slice(&full_box(b"esds", &es));
        mp4_box(b"mp4a", &p)
    };
    let audio = trak(&tkhd(1, 0, 0), &mdhd(48_000), &hdlr(b"soun"), &mp4a);
    let moov = mp4_box(b"moov", &audio);
    let file = [mp4_box(b"ftyp", b"isom\x00\x00\x02\x00isomiso2"), moov].concat();

    let path = temp_path("aac-header", "mp4");
    std::fs::write(&path, file).unwrap();
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
                names.iter().any(|n| n.contains("Aac")),
                "the audio track's decoder is plugged, got {names:?}"
            );
        }
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("Aac"),
                "the chain search starts from the selected AAC track: {msg}"
            );
        }
    }
    let _ = std::fs::remove_file(&path);
}

/// A real AV1 mp4 decodes end to end through the bare `decodebin` line with
/// only the codec-specific pure-Rust decoder compiled in.
#[tokio::test]
async fn bare_decodebin_decodes_a_real_av1_mp4() {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let clip = temp_path("clip", "mp4");
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
