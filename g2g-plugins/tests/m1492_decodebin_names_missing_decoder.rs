//! M1492: a bare `decodebin` on a container whose streams its demuxer cannot
//! forward names the codec, instead of plugging the demuxer's default port.

#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, ParseError};
use g2g_plugins::registry::default_registry;

fn le32(value: u32) -> [u8; 4] {
    value.to_le_bytes()
}

fn riff_chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = id.to_vec();
    out.extend_from_slice(&le32(body.len() as u32));
    out.extend_from_slice(body);
    if body.len() % 2 == 1 {
        out.push(0);
    }
    out
}

fn riff_list(form: &[u8; 4], body: &[u8]) -> Vec<u8> {
    riff_chunk(b"LIST", &[form.as_slice(), body].concat())
}

// one AASC video stream, its RIFF size claiming more than the file holds
fn truncated_aasc_avi() -> Vec<u8> {
    const STREAM_COUNT_AT: usize = 24;
    const STRH_LEN: usize = 56;
    const BITMAPINFO_LEN: usize = 40;
    const TRUNCATED_MOVI_LEN: u32 = 1 << 20;
    let mut avih = vec![0u8; 56];
    avih[STREAM_COUNT_AT..STREAM_COUNT_AT + 4].copy_from_slice(&le32(1));
    let mut strh = vec![0u8; STRH_LEN];
    strh[0..4].copy_from_slice(b"vids");
    strh[4..8].copy_from_slice(b"AASC");
    strh[20..24].copy_from_slice(&le32(1));
    strh[24..28].copy_from_slice(&le32(25));
    let mut strf = vec![0u8; BITMAPINFO_LEN];
    strf[0..4].copy_from_slice(&le32(BITMAPINFO_LEN as u32));
    strf[4..8].copy_from_slice(&le32(320));
    strf[8..12].copy_from_slice(&le32(176));
    strf[16..20].copy_from_slice(b"AASC");
    let strl = riff_list(
        b"strl",
        &[riff_chunk(b"strh", &strh), riff_chunk(b"strf", &strf)].concat(),
    );
    let hdrl = riff_list(b"hdrl", &[riff_chunk(b"avih", &avih), strl].concat());
    let mut movi_header = b"LIST".to_vec();
    movi_header.extend_from_slice(&le32(TRUNCATED_MOVI_LEN));
    movi_header.extend_from_slice(b"movi");
    let body = [b"AVI ".as_slice(), &hdrl, &movi_header].concat();
    let mut out = b"RIFF".to_vec();
    out.extend_from_slice(&le32(body.len() as u32 + TRUNCATED_MOVI_LEN));
    out.extend_from_slice(&body);
    out
}

fn ebml_size(value: usize) -> Vec<u8> {
    const EIGHT_BYTE_SIZE_MARKER: u8 = 0x01;
    let mut out = vec![EIGHT_BYTE_SIZE_MARKER];
    out.extend_from_slice(&(value as u64).to_be_bytes()[1..]);
    out
}

fn ebml(id: &[u8], body: &[u8]) -> Vec<u8> {
    [id, &ebml_size(body.len()), body].concat()
}

fn mkv_with_track(codec_id: &[u8]) -> Vec<u8> {
    let entry = [
        ebml(&[0xD7], &[1]),
        ebml(&[0x83], &[1]),
        ebml(&[0x86], codec_id),
    ]
    .concat();
    let tracks = ebml(&[0x16, 0x54, 0xAE, 0x6B], &ebml(&[0xAE], &entry));
    let segment = ebml(&[0x18, 0x53, 0x80, 0x67], &tracks);
    [ebml(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat()
}

fn mp4_box(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = ((body.len() + 8) as u32).to_be_bytes().to_vec();
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
    out
}

// one sound track whose sample entry is `sowt`
fn mov_with_sowt_track() -> Vec<u8> {
    const TKHD_V0_LEN: usize = 84;
    const MDHD_V0_LEN: usize = 24;
    const TIMESCALE: u32 = 44_100;
    let mut tkhd = vec![0u8; TKHD_V0_LEN];
    tkhd[12..16].copy_from_slice(&1u32.to_be_bytes());
    let mut mdhd = vec![0u8; MDHD_V0_LEN];
    mdhd[12..16].copy_from_slice(&TIMESCALE.to_be_bytes());
    let mut hdlr = vec![0u8; 24];
    hdlr[8..12].copy_from_slice(b"soun");
    let entry = mp4_box(b"sowt", &[0u8; 28]);
    let stsd = mp4_box(
        b"stsd",
        &[[0, 0, 0, 0, 0, 0, 0, 1].as_slice(), &entry].concat(),
    );
    let stbl = mp4_box(b"stbl", &stsd);
    let minf = mp4_box(b"minf", &stbl);
    let mdia = mp4_box(
        b"mdia",
        &[mp4_box(b"mdhd", &mdhd), mp4_box(b"hdlr", &hdlr), minf].concat(),
    );
    let trak = mp4_box(b"trak", &[mp4_box(b"tkhd", &tkhd), mdia].concat());
    let moov = mp4_box(b"moov", &trak);
    let ftyp = mp4_box(b"ftyp", b"qt  \0\0\0\0qt  ");
    [ftyp, moov].concat()
}

fn missing_decoders(tag: &str, ext: &str, bytes: &[u8], consumer: &str) -> Vec<String> {
    let path = std::env::temp_dir().join(format!("g2g-m1492-{tag}-{}.{ext}", std::process::id()));
    std::fs::write(&path, bytes).expect("write container");
    let line = format!(
        "filesrc location={} ! decodebin ! {consumer} ! fakesink",
        path.display()
    );
    let parsed = parse_launch(&default_registry(), &line);
    std::fs::remove_file(&path).ok();
    match parsed {
        Err(ParseError::NoStreamDecoder { codecs, .. }) => codecs,
        Err(other) => panic!("`{line}` fails naming no codec: {other}"),
        Ok(_) => panic!("`{line}` builds a graph"),
    }
}

#[test]
fn truncated_avi_names_its_video_codec() {
    let codecs = missing_decoders("aasc", "avi", &truncated_aasc_avi(), "videoconvert");
    assert_eq!(codecs, vec!["AASC".to_string()]);
}

#[test]
fn matroska_names_its_codec_id() {
    let codecs = missing_decoders(
        "rv40",
        "mkv",
        &mkv_with_track(b"V_REAL/RV40"),
        "videoconvert",
    );
    assert_eq!(codecs, vec!["V_REAL/RV40".to_string()]);
}

#[test]
fn mov_names_its_sample_entry() {
    let codecs = missing_decoders("sowt", "mov", &mov_with_sowt_track(), "audioconvert");
    assert_eq!(codecs, vec!["sowt".to_string()]);
}
