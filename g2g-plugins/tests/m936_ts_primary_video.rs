//! M936 - `decodebin` on MPEG-TS selects the PMT's video codec. `TsDemux` fixes
//! its output pad before parsing a byte (defaulting to H.264), so the
//! primary-stream sniff must name the actual video stream: previously it
//! declined any video-bearing TS, and an MPEG-2 (or H.265) transport stream
//! negotiated an H.264 decoder and failed `NotConfigured`.

#![cfg(feature = "std")]

use g2g_core::{ByteStreamEncoding, Caps};
use g2g_plugins::mpegts::{STREAM_TYPE_H264, STREAM_TYPE_MPEG1_AUDIO, STREAM_TYPE_MPEG2_VIDEO};
use g2g_plugins::uridecodebin::ts_primary_stream;

const TS_SYNC: u8 = 0x47;
const TS_PACKET_LEN: usize = 188;

// --- minimal MPEG-TS section builders (mirroring the m753 / m757 helpers) ---
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
fn es(stream_type: u8, pid: u16) -> Vec<u8> {
    vec![
        stream_type,
        0xE0 | (pid >> 8) as u8 & 0x1F,
        pid as u8,
        0xF0,
        0x00,
    ]
}
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

/// Write a PAT+PMT-only transport stream and return the sniff result.
fn sniff(tag: &str, streams: &[(u8, u16)]) -> Option<(String, String)> {
    let mut ts = pat(0x1000);
    ts.extend_from_slice(&pmt(streams));
    let path = std::env::temp_dir().join(format!("g2g-m936-{tag}-{}.ts", std::process::id()));
    std::fs::write(&path, &ts).expect("write ts");
    let caps = Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    };
    let primary = ts_primary_stream(path.to_str().expect("utf8 path"), &caps);
    std::fs::remove_file(&path).ok();
    primary.map(|p| {
        assert_eq!(p.demux, "tsdemux");
        let (k, v) = p.props.first().expect("stream selection prop").clone();
        (k, v)
    })
}

#[test]
fn mpeg2_video_ts_selects_mpeg2() {
    let sel = sniff("mpeg2", &[(STREAM_TYPE_MPEG2_VIDEO, 0x0100)]);
    assert_eq!(sel, Some(("stream".into(), "mpeg2".into())));
}

#[test]
fn h264_ts_selects_h264_explicitly() {
    let sel = sniff("h264", &[(STREAM_TYPE_H264, 0x0100)]);
    assert_eq!(sel, Some(("stream".into(), "h264".into())));
}

#[test]
fn video_wins_over_audio() {
    let sel = sniff(
        "av",
        &[
            (STREAM_TYPE_MPEG1_AUDIO, 0x0101),
            (STREAM_TYPE_MPEG2_VIDEO, 0x0100),
        ],
    );
    assert_eq!(sel, Some(("stream".into(), "mpeg2".into())));
}

#[test]
fn audio_only_ts_selects_audio() {
    let sel = sniff("mp2", &[(STREAM_TYPE_MPEG1_AUDIO, 0x0101)]);
    assert_eq!(sel, Some(("stream".into(), "mp2".into())));
}
