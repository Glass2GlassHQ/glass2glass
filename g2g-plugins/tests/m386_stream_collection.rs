//! M386 - stream discovery for MPEG-TS and MP4, the M376 pattern applied beyond
//! Matroska. `TsDemux` announces every elementary stream the PMT declares, and
//! `Mp4Src` announces its (single) video track, both as a
//! `BusMessage::StreamCollection` once the container's stream list is parsed, so
//! an app can discover the streams of a `.ts` / `.mp4` the same way it does an MKV.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::path::PathBuf;

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AudioFormat, Bus, BusMessage, ByteStreamEncoding, Caps, Dim, G2gError, OutputSink,
    PipelinePacket, PushOutcome, Rate, StreamType, VideoCodec,
};

use g2g_plugins::mp4mux::Mp4Mux;
use g2g_plugins::mp4src::Mp4Src;
use g2g_plugins::mpegts::{STREAM_TYPE_AAC, STREAM_TYPE_H264};
use g2g_plugins::tsdemux::TsDemux;

const TS_SYNC: u8 = 0x47;
const TS_PACKET_LEN: usize = 188;

// --- MPEG-TS section builders (mirroring the tsdemux unit-test helpers) ---
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

/// A two-stream PMT (one video, one audio), the common A/V multiplex shape.
fn pmt2(v_pid: u16, v_type: u8, a_pid: u16, a_type: u8) -> Vec<u8> {
    psi(
        0x1000,
        0x02,
        &[
            0x00,
            0x01,
            0xC1,
            0x00,
            0x00,
            0xE0 | (v_pid >> 8) as u8 & 0x1F,
            v_pid as u8, // PCR_PID
            0xF0,
            0x00,
            v_type,
            0xE0 | (v_pid >> 8) as u8 & 0x1F,
            v_pid as u8,
            0xF0,
            0x00,
            a_type,
            0xE0 | (a_pid >> 8) as u8 & 0x1F,
            a_pid as u8,
            0xF0,
            0x00,
        ],
    )
}

#[derive(Default)]
struct Drain {
    bytes: Vec<u8>,
    eos: bool,
}
impl OutputSink for Drain {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.bytes.extend_from_slice(s);
                    }
                }
                PipelinePacket::Eos => self.eos = true,
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

#[tokio::test]
async fn tsdemux_announces_a_stream_collection_from_the_pmt() {
    let v_pid = 0x0100u16; // 256
    let a_pid = 0x0101u16; // 257

    // PAT (points at PMT pid 0x1000) + a 2-stream PMT (H.264 video + AAC audio).
    let mut ts = pat(0x1000);
    ts.extend_from_slice(&pmt2(v_pid, STREAM_TYPE_H264, a_pid, STREAM_TYPE_AAC));

    let (bus, handle) = Bus::new(16);
    let mut demux = TsDemux::new().with_bus(handle);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        })
        .expect("configure");

    let mut sink = Drain::default();
    demux
        .process(
            PipelinePacket::DataFrame(Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(ts.into_boxed_slice())),
                FrameTiming::default(),
                0,
            )),
            &mut sink,
        )
        .await
        .unwrap();

    let collections: Vec<_> = core::iter::from_fn(|| bus.try_recv())
        .filter_map(|m| {
            if let BusMessage::StreamCollection(c) = m {
                Some(c)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(
        collections.len(),
        1,
        "exactly one StreamCollection from the PMT"
    );
    let c = &collections[0];
    assert_eq!(c.len(), 2, "both PMT streams listed");

    let video: Vec<_> = c.streams_of_type(StreamType::Video).collect();
    assert_eq!(video.len(), 1);
    assert_eq!(
        video[0].id, "mpegts-pid-256",
        "video stream keyed by its PID"
    );
    assert!(matches!(
        video[0].caps,
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            ..
        }
    ));

    let audio: Vec<_> = c.streams_of_type(StreamType::Audio).collect();
    assert_eq!(audio.len(), 1);
    assert_eq!(audio[0].id, "mpegts-pid-257");
    assert!(matches!(
        audio[0].caps,
        Caps::Audio {
            format: AudioFormat::Aac,
            ..
        }
    ));
}

// --- MP4 fixture: mux a tiny H.264 track with Mp4Mux, read it back with Mp4Src ---
fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(30 << 16),
    }
}
fn frame(data: Vec<u8>, i: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        FrameTiming {
            pts_ns: i * 33_000_000,
            dts_ns: i * 33_000_000,
            duration_ns: 33_000_000,
            ..FrameTiming::default()
        },
        i,
    ))
}
fn annexb(nals: &[&[u8]]) -> Vec<u8> {
    let mut v = Vec::new();
    for n in nals {
        v.extend_from_slice(&[0, 0, 0, 1]);
        v.extend_from_slice(n);
    }
    v
}

#[tokio::test]
async fn mp4src_announces_its_single_video_track() {
    let path: PathBuf = std::env::temp_dir().join(format!("g2g_m386_{}.mp4", std::process::id()));

    // Mux a 2-frame H.264 track.
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];
    let mut mux = Mp4Mux::new();
    let narrowed = mux.intercept_caps(&h264_caps()).expect("intercept H.264");
    mux.configure_pipeline(&narrowed).expect("configure mux");
    let mut cap = Drain::default();
    mux.process(frame(annexb(&[&sps, &pps, &idr]), 0), &mut cap)
        .await
        .unwrap();
    mux.process(frame(annexb(&[&[0x41u8, 0x9a, 0x00]]), 1), &mut cap)
        .await
        .unwrap();
    mux.process(PipelinePacket::Eos, &mut cap).await.unwrap();
    std::fs::write(&path, &cap.bytes).expect("write mp4");

    // Read it back with a bus attached; Mp4Src announces the one video track.
    let (bus, handle) = Bus::new(16);
    let mut src = Mp4Src::new(&path).with_bus(handle);
    let probed = src.intercept_caps().await.expect("probe");
    src.configure_pipeline(&probed).expect("configure src");
    let mut out = Drain::default();
    src.run(&mut out).await.expect("demux");
    let _ = std::fs::remove_file(&path);

    let collections: Vec<_> = core::iter::from_fn(|| bus.try_recv())
        .filter_map(|m| {
            if let BusMessage::StreamCollection(c) = m {
                Some(c)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(collections.len(), 1, "one StreamCollection for the MP4");
    let c = &collections[0];
    assert_eq!(c.len(), 1, "Mp4Src is single-track: one stream");
    let video: Vec<_> = c.streams_of_type(StreamType::Video).collect();
    assert_eq!(video.len(), 1);
    assert_eq!(video[0].id, "mp4-track-0");
    assert!(matches!(
        video[0].caps,
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(320),
            height: Dim::Fixed(240),
            ..
        }
    ));
}
