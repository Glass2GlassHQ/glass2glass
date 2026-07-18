//! M364 - Matroska/WebM demuxer seek (`MkvDemux` over a seekable `FileSrc`).
//! Like the other demuxers, it drives an upstream byte-seek and re-syncs from the
//! keyframe at or after the target. Matroska blocks carry a keyframe flag, so the
//! demuxer uses it directly.
//!
//! The synthetic WebM has keyframe blocks at 0 ms and 120 ms, with delta blocks
//! between. A seek to 80 ms resumes from the 120 ms keyframe.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::element::{AsyncElement, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, PipelinePacket};
use g2g_core::memory::MemoryDomain;
use g2g_core::runtime::{SeekController, SourceLoop};
use g2g_core::{ByteStreamEncoding, Caps, G2gError, Seek};
use g2g_plugins::filesrc::FileSrc;
use g2g_plugins::mkvdemux::{MkvDemux, MkvStream};

use std::path::PathBuf;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g_m364_{}_{}.webm", std::process::id(), name))
}

// --- minimal EBML/WebM builders (mirror the mkvdemux unit tests) ---
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
/// A SimpleBlock for `track` at relative timecode `rel`, flagged keyframe or not.
fn block(track: u64, rel: i16, keyframe: bool, frame: &[u8]) -> Vec<u8> {
    let mut b = vint(track);
    b.extend_from_slice(&rel.to_be_bytes());
    b.push(if keyframe { 0x80 } else { 0x00 }); // keyframe flag, no lacing
    b.extend_from_slice(frame);
    elem(&[0xA3], &b) // SimpleBlock
}
fn video_track(num: u64, codec: &[u8], w: u32, h: u32) -> Vec<u8> {
    let v = [
        elem(&[0xB0], &uint_body(w as u64)),
        elem(&[0xBA], &uint_body(h as u64)),
    ]
    .concat();
    let body = [
        elem(&[0xD7], &uint_body(num)),
        elem(&[0x86], codec),
        elem(&[0xE0], &v),
    ]
    .concat();
    elem(&[0xAE], &body)
}

/// One VP9 track; keyframe blocks at 0 ms and 120 ms (default 1 ms timescale).
fn webm() -> Vec<u8> {
    let tracks = elem(
        &[0x16, 0x54, 0xAE, 0x6B],
        &video_track(1, b"V_VP9", 320, 240),
    );
    let cluster = elem(
        &[0x1F, 0x43, 0xB6, 0x75],
        &[
            elem(&[0xE7], &uint_body(0)), // Cluster timecode 0
            block(1, 0, true, &[0x01]),
            block(1, 40, false, &[0x02]),
            block(1, 80, false, &[0x03]),
            block(1, 120, true, &[0x04]),
            block(1, 160, false, &[0x05]),
        ]
        .concat(),
    );
    let segment = elem(&[0x18, 0x53, 0x80, 0x67], &[tracks, cluster].concat());
    [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat()
}

#[derive(Default)]
struct Capture {
    frames: Vec<Vec<u8>>,
    flushes: usize,
    segments: usize,
}
impl OutputSink for Capture {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(Frame {
                    domain: MemoryDomain::System(s),
                    ..
                }) => {
                    self.frames.push(s.as_slice().to_vec());
                }
                PipelinePacket::Flush => self.flushes += 1,
                PipelinePacket::Segment(_) => self.segments += 1,
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

struct Chain<'a> {
    demux: &'a mut MkvDemux,
    capture: &'a mut Capture,
}
impl OutputSink for Chain<'_> {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            self.demux.process(packet, self.capture).await?;
            Ok(PushOutcome::Accepted)
        })
    }
}

#[tokio::test]
async fn mkvdemux_seeks_to_the_target_keyframe_over_filesrc() {
    let path = temp_path("seek");
    std::fs::write(&path, webm()).unwrap();

    let byte = SeekController::new();
    let time = SeekController::new();
    // Seek to 80 ms: resume from the next keyframe at 120 ms.
    time.seek(Seek::flush_to(80_000_000));

    let mut src = FileSrc::new(&path, Caps::ByteStream { encoding: ByteStreamEncoding::Matroska })
        .with_chunk_size(16) // small chunks: byte-seek observed mid-read
        .with_seek(byte.clone());
    let mut demux = MkvDemux::new()
        .with_stream(MkvStream::Vp9)
        .with_seek(time.clone(), byte.clone());

    let caps = {
        let c: Pin<Box<dyn Future<Output = _>>> = Box::pin(src.intercept_caps());
        c.await.expect("probe")
    };
    src.configure_pipeline(&caps).expect("configure src");
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        })
        .expect("configure demux");

    let mut capture = Capture::default();
    {
        let mut chain = Chain {
            demux: &mut demux,
            capture: &mut capture,
        };
        src.run(&mut chain).await.expect("filesrc runs");
    }

    // Re-synced from the 120 ms keyframe: blocks [0x04] (kf), [0x05].
    assert!(
        capture.flushes >= 1,
        "the upstream byte-seek flushed downstream"
    );
    assert!(capture.segments >= 1, "a resume segment was emitted");
    assert_eq!(
        capture.frames,
        vec![vec![0x04u8], vec![0x05u8]],
        "resumed from the 120 ms keyframe to the end, pre-target frames discarded"
    );
    let _ = std::fs::remove_file(&path);
}
