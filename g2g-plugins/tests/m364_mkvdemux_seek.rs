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
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
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
fn cluster(ts: u64, blocks: &[Vec<u8>]) -> Vec<u8> {
    let mut body = elem(&[0xE7], &uint_body(ts)); // Cluster Timestamp
    for b in blocks {
        body.extend_from_slice(b);
    }
    elem(&[0x1F, 0x43, 0xB6, 0x75], &body)
}
fn cue_point(time: u64, track: u64, pos: u64) -> Vec<u8> {
    let tp = [
        elem(&[0xF7], &uint_body(track)),
        elem(&[0xF1], &uint_body(pos)),
    ]
    .concat();
    let body = [elem(&[0xB3], &uint_body(time)), elem(&[0xB7], &tp)].concat();
    elem(&[0xBB], &body)
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

/// Two Clusters (keyframes at 0 ms and 120 ms) and a trailing `Cues` index.
/// Returns (whole file, absolute byte offset of Cluster1, Cluster1 bytes).
fn webm_with_cues() -> (Vec<u8>, u64, Vec<u8>) {
    let ebml = elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]);
    let tracks = elem(
        &[0x16, 0x54, 0xAE, 0x6B],
        &video_track(1, b"V_VP9", 320, 240),
    );
    let cluster0 = cluster(
        0,
        &[block(1, 0, true, &[0x01]), block(1, 40, false, &[0x02])],
    );
    let cluster1 = cluster(
        120,
        &[block(1, 0, true, &[0x04]), block(1, 40, false, &[0x05])],
    );
    let cluster0_pos = tracks.len() as u64;
    let cluster1_pos = (tracks.len() + cluster0.len()) as u64;
    let cues = elem(
        &[0x1C, 0x53, 0xBB, 0x6B],
        &[
            cue_point(0, 1, cluster0_pos),
            cue_point(120, 1, cluster1_pos),
        ]
        .concat(),
    );
    let body = [tracks, cluster0.clone(), cluster1.clone(), cues].concat();
    let segment = elem(&[0x18, 0x53, 0x80, 0x67], &body);
    let seg_data_pos = ebml.len() as u64 + (segment.len() - body.len()) as u64;
    let file = [ebml, segment].concat();
    (file, seg_data_pos + cluster1_pos, cluster1)
}

#[derive(Default)]
struct Capture {
    frames: Vec<Vec<u8>>,
    flushes: usize,
    segments: usize,
}
impl OutputSink for Capture {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
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

fn data_frame(bytes: &[u8]) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

struct Chain<'a> {
    demux: &'a mut MkvDemux,
    capture: &'a mut Capture,
}
impl OutputSink for Chain<'_> {
    fn poll_push(
        &mut self,
        cx: &mut core::task::Context<'_>,
        packet: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet.take().expect("poll_push without a packet");
        let mut fut = self.demux.process(packet, self.capture);
        match fut.as_mut().poll(cx) {
            core::task::Poll::Ready(r) => {
                core::task::Poll::Ready(r.map(|()| PushOutcome::Accepted))
            }
            // In-memory test elements never block: their only awaits are
            // pushes into always-ready capture sinks.
            core::task::Poll::Pending => panic!("element future did not resolve in one poll"),
        }
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

/// M864: the state-preserving flag belongs to the seek that set it. A plain
/// upstream flush arriving after an indexed seek finished is a discontinuity, so
/// the parser resets fully: the bytes after it may be anything, and without the
/// EBML header / `Tracks` nothing decodes.
#[tokio::test]
async fn mkvdemux_idle_flush_resets_the_parser_fully() {
    let (file, cluster1_offset, cluster1_bytes) = webm_with_cues();

    let byte = SeekController::new();
    let time = SeekController::new();
    let mut demux = MkvDemux::new()
        .with_stream(MkvStream::Vp9)
        .with_seek(time.clone(), byte.clone());
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        })
        .expect("configure demux");

    // Play the file (parsing the Cues), then run an indexed seek to 120 ms to
    // completion, which leaves the seek's state-preserving flag set.
    let mut pre = Capture::default();
    demux.process(data_frame(&file), &mut pre).await.unwrap();
    assert_eq!(
        pre.frames,
        vec![vec![0x01u8], vec![0x02], vec![0x04], vec![0x05]],
        "the whole file played"
    );
    time.seek(Seek::flush_to(120_000_000));
    let mut post = Capture::default();
    demux.process(data_frame(&[]), &mut post).await.unwrap();
    assert_eq!(
        byte.take_pending().map(|s| s.start),
        Some(cluster1_offset),
        "the Cues index resolved the target Cluster"
    );
    demux
        .process(PipelinePacket::Flush, &mut post)
        .await
        .unwrap();
    demux
        .process(data_frame(&cluster1_bytes), &mut post)
        .await
        .unwrap();
    assert_eq!(
        post.frames,
        vec![vec![0x04u8], vec![0x05]],
        "the indexed seek completed on Cluster1's keyframe"
    );

    // An upstream flush we did not ask for, then a mid-file Cluster with no EBML
    // header or Tracks ahead of it: nothing decodes.
    let mut after = Capture::default();
    demux
        .process(PipelinePacket::Flush, &mut after)
        .await
        .unwrap();
    demux
        .process(data_frame(&cluster1_bytes), &mut after)
        .await
        .unwrap();
    assert!(
        after.frames.is_empty(),
        "a headerless cluster after a full reset decodes nothing, got {:?}",
        after.frames
    );
}
