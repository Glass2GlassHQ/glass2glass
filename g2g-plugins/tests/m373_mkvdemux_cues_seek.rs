//! M373 - Matroska `Cues`-indexed seek. Unlike the M364 re-scan-from-zero seek,
//! once the `Cues` element has been parsed the demuxer seeks the upstream byte
//! source straight to the target Cluster's byte offset (mid-segment), keeping its
//! Tracks / TimestampScale / Cues so the landing Cluster decodes immediately.
//!
//! The synthetic WebM has two Clusters (keyframes at 0 ms and 120 ms) and a `Cues`
//! index at the end. The whole file is fed first (so `Cues` is parsed), then a
//! seek to 120 ms must request the *Cluster1* byte offset (not 0) and re-sync
//! there.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::element::{AsyncElement, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::SeekController;
use g2g_core::{ByteStreamEncoding, Caps, G2gError, Seek};
use g2g_plugins::mkvdemux::{MkvDemux, MkvStream};

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
fn block(track: u64, rel: i16, keyframe: bool, frame: &[u8]) -> Vec<u8> {
    let mut b = vint(track);
    b.extend_from_slice(&rel.to_be_bytes());
    b.push(if keyframe { 0x80 } else { 0x00 });
    b.extend_from_slice(frame);
    elem(&[0xA3], &b)
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
    // Cues last (the common layout); positions are relative to the Segment data.
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
    let body = [tracks.clone(), cluster0.clone(), cluster1.clone(), cues].concat();
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

fn data_frame(bytes: &[u8]) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

#[tokio::test]
async fn mkvdemux_cues_seek_jumps_straight_to_the_target_cluster() {
    let (file, cluster1_offset, cluster1_bytes) = webm_with_cues();

    let byte = SeekController::new(); // stands in for the FileSrc byte-seek channel
    let time = SeekController::new();
    let mut demux = MkvDemux::new()
        .with_stream(MkvStream::Vp9)
        .with_seek(time.clone(), byte.clone());
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        })
        .expect("configure");

    // 1. Feed the whole file: all frames demux and the Cues index is parsed.
    let mut pre = Capture::default();
    demux.process(data_frame(&file), &mut pre).await.unwrap();
    assert_eq!(
        pre.frames,
        vec![vec![0x01u8], vec![0x02], vec![0x04], vec![0x05]]
    );

    // 2. Seek to 120 ms. The next process polls it: with the Cues index parsed,
    // the demuxer requests the *Cluster1* byte offset on the upstream channel
    // (a re-scan would have requested 0).
    time.seek(Seek::flush_to(120_000_000));
    let mut post = Capture::default();
    demux.process(data_frame(&[]), &mut post).await.unwrap(); // dropped while awaiting flush
    let requested = byte
        .take_pending()
        .expect("an upstream byte-seek was requested");
    assert_eq!(
        requested.start, cluster1_offset,
        "Cues seek targets Cluster1, not a re-scan from 0"
    );
    assert_eq!(
        &file[cluster1_offset as usize..cluster1_offset as usize + 4],
        &[0x1F, 0x43, 0xB6, 0x75]
    );

    // 3. The byte source flushes and delivers bytes from the new offset (Cluster1).
    demux
        .process(PipelinePacket::Flush, &mut post)
        .await
        .unwrap();
    demux
        .process(data_frame(&cluster1_bytes), &mut post)
        .await
        .unwrap();

    assert_eq!(post.flushes, 1, "the flush is forwarded downstream");
    assert_eq!(
        post.segments, 1,
        "a resume segment is emitted at the landing keyframe"
    );
    assert_eq!(
        post.frames,
        vec![vec![0x04u8], vec![0x05]],
        "re-synced from Cluster1's 120 ms keyframe, having seeked straight to it"
    );
}
