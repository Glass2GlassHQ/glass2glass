// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! M374 - Matroska `SeekHead`-driven `Cues` prefetch. When the `Cues` index sits
//! at the end of the file (the common layout) it is not parsed during forward
//! playback, so the demuxer cannot use it for a seek yet. A `SeekHead` at the
//! Segment start locates `Cues`; on a seek the demuxer byte-seeks to the `Cues`
//! first, parses them, then byte-seeks to the target Cluster, all before any
//! downstream flush. This is the two-hop that makes the M373 index usable on the
//! first seek of a real (end-of-`Cues`) file.

#![cfg(feature = "std")]

use g2g_core::element::{AsyncElement, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::MemoryDomain;
use g2g_core::runtime::SeekController;
use g2g_core::{ByteStreamEncoding, Caps, G2gError, Seek};
use g2g_plugins::mkvdemux::{MkvDemux, MkvDemuxN, MkvStream};
use g2g_plugins::oneport::OnePortElement;

// --- minimal EBML/WebM builders ---
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
/// Fixed 8-byte SeekPosition: its length is independent of the value, so a
/// SeekHead that points past itself can be sized before the value is known.
fn uint8_body(v: u64) -> Vec<u8> {
    v.to_be_bytes().to_vec()
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
    let mut body = elem(&[0xE7], &uint_body(ts));
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
fn seek_entry(target_id: &[u8], pos: u64) -> Vec<u8> {
    let body = [
        elem(&[0x53, 0xAB], target_id),
        elem(&[0x53, 0xAC], &uint8_body(pos)),
    ]
    .concat();
    elem(&[0x4D, 0xBB], &body)
}

struct Built {
    file: Vec<u8>,
    prefix_end: usize, // end of Cluster0 (what's read before the seek)
    cues_offset: u64,
    cues_bytes: Vec<u8>,
    cluster1_offset: u64,
    cluster1_bytes: Vec<u8>,
}

/// A WebM with a SeekHead (locating the end-of-file Cues) at the Segment start,
/// Tracks, two Clusters (keyframes at 0 ms / 120 ms), and Cues last.
fn webm_with_seekhead() -> Built {
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

    // SeekHead length is fixed (8-byte SeekPositions), so positions that depend on
    // it can be computed before the real values are filled in.
    let seekhead = |cues_pos: u64| {
        elem(
            &[0x11, 0x4D, 0x9B, 0x74],
            &[
                seek_entry(&[0x16, 0x54, 0xAE, 0x6B], 0), // Tracks (placeholder)
                seek_entry(&[0x1C, 0x53, 0xBB, 0x6B], cues_pos), // Cues
            ]
            .concat(),
        )
    };
    let sh_len = seekhead(0).len() as u64;
    // Positions relative to the Segment data start: SeekHead, Tracks, Clusters, Cues.
    let cluster0_pos = sh_len + tracks.len() as u64;
    let cluster1_pos = cluster0_pos + cluster0.len() as u64;
    let cues_pos = cluster1_pos + cluster1.len() as u64;
    let cues = elem(
        &[0x1C, 0x53, 0xBB, 0x6B],
        &[
            cue_point(0, 1, cluster0_pos),
            cue_point(120, 1, cluster1_pos),
        ]
        .concat(),
    );

    let body = [
        seekhead(cues_pos),
        tracks.clone(),
        cluster0.clone(),
        cluster1.clone(),
        cues.clone(),
    ]
    .concat();
    let segment = elem(&[0x18, 0x53, 0x80, 0x67], &body);
    let seg_header = segment.len() - body.len();
    let seg_data_pos = ebml.len() + seg_header;
    let file = [ebml.clone(), segment].concat();

    // Bytes read before the seek: everything up to the end of Cluster0.
    let prefix_end = seg_data_pos + cluster1_pos as usize;
    Built {
        file,
        prefix_end,
        cues_offset: seg_data_pos as u64 + cues_pos,
        cues_bytes: cues,
        cluster1_offset: seg_data_pos as u64 + cluster1_pos,
        cluster1_bytes: cluster1,
    }
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
    PipelinePacket::DataFrame(Frame::from_system_bytes(
        bytes.to_vec(),
        FrameTiming::default(),
        0,
    ))
}

#[tokio::test]
async fn mkvdemux_prefetches_cues_via_seekhead_then_seeks() {
    prefetches_cues_via_seekhead_then_seeks(MkvDemux::new().with_stream(MkvStream::Vp9)).await;
}

#[tokio::test]
async fn the_multi_output_demuxer_prefetches_cues_via_seekhead_then_seeks() {
    let demux = MkvDemuxN::new(vec![MkvStream::Vp9]);
    prefetches_cues_via_seekhead_then_seeks(OnePortElement::wrap(demux, "mkvdemux")).await;
}

async fn prefetches_cues_via_seekhead_then_seeks(mut demux: impl AsyncElement) {
    let b = webm_with_seekhead();

    let byte = SeekController::new();
    let time = SeekController::new();
    // the prefetch reads to the end of the file, so it needs a source that parks there
    time.enable_seeks_after_end();
    demux.set_seek(time.clone(), byte.clone());
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        })
        .expect("configure");

    // 1. Read up to the end of Cluster0: the SeekHead is parsed (so Cues is
    // located) and Cluster0 plays, but the end-of-file Cues are not yet parsed.
    let mut pre = Capture::default();
    demux
        .process(data_frame(&b.file[..b.prefix_end]), &mut pre)
        .await
        .unwrap();
    assert_eq!(
        pre.frames,
        vec![vec![0x01u8], vec![0x02]],
        "Cluster0 played"
    );

    // 2. Seek to 120 ms. With only a SeekHead (no parsed Cues), the demuxer first
    // byte-seeks to the Cues element to prefetch the index.
    time.seek(Seek::flush_to(120_000_000));
    let mut post = Capture::default();
    demux.process(data_frame(&[]), &mut post).await.unwrap(); // dropped while awaiting flush
    let to_cues = byte.take_seek(Ok).expect("byte-seek to Cues issued");
    assert_eq!(
        to_cues.start, b.cues_offset,
        "first hop targets the Cues element"
    );

    // 3. The source flushes (internal, not forwarded) and delivers the Cues bytes.
    demux
        .process(PipelinePacket::Flush, &mut post)
        .await
        .unwrap();
    assert_eq!(
        post.flushes, 0,
        "the Cues-prefetch flush is consumed, not forwarded"
    );
    demux
        .process(data_frame(&b.cues_bytes), &mut post)
        .await
        .unwrap();

    // 4. With the index parsed, the demuxer byte-seeks to the target Cluster.
    let to_target = byte
        .take_seek(Ok)
        .expect("byte-seek to the target Cluster issued");
    assert_eq!(
        to_target.start, b.cluster1_offset,
        "second hop targets Cluster1"
    );

    // 5. The source flushes (the real seek, forwarded) and delivers Cluster1.
    demux
        .process(PipelinePacket::Flush, &mut post)
        .await
        .unwrap();
    demux
        .process(data_frame(&b.cluster1_bytes), &mut post)
        .await
        .unwrap();

    assert_eq!(
        post.flushes, 1,
        "exactly the real seek's flush reaches downstream"
    );
    assert_eq!(
        post.segments, 1,
        "a resume segment is emitted at the landing keyframe"
    );
    assert_eq!(
        post.frames,
        vec![vec![0x04u8], vec![0x05]],
        "re-synced from Cluster1's 120 ms keyframe after the SeekHead -> Cues -> target hops"
    );
}

// ffmpeg's matroska muxer writes a SeekHead at the start that locates the Cues at the end
const AV_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/av_seek_h264_aac.mkv"
);
// small enough that the demuxer has not read the Cues when the seek arrives
const BLOCKSIZE: usize = 256;
// once the sink has the frame at 0.2 s, to 1.02 s, between the keyframes at 0.8 s and 1.2 s
const SEEK_AT: &str = "0.2:1.02";
// the fixture's video: 25 fps, encoded with -g 10
const KEYFRAME_BEFORE_TARGET: usize = 20;

fn launch(pipeline: &str, options: &[&str]) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_g2g-launch"))
        .env("G2G_DEBUG", "demuxseek:6")
        .arg("-q")
        .args(options)
        .args(pipeline.split_whitespace())
        .output()
        .expect("g2g-launch starts")
}

fn succeeded(output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(output.status.success(), "g2g-launch fails: {stderr}");
    stderr
}

fn checksum_lines(output: &std::process::Output) -> Vec<String> {
    succeeded(output);
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::to_string)
        .collect()
}

#[test]
fn a_seek_during_playback_prefetches_the_cues_through_a_parked_source() {
    let pipeline = format!(
        "filesrc location={AV_FIXTURE} blocksize={BLOCKSIZE} ! matroskademux name=d \
         d.video_0 ! checksumsink d.audio_0 ! fakesink"
    );
    let unseeked = checksum_lines(&launch(&pipeline, &[]));
    let output = launch(&pipeline, &["--seek-at", SEEK_AT]);
    let seeked = checksum_lines(&output);
    // demuxseek logs the byte each seek reads from, a re-scan reads from byte 0
    let landings: Vec<u64> = String::from_utf8_lossy(&output.stderr)
        .split("from byte ")
        .skip(1)
        .filter_map(|rest| rest.split(',').next()?.parse().ok())
        .collect();
    assert!(
        !landings.is_empty() && landings.iter().all(|&byte| byte > 0),
        "the prefetched Cues place the seek, landings {landings:?}"
    );
    let resumed = seeked
        .iter()
        .zip(&unseeked)
        .position(|(seeked, unseeked)| seeked != unseeked)
        .expect("the seek changed the output");
    let expected = &unseeked[KEYFRAME_BEFORE_TARGET..];
    let pts = |lines: &[String]| {
        lines
            .iter()
            .map(|line| line.split_whitespace().next().map(str::to_string))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        pts(&seeked[resumed..]),
        pts(expected),
        "the run resumes at the keyframe before the target"
    );
    // the first frame after the flush carries the codec config, so its digest differs
    assert_eq!(seeked[resumed + 1..], expected[1..]);
}

#[test]
fn a_seek_during_playback_in_a_muxer_pipeline_is_served() {
    let output_location = format!("{}/m374_remuxed.ts", env!("CARGO_TARGET_TMPDIR"));
    // a muxer keeps the run from parking filesrc at the end, so reading the Cues there would end the stream
    let pipeline = format!(
        "filesrc location={AV_FIXTURE} blocksize={BLOCKSIZE} ! matroskademux ! h264parse \
         ! mpegtsmux ! filesink location={output_location}"
    );
    succeeded(&launch(&pipeline, &["--seek-at", SEEK_AT]));
    let _ = std::fs::remove_file(&output_location);
}
