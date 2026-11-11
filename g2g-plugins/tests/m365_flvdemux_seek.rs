// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! M365 - FLV demuxer seek (`FlvDemux` over a seekable `FileSrc`). Drives an
//! upstream byte-seek and re-syncs from the keyframe at or before the target,
//! using the FLV video tag's frame-type flag.
//!
//! The clip has keyframe video tags at 0 ms and 120 ms, with interframes between.
//! A seek to 140 ms resumes from the 120 ms keyframe.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::element::{AsyncElement, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, PipelinePacket};
use g2g_core::memory::MemoryDomain;
use g2g_core::runtime::{SeekController, SourceLoop};
use g2g_core::{ByteStreamEncoding, Caps, G2gError, Seek, SeekType};
use g2g_plugins::filesrc::FileSrc;
use g2g_plugins::flv::{FlvCodec, FlvMuxer};
use g2g_plugins::flvdemux::{FlvDemux, FlvStream};

use std::path::PathBuf;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g_m365_{}_{}.flv", std::process::id(), name))
}

/// Build an FLV byte stream: video tags `(pts_ms, keyframe, au)`.
fn make_flv(tags: &[(u32, bool, Vec<u8>)]) -> Vec<u8> {
    let mut mux = FlvMuxer::new(FlvCodec::H264);
    let mut out = Vec::new();
    for (pts, kf, au) in tags {
        out.extend_from_slice(&mux.push_video(au, *pts, 0, *kf));
    }
    out
}

#[derive(Default)]
struct Capture {
    frames: Vec<Vec<u8>>,
    flushes: usize,
    segment_starts: Vec<u64>,
    segment_stops: Vec<Option<u64>>,
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
                PipelinePacket::Segment(segment) => {
                    self.segment_starts.push(segment.start);
                    self.segment_stops.push(segment.stop);
                }
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

struct Chain<'a> {
    demux: &'a mut FlvDemux,
    capture: &'a mut Capture,
    chunks_read: usize,
}
impl OutputSink for Chain<'_> {
    fn poll_push(
        &mut self,
        cx: &mut core::task::Context<'_>,
        packet: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet.take().expect("poll_push without a packet");
        if let PipelinePacket::DataFrame(_) = packet {
            self.chunks_read += 1;
        }
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

const CHUNK_SIZE: usize = 16;

// the file source's chunks read and what the demuxer sent on, for `seek` queued before the run
async fn run_seek(name: &str, flv: &[u8], seek: Seek) -> (Capture, usize) {
    let path = temp_path(name);
    std::fs::write(&path, flv).unwrap();

    let byte = SeekController::new();
    let time = SeekController::new();
    time.seek(seek);

    let mut src = FileSrc::new(
        &path,
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Flv,
        },
    )
    .with_chunk_size(CHUNK_SIZE)
    .with_seek(byte.clone());
    let mut demux = FlvDemux::new()
        .with_stream(FlvStream::H264)
        .with_seek(time.clone(), byte.clone());

    let caps = {
        let c: Pin<Box<dyn Future<Output = _>>> = Box::pin(src.intercept_caps());
        c.await.expect("probe")
    };
    src.configure_pipeline(&caps).expect("configure src");
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Flv,
        })
        .expect("configure demux");

    let mut capture = Capture::default();
    let mut chain = Chain {
        demux: &mut demux,
        capture: &mut capture,
        chunks_read: 0,
    };
    src.run(&mut chain).await.expect("filesrc runs");
    let chunks_read = chain.chunks_read;
    let _ = std::fs::remove_file(&path);
    (capture, chunks_read)
}

#[tokio::test]
async fn flvdemux_seeks_to_the_target_keyframe_over_filesrc() {
    // Valid AVCC payloads (4-byte length prefix); the element re-frames them
    // Annex-B on the way out (M662).
    let flv = make_flv(&[
        (0, true, vec![0, 0, 0, 2, 0x65, 0x01]),
        (40, false, vec![0, 0, 0, 2, 0x41, 0x02]),
        (80, false, vec![0, 0, 0, 2, 0x41, 0x03]),
        (120, true, vec![0, 0, 0, 2, 0x65, 0x04]),
        (160, false, vec![0, 0, 0, 2, 0x41, 0x05]),
    ]);
    // Seek to 140 ms: resume from the keyframe before it, at 120 ms.
    let target_ns = 140_000_000;
    let (capture, _) = run_seek("seek", &flv, Seek::flush_to(target_ns)).await;

    assert!(
        capture.flushes >= 1,
        "the upstream byte-seek flushed downstream"
    );
    assert_eq!(
        capture.segment_starts.last(),
        Some(&target_ns),
        "the resume segment starts at the target"
    );
    assert_eq!(
        capture.frames,
        vec![vec![0, 0, 0, 1, 0x65, 0x04], vec![0, 0, 0, 1, 0x41, 0x05]],
        "resumed from the 120 ms keyframe to the end, earlier frames discarded"
    );
}

#[tokio::test]
async fn flvdemux_ends_the_stream_at_the_seek_stop() {
    const FRAME_PERIOD_MS: u32 = 40;
    const FRAMES: u32 = 50;
    let tags: Vec<_> = (0..FRAMES)
        .map(|index| {
            let keyframe = index % 5 == 0;
            let nal_header = if keyframe { 0x65 } else { 0x41 };
            (
                index * FRAME_PERIOD_MS,
                keyframe,
                vec![0, 0, 0, 2, nal_header, index as u8],
            )
        })
        .collect();
    let flv = make_flv(&tags);
    let total_chunks = flv.len().div_ceil(CHUNK_SIZE);
    let start_ns = 240_000_000;
    let stop_ns = 400_000_000;
    let seek = Seek {
        stop_type: SeekType::Set,
        stop: stop_ns,
        ..Seek::flush_to(start_ns)
    };
    let (capture, chunks_read) = run_seek("stop", &flv, seek).await;

    assert_eq!(capture.segment_stops.last(), Some(&Some(stop_ns)));
    let frame_indexes: Vec<u8> = capture.frames.iter().map(|frame| frame[5]).collect();
    // the keyframe at 200 ms decodes the 240 ms target, 400 ms is the last at or before the stop
    assert_eq!(frame_indexes, (5..=10).collect::<Vec<u8>>());
    assert_eq!(capture.flushes, 1, "the seek to the end is not forwarded");
    assert!(
        chunks_read < total_chunks / 2,
        "the source stopped reading at the stop, read {chunks_read} of {total_chunks} chunks"
    );
}
