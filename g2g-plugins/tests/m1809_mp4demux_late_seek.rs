// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "std")]

use core::task::{Context, Poll};

use g2g_core::element::{AsyncElement, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::SeekController;
use g2g_core::{ByteStreamEncoding, Caps, G2gError, Seek, SeekFlags, SeekType};
use g2g_plugins::mp4demux::Mp4Demux;

// ffmpeg -f lavfi -i testsrc=size=64x48:rate=25:duration=3 -c:v libx264 -g 10 -keyint_min 10 -sc_threshold 0 -bf 0
const MP4_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h264_64x48_25fps_gop10.mp4"
);
const FRAME_COUNT: usize = 75;
const SEEK_NS: u64 = 1_000_000_000;
// -g 10 at 25 fps puts a keyframe every 400 ms, the one at or before the target is at 800 ms
const KEYFRAME_BEFORE_SEEK_NS: u64 = 800_000_000;
const FRAME_PERIOD_NS: u64 = 40_000_000;

// queues `seek` once the last sample of the drain has arrived
struct SeekAfterLastFrame {
    app: SeekController,
    seek: Seek,
    frames: usize,
    segment_stop_ns: Option<u64>,
    // a segment seek parks the demuxer until the next seek, the shutdown ends it instead
    shutdown_after_seek: bool,
    eos_after_replay: bool,
    // the pts of every frame after the first Flush
    replayed_pts_ns: Option<Vec<u64>>,
}

impl OutputSink for SeekAfterLastFrame {
    fn poll_push(
        &mut self,
        _cx: &mut Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        match (packet_slot.take(), &mut self.replayed_pts_ns) {
            (Some(PipelinePacket::Flush), replayed @ None) => *replayed = Some(Vec::new()),
            (Some(PipelinePacket::Segment(segment)), Some(_)) => {
                self.segment_stop_ns = segment.stop
            }
            (Some(PipelinePacket::DataFrame(frame)), Some(replayed)) => {
                replayed.push(frame.timing.pts_ns)
            }
            (Some(PipelinePacket::Eos), Some(_)) => self.eos_after_replay = true,
            (Some(PipelinePacket::DataFrame(_)), None) => {
                self.frames += 1;
                if self.frames == FRAME_COUNT {
                    self.app.seek(self.seek);
                    if self.shutdown_after_seek {
                        self.app.shutdown();
                    }
                }
            }
            _ => {}
        }
        Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

async fn run_late_seek(seek: Seek, shutdown_after_seek: bool) -> SeekAfterLastFrame {
    let app = SeekController::new();
    let mut demux = Mp4Demux::new();
    AsyncElement::set_buffered_seek(&mut demux, app.clone());
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Mp4,
        })
        .expect("configure qtdemux");

    let mut sink = SeekAfterLastFrame {
        app,
        seek,
        frames: 0,
        segment_stop_ns: None,
        shutdown_after_seek,
        eos_after_replay: false,
        replayed_pts_ns: None,
    };
    let file = std::fs::read(MP4_FIXTURE).expect("read the fixture");
    let chunk = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(file.into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    demux
        .process(PipelinePacket::DataFrame(chunk), &mut sink)
        .await
        .expect("the file buffers");
    let drained = demux.process(PipelinePacket::Eos, &mut sink).await;

    drained.expect("the demuxer serves the late seek");
    assert_eq!(sink.frames, FRAME_COUNT, "every sample went out first");
    sink
}

#[tokio::test]
async fn a_seek_after_the_drain_replays_from_the_keyframe_before_the_target() {
    let sink = run_late_seek(Seek::flush_to(SEEK_NS), false).await;

    let replayed: Vec<u64> = (KEYFRAME_BEFORE_SEEK_NS..FRAME_COUNT as u64 * FRAME_PERIOD_NS)
        .step_by(FRAME_PERIOD_NS as usize)
        .collect();
    assert_eq!(
        sink.replayed_pts_ns,
        Some(replayed),
        "after a Flush the samples play again from the keyframe at or before the target"
    );
}

#[tokio::test]
async fn a_seek_stop_ends_the_replay_at_the_last_frame_by_the_stop() {
    let sink = run_late_seek(
        Seek {
            stop_type: SeekType::Set,
            stop: STOP_NS,
            ..Seek::flush_to(SEEK_NS)
        },
        false,
    )
    .await;
    assert_eq!(sink.replayed_pts_ns, Some(replayed_to_the_stop()));
    assert_eq!(sink.segment_stop_ns, Some(STOP_NS));
}

const STOP_NS: u64 = 1_500_000_000;
const DURATION_NS: u64 = FRAME_COUNT as u64 * FRAME_PERIOD_NS;

fn replayed_to_the_stop() -> Vec<u64> {
    (KEYFRAME_BEFORE_SEEK_NS..=STOP_NS)
        .step_by(FRAME_PERIOD_NS as usize)
        .collect()
}

#[tokio::test]
async fn a_stop_counted_from_the_end_lands_on_the_duration() {
    let sink = run_late_seek(
        Seek {
            stop_type: SeekType::End,
            stop: DURATION_NS - STOP_NS,
            ..Seek::flush_to(SEEK_NS)
        },
        false,
    )
    .await;
    assert_eq!(sink.replayed_pts_ns, Some(replayed_to_the_stop()));
    assert_eq!(sink.segment_stop_ns, Some(STOP_NS));
}

#[tokio::test]
async fn a_segment_seek_posts_segment_done_at_its_stop() {
    let sink = run_late_seek(
        Seek {
            flags: SeekFlags::FLUSH | SeekFlags::SEGMENT,
            stop_type: SeekType::Set,
            stop: STOP_NS,
            ..Seek::flush_to(SEEK_NS)
        },
        true,
    )
    .await;
    assert_eq!(sink.replayed_pts_ns, Some(replayed_to_the_stop()));
    assert_eq!(sink.app.take_segment_done(), Some(STOP_NS));
    assert!(sink.eos_after_replay, "the shutdown ends the stream");
}
