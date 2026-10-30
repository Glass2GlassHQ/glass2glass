#![cfg(feature = "std")]

use core::task::{Context, Poll};

use g2g_core::element::{AsyncElement, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::SeekController;
use g2g_core::{ByteStreamEncoding, Caps, G2gError, Seek};
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

// queues a seek once the last sample of the drain has arrived
struct SeekAfterLastFrame {
    app: SeekController,
    frames: usize,
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
            (Some(PipelinePacket::DataFrame(frame)), Some(replayed)) => {
                replayed.push(frame.timing.pts_ns)
            }
            (Some(PipelinePacket::DataFrame(_)), None) => {
                self.frames += 1;
                if self.frames == FRAME_COUNT {
                    self.app.seek(Seek::flush_to(SEEK_NS));
                }
            }
            _ => {}
        }
        Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

#[tokio::test]
async fn a_seek_after_the_drain_replays_from_the_keyframe_before_the_target() {
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
        frames: 0,
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
    let replayed: Vec<u64> = (KEYFRAME_BEFORE_SEEK_NS..FRAME_COUNT as u64 * FRAME_PERIOD_NS)
        .step_by(FRAME_PERIOD_NS as usize)
        .collect();
    assert_eq!(
        sink.replayed_pts_ns,
        Some(replayed),
        "after a Flush the samples play again from the keyframe at or before the target"
    );
}
