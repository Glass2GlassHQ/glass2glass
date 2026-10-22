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

// queues a seek once the last sample of the drain has arrived
struct SeekAfterLastFrame {
    app: SeekController,
    frames: usize,
}

impl OutputSink for SeekAfterLastFrame {
    fn poll_push(
        &mut self,
        _cx: &mut Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::DataFrame(_)) = packet_slot.take() {
            self.frames += 1;
            if self.frames == FRAME_COUNT {
                self.app.seek(Seek::flush_to(SEEK_NS));
            }
        }
        Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

#[tokio::test]
async fn a_seek_after_the_drain_fails_the_demuxer() {
    let app = SeekController::new();
    let mut demux = Mp4Demux::new();
    AsyncElement::set_buffered_seek(&mut demux, app.clone());
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Mp4,
        })
        .expect("configure qtdemux");

    let mut sink = SeekAfterLastFrame { app, frames: 0 };
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

    assert_eq!(sink.frames, FRAME_COUNT, "every sample went out first");
    assert_eq!(drained, Err(G2gError::SeekRefused));
}
