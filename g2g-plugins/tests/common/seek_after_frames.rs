#![allow(dead_code)] // not every test file uses every item here

use std::time::Duration;

use g2g_core::runtime::{SeekController, SeekTicket, SourceLoop};
use g2g_core::{G2gError, OutputSink, PipelinePacket, PushOutcome, Seek};
use tokio::sync::watch;

// longer than any run here takes, shorter than the reload waits the tests seek through
pub(crate) const LIVE_RUN_LIMIT: Duration = Duration::from_secs(10);

struct FrameCountingSink<S> {
    inner: S,
    frames: watch::Sender<usize>,
}

impl<S: OutputSink> OutputSink for FrameCountingSink<S> {
    fn poll_push(
        &mut self,
        cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let data_frame = matches!(packet_slot, Some(PipelinePacket::DataFrame(_)));
        let pushed = self.inner.poll_push(cx, packet_slot);
        if data_frame && pushed.is_ready() {
            self.frames.send_modify(|frames| *frames += 1);
        }
        pushed
    }
}

// plays a live `src` until it has pushed `frames_at_end` frames, issuing `seek` once it has pushed `frames_before_seek`
pub(crate) async fn run_seeking_after_frames<S: OutputSink + Default>(
    src: &mut impl SourceLoop,
    issuer: &SeekController,
    seek: Seek,
    frames_before_seek: usize,
    frames_at_end: usize,
) -> (S, SeekTicket) {
    let (frames, mut played) = watch::channel(0);
    let mut sink = FrameCountingSink {
        inner: S::default(),
        frames,
    };
    let ticket = {
        let run = src.run(&mut sink);
        let seek_then_play = async {
            played.wait_for(|n| *n >= frames_before_seek).await.unwrap();
            let ticket = issuer.seek(seek);
            played.wait_for(|n| *n >= frames_at_end).await.unwrap();
            ticket
        };
        let raced = async {
            tokio::select! {
                ended = run => panic!("the live run ended: {ended:?}"),
                ticket = seek_then_play => ticket,
            }
        };
        tokio::time::timeout(LIVE_RUN_LIMIT, raced)
            .await
            .expect("the run pushes every frame")
    };
    (sink.inner, ticket)
}
