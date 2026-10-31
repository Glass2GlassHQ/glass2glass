use g2g_core::element::{OutputSink, PushOutcome};
use g2g_core::frame::PipelinePacket;
use g2g_core::{G2gError, Reconfigure};

// answers the access unit at `request_at` with a keyframe request, as a WebRTC sink does on a PLI
#[derive(Debug)]
pub(crate) struct KeyframeRequestSink {
    request_at: usize,
    keyframes: Vec<bool>,
}

impl KeyframeRequestSink {
    pub(crate) fn new(request_at: usize) -> Self {
        Self {
            request_at,
            keyframes: Vec::new(),
        }
    }

    pub(crate) fn keyframe_indices(&self) -> Vec<usize> {
        (0..self.keyframes.len())
            .filter(|&index| self.keyframes[index])
            .collect()
    }
}

impl OutputSink for KeyframeRequestSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        let PipelinePacket::DataFrame(frame) = packet else {
            return core::task::Poll::Ready(Ok(PushOutcome::Accepted));
        };
        self.keyframes.push(frame.timing.keyframe);
        let outcome = if self.keyframes.len() == self.request_at + 1 {
            PushOutcome::Reconfigure(Reconfigure::ForceKeyframe)
        } else {
            PushOutcome::Accepted
        };
        core::task::Poll::Ready(Ok(outcome))
    }
}
