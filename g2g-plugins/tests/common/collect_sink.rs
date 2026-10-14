use g2g_core::element::{OutputSink, PushOutcome};
use g2g_core::frame::PipelinePacket;
use g2g_core::G2gError;

#[derive(Default)]
pub(crate) struct CollectSink {
    pub(crate) packets: Vec<PipelinePacket>,
}

impl OutputSink for CollectSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        self.packets
            .push(packet_slot.take().expect("poll_push without a packet"));
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}
