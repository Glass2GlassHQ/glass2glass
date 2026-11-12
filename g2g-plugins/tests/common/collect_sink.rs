// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use g2g_core::element::{AsyncElement, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::G2gError;

#[derive(Debug, Default)]
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

#[allow(dead_code)] // not every test file that collects packets feeds a byte stream
pub(crate) async fn feed_in_chunks<E: AsyncElement>(
    element: &mut E,
    bytes: &[u8],
    sink: &mut CollectSink,
) {
    const READ_CHUNK_LEN: usize = 4096;
    for chunk in bytes.chunks(READ_CHUNK_LEN) {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(chunk.to_vec().into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        element
            .process(PipelinePacket::DataFrame(frame), sink)
            .await
            .expect("process chunk");
    }
    element
        .process(PipelinePacket::Eos, sink)
        .await
        .expect("Eos");
}
