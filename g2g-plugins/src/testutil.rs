//! Shared helpers for crate-internal tests: collect packets from an
//! `AsyncElement` without each module reinventing a sink.

use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket,
    PushOutcome,
};

#[derive(Default)]
pub(crate) struct CollectSink {
    pub packets: Vec<PipelinePacket>,
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

pub(crate) fn frame(bytes: Vec<u8>) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    }
}

pub(crate) fn data_bytes(packets: &[PipelinePacket]) -> Vec<u8> {
    packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => {
                Some(f.domain.require_system_slice("test").unwrap().to_vec())
            }
            _ => None,
        })
        .flatten()
        .collect()
}

pub(crate) fn first_caps(packets: &[PipelinePacket]) -> Option<Caps> {
    packets.iter().find_map(|p| match p {
        PipelinePacket::CapsChanged(c) => Some(c.clone()),
        _ => None,
    })
}

/// Configure `element`, push each buffer then EOS, return collected packets.
pub(crate) fn run<E: AsyncElement>(element: &mut E, caps: &Caps, buffers: &[&[u8]]) -> CollectSink {
    element.configure_pipeline(caps).unwrap();
    let mut sink = CollectSink::default();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    for buf in buffers {
        runtime
            .block_on(element.process(PipelinePacket::DataFrame(frame(buf.to_vec())), &mut sink))
            .unwrap();
    }
    runtime
        .block_on(element.process(PipelinePacket::Eos, &mut sink))
        .unwrap();
    sink
}

/// Mux then parse: the samples that come out must equal `samples`, and the
/// parser must announce `audio`.
pub(crate) fn roundtrip<M, P>(
    mut mux: M,
    mut parse: P,
    audio: Caps,
    container: Caps,
    samples: &[u8],
) where
    M: AsyncElement,
    P: AsyncElement,
{
    let written = run(&mut mux, &audio, &[samples]);
    let file = data_bytes(&written.packets);
    let parsed = run(&mut parse, &container, &[&file]);
    assert_eq!(first_caps(&parsed.packets), Some(audio));
    assert_eq!(data_bytes(&parsed.packets), samples);
}
