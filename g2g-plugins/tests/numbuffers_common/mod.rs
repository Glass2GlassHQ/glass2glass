//! Helpers shared by the `num-buffers` batteries (`m1040_num_buffers_zero`,
//! `m1042_moqt_whep_num_buffers`): the gst `basesrc` property round-trip and the
//! collect-into-Vec sinks a zero-limit run pushes its EOS into. One definition,
//! included per test binary via `mod numbuffers_common;`.
#![allow(dead_code)] // no one battery uses every helper here

use g2g_core::{G2gError, MultiOutputSink, OutputSink, PipelinePacket, PushOutcome};

/// -1 reads back as -1, n as n, and 0 as 0: a real count, not a rejected value
/// and not a second spelling of "forever". The element's property trait
/// (`SourceLoop` / `MultiOutputSource`) has to be in scope at the call site.
macro_rules! assert_num_buffers_round_trips {
    ($source:expr) => {{
        let source = &mut $source;
        assert_eq!(
            source.get_property("num-buffers"),
            Some(g2g_core::PropValue::Int(-1)),
            "a fresh source is unlimited"
        );
        source
            .set_property("num-buffers", g2g_core::PropValue::Int(7))
            .unwrap();
        assert_eq!(
            source.get_property("num-buffers"),
            Some(g2g_core::PropValue::Int(7))
        );
        source
            .set_property("num-buffers", g2g_core::PropValue::Int(0))
            .unwrap();
        assert_eq!(
            source.get_property("num-buffers"),
            Some(g2g_core::PropValue::Int(0)),
            "0 is a count of zero, not unlimited"
        );
        source
            .set_property("num-buffers", g2g_core::PropValue::Int(-1))
            .unwrap();
        assert_eq!(
            source.get_property("num-buffers"),
            Some(g2g_core::PropValue::Int(-1))
        );
    }};
}
pub(crate) use assert_num_buffers_round_trips;

/// Collects every packet a directly-driven source pushes.
#[derive(Default)]
pub(crate) struct Collect {
    pub(crate) packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        self.packets.push(packet);
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

/// A zero limit produced exactly one packet and it was the EOS.
pub(crate) fn assert_only_eos(out: &Collect, emitted: u64) {
    assert_eq!(emitted, 0, "a zero limit emits no buffers");
    assert_eq!(out.packets.len(), 1, "Eos is the only packet");
    assert!(matches!(out.packets[0], PipelinePacket::Eos));
}

/// [`Collect`] for a multi-pad source, keeping each pad's packets apart.
pub(crate) struct CollectPorts {
    pub(crate) packets: Vec<Vec<PipelinePacket>>,
}

impl CollectPorts {
    pub(crate) fn new(ports: usize) -> Self {
        Self {
            packets: (0..ports).map(|_| Vec::new()).collect(),
        }
    }
}

impl MultiOutputSink for CollectPorts {
    fn poll_push_to(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        port: usize,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        self.packets[port].push(packet);
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }

    fn port_count(&self) -> usize {
        self.packets.len()
    }
}

/// A zero limit produced exactly one packet per pad and each was the EOS: a pad
/// left without one strands the branch behind it.
pub(crate) fn assert_only_eos_on_every_pad(out: &CollectPorts, emitted: u64) {
    assert_eq!(emitted, 0, "a zero limit emits no buffers");
    for (port, packets) in out.packets.iter().enumerate() {
        assert_eq!(packets.len(), 1, "pad {port}: Eos is the only packet");
        assert!(matches!(packets[0], PipelinePacket::Eos), "pad {port}");
    }
}
