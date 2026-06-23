//! Shared frame-emission helper for the packet-producing encoders
//! (`opusenc`, `vpxenc`, `av1enc`). They each turn a batch of encoded
//! `(payload, pts_ns)` packets into downstream `DataFrame`s, announcing the
//! output caps exactly once before the first frame, so the loop lives here.

use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    Caps, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket, PushOutcome, Reconfigure,
};

/// Push a batch of encoded `(payload, pts_ns)` packets downstream.
///
/// `caps` is announced via `CapsChanged` once, before the first frame is ever
/// emitted; `caps_sent` tracks that across calls so it fires at most once.
/// Each payload becomes a System-memory `DataFrame` with `dts == pts` and a
/// monotonic sequence number drawn from `emitted`. An empty batch is a no-op,
/// so the caps stay unannounced until real data arrives.
///
/// Returns `true` if a downstream element requested a keyframe
/// ([`Reconfigure::ForceKeyframe`], e.g. a WebRTC sink on a remote PLI) while
/// pushing this batch. A video encoder should force a key frame on its next
/// encode; an audio encoder ignores it (no keyframes).
pub(crate) async fn emit_packets(
    caps_sent: &mut bool,
    emitted: &mut u64,
    packets: Vec<(Vec<u8>, u64)>,
    caps: &Caps,
    out: &mut dyn OutputSink,
) -> Result<bool, G2gError> {
    if !packets.is_empty() && !*caps_sent {
        out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
        *caps_sent = true;
    }
    let mut force_keyframe = false;
    for (data, pts_ns) in packets {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            FrameTiming { pts_ns, dts_ns: pts_ns, ..FrameTiming::default() },
            *emitted,
        );
        *emitted += 1;
        if matches!(
            out.push(PipelinePacket::DataFrame(frame)).await?,
            PushOutcome::Reconfigure(Reconfigure::ForceKeyframe)
        ) {
            force_keyframe = true;
        }
    }
    Ok(force_keyframe)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;
    use core::future::Future;
    use core::pin::Pin;

    /// Sink whose push returns a fixed outcome, to drive the keyframe-request path.
    struct OutcomeSink(PushOutcome);
    impl OutputSink for OutcomeSink {
        fn push<'a>(
            &'a mut self,
            _packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            let outcome = self.0.clone();
            Box::pin(async move { Ok(outcome) })
        }
    }

    fn caps() -> Caps {
        Caps::Audio { format: g2g_core::AudioFormat::Opus, channels: 2, sample_rate: 48_000 }
    }

    #[tokio::test]
    async fn reports_downstream_force_keyframe() {
        // A downstream ForceKeyframe on a pushed frame propagates as `true`.
        let mut sent = false;
        let mut emitted = 0;
        let mut sink = OutcomeSink(PushOutcome::Reconfigure(Reconfigure::ForceKeyframe));
        let force = emit_packets(
            &mut sent,
            &mut emitted,
            Vec::from([(Vec::from([1u8, 2, 3]), 0u64)]),
            &caps(),
            &mut sink,
        )
        .await
        .unwrap();
        assert!(force, "downstream keyframe request is reported to the encoder");
        assert_eq!(emitted, 1);

        // A plain Accepted reports false.
        let mut sink = OutcomeSink(PushOutcome::Accepted);
        let force = emit_packets(
            &mut sent,
            &mut emitted,
            Vec::from([(Vec::from([4u8]), 1u64)]),
            &caps(),
            &mut sink,
        )
        .await
        .unwrap();
        assert!(!force);
    }

    #[tokio::test]
    async fn empty_batch_emits_nothing() {
        let mut sent = false;
        let mut emitted = 0;
        let mut sink = OutcomeSink(PushOutcome::Accepted);
        let force =
            emit_packets(&mut sent, &mut emitted, Vec::new(), &caps(), &mut sink).await.unwrap();
        assert!(!force);
        assert!(!sent, "caps stay unannounced until real data arrives");
        assert_eq!(emitted, 0);
    }
}
