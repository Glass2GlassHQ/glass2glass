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
/// Downstream feedback an encoder collects while pushing a batch, so it can
/// adapt its next encode: a keyframe request and/or a target bitrate.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct EmitFeedback {
    /// A downstream element asked for a keyframe ([`Reconfigure::ForceKeyframe`],
    /// e.g. a WebRTC sink on a remote PLI). A video encoder forces a key frame on
    /// its next encode; an audio encoder ignores it.
    pub force_keyframe: bool,
    /// The most recent target send bitrate (bits/second) reported downstream
    /// ([`PushOutcome::Bitrate`], a WebRTC sink's BWE estimate), or `None` if
    /// none was seen this batch.
    pub bitrate_bps: Option<u32>,
}

/// Push a batch and return any downstream feedback (see [`EmitFeedback`]).
pub(crate) async fn emit_packets(
    caps_sent: &mut bool,
    emitted: &mut u64,
    packets: Vec<(Vec<u8>, u64)>,
    caps: &Caps,
    out: &mut dyn OutputSink,
) -> Result<EmitFeedback, G2gError> {
    if !packets.is_empty() && !*caps_sent {
        out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
        *caps_sent = true;
    }
    let mut feedback = EmitFeedback::default();
    for (data, pts_ns) in packets {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            FrameTiming { pts_ns, dts_ns: pts_ns, ..FrameTiming::default() },
            *emitted,
        );
        *emitted += 1;
        match out.push(PipelinePacket::DataFrame(frame)).await? {
            PushOutcome::Reconfigure(Reconfigure::ForceKeyframe) => feedback.force_keyframe = true,
            PushOutcome::Bitrate(bps) => feedback.bitrate_bps = Some(bps),
            _ => {}
        }
    }
    Ok(feedback)
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
    async fn reports_downstream_force_keyframe_and_bitrate() {
        // A downstream ForceKeyframe on a pushed frame propagates.
        let mut sent = false;
        let mut emitted = 0;
        let mut sink = OutcomeSink(PushOutcome::Reconfigure(Reconfigure::ForceKeyframe));
        let fb = emit_packets(
            &mut sent,
            &mut emitted,
            Vec::from([(Vec::from([1u8, 2, 3]), 0u64)]),
            &caps(),
            &mut sink,
        )
        .await
        .unwrap();
        assert!(fb.force_keyframe, "downstream keyframe request is reported");
        assert_eq!(fb.bitrate_bps, None);
        assert_eq!(emitted, 1);

        // A downstream Bitrate estimate propagates as the target.
        let mut sink = OutcomeSink(PushOutcome::Bitrate(800_000));
        let fb = emit_packets(
            &mut sent,
            &mut emitted,
            Vec::from([(Vec::from([4u8]), 1u64)]),
            &caps(),
            &mut sink,
        )
        .await
        .unwrap();
        assert!(!fb.force_keyframe);
        assert_eq!(fb.bitrate_bps, Some(800_000));

        // A plain Accepted reports neither.
        let mut sink = OutcomeSink(PushOutcome::Accepted);
        let fb = emit_packets(
            &mut sent,
            &mut emitted,
            Vec::from([(Vec::from([5u8]), 2u64)]),
            &caps(),
            &mut sink,
        )
        .await
        .unwrap();
        assert!(!fb.force_keyframe);
        assert_eq!(fb.bitrate_bps, None);
    }

    #[tokio::test]
    async fn empty_batch_emits_nothing() {
        let mut sent = false;
        let mut emitted = 0;
        let mut sink = OutcomeSink(PushOutcome::Accepted);
        let fb =
            emit_packets(&mut sent, &mut emitted, Vec::new(), &caps(), &mut sink).await.unwrap();
        assert!(!fb.force_keyframe);
        assert_eq!(fb.bitrate_bps, None);
        assert!(!sent, "caps stay unannounced until real data arrives");
        assert_eq!(emitted, 0);
    }
}
