//! Shared frame-emission helper for the packet-producing encoders
//! (`opusenc`, `vpxenc`, `av1enc`). They each turn a batch of encoded
//! `(payload, pts_ns)` packets into downstream `DataFrame`s, announcing the
//! output caps exactly once before the first frame, so the loop lives here.

use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    Caps, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket, PushOutcome,
    Reconfigure, VideoCodec,
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

/// True when a new downstream bitrate estimate differs from the running rate by
/// at least 20%. No encoder here can retarget in place (rav1e fixes the rate at
/// `Context` construction, libavcodec at `avcodec_open2`), so each change costs a
/// rebuild and a keyframe; the gate damps a jittery BWE estimate.
#[cfg(any(
    feature = "av1-encode",
    all(target_os = "linux", feature = "openh264"),
    all(any(target_os = "linux", windows), feature = "ffmpeg")
))]
pub(crate) fn bitrate_change_is_significant(current: u64, target: u64) -> bool {
    let current = current.max(1);
    target.abs_diff(current) * 100 >= current * 20
}

pub(crate) trait PacketPayload {
    fn into_domain(self) -> MemoryDomain;
}

impl PacketPayload for Vec<u8> {
    fn into_domain(self) -> MemoryDomain {
        MemoryDomain::System(SystemSlice::from_boxed(self.into_boxed_slice()))
    }
}

impl PacketPayload for MemoryDomain {
    fn into_domain(self) -> MemoryDomain {
        self
    }
}

/// Push a batch and return any downstream feedback (see [`EmitFeedback`]).
pub(crate) async fn emit_packets(
    caps_sent: &mut bool,
    emitted: &mut u64,
    packets: Vec<(impl PacketPayload, u64)>,
    caps: &Caps,
    out: &mut dyn OutputSink,
) -> Result<EmitFeedback, G2gError> {
    if !packets.is_empty() && !*caps_sent {
        out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
        *caps_sent = true;
    }
    let mut feedback = EmitFeedback::default();
    for (payload, pts_ns) in packets {
        let domain = payload.into_domain();
        let keyframe = is_sync_point(caps, &domain);
        let frame = Frame::new(
            domain,
            FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                keyframe,
                ..FrameTiming::default()
            },
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

// read from the bitstream, which a muxer or segmenter downstream cuts on
fn is_sync_point(caps: &Caps, domain: &MemoryDomain) -> bool {
    let Some(bytes) = domain.as_system_slice() else {
        return false;
    };
    match caps {
        Caps::CompressedVideo {
            codec: VideoCodec::Vp8,
            ..
        } => crate::fmp4mux::vp8_keyframe(bytes),
        Caps::CompressedVideo {
            codec: VideoCodec::Vp9,
            ..
        } => crate::fmp4mux::vp9_keyframe(bytes),
        Caps::CompressedVideo { codec, .. } => crate::annexb::au_is_keyframe(*codec, bytes),
        Caps::Audio { .. } => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sink whose push returns a fixed outcome, to drive the keyframe-request path.
    struct OutcomeSink(PushOutcome);
    impl OutputSink for OutcomeSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            packet_slot.take();

            let outcome = self.0.clone();
            core::task::Poll::Ready(Ok(outcome))
        }
    }

    fn caps() -> Caps {
        Caps::Audio {
            format: g2g_core::AudioFormat::Opus,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }

    #[derive(Default)]
    struct TimingSink(Vec<bool>);
    impl OutputSink for TimingSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            if let Some(PipelinePacket::DataFrame(frame)) = packet_slot.take() {
                self.0.push(frame.timing.keyframe);
            }
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    #[tokio::test]
    async fn flags_the_keyframes_the_bitstream_marks() {
        // nal_unit_type 5 is an IDR slice, 1 a non-IDR slice
        let idr = Vec::from([0u8, 0, 0, 1, 0x65, 0x88]);
        let inter = Vec::from([0u8, 0, 0, 1, 0x41, 0x9a]);
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: g2g_core::Dim::Fixed(16),
            height: g2g_core::Dim::Fixed(16),
            framerate: g2g_core::Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        };
        let mut sink = TimingSink::default();
        emit_packets(
            &mut false,
            &mut 0,
            Vec::from([(idr, 0u64), (inter, 1)]),
            &h264,
            &mut sink,
        )
        .await
        .unwrap();
        assert_eq!(sink.0, [true, false]);

        let mut sink = TimingSink::default();
        emit_packets(
            &mut false,
            &mut 0,
            Vec::from([(Vec::from([1u8]), 0u64)]),
            &caps(),
            &mut sink,
        )
        .await
        .unwrap();
        assert_eq!(sink.0, [true], "every audio packet is a sync point");
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
        let no_packets: Vec<(Vec<u8>, u64)> = Vec::new();
        let fb = emit_packets(&mut sent, &mut emitted, no_packets, &caps(), &mut sink)
            .await
            .unwrap();
        assert!(!fb.force_keyframe);
        assert_eq!(fb.bitrate_bps, None);
        assert!(!sent, "caps stay unannounced until real data arrives");
        assert_eq!(emitted, 0);
    }
}
