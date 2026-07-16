//! Opus packet parser that refines source-side `Caps` from the TOC byte.
//!
//! The audio sibling of `aacparse`: it reads each `DataFrame`'s leading TOC
//! (table-of-contents) byte and recovers the mono/stereo channel count,
//! emitting a `CapsChanged` before forwarding the frame. This lets a raw Opus
//! elementary stream (RTP payload, a container that left channels unset) be
//! restreamed or muxed with a concrete channel count.
//!
//! Unlike H.264 / AAC there is no in-band syncword to hunt: the container
//! frames Opus packets, so the first byte of each packet *is* the TOC (RFC 6716
//! §3.1). Parsing is therefore reading byte 0, never a scan.
//!
//! `Caps::Audio` has no open (`Any`) field, so a source advertising Opus before
//! the first packet uses sentinel `channels`/`sample_rate` 0; the negotiated
//! constraint is `IdentityAny` (forward whatever Opus the upstream produces).
//! The Opus-only guard lives in `intercept_caps`.
//!
//! Scope: the TOC stereo bit distinguishes mono vs stereo, which covers Opus
//! channel-mapping family 0 (the common case). Multichannel (family 1) carries
//! its channel count in the `OpusHead` header, not the per-packet TOC, so a raw
//! parser can't recover it; that needs the container header and is deferred.
//! Opus always decodes at 48 kHz regardless of the coded bandwidth, so the
//! sample rate is the constant [`OPUS_RATE_HZ`].

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    G2gError, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
};

/// Opus decodes at 48 kHz for every coded bandwidth (NB..FB), so refined caps
/// always report this rate; the bandwidth only bounds the audio content.
pub const OPUS_RATE_HZ: u32 = 48_000;

#[derive(Debug, Default)]
pub struct OpusParse {
    configured: bool,
    last_emitted_caps: Option<Caps>,
    headers_emitted: u64,
}

impl OpusParse {
    pub fn new() -> Self {
        Self::default()
    }

    /// Count of `CapsChanged` packets pushed downstream, for tests asserting
    /// re-emission is suppressed when the channel count is unchanged.
    pub fn caps_changes_emitted(&self) -> u64 {
        self.headers_emitted
    }
}

impl AsyncElement for OpusParse {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::Audio { format: AudioFormat::Opus, .. } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Pass-through identity over Opus of any channel count (the parser refines
    /// that mid-stream from the TOC but never changes media type). `IdentityAny`,
    /// not `Identity(set)`, because audio caps cannot express "Opus at any
    /// channels" in a single `Caps`.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::Audio { format: AudioFormat::Opus, .. } => {
                self.configured = true;
                Ok(ConfigureOutcome::Accepted)
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Opus parser",
            "Codec/Parser/Audio",
            "Refines Opus caps (channel count) from each packet's TOC byte",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    if let g2g_core::MemoryDomain::System(slice) = &frame.domain {
                        if let Some(toc) = parse_toc(slice.as_slice()) {
                            let new_caps = Caps::Audio {
                                format: AudioFormat::Opus,
                                channels: toc.channels,
                                sample_rate: OPUS_RATE_HZ,
                            };
                            if self.last_emitted_caps.as_ref() != Some(&new_caps) {
                                out.push(PipelinePacket::CapsChanged(new_caps.clone())).await?;
                                self.last_emitted_caps = Some(new_caps);
                                self.headers_emitted += 1;
                            }
                        }
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                PipelinePacket::Flush => {
                    self.last_emitted_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                // Segment is control: forward unchanged.
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for OpusParse {
    fn pad_templates() -> Vec<PadTemplate> {
        // `Caps::Audio` has no open dims; pin the common stereo/48 kHz shape.
        let opus = Caps::Audio { format: AudioFormat::Opus, channels: 2, sample_rate: OPUS_RATE_HZ };
        Vec::from([
            PadTemplate::sink(CapsSet::one(opus.clone())),
            PadTemplate::source(CapsSet::one(opus)),
        ])
    }
}

/// Opus internal coder (RFC 6716 Table 2). SILK and CELT are the two base
/// coders; Hybrid layers CELT over SILK for the wider bandwidths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpusMode {
    SilkOnly,
    Hybrid,
    CeltOnly,
}

/// Coded audio bandwidth (RFC 6716 Table 1): narrow to full band.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpusBandwidth {
    Narrow,
    Medium,
    Wide,
    SuperWide,
    Full,
}

/// The fields decoded from an Opus packet's TOC byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpusToc {
    pub channels: u8,
    pub mode: OpusMode,
    pub bandwidth: OpusBandwidth,
    pub frame_duration_us: u32,
}

/// Decode the TOC byte of an Opus packet (RFC 6716 §3.1). `None` only for an
/// empty packet: every TOC byte value is structurally valid, so a non-empty
/// packet always yields a `OpusToc`.
fn parse_toc(packet: &[u8]) -> Option<OpusToc> {
    let toc = *packet.first()?;
    let config = toc >> 3; // top 5 bits select mode + bandwidth + frame size
    let stereo = (toc >> 2) & 0x01 == 1; // 1 bit
    let channels = if stereo { 2 } else { 1 };
    let (mode, bandwidth, frame_duration_us) = decode_config(config);
    Some(OpusToc { channels, mode, bandwidth, frame_duration_us })
}

/// Map a 5-bit TOC `config` (0..=31) to its coder, bandwidth, and frame
/// duration per RFC 6716 Table 2. The config space partitions into SILK
/// (0..=11, four durations each), Hybrid (12..=15, two durations each), and
/// CELT (16..=31, four durations each).
fn decode_config(config: u8) -> (OpusMode, OpusBandwidth, u32) {
    use OpusBandwidth::*;
    use OpusMode::*;
    const SILK_MS: [u32; 4] = [10_000, 20_000, 40_000, 60_000];
    const HYBRID_MS: [u32; 2] = [10_000, 20_000];
    const CELT_MS: [u32; 4] = [2_500, 5_000, 10_000, 20_000];
    match config {
        0..=3 => (SilkOnly, Narrow, SILK_MS[(config % 4) as usize]),
        4..=7 => (SilkOnly, Medium, SILK_MS[(config % 4) as usize]),
        8..=11 => (SilkOnly, Wide, SILK_MS[(config % 4) as usize]),
        12..=13 => (Hybrid, SuperWide, HYBRID_MS[(config % 2) as usize]),
        14..=15 => (Hybrid, Full, HYBRID_MS[(config % 2) as usize]),
        16..=19 => (CeltOnly, Narrow, CELT_MS[(config % 4) as usize]),
        20..=23 => (CeltOnly, Wide, CELT_MS[(config % 4) as usize]),
        24..=27 => (CeltOnly, SuperWide, CELT_MS[(config % 4) as usize]),
        // config is a 5-bit field, so 28..=31 is the only remaining range (FB).
        _ => (CeltOnly, Full, CELT_MS[(config % 4) as usize]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Build a one-byte Opus packet whose TOC encodes `config` / `stereo`, plus
    /// `payload_len` trailing zero bytes (the TOC parser ignores them).
    fn opus_packet(config: u8, stereo: bool, payload_len: usize) -> Vec<u8> {
        let toc = (config << 3) | ((stereo as u8) << 2); // frame-count code 0 (one frame)
        let mut p = vec![0u8; 1 + payload_len];
        p[0] = toc;
        p
    }

    #[test]
    fn recovers_mono_silk_wideband_20ms() {
        // config 9: SILK, wideband, 20 ms (group 8..=11, index 1).
        let toc = parse_toc(&opus_packet(9, false, 40)).expect("TOC must parse");
        assert_eq!(toc.channels, 1);
        assert_eq!(toc.mode, OpusMode::SilkOnly);
        assert_eq!(toc.bandwidth, OpusBandwidth::Wide);
        assert_eq!(toc.frame_duration_us, 20_000);
    }

    #[test]
    fn recovers_stereo_celt_fullband_20ms() {
        // config 31: CELT, fullband, 20 ms (group 28..=31, index 3).
        let toc = parse_toc(&opus_packet(31, true, 12)).expect("TOC must parse");
        assert_eq!(toc.channels, 2);
        assert_eq!(toc.mode, OpusMode::CeltOnly);
        assert_eq!(toc.bandwidth, OpusBandwidth::Full);
        assert_eq!(toc.frame_duration_us, 20_000);
    }

    #[test]
    fn decodes_hybrid_and_short_celt_frames() {
        // config 12: Hybrid, super-wideband, 10 ms.
        let hybrid = decode_config(12);
        assert_eq!(hybrid, (OpusMode::Hybrid, OpusBandwidth::SuperWide, 10_000));
        // config 16: CELT, narrowband, 2.5 ms (the shortest Opus frame).
        let celt = decode_config(16);
        assert_eq!(celt, (OpusMode::CeltOnly, OpusBandwidth::Narrow, 2_500));
    }

    #[test]
    fn every_config_decodes_to_a_valid_duration() {
        // The 5-bit config space is fully assigned; no value panics or yields a
        // zero duration.
        for config in 0u8..=31 {
            let (_, _, dur) = decode_config(config);
            assert!(dur >= 2_500, "config {config} has a real frame duration");
        }
    }

    #[test]
    fn empty_packet_yields_none() {
        assert!(parse_toc(&[]).is_none());
    }

    // -- Element-level tests (drive OpusParse::process directly) -------------

    use g2g_core::frame::Frame;
    use g2g_core::memory::SystemSlice;
    use g2g_core::{FrameTiming, MemoryDomain, PushOutcome};

    #[derive(Default)]
    struct RecordingSink {
        packets: Vec<PipelinePacket>,
    }

    impl OutputSink for RecordingSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                self.packets.push(packet);
                Ok(PushOutcome::Accepted)
            })
        }
    }

    fn frame_with_bytes(seq: u64, bytes: Vec<u8>) -> Frame {
        Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: seq,
            meta: Default::default(),
        }
    }

    fn opus_caps() -> Caps {
        // Sentinel pre-parse caps: format pinned, channels/rate unknown.
        Caps::Audio { format: AudioFormat::Opus, channels: 0, sample_rate: 0 }
    }

    #[tokio::test]
    async fn emits_caps_changed_before_first_data_frame() {
        let mut parse = OpusParse::new();
        parse.configure_pipeline(&opus_caps()).unwrap();
        let mut sink = RecordingSink::default();

        let frame = frame_with_bytes(0, opus_packet(31, true, 12));
        parse.process(PipelinePacket::DataFrame(frame), &mut sink).await.unwrap();

        assert_eq!(sink.packets.len(), 2, "expected CapsChanged then DataFrame");
        match &sink.packets[0] {
            PipelinePacket::CapsChanged(Caps::Audio {
                format: AudioFormat::Opus,
                channels,
                sample_rate,
            }) => {
                assert_eq!(*channels, 2);
                assert_eq!(*sample_rate, OPUS_RATE_HZ);
            }
            other => panic!("expected Opus CapsChanged first, got {other:?}"),
        }
        assert!(matches!(sink.packets[1], PipelinePacket::DataFrame(_)));
        assert_eq!(parse.caps_changes_emitted(), 1);
    }

    #[tokio::test]
    async fn does_not_re_emit_caps_when_unchanged() {
        let mut parse = OpusParse::new();
        parse.configure_pipeline(&opus_caps()).unwrap();
        let mut sink = RecordingSink::default();

        for seq in 0..3 {
            let frame = frame_with_bytes(seq, opus_packet(9, false, 40));
            parse.process(PipelinePacket::DataFrame(frame), &mut sink).await.unwrap();
        }

        let caps_count =
            sink.packets.iter().filter(|p| matches!(p, PipelinePacket::CapsChanged(_))).count();
        assert_eq!(caps_count, 1, "CapsChanged fires once for an unchanged channel count");
        assert_eq!(parse.caps_changes_emitted(), 1);
    }

    #[tokio::test]
    async fn re_emits_caps_on_channel_change() {
        let mut parse = OpusParse::new();
        parse.configure_pipeline(&opus_caps()).unwrap();
        let mut sink = RecordingSink::default();

        // mono then stereo.
        parse
            .process(PipelinePacket::DataFrame(frame_with_bytes(0, opus_packet(9, false, 40))), &mut sink)
            .await
            .unwrap();
        parse
            .process(PipelinePacket::DataFrame(frame_with_bytes(1, opus_packet(9, true, 40))), &mut sink)
            .await
            .unwrap();

        let channels: Vec<u8> = sink
            .packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::CapsChanged(Caps::Audio { channels, .. }) => Some(*channels),
                _ => None,
            })
            .collect();
        assert_eq!(channels, vec![1, 2]);
        assert_eq!(parse.caps_changes_emitted(), 2);
    }

    #[tokio::test]
    async fn rejects_non_opus_caps_in_intercept() {
        let parse = OpusParse::new();
        let aac = Caps::Audio { format: AudioFormat::Aac, channels: 2, sample_rate: 48_000 };
        assert_eq!(parse.intercept_caps(&aac), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn caps_constraint_is_identity_any() {
        let parse = OpusParse::new();
        assert!(matches!(parse.caps_constraint_as_transform(), CapsConstraint::IdentityAny));
    }
}
