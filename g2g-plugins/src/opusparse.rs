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
//! its channel count in the `OpusHead` header, not the per-packet TOC; when an
//! `OpusHead` (RFC 7845) arrives in-band it is parsed for the authoritative
//! channel count and consumed (it is codec config, not audio), and that count
//! then overrides the TOC for every following packet. Opus always decodes at
//! 48 kHz regardless of the coded bandwidth, so the sample rate is the constant
//! [`OPUS_RATE_HZ`].

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

/// Number of 48 kHz samples *one coded frame* of an Opus packet covers, from its
/// TOC byte (RFC 6716 §3.1). `0` for an empty packet. This is also the span an
/// in-band FEC (LBRR) copy carried by the packet covers, which is what
/// [`crate::opusdec`] recovers a lost packet with.
pub(crate) fn packet_frame_samples(pkt: &[u8]) -> u32 {
    parse_toc(pkt).map_or(0, |t| {
        ((t.frame_duration_us as u64 * OPUS_RATE_HZ as u64) / 1_000_000) as u32
    })
}

/// Number of 48 kHz samples an Opus packet decodes to: the per-frame duration
/// from the TOC config times the frame count code (low 2 bits). Opus is always
/// 48 kHz, so this maps directly to a duration. `0` for an empty packet. Shared
/// by [`crate::oggdemux`] (packet timing) and [`crate::oggmux`] (granule
/// positions).
pub(crate) fn packet_samples(pkt: &[u8]) -> u32 {
    let frame_samples = packet_frame_samples(pkt);
    if frame_samples == 0 {
        return 0;
    }
    let toc = pkt[0];
    let frames: u32 = match toc & 0x3 {
        0 => 1,
        1 | 2 => 2,
        // Code 3: the frame count is the low 6 bits of the following byte.
        _ => pkt.get(1).map(|b| (b & 0x3F) as u32).unwrap_or(1).max(1),
    };
    frame_samples.saturating_mul(frames)
}

/// Pre-skip written into a synthesized Opus header: libopus' encoder lookahead
/// at 48 kHz, which is what [`crate::opusenc::OpusEnc`] (and ffmpeg's libopus
/// wrapper) actually delays its output by. A remuxed stream carries the source
/// header instead, so this only applies to a freshly encoded one.
pub(crate) const OPUS_ENCODER_PRE_SKIP: u16 = 312;

/// Fixed part of an RFC 7845 `OpusHead`: 8-byte magic, version, channel count,
/// pre-skip, input sample rate, output gain, channel mapping family.
const OPUS_HEAD_FIXED: usize = 19;

/// A synthesized `OpusHead` (RFC 7845 §5.1) for a stream that carried none: a
/// freshly encoded one, where the pre-skip is the encoder's lookahead. Channel
/// mapping family 0, which is defined for mono and stereo only, so the count is
/// clamped to that. Used by every muxer that has to invent a header.
pub(crate) fn synth_opus_head(channels: u8, sample_rate: u32) -> Vec<u8> {
    let mut h = Vec::from(*b"OpusHead");
    h.push(1); // version
    h.push(channels.clamp(1, 2));
    h.extend_from_slice(&OPUS_ENCODER_PRE_SKIP.to_le_bytes());
    h.extend_from_slice(&sample_rate.max(1).to_le_bytes()); // original input rate
    h.extend_from_slice(&0i16.to_le_bytes()); // output gain
    h.push(0); // channel mapping family
    h
}

/// Fixed part of an RFC 8316 `dOps` payload: the same fields big-endian, with
/// the magic and version byte replaced by a single version byte. Only the MP4
/// elements speak `dOps`, and those are `std`-gated, hence the cfg on this and
/// the two converters below.
#[cfg(feature = "std")]
const DOPS_FIXED: usize = 11;

/// Whether `packet` is Opus codec config rather than audio: the RFC 7845
/// identification (`OpusHead`) or comment (`OpusTags`) header. Audio packets
/// start with a TOC byte, so the 8-byte magic separates them.
pub(crate) fn is_opus_config(packet: &[u8]) -> bool {
    packet.starts_with(b"OpusHead") || packet.starts_with(b"OpusTags")
}

/// Channel count and pre-skip from an in-band `OpusHead` (RFC 7845), or `None`
/// if `packet` is not one. Offset 9 is the total channel count for every mapping
/// family (family 1 multichannel included, which the per-packet TOC cannot
/// recover), offset 10 the LE u16 pre-skip. Family != 0 appends a channel
/// mapping table past the fixed part, which this does not read.
pub(crate) fn parse_opus_head(packet: &[u8]) -> Option<(u8, u16)> {
    let fixed = packet.get(..OPUS_HEAD_FIXED)?;
    if !fixed.starts_with(b"OpusHead") {
        return None;
    }
    let channels = fixed[9];
    let pre_skip = u16::from_le_bytes([fixed[10], fixed[11]]);
    (channels >= 1).then_some((channels, pre_skip))
}

/// Length of a channel mapping table for `channels` channels: stream count,
/// coupled count, then one output-channel index per channel (RFC 7845 §5.1.1).
/// Zero for mapping family 0, which has no table.
#[cfg(feature = "std")]
fn mapping_len(family: u8, channels: u8) -> usize {
    if family == 0 {
        0
    } else {
        2 + channels as usize
    }
}

/// Build an RFC 7845 `OpusHead` from an RFC 8316 `dOps` payload (the
/// OpusSpecificBox body, version byte first). The two carry the same fields;
/// `dOps` is big-endian and drops the magic, `OpusHead` is little-endian.
/// `None` for an unknown version, a zero channel count, or a truncated
/// channel-mapping table: every field is attacker-controlled.
#[cfg(feature = "std")]
pub(crate) fn opus_head_from_dops(dops: &[u8]) -> Option<Vec<u8>> {
    let fixed = dops.get(..DOPS_FIXED)?;
    if fixed[0] != 0 {
        return None; // OpusSpecificBox Version
    }
    let channels = fixed[1];
    if channels == 0 {
        return None;
    }
    let pre_skip = u16::from_be_bytes([fixed[2], fixed[3]]);
    let input_rate = u32::from_be_bytes([fixed[4], fixed[5], fixed[6], fixed[7]]);
    let output_gain = i16::from_be_bytes([fixed[8], fixed[9]]);
    let family = fixed[10];
    let mapping = dops.get(DOPS_FIXED..DOPS_FIXED + mapping_len(family, channels))?;

    let mut head = Vec::from(*b"OpusHead");
    head.push(1); // OpusHead version
    head.push(channels);
    head.extend_from_slice(&pre_skip.to_le_bytes());
    head.extend_from_slice(&input_rate.to_le_bytes());
    head.extend_from_slice(&output_gain.to_le_bytes());
    head.push(family);
    head.extend_from_slice(mapping);
    Some(head)
}

/// The inverse of [`opus_head_from_dops`]: the `dOps` payload carrying an in-band
/// `OpusHead`'s fields, so a remux into MP4 writes the source's real pre-skip,
/// output gain and channel mapping. `None` for anything that is not a complete
/// `OpusHead`.
#[cfg(feature = "std")]
pub(crate) fn dops_from_opus_head(head: &[u8]) -> Option<Vec<u8>> {
    let fixed = head.get(..OPUS_HEAD_FIXED)?;
    if !fixed.starts_with(b"OpusHead") {
        return None;
    }
    let channels = fixed[9];
    if channels == 0 {
        return None;
    }
    let pre_skip = u16::from_le_bytes([fixed[10], fixed[11]]);
    let input_rate = u32::from_le_bytes([fixed[12], fixed[13], fixed[14], fixed[15]]);
    let output_gain = i16::from_le_bytes([fixed[16], fixed[17]]);
    let family = fixed[18];
    let mapping = head.get(OPUS_HEAD_FIXED..OPUS_HEAD_FIXED + mapping_len(family, channels))?;

    let mut dops = Vec::new();
    dops.push(0); // OpusSpecificBox Version
    dops.push(channels);
    dops.extend_from_slice(&pre_skip.to_be_bytes());
    dops.extend_from_slice(&input_rate.to_be_bytes());
    dops.extend_from_slice(&output_gain.to_be_bytes());
    dops.push(family);
    dops.extend_from_slice(mapping);
    Some(dops)
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::opusparse::OpusParse;
///
/// let parse = OpusParse::new();
/// ```
#[derive(Debug, Default)]
pub struct OpusParse {
    configured: bool,
    last_emitted_caps: Option<Caps>,
    headers_emitted: u64,
    /// Authoritative channel count from an in-band `OpusHead` (RFC 7845), once
    /// seen. It overrides the per-packet TOC's mono/stereo guess, so a
    /// multichannel (mapping family 1) stream keeps its real channel count
    /// instead of collapsing to the TOC's internal-stream stereo bit.
    header_channels: Option<u8>,
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

    /// The caps to emit for `channels`, or `None` if unchanged from the last.
    /// Records the new caps and bumps the emit counter when it does change.
    fn caps_update(&mut self, channels: u8) -> Option<Caps> {
        let new_caps = Caps::Audio {
            format: AudioFormat::Opus,
            channels,
            sample_rate: OPUS_RATE_HZ,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        if self.last_emitted_caps.as_ref() == Some(&new_caps) {
            return None;
        }
        self.last_emitted_caps = Some(new_caps.clone());
        self.headers_emitted += 1;
        Some(new_caps)
    }
}

impl AsyncElement for OpusParse {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::Audio {
                format: AudioFormat::Opus,
                ..
            } => Ok(upstream_caps.clone()),
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
            Caps::Audio {
                format: AudioFormat::Opus,
                ..
            } => {
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
                        let bytes = slice.as_slice();
                        // An in-band OpusHead is codec config, not audio: parse the
                        // authoritative channel count from it, emit caps, and
                        // consume it (do not forward as a DataFrame).
                        if let Some((ch, _)) = parse_opus_head(bytes) {
                            self.header_channels = Some(ch);
                            if let Some(caps) = self.caps_update(ch) {
                                out.push(PipelinePacket::CapsChanged(caps)).await?;
                            }
                            return Ok(());
                        }
                        // Prefer the OpusHead count when known; else the TOC's
                        // mono/stereo bit (mapping family 0).
                        let channels = self
                            .header_channels
                            .or_else(|| parse_toc(bytes).map(|t| t.channels));
                        if let Some(ch) = channels {
                            if let Some(caps) = self.caps_update(ch) {
                                out.push(PipelinePacket::CapsChanged(caps)).await?;
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
        let opus = Caps::Audio {
            format: AudioFormat::Opus,
            channels: 2,
            sample_rate: OPUS_RATE_HZ,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
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
    Some(OpusToc {
        channels,
        mode,
        bandwidth,
        frame_duration_us,
    })
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

/// Fuzzing entry: parse both the Opus identification header (`OpusHead`) and a
/// packet's TOC byte / frame-count structure. Exposed only under `--cfg fuzzing`
/// (cargo-fuzz) so the normal public API is unchanged.
#[cfg(fuzzing)]
pub fn fuzz_parse(data: &[u8]) {
    let _ = parse_opus_head(data);
    let _ = parse_toc(data);
    #[cfg(feature = "std")]
    {
        let _ = opus_head_from_dops(data);
        let _ = dops_from_opus_head(data);
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
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            core::task::Poll::Ready({
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
        Caps::Audio {
            format: AudioFormat::Opus,
            channels: 0,
            sample_rate: 0,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }

    #[tokio::test]
    async fn emits_caps_changed_before_first_data_frame() {
        let mut parse = OpusParse::new();
        parse.configure_pipeline(&opus_caps()).unwrap();
        let mut sink = RecordingSink::default();

        let frame = frame_with_bytes(0, opus_packet(31, true, 12));
        parse
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        assert_eq!(sink.packets.len(), 2, "expected CapsChanged then DataFrame");
        match &sink.packets[0] {
            PipelinePacket::CapsChanged(Caps::Audio {
                format: AudioFormat::Opus,
                channels,
                sample_rate,
                ..
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
            parse
                .process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .unwrap();
        }

        let caps_count = sink
            .packets
            .iter()
            .filter(|p| matches!(p, PipelinePacket::CapsChanged(_)))
            .count();
        assert_eq!(
            caps_count, 1,
            "CapsChanged fires once for an unchanged channel count"
        );
        assert_eq!(parse.caps_changes_emitted(), 1);
    }

    #[tokio::test]
    async fn re_emits_caps_on_channel_change() {
        let mut parse = OpusParse::new();
        parse.configure_pipeline(&opus_caps()).unwrap();
        let mut sink = RecordingSink::default();

        // mono then stereo.
        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(0, opus_packet(9, false, 40))),
                &mut sink,
            )
            .await
            .unwrap();
        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(1, opus_packet(9, true, 40))),
                &mut sink,
            )
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

    /// An RFC 7845 OpusHead identification header for `channels` channels at the
    /// given mapping `family` (family != 0 appends an identity mapping table).
    fn opus_head(channels: u8, family: u8) -> Vec<u8> {
        let mut h = b"OpusHead".to_vec();
        h.push(1); // version
        h.push(channels);
        h.extend_from_slice(&[0, 0]); // pre-skip
        h.extend_from_slice(&48_000u32.to_le_bytes()); // input sample rate
        h.extend_from_slice(&[0, 0]); // output gain
        h.push(family);
        if family != 0 {
            h.push(1); // stream count
            h.push(0); // coupled count
            for i in 0..channels {
                h.push(i); // identity channel mapping
            }
        }
        h
    }

    #[tokio::test]
    async fn opus_head_recovers_multichannel_and_locks_channel_count() {
        let mut parse = OpusParse::new();
        parse.configure_pipeline(&opus_caps()).unwrap();
        let mut sink = RecordingSink::default();

        // A 6-channel family-1 OpusHead: caps report 6 channels, and the header
        // is consumed (codec config, never forwarded as an audio frame).
        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(0, opus_head(6, 1))),
                &mut sink,
            )
            .await
            .unwrap();
        assert_eq!(
            sink.packets.len(),
            1,
            "OpusHead emits caps only, no DataFrame"
        );
        match &sink.packets[0] {
            PipelinePacket::CapsChanged(Caps::Audio { channels, .. }) => assert_eq!(*channels, 6),
            other => panic!("expected 6-channel CapsChanged, got {other:?}"),
        }

        // A following stereo-TOC audio packet must not downgrade to 2 channels:
        // the header count wins. The frame is forwarded, no new CapsChanged.
        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(1, opus_packet(31, true, 12))),
                &mut sink,
            )
            .await
            .unwrap();
        assert_eq!(
            parse.caps_changes_emitted(),
            1,
            "TOC did not override the header count"
        );
        assert!(matches!(
            sink.packets.last().unwrap(),
            PipelinePacket::DataFrame(_)
        ));
    }

    #[tokio::test]
    async fn opus_head_family_zero_also_sets_channels() {
        let mut parse = OpusParse::new();
        parse.configure_pipeline(&opus_caps()).unwrap();
        let mut sink = RecordingSink::default();
        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(0, opus_head(2, 0))),
                &mut sink,
            )
            .await
            .unwrap();
        assert_eq!(sink.packets.len(), 1, "header consumed");
        assert!(matches!(
            sink.packets[0],
            PipelinePacket::CapsChanged(Caps::Audio { channels: 2, .. })
        ));
    }

    #[test]
    fn parse_opus_head_needs_magic_and_length() {
        assert_eq!(parse_opus_head(&opus_head(8, 1)), Some((8, 0)));
        assert_eq!(parse_opus_head(b"OpusHeadtooshort"), None, "under 19 bytes");
        assert_eq!(
            parse_opus_head(&opus_packet(31, true, 12)),
            None,
            "a TOC packet is not a header"
        );
    }

    /// `OpusHead` <-> `dOps` is the same field set in the other byte order, so a
    /// round trip must be the identity, channel-mapping table included.
    #[cfg(feature = "std")]
    #[test]
    fn opus_head_and_dops_round_trip_including_the_mapping_table() {
        for head in [opus_head(2, 0), opus_head(6, 1)] {
            let dops = dops_from_opus_head(&head).expect("a header converts");
            assert_eq!(dops[0], 0, "OpusSpecificBox version");
            assert_eq!(dops[1], head[9], "channel count");
            assert_eq!(
                u16::from_be_bytes([dops[2], dops[3]]),
                u16::from_le_bytes([head[10], head[11]]),
                "pre-skip flips endianness"
            );
            assert_eq!(
                opus_head_from_dops(&dops).as_deref(),
                Some(&head[..]),
                "the round trip is the identity"
            );
        }
    }

    /// Every `dOps` field is attacker-controlled: a bad version, a zero channel
    /// count, a truncated fixed part or a truncated mapping table must all
    /// decline rather than read past the box.
    #[cfg(feature = "std")]
    #[test]
    fn malformed_dops_declines() {
        let good = dops_from_opus_head(&opus_head(6, 1)).unwrap();
        assert!(opus_head_from_dops(&good).is_some(), "the baseline parses");

        let mut bad_version = good.clone();
        bad_version[0] = 1;
        assert_eq!(opus_head_from_dops(&bad_version), None, "unknown version");

        let mut no_channels = good.clone();
        no_channels[1] = 0;
        assert_eq!(opus_head_from_dops(&no_channels), None, "zero channels");

        for len in 0..good.len() {
            assert_eq!(
                opus_head_from_dops(&good[..len]),
                None,
                "a {len}-byte dOps is truncated"
            );
        }
        // The same truncation discipline the other way.
        let head = opus_head(6, 1);
        for len in 0..head.len() {
            assert_eq!(
                dops_from_opus_head(&head[..len]),
                None,
                "a {len}-byte OpusHead is truncated"
            );
        }
    }

    #[tokio::test]
    async fn rejects_non_opus_caps_in_intercept() {
        let parse = OpusParse::new();
        let aac = Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        assert_eq!(parse.intercept_caps(&aac), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn caps_constraint_is_identity_any() {
        let parse = OpusParse::new();
        assert!(matches!(
            parse.caps_constraint_as_transform(),
            CapsConstraint::IdentityAny
        ));
    }
}
