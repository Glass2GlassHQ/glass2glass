//! AAC ADTS access-unit parser that refines source-side `Caps`.
//!
//! The audio sibling of `h264parse` / `h265parse`: it scans each `DataFrame`
//! for an ADTS header (12-bit `0xFFF` syncword) and recovers the channel count
//! and sample rate, emitting a `CapsChanged` before forwarding the frame. This
//! lets a raw ADTS AAC elementary stream be restreamed or muxed with concrete
//! channel/rate caps.
//!
//! `Caps::Audio` has no open (`Any`) field, so a source advertising AAC before
//! the first header lands uses sentinel `channels`/`sample_rate` 0; the
//! negotiated constraint is therefore `IdentityAny` (forward whatever AAC the
//! upstream produces) rather than the video parsers' `Identity(any geometry)`.
//! The AAC-only guard lives in `intercept_caps`.
//!
//! Scope is ADTS, the common elementary-stream framing. LATM / LOAS (the
//! MPEG-TS / broadcast framing) is deferred. The ADTS header is plain bit
//! fields: no exp-Golomb, no emulation prevention, so this needs none of the
//! `annexb` machinery the H.264 / H.265 parsers share.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    G2gError, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
};

/// ADTS sampling-frequency-index table (ISO/IEC 14496-3). Indices 13/14 are
/// reserved and 15 (explicit rate) is forbidden in ADTS, so only 0..=12 map.
pub(crate) const SAMPLE_RATES: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350,
];

/// Synthesise the 2-byte AAC AudioSpecificConfig from an ADTS header.
pub(crate) fn asc_from_adts(au: &[u8]) -> Option<[u8; 2]> {
    if au.len() < 7 || au[0] != 0xFF || (au[1] & 0xF0) != 0xF0 {
        return None;
    }
    let object_type = ((au[2] >> 6) & 0x03) + 1; // profile + 1
    let sr_index = (au[2] >> 2) & 0x0F;
    let channel_config = ((au[2] & 0x01) << 2) | ((au[3] >> 6) & 0x03);
    Some([
        (object_type << 3) | (sr_index >> 1),
        ((sr_index & 1) << 7) | (channel_config << 3),
    ])
}

/// Strip the ADTS header (7 bytes, or 9 with CRC) from an AAC access unit.
pub(crate) fn strip_adts(au: &[u8]) -> &[u8] {
    if au.len() >= 7 && au[0] == 0xFF && (au[1] & 0xF0) == 0xF0 {
        let header = if au[1] & 0x01 == 0 { 9 } else { 7 }; // protection_absent==0 -> CRC
        au.get(header..).unwrap_or(&[])
    } else {
        au
    }
}

/// Build an ADTS-framed AAC access unit from the track's 2-byte
/// AudioSpecificConfig and the raw access unit: a 7-byte ADTS header (no CRC)
/// derived from the ASC's audio-object-type, sampling-frequency index, and
/// channel configuration, then the AU. The inverse of the muxers' de-ADTS write,
/// so the demuxed audio is self-describing. `None` when the ASC is too short, the
/// rate index / channel config is out of range, or the frame exceeds the 13-bit
/// ADTS length (then the AU is forwarded raw). Shared by the MP4 and FLV
/// demuxers (M662).
pub(crate) fn adts_from_asc(asc: &[u8], au: &[u8]) -> Option<Vec<u8>> {
    if asc.len() < 2 {
        return None;
    }
    let aot = asc[0] >> 3; // audio object type (5 bits)
    let sr_index = ((asc[0] & 0x07) << 1) | (asc[1] >> 7);
    let channel_config = (asc[1] >> 3) & 0x0F;
    if sr_index > 12 || channel_config == 0 {
        return None; // reserved/explicit rate or "config in stream": not ADTS-able
    }
    let profile = aot.saturating_sub(1) & 0x03; // ADTS profile = AOT - 1
    let frame_len = au.len() + 7;
    if frame_len > 0x1FFF {
        return None; // ADTS frame_length is 13 bits
    }
    let mut out = Vec::with_capacity(frame_len);
    out.extend_from_slice(&[
        0xFF,
        0xF1, // syncword | MPEG-4 | layer 0 | protection_absent (no CRC)
        (profile << 6) | (sr_index << 2) | ((channel_config >> 2) & 1),
        ((channel_config & 3) << 6) | ((frame_len >> 11) & 3) as u8,
        ((frame_len >> 3) & 0xFF) as u8,
        (((frame_len & 7) << 5) as u8) | 0x1F, // buffer fullness (top bits)
        0xFC, // buffer fullness (low) | num_raw_data_blocks = 0
    ]);
    out.extend_from_slice(au);
    Some(out)
}

#[derive(Debug, Default)]
pub struct AacParse {
    configured: bool,
    last_emitted_caps: Option<Caps>,
    headers_emitted: u64,
}

impl AacParse {
    pub fn new() -> Self {
        Self::default()
    }

    /// Count of `CapsChanged` packets pushed downstream, for tests asserting
    /// re-emission is suppressed when the ADTS parameters are unchanged.
    pub fn caps_changes_emitted(&self) -> u64 {
        self.headers_emitted
    }
}

impl AsyncElement for AacParse {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::Audio { format: AudioFormat::Aac, .. } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Pass-through identity over AAC of any channel/rate (the parser refines
    /// those mid-stream from the ADTS header but never changes media type).
    /// `IdentityAny`, not `Identity(set)`, because audio caps cannot express
    /// "AAC at any channels/rate" in a single `Caps`.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::Audio { format: AudioFormat::Aac, .. } => {
                self.configured = true;
                Ok(ConfigureOutcome::Accepted)
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "AAC parser",
            "Codec/Parser/Audio",
            "Parses an AAC ADTS stream and refines caps",
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
                        if let Some(info) = parse_adts(slice.as_slice()) {
                            let new_caps = Caps::Audio {
                                format: AudioFormat::Aac,
                                channels: info.channels,
                                sample_rate: info.sample_rate,
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

impl PadTemplates for AacParse {
    fn pad_templates() -> Vec<PadTemplate> {
        // `Caps::Audio` has no open dims; pin the common stereo/48 kHz shape.
        let aac = Caps::Audio { format: AudioFormat::Aac, channels: 2, sample_rate: 48_000 };
        Vec::from([
            PadTemplate::sink(CapsSet::one(aac.clone())),
            PadTemplate::source(CapsSet::one(aac)),
        ])
    }
}

/// Channel count and sample rate recovered from an ADTS header.
struct AdtsInfo {
    channels: u8,
    sample_rate: u32,
}

/// Scan `au` for the first valid ADTS header and decode its channel count and
/// sample rate. `None` if no header parses (no syncword, reserved sampling
/// index, or a channel configuration that doesn't pin a channel count).
fn parse_adts(au: &[u8]) -> Option<AdtsInfo> {
    // The fixed header is 7 bytes; we read fields up to byte 3.
    let last = au.len().checked_sub(7)?;
    for i in 0..=last {
        // Syncword 0xFFF (12 bits) + layer 00: byte0 all ones, byte1 high
        // nibble all ones and the two layer bits zero.
        if au[i] != 0xFF || (au[i + 1] & 0xF6) != 0xF0 {
            continue;
        }
        let b2 = au[i + 2];
        let b3 = au[i + 3];
        let freq_index = ((b2 >> 2) & 0x0F) as usize;
        let channel_config = ((b2 & 0x01) << 2) | (b3 >> 6);
        let Some(&sample_rate) = SAMPLE_RATES.get(freq_index) else {
            continue;
        };
        let channels = match channel_config {
            1..=6 => channel_config, // 1ch..5.1
            7 => 8,                  // 7.1
            _ => continue,           // 0 = carried in the AOT config, not ADTS
        };
        return Some(AdtsInfo { channels, sample_rate });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Build a 7-byte (no-CRC) ADTS header for `channel_config` at
    /// `freq_index`, followed by `payload_len` zero bytes, framed as one AAC-LC
    /// access unit.
    fn adts_frame(channel_config: u8, freq_index: u8, payload_len: usize) -> Vec<u8> {
        let frame_len = 7 + payload_len;
        let profile = 1u8; // AAC-LC (AOT 2, profile = AOT - 1)
        let mut f = vec![0u8; frame_len];
        f[0] = 0xFF;
        f[1] = 0xF1; // syncword low, MPEG-4, layer 00, protection_absent = 1
        f[2] = (profile << 6) | ((freq_index & 0x0F) << 2) | ((channel_config >> 2) & 0x01);
        f[3] = ((channel_config & 0x03) << 6) | (((frame_len >> 11) & 0x03) as u8);
        f[4] = ((frame_len >> 3) & 0xFF) as u8;
        f[5] = (((frame_len & 0x07) << 5) as u8) | 0x1F;
        f[6] = 0xFC; // buffer fullness low + num_raw_blocks (0)
        f
    }

    #[test]
    fn recovers_stereo_44100() {
        let info = parse_adts(&adts_frame(2, 4, 16)).expect("ADTS must parse");
        assert_eq!((info.channels, info.sample_rate), (2, 44_100));
    }

    #[test]
    fn recovers_mono_48000() {
        let info = parse_adts(&adts_frame(1, 3, 8)).expect("ADTS must parse");
        assert_eq!((info.channels, info.sample_rate), (1, 48_000));
    }

    #[test]
    fn maps_channel_config_7_to_eight_channels() {
        let info = parse_adts(&adts_frame(7, 3, 8)).expect("ADTS must parse");
        assert_eq!(info.channels, 8);
    }

    #[test]
    fn rejects_reserved_sampling_index() {
        // freq_index 13 is reserved; no valid rate, so the header is skipped.
        assert!(parse_adts(&adts_frame(2, 13, 16)).is_none());
    }

    #[test]
    fn rejects_channel_config_zero() {
        // config 0 means the channel count lives in the AOT config, not ADTS.
        assert!(parse_adts(&adts_frame(0, 4, 16)).is_none());
    }

    #[test]
    fn finds_header_after_leading_bytes() {
        let mut stream = vec![0x00, 0x11, 0x22];
        stream.extend_from_slice(&adts_frame(2, 4, 16));
        let info = parse_adts(&stream).expect("ADTS after junk must parse");
        assert_eq!((info.channels, info.sample_rate), (2, 44_100));
    }

    #[test]
    fn returns_none_on_non_adts() {
        assert!(parse_adts(&[0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0x00]).is_none());
        assert!(parse_adts(&[]).is_none());
    }

    // -- Element-level tests (drive AacParse::process directly) -------------

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

    fn aac_caps() -> Caps {
        // Sentinel pre-parse caps: format pinned, channels/rate unknown.
        Caps::Audio { format: AudioFormat::Aac, channels: 0, sample_rate: 0 }
    }

    #[tokio::test]
    async fn emits_caps_changed_before_first_data_frame() {
        let mut parse = AacParse::new();
        parse.configure_pipeline(&aac_caps()).unwrap();
        let mut sink = RecordingSink::default();

        let frame = frame_with_bytes(0, adts_frame(2, 4, 16));
        parse
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        assert_eq!(sink.packets.len(), 2, "expected CapsChanged then DataFrame");
        match &sink.packets[0] {
            PipelinePacket::CapsChanged(Caps::Audio {
                format: AudioFormat::Aac,
                channels,
                sample_rate,
            }) => {
                assert_eq!(*channels, 2);
                assert_eq!(*sample_rate, 44_100);
            }
            other => panic!("expected AAC CapsChanged first, got {other:?}"),
        }
        assert!(matches!(sink.packets[1], PipelinePacket::DataFrame(_)));
        assert_eq!(parse.caps_changes_emitted(), 1);
    }

    #[tokio::test]
    async fn does_not_re_emit_caps_when_unchanged() {
        let mut parse = AacParse::new();
        parse.configure_pipeline(&aac_caps()).unwrap();
        let mut sink = RecordingSink::default();

        for seq in 0..3 {
            let frame = frame_with_bytes(seq, adts_frame(2, 4, 16));
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
        assert_eq!(caps_count, 1, "CapsChanged fires once for identical ADTS params");
        assert_eq!(parse.caps_changes_emitted(), 1);
    }

    #[tokio::test]
    async fn re_emits_caps_on_parameter_change() {
        let mut parse = AacParse::new();
        parse.configure_pipeline(&aac_caps()).unwrap();
        let mut sink = RecordingSink::default();

        // stereo/44100 then mono/48000.
        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(0, adts_frame(2, 4, 16))),
                &mut sink,
            )
            .await
            .unwrap();
        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(1, adts_frame(1, 3, 8))),
                &mut sink,
            )
            .await
            .unwrap();

        let params: Vec<(u8, u32)> = sink
            .packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::CapsChanged(Caps::Audio { channels, sample_rate, .. }) => {
                    Some((*channels, *sample_rate))
                }
                _ => None,
            })
            .collect();
        assert_eq!(params, vec![(2, 44_100), (1, 48_000)]);
        assert_eq!(parse.caps_changes_emitted(), 2);
    }

    #[tokio::test]
    async fn rejects_non_aac_caps_in_intercept() {
        let parse = AacParse::new();
        let pcm = Caps::Audio { format: AudioFormat::PcmS16Le, channels: 2, sample_rate: 48_000 };
        assert_eq!(parse.intercept_caps(&pcm), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn caps_constraint_is_identity_any() {
        let parse = AacParse::new();
        assert!(matches!(
            parse.caps_constraint_as_transform(),
            CapsConstraint::IdentityAny
        ));
    }
}
