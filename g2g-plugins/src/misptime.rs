//! MISB ST 0604 MISP precision time stamps carried in the video bitstream
//! (M809, `no_std`): in a STANAG 4609 motion imagery stream every coded frame
//! carries an absolute microsecond time in an SEI `user_data_unregistered`
//! message, so the video stays correlated with the ST 0601 KLV telemetry (see
//! [`crate::klv`]) through remux and transcode. [`MispTimeInsert`] stamps a
//! compressed stream, [`MispTimeExtract`] mines the stamp back out.
//!
//! Wire layout, verified against GStreamer's shipping parser
//! (`gst-plugins-base/gst-libs/gst/video/video-sei.c`, function
//! `gst_video_sei_user_data_unregistered_parse_precision_time_stamp`, and the
//! UUID table in `video-sei.h`) plus the real-capture test vector in
//! `gst-plugins-bad/tests/check/elements/h264parse.c`
//! (`test_parse_sei_userdefinedunregistered`), cross-checked against the ST 0604
//! and ST 0603 text for the identifiers, the status byte, and the epoch (the
//! time is microseconds since 1970-01-01 UTC, not counting leap seconds). The
//! SEI payload is 28 bytes:
//!
//! ```text
//! [0..16)  UUID                 (per codec / unit, see the constants below)
//! [16]     status byte
//! [17..19) time bits 63..48     (big endian)
//! [19]     0xFF                 start code emulation prevention
//! [20..22) time bits 47..32
//! [22]     0xFF
//! [23..25) time bits 31..16
//! [25]     0xFF
//! [26..28) time bits 15..0
//! ```
//!
//! The three 0xFF bytes are the format's own emulation prevention: they cap any
//! zero run in the time field at two bytes, so the encoded SEI never needs an
//! RBSP escape and its byte offsets stay fixed. They are mandatory, so a payload
//! missing them is rejected rather than decoded into a plausible-looking time.

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, Rate, TextFormat, VideoCodec,
};

use crate::annexb::{
    add_emulation_prevention, h264_nal_type, h265_nal_type, nal_units_any, read_ff_extended,
    strip_emulation_prevention, vcl_start,
};
use crate::subparse::{Cue, CueSettings};

use core::future::Future;
use core::pin::Pin;

/// SEI `payloadType` 5, `user_data_unregistered`.
const SEI_USER_DATA_UNREGISTERED: usize = 5;

/// A MISP time SEI payload: 16-byte UUID plus the 12-byte body.
const MISP_PAYLOAD_LEN: usize = 28;

/// H.264 MISP microsecond time identifier, the ASCII bytes `MISPmicrosectime`.
/// ST 0604 notes this is not a valid UUID, which is why H.265 got a real one.
pub const MISP_MICROSECTIME_H264: [u8; 16] = *b"MISPmicrosectime";

/// H.265 MISP microsecond time UUID `a8687dd4-d759-3758-a5ce-f0338b6545f1`, the
/// version-3 UUID ST 0604 derives from the string `MISPmicrosectime-v2`.
pub const MISP_MICROSECTIME_H265: [u8; 16] = [
    0xA8, 0x68, 0x7D, 0xD4, 0xD7, 0x59, 0x37, 0x58, 0xA5, 0xCE, 0xF0, 0x33, 0x8B, 0x65, 0x45, 0xF1,
];

/// H.265 MISP nanosecond time UUID `cf848278-ee23-306c-9265-e8fef22fb8b8`.
/// Parsed but never written: g2g stamps the microsecond form, which is what
/// STANAG 4609 class 1 streams carry. These bytes come from GStreamer's table
/// alone; unlike the two above they were not cross-checked against ST 0604.
pub const MISP_NANOSECTIME_H265: [u8; 16] = [
    0xCF, 0x84, 0x82, 0x78, 0xEE, 0x23, 0x30, 0x6C, 0x92, 0x65, 0xE8, 0xFE, 0xF2, 0x2F, 0xB8, 0xB8,
];

/// Default status byte: locked, normal, forward, reserved bits set. Per ST 0603
/// the polarity is inverted from the intuitive reading, 0 is the healthy state:
/// bit 7 is 0 when the clock is locked to an absolute time reference, bit 6 is 0
/// when time increments linearly, bit 5 is 0 for a forward jump, and bits 4..0
/// are reserved. Real captures carry 0x1F (the GStreamer h264parse vector above).
pub const MISP_STATUS_DEFAULT: u8 = 0x1F;

/// The unit of a recovered [`MispTime`], fixed by which UUID the SEI carried.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MispTimeUnit {
    Microseconds,
    Nanoseconds,
}

/// A MISP time stamp recovered from an access unit's SEI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MispTime {
    /// The ST 0604 status byte, verbatim.
    pub status: u8,
    /// The 64-bit count, in [`unit`](Self::unit).
    pub value: u64,
    pub unit: MispTimeUnit,
}

impl MispTime {
    /// The time in microseconds since the UNIX epoch.
    pub fn micros(self) -> u64 {
        match self.unit {
            MispTimeUnit::Microseconds => self.value,
            MispTimeUnit::Nanoseconds => self.value / 1_000,
        }
    }

    /// The time in nanoseconds since the UNIX epoch.
    pub fn nanos(self) -> u64 {
        match self.unit {
            MispTimeUnit::Microseconds => self.value.saturating_mul(1_000),
            MispTimeUnit::Nanoseconds => self.value,
        }
    }
}

/// Build the Annex-B SEI NAL (4-byte start code) carrying `micros` as a MISP
/// microsecond time, the inverse of [`extract_misp_time`]. The NAL header is the
/// codec's SEI form (H.264 type 6, one byte; H.265 prefix-SEI type 39, two
/// bytes) and the UUID is the codec's microsecond UUID.
pub fn build_misp_time_sei(micros: u64, status: u8, codec: VideoCodec) -> Vec<u8> {
    let uuid = match codec {
        VideoCodec::H265 => MISP_MICROSECTIME_H265,
        _ => MISP_MICROSECTIME_H264,
    };
    let t = micros.to_be_bytes();
    let mut payload = Vec::with_capacity(MISP_PAYLOAD_LEN);
    payload.extend_from_slice(&uuid);
    payload.push(status);
    for (pair, chunk) in t.chunks_exact(2).enumerate() {
        if pair > 0 {
            payload.push(0xFF);
        }
        payload.extend_from_slice(chunk);
    }

    // payloadType and payloadSize are both under 0xFF, so neither needs the SEI
    // 0xFF run extension.
    let mut rbsp = Vec::with_capacity(MISP_PAYLOAD_LEN + 3);
    rbsp.push(SEI_USER_DATA_UNREGISTERED as u8);
    rbsp.push(MISP_PAYLOAD_LEN as u8);
    rbsp.extend_from_slice(&payload);
    rbsp.push(0x80); // rbsp_trailing_bits

    let mut nal = alloc::vec![0x00, 0x00, 0x00, 0x01];
    match codec {
        // H.265 prefix-SEI: nal_unit_type 39, layer 0, tid 1 -> 0x4E 0x01.
        VideoCodec::H265 => nal.extend_from_slice(&[0x4E, 0x01]),
        // H.264 SEI: nal_unit_type 6.
        _ => nal.push(0x06),
    }
    nal.extend_from_slice(&add_emulation_prevention(&rbsp));
    nal
}

/// Recover the first MISP time stamp from an access unit (Annex-B or AVCC
/// framed), or `None` when it carries none. Every length and offset comes from
/// the bitstream, so all of it is bounds-checked: a truncated or malformed SEI
/// yields `None` rather than a wrong time.
pub fn extract_misp_time(au: &[u8], codec: VideoCodec) -> Option<MispTime> {
    for nal in nal_units_any(au) {
        // SEI NAL header + RBSP offset differ by codec: H.264 SEI is NAL type 6
        // with a 1-byte header; H.265 prefix-SEI (39) / suffix-SEI (40) carry a
        // 2-byte header.
        let rbsp_off = match codec {
            VideoCodec::H265 => match h265_nal_type(nal) {
                Some(39) | Some(40) => 2,
                _ => continue,
            },
            _ => match h264_nal_type(nal) {
                Some(6) => 1,
                _ => continue,
            },
        };
        if nal.len() <= rbsp_off {
            continue;
        }
        let rbsp = strip_emulation_prevention(&nal[rbsp_off..]);
        if let Some(t) = misp_time_in_sei_rbsp(&rbsp) {
            return Some(t);
        }
    }
    None
}

/// Walk one SEI RBSP's messages for a MISP time payload. Each message is
/// `payloadType` then `payloadSize` (both extended by leading `0xFF` run bytes)
/// then that many payload bytes.
fn misp_time_in_sei_rbsp(rbsp: &[u8]) -> Option<MispTime> {
    let mut i = 0usize;
    // Stop once only the rbsp_trailing_bits (a lone 0x80) remain.
    while i + 1 < rbsp.len() {
        let (payload_type, next) = read_ff_extended(rbsp, i)?;
        let (payload_size, next) = read_ff_extended(rbsp, next)?;
        i = next;
        let end = match i.checked_add(payload_size) {
            Some(e) if e <= rbsp.len() => e,
            _ => return None,
        };
        if payload_type == SEI_USER_DATA_UNREGISTERED {
            if let Some(t) = parse_misp_payload(&rbsp[i..end]) {
                return Some(t);
            }
        }
        i = end;
    }
    None
}

/// Parse a `user_data_unregistered` payload as a MISP time, or `None` when it is
/// some other vendor's user data or is not well formed.
fn parse_misp_payload(payload: &[u8]) -> Option<MispTime> {
    if payload.len() < MISP_PAYLOAD_LEN {
        return None;
    }
    let uuid = &payload[..16];
    let unit = if uuid == MISP_MICROSECTIME_H264 || uuid == MISP_MICROSECTIME_H265 {
        MispTimeUnit::Microseconds
    } else if uuid == MISP_NANOSECTIME_H265 {
        MispTimeUnit::Nanoseconds
    } else {
        return None;
    };
    let b = &payload[16..MISP_PAYLOAD_LEN];
    // The separators are mandatory; without them this is not an ST 0604 body and
    // reading a time out of it would invent one.
    if b[3] != 0xFF || b[6] != 0xFF || b[9] != 0xFF {
        return None;
    }
    Some(MispTime {
        status: b[0],
        value: u64::from_be_bytes([b[1], b[2], b[4], b[5], b[7], b[8], b[10], b[11]]),
        unit,
    })
}

/// The compressed-video codecs whose SEI these elements read / write.
fn video_alternatives() -> CapsSet {
    CapsSet::from_alternatives(Vec::from([
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        },
        Caps::CompressedVideo {
            codec: VideoCodec::H265,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        },
    ]))
}

/// Stamp each access unit of a compressed H.264 / H.265 stream with a MISP
/// microsecond time SEI: compressed video in, the same stream out with one SEI
/// NAL added before the first VCL slice.
///
/// Frame timestamps are stream-relative, MISP time is absolute, so
/// `epoch-offset` supplies the UNIX-epoch microsecond that PTS 0 maps to.
#[derive(Debug)]
pub struct MispTimeInsert {
    /// Input codec, fixed at `configure_pipeline`; selects the SEI framing.
    codec: Option<VideoCodec>,
    caps: Option<Caps>,
    status: u8,
    epoch_offset_us: u64,
}

impl Default for MispTimeInsert {
    fn default() -> Self {
        Self::new()
    }
}

impl MispTimeInsert {
    pub fn new() -> Self {
        Self {
            codec: None,
            caps: None,
            status: MISP_STATUS_DEFAULT,
            epoch_offset_us: 0,
        }
    }

    /// Set the UNIX-epoch microsecond that PTS 0 maps to.
    pub fn with_epoch_offset_us(mut self, offset_us: u64) -> Self {
        self.epoch_offset_us = offset_us;
        self
    }

    /// Set the ST 0604 status byte written into each stamp.
    pub fn with_status(mut self, status: u8) -> Self {
        self.status = status;
        self
    }

    /// The absolute MISP time for a frame at `pts_ns`.
    fn time_for(&self, pts_ns: u64) -> u64 {
        (pts_ns / 1_000).saturating_add(self.epoch_offset_us)
    }
}

impl AsyncElement for MispTimeInsert {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::CompressedVideo {
                codec: VideoCodec::H264 | VideoCodec::H265,
                ..
            } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(video_alternatives())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::CompressedVideo {
                codec: codec @ (VideoCodec::H264 | VideoCodec::H265),
                ..
            } => {
                self.codec = Some(*codec);
                self.caps = Some(absolute_caps.clone());
                Ok(ConfigureOutcome::Accepted)
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "MISP time stamp inserter",
            "Codec/Parser/Video",
            "Writes a MISB ST 0604 MISP microsecond time SEI into each H.264 / H.265 access unit",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            let Some(codec) = self.codec else {
                return Err(G2gError::NotConfigured);
            };
            let PipelinePacket::DataFrame(frame) = packet else {
                return out.push(packet).await.map(|_| ());
            };
            let Some(au) = frame.domain.as_system_slice() else {
                // A non-system buffer carries no walkable bitstream; pass it through.
                return out.push(PipelinePacket::DataFrame(frame)).await.map(|_| ());
            };
            let sei = build_misp_time_sei(self.time_for(frame.timing.pts_ns), self.status, codec);
            let mut bytes = Vec::with_capacity(au.len() + sei.len());
            match vcl_start(au, codec) {
                Some(off) => {
                    bytes.extend_from_slice(&au[..off]);
                    bytes.extend_from_slice(&sei);
                    bytes.extend_from_slice(&au[off..]);
                }
                // No VCL slice (e.g. a parameter-set-only AU): leave it unchanged.
                None => bytes.extend_from_slice(au),
            }
            let new = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                frame.timing,
                frame.sequence,
            );
            out.push(PipelinePacket::DataFrame(new)).await.map(|_| ())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new(
                "epoch-offset",
                PropKind::Uint,
                "microseconds since the UNIX epoch that PTS 0 maps to",
            )
            .with_default("0"),
            PropertySpec::new("status", PropKind::Uint, "ST 0604 status byte").with_default("31"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "epoch-offset" => {
                self.epoch_offset_us = value.as_uint().ok_or(PropError::Type)?;
                Ok(())
            }
            "status" => {
                let v = value.as_uint().ok_or(PropError::Type)?;
                self.status = u8::try_from(v).map_err(|_| PropError::Value)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "epoch-offset" => Some(PropValue::Uint(self.epoch_offset_us)),
            "status" => Some(PropValue::Uint(self.status as u64)),
            _ => None,
        }
    }
}

impl PadTemplates for MispTimeInsert {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(video_alternatives()),
            PadTemplate::source(video_alternatives()),
        ])
    }
}

/// Mine the MISP time stamp out of each access unit of a compressed
/// H.264 / H.265 stream and emit it as a timed `Caps::Text{Utf8}` cue, the same
/// `ts=<microseconds>` field [`KlvDecode`](crate::klv::KlvDecode) renders, so the
/// video and telemetry branches read alike. A branch leaf like
/// [`CcExtract`](crate::ccextract::CcExtract): tee the parser output, one branch
/// to the decoder and the other here.
#[derive(Debug)]
pub struct MispTimeExtract {
    codec: Option<VideoCodec>,
    caps_emitted: bool,
    sequence: u64,
}

impl Default for MispTimeExtract {
    fn default() -> Self {
        Self::new()
    }
}

impl MispTimeExtract {
    pub fn new() -> Self {
        Self {
            codec: None,
            caps_emitted: false,
            sequence: 0,
        }
    }

    fn output_caps() -> Caps {
        Caps::Text {
            format: TextFormat::Utf8,
        }
    }
}

impl AsyncElement for MispTimeExtract {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::CompressedVideo {
                codec: VideoCodec::H264 | VideoCodec::H265,
                ..
            } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Decoder-style: a compressed H.264 / H.265 stream in, plain UTF-8 text out.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::CompressedVideo {
                codec: VideoCodec::H264 | VideoCodec::H265,
                ..
            } => CapsSet::one(Self::output_caps()),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::CompressedVideo {
                codec: codec @ (VideoCodec::H264 | VideoCodec::H265),
                ..
            } => {
                self.codec = Some(*codec);
                Ok(ConfigureOutcome::Accepted)
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "MISP time stamp extractor",
            "Codec/Parser/Video",
            "Extracts the MISB ST 0604 MISP time from H.264 / H.265 SEI into timed UTF-8 text",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            let Some(codec) = self.codec else {
                return Err(G2gError::NotConfigured);
            };
            let mut cues = Vec::new();
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    if let Some(au) = frame.domain.as_system_slice() {
                        if let Some(t) = extract_misp_time(au, codec) {
                            let start = frame.timing.pts_ns;
                            cues.push(Cue {
                                start_ns: start,
                                end_ns: start.saturating_add(frame.timing.duration_ns),
                                text: alloc::format!("ts={}", t.micros()),
                                settings: CueSettings::default(),
                            });
                        }
                    }
                }
                // The output caps are negotiated up front (DerivedOutput) and
                // announced at the first cue; an inbound video caps change carries
                // no timestamp effect.
                PipelinePacket::CapsChanged(_) | PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            crate::ccextract::push_cue_frames(out, cues, &mut self.caps_emitted, &mut self.sequence)
                .await
        })
    }
}

impl PadTemplates for MispTimeExtract {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(video_alternatives()),
            PadTemplate::source(CapsSet::one(Self::output_caps())),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use g2g_core::{FrameTiming, PushOutcome};

    /// The MISP SEI from a real STANAG 4609 capture, as it appears in GStreamer's
    /// `gst-plugins-bad/tests/check/elements/h264parse.c`
    /// (`test_parse_sei_userdefinedunregistered`). AVCC framed: a 4-byte length,
    /// the SEI NAL, then an IDR slice. The time decodes to 1444200632933395 us
    /// (2015-10-07 06:50:32.933395 UTC).
    fn gst_reference_au() -> Vec<u8> {
        vec![
            0x00, 0x00, 0x00, 0x20, 0x06, 0x05, 0x1c, 0x4d, 0x49, 0x53, 0x50, 0x6d, 0x69, 0x63,
            0x72, 0x6f, 0x73, 0x65, 0x63, 0x74, 0x69, 0x6d, 0x65, 0x1f, 0x00, 0x05, 0xff, 0x21,
            0x7e, 0xff, 0x29, 0xb5, 0xff, 0xdc, 0x13, 0x80, //
            0x00, 0x00, 0x00, 0x14, 0x65, 0x88, 0x84, 0x00, 0x10, 0xff, 0xfe, 0xf6, 0xf0, 0xfe,
            0x05, 0x36, 0x56, 0x04, 0x50, 0x96, 0x7b, 0x3f, 0x53, 0xe1,
        ]
    }

    const GST_REFERENCE_US: u64 = 1_444_200_632_933_395;

    #[test]
    fn parses_the_gstreamer_reference_capture() {
        let t = extract_misp_time(&gst_reference_au(), VideoCodec::H264).expect("MISP SEI found");
        assert_eq!(t.value, GST_REFERENCE_US);
        assert_eq!(t.unit, MispTimeUnit::Microseconds);
        assert_eq!(t.status, 0x1F);
        assert_eq!(t.nanos(), GST_REFERENCE_US * 1_000);
    }

    #[test]
    fn builds_the_gstreamer_reference_bytes() {
        // The builder reproduces the captured SEI byte for byte (start code in
        // place of the AVCC length prefix).
        let nal = build_misp_time_sei(GST_REFERENCE_US, 0x1F, VideoCodec::H264);
        let expected: Vec<u8> = [0x00, 0x00, 0x00, 0x01]
            .iter()
            .copied()
            .chain(gst_reference_au()[4..36].iter().copied())
            .collect();
        assert_eq!(nal, expected);
    }

    #[test]
    fn round_trips_both_codecs() {
        for codec in [VideoCodec::H264, VideoCodec::H265] {
            for micros in [0u64, 1, GST_REFERENCE_US, u64::MAX] {
                let nal = build_misp_time_sei(micros, MISP_STATUS_DEFAULT, codec);
                let t = extract_misp_time(&nal, codec).expect("round trips");
                assert_eq!(t.value, micros, "{codec:?}");
                assert_eq!(t.status, MISP_STATUS_DEFAULT);
            }
        }
    }

    #[test]
    fn codecs_use_their_own_uuid() {
        let h264 = build_misp_time_sei(1, MISP_STATUS_DEFAULT, VideoCodec::H264);
        let h265 = build_misp_time_sei(1, MISP_STATUS_DEFAULT, VideoCodec::H265);
        assert!(h264.windows(16).any(|w| w == MISP_MICROSECTIME_H264));
        assert!(h265.windows(16).any(|w| w == MISP_MICROSECTIME_H265));
    }

    #[test]
    fn the_ff_separators_make_escaping_unnecessary() {
        // A zero time is the worst case for start code emulation. The ST 0604
        // 0xFF bytes cap the zero run at two, so the RBSP needs no 0x03 escape
        // and the SEI keeps its fixed byte offsets.
        let nal = build_misp_time_sei(0, MISP_STATUS_DEFAULT, VideoCodec::H264);
        assert!(!nal[4..].windows(3).any(|w| w == [0x00, 0x00, 0x03]));
        assert_eq!(
            extract_misp_time(&nal, VideoCodec::H264).unwrap().value,
            0,
            "and it still round trips"
        );
    }

    #[test]
    fn a_zero_status_forces_escaping_and_still_round_trips() {
        // A status of 0 does put three zero bytes in a row, so the RBSP escape
        // fires; the parser strips it before reading the fixed offsets.
        let nal = build_misp_time_sei(0, 0x00, VideoCodec::H264);
        assert!(nal[4..].windows(3).any(|w| w == [0x00, 0x00, 0x03]));
        let t = extract_misp_time(&nal, VideoCodec::H264).expect("escaped SEI still parses");
        assert_eq!(t.value, 0);
        assert_eq!(t.status, 0);
    }

    #[test]
    fn truncated_or_malformed_sei_yields_no_time() {
        let full = build_misp_time_sei(GST_REFERENCE_US, MISP_STATUS_DEFAULT, VideoCodec::H264);
        // Any truncation that cuts into the payload yields no time, and no
        // truncation panics (the last byte is only the rbsp_trailing_bits, so
        // dropping it still leaves a complete payload).
        for n in 0..full.len() - 1 {
            assert!(
                extract_misp_time(&full[..n], VideoCodec::H264).is_none(),
                "truncated to {n} bytes"
            );
        }
        // A payload with the right UUID but a corrupted separator is not ST 0604.
        let mut bad = full.clone();
        let sep = bad.iter().position(|&b| b == 0xFF).expect("separator");
        bad[sep] = 0x00;
        assert!(extract_misp_time(&bad, VideoCodec::H264).is_none());
        // A payloadSize larger than the buffer is rejected, not trusted.
        let mut oversize = full.clone();
        oversize[6] = 0xFE;
        assert!(extract_misp_time(&oversize, VideoCodec::H264).is_none());
    }

    #[test]
    fn other_user_data_is_ignored() {
        // A user_data_unregistered SEI with a different UUID yields nothing.
        let mut nal = vec![0x00, 0x00, 0x00, 0x01, 0x06, 0x05, 0x1c];
        nal.extend_from_slice(b"NOTMISPTIMESTAMP");
        nal.extend_from_slice(&[0x1f, 0, 0, 0xff, 0, 0, 0xff, 0, 0, 0xff, 0, 0, 0x80]);
        assert!(extract_misp_time(&nal, VideoCodec::H264).is_none());
    }

    #[test]
    fn nanosecond_uuid_is_read_as_nanoseconds() {
        // Hand-build the H.265 nanosecond form: same body, different UUID.
        let mut payload = Vec::from(MISP_NANOSECTIME_H265);
        payload.extend_from_slice(&[
            0x1f, 0x00, 0x00, 0xff, 0x00, 0x00, 0xff, 0x00, 0x0f, 0xff, 0x42, 0x40,
        ]);
        let mut nal = vec![0x00, 0x00, 0x00, 0x01, 0x4e, 0x01, 0x05, 0x1c];
        nal.extend_from_slice(&payload);
        nal.push(0x80);
        let t = extract_misp_time(&nal, VideoCodec::H265).expect("nanosecond SEI parses");
        assert_eq!(t.unit, MispTimeUnit::Nanoseconds);
        assert_eq!(t.value, 1_000_000);
        assert_eq!(t.micros(), 1_000);
    }

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

    fn caps(codec: VideoCodec) -> Caps {
        Caps::CompressedVideo {
            codec,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    fn data_frame(au: Vec<u8>, pts: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
            FrameTiming {
                pts_ns: pts,
                ..Default::default()
            },
            0,
        ))
    }

    fn first_au(sink: &RecordingSink) -> Vec<u8> {
        sink.packets
            .iter()
            .find_map(|p| match p {
                PipelinePacket::DataFrame(f) => f.domain.as_system_slice().map(Vec::from),
                _ => None,
            })
            .expect("an access unit")
    }

    #[tokio::test]
    async fn insert_stamps_pts_plus_epoch_offset() {
        let mut el = MispTimeInsert::new().with_epoch_offset_us(1_700_000_000_000_000);
        el.configure_pipeline(&caps(VideoCodec::H264)).unwrap();
        let mut sink = RecordingSink::default();
        let au = vec![0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x00];
        el.process(data_frame(au, 2_500_000), &mut sink)
            .await
            .unwrap();
        let t = extract_misp_time(&first_au(&sink), VideoCodec::H264).expect("stamped");
        assert_eq!(t.value, 1_700_000_000_000_000 + 2_500);
    }

    #[tokio::test]
    async fn insert_requires_configure() {
        let mut el = MispTimeInsert::new();
        let mut sink = RecordingSink::default();
        let au = vec![0x00, 0x00, 0x00, 0x01, 0x65, 0x88];
        assert!(el.process(data_frame(au, 0), &mut sink).await.is_err());
    }

    #[tokio::test]
    async fn extract_emits_a_ts_cue_per_stamped_frame() {
        let mut el = MispTimeExtract::new();
        el.configure_pipeline(&caps(VideoCodec::H264)).unwrap();
        let mut sink = RecordingSink::default();
        let mut au = build_misp_time_sei(GST_REFERENCE_US, MISP_STATUS_DEFAULT, VideoCodec::H264);
        au.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x65, 0x88]);
        el.process(data_frame(au, 7_000), &mut sink).await.unwrap();

        // The text caps are announced before the first cue frame.
        assert!(matches!(
            sink.packets.first(),
            Some(PipelinePacket::CapsChanged(Caps::Text {
                format: TextFormat::Utf8
            }))
        ));
        let text = sink
            .packets
            .iter()
            .find_map(|p| match p {
                PipelinePacket::DataFrame(f) => f
                    .domain
                    .as_system_slice()
                    .map(|s| alloc::string::String::from_utf8_lossy(s).into_owned()),
                _ => None,
            })
            .expect("a cue");
        assert_eq!(text, alloc::format!("ts={GST_REFERENCE_US}"));
    }

    #[tokio::test]
    async fn extract_is_silent_on_unstamped_frames() {
        let mut el = MispTimeExtract::new();
        el.configure_pipeline(&caps(VideoCodec::H264)).unwrap();
        let mut sink = RecordingSink::default();
        let au = vec![0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x00];
        el.process(data_frame(au, 0), &mut sink).await.unwrap();
        assert!(sink.packets.is_empty());
    }

    #[test]
    fn insert_properties_round_trip_onto_the_fields_it_uses() {
        // Both halves parse_launch needs: declared in `properties()` and applied
        // by `set_property`, which the stamp then reflects.
        let mut el = MispTimeInsert::new();
        let names: Vec<&str> = el.properties().iter().map(|s| s.name).collect();
        assert!(names.contains(&"epoch-offset") && names.contains(&"status"));

        el.set_property("epoch-offset", PropValue::Uint(1_000_000))
            .unwrap();
        el.set_property("status", PropValue::Uint(0x9F)).unwrap();
        assert_eq!(
            el.get_property("epoch-offset"),
            Some(PropValue::Uint(1_000_000))
        );
        assert_eq!(el.get_property("status"), Some(PropValue::Uint(0x9F)));

        let sei = build_misp_time_sei(el.time_for(500_000), el.status, VideoCodec::H264);
        let t = extract_misp_time(&sei, VideoCodec::H264).unwrap();
        assert_eq!(t.value, 1_000_500);
        assert_eq!(t.status, 0x9F);

        // A status outside a byte is rejected rather than truncated.
        assert!(el.set_property("status", PropValue::Uint(256)).is_err());
    }

    #[test]
    fn negotiates_video_in_text_out() {
        let el = MispTimeExtract::new();
        assert!(el.intercept_caps(&caps(VideoCodec::H265)).is_ok());
        assert!(el
            .intercept_caps(&Caps::Text {
                format: TextFormat::Utf8
            })
            .is_err());
        let derived = match el.caps_constraint_as_transform() {
            CapsConstraint::DerivedOutput(f) => f(&caps(VideoCodec::H264)),
            _ => panic!("expected DerivedOutput"),
        };
        assert_eq!(derived.alternatives(), &[MispTimeExtract::output_caps()]);
    }
}
