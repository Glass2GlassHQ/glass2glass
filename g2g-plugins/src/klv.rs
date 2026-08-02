//! MISB ST 0601 UAS Datalink Local Set (M800): the KLV (SMPTE ST 336) metadata
//! a STANAG 4609 stream carries alongside its video, encoding platform / sensor
//! telemetry (position, attitude, field of view, frame center).
//!
//! Pure `no_std + alloc` codec plus the [`KlvDecode`] element (`klvdecode`):
//! `Caps::Klv` packets in, timed `Caps::Text{Utf8}` telemetry lines out, so a
//! demuxed drone stream overlays or logs its telemetry
//! (`tsdemux stream=klv ! klvdecode ! textoverlay`). The encode direction is the
//! [`UasDatalink::encode`] API (an app builds a local set and pushes it through
//! `appsrc` / `tsmuxn` with `Caps::Klv`).
//!
//! Every count / length here is attacker-controlled bitstream data: BER lengths
//! and BER-OID tags are bounds-checked, arithmetic saturates, and a malformed
//! packet fails the parse (checksum included) rather than panicking.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError, MemoryDomain,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec, TextFormat,
};

/// The 16-byte universal label of the ST 0601 UAS Datalink Local Set.
pub const UAS_LOCAL_SET_KEY: [u8; 16] = [
    0x06, 0x0E, 0x2B, 0x34, 0x02, 0x0B, 0x01, 0x01, 0x0E, 0x01, 0x03, 0x01, 0x01, 0x00, 0x00, 0x00,
];

/// Read a BER length at `buf[pos]`: short form (one byte < 0x80) or long form
/// (0x81..=0x84, that many big-endian bytes). Returns `(length, bytes_consumed)`.
/// The indefinite form (0x80) and lengths over 4 bytes are rejected, as is any
/// truncation, so a bogus length fails the parse instead of over-reading.
fn ber_length(buf: &[u8], pos: usize) -> Option<(usize, usize)> {
    let first = *buf.get(pos)?;
    if first < 0x80 {
        return Some((first as usize, 1));
    }
    let n = (first & 0x7F) as usize;
    if n == 0 || n > 4 {
        return None;
    }
    let mut len: usize = 0;
    for &b in buf.get(pos + 1..pos + 1 + n)? {
        len = len.checked_mul(256)?.checked_add(b as usize)?;
    }
    Some((len, 1 + n))
}

/// Write a BER length: short form when it fits, else the minimal long form.
fn push_ber_length(out: &mut Vec<u8>, len: usize) {
    if len < 0x80 {
        out.push(len as u8);
    } else if len <= 0xFF {
        out.push(0x81);
        out.push(len as u8);
    } else {
        out.push(0x82);
        out.push((len >> 8) as u8);
        out.push(len as u8);
    }
}

/// Read a BER-OID tag at `buf[pos]` (7 bits per byte, high bit continues).
/// Returns `(tag, bytes_consumed)`; bounded to 4 bytes (ST 0601 tags are small).
fn ber_oid_tag(buf: &[u8], pos: usize) -> Option<(u32, usize)> {
    let mut tag: u32 = 0;
    for i in 0..4 {
        let b = *buf.get(pos + i)?;
        tag = (tag << 7) | (b & 0x7F) as u32;
        if b & 0x80 == 0 {
            return Some((tag, i + 1));
        }
    }
    None
}

/// The ST 0601 running 16-bit checksum over `buf`: byte `i` is added into the
/// high half when `i` is even, the low half when odd (the standard's `bcc_16`).
fn checksum_16(buf: &[u8]) -> u16 {
    let mut bcc: u16 = 0;
    for (i, &b) in buf.iter().enumerate() {
        bcc = bcc.wrapping_add((b as u16) << (8 * ((i + 1) % 2)));
    }
    bcc
}

/// Split a buffer (one PES payload) into the KLV packets it carries: each starts
/// with the 4-byte SMPTE ST 336 prefix, a 16-byte key, then a BER length. Stops
/// at the first byte run that is not a KLV packet, so a truncated tail is
/// dropped rather than over-read.
pub fn split_klv_packets(mut buf: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    while buf.len() >= 17 && buf[..4] == [0x06, 0x0E, 0x2B, 0x34] {
        let Some((len, len_bytes)) = ber_length(buf, 16) else {
            break;
        };
        let Some(total) = len.checked_add(16 + len_bytes) else {
            break;
        };
        let Some(pkt) = buf.get(..total) else {
            break;
        };
        out.push(pkt);
        buf = &buf[total..];
    }
    out
}

/// Round-to-nearest for the scaled-integer encodes (`f64::round` is std-only).
fn round(x: f64) -> f64 {
    if x >= 0.0 {
        (x + 0.5) as i64 as f64
    } else {
        (x - 0.5) as i64 as f64
    }
}

/// The decoded core of an ST 0601 UAS Datalink Local Set: the platform / sensor
/// telemetry tags a ground station reads first. Every field is optional (a local
/// set carries only what changed); unknown tags are skipped on parse. Angles are
/// degrees, altitudes meters (MSL), the timestamp microseconds since the Unix
/// epoch (tag 2, mandated first in the set).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct UasDatalink {
    /// Tag 2: precision time stamp, microseconds since the Unix epoch.
    pub timestamp_us: Option<u64>,
    /// Tag 5: platform heading angle, 0..360 degrees.
    pub heading_deg: Option<f64>,
    /// Tag 6: platform pitch angle, +-20 degrees.
    pub pitch_deg: Option<f64>,
    /// Tag 7: platform roll angle, +-50 degrees.
    pub roll_deg: Option<f64>,
    /// Tag 13: sensor latitude, +-90 degrees.
    pub sensor_lat_deg: Option<f64>,
    /// Tag 14: sensor longitude, +-180 degrees.
    pub sensor_lon_deg: Option<f64>,
    /// Tag 15: sensor true altitude, -900..19000 m.
    pub sensor_alt_m: Option<f64>,
    /// Tag 16: sensor horizontal field of view, 0..180 degrees.
    pub hfov_deg: Option<f64>,
    /// Tag 17: sensor vertical field of view, 0..180 degrees.
    pub vfov_deg: Option<f64>,
    /// Tag 18: sensor relative azimuth (to platform nose), 0..360 degrees.
    pub rel_azimuth_deg: Option<f64>,
    /// Tag 19: sensor relative elevation, +-180 degrees.
    pub rel_elevation_deg: Option<f64>,
    /// Tag 20: sensor relative roll, 0..360 degrees.
    pub rel_roll_deg: Option<f64>,
    /// Tag 23: frame center latitude, +-90 degrees.
    pub frame_center_lat_deg: Option<f64>,
    /// Tag 24: frame center longitude, +-180 degrees.
    pub frame_center_lon_deg: Option<f64>,
    /// Tag 25: frame center elevation, -900..19000 m.
    pub frame_center_alt_m: Option<f64>,
    /// Tag 65: UAS Datalink LS version number.
    pub version: Option<u8>,
}

// Fixed-point scale factors from ST 0601: a mapped integer spans the tag's
// documented physical range.
const LAT_SCALE: f64 = 90.0 / (i32::MAX as f64);
const LON_SCALE: f64 = 180.0 / (i32::MAX as f64);
const HEADING_SCALE: f64 = 360.0 / (u16::MAX as f64);
const PITCH_SCALE: f64 = 20.0 / (i16::MAX as f64);
const ROLL_SCALE: f64 = 50.0 / (i16::MAX as f64);
const FOV_SCALE: f64 = 180.0 / (u16::MAX as f64);
const REL_AZ_SCALE: f64 = 360.0 / (u32::MAX as f64);
const REL_EL_SCALE: f64 = 180.0 / (i32::MAX as f64);
const ALT_SCALE: f64 = 19900.0 / (u16::MAX as f64);
const ALT_OFFSET: f64 = -900.0;

fn u16_at(v: &[u8]) -> Option<u16> {
    Some(u16::from_be_bytes(v.try_into().ok()?))
}
fn u32_at(v: &[u8]) -> Option<u32> {
    Some(u32::from_be_bytes(v.try_into().ok()?))
}
/// A signed 16-bit tag value; `i16::MIN` is the ST 0601 out-of-range indicator.
fn i16_valid(v: &[u8]) -> Option<i16> {
    Some(i16::from_be_bytes(v.try_into().ok()?)).filter(|&x| x != i16::MIN)
}
/// A signed 32-bit tag value; `i32::MIN` is the ST 0601 error indicator.
fn i32_valid(v: &[u8]) -> Option<i32> {
    Some(i32::from_be_bytes(v.try_into().ok()?)).filter(|&x| x != i32::MIN)
}

impl UasDatalink {
    /// Parse one complete KLV packet (16-byte UAS LS key, BER length, TLV body).
    /// `None` for a wrong key, a malformed / truncated structure, or a checksum
    /// (tag 1) that is absent or does not match: a corrupted set is rejected
    /// whole rather than half-read. Unknown tags and wrong-sized values are
    /// skipped (the set stays forward-compatible).
    pub fn parse(packet: &[u8]) -> Option<Self> {
        Self::parse_inner(packet, true)
    }

    /// [`parse`](Self::parse) without requiring the checksum to be present or
    /// match. Structure is still fully bounds-checked. For streams whose
    /// encoder writes a wrong checksum: even the published MISMMS reference
    /// packet declares 0xAA43 where the ST 0601 sum is 0x3E1E (klvdata computes
    /// the same), so field tooling tolerates this.
    pub fn parse_lenient(packet: &[u8]) -> Option<Self> {
        Self::parse_inner(packet, false)
    }

    fn parse_inner(packet: &[u8], require_checksum: bool) -> Option<Self> {
        if packet.get(..16)? != UAS_LOCAL_SET_KEY {
            return None;
        }
        let (body_len, len_bytes) = ber_length(packet, 16)?;
        let body_start = 16 + len_bytes;
        let body = packet.get(body_start..body_start.checked_add(body_len)?)?;

        let mut ls = Self::default();
        let mut checksum_ok = false;
        let mut pos = 0;
        while pos < body.len() {
            let (tag, tag_bytes) = ber_oid_tag(body, pos)?;
            let (len, l_bytes) = ber_length(body, pos + tag_bytes)?;
            let v_start = pos + tag_bytes + l_bytes;
            let value = body.get(v_start..v_start.checked_add(len)?)?;
            match tag {
                // The checksum covers the whole packet from the UL key up to and
                // including this tag + length (everything but the 2-byte value).
                1 => {
                    let covered = &packet[..body_start + v_start];
                    checksum_ok = u16_at(value) == Some(checksum_16(covered));
                }
                2 => {
                    if let Ok(v) = value.try_into() {
                        ls.timestamp_us = Some(u64::from_be_bytes(v));
                    }
                }
                5 => ls.heading_deg = u16_at(value).map(|v| v as f64 * HEADING_SCALE),
                6 => ls.pitch_deg = i16_valid(value).map(|v| v as f64 * PITCH_SCALE),
                7 => ls.roll_deg = i16_valid(value).map(|v| v as f64 * ROLL_SCALE),
                13 => ls.sensor_lat_deg = i32_valid(value).map(|v| v as f64 * LAT_SCALE),
                14 => ls.sensor_lon_deg = i32_valid(value).map(|v| v as f64 * LON_SCALE),
                15 => ls.sensor_alt_m = u16_at(value).map(|v| v as f64 * ALT_SCALE + ALT_OFFSET),
                16 => ls.hfov_deg = u16_at(value).map(|v| v as f64 * FOV_SCALE),
                17 => ls.vfov_deg = u16_at(value).map(|v| v as f64 * FOV_SCALE),
                18 => ls.rel_azimuth_deg = u32_at(value).map(|v| v as f64 * REL_AZ_SCALE),
                19 => ls.rel_elevation_deg = i32_valid(value).map(|v| v as f64 * REL_EL_SCALE),
                20 => ls.rel_roll_deg = u32_at(value).map(|v| v as f64 * REL_AZ_SCALE),
                23 => ls.frame_center_lat_deg = i32_valid(value).map(|v| v as f64 * LAT_SCALE),
                24 => ls.frame_center_lon_deg = i32_valid(value).map(|v| v as f64 * LON_SCALE),
                25 => {
                    ls.frame_center_alt_m = u16_at(value).map(|v| v as f64 * ALT_SCALE + ALT_OFFSET)
                }
                65 => ls.version = value.first().copied(),
                _ => {}
            }
            pos = v_start + len;
        }
        (checksum_ok || !require_checksum).then_some(ls)
    }

    /// Encode as one KLV packet: the UAS LS key, a BER length, the present tags
    /// (timestamp first, as ST 0601 mandates), and the trailing checksum (tag 1).
    pub fn encode(&self) -> Vec<u8> {
        let mut body = Vec::new();
        let mut put = |tag: u8, value: &[u8]| {
            body.push(tag);
            push_ber_length(&mut body, value.len());
            body.extend_from_slice(value);
        };
        if let Some(v) = self.timestamp_us {
            put(2, &v.to_be_bytes());
        }
        let scaled_u16 = |x: f64, scale: f64| (round(x / scale) as u16).to_be_bytes();
        let scaled_i16 = |x: f64, scale: f64| (round(x / scale) as i16).to_be_bytes();
        let scaled_u32 = |x: f64, scale: f64| (round(x / scale) as u32).to_be_bytes();
        let scaled_i32 = |x: f64, scale: f64| (round(x / scale) as i32).to_be_bytes();
        if let Some(v) = self.heading_deg {
            put(5, &scaled_u16(v, HEADING_SCALE));
        }
        if let Some(v) = self.pitch_deg {
            put(6, &scaled_i16(v, PITCH_SCALE));
        }
        if let Some(v) = self.roll_deg {
            put(7, &scaled_i16(v, ROLL_SCALE));
        }
        if let Some(v) = self.sensor_lat_deg {
            put(13, &scaled_i32(v, LAT_SCALE));
        }
        if let Some(v) = self.sensor_lon_deg {
            put(14, &scaled_i32(v, LON_SCALE));
        }
        if let Some(v) = self.sensor_alt_m {
            put(15, &scaled_u16(v - ALT_OFFSET, ALT_SCALE));
        }
        if let Some(v) = self.hfov_deg {
            put(16, &scaled_u16(v, FOV_SCALE));
        }
        if let Some(v) = self.vfov_deg {
            put(17, &scaled_u16(v, FOV_SCALE));
        }
        if let Some(v) = self.rel_azimuth_deg {
            put(18, &scaled_u32(v, REL_AZ_SCALE));
        }
        if let Some(v) = self.rel_elevation_deg {
            put(19, &scaled_i32(v, REL_EL_SCALE));
        }
        if let Some(v) = self.rel_roll_deg {
            put(20, &scaled_u32(v, REL_AZ_SCALE));
        }
        if let Some(v) = self.frame_center_lat_deg {
            put(23, &scaled_i32(v, LAT_SCALE));
        }
        if let Some(v) = self.frame_center_lon_deg {
            put(24, &scaled_i32(v, LON_SCALE));
        }
        if let Some(v) = self.frame_center_alt_m {
            put(25, &scaled_u16(v - ALT_OFFSET, ALT_SCALE));
        }
        if let Some(v) = self.version {
            put(65, &[v]);
        }
        // Trailing checksum: tag + length enter the sum, then the value is the
        // sum over everything before it (key + BER length included).
        body.push(1);
        body.push(2);
        let mut packet = Vec::with_capacity(16 + 3 + body.len() + 2);
        packet.extend_from_slice(&UAS_LOCAL_SET_KEY);
        push_ber_length(&mut packet, body.len() + 2);
        packet.extend_from_slice(&body);
        let bcc = checksum_16(&packet);
        packet.extend_from_slice(&bcc.to_be_bytes());
        packet
    }

    /// The telemetry as one `key=value` line (only the present fields), the
    /// text [`KlvDecode`] emits: readable on an overlay, splittable by a log
    /// consumer.
    pub fn to_line(&self) -> String {
        let mut s = String::new();
        let mut put = |part: String| {
            if !s.is_empty() {
                s.push(' ');
            }
            s.push_str(&part);
        };
        if let Some(v) = self.timestamp_us {
            put(alloc::format!("ts={v}"));
        }
        if let Some(v) = self.sensor_lat_deg {
            put(alloc::format!("lat={v:.6}"));
        }
        if let Some(v) = self.sensor_lon_deg {
            put(alloc::format!("lon={v:.6}"));
        }
        if let Some(v) = self.sensor_alt_m {
            put(alloc::format!("alt={v:.1}"));
        }
        if let Some(v) = self.heading_deg {
            put(alloc::format!("heading={v:.1}"));
        }
        if let Some(v) = self.pitch_deg {
            put(alloc::format!("pitch={v:.1}"));
        }
        if let Some(v) = self.roll_deg {
            put(alloc::format!("roll={v:.1}"));
        }
        if let Some(v) = self.hfov_deg {
            put(alloc::format!("hfov={v:.1}"));
        }
        if let Some(v) = self.vfov_deg {
            put(alloc::format!("vfov={v:.1}"));
        }
        if let Some(v) = self.rel_azimuth_deg {
            put(alloc::format!("az={v:.1}"));
        }
        if let Some(v) = self.rel_elevation_deg {
            put(alloc::format!("el={v:.1}"));
        }
        if let Some(v) = self.rel_roll_deg {
            put(alloc::format!("rel_roll={v:.1}"));
        }
        if let Some(v) = self.frame_center_lat_deg {
            put(alloc::format!("fc_lat={v:.6}"));
        }
        if let Some(v) = self.frame_center_lon_deg {
            put(alloc::format!("fc_lon={v:.6}"));
        }
        if let Some(v) = self.frame_center_alt_m {
            put(alloc::format!("fc_alt={v:.1}"));
        }
        s
    }
}

/// KLV telemetry decoder element (`klvdecode`): `Caps::Klv` frames in (each one
/// or more ST 336 packets, as a TS demux emits them), one timed
/// `Caps::Text{Utf8}` line per parsed ST 0601 local set out. A packet that is
/// not a valid ST 0601 set (wrong UL, bad checksum) is dropped, not forwarded;
/// `verify-checksum=false` tolerates a wrong checksum (encoders get it wrong,
/// see [`UasDatalink::parse_lenient`]).
#[derive(Debug)]
pub struct KlvDecode {
    configured: bool,
    emitted: u64,
    /// Whether a set with a missing / wrong checksum is dropped (the default).
    verify_checksum: bool,
}

impl Default for KlvDecode {
    fn default() -> Self {
        Self::new()
    }
}

impl KlvDecode {
    pub fn new() -> Self {
        Self {
            configured: false,
            emitted: 0,
            verify_checksum: true,
        }
    }

    /// Tolerate a missing / wrong checksum (default requires it to match).
    pub fn with_verify_checksum(mut self, verify: bool) -> Self {
        self.verify_checksum = verify;
        self
    }

    /// Count of text lines emitted.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn output_caps() -> Caps {
        Caps::Text {
            format: TextFormat::Utf8,
        }
    }
}

impl AsyncElement for KlvDecode {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Caps::Klv)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Klv => CapsSet::one(Self::output_caps()),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(absolute_caps, Caps::Klv) {
            return Err(G2gError::CapsMismatch);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
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
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    for pkt in split_klv_packets(slice) {
                        let parsed = if self.verify_checksum {
                            UasDatalink::parse(pkt)
                        } else {
                            UasDatalink::parse_lenient(pkt)
                        };
                        let Some(ls) = parsed else {
                            continue;
                        };
                        let line = ls.to_line();
                        let text = Frame::new(
                            MemoryDomain::System(SystemSlice::from_boxed(
                                line.into_bytes().into_boxed_slice(),
                            )),
                            frame.timing,
                            self.emitted,
                        );
                        self.emitted += 1;
                        out.push(PipelinePacket::DataFrame(text)).await?;
                    }
                }
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[PropertySpec::new(
            "verify-checksum",
            PropKind::Bool,
            "drop a local set whose checksum is missing or wrong",
        )
        .with_default("true")];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "verify-checksum" => {
                self.verify_checksum = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "verify-checksum" => Some(PropValue::Bool(self.verify_checksum)),
            _ => None,
        }
    }
}

impl PadTemplates for KlvDecode {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(Caps::Klv)),
            PadTemplate::source(CapsSet::one(Self::output_caps())),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> UasDatalink {
        UasDatalink {
            timestamp_us: Some(1_700_000_000_000_000),
            heading_deg: Some(87.3),
            pitch_deg: Some(1.2),
            roll_deg: Some(-3.5),
            sensor_lat_deg: Some(60.176822),
            sensor_lon_deg: Some(24.828835),
            sensor_alt_m: Some(145.2),
            hfov_deg: Some(54.9),
            vfov_deg: Some(31.2),
            rel_azimuth_deg: Some(46.3),
            rel_elevation_deg: Some(-4.9),
            rel_roll_deg: Some(358.2),
            frame_center_lat_deg: Some(60.18),
            frame_center_lon_deg: Some(24.84),
            frame_center_alt_m: Some(12.0),
            version: Some(19),
        }
    }

    /// Every field round-trips through encode + parse within its fixed-point
    /// quantization step.
    #[test]
    fn encode_parse_round_trip() {
        let ls = sample();
        let got = UasDatalink::parse(&ls.encode()).expect("valid packet");
        let close = |a: Option<f64>, b: Option<f64>, eps: f64| {
            let (a, b) = (a.unwrap(), b.unwrap());
            assert!((a - b).abs() <= eps, "{a} !~ {b}");
        };
        assert_eq!(got.timestamp_us, ls.timestamp_us);
        assert_eq!(got.version, ls.version);
        close(got.heading_deg, ls.heading_deg, HEADING_SCALE);
        close(got.pitch_deg, ls.pitch_deg, PITCH_SCALE);
        close(got.roll_deg, ls.roll_deg, ROLL_SCALE);
        close(got.sensor_lat_deg, ls.sensor_lat_deg, LAT_SCALE);
        close(got.sensor_lon_deg, ls.sensor_lon_deg, LON_SCALE);
        close(got.sensor_alt_m, ls.sensor_alt_m, ALT_SCALE);
        close(got.hfov_deg, ls.hfov_deg, FOV_SCALE);
        close(got.vfov_deg, ls.vfov_deg, FOV_SCALE);
        close(got.rel_azimuth_deg, ls.rel_azimuth_deg, REL_AZ_SCALE);
        close(got.rel_elevation_deg, ls.rel_elevation_deg, REL_EL_SCALE);
        close(got.rel_roll_deg, ls.rel_roll_deg, REL_AZ_SCALE);
        close(got.frame_center_lat_deg, ls.frame_center_lat_deg, LAT_SCALE);
        close(got.frame_center_lon_deg, ls.frame_center_lon_deg, LON_SCALE);
        close(got.frame_center_alt_m, ls.frame_center_alt_m, ALT_SCALE);
    }

    /// A flipped byte fails the checksum, so the whole set is rejected.
    #[test]
    fn corruption_fails_checksum() {
        let mut pkt = sample().encode();
        let mid = pkt.len() / 2;
        pkt[mid] ^= 0x01;
        assert_eq!(UasDatalink::parse(&pkt), None);
    }

    /// Malformed structure never panics: truncations, a bogus BER length, an
    /// over-long BER-OID tag, and a wrong key all fail cleanly.
    #[test]
    fn malformed_fails_cleanly() {
        let good = sample().encode();
        for cut in 0..good.len() {
            let _ = UasDatalink::parse(&good[..cut]);
        }
        let mut wrong_key = good.clone();
        wrong_key[15] ^= 0xFF;
        assert_eq!(UasDatalink::parse(&wrong_key), None);
        // A long-form BER length claiming more than the buffer holds.
        let mut huge = UAS_LOCAL_SET_KEY.to_vec();
        huge.extend_from_slice(&[0x84, 0xFF, 0xFF, 0xFF, 0xFF, 0x01]);
        assert_eq!(UasDatalink::parse(&huge), None);
        // An indefinite (0x80) BER length is rejected.
        let mut indefinite = UAS_LOCAL_SET_KEY.to_vec();
        indefinite.extend_from_slice(&[0x80, 0x01, 0x02]);
        assert_eq!(UasDatalink::parse(&indefinite), None);
    }

    /// Two packets in one buffer split cleanly; trailing junk is dropped.
    #[test]
    fn splits_concatenated_packets() {
        let a = sample().encode();
        let b = UasDatalink {
            timestamp_us: Some(1),
            ..Default::default()
        }
        .encode();
        let mut buf = a.clone();
        buf.extend_from_slice(&b);
        buf.extend_from_slice(&[0xDE, 0xAD]);
        let pkts = split_klv_packets(&buf);
        assert_eq!(pkts, alloc::vec![&a[..], &b[..]]);
    }

    /// The out-of-range indicators (i16::MIN / i32::MIN) decode to `None`, not
    /// a bogus angle.
    #[test]
    fn out_of_range_indicators_are_none() {
        // Hand-build a set with pitch = i16::MIN and lat = i32::MIN.
        let mut body = alloc::vec![
            6, 2, 0x80, 0x00, // pitch out of range
            13, 4, 0x80, 0x00, 0x00, 0x00, // lat error indicator
            1, 2, // checksum tag + len
        ];
        let mut pkt = UAS_LOCAL_SET_KEY.to_vec();
        push_ber_length(&mut pkt, body.len() + 2);
        pkt.append(&mut body);
        let bcc = checksum_16(&pkt);
        pkt.extend_from_slice(&bcc.to_be_bytes());
        let ls = UasDatalink::parse(&pkt).expect("valid structure");
        assert_eq!(ls.pitch_deg, None);
        assert_eq!(ls.sensor_lat_deg, None);
    }
}
