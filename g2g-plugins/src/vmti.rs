//! MISB ST 0903 VMTI (Video Moving Target Indicator) Local Set (M808): the
//! moving-target reports a detector puts alongside the video, either nested in
//! ST 0601 tag 74 (the usual STANAG 4609 case, where the enclosing set supplies
//! the platform telemetry and frame center) or as a KLV packet of its own.
//!
//! A [`VmtiLocalSet`] carries the frame geometry and a [`VTarget`] per reported
//! target: its id, where it sits in the frame (pixel number and / or row and
//! column, plus a bounding box as two pixel numbers), how sure the detector is,
//! and where it is on the ground. Pixel numbering is ST 0903's: 1 at the top
//! left, row major, `column + (row - 1) * frame_width`.
//!
//! Angles and heights use ST 1201 IMAPB, a linear map of a fixed range onto a
//! fixed-width integer, so the encodings are the current (ST 0903.4 and later)
//! ones; the pre-.4 scaled-integer forms are not read.
//!
//! With the `analytics` feature, [`vmti_from_analytics`] turns a frame's
//! `AnalyticsMeta` detections into a local set, so an in-pipeline detector emits
//! standards-compliant VMTI.
//!
//! Everything here is attacker-controlled bitstream data: lengths and counts are
//! bounds-checked, arithmetic is checked or saturating, and a malformed nested
//! set fails that field alone rather than panicking.

use alloc::string::String;
use alloc::vec::Vec;

use crate::klv::{
    ber_length, ber_oid_tag, checksum_16, push_ber_length, push_ber_oid, u8_one, utf8_string,
    MiisCoreId, BER_OID_MAX,
};

/// The 16-byte universal label of the ST 0903 VMTI Local Set, for the
/// standalone packet form ([`VmtiLocalSet::encode_klv`]).
pub const VMTI_LOCAL_SET_KEY: [u8; 16] = [
    0x06, 0x0E, 0x2B, 0x34, 0x02, 0x0B, 0x01, 0x01, 0x0E, 0x01, 0x03, 0x03, 0x06, 0x00, 0x00, 0x00,
];

/// The ST 0903 revision this codec writes: the float encodings below are the
/// IMAPB ones introduced in ST 0903.4, and a reader picks its parse rules from
/// this tag.
pub const ST0903_VERSION: u16 = 5;

/// The largest target id that survives the BER-OID round trip.
pub const MAX_TARGET_ID: u32 = BER_OID_MAX;

/// Largest frame dimension ST 0903 can express (the "V3" 3-byte counts).
const MAX_V3: u64 = 0xFF_FFFF;

/// Round toward negative infinity (`f64::floor` is std-only).
fn floor(x: f64) -> f64 {
    let truncated = x as i64 as f64;
    if x < truncated {
        truncated - 1.0
    } else {
        truncated
    }
}

/// `2^n` for a small signed exponent (`f64::powi` is std-only).
fn pow2(n: i32) -> f64 {
    let mut value = 1.0f64;
    for _ in 0..n.unsigned_abs() {
        value = if n > 0 { value * 2.0 } else { value / 2.0 };
    }
    value
}

/// `ceil(log2(range))` by doubling (`f64::log2` is std-only).
fn ceil_log2(range: f64) -> i32 {
    let mut power = 0;
    let mut value = 1.0f64;
    while value < range && power < 64 {
        value *= 2.0;
        power += 1;
    }
    while value / 2.0 >= range && power > -64 {
        value /= 2.0;
        power -= 1;
    }
    power
}

/// An ST 1201 IMAPB mapping: the range `[a, b]` packed into `len` bytes. The
/// constants are the standard's: `s_f` maps a value forward to the integer,
/// `s_r` back, and `z` shifts a range that straddles zero so that zero lands on
/// an integer.
#[derive(Debug, Clone, Copy)]
struct Imapb {
    a: f64,
    b: f64,
    s_f: f64,
    s_r: f64,
    z: f64,
    len: usize,
}

impl Imapb {
    fn new(a: f64, b: f64, len: usize) -> Self {
        let b_pow = ceil_log2(b - a);
        let d_pow = 8 * len as i32 - 1;
        let s_f = pow2(d_pow - b_pow);
        let z = if a < 0.0 && b > 0.0 {
            s_f * a - floor(s_f * a)
        } else {
            0.0
        };
        Self {
            a,
            b,
            s_f,
            s_r: pow2(b_pow - d_pow),
            z,
            len,
        }
    }

    /// Decode a value of exactly this mapping's width. ST 1201 flags special
    /// values (NaN, infinity, out of range) with the high bit of the first
    /// byte, which a normal value only reaches at the very top of the range;
    /// the out-of-range flags decode to the range ends and the rest to `None`,
    /// since this codec has no way to carry a NaN onward.
    fn decode(&self, v: &[u8]) -> Option<f64> {
        if v.len() != self.len {
            return None;
        }
        let d = v.iter().fold(0u64, |acc, &b| (acc << 8) | b as u64);
        if v[0] & 0x80 != 0 && d != 1u64 << (8 * self.len - 1) {
            return match v[0] {
                0xE0 => Some(self.a),
                0xE1 => Some(self.b),
                _ => None,
            };
        }
        Some(self.a + self.s_r * (d as f64 - self.z))
    }

    /// Encode a value, clamped into the mapping's range (and NaN to its
    /// minimum) so the result is always a plain mapped value.
    fn encode(&self, value: f64) -> Vec<u8> {
        let clamped = if value > self.b {
            self.b
        } else if value >= self.a {
            value
        } else {
            self.a
        };
        let d = floor(self.s_f * (clamped - self.a) + self.z) as u64;
        Vec::from(&d.to_be_bytes()[8 - self.len..])
    }
}

/// VMTI LS tags 11 / 12: horizontal and vertical field of view, degrees.
fn fov_imap() -> Imapb {
    Imapb::new(0.0, 180.0, 2)
}
/// VTarget tags 10 / 11: latitude / longitude offset from the frame center.
fn offset_imap() -> Imapb {
    Imapb::new(-19.2, 19.2, 3)
}
/// VTarget tag 12 and the location pack's third field: height above the WGS84
/// ellipsoid, meters.
fn hae_imap() -> Imapb {
    Imapb::new(-900.0, 19000.0, 2)
}
fn lat_imap() -> Imapb {
    Imapb::new(-90.0, 90.0, 4)
}
fn lon_imap() -> Imapb {
    Imapb::new(-180.0, 180.0, 4)
}

/// Read an ST 0903 variable-length big-endian unsigned integer, at most `max`
/// bytes (3 for the frame counts, 4 for a pixel row / column, 6 for a pixel
/// number). An empty or over-long value fails the tag.
fn var_uint(v: &[u8], max: usize) -> Option<u64> {
    if v.is_empty() || v.len() > max {
        return None;
    }
    Some(v.iter().fold(0u64, |acc, &b| (acc << 8) | b as u64))
}

/// Write an unsigned integer in the fewest bytes (at least one), the
/// bit-efficient form ST 0903 asks for.
fn var_uint_bytes(value: u64) -> Vec<u8> {
    let bytes = value.to_be_bytes();
    let skip = bytes.iter().take_while(|&&b| b == 0).count().min(7);
    Vec::from(&bytes[skip..])
}

/// VTarget tag 17: the target's absolute geodetic position.
///
/// `accuracy` keeps the pack's optional standard-deviation and correlation
/// groups (6 or 12 further bytes) verbatim. This codec does not decode them,
/// and dropping them would silently rewrite a pack it did not understand.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TargetLocation {
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub hae_m: f64,
    pub accuracy: Vec<u8>,
}

impl TargetLocation {
    fn parse(v: &[u8]) -> Option<Self> {
        // ST 0903 allows the coordinates alone, plus the sigma group, or plus
        // the sigma and rho groups.
        if !matches!(v.len(), 10 | 16 | 22) {
            return None;
        }
        Some(Self {
            lat_deg: lat_imap().decode(&v[..4])?,
            lon_deg: lon_imap().decode(&v[4..8])?,
            hae_m: hae_imap().decode(&v[8..10])?,
            accuracy: Vec::from(&v[10..]),
        })
    }

    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(10 + self.accuracy.len());
        out.extend_from_slice(&lat_imap().encode(self.lat_deg));
        out.extend_from_slice(&lon_imap().encode(self.lon_deg));
        out.extend_from_slice(&hae_imap().encode(self.hae_m));
        out.extend_from_slice(&self.accuracy);
        out
    }
}

/// One reported target: a VTarget pack from the VTarget series (VMTI LS tag
/// 101). The id leads the pack BER-OID encoded; everything after it is
/// optional, and a set carries only what its detector produced. Unknown tags
/// (the nested VMask / VObject / VTracker sets among them) are skipped.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VTarget {
    /// The target id, unique within the frame and stable across frames when the
    /// detector tracks. Bounded to [`MAX_TARGET_ID`] by its BER-OID encoding.
    pub id: u32,
    /// Tag 1: centroid as a pixel number.
    pub centroid_pixel: Option<u64>,
    /// Tag 2: bounding box top left corner as a pixel number.
    pub boundary_top_left_pixel: Option<u64>,
    /// Tag 3: bounding box bottom right corner as a pixel number.
    pub boundary_bottom_right_pixel: Option<u64>,
    /// Tag 4: priority or validity, 1 (highest) to 255.
    pub priority: Option<u8>,
    /// Tag 5: detector confidence, 0 to 100 percent.
    pub confidence_pct: Option<u8>,
    /// Tag 10: latitude offset from the frame center, +-19.2 degrees.
    pub location_offset_lat_deg: Option<f64>,
    /// Tag 11: longitude offset from the frame center, +-19.2 degrees.
    pub location_offset_lon_deg: Option<f64>,
    /// Tag 12: height above the WGS84 ellipsoid, -900..19000 m.
    pub hae_m: Option<f64>,
    /// Tag 17: absolute target location.
    pub location: Option<TargetLocation>,
    /// Tag 19: centroid row, 1-based.
    pub centroid_row: Option<u32>,
    /// Tag 20: centroid column, 1-based.
    pub centroid_col: Option<u32>,
}

impl VTarget {
    /// Parse one VTarget pack (the id, then TLV items). `None` for any
    /// malformed structure, which fails the whole series rather than reporting
    /// a target at a position we did not read.
    pub fn parse(pack: &[u8]) -> Option<Self> {
        let (id, id_bytes) = ber_oid_tag(pack, 0)?;
        let mut target = Self {
            id,
            ..Default::default()
        };
        let mut pos = id_bytes;
        while pos < pack.len() {
            let (tag, tag_bytes) = ber_oid_tag(pack, pos)?;
            let (len, l_bytes) = ber_length(pack, pos.checked_add(tag_bytes)?)?;
            let v_start = pos + tag_bytes + l_bytes;
            let value = pack.get(v_start..v_start.checked_add(len)?)?;
            match tag {
                1 => target.centroid_pixel = var_uint(value, 6),
                2 => target.boundary_top_left_pixel = var_uint(value, 6),
                3 => target.boundary_bottom_right_pixel = var_uint(value, 6),
                4 => target.priority = u8_one(value),
                5 => target.confidence_pct = u8_one(value),
                10 => target.location_offset_lat_deg = offset_imap().decode(value),
                11 => target.location_offset_lon_deg = offset_imap().decode(value),
                12 => target.hae_m = hae_imap().decode(value),
                17 => target.location = TargetLocation::parse(value),
                19 => target.centroid_row = var_uint(value, 4).map(|v| v as u32),
                20 => target.centroid_col = var_uint(value, 4).map(|v| v as u32),
                _ => {}
            }
            pos = v_start + len;
        }
        Some(target)
    }

    /// The pack bytes: the BER-OID id then the present tags in ascending order.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_ber_oid(&mut out, self.id.min(MAX_TARGET_ID));
        let mut put = |tag: u8, value: &[u8]| {
            out.push(tag);
            push_ber_length(&mut out, value.len());
            out.extend_from_slice(value);
        };
        if let Some(v) = self.centroid_pixel {
            put(1, &var_uint_bytes(v));
        }
        if let Some(v) = self.boundary_top_left_pixel {
            put(2, &var_uint_bytes(v));
        }
        if let Some(v) = self.boundary_bottom_right_pixel {
            put(3, &var_uint_bytes(v));
        }
        if let Some(v) = self.priority {
            put(4, &[v]);
        }
        if let Some(v) = self.confidence_pct {
            put(5, &[v]);
        }
        if let Some(v) = self.location_offset_lat_deg {
            put(10, &offset_imap().encode(v));
        }
        if let Some(v) = self.location_offset_lon_deg {
            put(11, &offset_imap().encode(v));
        }
        if let Some(v) = self.hae_m {
            put(12, &hae_imap().encode(v));
        }
        if let Some(v) = &self.location {
            put(17, &v.encode());
        }
        if let Some(v) = self.centroid_row {
            put(19, &var_uint_bytes(v as u64));
        }
        if let Some(v) = self.centroid_col {
            put(20, &var_uint_bytes(v as u64));
        }
        out
    }
}

/// The MISB ST 0903 VMTI Local Set: the frame the reports describe, and the
/// reported targets. Nested in ST 0601 tag 74 it has no key, no length and no
/// checksum of its own (the enclosing item supplies them); standalone it is a
/// KLV packet with the ST 0903 UL and a trailing checksum, which is
/// [`encode_klv`](Self::encode_klv) / [`parse_klv`](Self::parse_klv).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VmtiLocalSet {
    /// Tag 2: precision time stamp, microseconds since the Unix epoch.
    pub timestamp_us: Option<u64>,
    /// Tag 3: name or description of the VMTI system, free text.
    pub system_name: Option<String>,
    /// Tag 4: ST 0903 version number.
    pub version: Option<u16>,
    /// Tag 5: targets detected in the frame, before any culling.
    pub total_targets: Option<u32>,
    /// Tag 6: targets actually reported in this set.
    pub reported_targets: Option<u32>,
    /// Tag 7: frame number the reports describe.
    pub frame_number: Option<u32>,
    /// Tag 8: frame width in pixels (needed to read a pixel number back).
    pub frame_width: Option<u32>,
    /// Tag 9: frame height in pixels.
    pub frame_height: Option<u32>,
    /// Tag 10: the sensor the VMTI process ran on, free text.
    pub source_sensor: Option<String>,
    /// Tag 11: horizontal field of view of that sensor, 0..180 degrees.
    pub hfov_deg: Option<f64>,
    /// Tag 12: vertical field of view, 0..180 degrees.
    pub vfov_deg: Option<f64>,
    /// Tag 13: ST 1204 MIIS core identifier of the imagery.
    pub miis_core_id: Option<MiisCoreId>,
    /// Tag 101: the VTarget series.
    pub targets: Vec<VTarget>,
}

impl VmtiLocalSet {
    /// Parse the nested TLV body of ST 0601 tag 74. `None` when the nesting is
    /// not walkable (a bogus BER length, a value running past the end, a
    /// malformed target pack); a single tag whose value is the wrong size is
    /// skipped, leaving the rest of the set. A checksum (tag 1) is ignored
    /// here: it only means something on the standalone packet.
    pub fn parse(body: &[u8]) -> Option<Self> {
        let mut ls = Self::default();
        let mut pos = 0;
        while pos < body.len() {
            let (tag, tag_bytes) = ber_oid_tag(body, pos)?;
            let (len, l_bytes) = ber_length(body, pos.checked_add(tag_bytes)?)?;
            let v_start = pos + tag_bytes + l_bytes;
            let value = body.get(v_start..v_start.checked_add(len)?)?;
            match tag {
                2 => ls.timestamp_us = value.try_into().ok().map(u64::from_be_bytes),
                3 => ls.system_name = utf8_string(value),
                4 => ls.version = var_uint(value, 2).map(|v| v as u16),
                5 => ls.total_targets = var_uint(value, 3).map(|v| v as u32),
                6 => ls.reported_targets = var_uint(value, 3).map(|v| v as u32),
                7 => ls.frame_number = var_uint(value, 3).map(|v| v as u32),
                8 => ls.frame_width = var_uint(value, 3).map(|v| v as u32),
                9 => ls.frame_height = var_uint(value, 3).map(|v| v as u32),
                10 => ls.source_sensor = utf8_string(value),
                11 => ls.hfov_deg = fov_imap().decode(value),
                12 => ls.vfov_deg = fov_imap().decode(value),
                13 => ls.miis_core_id = MiisCoreId::parse(value),
                101 => ls.targets = parse_target_series(value)?,
                _ => {}
            }
            pos = v_start + len;
        }
        Some(ls)
    }

    /// The nested TLV body ST 0601 tag 74 carries: present tags in ascending
    /// order, no checksum.
    pub fn encode_body(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let mut put = |tag: u8, value: &[u8]| {
            out.push(tag);
            push_ber_length(&mut out, value.len());
            out.extend_from_slice(value);
        };
        if let Some(v) = self.timestamp_us {
            put(2, &v.to_be_bytes());
        }
        if let Some(v) = &self.system_name {
            put(3, v.as_bytes());
        }
        if let Some(v) = self.version {
            put(4, &var_uint_bytes(v as u64));
        }
        let counts = [
            (5, self.total_targets),
            (6, self.reported_targets),
            (7, self.frame_number),
            (8, self.frame_width),
            (9, self.frame_height),
        ];
        for (tag, count) in counts {
            if let Some(v) = count {
                put(tag, &var_uint_bytes((v as u64).min(MAX_V3)));
            }
        }
        if let Some(v) = &self.source_sensor {
            put(10, v.as_bytes());
        }
        if let Some(v) = self.hfov_deg {
            put(11, &fov_imap().encode(v));
        }
        if let Some(v) = self.vfov_deg {
            put(12, &fov_imap().encode(v));
        }
        if let Some(v) = &self.miis_core_id {
            put(13, &v.encode());
        }
        if !self.targets.is_empty() {
            let mut series = Vec::new();
            for target in &self.targets {
                let pack = target.encode();
                push_ber_length(&mut series, pack.len());
                series.extend_from_slice(&pack);
            }
            put(101, &series);
        }
        out
    }

    /// The standalone KLV packet: the ST 0903 UL, a BER length, the set, and
    /// the trailing checksum (tag 1), for a VMTI stream carried on its own
    /// rather than inside ST 0601.
    pub fn encode_klv(&self) -> Vec<u8> {
        let mut body = self.encode_body();
        // The checksum's tag and length enter the sum, then its value is the
        // sum over everything before it (key + BER length included).
        body.push(1);
        body.push(2);
        let mut packet = Vec::with_capacity(16 + 5 + body.len() + 2);
        packet.extend_from_slice(&VMTI_LOCAL_SET_KEY);
        push_ber_length(&mut packet, body.len() + 2);
        packet.extend_from_slice(&body);
        let bcc = checksum_16(&packet);
        packet.extend_from_slice(&bcc.to_be_bytes());
        packet
    }

    /// Parse a standalone VMTI KLV packet. `None` for a wrong key, a malformed
    /// structure, or a trailing checksum that does not match; a set with no
    /// checksum at all is accepted, since ST 0903 leaves it optional.
    pub fn parse_klv(packet: &[u8]) -> Option<Self> {
        if packet.get(..16)? != VMTI_LOCAL_SET_KEY {
            return None;
        }
        let (body_len, len_bytes) = ber_length(packet, 16)?;
        let body_start = 16 + len_bytes;
        let body = packet.get(body_start..body_start.checked_add(body_len)?)?;
        // ST 0903 puts the checksum last, so it is the final four body bytes
        // when present: tag 1, length 2, then the sum over all that precedes it.
        if let Some(tail) = body.len().checked_sub(4).and_then(|at| body.get(at..)) {
            if tail[0] == 1 && tail[1] == 2 {
                let covered = &packet[..body_start + body.len() - 2];
                if u16::from_be_bytes([tail[2], tail[3]]) != checksum_16(covered) {
                    return None;
                }
            }
        }
        Self::parse(body)
    }
}

/// Parse the VTarget series (VMTI LS tag 101): each pack is preceded by its own
/// BER length. Every pack is bounds-checked against real bytes before it is
/// pushed, so a bogus length allocates nothing.
fn parse_target_series(body: &[u8]) -> Option<Vec<VTarget>> {
    let mut targets = Vec::new();
    let mut pos = 0;
    while pos < body.len() {
        let (len, l_bytes) = ber_length(body, pos)?;
        let start = pos.checked_add(l_bytes)?;
        let pack = body.get(start..start.checked_add(len)?)?;
        targets.push(VTarget::parse(pack)?);
        pos = start + len;
    }
    Some(targets)
}

/// The 1-based index of the pixel a normalized coordinate falls in.
#[cfg(feature = "analytics")]
fn containing_pixel(v: f32, span: u32) -> u32 {
    let scaled = floor(v as f64 * span as f64);
    if scaled < 1.0 {
        1
    } else if scaled >= span as f64 {
        span
    } else {
        scaled as u32 + 1
    }
}

/// The 1-based index of the last pixel before a normalized exclusive edge.
#[cfg(feature = "analytics")]
fn last_pixel_before(v: f32, span: u32) -> u32 {
    let scaled = -floor(-(v as f64) * span as f64);
    if scaled < 1.0 {
        1
    } else if scaled > span as f64 {
        span
    } else {
        scaled as u32
    }
}

/// Turn a frame's detections into a VMTI local set (M808): the bridge from
/// g2g's in-pipeline analytics to the standard a ground station reads.
///
/// Every `AnalyticsNode::Detection` becomes a [`VTarget`] with its centroid and
/// bounding box as ST 0903 pixel numbers over `frame_width` x `frame_height`,
/// and its confidence as a percentage. A detection with a `Tracks` relation to
/// a `Tracking` node carries that tracker's `object_id` as the target id, so
/// ids stay stable across frames; an untracked one gets its 1-based position in
/// the frame (a graph that mixes the two can therefore collide, so track all or
/// none).
///
/// The box is normalized and half open, so its near edges map to the pixel that
/// contains them and its far edges to the last pixel inside; both are clamped
/// into the frame, and a box on the frame edge still names a real pixel.
#[cfg(feature = "analytics")]
pub fn vmti_from_analytics(
    meta: &g2g_core::AnalyticsMeta,
    frame_width: u32,
    frame_height: u32,
) -> VmtiLocalSet {
    use g2g_core::{AnalyticsNode, RelationKind};

    let width = frame_width.max(1);
    let height = frame_height.max(1);
    let pixel_at = |x: f32, y: f32| (containing_pixel(x, width), containing_pixel(y, height));
    let pixel_end = |x: f32, y: f32| (last_pixel_before(x, width), last_pixel_before(y, height));
    let pixel_number = |(col, row): (u32, u32)| -> u64 {
        (row as u64 - 1)
            .saturating_mul(width as u64)
            .saturating_add(col as u64)
    };

    let mut set = VmtiLocalSet {
        version: Some(ST0903_VERSION),
        frame_width: Some(width),
        frame_height: Some(height),
        ..Default::default()
    };
    for (index, node) in meta.nodes.iter().enumerate() {
        let AnalyticsNode::Detection(detection) = node else {
            continue;
        };
        let tracked = meta.relations.iter().find_map(|r| {
            (r.kind == RelationKind::Tracks && r.from == index)
                .then(|| match meta.nodes.get(r.to) {
                    Some(AnalyticsNode::Tracking(t)) => Some(t.object_id),
                    _ => None,
                })
                .flatten()
        });
        let id = tracked
            .unwrap_or(set.targets.len() as u64 + 1)
            .min(MAX_TARGET_ID as u64) as u32;
        let bbox = detection.bbox;
        let confidence = crate::klv::round(detection.confidence as f64 * 100.0);
        set.targets.push(VTarget {
            id,
            centroid_pixel: Some(pixel_number(pixel_at(
                bbox.x + bbox.w / 2.0,
                bbox.y + bbox.h / 2.0,
            ))),
            boundary_top_left_pixel: Some(pixel_number(pixel_at(bbox.x, bbox.y))),
            boundary_bottom_right_pixel: Some(pixel_number(pixel_end(
                bbox.x + bbox.w,
                bbox.y + bbox.h,
            ))),
            confidence_pct: Some(confidence.clamp(0.0, 100.0) as u8),
            ..Default::default()
        });
    }
    let reported = set.targets.len() as u32;
    set.total_targets = Some(reported);
    set.reported_targets = Some(reported);
    set
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> VmtiLocalSet {
        VmtiLocalSet {
            timestamp_us: Some(1_231_798_102_000_000),
            system_name: Some(String::from("g2g VMTI")),
            version: Some(ST0903_VERSION),
            total_targets: Some(28),
            reported_targets: Some(2),
            frame_number: Some(4_009),
            frame_width: Some(1920),
            frame_height: Some(1080),
            source_sensor: Some(String::from("EO Nose")),
            hfov_deg: Some(12.5),
            vfov_deg: Some(7.1),
            miis_core_id: None,
            targets: alloc::vec![
                VTarget {
                    id: 1,
                    centroid_pixel: Some(409_601),
                    boundary_top_left_pixel: Some(407_681),
                    boundary_bottom_right_pixel: Some(413_441),
                    priority: Some(3),
                    confidence_pct: Some(87),
                    location_offset_lat_deg: Some(-0.011),
                    location_offset_lon_deg: Some(0.014),
                    hae_m: Some(214.0),
                    location: None,
                    centroid_row: Some(214),
                    centroid_col: Some(641),
                },
                VTarget {
                    id: 4_211,
                    centroid_pixel: Some(1_000_000),
                    confidence_pct: Some(41),
                    location: Some(TargetLocation {
                        lat_deg: 60.176_822,
                        lon_deg: 24.828_835,
                        hae_m: 145.2,
                        accuracy: alloc::vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66],
                    }),
                    ..Default::default()
                },
            ],
        }
    }

    /// The nested body round-trips: every decoded field returns, floats within
    /// one IMAPB quantization step.
    #[test]
    fn nested_body_round_trip() {
        let ls = sample();
        let got = VmtiLocalSet::parse(&ls.encode_body()).expect("walkable");
        assert_eq!(got.timestamp_us, ls.timestamp_us);
        assert_eq!(got.system_name, ls.system_name);
        assert_eq!(got.version, ls.version);
        assert_eq!(got.total_targets, ls.total_targets);
        assert_eq!(got.reported_targets, ls.reported_targets);
        assert_eq!(got.frame_number, ls.frame_number);
        assert_eq!(got.frame_width, ls.frame_width);
        assert_eq!(got.frame_height, ls.frame_height);
        assert_eq!(got.source_sensor, ls.source_sensor);
        assert_eq!(got.targets.len(), 2);
        assert_eq!(got.targets[0].id, 1);
        assert_eq!(got.targets[0].centroid_pixel, Some(409_601));
        assert_eq!(got.targets[0].boundary_top_left_pixel, Some(407_681));
        assert_eq!(got.targets[0].priority, Some(3));
        assert_eq!(got.targets[0].confidence_pct, Some(87));
        assert_eq!(got.targets[0].centroid_row, Some(214));
        assert_eq!(got.targets[0].centroid_col, Some(641));
        assert_eq!(got.targets[1].id, 4_211);
        let close = |a: Option<f64>, b: Option<f64>, eps: f64| {
            let (a, b) = (a.expect("present"), b.expect("present"));
            assert!((a - b).abs() <= eps, "{a} !~ {b}");
        };
        // The IMAPB step is 2^(ceil(log2(range)) - (8 * bytes - 1)), and the
        // mapping truncates, so a value comes back within one step below.
        close(got.hfov_deg, ls.hfov_deg, 1.0 / 128.0);
        close(got.vfov_deg, ls.vfov_deg, 1.0 / 128.0);
        close(
            got.targets[0].location_offset_lat_deg,
            ls.targets[0].location_offset_lat_deg,
            1.0 / 131_072.0,
        );
        close(got.targets[0].hae_m, ls.targets[0].hae_m, 1.0);
        let location = got.targets[1].location.as_ref().expect("location pack");
        let want = ls.targets[1].location.as_ref().unwrap();
        assert!((location.lat_deg - want.lat_deg).abs() <= 1e-6);
        assert!((location.lon_deg - want.lon_deg).abs() <= 1e-6);
        assert_eq!(
            location.accuracy, want.accuracy,
            "the undecoded accuracy group is preserved verbatim"
        );
    }

    /// The standalone packet form carries the ST 0903 key and a checksum that
    /// a flipped byte breaks.
    #[test]
    fn standalone_packet_checksum() {
        let mut packet = sample().encode_klv();
        assert_eq!(&packet[..16], &VMTI_LOCAL_SET_KEY);
        assert!(VmtiLocalSet::parse_klv(&packet).is_some());
        let mid = packet.len() / 2;
        packet[mid] ^= 0x01;
        assert_eq!(VmtiLocalSet::parse_klv(&packet), None);
    }

    /// IMAPB matches the ST 1201 mapping at both ends of a range and re-encodes
    /// what it decoded.
    #[test]
    fn imapb_end_points() {
        let map = Imapb::new(-19.2, 19.2, 3);
        assert!(map.decode(&map.encode(-19.2)).unwrap() <= -19.199);
        assert!(map.decode(&map.encode(19.2)).unwrap() >= 19.199);
        assert!(map.decode(&map.encode(0.0)).unwrap().abs() < 1e-6);
        // Out-of-range flags decode to the range ends, not to a bogus angle.
        assert_eq!(map.decode(&[0xE0, 0x00, 0x00]), Some(-19.2));
        assert_eq!(map.decode(&[0xE1, 0x00, 0x00]), Some(19.2));
        // A NaN flag has no representation here, so the tag is skipped.
        assert_eq!(map.decode(&[0xD0, 0x00, 0x00]), None);
        // A wrong-width value is skipped rather than misread.
        assert_eq!(map.decode(&[0x00, 0x00]), None);
    }

    /// Every truncation of a nested set parses or fails, never panics.
    #[test]
    fn truncation_never_panics() {
        let body = sample().encode_body();
        for cut in 0..=body.len() {
            let _ = VmtiLocalSet::parse(&body[..cut]);
        }
        let packet = sample().encode_klv();
        for cut in 0..=packet.len() {
            let _ = VmtiLocalSet::parse_klv(&packet[..cut]);
        }
    }

    /// A target series whose pack length runs past the buffer fails the set,
    /// and allocates nothing on the way.
    #[test]
    fn bogus_target_length_fails() {
        // Tag 101, length 5, one pack claiming 0xFFFFFFFF bytes.
        let body = alloc::vec![101, 5, 0x84, 0xFF, 0xFF, 0xFF, 0xFF];
        assert_eq!(VmtiLocalSet::parse(&body), None);
        // A pack length longer than the series it sits in.
        let body = alloc::vec![101, 3, 0x40, 0x01, 0x02];
        assert_eq!(VmtiLocalSet::parse(&body), None);
    }

    #[test]
    fn variable_length_uint_round_trip() {
        for value in [0u64, 1, 255, 256, 65_535, 65_536, 16_777_215, 1_000_000] {
            let bytes = var_uint_bytes(value);
            assert_eq!(var_uint(&bytes, 8), Some(value), "{value}");
        }
        assert_eq!(var_uint_bytes(0), alloc::vec![0]);
        assert_eq!(var_uint(&[], 3), None, "an empty value is not a zero");
        assert_eq!(var_uint(&[1, 2, 3, 4], 3), None, "over-long value skipped");
    }

    #[cfg(feature = "analytics")]
    #[test]
    fn analytics_bridge_maps_detections() {
        use g2g_core::{
            AnalyticsMeta, AnalyticsNode, BBox, ObjectDetection, RelationKind, Tracking,
        };

        let mut meta = AnalyticsMeta::new();
        let det = meta.add_detection(ObjectDetection {
            bbox: BBox {
                x: 0.5,
                y: 0.5,
                w: 0.25,
                h: 0.25,
            },
            label: 3,
            confidence: 0.75,
        });
        let track = meta.push(AnalyticsNode::Tracking(Tracking { object_id: 77 }));
        meta.relate(det, track, RelationKind::Tracks);
        meta.add_detection(ObjectDetection {
            bbox: BBox {
                x: 0.0,
                y: 0.0,
                w: 1.0,
                h: 1.0,
            },
            label: 1,
            confidence: 1.0,
        });

        let set = vmti_from_analytics(&meta, 1920, 1080);
        assert_eq!(set.frame_width, Some(1920));
        assert_eq!(set.frame_height, Some(1080));
        assert_eq!(set.total_targets, Some(2));
        assert_eq!(set.reported_targets, Some(2));
        assert_eq!(set.targets.len(), 2);

        // Tracked detection: the tracker id becomes the target id, and the box
        // corners are pixel numbers over the declared frame width.
        let tracked = &set.targets[0];
        assert_eq!(tracked.id, 77);
        assert_eq!(tracked.confidence_pct, Some(75));
        assert_eq!(tracked.boundary_top_left_pixel, Some(540 * 1920 + 961));
        assert_eq!(tracked.boundary_bottom_right_pixel, Some(809 * 1920 + 1440));
        assert_eq!(tracked.centroid_pixel, Some(675 * 1920 + 1201));

        // Untracked: 1-based position, and a full-frame box clamps to the last
        // pixel rather than running off the end.
        let untracked = &set.targets[1];
        assert_eq!(untracked.id, 2);
        assert_eq!(untracked.boundary_top_left_pixel, Some(1));
        assert_eq!(
            untracked.boundary_bottom_right_pixel,
            Some(1079 * 1920 + 1920)
        );
    }
}
