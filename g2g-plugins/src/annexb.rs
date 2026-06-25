//! Annex-B/AVCC NAL splitting and RBSP bitstream helpers, shared by the
//! H.264/H.265 parsers, the WebCodecs payloader (`h264util`), and the RTP
//! packetizer (`rtppay`).

use alloc::vec::Vec;

/// Find the next Annex-B start code (`00 00 01` or `00 00 00 01`) at or after
/// `from`. Returns `(start_index, payload_index)`: the offset of the start code
/// and the offset of the NAL byte just past it.
pub(crate) fn next_start_code(data: &[u8], from: usize) -> Option<(usize, usize)> {
    let mut i = from;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 {
            if data[i + 2] == 1 {
                return Some((i, i + 3));
            }
            if i + 4 <= data.len() && data[i + 2] == 0 && data[i + 3] == 1 {
                return Some((i, i + 4));
            }
        }
        i += 1;
    }
    None
}

/// Iterator over the NAL units of an Annex-B buffer, start codes stripped.
pub(crate) struct NalUnits<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Iterator for NalUnits<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        let (_, begin) = next_start_code(self.data, self.pos)?;
        let end = match next_start_code(self.data, begin) {
            Some((sc, _)) => sc,
            None => self.data.len(),
        };
        self.pos = end;
        Some(&self.data[begin..end])
    }
}

pub(crate) fn nal_units(data: &[u8]) -> NalUnits<'_> {
    NalUnits { data, pos: 0 }
}

/// Heuristic: a buffer starting with an Annex-B start code is Annex-B, else it
/// is treated as AVCC (length-prefixed). Industry-standard (ffmpeg/gstreamer
/// h264parse do the same); a 4-byte AVCC length that happens to be `00 00 00
/// 01` is implausible for a real access unit and degrades to "no NAL found".
pub(crate) fn is_annex_b(data: &[u8]) -> bool {
    data.starts_with(&[0, 0, 0, 1]) || data.starts_with(&[0, 0, 1])
}

/// Iterator over AVCC NAL units: each NAL is preceded by a 4-byte big-endian
/// length (`lengthSizeMinusOne = 3`, the dominant case and what `retina` emits).
/// A truncated final length or NAL ends iteration rather than panicking.
pub(crate) struct AvccNals<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Iterator for AvccNals<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        if self.pos + 4 > self.data.len() {
            return None;
        }
        let len = u32::from_be_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ]) as usize;
        let start = self.pos + 4;
        let end = start.checked_add(len)?;
        if end > self.data.len() {
            return None;
        }
        self.pos = end;
        Some(&self.data[start..end])
    }
}

pub(crate) fn avcc_nal_units(data: &[u8]) -> AvccNals<'_> {
    AvccNals { data, pos: 0 }
}

/// NAL iterator over either framing, picked by [`is_annex_b`]. Yields the same
/// NAL payloads (start codes / length prefixes stripped) for a given stream.
pub(crate) enum NalUnitsAny<'a> {
    AnnexB(NalUnits<'a>),
    Avcc(AvccNals<'a>),
}

impl<'a> Iterator for NalUnitsAny<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        match self {
            NalUnitsAny::AnnexB(it) => it.next(),
            NalUnitsAny::Avcc(it) => it.next(),
        }
    }
}

pub(crate) fn nal_units_any(data: &[u8]) -> NalUnitsAny<'_> {
    if is_annex_b(data) {
        NalUnitsAny::AnnexB(nal_units(data))
    } else {
        NalUnitsAny::Avcc(avcc_nal_units(data))
    }
}

/// H.264 NAL unit type: the low 5 bits of the first NAL header byte.
#[cfg(any(
    all(target_os = "macos", feature = "vtdecode"),
    all(target_os = "android", feature = "mediacodec"),
    test
))]
pub(crate) fn h264_nal_type(nal: &[u8]) -> Option<u8> {
    nal.first().map(|b| b & 0x1F)
}

/// Collect the H.264 SPS (type 7) and PPS (type 8) NAL units from an access unit
/// (either framing), returned as owned copies so the caller can cache them across
/// frames. VideoToolbox builds its `CMVideoFormatDescription` from the parameter
/// sets (supplied out of band), not from NALs inside the decode sample, so the
/// decoder pulls these out and feeds only the VCL NALs to each frame.
#[cfg(any(
    all(target_os = "macos", feature = "vtdecode"),
    all(target_os = "android", feature = "mediacodec"),
    test
))]
pub(crate) fn h264_parameter_sets(au: &[u8]) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let mut sps = Vec::new();
    let mut pps = Vec::new();
    for nal in nal_units_any(au) {
        match h264_nal_type(nal) {
            Some(7) => sps.push(nal.to_vec()),
            Some(8) => pps.push(nal.to_vec()),
            _ => {}
        }
    }
    (sps, pps)
}

/// The H.265 (HEVC) NAL unit type: bits 1..=6 of the first header byte
/// (`(b >> 1) & 0x3F`), where H.264 uses the low 5 bits.
#[cfg(any(all(target_os = "android", feature = "mediacodec"), test))]
pub(crate) fn h265_nal_type(nal: &[u8]) -> Option<u8> {
    nal.first().map(|b| (b >> 1) & 0x3F)
}

/// VPS, SPS, and PPS NAL lists: the H.265 parameter sets.
#[cfg(any(all(target_os = "android", feature = "mediacodec"), test))]
pub(crate) type H265ParameterSets = (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<Vec<u8>>);

/// Collect the H.265 VPS (32), SPS (33), and PPS (34) NAL units from an access
/// unit, as owned copies the caller caches across frames. MediaCodec takes them
/// as the `csd-0` codec-specific data (VPS+SPS+PPS concatenated), the HEVC analog
/// of H.264's separate SPS / PPS.
#[cfg(any(all(target_os = "android", feature = "mediacodec"), test))]
pub(crate) fn h265_parameter_sets(au: &[u8]) -> H265ParameterSets {
    let mut vps = Vec::new();
    let mut sps = Vec::new();
    let mut pps = Vec::new();
    for nal in nal_units_any(au) {
        match h265_nal_type(nal) {
            Some(32) => vps.push(nal.to_vec()),
            Some(33) => sps.push(nal.to_vec()),
            Some(34) => pps.push(nal.to_vec()),
            _ => {}
        }
    }
    (vps, sps, pps)
}

/// Convert an access unit (Annex-B or AVCC) to AVCC form, each retained NAL
/// preceded by its 4-byte big-endian length (`lengthSizeMinusOne = 3`), keeping
/// only NALs for which `keep` returns true. VideoToolbox decode samples carry the
/// VCL (+ SEI) NALs length-prefixed; the parameter sets live in the format
/// description (see [`h264_parameter_sets`]), so the decoder excludes SPS / PPS /
/// AUD via `keep`. The inverse of [`avcc_nal_units`] for the kept NALs.
// AVCC framing is VideoToolbox-specific; MediaCodec takes Annex-B directly.
#[cfg(any(all(target_os = "macos", feature = "vtdecode"), test))]
pub(crate) fn to_avcc<F: Fn(&[u8]) -> bool>(au: &[u8], keep: F) -> Vec<u8> {
    let mut out = Vec::with_capacity(au.len() + 16);
    for nal in nal_units_any(au) {
        if !keep(nal) {
            continue;
        }
        out.extend_from_slice(&(nal.len() as u32).to_be_bytes());
        out.extend_from_slice(nal);
    }
    out
}

/// Convert AVCC (4-byte length-prefixed NALs) to Annex-B, each NAL preceded by a
/// 4-byte start code. The inverse of [`to_avcc`]: VideoToolbox's H.264 *encoder*
/// emits length-prefixed NALs, but the g2g pipeline is Annex-B framed
/// (downstream H.264 elements assume start codes), so the encoder converts on the
/// way out. Each NAL gets a 4-byte start code (`00 00 00 01`); the parameter sets
/// are prepended separately on keyframes (they live in the format description,
/// not the sample).
#[cfg(any(all(target_os = "macos", feature = "vtencode"), test))]
pub(crate) fn avcc_to_annexb(avcc: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(avcc.len() + 16);
    for nal in avcc_nal_units(avcc) {
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(nal);
    }
    out
}

/// Convert EBSP to RBSP by removing `0x03` emulation-prevention bytes that
/// follow two consecutive zero bytes (H.264 / H.265 share this encoding).
/// Always returns owned bytes for parser simplicity.
pub(crate) fn strip_emulation_prevention(ebsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ebsp.len());
    let mut zeros = 0usize;
    for &b in ebsp {
        if zeros >= 2 && b == 0x03 {
            zeros = 0;
            continue;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        out.push(b);
    }
    out
}

/// Convert RBSP to EBSP by inserting a `0x03` emulation-prevention byte before
/// any byte `<= 0x03` that follows two zero bytes, the inverse of
/// [`strip_emulation_prevention`]. Used by the SAMPLE-AES decryptor to re-escape
/// a NAL after decrypting its de-escaped payload.
#[cfg(any(feature = "hls", test))]
pub(crate) fn add_emulation_prevention(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len());
    let mut zeros = 0usize;
    for &b in rbsp {
        if zeros >= 2 && b <= 0x03 {
            out.push(0x03);
            zeros = 0;
        }
        out.push(b);
        zeros = if b == 0 { zeros + 1 } else { 0 };
    }
    out
}

/// MSB-first bit reader over a byte slice, the shared bitstream cursor for the
/// H.264 / H.265 SPS parsers. All readers return `None` on EOF rather than
/// panicking, so a partial / malformed header propagates as "field unknown"
/// instead of aborting the pipeline.
pub(crate) struct BitReader<'a> {
    buf: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Self { buf, bit_pos: 0 }
    }

    pub(crate) fn read_bit(&mut self) -> Option<u32> {
        let byte_idx = self.bit_pos / 8;
        let bit_off = 7 - (self.bit_pos % 8);
        if byte_idx >= self.buf.len() {
            return None;
        }
        let bit = u32::from((self.buf[byte_idx] >> bit_off) & 1);
        self.bit_pos += 1;
        Some(bit)
    }

    /// Read `n` (<= 32) bits MSB-first into a `u32`.
    pub(crate) fn read_bits(&mut self, n: u32) -> Option<u32> {
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | self.read_bit()?;
        }
        Some(value)
    }

    /// Advance past `n` bits without decoding them (e.g. H.265's fixed-size
    /// `profile_tier_level`). `None` if that would run past the end.
    pub(crate) fn skip_bits(&mut self, n: usize) -> Option<()> {
        let new_pos = self.bit_pos.checked_add(n)?;
        if new_pos > self.buf.len() * 8 {
            return None;
        }
        self.bit_pos = new_pos;
        Some(())
    }

    /// Unsigned exp-Golomb. Reads leading zeros to determine codeword length,
    /// then `n+1` bits of the codeword value, returns value - 1.
    pub(crate) fn read_ue(&mut self) -> Option<u32> {
        let mut leading_zeros = 0u32;
        loop {
            let b = self.read_bit()?;
            if b == 1 {
                break;
            }
            leading_zeros += 1;
            if leading_zeros > 31 {
                return None;
            }
        }
        let mut val = 1u32;
        for _ in 0..leading_zeros {
            val = (val << 1) | self.read_bit()?;
        }
        Some(val - 1)
    }

    /// Signed exp-Golomb, mapping ue to se per H.264 SS9.1.1.
    pub(crate) fn read_se(&mut self) -> Option<i32> {
        let ue = self.read_ue()?;
        Some(if ue & 1 == 1 {
            ((ue >> 1) + 1) as i32
        } else {
            -((ue >> 1) as i32)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn nal_iter_handles_3_and_4_byte_start_codes() {
        // 4-byte start code, then a 3-byte one.
        let mut au = Vec::new();
        au.extend_from_slice(&[0, 0, 0, 1, 0x67, 0x42]);
        au.extend_from_slice(&[0, 0, 1, 0x65, 0x88]);
        let nals: Vec<&[u8]> = nal_units(&au).collect();
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0], &[0x67, 0x42]);
        assert_eq!(nals[1], &[0x65, 0x88]);
    }

    #[test]
    fn avcc_iteration_matches_annexb_for_the_same_nals() {
        let sps: &[u8] = &[0x67, 0x42, 0xC0, 0x1E];
        let idr: &[u8] = &[0x65, 0x88, 0x84, 0x21, 0x0A];

        let mut annexb = Vec::new();
        annexb.extend_from_slice(&[0, 0, 0, 1]);
        annexb.extend_from_slice(sps);
        annexb.extend_from_slice(&[0, 0, 1]);
        annexb.extend_from_slice(idr);

        let mut avcc = Vec::new();
        avcc.extend_from_slice(&(sps.len() as u32).to_be_bytes());
        avcc.extend_from_slice(sps);
        avcc.extend_from_slice(&(idr.len() as u32).to_be_bytes());
        avcc.extend_from_slice(idr);

        assert!(is_annex_b(&annexb));
        assert!(!is_annex_b(&avcc));

        let from_annexb: Vec<&[u8]> = nal_units_any(&annexb).collect();
        let from_avcc: Vec<&[u8]> = nal_units_any(&avcc).collect();
        assert_eq!(from_annexb, vec![sps, idr]);
        assert_eq!(from_avcc, vec![sps, idr], "AVCC yields the same NALs");
    }

    #[test]
    fn avcc_to_annexb_prefixes_each_nal_with_a_start_code() {
        let sps: &[u8] = &[0x67, 0x42, 0xC0, 0x1E];
        let idr: &[u8] = &[0x65, 0x88, 0x84, 0x21, 0x0A];
        let mut avcc = Vec::new();
        avcc.extend_from_slice(&(sps.len() as u32).to_be_bytes());
        avcc.extend_from_slice(sps);
        avcc.extend_from_slice(&(idr.len() as u32).to_be_bytes());
        avcc.extend_from_slice(idr);

        let annexb = avcc_to_annexb(&avcc);
        assert!(is_annex_b(&annexb), "output is Annex-B framed");
        let nals: Vec<&[u8]> = nal_units(&annexb).collect();
        assert_eq!(nals, vec![sps, idr], "the same NALs, now start-code framed");
        // Every NAL is preceded by a 4-byte start code.
        assert_eq!(&annexb[..4], &[0, 0, 0, 1]);
    }

    #[test]
    fn avcc_stops_on_a_truncated_length() {
        // 4-byte length says 10 bytes but only 3 follow.
        let mut avcc = Vec::new();
        avcc.extend_from_slice(&10u32.to_be_bytes());
        avcc.extend_from_slice(&[0x67, 0x42, 0xC0]);
        assert_eq!(avcc_nal_units(&avcc).count(), 0, "truncated NAL is dropped");
    }

    #[test]
    fn parameter_sets_extracted_from_mixed_au() {
        // SPS (7), PPS (8), SEI (6), IDR slice (5), in Annex-B.
        let sps: &[u8] = &[0x67, 0x42, 0xE0, 0x1E];
        let pps: &[u8] = &[0x68, 0xCE, 0x3C, 0x80];
        let sei: &[u8] = &[0x06, 0x05, 0x01];
        let idr: &[u8] = &[0x65, 0x88, 0x84, 0x21];
        let mut au = Vec::new();
        for nal in [sps, pps, sei, idr] {
            au.extend_from_slice(&[0, 0, 0, 1]);
            au.extend_from_slice(nal);
        }
        let (got_sps, got_pps) = h264_parameter_sets(&au);
        assert_eq!(got_sps, vec![sps.to_vec()]);
        assert_eq!(got_pps, vec![pps.to_vec()]);
    }

    #[test]
    fn h265_parameter_sets_extracted_from_mixed_au() {
        // VPS (32), SPS (33), PPS (34), SEI (39), IDR slice (19), in Annex-B.
        // H.265 NAL header is 2 bytes; type = (byte0 >> 1) & 0x3F.
        let vps: &[u8] = &[0x40, 0x01, 0x0c];
        let sps: &[u8] = &[0x42, 0x01, 0x01];
        let pps: &[u8] = &[0x44, 0x01, 0xc0];
        let sei: &[u8] = &[0x4e, 0x01, 0x05];
        let idr: &[u8] = &[0x26, 0x01, 0xaf];
        let mut au = Vec::new();
        for nal in [vps, sps, pps, sei, idr] {
            au.extend_from_slice(&[0, 0, 0, 1]);
            au.extend_from_slice(nal);
        }
        let (got_vps, got_sps, got_pps) = h265_parameter_sets(&au);
        assert_eq!(got_vps, vec![vps.to_vec()]);
        assert_eq!(got_sps, vec![sps.to_vec()]);
        assert_eq!(got_pps, vec![pps.to_vec()]);
    }

    #[test]
    fn to_avcc_keeps_vcl_excludes_parameter_sets_and_round_trips() {
        let sps: &[u8] = &[0x67, 0x42, 0xE0, 0x1E];
        let pps: &[u8] = &[0x68, 0xCE];
        let sei: &[u8] = &[0x06, 0x05, 0x01];
        let idr: &[u8] = &[0x65, 0x88, 0x84, 0x21, 0x0A];
        let mut au = Vec::new();
        for nal in [sps, pps, sei, idr] {
            au.extend_from_slice(&[0, 0, 0, 1]);
            au.extend_from_slice(nal);
        }
        // Exclude SPS(7) / PPS(8) / AUD(9); keep SEI + VCL, like the decoder.
        let avcc = to_avcc(&au, |nal| !matches!(h264_nal_type(nal), Some(7..=9)));
        // The kept NALs, recovered by the AVCC iterator, are SEI then IDR.
        let kept: Vec<&[u8]> = avcc_nal_units(&avcc).collect();
        assert_eq!(kept, vec![sei, idr], "parameter sets dropped, order preserved");
    }
}
