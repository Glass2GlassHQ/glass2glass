//! Annex-B NAL unit splitting, shared by the WebCodecs payloader (`h264util`)
//! and the RTP packetizer (`rtppay`).

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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

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
    fn avcc_stops_on_a_truncated_length() {
        // 4-byte length says 10 bytes but only 3 follow.
        let mut avcc = Vec::new();
        avcc.extend_from_slice(&10u32.to_be_bytes());
        avcc.extend_from_slice(&[0x67, 0x42, 0xC0]);
        assert_eq!(avcc_nal_units(&avcc).count(), 0, "truncated NAL is dropped");
    }
}
