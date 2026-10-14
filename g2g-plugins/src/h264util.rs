//! H.264 helper for `WebCodecsDecode`: build the WebCodecs `codec` string from
//! the SPS. NAL splitting is shared via [`crate::annexb`]. Pure and
//! host-testable; the element itself only runs in a browser.

use alloc::string::String;

use crate::annexb::{h264_nal_type, nal_units};

/// Build the WebCodecs `codec` string for an H.264 stream from its SPS (NAL
/// type 7): `"avc1."` followed by profile_idc, the constraint-set byte, and
/// level_idc as six uppercase hex digits (e.g. `"avc1.42E01E"`). `None` if the
/// access unit carries no SPS.
pub(crate) fn h264_codec_string(au: &[u8]) -> Option<String> {
    let sps = nal_units(au).find(|nal| h264_nal_type(nal) == Some(7))?;
    // sps[0] is the NAL header; the next three bytes are profile_idc, the
    // constraint-set flags + reserved bits, and level_idc.
    let profile = *sps.get(1)?;
    let constraints = *sps.get(2)?;
    let level = *sps.get(3)?;
    Some(alloc::format!(
        "avc1.{profile:02X}{constraints:02X}{level:02X}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn wrap(nals: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        for nal in nals {
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(nal);
        }
        out
    }

    #[test]
    fn codec_string_from_sps() {
        let au = wrap(&[&[0x67, 0x42, 0xE0, 0x1E], &[0x65, 0x88]]);
        assert_eq!(h264_codec_string(&au).as_deref(), Some("avc1.42E01E"));
    }

    #[test]
    fn codec_string_none_without_sps() {
        let au = wrap(&[&[0x41, 0x9A]]);
        assert_eq!(h264_codec_string(&au), None);
    }
}
