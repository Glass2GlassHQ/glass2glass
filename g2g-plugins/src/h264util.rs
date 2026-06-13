//! H.264 helpers for `WebCodecsDecode`: detect IDR keyframes (so each
//! `EncodedVideoChunk` is tagged key/delta) and build the WebCodecs `codec`
//! string from the SPS. NAL splitting is shared via [`crate::annexb`]. Pure and
//! host-testable; the element itself only runs in a browser.

use alloc::string::String;

use crate::annexb::nal_units;

/// The H.264 NAL unit type is the low 5 bits of the first NAL byte.
fn nal_type(nal: &[u8]) -> Option<u8> {
    nal.first().map(|b| b & 0x1F)
}

/// True if the access unit contains an IDR NAL (type 5): a keyframe. WebCodecs
/// requires each `EncodedVideoChunk` tagged `key` or `delta`.
pub(crate) fn h264_au_is_keyframe(au: &[u8]) -> bool {
    nal_units(au).any(|nal| nal_type(nal) == Some(5))
}

/// Build the WebCodecs `codec` string for an H.264 stream from its SPS (NAL
/// type 7): `"avc1."` followed by profile_idc, the constraint-set byte, and
/// level_idc as six uppercase hex digits (e.g. `"avc1.42E01E"`). `None` if the
/// access unit carries no SPS.
pub(crate) fn h264_codec_string(au: &[u8]) -> Option<String> {
    let sps = nal_units(au).find(|nal| nal_type(nal) == Some(7))?;
    // sps[0] is the NAL header; the next three bytes are profile_idc, the
    // constraint-set flags + reserved bits, and level_idc.
    let profile = *sps.get(1)?;
    let constraints = *sps.get(2)?;
    let level = *sps.get(3)?;
    Some(alloc::format!("avc1.{profile:02X}{constraints:02X}{level:02X}"))
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
    fn keyframe_detected_from_idr_nal() {
        // SPS (0x67), PPS (0x68), IDR slice (0x65).
        let au = wrap(&[&[0x67, 0x42, 0xE0, 0x1E], &[0x68, 0xCE], &[0x65, 0x88]]);
        assert!(h264_au_is_keyframe(&au));
    }

    #[test]
    fn non_keyframe_has_no_idr() {
        // A single non-IDR slice (type 1, 0x41).
        let au = wrap(&[&[0x41, 0x9A, 0x00]]);
        assert!(!h264_au_is_keyframe(&au));
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
