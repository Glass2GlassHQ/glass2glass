use alloc::vec::Vec;

use crate::mpegcrc::{crc32_mpeg2, PSI_SECTION_CRC_INIT};

pub const TABLE_ID_SPLICE_INFO: u8 = 0xFC;
// the registration descriptor SCTE 35 puts in the PMT program_info loop
pub(crate) const CUEI_REGISTRATION: &[u8] = &[0x05, 4, b'C', b'U', b'E', b'I'];
// `section_length` is twelve bits
pub(crate) const MAX_SPLICE_INFO_SECTION_BYTES: usize = 3 + 0xFFF;

const SECTION_HEADER_LEN: usize = 3;
const CRC_LEN: usize = 4;
// protocol_version through descriptor_loop_length with an empty command, then the CRC
const MIN_SECTION_LENGTH: usize = 1 + 5 + 1 + 3 + 1 + 2 + CRC_LEN;
// the byte whose low bit is pts_adjustment's top bit, the next four hold the rest
const PTS_ADJUSTMENT_AT: usize = 4;
const PTS_ADJUSTMENT_TOP_BIT: u8 = 0x01;
const PTS_MODULUS: i128 = 1 << 33;
const SPLICE_NULL_COMMAND: u8 = 0x00;
// section_syntax_indicator 0, private_indicator 0, sap_type 3 (not specified)
const SAP_TYPE_UNSPECIFIED_FLAGS: u8 = 0x30;
// cw_index 0xFF, tier 0xFFF, splice_command_length 0
const UNENCRYPTED_UNTIERED_EMPTY_COMMAND: [u8; 4] = [0xFF, 0xFF, 0xF0, 0x00];

// `section` cut to its declared length with `ticks` added to its pts_adjustment, `None` unless it is a splice_info_section whose CRC checks out
pub fn shift_pts_adjustment(section: &[u8], ticks: i64) -> Option<Vec<u8>> {
    if section.first() != Some(&TABLE_ID_SPLICE_INFO) {
        return None;
    }
    let length_bytes = section.get(1..SECTION_HEADER_LEN)?;
    let section_length = usize::from(u16::from_be_bytes([
        length_bytes[0] & 0x0F,
        length_bytes[1],
    ]));
    if section_length < MIN_SECTION_LENGTH {
        return None;
    }
    let mut out = section.get(..SECTION_HEADER_LEN + section_length)?.to_vec();
    // a section followed by its own CRC sums to zero
    if crc32_mpeg2(PSI_SECTION_CRC_INIT, &out) != 0 {
        return None;
    }
    let field = &mut out[PTS_ADJUSTMENT_AT..PTS_ADJUSTMENT_AT + 5];
    let adjustment = (u64::from(field[0] & PTS_ADJUSTMENT_TOP_BIT) << 32)
        | u64::from(u32::from_be_bytes([field[1], field[2], field[3], field[4]]));
    let shifted = (i128::from(adjustment) + i128::from(ticks)).rem_euclid(PTS_MODULUS) as u64;
    field[0] = (field[0] & !PTS_ADJUSTMENT_TOP_BIT) | (shifted >> 32) as u8;
    field[1..].copy_from_slice(&(shifted as u32).to_be_bytes());
    let body_end = out.len() - CRC_LEN;
    let crc = crc32_mpeg2(PSI_SECTION_CRC_INIT, &out[..body_end]);
    out[body_end..].copy_from_slice(&crc.to_be_bytes());
    Some(out)
}

// a splice_null section, the heartbeat a muxer sends while no cue is due
pub fn splice_null() -> Vec<u8> {
    let mut section = Vec::with_capacity(SECTION_HEADER_LEN + MIN_SECTION_LENGTH);
    section.push(TABLE_ID_SPLICE_INFO);
    section.push(SAP_TYPE_UNSPECIFIED_FLAGS);
    section.push(MIN_SECTION_LENGTH as u8);
    // protocol_version 0, unencrypted, pts_adjustment 0
    section.extend_from_slice(&[0; 6]);
    section.extend_from_slice(&UNENCRYPTED_UNTIERED_EMPTY_COMMAND);
    section.push(SPLICE_NULL_COMMAND);
    // descriptor_loop_length
    section.extend_from_slice(&[0, 0]);
    let crc = crc32_mpeg2(PSI_SECTION_CRC_INIT, &section);
    section.extend_from_slice(&crc.to_be_bytes());
    section
}

#[cfg(test)]
mod tests {
    use super::*;

    // splice_insert for event 1 at pts 0x1_0000_0000
    fn splice_insert(pts_adjustment: u64) -> Vec<u8> {
        let mut body = Vec::from([0x00, 0x00 | (pts_adjustment >> 32) as u8]);
        body.extend_from_slice(&(pts_adjustment as u32).to_be_bytes());
        // cw_index, tier 0xFFF, splice_command_length 20
        body.extend_from_slice(&[0xFF, 0xFF, 0xF0, 0x14]);
        // splice_insert: event id 1, not cancelled, out of network, program splice, no duration, immediate 0, splice_time pts 0x1_0000_0000
        body.extend_from_slice(&[0x05, 0, 0, 0, 1, 0x7F, 0xEF, 0xFF, 0x00, 0x00, 0x00, 0x00]);
        // unique_program_id, avail_num, avails_expected
        body.extend_from_slice(&[0x00, 0x01, 0x00, 0x00]);
        body.extend_from_slice(&[0, 0, 0, 0]);
        body.extend_from_slice(&[0, 0]);
        let section_length = body.len() + CRC_LEN;
        let mut section = Vec::from([
            TABLE_ID_SPLICE_INFO,
            SAP_TYPE_UNSPECIFIED_FLAGS | (section_length >> 8) as u8,
            section_length as u8,
        ]);
        section.extend_from_slice(&body);
        let crc = crc32_mpeg2(PSI_SECTION_CRC_INIT, &section);
        section.extend_from_slice(&crc.to_be_bytes());
        section
    }

    #[test]
    fn a_shift_wraps_pts_adjustment_and_keeps_the_crc_valid() {
        let shifted = shift_pts_adjustment(&splice_insert(0x1_FFFF_FFF0), 0x20).expect("valid");
        assert_eq!(shifted, splice_insert(0x10));
        let back = shift_pts_adjustment(&shifted, -0x20).expect("valid");
        assert_eq!(back, splice_insert(0x1_FFFF_FFF0));
    }

    #[test]
    fn a_bad_crc_or_a_short_section_is_refused() {
        let mut corrupt = splice_insert(0);
        corrupt[14] ^= 0x01;
        assert!(shift_pts_adjustment(&corrupt, 0).is_none());
        let whole = splice_insert(0);
        assert!(shift_pts_adjustment(&whole[..whole.len() - 1], 0).is_none());
    }

    #[test]
    fn splice_null_is_a_valid_section() {
        let null = splice_null();
        assert_eq!(shift_pts_adjustment(&null, 0), Some(null.clone()));
        assert_eq!(null[13], SPLICE_NULL_COMMAND);
    }
}
