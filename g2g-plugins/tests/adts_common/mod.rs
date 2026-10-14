const ADTS_HEADER_LEN: usize = 7;
/// ADTS `profile` for AAC-LC: the audio object type (2) minus one.
const PROFILE_AAC_LC: u8 = 1;
const FREQ_INDEX_48000: u8 = 3;
const STEREO: u8 = 2;

/// A minimal ADTS AAC access unit (7-byte header, no CRC) at 48 kHz stereo
/// AAC-LC, carrying `payload`.
pub(crate) fn adts_au(payload: &[u8]) -> Vec<u8> {
    let frame_len = payload.len() + ADTS_HEADER_LEN;
    let mut au = Vec::from([
        0xFF,
        0xF1, // syncword | MPEG-4 | layer 0 | protection_absent
        (PROFILE_AAC_LC << 6) | (FREQ_INDEX_48000 << 2) | ((STEREO >> 2) & 1),
        ((STEREO & 3) << 6) | ((frame_len >> 11) & 3) as u8,
        ((frame_len >> 3) & 0xFF) as u8,
        (((frame_len & 7) << 5) as u8) | 0x1F, // buffer fullness (top bits)
        0xFC,                                  // buffer fullness (low) | one raw data block
    ]);
    au.extend_from_slice(payload);
    au
}
