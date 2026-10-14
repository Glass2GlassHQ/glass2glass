const POLYNOMIAL: u32 = 0x04C1_1DB7;

/// Initial value of the MPEG-2 systems CRC that trails every PSI section.
pub const PSI_SECTION_CRC_INIT: u32 = 0xFFFF_FFFF;

/// Initial value of the Ogg page checksum (RFC 3533 6).
pub const OGG_PAGE_CRC_INIT: u32 = 0;

const CRC_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut r = (i as u32) << 24;
        let mut bit = 0;
        while bit < 8 {
            r = if r & 0x8000_0000 != 0 {
                (r << 1) ^ POLYNOMIAL
            } else {
                r << 1
            };
            bit += 1;
        }
        table[i] = r;
        i += 1;
    }
    table
};

/// CRC-32 over polynomial `0x04C11DB7`, MSB first, no reflection and no final
/// inversion, from `init`: the MPEG-2 systems section CRC and the Ogg page
/// checksum differ only in that initial value.
pub fn crc32_mpeg2(init: u32, data: &[u8]) -> u32 {
    data.iter().fold(init, |crc, &byte| {
        (crc << 8) ^ CRC_TABLE[usize::from((crc >> 24) as u8 ^ byte)]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_check_value_for_each_initial_value() {
        assert_eq!(crc32_mpeg2(PSI_SECTION_CRC_INIT, b"123456789"), 0x0376_E6E7);
        assert_eq!(crc32_mpeg2(OGG_PAGE_CRC_INIT, b"123456789"), 0x89A1_897F);
    }
}
