use alloc::vec::Vec;

use crate::annexb::{BitReader, BitWriter};

const LINE_NUMBER_BITS: u32 = 11;
const HORIZONTAL_OFFSET_BITS: u32 = 12;
const WORD_BITS: u32 = 10;
const MAX_USER_DATA_WORDS: usize = 0xFF;
const LINE_NUMBER_UNSPECIFIED: u16 = 0x7FF;
const HORIZONTAL_OFFSET_UNSPECIFIED: u16 = 0xFFF;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AncFieldOverflow {
    LineNumber,
    HorizontalOffset,
    UserDataLength,
    UserDataWord,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AncPacket {
    pub c_not_y_channel: bool,
    pub line_number: u16,
    pub horizontal_offset: u16,
    pub did: u8,
    pub sdid: u8,
    pub user_data: Vec<u16>,
}

impl AncPacket {
    pub fn from_bytes(did: u8, sdid: u8, bytes: &[u8]) -> Self {
        Self {
            c_not_y_channel: false,
            line_number: LINE_NUMBER_UNSPECIFIED,
            horizontal_offset: HORIZONTAL_OFFSET_UNSPECIFIED,
            did,
            sdid,
            user_data: bytes.iter().map(|&b| word_with_parity(b)).collect(),
        }
    }

    pub fn user_data_bytes(&self) -> Vec<u8> {
        self.user_data.iter().map(|&w| w as u8).collect()
    }

    fn check_widths(&self) -> Result<(), AncFieldOverflow> {
        if u32::from(self.line_number) >> LINE_NUMBER_BITS != 0 {
            return Err(AncFieldOverflow::LineNumber);
        }
        if u32::from(self.horizontal_offset) >> HORIZONTAL_OFFSET_BITS != 0 {
            return Err(AncFieldOverflow::HorizontalOffset);
        }
        if self.user_data.len() > MAX_USER_DATA_WORDS {
            return Err(AncFieldOverflow::UserDataLength);
        }
        if self
            .user_data
            .iter()
            .any(|&w| u32::from(w) >> WORD_BITS != 0)
        {
            return Err(AncFieldOverflow::UserDataWord);
        }
        Ok(())
    }

    pub(crate) fn write(
        &self,
        w: &mut BitWriter,
        zero_bits_after_location: u32,
    ) -> Result<(), AncFieldOverflow> {
        self.check_widths()?;
        w.write_bit(u32::from(self.c_not_y_channel));
        w.write_bits(u32::from(self.line_number), LINE_NUMBER_BITS);
        w.write_bits(u32::from(self.horizontal_offset), HORIZONTAL_OFFSET_BITS);
        w.write_bits(0, zero_bits_after_location);
        let words = [
            word_with_parity(self.did),
            word_with_parity(self.sdid),
            word_with_parity(self.user_data.len() as u8),
        ]
        .into_iter()
        .chain(self.user_data.iter().copied());
        let mut sum = 0u32;
        for word in words {
            w.write_bits(u32::from(word), WORD_BITS);
            sum += u32::from(word) & 0x1FF;
        }
        w.write_bits(u32::from(checksum_word(sum)), WORD_BITS);
        Ok(())
    }

    pub(crate) fn read(r: &mut BitReader, skipped_bits_after_location: u32) -> Option<Self> {
        let c_not_y_channel = r.read_bit()? != 0;
        let line_number = r.read_bits(LINE_NUMBER_BITS)? as u16;
        let horizontal_offset = r.read_bits(HORIZONTAL_OFFSET_BITS)? as u16;
        r.skip_bits(skipped_bits_after_location as usize)?;
        let did = read_parity_word(r)?;
        let sdid = read_parity_word(r)?;
        let data_count = read_parity_word(r)?;
        let mut sum = (u32::from(did.word) & 0x1FF)
            + (u32::from(sdid.word) & 0x1FF)
            + (u32::from(data_count.word) & 0x1FF);
        let mut user_data = Vec::with_capacity(usize::from(data_count.value));
        for _ in 0..data_count.value {
            // ST 291 leaves user data word format to the packet type, ST 299 audio carries data in b8
            let word = r.read_bits(WORD_BITS)? as u16;
            sum += u32::from(word) & 0x1FF;
            user_data.push(word);
        }
        if r.read_bits(WORD_BITS)? as u16 != checksum_word(sum) {
            return None;
        }
        Some(Self {
            c_not_y_channel,
            line_number,
            horizontal_offset,
            did: did.value,
            sdid: sdid.value,
            user_data,
        })
    }
}

pub fn word_with_parity(value: u8) -> u16 {
    let parity = (value.count_ones() & 1) as u16;
    u16::from(value) | (parity << 8) | ((parity ^ 1) << 9)
}

fn checksum_word(sum: u32) -> u16 {
    let sum9 = (sum & 0x1FF) as u16;
    sum9 | ((((sum9 >> 8) & 1) ^ 1) << 9)
}

struct ParityWord {
    word: u16,
    value: u8,
}

fn read_parity_word(r: &mut BitReader) -> Option<ParityWord> {
    let word = r.read_bits(WORD_BITS)? as u16;
    let value = word as u8;
    (word == word_with_parity(value)).then_some(ParityWord { word, value })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn encode(packet: &AncPacket) -> Vec<u8> {
        let mut w = BitWriter::default();
        packet.write(&mut w, 0).expect("fields fit");
        w.into_bytes()
    }

    fn decode(bytes: &[u8]) -> Option<AncPacket> {
        AncPacket::read(&mut BitReader::new(bytes), 0)
    }

    #[test]
    fn word_parity_is_even_with_inverse_b9() {
        // 0x61 has three ones: b8 = 1, b9 = 0
        assert_eq!(word_with_parity(0x61), 0x161);
        // 0x03 has two ones: b8 = 0, b9 = 1
        assert_eq!(word_with_parity(0x03), 0x203);
    }

    #[test]
    fn packet_round_trips_with_ten_bit_user_data() {
        let mut packet = AncPacket::from_bytes(0x61, 0x01, &[0x96, 0x69, 0x52]);
        packet.line_number = 9;
        packet.horizontal_offset = 0;
        packet.c_not_y_channel = true;
        packet.user_data.push(0x1FF);
        packet.user_data.push(0x200);
        assert_eq!(decode(&encode(&packet)), Some(packet));
    }

    #[test]
    fn rejects_bad_parity_and_checksum() {
        let good = encode(&AncPacket::from_bytes(0x61, 0x01, &[0x11, 0x22]));
        let location_bits = 1 + LINE_NUMBER_BITS + HORIZONTAL_OFFSET_BITS;
        let did_b8 = location_bits + 1;
        let checksum_b0 = location_bits + WORD_BITS * 6 - 1;
        for bit in [did_b8, checksum_b0] {
            let mut bad = good.clone();
            bad[bit as usize / 8] ^= 0x80 >> (bit % 8);
            assert_eq!(decode(&bad), None, "bit {bit} flipped");
        }
    }

    #[test]
    fn write_rejects_overflowing_fields() {
        let mut w = BitWriter::default();
        let mut long = AncPacket::from_bytes(0x61, 0x01, &[]);
        long.user_data = vec![0; MAX_USER_DATA_WORDS + 1];
        assert_eq!(long.write(&mut w, 0), Err(AncFieldOverflow::UserDataLength));
        let mut wide = AncPacket::from_bytes(0x61, 0x01, &[]);
        wide.user_data = vec![1 << WORD_BITS];
        assert_eq!(wide.write(&mut w, 0), Err(AncFieldOverflow::UserDataWord));
        let mut line = AncPacket::from_bytes(0x61, 0x01, &[]);
        line.line_number = 1 << LINE_NUMBER_BITS;
        assert_eq!(line.write(&mut w, 0), Err(AncFieldOverflow::LineNumber));
    }
}
