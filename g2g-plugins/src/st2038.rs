use alloc::vec::Vec;

use crate::annexb::{BitReader, BitWriter};
use crate::st291::{AncFieldOverflow, AncPacket};

const ZERO_BITS_BEFORE_LOCATION: u32 = 6;
// ST 2038 PES payloads may end in 0xFF stuffing bytes, which no packet starts with
const STUFFING_BYTE: u8 = 0xFF;

pub fn parse_st2038(payload: &[u8]) -> Option<Vec<AncPacket>> {
    let mut r = BitReader::new(payload);
    let mut packets = Vec::new();
    loop {
        let next_byte = payload.get(r.bit_pos() / 8);
        if next_byte.is_none_or(|&b| b == STUFFING_BYTE) {
            return Some(packets);
        }
        if r.read_bits(ZERO_BITS_BEFORE_LOCATION)? != 0 {
            return None;
        }
        packets.push(AncPacket::read(&mut r, 0)?);
        while !r.bit_pos().is_multiple_of(8) {
            if r.read_bit()? != 1 {
                return None;
            }
        }
    }
}

pub fn write_st2038(packets: &[AncPacket]) -> Result<Vec<u8>, AncFieldOverflow> {
    let mut w = BitWriter::default();
    for packet in packets {
        w.write_bits(0, ZERO_BITS_BEFORE_LOCATION);
        packet.write(&mut w, 0)?;
        while !w.bit_pos().is_multiple_of(8) {
            w.write_bit(1);
        }
    }
    Ok(w.into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // one CEA-708 CDP packet written by gst-plugins-rs cctost2038anc, from its st2038 test
    const GST_CEA708_PACKET: &[u8] = include_bytes!("../tests/fixtures/st2038_cea708_gst.bin");

    #[test]
    fn parses_and_rewrites_a_gst_packet() {
        let packets = parse_st2038(GST_CEA708_PACKET).expect("valid ST 2038");
        assert_eq!(packets.len(), 1);
        let packet = &packets[0];
        // header values gst-plugins-rs's st2038 test records for this packet
        assert_eq!((packet.did, packet.sdid), (0x61, 0x01));
        assert_eq!((packet.line_number, packet.horizontal_offset), (9, 0));
        assert!(!packet.c_not_y_channel);
        assert_eq!(packet.user_data.len(), 73);
        assert!(crate::cea::parse_cdp(&packet.user_data_bytes()).is_some());
        assert_eq!(write_st2038(&packets).unwrap(), GST_CEA708_PACKET);
    }

    #[test]
    fn round_trips_packets_and_stops_at_stuffing() {
        let mut audio = AncPacket::from_bytes(0xE7, 0x00, &[]);
        audio.user_data = vec![0x1FF, 0x200, 0x3FF];
        audio.line_number = 12;
        audio.horizontal_offset = 3;
        let packets = vec![AncPacket::from_bytes(0x61, 0x02, &[0xAA, 0xBB]), audio];
        let mut payload = write_st2038(&packets).unwrap();
        payload.extend_from_slice(&[STUFFING_BYTE; 3]);
        assert_eq!(parse_st2038(&payload), Some(packets));
    }

    #[test]
    fn rejects_bad_framing_bits() {
        let good = write_st2038(&[AncPacket::from_bytes(0x61, 0x02, &[0xAA, 0xBB])]).unwrap();
        let mut leading = good.clone();
        leading[0] |= 0x80;
        assert_eq!(parse_st2038(&leading), None);
        let mut alignment = good.clone();
        *alignment.last_mut().unwrap() &= 0xFE;
        assert_eq!(parse_st2038(&alignment), None);
        assert_eq!(parse_st2038(&good[..good.len() - 1]), None);
    }
}
