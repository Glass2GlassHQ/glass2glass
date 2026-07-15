//! RFC 4588 RTP retransmission (RTX) framing. Sans-IO, `no_std + alloc`.
//!
//! A retransmission is sent on a distinct payload type (the `apt`'s associated
//! RTX type, negotiated in SDP) with the 2-byte original sequence number (OSN)
//! prepended to the payload, so the resend is unambiguous under heavy loss and
//! accountable separately from the original flow. The RTX packet keeps the
//! original packet's marker bit, timestamp, and CSRCs; only the payload type,
//! SSRC, and sequence number differ. Reconstruction restores those three and
//! strips the OSN, yielding the byte-exact original packet for the depayloader
//! / jitter buffer (so the marker bit, which the OSN alone does not carry, is
//! preserved faithfully).
//!
//! Both SSRC-multiplexed RTX (a separate RTX SSRC) and session-multiplexed RTX
//! (same SSRC, distinct PT) are supported: the caller supplies the SSRC to set
//! on the rebuilt original.

use alloc::vec::Vec;

/// Minimum RTP header length (no CSRCs, no extension).
const RTP_MIN_HEADER: usize = 12;

/// Offset of the RTP payload: `12 + 4*CC`, plus a 4-byte-aligned extension when
/// the X bit is set. `None` if the packet is shorter than the header it
/// advertises.
pub fn rtp_payload_offset(packet: &[u8]) -> Option<usize> {
    if packet.len() < RTP_MIN_HEADER {
        return None;
    }
    let cc = (packet[0] & 0x0F) as usize;
    let mut offset = RTP_MIN_HEADER + 4 * cc;
    // X bit: a 4-byte extension header (profile + length in 32-bit words), then body.
    if packet[0] & 0x10 != 0 {
        let ext_start = offset;
        if packet.len() < ext_start + 4 {
            return None;
        }
        let words = u16::from_be_bytes([packet[ext_start + 2], packet[ext_start + 3]]) as usize;
        offset = ext_start + 4 + 4 * words;
    }
    if packet.len() < offset {
        return None;
    }
    Some(offset)
}

/// Wrap an original RTP packet as an RTX packet: copy the header (marker bit,
/// timestamp, CSRCs intact), swap in the RTX payload type / SSRC / sequence
/// number, and insert the original sequence number (OSN) at the front of the
/// payload. `None` if `original` is not a parseable RTP packet.
pub fn build_rtx_packet(original: &[u8], rtx_pt: u8, rtx_ssrc: u32, rtx_seq: u16) -> Option<Vec<u8>> {
    let payload_off = rtp_payload_offset(original)?;
    let original_seq = u16::from_be_bytes([original[2], original[3]]);
    let mut out = Vec::with_capacity(original.len() + 2);
    out.extend_from_slice(&original[..payload_off]);
    // M/PT byte: preserve the marker (high bit), swap in the RTX payload type.
    out[1] = (original[1] & 0x80) | (rtx_pt & 0x7F);
    out[2..4].copy_from_slice(&rtx_seq.to_be_bytes());
    out[8..12].copy_from_slice(&rtx_ssrc.to_be_bytes());
    out.extend_from_slice(&original_seq.to_be_bytes()); // OSN
    out.extend_from_slice(&original[payload_off..]);
    Some(out)
}

/// Reconstruct the original RTP packet from an RTX packet: read the OSN, restore
/// the original payload type / SSRC / sequence number, and strip the OSN. `None`
/// if the packet is too short to be a valid RTX packet (header + OSN).
pub fn parse_rtx_packet(rtx: &[u8], original_pt: u8, original_ssrc: u32) -> Option<Vec<u8>> {
    let payload_off = rtp_payload_offset(rtx)?;
    if rtx.len() < payload_off + 2 {
        return None;
    }
    let osn = u16::from_be_bytes([rtx[payload_off], rtx[payload_off + 1]]);
    let mut out = Vec::with_capacity(rtx.len() - 2);
    out.extend_from_slice(&rtx[..payload_off]);
    out[1] = (rtx[1] & 0x80) | (original_pt & 0x7F);
    out[2..4].copy_from_slice(&osn.to_be_bytes());
    out[8..12].copy_from_slice(&original_ssrc.to_be_bytes());
    out.extend_from_slice(&rtx[payload_off + 2..]);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rtppay::RtpH264Packetizer;

    /// An RTX wrap followed by a parse must reproduce the original packet byte
    /// for byte, including the marker bit (which the OSN does not carry).
    #[test]
    fn rtx_round_trip_is_byte_exact() {
        let mut pkt = RtpH264Packetizer::new(96, 0x1234_5678);
        // A single small NAL -> one packet with the marker set (last of the AU).
        let packets = pkt.packetize(&[0u8, 0, 0, 1, 0x65, 0xAA, 0xBB], 9000);
        let original = &packets[0];
        assert_eq!(original[1] & 0x80, 0x80, "fixture packet has the marker set");

        let rtx = build_rtx_packet(original, 97, 0xDEAD_BEEF, 5).expect("built");
        // The RTX packet carries the RTX PT, RTX SSRC, RTX sequence, marker kept.
        assert_eq!(rtx[1] & 0x7F, 97, "rtx payload type");
        assert_eq!(rtx[1] & 0x80, 0x80, "marker preserved on the rtx packet");
        assert_eq!(u16::from_be_bytes([rtx[2], rtx[3]]), 5, "rtx sequence");
        assert_eq!(u32::from_be_bytes([rtx[8], rtx[9], rtx[10], rtx[11]]), 0xDEAD_BEEF);
        // OSN is the original sequence, prepended to the payload.
        let off = rtp_payload_offset(original).unwrap();
        assert_eq!(u16::from_be_bytes([rtx[off], rtx[off + 1]]), 0, "osn = original seq 0");
        assert_eq!(rtx.len(), original.len() + 2, "exactly the OSN was added");

        let restored = parse_rtx_packet(&rtx, 96, 0x1234_5678).expect("parsed");
        assert_eq!(&restored, original, "reconstruction is byte-exact");
    }

    #[test]
    fn payload_offset_accounts_for_csrc_and_extension() {
        // Base header, CC=2 (two CSRC words), no extension.
        let mut p = alloc::vec![0x82u8, 96, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0];
        p.extend_from_slice(&[0; 8]); // two CSRC identifiers
        p.push(0xFF); // one payload byte
        assert_eq!(rtp_payload_offset(&p), Some(12 + 8));

        // X bit set, one extension word.
        let mut e = alloc::vec![0x90u8, 96, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0];
        e.extend_from_slice(&[0xBE, 0xDE, 0, 1]); // ext header: 1 word follows
        e.extend_from_slice(&[0; 4]); // the extension word
        e.push(0xFF);
        assert_eq!(rtp_payload_offset(&e), Some(12 + 4 + 4));
    }

    #[test]
    fn rejects_truncated_packets() {
        assert_eq!(rtp_payload_offset(&[0u8; 8]), None);
        assert_eq!(parse_rtx_packet(&[0u8; 12], 96, 1), None, "no room for the OSN");
        assert_eq!(build_rtx_packet(&[0u8; 4], 97, 1, 0), None);
    }
}
