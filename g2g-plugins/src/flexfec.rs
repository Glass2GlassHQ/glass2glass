//! FlexFEC (RFC 8627) forward error correction for RTP. Sans-IO, `no_std + alloc`.
//!
//! Like ULPFEC ([`crate::ulpfec`]) this XORs a set of media RTP packets into a
//! repair packet and reconstructs a single lost member by XORing the survivors,
//! no round trip. FlexFEC differs in two ways that matter:
//!
//! - The repair rides a **dedicated FEC SSRC** (its own RTP stream), not folded
//!   into the media headers, so it is told apart by SSRC / payload type.
//! - A **variable-length bitmask** (15, 46, or 109 bits, RFC 8627 4.2.2.1)
//!   names which packets (by sequence offset from an `SN base`) a repair covers,
//!   so one repair can protect far more than ULPFEC's 16, and arbitrary
//!   (strided / 2-D) protection patterns are expressible.
//!
//! [`FlexFecEncoder`] emits one repair per group; [`FlexFecDecoder`] buffers
//! media + repairs and recovers any group missing exactly one member, chaining
//! recoveries so 2-D (row + column) protection reconstructs bursts. The recovery
//! math (the XORed RTP fields) matches ULPFEC; only the header / mask differ.
//! Real-peer interop is unverified here (sandbox); validated g2g <-> g2g.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec;
use alloc::vec::Vec;

/// Fixed RTP header length (we generate / protect packets with no CSRC list).
const RTP_HEADER: usize = 12;
/// FlexFEC fixed header before the per-SSRC block: R/F/P/X/CC (1) + M/PT (1) +
/// length recovery (2) + TS recovery (4) + SSRCCount (1) + reserved (3).
const FLEX_FIXED: usize = 12;
/// Per protected SSRC: the media SSRC (4) + SN base (2), then the bitmask.
const FLEX_SSRC_PREFIX: usize = 6;
/// Offset of the variable-length mask within a repair packet.
const MASK_OFF: usize = RTP_HEADER + FLEX_FIXED + FLEX_SSRC_PREFIX;
/// Largest sequence offset the 109-bit mask can express.
const MAX_OFFSET: u16 = 108;

fn seq_of(pkt: &[u8]) -> u16 {
    u16::from_be_bytes([pkt[2], pkt[3]])
}

/// Encode a protection bitmask: bit `o` (offset from the SN base) set means the
/// packet at `SN base + o` is protected. Picks the shortest of the 15 / 46 / 109
/// bit forms (2 / 6 / 14 bytes), the high bit of each word flagging a follow-on
/// word (RFC 8627 4.2.2.1). `None` if any offset exceeds 108.
fn encode_mask(offsets: &[u16]) -> Option<Vec<u8>> {
    let max = *offsets.iter().max()?;
    if max <= 14 {
        let mut w0: u16 = 0; // k = 0 (no continuation)
        for &o in offsets {
            w0 |= 1 << (14 - o);
        }
        Some(w0.to_be_bytes().to_vec())
    } else if max <= 45 {
        let mut w0: u16 = 0x8000; // k = 1, continues
        let mut w1: u32 = 0; // k = 0 (bit 31), ends
        for &o in offsets {
            if o <= 14 {
                w0 |= 1 << (14 - o);
            } else {
                w1 |= 1 << (45 - o);
            }
        }
        let mut v = w0.to_be_bytes().to_vec();
        v.extend_from_slice(&w1.to_be_bytes());
        Some(v)
    } else if max <= MAX_OFFSET {
        let mut w0: u16 = 0x8000; // k = 1
        let mut w1: u32 = 0x8000_0000; // k = 1
        let mut w2: u64 = 0; // k = 0 (bit 63), ends
        for &o in offsets {
            if o <= 14 {
                w0 |= 1 << (14 - o);
            } else if o <= 45 {
                w1 |= 1 << (45 - o);
            } else {
                w2 |= 1 << (108 - o);
            }
        }
        let mut v = w0.to_be_bytes().to_vec();
        v.extend_from_slice(&w1.to_be_bytes());
        v.extend_from_slice(&w2.to_be_bytes());
        Some(v)
    } else {
        None
    }
}

/// Decode a protection bitmask into `(offsets, mask byte length)`. Walks the
/// 15 / 46 / 109 bit words by their continuation (k) bits. `None` if truncated.
fn decode_mask(m: &[u8]) -> Option<(Vec<u16>, usize)> {
    if m.len() < 2 {
        return None;
    }
    let w0 = u16::from_be_bytes([m[0], m[1]]);
    let mut offs = Vec::new();
    for i in 0..15u16 {
        if w0 & (1 << (14 - i)) != 0 {
            offs.push(i);
        }
    }
    if w0 & 0x8000 == 0 {
        return Some((offs, 2));
    }
    if m.len() < 6 {
        return None;
    }
    let w1 = u32::from_be_bytes([m[2], m[3], m[4], m[5]]);
    for i in 0..31u16 {
        if w1 & (1 << (30 - i)) != 0 {
            offs.push(15 + i);
        }
    }
    if w1 & 0x8000_0000 == 0 {
        return Some((offs, 6));
    }
    if m.len() < 14 {
        return None;
    }
    let w2 = u64::from_be_bytes(m[6..14].try_into().ok()?);
    for i in 0..63u16 {
        if w2 & (1 << (62 - i)) != 0 {
            offs.push(46 + i);
        }
    }
    Some((offs, 14))
}

/// Build a FlexFEC repair packet protecting `media` (RTP packets sorted by
/// sequence, `media[0]` the SN base, spanning at most 109 sequence numbers). The
/// repair is an RTP packet on `fec_pt` / `fec_ssrc` / `fec_seq`. `None` if
/// `media` is empty, over 109 packets, or spans more than 109 sequence numbers.
pub fn build_flexfec_packet(
    media: &[&[u8]],
    fec_pt: u8,
    fec_ssrc: u32,
    fec_seq: u16,
) -> Option<Vec<u8>> {
    if media.is_empty() || media.len() > 109 {
        return None;
    }
    if media.iter().any(|p| p.len() < RTP_HEADER) {
        return None;
    }
    let sn_base = seq_of(media[0]);
    let media_ssrc = u32::from_be_bytes(media[0][8..12].try_into().ok()?);
    let offsets: Vec<u16> = media
        .iter()
        .map(|p| seq_of(p).wrapping_sub(sn_base))
        .collect();
    if offsets.iter().any(|&o| o > MAX_OFFSET) {
        return None; // out of order, or spans more than the mask can express
    }
    let mask = encode_mask(&offsets)?;

    // XOR-recover the protected RTP fields and payloads (same scheme as ULPFEC).
    let mut pxcc = 0u8; // P|X|CC (low 6 bits of byte 0)
    let mut mpt = 0u8; // M|PT (byte 1)
    let mut ts = 0u32;
    let mut len_recovery = 0u16;
    let protection_len = media
        .iter()
        .map(|p| p.len() - RTP_HEADER)
        .max()
        .unwrap_or(0);
    let mut payload = vec![0u8; protection_len];
    for p in media {
        pxcc ^= p[0] & 0x3F;
        mpt ^= p[1];
        ts ^= u32::from_be_bytes([p[4], p[5], p[6], p[7]]);
        len_recovery ^= (p.len() - RTP_HEADER) as u16;
        for (dst, src) in payload.iter_mut().zip(&p[RTP_HEADER..]) {
            *dst ^= *src;
        }
    }

    let mut out = Vec::with_capacity(MASK_OFF + mask.len() + protection_len);
    // Repair packet's own RTP header: V=2, FEC PT, on the FEC SSRC.
    out.push(0x80);
    out.push(fec_pt & 0x7F);
    out.extend_from_slice(&fec_seq.to_be_bytes());
    out.extend_from_slice(&0u32.to_be_bytes()); // repair packet's own timestamp
    out.extend_from_slice(&fec_ssrc.to_be_bytes());
    // FlexFEC header (R=0, F=0: flexible bitmask form).
    out.push(pxcc & 0x3F);
    out.push(mpt);
    out.extend_from_slice(&len_recovery.to_be_bytes());
    out.extend_from_slice(&ts.to_be_bytes());
    out.push(1); // SSRCCount: a single protected media stream
    out.extend_from_slice(&[0u8; 3]); // reserved
    out.extend_from_slice(&media_ssrc.to_be_bytes());
    out.extend_from_slice(&sn_base.to_be_bytes());
    out.extend_from_slice(&mask);
    out.extend_from_slice(&payload);
    Some(out)
}

/// Parse a repair packet's `(SN base, protected sequence numbers, payload
/// offset)`. `None` if truncated or not a single-SSRC FlexFEC repair.
fn flex_header(fec: &[u8]) -> Option<(u16, Vec<u16>, usize)> {
    if fec.len() < MASK_OFF + 2 || fec[RTP_HEADER + 8] != 1 {
        return None; // need the fixed header + one SSRC block + at least one mask word
    }
    let sn_base = u16::from_be_bytes([
        fec[RTP_HEADER + FLEX_FIXED + 4],
        fec[RTP_HEADER + FLEX_FIXED + 5],
    ]);
    let (offsets, mask_len) = decode_mask(&fec[MASK_OFF..])?;
    let seqs = offsets.iter().map(|o| sn_base.wrapping_add(*o)).collect();
    Some((sn_base, seqs, MASK_OFF + mask_len))
}

/// The sequence numbers a repair packet protects.
fn protected_seqs(fec: &[u8]) -> Option<Vec<u16>> {
    flex_header(fec).map(|(_, seqs, _)| seqs)
}

/// Recover the one missing media packet of a repair's group, given the surviving
/// members `present` (`(seq, packet)`). `None` unless exactly one protected
/// sequence is absent.
pub fn recover_packet(fec: &[u8], present: &[(u16, &[u8])]) -> Option<Vec<u8>> {
    let (_, seqs, payload_off) = flex_header(fec)?;
    let missing: Vec<u16> = seqs
        .iter()
        .copied()
        .filter(|s| !present.iter().any(|(p, _)| p == s))
        .collect();
    if missing.len() != 1 {
        return None; // a single repair recovers exactly one loss per group
    }
    let missing_seq = missing[0];
    let group: Vec<&[u8]> = present
        .iter()
        .filter(|(s, _)| seqs.contains(s))
        .map(|(_, p)| *p)
        .collect();
    if group.iter().any(|p| p.len() < RTP_HEADER) {
        return None;
    }

    let pxcc_r = fec[RTP_HEADER] & 0x3F;
    let mpt_r = fec[RTP_HEADER + 1];
    let len_r = u16::from_be_bytes(fec[RTP_HEADER + 2..RTP_HEADER + 4].try_into().ok()?);
    let ts_r = u32::from_be_bytes(fec[RTP_HEADER + 4..RTP_HEADER + 8].try_into().ok()?);
    let fec_payload = fec.get(payload_off..)?;

    let mut pxcc = pxcc_r;
    let mut mpt = mpt_r;
    let mut ts = ts_r;
    let mut len = len_r;
    let mut payload = fec_payload.to_vec();
    let mut ssrc = [0u8; 4];
    for p in &group {
        pxcc ^= p[0] & 0x3F;
        mpt ^= p[1];
        ts ^= u32::from_be_bytes([p[4], p[5], p[6], p[7]]);
        len ^= (p.len() - RTP_HEADER) as u16;
        for (dst, src) in payload.iter_mut().zip(&p[RTP_HEADER..]) {
            *dst ^= *src;
        }
        ssrc.copy_from_slice(&p[8..12]); // all group members share the media SSRC
    }
    let recovered_len = len as usize;
    if recovered_len > payload.len() {
        return None; // inconsistent length recovery
    }

    let mut out = Vec::with_capacity(RTP_HEADER + recovered_len);
    out.push(0x80 | (pxcc & 0x3F)); // V=2 + recovered P/X/CC
    out.push(mpt); // recovered M/PT
    out.extend_from_slice(&missing_seq.to_be_bytes());
    out.extend_from_slice(&ts.to_be_bytes());
    out.extend_from_slice(&ssrc);
    out.extend_from_slice(&payload[..recovered_len]);
    Some(out)
}

/// Emits one FlexFEC repair per group of `group` media packets (up to 109).
#[derive(Debug)]
pub struct FlexFecEncoder {
    group: usize,
    fec_pt: u8,
    fec_ssrc: u32,
    fec_seq: u16,
    pending: Vec<Vec<u8>>,
}

impl FlexFecEncoder {
    pub fn new(group: usize, fec_pt: u8, fec_ssrc: u32) -> Self {
        Self {
            group: group.clamp(1, 109),
            fec_pt: fec_pt & 0x7F,
            fec_ssrc,
            fec_seq: 0,
            pending: Vec::new(),
        }
    }

    /// Feed a media RTP packet; returns a repair packet when the group closes.
    pub fn push(&mut self, media: &[u8]) -> Option<Vec<u8>> {
        self.pending.push(media.to_vec());
        if self.pending.len() >= self.group {
            let refs: Vec<&[u8]> = self.pending.iter().map(|v| v.as_slice()).collect();
            let fec = build_flexfec_packet(&refs, self.fec_pt, self.fec_ssrc, self.fec_seq);
            self.fec_seq = self.fec_seq.wrapping_add(1);
            self.pending.clear();
            return fec;
        }
        None
    }
}

/// Buffers recent media + repair packets and recovers single losses per group,
/// chaining recoveries (a recovered packet can complete another group).
#[derive(Debug)]
pub struct FlexFecDecoder {
    media: BTreeMap<u16, Vec<u8>>,
    fecs: VecDeque<Vec<u8>>,
    recovered: Vec<Vec<u8>>,
    capacity: usize,
}

impl Default for FlexFecDecoder {
    fn default() -> Self {
        Self::new(256)
    }
}

impl FlexFecDecoder {
    pub fn new(capacity: usize) -> Self {
        Self {
            media: BTreeMap::new(),
            fecs: VecDeque::new(),
            recovered: Vec::new(),
            capacity: capacity.max(16),
        }
    }

    /// Record a received media packet and attempt recovery of any open group.
    pub fn push_media(&mut self, seq: u16, packet: &[u8]) {
        self.media.insert(seq, packet.to_vec());
        self.trim();
        self.try_recover();
    }

    /// Record a received repair packet and attempt recovery.
    pub fn push_fec(&mut self, packet: &[u8]) {
        if self.fecs.len() >= self.capacity {
            self.fecs.pop_front();
        }
        self.fecs.push_back(packet.to_vec());
        self.try_recover();
    }

    /// Take the media packets recovered so far (to inject into the jitter buffer).
    pub fn take_recovered(&mut self) -> Vec<Vec<u8>> {
        core::mem::take(&mut self.recovered)
    }

    fn trim(&mut self) {
        while self.media.len() > self.capacity {
            let first = *self.media.keys().next().expect("non-empty");
            self.media.remove(&first);
        }
    }

    fn try_recover(&mut self) {
        let mut progressed = true;
        while progressed {
            progressed = false;
            let mut spent = None;
            for (idx, fec) in self.fecs.iter().enumerate() {
                let Some(seqs) = protected_seqs(fec) else {
                    continue;
                };
                let present: Vec<(u16, &[u8])> = seqs
                    .iter()
                    .filter_map(|s| self.media.get(s).map(|p| (*s, p.as_slice())))
                    .collect();
                let missing = seqs.len().saturating_sub(present.len());
                if missing == 1 {
                    if let Some(rec) = recover_packet(fec, &present) {
                        let seq = u16::from_be_bytes([rec[2], rec[3]]);
                        self.media.insert(seq, rec.clone());
                        self.recovered.push(rec);
                        spent = Some(idx);
                        progressed = true;
                        break;
                    }
                } else if missing == 0 {
                    spent = Some(idx); // fully received, no longer useful
                    progressed = true;
                    break;
                }
            }
            if let Some(idx) = spent {
                self.fecs.remove(idx);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rtppay::RtpH264Packetizer;

    /// `n` consecutive RTP packets (distinct payloads, one NAL each).
    fn media_run(n: u16) -> Vec<Vec<u8>> {
        let mut pkt = RtpH264Packetizer::new(96, 0x1111_2222);
        (0..n)
            .map(|i| {
                let b = i as u8;
                let nal = [0u8, 0, 0, 1, 0x61, b, b.wrapping_mul(7), 0xCC];
                pkt.packetize(&nal, 1000 + i as u32 * 90).remove(0)
            })
            .collect()
    }

    fn present_except(media: &[Vec<u8>], lost: &[usize]) -> Vec<(u16, Vec<u8>)> {
        media
            .iter()
            .enumerate()
            .filter(|(i, _)| !lost.contains(i))
            .map(|(_, p)| (u16::from_be_bytes([p[2], p[3]]), p.clone()))
            .collect()
    }

    #[test]
    fn mask_round_trips_across_all_three_widths() {
        // 15-bit (max offset 14), 46-bit (max 45), 109-bit (max 108).
        for offsets in [
            vec![0u16, 3, 14],
            vec![0u16, 14, 15, 45],
            vec![0u16, 20, 46, 108],
        ] {
            let m = encode_mask(&offsets).expect("encode");
            let (decoded, len) = decode_mask(&m).expect("decode");
            assert_eq!(decoded, offsets, "offsets survive the mask round trip");
            assert_eq!(len, m.len(), "reported mask length matches the encoding");
        }
    }

    #[test]
    fn recovers_a_single_loss_by_xor() {
        let media = media_run(8);
        let refs: Vec<&[u8]> = media.iter().map(|v| v.as_slice()).collect();
        let fec = build_flexfec_packet(&refs, 110, 0xFEC0_0000, 0).expect("fec");
        let present = present_except(&media, &[3]);
        let present_refs: Vec<(u16, &[u8])> =
            present.iter().map(|(s, p)| (*s, p.as_slice())).collect();
        let recovered = recover_packet(&fec, &present_refs).expect("recovered");
        assert_eq!(
            recovered, media[3],
            "FlexFEC reconstructs the lost packet byte-exact"
        );
    }

    #[test]
    fn one_repair_protects_more_than_ulpfecs_sixteen() {
        // A 24-packet group: ULPFEC's single-level 16-bit mask cannot cover this;
        // FlexFEC's wider mask protects all 24 and recovers a loss among them.
        let media = media_run(24);
        let refs: Vec<&[u8]> = media.iter().map(|v| v.as_slice()).collect();
        let fec = build_flexfec_packet(&refs, 110, 0xFEC0_0000, 0).expect("24-packet fec");
        assert_eq!(
            protected_seqs(&fec).unwrap().len(),
            24,
            "all 24 packets protected by one repair"
        );
        let present = present_except(&media, &[20]);
        let present_refs: Vec<(u16, &[u8])> =
            present.iter().map(|(s, p)| (*s, p.as_slice())).collect();
        let recovered = recover_packet(&fec, &present_refs).expect("recovered the 21st packet");
        assert_eq!(recovered, media[20]);
    }

    #[test]
    fn encoder_decoder_round_trip_recovers_a_loss() {
        let mut enc = FlexFecEncoder::new(20, 110, 0xFEC0_0000);
        let mut dec = FlexFecDecoder::new(64);
        let media = media_run(20);

        let mut fec = None;
        for p in &media {
            if let Some(f) = enc.push(p) {
                fec = Some(f);
            }
        }
        let fec = fec.expect("a repair closed the group of 20");

        for (i, p) in media.iter().enumerate() {
            if i != 7 {
                dec.push_media(u16::from_be_bytes([p[2], p[3]]), p);
            }
        }
        dec.push_fec(&fec);
        let recovered = dec.take_recovered();
        assert_eq!(recovered.len(), 1, "the one loss was recovered");
        assert_eq!(recovered[0], media[7]);
    }

    #[test]
    fn two_dimensional_protection_recovers_a_burst() {
        // Row + column repairs over a 4x4 block: a 2-packet loss in one row is
        // beyond a single repair, but the column repairs complete the rows by
        // chained recovery (the FlexFEC payoff over single 1-D FEC).
        let media = media_run(16);
        let row =
            |r: usize| -> Vec<&[u8]> { (0..4).map(|c| media[r * 4 + c].as_slice()).collect() };
        let col =
            |c: usize| -> Vec<&[u8]> { (0..4).map(|r| media[r * 4 + c].as_slice()).collect() };

        let mut dec = FlexFecDecoder::new(64);
        let mut seq = 0u16;
        let mut repairs = Vec::new();
        for r in 0..4 {
            repairs.push(build_flexfec_packet(&row(r), 110, 0xFEC0_0001, seq).unwrap());
            seq += 1;
        }
        for c in 0..4 {
            repairs.push(build_flexfec_packet(&col(c), 110, 0xFEC0_0002, seq).unwrap());
            seq += 1;
        }

        // Lose two packets in the same row (indices 5 and 6): no single row repair
        // can recover both, but each sits in a different column.
        let lost = [5usize, 6];
        for (i, p) in media.iter().enumerate() {
            if !lost.contains(&i) {
                dec.push_media(u16::from_be_bytes([p[2], p[3]]), p);
            }
        }
        for fec in &repairs {
            dec.push_fec(fec);
        }
        let mut recovered = dec.take_recovered();
        recovered.sort_by_key(|p| u16::from_be_bytes([p[2], p[3]]));
        assert_eq!(recovered.len(), 2, "2-D protection recovered both losses");
        assert!(recovered.iter().any(|r| *r == media[5]));
        assert!(recovered.iter().any(|r| *r == media[6]));
    }
}
