//! RTP Forward Error Correction (ULPFEC, RFC 5109). Sans-IO, `no_std + alloc`.
//!
//! FEC trades bandwidth for latency-free recovery: the sender XORs a group of
//! media RTP packets into a repair packet, and the receiver reconstructs a
//! single lost packet of that group by XORing the repair with the survivors, no
//! round trip (the better fit when RTT is high or the path is one-way, unlike
//! NAK/RTX which need feedback). This is single-level ULPFEC (the `L=0` 16-bit
//! mask) protecting one contiguous run of up to 16 packets per repair packet.
//!
//! [`FecEncoder`] emits one repair packet per group; [`FecDecoder`] buffers
//! recent media + repair packets and recovers any group missing exactly one
//! member. The repair packets ride a distinct payload type (negotiated, like
//! RTX), so they are told apart from media at the receiver.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec;
use alloc::vec::Vec;

/// Fixed RTP header length (we generate / protect packets with no CSRC list).
const RTP_HEADER: usize = 12;
/// ULPFEC FEC header (RFC 5109 7.3) length for `L=0`.
const FEC_HEADER: usize = 10;
/// FEC level-0 header (protection length + 16-bit mask).
const FEC_LEVEL_HEADER: usize = 4;

/// Build a ULPFEC repair packet protecting `media` (contiguous-sequence RTP
/// packets, at most 16). The repair is itself an RTP packet on `fec_pt` /
/// `fec_ssrc` / `fec_seq`. `None` if `media` is empty or longer than 16.
pub fn build_fec_packet(media: &[&[u8]], fec_pt: u8, fec_ssrc: u32, fec_seq: u16) -> Option<Vec<u8>> {
    if media.is_empty() || media.len() > 16 {
        return None;
    }
    if media.iter().any(|p| p.len() < RTP_HEADER) {
        return None;
    }
    let sn_base = u16::from_be_bytes([media[0][2], media[0][3]]);

    // XOR-recover the protected header fields and the payloads.
    let mut pxcc = 0u8; // P|X|CC, the low 6 bits of byte 0
    let mut mpt = 0u8; // M|PT, byte 1
    let mut ts = 0u32;
    let mut len_recovery = 0u16;
    let protection_len = media.iter().map(|p| p.len() - RTP_HEADER).max().unwrap_or(0);
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

    // FEC level-0 mask: bit (15 - i) protects SN base + i.
    let mut mask = 0u16;
    for i in 0..media.len() {
        mask |= 1 << (15 - i);
    }

    let mut out = Vec::with_capacity(RTP_HEADER + FEC_HEADER + FEC_LEVEL_HEADER + protection_len);
    // Repair packet's own RTP header: V=2, no padding/ext/CSRC, M=0, FEC PT.
    out.push(0x80);
    out.push(fec_pt & 0x7F);
    out.extend_from_slice(&fec_seq.to_be_bytes());
    out.extend_from_slice(&0u32.to_be_bytes()); // the repair packet's own timestamp
    out.extend_from_slice(&fec_ssrc.to_be_bytes());
    // FEC header (E=0, L=0).
    out.push(pxcc); // E=0,L=0 in the top two bits (both clear)
    out.push(mpt);
    out.extend_from_slice(&sn_base.to_be_bytes());
    out.extend_from_slice(&ts.to_be_bytes());
    out.extend_from_slice(&len_recovery.to_be_bytes());
    // FEC level-0 header.
    out.extend_from_slice(&(protection_len as u16).to_be_bytes());
    out.extend_from_slice(&mask.to_be_bytes());
    out.extend_from_slice(&payload);
    Some(out)
}

/// The sequence numbers a repair packet protects (`SN base` + each set mask bit).
fn protected_seqs(fec: &[u8]) -> Option<Vec<u16>> {
    if fec.len() < RTP_HEADER + FEC_HEADER + FEC_LEVEL_HEADER {
        return None;
    }
    let sn_base = u16::from_be_bytes([fec[RTP_HEADER + 2], fec[RTP_HEADER + 3]]);
    let mask_off = RTP_HEADER + FEC_HEADER + 2;
    let mask = u16::from_be_bytes([fec[mask_off], fec[mask_off + 1]]);
    let mut seqs = Vec::new();
    for i in 0..16 {
        if mask & (1 << (15 - i)) != 0 {
            seqs.push(sn_base.wrapping_add(i));
        }
    }
    Some(seqs)
}

/// Recover the one missing media packet of a repair packet's group, given the
/// surviving members `present` (`(seq, packet)`). `None` unless exactly one of
/// the protected sequences is absent.
pub fn recover_packet(fec: &[u8], present: &[(u16, &[u8])]) -> Option<Vec<u8>> {
    let seqs = protected_seqs(fec)?;
    let missing: Vec<u16> = seqs.iter().copied().filter(|s| !present.iter().any(|(p, _)| p == s)).collect();
    if missing.len() != 1 {
        return None; // FEC recovers exactly one loss per group
    }
    let missing_seq = missing[0];
    // The survivors that belong to this group.
    let group: Vec<&[u8]> =
        present.iter().filter(|(s, _)| seqs.contains(s)).map(|(_, p)| *p).collect();
    if group.iter().any(|p| p.len() < RTP_HEADER) {
        return None;
    }

    let pxcc_r = fec[RTP_HEADER];
    let mpt_r = fec[RTP_HEADER + 1];
    let ts_r = u32::from_be_bytes(fec[RTP_HEADER + 4..RTP_HEADER + 8].try_into().ok()?);
    let len_r = u16::from_be_bytes(fec[RTP_HEADER + 8..RTP_HEADER + 10].try_into().ok()?);
    let prot_len =
        u16::from_be_bytes(fec[RTP_HEADER + FEC_HEADER..RTP_HEADER + FEC_HEADER + 2].try_into().ok()?)
            as usize;
    let fec_payload = &fec[RTP_HEADER + FEC_HEADER + FEC_LEVEL_HEADER..];
    if fec_payload.len() < prot_len {
        return None;
    }

    // XOR the repair fields with every survivor to recover the missing one.
    let mut pxcc = pxcc_r;
    let mut mpt = mpt_r;
    let mut ts = ts_r;
    let mut len = len_r;
    let mut payload = fec_payload[..prot_len].to_vec();
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

/// Emits one ULPFEC repair packet per group of `group` media packets.
#[derive(Debug)]
pub struct FecEncoder {
    group: usize,
    fec_pt: u8,
    fec_ssrc: u32,
    fec_seq: u16,
    pending: Vec<Vec<u8>>,
}

impl FecEncoder {
    pub fn new(group: usize, fec_pt: u8, fec_ssrc: u32) -> Self {
        Self { group: group.clamp(1, 16), fec_pt: fec_pt & 0x7F, fec_ssrc, fec_seq: 0, pending: Vec::new() }
    }

    /// Feed a media RTP packet; returns a repair packet when the group closes.
    pub fn push(&mut self, media: &[u8]) -> Option<Vec<u8>> {
        self.pending.push(media.to_vec());
        if self.pending.len() >= self.group {
            let refs: Vec<&[u8]> = self.pending.iter().map(|v| v.as_slice()).collect();
            let fec = build_fec_packet(&refs, self.fec_pt, self.fec_ssrc, self.fec_seq);
            self.fec_seq = self.fec_seq.wrapping_add(1);
            self.pending.clear();
            return fec;
        }
        None
    }
}

/// Buffers recent media + repair packets and recovers single losses per group.
#[derive(Debug)]
pub struct FecDecoder {
    media: BTreeMap<u16, Vec<u8>>,
    fecs: VecDeque<Vec<u8>>,
    recovered: Vec<Vec<u8>>,
    capacity: usize,
}

impl Default for FecDecoder {
    fn default() -> Self {
        Self::new(256)
    }
}

impl FecDecoder {
    pub fn new(capacity: usize) -> Self {
        Self { media: BTreeMap::new(), fecs: VecDeque::new(), recovered: Vec::new(), capacity: capacity.max(16) }
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

    /// Try every buffered repair packet; recover a group missing exactly one
    /// member, inject the recovery into the media map (so chained recovery and
    /// later groups see it), and retire the spent repair packet.
    fn try_recover(&mut self) {
        let mut progressed = true;
        while progressed {
            progressed = false;
            let mut spent = None;
            for (idx, fec) in self.fecs.iter().enumerate() {
                let Some(seqs) = protected_seqs(fec) else { continue };
                let present: Vec<(u16, &[u8])> =
                    seqs.iter().filter_map(|s| self.media.get(s).map(|p| (*s, p.as_slice()))).collect();
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

    /// Make `n` consecutive RTP packets (distinct payloads, one NAL each).
    fn media_run(n: u8) -> Vec<Vec<u8>> {
        let mut pkt = RtpH264Packetizer::new(96, 0x1111_2222);
        (0..n)
            .map(|i| {
                let nal = [0u8, 0, 0, 1, 0x61, i, i.wrapping_mul(7), 0xCC];
                pkt.packetize(&nal, 1000 + i as u32 * 90).remove(0)
            })
            .collect()
    }

    #[test]
    fn recovers_a_single_lost_packet_by_xor() {
        let media = media_run(4);
        let refs: Vec<&[u8]> = media.iter().map(|v| v.as_slice()).collect();
        let fec = build_fec_packet(&refs, 97, 0xFEC0_0000, 0).expect("fec built");

        // Lose the middle packet (index 2); recover from the FEC + the survivors.
        let lost_idx = 2;
        let present: Vec<(u16, &[u8])> = media
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != lost_idx)
            .map(|(_, p)| (u16::from_be_bytes([p[2], p[3]]), p.as_slice()))
            .collect();
        let recovered = recover_packet(&fec, &present).expect("recovered");
        assert_eq!(recovered, media[lost_idx], "FEC reconstructs the lost packet byte-exact");
    }

    #[test]
    fn two_losses_in_a_group_cannot_be_recovered() {
        let media = media_run(4);
        let refs: Vec<&[u8]> = media.iter().map(|v| v.as_slice()).collect();
        let fec = build_fec_packet(&refs, 97, 0, 0).unwrap();
        // Only two survivors -> two missing -> single-FEC cannot recover.
        let present: Vec<(u16, &[u8])> =
            media[..2].iter().map(|p| (u16::from_be_bytes([p[2], p[3]]), p.as_slice())).collect();
        assert!(recover_packet(&fec, &present).is_none());
    }

    #[test]
    fn encoder_emits_one_repair_per_group_decoder_recovers() {
        let mut enc = FecEncoder::new(4, 97, 0xFEC0_0000);
        let mut dec = FecDecoder::new(64);
        let media = media_run(4);

        let mut fec = None;
        for p in &media {
            if let Some(f) = enc.push(p) {
                fec = Some(f);
            }
        }
        let fec = fec.expect("a repair packet closed the group of 4");

        // Deliver all but packet index 1 to the decoder, then the FEC.
        for (i, p) in media.iter().enumerate() {
            if i != 1 {
                let seq = u16::from_be_bytes([p[2], p[3]]);
                dec.push_media(seq, p);
            }
        }
        dec.push_fec(&fec);
        let recovered = dec.take_recovered();
        assert_eq!(recovered.len(), 1, "the one loss was recovered");
        assert_eq!(recovered[0], media[1]);
    }
}
