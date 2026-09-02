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
//!
//! [`InterleavedFecEncoder`] is the burst-loss answer: over a block of
//! `rows x stride` packets it emits `stride` *column* repairs, where column `c`
//! protects the strided set `c, c+stride, c+2*stride, ...`. A burst of up to
//! `stride` consecutive losses then hits at most one packet per column, so each
//! column repair recovers its one loss and the whole burst is reconstructed,
//! where single-level FEC (one loss per contiguous group) would fail. The
//! decoder is unchanged: it reads each repair's mask generically and chains
//! recoveries, so column repairs just work (this is RFC 5109's interleaving via
//! a strided protection mask, the lighter cousin of full 2D row+column FEC).
//!
//! Every packet-XOR FEC scheme (RFC 2733 and its descendants ULPFEC, FlexFEC,
//! SMPTE 2022-1) shares the same algebra and the same receiver bookkeeping, and
//! differs only in how the recovery fields are packed on the wire. Both live
//! here as `XorFields` and `XorFecBuffer`, and [`crate::flexfec`] /
//! [`crate::st2022fec`] build their own wire formats on top.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec;
use alloc::vec::Vec;

/// Fixed RTP header length (we generate / protect packets with no CSRC list).
pub(crate) const RTP_HEADER: usize = 12;
/// ULPFEC FEC header (RFC 5109 7.3) length for `L=0`.
const FEC_HEADER: usize = 10;
/// FEC level-0 header (protection length + 16-bit mask).
const FEC_LEVEL_HEADER: usize = 4;

/// The XOR of the protected RTP fields over a set of media packets: the header
/// bits a receiver has to rebuild, plus the zero-padded payload. Folding a group
/// (encode) and undoing the survivors of a group (decode) are the same
/// operation, so both directions use [`XorFields::xor_in`].
#[derive(Debug, Default)]
pub(crate) struct XorFields {
    /// P|X|CC, the low 6 bits of RTP byte 0.
    pub(crate) pxcc: u8,
    /// M|PT, RTP byte 1.
    pub(crate) mpt: u8,
    pub(crate) ts: u32,
    /// Length recovery: the XOR of the payload lengths.
    pub(crate) len: u16,
    /// The XOR of the payloads, zero-padded to the longest.
    pub(crate) payload: Vec<u8>,
    /// Taken from the last packet folded in, not XORed: every member of a group
    /// carries the same media SSRC, so any survivor supplies it.
    pub(crate) ssrc: [u8; 4],
}

impl XorFields {
    /// Fold a whole group of media RTP packets. `None` if any is shorter than an
    /// RTP header.
    pub(crate) fn fold(media: &[&[u8]]) -> Option<Self> {
        let protection_len = media
            .iter()
            .map(|p| p.len().saturating_sub(RTP_HEADER))
            .max()
            .unwrap_or(0);
        let mut f = Self {
            payload: vec![0u8; protection_len],
            ..Self::default()
        };
        for p in media {
            f.xor_in(p)?;
        }
        Some(f)
    }

    /// Start from a repair packet's recovery fields, ready for the survivors to
    /// be XORed back out. `payload` is the repair's protected payload.
    pub(crate) fn from_repair(pxcc: u8, mpt: u8, ts: u32, len: u16, payload: Vec<u8>) -> Self {
        Self {
            pxcc: pxcc & 0x3F,
            mpt,
            ts,
            len,
            payload,
            ssrc: [0; 4],
        }
    }

    /// XOR one media packet in. `None` if it is shorter than an RTP header.
    pub(crate) fn xor_in(&mut self, p: &[u8]) -> Option<()> {
        if p.len() < RTP_HEADER {
            return None;
        }
        self.pxcc ^= p[0] & 0x3F;
        self.mpt ^= p[1];
        self.ts ^= u32::from_be_bytes([p[4], p[5], p[6], p[7]]);
        self.len ^= (p.len() - RTP_HEADER) as u16;
        for (dst, src) in self.payload.iter_mut().zip(&p[RTP_HEADER..]) {
            *dst ^= *src;
        }
        self.ssrc.copy_from_slice(&p[8..12]);
        Some(())
    }

    /// The media RTP packet these fields describe, on sequence `seq`. `None` if
    /// the recovered length does not fit the XORed payload (inconsistent input).
    pub(crate) fn to_rtp(&self, seq: u16) -> Option<Vec<u8>> {
        let len = self.len as usize;
        if len > self.payload.len() {
            return None;
        }
        let mut out = Vec::with_capacity(RTP_HEADER + len);
        out.push(0x80 | (self.pxcc & 0x3F)); // V=2 + recovered P/X/CC
        out.push(self.mpt); // recovered M/PT
        out.extend_from_slice(&seq.to_be_bytes());
        out.extend_from_slice(&self.ts.to_be_bytes());
        out.extend_from_slice(&self.ssrc);
        out.extend_from_slice(&self.payload[..len]);
        Some(out)
    }
}

/// Reads the sequence numbers a repair packet protects, in its wire format.
pub(crate) type ProtectedSeqsFn = fn(&[u8]) -> Option<Vec<u16>>;
/// Rebuilds the one missing member of a repair packet's group, in its wire
/// format, given the survivors as `(seq, packet)`.
pub(crate) type RecoverFn = fn(&[u8], &[(u16, &[u8])]) -> Option<Vec<u8>>;

/// Buffers received media + repair packets and chains single-loss recoveries.
/// The wire format is supplied as two function pointers: one reading a repair's
/// protected sequence set, one rebuilding a missing member from it.
#[derive(Debug)]
pub(crate) struct XorFecBuffer {
    media: BTreeMap<u16, Vec<u8>>,
    fecs: VecDeque<Vec<u8>>,
    recovered: Vec<Vec<u8>>,
    capacity: usize,
    protected_seqs: ProtectedSeqsFn,
    recover: RecoverFn,
}

impl XorFecBuffer {
    pub(crate) fn new(
        capacity: usize,
        protected_seqs: ProtectedSeqsFn,
        recover: RecoverFn,
    ) -> Self {
        Self {
            media: BTreeMap::new(),
            fecs: VecDeque::new(),
            recovered: Vec::new(),
            capacity: capacity.max(16),
            protected_seqs,
            recover,
        }
    }

    pub(crate) fn push_media(&mut self, seq: u16, packet: &[u8]) {
        self.media.insert(seq, packet.to_vec());
        while self.media.len() > self.capacity {
            let first = *self.media.keys().next().expect("non-empty");
            self.media.remove(&first);
        }
        self.try_recover();
    }

    pub(crate) fn push_fec(&mut self, packet: &[u8]) {
        if self.fecs.len() >= self.capacity {
            self.fecs.pop_front();
        }
        self.fecs.push_back(packet.to_vec());
        self.try_recover();
    }

    pub(crate) fn take_recovered(&mut self) -> Vec<Vec<u8>> {
        core::mem::take(&mut self.recovered)
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
                let Some(seqs) = (self.protected_seqs)(fec) else {
                    continue;
                };
                let present: Vec<(u16, &[u8])> = seqs
                    .iter()
                    .filter_map(|s| self.media.get(s).map(|p| (*s, p.as_slice())))
                    .collect();
                let missing = seqs.len().saturating_sub(present.len());
                if missing == 1 {
                    if let Some(rec) = (self.recover)(fec, &present) {
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

/// The survivors of `seqs` in `present`, and the one sequence that is missing.
/// `None` unless exactly one is absent (a single XOR repair recovers one loss).
pub(crate) fn split_group<'a>(
    seqs: &[u16],
    present: &[(u16, &'a [u8])],
) -> Option<(u16, Vec<&'a [u8]>)> {
    let mut missing = None;
    for s in seqs {
        if !present.iter().any(|(p, _)| p == s) {
            if missing.is_some() {
                return None;
            }
            missing = Some(*s);
        }
    }
    let group: Vec<&[u8]> = present
        .iter()
        .filter(|(s, _)| seqs.contains(s))
        .map(|(_, p)| *p)
        .collect();
    Some((missing?, group))
}

/// Build a ULPFEC repair packet protecting `media`, RTP packets sorted by
/// sequence and spanning at most 16 sequence numbers from the first (`media[0]`
/// is the SN base). The set may be *non-contiguous*: the level-0 mask is built
/// from each packet's sequence offset, so a strided / interleaved column (every
/// `stride`-th packet) is protected exactly like a contiguous run. The repair is
/// itself an RTP packet on `fec_pt` / `fec_ssrc` / `fec_seq`. `None` if `media`
/// is empty, longer than 16, or spans more than 16 sequence numbers.
pub fn build_fec_packet(
    media: &[&[u8]],
    fec_pt: u8,
    fec_ssrc: u32,
    fec_seq: u16,
) -> Option<Vec<u8>> {
    if media.is_empty() || media.len() > 16 {
        return None;
    }
    if media.iter().any(|p| p.len() < RTP_HEADER) {
        return None;
    }
    let sn_base = u16::from_be_bytes([media[0][2], media[0][3]]);

    // FEC level-0 mask: bit (15 - off) protects SN base + off, where off is each
    // packet's sequence distance from the base (0,1,2,... contiguous; 0,D,2D,...
    // interleaved). A span past 16 cannot be expressed in the 16-bit mask.
    let mut mask = 0u16;
    for p in media {
        let seq = u16::from_be_bytes([p[2], p[3]]);
        let off = seq.wrapping_sub(sn_base);
        if off >= 16 {
            return None; // out of order, or spans more than 16 sequence numbers
        }
        mask |= 1 << (15 - off);
    }

    // XOR-recover the protected header fields and the payloads.
    let f = XorFields::fold(media)?;
    let protection_len = f.payload.len();

    let mut out = Vec::with_capacity(RTP_HEADER + FEC_HEADER + FEC_LEVEL_HEADER + protection_len);
    // Repair packet's own RTP header: V=2, no padding/ext/CSRC, M=0, FEC PT.
    out.push(0x80);
    out.push(fec_pt & 0x7F);
    out.extend_from_slice(&fec_seq.to_be_bytes());
    out.extend_from_slice(&0u32.to_be_bytes()); // the repair packet's own timestamp
    out.extend_from_slice(&fec_ssrc.to_be_bytes());
    // FEC header (E=0, L=0).
    out.push(f.pxcc); // E=0,L=0 in the top two bits (both clear)
    out.push(f.mpt);
    out.extend_from_slice(&sn_base.to_be_bytes());
    out.extend_from_slice(&f.ts.to_be_bytes());
    out.extend_from_slice(&f.len.to_be_bytes());
    // FEC level-0 header.
    out.extend_from_slice(&(protection_len as u16).to_be_bytes());
    out.extend_from_slice(&mask.to_be_bytes());
    out.extend_from_slice(&f.payload);
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
    let (missing_seq, group) = split_group(&seqs, present)?;

    let pxcc_r = fec[RTP_HEADER];
    let mpt_r = fec[RTP_HEADER + 1];
    let ts_r = u32::from_be_bytes(fec[RTP_HEADER + 4..RTP_HEADER + 8].try_into().ok()?);
    let len_r = u16::from_be_bytes(fec[RTP_HEADER + 8..RTP_HEADER + 10].try_into().ok()?);
    let prot_len = u16::from_be_bytes(
        fec[RTP_HEADER + FEC_HEADER..RTP_HEADER + FEC_HEADER + 2]
            .try_into()
            .ok()?,
    ) as usize;
    let fec_payload = &fec[RTP_HEADER + FEC_HEADER + FEC_LEVEL_HEADER..];
    if fec_payload.len() < prot_len {
        return None;
    }

    // XOR the repair fields with every survivor to recover the missing one.
    let mut f =
        XorFields::from_repair(pxcc_r, mpt_r, ts_r, len_r, fec_payload[..prot_len].to_vec());
    for p in &group {
        f.xor_in(p)?;
    }
    f.to_rtp(missing_seq)
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
        Self {
            group: group.clamp(1, 16),
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
            let fec = build_fec_packet(&refs, self.fec_pt, self.fec_ssrc, self.fec_seq);
            self.fec_seq = self.fec_seq.wrapping_add(1);
            self.pending.clear();
            return fec;
        }
        None
    }
}

/// Emits interleaved (column) ULPFEC repairs for burst-loss recovery. Over a
/// block of `rows * stride` media packets it emits `stride` repair packets, one
/// per column; `rows * stride` must be at most 16 so each column fits the
/// level-0 mask. A burst of up to `stride` consecutive losses is recovered.
#[derive(Debug)]
pub struct InterleavedFecEncoder {
    rows: usize,
    stride: usize,
    fec_pt: u8,
    fec_ssrc: u32,
    fec_seq: u16,
    pending: Vec<Vec<u8>>,
}

impl InterleavedFecEncoder {
    /// `rows` x `stride` is the interleaving block; the product is clamped to 16
    /// (the level-0 mask width). `stride` is the burst length recovered.
    pub fn new(rows: usize, stride: usize, fec_pt: u8, fec_ssrc: u32) -> Self {
        let stride = stride.clamp(1, 16);
        // Keep rows * stride <= 16 so every column spans <= 16 sequence numbers.
        let rows = rows.clamp(1, 16 / stride);
        Self {
            rows,
            stride,
            fec_pt: fec_pt & 0x7F,
            fec_ssrc,
            fec_seq: 0,
            pending: Vec::new(),
        }
    }

    /// The interleaving block size (`rows * stride`), the number of media packets
    /// per emitted set of column repairs.
    pub fn block(&self) -> usize {
        self.rows * self.stride
    }

    /// Feed a media RTP packet; returns `stride` column repair packets when the
    /// block fills (empty otherwise). Each column `c` protects positions `c`,
    /// `c + stride`, ..., a strided set the decoder recovers independently.
    pub fn push(&mut self, media: &[u8]) -> Vec<Vec<u8>> {
        self.pending.push(media.to_vec());
        if self.pending.len() < self.rows * self.stride {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(self.stride);
        for c in 0..self.stride {
            let column: Vec<&[u8]> = (0..self.rows)
                .map(|r| self.pending[c + r * self.stride].as_slice())
                .collect();
            if let Some(fec) = build_fec_packet(&column, self.fec_pt, self.fec_ssrc, self.fec_seq) {
                self.fec_seq = self.fec_seq.wrapping_add(1);
                out.push(fec);
            }
        }
        self.pending.clear();
        out
    }
}

/// Buffers recent media + repair packets and recovers single losses per group.
#[derive(Debug)]
pub struct FecDecoder {
    inner: XorFecBuffer,
}

impl Default for FecDecoder {
    fn default() -> Self {
        Self::new(256)
    }
}

impl FecDecoder {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: XorFecBuffer::new(capacity, protected_seqs, recover_packet),
        }
    }

    /// Record a received media packet and attempt recovery of any open group.
    pub fn push_media(&mut self, seq: u16, packet: &[u8]) {
        self.inner.push_media(seq, packet);
    }

    /// Record a received repair packet and attempt recovery.
    pub fn push_fec(&mut self, packet: &[u8]) {
        self.inner.push_fec(packet);
    }

    /// Take the media packets recovered so far (to inject into the jitter buffer).
    pub fn take_recovered(&mut self) -> Vec<Vec<u8>> {
        self.inner.take_recovered()
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
        assert_eq!(
            recovered, media[lost_idx],
            "FEC reconstructs the lost packet byte-exact"
        );
    }

    #[test]
    fn two_losses_in_a_group_cannot_be_recovered() {
        let media = media_run(4);
        let refs: Vec<&[u8]> = media.iter().map(|v| v.as_slice()).collect();
        let fec = build_fec_packet(&refs, 97, 0, 0).unwrap();
        // Only two survivors -> two missing -> single-FEC cannot recover.
        let present: Vec<(u16, &[u8])> = media[..2]
            .iter()
            .map(|p| (u16::from_be_bytes([p[2], p[3]]), p.as_slice()))
            .collect();
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

    #[test]
    fn interleaved_fec_recovers_a_burst_single_level_cannot() {
        // 4 rows x 4 columns = a 16-packet block; column repairs recover a burst
        // of up to 4 consecutive losses (one per column).
        let mut enc = InterleavedFecEncoder::new(4, 4, 97, 0xFEC0_0000);
        assert_eq!(enc.block(), 16);
        let media = media_run(16);

        let mut columns = Vec::new();
        for p in &media {
            columns.extend(enc.push(p));
        }
        assert_eq!(columns.len(), 4, "one repair packet per column");

        // Lose a burst of 4 consecutive media packets (indices 5..=8): more than
        // one loss in any single contiguous group, so single-level FEC is stuck,
        // but the burst spans 4 distinct columns, one loss each.
        let burst = [5usize, 6, 7, 8];
        let mut dec = FecDecoder::new(64);
        for (i, p) in media.iter().enumerate() {
            if !burst.contains(&i) {
                let seq = u16::from_be_bytes([p[2], p[3]]);
                dec.push_media(seq, p);
            }
        }
        for col in &columns {
            dec.push_fec(col);
        }

        let mut recovered = dec.take_recovered();
        recovered.sort_by_key(|p| u16::from_be_bytes([p[2], p[3]]));
        assert_eq!(recovered.len(), 4, "every packet of the burst recovered");
        for &i in &burst {
            assert!(
                recovered.iter().any(|r| *r == media[i]),
                "burst packet {i} reconstructed byte-exact",
            );
        }
    }

    #[test]
    fn build_fec_packet_rejects_a_span_wider_than_16() {
        // Two packets 20 sequence numbers apart cannot share one level-0 mask.
        let mut pkt = RtpH264Packetizer::new(96, 7);
        let a = pkt.packetize(&[0, 0, 0, 1, 0x61, 1], 0).remove(0);
        // Force a far sequence by packetizing many throwaway packets.
        for _ in 0..19 {
            pkt.packetize(&[0, 0, 0, 1, 0x61, 2], 90);
        }
        let far = pkt.packetize(&[0, 0, 0, 1, 0x61, 3], 180).remove(0);
        assert!(
            build_fec_packet(&[a.as_slice(), far.as_slice()], 97, 0, 0).is_none(),
            "a > 16 sequence span does not fit the 16-bit mask",
        );
    }
}
