//! SMPTE ST 2022-1 (Pro-MPEG Code of Practice #3) FEC for RTP. Sans-IO,
//! `no_std + alloc`.
//!
//! The loss-recovery layer professional MPEG-TS contribution links expect, and
//! the one lossy datalinks carrying STANAG 4609 video are usually protected
//! with. The recovery math is the packet XOR of [`crate::ulpfec`] (both descend
//! from RFC 2733); ST 2022-1 differs in the wire format and the stream
//! structure:
//!
//! - Repairs ride **separate RTP sessions**, not a distinct payload type in the
//!   media session: with the media on port `p`, column FEC goes to `p + 2` and
//!   row FEC to `p + 4`, each with its own sequence number space.
//! - The group is described by a **block descriptor** (`SNBase`, `offset`,
//!   `NA`) rather than a bitmask: the protected sequences are
//!   `SNBase + i * offset` for `i` in `0..NA`.
//! - Protection is 2-D over an `L x D` block fed in sequence order. A *column*
//!   repair covers `D` packets at stride `L` (`offset = L`, `NA = D`), so a
//!   burst of up to `L` consecutive losses lands in `L` distinct columns and is
//!   fully recovered. A *row* repair covers `L` consecutive packets
//!   (`offset = 1`, `NA = L`), which costs less but only recovers isolated
//!   losses; running both lets the decoder chain recoveries.
//!
//! ## FEC header
//!
//! 16 bytes at the start of the repair packet's RTP payload. Verified against
//! FFmpeg's `libavformat/prompeg.c` (`prompeg_write_fec`, byte offsets 12..28
//! of the datagram) and GStreamer's `gst/rtpmanager/gstrtpst2022-1-fecenc.c`
//! (`queue_fec_packet`); the two agree field for field.
//!
//! ```text
//!  0                   1                   2                   3
//!  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |            SNBase low bits    |        Length recovery        |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |E|  PT recovery|                  Mask (0)                     |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |                          TS recovery                          |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |N|D|type |indx |     offset    |       NA      |SNBase ext bits|
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! ```
//!
//! `E = 1`, `N = 0`, `type = 0` (XOR), `index = 0`, `mask = 0`, `SNBase ext =
//! 0`; `D = 1` marks a row repair, `D = 0` a column repair. The remaining
//! recovery fields do not all live in the FEC header: the XORed P/X/CC bits go
//! in the repair's own RTP byte 0 and the XORed marker bit in its byte 1, so the
//! repair packet's header doubles as recovery state (`buf[0] = 0x80 | (b[0] &
//! 0x3f)`, `buf[1] = (b[1] & 0x80) | PT` in `prompeg_write_fec`).
//!
//! [`St2022FecEncoder`] emits the repair streams for a media stream;
//! [`St2022FecDecoder`] buffers media plus repairs and reconstructs the losses.
//! Interop with a real ST 2022-1 peer is unverified here (sandbox); the header
//! is byte-asserted against the two implementations above.

use alloc::vec::Vec;

use crate::ulpfec::{split_group, XorFecBuffer, XorFields, RTP_HEADER};

/// ST 2022-1 FEC header length, at the start of the repair packet's payload.
pub const FEC_HEADER: usize = 16;
/// The payload type FFmpeg sends ST 2022-1 repairs on, and GStreamer's default.
pub const DEFAULT_PT: u8 = 96;

/// Which of the two repair streams a packet belongs to. They are separate RTP
/// sessions with independent sequence numbers, so a sender has to keep them
/// apart on the way out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FecStream {
    /// `D` packets at stride `L`: recovers a burst of up to `L` losses.
    Column,
    /// `L` consecutive packets: cheaper, recovers isolated losses.
    Row,
}

/// A repair packet and the stream it belongs on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Repair {
    pub stream: FecStream,
    pub packet: Vec<u8>,
}

/// A parsed ST 2022-1 FEC header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FecHeader {
    /// Low 16 bits of the first protected sequence number.
    pub sn_base: u16,
    /// XOR of the protected payload lengths.
    pub len_recovery: u16,
    /// Set when the header is followed by the protected payload (always, here).
    pub e: bool,
    /// XOR of the protected payload types.
    pub pt_recovery: u8,
    /// 24-bit mask, unused by ST 2022-1 (RFC 2733 heritage).
    pub mask: u32,
    /// XOR of the protected RTP timestamps.
    pub ts_recovery: u32,
    pub n: bool,
    /// `true` = row FEC, `false` = column FEC.
    pub d: bool,
    /// Recovery algorithm; only `0` (XOR) is defined.
    pub fec_type: u8,
    pub index: u8,
    /// Sequence stride between protected packets.
    pub offset: u8,
    /// Number of protected packets.
    pub na: u8,
    /// High bits of a 24-bit `SNBase`, ignored for 16-bit RTP sequences.
    pub sn_base_ext: u8,
}

impl FecHeader {
    /// Parse the 16-byte header from a repair packet's RTP payload.
    pub fn parse(payload: &[u8]) -> Option<Self> {
        let h: &[u8; FEC_HEADER] = payload.get(..FEC_HEADER)?.try_into().ok()?;
        Some(Self {
            sn_base: u16::from_be_bytes([h[0], h[1]]),
            len_recovery: u16::from_be_bytes([h[2], h[3]]),
            e: h[4] & 0x80 != 0,
            pt_recovery: h[4] & 0x7F,
            mask: u32::from_be_bytes([0, h[5], h[6], h[7]]),
            ts_recovery: u32::from_be_bytes([h[8], h[9], h[10], h[11]]),
            n: h[12] & 0x80 != 0,
            d: h[12] & 0x40 != 0,
            fec_type: (h[12] >> 3) & 0x07,
            index: h[12] & 0x07,
            offset: h[13],
            na: h[14],
            sn_base_ext: h[15],
        })
    }

    /// The sequence numbers this repair protects: `SNBase + i * offset`. `None`
    /// for a descriptor that names no packets or a recovery algorithm other than
    /// XOR, both of which would otherwise produce garbage.
    pub fn protected_seqs(&self) -> Option<Vec<u16>> {
        if self.na == 0 || self.offset == 0 || self.fec_type != 0 {
            return None;
        }
        let mut seqs = Vec::with_capacity(self.na as usize);
        for i in 0..self.na as u16 {
            seqs.push(
                self.sn_base
                    .wrapping_add(i.wrapping_mul(self.offset as u16)),
            );
        }
        Some(seqs)
    }
}

/// Split a repair RTP packet into its FEC header and protected payload.
fn parse_repair(fec: &[u8]) -> Option<(FecHeader, &[u8])> {
    let payload = fec.get(RTP_HEADER..)?;
    let header = FecHeader::parse(payload)?;
    Some((header, &payload[FEC_HEADER..]))
}

/// The sequence numbers a repair packet protects.
pub fn protected_seqs(fec: &[u8]) -> Option<Vec<u16>> {
    parse_repair(fec)?.0.protected_seqs()
}

/// Build one repair packet over `media`, RTP packets in sequence order forming
/// the arithmetic progression the block descriptor names: consecutive for
/// [`FecStream::Row`], stride `l` for [`FecStream::Column`]. The repair is an
/// RTP packet on `pt` / `ssrc` / `seq`, carrying `ts` (senders use the last
/// media timestamp, as GStreamer does).
///
/// `None` if `media` is empty, longer than the 8-bit `NA` field, contains a
/// packet shorter than an RTP header, or is not that progression, since a
/// receiver derives the protected set from `SNBase` / `offset` / `NA` alone and
/// a repair whose members disagree would recover the wrong sequence number.
pub fn build_repair_packet(
    media: &[&[u8]],
    stream: FecStream,
    l: u8,
    pt: u8,
    ssrc: u32,
    seq: u16,
    ts: u32,
) -> Option<Vec<u8>> {
    if media.is_empty() || media.len() > u8::MAX as usize {
        return None;
    }
    let offset = match stream {
        FecStream::Row => 1u8,
        FecStream::Column => l,
    };
    if offset == 0 {
        return None;
    }
    let sn_base = u16::from_be_bytes([media[0].get(2).copied()?, media[0].get(3).copied()?]);
    for (i, p) in media.iter().enumerate() {
        let s = u16::from_be_bytes([*p.get(2)?, *p.get(3)?]);
        let want = sn_base.wrapping_add((i as u16).wrapping_mul(offset as u16));
        if s != want {
            return None;
        }
    }
    let f = XorFields::fold(media)?;

    let mut out = Vec::with_capacity(RTP_HEADER + FEC_HEADER + f.payload.len());
    // The repair's own RTP header also carries the XORed P/X/CC and marker bits.
    out.push(0x80 | (f.pxcc & 0x3F));
    out.push((f.mpt & 0x80) | (pt & 0x7F));
    out.extend_from_slice(&seq.to_be_bytes());
    out.extend_from_slice(&ts.to_be_bytes());
    out.extend_from_slice(&ssrc.to_be_bytes());
    // FEC header.
    out.extend_from_slice(&sn_base.to_be_bytes());
    out.extend_from_slice(&f.len.to_be_bytes());
    out.push(0x80 | (f.mpt & 0x7F)); // E=1, PT recovery
    out.extend_from_slice(&[0u8; 3]); // mask
    out.extend_from_slice(&f.ts.to_be_bytes());
    // N=0, D, type=0 (XOR), index=0
    out.push(match stream {
        FecStream::Row => 0x40,
        FecStream::Column => 0x00,
    });
    out.push(offset);
    out.push(media.len() as u8); // NA
    out.push(0); // SNBase ext bits
    out.extend_from_slice(&f.payload);
    Some(out)
}

/// Recover the one missing media packet of a repair packet's group, given the
/// surviving members `present` (`(seq, packet)`). `None` unless exactly one of
/// the protected sequences is absent, so a group with two losses reports no
/// recovery rather than a wrongly XORed packet.
pub fn recover_packet(fec: &[u8], present: &[(u16, &[u8])]) -> Option<Vec<u8>> {
    let (h, fec_payload) = parse_repair(fec)?;
    let seqs = h.protected_seqs()?;
    let (missing_seq, group) = split_group(&seqs, present)?;
    if group.is_empty() {
        return None; // the media SSRC only comes from a survivor
    }

    // The marker bit's recovery rides the repair's own RTP header, the payload
    // type's the FEC header; together they rebuild the media byte 1.
    let mpt_r = (fec[1] & 0x80) | h.pt_recovery;
    let mut f = XorFields::from_repair(
        fec[0],
        mpt_r,
        h.ts_recovery,
        h.len_recovery,
        fec_payload.to_vec(),
    );
    for p in &group {
        f.xor_in(p)?;
    }
    f.to_rtp(missing_seq)
}

/// Emits the ST 2022-1 repair streams for a media RTP stream fed in sequence
/// order, over an `L x D` block: `L` column repairs per block, and optionally a
/// row repair per row ([`with_row_fec`](Self::with_row_fec)).
///
/// Column repairs are emitted together when the block closes. A real sender
/// staggers them across the following block to smooth the bitrate, which is a
/// pacing choice for the network sink, not a wire-format one.
#[derive(Debug)]
pub struct St2022FecEncoder {
    l: usize,
    d: usize,
    columns: bool,
    rows: bool,
    pt: u8,
    ssrc: u32,
    column_seq: u16,
    row_seq: u16,
    /// The media packets of the current block, row-major, at most `l * d`.
    pending: Vec<Vec<u8>>,
}

impl St2022FecEncoder {
    /// `l` columns by `d` rows. ST 2022-1 profiles use `1 <= L <= 20` with
    /// `L * D <= 100`; the wire fields (`offset`, `NA`) cap both at 255.
    pub fn new(l: usize, d: usize, pt: u8, ssrc: u32) -> Self {
        Self {
            l: l.clamp(1, u8::MAX as usize),
            d: d.clamp(1, u8::MAX as usize),
            columns: true,
            rows: false,
            pt: pt & 0x7F,
            ssrc,
            column_seq: 0,
            row_seq: 0,
            pending: Vec::new(),
        }
    }

    /// Also emit the row repair stream (off by default: columns alone recover
    /// bursts, rows add a second dimension at another `1/D` of the bitrate).
    pub fn with_row_fec(mut self, on: bool) -> Self {
        self.rows = on;
        self
    }

    /// Emit the column repair stream (on by default).
    pub fn with_column_fec(mut self, on: bool) -> Self {
        self.columns = on;
        self
    }

    /// The block size `L * D`, the number of media packets per set of column
    /// repairs.
    pub fn block(&self) -> usize {
        self.l * self.d
    }

    /// Feed a media RTP packet; returns the repairs that became due: a row
    /// repair whenever a row of `L` closes, plus the `L` column repairs when the
    /// block completes. Empty otherwise, and empty for a group the descriptor
    /// cannot describe (a malformed packet, or a gap in the media sequence),
    /// which protects nothing rather than emitting a repair a receiver would
    /// misapply.
    pub fn push(&mut self, media: &[u8]) -> Vec<Repair> {
        self.pending.push(media.to_vec());
        let ts = media
            .get(4..8)
            .map(|t| u32::from_be_bytes([t[0], t[1], t[2], t[3]]))
            .unwrap_or(0);
        let mut out = Vec::new();

        if self.rows && self.pending.len().is_multiple_of(self.l) {
            let row: Vec<&[u8]> = self.pending[self.pending.len() - self.l..]
                .iter()
                .map(|v| v.as_slice())
                .collect();
            if let Some(p) = build_repair_packet(
                &row,
                FecStream::Row,
                self.l as u8,
                self.pt,
                self.ssrc,
                self.row_seq,
                ts,
            ) {
                self.row_seq = self.row_seq.wrapping_add(1);
                out.push(Repair {
                    stream: FecStream::Row,
                    packet: p,
                });
            }
        }

        if self.pending.len() >= self.block() {
            if self.columns {
                for c in 0..self.l {
                    let column: Vec<&[u8]> = self
                        .pending
                        .iter()
                        .skip(c)
                        .step_by(self.l)
                        .map(|v| v.as_slice())
                        .collect();
                    if let Some(p) = build_repair_packet(
                        &column,
                        FecStream::Column,
                        self.l as u8,
                        self.pt,
                        self.ssrc,
                        self.column_seq,
                        ts,
                    ) {
                        self.column_seq = self.column_seq.wrapping_add(1);
                        out.push(Repair {
                            stream: FecStream::Column,
                            packet: p,
                        });
                    }
                }
            }
            self.pending.clear();
        }
        out
    }
}

/// Buffers received media + repair packets from both repair streams and
/// reconstructs the losses, chaining recoveries so row and column repairs
/// complete each other.
#[derive(Debug)]
pub struct St2022FecDecoder {
    inner: XorFecBuffer,
}

impl Default for St2022FecDecoder {
    fn default() -> Self {
        Self::new(256)
    }
}

impl St2022FecDecoder {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: XorFecBuffer::new(capacity, protected_seqs, recover_packet),
        }
    }

    /// Record a received media packet and attempt recovery.
    pub fn push_media(&mut self, seq: u16, packet: &[u8]) {
        self.inner.push_media(seq, packet);
    }

    /// Record a received repair packet, from either repair stream, and attempt
    /// recovery.
    pub fn push_fec(&mut self, packet: &[u8]) {
        self.inner.push_fec(packet);
    }

    /// Take the media packets recovered so far (to inject into the jitter
    /// buffer).
    pub fn take_recovered(&mut self) -> Vec<Vec<u8>> {
        self.inner.take_recovered()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// An MPEG-TS-over-RTP media packet (PT 33) with a recognisable payload.
    fn media(seq: u16, ts: u32, fill: u8, len: usize) -> Vec<u8> {
        let mut p = Vec::with_capacity(RTP_HEADER + len);
        p.push(0x80);
        p.push(33);
        p.extend_from_slice(&seq.to_be_bytes());
        p.extend_from_slice(&ts.to_be_bytes());
        p.extend_from_slice(&0xABCD_0001u32.to_be_bytes());
        p.extend((0..len).map(|i| fill.wrapping_add(i as u8)));
        p
    }

    #[test]
    fn column_header_matches_ffmpeg_and_gstreamer_byte_for_byte() {
        // Offsets and values from FFmpeg libavformat/prompeg.c prompeg_write_fec
        // (datagram bytes 12..28) and GStreamer gstrtpst2022-1-fecenc.c
        // queue_fec_packet, which write identical fields.
        let l = 5u8;
        let seqs = [100u16, 105, 110];
        let tss = [0x0000_00FFu32, 0x0000_0F00, 0x000F_0000];
        let lens = [8usize, 10, 12];
        let group: Vec<Vec<u8>> = (0..3)
            .map(|i| media(seqs[i], tss[i], i as u8 * 16, lens[i]))
            .collect();
        let refs: Vec<&[u8]> = group.iter().map(|v| v.as_slice()).collect();
        let fec =
            build_repair_packet(&refs, FecStream::Column, l, 96, 0, 7, 0x1234_5678).expect("built");

        // Repair RTP header: V=2, no P/X/CC (no media packet has any), M=0, PT=96.
        assert_eq!(&fec[..2], &[0x80, 96]);
        assert_eq!(&fec[2..4], &7u16.to_be_bytes(), "own sequence number");
        assert_eq!(&fec[4..8], &0x1234_5678u32.to_be_bytes(), "own timestamp");

        let h = &fec[RTP_HEADER..RTP_HEADER + FEC_HEADER];
        assert_eq!(&h[0..2], &100u16.to_be_bytes(), "SNBase low bits");
        assert_eq!(&h[2..4], &(8u16 ^ 10 ^ 12).to_be_bytes(), "length recovery");
        assert_eq!(h[4], 0x80 | 33, "E=1, PT recovery = 33^33^33");
        assert_eq!(&h[5..8], &[0, 0, 0], "mask is unused");
        let ts_r = tss.iter().fold(0u32, |a, b| a ^ b);
        assert_eq!(&h[8..12], &ts_r.to_be_bytes(), "TS recovery");
        assert_eq!(h[12], 0x00, "N=0, D=0 (column), type=0 (XOR), index=0");
        assert_eq!(h[13], l, "offset = L for a column");
        assert_eq!(h[14], 3, "NA = the number of protected packets");
        assert_eq!(h[15], 0, "SNBase ext bits");
        assert_eq!(
            fec.len(),
            RTP_HEADER + FEC_HEADER + 12,
            "the protected payload is the longest member, shorter ones zero-padded",
        );
    }

    #[test]
    fn pt_recovery_is_the_xor_of_the_payload_types() {
        // An odd-sized group so the XOR of a constant PT does not cancel out.
        let group: Vec<Vec<u8>> = (0..3).map(|i| media(10 + i, 0, i as u8, 4)).collect();
        let refs: Vec<&[u8]> = group.iter().map(|v| v.as_slice()).collect();
        let fec = build_repair_packet(&refs, FecStream::Row, 3, 96, 0, 0, 0).unwrap();
        let h = FecHeader::parse(&fec[RTP_HEADER..]).expect("parsed");
        let pt_r = group.iter().fold(0u8, |a, p| a ^ (p[1] & 0x7F));
        assert_eq!(h.pt_recovery, pt_r, "XOR of the media payload types");
        assert_eq!(pt_r, 33, "an odd group does not cancel a constant PT");
        assert!(h.e, "E is set: the payload follows the header");
        assert!(h.d, "D=1 marks the row stream");
        assert_eq!(h.offset, 1, "row repairs protect consecutive packets");
        assert_eq!(h.na, 3);
        assert_eq!(h.len_recovery, 4, "XOR of three 4-byte payload lengths");
    }

    #[test]
    fn protected_seqs_walk_the_block_descriptor() {
        let group: Vec<Vec<u8>> = (0..4).map(|i| media(1000 + i * 5, 0, i as u8, 6)).collect();
        let refs: Vec<&[u8]> = group.iter().map(|v| v.as_slice()).collect();
        let fec = build_repair_packet(&refs, FecStream::Column, 5, 96, 0, 0, 0).unwrap();
        assert_eq!(
            protected_seqs(&fec).unwrap(),
            vec![1000u16, 1005, 1010, 1015],
        );
    }

    #[test]
    fn build_rejects_a_group_that_is_not_the_descriptors_progression() {
        // A gap in the media sequence: the receiver would derive 20, 25, 30 and
        // recover the wrong sequence number, so no repair is produced.
        let group = [media(20, 0, 0, 4), media(25, 0, 1, 4), media(31, 0, 2, 4)];
        let refs: Vec<&[u8]> = group.iter().map(|v| v.as_slice()).collect();
        assert!(build_repair_packet(&refs, FecStream::Column, 5, 96, 0, 0, 0).is_none());
        // Empty and over-long groups do not fit the descriptor either.
        assert!(build_repair_packet(&[], FecStream::Column, 5, 96, 0, 0, 0).is_none());
        let many: Vec<Vec<u8>> = (0..256).map(|i| media(i as u16, 0, 0, 4)).collect();
        let refs: Vec<&[u8]> = many.iter().map(|v| v.as_slice()).collect();
        assert!(build_repair_packet(&refs, FecStream::Row, 1, 96, 0, 0, 0).is_none());
    }

    #[test]
    fn a_non_xor_recovery_type_is_refused() {
        let group: Vec<Vec<u8>> = (0..3).map(|i| media(i, 0, i as u8, 4)).collect();
        let refs: Vec<&[u8]> = group.iter().map(|v| v.as_slice()).collect();
        let mut fec = build_repair_packet(&refs, FecStream::Row, 3, 96, 0, 0, 0).unwrap();
        fec[RTP_HEADER + 12] |= 0x08; // type = 1, an algorithm we do not implement
        assert!(protected_seqs(&fec).is_none());
        assert!(recover_packet(&fec, &[]).is_none());
    }

    #[test]
    fn truncated_and_corrupt_repairs_never_panic() {
        let group: Vec<Vec<u8>> = (0..4)
            .map(|i| media(i, 90 * i as u32, i as u8, 12))
            .collect();
        let refs: Vec<&[u8]> = group.iter().map(|v| v.as_slice()).collect();
        let fec = build_repair_packet(&refs, FecStream::Row, 4, 96, 0, 0, 0).unwrap();
        let present: Vec<(u16, &[u8])> = group[1..]
            .iter()
            .enumerate()
            .map(|(i, p)| (i as u16 + 1, p.as_slice()))
            .collect();

        // Every truncation of a valid repair, and every single-byte corruption
        // of its header, must be rejected or handled, not panic.
        for n in 0..fec.len() {
            let _ = protected_seqs(&fec[..n]);
            let _ = recover_packet(&fec[..n], &present);
        }
        for i in 0..RTP_HEADER + FEC_HEADER {
            for bit in [0x01u8, 0x40, 0x80, 0xFF] {
                let mut bad = fec.clone();
                bad[i] ^= bit;
                let _ = protected_seqs(&bad);
                let _ = recover_packet(&bad, &present);
            }
        }
    }

    #[test]
    fn a_bogus_length_recovery_allocates_nothing() {
        let group: Vec<Vec<u8>> = (0..3).map(|i| media(i, 0, i as u8, 8)).collect();
        let refs: Vec<&[u8]> = group.iter().map(|v| v.as_slice()).collect();
        let mut fec = build_repair_packet(&refs, FecStream::Row, 3, 96, 0, 0, 0).unwrap();
        // Claim a 64 KiB recovered payload behind an 8-byte protected payload.
        fec[RTP_HEADER + 2] = 0xFF;
        fec[RTP_HEADER + 3] = 0xFF;
        let present: Vec<(u16, &[u8])> = group[1..]
            .iter()
            .enumerate()
            .map(|(i, p)| (i as u16 + 1, p.as_slice()))
            .collect();
        assert!(
            recover_packet(&fec, &present).is_none(),
            "a length recovery past the XORed payload is inconsistent, not a 64 KiB packet",
        );
    }
}
