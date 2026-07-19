//! ST 2110-20 uncompressed video over RTP (M599): the packetizer and
//! depacketizer for uncompressed active video, RFC 4175.
//!
//! Unlike -30 audio (raw PCM straight in the payload), -20 video carries a small
//! per-packet media header: an Extended Sequence Number then one or more Sample
//! Row Data (SRD) line headers, each giving the scan line, pixel offset, and octet
//! length of a run of pixel data, so a packet can carry a partial line or several
//! partial lines and a receiver writes each run to the right place in the frame.
//! The RTP marker bit marks the last packet of a frame; every packet of one frame
//! shares that frame's [`MediaClock`] (90 kHz) timestamp, so a receiver on the
//! same grandmaster reconstructs the frame's PTP sampling instant, the video half
//! of A/V sync across devices.
//!
//! Sans-IO core (a frame buffer <-> RTP packets), like `st2110audio`: pure
//! `no_std` + alloc, CI round-trip testable, an element wrapper (`st2110videortp`)
//! sits on top. Three RFC 4175 samplings so far, mapped to the [`RawVideoFormat`]s
//! a pipeline already carries:
//!
//! - RGBA 8-bit (pgroup = 1 pixel / 4 octets, order R G B A), packed byte-for-byte.
//! - YCbCr-4:2:2 8-bit (pgroup = 2 pixels / 4 octets, wire order Cb0 Y0 Cr0 Y1),
//!   from the packed `Yuyv` buffer (Y0 Cb Y1 Cr), so the luma / chroma bytes swap.
//! - YCbCr-4:2:2 10-bit (pgroup = 2 pixels / 5 octets, the broadcast norm), from
//!   the planar `I422p10` buffer (separate Y / Cb / Cr planes of 16-bit little
//!   endian samples in the low 10 bits): the four 10-bit samples Cb0 Y0 Cr0 Y1 are
//!   MSB-first bit-packed into 5 octets, so this crosses both a planar-to-packed
//!   and a byte-to-bit boundary.
//!
//! Each sampling is a [`Layout`] that reads / writes one pgroup at a time, so the
//! packetizer / depacketizer are layout-agnostic (a per-pgroup gather / scatter,
//! not a contiguous copy; -20 is bandwidth-bound regardless).
//!
//! **Never trust the stream:** the SRD line / offset / length are attacker
//! controlled, so a run that names a line past the height, an offset off the
//! pgroup grid, or a length overrunning the row or the datagram is rejected (the
//! packet is dropped) rather than writing out of bounds.

use alloc::vec::Vec;

use g2g_core::rtp::{RtpHeader, RTP_HEADER_LEN};
use g2g_core::{MediaClock, RawVideoFormat};

/// RFC 4175 payload header before the SRD list: the Extended Sequence Number.
const EXT_SEQ_LEN: usize = 2;
/// One SRD line header: Length(16) + F|Line(16) + C|Offset(16).
const SRD_HEADER_LEN: usize = 6;
/// Largest pgroup we handle, in octets (RGBA 16-bit would be 8); sizes a scratch.
const MAX_PGROUP: usize = 8;

/// An RFC 4175 sampling, tying a [`RawVideoFormat`] to a wire pixel group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sampling {
    /// RGBA 8-bit: pgroup = 1 pixel / 4 octets, order R G B A ([`RawVideoFormat::Rgba8`]).
    Rgba8,
    /// YCbCr-4:2:2 8-bit: pgroup = 2 pixels / 4 octets, wire order Cb0 Y0 Cr0 Y1,
    /// from the packed `Yuyv` buffer (Y0 Cb Y1 Cr): the pgroup bytes are reordered.
    YCbCr422_8,
    /// YCbCr-4:2:2 10-bit: pgroup = 2 pixels / 5 octets, from the planar `I422p10`
    /// buffer: the four 10-bit samples Cb0 Y0 Cr0 Y1 are MSB-first bit-packed.
    YCbCr422_10,
}

impl Sampling {
    /// The RFC 4175 sampling for a [`RawVideoFormat`], or `None` for a format not
    /// (yet) mapped to a -20 sampling.
    pub fn from_format(f: RawVideoFormat) -> Option<Self> {
        match f {
            RawVideoFormat::Rgba8 => Some(Sampling::Rgba8),
            RawVideoFormat::Yuyv => Some(Sampling::YCbCr422_8),
            RawVideoFormat::I422p10 => Some(Sampling::YCbCr422_10),
            _ => None,
        }
    }

    /// Pixels per pgroup.
    pub const fn pixels_per_group(self) -> usize {
        match self {
            Sampling::Rgba8 => 1,
            Sampling::YCbCr422_8 | Sampling::YCbCr422_10 => 2,
        }
    }

    /// Octets per pgroup on the wire.
    pub const fn octets_per_group(self) -> usize {
        match self {
            Sampling::Rgba8 | Sampling::YCbCr422_8 => 4,
            Sampling::YCbCr422_10 => 5,
        }
    }

    /// The packed / planar [`RawVideoFormat`] this sampling maps to (the inverse of
    /// [`Self::from_format`]).
    pub const fn raw_format(self) -> RawVideoFormat {
        match self {
            Sampling::Rgba8 => RawVideoFormat::Rgba8,
            Sampling::YCbCr422_8 => RawVideoFormat::Yuyv,
            Sampling::YCbCr422_10 => RawVideoFormat::I422p10,
        }
    }
}

/// The buffer layout for a sampling at a fixed geometry: knows the total frame
/// size and how to read / write one pgroup (gather a wire pgroup from the buffer
/// on send, scatter it back on receive). This is the seam that keeps the
/// packetizer / depacketizer agnostic to packed vs planar and byte vs bit packing.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    sampling: Sampling,
    width: usize,
    height: usize,
}

/// Read a little-endian `u16` at `off`, masked to the low 10 bits.
fn read_le10(frame: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([frame[off], frame[off + 1]]) & 0x03FF
}

/// Write the low 10 bits of `v` as a little-endian `u16` at `off`.
fn write_le10(frame: &mut [u8], off: usize, v: u16) {
    frame[off..off + 2].copy_from_slice(&(v & 0x03FF).to_le_bytes());
}

impl Layout {
    /// A layout for `sampling` at `width` x `height`, or `None` if the width is not
    /// a whole number of pgroups (e.g. an odd width for 4:2:2) or the height is 0.
    pub fn new(sampling: Sampling, width: usize, height: usize) -> Option<Self> {
        if width == 0 || height == 0 || width % sampling.pixels_per_group() != 0 {
            return None;
        }
        Some(Self {
            sampling,
            width,
            height,
        })
    }

    /// pgroups per scan line.
    pub fn groups_per_line(&self) -> usize {
        self.width / self.sampling.pixels_per_group()
    }

    /// Total packed buffer size in octets.
    pub fn frame_size(&self) -> usize {
        match self.sampling {
            // Packed: one stride, wire octets per line == buffer octets per line.
            Sampling::Rgba8 | Sampling::YCbCr422_8 => {
                self.groups_per_line() * self.sampling.octets_per_group() * self.height
            }
            // Planar I422p10: Y (w*h) + Cb (w/2*h) + Cr (w/2*h) samples, 2 bytes each.
            Sampling::YCbCr422_10 => self.width * self.height * 4,
        }
    }

    /// Gather one pgroup at `(line, group)` from the buffer into `out` (must be at
    /// least [`Sampling::octets_per_group`] long) in RFC 4175 wire order.
    fn read_group(&self, frame: &[u8], line: usize, group: usize, out: &mut [u8]) {
        match self.sampling {
            Sampling::Rgba8 => {
                let off = (line * self.groups_per_line() + group) * 4;
                out[..4].copy_from_slice(&frame[off..off + 4]);
            }
            Sampling::YCbCr422_8 => {
                // Buffer Y0 Cb Y1 Cr -> wire Cb Y0 Cr Y1.
                let off = (line * self.groups_per_line() + group) * 4;
                let s = &frame[off..off + 4];
                out[..4].copy_from_slice(&[s[1], s[0], s[3], s[2]]);
            }
            Sampling::YCbCr422_10 => {
                let (cb, y0, cr, y1) = self.read_planar_422(frame, line, group);
                pack_2px_10(cb, y0, cr, y1, out);
            }
        }
    }

    /// Scatter one wire pgroup `src` (RFC 4175 order) into the buffer at `(line, group)`.
    fn write_group(&self, frame: &mut [u8], line: usize, group: usize, src: &[u8]) {
        match self.sampling {
            Sampling::Rgba8 => {
                let off = (line * self.groups_per_line() + group) * 4;
                frame[off..off + 4].copy_from_slice(&src[..4]);
            }
            Sampling::YCbCr422_8 => {
                // Wire Cb Y0 Cr Y1 -> buffer Y0 Cb Y1 Cr.
                let off = (line * self.groups_per_line() + group) * 4;
                frame[off..off + 4].copy_from_slice(&[src[1], src[0], src[3], src[2]]);
            }
            Sampling::YCbCr422_10 => {
                self.write_planar_422(frame, line, group, unpack_2px_10(src));
            }
        }
    }

    /// The three plane byte offsets (Y, Cb, Cr) of the planar I422p10 buffer.
    fn planes_422(&self) -> (usize, usize, usize) {
        let y = self.width * self.height * 2; // Y plane size
        let c = (self.width / 2) * self.height * 2; // each chroma plane size
        (0, y, y + c)
    }

    fn read_planar_422(&self, frame: &[u8], line: usize, group: usize) -> (u16, u16, u16, u16) {
        let (yp, cbp, crp) = self.planes_422();
        let y_row = line * self.width * 2;
        let c_row = line * (self.width / 2) * 2;
        let y0 = read_le10(frame, yp + y_row + (2 * group) * 2);
        let y1 = read_le10(frame, yp + y_row + (2 * group + 1) * 2);
        let cb = read_le10(frame, cbp + c_row + group * 2);
        let cr = read_le10(frame, crp + c_row + group * 2);
        (cb, y0, cr, y1)
    }

    fn write_planar_422(
        &self,
        frame: &mut [u8],
        line: usize,
        group: usize,
        (cb, y0, cr, y1): (u16, u16, u16, u16),
    ) {
        let (yp, cbp, crp) = self.planes_422();
        let y_row = line * self.width * 2;
        let c_row = line * (self.width / 2) * 2;
        write_le10(frame, yp + y_row + (2 * group) * 2, y0);
        write_le10(frame, yp + y_row + (2 * group + 1) * 2, y1);
        write_le10(frame, cbp + c_row + group * 2, cb);
        write_le10(frame, crp + c_row + group * 2, cr);
    }
}

/// MSB-first bit-pack four 10-bit samples (Cb0 Y0 Cr0 Y1) into 5 octets (RFC 4175
/// YCbCr-4:2:2 10-bit pgroup). Inputs are masked to 10 bits.
fn pack_2px_10(cb: u16, y0: u16, cr: u16, y1: u16, out: &mut [u8]) {
    let (cb, y0, cr, y1) = (cb & 0x3FF, y0 & 0x3FF, cr & 0x3FF, y1 & 0x3FF);
    out[0] = (cb >> 2) as u8;
    out[1] = (((cb & 0x3) << 6) | (y0 >> 4)) as u8;
    out[2] = (((y0 & 0xF) << 4) | (cr >> 6)) as u8;
    out[3] = (((cr & 0x3F) << 2) | (y1 >> 8)) as u8;
    out[4] = (y1 & 0xFF) as u8;
}

/// The inverse of [`pack_2px_10`]: unpack 5 octets into (Cb0, Y0, Cr0, Y1), each
/// in the low 10 bits.
fn unpack_2px_10(b: &[u8]) -> (u16, u16, u16, u16) {
    let cb = (u16::from(b[0]) << 2) | (u16::from(b[1]) >> 6);
    let y0 = ((u16::from(b[1]) & 0x3F) << 4) | (u16::from(b[2]) >> 4);
    let cr = ((u16::from(b[2]) & 0x0F) << 6) | (u16::from(b[3]) >> 2);
    let y1 = ((u16::from(b[3]) & 0x03) << 8) | u16::from(b[4]);
    (cb, y0, cr, y1)
}

/// Packetizes an uncompressed video frame into ST 2110-20 (RFC 4175) RTP packets.
#[derive(Debug)]
pub struct St2110VideoPacketizer {
    payload_type: u8,
    ssrc: u32,
    sequence: u32,
    sampling: Sampling,
    clock: MediaClock,
    /// Maximum RTP packet size in octets (header + payload); the frame is sliced
    /// into SRD runs to fit it.
    max_packet: usize,
}

impl St2110VideoPacketizer {
    /// A packetizer for `sampling`, capping each RTP packet at `max_packet` octets
    /// (a typical 1500-octet MTU leaves ~1460 after IP/UDP; pass that). `payload_type`
    /// is the dynamic RTP PT, `ssrc` the stream source id.
    pub fn new(payload_type: u8, ssrc: u32, sampling: Sampling, max_packet: usize) -> Self {
        // Floor the cap at a header plus one pgroup, else no data ever fits.
        let floor = RTP_HEADER_LEN + EXT_SEQ_LEN + SRD_HEADER_LEN + sampling.octets_per_group();
        Self {
            payload_type: payload_type & 0x7f,
            ssrc,
            sequence: 0,
            sampling,
            clock: MediaClock::video(),
            max_packet: max_packet.max(floor),
        }
    }

    /// The media clock (for recovering a frame's PTP time on the receive side).
    pub fn media_clock(&self) -> MediaClock {
        self.clock
    }

    /// Packetize one `width` x `height` frame (tightly packed, no row padding) at
    /// PTP/TAI sampling time `tai_ns`. Every packet shares the frame's 90 kHz RTP
    /// timestamp; the last carries the marker bit. Returns `None` if the width is
    /// not a whole number of pgroups or the buffer is too small for the geometry.
    pub fn packetize(
        &mut self,
        frame: &[u8],
        width: usize,
        height: usize,
        tai_ns: u64,
    ) -> Option<Vec<Vec<u8>>> {
        let opg = self.sampling.octets_per_group();
        let ppg = self.sampling.pixels_per_group();
        let layout = Layout::new(self.sampling, width, height)?;
        let groups_per_line = layout.groups_per_line();
        if frame.len() < layout.frame_size() {
            return None;
        }
        let rtp_ts = self.clock.rtp_timestamp(g2g_core::TaiNs(tai_ns)).get();

        let mut packets = Vec::new();
        let mut scratch = [0u8; MAX_PGROUP];
        // Cursor over the frame in pgroups, scan line then group within the line.
        let mut line = 0usize;
        let mut group = 0usize;
        while line < height {
            // Pack SRD runs into one packet up to the size cap.
            let mut srds: Vec<(usize, usize, usize)> = Vec::new(); // (line, offset_px, len_octets)
            let mut data: Vec<u8> = Vec::new();
            let mut avail = self.max_packet - RTP_HEADER_LEN - EXT_SEQ_LEN;
            while line < height {
                // Each new run costs an SRD header; need room for at least one pgroup.
                if avail < SRD_HEADER_LEN + opg {
                    break;
                }
                let by_budget = (avail - SRD_HEADER_LEN) / opg;
                let take = (groups_per_line - group).min(by_budget);
                if take == 0 {
                    break;
                }
                for gi in 0..take {
                    layout.read_group(frame, line, group + gi, &mut scratch[..opg]);
                    data.extend_from_slice(&scratch[..opg]);
                }
                srds.push((line, group * ppg, take * opg));
                avail -= SRD_HEADER_LEN + take * opg;
                group += take;
                if group == groups_per_line {
                    line += 1;
                    group = 0;
                }
            }
            let marker = line >= height;
            packets.push(self.build_packet(rtp_ts, marker, &srds, &data));
            self.sequence = self.sequence.wrapping_add(1);
        }
        Some(packets)
    }

    fn build_packet(
        &self,
        rtp_ts: u32,
        marker: bool,
        srds: &[(usize, usize, usize)],
        data: &[u8],
    ) -> Vec<u8> {
        let mut pkt = Vec::with_capacity(
            RTP_HEADER_LEN + EXT_SEQ_LEN + srds.len() * SRD_HEADER_LEN + data.len(),
        );
        // RTP header (the low 16 sequence bits; the high 16 ride the RFC 4175
        // extended sequence field below).
        let header = RtpHeader {
            payload_type: self.payload_type,
            marker,
            sequence: self.sequence as u16,
            timestamp: rtp_ts,
            ssrc: self.ssrc,
        };
        pkt.extend_from_slice(&header.to_bytes());
        // RFC 4175 payload header: Extended Sequence Number (high 16 bits).
        pkt.extend_from_slice(&((self.sequence >> 16) as u16).to_be_bytes());
        // SRD line headers; the last has C=0, the rest C=1.
        for (i, &(line, offset_px, len)) in srds.iter().enumerate() {
            let c = if i + 1 < srds.len() { 0x8000 } else { 0 };
            pkt.extend_from_slice(&(len as u16).to_be_bytes());
            pkt.extend_from_slice(&((line as u16) & 0x7FFF).to_be_bytes()); // F=0 | Line No
            pkt.extend_from_slice(&(c | (offset_px as u16 & 0x7FFF)).to_be_bytes());
            // C | Offset
        }
        pkt.extend_from_slice(data);
        pkt
    }
}

/// A completed ST 2110-20 frame: its RTP timestamp and the reassembled packed
/// pixel buffer (`RawVideoFormat` byte order, `stride * height` bytes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct St2110VideoFrame {
    pub rtp_timestamp: u32,
    pub bytes: Vec<u8>,
}

/// Reassembles ST 2110-20 (RFC 4175) RTP packets into whole frames. The geometry
/// (format, width, height) comes from the stream description (SDP) out of band,
/// not the payload, so it is fixed at construction.
#[derive(Debug)]
pub struct St2110VideoDepacketizer {
    layout: Layout,
    height: usize,
    groups_per_line: usize,
    /// The frame under reassembly; runs are written in place, the marker bit ends it.
    frame: Vec<u8>,
}

impl St2110VideoDepacketizer {
    /// A depacketizer for a `format` frame of `width` x `height`, or `None` if the
    /// format has no -20 sampling or the width is not a whole number of pgroups.
    pub fn new(format: RawVideoFormat, width: usize, height: usize) -> Option<Self> {
        let layout = Layout::new(Sampling::from_format(format)?, width, height)?;
        Some(Self {
            layout,
            height,
            groups_per_line: layout.groups_per_line(),
            frame: alloc::vec![0u8; layout.frame_size()],
        })
    }

    /// Feed one RTP packet. Writes its SRD runs into the frame under reassembly and,
    /// when the packet carries the marker bit (end of frame), returns the completed
    /// frame. Returns `None` while a frame is still incomplete or if the packet is
    /// malformed (too short, wrong version, or an SRD that would write out of
    /// bounds), in which case the packet is dropped.
    pub fn depacketize(&mut self, packet: &[u8]) -> Option<St2110VideoFrame> {
        if packet.len() < RTP_HEADER_LEN + EXT_SEQ_LEN || packet[0] & 0xC0 != 0x80 {
            return None;
        }
        let marker = packet[1] & 0x80 != 0;
        let rtp_timestamp = u32::from_be_bytes([packet[4], packet[5], packet[6], packet[7]]);

        // Payload: Extended Sequence Number, then the SRD headers, then sample data.
        let payload = &packet[RTP_HEADER_LEN..];
        let mut hi = EXT_SEQ_LEN; // header cursor
        let opg = self.layout.sampling.octets_per_group();
        let ppg = self.layout.sampling.pixels_per_group();

        // Parse the SRD header list (C=1 chains another header).
        let mut srds: Vec<(usize, usize, usize)> = Vec::new(); // (line, offset_px, len)
        loop {
            let hdr = payload.get(hi..hi + SRD_HEADER_LEN)?;
            let len = usize::from(u16::from_be_bytes([hdr[0], hdr[1]]));
            let line = usize::from(u16::from_be_bytes([hdr[2], hdr[3]]) & 0x7FFF);
            let co = u16::from_be_bytes([hdr[4], hdr[5]]);
            let offset_px = usize::from(co & 0x7FFF);
            srds.push((line, offset_px, len));
            hi += SRD_HEADER_LEN;
            if co & 0x8000 == 0 {
                break;
            }
        }

        // Sample data follows the header list; write each run into the frame. The
        // geometry checks (line, offset on the pgroup grid, run within the row) keep
        // every `write_group` in bounds (never trust the stream).
        let mut di = hi;
        for (line, offset_px, len) in srds {
            if line >= self.height || offset_px % ppg != 0 || len % opg != 0 {
                return None;
            }
            let group = offset_px / ppg;
            if group + len / opg > self.groups_per_line {
                return None;
            }
            let run = payload.get(di..di + len)?;
            for (gi, g) in run.chunks_exact(opg).enumerate() {
                self.layout
                    .write_group(&mut self.frame, line, group + gi, g);
            }
            di += len;
        }

        if marker {
            Some(St2110VideoFrame {
                rtp_timestamp,
                bytes: self.frame.clone(),
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic packed frame of `stride*height` bytes with distinct values.
    fn ramp(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 7 + 3) as u8).collect()
    }

    #[test]
    fn rgba_round_trips_across_multiple_packets() {
        // 8x4 RGBA: stride 32, 128 bytes. A small max_packet forces SRD splitting
        // (partial lines and multi-line packets).
        let (w, h) = (8usize, 4usize);
        let frame = ramp(w * 4 * h);
        let tai = 1_700_000_000_000_000_000u64;

        let mut tx =
            St2110VideoPacketizer::new(96, 0xF00D, Sampling::Rgba8, RTP_HEADER_LEN + 2 + 6 + 20);
        let packets = tx.packetize(&frame, w, h, tai).expect("packetizes");
        assert!(packets.len() > 1, "small MTU splits the frame");
        // Only the last packet has the marker bit.
        for (i, p) in packets.iter().enumerate() {
            assert_eq!(
                p[1] & 0x80 != 0,
                i + 1 == packets.len(),
                "marker only on last"
            );
            // Every packet shares the frame's 90 kHz timestamp.
            let ts = u32::from_be_bytes([p[4], p[5], p[6], p[7]]);
            assert_eq!(
                ts,
                MediaClock::video()
                    .rtp_timestamp(g2g_core::TaiNs(tai))
                    .get()
            );
        }

        let mut rx = St2110VideoDepacketizer::new(RawVideoFormat::Rgba8, w, h).unwrap();
        let mut done = None;
        for p in &packets {
            if let Some(f) = rx.depacketize(p) {
                done = Some(f);
            }
        }
        let f = done.expect("frame completes on the marker");
        assert_eq!(
            f.rtp_timestamp,
            MediaClock::video()
                .rtp_timestamp(g2g_core::TaiNs(tai))
                .get()
        );
        assert_eq!(f.bytes, frame, "RGBA pixels survive the round trip");
    }

    #[test]
    fn yuyv_422_reorders_on_the_wire_and_round_trips() {
        // 4x2 YUYV: stride 8 (4 px * 2 bytes), 16 bytes.
        let (w, h) = (4usize, 2usize);
        let frame = ramp(w * 2 * h);
        let mut tx = St2110VideoPacketizer::new(112, 1, Sampling::YCbCr422_8, 1400);
        let packets = tx.packetize(&frame, w, h, 0).expect("packetizes");

        // The wire order is Cb0 Y0 Cr0 Y1 while the buffer is Y0 Cb Y1 Cr, so the
        // first pgroup's first two octets on the wire are swapped vs the buffer. The
        // whole frame is one packet with one SRD per line (2 lines -> 2 headers).
        assert_eq!(packets.len(), 1);
        let first = &packets[0];
        let data_off = RTP_HEADER_LEN + EXT_SEQ_LEN + 2 * SRD_HEADER_LEN;
        assert_eq!(first[data_off], frame[1], "wire Cb0 == buffer byte 1");
        assert_eq!(first[data_off + 1], frame[0], "wire Y0 == buffer byte 0");

        let mut rx = St2110VideoDepacketizer::new(RawVideoFormat::Yuyv, w, h).unwrap();
        let mut done = None;
        for p in &packets {
            if let Some(f) = rx.depacketize(p) {
                done = Some(f);
            }
        }
        assert_eq!(
            done.unwrap().bytes,
            frame,
            "YUYV survives reorder there and back"
        );
    }

    #[test]
    fn ycbcr422_10bit_planar_round_trips() {
        // 4x2 I422p10: Y (4*2), Cb (2*2), Cr (2*2) samples, 2 bytes LE each = 32 B.
        // Fill every 10-bit sample with a distinct value; the round trip crosses
        // planar -> packed and byte -> 10-bit-MSB-first and back.
        let (w, h) = (4usize, 2usize);
        let mut frame = alloc::vec![0u8; w * h * 4];
        for (i, word) in frame.chunks_exact_mut(2).enumerate() {
            let val = ((i * 13 + 7) as u16) & 0x03FF;
            word.copy_from_slice(&val.to_le_bytes());
        }
        assert_eq!(Sampling::YCbCr422_10.octets_per_group(), 5);
        let mut tx = St2110VideoPacketizer::new(96, 1, Sampling::YCbCr422_10, 1400);
        let packets = tx.packetize(&frame, w, h, 0).expect("packetizes");
        let mut rx = St2110VideoDepacketizer::new(RawVideoFormat::I422p10, w, h).unwrap();
        let mut done = None;
        for p in &packets {
            if let Some(f) = rx.depacketize(p) {
                done = Some(f);
            }
        }
        assert_eq!(
            done.unwrap().bytes,
            frame,
            "10-bit 4:2:2 survives the round trip"
        );
    }

    #[test]
    fn pack_2px_10_bit_layout() {
        // Lock the MSB-first 4x10-bit -> 5-octet layout (helps future gear interop).
        let mut out = [0u8; 5];
        pack_2px_10(0x3FF, 0x000, 0x2AA, 0x155, &mut out);
        // Cb0 = all ones fills octet 0 and the top 2 bits of octet 1.
        assert_eq!(out[0], 0xFF);
        assert_eq!(out[1] >> 6, 0x3);
        assert_eq!(unpack_2px_10(&out), (0x3FF, 0x000, 0x2AA, 0x155));
    }

    #[test]
    fn rejects_bad_geometry_and_out_of_bounds_srd() {
        // Odd width is not a whole number of 4:2:2 pgroups.
        let mut tx = St2110VideoPacketizer::new(96, 1, Sampling::YCbCr422_8, 1400);
        assert!(
            tx.packetize(&ramp(64), 5, 2, 0).is_none(),
            "odd width for 4:2:2 rejected"
        );
        assert!(St2110VideoDepacketizer::new(RawVideoFormat::Yuyv, 5, 2).is_none());
        // Unmapped format has no sampling.
        assert!(St2110VideoDepacketizer::new(RawVideoFormat::Nv12, 4, 2).is_none());

        // A too-small buffer for the declared geometry is rejected.
        let mut tx2 = St2110VideoPacketizer::new(96, 1, Sampling::Rgba8, 1400);
        assert!(
            tx2.packetize(&ramp(16), 8, 4, 0).is_none(),
            "buffer too small"
        );

        // A hand-built packet whose SRD names a line past the height must not write
        // out of bounds; it is dropped.
        let mut rx = St2110VideoDepacketizer::new(RawVideoFormat::Rgba8, 2, 2).unwrap();
        let mut bad = alloc::vec![0x80, 0x80 | 96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; // RTP + marker
        bad.extend_from_slice(&[0, 0]); // ext seq
        bad.extend_from_slice(&[0, 8, 0, 99, 0, 0]); // Length 8, Line 99 (>= height), C=0, Off 0
        bad.extend_from_slice(&[0u8; 8]);
        assert!(
            rx.depacketize(&bad).is_none(),
            "SRD past the height is rejected"
        );
    }

    #[test]
    fn rejects_short_packets() {
        let mut rx = St2110VideoDepacketizer::new(RawVideoFormat::Rgba8, 2, 2).unwrap();
        assert!(
            rx.depacketize(&[0u8; 12]).is_none(),
            "no room for the ext-seq / SRD"
        );
        assert!(rx.depacketize(&[]).is_none());
    }
}
