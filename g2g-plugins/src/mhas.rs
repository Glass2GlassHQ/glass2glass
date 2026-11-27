// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::collections::VecDeque;
use alloc::vec::Vec;

use g2g_core::time::ticks_to_ns;
use g2g_core::G2gError;

use crate::annexb::BitReader;
use crate::mpeghconfig::mpegh_config_sample_rate;

// ISO/IEC 23008-3 MHASPacketType values
const MHAS_PACKET_TYPE_CONFIG: u64 = 1;
const MHAS_PACKET_TYPE_FRAME: u64 = 2;
// MHAS labels 1 to 16 belong to the main stream, higher ones to the sub-streams of a multi-stream
const LAST_MAIN_STREAM_LABEL: u64 = 16;
// mpeghdec only decodes coreSbrFrameLengthIndex 1, which is 1024 samples per access unit
pub(crate) const MPEGH_SAMPLES_PER_ACCESS_UNIT: u64 = 1024;
// MPEG-H level 5 caps an access unit at a few hundred kilobytes, so this much pending data is a corrupt stream
const MAX_PENDING_BYTES: usize = 1 << 20;

struct MhasPacketHeader {
    packet_type: u64,
    label: u64,
    header_len: usize,
    payload_len: usize,
}

fn escaped_value(reader: &mut BitReader, widths: [u32; 3]) -> Option<u64> {
    let [first_width, second_width, third_width] = widths;
    let mut value = u64::from(reader.read_bits(first_width)?);
    if value == (1 << first_width) - 1 {
        let second = u64::from(reader.read_bits(second_width)?);
        value += second;
        if second == (1 << second_width) - 1 {
            value += u64::from(reader.read_bits(third_width)?);
        }
    }
    Some(value)
}

// `None` until the whole header is in `data`
fn mhas_packet_header(data: &[u8]) -> Option<MhasPacketHeader> {
    let mut reader = BitReader::new(data);
    let packet_type = escaped_value(&mut reader, [3, 8, 8])?;
    let label = escaped_value(&mut reader, [2, 8, 32])?;
    let payload_len = usize::try_from(escaped_value(&mut reader, [11, 24, 24])?).ok()?;
    // every combination of escaped widths sums to whole bytes
    let header_len = reader.bit_pos() / 8;
    Some(MhasPacketHeader {
        packet_type,
        label,
        header_len,
        payload_len,
    })
}

#[derive(Debug, PartialEq)]
pub(crate) struct MhasAccessUnit {
    pub(crate) data: Vec<u8>,
    pub(crate) pts_ns: Option<u64>,
}

// Splits an MHAS byte stream into access units, however the container cut it. A buffer's
// pts belongs to the first access unit that starts in it (ISO/IEC 13818-1 PES rules), and
// the access units after it are timed 1024 samples apart.
#[derive(Debug, Default)]
pub(crate) struct MhasFramer {
    pending: Vec<u8>,
    // the offset in `pending` where a timestamped buffer starts, and its pts
    buffer_timestamps: VecDeque<(usize, u64)>,
    sample_rate: Option<u32>,
    // the last stamped access unit's pts, and how many access units followed it
    timeline: Option<(u64, u64)>,
}

impl MhasFramer {
    pub(crate) fn push(
        &mut self,
        buf: &[u8],
        pts_ns: Option<u64>,
    ) -> Result<Vec<MhasAccessUnit>, G2gError> {
        if let Some(pts_ns) = pts_ns {
            self.buffer_timestamps
                .push_back((self.pending.len(), pts_ns));
        }
        self.pending.extend_from_slice(buf);
        let boundaries = self.access_unit_ends()?;
        let mut units = Vec::with_capacity(boundaries.len());
        let mut start = 0;
        for end in boundaries {
            let pts_ns = self.access_unit_pts(start);
            units.push(MhasAccessUnit {
                data: self.pending[start..end].to_vec(),
                pts_ns,
            });
            start = end;
        }
        self.pending.drain(..start);
        for (offset, _) in &mut self.buffer_timestamps {
            *offset -= start;
        }
        if self.pending.len() > MAX_PENDING_BYTES {
            return Err(G2gError::CapsMismatch);
        }
        Ok(units)
    }

    // at end of stream the data left over is the last access unit
    pub(crate) fn finish(&mut self) -> Option<MhasAccessUnit> {
        if self.pending.is_empty() {
            return None;
        }
        let pts_ns = self.access_unit_pts(0);
        self.buffer_timestamps.clear();
        Some(MhasAccessUnit {
            data: core::mem::take(&mut self.pending),
            pts_ns,
        })
    }

    pub(crate) fn clear(&mut self) {
        self.pending.clear();
        self.buffer_timestamps.clear();
        self.timeline = None;
    }

    // The end offset of each access unit complete in `pending`. An access unit ends where a
    // main-stream packet follows its frame packet, or where the data ends right after a frame.
    fn access_unit_ends(&mut self) -> Result<Vec<usize>, G2gError> {
        let mut ends = Vec::new();
        let mut position = 0;
        let mut frame_seen = false;
        while let Some(header) = mhas_packet_header(&self.pending[position..]) {
            if frame_seen && header.label <= LAST_MAIN_STREAM_LABEL {
                ends.push(position);
                frame_seen = false;
            }
            let payload_start = position + header.header_len;
            let payload_end = payload_start
                .checked_add(header.payload_len)
                .ok_or(G2gError::CapsMismatch)?;
            let Some(payload) = self.pending.get(payload_start..payload_end) else {
                break;
            };
            match header.packet_type {
                MHAS_PACKET_TYPE_CONFIG => {
                    self.sample_rate =
                        Some(mpegh_config_sample_rate(payload).ok_or(G2gError::CapsMismatch)?);
                }
                MHAS_PACKET_TYPE_FRAME => frame_seen = true,
                _ => {}
            }
            position = payload_end;
        }
        if frame_seen && position == self.pending.len() {
            ends.push(position);
        }
        Ok(ends)
    }

    fn access_unit_pts(&mut self, start: usize) -> Option<u64> {
        let mut stamped = None;
        while let Some(&(offset, pts_ns)) = self.buffer_timestamps.front() {
            if offset > start {
                break;
            }
            stamped = Some(pts_ns);
            self.buffer_timestamps.pop_front();
        }
        if let Some(pts_ns) = stamped {
            self.timeline = Some((pts_ns, 0));
            return Some(pts_ns);
        }
        let (base_ns, count) = self.timeline.as_mut()?;
        *count += 1;
        let sample_rate = self.sample_rate?;
        Some(*base_ns + ticks_to_ns(*count * MPEGH_SAMPLES_PER_ACCESS_UNIT, sample_rate))
    }
}

#[cfg(test)]
pub(crate) mod test_packets {
    use alloc::vec::Vec;

    // a header with no escaped field: type in 3 bits, label in 2, payload length in 11
    pub(crate) fn mhas_packet(packet_type: u8, label: u8, payload: &[u8]) -> Vec<u8> {
        let length = u16::try_from(payload.len()).unwrap();
        assert!(packet_type < 7 && label < 3 && length < 2047);
        let header = (u16::from(packet_type) << 13) | (u16::from(label) << 11) | length;
        let mut packet = header.to_be_bytes().to_vec();
        packet.extend_from_slice(payload);
        packet
    }
}

#[cfg(test)]
mod tests {
    use super::test_packets::mhas_packet;
    use super::*;
    use crate::mpeghconfig::MPEGH_CONFIG_48K_PREFIX;

    const SYNC: u8 = 6;
    const FRAME_DURATION_NS: u64 = 1024 * 1_000_000_000 / 48_000;

    fn access_unit(frame_fill: u8) -> Vec<u8> {
        [
            mhas_packet(SYNC, 0, &[0xa5]),
            mhas_packet(MHAS_PACKET_TYPE_CONFIG as u8, 1, &MPEGH_CONFIG_48K_PREFIX),
            mhas_packet(MHAS_PACKET_TYPE_FRAME as u8, 1, &[frame_fill; 20]),
        ]
        .concat()
    }

    #[test]
    fn escaped_fields_read_the_way_the_standard_sums_them() {
        // type 20: 3 bits of 7, then 8 bits of 13; label 1 in 2 bits; length 2047 + 1 in 11 + 24 bits
        let mut bits = Vec::new();
        let mut push = |value: u64, width: u32| {
            for bit in (0..width).rev() {
                bits.push((value >> bit) & 1 == 1);
            }
        };
        push(7, 3);
        push(13, 8);
        push(1, 2);
        push(2047, 11);
        push(1, 24);
        let bytes: Vec<u8> = bits
            .chunks(8)
            .map(|byte| byte.iter().fold(0, |acc, &bit| (acc << 1) | u8::from(bit)))
            .collect();
        let header = mhas_packet_header(&bytes).expect("a whole header");
        assert_eq!(header.packet_type, 20);
        assert_eq!(header.label, 1);
        assert_eq!(header.payload_len, 2048);
        assert_eq!(header.header_len, bytes.len());
    }

    #[test]
    fn one_buffer_per_access_unit_passes_through_with_its_pts() {
        let mut framer = MhasFramer::default();
        for index in 0..3u64 {
            let unit = access_unit(index as u8);
            let pts_ns = index * FRAME_DURATION_NS;
            assert_eq!(
                framer.push(&unit, Some(pts_ns)).unwrap(),
                [MhasAccessUnit {
                    data: unit,
                    pts_ns: Some(pts_ns)
                }]
            );
        }
    }

    #[test]
    fn access_units_cut_across_buffers_take_the_pts_of_the_buffer_they_start_in() {
        let units: Vec<Vec<u8>> = (0..4).map(access_unit).collect();
        let stream = units.concat();
        let unit_len = units[0].len();
        // the second buffer starts inside the second access unit, so it stamps the third
        let cut = unit_len + unit_len / 2;
        let second_pts_ns = 7_000_000_000;
        let mut framer = MhasFramer::default();
        let mut framed = framer.push(&stream[..cut], Some(0)).unwrap();
        framed.extend(framer.push(&stream[cut..], Some(second_pts_ns)).unwrap());
        let expected = [
            Some(0),
            Some(FRAME_DURATION_NS),
            Some(second_pts_ns),
            Some(second_pts_ns + FRAME_DURATION_NS),
        ];
        assert_eq!(
            framed,
            units
                .into_iter()
                .zip(expected)
                .map(|(data, pts_ns)| MhasAccessUnit { data, pts_ns })
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_partial_packet_waits_for_the_rest() {
        let unit = access_unit(9);
        let mut framer = MhasFramer::default();
        assert!(framer
            .push(&unit[..unit.len() - 1], Some(0))
            .unwrap()
            .is_empty());
        assert_eq!(
            framer.push(&unit[unit.len() - 1..], None).unwrap(),
            [MhasAccessUnit {
                data: unit,
                pts_ns: Some(0)
            }]
        );
    }

    #[test]
    fn end_of_stream_hands_over_the_access_unit_cut_short() {
        let unit = access_unit(3);
        let mut framer = MhasFramer::default();
        assert!(framer
            .push(&unit[..unit.len() - 1], Some(0))
            .unwrap()
            .is_empty());
        assert_eq!(
            framer.finish(),
            Some(MhasAccessUnit {
                data: unit[..unit.len() - 1].to_vec(),
                pts_ns: Some(0)
            })
        );
        assert_eq!(framer.finish(), None);
    }

    #[test]
    fn a_flush_forgets_the_timeline() {
        let mut framer = MhasFramer::default();
        framer.push(&access_unit(0), Some(0)).unwrap();
        framer.clear();
        let unit = access_unit(1);
        assert_eq!(
            framer.push(&unit, None).unwrap(),
            [MhasAccessUnit {
                data: unit,
                pts_ns: None
            }]
        );
    }
}
