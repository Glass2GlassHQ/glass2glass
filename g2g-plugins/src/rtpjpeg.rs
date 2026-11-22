// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! RFC 2435 RTP/JPEG reassembly for 4:2:2 and 4:2:0 frames and fields.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use g2g_core::rtp::RtpHeader;

use crate::rtpdepay::{AccessUnit, MAX_ASSEMBLY_BYTES};
use crate::stillframe::{
    append_default_jpeg_huffman_tables, append_jpeg_segment_header, jpeg_frame_length,
    JPEG_END_OF_IMAGE, JPEG_MARKER_PREFIX, JPEG_SIGNATURE, JPEG_START_OF_SCAN,
};

const EXPLICIT_QUALITY: u8 = 128;
const VARIABLE_TABLES: u8 = 255;
const QUANTIZERS_PER_TABLE: usize = 64;
const DEFINE_QUANTIZATION: u8 = 0xdb;
const DEFINE_RESTART_INTERVAL: u8 = 0xdd;
const BASELINE_FRAME: u8 = 0xc0;
const EXTENDED_FRAME: u8 = 0xc1;

/// Reassembles JPEG images and emits interlaced frames as two JPEG codestreams.
#[derive(Debug, Default)]
pub struct RtpJpegDepayloader {
    frame: Option<JpegAssembly>,
    odd_field: Option<OddField>,
    ssrc: Option<u32>,
    tables: BTreeMap<u8, Quantization>,
}

#[derive(Debug)]
struct OddField {
    unit: AccessUnit,
    size: (u16, u16),
    next_sequence: u16,
}

#[derive(Debug)]
struct JpegAssembly {
    format: JpegFormat,
    unit: AccessUnit,
    scan_start: usize,
    next_sequence: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct JpegFormat {
    field: JpegField,
    width: u16,
    height: u16,
    luma_sampling: u8,
    restart_interval: u16,
    quality: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JpegField {
    Progressive,
    Odd,
    Even,
    Repeat,
}

impl JpegFormat {
    fn parse(payload: &[u8]) -> Option<(Self, usize, &[u8])> {
        let (&[field, a, b, c, kind, quality, width, height], mut body) =
            payload.split_first_chunk::<8>()?;
        if width == 0 || height == 0 {
            return None;
        }
        let field = match field {
            0 => JpegField::Progressive,
            1 => JpegField::Odd,
            2 => JpegField::Even,
            3 => JpegField::Repeat,
            _ => return None,
        };
        let sampling = match kind {
            0 | 64 => 0x21,
            1 | 65 => 0x22,
            _ => return None,
        };
        let restart_interval = if kind >= 64 {
            let (&[hi, lo, _, _], scan) = body.split_first_chunk::<4>()?;
            body = scan;
            let interval = u16::from_be_bytes([hi, lo]);
            if interval == 0 {
                return None;
            }
            interval
        } else {
            0
        };
        let offset = usize::try_from(u32::from_be_bytes([0, a, b, c])).ok()?;
        Some((
            Self {
                field,
                width: u16::from(width) * 8,
                height: u16::from(height) * 8,
                luma_sampling: sampling,
                restart_interval,
                quality,
            },
            offset,
            body,
        ))
    }

    fn jpeg_header(self, tables: &Quantization) -> Vec<u8> {
        let mut jpeg = Vec::from(JPEG_SIGNATURE);
        tables.append(&mut jpeg);
        if self.restart_interval != 0 {
            append_jpeg_segment_header(&mut jpeg, DEFINE_RESTART_INTERVAL, 2);
            jpeg.extend_from_slice(&self.restart_interval.to_be_bytes());
        }
        append_default_jpeg_huffman_tables(&mut jpeg);

        let frame_marker = if tables.precision == 0 {
            BASELINE_FRAME
        } else {
            EXTENDED_FRAME
        };
        append_jpeg_segment_header(&mut jpeg, frame_marker, 15);
        jpeg.push(8); // sample precision
        jpeg.extend_from_slice(&self.height.to_be_bytes());
        jpeg.extend_from_slice(&self.width.to_be_bytes());
        let chroma_table = u8::from(tables.count() == 2);
        jpeg.extend_from_slice(&[
            3,
            1,
            self.luma_sampling,
            0,
            2,
            0x11,
            chroma_table,
            3,
            0x11,
            chroma_table,
        ]);

        append_jpeg_segment_header(&mut jpeg, JPEG_START_OF_SCAN, 10);
        jpeg.extend_from_slice(&[3, 1, 0, 2, 0x11, 3, 0x11, 0, 63, 0]);
        jpeg
    }
}

impl RtpJpegDepayloader {
    pub fn depacketize(&mut self, packet: &[u8]) -> Option<AccessUnit> {
        let previous = self.frame.take();
        let mut odd = self.odd_field.take();
        let parsed = RtpHeader::parse(packet)?;
        let header = parsed.header;
        let payload = &packet[parsed.payload_offset..parsed.payload_offset + parsed.payload_len];
        let same_source = self.ssrc.replace(header.ssrc) == Some(header.ssrc);
        if !same_source {
            self.tables.clear();
        }
        let (format, offset, mut scan) = JpegFormat::parse(payload)?;
        let mut frame = if offset == 0 {
            let data = self.start_header(format, &mut scan)?;
            odd = odd.filter(|odd| {
                same_source
                    && format.field == JpegField::Even
                    && odd.next_sequence == header.sequence
                    && odd.size == (format.width, format.height)
            });
            if format.field == JpegField::Even && odd.is_none() {
                return None;
            }
            JpegAssembly {
                format,
                scan_start: data.len(),
                unit: AccessUnit {
                    data,
                    rtp_timestamp: header.timestamp,
                },
                next_sequence: header.sequence,
            }
        } else {
            previous.filter(|_| same_source)?
        };
        if format != frame.format
            || offset != frame.unit.data.len() - frame.scan_start
            || frame.next_sequence != header.sequence
            || frame.unit.rtp_timestamp != header.timestamp
            || scan.is_empty()
            || offset.saturating_add(scan.len()) > MAX_ASSEMBLY_BYTES
        {
            return None;
        }
        frame.unit.data.extend_from_slice(scan);
        if header.marker {
            // Explicit EOI can be followed by padding. Some senders omit it.
            if let Some(end) = jpeg_frame_length(&frame.unit.data).ok()? {
                frame.unit.data.truncate(end);
            } else {
                frame
                    .unit
                    .data
                    .extend_from_slice(&[JPEG_MARKER_PREFIX, JPEG_END_OF_IMAGE]);
            }
            return self.finish_image(frame, odd, header.sequence);
        }
        frame.next_sequence = header.sequence.wrapping_add(1);
        self.frame = Some(frame);
        self.odd_field = odd;
        None
    }

    fn finish_image(
        &mut self,
        mut frame: JpegAssembly,
        odd: Option<OddField>,
        sequence: u16,
    ) -> Option<AccessUnit> {
        match frame.format.field {
            JpegField::Progressive => {}
            JpegField::Odd => {
                self.odd_field = Some(OddField {
                    unit: frame.unit,
                    size: (frame.format.width, frame.format.height),
                    next_sequence: sequence.wrapping_add(1),
                });
                return None;
            }
            JpegField::Even => {
                // RFC 2435 sends the odd field first, but its rows go below the even field.
                let odd = odd?;
                frame.unit.rtp_timestamp = odd.unit.rtp_timestamp;
                frame.unit.data.extend(odd.unit.data);
            }
            JpegField::Repeat => frame.unit.data.extend_from_within(..),
        }
        Some(frame.unit)
    }

    fn start_header(&mut self, format: JpegFormat, scan: &mut &[u8]) -> Option<Vec<u8>> {
        let quality = format.quality;
        if quality < EXPLICIT_QUALITY {
            return Some(format.jpeg_header(&Quantization::from_quality(quality)?));
        }
        let (&[_, precision, hi, lo], rest) = scan.split_first_chunk::<4>()?;
        let length = usize::from(u16::from_be_bytes([hi, lo]));
        let (body, remaining) = rest.split_at_checked(length)?;
        *scan = remaining;
        if body.is_empty() {
            return Some(format.jpeg_header(self.tables.get(&quality)?));
        }
        let tables = Quantization::from_wire(precision, body)?;
        let jpeg = format.jpeg_header(&tables);
        if quality != VARIABLE_TABLES {
            self.tables.insert(quality, tables);
        }
        Some(jpeg)
    }
}

#[derive(Debug)]
struct Quantization {
    precision: u8,
    data: Vec<u8>,
}

impl Quantization {
    fn from_wire(precision: u8, data: &[u8]) -> Option<Self> {
        let first = table_length(precision, 0);
        let precision = match data.len() {
            len if len == first => precision & 1,
            len if len == first + table_length(precision, 1) => precision & 3,
            _ => return None,
        };
        Some(Self {
            precision,
            data: data.to_vec(),
        })
    }

    fn from_quality(quality: u8) -> Option<Self> {
        let quality = u16::from(quality);
        let scale = match quality {
            1..=50 => 5000 / quality,
            51..=99 => 200 - 2 * quality,
            _ => return None,
        };
        let data = [LUMA_QUANTIZERS, CHROMA_QUANTIZERS]
            .iter()
            .flat_map(|table| {
                ZIGZAG.iter().map(move |&index| {
                    ((u32::from(table[index]) * u32::from(scale) + 50) / 100).clamp(1, 255) as u8
                })
            })
            .collect();
        Some(Self { precision: 0, data })
    }

    fn count(&self) -> u8 {
        if self.data.len() == table_length(self.precision, 0) {
            1
        } else {
            2
        }
    }

    fn append(&self, jpeg: &mut Vec<u8>) {
        let count = self.count();
        append_jpeg_segment_header(
            jpeg,
            DEFINE_QUANTIZATION,
            self.data.len() as u16 + u16::from(count),
        );
        let mut rest = self.data.as_slice();
        for table in 0..count {
            let precision = (self.precision >> table) & 1;
            let (data, next) = rest.split_at(table_length(self.precision, table));
            jpeg.push((precision << 4) | table);
            jpeg.extend_from_slice(data);
            rest = next;
        }
    }
}

fn table_length(precision: u8, table: u8) -> usize {
    QUANTIZERS_PER_TABLE << ((precision >> table) & 1)
}

// ITU-T T.81 Tables K.1 and K.2, in natural order.
const LUMA_QUANTIZERS: [u8; QUANTIZERS_PER_TABLE] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];
const CHROMA_QUANTIZERS: [u8; QUANTIZERS_PER_TABLE] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];
const ZIGZAG: [usize; QUANTIZERS_PER_TABLE] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];
