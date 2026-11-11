// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Sans-IO H.264 and H.265 RTP depayloaders (RFC 3550 header + RFC 6184 /
//! RFC 7798 payload), the receive-side inverses of the
//! [`rtppay`](crate::rtppay) packetizers. They turn a stream of RTP packets
//! back into Annex-B access units: single-NAL and aggregation payloads pass
//! through; fragmentation units reassemble; the RTP marker bit closes an
//! access unit. No I/O: `UdpSrc` wraps this and feeds it the
//! datagrams a socket receives. Opus (RFC 7587) needs no reassembly: each RTP
//! payload is one Opus packet.
//!
//! This is the basic in-order depayloader. A jitter buffer (packet reorder,
//! loss concealment, RTCP) is the larger receive-side follow-up (design/TODO.md);
//! out-of-order or lost packets are detected via the sequence number and reset
//! the in-flight reassembly so a gap never welds two access units together.

use alloc::vec::Vec;

use g2g_core::mediaclock::MediaClock;
use g2g_core::rtp::RtpHeader;
use g2g_core::VideoCodec;

use crate::rtppay::{FU_A_TYPE, FU_END_BIT, FU_START_BIT, H265_FU_TYPE, H265_NAL_HEADER_LEN};
use crate::sdp::OPUS_CLOCK_HZ;

/// RFC 6184 aggregation packet NAL type.
const STAP_A_TYPE: u8 = 24;
const ANNEX_B_START_CODE: [u8; 4] = [0, 0, 0, 1];

/// One depayloaded access unit: Annex-B bytes for H.264, one Opus packet for
/// Opus, plus the RTP timestamp shared by its packets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessUnit {
    pub data: Vec<u8>,
    pub rtp_timestamp: u32,
}

/// Cap the in-flight FU-A and access-unit reassembly buffers. A never-ending
/// fragment run (start without end) or an access unit whose marker never arrives
/// would otherwise grow without bound on untrusted RTP; past this we drop the
/// in-flight data and resync at the next start-of-AU. 8 MiB covers a large
/// keyframe.
const MAX_ASSEMBLY_BYTES: usize = 8 * 1024 * 1024;

// RFC 3640 `mpeg4-generic` AAC in AAC-hbr or AAC-lbr mode, as its SDP fmtp declares it
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mpeg4GenericAac {
    pub size_length: u32,
    pub index_length: u32,
    pub index_delta_length: u32,
    pub clock_rate_hz: u32,
    pub audio_specific_config: Vec<u8>,
    // RTP clock ticks between consecutive access units in one packet
    pub ticks_per_access_unit: u32,
}

// RFC 6416 MP4A-LATM AAC whose StreamMuxConfig the SDP fmtp carries (`cpresent=0`)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mp4aLatmAac {
    pub clock_rate_hz: u32,
    // the StreamMuxConfig's AudioSpecificConfig, moved to a byte boundary
    pub audio_specific_config: Vec<u8>,
    // RTP clock ticks between consecutive access units in one packet
    pub ticks_per_access_unit: u32,
}

// aacparse and the decoders read the rate and channels off each ADTS header
#[cfg(any(feature = "udp-ingress", feature = "rtsp-server"))]
pub(crate) const ADTS_AAC_CAPS: g2g_core::Caps = g2g_core::Caps::Audio {
    format: g2g_core::AudioFormat::Aac,
    channels: 0,
    sample_rate: 0,
    channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
};

// the payload formats the plain RTP receive path depayloads, named as in gst `encoding-name`
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum RtpEncoding {
    #[default]
    H264,
    Opus,
    H265,
    Aac(Mpeg4GenericAac),
    AacLatm(Mp4aLatmAac),
}

impl RtpEncoding {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "H264" => Some(RtpEncoding::H264),
            "OPUS" => Some(RtpEncoding::Opus),
            "H265" => Some(RtpEncoding::H265),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            RtpEncoding::H264 => "H264",
            RtpEncoding::Opus => "OPUS",
            RtpEncoding::H265 => "H265",
            RtpEncoding::Aac(_) => "MPEG4-GENERIC",
            RtpEncoding::AacLatm(_) => "MP4A-LATM",
        }
    }

    pub fn clock_rate_hz(&self) -> u32 {
        match self {
            RtpEncoding::H264 | RtpEncoding::H265 => MediaClock::VIDEO_RATE_HZ,
            RtpEncoding::Opus => OPUS_CLOCK_HZ,
            RtpEncoding::Aac(format) => format.clock_rate_hz,
            RtpEncoding::AacLatm(format) => format.clock_rate_hz,
        }
    }

    pub fn video_codec(&self) -> Option<VideoCodec> {
        match self {
            RtpEncoding::H264 => Some(VideoCodec::H264),
            RtpEncoding::H265 => Some(VideoCodec::H265),
            RtpEncoding::Opus | RtpEncoding::Aac(_) | RtpEncoding::AacLatm(_) => None,
        }
    }

    // every Opus packet and AAC frame decodes without the ones before it
    pub fn is_keyframe(&self, unit: &[u8]) -> bool {
        match self {
            RtpEncoding::H264 => crate::annexb::au_is_keyframe(VideoCodec::H264, unit),
            RtpEncoding::H265 => crate::annexb::au_is_keyframe(VideoCodec::H265, unit),
            RtpEncoding::Opus | RtpEncoding::Aac(_) | RtpEncoding::AacLatm(_) => true,
        }
    }
}

pub type AccessUnits =
    core::iter::Chain<core::option::IntoIter<AccessUnit>, core::option::IntoIter<AccessUnit>>;

#[derive(Debug)]
pub enum RtpDepayloader {
    H264(RtpH264Depayloader),
    Opus,
    H265(RtpH265Depayloader),
    Aac(RtpAacDepayloader),
    AacLatm(RtpLatmDepayloader),
}

impl RtpDepayloader {
    pub fn new(encoding: &RtpEncoding) -> Self {
        match encoding {
            RtpEncoding::H264 => RtpDepayloader::H264(RtpH264Depayloader::new()),
            RtpEncoding::Opus => RtpDepayloader::Opus,
            RtpEncoding::H265 => RtpDepayloader::H265(RtpH265Depayloader::new()),
            RtpEncoding::Aac(format) => RtpDepayloader::Aac(RtpAacDepayloader::new(format.clone())),
            RtpEncoding::AacLatm(format) => {
                RtpDepayloader::AacLatm(RtpLatmDepayloader::new(format.clone()))
            }
        }
    }

    // an AAC packet can carry many access units, the other formats at most two
    pub fn depacketize(&mut self, packet: &[u8]) -> Vec<AccessUnit> {
        match self {
            RtpDepayloader::H264(depayloader) => depayloader.depacketize(packet).collect(),
            RtpDepayloader::H265(depayloader) => depayloader.depacketize(packet).collect(),
            RtpDepayloader::Opus => depacketize_opus(packet).into_iter().collect(),
            RtpDepayloader::Aac(depayloader) => depayloader.depacketize(packet),
            RtpDepayloader::AacLatm(depayloader) => depayloader.depacketize(packet),
        }
    }
}

// RFC 3640 3.2.1: the AU-headers-length field ahead of the AU headers, in bits
const AU_HEADERS_LENGTH_BYTES: usize = 2;

// a fragmented access unit (RFC 3640 3.2.3) still arriving
#[derive(Debug)]
struct AacFragment {
    timestamp: u32,
    size: usize,
    data: Vec<u8>,
}

// RFC 3640 AAC-hbr / AAC-lbr, each access unit ADTS-framed the way rtspsrc delivers AAC
#[derive(Debug)]
pub struct RtpAacDepayloader {
    format: Mpeg4GenericAac,
    fragment: Option<AacFragment>,
    last_seq: Option<u16>,
}

impl RtpAacDepayloader {
    pub fn new(format: Mpeg4GenericAac) -> Self {
        Self {
            format,
            fragment: None,
            last_seq: None,
        }
    }

    // a packet that does not parse, or a fragment run broken by loss, yields nothing
    pub fn depacketize(&mut self, packet: &[u8]) -> Vec<AccessUnit> {
        let Some(parsed) = RtpHeader::parse(packet) else {
            return Vec::new();
        };
        let seq = parsed.header.sequence;
        if self
            .last_seq
            .is_some_and(|prev| seq != prev.wrapping_add(1))
        {
            self.fragment = None;
        }
        self.last_seq = Some(seq);
        let payload = packet
            .get(parsed.payload_offset..parsed.payload_offset + parsed.payload_len)
            .unwrap_or_default();
        let Some((sizes, data)) = self.au_sizes(payload) else {
            self.fragment = None;
            return Vec::new();
        };
        let timestamp = parsed.header.timestamp;
        match sizes.as_slice() {
            [size] if *size > data.len() || self.fragment.is_some() => self
                .continue_fragment(timestamp, *size, data, parsed.header.marker)
                .into_iter()
                .collect(),
            _ => self.whole_access_units(timestamp, &sizes, data),
        }
    }

    // the AU sizes the header section lists and the data section behind it
    fn au_sizes<'a>(&self, payload: &'a [u8]) -> Option<(Vec<usize>, &'a [u8])> {
        let headers_bits = usize::from(u16::from_be_bytes(
            payload.get(..AU_HEADERS_LENGTH_BYTES)?.try_into().ok()?,
        ));
        let headers_end = AU_HEADERS_LENGTH_BYTES + headers_bits.div_ceil(8);
        let headers = payload.get(AU_HEADERS_LENGTH_BYTES..headers_end)?;
        let mut reader = crate::annexb::BitReader::new(headers);
        let mut sizes = Vec::new();
        let mut read_bits = 0;
        while read_bits < headers_bits {
            let index_bits = if sizes.is_empty() {
                self.format.index_length
            } else {
                self.format.index_delta_length
            };
            let size = reader.read_bits(self.format.size_length)?;
            let index = match index_bits {
                0 => 0,
                bits => reader.read_bits(bits)?,
            };
            // a nonzero index delta interleaves access units across packets
            if !sizes.is_empty() && index != 0 {
                return None;
            }
            sizes.push(size as usize);
            read_bits += (self.format.size_length + index_bits) as usize;
        }
        (read_bits == headers_bits && !sizes.is_empty()).then(|| (sizes, &payload[headers_end..]))
    }

    fn whole_access_units(
        &mut self,
        timestamp: u32,
        sizes: &[usize],
        mut data: &[u8],
    ) -> Vec<AccessUnit> {
        self.fragment = None;
        let mut units = Vec::with_capacity(sizes.len());
        for (index, &size) in sizes.iter().enumerate() {
            let Some(raw) = data.get(..size) else {
                break;
            };
            data = &data[size..];
            let offset = self.format.ticks_per_access_unit.wrapping_mul(index as u32);
            let rtp_timestamp = timestamp.wrapping_add(offset);
            units.extend(adts_unit(
                &self.format.audio_specific_config,
                rtp_timestamp,
                raw,
            ));
        }
        units
    }

    fn continue_fragment(
        &mut self,
        timestamp: u32,
        size: usize,
        data: &[u8],
        marker: bool,
    ) -> Option<AccessUnit> {
        let continues = self
            .fragment
            .as_ref()
            .is_some_and(|fragment| fragment.timestamp == timestamp && fragment.size == size);
        if !continues {
            self.fragment = Some(AacFragment {
                timestamp,
                size,
                data: Vec::with_capacity(size),
            });
        }
        let fragment = self.fragment.as_mut()?;
        fragment.data.extend_from_slice(data);
        if fragment.data.len() > fragment.size {
            self.fragment = None;
            return None;
        }
        if !marker {
            return None;
        }
        let whole = self.fragment.take()?;
        (whole.data.len() == whole.size)
            .then(|| {
                adts_unit(
                    &self.format.audio_specific_config,
                    whole.timestamp,
                    &whole.data,
                )
            })
            .flatten()
    }
}

fn adts_unit(audio_specific_config: &[u8], rtp_timestamp: u32, raw: &[u8]) -> Option<AccessUnit> {
    let data = crate::aacparse::adts_from_asc(audio_specific_config, raw)?;
    Some(AccessUnit {
        data,
        rtp_timestamp,
    })
}

// RFC 6416 MP4A-LATM, each AAC frame of an AudioMuxElement ADTS-framed like the RFC 3640 path
#[derive(Debug)]
pub struct RtpLatmDepayloader {
    format: Mp4aLatmAac,
    // the payload of an AudioMuxElement whose last packet, the one with the marker, is still due
    element: Vec<u8>,
    // a packet of `element` was lost or it outgrew the assembly cap
    element_broken: bool,
    last_seq: Option<u16>,
}

impl RtpLatmDepayloader {
    pub fn new(format: Mp4aLatmAac) -> Self {
        Self {
            format,
            element: Vec::new(),
            element_broken: false,
            last_seq: None,
        }
    }

    // an AudioMuxElement missing a packet yields nothing
    pub fn depacketize(&mut self, packet: &[u8]) -> Vec<AccessUnit> {
        let Some(parsed) = RtpHeader::parse(packet) else {
            return Vec::new();
        };
        let seq = parsed.header.sequence;
        if self
            .last_seq
            .is_some_and(|prev| seq != prev.wrapping_add(1))
        {
            self.element_broken = true;
        }
        self.last_seq = Some(seq);
        let payload = packet
            .get(parsed.payload_offset..parsed.payload_offset + parsed.payload_len)
            .unwrap_or_default();
        self.element.extend_from_slice(payload);
        if self.element.len() > MAX_ASSEMBLY_BYTES {
            self.element.clear();
            self.element_broken = true;
        }
        if !parsed.header.marker {
            return Vec::new();
        }
        let element = core::mem::take(&mut self.element);
        if core::mem::take(&mut self.element_broken) {
            return Vec::new();
        }
        // RFC 3016 4.1 lets one packet carry several AudioMuxElements back to back,
        // and each is its subframes back to back, so the frames are read in one run
        let mut units = Vec::new();
        let mut rest = element.as_slice();
        while !rest.is_empty() {
            let Some((frame, after)) = latm_payload(rest) else {
                break;
            };
            rest = after;
            let offset = self
                .format
                .ticks_per_access_unit
                .wrapping_mul(units.len() as u32);
            let rtp_timestamp = parsed.header.timestamp.wrapping_add(offset);
            units.extend(adts_unit(
                &self.format.audio_specific_config,
                rtp_timestamp,
                frame,
            ));
        }
        units
    }
}

// ISO/IEC 14496-3 PayloadLengthInfo: 255-valued bytes add up until a smaller one ends the length
fn latm_payload(element: &[u8]) -> Option<(&[u8], &[u8])> {
    const LENGTH_CONTINUES: u8 = 0xFF;
    let mut length = 0usize;
    let mut position = 0;
    loop {
        let byte = *element.get(position)?;
        position += 1;
        length += usize::from(byte);
        if byte != LENGTH_CONTINUES {
            break;
        }
    }
    let end = position.checked_add(length)?;
    Some((element.get(position..end)?, element.get(end..)?))
}

// RFC 7587 never fragments, so each payload is one whole Opus packet
fn depacketize_opus(packet: &[u8]) -> Option<AccessUnit> {
    let parsed = RtpHeader::parse(packet)?;
    let end = parsed.payload_offset.checked_add(parsed.payload_len)?;
    let payload = packet.get(parsed.payload_offset..end)?;
    if payload.is_empty() {
        return None;
    }
    Some(AccessUnit {
        data: payload.to_vec(),
        rtp_timestamp: parsed.header.timestamp,
    })
}

#[derive(Debug, Default)]
pub struct RtpH264Depayloader {
    assembler: NalAssembler,
}

impl RtpH264Depayloader {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one RTP packet. Yields the access unit a new RTP timestamp closed,
    /// then the one this packet's marker bit closed. Packets that are too short
    /// or use unsupported aggregation modes (STAP-B / MTAP / FU-B) are skipped.
    pub fn depacketize(&mut self, packet: &[u8]) -> AccessUnits {
        self.assembler.depacketize(packet, add_h264_nals)
    }
}

fn add_h264_nals(assembler: &mut NalAssembler, payload: &[u8]) {
    match payload[0] & 0x1F {
        // Single NAL unit (types 1..=23): the payload is the NAL verbatim.
        1..=23 => {
            assembler.abandon_fragment();
            assembler.push_nal(payload);
        }
        // STAP-A: [type 24][ (16-bit size)(NAL) ]* aggregated NALs, used
        // mostly to carry SPS+PPS in one packet.
        t if t == STAP_A_TYPE => {
            assembler.abandon_fragment();
            assembler.push_aggregated_nals(&payload[1..]);
        }
        // FU-A: a single NAL fragmented across packets. The original NAL
        // header byte is rebuilt from the FU indicator's F|NRI and the FU
        // header's type; Start/End bits bound the fragment run.
        t if t == FU_A_TYPE => {
            let Some(&fu_header) = payload.get(1) else {
                return;
            };
            if fu_header & FU_START_BIT != 0 {
                assembler.start_fragment(&[(payload[0] & 0xE0) | (fu_header & 0x1F)]);
            }
            assembler.extend_fragment(&payload[2..], fu_header & FU_END_BIT != 0);
        }
        // STAP-B (25), MTAP (26/27), FU-B (29): unsupported, skip.
        _ => {}
    }
}

/// Sans-IO H.265 RTP depayloader (RFC 7798), for senders that keep
/// `sprop-max-don-diff` at 0 and so carry no DONL fields.
#[derive(Debug, Default)]
pub struct RtpH265Depayloader {
    assembler: NalAssembler,
}

impl RtpH265Depayloader {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one RTP packet, with the same output as [`RtpH264Depayloader::depacketize`].
    /// PACI packets are skipped.
    pub fn depacketize(&mut self, packet: &[u8]) -> AccessUnits {
        self.assembler.depacketize(packet, add_h265_nals)
    }
}

/// RFC 7798 aggregation packet NAL type.
const H265_AP_TYPE: u8 = 48;

fn add_h265_nals(assembler: &mut NalAssembler, payload: &[u8]) {
    if payload.len() < H265_NAL_HEADER_LEN {
        return;
    }
    match (payload[0] >> 1) & 0x3F {
        0..=47 => {
            assembler.abandon_fragment();
            assembler.push_nal(payload);
        }
        H265_AP_TYPE => {
            assembler.abandon_fragment();
            assembler.push_aggregated_nals(&payload[H265_NAL_HEADER_LEN..]);
        }
        // the FU header after the payload header carries S, E and the fragmented NAL's type
        H265_FU_TYPE => {
            let Some(&fu_header) = payload.get(H265_NAL_HEADER_LEN) else {
                return;
            };
            if fu_header & FU_START_BIT != 0 {
                let nal_type = fu_header & 0x3F;
                assembler.start_fragment(&[(payload[0] & 0x81) | (nal_type << 1), payload[1]]);
            }
            assembler.extend_fragment(
                &payload[H265_NAL_HEADER_LEN + 1..],
                fu_header & FU_END_BIT != 0,
            );
        }
        _ => {}
    }
}

// rebuilds Annex-B access units from the NALs of consecutive RTP packets
#[derive(Debug, Default)]
struct NalAssembler {
    /// Annex-B NALs accumulated for the access unit currently being assembled.
    au: Vec<u8>,
    /// Where the fragmented NAL being reassembled starts in `au`, while one is.
    fragment_start: Option<usize>,
    /// Last RTP sequence number seen, for gap detection.
    last_seq: Option<u16>,
    /// RTP timestamp of the in-flight access unit.
    timestamp: u32,
    /// Part of the current access unit was lost or dropped, so its remaining
    /// packets are skipped until the next access-unit boundary.
    discarding: bool,
}

impl NalAssembler {
    fn depacketize(
        &mut self,
        packet: &[u8],
        add_nals: fn(&mut NalAssembler, &[u8]),
    ) -> AccessUnits {
        // One RTP header parser for the whole workspace (the CSRC / extension /
        // padding offset math lives in `g2g_core::rtp`, shared with the MCU RTP
        // source); a malformed or non-v2 datagram is rejected.
        let Some(parsed) = RtpHeader::parse(packet) else {
            return None.into_iter().chain(None);
        };
        let seq = parsed.header.sequence;

        // A sequence gap (loss / reorder) means the in-flight FU-A and access
        // unit can no longer be trusted; drop them so we never splice across
        // the discontinuity. The next start-of-AU rebuilds cleanly.
        if let Some(prev) = self.last_seq {
            if seq != prev.wrapping_add(1) {
                self.drop_access_unit();
            }
        }
        self.last_seq = Some(seq);

        // senders that omit the marker bit end an access unit only by starting
        // the next one with a new RTP timestamp
        let mut closed_by_timestamp = None;
        if parsed.header.timestamp != self.timestamp {
            self.abandon_fragment();
            if !self.discarding && !self.au.is_empty() {
                closed_by_timestamp = Some(self.take_access_unit());
            }
            self.au.clear();
            self.discarding = false;
            self.timestamp = parsed.header.timestamp;
        }

        let payload = packet
            .get(parsed.payload_offset..parsed.payload_offset + parsed.payload_len)
            .unwrap_or_default();
        if !self.discarding && !payload.is_empty() {
            add_nals(self, payload);
        }
        let closed_by_marker = self.close_on_marker(parsed.header.marker);
        closed_by_timestamp.into_iter().chain(closed_by_marker)
    }

    fn close_on_marker(&mut self, marker: bool) -> Option<AccessUnit> {
        if self.discarding {
            self.discarding = !marker;
            return None;
        }
        if !marker {
            return None;
        }
        self.abandon_fragment();
        if self.au.is_empty() {
            return None;
        }
        Some(self.take_access_unit())
    }

    // `[ (16-bit size)(NAL) ]*`, the body of a STAP-A or an H.265 aggregation packet
    fn push_aggregated_nals(&mut self, mut body: &[u8]) {
        while let [high, low, rest @ ..] = body {
            let size = usize::from(u16::from_be_bytes([*high, *low]));
            let Some(nal) = rest.get(..size) else {
                break;
            };
            self.push_nal(nal);
            body = &rest[size..];
        }
    }

    fn start_fragment(&mut self, nal_header: &[u8]) {
        self.abandon_fragment();
        self.fragment_start = Some(self.au.len());
        self.au.extend_from_slice(&ANNEX_B_START_CODE);
        self.au.extend_from_slice(nal_header);
    }

    fn extend_fragment(&mut self, data: &[u8], end: bool) {
        if self.fragment_start.is_none() {
            return;
        }
        self.au.extend_from_slice(data);
        if self.au.len() > MAX_ASSEMBLY_BYTES {
            // A fragment run that never ends; drop the in-flight data
            // and resync at the next start-of-AU.
            self.drop_access_unit();
            return;
        }
        if end {
            self.fragment_start = None;
        }
    }

    fn take_access_unit(&mut self) -> AccessUnit {
        // the next access unit is likely about as large as this one
        let next_capacity = self.au.len();
        let data = core::mem::replace(&mut self.au, Vec::with_capacity(next_capacity));
        AccessUnit {
            data,
            rtp_timestamp: self.timestamp,
        }
    }

    fn drop_access_unit(&mut self) {
        self.fragment_start = None;
        self.au.clear();
        self.discarding = true;
    }

    // a fragment run cut short by another NAL or the marker is incomplete, drop it
    fn abandon_fragment(&mut self) {
        if let Some(start) = self.fragment_start.take() {
            self.au.truncate(start);
        }
    }

    /// Append one NAL to the in-flight access unit as a 4-byte-start-code
    /// Annex-B unit.
    fn push_nal(&mut self, nal: &[u8]) {
        if nal.is_empty() {
            return;
        }
        self.au.extend_from_slice(&ANNEX_B_START_CODE);
        self.au.extend_from_slice(nal);
        if self.au.len() > MAX_ASSEMBLY_BYTES {
            // An access unit whose marker never arrives; drop it rather than
            // buffer without bound.
            self.drop_access_unit();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rtppay::RtpH264Packetizer;

    // AAC-LC, 48 kHz, stereo
    const AAC_CONFIG: [u8; 2] = [0x11, 0x90];
    const AAC_RATE_HZ: u32 = 48_000;

    // the format g2g's own AAC-hbr SDP declares, parsed back
    fn aac_hbr_format() -> Mpeg4GenericAac {
        let section = crate::sdp::aac_media_section(97, &AAC_CONFIG, AAC_RATE_HZ, 2, 0);
        crate::sdp::SdpMedia::parse(&crate::sdp::session_description(&[section]))
            .and_then(|media| media.mpeg4_generic_aac)
            .expect("an AAC-hbr section")
    }

    fn adts(raw: &[u8]) -> Vec<u8> {
        crate::aacparse::adts_from_asc(&AAC_CONFIG, raw).unwrap()
    }

    #[test]
    fn aac_hbr_units_come_back_adts_framed_whole_or_fragmented() {
        let format = aac_hbr_format();
        assert_eq!((format.size_length, format.index_length), (13, 3));
        let au: Vec<u8> = (0..40).collect();
        let mut depay = RtpDepayloader::new(&RtpEncoding::Aac(format));
        let mut packetizer = crate::rtppay::RtpAacPacketizer::new(97, 1);
        let whole = packetizer.packetize(&au, 1024).unwrap();
        assert_eq!(
            depay.depacketize(&whole[0]),
            [AccessUnit {
                data: adts(&au),
                rtp_timestamp: 1024
            }]
        );
        let mut fragmenting = crate::rtppay::RtpAacPacketizer::new(97, 1).with_max_payload(16);
        let fragments = fragmenting.packetize(&au, 2048).unwrap();
        assert!(fragments.len() > 2);
        let mut units = Vec::new();
        for fragment in &fragments {
            units.extend(depay.depacketize(fragment));
        }
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].data, adts(&au));
        // a lost middle fragment drops the unit
        let mut lossy = crate::rtppay::RtpAacPacketizer::new(97, 1).with_max_payload(16);
        let fragments = lossy.packetize(&au, 4096).unwrap();
        let mut units = Vec::new();
        for (index, fragment) in fragments.iter().enumerate() {
            if index != 1 {
                units.extend(depay.depacketize(fragment));
            }
        }
        assert!(units.is_empty());
    }

    // ffmpeg's RTP muxer puts several AAC frames behind one AU header section
    #[test]
    fn several_aac_hbr_units_in_one_packet_step_their_timestamps() {
        let format = aac_hbr_format();
        let ticks = format.ticks_per_access_unit;
        assert_eq!(ticks, 1024);
        let first: Vec<u8> = (0..5).collect();
        let second: Vec<u8> = (10..13).collect();
        // two 13-bit sizes, each with a zero 3-bit index or index delta
        let headers_bits: u16 = 2 * 16;
        let mut payload = headers_bits.to_be_bytes().to_vec();
        for unit in [&first, &second] {
            payload.extend_from_slice(&((unit.len() as u16) << 3).to_be_bytes());
        }
        payload.extend_from_slice(&first);
        payload.extend_from_slice(&second);
        let header = g2g_core::rtp::RtpHeader {
            marker: true,
            payload_type: 97,
            sequence: 7,
            timestamp: 9000,
            ssrc: 1,
        };
        let mut packet = header.to_bytes().to_vec();
        packet.extend_from_slice(&payload);
        let units = RtpDepayloader::new(&RtpEncoding::Aac(format)).depacketize(&packet);
        assert_eq!(
            units,
            [
                AccessUnit {
                    data: adts(&first),
                    rtp_timestamp: 9000
                },
                AccessUnit {
                    data: adts(&second),
                    rtp_timestamp: 9000 + ticks
                }
            ]
        );
    }

    fn latm_format() -> Mp4aLatmAac {
        Mp4aLatmAac {
            clock_rate_hz: AAC_RATE_HZ,
            audio_specific_config: AAC_CONFIG.to_vec(),
            ticks_per_access_unit: crate::aacparse::SAMPLES_PER_ACCESS_UNIT,
        }
    }

    // each frame behind its PayloadLengthInfo, as `rtpmp4apay` writes it
    fn latm_element(frames: &[&[u8]]) -> Vec<u8> {
        let mut element = Vec::new();
        for frame in frames {
            element.extend(core::iter::repeat_n(0xFF, frame.len() / 255));
            element.push((frame.len() % 255) as u8);
            element.extend_from_slice(frame);
        }
        element
    }

    fn rtp_packet(sequence: u16, timestamp: u32, marker: bool, payload: &[u8]) -> Vec<u8> {
        let header = g2g_core::rtp::RtpHeader {
            marker,
            payload_type: 96,
            sequence,
            timestamp,
            ssrc: 1,
        };
        let mut packet = header.to_bytes().to_vec();
        packet.extend_from_slice(payload);
        packet
    }

    #[test]
    fn latm_frames_come_back_adts_framed_whole_or_fragmented() {
        // long enough that its length needs a 255 continuation byte
        let frame: Vec<u8> = (0..300).map(|byte| byte as u8).collect();
        let element = latm_element(&[&frame]);
        assert_eq!(element[..2], [0xFF, 45]);
        let expected = [AccessUnit {
            data: adts(&frame),
            rtp_timestamp: 4800,
        }];

        let mut depay = RtpDepayloader::new(&RtpEncoding::AacLatm(latm_format()));
        assert_eq!(
            depay.depacketize(&rtp_packet(1, 4800, true, &element)),
            expected
        );

        let (head, tail) = element.split_at(100);
        assert!(depay
            .depacketize(&rtp_packet(2, 4800, false, head))
            .is_empty());
        assert_eq!(
            depay.depacketize(&rtp_packet(3, 4800, true, tail)),
            expected
        );
    }

    #[test]
    fn two_latm_frames_in_one_packet_step_their_timestamps() {
        let first: Vec<u8> = (0..5).collect();
        let second: Vec<u8> = (10..13).collect();
        let format = latm_format();
        let ticks = format.ticks_per_access_unit;
        let packet = rtp_packet(7, 9000, true, &latm_element(&[&first, &second]));
        assert_eq!(
            RtpDepayloader::new(&RtpEncoding::AacLatm(format)).depacketize(&packet),
            [
                AccessUnit {
                    data: adts(&first),
                    rtp_timestamp: 9000
                },
                AccessUnit {
                    data: adts(&second),
                    rtp_timestamp: 9000 + ticks
                }
            ]
        );
    }

    #[test]
    fn a_latm_element_missing_a_packet_yields_nothing() {
        let frame: Vec<u8> = (0..40).collect();
        let element = latm_element(&[&frame]);
        let (head, tail) = element.split_at(10);
        let mut depay = RtpDepayloader::new(&RtpEncoding::AacLatm(latm_format()));
        assert!(depay.depacketize(&rtp_packet(1, 0, false, head)).is_empty());
        // sequence 2 is lost
        assert!(depay.depacketize(&rtp_packet(3, 0, true, tail)).is_empty());
        // the next whole element comes through
        assert_eq!(
            depay.depacketize(&rtp_packet(4, 1024, true, &element)),
            [AccessUnit {
                data: adts(&frame),
                rtp_timestamp: 1024
            }]
        );
    }

    fn wrap(nals: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        for nal in nals {
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(nal);
        }
        out
    }

    #[test]
    fn round_trips_a_small_single_nal_access_unit() {
        let au = wrap(&[&[0x67, 0x42, 0x00], &[0x65, 1, 2, 3, 4]]);
        let mut pkt = RtpH264Packetizer::new(96, 7);
        let mut depay = RtpH264Depayloader::new();
        let packets = pkt.packetize(&au, 9000);

        let mut out = None;
        for p in &packets {
            for unit in depay.depacketize(p) {
                out = Some(unit);
            }
        }
        let unit = out.expect("marker closes the access unit");
        assert_eq!(unit.data, au, "depayloaded AU matches the original Annex-B");
        assert_eq!(unit.rtp_timestamp, 9000);
    }

    #[test]
    fn reassembles_fu_a_fragments() {
        // One oversized NAL packetized into FU-A fragments must reassemble to
        // the exact original NAL.
        let mut nal = alloc::vec![0x65u8];
        nal.extend_from_slice(&(0..40u8).collect::<Vec<_>>());
        let au = wrap(&[&nal]);
        let mut pkt = RtpH264Packetizer::new(96, 1).with_max_payload(8);
        let packets = pkt.packetize(&au, 3000);
        assert!(packets.len() > 1, "precondition: NAL actually fragmented");

        let mut depay = RtpH264Depayloader::new();
        let mut out = None;
        for p in &packets {
            for unit in depay.depacketize(p) {
                out = Some(unit);
            }
        }
        assert_eq!(out.expect("reassembled AU").data, au);
    }

    #[test]
    fn parses_stap_a_aggregated_nals() {
        // Hand-build a STAP-A packet carrying SPS + PPS, marker set.
        let sps: &[u8] = &[0x67, 0x42, 0x00, 0x1f];
        let pps: &[u8] = &[0x68, 0xce, 0x38, 0x80];
        let mut payload = alloc::vec![STAP_A_TYPE | 0x60];
        for nal in [sps, pps] {
            payload.extend_from_slice(&(nal.len() as u16).to_be_bytes());
            payload.extend_from_slice(nal);
        }
        let mut packet = alloc::vec![0x80u8, 0x80 | 96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        packet.extend_from_slice(&payload);

        let mut depay = RtpH264Depayloader::new();
        let unit = depay.depacketize(&packet).next().expect("marker closes AU");
        assert_eq!(unit.data, wrap(&[sps, pps]), "both NALs in start-code form");
    }

    #[test]
    fn sequence_gap_drops_partial_access_unit() {
        // First an unmarked single-NAL packet (AU still open), then a packet
        // with a non-consecutive sequence: the partial AU is discarded so the
        // gap never welds the two together.
        let mut depay = RtpH264Depayloader::new();
        // seq 0, no marker: starts an AU.
        let p0 = alloc::vec![0x80u8, 96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x61, 0xAA];
        assert!(depay.depacketize(&p0).next().is_none());
        // seq 5 (gap), marker set, a fresh single NAL of the next access unit.
        let p1 = alloc::vec![
            0x80u8,
            0x80 | 96,
            0,
            5,
            0,
            0,
            0x0B,
            0xB8,
            0,
            0,
            0,
            0,
            0x65,
            0xBB
        ];
        let unit = depay.depacketize(&p1).next().expect("marker closes AU");
        assert_eq!(
            unit.data,
            wrap(&[&[0x65, 0xBB]]),
            "only the post-gap NAL survives; the dropped one is gone"
        );
    }

    fn single_nal_packet(seq: u16, ts: u32, marker: bool, nal: &[u8]) -> Vec<u8> {
        let mut p = alloc::vec![0x80u8, if marker { 0x80 | 96 } else { 96 }];
        p.extend_from_slice(&seq.to_be_bytes());
        p.extend_from_slice(&ts.to_be_bytes());
        p.extend_from_slice(&[0, 0, 0, 0]);
        p.extend_from_slice(nal);
        p
    }

    #[test]
    fn gap_inside_an_access_unit_drops_its_remaining_packets() {
        let mut depay = RtpH264Depayloader::new();
        assert!(depay
            .depacketize(&single_nal_packet(0, 9000, false, &[0x65, 1]))
            .next()
            .is_none());
        assert!(
            depay
                .depacketize(&single_nal_packet(2, 9000, true, &[0x65, 3]))
                .next()
                .is_none(),
            "the tail of an access unit that lost a packet is not emitted"
        );
        let unit = depay
            .depacketize(&single_nal_packet(3, 12000, true, &[0x41, 4]))
            .next()
            .expect("the next access unit is complete");
        assert_eq!(unit.data, wrap(&[&[0x41, 4]]));
        assert_eq!(unit.rtp_timestamp, 12000);
    }

    #[test]
    fn timestamp_change_closes_an_unmarked_access_unit() {
        let mut depay = RtpH264Depayloader::new();
        assert!(depay
            .depacketize(&single_nal_packet(0, 9000, false, &[0x65, 1]))
            .next()
            .is_none());
        let unit = depay
            .depacketize(&single_nal_packet(1, 12000, false, &[0x41, 2]))
            .next()
            .expect("the new timestamp closes the previous access unit");
        assert_eq!(unit.data, wrap(&[&[0x65, 1]]));
        assert_eq!(unit.rtp_timestamp, 9000);
        let units: Vec<_> = depay
            .depacketize(&single_nal_packet(2, 15000, true, &[0x41, 3]))
            .map(|unit| (unit.data, unit.rtp_timestamp))
            .collect();
        assert_eq!(
            units,
            [(wrap(&[&[0x41, 2]]), 12000), (wrap(&[&[0x41, 3]]), 15000)],
            "the timestamp closes the unmarked unit and the marker closes this one"
        );
    }

    fn fu_a_packet(
        seq: u16,
        ts: u32,
        marker: bool,
        start: bool,
        end: bool,
        frag: &[u8],
    ) -> Vec<u8> {
        let mut p = alloc::vec![0x80u8, if marker { 0x80 | 96 } else { 96 }];
        p.extend_from_slice(&seq.to_be_bytes());
        p.extend_from_slice(&ts.to_be_bytes());
        p.extend_from_slice(&[0, 0, 0, 0]); // ssrc
        p.push(0x60 | FU_A_TYPE); // FU indicator (F=0, NRI=3, type 28)
        let mut fu_header = 0x05u8; // type 5 (IDR slice)
        if start {
            fu_header |= 0x80;
        }
        if end {
            fu_header |= 0x40;
        }
        p.push(fu_header);
        p.extend_from_slice(frag);
        p
    }

    #[test]
    fn unterminated_fu_a_run_is_bounded_and_resyncs() {
        let mut depay = RtpH264Depayloader::new();
        let frag = alloc::vec![0xABu8; 1024 * 1024]; // 1 MiB per packet
                                                     // Start a fragment run, then never send the end bit: the buffer must cap
                                                     // and emit nothing instead of growing without bound.
        assert!(depay
            .depacketize(&fu_a_packet(0, 1, false, true, false, &frag))
            .next()
            .is_none());
        for seq in 1..16u16 {
            assert!(depay
                .depacketize(&fu_a_packet(seq, 1, false, false, false, &frag))
                .next()
                .is_none());
        }
        // A fresh complete FU-A NAL still reassembles after the run is dropped.
        assert!(depay
            .depacketize(&fu_a_packet(100, 2, false, true, false, &[1, 2, 3]))
            .next()
            .is_none());
        let unit = depay
            .depacketize(&fu_a_packet(101, 2, true, false, true, &[4, 5, 6]))
            .next()
            .expect("resyncs and completes a new AU");
        assert_eq!(unit.data, wrap(&[&[0x65, 1, 2, 3, 4, 5, 6]]));
    }

    #[test]
    fn h265_access_unit_round_trips_through_fragmentation() {
        let mut idr = alloc::vec![0x26u8, 0x01];
        idr.extend((0..200u8).collect::<Vec<_>>());
        let au = wrap(&[
            &[0x40, 0x01, 0x0C],
            &[0x42, 0x01, 0x01],
            &[0x44, 0x01, 0xC1],
            &idr,
        ]);
        let mut pay = crate::rtppay::RtpH265Packetizer::new(96, 5).with_max_payload(64);
        let packets = pay.packetize(&au, 9000);
        assert!(packets.len() > 4, "precondition: the IDR is fragmented");
        let mut depay = RtpH265Depayloader::new();
        let units: Vec<_> = packets.iter().flat_map(|p| depay.depacketize(p)).collect();
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].data, au);
        assert!(RtpEncoding::H265.is_keyframe(&units[0].data));
    }

    #[test]
    fn h265_aggregation_packet_splits_into_its_nals() {
        let vps: &[u8] = &[0x40, 0x01, 0x0C];
        let sps: &[u8] = &[0x42, 0x01, 0x01, 0x60];
        let mut payload = alloc::vec![H265_AP_TYPE << 1, 0x01];
        for nal in [vps, sps] {
            payload.extend_from_slice(&(nal.len() as u16).to_be_bytes());
            payload.extend_from_slice(nal);
        }
        let mut packet = alloc::vec![0x80u8, 0x80 | 96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        packet.extend_from_slice(&payload);
        let mut depay = RtpH265Depayloader::new();
        let unit = depay
            .depacketize(&packet)
            .next()
            .expect("marker closes the unit");
        assert_eq!(unit.data, wrap(&[vps, sps]));
    }

    #[test]
    fn opus_payload_round_trips_and_short_packets_drop() {
        let opus = [0xFCu8, 1, 2, 3];
        let packets = crate::rtppay::RtpOpusPacketizer::new(111, 9).packetize(&opus, 960);
        let mut depay = RtpDepayloader::new(&RtpEncoding::Opus);
        let unit = depay
            .depacketize(&packets[0])
            .into_iter()
            .next()
            .expect("one RTP packet carries one Opus packet");
        assert_eq!(unit.data, opus);
        assert_eq!(unit.rtp_timestamp, 960);

        let header_only = &packets[0][..g2g_core::rtp::RTP_HEADER_LEN];
        assert!(depay.depacketize(header_only).is_empty(), "empty payload");
        for len in 0..g2g_core::rtp::RTP_HEADER_LEN {
            assert!(
                depay.depacketize(&packets[0][..len]).is_empty(),
                "{len} bytes"
            );
        }
    }

    #[test]
    fn fu_a_run_cut_short_by_another_nal_is_dropped() {
        let mut d = RtpH264Depayloader::new();
        assert!(d
            .depacketize(&fu_a_packet(0, 9000, false, true, false, &[1, 2]))
            .next()
            .is_none());
        let mut single = alloc::vec![0x80u8, 0x80 | 96];
        single.extend_from_slice(&1u16.to_be_bytes());
        single.extend_from_slice(&9000u32.to_be_bytes());
        single.extend_from_slice(&[0, 0, 0, 0]);
        single.extend_from_slice(&[0x41, 7]);
        let au = d
            .depacketize(&single)
            .next()
            .expect("the marker closes the access unit");
        assert_eq!(au.data, wrap(&[&[0x41, 7]]), "only the complete NAL");
    }
}
