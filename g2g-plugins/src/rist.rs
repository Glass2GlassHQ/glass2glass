//! Sans-IO RIST (VSF TR-06-1 Simple Profile) pieces on top of [`rtcp`]: the
//! SSRC rule that marks a retransmission, the range NACK and RTT echo APP
//! packets, the sender's retransmission history, the receiver's NACK schedule
//! and the TR-06-2 RTP header extension (sequence number extension and null
//! packet deletion, what GStreamer's `ristrtpext` / `ristrtpdeext` do).
//! `RistSink` / `RistSrc` (the `rist` feature) put them on sockets.

use alloc::borrow::Cow;
use alloc::collections::{BTreeMap, BTreeSet};
#[cfg(feature = "rist")]
use alloc::format;
#[cfg(feature = "rist")]
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::rtp::{RtpHeader, RTP_HEADER_LEN};
use g2g_core::G2gError;

use crate::mpegts::{SYNC_BYTE, TS_PACKET_LEN};
use crate::rtcp::{self, RtcpPacket};
use crate::rtx::SentPackets;

/// The APP packet name every RIST message carries.
pub const RIST_APP_NAME: [u8; 4] = *b"RIST";
pub const SUBTYPE_RANGE_NACK: u8 = 0;
pub const SUBTYPE_RTT_ECHO_REQUEST: u8 = 2;
pub const SUBTYPE_RTT_ECHO_RESPONSE: u8 = 3;
/// RFC 3551 static payload type of MPEG-2 transport stream over RTP.
pub const MP2T_PAYLOAD_TYPE: u8 = 33;
/// TR-06-1 caps a range NACK packet at this many range requests.
pub const MAX_RANGES_PER_NACK: usize = 16;
/// Default UDP port of the RTP media, both GStreamer elements' default.
pub const DEFAULT_PORT: u16 = 5004;
/// TR-06-1 upper bound on the gap between two RTCP packets, in ms.
pub const MAX_RTCP_INTERVAL_MS: u64 = 100;
/// Longest a socket wait blocks before the RTCP and NACK timers are checked.
pub const TIMER_CHECK_MS: u64 = 5;
/// Largest UDP payload, the receive buffer both elements read into.
pub const MAX_DATAGRAM: usize = 65_507;

/// SSRC low bit: 0 on original packets, 1 on retransmissions.
const RETRANSMISSION_SSRC_BIT: u32 = 1;
const SSRC_LOW_BYTE: usize = RTP_HEADER_LEN - 1;
/// Bytes of one range request: a start sequence and a count of followers.
const RANGE_ENTRY_LEN: usize = 4;
/// Echo request body: a 64-bit timestamp, then the 32-bit processing delay.
const RTT_ECHO_TIMESTAMP_LEN: usize = 8;
const RTT_ECHO_DELAY_LEN: usize = 4;
/// Distinct 16-bit RTP sequence numbers.
const SEQUENCE_SPACE: usize = 1 << 16;
/// Keep at most half the sequence space, so a sequence number names one packet.
const MAX_HISTORY_PACKETS: usize = SEQUENCE_SPACE / 2;

/// The original-stream SSRC a packet or request belongs to.
pub fn media_ssrc(ssrc: u32) -> u32 {
    ssrc & !RETRANSMISSION_SSRC_BIT
}

pub fn retransmission_ssrc(ssrc: u32) -> u32 {
    ssrc | RETRANSMISSION_SSRC_BIT
}

pub fn is_retransmission(ssrc: u32) -> bool {
    ssrc & RETRANSMISSION_SSRC_BIT != 0
}

/// `start` and the `additional` sequence numbers after it, as a range NACK
/// entry carries them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceRange {
    pub start: u16,
    pub additional: u16,
}

/// Group sequence numbers, given in stream order, into runs of consecutive ones.
fn sequence_ranges(sequences: &[u16]) -> Vec<SequenceRange> {
    let mut ranges: Vec<SequenceRange> = Vec::new();
    for &sequence in sequences {
        if let Some(last) = ranges.last_mut() {
            let follows_last = last.start.wrapping_add(last.additional).wrapping_add(1) == sequence;
            if follows_last && last.additional < u16::MAX {
                last.additional += 1;
                continue;
            }
        }
        ranges.push(SequenceRange {
            start: sequence,
            additional: 0,
        });
    }
    ranges
}

/// Range NACK APP packets (subtype 0) asking `media_ssrc` for `missing`, at
/// most [`MAX_RANGES_PER_NACK`] ranges each, concatenated.
pub fn build_range_nack(media_ssrc: u32, missing: &[u16]) -> Vec<u8> {
    let ranges = sequence_ranges(missing);
    let mut out = Vec::new();
    for chunk in ranges.chunks(MAX_RANGES_PER_NACK) {
        let mut data = Vec::with_capacity(chunk.len() * RANGE_ENTRY_LEN);
        for range in chunk {
            data.extend_from_slice(&range.start.to_be_bytes());
            data.extend_from_slice(&range.additional.to_be_bytes());
        }
        out.extend(rtcp::build_app(
            SUBTYPE_RANGE_NACK,
            media_ssrc,
            RIST_APP_NAME,
            &data,
        ));
    }
    out
}

/// The NACK for `missing` in whichever form is shorter on the wire, generic
/// (RFC 4585) on a tie, the way GStreamer's `ristsrc` picks.
pub fn build_nack_feedback(sender_ssrc: u32, media_ssrc: u32, missing: &[u16]) -> Vec<u8> {
    let generic = rtcp::build_nack(sender_ssrc, media_ssrc, missing);
    let range = build_range_nack(media_ssrc, missing);
    if range.len() < generic.len() {
        range
    } else {
        generic
    }
}

/// The ranges a NACK in either form asks the stream `media` to resend. `None`
/// when the packet is not a NACK for that stream.
pub fn requested_ranges(packet: &RtcpPacket, media: u32) -> Option<Vec<SequenceRange>> {
    match packet {
        RtcpPacket::Nack {
            media_ssrc: target,
            missing,
            ..
        } if media_ssrc(*target) == media => Some(
            missing
                .iter()
                .map(|&start| SequenceRange {
                    start,
                    additional: 0,
                })
                .collect(),
        ),
        RtcpPacket::App {
            subtype: SUBTYPE_RANGE_NACK,
            ssrc,
            name: RIST_APP_NAME,
            data,
        } if media_ssrc(*ssrc) == media => Some(
            data.as_chunks::<RANGE_ENTRY_LEN>()
                .0
                .iter()
                .map(
                    |&[start_high, start_low, additional_high, additional_low]| SequenceRange {
                        start: u16::from_be_bytes([start_high, start_low]),
                        additional: u16::from_be_bytes([additional_high, additional_low]),
                    },
                )
                .collect(),
        ),
        _ => None,
    }
}

/// The RTT echo response to an echo request: the timestamp and padding come
/// back unchanged with a zero processing delay. `None` for any other packet.
pub fn rtt_echo_response(packet: &RtcpPacket) -> Option<Vec<u8>> {
    let RtcpPacket::App {
        subtype: SUBTYPE_RTT_ECHO_REQUEST,
        ssrc,
        name: RIST_APP_NAME,
        data,
    } = packet
    else {
        return None;
    };
    if data.len() < RTT_ECHO_TIMESTAMP_LEN + RTT_ECHO_DELAY_LEN {
        return None;
    }
    let mut body = data.clone();
    body[RTT_ECHO_TIMESTAMP_LEN..RTT_ECHO_TIMESTAMP_LEN + RTT_ECHO_DELAY_LEN].fill(0);
    Some(rtcp::build_app(
        SUBTYPE_RTT_ECHO_RESPONSE,
        *ssrc,
        RIST_APP_NAME,
        &body,
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SentOn {
    EveryLink,
    Link(usize),
}

impl SentOn {
    pub(crate) fn includes(self, link: usize) -> bool {
        match self {
            SentOn::EveryLink => true,
            SentOn::Link(sent_link) => sent_link == link,
        }
    }
}

#[derive(Debug)]
struct SentPacket {
    sent_ns: u64,
    sent_on: SentOn,
    packet: Vec<u8>,
}

/// The sender's copy of every packet sent in the last `keep_ns`, indexed by
/// sequence number, for answering NACKs.
#[derive(Debug)]
pub struct RetransmissionHistory {
    keep_ns: u64,
    packets: SentPackets<SentPacket>,
}

impl RetransmissionHistory {
    pub fn new(keep_ns: u64) -> Self {
        Self {
            keep_ns,
            packets: SentPackets::default(),
        }
    }

    /// Remember the RTP `packet` with sequence number `sequence` just sent at
    /// `now_ns` and forget those older than the history keeps. A sequence that
    /// does not follow the last one restarts it.
    pub fn record(&mut self, sequence: u16, packet: Vec<u8>, sent_on: SentOn, now_ns: u64) {
        self.packets.push(
            sequence,
            SentPacket {
                sent_ns: now_ns,
                sent_on,
                packet,
            },
        );
        let oldest_kept_ns = now_ns.saturating_sub(self.keep_ns);
        while let Some(oldest) = self.packets.oldest() {
            let expired = oldest.sent_ns < oldest_kept_ns;
            if !expired && self.packets.len() <= MAX_HISTORY_PACKETS {
                break;
            }
            self.packets.pop_oldest();
        }
    }

    /// Retransmissions of every packet in `ranges` still held that went out on
    /// `link`: the original bytes with the SSRC low bit set.
    pub fn retransmissions(&self, ranges: &[SequenceRange], link: usize) -> Vec<Vec<u8>> {
        // no honest receiver asks for more than the whole sequence space, or for one sequence twice
        let requested = ranges
            .iter()
            .flat_map(|range| {
                (0..=range.additional).map(move |offset| range.start.wrapping_add(offset))
            })
            .take(SEQUENCE_SPACE);
        let mut resent = BTreeSet::new();
        let mut out = Vec::new();
        for sequence in requested {
            if !resent.insert(sequence) {
                continue;
            }
            let Some(sent) = self.packets.get(sequence) else {
                continue;
            };
            if !sent.sent_on.includes(link) {
                continue;
            }
            let mut packet = sent.packet.clone();
            // the SSRC is the last word of the fixed header, big-endian
            packet[SSRC_LOW_BYTE] |= RETRANSMISSION_SSRC_BIT as u8;
            out.push(packet);
        }
        out
    }
}

#[derive(Debug, Clone, Copy)]
struct PendingLoss {
    next_request_ns: u64,
    requests_sent: u32,
}

/// When the receiver asks again for each missing packet: first after the
/// reorder section, then `max_requests - 1` more times spread over the rest of
/// the receiver buffer.
#[derive(Debug)]
pub struct NackScheduler {
    reorder_ns: u64,
    retry_interval_ns: u64,
    max_requests: u32,
    pending: BTreeMap<u16, PendingLoss>,
}

impl NackScheduler {
    pub fn new(reorder_ns: u64, buffer_ns: u64, max_requests: u32) -> Self {
        Self {
            reorder_ns,
            retry_interval_ns: buffer_ns.saturating_sub(reorder_ns)
                / u64::from(max_requests.max(1)),
            max_requests,
            pending: BTreeMap::new(),
        }
    }

    pub fn forget_losses(&mut self) {
        self.pending.clear();
    }

    /// The sequence numbers to request at `now_ns`, at most `limit`, given the
    /// receive buffer's current holes. A hole seen for the first time starts
    /// its reorder wait, and one that has been filled is forgotten.
    pub fn due(&mut self, missing: &[u16], now_ns: u64, limit: usize) -> Vec<u16> {
        let mut requests = Vec::new();
        let mut still_missing = BTreeMap::new();
        for &sequence in missing {
            let mut loss = self.pending.remove(&sequence).unwrap_or(PendingLoss {
                next_request_ns: now_ns.saturating_add(self.reorder_ns),
                requests_sent: 0,
            });
            let can_request = loss.requests_sent < self.max_requests && requests.len() < limit;
            if can_request && now_ns >= loss.next_request_ns {
                requests.push(sequence);
                loss.requests_sent += 1;
                loss.next_request_ns = now_ns.saturating_add(self.retry_interval_ns);
            }
            still_missing.insert(sequence, loss);
        }
        self.pending = still_missing;
        requests
    }
}

// "RI", the profile field of GStreamer's `ristrtpext`
pub const RIST_EXTENSION_PROFILE: u16 = u16::from_be_bytes(*b"RI");
const RIST_EXTENSION_WORDS: u16 = 1;
const EXTENSION_HEADER_LEN: usize = 4;
const RIST_EXTENSION_BODY_LEN: usize = 4;
const RTP_VERSION: u8 = 2;
const RTP_VERSION_SHIFT: u8 = 6;
const RTP_PADDING_BIT: u8 = 0x20;
const RTP_EXTENSION_BIT: u8 = 0x10;
const RTP_CSRC_COUNT_MASK: u8 = 0x0F;
const RTP_CSRC_LEN: usize = 4;
const NULL_PACKETS_DELETED_FLAG: u8 = 0x80;
const SEQUENCE_EXTENSION_FLAG: u8 = 0x40;
const ORIGINAL_COUNT_SHIFT: u8 = 3;
const ORIGINAL_COUNT_MASK: u8 = 0x07;
const DVB_PACKET_SIZE_FLAG: u8 = 0x80;
const DELETED_PACKETS_MASK: u8 = 0x7F;
// the deletion bitmap covers the first seven TS packets, the first in its high bit
const NULL_DELETION_SLOTS: usize = 7;
const DVB_TS_PACKET_LEN: usize = 204;
const TS_PID_HIGH_MASK: u8 = 0x1F;
const NULL_PACKET_PID: u16 = 0x1FFF;
// PID 0x1FFF, payload only, continuity counter zero, then zeros, as `ristrtpdeext` writes it
pub const RESTORED_NULL_PACKET_HEADER: [u8; 4] = [SYNC_BYTE, 0x1F, 0xFF, 0x10];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RistExtensionSettings {
    pub drop_null_packets: bool,
    pub sequence_number_extension: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RistExtension {
    pub null_packets_deleted: bool,
    pub original_packet_count: u8,
    pub dvb_packet_size: bool,
    pub deleted_packets: u8,
    pub sequence_high_bits: Option<u16>,
}

fn deletion_bit(index: usize) -> u8 {
    1 << (NULL_DELETION_SLOTS - 1 - index)
}

fn ts_packet_pid(ts_packet: &[u8]) -> u16 {
    u16::from_be_bytes([ts_packet[1] & TS_PID_HIGH_MASK, ts_packet[2]])
}

// GStreamer deletes and restores null packets only in a payload of whole 188 or 204 byte packets
fn whole_ts_packet_len(payload_len: usize) -> Option<usize> {
    [TS_PACKET_LEN, DVB_TS_PACKET_LEN]
        .into_iter()
        .find(|ts_packet_len| payload_len.is_multiple_of(*ts_packet_len))
}

// a TS packet without the sync byte fails the send, as in `ristrtpext`
fn delete_null_packets(payload: &[u8], ts_packet_len: usize) -> Result<(Vec<u8>, u8), G2gError> {
    let mut remaining = Vec::with_capacity(payload.len());
    let mut deleted_packets = 0u8;
    for (index, ts_packet) in payload.chunks_exact(ts_packet_len).enumerate() {
        if index < NULL_DELETION_SLOTS {
            if ts_packet[0] != SYNC_BYTE {
                return Err(G2gError::CapsMismatch);
            }
            if ts_packet_pid(ts_packet) == NULL_PACKET_PID {
                deleted_packets |= deletion_bit(index);
                continue;
            }
        }
        remaining.extend_from_slice(ts_packet);
    }
    Ok((remaining, deleted_packets))
}

// `header` carries the low half of the 32-bit sequence number, the extension the high half
pub fn build_media_packet(
    header: RtpHeader,
    sequence_high_bits: u16,
    payload: &[u8],
    settings: RistExtensionSettings,
) -> Result<Vec<u8>, G2gError> {
    let extension_len = EXTENSION_HEADER_LEN + RIST_EXTENSION_BODY_LEN;
    let mut packet = Vec::with_capacity(RTP_HEADER_LEN + extension_len + payload.len());
    packet.extend_from_slice(&header.to_bytes());
    if !settings.drop_null_packets && !settings.sequence_number_extension {
        packet.extend_from_slice(payload);
        return Ok(packet);
    }
    packet[0] |= RTP_EXTENSION_BIT;
    let mut flags = 0u8;
    let mut ts_flags = 0u8;
    let mut kept = Cow::Borrowed(payload);
    let deletable_ts_packet_len = whole_ts_packet_len(payload.len());
    if let (true, Some(ts_packet_len)) = (settings.drop_null_packets, deletable_ts_packet_len) {
        let (remaining, deleted_packets) = delete_null_packets(payload, ts_packet_len)?;
        let original_count = payload.len() / ts_packet_len;
        flags |= NULL_PACKETS_DELETED_FLAG;
        if original_count <= NULL_DELETION_SLOTS {
            flags |= (original_count as u8) << ORIGINAL_COUNT_SHIFT;
        }
        if ts_packet_len == DVB_TS_PACKET_LEN {
            ts_flags |= DVB_PACKET_SIZE_FLAG;
        }
        ts_flags |= deleted_packets;
        kept = Cow::Owned(remaining);
    }
    let mut sequence_bytes = [0u8; 2];
    if settings.sequence_number_extension {
        flags |= SEQUENCE_EXTENSION_FLAG;
        sequence_bytes = sequence_high_bits.to_be_bytes();
    }
    packet.extend_from_slice(&RIST_EXTENSION_PROFILE.to_be_bytes());
    packet.extend_from_slice(&RIST_EXTENSION_WORDS.to_be_bytes());
    packet.extend_from_slice(&[flags, ts_flags]);
    packet.extend_from_slice(&sequence_bytes);
    packet.extend_from_slice(&kept);
    Ok(packet)
}

// another profile's extension, or one of another length, is not a RIST one
fn find_rist_extension(packet: &[u8]) -> Option<(usize, RistExtension)> {
    let first = *packet.first()?;
    if first >> RTP_VERSION_SHIFT != RTP_VERSION || first & RTP_EXTENSION_BIT == 0 {
        return None;
    }
    let extension_start = RTP_HEADER_LEN + usize::from(first & RTP_CSRC_COUNT_MASK) * RTP_CSRC_LEN;
    let extension_header = packet.get(extension_start..extension_start + EXTENSION_HEADER_LEN)?;
    let profile = u16::from_be_bytes([extension_header[0], extension_header[1]]);
    let words = u16::from_be_bytes([extension_header[2], extension_header[3]]);
    if profile != RIST_EXTENSION_PROFILE || words != RIST_EXTENSION_WORDS {
        return None;
    }
    let body_start = extension_start + EXTENSION_HEADER_LEN;
    let body = packet.get(body_start..body_start + RIST_EXTENSION_BODY_LEN)?;
    let extension = RistExtension {
        null_packets_deleted: body[0] & NULL_PACKETS_DELETED_FLAG != 0,
        original_packet_count: (body[0] >> ORIGINAL_COUNT_SHIFT) & ORIGINAL_COUNT_MASK,
        dvb_packet_size: body[1] & DVB_PACKET_SIZE_FLAG != 0,
        deleted_packets: body[1] & DELETED_PACKETS_MASK,
        sequence_high_bits: (body[0] & SEQUENCE_EXTENSION_FLAG != 0)
            .then(|| u16::from_be_bytes([body[2], body[3]])),
    };
    Some((extension_start, extension))
}

pub fn parse_rist_extension(packet: &[u8]) -> Option<RistExtension> {
    find_rist_extension(packet).map(|(_, extension)| extension)
}

// the deleted bits and the payload length set the packet count, never the sender's count field
fn restore_null_packets<'a>(payload: &'a [u8], extension: &RistExtension) -> Option<Cow<'a, [u8]>> {
    let ts_packet_len = if payload.is_empty() {
        if extension.dvb_packet_size {
            DVB_TS_PACKET_LEN
        } else {
            TS_PACKET_LEN
        }
    } else {
        match whole_ts_packet_len(payload.len()) {
            Some(ts_packet_len) => ts_packet_len,
            None => return Some(Cow::Borrowed(payload)),
        }
    };
    let deleted_count = extension.deleted_packets.count_ones() as usize;
    let packet_count = payload.len() / ts_packet_len + deleted_count;
    let mut kept = payload.chunks_exact(ts_packet_len);
    let mut restored = Vec::with_capacity(packet_count * ts_packet_len);
    for index in 0..packet_count {
        let was_deleted =
            index < NULL_DELETION_SLOTS && extension.deleted_packets & deletion_bit(index) != 0;
        if was_deleted {
            restored.extend_from_slice(&RESTORED_NULL_PACKET_HEADER);
            restored.resize(
                restored.len() + ts_packet_len - RESTORED_NULL_PACKET_HEADER.len(),
                0,
            );
        } else {
            restored.extend_from_slice(kept.next()?);
        }
    }
    Some(Cow::Owned(restored))
}

#[derive(Debug, Default)]
pub struct RistExtensionRemover {
    highest_extended_sequence: u32,
}

impl RistExtensionRemover {
    // `None` drops the packet, as `ristrtpdeext` drops one far behind the highest sequence seen
    pub fn strip<'a>(&mut self, packet: &'a [u8]) -> Option<Cow<'a, [u8]>> {
        let parsed = RtpHeader::parse(packet)?;
        let Some((extension_start, extension)) = find_rist_extension(packet) else {
            return Some(Cow::Borrowed(packet));
        };
        if let Some(high_bits) = extension.sequence_high_bits {
            let extended = (u32::from(high_bits) << u16::BITS) | u32::from(parsed.header.sequence);
            let behind = self.highest_extended_sequence.saturating_sub(extended);
            if behind > i16::MAX as u32 {
                return None;
            }
            self.highest_extended_sequence = self.highest_extended_sequence.max(extended);
        }
        let payload_end = parsed.payload_offset + parsed.payload_len;
        let payload = &packet[parsed.payload_offset..payload_end];
        let payload = if extension.null_packets_deleted && extension.deleted_packets != 0 {
            restore_null_packets(payload, &extension)?
        } else {
            Cow::Borrowed(payload)
        };
        let mut stripped = Vec::with_capacity(extension_start + payload.len());
        stripped.extend_from_slice(&packet[..extension_start]);
        stripped[0] &= !(RTP_EXTENSION_BIT | RTP_PADDING_BIT);
        stripped.extend_from_slice(&payload);
        Some(Cow::Owned(stripped))
    }
}

/// A random 32-bit value from the OS, for an SSRC or a first sequence number.
#[cfg(feature = "rist")]
pub(crate) fn random_u32() -> u32 {
    let mut bytes = [0u8; 4];
    getrandom::getrandom(&mut bytes).expect("OS RNG for the RIST SSRC");
    u32::from_be_bytes(bytes)
}

/// A receive error that only reports an earlier datagram the peer's port
/// refused (Windows surfaces the ICMP reply this way), not a broken socket.
#[cfg(feature = "rist")]
pub(crate) fn is_peer_unreachable(error: &std::io::Error) -> bool {
    matches!(
        error.kind(),
        std::io::ErrorKind::ConnectionReset | std::io::ErrorKind::ConnectionRefused
    )
}

/// Lowest RTP port TR-06-1 allows.
#[cfg(feature = "rist")]
const MIN_PORT: u64 = 2;
/// Highest even port, so RTCP still fits on the odd port above it.
#[cfg(feature = "rist")]
const MAX_PORT: u64 = u16::MAX as u64 - 1;
/// An SDES item length is one byte.
#[cfg(feature = "rist")]
const MAX_CNAME_LEN: usize = u8::MAX as usize;

#[cfg(feature = "rist")]
const DEFAULT_MULTICAST_TTL: u8 = 1;

#[cfg(feature = "rist")]
fn valid_port(port: u64) -> Result<u16, g2g_core::PropError> {
    if (MIN_PORT..=MAX_PORT).contains(&port) && port.is_multiple_of(2) {
        Ok(port as u16)
    } else {
        Err(g2g_core::PropError::Value)
    }
}

// one bonded link, `address:port` with an optional `/interface` in `bonding-addresses`
#[cfg(feature = "rist")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LinkAddress {
    address: String,
    port: u16,
    multicast_iface: Option<String>,
}

#[cfg(feature = "rist")]
impl LinkAddress {
    fn parse(entry: &str) -> Result<Self, g2g_core::PropError> {
        use g2g_core::PropError;
        let (address_and_port, multicast_iface) = match entry.rsplit_once('/') {
            Some((head, iface)) => (head, Some(iface.trim())),
            None => (entry, None),
        };
        let (address, port) = address_and_port.rsplit_once(':').ok_or(PropError::Value)?;
        let port = valid_port(port.trim().parse().map_err(|_| PropError::Value)?)?;
        let address = address.trim();
        let iface_is_valid = multicast_iface.is_none_or(|iface| !iface.is_empty());
        if address.is_empty() || !iface_is_valid {
            return Err(PropError::Value);
        }
        Ok(Self {
            address: address.to_string(),
            port,
            multicast_iface: multicast_iface.map(str::to_string),
        })
    }

    fn to_entry(&self) -> String {
        match &self.multicast_iface {
            Some(iface) => format!("{}:{}/{iface}", self.address, self.port),
            None => format!("{}:{}", self.address, self.port),
        }
    }

    // a host name is looked up
    pub(crate) fn resolve(&self) -> std::io::Result<std::net::SocketAddr> {
        use std::net::ToSocketAddrs;
        (self.address.as_str(), self.port)
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::NotFound))
    }
}

/// The properties `ristsink` and `ristsrc` share, under their GStreamer names.
// `address`, `port` and `multicast-iface` set the first bonded link, as in GStreamer
#[cfg(feature = "rist")]
#[derive(Debug, Clone)]
pub(crate) struct LinkSettings {
    links: Vec<LinkAddress>,
    min_rtcp_interval_ms: u64,
    pub(crate) cname: String,
    multicast_ttl: u8,
    multicast_loopback: bool,
}

#[cfg(feature = "rist")]
impl LinkSettings {
    pub(crate) fn new(address: &str, cname: String) -> Self {
        Self {
            links: alloc::vec![LinkAddress {
                address: address.to_string(),
                port: DEFAULT_PORT,
                multicast_iface: None,
            }],
            min_rtcp_interval_ms: MAX_RTCP_INTERVAL_MS,
            cname,
            multicast_ttl: DEFAULT_MULTICAST_TTL,
            multicast_loopback: false,
        }
    }

    pub(crate) fn links(&self) -> &[LinkAddress] {
        &self.links
    }

    // `bonding-addresses` never stores an empty list
    fn first_link(&mut self) -> &mut LinkAddress {
        &mut self.links[0]
    }

    /// `Some` when `name` is one of the shared properties.
    pub(crate) fn set_property(
        &mut self,
        name: &str,
        value: &g2g_core::PropValue,
    ) -> Option<Result<(), g2g_core::PropError>> {
        use g2g_core::PropError;
        let result = match name {
            "address" => match value.as_str() {
                Some("") => Err(PropError::Value),
                Some(address) => {
                    self.first_link().address = address.to_string();
                    Ok(())
                }
                None => Err(PropError::Type),
            },
            "port" => match value.as_uint() {
                Some(port) => valid_port(port).map(|port| self.first_link().port = port),
                None => Err(PropError::Type),
            },
            "bonding-addresses" => match value.as_str() {
                Some(entries) => entries
                    .split(',')
                    .map(LinkAddress::parse)
                    .collect::<Result<Vec<_>, _>>()
                    .map(|links| self.links = links),
                None => Err(PropError::Type),
            },
            "multicast-iface" => match value.as_str() {
                Some(iface) => {
                    self.first_link().multicast_iface =
                        (!iface.is_empty()).then(|| iface.to_string());
                    Ok(())
                }
                None => Err(PropError::Type),
            },
            "multicast-ttl" => match value.as_int() {
                Some(ttl) => u8::try_from(ttl)
                    .map(|ttl| self.multicast_ttl = ttl)
                    .map_err(|_| PropError::Value),
                None => Err(PropError::Type),
            },
            "multicast-loopback" => match value.as_bool() {
                Some(loopback) => {
                    self.multicast_loopback = loopback;
                    Ok(())
                }
                None => Err(PropError::Type),
            },
            "min-rtcp-interval" => match value.as_uint() {
                Some(ms) if ms <= MAX_RTCP_INTERVAL_MS => {
                    self.min_rtcp_interval_ms = ms;
                    Ok(())
                }
                Some(_) => Err(PropError::Value),
                None => Err(PropError::Type),
            },
            "cname" => match value.as_str() {
                Some(cname) if cname.len() <= MAX_CNAME_LEN => {
                    self.cname = cname.to_string();
                    Ok(())
                }
                Some(_) => Err(PropError::Value),
                None => Err(PropError::Type),
            },
            _ => return None,
        };
        Some(result)
    }

    pub(crate) fn get_property(&self, name: &str) -> Option<g2g_core::PropValue> {
        use g2g_core::PropValue;
        let first = &self.links[0];
        match name {
            "address" => Some(PropValue::Str(first.address.clone())),
            "port" => Some(PropValue::Uint(u64::from(first.port))),
            "bonding-addresses" => {
                let entries: Vec<String> = self.links.iter().map(LinkAddress::to_entry).collect();
                Some(PropValue::Str(entries.join(",")))
            }
            "multicast-iface" => Some(PropValue::Str(
                first.multicast_iface.clone().unwrap_or_default(),
            )),
            "multicast-ttl" => Some(PropValue::Int(i64::from(self.multicast_ttl))),
            "multicast-loopback" => Some(PropValue::Bool(self.multicast_loopback)),
            "min-rtcp-interval" => Some(PropValue::Uint(self.min_rtcp_interval_ms)),
            "cname" => Some(PropValue::Str(self.cname.clone())),
            _ => None,
        }
    }

    /// The gap between two regular RTCP packets.
    pub(crate) fn rtcp_interval_ns(&self) -> u64 {
        self.min_rtcp_interval_ms * 1_000_000
    }

    // a multicast socket shares its port with the other group members on this host
    pub(crate) fn link_socket(
        &self,
        link: &LinkAddress,
        local: std::net::SocketAddr,
        group: Option<std::net::IpAddr>,
    ) -> std::io::Result<std::net::UdpSocket> {
        use socket2::{Domain, InterfaceIndexOrAddress, Protocol, Socket, Type};
        use std::net::{IpAddr, Ipv4Addr};
        let socket = Socket::new(Domain::for_address(local), Type::DGRAM, Some(Protocol::UDP))?;
        if group.is_some() {
            socket.set_reuse_address(true)?;
        }
        let interface_index = match &link.multicast_iface {
            Some(iface) => Some(bind_to_interface(&socket, iface)?),
            None => None,
        };
        socket.bind(&local.into())?;
        let ttl = u32::from(self.multicast_ttl);
        match group {
            None => {}
            Some(IpAddr::V4(group)) => {
                match interface_index {
                    Some(index) => socket
                        .join_multicast_v4_n(&group, &InterfaceIndexOrAddress::Index(index))?,
                    None => socket.join_multicast_v4(&group, &Ipv4Addr::UNSPECIFIED)?,
                }
                socket.set_multicast_ttl_v4(ttl)?;
                socket.set_multicast_loop_v4(self.multicast_loopback)?;
            }
            Some(IpAddr::V6(group)) => {
                // interface 0 lets the kernel pick
                let index = interface_index.unwrap_or(0);
                socket.join_multicast_v6(&group, index)?;
                socket.set_multicast_if_v6(index)?;
                socket.set_multicast_hops_v6(ttl)?;
                socket.set_multicast_loop_v6(self.multicast_loopback)?;
            }
        }
        socket.set_nonblocking(true)?;
        Ok(socket.into())
    }
}

#[cfg(feature = "rist")]
pub(crate) fn unspecified_address(like: std::net::SocketAddr, port: u16) -> std::net::SocketAddr {
    use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr};
    if like.is_ipv6() {
        SocketAddr::from((Ipv6Addr::UNSPECIFIED, port))
    } else {
        SocketAddr::from((Ipv4Addr::UNSPECIFIED, port))
    }
}

// SO_BINDTODEVICE, what GStreamer applies for `multicast-iface`
#[cfg(all(feature = "rist", any(target_os = "linux", target_os = "android")))]
fn bind_to_interface(socket: &socket2::Socket, name: &str) -> std::io::Result<u32> {
    socket.bind_device(Some(name.as_bytes()))?;
    let name = std::ffi::CString::new(name)
        .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
    // SAFETY: `name` is a NUL-terminated string that outlives the call, which only reads it.
    let index = unsafe { libc::if_nametoindex(name.as_ptr()) };
    if index == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(index)
}

#[cfg(all(feature = "rist", not(any(target_os = "linux", target_os = "android"))))]
fn bind_to_interface(_socket: &socket2::Socket, _name: &str) -> std::io::Result<u32> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "multicast-iface needs SO_BINDTODEVICE",
    ))
}

#[cfg(feature = "rist")]
pub(crate) async fn any_readable<'a, Sockets>(sockets: Sockets) -> std::io::Result<()>
where
    Sockets: Iterator<Item = &'a tokio::net::UdpSocket> + Clone,
{
    core::future::poll_fn(|context| {
        for socket in sockets.clone() {
            if let core::task::Poll::Ready(result) = socket.poll_recv_ready(context) {
                return core::task::Poll::Ready(result);
            }
        }
        core::task::Poll::Pending
    })
    .await
}
