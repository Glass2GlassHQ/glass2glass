//! Sans-IO RIST (VSF TR-06-1 Simple Profile) pieces on top of [`rtcp`]: the
//! SSRC rule that marks a retransmission, the range NACK and RTT echo APP
//! packets, the sender's retransmission history and the receiver's NACK
//! schedule. `RistSink` / `RistSrc` (the `rist` feature) put them on sockets.

use alloc::collections::{BTreeMap, VecDeque};
#[cfg(feature = "rist")]
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::rtp::{RtpHeader, RTP_HEADER_LEN};

use crate::rtcp::{self, RtcpPacket};

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

#[derive(Debug)]
struct SentPacket {
    sent_ns: u64,
    header: RtpHeader,
    payload: Vec<u8>,
}

/// The sender's copy of every packet sent in the last `keep_ns`, indexed by
/// sequence number, for answering NACKs.
#[derive(Debug)]
pub struct RetransmissionHistory {
    keep_ns: u64,
    first_sequence: u16,
    packets: VecDeque<SentPacket>,
}

impl RetransmissionHistory {
    pub fn new(keep_ns: u64) -> Self {
        Self {
            keep_ns,
            first_sequence: 0,
            packets: VecDeque::new(),
        }
    }

    /// Remember a packet just sent at `now_ns` and forget those older than the
    /// history keeps. A sequence that does not follow the last one restarts it.
    pub fn record(&mut self, header: RtpHeader, payload: &[u8], now_ns: u64) {
        let expected = self.first_sequence.wrapping_add(self.packets.len() as u16);
        if self.packets.is_empty() || header.sequence != expected {
            self.packets.clear();
            self.first_sequence = header.sequence;
        }
        self.packets.push_back(SentPacket {
            sent_ns: now_ns,
            header,
            payload: payload.to_vec(),
        });
        let oldest_kept_ns = now_ns.saturating_sub(self.keep_ns);
        while let Some(oldest) = self.packets.front() {
            let expired = oldest.sent_ns < oldest_kept_ns;
            if !expired && self.packets.len() <= MAX_HISTORY_PACKETS {
                break;
            }
            self.packets.pop_front();
            self.first_sequence = self.first_sequence.wrapping_add(1);
        }
    }

    /// Retransmissions of every packet in `ranges` still held: the original
    /// bytes with the SSRC low bit set.
    pub fn retransmissions(&self, ranges: &[SequenceRange]) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        for range in ranges {
            for index in self.indices_in(*range) {
                let sent = &self.packets[index];
                let header = RtpHeader {
                    ssrc: retransmission_ssrc(sent.header.ssrc),
                    ..sent.header
                };
                let mut packet = Vec::with_capacity(RTP_HEADER_LEN + sent.payload.len());
                packet.extend_from_slice(&header.to_bytes());
                packet.extend_from_slice(&sent.payload);
                out.push(packet);
            }
        }
        out
    }

    /// History indices `range` covers. Its sequence numbers can wrap past
    /// 65535, so the covered span is up to two runs of indices.
    fn indices_in(&self, range: SequenceRange) -> impl Iterator<Item = usize> {
        let held = self.packets.len();
        let first = range.start.wrapping_sub(self.first_sequence) as usize;
        let last = first + range.additional as usize;
        let before_wrap = first..(last + 1).min(held);
        let after_wrap = 0..(last + 1).saturating_sub(SEQUENCE_SPACE).min(held);
        before_wrap.chain(after_wrap)
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

/// The properties `ristsink` and `ristsrc` share, under their GStreamer names.
#[cfg(feature = "rist")]
#[derive(Debug, Clone)]
pub(crate) struct LinkSettings {
    address: String,
    port: u16,
    min_rtcp_interval_ms: u64,
    pub(crate) cname: String,
}

#[cfg(feature = "rist")]
impl LinkSettings {
    pub(crate) fn new(address: &str, cname: String) -> Self {
        Self {
            address: address.to_string(),
            port: DEFAULT_PORT,
            min_rtcp_interval_ms: MAX_RTCP_INTERVAL_MS,
            cname,
        }
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
                    self.address = address.to_string();
                    Ok(())
                }
                None => Err(PropError::Type),
            },
            "port" => match value.as_uint() {
                Some(port) if (MIN_PORT..=MAX_PORT).contains(&port) && port.is_multiple_of(2) => {
                    self.port = port as u16;
                    Ok(())
                }
                Some(_) => Err(PropError::Value),
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
        match name {
            "address" => Some(PropValue::Str(self.address.clone())),
            "port" => Some(PropValue::Uint(u64::from(self.port))),
            "min-rtcp-interval" => Some(PropValue::Uint(self.min_rtcp_interval_ms)),
            "cname" => Some(PropValue::Str(self.cname.clone())),
            _ => None,
        }
    }

    /// The gap between two regular RTCP packets.
    pub(crate) fn rtcp_interval_ns(&self) -> u64 {
        self.min_rtcp_interval_ms * 1_000_000
    }

    /// The `address`:`port` pair, resolved (a host name is looked up).
    pub(crate) fn media_address(&self) -> std::io::Result<std::net::SocketAddr> {
        use std::net::ToSocketAddrs;
        (self.address.as_str(), self.port)
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::NotFound))
    }
}
