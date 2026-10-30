//! Sans-IO SRT (Secure Reliable Transport) wire layer. Pure `no_std + alloc`,
//! no sockets. Builds / parses the SRT packet header and the control packets a
//! caller and listener exchange, per the SRT protocol draft
//! (draft-sharabayko-srt): a 16-byte header followed by a payload (data) or a
//! Control Information Field (control).
//!
//! Scope: the HSv5 caller / listener handshake (induction + conclusion with the
//! HSREQ latency extension and the optional Stream-ID extension), data packets,
//! and the reliability control packets (ACK / NAK loss-report / ACKACK /
//! KEEPALIVE / SHUTDOWN). The handshake driver and the ARQ reliability layer
//! that sit on this are [`SrtHandshake`] and [`SrtReceiver`] / [`SrtSender`].
//! Encryption (AES / KMREQ) and TSBPD delivery timing ([`SrtReceiver::set_tsbpd`])
//! are wired; congestion control is a follow-up.

use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::log::Target;

// libsrt's SRTO_PEERIDLETIMEO default
pub const PEER_IDLE_TIMEOUT_NS: u64 = 5_000_000_000;

/// The 16-byte SRT packet header precedes every packet.
pub const HEADER_LEN: usize = 16;

/// SRT protocol version advertised in the HSv5 handshake (1.4.2-style).
pub const SRT_VERSION: u32 = 0x0001_0402;
/// Handshake CIF version for the induction phase.
pub const HS_VERSION_INDUCTION: u32 = 4;
/// Handshake CIF version for HSv5.
pub const HS_VERSION_5: u32 = 5;
/// SRT magic in the induction extension field (`SRT_MAGIC_CODE`).
pub const SRT_MAGIC: u16 = 0x4A17;

// Handshake request types (the "Handshake Type" CIF field, as i32 on the wire).
pub const URQ_INDUCTION: u32 = 1;
pub const URQ_WAVEAHAND: u32 = 0;
pub const URQ_CONCLUSION: u32 = 0xFFFF_FFFF; // -1
pub const URQ_AGREEMENT: u32 = 0xFFFF_FFFE; // -2
                                            // a handshake type at or past this rejects the connection, the excess is the reason
const URQ_FAILURE_TYPES: u32 = 1000;
// libsrt's SRT_REJ_UNSECURE, a passphrase or key the listener requires is missing or different
const REJECT_UNSECURE: u32 = URQ_FAILURE_TYPES + 11;
// libsrt's SRT_REJX_NOTFOUND, the listener does not serve the stream id
const REJECT_STREAM_NOT_FOUND: u32 = URQ_FAILURE_TYPES + 1404;
// the KM byte that holds the stream key length in 4-byte words
const KM_KEY_LENGTH_OFFSET: usize = 15;

// Control packet types (the 15-bit Control Type field).
pub const CTRL_HANDSHAKE: u16 = 0x0000;
pub const CTRL_KEEPALIVE: u16 = 0x0001;
pub const CTRL_ACK: u16 = 0x0002;
pub const CTRL_NAK: u16 = 0x0003;
pub const CTRL_SHUTDOWN: u16 = 0x0005;
pub const CTRL_ACKACK: u16 = 0x0006;
pub const CTRL_DROPREQ: u16 = 0x0007;
/// User-defined / extended control type; SRT carries a mid-stream Keying
/// Material update (rekey) here, with the subtype in the type-specific field.
pub const CTRL_USER: u16 = 0x7FFF;

// SRT handshake extension command types (the extension TLV "type").
pub const EXT_HSREQ: u16 = 1;
pub const EXT_HSRSP: u16 = 2;
pub const EXT_KMREQ: u16 = 3;
pub const EXT_KMRSP: u16 = 4;
pub const EXT_SID: u16 = 5;
// libsrt rejects a longer stream id, SRT_STREAMID_MAX
pub const MAX_STREAM_ID_BYTES: usize = 512;
// Extension Field flag (in the handshake CIF) signalling HSREQ/KMREQ/CONFIG present.
pub const HS_EXT_FLAG_HSREQ: u16 = 0x0001;
pub const HS_EXT_FLAG_KMREQ: u16 = 0x0002;
pub const HS_EXT_FLAG_CONFIG: u16 = 0x0004;
// HSREQ SRT flags (SRT_OPT_*), the set libsrt sends in live mode
const SRT_OPT_TSBPDSND: u32 = 0x01;
const SRT_OPT_TSBPDRCV: u32 = 0x02;
const SRT_OPT_TLPKTDROP: u32 = 0x08;
const SRT_OPT_NAKREPORT: u32 = 0x10;
const SRT_OPT_REXMITFLG: u32 = 0x20;
const LIVE_MODE_FLAGS: u32 =
    SRT_OPT_TSBPDSND | SRT_OPT_TSBPDRCV | SRT_OPT_TLPKTDROP | SRT_OPT_NAKREPORT | SRT_OPT_REXMITFLG;

pub const FLOW_WINDOW_PACKETS: u32 = 8192;
// libsrt keeps its own RTT estimate when an ACK carries exactly these initial values.
pub const INITIAL_RTT_US: u32 = 100_000;
pub const INITIAL_RTT_VARIANCE_US: u32 = INITIAL_RTT_US / 2;
const LIGHT_ACK_CIF_LEN: usize = 4;
// sequence, RTT, RTT variance, available buffer
const FULL_ACK_CIF_LEN: usize = 16;
// first and last dropped sequence
const DROP_REQUEST_CIF_LEN: usize = 8;
const SEQUENCE_MASK: u32 = 0x7FFF_FFFF;
// The receiver runs inside srtsrc, so its lines go under that element's category.
const RECEIVER_LOG_CATEGORY: &str = "srtsrc";

/// Whether a packet is a control packet (the top bit of the first word).
pub fn is_control(buf: &[u8]) -> bool {
    !buf.is_empty() && buf[0] & 0x80 != 0
}

/// Build a data packet: 31-bit sequence number, 26-bit message number (with the
/// PP=11 "solo" position and the order bit set), retransmit flag, timestamp,
/// destination socket id, then the payload.
pub fn build_data_packet(
    seq: u32,
    msg_no: u32,
    retransmit: bool,
    kk: u8,
    timestamp: u32,
    dst_socket_id: u32,
    payload: &[u8],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
    // word0: F=0 then the 31-bit sequence number.
    out.extend_from_slice(&(seq & 0x7FFF_FFFF).to_be_bytes());
    // word1: PP(2)=11 | O(1)=1 | KK(2) | R(1) | MsgNo(26). KK=00 cleartext,
    // 01 even key (the only encrypted case here).
    let mut word1 = (0b11u32 << 30) | (1u32 << 29) | ((kk as u32 & 0b11) << 27);
    if retransmit {
        word1 |= 1 << 26;
    }
    word1 |= msg_no & 0x03FF_FFFF;
    out.extend_from_slice(&word1.to_be_bytes());
    out.extend_from_slice(&timestamp.to_be_bytes());
    out.extend_from_slice(&dst_socket_id.to_be_bytes());
    out.extend_from_slice(payload);
    out
}

/// A parsed data packet: its sequence number, retransmit flag, and payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataPacket {
    pub seq: u32,
    pub retransmit: bool,
    /// Encryption key flag (KK): 0 cleartext, 1 even key.
    pub kk: u8,
    pub timestamp: u32,
    pub dst_socket_id: u32,
    pub payload: Vec<u8>,
}

/// Parse a data packet. `None` if it is too short or is a control packet.
pub fn parse_data_packet(buf: &[u8]) -> Option<DataPacket> {
    if buf.len() < HEADER_LEN || is_control(buf) {
        return None;
    }
    let seq = u32::from_be_bytes(buf[0..4].try_into().ok()?) & 0x7FFF_FFFF;
    let word1 = u32::from_be_bytes(buf[4..8].try_into().ok()?);
    let retransmit = word1 & (1 << 26) != 0;
    let kk = ((word1 >> 27) & 0b11) as u8;
    let timestamp = u32::from_be_bytes(buf[8..12].try_into().ok()?);
    let dst_socket_id = u32::from_be_bytes(buf[12..16].try_into().ok()?);
    Some(DataPacket {
        seq,
        retransmit,
        kk,
        timestamp,
        dst_socket_id,
        payload: buf[HEADER_LEN..].to_vec(),
    })
}

/// The control packets this implementation builds / parses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Control {
    /// HSv5 handshake (induction or conclusion).
    Handshake(Handshake),
    Keepalive,
    /// Full ACK: the acknowledged packet sequence number (everything before it
    /// is received), with the ACK sub-sequence number in the type-specific field.
    // libsrt sets its flow window to available_buffer on every full ACK.
    Ack {
        ack_no: u32,
        ack_seq: u32,
        rtt_us: u32,
        rtt_variance_us: u32,
        available_buffer: u32,
    },
    // libsrt shrinks its flow window by the acknowledged span on each light ACK.
    LightAck {
        ack_seq: u32,
    },
    /// ACKACK: confirms an ACK by its sub-sequence number.
    AckAck {
        ack_no: u32,
    },
    /// Loss report: the sequence numbers (or ranges) the receiver is missing.
    Nak {
        loss: Vec<u32>,
    },
    Shutdown,
    /// Mid-stream Keying Material update (rekey): the opaque KM blob (the new
    /// wrapped key + salt for the even or odd slot, `srtcrypto` builds /
    /// reads it). `rsp` distinguishes a request (KMREQ) from a response (KMRSP).
    KeyMaterial {
        rsp: bool,
        km: Vec<u8>,
    },
    // the sender no longer retransmits first_sequence..=last_sequence
    DropRequest {
        message_number: u32,
        first_sequence: u32,
        last_sequence: u32,
    },
}

/// The HSv5 handshake Control Information Field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Handshake {
    pub version: u32,
    pub encryption: u16,
    pub ext_field: u16,
    pub init_seq: u32,
    pub mtu: u32,
    pub flow_window: u32,
    pub hs_type: u32,
    pub srt_socket_id: u32,
    pub syn_cookie: u32,
    /// Peer IP (16 bytes, IPv4-mapped in the first 4 for v4).
    pub peer_ip: [u8; 16],
    /// HSREQ/HSRSP extension latency (ms), present in conclusion handshakes.
    pub latency_ms: Option<u16>,
    // SRT_OPT_* flags sent beside the latency
    pub srt_flags: u32,
    /// Stream-ID extension (the SRT `streamid`), present when set.
    pub stream_id: Option<String>,
    /// KMREQ/KMRSP extension: the Keying Material message bytes (the wrapped
    /// stream key + salt), present when the stream is encrypted. Opaque here;
    /// `srtcrypto` builds and interprets it.
    pub km: Option<Vec<u8>>,
    /// Whether this is a *response* handshake (a listener answering a caller's
    /// conclusion). The HSREQ/KMREQ extensions are then emitted as HSRSP/KMRSP:
    /// libsrt strictly distinguishes request from response TLV types (it rejects
    /// a KMREQ in a connect response with "KMREQ extracted without passing target
    /// buffer"), so a listener must not echo the caller's KM as a KMREQ.
    pub is_response: bool,
}

impl Handshake {
    /// A baseline induction request from a caller (no extensions yet).
    pub fn induction(socket_id: u32, init_seq: u32) -> Self {
        Self {
            version: HS_VERSION_INDUCTION,
            encryption: 0,
            ext_field: SRT_MAGIC,
            init_seq,
            mtu: 1500,
            flow_window: FLOW_WINDOW_PACKETS,
            hs_type: URQ_INDUCTION,
            srt_socket_id: socket_id,
            syn_cookie: 0,
            peer_ip: [0; 16],
            latency_ms: None,
            srt_flags: 0,
            stream_id: None,
            km: None,
            is_response: false,
        }
    }
}

/// Build a control packet. `dst_socket_id` and `timestamp` go in the header.
pub fn build_control(ctrl: &Control, timestamp: u32, dst_socket_id: u32) -> Vec<u8> {
    let (ctrl_type, type_info, cif) = match ctrl {
        Control::Handshake(hs) => (CTRL_HANDSHAKE, 0, build_handshake_cif(hs)),
        Control::Keepalive => (CTRL_KEEPALIVE, 0, Vec::new()),
        Control::Ack {
            ack_no,
            ack_seq,
            rtt_us,
            rtt_variance_us,
            available_buffer,
        } => {
            let mut cif = Vec::with_capacity(FULL_ACK_CIF_LEN);
            for word in [ack_seq, rtt_us, rtt_variance_us, available_buffer] {
                cif.extend_from_slice(&word.to_be_bytes());
            }
            (CTRL_ACK, *ack_no, cif)
        }
        Control::LightAck { ack_seq } => (CTRL_ACK, 0, ack_seq.to_be_bytes().to_vec()),
        Control::AckAck { ack_no } => (CTRL_ACKACK, *ack_no, Vec::new()),
        Control::Nak { loss } => (CTRL_NAK, 0, build_nak_cif(loss)),
        Control::Shutdown => (CTRL_SHUTDOWN, 0, Vec::new()),
        // Subtype (type-specific field): KMREQ (3) or KMRSP (4); CIF is the KM blob.
        Control::KeyMaterial { rsp, km } => (
            CTRL_USER,
            if *rsp {
                EXT_KMRSP as u32
            } else {
                EXT_KMREQ as u32
            },
            km.clone(),
        ),
        Control::DropRequest {
            message_number,
            first_sequence,
            last_sequence,
        } => {
            let mut cif = Vec::with_capacity(DROP_REQUEST_CIF_LEN);
            cif.extend_from_slice(&first_sequence.to_be_bytes());
            cif.extend_from_slice(&last_sequence.to_be_bytes());
            (CTRL_DROPREQ, *message_number, cif)
        }
    };
    let mut out = Vec::with_capacity(HEADER_LEN + cif.len());
    // word0: F=1 | Control Type (15 bits) | Subtype (16 bits, 0 here).
    let word0 = 0x8000_0000u32 | ((ctrl_type as u32) << 16);
    out.extend_from_slice(&word0.to_be_bytes());
    out.extend_from_slice(&type_info.to_be_bytes());
    out.extend_from_slice(&timestamp.to_be_bytes());
    out.extend_from_slice(&dst_socket_id.to_be_bytes());
    out.extend_from_slice(&cif);
    out
}

/// Parse a control packet. `None` if it is not a control packet or is malformed.
pub fn parse_control(buf: &[u8]) -> Option<Control> {
    if buf.len() < HEADER_LEN || !is_control(buf) {
        return None;
    }
    let word0 = u32::from_be_bytes(buf[0..4].try_into().ok()?);
    let ctrl_type = ((word0 >> 16) & 0x7FFF) as u16;
    let type_info = u32::from_be_bytes(buf[4..8].try_into().ok()?);
    let cif = &buf[HEADER_LEN..];
    // Only called where a guard has checked the CIF length.
    let cif_word = |offset: usize| u32::from_be_bytes(cif[offset..offset + 4].try_into().unwrap());
    match ctrl_type {
        CTRL_HANDSHAKE => Some(Control::Handshake(parse_handshake_cif(cif)?)),
        CTRL_KEEPALIVE => Some(Control::Keepalive),
        CTRL_ACK if cif.len() == LIGHT_ACK_CIF_LEN => Some(Control::LightAck {
            ack_seq: cif_word(0),
        }),
        CTRL_ACK if cif.len() >= FULL_ACK_CIF_LEN => Some(Control::Ack {
            ack_no: type_info,
            ack_seq: cif_word(0),
            rtt_us: cif_word(4),
            rtt_variance_us: cif_word(8),
            available_buffer: cif_word(12),
        }),
        CTRL_ACKACK => Some(Control::AckAck { ack_no: type_info }),
        CTRL_NAK => Some(Control::Nak {
            loss: parse_nak_cif(cif),
        }),
        CTRL_SHUTDOWN => Some(Control::Shutdown),
        CTRL_DROPREQ if cif.len() >= DROP_REQUEST_CIF_LEN => Some(Control::DropRequest {
            message_number: type_info,
            first_sequence: cif_word(0) & SEQUENCE_MASK,
            last_sequence: cif_word(4) & SEQUENCE_MASK,
        }),
        CTRL_USER if type_info == EXT_KMREQ as u32 || type_info == EXT_KMRSP as u32 => {
            Some(Control::KeyMaterial {
                rsp: type_info == EXT_KMRSP as u32,
                km: cif.to_vec(),
            })
        }
        _ => None,
    }
}

/// Build the HSv5 handshake CIF, appending the HSREQ + Stream-ID extensions when
/// the latency / stream id are set (the conclusion handshake).
fn build_handshake_cif(hs: &Handshake) -> Vec<u8> {
    let mut cif = Vec::new();
    cif.extend_from_slice(&hs.version.to_be_bytes());
    cif.extend_from_slice(&hs.encryption.to_be_bytes());
    cif.extend_from_slice(&hs.ext_field.to_be_bytes());
    cif.extend_from_slice(&hs.init_seq.to_be_bytes());
    cif.extend_from_slice(&hs.mtu.to_be_bytes());
    cif.extend_from_slice(&hs.flow_window.to_be_bytes());
    cif.extend_from_slice(&hs.hs_type.to_be_bytes());
    cif.extend_from_slice(&hs.srt_socket_id.to_be_bytes());
    cif.extend_from_slice(&hs.syn_cookie.to_be_bytes());
    cif.extend_from_slice(&hs.peer_ip);

    // HSREQ / HSRSP extension: SRT version, flags, then receiver + sender TSBPD delay.
    if let Some(latency) = hs.latency_ms {
        let hs_cmd = if hs.is_response { EXT_HSRSP } else { EXT_HSREQ };
        cif.extend_from_slice(&hs_cmd.to_be_bytes());
        cif.extend_from_slice(&3u16.to_be_bytes()); // length in 32-bit words
        cif.extend_from_slice(&SRT_VERSION.to_be_bytes());
        cif.extend_from_slice(&hs.srt_flags.to_be_bytes());
        // recv TSBPD delay (high 16) | send TSBPD delay (low 16).
        cif.extend_from_slice(&((latency as u32) << 16 | latency as u32).to_be_bytes());
    }
    // KMREQ / KMRSP extension: the opaque Keying Material blob, padded to a
    // 32-bit boundary. A listener answers the caller's KMREQ with a KMRSP (same
    // blob, response TLV type); libsrt rejects a KMREQ in a connect response.
    if let Some(km) = &hs.km {
        let words = km.len().div_ceil(4);
        let km_cmd = if hs.is_response { EXT_KMRSP } else { EXT_KMREQ };
        cif.extend_from_slice(&km_cmd.to_be_bytes());
        cif.extend_from_slice(&(words as u16).to_be_bytes());
        let mut padded = km.clone();
        padded.resize(words * 4, 0);
        cif.extend_from_slice(&padded);
    }
    // Stream-ID extension: the id padded with zeros to a 32-bit boundary, each
    // 32-bit word byte-reversed on the wire, as libsrt sends it.
    if let Some(sid) = &hs.stream_id {
        let bytes = sid.as_bytes();
        let words = bytes.len().div_ceil(4);
        cif.extend_from_slice(&EXT_SID.to_be_bytes());
        cif.extend_from_slice(&(words as u16).to_be_bytes());
        let mut padded = bytes.to_vec();
        padded.resize(words * 4, 0);
        reverse_word_bytes(&mut padded);
        cif.extend_from_slice(&padded);
    }
    cif
}

fn reverse_word_bytes(bytes: &mut [u8]) {
    for word in bytes.as_chunks_mut::<4>().0 {
        word.reverse();
    }
}

/// Parse the HSv5 handshake CIF and any HSREQ / Stream-ID extensions.
fn parse_handshake_cif(cif: &[u8]) -> Option<Handshake> {
    if cif.len() < 48 {
        return None;
    }
    let be32 = |o: usize| u32::from_be_bytes(cif[o..o + 4].try_into().unwrap());
    let be16 = |o: usize| u16::from_be_bytes(cif[o..o + 2].try_into().unwrap());
    let mut peer_ip = [0u8; 16];
    peer_ip.copy_from_slice(&cif[32..48]);
    let mut hs = Handshake {
        version: be32(0),
        encryption: be16(4),
        ext_field: be16(6),
        init_seq: be32(8),
        mtu: be32(12),
        flow_window: be32(16),
        hs_type: be32(20),
        srt_socket_id: be32(24),
        syn_cookie: be32(28),
        peer_ip,
        latency_ms: None,
        srt_flags: 0,
        stream_id: None,
        km: None,
        is_response: false,
    };

    // Walk the extension TLVs that follow the fixed CIF.
    let mut at = 48;
    while at + 4 <= cif.len() {
        let ext_type = u16::from_be_bytes(cif[at..at + 2].try_into().unwrap());
        let words = u16::from_be_bytes(cif[at + 2..at + 4].try_into().unwrap()) as usize;
        let body = at + 4;
        let end = body + words * 4;
        if end > cif.len() {
            break;
        }
        match ext_type {
            EXT_HSREQ | EXT_HSRSP if words >= 3 => {
                hs.srt_flags = u32::from_be_bytes(cif[body + 4..body + 8].try_into().unwrap());
                let tsbpd = u32::from_be_bytes(cif[body + 8..body + 12].try_into().unwrap());
                hs.latency_ms = Some((tsbpd >> 16) as u16);
                hs.is_response |= ext_type == EXT_HSRSP;
            }
            EXT_KMREQ | EXT_KMRSP => {
                hs.km = Some(cif[body..end].to_vec());
                hs.is_response |= ext_type == EXT_KMRSP;
            }
            EXT_SID => {
                let mut raw = cif[body..end].to_vec();
                reverse_word_bytes(&mut raw);
                let trimmed: Vec<u8> = raw.into_iter().take_while(|&b| b != 0).collect();
                if let Ok(s) = String::from_utf8(trimmed) {
                    hs.stream_id = Some(s);
                }
            }
            _ => {}
        }
        at = end;
    }
    Some(hs)
}

/// Build a NAK loss-report CIF. A single lost sequence is one 32-bit word with
/// the high bit clear; a contiguous range is two words, the first with the high
/// bit set (range start) and the second the range end (inclusive).
fn build_nak_cif(loss: &[u32]) -> Vec<u8> {
    let mut cif = Vec::new();
    let mut i = 0;
    while i < loss.len() {
        let start = loss[i];
        let mut end = start;
        while i + 1 < loss.len() && loss[i + 1] == end + 1 {
            end += 1;
            i += 1;
        }
        if end == start {
            cif.extend_from_slice(&(start & 0x7FFF_FFFF).to_be_bytes());
        } else {
            cif.extend_from_slice(&((start & 0x7FFF_FFFF) | 0x8000_0000).to_be_bytes());
            cif.extend_from_slice(&(end & 0x7FFF_FFFF).to_be_bytes());
        }
        i += 1;
    }
    cif
}

/// Upper bound on a materialized loss list. A real NAK never exceeds the flow
/// window (thousands of packets); the cap stops an attacker-chosen range or a
/// far-ahead sequence from expanding to billions of entries (OOM / CPU DoS).
const MAX_LOSS_LIST: usize = 1 << 16;

/// Expand a NAK loss-report CIF back into the explicit list of lost sequences.
fn parse_nak_cif(cif: &[u8]) -> Vec<u32> {
    let mut loss = Vec::new();
    let mut i = 0;
    while i + 4 <= cif.len() && loss.len() < MAX_LOSS_LIST {
        let word = u32::from_be_bytes(cif[i..i + 4].try_into().unwrap());
        i += 4;
        if word & 0x8000_0000 != 0 {
            // Range start; the next word is the inclusive end.
            let start = word & 0x7FFF_FFFF;
            if i + 4 <= cif.len() {
                let end = u32::from_be_bytes(cif[i..i + 4].try_into().unwrap()) & 0x7FFF_FFFF;
                i += 4;
                for s in start..=end {
                    loss.push(s);
                    if loss.len() >= MAX_LOSS_LIST {
                        break;
                    }
                }
            } else {
                loss.push(start);
            }
        } else {
            loss.push(word & 0x7FFF_FFFF);
        }
    }
    loss
}

/// Caller (connects out) or listener (accepts) role for the handshake.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Caller,
    Listener,
    Rendezvous(RendezvousState),
}

// libsrt's HSv5 rendezvous, where both sides call and a cookie contest picks the side that sends the HSREQ
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RendezvousState {
    Waving,
    Initiator,
    Responder,
    // answered the initiator's HSREQ, waiting for its agreement or its first other packet
    ResponderAnswered,
}

// libsrt's cookie contest: the side whose cookie is larger by 32-bit signed difference sends the HSREQ, a tie decides nothing
fn rendezvous_role(own_cookie: u32, peer_cookie: u32) -> Option<RendezvousState> {
    let contest = i64::from(own_cookie as i32) - i64::from(peer_cookie as i32);
    if contest & 0xFFFF_FFFF == 0 {
        return None;
    }
    if contest & 0x8000_0000 != 0 {
        return Some(RendezvousState::Responder);
    }
    Some(RendezvousState::Initiator)
}

/// What the handshake driver wants the I/O layer to do after a step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandshakeStep {
    /// A packet to send to the peer, if any.
    pub reply: Option<Vec<u8>>,
    /// The connection is now established.
    pub established: bool,
}

/// Sans-IO SRT HSv5 handshake driver. Feed received handshake packets to
/// [`on_packet`](Self::on_packet) and send the [`reply`](HandshakeStep::reply);
/// the caller kicks off with [`start`](Self::start). On completion the peer's
/// socket id (the destination for data packets) and initial sequence are known.
#[derive(Debug)]
pub struct SrtHandshake {
    role: Role,
    socket_id: u32,
    init_seq: u32,
    latency_ms: u16,
    stream_id: Option<String>,
    cookie: u32,
    peer_socket_id: u32,
    peer_init_seq: u32,
    established: bool,
    /// Keying Material this side advertises (the caller's wrapped key; the
    /// listener echoes the caller's). Opaque bytes; the element builds/parses it.
    km: Option<Vec<u8>>,
    /// Keying Material received from the peer (the caller's KM, read by the
    /// listener to derive the shared key).
    peer_km: Option<Vec<u8>>,
    peer_srt_flags: u32,
    // a listener answers a caller with another stream id or key length with a rejection
    required_stream_id: Option<String>,
    required_key_length: Option<usize>,
    rejection: Option<u32>,
}

// the stream key length in bytes a KM message carries
pub fn km_key_length(km: &[u8]) -> Option<usize> {
    km.get(KM_KEY_LENGTH_OFFSET)
        .map(|&words| usize::from(words) * 4)
}

impl SrtHandshake {
    /// `km` is the caller's Keying Material blob (from
    /// `srtcrypto`) for an encrypted stream, or `None` for cleartext.
    pub fn new_caller(
        socket_id: u32,
        init_seq: u32,
        latency_ms: u16,
        stream_id: Option<String>,
        km: Option<Vec<u8>>,
    ) -> Self {
        Self {
            role: Role::Caller,
            socket_id,
            init_seq,
            latency_ms,
            stream_id,
            cookie: 0,
            peer_socket_id: 0,
            peer_init_seq: 0,
            established: false,
            km,
            peer_km: None,
            peer_srt_flags: 0,
            required_stream_id: None,
            required_key_length: None,
            rejection: None,
        }
    }

    // `cookie` decides the contest for the initiator role, so it must differ from the peer's
    pub fn new_rendezvous(
        socket_id: u32,
        init_seq: u32,
        latency_ms: u16,
        stream_id: Option<String>,
        km: Option<Vec<u8>>,
        cookie: u32,
    ) -> Self {
        Self {
            role: Role::Rendezvous(RendezvousState::Waving),
            cookie,
            ..Self::new_caller(socket_id, init_seq, latency_ms, stream_id, km)
        }
    }

    /// `cookie` is the SYN cookie the listener offers in its induction reply and
    /// later validates is echoed in the conclusion. It must be an unpredictable
    /// nonzero value (the I/O layer seeds it from a clock / random source); a
    /// value derivable from the public socket id gives an off-path attacker the
    /// cookie for free and defeats the handshake's anti-spoof check.
    pub fn new_listener(socket_id: u32, latency_ms: u16, cookie: u32) -> Self {
        Self {
            role: Role::Listener,
            socket_id,
            init_seq: 0,
            latency_ms,
            stream_id: None,
            cookie,
            peer_socket_id: 0,
            peer_init_seq: 0,
            established: false,
            km: None,
            peer_km: None,
            peer_srt_flags: 0,
            required_stream_id: None,
            required_key_length: None,
            rejection: None,
        }
    }

    /// The Keying Material received from the peer, if the stream is encrypted.
    /// The listener parses this (with its passphrase) to recover the stream key.
    pub fn peer_km(&self) -> Option<&[u8]> {
        self.peer_km.as_deref()
    }

    pub fn admit_only(&mut self, stream_id: Option<String>, key_length: Option<usize>) {
        self.required_stream_id = stream_id;
        self.required_key_length = key_length;
    }

    // the handshake type of the rejection the peer answered with
    pub fn rejection(&self) -> Option<u32> {
        self.rejection
    }

    // the peer sets the retransmit bit on resent data packets, so a late packet without it was reordered
    pub fn peer_marks_retransmissions(&self) -> bool {
        self.peer_srt_flags & SRT_OPT_REXMITFLG != 0
    }

    pub fn is_established(&self) -> bool {
        self.established
    }

    /// The peer's SRT socket id (the destination socket id for data packets).
    pub fn peer_socket_id(&self) -> u32 {
        self.peer_socket_id
    }

    /// The peer's initial data sequence number.
    pub fn peer_init_seq(&self) -> u32 {
        self.peer_init_seq
    }

    // the larger of the two sides' latencies, which both sides apply
    pub fn latency_ms(&self) -> u16 {
        self.latency_ms
    }

    // the sequence number this side's first data packet carries
    pub fn init_seq(&self) -> u32 {
        self.init_seq
    }

    /// The caller's opening induction packet (listener returns `None` and waits).
    pub fn start(&self) -> Option<Vec<u8>> {
        match self.role {
            Role::Caller => Some(build_control(
                &Control::Handshake(Handshake::induction(self.socket_id, self.init_seq)),
                0,
                0,
            )),
            Role::Listener => None,
            Role::Rendezvous(_) => Some(build_control(
                &Control::Handshake(self.bare(URQ_WAVEAHAND)),
                0,
                0,
            )),
        }
    }

    /// Advance the handshake on a received packet.
    pub fn on_packet(&mut self, buf: &[u8]) -> HandshakeStep {
        let none = HandshakeStep {
            reply: None,
            established: self.established,
        };
        let Some(Control::Handshake(hs)) = parse_control(buf) else {
            // the initiator sends data once connected, which stands in for a lost agreement
            if self.role == Role::Rendezvous(RendezvousState::ResponderAnswered) {
                self.established = true;
            }
            return HandshakeStep {
                reply: None,
                established: self.established,
            };
        };
        // the conclusion and agreement types are negative on the wire, a rejection positive
        if i32::try_from(hs.hs_type).is_ok_and(|t| t >= URQ_FAILURE_TYPES as i32) {
            self.rejection = Some(hs.hs_type);
            return none;
        }
        match (self.role, hs.hs_type) {
            // Listener: a caller's induction -> reply induction with our cookie.
            (Role::Listener, t) if t == URQ_INDUCTION => {
                self.peer_socket_id = hs.srt_socket_id;
                let mut resp = Handshake::induction(self.socket_id, 0);
                resp.version = HS_VERSION_5;
                resp.syn_cookie = self.cookie;
                HandshakeStep {
                    reply: Some(build_control(
                        &Control::Handshake(resp),
                        0,
                        self.peer_socket_id,
                    )),
                    established: false,
                }
            }
            // Caller: the listener's induction response -> send conclusion.
            (Role::Caller, t) if t == URQ_INDUCTION => {
                // The listener's induction response echoes our own socket id in
                // its CIF (libsrt does this), so it is NOT the listener's id; the
                // listener's real id arrives in its conclusion response (below),
                // which is what data packets must target. During the handshake
                // packets are addressed to socket id 0: libsrt demuxes the
                // conclusion to its listening socket by dst == 0 and DROPS a
                // conclusion addressed to any other id (as belonging to no socket).
                self.cookie = hs.syn_cookie;
                let concl = self.conclusion();
                HandshakeStep {
                    reply: Some(build_control(&Control::Handshake(concl), 0, 0)),
                    established: false,
                }
            }
            // Listener: the caller's conclusion -> validate cookie, reply, done.
            (Role::Listener, t) if t == URQ_CONCLUSION => {
                if hs.syn_cookie != self.cookie {
                    return none; // a stale / spoofed conclusion; ignore
                }
                if let Some(reason) = self.refusal(&hs) {
                    let mut rejection = Handshake::induction(self.socket_id, 0);
                    rejection.version = HS_VERSION_5;
                    rejection.ext_field = 0;
                    rejection.hs_type = reason;
                    rejection.syn_cookie = self.cookie;
                    return HandshakeStep {
                        reply: Some(build_control(
                            &Control::Handshake(rejection),
                            0,
                            hs.srt_socket_id,
                        )),
                        established: false,
                    };
                }
                self.peer_socket_id = hs.srt_socket_id;
                self.peer_init_seq = hs.init_seq;
                // a libsrt caller refuses a conclusion with another ISN, and both directions start there
                self.init_seq = hs.init_seq;
                self.peer_srt_flags = hs.srt_flags;
                if let Some(l) = hs.latency_ms {
                    self.latency_ms = self.latency_ms.max(l);
                }
                // Echo the caller's Keying Material as our KMRSP, and keep it for
                // the element to derive the shared key from.
                self.peer_km = hs.km.clone();
                self.km = hs.km;
                self.established = true;
                let concl = self.conclusion();
                HandshakeStep {
                    reply: Some(build_control(
                        &Control::Handshake(concl),
                        0,
                        self.peer_socket_id,
                    )),
                    established: true,
                }
            }
            // Caller: the listener's conclusion response -> established.
            (Role::Caller, t) if t == URQ_CONCLUSION => {
                // The listener's real socket id (its conclusion carries it), the
                // destination for our subsequent data packets.
                self.peer_socket_id = hs.srt_socket_id;
                self.peer_init_seq = hs.init_seq;
                self.peer_srt_flags = hs.srt_flags;
                if let Some(l) = hs.latency_ms {
                    self.latency_ms = self.latency_ms.max(l);
                }
                self.peer_km = hs.km; // the listener's KMRSP (our echoed KM)
                self.established = true;
                HandshakeStep {
                    reply: None,
                    established: true,
                }
            }
            (Role::Rendezvous(state), _) => self.on_rendezvous_packet(state, hs),
            _ => none,
        }
    }

    // the HSv5 rendezvous flow of libsrt's handshake document, which tolerates any packet getting lost
    fn on_rendezvous_packet(&mut self, state: RendezvousState, hs: Handshake) -> HandshakeStep {
        self.peer_socket_id = hs.srt_socket_id;
        let state = match state {
            RendezvousState::Waving => match rendezvous_role(self.cookie, hs.syn_cookie) {
                Some(role) => role,
                None => {
                    return HandshakeStep {
                        reply: None,
                        established: false,
                    }
                }
            },
            decided => decided,
        };
        self.role = Role::Rendezvous(state);
        let carries_hs_extension = hs.latency_ms.is_some();
        let reply = match (state, hs.hs_type) {
            (RendezvousState::Initiator, t) if t == URQ_CONCLUSION && carries_hs_extension => {
                self.take_peer_parameters(&hs);
                self.established = true;
                Some(self.bare(URQ_AGREEMENT))
            }
            (RendezvousState::Initiator, t) if t == URQ_WAVEAHAND || t == URQ_CONCLUSION => {
                Some(self.conclusion())
            }
            (RendezvousState::Responder | RendezvousState::ResponderAnswered, t)
                if t == URQ_CONCLUSION && carries_hs_extension =>
            {
                self.take_peer_parameters(&hs);
                // the KMRSP echoes the initiator's key, as a listener's does
                self.km = hs.km;
                self.role = Role::Rendezvous(RendezvousState::ResponderAnswered);
                Some(self.conclusion())
            }
            (RendezvousState::ResponderAnswered, t) if t == URQ_AGREEMENT => {
                self.established = true;
                None
            }
            (RendezvousState::Responder, t) if t == URQ_WAVEAHAND || t == URQ_CONCLUSION => {
                Some(self.bare(URQ_CONCLUSION))
            }
            _ => None,
        };
        HandshakeStep {
            reply: reply.map(|hs| build_control(&Control::Handshake(hs), 0, self.peer_socket_id)),
            established: self.established,
        }
    }

    fn take_peer_parameters(&mut self, hs: &Handshake) {
        self.peer_init_seq = hs.init_seq;
        self.peer_srt_flags = hs.srt_flags;
        self.peer_km = hs.km.clone();
        if let Some(latency) = hs.latency_ms {
            self.latency_ms = self.latency_ms.max(latency);
        }
    }

    // a handshake of `hs_type` with no extensions
    fn bare(&self, hs_type: u32) -> Handshake {
        Handshake {
            version: HS_VERSION_5,
            ext_field: 0,
            hs_type,
            syn_cookie: self.cookie,
            ..Handshake::induction(self.socket_id, self.init_seq)
        }
    }

    fn refusal(&self, conclusion: &Handshake) -> Option<u32> {
        if self
            .required_stream_id
            .as_ref()
            .is_some_and(|id| conclusion.stream_id.as_ref() != Some(id))
        {
            return Some(REJECT_STREAM_NOT_FOUND);
        }
        let key_length = conclusion.km.as_deref().and_then(km_key_length);
        if self
            .required_key_length
            .is_some_and(|required| key_length != Some(required))
        {
            return Some(REJECT_UNSECURE);
        }
        None
    }

    /// Build this side's conclusion handshake (carries HSREQ latency + SID).
    fn conclusion(&self) -> Handshake {
        let ext_field = HS_EXT_FLAG_HSREQ
            | if self.stream_id.is_some() {
                HS_EXT_FLAG_CONFIG
            } else {
                0
            }
            | if self.km.is_some() {
                HS_EXT_FLAG_KMREQ
            } else {
                0
            };
        let responding = matches!(
            self.role,
            Role::Listener
                | Role::Rendezvous(RendezvousState::Responder | RendezvousState::ResponderAnswered)
        );
        // gosrt (mediamtx) rejects a caller whose request lacks the live-mode flags
        let srt_flags = if responding {
            // libsrt's listener asks for the retransmit bit only from a caller that offered it
            LIVE_MODE_FLAGS & (self.peer_srt_flags | !SRT_OPT_REXMITFLG)
        } else {
            LIVE_MODE_FLAGS
        };
        Handshake {
            version: HS_VERSION_5,
            encryption: if self.km.is_some() { 1 } else { 0 },
            ext_field,
            init_seq: self.init_seq,
            mtu: 1500,
            flow_window: FLOW_WINDOW_PACKETS,
            hs_type: URQ_CONCLUSION,
            srt_socket_id: self.socket_id,
            syn_cookie: self.cookie,
            peer_ip: [0; 16],
            latency_ms: Some(self.latency_ms),
            srt_flags,
            stream_id: self.stream_id.clone(),
            km: self.km.clone(),
            // A listener's conclusion answers the caller's request, so its
            // HSREQ/KMREQ extensions become HSRSP/KMRSP (libsrt is strict here).
            is_response: responding,
        }
    }
}

/// Sans-IO reliable SRT sender: assigns sequence numbers, keeps a bounded send
/// buffer, and retransmits on a NAK loss report (the ARQ the receiver drives).
#[derive(Debug)]
pub struct SrtSender {
    dst_socket_id: u32,
    next_seq: u32,
    msg_no: u32,
    /// Buffered packets, so a NAK resends the same ciphertext under the same
    /// sequence/IV *and the same key parity* even after a rekey has switched the
    /// active key (the receiver keeps both slots).
    buffer: alloc::collections::VecDeque<SentPacket>,
    capacity: usize,
    retransmits: u64,
    /// Which key slot stamps outgoing packets (`KK`): 0 cleartext, 1 even, 2 odd.
    active_kk: u8,
    /// The two stream-key slots; rekeying alternates the active one between them
    /// so in-flight packets under the previous key still decrypt. `None` until
    /// the KM is negotiated (cleartext).
    #[cfg(feature = "srt")]
    crypto_even: Option<crate::srtcrypto::SrtCrypto>,
    #[cfg(feature = "srt")]
    crypto_odd: Option<crate::srtcrypto::SrtCrypto>,
}

#[derive(Debug)]
struct SentPacket {
    seq: u32,
    kk: u8,
    // a receiver schedules the resent copy by this, as libsrt keeps the original
    timestamp: u32,
    // post-encryption bytes
    data: Vec<u8>,
}

impl SrtSender {
    pub fn new(dst_socket_id: u32, init_seq: u32, capacity: usize) -> Self {
        Self {
            dst_socket_id,
            next_seq: init_seq & 0x7FFF_FFFF,
            msg_no: 1,
            buffer: alloc::collections::VecDeque::new(),
            capacity: capacity.max(1),
            retransmits: 0,
            active_kk: 0,
            #[cfg(feature = "srt")]
            crypto_even: None,
            #[cfg(feature = "srt")]
            crypto_odd: None,
        }
    }

    /// Encrypt outgoing payloads with the negotiated stream key (the even slot;
    /// the active key at connection time).
    #[cfg(feature = "srt")]
    pub fn set_crypto(&mut self, crypto: crate::srtcrypto::SrtCrypto) {
        self.crypto_even = Some(crypto);
        self.active_kk = crate::srtcrypto::KM_KK_EVEN;
    }

    /// Install a stream key into the slot named by `kk` (`KM_KK_EVEN` /
    /// `KM_KK_ODD`) without changing which slot is active (the rekey pre-announce).
    #[cfg(feature = "srt")]
    pub fn install_key(&mut self, kk: u8, crypto: crate::srtcrypto::SrtCrypto) {
        match kk {
            crate::srtcrypto::KM_KK_EVEN => self.crypto_even = Some(crypto),
            crate::srtcrypto::KM_KK_ODD => self.crypto_odd = Some(crypto),
            _ => {}
        }
    }

    /// The active key parity stamped on outgoing packets.
    pub fn active_kk(&self) -> u8 {
        self.active_kk
    }

    /// Rekey: install `new` into the *inactive* slot, switch the active key to it,
    /// and return the KMREQ control packet announcing it (send this before the
    /// first packet under the new key so the receiver can install it). Subsequent
    /// [`send`](Self::send) packets carry the new parity; the previous key stays
    /// live for retransmits until the next rekey overwrites its slot.
    #[cfg(feature = "srt")]
    pub fn rekey(&mut self, new: crate::srtcrypto::SrtCrypto, passphrase: &str) -> Vec<u8> {
        let next_kk = if self.active_kk == crate::srtcrypto::KM_KK_ODD {
            crate::srtcrypto::KM_KK_EVEN
        } else {
            crate::srtcrypto::KM_KK_ODD
        };
        let km = new.build_km(passphrase, next_kk);
        self.install_key(next_kk, new);
        self.active_kk = next_kk;
        build_control(
            &Control::KeyMaterial { rsp: false, km },
            0,
            self.dst_socket_id,
        )
    }

    /// The active key slot, if any.
    #[cfg(feature = "srt")]
    fn active_crypto(&self) -> Option<&crate::srtcrypto::SrtCrypto> {
        match self.active_kk {
            crate::srtcrypto::KM_KK_EVEN => self.crypto_even.as_ref(),
            crate::srtcrypto::KM_KK_ODD => self.crypto_odd.as_ref(),
            _ => None,
        }
    }

    /// Wire-encode `payload` for sequence `seq`: encrypt with the active key when
    /// the stream is keyed, else pass through. Returns the bytes to put on the
    /// wire (and to buffer for retransmit).
    #[cfg(feature = "srt")]
    fn encode_payload(&self, seq: u32, payload: &[u8]) -> Vec<u8> {
        let mut data = payload.to_vec();
        if let Some(c) = self.active_crypto() {
            c.process(seq, &mut data);
        }
        data
    }
    #[cfg(not(feature = "srt"))]
    fn encode_payload(&self, _seq: u32, payload: &[u8]) -> Vec<u8> {
        payload.to_vec()
    }

    pub fn retransmits(&self) -> u64 {
        self.retransmits
    }

    /// Wrap `payload` as the next data packet, buffering it for possible resend.
    pub fn send(&mut self, payload: &[u8], timestamp: u32) -> Vec<u8> {
        let seq = self.next_seq;
        self.next_seq = (self.next_seq + 1) & 0x7FFF_FFFF;
        let msg_no = self.msg_no;
        self.msg_no = self.msg_no.wrapping_add(1) & 0x03FF_FFFF;
        let kk = self.active_kk;
        let data = self.encode_payload(seq, payload);
        let pkt = build_data_packet(seq, msg_no, false, kk, timestamp, self.dst_socket_id, &data);
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(SentPacket {
            seq,
            kk,
            timestamp,
            data,
        });
        pkt
    }

    /// React to a control packet from the receiver: NAK triggers retransmits
    /// (the R flag set), ACK trims the buffer and is answered with an ACKACK
    /// stamped `control_timestamp`. Returns packets to send.
    pub fn on_control(&mut self, ctrl: &Control, control_timestamp: u32) -> Vec<Vec<u8>> {
        match ctrl {
            Control::Nak { loss } => {
                let mut out = Vec::new();
                for &seq in loss {
                    // Resend with the parity the packet was originally encrypted
                    // under, not the current active key.
                    if let Some(sent) = self.buffered(seq) {
                        out.push(build_data_packet(
                            seq,
                            0,
                            true,
                            sent.kk,
                            sent.timestamp,
                            self.dst_socket_id,
                            &sent.data,
                        ));
                        self.retransmits += 1;
                    }
                }
                out
            }
            Control::LightAck { ack_seq } => {
                self.release_acknowledged(*ack_seq);
                Vec::new()
            }
            Control::Ack {
                ack_no, ack_seq, ..
            } => {
                self.release_acknowledged(*ack_seq);
                // libsrt's receiver measures RTT from this and tracks clock drift by its timestamp
                alloc::vec![build_control(
                    &Control::AckAck { ack_no: *ack_no },
                    control_timestamp,
                    self.dst_socket_id,
                )]
            }
            _ => Vec::new(),
        }
    }

    pub fn all_acknowledged(&self) -> bool {
        self.buffer.is_empty()
    }

    // everything before ack_seq is received
    fn release_acknowledged(&mut self, ack_seq: u32) {
        while self
            .buffer
            .front()
            .is_some_and(|sent| !seq_ge(sent.seq, ack_seq))
        {
            self.buffer.pop_front();
        }
    }

    /// The buffered packet `seq`, found by its offset from the oldest one since
    /// the buffer holds consecutive sequences in send order.
    fn buffered(&self, seq: u32) -> Option<&SentPacket> {
        let oldest = self.buffer.front()?.seq;
        let entry = self.buffer.get(seq_distance(seq, oldest) as usize)?;
        (entry.seq == seq).then_some(entry)
    }
}

/// Sans-IO reliable SRT receiver: reorders by sequence, reports gaps as a NAK
/// loss list, and delivers payloads in order. Mirrors the RTP jitter design.
///
/// TSBPD (Timestamp-Based Packet Delivery) is optional: with [`set_tsbpd`] the
/// receiver holds each in-order packet until its scheduled delivery time
/// (anchored so the first packet is released the configured latency after it
/// becomes deliverable, every later packet inheriting that fixed buffering delay
/// off the sender timestamps), smoothing network jitter into a steady output.
/// Without it, [`take_ready`] delivers as soon as a packet is in order.
///
/// [`set_tsbpd`]: Self::set_tsbpd
/// [`take_ready`]: Self::take_ready
#[derive(Debug)]
pub struct SrtReceiver {
    next_deliver: u32,
    have_base: bool,
    /// seq -> (payload, sender packet timestamp in microseconds).
    pending: alloc::collections::BTreeMap<u32, (Vec<u8>, u32)>,
    max_seen: u32,
    /// TSBPD target latency in microseconds (the held buffering delay); `None`
    /// delivers in order with no timing gate.
    tsbpd_latency_us: Option<u32>,
    // set by the first packet TSBPD schedules, then moved to each released one
    tsbpd_anchor: Option<TsbpdAnchor>,
    /// The two stream-key slots, selected per packet by its `KK` flag (even/odd),
    /// so a rekey can roll the active key while packets under the previous key
    /// still decrypt. `None` is cleartext.
    #[cfg(feature = "srt")]
    crypto_even: Option<crate::srtcrypto::SrtCrypto>,
    #[cfg(feature = "srt")]
    crypto_odd: Option<crate::srtcrypto::SrtCrypto>,
    // libsrt's SRTO_LOSSMAXTTL, the most packets a new gap waits for a reordered packet before its NAK
    loss_max_ttl: u32,
    // packets the next new gap waits, between 0 and loss_max_ttl
    reorder_tolerance: u32,
    held_losses: alloc::collections::VecDeque<HeldLoss>,
    ordered_streak: u32,
    early_reorder_streak: u32,
}

// libsrt lowers the reorder tolerance by one after this many packets in a row arrive in order
const ORDERED_PACKETS_PER_TOLERANCE_STEP: u32 = 50;
// or after this many reordered packets arrive with more than EARLY_REORDER_PACKETS_LEFT of their wait unused
const EARLY_REORDERS_PER_TOLERANCE_STEP: u32 = 10;
const EARLY_REORDER_PACKETS_LEFT: u32 = 2;

// a gap whose NAK waits until packets_left more packets arrive, libsrt's CRcvFreshLoss
#[derive(Debug, Clone, Copy)]
struct HeldLoss {
    first: u32,
    last: u32,
    packets_left: u32,
}

#[derive(Debug, Clone, Copy)]
struct TsbpdAnchor {
    timestamp: u32,
    delivery_us: i64,
}

impl TsbpdAnchor {
    // the sender timestamp wraps every 71 minutes, packets in flight are seconds apart
    fn delivery_us(&self, timestamp: u32) -> i64 {
        self.delivery_us + i64::from(timestamp.wrapping_sub(self.timestamp) as i32)
    }
}

impl Default for SrtReceiver {
    fn default() -> Self {
        Self::new()
    }
}

impl SrtReceiver {
    pub fn new() -> Self {
        Self {
            next_deliver: 0,
            have_base: false,
            pending: alloc::collections::BTreeMap::new(),
            max_seen: 0,
            tsbpd_latency_us: None,
            tsbpd_anchor: None,
            #[cfg(feature = "srt")]
            crypto_even: None,
            #[cfg(feature = "srt")]
            crypto_odd: None,
            loss_max_ttl: 0,
            reorder_tolerance: 0,
            held_losses: alloc::collections::VecDeque::new(),
            ordered_streak: 0,
            early_reorder_streak: 0,
        }
    }

    // only for a peer that marks retransmissions, otherwise every retransmission looks reordered
    pub fn set_loss_max_ttl(&mut self, packets: u32) {
        self.loss_max_ttl = packets;
        self.reorder_tolerance = packets;
    }

    /// Enable TSBPD with `latency_us` of buffering delay: [`take_ready_at`] then
    /// gates delivery on each packet's scheduled time instead of releasing it the
    /// moment it is in order. The advertised handshake latency drives this.
    ///
    /// [`take_ready_at`]: Self::take_ready_at
    pub fn set_tsbpd(&mut self, latency_us: u32) {
        self.tsbpd_latency_us = Some(latency_us);
    }

    /// Decrypt incoming payloads with the negotiated stream key (the even slot;
    /// the active key at connection time).
    #[cfg(feature = "srt")]
    pub fn set_crypto(&mut self, crypto: crate::srtcrypto::SrtCrypto) {
        self.crypto_even = Some(crypto);
    }

    /// Install a stream key into the slot named by `kk` (`KM_KK_EVEN` /
    /// `KM_KK_ODD`), the receiver side of a mid-stream rekey: a packet stamped
    /// with that parity then decrypts under this key.
    #[cfg(feature = "srt")]
    pub fn install_key(&mut self, kk: u8, crypto: crate::srtcrypto::SrtCrypto) {
        match kk {
            crate::srtcrypto::KM_KK_EVEN => self.crypto_even = Some(crypto),
            crate::srtcrypto::KM_KK_ODD => self.crypto_odd = Some(crypto),
            _ => {}
        }
    }

    /// Buffer a received data packet, decrypting its payload in place (keyed by
    /// its sequence) when the stream is encrypted. Returns the sequences to NAK
    /// now: a new gap at once, or a held gap whose wait for reordered packets ran out.
    pub fn on_data(&mut self, pkt: DataPacket) -> Vec<u32> {
        if !self.have_base {
            self.next_deliver = pkt.seq;
            // libsrt starts its highest-received sequence one before the first packet
            self.max_seen = pkt.seq.wrapping_sub(1) & SEQUENCE_MASK;
            self.have_base = true;
        }
        if !seq_ge(pkt.seq, self.next_deliver) {
            return Vec::new();
        }
        let tolerance = self.reorder_tolerance;
        let first_gap_seq = (self.max_seen + 1) & SEQUENCE_MASK;
        let gap = seq_gt(pkt.seq, first_gap_seq)
            .then(|| (first_gap_seq, pkt.seq.wrapping_sub(1) & SEQUENCE_MASK));
        let arrived_in_order = seq_gt(pkt.seq, self.max_seen);
        if arrived_in_order {
            self.max_seen = pkt.seq;
        } else {
            self.on_late_packet(pkt.seq, pkt.retransmit);
        }
        if self.pending.contains_key(&pkt.seq) {
            return Vec::new();
        }
        let payload = pkt.payload;
        // Decrypt in place using the key slot the packet's KK flag selects
        // (even/odd), so a rekeyed stream stays readable across the switch.
        // A no-op rebind for a cleartext stream (KK=0 / no installed key).
        #[cfg(feature = "srt")]
        let payload = {
            let mut p = payload;
            let slot = match pkt.kk {
                crate::srtcrypto::KM_KK_EVEN => self.crypto_even.as_ref(),
                crate::srtcrypto::KM_KK_ODD => self.crypto_odd.as_ref(),
                _ => None,
            };
            if let Some(c) = slot {
                c.process(pkt.seq, &mut p);
            }
            p
        };
        self.pending.insert(pkt.seq, (payload, pkt.timestamp));

        let mut nak = Vec::new();
        if let Some((first, last)) = gap {
            if tolerance == 0 {
                nak.extend(self.missing_in(first, last));
            } else {
                self.held_losses.push_back(HeldLoss {
                    first,
                    last,
                    packets_left: tolerance,
                });
            }
        }
        if tolerance > 0 {
            nak.extend(self.release_held_losses());
        }
        if arrived_in_order || pkt.retransmit {
            self.count_ordered_packet();
        }
        nak.truncate(MAX_LOSS_LIST);
        nak
    }

    // a late packet without the retransmit bit was reordered on the way, so later gaps wait longer
    fn on_late_packet(&mut self, seq: u32, retransmit: bool) {
        let reordered = !retransmit;
        let mut tolerance_raised = false;
        if reordered {
            let distance = seq_distance(self.max_seen, seq);
            if distance > self.reorder_tolerance {
                self.reorder_tolerance = distance.min(self.loss_max_ttl);
                tolerance_raised = true;
            }
        }
        if self.reorder_tolerance == 0 {
            return;
        }
        let packets_left = self.remove_held_loss(seq);
        if !reordered {
            return;
        }
        self.ordered_streak = 0;
        if tolerance_raised {
            self.early_reorder_streak = 0;
            return;
        }
        if !packets_left.is_some_and(|left| left > EARLY_REORDER_PACKETS_LEFT) {
            return;
        }
        self.early_reorder_streak += 1;
        if self.early_reorder_streak >= EARLY_REORDERS_PER_TOLERANCE_STEP {
            self.early_reorder_streak = 0;
            self.reorder_tolerance = self.reorder_tolerance.saturating_sub(1);
        }
    }

    fn count_ordered_packet(&mut self) {
        self.ordered_streak += 1;
        if self.ordered_streak >= ORDERED_PACKETS_PER_TOLERANCE_STEP {
            self.ordered_streak = 0;
            self.reorder_tolerance = self.reorder_tolerance.saturating_sub(1);
        }
    }

    // returns the packets the held gap containing seq had left to wait
    fn remove_held_loss(&mut self, seq: u32) -> Option<u32> {
        let index = self
            .held_losses
            .iter()
            .position(|loss| seq_ge(seq, loss.first) && seq_ge(loss.last, seq))?;
        let loss = self.held_losses[index];
        let before = seq.wrapping_sub(1) & SEQUENCE_MASK;
        let after = (seq + 1) & SEQUENCE_MASK;
        match (seq == loss.first, seq == loss.last) {
            (true, true) => {
                self.held_losses.remove(index);
            }
            (true, false) => self.held_losses[index].first = after,
            (false, true) => self.held_losses[index].last = before,
            (false, false) => {
                self.held_losses[index].last = before;
                self.held_losses.insert(
                    index + 1,
                    HeldLoss {
                        first: after,
                        ..loss
                    },
                );
            }
        }
        Some(loss.packets_left)
    }

    // libsrt NAKs the held gaps at the front whose wait ran out and counts down the rest
    fn release_held_losses(&mut self) -> Vec<u32> {
        let mut nak = Vec::new();
        while let Some(&loss) = self.held_losses.front() {
            if loss.packets_left > 0 {
                break;
            }
            self.held_losses.pop_front();
            nak.extend(self.missing_in(loss.first, loss.last));
        }
        for loss in &mut self.held_losses {
            loss.packets_left = loss.packets_left.saturating_sub(1);
        }
        nak
    }

    /// Pop every payload now deliverable in order (stops at the first gap). No
    /// timing gate: use [`take_ready_at`](Self::take_ready_at) for TSBPD.
    pub fn take_ready(&mut self) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        while let Some((payload, _ts)) = self.pending.remove(&self.next_deliver) {
            out.push(payload);
            self.next_deliver = (self.next_deliver + 1) & 0x7FFF_FFFF;
        }
        out
    }

    /// Pop every in-order payload whose TSBPD delivery time is due at `now_us`
    /// (a monotonic microsecond clock). Without TSBPD configured this is exactly
    /// [`take_ready`](Self::take_ready) (no timing gate). With it, the time base
    /// is anchored on the first packet to become deliverable so that packet
    /// releases `latency_us` after `now_us`, and every later packet inherits the
    /// same delay relative to the sender timestamps, holding the stream back into
    /// a steady, jitter-smoothed output. A packet that is due past a gap releases
    /// and drops the missing packets before it (libsrt's too-late packet drop),
    /// since a live sender has stopped retransmitting them by then.
    pub fn take_ready_at(&mut self, now_us: u64) -> Vec<Vec<u8>> {
        let Some(latency) = self.tsbpd_latency_us else {
            return self.take_ready();
        };
        let now_us = now_us as i64;
        let mut out = Vec::new();
        while let Some((seq, timestamp)) = self.first_pending() {
            let anchor = *self.tsbpd_anchor.get_or_insert(TsbpdAnchor {
                timestamp,
                delivery_us: now_us + i64::from(latency),
            });
            let delivery_us = anchor.delivery_us(timestamp);
            if now_us < delivery_us {
                break; // not yet this packet's scheduled delivery time
            }
            let (payload, _) = self.pending.remove(&seq).unwrap();
            out.push(payload);
            if seq != self.next_deliver {
                g2g_core::g2g_debug!(
                    Target::category(RECEIVER_LOG_CATEGORY),
                    "dropped {} packets too late to deliver before seq {seq}",
                    seq_distance(seq, self.next_deliver)
                );
            }
            self.next_deliver = (seq + 1) & SEQUENCE_MASK;
            self.tsbpd_anchor = Some(TsbpdAnchor {
                timestamp,
                delivery_us,
            });
        }
        out
    }

    // on_data buffers nothing behind the cursor, so the wrap-order first is the smallest key at or past it
    fn first_pending(&self) -> Option<(u32, u32)> {
        self.pending
            .range(self.next_deliver..)
            .next()
            .or_else(|| self.pending.range(..self.next_deliver).next())
            .map(|(&seq, &(_, timestamp))| (seq, timestamp))
    }

    // a range ahead of the cursor is left to the too-late drop in take_ready_at
    pub fn on_drop_request(&mut self, first_sequence: u32, last_sequence: u32) {
        let covers_cursor =
            seq_ge(self.next_deliver, first_sequence) && seq_ge(last_sequence, self.next_deliver);
        if !self.have_base || !covers_cursor {
            return;
        }
        self.next_deliver = (last_sequence + 1) & SEQUENCE_MASK;
        if seq_gt(last_sequence, self.max_seen) {
            self.max_seen = last_sequence;
        }
        let cursor = self.next_deliver;
        self.pending.retain(|&seq, _| seq_ge(seq, cursor));
    }

    /// The sequence numbers between the delivery cursor and the highest seen
    /// that have not arrived (the NAK loss list).
    pub fn missing(&self) -> Vec<u32> {
        if !self.have_base {
            return Vec::new();
        }
        self.missing_in(self.next_deliver, self.max_seen).collect()
    }

    // sequences in first..=last at or past the delivery cursor that have not arrived
    fn missing_in(&self, first: u32, last: u32) -> impl Iterator<Item = u32> + '_ {
        let start = if seq_gt(self.next_deliver, first) {
            self.next_deliver
        } else {
            first
        };
        let count = if seq_ge(last, start) {
            seq_distance(last, start) + 1
        } else {
            0
        };
        (0..count)
            .map(move |offset| (start + offset) & SEQUENCE_MASK)
            .filter(|seq| !self.pending.contains_key(seq))
            .take(MAX_LOSS_LIST)
    }

    /// The next-to-deliver sequence, the ACK point (everything before it is in).
    /// `None` until the first data packet, since an ACK then names no real
    /// sequence and libsrt drops the connection over it.
    pub fn ack_seq(&self) -> Option<u32> {
        self.have_base.then_some(self.next_deliver)
    }

    pub fn available_buffer(&self) -> u32 {
        let held = u32::try_from(self.pending.len()).unwrap_or(u32::MAX);
        FLOW_WINDOW_PACKETS.saturating_sub(held)
    }
}

// libsrt's ACK_WND_SIZE
const UNANSWERED_ACK_LIMIT: usize = 1024;
// libsrt's live-mode floor on the period of the repeated loss report
const MIN_NAK_REPORT_INTERVAL_US: u64 = 20_000;

// libsrt's receiver round-trip time, from the delay between a full ACK and the ACKACK echoing its number
#[derive(Debug)]
pub struct AckRoundTrip {
    // ack number and monotonic microsecond send time
    unanswered_acks: alloc::collections::VecDeque<(u32, u64)>,
    smoothed_rtt_us: u64,
    rtt_variance_us: u64,
    measured: bool,
}

impl Default for AckRoundTrip {
    fn default() -> Self {
        Self::new()
    }
}

impl AckRoundTrip {
    pub fn new() -> Self {
        Self {
            unanswered_acks: alloc::collections::VecDeque::new(),
            smoothed_rtt_us: INITIAL_RTT_US.into(),
            rtt_variance_us: INITIAL_RTT_VARIANCE_US.into(),
            measured: false,
        }
    }

    pub fn on_ack_sent(&mut self, ack_no: u32, now_us: u64) {
        if self.unanswered_acks.len() == UNANSWERED_ACK_LIMIT {
            self.unanswered_acks.pop_front();
        }
        self.unanswered_acks.push_back((ack_no, now_us));
    }

    pub fn on_ackack(&mut self, ack_no: u32, now_us: u64) {
        let Some(index) = self.unanswered_acks.iter().position(|&(n, _)| n == ack_no) else {
            return;
        };
        let (_, sent_us) = self.unanswered_acks[index];
        self.unanswered_acks.drain(..=index);
        let rtt_us = now_us.saturating_sub(sent_us);
        if !self.measured {
            self.smoothed_rtt_us = rtt_us;
            self.rtt_variance_us = rtt_us / 2;
            self.measured = true;
            return;
        }
        self.rtt_variance_us =
            (self.rtt_variance_us * 3 + self.smoothed_rtt_us.abs_diff(rtt_us)) / 4;
        self.smoothed_rtt_us = (self.smoothed_rtt_us * 7 + rtt_us) / 8;
    }

    // the ACK fields are 32-bit, an RTT past 71 minutes reports as the largest value
    pub fn smoothed_rtt_us(&self) -> u32 {
        u32::try_from(self.smoothed_rtt_us).unwrap_or(u32::MAX)
    }

    pub fn rtt_variance_us(&self) -> u32 {
        u32::try_from(self.rtt_variance_us).unwrap_or(u32::MAX)
    }

    // libsrt live mode repeats the loss report every half of RTT plus four deviations
    pub fn nak_report_interval_us(&self) -> u64 {
        ((self.smoothed_rtt_us + 4 * self.rtt_variance_us) / 2).max(MIN_NAK_REPORT_INTERVAL_US)
    }
}

/// SRT live-mode congestion control: rate-based output pacing. Rather than the
/// window-based AIMD of file transfer, SRT live mode paces packets so the egress
/// does not burst-flood the path and leaves headroom for retransmissions. This
/// either caps at a fixed bandwidth (`maxbw`) or follows the measured input rate
/// plus an overhead margin, and reports the inter-packet send interval
/// ([`snd_period_us`](Self::snd_period_us)) the I/O layer waits between packets.
/// Sans-IO + integer-only: the element feeds packet sizes and a monotonic clock.
#[derive(Debug)]
pub struct LiveCc {
    /// Hard bandwidth cap in bytes/sec; `None` follows the measured input rate.
    maxbw: Option<u64>,
    /// Headroom percent over the input rate for retransmits (rate-follow mode).
    overhead: u64,
    /// EWMA of the input rate in bytes/sec (the measured stream rate).
    est_rate: u64,
    /// Monotonic microsecond timestamp of the last fed packet.
    last_us: Option<u64>,
}

/// EWMA smoothing shift for the input-rate estimate (weight 1/8 on each sample).
const CC_EWMA_SHIFT: u64 = 3;
/// Default retransmit headroom (percent) in rate-follow mode (SRT's 25%).
pub const CC_DEFAULT_OVERHEAD: u64 = 25;

impl LiveCc {
    /// `maxbw` > 0 caps the egress at that many bytes/sec; `maxbw == 0` follows
    /// the measured input rate plus `overhead_percent` headroom (SRT semantics:
    /// maxbw 0 means relative-to-input).
    pub fn new(maxbw: u64, overhead_percent: u64) -> Self {
        Self {
            maxbw: (maxbw > 0).then_some(maxbw),
            overhead: overhead_percent,
            est_rate: 0,
            last_us: None,
        }
    }

    /// Record a sent packet of `len` bytes at monotonic `now_us`, updating the
    /// EWMA input-rate estimate (rate-follow mode). The first packet only seeds
    /// the clock; the estimate starts on the second (it needs an interval).
    pub fn on_packet(&mut self, len: usize, now_us: u64) {
        if let Some(prev) = self.last_us {
            let delta = now_us.saturating_sub(prev);
            // Instantaneous rate (bytes/sec) over this inter-packet gap; a zero
            // gap (checked_div -> None) contributes no sample.
            if let Some(inst) = (len as u64).saturating_mul(1_000_000).checked_div(delta) {
                self.est_rate = if self.est_rate == 0 {
                    inst
                } else {
                    self.est_rate - (self.est_rate >> CC_EWMA_SHIFT) + (inst >> CC_EWMA_SHIFT)
                };
            }
        }
        self.last_us = Some(now_us);
    }

    /// The current estimated input rate in bytes/sec.
    pub fn estimated_rate_bps(&self) -> u64 {
        self.est_rate
    }

    /// The rate the pacer targets: the fixed cap, or the estimate plus overhead.
    fn target_rate(&self) -> u64 {
        match self.maxbw {
            Some(bw) => bw,
            None => self
                .est_rate
                .saturating_add(self.est_rate.saturating_mul(self.overhead) / 100),
        }
    }

    /// Microseconds to wait before sending a packet of `next_len` bytes to hold
    /// the egress at the target rate. `0` when no rate is known yet (rate-follow
    /// before the estimate warms up), i.e. send without pacing.
    pub fn snd_period_us(&self, next_len: usize) -> u64 {
        // A zero target rate (rate-follow before the estimate warms up) yields no
        // pacing delay (checked_div -> None -> 0).
        (next_len as u64)
            .saturating_mul(1_000_000)
            .checked_div(self.target_rate())
            .unwrap_or(0)
    }
}

/// 31-bit sequence comparison (wrap-aware): is `a >= b`?
fn seq_ge(a: u32, b: u32) -> bool {
    a == b || seq_gt(a, b)
}

/// 31-bit sequence comparison (wrap-aware): is `a > b`?
fn seq_gt(a: u32, b: u32) -> bool {
    let diff = seq_distance(a, b);
    diff != 0 && diff < 0x4000_0000
}

/// How far `a` is past `b` on the 31-bit sequence circle.
fn seq_distance(a: u32, b: u32) -> u32 {
    a.wrapping_sub(b) & 0x7FFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{format, vec};

    #[test]
    fn data_packet_round_trips_and_sets_header_bits() {
        let pkt = build_data_packet(0x1234, 7, true, 0, 99, 0xABCD_0123, b"hello");
        assert!(!is_control(&pkt), "data packets clear the control bit");
        let d = parse_data_packet(&pkt).expect("parse");
        assert_eq!(d.seq, 0x1234);
        assert!(d.retransmit, "retransmit flag set");
        assert_eq!(d.kk, 0, "cleartext key flag");
        assert_eq!(d.timestamp, 99);
        assert_eq!(d.dst_socket_id, 0xABCD_0123);
        assert_eq!(d.payload, b"hello");
        // The sequence number must fit in 31 bits (control bit stays clear).
        assert_eq!(pkt[0] & 0x80, 0);
    }

    #[test]
    fn handshake_cif_round_trips_with_extensions() {
        let hs = Handshake {
            version: HS_VERSION_5,
            encryption: 0,
            ext_field: HS_EXT_FLAG_HSREQ | HS_EXT_FLAG_CONFIG,
            init_seq: 1000,
            mtu: 1500,
            flow_window: FLOW_WINDOW_PACKETS,
            hs_type: URQ_CONCLUSION,
            srt_socket_id: 0xDEAD_BEEF,
            syn_cookie: 0x0BAD_F00D,
            peer_ip: [1; 16],
            latency_ms: Some(120),
            srt_flags: LIVE_MODE_FLAGS,
            stream_id: Some("live/cam0".into()),
            km: Some(vec![0xAA, 0xBB, 0xCC, 0xDD, 0x11, 0x22, 0x33, 0x44]),
            is_response: false,
        };
        let bytes = build_control(&Control::Handshake(hs.clone()), 42, 99);
        assert!(is_control(&bytes), "control bit set");
        let Control::Handshake(parsed) = parse_control(&bytes).expect("parse") else {
            panic!("not a handshake");
        };
        assert_eq!(parsed.version, HS_VERSION_5);
        assert_eq!(parsed.hs_type, URQ_CONCLUSION);
        assert_eq!(parsed.srt_socket_id, 0xDEAD_BEEF);
        assert_eq!(parsed.syn_cookie, 0x0BAD_F00D);
        assert_eq!(parsed.latency_ms, Some(120), "HSREQ latency survives");
        assert_eq!(parsed.srt_flags, LIVE_MODE_FLAGS, "HSREQ flags survive");
        assert_eq!(
            parsed.stream_id.as_deref(),
            Some("live/cam0"),
            "stream id survives"
        );
        assert_eq!(
            parsed.km.as_deref(),
            Some(&[0xAA, 0xBB, 0xCC, 0xDD, 0x11, 0x22, 0x33, 0x44][..]),
            "KM extension bytes survive the cif round trip"
        );
    }

    // gosrt (mediamtx) rejects a conclusion without these, libsrt sends them
    #[test]
    fn caller_conclusion_requests_live_mode() {
        let caller = SrtHandshake::new_caller(1, 1, 120, None, None);
        let cif = build_handshake_cif(&caller.conclusion());
        // the HSREQ TLV header follows the 48-byte fixed CIF, the flags follow the version
        let flags = u32::from_be_bytes(cif[56..60].try_into().unwrap());
        let required = SRT_OPT_TSBPDSND
            | SRT_OPT_TSBPDRCV
            | SRT_OPT_TLPKTDROP
            | SRT_OPT_NAKREPORT
            | SRT_OPT_REXMITFLG;
        assert_eq!(flags & required, required);
    }

    // a libsrt caller sends periodic-NAK-aware retransmits and too-late drops only for these
    #[test]
    fn listener_answers_with_live_mode_flags() {
        const COOKIE: u32 = 0x1357_9BDF;
        let reply_flags = |caller_flags: u32| {
            let mut listener = SrtHandshake::new_listener(2, 120, COOKIE);
            let mut conclusion = Handshake::induction(1, 1000);
            conclusion.version = HS_VERSION_5;
            conclusion.hs_type = URQ_CONCLUSION;
            conclusion.syn_cookie = COOKIE;
            conclusion.latency_ms = Some(120);
            conclusion.srt_flags = caller_flags;
            let step = listener.on_packet(&build_control(&Control::Handshake(conclusion), 0, 0));
            let Some(Control::Handshake(reply)) = parse_control(&step.reply.expect("reply")) else {
                panic!("the listener answers with a handshake");
            };
            assert!(
                reply.is_response,
                "a plaintext reply carries an HSRSP block"
            );
            (reply.srt_flags, listener.peer_marks_retransmissions())
        };
        let without_retransmit_bit = LIVE_MODE_FLAGS & !SRT_OPT_REXMITFLG;
        assert_eq!(reply_flags(LIVE_MODE_FLAGS), (LIVE_MODE_FLAGS, true));
        assert_eq!(
            reply_flags(without_retransmit_bit),
            (without_retransmit_bit, false)
        );
    }

    #[test]
    fn stream_id_words_are_byte_reversed_on_the_wire() {
        let mut hs = Handshake::induction(1, 1);
        hs.stream_id = Some("publish:x".into());
        let cif = build_handshake_cif(&hs);
        // libsrt's layout for "publish:x": "publ" "ish:" "x" padded, each word reversed
        assert_eq!(&cif[cif.len() - 12..], b"lbup:hsi\0\0\0x");
        let parsed = parse_handshake_cif(&cif).expect("parse");
        assert_eq!(parsed.stream_id.as_deref(), Some("publish:x"));
    }

    #[test]
    fn induction_handshake_carries_no_extensions() {
        let hs = Handshake::induction(0x1111_2222, 500);
        let bytes = build_control(&Control::Handshake(hs), 0, 0);
        let Control::Handshake(p) = parse_control(&bytes).unwrap() else {
            panic!()
        };
        assert_eq!(p.version, HS_VERSION_INDUCTION);
        assert_eq!(p.ext_field, SRT_MAGIC);
        assert_eq!(p.hs_type, URQ_INDUCTION);
        assert_eq!(p.latency_ms, None);
        assert_eq!(p.stream_id, None);
    }

    #[test]
    fn ack_ackack_keepalive_shutdown_round_trip() {
        for (ctrl, name) in [
            (
                Control::Ack {
                    ack_no: 5,
                    ack_seq: 1000,
                    rtt_us: INITIAL_RTT_US,
                    rtt_variance_us: INITIAL_RTT_VARIANCE_US,
                    available_buffer: 4000,
                },
                "ack",
            ),
            (Control::LightAck { ack_seq: 1000 }, "light ack"),
            (Control::AckAck { ack_no: 5 }, "ackack"),
            (Control::Keepalive, "keepalive"),
            (Control::Shutdown, "shutdown"),
        ] {
            let bytes = build_control(&ctrl, 7, 13);
            assert_eq!(
                parse_control(&bytes).expect(name),
                ctrl,
                "{name} round trips"
            );
        }
    }

    #[test]
    fn caller_and_listener_complete_the_handshake() {
        let mut caller =
            SrtHandshake::new_caller(0x0A0A_0A0A, 1000, 120, Some("live".into()), None);
        let mut listener = SrtHandshake::new_listener(0x0B0B_0B0B, 80, 0x1357_9BDF);

        // Caller induction -> listener induction response -> caller conclusion ->
        // listener conclusion response -> both established.
        let mut pkt = caller.start().expect("caller opens");
        for _ in 0..4 {
            let step = listener.on_packet(&pkt);
            if let Some(reply) = step.reply {
                let cstep = caller.on_packet(&reply);
                if let Some(next) = cstep.reply {
                    pkt = next;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        assert!(caller.is_established(), "caller established");
        assert!(listener.is_established(), "listener established");
        // Each side learned the other's socket id (the data destination).
        assert_eq!(caller.peer_socket_id(), 0x0B0B_0B0B);
        assert_eq!(listener.peer_socket_id(), 0x0A0A_0A0A);
        assert_eq!(
            listener.peer_init_seq(),
            1000,
            "caller's ISN reached the listener"
        );
        assert_eq!(listener.init_seq(), 1000, "the listener sends from it");
        assert_eq!(caller.peer_init_seq(), 1000, "and echoes it back");
    }

    // runs a caller against a listener until one side stops answering
    fn exchange(caller: &mut SrtHandshake, listener: &mut SrtHandshake) {
        let mut pkt = caller.start().expect("caller opens");
        for _ in 0..4 {
            let Some(reply) = listener.on_packet(&pkt).reply else {
                return;
            };
            let Some(next) = caller.on_packet(&reply).reply else {
                return;
            };
            pkt = next;
        }
    }

    #[test]
    fn a_listener_rejects_a_caller_with_another_stream_id() {
        const COOKIE: u32 = 0x1357_9BDF;
        let mut caller = SrtHandshake::new_caller(1, 1000, 120, Some("other".into()), None);
        let mut listener = SrtHandshake::new_listener(2, 120, COOKIE);
        listener.admit_only(Some("live".into()), None);
        exchange(&mut caller, &mut listener);
        assert!(!listener.is_established());
        assert_eq!(caller.rejection(), Some(REJECT_STREAM_NOT_FOUND));

        let mut caller = SrtHandshake::new_caller(1, 1000, 120, Some("live".into()), None);
        let mut listener = SrtHandshake::new_listener(2, 120, COOKIE);
        listener.admit_only(Some("live".into()), None);
        exchange(&mut caller, &mut listener);
        assert!(caller.is_established() && listener.is_established());
    }

    #[test]
    fn a_listener_rejects_a_caller_without_the_key_length_it_requires() {
        const COOKIE: u32 = 0x1357_9BDF;
        const REQUIRED_KEY_BYTES: usize = 32;
        let mut km = vec![0u8; KM_KEY_LENGTH_OFFSET + 1];
        km[KM_KEY_LENGTH_OFFSET] = 16 / 4;
        let mut caller = SrtHandshake::new_caller(1, 1000, 120, None, Some(km));
        let mut listener = SrtHandshake::new_listener(2, 120, COOKIE);
        listener.admit_only(None, Some(REQUIRED_KEY_BYTES));
        exchange(&mut caller, &mut listener);
        assert!(!listener.is_established());
        assert_eq!(caller.rejection(), Some(REJECT_UNSECURE));
    }

    const LOW_COOKIE: u32 = 0x1000;
    const HIGH_COOKIE: u32 = 0x2000;

    fn rendezvous_side(socket_id: u32, cookie: u32) -> SrtHandshake {
        SrtHandshake::new_rendezvous(
            socket_id,
            0,
            120,
            None,
            Some(vec![socket_id as u8; 8]),
            cookie,
        )
    }

    // trades packets until both sides connect, `a_heard_first` drops b's opening wave
    fn meet(a: &mut SrtHandshake, b: &mut SrtHandshake, a_heard_first: bool) {
        let mut to_b = vec![a.start().unwrap()];
        let mut to_a = if a_heard_first {
            Vec::new()
        } else {
            vec![b.start().unwrap()]
        };
        for _ in 0..8 {
            let from_b: Vec<_> = to_b
                .drain(..)
                .filter_map(|p| b.on_packet(&p).reply)
                .collect();
            let from_a: Vec<_> = to_a
                .drain(..)
                .filter_map(|p| a.on_packet(&p).reply)
                .collect();
            to_a = from_b;
            to_b = from_a;
            if a.is_established() && b.is_established() {
                return;
            }
        }
        panic!("the rendezvous did not connect");
    }

    #[test]
    fn the_larger_cookie_sends_its_key_in_either_flow() {
        for a_heard_first in [false, true] {
            let mut a = rendezvous_side(1, HIGH_COOKIE);
            let mut b = rendezvous_side(2, LOW_COOKIE);
            meet(&mut a, &mut b, a_heard_first);
            assert_eq!(a.role, Role::Rendezvous(RendezvousState::Initiator));
            assert_eq!(b.peer_km(), Some(&[1u8; 8][..]), "b takes a's key");
            assert_eq!(a.peer_km(), Some(&[1u8; 8][..]), "b echoes it back");
            assert_eq!(b.peer_socket_id(), 1);
            assert_eq!(a.peer_socket_id(), 2);
            assert!(a.peer_marks_retransmissions() && b.peer_marks_retransmissions());
        }
        let mut a = rendezvous_side(1, LOW_COOKIE);
        let mut b = rendezvous_side(2, HIGH_COOKIE);
        meet(&mut a, &mut b, false);
        assert_eq!(b.role, Role::Rendezvous(RendezvousState::Initiator));
        assert_eq!(a.peer_km(), Some(&[2u8; 8][..]));
    }

    #[test]
    fn equal_cookies_decide_nothing() {
        let mut a = rendezvous_side(1, LOW_COOKIE);
        let b = rendezvous_side(2, LOW_COOKIE);
        let step = a.on_packet(&b.start().unwrap());
        assert_eq!(step.reply, None);
        assert_eq!(a.role, Role::Rendezvous(RendezvousState::Waving));
    }

    #[test]
    fn a_responder_whose_agreement_is_lost_connects_on_data() {
        let mut initiator = rendezvous_side(1, HIGH_COOKIE);
        let mut responder = rendezvous_side(2, LOW_COOKIE);
        let conclusion = initiator
            .on_packet(&responder.start().unwrap())
            .reply
            .unwrap();
        let answer = responder.on_packet(&conclusion).reply.unwrap();
        assert!(initiator.on_packet(&answer).established);
        assert!(!responder.is_established(), "the agreement never arrives");
        let data = build_data_packet(0, 1, false, 0, 0, 2, b"ts");
        assert!(responder.on_packet(&data).established);
    }

    #[test]
    fn listener_offers_the_injected_cookie() {
        // The cookie must be the value the I/O layer injected, not one derived
        // from the public socket id, so an off-path attacker can't predict it.
        let cookie = 0x1357_9BDF;
        let mut listener = SrtHandshake::new_listener(0x0B0B_0B0B, 80, cookie);
        let induction = build_control(
            &Control::Handshake(Handshake::induction(0x0A0A_0A0A, 1)),
            0,
            0,
        );
        let step = listener.on_packet(&induction);
        let reply = step.reply.expect("listener answers induction");
        let Control::Handshake(hs) = parse_control(&reply).expect("handshake reply") else {
            panic!("induction reply is a handshake");
        };
        assert_eq!(hs.syn_cookie, cookie, "offered cookie is the injected one");
    }

    #[test]
    fn listener_rejects_a_conclusion_with_a_bad_cookie() {
        let mut listener = SrtHandshake::new_listener(0x0B0B_0B0B, 80, 0x1357_9BDF);
        let induction = build_control(
            &Control::Handshake(Handshake::induction(0x0A0A_0A0A, 1)),
            0,
            0,
        );
        let _ = listener.on_packet(&induction); // learns peer, sets cookie
                                                // A conclusion echoing the wrong cookie must not establish.
        let mut bad = Handshake::induction(0x0A0A_0A0A, 1);
        bad.version = HS_VERSION_5;
        bad.hs_type = URQ_CONCLUSION;
        bad.syn_cookie = 0xDEAD_DEAD;
        bad.latency_ms = Some(50);
        let step = listener.on_packet(&build_control(&Control::Handshake(bad), 0, 0x0B0B_0B0B));
        assert!(!step.established, "a bad cookie is rejected");
        assert!(!listener.is_established());
    }

    #[test]
    fn arq_recovers_a_dropped_packet_in_order() {
        let mut sender = SrtSender::new(0x1234, 100, 64);
        let mut receiver = SrtReceiver::new();

        // Send seqs 100..105; drop 102 on the way.
        let mut wire = Vec::new();
        for i in 0..5 {
            let pkt = sender.send(format!("p{i}").as_bytes(), i);
            wire.push(pkt);
        }
        let dropped_seq = 102;
        for pkt in &wire {
            let d = parse_data_packet(pkt).unwrap();
            if d.seq != dropped_seq {
                receiver.on_data(d);
            }
        }
        // Only 100,101 deliver; 102 is a gap holding back 103,104.
        assert_eq!(receiver.take_ready(), vec![b"p0".to_vec(), b"p1".to_vec()]);
        let missing = receiver.missing();
        assert_eq!(missing, vec![102], "the gap is reported for NAK");

        // Receiver NAKs, sender retransmits, receiver recovers + drains in order.
        let resends = sender.on_control(&Control::Nak { loss: missing }, 0);
        assert_eq!(resends.len(), 1, "exactly the lost packet resent");
        let resent = parse_data_packet(&resends[0]).unwrap();
        assert!(resent.retransmit, "resend carries the R flag");
        assert_eq!(
            resent.timestamp,
            dropped_seq - 100,
            "resend keeps the timestamp the receiver schedules it by"
        );
        receiver.on_data(resent);
        assert_eq!(
            receiver.take_ready(),
            vec![b"p2".to_vec(), b"p3".to_vec(), b"p4".to_vec()],
            "the rest delivers in order once the gap is filled",
        );
        assert!(receiver.missing().is_empty(), "no gaps remain");
        assert_eq!(sender.retransmits(), 1);
    }

    // libsrt stalls after one flow window if every ACK is a 4-byte light ACK.
    #[test]
    fn full_ack_reports_the_free_receive_buffer_in_the_libsrt_layout() {
        const SENT: u32 = 5;
        const LOST: u32 = 2;
        let mut sender = SrtSender::new(0x1234, 100, 64);
        let mut receiver = SrtReceiver::new();
        for i in 0..SENT {
            let pkt = sender.send(b"p", i);
            if i != LOST {
                receiver.on_data(parse_data_packet(&pkt).unwrap());
            }
        }
        // Packets past the gap are held until the lost one arrives.
        receiver.take_ready();
        let held = SENT - LOST - 1;
        assert_eq!(receiver.available_buffer(), FLOW_WINDOW_PACKETS - held);

        let ack = Control::Ack {
            ack_no: 1,
            ack_seq: receiver.ack_seq().unwrap(),
            rtt_us: INITIAL_RTT_US,
            rtt_variance_us: INITIAL_RTT_VARIANCE_US,
            available_buffer: receiver.available_buffer(),
        };
        let bytes = build_control(&ack, 0, 0);
        let cif = &bytes[HEADER_LEN..];
        let word =
            |index: usize| u32::from_be_bytes(cif[index * 4..index * 4 + 4].try_into().unwrap());
        assert_eq!(cif.len(), FULL_ACK_CIF_LEN, "a full ACK, not a light one");
        assert_eq!(word(0), 100 + LOST, "acknowledged up to the gap");
        assert_eq!(word(1), INITIAL_RTT_US);
        assert_eq!(word(2), INITIAL_RTT_VARIANCE_US);
        assert_eq!(word(3), FLOW_WINDOW_PACKETS - held, "free buffer word");
    }

    #[test]
    fn receiver_has_no_ack_point_before_the_first_packet() {
        let mut sender = SrtSender::new(0x1234, 100, 64);
        let mut receiver = SrtReceiver::new();
        assert_eq!(receiver.ack_seq(), None);
        receiver.on_data(parse_data_packet(&sender.send(b"p0", 0)).unwrap());
        assert_eq!(receiver.ack_seq(), Some(100));
    }

    #[test]
    fn ack_across_the_sequence_wrap_keeps_unacked_packets() {
        const LAST_SEQUENCE: u32 = 0x7FFF_FFFF;
        const PACKETS_BEFORE_WRAP: u32 = 3;
        const PACKETS_AFTER_WRAP: u32 = 3;
        let first = LAST_SEQUENCE - (PACKETS_BEFORE_WRAP - 1);
        let mut sender = SrtSender::new(0x1234, first, 64);
        for i in 0..PACKETS_BEFORE_WRAP + PACKETS_AFTER_WRAP {
            sender.send(format!("p{i}").as_bytes(), i);
        }

        // Everything before the wrap arrived, the post-wrap packets 0, 1, 2 did not.
        sender.on_control(&Control::LightAck { ack_seq: 0 }, 0);
        let resends = sender.on_control(
            &Control::Nak {
                loss: vec![LAST_SEQUENCE, 0, 1, 2],
            },
            0,
        );
        let resent: Vec<u32> = resends
            .iter()
            .map(|p| parse_data_packet(p).unwrap().seq)
            .collect();
        assert_eq!(resent, [0, 1, 2], "acked packets dropped, unacked kept");
    }

    #[test]
    fn tsbpd_holds_each_packet_until_its_scheduled_delivery() {
        // Three packets, 20 ms apart in sender timestamps, 100 ms TSBPD latency.
        let mut rx = SrtReceiver::new();
        rx.set_tsbpd(100_000); // 100 ms in microseconds
        let pkt = |seq, ts| DataPacket {
            seq,
            retransmit: false,
            kk: 0,
            timestamp: ts,
            dst_socket_id: 0,
            payload: vec![seq as u8],
        };
        rx.on_data(pkt(10, 0));
        rx.on_data(pkt(11, 20_000));
        rx.on_data(pkt(12, 40_000));

        // At the anchor instant (now=1_000_000 us) nothing is due: the first
        // packet schedules to now+latency.
        assert!(
            rx.take_ready_at(1_000_000).is_empty(),
            "held for the latency"
        );
        // Just before the first packet's delivery time: still nothing.
        assert!(rx.take_ready_at(1_099_999).is_empty(), "not quite due");
        // At now+100 ms the first packet releases; the next two are not yet due
        // (they sit 20/40 ms further along).
        assert_eq!(
            rx.take_ready_at(1_100_000),
            vec![vec![10u8]],
            "first packet due"
        );
        assert!(
            rx.take_ready_at(1_110_000).is_empty(),
            "second not due for 10 more ms"
        );
        assert_eq!(
            rx.take_ready_at(1_120_000),
            vec![vec![11u8]],
            "second due at +20 ms"
        );
        assert_eq!(
            rx.take_ready_at(1_140_000),
            vec![vec![12u8]],
            "third due at +40 ms"
        );
    }

    #[test]
    fn tsbpd_delivers_reordered_packets_on_schedule_and_in_order() {
        // Packets arrive out of order; TSBPD must still release them in sequence
        // order and on their timestamp schedule, never the arrival order.
        let mut rx = SrtReceiver::new();
        rx.set_tsbpd(50_000); // 50 ms
        let pkt = |seq, ts| DataPacket {
            seq,
            retransmit: false,
            kk: 0,
            timestamp: ts,
            dst_socket_id: 0,
            payload: vec![seq as u8],
        };
        // 5 arrives before 4 (reordered on the wire).
        rx.on_data(pkt(3, 0));
        rx.on_data(pkt(5, 20_000));
        rx.on_data(pkt(4, 10_000));
        // Anchor at now=0; far in the future everything is due, in order.
        assert!(rx.take_ready_at(0).is_empty(), "held at the anchor");
        assert_eq!(
            rx.take_ready_at(1_000_000),
            vec![vec![3u8], vec![4u8], vec![5u8]],
            "delivered in sequence order regardless of arrival order",
        );
    }

    fn timestamped_packet(seq: u32, timestamp: u32) -> DataPacket {
        DataPacket {
            seq,
            retransmit: false,
            kk: 0,
            timestamp,
            dst_socket_id: 0,
            payload: vec![seq as u8],
        }
    }

    #[test]
    fn tsbpd_drops_a_gap_once_the_packet_after_it_is_due() {
        const LATENCY_US: u32 = 100_000;
        const PACKET_INTERVAL_US: u32 = 10_000;
        const ANCHOR_US: u64 = 1_000_000;
        const LOST: u32 = 11;
        let mut rx = SrtReceiver::new();
        rx.set_tsbpd(LATENCY_US);
        for seq in [10, 12, 13] {
            rx.on_data(timestamped_packet(seq, (seq - 10) * PACKET_INTERVAL_US));
        }
        let due_us = |seq: u32| ANCHOR_US + u64::from(LATENCY_US + (seq - 10) * PACKET_INTERVAL_US);

        assert!(rx.take_ready_at(ANCHOR_US).is_empty());
        assert_eq!(rx.take_ready_at(due_us(10)), vec![vec![10u8]]);
        assert!(
            rx.take_ready_at(due_us(12) - 1).is_empty(),
            "the gap waits for retransmission until the packet after it is due"
        );
        assert_eq!(rx.missing(), vec![LOST]);
        assert_eq!(
            rx.take_ready_at(due_us(12)),
            vec![vec![12u8]],
            "the due packet releases past the gap"
        );
        assert!(rx.missing().is_empty(), "the dropped packet is not NAKed");
        assert_eq!(rx.ack_seq(), Some(13));
        assert_eq!(rx.take_ready_at(due_us(13)), vec![vec![13u8]]);
    }

    #[test]
    fn tsbpd_schedules_across_the_sender_timestamp_wrap() {
        const LATENCY_US: u32 = 100_000;
        const BEFORE_WRAP_US: u32 = 10_000;
        const AFTER_WRAP_US: u32 = 10_000;
        let mut rx = SrtReceiver::new();
        rx.set_tsbpd(LATENCY_US);
        rx.on_data(timestamped_packet(1, u32::MAX - BEFORE_WRAP_US + 1));
        rx.on_data(timestamped_packet(2, AFTER_WRAP_US));

        assert!(rx.take_ready_at(0).is_empty());
        let first_due = u64::from(LATENCY_US);
        assert_eq!(
            rx.take_ready_at(first_due),
            vec![vec![1u8]],
            "the post-wrap packet is not due yet"
        );
        let second_due = first_due + u64::from(BEFORE_WRAP_US + AFTER_WRAP_US);
        assert!(rx.take_ready_at(second_due - 1).is_empty());
        assert_eq!(rx.take_ready_at(second_due), vec![vec![2u8]]);
    }

    #[test]
    fn a_drop_request_moves_delivery_past_the_dropped_range() {
        let mut rx = SrtReceiver::new();
        rx.on_data(timestamped_packet(10, 0));
        rx.on_data(timestamped_packet(13, 0));
        assert_eq!(rx.take_ready(), vec![vec![10u8]]);
        assert_eq!(rx.missing(), vec![11, 12]);

        let request = Control::DropRequest {
            message_number: 7,
            first_sequence: 11,
            last_sequence: 12,
        };
        let wire = build_control(&request, 0, 0);
        let Some(Control::DropRequest {
            first_sequence,
            last_sequence,
            ..
        }) = parse_control(&wire)
        else {
            panic!("the drop request does not parse back");
        };
        assert_eq!(parse_control(&wire), Some(request));
        rx.on_drop_request(first_sequence, last_sequence);
        assert!(rx.missing().is_empty());
        assert_eq!(rx.take_ready(), vec![vec![13u8]]);
    }

    #[test]
    fn without_tsbpd_take_ready_at_delivers_immediately() {
        let mut rx = SrtReceiver::new();
        let pkt = |seq| DataPacket {
            seq,
            retransmit: false,
            kk: 0,
            timestamp: 0,
            dst_socket_id: 0,
            payload: vec![seq as u8],
        };
        rx.on_data(pkt(1));
        rx.on_data(pkt(2));
        // No set_tsbpd: take_ready_at is a pass-through to take_ready (no gate).
        assert_eq!(rx.take_ready_at(0), vec![vec![1u8], vec![2u8]]);
    }

    #[test]
    fn key_material_control_round_trips() {
        let km = vec![0x12u8, 0x20, 0x29, 0x01, 0xDE, 0xAD, 0xBE, 0xEF];
        for rsp in [false, true] {
            let bytes = build_control(
                &Control::KeyMaterial {
                    rsp,
                    km: km.clone(),
                },
                7,
                42,
            );
            assert!(is_control(&bytes), "KM update is a control packet");
            let Control::KeyMaterial {
                rsp: got_rsp,
                km: got_km,
            } = parse_control(&bytes).expect("parse KM")
            else {
                panic!("not a KeyMaterial control");
            };
            assert_eq!(got_rsp, rsp, "request/response flag survives");
            assert_eq!(got_km, km, "KM blob survives the round trip");
        }
    }

    #[cfg(feature = "srt")]
    #[test]
    fn rekey_switches_keys_and_receiver_decrypts_across_the_switch() {
        use crate::srtcrypto::{AesKeySize, SrtCrypto, KM_KK_EVEN, KM_KK_ODD};

        let mut sender = SrtSender::new(0x1234, 100, 64);
        let mut receiver = SrtReceiver::new();
        // Connection-time even key, shared both sides.
        let even = SrtCrypto::generate(AesKeySize::Aes128);
        sender.set_crypto(even.clone());
        receiver.set_crypto(even);
        assert_eq!(sender.active_kk(), KM_KK_EVEN, "starts on the even key");

        // Two packets under the even key.
        let p0 = sender.send(b"aaa", 0);
        let p1 = sender.send(b"bbb", 1);

        // Rekey to a fresh odd key; the receiver installs it from the KM packet.
        let odd = SrtCrypto::generate(AesKeySize::Aes128);
        let km_ctrl = sender.rekey(odd, "pass");
        assert_eq!(sender.active_kk(), KM_KK_ODD, "active key flipped to odd");
        let Control::KeyMaterial { km, .. } = parse_control(&km_ctrl).expect("KM control") else {
            panic!("rekey did not emit a KeyMaterial packet");
        };
        let kk = SrtCrypto::km_kk(&km).expect("KM parity");
        receiver.install_key(kk, SrtCrypto::from_km(&km, "pass").expect("unwrap rekey"));

        // Two more under the odd key.
        let p2 = sender.send(b"ccc", 2);
        let p3 = sender.send(b"ddd", 3);

        // The wire carries the parity switch, and every payload decrypts with the
        // slot its KK selects, across the rekey.
        assert_eq!(
            parse_data_packet(&p1).unwrap().kk,
            KM_KK_EVEN,
            "pre-rekey packets are even"
        );
        assert_eq!(
            parse_data_packet(&p2).unwrap().kk,
            KM_KK_ODD,
            "post-rekey packets are odd"
        );
        for pkt in [&p0, &p1, &p2, &p3] {
            receiver.on_data(parse_data_packet(pkt).unwrap());
        }
        assert_eq!(
            receiver.take_ready(),
            vec![
                b"aaa".to_vec(),
                b"bbb".to_vec(),
                b"ccc".to_vec(),
                b"ddd".to_vec()
            ],
            "payloads decrypt across the even -> odd rekey",
        );
    }

    #[test]
    fn cc_maxbw_paces_at_the_fixed_cap() {
        // A 1 MB/s cap: a 1000-byte packet takes 1 ms, a 2000-byte one 2 ms.
        let cc = LiveCc::new(1_000_000, 0);
        assert_eq!(cc.snd_period_us(1000), 1000);
        assert_eq!(cc.snd_period_us(2000), 2000);
    }

    #[test]
    fn cc_rate_follow_tracks_input_and_adds_overhead() {
        // Feed 1250-byte packets 1 ms apart -> 1.25 MB/s input rate. Rate-follow
        // with 25% overhead paces at 1.5625 MB/s, so a 1250-byte packet -> 800 us
        // (faster than the 1000 us the bare input rate would give: headroom).
        let mut cc = LiveCc::new(0, CC_DEFAULT_OVERHEAD);
        let mut t = 0u64;
        for _ in 0..5 {
            cc.on_packet(1250, t);
            t += 1000;
        }
        assert_eq!(
            cc.estimated_rate_bps(),
            1_250_000,
            "EWMA settles on the steady input rate"
        );
        assert_eq!(
            cc.snd_period_us(1250),
            800,
            "25% headroom shortens the pacing interval"
        );
    }

    #[test]
    fn cc_rate_follow_does_not_pace_before_the_estimate_warms_up() {
        // No packets fed yet: the estimate is zero, so the pacer imposes no delay
        // (send freely until it learns the rate).
        let cc = LiveCc::new(0, CC_DEFAULT_OVERHEAD);
        assert_eq!(cc.snd_period_us(1000), 0);
    }

    #[test]
    fn parse_nak_cif_bounds_an_adversarial_range() {
        // A NAK range spanning nearly the whole sequence space must cap the
        // expansion instead of materializing billions of entries.
        let mut cif = Vec::new();
        cif.extend_from_slice(&0x8000_0000u32.to_be_bytes()); // range start 0 (high bit set)
        cif.extend_from_slice(&0x7FFF_FFFFu32.to_be_bytes()); // inclusive end
        assert_eq!(
            parse_nak_cif(&cif).len(),
            MAX_LOSS_LIST,
            "loss list is capped"
        );
    }

    #[test]
    fn missing_is_bounded_by_a_far_ahead_packet() {
        let mut receiver = SrtReceiver::new();
        let pkt = |seq| DataPacket {
            seq,
            retransmit: false,
            kk: 0,
            timestamp: 0,
            dst_socket_id: 0,
            payload: vec![0],
        };
        receiver.on_data(pkt(10)); // delivery base
                                   // one far-ahead packet jumps max_seen
        assert_eq!(
            receiver.on_data(pkt(10 + 3_000_000)).len(),
            MAX_LOSS_LIST,
            "the NAK for the new gap is capped"
        );
        assert_eq!(
            receiver.missing().len(),
            MAX_LOSS_LIST,
            "loss list is capped, not the full gap"
        );
    }

    fn packet(seq: u32, retransmit: bool) -> DataPacket {
        DataPacket {
            retransmit,
            ..timestamped_packet(seq, 0)
        }
    }

    #[test]
    fn a_gap_is_nacked_when_it_is_seen() {
        let mut receiver = SrtReceiver::new();
        assert!(receiver.on_data(packet(10, false)).is_empty());
        assert_eq!(receiver.on_data(packet(13, false)), vec![11, 12]);
        assert!(
            receiver.on_data(packet(14, false)).is_empty(),
            "the periodic report repeats an open gap, on_data NAKs it once"
        );
    }

    #[test]
    fn a_reordered_packet_inside_the_tolerance_is_never_nacked() {
        const LOSS_MAX_TTL: u32 = 3;
        let mut receiver = SrtReceiver::new();
        receiver.set_loss_max_ttl(LOSS_MAX_TTL);
        receiver.on_data(packet(10, false));
        assert!(receiver.on_data(packet(12, false)).is_empty());
        assert!(receiver.on_data(packet(11, false)).is_empty());
        for seq in 13..13 + LOSS_MAX_TTL {
            assert!(receiver.on_data(packet(seq, false)).is_empty());
        }
    }

    #[test]
    fn a_lost_packet_is_nacked_once_its_wait_runs_out() {
        const LOSS_MAX_TTL: u32 = 3;
        let mut receiver = SrtReceiver::new();
        receiver.set_loss_max_ttl(LOSS_MAX_TTL);
        receiver.on_data(packet(10, false));
        for seq in 12..12 + LOSS_MAX_TTL {
            assert!(receiver.on_data(packet(seq, false)).is_empty());
        }
        assert_eq!(receiver.on_data(packet(12 + LOSS_MAX_TTL, false)), vec![11]);
    }

    #[test]
    fn ordered_packets_shrink_the_reorder_tolerance() {
        const LOSS_MAX_TTL: u32 = 3;
        let mut receiver = SrtReceiver::new();
        receiver.set_loss_max_ttl(LOSS_MAX_TTL);
        for seq in 0..ORDERED_PACKETS_PER_TOLERANCE_STEP {
            receiver.on_data(packet(seq, false));
        }
        let lost = ORDERED_PACKETS_PER_TOLERANCE_STEP;
        let lowered_tolerance = LOSS_MAX_TTL - 1;
        for seq in lost + 1..lost + 1 + lowered_tolerance {
            assert!(receiver.on_data(packet(seq, false)).is_empty());
        }
        assert_eq!(
            receiver.on_data(packet(lost + 1 + lowered_tolerance, false)),
            vec![lost]
        );
    }

    #[test]
    fn a_reordered_packet_raises_the_tolerance_and_a_retransmission_does_not() {
        const LOSS_MAX_TTL: u32 = 3;
        let mut receiver = SrtReceiver::new();
        receiver.set_loss_max_ttl(LOSS_MAX_TTL);
        let ordered = ORDERED_PACKETS_PER_TOLERANCE_STEP * LOSS_MAX_TTL;
        for seq in 0..ordered {
            receiver.on_data(packet(seq, false));
        }
        assert_eq!(
            receiver.on_data(packet(ordered + 1, false)),
            vec![ordered],
            "the tolerance is down to 0"
        );
        receiver.on_data(packet(ordered + 2, false));
        receiver.on_data(packet(ordered, true));
        assert_eq!(
            receiver.on_data(packet(ordered + 4, false)),
            vec![ordered + 3],
            "a late retransmission leaves the tolerance at 0"
        );
        // one packet behind the highest received, without the retransmit bit
        receiver.on_data(packet(ordered + 3, false));
        assert!(receiver.on_data(packet(ordered + 6, false)).is_empty());
        assert_eq!(
            receiver.on_data(packet(ordered + 7, false)),
            vec![ordered + 5]
        );
    }

    #[test]
    fn a_full_ack_is_answered_with_an_ackack_and_a_light_ack_is_not() {
        const ACK_NO: u32 = 7;
        const CONTROL_TIMESTAMP: u32 = 1234;
        let mut sender = SrtSender::new(0x1234, 100, 64);
        sender.send(b"p0", 0);
        let light = sender.on_control(&Control::LightAck { ack_seq: 100 }, CONTROL_TIMESTAMP);
        assert!(light.is_empty());
        let full = Control::Ack {
            ack_no: ACK_NO,
            ack_seq: 101,
            rtt_us: INITIAL_RTT_US,
            rtt_variance_us: INITIAL_RTT_VARIANCE_US,
            available_buffer: FLOW_WINDOW_PACKETS,
        };
        let replies = sender.on_control(&full, CONTROL_TIMESTAMP);
        assert_eq!(replies.len(), 1);
        assert_eq!(
            parse_control(&replies[0]),
            Some(Control::AckAck { ack_no: ACK_NO })
        );
        let timestamp = u32::from_be_bytes(replies[0][8..12].try_into().unwrap());
        assert_eq!(timestamp, CONTROL_TIMESTAMP);
    }

    #[test]
    fn round_trip_follows_libsrt_smoothing() {
        let mut round_trip = AckRoundTrip::new();
        assert_eq!(
            round_trip.nak_report_interval_us(),
            u64::from(INITIAL_RTT_US + 4 * INITIAL_RTT_VARIANCE_US) / 2,
            "before a sample the initial estimate sets the report period"
        );
        round_trip.on_ack_sent(1, 0);
        round_trip.on_ack_sent(2, 10_000);
        round_trip.on_ackack(2, 50_000);
        assert_eq!(
            round_trip.smoothed_rtt_us(),
            40_000,
            "first sample taken whole"
        );
        assert_eq!(round_trip.rtt_variance_us(), 20_000);
        round_trip.on_ackack(1, 60_000);
        assert_eq!(
            round_trip.smoothed_rtt_us(),
            40_000,
            "an ACK older than the answered one is gone"
        );
        round_trip.on_ack_sent(3, 100_000);
        round_trip.on_ackack(3, 108_000);
        assert_eq!(round_trip.smoothed_rtt_us(), (40_000 * 7 + 8_000) / 8);
        assert_eq!(round_trip.rtt_variance_us(), (20_000 * 3 + 32_000) / 4);
    }

    #[test]
    fn nak_report_period_has_a_floor() {
        let mut round_trip = AckRoundTrip::new();
        round_trip.on_ack_sent(1, 0);
        round_trip.on_ackack(1, 100);
        assert_eq!(
            round_trip.nak_report_interval_us(),
            MIN_NAK_REPORT_INTERVAL_US
        );
    }

    #[test]
    fn nak_encodes_singletons_and_ranges() {
        // 5 is a singleton; 10..=13 a range; 20 a singleton again.
        let loss = vec![5u32, 10, 11, 12, 13, 20];
        let bytes = build_control(&Control::Nak { loss: loss.clone() }, 0, 0);
        let Control::Nak { loss: got } = parse_control(&bytes).unwrap() else {
            panic!()
        };
        assert_eq!(got, loss, "loss list round trips through range coding");
        // The range must use the compact 2-word form, not 4 singletons.
        let Control::Nak { .. } = parse_control(&bytes).unwrap() else {
            panic!()
        };
        // 1 (single 5) + 2 (range 10-13) + 1 (single 20) = 4 words = 16 bytes CIF.
        assert_eq!(bytes.len(), HEADER_LEN + 16, "range-coded CIF is compact");
    }
}
