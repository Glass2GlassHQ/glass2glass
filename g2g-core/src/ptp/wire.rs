//! PTP-over-UDP (IEEE 1588-2008 / SMPTE ST 2059-2) wire format, the messages a
//! SLAVE ordinary clock needs to parse and the Delay_Req it sends (M594).
//!
//! Only the subset a delay-request-response (E2E) slave uses: the common 34-byte
//! header, plus Sync / Follow_Up / Delay_Resp bodies, plus a Delay_Req builder.
//! Announce (BMCA), peer-delay (P2P), management and unicast are out of scope
//! (a SLAVE that just follows whatever master is sending on its domain).
//!
//! Every field is read from the network, so parsing is bounds-checked and returns
//! `None` on a short or malformed buffer rather than panicking, per the parser
//! rules in AGENTS.md. Sub-nanosecond correction bits are dropped (we time in ns).

/// PTP common-header length in bytes.
pub const HEADER_LEN: usize = 34;
/// A PTP timestamp on the wire: 48-bit seconds + 32-bit nanoseconds = 10 bytes.
pub const TIMESTAMP_LEN: usize = 10;
/// Byte offset of the first message body (right after the common header).
pub const BODY_OFFSET: usize = HEADER_LEN;
/// PTP version this parser targets (IEEE 1588-2008).
pub const PTP_VERSION: u8 = 2;
/// Total length of a Delay_Req: header + originTimestamp.
pub const DELAY_REQ_LEN: usize = HEADER_LEN + TIMESTAMP_LEN;

/// The message types a delay-request-response SLAVE cares about (the low nibble
/// of the first header octet); everything else is `Other`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PtpMessageType {
    Sync,
    DelayReq,
    FollowUp,
    DelayResp,
    Announce,
    Other(u8),
}

impl PtpMessageType {
    fn from_nibble(n: u8) -> Self {
        match n & 0x0f {
            0x0 => Self::Sync,
            0x1 => Self::DelayReq,
            0x8 => Self::FollowUp,
            0x9 => Self::DelayResp,
            0xb => Self::Announce,
            other => Self::Other(other),
        }
    }

    fn nibble(self) -> u8 {
        match self {
            Self::Sync => 0x0,
            Self::DelayReq => 0x1,
            Self::FollowUp => 0x8,
            Self::DelayResp => 0x9,
            Self::Announce => 0xb,
            Self::Other(o) => o & 0x0f,
        }
    }
}

/// The parsed PTP common header (the first 34 bytes of every message).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PtpHeader {
    pub message_type: PtpMessageType,
    pub version: u8,
    pub message_length: u16,
    pub domain: u8,
    pub flags: u16,
    /// correctionField in ns (the fractional-ns low 16 bits dropped).
    pub correction_ns: i64,
    pub source_clock_id: [u8; 8],
    pub source_port: u16,
    pub sequence_id: u16,
}

impl PtpHeader {
    /// Parse the common header from the front of `buf`, or `None` if too short.
    pub fn parse(buf: &[u8]) -> Option<Self> {
        if buf.len() < HEADER_LEN {
            return None;
        }
        let correction_raw = i64::from_be_bytes(buf[8..16].try_into().ok()?);
        Some(Self {
            message_type: PtpMessageType::from_nibble(buf[0]),
            version: buf[1] & 0x0f,
            message_length: u16::from_be_bytes([buf[2], buf[3]]),
            domain: buf[4],
            flags: u16::from_be_bytes([buf[6], buf[7]]),
            // correctionField is a 64-bit fixed-point ns value scaled by 2^16.
            correction_ns: correction_raw >> 16,
            source_clock_id: buf[20..28].try_into().ok()?,
            source_port: u16::from_be_bytes([buf[28], buf[29]]),
            sequence_id: u16::from_be_bytes([buf[30], buf[31]]),
        })
    }

    /// twoStepFlag (bit 1 of flagField octet 0): the accurate Sync TX time comes
    /// in a following Follow_Up rather than in the Sync itself.
    pub fn two_step(&self) -> bool {
        self.flags & 0x0200 != 0
    }
}

/// Read a 10-byte PTP timestamp (48-bit seconds + 32-bit ns) at `buf[off..]` as
/// total nanoseconds, or `None` if out of range / overflowing.
pub fn parse_timestamp(buf: &[u8], off: usize) -> Option<u64> {
    let b = buf.get(off..off + TIMESTAMP_LEN)?;
    let secs = (u64::from(b[0]) << 40)
        | (u64::from(b[1]) << 32)
        | (u64::from(b[2]) << 24)
        | (u64::from(b[3]) << 16)
        | (u64::from(b[4]) << 8)
        | u64::from(b[5]);
    let nanos = u64::from(u32::from_be_bytes([b[6], b[7], b[8], b[9]]));
    secs.checked_mul(1_000_000_000)?.checked_add(nanos)
}

/// The originTimestamp of a Sync (meaningful only for a one-step master; a
/// two-step master sends it as zero and the real value in the Follow_Up).
pub fn parse_sync_origin(buf: &[u8]) -> Option<u64> {
    parse_timestamp(buf, BODY_OFFSET)
}

/// The preciseOriginTimestamp of a Follow_Up (the accurate Sync TX time).
pub fn parse_follow_up_origin(buf: &[u8]) -> Option<u64> {
    parse_timestamp(buf, BODY_OFFSET)
}

/// A parsed Delay_Resp body: the master's receiveTimestamp of our Delay_Req plus
/// the requestingPortIdentity it echoes, so a slave can match its own request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DelayResp {
    pub receive_ts_ns: u64,
    pub requesting_clock_id: [u8; 8],
    pub requesting_port: u16,
}

impl DelayResp {
    /// Parse a Delay_Resp body (receiveTimestamp + requestingPortIdentity) from a
    /// full message buffer.
    pub fn parse(buf: &[u8]) -> Option<Self> {
        let receive_ts_ns = parse_timestamp(buf, BODY_OFFSET)?;
        // requestingPortIdentity follows the 10-byte timestamp.
        let id_off = BODY_OFFSET + TIMESTAMP_LEN;
        let clock = buf.get(id_off..id_off + 8)?;
        let port = buf.get(id_off + 8..id_off + 10)?;
        Some(Self {
            receive_ts_ns,
            requesting_clock_id: clock.try_into().ok()?,
            requesting_port: u16::from_be_bytes([port[0], port[1]]),
        })
    }
}

/// Build a Delay_Req message a SLAVE multicasts to the master. The
/// originTimestamp is left zero (a software slave times its own TX on send and
/// carries t3 locally, not in the message).
pub fn build_delay_req(
    domain: u8,
    clock_id: [u8; 8],
    port: u16,
    sequence_id: u16,
) -> [u8; DELAY_REQ_LEN] {
    let mut m = [0u8; DELAY_REQ_LEN];
    m[0] = PtpMessageType::DelayReq.nibble(); // majorSdoId 0 | messageType
    m[1] = PTP_VERSION; // minorVersion 0 | versionPTP 2
    let len = DELAY_REQ_LEN as u16;
    m[2..4].copy_from_slice(&len.to_be_bytes());
    m[4] = domain;
    // flagField 0, correctionField 0 (bytes 6..16 already zero).
    m[20..28].copy_from_slice(&clock_id);
    m[28..30].copy_from_slice(&port.to_be_bytes());
    m[30..32].copy_from_slice(&sequence_id.to_be_bytes());
    m[32] = 0x01; // controlField: Delay_Req (legacy but still set)
    m[33] = 0x7f; // logMessageInterval: 0x7f = "not set" for an event message
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-build a common header into `buf` for tests.
    fn put_header(buf: &mut [u8], mtype: u8, two_step: bool, domain: u8, seq: u16, corr_ns: i64) {
        buf[0] = mtype & 0x0f;
        buf[1] = PTP_VERSION;
        let len = buf.len() as u16;
        buf[2..4].copy_from_slice(&len.to_be_bytes());
        buf[4] = domain;
        let flags: u16 = if two_step { 0x0200 } else { 0 };
        buf[6..8].copy_from_slice(&flags.to_be_bytes());
        buf[8..16].copy_from_slice(&(corr_ns << 16).to_be_bytes());
        buf[20..28].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        buf[28..30].copy_from_slice(&1u16.to_be_bytes());
        buf[30..32].copy_from_slice(&seq.to_be_bytes());
    }

    fn put_timestamp(buf: &mut [u8], off: usize, secs: u64, nanos: u32) {
        buf[off] = (secs >> 40) as u8;
        buf[off + 1] = (secs >> 32) as u8;
        buf[off + 2] = (secs >> 24) as u8;
        buf[off + 3] = (secs >> 16) as u8;
        buf[off + 4] = (secs >> 8) as u8;
        buf[off + 5] = secs as u8;
        buf[off + 6..off + 10].copy_from_slice(&nanos.to_be_bytes());
    }

    #[test]
    fn rejects_short_buffers() {
        assert!(PtpHeader::parse(&[0u8; 10]).is_none());
        assert!(parse_timestamp(&[0u8; 5], 0).is_none());
        assert!(
            DelayResp::parse(&[0u8; HEADER_LEN]).is_none(),
            "no room for the body"
        );
    }

    #[test]
    fn parses_a_two_step_sync_header() {
        let mut buf = [0u8; HEADER_LEN + TIMESTAMP_LEN];
        put_header(&mut buf, 0x0, true, 0, 42, 0);
        let h = PtpHeader::parse(&buf).unwrap();
        assert_eq!(h.message_type, PtpMessageType::Sync);
        assert_eq!(h.version, PTP_VERSION);
        assert_eq!(h.domain, 0);
        assert_eq!(h.sequence_id, 42);
        assert_eq!(h.source_clock_id, [1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(h.two_step(), "twoStepFlag set");
    }

    #[test]
    fn parses_timestamps_and_correction() {
        let mut buf = [0u8; HEADER_LEN + TIMESTAMP_LEN];
        put_header(&mut buf, 0x8, false, 0, 7, 1234); // Follow_Up, correction 1234 ns
        put_timestamp(&mut buf, BODY_OFFSET, 1_700_000_000, 500_000_000);
        let h = PtpHeader::parse(&buf).unwrap();
        assert_eq!(h.message_type, PtpMessageType::FollowUp);
        assert_eq!(h.correction_ns, 1234);
        assert_eq!(
            parse_follow_up_origin(&buf),
            Some(1_700_000_000_500_000_000)
        );
    }

    #[test]
    fn parses_a_delay_resp_body() {
        let mut buf = [0u8; HEADER_LEN + TIMESTAMP_LEN + 10];
        put_header(&mut buf, 0x9, false, 0, 7, 0);
        put_timestamp(&mut buf, BODY_OFFSET, 1_700_000_001, 250);
        // requestingPortIdentity
        buf[BODY_OFFSET + TIMESTAMP_LEN..BODY_OFFSET + TIMESTAMP_LEN + 8]
            .copy_from_slice(&[9, 9, 9, 9, 9, 9, 9, 9]);
        buf[BODY_OFFSET + TIMESTAMP_LEN + 8..BODY_OFFSET + TIMESTAMP_LEN + 10]
            .copy_from_slice(&3u16.to_be_bytes());
        let r = DelayResp::parse(&buf).unwrap();
        assert_eq!(r.receive_ts_ns, 1_700_000_001_000_000_250);
        assert_eq!(r.requesting_clock_id, [9; 8]);
        assert_eq!(r.requesting_port, 3);
    }

    #[test]
    fn builds_a_parseable_delay_req() {
        let m = build_delay_req(0, [1, 2, 3, 4, 5, 6, 7, 8], 1, 99);
        assert_eq!(m.len(), DELAY_REQ_LEN);
        let h = PtpHeader::parse(&m).unwrap();
        assert_eq!(h.message_type, PtpMessageType::DelayReq);
        assert_eq!(h.version, PTP_VERSION);
        assert_eq!(h.sequence_id, 99);
        assert_eq!(h.source_clock_id, [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(h.source_port, 1);
        assert!(!h.two_step());
    }
}
