//! PTP management messages (IEEE 1588-2008 clause 15), the GET half plus the
//! data sets a client needs to read another clock's sync state (M998).
//!
//! This is what `pmc` speaks to `ptp4l`: a management message carries a
//! targetPortIdentity and one management TLV naming a managementId, and the
//! answering clock replies with the same TLV filled in. Only what a read-only
//! status query needs is here: a GET builder, a RESPONSE parser, and the
//! `PORT_DATA_SET` / `CURRENT_DATA_SET` bodies. No SET / COMMAND, no boundary-hop
//! forwarding.
//!
//! Sans-IO, like the rest of `ptp::wire`: the transport (a Unix datagram socket
//! to `ptp4l`, or UDP port 320) belongs to the caller. Every field is read from
//! another process, so parsing is bounds-checked and returns `None` on anything
//! short or malformed.

use crate::ptp::wire::{PtpHeader, PtpMessageType, HEADER_LEN, PTP_VERSION};

/// Length of the management message body: targetPortIdentity (10) +
/// startingBoundaryHops + boundaryHops + flags + reserved.
pub const MANAGEMENT_BODY_LEN: usize = 14;
/// Byte offset of the management TLV (right after header + management body).
pub const TLV_OFFSET: usize = HEADER_LEN + MANAGEMENT_BODY_LEN;
/// Management TLV header: tlvType (2) + lengthField (2) + managementId (2).
pub const TLV_HEADER_LEN: usize = 6;
/// Byte offset of the managed datum inside a management message.
pub const DATA_OFFSET: usize = TLV_OFFSET + TLV_HEADER_LEN;
/// Total length of a GET: header + body + an empty management TLV.
pub const GET_LEN: usize = DATA_OFFSET;

/// tlvType for a management TLV.
const TLV_MANAGEMENT: u16 = 0x0001;
/// actionField values (the low nibble of the management flags octet).
const ACTION_GET: u8 = 0;
const ACTION_RESPONSE: u8 = 2;
/// lengthField counts the managementId plus the datum.
const LENGTH_FIELD_ID_LEN: u16 = 2;
/// controlField for a management message (legacy IEEE 1588-2008 field).
const CONTROL_MANAGEMENT: u8 = 0x04;

/// managementId of the clock-wide currentDS (offset from master, path delay).
pub const CURRENT_DATA_SET: u16 = 0x2001;
/// managementId of the per-port portDS (the port's state machine state).
pub const PORT_DATA_SET: u16 = 0x2004;

/// A PTP port's state machine state (IEEE 1588-2008 clause 8.2.5.3.1). A clock
/// is synced to a grandmaster exactly when one of its ports is
/// [`Slave`](PortState::Slave).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortState {
    Initializing,
    Faulty,
    Disabled,
    Listening,
    PreMaster,
    Master,
    Passive,
    Uncalibrated,
    Slave,
    Other(u8),
}

impl PortState {
    /// Map the wire octet; unknown values (including linuxptp's non-standard
    /// GRAND_MASTER, which it reports as `Master` in a portDS) stay `Other`.
    pub fn from_octet(octet: u8) -> Self {
        match octet {
            1 => Self::Initializing,
            2 => Self::Faulty,
            3 => Self::Disabled,
            4 => Self::Listening,
            5 => Self::PreMaster,
            6 => Self::Master,
            7 => Self::Passive,
            8 => Self::Uncalibrated,
            9 => Self::Slave,
            other => Self::Other(other),
        }
    }
}

/// Build a GET for `management_id` addressed to every port of every clock the
/// message reaches (the wildcard target `pmc` uses by default), so a boundary
/// clock answers once per port.
pub fn build_get(
    domain: u8,
    clock_id: [u8; 8],
    port: u16,
    sequence_id: u16,
    management_id: u16,
) -> [u8; GET_LEN] {
    let mut m = [0u8; GET_LEN];
    m[0] = PtpMessageType::Management.nibble();
    m[1] = PTP_VERSION;
    m[2..4].copy_from_slice(&(GET_LEN as u16).to_be_bytes());
    m[4] = domain;
    m[20..28].copy_from_slice(&clock_id);
    m[28..30].copy_from_slice(&port.to_be_bytes());
    m[30..32].copy_from_slice(&sequence_id.to_be_bytes());
    m[32] = CONTROL_MANAGEMENT;
    m[33] = 0x7f; // logMessageInterval: not set

    // targetPortIdentity: all ones = any clock, any port.
    m[HEADER_LEN..HEADER_LEN + 10].fill(0xff);
    // startingBoundaryHops / boundaryHops 0: do not forward past the first clock.
    m[HEADER_LEN + 12] = ACTION_GET;
    m[TLV_OFFSET..TLV_OFFSET + 2].copy_from_slice(&TLV_MANAGEMENT.to_be_bytes());
    // An empty TLV body is the standard zero-length GET: length covers the id only.
    m[TLV_OFFSET + 2..TLV_OFFSET + 4].copy_from_slice(&LENGTH_FIELD_ID_LEN.to_be_bytes());
    m[TLV_OFFSET + 4..TLV_OFFSET + 6].copy_from_slice(&management_id.to_be_bytes());
    m
}

/// A management RESPONSE: which datum it carries and its raw body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ManagementResponse<'a> {
    pub header: PtpHeader,
    pub management_id: u16,
    pub data: &'a [u8],
}

impl<'a> ManagementResponse<'a> {
    /// Parse a management RESPONSE, or `None` for anything else (a short buffer,
    /// another message type, a GET echoed back, a management-error TLV).
    pub fn parse(buf: &'a [u8]) -> Option<Self> {
        let header = PtpHeader::parse(buf)?;
        if header.message_type != PtpMessageType::Management || header.version != PTP_VERSION {
            return None;
        }
        if buf.get(HEADER_LEN + 12)? & 0x0f != ACTION_RESPONSE {
            return None;
        }
        let tlv = buf.get(TLV_OFFSET..DATA_OFFSET)?;
        if u16::from_be_bytes([tlv[0], tlv[1]]) != TLV_MANAGEMENT {
            return None;
        }
        let length_field = u16::from_be_bytes([tlv[2], tlv[3]]);
        let data_len = usize::from(length_field.checked_sub(LENGTH_FIELD_ID_LEN)?);
        Some(Self {
            header,
            management_id: u16::from_be_bytes([tlv[4], tlv[5]]),
            data: buf.get(DATA_OFFSET..DATA_OFFSET + data_len)?,
        })
    }
}

/// Wire length of a portDS datum.
pub const PORT_DATA_SET_LEN: usize = 26;
/// Wire length of a currentDS datum.
pub const CURRENT_DATA_SET_LEN: usize = 18;

/// The portDS fields a status query reads: which port answered and its state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortDataSet {
    pub clock_id: [u8; 8],
    pub port_number: u16,
    pub port_state: PortState,
}

impl PortDataSet {
    /// Parse a portDS body.
    pub fn parse(data: &[u8]) -> Option<Self> {
        let body = data.get(..PORT_DATA_SET_LEN)?;
        Some(Self {
            clock_id: body[0..8].try_into().ok()?,
            port_number: u16::from_be_bytes([body[8], body[9]]),
            port_state: PortState::from_octet(body[10]),
        })
    }
}

/// The clock-wide currentDS: how far this clock is from its master.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CurrentDataSet {
    pub steps_removed: u16,
    pub offset_from_master_ns: i64,
    pub mean_path_delay_ns: i64,
}

impl CurrentDataSet {
    /// Parse a currentDS body. The two TimeInterval fields are ns scaled by
    /// 2^16, so the sub-ns bits are dropped as elsewhere in the PTP code.
    pub fn parse(data: &[u8]) -> Option<Self> {
        let body = data.get(..CURRENT_DATA_SET_LEN)?;
        Some(Self {
            steps_removed: u16::from_be_bytes([body[0], body[1]]),
            offset_from_master_ns: i64::from_be_bytes(body[2..10].try_into().ok()?) >> 16,
            mean_path_delay_ns: i64::from_be_bytes(body[10..18].try_into().ok()?) >> 16,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Wrap `data` in a management RESPONSE the way `ptp4l` answers a GET, built
    /// from the IEEE 1588 / linuxptp field layout independently of the parser.
    fn response(management_id: u16, data: &[u8]) -> alloc::vec::Vec<u8> {
        let mut m = alloc::vec![0u8; DATA_OFFSET + data.len()];
        m[0] = 0x0d; // messageType: Management
        m[1] = 0x12; // minorVersionPTP 1 | versionPTP 2, as linuxptp sends
        let len = m.len() as u16;
        m[2..4].copy_from_slice(&len.to_be_bytes());
        m[20..28].copy_from_slice(&[0x11; 8]);
        m[28..30].copy_from_slice(&1u16.to_be_bytes());
        m[30..32].copy_from_slice(&7u16.to_be_bytes());
        m[HEADER_LEN + 12] = ACTION_RESPONSE;
        m[TLV_OFFSET..TLV_OFFSET + 2].copy_from_slice(&TLV_MANAGEMENT.to_be_bytes());
        let length_field = LENGTH_FIELD_ID_LEN + data.len() as u16;
        m[TLV_OFFSET + 2..TLV_OFFSET + 4].copy_from_slice(&length_field.to_be_bytes());
        m[TLV_OFFSET + 4..TLV_OFFSET + 6].copy_from_slice(&management_id.to_be_bytes());
        m[DATA_OFFSET..].copy_from_slice(data);
        m
    }

    fn port_data_set_body(port_number: u16, state: u8) -> [u8; PORT_DATA_SET_LEN] {
        let mut d = [0u8; PORT_DATA_SET_LEN];
        d[0..8].copy_from_slice(&[0xaa; 8]);
        d[8..10].copy_from_slice(&port_number.to_be_bytes());
        d[10] = state;
        d
    }

    #[test]
    fn builds_a_wildcard_zero_length_get() {
        let m = build_get(0, [1, 2, 3, 4, 5, 6, 7, 8], 1, 42, PORT_DATA_SET);
        let h = PtpHeader::parse(&m).unwrap();
        assert_eq!(h.message_type, PtpMessageType::Management);
        assert_eq!(h.message_length as usize, GET_LEN);
        assert_eq!(h.sequence_id, 42);
        assert_eq!(h.source_clock_id, [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            &m[HEADER_LEN..HEADER_LEN + 10],
            &[0xff; 10],
            "wildcard target"
        );
        assert_eq!(m[HEADER_LEN + 12] & 0x0f, ACTION_GET);
        assert_eq!(
            &m[TLV_OFFSET..TLV_OFFSET + 2],
            &TLV_MANAGEMENT.to_be_bytes()
        );
        assert_eq!(
            u16::from_be_bytes([m[TLV_OFFSET + 2], m[TLV_OFFSET + 3]]),
            LENGTH_FIELD_ID_LEN,
            "a GET carries no datum"
        );
        assert_eq!(
            u16::from_be_bytes([m[TLV_OFFSET + 4], m[TLV_OFFSET + 5]]),
            PORT_DATA_SET
        );
    }

    #[test]
    fn parses_a_port_data_set_response() {
        let m = response(PORT_DATA_SET, &port_data_set_body(2, 9));
        let r = ManagementResponse::parse(&m).unwrap();
        assert_eq!(r.management_id, PORT_DATA_SET);
        assert_eq!(r.header.sequence_id, 7);
        let pds = PortDataSet::parse(r.data).unwrap();
        assert_eq!(pds.port_number, 2);
        assert_eq!(pds.port_state, PortState::Slave);
        assert_eq!(pds.clock_id, [0xaa; 8]);
    }

    #[test]
    fn parses_a_current_data_set_response_with_a_negative_offset() {
        let mut body = [0u8; CURRENT_DATA_SET_LEN];
        body[0..2].copy_from_slice(&1u16.to_be_bytes());
        // TimeInterval is ns << 16: -1234 ns offset, 5678 ns path delay.
        body[2..10].copy_from_slice(&(-1234i64 << 16).to_be_bytes());
        body[10..18].copy_from_slice(&(5678i64 << 16).to_be_bytes());
        let m = response(CURRENT_DATA_SET, &body);
        let r = ManagementResponse::parse(&m).unwrap();
        let cds = CurrentDataSet::parse(r.data).unwrap();
        assert_eq!(cds.steps_removed, 1);
        assert_eq!(cds.offset_from_master_ns, -1234);
        assert_eq!(cds.mean_path_delay_ns, 5678);
    }

    #[test]
    fn rejects_non_responses_and_short_buffers() {
        assert!(ManagementResponse::parse(&[0u8; 10]).is_none());
        assert!(
            ManagementResponse::parse(&build_get(0, [0; 8], 1, 1, PORT_DATA_SET)).is_none(),
            "a GET is not a RESPONSE"
        );

        let mut wrong_tlv = response(PORT_DATA_SET, &port_data_set_body(1, 9));
        wrong_tlv[TLV_OFFSET + 1] = 0x02; // TLV_MANAGEMENT_ERROR_STATUS
        assert!(ManagementResponse::parse(&wrong_tlv).is_none());

        let mut truncated = response(PORT_DATA_SET, &port_data_set_body(1, 9));
        truncated.truncate(truncated.len() - 1);
        assert!(
            ManagementResponse::parse(&truncated).is_none(),
            "lengthField promises more datum than the buffer holds"
        );

        let short_datum = response(PORT_DATA_SET, &[0u8; 4]);
        let r = ManagementResponse::parse(&short_datum).unwrap();
        assert!(PortDataSet::parse(r.data).is_none());
        assert!(CurrentDataSet::parse(r.data).is_none());
    }
}
