//! Datagram objects for IETF MoQ Transport draft-18 (§11.3): one object in one
//! QUIC datagram, unreliable and bounded by the path MTU, with no head-of-line
//! blocking against anything else on the session.
//!
//! Like the subgroup header, the type became a bit table. Two bits are new
//! relative to draft-16: `ZERO_OBJECT_ID` drops the object id field, and
//! `DEFAULT_PRIORITY` drops the priority byte, so the smallest draft-18 object
//! datagram is a type, a track alias and a group id.
//!
//! A datagram is a whole message that will never be continued, so a short one is
//! malformed rather than incomplete, and everything a peer put in it (ids,
//! properties length, payload length) is bounded before use.

use alloc::vec::Vec;

use super::super::reassembly::ReceivedObject;
use super::coding::{
    decode_kvps_length_prefixed, encode_kvps_length_prefixed, put_vi64, reader, MoqtError, Params,
};
use super::data::ObjectStatus;

/// OBJECT_DATAGRAM type bits (§11.3.1).
mod flag {
    pub(super) const PROPERTIES: u64 = 0x01;
    pub(super) const END_OF_GROUP: u64 = 0x02;
    pub(super) const ZERO_OBJECT_ID: u64 = 0x04;
    pub(super) const DEFAULT_PRIORITY: u64 = 0x08;
    pub(super) const STATUS: u64 = 0x20;
    /// Bits that must be zero for the type to match the form `0b00X0XXXX`.
    pub(super) const FORBIDDEN: u64 = 0xd0;
}

/// The bits of an OBJECT_DATAGRAM type byte.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DatagramType {
    pub properties: bool,
    /// No object with a greater object id exists in this group.
    pub end_of_group: bool,
    /// The object id field is omitted and the object id is 0.
    pub zero_object_id: bool,
    /// The priority byte is omitted; the subscription's priority applies.
    pub default_priority: bool,
    /// An Object Status replaces the payload.
    pub status: bool,
}

impl DatagramType {
    pub fn code(self) -> u64 {
        let mut code = 0u64;
        for (set, bit) in [
            (self.properties, flag::PROPERTIES),
            (self.end_of_group, flag::END_OF_GROUP),
            (self.zero_object_id, flag::ZERO_OBJECT_ID),
            (self.default_priority, flag::DEFAULT_PRIORITY),
            (self.status, flag::STATUS),
        ] {
            if set {
                code |= bit;
            }
        }
        code
    }

    pub fn from_code(code: u64) -> Result<Self, MoqtError> {
        // The form is 0b00X0XXXX: only bit 5 and the low four bits may be set.
        if code > 0x2f || code & flag::FORBIDDEN != 0 {
            return Err(MoqtError::Malformed);
        }
        let ty = Self {
            properties: code & flag::PROPERTIES != 0,
            end_of_group: code & flag::END_OF_GROUP != 0,
            zero_object_id: code & flag::ZERO_OBJECT_ID != 0,
            default_priority: code & flag::DEFAULT_PRIORITY != 0,
            status: code & flag::STATUS != 0,
        };
        // An object status message cannot also signal end of group.
        if ty.status && ty.end_of_group {
            return Err(MoqtError::Malformed);
        }
        Ok(ty)
    }
}

/// One object carried in a datagram.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatagramObject {
    pub datagram_type: DatagramType,
    pub track_alias: u64,
    pub group_id: u64,
    /// Zero for the types that omit the field.
    pub object_id: u64,
    /// Present only when `default_priority` is clear. Smaller is sent first.
    pub publisher_priority: Option<u8>,
    pub properties: Params,
    /// The status the datagram carries; `Normal` when the type carries none.
    pub status: ObjectStatus,
    /// Empty for the status-only types.
    pub payload: Vec<u8>,
}

impl DatagramObject {
    /// A normal media object with an explicit id and priority and no properties.
    pub fn media(
        track_alias: u64,
        group_id: u64,
        object_id: u64,
        publisher_priority: u8,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            datagram_type: DatagramType::default(),
            track_alias,
            group_id,
            object_id,
            publisher_priority: Some(publisher_priority),
            properties: Params::new(),
            status: ObjectStatus::Normal,
            payload,
        }
    }

    /// A status-only object closing `group_id` at `object_id`: nothing at or past
    /// that id will exist. This is what tells a subscriber a group carried only
    /// by datagrams is done, since no stream ends to say so.
    pub fn end_of_group(
        track_alias: u64,
        group_id: u64,
        object_id: u64,
        publisher_priority: u8,
    ) -> Self {
        Self {
            datagram_type: DatagramType {
                status: true,
                ..DatagramType::default()
            },
            status: ObjectStatus::EndOfGroup,
            payload: Vec::new(),
            ..Self::media(
                track_alias,
                group_id,
                object_id,
                publisher_priority,
                Vec::new(),
            )
        }
    }

    /// The status the receiver should act on: the type's end-of-group bit folded
    /// into the carried status.
    pub fn object_status(&self) -> ObjectStatus {
        if self.datagram_type.end_of_group {
            ObjectStatus::EndOfGroup
        } else {
            self.status
        }
    }

    pub fn into_received(self) -> ReceivedObject {
        ReceivedObject {
            group_id: self.group_id,
            object_id: self.object_id,
            status: self.object_status(),
            payload: self.payload,
        }
    }

    pub fn encode(&self, out: &mut Vec<u8>) -> Result<(), MoqtError> {
        let ty = self.datagram_type;
        // Every optional field's presence is fixed by the type, so a mismatch
        // would put bytes where the peer reads something else.
        if ty.properties == self.properties.0.is_empty() {
            return Err(MoqtError::Malformed);
        }
        if ty.default_priority == self.publisher_priority.is_some() {
            return Err(MoqtError::Malformed);
        }
        if ty.zero_object_id && self.object_id != 0 {
            return Err(MoqtError::Malformed);
        }
        if ty.status && !self.payload.is_empty() {
            return Err(MoqtError::Malformed);
        }
        // §11.3.1: only a Normal object can carry properties. The END_OF_GROUP
        // type bit does not make the object's own status non-Normal, so only
        // the wire status field is checked here, matching the decode side.
        if self.status != ObjectStatus::Normal && !self.properties.0.is_empty() {
            return Err(MoqtError::Malformed);
        }
        put_vi64(out, ty.code());
        put_vi64(out, self.track_alias);
        put_vi64(out, self.group_id);
        if !ty.zero_object_id {
            put_vi64(out, self.object_id);
        }
        if let Some(priority) = self.publisher_priority {
            out.push(priority);
        }
        if ty.properties {
            encode_kvps_length_prefixed(&self.properties, out)?;
        }
        if ty.status {
            put_vi64(out, self.status as u64);
        }
        out.extend_from_slice(&self.payload);
        Ok(())
    }

    /// Decode one whole datagram. `max_payload` caps the object, so a peer
    /// cannot make one datagram carry more than the subscriber allows.
    pub fn decode(bytes: &[u8], max_payload: usize) -> Result<Self, MoqtError> {
        Self::read(bytes, max_payload).map_err(|e| match e {
            // Nothing follows a datagram, so a prefix of one is a violation
            // rather than something to wait for.
            MoqtError::Incomplete => MoqtError::Malformed,
            e => e,
        })
    }

    fn read(bytes: &[u8], max_payload: usize) -> Result<Self, MoqtError> {
        let mut r = reader(bytes);
        let datagram_type = DatagramType::from_code(r.varint()?)?;
        let track_alias = r.varint()?;
        let group_id = r.varint()?;
        let object_id = if datagram_type.zero_object_id {
            0
        } else {
            r.varint()?
        };
        let publisher_priority = if datagram_type.default_priority {
            None
        } else {
            Some(r.u8()?)
        };
        let properties = if datagram_type.properties {
            let properties = decode_kvps_length_prefixed(&mut r)?;
            // A properties bit with a zero length is a violation, not an
            // alternative spelling of the plain type (§11.3.1).
            if properties.0.is_empty() {
                return Err(MoqtError::Malformed);
            }
            properties
        } else {
            Params::new()
        };
        let status = if datagram_type.status {
            ObjectStatus::from_code(r.varint()?)?
        } else {
            ObjectStatus::Normal
        };
        if status != ObjectStatus::Normal && !properties.0.is_empty() {
            return Err(MoqtError::Malformed);
        }
        // The rest of the datagram is the payload; a status-only type carries
        // none, and anything trailing it is a violation.
        let payload = if datagram_type.status {
            if !r.is_empty() {
                return Err(MoqtError::Malformed);
            }
            Vec::new()
        } else {
            let rest = r.rest();
            if rest.len() > max_payload {
                return Err(MoqtError::Malformed);
            }
            rest.to_vec()
        };
        Ok(Self {
            datagram_type,
            track_alias,
            group_id,
            object_id,
            publisher_priority,
            properties,
            status,
            payload,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn datagram_layouts_follow_the_bit_table() {
        // The plain media form: explicit id, explicit priority, payload.
        let mut out = Vec::new();
        DatagramObject::media(12, 10, 1234, 127, b"payload".to_vec())
            .encode(&mut out)
            .expect("encode");
        // type 0x00, alias 12, group 10, id 1234 (two-byte vi64), priority
        assert_eq!(
            out,
            vec![0x00, 0x0c, 0x0a, 0x84, 0xd2, 0x7f, b'p', b'a', b'y', b'l', b'o', b'a', b'd']
        );

        // The end-of-group marker: the STATUS bit, and no payload.
        let mut out = Vec::new();
        DatagramObject::end_of_group(4, 7, 3, 127)
            .encode(&mut out)
            .expect("encode");
        assert_eq!(out, vec![0x20, 0x04, 0x07, 0x03, 0x7f, 0x03]);

        // The two new bits together: no object id, no priority byte. Three
        // varints and a payload is the smallest draft-18 object datagram.
        let lean = DatagramObject {
            datagram_type: DatagramType {
                zero_object_id: true,
                default_priority: true,
                ..DatagramType::default()
            },
            object_id: 0,
            publisher_priority: None,
            ..DatagramObject::media(12, 10, 0, 0, b"x".to_vec())
        };
        let mut out = Vec::new();
        lean.encode(&mut out).expect("encode");
        assert_eq!(out, vec![0x0c, 0x0c, 0x0a, b'x']);
        assert_eq!(DatagramObject::decode(&out, 1 << 20), Ok(lean));
    }

    #[test]
    fn every_valid_type_round_trips() {
        let mut props = Params::new();
        props.set_int(super::super::coding::property::PRIOR_OBJECT_ID_GAP, 2);
        for code in (0x00..=0x0fu64).chain(0x20..=0x2f) {
            let Ok(ty) = DatagramType::from_code(code) else {
                // The eight STATUS + END_OF_GROUP combinations.
                assert!(
                    code & flag::STATUS != 0 && code & flag::END_OF_GROUP != 0,
                    "{code:#x} should be a valid datagram type"
                );
                continue;
            };
            assert_eq!(ty.code(), code, "{code:#x} re-encodes to itself");
            let object = DatagramObject {
                datagram_type: ty,
                object_id: if ty.zero_object_id { 0 } else { 9 },
                publisher_priority: (!ty.default_priority).then_some(12),
                properties: if ty.properties {
                    props.clone()
                } else {
                    Params::new()
                },
                status: if ty.status {
                    ObjectStatus::EndOfGroup
                } else {
                    ObjectStatus::Normal
                },
                payload: if ty.status {
                    Vec::new()
                } else {
                    b"media".to_vec()
                },
                ..DatagramObject::media(3, 4, 0, 0, Vec::new())
            };
            let mut out = Vec::new();
            // A status-only object may not carry properties, so those two bits
            // together have nothing legal to encode.
            if ty.status && ty.properties {
                assert_eq!(object.encode(&mut out), Err(MoqtError::Malformed));
                continue;
            }
            object.encode(&mut out).expect("encode");
            assert_eq!(
                DatagramObject::decode(&out, 1 << 20),
                Ok(object.clone()),
                "{code:#x} round trip"
            );
            assert_eq!(
                object.object_status() == ObjectStatus::EndOfGroup,
                ty.end_of_group || ty.status,
                "{code:#x} end of group"
            );
        }
    }

    #[test]
    fn malformed_datagrams_fail_the_decode() {
        let mut full = Vec::new();
        DatagramObject::media(12, 10, 1234, 127, b"payload".to_vec())
            .encode(&mut full)
            .expect("encode");
        // Every prefix of a datagram is malformed: nothing more is coming.
        for cut in 0..6 {
            assert_eq!(
                DatagramObject::decode(&full[..cut], 1 << 20),
                Err(MoqtError::Malformed),
                "a {cut}-byte datagram"
            );
        }

        // A payload past the bound is refused without allocating on it.
        assert_eq!(
            DatagramObject::decode(&full, 3),
            Err(MoqtError::Malformed),
            "seven payload bytes against a three byte bound"
        );

        // Types outside the form, and the STATUS + END_OF_GROUP combinations the
        // draft lists as invalid.
        for code in [0x10u64, 0x1f, 0x30, 0x40, 0x80, 0xff] {
            assert_eq!(
                DatagramType::from_code(code),
                Err(MoqtError::Malformed),
                "{code:#x} is outside the datagram form"
            );
        }
        for code in [0x22u64, 0x23, 0x26, 0x27, 0x2a, 0x2b, 0x2e, 0x2f] {
            assert_eq!(
                DatagramType::from_code(code),
                Err(MoqtError::Malformed),
                "{code:#x} claims a status and an end of group"
            );
        }

        // A properties block that is present but empty, which the bit forbids.
        assert_eq!(
            DatagramObject::decode(&[0x01, 0x0c, 0x0a, 0x01, 0x7f, 0x00], 1 << 20),
            Err(MoqtError::Malformed)
        );
        // ...and one whose length overruns the datagram.
        assert_eq!(
            DatagramObject::decode(&[0x01, 0x0c, 0x0a, 0x01, 0x7f, 0x20, 0x00], 1 << 20),
            Err(MoqtError::Malformed)
        );

        // A reserved status value.
        assert_eq!(
            DatagramObject::decode(&[0x20, 0x04, 0x07, 0x03, 0x7f, 0x01], 1 << 20),
            Err(MoqtError::Malformed)
        );
        // A status-only datagram with bytes after the status.
        assert_eq!(
            DatagramObject::decode(&[0x20, 0x04, 0x07, 0x03, 0x7f, 0x03, 0x00], 1 << 20),
            Err(MoqtError::Malformed)
        );
        // A non-normal status with properties.
        assert_eq!(
            DatagramObject::decode(
                &[0x21, 0x04, 0x07, 0x03, 0x7f, 0x02, 0x3e, 0x02, 0x03],
                1 << 20
            ),
            Err(MoqtError::Malformed)
        );
    }

    #[test]
    fn encoding_fields_the_type_does_not_carry_is_refused() {
        // Properties with no properties bit.
        let mut props = Params::new();
        props.set_int(super::super::coding::property::PRIOR_OBJECT_ID_GAP, 2);
        assert_eq!(
            DatagramObject {
                properties: props,
                ..DatagramObject::media(1, 1, 1, 1, b"x".to_vec())
            }
            .encode(&mut Vec::new()),
            Err(MoqtError::Malformed)
        );
        // A status-only type with a payload.
        assert_eq!(
            DatagramObject {
                payload: b"x".to_vec(),
                ..DatagramObject::end_of_group(1, 1, 1, 1)
            }
            .encode(&mut Vec::new()),
            Err(MoqtError::Malformed)
        );
        // A priority the type says is absent, and a non-zero id the type says is
        // zero.
        assert_eq!(
            DatagramObject {
                datagram_type: DatagramType {
                    default_priority: true,
                    ..DatagramType::default()
                },
                ..DatagramObject::media(1, 1, 1, 1, b"x".to_vec())
            }
            .encode(&mut Vec::new()),
            Err(MoqtError::Malformed)
        );
        assert_eq!(
            DatagramObject {
                datagram_type: DatagramType {
                    zero_object_id: true,
                    ..DatagramType::default()
                },
                ..DatagramObject::media(1, 1, 7, 1, b"x".to_vec())
            }
            .encode(&mut Vec::new()),
            Err(MoqtError::Malformed)
        );
    }
}
