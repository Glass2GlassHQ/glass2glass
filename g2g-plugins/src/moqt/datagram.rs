//! Datagram objects for IETF MoQ Transport draft-16: one object in one QUIC
//! datagram, unreliable and bounded by the path MTU, with no head-of-line
//! blocking against anything else on the session.
//!
//! Layout from `moq-rs/moq-transport/src/data/datagram.rs`. The type is a bit
//! table like the subgroup stream header: which of the object id, the extension
//! headers and the object status are present, whether a payload follows, and
//! whether the object ends its group. The payload has no length prefix, since
//! the datagram boundary already ends it.
//!
//! A datagram is a whole message that will never be continued, so a short one is
//! malformed rather than incomplete, and everything a peer put in it (ids,
//! extension length, payload length) is bounded before use.

use alloc::vec::Vec;

use super::coding::{put_varint, MoqtError, Params, Reader};
use super::data::ObjectStatus;
use super::reassembly::ReceivedObject;

/// Datagram header types (`data/datagram.rs`). The name says which optional
/// fields the datagram carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatagramType {
    ObjectIdPayload = 0x00,
    ObjectIdPayloadExt = 0x01,
    ObjectIdPayloadEndOfGroup = 0x02,
    ObjectIdPayloadExtEndOfGroup = 0x03,
    Payload = 0x04,
    PayloadExt = 0x05,
    PayloadEndOfGroup = 0x06,
    PayloadExtEndOfGroup = 0x07,
    ObjectIdStatus = 0x20,
    ObjectIdStatusExt = 0x21,
}

impl DatagramType {
    pub fn code(self) -> u64 {
        self as u64
    }

    pub fn from_code(code: u64) -> Result<Self, MoqtError> {
        Ok(match code {
            0x00 => Self::ObjectIdPayload,
            0x01 => Self::ObjectIdPayloadExt,
            0x02 => Self::ObjectIdPayloadEndOfGroup,
            0x03 => Self::ObjectIdPayloadExtEndOfGroup,
            0x04 => Self::Payload,
            0x05 => Self::PayloadExt,
            0x06 => Self::PayloadEndOfGroup,
            0x07 => Self::PayloadExtEndOfGroup,
            0x20 => Self::ObjectIdStatus,
            0x21 => Self::ObjectIdStatusExt,
            _ => return Err(MoqtError::Malformed),
        })
    }

    /// Whether an explicit object id follows the group id. Without one the
    /// object id is zero, which is how the reference subscriber reads it.
    pub fn has_object_id(self) -> bool {
        !matches!(
            self,
            Self::Payload | Self::PayloadExt | Self::PayloadEndOfGroup | Self::PayloadExtEndOfGroup
        )
    }

    pub fn has_extension_headers(self) -> bool {
        matches!(
            self,
            Self::ObjectIdPayloadExt
                | Self::ObjectIdPayloadExtEndOfGroup
                | Self::PayloadExt
                | Self::PayloadExtEndOfGroup
                | Self::ObjectIdStatusExt
        )
    }

    pub fn has_status(self) -> bool {
        matches!(self, Self::ObjectIdStatus | Self::ObjectIdStatusExt)
    }

    pub fn has_payload(self) -> bool {
        !self.has_status()
    }

    /// Whether this object is the last of its group.
    pub fn ends_group(self) -> bool {
        matches!(
            self,
            Self::ObjectIdPayloadEndOfGroup
                | Self::ObjectIdPayloadExtEndOfGroup
                | Self::PayloadEndOfGroup
                | Self::PayloadExtEndOfGroup
        )
    }
}

/// One object carried in a datagram.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatagramObject {
    pub datagram_type: DatagramType,
    pub track_alias: u64,
    pub group_id: u64,
    /// Zero for the types that carry no explicit id.
    pub object_id: u64,
    /// Smaller values are sent first.
    pub publisher_priority: u8,
    pub extension_headers: Params,
    /// The status the datagram carries; `Normal` for the types that carry none.
    pub status: ObjectStatus,
    /// Empty for the status-only types.
    pub payload: Vec<u8>,
}

impl DatagramObject {
    /// A normal media object, the form the reference publisher sends when it has
    /// no extension headers (`session/subscribed.rs`, `serve_datagrams`).
    pub fn media(
        track_alias: u64,
        group_id: u64,
        object_id: u64,
        publisher_priority: u8,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            datagram_type: DatagramType::ObjectIdPayload,
            track_alias,
            group_id,
            object_id,
            publisher_priority,
            extension_headers: Params::new(),
            status: ObjectStatus::Normal,
            payload,
        }
    }

    /// A status-only object closing `group_id` at `object_id`: nothing at or
    /// past that id will exist. This is what tells a subscriber a group carried
    /// only by datagrams is done, since no stream ends to say so.
    pub fn end_of_group(
        track_alias: u64,
        group_id: u64,
        object_id: u64,
        publisher_priority: u8,
    ) -> Self {
        Self {
            datagram_type: DatagramType::ObjectIdStatus,
            track_alias,
            group_id,
            object_id,
            publisher_priority,
            extension_headers: Params::new(),
            status: ObjectStatus::EndOfGroup,
            payload: Vec::new(),
        }
    }

    /// The status the receiver should act on: the type's end-of-group marking
    /// folded into the carried status.
    pub fn object_status(&self) -> ObjectStatus {
        if self.datagram_type.ends_group() {
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
        if ty.has_extension_headers() == self.extension_headers.0.is_empty() {
            // The type and the block must agree: an extension type with nothing
            // to write, or a plain type with headers that would land on the wire
            // where the peer reads something else.
            return Err(MoqtError::Malformed);
        }
        if !ty.has_payload() && !self.payload.is_empty() {
            return Err(MoqtError::Malformed);
        }
        if ty.has_status()
            && self.status != ObjectStatus::Normal
            && !self.extension_headers.0.is_empty()
        {
            return Err(MoqtError::Malformed);
        }
        put_varint(out, ty.code());
        put_varint(out, self.track_alias);
        put_varint(out, self.group_id);
        if ty.has_object_id() {
            put_varint(out, self.object_id);
        }
        out.push(self.publisher_priority);
        if ty.has_extension_headers() {
            self.extension_headers.encode_extension_headers(out)?;
        }
        if ty.has_status() {
            put_varint(out, self.status as u64);
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
        let mut r = Reader::new(bytes);
        let datagram_type = DatagramType::from_code(r.varint()?)?;
        let track_alias = r.varint()?;
        let group_id = r.varint()?;
        let object_id = if datagram_type.has_object_id() {
            r.varint()?
        } else {
            0
        };
        let publisher_priority = r.u8()?;
        let extension_headers = if datagram_type.has_extension_headers() {
            let headers = Params::decode_extension_headers(&mut r)?;
            // An extension type with an empty block is a violation, not an
            // alternative spelling of the plain type (`data/datagram.rs`).
            if headers.0.is_empty() {
                return Err(MoqtError::Malformed);
            }
            headers
        } else {
            Params::new()
        };
        let status = if datagram_type.has_status() {
            ObjectStatus::from_code(r.varint()?)?
        } else {
            ObjectStatus::Normal
        };
        if status != ObjectStatus::Normal && !extension_headers.0.is_empty() {
            return Err(MoqtError::Malformed);
        }
        // The rest of the datagram is the payload: status-only types carry none,
        // and the reference ignores anything trailing them.
        let payload = if datagram_type.has_payload() {
            let rest = r.rest();
            if rest.len() > max_payload {
                return Err(MoqtError::Malformed);
            }
            rest.to_vec()
        } else {
            Vec::new()
        };
        Ok(Self {
            datagram_type,
            track_alias,
            group_id,
            object_id,
            publisher_priority,
            extension_headers,
            status,
            payload,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Byte vectors printed by the reference encoder itself
    /// (`moq-transport/src/data/datagram.rs` through a throwaway example), not
    /// transcribed from the draft: a round trip alone cannot catch two fields
    /// swapped with each other.
    #[test]
    fn datagram_layouts_match_the_reference_encoder() {
        let mut out = Vec::new();
        DatagramObject::media(12, 10, 1234, 127, b"payload".to_vec())
            .encode(&mut out)
            .expect("encode");
        assert_eq!(
            out,
            vec![0x00, 0x0c, 0x0a, 0x44, 0xd2, 0x7f, b'p', b'a', b'y', b'l', b'o', b'a', b'd']
        );

        let mut ext = Params::new();
        ext.set_int(0, 42);
        let mut out = Vec::new();
        DatagramObject {
            datagram_type: DatagramType::ObjectIdPayloadExt,
            extension_headers: ext,
            ..DatagramObject::media(12, 10, 1234, 127, b"payload".to_vec())
        }
        .encode(&mut out)
        .expect("encode");
        assert_eq!(
            out,
            vec![
                0x01, 0x0c, 0x0a, 0x44, 0xd2, 0x7f, 0x02, 0x00, 0x2a, b'p', b'a', b'y', b'l', b'o',
                b'a', b'd'
            ]
        );

        let mut out = Vec::new();
        DatagramObject::end_of_group(4, 7, 3, 127)
            .encode(&mut out)
            .expect("encode");
        assert_eq!(out, vec![0x20, 0x04, 0x07, 0x03, 0x7f, 0x03]);

        // A type without an object id omits it entirely.
        let mut out = Vec::new();
        DatagramObject {
            datagram_type: DatagramType::PayloadEndOfGroup,
            ..DatagramObject::media(12, 10, 0, 127, b"payload".to_vec())
        }
        .encode(&mut out)
        .expect("encode");
        assert_eq!(
            out,
            vec![0x06, 0x0c, 0x0a, 0x7f, b'p', b'a', b'y', b'l', b'o', b'a', b'd']
        );
    }

    #[test]
    fn every_type_round_trips_and_ends_its_group_where_the_type_says() {
        for ty in [
            DatagramType::ObjectIdPayload,
            DatagramType::ObjectIdPayloadEndOfGroup,
            DatagramType::Payload,
            DatagramType::PayloadEndOfGroup,
        ] {
            let object = DatagramObject {
                datagram_type: ty,
                object_id: if ty.has_object_id() { 9 } else { 0 },
                ..DatagramObject::media(3, 4, 0, 12, b"media".to_vec())
            };
            let mut out = Vec::new();
            object.encode(&mut out).expect("encode");
            assert_eq!(DatagramObject::decode(&out, 1 << 20), Ok(object.clone()));
            assert_eq!(
                object.object_status() == ObjectStatus::EndOfGroup,
                ty.ends_group()
            );
            // An end-of-group datagram still carries its media.
            assert_eq!(object.into_received().payload, b"media".to_vec());
        }

        let marker = DatagramObject::end_of_group(4, 7, 3, 127);
        let mut out = Vec::new();
        marker.encode(&mut out).expect("encode");
        assert_eq!(DatagramObject::decode(&out, 1 << 20), Ok(marker.clone()));
        let received = marker.into_received();
        assert_eq!(received.status, ObjectStatus::EndOfGroup);
        assert!(received.payload.is_empty(), "a marker carries no media");
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

        // An extension block whose length overruns the datagram.
        assert_eq!(
            DatagramObject::decode(&[0x01, 0x0c, 0x0a, 0x01, 0x7f, 0x20, 0x00], 1 << 20),
            Err(MoqtError::Malformed)
        );
        // ...and one that is present but empty, which the type forbids.
        assert_eq!(
            DatagramObject::decode(&[0x01, 0x0c, 0x0a, 0x01, 0x7f, 0x00], 1 << 20),
            Err(MoqtError::Malformed)
        );

        // A reserved type, and a reserved status.
        assert_eq!(
            DatagramObject::decode(&[0x08, 0x0c, 0x0a, 0x01, 0x7f], 1 << 20),
            Err(MoqtError::Malformed)
        );
        assert_eq!(
            DatagramObject::decode(&[0x20, 0x04, 0x07, 0x03, 0x7f, 0x01], 1 << 20),
            Err(MoqtError::Malformed)
        );

        // A non-normal status with extension headers is a violation both ways.
        assert_eq!(
            DatagramObject::decode(
                &[0x21, 0x01, 0x01, 0x01, 0x7f, 0x02, 0x00, 0x01, 0x04],
                1 << 20
            ),
            Err(MoqtError::Malformed)
        );
        let mut ext = Params::new();
        ext.set_int(0, 1);
        assert_eq!(
            DatagramObject {
                datagram_type: DatagramType::ObjectIdStatusExt,
                extension_headers: ext,
                status: ObjectStatus::EndOfTrack,
                payload: Vec::new(),
                ..DatagramObject::media(1, 1, 1, 1, Vec::new())
            }
            .encode(&mut Vec::new()),
            Err(MoqtError::Malformed)
        );

        // Encoding headers a plain type cannot carry is refused rather than
        // written where the peer reads a payload.
        let mut ext = Params::new();
        ext.set_int(0, 1);
        assert_eq!(
            DatagramObject {
                extension_headers: ext,
                ..DatagramObject::media(1, 1, 1, 1, b"x".to_vec())
            }
            .encode(&mut Vec::new()),
            Err(MoqtError::Malformed)
        );
        // ...and a status-only type may not carry a payload.
        assert_eq!(
            DatagramObject {
                payload: b"x".to_vec(),
                ..DatagramObject::end_of_group(1, 1, 1, 1)
            }
            .encode(&mut Vec::new()),
            Err(MoqtError::Malformed)
        );
    }
}
