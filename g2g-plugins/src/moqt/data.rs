//! Data-stream headers for IETF MoQ Transport draft-16: the subgroup stream
//! header that opens a unidirectional QUIC stream and the per-object header
//! that precedes each payload.
//!
//! Layout from `moq-rs/moq-transport/src/data/{header.rs,subgroup.rs}`. The
//! stream header type is a bit table: which of the subgroup id, the extension
//! headers, and the end-of-group marker are present.

use alloc::vec::Vec;

use super::coding::{put_varint, MoqtError, Params, Reader};

/// Stream header types (`data/header.rs`). The name says which optional
/// fields the header and its objects carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamHeaderType {
    SubgroupZeroId = 0x10,
    SubgroupZeroIdExt = 0x11,
    SubgroupFirstObjectId = 0x12,
    SubgroupFirstObjectIdExt = 0x13,
    SubgroupId = 0x14,
    SubgroupIdExt = 0x15,
    SubgroupZeroIdEndOfGroup = 0x18,
    SubgroupZeroIdExtEndOfGroup = 0x19,
    SubgroupFirstObjectIdEndOfGroup = 0x1a,
    SubgroupFirstObjectIdExtEndOfGroup = 0x1b,
    SubgroupIdEndOfGroup = 0x1c,
    SubgroupIdExtEndOfGroup = 0x1d,
}

impl StreamHeaderType {
    pub fn code(self) -> u64 {
        self as u64
    }

    pub fn from_code(code: u64) -> Result<Self, MoqtError> {
        Ok(match code {
            0x10 => Self::SubgroupZeroId,
            0x11 => Self::SubgroupZeroIdExt,
            0x12 => Self::SubgroupFirstObjectId,
            0x13 => Self::SubgroupFirstObjectIdExt,
            0x14 => Self::SubgroupId,
            0x15 => Self::SubgroupIdExt,
            0x18 => Self::SubgroupZeroIdEndOfGroup,
            0x19 => Self::SubgroupZeroIdExtEndOfGroup,
            0x1a => Self::SubgroupFirstObjectIdEndOfGroup,
            0x1b => Self::SubgroupFirstObjectIdExtEndOfGroup,
            0x1c => Self::SubgroupIdEndOfGroup,
            0x1d => Self::SubgroupIdExtEndOfGroup,
            _ => return Err(MoqtError::Malformed),
        })
    }

    /// Whether an explicit subgroup id follows the group id.
    pub fn has_subgroup_id(self) -> bool {
        matches!(
            self,
            Self::SubgroupId
                | Self::SubgroupIdExt
                | Self::SubgroupIdEndOfGroup
                | Self::SubgroupIdExtEndOfGroup
        )
    }

    /// Whether each object carries an extension-header block.
    pub fn has_extension_headers(self) -> bool {
        matches!(
            self,
            Self::SubgroupZeroIdExt
                | Self::SubgroupFirstObjectIdExt
                | Self::SubgroupIdExt
                | Self::SubgroupZeroIdExtEndOfGroup
                | Self::SubgroupFirstObjectIdExtEndOfGroup
                | Self::SubgroupIdExtEndOfGroup
        )
    }
}

/// Object status values (§10.2.1.1). Only a zero-length object encodes one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectStatus {
    Normal = 0x0,
    EndOfGroup = 0x3,
    EndOfTrack = 0x4,
}

impl ObjectStatus {
    fn from_code(code: u64) -> Result<Self, MoqtError> {
        Ok(match code {
            0x0 => Self::Normal,
            0x3 => Self::EndOfGroup,
            0x4 => Self::EndOfTrack,
            _ => return Err(MoqtError::Malformed),
        })
    }
}

/// The header that opens a subgroup's unidirectional stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgroupHeader {
    pub header_type: StreamHeaderType,
    pub track_alias: u64,
    pub group_id: u64,
    /// Present only for the header types that carry it explicitly.
    pub subgroup_id: Option<u64>,
    /// Smaller values are sent first.
    pub publisher_priority: u8,
}

impl SubgroupHeader {
    pub fn encode(&self, out: &mut Vec<u8>) -> Result<(), MoqtError> {
        put_varint(out, self.header_type.code());
        put_varint(out, self.track_alias);
        put_varint(out, self.group_id);
        if self.header_type.has_subgroup_id() {
            put_varint(out, self.subgroup_id.ok_or(MoqtError::Malformed)?);
        }
        out.push(self.publisher_priority);
        Ok(())
    }

    pub fn decode(r: &mut Reader<'_>) -> Result<Self, MoqtError> {
        let header_type = StreamHeaderType::from_code(r.varint()?)?;
        let track_alias = r.varint()?;
        let group_id = r.varint()?;
        let subgroup_id = if header_type.has_subgroup_id() {
            Some(r.varint()?)
        } else {
            None
        };
        Ok(Self {
            header_type,
            track_alias,
            group_id,
            subgroup_id,
            publisher_priority: r.u8()?,
        })
    }
}

/// One object's header inside a subgroup stream. The payload follows it
/// directly and is not part of this struct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgroupObjectHeader {
    /// Distance to the previous object id, less one. The first object of a
    /// subgroup takes the delta as its absolute id, so zero means "0, then
    /// consecutive" (see `session/subscriber.rs` in the reference).
    pub object_id_delta: u64,
    pub extension_headers: Params,
    pub payload_length: usize,
    /// Only present, and only meaningful, when the payload is empty.
    pub status: Option<ObjectStatus>,
}

impl SubgroupObjectHeader {
    /// A normal object of `payload_length` bytes with no extensions.
    pub fn normal(object_id_delta: u64, payload_length: usize) -> Self {
        Self {
            object_id_delta,
            extension_headers: Params::new(),
            payload_length,
            status: None,
        }
    }

    pub fn encode(
        &self,
        header_type: StreamHeaderType,
        out: &mut Vec<u8>,
    ) -> Result<(), MoqtError> {
        put_varint(out, self.object_id_delta);
        if header_type.has_extension_headers() {
            self.extension_headers.encode_extension_headers(out)?;
        } else if !self.extension_headers.0.is_empty() {
            // The header type says there is no extension block, so writing one
            // would put bytes on the wire the peer will read as an object id.
            return Err(MoqtError::Malformed);
        }
        put_varint(out, self.payload_length as u64);
        if self.payload_length == 0 {
            put_varint(out, self.status.ok_or(MoqtError::Malformed)? as u64);
        }
        Ok(())
    }

    pub fn decode(header_type: StreamHeaderType, r: &mut Reader<'_>) -> Result<Self, MoqtError> {
        let object_id_delta = r.varint()?;
        let extension_headers = if header_type.has_extension_headers() {
            Params::decode_extension_headers(r)?
        } else {
            Params::new()
        };
        let payload_length = r.varint_usize()?;
        let status = if payload_length == 0 {
            let status = ObjectStatus::from_code(r.varint()?)?;
            // A non-normal status with extension headers is a protocol
            // violation (`data/subgroup.rs`).
            if status != ObjectStatus::Normal && !extension_headers.0.is_empty() {
                return Err(MoqtError::Malformed);
            }
            Some(status)
        } else {
            None
        };
        Ok(Self {
            object_id_delta,
            extension_headers,
            payload_length,
            status,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn subgroup_header_matches_the_reference_layout() {
        let header = SubgroupHeader {
            header_type: StreamHeaderType::SubgroupIdExt,
            track_alias: 4,
            group_id: 7,
            subgroup_id: Some(0),
            publisher_priority: 127,
        };
        let mut out = Vec::new();
        header.encode(&mut out).expect("encode");
        assert_eq!(out, vec![0x15, 0x04, 0x07, 0x00, 0x7f]);
        assert_eq!(
            SubgroupHeader::decode(&mut Reader::new(&out)).expect("decode"),
            header
        );
    }

    #[test]
    fn header_types_without_a_subgroup_id_omit_it() {
        let header = SubgroupHeader {
            header_type: StreamHeaderType::SubgroupZeroId,
            track_alias: 1,
            group_id: 2,
            subgroup_id: None,
            publisher_priority: 0,
        };
        let mut out = Vec::new();
        header.encode(&mut out).expect("encode");
        assert_eq!(out, vec![0x10, 0x01, 0x02, 0x00]);
        assert_eq!(
            SubgroupHeader::decode(&mut Reader::new(&out)).expect("decode"),
            header
        );
        assert!(!StreamHeaderType::SubgroupZeroId.has_extension_headers());
        assert!(StreamHeaderType::SubgroupIdExt.has_extension_headers());
        assert_eq!(
            StreamHeaderType::from_code(0x16),
            Err(MoqtError::Malformed),
            "0x16 is not a stream header type"
        );
    }

    #[test]
    fn object_header_carries_an_empty_extension_block() {
        let obj = SubgroupObjectHeader::normal(0, 5);
        let mut out = Vec::new();
        obj.encode(StreamHeaderType::SubgroupIdExt, &mut out)
            .expect("encode");
        // delta 0, extension length 0, payload length 5
        assert_eq!(out, vec![0x00, 0x00, 0x05]);
        assert_eq!(
            SubgroupObjectHeader::decode(StreamHeaderType::SubgroupIdExt, &mut Reader::new(&out))
                .expect("decode"),
            obj
        );

        // Without the Ext header type the extension block is absent entirely.
        let mut plain = Vec::new();
        obj.encode(StreamHeaderType::SubgroupId, &mut plain)
            .expect("encode");
        assert_eq!(plain, vec![0x00, 0x05]);
    }

    #[test]
    fn a_zero_length_object_must_state_a_status() {
        let mut header = SubgroupObjectHeader::normal(0, 0);
        assert_eq!(
            header.encode(StreamHeaderType::SubgroupIdExt, &mut Vec::new()),
            Err(MoqtError::Malformed)
        );
        header.status = Some(ObjectStatus::EndOfGroup);
        let mut out = Vec::new();
        header
            .encode(StreamHeaderType::SubgroupIdExt, &mut out)
            .expect("encode");
        assert_eq!(out, vec![0x00, 0x00, 0x00, 0x03]);
        assert_eq!(
            SubgroupObjectHeader::decode(StreamHeaderType::SubgroupIdExt, &mut Reader::new(&out))
                .expect("decode"),
            header
        );

        // A reserved status value fails the parse.
        assert_eq!(
            SubgroupObjectHeader::decode(
                StreamHeaderType::SubgroupIdExt,
                &mut Reader::new(&[0x00, 0x00, 0x00, 0x01])
            ),
            Err(MoqtError::Malformed)
        );
    }

    #[test]
    fn a_truncated_stream_prefix_is_incomplete() {
        let full = [0x15u8, 0x04, 0x07, 0x00, 0x7f];
        for cut in 0..full.len() {
            assert_eq!(
                SubgroupHeader::decode(&mut Reader::new(&full[..cut])),
                Err(MoqtError::Incomplete),
                "a {cut}-byte prefix needs more bytes"
            );
        }
    }
}
