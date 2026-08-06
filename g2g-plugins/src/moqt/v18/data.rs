//! Unidirectional stream types and the subgroup data plane for IETF MoQ
//! Transport draft-18 (§3.4, §11.4).
//!
//! Draft-16's twelve enumerated stream header types became a bit table: bit 4 is
//! always set, and the remaining bits say which fields the header and its
//! objects carry. Two of them are new, and both save bytes on every object:
//! `DEFAULT_PRIORITY` omits the priority byte entirely, and `SUBGROUP_ID_MODE`
//! can say "the subgroup id is the first object's id" without sending it.
//!
//! Object ordering is not version-specific, so [`super::super::reassembly`]'s
//! `Reassembler` and `ReceivedObject` are reused as they are; only the bytes
//! that produce them are decoded here.

use alloc::vec::Vec;

use super::super::reassembly::{next_object_id, ReceivedObject, DECODER_SLACK};
use super::coding::{
    decode_kvps_length_prefixed, encode_kvps_length_prefixed, put_vi64, reader, MoqtError, Params,
    Reader,
};

// Object status values are unchanged from draft-16 (§11.2.1.1), and a decoded
// object flows into the shared reassembler, so this is the same type.
pub use super::super::data::ObjectStatus;

/// FETCH_HEADER stream type (§11.4.4).
pub const FETCH_HEADER_TYPE: u64 = 0x05;

/// PADDING stream type (§11.5.1). Everything after it is zero bytes to discard.
pub const PADDING_STREAM_TYPE: u64 = 0x132b_3e28;

/// PADDING datagram type (§11.5.2).
pub const PADDING_DATAGRAM_TYPE: u64 = 0x132b_3e29;

/// SUBGROUP_HEADER type bits (§11.4.2).
mod flag {
    pub(super) const PROPERTIES: u64 = 0x01;
    pub(super) const SUBGROUP_ID_MODE: u64 = 0x06;
    pub(super) const END_OF_GROUP: u64 = 0x08;
    /// Bit 4, set on every subgroup header, is what distinguishes the form.
    pub(super) const SUBGROUP: u64 = 0x10;
    pub(super) const DEFAULT_PRIORITY: u64 = 0x20;
    pub(super) const FIRST_OBJECT: u64 = 0x40;
    /// Bits that must be zero for the type to match the form `0b0XX1XXXX`.
    pub(super) const FORBIDDEN: u64 = 0x80;
}

/// How a subgroup stream states its Subgroup ID (§11.4.2). `0b11` is reserved
/// and decodes as a protocol violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubgroupIdMode {
    /// No field: the subgroup id is 0.
    Zero,
    /// No field: the subgroup id is the first object's id.
    FirstObjectId,
    /// The subgroup id follows the group id in the header.
    Explicit,
}

/// The bits of a SUBGROUP_HEADER type byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubgroupHeaderType {
    /// Every object on this stream carries a properties block (length 0 when it
    /// has no properties).
    pub properties: bool,
    pub subgroup_id_mode: SubgroupIdMode,
    /// This subgroup holds the largest object in its group.
    pub end_of_group: bool,
    /// The priority byte is omitted; the subscription's priority applies.
    pub default_priority: bool,
    /// The stream's first object is the first ever published in the subgroup.
    pub first_object: bool,
}

impl SubgroupHeaderType {
    /// The form `moqtsink` publishes: an explicit subgroup id, a properties
    /// block on every object, and an explicit priority, which is the most
    /// information a relay can act on.
    pub fn explicit() -> Self {
        Self {
            properties: true,
            subgroup_id_mode: SubgroupIdMode::Explicit,
            end_of_group: false,
            default_priority: false,
            first_object: false,
        }
    }

    pub fn code(self) -> u64 {
        let mut code = flag::SUBGROUP;
        if self.properties {
            code |= flag::PROPERTIES;
        }
        code |= match self.subgroup_id_mode {
            SubgroupIdMode::Zero => 0,
            SubgroupIdMode::FirstObjectId => 1 << 1,
            SubgroupIdMode::Explicit => 2 << 1,
        };
        if self.end_of_group {
            code |= flag::END_OF_GROUP;
        }
        if self.default_priority {
            code |= flag::DEFAULT_PRIORITY;
        }
        if self.first_object {
            code |= flag::FIRST_OBJECT;
        }
        code
    }

    pub fn from_code(code: u64) -> Result<Self, MoqtError> {
        // The form is 0b0XX1XXXX: bit 4 set, bit 7 clear, nothing above.
        if code > 0x7f || code & flag::SUBGROUP == 0 || code & flag::FORBIDDEN != 0 {
            return Err(MoqtError::Malformed);
        }
        let subgroup_id_mode = match (code & flag::SUBGROUP_ID_MODE) >> 1 {
            0 => SubgroupIdMode::Zero,
            1 => SubgroupIdMode::FirstObjectId,
            2 => SubgroupIdMode::Explicit,
            // 0b11 is reserved for future use.
            _ => return Err(MoqtError::Malformed),
        };
        Ok(Self {
            properties: code & flag::PROPERTIES != 0,
            subgroup_id_mode,
            end_of_group: code & flag::END_OF_GROUP != 0,
            default_priority: code & flag::DEFAULT_PRIORITY != 0,
            first_object: code & flag::FIRST_OBJECT != 0,
        })
    }
}

/// What a unidirectional stream carries, from the varint that opens it (§3.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UniStreamType {
    /// The peer's control stream, opened with a SETUP message.
    Setup,
    /// A FETCH response stream, identified by its request id.
    FetchHeader,
    Subgroup(SubgroupHeaderType),
    /// Zero bytes to read and discard.
    Padding,
}

impl UniStreamType {
    /// An unknown stream type closes the session (§3.4), which is what
    /// [`MoqtError::Malformed`] means to the reader.
    pub fn from_code(code: u64) -> Result<Self, MoqtError> {
        Ok(match code {
            FETCH_HEADER_TYPE => Self::FetchHeader,
            super::message::msg_type::SETUP => Self::Setup,
            PADDING_STREAM_TYPE => Self::Padding,
            _ => Self::Subgroup(SubgroupHeaderType::from_code(code)?),
        })
    }
}

/// The header that opens a subgroup's unidirectional stream (§11.4.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgroupHeader {
    pub header_type: SubgroupHeaderType,
    pub track_alias: u64,
    pub group_id: u64,
    /// Present only when the mode is [`SubgroupIdMode::Explicit`].
    pub subgroup_id: Option<u64>,
    /// Present only when `default_priority` is clear. Smaller is sent first.
    pub publisher_priority: Option<u8>,
}

impl SubgroupHeader {
    pub fn encode(&self, out: &mut Vec<u8>) -> Result<(), MoqtError> {
        let explicit_id = self.header_type.subgroup_id_mode == SubgroupIdMode::Explicit;
        if explicit_id != self.subgroup_id.is_some()
            || self.header_type.default_priority == self.publisher_priority.is_some()
        {
            // The type says which fields exist, so a mismatch would put bytes
            // where the peer reads something else.
            return Err(MoqtError::Malformed);
        }
        put_vi64(out, self.header_type.code());
        put_vi64(out, self.track_alias);
        put_vi64(out, self.group_id);
        if let Some(id) = self.subgroup_id {
            put_vi64(out, id);
        }
        if let Some(priority) = self.publisher_priority {
            out.push(priority);
        }
        Ok(())
    }

    pub fn decode(r: &mut Reader<'_>) -> Result<Self, MoqtError> {
        let header_type = SubgroupHeaderType::from_code(r.varint()?)?;
        let track_alias = r.varint()?;
        let group_id = r.varint()?;
        let subgroup_id = match header_type.subgroup_id_mode {
            SubgroupIdMode::Explicit => Some(r.varint()?),
            _ => None,
        };
        let publisher_priority = if header_type.default_priority {
            None
        } else {
            Some(r.u8()?)
        };
        Ok(Self {
            header_type,
            track_alias,
            group_id,
            subgroup_id,
            publisher_priority,
        })
    }
}

/// One object's fields inside a subgroup stream (§11.4.2, figure 25). The
/// payload follows directly and is not part of this struct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgroupObjectHeader {
    /// Distance to the previous object id on this stream, less one. The first
    /// object of a stream takes the delta as its absolute id.
    pub object_id_delta: u64,
    pub properties: Params,
    pub payload_length: usize,
    /// Only present, and only meaningful, when the payload is empty.
    pub status: Option<ObjectStatus>,
}

impl SubgroupObjectHeader {
    /// A normal object of `payload_length` bytes with no properties.
    pub fn normal(object_id_delta: u64, payload_length: usize) -> Self {
        Self {
            object_id_delta,
            properties: Params::new(),
            payload_length,
            status: None,
        }
    }

    pub fn encode(
        &self,
        header_type: SubgroupHeaderType,
        out: &mut Vec<u8>,
    ) -> Result<(), MoqtError> {
        if !header_type.properties && !self.properties.0.is_empty() {
            // The stream type says no properties block follows, so writing one
            // would land where the peer reads a payload length.
            return Err(MoqtError::Malformed);
        }
        if self.status.is_some_and(|s| s != ObjectStatus::Normal) && !self.properties.0.is_empty() {
            // §11.2.1.2: only a Normal object can carry properties.
            return Err(MoqtError::Malformed);
        }
        put_vi64(out, self.object_id_delta);
        if header_type.properties {
            encode_kvps_length_prefixed(&self.properties, out)?;
        }
        put_vi64(out, self.payload_length as u64);
        if self.payload_length == 0 {
            put_vi64(out, self.status.ok_or(MoqtError::Malformed)? as u64);
        }
        Ok(())
    }

    pub fn decode(header_type: SubgroupHeaderType, r: &mut Reader<'_>) -> Result<Self, MoqtError> {
        let object_id_delta = r.varint()?;
        let properties = if header_type.properties {
            // Unlike a datagram, a subgroup object with the bit set and no
            // properties writes a zero length: that is legal here (§11.4.2).
            decode_kvps_length_prefixed(r)?
        } else {
            Params::new()
        };
        let payload_length = r.varint_usize()?;
        let status = if payload_length == 0 {
            let status = ObjectStatus::from_code(r.varint()?)?;
            if status != ObjectStatus::Normal && !properties.0.is_empty() {
                return Err(MoqtError::Malformed);
            }
            Some(status)
        } else {
            None
        };
        Ok(Self {
            object_id_delta,
            properties,
            payload_length,
            status,
        })
    }
}

/// What a [`SubgroupStreamDecoder`] produced from the bytes it holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamItem {
    /// The stream header, always the first item.
    Header(SubgroupHeader),
    Object(ReceivedObject),
}

/// Decodes one draft-18 subgroup stream incrementally: bytes in, whole objects
/// out. Objects are delivered whole because a MOQT object here is one CMAF
/// chunk and the demuxer downstream wants whole boxes anyway.
#[derive(Debug)]
pub struct SubgroupStreamDecoder {
    header: Option<SubgroupHeader>,
    prev_object_id: Option<u64>,
    buf: Vec<u8>,
    max_object_bytes: usize,
}

impl SubgroupStreamDecoder {
    pub fn new(max_object_bytes: usize) -> Self {
        Self {
            header: None,
            prev_object_id: None,
            buf: Vec::new(),
            max_object_bytes,
        }
    }

    pub fn header(&self) -> Option<&SubgroupHeader> {
        self.header.as_ref()
    }

    /// The subgroup id this stream carries, once its first object has resolved
    /// the [`SubgroupIdMode::FirstObjectId`] case.
    pub fn subgroup_id(&self) -> Option<u64> {
        let header = self.header.as_ref()?;
        match header.header_type.subgroup_id_mode {
            SubgroupIdMode::Zero => Some(0),
            SubgroupIdMode::Explicit => header.subgroup_id,
            SubgroupIdMode::FirstObjectId => self.prev_object_id,
        }
    }

    /// Append bytes read off the stream.
    pub fn push(&mut self, bytes: &[u8]) -> Result<(), MoqtError> {
        // A peer that opens an object it never finishes must not grow this
        // without limit; the per-object length is checked in `next_item`, this
        // bounds the header and properties block that precede it.
        if self.buf.len().saturating_add(bytes.len())
            > self.max_object_bytes.saturating_add(DECODER_SLACK)
        {
            return Err(MoqtError::Malformed);
        }
        self.buf.extend_from_slice(bytes);
        Ok(())
    }

    /// The next complete item, or `None` when more bytes are needed. Call until
    /// it returns `None` after every [`push`](Self::push).
    pub fn next_item(&mut self) -> Result<Option<StreamItem>, MoqtError> {
        let Some(header) = self.header.clone() else {
            let mut r = reader(&self.buf);
            return match SubgroupHeader::decode(&mut r) {
                Ok(header) => {
                    self.buf.drain(..r.position());
                    self.header = Some(header.clone());
                    Ok(Some(StreamItem::Header(header)))
                }
                Err(MoqtError::Incomplete) => Ok(None),
                Err(e) => Err(e),
            };
        };

        let mut r = reader(&self.buf);
        let object = match SubgroupObjectHeader::decode(header.header_type, &mut r) {
            Ok(object) => object,
            Err(MoqtError::Incomplete) => return Ok(None),
            Err(e) => return Err(e),
        };
        if object.payload_length > self.max_object_bytes {
            return Err(MoqtError::Malformed);
        }
        let payload = match r.bytes(object.payload_length) {
            Ok(payload) => payload.to_vec(),
            Err(MoqtError::Incomplete) => return Ok(None),
            Err(e) => return Err(e),
        };
        let object_id = next_object_id(self.prev_object_id, object.object_id_delta)?;
        self.prev_object_id = Some(object_id);
        self.buf.drain(..r.position());
        Ok(Some(StreamItem::Object(ReceivedObject {
            group_id: header.group_id,
            object_id,
            status: object.status.unwrap_or(ObjectStatus::Normal),
            payload,
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn explicit_header(group_id: u64) -> SubgroupHeader {
        SubgroupHeader {
            header_type: SubgroupHeaderType::explicit(),
            track_alias: 4,
            group_id,
            subgroup_id: Some(0),
            publisher_priority: Some(127),
        }
    }

    /// The bit table of §11.4.2, checked against the type ranges the draft
    /// spells out.
    #[test]
    fn subgroup_header_types_map_to_their_bits() {
        assert_eq!(SubgroupHeaderType::explicit().code(), 0x15);
        assert_eq!(
            SubgroupHeaderType::from_code(0x15),
            Ok(SubgroupHeaderType::explicit())
        );

        // Every value the draft calls valid decodes, and re-encodes to itself.
        let valid = (0x10..=0x1fu64)
            .chain(0x30..=0x3f)
            .chain(0x50..=0x5f)
            .chain(0x70..=0x7f);
        for code in valid {
            let reserved_mode = (code & super::flag::SUBGROUP_ID_MODE) >> 1 == 3;
            match SubgroupHeaderType::from_code(code) {
                Ok(ty) => {
                    assert!(!reserved_mode, "{code:#x} uses the reserved mode");
                    assert_eq!(ty.code(), code, "{code:#x} re-encodes to itself");
                }
                Err(e) => {
                    assert!(reserved_mode, "{code:#x} should decode, got {e:?}");
                    assert_eq!(e, MoqtError::Malformed);
                }
            }
        }

        // The reserved subgroup id mode, and every type outside the form.
        for code in [0x16u64, 0x1e, 0x36, 0x5f, 0x76, 0x7f] {
            assert_eq!(
                SubgroupHeaderType::from_code(code),
                Err(MoqtError::Malformed),
                "{code:#x} sets the reserved subgroup id mode"
            );
        }
        for code in [0x00u64, 0x05, 0x0f, 0x20, 0x2f, 0x80, 0x90, 0xff, 0x1_0000] {
            assert_eq!(
                SubgroupHeaderType::from_code(code),
                Err(MoqtError::Malformed),
                "{code:#x} is outside the subgroup form"
            );
        }
    }

    #[test]
    fn a_subgroup_header_omits_the_fields_its_type_says_are_absent() {
        let mut out = Vec::new();
        explicit_header(7).encode(&mut out).expect("encode");
        // type 0x15, alias 4, group 7, subgroup 0, priority 127
        assert_eq!(out, vec![0x15, 0x04, 0x07, 0x00, 0x7f]);
        assert_eq!(
            SubgroupHeader::decode(&mut reader(&out)).expect("decode"),
            explicit_header(7)
        );

        // Default priority and an implied subgroup id: the two new bits, and the
        // shortest header draft-18 allows.
        let lean = SubgroupHeader {
            header_type: SubgroupHeaderType {
                properties: false,
                subgroup_id_mode: SubgroupIdMode::Zero,
                end_of_group: true,
                default_priority: true,
                first_object: true,
            },
            track_alias: 1,
            group_id: 2,
            subgroup_id: None,
            publisher_priority: None,
        };
        let mut out = Vec::new();
        lean.encode(&mut out).expect("encode");
        assert_eq!(out, vec![0x78, 0x01, 0x02]);
        assert_eq!(SubgroupHeader::decode(&mut reader(&out)), Ok(lean));

        // A header whose fields disagree with its type is refused rather than
        // written where the peer reads something else.
        let mismatched = SubgroupHeader {
            subgroup_id: None,
            ..explicit_header(1)
        };
        assert_eq!(
            mismatched.encode(&mut Vec::new()),
            Err(MoqtError::Malformed)
        );
        let mismatched = SubgroupHeader {
            publisher_priority: None,
            ..explicit_header(1)
        };
        assert_eq!(
            mismatched.encode(&mut Vec::new()),
            Err(MoqtError::Malformed)
        );
    }

    #[test]
    fn an_object_carries_a_properties_block_only_when_the_type_says_so() {
        let obj = SubgroupObjectHeader::normal(0, 5);
        let mut out = Vec::new();
        obj.encode(SubgroupHeaderType::explicit(), &mut out)
            .expect("encode");
        // delta 0, properties length 0, payload length 5
        assert_eq!(out, vec![0x00, 0x00, 0x05]);
        assert_eq!(
            SubgroupObjectHeader::decode(SubgroupHeaderType::explicit(), &mut reader(&out)),
            Ok(obj.clone())
        );

        let bare = SubgroupHeaderType {
            properties: false,
            ..SubgroupHeaderType::explicit()
        };
        let mut out = Vec::new();
        obj.encode(bare, &mut out).expect("encode");
        assert_eq!(out, vec![0x00, 0x05], "no properties block at all");

        // Properties the type cannot carry are refused.
        let mut props = Params::new();
        props.set_int(super::super::coding::property::PRIOR_OBJECT_ID_GAP, 2);
        let with_props = SubgroupObjectHeader {
            properties: props.clone(),
            ..SubgroupObjectHeader::normal(0, 5)
        };
        assert_eq!(
            with_props.encode(bare, &mut Vec::new()),
            Err(MoqtError::Malformed)
        );
        let mut out = Vec::new();
        with_props
            .encode(SubgroupHeaderType::explicit(), &mut out)
            .expect("encode");
        assert_eq!(
            SubgroupObjectHeader::decode(SubgroupHeaderType::explicit(), &mut reader(&out)),
            Ok(with_props)
        );
    }

    #[test]
    fn a_zero_length_object_states_a_status_and_a_marker_carries_no_properties() {
        let mut header = SubgroupObjectHeader::normal(0, 0);
        assert_eq!(
            header.encode(SubgroupHeaderType::explicit(), &mut Vec::new()),
            Err(MoqtError::Malformed),
            "a zero-length object must say what it means"
        );
        header.status = Some(ObjectStatus::EndOfGroup);
        let mut out = Vec::new();
        header
            .encode(SubgroupHeaderType::explicit(), &mut out)
            .expect("encode");
        assert_eq!(out, vec![0x00, 0x00, 0x00, 0x03]);
        assert_eq!(
            SubgroupObjectHeader::decode(SubgroupHeaderType::explicit(), &mut reader(&out)),
            Ok(header.clone())
        );

        // A reserved status value fails the parse.
        assert_eq!(
            SubgroupObjectHeader::decode(
                SubgroupHeaderType::explicit(),
                &mut reader(&[0x00, 0x00, 0x00, 0x01])
            ),
            Err(MoqtError::Malformed)
        );
        // A non-normal status with properties is a violation both ways.
        let mut props = Params::new();
        props.set_int(super::super::coding::property::PRIOR_OBJECT_ID_GAP, 2);
        let marker = SubgroupObjectHeader {
            properties: props,
            ..header
        };
        assert_eq!(
            marker.encode(SubgroupHeaderType::explicit(), &mut Vec::new()),
            Err(MoqtError::Malformed)
        );
        assert_eq!(
            SubgroupObjectHeader::decode(
                SubgroupHeaderType::explicit(),
                &mut reader(&[0x00, 0x02, 0x3e, 0x02, 0x00, 0x03])
            ),
            Err(MoqtError::Malformed)
        );
    }

    #[test]
    fn uni_stream_types_dispatch_and_refuse_the_unknown() {
        assert_eq!(
            UniStreamType::from_code(0x05),
            Ok(UniStreamType::FetchHeader)
        );
        assert_eq!(UniStreamType::from_code(0x2f00), Ok(UniStreamType::Setup));
        assert_eq!(
            UniStreamType::from_code(0x132b_3e28),
            Ok(UniStreamType::Padding)
        );
        assert_eq!(
            UniStreamType::from_code(0x15),
            Ok(UniStreamType::Subgroup(SubgroupHeaderType::explicit()))
        );
        for code in [0x00u64, 0x04, 0x06, 0x2f01, 0x132b_3e29] {
            assert_eq!(
                UniStreamType::from_code(code),
                Err(MoqtError::Malformed),
                "{code:#x} is not a draft-18 stream type"
            );
        }
    }

    fn encoded_stream(header: &SubgroupHeader, objects: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        header.encode(&mut out).expect("header");
        for payload in objects {
            SubgroupObjectHeader::normal(0, payload.len())
                .encode(header.header_type, &mut out)
                .expect("object header");
            out.extend_from_slice(payload);
        }
        out
    }

    #[test]
    fn a_stream_decodes_to_consecutive_object_ids_one_byte_at_a_time() {
        let bytes = encoded_stream(&explicit_header(7), &[b"aaa", b"bb", b"c"]);
        let mut decoder = SubgroupStreamDecoder::new(1024);
        let mut items = Vec::new();
        for byte in &bytes {
            decoder.push(&[*byte]).expect("push");
            while let Some(item) = decoder.next_item().expect("decode") {
                items.push(item);
            }
        }
        assert_eq!(items.len(), 4, "the header plus three objects");
        assert_eq!(items[0], StreamItem::Header(explicit_header(7)));
        for (i, expected) in [b"aaa".as_slice(), b"bb", b"c"].iter().enumerate() {
            assert_eq!(
                items[i + 1],
                StreamItem::Object(ReceivedObject {
                    group_id: 7,
                    object_id: i as u64,
                    status: ObjectStatus::Normal,
                    payload: expected.to_vec(),
                })
            );
        }
        assert_eq!(decoder.subgroup_id(), Some(0));
    }

    /// The subgroup id mode that takes the first object's id resolves only once
    /// that object has arrived.
    #[test]
    fn the_first_object_id_mode_names_the_subgroup_after_its_first_object() {
        let header = SubgroupHeader {
            header_type: SubgroupHeaderType {
                properties: false,
                subgroup_id_mode: SubgroupIdMode::FirstObjectId,
                end_of_group: false,
                default_priority: false,
                first_object: true,
            },
            track_alias: 1,
            group_id: 2,
            subgroup_id: None,
            publisher_priority: Some(10),
        };
        let mut bytes = Vec::new();
        header.encode(&mut bytes).expect("header");
        // The first object's delta is its absolute id, so this subgroup is 5.
        SubgroupObjectHeader::normal(5, 1)
            .encode(header.header_type, &mut bytes)
            .expect("object");
        bytes.push(b'x');

        let mut decoder = SubgroupStreamDecoder::new(1024);
        decoder.push(&bytes).expect("push");
        assert_eq!(
            decoder.next_item(),
            Ok(Some(StreamItem::Header(header.clone())))
        );
        assert_eq!(decoder.subgroup_id(), None, "no object has arrived yet");
        assert_eq!(
            decoder.next_item(),
            Ok(Some(StreamItem::Object(ReceivedObject {
                group_id: 2,
                object_id: 5,
                status: ObjectStatus::Normal,
                payload: b"x".to_vec(),
            })))
        );
        assert_eq!(decoder.subgroup_id(), Some(5));
    }

    #[test]
    fn malformed_stream_input_fails_the_decode() {
        // A stream header type that is not a subgroup.
        let mut decoder = SubgroupStreamDecoder::new(1024);
        decoder.push(&[0x16, 0x04, 0x07, 0x7f]).expect("push");
        assert_eq!(decoder.next_item(), Err(MoqtError::Malformed));

        // A truncated object needs more bytes; it is not an error.
        let mut decoder = SubgroupStreamDecoder::new(1024);
        let bytes = encoded_stream(&explicit_header(1), &[b"payload"]);
        decoder.push(&bytes[..bytes.len() - 3]).expect("push");
        assert_eq!(
            decoder.next_item(),
            Ok(Some(StreamItem::Header(explicit_header(1))))
        );
        assert_eq!(decoder.next_item(), Ok(None));

        // A properties block one byte past the 64 KiB the codec allows.
        let mut decoder = SubgroupStreamDecoder::new(1 << 20);
        let mut bytes = Vec::new();
        explicit_header(1).encode(&mut bytes).expect("header");
        bytes.push(0x00); // delta 0
        put_vi64(&mut bytes, u16::MAX as u64 + 1);
        decoder.push(&bytes).expect("push");
        assert_eq!(
            decoder.next_item(),
            Ok(Some(StreamItem::Header(explicit_header(1))))
        );
        assert_eq!(decoder.next_item(), Err(MoqtError::Malformed));

        // A payload length past the per-object bound is refused without
        // allocating on it.
        let mut decoder = SubgroupStreamDecoder::new(16);
        let mut bytes = Vec::new();
        explicit_header(1).encode(&mut bytes).expect("header");
        bytes.extend_from_slice(&[0x00, 0x00]);
        put_vi64(&mut bytes, 1 << 40);
        decoder.push(&bytes).expect("push");
        assert_eq!(
            decoder.next_item(),
            Ok(Some(StreamItem::Header(explicit_header(1))))
        );
        assert_eq!(decoder.next_item(), Err(MoqtError::Malformed));

        // A stream that keeps sending without ever completing an object is
        // refused before its buffer grows past the bound.
        let mut decoder = SubgroupStreamDecoder::new(16);
        let mut bytes = Vec::new();
        explicit_header(1).encode(&mut bytes).expect("header");
        decoder.push(&bytes).expect("push");
        assert_eq!(
            decoder.next_item(),
            Ok(Some(StreamItem::Header(explicit_header(1))))
        );
        assert_eq!(
            decoder.push(&vec![0xffu8; 200 * 1024]),
            Err(MoqtError::Malformed)
        );

        // An object id that would leave u64.
        let mut decoder = SubgroupStreamDecoder::new(1024);
        let mut bytes = Vec::new();
        explicit_header(1).encode(&mut bytes).expect("header");
        for _ in 0..2 {
            put_vi64(&mut bytes, u64::MAX);
            bytes.extend_from_slice(&[0x00, 0x01, b'x']);
        }
        decoder.push(&bytes).expect("push");
        assert_eq!(
            decoder.next_item(),
            Ok(Some(StreamItem::Header(explicit_header(1))))
        );
        assert!(
            matches!(decoder.next_item(), Ok(Some(_))),
            "the first object"
        );
        assert_eq!(decoder.next_item(), Err(MoqtError::Malformed));
    }
}
