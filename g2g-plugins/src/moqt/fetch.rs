//! The FETCH response data stream: the header that opens it and the object
//! serialization that follows (draft-16 §10.4.4, draft-18 §11.4.4).
//!
//! A fetch response is one unidirectional stream carrying already-published
//! objects in the requested group order, so unlike a subgroup stream it names
//! the request rather than the track, and each object states its own group.
//!
//! One module serves both drafts, because only two things differ: the integer
//! flavour ([`MoqtVersion`] picks it), and whether the Group ID / Object ID
//! fields are absolute (draft-16) or deltas from the previous object
//! (draft-18). The Serialization Flags byte, its field order, and the
//! end-of-range values are identical.
//!
//! Everything here decodes bytes a publisher sent, so every id is folded with
//! checked arithmetic, the payload length is bounded before it is used, and the
//! decoder's buffer cannot grow on a peer that never finishes an object.

use alloc::vec::Vec;

use super::coding::{MoqtError, Params, Reader};
use super::data::ObjectStatus;
use super::reassembly::{ReceivedObject, DECODER_SLACK};
use super::v18::coding::decode_kvps_length_prefixed;
use super::MoqtVersion;

/// The unidirectional stream type that opens a FETCH response.
pub const FETCH_HEADER_TYPE: u64 = 0x05;

/// Serialization Flags bits (draft-16 Tables 5 and 6, draft-18 Tables 8 and 9).
mod flag {
    /// The two low bits say how the Subgroup ID is encoded.
    pub(super) const SUBGROUP_MODE: u64 = 0x03;
    pub(super) const OBJECT_ID: u64 = 0x04;
    pub(super) const GROUP_ID: u64 = 0x08;
    pub(super) const PRIORITY: u64 = 0x10;
    pub(super) const PROPERTIES: u64 = 0x20;
    pub(super) const DATAGRAM: u64 = 0x40;
    /// Largest value whose bits are flags at all; above it only the two
    /// end-of-range values are defined.
    pub(super) const MAX: u64 = 0x7f;
}

/// Serialization Flags value marking a run of objects that do not exist.
pub const END_OF_NON_EXISTENT_RANGE: u64 = 0x8c;
/// Serialization Flags value marking a run of objects whose status is unknown.
pub const END_OF_UNKNOWN_RANGE: u64 = 0x10c;

/// The previous object on a fetch stream: what the delta and "same as the prior
/// object" flags are measured against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Prior {
    group_id: u64,
    object_id: u64,
    subgroup_id: u64,
    priority: u8,
}

/// Serializes the objects of one FETCH response, in the order they are written.
///
/// Every object goes in subgroup 0: a fetch response is a single stream, and
/// the draft orders its objects by (group, object) with the subgroup id taking
/// no part.
#[derive(Debug)]
pub struct FetchWriter {
    version: MoqtVersion,
    prior: Option<Prior>,
}

impl FetchWriter {
    pub fn new(version: MoqtVersion) -> Self {
        Self {
            version,
            prior: None,
        }
    }

    /// The stream header: the type, then the request id these objects answer.
    pub fn header(&self, request_id: u64, out: &mut Vec<u8>) {
        self.version.put_int(out, FETCH_HEADER_TYPE);
        self.version.put_int(out, request_id);
    }

    /// Append one object. Objects must be written in ascending (group, object)
    /// order; anything else cannot be coded against the previous one and is
    /// refused rather than written as a value the subscriber reads differently.
    pub fn object(
        &mut self,
        group_id: u64,
        object_id: u64,
        priority: u8,
        payload: &[u8],
        out: &mut Vec<u8>,
    ) -> Result<(), MoqtError> {
        let mut flags = 0u64; // subgroup mode 0b00: the subgroup id is zero
        let mut fields = Vec::new();
        match self.prior {
            // The first object states both ids absolutely in either draft.
            None => {
                flags |= flag::GROUP_ID | flag::OBJECT_ID | flag::PRIORITY;
                self.version.put_int(&mut fields, group_id);
                self.version.put_int(&mut fields, object_id);
                fields.push(priority);
            }
            Some(prior) if prior.group_id != group_id => {
                flags |= flag::GROUP_ID | flag::OBJECT_ID;
                let group_field = match self.version {
                    MoqtVersion::V16 => group_id,
                    MoqtVersion::V18 => group_id
                        .checked_sub(prior.group_id)
                        .and_then(|d| d.checked_sub(1))
                        .ok_or(MoqtError::Malformed)?,
                };
                self.version.put_int(&mut fields, group_field);
                // With the group field present the object id is absolute.
                self.version.put_int(&mut fields, object_id);
                if priority != prior.priority {
                    flags |= flag::PRIORITY;
                    fields.push(priority);
                }
            }
            Some(prior) => {
                if object_id <= prior.object_id {
                    return Err(MoqtError::Malformed);
                }
                if object_id != prior.object_id.saturating_add(1) {
                    flags |= flag::OBJECT_ID;
                    let object_field = match self.version {
                        MoqtVersion::V16 => object_id,
                        MoqtVersion::V18 => object_id
                            .checked_sub(prior.object_id)
                            .ok_or(MoqtError::Malformed)?,
                    };
                    self.version.put_int(&mut fields, object_field);
                }
                if priority != prior.priority {
                    flags |= flag::PRIORITY;
                    fields.push(priority);
                }
            }
        }
        self.version.put_int(out, flags);
        out.extend_from_slice(&fields);
        self.version.put_int(out, payload.len() as u64);
        out.extend_from_slice(payload);
        self.prior = Some(Prior {
            group_id,
            object_id,
            subgroup_id: 0,
            priority,
        });
        Ok(())
    }
}

/// What a [`FetchStreamDecoder`] produced from the bytes it holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FetchItem {
    /// The stream header, always the first item.
    Header {
        request_id: u64,
    },
    Object(ReceivedObject),
    /// A run of objects the publisher states does not exist, or whose status it
    /// does not know: no media, but it moves the location the next object is
    /// coded against.
    Gap {
        group_id: u64,
        object_id: u64,
    },
}

/// Decodes one FETCH response stream incrementally: bytes in, whole objects
/// out, in the order the publisher wrote them.
#[derive(Debug)]
pub struct FetchStreamDecoder {
    version: MoqtVersion,
    request_id: Option<u64>,
    prior: Option<Prior>,
    buf: Vec<u8>,
    max_object_bytes: usize,
}

impl FetchStreamDecoder {
    pub fn new(version: MoqtVersion, max_object_bytes: usize) -> Self {
        Self {
            version,
            request_id: None,
            prior: None,
            buf: Vec::new(),
            max_object_bytes,
        }
    }

    /// The request id the header named, once it has arrived.
    pub fn request_id(&self) -> Option<u64> {
        self.request_id
    }

    /// Append bytes read off the stream.
    pub fn push(&mut self, bytes: &[u8]) -> Result<(), MoqtError> {
        // A publisher that opens an object it never finishes must not grow this
        // without limit; the per-object length is checked in `next_item`, this
        // bounds the fields and properties block that precede it.
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
    pub fn next_item(&mut self) -> Result<Option<FetchItem>, MoqtError> {
        if self.request_id.is_none() {
            let mut r = self.version.reader(&self.buf);
            return match read_header(&mut r) {
                Ok(request_id) => {
                    self.buf.drain(..r.position());
                    self.request_id = Some(request_id);
                    Ok(Some(FetchItem::Header { request_id }))
                }
                Err(MoqtError::Incomplete) => Ok(None),
                Err(e) => Err(e),
            };
        }
        let mut r = self.version.reader(&self.buf);
        let decoded = read_object(self.version, self.prior, self.max_object_bytes, &mut r);
        let used = r.position();
        let (item, prior) = match decoded {
            Ok(decoded) => decoded,
            Err(MoqtError::Incomplete) => return Ok(None),
            Err(e) => return Err(e),
        };
        self.buf.drain(..used);
        self.prior = Some(prior);
        Ok(Some(item))
    }
}

/// One object off a fetch stream, and the prior-object state the next one is
/// coded against. Free of the decoder so the reader can borrow its buffer.
fn read_object(
    version: MoqtVersion,
    prior: Option<Prior>,
    max_object_bytes: usize,
    r: &mut Reader<'_>,
) -> Result<(FetchItem, Prior), MoqtError> {
    let flags = r.varint()?;
    if flags == END_OF_NON_EXISTENT_RANGE || flags == END_OF_UNKNOWN_RANGE {
        let (group_id, object_id) = read_ids(version, prior, r, true, true)?;
        // The figure keeps the payload length outside the optional fields, and
        // an end-of-range marker carries no media.
        if r.varint()? != 0 {
            return Err(MoqtError::Malformed);
        }
        // §11.4.4.2: an object after the marker that codes against the prior
        // subgroup or priority means the last real object's, so those are kept.
        let moved = Prior {
            group_id,
            object_id,
            ..prior.unwrap_or(Prior {
                group_id,
                object_id,
                subgroup_id: 0,
                priority: 0,
            })
        };
        return Ok((
            FetchItem::Gap {
                group_id,
                object_id,
            },
            moved,
        ));
    }
    if flags > flag::MAX {
        return Err(MoqtError::Malformed);
    }
    let (group_id, object_id) = read_ids(
        version,
        prior,
        r,
        flags & flag::GROUP_ID != 0,
        flags & flag::OBJECT_ID != 0,
    )?;
    let subgroup_id = read_subgroup(prior, r, flags)?;
    let priority = if flags & flag::PRIORITY != 0 {
        r.u8()?
    } else {
        prior.ok_or(MoqtError::Malformed)?.priority
    };
    if flags & flag::PROPERTIES != 0 {
        // Bounded by the codec, and nothing downstream reads an object property
        // yet: decoding it is what keeps the stream framed.
        read_properties(version, r)?;
    }
    let payload_length = r.varint_usize()?;
    if payload_length > max_object_bytes {
        return Err(MoqtError::Malformed);
    }
    let payload = r.bytes(payload_length)?.to_vec();
    Ok((
        FetchItem::Object(ReceivedObject {
            group_id,
            object_id,
            // A fetch object carries no status field in either draft: the range
            // the response covers is what FETCH_OK states.
            status: ObjectStatus::Normal,
            payload,
        }),
        Prior {
            group_id,
            object_id,
            subgroup_id,
            priority,
        },
    ))
}

/// Group and object id, present as fields or taken from the previous object.
/// The fields come in that order, before the subgroup id and the priority.
fn read_ids(
    version: MoqtVersion,
    prior: Option<Prior>,
    r: &mut Reader<'_>,
    group_field: bool,
    object_field: bool,
) -> Result<(u64, u64), MoqtError> {
    let group_id = if group_field {
        let field = r.varint()?;
        match (version, prior) {
            // The first object's field is absolute in either draft.
            (_, None) | (MoqtVersion::V16, _) => field,
            (MoqtVersion::V18, Some(prior)) => prior
                .group_id
                .checked_add(field)
                .and_then(|g| g.checked_add(1))
                .ok_or(MoqtError::Malformed)?,
        }
    } else {
        prior.ok_or(MoqtError::Malformed)?.group_id
    };
    let object_id = match (object_field, prior) {
        (true, prior) => {
            let field = r.varint()?;
            match (version, prior, group_field) {
                // With the group field present the object id is absolute.
                (_, None, _) | (_, _, true) | (MoqtVersion::V16, ..) => field,
                (MoqtVersion::V18, Some(prior), false) => prior
                    .object_id
                    .checked_add(field)
                    .ok_or(MoqtError::Malformed)?,
            }
        }
        (false, Some(prior)) => prior.object_id.checked_add(1).ok_or(MoqtError::Malformed)?,
        // The first object must state both ids (§11.4.4.1).
        (false, None) => return Err(MoqtError::Malformed),
    };
    Ok((group_id, object_id))
}

fn read_subgroup(prior: Option<Prior>, r: &mut Reader<'_>, flags: u64) -> Result<u64, MoqtError> {
    // A datagram-preference object has no subgroup at all, and the two low bits
    // are to be ignored rather than read.
    if flags & flag::DATAGRAM != 0 {
        return Ok(0);
    }
    match flags & flag::SUBGROUP_MODE {
        0x0 => Ok(0),
        0x1 => Ok(prior.ok_or(MoqtError::Malformed)?.subgroup_id),
        0x2 => prior
            .ok_or(MoqtError::Malformed)?
            .subgroup_id
            .checked_add(1)
            .ok_or(MoqtError::Malformed),
        _ => r.varint(),
    }
}

fn read_properties(version: MoqtVersion, r: &mut Reader<'_>) -> Result<Params, MoqtError> {
    match version {
        MoqtVersion::V16 => Params::decode_extension_headers(r),
        MoqtVersion::V18 => decode_kvps_length_prefixed(r),
    }
}

fn read_header(r: &mut Reader<'_>) -> Result<u64, MoqtError> {
    if r.varint()? != FETCH_HEADER_TYPE {
        return Err(MoqtError::Malformed);
    }
    r.varint()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn encoded(version: MoqtVersion, objects: &[(u64, u64, &[u8])]) -> Vec<u8> {
        let mut writer = FetchWriter::new(version);
        let mut out = Vec::new();
        writer.header(7, &mut out);
        for (group, object, payload) in objects {
            writer
                .object(*group, *object, 127, payload, &mut out)
                .expect("encode object");
        }
        out
    }

    fn decoded(version: MoqtVersion, bytes: &[u8]) -> Vec<FetchItem> {
        let mut decoder = FetchStreamDecoder::new(version, 1 << 20);
        let mut items = Vec::new();
        // One byte at a time, so a partial field is `None` rather than an error.
        for byte in bytes {
            decoder.push(&[*byte]).expect("push");
            while let Some(item) = decoder.next_item().expect("decode") {
                items.push(item);
            }
        }
        items
    }

    fn object(group_id: u64, object_id: u64, payload: &[u8]) -> FetchItem {
        FetchItem::Object(ReceivedObject {
            group_id,
            object_id,
            status: ObjectStatus::Normal,
            payload: payload.to_vec(),
        })
    }

    #[test]
    fn a_fetch_stream_round_trips_on_both_drafts() {
        for version in [MoqtVersion::V16, MoqtVersion::V18] {
            let objects: &[(u64, u64, &[u8])] = &[
                (4, 0, b"a"),
                (4, 1, b"bb"),
                // A hole inside the group, then the next group.
                (4, 5, b"ccc"),
                (5, 0, b"d"),
                (7, 2, b"ee"),
            ];
            let bytes = encoded(version, objects);
            let items = decoded(version, &bytes);
            let mut want = vec![FetchItem::Header { request_id: 7 }];
            want.extend(objects.iter().map(|(g, o, p)| object(*g, *o, p)));
            assert_eq!(items, want, "{version:?}");
        }
    }

    /// The bytes the two drafts differ on: draft-16 writes absolute ids where
    /// draft-18 writes deltas, so a round trip alone would not catch one
    /// dialect's rule applied to the other's stream.
    #[test]
    fn the_wire_bytes_follow_each_drafts_id_rule() {
        // Group 4 object 0, then group 6 object 0, both priority 127.
        let objects: &[(u64, u64, &[u8])] = &[(4, 0, b"a"), (6, 0, b"b")];
        assert_eq!(
            encoded(MoqtVersion::V16, objects),
            vec![
                0x05, 0x07, // FETCH_HEADER, request 7
                0x1c, 0x04, 0x00, 0x7f, 0x01, b'a', // flags 0x1c: group 4, object 0
                0x0c, 0x06, 0x00, 0x01, b'b', // flags 0x0c: group 6 absolute
            ]
        );
        assert_eq!(
            encoded(MoqtVersion::V18, objects),
            vec![
                0x05, 0x07, //
                0x1c, 0x04, 0x00, 0x7f, 0x01, b'a', //
                0x0c, 0x01, 0x00, 0x01, b'b', // draft-18: group delta 1 = 6 - 4 - 1
            ]
        );
    }

    /// Consecutive object ids in one group cost one flags byte and nothing else.
    #[test]
    fn a_consecutive_object_omits_every_field() {
        let bytes = encoded(MoqtVersion::V18, &[(0, 0, b"a"), (0, 1, b"b")]);
        assert_eq!(
            bytes,
            vec![
                0x05, 0x07, //
                0x1c, 0x00, 0x00, 0x7f, 0x01, b'a', //
                0x00, 0x01, b'b', // flags 0: prior group, prior object + 1
            ]
        );
    }

    #[test]
    fn objects_out_of_order_are_refused_rather_than_written() {
        let mut writer = FetchWriter::new(MoqtVersion::V18);
        let mut out = Vec::new();
        writer.object(4, 1, 0, b"a", &mut out).expect("first");
        assert_eq!(
            writer.object(4, 1, 0, b"again", &mut out),
            Err(MoqtError::Malformed)
        );
        assert_eq!(
            writer.object(3, 0, 0, b"backwards", &mut out),
            Err(MoqtError::Malformed),
            "draft-18 codes the group as an ascending delta"
        );
    }

    #[test]
    fn malformed_fetch_streams_fail_the_decode() {
        for version in [MoqtVersion::V16, MoqtVersion::V18] {
            // A stream that does not open with FETCH_HEADER.
            let mut decoder = FetchStreamDecoder::new(version, 1 << 20);
            decoder.push(&[0x15, 0x04]).expect("push");
            assert_eq!(decoder.next_item(), Err(MoqtError::Malformed));

            // The first object references the prior object that does not exist.
            let mut decoder = FetchStreamDecoder::new(version, 1 << 20);
            decoder.push(&[0x05, 0x00, 0x00, 0x00]).expect("push");
            assert_eq!(
                decoder.next_item(),
                Ok(Some(FetchItem::Header { request_id: 0 }))
            );
            assert_eq!(decoder.next_item(), Err(MoqtError::Malformed));

            // A flags value that is neither a bit set nor an end-of-range value.
            let mut decoder = FetchStreamDecoder::new(version, 1 << 20);
            decoder.push(&[0x05, 0x00, 0x40, 0x81]).expect("push");
            assert_eq!(
                decoder.next_item(),
                Ok(Some(FetchItem::Header { request_id: 0 }))
            );
            assert_eq!(decoder.next_item(), Err(MoqtError::Malformed));

            // A payload length past the per-object bound is refused without
            // allocating on it.
            let mut decoder = FetchStreamDecoder::new(version, 16);
            let mut bytes = Vec::new();
            version.put_int(&mut bytes, FETCH_HEADER_TYPE);
            version.put_int(&mut bytes, 0);
            version.put_int(&mut bytes, 0x1c);
            version.put_int(&mut bytes, 0);
            version.put_int(&mut bytes, 0);
            bytes.push(0);
            version.put_int(&mut bytes, 1 << 30);
            decoder.push(&bytes).expect("push");
            assert_eq!(
                decoder.next_item(),
                Ok(Some(FetchItem::Header { request_id: 0 }))
            );
            assert_eq!(decoder.next_item(), Err(MoqtError::Malformed));

            // A publisher that never completes an object stops growing the
            // decoder's buffer.
            let mut decoder = FetchStreamDecoder::new(version, 16);
            assert_eq!(
                decoder.push(&vec![0xffu8; 200 * 1024]),
                Err(MoqtError::Malformed)
            );
        }
    }

    /// An id that would leave `u64` is a protocol violation on either dialect.
    #[test]
    fn ids_past_u64_fail_the_decode() {
        let mut bytes = Vec::new();
        MoqtVersion::V18.put_int(&mut bytes, FETCH_HEADER_TYPE);
        MoqtVersion::V18.put_int(&mut bytes, 0);
        // First object: group u64::MAX, object 0.
        MoqtVersion::V18.put_int(&mut bytes, 0x1c);
        MoqtVersion::V18.put_int(&mut bytes, u64::MAX);
        MoqtVersion::V18.put_int(&mut bytes, 0);
        bytes.push(0);
        MoqtVersion::V18.put_int(&mut bytes, 0);
        // Second object: a group delta of 0 means one past u64::MAX.
        MoqtVersion::V18.put_int(&mut bytes, 0x0c);
        MoqtVersion::V18.put_int(&mut bytes, 0);
        MoqtVersion::V18.put_int(&mut bytes, 0);
        MoqtVersion::V18.put_int(&mut bytes, 0);

        let mut decoder = FetchStreamDecoder::new(MoqtVersion::V18, 1 << 20);
        decoder.push(&bytes).expect("push");
        assert_eq!(
            decoder.next_item(),
            Ok(Some(FetchItem::Header { request_id: 0 }))
        );
        assert!(matches!(
            decoder.next_item(),
            Ok(Some(FetchItem::Object(_)))
        ));
        assert_eq!(decoder.next_item(), Err(MoqtError::Malformed));
    }

    /// An end-of-range marker yields no media but still moves the location the
    /// next object is coded against.
    #[test]
    fn an_end_of_range_marker_is_a_gap_not_an_object() {
        let version = MoqtVersion::V16;
        let mut bytes = Vec::new();
        version.put_int(&mut bytes, FETCH_HEADER_TYPE);
        version.put_int(&mut bytes, 3);
        version.put_int(&mut bytes, END_OF_NON_EXISTENT_RANGE);
        version.put_int(&mut bytes, 2); // group
        version.put_int(&mut bytes, 9); // object
        version.put_int(&mut bytes, 0); // payload length
                                        // A following object takes the prior group and object + 1.
        version.put_int(&mut bytes, 0x00);
        version.put_int(&mut bytes, 1);
        bytes.push(b'z');

        let items = decoded(version, &bytes);
        assert_eq!(
            items,
            vec![
                FetchItem::Header { request_id: 3 },
                FetchItem::Gap {
                    group_id: 2,
                    object_id: 9
                },
                object(2, 10, b"z"),
            ]
        );
    }
}
