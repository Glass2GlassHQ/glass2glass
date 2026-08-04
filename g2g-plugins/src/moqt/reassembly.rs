//! Receive side of the MoQ Transport data plane: decoding one subgroup stream,
//! and putting the objects from many concurrent subgroup streams back into
//! (group id, object id) order.
//!
//! A subgroup is a separate unidirectional QUIC stream, so streams for
//! different groups (and for different subgroups of one group) arrive at the
//! same time and in no particular order. A track's objects are ordered by
//! (group id, object id) across all of them, which is what
//! [`Reassembler`] restores.
//!
//! Object ids are delta-coded per stream: the first object's delta *is* its
//! absolute id, and every later one is `previous + delta + 1`
//! (`moq-transport/src/session/subscriber.rs`, `recv_subgroup_objects`).
//!
//! Everything here decodes bytes a relay sent us, so lengths and ids are
//! attacker-controlled: nothing is preallocated from them, the arithmetic is
//! checked, and both the per-object size and the total buffered amount are
//! bounded.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use super::coding::{MoqtError, Reader};
use super::data::{ObjectStatus, SubgroupHeader, SubgroupObjectHeader};

/// Headroom over `max_object_bytes` a stream decoder's buffer may hold: one
/// object header plus the 64 KiB the codec allows an extension block.
const DECODER_SLACK: usize = 128 * 1024;

/// Read size for a subgroup stream.
pub const DATA_READ_CHUNK: usize = 16 * 1024;

/// One whole object off a subgroup stream, with its absolute ids resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceivedObject {
    pub group_id: u64,
    pub object_id: u64,
    pub status: ObjectStatus,
    pub payload: Vec<u8>,
}

/// What a [`SubgroupStreamDecoder`] produced from the bytes it holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamItem {
    /// The stream header, always the first item.
    Header(SubgroupHeader),
    Object(ReceivedObject),
}

/// Decodes one subgroup stream incrementally: bytes in, whole objects out.
///
/// Objects are delivered whole rather than streamed, because a MoQT object here
/// is one CMAF chunk and the demuxer downstream wants whole boxes anyway.
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

    /// Append bytes read off the stream.
    pub fn push(&mut self, bytes: &[u8]) -> Result<(), MoqtError> {
        // A peer that opens an object it never finishes must not grow this
        // without limit; the per-object length is checked in `next_item`, this
        // bounds the header and extension block that precede it.
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
            let mut r = Reader::new(&self.buf);
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

        let mut r = Reader::new(&self.buf);
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
        let object_id = match self.prev_object_id {
            // The delta counts the distance to the previous id less one, so a
            // run of consecutive objects encodes as zeroes.
            Some(prev) => prev
                .checked_add(object.object_id_delta)
                .and_then(|id| id.checked_add(1))
                .ok_or(MoqtError::Malformed)?,
            None => object.object_id_delta,
        };
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

#[derive(Debug, Default)]
struct PendingGroup {
    objects: BTreeMap<u64, Vec<u8>>,
    bytes: usize,
    streams_opened: u32,
    streams_closed: u32,
    /// An `EndOfGroup` / `EndOfTrack` object, or a flush, says no more is coming.
    ended: bool,
}

impl PendingGroup {
    /// Nothing more will arrive for this group: it was marked ended, or every
    /// stream that carried it has finished.
    fn complete(&self) -> bool {
        self.ended || (self.streams_opened > 0 && self.streams_opened == self.streams_closed)
    }
}

#[derive(Debug, Clone, Copy)]
struct Cursor {
    group: u64,
    next_object: u64,
}

/// Counters for what the ordering policy had to throw away.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReassemblyStats {
    /// Groups abandoned without completing (a bound forced the skip).
    pub groups_dropped: u64,
    /// Objects thrown away: late (below the cursor), duplicated, or lost inside
    /// a group that completed with a hole.
    pub objects_dropped: u64,
    /// Objects handed downstream.
    pub objects_emitted: u64,
}

/// Reorders one track's objects into (group id, object id) order under a fixed
/// memory bound.
///
/// The policy, because a live subscriber cannot wait forever:
///
/// - Playback starts at the first group whose object 0 arrives, so joining
///   mid-group skips to the next group rather than emitting a partial one.
/// - Objects are emitted strictly in order from the cursor. Anything below it
///   (a late stream, a duplicate) is dropped and counted.
/// - A group is done when every stream that carried it has finished, or an
///   `EndOfGroup` object closed it. Then the cursor moves to the next group id
///   and waits there.
/// - A hole in a group that is already done cannot be filled, so the cursor
///   jumps to the lowest object still buffered in it.
/// - A group that never completes is bounded by `max_groups` and `max_bytes`:
///   when either is exceeded the oldest group is dropped whole and the cursor
///   moves past it. Buffering never grows, and the stream continues at the next
///   group boundary instead of stalling.
#[derive(Debug)]
pub struct Reassembler {
    max_groups: usize,
    max_bytes: usize,
    groups: BTreeMap<u64, PendingGroup>,
    cursor: Option<Cursor>,
    /// Groups below this were emitted, dropped or skipped: their objects are
    /// refused rather than reordered backwards.
    floor: u64,
    buffered_bytes: usize,
    stats: ReassemblyStats,
}

impl Reassembler {
    pub fn new(max_groups: usize, max_bytes: usize) -> Self {
        Self {
            // A zero bound would drop everything on arrival, so both floor at 1.
            max_groups: max_groups.max(1),
            max_bytes: max_bytes.max(1),
            groups: BTreeMap::new(),
            cursor: None,
            floor: 0,
            buffered_bytes: 0,
            stats: ReassemblyStats::default(),
        }
    }

    pub fn stats(&self) -> ReassemblyStats {
        self.stats
    }

    pub fn buffered_bytes(&self) -> usize {
        self.buffered_bytes
    }

    pub fn buffered_groups(&self) -> usize {
        self.groups.len()
    }

    /// A subgroup stream for `group_id` opened.
    pub fn stream_opened(&mut self, group_id: u64) {
        if group_id < self.floor {
            return;
        }
        self.groups.entry(group_id).or_default().streams_opened += 1;
    }

    /// A subgroup stream for `group_id` ended, cleanly or not. Either way it
    /// will deliver nothing more.
    pub fn stream_closed(&mut self, group_id: u64) {
        if let Some(group) = self.groups.get_mut(&group_id) {
            group.streams_closed = group.streams_closed.saturating_add(1);
        }
    }

    /// Buffer one object. Late and duplicate objects are dropped here.
    pub fn push(&mut self, object: ReceivedObject) {
        if object.group_id < self.floor {
            self.stats.objects_dropped += 1;
            return;
        }
        if let Some(cursor) = self.cursor {
            if object.group_id == cursor.group && object.object_id < cursor.next_object {
                self.stats.objects_dropped += 1;
                return;
            }
        }
        let ends_group = matches!(
            object.status,
            ObjectStatus::EndOfGroup | ObjectStatus::EndOfTrack
        );
        let group = self.groups.entry(object.group_id).or_default();
        if group.objects.contains_key(&object.object_id) {
            self.stats.objects_dropped += 1;
            return;
        }
        if ends_group {
            group.ended = true;
        }
        let len = object.payload.len();
        group.bytes = group.bytes.saturating_add(len);
        group.objects.insert(object.object_id, object.payload);
        self.buffered_bytes = self.buffered_bytes.saturating_add(len);
    }

    /// Every payload that is now in order, oldest first. Empty when the next
    /// object is still missing and no bound has been hit.
    pub fn drain(&mut self) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        loop {
            self.emit_ready(&mut out);
            if !self.enforce_bounds() {
                break;
            }
        }
        out
    }

    /// End of the track: nothing more will arrive, so every buffered group is
    /// complete and whatever is left comes out in order.
    pub fn flush(&mut self) -> Vec<Vec<u8>> {
        for group in self.groups.values_mut() {
            group.ended = true;
        }
        let mut out = Vec::new();
        loop {
            self.emit_ready(&mut out);
            if self.groups.is_empty() {
                break;
            }
            // The cursor is parked on a group that never arrived; skip to what
            // did.
            self.skip_to_lowest();
        }
        out
    }

    /// Emit from the cursor for as long as the next object is present.
    fn emit_ready(&mut self, out: &mut Vec<Vec<u8>>) {
        loop {
            let Some(cursor) = self.cursor else {
                // Not started: the first group whose object 0 is here decides
                // where playback begins.
                match self
                    .groups
                    .iter()
                    .find(|(_, g)| g.objects.contains_key(&0))
                    .map(|(id, _)| *id)
                {
                    Some(group) => {
                        self.drop_below(group);
                        self.cursor = Some(Cursor {
                            group,
                            next_object: 0,
                        });
                        continue;
                    }
                    None => return,
                }
            };
            let Some(group) = self.groups.get_mut(&cursor.group) else {
                // Nothing for this group has arrived yet: wait for it, or for a
                // bound to move us on.
                return;
            };
            if let Some(payload) = group.objects.remove(&cursor.next_object) {
                group.bytes = group.bytes.saturating_sub(payload.len());
                self.buffered_bytes = self.buffered_bytes.saturating_sub(payload.len());
                self.cursor = Some(Cursor {
                    next_object: cursor.next_object.saturating_add(1),
                    ..cursor
                });
                // A zero-length object is a status marker, not media.
                if !payload.is_empty() {
                    self.stats.objects_emitted += 1;
                    out.push(payload);
                }
                continue;
            }
            if !group.complete() {
                return;
            }
            match group.objects.keys().next().copied() {
                // A hole in a finished group can never be filled.
                Some(next) => {
                    self.stats.objects_dropped += next.saturating_sub(cursor.next_object);
                    self.cursor = Some(Cursor {
                        next_object: next,
                        ..cursor
                    });
                }
                // Drained and finished: on to the next group.
                None => self.close_group(cursor.group),
            }
        }
    }

    /// Drop the oldest group when a bound is exceeded. Returns whether it did,
    /// so the caller re-runs the emit loop.
    fn enforce_bounds(&mut self) -> bool {
        if self.groups.len() <= self.max_groups && self.buffered_bytes <= self.max_bytes {
            return false;
        }
        let Some(lowest) = self.groups.keys().next().copied() else {
            return false;
        };
        match self.cursor {
            // The cursor is parked on a group that never arrived: skip to the
            // oldest one that did, rather than dropping real data.
            Some(cursor) if cursor.group < lowest => {
                self.stats.groups_dropped += 1;
                self.drop_below(lowest);
                self.cursor = Some(Cursor {
                    group: lowest,
                    next_object: 0,
                });
            }
            _ => {
                self.stats.groups_dropped += 1;
                self.close_group(lowest);
            }
        }
        true
    }

    /// Retire `group` and park the cursor at the start of the next one.
    fn close_group(&mut self, group: u64) {
        self.remove_group(group);
        self.floor = group.saturating_add(1);
        self.cursor = Some(Cursor {
            group: self.floor,
            next_object: 0,
        });
    }

    /// Move the cursor to the oldest buffered group (used by `flush`, where
    /// waiting for a group that never arrived would stall).
    fn skip_to_lowest(&mut self) {
        let Some(lowest) = self.groups.keys().next().copied() else {
            return;
        };
        self.drop_below(lowest);
        let next_object = self
            .groups
            .get(&lowest)
            .and_then(|g| g.objects.keys().next().copied())
            .unwrap_or(0);
        self.cursor = Some(Cursor {
            group: lowest,
            next_object,
        });
    }

    fn drop_below(&mut self, group: u64) {
        let stale: Vec<u64> = self.groups.range(..group).map(|(id, _)| *id).collect();
        for id in stale {
            self.stats.groups_dropped += 1;
            self.remove_group(id);
        }
        self.floor = self.floor.max(group);
    }

    fn remove_group(&mut self, group: u64) {
        if let Some(dead) = self.groups.remove(&group) {
            self.stats.objects_dropped += dead.objects.len() as u64;
            self.buffered_bytes = self.buffered_bytes.saturating_sub(dead.bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moqt::data::StreamHeaderType;
    use alloc::vec;

    fn object(group_id: u64, object_id: u64, payload: &[u8]) -> ReceivedObject {
        ReceivedObject {
            group_id,
            object_id,
            status: ObjectStatus::Normal,
            payload: payload.to_vec(),
        }
    }

    /// One group's stream: opened, its objects, closed.
    fn deliver(r: &mut Reassembler, group: u64, objects: &[(u64, &[u8])]) {
        r.stream_opened(group);
        for (id, payload) in objects {
            r.push(object(group, *id, payload));
        }
        r.stream_closed(group);
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

    fn header(group_id: u64) -> SubgroupHeader {
        SubgroupHeader {
            header_type: StreamHeaderType::SubgroupIdExt,
            track_alias: 4,
            group_id,
            subgroup_id: Some(0),
            publisher_priority: 127,
        }
    }

    #[test]
    fn a_stream_decodes_to_consecutive_object_ids_one_byte_at_a_time() {
        let bytes = encoded_stream(&header(7), &[b"aaa", b"bb", b"c"]);
        let mut decoder = SubgroupStreamDecoder::new(1024);
        let mut items = Vec::new();
        for byte in &bytes {
            decoder.push(&[*byte]).expect("push");
            while let Some(item) = decoder.next_item().expect("decode") {
                items.push(item);
            }
        }
        assert_eq!(items.len(), 4, "the header plus three objects");
        assert_eq!(items[0], StreamItem::Header(header(7)));
        // The delta is zero for each, so the ids run 0, 1, 2.
        for (i, expected) in [b"aaa".as_slice(), b"bb", b"c"].iter().enumerate() {
            assert_eq!(
                items[i + 1],
                StreamItem::Object(object(7, i as u64, expected))
            );
        }
    }

    #[test]
    fn malformed_data_plane_input_fails_the_decode() {
        // A stream header type that is not a subgroup.
        let mut decoder = SubgroupStreamDecoder::new(1024);
        decoder.push(&[0x16, 0x04, 0x07, 0x7f]).expect("push");
        assert_eq!(decoder.next_item(), Err(MoqtError::Malformed));

        // A truncated object header needs more bytes, it is not an error.
        let mut decoder = SubgroupStreamDecoder::new(1024);
        let bytes = encoded_stream(&header(1), &[b"payload"]);
        decoder.push(&bytes[..bytes.len() - 3]).expect("push");
        assert_eq!(decoder.next_item(), Ok(Some(StreamItem::Header(header(1)))));
        assert_eq!(
            decoder.next_item(),
            Ok(None),
            "the payload is not all here yet"
        );

        // An extension block whose length overruns what is buffered needs more
        // bytes; one past the 64 KiB the codec allows is a violation.
        let mut decoder = SubgroupStreamDecoder::new(1 << 20);
        let mut bytes = Vec::new();
        header(1).encode(&mut bytes).expect("header");
        // delta 0, extension length 200 with two bytes behind it.
        bytes.extend_from_slice(&[0x00, 0x40, 0xc8, 0x01, 0x02]);
        decoder.push(&bytes).expect("push");
        assert_eq!(decoder.next_item(), Ok(Some(StreamItem::Header(header(1)))));
        assert_eq!(
            decoder.next_item(),
            Ok(None),
            "incomplete, not yet a violation"
        );

        let mut decoder = SubgroupStreamDecoder::new(1 << 20);
        let mut bytes = Vec::new();
        header(1).encode(&mut bytes).expect("header");
        // delta 0, then an extension length of 70000.
        bytes.extend_from_slice(&[0x00, 0x80, 0x01, 0x11, 0x70]);
        decoder.push(&bytes).expect("push");
        assert_eq!(decoder.next_item(), Ok(Some(StreamItem::Header(header(1)))));
        assert_eq!(decoder.next_item(), Err(MoqtError::Malformed));

        // A payload length past the per-object bound is refused without
        // allocating on it.
        let mut decoder = SubgroupStreamDecoder::new(16);
        let mut bytes = Vec::new();
        header(1).encode(&mut bytes).expect("header");
        bytes.extend_from_slice(&[0x00, 0x00, 0x80, 0x10, 0x00, 0x00]);
        decoder.push(&bytes).expect("push");
        assert_eq!(decoder.next_item(), Ok(Some(StreamItem::Header(header(1)))));
        assert_eq!(decoder.next_item(), Err(MoqtError::Malformed));

        // ...and a stream that keeps sending without ever completing an object
        // is refused before its buffer can grow past the bound.
        let mut decoder = SubgroupStreamDecoder::new(16);
        let mut bytes = Vec::new();
        header(1).encode(&mut bytes).expect("header");
        decoder.push(&bytes).expect("push");
        assert_eq!(decoder.next_item(), Ok(Some(StreamItem::Header(header(1)))));
        assert_eq!(
            decoder.push(&vec![0xffu8; 200 * 1024]),
            Err(MoqtError::Malformed)
        );
    }

    #[test]
    fn objects_come_out_in_group_and_object_order() {
        let mut r = Reassembler::new(8, 1 << 20);
        deliver(&mut r, 0, &[(0, b"a0"), (1, b"a1")]);
        assert_eq!(r.drain(), vec![b"a0".to_vec(), b"a1".to_vec()]);
        deliver(&mut r, 1, &[(0, b"b0"), (1, b"b1")]);
        assert_eq!(r.drain(), vec![b"b0".to_vec(), b"b1".to_vec()]);
        assert_eq!(r.stats().objects_emitted, 4);
        assert_eq!(r.stats().objects_dropped, 0);
    }

    #[test]
    fn a_later_group_waits_for_the_earlier_one() {
        let mut r = Reassembler::new(8, 1 << 20);
        // Group 1's whole stream lands before group 0 has finished arriving.
        r.stream_opened(0);
        r.push(object(0, 0, b"a0"));
        deliver(&mut r, 1, &[(0, b"b0"), (1, b"b1")]);
        assert_eq!(
            r.drain(),
            vec![b"a0".to_vec()],
            "group 1 is held until group 0 ends"
        );
        r.push(object(0, 1, b"a1"));
        r.stream_closed(0);
        assert_eq!(
            r.drain(),
            vec![b"a1".to_vec(), b"b0".to_vec(), b"b1".to_vec()]
        );
        assert_eq!(r.stats().groups_dropped, 0);
    }

    #[test]
    fn interleaved_subgroup_streams_of_one_group_merge_in_object_order() {
        let mut r = Reassembler::new(8, 1 << 20);
        // Two streams carry one group: odd objects on one, even on the other.
        r.stream_opened(3);
        r.stream_opened(3);
        r.push(object(3, 1, b"o1"));
        r.push(object(3, 3, b"o3"));
        assert!(r.drain().is_empty(), "object 0 is still missing");
        r.push(object(3, 0, b"o0"));
        assert_eq!(r.drain(), vec![b"o0".to_vec(), b"o1".to_vec()]);
        r.push(object(3, 2, b"o2"));
        assert_eq!(r.drain(), vec![b"o2".to_vec(), b"o3".to_vec()]);
        r.stream_closed(3);
        r.stream_closed(3);
        assert_eq!(r.stats().objects_dropped, 0);
    }

    #[test]
    fn a_duplicate_object_is_dropped_not_re_emitted() {
        let mut r = Reassembler::new(8, 1 << 20);
        r.stream_opened(0);
        r.push(object(0, 0, b"a0"));
        r.push(object(0, 0, b"a0 again"));
        assert_eq!(r.drain(), vec![b"a0".to_vec()]);
        // A copy that arrives after the cursor passed it is dropped too.
        r.push(object(0, 0, b"a0 later"));
        assert!(r.drain().is_empty());
        assert_eq!(r.stats().objects_dropped, 2);
        assert_eq!(r.stats().objects_emitted, 1);
    }

    #[test]
    fn joining_mid_group_starts_at_the_next_group() {
        let mut r = Reassembler::new(8, 1 << 20);
        // The relay's first stream drops us into the middle of group 4.
        r.stream_opened(4);
        r.push(object(4, 3, b"partial"));
        r.push(object(4, 4, b"partial2"));
        assert!(r.drain().is_empty(), "a partial group is not played");
        r.stream_closed(4);
        deliver(&mut r, 5, &[(0, b"c0"), (1, b"c1")]);
        assert_eq!(r.drain(), vec![b"c0".to_vec(), b"c1".to_vec()]);
        assert_eq!(r.stats().groups_dropped, 1);
    }

    #[test]
    fn a_group_that_never_completes_is_dropped_at_the_bound() {
        let mut r = Reassembler::new(2, 1 << 20);
        // Group 0 stalls after object 0: its stream never closes.
        r.stream_opened(0);
        r.push(object(0, 0, b"a0"));
        r.push(object(0, 2, b"a2"));
        assert_eq!(r.drain(), vec![b"a0".to_vec()]);
        deliver(&mut r, 1, &[(0, b"b0")]);
        deliver(&mut r, 2, &[(0, b"c0")]);
        // Three groups buffered against a bound of two: group 0 goes, and the
        // stream continues at the next group boundary instead of stalling.
        assert_eq!(r.drain(), vec![b"b0".to_vec(), b"c0".to_vec()]);
        assert_eq!(r.stats().groups_dropped, 1);
        assert_eq!(r.buffered_groups(), 0);
        assert_eq!(r.buffered_bytes(), 0);

        // Objects for the abandoned group are refused rather than reordered
        // backwards.
        r.push(object(0, 1, b"a1"));
        assert!(r.drain().is_empty());
        assert_eq!(r.stats().objects_dropped, 2, "a2 with the group, then a1");
    }

    #[test]
    fn the_byte_bound_holds_when_groups_never_end() {
        let mut r = Reassembler::new(64, 512);
        // Every group has a hole at object 1 and a stream that never ends, so
        // only its first object can ever be emitted. Buffering must stay under
        // the bound however long the publisher keeps this up.
        let big = vec![0xAAu8; 200];
        let mut emitted = 0usize;
        for group in 0..40u64 {
            r.stream_opened(group);
            r.push(object(group, 0, &big));
            r.push(object(group, 2, &big));
            emitted += r.drain().len();
            assert!(
                r.buffered_bytes() <= 512,
                "buffered {} bytes past the bound",
                r.buffered_bytes()
            );
        }
        assert_eq!(emitted, 40, "every group's first object still played");
        assert!(r.stats().groups_dropped > 0);
    }

    #[test]
    fn a_hole_in_a_finished_group_does_not_stall_the_stream() {
        let mut r = Reassembler::new(8, 1 << 20);
        r.stream_opened(0);
        r.push(object(0, 0, b"a0"));
        r.push(object(0, 2, b"a2"));
        // The stream ends without ever sending object 1: it is lost, not late.
        r.stream_closed(0);
        assert_eq!(r.drain(), vec![b"a0".to_vec(), b"a2".to_vec()]);
        assert_eq!(r.stats().objects_dropped, 1, "the missing object 1");
    }

    #[test]
    fn an_end_of_group_object_closes_the_group_without_the_stream_ending() {
        let mut r = Reassembler::new(8, 1 << 20);
        r.stream_opened(0);
        r.push(object(0, 0, b"a0"));
        r.push(ReceivedObject {
            group_id: 0,
            object_id: 1,
            status: ObjectStatus::EndOfGroup,
            payload: Vec::new(),
        });
        deliver(&mut r, 1, &[(0, b"b0")]);
        assert_eq!(
            r.drain(),
            vec![b"a0".to_vec(), b"b0".to_vec()],
            "the marker itself is not media"
        );
    }

    #[test]
    fn flush_empties_a_group_whose_stream_never_finished() {
        let mut r = Reassembler::new(8, 1 << 20);
        r.stream_opened(0);
        r.push(object(0, 0, b"a0"));
        r.push(object(0, 1, b"a1"));
        assert_eq!(r.drain(), vec![b"a0".to_vec(), b"a1".to_vec()]);
        r.push(object(0, 2, b"a2"));
        assert_eq!(r.flush(), vec![b"a2".to_vec()]);
        assert_eq!(r.buffered_groups(), 0);
    }
}
