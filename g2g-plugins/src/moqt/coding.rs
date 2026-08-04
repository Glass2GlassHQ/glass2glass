//! Primitive wire coding for IETF MoQ Transport draft-16: QUIC varints, the
//! byte/string forms, track namespaces and names, and the delta-coded
//! Key-Value-Pair sequences that carry every parameter.
//!
//! Layouts follow `moq-rs/moq-transport/src/coding/` (varint.rs, tuple.rs,
//! track_namespace.rs, kvp.rs). Everything here decodes peer bytes, so every
//! count, length and delta is bounded before it is used and nothing is
//! preallocated from a peer-supplied count.

use alloc::string::String;
use alloc::vec::Vec;

/// Why a decode did not produce a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoqtError {
    /// The buffer holds a valid prefix but not the whole item: read more and
    /// retry. Never a protocol violation.
    Incomplete,
    /// The bytes cannot be a draft-16 message (a bad length, an out-of-range
    /// count, a reserved value): a PROTOCOL_VIOLATION on the wire.
    Malformed,
}

/// Largest value a QUIC varint can carry (2^62 - 1).
pub const VARINT_MAX: u64 = (1 << 62) - 1;

/// A full track name (namespace fields plus track name) may not exceed this
/// (draft-16 §2.4.1), which also bounds a single namespace field.
pub const MAX_FULL_TRACK_NAME_LEN: usize = 4096;

/// A full track namespace has 1 to 32 fields; a prefix may have 0.
pub const MAX_NAMESPACE_FIELDS: usize = 32;

/// A KVP bytes value is length-prefixed by a varint but bounded to 16 bits.
const MAX_KVP_BYTES_LEN: usize = u16::MAX as usize;

/// A reason phrase is bounded to 1024 bytes, a session URI to 8192.
pub const MAX_REASON_PHRASE_LEN: usize = 1024;
pub const MAX_SESSION_URI_LEN: usize = 8192;

// ---------------------------------------------------------------- encoding

/// Append `v` as a QUIC varint. Values above [`VARINT_MAX`] are not
/// representable and saturate; every value this crate writes is either a
/// constant or came off the wire as a varint, so saturation is unreachable in
/// practice and never silently truncates a smaller field.
pub fn put_varint(out: &mut Vec<u8>, v: u64) {
    let v = v.min(VARINT_MAX);
    if v < (1 << 6) {
        out.push(v as u8);
    } else if v < (1 << 14) {
        out.extend_from_slice(&((v as u16) | 0x4000).to_be_bytes());
    } else if v < (1 << 30) {
        out.extend_from_slice(&((v as u32) | 0x8000_0000).to_be_bytes());
    } else {
        out.extend_from_slice(&(v | 0xC000_0000_0000_0000).to_be_bytes());
    }
}

/// Append a varint-length-prefixed byte string.
pub fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    put_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

// ---------------------------------------------------------------- decoding

/// A cursor over a byte slice that decodes draft-16 primitives.
#[derive(Debug)]
pub struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, at: 0 }
    }

    /// Bytes consumed so far.
    pub fn position(&self) -> usize {
        self.at
    }

    pub fn remaining(&self) -> usize {
        self.buf.len() - self.at
    }

    pub fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    pub fn varint(&mut self) -> Result<u64, MoqtError> {
        let first = *self.buf.get(self.at).ok_or(MoqtError::Incomplete)?;
        let len = 1usize << (first >> 6);
        let raw = self.bytes(len)?;
        let mut v = u64::from(raw[0] & 0x3F);
        for b in &raw[1..] {
            v = (v << 8) | u64::from(*b);
        }
        Ok(v)
    }

    /// A varint that has to index memory. A value beyond `usize` is a length
    /// that cannot be satisfied, which is malformed rather than incomplete.
    pub fn varint_usize(&mut self) -> Result<usize, MoqtError> {
        usize::try_from(self.varint()?).map_err(|_| MoqtError::Malformed)
    }

    pub fn u8(&mut self) -> Result<u8, MoqtError> {
        Ok(self.bytes(1)?[0])
    }

    pub fn u16(&mut self) -> Result<u16, MoqtError> {
        let b = self.bytes(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    /// A one-byte boolean; anything but 0 or 1 is a protocol violation.
    pub fn bool(&mut self) -> Result<bool, MoqtError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(MoqtError::Malformed),
        }
    }

    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8], MoqtError> {
        let end = self.at.checked_add(n).ok_or(MoqtError::Malformed)?;
        let out = self.buf.get(self.at..end).ok_or(MoqtError::Incomplete)?;
        self.at = end;
        Ok(out)
    }

    /// Consume and return everything left.
    pub fn rest(&mut self) -> &'a [u8] {
        let out = &self.buf[self.at..];
        self.at = self.buf.len();
        out
    }

    /// A varint-length-prefixed byte string, rejected past `max`.
    pub fn length_bytes(&mut self, max: usize) -> Result<&'a [u8], MoqtError> {
        let len = self.varint_usize()?;
        if len > max {
            return Err(MoqtError::Malformed);
        }
        self.bytes(len)
    }

    /// A bounded UTF-8 string (`ReasonPhrase` / `SessionUri`).
    pub fn string(&mut self, max: usize) -> Result<String, MoqtError> {
        let raw = self.length_bytes(max)?;
        core::str::from_utf8(raw)
            .map(String::from)
            .map_err(|_| MoqtError::Malformed)
    }
}

// ------------------------------------------------------ namespaces + names

/// A track namespace: an ordered tuple of non-empty byte fields.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TrackNamespace(pub Vec<Vec<u8>>);

impl TrackNamespace {
    /// Split a `/`-separated path into fields. Empty segments are dropped, so
    /// `/live/cam` and `live/cam` name the same namespace.
    pub fn from_path(path: &str) -> Self {
        Self(
            path.split('/')
                .filter(|s| !s.is_empty())
                .map(|s| s.as_bytes().to_vec())
                .collect(),
        )
    }

    /// Render as the leading-slash path form the reference catalog uses.
    pub fn to_path(&self) -> String {
        let mut out = String::new();
        for field in &self.0 {
            out.push('/');
            out.push_str(&String::from_utf8_lossy(field));
        }
        out
    }

    pub fn byte_len(&self) -> usize {
        self.0.iter().map(|f| f.len()).sum()
    }

    pub fn encode(&self, out: &mut Vec<u8>) -> Result<(), MoqtError> {
        encode_namespace(&self.0, 1, out)
    }

    pub fn decode(r: &mut Reader<'_>) -> Result<Self, MoqtError> {
        Ok(Self(decode_namespace(r, 1)?))
    }
}

/// A namespace prefix (`SUBSCRIBE_NAMESPACE`), which unlike a full namespace
/// may be empty.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TrackNamespacePrefix(pub Vec<Vec<u8>>);

impl TrackNamespacePrefix {
    pub fn encode(&self, out: &mut Vec<u8>) -> Result<(), MoqtError> {
        encode_namespace(&self.0, 0, out)
    }

    pub fn decode(r: &mut Reader<'_>) -> Result<Self, MoqtError> {
        Ok(Self(decode_namespace(r, 0)?))
    }
}

fn encode_namespace(fields: &[Vec<u8>], min: usize, out: &mut Vec<u8>) -> Result<(), MoqtError> {
    if fields.len() < min || fields.len() > MAX_NAMESPACE_FIELDS {
        return Err(MoqtError::Malformed);
    }
    let total: usize = fields.iter().map(|f| f.len()).sum();
    if total > MAX_FULL_TRACK_NAME_LEN {
        return Err(MoqtError::Malformed);
    }
    put_varint(out, fields.len() as u64);
    for field in fields {
        if field.is_empty() {
            return Err(MoqtError::Malformed);
        }
        put_bytes(out, field);
    }
    Ok(())
}

fn decode_namespace(r: &mut Reader<'_>, min: usize) -> Result<Vec<Vec<u8>>, MoqtError> {
    let count = r.varint_usize()?;
    if count < min || count > MAX_NAMESPACE_FIELDS {
        return Err(MoqtError::Malformed);
    }
    // `count` is bounded to 32 above, so reserving from it is safe here.
    let mut fields = Vec::with_capacity(count);
    let mut total = 0usize;
    for _ in 0..count {
        let field = r.length_bytes(MAX_FULL_TRACK_NAME_LEN)?;
        if field.is_empty() {
            return Err(MoqtError::Malformed);
        }
        total = total.saturating_add(field.len());
        if total > MAX_FULL_TRACK_NAME_LEN {
            return Err(MoqtError::Malformed);
        }
        fields.push(field.to_vec());
    }
    Ok(fields)
}

/// A track name: arbitrary bytes, possibly empty, bounded with its namespace.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TrackName(pub Vec<u8>);

impl TrackName {
    pub fn new(name: &str) -> Self {
        Self(name.as_bytes().to_vec())
    }

    pub fn as_str_lossy(&self) -> String {
        String::from_utf8_lossy(&self.0).into_owned()
    }

    pub fn encode(&self, out: &mut Vec<u8>) -> Result<(), MoqtError> {
        if self.0.len() > MAX_FULL_TRACK_NAME_LEN {
            return Err(MoqtError::Malformed);
        }
        put_bytes(out, &self.0);
        Ok(())
    }

    pub fn decode(r: &mut Reader<'_>) -> Result<Self, MoqtError> {
        Ok(Self(r.length_bytes(MAX_FULL_TRACK_NAME_LEN)?.to_vec()))
    }
}

/// Namespace plus track name must fit in 4096 bytes together.
pub fn validate_full_track_name(ns: &TrackNamespace, name: &TrackName) -> Result<(), MoqtError> {
    if ns.byte_len().saturating_add(name.0.len()) > MAX_FULL_TRACK_NAME_LEN {
        return Err(MoqtError::Malformed);
    }
    Ok(())
}

// ------------------------------------------------------- key-value pairs

/// A parameter value. The key's parity picks the form: even keys carry a
/// varint, odd keys a length-prefixed byte string (draft-16 §1.4.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParamValue {
    Int(u64),
    Bytes(Vec<u8>),
}

/// An ordered parameter list. Keys are absolute here; the delta coding is
/// applied only on the wire.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Params(pub Vec<(u64, ParamValue)>);

impl Params {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_int(&mut self, key: u64, value: u64) {
        self.set(key, ParamValue::Int(value));
    }

    pub fn set_bytes(&mut self, key: u64, value: Vec<u8>) {
        self.set(key, ParamValue::Bytes(value));
    }

    pub fn set(&mut self, key: u64, value: ParamValue) {
        match self.0.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 = value,
            None => self.0.push((key, value)),
        }
    }

    pub fn get_int(&self, key: u64) -> Option<u64> {
        self.0.iter().find_map(|(k, v)| match v {
            ParamValue::Int(i) if *k == key => Some(*i),
            _ => None,
        })
    }

    /// Count-prefixed sequence (control-message parameters).
    pub fn encode(&self, out: &mut Vec<u8>) -> Result<(), MoqtError> {
        put_varint(out, self.0.len() as u64);
        self.encode_pairs(out)
    }

    pub fn decode(r: &mut Reader<'_>) -> Result<Self, MoqtError> {
        let count = r.varint()?;
        let mut pairs = Vec::new();
        let mut prev = 0u64;
        for _ in 0..count {
            // No reserve from `count`: each iteration needs at least two bytes,
            // so a bogus count runs out of buffer instead of allocating on it.
            let (key, value) = decode_pair(r, prev)?;
            prev = key;
            pairs.push((key, value));
        }
        Ok(Self(pairs))
    }

    /// Byte-length-prefixed sequence with no count (data-plane extension
    /// headers, and the same body a track-extension field carries).
    pub fn encode_extension_headers(&self, out: &mut Vec<u8>) -> Result<(), MoqtError> {
        let mut body = Vec::new();
        self.encode_pairs(&mut body)?;
        put_bytes(out, &body);
        Ok(())
    }

    pub fn decode_extension_headers(r: &mut Reader<'_>) -> Result<Self, MoqtError> {
        let body = r.length_bytes(MAX_KVP_BYTES_LEN)?;
        Self::decode_to_end(&mut Reader::new(body))
    }

    /// Pairs running to the end of the buffer with neither count nor length
    /// (the track-extensions tail of SUBSCRIBE_OK / PUBLISH / FETCH_OK).
    pub fn decode_to_end(r: &mut Reader<'_>) -> Result<Self, MoqtError> {
        let mut pairs = Vec::new();
        let mut prev = 0u64;
        while !r.is_empty() {
            let (key, value) = decode_pair(r, prev)?;
            prev = key;
            pairs.push((key, value));
        }
        Ok(Self(pairs))
    }

    /// The bare pair sequence, no count and no length prefix.
    pub fn encode_pairs(&self, out: &mut Vec<u8>) -> Result<(), MoqtError> {
        // The wire delta must never go backwards, so encode in key order
        // regardless of insertion order.
        let mut sorted: Vec<&(u64, ParamValue)> = self.0.iter().collect();
        sorted.sort_by_key(|(k, _)| *k);
        let mut prev = 0u64;
        for (key, value) in sorted {
            let delta = key.checked_sub(prev).ok_or(MoqtError::Malformed)?;
            match value {
                ParamValue::Int(_) if key % 2 != 0 => return Err(MoqtError::Malformed),
                ParamValue::Bytes(_) if key % 2 == 0 => return Err(MoqtError::Malformed),
                _ => {}
            }
            put_varint(out, delta);
            match value {
                ParamValue::Int(v) => put_varint(out, *v),
                ParamValue::Bytes(b) => {
                    if b.len() > MAX_KVP_BYTES_LEN {
                        return Err(MoqtError::Malformed);
                    }
                    put_bytes(out, b);
                }
            }
            prev = *key;
        }
        Ok(())
    }
}

fn decode_pair(r: &mut Reader<'_>, prev: u64) -> Result<(u64, ParamValue), MoqtError> {
    let delta = r.varint()?;
    // Draft-16 §1.4.2: the running key must not overflow.
    let key = prev.checked_add(delta).ok_or(MoqtError::Malformed)?;
    let value = if key % 2 == 0 {
        ParamValue::Int(r.varint()?)
    } else {
        ParamValue::Bytes(r.length_bytes(MAX_KVP_BYTES_LEN)?.to_vec())
    };
    Ok((key, value))
}

/// Setup parameter types (draft-16 §9.3, `setup/param_types.rs`).
pub mod setup_param {
    pub const PATH: u64 = 0x1;
    pub const MAX_REQUEST_ID: u64 = 0x2;
    pub const AUTHORIZATION_TOKEN: u64 = 0x3;
    pub const MAX_AUTH_TOKEN_CACHE_SIZE: u64 = 0x4;
    pub const AUTHORITY: u64 = 0x5;
    pub const MOQT_IMPLEMENTATION: u64 = 0x7;
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn round_trip_varint(v: u64) {
        let mut out = Vec::new();
        put_varint(&mut out, v);
        assert_eq!(Reader::new(&out).varint().expect("decode"), v);
    }

    #[test]
    fn varints_match_the_quic_encoding() {
        // RFC 9000 §A.1 sample values plus each length boundary.
        let mut out = Vec::new();
        put_varint(&mut out, 37);
        assert_eq!(out, vec![0x25]);
        out.clear();
        put_varint(&mut out, 15293);
        assert_eq!(out, vec![0x7b, 0xbd]);
        out.clear();
        put_varint(&mut out, 494_878_333);
        assert_eq!(out, vec![0x9d, 0x7f, 0x3e, 0x7d]);
        out.clear();
        put_varint(&mut out, 151_288_809_941_952_652);
        assert_eq!(out, vec![0xc2, 0x19, 0x7c, 0x5e, 0xff, 0x14, 0xe8, 0x8c]);

        for v in [
            0,
            1,
            63,
            64,
            16383,
            16384,
            (1 << 30) - 1,
            1 << 30,
            VARINT_MAX,
        ] {
            round_trip_varint(v);
        }
    }

    #[test]
    fn truncated_varint_is_incomplete_not_a_panic() {
        // A 4-byte varint with only two bytes present.
        assert_eq!(
            Reader::new(&[0x9d, 0x7f]).varint(),
            Err(MoqtError::Incomplete)
        );
        assert_eq!(Reader::new(&[]).varint(), Err(MoqtError::Incomplete));
    }

    #[test]
    fn namespace_round_trips_and_matches_the_reference_tuple_layout() {
        let ns = TrackNamespace::from_path("test/ns");
        let mut out = Vec::new();
        ns.encode(&mut out).expect("encode");
        // count=2, len=4 "test", len=2 "ns"
        assert_eq!(
            out,
            vec![0x02, 0x04, b't', b'e', b's', b't', 0x02, b'n', b's']
        );
        assert_eq!(
            TrackNamespace::decode(&mut Reader::new(&out)).expect("decode"),
            ns
        );
        assert_eq!(ns.to_path(), "/test/ns");
        // A leading slash produces the same fields, not an empty leading field.
        assert_eq!(TrackNamespace::from_path("/test/ns"), ns);
    }

    #[test]
    fn namespace_rejects_absurd_and_empty_field_counts() {
        // 33 fields is over the draft's 32 limit.
        let mut out = Vec::new();
        put_varint(&mut out, 33);
        for _ in 0..33 {
            put_bytes(&mut out, b"x");
        }
        assert_eq!(
            TrackNamespace::decode(&mut Reader::new(&out)),
            Err(MoqtError::Malformed)
        );

        // Zero fields is legal for a prefix but not a full namespace.
        let zero = vec![0x00];
        assert_eq!(
            TrackNamespace::decode(&mut Reader::new(&zero)),
            Err(MoqtError::Malformed)
        );
        assert_eq!(
            TrackNamespacePrefix::decode(&mut Reader::new(&zero)).expect("prefix"),
            TrackNamespacePrefix(Vec::new())
        );

        // An empty field is a protocol violation.
        assert_eq!(
            TrackNamespace::decode(&mut Reader::new(&[0x01, 0x00])),
            Err(MoqtError::Malformed)
        );

        // A field length that overruns the buffer is incomplete, not a panic.
        assert_eq!(
            TrackNamespace::decode(&mut Reader::new(&[0x01, 0x08, b'a'])),
            Err(MoqtError::Incomplete)
        );
    }

    #[test]
    fn namespace_rejects_an_over_long_full_track_name() {
        let big = vec![b'a'; MAX_FULL_TRACK_NAME_LEN];
        let ns = TrackNamespace(vec![big]);
        assert_eq!(
            validate_full_track_name(&ns, &TrackName::new("x")),
            Err(MoqtError::Malformed)
        );
        assert_eq!(validate_full_track_name(&ns, &TrackName::default()), Ok(()));
    }

    #[test]
    fn params_delta_code_by_key_and_parity() {
        let mut params = Params::new();
        // Insert out of order: the encoder must sort so no delta goes backwards.
        params.set_int(2, 100);
        params.set_bytes(1, b"testpath".to_vec());
        let mut out = Vec::new();
        params.encode(&mut out).expect("encode");
        assert_eq!(
            out,
            vec![
                0x02, // 2 pairs
                0x01, 0x08, b't', b'e', b's', b't', b'p', b'a', b't', b'h', // key 1 (bytes)
                0x01, 0x40, 0x64, // delta 1 -> key 2 (int 100)
            ]
        );
        let decoded = Params::decode(&mut Reader::new(&out)).expect("decode");
        assert_eq!(decoded.get_int(2), Some(100));
        assert_eq!(
            decoded.0.iter().find(|(k, _)| *k == 1).map(|(_, v)| v),
            Some(&ParamValue::Bytes(b"testpath".to_vec()))
        );

        // Parity mismatches are refused on encode rather than written badly.
        let mut bad = Params::new();
        bad.set_int(1, 0);
        assert_eq!(bad.encode(&mut Vec::new()), Err(MoqtError::Malformed));
    }

    #[test]
    fn params_reject_delta_overflow_and_do_not_preallocate_on_a_huge_count() {
        // A count of 2^62-1 with no pairs behind it must fail on the buffer,
        // not on an allocation.
        let mut buf = Vec::new();
        put_varint(&mut buf, VARINT_MAX);
        assert_eq!(
            Params::decode(&mut Reader::new(&buf)),
            Err(MoqtError::Incomplete)
        );

        // Five maximum deltas overflow the u64 running key.
        let mut buf = Vec::new();
        put_varint(&mut buf, 5);
        for _ in 0..4 {
            put_varint(&mut buf, VARINT_MAX);
            // The running key alternates parity, but both forms read one varint
            // here: a zero length for the odd (bytes) keys, a zero value for the
            // even (int) ones.
            put_varint(&mut buf, 0);
        }
        put_varint(&mut buf, VARINT_MAX);
        assert_eq!(
            Params::decode(&mut Reader::new(&buf)),
            Err(MoqtError::Malformed)
        );
    }

    #[test]
    fn extension_headers_are_byte_length_prefixed() {
        let mut params = Params::new();
        params.set_int(0, 42);
        let mut out = Vec::new();
        params.encode_extension_headers(&mut out).expect("encode");
        assert_eq!(out, vec![0x02, 0x00, 0x2a]); // length 2, delta 0, value 42
        let decoded = Params::decode_extension_headers(&mut Reader::new(&out)).expect("decode");
        assert_eq!(decoded, params);

        let mut empty = Vec::new();
        Params::new()
            .encode_extension_headers(&mut empty)
            .expect("encode");
        assert_eq!(empty, vec![0x00]);
    }

    #[test]
    fn bounded_strings_reject_over_long_and_non_utf8() {
        let mut out = Vec::new();
        put_bytes(&mut out, b"why");
        assert_eq!(
            Reader::new(&out).string(MAX_REASON_PHRASE_LEN).as_deref(),
            Ok("why")
        );

        let mut too_long = Vec::new();
        put_varint(&mut too_long, MAX_REASON_PHRASE_LEN as u64 + 1);
        too_long.extend_from_slice(&[b'x'; 8]);
        assert_eq!(
            Reader::new(&too_long).string(MAX_REASON_PHRASE_LEN),
            Err(MoqtError::Malformed)
        );

        let mut bad_utf8 = Vec::new();
        put_bytes(&mut bad_utf8, &[0xff, 0xfe]);
        assert_eq!(
            Reader::new(&bad_utf8).string(MAX_REASON_PHRASE_LEN),
            Err(MoqtError::Malformed)
        );
    }
}
