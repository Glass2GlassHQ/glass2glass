//! Primitive wire coding for IETF MoQ Transport draft-18: the `vi64` variable
//! length integer, track namespaces and names, the delta-coded Key-Value-Pair
//! blocks, and the typed control-message parameters.
//!
//! Two things changed under the coding layer between draft-16 and draft-18, and
//! they are why this module exists rather than the draft-16 one being reused
//! wholesale:
//!
//! - every integer is a `vi64` (§1.4.1), not a QUIC varint: the length comes
//!   from the *leading 1 bits* of the first byte, spans 1 to 9 bytes, and
//!   reaches a full `u64` rather than 2^62 - 1.
//! - control-message parameters (§10.2) are no longer Key-Value-Pairs. Each
//!   parameter type has its own value encoding, so an unknown type cannot be
//!   skipped and the block is bounded by a count instead of a length. Receiving
//!   one is a session error, which is what [`MessageParams::decode`] returns.
//!
//! Key-Value-Pairs themselves (§1.4.3) kept their shape, so Setup Options and
//! Properties reuse [`Params`] and only the integer flavour differs.
//!
//! Everything here decodes peer bytes, so every count, length and delta is
//! bounded before it is used and nothing is preallocated from a peer-supplied
//! count.

use alloc::vec::Vec;

pub use super::super::coding::{
    validate_full_track_name, MoqtError, ParamValue, Params, Reader, TrackName, TrackNamespace,
    MAX_FULL_TRACK_NAME_LEN, MAX_KVP_BYTES_LEN, MAX_NAMESPACE_FIELDS, MAX_REASON_PHRASE_LEN,
    MAX_SESSION_URI_LEN,
};
// Draft-18 puts Location in the notational conventions (§1.4.2) rather than in
// a message definition, but it is the same `{group, object}` pair either way.
pub use super::super::message::Location;

// ---------------------------------------------------------------- encoding

/// Append `v` as a `vi64`, using the fewest bytes that hold it. Non-minimal
/// encodings are legal on the wire (§1.4.1) but there is no reason to write one.
pub fn put_vi64(out: &mut Vec<u8>, v: u64) {
    // Each of the first eight lengths carries 7 usable bits per byte; the ninth
    // carries a whole u64.
    let len = (1..=8usize)
        .find(|l| v < (1u64 << (7 * l)))
        .unwrap_or(9usize);
    if len == 9 {
        out.push(0xff);
        out.extend_from_slice(&v.to_be_bytes());
        return;
    }
    // `len - 1` leading ones, then the terminating zero, then the value's high
    // bits in whatever is left of the first byte.
    let prefix = ((((1u16 << (len - 1)) - 1) << (9 - len)) & 0xff) as u8;
    out.push(prefix | (v >> (8 * (len - 1))) as u8);
    out.extend_from_slice(&v.to_be_bytes()[9 - len..]);
}

/// Append a `vi64`-length-prefixed byte string.
pub fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    put_vi64(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

/// Append a bounded UTF-8 string (a Reason Phrase §1.4.4, or a session URI).
pub fn put_string(out: &mut Vec<u8>, s: &str, max: usize) -> Result<(), MoqtError> {
    if s.len() > max {
        return Err(MoqtError::Malformed);
    }
    put_bytes(out, s.as_bytes());
    Ok(())
}

/// A cursor over draft-18 bytes.
pub fn reader(buf: &[u8]) -> Reader<'_> {
    Reader::new_vi64(buf)
}

// ------------------------------------------------------ namespaces + names

/// Encode a Track Namespace (§2.4.1). Draft-18 allows 0 to 32 fields for every
/// namespace, so a prefix and a full namespace share one form.
pub fn encode_namespace(ns: &TrackNamespace, out: &mut Vec<u8>) -> Result<(), MoqtError> {
    if ns.0.len() > MAX_NAMESPACE_FIELDS || ns.byte_len() > MAX_FULL_TRACK_NAME_LEN {
        return Err(MoqtError::Malformed);
    }
    put_vi64(out, ns.0.len() as u64);
    for field in &ns.0 {
        if field.is_empty() {
            return Err(MoqtError::Malformed);
        }
        put_bytes(out, field);
    }
    Ok(())
}

pub fn decode_namespace(r: &mut Reader<'_>) -> Result<TrackNamespace, MoqtError> {
    let count = r.varint_usize()?;
    if count > MAX_NAMESPACE_FIELDS {
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
    Ok(TrackNamespace(fields))
}

pub fn encode_track_name(name: &TrackName, out: &mut Vec<u8>) -> Result<(), MoqtError> {
    if name.0.len() > MAX_FULL_TRACK_NAME_LEN {
        return Err(MoqtError::Malformed);
    }
    put_bytes(out, &name.0);
    Ok(())
}

pub fn decode_track_name(r: &mut Reader<'_>) -> Result<TrackName, MoqtError> {
    Ok(TrackName(r.length_bytes(MAX_FULL_TRACK_NAME_LEN)?.to_vec()))
}

/// A namespace plus a track name, bounded together (§2.4.1).
pub fn decode_full_track_name(
    r: &mut Reader<'_>,
) -> Result<(TrackNamespace, TrackName), MoqtError> {
    let namespace = decode_namespace(r)?;
    let name = decode_track_name(r)?;
    validate_full_track_name(&namespace, &name)?;
    Ok((namespace, name))
}

// ---------------------------------------------------------- location

pub fn encode_location(loc: &Location, out: &mut Vec<u8>) {
    put_vi64(out, loc.group_id);
    put_vi64(out, loc.object_id);
}

pub fn decode_location(r: &mut Reader<'_>) -> Result<Location, MoqtError> {
    Ok(Location {
        group_id: r.varint()?,
        object_id: r.varint()?,
    })
}

// ------------------------------------------------------- key-value pairs

/// Encode a bare Key-Value-Pair sequence (§1.4.3): no count, no length. The
/// caller bounds it, which is what Track Properties (the message tail) and
/// Setup Options (the whole payload) do.
pub fn encode_kvps(params: &Params, out: &mut Vec<u8>) -> Result<(), MoqtError> {
    // The wire delta must never go backwards, so encode in key order regardless
    // of insertion order.
    let mut sorted: Vec<&(u64, ParamValue)> = params.0.iter().collect();
    sorted.sort_by_key(|(k, _)| *k);
    let mut prev = 0u64;
    for (key, value) in sorted {
        let delta = key.checked_sub(prev).ok_or(MoqtError::Malformed)?;
        match value {
            ParamValue::Int(_) if key % 2 != 0 => return Err(MoqtError::Malformed),
            ParamValue::Bytes(_) if key % 2 == 0 => return Err(MoqtError::Malformed),
            _ => {}
        }
        put_vi64(out, delta);
        match value {
            ParamValue::Int(v) => put_vi64(out, *v),
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

/// Encode Object Properties (§11.2.1.2): a byte length, then the pairs.
pub fn encode_kvps_length_prefixed(params: &Params, out: &mut Vec<u8>) -> Result<(), MoqtError> {
    let mut body = Vec::new();
    encode_kvps(params, &mut body)?;
    put_bytes(out, &body);
    Ok(())
}

pub fn decode_kvps_length_prefixed(r: &mut Reader<'_>) -> Result<Params, MoqtError> {
    let body = r.length_bytes(MAX_KVP_BYTES_LEN)?;
    Params::decode_to_end(&mut reader(body))
}

/// Setup Option types (§15.4). This namespace is constant across MOQT versions.
pub mod setup_option {
    pub const PATH: u64 = 0x01;
    pub const AUTHORIZATION_TOKEN: u64 = 0x03;
    pub const MAX_AUTH_TOKEN_CACHE_SIZE: u64 = 0x04;
    pub const AUTHORITY: u64 = 0x05;
    pub const MOQT_IMPLEMENTATION: u64 = 0x07;
}

/// Property types (§15.8), carried as Key-Value-Pairs on tracks and objects.
pub mod property {
    pub const OBJECT_DELIVERY_TIMEOUT: u64 = 0x02;
    pub const MAX_CACHE_DURATION: u64 = 0x04;
    pub const SUBGROUP_DELIVERY_TIMEOUT: u64 = 0x06;
    pub const IMMUTABLE_PROPERTIES: u64 = 0x0b;
    pub const DEFAULT_PUBLISHER_PRIORITY: u64 = 0x0e;
    pub const DEFAULT_PUBLISHER_GROUP_ORDER: u64 = 0x22;
    pub const DYNAMIC_GROUPS: u64 = 0x30;
    pub const PRIOR_GROUP_ID_GAP: u64 = 0x3c;
    pub const PRIOR_OBJECT_ID_GAP: u64 = 0x3e;
}

// --------------------------------------------------- message parameters

/// Message Parameter types (§15.7).
pub mod param {
    pub const OBJECT_DELIVERY_TIMEOUT: u64 = 0x02;
    pub const AUTHORIZATION_TOKEN: u64 = 0x03;
    pub const RENDEZVOUS_TIMEOUT: u64 = 0x04;
    pub const SUBGROUP_DELIVERY_TIMEOUT: u64 = 0x06;
    pub const EXPIRES: u64 = 0x08;
    pub const LARGEST_OBJECT: u64 = 0x09;
    pub const FILL_TIMEOUT: u64 = 0x0a;
    pub const FORWARD: u64 = 0x10;
    pub const SUBSCRIBER_PRIORITY: u64 = 0x20;
    pub const SUBSCRIPTION_FILTER: u64 = 0x21;
    pub const GROUP_ORDER: u64 = 0x22;
    pub const NEW_GROUP_REQUEST: u64 = 0x32;
    pub const TRACK_NAMESPACE_PREFIX: u64 = 0x34;
}

/// One control-message parameter value. The form is fixed by the parameter
/// type, not by the type's parity: §10.2 defines uint8, varint, Location and
/// length-prefixed values, and TRACK_NAMESPACE_PREFIX adds the Track Namespace
/// form (§10.2.14).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MsgParam {
    Uint8(u8),
    Varint(u64),
    Location(Location),
    Bytes(Vec<u8>),
    Namespace(TrackNamespace),
}

impl MsgParam {
    /// The value form this parameter type is defined with, or `None` for a type
    /// this version of MOQT does not define.
    fn form_of(key: u64) -> Option<Form> {
        use param as p;
        Some(match key {
            // §10.2.3 / §10.2.4 / §10.2.10 / §10.2.13 name these varints;
            // RENDEZVOUS_TIMEOUT and FILL_TIMEOUT state a duration in
            // milliseconds without naming a form, which only a varint fits.
            p::OBJECT_DELIVERY_TIMEOUT
            | p::RENDEZVOUS_TIMEOUT
            | p::SUBGROUP_DELIVERY_TIMEOUT
            | p::EXPIRES
            | p::FILL_TIMEOUT
            | p::NEW_GROUP_REQUEST => Form::Varint,
            p::FORWARD | p::SUBSCRIBER_PRIORITY | p::GROUP_ORDER => Form::Uint8,
            p::LARGEST_OBJECT => Form::Location,
            p::AUTHORIZATION_TOKEN | p::SUBSCRIPTION_FILTER => Form::Bytes,
            p::TRACK_NAMESPACE_PREFIX => Form::Namespace,
            _ => return None,
        })
    }

    fn form(&self) -> Form {
        match self {
            Self::Uint8(_) => Form::Uint8,
            Self::Varint(_) => Form::Varint,
            Self::Location(_) => Form::Location,
            Self::Bytes(_) => Form::Bytes,
            Self::Namespace(_) => Form::Namespace,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Form {
    Uint8,
    Varint,
    Location,
    Bytes,
    Namespace,
}

/// An ordered control-message parameter list. Types are absolute here; the
/// delta coding is applied only on the wire.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MessageParams(pub Vec<(u64, MsgParam)>);

impl MessageParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Set a parameter, replacing any previous value for the same type. A value
    /// whose form does not match the type's definition is refused rather than
    /// written where the peer reads something else.
    pub fn set(&mut self, key: u64, value: MsgParam) -> Result<(), MoqtError> {
        if MsgParam::form_of(key) != Some(value.form()) {
            return Err(MoqtError::Malformed);
        }
        match self.0.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 = value,
            None => self.0.push((key, value)),
        }
        Ok(())
    }

    pub fn get(&self, key: u64) -> Option<&MsgParam> {
        self.0.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
    }

    pub fn get_varint(&self, key: u64) -> Option<u64> {
        match self.get(key)? {
            MsgParam::Varint(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_uint8(&self, key: u64) -> Option<u8> {
        match self.get(key)? {
            MsgParam::Uint8(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_bytes(&self, key: u64) -> Option<&[u8]> {
        match self.get(key)? {
            MsgParam::Bytes(v) => Some(v),
            _ => None,
        }
    }

    pub fn get_location(&self, key: u64) -> Option<Location> {
        match self.get(key)? {
            MsgParam::Location(v) => Some(*v),
            _ => None,
        }
    }

    /// Count, then the delta-typed parameters in ascending type order.
    pub fn encode(&self, out: &mut Vec<u8>) -> Result<(), MoqtError> {
        let mut sorted: Vec<&(u64, MsgParam)> = self.0.iter().collect();
        sorted.sort_by_key(|(k, _)| *k);
        put_vi64(out, sorted.len() as u64);
        let mut prev = 0u64;
        for (key, value) in sorted {
            if MsgParam::form_of(*key) != Some(value.form()) {
                return Err(MoqtError::Malformed);
            }
            let delta = key.checked_sub(prev).ok_or(MoqtError::Malformed)?;
            put_vi64(out, delta);
            match value {
                MsgParam::Uint8(v) => out.push(*v),
                MsgParam::Varint(v) => put_vi64(out, *v),
                MsgParam::Location(loc) => encode_location(loc, out),
                MsgParam::Bytes(b) => {
                    if b.len() > MAX_KVP_BYTES_LEN {
                        return Err(MoqtError::Malformed);
                    }
                    put_bytes(out, b);
                }
                MsgParam::Namespace(ns) => encode_namespace(ns, out)?,
            }
            prev = *key;
        }
        Ok(())
    }

    pub fn decode(r: &mut Reader<'_>) -> Result<Self, MoqtError> {
        let count = r.varint()?;
        let mut params: Vec<(u64, MsgParam)> = Vec::new();
        let mut prev = 0u64;
        for _ in 0..count {
            // No reserve from `count`: each iteration needs at least two bytes,
            // so a bogus count runs out of buffer instead of allocating on it.
            let delta = r.varint()?;
            let key = prev.checked_add(delta).ok_or(MoqtError::Malformed)?;
            // §10.2: an unknown parameter cannot be skipped, because nothing
            // says how long its value is. It is a session error, so the parse
            // stops here rather than resynchronizing.
            let value = match MsgParam::form_of(key).ok_or(MoqtError::Malformed)? {
                Form::Uint8 => MsgParam::Uint8(r.u8()?),
                Form::Varint => MsgParam::Varint(r.varint()?),
                Form::Location => MsgParam::Location(decode_location(r)?),
                Form::Bytes => MsgParam::Bytes(r.length_bytes(MAX_KVP_BYTES_LEN)?.to_vec()),
                Form::Namespace => MsgParam::Namespace(decode_namespace(r)?),
            };
            // §10.2: a repeated type is a violation unless the definition
            // allows it, and none of the ones the elements use do.
            if params.iter().any(|(k, _)| *k == key) {
                return Err(MoqtError::Malformed);
            }
            params.push((key, value));
            prev = key;
        }
        Ok(Self(params))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn round_trip_vi64(v: u64) {
        let mut out = Vec::new();
        put_vi64(&mut out, v);
        let mut r = reader(&out);
        assert_eq!(r.varint().expect("decode"), v, "vi64 {v}");
        assert!(r.is_empty(), "vi64 {v} consumed exactly its bytes");
    }

    /// The example encodings in draft-18 Table 2, which is the only place the
    /// leading-ones form is pinned down byte for byte.
    #[test]
    fn vi64_matches_the_draft_example_encodings() {
        for (bytes, value) in [
            (vec![0x25u8], 37u64),
            (vec![0xbb, 0xbd], 15_293),
            (vec![0xed, 0x7f, 0x3e, 0x7d], 226_442_877),
            (vec![0xfa, 0xa1, 0xa0, 0xe4, 0x03, 0xd8], 2_893_212_287_960),
            (
                vec![0xfc, 0x89, 0x98, 0xab, 0xc6, 0x6b, 0xc0],
                151_288_809_941_952,
            ),
            (
                vec![0xfe, 0xfa, 0x31, 0x8f, 0xa8, 0xe3, 0xca, 0x11],
                70_423_237_261_249_041,
            ),
            (
                vec![0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
                u64::MAX,
            ),
        ] {
            assert_eq!(
                reader(&bytes).varint(),
                Ok(value),
                "decoding {bytes:02x?} as {value}"
            );
            let mut out = Vec::new();
            put_vi64(&mut out, value);
            assert_eq!(out, bytes, "encoding {value} minimally");
        }
    }

    #[test]
    fn vi64_round_trips_every_length_boundary() {
        for len in 1..=8u32 {
            let max = (1u64 << (7 * len)) - 1;
            round_trip_vi64(max);
            round_trip_vi64(max.wrapping_add(1));
        }
        for v in [0, 1, 127, 128, u64::MAX - 1, u64::MAX] {
            round_trip_vi64(v);
        }
    }

    /// A non-minimal encoding is legal (§1.4.1) and must decode to the same
    /// value the minimal one does.
    #[test]
    fn vi64_accepts_non_minimal_encodings() {
        assert_eq!(reader(&[0x80, 0x25]).varint(), Ok(37));
        assert_eq!(reader(&[0xc0, 0x00, 0x25]).varint(), Ok(37));
        assert_eq!(
            reader(&[0xff, 0, 0, 0, 0, 0, 0, 0, 0x25]).varint(),
            Ok(37),
            "the nine-byte form of 37"
        );
    }

    #[test]
    fn a_truncated_vi64_is_incomplete_not_a_panic() {
        assert_eq!(reader(&[]).varint(), Err(MoqtError::Incomplete));
        for len in 2..=9usize {
            let mut out = Vec::new();
            // The largest value of this length, then every prefix of it.
            put_vi64(&mut out, (1u64 << (7 * (len - 1))) + 1);
            for cut in 0..out.len() {
                assert_eq!(
                    reader(&out[..cut]).varint(),
                    Err(MoqtError::Incomplete),
                    "a {cut}-byte prefix of a {len}-byte vi64"
                );
            }
        }
    }

    #[test]
    fn a_namespace_round_trips_and_may_be_empty() {
        let ns = TrackNamespace::from_path("test/ns");
        let mut out = Vec::new();
        encode_namespace(&ns, &mut out).expect("encode");
        assert_eq!(
            out,
            vec![0x02, 0x04, b't', b'e', b's', b't', 0x02, b'n', b's']
        );
        assert_eq!(decode_namespace(&mut reader(&out)).expect("decode"), ns);

        // Draft-18 allows 0 to 32 fields for every namespace, prefix or not.
        let empty = TrackNamespace::default();
        let mut out = Vec::new();
        encode_namespace(&empty, &mut out).expect("encode");
        assert_eq!(out, vec![0x00]);
        assert_eq!(decode_namespace(&mut reader(&out)).expect("decode"), empty);
    }

    #[test]
    fn a_namespace_rejects_absurd_counts_empty_fields_and_over_long_names() {
        // 33 fields is over the draft's 32.
        let mut out = Vec::new();
        put_vi64(&mut out, 33);
        for _ in 0..33 {
            put_bytes(&mut out, b"x");
        }
        assert_eq!(
            decode_namespace(&mut reader(&out)),
            Err(MoqtError::Malformed)
        );

        // An empty field, and a field length that overruns the buffer.
        assert_eq!(
            decode_namespace(&mut reader(&[0x01, 0x00])),
            Err(MoqtError::Malformed)
        );
        assert_eq!(
            decode_namespace(&mut reader(&[0x01, 0x08, b'a'])),
            Err(MoqtError::Incomplete)
        );

        // A count of u64::MAX must fail on the buffer, not on an allocation.
        let mut out = Vec::new();
        put_vi64(&mut out, u64::MAX);
        assert_eq!(
            decode_namespace(&mut reader(&out)),
            Err(MoqtError::Malformed)
        );

        // Namespace plus track name over 4096 bytes.
        let big = TrackNamespace(vec![vec![b'a'; MAX_FULL_TRACK_NAME_LEN]]);
        let mut out = Vec::new();
        encode_namespace(&big, &mut out).expect("encode");
        let mut r = reader(&out);
        let decoded = decode_namespace(&mut r).expect("decode");
        assert_eq!(
            validate_full_track_name(&decoded, &TrackName::new("x")),
            Err(MoqtError::Malformed)
        );
    }

    #[test]
    fn kvps_delta_code_by_type_and_parity() {
        let mut params = Params::new();
        // Insert out of order: the encoder must sort so no delta goes backwards.
        params.set_int(2, 100);
        params.set_bytes(1, b"tok".to_vec());
        let mut out = Vec::new();
        encode_kvps_length_prefixed(&params, &mut out).expect("encode");
        assert_eq!(
            out,
            vec![
                0x07, // 7 bytes of pairs
                0x01, 0x03, b't', b'o', b'k', // type 1, length 3
                0x01, 0x64, // delta 1 -> type 2, value 100
            ]
        );
        // Decode yields wire order (ascending type), not insertion order.
        let mut expected = params.clone();
        expected.0.sort_by_key(|(k, _)| *k);
        assert_eq!(
            decode_kvps_length_prefixed(&mut reader(&out)).expect("decode"),
            expected
        );

        // A parity mismatch is refused on encode.
        let mut bad = Params::new();
        bad.set_int(1, 0);
        assert_eq!(
            encode_kvps(&bad, &mut Vec::new()),
            Err(MoqtError::Malformed)
        );

        // A running type that overflows u64 is a violation.
        let mut out = Vec::new();
        put_vi64(&mut out, u64::MAX);
        put_vi64(&mut out, 0);
        put_vi64(&mut out, 1);
        assert_eq!(
            Params::decode_to_end(&mut reader(&out)),
            Err(MoqtError::Malformed)
        );
    }

    #[test]
    fn message_params_encode_each_type_in_its_own_form() {
        let mut params = MessageParams::new();
        params
            .set(param::SUBSCRIBER_PRIORITY, MsgParam::Uint8(200))
            .expect("uint8");
        params
            .set(param::EXPIRES, MsgParam::Varint(5000))
            .expect("varint");
        params
            .set(
                param::LARGEST_OBJECT,
                MsgParam::Location(Location {
                    group_id: 3,
                    object_id: 9,
                }),
            )
            .expect("location");
        params
            .set(param::AUTHORIZATION_TOKEN, MsgParam::Bytes(b"tok".to_vec()))
            .expect("bytes");
        params
            .set(
                param::TRACK_NAMESPACE_PREFIX,
                MsgParam::Namespace(TrackNamespace::from_path("live")),
            )
            .expect("namespace");

        let mut out = Vec::new();
        params.encode(&mut out).expect("encode");
        assert_eq!(
            out,
            vec![
                0x05, // five parameters
                0x03, 0x03, b't', b'o', b'k', // 0x03 AUTHORIZATION_TOKEN
                0x05, 0x93, 0x88, // delta 5 -> 0x08 EXPIRES, varint 5000
                0x01, 0x03, 0x09, // delta 1 -> 0x09 LARGEST_OBJECT
                0x17, 0xc8, // delta 0x17 -> 0x20 SUBSCRIBER_PRIORITY, uint8
                0x14, 0x01, 0x04, b'l', b'i', b'v', b'e', // 0x34 prefix
            ]
        );
        // Decode yields wire order (ascending type), not insertion order.
        let mut expected = params.clone();
        expected.0.sort_by_key(|(k, _)| *k);
        assert_eq!(
            MessageParams::decode(&mut reader(&out)).expect("decode"),
            expected
        );
    }

    #[test]
    fn a_message_param_form_that_does_not_match_its_type_is_refused() {
        let mut params = MessageParams::new();
        assert_eq!(
            params.set(param::EXPIRES, MsgParam::Uint8(1)),
            Err(MoqtError::Malformed),
            "EXPIRES is a varint"
        );
        assert_eq!(
            params.set(0x77, MsgParam::Varint(1)),
            Err(MoqtError::Malformed),
            "0x77 is not a defined parameter type"
        );
        // A hand-built list with the wrong form is refused on encode too.
        let wrong = MessageParams(vec![(param::FORWARD, MsgParam::Varint(1))]);
        assert_eq!(wrong.encode(&mut Vec::new()), Err(MoqtError::Malformed));
    }

    #[test]
    fn message_params_refuse_unknown_types_duplicates_and_bogus_counts() {
        // An unknown type cannot be skipped: §10.2 makes it a session error.
        assert_eq!(
            MessageParams::decode(&mut reader(&[0x01, 0x77, 0x00])),
            Err(MoqtError::Malformed)
        );
        // The same type twice.
        assert_eq!(
            MessageParams::decode(&mut reader(&[0x02, 0x10, 0x01, 0x00, 0x01])),
            Err(MoqtError::Malformed)
        );
        // A count of u64::MAX with nothing behind it runs out of buffer rather
        // than allocating on the count.
        let mut out = Vec::new();
        put_vi64(&mut out, u64::MAX);
        assert_eq!(
            MessageParams::decode(&mut reader(&out)),
            Err(MoqtError::Incomplete)
        );
        // A delta that overflows the running type.
        let mut out = Vec::new();
        put_vi64(&mut out, 2);
        put_vi64(&mut out, u64::MAX);
        assert_eq!(
            MessageParams::decode(&mut reader(&out)),
            Err(MoqtError::Malformed)
        );
    }

    #[test]
    fn a_bounded_string_rejects_over_long_and_non_utf8() {
        let mut out = Vec::new();
        put_string(&mut out, "why", MAX_REASON_PHRASE_LEN).expect("encode");
        assert_eq!(
            reader(&out).string(MAX_REASON_PHRASE_LEN).as_deref(),
            Ok("why")
        );

        let mut too_long = Vec::new();
        put_vi64(&mut too_long, MAX_REASON_PHRASE_LEN as u64 + 1);
        too_long.extend_from_slice(&[b'x'; 8]);
        assert_eq!(
            reader(&too_long).string(MAX_REASON_PHRASE_LEN),
            Err(MoqtError::Malformed)
        );

        let mut bad_utf8 = Vec::new();
        put_bytes(&mut bad_utf8, &[0xff, 0xfe]);
        assert_eq!(
            reader(&bad_utf8).string(MAX_REASON_PHRASE_LEN),
            Err(MoqtError::Malformed)
        );
    }
}
