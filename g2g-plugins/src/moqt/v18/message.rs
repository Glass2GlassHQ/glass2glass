//! Control-stream and request-stream messages for IETF MoQ Transport draft-18.
//!
//! Framing is unchanged from draft-16 (§10):
//!
//! ```text
//! Message Type (vi64), Message Length (16), Message Payload (..)
//! ```
//!
//! What changed is everything above it. Draft-18 splits the control plane in
//! two: a pair of unidirectional streams, each opened with a single [`SETUP`]
//! (type `0x2F00`, replacing the CLIENT_SETUP / SERVER_SETUP pair), and one
//! bidirectional stream per request. A response therefore carries no request id
//! at all: the stream identifies the request. Cancellation is a stream reset,
//! so UNSUBSCRIBE, FETCH_CANCEL, PUBLISH_NAMESPACE_CANCEL, MAX_REQUEST_ID and
//! REQUESTS_BLOCKED no longer exist.
//!
//! [`SETUP`]: ControlMessage::Setup
//!
//! Only the request messages ([`msg_type::SUBSCRIBE`] and friends) carry a
//! Request ID, and only the seven "First" messages of Table 5 may open a
//! bidirectional stream: [`ControlMessage::opens_request_stream`] says which.
//!
//! Type numbers and field lists are Table 5 and §10.3 to §10.20 of the draft.

use alloc::string::String;
use alloc::vec::Vec;

use super::coding::{
    decode_full_track_name, decode_location, decode_namespace, decode_track_name, encode_kvps,
    encode_location, encode_namespace, encode_track_name, put_string, put_vi64, reader, Location,
    MessageParams, MoqtError, Params, Reader, TrackName, TrackNamespace, MAX_REASON_PHRASE_LEN,
    MAX_SESSION_URI_LEN,
};

/// Message type numbers (draft-18 Table 5).
pub mod msg_type {
    pub const REQUEST_UPDATE: u64 = 0x2;
    pub const SUBSCRIBE: u64 = 0x3;
    pub const SUBSCRIBE_OK: u64 = 0x4;
    pub const REQUEST_ERROR: u64 = 0x5;
    pub const PUBLISH_NAMESPACE: u64 = 0x6;
    pub const REQUEST_OK: u64 = 0x7;
    pub const NAMESPACE: u64 = 0x8;
    pub const PUBLISH_DONE: u64 = 0xb;
    pub const TRACK_STATUS: u64 = 0xd;
    pub const NAMESPACE_DONE: u64 = 0xe;
    pub const PUBLISH_BLOCKED: u64 = 0xf;
    pub const GOAWAY: u64 = 0x10;
    pub const FETCH: u64 = 0x16;
    pub const FETCH_OK: u64 = 0x18;
    pub const PUBLISH: u64 = 0x1d;
    /// Table 5 gives PUBLISH_OK its own code point while §10.5 defines
    /// PUBLISH_OK as shorthand for a REQUEST_OK answering a PUBLISH. Both are
    /// accepted; the payload is REQUEST_OK's either way (see the module note on
    /// [`ControlMessage::PublishOk`]).
    pub const PUBLISH_OK: u64 = 0x1e;
    pub const SUBSCRIBE_NAMESPACE: u64 = 0x50;
    pub const SUBSCRIBE_TRACKS: u64 = 0x51;
    pub const SETUP: u64 = 0x2f00;
}

/// Session termination error codes (§3.5 / §15.10.1).
pub mod session_error_code {
    pub const NO_ERROR: u32 = 0x0;
    pub const INTERNAL_ERROR: u32 = 0x1;
    pub const UNAUTHORIZED: u32 = 0x2;
    pub const PROTOCOL_VIOLATION: u32 = 0x3;
    pub const INVALID_REQUEST_ID: u32 = 0x4;
    pub const DUPLICATE_TRACK_ALIAS: u32 = 0x5;
    pub const KEY_VALUE_FORMATTING_ERROR: u32 = 0x6;
    pub const GOAWAY_TIMEOUT: u32 = 0x10;
    pub const CONTROL_MESSAGE_TIMEOUT: u32 = 0x11;
    pub const DATA_STREAM_TIMEOUT: u32 = 0x12;
    pub const VERSION_NEGOTIATION_FAILED: u32 = 0x15;
}

/// REQUEST_ERROR codes (§15.10.2).
pub mod request_error_code {
    pub const INTERNAL_ERROR: u64 = 0x0;
    pub const UNAUTHORIZED: u64 = 0x1;
    pub const TIMEOUT: u64 = 0x2;
    pub const NOT_SUPPORTED: u64 = 0x3;
    pub const MALFORMED_AUTH_TOKEN: u64 = 0x4;
    pub const EXPIRED_AUTH_TOKEN: u64 = 0x5;
    pub const GOING_AWAY: u64 = 0x6;
    pub const EXCESSIVE_LOAD: u64 = 0x9;
    pub const DOES_NOT_EXIST: u64 = 0x10;
    pub const INVALID_RANGE: u64 = 0x11;
    pub const MALFORMED_TRACK: u64 = 0x12;
    pub const DUPLICATE_SUBSCRIPTION: u64 = 0x19;
    pub const UNINTERESTED: u64 = 0x20;
    pub const PREFIX_OVERLAP: u64 = 0x30;
    pub const NAMESPACE_TOO_LARGE: u64 = 0x31;
    pub const INVALID_JOINING_REQUEST_ID: u64 = 0x32;
    pub const UNSUPPORTED_EXTENSION: u64 = 0x33;
    /// The only code that carries a [`Redirect`] in REQUEST_ERROR.
    pub const REDIRECT: u64 = 0x34;
}

/// PUBLISH_DONE status codes (§15.10.3).
pub mod publish_done_code {
    pub const INTERNAL_ERROR: u64 = 0x0;
    pub const UNAUTHORIZED: u64 = 0x1;
    pub const TRACK_ENDED: u64 = 0x2;
    pub const SUBSCRIPTION_ENDED: u64 = 0x3;
    pub const GOING_AWAY: u64 = 0x4;
    pub const TOO_FAR_BEHIND: u64 = 0x5;
    pub const EXPIRED: u64 = 0x6;
    pub const UPDATE_FAILED: u64 = 0x8;
    pub const EXCESSIVE_LOAD: u64 = 0x9;
    pub const MALFORMED_TRACK: u64 = 0x12;
}

/// Stream reset error codes (§3.3.3 / §15.10.4). Draft-18 cancels a request by
/// resetting its stream, so these replace draft-16's cancel messages.
pub mod stream_error_code {
    pub const INTERNAL_ERROR: u32 = 0x0;
    pub const CANCELLED: u32 = 0x1;
    pub const DELIVERY_TIMEOUT: u32 = 0x2;
    pub const SESSION_CLOSED: u32 = 0x3;
    pub const GOING_AWAY: u32 = 0x4;
    pub const TOO_FAR_BEHIND: u32 = 0x5;
    pub const UNKNOWN_OBJECT_STATUS: u32 = 0x6;
    pub const EXPIRED_AUTH_TOKEN: u32 = 0x7;
    pub const EXCESSIVE_LOAD: u32 = 0x9;
    pub const MALFORMED_TRACK: u32 = 0x12;
}

/// FETCH variants (§10.12, Table 6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FetchType {
    Standalone,
    RelativeJoining,
    AbsoluteJoining,
}

impl FetchType {
    fn code(self) -> u64 {
        match self {
            Self::Standalone => 0x1,
            Self::RelativeJoining => 0x2,
            Self::AbsoluteJoining => 0x3,
        }
    }

    fn from_code(code: u64) -> Result<Self, MoqtError> {
        Ok(match code {
            0x1 => Self::Standalone,
            0x2 => Self::RelativeJoining,
            0x3 => Self::AbsoluteJoining,
            _ => return Err(MoqtError::Malformed),
        })
    }
}

/// The body of a standalone FETCH (§10.12.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StandaloneFetch {
    pub namespace: TrackNamespace,
    pub track_name: TrackName,
    pub start: Location,
    pub end: Location,
}

/// The body of a joining FETCH (§10.12.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JoiningFetch {
    pub joining_request_id: u64,
    pub joining_start: u64,
}

/// Where a peer wants a request retried (§10.6.1), carried by a REQUEST_ERROR
/// with code [`request_error_code::REDIRECT`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Redirect {
    /// Empty means "the current session's URI".
    pub connect_uri: String,
    /// An empty namespace and name together mean "the original request's".
    pub namespace: TrackNamespace,
    pub track_name: TrackName,
}

impl Redirect {
    fn encode(&self, out: &mut Vec<u8>) -> Result<(), MoqtError> {
        put_string(out, &self.connect_uri, MAX_SESSION_URI_LEN)?;
        encode_namespace(&self.namespace, out)?;
        encode_track_name(&self.track_name, out)
    }

    fn decode(r: &mut Reader<'_>) -> Result<Self, MoqtError> {
        // The draft does not bound the Connect URI beyond the 2^16-1 a control
        // message holds, so that is the bound rather than the tighter session
        // URI limit GOAWAY states.
        let connect_uri = r.string(u16::MAX as usize)?;
        Ok(Self {
            connect_uri,
            namespace: decode_namespace(r)?,
            track_name: decode_track_name(r)?,
        })
    }
}

/// Which objects a subscription asks for (§5.1.2), carried as the
/// [`super::coding::param::SUBSCRIPTION_FILTER`] parameter value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubscriptionFilter {
    /// From the group after the largest object's.
    NextGroupStart,
    /// From just past the largest object.
    LargestObject,
    /// From an explicit location, open ended.
    AbsoluteStart { start: Location },
    /// From an explicit location to `start.group_id + end_group_delta`.
    AbsoluteRange {
        start: Location,
        end_group_delta: u64,
    },
}

impl SubscriptionFilter {
    fn code(self) -> u64 {
        match self {
            Self::NextGroupStart => 0x1,
            Self::LargestObject => 0x2,
            Self::AbsoluteStart { .. } => 0x3,
            Self::AbsoluteRange { .. } => 0x4,
        }
    }

    /// The last group this filter passes, or `None` when it is open ended. A
    /// range whose end group would leave `u64` is a protocol violation (§5.1.2).
    pub fn end_group(self) -> Option<Result<u64, MoqtError>> {
        match self {
            Self::AbsoluteRange {
                start,
                end_group_delta,
            } => Some(
                start
                    .group_id
                    .checked_add(end_group_delta)
                    .ok_or(MoqtError::Malformed),
            ),
            _ => None,
        }
    }

    pub fn to_bytes(self) -> Result<Vec<u8>, MoqtError> {
        if let Some(end) = self.end_group() {
            end?;
        }
        let mut out = Vec::new();
        put_vi64(&mut out, self.code());
        match self {
            Self::NextGroupStart | Self::LargestObject => {}
            Self::AbsoluteStart { start } => encode_location(&start, &mut out),
            Self::AbsoluteRange {
                start,
                end_group_delta,
            } => {
                encode_location(&start, &mut out);
                put_vi64(&mut out, end_group_delta);
            }
        }
        Ok(out)
    }

    /// Decode a filter that must consume `bytes` exactly: a trailing byte means
    /// the peer and we disagree on the filter type's shape.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MoqtError> {
        let mut r = reader(bytes);
        let filter = match r.varint()? {
            0x1 => Self::NextGroupStart,
            0x2 => Self::LargestObject,
            0x3 => Self::AbsoluteStart {
                start: decode_location(&mut r)?,
            },
            0x4 => Self::AbsoluteRange {
                start: decode_location(&mut r)?,
                end_group_delta: r.varint()?,
            },
            _ => return Err(MoqtError::Malformed),
        };
        if !r.is_empty() {
            return Err(MoqtError::Malformed);
        }
        if let Some(end) = filter.end_group() {
            end?;
        }
        Ok(filter)
    }
}

/// Every draft-18 control or request message (Table 5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlMessage {
    /// The single message that opens each peer's control stream (§10.3). The
    /// options span the whole payload and unknown ones are ignored, so this
    /// decodes them all rather than refusing what it does not know.
    Setup {
        options: Params,
    },
    GoAway {
        /// Empty means "reuse the current URI"; a client MUST send it empty.
        uri: String,
        timeout_ms: u64,
        /// Present only on the control stream (§10.4). A request-stream GOAWAY
        /// omits it, and the payload length is what says which.
        request_id: Option<u64>,
    },
    Subscribe {
        id: u64,
        namespace: TrackNamespace,
        track_name: TrackName,
        params: MessageParams,
    },
    SubscribeOk {
        track_alias: u64,
        params: MessageParams,
        properties: Params,
    },
    Publish {
        id: u64,
        namespace: TrackNamespace,
        track_name: TrackName,
        track_alias: u64,
        params: MessageParams,
        properties: Params,
    },
    /// Table 5's `0x1E` code point. §10.5 makes PUBLISH_OK shorthand for a
    /// REQUEST_OK answering a PUBLISH, so the payload is REQUEST_OK's; the two
    /// stay separate variants because a peer may legitimately send either code
    /// point.
    PublishOk {
        params: MessageParams,
        properties: Params,
    },
    PublishDone {
        status_code: u64,
        stream_count: u64,
        reason: String,
    },
    Fetch {
        id: u64,
        fetch_type: FetchType,
        standalone: Option<StandaloneFetch>,
        joining: Option<JoiningFetch>,
        params: MessageParams,
    },
    FetchOk {
        end_of_track: bool,
        end: Location,
        params: MessageParams,
        properties: Params,
    },
    TrackStatus {
        id: u64,
        namespace: TrackNamespace,
        track_name: TrackName,
        params: MessageParams,
    },
    PublishNamespace {
        id: u64,
        namespace: TrackNamespace,
        params: MessageParams,
    },
    SubscribeNamespace {
        id: u64,
        prefix: TrackNamespace,
        params: MessageParams,
    },
    SubscribeTracks {
        id: u64,
        prefix: TrackNamespace,
        params: MessageParams,
    },
    Namespace {
        suffix: TrackNamespace,
    },
    NamespaceDone {
        suffix: TrackNamespace,
    },
    PublishBlocked {
        suffix: TrackNamespace,
        track_name: TrackName,
    },
    RequestUpdate {
        id: u64,
        params: MessageParams,
    },
    /// Answers PUBLISH, REQUEST_UPDATE, TRACK_STATUS, SUBSCRIBE_NAMESPACE,
    /// SUBSCRIBE_TRACKS and PUBLISH_NAMESPACE (§10.5). `properties` is only
    /// populated when it answers a TRACK_STATUS; which request a given
    /// REQUEST_OK answers is the stream's context, not the message's, so that
    /// rule is enforced by the session rather than here.
    RequestOk {
        params: MessageParams,
        properties: Params,
    },
    RequestError {
        error_code: u64,
        retry_interval: u64,
        reason: String,
        /// Present exactly when `error_code` is [`request_error_code::REDIRECT`].
        redirect: Option<Redirect>,
    },
}

impl ControlMessage {
    pub fn type_id(&self) -> u64 {
        use msg_type as t;
        match self {
            Self::Setup { .. } => t::SETUP,
            Self::GoAway { .. } => t::GOAWAY,
            Self::Subscribe { .. } => t::SUBSCRIBE,
            Self::SubscribeOk { .. } => t::SUBSCRIBE_OK,
            Self::Publish { .. } => t::PUBLISH,
            Self::PublishOk { .. } => t::PUBLISH_OK,
            Self::PublishDone { .. } => t::PUBLISH_DONE,
            Self::Fetch { .. } => t::FETCH,
            Self::FetchOk { .. } => t::FETCH_OK,
            Self::TrackStatus { .. } => t::TRACK_STATUS,
            Self::PublishNamespace { .. } => t::PUBLISH_NAMESPACE,
            Self::SubscribeNamespace { .. } => t::SUBSCRIBE_NAMESPACE,
            Self::SubscribeTracks { .. } => t::SUBSCRIBE_TRACKS,
            Self::Namespace { .. } => t::NAMESPACE,
            Self::NamespaceDone { .. } => t::NAMESPACE_DONE,
            Self::PublishBlocked { .. } => t::PUBLISH_BLOCKED,
            Self::RequestUpdate { .. } => t::REQUEST_UPDATE,
            Self::RequestOk { .. } => t::REQUEST_OK,
            Self::RequestError { .. } => t::REQUEST_ERROR,
        }
    }

    /// A human name for logging, matching the draft's message names.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Setup { .. } => "SETUP",
            Self::GoAway { .. } => "GOAWAY",
            Self::Subscribe { .. } => "SUBSCRIBE",
            Self::SubscribeOk { .. } => "SUBSCRIBE_OK",
            Self::Publish { .. } => "PUBLISH",
            Self::PublishOk { .. } => "PUBLISH_OK",
            Self::PublishDone { .. } => "PUBLISH_DONE",
            Self::Fetch { .. } => "FETCH",
            Self::FetchOk { .. } => "FETCH_OK",
            Self::TrackStatus { .. } => "TRACK_STATUS",
            Self::PublishNamespace { .. } => "PUBLISH_NAMESPACE",
            Self::SubscribeNamespace { .. } => "SUBSCRIBE_NAMESPACE",
            Self::SubscribeTracks { .. } => "SUBSCRIBE_TRACKS",
            Self::Namespace { .. } => "NAMESPACE",
            Self::NamespaceDone { .. } => "NAMESPACE_DONE",
            Self::PublishBlocked { .. } => "PUBLISH_BLOCKED",
            Self::RequestUpdate { .. } => "REQUEST_UPDATE",
            Self::RequestOk { .. } => "REQUEST_OK",
            Self::RequestError { .. } => "REQUEST_ERROR",
        }
    }

    /// The Request ID this message consumes (§10.1), or `None` for a response.
    pub fn request_id(&self) -> Option<u64> {
        match self {
            Self::Subscribe { id, .. }
            | Self::Publish { id, .. }
            | Self::Fetch { id, .. }
            | Self::TrackStatus { id, .. }
            | Self::PublishNamespace { id, .. }
            | Self::SubscribeNamespace { id, .. }
            | Self::SubscribeTracks { id, .. }
            | Self::RequestUpdate { id, .. } => Some(*id),
            _ => None,
        }
    }

    /// Whether this message is one of Table 5's seven "First" messages, the only
    /// ones that may open a bidirectional stream (§3.3). Anything else arriving
    /// first on a bidi stream is a PROTOCOL_VIOLATION.
    pub fn opens_request_stream(&self) -> bool {
        matches!(
            self,
            Self::TrackStatus { .. }
                | Self::Subscribe { .. }
                | Self::Publish { .. }
                | Self::Fetch { .. }
                | Self::PublishNamespace { .. }
                | Self::SubscribeNamespace { .. }
                | Self::SubscribeTracks { .. }
        )
    }

    /// Append the framed message (type, 16-bit length, payload).
    pub fn encode(&self, out: &mut Vec<u8>) -> Result<(), MoqtError> {
        let mut payload = Vec::new();
        self.encode_payload(&mut payload)?;
        if payload.len() > u16::MAX as usize {
            return Err(MoqtError::Malformed);
        }
        put_vi64(out, self.type_id());
        out.extend_from_slice(&(payload.len() as u16).to_be_bytes());
        out.extend_from_slice(&payload);
        Ok(())
    }

    /// Decode one framed message, returning it and the bytes it consumed.
    /// [`MoqtError::Incomplete`] means the frame is not fully buffered.
    pub fn decode(buf: &[u8]) -> Result<(Self, usize), MoqtError> {
        let mut r = reader(buf);
        let typ = r.varint()?;
        let len = r.u16()? as usize;
        let payload = r.bytes(len)?;
        let msg = Self::decode_payload(typ, payload)?;
        Ok((msg, r.position()))
    }

    /// Decode a payload already separated from its frame. It must be consumed
    /// exactly; leftover bytes are a protocol violation (§10).
    pub fn decode_payload(typ: u64, payload: &[u8]) -> Result<Self, MoqtError> {
        let mut r = reader(payload);
        let msg = Self::read(typ, &mut r)?;
        if !r.is_empty() {
            return Err(MoqtError::Malformed);
        }
        Ok(msg)
    }

    fn encode_payload(&self, out: &mut Vec<u8>) -> Result<(), MoqtError> {
        match self {
            Self::Setup { options } => encode_kvps(options, out)?,
            Self::GoAway {
                uri,
                timeout_ms,
                request_id,
            } => {
                put_string(out, uri, MAX_SESSION_URI_LEN)?;
                put_vi64(out, *timeout_ms);
                if let Some(id) = request_id {
                    put_vi64(out, *id);
                }
            }
            Self::Subscribe {
                id,
                namespace,
                track_name,
                params,
            }
            | Self::TrackStatus {
                id,
                namespace,
                track_name,
                params,
            } => {
                put_vi64(out, *id);
                encode_namespace(namespace, out)?;
                encode_track_name(track_name, out)?;
                params.encode(out)?;
            }
            Self::SubscribeOk {
                track_alias,
                params,
                properties,
            } => {
                put_vi64(out, *track_alias);
                params.encode(out)?;
                encode_kvps(properties, out)?;
            }
            Self::Publish {
                id,
                namespace,
                track_name,
                track_alias,
                params,
                properties,
            } => {
                put_vi64(out, *id);
                encode_namespace(namespace, out)?;
                encode_track_name(track_name, out)?;
                put_vi64(out, *track_alias);
                params.encode(out)?;
                encode_kvps(properties, out)?;
            }
            Self::RequestOk { params, properties } | Self::PublishOk { params, properties } => {
                params.encode(out)?;
                encode_kvps(properties, out)?;
            }
            Self::PublishDone {
                status_code,
                stream_count,
                reason,
            } => {
                put_vi64(out, *status_code);
                put_vi64(out, *stream_count);
                put_string(out, reason, MAX_REASON_PHRASE_LEN)?;
            }
            Self::Fetch {
                id,
                fetch_type,
                standalone,
                joining,
                params,
            } => {
                put_vi64(out, *id);
                put_vi64(out, fetch_type.code());
                match fetch_type {
                    FetchType::Standalone => {
                        let body = standalone.as_ref().ok_or(MoqtError::Malformed)?;
                        encode_namespace(&body.namespace, out)?;
                        encode_track_name(&body.track_name, out)?;
                        encode_location(&body.start, out);
                        encode_location(&body.end, out);
                    }
                    FetchType::RelativeJoining | FetchType::AbsoluteJoining => {
                        let body = joining.as_ref().ok_or(MoqtError::Malformed)?;
                        put_vi64(out, body.joining_request_id);
                        put_vi64(out, body.joining_start);
                    }
                }
                params.encode(out)?;
            }
            Self::FetchOk {
                end_of_track,
                end,
                params,
                properties,
            } => {
                out.push(u8::from(*end_of_track));
                encode_location(end, out);
                params.encode(out)?;
                encode_kvps(properties, out)?;
            }
            Self::PublishNamespace {
                id,
                namespace,
                params,
            } => {
                put_vi64(out, *id);
                encode_namespace(namespace, out)?;
                params.encode(out)?;
            }
            Self::SubscribeNamespace { id, prefix, params }
            | Self::SubscribeTracks { id, prefix, params } => {
                put_vi64(out, *id);
                encode_namespace(prefix, out)?;
                params.encode(out)?;
            }
            Self::Namespace { suffix } | Self::NamespaceDone { suffix } => {
                encode_namespace(suffix, out)?
            }
            Self::PublishBlocked { suffix, track_name } => {
                encode_namespace(suffix, out)?;
                encode_track_name(track_name, out)?;
            }
            Self::RequestUpdate { id, params } => {
                put_vi64(out, *id);
                params.encode(out)?;
            }
            Self::RequestError {
                error_code,
                retry_interval,
                reason,
                redirect,
            } => {
                put_vi64(out, *error_code);
                put_vi64(out, *retry_interval);
                put_string(out, reason, MAX_REASON_PHRASE_LEN)?;
                // §10.6.2 ties the Redirect's presence to the code, so a
                // mismatch would be unparseable at the peer.
                match (redirect, *error_code == request_error_code::REDIRECT) {
                    (Some(redirect), true) => redirect.encode(out)?,
                    (None, false) => {}
                    _ => return Err(MoqtError::Malformed),
                }
            }
        }
        Ok(())
    }

    fn read(typ: u64, r: &mut Reader<'_>) -> Result<Self, MoqtError> {
        use msg_type as t;
        Ok(match typ {
            t::SETUP => Self::Setup {
                options: Params::decode_to_end(r)?,
            },
            t::GOAWAY => Self::GoAway {
                uri: r.string(MAX_SESSION_URI_LEN)?,
                timeout_ms: r.varint()?,
                // The optional Request ID is present on a control stream and
                // absent on a request stream; the payload length is the only
                // thing that distinguishes them.
                request_id: if r.is_empty() {
                    None
                } else {
                    Some(r.varint()?)
                },
            },
            t::SUBSCRIBE => {
                let (id, namespace, track_name, params) = read_track_request(r)?;
                Self::Subscribe {
                    id,
                    namespace,
                    track_name,
                    params,
                }
            }
            t::TRACK_STATUS => {
                let (id, namespace, track_name, params) = read_track_request(r)?;
                Self::TrackStatus {
                    id,
                    namespace,
                    track_name,
                    params,
                }
            }
            t::SUBSCRIBE_OK => Self::SubscribeOk {
                track_alias: r.varint()?,
                params: MessageParams::decode(r)?,
                properties: Params::decode_to_end(r)?,
            },
            t::PUBLISH => {
                let id = r.varint()?;
                let (namespace, track_name) = decode_full_track_name(r)?;
                Self::Publish {
                    id,
                    namespace,
                    track_name,
                    track_alias: r.varint()?,
                    params: MessageParams::decode(r)?,
                    properties: Params::decode_to_end(r)?,
                }
            }
            t::PUBLISH_OK => Self::PublishOk {
                params: MessageParams::decode(r)?,
                properties: Params::decode_to_end(r)?,
            },
            t::REQUEST_OK => Self::RequestOk {
                params: MessageParams::decode(r)?,
                properties: Params::decode_to_end(r)?,
            },
            t::PUBLISH_DONE => Self::PublishDone {
                status_code: r.varint()?,
                stream_count: r.varint()?,
                reason: r.string(MAX_REASON_PHRASE_LEN)?,
            },
            t::FETCH => {
                let id = r.varint()?;
                let fetch_type = FetchType::from_code(r.varint()?)?;
                let (standalone, joining) = match fetch_type {
                    FetchType::Standalone => {
                        let (namespace, track_name) = decode_full_track_name(r)?;
                        (
                            Some(StandaloneFetch {
                                namespace,
                                track_name,
                                start: decode_location(r)?,
                                end: decode_location(r)?,
                            }),
                            None,
                        )
                    }
                    FetchType::RelativeJoining | FetchType::AbsoluteJoining => (
                        None,
                        Some(JoiningFetch {
                            joining_request_id: r.varint()?,
                            joining_start: r.varint()?,
                        }),
                    ),
                };
                Self::Fetch {
                    id,
                    fetch_type,
                    standalone,
                    joining,
                    params: MessageParams::decode(r)?,
                }
            }
            t::FETCH_OK => Self::FetchOk {
                end_of_track: r.bool()?,
                end: decode_location(r)?,
                params: MessageParams::decode(r)?,
                properties: Params::decode_to_end(r)?,
            },
            t::PUBLISH_NAMESPACE => Self::PublishNamespace {
                id: r.varint()?,
                namespace: decode_namespace(r)?,
                params: MessageParams::decode(r)?,
            },
            t::SUBSCRIBE_NAMESPACE => Self::SubscribeNamespace {
                id: r.varint()?,
                prefix: decode_namespace(r)?,
                params: MessageParams::decode(r)?,
            },
            t::SUBSCRIBE_TRACKS => Self::SubscribeTracks {
                id: r.varint()?,
                prefix: decode_namespace(r)?,
                params: MessageParams::decode(r)?,
            },
            t::NAMESPACE => Self::Namespace {
                suffix: decode_namespace(r)?,
            },
            t::NAMESPACE_DONE => Self::NamespaceDone {
                suffix: decode_namespace(r)?,
            },
            t::PUBLISH_BLOCKED => Self::PublishBlocked {
                suffix: decode_namespace(r)?,
                track_name: decode_track_name(r)?,
            },
            t::REQUEST_UPDATE => Self::RequestUpdate {
                id: r.varint()?,
                params: MessageParams::decode(r)?,
            },
            t::REQUEST_ERROR => {
                let error_code = r.varint()?;
                let retry_interval = r.varint()?;
                let reason = r.string(MAX_REASON_PHRASE_LEN)?;
                let redirect = if error_code == request_error_code::REDIRECT {
                    Some(Redirect::decode(r)?)
                } else {
                    None
                };
                Self::RequestError {
                    error_code,
                    retry_interval,
                    reason,
                    redirect,
                }
            }
            _ => return Err(MoqtError::Malformed),
        })
    }
}

/// The `(request id, namespace, track name, params)` shape SUBSCRIBE and
/// TRACK_STATUS share.
fn read_track_request(
    r: &mut Reader<'_>,
) -> Result<(u64, TrackNamespace, TrackName, MessageParams), MoqtError> {
    let id = r.varint()?;
    let (namespace, track_name) = decode_full_track_name(r)?;
    Ok((id, namespace, track_name, MessageParams::decode(r)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn encoded(msg: &ControlMessage) -> Vec<u8> {
        let mut out = Vec::new();
        msg.encode(&mut out).expect("encode");
        out
    }

    fn round_trip(msg: ControlMessage) {
        let bytes = encoded(&msg);
        let (decoded, used) = ControlMessage::decode(&bytes).expect("decode");
        assert_eq!(used, bytes.len(), "{} consumed its frame", msg.name());
        assert_eq!(decoded, msg, "{} round trip", msg.name());
    }

    fn namespace() -> TrackNamespace {
        TrackNamespace::from_path("ns")
    }

    fn params() -> MessageParams {
        let mut params = MessageParams::new();
        params
            .set(
                super::super::coding::param::SUBSCRIBER_PRIORITY,
                super::super::coding::MsgParam::Uint8(200),
            )
            .expect("priority");
        params
    }

    fn properties() -> Params {
        let mut props = Params::new();
        props.set_int(super::super::coding::property::DYNAMIC_GROUPS, 1);
        props
    }

    /// Field-by-field layouts, so a round trip alone cannot hide two neighbours
    /// swapped. Bytes derived from the figures in §10 of the draft.
    #[test]
    fn wire_layouts_match_the_draft_figures() {
        // SETUP: type 0x2F00 is a two-byte vi64 (0xAF 0x00), then a 16-bit
        // length and the Setup Options spanning the payload.
        assert_eq!(
            encoded(&ControlMessage::Setup {
                options: Params::new()
            }),
            vec![0xaf, 0x00, 0x00, 0x00],
            "an empty SETUP has a zero-length payload, not a zero count"
        );

        // SUBSCRIBE: request id, namespace, name, parameter count.
        assert_eq!(
            encoded(&ControlMessage::Subscribe {
                id: 0,
                namespace: namespace(),
                track_name: TrackName::new("t"),
                params: MessageParams::new(),
            }),
            vec![0x03, 0x00, 0x08, 0x00, 0x01, 0x02, b'n', b's', 0x01, b't', 0x00]
        );

        // SUBSCRIBE_OK: track alias first, and no request id at all.
        assert_eq!(
            encoded(&ControlMessage::SubscribeOk {
                track_alias: 7,
                params: MessageParams::new(),
                properties: Params::new(),
            }),
            vec![0x04, 0x00, 0x02, 0x07, 0x00]
        );

        // REQUEST_OK / PUBLISH_OK share a payload and differ only in type.
        assert_eq!(
            encoded(&ControlMessage::RequestOk {
                params: MessageParams::new(),
                properties: Params::new(),
            }),
            vec![0x07, 0x00, 0x01, 0x00]
        );
        assert_eq!(
            encoded(&ControlMessage::PublishOk {
                params: MessageParams::new(),
                properties: Params::new(),
            }),
            vec![0x1e, 0x00, 0x01, 0x00]
        );

        // REQUEST_ERROR carries no request id either.
        assert_eq!(
            encoded(&ControlMessage::RequestError {
                error_code: request_error_code::DOES_NOT_EXIST,
                retry_interval: 0,
                reason: String::from("no"),
                redirect: None,
            }),
            vec![0x05, 0x00, 0x05, 0x10, 0x00, 0x02, b'n', b'o']
        );

        // PUBLISH_DONE: status, stream count, reason.
        assert_eq!(
            encoded(&ControlMessage::PublishDone {
                status_code: publish_done_code::TRACK_ENDED,
                stream_count: 3,
                reason: String::new(),
            }),
            vec![0x0b, 0x00, 0x03, 0x02, 0x03, 0x00]
        );

        // PUBLISH_NAMESPACE: request id, namespace, parameter count.
        assert_eq!(
            encoded(&ControlMessage::PublishNamespace {
                id: 0,
                namespace: namespace(),
                params: MessageParams::new(),
            }),
            vec![0x06, 0x00, 0x06, 0x00, 0x01, 0x02, b'n', b's', 0x00]
        );

        // SUBSCRIBE_NAMESPACE lost draft-16's options field, and its type is
        // 0x50 rather than 0x11.
        assert_eq!(
            encoded(&ControlMessage::SubscribeNamespace {
                id: 0,
                prefix: TrackNamespace::default(),
                params: MessageParams::new(),
            }),
            vec![0x50, 0x00, 0x03, 0x00, 0x00, 0x00]
        );
        assert_eq!(
            encoded(&ControlMessage::SubscribeTracks {
                id: 2,
                prefix: TrackNamespace::default(),
                params: MessageParams::new(),
            }),
            vec![0x51, 0x00, 0x03, 0x02, 0x00, 0x00]
        );

        // GOAWAY on a control stream carries the trailing Request ID.
        assert_eq!(
            encoded(&ControlMessage::GoAway {
                uri: String::new(),
                timeout_ms: 100,
                request_id: Some(4),
            }),
            vec![0x10, 0x00, 0x03, 0x00, 0x64, 0x04]
        );
    }

    #[test]
    fn every_table_5_message_round_trips() {
        for msg in [
            ControlMessage::Setup {
                options: properties(),
            },
            ControlMessage::GoAway {
                uri: String::from("https://relay.example/2"),
                timeout_ms: 5000,
                request_id: Some(6),
            },
            // The request-stream form, which omits the Request ID.
            ControlMessage::GoAway {
                uri: String::new(),
                timeout_ms: 0,
                request_id: None,
            },
            ControlMessage::Subscribe {
                id: 4,
                namespace: TrackNamespace::from_path("live/cam"),
                track_name: TrackName::new("1.m4s"),
                params: params(),
            },
            ControlMessage::SubscribeOk {
                track_alias: 9,
                params: params(),
                properties: properties(),
            },
            ControlMessage::Publish {
                id: 12,
                namespace: TrackNamespace::from_path("live/cam"),
                track_name: TrackName::new("1.m4s"),
                track_alias: 12,
                params: params(),
                properties: properties(),
            },
            ControlMessage::PublishOk {
                params: params(),
                properties: Params::new(),
            },
            ControlMessage::PublishDone {
                status_code: publish_done_code::TRACK_ENDED,
                stream_count: 12,
                reason: String::from("eos"),
            },
            ControlMessage::Fetch {
                id: 10,
                fetch_type: FetchType::Standalone,
                standalone: Some(StandaloneFetch {
                    namespace: TrackNamespace::from_path("live/cam"),
                    track_name: TrackName::new("1.m4s"),
                    start: Location::default(),
                    end: Location {
                        group_id: 4,
                        object_id: 0,
                    },
                }),
                joining: None,
                params: params(),
            },
            ControlMessage::Fetch {
                id: 12,
                fetch_type: FetchType::RelativeJoining,
                standalone: None,
                joining: Some(JoiningFetch {
                    joining_request_id: 4,
                    joining_start: 2,
                }),
                params: MessageParams::new(),
            },
            ControlMessage::FetchOk {
                end_of_track: true,
                end: Location {
                    group_id: 3,
                    object_id: 9,
                },
                params: params(),
                properties: properties(),
            },
            ControlMessage::TrackStatus {
                id: 6,
                namespace: TrackNamespace::from_path("live/cam"),
                track_name: TrackName::new("1.m4s"),
                params: MessageParams::new(),
            },
            ControlMessage::PublishNamespace {
                id: 0,
                namespace: TrackNamespace::from_path("live/cam"),
                params: params(),
            },
            ControlMessage::SubscribeNamespace {
                id: 8,
                prefix: TrackNamespace(vec![b"live".to_vec()]),
                params: params(),
            },
            ControlMessage::SubscribeTracks {
                id: 10,
                prefix: TrackNamespace(vec![b"live".to_vec()]),
                params: params(),
            },
            ControlMessage::Namespace {
                suffix: TrackNamespace(vec![b"cam".to_vec()]),
            },
            ControlMessage::NamespaceDone {
                suffix: TrackNamespace(vec![b"cam".to_vec()]),
            },
            ControlMessage::PublishBlocked {
                suffix: TrackNamespace(vec![b"cam".to_vec()]),
                track_name: TrackName::new("1.m4s"),
            },
            ControlMessage::RequestUpdate {
                id: 2,
                params: params(),
            },
            ControlMessage::RequestOk {
                params: params(),
                properties: properties(),
            },
            ControlMessage::RequestError {
                error_code: request_error_code::DOES_NOT_EXIST,
                retry_interval: 0,
                reason: String::from("no such track"),
                redirect: None,
            },
            ControlMessage::RequestError {
                error_code: request_error_code::REDIRECT,
                retry_interval: 1,
                reason: String::from("elsewhere"),
                redirect: Some(Redirect {
                    connect_uri: String::from("https://other.example/"),
                    namespace: TrackNamespace::from_path("live/cam2"),
                    track_name: TrackName::new("1.m4s"),
                }),
            },
        ] {
            round_trip(msg);
        }
    }

    #[test]
    fn request_ids_and_stream_openers_match_table_5() {
        let sub = ControlMessage::Subscribe {
            id: 4,
            namespace: namespace(),
            track_name: TrackName::new("t"),
            params: MessageParams::new(),
        };
        assert_eq!(sub.request_id(), Some(4));
        assert!(sub.opens_request_stream());

        let ok = ControlMessage::SubscribeOk {
            track_alias: 4,
            params: MessageParams::new(),
            properties: Params::new(),
        };
        assert_eq!(ok.request_id(), None, "responses carry no request id");
        assert!(!ok.opens_request_stream());

        // NAMESPACE and PUBLISH_DONE travel on an existing request stream.
        assert!(!ControlMessage::Namespace {
            suffix: namespace()
        }
        .opens_request_stream());
        assert!(!ControlMessage::PublishDone {
            status_code: 0,
            stream_count: 0,
            reason: String::new(),
        }
        .opens_request_stream());
    }

    #[test]
    fn a_truncated_frame_is_incomplete_and_a_padded_payload_is_malformed() {
        let bytes = encoded(&ControlMessage::Subscribe {
            id: 0,
            namespace: namespace(),
            track_name: TrackName::new("t"),
            params: MessageParams::new(),
        });
        for cut in 0..bytes.len() {
            assert_eq!(
                ControlMessage::decode(&bytes[..cut]).map(|(_, n)| n),
                Err(MoqtError::Incomplete),
                "a prefix of {cut} bytes is a partial frame"
            );
        }

        // A declared length longer than the message needs leaves bytes over.
        let mut padded = bytes.clone();
        padded[2] += 1;
        padded.push(0x00);
        assert_eq!(
            ControlMessage::decode(&padded).map(|(_, n)| n),
            Err(MoqtError::Malformed)
        );
    }

    #[test]
    fn unknown_types_and_out_of_range_values_are_refused() {
        // An unassigned type number: §10 makes it a session error.
        assert_eq!(
            ControlMessage::decode(&[0x77, 0x00, 0x00]).map(|(_, n)| n),
            Err(MoqtError::Malformed)
        );
        // Draft-16's CLIENT_SETUP is explicitly reserved in draft-18.
        assert_eq!(
            ControlMessage::decode(&[0x20, 0x00, 0x01, 0x00]).map(|(_, n)| n),
            Err(MoqtError::Malformed)
        );
        // FETCH with an unassigned fetch type.
        assert_eq!(
            ControlMessage::decode_payload(msg_type::FETCH, &[0x00, 0x09]),
            Err(MoqtError::Malformed)
        );
        // FETCH_OK's End Of Track is a one-byte boolean.
        assert_eq!(
            ControlMessage::decode_payload(msg_type::FETCH_OK, &[0x02, 0x00, 0x00, 0x00]),
            Err(MoqtError::Malformed)
        );
        // A namespace count that overruns the payload, and one over the limit.
        assert_eq!(
            ControlMessage::decode_payload(msg_type::SUBSCRIBE, &[0x00, 0x20, 0x01, b'a']),
            Err(MoqtError::Incomplete)
        );
        assert_eq!(
            ControlMessage::decode_payload(msg_type::SUBSCRIBE, &[0x00, 0x21, 0x01, b'a']),
            Err(MoqtError::Malformed)
        );
        // A REQUEST_ERROR that claims REDIRECT without one, and a Redirect on a
        // code that does not carry it.
        assert_eq!(
            ControlMessage::decode_payload(msg_type::REQUEST_ERROR, &[0x34, 0x00, 0x00]),
            Err(MoqtError::Incomplete)
        );
        assert_eq!(
            ControlMessage::RequestError {
                error_code: request_error_code::INTERNAL_ERROR,
                retry_interval: 0,
                reason: String::new(),
                redirect: Some(Redirect::default()),
            }
            .encode(&mut Vec::new()),
            Err(MoqtError::Malformed)
        );
    }

    #[test]
    fn subscription_filters_round_trip_and_reject_bad_shapes() {
        for filter in [
            SubscriptionFilter::NextGroupStart,
            SubscriptionFilter::LargestObject,
            SubscriptionFilter::AbsoluteStart {
                start: Location {
                    group_id: 5,
                    object_id: 0,
                },
            },
            SubscriptionFilter::AbsoluteRange {
                start: Location {
                    group_id: 5,
                    object_id: 2,
                },
                end_group_delta: 3,
            },
        ] {
            let bytes = filter.to_bytes().expect("encode");
            assert_eq!(SubscriptionFilter::from_bytes(&bytes), Ok(filter));
        }

        assert_eq!(
            SubscriptionFilter::LargestObject.to_bytes().as_deref(),
            Ok(&[0x02u8][..])
        );
        // A filter type outside the four defined ones.
        assert_eq!(
            SubscriptionFilter::from_bytes(&[0x05]),
            Err(MoqtError::Malformed)
        );
        // A trailing byte means the peer's shape and ours disagree.
        assert_eq!(
            SubscriptionFilter::from_bytes(&[0x02, 0x00]),
            Err(MoqtError::Malformed)
        );
        // An end group past u64 is a violation both ways.
        let overflow = SubscriptionFilter::AbsoluteRange {
            start: Location {
                group_id: u64::MAX,
                object_id: 0,
            },
            end_group_delta: 1,
        };
        assert_eq!(overflow.to_bytes(), Err(MoqtError::Malformed));
        let mut bytes = Vec::new();
        put_vi64(&mut bytes, 0x4);
        put_vi64(&mut bytes, u64::MAX);
        put_vi64(&mut bytes, 0);
        put_vi64(&mut bytes, 1);
        assert_eq!(
            SubscriptionFilter::from_bytes(&bytes),
            Err(MoqtError::Malformed)
        );
    }
}
