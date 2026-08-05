//! Control-stream messages for IETF MoQ Transport draft-16.
//!
//! Every control message is framed the same way (§9):
//!
//! ```text
//! Message Type (i), Message Length (16), Message Payload (..)
//! ```
//!
//! The 16-bit length is big-endian and exact: a payload that does not consume
//! it is a protocol violation. Type numbers and field order follow
//! `moq-rs/moq-transport/src/message/mod.rs` (Table 1) and the per-message
//! files beside it; the SETUP pair (0x20 / 0x21) uses the same framing and
//! lives in `moq-rs/moq-transport/src/setup/`.
//!
//! Draft-16 negotiates the version through the WebTransport protocol
//! (`moqt-16`), not a version list in the SETUP payload, so CLIENT_SETUP and
//! SERVER_SETUP carry parameters only.

use alloc::string::String;
use alloc::vec::Vec;

use super::coding::{
    put_varint, validate_full_track_name, MoqtError, Params, Reader, TrackName, TrackNamespace,
    TrackNamespacePrefix, MAX_REASON_PHRASE_LEN, MAX_SESSION_URI_LEN,
};

/// Message type numbers (draft-16 Table 1).
pub mod msg_type {
    pub const REQUEST_UPDATE: u64 = 0x2;
    pub const SUBSCRIBE: u64 = 0x3;
    pub const SUBSCRIBE_OK: u64 = 0x4;
    pub const REQUEST_ERROR: u64 = 0x5;
    pub const PUBLISH_NAMESPACE: u64 = 0x6;
    pub const REQUEST_OK: u64 = 0x7;
    pub const NAMESPACE: u64 = 0x8;
    pub const PUBLISH_NAMESPACE_DONE: u64 = 0x9;
    pub const UNSUBSCRIBE: u64 = 0xa;
    pub const PUBLISH_DONE: u64 = 0xb;
    pub const PUBLISH_NAMESPACE_CANCEL: u64 = 0xc;
    pub const TRACK_STATUS: u64 = 0xd;
    pub const NAMESPACE_DONE: u64 = 0xe;
    pub const GOAWAY: u64 = 0x10;
    pub const SUBSCRIBE_NAMESPACE: u64 = 0x11;
    pub const MAX_REQUEST_ID: u64 = 0x15;
    pub const FETCH: u64 = 0x16;
    pub const FETCH_CANCEL: u64 = 0x17;
    pub const FETCH_OK: u64 = 0x18;
    pub const REQUESTS_BLOCKED: u64 = 0x1a;
    pub const CLIENT_SETUP: u64 = 0x20;
    pub const SERVER_SETUP: u64 = 0x21;
    pub const PUBLISH: u64 = 0x1d;
    pub const PUBLISH_OK: u64 = 0x1e;
}

/// REQUEST_ERROR codes (draft-16 §13.4.2).
pub mod request_error_code {
    pub const INTERNAL_ERROR: u64 = 0x0;
    pub const UNAUTHORIZED: u64 = 0x1;
    pub const TIMEOUT: u64 = 0x2;
    pub const NOT_SUPPORTED: u64 = 0x3;
    pub const DOES_NOT_EXIST: u64 = 0x10;
    pub const INVALID_RANGE: u64 = 0x11;
    pub const MALFORMED_TRACK: u64 = 0x12;
    pub const DUPLICATE_SUBSCRIPTION: u64 = 0x19;
    pub const UNINTERESTED: u64 = 0x20;
    pub const INVALID_JOINING_REQUEST_ID: u64 = 0x32;
}

/// Data-stream reset codes (draft-16 §13.4.4). A data stream that ends before
/// all of its objects have been written is reset with one of these rather than
/// finished.
pub mod stream_error_code {
    pub const CANCELLED: u32 = 0x1;
}

/// PUBLISH_DONE status codes (draft-16 §13.4.3).
pub mod publish_done_code {
    pub const INTERNAL_ERROR: u64 = 0x0;
    pub const TRACK_ENDED: u64 = 0x2;
    pub const SUBSCRIPTION_ENDED: u64 = 0x3;
    pub const GOING_AWAY: u64 = 0x4;
}

/// A group/object coordinate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Location {
    pub group_id: u64,
    pub object_id: u64,
}

impl Location {
    fn encode(&self, out: &mut Vec<u8>) {
        put_varint(out, self.group_id);
        put_varint(out, self.object_id);
    }

    fn decode(r: &mut Reader<'_>) -> Result<Self, MoqtError> {
        Ok(Self {
            group_id: r.varint()?,
            object_id: r.varint()?,
        })
    }
}

/// FETCH variants (draft-16 §9.19).
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
        match code {
            0x1 => Ok(Self::Standalone),
            0x2 => Ok(Self::RelativeJoining),
            0x3 => Ok(Self::AbsoluteJoining),
            _ => Err(MoqtError::Malformed),
        }
    }
}

/// What a SUBSCRIBE_NAMESPACE asks the peer to send back (draft-16 §9.25).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubscribeNamespaceOptions {
    Publish,
    Namespace,
    Both,
}

impl SubscribeNamespaceOptions {
    fn code(self) -> u64 {
        match self {
            Self::Publish => 0x0,
            Self::Namespace => 0x1,
            Self::Both => 0x2,
        }
    }

    fn from_code(code: u64) -> Result<Self, MoqtError> {
        match code {
            0x0 => Ok(Self::Publish),
            0x1 => Ok(Self::Namespace),
            0x2 => Ok(Self::Both),
            _ => Err(MoqtError::Malformed),
        }
    }
}

/// The body of a standalone FETCH.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StandaloneFetch {
    pub namespace: TrackNamespace,
    pub track_name: TrackName,
    pub start: Location,
    pub end: Location,
}

/// The body of a joining FETCH.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JoiningFetch {
    pub joining_request_id: u64,
    pub joining_start: u64,
}

/// Every control message this session speaks or tolerates.
///
/// A publisher acts on a handful of these; the rest are decoded so a peer's
/// stream stays framed and their content is observable, then ignored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlMessage {
    ClientSetup {
        params: Params,
    },
    ServerSetup {
        params: Params,
    },
    RequestUpdate {
        id: u64,
        existing_request_id: u64,
        params: Params,
    },
    Subscribe {
        id: u64,
        namespace: TrackNamespace,
        track_name: TrackName,
        params: Params,
    },
    SubscribeOk {
        id: u64,
        track_alias: u64,
        params: Params,
        extensions: Params,
    },
    RequestError {
        id: u64,
        error_code: u64,
        retry_interval: u64,
        reason: String,
    },
    PublishNamespace {
        id: u64,
        namespace: TrackNamespace,
        params: Params,
    },
    RequestOk {
        id: u64,
        params: Params,
    },
    Namespace {
        suffix: TrackNamespacePrefix,
    },
    PublishNamespaceDone {
        id: u64,
    },
    Unsubscribe {
        id: u64,
    },
    PublishDone {
        id: u64,
        status_code: u64,
        stream_count: u64,
        reason: String,
    },
    PublishNamespaceCancel {
        id: u64,
        error_code: u64,
        reason: String,
    },
    TrackStatus {
        id: u64,
        namespace: TrackNamespace,
        track_name: TrackName,
        params: Params,
    },
    NamespaceDone {
        suffix: TrackNamespacePrefix,
    },
    GoAway {
        uri: String,
    },
    SubscribeNamespace {
        id: u64,
        prefix: TrackNamespacePrefix,
        options: SubscribeNamespaceOptions,
        params: Params,
    },
    MaxRequestId {
        request_id: u64,
    },
    Fetch {
        id: u64,
        fetch_type: FetchType,
        standalone: Option<StandaloneFetch>,
        joining: Option<JoiningFetch>,
        params: Params,
    },
    FetchCancel {
        id: u64,
    },
    FetchOk {
        id: u64,
        end_of_track: bool,
        end: Location,
        params: Params,
        extensions: Params,
    },
    RequestsBlocked {
        max_request_id: u64,
    },
    Publish {
        id: u64,
        namespace: TrackNamespace,
        track_name: TrackName,
        track_alias: u64,
        params: Params,
        extensions: Params,
    },
    PublishOk {
        id: u64,
        params: Params,
    },
}

impl ControlMessage {
    /// The message type number this variant is sent as.
    pub fn type_id(&self) -> u64 {
        use msg_type as t;
        match self {
            Self::ClientSetup { .. } => t::CLIENT_SETUP,
            Self::ServerSetup { .. } => t::SERVER_SETUP,
            Self::RequestUpdate { .. } => t::REQUEST_UPDATE,
            Self::Subscribe { .. } => t::SUBSCRIBE,
            Self::SubscribeOk { .. } => t::SUBSCRIBE_OK,
            Self::RequestError { .. } => t::REQUEST_ERROR,
            Self::PublishNamespace { .. } => t::PUBLISH_NAMESPACE,
            Self::RequestOk { .. } => t::REQUEST_OK,
            Self::Namespace { .. } => t::NAMESPACE,
            Self::PublishNamespaceDone { .. } => t::PUBLISH_NAMESPACE_DONE,
            Self::Unsubscribe { .. } => t::UNSUBSCRIBE,
            Self::PublishDone { .. } => t::PUBLISH_DONE,
            Self::PublishNamespaceCancel { .. } => t::PUBLISH_NAMESPACE_CANCEL,
            Self::TrackStatus { .. } => t::TRACK_STATUS,
            Self::NamespaceDone { .. } => t::NAMESPACE_DONE,
            Self::GoAway { .. } => t::GOAWAY,
            Self::SubscribeNamespace { .. } => t::SUBSCRIBE_NAMESPACE,
            Self::MaxRequestId { .. } => t::MAX_REQUEST_ID,
            Self::Fetch { .. } => t::FETCH,
            Self::FetchCancel { .. } => t::FETCH_CANCEL,
            Self::FetchOk { .. } => t::FETCH_OK,
            Self::RequestsBlocked { .. } => t::REQUESTS_BLOCKED,
            Self::Publish { .. } => t::PUBLISH,
            Self::PublishOk { .. } => t::PUBLISH_OK,
        }
    }

    /// A human name for logging, matching the draft's message names.
    pub fn name(&self) -> &'static str {
        match self {
            Self::ClientSetup { .. } => "CLIENT_SETUP",
            Self::ServerSetup { .. } => "SERVER_SETUP",
            Self::RequestUpdate { .. } => "REQUEST_UPDATE",
            Self::Subscribe { .. } => "SUBSCRIBE",
            Self::SubscribeOk { .. } => "SUBSCRIBE_OK",
            Self::RequestError { .. } => "REQUEST_ERROR",
            Self::PublishNamespace { .. } => "PUBLISH_NAMESPACE",
            Self::RequestOk { .. } => "REQUEST_OK",
            Self::Namespace { .. } => "NAMESPACE",
            Self::PublishNamespaceDone { .. } => "PUBLISH_NAMESPACE_DONE",
            Self::Unsubscribe { .. } => "UNSUBSCRIBE",
            Self::PublishDone { .. } => "PUBLISH_DONE",
            Self::PublishNamespaceCancel { .. } => "PUBLISH_NAMESPACE_CANCEL",
            Self::TrackStatus { .. } => "TRACK_STATUS",
            Self::NamespaceDone { .. } => "NAMESPACE_DONE",
            Self::GoAway { .. } => "GOAWAY",
            Self::SubscribeNamespace { .. } => "SUBSCRIBE_NAMESPACE",
            Self::MaxRequestId { .. } => "MAX_REQUEST_ID",
            Self::Fetch { .. } => "FETCH",
            Self::FetchCancel { .. } => "FETCH_CANCEL",
            Self::FetchOk { .. } => "FETCH_OK",
            Self::RequestsBlocked { .. } => "REQUESTS_BLOCKED",
            Self::Publish { .. } => "PUBLISH",
            Self::PublishOk { .. } => "PUBLISH_OK",
        }
    }

    /// Append the framed message (type, 16-bit length, payload).
    pub fn encode(&self, out: &mut Vec<u8>) -> Result<(), MoqtError> {
        let mut payload = Vec::new();
        self.encode_payload(&mut payload)?;
        if payload.len() > u16::MAX as usize {
            return Err(MoqtError::Malformed);
        }
        put_varint(out, self.type_id());
        out.extend_from_slice(&(payload.len() as u16).to_be_bytes());
        out.extend_from_slice(&payload);
        Ok(())
    }

    /// Decode one framed message, returning it and the bytes it consumed.
    /// [`MoqtError::Incomplete`] means the frame is not fully buffered.
    pub fn decode(buf: &[u8]) -> Result<(Self, usize), MoqtError> {
        let mut r = Reader::new(buf);
        let typ = r.varint()?;
        let len = r.u16()? as usize;
        let payload = r.bytes(len)?;
        let msg = Self::decode_payload(typ, payload)?;
        Ok((msg, r.position()))
    }

    /// Decode a payload already separated from its frame. The payload must be
    /// consumed exactly; leftover bytes are a protocol violation (§9).
    pub fn decode_payload(typ: u64, payload: &[u8]) -> Result<Self, MoqtError> {
        let mut r = Reader::new(payload);
        let msg = Self::read(typ, &mut r)?;
        if !r.is_empty() {
            return Err(MoqtError::Malformed);
        }
        Ok(msg)
    }

    fn encode_payload(&self, out: &mut Vec<u8>) -> Result<(), MoqtError> {
        match self {
            Self::ClientSetup { params } | Self::ServerSetup { params } => params.encode(out)?,
            Self::RequestUpdate {
                id,
                existing_request_id,
                params,
            } => {
                put_varint(out, *id);
                put_varint(out, *existing_request_id);
                params.encode(out)?;
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
                put_varint(out, *id);
                namespace.encode(out)?;
                track_name.encode(out)?;
                params.encode(out)?;
            }
            Self::SubscribeOk {
                id,
                track_alias,
                params,
                extensions,
            } => {
                put_varint(out, *id);
                put_varint(out, *track_alias);
                params.encode(out)?;
                extensions.encode_pairs(out)?;
            }
            Self::RequestError {
                id,
                error_code,
                retry_interval,
                reason,
            } => {
                put_varint(out, *id);
                put_varint(out, *error_code);
                put_varint(out, *retry_interval);
                encode_string(out, reason, MAX_REASON_PHRASE_LEN)?;
            }
            Self::PublishNamespace {
                id,
                namespace,
                params,
            } => {
                put_varint(out, *id);
                namespace.encode(out)?;
                params.encode(out)?;
            }
            Self::RequestOk { id, params } | Self::PublishOk { id, params } => {
                put_varint(out, *id);
                params.encode(out)?;
            }
            Self::Namespace { suffix } | Self::NamespaceDone { suffix } => suffix.encode(out)?,
            Self::PublishNamespaceDone { id }
            | Self::Unsubscribe { id }
            | Self::FetchCancel { id } => put_varint(out, *id),
            Self::PublishDone {
                id,
                status_code,
                stream_count,
                reason,
            } => {
                put_varint(out, *id);
                put_varint(out, *status_code);
                put_varint(out, *stream_count);
                encode_string(out, reason, MAX_REASON_PHRASE_LEN)?;
            }
            Self::PublishNamespaceCancel {
                id,
                error_code,
                reason,
            } => {
                put_varint(out, *id);
                put_varint(out, *error_code);
                encode_string(out, reason, MAX_REASON_PHRASE_LEN)?;
            }
            Self::GoAway { uri } => encode_string(out, uri, MAX_SESSION_URI_LEN)?,
            Self::SubscribeNamespace {
                id,
                prefix,
                options,
                params,
            } => {
                put_varint(out, *id);
                prefix.encode(out)?;
                put_varint(out, options.code());
                params.encode(out)?;
            }
            Self::MaxRequestId { request_id } => put_varint(out, *request_id),
            Self::RequestsBlocked { max_request_id } => put_varint(out, *max_request_id),
            Self::Fetch {
                id,
                fetch_type,
                standalone,
                joining,
                params,
            } => {
                put_varint(out, *id);
                put_varint(out, fetch_type.code());
                match fetch_type {
                    FetchType::Standalone => {
                        let body = standalone.as_ref().ok_or(MoqtError::Malformed)?;
                        body.namespace.encode(out)?;
                        body.track_name.encode(out)?;
                        body.start.encode(out);
                        body.end.encode(out);
                    }
                    FetchType::RelativeJoining | FetchType::AbsoluteJoining => {
                        let body = joining.as_ref().ok_or(MoqtError::Malformed)?;
                        put_varint(out, body.joining_request_id);
                        put_varint(out, body.joining_start);
                    }
                }
                params.encode(out)?;
            }
            Self::FetchOk {
                id,
                end_of_track,
                end,
                params,
                extensions,
            } => {
                put_varint(out, *id);
                out.push(u8::from(*end_of_track));
                end.encode(out);
                params.encode(out)?;
                extensions.encode_pairs(out)?;
            }
            Self::Publish {
                id,
                namespace,
                track_name,
                track_alias,
                params,
                extensions,
            } => {
                put_varint(out, *id);
                namespace.encode(out)?;
                track_name.encode(out)?;
                put_varint(out, *track_alias);
                params.encode(out)?;
                extensions.encode_pairs(out)?;
            }
        }
        Ok(())
    }

    fn read(typ: u64, r: &mut Reader<'_>) -> Result<Self, MoqtError> {
        use msg_type as t;
        Ok(match typ {
            t::CLIENT_SETUP => Self::ClientSetup {
                params: Params::decode(r)?,
            },
            t::SERVER_SETUP => Self::ServerSetup {
                params: Params::decode(r)?,
            },
            t::REQUEST_UPDATE => Self::RequestUpdate {
                id: r.varint()?,
                existing_request_id: r.varint()?,
                params: Params::decode(r)?,
            },
            t::SUBSCRIBE => {
                let (id, namespace, track_name, params) = read_full_track_request(r)?;
                Self::Subscribe {
                    id,
                    namespace,
                    track_name,
                    params,
                }
            }
            t::TRACK_STATUS => {
                let (id, namespace, track_name, params) = read_full_track_request(r)?;
                Self::TrackStatus {
                    id,
                    namespace,
                    track_name,
                    params,
                }
            }
            t::SUBSCRIBE_OK => Self::SubscribeOk {
                id: r.varint()?,
                track_alias: r.varint()?,
                params: Params::decode(r)?,
                extensions: Params::decode_to_end(r)?,
            },
            t::REQUEST_ERROR => Self::RequestError {
                id: r.varint()?,
                error_code: r.varint()?,
                retry_interval: r.varint()?,
                reason: r.string(MAX_REASON_PHRASE_LEN)?,
            },
            t::PUBLISH_NAMESPACE => Self::PublishNamespace {
                id: r.varint()?,
                namespace: TrackNamespace::decode(r)?,
                params: Params::decode(r)?,
            },
            t::REQUEST_OK => Self::RequestOk {
                id: r.varint()?,
                params: Params::decode(r)?,
            },
            t::PUBLISH_OK => Self::PublishOk {
                id: r.varint()?,
                params: Params::decode(r)?,
            },
            t::NAMESPACE => Self::Namespace {
                suffix: TrackNamespacePrefix::decode(r)?,
            },
            t::NAMESPACE_DONE => Self::NamespaceDone {
                suffix: TrackNamespacePrefix::decode(r)?,
            },
            t::PUBLISH_NAMESPACE_DONE => Self::PublishNamespaceDone { id: r.varint()? },
            t::UNSUBSCRIBE => Self::Unsubscribe { id: r.varint()? },
            t::FETCH_CANCEL => Self::FetchCancel { id: r.varint()? },
            t::PUBLISH_DONE => Self::PublishDone {
                id: r.varint()?,
                status_code: r.varint()?,
                stream_count: r.varint()?,
                reason: r.string(MAX_REASON_PHRASE_LEN)?,
            },
            t::PUBLISH_NAMESPACE_CANCEL => Self::PublishNamespaceCancel {
                id: r.varint()?,
                error_code: r.varint()?,
                reason: r.string(MAX_REASON_PHRASE_LEN)?,
            },
            t::GOAWAY => Self::GoAway {
                uri: r.string(MAX_SESSION_URI_LEN)?,
            },
            t::SUBSCRIBE_NAMESPACE => Self::SubscribeNamespace {
                id: r.varint()?,
                prefix: TrackNamespacePrefix::decode(r)?,
                options: SubscribeNamespaceOptions::from_code(r.varint()?)?,
                params: Params::decode(r)?,
            },
            t::MAX_REQUEST_ID => Self::MaxRequestId {
                request_id: r.varint()?,
            },
            t::REQUESTS_BLOCKED => Self::RequestsBlocked {
                max_request_id: r.varint()?,
            },
            t::FETCH => {
                let id = r.varint()?;
                let fetch_type = FetchType::from_code(r.varint()?)?;
                let (standalone, joining) = match fetch_type {
                    FetchType::Standalone => {
                        let namespace = TrackNamespace::decode(r)?;
                        let track_name = TrackName::decode(r)?;
                        validate_full_track_name(&namespace, &track_name)?;
                        (
                            Some(StandaloneFetch {
                                namespace,
                                track_name,
                                start: Location::decode(r)?,
                                end: Location::decode(r)?,
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
                    params: Params::decode(r)?,
                }
            }
            t::FETCH_OK => Self::FetchOk {
                id: r.varint()?,
                end_of_track: r.bool()?,
                end: Location::decode(r)?,
                params: Params::decode(r)?,
                extensions: Params::decode_to_end(r)?,
            },
            t::PUBLISH => {
                let id = r.varint()?;
                let namespace = TrackNamespace::decode(r)?;
                let track_name = TrackName::decode(r)?;
                validate_full_track_name(&namespace, &track_name)?;
                Self::Publish {
                    id,
                    namespace,
                    track_name,
                    track_alias: r.varint()?,
                    params: Params::decode(r)?,
                    extensions: Params::decode_to_end(r)?,
                }
            }
            _ => return Err(MoqtError::Malformed),
        })
    }
}

/// The `(request id, namespace, track name, params)` shape SUBSCRIBE and
/// TRACK_STATUS share.
fn read_full_track_request(
    r: &mut Reader<'_>,
) -> Result<(u64, TrackNamespace, TrackName, Params), MoqtError> {
    let id = r.varint()?;
    let namespace = TrackNamespace::decode(r)?;
    let track_name = TrackName::decode(r)?;
    validate_full_track_name(&namespace, &track_name)?;
    Ok((id, namespace, track_name, Params::decode(r)?))
}

fn encode_string(out: &mut Vec<u8>, s: &str, max: usize) -> Result<(), MoqtError> {
    if s.len() > max {
        return Err(MoqtError::Malformed);
    }
    put_varint(out, s.len() as u64);
    out.extend_from_slice(s.as_bytes());
    Ok(())
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

    /// Byte layouts taken from the reference's own assertions in
    /// `moq-rs/moq-transport/src/message/mod.rs` (`draft16_wire_layouts_for_changed_control_messages`)
    /// and `setup/{client,server}.rs`. A round trip alone would not catch a
    /// field swapped with its neighbour.
    #[test]
    fn wire_layouts_match_the_reference() {
        assert_eq!(
            encoded(&ControlMessage::Subscribe {
                id: 0,
                namespace: namespace(),
                track_name: TrackName::new("t"),
                params: Params::new(),
            }),
            vec![0x03, 0x00, 0x08, 0x00, 0x01, 0x02, b'n', b's', 0x01, b't', 0x00]
        );

        assert_eq!(
            encoded(&ControlMessage::SubscribeOk {
                id: 0,
                track_alias: 1,
                params: Params::new(),
                extensions: Params::new(),
            }),
            vec![0x04, 0x00, 0x03, 0x00, 0x01, 0x00]
        );

        assert_eq!(
            encoded(&ControlMessage::TrackStatus {
                id: 0,
                namespace: namespace(),
                track_name: TrackName::new("t"),
                params: Params::new(),
            }),
            vec![0x0d, 0x00, 0x08, 0x00, 0x01, 0x02, b'n', b's', 0x01, b't', 0x00]
        );

        assert_eq!(
            encoded(&ControlMessage::Publish {
                id: 0,
                namespace: namespace(),
                track_name: TrackName::new("t"),
                track_alias: 5,
                params: Params::new(),
                extensions: Params::new(),
            }),
            vec![0x1d, 0x00, 0x09, 0x00, 0x01, 0x02, b'n', b's', 0x01, b't', 0x05, 0x00]
        );

        assert_eq!(
            encoded(&ControlMessage::PublishOk {
                id: 0,
                params: Params::new(),
            }),
            vec![0x1e, 0x00, 0x02, 0x00, 0x00]
        );

        assert_eq!(
            encoded(&ControlMessage::Fetch {
                id: 0,
                fetch_type: FetchType::Standalone,
                standalone: Some(StandaloneFetch {
                    namespace: namespace(),
                    track_name: TrackName::new("t"),
                    start: Location::default(),
                    end: Location {
                        group_id: 0,
                        object_id: 1
                    },
                }),
                joining: None,
                params: Params::new(),
            }),
            vec![
                0x16, 0x00, 0x0d, 0x00, 0x01, 0x01, 0x02, b'n', b's', 0x01, b't', 0x00, 0x00, 0x00,
                0x01, 0x00
            ]
        );

        assert_eq!(
            encoded(&ControlMessage::FetchOk {
                id: 0,
                end_of_track: false,
                end: Location {
                    group_id: 0,
                    object_id: 1
                },
                params: Params::new(),
                extensions: Params::new(),
            }),
            vec![0x18, 0x00, 0x05, 0x00, 0x00, 0x00, 0x01, 0x00]
        );

        assert_eq!(
            encoded(&ControlMessage::SubscribeNamespace {
                id: 0,
                prefix: TrackNamespacePrefix::default(),
                options: SubscribeNamespaceOptions::Both,
                params: Params::new(),
            }),
            vec![0x11, 0x00, 0x04, 0x00, 0x00, 0x02, 0x00]
        );

        // CLIENT_SETUP / SERVER_SETUP carry parameters only in draft-16: type,
        // 16-bit length, then a KVP count of zero.
        assert_eq!(
            encoded(&ControlMessage::ClientSetup {
                params: Params::new()
            }),
            vec![0x20, 0x00, 0x01, 0x00]
        );
        assert_eq!(
            encoded(&ControlMessage::ServerSetup {
                params: Params::new()
            }),
            vec![0x21, 0x00, 0x01, 0x00]
        );

        // PUBLISH_NAMESPACE: request id, namespace tuple, parameters.
        assert_eq!(
            encoded(&ControlMessage::PublishNamespace {
                id: 0,
                namespace: namespace(),
                params: Params::new(),
            }),
            vec![0x06, 0x00, 0x06, 0x00, 0x01, 0x02, b'n', b's', 0x00]
        );
    }

    #[test]
    fn every_message_round_trips() {
        let mut params = Params::new();
        params.set_int(2, 100);
        params.set_bytes(3, b"token".to_vec());

        for msg in [
            ControlMessage::ClientSetup {
                params: params.clone(),
            },
            ControlMessage::ServerSetup {
                params: params.clone(),
            },
            ControlMessage::RequestUpdate {
                id: 2,
                existing_request_id: 0,
                params: params.clone(),
            },
            ControlMessage::Subscribe {
                id: 4,
                namespace: TrackNamespace::from_path("live/cam"),
                track_name: TrackName::new("1.m4s"),
                params: params.clone(),
            },
            ControlMessage::SubscribeOk {
                id: 4,
                track_alias: 4,
                params: params.clone(),
                extensions: Params(vec![(0, super::super::coding::ParamValue::Int(7))]),
            },
            ControlMessage::RequestError {
                id: 4,
                error_code: request_error_code::DOES_NOT_EXIST,
                retry_interval: 0,
                reason: String::from("no such track"),
            },
            ControlMessage::PublishNamespace {
                id: 0,
                namespace: TrackNamespace::from_path("live/cam"),
                params: params.clone(),
            },
            ControlMessage::RequestOk {
                id: 0,
                params: params.clone(),
            },
            ControlMessage::Namespace {
                suffix: TrackNamespacePrefix(vec![b"cam".to_vec()]),
            },
            ControlMessage::NamespaceDone {
                suffix: TrackNamespacePrefix(vec![b"cam".to_vec()]),
            },
            ControlMessage::PublishNamespaceDone { id: 0 },
            ControlMessage::Unsubscribe { id: 4 },
            ControlMessage::PublishDone {
                id: 4,
                status_code: publish_done_code::TRACK_ENDED,
                stream_count: 12,
                reason: String::from("eos"),
            },
            ControlMessage::PublishNamespaceCancel {
                id: 0,
                error_code: 1,
                reason: String::from("expired"),
            },
            ControlMessage::TrackStatus {
                id: 6,
                namespace: TrackNamespace::from_path("live/cam"),
                track_name: TrackName::new("1.m4s"),
                params: params.clone(),
            },
            ControlMessage::GoAway {
                uri: String::from("https://relay.example/2"),
            },
            ControlMessage::SubscribeNamespace {
                id: 8,
                prefix: TrackNamespacePrefix(vec![b"live".to_vec()]),
                options: SubscribeNamespaceOptions::Namespace,
                params: params.clone(),
            },
            ControlMessage::MaxRequestId { request_id: 100 },
            ControlMessage::RequestsBlocked {
                max_request_id: 100,
            },
            ControlMessage::Fetch {
                id: 10,
                fetch_type: FetchType::RelativeJoining,
                standalone: None,
                joining: Some(JoiningFetch {
                    joining_request_id: 4,
                    joining_start: 2,
                }),
                params: params.clone(),
            },
            ControlMessage::FetchCancel { id: 10 },
            ControlMessage::FetchOk {
                id: 10,
                end_of_track: true,
                end: Location {
                    group_id: 3,
                    object_id: 9,
                },
                params: params.clone(),
                extensions: Params::new(),
            },
            ControlMessage::Publish {
                id: 12,
                namespace: TrackNamespace::from_path("live/cam"),
                track_name: TrackName::new("1.m4s"),
                track_alias: 12,
                params: params.clone(),
                extensions: Params::new(),
            },
            ControlMessage::PublishOk {
                id: 12,
                params: params.clone(),
            },
        ] {
            round_trip(msg);
        }
    }

    #[test]
    fn a_truncated_frame_is_incomplete_and_a_short_payload_is_malformed() {
        let bytes = encoded(&ControlMessage::Subscribe {
            id: 0,
            namespace: namespace(),
            track_name: TrackName::new("t"),
            params: Params::new(),
        });
        for cut in 0..bytes.len() {
            assert_eq!(
                ControlMessage::decode(&bytes[..cut]).map(|(_, n)| n),
                Err(MoqtError::Incomplete),
                "prefix of {cut} bytes is a partial frame"
            );
        }

        // A declared length longer than the payload the message needs leaves
        // bytes over: a protocol violation, not a silent accept.
        let mut padded = bytes.clone();
        padded[2] += 1;
        padded.push(0x00);
        assert_eq!(
            ControlMessage::decode(&padded).map(|(_, n)| n),
            Err(MoqtError::Malformed)
        );
    }

    #[test]
    fn unknown_message_types_and_reserved_values_are_refused() {
        assert_eq!(
            ControlMessage::decode(&[0x40, 0x99, 0x00, 0x00]).map(|(_, n)| n),
            Err(MoqtError::Malformed),
            "an unassigned type number is a protocol violation"
        );
        // SUBSCRIBE_NAMESPACE with an out-of-range option value.
        assert_eq!(
            ControlMessage::decode(&[0x11, 0x00, 0x04, 0x00, 0x00, 0x09, 0x00]).map(|(_, n)| n),
            Err(MoqtError::Malformed)
        );
        // FETCH with an unassigned fetch type.
        assert_eq!(
            ControlMessage::decode(&[0x16, 0x00, 0x02, 0x00, 0x09]).map(|(_, n)| n),
            Err(MoqtError::Malformed)
        );
    }

    #[test]
    fn a_namespace_count_that_overruns_the_payload_fails_the_parse() {
        // SUBSCRIBE whose namespace claims 32 fields inside a 4-byte payload.
        let payload = vec![0x00, 0x20, 0x01, b'a'];
        assert_eq!(
            ControlMessage::decode_payload(msg_type::SUBSCRIBE, &payload),
            Err(MoqtError::Incomplete)
        );
        // ...and one that claims more fields than the draft allows at all.
        let payload = vec![0x00, 0x40, 0xff, 0x01, b'a'];
        assert_eq!(
            ControlMessage::decode_payload(msg_type::SUBSCRIBE, &payload),
            Err(MoqtError::Malformed)
        );
    }
}
