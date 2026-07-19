//! LiveKit signalling seam for the str0m WebRTC stack: the WebSocket + protobuf
//! layer that sits on top of the T1 engine, the way gst-plugins-rs layers a
//! signaller over `webrtcbin`. This module is transport + protocol only; the
//! media PeerConnection is a plain str0m `Rtc` driven by the element that owns
//! this signaller (see [`crate::livekitsink`]).
//!
//! LiveKit speaks length-agnostic protobuf envelopes over one WebSocket: the
//! client always sends a `SignalRequest`, the server replies with a
//! `SignalResponse`. Rather than pull the `livekit-protocol` crate (prost + its
//! whole room/egress/ingress/agent tree) we hand-roll encode/decode for just the
//! publish subset, matching how the `srt` / `rtmp` / `hls` protocol cores are
//! hand-rolled sans-IO here. The protobuf wire format is plain tag/length TLV.
//!
//! LiveKit uses two PeerConnections per client: the publisher PC is
//! client-offered (`SignalRequest.offer`) and the subscriber PC is server-offered
//! (`SignalResponse.offer`), disambiguated by direction plus the
//! [`SignalTarget`] tag on trickled candidates. This publish milestone drives the
//! publisher PC; the same envelopes host the subscriber (answerer) role a later
//! ingest milestone needs.
//!
//! Auth is a hand-rolled HS256 JWT (LiveKit access token): base64url header +
//! claims, HMAC-SHA256 over them with the API secret. See [`mint_token`].

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use base64::Engine as _;
use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

const B64URL: base64::engine::general_purpose::GeneralPurpose =
    base64::engine::general_purpose::URL_SAFE_NO_PAD;

/// The protocol version the client advertises in the `/rtc?protocol=` query. 17
/// matches the current client / rust SDKs; the dev server does not gate on it.
pub const PROTOCOL_VERSION: u32 = 17;

// ---------------------------------------------------------------------------
// Access token (HS256 JWT)
// ---------------------------------------------------------------------------

/// The LiveKit video grant subset this crate needs. Serialized as the token's
/// `video` claim; `false` / empty fields are omitted so the token stays minimal.
#[derive(Debug, Clone, Default)]
pub struct VideoGrant {
    pub room_join: bool,
    pub room: String,
    pub can_publish: bool,
    pub can_subscribe: bool,
    pub can_publish_data: bool,
    /// RoomService admin (ListParticipants etc.); used by the validation harness,
    /// not the publish path.
    pub room_admin: bool,
}

impl VideoGrant {
    /// A publisher grant: join `room`, publish media + data, do not subscribe.
    pub fn publisher(room: impl Into<String>) -> Self {
        Self {
            room_join: true,
            room: room.into(),
            can_publish: true,
            can_subscribe: false,
            can_publish_data: true,
            room_admin: false,
        }
    }

    fn to_json(&self) -> String {
        let mut fields: Vec<String> = Vec::new();
        if self.room_join {
            fields.push("\"roomJoin\":true".to_string());
        }
        if !self.room.is_empty() {
            fields.push(format!("\"room\":{}", json_string(&self.room)));
        }
        // canPublish / canSubscribe default to TRUE server-side when absent, so
        // they are always emitted explicitly: an omitted canSubscribe=false
        // leaves the server in subscriber-primary mode, and a publish-only
        // client that never answers the subscriber offer is killed with
        // JOIN_TIMEOUT 60s in (found live against livekit-server 1.13.4).
        fields.push(format!("\"canPublish\":{}", self.can_publish));
        fields.push(format!("\"canSubscribe\":{}", self.can_subscribe));
        if self.can_publish_data {
            fields.push("\"canPublishData\":true".to_string());
        }
        if self.room_admin {
            fields.push("\"roomAdmin\":true".to_string());
        }
        format!("{{{}}}", fields.join(","))
    }
}

/// Mint a LiveKit access token (HS256 JWT) for `identity`, signed with
/// `api_secret` and issued by `api_key`. `now_unix` is the current time in
/// seconds (caller-supplied so the mint stays pure/testable); the token is valid
/// from `now_unix` for `ttl_secs`.
pub fn mint_token(
    api_key: &str,
    api_secret: &str,
    identity: &str,
    grant: &VideoGrant,
    now_unix: u64,
    ttl_secs: u64,
) -> String {
    let header = B64URL.encode(br#"{"alg":"HS256","typ":"JWT"}"#);
    let claims = format!(
        "{{\"iss\":{},\"sub\":{},\"nbf\":{},\"exp\":{},\"video\":{}}}",
        json_string(api_key),
        json_string(identity),
        now_unix,
        now_unix + ttl_secs,
        grant.to_json(),
    );
    let payload = B64URL.encode(claims.as_bytes());
    let signing_input = format!("{header}.{payload}");
    let mut mac =
        HmacSha256::new_from_slice(api_secret.as_bytes()).expect("HMAC accepts any key length");
    mac.update(signing_input.as_bytes());
    let sig = B64URL.encode(mac.finalize().into_bytes());
    format!("{signing_input}.{sig}")
}

/// Minimal JSON string escaping (quotes, backslash, control chars) for the two
/// caller-controlled strings that reach the token / candidateInit JSON.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// ---------------------------------------------------------------------------
// Protobuf wire primitives (tag/length TLV, no schema/reflection)
// ---------------------------------------------------------------------------

mod pb {
    use alloc::vec::Vec;

    pub(super) fn put_varint(buf: &mut Vec<u8>, mut v: u64) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v != 0 {
                buf.push(b | 0x80);
            } else {
                buf.push(b);
                return;
            }
        }
    }

    fn put_tag(buf: &mut Vec<u8>, field: u32, wire: u32) {
        put_varint(buf, ((field as u64) << 3) | wire as u64);
    }

    pub(super) fn put_len_field(buf: &mut Vec<u8>, field: u32, data: &[u8]) {
        put_tag(buf, field, 2);
        put_varint(buf, data.len() as u64);
        buf.extend_from_slice(data);
    }

    pub(super) fn put_string_field(buf: &mut Vec<u8>, field: u32, s: &str) {
        put_len_field(buf, field, s.as_bytes());
    }

    pub(super) fn put_varint_field(buf: &mut Vec<u8>, field: u32, v: u64) {
        if v == 0 {
            return; // proto3 default: omit zero
        }
        put_tag(buf, field, 0);
        put_varint(buf, v);
    }

    /// A signed int64 field (protobuf encodes it as a plain varint of the two's
    /// complement), emitted even when zero because callers use it for a real
    /// timestamp payload.
    pub(super) fn put_int64_field(buf: &mut Vec<u8>, field: u32, v: i64) {
        put_tag(buf, field, 0);
        put_varint(buf, v as u64);
    }

    fn read_varint(buf: &[u8], pos: &mut usize) -> Option<u64> {
        let mut result = 0u64;
        let mut shift = 0u32;
        loop {
            let b = *buf.get(*pos)?;
            *pos += 1;
            result |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Some(result);
            }
            shift += 7;
            if shift >= 64 {
                return None;
            }
        }
    }

    /// One decoded protobuf field of a wire type this subset reads: its number
    /// plus the payload. Fixed 32/64-bit fields are not among the messages we
    /// decode, so `next_field` skips them rather than surface a dead variant.
    pub(super) enum Field<'a> {
        Varint(u32, u64),
        Len(u32, &'a [u8]),
    }

    /// Read the next varint / length-delimited field from `buf` at `pos`,
    /// skipping any fixed-width fields, and advancing past whatever it reads.
    /// `None` at end of buffer or on a malformed field (attacker-controlled
    /// input: never panic).
    pub(super) fn next_field<'a>(buf: &'a [u8], pos: &mut usize) -> Option<Field<'a>> {
        loop {
            if *pos >= buf.len() {
                return None;
            }
            let tag = read_varint(buf, pos)?;
            let field = (tag >> 3) as u32;
            let wire = (tag & 0x7) as u32;
            match wire {
                0 => return Some(Field::Varint(field, read_varint(buf, pos)?)),
                2 => {
                    let len = read_varint(buf, pos)? as usize;
                    let end = pos.checked_add(len)?;
                    let slice = buf.get(*pos..end)?;
                    *pos = end;
                    return Some(Field::Len(field, slice));
                }
                // Skip 64-bit / 32-bit fixed fields (none of the decoded messages
                // use them); loop to the next field.
                1 => *pos = pos.checked_add(8).filter(|e| *e <= buf.len())?,
                5 => *pos = pos.checked_add(4).filter(|e| *e <= buf.len())?,
                _ => return None,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Typed messages (the publish subset of livekit_rtc.proto)
// ---------------------------------------------------------------------------

/// Which PeerConnection a trickled candidate / SDP applies to
/// (livekit_rtc.proto `SignalTarget`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalTarget {
    Publisher,
    Subscriber,
}

impl SignalTarget {
    fn wire(self) -> u64 {
        match self {
            SignalTarget::Publisher => 0,
            SignalTarget::Subscriber => 1,
        }
    }
    fn from_wire(v: u64) -> Self {
        match v {
            1 => SignalTarget::Subscriber,
            _ => SignalTarget::Publisher,
        }
    }
}

/// LiveKit track kind (livekit_models.proto `TrackType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackType {
    Audio,
    Video,
}

impl TrackType {
    fn wire(self) -> u64 {
        match self {
            TrackType::Audio => 0,
            TrackType::Video => 1,
        }
    }
}

/// LiveKit track source (livekit_models.proto `TrackSource`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackSource {
    Camera,
    Microphone,
}

impl TrackSource {
    fn wire(self) -> u64 {
        match self {
            TrackSource::Camera => 1,
            TrackSource::Microphone => 2,
        }
    }
}

/// LiveKit video quality tier (livekit_models.proto `VideoQuality`). A simulcast
/// layer's rid maps to a tier by the server's convention (`q` = low, `h` = mid,
/// `f` = high), so the SFU knows which quality each published rid carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoQuality {
    Low,
    Medium,
    High,
}

impl VideoQuality {
    fn wire(self) -> u64 {
        match self {
            VideoQuality::Low => 0,
            VideoQuality::Medium => 1,
            VideoQuality::High => 2,
        }
    }

    /// The tier for a simulcast rid, matching the server's rid->layer table.
    pub fn for_rid(rid: &str) -> VideoQuality {
        match rid {
            "f" => VideoQuality::High,
            "h" => VideoQuality::Medium,
            _ => VideoQuality::Low,
        }
    }
}

/// One published simulcast layer's metadata (livekit_models.proto `VideoLayer`):
/// the quality tier and the layer's resolution. Announced in [`AddTrackRequest`]
/// so the SFU can map each rid to a quality and forward the right layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoLayer {
    pub quality: VideoQuality,
    pub width: u32,
    pub height: u32,
}

impl VideoLayer {
    fn encode_into(&self, buf: &mut Vec<u8>) {
        pb::put_varint_field(buf, 1, self.quality.wire());
        pb::put_varint_field(buf, 2, self.width as u64);
        pb::put_varint_field(buf, 3, self.height as u64);
    }
}

/// SDP session description (`type` = "offer" / "answer", plus the SDP blob).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionDescription {
    pub sdp_type: String,
    pub sdp: String,
}

impl SessionDescription {
    fn encode_into(&self, buf: &mut Vec<u8>) {
        pb::put_string_field(buf, 1, &self.sdp_type);
        pb::put_string_field(buf, 2, &self.sdp);
    }

    fn decode(buf: &[u8]) -> Self {
        let mut sdp_type = String::new();
        let mut sdp = String::new();
        let mut pos = 0;
        while let Some(f) = pb::next_field(buf, &mut pos) {
            match f {
                pb::Field::Len(1, v) => sdp_type = String::from_utf8_lossy(v).into_owned(),
                pb::Field::Len(2, v) => sdp = String::from_utf8_lossy(v).into_owned(),
                _ => {}
            }
        }
        SessionDescription { sdp_type, sdp }
    }
}

/// A trickled ICE candidate: `candidate_init` is the JSON-serialized
/// RTCIceCandidateInit, `target` names the PeerConnection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrickleRequest {
    pub candidate_init: String,
    pub target: SignalTarget,
}

impl TrickleRequest {
    fn encode_into(&self, buf: &mut Vec<u8>) {
        pb::put_string_field(buf, 1, &self.candidate_init);
        pb::put_varint_field(buf, 2, self.target.wire());
    }

    fn decode(buf: &[u8]) -> Self {
        let mut candidate_init = String::new();
        let mut target = SignalTarget::Publisher;
        let mut pos = 0;
        while let Some(f) = pb::next_field(buf, &mut pos) {
            match f {
                pb::Field::Len(1, v) => candidate_init = String::from_utf8_lossy(v).into_owned(),
                pb::Field::Varint(2, v) => target = SignalTarget::from_wire(v),
                _ => {}
            }
        }
        TrickleRequest {
            candidate_init,
            target,
        }
    }
}

/// Announce a track to publish (client-id, kind, source, geometry).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddTrackRequest {
    pub cid: String,
    pub name: String,
    pub track_type: TrackType,
    pub width: u32,
    pub height: u32,
    pub source: TrackSource,
    /// Simulcast layers, empty for a single-stream track. Each carries a quality
    /// tier + resolution so the SFU can forward the right layer per subscriber.
    pub layers: Vec<VideoLayer>,
}

impl AddTrackRequest {
    fn encode_into(&self, buf: &mut Vec<u8>) {
        pb::put_string_field(buf, 1, &self.cid);
        pb::put_string_field(buf, 2, &self.name);
        pb::put_varint_field(buf, 3, self.track_type.wire());
        pb::put_varint_field(buf, 4, self.width as u64);
        pb::put_varint_field(buf, 5, self.height as u64);
        pb::put_varint_field(buf, 8, self.source.wire());
        for layer in &self.layers {
            let mut inner = Vec::new();
            layer.encode_into(&mut inner);
            pb::put_len_field(buf, 9, &inner);
        }
    }
}

/// One ICE server from the JoinResponse (STUN/TURN URLs + optional credentials).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IceServer {
    pub urls: Vec<String>,
    pub username: String,
    pub credential: String,
}

impl IceServer {
    fn decode(buf: &[u8]) -> Self {
        let mut s = IceServer::default();
        let mut pos = 0;
        while let Some(f) = pb::next_field(buf, &mut pos) {
            match f {
                pb::Field::Len(1, v) => s.urls.push(String::from_utf8_lossy(v).into_owned()),
                pb::Field::Len(2, v) => s.username = String::from_utf8_lossy(v).into_owned(),
                pb::Field::Len(3, v) => s.credential = String::from_utf8_lossy(v).into_owned(),
                _ => {}
            }
        }
        s
    }
}

/// The publish-relevant slice of JoinResponse.
#[derive(Debug, Clone, Default)]
pub struct JoinResponse {
    pub ice_servers: Vec<IceServer>,
    pub subscriber_primary: bool,
    pub ping_interval: i32,
    pub ping_timeout: i32,
    pub participant_sid: String,
    pub participant_identity: String,
}

impl JoinResponse {
    fn decode(buf: &[u8]) -> Self {
        let mut jr = JoinResponse::default();
        let mut pos = 0;
        while let Some(f) = pb::next_field(buf, &mut pos) {
            match f {
                // participant (ParticipantInfo): sid=1, identity=2
                pb::Field::Len(2, v) => {
                    let mut p = 0;
                    while let Some(pf) = pb::next_field(v, &mut p) {
                        match pf {
                            pb::Field::Len(1, s) => {
                                jr.participant_sid = String::from_utf8_lossy(s).into_owned()
                            }
                            pb::Field::Len(2, s) => {
                                jr.participant_identity = String::from_utf8_lossy(s).into_owned()
                            }
                            _ => {}
                        }
                    }
                }
                pb::Field::Len(5, v) => jr.ice_servers.push(IceServer::decode(v)),
                pb::Field::Varint(6, v) => jr.subscriber_primary = v != 0,
                pb::Field::Varint(10, v) => jr.ping_timeout = v as i32,
                pb::Field::Varint(11, v) => jr.ping_interval = v as i32,
                _ => {}
            }
        }
        jr
    }
}

/// Server confirmation that a track was published, echoing the request `cid` and
/// carrying the assigned track SID.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TrackPublishedResponse {
    pub cid: String,
    pub track_sid: String,
}

impl TrackPublishedResponse {
    fn decode(buf: &[u8]) -> Self {
        let mut r = TrackPublishedResponse::default();
        let mut pos = 0;
        while let Some(f) = pb::next_field(buf, &mut pos) {
            match f {
                pb::Field::Len(1, v) => r.cid = String::from_utf8_lossy(v).into_owned(),
                // track (TrackInfo): sid=1
                pb::Field::Len(2, v) => {
                    let mut p = 0;
                    while let Some(tf) = pb::next_field(v, &mut p) {
                        if let pb::Field::Len(1, s) = tf {
                            r.track_sid = String::from_utf8_lossy(s).into_owned();
                        }
                    }
                }
                _ => {}
            }
        }
        r
    }
}

/// A client -> server signalling message (the publish subset of SignalRequest).
#[derive(Debug, Clone)]
pub enum SignalRequest {
    Offer(SessionDescription),
    Answer(SessionDescription),
    Trickle(TrickleRequest),
    AddTrack(AddTrackRequest),
    /// legacy int64 ping (unix-ms); the dev server pongs it on field 18.
    Ping(i64),
    Leave,
}

impl SignalRequest {
    /// Encode to the protobuf envelope bytes carried in one binary WS message.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        match self {
            SignalRequest::Offer(sd) => {
                let mut inner = Vec::new();
                sd.encode_into(&mut inner);
                pb::put_len_field(&mut buf, 1, &inner);
            }
            SignalRequest::Answer(sd) => {
                let mut inner = Vec::new();
                sd.encode_into(&mut inner);
                pb::put_len_field(&mut buf, 2, &inner);
            }
            SignalRequest::Trickle(t) => {
                let mut inner = Vec::new();
                t.encode_into(&mut inner);
                pb::put_len_field(&mut buf, 3, &inner);
            }
            SignalRequest::AddTrack(a) => {
                let mut inner = Vec::new();
                a.encode_into(&mut inner);
                pb::put_len_field(&mut buf, 4, &inner);
            }
            SignalRequest::Ping(ts) => pb::put_int64_field(&mut buf, 14, *ts),
            SignalRequest::Leave => {
                // LeaveRequest with all defaults (empty message body).
                pb::put_len_field(&mut buf, 8, &[]);
            }
        }
        buf
    }
}

/// A server -> client signalling message (the publish subset of SignalResponse).
/// Unrecognized envelopes decode to [`SignalResponse::Other`] and are ignored.
#[derive(Debug, Clone)]
pub enum SignalResponse {
    Join(JoinResponse),
    Answer(SessionDescription),
    Offer(SessionDescription),
    Trickle(TrickleRequest),
    TrackPublished(TrackPublishedResponse),
    Pong(i64),
    Leave,
    Other,
}

impl SignalResponse {
    /// Decode one protobuf envelope. `None` only if the buffer is not a single
    /// well-formed top-level field; unknown message types map to `Other`.
    pub fn decode(buf: &[u8]) -> Option<Self> {
        let mut pos = 0;
        let f = pb::next_field(buf, &mut pos)?;
        Some(match f {
            pb::Field::Len(1, v) => SignalResponse::Join(JoinResponse::decode(v)),
            pb::Field::Len(2, v) => SignalResponse::Answer(SessionDescription::decode(v)),
            pb::Field::Len(3, v) => SignalResponse::Offer(SessionDescription::decode(v)),
            pb::Field::Len(4, v) => SignalResponse::Trickle(TrickleRequest::decode(v)),
            pb::Field::Len(6, v) => {
                SignalResponse::TrackPublished(TrackPublishedResponse::decode(v))
            }
            pb::Field::Len(8, _) => SignalResponse::Leave,
            pb::Field::Varint(18, v) => SignalResponse::Pong(v as i64),
            _ => SignalResponse::Other,
        })
    }
}

/// Build the RTCIceCandidateInit JSON LiveKit expects in a TrickleRequest, from
/// a str0m candidate string and the m-line it belongs to.
pub fn candidate_init_json(candidate: &str, mid: &str, m_line_index: u32) -> String {
    format!(
        "{{\"candidate\":{},\"sdpMid\":{},\"sdpMLineIndex\":{}}}",
        json_string(candidate),
        json_string(mid),
        m_line_index,
    )
}

/// Extract the raw `candidate:` string from a received candidateInit JSON. Best
/// effort string scan (no JSON dep); returns `None` if the field is absent.
pub fn candidate_from_init_json(json: &str) -> Option<String> {
    let key = "\"candidate\"";
    let start = json.find(key)? + key.len();
    let rest = &json[start..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    let mut chars = after.char_indices();
    if chars.next()?.1 != '"' {
        return None;
    }
    let mut out = String::new();
    let mut escaped = false;
    for (_, c) in chars {
        if escaped {
            out.push(c);
            escaped = false;
        } else if c == '\\' {
            escaped = true;
        } else if c == '"' {
            return Some(out);
        } else {
            out.push(c);
        }
    }
    None
}

/// Build the LiveKit signalling WebSocket URL from a base ws(s):// host URL, the
/// access token, and `auto_subscribe`. Appends the `/rtc` path and the query.
pub fn signal_ws_url(base_url: &str, token: &str, auto_subscribe: bool) -> String {
    let base = base_url.trim_end_matches('/');
    format!(
        "{base}/rtc?access_token={token}&auto_subscribe={}&protocol={}&sdk=go",
        auto_subscribe, PROTOCOL_VERSION,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mint_token_matches_known_hs256_vector() {
        // Deterministic inputs -> a stable token whose parts decode back to the
        // claims, and whose signature verifies under the secret.
        let grant = VideoGrant::publisher("myroom");
        let tok = mint_token(
            "devkey",
            "secret",
            "publisher-1",
            &grant,
            1_700_000_000,
            3600,
        );
        let parts: Vec<&str> = tok.split('.').collect();
        assert_eq!(parts.len(), 3, "header.payload.signature");

        let header = B64URL.decode(parts[0]).unwrap();
        assert_eq!(header, br#"{"alg":"HS256","typ":"JWT"}"#);

        let claims = String::from_utf8(B64URL.decode(parts[1]).unwrap()).unwrap();
        assert!(claims.contains("\"iss\":\"devkey\""), "{claims}");
        assert!(claims.contains("\"sub\":\"publisher-1\""), "{claims}");
        assert!(claims.contains("\"nbf\":1700000000"), "{claims}");
        assert!(claims.contains("\"exp\":1700003600"), "{claims}");
        assert!(claims.contains("\"roomJoin\":true"), "{claims}");
        assert!(claims.contains("\"room\":\"myroom\""), "{claims}");
        assert!(claims.contains("\"canPublish\":true"), "{claims}");

        // Recompute the signature the way a verifier would.
        let mut mac = HmacSha256::new_from_slice(b"secret").unwrap();
        mac.update(format!("{}.{}", parts[0], parts[1]).as_bytes());
        let expected = B64URL.encode(mac.finalize().into_bytes());
        assert_eq!(parts[2], expected);
    }

    #[test]
    fn mint_token_is_stable_for_fixed_inputs() {
        let grant = VideoGrant::publisher("r");
        let a = mint_token("k", "s", "id", &grant, 1000, 60);
        let b = mint_token("k", "s", "id", &grant, 1000, 60);
        assert_eq!(a, b, "same inputs must yield the same token");
    }

    #[test]
    fn signal_request_offer_round_trips_as_a_response() {
        // Field 1 (offer) in a SignalRequest is field 3 (offer) in a
        // SignalResponse: same SessionDescription encoding, so encode a request
        // offer and confirm the SessionDescription bytes decode back intact.
        let sd = SessionDescription {
            sdp_type: "offer".into(),
            sdp: "v=0\r\no=- 1 1 IN IP4 0.0.0.0\r\n".into(),
        };
        let bytes = SignalRequest::Offer(sd.clone()).encode();
        // Top-level field 1, length-delimited: decode the inner SessionDescription.
        let mut pos = 0;
        let f = pb::next_field(&bytes, &mut pos).unwrap();
        match f {
            pb::Field::Len(1, inner) => {
                assert_eq!(SessionDescription::decode(inner), sd);
            }
            _ => panic!("expected field 1 length-delimited"),
        }
    }

    #[test]
    fn add_track_encodes_expected_fields() {
        let a = AddTrackRequest {
            cid: "vid-cid".into(),
            name: "video".into(),
            track_type: TrackType::Video,
            width: 640,
            height: 480,
            source: TrackSource::Camera,
            layers: Vec::new(),
        };
        let bytes = SignalRequest::AddTrack(a.clone()).encode();
        let mut pos = 0;
        let pb::Field::Len(4, inner) = pb::next_field(&bytes, &mut pos).unwrap() else {
            panic!("expected add_track field 4");
        };
        // Walk the inner AddTrackRequest fields.
        let (mut cid, mut width, mut height, mut ttype, mut source) =
            (String::new(), 0u64, 0u64, 0u64, 0u64);
        let mut p = 0;
        while let Some(f) = pb::next_field(inner, &mut p) {
            match f {
                pb::Field::Len(1, v) => cid = String::from_utf8_lossy(v).into_owned(),
                pb::Field::Varint(3, v) => ttype = v,
                pb::Field::Varint(4, v) => width = v,
                pb::Field::Varint(5, v) => height = v,
                pb::Field::Varint(8, v) => source = v,
                _ => {}
            }
        }
        assert_eq!(cid, "vid-cid");
        assert_eq!(ttype, 1); // VIDEO
        assert_eq!(width, 640);
        assert_eq!(height, 480);
        assert_eq!(source, 1); // CAMERA
    }

    #[test]
    fn add_track_encodes_simulcast_layers() {
        // A two-layer simulcast track: field 9 repeats a VideoLayer per rid, each
        // carrying its quality tier + resolution.
        let a = AddTrackRequest {
            cid: "vid-cid".into(),
            name: "video".into(),
            track_type: TrackType::Video,
            width: 640,
            height: 480,
            source: TrackSource::Camera,
            layers: alloc::vec![
                VideoLayer {
                    quality: VideoQuality::Medium,
                    width: 640,
                    height: 480,
                },
                VideoLayer {
                    quality: VideoQuality::Low,
                    width: 320,
                    height: 240,
                },
            ],
        };
        let bytes = SignalRequest::AddTrack(a).encode();
        let mut pos = 0;
        let pb::Field::Len(4, inner) = pb::next_field(&bytes, &mut pos).unwrap() else {
            panic!("expected add_track field 4");
        };
        // Collect the two VideoLayer submessages (field 9) and decode their
        // quality + width.
        let mut layers = Vec::new();
        let mut p = 0;
        while let Some(f) = pb::next_field(inner, &mut p) {
            if let pb::Field::Len(9, sub) = f {
                let (mut quality, mut width) = (0u64, 0u64);
                let mut q = 0;
                while let Some(lf) = pb::next_field(sub, &mut q) {
                    match lf {
                        pb::Field::Varint(1, v) => quality = v,
                        pb::Field::Varint(2, v) => width = v,
                        _ => {}
                    }
                }
                layers.push((quality, width));
            }
        }
        assert_eq!(layers, alloc::vec![(1, 640), (0, 320)]);
    }

    #[test]
    fn video_quality_maps_from_rid() {
        assert_eq!(VideoQuality::for_rid("q"), VideoQuality::Low);
        assert_eq!(VideoQuality::for_rid("h"), VideoQuality::Medium);
        assert_eq!(VideoQuality::for_rid("f"), VideoQuality::High);
    }

    #[test]
    fn decodes_join_response_with_ice_and_ping() {
        // Hand-build a SignalResponse{ join: JoinResponse{ participant, ice, ping } }.
        let mut participant = Vec::new();
        pb::put_string_field(&mut participant, 1, "PA_sid");
        pb::put_string_field(&mut participant, 2, "publisher-1");

        let mut ice = Vec::new();
        pb::put_string_field(&mut ice, 1, "stun:stun.l.google.com:19302");

        let mut join = Vec::new();
        pb::put_len_field(&mut join, 2, &participant);
        pb::put_len_field(&mut join, 5, &ice);
        pb::put_varint_field(&mut join, 6, 1); // subscriber_primary
        pb::put_varint_field(&mut join, 11, 3); // ping_interval

        let mut env = Vec::new();
        pb::put_len_field(&mut env, 1, &join);

        let msg = SignalResponse::decode(&env).unwrap();
        let SignalResponse::Join(jr) = msg else {
            panic!("expected join");
        };
        assert_eq!(jr.participant_identity, "publisher-1");
        assert_eq!(jr.participant_sid, "PA_sid");
        assert_eq!(jr.ice_servers.len(), 1);
        assert_eq!(
            jr.ice_servers[0].urls,
            alloc::vec!["stun:stun.l.google.com:19302"]
        );
        assert!(jr.subscriber_primary);
        assert_eq!(jr.ping_interval, 3);
    }

    #[test]
    fn decodes_answer_and_track_published_and_pong() {
        // answer (field 2)
        let mut sd = Vec::new();
        SessionDescription {
            sdp_type: "answer".into(),
            sdp: "v=0\r\n".into(),
        }
        .encode_into(&mut sd);
        let mut env = Vec::new();
        pb::put_len_field(&mut env, 2, &sd);
        assert!(matches!(
            SignalResponse::decode(&env),
            Some(SignalResponse::Answer(s)) if s.sdp_type == "answer"
        ));

        // track_published (field 6) with a nested TrackInfo.sid
        let mut track = Vec::new();
        pb::put_string_field(&mut track, 1, "TR_xyz");
        let mut tp = Vec::new();
        pb::put_string_field(&mut tp, 1, "vid-cid");
        pb::put_len_field(&mut tp, 2, &track);
        let mut env2 = Vec::new();
        pb::put_len_field(&mut env2, 6, &tp);
        let Some(SignalResponse::TrackPublished(r)) = SignalResponse::decode(&env2) else {
            panic!("expected track_published");
        };
        assert_eq!(r.cid, "vid-cid");
        assert_eq!(r.track_sid, "TR_xyz");

        // pong (field 18, int64)
        let mut env3 = Vec::new();
        pb::put_int64_field(&mut env3, 18, 42);
        assert!(matches!(
            SignalResponse::decode(&env3),
            Some(SignalResponse::Pong(42))
        ));
    }

    #[test]
    fn trickle_round_trips_with_target() {
        let t = TrickleRequest {
            candidate_init: candidate_init_json(
                "candidate:1 1 udp 2113 10.0.0.2 5000 typ host",
                "0",
                0,
            ),
            target: SignalTarget::Publisher,
        };
        let mut inner = Vec::new();
        t.encode_into(&mut inner);
        let decoded = TrickleRequest::decode(&inner);
        assert_eq!(decoded.target, SignalTarget::Publisher);
        let cand = candidate_from_init_json(&decoded.candidate_init).unwrap();
        assert_eq!(cand, "candidate:1 1 udp 2113 10.0.0.2 5000 typ host");
    }

    #[test]
    fn candidate_from_init_json_handles_absent_field() {
        assert!(candidate_from_init_json("{\"sdpMid\":\"0\"}").is_none());
    }

    #[test]
    fn ws_url_has_path_and_query() {
        let u = signal_ws_url("ws://localhost:7880/", "TOKEN", false);
        assert_eq!(
            u,
            format!("ws://localhost:7880/rtc?access_token=TOKEN&auto_subscribe=false&protocol={PROTOCOL_VERSION}&sdk=go")
        );
    }
}
