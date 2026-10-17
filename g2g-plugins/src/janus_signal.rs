use alloc::format;
use alloc::string::{String, ToString};

use std::time::Duration;

use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{Sink, SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::{Error as WsError, Message};
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

use str0m::Candidate;

use crate::webrtc_util::random_hex_id;

pub const DEFAULT_ENDPOINT: &str = "ws://127.0.0.1:8188";
const JANUS_SUBPROTOCOL: &str = "janus-protocol";
const VIDEOROOM_PLUGIN: &str = "janus.plugin.videoroom";
// janus drops a session after 60 s without traffic
pub(crate) const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(10);
const SIGNAL_TIMEOUT: Duration = Duration::from_secs(20);
const LEAVE_ACK_TIMEOUT: Duration = Duration::from_millis(500);

pub(crate) type Ws = WebSocketStream<MaybeTlsStream<TcpStream>>;
pub(crate) type WsWrite = SplitSink<Ws, Message>;
pub(crate) type WsRead = SplitStream<Ws>;

/// A room or feed id in the JSON shape the server expects.
pub(crate) fn id_value(id: &str, string_ids: bool) -> Option<Value> {
    if string_ids {
        return Some(Value::String(id.to_string()));
    }
    id.parse::<u64>().ok().map(Value::from)
}

/// The core request envelope: kind, a fresh transaction, and the ids a
/// request past `create` / `attach` carries.
fn envelope(
    kind: &str,
    session_id: Option<u64>,
    handle_id: Option<u64>,
    secret: Option<&str>,
) -> Value {
    let mut message = json!({ "janus": kind, "transaction": random_hex_id() });
    if let Some(session_id) = session_id {
        message["session_id"] = json!(session_id);
    }
    if let Some(handle_id) = handle_id {
        message["handle_id"] = json!(handle_id);
    }
    if let Some(secret) = secret {
        message["apisecret"] = json!(secret);
    }
    message
}

/// The attached session + plugin handle every later request names.
#[derive(Debug, Clone)]
pub(crate) struct Signaller {
    session_id: u64,
    handle_id: u64,
    api_secret: Option<String>,
}

impl Signaller {
    fn envelope(&self, kind: &str) -> Value {
        envelope(
            kind,
            Some(self.session_id),
            Some(self.handle_id),
            self.api_secret.as_deref(),
        )
    }

    /// A videoroom plugin request (`body`), with a JSEP when it carries SDP.
    pub(crate) fn plugin_message(&self, body: Value, jsep: Option<Value>) -> Value {
        let mut message = self.envelope("message");
        message["body"] = body;
        if let Some(jsep) = jsep {
            message["jsep"] = jsep;
        }
        message
    }

    pub(crate) fn keepalive(&self) -> Value {
        envelope(
            "keepalive",
            Some(self.session_id),
            None,
            self.api_secret.as_deref(),
        )
    }
}

pub(crate) fn plugin_data(message: &Value) -> &Value {
    &message["plugindata"]["data"]
}

fn success_id(message: &Value) -> Option<u64> {
    (message["janus"] == "success")
        .then(|| message["data"]["id"].as_u64())
        .flatten()
}

/// Why the server ended things, when `message` says so.
pub(crate) fn failure(message: &Value) -> Option<String> {
    match message["janus"].as_str() {
        Some("error") => Some(format!(
            "janus error {}: {}",
            message["error"]["code"], message["error"]["reason"]
        )),
        Some("hangup") => Some(format!("janus hangup: {}", message["reason"])),
        _ => {
            let data = plugin_data(message);
            data.get("error_code")
                .map(|code| format!("videoroom error {code}: {}", data["error"]))
        }
    }
}

pub(crate) async fn send_json<S>(ws: &mut S, message: Value) -> Result<(), String>
where
    S: Sink<Message, Error = WsError> + Unpin,
{
    ws.send(Message::Text(message.to_string()))
        .await
        .map_err(|e| format!("websocket send: {e}"))
}

async fn recv_json(ws: &mut Ws) -> Result<Value, String> {
    loop {
        let frame = ws
            .next()
            .await
            .ok_or("websocket closed")?
            .map_err(|e| format!("websocket: {e}"))?;
        if let Some(message) = parse_frame(&frame)? {
            return Ok(message);
        }
    }
}

/// The JSON of a text or binary frame; control frames give `None`.
pub(crate) fn parse_frame(frame: &Message) -> Result<Option<Value>, String> {
    let bytes: &[u8] = match frame {
        Message::Text(text) => text.as_bytes(),
        Message::Binary(bytes) => bytes,
        Message::Close(_) => return Err("websocket closed".to_string()),
        _ => return Ok(None),
    };
    serde_json::from_slice(bytes)
        .map(Some)
        .map_err(|e| format!("janus sent invalid json: {e}"))
}

/// Read until `accept` yields, failing on a server error or the timeout.
pub(crate) async fn wait_for<T>(
    ws: &mut Ws,
    mut accept: impl FnMut(&Value) -> Option<T>,
) -> Result<T, String> {
    let deadline = tokio::time::Instant::now() + SIGNAL_TIMEOUT;
    loop {
        let message = tokio::time::timeout_at(deadline, recv_json(ws))
            .await
            .map_err(|_| "janus did not answer in time".to_string())??;
        if let Some(reason) = failure(&message) {
            return Err(reason);
        }
        if let Some(value) = accept(&message) {
            return Ok(value);
        }
    }
}

/// A `trickle` event's candidate line, `None` for anything else (including the
/// end-of-candidates marker).
pub(crate) fn trickle_candidate(message: &Value) -> Option<Candidate> {
    if message["janus"] != "trickle" {
        return None;
    }
    let line = message["candidate"]["candidate"].as_str()?;
    Candidate::from_sdp_string(line).ok()
}

/// Tell the room we are gone, wait briefly for its `leaving` (publisher) or
/// `left` (subscriber) reply, and close the socket.
pub(crate) async fn leave(ws_write: &mut WsWrite, ws_read: &mut WsRead, signaller: &Signaller) {
    let leave = signaller.plugin_message(json!({ "request": "leave" }), None);
    if send_json(ws_write, leave).await.is_ok() {
        let deadline = tokio::time::Instant::now() + LEAVE_ACK_TIMEOUT;
        while let Ok(Some(Ok(frame))) = tokio::time::timeout_at(deadline, ws_read.next()).await {
            let acknowledged = matches!(parse_frame(&frame), Ok(Some(m))
                if plugin_data(&m)["leaving"] == "ok" || plugin_data(&m)["left"] == "ok");
            if acknowledged {
                break;
            }
        }
    }
    let _ = ws_write.close().await;
}

pub(crate) async fn open_session(
    endpoint: &str,
    api_secret: Option<&str>,
) -> Result<(Ws, Signaller), String> {
    let mut request = endpoint
        .into_client_request()
        .map_err(|e| format!("janus-endpoint {endpoint}: {e}"))?;
    request.headers_mut().insert(
        "Sec-WebSocket-Protocol",
        JANUS_SUBPROTOCOL
            .parse()
            .map_err(|_| "subprotocol header".to_string())?,
    );
    let (mut ws, _) = tokio::time::timeout(SIGNAL_TIMEOUT, connect_async(request))
        .await
        .map_err(|_| format!("connect to {endpoint} timed out"))?
        .map_err(|e| format!("connect to {endpoint} failed: {e}"))?;

    send_json(&mut ws, envelope("create", None, None, api_secret)).await?;
    let session_id = wait_for(&mut ws, success_id).await?;
    let mut attach = envelope("attach", Some(session_id), None, api_secret);
    attach["plugin"] = json!(VIDEOROOM_PLUGIN);
    send_json(&mut ws, attach).await?;
    let handle_id = wait_for(&mut ws, success_id).await?;
    let signaller = Signaller {
        session_id,
        handle_id,
        api_secret: api_secret.map(str::to_string),
    };
    Ok((ws, signaller))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_follow_the_string_ids_switch() {
        assert_eq!(id_value("1234", false), Some(json!(1234)));
        assert_eq!(id_value("1234", true), Some(json!("1234")));
        assert_eq!(id_value("lobby", false), None);
        assert_eq!(id_value("lobby", true), Some(json!("lobby")));
    }

    #[test]
    fn requests_carry_the_session_handle_and_secret() {
        let signaller = Signaller {
            session_id: 7,
            handle_id: 9,
            api_secret: Some("s3".into()),
        };
        let publish = signaller.plugin_message(
            json!({ "request": "publish" }),
            Some(json!({ "type": "offer", "sdp": "v=0", "trickle": false })),
        );
        assert_eq!(publish["janus"], "message");
        assert_eq!(publish["session_id"], 7);
        assert_eq!(publish["handle_id"], 9);
        assert_eq!(publish["apisecret"], "s3");
        assert_eq!(publish["body"]["request"], "publish");
        assert_eq!(publish["jsep"]["type"], "offer");
        assert_eq!(publish["transaction"].as_str().map(str::len), Some(16));
        let keepalive = signaller.keepalive();
        assert_eq!(keepalive["janus"], "keepalive");
        assert_eq!(keepalive["session_id"], 7);
        assert!(keepalive.get("handle_id").is_none());
        let create = envelope("create", None, None, None);
        assert!(create.get("session_id").is_none());
        assert!(create.get("apisecret").is_none());
    }

    #[test]
    fn server_failures_are_named() {
        assert_eq!(
            failure(
                &json!({ "janus": "error", "error": { "code": 403, "reason": "Unauthorized request" } })
            ),
            Some("janus error 403: \"Unauthorized request\"".to_string())
        );
        assert_eq!(
            failure(&json!({ "janus": "hangup", "reason": "DTLS alert" })),
            Some("janus hangup: \"DTLS alert\"".to_string())
        );
        let room_error = json!({ "janus": "event", "plugindata": { "plugin": VIDEOROOM_PLUGIN,
            "data": { "videoroom": "event", "error_code": 426, "error": "No such room" } } });
        assert_eq!(
            failure(&room_error),
            Some("videoroom error 426: \"No such room\"".to_string())
        );
        assert_eq!(failure(&json!({ "janus": "ack" })), None);
        assert_eq!(failure(&json!({ "janus": "webrtcup", "sender": 9 })), None);
    }

    #[test]
    fn success_and_trickle_messages_parse() {
        assert_eq!(
            success_id(&json!({ "janus": "success", "data": { "id": 42 } })),
            Some(42)
        );
        assert_eq!(success_id(&json!({ "janus": "ack" })), None);
        let trickle = json!({ "janus": "trickle", "sender": 9, "candidate": {
            "candidate": "candidate:1 1 udp 2130706431 192.0.2.1 5000 typ host", "sdpMid": "0", "sdpMLineIndex": 0 } });
        assert!(trickle_candidate(&trickle).is_some());
        let completed = json!({ "janus": "trickle", "candidate": { "completed": true } });
        assert!(trickle_candidate(&completed).is_none());
        assert!(trickle_candidate(&json!({ "janus": "ack" })).is_none());
    }
}
