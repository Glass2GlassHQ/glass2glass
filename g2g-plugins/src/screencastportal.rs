//! xdg-desktop-portal `org.freedesktop.portal.ScreenCast` handshake: the D-Bus
//! half of `pipewirevideosrc portal=true`, so a launch line can capture the
//! screen on a stock Wayland desktop without the app driving D-Bus itself.
//!
//! The dance is `CreateSession` -> `SelectSources` -> `Start` ->
//! `OpenPipeWireRemote`, ending in a private PipeWire remote fd plus the node id
//! the compositor granted. The first three are asynchronous in the portal sense:
//! the method returns an `org.freedesktop.portal.Request` object path and the
//! result arrives later as a `Response` signal on that object. The portal can
//! emit that signal before the method call returns, so each step derives the
//! path from its own `handle_token`, subscribes there, and only then calls.
//!
//! `Start` is the step a human consents to, so every wait is bounded by a
//! deadline. The signal wait runs on a helper thread; on a timeout, closing the
//! bus connection ends the signal stream and so the thread.
//!
//! Nothing in a `Response` is trusted: a missing key, a wrong variant type or an
//! empty stream list fails the handshake with a named error.

use alloc::format;
use alloc::string::{String, ToString};
use core::fmt;
use core::time::Duration;
use std::collections::HashMap;
use std::os::fd::OwnedFd;

use zbus::blocking::proxy::SignalIterator;
use zbus::blocking::{Connection, Proxy};
use zbus::zvariant::{Array, ObjectPath, OwnedObjectPath, OwnedValue, Structure, Value};

const PORTAL_BUS_NAME: &str = "org.freedesktop.portal.Desktop";
const PORTAL_OBJECT_PATH: &str = "/org/freedesktop/portal/desktop";
const SCREEN_CAST_INTERFACE: &str = "org.freedesktop.portal.ScreenCast";
const REQUEST_INTERFACE: &str = "org.freedesktop.portal.Request";
const RESPONSE_SIGNAL: &str = "Response";
const REQUEST_PATH_PREFIX: &str = "/org/freedesktop/portal/desktop/request/";
/// Prefix of every `handle_token` we hand the portal. It only has to be unique
/// within one bus connection, which is unique in itself, so a counter suffices.
const HANDLE_TOKEN_PREFIX: &str = "g2g";

/// `Response` signal codes (xdg-desktop-portal `org.freedesktop.portal.Request`).
const RESPONSE_SUCCESS: u32 = 0;
const RESPONSE_CANCELLED: u32 = 1;

const SESSION_HANDLE_FIELD: &str = "session_handle";
const STREAMS_FIELD: &str = "streams";
const RESTORE_TOKEN_FIELD: &str = "restore_token";

/// `persist_mode` asking the portal to remember the grant for as long as the
/// application lives and hand back a `restore_token` (2 = "persist until
/// revoked"). Only sent when the caller wants a token.
const PERSIST_MODE_UNTIL_REVOKED: u32 = 2;

/// What the user is asked to share. The portal takes a bitmask; these are the
/// combinations worth a property.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PortalSourceTypes {
    /// A whole output.
    #[default]
    Monitor,
    /// A single application window.
    Window,
    /// Let the user pick either.
    Any,
}

impl PortalSourceTypes {
    const MONITOR_BIT: u32 = 1;
    const WINDOW_BIT: u32 = 2;

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "monitor" => Some(Self::Monitor),
            "window" => Some(Self::Window),
            "any" => Some(Self::Any),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Monitor => "monitor",
            Self::Window => "window",
            Self::Any => "any",
        }
    }

    fn bits(self) -> u32 {
        match self {
            Self::Monitor => Self::MONITOR_BIT,
            Self::Window => Self::WINDOW_BIT,
            Self::Any => Self::MONITOR_BIT | Self::WINDOW_BIT,
        }
    }
}

/// Why a handshake did not produce a stream.
#[derive(Debug)]
pub enum PortalError {
    /// The session bus, the portal service, or one method call was unreachable.
    Bus(String),
    /// The bus gave us a unique name an object path cannot be built from.
    SenderName(String),
    /// The portal answered a different `Request` object than the one we asked on.
    RequestPathMismatch { expected: String, returned: String },
    /// The user declined the share.
    Cancelled,
    /// The portal ended the request some other way (code 2 and up).
    Refused(u32),
    /// Nobody answered within the deadline (usually an unattended consent dialog).
    TimedOut(Duration),
    /// The signal stream ended without a response.
    NoResponse,
    /// A `Response` result the step needs is absent.
    MissingField(&'static str),
    /// A `Response` result carries a type this step cannot read.
    WrongType(&'static str),
    /// `Start` succeeded but granted no stream.
    NoStreams,
}

impl fmt::Display for PortalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bus(detail) => write!(f, "screencast portal is unreachable: {detail}"),
            Self::SenderName(name) => write!(f, "unusable D-Bus unique name {name:?}"),
            Self::RequestPathMismatch { expected, returned } => write!(
                f,
                "portal answered request {returned:?}, expected {expected:?}"
            ),
            Self::Cancelled => write!(f, "the screen share was declined"),
            Self::Refused(code) => write!(f, "the portal ended the request (response {code})"),
            Self::TimedOut(after) => write!(
                f,
                "no portal response within {} s (nobody answered the consent dialog?)",
                after.as_secs()
            ),
            Self::NoResponse => write!(f, "the portal connection ended before it responded"),
            Self::MissingField(key) => write!(f, "portal response has no {key:?}"),
            Self::WrongType(key) => write!(f, "portal response field {key:?} has the wrong type"),
            Self::NoStreams => write!(f, "the portal granted no stream"),
        }
    }
}

/// What the caller wants shared.
#[derive(Debug, Clone)]
pub struct PortalRequest {
    pub source_types: PortalSourceTypes,
    /// A token from an earlier grant, to re-open it without asking again. An
    /// unknown or stale token makes the portal ask normally.
    pub restore_token: Option<String>,
    /// How long to wait for each `Response`, the consent dialog included.
    pub timeout: Duration,
}

/// A granted screen share: the portal's own PipeWire remote plus the node on it.
#[derive(Debug)]
pub struct PortalScreenCast {
    pub remote_fd: OwnedFd,
    pub node_id: u32,
    /// Present when the portal persisted the grant; feeding it back into
    /// [`PortalRequest::restore_token`] skips the dialog next time.
    pub restore_token: Option<String>,
}

/// Run the whole handshake and return the granted stream.
pub fn open_screen_cast(request: &PortalRequest) -> Result<PortalScreenCast, PortalError> {
    let connection = Connection::session().map_err(bus_error)?;
    let unique_name = connection
        .unique_name()
        .ok_or_else(|| PortalError::SenderName(String::new()))?
        .to_string();
    let sender_token = sender_token(&unique_name)?;
    let screen_cast = Proxy::new_owned(
        connection.clone(),
        PORTAL_BUS_NAME,
        PORTAL_OBJECT_PATH,
        SCREEN_CAST_INTERFACE,
    )
    .map_err(bus_error)?;

    let create_token = handle_token(0);
    let session_token = handle_token(1);
    let mut options = HashMap::new();
    options.insert("handle_token", Value::from(create_token.as_str()));
    options.insert("session_handle_token", Value::from(session_token.as_str()));
    let results = call_request(
        &connection,
        &sender_token,
        &create_token,
        request.timeout,
        || screen_cast.call("CreateSession", &(options,)),
    )?;
    let session_handle = string_field(&results, SESSION_HANDLE_FIELD)?;
    let session_path = ObjectPath::try_from(session_handle)
        .map_err(|_| PortalError::WrongType(SESSION_HANDLE_FIELD))?;

    let select_token = handle_token(2);
    let mut options = HashMap::new();
    options.insert("handle_token", Value::from(select_token.as_str()));
    options.insert("types", Value::from(request.source_types.bits()));
    options.insert("multiple", Value::from(false));
    // the portal only hands back a restore token when persistence is asked for,
    // so ask every time: that is what makes the token property usable at all
    options.insert("persist_mode", Value::from(PERSIST_MODE_UNTIL_REVOKED));
    if let Some(token) = request.restore_token.as_deref() {
        options.insert("restore_token", Value::from(token));
    }
    call_request(
        &connection,
        &sender_token,
        &select_token,
        request.timeout,
        || screen_cast.call("SelectSources", &(&session_path, options)),
    )?;

    let start_token = handle_token(3);
    let mut options = HashMap::new();
    options.insert("handle_token", Value::from(start_token.as_str()));
    let results = call_request(
        &connection,
        &sender_token,
        &start_token,
        request.timeout,
        || screen_cast.call("Start", &(&session_path, "", options)),
    )?;
    let node_id = first_stream_node_id(&results)?;
    let restore_token = optional_string_field(&results, RESTORE_TOKEN_FIELD);

    let options: HashMap<&str, Value> = HashMap::new();
    let remote_fd: zbus::zvariant::OwnedFd = screen_cast
        .call("OpenPipeWireRemote", &(&session_path, options))
        .map_err(bus_error)?;

    Ok(PortalScreenCast {
        remote_fd: OwnedFd::from(remote_fd),
        node_id,
        restore_token,
    })
}

fn bus_error(error: zbus::Error) -> PortalError {
    PortalError::Bus(error.to_string())
}

/// The `handle_token` of the n-th step. Only has to be unique per connection.
fn handle_token(step: u32) -> String {
    format!("{HANDLE_TOKEN_PREFIX}{step}")
}

/// The bus unique name (`:1.42`) as it appears inside a `Request` object path:
/// no leading colon and dots become underscores, since an object path element
/// is `[A-Za-z0-9_]`.
fn sender_token(unique_name: &str) -> Result<String, PortalError> {
    let bare = unique_name
        .strip_prefix(':')
        .ok_or_else(|| PortalError::SenderName(unique_name.to_string()))?;
    let usable = !bare.is_empty()
        && bare
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_');
    if !usable {
        return Err(PortalError::SenderName(unique_name.to_string()));
    }
    Ok(bare.replace('.', "_"))
}

/// Where the portal will answer a call made with `handle_token`.
fn request_path(sender_token: &str, handle_token: &str) -> String {
    format!("{REQUEST_PATH_PREFIX}{sender_token}/{handle_token}")
}

/// One portal step: subscribe to the `Response` we can predict the path of, make
/// the call, then wait for that response and unwrap its result dictionary.
fn call_request(
    connection: &Connection,
    sender_token: &str,
    handle_token: &str,
    timeout: Duration,
    call: impl FnOnce() -> zbus::Result<OwnedObjectPath>,
) -> Result<HashMap<String, OwnedValue>, PortalError> {
    let expected = request_path(sender_token, handle_token);
    let request = Proxy::new_owned(
        connection.clone(),
        PORTAL_BUS_NAME,
        expected.clone(),
        REQUEST_INTERFACE,
    )
    .map_err(bus_error)?;
    let signals = request.receive_signal(RESPONSE_SIGNAL).map_err(bus_error)?;

    let returned = call().map_err(bus_error)?;
    if returned.as_str() != expected {
        return Err(PortalError::RequestPathMismatch {
            expected,
            returned: returned.as_str().to_string(),
        });
    }

    let message = wait_for_response(connection, signals, timeout)?;
    let (code, results): (u32, HashMap<String, OwnedValue>) = message
        .body()
        .deserialize()
        .map_err(|_| PortalError::WrongType(RESPONSE_SIGNAL))?;
    response_results(code, results)
}

/// A `Response` body split into "granted" and the reasons it was not.
fn response_results(
    code: u32,
    results: HashMap<String, OwnedValue>,
) -> Result<HashMap<String, OwnedValue>, PortalError> {
    match code {
        RESPONSE_SUCCESS => Ok(results),
        RESPONSE_CANCELLED => Err(PortalError::Cancelled),
        other => Err(PortalError::Refused(other)),
    }
}

/// Block for one `Response`, but never past `timeout`. The iterator has no timed
/// read, so it runs on a helper thread; closing the connection ends its stream,
/// which is what lets that thread finish after we have given up.
fn wait_for_response(
    connection: &Connection,
    signals: SignalIterator<'static>,
    timeout: Duration,
) -> Result<zbus::Message, PortalError> {
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    let waiter = std::thread::Builder::new()
        .name(String::from("g2g-portal-response"))
        .spawn(move || {
            let mut signals = signals;
            let _ = tx.send(signals.next());
        })
        .map_err(|e| PortalError::Bus(e.to_string()))?;

    let outcome = match rx.recv_timeout(timeout) {
        Ok(Some(message)) => Ok(message),
        Ok(None) => Err(PortalError::NoResponse),
        Err(_) => {
            let _ = connection.clone().close();
            Err(PortalError::TimedOut(timeout))
        }
    };
    let _ = waiter.join();
    outcome
}

fn string_field(
    results: &HashMap<String, OwnedValue>,
    key: &'static str,
) -> Result<String, PortalError> {
    let value = results.get(key).ok_or(PortalError::MissingField(key))?;
    let text: &str = value
        .downcast_ref()
        .map_err(|_| PortalError::WrongType(key))?;
    Ok(text.to_string())
}

fn optional_string_field(
    results: &HashMap<String, OwnedValue>,
    key: &'static str,
) -> Option<String> {
    let value = results.get(key)?;
    let text: &str = value.downcast_ref().ok()?;
    Some(text.to_string())
}

/// The node id of the first granted stream. `streams` is `a(ua{sv})`, and only
/// the leading `u` of the first entry matters: `multiple` was false, so the
/// portal grants at most one.
fn first_stream_node_id(results: &HashMap<String, OwnedValue>) -> Result<u32, PortalError> {
    let value = results
        .get(STREAMS_FIELD)
        .ok_or(PortalError::MissingField(STREAMS_FIELD))?;
    let streams: &Array = value
        .downcast_ref()
        .map_err(|_| PortalError::WrongType(STREAMS_FIELD))?;
    let first = streams.first().ok_or(PortalError::NoStreams)?;
    let stream: &Structure = first
        .downcast_ref()
        .map_err(|_| PortalError::WrongType(STREAMS_FIELD))?;
    let node_id = stream
        .fields()
        .first()
        .ok_or(PortalError::WrongType(STREAMS_FIELD))?;
    node_id
        .downcast_ref()
        .map_err(|_| PortalError::WrongType(STREAMS_FIELD))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use zbus::zvariant::Signature;

    fn results(pairs: Vec<(&str, Value<'static>)>) -> HashMap<String, OwnedValue> {
        pairs
            .into_iter()
            .map(|(k, v)| (k.to_string(), OwnedValue::try_from(v).expect("owned value")))
            .collect()
    }

    /// `a(ua{sv})` as the portal sends it, with `node_id` in the leading field.
    fn streams_value(node_ids: &[u32]) -> Value<'static> {
        let signature = Signature::try_from("(ua{sv})").expect("stream signature");
        let mut array = Array::new(&signature);
        for id in node_ids {
            let properties: HashMap<String, Value<'static>> = HashMap::new();
            let entry = zbus::zvariant::StructureBuilder::new()
                .add_field(*id)
                .add_field(properties)
                .build()
                .expect("stream structure");
            array.append(Value::from(entry)).expect("append stream");
        }
        Value::from(array)
    }

    #[test]
    fn the_request_path_is_the_one_the_portal_will_answer_on() {
        let sender = sender_token(":1.427").expect("unique name");
        assert_eq!(sender, "1_427");
        assert_eq!(
            request_path(&sender, &handle_token(3)),
            "/org/freedesktop/portal/desktop/request/1_427/g2g3"
        );
    }

    #[test]
    fn every_step_of_one_handshake_gets_its_own_token() {
        let tokens: Vec<String> = (0..4).map(handle_token).collect();
        let unique: std::collections::BTreeSet<&String> = tokens.iter().collect();
        assert_eq!(unique.len(), tokens.len());
    }

    #[test]
    fn a_unique_name_an_object_path_cannot_carry_is_rejected() {
        for name in ["1.427", "", ":", ":1/427", ":1.4-27"] {
            assert!(
                matches!(sender_token(name), Err(PortalError::SenderName(_))),
                "{name:?} should not make a request path"
            );
        }
    }

    #[test]
    fn source_types_map_to_the_portal_bitmask() {
        assert_eq!(PortalSourceTypes::Monitor.bits(), 1);
        assert_eq!(PortalSourceTypes::Window.bits(), 2);
        assert_eq!(PortalSourceTypes::Any.bits(), 3);
        assert_eq!(
            PortalSourceTypes::from_name("window"),
            Some(PortalSourceTypes::Window)
        );
        assert_eq!(PortalSourceTypes::from_name("screen"), None);
    }

    #[test]
    fn a_declined_share_is_not_a_missing_field() {
        assert!(matches!(
            response_results(RESPONSE_CANCELLED, results(Vec::new())),
            Err(PortalError::Cancelled)
        ));
        assert!(matches!(
            response_results(2, results(Vec::new())),
            Err(PortalError::Refused(2))
        ));
        assert!(response_results(RESPONSE_SUCCESS, results(Vec::new())).is_ok());
    }

    #[test]
    fn the_session_handle_is_read_out_of_a_create_session_response() {
        let ok = results(Vec::from([(
            SESSION_HANDLE_FIELD,
            Value::from("/org/freedesktop/portal/desktop/session/1_427/g2g1"),
        )]));
        assert_eq!(
            string_field(&ok, SESSION_HANDLE_FIELD).expect("session handle"),
            "/org/freedesktop/portal/desktop/session/1_427/g2g1"
        );
    }

    #[test]
    fn a_malformed_create_session_response_fails_instead_of_panicking() {
        let empty = results(Vec::new());
        assert!(matches!(
            string_field(&empty, SESSION_HANDLE_FIELD),
            Err(PortalError::MissingField(SESSION_HANDLE_FIELD))
        ));

        let wrong_type = results(Vec::from([(SESSION_HANDLE_FIELD, Value::from(42u32))]));
        assert!(matches!(
            string_field(&wrong_type, SESSION_HANDLE_FIELD),
            Err(PortalError::WrongType(SESSION_HANDLE_FIELD))
        ));
    }

    #[test]
    fn the_node_id_comes_from_the_first_granted_stream() {
        let granted = results(Vec::from([(STREAMS_FIELD, streams_value(&[93, 94]))]));
        assert_eq!(first_stream_node_id(&granted).expect("node id"), 93);
    }

    #[test]
    fn a_malformed_start_response_fails_instead_of_panicking() {
        let empty = results(Vec::new());
        assert!(matches!(
            first_stream_node_id(&empty),
            Err(PortalError::MissingField(STREAMS_FIELD))
        ));

        let no_streams = results(Vec::from([(STREAMS_FIELD, streams_value(&[]))]));
        assert!(matches!(
            first_stream_node_id(&no_streams),
            Err(PortalError::NoStreams)
        ));

        let not_an_array = results(Vec::from([(STREAMS_FIELD, Value::from(93u32))]));
        assert!(matches!(
            first_stream_node_id(&not_an_array),
            Err(PortalError::WrongType(STREAMS_FIELD))
        ));

        let signature = Signature::try_from("s").expect("signature");
        let mut wrong_elements = Array::new(&signature);
        wrong_elements
            .append(Value::from("not a stream"))
            .expect("append");
        let wrong_elements = results(Vec::from([(STREAMS_FIELD, Value::from(wrong_elements))]));
        assert!(matches!(
            first_stream_node_id(&wrong_elements),
            Err(PortalError::WrongType(STREAMS_FIELD))
        ));
    }

    #[test]
    fn a_restore_token_is_optional_and_typed() {
        let with_token = results(Vec::from([(RESTORE_TOKEN_FIELD, Value::from("abc123"))]));
        assert_eq!(
            optional_string_field(&with_token, RESTORE_TOKEN_FIELD).as_deref(),
            Some("abc123")
        );
        assert!(optional_string_field(&results(Vec::new()), RESTORE_TOKEN_FIELD).is_none());
        let wrong_type = results(Vec::from([(RESTORE_TOKEN_FIELD, Value::from(7u32))]));
        assert!(optional_string_field(&wrong_type, RESTORE_TOKEN_FIELD).is_none());
    }

    #[test]
    fn every_failure_says_what_went_wrong() {
        let rendered = [
            PortalError::Cancelled.to_string(),
            PortalError::TimedOut(Duration::from_secs(60)).to_string(),
            PortalError::MissingField(STREAMS_FIELD).to_string(),
            PortalError::NoStreams.to_string(),
        ];
        assert!(rendered.iter().all(|line| line.len() > 10));
        assert!(rendered[1].contains("60"));
        assert!(rendered[2].contains(STREAMS_FIELD));
    }
}
