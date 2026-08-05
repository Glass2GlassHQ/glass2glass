//! MoQ Transport draft-18 session over the M901 WebTransport carrier.
//!
//! Draft-18 restructured the control plane, and this module is where that shows
//! up:
//!
//! - **A pair of unidirectional control streams** (§3.3), one opened by each
//!   peer, each beginning with a single SETUP message. Either peer may send
//!   first, so [`Session18::connect_over`] writes ours before waiting for
//!   theirs.
//! - **One bidirectional stream per request.** A request's response comes back
//!   on the same stream and carries no request id, so the stream *is* the
//!   correlation. [`Session18::open_request`] opens one; [`PeerRequest`]s the
//!   peer opens arrive on a channel, so a request is answered when it lands
//!   rather than when the element next has work.
//! - **Cancellation is a stream reset** (§3.3.2), which is why there is no
//!   UNSUBSCRIBE here.
//!
//! One task accepts every unidirectional stream and dispatches on the type
//! varint that opens it (§3.4): the peer's control stream, a subgroup of
//! objects, a FETCH response, or padding to discard. An unknown type closes the
//! session.

use core::sync::atomic::{AtomicBool, Ordering};

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use tokio::sync::{mpsc, oneshot};
use web_transport_quinn::{RecvStream, SendStream, Session};

use g2g_core::{G2gError, HardwareError};

use crate::remotewtio::wt_err;

use super::super::reassembly::DATA_READ_CHUNK;
// The data plane's shape is not version specific: a subscriber reorders objects
// the same way whichever draft delivered them.
pub use super::super::session::DataEvent;
use super::coding::{setup_option, MoqtError, Params};
use super::data::{
    SubgroupHeader, SubgroupStreamDecoder, UniStreamType, PADDING_DATAGRAM_TYPE,
    PADDING_STREAM_TYPE,
};
use super::datagram::DatagramObject;
use super::message::{session_error_code, ControlMessage};

/// The WebTransport subprotocol that names draft-18 (§3.1).
pub const MOQT_PROTOCOL: &str = "moqt-18";

/// Read size for a control or request stream. Control messages are 16-bit-length
/// bounded, so this only affects how often the reader syscalls.
const CONTROL_READ_CHUNK: usize = 8192;

/// Most a control-stream buffer may hold before the peer is framing garbage: the
/// largest message (type, 16-bit length, payload) plus a little slack.
const MAX_CONTROL_BUFFER: usize = u16::MAX as usize + 64;

/// Most a padding stream may deliver before we stop reading it. Padding carries
/// no application data, so this only keeps a peer from spending our time.
const MAX_PADDING_BYTES: u64 = 16 * 1024 * 1024;

fn protocol_err() -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

fn coding_err(_: MoqtError) -> G2gError {
    protocol_err()
}

/// A request stream the peer opened, with its first message and both halves. The
/// response goes on `tx`; `rx` carries any REQUEST_UPDATE and, by ending, the
/// peer's cancellation.
#[derive(Debug)]
pub struct PeerRequest {
    pub first: ControlMessage,
    pub tx: SendStream,
    pub rx: RecvStream,
}

/// A connected draft-18 session: both SETUPs have been exchanged.
#[derive(Debug)]
pub struct Session18 {
    session: Session,
    control_tx: SendStream,
    /// The Setup Options the peer sent. Unknown ones are kept rather than
    /// refused, which is what §10.3 asks for.
    peer_options: Params,
    requests: Option<mpsc::UnboundedReceiver<PeerRequest>>,
    data: Option<mpsc::UnboundedReceiver<DataEvent>>,
    /// Next request id to allocate. A client uses even ids from 0 (§10.1).
    next_request_id: u64,
    closed: Arc<AtomicBool>,
}

impl Session18 {
    /// Complete the draft-18 handshake over an already dialled WebTransport
    /// session: write our SETUP, then accept the peer's control stream and its
    /// SETUP. `max_object_bytes` bounds a single object on the data plane.
    ///
    /// The caller dials, because the version is negotiated by the WebTransport
    /// subprotocol and so is known only after the CONNECT response.
    pub async fn connect_over(
        session: Session,
        implementation: &str,
        max_object_bytes: usize,
    ) -> Result<Self, G2gError> {
        let mut control_tx = session.open_uni().await.map_err(wt_err)?;
        let mut options = Params::new();
        if !implementation.is_empty() {
            options.set_bytes(
                setup_option::MOQT_IMPLEMENTATION,
                implementation.as_bytes().to_vec(),
            );
        }
        // Over WebTransport the path and authority are the CONNECT URL's, and
        // sending PATH or AUTHORITY there is a session error (§10.3.1).
        write_message(&mut control_tx, &ControlMessage::Setup { options }).await?;

        let closed = Arc::new(AtomicBool::new(false));
        let (request_tx, requests) = mpsc::unbounded_channel();
        let (data_tx, data) = mpsc::unbounded_channel();
        let (setup_tx, setup_rx) = oneshot::channel();

        tokio::spawn(accept_uni(
            session.clone(),
            setup_tx,
            data_tx.clone(),
            Arc::clone(&closed),
            max_object_bytes,
        ));
        tokio::spawn(accept_bi(
            session.clone(),
            request_tx,
            Arc::clone(&closed),
        ));
        tokio::spawn(read_datagrams(
            session.clone(),
            data_tx,
            max_object_bytes,
        ));

        // Our SETUP is already written, so waiting here cannot deadlock against
        // a peer that also waits to send first.
        let peer_options = setup_rx.await.map_err(|_| protocol_err())??;

        Ok(Self {
            session,
            control_tx,
            peer_options,
            requests: Some(requests),
            data: Some(data),
            next_request_id: 0,
            closed,
        })
    }

    /// The Setup Options the peer sent.
    pub fn peer_options(&self) -> &Params {
        &self.peer_options
    }

    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Relaxed)
    }

    /// Take the next client request id: even, from 0, stepping by 2 (§10.1).
    /// Draft-18 dropped MAX_REQUEST_ID, so this never runs out.
    pub fn allocate_request_id(&mut self) -> u64 {
        let id = self.next_request_id;
        self.next_request_id = self.next_request_id.saturating_add(2);
        id
    }

    /// Open a bidirectional stream for one request and write its first message.
    /// The response arrives on the returned read half.
    pub async fn open_request(
        &mut self,
        msg: &ControlMessage,
    ) -> Result<(SendStream, RecvStream), G2gError> {
        if !msg.opens_request_stream() {
            // §3.3 lists the seven messages that may open a request stream;
            // anything else would be a violation at the peer.
            return Err(protocol_err());
        }
        let (mut tx, rx) = self.session.open_bi().await.map_err(wt_err)?;
        write_message(&mut tx, msg).await?;
        Ok((tx, rx))
    }

    /// Take the channel of request streams the peer opens. Call once: a second
    /// reader would race the first.
    pub fn take_requests(&mut self) -> Option<mpsc::UnboundedReceiver<PeerRequest>> {
        self.requests.take()
    }

    /// Take the channel of data-plane events (subgroup objects and datagram
    /// objects, interleaved). Call once.
    pub fn take_data(&mut self) -> Option<mpsc::UnboundedReceiver<DataEvent>> {
        self.data.take()
    }

    /// Open a unidirectional stream and write a subgroup header on it. The
    /// caller then writes object headers and payloads.
    pub async fn open_subgroup(&mut self, header: &SubgroupHeader) -> Result<SendStream, G2gError> {
        let mut bytes = Vec::new();
        header.encode(&mut bytes).map_err(coding_err)?;
        let mut stream = self.session.open_uni().await.map_err(wt_err)?;
        // Smaller publisher priority is sent first, and QUIC send priority runs
        // the other way, so the stream priority is the negated byte.
        if let Some(priority) = header.publisher_priority {
            let _ = stream.set_priority(-i32::from(priority));
        }
        stream.write_all(&bytes).await.map_err(wt_err)?;
        Ok(stream)
    }

    /// Send one datagram object. Fails when the encoded object does not fit the
    /// path MTU, or when the peer accepts no datagrams at all: either way the
    /// caller has to carry the object on a stream instead.
    pub fn send_datagram(&self, object: &DatagramObject) -> Result<(), G2gError> {
        let mut bytes = Vec::new();
        object.encode(&mut bytes).map_err(coding_err)?;
        self.session
            .send_datagram(bytes.into())
            .map_err(wt_err)
            .map(|_| ())
    }

    /// Send a message on our control stream (§10 lists GOAWAY as the only one
    /// after SETUP).
    pub async fn send_control(&mut self, msg: &ControlMessage) -> Result<(), G2gError> {
        write_message(&mut self.control_tx, msg).await
    }

    /// Finish our control stream and close the QUIC connection.
    pub async fn close(&mut self, code: u32, reason: &str) {
        let _ = self.control_tx.finish();
        self.session.close(code, reason.as_bytes());
        self.closed.store(true, Ordering::Relaxed);
    }
}

/// Append a framed control message to a stream.
pub async fn write_message(stream: &mut SendStream, msg: &ControlMessage) -> Result<(), G2gError> {
    let mut bytes = Vec::new();
    msg.encode(&mut bytes).map_err(coding_err)?;
    stream.write_all(&bytes).await.map_err(wt_err)
}

/// Buffered reader over one control or request stream.
#[derive(Debug, Default)]
pub struct MessageReader {
    buf: Vec<u8>,
}

impl MessageReader {
    pub fn new() -> Self {
        Self::default()
    }

    /// The next whole message, or `None` once the stream ends. A framing
    /// violation, or a peer that never completes a message, is an error rather
    /// than something to keep buffering.
    pub async fn next(&mut self, rx: &mut RecvStream) -> Result<Option<ControlMessage>, G2gError> {
        let mut chunk = vec![0u8; CONTROL_READ_CHUNK];
        loop {
            match ControlMessage::decode(&self.buf) {
                Ok((msg, used)) => {
                    self.buf.drain(..used);
                    return Ok(Some(msg));
                }
                Err(MoqtError::Incomplete) => {}
                // A malformed control message is a PROTOCOL_VIOLATION: stop
                // rather than resynchronize on framing we no longer trust.
                Err(MoqtError::Malformed) => return Err(protocol_err()),
            }
            if self.buf.len() > MAX_CONTROL_BUFFER {
                return Err(protocol_err());
            }
            match rx.read(&mut chunk).await {
                Ok(Some(n)) if n > 0 => self.buf.extend_from_slice(&chunk[..n]),
                // A clean finish, a zero read, or a reset: the stream is over.
                // Bytes left over are half a message, which §11.4 makes a
                // violation.
                Ok(_) => {
                    return if self.buf.is_empty() {
                        Ok(None)
                    } else {
                        Err(protocol_err())
                    }
                }
                Err(_) => return Ok(None),
            }
        }
    }
}

/// Accept every bidirectional stream the peer opens and read its first message.
/// A stream that does not begin with one of Table 5's seven "First" messages is
/// a PROTOCOL_VIOLATION (§3.3).
async fn accept_bi(
    session: Session,
    out: mpsc::UnboundedSender<PeerRequest>,
    closed: Arc<AtomicBool>,
) {
    while let Ok((tx, mut rx)) = session.accept_bi().await {
        let mut reader = MessageReader::new();
        let first = match reader.next(&mut rx).await {
            Ok(Some(msg)) => msg,
            // A stream that closes before saying anything cost us nothing.
            Ok(None) => continue,
            Err(_) => break,
        };
        if !first.opens_request_stream() {
            break;
        }
        // §10.1: a server's request ids are odd. We are always the client, so an
        // even one from the peer is INVALID_REQUEST_ID.
        if first.request_id().is_some_and(|id| id % 2 == 0) {
            break;
        }
        if out.send(PeerRequest { first, tx, rx }).is_err() {
            return;
        }
    }
    session.close(
        session_error_code::PROTOCOL_VIOLATION,
        b"bidirectional stream",
    );
    closed.store(true, Ordering::Relaxed);
}

/// Accept every unidirectional stream and dispatch on the type varint that opens
/// it (§3.4). The peer's control stream is one of them, so its SETUP arrives
/// through `setup`.
async fn accept_uni(
    session: Session,
    setup: oneshot::Sender<Result<Params, G2gError>>,
    data: mpsc::UnboundedSender<DataEvent>,
    closed: Arc<AtomicBool>,
    max_object_bytes: usize,
) {
    let mut setup = Some(setup);
    while let Ok(mut stream) = session.accept_uni().await {
        let Ok((code, prefix)) = read_stream_type(&mut stream).await else {
            continue;
        };
        match UniStreamType::from_code(code) {
            Ok(UniStreamType::Setup) => {
                let Some(slot) = setup.take() else {
                    // §3.3: exactly one control stream per peer.
                    break;
                };
                tokio::spawn(read_control(
                    stream,
                    prefix,
                    slot,
                    Arc::clone(&closed),
                    session.clone(),
                ));
            }
            Ok(UniStreamType::Subgroup(_)) => {
                tokio::spawn(read_subgroup(
                    stream,
                    prefix,
                    data.clone(),
                    max_object_bytes,
                ));
            }
            // We never send a FETCH, so a FETCH response is unsolicited; a
            // padding stream is data to throw away. Both are read to their end
            // so they do not hold flow control.
            Ok(UniStreamType::FetchHeader) | Ok(UniStreamType::Padding) => {
                tokio::spawn(drain(stream));
            }
            // §3.4: an unknown stream type closes the session.
            Err(_) => break,
        }
    }
    if let Some(slot) = setup.take() {
        let _ = slot.send(Err(protocol_err()));
    }
    session.close(session_error_code::PROTOCOL_VIOLATION, b"stream type");
    closed.store(true, Ordering::Relaxed);
}

/// Read just enough of a unidirectional stream to decode the type varint that
/// opens it, returning it and every byte read (the type included, since a
/// subgroup decoder needs it).
async fn read_stream_type(stream: &mut RecvStream) -> Result<(u64, Vec<u8>), G2gError> {
    // A vi64 is at most nine bytes, so this cannot grow on a peer's say-so.
    let mut buf = Vec::with_capacity(9);
    let mut chunk = [0u8; 9];
    loop {
        match super::coding::reader(&buf).varint() {
            Ok(code) => return Ok((code, buf)),
            Err(MoqtError::Incomplete) if buf.len() < 9 => {}
            Err(_) => return Err(protocol_err()),
        }
        match stream.read(&mut chunk[..9 - buf.len()]).await {
            Ok(Some(n)) if n > 0 => buf.extend_from_slice(&chunk[..n]),
            _ => return Err(protocol_err()),
        }
    }
}

/// Read the peer's control stream: the SETUP that opens it, then whatever
/// follows (GOAWAY is the only message the draft puts here).
async fn read_control(
    mut stream: RecvStream,
    prefix: Vec<u8>,
    setup: oneshot::Sender<Result<Params, G2gError>>,
    closed: Arc<AtomicBool>,
    session: Session,
) {
    let mut reader = MessageReader { buf: prefix };
    match reader.next(&mut stream).await {
        Ok(Some(ControlMessage::Setup { options })) => {
            if setup.send(Ok(options)).is_err() {
                return;
            }
        }
        // Anything but SETUP first, or a stream that ends there, fails the
        // handshake (§3.3).
        _ => {
            let _ = setup.send(Err(protocol_err()));
            closed.store(true, Ordering::Relaxed);
            return;
        }
    }
    let mut goaway_seen = false;
    loop {
        match reader.next(&mut stream).await {
            // A GOAWAY means the peer is going away soon; more than one on the
            // control stream is a violation (§10.4).
            Ok(Some(ControlMessage::GoAway { .. })) if !goaway_seen => goaway_seen = true,
            Ok(Some(_)) | Err(_) => break,
            // §3.3: closing a control stream during the session is a violation.
            Ok(None) => break,
        }
    }
    // Either way the session is finished as far as the element is concerned.
    session.close(session_error_code::NO_ERROR, b"control stream");
    closed.store(true, Ordering::Relaxed);
}

/// Read one subgroup stream to its end, reporting the header, each whole object,
/// and the close. A malformed stream ends here rather than failing the session:
/// a publisher may reset an individual data stream, so the subscription survives
/// losing one.
async fn read_subgroup(
    mut stream: RecvStream,
    prefix: Vec<u8>,
    out: mpsc::UnboundedSender<DataEvent>,
    max_object_bytes: usize,
) {
    let mut decoder = SubgroupStreamDecoder::new(max_object_bytes);
    if decoder.push(&prefix).is_err() {
        return;
    }
    let mut chunk = vec![0u8; DATA_READ_CHUNK];
    let mut route: Option<(u64, u64)> = None;
    'read: loop {
        loop {
            match decoder.next_item() {
                Ok(Some(super::data::StreamItem::Header(header))) => {
                    route = Some((header.track_alias, header.group_id));
                    if out
                        .send(DataEvent::StreamOpened {
                            track_alias: header.track_alias,
                            group_id: header.group_id,
                        })
                        .is_err()
                    {
                        return;
                    }
                }
                Ok(Some(super::data::StreamItem::Object(object))) => {
                    let Some((track_alias, _)) = route else {
                        break 'read; // an object before the header is impossible
                    };
                    if out
                        .send(DataEvent::Object {
                            track_alias,
                            object,
                        })
                        .is_err()
                    {
                        return;
                    }
                }
                Ok(None) => break,
                Err(_) => break 'read,
            }
        }
        match stream.read(&mut chunk).await {
            Ok(Some(n)) if n > 0 => {
                if decoder.push(&chunk[..n]).is_err() {
                    break;
                }
            }
            // A clean finish, a zero read, or a reset: the stream is over.
            _ => break,
        }
    }
    if let Some((track_alias, group_id)) = route {
        let _ = out.send(DataEvent::StreamClosed {
            track_alias,
            group_id,
        });
    }
}

/// Read and discard a stream, bounded, so it releases flow control without
/// buffering anything (§11.5.1 requires exactly this of a padding stream).
async fn drain(mut stream: RecvStream) {
    let mut chunk = vec![0u8; DATA_READ_CHUNK];
    let mut seen = 0u64;
    while let Ok(Some(n)) = stream.read(&mut chunk).await {
        if n == 0 {
            return;
        }
        seen = seen.saturating_add(n as u64);
        if seen > MAX_PADDING_BYTES {
            let _ = stream.stop(session_error_code::NO_ERROR);
            return;
        }
    }
}

/// Read datagram objects onto the data channel, so both carriages reorder
/// together: a datagram object is an object like any other, it just arrives
/// without a stream to open or close.
async fn read_datagrams(
    session: Session,
    out: mpsc::UnboundedSender<DataEvent>,
    max_object_bytes: usize,
) {
    while let Ok(bytes) = session.read_datagram().await {
        // A padding datagram carries nothing (§11.5.2).
        if super::coding::reader(&bytes).varint() == Ok(PADDING_DATAGRAM_TYPE) {
            continue;
        }
        // A datagram that does not decode is one lost object on a carriage that
        // already loses them: dropping it keeps the rest of the session, where
        // killing it would lose everything.
        let Ok(object) = DatagramObject::decode(&bytes, max_object_bytes) else {
            continue;
        };
        let event = DataEvent::Object {
            track_alias: object.track_alias,
            object: object.into_received(),
        };
        if out.send(event).is_err() {
            return;
        }
    }
}

/// A padding stream this build would send, if it ever needed to probe bandwidth.
/// Exposed so the type number has one definition rather than a literal at each
/// use.
pub fn padding_stream_prefix() -> Vec<u8> {
    let mut out = Vec::new();
    super::coding::put_vi64(&mut out, PADDING_STREAM_TYPE);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::coding::reader;

    #[test]
    fn the_padding_stream_type_encodes_as_five_bytes() {
        assert_eq!(
            padding_stream_prefix(),
            vec![0xf0, 0x13, 0x2b, 0x3e, 0x28],
            "0x132B3E28 needs 29 bits, so a five-byte vi64"
        );
        assert_eq!(
            reader(&padding_stream_prefix()).varint(),
            Ok(PADDING_STREAM_TYPE)
        );
    }

    /// The control stream's type varint must not collide with a subgroup
    /// header's, since one accept loop tells them apart by it.
    #[test]
    fn the_control_stream_type_is_distinct_from_every_subgroup_header() {
        let mut setup = Vec::new();
        super::super::coding::put_vi64(&mut setup, super::super::message::msg_type::SETUP);
        assert_eq!(setup, vec![0xaf, 0x00]);
        assert_eq!(
            UniStreamType::from_code(super::super::message::msg_type::SETUP),
            Ok(UniStreamType::Setup)
        );
        for code in (0x10..=0x1fu64).chain(0x30..=0x3f).chain(0x70..=0x7f) {
            assert_ne!(code, super::super::message::msg_type::SETUP);
            // Every subgroup type is a single byte, so it cannot be mistaken for
            // the two-byte SETUP prefix.
            let mut out = Vec::new();
            super::super::coding::put_vi64(&mut out, code);
            assert_eq!(out.len(), 1, "{code:#x} is a one-byte stream type");
        }
    }
}
