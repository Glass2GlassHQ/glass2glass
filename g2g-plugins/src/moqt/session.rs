//! MoQ Transport draft-16 session over the M901 WebTransport carrier: the
//! SETUP exchange, the control stream, and opening subgroup data streams.
//!
//! The QUIC ALPN for WebTransport is always `h3`, so draft-16 states its
//! version as the WebTransport subprotocol [`MOQT_PROTOCOL`] on the HTTP/3
//! CONNECT request; the SETUP payloads carry parameters only
//! (`moq-native-ietf/src/quic.rs` requests the same subprotocol, and
//! `moq-transport/src/setup/client.rs` documents the payload change).
//!
//! The control stream is the session's first bidirectional stream. Its read
//! half runs in its own task so a SUBSCRIBE is decoded as it arrives rather
//! than when the element next has a frame to push; the write half stays with
//! the caller, which is the only thing that sends control messages.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use tokio::sync::mpsc;
use web_transport_quinn::{RecvStream, SendStream, Session};

use g2g_core::{G2gError, HardwareError};

use crate::remotewtio::{dial, wt_err};

use super::coding::{setup_param, MoqtError, Params};
use super::data::SubgroupHeader;
use super::message::ControlMessage;

/// The WebTransport subprotocol that names draft-16
/// (`moq-transport/src/setup/mod.rs`).
pub const MOQT_PROTOCOL: &str = "moqt-16";

/// Draft version this session speaks, for the record: `0xff000010`.
pub const MOQT_VERSION: u32 = 0xff00_0010;

/// Read size for the control stream. Control messages are 16-bit-length
/// bounded, so this only affects how often the reader syscalls.
const CONTROL_READ_CHUNK: usize = 8192;

/// A peer that violates the control-stream framing, or a transport fault, is
/// the same thing to the pipeline: the session is gone.
fn protocol_err(_: MoqtError) -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

/// A connected MoQ Transport session: the SETUP exchange has completed and the
/// control stream is live.
#[derive(Debug)]
pub struct MoqtSession {
    session: Session,
    control_tx: SendStream,
    inbound: mpsc::UnboundedReceiver<ControlMessage>,
    /// `MAX_REQUEST_ID` the peer advertised: request ids we may allocate stay
    /// below it.
    peer_max_request_id: u64,
    /// Next request id to allocate. A client uses even ids (§6.2).
    next_request_id: u64,
    /// Set once the control stream ends or fails.
    closed: bool,
}

impl MoqtSession {
    /// Dial `url`, complete the CONNECT handshake requesting [`MOQT_PROTOCOL`],
    /// open the control stream, and exchange CLIENT_SETUP / SERVER_SETUP.
    ///
    /// `cert_hashes` is the `server-certificate-hashes` form the M901 carrier
    /// takes (empty means "a system root must sign it"). `max_request_id` is
    /// the limit we advertise to the peer; `implementation` is the optional
    /// MOQT_IMPLEMENTATION string.
    pub async fn connect(
        url: &str,
        cert_hashes: &str,
        max_request_id: u64,
        implementation: &str,
    ) -> Result<Self, G2gError> {
        let session = dial(url, cert_hashes, Some(MOQT_PROTOCOL)).await?;
        let (mut control_tx, control_rx) = session.open_bi().await.map_err(wt_err)?;

        let mut params = Params::new();
        params.set_int(setup_param::MAX_REQUEST_ID, max_request_id);
        if !implementation.is_empty() {
            params.set_bytes(
                setup_param::MOQT_IMPLEMENTATION,
                implementation.as_bytes().to_vec(),
            );
        }
        // WebTransport carries the path in the CONNECT URL, so PATH / AUTHORITY
        // are only sent on a raw-QUIC session (§9.3.1.1).
        write_message(&mut control_tx, &ControlMessage::ClientSetup { params }).await?;

        let (inbound_tx, mut inbound) = mpsc::unbounded_channel();
        tokio::spawn(read_control(control_rx, inbound_tx));

        let server_params = match inbound.recv().await {
            Some(ControlMessage::ServerSetup { params }) => params,
            // Anything before SERVER_SETUP, or a closed stream, fails the dial.
            _ => return Err(G2gError::Hardware(HardwareError::Other)),
        };
        let peer_max_request_id = server_params
            .get_int(setup_param::MAX_REQUEST_ID)
            .unwrap_or(0);

        Ok(Self {
            session,
            control_tx,
            inbound,
            peer_max_request_id,
            next_request_id: 0,
            closed: false,
        })
    }

    /// Whether the control stream has ended.
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Take the next client request id (even, §6.2), or `None` when the peer's
    /// MAX_REQUEST_ID leaves no room.
    pub fn allocate_request_id(&mut self) -> Option<u64> {
        if self.next_request_id >= self.peer_max_request_id {
            return None;
        }
        let id = self.next_request_id;
        self.next_request_id = self.next_request_id.saturating_add(2);
        Some(id)
    }

    /// Raise the peer's advertised request-id limit (MAX_REQUEST_ID never
    /// decreases).
    pub fn set_peer_max_request_id(&mut self, max: u64) {
        self.peer_max_request_id = self.peer_max_request_id.max(max);
    }

    pub async fn send(&mut self, msg: &ControlMessage) -> Result<(), G2gError> {
        write_message(&mut self.control_tx, msg).await
    }

    /// The next control message already decoded by the reader task, or `None`
    /// when none is waiting. Never blocks on the network.
    pub fn poll_control(&mut self) -> Option<ControlMessage> {
        match self.inbound.try_recv() {
            Ok(msg) => Some(msg),
            Err(mpsc::error::TryRecvError::Empty) => None,
            Err(mpsc::error::TryRecvError::Disconnected) => {
                self.closed = true;
                None
            }
        }
    }

    /// Open a unidirectional stream and write a subgroup header on it. The
    /// caller then writes object headers and payloads.
    pub async fn open_subgroup(&mut self, header: &SubgroupHeader) -> Result<SendStream, G2gError> {
        let mut bytes = Vec::new();
        header.encode(&mut bytes).map_err(protocol_err)?;
        let mut stream = self.session.open_uni().await.map_err(wt_err)?;
        // Smaller publisher priority is sent first, and QUIC send priority runs
        // the other way, so the stream priority is the negated byte.
        let _ = stream.set_priority(-i32::from(header.publisher_priority));
        stream.write_all(&bytes).await.map_err(wt_err)?;
        Ok(stream)
    }

    /// Finish the control stream and close the QUIC connection.
    pub async fn close(&mut self, reason: &str) {
        let _ = self.control_tx.finish();
        self.session.close(0, reason.as_bytes());
        self.closed = true;
    }
}

async fn write_message(stream: &mut SendStream, msg: &ControlMessage) -> Result<(), G2gError> {
    let mut bytes = Vec::new();
    msg.encode(&mut bytes).map_err(protocol_err)?;
    stream.write_all(&bytes).await.map_err(wt_err)
}

/// Decode control messages off the control stream until it ends or a peer
/// violates the framing. Dropping the sender is how the session learns it
/// ended.
async fn read_control(mut stream: RecvStream, out: mpsc::UnboundedSender<ControlMessage>) {
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = vec![0u8; CONTROL_READ_CHUNK];
    loop {
        // Drain every whole message already buffered before reading again.
        loop {
            match ControlMessage::decode(&buf) {
                Ok((msg, used)) => {
                    buf.drain(..used);
                    if out.send(msg).is_err() {
                        return;
                    }
                }
                Err(MoqtError::Incomplete) => break,
                // A malformed control message is a PROTOCOL_VIOLATION: stop
                // reading rather than resynchronizing on a stream whose framing
                // we no longer trust.
                Err(MoqtError::Malformed) => return,
            }
        }
        match stream.read(&mut chunk).await {
            Ok(Some(n)) if n > 0 => buf.extend_from_slice(&chunk[..n]),
            // `Ok(None)` is a clean finish; a zero read or an error ends it too.
            _ => return,
        }
    }
}

/// The MOQT_IMPLEMENTATION string this build advertises.
pub fn implementation_name() -> String {
    String::from(concat!("glass2glass/", env!("CARGO_PKG_VERSION")))
}
