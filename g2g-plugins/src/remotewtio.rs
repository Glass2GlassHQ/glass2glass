//! Native WebTransport <-> wire-codec I/O shared by the `webtransport` elements
//! (`RemoteWtSink`, `RemoteWtSrc`, `RemoteWtTransform`): dial or accept a session,
//! and carry the [`PipelinePacket`] stream over the single bidirectional stream it
//! opens.
//!
//! A WebTransport stream is a QUIC stream, so it is a *byte* stream with no
//! message boundaries: the framing is the `remote` TCP pair's `u32` length prefix
//! (shared with it in [`crate::remotewire`]), not the WebSocket pair's
//! one-message-per-packet.
//!
//! The M911 datagram carrier rides the same session: with `datagrams` set, a
//! sender puts each data frame in one QUIC datagram (unreliable, unordered,
//! MTU-bounded) and keeps the control packets (caps, segment, flush, `Eos`) on
//! the stream, since losing one of those loses the stream itself. One datagram is
//! one packet, so a receiver reassembles nothing and a drop simply loses that
//! frame. A frame too large for the path falls back to the stream: truncating it
//! or dropping it silently would both be worse than arriving late. Receiving is
//! mode-free, [`WtStream::recv`] takes packets from whichever the peer used.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::time::Duration;

use web_transport_quinn::proto::ConnectRequest;
use web_transport_quinn::quinn::rustls::pki_types::pem::PemObject;
use web_transport_quinn::quinn::rustls::pki_types::{CertificateDer, PrivateKeyDer};
use web_transport_quinn::quinn::Connection;
use web_transport_quinn::{ClientBuilder, CongestionControl, RecvStream, SendStream, Session};

use g2g_core::wire::{decode_packet, encode_packet};
use g2g_core::{
    G2gError, HardwareError, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::remotewire::{map_wire, send_framed, FramedRead};

/// Any WebTransport / QUIC / TLS failure maps to an internal hardware error (the
/// network boundary), matching how the other transports treat a transport fault.
pub(crate) fn wt_err<E>(_: E) -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

/// How long a half-close waits for the peer to acknowledge what was already
/// written. Closing a QUIC connection discards anything unacknowledged, so
/// without this the trailing `Eos` can be lost; bounded so a peer that stopped
/// reading cannot wedge the pipeline.
const FLUSH_TIMEOUT: Duration = Duration::from_millis(500);

/// A WebTransport session plus the one bidirectional stream carrying the wire
/// codec. The session is held alongside the stream because dropping it closes the
/// QUIC connection underneath, and because the datagram carrier sends and reads
/// on it. Opaque: it is the `RemoteWtSrc` connection type, so it has to be
/// nameable, but nothing outside the crate operates on it.
#[derive(Debug)]
pub struct WtStream {
    session: Session,
    tx: SendStream,
    rx: RecvStream,
    /// Stream-side read state, kept across calls so [`Self::recv`] survives being
    /// cancelled by the datagram branch of its `select!`.
    reader: FramedRead,
    /// Send data frames as datagrams (the sender's `datagrams` property).
    datagrams: bool,
    /// Whether the peer's first packet is still outstanding. See [`Self::recv`].
    leading: bool,
}

impl WtStream {
    pub(crate) fn new(session: Session, tx: SendStream, rx: RecvStream, datagrams: bool) -> Self {
        Self {
            session,
            tx,
            rx,
            reader: FramedRead::default(),
            datagrams,
            leading: true,
        }
    }

    /// Serialize and send one packet: length-framed on the session's stream, or,
    /// in datagram mode, a data frame that fits the path as one QUIC datagram.
    /// `Ok(true)` when it went out as a datagram.
    pub(crate) async fn send(&mut self, packet: &PipelinePacket) -> Result<bool, G2gError> {
        if self.datagrams && matches!(packet, PipelinePacket::DataFrame(_)) {
            let body = encode_packet(packet).map_err(map_wire)?;
            if body.len() <= self.session.max_datagram_size() {
                self.session.send_datagram(body.into()).map_err(wt_err)?;
                return Ok(true);
            }
            // Too large for the path: the reliable stream carries it instead of
            // the frame being truncated or silently dropped.
        }
        send_framed(&mut self.tx, packet).await?;
        Ok(false)
    }

    /// Read the next packet, from the session's stream or its datagram flow
    /// (whichever the peer used). `Ok(None)` once the peer finishes its send
    /// stream.
    pub(crate) async fn recv(&mut self) -> Result<Option<PipelinePacket>, G2gError> {
        if self.leading {
            // A peer's first packet is its negotiated caps, and those are always
            // on the stream: taking a datagram ahead of them would lose the
            // receiver's caps discovery.
            let packet = self.reader.recv(&mut self.rx).await?;
            self.leading = false;
            return Ok(packet);
        }
        let datagram = {
            let Self {
                session,
                rx,
                reader,
                ..
            } = self;
            tokio::select! {
                // Datagrams first: a stream keeps whatever it buffered until it
                // is read, while an undrained datagram queue loses frames, and
                // taking the stream's Eos first would end the flow with frames
                // still queued.
                biased;
                datagram = session.read_datagram() => datagram,
                framed = reader.recv(rx) => return framed,
            }
        };
        match datagram {
            Ok(bytes) => Ok(Some(decode_packet(&bytes).map_err(map_wire)?)),
            // The datagram flow ends with the session, well before the stream has
            // been read out: let the stream say how this ended.
            Err(_) => self.reader.recv(&mut self.rx).await,
        }
    }

    /// Half-close: finish our send direction and wait (bounded) for the peer to
    /// acknowledge the data, so the far side reads the stream to its end.
    pub(crate) async fn finish(&mut self) {
        let _ = self.tx.finish();
        let _ = tokio::time::timeout(FLUSH_TIMEOUT, self.tx.stopped()).await;
        self.session.close(0, b"eos");
    }
}

/// Dial `url` (e.g. `https://127.0.0.1:9603`) and open the session's single
/// bidirectional stream. `hashes` is the `server-certificate-hashes` property:
/// empty accepts any certificate a system root signs, otherwise only the listed
/// certificates are accepted. `datagrams` puts data frames on the datagram
/// carrier (see the module docs).
pub(crate) async fn connect(
    url: &str,
    hashes: &str,
    datagrams: bool,
    congestion: &str,
) -> Result<WtStream, G2gError> {
    let session = dial(url, hashes, &[], congestion).await?;
    // `Session::max_datagram_size` panics on a peer that refused datagram
    // support, so the configured mode is checked once here rather than per frame.
    if datagrams && Connection::max_datagram_size(&session).is_none() {
        return Err(G2gError::NotConfigured);
    }
    let (tx, rx) = session.open_bi().await.map_err(wt_err)?;
    Ok(WtStream::new(session, tx, rx, datagrams))
}

/// Dial `url` and complete the HTTP/3 CONNECT handshake, leaving stream opening
/// to the caller. `hashes` is the `server-certificate-hashes` property (see
/// [`connect`]); `protocols` names the WebTransport subprotocols offered in
/// preference order, which is how a protocol layered on the carrier (MoQT's
/// `moqt-16` / `moqt-18`) states its version, since the QUIC ALPN is always
/// `h3`. The server's pick is `session.response().protocol`. `congestion` is a
/// [`CONGESTION_PROP`] nick.
pub(crate) async fn dial(
    url: &str,
    hashes: &str,
    protocols: &[&str],
    congestion: &str,
) -> Result<Session, G2gError> {
    let url = url::Url::parse(url).map_err(wt_err)?;
    let hashes = parse_cert_hashes(hashes)?;
    let builder = ClientBuilder::new().with_congestion_control(congestion_control(congestion));
    let client = if hashes.is_empty() {
        builder.with_system_roots().map_err(wt_err)?
    } else {
        builder
            .with_server_certificate_hashes(hashes)
            .map_err(wt_err)?
    };
    let mut request = ConnectRequest::new(url);
    for protocol in protocols {
        request = request.with_protocol(*protocol);
    }
    client.connect(request).await.map_err(wt_err)
}

/// The `congestion-control` spec, shared by the WebTransport client and server
/// elements (`web-transport-quinn` takes the same choice on both builders).
pub(crate) const CONGESTION_PROP: PropertySpec = PropertySpec::new(
    "congestion-control",
    PropKind::Str,
    "QUIC congestion controller: default or throughput (CUBIC), low-latency (BBR)",
)
.with_default("default")
.with_enum_values("default | throughput | low-latency");

/// Map a `congestion-control` nick to the algorithm both builders take. `None`
/// for anything else, which is how [`set_congestion`] rejects it.
fn congestion(nick: &str) -> Option<CongestionControl> {
    match nick {
        "default" => Some(CongestionControl::Default),
        "throughput" => Some(CongestionControl::Throughput),
        "low-latency" => Some(CongestionControl::LowLatency),
        _ => None,
    }
}

/// The stored nick as an algorithm, for a builder. Unknown text never gets
/// stored, so the fallback only covers a never-set value.
pub(crate) fn congestion_control(nick: &str) -> CongestionControl {
    congestion(nick).unwrap_or(CongestionControl::Default)
}

/// The `congestion-control` half of `set_property`: store the nick if it names
/// an algorithm. A launch line is checked against `enum_values` before it gets
/// here, a direct `set_property` call is not.
pub(crate) fn set_congestion(target: &mut String, value: &PropValue) -> Result<(), PropError> {
    let nick = value.as_str().ok_or(PropError::Type)?;
    congestion(nick).ok_or(PropError::Value)?;
    *target = nick.to_string();
    Ok(())
}

/// Load a PEM certificate chain and its private key from file paths (the
/// `certificate` / `private-key` properties). QUIC is always TLS, so a
/// WebTransport server cannot start without them.
pub(crate) fn load_certificate(
    cert: &str,
    key: &str,
) -> Result<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>), G2gError> {
    if cert.is_empty() || key.is_empty() {
        return Err(G2gError::NotConfigured);
    }
    let chain: Vec<CertificateDer<'static>> = CertificateDer::pem_file_iter(cert)
        .map_err(wt_err)?
        .collect::<Result<_, _>>()
        .map_err(wt_err)?;
    if chain.is_empty() {
        return Err(G2gError::NotConfigured);
    }
    let key = PrivateKeyDer::from_pem_file(key).map_err(wt_err)?;
    Ok((chain, key))
}

/// Parse a comma-separated list of hex SHA-256 certificate digests (the browser
/// WebTransport API's `serverCertificateHashes`, for a self-signed or short-lived
/// certificate no system root covers). Empty means "use the system roots".
pub(crate) fn parse_cert_hashes(spec: &str) -> Result<Vec<Vec<u8>>, G2gError> {
    let mut out = Vec::new();
    for item in spec.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let bytes = hex_bytes(item).ok_or(G2gError::NotConfigured)?;
        // A SHA-256 digest and nothing else: the verifier compares raw bytes, so
        // a wrong-length entry would silently never match.
        if bytes.len() != 32 {
            return Err(G2gError::NotConfigured);
        }
        out.push(bytes);
    }
    Ok(out)
}

/// Decode an even-length hex string, tolerating `:` separators.
fn hex_bytes(s: &str) -> Option<Vec<u8>> {
    let digits: Vec<u8> = s.bytes().filter(|b| *b != b':').collect();
    if !digits.len().is_multiple_of(2) {
        return None;
    }
    digits
        .chunks(2)
        .map(|pair| {
            let hi = (pair[0] as char).to_digit(16)?;
            let lo = (pair[1] as char).to_digit(16)?;
            Some((hi * 16 + lo) as u8)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cert_hashes_parse_and_reject_malformed() {
        let one = "aa".repeat(32);
        let parsed = parse_cert_hashes(&one).expect("one digest");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].len(), 32);
        assert_eq!(parsed[0][0], 0xaa);

        let two = alloc::format!("{one},{one}");
        assert_eq!(parse_cert_hashes(&two).expect("two digests").len(), 2);
        assert!(parse_cert_hashes("")
            .expect("empty is system roots")
            .is_empty());
        // Too short (a truncated digest would never match) and non-hex.
        assert!(parse_cert_hashes("aabb").is_err());
        assert!(parse_cert_hashes(&"zz".repeat(32)).is_err());
    }
}
