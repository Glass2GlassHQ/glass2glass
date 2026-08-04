//! Native WebTransport <-> wire-codec I/O shared by the `webtransport` elements
//! (`RemoteWtSink`, `RemoteWtSrc`, `RemoteWtTransform`): dial or accept a session,
//! and carry the [`PipelinePacket`] stream over the single bidirectional stream it
//! opens.
//!
//! A WebTransport stream is a QUIC stream, so it is a *byte* stream with no
//! message boundaries: the framing is the `remote` TCP pair's `u32` length prefix
//! (shared with it in [`crate::remotewire`]), not the WebSocket pair's
//! one-message-per-packet. Datagram mode (unreliable, MTU-bounded) is a separate
//! carrier and is not used here.

use alloc::vec::Vec;
use core::time::Duration;

use web_transport_quinn::proto::ConnectRequest;
use web_transport_quinn::quinn::rustls::pki_types::pem::PemObject;
use web_transport_quinn::quinn::rustls::pki_types::{CertificateDer, PrivateKeyDer};
use web_transport_quinn::{ClientBuilder, RecvStream, SendStream, Session};

use g2g_core::{G2gError, HardwareError, PipelinePacket};

use crate::remotewire::{recv_framed, send_framed};

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
/// QUIC connection underneath. Opaque: it is the `RemoteWtSrc` connection type, so
/// it has to be nameable, but nothing outside the crate operates on it.
#[derive(Debug)]
pub struct WtStream {
    session: Session,
    tx: SendStream,
    rx: RecvStream,
}

impl WtStream {
    pub(crate) fn new(session: Session, tx: SendStream, rx: RecvStream) -> Self {
        Self { session, tx, rx }
    }

    /// Serialize and send one packet, length-framed.
    pub(crate) async fn send(&mut self, packet: &PipelinePacket) -> Result<(), G2gError> {
        send_framed(&mut self.tx, packet).await
    }

    /// Read the next packet. `Ok(None)` once the peer finishes its send stream.
    pub(crate) async fn recv(&mut self) -> Result<Option<PipelinePacket>, G2gError> {
        recv_framed(&mut self.rx).await
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
/// certificates are accepted.
pub(crate) async fn connect(url: &str, hashes: &str) -> Result<WtStream, G2gError> {
    let session = dial(url, hashes, None).await?;
    let (tx, rx) = session.open_bi().await.map_err(wt_err)?;
    Ok(WtStream::new(session, tx, rx))
}

/// Dial `url` and complete the HTTP/3 CONNECT handshake, leaving stream opening
/// to the caller. `hashes` is the `server-certificate-hashes` property (see
/// [`connect`]); `protocol` names a WebTransport subprotocol to request, which
/// is how a protocol layered on the carrier (MoQT's `moqt-16`) states its
/// version, since the QUIC ALPN is always `h3`.
pub(crate) async fn dial(
    url: &str,
    hashes: &str,
    protocol: Option<&str>,
) -> Result<Session, G2gError> {
    let url = url::Url::parse(url).map_err(wt_err)?;
    let hashes = parse_cert_hashes(hashes)?;
    let builder = ClientBuilder::new();
    let client = if hashes.is_empty() {
        builder.with_system_roots().map_err(wt_err)?
    } else {
        builder
            .with_server_certificate_hashes(hashes)
            .map_err(wt_err)?
    };
    let mut request = ConnectRequest::new(url);
    if let Some(protocol) = protocol {
        request = request.with_protocol(protocol);
    }
    client.connect(request).await.map_err(wt_err)
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
    if digits.len() % 2 != 0 {
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
