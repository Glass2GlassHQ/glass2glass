// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Native WebTransport and raw QUIC <-> wire-codec I/O shared by the `webtransport` elements
//! (`RemoteWtSink`, `RemoteWtSrc`, `RemoteWtTransform`): dial or accept a session,
//! and carry the [`PipelinePacket`] stream over the single bidirectional stream it
//! opens.
//!
//! A WebTransport stream is a QUIC stream, so it is a *byte* stream with no
//! message boundaries: the framing is the `remote` TCP pair's `u32` length prefix
//! (shared with it in [`crate::remotewire`]), not the WebSocket pair's
//! one-message-per-packet.
//!
//! The datagram carrier rides the same session: with `datagrams` set, a
//! sender puts each data frame in one QUIC datagram (unreliable, unordered,
//! MTU-bounded) and keeps the control packets (caps, segment, flush, `Eos`) on
//! the stream, since losing one of those loses the stream itself. One datagram is
//! one packet, so a receiver reassembles nothing and a drop simply loses that
//! frame. A frame too large for the path falls back to the stream: truncating it
//! or dropping it silently would both be worse than arriving late. Receiving is
//! mode-free, [`WtStream::recv`] takes packets from whichever the peer used.
//!
//! A data channel (`open_data_channel` on the sink and source) is a second
//! bidirectional stream carrying length-framed byte messages, opened by the
//! source and accepted by the sink.

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::time::Duration;

use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;
use web_transport_quinn::proto::ConnectRequest;
use web_transport_quinn::quinn::rustls::pki_types::pem::PemObject;
use web_transport_quinn::quinn::rustls::pki_types::{CertificateDer, PrivateKeyDer};

use web_transport_quinn::quinn::crypto::rustls::{
    HandshakeData, QuicClientConfig, QuicServerConfig,
};
use web_transport_quinn::quinn::rustls;
use web_transport_quinn::quinn::rustls::client::danger::{
    HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier,
};
use web_transport_quinn::quinn::rustls::pki_types::{ServerName, UnixTime};
use web_transport_quinn::quinn::{self, Connection, Endpoint};
use web_transport_quinn::{
    crypto, ClientBuilder, CongestionControl, RecvStream, Request, SendStream, Session,
};

use g2g_core::g2g_warn;
use g2g_core::log::Target;
use g2g_core::wire::{decode_packet, encode_packet};
use g2g_core::{
    G2gError, NetworkFailure, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::abortondrop::AbortOnDrop;
use crate::datachannel::{DataChannelTransportEnds, MAX_MESSAGE_SIZE};
use crate::remotewire::{map_wire, send_frame_bytes, send_framed, FramedRead};
use crate::textencoding::hex_decode;

/// Any WebTransport / QUIC / TLS failure ends the session.
pub(crate) fn wt_err<E>(_: E) -> G2gError {
    G2gError::Network(NetworkFailure::Stream)
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
    _data_channel: Option<AbortOnDrop>,
}

impl WtStream {
    pub(crate) fn new(
        session: Session,
        tx: SendStream,
        rx: RecvStream,
        datagrams: bool,
        data_channel: Option<AbortOnDrop>,
    ) -> Self {
        Self {
            session,
            tx,
            rx,
            reader: FramedRead::default(),
            datagrams,
            leading: true,
            _data_channel: data_channel,
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

/// Open the sender's single bidirectional stream on `session`. `datagrams` puts
/// data frames on the datagram carrier (see the module docs).
pub(crate) async fn open_stream(
    session: Session,
    datagrams: bool,
    data_channel: Option<&SharedDataChannel>,
) -> Result<WtStream, G2gError> {
    // `Session::max_datagram_size` panics on a peer that refused datagram
    // support, so the configured mode is checked once here rather than per frame.
    if datagrams && Connection::max_datagram_size(&session).is_none() {
        return Err(G2gError::NotConfigured);
    }
    let pump = data_channel.map(|channel| accept_data_channel(&session, channel.clone()));
    let (tx, rx) = session.open_bi().await.map_err(wt_err)?;
    Ok(WtStream::new(session, tx, rx, datagrams, pump))
}

pub(crate) type SharedDataChannel = Arc<tokio::sync::Mutex<DataChannelTransportEnds>>;

pub(crate) fn open_data_channel(session: &Session, channel: SharedDataChannel) -> AbortOnDrop {
    let session = session.clone();
    AbortOnDrop(tokio::spawn(async move {
        if let Ok((tx, rx)) = session.open_bi().await {
            carry_messages(tx, rx, channel).await;
        }
    }))
}

fn accept_data_channel(session: &Session, channel: SharedDataChannel) -> AbortOnDrop {
    let session = session.clone();
    AbortOnDrop(tokio::spawn(async move {
        if let Ok((tx, rx)) = session.accept_bi().await {
            carry_messages(tx, rx, channel).await;
        }
    }))
}

async fn carry_messages(mut tx: SendStream, mut rx: RecvStream, channel: SharedDataChannel) {
    // tokio releases the lock when a replaced connection's task is aborted
    let mut ends = channel.lock().await;
    let DataChannelTransportEnds { outgoing, incoming } = &mut *ends;
    let mut reader = FramedRead::default();
    let mut sending = true;
    loop {
        tokio::select! {
            message = outgoing.recv(), if sending => match message {
                Some(message) => {
                    if send_frame_bytes(&mut tx, &message).await.is_err() {
                        return;
                    }
                }
                None => {
                    let _ = tx.finish();
                    sending = false;
                }
            },
            received = reader.recv_bytes(&mut rx, MAX_MESSAGE_SIZE) => match received {
                Ok(Some(message)) => {
                    if incoming.send(message.to_vec()).is_err() {
                        return;
                    }
                }
                Ok(None) => return,
                Err(error) => {
                    g2g_warn!(
                        Target::category("remotewt"),
                        "data channel closed on a bad message: {error:?}"
                    );
                    return;
                }
            },
        }
    }
}

pub(crate) const RAW_QUIC_ALPN: &str = "g2g-remote";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuicMode {
    Caller,
    Listener,
}

impl QuicMode {
    pub(crate) fn name(self) -> &'static str {
        match self {
            QuicMode::Caller => "caller",
            QuicMode::Listener => "listener",
        }
    }

    pub(crate) fn from_name(name: &str) -> Option<Self> {
        match name {
            "caller" => Some(QuicMode::Caller),
            "listener" => Some(QuicMode::Listener),
            _ => None,
        }
    }
}

pub(crate) fn set_mode(target: &mut QuicMode, value: &PropValue) -> Result<(), PropError> {
    let name = value.as_str().ok_or(PropError::Type)?;
    *target = QuicMode::from_name(name).ok_or(PropError::Value)?;
    Ok(())
}

fn url_host(url: &url::Url) -> Result<&str, G2gError> {
    // url leaves the host of a non-special scheme such as quic:// unparsed, brackets included
    let host = url.host_str().ok_or(G2gError::NotConfigured)?;
    Ok(host.trim_start_matches('[').trim_end_matches(']'))
}

pub(crate) fn bind_address(location: &str) -> Result<SocketAddr, G2gError> {
    let url = url::Url::parse(location).map_err(|_| G2gError::NotConfigured)?;
    let port = url.port().ok_or(G2gError::NotConfigured)?;
    let ip: core::net::IpAddr = url_host(&url)?
        .parse()
        .map_err(|_| G2gError::NotConfigured)?;
    Ok(SocketAddr::new(ip, port))
}
const RAW_QUIC_SCHEME: &str = "quic";

fn transport_config(congestion: &str) -> Arc<quinn::TransportConfig> {
    let mut transport = quinn::TransportConfig::default();
    // web-transport-quinn keeps its factory mapping private, so raw QUIC repeats it
    match congestion_control(congestion) {
        CongestionControl::LowLatency => {
            transport
                .congestion_controller_factory(Arc::new(quinn::congestion::BbrConfig::default()));
        }
        CongestionControl::Throughput => {
            transport
                .congestion_controller_factory(Arc::new(quinn::congestion::CubicConfig::default()));
        }
        CongestionControl::Default => {}
    }
    Arc::new(transport)
}

#[derive(Debug)]
struct PinnedCertificates {
    provider: crypto::Provider,
    hashes: Vec<Vec<u8>>,
}

impl ServerCertVerifier for PinnedCertificates {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        let hash = crypto::sha256(&self.provider, end_entity);
        if self
            .hashes
            .iter()
            .any(|pinned| pinned.as_slice() == hash.as_ref())
        {
            return Ok(ServerCertVerified::assertion());
        }
        Err(rustls::Error::InvalidCertificate(
            rustls::CertificateError::UnknownIssuer,
        ))
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        signature: &rustls::DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            signature,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        signature: &rustls::DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            signature,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }
}

fn raw_client_tls(hashes: Vec<Vec<u8>>) -> Result<rustls::ClientConfig, G2gError> {
    let provider = crypto::default_provider();
    let builder = rustls::ClientConfig::builder_with_provider(provider.clone())
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(wt_err)?;
    let mut tls = if hashes.is_empty() {
        let mut roots = rustls::RootCertStore::empty();
        for cert in rustls_native_certs::load_native_certs().certs {
            roots.add(cert).map_err(wt_err)?;
        }
        builder.with_root_certificates(roots).with_no_client_auth()
    } else {
        builder
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(PinnedCertificates { provider, hashes }))
            .with_no_client_auth()
    };
    tls.alpn_protocols = vec![RAW_QUIC_ALPN.as_bytes().to_vec()];
    Ok(tls)
}

async fn dial_raw(url: &url::Url, hashes: &str, congestion: &str) -> Result<Session, G2gError> {
    let host = url_host(url)?;
    let port = url.port().ok_or(G2gError::NotConfigured)?;
    let address = tokio::net::lookup_host((host, port))
        .await
        .map_err(wt_err)?
        .next()
        .ok_or(G2gError::Network(NetworkFailure::Stream))?;
    let tls = raw_client_tls(parse_cert_hashes(hashes)?)?;
    let mut config =
        quinn::ClientConfig::new(Arc::new(QuicClientConfig::try_from(tls).map_err(wt_err)?));
    config.transport_config(transport_config(congestion));
    let unspecified: SocketAddr = match address {
        SocketAddr::V4(_) => (Ipv4Addr::UNSPECIFIED, 0).into(),
        SocketAddr::V6(_) => (Ipv6Addr::UNSPECIFIED, 0).into(),
    };
    // quinn keeps the endpoint running while any of its connections is open
    let endpoint = Endpoint::client(unspecified).map_err(wt_err)?;
    let connection = endpoint
        .connect_with(config, address, host)
        .map_err(wt_err)?
        .await
        .map_err(wt_err)?;
    Ok(Session::raw(connection))
}

pub(crate) fn listen(
    bind: SocketAddr,
    chain: Vec<CertificateDer<'static>>,
    key: PrivateKeyDer<'static>,
    congestion: &str,
) -> Result<Endpoint, G2gError> {
    let mut tls = rustls::ServerConfig::builder_with_provider(crypto::default_provider())
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(wt_err)?
        .with_no_client_auth()
        .with_single_cert(chain, key)
        .map_err(wt_err)?;
    tls.alpn_protocols = vec![
        web_transport_quinn::ALPN.as_bytes().to_vec(),
        RAW_QUIC_ALPN.as_bytes().to_vec(),
    ];
    let mut config = quinn::ServerConfig::with_crypto(Arc::new(
        QuicServerConfig::try_from(tls).map_err(wt_err)?,
    ));
    config.transport_config(transport_config(congestion));
    Endpoint::server(config, bind).map_err(wt_err)
}

fn negotiated_alpn(connection: &Connection) -> Option<Vec<u8>> {
    connection
        .handshake_data()?
        .downcast::<HandshakeData>()
        .ok()?
        .protocol
}

pub(crate) async fn accept(endpoint: &Endpoint) -> Result<Session, G2gError> {
    let incoming = endpoint.accept().await.ok_or(G2gError::NotConfigured)?;
    let connection = incoming.await.map_err(wt_err)?;
    if negotiated_alpn(&connection).as_deref() == Some(RAW_QUIC_ALPN.as_bytes()) {
        return Ok(Session::raw(connection));
    }
    let request = Request::accept(connection).await.map_err(wt_err)?;
    request.ok().await.map_err(wt_err)
}

/// Dial `url` and complete the HTTP/3 CONNECT handshake, or the raw QUIC
/// handshake for a `quic://` URL, leaving stream opening to the caller. `hashes`
/// is the `server-certificate-hashes` property: empty accepts any certificate a
/// system root signs, otherwise only the listed certificates. `protocols` names the WebTransport subprotocols offered in
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
    if url.scheme() == RAW_QUIC_SCHEME {
        // raw QUIC states its version in the ALPN, so it has no subprotocols to offer
        if !protocols.is_empty() {
            return Err(G2gError::NotConfigured);
        }
        return dial_raw(&url, hashes, congestion).await;
    }
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

pub(crate) const CERTIFICATE_PROP: PropertySpec = PropertySpec::new(
    "certificate",
    PropKind::Str,
    "path to the TLS certificate chain (PEM) the listening side presents",
);

pub(crate) const PRIVATE_KEY_PROP: PropertySpec = PropertySpec::new(
    "private-key",
    PropKind::Str,
    "path to the TLS private key (PEM) for `certificate`",
);

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
        let bytes = hex_decode(&item.replace(':', "")).ok_or(G2gError::NotConfigured)?;
        // A SHA-256 digest and nothing else: the verifier compares raw bytes, so
        // a wrong-length entry would silently never match.
        if bytes.len() != 32 {
            return Err(G2gError::NotConfigured);
        }
        out.push(bytes);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn listener_location_binds_an_ip_literal_for_either_scheme() {
        let v4: SocketAddr = "127.0.0.1:4433".parse().unwrap();
        let v6: SocketAddr = "[::1]:4433".parse().unwrap();
        assert_eq!(bind_address("quic://127.0.0.1:4433"), Ok(v4));
        assert_eq!(bind_address("https://127.0.0.1:4433"), Ok(v4));
        assert_eq!(bind_address("quic://[::1]:4433"), Ok(v6));
        assert_eq!(bind_address("https://[::1]:4433/path"), Ok(v6));
        // a listener binds an address, so a name or a missing port is refused
        assert_eq!(
            bind_address("quic://localhost:4433"),
            Err(G2gError::NotConfigured)
        );
        assert_eq!(
            bind_address("quic://127.0.0.1"),
            Err(G2gError::NotConfigured)
        );
    }

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
