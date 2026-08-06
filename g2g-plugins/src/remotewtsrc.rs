//! Remote WebTransport source (M901, `webtransport` feature): the WebTransport
//! receive half of the distributed-graph primitive, the inverse of
//! [`RemoteWtSink`](crate::remotewtsink) and the sibling of the TCP
//! [`RemoteSrc`](crate::remotesrc) / WebSocket [`RemoteWsSrc`](crate::remotewssrc).
//!
//! `RemoteWtSrc` is the WebTransport *server*: it binds a QUIC endpoint, accepts
//! one session (answering its HTTP/3 CONNECT with 200 whatever path it asks for,
//! since the element is media-agnostic), takes the client's single bidirectional
//! stream, and reconstructs the `PipelinePacket` stream the sender serialized
//! ([`g2g_core::wire`], `u32` length-framed as on TCP). The stream's first packet
//! is the sender's negotiated `CapsChanged`, so the source *discovers* the media
//! type from the wire in `intercept_caps` (the async caps-discovery pattern
//! `RtspSrc` / `RemoteSrc` use), then re-emits the leading `CapsChanged` and every
//! subsequent packet in `run`, ending on the sender's `Eos` (or the stream's end).
//!
//! QUIC is always TLS, so unlike the TCP and WebSocket servers this one cannot
//! start without a certificate: `certificate` / `private-key` are PEM file paths.
//! A browser peer that trusts a self-signed certificate by digest
//! (`serverCertificateHashes`) requires the certificate be ECDSA P-256 and valid
//! for at most 14 days, a peer-side constraint this element does not police.
//!
//! There is no receive-side carrier switch: a sender running the `datagrams`
//! carrier of [`RemoteWtSink`](crate::remotewtsink) puts its data frames in QUIC
//! datagrams and its control packets on the stream, and this source takes packets
//! from both, so the two sides cannot be configured out of step. Drops are the
//! sender's chosen trade there and simply leave gaps.
//!
//! The shared server machinery lives in [`RemoteSource`](crate::remotesource);
//! this file supplies only the WebTransport transport (`WtTransport`).

use alloc::boxed::Box;
use alloc::string::{String, ToString};

use std::net::{SocketAddr, TcpListener as StdTcpListener};

use web_transport_quinn::{Server, ServerBuilder};

use g2g_core::runtime::SourceLoop;
use g2g_core::{Caps, G2gError, PipelinePacket, PropError, PropKind, PropValue, PropertySpec};

use crate::remotesource::{leading_caps, PacketTransport, RemoteSource, TransportFuture};
use crate::remotewtio::{
    congestion_control, load_certificate, set_congestion, wt_err, WtStream, CONGESTION_PROP,
};

/// WebTransport `RemoteWtSrc`: a length-framed [`g2g_core::wire`] stream over one
/// bidirectional WebTransport stream, from a [`RemoteWtSink`](crate::remotewtsink)
/// or any peer speaking the same codec (a browser `WebTransport` client included).
pub type RemoteWtSrc = RemoteSource<WtTransport>;

impl RemoteWtSrc {
    /// The TLS certificate chain and private key (PEM file paths) the QUIC server
    /// presents. Required: there is no plaintext WebTransport. The same knobs as
    /// the `certificate` / `private-key` properties, set through the same path so
    /// a builder and a launch line cannot drift.
    pub fn with_certificate(mut self, cert: impl Into<String>, key: impl Into<String>) -> Self {
        SourceLoop::set_property(&mut self, "certificate", PropValue::Str(cert.into()))
            .expect("certificate is a string");
        SourceLoop::set_property(&mut self, "private-key", PropValue::Str(key.into()))
            .expect("private-key is a string");
        self
    }
}

/// WebTransport transport for [`RemoteSource`].
#[derive(Debug)]
pub struct WtTransport {
    cert: String,
    key: String,
    /// The `congestion-control` nick applied to accepted connections.
    congestion: String,
}

impl Default for WtTransport {
    fn default() -> Self {
        Self {
            cert: String::new(),
            key: String::new(),
            congestion: "default".to_string(),
        }
    }
}

impl PacketTransport for WtTransport {
    type Conn = WtStream;
    type Listener = Server;
    const NAME: &'static str = "Remote WebTransport source";
    const DESCRIPTION: &'static str =
        "Receives a serialized PipelinePacket stream over WebTransport from a remote RemoteWtSink";
    const PROPERTIES: &'static [PropertySpec] = &[
        PropertySpec::new(
            "address",
            PropKind::Str,
            "local bind address (IP to listen on)",
        )
        .with_default("0.0.0.0"),
        PropertySpec::new(
            "port",
            PropKind::Uint,
            "local UDP port to listen on for WebTransport (QUIC) clients",
        )
        .with_range("0", "65535"),
        PropertySpec::new(
            "keep-listening",
            PropKind::Bool,
            "accept a replacement client when one drops without Eos",
        )
        .with_default("false"),
        PropertySpec::new(
            "certificate",
            PropKind::Str,
            "path to the TLS certificate chain (PEM) the QUIC server presents",
        ),
        PropertySpec::new(
            "private-key",
            PropKind::Str,
            "path to the TLS private key (PEM) for `certificate`",
        ),
        CONGESTION_PROP,
    ];

    fn listen(
        &mut self,
        bind: SocketAddr,
        // A QUIC endpoint binds its own UDP socket, so there is no pre-bound TCP
        // listener to adopt (`from_listener` is constrained to the TCP carriers);
        // `RemoteSource::listen` reports the ephemeral port instead.
        _adopt: Option<StdTcpListener>,
    ) -> TransportFuture<'_, Self::Listener> {
        Box::pin(async move {
            let (chain, key) = load_certificate(&self.cert, &self.key)?;
            ServerBuilder::new()
                .with_addr(bind)
                .with_congestion_control(congestion_control(&self.congestion))
                .with_certificate(chain, key)
                .map_err(wt_err)
        })
    }

    fn listen_addr(listener: &Self::Listener) -> Option<SocketAddr> {
        listener.local_addr().ok()
    }

    fn accept(listener: &mut Self::Listener) -> TransportFuture<'_, (Self::Conn, Caps)> {
        Box::pin(async move {
            let request = listener.accept().await.ok_or(G2gError::NotConfigured)?;
            let session = request.ok().await.map_err(wt_err)?;
            // The sender opens exactly one bidirectional stream and writes the
            // wire codec into it.
            let (tx, rx) = session.accept_bi().await.map_err(wt_err)?;
            // Receiving is mode-free: `WtStream::recv` takes packets off the
            // stream or the datagram flow, whichever the sender used.
            let mut conn = WtStream::new(session, tx, rx, false);
            let caps = leading_caps(conn.recv().await?)?;
            Ok((conn, caps))
        })
    }

    fn recv(conn: &mut Self::Conn) -> TransportFuture<'_, Option<PipelinePacket>> {
        Box::pin(async move { conn.recv().await })
    }

    fn set_transport_prop(
        &mut self,
        name: &str,
        value: &PropValue,
    ) -> Option<Result<(), PropError>> {
        if name == "congestion-control" {
            return Some(set_congestion(&mut self.congestion, value));
        }
        let target = match name {
            "certificate" => &mut self.cert,
            "private-key" => &mut self.key,
            _ => return None,
        };
        Some(match value.as_str() {
            Some(s) => {
                *target = s.to_string();
                Ok(())
            }
            None => Err(PropError::Type),
        })
    }

    fn get_transport_prop(&self, name: &str) -> Option<PropValue> {
        match name {
            "certificate" => Some(PropValue::Str(self.cert.clone())),
            "private-key" => Some(PropValue::Str(self.key.clone())),
            "congestion-control" => Some(PropValue::Str(self.congestion.clone())),
            _ => None,
        }
    }
}
