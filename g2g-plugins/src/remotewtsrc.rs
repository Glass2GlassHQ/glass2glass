// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Remote WebTransport source (`webtransport` feature): the WebTransport
//! receive half of the distributed-graph primitive, the inverse of
//! [`RemoteWtSink`](crate::remotewtsink) and the sibling of the TCP
//! [`RemoteSrc`](crate::remotesrc) / WebSocket [`RemoteWsSrc`](crate::remotewssrc).
//!
//! `RemoteWtSrc` is the server by default (`mode=caller` makes it dial a
//! listening sink at `location` instead): it binds a QUIC endpoint, accepts
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
//! There is no receive-side carrier switch: whichever `video-mode` the
//! [`RemoteWtSink`](crate::remotewtsink) runs, this source takes its packets and
//! restores send order, so the two sides cannot be configured out of step. A
//! datagram frame still missing after `frame-timeout` is skipped, with the
//! frames after it up to the next keyframe, and the sender is asked for a
//! keyframe. A keyframe request from downstream, a decoder that lost its
//! reference, goes to the sender the same way.
//!
//! The shared server machinery lives in [`RemoteSource`](crate::remotesource);
//! this file supplies only the WebTransport transport (`WtTransport`).

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use core::time::Duration;

use std::net::{SocketAddr, TcpListener as StdTcpListener};
use std::sync::Arc;

use tokio::sync::Mutex;

use web_transport_quinn::quinn::Endpoint;

use g2g_core::runtime::SourceLoop;
use g2g_core::{Caps, PipelinePacket, PropError, PropKind, PropValue, PropertySpec};

use crate::datachannel::{data_channel, DataChannelEnds};
use crate::remotesource::{leading_caps, PacketTransport, RemoteSource, TransportFuture};
use crate::remotewtio::{
    self, load_certificate, set_congestion, wt_err, QuicMode, SharedDataChannel, VideoSettings,
    WtStream, CERTIFICATE_PROP, CONGESTION_PROP, PRIVATE_KEY_PROP,
};
use crate::remotewtsink::CERT_HASHES_PROP;
use crate::remotewtvideo::VideoMode;

/// WebTransport `RemoteWtSrc`: a length-framed [`g2g_core::wire`] stream over one
/// bidirectional WebTransport stream, from a [`RemoteWtSink`](crate::remotewtsink)
/// or any peer speaking the same codec (a browser `WebTransport` client included).
pub type RemoteWtSrc = RemoteSource<WtTransport>;

impl RemoteWtSrc {
    /// The TLS certificate chain and private key (PEM file paths) the QUIC server
    /// presents. Required: there is no plaintext WebTransport. The same knobs as
    /// the `certificate` / `private-key` properties, set through the same path so
    /// a builder and a launch line cannot drift.
    pub fn with_caller(mut self, location: impl Into<String>, hashes: impl Into<String>) -> Self {
        SourceLoop::set_property(&mut self, "mode", PropValue::Str("caller".into()))
            .expect("mode is a string");
        SourceLoop::set_property(&mut self, "location", PropValue::Str(location.into()))
            .expect("location is a string");
        SourceLoop::set_property(
            &mut self,
            "server-certificate-hashes",
            PropValue::Str(hashes.into()),
        )
        .expect("server-certificate-hashes is a string");
        self
    }

    pub fn open_data_channel(&mut self) -> DataChannelEnds {
        let transport = self.transport_mut();
        assert!(
            !transport.listened,
            "the data channel opens before the source listens"
        );
        let (application, transport_ends) = data_channel();
        transport.data_channel = Some(Arc::new(Mutex::new(transport_ends)));
        application
    }

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
    mode: QuicMode,
    location: String,
    cert_hashes: String,
    cert: String,
    key: String,
    /// The `congestion-control` nick applied to accepted connections.
    congestion: String,
    frame_timeout_ms: u64,
    data_channel: Option<SharedDataChannel>,
    listened: bool,
}

impl Default for WtTransport {
    fn default() -> Self {
        Self {
            mode: QuicMode::Listener,
            location: String::new(),
            cert_hashes: String::new(),
            cert: String::new(),
            key: String::new(),
            congestion: "default".to_string(),
            frame_timeout_ms: 0,
            data_channel: None,
            listened: false,
        }
    }
}

#[derive(Debug)]
pub struct WtListener {
    kind: ListenerKind,
    frame_timeout_ms: u64,
    data_channel: Option<SharedDataChannel>,
}

#[derive(Debug)]
enum ListenerKind {
    Bound(Endpoint),
    Caller {
        location: String,
        cert_hashes: String,
        congestion: String,
    },
}

impl PacketTransport for WtTransport {
    type Conn = WtStream;
    type Listener = WtListener;
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
        CERTIFICATE_PROP,
        PRIVATE_KEY_PROP,
        CONGESTION_PROP,
        PropertySpec::new(
            "mode",
            PropKind::Str,
            "listener binds address:port and waits for one sender, caller dials location",
        )
        .with_default("listener")
        .with_enum_values("caller | listener"),
        PropertySpec::new(
            "location",
            PropKind::Str,
            "in caller mode, the sender's URL: https:// for WebTransport or quic:// for raw QUIC",
        ),
        CERT_HASHES_PROP,
        PropertySpec::new(
            "frame-timeout",
            PropKind::Uint,
            "milliseconds a missing datagram frame is waited for before it is skipped and a keyframe requested (0 = one frame period of the caps)",
        )
        .with_default("0"),
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
            self.listened = true;
            let kind = match self.mode {
                QuicMode::Caller => ListenerKind::Caller {
                    location: self.location.clone(),
                    cert_hashes: self.cert_hashes.clone(),
                    congestion: self.congestion.clone(),
                },
                QuicMode::Listener => {
                    let (chain, key) = load_certificate(&self.cert, &self.key)?;
                    ListenerKind::Bound(remotewtio::listen(bind, chain, key, &self.congestion)?)
                }
            };
            Ok(WtListener {
                kind,
                frame_timeout_ms: self.frame_timeout_ms,
                data_channel: self.data_channel.clone(),
            })
        })
    }

    fn listen_addr(listener: &Self::Listener) -> Option<SocketAddr> {
        match &listener.kind {
            ListenerKind::Bound(endpoint) => endpoint.local_addr().ok(),
            ListenerKind::Caller { .. } => None,
        }
    }

    fn accept(listener: &mut Self::Listener) -> TransportFuture<'_, (Self::Conn, Caps)> {
        Box::pin(async move {
            let session = match &listener.kind {
                ListenerKind::Bound(endpoint) => remotewtio::accept(endpoint).await?,
                ListenerKind::Caller {
                    location,
                    cert_hashes,
                    congestion,
                } => remotewtio::dial(location, cert_hashes, &[], congestion).await?,
            };
            let pump = listener
                .data_channel
                .as_ref()
                .map(|channel| remotewtio::open_data_channel(&session, channel.clone()));
            // The sender opens exactly one bidirectional stream and writes the
            // wire codec into it.
            let (tx, rx) = session.accept_bi().await.map_err(wt_err)?;
            // Receiving is mode-free: `WtStream::recv` takes packets off the
            // stream or the datagram flow, whichever the sender used.
            let settings = VideoSettings {
                mode: VideoMode::Stream,
                stall_timeout: None,
                missing_frame_timeout: (listener.frame_timeout_ms != 0)
                    .then(|| Duration::from_millis(listener.frame_timeout_ms)),
            };
            let mut conn = WtStream::new(session, tx, rx, settings, pump);
            let caps = leading_caps(conn.recv().await?)?;
            Ok((conn, caps))
        })
    }

    fn recv(conn: &mut Self::Conn) -> TransportFuture<'_, Option<PipelinePacket>> {
        Box::pin(async move { conn.recv().await })
    }

    fn request_keyframe(conn: &mut Self::Conn) -> TransportFuture<'_, ()> {
        Box::pin(async move { conn.request_keyframe().await })
    }

    fn set_transport_prop(
        &mut self,
        name: &str,
        value: &PropValue,
    ) -> Option<Result<(), PropError>> {
        match name {
            "congestion-control" => return Some(set_congestion(&mut self.congestion, value)),
            "mode" => return Some(remotewtio::set_mode(&mut self.mode, value)),
            "frame-timeout" => {
                return Some(match value.as_uint() {
                    Some(milliseconds) => {
                        self.frame_timeout_ms = milliseconds;
                        Ok(())
                    }
                    None => Err(PropError::Type),
                })
            }
            _ => {}
        }
        let target = match name {
            "certificate" => &mut self.cert,
            "private-key" => &mut self.key,
            "location" => &mut self.location,
            "server-certificate-hashes" => &mut self.cert_hashes,
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
            "mode" => Some(PropValue::Str(self.mode.name().to_string())),
            "frame-timeout" => Some(PropValue::Uint(self.frame_timeout_ms)),
            "location" => Some(PropValue::Str(self.location.clone())),
            "server-certificate-hashes" => Some(PropValue::Str(self.cert_hashes.clone())),
            _ => None,
        }
    }
}
