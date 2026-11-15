// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Remote WebTransport sink (`webtransport` feature): the WebTransport send
//! half of the distributed-graph primitive, the sibling of the TCP
//! [`RemoteSink`](crate::remotesink) and the WebSocket
//! [`RemoteWsSink`](crate::remotewssink).
//!
//! `RemoteWtSink` accepts *any* caps and forwards the whole `PipelinePacket`
//! stream (the leading `CapsChanged`, `Segment`, every `DataFrame`, mid-stream
//! caps refinement, `Flush`, `Eos`) over one WebTransport session, each packet
//! serialized by [`g2g_core::wire`]. The session's single bidirectional stream is
//! a QUIC stream, so it is an ordered reliable *byte* stream: the framing is the
//! TCP pair's `u32` length prefix, not the WebSocket pair's one-message-per-packet.
//! The receiving half is [`RemoteWtSrc`](crate::remotewtsrc).
//!
//! What WebTransport adds over the WebSocket carrier is the QUIC connection under
//! it: head-of-line blocking is per stream rather than per connection, the
//! handshake is 1-RTT, and a browser peer can speak it directly (`new
//! WebTransport(url)`) without a TLS-terminating proxy in front. Only CPU-memory
//! frames cross the wire; a device-resident frame yields
//! [`G2gError::UnsupportedDomain`](g2g_core::G2gError), exactly as any CPU sink
//! already requires.
//!
//! `RemoteWtSink` is the client by default (it dials the [`RemoteWtSrc`]
//! server); `mode=listener` makes it bind `location` and wait for a calling
//! source instead. A `quic://` location is raw QUIC, `https://` WebTransport. The QUIC +
//! HTTP/3 CONNECT handshake is async, so as with the WebSocket sink the connect is
//! deferred to the first `process` call, where a runtime context is guaranteed.
//! QUIC is always TLS: the server's certificate must chain to a system root, or be
//! named by its SHA-256 digest in `server-certificate-hashes` (what a browser's
//! `serverCertificateHashes` option does for a self-signed certificate).
//!
//! `video-mode` picks how frames travel, see [`remotewtio`](crate::remotewtio).
//! In the modes that reset or drop frames, `frame-timeout` asks the encoder for
//! a keyframe once a frame stream goes unacknowledged that long, and a keyframe
//! request from the receiving source reaches the encoder the same way.
//! `congestion-control` picks the QUIC controller (CUBIC by default, BBR under
//! `low-latency`).
//!
//! The shared client machinery (caps-dedup, the reconnect/retry `deliver` loop,
//! the `AsyncElement` glue) lives in [`RemoteClient`](crate::remoteclient); this
//! file supplies only the WebTransport transport (`WtClient`).

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use core::time::Duration;

use std::net::SocketAddr;
use std::sync::Arc;

use tokio::sync::Mutex;

use g2g_core::{
    AsyncElement, G2gError, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::datachannel::{data_channel, DataChannelEnds};
use crate::remoteclient::{PacketClient, RemoteClient};
use crate::remotesource::TransportFuture;
use crate::remotewtio::{
    self, Connected, Listeners, QuicMode, SelfSignedCertificate, SharedDataChannel, VideoSettings,
    WtStream,
};
use crate::remotewtvideo::{set_video_mode, VideoMode};
use crate::remotewtws::WsLink;

/// WebTransport `RemoteWtSink`: a length-framed [`g2g_core::wire`] stream over one
/// bidirectional WebTransport stream, received by
/// [`RemoteWtSrc`](crate::remotewtsrc).
pub type RemoteWtSink = RemoteClient<WtClient>;

impl RemoteWtSink {
    /// Send the packet stream to `url` (the [`RemoteWtSrc`](crate::remotewtsrc)
    /// server, e.g. `https://127.0.0.1:9603`).
    pub fn new(url: impl Into<String>) -> Self {
        RemoteClient::from_transport(WtClient::new(url))
    }

    /// Accept only server certificates whose SHA-256 digest is listed (hex,
    /// comma-separated), instead of requiring a system root. The same knob as the
    /// `server-certificate-hashes` property, set through the same path so a
    /// builder and a launch line cannot drift.
    pub fn with_server_certificate_hashes(mut self, hashes: impl Into<String>) -> Self {
        AsyncElement::set_property(
            &mut self,
            "server-certificate-hashes",
            PropValue::Str(hashes.into()),
        )
        .expect("server-certificate-hashes is a string");
        self
    }

    pub fn with_listener(mut self, cert: impl Into<String>, key: impl Into<String>) -> Self {
        let transport = self.transport_mut();
        transport.mode = QuicMode::Listener;
        transport.cert = cert.into();
        transport.key = key.into();
        self
    }

    pub fn with_self_signed_listener(mut self, certificate: SelfSignedCertificate) -> Self {
        let transport = self.transport_mut();
        transport.mode = QuicMode::Listener;
        transport.self_signed = Some(certificate);
        self
    }

    pub fn listen(&mut self) -> Result<SocketAddr, G2gError> {
        self.transport_mut().listeners()?.local_addr()
    }

    pub fn open_data_channel(&mut self) -> DataChannelEnds {
        let (application, transport) = data_channel();
        self.transport_mut().data_channel = Some(Arc::new(Mutex::new(transport)));
        application
    }

    pub fn with_video_mode(mut self, mode: VideoMode) -> Self {
        AsyncElement::set_property(&mut self, "video-mode", PropValue::Str(mode.name().into()))
            .expect("video-mode names a mode");
        self
    }
}

/// WebTransport transport for [`RemoteClient`] (and, read back, for the
/// [`RemoteWtTransform`](crate::remotewttransform) round trip).
#[derive(Debug)]
pub struct WtClient {
    /// WebTransport URL of the server (e.g. `https://127.0.0.1:9603`).
    url: String,
    /// Comma-separated hex SHA-256 digests of the accepted server certificates;
    /// empty means "anything a system root signs".
    cert_hashes: String,
    /// The `congestion-control` nick applied to the QUIC connection.
    congestion: String,
    mode: QuicMode,
    cert: String,
    key: String,
    self_signed: Option<SelfSignedCertificate>,
    listeners: Option<Listeners>,
    video_mode: VideoMode,
    frame_timeout_ms: u64,
    send_window: u64,
    /// Opened lazily on the first send (the QUIC / CONNECT handshake is async).
    stream: Option<WtStream>,
    data_channel: Option<SharedDataChannel>,
}

impl WtClient {
    /// Dial `url` on first use.
    pub(crate) fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            cert_hashes: String::new(),
            congestion: "default".to_string(),
            mode: QuicMode::Caller,
            cert: String::new(),
            key: String::new(),
            self_signed: None,
            listeners: None,
            video_mode: VideoMode::Stream,
            frame_timeout_ms: DEFAULT_FRAME_TIMEOUT_MS,
            send_window: QUINN_SEND_WINDOW,
            stream: None,
            data_channel: None,
        }
    }

    /// The live session stream, for the transform's reply read.
    fn listeners(&mut self) -> Result<&Listeners, G2gError> {
        if self.listeners.is_none() {
            let (chain, key) = match &self.self_signed {
                Some(certificate) => remotewtio::parse_certificate(
                    certificate.certificate_pem.as_bytes(),
                    certificate.private_key_pem.as_bytes(),
                )?,
                None => remotewtio::load_certificate(&self.cert, &self.key)?,
            };
            let bind = remotewtio::bind_address(&self.url)?;
            self.listeners = Some(remotewtio::listen_with_fallback(
                bind,
                chain,
                key,
                &self.congestion,
            )?);
        }
        self.listeners.as_ref().ok_or(G2gError::NotConfigured)
    }

    pub(crate) fn stream_mut(&mut self) -> Option<&mut WtStream> {
        self.stream.as_mut()
    }

    /// Shared by the sink and the transform: the knobs a client has. `video-mode`
    /// and `frame-timeout` are the sink's alone (see the transform's module docs),
    /// but they are stored here because the transport is.
    pub(crate) fn set_client_prop(
        &mut self,
        name: &str,
        value: &PropValue,
    ) -> Option<Result<(), PropError>> {
        match name {
            "congestion-control" => {
                return Some(remotewtio::set_congestion(&mut self.congestion, value))
            }
            "video-mode" => return Some(set_video_mode(&mut self.video_mode, value)),
            "frame-timeout" => {
                return Some(match value.as_uint() {
                    Some(milliseconds) => {
                        self.frame_timeout_ms = milliseconds;
                        Ok(())
                    }
                    None => Err(PropError::Type),
                })
            }
            "send-window" => {
                return Some(match value.as_uint() {
                    Some(0) => Err(PropError::Value),
                    Some(bytes) => {
                        self.send_window = bytes;
                        Ok(())
                    }
                    None => Err(PropError::Type),
                })
            }
            "mode" => return Some(remotewtio::set_mode(&mut self.mode, value)),
            _ => {}
        }
        let target = match name {
            "location" => &mut self.url,
            "server-certificate-hashes" => &mut self.cert_hashes,
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

    pub(crate) fn get_client_prop(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.url.clone())),
            "server-certificate-hashes" => Some(PropValue::Str(self.cert_hashes.clone())),
            "congestion-control" => Some(PropValue::Str(self.congestion.clone())),
            "video-mode" => Some(PropValue::Str(self.video_mode.name().to_string())),
            "frame-timeout" => Some(PropValue::Uint(self.frame_timeout_ms)),
            "send-window" => Some(PropValue::Uint(self.send_window)),
            "mode" => Some(PropValue::Str(self.mode.name().to_string())),
            "certificate" => Some(PropValue::Str(self.cert.clone())),
            "private-key" => Some(PropValue::Str(self.key.clone())),
            _ => None,
        }
    }
}

// quinn's TransportConfig default, eight times its 1.25 MB stream receive window
const QUINN_SEND_WINDOW: u64 = 10_000_000;
const DEFAULT_FRAME_TIMEOUT_MS: u64 = 100;

/// The `server-certificate-hashes` spec, shared by the sink and the transform.
pub(crate) const CERT_HASHES_PROP: PropertySpec = PropertySpec::new(
    "server-certificate-hashes",
    PropKind::Str,
    "accept only server certificates with these SHA-256 digests (hex, comma-separated); empty = system roots",
);

impl PacketClient for WtClient {
    const NAME: &'static str = "Remote WebTransport sink";
    const DESCRIPTION: &'static str =
        "Serializes the PipelinePacket stream and sends it over WebTransport to a remote RemoteWtSrc";
    const PROPERTIES: &'static [PropertySpec] = &[
        PropertySpec::new(
            "location",
            PropKind::Str,
            "URL of the RemoteWtSrc server, https:// for WebTransport or quic:// for raw QUIC; in listener mode the address to bind",
        )
        .with_default("https://127.0.0.1:9603"),
        PropertySpec::new(
            "mode",
            PropKind::Str,
            "caller dials location, listener binds it and waits for one client",
        )
        .with_default("caller")
        .with_enum_values("caller | listener"),
        CERT_HASHES_PROP,
        remotewtio::CERTIFICATE_PROP,
        remotewtio::PRIVATE_KEY_PROP,
        remotewtio::CONGESTION_PROP,
        PropertySpec::new(
            "reconnect-attempts",
            PropKind::Uint,
            "retry a failed connect / send up to N times (0 = off)",
        )
        .with_default("0"),
        PropertySpec::new(
            "video-mode",
            PropKind::Str,
            "stream sends every packet in order on one stream; stream-per-frame sends each frame on its own stream and resets unacknowledged frames when a keyframe goes out; datagrams splits each frame into unreliable datagram shards plus Reed-Solomon repair sized to the path's loss",
        )
        .with_default("stream")
        .with_enum_values("stream | stream-per-frame | datagrams"),
        PropertySpec::new(
            "frame-timeout",
            PropKind::Uint,
            "milliseconds a frame stream may go unacknowledged before the encoder is asked for a keyframe (0 = never)",
        )
        .with_default("100"),
        PropertySpec::new(
            "send-window",
            PropKind::Uint,
            "bytes written but not yet acknowledged before a send blocks; smaller keeps a stalled link from queueing video",
        )
        .with_default("10000000"),
    ];

    fn is_connected(&self) -> bool {
        self.stream.is_some()
    }

    fn connect(&mut self) -> TransportFuture<'_, ()> {
        Box::pin(async move {
            let connected = match self.mode {
                QuicMode::Caller => Connected::Quic(Box::new(
                    remotewtio::dial(&self.url, &self.cert_hashes, &[], &self.congestion).await?,
                )),
                QuicMode::Listener => remotewtio::accept_either(self.listeners()?).await?,
            };
            let session = match connected {
                Connected::Quic(session) => *session,
                Connected::WebSocket(websocket) => {
                    let link = WsLink::new(*websocket, self.data_channel.as_ref()).await;
                    self.stream = Some(WtStream::WebSocket(link));
                    return Ok(());
                }
            };
            session.set_send_window(self.send_window);
            let settings = VideoSettings {
                mode: self.video_mode,
                stall_timeout: (self.frame_timeout_ms != 0)
                    .then(|| Duration::from_millis(self.frame_timeout_ms)),
                missing_frame_timeout: None,
            };
            let stream =
                remotewtio::open_stream(session, settings, self.data_channel.as_ref()).await?;
            self.stream = Some(WtStream::Quic(Box::new(stream)));
            Ok(())
        })
    }

    fn send<'a>(&'a mut self, packet: &'a PipelinePacket) -> TransportFuture<'a, ()> {
        Box::pin(async move {
            let stream = self.stream.as_mut().ok_or(G2gError::NotConfigured)?;
            stream.send(packet).await
        })
    }

    fn reset(&mut self) {
        self.stream = None;
    }

    fn take_keyframe_request(&mut self) -> bool {
        self.stream
            .as_mut()
            .is_some_and(WtStream::take_keyframe_request)
    }

    fn close(&mut self) -> TransportFuture<'_, ()> {
        Box::pin(async move {
            if let Some(stream) = self.stream.as_mut() {
                stream.finish().await;
            }
            Ok(())
        })
    }

    fn configure_connect(&mut self, _eager: bool) -> Result<(), G2gError> {
        // The QUIC + CONNECT handshake is async and needs a runtime, so the
        // connect is always deferred to the first send (as for the WebSocket sink).
        Ok(())
    }

    fn set_transport_prop(
        &mut self,
        name: &str,
        value: &PropValue,
    ) -> Option<Result<(), PropError>> {
        self.set_client_prop(name, value)
    }

    fn get_transport_prop(&self, name: &str) -> Option<PropValue> {
        self.get_client_prop(name)
    }
}
