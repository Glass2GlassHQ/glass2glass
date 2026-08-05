//! Remote WebTransport transform (M901, `webtransport` feature): a media-agnostic
//! *remote stage* over WebTransport, the sibling of
//! [`RemoteWsTransform`](crate::remotewstransform). Where the sink / source pair
//! cuts an edge one-way, this keeps the graph shape and offloads a single middle
//! stage: each input packet goes to a remote peer over the session's one
//! bidirectional stream and the processed packet it returns is emitted downstream.
//!
//! The shared machinery (the FIFO round-trip protocol, caps dedup, the
//! `AsyncElement` glue) lives in [`RemoteTransform`](crate::remotetransform); this
//! file supplies only the WebTransport transport's transform role. The transport
//! itself is the sink's [`WtClient`](crate::remotewtsink::WtClient).
//!
//! The sink's `datagrams` carrier is deliberately not offered here: the round
//! trip pairs one reply with each request, so a dropped request would leave the
//! stage waiting for a reply that is never sent. `congestion-control` is offered,
//! it is a property of the connection rather than of the framing.

use alloc::boxed::Box;
use alloc::string::String;

use g2g_core::{AsyncElement, G2gError, PipelinePacket, PropKind, PropValue, PropertySpec};

use crate::remotesource::TransportFuture;
use crate::remotetransform::{PacketDuplex, RemoteTransform};
use crate::remotewtsink::{WtClient, CERT_HASHES_PROP};

/// WebTransport `RemoteWtTransform`: the length-framed [`g2g_core::wire`] stream
/// out and back over one bidirectional WebTransport stream, against a peer running
/// the offloaded stage.
pub type RemoteWtTransform = RemoteTransform<WtClient>;

impl RemoteWtTransform {
    /// Offload the middle stage to `url` (a remote peer that reads the wire
    /// stream, processes each frame, and replies one processed frame each).
    pub fn new(url: impl Into<String>) -> Self {
        RemoteTransform::from_transport(WtClient::new(url))
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
}

impl PacketDuplex for WtClient {
    const NAME: &'static str = "Remote WebTransport transform";
    const DESCRIPTION: &'static str = "Offloads a middle stage: ships each frame to a remote peer over WebTransport and emits the processed frame it returns";
    const PROPERTIES: &'static [PropertySpec] = &[
        PropertySpec::new(
            "location",
            PropKind::Str,
            "WebTransport URL of the remote stage server (e.g. https://host:port)",
        )
        .with_default("https://127.0.0.1:9604"),
        CERT_HASHES_PROP,
        crate::remotewtio::CONGESTION_PROP,
    ];

    fn recv(&mut self) -> TransportFuture<'_, Option<PipelinePacket>> {
        Box::pin(async move {
            let stream = self.stream_mut().ok_or(G2gError::NotConfigured)?;
            stream.recv().await
        })
    }
}
