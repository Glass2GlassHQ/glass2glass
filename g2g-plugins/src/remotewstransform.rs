//! Remote WebSocket transform (M555, `remote-ws` feature): a media-agnostic
//! *remote stage* over a WebSocket. Where the M554 `RemoteWsSink` / `RemoteWsSrc`
//! pair cuts an edge one-way, `RemoteWsTransform` keeps the graph shape and
//! offloads a single middle stage, the bidirectional generalization of the bespoke
//! M549 `WebRemoteDetect` shim (which hand-rolled an RGBA-up / boxes-down protocol
//! that knew about detection).
//!
//! The shared machinery (the FIFO round-trip protocol, caps dedup, the
//! `AsyncElement` glue) lives in [`RemoteTransform`](crate::remotetransform); this
//! file supplies only the WebSocket transport's transform role. The transport
//! itself is the sink's [`WsClient`](crate::remotewssink::WsClient).

use alloc::boxed::Box;
use alloc::string::String;

use g2g_core::{G2gError, PipelinePacket, PropKind, PropertySpec};

use crate::remotesource::TransportFuture;
use crate::remotetransform::{PacketDuplex, RemoteTransform};
use crate::remotewsio::recv_wire;
use crate::remotewssink::WsClient;

/// WebSocket `RemoteWsTransform`: one [`g2g_core::wire`] packet per binary
/// WebSocket message, out and back, against a peer running the offloaded stage.
pub type RemoteWsTransform = RemoteTransform<WsClient>;

impl RemoteWsTransform {
    /// Offload the middle stage to `url` (a remote peer that reads the wire
    /// stream, processes each frame, and replies one processed frame each).
    pub fn new(url: impl Into<String>) -> Self {
        RemoteTransform::from_transport(WsClient::new(url))
    }
}

impl PacketDuplex for WsClient {
    const NAME: &'static str = "Remote WebSocket transform";
    const DESCRIPTION: &'static str = "Offloads a middle stage: ships each frame to a remote peer over a WebSocket and emits the processed frame it returns";
    const PROPERTIES: &'static [PropertySpec] = &[PropertySpec::new(
        "location",
        PropKind::Str,
        "WebSocket URL of the remote stage server (e.g. ws://host:port)",
    )
    .with_default("ws://127.0.0.1:9602")];

    fn recv(&mut self) -> TransportFuture<'_, Option<PipelinePacket>> {
        Box::pin(async move {
            let sock = self.socket_mut().ok_or(G2gError::NotConfigured)?;
            recv_wire(sock).await
        })
    }
}
