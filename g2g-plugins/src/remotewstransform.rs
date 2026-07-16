//! Remote WebSocket transform (M555, `remote-ws` feature): a media-agnostic
//! *remote stage*. Where the M554 `RemoteWsSink` / `RemoteWsSrc` pair cuts an
//! edge one-way (the whole downstream subgraph runs remotely), `RemoteWsTransform`
//! keeps the graph shape and offloads a single middle stage: it ships each input
//! packet to a remote peer over one WebSocket and emits the processed packet it
//! gets back. That is the shape a browser detection offload needs, where the
//! stages *around* the remote one (decode, overlay, present) must stay local: the
//! bidirectional, round-trip generalization of the bespoke M549 `WebRemoteDetect`
//! shim (which hand-rolled an RGBA-up / boxes-down protocol that knew about
//! detection). Here the element knows nothing about the stage it offloads; the
//! remote peer runs whatever g2g subgraph it likes and returns a processed packet.
//!
//! Caps are identity (pixels and geometry pass through; the remote stage may
//! attach `metadata`, e.g. `AnalyticsMeta` detections, which crosses the wire
//! in band). Protocol over the single socket, kept strictly FIFO so each
//! per-frame read pairs with its own frame:
//!   client -> peer: the leading `CapsChanged` (config, no reply), then one
//!                   `DataFrame` per frame, then `Eos`.
//!   peer -> client: exactly one processed `DataFrame` per `DataFrame` received;
//!                   no echoed caps / segment / control.
//! `Segment` / `Flush` therefore pass through locally (they are not sent to the
//! peer), so the reply stream stays one-packet-per-frame. Per-frame timing still
//! crosses (the wire codec carries each frame's `FrameTiming`).
//!
//! Bandwidth note: this round-trips the whole frame both ways (unlike M549's
//! bespoke frame-up / boxes-down), the honest cost of a generic packet-in /
//! packet-out transform. Fine on a LAN / localhost; a `metadata`-only return
//! (retain the frame locally, receive only the attached meta) is a future
//! optimization for the pixels-unchanged case.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::{String, ToString};

use tokio::net::TcpStream;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError,
    HardwareError, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec,
};

use crate::remotewsio::{recv_wire, send_wire, ws_err};

#[derive(Debug)]
pub struct RemoteWsTransform {
    /// WebSocket URL of the remote stage server (e.g. `ws://127.0.0.1:9602`).
    url: String,
    /// Opened lazily on the first `process` (the handshake is async).
    socket: Option<WebSocketStream<MaybeTlsStream<TcpStream>>>,
    /// Caps recorded in `configure_pipeline`, sent to the peer (deduped against
    /// `last_sent`) as the leading `CapsChanged` so its subgraph configures.
    configured_caps: Option<Caps>,
    last_sent: Option<Caps>,
    configured: bool,
    emitted: u64,
}

impl RemoteWsTransform {
    /// Offload the middle stage to `url` (a remote peer that reads the wire
    /// stream, processes each frame, and replies one processed frame each).
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            socket: None,
            configured_caps: None,
            last_sent: None,
            configured: false,
            emitted: 0,
        }
    }

    /// Count of processed frames emitted downstream. Useful in tests.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// Send the current caps to the peer as an ordered `CapsChanged`, once,
    /// deduped, so its subgraph configures before the first frame.
    async fn send_caps_if_new(&mut self) -> Result<(), G2gError> {
        if let Some(caps) = self.configured_caps.clone() {
            if self.last_sent.as_ref() != Some(&caps) {
                let sock = self.socket.as_mut().ok_or(G2gError::NotConfigured)?;
                send_wire(sock, &PipelinePacket::CapsChanged(caps.clone())).await?;
                self.last_sent = Some(caps);
            }
        }
        Ok(())
    }
}

impl AsyncElement for RemoteWsTransform {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Identity, media-agnostic: whatever arrives is what flows on (the remote
        // stage may attach metadata but does not change the format).
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| CapsSet::one(input.clone())))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // The WebSocket handshake is async, so the connect is deferred to
        // `process`; record the caps for the leading wire CapsChanged.
        self.configured_caps = Some(absolute_caps.clone());
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            // Open the socket once, on first use.
            if self.socket.is_none() {
                let (socket, _resp) = connect_async(&self.url).await.map_err(ws_err)?;
                self.socket = Some(socket);
            }
            // The peer's subgraph must see the caps before the first frame.
            self.send_caps_if_new().await?;

            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let sock = self.socket.as_mut().ok_or(G2gError::NotConfigured)?;
                    send_wire(sock, &PipelinePacket::DataFrame(frame)).await?;
                    // Exactly one processed packet comes back per frame (the peer
                    // never echoes control), so this read pairs with our frame.
                    let processed = recv_wire(sock)
                        .await?
                        .ok_or(G2gError::Hardware(HardwareError::Other))?;
                    self.emitted += 1;
                    out.push(processed).await?;
                }
                PipelinePacket::CapsChanged(caps) => {
                    // Forward mid-stream refinement to the peer (deduped) and
                    // downstream. The dedup above already sent it if unchanged.
                    if self.last_sent.as_ref() != Some(&caps) {
                        let sock = self.socket.as_mut().ok_or(G2gError::NotConfigured)?;
                        send_wire(sock, &PipelinePacket::CapsChanged(caps.clone())).await?;
                        self.last_sent = Some(caps.clone());
                    }
                    out.push(PipelinePacket::CapsChanged(caps)).await?;
                }
                PipelinePacket::Eos => {
                    // Tell the peer we are done and close; the runner's transform
                    // arm forwards EOS downstream, so we do not push it.
                    let sock = self.socket.as_mut().ok_or(G2gError::NotConfigured)?;
                    let _ = send_wire(sock, &PipelinePacket::Eos).await;
                    let _ = sock.close(None).await;
                }
                // Segment / Flush pass through locally (not sent to the peer, so
                // the reply stream stays one packet per frame).
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Remote WebSocket transform",
            "Filter/Network",
            "Offloads a middle stage: ships each frame to a remote peer over a WebSocket and emits the processed frame it returns",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[PropertySpec::new(
            "location",
            PropKind::Str,
            "WebSocket URL of the remote stage server (e.g. ws://host:port)",
        )
        .with_default("ws://127.0.0.1:9602")];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" => {
                self.url = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.url.clone())),
            _ => None,
        }
    }
}

impl PadTemplates for RemoteWsTransform {
    /// Wildcard sink (media-agnostic); the identity source side is expressed at
    /// runtime by `caps_constraint_as_transform`, so only the sink is declared
    /// statically (a wildcard source pad is degenerate).
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink_any()])
    }
}
