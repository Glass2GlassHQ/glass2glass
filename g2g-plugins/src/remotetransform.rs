//! Shared core for the distributed-graph *remote transform* elements.
//!
//! Where a `Remote*Sink` / `Remote*Src` pair cuts an edge one-way (the whole
//! downstream subgraph runs remotely), a remote transform keeps the graph shape
//! and offloads a single middle stage: it ships each input packet to a remote peer
//! over one connection and emits the processed packet it gets back. That is the
//! shape a browser detection offload needs, where the stages *around* the remote
//! one (decode, overlay, present) must stay local. The element knows nothing about
//! the stage it offloads; the peer runs whatever g2g subgraph it likes and returns
//! a processed packet.
//!
//! Caps are identity (pixels and geometry pass through; the remote stage may
//! attach `metadata`, e.g. `AnalyticsMeta` detections, which crosses the wire in
//! band). Protocol over the single connection, kept strictly FIFO so each
//! per-frame read pairs with its own frame:
//!   client -> peer: the leading `CapsChanged` (config, no reply), then one
//!                   `DataFrame` per frame, then `Eos`.
//!   peer -> client: exactly one processed `DataFrame` per `DataFrame` received;
//!                   no echoed caps / segment / control.
//! `Segment` / `Flush` therefore pass through locally (they are not sent to the
//! peer), so the reply stream stays one-packet-per-frame. Per-frame timing still
//! crosses (the wire codec carries each frame's `FrameTiming`).
//!
//! Bandwidth note: this round-trips the whole frame both ways, the honest cost of
//! a generic packet-in / packet-out transform. Fine on a LAN / localhost; a
//! `metadata`-only return (retain the frame locally, receive only the attached
//! meta) is a future optimization for the pixels-unchanged case.
//!
//! `RemoteTransform<T>` holds the shared machinery; a [`PacketDuplex`] transport
//! supplies the connection. [`RemoteWsTransform`](crate::remotewstransform) and
//! [`RemoteWtTransform`](crate::remotewttransform) are type aliases over it.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError,
    HardwareError, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropValue,
    PropertySpec,
};

use crate::remoteclient::PacketClient;
use crate::remotesource::TransportFuture;

/// A [`PacketClient`] whose connection is also read back, as the remote-transform
/// round trip needs. The client-role identity (`NAME` / `PROPERTIES`) describes
/// the sink; these consts describe the same transport in the transform role.
pub trait PacketDuplex: PacketClient {
    /// `ElementMetadata` long name.
    const NAME: &'static str;
    /// `ElementMetadata` description.
    const DESCRIPTION: &'static str;
    /// The transform's runtime property specs (no `reconnect-attempts`: the
    /// round-trip protocol has no resend semantics for an in-flight frame).
    const PROPERTIES: &'static [PropertySpec];

    /// Read the peer's next packet. `Ok(None)` on a clean close.
    fn recv(&mut self) -> TransportFuture<'_, Option<PipelinePacket>>;
}

/// Distributed-graph remote transform generic over a [`PacketDuplex`]. See module
/// docs.
pub struct RemoteTransform<T: PacketDuplex> {
    transport: T,
    /// Caps recorded in `configure_pipeline`, sent to the peer (deduped against
    /// `last_sent`) as the leading `CapsChanged` so its subgraph configures.
    configured_caps: Option<Caps>,
    last_sent: Option<Caps>,
    configured: bool,
    emitted: u64,
}

impl<T: PacketDuplex + core::fmt::Debug> core::fmt::Debug for RemoteTransform<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RemoteTransform")
            .field("transport", &self.transport)
            .field("configured", &self.configured)
            .field("emitted", &self.emitted)
            .finish_non_exhaustive()
    }
}

impl<T: PacketDuplex> RemoteTransform<T> {
    /// Build a transform around `transport` (called by the per-transport `new`).
    pub(crate) fn from_transport(transport: T) -> Self {
        Self {
            transport,
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

    /// Send `caps` to the peer unless it already has them, so its subgraph
    /// configures before the first frame.
    async fn send_caps_if_new(&mut self, caps: Caps) -> Result<(), G2gError> {
        if self.last_sent.as_ref() != Some(&caps) {
            self.transport
                .send(&PipelinePacket::CapsChanged(caps.clone()))
                .await?;
            self.last_sent = Some(caps);
        }
        Ok(())
    }
}

impl<T: PacketDuplex> AsyncElement for RemoteTransform<T> {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
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
        // Every transport here has an async handshake, so the connect is deferred
        // to `process`; record the caps for the leading wire CapsChanged.
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
            // Open the connection once, on first use.
            if !self.transport.is_connected() {
                self.transport.connect().await?;
            }
            // The peer's subgraph must see the caps before the first frame.
            if let Some(caps) = self.configured_caps.clone() {
                self.send_caps_if_new(caps).await?;
            }

            match packet {
                PipelinePacket::DataFrame(frame) => {
                    self.transport
                        .send(&PipelinePacket::DataFrame(frame))
                        .await?;
                    // Exactly one processed packet comes back per frame (the peer
                    // never echoes control), so this read pairs with our frame.
                    let processed = self
                        .transport
                        .recv()
                        .await?
                        .ok_or(G2gError::Hardware(HardwareError::Other))?;
                    self.emitted += 1;
                    out.push(processed).await?;
                }
                PipelinePacket::CapsChanged(caps) => {
                    // Forward mid-stream refinement to the peer (deduped) and
                    // downstream. The dedup above already sent it if unchanged.
                    self.send_caps_if_new(caps.clone()).await?;
                    out.push(PipelinePacket::CapsChanged(caps)).await?;
                }
                PipelinePacket::Eos => {
                    // Tell the peer we are done and close; the runner's transform
                    // arm forwards EOS downstream, so we do not push it.
                    let _ = self.transport.send(&PipelinePacket::Eos).await;
                    let _ = self.transport.close().await;
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
            <T as PacketDuplex>::NAME,
            "Filter/Network",
            <T as PacketDuplex>::DESCRIPTION,
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        <T as PacketDuplex>::PROPERTIES
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        self.transport
            .set_transport_prop(name, &value)
            .unwrap_or(Err(PropError::Unknown))
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        self.transport.get_transport_prop(name)
    }
}

impl<T: PacketDuplex> PadTemplates for RemoteTransform<T> {
    /// Wildcard sink (media-agnostic); the identity source side is expressed at
    /// runtime by `caps_constraint_as_transform`, so only the sink is declared
    /// statically (a wildcard source pad is degenerate).
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink_any()])
    }
}
