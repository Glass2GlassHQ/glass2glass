//! Shared send-side core for the distributed-graph sink elements.
//!
//! [`RemoteSink`](crate::remotesink) (TCP) and [`RemoteWsSink`](crate::remotewssink)
//! (WebSocket) are the same element: a client that dials a `Remote*Src` server,
//! sends the negotiated caps as the stream's leading `CapsChanged`, then forwards
//! every subsequent `PipelinePacket` ([`g2g_core::wire`]-serialized). Both dedup
//! the caps against the last sent, and both optionally reconnect: a failed connect
//! or send is retried with a short backoff (re-sending caps on the new
//! connection), tolerating a late-starting or restarting peer. They differ only in
//! the transport primitive: how the connection is dialed, how one packet is sent,
//! and the destination knob (a host/port vs a WebSocket URL).
//!
//! `RemoteClient<T>` holds the shared machinery; a [`PacketClient`] supplies the
//! transport. `RemoteSink` / `RemoteWsSink` are type aliases over it, each adding a
//! transport-specific `new` constructor.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use tokio::time::sleep;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, PropError, PropValue, PropertySpec,
};

use crate::remotesource::TransportFuture;

/// Backoff between reconnect attempts. Short and fixed: the common case is a
/// downstream node coming up within a second or two, not a long outage.
const RECONNECT_BACKOFF: core::time::Duration = core::time::Duration::from_millis(50);

/// Transport-specific hooks for [`RemoteClient`]. The implementor owns its own
/// connection state and destination (a host/port or a URL).
pub trait PacketClient: Send + 'static {
    /// `ElementMetadata` long name.
    const NAME: &'static str;
    /// `ElementMetadata` description.
    const DESCRIPTION: &'static str;
    /// The element's transport-specific runtime property specs (`host`/`port` or
    /// `location`); the shared `reconnect-attempts` is added by [`RemoteClient`].
    const PROPERTIES: &'static [PropertySpec];

    /// Whether a live connection exists.
    fn is_connected(&self) -> bool;
    /// (Re)establish the connection. Called only when [`is_connected`](Self::is_connected)
    /// is false; on return a fresh connection exists and the caller re-sends caps.
    fn connect(&mut self) -> TransportFuture<'_, ()>;
    /// Serialize and send one packet over the live connection.
    fn send<'a>(&'a mut self, packet: &'a PipelinePacket) -> TransportFuture<'a, ()>;
    /// Drop the connection so the next [`connect`](Self::connect) reconnects.
    fn reset(&mut self);
    /// Half-close the connection after `Eos` so the far side reads the end.
    fn close(&mut self) -> TransportFuture<'_, ()>;
    /// Connect eagerly at configure time when `eager` (a synchronous dial, so a
    /// failure surfaces at configure). A transport whose handshake must be async
    /// is a no-op here and defers the connect to the first send.
    fn configure_connect(&mut self, eager: bool) -> Result<(), G2gError>;
    /// Handle the transport-specific half of `set_property`; `None` if `name` is
    /// not one of this transport's knobs (the caller's `match` falls through).
    fn set_transport_prop(
        &mut self,
        name: &str,
        value: &PropValue,
    ) -> Option<Result<(), PropError>>;
    /// Handle the transport-specific half of `get_property`.
    fn get_transport_prop(&self, name: &str) -> Option<PropValue>;
}

/// Distributed-graph sink generic over a [`PacketClient`]. See module docs.
pub struct RemoteClient<T: PacketClient> {
    transport: T,
    /// The caps the runner last handed `configure_pipeline`. Sent over the wire
    /// (deduped against `last_sent`) before the next data frame, so the far side
    /// always sees the current caps as an ordered `CapsChanged`.
    configured_caps: Option<Caps>,
    last_sent: Option<Caps>,
    configured: bool,
    sent: u64,
    /// Reconnect budget: 0 (default) = no reconnect (a failure is a hard error);
    /// N = retry the connect / a failed send up to N times with a short backoff.
    reconnect_attempts: u32,
}

impl<T: PacketClient + core::fmt::Debug> core::fmt::Debug for RemoteClient<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RemoteClient")
            .field("transport", &self.transport)
            .field("configured", &self.configured)
            .field("sent", &self.sent)
            .field("reconnect_attempts", &self.reconnect_attempts)
            .finish_non_exhaustive()
    }
}

impl<T: PacketClient> RemoteClient<T> {
    /// Build a sink around `transport` (called by the per-transport `new`).
    pub(crate) fn from_transport(transport: T) -> Self {
        Self {
            transport,
            configured_caps: None,
            last_sent: None,
            configured: false,
            sent: 0,
            reconnect_attempts: 0,
        }
    }

    /// Tolerate a late-starting or restarting peer: retry the connect and any
    /// failed send up to `attempts` times (with a short backoff), re-sending the
    /// current caps on each new connection. `0` restores the default (a failure
    /// ends the pipeline).
    pub fn with_reconnect(mut self, attempts: u32) -> Self {
        self.reconnect_attempts = attempts;
        self
    }

    /// Count of wire packets sent (caps + data + control). Useful in tests.
    pub fn sent(&self) -> u64 {
        self.sent
    }

    /// Ensure the connection is up and the current caps are sent, then send
    /// `packet` (when `Some`). On any failure, if reconnect is enabled and the
    /// budget is not spent, drop the connection, back off, and retry the whole
    /// cycle (reconnect + caps + packet); otherwise return the error.
    async fn deliver(&mut self, packet: Option<&PipelinePacket>) -> Result<(), G2gError> {
        let mut tries = 0u32;
        loop {
            let attempt = async {
                if !self.transport.is_connected() {
                    self.transport.connect().await?;
                    // A new connection needs the caps re-sent as its first packet.
                    self.last_sent = None;
                }
                if let Some(caps) = self.configured_caps.clone() {
                    if self.last_sent.as_ref() != Some(&caps) {
                        self.transport
                            .send(&PipelinePacket::CapsChanged(caps.clone()))
                            .await?;
                        self.sent += 1;
                        self.last_sent = Some(caps);
                    }
                }
                if let Some(p) = packet {
                    self.transport.send(p).await?;
                    self.sent += 1;
                }
                Ok::<(), G2gError>(())
            }
            .await;
            match attempt {
                Ok(()) => return Ok(()),
                Err(e) => {
                    self.transport.reset();
                    self.last_sent = None;
                    if tries >= self.reconnect_attempts {
                        return Err(e);
                    }
                    tries += 1;
                    sleep(RECONNECT_BACKOFF).await;
                }
            }
        }
    }
}

impl<T: PacketClient> AsyncElement for RemoteClient<T> {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // A generic transport: whatever upstream produces is what we ship.
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // Without reconnect, connect eagerly so a connect failure surfaces at
        // configure time; with reconnect, defer so it can be retried against a
        // late-starting peer. (A transport with an async handshake always defers.)
        self.transport
            .configure_connect(self.reconnect_attempts == 0)?;
        // The caps send is deferred to `process`; record the current caps so the
        // next send emits them if they changed.
        self.configured_caps = Some(absolute_caps.clone());
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            // `deliver` (re)connects on first use, emits the current caps as an
            // ordered leading `CapsChanged` (deduped against `last_sent`), then
            // sends the packet, retrying the whole cycle on failure when reconnect
            // is enabled.
            match packet {
                PipelinePacket::CapsChanged(caps) => {
                    // Mid-stream refinement: adopt it as the current caps and let
                    // `deliver` send it (deduped). No trailing data packet.
                    self.configured_caps = Some(caps);
                    self.deliver(None).await?;
                }
                PipelinePacket::Eos => {
                    self.deliver(Some(&PipelinePacket::Eos)).await?;
                    // Half-close so the far side reads the end after draining Eos.
                    self.transport.close().await?;
                }
                other => self.deliver(Some(&other)).await?,
            }
            Ok(())
        })
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(T::NAME, "Sink/Network", T::DESCRIPTION, "g2g")
    }

    fn properties(&self) -> &'static [PropertySpec] {
        T::PROPERTIES
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if name == "reconnect-attempts" {
            let n = value.as_uint().ok_or(PropError::Type)?;
            self.reconnect_attempts = n.min(u32::MAX as u64) as u32;
            return Ok(());
        }
        self.transport
            .set_transport_prop(name, &value)
            .unwrap_or(Err(PropError::Unknown))
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if name == "reconnect-attempts" {
            return Some(PropValue::Uint(self.reconnect_attempts as u64));
        }
        self.transport.get_transport_prop(name)
    }
}

impl<T: PacketClient> PadTemplates for RemoteClient<T> {
    /// Wildcard sink: accepts any caps (a media-agnostic transport), matching
    /// `caps_constraint_as_sink` of `AcceptsAny`.
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink_any()])
    }
}
