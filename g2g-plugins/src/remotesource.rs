//! Shared receive-side core for the distributed-graph source elements.
//!
//! [`RemoteSrc`](crate::remotesrc) (TCP) and [`RemoteWsSrc`](crate::remotewssrc)
//! (WebSocket) are the same element: a server that listens, accepts one client,
//! discovers the media type from the stream's leading `CapsChanged`
//! ([`g2g_core::wire`]), then re-emits that packet and every subsequent one until
//! the sender's `Eos` (or a clean close). With `keep_listening` a client that
//! drops without `Eos` is replaced instead of ending the stream. They differ only
//! in the transport primitive: how a client is accepted (plus any handshake) and
//! how one packet is read.
//!
//! `RemoteSource<T>` holds the shared machinery; a [`PacketTransport`] supplies the
//! transport-specific `accept` / `recv` plus the element's identity. `RemoteSrc` /
//! `RemoteWsSrc` are type aliases over it.

use core::future::Future;
use core::marker::PhantomData;
use core::pin::Pin;

use alloc::boxed::Box;

use std::net::{SocketAddr, TcpListener as StdTcpListener};

use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
    PipelinePacket, PropError, PropValue, PropertySpec,
};

use crate::filesink::io_err;

/// A future returned by a [`PacketTransport`] method, borrowing its argument for
/// the future's lifetime (the codebase's dyn-safe boxed-future idiom).
pub type TransportFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, G2gError>> + 'a>>;

/// Transport-specific hooks for [`RemoteSource`]. Implemented by zero-sized
/// markers (`TcpTransport`, `WsTransport`).
pub trait PacketTransport: Send + Sync + 'static {
    /// The live client connection this transport accepts and reads from.
    type Conn: Send;
    /// `ElementMetadata` long name.
    const NAME: &'static str;
    /// `ElementMetadata` description.
    const DESCRIPTION: &'static str;
    /// The element's runtime property specs (`address` / `port` / `keep-listening`;
    /// the `port` help text names the transport, so it varies).
    const PROPERTIES: &'static [PropertySpec];

    /// Accept one client on `listener`, perform any handshake, read the leading
    /// `CapsChanged`, and return the live connection plus the discovered caps.
    fn accept(listener: &tokio::net::TcpListener) -> TransportFuture<'_, (Self::Conn, Caps)>;
    /// Read the next decoded packet, or `Ok(None)` on a clean close at a message
    /// boundary (the stream's natural end).
    fn recv(conn: &mut Self::Conn) -> TransportFuture<'_, Option<PipelinePacket>>;
}

/// Distributed-graph source generic over a [`PacketTransport`]. See module docs.
pub struct RemoteSource<T: PacketTransport> {
    bind: SocketAddr,
    /// A pre-bound listener (so a test can pick an ephemeral port before the
    /// sender connects); otherwise caps discovery binds `bind`.
    std_listener: Option<StdTcpListener>,
    /// The tokio listener, kept alive after the first accept so a dropped client
    /// can be replaced (`keep_listening`).
    listener: Option<tokio::net::TcpListener>,
    /// The accepted client connection, established during caps discovery.
    conn: Option<T::Conn>,
    /// Caps read from the first wire packet; memoized so a repeated
    /// `intercept_caps` / `caps_constraint` does not accept a second connection.
    discovered: Option<Caps>,
    configured: bool,
    frame_limit: u64,
    /// When set, a client that drops without a clean `Eos` is not the end of the
    /// stream: the source waits for a replacement client (which re-sends its
    /// leading caps) and continues. Only an explicit `Eos` (or `frame_limit`) ends
    /// it.
    keep_listening: bool,
    _transport: PhantomData<T>,
}

impl<T: PacketTransport> core::fmt::Debug for RemoteSource<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RemoteSource")
            .field("name", &T::NAME)
            .field("bind", &self.bind)
            .field("configured", &self.configured)
            .field("keep_listening", &self.keep_listening)
            .finish_non_exhaustive()
    }
}

impl<T: PacketTransport> RemoteSource<T> {
    /// Listen for a sender on `bind` (e.g. `0.0.0.0:9600`).
    pub fn new(bind: SocketAddr) -> Self {
        Self {
            bind,
            std_listener: None,
            listener: None,
            conn: None,
            discovered: None,
            configured: false,
            frame_limit: 0,
            keep_listening: false,
            _transport: PhantomData,
        }
    }

    /// Tolerate a sender that drops without a clean `Eos`: keep the listener open
    /// and accept a replacement client (which re-sends its leading caps) instead
    /// of ending the stream. Pairs with the sink's `with_reconnect`.
    pub fn with_reconnect(mut self) -> Self {
        self.keep_listening = true;
        self
    }

    /// Use an already-bound listener (so a test can bind port 0 and read the
    /// actual port before the sender connects).
    pub fn from_listener(listener: StdTcpListener) -> Result<Self, G2gError> {
        let bind = listener.local_addr().map_err(io_err)?;
        Ok(Self {
            std_listener: Some(listener),
            ..Self::new(bind)
        })
    }

    /// Stop after `n` data frames and emit EOS (the bounded / test path).
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    /// The bound port, once a listener exists.
    pub fn local_port(&self) -> Option<u16> {
        self.std_listener
            .as_ref()
            .and_then(|l| l.local_addr().ok())
            .map(|a| a.port())
    }

    /// Bind / reuse the listener, accept the sender, and read its first packet
    /// (the caps). Idempotent: once discovered, returns the cached caps without
    /// accepting again.
    async fn ensure_connected(&mut self) -> Result<Caps, G2gError> {
        if let Some(caps) = &self.discovered {
            return Ok(caps.clone());
        }
        let listener = match self.std_listener.take() {
            Some(l) => {
                l.set_nonblocking(true).map_err(io_err)?;
                tokio::net::TcpListener::from_std(l).map_err(io_err)?
            }
            None => tokio::net::TcpListener::bind(self.bind)
                .await
                .map_err(io_err)?,
        };
        let (conn, caps) = T::accept(&listener).await?;
        self.conn = Some(conn);
        // Keep the listener so a dropped client can be replaced (keep_listening).
        self.listener = Some(listener);
        self.discovered = Some(caps.clone());
        Ok(caps)
    }
}

impl<T: PacketTransport> SourceLoop for RemoteSource<T> {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = Pin<Box<dyn Future<Output = Result<Caps, G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        Box::pin(async move { self.ensure_connected().await })
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        Box::pin(async move {
            let caps = self.ensure_connected().await?;
            Ok(CapsConstraint::Produces(CapsSet::one(caps)))
        })
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // The connection is already established (caps discovery); just arm.
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(T::NAME, "Source/Network", T::DESCRIPTION, "g2g")
    }

    fn properties(&self) -> &'static [PropertySpec] {
        T::PROPERTIES
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if let Some(r) = crate::netprop::set_addr_prop(&mut self.bind, "address", name, &value) {
            return r;
        }
        match name {
            "keep-listening" => {
                self.keep_listening = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if let Some(v) = crate::netprop::get_addr_prop(&self.bind, "address", name) {
            return Some(v);
        }
        match name {
            "keep-listening" => Some(PropValue::Bool(self.keep_listening)),
            _ => None,
        }
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            // Re-emit the discovered caps as the leading ordered CapsChanged.
            let caps = self.discovered.clone().ok_or(G2gError::NotConfigured)?;
            out.push(PipelinePacket::CapsChanged(caps)).await?;

            let limit = self.frame_limit;
            let mut emitted = 0u64;
            loop {
                let conn = self.conn.as_mut().ok_or(G2gError::NotConfigured)?;
                let pkt = match T::recv(conn).await? {
                    Some(p) => p,
                    // Sender closed the connection without a clean Eos.
                    None => {
                        if self.keep_listening {
                            // Wait for a replacement client and continue. It
                            // re-sends its leading caps, which we forward so a
                            // downstream re-negotiates if they changed.
                            let listener = self.listener.take().ok_or(G2gError::NotConfigured)?;
                            let (conn, caps) = T::accept(&listener).await?;
                            self.conn = Some(conn);
                            self.listener = Some(listener);
                            out.push(PipelinePacket::CapsChanged(caps)).await?;
                            continue;
                        }
                        out.push(PipelinePacket::Eos).await?;
                        break;
                    }
                };
                match pkt {
                    PipelinePacket::Eos => {
                        out.push(PipelinePacket::Eos).await?;
                        break;
                    }
                    PipelinePacket::DataFrame(frame) => {
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                        emitted += 1;
                        if limit != 0 && emitted >= limit {
                            out.push(PipelinePacket::Eos).await?;
                            break;
                        }
                    }
                    // CapsChanged / Segment / Flush forwarded unchanged.
                    other => {
                        out.push(other).await?;
                    }
                }
            }
            Ok(emitted)
        })
    }
}
