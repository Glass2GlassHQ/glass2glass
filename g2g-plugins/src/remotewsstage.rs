//! The peer side of a remote transform: host a subgraph over one WebSocket.
//!
//! [`RemoteWsTransform`](crate::remotewstransform) offloads a middle stage and
//! expects one processed frame per frame it sends. That peer had to be written
//! by hand. [`serve_ws_stage`] is the other half: it accepts one client, runs
//! every arriving frame through the chain of transforms it was given, and sends
//! each result back down the same connection, so a whole subgraph (a `Bin`'s
//! interior, flattened into its stages) becomes the offloaded stage.
//!
//! The connection is split once: the read half feeds [`WireStageSrc`], a source
//! that discovers the caps from the leading wire message and emits every packet
//! after it, and the write half backs [`WireStageSink`], which returns the
//! processed frames. Whole packets cross both ways, so timing, sequence and
//! metadata survive the hop; the runner's links carry the backpressure.
//!
//! The reply stream is frames only. The protocol pairs each reply with the frame
//! that caused it, so echoing the caps or a segment would desynchronise the
//! client's per-frame read. With `meta_only` the reply carries the metadata and
//! an empty payload, the mode [`crate::metaonly`] describes, for a chain that
//! leaves the pixels alone.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{accept_async, MaybeTlsStream, WebSocketStream};

use g2g_core::element::DynAsyncElement;
use g2g_core::frame::Frame;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{run_linear_chain, LinkCapacity, RunStats, SourceLoop};
use g2g_core::wire::{decode_packet, encode_packet};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError, HardwareError,
    OutputSink, PipelineClock, PipelinePacket,
};

use crate::filesink::io_err;
use crate::remotewire::map_wire;
use crate::remotewsio::{bind_addr_of, ws_err};

type ServeSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;
type ServeWrite = SplitSink<ServeSocket, Message>;
type ServeRead = SplitStream<ServeSocket>;

/// Reads the client's packets off the connection's read half. Discovers the
/// caps from the leading wire message, the way
/// [`RemoteWsSrc`](crate::remotewssrc) does.
#[derive(Debug)]
pub struct WireStageSrc {
    read: ServeRead,
    discovered: Option<Caps>,
    configured: bool,
}

impl WireStageSrc {
    /// The next wire packet, `None` once the client closes.
    async fn next_packet(&mut self) -> Result<Option<PipelinePacket>, G2gError> {
        loop {
            match self.read.next().await {
                Some(Ok(Message::Binary(bytes))) => {
                    return Ok(Some(decode_packet(&bytes).map_err(map_wire)?))
                }
                Some(Ok(Message::Close(_))) | None => return Ok(None),
                Some(Ok(_)) => continue,
                Some(Err(e)) => return Err(ws_err(e)),
            }
        }
    }
}

impl SourceLoop for WireStageSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = Pin<Box<dyn Future<Output = Result<Caps, G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        Box::pin(async move {
            if let Some(caps) = self.discovered.clone() {
                return Ok(caps);
            }
            let caps = match self.next_packet().await? {
                Some(PipelinePacket::CapsChanged(caps)) => caps,
                // The client sends its caps first; anything else breaks the
                // protocol before a frame can be paired with a reply.
                _ => return Err(G2gError::Hardware(HardwareError::Other)),
            };
            self.discovered = Some(caps.clone());
            Ok(caps)
        })
    }

    async fn caps_constraint(&mut self) -> Result<CapsConstraint<'_>, G2gError> {
        let caps = self.intercept_caps().await?;
        Ok(CapsConstraint::Produces(CapsSet::one(caps)))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            if let Some(caps) = self.discovered.clone() {
                out.push(PipelinePacket::CapsChanged(caps)).await?;
            }
            let mut frames = 0u64;
            loop {
                let packet = match self.next_packet().await? {
                    Some(packet) => packet,
                    None => break,
                };
                if matches!(packet, PipelinePacket::Eos) {
                    break;
                }
                let is_frame = matches!(packet, PipelinePacket::DataFrame(_));
                out.push(packet).await?;
                if is_frame {
                    frames += 1;
                }
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(frames)
        })
    }
}

/// Sends each processed frame back down the connection's write half. Frames
/// only: a reply pairs with the frame that caused it, so control packets stay
/// local.
#[derive(Debug)]
pub struct WireStageSink {
    write: ServeWrite,
    /// Reply with the metadata and an empty payload (see [`crate::metaonly`]).
    meta_only: bool,
    sent: u64,
}

impl WireStageSink {
    /// Count of replies sent.
    pub fn sent(&self) -> u64 {
        self.sent
    }

    /// The frame to send back: the processed one, or its metadata alone.
    fn reply_of(&self, frame: Frame) -> Frame {
        if !self.meta_only {
            return frame;
        }
        let mut reply = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(Vec::new().into_boxed_slice())),
            frame.timing,
            frame.sequence,
        );
        reply.meta = frame.meta;
        reply
    }
}

impl AsyncElement for WireStageSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let reply = PipelinePacket::DataFrame(self.reply_of(frame));
                    let body = encode_packet(&reply).map_err(map_wire)?;
                    self.write
                        .send(Message::Binary(body))
                        .await
                        .map_err(ws_err)?;
                    self.sent += 1;
                }
                PipelinePacket::Eos => {
                    let _ = self.write.close().await;
                }
                // Control stays local: an echoed caps / segment would
                // desynchronise the client's per-frame read.
                _ => {}
            }
            Ok(())
        })
    }
}

/// Accept one client on `bind` and run `transforms` as the stage it offloads,
/// returning when the client's stream ends. `bind` is the address to listen on
/// (`ws://host:port` or a bare `host:port`); `meta_only` returns each frame's
/// metadata alone, for a chain that leaves the pixels alone.
pub async fn serve_ws_stage(
    bind: &str,
    transforms: Vec<&mut dyn DynAsyncElement>,
    clock: &impl PipelineClock,
    link_capacity: impl Into<LinkCapacity>,
    meta_only: bool,
) -> Result<RunStats, G2gError> {
    let address = bind_addr_of(bind)?;
    let listener = std::net::TcpListener::bind(address).map_err(io_err)?;
    serve_ws_stage_on(listener, transforms, clock, link_capacity, meta_only).await
}

/// As [`serve_ws_stage`], on a listener that is already bound. A client whose
/// connect does not retry needs the port listening before it dials, and only
/// the caller knows when that is.
pub async fn serve_ws_stage_on(
    listener: std::net::TcpListener,
    transforms: Vec<&mut dyn DynAsyncElement>,
    clock: &impl PipelineClock,
    link_capacity: impl Into<LinkCapacity>,
    meta_only: bool,
) -> Result<RunStats, G2gError> {
    listener.set_nonblocking(true).map_err(io_err)?;
    let listener = tokio::net::TcpListener::from_std(listener).map_err(io_err)?;
    let (tcp, _peer) = listener.accept().await.map_err(io_err)?;
    let socket = accept_async(MaybeTlsStream::Plain(tcp))
        .await
        .map_err(ws_err)?;
    let (write, read) = socket.split();
    let mut source = WireStageSrc {
        read,
        discovered: None,
        configured: false,
    };
    let mut sink = WireStageSink {
        write,
        meta_only,
        sent: 0,
    };
    run_linear_chain(&mut source, transforms, &mut sink, clock, link_capacity).await
}
