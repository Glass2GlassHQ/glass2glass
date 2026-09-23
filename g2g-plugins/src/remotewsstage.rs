//! The peer side of a remote transform: host a subgraph over one WebSocket.
//!
//! [`RemoteWsTransform`](crate::remotewstransform) offloads a middle stage and
//! expects one processed frame per frame it sends. That peer had to be written
//! by hand. [`serve_ws_stage`] is the other half: it accepts one client, runs
//! every arriving frame through the [`Bin`] it was given, and sends each result
//! back down the same connection, so a whole subgraph (tees, side branches,
//! muxers that rejoin them) becomes the offloaded stage.
//!
//! The connection is split once: the read half feeds [`WireStageSrc`], a source
//! that discovers the caps from the leading wire message and emits every packet
//! after it, and the write half backs [`WireStageSink`], which returns the
//! processed frames. The source links to the bin's one ghost input and the sink
//! to its one ghost output, and `run_graph` drives the flattened whole. Whole
//! packets cross both ways, so timing, sequence and metadata survive the hop.
//! The runner's links carry the backpressure.
//!
//! The reply stream is frames only. The protocol pairs each reply with the frame
//! that caused it, so echoing the caps or a segment would desynchronise the
//! client's per-frame read. With `meta_only` the reply carries the metadata and
//! an empty payload, the mode [`crate::metaonly`] describes, for a subgraph that
//! leaves the pixels alone.
//!
//! For the pairing to hold, the bin must deliver exactly one frame to its ghost
//! output per frame on its ghost input, in order, carrying that frame's
//! `sequence`. Branches that end in their own sinks are free. The sink checks
//! each reply's sequence against the frames still owed one and fails the run on
//! an extra, missing or reordered reply. The client waits for each reply before
//! sending the next frame, so a bin that drops a frame, or holds one until a
//! later frame arrives, stalls both ends instead.

use core::future::Future;
use core::pin::{pin, Pin};

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::sync::Arc;
use alloc::vec::Vec;
use std::sync::Mutex;

use futures_util::future::{select, Either};
use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio::sync::Notify;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{accept_async, MaybeTlsStream, WebSocketStream};

use g2g_core::frame::Frame;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{run_graph, GraphNodeRef, LinkCapacity, RunStats, SourceLoop};
use g2g_core::wire::{decode_packet, encode_packet};
use g2g_core::{
    AsyncElement, Bin, Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError, Graph,
    HardwareError, OutputSink, PipelineClock, PipelinePacket,
};

use crate::filesink::io_err;
use crate::remotewire::map_wire;
use crate::remotewsio::{bind_addr_of, ws_err};

type ServeSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;
type ServeWrite = SplitSink<ServeSocket, Message>;
type ServeRead = SplitStream<ServeSocket>;

/// A reply that does not pair with the oldest frame still owed one.
const REPLY_DESYNC: G2gError = G2gError::Hardware(HardwareError::Other);

/// The frames read off the wire that have no reply yet, recorded by the source
/// and answered by the sink.
#[derive(Debug, Default)]
struct ReplyLedger {
    /// Sequence numbers still owed a reply, oldest first.
    owed: Mutex<VecDeque<u64>>,
    desynced: Notify,
}

impl ReplyLedger {
    fn record(&self, sequence: u64) {
        self.owed.lock().expect("reply ledger").push_back(sequence);
    }

    /// Pair a reply carrying `sequence` with the oldest frame owed one.
    fn answer(&self, sequence: u64) -> Result<(), G2gError> {
        if self.owed.lock().expect("reply ledger").pop_front() == Some(sequence) {
            return Ok(());
        }
        self.fail()
    }

    /// Check at end of stream that every frame got its reply.
    fn settle(&self) -> Result<(), G2gError> {
        if self.owed.lock().expect("reply ledger").is_empty() {
            return Ok(());
        }
        self.fail()
    }

    fn fail(&self) -> Result<(), G2gError> {
        self.desynced.notify_one();
        Err(REPLY_DESYNC)
    }
}

/// Reads the client's packets off the connection's read half. Discovers the
/// caps from the leading wire message, the way
/// [`RemoteWsSrc`](crate::remotewssrc) does.
#[derive(Debug)]
pub struct WireStageSrc {
    read: ServeRead,
    discovered: Option<Caps>,
    configured: bool,
    ledger: Arc<ReplyLedger>,
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
                if let PipelinePacket::DataFrame(frame) = &packet {
                    self.ledger.record(frame.sequence);
                    frames += 1;
                }
                out.push(packet).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(frames)
        })
    }
}

/// Sends each processed frame back down the connection's write half. Frames
/// only: a reply pairs with the frame that caused it, so control packets stay
/// local. A frame whose sequence is not the oldest one owed a reply fails the
/// run, as does an end of stream with replies still owed.
#[derive(Debug)]
pub struct WireStageSink {
    write: ServeWrite,
    /// Reply with the metadata and an empty payload (see [`crate::metaonly`]).
    meta_only: bool,
    ledger: Arc<ReplyLedger>,
}

impl WireStageSink {
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
                    self.ledger.answer(frame.sequence)?;
                    let reply = PipelinePacket::DataFrame(self.reply_of(frame));
                    let body = encode_packet(&reply).map_err(map_wire)?;
                    self.write
                        .send(Message::Binary(body))
                        .await
                        .map_err(ws_err)?;
                }
                PipelinePacket::Eos => {
                    self.ledger.settle()?;
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

/// Accept one client on `bind` and run `stage` as the stage it offloads,
/// returning when the client's stream ends. `stage` exposes exactly one ghost
/// input, fed the client's frames, and one ghost output, whose frames are the
/// replies. `bind` is the address to listen on (`ws://host:port` or a bare
/// `host:port`). `meta_only` returns each frame's metadata alone, for a stage
/// that leaves the pixels alone.
pub async fn serve_ws_stage(
    bind: &str,
    stage: Bin<GraphNodeRef<'_>>,
    clock: &impl PipelineClock,
    link_capacity: impl Into<LinkCapacity>,
    meta_only: bool,
) -> Result<RunStats, G2gError> {
    let address = bind_addr_of(bind)?;
    let listener = std::net::TcpListener::bind(address).map_err(io_err)?;
    serve_ws_stage_on(listener, stage, clock, link_capacity, meta_only).await
}

/// As [`serve_ws_stage`], on a listener that is already bound. A client whose
/// connect does not retry needs the port listening before it dials, and only
/// the caller knows when that is.
pub async fn serve_ws_stage_on(
    listener: std::net::TcpListener,
    stage: Bin<GraphNodeRef<'_>>,
    clock: &impl PipelineClock,
    link_capacity: impl Into<LinkCapacity>,
    meta_only: bool,
) -> Result<RunStats, G2gError> {
    let mut graph = Graph::new();
    let stage = graph.add_bin(stage);
    if stage.input_count() != 1 || stage.output_count() != 1 {
        return Err(G2gError::CapsMismatch);
    }
    listener.set_nonblocking(true).map_err(io_err)?;
    let listener = tokio::net::TcpListener::from_std(listener).map_err(io_err)?;
    let (tcp, _peer) = listener.accept().await.map_err(io_err)?;
    let socket = accept_async(MaybeTlsStream::Plain(tcp))
        .await
        .map_err(ws_err)?;
    let (write, read) = socket.split();
    let ledger = Arc::new(ReplyLedger::default());
    let source = graph.add_source(GraphNodeRef::source(WireStageSrc {
        read,
        discovered: None,
        configured: false,
        ledger: ledger.clone(),
    }));
    let sink = graph.add_sink(GraphNodeRef::element(WireStageSink {
        write,
        meta_only,
        ledger: ledger.clone(),
    }));
    graph
        .link(source, stage.input(0))
        .map_err(|_| G2gError::CapsMismatch)?;
    graph
        .link(stage.output(0), sink)
        .map_err(|_| G2gError::CapsMismatch)?;
    let run = pin!(run_graph(graph, clock, link_capacity));
    // The source keeps reading a client that waits on the reply that failed.
    let desynced = pin!(ledger.desynced.notified());
    match select(run, desynced).await {
        Either::Left((result, _)) => result,
        Either::Right(_) => Err(REPLY_DESYNC),
    }
}
