// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Native WebTransport and raw QUIC <-> wire-codec I/O shared by the `webtransport` elements
//! (`RemoteWtSink`, `RemoteWtSrc`, `RemoteWtTransform`): dial or accept a session,
//! and carry the [`PipelinePacket`] stream over the single bidirectional stream it
//! opens.
//!
//! A WebTransport stream is a QUIC stream, so it is a *byte* stream with no
//! message boundaries: the framing is the `remote` TCP pair's `u32` length prefix
//! (shared with it in [`crate::remotewire`]), not the WebSocket pair's
//! one-message-per-packet.
//!
//! The other video modes ride the same session. `stream-per-frame` puts every
//! packet after the leading caps on its own unidirectional stream, and a
//! keyframe resets the frame streams not yet acknowledged. `datagrams` splits each
//! data frame into Reed-Solomon shards, one QUIC datagram each, plus repair shards
//! sized to the path's loss, and sends anything else as a unidirectional stream.
//! Each packet carries a [`PacketOrder`], from which the receiver restores send order and skips frames that will not arrive. Receiving
//! is mode-free, [`QuicStream::recv`] takes packets from whichever the peer used.
//! The receiver writes keyframe requests on its half of the bidirectional stream.
//!
//! A data channel (`open_data_channel` on the sink and source) is a second
//! bidirectional stream carrying length-framed byte messages, opened by the
//! source and accepted by the sink.

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::future::Future;
use core::pin::pin;
use core::task::{Context, Poll, Waker};
use core::time::Duration;

use bytes::Bytes;
use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;
use tokio::task::{JoinError, JoinSet};
use tokio::time::Instant;
use web_transport_quinn::proto::ConnectRequest;
use web_transport_quinn::quinn::rustls::pki_types::pem::PemObject;
use web_transport_quinn::quinn::rustls::pki_types::{CertificateDer, PrivateKeyDer};

use web_transport_quinn::quinn::crypto::rustls::{
    HandshakeData, QuicClientConfig, QuicServerConfig,
};
use web_transport_quinn::quinn::rustls;
use web_transport_quinn::quinn::rustls::client::danger::{
    HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier,
};
use web_transport_quinn::quinn::rustls::pki_types::{ServerName, UnixTime};
use web_transport_quinn::quinn::{self, Connection, Endpoint};
use web_transport_quinn::{
    crypto, ClientBuilder, CongestionControl, ReadToEndError, RecvStream, Request, SendStream,
    Session, SessionError,
};

use g2g_core::caps::framerate_q16_period_ns;
use g2g_core::g2g_warn;
use g2g_core::log::Target;
use g2g_core::query::LatencyReport;
use g2g_core::{
    Caps, G2gError, NetworkFailure, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
    Rate,
};

use crate::abortondrop::AbortOnDrop;
use crate::datachannel::{DataChannelTransportEnds, MAX_MESSAGE_SIZE};
use crate::remotewire::{send_frame_bytes, send_framed, FramedRead};
use crate::remotewtfec::{shard_frame, PathCounts, PathLoss, ShardAssembly};
use crate::remotewtvideo::{NextPacket, PacketOrder, PacketReorder, VideoMode, KEYFRAME_REQUEST};
use crate::remotewtws::{TlsWebSocket, WsLink, WEBSOCKET_SCHEME};
use crate::textencoding::{hex_decode, hex_encode};

/// Any WebTransport / QUIC / TLS failure ends the session.
pub(crate) fn wt_err<E>(_: E) -> G2gError {
    G2gError::Network(NetworkFailure::Stream)
}

/// How long a half-close waits for the peer to acknowledge what was already
/// written. Closing a QUIC connection discards anything unacknowledged, so
/// without this the trailing `Eos` can be lost; bounded so a peer that stopped
/// reading cannot wedge the pipeline.
const FLUSH_TIMEOUT: Duration = Duration::from_millis(500);

#[derive(Debug)]
pub enum WtStream {
    Quic(Box<QuicStream>),
    WebSocket(WsLink),
}

impl WtStream {
    pub(crate) async fn send(&mut self, packet: &PipelinePacket) -> Result<(), G2gError> {
        match self {
            WtStream::Quic(stream) => stream.send(packet).await,
            WtStream::WebSocket(link) => link.send(packet).await,
        }
    }

    pub(crate) fn take_keyframe_request(&mut self) -> bool {
        match self {
            WtStream::Quic(stream) => stream.take_keyframe_request(),
            WtStream::WebSocket(link) => link.take_keyframe_request(),
        }
    }

    pub(crate) async fn request_keyframe(&mut self) -> Result<(), G2gError> {
        match self {
            WtStream::Quic(stream) => stream.request_keyframe().await,
            WtStream::WebSocket(link) => link.request_keyframe(),
        }
    }

    pub(crate) fn latency(&self) -> LatencyReport {
        match self {
            WtStream::Quic(stream) => stream.latency(),
            WtStream::WebSocket(_) => LatencyReport::live(0, None),
        }
    }

    pub(crate) async fn recv(&mut self) -> Result<Option<PipelinePacket>, G2gError> {
        match self {
            WtStream::Quic(stream) => stream.recv().await,
            WtStream::WebSocket(link) => link.recv().await,
        }
    }

    pub(crate) async fn finish(&mut self) {
        match self {
            WtStream::Quic(stream) => stream.finish().await,
            WtStream::WebSocket(link) => link.finish(FLUSH_TIMEOUT).await,
        }
    }
}

/// A WebTransport session plus the one bidirectional stream carrying the wire
/// codec. The session is held alongside the stream because dropping it closes the
/// QUIC connection underneath, and because the datagram and stream-per-frame
/// carriers send and read on it.
#[derive(Debug)]
pub struct QuicStream {
    session: Session,
    tx: SendStream,
    rx: RecvStream,
    /// Stream-side read state, kept across calls so [`Self::recv`] survives being
    /// cancelled by the other branches of its `select!`.
    reader: FramedRead,
    /// Whether the first packet, the negotiated caps, is still outstanding. It
    /// always travels on the bidirectional stream.
    leading: bool,
    sending: SendOrdering,
    receiving: ReceiveOrdering,
    _data_channel: Option<AbortOnDrop>,
}

#[derive(Debug)]
struct SendOrdering {
    mode: VideoMode,
    stall_timeout: Option<Duration>,
    frames_sent: u64,
    controls_sent: u64,
    unacknowledged: VecDeque<UnacknowledgedStream>,
    stall_keyframe_requested: bool,
    path_loss: PathLoss,
}

#[derive(Debug)]
struct UnacknowledgedStream {
    stream: SendStream,
    // None for a control packet, which is never reset
    frame_sent_at: Option<Instant>,
}

const FRAME_SUPERSEDED: u32 = 0;

impl SendOrdering {
    fn forget_acknowledged(&mut self) {
        let mut context = Context::from_waker(Waker::noop());
        self.unacknowledged.retain(|entry| {
            let stopped = pin!(entry.stream.stopped());
            stopped.poll(&mut context).is_pending()
        });
    }

    fn reset_unfinished_frames(&mut self) {
        self.stall_keyframe_requested = false;
        self.unacknowledged.retain_mut(|entry| {
            if entry.frame_sent_at.is_none() {
                return true;
            }
            // quinn refuses a reset once every byte is acknowledged, and the frame then arrived anyway
            let _ = entry.stream.reset(FRAME_SUPERSEDED);
            false
        });
    }

    fn stalled(&mut self) -> bool {
        let Some(timeout) = self.stall_timeout else {
            return false;
        };
        if self.stall_keyframe_requested {
            return false;
        }
        let oldest = self
            .unacknowledged
            .iter()
            .find_map(|entry| entry.frame_sent_at);
        self.stall_keyframe_requested = oldest.is_some_and(|sent| sent.elapsed() >= timeout);
        self.stall_keyframe_requested
    }
}

#[derive(Debug)]
struct ReceiveOrdering {
    frame_timeout: Option<Duration>,
    reorder: PacketReorder,
    shards: ShardAssembly,
    reads: JoinSet<Result<Vec<u8>, ReadToEndError>>,
    ordered: bool,
    datagrams_seen: bool,
    stream_done: bool,
    datagrams_closed: bool,
    uni_closed: bool,
}

#[derive(Debug)]
enum ReceiveEvent {
    Datagram(Result<Bytes, SessionError>),
    UniStream(Result<RecvStream, SessionError>),
    UniRead(Result<Result<Vec<u8>, ReadToEndError>, JoinError>),
    Stream(Result<Option<PipelinePacket>, G2gError>),
    Deadline,
}

impl ReceiveOrdering {
    fn insert(&mut self, bytes: &[u8]) -> Result<(), G2gError> {
        let (order, packet) = PacketOrder::decode(bytes)?;
        self.reorder.insert(order, packet)
    }

    fn insert_read(
        &mut self,
        read: Result<Result<Vec<u8>, ReadToEndError>, JoinError>,
    ) -> Result<(), G2gError> {
        match read.map_err(wt_err)? {
            Ok(bytes) => self.insert(&bytes),
            // senders reset a frame stream that a newer keyframe replaces
            Err(_) => Ok(()),
        }
    }

    fn missing_frame_timeout(&self) -> Result<Option<Duration>, G2gError> {
        if !self.datagrams_seen {
            return Ok(None);
        }
        if self.frame_timeout.is_none() {
            g2g_warn!(
                Target::category("remotewt"),
                "datagram video needs frame-timeout when the caps have no fixed framerate"
            );
        }
        self.frame_timeout.map(Some).ok_or(G2gError::NotConfigured)
    }

    fn finished(&self) -> bool {
        self.stream_done && self.datagrams_closed && self.uni_closed && self.reads.is_empty()
    }
}

impl QuicStream {
    pub(crate) fn new(
        session: Session,
        tx: SendStream,
        rx: RecvStream,
        settings: VideoSettings,
        data_channel: Option<AbortOnDrop>,
    ) -> Self {
        Self {
            session,
            tx,
            rx,
            reader: FramedRead::default(),
            leading: true,
            sending: SendOrdering {
                mode: settings.mode,
                stall_timeout: settings.stall_timeout,
                frames_sent: 0,
                controls_sent: 0,
                unacknowledged: VecDeque::new(),
                stall_keyframe_requested: false,
                path_loss: PathLoss::new(Instant::now()),
            },
            receiving: ReceiveOrdering {
                frame_timeout: settings.missing_frame_timeout,
                reorder: PacketReorder::default(),
                shards: ShardAssembly::default(),
                reads: JoinSet::new(),
                ordered: false,
                datagrams_seen: false,
                stream_done: false,
                datagrams_closed: false,
                uni_closed: false,
            },
            _data_channel: data_channel,
        }
    }

    /// Serialize and send one packet. The leading caps and every packet in
    /// `stream` mode go length-framed on the bidirectional stream. In the other
    /// modes each packet carries a [`PacketOrder`] and goes on its own
    /// unidirectional stream, or as datagram shards for a data frame in
    /// `datagrams` mode.
    pub(crate) async fn send(&mut self, packet: &PipelinePacket) -> Result<(), G2gError> {
        self.sending.forget_acknowledged();
        if self.leading || self.sending.mode == VideoMode::Stream {
            self.leading = false;
            return send_framed(&mut self.tx, packet).await;
        }
        let order = PacketOrder {
            frames_before: self.sending.frames_sent,
            controls_before: self.sending.controls_sent,
        };
        let bytes = order.encode(packet)?;
        let PipelinePacket::DataFrame(frame) = packet else {
            self.sending.controls_sent += 1;
            let stream = self.send_on_new_stream(&bytes).await?;
            self.sending.unacknowledged.push_back(UnacknowledgedStream {
                stream,
                frame_sent_at: None,
            });
            return Ok(());
        };
        self.sending.frames_sent += 1;
        if frame.timing.keyframe {
            self.sending.reset_unfinished_frames();
        }
        if self.sending.mode == VideoMode::Datagrams {
            let path = Connection::stats(&self.session).path;
            self.sending.path_loss.observe(
                Instant::now(),
                PathCounts {
                    lost_packets: path.lost_packets,
                    sent_packets: path.sent_packets,
                },
            );
            let max_datagram_len = self.session.max_datagram_size();
            for shard in shard_frame(
                order.frames_before,
                &bytes,
                max_datagram_len,
                &self.sending.path_loss,
            )? {
                self.session.send_datagram(shard).map_err(wt_err)?;
            }
            return Ok(());
        }
        let stream = self.send_on_new_stream(&bytes).await?;
        self.sending.unacknowledged.push_back(UnacknowledgedStream {
            stream,
            frame_sent_at: Some(Instant::now()),
        });
        Ok(())
    }

    async fn send_on_new_stream(&mut self, bytes: &[u8]) -> Result<SendStream, G2gError> {
        let mut stream = self.session.open_uni().await.map_err(wt_err)?;
        stream.write_all(bytes).await.map_err(wt_err)?;
        stream.finish().map_err(wt_err)?;
        Ok(stream)
    }

    /// Whether the receiver asked for a keyframe, or a frame stream has gone
    /// unacknowledged past the stall timeout, since the last call.
    pub(crate) fn take_keyframe_request(&mut self) -> bool {
        let requested = self.read_keyframe_requests();
        self.sending.forget_acknowledged();
        self.sending.stalled() || requested
    }

    fn read_keyframe_requests(&mut self) -> bool {
        let mut context = Context::from_waker(Waker::noop());
        let mut bytes = [0u8; 64];
        let mut requested = false;
        loop {
            let polled = pin!(self.rx.read(&mut bytes)).poll(&mut context);
            let Poll::Ready(Ok(Some(count))) = polled else {
                return requested;
            };
            let received = &bytes[..count];
            if received.iter().any(|byte| *byte != KEYFRAME_REQUEST) {
                g2g_warn!(
                    Target::category("remotewt"),
                    "ignoring unknown bytes on the keyframe request stream"
                );
            }
            requested |= received.contains(&KEYFRAME_REQUEST);
        }
    }

    pub(crate) async fn request_keyframe(&mut self) -> Result<(), G2gError> {
        self.tx.write_all(&[KEYFRAME_REQUEST]).await.map_err(wt_err)
    }

    /// Read the next packet, from the bidirectional stream or, once the peer
    /// sends in another video mode, from its unidirectional streams and
    /// datagrams in send order. `Ok(None)` once the peer finishes sending.
    pub(crate) fn latency(&self) -> LatencyReport {
        let held_ns = match (self.receiving.datagrams_seen, self.receiving.frame_timeout) {
            (true, Some(timeout)) => timeout.as_nanos() as u64,
            _ => 0,
        };
        LatencyReport::live(held_ns, None)
    }

    pub(crate) async fn recv(&mut self) -> Result<Option<PipelinePacket>, G2gError> {
        let mut packet = self.recv_packet().await?;
        let stamped = match &mut packet {
            Some(PipelinePacket::DataFrame(frame)) if frame.timing.arrival_ns != 0 => Some(frame),
            _ => None,
        };
        if let Some(frame) = stamped {
            // the wire age stops at the sender, quinn's smoothed round trip covers the path
            let one_way_ns = u64::try_from((self.session.rtt() / 2).as_nanos()).unwrap_or(u64::MAX);
            frame.timing.arrival_ns = frame.timing.arrival_ns.saturating_sub(one_way_ns).max(1);
        }
        Ok(packet)
    }

    async fn recv_packet(&mut self) -> Result<Option<PipelinePacket>, G2gError> {
        if self.leading {
            let packet = self.reader.recv(&mut self.rx).await?;
            self.leading = false;
            if let Some(PipelinePacket::CapsChanged(caps)) = &packet {
                if self.receiving.frame_timeout.is_none() {
                    self.receiving.frame_timeout = frame_period(caps);
                }
            }
            return Ok(packet);
        }
        loop {
            while let Some(read) = self.receiving.reads.try_join_next() {
                self.receiving.insert_read(read)?;
            }
            let timeout = self.receiving.missing_frame_timeout()?;
            let next = self.receiving.reorder.pop(Instant::now(), timeout);
            if self.receiving.reorder.take_keyframe_wanted() {
                self.request_keyframe().await?;
            }
            let deadline = match next {
                NextPacket::Ready(PipelinePacket::Eos) => {
                    // the sender waits for this close before closing, so its last streams are read
                    self.session.close(0, b"eos");
                    return Ok(Some(PipelinePacket::Eos));
                }
                NextPacket::Ready(packet) => return Ok(Some(packet)),
                NextPacket::Wait { deadline } => deadline,
            };
            if self.receiving.finished() {
                return Ok(None);
            }
            match self.next_event(deadline).await {
                ReceiveEvent::Datagram(Ok(bytes)) => {
                    self.receiving.ordered = true;
                    self.receiving.datagrams_seen = true;
                    let next_frame = self.receiving.reorder.next_frame();
                    if let Some(packet) = self.receiving.shards.insert(&bytes, next_frame)? {
                        self.receiving.insert(&packet)?;
                    }
                }
                ReceiveEvent::Datagram(Err(_)) => self.receiving.datagrams_closed = true,
                ReceiveEvent::UniStream(Ok(mut stream)) => {
                    self.receiving.ordered = true;
                    // wire bodies carry their own length checks, the stream end bounds the read
                    self.receiving
                        .reads
                        .spawn(async move { stream.read_to_end(usize::MAX).await });
                }
                ReceiveEvent::UniStream(Err(_)) => self.receiving.uni_closed = true,
                ReceiveEvent::UniRead(read) => self.receiving.insert_read(read)?,
                ReceiveEvent::Stream(Ok(Some(packet))) => return Ok(Some(packet)),
                ReceiveEvent::Stream(Ok(None)) => {
                    self.receiving.stream_done = true;
                    if !self.receiving.ordered {
                        return Ok(None);
                    }
                }
                ReceiveEvent::Stream(Err(error)) => return Err(error),
                ReceiveEvent::Deadline => {}
            }
        }
    }

    async fn next_event(&mut self, deadline: Option<Instant>) -> ReceiveEvent {
        let Self {
            session,
            rx,
            reader,
            receiving,
            ..
        } = self;
        let ReceiveOrdering {
            reads,
            stream_done,
            datagrams_closed,
            uni_closed,
            ..
        } = receiving;
        let wake_at = deadline.unwrap_or_else(Instant::now);
        tokio::select! {
            // Datagrams first: a stream keeps whatever it buffered until it is
            // read, while an undrained datagram queue loses frames.
            biased;
            datagram = session.read_datagram(), if !*datagrams_closed => ReceiveEvent::Datagram(datagram),
            Some(read) = reads.join_next(), if !reads.is_empty() => ReceiveEvent::UniRead(read),
            stream = session.accept_uni(), if !*uni_closed => ReceiveEvent::UniStream(stream),
            framed = reader.recv(rx), if !*stream_done => ReceiveEvent::Stream(framed),
            () = tokio::time::sleep_until(wake_at), if deadline.is_some() => ReceiveEvent::Deadline,
        }
    }

    /// Half-close: finish our send direction and wait (bounded) for the peer to
    /// acknowledge the data, so the far side reads every stream to its end.
    pub(crate) async fn finish(&mut self) {
        let _ = self.tx.finish();
        let unacknowledged = core::mem::take(&mut self.sending.unacknowledged);
        let tx = &self.tx;
        let session = &self.session;
        let ordered = self.sending.mode != VideoMode::Stream;
        let acknowledged = async {
            let _ = tx.stopped().await;
            for entry in &unacknowledged {
                let _ = entry.stream.stopped().await;
            }
            if ordered {
                // quinn fails reads of acknowledged but unread stream data once the connection closes
                session.closed().await;
            }
        };
        let _ = tokio::time::timeout(FLUSH_TIMEOUT, acknowledged).await;
        self.session.close(0, b"eos");
    }
}

fn frame_period(caps: &Caps) -> Option<Duration> {
    let Rate::Fixed(q16) = caps.framerate() else {
        return None;
    };
    framerate_q16_period_ns(q16).map(Duration::from_nanos)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct VideoSettings {
    pub(crate) mode: VideoMode,
    pub(crate) stall_timeout: Option<Duration>,
    pub(crate) missing_frame_timeout: Option<Duration>,
}

/// Open the sender's bidirectional stream on `session`.
pub(crate) async fn open_stream(
    session: Session,
    settings: VideoSettings,
    data_channel: Option<&SharedDataChannel>,
) -> Result<QuicStream, G2gError> {
    // `Session::max_datagram_size` panics on a peer that refused datagram
    // support, so the configured mode is checked once here rather than per frame.
    if settings.mode == VideoMode::Datagrams && Connection::max_datagram_size(&session).is_none() {
        return Err(G2gError::NotConfigured);
    }
    let pump = data_channel.map(|channel| accept_data_channel(&session, channel.clone()));
    let (tx, rx) = session.open_bi().await.map_err(wt_err)?;
    Ok(QuicStream::new(session, tx, rx, settings, pump))
}

pub(crate) type SharedDataChannel = Arc<tokio::sync::Mutex<DataChannelTransportEnds>>;

pub(crate) fn open_data_channel(session: &Session, channel: SharedDataChannel) -> AbortOnDrop {
    let session = session.clone();
    AbortOnDrop(tokio::spawn(async move {
        if let Ok((tx, rx)) = session.open_bi().await {
            carry_messages(tx, rx, channel).await;
        }
    }))
}

fn accept_data_channel(session: &Session, channel: SharedDataChannel) -> AbortOnDrop {
    let session = session.clone();
    AbortOnDrop(tokio::spawn(async move {
        if let Ok((tx, rx)) = session.accept_bi().await {
            carry_messages(tx, rx, channel).await;
        }
    }))
}

async fn carry_messages(mut tx: SendStream, mut rx: RecvStream, channel: SharedDataChannel) {
    // tokio releases the lock when a replaced connection's task is aborted
    let mut ends = channel.lock().await;
    let DataChannelTransportEnds { outgoing, incoming } = &mut *ends;
    let mut reader = FramedRead::default();
    let mut sending = true;
    let mut receiving = true;
    // a peer can finish its send side and keep receiving
    while sending || receiving {
        tokio::select! {
            message = outgoing.recv(), if sending => match message {
                Some(message) => {
                    if send_frame_bytes(&mut tx, &message).await.is_err() {
                        return;
                    }
                }
                None => {
                    let _ = tx.finish();
                    sending = false;
                }
            },
            received = reader.recv_bytes(&mut rx, MAX_MESSAGE_SIZE), if receiving => match received {
                Ok(Some(message)) => {
                    if incoming.send(message.to_vec()).is_err() {
                        return;
                    }
                }
                Ok(None) => receiving = false,
                Err(error) => {
                    g2g_warn!(
                        Target::category("remotewt"),
                        "data channel closed on a bad message: {error:?}"
                    );
                    return;
                }
            },
        }
    }
}

pub(crate) const RAW_QUIC_ALPN: &str = "g2g-remote";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuicMode {
    Caller,
    Listener,
}

impl QuicMode {
    pub(crate) fn name(self) -> &'static str {
        match self {
            QuicMode::Caller => "caller",
            QuicMode::Listener => "listener",
        }
    }

    pub(crate) fn from_name(name: &str) -> Option<Self> {
        match name {
            "caller" => Some(QuicMode::Caller),
            "listener" => Some(QuicMode::Listener),
            _ => None,
        }
    }
}

pub(crate) fn set_mode(target: &mut QuicMode, value: &PropValue) -> Result<(), PropError> {
    let name = value.as_str().ok_or(PropError::Type)?;
    *target = QuicMode::from_name(name).ok_or(PropError::Value)?;
    Ok(())
}

pub(crate) fn url_host(url: &url::Url) -> Result<&str, G2gError> {
    // url leaves the host of a non-special scheme such as quic:// unparsed, brackets included
    let host = url.host_str().ok_or(G2gError::NotConfigured)?;
    Ok(host.trim_start_matches('[').trim_end_matches(']'))
}

pub(crate) fn bind_address(location: &str) -> Result<SocketAddr, G2gError> {
    let url = url::Url::parse(location).map_err(|_| G2gError::NotConfigured)?;
    let port = url.port().ok_or(G2gError::NotConfigured)?;
    let ip: core::net::IpAddr = url_host(&url)?
        .parse()
        .map_err(|_| G2gError::NotConfigured)?;
    Ok(SocketAddr::new(ip, port))
}
const RAW_QUIC_SCHEME: &str = "quic";

// a peer that vanishes without a close ends its session after this long
const IDLE_TIMEOUT: Duration = Duration::from_secs(5);
// quinn's idle timer restarts on any packet from the peer, and keep-alives make an idle peer send one
const KEEP_ALIVE_INTERVAL: Duration = Duration::from_secs(1);

fn transport_config(congestion: &str) -> Arc<quinn::TransportConfig> {
    let mut transport = quinn::TransportConfig::default();
    transport
        .keep_alive_interval(Some(KEEP_ALIVE_INTERVAL))
        .max_idle_timeout(Some(
            quinn::IdleTimeout::try_from(IDLE_TIMEOUT).expect("5 s fits a QUIC varint"),
        ));
    // web-transport-quinn keeps its factory mapping private, so raw QUIC repeats it
    match congestion_control(congestion) {
        CongestionControl::LowLatency => {
            transport
                .congestion_controller_factory(Arc::new(quinn::congestion::BbrConfig::default()));
        }
        CongestionControl::Throughput => {
            transport
                .congestion_controller_factory(Arc::new(quinn::congestion::CubicConfig::default()));
        }
        CongestionControl::Default => {}
    }
    Arc::new(transport)
}

#[derive(Debug)]
struct PinnedCertificates {
    provider: crypto::Provider,
    hashes: Vec<Vec<u8>>,
}

impl ServerCertVerifier for PinnedCertificates {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        let hash = crypto::sha256(&self.provider, end_entity);
        if self
            .hashes
            .iter()
            .any(|pinned| pinned.as_slice() == hash.as_ref())
        {
            return Ok(ServerCertVerified::assertion());
        }
        Err(rustls::Error::InvalidCertificate(
            rustls::CertificateError::UnknownIssuer,
        ))
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        signature: &rustls::DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            signature,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        signature: &rustls::DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            signature,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }
}

pub(crate) fn client_tls(hashes: Vec<Vec<u8>>) -> Result<rustls::ClientConfig, G2gError> {
    let provider = crypto::default_provider();
    let builder = rustls::ClientConfig::builder_with_provider(provider.clone())
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(wt_err)?;
    let tls = if hashes.is_empty() {
        let mut roots = rustls::RootCertStore::empty();
        for cert in rustls_native_certs::load_native_certs().certs {
            roots.add(cert).map_err(wt_err)?;
        }
        builder.with_root_certificates(roots).with_no_client_auth()
    } else {
        builder
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(PinnedCertificates { provider, hashes }))
            .with_no_client_auth()
    };
    Ok(tls)
}

async fn dial_raw(url: &url::Url, hashes: &str, congestion: &str) -> Result<Session, G2gError> {
    let host = url_host(url)?;
    let port = url.port().ok_or(G2gError::NotConfigured)?;
    let address = tokio::net::lookup_host((host, port))
        .await
        .map_err(wt_err)?
        .next()
        .ok_or(G2gError::Network(NetworkFailure::Stream))?;
    let mut tls = client_tls(parse_cert_hashes(hashes)?)?;
    tls.alpn_protocols = vec![RAW_QUIC_ALPN.as_bytes().to_vec()];
    let mut config =
        quinn::ClientConfig::new(Arc::new(QuicClientConfig::try_from(tls).map_err(wt_err)?));
    config.transport_config(transport_config(congestion));
    let unspecified: SocketAddr = match address {
        SocketAddr::V4(_) => (Ipv4Addr::UNSPECIFIED, 0).into(),
        SocketAddr::V6(_) => (Ipv6Addr::UNSPECIFIED, 0).into(),
    };
    // quinn keeps the endpoint running while any of its connections is open
    let endpoint = Endpoint::client(unspecified).map_err(wt_err)?;
    let connection = endpoint
        .connect_with(config, address, host)
        .map_err(wt_err)?
        .await
        .map_err(wt_err)?;
    Ok(Session::raw(connection))
}

pub(crate) fn listen(
    bind: SocketAddr,
    chain: Vec<CertificateDer<'static>>,
    key: PrivateKeyDer<'static>,
    congestion: &str,
) -> Result<Endpoint, G2gError> {
    let mut tls = rustls::ServerConfig::builder_with_provider(crypto::default_provider())
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(wt_err)?
        .with_no_client_auth()
        .with_single_cert(chain, key)
        .map_err(wt_err)?;
    tls.alpn_protocols = vec![
        web_transport_quinn::ALPN.as_bytes().to_vec(),
        RAW_QUIC_ALPN.as_bytes().to_vec(),
    ];
    let mut config = quinn::ServerConfig::with_crypto(Arc::new(
        QuicServerConfig::try_from(tls).map_err(wt_err)?,
    ));
    config.transport_config(transport_config(congestion));
    Endpoint::server(config, bind).map_err(wt_err)
}

#[derive(Debug)]
pub(crate) struct Listeners {
    quic: Endpoint,
    tcp: tokio::net::TcpListener,
    tls: Arc<rustls::ServerConfig>,
}

impl Listeners {
    pub(crate) fn local_addr(&self) -> Result<SocketAddr, G2gError> {
        self.quic.local_addr().map_err(wt_err)
    }
}

// a port the OS picked for UDP can already be taken on TCP
const SAME_PORT_ATTEMPTS: usize = 8;

pub(crate) fn listen_with_fallback(
    bind: SocketAddr,
    chain: Vec<CertificateDer<'static>>,
    key: PrivateKeyDer<'static>,
    congestion: &str,
) -> Result<Listeners, G2gError> {
    let tls = crate::remotewtws::server_tls(chain.clone(), key.clone_key())?;
    let attempts = if bind.port() == 0 {
        SAME_PORT_ATTEMPTS
    } else {
        1
    };
    let mut last_error = G2gError::NotConfigured;
    for _ in 0..attempts {
        let quic = listen(bind, chain.clone(), key.clone_key(), congestion)?;
        let port = quic.local_addr().map_err(wt_err)?.port();
        match std::net::TcpListener::bind(SocketAddr::new(bind.ip(), port)) {
            Ok(tcp) => {
                tcp.set_nonblocking(true).map_err(wt_err)?;
                let tcp = tokio::net::TcpListener::from_std(tcp).map_err(wt_err)?;
                return Ok(Listeners { quic, tcp, tls });
            }
            Err(error) => last_error = wt_err(error),
        }
    }
    Err(last_error)
}

#[derive(Debug)]
pub(crate) enum Connected {
    Quic(Box<Session>),
    WebSocket(Box<TlsWebSocket<tokio::net::TcpStream>>),
}

pub(crate) async fn accept_either(listeners: &Listeners) -> Result<Connected, G2gError> {
    loop {
        tokio::select! {
            session = accept(&listeners.quic) => return session.map(|session| Connected::Quic(Box::new(session))),
            websocket = crate::remotewtws::accept(&listeners.tcp, &listeners.tls) => match websocket {
                Ok(websocket) => return Ok(Connected::WebSocket(Box::new(websocket))),
                Err(_) => g2g_warn!(
                    Target::category("remotewt"),
                    "dropped a TCP connection that did not complete the WebSocket handshake"
                ),
            },
        }
    }
}

async fn dial_websocket(
    url: &url::Url,
    hashes: &str,
) -> Result<TlsWebSocket<tokio::net::TcpStream>, G2gError> {
    if url.scheme() != WEBSOCKET_SCHEME {
        return Err(G2gError::NotConfigured);
    }
    crate::remotewtws::dial(url, client_tls(parse_cert_hashes(hashes)?)?).await
}

const FALLBACK_DELAY: Duration = Duration::from_secs(1);

pub(crate) async fn dial_with_fallback(
    location: &str,
    fallback_location: &str,
    hashes: &str,
    congestion: &str,
) -> Result<Connected, G2gError> {
    let url = url::Url::parse(location).map_err(wt_err)?;
    if url.scheme() == WEBSOCKET_SCHEME {
        return dial_websocket(&url, hashes)
            .await
            .map(|websocket| Connected::WebSocket(Box::new(websocket)));
    }
    if fallback_location.is_empty() {
        return dial(location, hashes, &[], congestion)
            .await
            .map(|session| Connected::Quic(Box::new(session)));
    }
    let fallback = url::Url::parse(fallback_location).map_err(wt_err)?;
    let mut quic = pin!(dial(location, hashes, &[], congestion));
    let mut websocket = pin!(async {
        tokio::time::sleep(FALLBACK_DELAY).await;
        dial_websocket(&fallback, hashes).await
    });
    tokio::select! {
        session = &mut quic => match session {
            Ok(session) => Ok(Connected::Quic(Box::new(session))),
            Err(_) => websocket.await.map(|websocket| Connected::WebSocket(Box::new(websocket))),
        },
        websocket_dialed = &mut websocket => match websocket_dialed {
            Ok(websocket) => Ok(Connected::WebSocket(Box::new(websocket))),
            Err(_) => quic.await.map(|session| Connected::Quic(Box::new(session))),
        },
    }
}

fn negotiated_alpn(connection: &Connection) -> Option<Vec<u8>> {
    connection
        .handshake_data()?
        .downcast::<HandshakeData>()
        .ok()?
        .protocol
}

pub(crate) async fn accept(endpoint: &Endpoint) -> Result<Session, G2gError> {
    let incoming = endpoint.accept().await.ok_or(G2gError::NotConfigured)?;
    let connection = incoming.await.map_err(wt_err)?;
    if negotiated_alpn(&connection).as_deref() == Some(RAW_QUIC_ALPN.as_bytes()) {
        return Ok(Session::raw(connection));
    }
    let request = Request::accept(connection).await.map_err(wt_err)?;
    request.ok().await.map_err(wt_err)
}

/// Dial `url` and complete the HTTP/3 CONNECT handshake, or the raw QUIC
/// handshake for a `quic://` URL, leaving stream opening to the caller. `hashes`
/// is the `server-certificate-hashes` property: empty accepts any certificate a
/// system root signs, otherwise only the listed certificates. `protocols` names the WebTransport subprotocols offered in
/// preference order, which is how a protocol layered on the carrier (MoQT's
/// `moqt-16` / `moqt-18`) states its version, since the QUIC ALPN is always
/// `h3`. The server's pick is `session.response().protocol`. `congestion` is a
/// [`CONGESTION_PROP`] nick.
pub(crate) async fn dial(
    url: &str,
    hashes: &str,
    protocols: &[&str],
    congestion: &str,
) -> Result<Session, G2gError> {
    let url = url::Url::parse(url).map_err(wt_err)?;
    if url.scheme() == RAW_QUIC_SCHEME {
        // raw QUIC states its version in the ALPN, so it has no subprotocols to offer
        if !protocols.is_empty() {
            return Err(G2gError::NotConfigured);
        }
        return dial_raw(&url, hashes, congestion).await;
    }
    let hashes = parse_cert_hashes(hashes)?;
    let builder = ClientBuilder::new().with_congestion_control(congestion_control(congestion));
    let client = if hashes.is_empty() {
        builder.with_system_roots().map_err(wt_err)?
    } else {
        builder
            .with_server_certificate_hashes(hashes)
            .map_err(wt_err)?
    };
    let mut request = ConnectRequest::new(url);
    for protocol in protocols {
        request = request.with_protocol(*protocol);
    }
    client.connect(request).await.map_err(wt_err)
}

pub(crate) const CERTIFICATE_PROP: PropertySpec = PropertySpec::new(
    "certificate",
    PropKind::Str,
    "path to the TLS certificate chain (PEM) the listening side presents",
);

pub(crate) const PRIVATE_KEY_PROP: PropertySpec = PropertySpec::new(
    "private-key",
    PropKind::Str,
    "path to the TLS private key (PEM) for `certificate`",
);

/// The `congestion-control` spec, shared by the WebTransport client and server
/// elements (`web-transport-quinn` takes the same choice on both builders).
pub(crate) const CONGESTION_PROP: PropertySpec = PropertySpec::new(
    "congestion-control",
    PropKind::Str,
    "QUIC congestion controller: default or throughput (CUBIC), low-latency (BBR)",
)
.with_default("default")
.with_enum_values("default | throughput | low-latency");

/// Map a `congestion-control` nick to the algorithm both builders take. `None`
/// for anything else, which is how [`set_congestion`] rejects it.
fn congestion(nick: &str) -> Option<CongestionControl> {
    match nick {
        "default" => Some(CongestionControl::Default),
        "throughput" => Some(CongestionControl::Throughput),
        "low-latency" => Some(CongestionControl::LowLatency),
        _ => None,
    }
}

/// The stored nick as an algorithm, for a builder. Unknown text never gets
/// stored, so the fallback only covers a never-set value.
pub(crate) fn congestion_control(nick: &str) -> CongestionControl {
    congestion(nick).unwrap_or(CongestionControl::Default)
}

/// The `congestion-control` half of `set_property`: store the nick if it names
/// an algorithm. A launch line is checked against `enum_values` before it gets
/// here, a direct `set_property` call is not.
pub(crate) fn set_congestion(target: &mut String, value: &PropValue) -> Result<(), PropError> {
    let nick = value.as_str().ok_or(PropError::Type)?;
    congestion(nick).ok_or(PropError::Value)?;
    *target = nick.to_string();
    Ok(())
}

/// Load a PEM certificate chain and its private key from file paths (the
/// `certificate` / `private-key` properties). QUIC is always TLS, so a
/// WebTransport server cannot start without them.
pub(crate) fn load_certificate(
    cert: &str,
    key: &str,
) -> Result<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>), G2gError> {
    if cert.is_empty() || key.is_empty() {
        return Err(G2gError::NotConfigured);
    }
    let cert = std::fs::read(cert).map_err(wt_err)?;
    let key = std::fs::read(key).map_err(wt_err)?;
    parse_certificate(&cert, &key)
}

pub(crate) fn parse_certificate(
    cert: &[u8],
    key: &[u8],
) -> Result<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>), G2gError> {
    let chain: Vec<CertificateDer<'static>> = CertificateDer::pem_slice_iter(cert)
        .collect::<Result<_, _>>()
        .map_err(wt_err)?;
    if chain.is_empty() {
        return Err(G2gError::NotConfigured);
    }
    let key = PrivateKeyDer::from_pem_slice(key).map_err(wt_err)?;
    Ok((chain, key))
}

#[derive(Clone)]
pub struct SelfSignedCertificate {
    pub certificate_pem: String,
    pub private_key_pem: String,
    pub sha256_hex: String,
}

impl core::fmt::Debug for SelfSignedCertificate {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SelfSignedCertificate")
            .field("sha256_hex", &self.sha256_hex)
            .finish_non_exhaustive()
    }
}

// browsers accept a serverCertificateHashes certificate only when it is valid for at most 14 days
const CERTIFICATE_LIFETIME: time::Duration = time::Duration::days(13);
// clients whose clock runs behind the host's still see the certificate as valid
const CERTIFICATE_BACKDATE: time::Duration = time::Duration::hours(1);

pub fn generate_certificate() -> Result<SelfSignedCertificate, rcgen::Error> {
    let mut params = rcgen::CertificateParams::new(vec!["localhost".to_string()])?;
    let now = time::OffsetDateTime::now_utc();
    params.not_before = now - CERTIFICATE_BACKDATE;
    params.not_after = now + CERTIFICATE_LIFETIME;
    let key = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256)?;
    let certificate = params.self_signed(&key)?;
    let digest = crypto::sha256(&crypto::default_provider(), certificate.der());
    Ok(SelfSignedCertificate {
        certificate_pem: certificate.pem(),
        private_key_pem: key.serialize_pem(),
        sha256_hex: hex_encode(digest.as_ref()),
    })
}

/// Parse a comma-separated list of hex SHA-256 certificate digests (the browser
/// WebTransport API's `serverCertificateHashes`, for a self-signed or short-lived
/// certificate no system root covers). Empty means "use the system roots".
pub(crate) fn parse_cert_hashes(spec: &str) -> Result<Vec<Vec<u8>>, G2gError> {
    let mut out = Vec::new();
    for item in spec.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let bytes = hex_decode(&item.replace(':', "")).ok_or(G2gError::NotConfigured)?;
        // A SHA-256 digest and nothing else: the verifier compares raw bytes, so
        // a wrong-length entry would silently never match.
        if bytes.len() != 32 {
            return Err(G2gError::NotConfigured);
        }
        out.push(bytes);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn listener_location_binds_an_ip_literal_for_either_scheme() {
        let v4: SocketAddr = "127.0.0.1:4433".parse().unwrap();
        let v6: SocketAddr = "[::1]:4433".parse().unwrap();
        assert_eq!(bind_address("quic://127.0.0.1:4433"), Ok(v4));
        assert_eq!(bind_address("https://127.0.0.1:4433"), Ok(v4));
        assert_eq!(bind_address("quic://[::1]:4433"), Ok(v6));
        assert_eq!(bind_address("https://[::1]:4433/path"), Ok(v6));
        // a listener binds an address, so a name or a missing port is refused
        assert_eq!(
            bind_address("quic://localhost:4433"),
            Err(G2gError::NotConfigured)
        );
        assert_eq!(
            bind_address("quic://127.0.0.1"),
            Err(G2gError::NotConfigured)
        );
    }

    #[test]
    fn cert_hashes_parse_and_reject_malformed() {
        let one = "aa".repeat(32);
        let parsed = parse_cert_hashes(&one).expect("one digest");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].len(), 32);
        assert_eq!(parsed[0][0], 0xaa);

        let two = alloc::format!("{one},{one}");
        assert_eq!(parse_cert_hashes(&two).expect("two digests").len(), 2);
        assert!(parse_cert_hashes("")
            .expect("empty is system roots")
            .is_empty());
        // Too short (a truncated digest would never match) and non-hex.
        assert!(parse_cert_hashes("aabb").is_err());
        assert!(parse_cert_hashes(&"zz".repeat(32)).is_err());
    }
}
