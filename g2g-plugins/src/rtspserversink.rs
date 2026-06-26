//! RTSP server sink (RtspServerSink, `rtsp-server` feature): hosts a pipeline's
//! H.264 as an RTSP endpoint and serves it as RTP/UDP to connecting players (the
//! OBS / surveillance / contribution-server shape). The sans-IO
//! [`rtspserver::RtspResponder`](crate::rtspserver) does the protocol work
//! (OPTIONS / DESCRIBE / SETUP / PLAY); this element is the tokio TCP control
//! channel + the UDP RTP transport around it, reusing the
//! [`RtpH264Packetizer`](crate::rtppay) the UDP sink uses.
//!
//! Multi-client: the listener is bound in `configure_pipeline`; the first buffer
//! blocks until one player has connected and PLAYed (so a stream that is only
//! watched by one viewer behaves predictably), and from then on every buffer
//! also opportunistically accepts new players and advances their handshakes
//! without blocking, broadcasting each frame to every PLAYing player on its own
//! RTP session. Players that disconnect are reaped. One shared RTP UDP socket
//! sends to each player's negotiated address. ANNOUNCE/RECORD ingest is the
//! separate [`RtspServerSrc`](crate::rtspserversrc).

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::vec::Vec;

use std::net::{IpAddr, SocketAddr, TcpListener as StdTcpListener};

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    HardwareError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, Rate,
    VideoCodec,
};

use crate::filesink::io_err;
use crate::rtppay::RtpH264Packetizer;
use crate::rtspserver::{sdp_h264, RtspEvent, RtspRequest, RtspResponder};

/// H.264 RTP media clock (RFC 6184): 90 kHz.
const RTP_CLOCK_HZ: u64 = 90_000;
/// Default dynamic RTP payload type for H.264.
const DEFAULT_PAYLOAD_TYPE: u8 = 96;
/// Default max RTP payload bytes, leaving headroom under a 1500-byte MTU.
const DEFAULT_MAX_PAYLOAD: usize = 1400;
/// TCP read buffer for RTSP control requests.
const CTRL_BUF: usize = 8192;
/// Cap on buffered-but-unparsed control bytes. A real RTSP request (even an
/// ANNOUNCE carrying SDP) is far smaller; the bound reaps a client that drips a
/// never-terminating request or an oversized Content-Length (slow-loris DoS).
const MAX_PENDING: usize = 64 * 1024;

/// H.264-at-any-geometry caps (geometry rides in-band in the SPS).
fn h264_any() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

/// One connected player: its control channel, protocol responder, the RTP
/// destination negotiated at SETUP, and the packetizer (its own RTP session,
/// armed at PLAY).
#[derive(Debug)]
struct Client {
    control: tokio::net::TcpStream,
    responder: RtspResponder,
    pending: Vec<u8>,
    peer_ip: IpAddr,
    dest: Option<SocketAddr>,
    packetizer: Option<RtpH264Packetizer>,
}

impl Client {
    /// PLAYing once SETUP gave us a destination and PLAY armed the packetizer.
    fn playing(&self) -> bool {
        self.dest.is_some() && self.packetizer.is_some()
    }

    /// Drain whatever control bytes are readable now (non-blocking) and answer
    /// the requests they complete, advancing toward PLAY. Returns `false` if the
    /// player disconnected or tore down (the caller reaps it).
    async fn advance(&mut self, payload_type: u8, ssrc: u32, max_payload: usize) -> bool {
        let mut buf = [0u8; CTRL_BUF];
        loop {
            match self.control.try_read(&mut buf) {
                Ok(0) => return false, // closed
                Ok(n) => self.pending.extend_from_slice(&buf[..n]),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(_) => return false,
            }
        }
        while let Some((req, consumed)) = RtspRequest::parse(&self.pending) {
            self.pending.drain(..consumed);
            let (response, event) = self.responder.handle_request(&req);
            if self.control.write_all(&response).await.is_err() {
                return false;
            }
            match event {
                RtspEvent::Setup { client_rtp_port } => {
                    self.dest = Some(SocketAddr::new(self.peer_ip, client_rtp_port));
                }
                RtspEvent::Play => {
                    self.packetizer = Some(
                        RtpH264Packetizer::new(payload_type, ssrc).with_max_payload(max_payload),
                    );
                }
                RtspEvent::Teardown => return false,
                _ => {}
            }
        }
        // A partial request that never completes must not grow without bound.
        if self.pending.len() > MAX_PENDING {
            return false;
        }
        true
    }

    /// Packetize `bytes` and send every RTP packet to this player. Returns the
    /// packet count, or `Err` if the send failed (the caller reaps it).
    async fn send_frame(
        &mut self,
        socket: &tokio::net::UdpSocket,
        bytes: &[u8],
        timestamp: u32,
    ) -> Result<u64, ()> {
        let (Some(packetizer), Some(dest)) = (self.packetizer.as_mut(), self.dest) else {
            return Ok(0);
        };
        let mut sent = 0;
        for pkt in &packetizer.packetize(bytes, timestamp) {
            if socket.send_to(pkt, dest).await.is_err() {
                return Err(());
            }
            sent += 1;
        }
        Ok(sent)
    }
}

#[derive(Debug)]
pub struct RtspServerSink {
    rtsp_addr: SocketAddr,
    payload_type: u8,
    ssrc: u32,
    max_payload: usize,
    listener: Option<StdTcpListener>,
    // Runtime, established lazily on the first buffer.
    tcp: Option<tokio::net::TcpListener>,
    rtp_socket: Option<tokio::net::UdpSocket>,
    clients: Vec<Client>,
    started: bool,
    configured: bool,
    frames_sent: u64,
    packets_sent: u64,
    eos_seen: bool,
}

impl RtspServerSink {
    /// Listen for RTSP players on `rtsp_addr` (e.g. `0.0.0.0:8554`).
    pub fn new(rtsp_addr: SocketAddr) -> Self {
        Self {
            rtsp_addr,
            payload_type: DEFAULT_PAYLOAD_TYPE,
            ssrc: 0,
            max_payload: DEFAULT_MAX_PAYLOAD,
            listener: None,
            tcp: None,
            rtp_socket: None,
            clients: Vec::new(),
            started: false,
            configured: false,
            frames_sent: 0,
            packets_sent: 0,
            eos_seen: false,
        }
    }

    /// Use an already-bound listener (so a test can pick an ephemeral port).
    pub fn from_listener(listener: StdTcpListener) -> Result<Self, G2gError> {
        let addr = listener.local_addr().map_err(io_err)?;
        Ok(Self { listener: Some(listener), configured: true, ..Self::new(addr) })
    }

    /// Set the RTP payload type and SSRC carried in every packet.
    pub fn with_rtp(mut self, payload_type: u8, ssrc: u32) -> Self {
        self.payload_type = payload_type & 0x7F;
        self.ssrc = ssrc;
        self
    }

    pub fn frames_sent(&self) -> u64 {
        self.frames_sent
    }

    pub fn packets_sent(&self) -> u64 {
        self.packets_sent
    }

    /// Number of players currently connected (PLAYing or mid-handshake).
    pub fn client_count(&self) -> usize {
        self.clients.len()
    }

    pub fn eos_seen(&self) -> bool {
        self.eos_seen
    }

    /// 90 kHz RTP timestamp for a presentation time.
    fn rtp_timestamp(pts_ns: u64) -> u32 {
        ((pts_ns as u128 * RTP_CLOCK_HZ as u128) / 1_000_000_000) as u32
    }

    /// Bind the shared RTP socket, promote the listener to tokio, then accept one
    /// player and drive its handshake to PLAY (blocking), so the first buffer is
    /// not dropped before anyone is watching. Subsequent players join without
    /// blocking via [`accept_new`](Self::accept_new).
    async fn bootstrap(&mut self) -> Result<(), G2gError> {
        let std_listener = self.listener.take().ok_or(G2gError::NotConfigured)?;
        std_listener.set_nonblocking(true).map_err(io_err)?;
        let listener = tokio::net::TcpListener::from_std(std_listener).map_err(io_err)?;
        let rtp_socket = tokio::net::UdpSocket::bind(("0.0.0.0", 0)).await.map_err(io_err)?;
        let server_rtp_port = rtp_socket.local_addr().map_err(io_err)?.port();

        let (mut control, peer) = listener.accept().await.map_err(io_err)?;
        let mut responder = RtspResponder::new(sdp_h264(self.payload_type), server_rtp_port, self.ssrc);
        let mut pending: Vec<u8> = Vec::new();
        let mut buf = [0u8; CTRL_BUF];
        let mut dest = None;
        let packetizer;
        'handshake: loop {
            let n = control.read(&mut buf).await.map_err(io_err)?;
            if n == 0 {
                return Err(G2gError::Hardware(HardwareError::Other)); // closed before PLAY
            }
            pending.extend_from_slice(&buf[..n]);
            while let Some((req, consumed)) = RtspRequest::parse(&pending) {
                pending.drain(..consumed);
                let (response, event) = responder.handle_request(&req);
                control.write_all(&response).await.map_err(io_err)?;
                match event {
                    RtspEvent::Setup { client_rtp_port } => {
                        dest = Some(SocketAddr::new(peer.ip(), client_rtp_port));
                    }
                    RtspEvent::Play => {
                        packetizer = Some(
                            RtpH264Packetizer::new(self.payload_type, self.ssrc)
                                .with_max_payload(self.max_payload),
                        );
                        break 'handshake;
                    }
                    RtspEvent::Teardown => return Err(G2gError::Shutdown),
                    _ => {}
                }
            }
        }
        self.clients.push(Client {
            control,
            responder,
            pending,
            peer_ip: peer.ip(),
            dest,
            packetizer,
        });
        self.tcp = Some(listener);
        self.rtp_socket = Some(rtp_socket);
        Ok(())
    }

    /// Accept every player whose TCP connection is already queued (non-blocking),
    /// adding each as a handshaking client.
    async fn accept_new(&mut self) {
        let (Some(listener), Some(rtp)) = (self.tcp.as_ref(), self.rtp_socket.as_ref()) else {
            return;
        };
        let Ok(server_rtp_port) = rtp.local_addr().map(|a| a.port()) else { return };
        // A zero timeout polls accept once: take a queued connection or stop.
        while let Ok(Ok((control, peer))) =
            tokio::time::timeout(Duration::from_millis(0), listener.accept()).await
        {
            let responder =
                RtspResponder::new(sdp_h264(self.payload_type), server_rtp_port, self.ssrc);
            self.clients.push(Client {
                control,
                responder,
                pending: Vec::new(),
                peer_ip: peer.ip(),
                dest: None,
                packetizer: None,
            });
        }
    }

    /// Advance every still-handshaking client (non-blocking), reaping any that
    /// disconnected.
    async fn advance_handshakes(&mut self) {
        let (pt, ssrc, mp) = (self.payload_type, self.ssrc, self.max_payload);
        let mut i = 0;
        while i < self.clients.len() {
            // A PLAYing client needs no handshake advance; keep it. Otherwise
            // advance it, reaping the connection if it died.
            let keep = self.clients[i].playing() || self.clients[i].advance(pt, ssrc, mp).await;
            if keep {
                i += 1;
            } else {
                self.clients.swap_remove(i);
            }
        }
    }

    /// Broadcast one frame to every PLAYing client, reaping any whose send fails.
    async fn broadcast(&mut self, bytes: &[u8], timestamp: u32) -> Result<(), G2gError> {
        let socket = self.rtp_socket.as_ref().ok_or(G2gError::NotConfigured)?;
        let mut i = 0;
        while i < self.clients.len() {
            if !self.clients[i].playing() {
                i += 1;
                continue;
            }
            match self.clients[i].send_frame(socket, bytes, timestamp).await {
                Ok(pkts) => {
                    self.packets_sent += pkts;
                    i += 1;
                }
                Err(()) => {
                    self.clients.swap_remove(i);
                }
            }
        }
        Ok(())
    }
}

impl AsyncElement for RtspServerSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&h264_any())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(h264_any()))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::CompressedVideo { codec: VideoCodec::H264, .. } => {}
            _ => return Err(G2gError::CapsMismatch),
        }
        if self.listener.is_none() && self.tcp.is_none() {
            self.listener = Some(StdTcpListener::bind(self.rtsp_addr).map_err(io_err)?);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "RTSP server sink",
            "Sink/Network",
            "Hosts an RTSP endpoint serving H.264 over RTP/UDP to multiple players",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    if !self.configured {
                        return Err(G2gError::NotConfigured);
                    }
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    // Block on the first buffer until one player connects + PLAYs,
                    // then serve every connected player without blocking.
                    if !self.started {
                        self.bootstrap().await?;
                        self.started = true;
                    } else {
                        self.accept_new().await;
                        self.advance_handshakes().await;
                    }
                    let timestamp = Self::rtp_timestamp(frame.timing.pts_ns);
                    let bytes = slice.as_slice();
                    self.broadcast(bytes, timestamp).await?;
                    self.frames_sent += 1;
                }
                // Best-effort TEARDOWN courtesy is a follow-up; RTP has no in-band
                // end marker, so a player times out on the stream stopping.
                PipelinePacket::Eos => self.eos_seen = true,
                // Sequence numbers persist across a seek (loss is tracked by gaps).
                PipelinePacket::Flush => {}
                // Geometry refinement lives in the in-band SPS, not in RTP/SDP.
                PipelinePacket::CapsChanged(_) => {}
                PipelinePacket::Segment(_) => {}
            }
            Ok(())
        })
    }
}

impl PadTemplates for RtspServerSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(CapsSet::one(h264_any()))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use tokio::io::AsyncWriteExt;

    async fn client_pair() -> (Client, tokio::net::TcpStream) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let peer = tokio::net::TcpStream::connect(addr).await.unwrap();
        let (server, _) = listener.accept().await.unwrap();
        let client = Client {
            control: server,
            responder: RtspResponder::new(sdp_h264(96), 6000, 0x1234_5678),
            pending: Vec::new(),
            peer_ip: std::net::IpAddr::from([127, 0, 0, 1]),
            dest: None,
            packetizer: None,
        };
        (client, peer)
    }

    #[tokio::test]
    async fn oversized_pending_request_reaps_the_client() {
        let (mut client, mut peer) = client_pair().await;
        // A never-terminating request (no double CRLF): the writer keeps the
        // connection open, so any reap is from the buffer cap, not a close.
        let writer = tokio::spawn(async move {
            let junk = vec![b'A'; MAX_PENDING + CTRL_BUF + 16];
            let _ = peer.write_all(&junk).await;
            peer // hold the socket open
        });
        let mut reaped = false;
        for _ in 0..10_000 {
            if !client.advance(96, 0x1234_5678, 1400).await {
                reaped = true;
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(reaped, "a client overflowing the control buffer is reaped");
        let _ = writer.await;
    }
}
