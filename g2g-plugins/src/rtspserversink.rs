//! RTSP server sink (RtspServerSink, `rtsp-server` feature): hosts a pipeline's
//! H.264 as an RTSP endpoint and serves it over RTP to connecting players (the
//! OBS / surveillance / contribution-server shape), on either transport a player
//! SETUPs: unicast RTP/UDP, or TCP-interleaved (RFC 2326 §10.12, `$`-framed RTP
//! on the control connection, what `ffmpeg -rtsp_transport tcp` uses, for players
//! behind a firewall that blocks the UDP ports). The sans-IO
//! [`rtspserver::RtspResponder`](crate::rtspserver) does the protocol work
//! (OPTIONS / DESCRIBE / SETUP / PLAY); this element is the tokio TCP control
//! channel + the RTP transport around it, reusing the
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
//!
//! RTCP + keepalive during PLAY: sender reports (NTP <-> RTP mapping, RFC 3550)
//! go to each player periodically, over UDP from the socket adjacent to the RTP
//! one (the advertised `server_port` pair) or `$`-framed on the RTCP channel for
//! an interleaved player, with a BYE at EOS. The SETUP response advertises
//! `Session: id;timeout=N`; a player silent past the timeout on both the control
//! channel (GET_PARAMETER / OPTIONS keepalive) and RTCP (receiver reports) is
//! reaped, so departed players do not accumulate.

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::vec::Vec;

use std::net::{IpAddr, SocketAddr, TcpListener as StdTcpListener};

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    HardwareError, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, Rate, VideoCodec,
};

use crate::filesink::io_err;
use crate::rtcp;
use crate::rtppay::RtpH264Packetizer;
use crate::rtspserver::{
    sdp_h264, RtspEvent, RtspRequest, RtspResponder, DEFAULT_SESSION_TIMEOUT_SECS,
};

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
/// Default RTCP sender-report interval (RFC 3550's nominal 5 s).
const DEFAULT_SR_INTERVAL_NS: u64 = 5_000_000_000;

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
    /// Where this player's RTCP lives when on UDP: the second port of the
    /// `client_port` pair. Sender reports go there; its receiver reports come
    /// from there.
    rtcp_dest: Option<SocketAddr>,
    /// The RTP channel once a TCP-interleaved SETUP was negotiated (RFC 2326
    /// §10.12): RTP rides the control connection as `$`-framed binary instead of
    /// its own UDP port. Mutually exclusive with `dest` in practice.
    interleaved: Option<u8>,
    /// The RTCP channel for an interleaved player: SRs out, RRs in, `$`-framed.
    rtcp_channel: Option<u8>,
    packetizer: Option<RtpH264Packetizer>,
    /// RFC 3550 sender counters for this player's RTP session.
    rtp_packets: u32,
    rtp_octets: u32,
    last_rtp_ts: u32,
    last_sr_ns: u64,
    /// Last control-channel or RTCP activity, for session-timeout reaping.
    last_activity_ns: u64,
}

impl Client {
    fn new(control: tokio::net::TcpStream, responder: RtspResponder, peer_ip: IpAddr) -> Self {
        Self {
            control,
            responder,
            pending: Vec::new(),
            peer_ip,
            dest: None,
            rtcp_dest: None,
            interleaved: None,
            rtcp_channel: None,
            packetizer: None,
            rtp_packets: 0,
            rtp_octets: 0,
            last_rtp_ts: 0,
            last_sr_ns: 0,
            last_activity_ns: g2g_core::metrics::monotonic_ns(),
        }
    }

    /// PLAYing once SETUP gave a transport (a UDP destination or an interleaved
    /// channel) and PLAY armed the packetizer.
    fn playing(&self) -> bool {
        self.packetizer.is_some() && (self.dest.is_some() || self.interleaved.is_some())
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
        loop {
            // An interleaved player sends its RTCP (receiver reports) `$`-framed
            // on the control connection; consume the frame before request parsing
            // so the binary is not misread as a request, and count it as
            // keepalive activity.
            if self.pending.first() == Some(&0x24) {
                if self.pending.len() < 4 {
                    break; // frame header not fully arrived
                }
                let len = u16::from_be_bytes([self.pending[2], self.pending[3]]) as usize;
                if self.pending.len() < 4 + len {
                    break;
                }
                self.pending.drain(..4 + len);
                self.last_activity_ns = g2g_core::metrics::monotonic_ns();
                continue;
            }
            let Some((req, consumed)) = RtspRequest::parse(&self.pending) else {
                break;
            };
            self.pending.drain(..consumed);
            self.last_activity_ns = g2g_core::metrics::monotonic_ns();
            let (response, event) = self.responder.handle_request(&req);
            if self.control.write_all(&response).await.is_err() {
                return false;
            }
            match event {
                RtspEvent::Setup { client_rtp_port } => {
                    self.dest = Some(SocketAddr::new(self.peer_ip, client_rtp_port));
                    self.rtcp_dest = Some(SocketAddr::new(
                        self.peer_ip,
                        client_rtp_port.saturating_add(1),
                    ));
                }
                RtspEvent::SetupInterleaved {
                    rtp_channel,
                    rtcp_channel,
                } => {
                    self.interleaved = Some(rtp_channel);
                    self.rtcp_channel = Some(rtcp_channel);
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

    /// Packetize `bytes` and send every RTP packet to this player over its
    /// negotiated transport: UDP to `dest`, or `$`-framed on the control
    /// connection for a TCP-interleaved client. Returns the packet count, or
    /// `Err` if the send failed (the caller reaps it).
    async fn send_frame(
        &mut self,
        socket: &tokio::net::UdpSocket,
        bytes: &[u8],
        timestamp: u32,
    ) -> Result<u64, ()> {
        let pkts = match self.packetizer.as_mut() {
            Some(p) => p.packetize(bytes, timestamp),
            None => return Ok(0),
        };
        self.last_rtp_ts = timestamp;
        let mut sent = 0;
        if let Some(channel) = self.interleaved {
            // RFC 2326 §10.12: `$` | channel | 16-bit length | RTP, on the TCP
            // control connection. An RTP packet is always well under 64 KiB.
            for pkt in &pkts {
                let mut framed = Vec::with_capacity(4 + pkt.len());
                framed.push(0x24);
                framed.push(channel);
                framed.extend_from_slice(&(pkt.len() as u16).to_be_bytes());
                framed.extend_from_slice(pkt);
                if self.control.write_all(&framed).await.is_err() {
                    return Err(());
                }
                sent += 1;
            }
        } else if let Some(dest) = self.dest {
            for pkt in &pkts {
                if socket.send_to(pkt, dest).await.is_err() {
                    return Err(());
                }
                sent += 1;
            }
        }
        // SR sender counters: media packets and their payload octets (past the
        // 12-byte RTP header).
        for pkt in &pkts {
            self.rtp_packets = self.rtp_packets.wrapping_add(1);
            self.rtp_octets = self
                .rtp_octets
                .wrapping_add(pkt.len().saturating_sub(12) as u32);
        }
        Ok(sent)
    }

    /// Send one RTCP packet over this player's negotiated RTCP transport:
    /// `$`-framed on the RTCP channel for an interleaved player, else UDP from
    /// the server RTCP socket to the client's RTCP port. `Err` if the send
    /// failed (the caller reaps it); `Ok(false)` if no transport is negotiated.
    async fn send_rtcp(
        &mut self,
        rtcp_socket: &tokio::net::UdpSocket,
        packet: &[u8],
    ) -> Result<bool, ()> {
        if let Some(channel) = self.rtcp_channel {
            let mut framed = Vec::with_capacity(4 + packet.len());
            framed.push(0x24);
            framed.push(channel);
            framed.extend_from_slice(&(packet.len() as u16).to_be_bytes());
            framed.extend_from_slice(packet);
            if self.control.write_all(&framed).await.is_err() {
                return Err(());
            }
            Ok(true)
        } else if let Some(dest) = self.rtcp_dest {
            if rtcp_socket.send_to(packet, dest).await.is_err() {
                return Err(());
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

/// # Example
///
/// ```no_run
/// use core::time::Duration;
/// use g2g_plugins::rtspserversink::RtspServerSink;
///
/// let sink = RtspServerSink::new("0.0.0.0:8554".parse().unwrap())
///     .with_rtp(96, 0x1234_5678)
///     .with_rtcp_sr_interval(Duration::from_secs(5));
/// ```
#[derive(Debug)]
pub struct RtspServerSink {
    rtsp_addr: SocketAddr,
    payload_type: u8,
    ssrc: u32,
    max_payload: usize,
    listener: Option<StdTcpListener>,
    sr_interval_ns: u64,
    session_timeout_ns: u64,
    // Runtime, established lazily on the first buffer.
    tcp: Option<tokio::net::TcpListener>,
    rtp_socket: Option<tokio::net::UdpSocket>,
    rtcp_socket: Option<tokio::net::UdpSocket>,
    clients: Vec<Client>,
    started: bool,
    configured: bool,
    frames_sent: u64,
    packets_sent: u64,
    sender_reports_sent: u64,
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
            sr_interval_ns: DEFAULT_SR_INTERVAL_NS,
            session_timeout_ns: DEFAULT_SESSION_TIMEOUT_SECS as u64 * 1_000_000_000,
            tcp: None,
            rtp_socket: None,
            rtcp_socket: None,
            clients: Vec::new(),
            started: false,
            configured: false,
            frames_sent: 0,
            packets_sent: 0,
            sender_reports_sent: 0,
            eos_seen: false,
        }
    }

    /// Use an already-bound listener (so a test can pick an ephemeral port).
    pub fn from_listener(listener: StdTcpListener) -> Result<Self, G2gError> {
        let addr = listener.local_addr().map_err(io_err)?;
        Ok(Self {
            listener: Some(listener),
            configured: true,
            ..Self::new(addr)
        })
    }

    /// Set the RTP payload type and SSRC carried in every packet.
    pub fn with_rtp(mut self, payload_type: u8, ssrc: u32) -> Self {
        self.payload_type = payload_type & 0x7F;
        self.ssrc = ssrc;
        self
    }

    /// RTCP sender-report interval (default 5 s).
    pub fn with_rtcp_sr_interval(mut self, interval: Duration) -> Self {
        self.sr_interval_ns = interval.as_nanos() as u64;
        self
    }

    /// Session timeout: a player silent past this on both the control channel
    /// and RTCP is reaped. Advertised (rounded up to whole seconds) in the
    /// SETUP response so clients know their keepalive cadence.
    pub fn with_session_timeout(mut self, timeout: Duration) -> Self {
        self.session_timeout_ns = (timeout.as_nanos() as u64).max(1);
        self
    }

    /// The timeout to advertise in `Session: id;timeout=N`, in whole seconds.
    /// With reaping disabled the RFC default is advertised, so clients keep a
    /// standard keepalive cadence.
    fn session_timeout_secs(&self) -> u32 {
        if self.session_timeout_ns == u64::MAX {
            return DEFAULT_SESSION_TIMEOUT_SECS;
        }
        self.session_timeout_ns.div_ceil(1_000_000_000).max(1) as u32
    }

    fn responder(&self, server_rtp_port: u16) -> RtspResponder {
        RtspResponder::new(sdp_h264(self.payload_type), server_rtp_port, self.ssrc)
            .with_session_timeout_secs(self.session_timeout_secs())
    }

    pub fn frames_sent(&self) -> u64 {
        self.frames_sent
    }

    pub fn packets_sent(&self) -> u64 {
        self.packets_sent
    }

    pub fn sender_reports_sent(&self) -> u64 {
        self.sender_reports_sent
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
        let (rtp_socket, rtcp_socket) = bind_rtp_pair().await?;
        let server_rtp_port = rtp_socket.local_addr().map_err(io_err)?.port();

        let (mut control, peer) = listener.accept().await.map_err(io_err)?;
        let mut responder = self.responder(server_rtp_port);
        let mut pending: Vec<u8> = Vec::new();
        let mut buf = [0u8; CTRL_BUF];
        let mut dest = None;
        let mut rtcp_dest = None;
        let mut interleaved = None;
        let mut rtcp_channel = None;
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
                        rtcp_dest = Some(SocketAddr::new(
                            peer.ip(),
                            client_rtp_port.saturating_add(1),
                        ));
                    }
                    RtspEvent::SetupInterleaved {
                        rtp_channel,
                        rtcp_channel: rtcp_ch,
                    } => {
                        interleaved = Some(rtp_channel);
                        rtcp_channel = Some(rtcp_ch);
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
        let mut client = Client::new(control, responder, peer.ip());
        client.pending = pending;
        client.dest = dest;
        client.rtcp_dest = rtcp_dest;
        client.interleaved = interleaved;
        client.rtcp_channel = rtcp_channel;
        client.packetizer = packetizer;
        self.clients.push(client);
        self.tcp = Some(listener);
        self.rtp_socket = Some(rtp_socket);
        self.rtcp_socket = Some(rtcp_socket);
        Ok(())
    }

    /// Accept every player whose TCP connection is already queued (non-blocking),
    /// adding each as a handshaking client.
    async fn accept_new(&mut self) {
        let (Some(listener), Some(rtp)) = (self.tcp.as_ref(), self.rtp_socket.as_ref()) else {
            return;
        };
        let Ok(server_rtp_port) = rtp.local_addr().map(|a| a.port()) else {
            return;
        };
        // A zero timeout polls accept once: take a queued connection or stop.
        while let Ok(Ok((control, peer))) =
            tokio::time::timeout(Duration::from_millis(0), listener.accept()).await
        {
            let responder = self.responder(server_rtp_port);
            self.clients
                .push(Client::new(control, responder, peer.ip()));
        }
    }

    /// Service every client's control channel (non-blocking), reaping any that
    /// disconnected, tore down, or fell silent past the session timeout.
    /// PLAYing clients are advanced too, so a mid-stream TEARDOWN or
    /// control-channel close is detected and keepalive requests are answered.
    async fn advance_handshakes(&mut self) {
        let (pt, ssrc, mp) = (self.payload_type, self.ssrc, self.max_payload);
        let now = g2g_core::metrics::monotonic_ns();
        let mut i = 0;
        while i < self.clients.len() {
            let keep = self.clients[i].advance(pt, ssrc, mp).await
                && now.saturating_sub(self.clients[i].last_activity_ns) <= self.session_timeout_ns;
            if keep {
                i += 1;
            } else {
                self.clients.swap_remove(i);
            }
        }
    }

    /// The RTCP pass: drain the server RTCP socket (a player's receiver report
    /// counts as keepalive activity), then send a sender report to every PLAYing
    /// player whose interval elapsed, reaping any whose send fails.
    async fn service_rtcp(&mut self) {
        let Some(rtcp_socket) = self.rtcp_socket.as_ref() else {
            return;
        };
        let now = g2g_core::metrics::monotonic_ns();
        let mut buf = [0u8; 1500];
        while let Ok((n, src)) = rtcp_socket.try_recv_from(&mut buf) {
            if !rtcp::is_rtcp(&buf[..n]) {
                continue;
            }
            // Attribute by the negotiated RTCP address, falling back to the
            // peer IP (a client may send from an unbound source port).
            let by_addr = self.clients.iter_mut().find(|c| c.rtcp_dest == Some(src));
            let stamped = match by_addr {
                Some(c) => Some(c),
                None => self.clients.iter_mut().find(|c| c.peer_ip == src.ip()),
            };
            if let Some(c) = stamped {
                c.last_activity_ns = now;
            }
        }
        let mut i = 0;
        while i < self.clients.len() {
            let c = &mut self.clients[i];
            if !c.playing() || now.saturating_sub(c.last_sr_ns) < self.sr_interval_ns {
                i += 1;
                continue;
            }
            let sr = rtcp::build_sender_report(
                self.ssrc,
                rtcp::ntp_now(),
                c.last_rtp_ts,
                c.rtp_packets,
                c.rtp_octets,
                &[],
            );
            match c.send_rtcp(rtcp_socket, &sr).await {
                Ok(sent) => {
                    if sent {
                        c.last_sr_ns = now;
                        self.sender_reports_sent += 1;
                    }
                    i += 1;
                }
                Err(()) => {
                    self.clients.swap_remove(i);
                }
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

/// Bind the server's RTP/RTCP UDP sockets on adjacent ports, so the SETUP
/// response's `server_port=N-(N+1)` pair is real on both halves. The ephemeral
/// RTP port's neighbor may be taken; retry with a fresh pair a few times.
async fn bind_rtp_pair() -> Result<(tokio::net::UdpSocket, tokio::net::UdpSocket), G2gError> {
    for _ in 0..16 {
        let rtp = tokio::net::UdpSocket::bind(("0.0.0.0", 0))
            .await
            .map_err(io_err)?;
        let port = rtp.local_addr().map_err(io_err)?.port();
        let Some(rtcp_port) = port.checked_add(1) else {
            continue;
        };
        if let Ok(rtcp) = tokio::net::UdpSocket::bind(("0.0.0.0", rtcp_port)).await {
            return Ok((rtp, rtcp));
        }
    }
    Err(G2gError::Hardware(HardwareError::Other))
}

impl AsyncElement for RtspServerSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&h264_any())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(h264_any()))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            } => {}
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

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new("address", PropKind::Str, "IP to listen for RTSP players on")
                .with_default("0.0.0.0"),
            PropertySpec::new("port", PropKind::Uint, "RTSP TCP port to listen on")
                .with_default("8554")
                .with_range("0", "65535"),
            PropertySpec::new(
                "payload-type",
                PropKind::Uint,
                "RTP payload type (96..=127)",
            )
            .with_default("96")
            .with_range("0", "127"),
            PropertySpec::new("ssrc", PropKind::Uint, "RTP synchronization source id"),
            PropertySpec::new(
                "rtcp-sr-interval",
                PropKind::Uint,
                "RTCP sender-report interval in milliseconds",
            )
            .with_default("5000")
            .with_range("1", "60000"),
            PropertySpec::new(
                "timeout",
                PropKind::Uint,
                "session timeout in seconds (a player silent past it is reaped; 0 = never)",
            )
            .with_default("60")
            .with_range("0", "604800"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if let Some(r) = crate::netprop::set_addr_prop(&mut self.rtsp_addr, "address", name, &value)
        {
            return r;
        }
        match name {
            "payload-type" => {
                let pt = value.as_uint().ok_or(PropError::Type)?;
                if pt > 127 {
                    return Err(PropError::Value);
                }
                self.payload_type = pt as u8;
                Ok(())
            }
            "ssrc" => {
                self.ssrc = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "rtcp-sr-interval" => {
                let ms = value.as_uint().ok_or(PropError::Type)?;
                if !(1..=60_000).contains(&ms) {
                    return Err(PropError::Value);
                }
                self.sr_interval_ns = ms * 1_000_000;
                Ok(())
            }
            "timeout" => {
                let secs = value.as_uint().ok_or(PropError::Type)?;
                if secs > 604_800 {
                    return Err(PropError::Value);
                }
                self.session_timeout_ns = match secs {
                    0 => u64::MAX, // never reap
                    s => s * 1_000_000_000,
                };
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if let Some(v) = crate::netprop::get_addr_prop(&self.rtsp_addr, "address", name) {
            return Some(v);
        }
        match name {
            "payload-type" => Some(PropValue::Uint(self.payload_type as u64)),
            "ssrc" => Some(PropValue::Uint(self.ssrc as u64)),
            "rtcp-sr-interval" => Some(PropValue::Uint(self.sr_interval_ns / 1_000_000)),
            "timeout" => Some(PropValue::Uint(match self.session_timeout_ns {
                u64::MAX => 0,
                _ => self.session_timeout_secs() as u64,
            })),
            _ => None,
        }
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
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
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
                    let bytes = slice;
                    self.broadcast(bytes, timestamp).await?;
                    self.frames_sent += 1;
                    self.service_rtcp().await;
                }
                // RTP has no in-band end marker: an RTCP BYE tells each player
                // the stream ended cleanly rather than stalling to a timeout.
                PipelinePacket::Eos => {
                    self.eos_seen = true;
                    if let Some(rtcp_socket) = self.rtcp_socket.as_ref() {
                        let bye = rtcp::build_bye(self.ssrc);
                        for c in &mut self.clients {
                            if c.playing() {
                                let _ = c.send_rtcp(rtcp_socket, &bye).await;
                            }
                        }
                    }
                }
                // Sequence numbers persist across a seek (loss is tracked by gaps).
                PipelinePacket::Flush => {}
                // Geometry refinement lives in the in-band SPS, not in RTP/SDP.
                PipelinePacket::CapsChanged(_) => {}
                PipelinePacket::Segment(_) => {}
                // future PipelinePacket variants: no-op (terminal sink).
                _ => {}
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
        let client = Client::new(
            server,
            RtspResponder::new(sdp_h264(96), 6000, 0x1234_5678),
            std::net::IpAddr::from([127, 0, 0, 1]),
        );
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

    #[tokio::test]
    async fn playing_client_is_reaped_when_control_channel_closes() {
        let (mut client, peer) = client_pair().await;
        client.dest = Some(SocketAddr::new(client.peer_ip, 5000));
        client.packetizer = Some(RtpH264Packetizer::new(96, 0x1234_5678));
        assert!(client.playing());
        drop(peer); // peer closes the control channel mid-stream
        let mut reaped = false;
        for _ in 0..10_000 {
            if !client.advance(96, 0x1234_5678, 1400).await {
                reaped = true;
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(
            reaped,
            "a playing client whose control channel closed is reaped"
        );
    }
}
