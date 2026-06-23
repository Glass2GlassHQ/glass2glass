//! RTSP server sink (RtspServerSink, `rtsp-server` feature): hosts a pipeline's
//! H.264 as an RTSP endpoint and serves it as RTP/UDP to a connecting player
//! (the OBS / surveillance / contribution-server shape). The sans-IO
//! [`rtspserver::RtspResponder`](crate::rtspserver) does the protocol work
//! (OPTIONS / DESCRIBE / SETUP / PLAY); this element is the tokio TCP control
//! channel + the UDP RTP transport around it, reusing the
//! [`RtpH264Packetizer`](crate::rtppay) the UDP sink uses.
//!
//! Scope: one client / one session, unicast UDP, the PLAY (serving) direction.
//! The TCP listener is bound in `configure_pipeline`; on the first buffer a
//! client is accepted and the RTSP handshake is driven to PLAY before media
//! flows. ANNOUNCE/RECORD ingest (a publisher pushing in) is a follow-up; the
//! responder already speaks it.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use std::net::{SocketAddr, TcpListener as StdTcpListener};

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, Rate, VideoCodec,
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

/// H.264-at-any-geometry caps (geometry rides in-band in the SPS).
fn h264_any() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
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
    control: Option<tokio::net::TcpStream>,
    rtp_socket: Option<tokio::net::UdpSocket>,
    packetizer: Option<RtpH264Packetizer>,
    playing: bool,
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
            control: None,
            rtp_socket: None,
            packetizer: None,
            playing: false,
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

    pub fn eos_seen(&self) -> bool {
        self.eos_seen
    }

    /// 90 kHz RTP timestamp for a presentation time.
    fn rtp_timestamp(pts_ns: u64) -> u32 {
        ((pts_ns as u128 * RTP_CLOCK_HZ as u128) / 1_000_000_000) as u32
    }

    /// Accept one player and drive the RTSP handshake to PLAY: bind the RTP UDP
    /// socket, run OPTIONS/DESCRIBE/SETUP/PLAY over the TCP control channel, then
    /// connect the RTP socket to the client's negotiated address and arm the
    /// packetizer. Leaves `self.playing` true on success.
    async fn accept_and_handshake(&mut self) -> Result<(), G2gError> {
        let std_listener = self.listener.take().ok_or(G2gError::NotConfigured)?;
        std_listener.set_nonblocking(true).map_err(io_err)?;
        let listener = tokio::net::TcpListener::from_std(std_listener).map_err(io_err)?;
        let (mut control, peer) = listener.accept().await.map_err(io_err)?;

        // Our RTP send socket; its local port is advertised in SETUP.
        let rtp_socket = tokio::net::UdpSocket::bind(("0.0.0.0", 0)).await.map_err(io_err)?;
        let server_rtp_port = rtp_socket.local_addr().map_err(io_err)?.port();

        let mut responder = RtspResponder::new(sdp_h264(self.payload_type), server_rtp_port, self.ssrc);
        let mut pending: Vec<u8> = Vec::new();
        let mut buf = [0u8; CTRL_BUF];
        let mut client_rtp_port = None;

        loop {
            let n = control.read(&mut buf).await.map_err(io_err)?;
            if n == 0 {
                return Err(G2gError::Hardware(g2g_core::HardwareError::Other)); // closed before PLAY
            }
            pending.extend_from_slice(&buf[..n]);

            // Drain every complete request the buffer now holds.
            while let Some((req, consumed)) = RtspRequest::parse(&pending) {
                pending.drain(..consumed);
                let (response, event) = responder.handle_request(&req);
                control.write_all(&response).await.map_err(io_err)?;
                match event {
                    RtspEvent::Setup { client_rtp_port: port } => client_rtp_port = Some(port),
                    RtspEvent::Play => {
                        let port = client_rtp_port.ok_or(G2gError::NotConfigured)?;
                        let dest = SocketAddr::new(peer.ip(), port);
                        rtp_socket.connect(dest).await.map_err(io_err)?;
                        self.rtp_socket = Some(rtp_socket);
                        self.control = Some(control);
                        self.packetizer = Some(
                            RtpH264Packetizer::new(self.payload_type, self.ssrc)
                                .with_max_payload(self.max_payload),
                        );
                        self.playing = true;
                        return Ok(());
                    }
                    RtspEvent::Teardown => {
                        return Err(G2gError::Shutdown); // client gave up before PLAY
                    }
                    _ => {}
                }
            }
        }
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
        if self.listener.is_none() {
            self.listener = Some(StdTcpListener::bind(self.rtsp_addr).map_err(io_err)?);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "RTSP server sink",
            "Sink/Network",
            "Hosts an RTSP endpoint serving H.264 over RTP/UDP",
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
                    // Block on the first buffer until a player connects and PLAYs.
                    if !self.playing {
                        self.accept_and_handshake().await?;
                    }
                    let timestamp = Self::rtp_timestamp(frame.timing.pts_ns);
                    let packets = self
                        .packetizer
                        .as_mut()
                        .ok_or(G2gError::NotConfigured)?
                        .packetize(slice.as_slice(), timestamp);
                    let socket = self.rtp_socket.as_ref().ok_or(G2gError::NotConfigured)?;
                    for pkt in &packets {
                        socket.send(pkt).await.map_err(io_err)?;
                        self.packets_sent += 1;
                    }
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
