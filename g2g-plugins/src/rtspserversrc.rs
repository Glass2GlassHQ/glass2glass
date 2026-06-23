//! RTSP server source (RtspServerSrc, `rtsp-server` feature): hosts an RTSP
//! endpoint that a *publisher* pushes media into (the ANNOUNCE / SETUP / RECORD
//! direction, the inverse of [`RtspServerSink`](crate::rtspserversink) which
//! serves players). The sans-IO [`rtspserver::RtspResponder`](crate::rtspserver)
//! already speaks ANNOUNCE/RECORD; this element wraps it in the tokio TCP
//! control channel and the UDP RTP receive transport, reusing the depayloader
//! ([`RtpH264Depayloader`](crate::rtpdepay)) the UDP source uses.
//!
//! Shape: a contribution endpoint (e.g. an encoder/camera that publishes to an
//! RTSP server with `ffmpeg -f rtsp -rtsp_transport udp ...`). The TCP listener
//! is bound in `configure_pipeline`; `run` accepts one publisher, drives the
//! handshake to RECORD, then depayloads the RTP it receives into H.264 access
//! units emitted downstream.
//!
//! Scope: one publisher / one session, unicast UDP, in-order receive (localhost
//! / a clean link). A receive-side jitter buffer + RTCP, and multi-client, are
//! follow-ups; `UdpSrc` already has the jitter/RTCP machinery to lift in.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use std::net::{SocketAddr, TcpListener as StdTcpListener};

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, FrameTiming, G2gError,
    LatencyReport, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, Rate,
    VideoCodec,
};

use crate::filesink::io_err;
use crate::rtpdepay::RtpH264Depayloader;
use crate::rtspserver::{sdp_h264, RtspEvent, RtspRequest, RtspResponder};

/// H.264 RTP media clock (RFC 6184): 90 kHz.
const RTP_CLOCK_HZ: u64 = 90_000;
/// Default dynamic RTP payload type for H.264.
const DEFAULT_PAYLOAD_TYPE: u8 = 96;
/// Declared geometry hint (SPS is authoritative; a downstream decoder corrects).
const DEFAULT_WIDTH: u32 = 1280;
const DEFAULT_HEIGHT: u32 = 720;
const DEFAULT_FPS: u32 = 30;
/// TCP read buffer for RTSP control requests.
const CTRL_BUF: usize = 8192;
/// UDP receive buffer for RTP datagrams.
const RECV_BUF: usize = 65536;

#[derive(Debug)]
pub struct RtspServerSrc {
    rtsp_addr: SocketAddr,
    payload_type: u8,
    ssrc: u32,
    width: u32,
    height: u32,
    fps: u32,
    /// 0 means run until the connection drops / downstream shuts down; otherwise
    /// stop after this many access units and emit EOS (the test / bounded path).
    frame_limit: u64,
    listener: Option<StdTcpListener>,
    configured: bool,
}

impl RtspServerSrc {
    /// Listen for an RTSP publisher on `rtsp_addr` (e.g. `0.0.0.0:8554`).
    pub fn new(rtsp_addr: SocketAddr) -> Self {
        Self {
            rtsp_addr,
            payload_type: DEFAULT_PAYLOAD_TYPE,
            ssrc: 0,
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            fps: DEFAULT_FPS,
            frame_limit: 0,
            listener: None,
            configured: false,
        }
    }

    /// Use an already-bound listener (so a test can pick an ephemeral port).
    pub fn from_listener(listener: StdTcpListener) -> Result<Self, G2gError> {
        let addr = listener.local_addr().map_err(io_err)?;
        Ok(Self { listener: Some(listener), configured: true, ..Self::new(addr) })
    }

    /// Set the RTP payload type and SSRC negotiated in SETUP.
    pub fn with_rtp(mut self, payload_type: u8, ssrc: u32) -> Self {
        self.payload_type = payload_type & 0x7F;
        self.ssrc = ssrc;
        self
    }

    /// Declared output geometry (a negotiation hint; SPS is authoritative).
    pub fn with_video_size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Declared output frame rate (a negotiation hint).
    pub fn with_framerate(mut self, fps: u32) -> Self {
        self.fps = fps;
        self
    }

    /// Stop after `n` access units and emit EOS. Without this the source runs
    /// until the publisher disconnects (RTP has no in-band end marker).
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    /// The TCP control port actually bound, once a listener exists (ephemeral
    /// lookup for tests).
    pub fn local_port(&self) -> Option<u16> {
        self.listener.as_ref().and_then(|l| l.local_addr().ok()).map(|a| a.port())
    }

    fn caps(&self) -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: Rate::Fixed(self.fps << 16),
        }
    }

    /// Accept one publisher and drive the RTSP handshake to RECORD: bind the RTP
    /// UDP receive socket (its port is advertised in SETUP), run
    /// OPTIONS/ANNOUNCE/SETUP/RECORD over the TCP control channel, and return the
    /// bound UDP socket once the publisher has issued RECORD.
    async fn accept_and_handshake(&mut self) -> Result<tokio::net::UdpSocket, G2gError> {
        let std_listener = self.listener.take().ok_or(G2gError::NotConfigured)?;
        std_listener.set_nonblocking(true).map_err(io_err)?;
        let listener = tokio::net::TcpListener::from_std(std_listener).map_err(io_err)?;
        let (mut control, _peer) = listener.accept().await.map_err(io_err)?;

        // The UDP socket the publisher will push RTP to; its local port is
        // advertised in the SETUP response (server_port).
        let rtp_socket = tokio::net::UdpSocket::bind(("0.0.0.0", 0)).await.map_err(io_err)?;
        let server_rtp_port = rtp_socket.local_addr().map_err(io_err)?.port();

        let mut responder =
            RtspResponder::new(sdp_h264(self.payload_type), server_rtp_port, self.ssrc);
        let mut pending: Vec<u8> = Vec::new();
        let mut buf = [0u8; CTRL_BUF];

        loop {
            let n = control.read(&mut buf).await.map_err(io_err)?;
            if n == 0 {
                // Publisher closed before RECORD.
                return Err(G2gError::Hardware(g2g_core::HardwareError::Other));
            }
            pending.extend_from_slice(&buf[..n]);

            while let Some((req, consumed)) = RtspRequest::parse(&pending) {
                pending.drain(..consumed);
                let (response, event) = responder.handle_request(&req);
                control.write_all(&response).await.map_err(io_err)?;
                match event {
                    RtspEvent::Record => {
                        // Keep the control channel alive for the session (a
                        // TEARDOWN watch is a follow-up); media now arrives on
                        // the RTP socket.
                        return Ok(rtp_socket);
                    }
                    RtspEvent::Teardown => return Err(G2gError::Shutdown),
                    _ => {}
                }
            }
        }
    }
}

impl SourceLoop for RtspServerSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.caps()))
    }

    /// Produces the declared H.264 hint caps (no I/O at negotiation; the TCP
    /// listener binds in `configure_pipeline`). A downstream decoder corrects the
    /// real geometry from the in-band SPS via a mid-stream `CapsChanged`.
    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(self.caps()))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if self.listener.is_none() {
            self.listener = Some(StdTcpListener::bind(self.rtsp_addr).map_err(io_err)?);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "RTSP server source",
            "Source/Network",
            "Hosts an RTSP endpoint a publisher pushes H.264 into (ANNOUNCE/RECORD)",
            "g2g",
        )
    }

    /// Live source: contributes one frame period so the sink keeps a frame in
    /// hand and never runs dry waiting on the network.
    fn latency(&self) -> LatencyReport {
        let period_ns = if self.fps > 0 { 1_000_000_000 / self.fps as u64 } else { 0 };
        LatencyReport::live(period_ns, None)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let socket = self.accept_and_handshake().await?;

            let mut depay = RtpH264Depayloader::new();
            let mut buf = vec![0u8; RECV_BUF];
            let limit = self.frame_limit;
            let mut seq = 0u64;
            // RTP timestamps start at a random offset; rebase to near zero.
            let mut ts_base: Option<u32> = None;

            loop {
                let (n, _src) = socket.recv_from(&mut buf).await.map_err(io_err)?;
                // In-order receive (one publisher, clean link); a reorder buffer
                // is a follow-up. The marker bit closes each access unit.
                let Some(au) = depay.depacketize(&buf[..n]) else {
                    continue;
                };
                let base = *ts_base.get_or_insert(au.rtp_timestamp);
                let rel = au.rtp_timestamp.wrapping_sub(base) as u64;
                let pts = rel * 1_000_000_000 / RTP_CLOCK_HZ;
                let arrival_ns = g2g_core::metrics::monotonic_ns();
                // IDR NAL => keyframe; false (the safe default) otherwise.
                let keyframe = crate::h264util::h264_au_is_keyframe(&au.data);
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(au.data.into_boxed_slice())),
                    timing: FrameTiming {
                        pts_ns: pts,
                        dts_ns: pts,
                        duration_ns: 0,
                        capture_ns: pts,
                        arrival_ns,
                        keyframe,
                    },
                    sequence: seq,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
                seq += 1;
                if limit != 0 && seq >= limit {
                    out.push(PipelinePacket::Eos).await?;
                    return Ok(seq);
                }
            }
        })
    }
}

impl PadTemplates for RtspServerSrc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::source(CapsSet::one(Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }))])
    }
}
