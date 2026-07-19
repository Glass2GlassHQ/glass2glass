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
//! Transport: both unicast UDP (`RTP/AVP;client_port=`, the jitter/RTCP/FEC
//! receive path shared with `UdpSrc`) and TCP-interleaved (`RTP/AVP/TCP;
//! interleaved=`, RFC 2326 §10.12: RTP rides the control connection as `$`-framed
//! binary, in order, so no jitter buffer is needed), chosen by what the publisher
//! negotiates in SETUP. What `ffmpeg -rtsp_transport tcp` uses.
//!
//! Scope: one publisher / one session. Multi-client is a follow-up.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use std::net::{SocketAddr, TcpListener as StdTcpListener};

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError, LatencyReport,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, Rate, VideoCodec,
};

use crate::filesink::io_err;
use crate::rtpdepay::RtpH264Depayloader;
use crate::rtpjitter::JitterConfig;
use crate::rtprecv::{push_access_unit, RtpRecvConfig};
use crate::rtspserver::{sdp_h264, RtspEvent, RtspRequest, RtspResponder};

/// Default dynamic RTP payload type for H.264.
const DEFAULT_PAYLOAD_TYPE: u8 = 96;
/// Declared geometry hint (SPS is authoritative; a downstream decoder corrects).
const DEFAULT_WIDTH: u32 = 1280;
const DEFAULT_HEIGHT: u32 = 720;
const DEFAULT_FPS: u32 = 30;
/// TCP read buffer for RTSP control requests.
const CTRL_BUF: usize = 8192;

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
    /// Receive-path tuning (jitter reorder + optional RTCP/NACK), shared with
    /// [`UdpSrc`](crate::udpsrc) via [`crate::rtprecv`]. RTCP defaults off: a
    /// classic RTSP publisher puts RTCP on a separate port (not muxed onto the
    /// RTP socket), so receiver-report / NACK feedback needs `with_rtcp` plus a
    /// negotiated `rtcp-mux` (a follow-up) to actually reach the sender.
    recv: RtpRecvConfig,
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
            recv: RtpRecvConfig {
                jitter: JitterConfig::default(),
                // Off until a separate RTCP port / rtcp-mux is negotiated.
                rtcp_rr_interval_ms: 0,
                nack_enabled: false,
                rtx: None,
                fec_pt: None,
                flexfec_pt: None,
            },
            listener: None,
            configured: false,
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

    /// Tune the receive-side jitter buffer: hold a gap up to `max_hold_ms`
    /// before declaring it lost, buffering at most `max_depth` packets. A
    /// `max_depth` of 0 disables reordering (in-order passthrough). Default is
    /// [`JitterConfig::default`] (50 ms / 64 packets), so a lossy / reordering
    /// link is tolerated even without RTCP retransmission.
    pub fn with_jitter(mut self, max_hold_ms: u64, max_depth: usize) -> Self {
        self.recv.jitter = JitterConfig::new(max_hold_ms, max_depth);
        self
    }

    /// Enable RTCP receiver reports (every `rr_interval_ms`, 0 disables) and
    /// Generic NACK (when `nack`) on the RTP socket. Off by default: it is only
    /// useful once the publisher muxes RTCP onto the RTP port (RFC 5761), which
    /// a classic RTSP publisher does not do without a negotiated `rtcp-mux`.
    pub fn with_rtcp(mut self, rr_interval_ms: u64, nack: bool) -> Self {
        self.recv.rtcp_rr_interval_ms = rr_interval_ms;
        self.recv.nack_enabled = nack;
        self
    }

    /// The TCP control port actually bound, once a listener exists (ephemeral
    /// lookup for tests).
    pub fn local_port(&self) -> Option<u16> {
        self.listener
            .as_ref()
            .and_then(|l| l.local_addr().ok())
            .map(|a| a.port())
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
    /// UDP receive socket (its port is advertised in a UDP SETUP), run
    /// OPTIONS/ANNOUNCE/SETUP/RECORD over the TCP control channel, and return the
    /// negotiated [`RecordTransport`] once the publisher has issued RECORD. Either
    /// way the control stream is handed back (never dropped): a real RTSP
    /// publisher (ffmpeg) holds the control connection open while it records and
    /// treats the server closing it as a fatal "broken pipe", so dropping it would
    /// abort the publish. For TCP-interleaved, that same stream *is* the RTP
    /// transport, and any bytes already buffered past RECORD are handed on as
    /// `leftover` so a pipelined first frame is not lost.
    async fn accept_and_handshake(&mut self) -> Result<RecordTransport, G2gError> {
        let std_listener = self.listener.take().ok_or(G2gError::NotConfigured)?;
        std_listener.set_nonblocking(true).map_err(io_err)?;
        let listener = tokio::net::TcpListener::from_std(std_listener).map_err(io_err)?;
        let (mut control, _peer) = listener.accept().await.map_err(io_err)?;

        // The UDP socket the publisher will push RTP to (UDP transport); its local
        // port is advertised in the SETUP response (server_port). Unused, but
        // still bound, if the publisher instead picks TCP-interleaved.
        let rtp_socket = tokio::net::UdpSocket::bind(("0.0.0.0", 0))
            .await
            .map_err(io_err)?;
        let server_rtp_port = rtp_socket.local_addr().map_err(io_err)?.port();

        let mut responder =
            RtspResponder::new(sdp_h264(self.payload_type), server_rtp_port, self.ssrc);
        // Set when SETUP negotiated TCP-interleaved: the RTP channel to demux the
        // `$`-framed control-connection binary on.
        let mut interleaved_rtp_channel: Option<u8> = None;
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
                    RtspEvent::SetupInterleaved { rtp_channel, .. } => {
                        interleaved_rtp_channel = Some(rtp_channel);
                    }
                    RtspEvent::Record => {
                        // Media now flows; hand the control stream back so the
                        // caller keeps it open for the session (interleaved also
                        // receives its RTP on it). `pending` holds any bytes read
                        // past RECORD (a pipelined first interleaved frame).
                        return Ok(match interleaved_rtp_channel {
                            Some(rtp_channel) => RecordTransport::Interleaved {
                                control,
                                rtp_channel,
                                leftover: pending,
                            },
                            None => RecordTransport::Udp {
                                rtp_socket,
                                control,
                            },
                        });
                    }
                    RtspEvent::Teardown => return Err(G2gError::Shutdown),
                    _ => {}
                }
            }
        }
    }
}

/// The transport a publisher negotiated by RECORD time. Both keep the control
/// `TcpStream` (a UDP publisher needs it open; an interleaved one receives its RTP
/// on it).
#[derive(Debug)]
enum RecordTransport {
    /// Unicast UDP: RTP arrives on `rtp_socket`; `control` is held open.
    Udp {
        rtp_socket: tokio::net::UdpSocket,
        control: tokio::net::TcpStream,
    },
    /// TCP-interleaved (RFC 2326 §10.12): RTP arrives on `control` as `$`-framed
    /// binary on `rtp_channel`; `leftover` is any binary already buffered.
    Interleaved {
        control: tokio::net::TcpStream,
        rtp_channel: u8,
        leftover: Vec<u8>,
    },
}

/// One parsed item from an interleaved control stream (RFC 2326 §10.12).
#[derive(Debug, PartialEq, Eq)]
enum Interleaved {
    /// A `$`-framed binary packet on `channel`; its payload is `buf[start..end]`,
    /// and `consumed` bytes (header + payload) form the whole frame.
    Binary {
        channel: u8,
        start: usize,
        end: usize,
        consumed: usize,
    },
    /// An embedded RTSP request occupying `consumed` bytes (`teardown` if it was a
    /// TEARDOWN), interleaved between binary frames.
    Rtsp { teardown: bool, consumed: usize },
    /// Not enough bytes buffered yet for a complete item.
    NeedMore,
}

/// Parse the next interleaved item at the front of `buf`. A `$` (0x24) begins a
/// 4-byte binary header (`$`, channel, 2-byte big-endian length) then that many
/// payload bytes; anything else is an interleaved RTSP request.
fn next_interleaved(buf: &[u8]) -> Interleaved {
    match buf.first() {
        None => Interleaved::NeedMore,
        Some(&0x24) => {
            if buf.len() < 4 {
                return Interleaved::NeedMore;
            }
            let channel = buf[1];
            let len = u16::from_be_bytes([buf[2], buf[3]]) as usize;
            let consumed = 4 + len;
            if buf.len() < consumed {
                return Interleaved::NeedMore;
            }
            Interleaved::Binary {
                channel,
                start: 4,
                end: consumed,
                consumed,
            }
        }
        // Not a binary frame: an interleaved RTSP request (e.g. TEARDOWN).
        Some(_) => match RtspRequest::parse(buf) {
            Some((req, consumed)) => Interleaved::Rtsp {
                teardown: req.method == "TEARDOWN",
                consumed,
            },
            None => Interleaved::NeedMore,
        },
    }
}

/// Receive H.264 over a TCP-interleaved control stream (RFC 2326 §10.12): demux
/// `$`-framed RTP on `rtp_channel` and depayload it into access units. TCP is
/// ordered and lossless, so no jitter buffer / RTCP / FEC is needed (unlike the
/// UDP path); packets depayload straight through. Ends on the publisher closing
/// the connection, a TEARDOWN, or `frame_limit` access units. `leftover` is any
/// binary already buffered past RECORD.
async fn receive_interleaved(
    mut control: tokio::net::TcpStream,
    rtp_channel: u8,
    mut pending: Vec<u8>,
    frame_limit: u64,
    seq_base: u64,
    out: &mut dyn OutputSink,
) -> Result<u64, G2gError> {
    let mut depay = RtpH264Depayloader::new();
    let mut seq = seq_base;
    let mut ts_base: Option<u32> = None;
    let mut buf = [0u8; CTRL_BUF];
    loop {
        // Drain every complete interleaved item currently buffered.
        loop {
            match next_interleaved(&pending) {
                Interleaved::Binary {
                    channel,
                    start,
                    end,
                    consumed,
                } => {
                    // Depayload only the RTP channel (skip RTCP / other channels).
                    if channel == rtp_channel {
                        if let Some(au) = depay.depacketize(&pending[start..end]) {
                            if push_access_unit(
                                au,
                                &mut ts_base,
                                &mut seq,
                                seq_base,
                                frame_limit,
                                out,
                            )
                            .await?
                            {
                                return Ok(seq - seq_base);
                            }
                        }
                    }
                    pending.drain(..consumed);
                }
                Interleaved::Rtsp { teardown, consumed } => {
                    pending.drain(..consumed);
                    if teardown {
                        out.push(PipelinePacket::Eos).await?;
                        return Ok(seq - seq_base);
                    }
                }
                Interleaved::NeedMore => break,
            }
        }
        let n = control.read(&mut buf).await.map_err(io_err)?;
        if n == 0 {
            // Publisher closed the connection: end the stream (RTP has no in-band
            // end marker, so the close is the EOS signal).
            out.push(PipelinePacket::Eos).await?;
            return Ok(seq - seq_base);
        }
        pending.extend_from_slice(&buf[..n]);
    }
}

impl SourceLoop for RtspServerSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
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
        let period_ns = if self.fps > 0 {
            1_000_000_000 / self.fps as u64
        } else {
            0
        };
        LatencyReport::live(period_ns, None)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match self.accept_and_handshake().await? {
                // UDP: `_control` is held for the whole receive loop (a real RTSP
                // publisher aborts if the server closes it); the jitter + (optional)
                // RTCP + depayload path is shared with UdpSrc.
                RecordTransport::Udp {
                    rtp_socket,
                    control: _control,
                } => {
                    crate::rtprecv::receive_rtp_h264(
                        &rtp_socket,
                        &self.recv,
                        self.frame_limit,
                        0,
                        out,
                    )
                    .await
                }
                // TCP-interleaved: the control stream itself carries `$`-framed RTP.
                RecordTransport::Interleaved {
                    control,
                    rtp_channel,
                    leftover,
                } => {
                    receive_interleaved(control, rtp_channel, leftover, self.frame_limit, 0, out)
                        .await
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
