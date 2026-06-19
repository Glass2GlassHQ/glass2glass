//! UDP ingress source for H.264 over RTP (M91): the receive-side inverse of
//! [`UdpSink`](crate::udpsink). It binds a UDP socket, receives RTP packets,
//! and depayloads them (via [`rtpdepay`](crate::rtpdepay)) into Annex-B access
//! units pushed downstream as `CompressedVideo` H.264, ready for a decoder.
//!
//! This is **raw RTP** (no RTSP/SDP). There is no out-of-band stream
//! description, so the output geometry is a declared hint (`with_video_size` /
//! `with_framerate`, default 1280x720@30): H.264 carries its real dimensions
//! in-band in the SPS, and a downstream decoder re-derives and corrects them.
//! A receive-side jitter buffer (reorder / loss concealment / RTCP) and
//! SDP/SPS-driven caps discovery are the larger follow-ups (DESIGN_TODO).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;

use std::net::{SocketAddr, UdpSocket as StdUdpSocket};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, ConfigureOutcome, Dim, FrameTiming, G2gError, LatencyReport, MemoryDomain, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, Rate, VideoCodec,
};

use crate::filesink::io_err;
use crate::rtpdepay::RtpH264Depayloader;

/// H.264 RTP media clock (RFC 6184): timestamps tick at 90 kHz.
const RTP_CLOCK_HZ: u64 = 90_000;
/// Receive buffer: a UDP datagram tops out at 65507 payload bytes.
const RECV_BUF: usize = 65_536;
const DEFAULT_WIDTH: u32 = 1280;
const DEFAULT_HEIGHT: u32 = 720;
const DEFAULT_FPS: u32 = 30;

#[derive(Debug)]
pub struct UdpSrc {
    bind: SocketAddr,
    width: u32,
    height: u32,
    fps: u32,
    /// 0 means run until error / downstream shutdown; otherwise stop after this
    /// many access units and emit EOS (the test / bounded path).
    frame_limit: u64,
    /// Bound synchronously in `configure_pipeline` (or supplied pre-bound via
    /// `from_socket`); promoted to a tokio socket in `run`, where a runtime
    /// context is guaranteed.
    std_socket: Option<StdUdpSocket>,
    configured: bool,
}

impl UdpSrc {
    /// Receive RTP on `bind` (e.g. `0.0.0.0:5004`).
    pub fn new(bind: SocketAddr) -> Self {
        Self {
            bind,
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            fps: DEFAULT_FPS,
            frame_limit: 0,
            std_socket: None,
            configured: false,
        }
    }

    /// Use an already-bound socket instead of binding `bind` ourselves. Lets a
    /// caller (e.g. a test) pick an ephemeral port and learn it up front.
    pub fn from_socket(socket: StdUdpSocket) -> Result<Self, G2gError> {
        let bind = socket.local_addr().map_err(io_err)?;
        socket.set_nonblocking(true).map_err(io_err)?;
        Ok(Self {
            std_socket: Some(socket),
            ..Self::new(bind)
        })
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
    /// until a socket error (RTP has no in-band end marker).
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    /// The port actually bound, once a socket exists (ephemeral-port lookup).
    pub fn local_port(&self) -> Option<u16> {
        self.std_socket
            .as_ref()
            .and_then(|s| s.local_addr().ok())
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
}

impl SourceLoop for UdpSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.caps()))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if self.std_socket.is_none() {
            let socket = StdUdpSocket::bind(self.bind).map_err(io_err)?;
            socket.set_nonblocking(true).map_err(io_err)?;
            self.std_socket = Some(socket);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
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
            let std = self.std_socket.take().ok_or(G2gError::NotConfigured)?;
            let socket = tokio::net::UdpSocket::from_std(std).map_err(io_err)?;

            let mut depay = RtpH264Depayloader::new();
            let mut buf = vec![0u8; RECV_BUF];
            let limit = self.frame_limit;
            let mut seq = 0u64;
            // RTP timestamps start at a random offset; rebase so downstream
            // sees PTS near zero.
            let mut ts_base: Option<u32> = None;

            while limit == 0 || seq < limit {
                let (n, _from) = socket.recv_from(&mut buf).await.map_err(io_err)?;
                let Some(au) = depay.depacketize(&buf[..n]) else {
                    continue;
                };

                let base = *ts_base.get_or_insert(au.rtp_timestamp);
                let rel = au.rtp_timestamp.wrapping_sub(base) as u64;
                let pts = rel * 1_000_000_000 / RTP_CLOCK_HZ;
                let arrival_ns = g2g_core::metrics::monotonic_ns();
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(au.data.into_boxed_slice())),
                    timing: FrameTiming {
                        pts_ns: pts,
                        dts_ns: pts,
                        duration_ns: 0,
                        capture_ns: pts,
                        arrival_ns,
                    },
                    sequence: seq,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
                seq += 1;
            }

            out.push(PipelinePacket::Eos).await?;
            Ok(seq)
        })
    }
}

impl PadTemplates for UdpSrc {
    /// Produces H.264 at any geometry; an instance fixes the declared hint.
    fn pad_templates() -> alloc::vec::Vec<PadTemplate> {
        alloc::vec::Vec::from([PadTemplate::source(g2g_core::CapsSet::one(
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
            },
        ))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builders_set_hint_and_limit() {
        let src = UdpSrc::new("127.0.0.1:5004".parse().unwrap())
            .with_video_size(640, 480)
            .with_framerate(25)
            .with_frame_limit(5);
        assert_eq!((src.width, src.height, src.fps), (640, 480, 25));
        assert_eq!(src.frame_limit, 5);
        assert!(matches!(src.caps(), Caps::CompressedVideo { codec: VideoCodec::H264, .. }));
    }

    #[test]
    fn from_socket_adopts_the_bound_port() {
        let sock = StdUdpSocket::bind("127.0.0.1:0").unwrap();
        let port = sock.local_addr().unwrap().port();
        let src = UdpSrc::from_socket(sock).unwrap();
        assert_eq!(src.local_port(), Some(port), "adopts the pre-bound port");
    }

    #[tokio::test]
    async fn run_before_configure_is_not_configured() {
        // Drive run() directly with a throwaway sink to assert the guard fires
        // before any socket work.
        struct NullSink;
        impl OutputSink for NullSink {
            fn push<'a>(
                &'a mut self,
                _p: PipelinePacket,
            ) -> Pin<Box<dyn Future<Output = Result<g2g_core::PushOutcome, G2gError>> + 'a>> {
                Box::pin(async { Ok(g2g_core::PushOutcome::Accepted) })
            }
        }
        let mut src = UdpSrc::new("127.0.0.1:0".parse().unwrap());
        let mut sink = NullSink;
        let res = src.run(&mut sink).await;
        assert_eq!(res, Err(G2gError::NotConfigured));
    }
}
