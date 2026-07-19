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

use std::net::{SocketAddr, UdpSocket as StdUdpSocket};

use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError, LatencyReport,
    OutputSink, PadTemplate, PadTemplates, PropError, PropKind, PropValue, PropertySpec, Rate,
    VideoCodec,
};

use crate::filesink::io_err;
use crate::rtpjitter::JitterConfig;
use crate::rtprecv::{RtpRecvConfig, DEFAULT_RR_INTERVAL_MS};

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
    /// Reorder/loss-resilience policy for the receive path.
    jitter: JitterConfig,
    /// RTCP receiver-report cadence in ms (0 disables RTCP feedback entirely).
    rtcp_rr_interval_ms: u64,
    /// Emit RTPFB Generic NACK for detected gaps (requests retransmission).
    nack_enabled: bool,
    /// RFC 4588 RTX: when set, packets on this `(rtx payload type, apt)` are
    /// reconstructed to the original stream before reordering. `apt` is the
    /// associated (original) payload type the rebuilt packet is restamped with.
    rtx: Option<(u8, u8)>,
    /// RFC 5109 ULPFEC: when set, packets on this payload type are repair packets
    /// fed to the ULPFEC decoder, which recovers single per-group media losses.
    fec_pt: Option<u8>,
    /// RFC 8627 FlexFEC: when set, packets on this payload type are FlexFEC repair
    /// packets fed to the FlexFEC decoder (the wide-mask sibling of `fec_pt`).
    flexfec_pt: Option<u8>,
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
            jitter: JitterConfig::default(),
            rtcp_rr_interval_ms: DEFAULT_RR_INTERVAL_MS,
            nack_enabled: true,
            rtx: None,
            fec_pt: None,
            flexfec_pt: None,
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

    /// Tune the receive-side jitter buffer: hold a gap up to `max_hold_ms`
    /// before declaring it lost, and buffer at most `max_depth` packets. A
    /// `max_depth` of 0 disables reordering (in-order passthrough). Default is
    /// [`JitterConfig::default`] (50 ms / 64 packets).
    pub fn with_jitter(mut self, max_hold_ms: u64, max_depth: usize) -> Self {
        self.jitter = JitterConfig::new(max_hold_ms, max_depth);
        self
    }

    /// Configure RTCP feedback (RTP/RTCP-muxed on the same socket, RFC 5761):
    /// send a receiver report every `rr_interval_ms` (0 disables RTCP), and emit
    /// a Generic NACK for each detected gap when `nack` is set. Default is on
    /// (1 s reports, NACK enabled).
    pub fn with_rtcp(mut self, rr_interval_ms: u64, nack: bool) -> Self {
        self.rtcp_rr_interval_ms = rr_interval_ms;
        self.nack_enabled = nack;
        self
    }

    /// Reconstruct RFC 4588 RTX packets: those whose payload type is
    /// `rtx_payload_type` carry an original packet (sequence prepended) of
    /// payload type `apt`. The rebuilt original is fed to the jitter buffer like
    /// any other packet, so a retransmission fills its gap.
    pub fn with_rtx(mut self, rtx_payload_type: u8, apt: u8) -> Self {
        self.rtx = Some((rtx_payload_type & 0x7F, apt & 0x7F));
        self
    }

    /// Decode RFC 5109 ULPFEC repair packets (this payload type) and inject any
    /// recovered media into the jitter buffer, filling a single per-group loss
    /// with no retransmission round trip.
    pub fn with_fec(mut self, fec_payload_type: u8) -> Self {
        self.fec_pt = Some(fec_payload_type & 0x7F);
        self
    }

    /// Decode RFC 8627 FlexFEC repair packets (this payload type) and inject any
    /// recovered media into the jitter buffer. FlexFEC's wide mask protects more
    /// than ULPFEC's 16 packets per repair (the sender's `with_flexfec`).
    pub fn with_flexfec(mut self, fec_payload_type: u8) -> Self {
        self.flexfec_pt = Some(fec_payload_type & 0x7F);
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

    /// Produces the declared H.264 hint caps (no I/O at negotiation; the socket
    /// binds in `configure_pipeline`). A downstream decoder corrects the real
    /// geometry from the in-band SPS via a mid-stream `CapsChanged`.
    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(self.caps()))))
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

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "UDP RTP source",
            "Source/Network",
            "Receives raw RTP H.264 over UDP with a jitter buffer",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new(
                "address",
                PropKind::Str,
                "local bind address (IP to listen on)",
            )
            .with_default("0.0.0.0"),
            PropertySpec::new("port", PropKind::Uint, "local UDP port to receive on")
                .with_range("0", "65535"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        crate::netprop::set_addr_prop(&mut self.bind, "address", name, &value)
            .unwrap_or(Err(PropError::Unknown))
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        crate::netprop::get_addr_prop(&self.bind, "address", name)
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
            let std = self.std_socket.take().ok_or(G2gError::NotConfigured)?;
            let socket = tokio::net::UdpSocket::from_std(std).map_err(io_err)?;

            // The jitter + RTCP RR/NACK + FEC/RTX + depayload receive path is
            // shared with RtspServerSrc; assemble our tuning and drive it.
            let cfg = RtpRecvConfig {
                jitter: self.jitter,
                rtcp_rr_interval_ms: self.rtcp_rr_interval_ms,
                nack_enabled: self.nack_enabled,
                rtx: self.rtx,
                fec_pt: self.fec_pt,
                flexfec_pt: self.flexfec_pt,
            };
            crate::rtprecv::receive_rtp_h264(&socket, &cfg, self.frame_limit, 0, out).await
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
    use g2g_core::PipelinePacket;

    #[test]
    fn builders_set_hint_and_limit() {
        let src = UdpSrc::new("127.0.0.1:5004".parse().unwrap())
            .with_video_size(640, 480)
            .with_framerate(25)
            .with_frame_limit(5);
        assert_eq!((src.width, src.height, src.fps), (640, 480, 25));
        assert_eq!(src.frame_limit, 5);
        assert!(matches!(
            src.caps(),
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            }
        ));
    }

    #[test]
    fn from_socket_adopts_the_bound_port() {
        let sock = StdUdpSocket::bind("127.0.0.1:0").unwrap();
        let port = sock.local_addr().unwrap().port();
        let src = UdpSrc::from_socket(sock).unwrap();
        assert_eq!(src.local_port(), Some(port), "adopts the pre-bound port");
    }

    #[tokio::test]
    async fn caps_constraint_is_produces_declared_h264() {
        let mut src = UdpSrc::new("127.0.0.1:5004".parse().unwrap())
            .with_video_size(640, 480)
            .with_framerate(25);
        let expected = src.caps();
        match src.caps_constraint().await.unwrap() {
            CapsConstraint::Produces(set) => assert_eq!(set.alternatives(), &[expected]),
            other => panic!("expected Produces, got {other:?}"),
        };
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
            ) -> Pin<Box<dyn Future<Output = Result<g2g_core::PushOutcome, G2gError>> + 'a>>
            {
                Box::pin(async { Ok(g2g_core::PushOutcome::Accepted) })
            }
        }
        let mut src = UdpSrc::new("127.0.0.1:0".parse().unwrap());
        let mut sink = NullSink;
        let res = src.run(&mut sink).await;
        assert_eq!(res, Err(G2gError::NotConfigured));
    }
}
