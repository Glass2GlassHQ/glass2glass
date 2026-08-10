//! UDP ingress source for H.264 over RTP (M91): the receive-side inverse of
//! [`UdpSink`](crate::udpsink). It binds a UDP socket, receives RTP packets,
//! and depayloads them (via [`rtpdepay`](crate::rtpdepay)) into Annex-B access
//! units pushed downstream as `CompressedVideo` H.264, ready for a decoder.
//!
//! Caps come from three places. The `sdp` property takes the description a
//! sender publishes (`ffmpeg -sdp_file`, an RTSP `DESCRIBE` body) and sets the
//! codec, geometry, frame rate and receive port from it ([`crate::sdp`]), so
//! negotiation runs on real values. The stream's own SPS then corrects whatever
//! is in force, as a `CapsChanged` before the frame it describes: it is
//! authoritative and covers the no-SDP case. The declared hint
//! (`with_video_size` / `with_framerate`, default 1280x720@30) is the fallback
//! until one of those lands, and stays in force for the fields neither supplies.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::{String, ToString};

use std::net::{SocketAddr, UdpSocket as StdUdpSocket};

use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError, LatencyReport,
    OutputSink, PadTemplate, PadTemplates, PropError, PropKind, PropValue, PropertySpec, Rate,
    VideoCodec,
};

use crate::filesink::io_err;
use crate::rtpjitter::JitterConfig;
use crate::rtprecv::RtpRecvConfig;
use crate::sdp::SdpMedia;

/// Resolve an `sdp` value: a document starts with its RFC 4566 `v=` version
/// line, anything else is a path to read it from.
fn read_sdp(value: &str) -> Result<String, G2gError> {
    let trimmed = value.trim_start();
    if trimmed.starts_with("v=") {
        return Ok(trimmed.to_string());
    }
    std::fs::read_to_string(value).map_err(io_err)
}

const DEFAULT_WIDTH: u32 = 1280;
const DEFAULT_HEIGHT: u32 = 720;
const DEFAULT_FPS: u32 = 30;

/// # Example
///
/// ```no_run
/// use g2g_plugins::udpsrc::UdpSrc;
///
/// let src = UdpSrc::new("0.0.0.0:5004".parse().unwrap())
///     .with_video_size(1920, 1080)
///     .with_framerate(30);
/// ```
#[derive(Debug)]
pub struct UdpSrc {
    bind: SocketAddr,
    width: u32,
    height: u32,
    fps: u32,
    /// 0 means run until error / downstream shutdown; otherwise stop after this
    /// many access units and emit EOS (the test / bounded path).
    frame_limit: u64,
    /// Receive-path tuning (jitter reorder, RTCP RR/NACK, RTX, ULPFEC/FlexFEC),
    /// the shared-path config handed to [`crate::rtprecv::receive_rtp_h264`].
    recv: RtpRecvConfig,
    /// The `sdp` property as set (inline text or a file path), kept only so the
    /// property reads back; its effect is already folded into the fields above.
    sdp: Option<String>,
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
            recv: RtpRecvConfig::default(),
            sdp: None,
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

    /// Configure from an SDP: either the document text or the path to a `.sdp`
    /// file. The first H.264 media section sets the output geometry / frame rate
    /// (from its `fmtp` parameter sets and `a=framerate`) and the receive port,
    /// so a published description is all a receiver needs. Fields the SDP leaves
    /// out keep their declared values.
    pub fn with_sdp(mut self, sdp: &str) -> Result<Self, G2gError> {
        let text = read_sdp(sdp)?;
        let media = SdpMedia::parse(&text).ok_or(G2gError::CapsMismatch)?;
        if !self.apply_sdp(&media) {
            return Err(G2gError::CapsMismatch);
        }
        self.sdp = Some(sdp.to_string());
        Ok(self)
    }

    /// Apply one parsed media description. Returns `false` (changing nothing)
    /// for a description this source cannot receive: it depayloads H.264 only.
    pub fn apply_sdp(&mut self, media: &SdpMedia) -> bool {
        let Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width,
            height,
            framerate,
        } = &media.caps
        else {
            return false;
        };
        if let (Dim::Fixed(w), Dim::Fixed(h)) = (width, height) {
            self.width = *w;
            self.height = *h;
        }
        if let Rate::Fixed(q16) = framerate {
            self.fps = (q16 >> 16).max(1);
        }
        // The m= port is where the sender is sending; 0 means the SDP pins none.
        if media.port != 0 {
            self.bind.set_port(media.port);
        }
        true
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
        self.recv.jitter = JitterConfig::new(max_hold_ms, max_depth);
        self
    }

    /// Configure RTCP feedback (RTP/RTCP-muxed on the same socket, RFC 5761):
    /// send a receiver report every `rr_interval_ms` (0 disables RTCP), and emit
    /// a Generic NACK for each detected gap when `nack` is set. Default is on
    /// (1 s reports, NACK enabled).
    pub fn with_rtcp(mut self, rr_interval_ms: u64, nack: bool) -> Self {
        self.recv.rtcp_rr_interval_ms = rr_interval_ms;
        self.recv.nack_enabled = nack;
        self
    }

    /// Reconstruct RFC 4588 RTX packets: those whose payload type is
    /// `rtx_payload_type` carry an original packet (sequence prepended) of
    /// payload type `apt`. The rebuilt original is fed to the jitter buffer like
    /// any other packet, so a retransmission fills its gap.
    pub fn with_rtx(mut self, rtx_payload_type: u8, apt: u8) -> Self {
        self.recv.rtx = Some((rtx_payload_type & 0x7F, apt & 0x7F));
        self
    }

    /// Decode RFC 5109 ULPFEC repair packets (this payload type) and inject any
    /// recovered media into the jitter buffer, filling a single per-group loss
    /// with no retransmission round trip.
    pub fn with_fec(mut self, fec_payload_type: u8) -> Self {
        self.recv.fec_pt = Some(fec_payload_type & 0x7F);
        self
    }

    /// Decode RFC 8627 FlexFEC repair packets (this payload type) and inject any
    /// recovered media into the jitter buffer. FlexFEC's wide mask protects more
    /// than ULPFEC's 16 packets per repair (the sender's `with_flexfec`).
    pub fn with_flexfec(mut self, fec_payload_type: u8) -> Self {
        self.recv.flexfec_pt = Some(fec_payload_type & 0x7F);
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
            "Receives raw RTP H.264 over UDP with a jitter buffer, caps from an SDP or the stream's SPS",
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
            PropertySpec::new(
                "width",
                PropKind::Uint,
                "declared frame width hint (the in-band SPS corrects it)",
            )
            .with_default("1280"),
            PropertySpec::new(
                "height",
                PropKind::Uint,
                "declared frame height hint (the in-band SPS corrects it)",
            )
            .with_default("720"),
            PropertySpec::new("framerate", PropKind::Uint, "declared frame rate hint, fps")
                .with_default("30"),
            PropertySpec::new(
                "sdp",
                PropKind::Str,
                "SDP describing the stream (document text or a .sdp file path): sets geometry, frame rate, and port",
            ),
            PropertySpec::new(
                "num-buffers",
                PropKind::Int,
                "access units to emit then EOS (-1 = until error/shutdown)",
            )
            .with_default("-1")
            .with_range("-1", "9223372036854775807"),
            PropertySpec::new(
                "jitter-latency",
                PropKind::Uint,
                "max time to hold a sequence gap before declaring it lost, milliseconds",
            )
            .with_default("50"),
            PropertySpec::new(
                "jitter-depth",
                PropKind::Uint,
                "max packets buffered for reorder (0 = in-order passthrough)",
            )
            .with_default("64"),
            PropertySpec::new(
                "rtcp-rr-interval",
                PropKind::Uint,
                "RTCP receiver-report interval in milliseconds (0 = RTCP off)",
            )
            .with_default("1000"),
            PropertySpec::new(
                "nack",
                PropKind::Bool,
                "request retransmission of detected gaps via RTPFB Generic NACK",
            )
            .with_default("true"),
            PropertySpec::new(
                "rtx-payload-type",
                PropKind::Uint,
                "RFC 4588 RTX stream payload type (0 = off; set rtx-apt too)",
            )
            .with_default("0")
            .with_range("0", "127"),
            PropertySpec::new(
                "rtx-apt",
                PropKind::Uint,
                "original (associated) payload type RTX packets rebuild to",
            )
            .with_default("0")
            .with_range("0", "127"),
            PropertySpec::new(
                "fec-payload-type",
                PropKind::Uint,
                "RFC 5109 ULPFEC repair-stream payload type (0 = off)",
            )
            .with_default("0")
            .with_range("0", "127"),
            PropertySpec::new(
                "flexfec-payload-type",
                PropKind::Uint,
                "RFC 8627 FlexFEC repair-stream payload type (0 = off)",
            )
            .with_default("0")
            .with_range("0", "127"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if let Some(r) = crate::netprop::set_addr_prop(&mut self.bind, "address", name, &value) {
            return r;
        }
        if let Some(r) = crate::rtprecv::set_recv_prop(&mut self.recv, name, &value) {
            return r;
        }
        match name {
            "width" => {
                self.width = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "height" => {
                self.height = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "framerate" => {
                self.fps = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "sdp" => {
                let raw = value.as_str().ok_or(PropError::Type)?;
                let text = read_sdp(raw).map_err(|_| PropError::Value)?;
                let media = SdpMedia::parse(&text).ok_or(PropError::Value)?;
                if !self.apply_sdp(&media) {
                    return Err(PropError::Value);
                }
                self.sdp = Some(raw.to_string());
                Ok(())
            }
            "num-buffers" => crate::netprop::set_frame_limit(&mut self.frame_limit, &value),
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if let Some(v) = crate::netprop::get_addr_prop(&self.bind, "address", name) {
            return Some(v);
        }
        if let Some(v) = crate::rtprecv::get_recv_prop(&self.recv, name) {
            return Some(v);
        }
        match name {
            "width" => Some(PropValue::Uint(self.width as u64)),
            "height" => Some(PropValue::Uint(self.height as u64)),
            "framerate" => Some(PropValue::Uint(self.fps as u64)),
            "sdp" => Some(PropValue::Str(self.sdp.clone().unwrap_or_default())),
            "num-buffers" => Some(crate::netprop::get_frame_limit(self.frame_limit)),
            _ => None,
        }
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
            // shared with RtspServerSrc; the caps in force ride along so the
            // stream's SPS can refine them mid-flight.
            let mut recv = self.recv.clone();
            recv.declared_caps = Some(self.caps());
            crate::rtprecv::receive_rtp_h264(&socket, &recv, self.frame_limit, 0, out).await
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
            fn poll_push(
                &mut self,
                _cx: &mut core::task::Context<'_>,
                packet_slot: &mut Option<PipelinePacket>,
            ) -> core::task::Poll<Result<g2g_core::PushOutcome, G2gError>> {
                packet_slot.take();
                core::task::Poll::Ready(Ok(g2g_core::PushOutcome::Accepted))
            }
        }
        let mut src = UdpSrc::new("127.0.0.1:0".parse().unwrap());
        let mut sink = NullSink;
        let res = src.run(&mut sink).await;
        assert_eq!(res, Err(G2gError::NotConfigured));
    }
}
