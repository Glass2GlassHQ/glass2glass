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
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, FrameTiming, G2gError,
    LatencyReport, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, Rate,
    VideoCodec,
};

use crate::filesink::io_err;
use crate::rtcp::{self, ReceptionStats, RtcpPacket};
use crate::rtpdepay::RtpH264Depayloader;
use crate::rtpjitter::{JitterConfig, RtpJitterBuffer};
use crate::rtx;
use crate::ulpfec::FecDecoder;

/// H.264 RTP media clock (RFC 6184): timestamps tick at 90 kHz.
const RTP_CLOCK_HZ: u64 = 90_000;
/// Receive buffer: a UDP datagram tops out at 65507 payload bytes.
const RECV_BUF: usize = 65_536;
const DEFAULT_WIDTH: u32 = 1280;
const DEFAULT_HEIGHT: u32 = 720;
const DEFAULT_FPS: u32 = 30;
/// Default receiver-report cadence.
const DEFAULT_RR_INTERVAL_MS: u64 = 1000;
/// Minimum spacing between NACK feedback packets, so a persistent gap is not
/// re-requested every datagram (give a retransmit time to arrive).
const NACK_MIN_INTERVAL_NS: u64 = 20_000_000;
/// Our own SSRC as the RTCP reporter (`g2g` + 1).
const LOCAL_SSRC: u32 = 0x6732_6701;

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
    /// fed to the FEC decoder, which recovers single per-group media losses.
    fec_pt: Option<u8>,
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
            let mut jitter = RtpJitterBuffer::new(self.jitter);
            let mut stats = ReceptionStats::new(0, RTP_CLOCK_HZ as u32);
            let mut buf = vec![0u8; RECV_BUF];
            let limit = self.frame_limit;
            let mut seq = 0u64;
            // RTP timestamps start at a random offset; rebase so downstream
            // sees PTS near zero.
            let mut ts_base: Option<u32> = None;
            // The original media stream's SSRC, learned from the first non-RTX
            // packet; an RFC 4588 RTX resend is restamped back onto it.
            let mut media_ssrc_seen: Option<u32> = None;
            // ULPFEC decoder; only consulted when a FEC payload type is set.
            let mut fec_decoder = FecDecoder::default();

            // RTCP feedback state (RTP/RTCP-muxed on this socket): the peer to
            // report to (learned from the first datagram) and report timers.
            let rtcp_on = self.rtcp_rr_interval_ms > 0;
            let rr_interval_ns = self.rtcp_rr_interval_ms * 1_000_000;
            let mut peer: Option<SocketAddr> = None;
            let mut last_rr_ns = g2g_core::metrics::monotonic_ns();
            let mut last_nack_ns = 0u64;

            loop {
                // Drain every packet the jitter buffer will release now, turning
                // each completed access unit into a downstream frame.
                while let Some(packet) = jitter.pop(g2g_core::metrics::monotonic_ns()) {
                    let Some(au) = depay.depacketize(&packet) else {
                        continue;
                    };
                    let base = *ts_base.get_or_insert(au.rtp_timestamp);
                    let rel = au.rtp_timestamp.wrapping_sub(base) as u64;
                    let pts = rel * 1_000_000_000 / RTP_CLOCK_HZ;
                    let arrival_ns = g2g_core::metrics::monotonic_ns();
                    // IDR NAL => keyframe; false (the safe default) otherwise.
                    let keyframe = crate::h264util::h264_au_is_keyframe(&au.data);
                    let frame = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(
                            au.data.into_boxed_slice(),
                        )),
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

                // Periodic receiver report back to the peer (best-effort).
                let now = g2g_core::metrics::monotonic_ns();
                if rtcp_on {
                    if let Some(dest) = peer {
                        if now.saturating_sub(last_rr_ns) >= rr_interval_ns {
                            let rr = rtcp::build_receiver_report(LOCAL_SSRC, &[stats.report_block(now)]);
                            let _ = socket.send_to(&rr, dest).await;
                            last_rr_ns = now;
                        }
                    }
                }

                // Wait for the next datagram, but no longer than the soonest of
                // the jitter hold deadline (flush a gap) and the next report.
                let now = g2g_core::metrics::monotonic_ns();
                let jitter_deadline = jitter.next_deadline_ns(now);
                let rr_deadline = if rtcp_on && peer.is_some() {
                    Some(rr_interval_ns.saturating_sub(now.saturating_sub(last_rr_ns)))
                } else {
                    None
                };
                let timeout = match (jitter_deadline, rr_deadline) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                };

                let recv = socket.recv_from(&mut buf);
                let received = match timeout {
                    Some(delay) if delay > 0 => {
                        // Err (deadline elapsed, no packet) => None: loop to flush / report.
                        tokio::time::timeout(core::time::Duration::from_nanos(delay), recv)
                            .await
                            .ok()
                    }
                    Some(_) => Some(recv.await),
                    None => Some(recv.await),
                };

                let Some(r) = received else { continue };
                let (n, from) = r.map_err(io_err)?;
                let now = g2g_core::metrics::monotonic_ns();

                // RTCP (muxed) feedback from the sender: a sender report fills the
                // LSR/DLSR fields for round-trip estimation.
                if rtcp_on && rtcp::is_rtcp(&buf[..n]) {
                    for p in rtcp::parse_compound(&buf[..n]) {
                        if let RtcpPacket::SenderReport { ntp, .. } = p {
                            stats.on_sender_report(ntp, now);
                        }
                    }
                    continue;
                }

                // RTP media: account it for reception stats, buffer it for
                // reordering, and remember the peer for feedback.
                if n >= 12 {
                    // RFC 4588: a packet on the RTX payload type carries an
                    // original packet (sequence prepended); rebuild it onto the
                    // media stream's SSRC before reordering, so the resend simply
                    // fills its gap. Drop it if no original SSRC is known yet.
                    let pt = buf[1] & 0x7F;
                    // ULPFEC repair packet: feed the decoder and inject any
                    // recovered media into the jitter buffer (no NAK round trip).
                    if self.fec_pt == Some(pt) {
                        fec_decoder.push_fec(&buf[..n]);
                        for rec in fec_decoder.take_recovered() {
                            jitter.push(&rec, now);
                        }
                        continue;
                    }
                    let is_rtx = self.rtx.is_some_and(|(rtx_pt, _)| pt == rtx_pt);
                    let reconstructed = match (is_rtx, self.rtx, media_ssrc_seen) {
                        (true, Some((_, apt)), Some(ssrc)) => {
                            rtx::parse_rtx_packet(&buf[..n], apt, ssrc)
                        }
                        _ => None,
                    };
                    if is_rtx && reconstructed.is_none() {
                        continue; // unusable RTX packet (no media SSRC yet / malformed)
                    }
                    let pkt: &[u8] = reconstructed.as_deref().unwrap_or(&buf[..n]);

                    let media_ssrc = u32::from_be_bytes([pkt[8], pkt[9], pkt[10], pkt[11]]);
                    let pkt_seq = u16::from_be_bytes([pkt[2], pkt[3]]);
                    let rtp_ts = u32::from_be_bytes([pkt[4], pkt[5], pkt[6], pkt[7]]);
                    if !is_rtx {
                        media_ssrc_seen = Some(media_ssrc);
                    }
                    stats.on_rtp(media_ssrc, pkt_seq, rtp_ts, now);
                    peer = Some(from);
                    jitter.push(pkt, now);
                    // Record the packet for FEC and inject anything its arrival
                    // now lets a buffered repair packet recover.
                    if self.fec_pt.is_some() {
                        fec_decoder.push_media(pkt_seq, pkt);
                        for rec in fec_decoder.take_recovered() {
                            jitter.push(&rec, now);
                        }
                    }

                    // Request retransmission of any open gaps, rate-limited.
                    if self.nack_enabled && now.saturating_sub(last_nack_ns) >= NACK_MIN_INTERVAL_NS {
                        let missing = jitter.missing_seqs();
                        if !missing.is_empty() {
                            let nack = rtcp::build_nack(LOCAL_SSRC, media_ssrc, &missing);
                            let _ = socket.send_to(&nack, from).await;
                            last_nack_ns = now;
                        }
                    }
                }
            }
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
