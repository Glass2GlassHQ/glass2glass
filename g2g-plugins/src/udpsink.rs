//! UDP egress sink for H.264 over RTP (M47). The I/O half of the live-egress
//! path whose Sans-IO half is `rtppay::RtpH264Packetizer` (M46): this sink
//! drives the packetizer over each Annex-B access unit and sends the resulting
//! RTP packets to a destination on a UDP socket, the send-side inverse of
//! `RtspSrc`'s receive path.
//!
//! The RTP timestamp is derived from `FrameTiming::pts_ns` at the 90 kHz H.264
//! media clock; sequence numbers and the per-AU marker bit come from the
//! packetizer. RTCP sender reports and the RTSP `ANNOUNCE`/`RECORD` handshake
//! for Wowza-style ingest are the remaining live-egress follow-ups (they need
//! the network port the sandbox blocks).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use std::net::{SocketAddr, UdpSocket as StdUdpSocket};

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, MemoryDomain,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, Rate, VideoCodec,
};

use crate::filesink::io_err;
use crate::rtppay::RtpH264Packetizer;

/// H.264 RTP media clock (RFC 6184): timestamps tick at 90 kHz.
const RTP_CLOCK_HZ: u64 = 90_000;
/// Default dynamic RTP payload type for H.264.
const DEFAULT_PAYLOAD_TYPE: u8 = 96;
/// Default max RTP payload bytes, leaving headroom under a 1500-byte MTU.
const DEFAULT_MAX_PAYLOAD: usize = 1400;

/// The H.264-at-any-geometry caps the sink accepts. Geometry rides in-band in
/// the SPS, so the sink imposes no concrete dimensions.
fn h264_any() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

#[derive(Debug)]
pub struct UdpSink {
    dest: SocketAddr,
    payload_type: u8,
    ssrc: u32,
    max_payload: usize,
    packetizer: Option<RtpH264Packetizer>,
    // Bound synchronously in `configure_pipeline` (no runtime needed); wrapped
    // into the tokio socket lazily on first `process`, where a runtime context
    // is guaranteed (`UdpSocket::from_std` requires one).
    std_socket: Option<StdUdpSocket>,
    socket: Option<tokio::net::UdpSocket>,
    packets_sent: u64,
    bytes_sent: u64,
    frames_sent: u64,
    eos_seen: bool,
}

impl UdpSink {
    pub fn new(dest: SocketAddr) -> Self {
        Self {
            dest,
            payload_type: DEFAULT_PAYLOAD_TYPE,
            ssrc: 0,
            max_payload: DEFAULT_MAX_PAYLOAD,
            packetizer: None,
            std_socket: None,
            socket: None,
            packets_sent: 0,
            bytes_sent: 0,
            frames_sent: 0,
            eos_seen: false,
        }
    }

    /// Set the RTP payload type (commonly 96..=127) and the synchronization
    /// source identifier carried in every packet.
    pub fn with_rtp(mut self, payload_type: u8, ssrc: u32) -> Self {
        self.payload_type = payload_type;
        self.ssrc = ssrc;
        self
    }

    /// Max RTP payload bytes per packet; larger NALs fragment into FU-A.
    pub fn with_max_payload(mut self, bytes: usize) -> Self {
        self.max_payload = bytes;
        self
    }

    pub fn packets_sent(&self) -> u64 {
        self.packets_sent
    }

    pub fn bytes_sent(&self) -> u64 {
        self.bytes_sent
    }

    pub fn frames_sent(&self) -> u64 {
        self.frames_sent
    }

    pub fn eos_seen(&self) -> bool {
        self.eos_seen
    }

    /// 90 kHz RTP timestamp for a presentation time. Wraps the u32 RTP field
    /// as the protocol expects.
    fn rtp_timestamp(pts_ns: u64) -> u32 {
        ((pts_ns as u128 * RTP_CLOCK_HZ as u128) / 1_000_000_000) as u32
    }

    fn ensure_socket(&mut self) -> Result<(), G2gError> {
        if self.socket.is_none() {
            let std = self.std_socket.take().ok_or(G2gError::NotConfigured)?;
            self.socket = Some(tokio::net::UdpSocket::from_std(std).map_err(io_err)?);
        }
        Ok(())
    }
}

impl AsyncElement for UdpSink {
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
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            } => {}
            _ => return Err(G2gError::CapsMismatch),
        }
        self.packetizer =
            Some(RtpH264Packetizer::new(self.payload_type, self.ssrc).with_max_payload(self.max_payload));
        let socket = StdUdpSocket::bind(("0.0.0.0", 0)).map_err(io_err)?;
        socket.set_nonblocking(true).map_err(io_err)?;
        socket.connect(self.dest).map_err(io_err)?;
        self.std_socket = Some(socket);
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let timestamp = Self::rtp_timestamp(frame.timing.pts_ns);
                    let packets = {
                        let packetizer = self.packetizer.as_mut().ok_or(G2gError::NotConfigured)?;
                        packetizer.packetize(slice.as_slice(), timestamp)
                    };
                    self.ensure_socket()?;
                    let socket = self.socket.as_ref().ok_or(G2gError::NotConfigured)?;
                    let mut sent = 0u64;
                    let mut bytes = 0u64;
                    for pkt in &packets {
                        socket.send(pkt).await.map_err(io_err)?;
                        sent += 1;
                        bytes += pkt.len() as u64;
                    }
                    self.packets_sent += sent;
                    self.bytes_sent += bytes;
                    self.frames_sent += 1;
                }
                // RTP carries no in-band end marker; an RTCP BYE is the M47
                // follow-up. Sequence numbers persist so a receiver sees clean
                // termination of the flow.
                PipelinePacket::Eos => self.eos_seen = true,
                // A seek does not reset the RTP sequence: a receiver tracks loss
                // by gaps, so the numbering continues across the flush.
                PipelinePacket::Flush => {}
                // Geometry refinement lives in the in-band SPS, not in RTP.
                PipelinePacket::CapsChanged(_) => {}
            }
            Ok(())
        })
    }
}

impl PadTemplates for UdpSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(CapsSet::one(h264_any()))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h264(w: u32, h: u32) -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(30 << 16),
        }
    }

    fn rgba(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: g2g_core::RawVideoFormat::Rgba8,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(30 << 16),
        }
    }

    fn sink() -> UdpSink {
        UdpSink::new("127.0.0.1:5004".parse().unwrap())
    }

    #[test]
    fn intercept_narrows_h264_and_rejects_raw() {
        let s = sink();
        assert!(s.intercept_caps(&h264(640, 480)).is_ok());
        assert_eq!(
            s.intercept_caps(&rgba(640, 480)),
            Err(G2gError::CapsMismatch),
            "an RTP H.264 packetizer cannot take raw video"
        );
    }

    #[test]
    fn configure_rejects_non_h264_before_binding() {
        let mut s = sink();
        let err = s
            .configure_pipeline(&rgba(640, 480))
            .expect_err("non-h264 caps must be rejected");
        assert_eq!(err, G2gError::CapsMismatch);
        assert!(s.std_socket.is_none(), "no socket bound on a rejected caps");
    }

    #[test]
    fn rtp_timestamp_is_90khz_of_pts() {
        assert_eq!(UdpSink::rtp_timestamp(0), 0);
        // 1 second of pts -> 90000 ticks.
        assert_eq!(UdpSink::rtp_timestamp(1_000_000_000), 90_000);
        // 1/30 s -> 3000 ticks.
        assert_eq!(UdpSink::rtp_timestamp(33_333_333), 2999);
    }
}
