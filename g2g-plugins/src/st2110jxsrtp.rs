//! ST 2110-22 JPEG XS network elements (M604): a sink and source that put the
//! sans-IO `st2110jxs` core (RFC 9134) on the wire over UDP, with RTP timestamps
//! from the PTP media clock.
//!
//! [`St2110JxsSink`] takes `CompressedVideo{JpegXs}` `DataFrame`s (one JPEG XS
//! codestream per frame, from `FfmpegJpegXsEnc`), maps each frame's PTS through the
//! elected clock to a PTP/TAI sampling instant, slices the codestream into RFC 9134
//! codestream-mode packets, and sends the RTP packets to a UDP destination.
//! [`St2110JxsSrc`] binds a UDP socket, reassembles received -22 packets into whole
//! codestreams (the geometry from its properties / SDP, since -22 carries it out of
//! band), reconstructs each frame's PTS from the RTP timestamp, and emits
//! `CompressedVideo{JpegXs}` frames for a JPEG XS decoder. Because the timestamp is
//! the shared 90 kHz media clock, a receiver on the same grandmaster stays
//! frame-locked to the source.
//!
//! The codestream is opaque to this layer (the codec is `ffmpegjpegxs`); this is the
//! -20 sibling for compressed mezzanine video. The blocking recv in the source runs
//! with a read timeout so it stays cooperative and ends cleanly on a gap.

use core::future::{ready, Future, Ready};
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use std::net::UdpSocket;
use std::time::Duration;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ClockSync, ConfigureOutcome, Dim, ElementMetadata,
    FrameTiming, G2gError, MediaClock, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate, VideoCodec,
};

use crate::capsfilter::parse_raw_format;
use crate::filesink::io_err;
use crate::st2110jxs::{St2110JxsDepacketizer, St2110JxsPacketizer};
use crate::st2110pacing::{frame_period_ns, pace_send, Pacer, PacingProfile};
use crate::st2110sdp::{St2110Essence, St2110Sdp};
use crate::st2110video::Sampling;

/// The whole-fps rate from a `Caps::CompressedVideo` framerate (`Rate::Fixed` is
/// fps << 16), or 0 if not fixed (do not pace).
fn caps_fps(caps: &Caps) -> u32 {
    match caps {
        Caps::CompressedVideo { framerate: Rate::Fixed(q), .. } => q >> 16,
        _ => 0,
    }
}

/// Default reassembly-buffer ceiling for one codestream (never trust the stream): a
/// generous 4 MiB, far above a JPEG XS HD frame at typical mezzanine bitrates.
const DEFAULT_MAX_FRAME_BYTES: usize = 4 << 20;

/// Extract the geometry from an absolute `Caps::CompressedVideo{JpegXs}`:
/// returns (width, height). Errors on a non-JPEG-XS caps or a non-fixed dimension.
fn jxs_geometry(caps: &Caps) -> Result<(u32, u32), G2gError> {
    match caps {
        Caps::CompressedVideo {
            codec: VideoCodec::JpegXs,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        } => Ok((*w, *h)),
        _ => Err(G2gError::CapsMismatch),
    }
}

/// Map a sampling-name property (raw-format token) to the descriptive [`Sampling`]
/// the -22 SDP advertises; `None` if the token has no -20-style sampling.
fn parse_sampling(s: &str) -> Option<Sampling> {
    Sampling::from_format(parse_raw_format(s)?)
}

// ================================================================
// Sink
// ================================================================

/// ST 2110-22 JPEG XS sink: `CompressedVideo{JpegXs}` `DataFrame`s -> RFC 9134 RTP
/// over UDP.
pub struct St2110JxsSink {
    host: String,
    port: u16,
    payload_type: u8,
    ssrc: u32,
    max_packet: usize,
    width: u32,
    height: u32,
    /// Descriptive sampling for the SDP `fmtp` (the wire payload is compressed, so
    /// this is informative, not a pixel layout). Defaults to the broadcast norm.
    sampling: Sampling,
    packetizer: Option<St2110JxsPacketizer>,
    socket: Option<UdpSocket>,
    clock_sync: Option<ClockSync>,
    /// ST 2110-21 sender pacing profile (None = burst all packets, the default).
    pacing: Option<PacingProfile>,
    /// Frame period in ns (from the negotiated framerate), for the -21 schedule.
    frame_period_ns: u64,
}

impl core::fmt::Debug for St2110JxsSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("St2110JxsSink")
            .field("host", &self.host)
            .field("port", &self.port)
            .field("width", &self.width)
            .field("height", &self.height)
            .finish()
    }
}

impl Default for St2110JxsSink {
    fn default() -> Self {
        Self::new()
    }
}

impl St2110JxsSink {
    /// A sink to `127.0.0.1:5010` (RTP), dynamic PT 112, 1460-octet packets.
    pub fn new() -> Self {
        Self {
            host: String::from("127.0.0.1"),
            port: 5010,
            payload_type: 112,
            ssrc: 0x4A58_5300, // "JXS\0"
            max_packet: 1460,
            width: 0,
            height: 0,
            sampling: Sampling::YCbCr422_10,
            packetizer: None,
            socket: None,
            clock_sync: None,
            pacing: None,
            frame_period_ns: 0,
        }
    }

    /// Build the ST 2110-22 SDP advertising this sink's stream, given the source
    /// `exact_fps` (numerator, denominator; the sink does not observe the frame rate
    /// itself). `None` until the sink is configured. A publisher hands this SDP to
    /// receivers, whose [`St2110JxsSrc::apply_sdp`] auto-configures them.
    pub fn sdp(&self, exact_fps: (u32, u32)) -> Option<St2110Sdp> {
        if self.width == 0 || self.height == 0 {
            return None;
        }
        Some(St2110Sdp {
            essence: St2110Essence::JpegXs {
                sampling: self.sampling,
                width: self.width,
                height: self.height,
                exact_fps,
            },
            payload_type: self.payload_type,
            address: self.host.clone(),
            port: self.port,
            ptp: None,
        })
    }
}

impl AsyncElement for St2110JxsSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>> where Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        jxs_geometry(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::LegacySink(Box::new(|c: &Caps| {
            jxs_geometry(c)?;
            Ok(c.clone())
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (width, height) = jxs_geometry(absolute_caps)?;
        self.width = width;
        self.height = height;
        self.frame_period_ns = frame_period_ns(caps_fps(absolute_caps));
        self.packetizer =
            Some(St2110JxsPacketizer::new(self.payload_type, self.ssrc, self.max_packet));
        let sock = UdpSocket::bind(("0.0.0.0", 0)).map_err(io_err)?;
        sock.connect((self.host.as_str(), self.port)).map_err(io_err)?;
        self.socket = Some(sock);
        Ok(ConfigureOutcome::Accepted)
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.clock_sync = Some(sync);
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "ST 2110-22 JPEG XS sink",
            "Sink/Network",
            "Sends a JPEG XS codestream as ST 2110-22 (RFC 9134) RTP over UDP",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new("host", PropKind::Str, "Destination host / multicast group")
                .with_default("127.0.0.1"),
            PropertySpec::new("port", PropKind::Uint, "Destination UDP port").with_default("5010"),
            PropertySpec::new("payload-type", PropKind::Uint, "Dynamic RTP payload type")
                .with_default("112"),
            PropertySpec::new("ssrc", PropKind::Uint, "RTP SSRC"),
            PropertySpec::new("max-packet", PropKind::Uint, "Max RTP packet size in octets")
                .with_default("1460"),
            PropertySpec::new(
                "sampling",
                PropKind::Str,
                "Descriptive sampling for the SDP fmtp (rgba / yuyv / i422_10le)",
            )
            .with_default("i422_10le"),
            PropertySpec::new(
                "pacing",
                PropKind::Str,
                "ST 2110-21 sender pacing: off / linear / gapped",
            )
            .with_default("off"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "host" => {
                self.host = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            "port" => {
                self.port = u16::try_from(value.as_uint().ok_or(PropError::Type)?)
                    .map_err(|_| PropError::Value)?;
                Ok(())
            }
            "payload-type" => {
                self.payload_type = (value.as_uint().ok_or(PropError::Type)? as u8) & 0x7f;
                Ok(())
            }
            "ssrc" => {
                self.ssrc = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "max-packet" => {
                self.max_packet = value.as_uint().ok_or(PropError::Type)? as usize;
                Ok(())
            }
            "sampling" => {
                self.sampling =
                    parse_sampling(value.as_str().ok_or(PropError::Type)?).ok_or(PropError::Value)?;
                Ok(())
            }
            "pacing" => {
                self.pacing = match value.as_str().ok_or(PropError::Type)? {
                    "off" => None,
                    "linear" => Some(PacingProfile::Linear),
                    "gapped" => Some(PacingProfile::Gapped),
                    _ => return Err(PropError::Value),
                };
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "host" => Some(PropValue::Str(self.host.clone())),
            "port" => Some(PropValue::Uint(u64::from(self.port))),
            "payload-type" => Some(PropValue::Uint(u64::from(self.payload_type))),
            "ssrc" => Some(PropValue::Uint(u64::from(self.ssrc))),
            "max-packet" => Some(PropValue::Uint(self.max_packet as u64)),
            "pacing" => Some(PropValue::Str(
                match self.pacing {
                    None => "off",
                    Some(PacingProfile::Linear) => "linear",
                    Some(PacingProfile::Gapped) => "gapped",
                }
                .into(),
            )),
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
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let pkt = self.packetizer.as_mut().ok_or(G2gError::NotConfigured)?;
                    let sock = self.socket.as_ref().ok_or(G2gError::NotConfigured)?;
                    // The frame's sampling instant on the PTP timeline: base + PTS.
                    let base = self.clock_sync.as_ref().map_or(0, ClockSync::base_time);
                    let tai = base.saturating_add(frame.timing.pts_ns);
                    let packets = pkt.packetize(slice.as_slice(), tai);
                    match self.pacing {
                        // ST 2110-21: spread the codestream's packets across the frame
                        // period on the async timer (needs a real framerate + a tokio
                        // reactor); falls back to a burst if the period is unknown. The
                        // schedule + waits are shared with the -20 sink via `pace_send`.
                        Some(profile) if self.frame_period_ns > 0 => {
                            let pacer = Pacer::new(profile, packets.len(), self.frame_period_ns);
                            pace_send(sock, &packets, &pacer).await.map_err(io_err)?;
                        }
                        _ => {
                            for p in &packets {
                                sock.send(p).map_err(io_err)?;
                            }
                        }
                    }
                    Ok(())
                }
                PipelinePacket::CapsChanged(c) => {
                    jxs_geometry(&c)?;
                    Ok(())
                }
                _ => Ok(()),
            }
        })
    }
}

impl PadTemplates for St2110JxsSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(CapsSet::one(Caps::CompressedVideo {
            codec: VideoCodec::JpegXs,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }))])
    }
}

// ================================================================
// Source
// ================================================================

/// ST 2110-22 JPEG XS source: RFC 9134 RTP over UDP -> `CompressedVideo{JpegXs}`
/// `DataFrame`s.
pub struct St2110JxsSrc {
    address: String,
    port: u16,
    width: u32,
    height: u32,
    framerate_fps: u32,
    recv_timeout_ms: u64,
    max_frame_bytes: usize,
    socket: Option<UdpSocket>,
}

impl core::fmt::Debug for St2110JxsSrc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("St2110JxsSrc")
            .field("address", &self.address)
            .field("port", &self.port)
            .field("width", &self.width)
            .field("height", &self.height)
            .finish()
    }
}

impl Default for St2110JxsSrc {
    fn default() -> Self {
        Self::new()
    }
}

impl St2110JxsSrc {
    /// A source binding `0.0.0.0:5010`, 1920x1080 at 60 fps, 500 ms gap timeout.
    pub fn new() -> Self {
        Self {
            address: String::from("0.0.0.0"),
            port: 5010,
            width: 1920,
            height: 1080,
            framerate_fps: 60,
            recv_timeout_ms: 500,
            max_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
            socket: None,
        }
    }

    /// The bound local UDP port after `configure_pipeline` (for tests binding an
    /// ephemeral port with `port = 0`).
    pub fn local_port(&self) -> Option<u16> {
        self.socket.as_ref().and_then(|s| s.local_addr().ok()).map(|a| a.port())
    }

    /// Auto-configure this source's geometry from a parsed JPEG XS [`St2110Sdp`] (the
    /// receiver path). Returns false, leaving the source unchanged, if the SDP is not
    /// a JPEG XS essence. Call before `configure_pipeline`.
    pub fn apply_sdp(&mut self, sdp: &St2110Sdp) -> bool {
        let St2110Essence::JpegXs { width, height, exact_fps, .. } = &sdp.essence else {
            return false;
        };
        self.width = *width;
        self.height = *height;
        self.framerate_fps = match exact_fps.1 {
            0 | 1 => exact_fps.0,
            den => (exact_fps.0 + den / 2) / den,
        };
        self.port = sdp.port;
        true
    }

    fn caps(&self) -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::JpegXs,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: Rate::Fixed(self.framerate_fps << 16),
        }
    }
}

impl SourceLoop for St2110JxsSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>> where Self: 'a;
    type CapsFuture<'a> = Ready<Result<Caps, G2gError>> where Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        ready(Ok(self.caps()))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let sock = UdpSocket::bind((self.address.as_str(), self.port)).map_err(io_err)?;
        sock.set_read_timeout(Some(Duration::from_millis(self.recv_timeout_ms))).map_err(io_err)?;
        self.socket = Some(sock);
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let sock = self.socket.as_ref().ok_or(G2gError::NotConfigured)?;
            let mut depack = St2110JxsDepacketizer::new(self.max_frame_bytes);
            let clock = MediaClock::video();
            let mut base_rtp: Option<u32> = None;
            let mut count = 0u64;
            let mut buf = [0u8; 65_536];
            loop {
                let n = match sock.recv_from(&mut buf) {
                    Ok((n, _)) => n,
                    // A gap (read timeout) ends the stream cleanly.
                    Err(e)
                        if matches!(
                            e.kind(),
                            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                        ) =>
                    {
                        break
                    }
                    Err(e) => return Err(io_err(e)),
                };
                let Some(jxs) = depack.depacketize(&buf[..n]) else { continue };
                // PTS = the frame's media-clock offset from the first frame (the PTP
                // grandmaster supplies absolute time upstream).
                let base = *base_rtp.get_or_insert(jxs.rtp_timestamp);
                let pts_ns = clock.ticks_to_ns(u64::from(jxs.rtp_timestamp.wrapping_sub(base)));
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(
                        jxs.codestream.into_boxed_slice(),
                    )),
                    timing: FrameTiming { pts_ns, ..FrameTiming::default() },
                    sequence: count,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
                count += 1;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(count)
        })
    }
}

impl PadTemplates for St2110JxsSrc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::source(CapsSet::one(Caps::CompressedVideo {
            codec: VideoCodec::JpegXs,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }))])
    }
}

// The source's geometry is set via properties (the -22 stream description is out of
// band); expose them so `gst-launch`-style config works.
impl St2110JxsSrc {
    /// The runtime property specs for the source geometry.
    pub const PROPS: &'static [PropertySpec] = &[
        PropertySpec::new("address", PropKind::Str, "Local bind address").with_default("0.0.0.0"),
        PropertySpec::new("port", PropKind::Uint, "Local UDP port").with_default("5010"),
        PropertySpec::new("width", PropKind::Uint, "Frame width in pixels").with_default("1920"),
        PropertySpec::new("height", PropKind::Uint, "Frame height in pixels").with_default("1080"),
        PropertySpec::new("framerate", PropKind::Uint, "Frame rate in fps").with_default("60"),
    ];

    /// Set a geometry property (mirrors the [`AsyncElement`] property convention on
    /// this [`SourceLoop`], so `parse_launch` can configure it).
    pub fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "address" => {
                self.address = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            "port" => {
                self.port = u16::try_from(value.as_uint().ok_or(PropError::Type)?)
                    .map_err(|_| PropError::Value)?;
                Ok(())
            }
            "width" => {
                self.width = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "height" => {
                self.height = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "framerate" => {
                self.framerate_fps = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    /// Read a geometry property.
    pub fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "address" => Some(PropValue::Str(self.address.clone())),
            "port" => Some(PropValue::Uint(u64::from(self.port))),
            "width" => Some(PropValue::Uint(u64::from(self.width))),
            "height" => Some(PropValue::Uint(u64::from(self.height))),
            "framerate" => Some(PropValue::Uint(u64::from(self.framerate_fps))),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::runtime::block_on;
    use g2g_core::{MonotonicClock, PushOutcome};
    use std::sync::Arc;

    /// Collects the codestream bytes of every DataFrame the source emits.
    #[derive(Default)]
    struct Capture {
        frames: Vec<Vec<u8>>,
        eos: bool,
    }
    impl OutputSink for Capture {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                match packet {
                    PipelinePacket::DataFrame(f) => {
                        if let MemoryDomain::System(s) = &f.domain {
                            self.frames.push(s.as_slice().to_vec());
                        }
                    }
                    PipelinePacket::Eos => self.eos = true,
                    _ => {}
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    fn jxs_frame(bytes: Vec<u8>, pts_ns: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            timing: FrameTiming { pts_ns, ..FrameTiming::default() },
            sequence: 0,
            meta: Default::default(),
        })
    }

    fn caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::JpegXs,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Fixed(60 << 16),
        }
    }

    #[test]
    fn jxs_sink_to_src_over_udp_loopback() {
        // End to end on real UDP (localhost): sink -> UDP -> src. A small MTU splits
        // the codestream across many packets; UDP buffers them so we send then
        // receive sequentially, no threads. The codestream is opaque bytes here.
        let mut src = St2110JxsSrc::new();
        src.set_property("address", PropValue::Str("127.0.0.1".into())).unwrap();
        src.set_property("port", PropValue::Uint(0)).unwrap();
        src.recv_timeout_ms = 300;
        src.configure_pipeline(&caps()).expect("src binds");
        let port = src.local_port().expect("bound port");

        let mut sink = St2110JxsSink::new();
        sink.host = String::from("127.0.0.1");
        sink.port = port;
        sink.max_packet = 200; // force many packets across the codestream
        sink.configure_pipeline(&caps()).expect("sink configures");
        let clock: Arc<dyn g2g_core::PipelineClock + Send + Sync> = Arc::new(MonotonicClock);
        sink.set_clock_sync(ClockSync::new(clock, 1_700_000_000_000_000_000));

        // A distinct pseudo-codestream (opaque to the transport).
        let cs: Vec<u8> = (0..3000).map(|i| (i * 13 + 1) as u8).collect();
        let mut null = Capture::default();
        block_on(sink.process(jxs_frame(cs.clone(), 0), &mut null)).expect("sink sends");

        let mut cap = Capture::default();
        let n = block_on(src.run(&mut cap)).expect("src runs");

        assert_eq!(n, 1, "one codestream reassembled");
        assert!(cap.eos, "source emitted EOS on the gap");
        assert_eq!(cap.frames.len(), 1);
        assert_eq!(cap.frames[0], cs, "JPEG XS codestream survives sink -> UDP -> src");
    }

    #[test]
    fn sink_properties_round_trip() {
        let mut sink = St2110JxsSink::new();
        sink.set_property("host", PropValue::Str("239.22.0.9".into())).unwrap();
        sink.set_property("max-packet", PropValue::Uint(9000)).unwrap();
        sink.set_property("sampling", PropValue::Str("yuyv".into())).unwrap();
        assert_eq!(sink.get_property("host"), Some(PropValue::Str("239.22.0.9".into())));
        assert_eq!(sink.get_property("max-packet"), Some(PropValue::Uint(9000)));
        assert_eq!(
            sink.set_property("port", PropValue::Uint(70_000)),
            Err(PropError::Value),
            "a port past u16 is rejected"
        );
        assert_eq!(
            sink.set_property("sampling", PropValue::Str("nv12".into())),
            Err(PropError::Value),
            "a format with no -20-style sampling is rejected"
        );
    }

    #[test]
    fn pacing_property_round_trips_and_rejects_unknown() {
        let mut sink = St2110JxsSink::new();
        assert_eq!(sink.get_property("pacing"), Some(PropValue::Str("off".into())));
        sink.set_property("pacing", PropValue::Str("linear".into())).unwrap();
        assert_eq!(sink.get_property("pacing"), Some(PropValue::Str("linear".into())));
        assert_eq!(sink.pacing, Some(PacingProfile::Linear));
        assert_eq!(
            sink.set_property("pacing", PropValue::Str("bogus".into())),
            Err(PropError::Value)
        );
    }

    #[tokio::test]
    async fn gapped_pacing_spreads_a_codestream_over_its_period() {
        // Under a tokio reactor, pacing sleeps between packets so a codestream's packets
        // are emitted across (most of) the frame period instead of at once.
        let caps = Caps::CompressedVideo {
            codec: VideoCodec::JpegXs,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Fixed(50 << 16), // 20 ms period
        };
        let rx = UdpSocket::bind(("127.0.0.1", 0)).unwrap();
        rx.set_nonblocking(true).unwrap();
        let port = rx.local_addr().unwrap().port();

        let mut sink = St2110JxsSink::new();
        sink.host = String::from("127.0.0.1");
        sink.port = port;
        sink.max_packet = 200; // force several packets across the codestream
        sink.set_property("pacing", PropValue::Str("gapped".into())).unwrap();
        sink.configure_pipeline(&caps).expect("configures");
        assert_eq!(sink.frame_period_ns, 20_000_000, "50 fps -> 20 ms period");

        let cs: Vec<u8> = (0..3000).map(|i| (i * 13 + 1) as u8).collect();
        let start = std::time::Instant::now();
        let mut null = Capture::default();
        sink.process(jxs_frame(cs, 0), &mut null).await.expect("sink sends");
        let elapsed = start.elapsed();

        // Gapped packs into the active 1080/1125, but still spreads over milliseconds
        // (a burst finishes in microseconds).
        assert!(elapsed.as_millis() >= 8, "gapped pacing spread the codestream: {elapsed:?}");
        let mut count = 0;
        let mut buf = [0u8; 2048];
        while rx.recv_from(&mut buf).is_ok() {
            count += 1;
        }
        assert!(count > 1, "several paced packets arrived: {count}");
    }

    #[test]
    fn sdp_generated_by_sink_configures_a_src() {
        // The full out-of-band loop: a sink's -22 SDP -> text -> parse -> a receiver's
        // St2110JxsSrc auto-configured from it, no sockets.
        let mut sink = St2110JxsSink::new();
        sink.host = String::from("239.22.30.40");
        sink.port = 5010;
        sink.configure_pipeline(&caps()).expect("configures");

        let sdp = sink.sdp((60000, 1001)).expect("sink is configured");
        let text = sdp.to_sdp();
        assert!(text.contains("jxsv/90000"), "advertises the -22 rtpmap\n{text}");
        let parsed = St2110Sdp::parse(&text).expect("parses");

        let mut src = St2110JxsSrc::new();
        assert!(src.apply_sdp(&parsed), "JPEG XS SDP configures the src");
        assert_eq!(src.get_property("width"), Some(PropValue::Uint(1920)));
        assert_eq!(src.get_property("height"), Some(PropValue::Uint(1080)));
        assert_eq!(src.get_property("framerate"), Some(PropValue::Uint(60))); // 59.94 -> 60
        assert_eq!(src.get_property("port"), Some(PropValue::Uint(5010)));
        // A non-JPEG-XS SDP leaves the src unchanged.
        let anc = St2110Sdp {
            essence: St2110Essence::Ancillary,
            payload_type: 100,
            address: "239.1.1.1".into(),
            port: 6000,
            ptp: None,
        };
        assert!(!src.apply_sdp(&anc), "non-JPEG-XS SDP is rejected");
        assert_eq!(src.get_property("port"), Some(PropValue::Uint(5010)), "unchanged");
    }
}
