//! ST 2110-20 uncompressed video network elements (M599): a sink and source that
//! put the sans-IO `st2110video` core (RFC 4175) on the wire over UDP, with RTP
//! timestamps from the PTP media clock.
//!
//! [`St2110VideoSink`] takes packed raw-video `DataFrame`s (RGBA 8-bit or YUYV
//! 4:2:2), maps each frame's PTS through the elected clock to a PTP/TAI sampling
//! instant, slices the frame into RFC 4175 SRD line runs, and sends the RTP
//! packets to a UDP destination. [`St2110VideoSrc`] binds a UDP socket, reassembles
//! received -20 packets into whole frames (the geometry from its properties, since
//! -20 carries it out of band in SDP), and reconstructs each frame's PTS from the
//! RTP timestamp. Because the timestamp is the shared 90 kHz media clock, a
//! receiver on the same grandmaster stays frame-locked to the source.
//!
//! Uncompressed HD is multi-Gbps, so this is bandwidth-heavy; the loopback CI test
//! uses a tiny frame. 10-bit 4:2:2 (the broadcast norm) follows the wider sampling
//! in the core. The blocking recv in the source runs with a read timeout so it
//! stays cooperative and ends cleanly on a gap.

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
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat,
};

use crate::capsfilter::parse_raw_format;
use crate::filesink::io_err;
use crate::st2110dup::RedundantRtpReceiver;
use crate::st2110pacing::{frame_period_ns, pace_send, Pacer, PacingProfile};
use crate::st2110sdp::{St2110Essence, St2110Sdp};
use crate::st2110video::{Sampling, St2110VideoDepacketizer, St2110VideoPacketizer};
use crate::videoconvert::raw_format_to_str;

/// The whole-fps rate from a `Caps::RawVideo` framerate (`Rate::Fixed` is fps << 16),
/// or 0 if not fixed (do not pace).
fn caps_fps(caps: &Caps) -> u32 {
    match caps {
        Caps::RawVideo {
            framerate: Rate::Fixed(q),
            ..
        } => q >> 16,
        _ => 0,
    }
}

/// Extract the geometry from an absolute `Caps::RawVideo` with a -20-mappable
/// format: returns (format, width, height). Errors on a non-raw caps, an unmapped
/// format, or a non-fixed dimension.
fn video_geometry(caps: &Caps) -> Result<(RawVideoFormat, usize, usize), G2gError> {
    match caps {
        Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        } if Sampling::from_format(*format).is_some() => Ok((*format, *w as usize, *h as usize)),
        _ => Err(G2gError::CapsMismatch),
    }
}

// ================================================================
// Sink
// ================================================================

/// ST 2110-20 video sink: packed raw-video `DataFrame`s -> RFC 4175 RTP over UDP.
pub struct St2110VideoSink {
    host: String,
    port: u16,
    payload_type: u8,
    ssrc: u32,
    max_packet: usize,
    width: usize,
    height: usize,
    sampling: Option<Sampling>,
    packetizer: Option<St2110VideoPacketizer>,
    socket: Option<UdpSocket>,
    clock_sync: Option<ClockSync>,
    /// ST 2110-21 sender pacing profile (None = burst all packets, the default).
    pacing: Option<PacingProfile>,
    /// Frame period in ns (from the negotiated framerate), for the -21 schedule.
    frame_period_ns: u64,
}

impl core::fmt::Debug for St2110VideoSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("St2110VideoSink")
            .field("host", &self.host)
            .field("port", &self.port)
            .field("width", &self.width)
            .field("height", &self.height)
            .finish()
    }
}

impl Default for St2110VideoSink {
    fn default() -> Self {
        Self::new()
    }
}

impl St2110VideoSink {
    /// A sink to `127.0.0.1:5008` (RTP), dynamic PT 96, 1460-octet packets.
    pub fn new() -> Self {
        Self {
            host: String::from("127.0.0.1"),
            port: 5008,
            payload_type: 96,
            ssrc: 0x3273_3230, // "s2 0"
            max_packet: 1460,
            width: 0,
            height: 0,
            sampling: None,
            packetizer: None,
            socket: None,
            clock_sync: None,
            pacing: None,
            frame_period_ns: 0,
        }
    }

    /// Build the ST 2110-20 SDP advertising this sink's stream, given the source
    /// `exact_fps` (numerator, denominator; the sink does not observe the frame
    /// rate itself). `None` until the sink is configured. A publisher hands this
    /// SDP to receivers, whose [`St2110VideoSrc::apply_sdp`] auto-configures them.
    pub fn sdp(&self, exact_fps: (u32, u32)) -> Option<St2110Sdp> {
        let sampling = self.sampling?;
        Some(St2110Sdp {
            essence: St2110Essence::Video {
                sampling,
                width: self.width as u32,
                height: self.height as u32,
                exact_fps,
            },
            payload_type: self.payload_type,
            address: self.host.clone(),
            port: self.port,
            ptp: None,
        })
    }
}

impl AsyncElement for St2110VideoSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        video_geometry(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::LegacySink(Box::new(|c: &Caps| {
            video_geometry(c)?;
            Ok(c.clone())
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (format, width, height) = video_geometry(absolute_caps)?;
        let sampling = Sampling::from_format(format).ok_or(G2gError::CapsMismatch)?;
        self.width = width;
        self.height = height;
        self.sampling = Some(sampling);
        self.frame_period_ns = frame_period_ns(caps_fps(absolute_caps));
        self.packetizer = Some(St2110VideoPacketizer::new(
            self.payload_type,
            self.ssrc,
            sampling,
            self.max_packet,
        ));
        let sock = UdpSocket::bind(("0.0.0.0", 0)).map_err(io_err)?;
        sock.connect((self.host.as_str(), self.port))
            .map_err(io_err)?;
        self.socket = Some(sock);
        Ok(ConfigureOutcome::Accepted)
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.clock_sync = Some(sync);
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "ST 2110-20 video sink",
            "Sink/Network",
            "Sends uncompressed video as ST 2110-20 (RFC 4175) RTP over UDP",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new("host", PropKind::Str, "Destination host / multicast group")
                .with_default("127.0.0.1"),
            PropertySpec::new("port", PropKind::Uint, "Destination UDP port").with_default("5008"),
            PropertySpec::new("payload-type", PropKind::Uint, "Dynamic RTP payload type")
                .with_default("96"),
            PropertySpec::new("ssrc", PropKind::Uint, "RTP SSRC"),
            PropertySpec::new(
                "max-packet",
                PropKind::Uint,
                "Max RTP packet size in octets",
            )
            .with_default("1460"),
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
                    let packets = pkt
                        .packetize(slice.as_slice(), self.width, self.height, tai)
                        .ok_or(G2gError::CapsMismatch)?;
                    match self.pacing {
                        // ST 2110-21: spread the frame's packets across the frame
                        // period on the async timer, so the network sees a smooth
                        // flow instead of a burst. Needs a real framerate + a tokio
                        // reactor (the production runner); falls back to a burst if
                        // the period is unknown.
                        Some(profile) if self.frame_period_ns > 0 => {
                            let pacer = Pacer::new(profile, packets.len(), self.frame_period_ns);
                            pace_send(sock, &packets, &pacer).await.map_err(io_err)?;
                        }
                        _ => {
                            for p in packets {
                                sock.send(&p).map_err(io_err)?;
                            }
                        }
                    }
                    Ok(())
                }
                PipelinePacket::CapsChanged(c) => {
                    video_geometry(&c)?;
                    Ok(())
                }
                _ => Ok(()),
            }
        })
    }
}

impl PadTemplates for St2110VideoSink {
    fn pad_templates() -> Vec<PadTemplate> {
        let alts = [
            RawVideoFormat::Rgba8,
            RawVideoFormat::Yuyv,
            RawVideoFormat::I422p10,
        ]
        .map(|format| Caps::RawVideo {
            format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        })
        .to_vec();
        Vec::from([PadTemplate::sink(CapsSet::from_alternatives(alts))])
    }
}

// ================================================================
// Source
// ================================================================

/// ST 2110-20 video source: RFC 4175 RTP over UDP -> packed raw-video `DataFrame`s.
pub struct St2110VideoSrc {
    address: String,
    port: u16,
    format: RawVideoFormat,
    width: usize,
    height: usize,
    framerate_fps: u32,
    recv_timeout_ms: u64,
    socket: Option<UdpSocket>,
    /// ST 2110-7 seamless protection: when set, a second ("blue") path is bound and
    /// the two streams are merged by RTP sequence number (see [`RedundantRtpReceiver`]).
    redundant: bool,
    redundant_address: String,
    redundant_port: u16,
    socket2: Option<UdpSocket>,
}

/// Poll granularity for the redundant receiver: how often each path is checked while
/// waiting for the next packet. Bounded so an idle path is noticed promptly without a
/// busy loop; the whole-stream end still waits `recv_timeout_ms`.
const REDUNDANT_POLL_MS: u64 = 20;

/// Build one output `Frame` from a reassembled -20 frame's bytes and RTP timestamp,
/// reconstructing the PTS as the media-clock offset from the first frame. Shared by
/// the single-path and redundant (-7) receive loops.
fn build_frame(
    bytes: Vec<u8>,
    rtp_timestamp: u32,
    clock: &MediaClock,
    base_rtp: &mut Option<u32>,
    count: u64,
) -> Frame {
    let base = *base_rtp.get_or_insert(rtp_timestamp);
    let pts_ns = clock.ticks_to_ns(u64::from(rtp_timestamp.wrapping_sub(base)));
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns,
            ..FrameTiming::default()
        },
        sequence: count,
        meta: Default::default(),
    }
}

impl core::fmt::Debug for St2110VideoSrc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("St2110VideoSrc")
            .field("address", &self.address)
            .field("port", &self.port)
            .field("format", &self.format)
            .field("width", &self.width)
            .field("height", &self.height)
            .finish()
    }
}

impl Default for St2110VideoSrc {
    fn default() -> Self {
        Self::new()
    }
}

impl St2110VideoSrc {
    /// A source binding `0.0.0.0:5008`, 1280x720 YUYV at 60 fps, 500 ms gap timeout.
    pub fn new() -> Self {
        Self {
            address: String::from("0.0.0.0"),
            port: 5008,
            format: RawVideoFormat::Yuyv,
            width: 1280,
            height: 720,
            framerate_fps: 60,
            recv_timeout_ms: 500,
            socket: None,
            redundant: false,
            redundant_address: String::from("0.0.0.0"),
            redundant_port: 5009,
            socket2: None,
        }
    }

    /// The bound local UDP port of the primary ("red") path after
    /// `configure_pipeline` (for tests binding an ephemeral port with `port = 0`).
    pub fn local_port(&self) -> Option<u16> {
        self.socket
            .as_ref()
            .and_then(|s| s.local_addr().ok())
            .map(|a| a.port())
    }

    /// The bound local UDP port of the redundant ("blue") path, or `None` if
    /// redundancy is off or unbound.
    pub fn redundant_local_port(&self) -> Option<u16> {
        self.socket2
            .as_ref()
            .and_then(|s| s.local_addr().ok())
            .map(|a| a.port())
    }

    /// Auto-configure this source's geometry from a parsed video [`St2110Sdp`] (the
    /// receiver path: a publisher's SDP names the format / size / rate / group /
    /// port). Returns false, leaving the source unchanged, if the SDP is not a video
    /// essence. Call before `configure_pipeline`. The bind address stays as
    /// configured (a receiver joins the group on its chosen interface); the group
    /// address is available on the SDP for a caller that wants to bind it directly.
    pub fn apply_sdp(&mut self, sdp: &St2110Sdp) -> bool {
        let St2110Essence::Video {
            sampling,
            width,
            height,
            exact_fps,
        } = &sdp.essence
        else {
            return false;
        };
        self.format = sampling.raw_format();
        self.width = *width as usize;
        self.height = *height as usize;
        // exactframerate (num/den) rounded to whole fps for the emitted caps Rate.
        self.framerate_fps = match exact_fps.1 {
            0 | 1 => exact_fps.0,
            den => (exact_fps.0 + den / 2) / den,
        };
        self.port = sdp.port;
        true
    }

    fn caps(&self) -> Caps {
        Caps::RawVideo {
            format: self.format,
            width: Dim::Fixed(self.width as u32),
            height: Dim::Fixed(self.height as u32),
            framerate: Rate::Fixed(self.framerate_fps << 16),
        }
    }
}

impl SourceLoop for St2110VideoSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        ready(Ok(self.caps()))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // Validate the geometry maps to a -20 sampling before binding.
        St2110VideoDepacketizer::new(self.format, self.width, self.height)
            .ok_or(G2gError::CapsMismatch)?;
        let sock = UdpSocket::bind((self.address.as_str(), self.port)).map_err(io_err)?;
        sock.set_read_timeout(Some(Duration::from_millis(self.recv_timeout_ms)))
            .map_err(io_err)?;
        self.socket = Some(sock);
        // ST 2110-7: bind the redundant ("blue") path too; the receiver sets its own
        // (shorter) poll timeout on both sockets when the run loop starts.
        if self.redundant {
            let sock2 = UdpSocket::bind((self.redundant_address.as_str(), self.redundant_port))
                .map_err(io_err)?;
            self.socket2 = Some(sock2);
        }
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let mut depack = St2110VideoDepacketizer::new(self.format, self.width, self.height)
                .ok_or(G2gError::CapsMismatch)?;
            let clock = MediaClock::video();
            let mut base_rtp: Option<u32> = None;
            let mut count = 0u64;
            let mut buf = [0u8; 65_536];
            if self.redundant {
                // ST 2110-7: drain both paths through the dedup, depacketizing each RTP
                // sequence number exactly once. A packet lost on one path arrives on the
                // other, so the frame still completes.
                let s1 = self.socket.as_ref().ok_or(G2gError::NotConfigured)?;
                let s2 = self.socket2.as_ref().ok_or(G2gError::NotConfigured)?;
                let socks = [s1, s2];
                let poll = Duration::from_millis(self.recv_timeout_ms.min(REDUNDANT_POLL_MS));
                let idle = Duration::from_millis(self.recv_timeout_ms);
                let mut rx = RedundantRtpReceiver::new(&socks, poll, idle).map_err(io_err)?;
                while let Some(n) = rx.recv_novel(&mut buf).map_err(io_err)? {
                    let Some(vframe) = depack.depacketize(&buf[..n]) else {
                        continue;
                    };
                    let frame = build_frame(
                        vframe.bytes,
                        vframe.rtp_timestamp,
                        &clock,
                        &mut base_rtp,
                        count,
                    );
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                    count += 1;
                }
            } else {
                let sock = self.socket.as_ref().ok_or(G2gError::NotConfigured)?;
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
                    let Some(vframe) = depack.depacketize(&buf[..n]) else {
                        continue;
                    };
                    // PTS = the frame's media-clock offset from the first frame (the PTP
                    // grandmaster supplies absolute time upstream).
                    let frame = build_frame(
                        vframe.bytes,
                        vframe.rtp_timestamp,
                        &clock,
                        &mut base_rtp,
                        count,
                    );
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                    count += 1;
                }
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(count)
        })
    }
}

impl PadTemplates for St2110VideoSrc {
    fn pad_templates() -> Vec<PadTemplate> {
        let alts = [
            RawVideoFormat::Rgba8,
            RawVideoFormat::Yuyv,
            RawVideoFormat::I422p10,
        ]
        .map(|format| Caps::RawVideo {
            format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        })
        .to_vec();
        Vec::from([PadTemplate::source(CapsSet::from_alternatives(alts))])
    }
}

// The source's geometry is set via properties (the -20 stream description is out
// of band); expose them so `gst-launch`-style config works.
impl St2110VideoSrc {
    /// The runtime property specs for the source geometry.
    pub const PROPS: &'static [PropertySpec] = &[
        PropertySpec::new("address", PropKind::Str, "Local bind address").with_default("0.0.0.0"),
        PropertySpec::new("port", PropKind::Uint, "Local UDP port").with_default("5008"),
        PropertySpec::new("format", PropKind::Str, "Raw video format (rgba / yuyv)")
            .with_default("yuyv"),
        PropertySpec::new("width", PropKind::Uint, "Frame width in pixels").with_default("1280"),
        PropertySpec::new("height", PropKind::Uint, "Frame height in pixels").with_default("720"),
        PropertySpec::new("framerate", PropKind::Uint, "Frame rate in fps").with_default("60"),
        PropertySpec::new(
            "redundant",
            PropKind::Bool,
            "ST 2110-7 seamless protection: join a second (blue) path",
        )
        .with_default("false"),
        PropertySpec::new("redundant-address", PropKind::Str, "Blue-path bind address")
            .with_default("0.0.0.0"),
        PropertySpec::new("redundant-port", PropKind::Uint, "Blue-path UDP port")
            .with_default("5009"),
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
            "format" => {
                let f = parse_raw_format(value.as_str().ok_or(PropError::Type)?)
                    .filter(|f| Sampling::from_format(*f).is_some())
                    .ok_or(PropError::Value)?;
                self.format = f;
                Ok(())
            }
            "width" => {
                self.width = value.as_uint().ok_or(PropError::Type)? as usize;
                Ok(())
            }
            "height" => {
                self.height = value.as_uint().ok_or(PropError::Type)? as usize;
                Ok(())
            }
            "framerate" => {
                self.framerate_fps = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "redundant" => {
                self.redundant = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "redundant-address" => {
                self.redundant_address = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            "redundant-port" => {
                self.redundant_port = u16::try_from(value.as_uint().ok_or(PropError::Type)?)
                    .map_err(|_| PropError::Value)?;
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
            "format" => Some(PropValue::Str(raw_format_to_str(self.format).into())),
            "width" => Some(PropValue::Uint(self.width as u64)),
            "height" => Some(PropValue::Uint(self.height as u64)),
            "framerate" => Some(PropValue::Uint(u64::from(self.framerate_fps))),
            "redundant" => Some(PropValue::Bool(self.redundant)),
            "redundant-address" => Some(PropValue::Str(self.redundant_address.clone())),
            "redundant-port" => Some(PropValue::Uint(u64::from(self.redundant_port))),
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

    /// Collects the packed bytes of every DataFrame the source emits.
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

    fn raw_frame(bytes: Vec<u8>, pts_ns: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            timing: FrameTiming {
                pts_ns,
                ..FrameTiming::default()
            },
            sequence: 0,
            meta: Default::default(),
        })
    }

    #[test]
    fn video_sink_to_src_over_udp_loopback() {
        // End to end on real UDP (localhost): sink -> UDP -> src. A small frame with
        // a small MTU exercises SRD splitting; UDP buffers the packets so we send
        // then receive sequentially, no threads.
        let (w, h) = (16usize, 8usize);
        let caps = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(w as u32),
            height: Dim::Fixed(h as u32),
            framerate: Rate::Fixed(30 << 16),
        };

        let mut src = St2110VideoSrc::new();
        src.set_property("address", PropValue::Str("127.0.0.1".into()))
            .unwrap();
        src.set_property("port", PropValue::Uint(0)).unwrap();
        src.set_property("format", PropValue::Str("rgba".into()))
            .unwrap();
        src.set_property("width", PropValue::Uint(w as u64))
            .unwrap();
        src.set_property("height", PropValue::Uint(h as u64))
            .unwrap();
        src.recv_timeout_ms = 300;
        src.configure_pipeline(&caps).expect("src binds");
        let port = src.local_port().expect("bound port");

        let mut sink = St2110VideoSink::new();
        sink.host = String::from("127.0.0.1");
        sink.port = port;
        sink.max_packet = 80; // force many SRD-split packets across the frame
        sink.configure_pipeline(&caps).expect("sink configures");
        let clock: Arc<dyn g2g_core::PipelineClock + Send + Sync> = Arc::new(MonotonicClock);
        sink.set_clock_sync(ClockSync::new(clock, 1_700_000_000_000_000_000));

        // A distinct RGBA test frame.
        let input: Vec<u8> = (0..w * 4 * h).map(|i| (i * 11 + 5) as u8).collect();
        let mut null = Capture::default();
        block_on(sink.process(raw_frame(input.clone(), 0), &mut null)).expect("sink sends");

        let mut cap = Capture::default();
        let n = block_on(src.run(&mut cap)).expect("src runs");

        assert_eq!(n, 1, "one frame reassembled");
        assert!(cap.eos, "source emitted EOS on the gap");
        assert_eq!(cap.frames.len(), 1);
        assert_eq!(
            cap.frames[0], input,
            "RGBA frame survives sink -> UDP -> src"
        );
    }

    #[test]
    fn video_10bit_422_sink_to_src_over_udp_loopback() {
        // The planar I422p10 path end to end: sink bit-packs to 10-bit pgroups on
        // the wire, src unpacks back to the three planes.
        let (w, h) = (4usize, 4usize);
        let caps = Caps::RawVideo {
            format: RawVideoFormat::I422p10,
            width: Dim::Fixed(w as u32),
            height: Dim::Fixed(h as u32),
            framerate: Rate::Fixed(60 << 16),
        };

        let mut src = St2110VideoSrc::new();
        src.set_property("address", PropValue::Str("127.0.0.1".into()))
            .unwrap();
        src.set_property("port", PropValue::Uint(0)).unwrap();
        src.set_property("format", PropValue::Str("i422_10le".into()))
            .unwrap();
        src.set_property("width", PropValue::Uint(w as u64))
            .unwrap();
        src.set_property("height", PropValue::Uint(h as u64))
            .unwrap();
        src.recv_timeout_ms = 300;
        src.configure_pipeline(&caps).expect("src binds");
        let port = src.local_port().expect("bound port");

        let mut sink = St2110VideoSink::new();
        sink.host = String::from("127.0.0.1");
        sink.port = port;
        sink.configure_pipeline(&caps).expect("sink configures");

        // A planar I422p10 buffer of distinct 10-bit samples.
        let mut input = alloc::vec![0u8; w * h * 4];
        for (i, word) in input.chunks_exact_mut(2).enumerate() {
            word.copy_from_slice(&(((i * 17 + 3) as u16) & 0x03FF).to_le_bytes());
        }
        let mut null = Capture::default();
        block_on(sink.process(raw_frame(input.clone(), 0), &mut null)).expect("sink sends");

        let mut cap = Capture::default();
        let n = block_on(src.run(&mut cap)).expect("src runs");
        assert_eq!(n, 1);
        assert_eq!(
            cap.frames[0], input,
            "10-bit 4:2:2 frame survives sink -> UDP -> src"
        );
    }

    #[test]
    fn redundant_src_reconstructs_a_frame_from_two_lossy_paths() {
        // ST 2110-7: the source binds two paths (red + blue). We packetize one frame
        // with the -20 packetizer and split its packets across the paths so each drops
        // a different third (and one third arrives on both, a real duplicate the dedup
        // must discard), with no packet lost on both. The frame must still complete
        // byte-exact.
        let (w, h) = (16usize, 8usize);
        let caps = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(w as u32),
            height: Dim::Fixed(h as u32),
            framerate: Rate::Fixed(30 << 16),
        };

        let mut src = St2110VideoSrc::new();
        src.set_property("address", PropValue::Str("127.0.0.1".into()))
            .unwrap();
        src.set_property("port", PropValue::Uint(0)).unwrap();
        src.set_property("redundant", PropValue::Bool(true))
            .unwrap();
        src.set_property("redundant-address", PropValue::Str("127.0.0.1".into()))
            .unwrap();
        src.set_property("redundant-port", PropValue::Uint(0))
            .unwrap();
        src.set_property("format", PropValue::Str("rgba".into()))
            .unwrap();
        src.set_property("width", PropValue::Uint(w as u64))
            .unwrap();
        src.set_property("height", PropValue::Uint(h as u64))
            .unwrap();
        src.recv_timeout_ms = 300;
        src.configure_pipeline(&caps).expect("src binds both paths");
        let red = src.local_port().expect("red bound");
        let blue = src.redundant_local_port().expect("blue bound");
        assert_ne!(red, blue, "the two paths bind distinct ports");

        // Packetize the frame ourselves (as the sink would). Both paths carry the
        // complete stream (that is what -7 redundancy is); we simulate a little loss by
        // dropping a few different mid-stream packets on each path. No packet is lost on
        // both, and the marker (last) packet is never dropped, so -7 recovers the frame.
        let input: Vec<u8> = (0..w * 4 * h).map(|i| (i * 11 + 5) as u8).collect();
        let mut tx = St2110VideoPacketizer::new(96, 0xABCD, Sampling::Rgba8, 60);
        let packets = tx
            .packetize(&input, w, h, 1_000_000_000)
            .expect("packetizes");
        assert!(packets.len() > 6, "frame split into several packets");
        let last = packets.len() - 1;
        let red_drops = [2usize];
        let blue_drops = [3usize, 5usize];

        let sender = UdpSocket::bind(("127.0.0.1", 0)).unwrap();
        // Interleave sends per sequence, as two concurrent in-order paths would arrive.
        for (i, p) in packets.iter().enumerate() {
            let on_red = i == last || !red_drops.contains(&i);
            let on_blue = i == last || !blue_drops.contains(&i);
            assert!(on_red || on_blue, "no packet lost on both paths");
            if on_red {
                sender.send_to(p, ("127.0.0.1", red)).unwrap();
            }
            if on_blue {
                sender.send_to(p, ("127.0.0.1", blue)).unwrap();
            }
        }

        let mut cap = Capture::default();
        let n = block_on(src.run(&mut cap)).expect("src runs");
        assert_eq!(n, 1, "one frame reassembled from the merged paths");
        assert!(cap.eos, "source emitted EOS on the gap");
        assert_eq!(
            cap.frames[0], input,
            "the RGBA frame is reconstructed byte-exact via -7 merge"
        );
    }

    #[test]
    fn redundant_properties_round_trip() {
        let mut src = St2110VideoSrc::new();
        assert_eq!(src.get_property("redundant"), Some(PropValue::Bool(false)));
        src.set_property("redundant", PropValue::Bool(true))
            .unwrap();
        src.set_property("redundant-address", PropValue::Str("239.0.0.2".into()))
            .unwrap();
        src.set_property("redundant-port", PropValue::Uint(5009))
            .unwrap();
        assert_eq!(src.get_property("redundant"), Some(PropValue::Bool(true)));
        assert_eq!(
            src.get_property("redundant-address"),
            Some(PropValue::Str("239.0.0.2".into()))
        );
        assert_eq!(
            src.get_property("redundant-port"),
            Some(PropValue::Uint(5009))
        );
    }

    #[test]
    fn sink_properties_round_trip() {
        let mut sink = St2110VideoSink::new();
        sink.set_property("host", PropValue::Str("239.0.0.9".into()))
            .unwrap();
        sink.set_property("max-packet", PropValue::Uint(9000))
            .unwrap();
        assert_eq!(
            sink.get_property("host"),
            Some(PropValue::Str("239.0.0.9".into()))
        );
        assert_eq!(sink.get_property("max-packet"), Some(PropValue::Uint(9000)));
        assert_eq!(
            sink.set_property("port", PropValue::Uint(70_000)),
            Err(PropError::Value),
            "a port past u16 is rejected"
        );
    }

    #[test]
    fn sdp_generated_by_sink_configures_a_src() {
        // The full out-of-band loop: a sink's SDP -> text -> parse -> a receiver's
        // St2110VideoSrc auto-configured from it, no sockets.
        let caps = Caps::RawVideo {
            format: RawVideoFormat::I422p10,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Fixed(60 << 16),
        };
        let mut sink = St2110VideoSink::new();
        sink.host = String::from("239.20.30.40");
        sink.port = 5008;
        sink.configure_pipeline(&caps).expect("configures");

        // Publish, serialize, parse (as a receiver would).
        let sdp = sink.sdp((60000, 1001)).expect("sink is configured");
        let text = sdp.to_sdp();
        let parsed = St2110Sdp::parse(&text).expect("parses");

        let mut src = St2110VideoSrc::new();
        assert!(src.apply_sdp(&parsed), "video SDP configures the src");
        assert_eq!(
            src.get_property("format"),
            Some(PropValue::Str("I422_10LE".into()))
        );
        assert_eq!(src.get_property("width"), Some(PropValue::Uint(1920)));
        assert_eq!(src.get_property("height"), Some(PropValue::Uint(1080)));
        assert_eq!(src.get_property("framerate"), Some(PropValue::Uint(60))); // 59.94 -> 60
        assert_eq!(src.get_property("port"), Some(PropValue::Uint(5008)));
        // A non-video SDP leaves the src unchanged.
        let audio = St2110Sdp {
            essence: St2110Essence::Ancillary,
            payload_type: 100,
            address: "239.1.1.1".into(),
            port: 6000,
            ptp: None,
        };
        assert!(!src.apply_sdp(&audio), "non-video SDP is rejected");
        assert_eq!(
            src.get_property("port"),
            Some(PropValue::Uint(5008)),
            "unchanged"
        );
    }

    #[test]
    fn src_rejects_unmapped_format_property() {
        let mut src = St2110VideoSrc::new();
        assert_eq!(
            src.set_property("format", PropValue::Str("nv12".into())),
            Err(PropError::Value),
            "NV12 has no -20 sampling"
        );
        src.set_property("format", PropValue::Str("rgba".into()))
            .unwrap();
        assert_eq!(
            src.get_property("format"),
            Some(PropValue::Str("RGBA".into()))
        );
    }

    #[test]
    fn pacing_property_round_trips_and_rejects_unknown() {
        let mut sink = St2110VideoSink::new();
        assert_eq!(
            sink.get_property("pacing"),
            Some(PropValue::Str("off".into()))
        );
        sink.set_property("pacing", PropValue::Str("gapped".into()))
            .unwrap();
        assert_eq!(
            sink.get_property("pacing"),
            Some(PropValue::Str("gapped".into()))
        );
        assert_eq!(sink.pacing, Some(PacingProfile::Gapped));
        assert_eq!(
            sink.set_property("pacing", PropValue::Str("bogus".into())),
            Err(PropError::Value)
        );
    }

    #[tokio::test]
    async fn linear_pacing_spreads_a_frame_over_its_period() {
        // Under a tokio reactor, linear pacing sleeps between packets so a frame's
        // packets are emitted across (most of) the frame period instead of at once.
        let (w, h) = (16usize, 8usize);
        let caps = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(w as u32),
            height: Dim::Fixed(h as u32),
            framerate: Rate::Fixed(50 << 16), // 20 ms period
        };
        // A receiver to count the packets that actually arrive.
        let rx = UdpSocket::bind(("127.0.0.1", 0)).unwrap();
        rx.set_nonblocking(true).unwrap();
        let port = rx.local_addr().unwrap().port();

        let mut sink = St2110VideoSink::new();
        sink.host = String::from("127.0.0.1");
        sink.port = port;
        sink.max_packet = 60; // force several packets across the frame
        sink.set_property("pacing", PropValue::Str("linear".into()))
            .unwrap();
        sink.configure_pipeline(&caps).expect("configures");
        assert_eq!(sink.frame_period_ns, 20_000_000, "50 fps -> 20 ms period");

        let input: Vec<u8> = (0..w * 4 * h).map(|i| (i * 7) as u8).collect();
        let start = std::time::Instant::now();
        let mut null = Capture::default();
        sink.process(raw_frame(input, 0), &mut null)
            .await
            .expect("sink sends");
        let elapsed = start.elapsed();

        // The frame was paced: sending took a meaningful fraction of the 20 ms period
        // (a burst would finish in microseconds).
        assert!(
            elapsed.as_millis() >= 10,
            "linear pacing spread the frame: {elapsed:?}"
        );

        let mut count = 0;
        let mut buf = [0u8; 2048];
        while rx.recv_from(&mut buf).is_ok() {
            count += 1;
        }
        assert!(count > 1, "several paced packets arrived: {count}");
    }
}
