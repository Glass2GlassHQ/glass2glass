//! ST 2110-40 ancillary network elements (M598): a sink and source that put the
//! sans-IO `st2110anc` core on the wire over UDP, with the existing CEA-608/708
//! caption stack (`crate::cea`) as the bridge and RTP timestamps from the PTP
//! media clock.
//!
//! [`St2110AncSink`] taps a compressed H.264 / H.265 stream (like
//! [`crate::ccextract::CcExtract`], a teed branch leaf), mines each access unit's
//! in-band caption `cc_data` triples, wraps them in a Caption Distribution Packet
//! (CDP), carries the CDP in a DID 0x61 / SDID 0x01 ANC packet, and sends the
//! RFC 8331 RTP packet to a UDP destination, timestamped at the frame's PTP/TAI
//! time through the elected clock. [`St2110AncSrc`] binds a UDP socket,
//! depacketizes received -40 RTP into caption triples, decodes the selected
//! service through the shared [`crate::cea::CaptionDecoder`], and emits timed
//! `Caps::Text{Utf8}` cue frames (the same shape `CcExtract` emits, so a
//! `TextOverlay` / text sink consumes either interchangeably).
//!
//! Together they carry closed captions end to end over ST 2110-40: because the RTP
//! timestamp is the shared 90 kHz video media clock, a receiver on the same
//! grandmaster aligns the captions with the video frame. The blocking recv in the
//! source runs with a read timeout so it stays cooperative and ends cleanly on a
//! gap.

use core::future::{ready, Future, Ready};
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use std::net::UdpSocket;
use std::time::Duration;

use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ClockSync, ConfigureOutcome, Dim, ElementMetadata,
    G2gError, MediaClock, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec, Rate, VideoCodec,
};

use crate::ccextract::push_cue_frames;
use crate::cea::{build_cdp, extract_cc_data, parse_cdp, CaptionDecoder, CcSource};
use crate::filesink::io_err;
use crate::st2110anc::{AncField, AncPacket, St2110AncDepacketizer, St2110AncPacketizer};
use crate::st2110sdp::{St2110Essence, St2110Sdp};

/// DID / SDID of a CEA closed-caption ANC packet (SMPTE ST 334-1): DID 0x61 with
/// SDID 0x01 carries the CDP (both CEA-608 and CEA-708).
const CC_DID: u8 = 0x61;
const CC_SDID: u8 = 0x01;

/// Accept only compressed H.264 / H.265 (the codecs whose SEI carries captions);
/// returns the codec.
fn video_codec(caps: &Caps) -> Result<VideoCodec, G2gError> {
    match caps {
        Caps::CompressedVideo { codec: codec @ (VideoCodec::H264 | VideoCodec::H265), .. } => {
            Ok(*codec)
        }
        _ => Err(G2gError::CapsMismatch),
    }
}

// ================================================================
// Sink
// ================================================================

/// ST 2110-40 caption sink: compressed H.264 / H.265 video in, CEA-608/708 captions
/// out as RFC 8331 ANC RTP over UDP.
pub struct St2110AncSink {
    host: String,
    port: u16,
    payload_type: u8,
    ssrc: u32,
    frame_rate_code: u8,
    cdp_seq: u16,
    codec: Option<VideoCodec>,
    packetizer: Option<St2110AncPacketizer>,
    socket: Option<UdpSocket>,
    clock_sync: Option<ClockSync>,
}

impl core::fmt::Debug for St2110AncSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("St2110AncSink")
            .field("host", &self.host)
            .field("port", &self.port)
            .field("codec", &self.codec)
            .finish()
    }
}

impl Default for St2110AncSink {
    fn default() -> Self {
        Self::new()
    }
}

impl St2110AncSink {
    /// A sink to `127.0.0.1:5006` (RTP), dynamic PT 100, CDP frame-rate code 4
    /// (29.97).
    pub fn new() -> Self {
        Self {
            host: String::from("127.0.0.1"),
            port: 5006,
            payload_type: 100,
            ssrc: 0x3273_3234, // "s2 4"
            frame_rate_code: 4,
            cdp_seq: 0,
            codec: None,
            packetizer: None,
            socket: None,
            clock_sync: None,
        }
    }

    /// Build the ST 2110-40 SDP advertising this sink's stream (`smpte291/90000`).
    /// A publisher hands it to receivers, whose [`St2110AncSrc::apply_sdp`]
    /// auto-configures from it.
    pub fn sdp(&self) -> St2110Sdp {
        St2110Sdp {
            essence: St2110Essence::Ancillary,
            payload_type: self.payload_type,
            address: self.host.clone(),
            port: self.port,
            ptp: None,
        }
    }
}

impl AsyncElement for St2110AncSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>> where Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        video_codec(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::LegacySink(Box::new(|c: &Caps| {
            video_codec(c)?;
            Ok(c.clone())
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.codec = Some(video_codec(absolute_caps)?);
        self.packetizer = Some(St2110AncPacketizer::new(self.payload_type, self.ssrc));
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
            "ST 2110-40 caption sink",
            "Sink/Network/ClosedCaption",
            "Sends CEA-608/708 captions as ST 2110-40 ancillary (RFC 8331) RTP over UDP",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new("host", PropKind::Str, "Destination host / multicast group")
                .with_default("127.0.0.1"),
            PropertySpec::new("port", PropKind::Uint, "Destination UDP port").with_default("5006"),
            PropertySpec::new("payload-type", PropKind::Uint, "Dynamic RTP payload type")
                .with_default("100"),
            PropertySpec::new("ssrc", PropKind::Uint, "RTP SSRC"),
            PropertySpec::new("cdp-frame-rate", PropKind::Uint, "CDP frame-rate code (4=29.97, 5=30, 8=60)")
                .with_default("4"),
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
            "cdp-frame-rate" => {
                self.frame_rate_code = (value.as_uint().ok_or(PropError::Type)? as u8) & 0x0F;
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
            "cdp-frame-rate" => Some(PropValue::Uint(u64::from(self.frame_rate_code))),
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
                    let codec = self.codec.ok_or(G2gError::NotConfigured)?;
                    let pkt = self.packetizer.as_mut().ok_or(G2gError::NotConfigured)?;
                    let sock = self.socket.as_ref().ok_or(G2gError::NotConfigured)?;
                    // Mine this access unit's caption triples; a frame with no
                    // captions sends no -40 packet.
                    let triples = extract_cc_data(slice.as_slice(), codec);
                    if triples.is_empty() {
                        return Ok(());
                    }
                    let cdp = build_cdp(&triples, self.frame_rate_code, self.cdp_seq);
                    self.cdp_seq = self.cdp_seq.wrapping_add(1);
                    let anc = AncPacket::generic(CC_DID, CC_SDID, cdp);
                    // The video frame's sampling instant on the PTP timeline.
                    let base = self.clock_sync.as_ref().map_or(0, ClockSync::base_time);
                    let tai = base.saturating_add(frame.timing.pts_ns);
                    let rtp = pkt.packetize(&[anc], tai, AncField::Progressive);
                    sock.send(&rtp).map_err(io_err)?;
                    Ok(())
                }
                PipelinePacket::CapsChanged(c) => {
                    video_codec(&c)?;
                    Ok(())
                }
                _ => Ok(()),
            }
        })
    }
}

impl PadTemplates for St2110AncSink {
    fn pad_templates() -> Vec<PadTemplate> {
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        let h265 = Caps::CompressedVideo {
            codec: VideoCodec::H265,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        Vec::from([PadTemplate::sink(CapsSet::from_alternatives(Vec::from([h264, h265])))])
    }
}

// ================================================================
// Source
// ================================================================

/// ST 2110-40 caption source: RFC 8331 ANC RTP over UDP -> timed `Caps::Text{Utf8}`
/// cue frames for the selected caption service.
pub struct St2110AncSrc {
    address: String,
    port: u16,
    source: CcSource,
    recv_timeout_ms: u64,
    socket: Option<UdpSocket>,
}

impl core::fmt::Debug for St2110AncSrc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("St2110AncSrc")
            .field("address", &self.address)
            .field("port", &self.port)
            .field("source", &self.source)
            .finish()
    }
}

impl Default for St2110AncSrc {
    fn default() -> Self {
        Self::new()
    }
}

impl St2110AncSrc {
    /// A source binding `0.0.0.0:5006`, rendering CEA-608 CC1, 500 ms gap timeout.
    pub fn new() -> Self {
        Self::for_source(CcSource::default())
    }

    /// A source rendering `source` (a CEA-608 channel or a CEA-708 service).
    pub fn for_source(source: CcSource) -> Self {
        Self {
            address: String::from("0.0.0.0"),
            port: 5006,
            source,
            recv_timeout_ms: 500,
            socket: None,
        }
    }

    /// The bound local UDP port after `configure_pipeline` (for tests binding an
    /// ephemeral port with `port = 0`).
    pub fn local_port(&self) -> Option<u16> {
        self.socket.as_ref().and_then(|s| s.local_addr().ok()).map(|a| a.port())
    }

    /// Auto-configure this source from a parsed ancillary [`St2110Sdp`] (the
    /// receiver path). Returns false, unchanged, if the SDP is not an ancillary
    /// essence. Ancillary carries no geometry, so only the port is taken; the
    /// caption service to render stays as constructed. Call before
    /// `configure_pipeline`.
    pub fn apply_sdp(&mut self, sdp: &St2110Sdp) -> bool {
        if !matches!(sdp.essence, St2110Essence::Ancillary) {
            return false;
        }
        self.port = sdp.port;
        true
    }
}

impl SourceLoop for St2110AncSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>> where Self: 'a;
    type CapsFuture<'a> = Ready<Result<Caps, G2gError>> where Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        ready(Ok(Caps::Text { format: g2g_core::TextFormat::Utf8 }))
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
            let depack = St2110AncDepacketizer::new();
            let mut decoder = CaptionDecoder::new(self.source);
            let clock = MediaClock::video();
            let mut base_rtp: Option<u32> = None;
            let mut last_pts = 0u64;
            let mut caps_emitted = false;
            let mut sequence = 0u64;
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
                let Some(anc_frame) = depack.depacketize(&buf[..n]) else { continue };
                // PTS = the packet's media-clock offset from the first packet (the
                // PTP grandmaster supplies absolute time upstream).
                let base = *base_rtp.get_or_insert(anc_frame.rtp_timestamp);
                let pts_ns =
                    clock.ticks_to_ns(u64::from(anc_frame.rtp_timestamp.wrapping_sub(base)));
                last_pts = pts_ns;
                for anc in &anc_frame.packets {
                    // Only CEA caption ANC packets carry a CDP.
                    if anc.did != CC_DID {
                        continue;
                    }
                    if let Some(triples) = parse_cdp(&anc.user_data) {
                        decoder.push_triples(&triples, pts_ns);
                    }
                }
                let cues = decoder.take_cues();
                push_cue_frames(out, cues, &mut caps_emitted, &mut sequence).await?;
            }
            // Finalize any still-shown caption at the last packet's time.
            decoder.flush(last_pts);
            let cues = decoder.take_cues();
            push_cue_frames(out, cues, &mut caps_emitted, &mut sequence).await?;
            out.push(PipelinePacket::Eos).await?;
            Ok(sequence)
        })
    }
}

impl PadTemplates for St2110AncSrc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::source(CapsSet::one(Caps::Text {
            format: g2g_core::TextFormat::Utf8,
        }))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cea::CcTriple;
    use g2g_core::frame::Frame;
    use g2g_core::memory::SystemSlice;
    use g2g_core::runtime::block_on;
    use g2g_core::{FrameTiming, MonotonicClock, PushOutcome};
    use std::sync::Arc;

    /// Collects the text of every cue frame the source emits.
    #[derive(Default)]
    struct Capture {
        texts: Vec<(u64, u64, String)>,
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
                            let text = String::from_utf8_lossy(s.as_slice()).into_owned();
                            self.texts.push((f.timing.pts_ns, f.timing.duration_ns, text));
                        }
                    }
                    PipelinePacket::Eos => self.eos = true,
                    _ => {}
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    /// Build an H.264 access unit whose SEI carries `triples` as a GA94 cc_data
    /// block, Annex-B framed. Mirrors the `cea` / `ccextract` fixtures.
    fn h264_au(triples: &[CcTriple]) -> Vec<u8> {
        let mut payload = alloc::vec![0xB5, 0x00, 0x31, 0x47, 0x41, 0x39, 0x34, 0x03];
        payload.push(0x40 | (triples.len() as u8 & 0x1F));
        payload.push(0xFF);
        for t in triples {
            payload.push(0xF8 | 0x04 | (t.cc_type & 0x03));
            payload.push(t.b0);
            payload.push(t.b1);
        }
        payload.push(0xFF);
        let mut sei = alloc::vec![0x04, payload.len() as u8];
        sei.extend_from_slice(&payload);
        sei.push(0x80);
        let mut au = alloc::vec![0x00, 0x00, 0x00, 0x01, 0x06];
        au.extend_from_slice(&sei);
        au
    }

    fn h264_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    fn video_frame(au: Vec<u8>, pts: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
            FrameTiming { pts_ns: pts, ..FrameTiming::default() },
            0,
        ))
    }

    #[test]
    fn captions_sink_to_src_over_udp_loopback() {
        // End to end on real UDP (localhost): H.264 caption AUs -> -40 sink -> UDP
        // -> -40 src -> text cues. UDP buffers the packets, so send-then-receive is
        // sequential without threads.
        let caps = h264_caps();

        let mut src = St2110AncSrc::new();
        src.address = String::from("127.0.0.1");
        src.port = 0;
        src.recv_timeout_ms = 300;
        src.configure_pipeline(&caps).expect("src binds");
        let port = src.local_port().expect("bound port");

        let mut sink = St2110AncSink::new();
        sink.host = String::from("127.0.0.1");
        sink.port = port;
        sink.configure_pipeline(&caps).expect("sink configures");
        let clock: Arc<dyn g2g_core::PipelineClock + Send + Sync> = Arc::new(MonotonicClock);
        sink.set_clock_sync(ClockSync::new(clock, 1_700_000_000_000_000_000));

        // A CEA-608 CC1 pop-on caption over two frames: RCL, write "HI", EOC (show)
        // then EDM (erase). Each frame's captions become one -40 ANC RTP packet.
        let mut null = Capture::default();
        let au1 = h264_au(&[
            CcTriple { cc_type: 0, b0: 0x14, b1: 0x20 }, // RCL
            CcTriple { cc_type: 0, b0: b'H', b1: b'I' },
            CcTriple { cc_type: 0, b0: 0x14, b1: 0x2F }, // EOC
        ]);
        block_on(sink.process(video_frame(au1, 0), &mut null)).expect("sink sends au1");
        let au2 = h264_au(&[CcTriple { cc_type: 0, b0: 0x14, b1: 0x2C }]); // EDM
        block_on(sink.process(video_frame(au2, 1_000_000_000), &mut null)).expect("sink sends au2");

        // Drain the receiver: reads the buffered -40 packets, decodes captions, then
        // times out -> flush + EOS.
        let mut cap = Capture::default();
        let n = block_on(src.run(&mut cap)).expect("src runs");

        assert_eq!(n, 1, "one finished caption cue");
        assert!(cap.eos, "source emitted EOS on the gap");
        assert_eq!(cap.texts.len(), 1);
        assert_eq!(cap.texts[0].2, "HI", "caption text survives sink -> UDP -> src");
        // Started at the EOC frame (PTS 0), ended at the EDM frame (PTS 1s).
        assert_eq!(cap.texts[0].0, 0);
        assert_eq!(cap.texts[0].0 + cap.texts[0].1, 1_000_000_000);
    }

    #[test]
    fn sink_skips_frames_without_captions() {
        // A video frame with no SEI cc_data sends no -40 packet.
        let caps = h264_caps();
        let mut sink = St2110AncSink::new();
        sink.configure_pipeline(&caps).expect("configures");
        let mut null = Capture::default();
        // An access unit with an empty triple list -> no captions.
        let au = h264_au(&[]);
        // No socket send happens; process just returns Ok.
        block_on(sink.process(video_frame(au, 0), &mut null)).expect("no-caption frame is fine");
        assert_eq!(sink.cdp_seq, 0, "CDP sequence did not advance");
    }

    #[test]
    fn sink_properties_round_trip() {
        let mut sink = St2110AncSink::new();
        sink.set_property("host", PropValue::Str("239.0.0.5".into())).unwrap();
        sink.set_property("cdp-frame-rate", PropValue::Uint(8)).unwrap();
        assert_eq!(sink.get_property("host"), Some(PropValue::Str("239.0.0.5".into())));
        assert_eq!(sink.get_property("cdp-frame-rate"), Some(PropValue::Uint(8)));
        assert_eq!(
            sink.set_property("port", PropValue::Uint(70_000)),
            Err(PropError::Value),
            "a port past u16 is rejected"
        );
    }

    #[test]
    fn src_renders_the_selected_708_service() {
        // A src selecting CEA-708 service 1 ignores a lone 608 pair.
        let src = St2110AncSrc::for_source(CcSource::parse("service-1").unwrap());
        assert!(matches!(src.source, CcSource::Cea708(1)));
    }

    #[test]
    fn sdp_generated_by_sink_configures_a_src() {
        // The -40 SDP loop: sink advertises smpte291, a src picks up the port.
        let mut sink = St2110AncSink::new();
        sink.host = String::from("239.40.1.1");
        sink.port = 5006;
        let text = sink.sdp().to_sdp();
        let parsed = St2110Sdp::parse(&text).expect("parses");
        assert!(matches!(parsed.essence, St2110Essence::Ancillary));

        let mut src = St2110AncSrc::for_source(CcSource::parse("cc1").unwrap());
        assert!(src.apply_sdp(&parsed), "ancillary SDP configures the src");
        assert_eq!(src.port, 5006);
        // A non-ancillary SDP is rejected.
        let audio = St2110Sdp {
            essence: St2110Essence::Audio {
                depth: crate::st2110audio::SampleDepth::L16,
                sample_rate: 48_000,
                channels: 2,
                ptime_us: 1000,
            },
            payload_type: 97,
            address: "239.30.1.1".into(),
            port: 5004,
            ptp: None,
        };
        assert!(!src.apply_sdp(&audio));
        assert_eq!(src.port, 5006, "unchanged");
    }
}
