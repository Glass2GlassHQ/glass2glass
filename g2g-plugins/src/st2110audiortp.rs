//! ST 2110-30 audio network elements (M597): a sink and source that put the
//! sans-IO `st2110audio` core on the wire over UDP, with RTP timestamps from the
//! PTP media clock.
//!
//! [`St2110AudioSink`] takes PCM `DataFrame`s, maps each frame's PTS through the
//! elected clock to a PTP/TAI sampling instant, packetizes with the ST 2110-30
//! core, and sends the RTP packets to a UDP destination (typically a multicast
//! group). [`St2110AudioSrc`] binds a UDP socket, depacketizes received RTP into
//! PCM `DataFrame`s, and reconstructs each frame's PTS from the RTP timestamp.
//! Together they make a full ST 2110-30 sender/receiver; the audio essence now
//! runs end to end on the wire, and, because the timestamp is the shared media
//! clock, a receiver on the same grandmaster stays in sync with the source.
//!
//! `PcmS16Le` rides the wire as L16 and `PcmF32Le` as L24 (float carries more than
//! 16 bits, so it maps to the 24-bit wire, scaled from [-1, 1]). The blocking recv
//! in the source runs with a read timeout so it stays cooperative and ends cleanly
//! on a silence gap.

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
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ClockSync, ConfigureOutcome,
    ElementMetadata, FrameTiming, G2gError, MediaClock, MemoryDomain, OutputSink, PadTemplate,
    PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::filesink::io_err;
use crate::st2110audio::{SampleDepth, St2110AudioDepacketizer, St2110AudioPacketizer};
use crate::st2110sdp::{St2110Essence, St2110Sdp};

/// The supported PCM formats and their (format, channels, sample_rate). S16 rides
/// the wire as L16, F32 as L24 (see [`wire_depth`]).
fn audio_params(caps: &Caps) -> Result<(AudioFormat, u8, u32), G2gError> {
    match caps {
        Caps::Audio {
            format:
                format @ (AudioFormat::PcmS16Le | AudioFormat::PcmF32Le | AudioFormat::PcmS24Le),
            channels,
            sample_rate,
        } => Ok((*format, *channels, *sample_rate)),
        _ => Err(G2gError::CapsMismatch),
    }
}

/// The ST 2110-30 wire depth a PCM format maps to: S16 -> L16, F32 / S24 -> L24
/// (float carries more than 16 bits of precision, S24 is natively 24-bit, so both
/// ride the 24-bit wire).
fn wire_depth(format: AudioFormat) -> Option<SampleDepth> {
    match format {
        AudioFormat::PcmS16Le => Some(SampleDepth::L16),
        AudioFormat::PcmF32Le | AudioFormat::PcmS24Le => Some(SampleDepth::L24),
        _ => None,
    }
}

/// Interleaved-PCM bytes of `format` -> the `i32` samples the -30 core packetizes:
/// S16 sign-extends; F32 [-1, 1] scales to signed 24-bit (clamped so +1.0 does not
/// wrap to full-scale negative).
fn pcm_to_samples(format: AudioFormat, bytes: &[u8]) -> Vec<i32> {
    match format {
        AudioFormat::PcmS16Le => {
            bytes.chunks_exact(2).map(|c| i32::from(i16::from_le_bytes([c[0], c[1]]))).collect()
        }
        AudioFormat::PcmF32Le => bytes
            .chunks_exact(4)
            .map(|c| {
                let f = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                (f * 8_388_608.0).clamp(-8_388_608.0, 8_388_607.0) as i32
            })
            .collect(),
        // 24-bit little-endian, sign-extended from the top byte.
        AudioFormat::PcmS24Le => bytes
            .chunks_exact(3)
            .map(|c| i32::from(c[0]) | (i32::from(c[1]) << 8) | (i32::from(c[2] as i8) << 16))
            .collect(),
        _ => Vec::new(),
    }
}

/// The inverse of [`pcm_to_samples`]: `i32` samples -> interleaved-PCM bytes.
fn samples_to_pcm(format: AudioFormat, samples: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * bytes_per_sample(format));
    match format {
        AudioFormat::PcmS16Le => {
            for &s in samples {
                out.extend_from_slice(&(s as i16).to_le_bytes());
            }
        }
        AudioFormat::PcmF32Le => {
            for &s in samples {
                out.extend_from_slice(&((s as f32) / 8_388_608.0).to_le_bytes());
            }
        }
        AudioFormat::PcmS24Le => {
            for &s in samples {
                // Low 24 bits, little-endian (3 bytes).
                out.extend_from_slice(&[s as u8, (s >> 8) as u8, (s >> 16) as u8]);
            }
        }
        _ => {}
    }
    out
}

/// Bytes per sample of a supported PCM format.
fn bytes_per_sample(format: AudioFormat) -> usize {
    match format {
        AudioFormat::PcmS16Le => 2,
        AudioFormat::PcmF32Le => 4,
        AudioFormat::PcmS24Le => 3,
        _ => 0,
    }
}

/// Sample-frames per packet for a packet time in microseconds (e.g. 1000 = 1 ms).
fn frames_per_packet(sample_rate: u32, ptime_us: u32) -> usize {
    ((u64::from(sample_rate) * u64::from(ptime_us)) / 1_000_000).max(1) as usize
}

// ================================================================
// Sink
// ================================================================

/// ST 2110-30 audio sink: PCM `DataFrame`s -> RTP over UDP.
pub struct St2110AudioSink {
    host: String,
    port: u16,
    payload_type: u8,
    ssrc: u32,
    ptime_us: u32,
    format: Option<AudioFormat>,
    packetizer: Option<St2110AudioPacketizer>,
    socket: Option<UdpSocket>,
    clock_sync: Option<ClockSync>,
    caps: Option<Caps>,
}

impl core::fmt::Debug for St2110AudioSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("St2110AudioSink")
            .field("host", &self.host)
            .field("port", &self.port)
            .field("ptime_us", &self.ptime_us)
            .finish()
    }
}

impl Default for St2110AudioSink {
    fn default() -> Self {
        Self::new()
    }
}

impl St2110AudioSink {
    /// A sink to `127.0.0.1:5004` (RTP), dynamic PT 97, 1 ms packets.
    pub fn new() -> Self {
        Self {
            host: String::from("127.0.0.1"),
            port: 5004,
            payload_type: 97,
            ssrc: 0x3273_3230, // "s2 0"
            ptime_us: 1000,
            format: None,
            packetizer: None,
            socket: None,
            clock_sync: None,
            caps: None,
        }
    }

    /// Build the ST 2110-30 SDP advertising this sink's stream (`None` until
    /// configured). A publisher hands it to receivers, whose
    /// [`St2110AudioSrc::apply_sdp`] auto-configures from it.
    pub fn sdp(&self) -> Option<St2110Sdp> {
        let format = self.format?;
        let (_f, channels, sample_rate) = audio_params(self.caps.as_ref()?).ok()?;
        Some(St2110Sdp {
            essence: St2110Essence::Audio {
                depth: wire_depth(format)?,
                sample_rate,
                channels: u16::from(channels),
                ptime_us: self.ptime_us,
            },
            payload_type: self.payload_type,
            address: self.host.clone(),
            port: self.port,
            ptp: None,
        })
    }
}

impl AsyncElement for St2110AudioSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>> where Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        audio_params(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::LegacySink(Box::new(|c: &Caps| {
            audio_params(c)?;
            Ok(c.clone())
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (format, channels, sample_rate) = audio_params(absolute_caps)?;
        let depth = wire_depth(format).ok_or(G2gError::CapsMismatch)?;
        let fpp = frames_per_packet(sample_rate, self.ptime_us);
        self.format = Some(format);
        self.packetizer = Some(St2110AudioPacketizer::new(
            self.payload_type,
            self.ssrc,
            sample_rate,
            u16::from(channels),
            depth,
            fpp,
        ));
        let sock = UdpSocket::bind(("0.0.0.0", 0)).map_err(io_err)?;
        sock.connect((self.host.as_str(), self.port)).map_err(io_err)?;
        self.socket = Some(sock);
        self.caps = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.clock_sync = Some(sync);
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "ST 2110-30 audio sink",
            "Sink/Network",
            "Sends PCM as ST 2110-30 (AES67) RTP over UDP",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new("host", PropKind::Str, "Destination host / multicast group")
                .with_default("127.0.0.1"),
            PropertySpec::new("port", PropKind::Uint, "Destination UDP port").with_default("5004"),
            PropertySpec::new("payload-type", PropKind::Uint, "Dynamic RTP payload type")
                .with_default("97"),
            PropertySpec::new("ssrc", PropKind::Uint, "RTP SSRC"),
            PropertySpec::new("ptime-us", PropKind::Uint, "Packet time in microseconds")
                .with_default("1000"),
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
            "ptime-us" => {
                self.ptime_us = value.as_uint().ok_or(PropError::Type)? as u32;
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
            "ptime-us" => Some(PropValue::Uint(u64::from(self.ptime_us))),
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
                    let format = self.format.ok_or(G2gError::NotConfigured)?;
                    let pkt = self.packetizer.as_mut().ok_or(G2gError::NotConfigured)?;
                    let sock = self.socket.as_ref().ok_or(G2gError::NotConfigured)?;
                    // Interleaved PCM bytes -> i32 samples for the -30 core (S16 or F32).
                    let samples = pcm_to_samples(format, slice.as_slice());
                    // The sampling instant on the PTP timeline: base time + the
                    // frame's running time (its PTS). Without an elected clock,
                    // fall back to the PTS directly.
                    let base = self.clock_sync.as_ref().map_or(0, ClockSync::base_time);
                    let tai = base.saturating_add(frame.timing.pts_ns);
                    for p in pkt.packetize(&samples, tai) {
                        sock.send(&p).map_err(io_err)?;
                    }
                    Ok(())
                }
                PipelinePacket::CapsChanged(c) => {
                    audio_params(&c)?;
                    Ok(())
                }
                _ => Ok(()),
            }
        })
    }
}

impl PadTemplates for St2110AudioSink {
    fn pad_templates() -> Vec<PadTemplate> {
        let alts = [AudioFormat::PcmS16Le, AudioFormat::PcmF32Le, AudioFormat::PcmS24Le]
            .map(|format| Caps::Audio { format, channels: 2, sample_rate: 48_000 })
            .to_vec();
        Vec::from([PadTemplate::sink(CapsSet::from_alternatives(alts))])
    }
}

// ================================================================
// Source
// ================================================================

/// ST 2110-30 audio source: RTP over UDP -> PCM `DataFrame`s.
pub struct St2110AudioSrc {
    address: String,
    port: u16,
    format: AudioFormat,
    channels: u8,
    sample_rate: u32,
    recv_timeout_ms: u64,
    socket: Option<UdpSocket>,
}

impl core::fmt::Debug for St2110AudioSrc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("St2110AudioSrc")
            .field("address", &self.address)
            .field("port", &self.port)
            .field("format", &self.format)
            .field("channels", &self.channels)
            .field("sample_rate", &self.sample_rate)
            .finish()
    }
}

impl Default for St2110AudioSrc {
    fn default() -> Self {
        Self::new()
    }
}

impl St2110AudioSrc {
    /// A source binding `0.0.0.0:5004`, stereo 48 kHz, 500 ms silence-gap timeout.
    pub fn new() -> Self {
        Self {
            address: String::from("0.0.0.0"),
            port: 5004,
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
            recv_timeout_ms: 500,
            socket: None,
        }
    }

    /// The bound local UDP port after `configure_pipeline` (for tests binding an
    /// ephemeral port with `port = 0`).
    pub fn local_port(&self) -> Option<u16> {
        self.socket.as_ref().and_then(|s| s.local_addr().ok()).map(|a| a.port())
    }

    /// Auto-configure this source from a parsed audio [`St2110Sdp`] (the receiver
    /// path). Returns false, unchanged, if the SDP is not an audio essence. The wire
    /// depth picks the output PCM format (L16 -> S16, L24 -> F32). Call before
    /// `configure_pipeline`.
    pub fn apply_sdp(&mut self, sdp: &St2110Sdp) -> bool {
        let St2110Essence::Audio { depth, sample_rate, channels, .. } = &sdp.essence else {
            return false;
        };
        self.format = match depth {
            SampleDepth::L16 => AudioFormat::PcmS16Le,
            SampleDepth::L24 => AudioFormat::PcmF32Le,
        };
        self.sample_rate = *sample_rate;
        self.channels = *channels as u8;
        self.port = sdp.port;
        true
    }

    fn caps(&self) -> Caps {
        Caps::Audio {
            format: self.format,
            channels: self.channels,
            sample_rate: self.sample_rate,
        }
    }
}

impl SourceLoop for St2110AudioSrc {
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
            let depth = wire_depth(self.format).ok_or(G2gError::CapsMismatch)?;
            let depack = St2110AudioDepacketizer::new(u16::from(self.channels), depth);
            let clock = MediaClock::audio(self.sample_rate);
            let mut base_rtp: Option<u32> = None;
            let mut count = 0u64;
            let mut buf = [0u8; 65_536];
            loop {
                let n = match sock.recv_from(&mut buf) {
                    Ok((n, _)) => n,
                    // A silence gap (read timeout) ends the stream cleanly.
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
                let Some(p) = depack.depacketize(&buf[..n]) else { continue };
                // PTS = the packet's media-clock offset from the first packet, so
                // the receiver reproduces the sender's timing without needing a
                // shared epoch here (the PTP clock supplies absolute time upstream).
                let base = *base_rtp.get_or_insert(p.rtp_timestamp);
                let pts_ns = clock.ticks_to_ns(u64::from(p.rtp_timestamp.wrapping_sub(base)));

                let bytes = samples_to_pcm(self.format, &p.samples);
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
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

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::runtime::block_on;
    use g2g_core::{PushOutcome, MonotonicClock};
    use std::sync::Arc;

    /// Collects the samples of every DataFrame the source emits.
    #[derive(Default)]
    struct Capture {
        frames: Vec<Vec<i16>>,
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
                            self.frames.push(
                                s.as_slice()
                                    .chunks_exact(2)
                                    .map(|c| i16::from_le_bytes([c[0], c[1]]))
                                    .collect(),
                            );
                        }
                    }
                    PipelinePacket::Eos => self.eos = true,
                    _ => {}
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    fn pcm_frame(samples: &[i16], pts_ns: u64) -> PipelinePacket {
        let mut bytes = Vec::new();
        for &s in samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            timing: FrameTiming { pts_ns, ..FrameTiming::default() },
            sequence: 0,
            meta: Default::default(),
        })
    }

    #[test]
    fn audio_sink_to_src_over_udp_loopback() {
        // End to end on real UDP (localhost): sink -> UDP -> src. UDP buffers the
        // packets, so we can send then receive sequentially, no threads.
        let caps = Caps::Audio { format: AudioFormat::PcmS16Le, channels: 2, sample_rate: 48_000 };

        // Receiver binds an ephemeral port.
        let mut src = St2110AudioSrc::new();
        src.address = String::from("127.0.0.1");
        src.port = 0;
        src.recv_timeout_ms = 300;
        src.configure_pipeline(&caps).expect("src binds");
        let port = src.local_port().expect("bound port");

        // Sender aims at it. A fake elected clock so PTS maps to a PTP instant.
        let mut sink = St2110AudioSink::new();
        sink.host = String::from("127.0.0.1");
        sink.port = port;
        sink.configure_pipeline(&caps).expect("sink configures");
        let clock: Arc<dyn g2g_core::PipelineClock + Send + Sync> = Arc::new(MonotonicClock);
        sink.set_clock_sync(ClockSync::new(clock, 1_700_000_000_000_000_000));

        // 96 stereo frames (2 ms at 48 kHz -> 2 packets of 48 frames) at PTS 0.
        let input: Vec<i16> = (0..96i16).flat_map(|i| [i * 200 - 3000, -(i * 91) - 1]).collect();
        let mut null = Capture::default();
        block_on(sink.process(pcm_frame(&input, 0), &mut null)).expect("sink sends");

        // Drain the receiver: it reads the buffered packets, then times out -> EOS.
        let mut cap = Capture::default();
        let n = block_on(src.run(&mut cap)).expect("src runs");

        assert_eq!(n, 2, "two 48-frame packets received");
        assert!(cap.eos, "source emitted EOS on the silence gap");
        let received: Vec<i16> = cap.frames.concat();
        assert_eq!(received, input, "PCM survives sink -> UDP -> src");
    }

    #[test]
    fn sdp_generated_by_sink_configures_a_src() {
        // Full out-of-band loop for -30 audio: sink SDP -> text -> parse -> src.
        let caps = Caps::Audio { format: AudioFormat::PcmF32Le, channels: 2, sample_rate: 48_000 };
        let mut sink = St2110AudioSink::new();
        sink.host = String::from("239.30.1.1");
        sink.port = 5004;
        sink.ptime_us = 125;
        sink.configure_pipeline(&caps).expect("configures");

        let text = sink.sdp().expect("configured").to_sdp();
        let parsed = crate::st2110sdp::St2110Sdp::parse(&text).expect("parses");

        let mut src = St2110AudioSrc::new();
        assert!(src.apply_sdp(&parsed), "audio SDP configures the src");
        // L24 (from F32) -> the src outputs F32; geometry matches.
        assert_eq!(src.format, AudioFormat::PcmF32Le);
        assert_eq!(src.channels, 2);
        assert_eq!(src.sample_rate, 48_000);
        assert_eq!(src.port, 5004);
        // A non-audio SDP is rejected, leaving the src unchanged.
        let video = crate::st2110sdp::St2110Sdp {
            essence: St2110Essence::Ancillary,
            payload_type: 100,
            address: "239.1.1.1".into(),
            port: 6000,
            ptp: None,
        };
        assert!(!src.apply_sdp(&video));
        assert_eq!(src.port, 5004, "unchanged");
    }

    #[test]
    fn f32_audio_rides_l24_over_udp_loopback() {
        // Float PCM maps to the 24-bit wire (L24). A byte-capturing sink so we can
        // read back f32 samples and check they survive within the 24-bit quantum.
        #[derive(Default)]
        struct RawCapture {
            bytes: Vec<u8>,
        }
        impl OutputSink for RawCapture {
            fn push<'a>(
                &'a mut self,
                packet: PipelinePacket,
            ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
                Box::pin(async move {
                    if let PipelinePacket::DataFrame(f) = packet {
                        if let MemoryDomain::System(s) = &f.domain {
                            self.bytes.extend_from_slice(s.as_slice());
                        }
                    }
                    Ok(PushOutcome::Accepted)
                })
            }
        }

        let caps = Caps::Audio { format: AudioFormat::PcmF32Le, channels: 2, sample_rate: 48_000 };
        let mut src = St2110AudioSrc::new();
        src.address = String::from("127.0.0.1");
        src.port = 0;
        src.format = AudioFormat::PcmF32Le;
        src.recv_timeout_ms = 300;
        src.configure_pipeline(&caps).expect("src binds");
        let port = src.local_port().expect("bound port");

        let mut sink = St2110AudioSink::new();
        sink.host = String::from("127.0.0.1");
        sink.port = port;
        sink.configure_pipeline(&caps).expect("sink configures");

        // Stereo float samples across the range incl the extremes.
        let input: Vec<f32> =
            (0..48).flat_map(|i| [(i as f32 / 47.0) * 2.0 - 1.0, -(i as f32 / 47.0)]).collect();
        let mut bytes = Vec::new();
        for &s in &input {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let frame = PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            timing: FrameTiming { pts_ns: 0, ..FrameTiming::default() },
            sequence: 0,
            meta: Default::default(),
        });
        let mut null = Capture::default();
        block_on(sink.process(frame, &mut null)).expect("sink sends");

        let mut cap = RawCapture::default();
        block_on(src.run(&mut cap)).expect("src runs");
        let received: Vec<f32> =
            cap.bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert_eq!(received.len(), input.len(), "all float samples returned");
        for (got, want) in received.iter().zip(&input) {
            // Round-trip through 24-bit is lossy by at most one 24-bit quantum.
            assert!((got - want).abs() < 1e-6, "F32 survives L24 round trip: {got} vs {want}");
        }
    }

    #[test]
    fn s24_integer_audio_rides_l24_over_udp_loopback() {
        // 24-bit integer PCM maps to L24 with no float detour, so it survives the
        // round trip *exactly* (unlike F32, which is lossy by up to one quantum).
        let caps = Caps::Audio { format: AudioFormat::PcmS24Le, channels: 2, sample_rate: 48_000 };
        let mut src = St2110AudioSrc::new();
        src.address = String::from("127.0.0.1");
        src.port = 0;
        src.format = AudioFormat::PcmS24Le;
        src.recv_timeout_ms = 300;
        src.configure_pipeline(&caps).expect("src binds");
        let port = src.local_port().expect("bound port");

        let mut sink = St2110AudioSink::new();
        sink.host = String::from("127.0.0.1");
        sink.port = port;
        sink.configure_pipeline(&caps).expect("sink configures");

        // Stereo 24-bit samples incl the signed extremes (+/- full scale) and zero.
        let samples: Vec<i32> = [0, 1, -1, 8_388_607, -8_388_608, 12_345, -6_000, 100_000]
            .into_iter()
            .cycle()
            .take(96)
            .collect();
        let mut bytes = Vec::new();
        for &s in &samples {
            bytes.extend_from_slice(&[s as u8, (s >> 8) as u8, (s >> 16) as u8]);
        }
        let frame = PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.clone().into_boxed_slice())),
            timing: FrameTiming { pts_ns: 0, ..FrameTiming::default() },
            sequence: 0,
            meta: Default::default(),
        });
        let mut null = Capture::default();
        block_on(sink.process(frame, &mut null)).expect("sink sends");

        #[derive(Default)]
        struct RawCapture {
            bytes: Vec<u8>,
        }
        impl OutputSink for RawCapture {
            fn push<'a>(
                &'a mut self,
                packet: PipelinePacket,
            ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
                Box::pin(async move {
                    if let PipelinePacket::DataFrame(f) = packet {
                        if let MemoryDomain::System(s) = &f.domain {
                            self.bytes.extend_from_slice(s.as_slice());
                        }
                    }
                    Ok(PushOutcome::Accepted)
                })
            }
        }
        let mut cap = RawCapture::default();
        block_on(src.run(&mut cap)).expect("src runs");
        assert_eq!(cap.bytes, bytes, "24-bit integer PCM survives S24 -> L24 -> S24 byte-exact");
    }

    #[test]
    fn sink_properties_round_trip() {
        let mut sink = St2110AudioSink::new();
        sink.set_property("host", PropValue::Str("239.0.0.1".into())).unwrap();
        sink.set_property("port", PropValue::Uint(5004)).unwrap();
        sink.set_property("ptime-us", PropValue::Uint(125)).unwrap();
        assert_eq!(sink.get_property("host"), Some(PropValue::Str("239.0.0.1".into())));
        assert_eq!(sink.get_property("ptime-us"), Some(PropValue::Uint(125)));
        // 125 us at 48 kHz = 6 sample-frames per packet.
        assert_eq!(frames_per_packet(48_000, 125), 6);
        assert_eq!(
            sink.set_property("port", PropValue::Uint(70_000)),
            Err(PropError::Value),
            "a port past u16 is rejected"
        );
    }
}
