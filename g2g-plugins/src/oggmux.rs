//! Ogg muxer element (M789): one audio elementary stream in
//! (`Caps::Audio{Opus|Vorbis|Flac}`), an Ogg byte stream out
//! (`Caps::ByteStream{Ogg}`).
//!
//! Wraps the pure [`crate::ogg::OggPageWriter`], the inverse of
//! [`crate::oggdemux::OggDemux`]: the codec's header packets go on their own
//! pages up front (granule 0), then audio packets are framed into data pages
//! whose granule position is the codec mapping's running sample count. CPU,
//! `no_std` baseline.
//!
//! ```text
//! ... ! opusenc ! oggmux ! filesink location=out.opus
//! filesrc location=in.ogg ! oggdemux stream=vorbis ! oggmux ! filesink location=out.ogg
//! ```
//!
//! Codec mappings:
//! - **Opus** (RFC 7845): an in-band `OpusHead` (a remux from `oggdemux`) is
//!   carried verbatim, otherwise one is synthesized from the caps. Granule is
//!   the cumulative 48 kHz sample count, pre-skip included.
//! - **Vorbis**: remux only (g2g has no Vorbis encoder). The three headers
//!   arrive in-band; ident goes alone on the beginning-of-stream page, comment
//!   and setup on the next. Granule is the lapped `(prev + cur) / 4` sample
//!   count from the [`crate::ogg::VorbisTiming`] mode tables, the inverse of
//!   the demux-side durations.
//! - **FLAC**: the in-band `fLaC` header becomes the mapping's `\x7fFLAC`
//!   beginning-of-stream packet; granule is the cumulative block-size count.
//!
//! A frame that declares a `duration_ns` shorter than the packet's natural
//! sample count caps that packet's granule contribution, so a remux reproduces
//! the source's end-of-stream trim exactly.
//!
//! Scope (v1): one logical bitstream (a single input pad), mirroring the
//! single-stream `OggDemux`. Multi-stream (grouped) Ogg is a follow-up.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec,
};

use crate::ogg::{OggPageWriter, VorbisTiming};
use crate::opusparse::{packet_samples as opus_packet_samples, OPUS_RATE_HZ};

/// Default logical-bitstream serial number, overridable via the `serial`
/// property. Any value identifies the stream as long as it is unique within the
/// file, and this muxer writes one stream.
const DEFAULT_SERIAL: u32 = 0x6732_6732; // "g2g2"

/// Pre-skip written into a synthesized `OpusHead`: libopus' encoder lookahead at
/// 48 kHz, which is what [`crate::opusenc::OpusEnc`] (and ffmpeg's libopus
/// wrapper) actually delays its output by. A remuxed stream carries the source
/// header instead, so this only applies to a freshly encoded one.
const OPUS_ENCODER_PRE_SKIP: u16 = 312;

/// The vendor string written into synthesized comment headers.
const VENDOR: &[u8] = b"g2g";

/// Muxes one audio elementary stream into an Ogg byte stream.
#[derive(Debug)]
pub struct OggMux {
    writer: OggPageWriter,
    serial: u32,
    /// The input codec, set at configure.
    format: Option<AudioFormat>,
    channels: u8,
    sample_rate: u32,
    configured: bool,
    emitted: u64,
    /// Header packets collected in-band, in arrival order.
    headers: Vec<Vec<u8>>,
    /// Whether the header pages have been written (done lazily, so every in-band
    /// header has landed first).
    headers_written: bool,
    /// Cumulative samples the audio packets decode to, per the codec mapping.
    natural_samples: u64,
    /// Cumulative `duration_ns` the input frames declared.
    declared_ns: u64,
    /// Whether upstream left an audio packet untimed, which voids `declared_ns`
    /// as a bound on the granule position.
    untimed: bool,
    /// Vorbis packet-duration tables, parsed once ident + setup have landed.
    vorbis: Option<VorbisTiming>,
    /// The previous Vorbis packet's block size. The first audio packet has no
    /// predecessor to lap against, so it decodes to nothing and adds 0.
    prev_blocksize: Option<u32>,
}

impl Default for OggMux {
    fn default() -> Self {
        Self::new()
    }
}

impl OggMux {
    pub fn new() -> Self {
        Self {
            writer: OggPageWriter::new(DEFAULT_SERIAL),
            serial: DEFAULT_SERIAL,
            format: None,
            channels: 0,
            sample_rate: 0,
            configured: false,
            emitted: 0,
            headers: Vec::new(),
            headers_written: false,
            natural_samples: 0,
            declared_ns: 0,
            untimed: false,
            vorbis: None,
            prev_blocksize: None,
        }
    }

    /// Set the logical bitstream's serial number.
    pub fn with_serial(mut self, serial: u32) -> Self {
        self.serial = serial;
        self.writer = OggPageWriter::new(serial);
        self
    }

    /// Count of Ogg byte frames forwarded.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// The output it produces: an Ogg byte stream.
    fn output_caps() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Ogg,
        }
    }

    /// The elementary streams this muxer accepts on its sink pad.
    fn input_alternatives() -> Vec<Caps> {
        Vec::from(
            [AudioFormat::Opus, AudioFormat::Vorbis, AudioFormat::Flac].map(|format| Caps::Audio {
                format,
                channels: 0,
                sample_rate: 0,
            }),
        )
    }

    /// The codec of `caps`, or `None` for a stream Ogg cannot carry here.
    fn format_of(caps: &Caps) -> Option<AudioFormat> {
        match caps {
            Caps::Audio {
                format: f @ (AudioFormat::Opus | AudioFormat::Vorbis | AudioFormat::Flac),
                ..
            } => Some(*f),
            _ => None,
        }
    }

    /// Whether `packet` is codec config rather than audio, for the configured
    /// codec's in-band convention.
    fn is_header(&self, packet: &[u8]) -> bool {
        match self.format {
            // `OpusHead` / `OpusTags` (RFC 7845); audio packets start with a TOC.
            Some(AudioFormat::Opus) => {
                packet.starts_with(b"OpusHead") || packet.starts_with(b"OpusTags")
            }
            // Vorbis header packets have bit 0 of the packet type set and carry
            // the "vorbis" magic; audio packets have it clear.
            Some(AudioFormat::Vorbis) => {
                packet.len() > 7 && packet[0] & 1 == 1 && &packet[1..7] == b"vorbis"
            }
            // The native `fLaC` marker + metadata blocks; audio frames start on
            // the 0xFF sync.
            Some(AudioFormat::Flac) => packet.starts_with(b"fLaC"),
            _ => false,
        }
    }

    /// The header packet with the given Vorbis packet type (1 ident, 3 comment,
    /// 5 setup), if it arrived in-band.
    fn vorbis_header(&self, kind: u8) -> Option<&[u8]> {
        self.headers
            .iter()
            .find(|h| h.first() == Some(&kind) && h[1..].starts_with(b"vorbis"))
            .map(|h| h.as_slice())
    }

    /// The header packets that go before the audio: the beginning-of-stream
    /// packet, then the rest (which share the following page). `None` when the
    /// codec's mandatory in-band config has not arrived.
    fn header_packets(&self) -> Option<(Vec<u8>, Vec<Vec<u8>>)> {
        match self.format? {
            AudioFormat::Opus => {
                let head = self
                    .headers
                    .iter()
                    .find(|h| h.starts_with(b"OpusHead"))
                    .cloned()
                    .unwrap_or_else(|| opus_head(self.channels, self.sample_rate));
                let tags = self
                    .headers
                    .iter()
                    .find(|h| h.starts_with(b"OpusTags"))
                    .cloned()
                    .unwrap_or_else(|| vorbis_comment(b"OpusTags"));
                Some((head, Vec::from([tags])))
            }
            AudioFormat::Vorbis => {
                // Remux only: without the codebooks there is no decodable stream
                // to write, so the ident and setup headers must arrive in-band.
                let ident = self.vorbis_header(1)?.to_vec();
                let setup = self.vorbis_header(5)?.to_vec();
                let comment = self
                    .vorbis_header(3)
                    .map(|c| c.to_vec())
                    .unwrap_or_else(|| vorbis_comment(b"\x03vorbis"));
                Some((ident, Vec::from([comment, setup])))
            }
            AudioFormat::Flac => flac_headers(self.headers.first()?),
            _ => None,
        }
    }

    /// Write the header pages: the beginning-of-stream packet alone on the first
    /// page, the remaining headers on the next, both at granule 0 (the mappings
    /// require audio to start on a fresh page).
    fn write_headers(&mut self) -> Result<Vec<u8>, G2gError> {
        let (bos, rest) = self.header_packets().ok_or(G2gError::NotConfigured)?;
        // Vorbis timing needs the mode tables from ident + setup.
        if self.format == Some(AudioFormat::Vorbis) {
            self.vorbis = match (self.vorbis_header(1), self.vorbis_header(5)) {
                (Some(i), Some(s)) => VorbisTiming::parse(i, s),
                _ => None,
            };
        }
        let mut out = Vec::new();
        self.writer.push_packet(bos, 0);
        out.extend_from_slice(&self.writer.flush(false));
        for packet in rest {
            self.writer.push_packet(packet, 0);
        }
        out.extend_from_slice(&self.writer.flush(false));
        self.headers_written = true;
        Ok(out)
    }

    /// Advance the granule position by `packet`, returning where the stream now
    /// stands. The natural sample count comes from the codec mapping; when
    /// upstream times every packet, the position is also held to the total
    /// duration declared so far, which is how a remux inherits the source's
    /// end-of-stream trim (the tail of a Vorbis / Opus stream is padding the
    /// encoder wrote but the granule does not cover).
    fn advance_granule(&mut self, packet: &[u8], duration_ns: u64) -> u64 {
        let natural = match self.format {
            Some(AudioFormat::Opus) => u64::from(opus_packet_samples(packet)),
            Some(AudioFormat::Flac) => crate::flacparse::parse_frame_header(packet)
                .map(|h| u64::from(h.block_size))
                .unwrap_or(0),
            // The lapped Vorbis overlap: packet n outputs the second half of
            // window n-1 plus the first half of window n, and the stream's first
            // packet (no predecessor) outputs nothing.
            Some(AudioFormat::Vorbis) => {
                match self
                    .vorbis
                    .as_ref()
                    .and_then(|t| t.packet_blocksize(packet))
                {
                    Some(cur) => {
                        let step = match self.prev_blocksize {
                            Some(prev) => u64::from((prev + cur) / 4),
                            None => 0,
                        };
                        self.prev_blocksize = Some(cur);
                        step
                    }
                    None => 0,
                }
            }
            _ => 0,
        };
        self.natural_samples = self.natural_samples.saturating_add(natural);
        // A packet that decodes to something but declares no duration means
        // upstream is not timing this stream, so its running total says nothing.
        if duration_ns == 0 && natural > 0 {
            self.untimed = true;
        }
        self.declared_ns = self.declared_ns.saturating_add(duration_ns);
        match declared_samples(self.declared_ns, self.granule_rate()) {
            Some(declared) if !self.untimed => self.natural_samples.min(declared),
            _ => self.natural_samples,
        }
    }

    /// The sample rate the granule position counts in: 48 kHz for Opus whatever
    /// the input rate, the stream's own rate otherwise.
    fn granule_rate(&self) -> u32 {
        match self.format {
            Some(AudioFormat::Opus) => OPUS_RATE_HZ,
            _ => self.sample_rate,
        }
    }

    /// Wrap muxed bytes as an output frame, or `None` when nothing was produced.
    fn byte_frame(&mut self, bytes: Vec<u8>) -> Option<PipelinePacket> {
        if bytes.is_empty() {
            return None;
        }
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            FrameTiming::default(),
            self.emitted,
        );
        self.emitted += 1;
        Some(PipelinePacket::DataFrame(frame))
    }
}

/// Samples a frame declares it decodes to, from its stamped duration (`None`
/// when untimed). Rounded, so a demuxer's sample -> ns conversion inverts
/// exactly.
fn declared_samples(duration_ns: u64, rate: u32) -> Option<u64> {
    if duration_ns == 0 || rate == 0 {
        return None;
    }
    let ns = u128::from(duration_ns) * u128::from(rate) + 500_000_000;
    Some((ns / 1_000_000_000) as u64)
}

/// A synthesized `OpusHead` (RFC 7845 §5.1): version 1, channel mapping family
/// 0 (mono / stereo), the encoder-lookahead pre-skip. All fields little-endian.
fn opus_head(channels: u8, sample_rate: u32) -> Vec<u8> {
    let mut h = Vec::from(*b"OpusHead");
    h.push(1); // version
    h.push(channels.clamp(1, 2));
    h.extend_from_slice(&OPUS_ENCODER_PRE_SKIP.to_le_bytes());
    h.extend_from_slice(&sample_rate.max(1).to_le_bytes()); // original input rate
    h.extend_from_slice(&0i16.to_le_bytes()); // output gain
    h.push(0); // channel mapping family
    h
}

/// A minimal VorbisComment header behind `magic`: the vendor string and an empty
/// field list (RFC 7845 §5.2). The Vorbis flavour needs the framing bit that its
/// own mapping mandates; `OpusTags` does not carry one.
fn vorbis_comment(magic: &[u8]) -> Vec<u8> {
    let mut p = Vec::from(magic);
    p.extend_from_slice(&(VENDOR.len() as u32).to_le_bytes());
    p.extend_from_slice(VENDOR);
    p.extend_from_slice(&0u32.to_le_bytes()); // no user comments
    if magic.starts_with(b"\x03") {
        p.push(1); // framing bit
    }
    p
}

/// Split a native FLAC header (`fLaC` + metadata blocks) into the Ogg-FLAC
/// mapping's beginning-of-stream packet and the header packets that follow.
///
/// The mapping's first packet is `0x7F "FLAC" 1 0`, a big-endian count of the
/// following header packets, then the native marker + STREAMINFO. The remaining
/// metadata blocks ride their own packets; the mapping mandates a
/// VORBIS_COMMENT, so one is synthesized when the source carried none. `None`
/// when `native` is too short to hold the mandatory STREAMINFO (a 4-byte block
/// header then 34 bytes), which is not a FLAC header at all.
fn flac_headers(native: &[u8]) -> Option<(Vec<u8>, Vec<Vec<u8>>)> {
    let streaminfo = native.get(4..42)?;
    let comment = comment_block(native).unwrap_or_else(|| {
        let body = vorbis_comment(b"");
        let mut b = Vec::from([0x84u8]); // VORBIS_COMMENT, last metadata block
        b.extend_from_slice(&(body.len() as u32).to_be_bytes()[1..]);
        b.extend_from_slice(&body);
        b
    });
    let mut bos = Vec::from([0x7Fu8]);
    bos.extend_from_slice(b"FLAC");
    bos.extend_from_slice(&[1, 0]); // mapping version 1.0
    bos.extend_from_slice(&1u16.to_be_bytes()); // one further header packet
    bos.extend_from_slice(b"fLaC");
    bos.extend_from_slice(streaminfo);
    // STREAMINFO is not the last block here: the comment packet follows.
    bos[13] &= 0x7F;
    Some((bos, Vec::from([comment])))
}

/// The VORBIS_COMMENT metadata block (type 4) of a native FLAC header, with its
/// last-block flag set so it terminates the mapping's header packets. `None`
/// when the header carries none. Block lengths come from the stream, so the walk
/// is bounded by the buffer at every step.
fn comment_block(native: &[u8]) -> Option<Vec<u8>> {
    let mut at = 4usize; // past the `fLaC` marker
    loop {
        let header = native.get(at..at.checked_add(4)?)?;
        let last = header[0] & 0x80 != 0;
        let kind = header[0] & 0x7F;
        let len = u32::from_be_bytes([0, header[1], header[2], header[3]]) as usize;
        let end = at.checked_add(4)?.checked_add(len)?;
        if kind == 4 {
            let mut block = native.get(at..end)?.to_vec();
            block[0] = 0x84;
            return Some(block);
        }
        if last {
            return None;
        }
        at = end;
    }
}

impl AsyncElement for OggMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if Self::format_of(upstream_caps).is_some() {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| {
            if Self::format_of(input).is_some() {
                CapsSet::one(Self::output_caps())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::Audio {
            channels,
            sample_rate,
            ..
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        self.format = Some(Self::format_of(absolute_caps).ok_or(G2gError::CapsMismatch)?);
        self.channels = *channels;
        self.sample_rate = *sample_rate;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[PropertySpec::new(
            "serial",
            PropKind::Uint,
            "serial number of the logical bitstream",
        )];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "serial" => {
                let raw = value.as_uint().ok_or(PropError::Type)?;
                let serial = u32::try_from(raw).map_err(|_| PropError::Value)?;
                self.serial = serial;
                // Only meaningful before the first page; afterwards the stream is
                // already identified by the old serial.
                self.writer = OggPageWriter::new(serial);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "serial" => Some(PropValue::Uint(u64::from(self.serial))),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    // Codec config arrives in-band ahead of the audio; hold it
                    // until the first audio packet, then write the header pages.
                    if !self.headers_written && self.is_header(slice) {
                        self.headers.push(slice.to_vec());
                        return Ok(());
                    }
                    let mut bytes = if self.headers_written {
                        Vec::new()
                    } else {
                        self.write_headers()?
                    };
                    let granule = self.advance_granule(slice, frame.timing.duration_ns);
                    bytes.extend_from_slice(&self.writer.push_packet(slice.to_vec(), granule));
                    if let Some(p) = self.byte_frame(bytes) {
                        out.push(p).await?;
                    }
                }
                // Flush the queued packets and close the logical bitstream; the
                // runner's transform arm forwards EOS after this.
                PipelinePacket::Eos => {
                    if self.headers_written {
                        let bytes = self.writer.flush(true);
                        if let Some(p) = self.byte_frame(bytes) {
                            out.push(p).await?;
                        }
                    }
                }
                // Channels / rate only feed the synthesized headers, which are
                // written before any audio flows.
                PipelinePacket::CapsChanged(caps) => {
                    if !self.headers_written {
                        if let Caps::Audio {
                            channels,
                            sample_rate,
                            ..
                        } = &caps
                        {
                            self.channels = *channels;
                            self.sample_rate = *sample_rate;
                        }
                    }
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for OggMux {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::from_alternatives(Self::input_alternatives())),
            PadTemplate::source(CapsSet::one(Self::output_caps())),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ogg::{OggCodec, OggDemuxer};
    use alloc::vec;
    use g2g_core::PushOutcome;

    #[derive(Default)]
    struct CaptureSink {
        bytes: Vec<u8>,
    }
    impl OutputSink for CaptureSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                if let PipelinePacket::DataFrame(f) = packet {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.bytes.extend_from_slice(s);
                    }
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    fn opus_caps() -> Caps {
        Caps::Audio {
            format: AudioFormat::Opus,
            channels: 2,
            sample_rate: 48_000,
        }
    }

    fn frame(data: Vec<u8>) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            FrameTiming::default(),
            0,
        ))
    }

    /// A 20 ms CELT-FB stereo Opus packet (TOC config 31, one frame).
    fn opus_packet(fill: u8) -> Vec<u8> {
        vec![(31 << 3) | 0x04, fill, fill]
    }

    #[test]
    fn caps_audio_in_ogg_byte_stream_out() {
        let m = OggMux::new();
        assert!(m.intercept_caps(&opus_caps()).is_ok());
        let aac = Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
        };
        assert!(m.intercept_caps(&aac).is_err(), "AAC has no Ogg mapping");

        let CapsConstraint::DerivedOutput(f) = m.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        assert!(matches!(
            f(&opus_caps()).alternatives(),
            [Caps::ByteStream {
                encoding: ByteStreamEncoding::Ogg
            }]
        ));
    }

    #[tokio::test]
    async fn opus_round_trips_through_the_demuxer() {
        let mut mux = OggMux::new();
        mux.configure_pipeline(&opus_caps()).unwrap();
        let mut sink = CaptureSink::default();
        let packets: Vec<Vec<u8>> = (0..5).map(opus_packet).collect();
        for p in &packets {
            mux.process(frame(p.clone()), &mut sink).await.unwrap();
        }
        mux.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        let mut d = OggDemuxer::new();
        d.push_data(&sink.bytes);
        let info = d.info().expect("identification header parsed");
        assert_eq!(info.codec, OggCodec::Opus);
        assert_eq!(info.channels, 2);
        assert_eq!(info.pre_skip, OPUS_ENCODER_PRE_SKIP);
        assert_eq!(d.take_packets(), packets, "audio packets survive the mux");
        // Five 20 ms packets at 48 kHz.
        assert_eq!(d.end_granule(), Some(5 * 960));
    }

    #[tokio::test]
    async fn in_band_opus_head_is_carried_verbatim() {
        let mut head = Vec::from(*b"OpusHead");
        head.extend_from_slice(&[1, 1]);
        head.extend_from_slice(&777u16.to_le_bytes()); // a source pre-skip
        head.extend_from_slice(&48_000u32.to_le_bytes());
        head.extend_from_slice(&[0, 0, 0]);

        let mut mux = OggMux::new();
        mux.configure_pipeline(&opus_caps()).unwrap();
        let mut sink = CaptureSink::default();
        mux.process(frame(head.clone()), &mut sink).await.unwrap();
        mux.process(frame(opus_packet(1)), &mut sink).await.unwrap();
        mux.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        let mut d = OggDemuxer::new();
        d.push_data(&sink.bytes);
        assert_eq!(d.head_header(), Some(head.as_slice()));
        assert_eq!(d.info().unwrap().pre_skip, 777);
    }

    /// A timed frame: `samples` of the packet's output are real audio.
    fn timed_frame(data: Vec<u8>, samples: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            FrameTiming {
                duration_ns: samples * 1_000_000_000 / u64::from(OPUS_RATE_HZ),
                ..FrameTiming::default()
            },
            0,
        ))
    }

    #[tokio::test]
    async fn declared_durations_cap_the_final_granule() {
        let mut mux = OggMux::new();
        mux.configure_pipeline(&opus_caps()).unwrap();
        let mut sink = CaptureSink::default();
        mux.process(timed_frame(opus_packet(0), 960), &mut sink)
            .await
            .unwrap();
        // A short final packet: 400 of its 960 samples are real, the rest padding.
        mux.process(timed_frame(opus_packet(1), 400), &mut sink)
            .await
            .unwrap();
        mux.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        let mut d = OggDemuxer::new();
        d.push_data(&sink.bytes);
        assert_eq!(d.end_granule(), Some(960 + 400));
    }

    /// An untimed stream (a live encoder stamps no duration) keeps the codec's
    /// own sample counts, so the cap never truncates it to nothing.
    #[tokio::test]
    async fn untimed_input_keeps_the_natural_granule() {
        let mut mux = OggMux::new();
        mux.configure_pipeline(&opus_caps()).unwrap();
        let mut sink = CaptureSink::default();
        for i in 0..3 {
            mux.process(frame(opus_packet(i)), &mut sink).await.unwrap();
        }
        mux.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        let mut d = OggDemuxer::new();
        d.push_data(&sink.bytes);
        assert_eq!(d.end_granule(), Some(3 * 960));
    }

    #[tokio::test]
    async fn flac_header_becomes_the_ogg_mapping_packet() {
        // A native `fLaC` header: STREAMINFO (last block) for 44.1 kHz stereo.
        let mut native = Vec::from(*b"fLaC");
        native.extend_from_slice(&[0x80, 0, 0, 34]);
        let mut body = [0u8; 34];
        let rate = 44_100u32;
        body[10] = (rate >> 12) as u8;
        body[11] = (rate >> 4) as u8;
        body[12] = (((rate & 0xF) as u8) << 4) | (1 << 1);
        native.extend_from_slice(&body);
        // One FLAC frame: 4096-sample block, 44.1 kHz, stereo, with its CRC-8.
        let audio = vec![0xFFu8, 0xF8, 0xC9, 0x18, 0x00, 0xC2];

        let mut mux = OggMux::new();
        mux.configure_pipeline(&Caps::Audio {
            format: AudioFormat::Flac,
            channels: 2,
            sample_rate: rate,
        })
        .unwrap();
        let mut sink = CaptureSink::default();
        mux.process(frame(native), &mut sink).await.unwrap();
        mux.process(frame(audio.clone()), &mut sink).await.unwrap();
        mux.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        let mut d = OggDemuxer::new();
        d.push_data(&sink.bytes);
        let info = d.info().expect("mapping header parsed");
        assert_eq!(info.codec, OggCodec::Flac);
        assert_eq!(info.sample_rate, rate);
        assert_eq!(info.channels, 2);
        assert!(
            d.comment_header().is_some(),
            "the mapping's mandatory VorbisComment packet is written"
        );
        assert_eq!(d.take_packets(), vec![audio]);
        assert_eq!(d.end_granule(), Some(4096), "one 4096-sample block");
    }
}
