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
//! One logical bitstream (a single input pad), mirroring the single-stream
//! `OggDemux`. The grouped multi-stream case is the fan-in sibling
//! [`crate::oggmuxn::OggMuxN`], which reuses this module's [`OggStreamMux`] once
//! per input pad.
//!
//! **Chained streams (M858).** A `Segment` arriving after audio has flowed ends
//! the logical bitstream and opens the next link of a chained file: the current
//! link's pages are flushed with the end-of-stream flag, then the next link
//! writes its own beginning-of-stream page on a fresh serial. That is the write
//! side of the chains [`crate::oggdemux::OggDemux`] reads, which announces each
//! one with exactly that `Segment`, so a chained file survives a demux -> mux
//! round trip; it is also what a track change on a concatenating pipeline
//! (gapless playback, an icecast-style feed) looks like.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec, Segment,
};

use crate::ogg::{OggPageWriter, VorbisTiming};
use crate::opusparse::{
    is_opus_config, packet_samples as opus_packet_samples, synth_opus_head, OPUS_RATE_HZ,
};

/// Default logical-bitstream serial number, overridable via the `serial`
/// property. Any value identifies the stream as long as it is unique within the
/// file, and this muxer writes one stream.
pub(crate) const DEFAULT_SERIAL: u32 = 0x6732_6732; // "g2g2"

/// The vendor string written into synthesized comment headers.
const VENDOR: &[u8] = b"g2g";

/// The Ogg byte stream both muxers produce.
pub(crate) fn ogg_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::Ogg,
    }
}

/// The elementary streams an Ogg muxer accepts on a sink pad.
pub(crate) fn input_alternatives() -> Vec<Caps> {
    Vec::from(
        [AudioFormat::Opus, AudioFormat::Vorbis, AudioFormat::Flac].map(|format| Caps::Audio {
            format,
            channels: 0,
            sample_rate: 0,
        }),
    )
}

/// The codec of `caps`, or `None` for a stream Ogg cannot carry here.
pub(crate) fn format_of(caps: &Caps) -> Option<AudioFormat> {
    match caps {
        Caps::Audio {
            format: f @ (AudioFormat::Opus | AudioFormat::Vorbis | AudioFormat::Flac),
            ..
        } => Some(*f),
        _ => None,
    }
}

/// One logical bitstream being written: its page writer, codec mapping, in-band
/// headers and granule accounting. The single-input [`OggMux`] owns one; the
/// grouped [`crate::oggmuxn::OggMuxN`] owns one per input pad, which is why the
/// header writing is split into [`write_bos`](Self::write_bos) and
/// [`write_rest`](Self::write_rest): RFC 3533 grouping puts every stream's
/// beginning-of-stream page ahead of every other page.
#[derive(Debug)]
pub(crate) struct OggStreamMux {
    writer: OggPageWriter,
    /// The input codec, set at configure.
    format: Option<AudioFormat>,
    channels: u8,
    sample_rate: u32,
    /// Header packets collected in-band, in arrival order.
    headers: Vec<Vec<u8>>,
    /// Whether the header pages have been written (done lazily, so every in-band
    /// header has landed first).
    headers_written: bool,
    /// Header packets `write_bos` held back for `write_rest`.
    pending_rest: Vec<Vec<u8>>,
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

impl OggStreamMux {
    pub(crate) fn new(serial: u32) -> Self {
        Self {
            writer: OggPageWriter::new(serial),
            format: None,
            channels: 0,
            sample_rate: 0,
            headers: Vec::new(),
            headers_written: false,
            pending_rest: Vec::new(),
            natural_samples: 0,
            declared_ns: 0,
            untimed: false,
            vorbis: None,
            prev_blocksize: None,
        }
    }

    /// Re-key the bitstream. Only meaningful before its first page, which
    /// identifies the stream by the serial then in force.
    pub(crate) fn set_serial(&mut self, serial: u32) {
        self.writer = OggPageWriter::new(serial);
    }

    pub(crate) fn serial(&self) -> u32 {
        self.writer.serial()
    }

    /// Pin the codec mapping from negotiated caps.
    pub(crate) fn configure(&mut self, caps: &Caps) -> Result<(), G2gError> {
        let Caps::Audio {
            channels,
            sample_rate,
            ..
        } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        self.format = Some(format_of(caps).ok_or(G2gError::CapsMismatch)?);
        self.channels = *channels;
        self.sample_rate = *sample_rate;
        Ok(())
    }

    /// Refine the channel count / rate a synthesized header will carry. Ignored
    /// once the headers are on the wire.
    pub(crate) fn refine_caps(&mut self, caps: &Caps) {
        if self.headers_written {
            return;
        }
        if let Caps::Audio {
            channels,
            sample_rate,
            ..
        } = caps
        {
            self.channels = *channels;
            self.sample_rate = *sample_rate;
        }
    }

    pub(crate) fn headers_written(&self) -> bool {
        self.headers_written
    }

    /// Collect an in-band codec-config packet, to be written as a header page.
    pub(crate) fn push_header(&mut self, packet: &[u8]) {
        self.headers.push(packet.to_vec());
    }

    /// Write this stream's beginning-of-stream page (the mapping's first packet
    /// alone, granule 0), stashing the remaining header packets for
    /// [`write_rest`](Self::write_rest).
    pub(crate) fn write_bos(&mut self) -> Result<Vec<u8>, G2gError> {
        let (bos, rest) = self.header_packets().ok_or(G2gError::NotConfigured)?;
        // Vorbis timing needs the mode tables from ident + setup.
        if self.format == Some(AudioFormat::Vorbis) {
            self.vorbis = match (self.vorbis_header(1), self.vorbis_header(5)) {
                (Some(i), Some(s)) => VorbisTiming::parse(i, s),
                _ => None,
            };
        }
        self.pending_rest = rest;
        self.writer.push_packet(bos, 0);
        Ok(self.writer.flush(false))
    }

    /// Write the header packets after the beginning-of-stream one, on their own
    /// page at granule 0 (the mappings require audio to start on a fresh page).
    pub(crate) fn write_rest(&mut self) -> Vec<u8> {
        for packet in core::mem::take(&mut self.pending_rest) {
            self.writer.push_packet(packet, 0);
        }
        self.headers_written = true;
        self.writer.flush(false)
    }

    /// Queue one audio packet, returning whatever pages completed.
    pub(crate) fn push_audio(&mut self, packet: &[u8], duration_ns: u64) -> Vec<u8> {
        let granule = self.advance_granule(packet, duration_ns);
        self.writer.push_packet(packet.to_vec(), granule)
    }

    /// Flush the queued packets; `eos` closes the logical bitstream.
    pub(crate) fn flush(&mut self, eos: bool) -> Vec<u8> {
        self.writer.flush(eos)
    }

    /// Close this logical bitstream and restart on `serial` as the next link of
    /// a chained stream (M858), returning the closed link's remaining pages with
    /// the last flagged end-of-stream. The codec mapping survives; everything
    /// the link accumulated (headers, granule, Vorbis lapping) restarts, so the
    /// new link writes its own header pages from granule 0.
    pub(crate) fn begin_chain(&mut self, serial: u32) -> Vec<u8> {
        let tail = if self.headers_written {
            self.writer.flush(true)
        } else {
            Vec::new()
        };
        *self = Self {
            format: self.format,
            channels: self.channels,
            sample_rate: self.sample_rate,
            ..Self::new(serial)
        };
        tail
    }

    /// Whether `packet` is codec config rather than audio, for the configured
    /// codec's in-band convention.
    pub(crate) fn is_header(&self, packet: &[u8]) -> bool {
        match self.format {
            // `OpusHead` / `OpusTags` (RFC 7845); audio packets start with a TOC.
            Some(AudioFormat::Opus) => is_opus_config(packet),
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
                    .unwrap_or_else(|| synth_opus_head(self.channels, self.sample_rate));
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
}

/// How many link serials a chained stream remembers, so a derived serial never
/// collides with one still in the file's recent history. A radio-style stream of
/// unbounded chains cannot keep them all.
const RECENT_SERIALS: usize = 16;

/// Muxes one audio elementary stream into an Ogg byte stream.
#[derive(Debug)]
pub struct OggMux {
    stream: OggStreamMux,
    /// Base serial (the `serial` property): the first link's, and what the later
    /// links' serials are derived from.
    serial: u32,
    /// Chain links written so far, `0` for the first.
    link: u32,
    /// Serials already handed out, most recent last.
    recent_serials: Vec<u32>,
    /// The segment in force, so a repeat of it is not a chain boundary.
    last_segment: Option<Segment>,
    configured: bool,
    emitted: u64,
}

impl Default for OggMux {
    fn default() -> Self {
        Self::new()
    }
}

impl OggMux {
    pub fn new() -> Self {
        Self {
            stream: OggStreamMux::new(DEFAULT_SERIAL),
            serial: DEFAULT_SERIAL,
            link: 0,
            recent_serials: Vec::from([DEFAULT_SERIAL]),
            last_segment: None,
            configured: false,
            emitted: 0,
        }
    }

    /// Set the logical bitstream's serial number.
    pub fn with_serial(mut self, serial: u32) -> Self {
        self.set_base_serial(serial);
        self
    }

    /// Re-key the first link. Only meaningful before its first page, which
    /// identifies the stream by the serial then in force.
    fn set_base_serial(&mut self, serial: u32) {
        self.serial = serial;
        self.stream.set_serial(serial);
        self.recent_serials = Vec::from([serial]);
    }

    /// The serial the next chain link takes: mixed from the base serial and the
    /// link index, then stepped past the serials the recent links used, since a
    /// serial must not repeat within one physical stream. Deterministic: the
    /// `no_std` core has no entropy source, and the same input must mux to the
    /// same file twice.
    fn next_serial(&mut self) -> u32 {
        self.link = self.link.saturating_add(1);
        let mut serial = mix_serial(self.serial, self.link);
        for salt in 1..8 {
            if !self.recent_serials.contains(&serial) {
                break;
            }
            serial = mix_serial(serial, salt);
        }
        if self.recent_serials.len() == RECENT_SERIALS {
            self.recent_serials.remove(0);
        }
        self.recent_serials.push(serial);
        serial
    }

    /// Count of Ogg byte frames forwarded.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// Write the header pages: the beginning-of-stream packet alone on the first
    /// page, the remaining headers on the next.
    fn write_headers(&mut self) -> Result<Vec<u8>, G2gError> {
        let mut out = self.stream.write_bos()?;
        out.extend_from_slice(&self.stream.write_rest());
        Ok(out)
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

/// A serial for chain link `link`, from the base serial through the SplitMix64
/// finalizer: unrelated to the serials around it (what the format expects) but
/// reproducible.
fn mix_serial(base: u32, link: u32) -> u32 {
    let mut h = u64::from(base) ^ u64::from(link).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    (h >> 32) as u32
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
        if format_of(upstream_caps).is_some() {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| {
            if format_of(input).is_some() {
                CapsSet::one(ogg_caps())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.stream.configure(absolute_caps)?;
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
                // Only meaningful before the first page; afterwards the stream is
                // already identified by the old serial.
                self.set_base_serial(serial);
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
                    if !self.stream.headers_written() && self.stream.is_header(slice) {
                        self.stream.push_header(slice);
                        return Ok(());
                    }
                    let mut bytes = if self.stream.headers_written() {
                        Vec::new()
                    } else {
                        self.write_headers()?
                    };
                    bytes.extend_from_slice(
                        &self.stream.push_audio(slice, frame.timing.duration_ns),
                    );
                    if let Some(p) = self.byte_frame(bytes) {
                        out.push(p).await?;
                    }
                }
                // Flush the queued packets and close the logical bitstream; the
                // runner's transform arm forwards EOS after this.
                PipelinePacket::Eos => {
                    if self.stream.headers_written() {
                        let bytes = self.stream.flush(true);
                        if let Some(p) = self.byte_frame(bytes) {
                            out.push(p).await?;
                        }
                    }
                }
                // Channels / rate only feed the synthesized headers, which are
                // written before any audio flows.
                PipelinePacket::CapsChanged(caps) => self.stream.refine_caps(&caps),
                // A segment after audio has flowed is a chain boundary (M858):
                // the logical bitstream so far is complete, so close it and let
                // the next link open on a fresh serial. The segment every stream
                // opens with, and any repeat of the one in force, is not one.
                PipelinePacket::Segment(seg) => {
                    let boundary = self.stream.headers_written() && self.last_segment != Some(seg);
                    self.last_segment = Some(seg);
                    if boundary {
                        let serial = self.next_serial();
                        let bytes = self.stream.begin_chain(serial);
                        if let Some(p) = self.byte_frame(bytes) {
                            out.push(p).await?;
                        }
                    }
                    out.push(PipelinePacket::Segment(seg)).await?;
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
            PadTemplate::sink(CapsSet::from_alternatives(input_alternatives())),
            PadTemplate::source(CapsSet::one(ogg_caps())),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ogg::{page_flags, OggCodec, OggDemuxer};
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
        assert_eq!(info.pre_skip, crate::opusparse::OPUS_ENCODER_PRE_SKIP);
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

    /// Mux `links` runs of Opus packets, a `Segment` between consecutive ones.
    async fn mux_chained(links: &[Vec<Vec<u8>>]) -> Vec<u8> {
        let mut mux = OggMux::new();
        mux.configure_pipeline(&opus_caps()).unwrap();
        let mut sink = CaptureSink::default();
        for (i, packets) in links.iter().enumerate() {
            if i > 0 {
                let start = i as u64 * 1_000_000_000;
                let seg = PipelinePacket::Segment(Segment {
                    start,
                    base: start,
                    ..Segment::new()
                });
                mux.process(seg, &mut sink).await.unwrap();
            }
            for p in packets {
                mux.process(frame(p.clone()), &mut sink).await.unwrap();
            }
        }
        mux.process(PipelinePacket::Eos, &mut sink).await.unwrap();
        sink.bytes
    }

    #[tokio::test]
    async fn a_segment_after_audio_opens_the_next_chain() {
        let first: Vec<Vec<u8>> = (0..3).map(opus_packet).collect();
        let second: Vec<Vec<u8>> = (10..14).map(opus_packet).collect();
        let bytes = mux_chained(&[first.clone(), second.clone()]).await;

        let pages = page_flags(&bytes);
        let bos: Vec<usize> = (0..pages.len())
            .filter(|&i| pages[i].1 & 0x02 != 0)
            .collect();
        let eos: Vec<usize> = (0..pages.len())
            .filter(|&i| pages[i].1 & 0x04 != 0)
            .collect();
        assert_eq!(bos.len(), 2, "one beginning-of-stream page per link");
        assert_eq!(eos.len(), 2, "each link closes");
        assert!(
            eos[0] < bos[1],
            "the first link ends before the second begins"
        );
        assert_ne!(
            pages[bos[0]].0, pages[bos[1]].0,
            "each link carries its own serial"
        );

        let mut d = OggDemuxer::new();
        d.push_data(&bytes);
        assert_eq!(d.chain(), 1, "the demuxer reads two physical streams");
        assert_eq!(d.streams().len(), 2);
        assert_eq!(d.streams()[1].chain(), 1);
        assert_eq!(d.stream_mut(0).unwrap().take_packets(), first);
        assert_eq!(d.stream_mut(1).unwrap().take_packets(), second);
        // Each link times from its own zero.
        assert_eq!(d.streams()[0].end_granule(), Some(3 * 960));
        assert_eq!(d.streams()[1].end_granule(), Some(4 * 960));
    }

    /// The segment every stream opens with, and a repeat of the one in force,
    /// leave the bitstream alone: only a change of segment is a boundary.
    #[tokio::test]
    async fn an_unchanged_segment_is_not_a_chain_boundary() {
        let mut mux = OggMux::new();
        mux.configure_pipeline(&opus_caps()).unwrap();
        let mut sink = CaptureSink::default();
        let seg = || PipelinePacket::Segment(Segment::new());
        mux.process(seg(), &mut sink).await.unwrap();
        mux.process(frame(opus_packet(0)), &mut sink).await.unwrap();
        mux.process(seg(), &mut sink).await.unwrap();
        mux.process(frame(opus_packet(1)), &mut sink).await.unwrap();
        mux.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        let pages = page_flags(&sink.bytes);
        assert_eq!(
            pages.iter().filter(|(_, ht)| ht & 0x02 != 0).count(),
            1,
            "one logical bitstream"
        );
        let mut d = OggDemuxer::new();
        d.push_data(&sink.bytes);
        assert_eq!(d.chain(), 0);
        assert_eq!(d.end_granule(), Some(2 * 960));
    }

    /// Link serials are derived, not counted: distinct, and the same every run.
    #[test]
    fn chain_serials_are_deterministic_and_distinct() {
        let mut a = OggMux::new().with_serial(7);
        let mut b = OggMux::new().with_serial(7);
        let mut seen = vec![7u32];
        for _ in 0..8 {
            let s = a.next_serial();
            assert_eq!(s, b.next_serial(), "same base, same serials");
            assert!(!seen.contains(&s), "serial {s} repeats");
            seen.push(s);
        }
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
