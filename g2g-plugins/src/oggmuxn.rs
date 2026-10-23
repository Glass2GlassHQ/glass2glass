//! Ogg multiplexer element (M789, M790): N audio elementary streams in
//! (`Caps::Audio{Opus|Vorbis|Flac}` per pad), one Ogg byte stream out, grouped
//! when there are several. The inverse of [`crate::oggdemux::OggDemux`] and the
//! write side of the grouped files [`crate::oggdemux::OggDemuxN`] reads.
//!
//! A [`MultiInputElement`]: each pad becomes its own logical bitstream with its
//! own serial number, framed by [`crate::ogg::OggPageWriter`] under the pad's
//! codec mapping. Packets interleave by presentation timestamp through the M204
//! [`InputAggregator::take_earliest_by`](g2g_core::InputAggregator::take_earliest_by)
//! merge, so the data pages of the streams alternate the way a player expects.
//!
//! Page order follows RFC 3533 §4 grouping: every stream's beginning-of-stream
//! page first, in pad order, then the remaining header pages per stream, then
//! the interleaved data pages. That block is written when the merge first
//! releases a packet, which is also the first moment every pad's in-band codec
//! config is known to have arrived.
//!
//! ```text
//! ... ! opusenc ! oggmux ! filesink location=out.opus
//! filesrc location=in.ogg ! oggdemux stream=vorbis ! oggmux ! filesink location=out.ogg
//! a.! m.  b.! m.  oggmux name=m ! filesink location=out.ogg
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
//! **Chained streams (M858).** With one input pad, a `Segment` arriving after
//! audio has flowed ends the logical bitstream and opens the next link of a
//! chained file: the current link's pages are flushed with the end-of-stream
//! flag, then the next link writes its own beginning-of-stream page on a fresh
//! serial. That is the write side of the chains [`crate::oggdemux::OggDemux`]
//! reads, which announces each one with exactly that `Segment`, so a chained file
//! survives a demux -> mux round trip; it is also what a track change on a
//! concatenating pipeline (gapless playback, an icecast-style feed) looks like.
//!
//! Registered only as the `oggmux` muxer in
//! [`default_registry`](crate::registry::default_registry), so any number of
//! input links builds this element.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, FrameTiming, G2gError, InputAggregator, MemoryDomain, MultiInputElement,
    OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Segment,
};

use crate::mux::{audio_track_params, refused_input};
use crate::ogg::{OggPageWriter, VorbisTiming};
use crate::opusparse::{
    is_opus_config, packet_samples as opus_packet_samples, synth_opus_head, OPUS_ENCODER_PRE_SKIP,
    OPUS_RATE_HZ,
};
use g2g_core::log::short_type_name;

/// Default serial number of the first logical bitstream, overridable via the
/// `serial` property. Any value identifies a stream as long as it is unique
/// within the file.
const DEFAULT_SERIAL: u32 = 0x6732_6732; // "g2g2"

/// The Ogg byte stream the muxer produces.
fn ogg_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::Ogg,
    }
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

// a stream's beginning-of-stream packet and the header packets after it
type HeaderPackets = (Vec<u8>, Vec<Vec<u8>>);

/// One logical bitstream being written: its page writer, codec mapping, in-band
/// headers and granule accounting. [`OggMuxN`] holds one per input pad, which is
/// why the header writing is split into [`write_bos`](Self::write_bos) and
/// [`write_rest`](Self::write_rest): RFC 3533 grouping puts every stream's
/// beginning-of-stream page ahead of every other page.
#[derive(Debug)]
struct OggStreamMux {
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
    fn new(serial: u32) -> Self {
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
    fn set_serial(&mut self, serial: u32) {
        self.writer = OggPageWriter::new(serial);
    }

    fn serial(&self) -> u32 {
        self.writer.serial()
    }

    /// Pin the codec mapping from negotiated caps.
    fn configure(&mut self, caps: &Caps) -> Result<(), G2gError> {
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
    fn refine_caps(&mut self, caps: &Caps) {
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

    /// Collect an in-band codec-config packet, to be written as a header page.
    fn push_header(&mut self, packet: &[u8]) {
        self.headers.push(packet.to_vec());
    }

    /// Write this stream's beginning-of-stream page (the mapping's first packet
    /// alone, granule 0), stashing the remaining header packets for
    /// [`write_rest`](Self::write_rest).
    fn write_bos(&mut self) -> Result<Vec<u8>, G2gError> {
        let (bos, rest) = self.header_packets()?.ok_or(G2gError::NotConfigured)?;
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
    fn write_rest(&mut self) -> Vec<u8> {
        for packet in core::mem::take(&mut self.pending_rest) {
            self.writer.push_packet(packet, 0);
        }
        self.headers_written = true;
        self.writer.flush(false)
    }

    /// Queue one audio packet, returning whatever pages completed.
    fn push_audio(&mut self, packet: &[u8], duration_ns: u64) -> Vec<u8> {
        let granule = self.advance_granule(packet, duration_ns);
        self.writer.push_packet(packet.to_vec(), granule)
    }

    /// Flush the queued packets; `eos` closes the logical bitstream.
    fn flush(&mut self, eos: bool) -> Vec<u8> {
        self.writer.flush(eos)
    }

    /// Close this logical bitstream and restart on `serial` as the next link of
    /// a chained stream (M858), returning the closed link's remaining pages with
    /// the last flagged end-of-stream. The codec mapping survives; everything
    /// the link accumulated (headers, granule, Vorbis lapping) restarts, so the
    /// new link writes its own header pages from granule 0.
    fn begin_chain(&mut self, serial: u32) -> Vec<u8> {
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
    fn is_header(&self, packet: &[u8]) -> bool {
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
    fn header_packets(&self) -> Result<Option<HeaderPackets>, G2gError> {
        let Some(format) = self.format else {
            return Ok(None);
        };
        match format {
            AudioFormat::Opus => {
                let head = match self.headers.iter().find(|h| h.starts_with(b"OpusHead")) {
                    Some(head) => head.clone(),
                    None => {
                        let params = audio_track_params(
                            g2g_core::log::short_type_name::<Self>(),
                            format,
                            self.channels,
                            self.sample_rate,
                            &[],
                        )?;
                        synth_opus_head(params.channels, params.sample_rate, OPUS_ENCODER_PRE_SKIP)
                    }
                };
                let tags = self
                    .headers
                    .iter()
                    .find(|h| h.starts_with(b"OpusTags"))
                    .cloned()
                    .unwrap_or_else(|| vorbis_comment(b"OpusTags"));
                Ok(Some((head, Vec::from([tags]))))
            }
            AudioFormat::Vorbis => {
                // Remux only: without the codebooks there is no decodable stream
                // to write, so the ident and setup headers must arrive in-band.
                let (Some(ident), Some(setup)) = (self.vorbis_header(1), self.vorbis_header(5))
                else {
                    return Ok(None);
                };
                let comment = self
                    .vorbis_header(3)
                    .map(|c| c.to_vec())
                    .unwrap_or_else(|| vorbis_comment(b"\x03vorbis"));
                Ok(Some((ident.to_vec(), Vec::from([comment, setup.to_vec()]))))
            }
            AudioFormat::Flac => Ok(self.headers.first().and_then(|first| flac_headers(first))),
            _ => Ok(None),
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

/// A minimal VorbisComment header behind `magic`: the vendor string and no
/// fields, for a mapping that mandates the packet when the source carried none.
fn vorbis_comment(magic: &[u8]) -> Vec<u8> {
    crate::vorbiscomment::vorbis_comment(
        magic,
        crate::vorbiscomment::VENDOR,
        &g2g_core::TagList::new(),
    )
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
fn flac_headers(native: &[u8]) -> Option<HeaderPackets> {
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

/// Muxes N audio elementary streams into one Ogg byte stream, grouped when N > 1.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::oggmuxn::OggMuxN;
///
/// let mux = OggMuxN::new(2).with_serial(0x1234);
/// assert_eq!(mux.emitted(), 0);
/// ```
#[derive(Debug)]
pub struct OggMuxN {
    inputs: usize,
    /// One logical bitstream per input pad, in pad order.
    streams: Vec<OggStreamMux>,
    /// Per-input packet buffer; releases the globally earliest-PTS packet (M204).
    agg: InputAggregator<Frame>,
    /// Serial of the first stream; pad `i` takes `serial + i`.
    serial: u32,
    /// Whether the grouped header block has been written.
    headers_written: bool,
    /// Whether the end-of-stream pages have been written, so the close happens
    /// once however many per-input EOS packets arrive.
    closed: bool,
    /// Chain links written so far, `0` for the first.
    link: u32,
    /// Serials already handed out, most recent last.
    recent_serials: Vec<u32>,
    /// The segment in force, so a repeat of it is not a chain boundary.
    last_segment: Option<Segment>,
    emitted: u64,
}

impl OggMuxN {
    /// A muxer with `inputs` input pads. Pad `i` becomes the `i`-th logical
    /// bitstream of the file, and its codec mapping comes from its negotiated
    /// caps.
    pub fn new(inputs: usize) -> Self {
        assert!(inputs > 0, "OggMuxN needs at least one input");
        Self {
            inputs,
            streams: (0..inputs)
                .map(|i| OggStreamMux::new(serial_for(DEFAULT_SERIAL, i)))
                .collect(),
            agg: InputAggregator::new(inputs),
            serial: DEFAULT_SERIAL,
            headers_written: false,
            closed: false,
            link: 0,
            recent_serials: Vec::from([DEFAULT_SERIAL]),
            last_segment: None,
            emitted: 0,
        }
    }

    /// Set the first stream's serial number; pad `i` takes `serial + i`.
    pub fn with_serial(mut self, serial: u32) -> Self {
        self.set_serials(serial);
        self
    }

    /// Count of Ogg byte frames emitted.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// The serial numbers of the logical bitstreams, in pad order.
    pub fn serials(&self) -> Vec<u32> {
        self.streams.iter().map(|s| s.serial()).collect()
    }

    fn set_serials(&mut self, serial: u32) {
        self.serial = serial;
        for (i, stream) in self.streams.iter_mut().enumerate() {
            stream.set_serial(serial_for(serial, i));
        }
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

    /// Write the grouped header block: every stream's beginning-of-stream page in
    /// pad order, then each stream's remaining header pages (RFC 3533 §4).
    fn write_headers(&mut self) -> Result<Vec<u8>, G2gError> {
        let mut out = Vec::new();
        for stream in &mut self.streams {
            out.extend_from_slice(&stream.write_bos()?);
        }
        for stream in &mut self.streams {
            out.extend_from_slice(&stream.write_rest());
        }
        self.headers_written = true;
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

/// The serial of pad `index` given the base. Wrapping, so a base near the top of
/// the range still yields distinct serials.
fn serial_for(base: u32, index: usize) -> u32 {
    base.wrapping_add(index as u32)
}

impl MultiInputElement for OggMuxN {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so every pad takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn input_count(&self) -> usize {
        self.inputs
    }

    /// Named request pads (M481): every input is an audio slot, so `audio_%u` /
    /// `sink_%u` each claim the next positional one.
    fn input_pad_index(
        &self,
        _req: &g2g_core::runtime::PadRequest,
        ordinal: usize,
    ) -> Option<usize> {
        (ordinal < self.inputs).then_some(ordinal)
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if format_of(upstream_caps).is_some() {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        // Each pad forwards its stream verbatim; `configure_pipeline` rejects an
        // unsupported caps. `AcceptsAny` is the native muxer-input shape.
        CapsConstraint::AcceptsAny
    }

    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        Ok(CapsConstraint::Produces(CapsSet::one(ogg_caps())))
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        if format_of(absolute_caps).is_none() {
            return Err(refused_input(short_type_name::<Self>(), absolute_caps));
        }
        self.streams[input].configure(absolute_caps)?;
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(ogg_caps())
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Ogg muxer",
            "Codec/Muxer",
            "Muxes audio streams into an Ogg bitstream",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[PropertySpec::new(
            "serial",
            PropKind::Uint,
            "serial number of the first logical bitstream; pad i takes serial + i",
        )];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "serial" => {
                // The serials are baked into every page, so this only takes effect
                // before the header block is written.
                if self.headers_written || self.link > 0 {
                    return Err(PropError::ReadOnly);
                }
                let raw = value.as_uint().ok_or(PropError::Type)?;
                self.set_serials(u32::try_from(raw).map_err(|_| PropError::Value)?);
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
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    // Codec config arrives in-band ahead of the audio. It never
                    // enters the merge: the header block needs every stream's
                    // config before any of it can be written.
                    if !self.headers_written && self.streams[input].is_header(slice) {
                        self.streams[input].push_header(slice);
                        return Ok(());
                    }
                    self.agg.push(input, frame);
                }
                // A per-input Eos lets the merge release packets held waiting on
                // this input; the runner emits the merged Eos.
                PipelinePacket::Eos => self.agg.mark_ended(input),
                // Channels / rate only feed a synthesized header, written before
                // any audio flows.
                PipelinePacket::CapsChanged(caps) => {
                    self.streams[input].refine_caps(&caps);
                    return Ok(());
                }
                // A per-input `Segment` maps that stream to running time; a muxed
                // container carries its own timestamps, so it is consumed rather
                // than forwarded into the byte stream.
                PipelinePacket::Segment(segment) => {
                    // a changed segment after audio on a lone pad opens the next link of a chained file (M858)
                    // a grouped link closes as a whole, with other pads' packets still queued in the merge
                    let boundary = self.inputs == 1
                        && self.headers_written
                        && self.last_segment != Some(segment);
                    self.last_segment = Some(segment);
                    if boundary {
                        let serial = self.next_serial();
                        let tail = self.streams[0].begin_chain(serial);
                        self.headers_written = false;
                        if let Some(p) = self.byte_frame(tail) {
                            out.push(p).await?;
                        }
                    }
                    return Ok(());
                }
                other => {
                    out.push(other).await?;
                    return Ok(());
                }
            }

            let mut bytes = Vec::new();
            // Drain every packet now safe to emit, in global PTS order. The first
            // release means every contributing pad has delivered audio, so every
            // stream's in-band config has landed and the header block can go out.
            while let Some((stream, frame)) = self.agg.take_earliest_by(|f| f.timing.pts_ns) {
                if !self.headers_written {
                    bytes.extend_from_slice(&self.write_headers()?);
                }
                let slice = frame
                    .domain
                    .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                bytes.extend_from_slice(
                    &self.streams[stream].push_audio(slice, frame.timing.duration_ns),
                );
            }
            // Once every pad has ended and drained, close each logical bitstream.
            if self.headers_written
                && !self.closed
                && (0..self.inputs).all(|i| self.agg.is_ended(i))
            {
                self.closed = true;
                for stream in &mut self.streams {
                    bytes.extend_from_slice(&stream.flush(true));
                }
            }
            if let Some(p) = self.byte_frame(bytes) {
                out.push(p).await?;
            }
            Ok(())
        })
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
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            core::task::Poll::Ready({
                if let PipelinePacket::DataFrame(f) = packet {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.bytes.extend_from_slice(s);
                    }
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    fn audio_caps(format: AudioFormat, channels: u8, sample_rate: u32) -> Caps {
        Caps::Audio {
            format,
            channels,
            sample_rate,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }

    fn frame(data: Vec<u8>, pts_ns: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            FrameTiming {
                pts_ns,
                ..FrameTiming::default()
            },
            0,
        ))
    }

    /// A 20 ms CELT-FB stereo Opus packet (TOC config 31, one frame).
    fn opus_packet(fill: u8) -> Vec<u8> {
        vec![(31 << 3) | 0x04, fill, fill]
    }

    /// A native FLAC header (`fLaC` + STREAMINFO) for `rate`, stereo.
    fn flac_header(rate: u32) -> Vec<u8> {
        let mut native = Vec::from(*b"fLaC");
        native.extend_from_slice(&[0x80, 0, 0, 34]);
        let mut body = [0u8; 34];
        body[10] = (rate >> 12) as u8;
        body[11] = (rate >> 4) as u8;
        body[12] = (((rate & 0xF) as u8) << 4) | (1 << 1);
        native.extend_from_slice(&body);
        native
    }

    /// One 4096-sample FLAC frame header at 44.1 kHz stereo, with its CRC-8.
    fn flac_frame() -> Vec<u8> {
        vec![0xFFu8, 0xF8, 0xC9, 0x18, 0x00, 0xC2]
    }

    /// Mux an Opus stream on pad 0 and an Ogg-FLAC stream on pad 1.
    async fn mux_opus_and_flac() -> Vec<u8> {
        let mut mux = OggMuxN::new(2);
        mux.configure_pipeline(0, &audio_caps(AudioFormat::Opus, 2, 48_000))
            .unwrap();
        mux.configure_pipeline(1, &audio_caps(AudioFormat::Flac, 2, 44_100))
            .unwrap();
        let mut sink = CaptureSink::default();
        // FLAC's in-band header, then interleaved audio (Opus has no in-band
        // header here, so its `OpusHead` is synthesized).
        mux.process(1, frame(flac_header(44_100), 0), &mut sink)
            .await
            .unwrap();
        for i in 0..3u64 {
            mux.process(0, frame(opus_packet(i as u8), i * 20_000_000), &mut sink)
                .await
                .unwrap();
            mux.process(1, frame(flac_frame(), i * 92_879_818), &mut sink)
                .await
                .unwrap();
        }
        mux.process(0, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();
        mux.process(1, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();
        sink.bytes
    }

    #[tokio::test]
    async fn beginning_of_stream_pages_lead_and_serials_are_distinct() {
        let bytes = mux_opus_and_flac().await;
        let pages = page_flags(&bytes);
        let bos: Vec<u32> = pages
            .iter()
            .filter(|(_, ht)| ht & 0x02 != 0)
            .map(|(s, _)| *s)
            .collect();
        assert_eq!(bos.len(), 2, "one beginning-of-stream page per stream");
        assert_ne!(bos[0], bos[1], "distinct serial numbers");
        assert_eq!(
            &pages[..2].iter().map(|(s, _)| *s).collect::<Vec<_>>(),
            &bos,
            "RFC 3533 grouping: both BOS pages precede every other page"
        );
        // Exactly one end-of-stream page per stream, and both streams close.
        let eos: Vec<u32> = pages
            .iter()
            .filter(|(_, ht)| ht & 0x04 != 0)
            .map(|(s, _)| *s)
            .collect();
        assert_eq!(eos.len(), 2);
        assert_ne!(eos[0], eos[1]);
    }

    #[tokio::test]
    async fn both_streams_round_trip_through_the_demuxer() {
        let bytes = mux_opus_and_flac().await;
        let mut d = OggDemuxer::new();
        d.push_data(&bytes);

        assert_eq!(d.streams().len(), 2);
        assert_eq!(d.streams()[0].info().unwrap().codec, OggCodec::Opus);
        assert_eq!(d.streams()[1].info().unwrap().codec, OggCodec::Flac);
        assert_eq!(d.streams()[1].info().unwrap().sample_rate, 44_100);
        assert_eq!(
            d.stream_mut(0).unwrap().take_packets(),
            (0..3).map(|i| opus_packet(i as u8)).collect::<Vec<_>>()
        );
        assert_eq!(
            d.stream_mut(1).unwrap().take_packets(),
            vec![flac_frame(); 3]
        );
        // Per-stream granules: three 20 ms Opus packets, three 4096-sample blocks.
        assert_eq!(d.streams()[0].end_granule(), Some(3 * 960));
        assert_eq!(d.streams()[1].end_granule(), Some(3 * 4096));
    }

    #[tokio::test]
    async fn the_serial_property_re_keys_every_stream() {
        let mut mux = OggMuxN::new(2);
        mux.set_property("serial", PropValue::Uint(1000)).unwrap();
        assert_eq!(mux.serials(), vec![1000, 1001]);
        assert_eq!(mux.get_property("serial"), Some(PropValue::Uint(1000)));
        assert!(
            mux.set_property("serial", PropValue::Uint(1 << 40))
                .is_err(),
            "a serial number is 32 bits"
        );
    }

    fn opus_caps() -> Caps {
        audio_caps(AudioFormat::Opus, 2, 48_000)
    }

    // mux `packets` through a one-pad muxer configured with `caps`
    async fn mux_one_pad(caps: &Caps, packets: Vec<PipelinePacket>) -> Vec<u8> {
        let mut mux = OggMuxN::new(1);
        mux.configure_pipeline(0, caps).unwrap();
        let mut sink = CaptureSink::default();
        for packet in packets {
            mux.process(0, packet, &mut sink).await.unwrap();
        }
        mux.process(0, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();
        sink.bytes
    }

    #[test]
    fn caps_audio_in_ogg_byte_stream_out() {
        let m = OggMuxN::new(1);
        assert!(m.intercept_caps(0, &opus_caps()).is_ok());
        let aac = audio_caps(AudioFormat::Aac, 2, 48_000);
        assert!(m.intercept_caps(0, &aac).is_err(), "AAC has no Ogg mapping");

        let Ok(CapsConstraint::Produces(set)) = m.caps_constraint_for_output() else {
            panic!("expected Produces");
        };
        assert!(matches!(
            set.alternatives(),
            [Caps::ByteStream {
                encoding: ByteStreamEncoding::Ogg
            }]
        ));
    }

    #[tokio::test]
    async fn opus_round_trips_through_the_demuxer() {
        let packets: Vec<Vec<u8>> = (0..5).map(opus_packet).collect();
        let bytes = mux_one_pad(
            &opus_caps(),
            packets.iter().map(|p| frame(p.clone(), 0)).collect(),
        )
        .await;

        let mut d = OggDemuxer::new();
        d.push_data(&bytes);
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

        let bytes = mux_one_pad(
            &opus_caps(),
            vec![frame(head.clone(), 0), frame(opus_packet(1), 0)],
        )
        .await;

        let mut d = OggDemuxer::new();
        d.push_data(&bytes);
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
        // A short final packet: 400 of its 960 samples are real, the rest padding.
        let bytes = mux_one_pad(
            &opus_caps(),
            vec![
                timed_frame(opus_packet(0), 960),
                timed_frame(opus_packet(1), 400),
            ],
        )
        .await;

        let mut d = OggDemuxer::new();
        d.push_data(&bytes);
        assert_eq!(d.end_granule(), Some(960 + 400));
    }

    /// An untimed stream (a live encoder stamps no duration) keeps the codec's
    /// own sample counts, so the cap never truncates it to nothing.
    #[tokio::test]
    async fn untimed_input_keeps_the_natural_granule() {
        let bytes = mux_one_pad(
            &opus_caps(),
            (0..3).map(|i| frame(opus_packet(i), 0)).collect(),
        )
        .await;

        let mut d = OggDemuxer::new();
        d.push_data(&bytes);
        assert_eq!(d.end_granule(), Some(3 * 960));
    }

    /// Mux `links` runs of Opus packets, a `Segment` between consecutive ones.
    async fn mux_chained(links: &[Vec<Vec<u8>>]) -> Vec<u8> {
        let mut packets = Vec::new();
        for (i, link) in links.iter().enumerate() {
            if i > 0 {
                let start = i as u64 * 1_000_000_000;
                packets.push(PipelinePacket::Segment(Segment {
                    start,
                    base: start,
                    ..Segment::new()
                }));
            }
            packets.extend(link.iter().map(|p| frame(p.clone(), 0)));
        }
        mux_one_pad(&opus_caps(), packets).await
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
        let seg = || PipelinePacket::Segment(Segment::new());
        let bytes = mux_one_pad(
            &opus_caps(),
            vec![
                seg(),
                frame(opus_packet(0), 0),
                seg(),
                frame(opus_packet(1), 0),
            ],
        )
        .await;

        let pages = page_flags(&bytes);
        assert_eq!(
            pages.iter().filter(|(_, ht)| ht & 0x02 != 0).count(),
            1,
            "one logical bitstream"
        );
        let mut d = OggDemuxer::new();
        d.push_data(&bytes);
        assert_eq!(d.chain(), 0);
        assert_eq!(d.end_granule(), Some(2 * 960));
    }

    /// Link serials are derived, not counted: distinct, and the same every run.
    #[test]
    fn chain_serials_are_deterministic_and_distinct() {
        let mut a = OggMuxN::new(1).with_serial(7);
        let mut b = OggMuxN::new(1).with_serial(7);
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
        let rate = 44_100;
        let bytes = mux_one_pad(
            &audio_caps(AudioFormat::Flac, 2, rate),
            vec![frame(flac_header(rate), 0), frame(flac_frame(), 0)],
        )
        .await;

        let mut d = OggDemuxer::new();
        d.push_data(&bytes);
        let info = d.info().expect("mapping header parsed");
        assert_eq!(info.codec, OggCodec::Flac);
        assert_eq!(info.sample_rate, rate);
        assert_eq!(info.channels, 2);
        assert!(
            d.comment_header().is_some(),
            "the mapping's mandatory VorbisComment packet is written"
        );
        assert_eq!(d.take_packets(), vec![flac_frame()]);
        assert_eq!(d.end_granule(), Some(4096), "one 4096-sample block");
    }
}
