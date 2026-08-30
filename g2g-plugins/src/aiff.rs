//! AIFF / AIFC: `aiffparse` reads a `FORM`/`AIFF` (or `AIFC`) byte stream into
//! PCM, `aiffmux` writes one. The Mac / interchange sibling of WAVE.
//!
//! ```text
//! filesrc location=in.aiff ! aiffparse ! audioconvert ! autoaudiosink
//! audiotestsrc ! aiffmux ! filesink location=out.aiff
//! ```
//!
//! EA IFF 85 is a chunk list: a 12-byte `FORM` + size + form-type header, then
//! `id` + size chunks whose sizes are big-endian. `COMM` names the sample format
//! (channels, bit depth, an 80-bit IEEE extended rate); `SSND` is the samples,
//! optionally with an offset into the sound data. Anything else (`NAME`, `AUTH`,
//! `COMT`, an AIFC `FVER`) is skipped by its declared size. Sizes come from the
//! file, so they are read with checked arithmetic and a chunk that overruns
//! fails the parse.
//!
//! Multi-byte PCM is big-endian on the wire (AIFC `sowt` is the little-endian
//! exception) and 8-bit samples are signed. Both are converted at this boundary
//! so the rest of the graph sees the little-endian / unsigned `AudioFormat`s.
//! The muxer holds the samples until EOS so the FORM, COMM, and SSND sizes are
//! valid before the header is sent to a forward-only sink.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, G2gError, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
};

use crate::audiocontainer::{
    container_caps, emit_pcm, mux_constraint, mux_input_alternatives, parse_constraint,
    parse_output_alternatives, PcmContainer, PcmMux, MAX_CHANNELS, MAX_SAMPLE_RATE, STREAMING_SIZE,
};
use crate::pcmendian::{read_fourcc, read_u16_be, read_u32_be, PcmWire};

/// Holds the samples until EOS so the FORM, COMM, and SSND sizes are final, and
/// pads the sound data to an even length as EA IFF 85 requires.
static AIFF_CONTAINER: PcmContainer = PcmContainer {
    encoding: ByteStreamEncoding::Aiff,
    supported,
    header,
    finalize_at_eos: true,
    data_alignment: 2,
};

fn aiff_caps() -> Caps {
    container_caps(ByteStreamEncoding::Aiff)
}

/// `FORM` + size + form type.
const FORM_HEADER_LEN: usize = 12;
/// A chunk's 4-byte id plus its 4-byte big-endian size.
const CHUNK_HEADER_LEN: usize = 8;
/// The fields every `COMM` chunk carries before any AIFC compression type.
const COMM_FIXED_LEN: usize = 18;
/// `SSND` offset + blockSize prefix ahead of the samples.
const SSND_PREFIX_LEN: usize = 8;
/// Largest header this accepts: a skipped chunk body (`COMM`, `NAME`, `AUTH`)
/// or the `SSND` offset ahead of the samples. More than this is corrupt, not a
/// comment.
const MAX_HEADER_CHUNK: usize = 1 << 20;

/// A chunk body is padded to an even length. `None` on overflow.
fn padded_len(size: usize) -> Option<usize> {
    size.checked_add(size % 2)
}

/// Integer sample rate of an Apple SANE 80-bit IEEE extended value, `None` when
/// the encoding is not a positive integer in `1..=MAX_SAMPLE_RATE`.
fn ieee80_to_u32(bytes: &[u8]) -> Option<u32> {
    let bytes: &[u8; 10] = bytes.get(..10)?.try_into().ok()?;
    if bytes[0] & 0x80 != 0 {
        return None;
    }
    let exp = u16::from_be_bytes([bytes[0] & 0x7f, bytes[1]]);
    if exp == 0 || exp == 0x7fff {
        return None;
    }
    let mantissa = u64::from_be_bytes(bytes[2..10].try_into().ok()?);
    if mantissa == 0 {
        return None;
    }
    let unbiased = i32::from(exp) - 16383;
    let shift = unbiased - 63;
    let value = if shift >= 0 {
        mantissa.checked_shl(shift as u32)?
    } else {
        let right = (-shift) as u32;
        if right >= 64 {
            return None;
        }
        if mantissa & ((1u64 << right) - 1) != 0 {
            return None;
        }
        mantissa >> right
    };
    let rate = u32::try_from(value).ok()?;
    (rate > 0 && rate <= MAX_SAMPLE_RATE).then_some(rate)
}

/// The 80-bit Apple extended encoding of a positive integer sample rate.
fn u32_to_ieee80(rate: u32) -> [u8; 10] {
    if rate == 0 {
        return [0; 10];
    }
    let unbiased = 31 - rate.leading_zeros();
    let exp = (16383 + unbiased) as u16;
    let mantissa = (u64::from(rate)) << (63 - unbiased);
    let mut out = [0u8; 10];
    out[0] = (exp >> 8) as u8;
    out[1] = exp as u8;
    out[2..10].copy_from_slice(&mantissa.to_be_bytes());
    out
}

/// The graph format an uncompressed `COMM` bit depth describes.
fn pcm_from_bits(bits: u16) -> Option<AudioFormat> {
    Some(match bits {
        8 => AudioFormat::PcmU8,
        16 => AudioFormat::PcmS16Le,
        24 => AudioFormat::PcmS24Le,
        32 => AudioFormat::PcmS32Le,
        _ => return None,
    })
}

/// The graph format a (`bits`, AIFC compression) pair describes, plus whether
/// multi-byte samples are big-endian on the wire. `None` for an unmodeled
/// payload (ima4, GSM, 64-bit float).
fn audio_from_comm(bits: u16, compression: Option<[u8; 4]>) -> Option<(AudioFormat, bool)> {
    let fourcc = compression.as_ref();
    let sowt = fourcc == Some(b"sowt");
    let linear = sowt || fourcc.is_none() || fourcc == Some(b"NONE") || fourcc == Some(b"none");
    if linear {
        return Some((pcm_from_bits(bits)?, !sowt));
    }
    match fourcc {
        Some(b"fl32") | Some(b"FL32") if bits == 32 => Some((AudioFormat::PcmF32Le, true)),
        Some(b"ulaw") | Some(b"ULAW") => Some((AudioFormat::Mulaw, false)),
        Some(b"alaw") | Some(b"ALAW") => Some((AudioFormat::Alaw, false)),
        _ => None,
    }
}

fn parse_comm(body: &[u8], aifc: bool) -> Option<PcmWire> {
    if body.len() < COMM_FIXED_LEN {
        return None;
    }
    let channels = read_u16_be(body, 0)?;
    let bits = read_u16_be(body, 6)?;
    let sample_rate = ieee80_to_u32(body.get(8..18)?)?;
    if channels == 0 || channels > MAX_CHANNELS {
        return None;
    }
    let compression = if aifc {
        if body.len() < COMM_FIXED_LEN + 4 {
            return None;
        }
        Some(read_fourcc(body, COMM_FIXED_LEN)?)
    } else {
        None
    };
    let (format, big_endian) = audio_from_comm(bits, compression)?;
    Some(PcmWire {
        format,
        channels: channels as u8,
        sample_rate,
        big_endian,
    })
}

/// The AIFC compression type (and whether the FORM is AIFC) for a graph format.
fn wire_kind(format: AudioFormat) -> Option<(&'static [u8; 4], u16, bool)> {
    match format {
        AudioFormat::PcmU8 => Some((b"NONE", 8, false)),
        AudioFormat::PcmS16Le => Some((b"NONE", 16, false)),
        AudioFormat::PcmS24Le => Some((b"NONE", 24, false)),
        AudioFormat::PcmS32Le => Some((b"NONE", 32, false)),
        AudioFormat::PcmF32Le => Some((b"fl32", 32, true)),
        AudioFormat::Mulaw => Some((b"ulaw", 8, true)),
        AudioFormat::Alaw => Some((b"alaw", 8, true)),
        _ => None,
    }
}

/// The FORM/COMM/SSND header for this complete stream.
fn header(
    format: AudioFormat,
    channels: u16,
    sample_rate: u32,
    payload_size: usize,
) -> Option<Vec<u8>> {
    let (compression, bits, aifc) = wire_kind(format)?;
    let comm_ext: &[u8] = if aifc {
        // compression type plus an empty Pascal name (length 0, pad byte).
        &[
            compression[0],
            compression[1],
            compression[2],
            compression[3],
            0,
            0,
        ]
    } else {
        &[]
    };
    let comm_len = (COMM_FIXED_LEN + comm_ext.len()) as u32;
    let frame_bytes = crate::pcmendian::sample_width(format)?.checked_mul(channels as usize)?;
    if frame_bytes == 0 {
        return None;
    }
    let sample_frames = u32::try_from(payload_size / frame_bytes).ok()?;
    let ssnd_size = SSND_PREFIX_LEN.checked_add(payload_size)?;
    let padded_ssnd_size = padded_len(ssnd_size)?;
    let comm_chunk_size = CHUNK_HEADER_LEN.checked_add(comm_len as usize)?;
    let sound_chunk_size = CHUNK_HEADER_LEN.checked_add(padded_ssnd_size)?;
    let form_size = 4usize
        .checked_add(comm_chunk_size)?
        .checked_add(sound_chunk_size)?;
    let form = if aifc { *b"AIFC" } else { *b"AIFF" };
    let mut out = Vec::with_capacity(FORM_HEADER_LEN + CHUNK_HEADER_LEN + comm_len as usize + 16);
    out.extend_from_slice(b"FORM");
    out.extend_from_slice(&u32::try_from(form_size).ok()?.to_be_bytes());
    out.extend_from_slice(&form);
    out.extend_from_slice(b"COMM");
    out.extend_from_slice(&comm_len.to_be_bytes());
    out.extend_from_slice(&channels.to_be_bytes());
    out.extend_from_slice(&sample_frames.to_be_bytes());
    out.extend_from_slice(&bits.to_be_bytes());
    out.extend_from_slice(&u32_to_ieee80(sample_rate));
    out.extend_from_slice(comm_ext);
    out.extend_from_slice(b"SSND");
    out.extend_from_slice(&u32::try_from(ssnd_size).ok()?.to_be_bytes());
    out.extend_from_slice(&0u32.to_be_bytes()); // offset
    out.extend_from_slice(&0u32.to_be_bytes()); // blockSize
    Some(out)
}

/// Parses an AIFF / AIFC byte stream into the PCM stream it carries.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::aiff::AiffParse;
///
/// let parse = AiffParse::new();
/// ```
#[derive(Debug, Default)]
pub struct AiffParse {
    configured: bool,
    buf: Vec<u8>,
    form_seen: bool,
    aifc: bool,
    format: Option<PcmWire>,
    ssnd_prefix_pending: bool,
    ssnd_offset_remaining: usize,
    data_remaining: Option<usize>,
    in_data: bool,
    emitted: u64,
}

impl AiffParse {
    pub fn new() -> Self {
        Self::default()
    }

    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if !self.form_seen {
            if self.buf.len() < FORM_HEADER_LEN {
                return Ok(());
            }
            if read_fourcc(&self.buf, 0) != Some(*b"FORM") {
                return Err(G2gError::CapsMismatch);
            }
            let form = read_fourcc(&self.buf, 8).ok_or(G2gError::CapsMismatch)?;
            self.aifc = match &form {
                b"AIFF" => false,
                b"AIFC" => true,
                _ => return Err(G2gError::CapsMismatch),
            };
            self.buf.drain(..FORM_HEADER_LEN);
            self.form_seen = true;
        }
        while !self.in_data {
            if self.buf.len() < CHUNK_HEADER_LEN {
                return Ok(());
            }
            let id = read_fourcc(&self.buf, 0).ok_or(G2gError::CapsMismatch)?;
            let size = read_u32_be(&self.buf, 4).ok_or(G2gError::CapsMismatch)? as usize;
            // SSND is the sample payload and is not buffered whole.
            if &id != b"SSND" && size > MAX_HEADER_CHUNK {
                return Err(G2gError::CapsMismatch);
            }
            if &id == b"SSND" {
                let wire = self.format.ok_or(G2gError::CapsMismatch)?;
                if size < SSND_PREFIX_LEN {
                    return Err(G2gError::CapsMismatch);
                }
                self.buf.drain(..CHUNK_HEADER_LEN);
                self.ssnd_prefix_pending = true;
                self.data_remaining =
                    (size != STREAMING_SIZE as usize).then_some(size - SSND_PREFIX_LEN);
                out.push(PipelinePacket::CapsChanged(Caps::Audio {
                    format: wire.format,
                    channels: wire.channels,
                    sample_rate: wire.sample_rate,
                }))
                .await?;
                self.in_data = true;
                break;
            }
            let padded = padded_len(size).ok_or(G2gError::CapsMismatch)?;
            let total = padded
                .checked_add(CHUNK_HEADER_LEN)
                .ok_or(G2gError::CapsMismatch)?;
            if self.buf.len() < total {
                return Ok(());
            }
            if &id == b"COMM" {
                self.format = Some(
                    parse_comm(
                        &self.buf[CHUNK_HEADER_LEN..CHUNK_HEADER_LEN + size],
                        self.aifc,
                    )
                    .ok_or(G2gError::CapsMismatch)?,
                );
            }
            self.buf.drain(..total);
        }
        if self.ssnd_prefix_pending {
            if self.buf.len() < SSND_PREFIX_LEN {
                return Ok(());
            }
            let offset = read_u32_be(&self.buf, 0).ok_or(G2gError::CapsMismatch)? as usize;
            if offset > MAX_HEADER_CHUNK {
                return Err(G2gError::CapsMismatch);
            }
            if let Some(remaining) = self.data_remaining.as_mut() {
                *remaining = remaining
                    .checked_sub(offset)
                    .ok_or(G2gError::CapsMismatch)?;
            }
            self.buf.drain(..SSND_PREFIX_LEN);
            self.ssnd_prefix_pending = false;
            self.ssnd_offset_remaining = offset;
        }
        if self.ssnd_offset_remaining > 0 {
            let skipped = self.ssnd_offset_remaining.min(self.buf.len());
            self.buf.drain(..skipped);
            self.ssnd_offset_remaining -= skipped;
            if self.ssnd_offset_remaining > 0 {
                return Ok(());
            }
        }
        let wire = self.format.ok_or(G2gError::CapsMismatch)?;
        emit_pcm(
            &mut self.buf,
            &mut self.data_remaining,
            &wire,
            &mut self.emitted,
            out,
        )
        .await
    }
}

impl AsyncElement for AiffParse {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "AIFF parser",
            "Codec/Demuxer/Audio",
            "Reads the PCM stream out of an AIFF / AIFC byte stream",
            "g2g",
        )
    }

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&aiff_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        parse_constraint(ByteStreamEncoding::Aiff)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(
            absolute_caps,
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Aiff
            }
        ) {
            return Err(G2gError::CapsMismatch);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
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
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    self.buf.extend_from_slice(slice);
                    self.drain(out).await?;
                }
                PipelinePacket::Eos => {
                    self.drain(out).await?;
                }
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for AiffParse {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(aiff_caps())),
            PadTemplate::source(parse_output_alternatives()),
        ])
    }
}

/// Writes a PCM stream as an AIFF / AIFC byte stream.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::aiff::AiffMux;
///
/// let element = AiffMux::new();
/// ```
#[derive(Debug, Default)]
pub struct AiffMux {
    inner: PcmMux,
}

impl AiffMux {
    pub fn new() -> Self {
        Self::default()
    }
}

fn supported(format: AudioFormat) -> bool {
    wire_kind(format).is_some()
}

impl AsyncElement for AiffMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn is_format_boundary(&self) -> bool {
        true
    }

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        PcmMux::intercept(upstream_caps, &AIFF_CONTAINER)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        mux_constraint(&AIFF_CONTAINER)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.inner.configure(absolute_caps, &AIFF_CONTAINER)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        self.inner.process(packet, out, &AIFF_CONTAINER)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "AIFF encoder",
            "Codec/Muxer/Audio",
            "Wraps raw PCM in an AIFF / AIFC byte stream",
            "g2g",
        )
    }
}

impl PadTemplates for AiffMux {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::from_alternatives(mux_input_alternatives())),
            PadTemplate::source(CapsSet::one(aiff_caps())),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcmendian::{convert_wire_layout, sample_width, CARRIED};
    use crate::testutil::{data_bytes, first_caps, frame, roundtrip, run, CollectSink};
    use crate::typefind::sniff;
    use alloc::vec::Vec;
    use g2g_core::AudioFormat;

    /// Apple SANE encoding of 44100 Hz, from the AIFF spec's example rate.
    const RATE_44100_EXTENDED: [u8; 10] = [0x40, 0x0E, 0xAC, 0x44, 0, 0, 0, 0, 0, 0];

    fn audio(format: AudioFormat, channels: u8, rate: u32) -> Caps {
        Caps::Audio {
            format,
            channels,
            sample_rate: rate,
        }
    }

    fn samples(format: AudioFormat, channels: u8, frames: usize) -> Vec<u8> {
        let n = sample_width(format).unwrap() * channels as usize * frames;
        (0..n).map(|i| i as u8).collect()
    }

    #[test]
    fn extended_rate_round_trips_common_rates() {
        assert_eq!(u32_to_ieee80(44_100), RATE_44100_EXTENDED);
        for rate in [8_000, 11_025, 16_000, 22_050, 44_100, 48_000, 96_000] {
            assert_eq!(ieee80_to_u32(&u32_to_ieee80(rate)), Some(rate));
        }
    }

    #[test]
    fn a_bogus_extended_rate_is_rejected() {
        assert_eq!(ieee80_to_u32(&[0; 10]), None);
        let mut inf = [0u8; 10];
        inf[0] = 0x7F;
        inf[1] = 0xFF;
        inf[2] = 0x80;
        assert_eq!(ieee80_to_u32(&inf), None);
        let mut frac = RATE_44100_EXTENDED;
        *frac.last_mut().unwrap() = 1;
        assert_eq!(ieee80_to_u32(&frac), None);
    }

    #[test]
    fn mux_header_is_form_then_comm_then_ssnd() {
        for format in CARRIED {
            let channels = 2u16;
            let rate = 48_000u32;
            let bytes = header(format, channels, rate, 0).unwrap();
            assert_eq!(&bytes[..4], b"FORM");
            let form = &bytes[8..12];
            assert!(form == b"AIFF" || form == b"AIFC", "{format:?}");
            assert!(bytes.windows(4).any(|w| w == b"COMM"), "{format:?}");
            assert!(bytes.windows(4).any(|w| w == b"SSND"), "{format:?}");
            let comm = bytes.windows(4).position(|w| w == b"COMM").unwrap();
            assert_eq!(
                u16::from_be_bytes(bytes[comm + 8..comm + 10].try_into().unwrap()),
                channels
            );
            assert_eq!(
                ieee80_to_u32(&bytes[comm + 16..comm + 26]),
                Some(rate),
                "{format:?}"
            );
        }
    }

    #[test]
    fn every_carried_format_round_trips() {
        for format in CARRIED {
            for channels in [1u8, 2] {
                let pcm = samples(format, channels, 8);
                roundtrip(
                    AiffMux::new(),
                    AiffParse::new(),
                    audio(format, channels, 48_000),
                    aiff_caps(),
                    &pcm,
                );
            }
        }
    }

    #[test]
    fn a_written_file_types_as_aiff() {
        let out = run(
            &mut AiffMux::new(),
            &audio(AudioFormat::PcmS16Le, 1, 8_000),
            &[&samples(AudioFormat::PcmS16Le, 1, 4)],
        );
        assert_eq!(
            sniff(&data_bytes(&out.packets)),
            Some(ByteStreamEncoding::Aiff)
        );
    }

    #[test]
    fn mux_writes_final_form_sample_and_sound_sizes() {
        let pcm = samples(AudioFormat::PcmS16Le, 2, 8);
        let written = run(
            &mut AiffMux::new(),
            &audio(AudioFormat::PcmS16Le, 2, 48_000),
            &[&pcm],
        );
        let file = data_bytes(&written.packets);
        assert_eq!(read_u32_be(&file, 4), Some((file.len() - 8) as u32));

        let comm = file.windows(4).position(|bytes| bytes == b"COMM").unwrap();
        assert_eq!(read_u32_be(&file, comm + 10), Some(8));

        let ssnd = file.windows(4).position(|bytes| bytes == b"SSND").unwrap();
        assert_eq!(
            read_u32_be(&file, ssnd + 4),
            Some((SSND_PREFIX_LEN + pcm.len()) as u32)
        );
    }

    #[test]
    fn a_split_header_still_parses() {
        let written = run(
            &mut AiffMux::new(),
            &audio(AudioFormat::PcmS16Le, 1, 8_000),
            &[&samples(AudioFormat::PcmS16Le, 1, 8)],
        );
        let file = data_bytes(&written.packets);
        let mut parse = AiffParse::new();
        let chunks: Vec<&[u8]> = file.chunks(3).collect();
        let parsed = run(&mut parse, &aiff_caps(), &chunks);
        assert_eq!(
            first_caps(&parsed.packets),
            Some(audio(AudioFormat::PcmS16Le, 1, 8_000))
        );
        assert_eq!(
            data_bytes(&parsed.packets),
            samples(AudioFormat::PcmS16Le, 1, 8)
        );
    }

    #[test]
    fn ssnd_offset_split_across_buffers_is_skipped_once() {
        let pcm = samples(AudioFormat::PcmS16Le, 1, 4);
        let mut wire_pcm = pcm.clone();
        convert_wire_layout(&mut wire_pcm, AudioFormat::PcmS16Le, true);
        let offset = [9u8, 8, 7, 6];
        let mut file = header(AudioFormat::PcmS16Le, 1, 8_000, 0).unwrap();
        let ssnd = file.windows(4).position(|bytes| bytes == b"SSND").unwrap();
        file[ssnd + 4..ssnd + 8].copy_from_slice(
            &((SSND_PREFIX_LEN + offset.len() + wire_pcm.len()) as u32).to_be_bytes(),
        );
        file[ssnd + 8..ssnd + 12].copy_from_slice(&(offset.len() as u32).to_be_bytes());

        let mut parse = AiffParse::new();
        let parsed = run(&mut parse, &aiff_caps(), &[&file, &offset, &wire_pcm]);
        assert_eq!(data_bytes(&parsed.packets), pcm);
    }

    #[test]
    fn declared_ssnd_size_excludes_trailing_chunks() {
        let pcm = samples(AudioFormat::PcmS16Le, 1, 4);
        let mut wire_pcm = pcm.clone();
        convert_wire_layout(&mut wire_pcm, AudioFormat::PcmS16Le, true);
        let mut file = header(AudioFormat::PcmS16Le, 1, 8_000, 0).unwrap();
        let ssnd = file.windows(4).position(|bytes| bytes == b"SSND").unwrap();
        file[ssnd + 4..ssnd + 8]
            .copy_from_slice(&((SSND_PREFIX_LEN + wire_pcm.len()) as u32).to_be_bytes());
        file.extend_from_slice(&wire_pcm);
        file.extend_from_slice(b"NAME\0\0\0\x04junk");

        let parsed = run(&mut AiffParse::new(), &aiff_caps(), &[&file]);
        assert_eq!(data_bytes(&parsed.packets), pcm);
    }

    #[test]
    fn a_truncated_header_emits_nothing() {
        let out = run(&mut AiffParse::new(), &aiff_caps(), &[b"FORM"]);
        assert!(out.packets.is_empty());
    }

    #[tokio::test]
    async fn mux_rejects_a_partial_sample_frame_before_writing() {
        let mut mux = AiffMux::new();
        mux.configure_pipeline(&audio(AudioFormat::PcmS16Le, 2, 8_000))
            .unwrap();
        let mut output = CollectSink::default();
        let result = mux
            .process(
                PipelinePacket::DataFrame(frame(alloc::vec![0, 1, 2])),
                &mut output,
            )
            .await;
        assert!(matches!(result, Err(G2gError::CapsMismatch)));
        assert!(output.packets.is_empty());
    }
}
