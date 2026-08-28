//! WAV encoder: PCM or a companded / ADPCM payload in (`Caps::Audio`), a
//! RIFF/WAVE byte stream out (`Caps::ByteStream{Wav}`), so a separated or
//! synthesized track lands as a file an audio tool opens.
//!
//! ```text
//! ... ! audioconvert ! wavenc ! filesink location=out.wav
//! ```
//!
//! The 44-byte canonical header goes out ahead of the first sample, with the two
//! size fields written as the streaming sentinel `0xFFFFFFFF` (read the data to
//! end of file). A g2g sink writes forward only, so there is no seek back to
//! patch the real sizes in at end of stream, and a sentinel is what a reader
//! copes with; a zero would have it believe the file is empty. Samples then pass
//! through untouched: WAV is a header plus the payload as it arrived.
//!
//! G.711 and IMA ADPCM are written too (M1092), which is what `wavparse` already
//! reads back: `... ! mulawenc ! wavenc` files a telephony recording at half the
//! size. ADPCM's `fmt ` chunk has to state the block size, and `Caps::Audio`
//! carries none, so it comes from the `blockalign` property and must match the
//! `adpcmenc blockalign=` that produced the blocks.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    pcm_formats, AsyncElement, AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet,
    ConfigureOutcome, ElementMetadata, G2gError, MemoryDomain, OutputSink, PadTemplate,
    PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use g2g_mcu::adpcm::{samples_per_block, BLOCK_HEADER};

/// Size fields of a stream whose length is not known when the header is written.
/// A reader takes the `data` chunk as running to the end of the file.
const STREAMING_SIZE: u32 = u32::MAX;

/// `wFormatTag` values: integer PCM, IEEE float for the 32-bit float input, and
/// the companded / ADPCM payloads `wavparse` reads.
const FORMAT_PCM: u16 = 1;
const FORMAT_IEEE_FLOAT: u16 = 3;
const FORMAT_ALAW: u16 = 6;
const FORMAT_MULAW: u16 = 7;
const FORMAT_IMA_ADPCM: u16 = 0x11;

/// The coded payloads a WAV file can carry beside PCM.
const CODED_FORMATS: [AudioFormat; 3] =
    [AudioFormat::Mulaw, AudioFormat::Alaw, AudioFormat::ImaAdpcm];

/// The inputs a WAV file can carry.
fn input_alternatives() -> Vec<Caps> {
    let any_layout = |format| Caps::Audio {
        format,
        channels: 0,
        sample_rate: 0,
    };
    let mut alternatives = Vec::from(pcm_formats().map(any_layout));
    alternatives.extend(CODED_FORMATS.map(any_layout));
    alternatives
}

/// `(wFormatTag, bits per sample)` of a format, `None` for anything WAV cannot
/// describe in a `fmt ` chunk this way.
fn wave_format(format: AudioFormat) -> Option<(u16, u16)> {
    match format {
        AudioFormat::PcmU8 => Some((FORMAT_PCM, 8)),
        AudioFormat::PcmS16Le => Some((FORMAT_PCM, 16)),
        AudioFormat::PcmS24Le => Some((FORMAT_PCM, 24)),
        AudioFormat::PcmS32Le => Some((FORMAT_PCM, 32)),
        AudioFormat::PcmF32Le => Some((FORMAT_IEEE_FLOAT, 32)),
        AudioFormat::Mulaw => Some((FORMAT_MULAW, G711_BITS)),
        AudioFormat::Alaw => Some((FORMAT_ALAW, G711_BITS)),
        AudioFormat::ImaAdpcm => Some((FORMAT_IMA_ADPCM, IMA_ADPCM_BITS)),
        _ => None,
    }
}

/// Bits per sample of the companded G.711 payloads and of an ADPCM nibble, the
/// numbers their `fmt ` chunks state.
const G711_BITS: u16 = 8;
const IMA_ADPCM_BITS: u16 = 4;

/// The RIFF/WAVE header for this stream: 44 bytes for PCM (the canonical
/// layout), 46 for a companded payload (`cbSize` 0, as ffmpeg writes) and 48 for
/// ADPCM (`cbSize` 2 plus the samples a block holds).
fn header(
    format: AudioFormat,
    channels: u16,
    sample_rate: u32,
    block_align_property: usize,
) -> Option<Vec<u8>> {
    let (tag, bits) = wave_format(format)?;
    let samples_per_block = match format {
        AudioFormat::ImaAdpcm => Some(samples_per_block(block_align_property)),
        _ => None,
    };
    let block_align = match format {
        // One block, not one sample, is ADPCM's addressable unit.
        AudioFormat::ImaAdpcm => u16::try_from(block_align_property).ok()?,
        _ => channels * bits / 8,
    };
    let byte_rate = match samples_per_block {
        // The coded rate: how many block-sized chunks a second of audio needs.
        // Folded in one expression, since a block holds thousands of samples and
        // dividing first would round the rate away. (ffmpeg writes the decoded
        // PCM rate in this field instead, which readers ignore either way.)
        Some(samples) => u32::try_from(
            u64::from(sample_rate) * u64::from(channels) * u64::from(block_align) / samples as u64,
        )
        .ok()?,
        None => sample_rate * block_align as u32,
    };
    // The `fmt ` extension: absent for PCM, empty for G.711, one field for ADPCM.
    let extension: Vec<u8> = match samples_per_block {
        Some(samples) => {
            let mut bytes = Vec::from(2u16.to_le_bytes());
            bytes.extend_from_slice(&u16::try_from(samples).ok()?.to_le_bytes());
            bytes
        }
        None if tag == FORMAT_PCM || tag == FORMAT_IEEE_FLOAT => Vec::new(),
        None => Vec::from(0u16.to_le_bytes()),
    };
    /// The fields every `fmt ` chunk carries, before any extension.
    const FMT_FIXED_LEN: u32 = 16;

    let mut out = Vec::with_capacity(48);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&STREAMING_SIZE.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&(FMT_FIXED_LEN + extension.len() as u32).to_le_bytes());
    out.extend_from_slice(&tag.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(&extension);
    out.extend_from_slice(b"data");
    out.extend_from_slice(&STREAMING_SIZE.to_le_bytes());
    Some(out)
}

/// The WAV byte stream this element produces.
fn wav_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::Wav,
    }
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::wavenc::WavEnc;
///
/// let element = WavEnc::new();
/// ```
#[derive(Debug)]
pub struct WavEnc {
    /// The negotiated stream, captured at configure time so the header can
    /// describe it.
    input: Option<(AudioFormat, u16, u32)>,
    /// ADPCM's block size, which the `fmt ` chunk states and the caps cannot
    /// carry. Ignored by every other format.
    block_align: usize,
    header_written: bool,
    configured: bool,
    emitted: u64,
}

impl Default for WavEnc {
    fn default() -> Self {
        Self {
            input: None,
            block_align: crate::adpcm::DEFAULT_BLOCK_ALIGN,
            header_written: false,
            configured: false,
            emitted: 0,
        }
    }
}

impl WavEnc {
    pub fn new() -> Self {
        Self::default()
    }

    /// The ADPCM block size the header states; must match the `adpcmenc` that
    /// produced the blocks.
    pub fn with_block_align(mut self, bytes: usize) -> Self {
        if bytes > BLOCK_HEADER {
            self.block_align = bytes;
        }
        self
    }

    /// Count of packets pushed downstream, the header included.
    pub fn emitted_count(&self) -> u64 {
        self.emitted
    }

    fn frame(&mut self, bytes: Vec<u8>) -> PipelinePacket {
        self.emitted += 1;
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            timing: Default::default(),
            sequence: self.emitted,
            meta: Default::default(),
        })
    }
}

impl AsyncElement for WavEnc {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn is_format_boundary(&self) -> bool {
        true
    }

    fn properties(&self) -> &'static [PropertySpec] {
        static PROPS: &[PropertySpec] = &[PropertySpec::new(
            "blockalign",
            PropKind::Uint,
            "ADPCM block size the `fmt ` chunk states, which must match the `adpcmenc blockalign=` that produced the blocks (ignored for every other format)",
        )
        .with_default("1024")];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "blockalign" => {
                let bytes = value.as_uint().ok_or(PropError::Type)?;
                let bytes = usize::try_from(bytes).map_err(|_| PropError::Value)?;
                // A block smaller than its own header holds no samples, and the
                // field is 16-bit in the chunk.
                if bytes <= BLOCK_HEADER || u16::try_from(bytes).is_err() {
                    return Err(PropError::Value);
                }
                self.block_align = bytes;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "blockalign" => Some(PropValue::Uint(self.block_align as u64)),
            _ => None,
        }
    }

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    /// The input side: the PCM this can describe in a `fmt ` chunk, at the rate
    /// and channel count upstream negotiated.
    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::Audio { format, .. } if wave_format(*format).is_some() => {
                Ok(upstream_caps.clone())
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Audio { format, .. } if wave_format(*format).is_some() => {
                CapsSet::one(wav_caps())
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::Audio {
            format,
            channels,
            sample_rate,
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if wave_format(*format).is_none() {
            return Err(G2gError::CapsMismatch);
        }
        // A coded stream negotiates at the `ANY_CHANNELS` / `ANY_SAMPLE_RATE`
        // sentinels and states its real layout in a runtime `CapsChanged`, which
        // still arrives before the first frame, so the header can wait for it.
        self.input = Some((*format, *channels as u16, *sample_rate));
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
                    let (format, channels, sample_rate) =
                        self.input.ok_or(G2gError::NotConfigured)?;
                    if !self.header_written {
                        // Nothing in a `fmt ` chunk can say "unknown", so a
                        // layout that never arrived fails here rather than
                        // writing a header of zeroes.
                        if channels == 0 || sample_rate == 0 {
                            return Err(G2gError::CapsMismatch);
                        }
                        out.push(PipelinePacket::CapsChanged(wav_caps())).await?;
                        let bytes = header(format, channels, sample_rate, self.block_align)
                            .ok_or(G2gError::CapsMismatch)?;
                        let packet = self.frame(bytes);
                        out.push(packet).await?;
                        self.header_written = true;
                    }
                    let samples = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let packet = self.frame(samples.to_vec());
                    out.push(packet).await?;
                }
                // The real layout of a coded stream arrives here, and the
                // header needs it, so take it while the file can still change.
                // Once written, a change would need a second header, which a
                // written file cannot take: the output is the WAV stream this
                // element already announced, so the caps are not forwarded.
                PipelinePacket::CapsChanged(caps) => {
                    if !self.header_written {
                        if let Caps::Audio {
                            format,
                            channels,
                            sample_rate,
                        } = caps
                        {
                            if wave_format(format).is_some() {
                                self.input = Some((format, channels as u16, sample_rate));
                            }
                        }
                    }
                }
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "WAV encoder",
            "Codec/Muxer/Audio",
            "Wraps raw PCM in a RIFF/WAVE byte stream",
            "g2g",
        )
    }
}

impl PadTemplates for WavEnc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::from_alternatives(input_alternatives())),
            PadTemplate::source(CapsSet::one(wav_caps())),
        ])
    }
}
