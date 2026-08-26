//! IMA ADPCM elements (M1073): `adpcmenc` packs mono `Audio{PcmS16Le}` into
//! the WAV / DVI block layout at 4 bits per sample, `adpcmdec` unpacks it.
//! The block math is [`g2g_mcu::adpcm`], the heap-free MCU codec validated
//! bit-exact against ffmpeg's `adpcm_ima_wav`; these elements are the `alloc`
//! host wrappers around it.
//!
//! `layout=dvi` is the only layout g2g models, so it is not a property the way
//! GStreamer's `adpcmenc` has one. Blocks are self-contained and mono, so a
//! stream is a run of `blockalign`-sized blocks: the encoder buffers samples
//! until a whole block is available (padding the last one with silence at EOS,
//! as ffmpeg does) and the decoder buffers bytes until a whole block is, so a
//! block split across input frames is carried over rather than dropped.
//!
//! `Caps::Audio` has no block-size field, so the decoder cannot read the WAV
//! `fmt ` chunk's block align off its input caps: a file written with a
//! non-default block size needs `adpcmdec blockalign=N` on the launch line.
//!
//! The encoder takes mono only. An unknown channel count fixates to stereo, so
//! a source that announces its layout mid-stream (`wavparse`) needs an explicit
//! `audioconvert channels=1` ahead of `adpcmenc`.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec, ANY_CHANNELS, ANY_SAMPLE_RATE,
};

use g2g_mcu::adpcm::{decode_block, encode_block, samples_per_block};

/// Bytes per block, the `blockalign` default: what WAV writers (ffmpeg's
/// `adpcm_ima_wav`, GStreamer's `adpcmenc`) use.
pub const DEFAULT_BLOCK_ALIGN: usize = 1024;
/// Smallest accepted `blockalign`, GStreamer's lower bound.
const MIN_BLOCK_ALIGN: usize = 64;
/// Largest accepted `blockalign`, GStreamer's upper bound.
const MAX_BLOCK_ALIGN: usize = 8192;
/// The only channel layout the block math covers.
const ADPCM_CHANNELS: u8 = 1;
/// The rate the decoded caps fixate on while nothing has announced the real one
/// (the placeholder `wavparse` negotiates with): the stream's rate arrives in a
/// `CapsChanged` before the first block is decoded.
const FIXATE_SAMPLE_RATE_HZ: u32 = 48_000;
/// Bytes per S16LE sample, the width both sides convert against.
const PCM_SAMPLE_BYTES: usize = 2;

/// The `blockalign` property, shared by both elements: the byte size of one
/// IMA block, which fixes how many samples it carries.
const BLOCK_ALIGN_PROPERTY: &[PropertySpec] =
    &[
        PropertySpec::new("blockalign", PropKind::Uint, "bytes per IMA ADPCM block")
            .with_default("1024")
            .with_range("64", "8192"),
    ];

/// Read a `blockalign` property value, rejecting anything outside the accepted
/// range.
fn parse_block_align(value: PropValue) -> Result<usize, PropError> {
    let bytes = value.as_uint().ok_or(PropError::Type)?;
    if !(MIN_BLOCK_ALIGN as u64..=MAX_BLOCK_ALIGN as u64).contains(&bytes) {
        return Err(PropError::Value);
    }
    Ok(bytes as usize)
}

/// Mono ADPCM caps at the given layout.
fn coded_caps(sample_rate: u32) -> Caps {
    Caps::Audio {
        format: AudioFormat::ImaAdpcm,
        channels: ADPCM_CHANNELS,
        sample_rate,
    }
}

/// Encodes mono interleaved S16LE PCM into IMA ADPCM blocks.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::adpcm::AdpcmEnc;
///
/// let encoder = AdpcmEnc::new().with_block_align(512);
/// ```
#[derive(Debug)]
pub struct AdpcmEnc {
    block_align: usize,
    sample_rate: u32,
    /// S16LE bytes not yet packed into a whole block.
    pending: Vec<u8>,
    /// The step index carried across blocks, as the reference encoder does.
    step_index: u8,
    /// PTS of the next block, anchored to the first input frame's PTS.
    next_pts_ns: Option<u64>,
    last_out: Option<Caps>,
    sequence: u64,
    configured: bool,
}

impl Default for AdpcmEnc {
    fn default() -> Self {
        Self::new()
    }
}

impl AdpcmEnc {
    pub fn new() -> Self {
        Self {
            block_align: DEFAULT_BLOCK_ALIGN,
            sample_rate: 0,
            pending: Vec::new(),
            step_index: 0,
            next_pts_ns: None,
            last_out: None,
            sequence: 0,
            configured: false,
        }
    }

    /// Set the byte size of one emitted block (64..=8192, default 1024).
    /// Out-of-range values clamp; the `blockalign` property rejects them.
    pub fn with_block_align(mut self, bytes: usize) -> Self {
        self.block_align = bytes.clamp(MIN_BLOCK_ALIGN, MAX_BLOCK_ALIGN);
        self
    }

    /// The byte size of one emitted block.
    pub fn block_align(&self) -> usize {
        self.block_align
    }

    /// The PCM shape this encoder takes: mono interleaved S16LE. An unknown
    /// channel count is refused rather than assumed, since it fixates to stereo.
    fn pcm_shape(caps: &Caps) -> Option<u32> {
        match caps {
            Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: ADPCM_CHANNELS,
                sample_rate,
            } => Some(*sample_rate),
            _ => None,
        }
    }

    /// Bytes of S16LE input one block consumes.
    fn source_block_bytes(&self) -> usize {
        samples_per_block(self.block_align) * PCM_SAMPLE_BYTES
    }

    /// Nanoseconds of audio one block carries.
    fn block_duration_ns(&self) -> u64 {
        samples_per_block(self.block_align) as u64 * 1_000_000_000 / self.sample_rate.max(1) as u64
    }

    /// Encode every whole block the pending samples hold, and emit each as a
    /// frame. `flush` pads a partial tail with silence first, so no audio is
    /// lost at EOS.
    async fn drain(&mut self, flush: bool, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let source_block = self.source_block_bytes();
        if flush && !self.pending.is_empty() {
            self.pending
                .resize(self.pending.len().next_multiple_of(source_block), 0);
        }
        while self.pending.len() >= source_block {
            let mut block = alloc::vec![0u8; self.block_align];
            let samples: Vec<u8> = self.pending.drain(..source_block).collect();
            self.step_index = encode_block(self.step_index, &samples, &mut block)
                .ok_or(G2gError::CapsMismatch)?;
            let new_caps = coded_caps(self.sample_rate);
            if self.last_out.as_ref() != Some(&new_caps) {
                out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                    .await?;
                self.last_out = Some(new_caps);
            }
            let pts_ns = self.next_pts_ns.unwrap_or(0);
            let duration_ns = self.block_duration_ns();
            self.next_pts_ns = Some(pts_ns + duration_ns);
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(block.into_boxed_slice())),
                FrameTiming {
                    pts_ns,
                    dts_ns: pts_ns,
                    duration_ns,
                    ..FrameTiming::default()
                },
                self.sequence,
            );
            self.sequence += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }
}

impl AsyncElement for AdpcmEnc {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    /// Mono S16LE only, so an `audioconvert` is placed ahead of anything else.
    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Self::pcm_shape(upstream_caps)
            .map(|_| upstream_caps.clone())
            .ok_or(G2gError::CapsMismatch)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match Self::pcm_shape(input) {
            Some(sample_rate) => CapsSet::one(coded_caps(sample_rate)),
            None => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.sample_rate = Self::pcm_shape(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "ADPCM encoder",
            "Codec/Encoder/Audio",
            "Encodes mono S16LE PCM to IMA ADPCM (dvi layout)",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        BLOCK_ALIGN_PROPERTY
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "blockalign" => {
                self.block_align = parse_block_align(value)?;
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
                    if self.next_pts_ns.is_none() {
                        self.next_pts_ns = Some(frame.timing.pts_ns);
                    }
                    self.pending.extend_from_slice(slice);
                    self.drain(false, out).await?;
                }
                PipelinePacket::Eos => {
                    self.drain(true, out).await?;
                }
                PipelinePacket::Flush => {
                    self.pending.clear();
                    self.step_index = 0;
                    self.next_pts_ns = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::CapsChanged(c) => match &c {
                    // An upstream refine of the PCM rate: blocks are timed
                    // against it, and it is the coded rate too.
                    Caps::Audio {
                        format: AudioFormat::PcmS16Le,
                        sample_rate,
                        ..
                    } => self.sample_rate = *sample_rate,
                    // The runner's pre-fixed output caps: forward them on.
                    Caps::Audio {
                        format: AudioFormat::ImaAdpcm,
                        ..
                    } => {
                        out.push(PipelinePacket::CapsChanged(c.clone())).await?;
                        self.last_out = Some(c);
                    }
                    _ => return Err(G2gError::CapsMismatch),
                },
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for AdpcmEnc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: ADPCM_CHANNELS,
                sample_rate: ANY_SAMPLE_RATE,
            })),
            PadTemplate::source(CapsSet::one(coded_caps(ANY_SAMPLE_RATE))),
        ])
    }
}

/// Decodes IMA ADPCM blocks back to mono interleaved S16LE PCM.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::adpcm::AdpcmDec;
///
/// let decoder = AdpcmDec::new();
/// ```
#[derive(Debug)]
pub struct AdpcmDec {
    block_align: usize,
    sample_rate: u32,
    /// Coded bytes not yet forming a whole block.
    pending: Vec<u8>,
    /// PTS of the next block's PCM, anchored to the first input frame's PTS.
    next_pts_ns: Option<u64>,
    last_out: Option<Caps>,
    sequence: u64,
    configured: bool,
}

impl Default for AdpcmDec {
    fn default() -> Self {
        Self::new()
    }
}

impl AdpcmDec {
    pub fn new() -> Self {
        Self {
            block_align: DEFAULT_BLOCK_ALIGN,
            sample_rate: 0,
            pending: Vec::new(),
            next_pts_ns: None,
            last_out: None,
            sequence: 0,
            configured: false,
        }
    }

    /// Set the byte size of one consumed block (64..=8192, default 1024). It
    /// must match the block size the stream was written with: `Caps::Audio`
    /// carries no block-size field for a parser to hand over.
    pub fn with_block_align(mut self, bytes: usize) -> Self {
        self.block_align = bytes.clamp(MIN_BLOCK_ALIGN, MAX_BLOCK_ALIGN);
        self
    }

    /// The byte size of one consumed block.
    pub fn block_align(&self) -> usize {
        self.block_align
    }

    /// The coded layout `caps` carries, sentinels included.
    fn coded_shape(caps: &Caps) -> Option<u32> {
        match caps {
            Caps::Audio {
                format: AudioFormat::ImaAdpcm,
                channels,
                sample_rate,
            } if *channels == ADPCM_CHANNELS || *channels == ANY_CHANNELS => Some(*sample_rate),
            _ => None,
        }
    }

    fn output_caps(&self) -> Caps {
        Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: ADPCM_CHANNELS,
            sample_rate: self.sample_rate,
        }
    }

    /// Nanoseconds of audio one block carries.
    fn block_duration_ns(&self) -> u64 {
        samples_per_block(self.block_align) as u64 * 1_000_000_000 / self.sample_rate.max(1) as u64
    }

    /// Decode every whole block the pending bytes hold; a partial one stays
    /// buffered for the next input frame.
    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let pcm_block = samples_per_block(self.block_align) * PCM_SAMPLE_BYTES;
        while self.pending.len() >= self.block_align {
            // IMA ADPCM has no default rate the way G.711 has 8 kHz: without
            // one announced there is nothing to stamp the PCM with.
            if self.sample_rate == ANY_SAMPLE_RATE {
                return Err(G2gError::CapsMismatch);
            }
            let block: Vec<u8> = self.pending.drain(..self.block_align).collect();
            let mut pcm = alloc::vec![0u8; pcm_block];
            decode_block(&block, &mut pcm).ok_or(G2gError::CapsMismatch)?;
            let new_caps = self.output_caps();
            if self.last_out.as_ref() != Some(&new_caps) {
                out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                    .await?;
                self.last_out = Some(new_caps);
            }
            let pts_ns = self.next_pts_ns.unwrap_or(0);
            let duration_ns = self.block_duration_ns();
            self.next_pts_ns = Some(pts_ns + duration_ns);
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(pcm.into_boxed_slice())),
                FrameTiming {
                    pts_ns,
                    dts_ns: pts_ns,
                    duration_ns,
                    ..FrameTiming::default()
                },
                self.sequence,
            );
            self.sequence += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }
}

impl AsyncElement for AdpcmDec {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Self::coded_shape(upstream_caps)
            .map(|_| upstream_caps.clone())
            .ok_or(G2gError::CapsMismatch)
    }

    /// Mono S16LE at the coded rate. The wildcard alternative keeps a
    /// still-unknown rate negotiable; the real one arrives in the `CapsChanged`
    /// the parser emits ahead of the first block.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match Self::coded_shape(input) {
            Some(sample_rate) => {
                let pcm = |sample_rate| Caps::Audio {
                    format: AudioFormat::PcmS16Le,
                    channels: ADPCM_CHANNELS,
                    sample_rate,
                };
                let fixatable = if sample_rate == ANY_SAMPLE_RATE {
                    FIXATE_SAMPLE_RATE_HZ
                } else {
                    sample_rate
                };
                CapsSet::from_alternatives(alloc::vec![pcm(fixatable), pcm(ANY_SAMPLE_RATE)])
            }
            None => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.sample_rate = Self::coded_shape(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "ADPCM decoder",
            "Codec/Decoder/Audio",
            "Decodes IMA ADPCM (dvi layout) to mono S16LE PCM",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        BLOCK_ALIGN_PROPERTY
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "blockalign" => {
                self.block_align = parse_block_align(value)?;
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
                    if self.next_pts_ns.is_none() {
                        self.next_pts_ns = Some(frame.timing.pts_ns);
                    }
                    self.pending.extend_from_slice(slice);
                    self.drain(out).await?;
                }
                PipelinePacket::Flush => {
                    self.pending.clear();
                    self.next_pts_ns = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::CapsChanged(c) => match &c {
                    // An upstream refine of the coded rate (the `fmt ` chunk
                    // landing): decoded PCM inherits it.
                    Caps::Audio {
                        format: AudioFormat::ImaAdpcm,
                        sample_rate,
                        ..
                    } => self.sample_rate = *sample_rate,
                    // The runner's pre-fixed output caps: forward them on.
                    Caps::Audio {
                        format: AudioFormat::PcmS16Le,
                        ..
                    } => {
                        out.push(PipelinePacket::CapsChanged(c.clone())).await?;
                        self.last_out = Some(c);
                    }
                    _ => return Err(G2gError::CapsMismatch),
                },
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for AdpcmDec {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(coded_caps(ANY_SAMPLE_RATE))),
            PadTemplate::source(CapsSet::one(Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: ADPCM_CHANNELS,
                sample_rate: ANY_SAMPLE_RATE,
            })),
        ])
    }
}
