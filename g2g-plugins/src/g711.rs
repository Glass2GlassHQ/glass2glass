//! G.711 companding elements (M1073): `mulawenc` / `alawenc` turn
//! `Audio{PcmS16Le}` into one companded byte per sample, `mulawdec` /
//! `alawdec` expand it back. The sample math is [`g2g_mcu::g711`], the
//! heap-free MCU codec validated bit-exact against ffmpeg's `pcm_mulaw` /
//! `pcm_alaw`; these elements are the `alloc` host wrappers around it.
//!
//! Each law is a type ([`Mulaw`], [`Alaw`]) so the four elements are four
//! types with their own pad templates and auto-plug candidacy, over one
//! implementation.
//!
//! Companding is per-sample, so channel count and rate pass through and the
//! frame timing is carried over unchanged. The compressed side may arrive
//! with the unknown-channels / unknown-rate sentinels (an RTSP source that
//! has not read the SDP yet, the way `aacparse` documents); a decoder then
//! falls back to the telephony default RFC 3551 fixes for PCMU / PCMA, mono
//! at 8 kHz, and announces the concrete output caps in a `CapsChanged`
//! before its first frame.

use core::future::Future;
use core::marker::PhantomData;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, ANY_CHANNELS,
    ANY_SAMPLE_RATE,
};

use g2g_mcu::g711::Law;

/// The clock rate RFC 3551 fixes for the PCMU / PCMA payload types, and the
/// rate a decoder assumes when the coded caps carry the unknown-rate sentinel.
pub const G711_CLOCK_RATE_HZ: u32 = 8_000;

/// The channel count that goes with [`G711_CLOCK_RATE_HZ`] when the coded caps
/// carry the unknown-channels sentinel.
pub const G711_DEFAULT_CHANNELS: u8 = 1;

/// One companding law as a type: which [`Law`] the shared encoder / decoder
/// applies, and how each half describes itself to `gst-inspect`.
pub trait G711Law: core::fmt::Debug + Send + 'static {
    /// The law the sample math applies.
    const LAW: Law;
    /// Metadata of this law's encoder element.
    const ENCODER_METADATA: ElementMetadata;
    /// Metadata of this law's decoder element.
    const DECODER_METADATA: ElementMetadata;
}

/// Mu-law (RTP payload type 0, PCMU).
#[derive(Debug, Default, Clone, Copy)]
pub struct Mulaw;

/// A-law (RTP payload type 8, PCMA).
#[derive(Debug, Default, Clone, Copy)]
pub struct Alaw;

impl G711Law for Mulaw {
    const LAW: Law = Law::Mulaw;
    const ENCODER_METADATA: ElementMetadata = ElementMetadata::new(
        "Mu Law audio encoder",
        "Codec/Encoder/Audio",
        "Convert 16bit PCM to 8bit mu law",
        "g2g",
    );
    const DECODER_METADATA: ElementMetadata = ElementMetadata::new(
        "Mu Law audio decoder",
        "Codec/Decoder/Audio",
        "Convert 8bit mu law to 16bit PCM",
        "g2g",
    );
}

impl G711Law for Alaw {
    const LAW: Law = Law::Alaw;
    const ENCODER_METADATA: ElementMetadata = ElementMetadata::new(
        "A Law audio encoder",
        "Codec/Encoder/Audio",
        "Convert 16bit PCM to 8bit A law",
        "g2g",
    );
    const DECODER_METADATA: ElementMetadata = ElementMetadata::new(
        "A Law audio decoder",
        "Codec/Decoder/Audio",
        "Convert 8bit A law to 16bit PCM",
        "g2g",
    );
}

/// The concrete channel count and rate to decode at: the coded caps' own where
/// they carry them, else the telephony default (see [`G711_CLOCK_RATE_HZ`]).
fn resolve_layout(channels: u8, sample_rate: u32) -> (u8, u32) {
    (
        if channels == ANY_CHANNELS {
            G711_DEFAULT_CHANNELS
        } else {
            channels
        },
        if sample_rate == ANY_SAMPLE_RATE {
            G711_CLOCK_RATE_HZ
        } else {
            sample_rate
        },
    )
}

/// Companded caps for `L`, at the sentinel layout a demuxer starts from.
fn coded_template<L: G711Law>() -> Caps {
    Caps::Audio {
        format: L::LAW.format(),
        channels: ANY_CHANNELS,
        sample_rate: ANY_SAMPLE_RATE,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

/// Encodes interleaved S16LE PCM to one G.711 byte per sample.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::g711::MulawEnc;
///
/// let encoder = MulawEnc::new();
/// ```
#[derive(Debug)]
pub struct G711Enc<L: G711Law> {
    channels: u8,
    sample_rate: u32,
    /// The low byte of a sample split across two input frames.
    partial: Option<u8>,
    /// Last emitted output caps, to suppress an unchanged `CapsChanged`.
    last_out: Option<Caps>,
    configured: bool,
    law: PhantomData<L>,
}

/// `mulawenc`: S16LE PCM to mu-law.
pub type MulawEnc = G711Enc<Mulaw>;
/// `alawenc`: S16LE PCM to A-law.
pub type AlawEnc = G711Enc<Alaw>;
/// `mulawdec`: mu-law to S16LE PCM.
pub type MulawDec = G711Dec<Mulaw>;
/// `alawdec`: A-law to S16LE PCM.
pub type AlawDec = G711Dec<Alaw>;

impl<L: G711Law> Default for G711Enc<L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<L: G711Law> G711Enc<L> {
    pub fn new() -> Self {
        Self {
            channels: 0,
            sample_rate: 0,
            partial: None,
            last_out: None,
            configured: false,
            law: PhantomData,
        }
    }

    /// The PCM shape this encoder takes: interleaved S16LE, any layout.
    fn pcm_shape(caps: &Caps) -> Option<(u8, u32)> {
        match caps {
            Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels,
                sample_rate,
                ..
            } => Some((*channels, *sample_rate)),
            _ => None,
        }
    }

    fn output_caps(&self) -> Caps {
        Caps::Audio {
            format: L::LAW.format(),
            channels: self.channels,
            sample_rate: self.sample_rate,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }
}

impl<L: G711Law> AsyncElement for G711Enc<L> {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    /// S16LE only, so an `audioconvert` is placed ahead of any other PCM width.
    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Self::pcm_shape(upstream_caps)
            .map(|_| upstream_caps.clone())
            .ok_or(G2gError::CapsMismatch)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match Self::pcm_shape(input) {
            Some((channels, sample_rate)) => CapsSet::one(Caps::Audio {
                format: L::LAW.format(),
                channels,
                sample_rate,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            }),
            None => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (channels, sample_rate) =
            Self::pcm_shape(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        self.channels = channels;
        self.sample_rate = sample_rate;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        L::ENCODER_METADATA
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
                    let mut coded = Vec::with_capacity(slice.len().div_ceil(2) + 1);
                    let mut pcm = slice;
                    // A sample split across two input frames: its low byte was
                    // held back last time.
                    if let Some(low) = self.partial.take() {
                        match pcm.split_first() {
                            Some((high, rest)) => {
                                coded.push(L::LAW.encode(i16::from_le_bytes([low, *high])));
                                pcm = rest;
                            }
                            None => self.partial = Some(low),
                        }
                    }
                    let (samples, tail) = pcm.as_chunks::<2>();
                    coded.extend(
                        samples
                            .iter()
                            .map(|s| L::LAW.encode(i16::from_le_bytes(*s))),
                    );
                    self.partial = tail.first().copied();
                    if coded.is_empty() {
                        return Ok(());
                    }
                    let new_caps = self.output_caps();
                    if self.last_out.as_ref() != Some(&new_caps) {
                        out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                            .await?;
                        self.last_out = Some(new_caps);
                    }
                    let out_frame = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(coded.into_boxed_slice())),
                        frame.timing,
                        frame.sequence,
                    );
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                PipelinePacket::Flush => {
                    self.partial = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::CapsChanged(c) => match &c {
                    // An upstream refine of the PCM layout: it passes straight
                    // through companding, so it becomes the new output caps.
                    Caps::Audio {
                        format: AudioFormat::PcmS16Le,
                        channels,
                        sample_rate,
                        ..
                    } => {
                        self.channels = *channels;
                        self.sample_rate = *sample_rate;
                    }
                    // The runner's pre-fixed output caps: forward them on.
                    Caps::Audio { format, .. } if *format == L::LAW.format() => {
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

impl<L: G711Law> PadTemplates for G711Enc<L> {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: ANY_CHANNELS,
                sample_rate: ANY_SAMPLE_RATE,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            })),
            PadTemplate::source(CapsSet::one(coded_template::<L>())),
        ])
    }
}

/// Expands G.711 bytes back to interleaved S16LE PCM.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::g711::MulawDec;
///
/// let decoder = MulawDec::new();
/// ```
#[derive(Debug)]
pub struct G711Dec<L: G711Law> {
    channels: u8,
    sample_rate: u32,
    /// Last emitted output caps, to suppress an unchanged `CapsChanged`.
    last_out: Option<Caps>,
    configured: bool,
    law: PhantomData<L>,
}

impl<L: G711Law> Default for G711Dec<L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<L: G711Law> G711Dec<L> {
    pub fn new() -> Self {
        Self {
            channels: G711_DEFAULT_CHANNELS,
            sample_rate: G711_CLOCK_RATE_HZ,
            last_out: None,
            configured: false,
            law: PhantomData,
        }
    }

    /// The coded layout `caps` carries, sentinels included.
    fn coded_shape(caps: &Caps) -> Option<(u8, u32)> {
        match caps {
            Caps::Audio {
                format,
                channels,
                sample_rate,
                ..
            } if *format == L::LAW.format() => Some((*channels, *sample_rate)),
            _ => None,
        }
    }

    fn output_caps(&self) -> Caps {
        Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: self.channels,
            sample_rate: self.sample_rate,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }
}

impl<L: G711Law> AsyncElement for G711Dec<L> {
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

    /// The decoded layout is the coded one, with the telephony default standing
    /// in for a sentinel. The wildcard alternative keeps a still-unknown layout
    /// negotiable; the real values arrive in the `CapsChanged` ahead of the
    /// first frame.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match Self::coded_shape(input) {
            Some((channels, sample_rate)) => {
                let (channels, sample_rate) = resolve_layout(channels, sample_rate);
                CapsSet::from_alternatives(alloc::vec![
                    Caps::Audio {
                        format: AudioFormat::PcmS16Le,
                        channels,
                        sample_rate,
                        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
                    },
                    Caps::Audio {
                        format: AudioFormat::PcmS16Le,
                        channels: ANY_CHANNELS,
                        sample_rate: ANY_SAMPLE_RATE,
                        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
                    },
                ])
            }
            None => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (channels, sample_rate) =
            Self::coded_shape(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        (self.channels, self.sample_rate) = resolve_layout(channels, sample_rate);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        L::DECODER_METADATA
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
                    if slice.is_empty() {
                        return Ok(());
                    }
                    let mut pcm = Vec::with_capacity(slice.len() * 2);
                    for &code in slice {
                        pcm.extend_from_slice(&L::LAW.decode(code).to_le_bytes());
                    }
                    let new_caps = self.output_caps();
                    if self.last_out.as_ref() != Some(&new_caps) {
                        out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                            .await?;
                        self.last_out = Some(new_caps);
                    }
                    let out_frame = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(pcm.into_boxed_slice())),
                        frame.timing,
                        frame.sequence,
                    );
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => match &c {
                    // An upstream refine of the coded layout (the SDP / `fmt `
                    // chunk landing): decoded PCM inherits it.
                    Caps::Audio {
                        format,
                        channels,
                        sample_rate,
                        ..
                    } if *format == L::LAW.format() => {
                        (self.channels, self.sample_rate) = resolve_layout(*channels, *sample_rate);
                    }
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

impl<L: G711Law> PadTemplates for G711Dec<L> {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            // Both the sentinel layout a demuxer starts from and the concrete
            // telephony one an SDP announces: a compressed `sample_rate` is
            // matched for equality, so the RTSP rate needs its own alternative.
            PadTemplate::sink(CapsSet::from_alternatives(alloc::vec![
                coded_template::<L>(),
                Caps::Audio {
                    format: L::LAW.format(),
                    channels: ANY_CHANNELS,
                    sample_rate: G711_CLOCK_RATE_HZ,
                    channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
                },
            ])),
            PadTemplate::source(CapsSet::one(Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: G711_DEFAULT_CHANNELS,
                sample_rate: G711_CLOCK_RATE_HZ,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            })),
        ])
    }
}
