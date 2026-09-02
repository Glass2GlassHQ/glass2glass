//! Software PCM converter (M34), the audio analog of `VideoConvert`. Converts
//! interleaved PCM between sample formats (`PcmU8` / `PcmS16Le` / `PcmS24Le` /
//! `PcmS32Le` / `PcmF32Le`, all via an f32 intermediate) and
//! between channel counts (mono <-> multi-channel) at the same sample rate, so
//! audio chains compose across format boundaries: `WasapiSrc (F32, 2ch) ->
//! AudioConvert -> WavSink (S16)`, or feeding an encoder that wants a specific
//! layout.
//!
//! Channel conversion handles any count to any count: identity, mono <-> stereo
//! (replicate / average), and position-aware matrix mixing for multichannel
//! (either side > 2 channels). Speaker positions come from the per-count
//! default layout convention ([`ChannelLayout::default_for`], what the ffmpeg
//! decode path emits): downmix applies the ITU BS.775-style coefficients
//! (surrounds and center at 1/sqrt(2), back center at 0.5 into each front, LFE
//! dropped, matrix normalized against clipping, matching ffmpeg's default
//! rematrix), and upmix places each input at its own speaker, leaving the rest
//! silent. Counts past the layout table (> 8) fall back to a layout-agnostic
//! round-robin fold/replicate so no channel is silently dropped. Sample rate is
//! preserved (no resampler). CPU-only and `no_std`: this element lives in the
//! crate baseline.
//!
//! A conversion that drops resolution (F32 or a wider integer down to, say,
//! S16) adds dither before rounding (tpdf by default, as in gst) and can feed
//! the rounding error back through a noise-shaping filter, per the `dithering`
//! / `dithering-threshold` / `noise-shaping` properties. Neither runs on a
//! lossless conversion, so a passthrough or an equal-depth format change stays
//! byte for byte identical.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, AudioShape, Caps, CapsConstraint, CapsSet, CapsTransform,
    ChannelLayout, ChannelPosition, ConfigureOutcome, ElementMetadata, FieldTransform, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, ANY_CHANNELS,
};

use crate::random::{next_random, XORSHIFT_BASE_STATE};

/// The PCM sample formats `AudioConvert` / `AudioResample` read and write: every
/// one g2g has, since the converter is what makes any of them reachable.
pub(crate) fn pcm_formats() -> [AudioFormat; 5] {
    g2g_core::pcm_formats()
}

/// # Example
///
/// ```no_run
/// use g2g_core::AudioFormat;
/// use g2g_plugins::audioconvert::AudioConvert;
///
/// let element = AudioConvert::new(AudioFormat::PcmS16Le, 2);
/// ```
#[derive(Debug)]
pub struct AudioConvert {
    /// Target sample format, or `None` for caps-driven (take it from a downstream
    /// capsfilter, else passthrough the input format).
    target_format: Option<AudioFormat>,
    /// Target channel count, or `None` for caps-driven.
    target_channels: Option<u8>,
    /// Output format/channels resolved from the negotiated output caps (a
    /// downstream capsfilter), set by `configure_output`. Used when the matching
    /// target is caps-driven; `None` until then.
    resolved: Option<(AudioFormat, u8)>,
    /// Input format/channels/rate of the configured stream, updated by a
    /// mid-stream `CapsChanged`.
    input: Option<(AudioFormat, u8, u32)>,
    /// Speaker layout the input caps declared, `UNSPECIFIED` when they did not.
    input_layout: ChannelLayout,
    /// Speaker layout the negotiated output caps pinned (a downstream
    /// `channel-mask`), `UNSPECIFIED` when they left it open.
    output_layout: ChannelLayout,
    /// Dither + noise-shaping settings and their per-channel state, applied
    /// only where the output format drops resolution.
    quantizer: Quantizer,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl AudioConvert {
    pub fn new(target_format: AudioFormat, target_channels: u8) -> Self {
        assert!(target_channels > 0, "target channels must be non-zero");
        assert!(
            pcm_formats().contains(&target_format),
            "AudioConvert is a raw-PCM converter; target must be a PCM format"
        );
        Self {
            target_format: Some(target_format),
            target_channels: Some(target_channels),
            ..Self::auto()
        }
    }

    /// Caps-driven: take the output format + channel count from the negotiated
    /// caps (a downstream capsfilter), the gst idiom. With no downstream
    /// constraint it passes the input through unchanged.
    pub fn auto() -> Self {
        Self {
            target_format: None,
            target_channels: None,
            resolved: None,
            input: None,
            input_layout: ChannelLayout::UNSPECIFIED,
            output_layout: ChannelLayout::UNSPECIFIED,
            quantizer: Quantizer {
                dither: DEFAULT_DITHER,
                threshold: DITHERING_THRESHOLD,
                ..Quantizer::default()
            },
            configured: false,
            last_caps: None,
            emitted: 0,
        }
    }

    /// Effective output format: the property when set, else the caps-resolved
    /// format (auto), else the input format (passthrough).
    fn out_format(&self, in_format: AudioFormat) -> AudioFormat {
        self.target_format
            .or(self.resolved.map(|(f, _)| f))
            .unwrap_or(in_format)
    }

    /// Effective output channel count: the property when set, else the
    /// caps-resolved count (auto), else the input count (passthrough).
    fn out_channels(&self, in_channels: u8) -> u8 {
        self.target_channels
            .or(self.resolved.map(|(_, c)| c))
            .unwrap_or(in_channels)
    }

    pub fn target_format(&self) -> AudioFormat {
        self.out_format(AudioFormat::PcmS16Le)
    }

    pub fn target_channels(&self) -> u8 {
        self.out_channels(2)
    }

    /// Speaker layout to put on the output caps: the one a downstream
    /// `channel-mask` pinned, else the input's when the channel count is
    /// unchanged (nothing was remixed), else unspecified, since a remix lands on
    /// the target count's conventional layout, which is what an unspecified
    /// layout already means.
    fn out_layout(&self, in_channels: u8, out_channels: u8) -> ChannelLayout {
        if !self.output_layout.is_unspecified() {
            return self.output_layout;
        }
        if in_channels == out_channels {
            return self.input_layout;
        }
        ChannelLayout::UNSPECIFIED
    }

    /// Validate a PCM caps as a convertible input, returning its
    /// format/channels/rate. Any concrete input channel count converts to the
    /// target (identity / fan-out / downmix / layout-agnostic remap);
    /// `ANY_CHANNELS` (0) is the negotiation placeholder, accepted here with the
    /// real count arriving via a `CapsChanged` before the first frame.
    fn accept_input(&self, caps: &Caps) -> Result<(AudioFormat, u8, u32, ChannelLayout), G2gError> {
        let Caps::Audio {
            format,
            channels,
            sample_rate,
            channel_layout,
        } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !pcm_formats().contains(format) {
            return Err(G2gError::CapsMismatch);
        }
        Ok((*format, *channels, *sample_rate, *channel_layout))
    }
}

/// Bytes per sample. `PcmS24Le` is 3-byte packed (no 32-bit container), the
/// ST 2110-30 / AES67 L24 layout.
pub(crate) fn sample_bytes(format: AudioFormat) -> usize {
    match format {
        AudioFormat::PcmU8 => 1,
        AudioFormat::PcmS16Le => 2,
        AudioFormat::PcmS24Le => 3,
        AudioFormat::PcmF32Le | AudioFormat::PcmS32Le => 4,
        // not reachable: only the PCM formats pass negotiation.
        _ => 0,
    }
}

const NS_PER_SECOND: u64 = 1_000_000_000;

/// The sample format an element declares when the solver needs its caps before
/// the pads negotiate (a fan-in's merged output, a fan-out's ports). Matches
/// `audiotestsrc` and `audiomixer`'s nominal output.
pub(crate) const DEFAULT_PCM_FORMAT: AudioFormat = AudioFormat::PcmS16Le;

/// The same value as declared text, for `gst-inspect`.
pub(crate) const DEFAULT_PCM_FORMAT_TEXT: &str = "S16LE";

/// The sample rate that goes with [`DEFAULT_PCM_FORMAT`].
pub(crate) const DEFAULT_PCM_RATE: u32 = 48_000;

/// The same value as declared text, for `gst-inspect`.
pub(crate) const DEFAULT_PCM_RATE_TEXT: &str = "48000";

/// The `format` / `rate` properties an element exposes when it declares its PCM
/// shape rather than learning it from the pads ([`DEFAULT_PCM_FORMAT`]).
pub(crate) static PCM_SHAPE_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "format",
        PropKind::Str,
        "sample format on every pad: S16LE | F32LE | S24LE | S32LE | U8",
    )
    .with_default(DEFAULT_PCM_FORMAT_TEXT),
    PropertySpec::new("rate", PropKind::Uint, "sample rate on every pad")
        .with_default(DEFAULT_PCM_RATE_TEXT),
];

/// Apply one [`PCM_SHAPE_PROPS`] property to the shape an element declares.
pub(crate) fn set_pcm_shape_property(
    format: &mut AudioFormat,
    rate: &mut u32,
    name: &str,
    value: PropValue,
) -> Result<(), PropError> {
    match name {
        "format" => {
            let text = value.as_str().ok_or(PropError::Type)?;
            *format = audio_format_from_str(text).ok_or(PropError::Value)?;
        }
        "rate" => {
            let value = value.as_uint().ok_or(PropError::Type)? as u32;
            if value == 0 {
                return Err(PropError::Value);
            }
            *rate = value;
        }
        _ => return Err(PropError::Unknown),
    }
    Ok(())
}

/// Read one [`PCM_SHAPE_PROPS`] property back.
pub(crate) fn get_pcm_shape_property(
    format: AudioFormat,
    rate: u32,
    name: &str,
) -> Option<PropValue> {
    match name {
        "format" => Some(PropValue::Str(audio_format_to_str(format).into())),
        "rate" => Some(PropValue::Uint(rate as u64)),
        _ => None,
    }
}

/// Byte a silent sample of this format is made of: `PcmU8` is offset-binary
/// (silence at the 0x80 midpoint), the signed and float formats are all-zero.
pub(crate) fn silence_byte(format: AudioFormat) -> u8 {
    match format {
        AudioFormat::PcmU8 => 0x80,
        _ => 0,
    }
}

/// Nanoseconds `samples` sample frames occupy at `rate`.
pub(crate) fn samples_to_ns(samples: u64, rate: u32) -> u64 {
    if rate == 0 {
        return 0;
    }
    (samples as u128 * NS_PER_SECOND as u128 / rate as u128) as u64
}

/// Sample frames spanning `ns` at `rate`, rounded to nearest so half a sample
/// of jitter does not leave a permanent one-sample offset.
pub(crate) fn ns_to_samples(ns: u64, rate: u32) -> u64 {
    if rate == 0 {
        return 0;
    }
    let scale = NS_PER_SECOND as u128;
    ((ns as u128 * rate as u128 + scale / 2) / scale) as u64
}

impl AsyncElement for AudioConvert {
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
        self.accept_input(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    /// Native `DerivedFields`: a supported PCM input maps to the target format +
    /// channel count at the same sample rate (rate is the one passthrough field).
    /// A fixed target emits that single output; a caps-driven (`auto`) target
    /// advertises the passthrough as the preferred alternative plus the retarget
    /// options (every other PCM format, and an `ANY_CHANNELS` wildcard) so a
    /// downstream capsfilter pins the real format / channel count.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        // Candidate output formats: the fixed target, or (auto) the input format
        // first (passthrough) then every PCM format, the duplicate collapsing into
        // the passthrough.
        let formats: Vec<FieldTransform<AudioFormat>> = match self.target_format {
            Some(f) => alloc::vec![FieldTransform::Fixed(f)],
            None => {
                let mut v = alloc::vec![FieldTransform::Identity];
                v.extend(pcm_formats().iter().copied().map(FieldTransform::Fixed));
                v
            }
        };
        // Candidate channel counts: the fixed target, or (auto) the input count
        // (passthrough) then the `ANY_CHANNELS` wildcard. A `0` (ANY_CHANNELS)
        // input is the decoder's pre-decode placeholder, where the two coincide:
        // only the wildcard is advertised, so a downstream capsfilter pins it
        // (else it fixates to stereo) and the real count flows in a runtime
        // `CapsChanged`.
        let chans: Vec<FieldTransform<u8>> = match self.target_channels {
            Some(c) => alloc::vec![FieldTransform::Fixed(c)],
            None => alloc::vec![
                FieldTransform::Identity,
                FieldTransform::Fixed(ANY_CHANNELS)
            ],
        };
        let mut shapes = Vec::with_capacity(formats.len() * chans.len());
        for f in &formats {
            for c in &chans {
                shapes.push(
                    AudioShape::PASSTHROUGH
                        .with_format(f.clone())
                        .with_channels(c.clone()),
                );
            }
        }
        CapsConstraint::DerivedFields(CapsTransform::Audio {
            accept: pcm_formats().to_vec(),
            produce: Vec::new(),
            shapes,
        })
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (format, channels, rate, channel_layout) = self.accept_input(absolute_caps)?;
        self.input = Some((format, channels, rate));
        self.input_layout = channel_layout;
        self.quantizer.reset();
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    /// Caps-driven: take the output format + channel count from the negotiated
    /// output caps when a target is unset (auto). Already fixated, so concrete.
    fn configure_output(&mut self, output_caps: &Caps) -> Result<(), G2gError> {
        let Caps::Audio {
            format,
            channels,
            channel_layout,
            ..
        } = output_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !pcm_formats().contains(format) || *channels == ANY_CHANNELS {
            return Err(G2gError::CapsMismatch);
        }
        self.resolved = Some((*format, *channels));
        self.output_layout = *channel_layout;
        Ok(())
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
                    let (in_format, in_channels, rate) =
                        self.input.ok_or(G2gError::NotConfigured)?;
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let out_format = self.out_format(in_format);
                    let out_channels = self.out_channels(in_channels);
                    let out_layout = self.out_layout(in_channels, out_channels);
                    let converted = convert_pcm(
                        slice,
                        PcmStream {
                            format: in_format,
                            channels: in_channels,
                            layout: self.input_layout,
                        },
                        PcmStream {
                            format: out_format,
                            channels: out_channels,
                            layout: out_layout,
                        },
                        &mut self.quantizer,
                    )?;

                    let new_caps = Caps::Audio {
                        format: out_format,
                        channels: out_channels,
                        sample_rate: rate,
                        channel_layout: out_layout,
                    };
                    if self.last_caps.as_ref() != Some(&new_caps) {
                        out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                            .await?;
                        self.last_caps = Some(new_caps);
                    }
                    let out_frame = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(converted)),
                        timing: frame.timing,
                        sequence: self.emitted,
                        meta: Default::default(),
                    };
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    // The runner's transform arm calls `configure_pipeline` (input)
                    // then `configure_output` (output) immediately before pushing
                    // this packet, whose caps `c` is the arm's pre-fixed forward
                    // *output*, not a new input. Forward it and record `last_caps`
                    // to suppress the duplicate emit from the data path. Do NOT
                    // `accept_input` here: `c` is our output, and adopting it as
                    // the input corrupts the next frame (the stacked-convert bug;
                    // see videoconvert.rs). The real input is set by
                    // `configure_pipeline`.
                    out.push(PipelinePacket::CapsChanged(c.clone())).await?;
                    self.last_caps = Some(c);
                }
                PipelinePacket::Flush => {
                    self.quantizer.reset();
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                // Segment is control: forward unchanged.
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                // the runner forwards Eos; the transform does not re-emit it.
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        AUDIOCONVERT_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Audio format converter",
            "Filter/Converter/Audio",
            "Converts between raw audio sample formats and channel layouts",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "format" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.target_format = Some(audio_format_from_str(s).ok_or(PropError::Value)?);
                Ok(())
            }
            "channels" => {
                let c = value.as_uint().ok_or(PropError::Type)? as u8;
                if c == 0 {
                    return Err(PropError::Value);
                }
                self.target_channels = Some(c);
                Ok(())
            }
            "dithering" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.quantizer.dither = dither_from_nick(s).ok_or(PropError::Value)?;
                self.quantizer.reset();
                Ok(())
            }
            "dithering-threshold" => {
                let bits = value.as_uint().ok_or(PropError::Type)?;
                if bits > MAX_DITHERING_THRESHOLD {
                    return Err(PropError::Value);
                }
                self.quantizer.threshold = bits as u8;
                Ok(())
            }
            "noise-shaping" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.quantizer.shaping = noise_shaping_from_nick(s).ok_or(PropError::Value)?;
                // the filter history is method-sized; drop it so the next
                // buffer rebuilds it.
                self.quantizer.reset();
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "format" => Some(PropValue::Str(
                audio_format_to_str(self.target_format()).into(),
            )),
            "channels" => Some(PropValue::Uint(self.target_channels() as u64)),
            "dithering" => Some(PropValue::Str(dither_nick(self.quantizer.dither).into())),
            "dithering-threshold" => Some(PropValue::Uint(u64::from(self.quantizer.threshold))),
            "noise-shaping" => Some(PropValue::Str(
                noise_shaping_nick(self.quantizer.shaping).into(),
            )),
            _ => None,
        }
    }
}

/// `AudioConvert`'s settable properties (M107).
static AUDIOCONVERT_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "format",
        PropKind::Str,
        "output sample format: S16LE | F32LE | S24LE | S32LE | U8",
    ),
    PropertySpec::new("channels", PropKind::Uint, "output channel count"),
    PropertySpec::new(
        "dithering",
        PropKind::Str,
        "dither added before rounding, on a bit-depth reduction",
    )
    .with_enum_values(DITHER_NICKS)
    .with_default(dither_nick(DEFAULT_DITHER)),
    PropertySpec::new(
        "dithering-threshold",
        PropKind::Uint,
        "output bit depth at or below which to dither",
    )
    .with_range(MIN_DITHERING_THRESHOLD_TEXT, MAX_DITHERING_THRESHOLD_TEXT)
    .with_default(DITHERING_THRESHOLD_TEXT),
    PropertySpec::new(
        "noise-shaping",
        PropKind::Str,
        "error-feedback filter on the rounding error, on a bit-depth reduction",
    )
    .with_enum_values(NOISE_SHAPING_NICKS)
    .with_default("none"),
];

/// Parse an audio-format property string to an [`AudioFormat`]. Shared with the
/// `gst-launch` DSL. GStreamer names raw sample formats uppercase (S16LE,
/// F32LE); accept any case and the historical lowercase spellings as aliases.
pub(crate) fn audio_format_from_str(s: &str) -> Option<AudioFormat> {
    // Only the PCM formats are valid AudioConvert targets; AAC/OPUS are encoder
    // outputs, not something a raw-sample converter can produce.
    g2g_core::pcm_from_gst_format(s)
}

/// The canonical (GStreamer) property string for an [`AudioFormat`].
pub(crate) fn audio_format_to_str(f: AudioFormat) -> &'static str {
    if let Some(name) = g2g_core::pcm_gst_format(f) {
        return name;
    }
    match f {
        AudioFormat::Aac => "AAC",
        AudioFormat::Opus => "OPUS",
        // A format added since: no canonical string here, fail loud.
        _ => unreachable!("unnamed AudioFormat: {f:?}"),
    }
}

impl PadTemplates for AudioConvert {
    /// Static superset: PCM in, PCM out. `Caps::Audio` has no open dims, so the
    /// templates pin the common stereo/48 kHz shape per format.
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = |format| Caps::Audio {
            format,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        let set = CapsSet::from_alternatives(pcm_formats().map(pcm).to_vec());
        Vec::from([PadTemplate::sink(set.clone()), PadTemplate::source(set)])
    }
}

/// The interleaved PCM on one end of a conversion: its sample format, how many
/// channels it carries, and which speakers those channels feed
/// ([`ChannelLayout::UNSPECIFIED`] when the caps did not say).
#[derive(Clone, Copy, Debug)]
struct PcmStream {
    format: AudioFormat,
    channels: u8,
    layout: ChannelLayout,
}

/// Read interleaved PCM and re-emit it in the target format/channel count.
/// Samples pass through an f32 intermediate; channel mapping is identity,
/// mono fan-out, or downmix-to-mono average. `quantizer` carries the dither and
/// noise-shaping state, and only engages on a conversion that drops resolution.
fn convert_pcm(
    src: &[u8],
    input: PcmStream,
    output: PcmStream,
    quantizer: &mut Quantizer,
) -> Result<Box<[u8]>, G2gError> {
    let (in_format, out_format) = (input.format, output.format);
    let in_bytes = sample_bytes(in_format);
    let out_bytes = sample_bytes(out_format);
    let in_ch = input.channels as usize;
    let out_ch = output.channels as usize;
    let in_frame = in_bytes * in_ch;
    if in_frame == 0 || !src.len().is_multiple_of(in_frame) {
        return Err(G2gError::CapsMismatch);
    }
    let frames = src.len() / in_frame;

    // Position-aware mixing whenever either side is true multichannel and both
    // sides have a usable layout; None keeps the layout-agnostic paths.
    let matrix = if in_ch != out_ch && in_ch.max(out_ch) > 2 {
        match (
            mixing_layout(input.layout, in_ch),
            mixing_layout(output.layout, out_ch),
        ) {
            (Some(i), Some(o)) => Some(mix_matrix(i, o)),
            _ => None,
        }
    } else {
        None
    };

    let plan = quantizer.plan(in_format, out_format);
    quantizer.prepare(out_ch, plan);

    let mut dst = Vec::with_capacity(frames * out_bytes * out_ch);
    let mut in_samples = alloc::vec![0f32; in_ch];
    for f in 0..frames {
        let base = f * in_frame;
        for (c, slot) in in_samples.iter_mut().enumerate() {
            *slot = read_sample(&src[base + c * in_bytes..], in_format);
        }
        for oc in 0..out_ch {
            let v = match &matrix {
                Some(m) => m[oc * in_ch..][..in_ch]
                    .iter()
                    .zip(&in_samples)
                    .map(|(coef, s)| coef * s)
                    .sum(),
                None => map_channel(&in_samples, oc, out_ch),
            };
            quantizer.write(&mut dst, v, out_format, oc, plan);
        }
    }
    Ok(dst.into_boxed_slice())
}

/// The `dithering` choices, matching gst audioconvert's `GstAudioDitherMethod`
/// nicks (`gst-inspect-1.0 audioconvert`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum DitherMethod {
    #[default]
    None,
    /// One uniform draw, rectangular over +/- 1 LSB.
    Rpdf,
    /// Two uniform half-LSB draws summed, triangular over +/- 1 LSB.
    Tpdf,
    /// A triangular draw differenced against the channel's previous one, which
    /// tilts the dither noise itself toward high frequency.
    TpdfHf,
}

/// The `noise-shaping` choices, matching gst audioconvert's
/// `GstAudioNoiseShapingMethod` nicks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum NoiseShapingMethod {
    #[default]
    None,
    ErrorFeedback,
    Simple,
    Medium,
    High,
}

const DITHER_NICKS: &str = "none | rpdf | tpdf | tpdf-hf";
const NOISE_SHAPING_NICKS: &str = "none | error-feedback | simple | medium | high";

/// gst's `dithering` default.
const DEFAULT_DITHER: DitherMethod = DitherMethod::Tpdf;

/// gst's `dithering-threshold` default: dither only where the output resolution
/// is this many bits or fewer.
const DITHERING_THRESHOLD: u8 = 20;
const DITHERING_THRESHOLD_TEXT: &str = "20";
/// gst's declared bound on `dithering-threshold` (Unsigned Integer, 0 - 32).
const MAX_DITHERING_THRESHOLD: u64 = 32;
const MIN_DITHERING_THRESHOLD_TEXT: &str = "0";
const MAX_DITHERING_THRESHOLD_TEXT: &str = "32";

/// The name a [`DitherMethod`] takes in a launch line.
const fn dither_nick(method: DitherMethod) -> &'static str {
    match method {
        DitherMethod::None => "none",
        DitherMethod::Rpdf => "rpdf",
        DitherMethod::Tpdf => "tpdf",
        DitherMethod::TpdfHf => "tpdf-hf",
    }
}

fn dither_from_nick(nick: &str) -> Option<DitherMethod> {
    match nick {
        "none" => Some(DitherMethod::None),
        "rpdf" => Some(DitherMethod::Rpdf),
        "tpdf" => Some(DitherMethod::Tpdf),
        "tpdf-hf" => Some(DitherMethod::TpdfHf),
        _ => None,
    }
}

fn noise_shaping_nick(method: NoiseShapingMethod) -> &'static str {
    match method {
        NoiseShapingMethod::None => "none",
        NoiseShapingMethod::ErrorFeedback => "error-feedback",
        NoiseShapingMethod::Simple => "simple",
        NoiseShapingMethod::Medium => "medium",
        NoiseShapingMethod::High => "high",
    }
}

fn noise_shaping_from_nick(nick: &str) -> Option<NoiseShapingMethod> {
    match nick {
        "none" => Some(NoiseShapingMethod::None),
        "error-feedback" => Some(NoiseShapingMethod::ErrorFeedback),
        "simple" => Some(NoiseShapingMethod::Simple),
        "medium" => Some(NoiseShapingMethod::Medium),
        "high" => Some(NoiseShapingMethod::High),
        _ => None,
    }
}

/// Error-feedback weights per shaping method, oldest error first: the last
/// entry weights the error made on the previous sample. Taken from GStreamer's
/// `audio-quantize.c` (`ns_simple_coeffs` / `ns_medium_coeffs` /
/// `ns_high_coeffs`); `error-feedback` is the plain one-tap case.
fn shaping_coefficients(method: NoiseShapingMethod) -> &'static [f32] {
    match method {
        NoiseShapingMethod::None => &[],
        NoiseShapingMethod::ErrorFeedback => &[1.0],
        NoiseShapingMethod::Simple => &[-0.5, 1.0],
        NoiseShapingMethod::Medium => &[0.6149, -1.590, 1.959, -2.165, 2.033],
        NoiseShapingMethod::High => &[
            -0.340122, 0.876066, -1.72008, 2.61339, -3.31399, 3.27918, -2.92975, 2.08484,
        ],
    }
}

/// What the quantizer does over one conversion, decided once from the format
/// pair: `dither` is `None` where the output loses nothing or is wider than
/// `dithering-threshold`, `shaping` is `None` where the output loses nothing.
#[derive(Debug, Clone, Copy)]
struct QuantizePlan {
    dither: DitherMethod,
    shaping: NoiseShapingMethod,
}

impl QuantizePlan {
    /// Neither knob applies, so the samples take the plain rounding path and a
    /// lossless conversion stays byte for byte identical.
    fn is_plain(&self) -> bool {
        self.dither == DitherMethod::None && self.shaping == NoiseShapingMethod::None
    }
}

/// Quantizes the f32 intermediate to an integer output format, adding dither
/// and feeding the rounding error back through a shaping filter. The state is
/// per channel and lives across buffers, so the shaping filter stays continuous
/// over the stream.
#[derive(Debug, Default)]
struct Quantizer {
    dither: DitherMethod,
    shaping: NoiseShapingMethod,
    /// Dither only where the output resolution is at most this many bits.
    threshold: u8,
    random_state: u32,
    /// Per channel, the last `shaping_coefficients().len()` rounding errors in
    /// output LSBs, oldest first.
    errors: Vec<Vec<f32>>,
    /// Per channel, the previous dither draw, for [`DitherMethod::TpdfHf`].
    previous_draw: Vec<f32>,
}

impl Quantizer {
    /// Drop the filter history and restart the generator, so a configure or a
    /// flush replays the same noise from the top.
    fn reset(&mut self) {
        self.random_state = XORSHIFT_BASE_STATE;
        self.errors.clear();
        self.previous_draw.clear();
    }

    /// Size the per-channel state for this conversion, keeping what is already
    /// there when the shape has not moved.
    fn prepare(&mut self, channels: usize, plan: QuantizePlan) {
        let taps = shaping_coefficients(plan.shaping).len();
        if self.errors.len() == channels && self.errors.first().is_none_or(|e| e.len() == taps) {
            return;
        }
        self.random_state = XORSHIFT_BASE_STATE;
        self.errors = alloc::vec![alloc::vec![0f32; taps]; channels];
        self.previous_draw = alloc::vec![0f32; channels];
    }

    /// The dither and shaping this format pair calls for. Both need the output
    /// to actually drop resolution: an equal or wider integer output, or a
    /// float one, loses nothing to shape.
    fn plan(&self, in_format: AudioFormat, out_format: AudioFormat) -> QuantizePlan {
        let plain = QuantizePlan {
            dither: DitherMethod::None,
            shaping: NoiseShapingMethod::None,
        };
        let Some(out_bits) = quantization_bits(out_format) else {
            return plain;
        };
        let reduces = match quantization_bits(in_format) {
            Some(in_bits) => in_bits > out_bits,
            // float input carries more resolution than any integer output.
            None => true,
        };
        if !reduces {
            return plain;
        }
        QuantizePlan {
            dither: if out_bits <= self.threshold {
                self.dither
            } else {
                DitherMethod::None
            },
            shaping: self.shaping,
        }
    }

    /// One dither draw for `channel`, in output LSBs.
    fn draw(&mut self, method: DitherMethod, channel: usize) -> f32 {
        // gst's rpdf spans a whole quantizer step each way; its tpdf sums two
        // half-step draws, the textbook triangular dither.
        const RPDF_HALF_WIDTH: f32 = 1.0;
        const TPDF_HALF_WIDTH: f32 = 0.5;
        match method {
            DitherMethod::None => 0.0,
            DitherMethod::Rpdf => self.uniform(RPDF_HALF_WIDTH),
            DitherMethod::Tpdf => self.uniform(TPDF_HALF_WIDTH) + self.uniform(TPDF_HALF_WIDTH),
            DitherMethod::TpdfHf => {
                let drawn = self.uniform(TPDF_HALF_WIDTH);
                let shaped = drawn - self.previous_draw[channel];
                self.previous_draw[channel] = drawn;
                shaped
            }
        }
    }

    /// A uniform draw over `[-half_width, half_width)`.
    fn uniform(&mut self, half_width: f32) -> f32 {
        let unit = next_random(&mut self.random_state) as f32 / (u32::MAX as f32 + 1.0);
        (unit - 0.5) * 2.0 * half_width
    }

    /// Encode one sample of `channel` under `plan`.
    fn write(
        &mut self,
        dst: &mut Vec<u8>,
        v: f32,
        format: AudioFormat,
        channel: usize,
        plan: QuantizePlan,
    ) {
        let scale = match (plan.is_plain(), full_scale(format)) {
            (false, Some(scale)) => scale,
            _ => return write_sample(dst, v, format),
        };
        let coefficients = shaping_coefficients(plan.shaping);
        // Subtracting the filtered past errors puts the quantization noise
        // through `1 - sum(coefficient * z^-k)`, which lifts it out of the band
        // the ear hears best.
        let feedback: f32 = coefficients
            .iter()
            .zip(&self.errors[channel])
            .map(|(c, e)| c * e)
            .sum();
        let target = v.clamp(-1.0, 1.0) * scale - feedback;
        let dithered = target + self.draw(plan.dither, channel);
        let quantized = round_to_integer(dithered);
        if !coefficients.is_empty() {
            let history = &mut self.errors[channel];
            history.remove(0);
            history.push(quantized - dithered);
        }
        emit_quantized(dst, quantized, format);
    }
}

/// ITU BS.775-style surround coefficient: center and surrounds mix into the
/// fronts at 1/sqrt(2).
const SURROUND_MIX: f32 = core::f32::consts::FRAC_1_SQRT_2;

/// Build the position-aware mixing matrix (flattened `out_ch` rows of `in_ch`
/// coefficients) between two speaker layouts.
///
/// Each input position routes to the output: itself at 1.0 when present; a
/// side <-> back counterpart at 1.0; otherwise folded frontward (center and
/// surrounds at 1/sqrt(2), back center at 0.5 into each front, surrounds at 0.5
/// into a mono center). LFE is dropped unless the output has one. The matrix is
/// then normalized by its largest output-row sum so a full-scale input cannot
/// clip. Coefficients match ffmpeg's default rematrix (verified against
/// `swr_build_matrix` output for 5.1 / 6.1 / 7.1 to stereo and mono).
fn mix_matrix(in_layout: ChannelLayout, out_layout: ChannelLayout) -> Vec<f32> {
    use ChannelPosition::*;
    let in_ch = in_layout.channels() as usize;
    let out_ch = out_layout.channels() as usize;
    let mut m = alloc::vec![0f32; in_ch * out_ch];

    // Add `gain` of input index `ii` into output position `pos`, if present.
    let add = |m: &mut Vec<f32>, pos: ChannelPosition, ii: usize, gain: f32| -> bool {
        match out_layout.index_of(pos) {
            Some(oi) => {
                m[oi * in_ch + ii] += gain;
                true
            }
            None => false,
        }
    };

    for (ii, pos) in in_layout.positions().enumerate() {
        if add(&mut m, pos, ii, 1.0) {
            continue;
        }
        match pos {
            Fl | Flc => {
                let _ = add(&mut m, Fc, ii, SURROUND_MIX);
            }
            Fr | Frc => {
                let _ = add(&mut m, Fc, ii, SURROUND_MIX);
            }
            Fc => {
                let _ = add(&mut m, Fl, ii, SURROUND_MIX);
                let _ = add(&mut m, Fr, ii, SURROUND_MIX);
            }
            Lfe => {} // dropped: BS.775 / ffmpeg default LFE mix level 0
            Bl | Sl => {
                let side_back = if pos == Bl { Sl } else { Bl };
                if !add(&mut m, side_back, ii, 1.0) && !add(&mut m, Fl, ii, SURROUND_MIX) {
                    let _ = add(&mut m, Fc, ii, SURROUND_MIX * SURROUND_MIX);
                }
            }
            Br | Sr => {
                let side_back = if pos == Br { Sr } else { Br };
                if !add(&mut m, side_back, ii, 1.0) && !add(&mut m, Fr, ii, SURROUND_MIX) {
                    let _ = add(&mut m, Fc, ii, SURROUND_MIX * SURROUND_MIX);
                }
            }
            Bc => {
                if !(add(&mut m, Bl, ii, SURROUND_MIX) | add(&mut m, Br, ii, SURROUND_MIX))
                    && !(add(&mut m, Sl, ii, SURROUND_MIX) | add(&mut m, Sr, ii, SURROUND_MIX))
                    && !(add(&mut m, Fl, ii, 0.5) | add(&mut m, Fr, ii, 0.5))
                {
                    let _ = add(&mut m, Fc, ii, SURROUND_MIX);
                }
            }
        }
    }

    // Normalize against clipping: scale so the loudest output row sums to 1.
    let max_sum = (0..out_ch)
        .map(|oc| m[oc * in_ch..][..in_ch].iter().map(|c| c.abs()).sum())
        .fold(1.0f32, f32::max);
    if max_sum > 1.0 {
        for coef in &mut m {
            *coef /= max_sum;
        }
    }
    m
}

/// The layout to mix a side by: the one its caps declared, else the count's
/// conventional layout. `None` when there is no usable layout (a count over 8
/// with nothing declared, or a declared layout whose speaker count disagrees
/// with the caps' channel count), which drops back to the position-unaware path.
fn mixing_layout(declared: ChannelLayout, channels: usize) -> Option<ChannelLayout> {
    let count = u8::try_from(channels).ok()?;
    declared
        .or_default_for(count)
        .filter(|layout| layout.channels() == count)
}

/// Output sample for channel `oc`, given the interleaved input frame: the
/// position-unaware path, used for the mono / stereo cases and as the fallback
/// when a count has no conventional layout (`mix_matrix` returned `None`).
/// Identity when counts match; mono fan-out (replicate the one input); downmix
/// to mono (average all inputs); a round-robin fold for a general downmix
/// (`out_ch` < `in_ch`, `out_ch` >= 2: output `oc` averages inputs
/// `oc, oc+out_ch, oc+2*out_ch, ...`, so no channel is dropped); and a
/// round-robin replicate for upmix (`out_ch` > `in_ch`, `in_ch` >= 2).
fn map_channel(in_samples: &[f32], oc: usize, out_ch: usize) -> f32 {
    let in_ch = in_samples.len();
    if in_ch == out_ch {
        in_samples[oc] // identity
    } else if in_ch == 1 {
        in_samples[0] // mono fan-out
    } else if out_ch == 1 {
        in_samples.iter().sum::<f32>() / in_ch as f32 // downmix to mono
    } else if out_ch < in_ch {
        // General downmix: fold input channels into outputs round-robin and
        // average each group, so every input contributes (no silent drop).
        let mut sum = 0.0;
        let mut n = 0u32;
        let mut i = oc;
        while i < in_ch {
            sum += in_samples[i];
            n += 1;
            i += out_ch;
        }
        sum / n as f32
    } else {
        in_samples[oc % in_ch] // upmix: round-robin replicate
    }
}

/// Round half away from zero without libm.
fn round_away(v: f32) -> f32 {
    if v >= 0.0 {
        v + 0.5
    } else {
        v - 0.5
    }
}

/// The integer value `round_away` plus the truncating cast in
/// [`emit_quantized`] land on, so a shaping filter can weigh the error the
/// emitted sample actually makes.
fn round_to_integer(v: f32) -> f32 {
    round_away(v) as i64 as f32
}

/// Full-scale integer magnitude of a PCM format: the multiplier from the
/// `[-1, 1]` float intermediate, and the format's largest positive code.
/// `None` for float output, which has no quantizer.
fn full_scale(format: AudioFormat) -> Option<f32> {
    match format {
        AudioFormat::PcmU8 => Some(127.0),
        AudioFormat::PcmS16Le => Some(32_767.0),
        AudioFormat::PcmS24Le => Some(8_388_607.0),
        AudioFormat::PcmS32Le => Some(2_147_483_647.0),
        _ => None,
    }
}

/// Resolution of a PCM format in bits, what [`DITHERING_THRESHOLD`] compares
/// against. `None` for float, which is treated as losing nothing.
fn quantization_bits(format: AudioFormat) -> Option<u8> {
    match format {
        AudioFormat::PcmU8 => Some(8),
        AudioFormat::PcmS16Le => Some(16),
        AudioFormat::PcmS24Le => Some(24),
        AudioFormat::PcmS32Le => Some(32),
        _ => None,
    }
}

/// Append an already-rounded integer sample value, clamped to the format's
/// code range. The clamp matters once dither pushes a full-scale sample past
/// the top code, where `as u8` would wrap rather than saturate.
fn emit_quantized(dst: &mut Vec<u8>, quantized: f32, format: AudioFormat) {
    let Some(scale) = full_scale(format) else {
        return;
    };
    // the negative side has one more code than the positive one.
    let q = quantized.clamp(-(scale + 1.0), scale);
    match format {
        // u8 PCM is offset-binary: silence sits at 0x80.
        AudioFormat::PcmU8 => dst.push((q as i32 + 128) as u8),
        AudioFormat::PcmS16Le => dst.extend_from_slice(&(q as i16).to_le_bytes()),
        // 3-byte packed, little-endian.
        AudioFormat::PcmS24Le => dst.extend_from_slice(&(q as i32).to_le_bytes()[..3]),
        // the +0.5 in `round_away` is below the f32 quantum at this magnitude;
        // `as i32` saturates rather than wrapping at full scale.
        AudioFormat::PcmS32Le => dst.extend_from_slice(&(q as i32).to_le_bytes()),
        _ => {}
    }
}

/// Decode one sample to f32 in [-1, 1). The slice starts at the sample.
pub(crate) fn read_sample(at: &[u8], format: AudioFormat) -> f32 {
    match format {
        AudioFormat::PcmU8 => (at[0] as f32 - 128.0) / 128.0,
        AudioFormat::PcmS16Le => {
            let s = i16::from_le_bytes([at[0], at[1]]);
            s as f32 / 32768.0
        }
        AudioFormat::PcmS24Le => {
            // 3-byte packed, little-endian: sign-extend from the top byte.
            let s = at[0] as i32 | (at[1] as i32) << 8 | (at[2] as i8 as i32) << 16;
            s as f32 / 8_388_608.0
        }
        AudioFormat::PcmS32Le => {
            let s = i32::from_le_bytes([at[0], at[1], at[2], at[3]]);
            s as f32 / 2_147_483_648.0
        }
        AudioFormat::PcmF32Le => f32::from_le_bytes([at[0], at[1], at[2], at[3]]),
        _ => 0.0,
    }
}

/// Encode one f32 sample, appending its little-endian bytes. Plain rounding,
/// no dither: the resampler and the audio filters call this, and a
/// [`Quantizer`] is what carries the per-channel dither state.
pub(crate) fn write_sample(dst: &mut Vec<u8>, v: f32, format: AudioFormat) {
    if format == AudioFormat::PcmF32Le {
        dst.extend_from_slice(&v.to_le_bytes());
        return;
    }
    let Some(scale) = full_scale(format) else {
        return;
    };
    emit_quantized(dst, round_away(v.clamp(-1.0, 1.0) * scale), format);
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::PassthroughFields;

    fn audio(format: AudioFormat, channels: u8, rate: u32) -> Caps {
        Caps::Audio {
            format,
            channels,
            sample_rate: rate,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }

    /// A quantizer with dither and shaping off, so the output is the plain
    /// rounding these expectations were written against.
    fn plain_quantizer() -> Quantizer {
        Quantizer {
            dither: DitherMethod::None,
            threshold: DITHERING_THRESHOLD,
            ..Quantizer::default()
        }
    }

    fn convert(
        src: &[u8],
        in_format: AudioFormat,
        in_channels: u8,
        out_format: AudioFormat,
        out_channels: u8,
    ) -> Result<Box<[u8]>, G2gError> {
        convert_pcm(
            src,
            mono_stream(in_format, in_channels),
            mono_stream(out_format, out_channels),
            &mut plain_quantizer(),
        )
    }

    /// One output LSB of a 16-bit sample, in the `[-1, 1]` float intermediate.
    fn s16_step() -> f64 {
        1.0 / f64::from(full_scale(AudioFormat::PcmS16Le).unwrap())
    }

    fn s16_samples(bytes: &[u8]) -> Vec<i16> {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| i16::from_le_bytes(*b))
            .collect()
    }

    #[test]
    fn dither_off_keeps_a_reduction_on_the_plain_rounding_path() {
        // With dither off, a float-to-16-bit reduction is exactly what
        // `write_sample` produces on its own.
        let src: Vec<u8> = (0..64u8)
            .flat_map(|i| (f32::from(i) / 64.0 - 0.5).to_le_bytes())
            .collect();
        let got = convert(&src, AudioFormat::PcmF32Le, 1, AudioFormat::PcmS16Le, 1).unwrap();
        let mut want = Vec::new();
        for chunk in src.as_chunks::<4>().0 {
            write_sample(&mut want, f32::from_le_bytes(*chunk), AudioFormat::PcmS16Le);
        }
        assert_eq!(&got[..], &want[..]);
    }

    #[test]
    fn dither_skips_conversions_that_lose_nothing() {
        let src: Vec<u8> = (-7i16..8).flat_map(|s| (s * 4096).to_le_bytes()).collect();
        let mut loud = plain_quantizer();
        loud.dither = DitherMethod::Tpdf;
        loud.shaping = NoiseShapingMethod::High;
        let same_depth = |quantizer: &mut Quantizer, out_format| {
            convert_pcm(
                &src,
                mono_stream(AudioFormat::PcmS16Le, 1),
                mono_stream(out_format, 1),
                quantizer,
            )
            .unwrap()
        };
        // Equal depth, and a widening: neither drops a bit, so the requested
        // dither and shaping must not run.
        for out_format in [AudioFormat::PcmS16Le, AudioFormat::PcmS32Le] {
            assert_eq!(
                same_depth(&mut loud, out_format),
                same_depth(&mut plain_quantizer(), out_format),
                "{out_format:?} loses nothing to dither"
            );
        }
    }

    #[test]
    fn dithering_threshold_gates_the_output_depth() {
        // S24 is wider than the declared threshold, so a reduction into it is
        // still left alone; S16 is at or below it.
        let src: Vec<u8> = (0..64u8)
            .flat_map(|i| (f32::from(i) / 64.0 - 0.5).to_le_bytes())
            .collect();
        let mut tpdf = plain_quantizer();
        tpdf.dither = DitherMethod::Tpdf;
        assert!(tpdf.threshold < quantization_bits(AudioFormat::PcmS24Le).unwrap());
        assert_eq!(
            tpdf.plan(AudioFormat::PcmF32Le, AudioFormat::PcmS24Le)
                .dither,
            DitherMethod::None
        );
        let dithered = convert_pcm(
            &src,
            mono_stream(AudioFormat::PcmF32Le, 1),
            mono_stream(AudioFormat::PcmS24Le, 1),
            &mut tpdf,
        )
        .unwrap();
        let plain = convert(&src, AudioFormat::PcmF32Le, 1, AudioFormat::PcmS24Le, 1).unwrap();
        assert_eq!(dithered, plain);
    }

    #[test]
    fn tpdf_dither_recovers_a_tone_below_the_quantization_step() {
        // A sine whose peak sits under half an output LSB rounds to nothing:
        // plain quantization returns silence and the tone is gone. Dither
        // decorrelates the rounding error from the signal, and the tone comes
        // back in the average.
        const TONE_PERIOD: usize = 8;
        const FRAMES: usize = TONE_PERIOD * 512;
        const AMPLITUDE_LSB: f64 = 0.3;
        let phase = |i: usize| core::f64::consts::TAU * i as f64 / TONE_PERIOD as f64;
        let mut src = Vec::new();
        for i in 0..FRAMES {
            let v = AMPLITUDE_LSB * s16_step() * crate::mathf::sin(phase(i));
            src.extend_from_slice(&(v as f32).to_le_bytes());
        }
        let quantized = |dither| {
            let mut q = plain_quantizer();
            q.dither = dither;
            s16_samples(
                &convert_pcm(
                    &src,
                    mono_stream(AudioFormat::PcmF32Le, 1),
                    mono_stream(AudioFormat::PcmS16Le, 1),
                    &mut q,
                )
                .unwrap(),
            )
        };

        assert!(
            quantized(DitherMethod::None).iter().all(|&s| s == 0),
            "plain rounding loses the tone entirely"
        );

        // Project the output onto the tone: for a sine of amplitude A,
        // (2 / N) * sum(x[i] * sin) estimates A.
        let dithered = quantized(DitherMethod::Tpdf);
        let recovered = 2.0 / FRAMES as f64
            * dithered
                .iter()
                .enumerate()
                .map(|(i, &s)| f64::from(s) * crate::mathf::sin(phase(i)))
                .sum::<f64>();
        // What is left over is the dithered quantizer's noise, variance a
        // quarter of an LSB squared, so this estimator's standard error is
        // sqrt(2 * 0.25 / FRAMES), about 0.011 LSB. The tolerance is four of
        // those.
        const TOLERANCE_LSB: f64 = 0.05;
        assert!(
            (recovered - AMPLITUDE_LSB).abs() < TOLERANCE_LSB,
            "recovered {recovered} LSB from a {AMPLITUDE_LSB} LSB tone"
        );
    }

    #[test]
    fn error_feedback_shaping_carries_a_sub_step_level_into_the_output() {
        // Error feedback subtracts the previous rounding error from the next
        // sample, so the errors telescope and only the first and last survive
        // in the sum. Each is under half an LSB, so over N samples the mean
        // output is within 1 / N of a DC level plain rounding would erase.
        const FRAMES: usize = 1000;
        const LEVEL_LSB: f64 = 0.3;
        let level = (LEVEL_LSB * s16_step()) as f32;
        let src: Vec<u8> = (0..FRAMES).flat_map(|_| level.to_le_bytes()).collect();
        let mut shaped = plain_quantizer();
        shaped.shaping = NoiseShapingMethod::ErrorFeedback;
        let got = s16_samples(
            &convert_pcm(
                &src,
                mono_stream(AudioFormat::PcmF32Le, 1),
                mono_stream(AudioFormat::PcmS16Le, 1),
                &mut shaped,
            )
            .unwrap(),
        );
        let mean = got.iter().map(|&s| f64::from(s)).sum::<f64>() / FRAMES as f64;
        assert!(
            (mean - LEVEL_LSB).abs() <= 1.0 / FRAMES as f64,
            "mean {mean} LSB from a {LEVEL_LSB} LSB level"
        );
    }

    #[test]
    fn the_element_dithers_by_default_like_gst() {
        // gst-inspect-1.0 audioconvert: dithering Default: 2, "tpdf".
        let conv = AudioConvert::auto();
        assert_eq!(
            conv.get_property("dithering"),
            Some(PropValue::Str("tpdf".into()))
        );
        assert_eq!(
            conv.quantizer
                .plan(AudioFormat::PcmF32Le, AudioFormat::PcmS16Le)
                .dither,
            DitherMethod::Tpdf,
            "a reduction dithers with nobody setting the property"
        );
        assert!(
            conv.quantizer
                .plan(AudioFormat::PcmS16Le, AudioFormat::PcmS16Le)
                .is_plain(),
            "an equal-depth conversion stays on the plain rounding path"
        );
    }

    #[test]
    fn dither_properties_round_trip_and_report_their_declared_defaults() {
        let mut conv = AudioConvert::new(AudioFormat::PcmS16Le, 2);
        for name in ["dithering", "dithering-threshold", "noise-shaping"] {
            let spec = conv
                .properties()
                .iter()
                .find(|s| s.name == name)
                .unwrap_or_else(|| panic!("`{name}` is declared"));
            let declared = spec.default.expect("declares a default");
            assert_eq!(
                conv.get_property(name),
                Some(spec.parse_value(declared).expect("the default parses")),
                "`{name}` reports the default it declares"
            );
        }
        conv.set_property("dithering", PropValue::Str("tpdf-hf".into()))
            .unwrap();
        assert_eq!(
            conv.get_property("dithering"),
            Some(PropValue::Str("tpdf-hf".into()))
        );
        conv.set_property("noise-shaping", PropValue::Str("medium".into()))
            .unwrap();
        assert_eq!(
            conv.get_property("noise-shaping"),
            Some(PropValue::Str("medium".into()))
        );
        conv.set_property("dithering-threshold", PropValue::Uint(16))
            .unwrap();
        assert_eq!(
            conv.get_property("dithering-threshold"),
            Some(PropValue::Uint(16))
        );
        assert_eq!(
            conv.set_property("dithering", PropValue::Str("gaussian".into())),
            Err(PropError::Value)
        );
        assert_eq!(
            conv.set_property(
                "dithering-threshold",
                PropValue::Uint(MAX_DITHERING_THRESHOLD + 1)
            ),
            Err(PropError::Value)
        );
    }

    #[test]
    fn rejects_compressed_target_format() {
        let mut conv = AudioConvert::new(AudioFormat::PcmS16Le, 2);
        // AAC/OPUS are not raw-PCM formats; setting them must fail loud rather
        // than silently emit empty frames.
        assert_eq!(
            conv.set_property("format", PropValue::Str("aac".into())),
            Err(PropError::Value)
        );
        assert_eq!(
            conv.set_property("format", PropValue::Str("opus".into())),
            Err(PropError::Value)
        );
        assert!(conv
            .set_property("format", PropValue::Str("f32le".into()))
            .is_ok());
        assert_eq!(conv.target_format(), AudioFormat::PcmF32Le);
    }

    #[test]
    fn fixed_target_maps_pcm_to_target() {
        let conv = AudioConvert::new(AudioFormat::PcmS16Le, 2);
        let CapsConstraint::DerivedFields(t) = conv.caps_constraint_as_transform() else {
            panic!("expected DerivedFields");
        };
        // only sample_rate is preserved; format + channels are retargeted.
        assert_eq!(t.passthrough(), PassthroughFields::NONE.with_sample_rate());
        let f = |c: &Caps| t.derive(c);
        let out = f(&audio(AudioFormat::PcmF32Le, 2, 44_100));
        assert_eq!(
            out.alternatives(),
            &[audio(AudioFormat::PcmS16Le, 2, 44_100)]
        );
        // compressed audio is not convertible
        assert!(f(&audio(AudioFormat::Aac, 2, 48_000)).is_empty());
        // a multi-channel remap (3 -> 2) now produces the target layout.
        assert_eq!(
            f(&audio(AudioFormat::PcmF32Le, 3, 48_000)).alternatives(),
            &[audio(AudioFormat::PcmS16Le, 2, 48_000)]
        );
    }

    #[test]
    fn auto_target_advertises_passthrough_and_retarget_options() {
        // Caps-driven: a downstream capsfilter should be able to pin either PCM
        // format and any channel count, so the derive advertises the passthrough
        // (input) shape first plus the retarget alternatives.
        let conv = AudioConvert::auto();
        let CapsConstraint::DerivedFields(t) = conv.caps_constraint_as_transform() else {
            panic!("expected DerivedFields");
        };
        let f = |c: &Caps| t.derive(c);
        let out = f(&audio(AudioFormat::PcmS16Le, 2, 48_000));
        let alts = out.alternatives();
        // passthrough (S16, 2) is the preferred first alternative.
        assert_eq!(alts[0], audio(AudioFormat::PcmS16Le, 2, 48_000));
        // a mono capsfilter pins through the ANY_CHANNELS wildcard alternative.
        assert!(alts.contains(&audio(AudioFormat::PcmS16Le, ANY_CHANNELS, 48_000)));
        // the other PCM format is offered so a format-changing capsfilter matches.
        assert!(alts.iter().any(|c| matches!(
            c,
            Caps::Audio {
                format: AudioFormat::PcmF32Le,
                ..
            }
        )));
        // the decoder's pre-decode ANY_CHANNELS placeholder still derives an
        // output (the wildcard), rather than an empty set.
        assert!(!f(&audio(AudioFormat::PcmS16Le, ANY_CHANNELS, 48_000)).is_empty());
    }

    #[test]
    fn f32_to_s16_round_trips_within_a_quantum() {
        // a few f32 values -> s16 -> f32 must stay within one 16-bit step.
        let src_f32: Vec<u8> = [0.0f32, 0.5, -0.5, 1.0, -1.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let s16 = convert(&src_f32, AudioFormat::PcmF32Le, 1, AudioFormat::PcmS16Le, 1).unwrap();
        let back = convert(&s16, AudioFormat::PcmS16Le, 1, AudioFormat::PcmF32Le, 1).unwrap();
        for (i, chunk) in back.as_chunks::<4>().0.iter().enumerate() {
            let got = f32::from_le_bytes(*chunk);
            let want = [0.0f32, 0.5, -0.5, 1.0, -1.0][i];
            assert!(
                (got - want).abs() < 1.0 / 32767.0 + 1e-6,
                "sample {i}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn s16_peak_maps_near_full_scale_float() {
        // i16 max -> ~1.0 f32.
        let s16: Vec<u8> = i16::MAX.to_le_bytes().to_vec();
        let f32b = convert(&s16, AudioFormat::PcmS16Le, 1, AudioFormat::PcmF32Le, 1).unwrap();
        let v = f32::from_le_bytes(f32b[..4].try_into().unwrap());
        assert!((v - 1.0).abs() < 1e-3, "got {v}");
    }

    #[test]
    fn s24_packs_three_bytes_little_endian() {
        // one f32 sample -> 3 bytes, not a 32-bit container.
        let src: Vec<u8> = 0.5f32.to_le_bytes().to_vec();
        let s24 = convert(&src, AudioFormat::PcmF32Le, 1, AudioFormat::PcmS24Le, 1).unwrap();
        assert_eq!(s24.len(), 3);
        let v = s24[0] as i32 | (s24[1] as i32) << 8 | (s24[2] as i8 as i32) << 16;
        assert!((v - 4_194_303).abs() <= 1, "got {v}");
        // negative values sign-extend out of the top byte.
        let src: Vec<u8> = (-0.5f32).to_le_bytes().to_vec();
        let s24 = convert(&src, AudioFormat::PcmF32Le, 1, AudioFormat::PcmS24Le, 1).unwrap();
        let v = s24[0] as i32 | (s24[1] as i32) << 8 | (s24[2] as i8 as i32) << 16;
        assert!((v + 4_194_303).abs() <= 1, "got {v}");
    }

    #[test]
    fn the_wide_formats_round_trip_within_their_quantum() {
        let values = [0.0f32, 0.5, -0.5, 1.0, -1.0, 0.123];
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        // quantum per format: u8 is coarse, s24 / s32 are finer than f32's error.
        for (format, tol) in [
            (AudioFormat::PcmU8, 1.0 / 127.0),
            (AudioFormat::PcmS24Le, 1e-6),
            (AudioFormat::PcmS32Le, 1e-6),
        ] {
            let packed = convert(&src, AudioFormat::PcmF32Le, 1, format, 1).unwrap();
            assert_eq!(
                packed.len(),
                values.len() * sample_bytes(format),
                "{format:?}"
            );
            let back = convert(&packed, format, 1, AudioFormat::PcmF32Le, 1).unwrap();
            for (i, chunk) in back.as_chunks::<4>().0.iter().enumerate() {
                let got = f32::from_le_bytes(*chunk);
                assert!(
                    (got - values[i]).abs() <= tol,
                    "{format:?} sample {i}: {got} vs {}",
                    values[i]
                );
            }
        }
    }

    #[test]
    fn u8_silence_sits_at_the_offset_binary_midpoint() {
        let src: Vec<u8> = 0.0f32.to_le_bytes().to_vec();
        let u8s = convert(&src, AudioFormat::PcmF32Le, 1, AudioFormat::PcmU8, 1).unwrap();
        assert_eq!(&u8s[..], [128]);
        // and it reads back as silence.
        let back = convert(&[128], AudioFormat::PcmU8, 1, AudioFormat::PcmF32Le, 1).unwrap();
        assert_eq!(f32::from_le_bytes(back[..4].try_into().unwrap()), 0.0);
    }

    #[test]
    fn mono_fans_out_to_stereo() {
        // one s16 sample (value 1000) -> two identical channels.
        let mono: Vec<u8> = 1000i16.to_le_bytes().to_vec();
        let stereo = convert(&mono, AudioFormat::PcmS16Le, 1, AudioFormat::PcmS16Le, 2).unwrap();
        assert_eq!(stereo.len(), 4);
        assert_eq!(i16::from_le_bytes([stereo[0], stereo[1]]), 1000);
        assert_eq!(i16::from_le_bytes([stereo[2], stereo[3]]), 1000);
    }

    #[test]
    fn stereo_downmixes_to_mono_average() {
        // L=1000, R=2000 -> mono 1500.
        let mut stereo = Vec::new();
        stereo.extend_from_slice(&1000i16.to_le_bytes());
        stereo.extend_from_slice(&2000i16.to_le_bytes());
        let mono = convert(&stereo, AudioFormat::PcmS16Le, 2, AudioFormat::PcmS16Le, 1).unwrap();
        assert_eq!(mono.len(), 2);
        let v = i16::from_le_bytes([mono[0], mono[1]]);
        assert!((v - 1500).abs() <= 1, "got {v}");
    }

    #[test]
    fn ragged_input_fails_loud() {
        // 3 bytes is not a whole s16 stereo frame (4 bytes).
        assert_eq!(
            convert(
                &[0, 0, 0],
                AudioFormat::PcmS16Le,
                2,
                AudioFormat::PcmS16Le,
                2
            ),
            Err(G2gError::CapsMismatch)
        );
    }

    #[test]
    fn configure_accepts_any_channel_count_and_wildcard() {
        let mut conv = AudioConvert::new(AudioFormat::PcmS16Le, 2);
        // 5.1 -> stereo now configures (a real runtime CapsChanged for multichannel
        // content); identity is fine; ANY_CHANNELS (0) is the negotiation placeholder.
        assert!(conv
            .configure_pipeline(&audio(AudioFormat::PcmF32Le, 6, 48_000))
            .is_ok());
        assert!(conv
            .configure_pipeline(&audio(AudioFormat::PcmF32Le, 2, 48_000))
            .is_ok());
        assert!(conv
            .configure_pipeline(&audio(AudioFormat::PcmF32Le, 0, 48_000))
            .is_ok());
        // a non-PCM input still fails loud.
        assert!(matches!(
            conv.configure_pipeline(&audio(AudioFormat::Aac, 2, 48_000)),
            Err(G2gError::CapsMismatch)
        ));
    }

    /// One interleaved f32 frame in, one out, through `convert_pcm`.
    fn mix_f32(input: &[f32], out_ch: u8) -> Vec<f32> {
        let mut bytes = Vec::new();
        for v in input {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let out = convert(
            &bytes,
            AudioFormat::PcmF32Le,
            input.len() as u8,
            AudioFormat::PcmF32Le,
            out_ch,
        )
        .unwrap();
        out.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Mix `input` (one interleaved frame) down to `out_ch`, taking the input
    /// speakers from `in_layout` rather than the count convention.
    fn mix_f32_with_layout(input: &[f32], in_layout: ChannelLayout, out_ch: u8) -> Vec<f32> {
        let mut bytes = Vec::new();
        for v in input {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let out = convert_pcm(
            &bytes,
            PcmStream {
                format: AudioFormat::PcmF32Le,
                channels: input.len() as u8,
                layout: in_layout,
            },
            mono_stream(AudioFormat::PcmF32Le, out_ch),
            &mut plain_quantizer(),
        )
        .unwrap();
        out.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn a_declared_layout_picks_different_downmix_coefficients() {
        // Quad (FL FR BL BR) against the 4-channel default (FL FR FC BC): the
        // back pair folds to its own side at 1/sqrt(2), while a center + back
        // center spreads across both fronts, so the two rows differ.
        let quad = ChannelLayout::of(&[
            ChannelPosition::Fl,
            ChannelPosition::Fr,
            ChannelPosition::Bl,
            ChannelPosition::Br,
        ]);
        let default_four = ChannelLayout::default_for(4).unwrap();
        assert_ne!(quad, default_four);

        let quad_matrix = mix_matrix(quad, ChannelLayout::STEREO);
        let default_matrix = mix_matrix(default_four, ChannelLayout::STEREO);
        assert_ne!(quad_matrix, default_matrix);
        // FL keeps 1.0 and BL folds in at 1/sqrt(2), normalized by the row sum.
        let quad_norm = 1.0 + SURROUND_MIX;
        assert_close(
            &quad_matrix,
            &[
                1.0 / quad_norm,
                0.0,
                SURROUND_MIX / quad_norm,
                0.0,
                0.0,
                1.0 / quad_norm,
                0.0,
                SURROUND_MIX / quad_norm,
            ],
        );

        // The same samples, converted: an unspecified layout keeps today's
        // count-convention result, a declared quad does not.
        let frame = [1.0f32, 0.0, 1.0, 0.0];
        let as_declared = mix_f32_with_layout(&frame, quad, 2);
        let as_unspecified = mix_f32_with_layout(&frame, ChannelLayout::UNSPECIFIED, 2);
        assert_close(&as_unspecified, &mix_f32(&frame, 2));
        assert_ne!(as_declared, as_unspecified);
        assert_close(&as_declared, &[(1.0 + SURROUND_MIX) / quad_norm, 0.0]);
    }

    /// A stream of `channels` channels with no declared speaker layout, the
    /// shape every caps carried before a layout could be declared.
    fn mono_stream(format: AudioFormat, channels: u8) -> PcmStream {
        PcmStream {
            format,
            channels,
            layout: ChannelLayout::UNSPECIFIED,
        }
    }

    fn assert_close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < 1e-4, "ch{i}: got {g}, want {w}");
        }
    }

    // The downmix expectations below are ffmpeg's default rematrix coefficients
    // (swresample `Matrix coefficients` debug dump), the BS.775-derived
    // reference this matrix reproduces.

    #[test]
    fn five_one_downmixes_to_stereo_with_itu_coefficients() {
        // 5.1 order FL FR FC LFE BL BR.
        // L = 0.414214 FL + 0.292893 FC + 0.292893 BL, R symmetric, LFE dropped.
        let out = mix_f32(&[0.2, 0.1, 0.4, 0.9, 0.3, 0.05], 2);
        let l = 0.414214 * 0.2 + 0.292893 * 0.4 + 0.292893 * 0.3;
        let r = 0.414214 * 0.1 + 0.292893 * 0.4 + 0.292893 * 0.05;
        assert_close(&out, &[l, r]);
    }

    #[test]
    fn five_one_downmixes_to_mono() {
        // M = 0.207107 (FL+FR) + 0.292893 FC + 0.146447 (BL+BR).
        let out = mix_f32(&[0.2, 0.1, 0.4, 0.9, 0.3, 0.05], 1);
        let m = 0.207107 * (0.2 + 0.1) + 0.292893 * 0.4 + 0.146447 * (0.3 + 0.05);
        assert_close(&out, &[m]);
    }

    #[test]
    fn seven_one_downmixes_to_stereo() {
        // 7.1 order FL FR FC LFE BL BR SL SR.
        // L = 0.320377 FL + 0.226541 (FC + BL + SL).
        let out = mix_f32(&[0.2, 0.1, 0.4, 0.9, 0.3, 0.05, 0.25, 0.15], 2);
        let l = 0.320377 * 0.2 + 0.226541 * (0.4 + 0.3 + 0.25);
        let r = 0.320377 * 0.1 + 0.226541 * (0.4 + 0.05 + 0.15);
        assert_close(&out, &[l, r]);
    }

    #[test]
    fn six_one_back_center_splits_into_both_fronts() {
        // 6.1 order FL FR FC LFE BC SL SR.
        // L = 0.343146 FL + 0.242641 (FC + SL) + 0.171573 BC.
        let out = mix_f32(&[0.2, 0.1, 0.4, 0.9, 0.3, 0.25, 0.15], 2);
        let l = 0.343146 * 0.2 + 0.242641 * (0.4 + 0.25) + 0.171573 * 0.3;
        let r = 0.343146 * 0.1 + 0.242641 * (0.4 + 0.15) + 0.171573 * 0.3;
        assert_close(&out, &[l, r]);
    }

    #[test]
    fn upmix_places_positions_and_leaves_the_rest_silent() {
        // Stereo -> 5.1: FL/FR at their own speakers, everything else silent.
        let out = mix_f32(&[0.5, -0.25], 6);
        assert_close(&out, &[0.5, -0.25, 0.0, 0.0, 0.0, 0.0]);
        // Mono (FC) -> 5.1 goes to the center speaker only.
        let out = mix_f32(&[0.5], 6);
        assert_close(&out, &[0.0, 0.0, 0.5, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn lfe_is_dropped_on_downmix() {
        let out = mix_f32(&[0.0, 0.0, 0.0, 0.9, 0.0, 0.0], 2);
        assert_close(&out, &[0.0, 0.0]);
    }

    #[test]
    fn counts_past_the_layout_table_fall_back_to_round_robin() {
        // 9 channels have no conventional layout: the layout-agnostic fold
        // applies (L = avg of ch 0,2,4,6,8; R = avg of ch 1,3,5,7).
        let input: Vec<f32> = (0..9).map(|c| c as f32 / 10.0).collect();
        let out = mix_f32(&input, 2);
        let l = (0.0 + 0.2 + 0.4 + 0.6 + 0.8) / 5.0;
        let r = (0.1 + 0.3 + 0.5 + 0.7) / 4.0;
        assert_close(&out, &[l, r]);
    }
}
