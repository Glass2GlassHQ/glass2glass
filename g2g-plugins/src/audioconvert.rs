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

/// The PCM sample formats `AudioConvert` / `AudioResample` read and write.
pub(crate) const PCM_FORMATS: [AudioFormat; 5] = [
    AudioFormat::PcmS16Le,
    AudioFormat::PcmF32Le,
    AudioFormat::PcmS24Le,
    AudioFormat::PcmS32Le,
    AudioFormat::PcmU8,
];

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
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl AudioConvert {
    pub fn new(target_format: AudioFormat, target_channels: u8) -> Self {
        assert!(target_channels > 0, "target channels must be non-zero");
        assert!(
            PCM_FORMATS.contains(&target_format),
            "AudioConvert is a raw-PCM converter; target must be a PCM format"
        );
        Self {
            target_format: Some(target_format),
            target_channels: Some(target_channels),
            resolved: None,
            input: None,
            configured: false,
            last_caps: None,
            emitted: 0,
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

    /// Validate a PCM caps as a convertible input, returning its
    /// format/channels/rate. Any concrete input channel count converts to the
    /// target (identity / fan-out / downmix / layout-agnostic remap);
    /// `ANY_CHANNELS` (0) is the negotiation placeholder, accepted here with the
    /// real count arriving via a `CapsChanged` before the first frame.
    fn accept_input(&self, caps: &Caps) -> Result<(AudioFormat, u8, u32), G2gError> {
        let Caps::Audio {
            format,
            channels,
            sample_rate,
        } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !PCM_FORMATS.contains(format) {
            return Err(G2gError::CapsMismatch);
        }
        Ok((*format, *channels, *sample_rate))
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
        // not reachable: only PCM_FORMATS pass negotiation.
        _ => 0,
    }
}

impl AsyncElement for AudioConvert {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

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
                v.extend(PCM_FORMATS.iter().copied().map(FieldTransform::Fixed));
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
            accept: PCM_FORMATS.to_vec(),
            produce: Vec::new(),
            shapes,
        })
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (format, channels, rate) = self.accept_input(absolute_caps)?;
        self.input = Some((format, channels, rate));
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    /// Caps-driven: take the output format + channel count from the negotiated
    /// output caps when a target is unset (auto). Already fixated, so concrete.
    fn configure_output(&mut self, output_caps: &Caps) -> Result<(), G2gError> {
        let Caps::Audio {
            format, channels, ..
        } = output_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !PCM_FORMATS.contains(format) || *channels == ANY_CHANNELS {
            return Err(G2gError::CapsMismatch);
        }
        self.resolved = Some((*format, *channels));
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
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let out_format = self.out_format(in_format);
                    let out_channels = self.out_channels(in_channels);
                    let converted =
                        convert_pcm(slice, in_format, in_channels, out_format, out_channels)?;

                    let new_caps = Caps::Audio {
                        format: out_format,
                        channels: out_channels,
                        sample_rate: rate,
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
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "format" => Some(PropValue::Str(
                audio_format_to_str(self.target_format()).into(),
            )),
            "channels" => Some(PropValue::Uint(self.target_channels() as u64)),
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
];

/// Parse an audio-format property string to an [`AudioFormat`]. Shared with the
/// `gst-launch` DSL. GStreamer names raw sample formats uppercase (S16LE,
/// F32LE); accept any case and the historical lowercase spellings as aliases.
pub(crate) fn audio_format_from_str(s: &str) -> Option<AudioFormat> {
    // Only the PCM formats are valid AudioConvert targets; AAC/OPUS are encoder
    // outputs, not something a raw-sample converter can produce.
    match s.to_ascii_lowercase().as_str() {
        "s16le" => Some(AudioFormat::PcmS16Le),
        "f32le" => Some(AudioFormat::PcmF32Le),
        "s24le" => Some(AudioFormat::PcmS24Le),
        "s32le" => Some(AudioFormat::PcmS32Le),
        "u8" => Some(AudioFormat::PcmU8),
        _ => None,
    }
}

/// The canonical (GStreamer) property string for an [`AudioFormat`].
pub(crate) fn audio_format_to_str(f: AudioFormat) -> &'static str {
    match f {
        AudioFormat::PcmS16Le => "S16LE",
        AudioFormat::PcmF32Le => "F32LE",
        AudioFormat::PcmS24Le => "S24LE",
        AudioFormat::PcmS32Le => "S32LE",
        AudioFormat::PcmU8 => "U8",
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
        };
        let set = CapsSet::from_alternatives(PCM_FORMATS.map(pcm).to_vec());
        Vec::from([PadTemplate::sink(set.clone()), PadTemplate::source(set)])
    }
}

/// Read interleaved PCM and re-emit it in the target format/channel count.
/// Samples pass through an f32 intermediate; channel mapping is identity,
/// mono fan-out, or downmix-to-mono average.
fn convert_pcm(
    src: &[u8],
    in_format: AudioFormat,
    in_channels: u8,
    out_format: AudioFormat,
    out_channels: u8,
) -> Result<Box<[u8]>, G2gError> {
    let in_bytes = sample_bytes(in_format);
    let out_bytes = sample_bytes(out_format);
    let in_ch = in_channels as usize;
    let out_ch = out_channels as usize;
    let in_frame = in_bytes * in_ch;
    if in_frame == 0 || src.len() % in_frame != 0 {
        return Err(G2gError::CapsMismatch);
    }
    let frames = src.len() / in_frame;

    // Position-aware mixing whenever either side is true multichannel and both
    // counts have a conventional layout; None keeps the layout-agnostic paths.
    let matrix = if in_ch != out_ch && in_ch.max(out_ch) > 2 {
        mix_matrix(in_ch, out_ch)
    } else {
        None
    };

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
            write_sample(&mut dst, v, out_format);
        }
    }
    Ok(dst.into_boxed_slice())
}

/// ITU BS.775-style surround coefficient: center and surrounds mix into the
/// fronts at 1/sqrt(2).
const SURROUND_MIX: f32 = core::f32::consts::FRAC_1_SQRT_2;

/// Build the position-aware mixing matrix (flattened `out_ch` rows of `in_ch`
/// coefficients) between the conventional layouts of two channel counts, or
/// `None` when either count has no conventional layout (> 8 channels).
///
/// Each input position routes to the output: itself at 1.0 when present; a
/// side <-> back counterpart at 1.0; otherwise folded frontward (center and
/// surrounds at 1/sqrt(2), back center at 0.5 into each front, surrounds at 0.5
/// into a mono center). LFE is dropped unless the output has one. The matrix is
/// then normalized by its largest output-row sum so a full-scale input cannot
/// clip. Coefficients match ffmpeg's default rematrix (verified against
/// `swr_build_matrix` output for 5.1 / 6.1 / 7.1 to stereo and mono).
fn mix_matrix(in_ch: usize, out_ch: usize) -> Option<Vec<f32>> {
    use ChannelPosition::*;
    let in_layout = ChannelLayout::default_for(in_ch as u8)?;
    let out_layout = ChannelLayout::default_for(out_ch as u8)?;
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
    Some(m)
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

/// Encode one f32 sample, appending its little-endian bytes.
pub(crate) fn write_sample(dst: &mut Vec<u8>, v: f32, format: AudioFormat) {
    let c = v.clamp(-1.0, 1.0);
    match format {
        // u8 PCM is offset-binary: silence sits at 0x80.
        AudioFormat::PcmU8 => dst.push((round_away(c * 127.0) as i32 + 128) as u8),
        AudioFormat::PcmS16Le => {
            dst.extend_from_slice(&(round_away(c * 32767.0) as i16).to_le_bytes());
        }
        AudioFormat::PcmS24Le => {
            let s = round_away(c * 8_388_607.0) as i32;
            dst.extend_from_slice(&s.to_le_bytes()[..3]);
        }
        // the +0.5 is below the f32 quantum at this magnitude; `as i32`
        // saturates rather than wrapping at full scale.
        AudioFormat::PcmS32Le => {
            dst.extend_from_slice(&(round_away(c * 2_147_483_647.0) as i32).to_le_bytes());
        }
        AudioFormat::PcmF32Le => dst.extend_from_slice(&v.to_le_bytes()),
        _ => {}
    }
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
        }
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
        let s16 =
            convert_pcm(&src_f32, AudioFormat::PcmF32Le, 1, AudioFormat::PcmS16Le, 1).unwrap();
        let back = convert_pcm(&s16, AudioFormat::PcmS16Le, 1, AudioFormat::PcmF32Le, 1).unwrap();
        for (i, chunk) in back.chunks_exact(4).enumerate() {
            let got = f32::from_le_bytes(chunk.try_into().unwrap());
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
        let f32b = convert_pcm(&s16, AudioFormat::PcmS16Le, 1, AudioFormat::PcmF32Le, 1).unwrap();
        let v = f32::from_le_bytes(f32b[..4].try_into().unwrap());
        assert!((v - 1.0).abs() < 1e-3, "got {v}");
    }

    #[test]
    fn s24_packs_three_bytes_little_endian() {
        // one f32 sample -> 3 bytes, not a 32-bit container.
        let src: Vec<u8> = 0.5f32.to_le_bytes().to_vec();
        let s24 = convert_pcm(&src, AudioFormat::PcmF32Le, 1, AudioFormat::PcmS24Le, 1).unwrap();
        assert_eq!(s24.len(), 3);
        let v = s24[0] as i32 | (s24[1] as i32) << 8 | (s24[2] as i8 as i32) << 16;
        assert!((v - 4_194_303).abs() <= 1, "got {v}");
        // negative values sign-extend out of the top byte.
        let src: Vec<u8> = (-0.5f32).to_le_bytes().to_vec();
        let s24 = convert_pcm(&src, AudioFormat::PcmF32Le, 1, AudioFormat::PcmS24Le, 1).unwrap();
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
            let packed = convert_pcm(&src, AudioFormat::PcmF32Le, 1, format, 1).unwrap();
            assert_eq!(
                packed.len(),
                values.len() * sample_bytes(format),
                "{format:?}"
            );
            let back = convert_pcm(&packed, format, 1, AudioFormat::PcmF32Le, 1).unwrap();
            for (i, chunk) in back.chunks_exact(4).enumerate() {
                let got = f32::from_le_bytes(chunk.try_into().unwrap());
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
        let u8s = convert_pcm(&src, AudioFormat::PcmF32Le, 1, AudioFormat::PcmU8, 1).unwrap();
        assert_eq!(&u8s[..], [128]);
        // and it reads back as silence.
        let back = convert_pcm(&[128], AudioFormat::PcmU8, 1, AudioFormat::PcmF32Le, 1).unwrap();
        assert_eq!(f32::from_le_bytes(back[..4].try_into().unwrap()), 0.0);
    }

    #[test]
    fn mono_fans_out_to_stereo() {
        // one s16 sample (value 1000) -> two identical channels.
        let mono: Vec<u8> = 1000i16.to_le_bytes().to_vec();
        let stereo =
            convert_pcm(&mono, AudioFormat::PcmS16Le, 1, AudioFormat::PcmS16Le, 2).unwrap();
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
        let mono =
            convert_pcm(&stereo, AudioFormat::PcmS16Le, 2, AudioFormat::PcmS16Le, 1).unwrap();
        assert_eq!(mono.len(), 2);
        let v = i16::from_le_bytes([mono[0], mono[1]]);
        assert!((v - 1500).abs() <= 1, "got {v}");
    }

    #[test]
    fn ragged_input_fails_loud() {
        // 3 bytes is not a whole s16 stereo frame (4 bytes).
        assert_eq!(
            convert_pcm(
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
        let out = convert_pcm(
            &bytes,
            AudioFormat::PcmF32Le,
            input.len() as u8,
            AudioFormat::PcmF32Le,
            out_ch,
        )
        .unwrap();
        out.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
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
