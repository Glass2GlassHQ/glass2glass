//! Software PCM converter (M34), the audio analog of `VideoConvert`. Converts
//! interleaved PCM between sample formats (`PcmS16Le` <-> `PcmF32Le`) and
//! between channel counts (mono <-> multi-channel) at the same sample rate, so
//! audio chains compose across format boundaries: `WasapiSrc (F32, 2ch) ->
//! AudioConvert -> WavSink (S16)`, or feeding an encoder that wants a specific
//! layout.
//!
//! Channel conversion handles any count to any count: identity, mono fan-out to
//! N channels (replicate), downmix to mono (average), and a layout-agnostic
//! round-robin fold/replicate for the mixed multi-channel cases (e.g. 5.1 ->
//! stereo). The fold is position-unaware (we don't track speaker layout), so it
//! never silently drops a channel rather than applying ITU downmix coefficients.
//! Sample rate is preserved (no resampler). CPU-only and `no_std`: this element
//! lives in the crate baseline.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PassthroughFields,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, ANY_CHANNELS,
};

/// The PCM sample formats this element reads and writes.
const FORMATS: [AudioFormat; 2] = [AudioFormat::PcmS16Le, AudioFormat::PcmF32Le];

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
            FORMATS.contains(&target_format),
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
        if !FORMATS.contains(format) {
            return Err(G2gError::CapsMismatch);
        }
        Ok((*format, *channels, *sample_rate))
    }
}

pub(crate) fn sample_bytes(format: AudioFormat) -> usize {
    match format {
        AudioFormat::PcmS16Le => 2,
        AudioFormat::PcmF32Le => 4,
        // not reachable: only FORMATS pass negotiation.
        _ => 0,
    }
}

/// The PCM sample formats `AudioConvert` / `AudioResample` read and write.
pub(crate) const PCM_FORMATS: [AudioFormat; 2] = [AudioFormat::PcmS16Le, AudioFormat::PcmF32Le];

impl AsyncElement for AudioConvert {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        self.accept_input(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    /// Native `DerivedCoupled`: a supported PCM input maps to the target format +
    /// channel count at the same sample rate (rate is the one passthrough field).
    /// A fixed target emits that single output; a caps-driven (`auto`) target
    /// advertises the passthrough as the preferred alternative plus the retarget
    /// options (the other PCM format, and an `ANY_CHANNELS` wildcard) so a
    /// downstream capsfilter pins the real format / channel count.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let target_format = self.target_format;
        let target_channels = self.target_channels;
        // Only sample_rate is preserved; format + channels are retargeted.
        let passthrough = PassthroughFields::NONE.with_sample_rate();
        let derive = Box::new(move |input: &Caps| match input {
            Caps::Audio {
                format,
                channels,
                sample_rate,
            } if FORMATS.contains(format) => {
                // Candidate output formats: the fixed target, or (auto) the input
                // format first (passthrough) then the other PCM format.
                let formats: Vec<AudioFormat> = match target_format {
                    Some(f) => alloc::vec![f],
                    None => {
                        let mut v = alloc::vec![*format];
                        v.extend(FORMATS.iter().copied().filter(|f| f != format));
                        v
                    }
                };
                // Candidate channel counts: the fixed target, or (auto) the input
                // count (passthrough) then the `ANY_CHANNELS` wildcard. A `0`
                // (ANY_CHANNELS) input is the decoder's pre-decode placeholder:
                // advertise only the wildcard so a downstream capsfilter pins it
                // (else it fixates to stereo) and the real count flows in a
                // runtime `CapsChanged`.
                let chans: Vec<u8> = match (target_channels, *channels) {
                    (Some(c), _) => alloc::vec![c],
                    (None, ANY_CHANNELS) => alloc::vec![ANY_CHANNELS],
                    (None, c) => alloc::vec![c, ANY_CHANNELS],
                };
                let mut alts = Vec::new();
                for f in &formats {
                    for c in &chans {
                        alts.push(Caps::Audio {
                            format: *f,
                            channels: *c,
                            sample_rate: *sample_rate,
                        });
                    }
                }
                CapsSet::from_alternatives(alts)
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        });
        CapsConstraint::DerivedCoupled {
            derive,
            passthrough,
        }
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
        if !FORMATS.contains(format) || *channels == ANY_CHANNELS {
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
        "output sample format: S16LE | F32LE",
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
        _ => None,
    }
}

/// The canonical (GStreamer) property string for an [`AudioFormat`].
pub(crate) fn audio_format_to_str(f: AudioFormat) -> &'static str {
    match f {
        AudioFormat::PcmS16Le => "S16LE",
        AudioFormat::PcmF32Le => "F32LE",
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
        let set = CapsSet::from_alternatives(FORMATS.map(pcm).to_vec());
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

    let mut dst = Vec::with_capacity(frames * out_bytes * out_ch);
    let mut in_samples = alloc::vec![0f32; in_ch];
    for f in 0..frames {
        let base = f * in_frame;
        for (c, slot) in in_samples.iter_mut().enumerate() {
            *slot = read_sample(&src[base + c * in_bytes..], in_format);
        }
        for oc in 0..out_ch {
            let v = map_channel(&in_samples, oc, out_ch);
            write_sample(&mut dst, v, out_format);
        }
    }
    Ok(dst.into_boxed_slice())
}

/// Output sample for channel `oc`, given the interleaved input frame. Covers the
/// full N -> M space, position-unaware: identity when counts match; mono fan-out
/// (replicate the one input); downmix to mono (average all inputs); a round-robin
/// fold for a general downmix (`out_ch` < `in_ch`, `out_ch` >= 2: output `oc`
/// averages inputs `oc, oc+out_ch, oc+2*out_ch, ...`, so no channel is dropped);
/// and a round-robin replicate for upmix (`out_ch` > `in_ch`, `in_ch` >= 2).
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

/// Decode one sample to f32 in [-1, 1). The slice starts at the sample.
pub(crate) fn read_sample(at: &[u8], format: AudioFormat) -> f32 {
    match format {
        AudioFormat::PcmS16Le => {
            let s = i16::from_le_bytes([at[0], at[1]]);
            s as f32 / 32768.0
        }
        AudioFormat::PcmF32Le => f32::from_le_bytes([at[0], at[1], at[2], at[3]]),
        _ => 0.0,
    }
}

/// Encode one f32 sample, appending its little-endian bytes.
pub(crate) fn write_sample(dst: &mut Vec<u8>, v: f32, format: AudioFormat) {
    match format {
        AudioFormat::PcmS16Le => {
            let scaled = v.clamp(-1.0, 1.0) * 32767.0;
            // round half away from zero without libm.
            let rounded = if scaled >= 0.0 {
                scaled + 0.5
            } else {
                scaled - 0.5
            };
            dst.extend_from_slice(&(rounded as i16).to_le_bytes());
        }
        AudioFormat::PcmF32Le => dst.extend_from_slice(&v.to_le_bytes()),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let CapsConstraint::DerivedCoupled {
            derive: f,
            passthrough,
        } = conv.caps_constraint_as_transform()
        else {
            panic!("expected DerivedCoupled");
        };
        // only sample_rate is preserved; format + channels are retargeted.
        assert_eq!(passthrough, PassthroughFields::NONE.with_sample_rate());
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
        let CapsConstraint::DerivedCoupled { derive: f, .. } = conv.caps_constraint_as_transform()
        else {
            panic!("expected DerivedCoupled");
        };
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

    #[test]
    fn six_channel_downmixes_to_stereo_round_robin() {
        // 5.1 frame ch0..ch5 = 0,100,200,300,400,500 (s16). Round-robin fold:
        // L = avg(ch0, ch2, ch4) = 200; R = avg(ch1, ch3, ch5) = 300.
        let mut six = Vec::new();
        for v in [0i16, 100, 200, 300, 400, 500] {
            six.extend_from_slice(&v.to_le_bytes());
        }
        let stereo = convert_pcm(&six, AudioFormat::PcmS16Le, 6, AudioFormat::PcmS16Le, 2).unwrap();
        assert_eq!(stereo.len(), 4);
        let l = i16::from_le_bytes([stereo[0], stereo[1]]);
        let r = i16::from_le_bytes([stereo[2], stereo[3]]);
        assert!((l - 200).abs() <= 1, "L={l}");
        assert!((r - 300).abs() <= 1, "R={r}");
    }

    #[test]
    fn stereo_upmixes_to_six_round_robin() {
        // L=1000, R=2000 -> six channels replicate round-robin: 1000,2000,1000,...
        let mut stereo = Vec::new();
        stereo.extend_from_slice(&1000i16.to_le_bytes());
        stereo.extend_from_slice(&2000i16.to_le_bytes());
        let six = convert_pcm(&stereo, AudioFormat::PcmS16Le, 2, AudioFormat::PcmS16Le, 6).unwrap();
        assert_eq!(six.len(), 12);
        for (i, chunk) in six.chunks_exact(2).enumerate() {
            let want = if i % 2 == 0 { 1000 } else { 2000 };
            assert_eq!(i16::from_le_bytes([chunk[0], chunk[1]]), want, "ch{i}");
        }
    }
}
