//! Matrix channel mixer (`audiomixmatrix`). Every output channel is a weighted
//! sum of every input channel, so one element downmixes 5.1 to stereo, folds
//! stereo to mono, or reorders channels. Preserves format and sample rate and
//! changes the channel count. CPU-only `no_std`.
//!
//! Matches GStreamer's `audiomixmatrix`: `out[o] = sum over i of in[i] *
//! matrix[o][i]`, row-major with one row per output channel.
//!
//! `PropKind` has no array kind, so the reference's nested GstValueArray
//! `matrix=<<1.0,0.0>,<0.0,1.0>>` is written here as a string: coefficients
//! separated by `,` and rows by `;`, `matrix="1.0,0.0;0.0,1.0"`. The row count
//! must be `out-channels` and every row must be `in-channels` long.
//!
//! In `first-channels` mode the matrix is a truncated identity, as in the
//! reference. The reference then drops the channel count from its caps and
//! lets negotiation pick it; g2g fixes the output count from `out-channels`
//! instead (0 keeps the input's count), because the solver needs a definite
//! output shape.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec,
};

use crate::audiofx;

/// The reference's channel bounds, and its "unset" default.
const CHANNELS_MAX: u64 = 64;
const CHANNELS_MIN_TEXT: &str = "0";
const CHANNELS_MAX_TEXT: &str = "64";
const DEFAULT_CHANNELS_TEXT: &str = "0";

/// Row separator of the `matrix` string form, one row per output channel.
const MATRIX_ROW_SEPARATOR: char = ';';

/// Whether the matrix is written down or derived from the channel counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixMatrixMode {
    /// `in-channels`, `out-channels` and `matrix` are all set by hand.
    Manual,
    /// The matrix is a truncated identity: output `n` takes input `n`.
    FirstChannels,
}

impl MixMatrixMode {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "manual" => Some(Self::Manual),
            "first-channels" => Some(Self::FirstChannels),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Manual => "manual",
            Self::FirstChannels => "first-channels",
        }
    }
}

/// The spellings and order `gst-inspect` prints for `mode`.
const MIX_MATRIX_MODE_VALUES: &str = "manual | first-channels";

/// Parse the `matrix` string form into `out_channels` rows of `in_channels`
/// gains. An empty string is an empty matrix.
fn parse_matrix(text: &str) -> Result<Vec<Vec<f64>>, PropError> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let mut rows = Vec::new();
    for row in text.split(MATRIX_ROW_SEPARATOR) {
        rows.push(audiofx::parse_coefficients(row)?);
    }
    Ok(rows)
}

/// A matrix back as the text [`parse_matrix`] reads.
fn format_matrix(rows: &[Vec<f64>]) -> String {
    let mut out = String::new();
    for (index, row) in rows.iter().enumerate() {
        if index > 0 {
            out.push(MATRIX_ROW_SEPARATOR);
        }
        out.push_str(&audiofx::format_coefficients(row));
    }
    out
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::audiomixmatrix::AudioMixMatrix;
///
/// // stereo down to mono, each side at half gain.
/// let downmix = AudioMixMatrix::new()
///     .with_in_channels(2)
///     .with_out_channels(1)
///     .with_matrix("0.5,0.5");
/// ```
#[derive(Debug)]
pub struct AudioMixMatrix {
    mode: MixMatrixMode,
    in_channels: u8,
    out_channels: u8,
    /// One row per output channel, each `in_channels` long. Empty until set.
    matrix: Vec<Vec<f64>>,
    format: AudioFormat,
    sample_rate: u32,
    /// The negotiated input shape, set by `configure_pipeline`.
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl Default for AudioMixMatrix {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioMixMatrix {
    pub fn new() -> Self {
        Self {
            mode: MixMatrixMode::Manual,
            in_channels: 0,
            out_channels: 0,
            matrix: Vec::new(),
            format: AudioFormat::PcmS16Le,
            sample_rate: 0,
            configured: false,
            last_caps: None,
            emitted: 0,
        }
    }

    pub fn with_mode(mut self, mode: MixMatrixMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_in_channels(mut self, channels: u8) -> Self {
        self.in_channels = channels;
        self
    }

    pub fn with_out_channels(mut self, channels: u8) -> Self {
        self.out_channels = channels;
        self
    }

    /// Set the matrix from the string form: rows of comma-separated gains,
    /// rows separated by `;`.
    pub fn with_matrix(mut self, matrix: &str) -> Self {
        self.matrix = parse_matrix(matrix).unwrap_or_default();
        self
    }

    /// Output channel count for an input of `in_channels`, or `None` when the
    /// settings do not describe a mix of that input.
    fn resolve_out_channels(&self, in_channels: u8) -> Option<u8> {
        match self.mode {
            MixMatrixMode::FirstChannels => Some(if self.out_channels == 0 {
                in_channels
            } else {
                self.out_channels
            }),
            MixMatrixMode::Manual => {
                if self.in_channels != in_channels || self.out_channels == 0 {
                    return None;
                }
                if self.matrix.len() != self.out_channels as usize {
                    return None;
                }
                if self
                    .matrix
                    .iter()
                    .any(|row| row.len() != in_channels as usize)
                {
                    return None;
                }
                Some(self.out_channels)
            }
        }
    }

    /// The gains actually applied to an input of `in_channels`: the written
    /// matrix in `manual` mode, a truncated identity in `first-channels`.
    fn gains(&self, in_channels: u8, out_channels: u8) -> Vec<Vec<f64>> {
        match self.mode {
            MixMatrixMode::Manual => self.matrix.clone(),
            MixMatrixMode::FirstChannels => (0..out_channels as usize)
                .map(|out| {
                    (0..in_channels as usize)
                        .map(|input| if input == out { 1.0 } else { 0.0 })
                        .collect()
                })
                .collect(),
        }
    }

    /// Input caps this element accepts, and the output caps it makes of them.
    fn derive(&self, input: &Caps) -> Result<(Caps, u8, u8), G2gError> {
        let (format, in_channels, sample_rate) = audiofx::accept_audio(input, None)?;
        let in_channels = u8::try_from(in_channels).map_err(|_| G2gError::CapsMismatch)?;
        let out_channels = self
            .resolve_out_channels(in_channels)
            .ok_or(G2gError::CapsMismatch)?;
        let output = Caps::Audio {
            format,
            channels: out_channels,
            sample_rate,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        Ok((output, in_channels, out_channels))
    }

    /// Mix one interleaved buffer, returning `out_channels`-wide frames.
    fn mix(&self, samples: &[f32], in_channels: u8, out_channels: u8) -> Vec<f32> {
        let gains = self.gains(in_channels, out_channels);
        let in_channels = in_channels as usize;
        let frames = samples.len() / in_channels.max(1);
        let mut out = vec![0.0f32; frames * out_channels as usize];
        for (index, frame) in samples.chunks_exact(in_channels).enumerate() {
            for (channel, row) in gains.iter().enumerate() {
                let mut sum = 0.0f64;
                for (input, gain) in row.iter().enumerate() {
                    sum += frame[input] as f64 * gain;
                }
                out[index * out_channels as usize + channel] = sum as f32;
            }
        }
        out
    }
}

impl AsyncElement for AudioMixMatrix {
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
        let (output, _, _) = self.derive(upstream_caps)?;
        Ok(output)
    }

    /// The channel count is the one field the mix changes; format and rate pass
    /// through.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match self.derive(input) {
            Ok((output, _, _)) => CapsSet::one(output),
            Err(_) => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (_, in_channels, _) = self.derive(absolute_caps)?;
        let (format, _, sample_rate) = audiofx::accept_audio(absolute_caps, None)?;
        self.format = format;
        self.sample_rate = sample_rate;
        self.in_channels = in_channels;
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
                    let input = Caps::Audio {
                        format: self.format,
                        channels: self.in_channels,
                        sample_rate: self.sample_rate,
                        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
                    };
                    let (output, in_channels, out_channels) = self.derive(&input)?;
                    let src = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let samples = audiofx::decode(src, self.format);
                    let mixed = self.mix(&samples, in_channels, out_channels);
                    let dst = audiofx::encode(&mixed, self.format);

                    if self.last_caps.as_ref() != Some(&output) {
                        out.push(PipelinePacket::CapsChanged(output.clone()))
                            .await?;
                        self.last_caps = Some(output);
                    }
                    let out_frame = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(dst)),
                        timing: frame.timing,
                        sequence: self.emitted,
                        meta: Default::default(),
                    };
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                // The runner's transform arm pre-fixes the forward output caps
                // and delivers them here, so `c` is this element's own output,
                // not a new input. Forward it and record it so the data path
                // does not emit it a second time; the input comes from
                // `configure_pipeline`.
                PipelinePacket::CapsChanged(c) => {
                    out.push(PipelinePacket::CapsChanged(c.clone())).await?;
                    self.last_caps = Some(c);
                }
                PipelinePacket::Flush => {
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        AUDIOMIXMATRIX_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Matrix audio mix",
            "Filter/Audio",
            "Mixes a number of input channels into a number of output channels according to a transformation matrix",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "mode" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.mode = MixMatrixMode::from_str(s).ok_or(PropError::Value)?;
            }
            "in-channels" => {
                let channels = value.as_uint().ok_or(PropError::Type)?;
                if channels > CHANNELS_MAX {
                    return Err(PropError::Value);
                }
                self.in_channels = channels as u8;
            }
            "out-channels" => {
                let channels = value.as_uint().ok_or(PropError::Type)?;
                if channels > CHANNELS_MAX {
                    return Err(PropError::Value);
                }
                self.out_channels = channels as u8;
            }
            "matrix" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.matrix = parse_matrix(s)?;
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "mode" => Some(PropValue::Str(self.mode.as_str().to_string())),
            "in-channels" => Some(PropValue::Uint(self.in_channels as u64)),
            "out-channels" => Some(PropValue::Uint(self.out_channels as u64)),
            "matrix" => Some(PropValue::Str(format_matrix(&self.matrix))),
            _ => None,
        }
    }
}

static AUDIOMIXMATRIX_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "mode",
        PropKind::Str,
        "whether the matrix is written down or a truncated identity",
    )
    .with_enum_values(MIX_MATRIX_MODE_VALUES)
    .with_default("manual"),
    PropertySpec::new(
        "in-channels",
        PropKind::Uint,
        "how many audio channels we have on the input side",
    )
    .with_range(CHANNELS_MIN_TEXT, CHANNELS_MAX_TEXT)
    .with_default(DEFAULT_CHANNELS_TEXT),
    PropertySpec::new(
        "out-channels",
        PropKind::Uint,
        "how many audio channels we have on the output side",
    )
    .with_range(CHANNELS_MIN_TEXT, CHANNELS_MAX_TEXT)
    .with_default(DEFAULT_CHANNELS_TEXT),
    PropertySpec::new(
        "matrix",
        PropKind::Str,
        "transformation matrix for input/output channels: rows of comma-separated gains, one row per output channel, rows separated by ';'",
    ),
];

impl PadTemplates for AudioMixMatrix {
    fn pad_templates() -> Vec<PadTemplate> {
        audiofx::default_pad_templates()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(channels: u8) -> Caps {
        Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }

    #[test]
    fn matrix_syntax_round_trips() {
        let mut element = AudioMixMatrix::new();
        element
            .set_property("matrix", PropValue::Str("0.5,0.5;1,0".into()))
            .unwrap();
        assert_eq!(
            element.get_property("matrix"),
            Some(PropValue::Str("0.5,0.5;1,0".into()))
        );
    }

    #[test]
    fn a_malformed_matrix_entry_is_rejected() {
        let mut element = AudioMixMatrix::new();
        assert_eq!(
            element
                .set_property("matrix", PropValue::Str("0.5,x".into()))
                .unwrap_err(),
            PropError::Value
        );
    }

    #[test]
    fn manual_mode_needs_the_matrix_to_match_the_channel_counts() {
        let mut element = AudioMixMatrix::new()
            .with_in_channels(2)
            .with_out_channels(1)
            .with_matrix("0.5,0.5");
        assert_eq!(element.derive(&caps(2)).unwrap().0, caps(1));
        // the input caps disagree with `in-channels`.
        assert!(element.derive(&caps(4)).is_err());
        // a row of the wrong width is not a mix of this input.
        element.matrix = vec![vec![0.5, 0.5, 0.5]];
        assert!(element.derive(&caps(2)).is_err());
    }

    #[test]
    fn first_channels_mode_truncates_to_the_identity() {
        let element = AudioMixMatrix::new()
            .with_mode(MixMatrixMode::FirstChannels)
            .with_out_channels(2);
        assert_eq!(element.derive(&caps(6)).unwrap().0, caps(2));
        let gains = element.gains(6, 2);
        assert_eq!(
            gains,
            vec![
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        );
    }

    #[test]
    fn first_channels_mode_keeps_the_input_count_when_out_channels_is_unset() {
        let element = AudioMixMatrix::new().with_mode(MixMatrixMode::FirstChannels);
        assert_eq!(element.derive(&caps(6)).unwrap().0, caps(6));
    }

    #[test]
    fn a_downmix_is_the_weighted_sum() {
        let element = AudioMixMatrix::new()
            .with_in_channels(2)
            .with_out_channels(1)
            .with_matrix("0.25,0.75");
        let mixed = element.mix(&[1.0, 0.0, 0.0, 1.0], 2, 1);
        assert_eq!(mixed, vec![0.25, 0.75]);
    }
}
