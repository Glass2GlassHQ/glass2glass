//! Software PCM sample-rate converter (the last Tier-1 audio transform, the
//! resampler `AudioConvert` deliberately left out). Converts interleaved PCM
//! (`PcmS16Le` / `PcmF32Le`) from its input rate to a configured target rate,
//! preserving sample format and channel count, so audio chains can bridge a
//! rate mismatch: `WasapiSrc (44.1 kHz) -> AudioResample (48 kHz) -> AacEncode`.
//!
//! Algorithm: per-channel linear interpolation. A resampler is inherently
//! stateful, the output sample grid does not align to buffer boundaries, so the
//! element carries the last input sample of each channel and a fractional read
//! position (`phase`) across `process` calls. Linear interpolation is the cheap,
//! `no_std`, allocation-free-per-sample baseline (a windowed-sinc / polyphase
//! filter is a quality follow-up); it is exact at rate 1:1 and introduces the
//! usual linear-interp high-frequency rolloff otherwise. CPU-only and `no_std`:
//! this element lives in the crate baseline alongside `AudioConvert`.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec,
};

use crate::audioconvert::{read_sample, sample_bytes, write_sample, PCM_FORMATS};

/// `f64::floor` without `std` / libm: truncation rounds toward zero, so a
/// negative non-integer needs one subtracted. `rel` lives in a small range
/// (roughly `[-1, buffer_len)`), well within `isize`.
fn floor_isize(x: f64) -> isize {
    let truncated = x as isize;
    if x < 0.0 && (truncated as f64) != x {
        truncated - 1
    } else {
        truncated
    }
}

#[derive(Debug)]
pub struct AudioResample {
    target_rate: u32,
    /// Input format/channels/rate of the configured stream, updated by a
    /// mid-stream `CapsChanged`.
    input: Option<(AudioFormat, u8, u32)>,
    /// Per-channel last input sample carried from the previous buffer, so an
    /// output sample whose read position falls before the current buffer's
    /// first sample interpolates against the real predecessor. `None` until the
    /// first buffer is seen (or after a flush / rate change).
    prev: Option<Vec<f32>>,
    /// Read position of the next output sample relative to the current buffer's
    /// sample 0, in input samples. Negative means "between `prev` and sample 0".
    phase: f64,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl AudioResample {
    pub fn new(target_rate: u32) -> Self {
        assert!(target_rate > 0, "target sample rate must be non-zero");
        Self {
            target_rate,
            input: None,
            prev: None,
            phase: 0.0,
            configured: false,
            last_caps: None,
            emitted: 0,
        }
    }

    pub fn target_rate(&self) -> u32 {
        self.target_rate
    }

    /// Validate a PCM caps as a resamplable input, returning its
    /// format/channels/rate.
    fn accept_input(&self, caps: &Caps) -> Result<(AudioFormat, u8, u32), G2gError> {
        let Caps::Audio {
            format,
            channels,
            sample_rate,
        } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !PCM_FORMATS.contains(format) || *channels == 0 || *sample_rate == 0 {
            return Err(G2gError::CapsMismatch);
        }
        Ok((*format, *channels, *sample_rate))
    }

    /// Reset the streaming state (on flush or a rate/format change), so the next
    /// buffer restarts the interpolation grid without carrying a stale sample.
    fn reset_state(&mut self) {
        self.prev = None;
        self.phase = 0.0;
    }
}

impl AsyncElement for AudioResample {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        self.accept_input(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    /// Native `DerivedOutput`: a supported PCM input maps to the same format +
    /// channels at the target sample rate.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let target_rate = self.target_rate;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::Audio {
                format,
                channels,
                sample_rate: _,
            } if PCM_FORMATS.contains(format) && *channels > 0 => CapsSet::one(Caps::Audio {
                format: *format,
                channels: *channels,
                sample_rate: target_rate,
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (format, channels, rate) = self.accept_input(absolute_caps)?;
        self.input = Some((format, channels, rate));
        self.reset_state();
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
                    let (in_format, in_channels, in_rate) =
                        self.input.ok_or(G2gError::NotConfigured)?;
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let resampled = self.resample(slice.as_slice(), in_format, in_channels, in_rate)?;

                    let new_caps = Caps::Audio {
                        format: in_format,
                        channels: in_channels,
                        sample_rate: self.target_rate,
                    };
                    if self.last_caps.as_ref() != Some(&new_caps) {
                        out.push(PipelinePacket::CapsChanged(new_caps.clone())).await?;
                        self.last_caps = Some(new_caps);
                    }
                    let out_frame = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(resampled)),
                        timing: frame.timing,
                        sequence: self.emitted,
                        meta: Default::default(),
                    };
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    let (format, channels, rate) = self.accept_input(&c)?;
                    self.input = Some((format, channels, rate));
                    // A rate / format change invalidates the carried sample.
                    self.reset_state();
                }
                PipelinePacket::Flush => {
                    self.reset_state();
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                // Segment is control: forward unchanged.
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                // the runner forwards Eos; the transform does not re-emit it.
                PipelinePacket::Eos => {}
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        AUDIORESAMPLE_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "samplerate" => {
                self.target_rate = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "samplerate" => Some(PropValue::Uint(self.target_rate as u64)),
            _ => None,
        }
    }
}

/// `AudioResample`'s settable properties (M107): the output sample rate.
static AUDIORESAMPLE_PROPS: &[PropertySpec] =
    &[PropertySpec::new("samplerate", PropKind::Uint, "output samples per second")];

impl AudioResample {
    /// Resample one interleaved PCM buffer from `in_rate` to `self.target_rate`,
    /// advancing the per-channel carry + fractional phase. At rate 1:1 the
    /// output equals the input (phase stays integral, interpolation is exact).
    fn resample(
        &mut self,
        src: &[u8],
        in_format: AudioFormat,
        in_channels: u8,
        in_rate: u32,
    ) -> Result<Box<[u8]>, G2gError> {
        let bytes = sample_bytes(in_format);
        let ch = in_channels as usize;
        let in_frame = bytes * ch;
        if in_frame == 0 || src.len() % in_frame != 0 {
            return Err(G2gError::CapsMismatch);
        }
        let n = src.len() / in_frame;
        if n == 0 {
            return Ok(Vec::new().into_boxed_slice());
        }

        // Decode the buffer to per-channel f32 (channels-major), so the inner
        // interpolation loop is index math, not byte decoding.
        let mut planes: Vec<Vec<f32>> = alloc::vec![Vec::with_capacity(n); ch];
        for f in 0..n {
            let base = f * in_frame;
            for (c, plane) in planes.iter_mut().enumerate() {
                plane.push(read_sample(&src[base + c * bytes..], in_format));
            }
        }

        // input samples advanced per output sample.
        let step = in_rate as f64 / self.target_rate as f64;
        // `phase` is the read position relative to this buffer's sample 0; it is
        // carried across calls (typically negative, pointing into `prev`).
        let mut rel = self.phase;
        let prev = self.prev.as_ref();

        // sample(c, i): input value at integer index `i` relative to this
        // buffer. i == -1 is the carried predecessor; 0..n-1 is this buffer.
        let sample = |planes: &[Vec<f32>], c: usize, i: isize| -> f32 {
            if i < 0 {
                // before the buffer: the carried sample, or sample 0 if this is
                // the first buffer (phase starts at 0, so this only fires when a
                // predecessor exists).
                prev.map(|p| p[c]).unwrap_or(planes[c][0])
            } else {
                planes[c][i as usize]
            }
        };

        let mut dst = Vec::new();
        // Produce while both interpolation endpoints (floor(rel), floor(rel)+1)
        // are available, i.e. floor(rel)+1 <= n-1  =>  rel < n-1.
        while rel < (n - 1) as f64 {
            let i = floor_isize(rel);
            let frac = (rel - i as f64) as f32;
            for c in 0..ch {
                let a = sample(&planes, c, i);
                let b = sample(&planes, c, i + 1);
                write_sample(&mut dst, a + (b - a) * frac, in_format);
            }
            rel += step;
        }

        // Carry: this buffer's last sample becomes the predecessor, and the read
        // position shifts to be relative to the next buffer's sample 0.
        let mut last = alloc::vec![0f32; ch];
        for (c, slot) in last.iter_mut().enumerate() {
            *slot = planes[c][n - 1];
        }
        self.prev = Some(last);
        // The next buffer's sample 0 is at absolute index `n`, so a read
        // position `rel` (relative to this buffer's sample 0) becomes `rel - n`
        // relative to the next buffer; the carried `prev` then sits at its
        // relative index -1, exactly where the boundary interpolation reads it.
        self.phase = rel - n as f64;
        Ok(dst.into_boxed_slice())
    }
}

impl PadTemplates for AudioResample {
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

    /// Build an interleaved f32 buffer from per-channel sample slices.
    fn interleave_f32(channels: &[&[f32]]) -> Vec<u8> {
        let n = channels[0].len();
        let mut v = Vec::new();
        for f in 0..n {
            for ch in channels {
                v.extend_from_slice(&ch[f].to_le_bytes());
            }
        }
        v
    }

    fn f32_samples(bytes: &[u8]) -> Vec<f32> {
        bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    #[test]
    fn derived_output_retargets_rate_only() {
        let r = AudioResample::new(48_000);
        let CapsConstraint::DerivedOutput(f) = r.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        // format + channels preserved, rate retargeted.
        let out = f(&audio(AudioFormat::PcmS16Le, 2, 44_100));
        assert_eq!(out.alternatives(), &[audio(AudioFormat::PcmS16Le, 2, 48_000)]);
        // compressed audio is not resamplable.
        assert!(f(&audio(AudioFormat::Aac, 2, 48_000)).is_empty());
    }

    #[test]
    fn identity_rate_passes_samples_through() {
        let mut r = AudioResample::new(48_000);
        r.configure_pipeline(&audio(AudioFormat::PcmF32Le, 1, 48_000)).unwrap();
        let src = interleave_f32(&[&[0.0, 0.25, 0.5, 0.75]]);
        let out = r.resample(&src, AudioFormat::PcmF32Le, 1, 48_000).unwrap();
        let got = f32_samples(&out);
        // 1:1 produces n-1 outputs from this buffer (the last sample is carried
        // to interpolate with the next buffer) and reproduces them exactly.
        assert_eq!(got, &[0.0, 0.25, 0.5]);
    }

    #[test]
    fn upsampling_2x_doubles_length_and_interpolates_midpoints() {
        let mut r = AudioResample::new(96_000);
        r.configure_pipeline(&audio(AudioFormat::PcmF32Le, 1, 48_000)).unwrap();
        // ramp 0,1,2,3; upsample 2x -> step 0.5 -> 0,0.5,1,1.5,2,2.5 (stops
        // before the last sample, which is carried).
        let src = interleave_f32(&[&[0.0, 1.0, 2.0, 3.0]]);
        let out = r.resample(&src, AudioFormat::PcmF32Le, 1, 48_000).unwrap();
        let got = f32_samples(&out);
        assert_eq!(got.len(), 6, "2x upsample of 4 samples yields ~2*(n-1) outputs");
        let want = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-5, "got {g} want {w}");
        }
    }

    #[test]
    fn downsampling_halves_length() {
        let mut r = AudioResample::new(24_000);
        r.configure_pipeline(&audio(AudioFormat::PcmF32Le, 1, 48_000)).unwrap();
        // step 2.0 over indices 0..7 -> reads at 0,2,4,6 (the loop runs while
        // rel < n-1 = 7, so rel=6 still interpolates 6..7).
        let src = interleave_f32(&[&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]);
        let out = r.resample(&src, AudioFormat::PcmF32Le, 1, 48_000).unwrap();
        let got = f32_samples(&out);
        assert_eq!(got, &[0.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn phase_carries_across_buffers() {
        // Upsample 2x across two buffers; the interpolation grid must continue
        // seamlessly, using the carried last sample of buffer 1 to interpolate
        // the boundary value (3 -> 4 midpoint = 3.5).
        let mut r = AudioResample::new(96_000);
        r.configure_pipeline(&audio(AudioFormat::PcmF32Le, 1, 48_000)).unwrap();
        let b1 = interleave_f32(&[&[0.0, 1.0, 2.0, 3.0]]);
        let b2 = interleave_f32(&[&[4.0, 5.0, 6.0, 7.0]]);
        let o1 = f32_samples(&r.resample(&b1, AudioFormat::PcmF32Le, 1, 48_000).unwrap());
        let o2 = f32_samples(&r.resample(&b2, AudioFormat::PcmF32Le, 1, 48_000).unwrap());
        assert_eq!(o1, &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
        // buffer 2 resumes at read position 3.0 (carried): 3,3.5,4,4.5,5,5.5,6.5?
        // grid: 3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5 stopping before last (index 7).
        assert_eq!(o2.first().copied(), Some(3.0), "resumes exactly where it left off");
        assert!((o2[1] - 3.5).abs() < 1e-5, "boundary midpoint uses carried sample");
    }

    #[test]
    fn ragged_input_fails_loud() {
        let mut r = AudioResample::new(48_000);
        r.configure_pipeline(&audio(AudioFormat::PcmS16Le, 2, 44_100)).unwrap();
        // 3 bytes is not a whole s16 stereo frame (4 bytes).
        assert_eq!(
            r.resample(&[0, 0, 0], AudioFormat::PcmS16Le, 2, 44_100),
            Err(G2gError::CapsMismatch)
        );
    }
}
