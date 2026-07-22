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
//!
//! The per-buffer loop defers the last input sample into the carry, so at end
//! of stream `process(Eos)` flushes the pending output positions in the final
//! window, interpolating toward a held last sample (sample-and-hold), landing
//! the total output at the rate-ratio count `ceil(n_in * out/in)`.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PassthroughFields,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, ANY_SAMPLE_RATE,
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
    /// Target output rate from the `samplerate` property. `0` means "auto": take
    /// the output rate from the negotiated caps (a downstream capsfilter), the
    /// gst caps-driven idiom (M187).
    target_rate: u32,
    /// Input format/channels/rate of the configured stream, updated by a
    /// mid-stream `CapsChanged`.
    input: Option<(AudioFormat, u8, u32)>,
    /// Output rate resolved from the negotiated output caps (M187), set by
    /// `configure_output`. Used in auto mode; `None` until then so `process`
    /// falls back to the property and runners that don't deliver output caps
    /// keep the property-driven behavior.
    resolved: Option<u32>,
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
    /// Timing of the last input `DataFrame`, reused to stamp the EOS tail frame
    /// (its exact timestamp is not critical, the tail is a fraction of a buffer).
    last_timing: FrameTiming,
}

impl AudioResample {
    pub fn new(target_rate: u32) -> Self {
        assert!(target_rate > 0, "target sample rate must be non-zero");
        Self {
            target_rate,
            input: None,
            resolved: None,
            prev: None,
            phase: 0.0,
            configured: false,
            last_caps: None,
            emitted: 0,
            last_timing: FrameTiming::default(),
        }
    }

    /// Caps-driven (M187): take the output rate from the negotiated caps (a
    /// downstream capsfilter). With no downstream constraint it defaults to
    /// passthrough (no resampling).
    pub fn auto() -> Self {
        Self {
            target_rate: 0,
            input: None,
            resolved: None,
            prev: None,
            phase: 0.0,
            configured: false,
            last_caps: None,
            emitted: 0,
            last_timing: FrameTiming::default(),
        }
    }

    pub fn target_rate(&self) -> u32 {
        self.target_rate
    }

    /// The effective output rate: the property when set, else the caps-resolved
    /// rate (auto). `None` for an unconfigured auto instance.
    fn out_rate(&self) -> Option<u32> {
        if self.target_rate != 0 {
            Some(self.target_rate)
        } else {
            self.resolved
        }
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
        // A 0 rate (`ANY_SAMPLE_RATE`) or 0 channel count (`ANY_CHANNELS`) is the
        // negotiation placeholder a decoder advertises before it has decoded a
        // frame; accept both deferred (the real values arrive as a `CapsChanged`,
        // which the runner turns into a fresh `configure_pipeline`, and channels is
        // a passthrough field so a downstream capsfilter pins it). A `DataFrame`
        // never precedes that `CapsChanged`, so `resample` never interpolates at a
        // placeholder rate / channel count (guarded in `process` / `resample`).
        if !PCM_FORMATS.contains(format) {
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
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        self.accept_input(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    /// Native `DerivedOutput`: a supported PCM input maps to the same format +
    /// channels at the target sample rate.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        // Property, or the caps-resolved target from startup (M755). Reflecting the
        // resolved rate (not just the property) lets a mid-stream input-rate change
        // re-derive to a single fixed output, so the element keeps its 48 kHz target
        // even when its downstream feasibility snapshot is blanked by an intervening
        // format converter (`audioresample ! audioconvert ! rate-pin`, where the
        // converter retargets `format`, a scalar with no wildcard, so the backward
        // feasibility projection is empty). Auto + unresolved (startup) stays the
        // passthrough+wildcard set, so startup negotiation is unchanged.
        let out_rate = self.out_rate();
        // Passthrough format + channels (retarget sample_rate only).
        let passthrough = PassthroughFields::NONE.with_format().with_channels();
        let derive = Box::new(move |input: &Caps| match input {
            // `channels` passes through untouched, so an `ANY_CHANNELS` (0)
            // placeholder input derives an `ANY_CHANNELS` output (a downstream
            // capsfilter pins it); do not require a concrete count here, else a
            // decoder's pre-decode placeholder collapses the derived set to empty
            // and the solver reads it as an unsatisfiable link.
            Caps::Audio {
                format,
                channels,
                sample_rate,
            } if PCM_FORMATS.contains(format) => {
                let mk = |rate| Caps::Audio {
                    format: *format,
                    channels: *channels,
                    sample_rate: rate,
                };
                match out_rate {
                    // Property-driven, or a caps-resolved target: the fixed rate.
                    Some(rate) => CapsSet::one(mk(rate)),
                    // Caps-driven (auto), not yet resolved: default to passthrough
                    // (the input rate, no resampling), but advertise "any rate" so a
                    // downstream capsfilter pins the target. Passthrough is the
                    // preferred (first) alternative.
                    None => CapsSet::from_alternatives(alloc::vec![
                        mk(*sample_rate),
                        mk(ANY_SAMPLE_RATE)
                    ]),
                }
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
        self.reset_state();
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    /// M187: take the output rate from the negotiated output caps when the
    /// `samplerate` property is unset (caps-driven). The rate is already
    /// fixated, so it is concrete (non-zero).
    fn configure_output(&mut self, output_caps: &Caps) -> Result<(), G2gError> {
        let Caps::Audio { sample_rate, .. } = output_caps else {
            return Err(G2gError::CapsMismatch);
        };
        if *sample_rate == ANY_SAMPLE_RATE {
            return Err(G2gError::CapsMismatch);
        }
        self.resolved = Some(*sample_rate);
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
                    let (in_format, in_channels, in_rate) =
                        self.input.ok_or(G2gError::NotConfigured)?;
                    // The deferred `ANY_SAMPLE_RATE` placeholder must have been
                    // resolved by a real input `CapsChanged` before any data; if not,
                    // fail loud rather than divide by a zero rate.
                    if in_rate == 0 {
                        return Err(G2gError::NotConfigured);
                    }
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.last_timing = frame.timing;
                    // Effective output rate: property, or caps-resolved (auto).
                    let out_rate = self.out_rate().ok_or(G2gError::NotConfigured)?;
                    let resampled =
                        self.resample(slice, in_format, in_channels, in_rate, out_rate)?;

                    let new_caps = Caps::Audio {
                        format: in_format,
                        channels: in_channels,
                        sample_rate: out_rate,
                    };
                    if self.last_caps.as_ref() != Some(&new_caps) {
                        out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                            .await?;
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
                    // The runner's transform arm calls `configure_pipeline` (input)
                    // then `configure_output` (output) immediately before pushing
                    // this packet, whose caps `c` is the arm's pre-fixed forward
                    // *output*, not a new input. `configure_pipeline` already set
                    // the input and reset the resampler state, so just forward the
                    // output caps and record `last_caps`. Do NOT `accept_input`
                    // here: `c` is our output, and adopting it as the input
                    // corrupts the next frame (the stacked-transform bug; see
                    // videoconvert.rs).
                    out.push(PipelinePacket::CapsChanged(c.clone())).await?;
                    self.last_caps = Some(c);
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
                // The runner's transform arm calls `process(Eos)` before it
                // forwards Eos downstream, so the flushed tail frame lands ahead
                // of Eos. Flush the pending final-window output, then let the
                // runner emit Eos (do not re-emit it here).
                PipelinePacket::Eos => {
                    if let Some(tail) = self.flush_tail()? {
                        let out_frame = Frame {
                            domain: MemoryDomain::System(SystemSlice::from_boxed(tail)),
                            timing: self.last_timing,
                            sequence: self.emitted,
                            meta: Default::default(),
                        };
                        self.emitted += 1;
                        out.push(PipelinePacket::DataFrame(out_frame)).await?;
                    }
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        AUDIORESAMPLE_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Audio resampler",
            "Filter/Converter/Audio",
            "Resamples raw audio to a different sample rate",
            "g2g",
        )
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
            "samplerate" => self.out_rate().map(|r| PropValue::Uint(r as u64)),
            _ => None,
        }
    }
}

/// `AudioResample`'s settable properties (M107): the output sample rate.
static AUDIORESAMPLE_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "samplerate",
    PropKind::Uint,
    "output samples per second",
)];

impl AudioResample {
    /// Resample one interleaved PCM buffer from `in_rate` to `out_rate`,
    /// advancing the per-channel carry + fractional phase. Rate 1:1 short-circuits
    /// to a byte-exact pass-through (no carry, no interpolation).
    fn resample(
        &mut self,
        src: &[u8],
        in_format: AudioFormat,
        in_channels: u8,
        in_rate: u32,
        out_rate: u32,
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
        // Rate 1:1 is a byte-exact pass-through. The interpolation loop below
        // would defer each buffer's last sample into the carry, and the final
        // one is never flushed at end of stream (one sample lost per stream,
        // caught by calliope's opus differential). Skip the loop entirely; the
        // carry state stays reset (a mid-stream rate change reconfigures first).
        if in_rate == out_rate {
            return Ok(src.to_vec().into_boxed_slice());
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
        let step = in_rate as f64 / out_rate as f64;
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

    /// At end of stream, emit the output samples the per-buffer loop deferred:
    /// the read positions in the final input window `[n-1, n)` that `resample`
    /// left pending (it stops at `rel < n-1` and carries the buffer's last
    /// sample into `prev`). Interpolates toward a held last sample (b = a,
    /// sample-and-hold), so the total stream output lands at `ceil(n_in *
    /// out/in)`. Returns `None` when there is no carry: rate 1:1 (bypass) or a
    /// stream that ended before any resampled frame.
    fn flush_tail(&mut self) -> Result<Option<Box<[u8]>>, G2gError> {
        let Some(prev) = self.prev.take() else {
            return Ok(None);
        };
        let (in_format, in_channels, in_rate) = self.input.ok_or(G2gError::NotConfigured)?;
        let out_rate = self.out_rate().ok_or(G2gError::NotConfigured)?;
        // 1:1 bypasses and never populates the carry; guard against a zero rate.
        if in_rate == out_rate || in_rate == 0 || out_rate == 0 {
            return Ok(None);
        }
        let ch = in_channels as usize;
        if prev.len() != ch {
            return Err(G2gError::CapsMismatch);
        }
        let step = in_rate as f64 / out_rate as f64;
        // After the last buffer `phase` is `rel - n` relative to that buffer's
        // sample 0, so the carried `prev` sits at relative index -1 and the
        // pending window is `phase in [-1, 0)`. Emit while `phase < 0`, holding
        // `prev` as both interpolation endpoints (the signal past the last
        // sample is held constant).
        let mut rel = self.phase;
        let mut dst = Vec::new();
        while rel < 0.0 {
            for &sample in &prev {
                write_sample(&mut dst, sample, in_format);
            }
            rel += step;
        }
        self.phase = 0.0;
        if dst.is_empty() {
            return Ok(None);
        }
        Ok(Some(dst.into_boxed_slice()))
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
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn derived_output_retargets_rate_only() {
        let r = AudioResample::new(48_000);
        let CapsConstraint::DerivedCoupled {
            derive: f,
            passthrough,
        } = r.caps_constraint_as_transform()
        else {
            panic!("expected DerivedCoupled");
        };
        assert_eq!(
            passthrough,
            PassthroughFields::NONE.with_format().with_channels()
        );
        // format + channels preserved, rate retargeted.
        let out = f(&audio(AudioFormat::PcmS16Le, 2, 44_100));
        assert_eq!(
            out.alternatives(),
            &[audio(AudioFormat::PcmS16Le, 2, 48_000)]
        );
        // compressed audio is not resamplable.
        assert!(f(&audio(AudioFormat::Aac, 2, 48_000)).is_empty());
    }

    #[test]
    fn identity_rate_passes_samples_through() {
        let mut r = AudioResample::new(48_000);
        r.configure_pipeline(&audio(AudioFormat::PcmF32Le, 1, 48_000))
            .unwrap();
        let src = interleave_f32(&[&[0.0, 0.25, 0.5, 0.75]]);
        let out = r
            .resample(&src, AudioFormat::PcmF32Le, 1, 48_000, 48_000)
            .unwrap();
        let got = f32_samples(&out);
        // 1:1 is a byte-exact pass-through: every sample, including the last
        // (the old carry deferred it and lost the stream's final sample at EOS).
        assert_eq!(got, &[0.0, 0.25, 0.5, 0.75]);
    }

    #[test]
    fn upsampling_2x_doubles_length_and_interpolates_midpoints() {
        let mut r = AudioResample::new(96_000);
        r.configure_pipeline(&audio(AudioFormat::PcmF32Le, 1, 48_000))
            .unwrap();
        // ramp 0,1,2,3; upsample 2x -> step 0.5 -> 0,0.5,1,1.5,2,2.5 (stops
        // before the last sample, which is carried).
        let src = interleave_f32(&[&[0.0, 1.0, 2.0, 3.0]]);
        let out = r
            .resample(&src, AudioFormat::PcmF32Le, 1, 48_000, 96_000)
            .unwrap();
        let got = f32_samples(&out);
        assert_eq!(
            got.len(),
            6,
            "2x upsample of 4 samples yields ~2*(n-1) outputs"
        );
        let want = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-5, "got {g} want {w}");
        }
    }

    #[test]
    fn downsampling_halves_length() {
        let mut r = AudioResample::new(24_000);
        r.configure_pipeline(&audio(AudioFormat::PcmF32Le, 1, 48_000))
            .unwrap();
        // step 2.0 over indices 0..7 -> reads at 0,2,4,6 (the loop runs while
        // rel < n-1 = 7, so rel=6 still interpolates 6..7).
        let src = interleave_f32(&[&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]);
        let out = r
            .resample(&src, AudioFormat::PcmF32Le, 1, 48_000, 24_000)
            .unwrap();
        let got = f32_samples(&out);
        assert_eq!(got, &[0.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn eos_flush_emits_deferred_tail() {
        // 2x upsample of 4 samples: the per-buffer loop yields 6 (deferring the
        // last window), then the EOS flush emits the held tail so the total lands
        // at round(4*2) = 8 = ceil(4 / 0.5).
        let mut r = AudioResample::new(96_000);
        r.configure_pipeline(&audio(AudioFormat::PcmF32Le, 1, 48_000))
            .unwrap();
        let src = interleave_f32(&[&[0.0, 1.0, 2.0, 3.0]]);
        let body = f32_samples(
            &r.resample(&src, AudioFormat::PcmF32Le, 1, 48_000, 96_000)
                .unwrap(),
        );
        assert_eq!(body.len(), 6);
        let tail = f32_samples(&r.flush_tail().unwrap().expect("tail emitted"));
        // held last sample (3.0) fills positions 3.0 and 3.5.
        assert_eq!(tail, &[3.0, 3.0]);
        assert_eq!(body.len() + tail.len(), 8);
        // flush is idempotent: the carry is consumed, a second flush emits nothing.
        assert!(r.flush_tail().unwrap().is_none());
    }

    #[test]
    fn eos_flush_without_data_emits_nothing() {
        // A stream that ends before any DataFrame has no carry to flush.
        let mut r = AudioResample::new(48_000);
        r.configure_pipeline(&audio(AudioFormat::PcmF32Le, 1, 44_100))
            .unwrap();
        assert!(r.flush_tail().unwrap().is_none());
    }

    #[test]
    fn phase_carries_across_buffers() {
        // Upsample 2x across two buffers; the interpolation grid must continue
        // seamlessly, using the carried last sample of buffer 1 to interpolate
        // the boundary value (3 -> 4 midpoint = 3.5).
        let mut r = AudioResample::new(96_000);
        r.configure_pipeline(&audio(AudioFormat::PcmF32Le, 1, 48_000))
            .unwrap();
        let b1 = interleave_f32(&[&[0.0, 1.0, 2.0, 3.0]]);
        let b2 = interleave_f32(&[&[4.0, 5.0, 6.0, 7.0]]);
        let o1 = f32_samples(
            &r.resample(&b1, AudioFormat::PcmF32Le, 1, 48_000, 96_000)
                .unwrap(),
        );
        let o2 = f32_samples(
            &r.resample(&b2, AudioFormat::PcmF32Le, 1, 48_000, 96_000)
                .unwrap(),
        );
        assert_eq!(o1, &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
        // buffer 2 resumes at read position 3.0 (carried): 3,3.5,4,4.5,5,5.5,6.5?
        // grid: 3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5 stopping before last (index 7).
        assert_eq!(
            o2.first().copied(),
            Some(3.0),
            "resumes exactly where it left off"
        );
        assert!(
            (o2[1] - 3.5).abs() < 1e-5,
            "boundary midpoint uses carried sample"
        );
    }

    #[test]
    fn ragged_input_fails_loud() {
        let mut r = AudioResample::new(48_000);
        r.configure_pipeline(&audio(AudioFormat::PcmS16Le, 2, 44_100))
            .unwrap();
        // 3 bytes is not a whole s16 stereo frame (4 bytes).
        assert_eq!(
            r.resample(&[0, 0, 0], AudioFormat::PcmS16Le, 2, 44_100, 48_000),
            Err(G2gError::CapsMismatch)
        );
    }
}
