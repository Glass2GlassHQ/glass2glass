//! Software PCM sample-rate converter (the last Tier-1 audio transform, the
//! resampler `AudioConvert` deliberately left out). Converts interleaved PCM
//! (`PcmS16Le` / `PcmF32Le`) from its input rate to a configured target rate,
//! preserving sample format and channel count, so audio chains can bridge a
//! rate mismatch: `WasapiSrc (44.1 kHz) -> AudioResample (48 kHz) -> AacEncode`.
//!
//! Algorithm: per-channel interpolation on a fractional read grid. A resampler
//! is inherently stateful, the output sample grid does not align to buffer
//! boundaries, so the element carries a tail of input samples per channel and a
//! fractional read position (`phase`) across `process` calls. The `quality`
//! property picks the kernel: 0 is two-point linear, cheap and table-free but
//! with the usual high-frequency rolloff; 1..=10 is a windowed sinc whose tap
//! count grows with quality, read from an oversampled phase table. Rate 1:1 is
//! a byte-exact pass-through either way. CPU-only and `no_std`: this element
//! lives in the crate baseline alongside `AudioConvert`.
//!
//! An output sample can only be emitted once its whole kernel window has
//! arrived, so the per-buffer loop defers the last `taps / 2` input samples
//! into the carry. At end of stream `process(Eos)` holds the final sample for
//! that many positions (sample-and-hold) and flushes the pending output,
//! landing the total output at the rate-ratio count `ceil(n_in * out/in)`.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, AudioShape, Caps, CapsConstraint, CapsSet, CapsTransform,
    ConfigureOutcome, ElementMetadata, FieldTransform, G2gError, MemoryDomain, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
    ANY_SAMPLE_RATE,
};

use crate::audioconvert::{pcm_formats, read_sample, sample_bytes, write_sample};

/// `f64::floor` without `std` / libm: truncation rounds toward zero, so a
/// negative non-integer needs one subtracted. `rel` lives in a small range
/// (roughly `[-taps, buffer_len)`), well within `isize`.
fn floor_isize(x: f64) -> isize {
    let truncated = x as isize;
    if x < 0.0 && (truncated as f64) != x {
        truncated - 1
    } else {
        truncated
    }
}

/// gst audioresample's `quality` bounds and default, from `gst-inspect-1.0
/// audioresample` (Integer, range 0 - 10, default 4).
const MIN_QUALITY: i64 = 0;
const MAX_QUALITY: i64 = 10;
const DEFAULT_QUALITY: u8 = 4;
/// The same value as declared text, for `gst-inspect`.
const MIN_QUALITY_TEXT: &str = "0";
const MAX_QUALITY_TEXT: &str = "10";
const DEFAULT_QUALITY_TEXT: &str = "4";

/// Kernel width per `quality` level, indexed by quality. Level 0 is the
/// two-point linear path; the sinc levels grow to 64 taps, which at 48 kHz puts
/// the kernel's transition band inside the top octave. gst also maps quality to
/// filter length, but its table is Kaiser-parameterised and is not reproduced
/// here.
const QUALITY_TAPS: [usize; 11] = [2, 8, 12, 16, 32, 36, 40, 44, 48, 56, 64];

/// Fractional read positions the sinc table holds, over `[0, 1]`. A runtime
/// position lands between two of them and interpolates linearly, gst's
/// `sinc-filter-interpolation=linear`: at this spacing the residual is far
/// below the kernel's own stop band.
const SINC_PHASE_STEPS: usize = 256;

/// Blackman-Nuttall window over `t` in `[0, 1]`, its own coefficients. The
/// ~-98 dB side lobes keep the truncated sinc's stop band below the
/// interpolation error the kernel is built to remove.
fn blackman_nuttall(t: f64) -> f64 {
    const A0: f64 = 0.363_581_9;
    const A1: f64 = 0.489_177_5;
    const A2: f64 = 0.136_599_5;
    const A3: f64 = 0.010_641_1;
    let turn = core::f64::consts::TAU * t;
    A0 - A1 * crate::mathf::cos(turn) + A2 * crate::mathf::cos(2.0 * turn)
        - A3 * crate::mathf::cos(3.0 * turn)
}

/// Normalized sinc, `sin(pi x) / (pi x)`, with the removable singularity filled.
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let arg = core::f64::consts::PI * x;
    crate::mathf::sin(arg) / arg
}

/// An oversampled windowed-sinc interpolation table: one kernel per fractional
/// read position in [`SINC_PHASE_STEPS`] steps, each row normalized to unity
/// DC gain so a constant signal survives the resampler untouched.
#[derive(Debug)]
struct SincTable {
    /// Taps per kernel, always even: `taps / 2` input samples each side of the
    /// read position.
    taps: usize,
    /// `(SINC_PHASE_STEPS + 1) * taps` weights, phase-major.
    weights: Vec<f32>,
    /// The `(quality, in_rate, out_rate)` this table was built for, so a
    /// mid-stream rate or quality change rebuilds it.
    built_for: (u8, u32, u32),
}

impl SincTable {
    fn build(quality: u8, in_rate: u32, out_rate: u32) -> Self {
        let taps = QUALITY_TAPS[quality as usize];
        let half = taps / 2;
        // Downsampling has to band-limit to the *output* Nyquist, so the sinc
        // widens by the rate ratio; upsampling keeps the input Nyquist.
        let cutoff = (f64::from(out_rate) / f64::from(in_rate)).min(1.0);
        let mut weights = Vec::with_capacity((SINC_PHASE_STEPS + 1) * taps);
        for phase in 0..=SINC_PHASE_STEPS {
            let frac = phase as f64 / SINC_PHASE_STEPS as f64;
            let row = weights.len();
            for k in 0..taps {
                // tap k reads input index floor(rel) - half + 1 + k, so its
                // distance from the read position is:
                let x = (k as f64 - (half as f64 - 1.0)) - frac;
                // the window spans [-half, half] in the same units.
                weights.push(
                    (sinc(cutoff * x) * blackman_nuttall((x + half as f64) / taps as f64)) as f32,
                );
            }
            let sum: f32 = weights[row..].iter().sum();
            if sum != 0.0 {
                for w in &mut weights[row..] {
                    *w /= sum;
                }
            }
        }
        Self {
            taps,
            weights,
            built_for: (quality, in_rate, out_rate),
        }
    }

    /// The kernel for fractional position `frac` in `[0, 1)`, linearly blended
    /// between the two neighbouring table rows.
    fn tap_weights(&self, frac: f64, into: &mut Vec<f32>) {
        let scaled = frac * SINC_PHASE_STEPS as f64;
        let low = (scaled as usize).min(SINC_PHASE_STEPS - 1);
        let blend = (scaled - low as f64) as f32;
        let (a, b) = (low * self.taps, (low + 1) * self.taps);
        into.clear();
        for k in 0..self.taps {
            let lower = self.weights[a + k];
            into.push(lower + (self.weights[b + k] - lower) * blend);
        }
    }
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::audioresample::AudioResample;
///
/// // audioresample samplerate=48000
/// let resample = AudioResample::new(48_000);
/// ```
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
    /// Per-channel tail of the input stream the next output sample's kernel
    /// window still reaches back into, carried from the previous buffer. Empty
    /// until the first buffer is seen (or after a flush / rate change).
    carry: Vec<Vec<f32>>,
    /// Read position of the next output sample relative to `carry`'s sample 0,
    /// in input samples.
    phase: f64,
    /// Interpolation quality, 0 (linear) to [`MAX_QUALITY`].
    quality: u8,
    /// The windowed-sinc table for the current quality and rate pair, built on
    /// first use. `None` at quality 0, where the kernel is two-point linear.
    sinc: Option<SincTable>,
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
            ..Self::auto()
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
            carry: Vec::new(),
            phase: 0.0,
            quality: DEFAULT_QUALITY,
            sinc: None,
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
            ..
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
        if !pcm_formats().contains(format) {
            return Err(G2gError::CapsMismatch);
        }
        Ok((*format, *channels, *sample_rate))
    }

    /// Reset the streaming state (on flush or a rate/format change), so the next
    /// buffer restarts the interpolation grid without carrying stale samples.
    fn reset_state(&mut self) {
        self.carry.clear();
        self.phase = 0.0;
    }
}

impl AsyncElement for AudioResample {
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

    /// Native `DerivedFields`: a supported PCM input maps to the same format +
    /// channels (the coupled fields) at the target sample rate.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        // Property, or the caps-resolved target from startup (M755). Reflecting the
        // resolved rate (not just the property) lets a mid-stream input-rate change
        // re-derive to a single fixed output, so the element keeps its 48 kHz target
        // even when its downstream feasibility snapshot is blanked by an intervening
        // format converter (`audioresample ! audioconvert ! rate-pin`, where the
        // converter retargets `format`, a scalar with no wildcard, so the backward
        // feasibility projection is empty). Auto + unresolved (startup) stays the
        // passthrough+wildcard set, so startup negotiation is unchanged.
        // `channels` passes through untouched, so an `ANY_CHANNELS` (0)
        // placeholder input derives an `ANY_CHANNELS` output (a downstream
        // capsfilter pins it); the shapes never require a concrete count, else a
        // decoder's pre-decode placeholder would collapse the derived set to empty
        // and the solver would read it as an unsatisfiable link.
        let shapes = match self.out_rate() {
            // Property-driven, or a caps-resolved target: the fixed rate.
            Some(rate) => {
                alloc::vec![AudioShape::PASSTHROUGH.with_sample_rate(FieldTransform::Fixed(rate))]
            }
            // Caps-driven (auto), not yet resolved: default to passthrough (the
            // input rate, no resampling), but advertise "any rate" so a downstream
            // capsfilter pins the target. Passthrough is the preferred (first)
            // alternative.
            None => alloc::vec![
                AudioShape::PASSTHROUGH,
                AudioShape::PASSTHROUGH.with_sample_rate(FieldTransform::Fixed(ANY_SAMPLE_RATE)),
            ],
        };
        CapsConstraint::DerivedFields(CapsTransform::Audio {
            accept: pcm_formats().to_vec(),
            produce: Vec::new(),
            shapes,
        })
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
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    self.last_timing = frame.timing;
                    // Effective output rate: property, or caps-resolved (auto).
                    let out_rate = self.out_rate().ok_or(G2gError::NotConfigured)?;
                    let resampled =
                        self.resample(slice, in_format, in_channels, in_rate, out_rate)?;

                    let new_caps = Caps::Audio {
                        format: in_format,
                        channels: in_channels,
                        sample_rate: out_rate,
                        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
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
            "quality" => {
                let q = value.as_int().ok_or(PropError::Type)?;
                if !(MIN_QUALITY..=MAX_QUALITY).contains(&q) {
                    return Err(PropError::Value);
                }
                self.quality = q as u8;
                // The table is quality-specific; drop it so the next buffer
                // rebuilds at the new width.
                self.sinc = None;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "samplerate" => self.out_rate().map(|r| PropValue::Uint(r as u64)),
            "quality" => Some(PropValue::Int(i64::from(self.quality))),
            _ => None,
        }
    }
}

/// `AudioResample`'s settable properties (M107): the output sample rate and the
/// interpolation quality.
static AUDIORESAMPLE_PROPS: &[PropertySpec] = &[
    PropertySpec::new("samplerate", PropKind::Uint, "output samples per second"),
    PropertySpec::new(
        "quality",
        PropKind::Int,
        "interpolation quality: 0 is two-point linear, 1 and up windowed sinc",
    )
    .with_range(MIN_QUALITY_TEXT, MAX_QUALITY_TEXT)
    .with_default(DEFAULT_QUALITY_TEXT),
];

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
        if in_frame == 0 || !src.len().is_multiple_of(in_frame) {
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

        self.ensure_kernel(in_rate, out_rate);

        // Append the buffer, decoded to per-channel f32, to the carry, so the
        // inner interpolation loop is index math over one contiguous plane and
        // never has to special-case a read that reaches back before the buffer.
        let mut planes = core::mem::take(&mut self.carry);
        if planes.len() != ch {
            planes = alloc::vec![Vec::new(); ch];
        }
        for plane in &mut planes {
            plane.reserve(n);
        }
        for f in 0..n {
            let base = f * in_frame;
            for (c, plane) in planes.iter_mut().enumerate() {
                plane.push(read_sample(&src[base + c * bytes..], in_format));
            }
        }

        // input samples advanced per output sample.
        let step = in_rate as f64 / out_rate as f64;
        let mut dst = Vec::new();
        let rel = self.emit_window(&planes, self.phase, step, in_format, &mut dst);

        // Keep only the samples the next output's kernel window still reaches:
        // everything from its leftmost tap on. The read position moves with it.
        let total = planes[0].len() as isize;
        let keep_from =
            (floor_isize(rel) - self.half_width() as isize + 1).clamp(0, total) as usize;
        for plane in &mut planes {
            plane.drain(..keep_from);
        }
        self.phase = rel - keep_from as f64;
        self.carry = planes;
        Ok(dst.into_boxed_slice())
    }

    /// Half the kernel width: the input samples the kernel reads on each side of
    /// the read position. 1 for the linear path (its two endpoints).
    fn half_width(&self) -> usize {
        self.sinc.as_ref().map_or(1, |t| t.taps / 2)
    }

    /// Build (or rebuild) the sinc table for the current quality and rate pair.
    /// Quality 0 drops it: that path is two-point linear and needs no table.
    fn ensure_kernel(&mut self, in_rate: u32, out_rate: u32) {
        if self.quality == 0 {
            self.sinc = None;
            return;
        }
        let want = (self.quality, in_rate, out_rate);
        if self.sinc.as_ref().is_some_and(|t| t.built_for == want) {
            return;
        }
        self.sinc = Some(SincTable::build(self.quality, in_rate, out_rate));
    }

    /// Emit every output sample whose whole kernel window fits inside `planes`,
    /// starting at read position `rel`, and return the position left over.
    ///
    /// Tap `k` weights input index `floor(rel) - half + 1 + k`. An index before
    /// the plane's start only happens at the head of a stream, where it holds
    /// sample 0; the loop bound keeps the right-hand taps in range.
    fn emit_window(
        &self,
        planes: &[Vec<f32>],
        mut rel: f64,
        step: f64,
        format: AudioFormat,
        dst: &mut Vec<u8>,
    ) -> f64 {
        let half = self.half_width();
        let total = planes[0].len();
        let last = total.saturating_sub(1) as isize;
        let mut weights: Vec<f32> = Vec::with_capacity(2 * half);
        while rel < total as f64 - half as f64 {
            let start = floor_isize(rel) - half as isize + 1;
            let frac = rel - floor_isize(rel) as f64;
            match &self.sinc {
                Some(table) => table.tap_weights(frac, &mut weights),
                None => {
                    weights.clear();
                    weights.push(1.0 - frac as f32);
                    weights.push(frac as f32);
                }
            }
            for plane in planes {
                let mut acc = 0f32;
                for (k, w) in weights.iter().enumerate() {
                    let index = (start + k as isize).clamp(0, last) as usize;
                    acc += w * plane[index];
                }
                write_sample(dst, acc, format);
            }
            rel += step;
        }
        rel
    }

    /// At end of stream, emit the output samples the per-buffer loop deferred:
    /// the read positions whose kernel window ran past the last input sample
    /// (`resample` stops at `rel < total - half` and keeps the rest in the
    /// carry). The signal past the last sample is held constant, so appending
    /// `half` copies of it lets the same kernel run the grid out, landing the
    /// total stream output at `ceil(n_in * out/in)`. Returns `None` when there
    /// is no carry: rate 1:1 (bypass) or a stream that ended before any
    /// resampled frame.
    fn flush_tail(&mut self) -> Result<Option<Box<[u8]>>, G2gError> {
        let mut planes = core::mem::take(&mut self.carry);
        if planes.first().is_none_or(Vec::is_empty) {
            return Ok(None);
        }
        let (in_format, in_channels, in_rate) = self.input.ok_or(G2gError::NotConfigured)?;
        let out_rate = self.out_rate().ok_or(G2gError::NotConfigured)?;
        // 1:1 bypasses and never populates the carry; guard against a zero rate.
        if in_rate == out_rate || in_rate == 0 || out_rate == 0 {
            return Ok(None);
        }
        if planes.len() != in_channels as usize {
            return Err(G2gError::CapsMismatch);
        }
        let half = self.half_width();
        for plane in &mut planes {
            let held = plane[plane.len() - 1];
            plane.resize(plane.len() + half, held);
        }
        let step = in_rate as f64 / out_rate as f64;
        let mut dst = Vec::new();
        self.emit_window(&planes, self.phase, step, in_format, &mut dst);
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
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        let set = CapsSet::from_alternatives(pcm_formats().map(pcm).to_vec());
        Vec::from([PadTemplate::sink(set.clone()), PadTemplate::source(set)])
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
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    /// A resampler pinned to the two-point linear kernel, the exact grid the
    /// tests below were written against. The element's default is sinc.
    fn linear(target_rate: u32) -> AudioResample {
        let mut r = AudioResample::new(target_rate);
        r.set_property("quality", PropValue::Int(0)).unwrap();
        r
    }

    /// The probe tone, as turns of a sine per input sample: 7 cycles every 48
    /// samples (7 kHz at 48 kHz). The ratio is not a divisor of a 2x output
    /// grid, so the output positions sample the sine's phase densely instead of
    /// landing on a handful of points.
    const TONE_CYCLES: usize = 7;
    const TONE_PERIOD: usize = 48;

    /// The tone at a fractional input-sample position. The phase is reduced to
    /// one turn first, so a long run does not lose bits in `sin`'s reduction.
    fn tone(input_position: f64) -> f64 {
        let turns = input_position * TONE_CYCLES as f64 / TONE_PERIOD as f64;
        crate::mathf::sin(core::f64::consts::TAU * (turns - crate::mathf::floor(turns)))
    }

    #[test]
    fn derived_output_retargets_rate_only() {
        let r = AudioResample::new(48_000);
        let CapsConstraint::DerivedFields(t) = r.caps_constraint_as_transform() else {
            panic!("expected DerivedFields");
        };
        assert_eq!(
            t.passthrough(),
            PassthroughFields::NONE.with_format().with_channels()
        );
        let f = |c: &Caps| t.derive(c);
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
        let mut r = linear(96_000);
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
        let mut r = linear(24_000);
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
        let mut r = linear(96_000);
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
        let mut r = linear(96_000);
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
    fn quality_round_trips_and_rejects_out_of_range() {
        let mut r = AudioResample::new(48_000);
        assert_eq!(
            r.get_property("quality"),
            Some(PropValue::Int(i64::from(DEFAULT_QUALITY))),
            "a fresh element reports the default the spec declares"
        );
        r.set_property("quality", PropValue::Int(MAX_QUALITY))
            .unwrap();
        assert_eq!(r.get_property("quality"), Some(PropValue::Int(MAX_QUALITY)));
        assert_eq!(
            r.set_property("quality", PropValue::Int(MAX_QUALITY + 1)),
            Err(PropError::Value)
        );
    }

    #[test]
    fn sinc_tracks_a_sine_far_closer_than_linear() {
        const IN_RATE: u32 = 48_000;
        const OUT_RATE: u32 = 96_000;
        const FRAMES: usize = 512;
        // Two-point interpolation reads the chord where the sine takes the arc.
        // At a midpoint the two differ by a factor cos(pi * f / fs), so over a
        // grid that samples every phase the peak error is 1 - cos(pi * f / fs).
        let linear_bound = 1.0
            - crate::mathf::cos(core::f64::consts::PI * TONE_CYCLES as f64 / TONE_PERIOD as f64);

        let samples: Vec<f32> = (0..FRAMES).map(|i| tone(i as f64) as f32).collect();
        let src = interleave_f32(&[&samples]);
        // Skip the head, where the kernel window runs off the front of the
        // stream and holds sample 0; the widest kernel covers every quality.
        let skip = QUALITY_TAPS[QUALITY_TAPS.len() - 1];
        let step = f64::from(IN_RATE) / f64::from(OUT_RATE);

        let peak_error = |quality: i64| -> f64 {
            let mut r = AudioResample::new(OUT_RATE);
            r.set_property("quality", PropValue::Int(quality)).unwrap();
            r.configure_pipeline(&audio(AudioFormat::PcmF32Le, 1, IN_RATE))
                .unwrap();
            let got = f32_samples(
                &r.resample(&src, AudioFormat::PcmF32Le, 1, IN_RATE, OUT_RATE)
                    .unwrap(),
            );
            assert!(got.len() > skip, "the probe is long enough to measure");
            got.iter()
                .enumerate()
                .skip(skip)
                .map(|(j, v)| (f64::from(*v) - tone(j as f64 * step)).abs())
                .fold(0.0f64, f64::max)
        };

        let linear_error = peak_error(0);

        let sinc_error = peak_error(i64::from(DEFAULT_QUALITY));
        assert!(
            linear_error <= linear_bound && linear_error > linear_bound * 0.99,
            "linear peak error {linear_error} sits at its analytic bound {linear_bound}"
        );
        // At 7 kHz the default 32-tap kernel's pass band is flat, so what is
        // left is stop-band leakage: it lands near 5e-6, the resolution of the
        // f32 samples it reads. A thousandth of the linear bound is the floor
        // asserted here, not the error the kernel actually makes.
        assert!(
            sinc_error < linear_bound / 1000.0,
            "sinc peak error {sinc_error} vs linear {linear_error}"
        );
    }

    #[test]
    fn sinc_eos_flush_lands_the_rate_ratio_count() {
        // The sinc kernel defers `taps / 2` samples instead of one, so the EOS
        // flush carries more of the tail; the total must still be the rate-ratio
        // count and no input sample may be dropped.
        const FRAMES: usize = 256;
        let mut r = AudioResample::new(96_000);
        r.configure_pipeline(&audio(AudioFormat::PcmF32Le, 1, 48_000))
            .unwrap();
        let samples: Vec<f32> = (0..FRAMES).map(|i| tone(i as f64) as f32).collect();
        let src = interleave_f32(&[&samples]);
        let body = f32_samples(
            &r.resample(&src, AudioFormat::PcmF32Le, 1, 48_000, 96_000)
                .unwrap(),
        );
        let tail = f32_samples(&r.flush_tail().unwrap().expect("tail emitted"));
        assert_eq!(body.len() + tail.len(), FRAMES * 2);
        assert!(r.flush_tail().unwrap().is_none(), "the carry is consumed");
    }

    #[test]
    fn sinc_preserves_a_constant_signal() {
        // Every phase row is normalized to unity DC gain, so a held level comes
        // out untouched, transient head included.
        const LEVEL: f32 = 0.25;
        let mut r = AudioResample::new(44_100);
        r.configure_pipeline(&audio(AudioFormat::PcmF32Le, 1, 48_000))
            .unwrap();
        let src = interleave_f32(&[&[LEVEL; 128]]);
        let got = f32_samples(
            &r.resample(&src, AudioFormat::PcmF32Le, 1, 48_000, 44_100)
                .unwrap(),
        );
        assert!(!got.is_empty());
        for v in got {
            assert!((v - LEVEL).abs() < 1e-6, "got {v} want {LEVEL}");
        }
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
