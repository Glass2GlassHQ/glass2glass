//! EBU R128 loudness meter (`ebur128`). A passthrough that measures momentary
//! (400 ms), short-term (3 s) and gated integrated loudness in LUFS per
//! ITU-R BS.1770-4, and exposes them via getters, the way
//! [`Level`](crate::level::Level) exposes peak and RMS. ffmpeg calls the same
//! measurement `ebur128`; GStreamer has no equivalent element. The buffer is
//! forwarded byte for byte. CPU-only `no_std`, no math dep.
//!
//! The chain per BS.1770-4 is: K-weighting (a +4 dB high shelf at 1682 Hz, then
//! an RLB high-pass at 38 Hz, both as biquads), the mean square of each
//! K-weighted channel over a 400 ms block, and the channel-weighted sum of those
//! turned into `-0.691 + 10 log10(...)` LUFS. Blocks overlap by 75%, so one
//! starts every 100 ms.
//!
//! The integrated measurement is gated twice: a block below -70 LUFS is dropped
//! outright, and of what is left, a block more than 10 LU below the mean of the
//! survivors is dropped too. It keeps one `f64` per 100 ms block of the whole
//! stream, since the relative gate cannot be applied until the mean is known.
//!
//! BS.1770 tabulates its coefficients at 48 kHz. This builds them from the
//! analog shelf and high-pass the table came from, so any rate whose Nyquist is
//! above the shelf frequency measures correctly; a rate below that is refused at
//! negotiation rather than measured wrong.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    G2gError, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, ANY_CHANNELS, ANY_SAMPLE_RATE,
};

use crate::audioconvert::{ns_to_samples, pcm_formats, samples_to_ns};
use crate::audiofx::{self, IirCascade, IirSection};
use crate::mathf;

/// The analog prototype BS.1770-4's 48 kHz coefficient table is derived from:
/// a high shelf of `SHELF_GAIN_DB` at `SHELF_FREQUENCY_HZ`, whose mid-band gain
/// is the shelf gain raised to `SHELF_MID_GAIN_EXPONENT`.
const SHELF_FREQUENCY_HZ: f64 = 1681.974450955533;
const SHELF_GAIN_DB: f64 = 3.999843853973347;
const SHELF_Q: f64 = 0.7071752369554196;
const SHELF_MID_GAIN_EXPONENT: f64 = 0.4996667741545416;
/// The RLB stage: a second-order high-pass.
const HIGHPASS_FREQUENCY_HZ: f64 = 38.13547087602444;
const HIGHPASS_Q: f64 = 0.5003270373238773;

/// The calibration term of `L = -0.691 + 10 log10(sum G z)`.
const LOUDNESS_OFFSET_LU: f64 = -0.691;
/// A momentary block is 400 ms and blocks overlap by 75%, so a new one starts
/// every quarter of a block.
const BLOCK_DURATION_NS: u64 = 400_000_000;
const BLOCKS_PER_OVERLAP: u64 = 4;
const STEP_DURATION_NS: u64 = BLOCK_DURATION_NS / BLOCKS_PER_OVERLAP;
const SHORT_TERM_DURATION_NS: u64 = 3_000_000_000;
const MOMENTARY_STEPS: usize = (BLOCK_DURATION_NS / STEP_DURATION_NS) as usize;
const SHORT_TERM_STEPS: usize = (SHORT_TERM_DURATION_NS / STEP_DURATION_NS) as usize;

/// The absolute gate: a block quieter than this never enters the integrated
/// measurement. The relative gate then drops what sits this far below the mean
/// of the survivors.
const ABSOLUTE_GATE_LUFS: f64 = -70.0;
const RELATIVE_GATE_LU: f64 = -10.0;

/// BS.1770-4 Table 3: the surround channels count for more than the front ones,
/// and the LFE not at all.
const FRONT_WEIGHT: f64 = 1.0;
const SURROUND_WEIGHT: f64 = 1.41;
const LFE_WEIGHT: f64 = 0.0;

/// gst `level`'s reporting interval, 100 ms, which is also one block step.
const DEFAULT_INTERVAL_NS: u64 = 100_000_000;
const DEFAULT_INTERVAL_TEXT: &str = "100000000";
const DEFAULT_POST_MESSAGES: bool = true;
const DEFAULT_POST_MESSAGES_TEXT: &str = "true";

/// The weight of each channel, laid over the interleave order the channel count
/// implies (SMPTE: L, R, C, LFE, Ls, Rs). A count whose layout we cannot name
/// gets the front weight throughout.
fn channel_weights(channels: usize) -> Vec<f64> {
    match channels {
        4 => vec![FRONT_WEIGHT, FRONT_WEIGHT, SURROUND_WEIGHT, SURROUND_WEIGHT],
        5 => vec![
            FRONT_WEIGHT,
            FRONT_WEIGHT,
            FRONT_WEIGHT,
            SURROUND_WEIGHT,
            SURROUND_WEIGHT,
        ],
        6 => vec![
            FRONT_WEIGHT,
            FRONT_WEIGHT,
            FRONT_WEIGHT,
            LFE_WEIGHT,
            SURROUND_WEIGHT,
            SURROUND_WEIGHT,
        ],
        _ => vec![FRONT_WEIGHT; channels],
    }
}

/// `IirSection` runs `y = sum b x + sum a y`, so the denominator arrives with
/// its sign already flipped.
fn section(b: [f64; 3], a1: f64, a2: f64) -> IirSection {
    IirSection {
        b: [b[0], b[1], b[2], 0.0, 0.0],
        a: [0.0, -a1, -a2, 0.0, 0.0],
    }
}

/// The K-weighting high shelf at `sample_rate`, by the bilinear transform of the
/// analog prototype. At 48 kHz this reproduces BS.1770-4's tabulated stage 1.
fn shelf_section(sample_rate: u32) -> IirSection {
    let k = mathf::tan(core::f64::consts::PI * SHELF_FREQUENCY_HZ / sample_rate as f64);
    let high_gain = mathf::powf(10.0, SHELF_GAIN_DB / 20.0);
    let mid_gain = mathf::powf(high_gain, SHELF_MID_GAIN_EXPONENT);
    let denominator = 1.0 + k / SHELF_Q + k * k;
    section(
        [
            (high_gain + mid_gain * k / SHELF_Q + k * k) / denominator,
            2.0 * (k * k - high_gain) / denominator,
            (high_gain - mid_gain * k / SHELF_Q + k * k) / denominator,
        ],
        2.0 * (k * k - 1.0) / denominator,
        (1.0 - k / SHELF_Q + k * k) / denominator,
    )
}

/// The RLB high-pass at `sample_rate`, BS.1770-4's tabulated stage 2 at 48 kHz.
fn highpass_section(sample_rate: u32) -> IirSection {
    let k = mathf::tan(core::f64::consts::PI * HIGHPASS_FREQUENCY_HZ / sample_rate as f64);
    let denominator = 1.0 + k / HIGHPASS_Q + k * k;
    section(
        [1.0, -2.0, 1.0],
        2.0 * (k * k - 1.0) / denominator,
        (1.0 - k / HIGHPASS_Q + k * k) / denominator,
    )
}

/// The shelf has to sit below Nyquist for the bilinear transform to place it at
/// all, so a rate this low cannot carry a K-weighted measurement.
fn rate_carries_the_shelf(sample_rate: u32) -> bool {
    sample_rate as f64 > 2.0 * SHELF_FREQUENCY_HZ
}

/// `10 log10(power)`, the dB the loudness formula is written in.
fn power_decibels(power: f64) -> f64 {
    10.0 * mathf::log2(power) / core::f64::consts::LOG2_10
}

/// The loudness of a channel-weighted mean square. Silence has none.
fn loudness_of(power: f64) -> Option<f64> {
    if power <= 0.0 {
        return None;
    }
    Some(LOUDNESS_OFFSET_LU + power_decibels(power))
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::ebur128::Ebur128;
///
/// let meter = Ebur128::new();
/// assert!(meter.integrated_lufs().is_none());
/// ```
#[derive(Debug)]
pub struct Ebur128 {
    interval_ns: u64,
    post_messages: bool,

    format: AudioFormat,
    channels: usize,
    sample_rate: u32,
    configured: bool,
    weights: Vec<f64>,
    cascade: IirCascade,

    /// Sample frames one 100 ms step holds.
    step_frames: u64,
    /// Channel-weighted sum of squares of the step being filled, and how many
    /// frames have gone into it.
    step_power: f64,
    step_filled: u64,
    /// The last [`SHORT_TERM_STEPS`] completed steps, newest last.
    steps: VecDeque<f64>,
    /// Mean power of every 400 ms block that passed the absolute gate, which the
    /// relative gate is applied to once the whole stream has been seen.
    gated_powers: Vec<f64>,

    frames_seen: u64,
    next_report_ns: u64,
    momentary: Option<f64>,
    short_term: Option<f64>,
    integrated: Option<f64>,
}

impl Default for Ebur128 {
    fn default() -> Self {
        Self::new()
    }
}

impl Ebur128 {
    pub fn new() -> Self {
        Self {
            interval_ns: DEFAULT_INTERVAL_NS,
            post_messages: DEFAULT_POST_MESSAGES,
            format: AudioFormat::PcmS16Le,
            channels: 0,
            sample_rate: 0,
            configured: false,
            weights: Vec::new(),
            cascade: IirCascade::default(),
            step_frames: 0,
            step_power: 0.0,
            step_filled: 0,
            steps: VecDeque::new(),
            gated_powers: Vec::new(),
            frames_seen: 0,
            next_report_ns: 0,
            momentary: None,
            short_term: None,
            integrated: None,
        }
    }

    pub fn with_interval(mut self, interval_ns: u64) -> Self {
        self.interval_ns = interval_ns;
        self
    }

    /// Loudness of the last 400 ms, as of the most recent report.
    pub fn momentary_lufs(&self) -> Option<f64> {
        self.momentary
    }

    /// Loudness of the last 3 s, as of the most recent report.
    pub fn short_term_lufs(&self) -> Option<f64> {
        self.short_term
    }

    /// Gated loudness of everything measured so far, as of the most recent
    /// report. `None` until one block clears the absolute gate.
    pub fn integrated_lufs(&self) -> Option<f64> {
        self.integrated
    }

    /// The K-weighting cascade's magnitude at `frequency`, so the loudness a
    /// tone should measure can be derived rather than guessed.
    pub fn response_at(&self, frequency: f64) -> f64 {
        let w = core::f64::consts::TAU * frequency / self.sample_rate.max(1) as f64;
        self.cascade.magnitude(mathf::cos(w), mathf::sin(w))
    }

    fn accept_input(&self, caps: &Caps) -> Result<(AudioFormat, usize, u32), G2gError> {
        let Caps::Audio {
            format,
            channels,
            sample_rate,
            ..
        } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !pcm_formats().contains(format)
            || *channels == 0
            || *channels == ANY_CHANNELS
            || *sample_rate == ANY_SAMPLE_RATE
            || !rate_carries_the_shelf(*sample_rate)
        {
            return Err(G2gError::CapsMismatch);
        }
        Ok((*format, *channels as usize, *sample_rate))
    }

    fn configure(&mut self, caps: &Caps) -> Result<(), G2gError> {
        let input = self.accept_input(caps)?;
        // The runner re-announces the solved caps mid-stream, so rebuilding
        // unconditionally would throw away the integrated measurement.
        if (self.format, self.channels, self.sample_rate) != input {
            let (format, channels, sample_rate) = input;
            self.format = format;
            self.channels = channels;
            self.sample_rate = sample_rate;
            self.weights = channel_weights(channels);
            self.cascade.set_sections(
                Vec::from([shelf_section(sample_rate), highpass_section(sample_rate)]),
                channels,
            );
            self.step_frames = ns_to_samples(STEP_DURATION_NS, sample_rate);
            self.reset_measurement();
        }
        self.configured = true;
        Ok(())
    }

    fn reset_measurement(&mut self) {
        self.cascade.reset();
        self.step_power = 0.0;
        self.step_filled = 0;
        self.steps.clear();
        self.gated_powers.clear();
        self.frames_seen = 0;
        self.next_report_ns = self.interval_ns;
        self.momentary = None;
        self.short_term = None;
        self.integrated = None;
    }

    /// Mean power over the last `steps` completed steps, once there are that
    /// many.
    fn window_power(&self, steps: usize) -> Option<f64> {
        if self.steps.len() < steps || self.step_frames == 0 {
            return None;
        }
        let sum: f64 = self.steps.iter().rev().take(steps).sum();
        Some(sum / (steps as u64 * self.step_frames) as f64)
    }

    /// Close the 100 ms step just filled, and with it the 400 ms block that ends
    /// on it.
    fn close_step(&mut self) {
        self.steps.push_back(self.step_power);
        if self.steps.len() > SHORT_TERM_STEPS {
            self.steps.pop_front();
        }
        self.step_power = 0.0;
        self.step_filled = 0;

        if let Some(power) = self.window_power(MOMENTARY_STEPS) {
            if loudness_of(power).is_some_and(|block| block > ABSOLUTE_GATE_LUFS) {
                self.gated_powers.push(power);
            }
        }
    }

    /// The integrated loudness: the mean of the absolutely gated blocks that
    /// also clear the relative gate.
    fn gated_loudness(&self) -> Option<f64> {
        if self.gated_powers.is_empty() {
            return None;
        }
        let mean = |powers: &[f64]| powers.iter().sum::<f64>() / powers.len() as f64;
        let relative_gate = loudness_of(mean(&self.gated_powers))? + RELATIVE_GATE_LU;
        let kept: Vec<f64> = self
            .gated_powers
            .iter()
            .copied()
            .filter(|&power| loudness_of(power).is_some_and(|block| block > relative_gate))
            .collect();
        if kept.is_empty() {
            return None;
        }
        loudness_of(mean(&kept))
    }

    fn report(&mut self) {
        self.momentary = self.window_power(MOMENTARY_STEPS).and_then(loudness_of);
        self.short_term = self.window_power(SHORT_TERM_STEPS).and_then(loudness_of);
        self.integrated = self.gated_loudness();
    }

    /// Feed one buffer through the K-weighting and into the block accumulators.
    fn observe(&mut self, bytes: &[u8]) {
        if self.channels == 0 || self.step_frames == 0 {
            return;
        }
        let mut samples = audiofx::decode(bytes, self.format);
        self.cascade.run(&mut samples);
        for frame in samples.chunks_exact(self.channels) {
            for (channel, &sample) in frame.iter().enumerate() {
                let value = sample as f64;
                self.step_power += self.weights[channel] * value * value;
            }
            self.step_filled += 1;
            if self.step_filled == self.step_frames {
                self.close_step();
            }
        }
        self.frames_seen += (samples.len() / self.channels) as u64;

        while self.interval_ns > 0
            && samples_to_ns(self.frames_seen, self.sample_rate) >= self.next_report_ns
        {
            self.next_report_ns += self.interval_ns;
            self.report();
        }
    }
}

impl AsyncElement for Ebur128 {
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

    /// Pure passthrough: the meter never changes the stream.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Audio {
                format,
                sample_rate,
                ..
            } if pcm_formats().contains(format) && rate_carries_the_shelf(*sample_rate) => {
                CapsSet::one(input.clone())
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configure(absolute_caps)?;
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
                    if self.post_messages {
                        if let Some(slice) = frame.domain.as_system_slice() {
                            self.observe(slice);
                        }
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(caps) => {
                    self.configure(&caps)?;
                    out.push(PipelinePacket::CapsChanged(caps)).await?;
                }
                PipelinePacket::Flush => {
                    self.reset_measurement();
                    out.push(PipelinePacket::Flush).await?;
                }
                // The runner emits the final Eos after process(Eos) returns (the
                // transform contract); the tail since the last interval is
                // reported here so nothing measured goes unreported.
                PipelinePacket::Eos => {
                    if self.post_messages {
                        self.report();
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
        EBUR128_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "EBU R128 loudness meter",
            "Filter/Analyzer/Audio",
            "Measures momentary, short-term and gated integrated loudness in LUFS",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "interval" => self.interval_ns = value.as_uint().ok_or(PropError::Type)?,
            "post-messages" => self.post_messages = value.as_bool().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "interval" => Some(PropValue::Uint(self.interval_ns)),
            "post-messages" => Some(PropValue::Bool(self.post_messages)),
            _ => None,
        }
    }
}

static EBUR128_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "interval",
        PropKind::Uint,
        "nanoseconds of audio between measurement reports",
    )
    .with_default(DEFAULT_INTERVAL_TEXT),
    PropertySpec::new(
        "post-messages",
        PropKind::Bool,
        "run the measurement when true",
    )
    .with_default(DEFAULT_POST_MESSAGES_TEXT),
];

impl PadTemplates for Ebur128 {
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = |format| Caps::Audio {
            format,
            channels: ANY_CHANNELS,
            sample_rate: ANY_SAMPLE_RATE,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        let set = CapsSet::from_alternatives(pcm_formats().iter().copied().map(pcm).collect());
        vec![PadTemplate::sink(set.clone()), PadTemplate::source(set)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REFERENCE_RATE: u32 = 48_000;

    /// BS.1770-4 Table 1, the K-weighting shelf at 48 kHz.
    const TABLE_SHELF_B: [f64; 3] = [1.53512485958697, -2.69169618940638, 1.19839281085285];
    const TABLE_SHELF_A: [f64; 2] = [-1.69065929318241, 0.73248077421585];
    /// BS.1770-4 Table 2, the RLB high-pass at 48 kHz.
    const TABLE_HIGHPASS_B: [f64; 3] = [1.0, -2.0, 1.0];
    const TABLE_HIGHPASS_A: [f64; 2] = [-1.99004745483398, 0.99007225036621];
    /// The table is printed to fourteen decimals.
    const TABLE_TOLERANCE: f64 = 1e-11;

    fn assert_matches_table(built: &IirSection, b: [f64; 3], a: [f64; 2]) {
        for (index, expected) in b.iter().enumerate() {
            assert!(
                (built.b[index] - expected).abs() < TABLE_TOLERANCE,
                "b{index} is {} not {expected}",
                built.b[index]
            );
        }
        // the section negates the denominator, so the table's a comes back flipped.
        for (index, expected) in a.iter().enumerate() {
            assert!(
                (-built.a[index + 1] - expected).abs() < TABLE_TOLERANCE,
                "a{} is {} not {expected}",
                index + 1,
                -built.a[index + 1]
            );
        }
    }

    #[test]
    fn the_derived_coefficients_are_the_published_ones_at_48_khz() {
        assert_matches_table(&shelf_section(REFERENCE_RATE), TABLE_SHELF_B, TABLE_SHELF_A);
        assert_matches_table(
            &highpass_section(REFERENCE_RATE),
            TABLE_HIGHPASS_B,
            TABLE_HIGHPASS_A,
        );
    }

    #[test]
    fn a_rate_below_the_shelf_is_refused() {
        let meter = Ebur128::new();
        // 3 kHz puts Nyquist under the 1682 Hz shelf.
        let caps = Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 2,
            sample_rate: 3_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        assert_eq!(
            meter.accept_input(&caps).unwrap_err(),
            G2gError::CapsMismatch
        );
    }

    #[test]
    fn surround_weights_follow_the_table() {
        assert_eq!(channel_weights(2), [FRONT_WEIGHT, FRONT_WEIGHT]);
        assert_eq!(
            channel_weights(6),
            [
                FRONT_WEIGHT,
                FRONT_WEIGHT,
                FRONT_WEIGHT,
                LFE_WEIGHT,
                SURROUND_WEIGHT,
                SURROUND_WEIGHT
            ]
        );
    }

    #[test]
    fn the_block_grid_is_four_steps_of_a_hundred_milliseconds() {
        assert_eq!(STEP_DURATION_NS, 100_000_000);
        assert_eq!(MOMENTARY_STEPS, 4);
        assert_eq!(SHORT_TERM_STEPS, 30);
    }
}
