//! WebRTC voice processing (`webrtcdsp`) and its far-end tap (`webrtcechoprobe`),
//! the GStreamer elements of the same names, over `sonora`, a pure-Rust port of
//! WebRTC's audio processing module. The probe sits on the playback path and
//! keeps what it forwards, and the dsp on the capture path pulls the 10 ms of
//! it whose timestamps match each 10 ms it processes, as GStreamer's pair does.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;

use sonora::config::{
    AdaptiveDigital, EchoCanceller, FixedDigital, GainController2, HighPassFilter,
    NoiseSuppression, NoiseSuppressionLevel,
};
use sonora::{AudioProcessing, Config, StreamConfig};
use spin::Mutex;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::{DomainSet, MemoryDomainKind, SystemSlice};
use g2g_core::query::LatencyReport;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec, ANY_CHANNELS,
};

use crate::audioconvert::{ns_to_samples, samples_to_ns};
use crate::audiofx::{decode, encode, int_in_range, AUDIOFX_FORMATS};

const DEFAULT_PROBE_NAME: &str = "webrtcechoprobe0";
// GStreamer's rate set, all inside the 8 to 384 kHz the processor takes
const SUPPORTED_SAMPLE_RATES: [u32; 4] = [48_000, 32_000, 16_000, 8_000];
// the processor works on 10 ms periods
const PERIODS_PER_SECOND: u32 = 100;
const NS_PER_SECOND: u64 = 1_000_000_000;
const PERIOD_NS: u64 = NS_PER_SECOND / PERIODS_PER_SECOND as u64;
const NS_PER_MS: i128 = 1_000_000;
// `set_stream_delay_ms` clamps to this
const MAX_STREAM_DELAY_MS: i128 = 500;
// far-end audio a probe keeps for a dsp that has not caught up yet
const FAR_END_HISTORY_NS: u64 = 2 * NS_PER_SECOND;

// every processing stage is on by default, as in GStreamer
const ENABLED_BY_DEFAULT: bool = true;
const ENABLED_TEXT: &str = "true";
const DEFAULT_NOISE_SUPPRESSION_LEVEL_TEXT: &str = "moderate";
const DEFAULT_GAIN_CONTROL_MODE_TEXT: &str = "adaptive-digital";
const GAIN_MIN: i64 = 0;
const GAIN_MIN_TEXT: &str = "0";
const DEFAULT_TARGET_LEVEL_DBFS: i64 = 3;
const DEFAULT_TARGET_LEVEL_DBFS_TEXT: &str = "3";
const TARGET_LEVEL_DBFS_MAX: i64 = 31;
const TARGET_LEVEL_DBFS_MAX_TEXT: &str = "31";
const DEFAULT_COMPRESSION_GAIN_DB: i64 = 9;
const DEFAULT_COMPRESSION_GAIN_DB_TEXT: &str = "9";
// the processor refuses a fixed gain of 50 dB or more
const COMPRESSION_GAIN_DB_MAX: i64 = 49;
const COMPRESSION_GAIN_DB_MAX_TEXT: &str = "49";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StreamShape {
    format: AudioFormat,
    channels: usize,
    rate: u32,
}

impl StreamShape {
    fn accept(caps: &Caps) -> Result<Self, G2gError> {
        match caps {
            Caps::Audio {
                format,
                channels,
                sample_rate,
                ..
            } if AUDIOFX_FORMATS.contains(format)
                && *channels > 0
                && *channels != ANY_CHANNELS
                && SUPPORTED_SAMPLE_RATES.contains(sample_rate) =>
            {
                Ok(Self {
                    format: *format,
                    channels: *channels as usize,
                    rate: *sample_rate,
                })
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn period_frames(&self) -> usize {
        (self.rate / PERIODS_PER_SECOND) as usize
    }

    fn stream_config(&self) -> StreamConfig {
        StreamConfig::new(self.rate, self.channels as u16)
    }
}

fn passthrough_constraint() -> CapsConstraint<'static> {
    CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match StreamShape::accept(input) {
        Ok(_) => CapsSet::one(input.clone()),
        Err(_) => CapsSet::from_alternatives(Vec::new()),
    }))
}

fn pcm_pad_templates() -> Vec<PadTemplate> {
    let mut alternatives = Vec::new();
    for format in AUDIOFX_FORMATS {
        for rate in SUPPORTED_SAMPLE_RATES {
            alternatives.push(Caps::Audio {
                format,
                channels: ANY_CHANNELS,
                sample_rate: rate,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            });
        }
    }
    let set = CapsSet::from_alternatives(alternatives);
    Vec::from([PadTemplate::sink(set.clone()), PadTemplate::source(set)])
}

#[derive(Debug)]
struct TimedChunk {
    pts_ns: u64,
    samples: Vec<f32>,
    consumed_frames: usize,
}

// GStreamer's adapter: samples queued with the pts of the buffer each came in
#[derive(Debug, Default)]
struct TimedSamples {
    chunks: VecDeque<TimedChunk>,
    channels: usize,
    frames: usize,
}

impl TimedSamples {
    fn reset(&mut self, channels: usize) {
        self.chunks.clear();
        self.channels = channels;
        self.frames = 0;
    }

    fn push(&mut self, pts_ns: u64, samples: Vec<f32>) {
        let frames = samples.len() / self.channels;
        if frames == 0 {
            return;
        }
        self.frames += frames;
        self.chunks.push_back(TimedChunk {
            pts_ns,
            samples,
            consumed_frames: 0,
        });
    }

    fn head_pts(&self, rate: u32) -> Option<u64> {
        let chunk = self.chunks.front()?;
        if chunk.pts_ns == FrameTiming::PTS_NONE {
            return None;
        }
        Some(chunk.pts_ns + samples_to_ns(chunk.consumed_frames as u64, rate))
    }

    fn discard(&mut self, frames: usize) {
        self.drain(frames, |_| {});
    }

    fn take_into(&mut self, frames: usize, out: &mut Vec<f32>) {
        self.drain(frames, |samples| out.extend_from_slice(samples));
    }

    fn drain(&mut self, mut frames: usize, mut sink: impl FnMut(&[f32])) {
        while frames > 0 {
            let Some(chunk) = self.chunks.front_mut() else {
                return;
            };
            let chunk_frames = chunk.samples.len() / self.channels;
            let step = frames.min(chunk_frames - chunk.consumed_frames);
            let start = chunk.consumed_frames * self.channels;
            sink(&chunk.samples[start..start + step * self.channels]);
            chunk.consumed_frames += step;
            frames -= step;
            self.frames -= step;
            if chunk.consumed_frames == chunk_frames {
                self.chunks.pop_front();
            }
        }
    }
}

#[derive(Debug)]
struct ProbeState {
    name: String,
    acquired: bool,
    shape: Option<StreamShape>,
    far_end: TimedSamples,
}

type ProbeHandle = Arc<Mutex<ProbeState>>;

static PROBES: Mutex<Vec<ProbeHandle>> = Mutex::new(Vec::new());

fn acquire_probe(name: &str) -> Option<ProbeHandle> {
    let probes = PROBES.lock();
    probes
        .iter()
        .find(|probe| {
            let mut state = probe.lock();
            let free = !state.acquired && state.name == name;
            state.acquired |= free;
            free
        })
        .cloned()
}

fn release_probe(probe: &ProbeHandle) {
    probe.lock().acquired = false;
}

#[derive(Debug, Default)]
struct FarEndPeriod {
    samples: Vec<f32>,
    channels: usize,
    delay_ms: i32,
}

// `gst_webrtc_echo_probe_read` with a sink latency of zero, g2g has no latency query
fn read_far_end(
    state: &mut ProbeState,
    capture: StreamShape,
    capture_pts: Option<u64>,
    period: &mut FarEndPeriod,
) -> Result<(), G2gError> {
    period.samples.clear();
    period.delay_ms = 0;
    let Some(shape) = state.shape else {
        period.channels = capture.channels;
        period
            .samples
            .resize(capture.period_frames() * capture.channels, 0.0);
        return Ok(());
    };
    if shape.rate != capture.rate {
        return Err(G2gError::CapsMismatch);
    }
    period.channels = shape.channels;
    let period_frames = shape.period_frames();
    let available = state.far_end.frames;
    let head_pts = state.far_end.head_pts(shape.rate);

    let (skip, offset, size) = match (capture_pts, head_pts) {
        (None, _) if available >= period_frames => (0, 0, period_frames),
        (None, _) => (0, 0, 0),
        (Some(_), _) if available == 0 => (period_frames, 0, 0),
        (Some(_), None) => (0, 0, available.min(period_frames)),
        (Some(capture_ns), Some(far_ns)) => {
            let diff_ns = far_ns as i128 - capture_ns as i128;
            let diff_frames = ns_to_samples(diff_ns.unsigned_abs() as u64, shape.rate) as usize;
            if diff_ns > 0 {
                let skip = diff_frames.min(period_frames);
                period.delay_ms = (diff_ns / NS_PER_MS).min(MAX_STREAM_DELAY_MS) as i32;
                (skip, 0, available.min(period_frames - skip))
            } else {
                let offset = diff_frames.min(available);
                (0, offset, (available - offset).min(period_frames))
            }
        }
    };

    state.far_end.discard(offset);
    period.samples.resize(skip * shape.channels, 0.0);
    state.far_end.take_into(size, &mut period.samples);
    period.samples.resize(period_frames * shape.channels, 0.0);
    Ok(())
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::webrtcdsp::WebrtcEchoProbe;
///
/// let probe = WebrtcEchoProbe::new().with_probe_name("speaker");
/// ```
#[derive(Debug)]
pub struct WebrtcEchoProbe {
    state: ProbeHandle,
    shape: Option<StreamShape>,
    last_caps: Option<Caps>,
}

impl Default for WebrtcEchoProbe {
    fn default() -> Self {
        Self::new()
    }
}

impl WebrtcEchoProbe {
    pub fn new() -> Self {
        let state = Arc::new(Mutex::new(ProbeState {
            name: DEFAULT_PROBE_NAME.to_string(),
            acquired: false,
            shape: None,
            far_end: TimedSamples::default(),
        }));
        PROBES.lock().push(state.clone());
        Self {
            state,
            shape: None,
            last_caps: None,
        }
    }

    pub fn with_probe_name(self, name: &str) -> Self {
        self.state.lock().name = name.to_string();
        self
    }

    fn configure(&mut self, caps: &Caps) -> Result<(), G2gError> {
        let shape = StreamShape::accept(caps)?;
        let mut state = self.state.lock();
        if state.shape != Some(shape) {
            state.shape = Some(shape);
            state.far_end.reset(shape.channels);
        }
        self.shape = Some(shape);
        Ok(())
    }

    fn keep(&self, shape: StreamShape, frame: &Frame) -> Result<(), G2gError> {
        let bytes = frame
            .domain
            .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
        let mut state = self.state.lock();
        state
            .far_end
            .push(frame.timing.pts_ns, decode(bytes, shape.format));
        let history_frames = ns_to_samples(FAR_END_HISTORY_NS, shape.rate) as usize;
        let excess = state.far_end.frames.saturating_sub(history_frames);
        state.far_end.discard(excess);
        Ok(())
    }
}

impl Drop for WebrtcEchoProbe {
    fn drop(&mut self) {
        PROBES
            .lock()
            .retain(|probe| !Arc::ptr_eq(probe, &self.state));
    }
}

impl AsyncElement for WebrtcEchoProbe {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        StreamShape::accept(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        passthrough_constraint()
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
            let Some(shape) = self.shape else {
                return Err(G2gError::NotConfigured);
            };
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    self.keep(shape, &frame)?;
                    let caps = Caps::Audio {
                        format: shape.format,
                        channels: shape.channels as u8,
                        sample_rate: shape.rate,
                        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
                    };
                    if self.last_caps.as_ref() != Some(&caps) {
                        out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                        self.last_caps = Some(caps);
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(caps) => self.configure(&caps)?,
                PipelinePacket::Flush => {
                    self.state.lock().far_end.reset(shape.channels);
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
        WEBRTCECHOPROBE_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Acoustic Echo Canceller probe",
            "Generic/Audio",
            "Gathers playback buffers for webrtcdsp",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "probe-name" => {
                self.state.lock().name = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "probe-name" => Some(PropValue::Str(self.state.lock().name.clone())),
            _ => None,
        }
    }
}

impl PadTemplates for WebrtcEchoProbe {
    fn pad_templates() -> Vec<PadTemplate> {
        pcm_pad_templates()
    }
}

static WEBRTCECHOPROBE_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "probe-name",
    PropKind::Str,
    "name a webrtcdsp `probe` property selects this probe by",
)
.with_default(DEFAULT_PROBE_NAME)];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseSuppressionStrength {
    Low,
    Moderate,
    High,
    VeryHigh,
}

impl NoiseSuppressionStrength {
    const NICKS: &'static str = "low | moderate | high | very-high";

    fn from_nick(nick: &str) -> Option<Self> {
        match nick {
            "low" => Some(Self::Low),
            "moderate" => Some(Self::Moderate),
            "high" => Some(Self::High),
            "very-high" => Some(Self::VeryHigh),
            _ => None,
        }
    }

    fn nick(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Moderate => "moderate",
            Self::High => "high",
            Self::VeryHigh => "very-high",
        }
    }

    fn processor_level(self) -> NoiseSuppressionLevel {
        match self {
            Self::Low => NoiseSuppressionLevel::Low,
            Self::Moderate => NoiseSuppressionLevel::Moderate,
            Self::High => NoiseSuppressionLevel::High,
            Self::VeryHigh => NoiseSuppressionLevel::VeryHigh,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GainControlMode {
    AdaptiveDigital,
    FixedDigital,
}

impl GainControlMode {
    const NICKS: &'static str = "adaptive-digital | fixed-digital";

    fn from_nick(nick: &str) -> Option<Self> {
        match nick {
            "adaptive-digital" => Some(Self::AdaptiveDigital),
            "fixed-digital" => Some(Self::FixedDigital),
            _ => None,
        }
    }

    fn nick(self) -> &'static str {
        match self {
            Self::AdaptiveDigital => "adaptive-digital",
            Self::FixedDigital => "fixed-digital",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct DspSettings {
    probe_name: String,
    echo_cancel: bool,
    noise_suppression: bool,
    noise_suppression_level: NoiseSuppressionStrength,
    gain_control: bool,
    gain_control_mode: GainControlMode,
    target_level_dbfs: i64,
    compression_gain_db: i64,
    high_pass_filter: bool,
}

impl Default for DspSettings {
    fn default() -> Self {
        Self {
            probe_name: DEFAULT_PROBE_NAME.to_string(),
            echo_cancel: ENABLED_BY_DEFAULT,
            noise_suppression: ENABLED_BY_DEFAULT,
            noise_suppression_level: NoiseSuppressionStrength::Moderate,
            gain_control: ENABLED_BY_DEFAULT,
            gain_control_mode: GainControlMode::AdaptiveDigital,
            target_level_dbfs: DEFAULT_TARGET_LEVEL_DBFS,
            compression_gain_db: DEFAULT_COMPRESSION_GAIN_DB,
            high_pass_filter: ENABLED_BY_DEFAULT,
        }
    }
}

impl DspSettings {
    fn gain_controller(&self) -> GainController2 {
        match self.gain_control_mode {
            GainControlMode::AdaptiveDigital => GainController2 {
                adaptive_digital: Some(AdaptiveDigital {
                    headroom_db: self.target_level_dbfs as f32,
                    ..Default::default()
                }),
                ..Default::default()
            },
            GainControlMode::FixedDigital => GainController2 {
                fixed_digital: FixedDigital {
                    gain_db: self.compression_gain_db as f32,
                },
                ..Default::default()
            },
        }
    }

    fn processor_config(&self) -> Config {
        Config {
            high_pass_filter: self.high_pass_filter.then(HighPassFilter::default),
            echo_canceller: self.echo_cancel.then(|| EchoCanceller {
                enforce_high_pass_filtering: self.high_pass_filter,
                ..Default::default()
            }),
            noise_suppression: self.noise_suppression.then(|| NoiseSuppression {
                level: self.noise_suppression_level.processor_level(),
                ..Default::default()
            }),
            gain_controller2: self.gain_control.then(|| self.gain_controller()),
            ..Default::default()
        }
    }
}

#[derive(Debug, Default)]
struct PlanarBuffers {
    input: Vec<Vec<f32>>,
    output: Vec<Vec<f32>>,
}

impl PlanarBuffers {
    fn load(&mut self, interleaved: &[f32], channels: usize) {
        let frames = interleaved.len() / channels;
        self.input.resize_with(channels, Vec::new);
        self.output.resize_with(channels, Vec::new);
        for (channel, (input, output)) in self.input.iter_mut().zip(&mut self.output).enumerate() {
            input.clear();
            input.extend(interleaved.iter().skip(channel).step_by(channels));
            output.clear();
            output.resize(frames, 0.0);
        }
    }

    fn store(&self, interleaved: &mut [f32], channels: usize) {
        for (channel, output) in self.output.iter().take(channels).enumerate() {
            for (sample, value) in interleaved
                .iter_mut()
                .skip(channel)
                .step_by(channels)
                .zip(output)
            {
                *sample = *value;
            }
        }
    }

    fn run(
        &mut self,
        channels: usize,
        step: impl FnOnce(&[&[f32]], &mut [&mut [f32]]) -> Result<(), sonora::Error>,
    ) -> Result<(), G2gError> {
        let input: Vec<&[f32]> = self
            .input
            .iter()
            .take(channels)
            .map(Vec::as_slice)
            .collect();
        let mut output: Vec<&mut [f32]> = self
            .output
            .iter_mut()
            .take(channels)
            .map(Vec::as_mut_slice)
            .collect();
        step(&input, &mut output).map_err(|_| G2gError::CapsMismatch)
    }
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::webrtcdsp::{NoiseSuppressionStrength, WebrtcDsp};
///
/// let dsp = WebrtcDsp::new()
///     .with_probe("speaker")
///     .with_noise_suppression_level(NoiseSuppressionStrength::High);
/// ```
#[derive(Debug)]
pub struct WebrtcDsp {
    settings: DspSettings,
    shape: Option<StreamShape>,
    processor: Option<AudioProcessing>,
    probe: Option<ProbeHandle>,
    near_end: TimedSamples,
    far_end: FarEndPeriod,
    planar: PlanarBuffers,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl Default for WebrtcDsp {
    fn default() -> Self {
        Self::new()
    }
}

impl WebrtcDsp {
    pub fn new() -> Self {
        Self {
            settings: DspSettings::default(),
            shape: None,
            processor: None,
            probe: None,
            near_end: TimedSamples::default(),
            far_end: FarEndPeriod::default(),
            planar: PlanarBuffers::default(),
            last_caps: None,
            emitted: 0,
        }
    }

    pub fn with_probe(mut self, name: &str) -> Self {
        self.settings.probe_name = name.to_string();
        self
    }

    pub fn with_echo_cancel(mut self, enabled: bool) -> Self {
        self.settings.echo_cancel = enabled;
        self
    }

    pub fn with_noise_suppression(mut self, enabled: bool) -> Self {
        self.settings.noise_suppression = enabled;
        self
    }

    pub fn with_noise_suppression_level(mut self, level: NoiseSuppressionStrength) -> Self {
        self.settings.noise_suppression_level = level;
        self
    }

    pub fn with_gain_control(mut self, enabled: bool) -> Self {
        self.settings.gain_control = enabled;
        self
    }

    pub fn with_gain_control_mode(mut self, mode: GainControlMode) -> Self {
        self.settings.gain_control_mode = mode;
        self
    }

    pub fn with_high_pass_filter(mut self, enabled: bool) -> Self {
        self.settings.high_pass_filter = enabled;
        self
    }

    fn release(&mut self) {
        if let Some(probe) = self.probe.take() {
            release_probe(&probe);
        }
    }

    fn acquired_probe(&mut self) -> Result<ProbeHandle, G2gError> {
        if let Some(probe) = &self.probe {
            return Ok(probe.clone());
        }
        let probe = acquire_probe(&self.settings.probe_name).ok_or(G2gError::NotConfigured)?;
        self.probe = Some(probe.clone());
        Ok(probe)
    }

    fn configure(&mut self, caps: &Caps) -> Result<(), G2gError> {
        let shape = StreamShape::accept(caps)?;
        let mut render = shape.stream_config();
        if self.settings.echo_cancel {
            let probe = self.acquired_probe()?;
            let far_shape = probe.lock().shape;
            if let Some(far_shape) = far_shape {
                if far_shape.rate != shape.rate {
                    return Err(G2gError::CapsMismatch);
                }
                render = far_shape.stream_config();
            }
        }
        if self.shape != Some(shape) || self.processor.is_none() {
            self.processor = Some(
                AudioProcessing::builder()
                    .config(self.settings.processor_config())
                    .capture_config(shape.stream_config())
                    .render_config(render)
                    .build(),
            );
            self.near_end.reset(shape.channels);
            self.shape = Some(shape);
        }
        Ok(())
    }

    fn apply_settings(&mut self) {
        if let Some(processor) = &mut self.processor {
            processor.apply_config(self.settings.processor_config());
        }
    }

    fn process_period(
        &mut self,
        shape: StreamShape,
        capture_pts: Option<u64>,
        samples: &mut [f32],
    ) -> Result<(), G2gError> {
        if self.settings.echo_cancel {
            let probe = self.acquired_probe()?;
            read_far_end(&mut probe.lock(), shape, capture_pts, &mut self.far_end)?;
            let far_channels = self.far_end.channels;
            let render = StreamConfig::new(shape.rate, far_channels as u16);
            let processor = self.processor.as_mut().ok_or(G2gError::NotConfigured)?;
            processor
                .set_stream_delay_ms(self.far_end.delay_ms)
                .map_err(|_| G2gError::CapsMismatch)?;
            self.planar.load(&self.far_end.samples, far_channels);
            self.planar.run(far_channels, |input, output| {
                processor.process_render_f32_with_config(input, &render, &render, output)
            })?;
        }
        let processor = self.processor.as_mut().ok_or(G2gError::NotConfigured)?;
        self.planar.load(samples, shape.channels);
        self.planar.run(shape.channels, |input, output| {
            processor.process_capture_f32(input, output)
        })?;
        self.planar.store(samples, shape.channels);
        Ok(())
    }

    // `drain` also processes the final partial period, padded with silence
    async fn emit_periods(
        &mut self,
        shape: StreamShape,
        drain: bool,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let period_frames = shape.period_frames();
        let mut processed = Vec::new();
        let mut first_pts = None;
        let mut period = Vec::with_capacity(period_frames * shape.channels);
        while self.near_end.frames >= period_frames || (drain && self.near_end.frames > 0) {
            let real_frames = self.near_end.frames.min(period_frames);
            let pts = self.near_end.head_pts(shape.rate);
            first_pts = first_pts.or(Some(pts));
            period.clear();
            self.near_end.take_into(real_frames, &mut period);
            period.resize(period_frames * shape.channels, 0.0);
            self.process_period(shape, pts, &mut period)?;
            processed.extend_from_slice(&period[..real_frames * shape.channels]);
        }
        let Some(pts) = first_pts else {
            return Ok(());
        };

        let caps = Caps::Audio {
            format: shape.format,
            channels: shape.channels as u8,
            sample_rate: shape.rate,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        if self.last_caps.as_ref() != Some(&caps) {
            out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
            self.last_caps = Some(caps);
        }
        let pts_ns = pts.unwrap_or(FrameTiming::PTS_NONE);
        let frames = (processed.len() / shape.channels) as u64;
        let timing = FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            duration_ns: samples_to_ns(frames, shape.rate),
            ..Default::default()
        };
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(encode(&processed, shape.format))),
            timing,
            self.emitted,
        );
        self.emitted += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }
}

impl Drop for WebrtcDsp {
    fn drop(&mut self) {
        self.release();
    }
}

impl AsyncElement for WebrtcDsp {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        StreamShape::accept(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        passthrough_constraint()
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configure(absolute_caps)?;
        Ok(ConfigureOutcome::Accepted)
    }

    fn latency(&self) -> LatencyReport {
        LatencyReport::buffered(PERIOD_NS, Some(PERIOD_NS))
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            let Some(shape) = self.shape else {
                return Err(G2gError::NotConfigured);
            };
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let bytes = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    self.near_end
                        .push(frame.timing.pts_ns, decode(bytes, shape.format));
                    self.emit_periods(shape, false, out).await?;
                }
                PipelinePacket::CapsChanged(caps) => self.configure(&caps)?,
                PipelinePacket::Flush => {
                    self.near_end.reset(shape.channels);
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                // the runner emits the end itself, this only drains the tail
                PipelinePacket::Eos => self.emit_periods(shape, true, out).await?,
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        WEBRTCDSP_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Voice Processor (AGC, AEC, filters, etc.)",
            "Generic/Audio",
            "Pre-processes voice with the WebRTC audio processing module",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let settings = &mut self.settings;
        match name {
            "probe" => {
                settings.probe_name = value.as_str().ok_or(PropError::Type)?.to_string();
                self.release();
            }
            "echo-cancel" => settings.echo_cancel = value.as_bool().ok_or(PropError::Type)?,
            "noise-suppression" => {
                settings.noise_suppression = value.as_bool().ok_or(PropError::Type)?
            }
            "noise-suppression-level" => {
                let nick = value.as_str().ok_or(PropError::Type)?;
                settings.noise_suppression_level =
                    NoiseSuppressionStrength::from_nick(nick).ok_or(PropError::Value)?;
            }
            "gain-control" => settings.gain_control = value.as_bool().ok_or(PropError::Type)?,
            "gain-control-mode" => {
                let nick = value.as_str().ok_or(PropError::Type)?;
                settings.gain_control_mode =
                    GainControlMode::from_nick(nick).ok_or(PropError::Value)?;
            }
            "target-level-dbfs" => {
                settings.target_level_dbfs = int_in_range(value, GAIN_MIN, TARGET_LEVEL_DBFS_MAX)?
            }
            "compression-gain-db" => {
                settings.compression_gain_db =
                    int_in_range(value, GAIN_MIN, COMPRESSION_GAIN_DB_MAX)?
            }
            "high-pass-filter" => {
                settings.high_pass_filter = value.as_bool().ok_or(PropError::Type)?
            }
            _ => return Err(PropError::Unknown),
        }
        self.apply_settings();
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        let settings = &self.settings;
        match name {
            "probe" => Some(PropValue::Str(settings.probe_name.clone())),
            "echo-cancel" => Some(PropValue::Bool(settings.echo_cancel)),
            "noise-suppression" => Some(PropValue::Bool(settings.noise_suppression)),
            "noise-suppression-level" => Some(PropValue::Str(
                settings.noise_suppression_level.nick().to_string(),
            )),
            "gain-control" => Some(PropValue::Bool(settings.gain_control)),
            "gain-control-mode" => Some(PropValue::Str(
                settings.gain_control_mode.nick().to_string(),
            )),
            "target-level-dbfs" => Some(PropValue::Int(settings.target_level_dbfs)),
            "compression-gain-db" => Some(PropValue::Int(settings.compression_gain_db)),
            "high-pass-filter" => Some(PropValue::Bool(settings.high_pass_filter)),
            _ => None,
        }
    }
}

impl PadTemplates for WebrtcDsp {
    fn pad_templates() -> Vec<PadTemplate> {
        pcm_pad_templates()
    }
}

static WEBRTCDSP_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "probe",
        PropKind::Str,
        "probe-name of the webrtcechoprobe on the playback path",
    )
    .with_default(DEFAULT_PROBE_NAME),
    PropertySpec::new("echo-cancel", PropKind::Bool, "enable the echo canceller")
        .with_default(ENABLED_TEXT),
    PropertySpec::new(
        "noise-suppression",
        PropKind::Bool,
        "enable noise suppression",
    )
    .with_default(ENABLED_TEXT),
    PropertySpec::new(
        "noise-suppression-level",
        PropKind::Str,
        "noise suppression aggressiveness",
    )
    .with_enum_values(NoiseSuppressionStrength::NICKS)
    .with_default(DEFAULT_NOISE_SUPPRESSION_LEVEL_TEXT),
    PropertySpec::new(
        "gain-control",
        PropKind::Bool,
        "enable automatic digital gain control",
    )
    .with_default(ENABLED_TEXT),
    PropertySpec::new(
        "gain-control-mode",
        PropKind::Str,
        "adaptive gain toward target-level-dbfs, or a fixed compression-gain-db",
    )
    .with_enum_values(GainControlMode::NICKS)
    .with_default(DEFAULT_GAIN_CONTROL_MODE_TEXT),
    PropertySpec::new(
        "target-level-dbfs",
        PropKind::Int,
        "adaptive-digital target level in dB below full scale",
    )
    .with_range(GAIN_MIN_TEXT, TARGET_LEVEL_DBFS_MAX_TEXT)
    .with_default(DEFAULT_TARGET_LEVEL_DBFS_TEXT),
    PropertySpec::new(
        "compression-gain-db",
        PropKind::Int,
        "fixed-digital gain in dB ahead of the limiter",
    )
    .with_range(GAIN_MIN_TEXT, COMPRESSION_GAIN_DB_MAX_TEXT)
    .with_default(DEFAULT_COMPRESSION_GAIN_DB_TEXT),
    PropertySpec::new(
        "high-pass-filter",
        PropKind::Bool,
        "enable the high-pass filter",
    )
    .with_default(ENABLED_TEXT),
];
