//! Silence remover (`removesilence`). A voice-activity detector classifies each
//! buffer, and with `remove` on the silent ones are dropped. Preserves format,
//! channel count, and sample rate. CPU-only `no_std`.
//!
//! Matches GStreamer's `removesilence` and the `vad_private.c` detector behind
//! it. A buffer is voice when its smoothed power is above `threshold` and its
//! zero-crossing count is negative, meaning fewer than half of the adjacent
//! sample pairs in the last [`VAD_WINDOW_SAMPLES`] change sign. The power is an
//! exponential average with weight [`VAD_POWER_ALPHA`], and the threshold in dB
//! is read as the reference reads it, `10^trunc(dB / 10)` of full-scale mean
//! square, so -60 and -65 dB both mean the same 1e-6. Only a voice-to-silence
//! transition waits: it needs `hysteresis` samples of agreement, while silence
//! to voice switches at once.
//!
//! The reference posts a `silence_detected` / `silence_finished` element
//! message on each transition, which `silent` turns off. g2g's bus has no
//! element message with named fields, so the two are posted as
//! [`BusMessage::Info`] naming the transition and the buffer's pts, and counted
//! in [`silence_transitions`](RemoveSilence::silence_transitions) for a caller
//! that attached no bus. `silent` gates both.
//!
//! The reference's detector works on the S16 samples directly; every g2g
//! audiofx filter works in f32, so the power runs on the normalized signal,
//! which is where the reference's fixed-point rounding is left behind.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::{
    AsyncElement, AudioFormat, BusHandle, BusMessage, Caps, CapsConstraint, ConfigureOutcome,
    ElementMetadata, G2gError, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec,
};

use crate::audiofx;
use crate::mathf;

/// Samples the zero-crossing count runs over, the reference's VAD ring.
pub const VAD_WINDOW_SAMPLES: usize = 256;

/// Weight of the newest sample in the power average, the reference's
/// `VAD_POWER_ALPHA` of 0x0800 in Q16.
pub const VAD_POWER_ALPHA: f64 = 2048.0 / 65536.0;

/// Weight of the accumulated power, the reference's `0xFFFF - VAD_POWER_ALPHA`
/// in Q16. It is a hair under `1 - VAD_POWER_ALPHA`, so the average leaks.
const VAD_POWER_DECAY: f64 = 63487.0 / 65536.0;

/// A buffer is voice only when the zero-crossing count is under this, i.e.
/// fewer sign changes than steady stretches.
const VAD_ZERO_CROSSING_THRESHOLD: i64 = 0;

/// The reference's defaults and bounds.
const DEFAULT_REMOVE: bool = false;
const DEFAULT_HYSTERESIS: u64 = 480;
const HYSTERESIS_MIN: u64 = 1;
const DEFAULT_THRESHOLD_DB: i64 = -60;
const THRESHOLD_DB_MIN: i64 = -70;
const THRESHOLD_DB_MAX: i64 = 70;
const DEFAULT_SQUASH: bool = false;
const DEFAULT_SILENT: bool = true;
const DEFAULT_MINIMUM_SILENCE_BUFFERS: u64 = 0;
const MINIMUM_SILENCE_BUFFERS_MAX: u64 = 10_000;
const DEFAULT_MINIMUM_SILENCE_TIME_NS: u64 = 0;
const MINIMUM_SILENCE_TIME_NS_MAX: u64 = 10_000_000_000;

const DEFAULT_REMOVE_TEXT: &str = "false";
const DEFAULT_HYSTERESIS_TEXT: &str = "480";
const DEFAULT_THRESHOLD_DB_TEXT: &str = "-60";
const DEFAULT_SQUASH_TEXT: &str = "false";
const DEFAULT_SILENT_TEXT: &str = "true";
const DEFAULT_MINIMUM_SILENCE_BUFFERS_TEXT: &str = "0";
const DEFAULT_MINIMUM_SILENCE_TIME_TEXT: &str = "0";

/// The two transitions the reference names its element messages after.
const SILENCE_DETECTED_MESSAGE: &str = "silence_detected";
const SILENCE_FINISHED_MESSAGE: &str = "silence_finished";

/// What the detector decided about the last buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadState {
    Silence,
    Voice,
}

/// The reference's voice-activity detector: a leaky power average plus a
/// zero-crossing count over a fixed sample window.
#[derive(Debug)]
struct Vad {
    /// Whether each of the last [`VAD_WINDOW_SAMPLES`] samples was negative, a
    /// ring indexed by `written`.
    signs: [bool; VAD_WINDOW_SAMPLES],
    written: u64,
    state: VadState,
    /// Samples the current voice-to-silence transition has agreed for.
    pending_samples: u64,
    power: f64,
    /// Mean-square power above which a buffer can be voice.
    threshold: f64,
}

impl Vad {
    fn new(threshold_db: i64) -> Self {
        Self {
            signs: [false; VAD_WINDOW_SAMPLES],
            written: 0,
            state: VadState::Silence,
            pending_samples: 0,
            power: 0.0,
            threshold: threshold_from_db(threshold_db),
        }
    }

    fn set_threshold(&mut self, threshold_db: i64) {
        self.threshold = threshold_from_db(threshold_db);
    }

    fn reset(&mut self) {
        self.signs = [false; VAD_WINDOW_SAMPLES];
        self.written = 0;
        self.state = VadState::Silence;
        self.pending_samples = 0;
        self.power = 0.0;
    }

    /// Sign changes minus steady pairs across the window: negative when the
    /// signal crosses zero rarely, which is what the reference reads as voice.
    fn zero_crossings(&self) -> i64 {
        let window = VAD_WINDOW_SAMPLES as u64;
        let filled = self.written.min(window);
        if filled < 2 {
            return 0;
        }
        let start = ((self.written - filled) % window) as usize;
        let mut sum = 0i64;
        for step in 0..filled as usize - 1 {
            let older = self.signs[(start + step) % VAD_WINDOW_SAMPLES];
            let newer = self.signs[(start + step + 1) % VAD_WINDOW_SAMPLES];
            sum += if older != newer { 1 } else { -1 };
        }
        sum
    }

    /// Classify one buffer of interleaved samples.
    fn update(&mut self, samples: &[f32], hysteresis: u64) -> VadState {
        for sample in samples {
            let value = *sample as f64;
            self.power = VAD_POWER_ALPHA * value * value + VAD_POWER_DECAY * self.power;
            self.signs[(self.written % VAD_WINDOW_SAMPLES as u64) as usize] = *sample < 0.0;
            self.written += 1;
        }

        let observed =
            if self.power > self.threshold && self.zero_crossings() < VAD_ZERO_CROSSING_THRESHOLD {
                VadState::Voice
            } else {
                VadState::Silence
            };

        if observed == self.state {
            self.pending_samples = 0;
            return self.state;
        }
        // Silence to voice switches at once; the other way has to hold.
        if self.state == VadState::Voice {
            self.pending_samples += samples.len() as u64;
            if self.pending_samples >= hysteresis {
                self.state = observed;
                self.pending_samples = 0;
            }
        } else {
            self.state = observed;
            self.pending_samples = 0;
        }
        self.state
    }
}

/// The mean-square power a dB threshold names, read the reference's way: the dB
/// value is divided by ten and truncated toward zero before the exponent, so
/// -60 and -65 dB are the same threshold.
fn threshold_from_db(threshold_db: i64) -> f64 {
    mathf::powf(10.0, (threshold_db / 10) as f64)
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::removesilence::RemoveSilence;
///
/// let strip = RemoveSilence::new().with_remove(true).with_squash(true);
/// ```
#[derive(Debug)]
pub struct RemoveSilence {
    remove: bool,
    hysteresis: u64,
    threshold_db: i64,
    squash: bool,
    silent: bool,
    minimum_silence_buffers: u64,
    minimum_silence_time_ns: u64,
    vad: Vad,
    /// Consecutive silent buffers and their total duration, the two
    /// `minimum-silence-*` counters.
    consecutive_silence_buffers: u64,
    consecutive_silence_time_ns: u64,
    silence_detected: bool,
    /// Total duration dropped so far, subtracted from every pts under `squash`.
    timestamp_offset_ns: u64,
    /// Silence-detected / silence-finished events posted so far, for a caller
    /// that attached no bus.
    transitions: u64,
    bus: Option<BusHandle>,
    format: AudioFormat,
    caps: Option<Caps>,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl Default for RemoveSilence {
    fn default() -> Self {
        Self::new()
    }
}

impl RemoveSilence {
    pub fn new() -> Self {
        Self {
            remove: DEFAULT_REMOVE,
            hysteresis: DEFAULT_HYSTERESIS,
            threshold_db: DEFAULT_THRESHOLD_DB,
            squash: DEFAULT_SQUASH,
            silent: DEFAULT_SILENT,
            minimum_silence_buffers: DEFAULT_MINIMUM_SILENCE_BUFFERS,
            minimum_silence_time_ns: DEFAULT_MINIMUM_SILENCE_TIME_NS,
            vad: Vad::new(DEFAULT_THRESHOLD_DB),
            consecutive_silence_buffers: 0,
            consecutive_silence_time_ns: 0,
            silence_detected: false,
            timestamp_offset_ns: 0,
            transitions: 0,
            bus: None,
            format: AudioFormat::PcmS16Le,
            caps: None,
            last_caps: None,
            emitted: 0,
        }
    }

    /// Attach the pipeline bus the silence transitions are posted on.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    pub fn with_remove(mut self, remove: bool) -> Self {
        self.remove = remove;
        self
    }

    pub fn with_hysteresis(mut self, hysteresis: u64) -> Self {
        self.hysteresis = hysteresis.max(HYSTERESIS_MIN);
        self
    }

    pub fn with_threshold(mut self, threshold_db: i64) -> Self {
        self.threshold_db = threshold_db.clamp(THRESHOLD_DB_MIN, THRESHOLD_DB_MAX);
        self.vad.set_threshold(self.threshold_db);
        self
    }

    pub fn with_squash(mut self, squash: bool) -> Self {
        self.squash = squash;
        self
    }

    pub fn with_silent(mut self, silent: bool) -> Self {
        self.silent = silent;
        self
    }

    pub fn with_minimum_silence_buffers(mut self, buffers: u64) -> Self {
        self.minimum_silence_buffers = buffers.min(MINIMUM_SILENCE_BUFFERS_MAX);
        self
    }

    pub fn with_minimum_silence_time(mut self, time_ns: u64) -> Self {
        self.minimum_silence_time_ns = time_ns.min(MINIMUM_SILENCE_TIME_NS_MAX);
        self
    }

    /// Silence-detected and silence-finished events posted since the start, for
    /// a caller that attached no bus. Stays zero while `silent` is on.
    pub fn silence_transitions(&self) -> u64 {
        self.transitions
    }

    /// Report one silence transition, unless `silent` turns the reporting off.
    fn post_transition(&mut self, message: &str, pts_ns: u64) {
        if self.silent {
            return;
        }
        self.transitions += 1;
        if let Some(bus) = &self.bus {
            bus.try_post(BusMessage::Info(format!("{message} at {pts_ns} ns")));
        }
    }

    fn configure(&mut self, caps: &Caps) -> Result<(), G2gError> {
        let (format, _, _) = audiofx::accept_audio(caps, None)?;
        self.format = format;
        self.caps = Some(caps.clone());
        Ok(())
    }

    /// Whether enough consecutive silence has piled up to act on. With both
    /// minimums disabled every silent buffer counts.
    fn consecutive_silence_reached(&self) -> bool {
        if self.minimum_silence_buffers == 0 && self.minimum_silence_time_ns == 0 {
            return true;
        }
        (self.minimum_silence_buffers > 0
            && self.consecutive_silence_buffers >= self.minimum_silence_buffers)
            || (self.minimum_silence_time_ns > 0
                && self.consecutive_silence_time_ns >= self.minimum_silence_time_ns)
    }

    /// Classify one buffer and decide whether it is dropped. Updates the
    /// silence run and, under `squash`, the timestamp offset.
    fn judge(&mut self, samples: &[f32], pts_ns: u64, duration_ns: u64) -> bool {
        let state = self.vad.update(samples, self.hysteresis);
        if state == VadState::Voice {
            self.consecutive_silence_buffers = 0;
            self.consecutive_silence_time_ns = 0;
            if self.silence_detected {
                self.silence_detected = false;
                self.post_transition(SILENCE_FINISHED_MESSAGE, pts_ns);
            }
            return false;
        }

        self.consecutive_silence_buffers += 1;
        self.consecutive_silence_time_ns =
            self.consecutive_silence_time_ns.saturating_add(duration_ns);
        let reached = self.consecutive_silence_reached();
        if !self.silence_detected && reached {
            self.silence_detected = true;
            self.post_transition(SILENCE_DETECTED_MESSAGE, pts_ns);
        }
        if self.remove && reached {
            if self.squash {
                self.timestamp_offset_ns = self.timestamp_offset_ns.saturating_add(duration_ns);
            }
            return true;
        }
        false
    }

    fn reset(&mut self) {
        self.vad.reset();
        self.consecutive_silence_buffers = 0;
        self.consecutive_silence_time_ns = 0;
        self.silence_detected = false;
        self.timestamp_offset_ns = 0;
    }
}

impl AsyncElement for RemoveSilence {
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
        audiofx::accept_audio(upstream_caps, None)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        audiofx::passthrough_constraint(None)
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
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let caps = self.caps.clone().ok_or(G2gError::NotConfigured)?;
                    let src = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let samples = audiofx::decode(src, self.format);
                    if self.judge(&samples, frame.timing.pts_ns, frame.timing.duration_ns) {
                        return Ok(());
                    }

                    if self.last_caps.as_ref() != Some(&caps) {
                        out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                        self.last_caps = Some(caps);
                    }
                    let mut timing = frame.timing;
                    if self.squash {
                        timing.pts_ns = timing.pts_ns.saturating_sub(self.timestamp_offset_ns);
                        timing.dts_ns = timing.dts_ns.saturating_sub(self.timestamp_offset_ns);
                    }
                    let out_frame = Frame {
                        domain: frame.domain,
                        timing,
                        sequence: self.emitted,
                        meta: Default::default(),
                    };
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    self.configure(&c)?;
                }
                PipelinePacket::Flush => {
                    self.reset();
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
        REMOVESILENCE_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "RemoveSilence",
            "Filter/Effect/Audio",
            "Removes all the silence periods from the audio stream",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "remove" => self.remove = value.as_bool().ok_or(PropError::Type)?,
            "hysteresis" => {
                let hysteresis = value.as_uint().ok_or(PropError::Type)?;
                if hysteresis < HYSTERESIS_MIN {
                    return Err(PropError::Value);
                }
                self.hysteresis = hysteresis;
            }
            "threshold" => {
                self.threshold_db =
                    audiofx::int_in_range(value, THRESHOLD_DB_MIN, THRESHOLD_DB_MAX)?;
                self.vad.set_threshold(self.threshold_db);
            }
            "squash" => self.squash = value.as_bool().ok_or(PropError::Type)?,
            "silent" => self.silent = value.as_bool().ok_or(PropError::Type)?,
            "minimum-silence-buffers" => {
                let buffers = value.as_uint().ok_or(PropError::Type)?;
                if buffers > MINIMUM_SILENCE_BUFFERS_MAX {
                    return Err(PropError::Value);
                }
                self.minimum_silence_buffers = buffers;
            }
            "minimum-silence-time" => {
                let time_ns = value.as_uint().ok_or(PropError::Type)?;
                if time_ns > MINIMUM_SILENCE_TIME_NS_MAX {
                    return Err(PropError::Value);
                }
                self.minimum_silence_time_ns = time_ns;
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "remove" => Some(PropValue::Bool(self.remove)),
            "hysteresis" => Some(PropValue::Uint(self.hysteresis)),
            "threshold" => Some(PropValue::Int(self.threshold_db)),
            "squash" => Some(PropValue::Bool(self.squash)),
            "silent" => Some(PropValue::Bool(self.silent)),
            "minimum-silence-buffers" => Some(PropValue::Uint(self.minimum_silence_buffers)),
            "minimum-silence-time" => Some(PropValue::Uint(self.minimum_silence_time_ns)),
            _ => None,
        }
    }
}

static REMOVESILENCE_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "remove",
        PropKind::Bool,
        "drop the silent buffers instead of only detecting them",
    )
    .with_default(DEFAULT_REMOVE_TEXT),
    PropertySpec::new(
        "hysteresis",
        PropKind::Uint,
        "samples a voice-to-silence transition has to hold before it takes",
    )
    .with_range("1", "18446744073709551615")
    .with_default(DEFAULT_HYSTERESIS_TEXT),
    PropertySpec::new(
        "threshold",
        PropKind::Int,
        "silence threshold used on the internal VAD in dB",
    )
    .with_range("-70", "70")
    .with_default(DEFAULT_THRESHOLD_DB_TEXT),
    PropertySpec::new(
        "squash",
        PropKind::Bool,
        "retimestamp the kept buffers so removing silence leaves no gap",
    )
    .with_default(DEFAULT_SQUASH_TEXT),
    PropertySpec::new(
        "silent",
        PropKind::Bool,
        "stop counting silence-detected / silence-finished transitions",
    )
    .with_default(DEFAULT_SILENT_TEXT),
    PropertySpec::new(
        "minimum-silence-buffers",
        PropKind::Uint,
        "consecutive silent buffers before silence is removed, 0 disables",
    )
    .with_range("0", "10000")
    .with_default(DEFAULT_MINIMUM_SILENCE_BUFFERS_TEXT),
    PropertySpec::new(
        "minimum-silence-time",
        PropKind::Uint,
        "consecutive silence in nanoseconds before silence is removed, 0 disables",
    )
    .with_range("0", "10000000000")
    .with_default(DEFAULT_MINIMUM_SILENCE_TIME_TEXT),
];

impl PadTemplates for RemoveSilence {
    fn pad_templates() -> Vec<PadTemplate> {
        audiofx::default_pad_templates()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// A tone low enough that the zero-crossing count stays negative, which is
    /// what the reference's detector calls voice.
    fn tone(frames: usize) -> Vec<f32> {
        (0..frames)
            .map(|i| {
                let turns = i as f32 / 32.0;
                0.5 * mathf::sin_turns(turns)
            })
            .collect()
    }

    #[test]
    fn a_threshold_in_db_truncates_to_a_power_of_ten() {
        assert!((threshold_from_db(-60) - 1e-6).abs() < 1e-18);
        // -65 dB truncates the same way -60 does.
        assert_eq!(threshold_from_db(-65), threshold_from_db(-60));
    }

    #[test]
    fn silence_is_detected_and_a_tone_is_not() {
        let mut element = RemoveSilence::new();
        let quiet = vec![0.0f32; VAD_WINDOW_SAMPLES * 4];
        assert_eq!(
            element.vad.update(&quiet, DEFAULT_HYSTERESIS),
            VadState::Silence
        );
        let loud = tone(VAD_WINDOW_SAMPLES * 4);
        assert_eq!(
            element.vad.update(&loud, DEFAULT_HYSTERESIS),
            VadState::Voice
        );
    }

    #[test]
    fn remove_drops_only_the_silent_buffers() {
        let mut element = RemoveSilence::new().with_remove(true);
        let loud = tone(VAD_WINDOW_SAMPLES * 4);
        assert!(!element.judge(&loud, 0, 0), "a tone is kept");
        let quiet = vec![0.0f32; VAD_WINDOW_SAMPLES * 4];
        // the voice-to-silence transition waits out the hysteresis first.
        let mut dropped = false;
        for _ in 0..8 {
            dropped |= element.judge(&quiet, 0, 0);
        }
        assert!(dropped, "silence is dropped once the hysteresis is served");
    }

    #[test]
    fn detection_without_remove_keeps_every_buffer() {
        let mut element = RemoveSilence::new();
        let quiet = vec![0.0f32; VAD_WINDOW_SAMPLES * 4];
        for _ in 0..8 {
            assert!(!element.judge(&quiet, 0, 0));
        }
        assert!(element.silence_detected, "still reported");
    }

    #[test]
    fn a_minimum_silence_run_has_to_be_reached() {
        let mut element = RemoveSilence::new()
            .with_remove(true)
            .with_minimum_silence_buffers(4);
        let quiet = vec![0.0f32; VAD_WINDOW_SAMPLES];
        assert!(
            !element.judge(&quiet, 0, 0),
            "the first buffer is short of the run"
        );
        assert_eq!(element.consecutive_silence_buffers, 1);
    }

    #[test]
    fn squash_accumulates_the_dropped_duration() {
        let mut element = RemoveSilence::new().with_remove(true).with_squash(true);
        let quiet = vec![0.0f32; VAD_WINDOW_SAMPLES];
        let duration_ns = 10_000_000;
        assert!(element.judge(&quiet, 0, duration_ns));
        assert_eq!(element.timestamp_offset_ns, duration_ns);
    }
}
