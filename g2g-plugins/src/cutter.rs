//! Silence detector (`cutter`). A passthrough that watches the S16LE signal
//! level and flips between "above" (signal present) and "below" (silence) state,
//! the g2g analog of GStreamer's `cutter` (which posts the transitions on the
//! bus). The current state and transition counts are exposed via getters.
//! CPU-only `no_std`.
//!
//! `threshold` is a linear RMS level (0..1, full scale), compared without a
//! `sqrt` by squaring the threshold. Rising above the threshold flips to "above"
//! immediately; falling below it flips to "below" only after `run-length` ns of
//! continuous silence (debounce), matching `cutter`.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec,
};

/// The debounced silence state machine. Kept separate from the element so the
/// transition logic is unit-testable without a pipeline.
#[derive(Debug)]
struct CutterState {
    threshold_sq: f64,
    run_length_ns: u64,
    silent: bool,
    silence_ns: u64,
}

impl CutterState {
    fn new(threshold: f64, run_length_ns: u64) -> Self {
        Self { threshold_sq: threshold * threshold, run_length_ns, silent: true, silence_ns: 0 }
    }

    /// Feed one buffer's mean-square level and its duration. Returns `Some(true)`
    /// when it just went silent, `Some(false)` when it just went loud, `None` when
    /// the state is unchanged.
    fn push(&mut self, mean_square: f64, dur_ns: u64) -> Option<bool> {
        if mean_square >= self.threshold_sq {
            self.silence_ns = 0;
            if self.silent {
                self.silent = false;
                return Some(false);
            }
        } else {
            self.silence_ns = self.silence_ns.saturating_add(dur_ns);
            if !self.silent && self.silence_ns >= self.run_length_ns {
                self.silent = true;
                return Some(true);
            }
        }
        None
    }
}

#[derive(Debug)]
pub struct Cutter {
    threshold: f64,
    run_length_ns: u64,
    channels: u32,
    sample_rate: u32,
    state: CutterState,
    above_count: u64,
    below_count: u64,
    configured: bool,
}

impl Default for Cutter {
    fn default() -> Self {
        Self::new()
    }
}

impl Cutter {
    /// Threshold 0.001 (about -60 dB), 500ms debounce.
    pub fn new() -> Self {
        let threshold = 0.001;
        let run_length_ns = 500_000_000;
        Self {
            threshold,
            run_length_ns,
            channels: 0,
            sample_rate: 0,
            state: CutterState::new(threshold, run_length_ns),
            above_count: 0,
            below_count: 0,
            configured: false,
        }
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self.state = CutterState::new(self.threshold, self.run_length_ns);
        self
    }

    /// True while the input is below threshold (silence).
    pub fn is_silent(&self) -> bool {
        self.state.silent
    }

    /// Number of below->above (signal returned) transitions seen.
    pub fn above_count(&self) -> u64 {
        self.above_count
    }

    /// Number of above->below (fell silent) transitions seen.
    pub fn below_count(&self) -> u64 {
        self.below_count
    }

    fn accept_input(&self, caps: &Caps) -> Result<(u32, u32), G2gError> {
        match caps {
            Caps::Audio { format: AudioFormat::PcmS16Le, channels, sample_rate } if *channels > 0 => {
                Ok((*channels as u32, *sample_rate))
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn observe(&mut self, src: &[u8]) {
        let total = src.len() / 2;
        if total == 0 || self.sample_rate == 0 {
            return;
        }
        let mut sumsq = 0.0f64;
        for s in src.chunks_exact(2) {
            let v = (i16::from_le_bytes([s[0], s[1]]) as f64) / 32768.0;
            sumsq += v * v;
        }
        let mean_square = sumsq / total as f64;
        let frames = (total / self.channels.max(1) as usize) as u64;
        let dur_ns = frames.saturating_mul(1_000_000_000) / self.sample_rate.max(1) as u64;
        match self.state.push(mean_square, dur_ns) {
            Some(true) => self.below_count += 1,
            Some(false) => self.above_count += 1,
            None => {}
        }
    }
}

impl AsyncElement for Cutter {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        self.accept_input(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    /// Pure passthrough: the detector never changes the stream.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Audio { format: AudioFormat::PcmS16Le, .. } => CapsSet::one(input.clone()),
            _ => CapsSet::from_alternatives(alloc::vec::Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (channels, rate) = self.accept_input(absolute_caps)?;
        self.channels = channels;
        self.sample_rate = rate;
        self.configured = true;
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
                    if let MemoryDomain::System(slice) = &frame.domain {
                        self.observe(slice.as_slice());
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    let (channels, rate) = self.accept_input(&c)?;
                    self.channels = channels;
                    self.sample_rate = rate;
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                // The runner emits the final Eos after process(Eos) returns.
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        CUTTER_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new("Cutter", "Filter/Analyzer/Audio", "Detects silence in an audio stream", "g2g")
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "threshold" => self.threshold = value.as_double().ok_or(PropError::Type)?,
            "run-length" => self.run_length_ns = value.as_uint().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        // Rebuild the state machine's derived bounds, preserving the current
        // silent/accumulator so a live tweak does not spuriously flip state.
        let (silent, silence_ns) = (self.state.silent, self.state.silence_ns);
        self.state = CutterState::new(self.threshold, self.run_length_ns);
        self.state.silent = silent;
        self.state.silence_ns = silence_ns;
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "threshold" => Some(PropValue::Double(self.threshold)),
            "run-length" => Some(PropValue::Uint(self.run_length_ns)),
            _ => None,
        }
    }
}

static CUTTER_PROPS: &[PropertySpec] = &[
    PropertySpec::new("threshold", PropKind::Double, "linear RMS silence threshold, 0..1"),
    PropertySpec::new("run-length", PropKind::Uint, "continuous silence before 'below', ns"),
];

impl PadTemplates for Cutter {
    fn pad_templates() -> alloc::vec::Vec<PadTemplate> {
        let pcm = Caps::Audio { format: AudioFormat::PcmS16Le, channels: 2, sample_rate: 48_000 };
        alloc::vec::Vec::from([
            PadTemplate::sink(CapsSet::one(pcm.clone())),
            PadTemplate::source(CapsSet::one(pcm)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rises_above_immediately() {
        let mut st = CutterState::new(0.1, 1_000_000_000);
        // starts silent; a loud buffer flips to above at once.
        assert_eq!(st.push(0.25, 100_000_000), Some(false));
        assert!(!st.silent);
        // staying loud does not re-fire.
        assert_eq!(st.push(0.25, 100_000_000), None);
    }

    #[test]
    fn falls_below_only_after_run_length() {
        let mut st = CutterState::new(0.1, 1_000_000_000);
        st.push(0.25, 100_000_000); // go loud
                                    // 0.9s of silence: not yet below.
        assert_eq!(st.push(0.0, 900_000_000), None);
        assert!(!st.silent);
        // crossing 1s total flips to below.
        assert_eq!(st.push(0.0, 200_000_000), Some(true));
        assert!(st.silent);
    }

    #[test]
    fn brief_silence_does_not_flip() {
        let mut st = CutterState::new(0.1, 1_000_000_000);
        st.push(0.25, 100_000_000); // loud
        st.push(0.0, 500_000_000); // 0.5s silence
        assert_eq!(st.push(0.25, 100_000_000), None); // loud again resets
        assert!(!st.silent);
        assert_eq!(st.silence_ns, 0);
    }

    #[test]
    fn configure_rejects_non_s16le() {
        let mut c = Cutter::new();
        let bad = Caps::Audio { format: AudioFormat::PcmF32Le, channels: 2, sample_rate: 48_000 };
        assert_eq!(c.configure_pipeline(&bad).unwrap_err(), G2gError::CapsMismatch);
        let ok = Caps::Audio { format: AudioFormat::PcmS16Le, channels: 2, sample_rate: 48_000 };
        assert!(c.configure_pipeline(&ok).is_ok());
    }
}
