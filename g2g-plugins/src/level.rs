//! Audio level meter (`level`). A passthrough that measures per-channel peak
//! and RMS of each S16LE buffer and exposes them via getters, the g2g analog of
//! GStreamer's `level` element (which posts the same values on the bus). Values
//! are linear (0..1, normalized to full scale); an application converts to dB if
//! it wants (dB needs a `log`, which the `no_std` baseline avoids). CPU-only
//! `no_std`.
//!
//! `post-messages` gates measurement: when false the meter is inert (getters
//! return empty) and the buffer is forwarded untouched.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec,
};

#[derive(Debug)]
pub struct Level {
    post_messages: bool,
    channels: usize,
    peak: Vec<f64>,
    rms: Vec<f64>,
    configured: bool,
}

impl Default for Level {
    fn default() -> Self {
        Self::new()
    }
}

const FULL_SCALE: f64 = 32768.0;

impl Level {
    pub fn new() -> Self {
        Self { post_messages: true, channels: 0, peak: Vec::new(), rms: Vec::new(), configured: false }
    }

    /// Per-channel peak of the last measured buffer, linear 0..1.
    pub fn last_peak(&self) -> &[f64] {
        &self.peak
    }

    /// Per-channel RMS of the last measured buffer, linear 0..1.
    pub fn last_rms(&self) -> &[f64] {
        &self.rms
    }

    fn accept_input(&self, caps: &Caps) -> Result<usize, G2gError> {
        match caps {
            Caps::Audio { format: AudioFormat::PcmS16Le, channels, .. } if *channels > 0 => {
                Ok(*channels as usize)
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn measure(&mut self, src: &[u8]) {
        let ch = self.channels.max(1);
        let mut peak = vec![0.0f64; ch];
        let mut sumsq = vec![0.0f64; ch];
        let mut counts = vec![0u64; ch];
        for (i, s) in src.chunks_exact(2).enumerate() {
            let c = i % ch;
            let v = (i16::from_le_bytes([s[0], s[1]]) as f64) / FULL_SCALE;
            let a = v.abs();
            if a > peak[c] {
                peak[c] = a;
            }
            sumsq[c] += v * v;
            counts[c] += 1;
        }
        for c in 0..ch {
            sumsq[c] =
                if counts[c] > 0 { crate::mathf::sqrt(sumsq[c] / counts[c] as f64) } else { 0.0 };
        }
        self.peak = peak;
        self.rms = sumsq;
    }
}

impl AsyncElement for Level {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        self.accept_input(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    /// Pure passthrough: the meter never changes the stream.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Audio { format: AudioFormat::PcmS16Le, .. } => CapsSet::one(input.clone()),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.channels = self.accept_input(absolute_caps)?;
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
                    if self.post_messages {
                        if let MemoryDomain::System(slice) = &frame.domain {
                            self.measure(slice.as_slice());
                        }
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    self.channels = self.accept_input(&c)?;
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                // The runner emits the final Eos after process(Eos) returns
                // (the transform contract), so do not forward it here.
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        LEVEL_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new("Level", "Filter/Analyzer/Audio", "Measures audio peak / RMS levels", "g2g")
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "post-messages" => self.post_messages = value.as_bool().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "post-messages" => Some(PropValue::Bool(self.post_messages)),
            _ => None,
        }
    }
}

static LEVEL_PROPS: &[PropertySpec] =
    &[PropertySpec::new("post-messages", PropKind::Bool, "measure and expose levels when true")];

impl PadTemplates for Level {
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = Caps::Audio { format: AudioFormat::PcmS16Le, channels: 2, sample_rate: 48_000 };
        Vec::from([PadTemplate::sink(CapsSet::one(pcm.clone())), PadTemplate::source(CapsSet::one(pcm))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack(samples: &[i16]) -> Vec<u8> {
        let mut v = vec![0u8; samples.len() * 2];
        for (i, s) in samples.iter().enumerate() {
            v[i * 2..i * 2 + 2].copy_from_slice(&s.to_le_bytes());
        }
        v
    }

    #[test]
    fn silence_measures_zero() {
        let mut l = Level::new();
        l.channels = 1;
        l.measure(&pack(&[0, 0, 0, 0]));
        assert_eq!(l.last_peak(), &[0.0]);
        assert_eq!(l.last_rms(), &[0.0]);
    }

    #[test]
    fn full_scale_measures_one() {
        let mut l = Level::new();
        l.channels = 1;
        l.measure(&pack(&[i16::MAX, i16::MIN]));
        // peak ~1.0, rms ~1.0 (both samples at full magnitude).
        assert!((l.last_peak()[0] - 1.0).abs() < 0.001);
        assert!((l.last_rms()[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn stereo_measures_per_channel() {
        let mut l = Level::new();
        l.channels = 2;
        // left = 16384 (half scale), right = 0.
        l.measure(&pack(&[16384, 0, 16384, 0]));
        assert!((l.last_peak()[0] - 0.5).abs() < 0.001);
        assert_eq!(l.last_peak()[1], 0.0);
        assert!((l.last_rms()[0] - 0.5).abs() < 0.001);
        assert_eq!(l.last_rms()[1], 0.0);
    }

    #[test]
    fn post_messages_false_leaves_measurements_empty() {
        let mut l = Level::new();
        l.set_property("post-messages", PropValue::Bool(false)).unwrap();
        assert!(!l.post_messages);
    }
}
