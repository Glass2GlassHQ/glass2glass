//! Audio echo / delay (`audioecho`). Adds a delayed, attenuated copy of the
//! signal back onto itself, with optional feedback, preserving format, channel
//! count, and sample rate. CPU-only `no_std`.
//!
//! Matches GStreamer's `audioecho`: `out = in + intensity * echo`, and the value
//! written into the delay line is `in + feedback * echo`, per channel. `delay`
//! and `max-delay` are in nanoseconds; the delay-line length is fixed at
//! `max-delay` when the caps are known, so `delay` can be tuned live up to that
//! bound. Only `PcmS16Le` is handled.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec,
};

/// The interleaved-S16LE delay line: a flat ring of `frames * channels` samples,
/// indexed by an absolute sample counter so each channel's echo stays phase
/// aligned (the per-channel delay is `delay_frames * channels` samples back).
#[derive(Debug)]
struct EchoState {
    ring: Vec<i16>,
    channels: usize,
    max_delay_frames: usize,
    pos: usize,
}

impl EchoState {
    fn new(channels: usize, max_delay_frames: usize) -> Self {
        let frames = max_delay_frames.max(1);
        Self {
            ring: vec![0i16; frames * channels],
            channels,
            max_delay_frames: frames,
            pos: 0,
        }
    }

    /// Process one interleaved S16LE buffer in place semantics (writes `dst`).
    fn process(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        delay_frames: usize,
        intensity: f64,
        feedback: f64,
    ) {
        let delay = delay_frames.clamp(1, self.max_delay_frames);
        let back = delay * self.channels;
        let len = self.ring.len();
        for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
            let input = i16::from_le_bytes([s[0], s[1]]) as f64;
            let echo = self.ring[(self.pos + len - back) % len] as f64;
            let out = clamp_i16(input + intensity * echo);
            self.ring[self.pos % len] = clamp_i16(input + feedback * echo);
            self.pos = (self.pos + 1) % len;
            d.copy_from_slice(&out.to_le_bytes());
        }
    }
}

fn clamp_i16(v: f64) -> i16 {
    let rounded = if v >= 0.0 { v + 0.5 } else { v - 0.5 };
    rounded.clamp(i16::MIN as f64, i16::MAX as f64) as i16
}

fn delay_frames(delay_ns: u64, sample_rate: u32) -> usize {
    ((delay_ns as u128 * sample_rate as u128) / 1_000_000_000u128) as usize
}

#[derive(Debug)]
pub struct AudioEcho {
    delay_ns: u64,
    max_delay_ns: u64,
    intensity: f64,
    feedback: f64,
    caps: Option<Caps>,
    state: Option<EchoState>,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl Default for AudioEcho {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioEcho {
    /// 1s max delay, 500ms delay, half-intensity echo, no feedback.
    pub fn new() -> Self {
        Self {
            delay_ns: 500_000_000,
            max_delay_ns: 1_000_000_000,
            intensity: 0.5,
            feedback: 0.0,
            caps: None,
            state: None,
            last_caps: None,
            emitted: 0,
        }
    }

    pub fn with_delay(mut self, delay_ns: u64) -> Self {
        self.delay_ns = delay_ns;
        self
    }

    pub fn with_intensity(mut self, intensity: f64) -> Self {
        self.intensity = intensity;
        self
    }

    pub fn with_feedback(mut self, feedback: f64) -> Self {
        self.feedback = feedback;
        self
    }

    fn accept_input(&self, caps: &Caps) -> Result<(u32, u32), G2gError> {
        match caps {
            Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels,
                sample_rate,
            } if *channels > 0 => Ok((*channels as u32, *sample_rate)),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn build_state(&mut self, caps: &Caps) -> Result<(), G2gError> {
        let (channels, rate) = self.accept_input(caps)?;
        let max_frames = delay_frames(self.max_delay_ns, rate).max(1);
        self.state = Some(EchoState::new(channels as usize, max_frames));
        self.caps = Some(caps.clone());
        Ok(())
    }
}

impl AsyncElement for AudioEcho {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        self.accept_input(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    /// Native `DerivedOutput`: the echo preserves format, channels, and rate.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Audio {
                format: AudioFormat::PcmS16Le,
                ..
            } => CapsSet::one(input.clone()),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.build_state(absolute_caps)?;
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
                    let rate = match &caps {
                        Caps::Audio { sample_rate, .. } => *sample_rate,
                        _ => return Err(G2gError::NotConfigured),
                    };
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let src = slice.as_slice();
                    let mut dst = vec![0u8; src.len()].into_boxed_slice();
                    let df = delay_frames(self.delay_ns, rate);
                    let (intensity, feedback) = (self.intensity, self.feedback);
                    let state = self.state.as_mut().ok_or(G2gError::NotConfigured)?;
                    state.process(src, &mut dst, df, intensity, feedback);

                    if self.last_caps.as_ref() != Some(&caps) {
                        out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                        self.last_caps = Some(caps);
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
                PipelinePacket::CapsChanged(c) => {
                    self.build_state(&c)?;
                }
                PipelinePacket::Flush => {
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
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
        AUDIOECHO_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Audio echo",
            "Filter/Effect/Audio",
            "Adds an echo / delay to an audio stream",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "delay" => self.delay_ns = value.as_uint().ok_or(PropError::Type)?,
            "max-delay" => self.max_delay_ns = value.as_uint().ok_or(PropError::Type)?,
            "intensity" => self.intensity = value.as_double().ok_or(PropError::Type)?,
            "feedback" => self.feedback = value.as_double().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "delay" => Some(PropValue::Uint(self.delay_ns)),
            "max-delay" => Some(PropValue::Uint(self.max_delay_ns)),
            "intensity" => Some(PropValue::Double(self.intensity)),
            "feedback" => Some(PropValue::Double(self.feedback)),
            _ => None,
        }
    }
}

/// `AudioEcho`'s settable properties. `max-delay` fixes the delay-line length,
/// so it only takes effect at the next negotiation.
static AUDIOECHO_PROPS: &[PropertySpec] = &[
    PropertySpec::new("delay", PropKind::Uint, "echo delay in ns (<= max-delay)"),
    PropertySpec::new("max-delay", PropKind::Uint, "delay-line length in ns"),
    PropertySpec::new("intensity", PropKind::Double, "echo mix, 0..1"),
    PropertySpec::new("feedback", PropKind::Double, "delay-line feedback, 0..1"),
];

impl PadTemplates for AudioEcho {
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(pcm.clone())),
            PadTemplate::source(CapsSet::one(pcm)),
        ])
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

    fn unpack(bytes: &[u8]) -> Vec<i16> {
        bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect()
    }

    #[test]
    fn delay_frames_maps_ns_to_samples() {
        assert_eq!(delay_frames(1_000_000_000, 48_000), 48_000);
        assert_eq!(delay_frames(500_000_000, 48_000), 24_000);
        assert_eq!(delay_frames(0, 48_000), 0);
    }

    #[test]
    fn mono_echo_delays_and_mixes() {
        // 1 channel, delay of 2 frames, intensity 0.5, no feedback. An impulse of
        // 1000 at t=0 reappears attenuated (500) two samples later.
        let mut st = EchoState::new(1, 8);
        let src = pack(&[1000, 0, 0, 0, 0]);
        let mut dst = vec![0u8; src.len()];
        st.process(&src, &mut dst, 2, 0.5, 0.0);
        assert_eq!(unpack(&dst), [1000, 0, 500, 0, 0]);
    }

    #[test]
    fn feedback_re_injects_the_echo() {
        // With feedback the delay line accumulates: the echo at t=2 (500) is
        // written back, so a further echo appears at t=4 (250).
        let mut st = EchoState::new(1, 8);
        let src = pack(&[1000, 0, 0, 0, 0, 0, 0]);
        let mut dst = vec![0u8; src.len()];
        st.process(&src, &mut dst, 2, 0.5, 0.5);
        assert_eq!(unpack(&dst), [1000, 0, 500, 0, 250, 0, 125]);
    }

    #[test]
    fn stereo_keeps_channels_separate() {
        // 2 channels, delay 1 frame. Left impulse must echo only into left.
        let mut st = EchoState::new(2, 4);
        let src = pack(&[1000, 2000, 0, 0, 0, 0]);
        let mut dst = vec![0u8; src.len()];
        st.process(&src, &mut dst, 1, 0.5, 0.0);
        // frame 1 echoes frame 0: L += 500, R += 1000.
        assert_eq!(unpack(&dst), [1000, 2000, 500, 1000, 0, 0]);
    }

    #[test]
    fn configure_rejects_non_s16le() {
        let mut e = AudioEcho::new();
        let bad = Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 2,
            sample_rate: 48_000,
        };
        assert_eq!(
            e.configure_pipeline(&bad).unwrap_err(),
            G2gError::CapsMismatch
        );
        let ok = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
        };
        assert!(e.configure_pipeline(&ok).is_ok());
    }
}
