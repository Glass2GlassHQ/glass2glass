//! Channel interleaver (`interleave`), the fan-in that turns N mono PCM inputs
//! into one N-channel stream: input pad `i` becomes channel `i` of every output
//! sample frame. Nothing here resamples or converts, so every pad must carry the
//! same format and sample rate; a pad that does not is rejected in negotiation
//! and at `configure_pipeline`. CPU-only, `no_std` baseline.
//!
//! A fan-in declares its merged output caps before any input pad negotiates, so
//! the PCM shape is set on the element rather than learned from the pads:
//! `interleave format=F32LE rate=44100`. The defaults match `audiotestsrc` and
//! `audiomixer`'s nominal output (S16LE, 48 kHz).
//!
//! Alignment is by queued samples, not timestamps: each pad's bytes queue, every
//! push emits the sample prefix all pads have delivered, and the remainder stays
//! queued (bounded by the runner's per-pad link capacity, as `audiomixer`'s
//! accumulator is). Output pts runs off the emitted sample count from the first
//! pad's first pts, so a rate that does not divide a second never drifts. A pad
//! at `Eos` contributes silence from there on, so the output spans the longest
//! pad.
//!
//! gst's `channel-positions` / `channel-positions-from-input` are not exposed:
//! input pad `i` always becomes interleaved channel `i`, and ascending bit order
//! is the interleave order of every [`ChannelLayout`](g2g_core::ChannelLayout),
//! so pad `i` feeds the `i`-th speaker of whatever layout the output caps carry
//! (the count's [`default_for`](g2g_core::ChannelLayout::default_for)
//! convention when they carry none). The element declares no layout of its own,
//! since the pad order is a convention rather than something a stream told it;
//! a downstream `channel-mask` pins the positions when they matter.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError,
    MemoryDomain, MultiInputElement, OutputSink, PipelinePacket, PropError, PropValue,
    PropertySpec,
};

use crate::audioconvert::{
    get_pcm_shape_property, sample_bytes, samples_to_ns, set_pcm_shape_property, silence_byte,
    DEFAULT_PCM_FORMAT, DEFAULT_PCM_RATE, PCM_SHAPE_PROPS,
};

/// # Example
///
/// ```no_run
/// use g2g_plugins::interleave::Interleave;
///
/// // two mono pads in, one stereo stream out.
/// let element = Interleave::new(2);
/// ```
#[derive(Debug)]
pub struct Interleave {
    inputs: usize,
    format: AudioFormat,
    sample_rate: u32,
    /// Per pad: samples delivered and not yet emitted, as raw bytes.
    queued: Vec<Vec<u8>>,
    done: Vec<bool>,
    /// Timestamp of output sample 0, from the first pad's first frame.
    base_pts: Option<u64>,
    emitted_samples: u64,
    emitted: u64,
}

impl Interleave {
    /// An interleaver with `inputs` mono input pads, producing `inputs` channels
    /// at the default shape. Panics if `inputs` is zero (a fan-in needs a pad).
    pub fn new(inputs: usize) -> Self {
        assert!(
            inputs > 0 && inputs <= u8::MAX as usize,
            "Interleave takes 1 to {} inputs, one per output channel",
            u8::MAX
        );
        Self {
            inputs,
            format: DEFAULT_PCM_FORMAT,
            sample_rate: DEFAULT_PCM_RATE,
            queued: vec![Vec::new(); inputs],
            done: vec![false; inputs],
            base_pts: None,
            emitted_samples: 0,
            emitted: 0,
        }
    }

    pub fn with_format(mut self, format: AudioFormat) -> Self {
        self.format = format;
        self
    }

    pub fn with_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// The caps one input pad carries: mono, in this element's declared format
    /// and rate.
    fn input_caps(&self) -> Caps {
        Caps::Audio {
            format: self.format,
            channels: 1,
            sample_rate: self.sample_rate,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }

    fn accept_input(&self, caps: &Caps) -> Result<(), G2gError> {
        if caps == &self.input_caps() {
            Ok(())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    /// Samples the interleave can commit now: the prefix every still-open pad has
    /// delivered, or, once every pad is at `Eos`, everything left (the shorter
    /// pads drained with silence).
    fn ready_samples(&self) -> usize {
        let per_sample = sample_bytes(self.format);
        if per_sample == 0 {
            return 0;
        }
        let queued_samples = |pad: usize| self.queued[pad].len() / per_sample;
        match (0..self.inputs)
            .filter(|&pad| !self.done[pad])
            .map(queued_samples)
            .min()
        {
            Some(open_min) => open_min,
            None => (0..self.inputs).map(queued_samples).max().unwrap_or(0),
        }
    }

    async fn emit_ready(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let samples = self.ready_samples();
        if samples == 0 {
            return Ok(());
        }
        let per_sample = sample_bytes(self.format);
        let stride = per_sample * self.inputs;
        // Silence pre-fill, so a pad that ran out (at Eos) needs no second pass.
        let mut bytes = vec![silence_byte(self.format); samples * stride].into_boxed_slice();
        for pad in 0..self.inputs {
            let taken = (self.queued[pad].len() / per_sample).min(samples);
            for sample in 0..taken {
                let src = sample * per_sample;
                let dst = sample * stride + pad * per_sample;
                bytes[dst..dst + per_sample]
                    .copy_from_slice(&self.queued[pad][src..src + per_sample]);
            }
            self.queued[pad].drain(0..taken * per_sample);
        }

        let base = self.base_pts.unwrap_or(0);
        let pts_ns = base.saturating_add(samples_to_ns(self.emitted_samples, self.sample_rate));
        let end_ns = base.saturating_add(samples_to_ns(
            self.emitted_samples + samples as u64,
            self.sample_rate,
        ));
        self.emitted_samples += samples as u64;
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes)),
            FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                duration_ns: end_ns - pts_ns,
                ..FrameTiming::default()
            },
            self.emitted,
        );
        self.emitted += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }
}

impl MultiInputElement for Interleave {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Shuffles host samples, so every pad takes system frames only. The
    /// allocation cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn input_count(&self) -> usize {
        self.inputs
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&self.input_caps())
    }

    /// Every pad carries the same mono shape, so a source that would deliver a
    /// different format, rate, or channel count fails to negotiate rather than
    /// being silently converted.
    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(self.input_caps()))
    }

    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        Ok(CapsConstraint::Produces(CapsSet::one(self.output_caps()?)))
    }

    fn configure_pipeline(
        &mut self,
        _input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        self.accept_input(absolute_caps)?;
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(Caps::Audio {
            format: self.format,
            channels: self.inputs as u8,
            sample_rate: self.sample_rate,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        })
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Audio interleaver",
            "Filter/Converter/Audio",
            "Merges N mono PCM inputs into one N-channel interleaved stream (the gst `interleave` analog).",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let bytes = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let per_sample = sample_bytes(self.format);
                    if per_sample == 0 || bytes.len() % per_sample != 0 {
                        return Err(G2gError::CapsMismatch);
                    }
                    if self.base_pts.is_none() && input == 0 {
                        self.base_pts = frame.timing.pts();
                    }
                    self.queued[input].extend_from_slice(bytes);
                }
                // A per-input Eos is informational: the runner emits the single
                // merged Eos once every input ends. The finished pad contributes
                // silence from here, so the output spans the longest pad.
                PipelinePacket::Eos => self.done[input] = true,
                PipelinePacket::Flush => {
                    self.queued.iter_mut().for_each(Vec::clear);
                    self.base_pts = None;
                    self.emitted_samples = 0;
                }
                // A pad may only ever carry the declared mono shape; a mid-stream
                // change to anything else would break the interleave.
                PipelinePacket::CapsChanged(caps) => self.accept_input(&caps)?,
                PipelinePacket::Segment(_) => {}
                // future PipelinePacket variants: no-op.
                _ => {}
            }
            self.emit_ready(out).await
        })
    }

    /// The declared output shape (M1072). gst takes format and rate from the
    /// sink pads' caps, which a fan-in here cannot read before it declares its
    /// output.
    fn properties(&self) -> &'static [PropertySpec] {
        PCM_SHAPE_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        set_pcm_shape_property(&mut self.format, &mut self.sample_rate, name, value)
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        get_pcm_shape_property(self.format, self.sample_rate, name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_declared_output_carries_one_channel_per_pad() {
        let element = Interleave::new(6).with_format(AudioFormat::PcmF32Le);
        assert_eq!(
            element.output_caps().unwrap(),
            Caps::Audio {
                format: AudioFormat::PcmF32Le,
                channels: 6,
                sample_rate: DEFAULT_PCM_RATE,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            }
        );
    }
}
