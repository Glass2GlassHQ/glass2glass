//! Audio mixer (`audiomixer`), the summing fan-in counterpart to the
//! passthrough `InterleaveMux`. Combines N interleaved S16LE inputs into one
//! output by adding samples and clamping to the i16 range. CPU-only `no_std`.
//!
//! Alignment is PTS-based: each input buffer lands on a shared output timeline
//! at the sample-frame its PTS maps to, so streams that arrive out of step still
//! mix at the right instant. Overlapping regions sum; a gap between an input's
//! buffers (a PTS jump) mixes as silence for that input. A buffer whose PTS is
//! at or behind where the input already wrote is treated as continuous and
//! appended (covers zero / duplicate PTS). A span is emitted once every open
//! input has covered it; an input at EOS stops driving the timeline (its place
//! mixed as silence). Sample-rate and channel reconciliation stay upstream
//! (`audioresample` / `audioconvert`), matching gst `audiomixer`.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError,
    MemoryDomain, MultiInputElement, OutputSink, PipelinePacket,
};

#[derive(Debug)]
pub struct AudioMixer {
    inputs: usize,
    output: Caps,
    channels: usize,
    sample_rate: u32,
    done: Vec<bool>,
    /// Per input: the absolute sample-frame up to which the input has delivered
    /// (where its next buffer appends when continuous).
    end_frame: Vec<u64>,
    /// Interleaved i32 accumulator of summed samples, `acc[0]` at `acc_start`.
    acc: Vec<i32>,
    /// Absolute sample-frame index of the first slot in `acc`.
    acc_start: u64,
    /// Whether `acc_start` has been anchored to the first buffer's position.
    started: bool,
    /// Whether any span has been emitted (locks `acc_start` from moving down).
    emitted_any: bool,
    emitted: u64,
}

impl AudioMixer {
    /// A mixer with `inputs` input pads producing `output` caps. `output` is the
    /// fixed merged-output caps (the negotiated S16LE shape); its channel count
    /// and sample rate drive the PTS-to-sample-frame mapping.
    pub fn new(inputs: usize, output: Caps) -> Self {
        assert!(inputs > 0, "AudioMixer needs at least one input");
        let (channels, sample_rate) = match &output {
            Caps::Audio { channels, sample_rate, .. } => (*channels as usize, *sample_rate),
            _ => panic!("AudioMixer output must be Caps::Audio"),
        };
        Self {
            inputs,
            output,
            channels,
            sample_rate,
            done: vec![false; inputs],
            end_frame: vec![0; inputs],
            acc: Vec::new(),
            acc_start: 0,
            started: false,
            emitted_any: false,
            emitted: 0,
        }
    }

    /// Place an input buffer on the timeline: sum its samples into `acc` at the
    /// sample-frame its PTS maps to, and advance the input's coverage.
    fn accumulate_frame(&mut self, input: usize, pts_ns: u64, bytes: &[u8]) {
        let bytes_per_frame = self.channels * 2;
        if bytes_per_frame == 0 {
            return;
        }
        let n_frames = bytes.len() / bytes_per_frame;
        if n_frames == 0 {
            return;
        }
        // Continuous append when the PTS is at or behind the input's own
        // position (zero / duplicate / slightly-behind PTS); honor a forward
        // gap otherwise.
        let pts_frame = pts_to_frame(pts_ns, self.sample_rate);
        let start = pts_frame.max(self.end_frame[input]);

        if !self.started {
            self.acc_start = start;
            self.started = true;
        } else if start < self.acc_start && !self.emitted_any {
            // A later-arriving input opened earlier than the current origin:
            // grow the accumulator downward. Only reachable before the first
            // emit (afterwards every input's coverage is >= acc_start).
            let shift = (self.acc_start - start) as usize * self.channels;
            let mut grown = vec![0i32; shift];
            grown.extend_from_slice(&self.acc);
            self.acc = grown;
            self.acc_start = start;
        }

        let off = (start - self.acc_start) as usize * self.channels;
        let needed = off + n_frames * self.channels;
        if self.acc.len() < needed {
            self.acc.resize(needed, 0);
        }
        accumulate(&mut self.acc[off..], &bytes[..n_frames * bytes_per_frame]);
        self.end_frame[input] = start + n_frames as u64;
    }

    /// Emit every span the timeline can now safely commit: contiguous from
    /// `acc_start` up to the earliest coverage among still-open inputs (or all
    /// remaining samples once every input is done).
    async fn emit_ready(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let safe = match (0..self.inputs)
            .filter(|&i| !self.done[i])
            .map(|i| self.end_frame[i])
            .min()
        {
            Some(open_min) => open_min,
            None => self.end_frame.iter().copied().max().unwrap_or(self.acc_start),
        };
        if !self.started || safe <= self.acc_start {
            return Ok(());
        }

        let n_samples = ((safe - self.acc_start) as usize * self.channels).min(self.acc.len());
        let pts_ns = frame_to_ns(self.acc_start, self.sample_rate);
        let duration_ns = frame_to_ns(safe, self.sample_rate).saturating_sub(pts_ns);
        let bytes = clamp_to_bytes(&self.acc[..n_samples]);
        self.acc.drain(0..n_samples);
        self.acc_start = safe;
        self.emitted_any = true;

        let out_frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes)),
            timing: FrameTiming { pts_ns, duration_ns, ..FrameTiming::default() },
            sequence: self.emitted,
            meta: Default::default(),
        };
        self.emitted += 1;
        out.push(PipelinePacket::DataFrame(out_frame)).await?;
        Ok(())
    }
}

impl MultiInputElement for AudioMixer {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.inputs
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// Each input accepts whatever upstream offers; `configure_pipeline` enforces
    /// S16LE, matching `InterleaveMux`'s native per-input wildcard.
    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    /// The merged output caps are fixed at construction (the negotiated S16LE
    /// shape), like `InterleaveMux`.
    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        Ok(CapsConstraint::Produces(CapsSet::one(self.output.clone())))
    }

    fn configure_pipeline(
        &mut self,
        _input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::Audio { format: AudioFormat::PcmS16Le, .. } => Ok(ConfigureOutcome::Accepted),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(self.output.clone())
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Audio mixer",
            "Filter/Audio",
            "Sums N time-aligned PCM audio inputs into one output stream (the gst `audiomixer` analog).",
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
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let pts_ns = frame.timing.pts_ns;
                    let bytes = slice.as_slice();
                    self.accumulate_frame(input, pts_ns, bytes);
                }
                // A per-input Eos is informational: the runner emits the single
                // merged Eos once every input ends (the InterleaveMux contract).
                // The finished input stops driving the timeline; its place mixes
                // as silence past its coverage.
                PipelinePacket::Eos => self.done[input] = true,
                PipelinePacket::Flush => {
                    self.acc.clear();
                    self.acc_start = 0;
                    self.started = false;
                    self.end_frame.iter_mut().for_each(|e| *e = 0);
                }
                // Output caps are fixed; per-input caps / segments are not
                // forwarded (the merged stream carries its own).
                PipelinePacket::CapsChanged(_) | PipelinePacket::Segment(_) => {}
                // future PipelinePacket variants: no-op.
                _ => {}
            }

            self.emit_ready(out).await
        })
    }
}

/// The absolute output sample-frame a PTS maps to, rounded to nearest.
fn pts_to_frame(pts_ns: u64, sample_rate: u32) -> u64 {
    if sample_rate == 0 {
        return 0;
    }
    ((pts_ns as u128 * sample_rate as u128 + 500_000_000) / 1_000_000_000) as u64
}

/// The PTS of an absolute output sample-frame.
fn frame_to_ns(frame: u64, sample_rate: u32) -> u64 {
    if sample_rate == 0 {
        return 0;
    }
    (frame as u128 * 1_000_000_000 / sample_rate as u128) as u64
}

/// Add S16LE `bytes` into an i32 accumulator lane-for-lane; excess bytes past
/// `acc` are ignored.
fn accumulate(acc: &mut [i32], bytes: &[u8]) {
    for (dst, chunk) in acc.iter_mut().zip(bytes.chunks_exact(2)) {
        *dst += i16::from_le_bytes([chunk[0], chunk[1]]) as i32;
    }
}

/// Clamp an i32 accumulator to the i16 range as S16LE bytes.
fn clamp_to_bytes(acc: &[i32]) -> Box<[u8]> {
    let mut out = vec![0u8; acc.len() * 2].into_boxed_slice();
    for (v, chunk) in acc.iter().zip(out.chunks_exact_mut(2)) {
        let s = (*v).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        chunk.copy_from_slice(&s.to_le_bytes());
    }
    out
}

/// Sum N interleaved S16LE buffers sample-wise, clamping to the i16 range.
/// Shorter buffers contribute zero past their end, so the result spans the
/// longest input.
#[cfg(test)]
fn mix(buffers: &[&[u8]]) -> Box<[u8]> {
    let max_samples = buffers.iter().map(|b| b.len() / 2).max().unwrap_or(0);
    let mut acc = vec![0i32; max_samples];
    for b in buffers {
        accumulate(&mut acc, b);
    }
    clamp_to_bytes(&acc)
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
        bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
    }

    #[test]
    fn mixes_and_clamps() {
        let a = pack(&[100, -200, 20000]);
        let b = pack(&[50, 100, 20000]);
        // 100+50, -200+100, 20000+20000 (saturates).
        assert_eq!(unpack(&mix(&[&a, &b])), [150, -100, i16::MAX]);
    }

    #[test]
    fn unequal_lengths_span_the_longest() {
        let a = pack(&[1000, 2000, 3000]);
        let b = pack(&[1, 2]);
        // third sample takes a alone (b is shorter).
        assert_eq!(unpack(&mix(&[&a, &b])), [1001, 2002, 3000]);
    }

    #[test]
    fn single_buffer_passes_through() {
        let a = pack(&[7, -7]);
        assert_eq!(unpack(&mix(&[&a])), [7, -7]);
    }
}
