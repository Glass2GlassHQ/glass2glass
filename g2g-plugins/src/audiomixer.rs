//! Audio mixer (`audiomixer`), the summing fan-in counterpart to the
//! passthrough `InterleaveMux`. Combines N interleaved S16LE inputs into one
//! output by adding samples and clamping to the i16 range. CPU-only `no_std`.
//!
//! v1 aligns inputs by arrival: it emits a mixed buffer once every still-open
//! input has delivered one, so it suits roughly synchronised equal-rate sources
//! (PTS-based alignment and sample-rate / channel reconciliation are follow-ups).
//! An input that reaches EOS early stops contributing; the others continue, its
//! place mixed as silence. Buffers of unequal length sum over the longer one.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError, MemoryDomain,
    MultiInputElement, OutputSink, PipelinePacket,
};

#[derive(Debug)]
pub struct AudioMixer {
    inputs: usize,
    output: Caps,
    queues: Vec<VecDeque<Frame>>,
    done: Vec<bool>,
    emitted: u64,
}

impl AudioMixer {
    /// A mixer with `inputs` input pads producing `output` caps. `output` is the
    /// fixed merged-output caps (the negotiated S16LE shape).
    pub fn new(inputs: usize, output: Caps) -> Self {
        assert!(inputs > 0, "AudioMixer needs at least one input");
        Self {
            inputs,
            output,
            queues: (0..inputs).map(|_| VecDeque::new()).collect(),
            done: vec![false; inputs],
            emitted: 0,
        }
    }

    /// Can a mix be emitted now: every input either has a queued buffer or is
    /// finished, and at least one has a buffer to contribute.
    fn ready(&self) -> bool {
        (0..self.inputs).all(|i| !self.queues[i].is_empty() || self.done[i])
            && (0..self.inputs).any(|i| !self.queues[i].is_empty())
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

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(_) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.queues[input].push_back(frame);
                }
                // A per-input Eos is informational: the runner emits the single
                // merged Eos once every input ends (the InterleaveMux contract).
                PipelinePacket::Eos => self.done[input] = true,
                PipelinePacket::Flush => self.queues[input].clear(),
                // Output caps are fixed; per-input caps / segments are not
                // forwarded (the merged stream carries its own).
                PipelinePacket::CapsChanged(_) | PipelinePacket::Segment(_) => {}
            }

            while self.ready() {
                let mut frames: Vec<Frame> = Vec::with_capacity(self.inputs);
                for i in 0..self.inputs {
                    if let Some(f) = self.queues[i].pop_front() {
                        frames.push(f);
                    }
                }
                let timing = frames[0].timing;
                let slices: Vec<&[u8]> = frames
                    .iter()
                    .filter_map(|f| match &f.domain {
                        MemoryDomain::System(s) => Some(s.as_slice()),
                        _ => None,
                    })
                    .collect();
                let mixed = mix(&slices);
                let out_frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(mixed)),
                    timing,
                    sequence: self.emitted,
                    meta: Default::default(),
                };
                self.emitted += 1;
                out.push(PipelinePacket::DataFrame(out_frame)).await?;
            }
            Ok(())
        })
    }
}

/// Sum N interleaved S16LE buffers sample-wise, clamping to the i16 range.
/// Shorter buffers contribute zero past their end, so the result spans the
/// longest input.
fn mix(buffers: &[&[u8]]) -> Box<[u8]> {
    let max_len = buffers.iter().map(|b| b.len()).max().unwrap_or(0);
    let samples = max_len / 2;
    let mut out = vec![0u8; samples * 2].into_boxed_slice();
    for (s, chunk) in out.chunks_exact_mut(2).enumerate() {
        let off = s * 2;
        let mut acc: i32 = 0;
        for b in buffers {
            if off + 2 <= b.len() {
                acc += i16::from_le_bytes([b[off], b[off + 1]]) as i32;
            }
        }
        let v = acc.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        chunk.copy_from_slice(&v.to_le_bytes());
    }
    out
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
