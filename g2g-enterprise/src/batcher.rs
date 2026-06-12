//! Bounded multi-stream tensor batcher (DESIGN.md §5.3).
//!
//! M22: `TensorBatcher` is a `MultiInputElement` that gathers one tensor
//! frame from each input stream and emits the gathered round as a single
//! batched tensor frame, stacked along the leading batch dimension (for the
//! per-slot caps `[1, d...]`, a full round of `N` inputs emits `[N, d...]`;
//! stacking along dim 0 of a dense row-major tensor is plain byte
//! concatenation in input order). Feed the output to a dynamic-batch
//! inference session to amortize one execution across N camera streams.
//!
//! Liveness over completeness: an input that reaches end-of-stream stops
//! gating the gather (a dead camera must not stall the others). Batches then
//! shrink to the surviving inputs and the batcher emits a `CapsChanged` with
//! the smaller batch dim before the first shrunken frame. Queued frames of
//! an ended input still drain into batches first. This needs the per-input
//! `Eos` delivery added to `MultiInputElement::process` in M22.
//!
//! Owed: a deadline-based partial-batch flush (the "Timeout" half of §5.3's
//! select/timeout), gated on a runtime timer primitive; today a stalled
//! (but not ended) input stalls its round, which backpressures the other
//! inputs once links fill.

use core::future::Future;
use core::pin::Pin;
use std::collections::VecDeque;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError, MemoryDomain, MultiInputElement,
    OutputSink, PipelinePacket, TensorDType, TensorShape,
};

#[derive(Debug)]
pub struct TensorBatcher {
    /// Per-input caps: a tensor with leading batch dim 1.
    slot: Caps,
    slot_bytes: usize,
    queues: Vec<VecDeque<Frame>>,
    ended: Vec<bool>,
    configured: Vec<Option<Caps>>,
    /// Batch caps last advertised downstream. Seeded with the full-batch
    /// startup caps on first emit, so only a shrink emits `CapsChanged`.
    last_caps: Option<Caps>,
    emitted: u64,
}

impl TensorBatcher {
    /// `slot` is each input's tensor caps and must have a leading batch dim
    /// of 1 and static dims; the merged output is `slot` with the batch dim
    /// set to the number of currently contributing inputs.
    pub fn new(inputs: usize, slot: Caps) -> Result<Self, G2gError> {
        assert!(inputs > 0, "TensorBatcher needs at least one input");
        let slot_bytes = slot_byte_size(&slot).ok_or(G2gError::CapsMismatch)?;
        Ok(Self {
            slot,
            slot_bytes,
            queues: (0..inputs).map(|_| VecDeque::new()).collect(),
            ended: vec![false; inputs],
            configured: vec![None; inputs],
            last_caps: None,
            emitted: 0,
        })
    }

    /// Count of batched frames pushed downstream. Useful in tests.
    pub fn batches_emitted(&self) -> u64 {
        self.emitted
    }

    /// The caps input pad `input` was configured with.
    pub fn input_caps(&self, input: usize) -> Option<&Caps> {
        self.configured.get(input).and_then(|c| c.as_ref())
    }

    fn batched_caps(&self, batch: u32) -> Caps {
        let Caps::Tensor { dtype, shape, layout } = &self.slot else {
            unreachable!("slot validated at construction");
        };
        let mut dims = shape.0.clone();
        dims[0] = batch;
        Caps::Tensor {
            dtype: *dtype,
            shape: TensorShape(dims),
            layout: *layout,
        }
    }

    /// Inputs that still gate or feed a gather round: everything except an
    /// ended input whose queue has drained.
    fn contributors(&self) -> Vec<usize> {
        (0..self.queues.len())
            .filter(|&i| !(self.ended[i] && self.queues[i].is_empty()))
            .collect()
    }

    /// Emit every gather round that is currently complete.
    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        loop {
            let contributors = self.contributors();
            if contributors.is_empty()
                || contributors.iter().any(|&i| self.queues[i].is_empty())
            {
                return Ok(());
            }

            let parts: Vec<Frame> = contributors
                .iter()
                .map(|&i| self.queues[i].pop_front().expect("checked non-empty"))
                .collect();

            let new_caps = self.batched_caps(contributors.len() as u32);
            match &self.last_caps {
                // first batch: the startup-negotiated output is the full
                // batch, so emit CapsChanged only if this round already
                // shrank below it.
                None => {
                    let startup = self.batched_caps(self.queues.len() as u32);
                    if new_caps != startup {
                        out.push(PipelinePacket::CapsChanged(new_caps.clone())).await?;
                    }
                }
                Some(prev) if *prev != new_caps => {
                    out.push(PipelinePacket::CapsChanged(new_caps.clone())).await?;
                }
                _ => {}
            }
            self.last_caps = Some(new_caps);

            let mut bytes = Vec::with_capacity(self.slot_bytes * parts.len());
            let mut timing = parts[0].timing;
            for p in &parts {
                let MemoryDomain::System(slice) = &p.domain else {
                    unreachable!("validated on enqueue");
                };
                bytes.extend_from_slice(slice.as_slice());
                // batch pts is the newest constituent (the batch exists once
                // its last frame arrives); arrival keeps the oldest non-zero
                // stamp so glass-to-glass latency reports the worst case.
                timing.pts_ns = timing.pts_ns.max(p.timing.pts_ns);
                timing.dts_ns = timing.pts_ns;
                timing.capture_ns = timing.capture_ns.max(p.timing.capture_ns);
                if p.timing.arrival_ns != 0
                    && (timing.arrival_ns == 0 || p.timing.arrival_ns < timing.arrival_ns)
                {
                    timing.arrival_ns = p.timing.arrival_ns;
                }
            }

            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                timing,
                sequence: self.emitted,
            };
            self.emitted += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
    }
}

impl MultiInputElement for TensorBatcher {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.queues.len()
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&self.slot)
    }

    /// Every input must negotiate exactly the slot caps; batching is only
    /// well-defined over identical tensors.
    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(self.slot.clone()))
    }

    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        Ok(CapsConstraint::Produces(CapsSet::one(
            self.batched_caps(self.queues.len() as u32),
        )))
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        absolute_caps.intersect(&self.slot)?;
        self.configured[input] = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(self.batched_caps(self.queues.len() as u32))
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
                    if slice.as_slice().len() != self.slot_bytes {
                        return Err(G2gError::CapsMismatch);
                    }
                    self.queues[input].push_back(frame);
                    self.drain(out).await?;
                }
                PipelinePacket::Eos => {
                    // per-input end (M22 contract): stop gating on this
                    // input; its queued frames still drain. The runner owns
                    // the merged downstream Eos.
                    self.ended[input] = true;
                    self.drain(out).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    // the slot shape is pinned; a mid-stream change to
                    // anything else is a hard error.
                    c.intersect(&self.slot)?;
                }
                PipelinePacket::Flush => {
                    for q in &mut self.queues {
                        q.clear();
                    }
                    out.push(PipelinePacket::Flush).await?;
                }
            }
            Ok(())
        })
    }
}

/// Byte size of one slot tensor: element size times the product of its dims.
/// `None` for non-tensor caps, a batch dim other than 1, or any non-static
/// dim.
fn slot_byte_size(slot: &Caps) -> Option<usize> {
    let Caps::Tensor { dtype, shape, .. } = slot else {
        return None;
    };
    let dims = &shape.0;
    if dims.first() != Some(&1) || dims.contains(&0) {
        return None;
    }
    let elem = match dtype {
        TensorDType::F16 => 2usize,
        TensorDType::F32 => 4,
        TensorDType::I8 | TensorDType::U8 => 1,
    };
    dims.iter()
        .try_fold(elem, |acc, d| acc.checked_mul(*d as usize))
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::TensorLayout;

    fn slot(dims: Vec<u32>) -> Caps {
        Caps::Tensor {
            dtype: TensorDType::U8,
            shape: TensorShape(dims),
            layout: TensorLayout::Nchw,
        }
    }

    #[test]
    fn slot_byte_size_requires_batch_one_static_dims() {
        assert_eq!(slot_byte_size(&slot(vec![1, 4])), Some(4));
        assert_eq!(slot_byte_size(&slot(vec![1, 3, 2, 2])), Some(12));
        assert_eq!(slot_byte_size(&slot(vec![2, 4])), None, "batch must be 1");
        assert_eq!(slot_byte_size(&slot(vec![1, 0])), None, "dims must be static");
    }

    #[test]
    fn output_caps_stack_along_batch_dim() {
        let b = TensorBatcher::new(3, slot(vec![1, 3, 2, 2])).unwrap();
        assert_eq!(
            MultiInputElement::output_caps(&b).unwrap(),
            slot(vec![3, 3, 2, 2])
        );
    }

    #[test]
    fn construction_rejects_non_tensor_slots() {
        use g2g_core::{Dim, Rate, RawVideoFormat};
        let video = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: Rate::Any,
        };
        assert_eq!(
            TensorBatcher::new(2, video).err(),
            Some(G2gError::CapsMismatch)
        );
    }
}
