//! Tensor post-processing element (M27): the classification head that turns
//! a model's raw logits into something a consumer can act on, in-graph.
//! Composes after `OrtInference`: `... -> OrtInference -> TensorPostprocess`.
//!
//! Two operations over f32 tensors (treated as one flat vector, the
//! conventional reading of `[1, N]` logits; documented limitation for
//! higher-rank inputs):
//! - `softmax()`: numerically stable softmax, same caps out, values sum to 1
//! - `argmax()`: emits a `[1, 2]` f32 tensor `[winning index, winning value]`
//!
//! Pure Rust, no dependencies; always available in `g2g-ml` (no feature
//! gate).

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError, MemoryDomain,
    OutputSink, PipelinePacket, TensorDType, TensorLayout, TensorShape,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Softmax,
    ArgMax,
}

#[derive(Debug)]
pub struct TensorPostprocess {
    op: Op,
    configured: bool,
    /// Negotiated input tensor caps (configure + mid-stream changes);
    /// softmax echoes these on its output.
    input_caps: Option<Caps>,
    /// Output caps last advertised downstream (emission suppression).
    last_caps: Option<Caps>,
    emitted: u64,
}

impl TensorPostprocess {
    /// Numerically stable softmax over the flat tensor; output caps equal
    /// the input caps.
    pub fn softmax() -> Self {
        Self::new(Op::Softmax)
    }

    /// Winning class: emits `[1, 2]` f32 `[index, value]` of the flat
    /// tensor's maximum.
    pub fn argmax() -> Self {
        Self::new(Op::ArgMax)
    }

    fn new(op: Op) -> Self {
        Self {
            op,
            configured: false,
            input_caps: None,
            last_caps: None,
            emitted: 0,
        }
    }

    /// Count of processed frames pushed downstream. Useful in tests.
    pub fn processed_count(&self) -> u64 {
        self.emitted
    }

    fn derive(op: Op, input: &Caps) -> CapsSet {
        match input {
            Caps::Tensor {
                dtype: TensorDType::F32,
                ..
            } => CapsSet::one(match op {
                Op::Softmax => input.clone(),
                Op::ArgMax => argmax_caps(),
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        }
    }
}

fn argmax_caps() -> Caps {
    Caps::Tensor {
        dtype: TensorDType::F32,
        shape: TensorShape::new([1, 2]),
        layout: TensorLayout::Nchw,
    }
}

impl AsyncElement for TensorPostprocess {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::Tensor {
                dtype: TensorDType::F32,
                ..
            } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Native `DerivedOutput`: any f32 tensor in; softmax preserves the
    /// caps, argmax derives the fixed `[1, 2]` result shape.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let op = self.op;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| Self::derive(op, input)))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.intercept_caps(absolute_caps)?;
        self.input_caps = Some(absolute_caps.clone());
        self.configured = true;
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
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let bytes = slice.as_slice();
                    if bytes.is_empty() || bytes.len() % 4 != 0 {
                        return Err(G2gError::CapsMismatch);
                    }
                    let values: Vec<f32> = bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    // a model diverged to NaN/inf yields a meaningless class
                    // (argmax would emit (0, -inf), softmax all-NaN); reject
                    // loud rather than pass it downstream.
                    if values.iter().any(|v| !v.is_finite()) {
                        return Err(G2gError::CapsMismatch);
                    }

                    let (result, new_caps) = match self.op {
                        Op::Softmax => {
                            let probs = softmax(&values);
                            // softmax preserves the negotiated input caps;
                            // a flat [1, N] is the fallback when only the
                            // element count is known.
                            let caps = self.input_caps.clone().unwrap_or_else(|| Caps::Tensor {
                                dtype: TensorDType::F32,
                                shape: TensorShape::new([1, probs.len() as u32]),
                                layout: TensorLayout::Nchw,
                            });
                            (probs, caps)
                        }
                        Op::ArgMax => {
                            let (idx, val) = argmax(&values);
                            (vec![idx as f32, val], argmax_caps())
                        }
                    };

                    if self.last_caps.as_ref() != Some(&new_caps) {
                        out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                            .await?;
                        self.last_caps = Some(new_caps);
                    }
                    let mut out_bytes = Vec::with_capacity(result.len() * 4);
                    for v in &result {
                        out_bytes.extend_from_slice(&v.to_le_bytes());
                    }
                    let out_frame = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(
                            out_bytes.into_boxed_slice(),
                        )),
                        timing: frame.timing,
                        sequence: self.emitted,
                        meta: Default::default(),
                    };
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    // Runner contract (the videoconvert / videoscale convention):
                    // the transform arm pushes our pre-fixed forward *output* caps
                    // here (the input is already set via `configure_pipeline`), so
                    // `c` is our output (argmax's `[1, 2]`, softmax's echoed input).
                    // Forward it and record `last_caps` to suppress the data path's
                    // duplicate; do not adopt it as the input.
                    self.last_caps = Some(c.clone());
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                PipelinePacket::Flush => {
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                // Segment is a timing marker: forward unchanged.
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
}

/// Numerically stable softmax: shift by the max before exponentiating.
fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = values.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Index and value of the maximum; the first occurrence wins ties.
fn argmax(values: &[f32]) -> (usize, f32) {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, v) in values.iter().enumerate() {
        if *v > best.1 {
            best = (i, *v);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one_and_orders_correctly() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(p[2] > p[1] && p[1] > p[0]);
    }

    #[test]
    fn softmax_is_stable_for_large_logits() {
        // naive exp(1000) overflows to inf; the shifted form must not.
        let p = softmax(&[1000.0, 999.0]);
        assert!(p.iter().all(|v| v.is_finite()));
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[0] > p[1]);
    }

    #[test]
    fn argmax_picks_first_maximum() {
        assert_eq!(argmax(&[0.5, 3.0, -1.0]), (1, 3.0));
        assert_eq!(argmax(&[2.0, 2.0]), (0, 2.0), "first occurrence wins");
    }

    #[test]
    fn derived_output_for_argmax_is_one_by_two() {
        let logits = Caps::Tensor {
            dtype: TensorDType::F32,
            shape: TensorShape::new([1, 10]),
            layout: TensorLayout::Nchw,
        };
        assert_eq!(
            TensorPostprocess::derive(Op::ArgMax, &logits).alternatives(),
            &[argmax_caps()]
        );
        assert_eq!(
            TensorPostprocess::derive(Op::Softmax, &logits).alternatives(),
            core::slice::from_ref(&logits)
        );
        // non-f32 input rejected
        let bytes = Caps::Tensor {
            dtype: TensorDType::U8,
            shape: TensorShape::new([1, 10]),
            layout: TensorLayout::Nchw,
        };
        assert!(TensorPostprocess::derive(Op::Softmax, &bytes).is_empty());
    }
}
