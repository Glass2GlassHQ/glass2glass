//! Pure-Rust Burn inference element (`burn` backend, DESIGN.md §5.2).
//!
//! `BurnInference` is the no-C++ counterpart of `OrtInference`: it negotiates
//! `Caps::RawVideo` (RGBA) on its input pad and `Caps::Tensor` on its output
//! pad, and runs the inference on burn's `wgpu` backend (any D3D12 / Vulkan /
//! Metal GPU, and WebGPU on wasm). v1 ships a single linear layer
//! (`output = input . W + b`): each RGBA frame is normalized to a flat f32 NCHW
//! RGB vector (`value / 255`, the same preprocessing `OrtInference` does) and
//! multiplied by the supplied weight matrix, emitting the `[1, N]` logits as an
//! f32 `Caps::Tensor`. The weights are caller-supplied and deterministic, so the
//! output is exactly verifiable on the CPU.
//!
//! This is the backend foundation, not a model zoo. ONNX import (burn-import is
//! build-time codegen, not a runtime loader) and richer layers (conv, the burn
//! `Module` path with trained weights) are follow-ups; the `AsyncElement` /
//! caps contract here is what they slot into.

use core::future::Future;
use core::pin::Pin;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::tensor::{Tensor, TensorData};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, HardwareError,
    MemoryDomain, OutputSink, PipelinePacket, Rate, RawVideoFormat, TensorDType, TensorLayout,
    TensorShape,
};

type B = Wgpu;

/// RGBA8 -> normalized f32 NCHW RGB (`value / 255`), flattened in plane order
/// (R plane, then G, then B). Shared by the element and its test so the linear
/// layer's input is defined in exactly one place.
pub fn normalize_rgba_nchw(rgba: &[u8], width: usize, height: usize) -> Vec<f32> {
    let plane = width * height;
    let mut flat = vec![0f32; 3 * plane];
    for px in 0..plane {
        let src = px * 4;
        flat[px] = rgba[src] as f32 / 255.0;
        flat[plane + px] = rgba[src + 1] as f32 / 255.0;
        flat[2 * plane + px] = rgba[src + 2] as f32 / 255.0;
    }
    flat
}

#[derive(Debug)]
pub struct BurnInference {
    width: u32,
    height: u32,
    num_outputs: usize,
    /// Row-major `[K, N]` weight matrix, `K = 3 * W * H`.
    weights: Vec<f32>,
    /// `[N]` bias.
    bias: Vec<f32>,
    device: WgpuDevice,
    weight_t: Option<Tensor<B, 2>>,
    bias_t: Option<Tensor<B, 2>>,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl BurnInference {
    /// A linear layer over RGBA frames of `width x height`. `weights` is the
    /// row-major `[K, N]` matrix (`K = 3 * width * height`) and `bias` is `[N]`;
    /// `N`, the output count, is `bias.len()`. Fails loud on a dimension
    /// mismatch.
    pub fn linear(
        width: u32,
        height: u32,
        weights: Vec<f32>,
        bias: Vec<f32>,
    ) -> Result<Self, G2gError> {
        let num_outputs = bias.len();
        // Fold the weight-matrix size with checked ops so an overflowing
        // (width, height, num_outputs) fails the validation gate instead of
        // wrapping to a value that admits a short weight buffer.
        let k = (width as usize)
            .checked_mul(height as usize)
            .and_then(|wh| wh.checked_mul(3));
        let expected = k.and_then(|k| k.checked_mul(num_outputs));
        if num_outputs == 0 || k == Some(0) || Some(weights.len()) != expected {
            return Err(G2gError::CapsMismatch);
        }
        Ok(Self {
            width,
            height,
            num_outputs,
            weights,
            bias,
            device: WgpuDevice::default(),
            weight_t: None,
            bias_t: None,
            configured: false,
            last_caps: None,
            emitted: 0,
        })
    }

    /// Count of tensor `DataFrame`s pushed downstream. Useful in tests.
    pub fn inferred_count(&self) -> u64 {
        self.emitted
    }

    fn supported_input(&self) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: Rate::Any,
        }
    }

    fn output_caps(&self) -> Caps {
        Caps::Tensor {
            dtype: TensorDType::F32,
            shape: TensorShape::new([1, self.num_outputs as u32]),
            layout: TensorLayout::Nchw,
        }
    }

    /// Normalize RGBA, run `input . W + b` on the GPU, return the `[1, N]`
    /// logits as little-endian f32 bytes (the `OrtInference` output format).
    fn infer(&self, rgba: &[u8]) -> Result<Box<[u8]>, G2gError> {
        let (w, h) = (self.width as usize, self.height as usize);
        if rgba.len() < w * h * 4 {
            return Err(G2gError::CapsMismatch);
        }
        let weight = self.weight_t.as_ref().ok_or(G2gError::NotConfigured)?;
        let bias = self.bias_t.as_ref().ok_or(G2gError::NotConfigured)?;

        let flat = normalize_rgba_nchw(rgba, w, h);
        let k = flat.len();
        let input = Tensor::<B, 2>::from_data(TensorData::new(flat, [1, k]), &self.device);
        let logits = input.matmul(weight.clone()).add(bias.clone());

        let values = logits
            .into_data()
            .to_vec::<f32>()
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Ok(bytes.into_boxed_slice())
    }
}

impl AsyncElement for BurnInference {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&self.supported_input())
    }

    /// Native `DerivedOutput`: RGBA at the fixed geometry in, the `[1, N]`
    /// tensor out. Non-matching input yields an empty set, rejected at solve.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let supported = self.supported_input();
        let out = self.output_caps();
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| {
            if input.intersect(&supported).is_ok() {
                CapsSet::one(out.clone())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // Validate caps before touching the GPU so a mismatch fails cheaply.
        absolute_caps.intersect(&self.supported_input())?;
        let k = 3 * self.width as usize * self.height as usize;
        self.weight_t = Some(Tensor::<B, 2>::from_data(
            TensorData::new(self.weights.clone(), [k, self.num_outputs]),
            &self.device,
        ));
        self.bias_t = Some(Tensor::<B, 2>::from_data(
            TensorData::new(self.bias.clone(), [1, self.num_outputs]),
            &self.device,
        ));
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
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let bytes = self.infer(slice)?;
                    let new_caps = self.output_caps();
                    if self.last_caps.as_ref() != Some(&new_caps) {
                        out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                            .await?;
                        self.last_caps = Some(new_caps);
                    }
                    let tensor = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes)),
                        // per-frame inference: inherit source timing so latency
                        // stays traceable.
                        timing: frame.timing,
                        sequence: self.emitted,
                        meta: Default::default(),
                    };
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(tensor)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    // geometry is pinned at construction; anything else is a
                    // hard error.
                    c.intersect(&self.supported_input())?;
                }
                PipelinePacket::Flush => {
                    out.push(PipelinePacket::Flush).await?;
                }
                // Segment is a timing marker: forward unchanged.
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                // stateless per-frame inference: nothing to drain.
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

/// Whether burn's wgpu backend can run on this host. Probes by running a
/// trivial op and catching the adapter-acquisition panic, so tests skip
/// gracefully on a headless machine.
pub fn gpu_available() -> bool {
    std::panic::catch_unwind(|| {
        let device = WgpuDevice::default();
        let t = Tensor::<B, 1>::from_data(TensorData::new(vec![1.0f32], [1]), &device);
        let _ = t.into_data();
    })
    .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgba(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
        }
    }

    fn nv12(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
        }
    }

    #[test]
    fn linear_validates_weight_dimensions() {
        // 2x2 RGBA -> K = 12; N = 2 -> 24 weights expected.
        assert!(BurnInference::linear(2, 2, vec![0.0; 24], vec![0.0; 2]).is_ok());
        assert_eq!(
            BurnInference::linear(2, 2, vec![0.0; 23], vec![0.0; 2]).err(),
            Some(G2gError::CapsMismatch),
            "weights must be K*N"
        );
        assert_eq!(
            BurnInference::linear(2, 2, vec![0.0; 0], vec![]).err(),
            Some(G2gError::CapsMismatch),
            "needs at least one output"
        );
    }

    #[test]
    fn intercept_narrows_rgba_and_rejects_nv12() {
        let e = BurnInference::linear(4, 4, vec![0.0; 3 * 16 * 2], vec![0.0; 2]).unwrap();
        assert!(e.intercept_caps(&rgba(4, 4)).is_ok());
        assert_eq!(e.intercept_caps(&nv12(4, 4)), Err(G2gError::CapsMismatch));
        // wrong geometry is also rejected (fixed by the model).
        assert_eq!(e.intercept_caps(&rgba(8, 8)), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn configure_rejects_non_rgba_before_gpu() {
        let mut e = BurnInference::linear(2, 2, vec![0.0; 24], vec![0.0; 2]).unwrap();
        assert_eq!(
            e.configure_pipeline(&nv12(2, 2)).err(),
            Some(G2gError::CapsMismatch)
        );
        assert!(
            e.weight_t.is_none(),
            "no GPU tensors built on rejected caps"
        );
    }
}
