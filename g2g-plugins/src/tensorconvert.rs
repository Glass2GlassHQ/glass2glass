//! Tensor dtype converter (M441): the tensor-domain sibling of
//! [`VideoConvert`](crate::videoconvert::VideoConvert). Where `VideoConvert`
//! converts between raw-video pixel formats, `TensorConvert` converts between
//! [`TensorDType`]s: it *quantizes* an f32 tensor to int8 / uint8
//! (`q = round(x / scale) + zero_point`, clamped to the dtype range) or
//! *dequantizes* the inverse (`x = (q - zero_point) * scale`). Shape and layout
//! pass through unchanged; only the element dtype changes.
//!
//! This is the missing step for a *fully* on-NPU inference: a quantized model
//! whose input is int8 / uint8 runs end to end on an accelerator like the Edge
//! TPU, where a float-input model leaves the boundary `QuantizeLinear` on the CPU
//! (M440). `f32 tensor -> TensorConvert(quantize) -> uint8 tensor -> inference`
//! moves that quantize out of the model so the whole graph is accelerator-eligible.
//!
//! Scope (v1): the F32 <-> U8 / I8 affine quantization, in the tensor's existing
//! layout. Layout conversion (NCHW <-> NHWC) and F16 are follow-ups; both fit the
//! same element (a layout transpose is a strided copy, F16 another dtype arm).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, TensorDType, TensorLayout, TensorShape,
};

/// Round half away from zero without `f32::round` (which is std / libm only; the
/// baseline is `no_std`). Quantization tolerates either rounding rule.
fn round_to_i32(v: f32) -> i32 {
    if v >= 0.0 {
        (v + 0.5) as i32
    } else {
        (v - 0.5) as i32
    }
}

/// The clamp range of an integer quantization dtype.
fn int_range(dtype: TensorDType) -> Option<(i32, i32)> {
    match dtype {
        TensorDType::U8 => Some((0, 255)),
        TensorDType::I8 => Some((-128, 127)),
        _ => None,
    }
}

/// Quantize f32 elements to `target` (U8 / I8) bytes: one byte per element,
/// `clamp(round(x / scale) + zero_point)`. An I8 value is stored as its two's
/// complement byte (`q as u8`). `scale` 0 is treated as 1 (a degenerate model
/// would otherwise divide by zero); a real quantized model always carries a
/// nonzero scale.
pub fn quantize_f32(src: &[f32], target: TensorDType, scale: f32, zero_point: i32) -> Option<Vec<u8>> {
    let (lo, hi) = int_range(target)?;
    let scale = if scale == 0.0 { 1.0 } else { scale };
    Some(
        src.iter()
            .map(|&x| {
                let q = (round_to_i32(x / scale) + zero_point).clamp(lo, hi);
                q as u8
            })
            .collect(),
    )
}

/// Dequantize int8 / uint8 bytes back to f32: `(q - zero_point) * scale`, the
/// inverse of [`quantize_f32`]. `src_dtype` selects the byte interpretation
/// (I8 = signed two's complement).
pub fn dequantize_to_f32(src: &[u8], src_dtype: TensorDType, scale: f32, zero_point: i32) -> Option<Vec<f32>> {
    int_range(src_dtype)?;
    Some(
        src.iter()
            .map(|&b| {
                let q = match src_dtype {
                    TensorDType::I8 => (b as i8) as i32,
                    _ => b as i32,
                };
                (q - zero_point) as f32 * scale
            })
            .collect(),
    )
}

/// Element count of a tensor shape (product of dims).
fn numel(shape: &TensorShape) -> usize {
    shape.0.iter().map(|&d| d as usize).product()
}

#[derive(Debug)]
pub struct TensorConvert {
    /// Output dtype. An integer target (U8 / I8) means quantize an f32 input; an
    /// F32 target means dequantize an int8 / uint8 input.
    target: TensorDType,
    /// Affine quantization parameters, matched to the model's input (quantize) or
    /// the producer's output (dequantize) tensor.
    scale: f32,
    zero_point: i32,
    /// The configured input dtype / shape / layout (from `configure_pipeline`).
    input: Option<(TensorDType, TensorShape, TensorLayout)>,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl TensorConvert {
    /// Quantize an f32 tensor to `target` (U8 or I8) with the given affine params.
    pub fn quantize(target: TensorDType, scale: f32, zero_point: i32) -> Self {
        Self { target, scale, zero_point, input: None, configured: false, last_caps: None, emitted: 0 }
    }

    /// Dequantize an int8 / uint8 tensor back to f32 with the given affine params.
    pub fn dequantize(scale: f32, zero_point: i32) -> Self {
        Self {
            target: TensorDType::F32,
            scale,
            zero_point,
            input: None,
            configured: false,
            last_caps: None,
            emitted: 0,
        }
    }

    /// The dtypes this instance accepts as input: int8 / uint8 when dequantizing
    /// (target F32), else f32 when quantizing.
    fn source_dtypes(&self) -> &'static [TensorDType] {
        if self.target == TensorDType::F32 {
            &[TensorDType::U8, TensorDType::I8]
        } else {
            &[TensorDType::F32]
        }
    }

    /// Validate a tensor caps as a convertible input and return its parts.
    fn accept_input(&self, caps: &Caps) -> Result<(TensorDType, TensorShape, TensorLayout), G2gError> {
        let Caps::Tensor { dtype, shape, layout } = caps else {
            return Err(G2gError::CapsMismatch);
        };
        if !self.source_dtypes().contains(dtype) {
            return Err(G2gError::CapsMismatch);
        }
        Ok((*dtype, shape.clone(), *layout))
    }

    /// Expected input byte length for the configured tensor.
    fn input_bytes(dtype: TensorDType, shape: &TensorShape) -> usize {
        numel(shape) * dtype.size()
    }
}

impl AsyncElement for TensorConvert {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // A tensor caps carries a concrete shape, so there is nothing to narrow:
        // accept it when its dtype is a valid source for this conversion.
        self.accept_input(upstream_caps).map(|_| upstream_caps.clone())
    }

    /// Native `DerivedOutput`: a tensor of a valid source dtype maps to the target
    /// dtype at the same shape and layout.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let target = self.target;
        let sources = self.source_dtypes();
        let derive = Box::new(move |input: &Caps| match input {
            Caps::Tensor { dtype, shape, layout } if sources.contains(dtype) => {
                CapsSet::one(Caps::Tensor { dtype: target, shape: shape.clone(), layout: *layout })
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        });
        CapsConstraint::DerivedOutput(derive)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.input = Some(self.accept_input(absolute_caps)?);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(&'a mut self, packet: PipelinePacket, out: &'a mut dyn OutputSink) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let (in_dtype, shape, layout) = self.input.clone().ok_or(G2gError::NotConfigured)?;
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let src = slice.as_slice();
                    if src.len() < Self::input_bytes(in_dtype, &shape) {
                        return Err(G2gError::CapsMismatch);
                    }

                    let converted: Vec<u8> = if in_dtype == TensorDType::F32 {
                        // Quantize: reinterpret the input bytes as f32 elements.
                        let floats: Vec<f32> = src
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        quantize_f32(&floats, self.target, self.scale, self.zero_point)
                            .ok_or(G2gError::CapsMismatch)?
                    } else {
                        // Dequantize: int bytes -> f32 -> little-endian bytes.
                        let n = numel(&shape);
                        let floats = dequantize_to_f32(&src[..n], in_dtype, self.scale, self.zero_point)
                            .ok_or(G2gError::CapsMismatch)?;
                        let mut bytes = Vec::with_capacity(floats.len() * 4);
                        for f in floats {
                            bytes.extend_from_slice(&f.to_le_bytes());
                        }
                        bytes
                    };

                    let new_caps = Caps::Tensor { dtype: self.target, shape: shape.clone(), layout };
                    if self.last_caps.as_ref() != Some(&new_caps) {
                        out.push(PipelinePacket::CapsChanged(new_caps.clone())).await?;
                        self.last_caps = Some(new_caps);
                    }
                    let out_frame = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(converted.into_boxed_slice())),
                        timing: frame.timing,
                        sequence: self.emitted,
                        meta: Default::default(),
                    };
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                // The runner calls configure_pipeline then pushes the forward output
                // caps; forward it and suppress the data path's duplicate emit.
                PipelinePacket::CapsChanged(c) => {
                    out.push(PipelinePacket::CapsChanged(c.clone())).await?;
                    self.last_caps = Some(c);
                }
                PipelinePacket::Flush => {
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        TENSORCONVERT_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Tensor dtype converter",
            "Filter/Converter/Tensor",
            "Quantizes f32 tensors to int8/uint8 or dequantizes the inverse",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "dtype" => {
                self.target = dtype_from_str(value.as_str().ok_or(PropError::Type)?).ok_or(PropError::Value)?;
                Ok(())
            }
            "scale" => match value {
                PropValue::Double(v) => {
                    self.scale = v as f32;
                    Ok(())
                }
                _ => Err(PropError::Type),
            },
            "zero-point" => match value {
                PropValue::Int(v) => {
                    self.zero_point = v as i32;
                    Ok(())
                }
                _ => Err(PropError::Type),
            },
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "dtype" => Some(PropValue::Str(dtype_to_str(self.target).into())),
            "scale" => Some(PropValue::Double(self.scale as f64)),
            "zero-point" => Some(PropValue::Int(self.zero_point as i64)),
            _ => None,
        }
    }
}

impl PadTemplates for TensorConvert {
    /// A tensor caps carries a concrete shape, so there is no meaningful static
    /// superset to advertise (unlike `VideoConvert`'s `Dim::Any` geometry). The
    /// element is placed explicitly and negotiates via `caps_constraint`, so it
    /// declares no templates (it is not part of the auto-plug pool).
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::new()
    }
}

/// Parse a tensor-dtype property string.
fn dtype_from_str(s: &str) -> Option<TensorDType> {
    match s.to_ascii_lowercase().as_str() {
        "f32" | "float32" => Some(TensorDType::F32),
        "f16" | "float16" => Some(TensorDType::F16),
        "i8" | "int8" => Some(TensorDType::I8),
        "u8" | "uint8" => Some(TensorDType::U8),
        _ => None,
    }
}

fn dtype_to_str(d: TensorDType) -> &'static str {
    match d {
        TensorDType::F32 => "f32",
        TensorDType::F16 => "f16",
        TensorDType::I8 => "i8",
        TensorDType::U8 => "u8",
    }
}

static TENSORCONVERT_PROPS: &[PropertySpec] = &[
    PropertySpec::new("dtype", PropKind::Str, "output dtype: f32 | f16 | i8 | u8 (int target = quantize, f32 = dequantize)"),
    PropertySpec::new("scale", PropKind::Double, "affine quantization scale"),
    PropertySpec::new("zero-point", PropKind::Int, "affine quantization zero point"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_then_dequantize_round_trips_within_a_step() {
        // uint8 affine: scale 0.05, zp 128 covers roughly [-6.4, 6.35].
        let (scale, zp) = (0.05f32, 128);
        let xs = [0.0f32, 0.5, -0.5, 1.0, -1.0, 3.0, -3.0];
        let q = quantize_f32(&xs, TensorDType::U8, scale, zp).expect("u8 quantize");
        assert_eq!(q.len(), xs.len(), "one byte per element");
        let back = dequantize_to_f32(&q, TensorDType::U8, scale, zp).expect("dequantize");
        for (x, r) in xs.iter().zip(&back) {
            assert!((x - r).abs() <= scale, "dequant within one step: {x} vs {r}");
        }
    }

    #[test]
    fn quantize_clamps_to_the_dtype_range() {
        let (scale, zp) = (0.1f32, 0);
        // Way out of range both directions -> clamps, not wraps.
        let q = quantize_f32(&[1000.0, -1000.0], TensorDType::I8, scale, zp).expect("i8");
        assert_eq!(q[0] as i8, 127);
        assert_eq!(q[1] as i8, -128);
        // uint8 clamps at 0 and 255.
        let q = quantize_f32(&[1000.0, -1000.0], TensorDType::U8, scale, zp).expect("u8");
        assert_eq!(q, [255, 0]);
    }

    #[test]
    fn dequantize_handles_signed_int8_bytes() {
        // i8 byte 0xFF = -1; (-1 - 0) * 0.5 = -0.5.
        let back = dequantize_to_f32(&[0xFF], TensorDType::I8, 0.5, 0).expect("i8");
        assert_eq!(back, [-0.5]);
    }

    #[test]
    fn float_target_is_dequantize_int_target_is_quantize() {
        assert_eq!(TensorConvert::quantize(TensorDType::U8, 1.0, 0).source_dtypes(), &[TensorDType::F32]);
        assert_eq!(TensorConvert::dequantize(1.0, 0).source_dtypes(), &[TensorDType::U8, TensorDType::I8]);
    }
}
