//! Tensor dtype / layout converter (M441, M445): the tensor-domain sibling of
//! [`VideoConvert`](crate::videoconvert::VideoConvert). Where `VideoConvert`
//! converts between raw-video pixel formats, `TensorConvert` converts between
//! [`TensorDType`]s and [`TensorLayout`]s:
//!
//! - *quantize* an f32 tensor to int8 / uint8
//!   (`q = round(x / scale) + zero_point`, clamped to the dtype range),
//! - *dequantize* the inverse (`x = (q - zero_point) * scale`),
//! - *narrow* f32 to IEEE-754 half (`F16`) or *widen* half back to f32,
//! - *transpose* the layout `NCHW <-> NHWC`.
//!
//! A dtype change and a layout change compose in a single pass: the elementwise
//! dtype op runs first (positionally independent, so it commutes with the
//! transpose), then the result is reordered into the output layout. This is the
//! step most real on-NPU models need: an NNAPI / TFLite quantized vision model
//! wants `NHWC uint8`, while an ONNX export from PyTorch is `NCHW`, and a camera
//! frame arrives interleaved (NHWC). One element does `f32 NCHW ->
//! quantize+transpose -> uint8 NHWC`, moving both the quantize and the layout
//! shuffle out of the model so the whole graph stays accelerator-eligible (the
//! float boundary `QuantizeLinear` / `Transpose` no longer pin work on the CPU,
//! cf. M440 / M442).
//!
//! Scope: F32 <-> U8 / I8 affine quantization, F32 <-> F16 float narrowing, and
//! 4D NCHW <-> NHWC transposition. A direct int8 <-> uint8 or int <-> f16
//! conversion is rejected (no defined affine); layout transposition is defined
//! only for rank-4 tensors.

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

/// IEEE-754 half-precision (binary16) encode of an `f32`: round to nearest even,
/// overflow to +/-inf, underflow to a subnormal or signed zero. Pure bit
/// manipulation, no `std`/libm (the affine int quantization above is lossy by
/// design; this is its lossy *float* narrowing sibling).
fn f32_to_f16_bits(value: f32) -> u16 {
    let x = value.to_bits();
    let sign = x & 0x8000_0000;
    let exp = x & 0x7F80_0000;
    let man = x & 0x007F_FFFF;

    // Inf / NaN: exponent field all ones.
    if exp == 0x7F80_0000 {
        // Nonzero mantissa is NaN (kept quiet); zero mantissa is infinity.
        let nan = if man != 0 { 0x0200 } else { 0 };
        return ((sign >> 16) | 0x7C00 | nan | (man >> 13)) as u16;
    }

    let half_sign = sign >> 16;
    let unbiased = ((exp >> 23) as i32) - 127;
    let half_exp = unbiased + 15;

    // Half exponent overflow -> infinity.
    if half_exp >= 0x1F {
        return (half_sign | 0x7C00) as u16;
    }

    // Subnormal half or zero.
    if half_exp <= 0 {
        // Too small to round even into the smallest subnormal -> signed zero.
        if 14 - half_exp > 24 {
            return half_sign as u16;
        }
        let man = man | 0x0080_0000; // restore the implicit leading 1
        let mut half_man = man >> (14 - half_exp);
        // Round to nearest, ties to even.
        let round_bit = 1 << (13 - half_exp);
        if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
            half_man += 1;
        }
        return (half_sign | half_man) as u16;
    }

    // Normalized.
    let half_man = man >> 13;
    let round_bit = 0x0000_1000;
    let bits = half_sign | ((half_exp as u32) << 10) | half_man;
    if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
        // Carry from a mantissa overflow naturally bumps the exponent.
        (bits + 1) as u16
    } else {
        bits as u16
    }
}

/// IEEE-754 half (binary16) decode back to `f32`, the inverse of
/// [`f32_to_f16_bits`]. Always exact: every half value is representable in f32.
fn f16_bits_to_f32(h: u16) -> f32 {
    let h = h as u32;
    let sign = (h & 0x8000) << 16;
    let exp = (h & 0x7C00) >> 10;
    let man = h & 0x03FF;

    if exp == 0 {
        if man == 0 {
            return f32::from_bits(sign); // signed zero
        }
        // Subnormal: value = man * 2^-24, exactly representable in f32.
        let v = (man as f32) * (1.0 / 16_777_216.0);
        return if sign != 0 { -v } else { v };
    }
    if exp == 0x1F {
        // Inf / NaN.
        return f32::from_bits(sign | 0x7F80_0000 | (man << 13));
    }
    // Normalized: rebias the exponent (15 -> 127), widen the mantissa.
    let exp = ((exp as i32 - 15 + 127) as u32) << 23;
    f32::from_bits(sign | exp | (man << 13))
}

/// The elementwise dtype operation an instance applies, resolved from the
/// concrete input dtype and the configured target (`None` = preserve input).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Conversion {
    /// Output dtype equals input dtype: a raw copy (used for layout-only work).
    Identity,
    /// f32 -> U8 / I8 affine quantization.
    Quantize,
    /// U8 / I8 -> f32 affine dequantization.
    Dequantize,
    /// f32 -> F16 float narrowing.
    NarrowF16,
    /// F16 -> f32 float widening.
    WidenF16,
}

/// Resolve `(input dtype, target)` to the output dtype and the conversion to
/// apply. `target` `None` preserves the input dtype (identity, for a
/// layout-only transpose). Returns `None` for an undefined pairing (a direct
/// int8 <-> uint8, or any int <-> f16, has no affine and is rejected).
fn resolve(in_dtype: TensorDType, target: Option<TensorDType>) -> Option<(TensorDType, Conversion)> {
    let out = target.unwrap_or(in_dtype);
    use TensorDType::{F16, F32, I8, U8};
    let conv = match (in_dtype, out) {
        (a, b) if a == b => Conversion::Identity,
        (F32, U8 | I8) => Conversion::Quantize,
        (U8 | I8, F32) => Conversion::Dequantize,
        (F32, F16) => Conversion::NarrowF16,
        (F16, F32) => Conversion::WidenF16,
        _ => return None,
    };
    Some((out, conv))
}

/// For a layout change, the output shape (in `to` order) and a permutation
/// `perm` where output element `i` is sourced from input element `perm[i]`.
/// `from == to` is the identity permutation at any rank; an actual transpose is
/// defined only for rank-4 tensors (returns `None` otherwise).
fn layout_permutation(
    in_shape: &TensorShape,
    from: TensorLayout,
    to: TensorLayout,
) -> Option<(TensorShape, Vec<usize>)> {
    if from == to {
        let n = in_shape.elements();
        return Some((*in_shape, (0..n).collect()));
    }
    if in_shape.dims().len() != 4 {
        return None;
    }
    let d: Vec<usize> = in_shape.dims().iter().map(|&x| x as usize).collect();
    match (from, to) {
        (TensorLayout::Nchw, TensorLayout::Nhwc) => {
            let (n, c, h, w) = (d[0], d[1], d[2], d[3]);
            let out = TensorShape::new([n as u32, h as u32, w as u32, c as u32]);
            let mut perm = Vec::with_capacity(n * c * h * w);
            for nn in 0..n {
                for hh in 0..h {
                    for ww in 0..w {
                        for cc in 0..c {
                            perm.push(((nn * c + cc) * h + hh) * w + ww);
                        }
                    }
                }
            }
            Some((out, perm))
        }
        (TensorLayout::Nhwc, TensorLayout::Nchw) => {
            let (n, h, w, c) = (d[0], d[1], d[2], d[3]);
            let out = TensorShape::new([n as u32, c as u32, h as u32, w as u32]);
            let mut perm = Vec::with_capacity(n * c * h * w);
            for nn in 0..n {
                for cc in 0..c {
                    for hh in 0..h {
                        for ww in 0..w {
                            perm.push(((nn * h + hh) * w + ww) * c + cc);
                        }
                    }
                }
            }
            Some((out, perm))
        }
        _ => None,
    }
}

/// The output caps an instance derives from a tensor input, matching exactly
/// what [`TensorConvert::process`] emits. `None` if the input is not a tensor or
/// the (dtype, layout) conversion is undefined for it.
fn derive_output_caps(input: &Caps, target: Option<TensorDType>, out_layout: Option<TensorLayout>) -> Option<Caps> {
    let Caps::Tensor { dtype, shape, layout } = input else {
        return None;
    };
    let (out_dtype, _) = resolve(*dtype, target)?;
    let to = out_layout.unwrap_or(*layout);
    let (out_shape, _) = layout_permutation(shape, *layout, to)?;
    Some(Caps::Tensor { dtype: out_dtype, shape: out_shape, layout: to })
}

#[derive(Debug)]
pub struct TensorConvert {
    /// Output dtype. `None` preserves the input dtype (a layout-only transpose).
    /// An integer target (U8 / I8) quantizes an f32 input; F32 dequantizes an
    /// int input or widens an F16 input; F16 narrows an f32 input.
    target: Option<TensorDType>,
    /// Affine quantization parameters, matched to the model's input (quantize)
    /// or the producer's output (dequantize). Ignored for the F16 / identity
    /// conversions.
    scale: f32,
    zero_point: i32,
    /// Output layout. `None` preserves the input layout; `Some` transposes a 4D
    /// tensor when it differs from the input layout.
    out_layout: Option<TensorLayout>,
    /// The configured input dtype / shape / layout (from `configure_pipeline`).
    input: Option<(TensorDType, TensorShape, TensorLayout)>,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl TensorConvert {
    /// Quantize an f32 tensor to `target` (U8 or I8) with the given affine params.
    pub fn quantize(target: TensorDType, scale: f32, zero_point: i32) -> Self {
        Self::with(Some(target), scale, zero_point)
    }

    /// Dequantize an int8 / uint8 tensor back to f32 with the given affine params.
    pub fn dequantize(scale: f32, zero_point: i32) -> Self {
        Self::with(Some(TensorDType::F32), scale, zero_point)
    }

    /// Narrow an f32 tensor to IEEE-754 half (`F16`). No affine: a lossy float cast.
    pub fn narrow_f16() -> Self {
        Self::with(Some(TensorDType::F16), 1.0, 0)
    }

    /// Widen an `F16` tensor back to f32 (exact). The float sibling of
    /// [`dequantize`](Self::dequantize), which it shares an F32 target with; the
    /// actual conversion is chosen by the input dtype at configure time.
    pub fn widen_f16() -> Self {
        Self::with(Some(TensorDType::F32), 1.0, 0)
    }

    /// Transpose the layout to `layout` (NCHW <-> NHWC) without changing dtype.
    pub fn transpose(layout: TensorLayout) -> Self {
        let mut s = Self::with(None, 1.0, 0);
        s.out_layout = Some(layout);
        s
    }

    /// Builder: also emit the output in `layout`, transposing if it differs from
    /// the input layout. Composes with any dtype conversion (e.g.
    /// `TensorConvert::quantize(U8, s, z).to_layout(Nhwc)`).
    pub fn to_layout(mut self, layout: TensorLayout) -> Self {
        self.out_layout = Some(layout);
        self
    }

    fn with(target: Option<TensorDType>, scale: f32, zero_point: i32) -> Self {
        Self {
            target,
            scale,
            zero_point,
            out_layout: None,
            input: None,
            configured: false,
            last_caps: None,
            emitted: 0,
        }
    }

    /// Validate a tensor caps as a convertible input and return its parts.
    fn accept_input(&self, caps: &Caps) -> Result<(TensorDType, TensorShape, TensorLayout), G2gError> {
        derive_output_caps(caps, self.target, self.out_layout).ok_or(G2gError::CapsMismatch)?;
        let Caps::Tensor { dtype, shape, layout } = caps else {
            return Err(G2gError::CapsMismatch);
        };
        Ok((*dtype, *shape, *layout))
    }

    /// Convert one frame's tensor bytes per the configured (dtype, layout) change.
    fn convert_frame(&self, src: &[u8]) -> Result<(Caps, Vec<u8>), G2gError> {
        let (in_dtype, in_shape, in_layout) = self.input.ok_or(G2gError::NotConfigured)?;
        let (out_dtype, conv) = resolve(in_dtype, self.target).ok_or(G2gError::CapsMismatch)?;
        let count = in_shape.elements();
        if src.len() < count * in_dtype.size() {
            return Err(G2gError::CapsMismatch);
        }

        // Step 1: elementwise dtype op, output bytes in the input element order.
        let elems: Vec<u8> = match conv {
            Conversion::Identity => src[..count * in_dtype.size()].to_vec(),
            Conversion::Quantize => {
                let floats: Vec<f32> = src
                    .chunks_exact(4)
                    .take(count)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                quantize_f32(&floats, out_dtype, self.scale, self.zero_point).ok_or(G2gError::CapsMismatch)?
            }
            Conversion::Dequantize => {
                let floats = dequantize_to_f32(&src[..count], in_dtype, self.scale, self.zero_point)
                    .ok_or(G2gError::CapsMismatch)?;
                let mut bytes = Vec::with_capacity(count * 4);
                for f in floats {
                    bytes.extend_from_slice(&f.to_le_bytes());
                }
                bytes
            }
            Conversion::NarrowF16 => {
                let mut bytes = Vec::with_capacity(count * 2);
                for c in src.chunks_exact(4).take(count) {
                    let f = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    bytes.extend_from_slice(&f32_to_f16_bits(f).to_le_bytes());
                }
                bytes
            }
            Conversion::WidenF16 => {
                let mut bytes = Vec::with_capacity(count * 4);
                for c in src.chunks_exact(2).take(count) {
                    let f = f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]));
                    bytes.extend_from_slice(&f.to_le_bytes());
                }
                bytes
            }
        };

        // Step 2: layout transpose (if requested and the layout actually differs).
        let to = self.out_layout.unwrap_or(in_layout);
        let (out_shape, out_bytes) = if to != in_layout {
            let (out_shape, perm) =
                layout_permutation(&in_shape, in_layout, to).ok_or(G2gError::CapsMismatch)?;
            let es = out_dtype.size();
            let mut reordered = alloc::vec![0u8; elems.len()];
            for (i, &s) in perm.iter().enumerate() {
                reordered[i * es..(i + 1) * es].copy_from_slice(&elems[s * es..(s + 1) * es]);
            }
            (out_shape, reordered)
        } else {
            (in_shape, elems)
        };

        let caps = Caps::Tensor { dtype: out_dtype, shape: out_shape, layout: to };
        Ok((caps, out_bytes))
    }
}

impl AsyncElement for TensorConvert {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // A tensor caps carries a concrete shape, so there is nothing to narrow:
        // accept it when the configured conversion is defined for its dtype/layout.
        self.accept_input(upstream_caps).map(|_| upstream_caps.clone())
    }

    /// Native `DerivedOutput`: a tensor input maps to the converted dtype, the
    /// requested layout, and the (possibly transposed) shape.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let target = self.target;
        let out_layout = self.out_layout;
        let derive = Box::new(move |input: &Caps| match derive_output_caps(input, target, out_layout) {
            Some(caps) => CapsSet::one(caps),
            None => CapsSet::from_alternatives(Vec::new()),
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
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let (new_caps, converted) = self.convert_frame(slice.as_slice())?;
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
                // Runner contract (the videoconvert / videoscale convention): the
                // transform arm calls `configure_pipeline` (the new input) then
                // pushes this packet carrying our pre-fixed forward *output* caps,
                // not a new input. So `c` is already our output: forward it and
                // record `last_caps` to suppress the data path's duplicate. Do NOT
                // `accept_input(c)` here, adopting our own output as the next input
                // is the stacked-convert bug (a u8 output read back as a u8 input
                // would resolve to an identity pass-through); the real input is set
                // by `configure_pipeline`.
                PipelinePacket::CapsChanged(c) => {
                    self.last_caps = Some(c.clone());
                    out.push(PipelinePacket::CapsChanged(c)).await?;
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
            "Tensor dtype / layout converter",
            "Filter/Converter/Tensor",
            "Quantizes/dequantizes f32 <-> int8/uint8, narrows/widens f32 <-> f16, transposes NCHW <-> NHWC",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "dtype" => {
                self.target = Some(dtype_from_str(value.as_str().ok_or(PropError::Type)?).ok_or(PropError::Value)?);
                Ok(())
            }
            "layout" => {
                self.out_layout = Some(layout_from_str(value.as_str().ok_or(PropError::Type)?).ok_or(PropError::Value)?);
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
            "dtype" => self.target.map(|d| PropValue::Str(dtype_to_str(d).into())),
            "layout" => self.out_layout.map(|l| PropValue::Str(layout_to_str(l).into())),
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
        // Only dtypes produced by `dtype_from_str` are ever stored, so any
        // future `TensorDType` variant cannot reach here.
        _ => unreachable!("tensorconvert names only dtypes it parsed"),
    }
}

/// Parse a tensor-layout property string.
fn layout_from_str(s: &str) -> Option<TensorLayout> {
    match s.to_ascii_lowercase().as_str() {
        "nchw" => Some(TensorLayout::Nchw),
        "nhwc" => Some(TensorLayout::Nhwc),
        _ => None,
    }
}

fn layout_to_str(l: TensorLayout) -> &'static str {
    match l {
        TensorLayout::Nchw => "nchw",
        TensorLayout::Nhwc => "nhwc",
        // Only layouts produced by `layout_from_str` are ever stored, so any
        // future `TensorLayout` variant cannot reach here.
        _ => unreachable!("tensorconvert names only layouts it parsed"),
    }
}

static TENSORCONVERT_PROPS: &[PropertySpec] = &[
    PropertySpec::new("dtype", PropKind::Str, "output dtype: f32 | f16 | i8 | u8 (int target = quantize, f16 = narrow, f32 = dequantize/widen)"),
    PropertySpec::new("layout", PropKind::Str, "output layout: nchw | nhwc (transpose a 4D tensor)"),
    PropertySpec::new("scale", PropKind::Double, "affine quantization scale"),
    PropertySpec::new("zero-point", PropKind::Int, "affine quantization zero point"),
];

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::vec;
    use core::pin::Pin;

    use g2g_core::frame::Frame;
    use g2g_core::memory::SystemSlice;
    use g2g_core::{FrameTiming, PushOutcome};

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
    fn resolve_picks_the_right_conversion_and_rejects_nonsense() {
        use TensorDType::*;
        assert_eq!(resolve(F32, Some(U8)), Some((U8, Conversion::Quantize)));
        assert_eq!(resolve(I8, Some(F32)), Some((F32, Conversion::Dequantize)));
        assert_eq!(resolve(F32, Some(F16)), Some((F16, Conversion::NarrowF16)));
        assert_eq!(resolve(F16, Some(F32)), Some((F32, Conversion::WidenF16)));
        assert_eq!(resolve(U8, None), Some((U8, Conversion::Identity)));
        // No defined affine for these:
        assert_eq!(resolve(U8, Some(I8)), None);
        assert_eq!(resolve(F16, Some(U8)), None);
    }

    #[test]
    fn f16_known_bit_patterns_round_trip() {
        assert_eq!(f32_to_f16_bits(1.0), 0x3C00);
        assert_eq!(f32_to_f16_bits(-2.0), 0xC000);
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(65504.0), 0x7BFF, "max normal half");
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7C00);
        assert_eq!(f16_bits_to_f32(0x3C00), 1.0);
        assert_eq!(f16_bits_to_f32(0x3800), 0.5);
        assert_eq!(f16_bits_to_f32(0xC000), -2.0);
        assert!(f16_bits_to_f32(0x7C00).is_infinite());
    }

    #[test]
    fn f16_narrow_widen_within_half_precision() {
        for &x in &[0.1f32, 1.0 / 3.0, -7.25, 100.0, 0.001, 12345.0] {
            let back = f16_bits_to_f32(f32_to_f16_bits(x));
            // ~2^-10 relative resolution, plus an absolute floor for tiny values.
            let tol = x.abs() / 1024.0 + 1e-6;
            assert!((x - back).abs() <= tol, "{x} -> {back}");
        }
    }

    #[test]
    fn layout_permutation_nchw_to_nhwc_reorders_channels_last() {
        // [1, 2, 2, 2]: NCHW flat index is the value at each position.
        let shape = TensorShape::new([1, 2, 2, 2]);
        let (out_shape, perm) =
            layout_permutation(&shape, TensorLayout::Nchw, TensorLayout::Nhwc).expect("4D transpose");
        assert_eq!(out_shape, TensorShape::new([1, 2, 2, 2]));
        assert_eq!(perm, vec![0, 4, 1, 5, 2, 6, 3, 7]);
        // Round trip back to NCHW is the identity permutation over the data.
        let (_, back) =
            layout_permutation(&out_shape, TensorLayout::Nhwc, TensorLayout::Nchw).expect("inverse");
        let composed: Vec<usize> = back.iter().map(|&i| perm[i]).collect();
        assert_eq!(composed, (0..8).collect::<Vec<_>>(), "NCHW->NHWC->NCHW is identity");
    }

    #[test]
    fn layout_transpose_rejected_for_non_4d() {
        let shape = TensorShape::new([1, 10]);
        assert!(layout_permutation(&shape, TensorLayout::Nchw, TensorLayout::Nhwc).is_none());
    }

    #[test]
    fn derive_output_caps_reflects_dtype_and_layout() {
        let nchw_f32 = Caps::Tensor {
            dtype: TensorDType::F32,
            shape: TensorShape::new([1, 3, 4, 4]),
            layout: TensorLayout::Nchw,
        };
        // quantize + transpose to NHWC uint8.
        let out = derive_output_caps(&nchw_f32, Some(TensorDType::U8), Some(TensorLayout::Nhwc)).unwrap();
        assert_eq!(
            out,
            Caps::Tensor {
                dtype: TensorDType::U8,
                shape: TensorShape::new([1, 4, 4, 3]),
                layout: TensorLayout::Nhwc,
            }
        );
        // narrow_f16 keeps shape and layout.
        let out = derive_output_caps(&nchw_f32, Some(TensorDType::F16), None).unwrap();
        assert!(matches!(out, Caps::Tensor { dtype: TensorDType::F16, layout: TensorLayout::Nchw, .. }));
    }

    // -- Element-level tests (drive TensorConvert::process directly) ---------

    #[derive(Default)]
    struct RecordingSink {
        packets: Vec<PipelinePacket>,
    }

    impl OutputSink for RecordingSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                self.packets.push(packet);
                Ok(PushOutcome::Accepted)
            })
        }
    }

    fn frame(bytes: Vec<u8>) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: 0,
            meta: Default::default(),
        })
    }

    fn first_data(sink: &RecordingSink) -> Vec<u8> {
        for p in &sink.packets {
            if let PipelinePacket::DataFrame(f) = p {
                let MemoryDomain::System(s) = &f.domain else { panic!("system frame") };
                return s.as_slice().to_vec();
            }
        }
        panic!("no DataFrame emitted");
    }

    fn first_caps(sink: &RecordingSink) -> Caps {
        for p in &sink.packets {
            if let PipelinePacket::CapsChanged(c) = p {
                return c.clone();
            }
        }
        panic!("no CapsChanged emitted");
    }

    #[tokio::test]
    async fn transpose_only_reorders_bytes_and_flips_layout() {
        // u8 [1,2,2,2] NCHW data 0..8 -> NHWC reorders to [0,4,1,5,2,6,3,7].
        let mut conv = TensorConvert::transpose(TensorLayout::Nhwc);
        let in_caps = Caps::Tensor {
            dtype: TensorDType::U8,
            shape: TensorShape::new([1, 2, 2, 2]),
            layout: TensorLayout::Nchw,
        };
        conv.configure_pipeline(&in_caps).unwrap();
        let mut sink = RecordingSink::default();
        conv.process(frame((0u8..8).collect()), &mut sink).await.unwrap();

        assert_eq!(first_data(&sink), vec![0, 4, 1, 5, 2, 6, 3, 7]);
        match first_caps(&sink) {
            Caps::Tensor { dtype, layout, .. } => {
                assert_eq!(dtype, TensorDType::U8);
                assert_eq!(layout, TensorLayout::Nhwc);
            }
            _ => panic!("tensor caps"),
        }
    }

    #[tokio::test]
    async fn quantize_and_transpose_compose_in_one_pass() {
        // f32 NCHW [1,2,1,2]: c0=[0.0, 0.5], c1=[1.0, 1.5]; scale 0.5 zp 0 -> u8.
        // Quantized NCHW = [0, 1, 2, 3]; NHWC (h,w,c) reorders to [0, 2, 1, 3].
        let mut conv = TensorConvert::quantize(TensorDType::U8, 0.5, 0).to_layout(TensorLayout::Nhwc);
        let in_caps = Caps::Tensor {
            dtype: TensorDType::F32,
            shape: TensorShape::new([1, 2, 1, 2]),
            layout: TensorLayout::Nchw,
        };
        conv.configure_pipeline(&in_caps).unwrap();
        let mut bytes = Vec::new();
        for v in [0.0f32, 0.5, 1.0, 1.5] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let mut sink = RecordingSink::default();
        conv.process(frame(bytes), &mut sink).await.unwrap();

        assert_eq!(first_data(&sink), vec![0, 2, 1, 3]);
        assert_eq!(
            first_caps(&sink),
            Caps::Tensor {
                dtype: TensorDType::U8,
                shape: TensorShape::new([1, 1, 2, 2]),
                layout: TensorLayout::Nhwc,
            }
        );
    }

    #[tokio::test]
    async fn narrow_to_f16_halves_the_byte_width() {
        let mut conv = TensorConvert::narrow_f16();
        let in_caps = Caps::Tensor {
            dtype: TensorDType::F32,
            shape: TensorShape::new([1, 4]),
            layout: TensorLayout::Nchw,
        };
        conv.configure_pipeline(&in_caps).unwrap();
        let mut bytes = Vec::new();
        for v in [1.0f32, 0.5, -2.0, 0.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let mut sink = RecordingSink::default();
        conv.process(frame(bytes), &mut sink).await.unwrap();

        let out = first_data(&sink);
        assert_eq!(out.len(), 4 * 2, "one f16 (2 bytes) per element");
        let halves: Vec<u16> = out.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        assert_eq!(halves, vec![0x3C00, 0x3800, 0xC000, 0x0000]);
    }
}
