//! Alpha control / chroma key (`alpha`). Rewrites the alpha channel of a packed
//! RGBA / BGRA frame, leaving the colour channels untouched. CPU-only `no_std`.
//!
//! `set` replaces alpha with a constant (`alpha` 0..1); `green` / `blue` are a
//! simple chroma key that makes a pixel transparent when the key channel
//! dominates the other two by [`KEY_MARGIN`], opaque otherwise. The key is a
//! dominance test, not the full YUV-distance keyer GStreamer's `alpha` ships, so
//! it stays integer-only and libm-free.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, MemoryDomain,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec, Rate, RawVideoFormat,
};

use crate::pixel::rgba_rb_offsets;

const FORMATS: [RawVideoFormat; 2] = [RawVideoFormat::Rgba8, RawVideoFormat::Bgra8];

/// How far (0..255) the key channel must exceed the other two colour channels
/// for a chroma-key pixel to be treated as background and made transparent.
const KEY_MARGIN: i16 = 40;

/// What the element does to the alpha channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaMethod {
    /// Replace alpha with the constant `alpha` value for every pixel.
    Set,
    /// Chroma key: green pixels become transparent, the rest opaque.
    Green,
    /// Chroma key: blue pixels become transparent, the rest opaque.
    Blue,
}

#[derive(Debug)]
pub struct Alpha {
    method: AlphaMethod,
    alpha: f64,
    input: Option<(RawVideoFormat, u32, u32, Rate)>,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl Default for Alpha {
    fn default() -> Self {
        Self::new()
    }
}

impl Alpha {
    /// A `set` element at full opacity (alpha 1.0); use the builders or the
    /// properties to pick a method or constant.
    pub fn new() -> Self {
        Self {
            method: AlphaMethod::Set,
            alpha: 1.0,
            input: None,
            configured: false,
            last_caps: None,
            emitted: 0,
        }
    }

    pub fn with_method(mut self, method: AlphaMethod) -> Self {
        self.method = method;
        self
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    fn accept_input(&self, caps: &Caps) -> Result<(RawVideoFormat, u32, u32, Rate), G2gError> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate,
        } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !FORMATS.contains(format) || *w == 0 || *h == 0 {
            return Err(G2gError::CapsMismatch);
        }
        Ok((*format, *w, *h, framerate.clone()))
    }
}

impl AsyncElement for Alpha {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        for format in FORMATS {
            let candidate = Caps::RawVideo {
                format,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
            };
            if let Ok(narrowed) = upstream_caps.intersect(&candidate) {
                return Ok(narrowed);
            }
        }
        Err(G2gError::CapsMismatch)
    }

    /// Native `DerivedOutput`: rewriting alpha preserves format, geometry, and
    /// framerate, so the output caps equal the input for any supported format.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::RawVideo { format, .. } if FORMATS.contains(format) => {
                CapsSet::one(input.clone())
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.input = Some(self.accept_input(absolute_caps)?);
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
                    let (format, w, h, rate) = match &self.input {
                        Some((f, w, h, r)) => (*f, *w, *h, r.clone()),
                        None => return Err(G2gError::NotConfigured),
                    };
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let src = slice.as_slice();
                    let bytes = (w as usize) * (h as usize) * 4;
                    if src.len() < bytes {
                        return Err(G2gError::CapsMismatch);
                    }
                    let mut dst = vec![0u8; bytes].into_boxed_slice();
                    apply_alpha(
                        format,
                        &src[..bytes],
                        &mut dst,
                        self.method,
                        alpha_u8(self.alpha),
                    );

                    let new_caps = Caps::RawVideo {
                        format,
                        width: Dim::Fixed(w),
                        height: Dim::Fixed(h),
                        framerate: rate,
                    };
                    if self.last_caps.as_ref() != Some(&new_caps) {
                        out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                            .await?;
                        self.last_caps = Some(new_caps);
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
                    self.input = Some(self.accept_input(&c)?);
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
        ALPHA_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "method" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.method = method_from_str(s).ok_or(PropError::Value)?;
            }
            "alpha" => self.alpha = value.as_double().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "method" => Some(PropValue::Str(method_to_str(self.method).into())),
            "alpha" => Some(PropValue::Double(self.alpha)),
            _ => None,
        }
    }
}

/// `Alpha`'s settable properties (M104).
static ALPHA_PROPS: &[PropertySpec] = &[
    PropertySpec::new("method", PropKind::Str, "alpha op: set | green | blue"),
    PropertySpec::new("alpha", PropKind::Double, "constant alpha for 'set', 0..1"),
];

fn method_from_str(s: &str) -> Option<AlphaMethod> {
    match s {
        "set" => Some(AlphaMethod::Set),
        "green" => Some(AlphaMethod::Green),
        "blue" => Some(AlphaMethod::Blue),
        _ => None,
    }
}

fn method_to_str(m: AlphaMethod) -> &'static str {
    match m {
        AlphaMethod::Set => "set",
        AlphaMethod::Green => "green",
        AlphaMethod::Blue => "blue",
    }
}

fn alpha_u8(alpha: f64) -> u8 {
    // round to nearest without libm; the value is non-negative after clamping.
    (alpha.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

impl PadTemplates for Alpha {
    fn pad_templates() -> Vec<PadTemplate> {
        let any_geometry = |format| Caps::RawVideo {
            format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        let set = CapsSet::from_alternatives(FORMATS.map(any_geometry).to_vec());
        Vec::from([PadTemplate::sink(set.clone()), PadTemplate::source(set)])
    }
}

/// The new alpha byte for one pixel under `method`. `set` returns the constant;
/// the chroma keys return 0 (transparent) when the key channel dominates, else
/// 255 (opaque).
fn pixel_alpha(format: RawVideoFormat, px: &[u8], method: AlphaMethod, set_alpha: u8) -> u8 {
    match method {
        AlphaMethod::Set => set_alpha,
        AlphaMethod::Green | AlphaMethod::Blue => {
            let (r_idx, b_idx) = rgba_rb_offsets(format);
            let r = px[r_idx] as i16;
            let g = px[1] as i16;
            let b = px[b_idx] as i16;
            let keyed = match method {
                AlphaMethod::Green => g - r > KEY_MARGIN && g - b > KEY_MARGIN,
                AlphaMethod::Blue => b - r > KEY_MARGIN && b - g > KEY_MARGIN,
                AlphaMethod::Set => unreachable!(),
            };
            if keyed {
                0
            } else {
                255
            }
        }
    }
}

/// Rewrite the alpha channel of every pixel, copying the colour channels through.
fn apply_alpha(
    format: RawVideoFormat,
    src: &[u8],
    dst: &mut [u8],
    method: AlphaMethod,
    set_alpha: u8,
) {
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
        d[3] = pixel_alpha(format, s, method, set_alpha);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_replaces_alpha_and_keeps_colour() {
        // alpha 0.5 -> round(127.5) = 128; RGB untouched.
        let mut dst = [0u8; 4];
        apply_alpha(
            RawVideoFormat::Rgba8,
            &[10, 20, 30, 255],
            &mut dst,
            AlphaMethod::Set,
            alpha_u8(0.5),
        );
        assert_eq!(dst, [10, 20, 30, 128]);
        assert_eq!(alpha_u8(1.0), 255);
        assert_eq!(alpha_u8(0.0), 0);
    }

    #[test]
    fn green_key_makes_green_transparent_only() {
        let key = |px: &[u8; 4]| pixel_alpha(RawVideoFormat::Rgba8, px, AlphaMethod::Green, 255);
        assert_eq!(key(&[0, 255, 0, 255]), 0, "green keyed out");
        assert_eq!(key(&[255, 0, 0, 255]), 255, "red opaque");
        assert_eq!(
            key(&[128, 128, 128, 255]),
            255,
            "grey opaque (no dominance)"
        );
    }

    #[test]
    fn blue_key_respects_bgra_channel_order() {
        // BGRA blue pixel is [255, 0, 0, A]; the key must read blue at index 0,
        // not the byte that would be red in RGBA.
        let bgra_blue = [255u8, 0, 0, 255];
        assert_eq!(
            pixel_alpha(RawVideoFormat::Bgra8, &bgra_blue, AlphaMethod::Blue, 255),
            0
        );
        // The same bytes read as RGBA are a red pixel, untouched by a blue key.
        assert_eq!(
            pixel_alpha(RawVideoFormat::Rgba8, &bgra_blue, AlphaMethod::Blue, 255),
            255
        );
    }
}
