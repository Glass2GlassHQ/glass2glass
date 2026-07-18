//! Software colour balance (`videobalance`). Adjusts brightness, contrast, and
//! saturation of a packed RGBA / BGRA frame per pixel, preserving format and
//! geometry. CPU-only `no_std` baseline.
//!
//! `brightness` (-1..1) adds an offset, `contrast` (0..2) scales around mid-grey
//! (128), `saturation` (0..2) lerps each channel toward the pixel's Rec.601 luma
//! (0 = greyscale, 1 = unchanged, >1 = boosted), and `hue` (-1..1 = -180..180deg)
//! rotates the per-pixel chroma vector `(r-luma, g-luma, b-luma)` about the
//! Rec.601 luma axis (so luma is preserved, grey stays grey, and hue=0 is exactly
//! the pre-hue behaviour). The
//! `sin`/`cos` of the hue angle come from the crate's `libm`-free approximation
//! ([`crate::mathf`]), computed once per frame, not per pixel.

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

const FORMATS: [RawVideoFormat; 2] = [RawVideoFormat::Rgba8, RawVideoFormat::Bgra8];

#[derive(Debug)]
pub struct VideoBalance {
    brightness: f64,
    contrast: f64,
    saturation: f64,
    hue: f64,
    input: Option<(RawVideoFormat, u32, u32, Rate)>,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl Default for VideoBalance {
    fn default() -> Self {
        Self::new()
    }
}

impl VideoBalance {
    /// An identity balance (brightness 0, contrast 1, saturation 1, hue 0); use
    /// the `with_*` builders or the properties to adjust.
    pub fn new() -> Self {
        Self {
            brightness: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            hue: 0.0,
            input: None,
            configured: false,
            last_caps: None,
            emitted: 0,
        }
    }

    pub fn with_brightness(mut self, brightness: f64) -> Self {
        self.brightness = brightness;
        self
    }

    pub fn with_contrast(mut self, contrast: f64) -> Self {
        self.contrast = contrast;
        self
    }

    pub fn with_saturation(mut self, saturation: f64) -> Self {
        self.saturation = saturation;
        self
    }

    pub fn with_hue(mut self, hue: f64) -> Self {
        self.hue = hue;
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

impl AsyncElement for VideoBalance {
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

    /// Native `DerivedOutput`: a colour adjustment preserves format, geometry,
    /// and framerate, so the output caps equal the input for any supported raw
    /// format.
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
                    let bal =
                        Balance::new(self.brightness, self.contrast, self.saturation, self.hue);
                    apply_balance(format, &src[..bytes], &mut dst, &bal);

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
        VIDEOBALANCE_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let v = value.as_double().ok_or(PropError::Type)?;
        match name {
            "brightness" => self.brightness = v,
            "contrast" => self.contrast = v,
            "saturation" => self.saturation = v,
            "hue" => self.hue = v,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "brightness" => Some(PropValue::Double(self.brightness)),
            "contrast" => Some(PropValue::Double(self.contrast)),
            "saturation" => Some(PropValue::Double(self.saturation)),
            "hue" => Some(PropValue::Double(self.hue)),
            _ => None,
        }
    }
}

/// `VideoBalance`'s settable properties (M104).
static VIDEOBALANCE_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "brightness",
        PropKind::Double,
        "additive brightness, -1..1 (0 = none)",
    ),
    PropertySpec::new(
        "contrast",
        PropKind::Double,
        "contrast about mid-grey, 0..2 (1 = none)",
    ),
    PropertySpec::new(
        "saturation",
        PropKind::Double,
        "saturation, 0..2 (0 = grey, 1 = none)",
    ),
    PropertySpec::new(
        "hue",
        PropKind::Double,
        "hue rotation, -1..1 (-180..180 deg, 0 = none)",
    ),
];

impl PadTemplates for VideoBalance {
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

fn clamp_u8(v: f64) -> u8 {
    v.clamp(0.0, 255.0) as u8
}

/// Frame-constant balance coefficients: the raw brightness / contrast /
/// saturation plus the precomputed sine / cosine of the hue angle, so the trig
/// runs once per frame rather than per pixel.
#[derive(Debug, Clone, Copy)]
struct Balance {
    bright: f64,
    contrast: f64,
    sat: f64,
    cos_h: f64,
    sin_h: f64,
}

impl Balance {
    fn new(bright: f64, contrast: f64, sat: f64, hue: f64) -> Self {
        // hue -1..1 -> angle -pi..pi, i.e. turns t = hue / 2.
        let t = (hue * 0.5) as f32;
        Self {
            bright,
            contrast,
            sat,
            cos_h: crate::mathf::cos_turns(t) as f64,
            sin_h: crate::mathf::sin_turns(t) as f64,
        }
    }
}

/// The Rec.601 luma weights normalized to a unit rotation axis. Rotating the
/// chroma vector about this axis preserves luma exactly: the chroma `d` already
/// satisfies `w.d = 0` (the weights sum to one), so an in-plane rotation leaves
/// `w.d` at zero and the output luma unchanged.
const LUMA_AXIS: [f64; 3] = [0.447_179_5, 0.877_952_5, 0.170_502_5];

/// Apply brightness / contrast / saturation / hue to one RGBA / BGRA pixel.
/// Contrast scales about 128 and brightness adds `b*255`; the chroma vector
/// `(r-luma, g-luma, b-luma)` is then rotated about the Rec.601 luma axis by the
/// hue angle (Rodrigues, luma-preserving) and scaled by saturation before luma
/// is added back.
fn balance_pixel(format: RawVideoFormat, px: &[u8], bal: &Balance) -> [u8; 4] {
    let (r_idx, b_idx) = crate::pixel::rgba_rb_offsets(format);
    let adj = |c: u8| (c as f64 - 128.0) * bal.contrast + 128.0 + bal.bright * 255.0;
    let r = adj(px[r_idx]);
    let g = adj(px[1]);
    let b = adj(px[b_idx]);
    let luma = 0.299 * r + 0.587 * g + 0.114 * b;

    // chroma about luma, rotated around the luma axis by the hue angle. Because
    // k is parallel to the weights, k.d = 0, so Rodrigues collapses to
    // d' = d*cos + (k x d)*sin.
    let (dr, dg, db) = (r - luma, g - luma, b - luma);
    let (c, s) = (bal.cos_h, bal.sin_h);
    let [kr, kg, kb] = LUMA_AXIS;
    let rr = dr * c + (kg * db - kb * dg) * s;
    let rg = dg * c + (kb * dr - kr * db) * s;
    let rb = db * c + (kr * dg - kg * dr) * s;

    let mut out = [0u8; 4];
    out[r_idx] = clamp_u8(luma + bal.sat * rr);
    out[1] = clamp_u8(luma + bal.sat * rg);
    out[b_idx] = clamp_u8(luma + bal.sat * rb);
    out[3] = px[3];
    out
}

/// Apply the balance to every pixel of a packed 4-channel frame.
fn apply_balance(format: RawVideoFormat, src: &[u8], dst: &mut [u8], bal: &Balance) {
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        d.copy_from_slice(&balance_pixel(format, s, bal));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bal(bright: f64, contrast: f64, sat: f64, hue: f64) -> Balance {
        Balance::new(bright, contrast, sat, hue)
    }

    #[test]
    fn identity_is_exact() {
        // brightness 0, contrast 1, saturation 1, hue 0 reproduces the input byte
        // for byte (integer values round-trip through f64 exactly; hue=0 gives an
        // exact cos=1/sin=0 so the rotation is the identity).
        let src: Vec<u8> = (0..(4 * 4 * 4) as u8).collect();
        let mut dst = vec![0u8; src.len()];
        apply_balance(
            RawVideoFormat::Rgba8,
            &src,
            &mut dst,
            &bal(0.0, 1.0, 1.0, 0.0),
        );
        assert_eq!(dst, src);
    }

    #[test]
    fn brightness_adds_a_scaled_offset() {
        // +0.2 brightness adds 0.2*255 = 51 to each channel; alpha is untouched.
        let px = balance_pixel(
            RawVideoFormat::Rgba8,
            &[100, 100, 100, 200],
            &bal(0.2, 1.0, 1.0, 0.0),
        );
        assert_eq!(px, [151, 151, 151, 200]);
    }

    #[test]
    fn contrast_pivots_around_mid_grey() {
        // contrast 2 doubles the distance from 128: 100 -> 72, 128 stays put.
        assert_eq!(
            balance_pixel(
                RawVideoFormat::Rgba8,
                &[100, 100, 100, 255],
                &bal(0.0, 2.0, 1.0, 0.0)
            )[0],
            72
        );
        assert_eq!(
            balance_pixel(
                RawVideoFormat::Rgba8,
                &[128, 128, 128, 255],
                &bal(0.0, 2.0, 1.0, 0.0)
            )[0],
            128
        );
    }

    #[test]
    fn saturation_zero_is_greyscale() {
        // saturation 0 collapses every channel to the pixel's luma, so R=G=B.
        let px = balance_pixel(
            RawVideoFormat::Rgba8,
            &[200, 100, 50, 255],
            &bal(0.0, 1.0, 0.0, 0.0),
        );
        assert_eq!(px[0], px[1]);
        assert_eq!(px[1], px[2]);
        // Rec.601 luma of (200,100,50) = 59.8 + 58.7 + 5.7 = 124.2 -> 124.
        assert_eq!(px[0], 124);
    }

    #[test]
    fn bgra_weights_luma_by_true_colour() {
        // Same colour as above but stored BGRA: bytes [50,100,200,255]. Luma must
        // match the RGBA case (channel roles respected), not be computed on the
        // raw byte order.
        let px = balance_pixel(
            RawVideoFormat::Bgra8,
            &[50, 100, 200, 255],
            &bal(0.0, 1.0, 0.0, 0.0),
        );
        assert_eq!(px[0], px[2]);
        assert_eq!(px[0], 124, "luma uses true R/G/B, not byte order");
    }

    #[test]
    fn hue_leaves_grey_untouched() {
        // A grey pixel has zero chroma, so any hue rotation is a no-op (to within
        // one LSB of u8 quantization; the luma of grey is not exactly integral in
        // f64, so a rotated near-zero chroma can truncate a step).
        for hue in [-0.5, 0.25, 0.75, 1.0] {
            let px = balance_pixel(
                RawVideoFormat::Rgba8,
                &[128, 128, 128, 255],
                &bal(0.0, 1.0, 1.0, hue),
            );
            for (ch, &v) in px[..3].iter().enumerate() {
                assert!(
                    (v as i32 - 128).abs() <= 1,
                    "hue {hue} shifted grey channel {ch} to {v}"
                );
            }
            assert_eq!(px[3], 255);
        }
    }

    #[test]
    fn hue_rotation_cycles_red_toward_green() {
        // +120 deg (hue 2/3) about the grey axis maps red chroma toward green:
        // the green channel rises well above red, and luma stays close.
        let src = [200u8, 40, 40, 255];
        let px = balance_pixel(RawVideoFormat::Rgba8, &src, &bal(0.0, 1.0, 1.0, 2.0 / 3.0));
        assert!(px[1] > px[0], "green {} should exceed red {}", px[1], px[0]);
        assert!(
            px[1] > px[2],
            "green {} should exceed blue {}",
            px[1],
            px[2]
        );
        // the rotation is luma-preserving by construction; only u8 rounding drifts.
        let luma_in = 0.299 * 200.0 + 0.587 * 40.0 + 0.114 * 40.0;
        let luma_out = 0.299 * px[0] as f64 + 0.587 * px[1] as f64 + 0.114 * px[2] as f64;
        assert!(
            (luma_out - luma_in).abs() < 2.0,
            "luma drifted: {luma_in} -> {luma_out}"
        );
    }

    #[test]
    fn hue_sign_is_opposite_for_opposite_angles() {
        // +h and -h push a colour in opposite chroma directions.
        let src = [200u8, 40, 40, 255];
        let plus = balance_pixel(RawVideoFormat::Rgba8, &src, &bal(0.0, 1.0, 1.0, 0.25));
        let minus = balance_pixel(RawVideoFormat::Rgba8, &src, &bal(0.0, 1.0, 1.0, -0.25));
        assert_ne!(plus, minus);
        // green rises under one sign and blue under the other.
        assert!(
            (plus[1] as i32 - minus[1] as i32).signum()
                != (plus[2] as i32 - minus[2] as i32).signum()
        );
    }
}
