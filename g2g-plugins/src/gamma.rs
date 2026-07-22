//! Gamma correction (`gamma`). Applies `out = 255 * (in/255)^(1/gamma)` to the
//! R/G/B channels of a packed RGBA / BGRA frame via a 256-entry lookup table,
//! leaving alpha and geometry unchanged. `gamma > 1` brightens mid-tones,
//! `gamma < 1` darkens them, `gamma == 1` is the identity. CPU-only `no_std`.
//!
//! The `pow` comes from the crate's `libm`-free approximation
//! ([`crate::mathf::powf`]), evaluated 256 times per gamma change (the LUT is
//! rebuilt only when `gamma` changes), not per pixel.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, Rate, RawVideoFormat,
};

const FORMATS: [RawVideoFormat; 2] = [RawVideoFormat::Rgba8, RawVideoFormat::Bgra8];

/// Build the gamma LUT: `lut[v] = round(255 * (v/255)^(1/gamma))`. `gamma <= 0`
/// or `gamma == 1` yields the identity table.
fn build_lut(gamma: f64) -> [u8; 256] {
    let mut lut = [0u8; 256];
    if gamma <= 0.0 || gamma == 1.0 {
        for (v, e) in lut.iter_mut().enumerate() {
            *e = v as u8;
        }
        return lut;
    }
    let inv = 1.0 / gamma;
    for (v, e) in lut.iter_mut().enumerate() {
        let normalized = v as f64 / 255.0;
        let corrected = crate::mathf::powf(normalized, inv) * 255.0;
        *e = (corrected + 0.5).clamp(0.0, 255.0) as u8;
    }
    lut
}

#[derive(Debug)]
pub struct Gamma {
    gamma: f64,
    lut: [u8; 256],
    input: Option<(RawVideoFormat, u32, u32, Rate)>,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl Default for Gamma {
    fn default() -> Self {
        Self::new()
    }
}

impl Gamma {
    /// Identity gamma (1.0); use the builder or the `gamma` property to adjust.
    pub fn new() -> Self {
        Self {
            gamma: 1.0,
            lut: build_lut(1.0),
            input: None,
            configured: false,
            last_caps: None,
            emitted: 0,
        }
    }

    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self.lut = build_lut(gamma);
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

impl AsyncElement for Gamma {
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

    /// Native `DerivedOutput`: gamma preserves format, geometry, and framerate.
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
                    let Some(src) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let bytes = (w as usize) * (h as usize) * 4;
                    if src.len() < bytes {
                        return Err(G2gError::CapsMismatch);
                    }
                    let mut dst = vec![0u8; bytes].into_boxed_slice();
                    apply_gamma(&src[..bytes], &mut dst, &self.lut);

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
        GAMMA_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Gamma",
            "Filter/Effect/Video",
            "Performs gamma correction",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "gamma" => {
                self.gamma = value.as_double().ok_or(PropError::Type)?;
                self.lut = build_lut(self.gamma);
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "gamma" => Some(PropValue::Double(self.gamma)),
            _ => None,
        }
    }
}

static GAMMA_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "gamma",
    PropKind::Double,
    "gamma value (>1 brightens, 1 = none)",
)];

impl PadTemplates for Gamma {
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

/// Apply the LUT to R/G/B of every packed pixel, leaving alpha (byte 3) as is.
/// The LUT is symmetric in R/G/B, so RGBA and BGRA both map channels 0/1/2.
fn apply_gamma(src: &[u8], dst: &mut [u8], lut: &[u8; 256]) {
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        d[0] = lut[s[0] as usize];
        d[1] = lut[s[1] as usize];
        d[2] = lut[s[2] as usize];
        d[3] = s[3];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_gamma_is_exact() {
        let lut = build_lut(1.0);
        for v in 0..=255u8 {
            assert_eq!(lut[v as usize], v);
        }
    }

    #[test]
    fn endpoints_are_fixed() {
        let lut = build_lut(2.2);
        assert_eq!(lut[0], 0);
        assert_eq!(lut[255], 255);
    }

    #[test]
    fn lut_is_monotonic() {
        let lut = build_lut(2.2);
        for v in 1..=255usize {
            assert!(lut[v] >= lut[v - 1], "non-monotonic at {v}");
        }
    }

    #[test]
    fn gamma_above_one_brightens_midtones() {
        let lut = build_lut(2.0);
        // mid-grey rises: (128/255)^(1/2) * 255 ~ 181.
        assert!(
            lut[128] > 128,
            "gamma 2.0 should brighten 128, got {}",
            lut[128]
        );
        assert!((lut[128] as i32 - 181).abs() <= 1);
    }

    #[test]
    fn alpha_is_preserved() {
        let lut = build_lut(2.0);
        let src = [10u8, 20, 30, 200];
        let mut dst = [0u8; 4];
        apply_gamma(&src, &mut dst, &lut);
        assert_eq!(dst[3], 200);
    }
}
