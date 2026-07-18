//! Deinterlace (`deinterlace`). Removes interlacing combs from a packed RGBA /
//! BGRA frame by vertical interpolation, preserving format and geometry. CPU-only
//! `no_std`.
//!
//! Two methods (a subset of GStreamer's `deinterlace`):
//! - `linear` (default): keep the even (top-field) lines, replace each odd line
//!   with the average of the even lines above and below it, so the bottom field's
//!   comb is discarded and interpolated.
//! - `blend`: each output line is the average of it and the line below, a soft
//!   vertical blur that suppresses combing without dropping a field.

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeinterlaceMethod {
    Linear,
    Blend,
}

impl DeinterlaceMethod {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "linear" => Some(Self::Linear),
            "blend" => Some(Self::Blend),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Blend => "blend",
        }
    }
}

#[derive(Debug)]
pub struct Deinterlace {
    method: DeinterlaceMethod,
    input: Option<(RawVideoFormat, u32, u32, Rate)>,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl Default for Deinterlace {
    fn default() -> Self {
        Self::new()
    }
}

impl Deinterlace {
    pub fn new() -> Self {
        Self {
            method: DeinterlaceMethod::Linear,
            input: None,
            configured: false,
            last_caps: None,
            emitted: 0,
        }
    }

    pub fn with_method(mut self, method: DeinterlaceMethod) -> Self {
        self.method = method;
        self
    }

    fn accept_input(&self, caps: &Caps) -> Result<(RawVideoFormat, u32, u32, Rate), G2gError> {
        let Caps::RawVideo { format, width: Dim::Fixed(w), height: Dim::Fixed(h), framerate } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !FORMATS.contains(format) || *w == 0 || *h == 0 {
            return Err(G2gError::CapsMismatch);
        }
        Ok((*format, *w, *h, framerate.clone()))
    }
}

impl AsyncElement for Deinterlace {
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

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::RawVideo { format, .. } if FORMATS.contains(format) => CapsSet::one(input.clone()),
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
                    deinterlace(&src[..bytes], &mut dst, w as usize, h as usize, self.method);

                    let new_caps = Caps::RawVideo {
                        format,
                        width: Dim::Fixed(w),
                        height: Dim::Fixed(h),
                        framerate: rate,
                    };
                    if self.last_caps.as_ref() != Some(&new_caps) {
                        out.push(PipelinePacket::CapsChanged(new_caps.clone())).await?;
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
        DEINTERLACE_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new("Deinterlace", "Filter/Effect/Video/Deinterlace", "Deinterlaces video", "g2g")
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "method" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.method = DeinterlaceMethod::from_str(s).ok_or(PropError::Value)?;
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "method" => Some(PropValue::Str(self.method.as_str().into())),
            _ => None,
        }
    }
}

static DEINTERLACE_PROPS: &[PropertySpec] =
    &[PropertySpec::new("method", PropKind::Str, "deinterlace method: linear | blend")];

impl PadTemplates for Deinterlace {
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

fn avg(a: u8, b: u8) -> u8 {
    ((a as u16 + b as u16) / 2) as u8
}

/// Deinterlace one packed 4-channel frame. `linear` interpolates odd lines from
/// the even lines above / below; `blend` averages each line with the one below.
fn deinterlace(src: &[u8], dst: &mut [u8], w: usize, h: usize, method: DeinterlaceMethod) {
    let stride = w * 4;
    let line = |y: usize| &src[y * stride..y * stride + stride];
    match method {
        DeinterlaceMethod::Linear => {
            for y in 0..h {
                let out = &mut dst[y * stride..y * stride + stride];
                if y % 2 == 0 || y + 1 >= h {
                    // even line (top field) or last row: passthrough.
                    out.copy_from_slice(line(y));
                } else {
                    let above = line(y - 1);
                    let below = line(y + 1);
                    for i in 0..stride {
                        out[i] = avg(above[i], below[i]);
                    }
                }
            }
        }
        DeinterlaceMethod::Blend => {
            for y in 0..h {
                let out = &mut dst[y * stride..y * stride + stride];
                let cur = line(y);
                if y + 1 >= h {
                    out.copy_from_slice(cur);
                } else {
                    let below = line(y + 1);
                    for i in 0..stride {
                        out[i] = avg(cur[i], below[i]);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1px-wide, 4px-tall RGBA frame with alternating black/white lines (a comb).
    fn comb() -> Vec<u8> {
        let mut v = Vec::new();
        for y in 0..4 {
            let c = if y % 2 == 0 { 0u8 } else { 255u8 };
            v.extend_from_slice(&[c, c, c, 255]);
        }
        v
    }

    #[test]
    fn linear_interpolates_odd_lines() {
        let src = comb();
        let mut dst = vec![0u8; src.len()];
        deinterlace(&src, &mut dst, 1, 4, DeinterlaceMethod::Linear);
        // even lines (0,2) stay 0; odd line 1 = avg(0,0)=0; line 3 is last -> passthrough 255.
        assert_eq!(dst[0..4], [0, 0, 0, 255]);
        assert_eq!(dst[4..8], [0, 0, 0, 255]);
        assert_eq!(dst[8..12], [0, 0, 0, 255]);
        assert_eq!(dst[12..16], [255, 255, 255, 255]);
    }

    #[test]
    fn blend_softens_edges() {
        let src = comb();
        let mut dst = vec![0u8; src.len()];
        deinterlace(&src, &mut dst, 1, 4, DeinterlaceMethod::Blend);
        // line 0 = avg(0,255)=127; the comb is reduced (no full 0/255 jump).
        assert_eq!(dst[0], 127);
        assert_eq!(dst[4], 127);
    }

    #[test]
    fn method_property_round_trips() {
        let mut d = Deinterlace::new();
        d.set_property("method", PropValue::Str("blend".into())).unwrap();
        assert_eq!(d.get_property("method"), Some(PropValue::Str("blend".into())));
        assert_eq!(
            d.set_property("method", PropValue::Str("nope".into())).unwrap_err(),
            PropError::Value
        );
    }
}
