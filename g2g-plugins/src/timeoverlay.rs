//! Buffer-time overlay (`timeoverlay`). Burns each frame's PTS, formatted as
//! `HH:MM:SS.mmm`, into the top-left corner of a packed RGBA / BGRA frame with
//! the embedded 8x8 [`bitmapfont`], preserving format and geometry. CPU-only
//! `no_std`.
//!
//! The g2g analog of GStreamer's `timeoverlay` (buffer-time mode). Text is white
//! over a translucent black box for legibility; `scale` sets the integer font
//! magnification.
//!
//! [`bitmapfont`]: crate::bitmapfont

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, Rate, RawVideoFormat,
};

use crate::bitmapfont::{glyph, GLYPH_ADVANCE, GLYPH_HEIGHT};
use crate::paint::blend_px;

const FORMATS: [RawVideoFormat; 2] = [RawVideoFormat::Rgba8, RawVideoFormat::Bgra8];

/// Format a nanosecond timestamp as `HH:MM:SS.mmm`.
fn format_time(pts_ns: u64) -> String {
    let total_ms = pts_ns / 1_000_000;
    let ms = total_ms % 1000;
    let s = (total_ms / 1000) % 60;
    let m = (total_ms / 60_000) % 60;
    let h = total_ms / 3_600_000;
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}

#[derive(Debug)]
pub struct TimeOverlay {
    scale: u32,
    input: Option<(RawVideoFormat, u32, u32, Rate)>,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl Default for TimeOverlay {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeOverlay {
    pub fn new() -> Self {
        Self { scale: 2, input: None, configured: false, last_caps: None, emitted: 0 }
    }

    pub fn with_scale(mut self, scale: u32) -> Self {
        self.scale = scale.max(1);
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

impl AsyncElement for TimeOverlay {
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
                    dst.copy_from_slice(&src[..bytes]);
                    let text = format_time(frame.timing.pts_ns);
                    draw_text(&mut dst, w as usize, h as usize, &text, self.scale.max(1));

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
        TIMEOVERLAY_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new("Time overlay", "Filter/Editor/Video", "Overlays the buffer time on video", "g2g")
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "scale" => {
                let s = value.as_uint().ok_or(PropError::Type)?;
                if s == 0 {
                    return Err(PropError::Value);
                }
                self.scale = s as u32;
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "scale" => Some(PropValue::Uint(self.scale as u64)),
            _ => None,
        }
    }
}

static TIMEOVERLAY_PROPS: &[PropertySpec] =
    &[PropertySpec::new("scale", PropKind::Uint, "integer font magnification (>= 1)")];

impl PadTemplates for TimeOverlay {
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

/// Draw `text` at the top-left of a packed RGBA/BGRA buffer: a translucent black
/// box, then white glyphs. The glyph bitmap is channel-symmetric, so it renders
/// the same in RGBA and BGRA.
fn draw_text(buf: &mut [u8], w: usize, h: usize, text: &str, scale: u32) {
    let scale = scale as i32;
    let margin = 2 * scale;
    let cell_w = GLYPH_ADVANCE as i32 * scale;
    let glyph_h = GLYPH_HEIGHT as i32 * scale;
    let box_w = margin * 2 + cell_w * text.chars().count() as i32;
    let box_h = margin * 2 + glyph_h;
    let dims = (w, h);
    fill_rect(buf, dims, 0, 0, box_w, box_h, [0, 0, 0, 160]);
    let white = [255u8, 255, 255, 255];
    for (i, c) in text.chars().enumerate() {
        let gx = margin + i as i32 * cell_w;
        blit_glyph(buf, dims, gx, margin, scale, glyph(c), white);
    }
}

fn fill_rect(buf: &mut [u8], dims: (usize, usize), x: i32, y: i32, rw: i32, rh: i32, color: [u8; 4]) {
    let (wi, hi) = (dims.0 as i32, dims.1 as i32);
    for py in y..y + rh {
        if py < 0 || py >= hi {
            continue;
        }
        for px in x..x + rw {
            if px < 0 || px >= wi {
                continue;
            }
            blend_px(buf, ((py * wi + px) * 4) as usize, color, 255);
        }
    }
}

fn blit_glyph(buf: &mut [u8], dims: (usize, usize), gx: i32, gy: i32, scale: i32, rows: [u8; 8], color: [u8; 4]) {
    for (ry, bits) in rows.iter().enumerate() {
        if *bits == 0 {
            continue;
        }
        for col in 0..8i32 {
            if bits & (0x80 >> col) != 0 {
                fill_rect(buf, dims, gx + col * scale, gy + ry as i32 * scale, scale, scale, color);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_time_as_hms() {
        assert_eq!(format_time(0), "00:00:00.000");
        assert_eq!(format_time(1_500_000_000), "00:00:01.500");
        assert_eq!(format_time(3_661_250_000_000), "01:01:01.250");
    }

    #[test]
    fn draws_something_onto_a_blank_frame() {
        // 128x16 white RGBA frame; after overlay some pixels must differ (the box
        // + glyphs), proving the overlay actually wrote to the buffer.
        let (w, h) = (128usize, 16usize);
        let mut buf = vec![255u8; w * h * 4];
        let before = buf.clone();
        draw_text(&mut buf, w, h, "00:00:01.000", 1);
        assert_ne!(buf, before);
        // top-left pixel is inside the translucent black box, so it darkened.
        assert!(buf[0] < 255);
    }
}
