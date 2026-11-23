// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Draw ordered bitmap overlays over raw video or attach them for the sink.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::future::Future;
use core::pin::Pin;

use g2g_core::meta::{MetaRequests, PlaneLayout};
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, ElementMetadata,
    Frame, G2gError, MemoryDomain, OutputSink, OverlayCompositionMeta, OverlayPlacement,
    OverlayRectangle, PadTemplate, PadTemplates, PipelinePacket, RawVideoFormat,
};

use crate::compositor::blend_over_scaled;
use crate::paint::{blend_layer_into_nv12, over_premultiplied};
use crate::pixel::{plane_shapes, rgba_rb_offsets};
use crate::yuvmatrix::YuvRgbMatrix;

#[cfg(any(feature = "wgpu-sink", feature = "wayland-sink", feature = "kms-sink"))]
pub(crate) const RENDER_OVERLAYS_NAME: &str = "render-overlays";

#[cfg(any(feature = "wgpu-sink", feature = "wayland-sink", feature = "kms-sink"))]
pub(crate) const RENDER_OVERLAYS_PROPERTY: g2g_core::PropertySpec = g2g_core::PropertySpec::new(
    RENDER_OVERLAYS_NAME,
    g2g_core::PropKind::Bool,
    "render bitmap overlay compositions attached to video frames",
)
.with_default("false");

#[cfg(any(feature = "wgpu-sink", feature = "wayland-sink", feature = "kms-sink"))]
pub(crate) fn render_requests(render: bool) -> MetaRequests {
    if render {
        MetaRequests::new().request_from_every_consumer::<OverlayCompositionMeta>()
    } else {
        MetaRequests::new()
    }
}

type Draw = dyn FnMut(&Frame, (u32, u32)) -> Option<OverlayCompositionMeta> + Send;

/// Uses a per-frame draw callback and any composition already on the frame.
/// Without a callback it renders or forwards the incoming composition alone.
#[derive(Default)]
pub struct OverlayComposition {
    draw: Option<Box<Draw>>,
    picture: Option<Picture>,
    downstream_renders: bool,
    layer: Vec<u8>,
}

impl core::fmt::Debug for OverlayComposition {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OverlayComposition")
            .field("picture", &self.picture)
            .field("downstream_renders", &self.downstream_renders)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct Picture {
    format: RawVideoFormat,
    width: u32,
    height: u32,
    matrix: YuvRgbMatrix,
}

impl OverlayComposition {
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate rectangles from the current frame and its picture size. The
    /// returned rectangles go on top of the upstream composition on the frame.
    pub fn with_draw(
        mut self,
        draw: impl FnMut(&Frame, (u32, u32)) -> Option<OverlayCompositionMeta> + Send + 'static,
    ) -> Self {
        self.draw = Some(Box::new(draw));
        self
    }

    fn compose(&mut self, frame: &mut Frame) -> Result<(), G2gError> {
        let picture = self.picture.as_ref().ok_or(G2gError::NotConfigured)?;
        if let Some(composition) = self
            .draw
            .as_mut()
            .and_then(|draw| draw(frame, (picture.width, picture.height)))
        {
            if !composition.rectangles().is_empty() {
                let composition = match frame.meta.get::<OverlayCompositionMeta>() {
                    Some(upstream) => OverlayCompositionMeta::new(
                        upstream
                            .rectangles()
                            .iter()
                            .chain(composition.rectangles())
                            .cloned()
                            .collect(),
                    ),
                    None => composition,
                };
                frame.meta.attach(composition);
            }
        }
        if !self.downstream_renders {
            if let Some(composition) = frame.meta.get::<OverlayCompositionMeta>() {
                picture.blend(
                    composition,
                    &mut frame.domain,
                    frame.meta.get::<PlaneLayout>(),
                    &mut self.layer,
                )?;
                frame.meta.remove::<OverlayCompositionMeta>();
            }
        }
        Ok(())
    }
}

impl Picture {
    fn blend(
        &self,
        composition: &OverlayCompositionMeta,
        domain: &mut MemoryDomain,
        layout: Option<&PlaneLayout>,
        layer: &mut Vec<u8>,
    ) -> Result<(), G2gError> {
        if !composition
            .rectangles()
            .iter()
            .any(|rectangle| visible(rectangle, self.width, self.height))
        {
            return Ok(());
        }
        if !matches!(
            self.format,
            RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8 | RawVideoFormat::Nv12
        ) {
            return Err(G2gError::CapsMismatch);
        }
        let MemoryDomain::System(system) = domain else {
            return Err(G2gError::UnsupportedDomain);
        };
        let tight = PlaneLayout::tight(self.format, self.width, self.height)
            .ok_or(G2gError::CapsMismatch)?;
        let layout = layout.unwrap_or(&tight);
        let (width, height) = (self.width as usize, self.height as usize);
        let shapes = plane_shapes(self.format, width, height).ok_or(G2gError::CapsMismatch)?;
        if shapes.len() != layout.count()
            || shapes
                .iter()
                .enumerate()
                .any(|(plane, &(row_bytes, rows))| {
                    layout
                        .row_range(plane, rows - 1, row_bytes)
                        .is_none_or(|range| range.end > system.as_slice().len())
                })
        {
            return Err(G2gError::CapsMismatch);
        }
        let layer = composition_layer(composition, (self.width, self.height), layer)?;
        let bytes = system.as_mut_slice();
        if self.format == RawVideoFormat::Nv12 {
            blend_layer_into_nv12(layer, bytes, layout, (width, height), &self.matrix)
        } else {
            let (red, blue) = rgba_rb_offsets(self.format);
            for (y, layer_row) in layer.chunks_exact(width * 4).enumerate() {
                let row = layout
                    .row_range(0, y, layer_row.len())
                    .ok_or(G2gError::CapsMismatch)?;
                for (dst, src) in bytes[row]
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .zip(layer_row.as_chunks::<4>().0)
                {
                    over_premultiplied(dst, [src[red], src[1], src[blue], src[3]]);
                }
            }
            Ok(())
        }
    }
}

fn visible(rectangle: &OverlayRectangle, width: u32, height: u32) -> bool {
    rectangle.placement().clipped_to(width, height).is_some()
}

pub(crate) fn composition_layer<'a>(
    composition: &'a OverlayCompositionMeta,
    (width, height): (u32, u32),
    layer: &'a mut Vec<u8>,
) -> Result<&'a [u8], G2gError> {
    let whole_picture = OverlayPlacement {
        x: 0,
        y: 0,
        width,
        height,
    };
    if let [rectangle] = composition.rectangles() {
        if rectangle.placement() == whole_picture && rectangle.pixel_size() == (width, height) {
            return Ok(rectangle.pixels());
        }
    }
    let (width, height) = (width as usize, height as usize);
    let layer_len = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(G2gError::CapsMismatch)?;
    layer.resize(layer_len, 0);
    layer.fill(0);
    for rectangle in composition
        .rectangles()
        .iter()
        .filter(|rectangle| visible(rectangle, whole_picture.width, whole_picture.height))
    {
        let placement = rectangle.placement();
        let (sw, sh) = rectangle.pixel_size();
        blend_over_scaled(
            layer,
            width,
            height,
            rectangle.pixels(),
            sw as usize,
            sh as usize,
            placement.x,
            placement.y,
            placement.width as usize,
            placement.height as usize,
            255,
            paint_layer,
        );
    }
    Ok(layer)
}

// The scratch layer holds premultiplied channels so resampling and stacking
// translucent overlays do not pull in the RGB values of transparent pixels.
fn paint_layer(layer: &mut [u8], offset: usize, source: [u8; 4], _alpha: u8) {
    let inverse = 255 - u16::from(source[3]);
    for (dst, src) in layer[offset..offset + 4].iter_mut().zip(source) {
        let value = u16::from(src) + (u16::from(*dst) * inverse + 127) / 255;
        *dst = value.min(255) as u8;
    }
}

impl AsyncElement for OverlayComposition {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Overlay composition",
            "Filter/Video",
            "Attaches or renders bitmap overlay rectangles",
            crate::AARON_BOXER,
        )
    }

    fn output_domain_follows_input(&self) -> bool {
        true
    }

    fn intercept_caps(&self, caps: &Caps) -> Result<Caps, G2gError> {
        match caps {
            Caps::RawVideo { .. } => Ok(caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(crate::fakemediasink::raw_video_caps())
    }

    fn configure_pipeline(&mut self, caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match *caps {
            Caps::RawVideo {
                format,
                width: Dim::Fixed(width),
                height: Dim::Fixed(height),
                colorimetry,
                ..
            } if width > 0 && height > 0 => {
                self.picture = Some(Picture {
                    format,
                    width,
                    height,
                    matrix: YuvRgbMatrix::new(colorimetry),
                });
                Ok(ConfigureOutcome::Accepted)
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn configure_allocation(&mut self, params: &AllocationParams) {
        self.downstream_renders = params.meta_requests.wants::<OverlayCompositionMeta>();
    }

    fn meta_requests(&self) -> MetaRequests {
        MetaRequests::new().request_from_every_consumer::<OverlayCompositionMeta>()
    }

    fn process<'a>(
        &'a mut self,
        mut packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match &mut packet {
                PipelinePacket::DataFrame(frame) => self.compose(frame)?,
                PipelinePacket::CapsChanged(caps) => {
                    self.configure_pipeline(caps)?;
                }
                PipelinePacket::Eos => return Ok(()),
                _ => {}
            }
            out.push(packet).await?;
            Ok(())
        })
    }
}

impl PadTemplates for OverlayComposition {
    fn pad_templates() -> Vec<PadTemplate> {
        crate::pcm_passthrough::pad_templates(crate::fakemediasink::raw_video_caps())
    }
}
