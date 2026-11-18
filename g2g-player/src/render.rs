// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::ffi::c_void;
use std::sync::Arc;

use g2g_core::{Caps, G2gError};
use g2g_plugins::glnv12::{GlTarget, GlVideoRenderer};
use g2g_plugins::{FilledRectangle, OverlayImage, ViewportRectangle};

use crate::picture_filters::video_size_of;
use crate::subtitles::ShowingCues;
use crate::videosink::DueFrame;
use crate::{OsdDimensions, Player, Shared};

#[derive(Debug)]
struct FrameFeed {
    shared: Arc<Shared>,
    current: Option<DueFrame>,
    configured_caps: Option<Caps>,
    subtitle_image: Option<SubtitleImage>,
    subtitle_images_drawn: u64,
}

#[derive(Debug)]
struct SubtitleImage {
    showing: ShowingCues,
    image: OverlayImage,
}

impl FrameFeed {
    fn new(player: &Player) -> Self {
        Self {
            shared: player.shared.clone(),
            current: None,
            configured_caps: None,
            subtitle_image: None,
            subtitle_images_drawn: 0,
        }
    }

    fn latest(&mut self) -> Option<(&DueFrame, Option<&Caps>)> {
        if let Some(frame) = self.shared.frames.take_for_render() {
            self.current = Some(frame);
        }
        let current = self.current.as_ref()?;
        if self.configured_caps.as_ref() == Some(&current.caps) {
            return Some((current, None));
        }
        self.configured_caps = Some(current.caps.clone());
        Some((current, Some(&current.caps)))
    }

    fn overlay_in_target(&self, picture: ViewportRectangle) -> Vec<FilledRectangle> {
        let overlay = self.shared.overlay.lock().expect("overlay lock");
        if overlay.is_empty() {
            return Vec::new();
        }
        let drawn = self.current.as_ref().expect("a frame was drawn");
        let size = video_size_of(&drawn.caps).expect("playervideosink caps have a fixed size");
        let scale_x = picture.width as f32 / size.width as f32;
        let scale_y = picture.height as f32 / size.height as f32;
        overlay
            .iter()
            .map(|rectangle| {
                let [red, green, blue] = rectangle.colour;
                FilledRectangle {
                    x: picture.x as f32 + rectangle.x as f32 * scale_x,
                    y: picture.y as f32 + rectangle.y as f32 * scale_y,
                    width: rectangle.width as f32 * scale_x,
                    height: rectangle.height as f32 * scale_y,
                    rgba: [red, green, blue, rectangle.alpha],
                }
            })
            .collect()
    }

    fn subtitles_in_target(&mut self, picture: ViewportRectangle) -> Option<&OverlayImage> {
        let pts_ns = self.current.as_ref()?.frame.timing.pts_ns;
        let mut subtitles = self.shared.subtitles.lock().expect("subtitles lock");
        let showing = subtitles.showing_at(pts_ns);
        let size = (picture.width, picture.height);
        if showing.is_empty() || picture.width == 0 || picture.height == 0 {
            self.subtitle_image = None;
            return None;
        }
        let cached = self.subtitle_image.take().filter(|cached| {
            cached.showing == showing && (cached.image.width, cached.image.height) == size
        });
        let mut drawn = match cached {
            Some(cached) => cached,
            None => {
                let mut rgba = vec![0; picture.width as usize * picture.height as usize * 4];
                subtitles.render(&mut rgba, size, pts_ns);
                self.subtitle_images_drawn += 1;
                SubtitleImage {
                    showing,
                    image: OverlayImage {
                        placement: picture,
                        width: picture.width,
                        height: picture.height,
                        rgba: rgba.into(),
                        generation: self.subtitle_images_drawn,
                    },
                }
            }
        };
        drawn.image.placement = picture;
        Some(&self.subtitle_image.insert(drawn).image)
    }

    fn report_drawn(&self, picture: ViewportRectangle, (width, height): (u32, u32)) {
        self.shared.set_osd_dimensions(OsdDimensions {
            width,
            height,
            margin_left: picture.x,
            margin_top: picture.y,
            margin_right: width - picture.x - picture.width,
            margin_bottom: height - picture.y - picture.height,
        });
    }
}

#[derive(Debug)]
pub struct GlRenderContext {
    feed: FrameFeed,
    renderer: GlVideoRenderer,
}

impl GlRenderContext {
    pub fn new(
        player: &Player,
        get_proc_address: impl Fn(&str) -> *const c_void + 'static,
    ) -> Self {
        Self {
            feed: FrameFeed::new(player),
            renderer: GlVideoRenderer::new(get_proc_address),
        }
    }

    /// # Safety
    /// The GL context `get_proc_address` resolves entry points for must be
    /// current, with `target.framebuffer` a framebuffer of it.
    pub unsafe fn render(&mut self, target: GlTarget) -> Result<(), G2gError> {
        let Some((frame, new_caps)) = self.feed.latest() else {
            return Ok(());
        };
        if let Some(caps) = new_caps {
            self.renderer.configure(caps)?;
        }
        // SAFETY: forwarded from the caller.
        let picture = unsafe { self.renderer.draw(&frame.frame.domain, target) }?;
        let overlay = self.feed.overlay_in_target(picture);
        if !overlay.is_empty() {
            // SAFETY: forwarded from the caller.
            unsafe { self.renderer.draw_rectangles(&overlay, target) }?;
        }
        if let Some(subtitles) = self.feed.subtitles_in_target(picture) {
            // SAFETY: forwarded from the caller.
            unsafe { self.renderer.draw_image(subtitles, target) }?;
        }
        self.feed
            .report_drawn(picture, (target.width, target.height));
        Ok(())
    }

    /// # Safety
    /// The GL context `get_proc_address` resolves entry points for must be current.
    pub unsafe fn destroy(self) {
        // SAFETY: forwarded from the caller.
        unsafe { self.renderer.destroy() }
    }
}

#[cfg(feature = "wgpu")]
pub use wgpu_target::WgpuRenderContext;

#[cfg(feature = "wgpu")]
mod wgpu_target {
    use g2g_core::G2gError;
    use g2g_plugins::gpu::GpuContext;
    use g2g_plugins::wgpusink::VideoRenderer;

    use super::FrameFeed;
    use crate::Player;

    fn dither_bits(format: wgpu::TextureFormat) -> Option<u32> {
        match format {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm => Some(8),
            wgpu::TextureFormat::Rgb10a2Unorm => Some(10),
            // srgb targets encode the shader's output afterwards, and float targets have no fixed step
            _ => None,
        }
    }

    #[derive(Debug)]
    pub struct WgpuRenderContext {
        feed: FrameFeed,
        context: GpuContext,
        renderer: Option<VideoRenderer>,
    }

    impl WgpuRenderContext {
        pub fn new(player: &Player, context: GpuContext) -> Self {
            Self {
                feed: FrameFeed::new(player),
                context,
                renderer: None,
            }
        }

        pub fn render(&mut self, target: &wgpu::Texture) -> Result<(), G2gError> {
            let format = target.format();
            if self
                .renderer
                .as_ref()
                .is_none_or(|renderer| renderer.target_format() != format)
            {
                self.renderer = Some(VideoRenderer::new(
                    self.context.clone(),
                    format,
                    dither_bits(format),
                ));
                self.feed.configured_caps = None;
            }
            let renderer = self.renderer.as_mut().expect("built above");
            let Some((frame, new_caps)) = self.feed.latest() else {
                return Ok(());
            };
            if let Some(caps) = new_caps {
                renderer.configure(caps)?;
            }
            let view = target.create_view(&wgpu::TextureViewDescriptor::default());
            let target_size = (target.width(), target.height());
            let picture = renderer.draw(&frame.frame.domain, &view, target_size)?;
            renderer.draw_rectangles(&self.feed.overlay_in_target(picture), &view, target_size);
            if let Some(subtitles) = self.feed.subtitles_in_target(picture) {
                renderer.draw_image(subtitles, &view)?;
            }
            self.feed.report_drawn(picture, target_size);
            Ok(())
        }
    }
}
