// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::ffi::c_void;
use std::sync::Arc;

use g2g_core::{Caps, G2gError};
use g2g_plugins::glnv12::{GlTarget, GlVideoRenderer};
use g2g_plugins::{FilledRectangle, ViewportRectangle};

use crate::picture_filters::video_size_of;
use crate::videosink::DueFrame;
use crate::{OsdDimensions, Player, Shared};

#[derive(Debug)]
struct FrameFeed {
    shared: Arc<Shared>,
    current: Option<DueFrame>,
    configured_caps: Option<Caps>,
}

impl FrameFeed {
    fn new(player: &Player) -> Self {
        Self {
            shared: player.shared.clone(),
            current: None,
            configured_caps: None,
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
                self.renderer = Some(VideoRenderer::new(self.context.clone(), format));
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
            self.feed.report_drawn(picture, target_size);
            Ok(())
        }
    }
}
