// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::ffi::c_void;
use std::sync::Arc;

use g2g_core::{Caps, G2gError};
use g2g_plugins::glnv12::{GlTarget, GlVideoRenderer};

use crate::videosink::DueFrame;
use crate::{Player, Shared};

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
        if let Some(frame) = self.shared.frames.take() {
            self.current = Some(frame);
        }
        let current = self.current.as_ref()?;
        if self.configured_caps.as_ref() == Some(&current.caps) {
            return Some((current, None));
        }
        self.configured_caps = Some(current.caps.clone());
        Some((current, Some(&current.caps)))
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
        unsafe { self.renderer.draw(&frame.frame.domain, target) }
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
            renderer.draw(
                &frame.frame.domain,
                &view,
                (target.width(), target.height()),
            )
        }
    }
}
