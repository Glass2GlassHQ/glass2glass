//! M575: HDR swapchain present.
//!
//! The last HDR layer: present a decoded HDR texture on screen through a
//! swapchain that carries an HDR colour space + mastering metadata, so an HDR
//! display shows it as HDR (not clipped / tone-mapped to SDR). The decoder's
//! `Passthrough` GPU output is an `Rgba16Float` texture holding the stream's
//! PQ-encoded BT.2020 R'G'B' - exactly what an HDR10 (`ST2084`) swapchain wants,
//! so present is a straight image copy with no further colour maths.
//!
//! wgpu 29's `SurfaceConfiguration` has no colour-space knob (it can only pick
//! `EXTENDED_SRGB_LINEAR` scRGB automatically for an `Rgba16Float` surface), so a
//! true HDR10-PQ swapchain needs raw Vulkan. [`VulkanHdrSink`] therefore owns a
//! raw `VK_KHR_swapchain` created on the **decode device's** `VkInstance` (the
//! extensions are enabled in `open_decode_device`): the decoded texture and the
//! swapchain share one device, so present is zero-copy on the GPU. It negotiates
//! the best colour space the surface offers (`HDR10_ST2084` PQ, else scRGB linear,
//! else SDR) and, when `VK_EXT_hdr_metadata` is present, attaches the mastering
//! display via `vkSetHdrMetadataEXT`.
//!
//! Present is CPU-synchronised (acquire fence, blit, submit fence, present) for
//! correctness over throughput: the decode + ycbcr convert already completed
//! synchronously before the texture reaches the sink, so the queue is idle.
//!
//! On-screen HDR output is display + compositor dependent and is validated live
//! by running `examples/vulkan_video_hdr_on_screen.rs` on an HDR display in HDR
//! mode; the headless-testable parts (surface-format / colour-space selection,
//! the `VkHdrMetadataEXT` construction) have unit tests, and the device opens with
//! the present extensions on the RTX 3060.

use ash::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::vulkanvideo::{PresentContext, VulkanVideoDevice, VulkanVideoError};

/// The colour space negotiated for the swapchain, in preference order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdrColorSpace {
    /// HDR10 PQ (SMPTE ST 2084), BT.2020 primaries. The swapchain carries
    /// PQ-encoded values (the decoder's passthrough output presents directly).
    Hdr10Pq,
    /// scRGB: extended-range linear, BT.709 primaries (values may exceed 1.0 for
    /// highlights). The wgpu-native HDR path; needs a linear-light input.
    ScRgbLinear,
    /// SDR sRGB (nonlinear). The fallback when the surface offers no HDR space.
    Sdr,
}

impl HdrColorSpace {
    fn from_vk(cs: vk::ColorSpaceKHR) -> Option<Self> {
        match cs {
            vk::ColorSpaceKHR::HDR10_ST2084_EXT => Some(HdrColorSpace::Hdr10Pq),
            vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT => Some(HdrColorSpace::ScRgbLinear),
            vk::ColorSpaceKHR::SRGB_NONLINEAR => Some(HdrColorSpace::Sdr),
            _ => None,
        }
    }

    /// Preference rank (higher = more preferred): HDR10 PQ > scRGB > SDR.
    fn rank(self) -> u8 {
        match self {
            HdrColorSpace::Hdr10Pq => 2,
            HdrColorSpace::ScRgbLinear => 1,
            HdrColorSpace::Sdr => 0,
        }
    }
}

/// Mastering-display metadata for the HDR10 (`VkHdrMetadataEXT`) present. Streams
/// here carry no mastering metadata, so this defaults to BT.2020 primaries + D65
/// white and a 1000-nit peak (typical HDR10). Luminances are in nits (cd/m^2).
#[derive(Debug, Clone, Copy)]
pub struct HdrMasteringDisplay {
    pub max_luminance: f32,
    pub min_luminance: f32,
    pub max_content_light_level: f32,
    pub max_frame_average_light_level: f32,
}

impl Default for HdrMasteringDisplay {
    fn default() -> Self {
        Self {
            max_luminance: 1000.0,
            min_luminance: 0.005,
            max_content_light_level: 1000.0,
            max_frame_average_light_level: 400.0,
        }
    }
}

/// Pick the best (surface format, colour space) from what the surface offers,
/// preferring HDR10 PQ, then scRGB linear, then SDR. Within a colour space, a
/// 10-bit (`A2B10G10R10` / `A2R10G10B10`) or 16-bit-float format is preferred over
/// 8-bit. Returns `None` if the surface offers no format this sink recognises.
/// Pure (no Vulkan calls); unit-tested against synthetic surface-format lists.
fn select_format(
    formats: &[vk::SurfaceFormatKHR],
) -> Option<(vk::SurfaceFormatKHR, HdrColorSpace)> {
    // Score a format: colour-space rank dominates, then bit depth.
    let depth_score = |f: vk::Format| -> u8 {
        match f {
            vk::Format::R16G16B16A16_SFLOAT => 3,
            vk::Format::A2B10G10R10_UNORM_PACK32 | vk::Format::A2R10G10B10_UNORM_PACK32 => 2,
            vk::Format::R8G8B8A8_UNORM
            | vk::Format::B8G8R8A8_UNORM
            | vk::Format::R8G8B8A8_SRGB
            | vk::Format::B8G8R8A8_SRGB => 1,
            _ => 0,
        }
    };
    formats
        .iter()
        .filter_map(|sf| {
            let cs = HdrColorSpace::from_vk(sf.color_space)?;
            let d = depth_score(sf.format);
            if d == 0 {
                return None;
            }
            // HDR colour spaces need a >8-bit format to be worth taking (an 8-bit
            // HDR10 surface would band badly); SDR takes any recognised format.
            if cs != HdrColorSpace::Sdr && d < 2 {
                return None;
            }
            Some(((*sf, cs), (cs.rank(), d)))
        })
        .max_by_key(|&(_, key)| key)
        .map(|(v, _)| v)
}

/// A CIE xy chromaticity as the `VkXYColorEXT` the metadata wants.
fn xy(x: f32, y: f32) -> vk::XYColorEXT {
    vk::XYColorEXT { x, y }
}

/// Build `VkHdrMetadataEXT` for BT.2020 primaries + D65 white from a mastering
/// display. Pure; the primaries are the fixed BT.2020 (Rec.2100) chromaticities.
fn hdr10_metadata(m: HdrMasteringDisplay) -> vk::HdrMetadataEXT<'static> {
    vk::HdrMetadataEXT::default()
        .display_primary_red(xy(0.708, 0.292))
        .display_primary_green(xy(0.170, 0.797))
        .display_primary_blue(xy(0.131, 0.046))
        .white_point(xy(0.3127, 0.3290))
        .max_luminance(m.max_luminance)
        .min_luminance(m.min_luminance)
        .max_content_light_level(m.max_content_light_level)
        .max_frame_average_light_level(m.max_frame_average_light_level)
}

/// An on-screen HDR present sink: a raw Vulkan swapchain on the decode device
/// that presents decoded HDR textures with an HDR colour space + metadata. Owns
/// the surface, swapchain, and per-present command buffer / fences; frees them on
/// drop. The application owns the window (and its lifetime must outlive the sink).
pub struct VulkanHdrSink {
    ctx: PresentContext,
    surface_fn: ash::khr::surface::Instance,
    swapchain_fn: ash::khr::swapchain::Device,
    hdr_fn: Option<ash::ext::hdr_metadata::Device>,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    images: alloc::vec::Vec<vk::Image>,
    surface_format: vk::SurfaceFormatKHR,
    color_space: HdrColorSpace,
    extent: vk::Extent2D,
    cmd_pool: vk::CommandPool,
    cmd_buf: vk::CommandBuffer,
    /// Signalled by `acquire_next_image` when the acquired image is ready; the
    /// blit submit waits it (GPU-side, no CPU stall). One frame in flight, so a
    /// single binary semaphore suffices (it is consumed by the submit each frame,
    /// which `in_flight` confirms complete before the next acquire re-signals it).
    image_available: vk::Semaphore,
    /// Signalled by the blit submit, waited by present, one per swapchain image:
    /// an image is only re-acquired once its present consumed this, so re-signalling
    /// `render_finished[i]` is always safe (the textbook per-image present sync).
    render_finished: alloc::vec::Vec<vk::Semaphore>,
    /// Signalled by the blit submit; waited at the top of the next `present` so the
    /// command buffer + `image_available` are free to reuse. Created signalled so
    /// the first frame's wait passes. This is the only CPU wait, and it overlaps
    /// the caller's between-frame work (decode / pacing) rather than stalling
    /// mid-present as the old acquire-fence + submit-fence pair did.
    in_flight: vk::Fence,
    mastering: HdrMasteringDisplay,
    presented: u64,
}

impl core::fmt::Debug for VulkanHdrSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VulkanHdrSink")
            .field("color_space", &self.color_space)
            .field("format", &self.surface_format.format)
            .field("extent", &self.extent)
            .field("images", &self.images.len())
            .field("presented", &self.presented)
            .finish_non_exhaustive()
    }
}

impl VulkanHdrSink {
    /// Build an HDR present sink for `device` presenting to the window described
    /// by `display` + `window` (from the app's windowing library, e.g. winit's
    /// `raw-window-handle`). `width` x `height` is the initial drawable size.
    /// Returns [`VulkanVideoError::PresentUnsupported`] if the device cannot
    /// present or the platform handle is unsupported.
    ///
    /// # Safety
    /// `display` / `window` must be valid handles to a live window that outlives
    /// the returned sink, on the platform the decode device's instance enabled the
    /// matching surface extension for (which wgpu-hal does for the running WSI).
    pub unsafe fn new(
        device: &VulkanVideoDevice,
        display: RawDisplayHandle,
        window: RawWindowHandle,
        width: u32,
        height: u32,
        mastering: HdrMasteringDisplay,
    ) -> Result<Self, VulkanVideoError> {
        let ctx = device
            .present_context()
            .ok_or(VulkanVideoError::PresentUnsupported)?;
        let surface_fn = ash::khr::surface::Instance::new(&ctx.entry, &ctx.instance);
        let swapchain_fn = ash::khr::swapchain::Device::new(&ctx.instance, &ctx.device);
        let hdr_fn = ctx
            .hdr_metadata
            .then(|| ash::ext::hdr_metadata::Device::new(&ctx.instance, &ctx.device));

        // SAFETY: handles valid per the contract; every raw object created here is
        // stored on the sink and destroyed once in `Drop` (surface last).
        let surface = unsafe { create_surface(&ctx.entry, &ctx.instance, display, window)? };

        let cleanup_surface = |s: vk::SurfaceKHR| {
            // SAFETY: `s` was just created from `surface_fn`'s instance.
            unsafe { surface_fn.destroy_surface(s, None) };
        };

        // The present queue family must support this surface.
        // SAFETY: valid surface + physical device + family.
        let supported = unsafe {
            surface_fn.get_physical_device_surface_support(ctx.phys, ctx.queue_family, surface)
        }
        .unwrap_or(false);
        if !supported {
            cleanup_surface(surface);
            return Err(VulkanVideoError::PresentUnsupported);
        }

        let cmd_pool = {
            let ci = vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.queue_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            // SAFETY: valid create info.
            match unsafe { ctx.device.create_command_pool(&ci, None) } {
                Ok(p) => p,
                Err(e) => {
                    cleanup_surface(surface);
                    return Err(VulkanVideoError::QueryFailed(e));
                }
            }
        };
        // SAFETY: pool just created; one primary buffer.
        let cmd_buf = match unsafe {
            ctx.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        } {
            Ok(v) => v[0],
            Err(e) => {
                // SAFETY: pool + surface created above.
                unsafe { ctx.device.destroy_command_pool(cmd_pool, None) };
                cleanup_surface(surface);
                return Err(VulkanVideoError::QueryFailed(e));
            }
        };
        // The image-available semaphore and the in-flight fence (created signalled
        // so the first present's wait-at-top passes). Per-image `render_finished`
        // semaphores are built with the swapchain. On failure free the pool +
        // surface + whatever succeeded.
        let free_pool_surface = |sem: Option<vk::Semaphore>, fence: Option<vk::Fence>| {
            // SAFETY: destroy the handles created above (if any) once each.
            unsafe {
                if let Some(s) = sem {
                    ctx.device.destroy_semaphore(s, None);
                }
                if let Some(f) = fence {
                    ctx.device.destroy_fence(f, None);
                }
                ctx.device.destroy_command_pool(cmd_pool, None);
            }
            cleanup_surface(surface);
        };
        // SAFETY: valid create info.
        let image_available = match unsafe {
            ctx.device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
        } {
            Ok(s) => s,
            Err(e) => {
                free_pool_surface(None, None);
                return Err(VulkanVideoError::QueryFailed(e));
            }
        };
        // SAFETY: valid create info; SIGNALED so the first wait-at-top passes.
        let in_flight = match unsafe {
            ctx.device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
        } {
            Ok(f) => f,
            Err(e) => {
                free_pool_surface(Some(image_available), None);
                return Err(VulkanVideoError::QueryFailed(e));
            }
        };

        let mut sink = VulkanHdrSink {
            ctx,
            surface_fn,
            swapchain_fn,
            hdr_fn,
            surface,
            swapchain: vk::SwapchainKHR::null(),
            images: alloc::vec::Vec::new(),
            surface_format: vk::SurfaceFormatKHR::default(),
            color_space: HdrColorSpace::Sdr,
            extent: vk::Extent2D { width, height },
            cmd_pool,
            cmd_buf,
            image_available,
            render_finished: alloc::vec::Vec::new(),
            in_flight,
            mastering,
            presented: 0,
        };
        // SAFETY: the sink's handles are all valid; on failure `Drop` cleans up.
        unsafe { sink.build_swapchain(width, height) }?;
        Ok(sink)
    }

    /// The colour space the swapchain was created with (what the surface offered).
    pub fn color_space(&self) -> HdrColorSpace {
        self.color_space
    }

    /// The swapchain image format.
    pub fn format(&self) -> vk::Format {
        self.surface_format.format
    }

    /// Frames presented so far.
    pub fn presented_count(&self) -> u64 {
        self.presented
    }

    /// (Re)build the swapchain at `width` x `height`, selecting the best HDR
    /// colour space the surface offers and attaching HDR metadata when available.
    /// Called by `new` and `resize`.
    ///
    /// # Safety
    /// The sink's surface / device handles must be valid and no present may be in
    /// flight (`present` is CPU-synchronised, so this holds between frames).
    unsafe fn build_swapchain(&mut self, width: u32, height: u32) -> Result<(), VulkanVideoError> {
        let dev = &self.ctx.device;
        // SAFETY: valid surface + physical device.
        let caps = unsafe {
            self.surface_fn
                .get_physical_device_surface_capabilities(self.ctx.phys, self.surface)
        }
        .map_err(VulkanVideoError::QueryFailed)?;
        // SAFETY: valid surface + physical device.
        let formats = unsafe {
            self.surface_fn
                .get_physical_device_surface_formats(self.ctx.phys, self.surface)
        }
        .map_err(VulkanVideoError::QueryFailed)?;
        let (surface_format, color_space) =
            select_format(&formats).ok_or(VulkanVideoError::PresentUnsupported)?;

        // Clamp the requested extent to the surface's allowed range; a
        // `current_extent` of u32::MAX means the surface takes whatever we pick.
        let extent = if caps.current_extent.width == u32::MAX {
            vk::Extent2D {
                width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        } else {
            caps.current_extent
        };
        if extent.width == 0 || extent.height == 0 {
            // Minimised window: keep the old swapchain, skip the rebuild.
            return Ok(());
        }
        let min_images = (caps.min_image_count + 1).min(if caps.max_image_count == 0 {
            u32::MAX
        } else {
            caps.max_image_count
        });
        // FIFO is always supported (vsync); the blit dst needs TRANSFER_DST.
        let usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT;
        if !caps
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_DST)
        {
            return Err(VulkanVideoError::PresentUnsupported);
        }
        let ci = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(min_images)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(self.swapchain);

        // SAFETY: valid create info; the old swapchain (if any) is retired below.
        let new_swapchain = unsafe { self.swapchain_fn.create_swapchain(&ci, None) }
            .map_err(VulkanVideoError::QueryFailed)?;
        // Retire the previous swapchain now that the new one is created.
        if self.swapchain != vk::SwapchainKHR::null() {
            // SAFETY: no present in flight (CPU-synchronised); images not in use.
            unsafe { self.swapchain_fn.destroy_swapchain(self.swapchain, None) };
        }
        self.swapchain = new_swapchain;
        // SAFETY: fresh swapchain.
        self.images = unsafe { self.swapchain_fn.get_swapchain_images(self.swapchain) }
            .map_err(VulkanVideoError::QueryFailed)?;
        self.surface_format = surface_format;
        self.color_space = color_space;
        self.extent = extent;

        // (Re)create one `render_finished` semaphore per swapchain image. Old ones
        // are freed after `device_wait_idle` (resize) or at first build (empty).
        for s in self.render_finished.drain(..) {
            // SAFETY: no present in flight; each old semaphore destroyed once.
            unsafe { dev.destroy_semaphore(s, None) };
        }
        for _ in 0..self.images.len() {
            // SAFETY: valid create info.
            match unsafe { dev.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) } {
                Ok(s) => self.render_finished.push(s),
                Err(e) => return Err(VulkanVideoError::QueryFailed(e)),
            }
        }

        // Attach HDR mastering metadata when the colour space is HDR and the
        // device has VK_EXT_hdr_metadata (best-effort; not fatal if absent).
        if color_space != HdrColorSpace::Sdr {
            if let Some(hdr) = &self.hdr_fn {
                let md = hdr10_metadata(self.mastering);
                // SAFETY: one swapchain + one metadata, both live.
                unsafe { hdr.set_hdr_metadata(&[self.swapchain], &[md]) };
            }
        }
        Ok(())
    }

    /// Rebuild the swapchain for a new drawable size (window resize). Safe to call
    /// between presents.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), VulkanVideoError> {
        // SAFETY: present is CPU-synchronised, so nothing is in flight here.
        unsafe {
            self.ctx
                .device
                .device_wait_idle()
                .map_err(VulkanVideoError::QueryFailed)?;
            self.build_swapchain(width, height)
        }
    }

    /// Present `texture` (a decoded HDR/SDR `wgpu::Texture` on the decode device)
    /// to the swapchain: acquire an image, blit the texture onto it (scaling to
    /// the swapchain extent), and present. The acquire -> blit -> present chain is
    /// ordered by GPU semaphores (`image_available`, per-image `render_finished`);
    /// the only CPU wait is the previous frame's `in_flight` fence at the top,
    /// which overlaps the caller's between-frame work. An out-of-date swapchain
    /// (resize) is skipped; call [`resize`](Self::resize) then retry.
    ///
    /// # Safety
    /// `texture` must be a live colour texture on this sink's device left in
    /// `SHADER_READ_ONLY_OPTIMAL` layout (the decoder's GPU-texture output is), and
    /// no other queue work may touch the decode device concurrently (the pipeline
    /// is cooperative and the decode already completed before this call).
    pub unsafe fn present(&mut self, texture: &wgpu::Texture) -> Result<(), VulkanVideoError> {
        if self.swapchain == vk::SwapchainKHR::null() || self.images.is_empty() {
            return Ok(());
        }
        // The decoded image handle (raw VkImage behind the wgpu texture).
        // SAFETY: caller guarantees a live Vulkan texture on this device.
        let src_image = match unsafe { texture.as_hal::<wgpu_hal::api::Vulkan>() } {
            // SAFETY: the hal texture is not destroyed; we only read its VkImage.
            Some(t) => unsafe { t.raw_handle() },
            None => return Err(VulkanVideoError::NoVulkanAdapter),
        };
        let src_w = texture.width();
        let src_h = texture.height();
        let dev = &self.ctx.device;

        // Wait the previous frame's blit (so the command buffer + `image_available`
        // are free to reuse), but do NOT reset the fence yet: if the acquire below
        // fails (out-of-date), we return with the fence still signalled so the next
        // frame's wait still passes (resetting-then-not-submitting would deadlock).
        // SAFETY: `in_flight` is created signalled and signalled by each submit.
        unsafe { dev.wait_for_fences(&[self.in_flight], true, u64::MAX) }
            .map_err(VulkanVideoError::QueryFailed)?;

        // Acquire the next image, signalling `image_available` (GPU-side; the blit
        // submit waits it, so there is no CPU stall between acquire and blit).
        // SAFETY: valid swapchain + semaphore.
        let (index, _sub) = match unsafe {
            self.swapchain_fn.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available,
                vk::Fence::null(),
            )
        } {
            Ok(v) => v,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(()), // caller resizes
            Err(e) => return Err(VulkanVideoError::QueryFailed(e)),
        };
        // The image is ours now; reset the fence before submitting.
        // SAFETY: waited above; reset once before the submit re-signals it.
        unsafe { dev.reset_fences(&[self.in_flight]) }.map_err(VulkanVideoError::QueryFailed)?;

        let dst_image = self.images[index as usize];
        // SAFETY: record the blit (source SHADER_READ_ONLY per the contract, dest
        // swapchain image discarded then filled) with explicit layout barriers.
        unsafe { self.record_blit(src_image, dst_image, src_w, src_h)? };

        // Submit: wait `image_available` at the transfer stage (where the blit uses
        // the image), signal `render_finished[index]` (present waits it) and
        // `in_flight` (next frame waits it). No CPU wait here.
        let cbs = [self.cmd_buf];
        let wait = [self.image_available];
        let wait_stage = [vk::PipelineStageFlags::TRANSFER];
        let signal = [self.render_finished[index as usize]];
        let submit = vk::SubmitInfo::default()
            .command_buffers(&cbs)
            .wait_semaphores(&wait)
            .wait_dst_stage_mask(&wait_stage)
            .signal_semaphores(&signal);
        // SAFETY: the command buffer is recorded; semaphores + fence are live.
        unsafe { dev.queue_submit(self.ctx.queue, &[submit], self.in_flight) }
            .map_err(VulkanVideoError::QueryFailed)?;

        // Present, waiting `render_finished[index]` (the present engine waits the
        // blit on the GPU side). One `render_finished` per image makes re-signalling
        // it next time this image is acquired safe (present consumed it by then).
        let swapchains = [self.swapchain];
        let indices = [index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal)
            .swapchains(&swapchains)
            .image_indices(&indices);
        // SAFETY: valid present info; queue supports present to this surface.
        match unsafe {
            self.swapchain_fn
                .queue_present(self.ctx.queue, &present_info)
        } {
            Ok(_) => {
                self.presented += 1;
                Ok(())
            }
            // Out-of-date / suboptimal: the caller resizes and retries.
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => Ok(()),
            Err(e) => Err(VulkanVideoError::QueryFailed(e)),
        }
    }

    /// Record the source->swapchain blit into `cmd_buf` with explicit layout
    /// transitions. Source: `SHADER_READ_ONLY_OPTIMAL` -> `TRANSFER_SRC` -> restore.
    /// Dest (swapchain): `UNDEFINED` -> `TRANSFER_DST` -> `PRESENT_SRC_KHR`.
    ///
    /// # Safety
    /// `cmd_buf` is recordable; `src` is in `SHADER_READ_ONLY_OPTIMAL`; `dst` is a
    /// swapchain image owned by this sink.
    unsafe fn record_blit(
        &self,
        src: vk::Image,
        dst: vk::Image,
        src_w: u32,
        src_h: u32,
    ) -> Result<(), VulkanVideoError> {
        let dev = &self.ctx.device;
        let cb = self.cmd_buf;
        let color = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let barrier = |image, old, new, src_access, dst_access| {
            vk::ImageMemoryBarrier::default()
                .old_layout(old)
                .new_layout(new)
                .src_access_mask(src_access)
                .dst_access_mask(dst_access)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color)
        };
        // SAFETY: contract above; barriers + a single blit, then transitions to
        // PRESENT_SRC (dest) and back to SHADER_READ_ONLY (source).
        unsafe {
            dev.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                .map_err(VulkanVideoError::QueryFailed)?;
            dev.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(VulkanVideoError::QueryFailed)?;
            let pre = [
                barrier(
                    src,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::AccessFlags::SHADER_READ,
                    vk::AccessFlags::TRANSFER_READ,
                ),
                barrier(
                    dst,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                ),
            ];
            dev.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &pre,
            );
            let sub = vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1);
            let region = vk::ImageBlit::default()
                .src_subresource(sub)
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: src_w as i32,
                        y: src_h as i32,
                        z: 1,
                    },
                ])
                .dst_subresource(sub)
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: self.extent.width as i32,
                        y: self.extent.height as i32,
                        z: 1,
                    },
                ]);
            dev.cmd_blit_image(
                cb,
                src,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
                vk::Filter::LINEAR,
            );
            let post = [
                barrier(
                    dst,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::empty(),
                ),
                barrier(
                    src,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::AccessFlags::TRANSFER_READ,
                    vk::AccessFlags::SHADER_READ,
                ),
            ];
            dev.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &post,
            );
            dev.end_command_buffer(cb)
                .map_err(VulkanVideoError::QueryFailed)?;
        }
        Ok(())
    }
}

impl Drop for VulkanHdrSink {
    fn drop(&mut self) {
        let dev = &self.ctx.device;
        // SAFETY: every handle was created by this sink; wait idle first so nothing
        // is in flight, then destroy each once (swapchain + surface last).
        unsafe {
            let _ = dev.device_wait_idle();
            dev.destroy_fence(self.in_flight, None);
            dev.destroy_semaphore(self.image_available, None);
            for &s in &self.render_finished {
                dev.destroy_semaphore(s, None);
            }
            dev.destroy_command_pool(self.cmd_pool, None);
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_fn.destroy_swapchain(self.swapchain, None);
            }
            self.surface_fn.destroy_surface(self.surface, None);
        }
    }
}

/// Create a `VkSurfaceKHR` from raw window/display handles via the platform WSI
/// extension (which wgpu-hal enabled on its instance for the running window
/// system). Supports Wayland / Xlib / Xcb on Linux and Win32 on Windows.
///
/// # Safety
/// The handles must be valid and outlive the returned surface.
unsafe fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    display: RawDisplayHandle,
    window: RawWindowHandle,
) -> Result<vk::SurfaceKHR, VulkanVideoError> {
    let err = VulkanVideoError::QueryFailed;
    match (display, window) {
        #[cfg(target_os = "linux")]
        (RawDisplayHandle::Wayland(d), RawWindowHandle::Wayland(w)) => {
            let sfn = ash::khr::wayland_surface::Instance::new(entry, instance);
            let ci = vk::WaylandSurfaceCreateInfoKHR::default()
                .display(d.display.as_ptr())
                .surface(w.surface.as_ptr());
            // SAFETY: valid Wayland display + surface pointers.
            unsafe { sfn.create_wayland_surface(&ci, None) }.map_err(err)
        }
        #[cfg(target_os = "linux")]
        (RawDisplayHandle::Xlib(d), RawWindowHandle::Xlib(w)) => {
            let sfn = ash::khr::xlib_surface::Instance::new(entry, instance);
            let ci = vk::XlibSurfaceCreateInfoKHR::default()
                .dpy(
                    d.display
                        .map_or(core::ptr::null_mut(), |p| p.as_ptr().cast()),
                )
                .window(w.window);
            // SAFETY: valid Xlib display + window.
            unsafe { sfn.create_xlib_surface(&ci, None) }.map_err(err)
        }
        #[cfg(target_os = "linux")]
        (RawDisplayHandle::Xcb(d), RawWindowHandle::Xcb(w)) => {
            let sfn = ash::khr::xcb_surface::Instance::new(entry, instance);
            let ci = vk::XcbSurfaceCreateInfoKHR::default()
                .connection(d.connection.map_or(core::ptr::null_mut(), |p| p.as_ptr()))
                .window(w.window.get());
            // SAFETY: valid Xcb connection + window.
            unsafe { sfn.create_xcb_surface(&ci, None) }.map_err(err)
        }
        #[cfg(target_os = "windows")]
        (RawDisplayHandle::Windows(_), RawWindowHandle::Win32(w)) => {
            let sfn = ash::khr::win32_surface::Instance::new(entry, instance);
            let ci = vk::Win32SurfaceCreateInfoKHR::default()
                .hinstance(w.hinstance.map_or(0, |h| h.get()))
                .hwnd(w.hwnd.get());
            // SAFETY: valid Win32 HWND / HINSTANCE.
            unsafe { sfn.create_win32_surface(&ci, None) }.map_err(err)
        }
        _ => Err(VulkanVideoError::PresentUnsupported),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sf(format: vk::Format, cs: vk::ColorSpaceKHR) -> vk::SurfaceFormatKHR {
        vk::SurfaceFormatKHR {
            format,
            color_space: cs,
        }
    }

    #[test]
    fn prefers_hdr10_pq_10bit() {
        // A surface offering SDR 8-bit, scRGB 16f, and HDR10 10-bit -> pick HDR10.
        let formats = [
            sf(
                vk::Format::B8G8R8A8_UNORM,
                vk::ColorSpaceKHR::SRGB_NONLINEAR,
            ),
            sf(
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT,
            ),
            sf(
                vk::Format::A2B10G10R10_UNORM_PACK32,
                vk::ColorSpaceKHR::HDR10_ST2084_EXT,
            ),
        ];
        let (chosen, cs) = select_format(&formats).expect("a format");
        assert_eq!(cs, HdrColorSpace::Hdr10Pq);
        assert_eq!(chosen.format, vk::Format::A2B10G10R10_UNORM_PACK32);
    }

    #[test]
    fn falls_back_to_scrgb_then_sdr() {
        // No HDR10 -> scRGB linear.
        let scrgb = [
            sf(
                vk::Format::B8G8R8A8_UNORM,
                vk::ColorSpaceKHR::SRGB_NONLINEAR,
            ),
            sf(
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT,
            ),
        ];
        assert_eq!(select_format(&scrgb).unwrap().1, HdrColorSpace::ScRgbLinear);
        // Only SDR -> SDR.
        let sdr = [sf(
            vk::Format::B8G8R8A8_SRGB,
            vk::ColorSpaceKHR::SRGB_NONLINEAR,
        )];
        assert_eq!(select_format(&sdr).unwrap().1, HdrColorSpace::Sdr);
        // An 8-bit HDR10 surface is rejected (would band); nothing recognised.
        let bad = [sf(
            vk::Format::B8G8R8A8_UNORM,
            vk::ColorSpaceKHR::HDR10_ST2084_EXT,
        )];
        assert!(select_format(&bad).is_none());
        // Empty / unrecognised -> None.
        assert!(select_format(&[]).is_none());
    }

    #[test]
    fn hdr10_metadata_uses_bt2020_primaries() {
        let md = hdr10_metadata(HdrMasteringDisplay {
            max_luminance: 1200.0,
            ..Default::default()
        });
        // BT.2020 red primary + D65 white + the peak we passed.
        assert!((md.display_primary_red.x - 0.708).abs() < 1e-6);
        assert!((md.display_primary_green.y - 0.797).abs() < 1e-6);
        assert!((md.white_point.x - 0.3127).abs() < 1e-6);
        assert!((md.max_luminance - 1200.0).abs() < 1e-3);
    }
}
