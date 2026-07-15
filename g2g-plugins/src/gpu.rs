//! Shared wgpu device context for the GPU elements (M103): the Vello overlay
//! producer ([`crate::vellooverlay`]) and the [`WgpuSink`](crate::wgpusink)
//! consumer.
//!
//! A `wgpu::Texture` is bound to the device that created it, so a producer and a
//! sink can only hand a [`MemoryDomain::WgpuTexture`](g2g_core::MemoryDomain)
//! frame across with no copy if they share **one** device. [`GpuContext`] is that
//! shared handle: build it once, clone it into both elements. `wgpu::Device` /
//! `Queue` / `Adapter` / `Instance` are all cheap `Clone`s (reference-counted
//! inside wgpu), so cloning a `GpuContext` shares the GPU rather than duplicating
//! it.
//!
//! Gated on the GPU features (`vello-overlay` / `wgpu-sink` / `cuda-wgpu`); the
//! last reuses [`texture_of`] / [`WgpuTextureKeepAlive`] to read an upstream
//! producer's texture in the wgpu -> CUDA bridge element.

use g2g_core::memory::OwnedWgpuTexture;
use g2g_core::{G2gError, HardwareError, WgpuKeepAlive};

/// A shared wgpu device context. Clone it into each GPU element so they render
/// and present on the same device (the prerequisite for a copy-free
/// `WgpuTexture` handoff between a producer and a sink).
#[derive(Debug, Clone)]
pub struct GpuContext {
    /// The wgpu instance (used by an application to create a surface).
    pub instance: wgpu::Instance,
    /// The adapter the device was opened on (used for surface capabilities).
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Build a headless context (no surface): for the overlay, the offscreen
    /// sink, and tests. Picks a high-performance adapter.
    pub async fn headless() -> Result<Self, G2gError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(gpu_err)?;
        Self::from_adapter(instance, adapter).await
    }

    /// Build a context whose adapter can present to `surface`. Use this for the
    /// on-screen sink: an application creates the window's `wgpu::Surface` from
    /// `instance`, then opens a compatible device here.
    pub async fn for_surface(
        instance: wgpu::Instance,
        surface: &wgpu::Surface<'_>,
    ) -> Result<Self, G2gError> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(surface),
                ..Default::default()
            })
            .await
            .map_err(gpu_err)?;
        Self::from_adapter(instance, adapter).await
    }

    /// Wrap an application's *existing* wgpu device as the shared context, rather
    /// than opening one (M263). For an embedder that already owns a `wgpu::Device`
    /// (a game engine, a Bevy / Tauri app, an editor's renderer): hand its
    /// device / queue / adapter / instance here and every GPU element produces
    /// textures *on that device*, so a decoded frame's
    /// [`MemoryDomain::WgpuTexture`](g2g_core::MemoryDomain) is bindable directly
    /// in the embedder's own render graph with no extra device, no surface
    /// hand-off, and no copy. `wgpu` handles are reference-counted, so cloning the
    /// engine's device in (and keeping a clone to render with) shares one GPU. The
    /// inverse of [`for_surface`](Self::for_surface), where g2g opens the device:
    /// here the embedder owns it and g2g joins.
    pub fn from_wgpu(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Self {
        Self { instance, adapter, device, queue }
    }

    async fn from_adapter(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
    ) -> Result<Self, G2gError> {
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("g2g-gpu"),
                // The full pipeline fits within the default limits on a discrete
                // GPU; pass the adapter's so a constrained default tier does not
                // reject Vello's renderer construction.
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .map_err(gpu_err)?;
        Ok(Self { instance, adapter, device, queue })
    }
}

/// Keep-alive owner for a rendered wgpu texture (the [`WgpuKeepAlive`] payload of
/// [`MemoryDomain::WgpuTexture`](g2g_core::MemoryDomain)). Owns the
/// `wgpu::Texture`; the consuming sink recovers it via [`texture_of`]. Shared so
/// the overlay producer and the sink agree on the concrete type to downcast to.
#[derive(Debug)]
pub struct WgpuTextureKeepAlive(pub wgpu::Texture);

impl WgpuKeepAlive for WgpuTextureKeepAlive {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

/// Recover the `wgpu::Texture` from a [`OwnedWgpuTexture`], whatever g2g GPU
/// producer wrapped it. Returns `None` if the frame's keep-alive is some other
/// (foreign) producer's type this sink cannot present.
///
/// Two in-tree producers wrap their texture differently: the overlay / blit path
/// uses [`WgpuTextureKeepAlive`], and the Android MediaCodec GPU decode (M304/M305)
/// uses [`mediacodec_wgpu::WgpuRgbaTexture`](crate::mediacodec_wgpu::WgpuRgbaTexture).
/// Recognising both lets the one [`WgpuSink`](crate::wgpusink) present either.
pub fn texture_of(owned: &OwnedWgpuTexture) -> Option<&wgpu::Texture> {
    let any = owned.keep_alive().as_any();
    if let Some(k) = any.downcast_ref::<WgpuTextureKeepAlive>() {
        return Some(&k.0);
    }
    // The Android decoder's RGBA output: same device, different wrapper.
    #[cfg(all(target_os = "android", feature = "mediacodec-wgpu"))]
    if let Some(k) = any.downcast_ref::<crate::mediacodec_wgpu::WgpuRgbaTexture>() {
        return Some(k.texture());
    }
    None
}

/// Read an RGBA8 `wgpu::Texture` back to a packed `Vec<u8>` (`width * height * 4`)
/// on the CPU, using the device/queue of a shared [`GpuContext`]. The texture must
/// be bound to that context's device (e.g. a zero-copy decode texture whose context
/// the caller shares). Handles the 256-byte `bytes_per_row` alignment wgpu requires
/// for buffer copies, unpacking back to a tight stride.
///
/// This is a verification / interop helper for GPU-resident frames (the Tier B
/// zero-copy path): a real consumer keeps the texture on the GPU, but a debugger or
/// a cross-device fallback needs the pixels. Not part of the zero-copy fast path.
pub fn read_rgba_texture(ctx: &GpuContext, texture: &wgpu::Texture) -> alloc::vec::Vec<u8> {
    read_rgba_texture_dq(&ctx.device, &ctx.queue, texture).expect("rgba texture readback")
}

/// The core of [`read_rgba_texture`] over an explicit device + queue (rather than a
/// [`GpuContext`]), so callers that hold their own wgpu device/queue (the
/// Vulkan-Video and Android MediaCodec decode paths) share the one copy of the
/// 256-byte-row alignment + de-pad readback. `texture` must be an uncompressed
/// colour format (its texel block size gives the row stride: 4 for `Rgba8Unorm`,
/// 8 for the 10-bit `Rgba16Float` target) and bound to `device`. The returned
/// bytes are tightly packed at that texel size. Fails only if the device is lost
/// during the readback poll.
pub(crate) fn read_rgba_texture_dq(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
) -> Result<alloc::vec::Vec<u8>, G2gError> {
    let w = texture.width();
    let h = texture.height();
    let bpp = texture.format().block_copy_size(None).unwrap_or(4) as usize;
    let tight = w as usize * bpp;
    // wgpu requires bytes_per_row be a multiple of 256 for texture->buffer copies.
    let padded = tight.div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read_rgba_texture"),
        size: (padded * h as usize) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded as u32),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit([enc.finish()]);
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
        .map_err(gpu_err)?;
    rx.recv().map_err(gpu_err)?.map_err(gpu_err)?;
    let mapped = slice.get_mapped_range();
    // Drop the per-row padding back to a tight w*4 stride.
    let mut out = alloc::vec::Vec::with_capacity(tight * h as usize);
    for row in 0..h as usize {
        let start = row * padded;
        out.extend_from_slice(&mapped[start..start + tight]);
    }
    drop(mapped);
    buffer.unmap();
    Ok(out)
}

/// Map any wgpu / Vello failure to a structured hardware error.
pub(crate) fn gpu_err<E>(_e: E) -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

/// Import an externally-owned Vulkan image as a zero-copy `wgpu::Texture` on
/// `device` (the interop Vulkan wgpu device the image was allocated on, which for
/// a zero-copy import is necessarily the device wgpu wraps). wgpu does not own the
/// backing memory (`TextureMemory::External`); `image` and `memory` are freed by
/// the texture's drop callback when wgpu drops the texture. `hal_usage` /
/// `wgpu_usage` are the matching HAL / wgpu usage masks (they differ per caller:
/// a sampled NV12 plane vs a render-target RGBA surface). Returns `None` if
/// `device` is not a Vulkan wgpu device. Shared by the CUDA-interop (`cudawgpu`)
/// and Vulkan-Video (`vulkanvideo`) egress paths.
///
/// # Safety
/// `device` must be the Vulkan wgpu device `image` / `memory` were allocated on,
/// and this takes ownership of them: they must not be freed by any other path.
#[cfg(any(feature = "cuda-wgpu", feature = "vulkan-video", feature = "mediacodec-wgpu"))]
// A thin wrapper over the raw Vulkan image + wgpu-hal descriptors; each argument
// is a distinct piece of the import (image, memory, geometry, format, the two
// usage masks, label) with no natural grouping.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn import_vk_image_as_wgpu_texture(
    device: &wgpu::Device,
    image: ash::vk::Image,
    memory: ash::vk::DeviceMemory,
    size: wgpu::Extent3d,
    format: wgpu::TextureFormat,
    hal_usage: wgpu::TextureUses,
    wgpu_usage: wgpu::TextureUsages,
    label: &str,
) -> Option<wgpu::Texture> {
    // SAFETY: the caller's contract guarantees `device` owns `image`/`memory` and
    // transfers ownership here; the drop callback frees them once, when wgpu drops
    // the texture (the GPU idle by then).
    unsafe {
        let hal_device = device.as_hal::<wgpu_hal::api::Vulkan>()?;
        let raw = hal_device.raw_device().clone();
        let drop_cb: wgpu_hal::DropCallback = alloc::boxed::Box::new(move || {
            raw.destroy_image(image, None);
            raw.free_memory(memory, None);
        });
        let hal_desc = wgpu_hal::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: hal_usage,
            memory_flags: wgpu_hal::MemoryFlags::empty(),
            view_formats: alloc::vec::Vec::new(),
        };
        let hal_tex = hal_device.texture_from_raw(
            image,
            &hal_desc,
            Some(drop_cb),
            wgpu_hal::vulkan::TextureMemory::External,
        );
        let wgpu_desc = wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu_usage,
            view_formats: &[],
        };
        Some(device.create_texture_from_hal::<wgpu_hal::api::Vulkan>(hal_tex, &wgpu_desc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use g2g_core::frame::{Frame, FrameTiming};
    use g2g_core::memory::MemoryDomain;

    // 64 px * 4 bytes = 256, the wgpu COPY_BYTES_PER_ROW alignment, so the
    // read-back needs no row padding.
    const W: u32 = 64;
    const H: u32 = 2;

    /// Stand in for an embedding application (a game engine / Tauri app) that
    /// already owns a wgpu device: open one the ordinary way. `None` if the host
    /// has no adapter (CI), so the test skips.
    async fn embedder_device(
    ) -> Option<(wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .ok()?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("embedder"),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .ok()?;
        Some((instance, adapter, device, queue))
    }

    /// Top row red, bottom row blue (RGBA8).
    fn pattern() -> Vec<u8> {
        let mut p = Vec::with_capacity((W * H * 4) as usize);
        for y in 0..H {
            let px = if y == 0 { [255, 0, 0, 255] } else { [0, 0, 255, 255] };
            for _ in 0..W {
                p.extend_from_slice(&px);
            }
        }
        p
    }

    /// Upload `pixels` to an RGBA8 texture on `device` (the "decoded frame" a GPU
    /// element would emit), readable back via COPY_SRC.
    fn upload(device: &wgpu::Device, queue: &wgpu::Queue, pixels: &[u8]) -> wgpu::Texture {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("decoded-frame"),
            size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(W * 4),
                rows_per_image: Some(H),
            },
            wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        );
        texture
    }

    /// Read an RGBA8 `W x H` texture back to bytes using `device` / `queue`. If
    /// `texture` was created on a *different* device this is a wgpu validation
    /// error, which is exactly the property the test relies on.
    fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u8> {
        let size = (W * 4 * H) as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(W * 4),
                    rows_per_image: Some(H),
                },
            },
            wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        );
        queue.submit([enc.finish()]);
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range().to_vec();
        buffer.unmap();
        data
    }

    /// M263 keystone for the game-engine / lightweight-app wedge: a frame produced
    /// on an *embedder-supplied* device (via `GpuContext::from_wgpu`) carries a
    /// `WgpuTexture` the embedder can use on its *own* device with no copy. If g2g
    /// had opened its own device (the `headless` path), the texture would be bound
    /// to a different device and this read-back would be a validation error.
    #[tokio::test]
    async fn from_wgpu_texture_is_usable_on_the_embedders_own_device() {
        let Some((instance, adapter, device, queue)) = embedder_device().await else {
            std::eprintln!("no wgpu adapter; skipping bring-your-own-device test");
            return;
        };
        // The embedder keeps its own device/queue handles to render with.
        let (embedder_device, embedder_queue) = (device.clone(), queue.clone());

        // g2g joins the embedder's device instead of opening one.
        let ctx = GpuContext::from_wgpu(instance, adapter, device, queue);

        // A "decoded frame" a g2g GPU element emits, produced on `ctx` (= the
        // embedder's device).
        let pixels = pattern();
        let texture = upload(&ctx.device, &ctx.queue, &pixels);
        let frame = Frame::new(
            MemoryDomain::WgpuTexture(OwnedWgpuTexture::new(
                W,
                H,
                Arc::new(WgpuTextureKeepAlive(texture)),
            )),
            FrameTiming::default(),
            0,
        );

        // The embedder recovers the texture from the frame and reads it back on
        // its *own* device handles, never touching `ctx`: the texture is a
        // first-class object in the embedder's render graph, zero-copy.
        let MemoryDomain::WgpuTexture(owned) = &frame.domain else {
            panic!("expected a WgpuTexture frame");
        };
        let tex = texture_of(owned).expect("recover the wgpu texture");
        let got = read_back(&embedder_device, &embedder_queue, tex);
        assert_eq!(got, pixels, "g2g's texture reads back correctly on the embedder's own device");
    }
}
