//! Zero-copy DMABUF -> GPU import element (`dmabuftowgpu`).
//!
//! Consumes a [`MemoryDomain::DmaBuf`] frame (e.g. imported into the pipeline by
//! `g2g-bridge` from a GStreamer `GstDmaBufMemory`, or from a GPU/CSI producer)
//! and emits a GPU-resident [`MemoryDomain::WgpuBuffer`] that aliases the same
//! memory with no CPU copy: the dma-buf fd is imported into a Vulkan buffer via
//! `VK_EXT_external_memory_dma_buf`, wrapped as a `wgpu::Buffer`, and handed
//! downstream for a wgpu compute stage (`WgpuPreprocess` / `WgpuInference`).
//!
//! This is the GPU-consuming counterpart to the bridge's dma-buf ingest side
//! (`AppSrcFeed::push_dmabuf`): together they are the `GstDmaBufMemory` -> g2g
//! GPU zero-copy path of DESIGN.md §7.
//!
//! Hardware note (measured, see also `libcamera_dmabuf`): a discrete GPU imports
//! only *GPU-visible* dma-bufs (allocated by a GPU / CSI-ISP, or GPU-exported),
//! not a CPU/vmalloc-backed one (a USB webcam, a udmabuf). The import reports a
//! clear error (`UnsupportedDomain`) when the driver cannot bind the fd, so the
//! caller can fall back to a CPU download path.

use core::any::Any;
use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::sync::Arc;

use std::os::fd::{BorrowedFd, FromRawFd, IntoRawFd, OwnedFd};
use std::time::{Duration, Instant};

use ash::vk;

use g2g_core::memory::{
    DomainSet, MemoryDomain, MemoryDomainKind, OwnedWgpuBuffer, WgpuBufferKeepAlive,
};
use g2g_core::pad_template::{PadTemplate, PadTemplates};
use g2g_core::{
    AsyncElement, Caps, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError, HardwareError,
    OutputSink, PipelinePacket, Rate, RawVideoFormat,
};

/// Raw formats the import accepts (the pixel caps pass through unchanged; only
/// the memory domain changes from dma-buf to a GPU buffer).
const FORMATS: [RawVideoFormat; 4] = [
    RawVideoFormat::Rgba8,
    RawVideoFormat::Bgra8,
    RawVideoFormat::Nv12,
    RawVideoFormat::I420,
];

fn gpu_err() -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

/// Total packed byte size of one frame for a dma-buf laid out with luma row
/// stride `stride` and `height` rows. Packed RGBA/BGRA is one plane
/// (`stride * height`); 8-bit NV12 / I420 add a half-height chroma region
/// (`stride * ceil(height/2)`), which is the same total (`stride * height * 3/2`)
/// whether the chroma is interleaved (NV12) or split (I420) as long as the luma
/// stride is used. Shared by the import ([`DmaBufToWgpu`]) and export
/// ([`crate::wgpudmabuf::WgpuToDmaBuf`]) so both agree on the buffer size.
/// Returns `None` for a format this element does not carry.
pub(crate) fn dmabuf_frame_bytes(format: RawVideoFormat, stride: u64, height: u64) -> Option<u64> {
    let luma = stride.checked_mul(height)?;
    match format {
        RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8 => Some(luma),
        RawVideoFormat::Nv12 | RawVideoFormat::I420 => {
            let chroma = stride.checked_mul(height.div_ceil(2))?;
            luma.checked_add(chroma)
        }
        _ => None,
    }
}

/// Row stride (bytes) for the luma / packed plane of `format` at `width`. RGBA is
/// 4 bytes/pixel; 8-bit NV12 / I420 luma is 1 byte/pixel.
pub(crate) fn dmabuf_row_stride(format: RawVideoFormat, width: u32) -> Option<u32> {
    match format {
        RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8 => width.checked_mul(4),
        RawVideoFormat::Nv12 | RawVideoFormat::I420 => Some(width),
        _ => None,
    }
}

/// Keep-alive owner for a [`MemoryDomain::WgpuBuffer`] backed by an imported
/// dma-buf: holds the `wgpu::Buffer` (which, via `from_raw_managed`, owns the
/// imported `VkDeviceMemory` and closes the dup'ed fd on drop) and the device
/// needed to use it. A downstream wgpu consumer downcasts via [`Any`] to recover
/// the buffer.
#[derive(Debug)]
pub struct DmaBufWgpuBuffer {
    // Field order is drop order: the buffer (and its backing imported memory) is
    // released before the device.
    buffer: wgpu::Buffer,
    _device: wgpu::Device,
}

impl DmaBufWgpuBuffer {
    /// The imported GPU buffer, for a downstream stage that links wgpu.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl WgpuBufferKeepAlive for DmaBufWgpuBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// DMABUF -> `wgpu::Buffer` import element. See the module docs.
#[derive(Debug)]
pub struct DmaBufToWgpu {
    configured: bool,
    /// The Vulkan wgpu device carrying the external-memory import extensions,
    /// built lazily on the first frame (device creation is async) and reused.
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    /// Frame height, from the negotiated caps: the imported buffer size is
    /// `offset + dmabuf_frame_bytes(format, stride, height)`.
    height: u32,
    /// Pixel format, from the negotiated caps (drives the plane-aware size).
    format: RawVideoFormat,
    /// Frames imported so far.
    imported: u64,
    /// The producer's timeline semaphore, imported once from the first frame that
    /// carries a sync fd (see [`OwnedDmaBuf::sync_fd`]) and reused. When present,
    /// each synced frame's completion value is polled (yielding cooperatively
    /// between polls) before the buffer is handed downstream, so the producer never
    /// blocks on the copy and the consumer never blocks the runtime.
    semaphore: Option<vk::Semaphore>,
}

impl Default for DmaBufToWgpu {
    fn default() -> Self {
        Self::new()
    }
}

impl DmaBufToWgpu {
    pub fn new() -> Self {
        Self {
            configured: false,
            device: None,
            queue: None,
            height: 0,
            format: RawVideoFormat::Rgba8,
            imported: 0,
            semaphore: None,
        }
    }

    /// Frames imported so far. Useful in tests.
    pub fn imported(&self) -> u64 {
        self.imported
    }

    /// The import device, once built (on the first frame). Lets a consumer read
    /// back an imported buffer, which lives on this device.
    pub fn device(&self) -> Option<&wgpu::Device> {
        self.device.as_ref()
    }

    /// The import queue, once built.
    pub fn queue(&self) -> Option<&wgpu::Queue> {
        self.queue.as_ref()
    }
}

impl Drop for DmaBufToWgpu {
    fn drop(&mut self) {
        if let (Some(sem), Some(device)) = (self.semaphore.take(), self.device.as_ref()) {
            // The host waits above already blocked until each signalled value was
            // reached, so no submission still references the semaphore; destroying
            // it is legal. A device poll first is belt-and-braces for shutdown.
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            // SAFETY: `sem` was imported on this device and is no longer in use.
            unsafe {
                if let Some(hal) = device.as_hal::<wgpu_hal::api::Vulkan>() {
                    hal.raw_device().destroy_semaphore(sem, None);
                }
            }
        }
    }
}

impl PadTemplates for DmaBufToWgpu {
    fn pad_templates() -> alloc::vec::Vec<PadTemplate> {
        let any = |format| Caps::RawVideo {
            format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        let set = CapsSet::from_alternatives(FORMATS.map(any).to_vec());
        alloc::vec![PadTemplate::sink(set.clone()), PadTemplate::source(set)]
    }
}

impl AsyncElement for DmaBufToWgpu {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "DMABUF to wgpu buffer import",
            "Filter/Converter/Video/GPU",
            "Zero-copy import of a dma-buf frame into a GPU-resident wgpu buffer",
            "g2g",
        )
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Only the memory domain changes (DmaBuf -> WgpuBuffer); the pixel caps
        // pass through unchanged.
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        (self.format, self.height) = match absolute_caps {
            Caps::RawVideo {
                format,
                height: g2g_core::Dim::Fixed(h),
                ..
            } => (*format, *h),
            _ => return Err(G2gError::CapsMismatch),
        };
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    /// Input pad accepts only dma-buf frames (M354 domain nego); the auto-plug
    /// splices this element where an upstream produces a dma-buf.
    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::DmaBuf)
    }

    fn output_memory(&self) -> MemoryDomainKind {
        MemoryDomainKind::WgpuBuffer
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
                    let MemoryDomain::DmaBuf(dmabuf) = &frame.domain else {
                        // Import path only; a system-memory frame is not ours.
                        return Err(G2gError::UnsupportedDomain);
                    };
                    if self.device.is_none() {
                        let (device, queue) = create_import_device().await?;
                        self.device = Some(device);
                        self.queue = Some(queue);
                    }
                    let device = self.device.clone().unwrap();

                    let stride = u64::from(dmabuf.stride);
                    // Plane-aware size: RGBA is one plane, NV12 / I420 add the
                    // half-height chroma region (a bare stride*height would import
                    // only the luma plane of a planar frame).
                    let plane_bytes =
                        dmabuf_frame_bytes(self.format, stride, u64::from(self.height))
                            .ok_or(G2gError::CapsMismatch)?;
                    let size = u64::from(dmabuf.offset) + plane_bytes;
                    if size == 0 {
                        return Err(G2gError::CapsMismatch);
                    }
                    // Cross-process GPU sync: if the producer attached a timeline
                    // semaphore (zero-stall `WgpuToDmaBuf`), import it once and
                    // host-wait this frame's value before the buffer is read
                    // downstream. Without a sync fd the buffer is assumed complete
                    // (the producer synchronised itself).
                    let sync = dmabuf.sync_fd().zip(dmabuf.sync_value());
                    if let Some((fd, value)) = sync {
                        if self.semaphore.is_none() {
                            // SAFETY: `device` carries VK_KHR_external_semaphore_fd;
                            // `fd` is a live exported timeline-semaphore fd owned by
                            // `frame` (duplicated before import).
                            self.semaphore = Some(unsafe { import_timeline(&device, fd)? });
                        }
                        let sem = self.semaphore.unwrap();
                        // Cooperative wait for the producer's copy: poll the timeline
                        // counter and yield to the executor between polls, rather than
                        // a blocking `vkWaitSemaphores` that would stall the whole
                        // runtime (and any sibling task, e.g. the source pulling the
                        // next frame) while the copy is in flight. The common case -
                        // the copy already finished by the time the fd crossed the
                        // socket - passes on the first poll with no yield. A 5 s
                        // deadline guards a producer that never signals. (wgpu-hal 29
                        // exposes no wait-semaphore injection, so a GPU-queue wait is
                        // not available; this keeps the CPU wait off the hot path.)
                        let deadline = Instant::now() + Duration::from_secs(5);
                        loop {
                            // SAFETY: `sem` is a live imported timeline on `device`.
                            if unsafe { timeline_counter(&device, sem)? } >= value {
                                break;
                            }
                            if Instant::now() >= deadline {
                                return Err(gpu_err());
                            }
                            tokio::task::yield_now().await;
                        }
                    }
                    // SAFETY: `device` carries VK_EXT_external_memory_dma_buf; the
                    // fd is a live dma-buf owned by `frame` for this call and is
                    // duplicated before Vulkan takes ownership.
                    let buffer = unsafe { import_dmabuf(&device, dmabuf.as_raw(), size)? };
                    let owner = DmaBufWgpuBuffer {
                        buffer,
                        _device: device.clone(),
                    };
                    let mut gpu_frame = frame;
                    gpu_frame.domain = MemoryDomain::WgpuBuffer(OwnedWgpuBuffer::new(
                        size as usize,
                        Arc::new(owner),
                    ));
                    self.imported += 1;
                    out.push(PipelinePacket::DataFrame(gpu_frame)).await?;
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

/// Build a Vulkan wgpu device with the dma-buf import extensions. Async because
/// adapter/device creation is; called once and the device reused.
async fn create_import_device() -> Result<(wgpu::Device, wgpu::Queue), G2gError> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .map_err(|_| gpu_err())?;
    // SAFETY: read the hal adapter only to open a device carrying the dma-buf
    // import extensions; the guard outlives the open call.
    let open = unsafe {
        let hal_adapter = adapter
            .as_hal::<wgpu_hal::api::Vulkan>()
            .ok_or_else(gpu_err)?;
        hal_adapter.open_with_callback(
            wgpu::Features::empty(),
            &wgpu::Limits::default(),
            &wgpu::MemoryHints::default(),
            Some(Box::new(
                |args: wgpu_hal::vulkan::CreateDeviceCallbackArgs| {
                    args.extensions.push(ash::khr::external_memory_fd::NAME);
                    args.extensions
                        .push(ash::ext::external_memory_dma_buf::NAME);
                    // For the optional zero-stall timeline-semaphore import path.
                    args.extensions.push(ash::khr::external_semaphore_fd::NAME);
                },
            )),
        )
    }
    .map_err(|_| gpu_err())?;
    // SAFETY: `open` came from this adapter's hal.
    let (device, queue) = unsafe {
        adapter.create_device_from_hal(
            open,
            &wgpu::DeviceDescriptor {
                label: Some("dmabuftowgpu"),
                ..Default::default()
            },
        )
    }
    .map_err(|_| gpu_err())?;
    Ok((device, queue))
}

/// Import an exported timeline-semaphore `fd` (from the producer's
/// [`WgpuToDmaBuf`](crate::wgpudmabuf::WgpuToDmaBuf)) into a timeline semaphore on
/// `device`, so its counter can be host-waited. Permanent import; the caller's
/// `fd` is left untouched (a dup is handed to Vulkan).
///
/// # Safety
/// `device` must carry `VK_KHR_external_semaphore_fd`; `fd` must be a valid open
/// exported OPAQUE_FD timeline semaphore owned by the caller.
unsafe fn import_timeline(device: &wgpu::Device, fd: i32) -> Result<vk::Semaphore, G2gError> {
    // SAFETY: raw device from the live wgpu device; the semaphore created here is
    // returned to the caller (destroyed on the element's drop).
    unsafe {
        let hal = device
            .as_hal::<wgpu_hal::api::Vulkan>()
            .ok_or_else(gpu_err)?;
        let raw = hal.raw_device();
        let instance = hal.shared_instance().raw_instance();

        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let sem = raw
            .create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut type_info),
                None,
            )
            .map_err(|_| gpu_err())?;

        // Dup the caller's fd so the frame keeps ownership of its own.
        let dup = match BorrowedFd::borrow_raw(fd).try_clone_to_owned() {
            Ok(f) => f.into_raw_fd(),
            Err(_) => {
                raw.destroy_semaphore(sem, None);
                return Err(gpu_err());
            }
        };
        let loader = ash::khr::external_semaphore_fd::Device::new(instance, raw);
        if loader
            .import_semaphore_fd(
                &vk::ImportSemaphoreFdInfoKHR::default()
                    .semaphore(sem)
                    .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD)
                    .flags(vk::SemaphoreImportFlags::empty()) // permanent
                    .fd(dup),
            )
            .is_err()
        {
            let _ = OwnedFd::from_raw_fd(dup); // import failed: close the dup
            raw.destroy_semaphore(sem, None);
            return Err(gpu_err());
        }
        Ok(sem)
    }
}

/// Non-blocking read of the imported timeline semaphore's current value, so the
/// import element can poll for the producer's completion and yield between polls
/// instead of blocking the runtime on `vkWaitSemaphores`.
///
/// # Safety
/// `sem` must be a live timeline semaphore on `device`.
unsafe fn timeline_counter(device: &wgpu::Device, sem: vk::Semaphore) -> Result<u64, G2gError> {
    // SAFETY: raw device from the live wgpu device; reading the counter is a
    // non-blocking query.
    unsafe {
        let hal = device
            .as_hal::<wgpu_hal::api::Vulkan>()
            .ok_or_else(gpu_err)?;
        hal.raw_device()
            .get_semaphore_counter_value(sem)
            .map_err(|_| gpu_err())
    }
}

/// Import a dma-buf `fd` (`size` bytes) into a `wgpu::Buffer` that aliases it.
/// The buffer takes ownership of the imported `VkDeviceMemory` (freed, with the
/// dup'ed fd, when the buffer drops). Returns `UnsupportedDomain` when the GPU
/// cannot bind the fd (a CPU-backed dma-buf on a discrete GPU).
///
/// # Safety
/// `device` must carry `VK_EXT_external_memory_dma_buf`; `fd` must be a valid
/// open dma-buf of at least `size` bytes, owned by the caller (it is duplicated
/// before Vulkan takes ownership).
unsafe fn import_dmabuf(
    device: &wgpu::Device,
    fd: i32,
    size: u64,
) -> Result<wgpu::Buffer, G2gError> {
    // SAFETY: raw device from the live wgpu device; the raw objects created here
    // are either handed to wgpu (on success) or freed (on failure).
    let (vk_buffer, vk_memory) = unsafe {
        let hal_device = device
            .as_hal::<wgpu_hal::api::Vulkan>()
            .ok_or_else(gpu_err)?;
        let raw = hal_device.raw_device();
        let instance = hal_device.shared_instance().raw_instance();
        let loader = ash::khr::external_memory_fd::Device::new(instance, raw);

        let mut props = vk::MemoryFdPropertiesKHR::default();
        loader
            .get_memory_fd_properties(
                vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
                fd,
                &mut props,
            )
            .map_err(|_| gpu_err())?;
        let bits = props.memory_type_bits;
        if bits == 0 {
            return Err(G2gError::UnsupportedDomain);
        }

        let mut ext_buf = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let buf_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut ext_buf);
        let buffer = raw.create_buffer(&buf_info, None).map_err(|_| gpu_err())?;

        // The query mask is optimistic; try each candidate memory type until an
        // import + bind succeeds.
        let mut bound = None;
        for type_index in 0..32u32 {
            if bits & (1 << type_index) == 0 {
                continue;
            }
            let dup_fd = match BorrowedFd::borrow_raw(fd).try_clone_to_owned() {
                Ok(f) => f.into_raw_fd(),
                Err(_) => continue,
            };
            let mut import_info = vk::ImportMemoryFdInfoKHR::default()
                .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
                .fd(dup_fd);
            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(size)
                .memory_type_index(type_index)
                .push_next(&mut import_info);
            if let Ok(memory) = raw.allocate_memory(&alloc_info, None) {
                if raw.bind_buffer_memory(buffer, memory, 0).is_ok() {
                    bound = Some(memory);
                    break;
                }
                raw.free_memory(memory, None); // closes the imported dup fd
            }
            // On allocate failure the fd's ownership is driver-dependent; leaking
            // a probe handle is safer than risking a double close.
        }

        match bound {
            Some(memory) => (buffer, memory),
            None => {
                raw.destroy_buffer(buffer, None);
                return Err(G2gError::UnsupportedDomain);
            }
        }
    };

    // Hand the imported buffer + memory to wgpu, which now owns them (frees both,
    // closing the dup fd, when the returned `wgpu::Buffer` drops).
    // SAFETY: `vk_buffer` is bound to `vk_memory` at offset 0 for `size` bytes,
    // and we relinquish all further use of the raw handles here.
    let hal_buffer =
        unsafe { wgpu_hal::vulkan::Buffer::from_raw_managed(vk_buffer, vk_memory, 0, size) };
    // SAFETY: `hal_buffer` was produced from this device's hal.
    let buffer = unsafe {
        device.create_buffer_from_hal::<wgpu_hal::api::Vulkan>(
            hal_buffer,
            &wgpu::BufferDescriptor {
                label: Some("dmabuf-imported"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )
    };
    Ok(buffer)
}
