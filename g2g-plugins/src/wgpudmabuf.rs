//! Zero-copy GPU -> DMABUF *export* element (`wgputodmabuf`), the producer half
//! that pairs with the [`DmaBufToWgpu`](crate::dmabufwgpu::DmaBufToWgpu) importer
//! across a process boundary.
//!
//! `WgpuToDmaBuf` consumes a GPU-resident [`MemoryDomain::WgpuBuffer`] frame and
//! emits a [`MemoryDomain::DmaBuf`] one referencing the *same* pixels, so a
//! rendered / decoded GPU frame can leave the process with no CPU copy: feed the
//! emitted dma-buf to a [`DmaBufSink`](crate::localdmabuf::DmaBufSink) (M557) and
//! the peer re-imports it with `DmaBufToWgpu`. This is the export mirror of the
//! import side of DESIGN.md 7 and the GPU producer named as the M557 follow-up.
//!
//! # How the export works
//!
//! A `wgpu::Buffer` wgpu allocated itself is not exportable (wgpu does not request
//! external-memory flags), so the element allocates its *own* Vulkan buffer backed
//! by `VkExportMemoryAllocateInfo` with the dma-buf handle type, copies the input
//! into it on the GPU (`copy_buffer_to_buffer`), and exports the backing memory as
//! a dma-buf fd with `vkGetMemoryFdKHR`. Because the input and the exportable
//! buffer must share one `wgpu::Device` for that GPU copy, a producer feeding this
//! element must allocate on the element's device (see [`WgpuToDmaBuf::gpu`] and
//! [`WgpuToDmaBuf::wrap_buffer`]); a cross-device hand-off is impossible without a
//! copy anyway.
//!
//! # Lifetime
//!
//! An exported dma-buf fd is an *independent* reference to the underlying buffer
//! (standard dma-buf refcounting), so once the fd is exported the element frees
//! its own Vulkan handles immediately and the buffer stays alive through the fd
//! (and, once `DmaBufSink` sends it, through the receiver's `SCM_RIGHTS` dup).
//! This is validated end-to-end (export -> free -> re-import on a second device ->
//! read back) in `wgpu_dmabuf_roundtrip`.
//!
//! # Synchronisation
//!
//! By default the element waits for the copy to complete on its own device
//! (`device.poll(Wait)`) before exporting, so a consumer that reads the dma-buf
//! sees finished pixels. That trades a per-frame stall for correctness without a
//! shared semaphore.
//!
//! [`with_external_semaphore`](WgpuToDmaBuf::with_external_semaphore)`(true)` drops
//! that stall (M562): the element exports a persistent `VK_KHR_external_semaphore_fd`
//! *timeline* semaphore once, signals the next value on each frame's copy submit
//! (via `wgpu_hal::vulkan::Queue::add_signal_semaphore`, no `poll(Wait)`), and
//! attaches the semaphore fd + value to the emitted dma-buf ([`OwnedDmaBuf::with_sync`]).
//! A sem-aware [`DmaBufToWgpu`](crate::dmabufwgpu::DmaBufToWgpu) imports the semaphore
//! and host-waits each value before reading, so the wait moves to the consumer and
//! the producer pipelines ahead. The exportable copy buffers are reclaimed lazily
//! once the timeline counter passes their value (a non-blocking poll), so nothing
//! stalls yet no buffer is freed while its copy is still in flight. Leave the mode
//! off if the dma-buf may be consumed by something that does not honour the
//! semaphore. Cross-device / cross-process timeline share is validated on the RTX
//! 3060 (`dmabuf_timeline_probe`, `m562_dmabuf_semaphore_sync`).
//!
//! Hardware: needs a Vulkan device with `VK_KHR_external_memory_fd` +
//! `VK_EXT_external_memory_dma_buf` *export* support (validated on the RTX 3060 via
//! `dmabuf_export_probe`). CI-excluded like the rest of the GPU stack.

use core::any::Any;
use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::sync::Arc;

use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};

use ash::vk;

use g2g_core::memory::{
    DomainSet, MemoryDomain, MemoryDomainKind, OwnedDmaBuf, OwnedWgpuBuffer, SyncFd,
    WgpuBufferKeepAlive,
};
use g2g_core::pad_template::{PadTemplate, PadTemplates};
use g2g_core::{
    AsyncElement, Caps, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError, HardwareError,
    OutputSink, PipelinePacket, Rate, RawVideoFormat,
};

use crate::dmabufwgpu::{dmabuf_frame_bytes, dmabuf_row_stride, DmaBufWgpuBuffer};

/// Formats this element exports: packed RGBA/BGRA (one plane) and 8-bit NV12 (a
/// packed luma + interleaved-chroma buffer, luma stride). The frame byte size and
/// row stride come from the shared [`dmabuf_frame_bytes`] / [`dmabuf_row_stride`]
/// helpers so the export and the [`DmaBufToWgpu`] import agree.
const FORMATS: [RawVideoFormat; 3] =
    [RawVideoFormat::Rgba8, RawVideoFormat::Bgra8, RawVideoFormat::Nv12];

fn gpu_err() -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

fn supported(format: RawVideoFormat) -> bool {
    FORMATS.contains(&format)
}

/// Owner for a plain exportable-device `wgpu::Buffer`: what
/// [`WgpuToDmaBuf::wrap_buffer`] attaches so a producer on the element's device
/// hands it a `WgpuBuffer` frame this element can recover and copy from.
#[derive(Debug)]
pub struct PlainWgpuBuffer {
    buffer: wgpu::Buffer,
    _device: wgpu::Device,
}

impl PlainWgpuBuffer {
    /// The wrapped buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl WgpuBufferKeepAlive for PlainWgpuBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Recover the input `wgpu::Buffer` from a `WgpuBuffer` frame's keep-alive. Known
/// owners: this element's [`PlainWgpuBuffer`] and the importer's
/// [`DmaBufWgpuBuffer`] (so an import -> GPU-work -> export chain on one device
/// flows through).
fn input_buffer(owned: &OwnedWgpuBuffer) -> Option<&wgpu::Buffer> {
    let any = owned.keep_alive().as_any();
    if let Some(p) = any.downcast_ref::<PlainWgpuBuffer>() {
        return Some(p.buffer());
    }
    if let Some(d) = any.downcast_ref::<DmaBufWgpuBuffer>() {
        return Some(d.buffer());
    }
    None
}

/// GPU -> DMABUF export element. See the module docs.
#[derive(Debug)]
pub struct WgpuToDmaBuf {
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    configured: bool,
    /// Pixel format, luma/packed row stride, and height from the negotiated caps;
    /// the exported buffer is `dmabuf_frame_bytes(format, stride, height)` bytes.
    format: RawVideoFormat,
    stride: u32,
    height: u32,
    exported: u64,
    /// Zero-stall sync mode (see module docs "Synchronisation"): when set, signal a
    /// timeline semaphore on the copy submit and attach it to the frame instead of
    /// blocking on `device.poll(Wait)`. Off by default (a consumer that does not
    /// honour the semaphore would read torn pixels).
    external_semaphore: bool,
    /// The persistent exportable timeline semaphore (created once on the export
    /// device, signalled per frame, destroyed on drop) and its exported fd shared
    /// into every frame. Only used when `external_semaphore` is set.
    semaphore: Option<vk::Semaphore>,
    sync_fd: Option<SyncFd>,
    /// Monotonic timeline value; frame N signals `signal_value` = N.
    signal_value: u64,
    /// Exportable `dst` buffers whose copy submit may still be in flight, with the
    /// timeline value that retires them. Freed lazily once the semaphore counter
    /// passes their value (a non-blocking poll), so the element never stalls on the
    /// copy yet never frees a buffer the GPU is still reading (only in semaphore
    /// mode; the poll(Wait) path frees `dst` inline).
    pending: alloc::vec::Vec<(u64, wgpu::Buffer)>,
}

impl Default for WgpuToDmaBuf {
    fn default() -> Self {
        Self::new()
    }
}

impl WgpuToDmaBuf {
    pub fn new() -> Self {
        Self {
            device: None,
            queue: None,
            configured: false,
            format: RawVideoFormat::Rgba8,
            stride: 0,
            height: 0,
            exported: 0,
            external_semaphore: false,
            semaphore: None,
            sync_fd: None,
            signal_value: 0,
            pending: alloc::vec::Vec::new(),
        }
    }

    /// Enable zero-stall cross-process sync (default off): export a timeline
    /// semaphore signalled by each frame's GPU copy and attach it to the emitted
    /// dma-buf, instead of blocking on the copy with `device.poll(Wait)`. The
    /// downstream consumer (a sem-aware [`DmaBufToWgpu`], across
    /// [`DmaBufSink`](crate::localdmabuf::DmaBufSink)) waits on the semaphore before
    /// reading. Leave off when the dma-buf may be consumed by something that does
    /// not honour the semaphore.
    pub fn with_external_semaphore(mut self, on: bool) -> Self {
        self.external_semaphore = on;
        self
    }

    /// Frames exported so far. Useful in tests.
    pub fn exported(&self) -> u64 {
        self.exported
    }

    /// Ensure the export device exists and return clones of it and its queue, so a
    /// producer (or a test) can allocate its input `wgpu::Buffer` on the *same*
    /// device (required for the GPU copy into the exportable buffer).
    pub async fn gpu(&mut self) -> Result<(wgpu::Device, wgpu::Queue), G2gError> {
        if self.device.is_none() {
            let (device, queue) = create_export_device().await?;
            self.device = Some(device);
            self.queue = Some(queue);
        }
        Ok((self.device.clone().unwrap(), self.queue.clone().unwrap()))
    }

    /// Wrap a `wgpu::Buffer` (allocated on this element's [`gpu`](Self::gpu)
    /// device, with `COPY_SRC` usage) as a `WgpuBuffer` frame domain this element
    /// accepts.
    pub fn wrap_buffer(device: &wgpu::Device, buffer: wgpu::Buffer, len: usize) -> OwnedWgpuBuffer {
        OwnedWgpuBuffer::new(len, Arc::new(PlainWgpuBuffer { buffer, _device: device.clone() }))
    }

    /// Create the persistent exportable timeline semaphore and export its fd, once.
    /// Requires the export device (call [`gpu`](Self::gpu) first).
    fn ensure_semaphore(&mut self) -> Result<(), G2gError> {
        if self.semaphore.is_some() {
            return Ok(());
        }
        let device = self.device.as_ref().ok_or_else(gpu_err)?;
        // SAFETY: `device` is a Vulkan export device carrying
        // VK_KHR_external_semaphore_fd (added in `create_export_device`).
        let (sem, fd) = unsafe { create_export_semaphore(device)? };
        self.semaphore = Some(sem);
        // SAFETY: `fd` is a fresh, owned OPAQUE_FD from the export above.
        self.sync_fd = Some(unsafe { SyncFd::from_raw(fd) });
        Ok(())
    }

    /// Free any pending exportable buffer whose copy submit has retired (the
    /// timeline counter reached its signalled value). Non-blocking: the element
    /// never stalls on the copy, it just reclaims buffers a frame or two later.
    fn reap_retired(&mut self) {
        let Some(sem) = self.semaphore else { return };
        let Some(device) = self.device.as_ref() else { return };
        // SAFETY: `sem` is a live timeline semaphore on `device`; reading its
        // counter is a non-blocking query.
        let counter = unsafe {
            device
                .as_hal::<wgpu_hal::api::Vulkan>()
                .and_then(|hal| hal.raw_device().get_semaphore_counter_value(sem).ok())
        };
        let Some(counter) = counter else { return };
        // A copy signalling `value` has retired once the counter reaches `value`.
        self.pending.retain(|(value, _)| *value > counter);
    }
}

impl Drop for WgpuToDmaBuf {
    fn drop(&mut self) {
        let Some(device) = self.device.as_ref() else { return };
        // Any in-flight copy must finish before we free its `dst` and destroy the
        // semaphore it signals. A blocking poll is fine here (shutdown, not the hot
        // path). Then the pending buffers drop (freeing their Vulkan handles; the
        // exported dma-buf fds keep the memory alive for consumers) and the
        // timeline semaphore is destroyed.
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        self.pending.clear();
        if let Some(sem) = self.semaphore.take() {
            // SAFETY: the semaphore was created on this device; after the wait above
            // no submission still references it, so destroying it is legal.
            unsafe {
                if let Some(hal) = device.as_hal::<wgpu_hal::api::Vulkan>() {
                    hal.raw_device().destroy_semaphore(sem, None);
                }
            }
        }
    }
}

impl PadTemplates for WgpuToDmaBuf {
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

impl AsyncElement for WgpuToDmaBuf {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "wgpu buffer to DMABUF export",
            "Filter/Converter/Video/GPU",
            "Exports a GPU-resident wgpu buffer as a dma-buf fd (zero-copy GPU frame egress)",
            "g2g",
        )
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Only the memory domain changes (WgpuBuffer -> DmaBuf); pixel caps pass
        // through unchanged.
        match upstream_caps {
            Caps::RawVideo { format, .. } if supported(*format) => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::WgpuBuffer)
    }

    fn output_memory(&self) -> MemoryDomainKind {
        MemoryDomainKind::DmaBuf
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (format, w, h) = match absolute_caps {
            Caps::RawVideo { format, width: Dim::Fixed(w), height: Dim::Fixed(h), .. } => {
                (*format, *w, *h)
            }
            _ => return Err(G2gError::CapsMismatch),
        };
        self.stride = dmabuf_row_stride(format, w).ok_or(G2gError::CapsMismatch)?;
        self.format = format;
        self.height = h;
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
                    let MemoryDomain::WgpuBuffer(owned) = &frame.domain else {
                        // Export path only; a non-wgpu frame is not ours.
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let src = input_buffer(owned).ok_or(G2gError::UnsupportedDomain)?;
                    self.gpu().await?;

                    let size =
                        dmabuf_frame_bytes(self.format, u64::from(self.stride), u64::from(self.height))
                            .ok_or(G2gError::CapsMismatch)?;
                    if size == 0 || (owned.len as u64) < size {
                        return Err(G2gError::CapsMismatch);
                    }

                    // Zero-stall mode: create the export semaphore once, signal the
                    // next timeline value on the copy submit, and attach it to the
                    // frame. Otherwise the copy is drained inline (poll(Wait)).
                    let sync = if self.external_semaphore {
                        self.ensure_semaphore()?;
                        self.signal_value += 1;
                        Some((self.semaphore.unwrap(), self.signal_value))
                    } else {
                        None
                    };

                    let device = self.device.as_ref().unwrap();
                    let queue = self.queue.as_ref().unwrap();
                    // SAFETY: `device` carries the dma-buf export extensions; `src`
                    // is a live buffer on `device` of at least `size` bytes; `sync`,
                    // when set, names this device's timeline semaphore.
                    let (fd, dst) = unsafe { export_copy(device, queue, src, size, sync)? };

                    // SAFETY: `fd` is a fresh dma-buf fd owned by this process;
                    // OwnedDmaBuf closes it once on drop. Stride = tight row bytes,
                    // offset 0 (packed single plane).
                    let mut dmabuf = unsafe { OwnedDmaBuf::from_raw(fd, self.stride, 0) };
                    if let (Some(sf), Some((_, value))) = (&self.sync_fd, sync) {
                        dmabuf = dmabuf.with_sync(sf.clone(), value);
                    }
                    if let Some(dst) = dst {
                        // Keep the exportable buffer alive until its copy retires,
                        // then free lazily (no stall). See `pending`.
                        self.pending.push((self.signal_value, dst));
                        self.reap_retired();
                    }
                    let mut out_frame = frame;
                    out_frame.domain = MemoryDomain::DmaBuf(dmabuf);
                    self.exported += 1;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

/// Build a Vulkan wgpu device with the dma-buf export extensions.
async fn create_export_device() -> Result<(wgpu::Device, wgpu::Queue), G2gError> {
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
    // SAFETY: read the hal adapter only to open a device carrying the export
    // extensions; the guard outlives the open call.
    let open = unsafe {
        let hal = adapter.as_hal::<wgpu_hal::api::Vulkan>().ok_or_else(gpu_err)?;
        hal.open_with_callback(
            wgpu::Features::empty(),
            &wgpu::Limits::default(),
            &wgpu::MemoryHints::default(),
            Some(Box::new(|args: wgpu_hal::vulkan::CreateDeviceCallbackArgs| {
                args.extensions.push(ash::khr::external_memory_fd::NAME);
                args.extensions.push(ash::ext::external_memory_dma_buf::NAME);
                // For the optional zero-stall timeline-semaphore export path.
                args.extensions.push(ash::khr::external_semaphore_fd::NAME);
            })),
        )
    }
    .map_err(|_| gpu_err())?;
    // SAFETY: `open` came from this adapter's hal.
    let (device, queue) = unsafe {
        adapter.create_device_from_hal(
            open,
            &wgpu::DeviceDescriptor { label: Some("wgputodmabuf"), ..Default::default() },
        )
    }
    .map_err(|_| gpu_err())?;
    Ok((device, queue))
}

/// Create a persistent exportable *timeline* semaphore on `device` (initial value
/// 0) and export its `OPAQUE_FD`. The semaphore lives for the element's lifetime
/// (signalled per frame, destroyed on drop); the fd is shared into every exported
/// frame. Cross-device / cross-process timeline share is validated on the RTX 3060
/// by `dmabuf_timeline_probe`.
///
/// # Safety
/// `device` must be a Vulkan-backend wgpu device carrying
/// `VK_KHR_external_semaphore_fd` (added in [`create_export_device`]).
unsafe fn create_export_semaphore(device: &wgpu::Device) -> Result<(vk::Semaphore, i32), G2gError> {
    // SAFETY: caller guarantees the export device; the hal guard is held for the
    // whole creation + export.
    unsafe {
        let hal = device.as_hal::<wgpu_hal::api::Vulkan>().ok_or_else(gpu_err)?;
        let raw = hal.raw_device();
        let instance = hal.shared_instance().raw_instance();

        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let mut export = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
        let info =
            vk::SemaphoreCreateInfo::default().push_next(&mut type_info).push_next(&mut export);
        let sem = raw.create_semaphore(&info, None).map_err(|_| gpu_err())?;

        let loader = ash::khr::external_semaphore_fd::Device::new(instance, raw);
        let fd = loader.get_semaphore_fd(
            &vk::SemaphoreGetFdInfoKHR::default()
                .semaphore(sem)
                .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD),
        );
        match fd {
            Ok(fd) if fd >= 0 => Ok((sem, fd)),
            _ => {
                raw.destroy_semaphore(sem, None);
                Err(gpu_err())
            }
        }
    }
}

/// Pick a memory type satisfying `type_bits` with `flags` set.
fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| {
        (type_bits & (1 << i)) != 0 && props.memory_types[i as usize].property_flags.contains(flags)
    })
}

/// Allocate an exportable `size`-byte Vulkan buffer, copy `src` into it on the
/// GPU, and export the backing memory as a dma-buf fd.
///
/// Synchronisation depends on `sync`:
/// - `None`: block on `device.poll(Wait)` until the copy finishes, then free the
///   exportable `dst` buffer inline and return `(fd, None)`. A consumer may read
///   the dma-buf immediately.
/// - `Some((sem, value))`: inject a timeline signal of `value` on `sem` into the
///   copy submit and return without waiting, handing `dst` back as `(fd, Some(dst))`
///   so the caller keeps it alive until the copy retires (freeing it while the
///   submit is in flight would be a GPU use-after-free). The consumer host-waits
///   `value` on the imported semaphore before reading.
///
/// In both cases the exported dma-buf fd is an independent reference to the
/// underlying memory, so freeing `dst` later does not invalidate it.
///
/// # Safety
/// `device` must be a Vulkan-backend wgpu device carrying
/// `VK_KHR_external_memory_fd` + `VK_EXT_external_memory_dma_buf` (and
/// `VK_KHR_external_semaphore_fd` when `sync` is set), and `src` a live buffer on
/// it of at least `size` bytes with `COPY_SRC` usage; `sem` (when set) is a
/// timeline semaphore on `device`.
unsafe fn export_copy(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    size: u64,
    sync: Option<(vk::Semaphore, u64)>,
) -> Result<(i32, Option<wgpu::Buffer>), G2gError> {
    // Create the exportable Vulkan buffer + dedicated exported memory, and export
    // its fd, all through the raw device.
    let (vk_buffer, vk_memory, fd) = {
        // SAFETY: caller guarantees a Vulkan export device; the hal guard is held
        // for the whole allocation and the raw handles are handed to wgpu below.
        unsafe {
            let hal = device.as_hal::<wgpu_hal::api::Vulkan>().ok_or_else(gpu_err)?;
            let raw = hal.raw_device();
            let instance = hal.shared_instance().raw_instance();
            let phys = hal.raw_physical_device();

            let mut ext = vk::ExternalMemoryBufferCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let buf_info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::TRANSFER_SRC,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .push_next(&mut ext);
            let buffer = raw.create_buffer(&buf_info, None).map_err(|_| gpu_err())?;

            let reqs = raw.get_buffer_memory_requirements(buffer);
            let props = instance.get_physical_device_memory_properties(phys);
            let Some(mem_type) =
                find_memory_type(&props, reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            else {
                raw.destroy_buffer(buffer, None);
                return Err(gpu_err());
            };

            // Export as dma-buf, dedicated to this buffer.
            let mut export = vk::ExportMemoryAllocateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(reqs.size)
                .memory_type_index(mem_type)
                .push_next(&mut export)
                .push_next(&mut dedicated);
            let memory = match raw.allocate_memory(&alloc, None) {
                Ok(m) => m,
                Err(_) => {
                    raw.destroy_buffer(buffer, None);
                    return Err(gpu_err());
                }
            };
            if raw.bind_buffer_memory(buffer, memory, 0).is_err() {
                raw.free_memory(memory, None);
                raw.destroy_buffer(buffer, None);
                return Err(gpu_err());
            }

            let loader = ash::khr::external_memory_fd::Device::new(instance, raw);
            let fd = loader.get_memory_fd(
                &vk::MemoryGetFdInfoKHR::default()
                    .memory(memory)
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT),
            );
            match fd {
                Ok(fd) if fd >= 0 => (buffer, memory, fd),
                _ => {
                    raw.free_memory(memory, None);
                    raw.destroy_buffer(buffer, None);
                    return Err(gpu_err());
                }
            }
        }
    };

    // Own the fd immediately so any early return closes it.
    // SAFETY: `fd` is a fresh dma-buf fd just exported and owned by this process.
    let owned_fd = unsafe { OwnedFd::from_raw_fd(fd) };

    // Wrap the exportable buffer as wgpu (which takes ownership of the raw handles
    // and frees them on drop) and copy the input into it on the GPU.
    // SAFETY: `vk_buffer` is bound to `vk_memory` at offset 0 for `size` bytes and
    // we relinquish the raw handles to wgpu here.
    let hal_buffer =
        unsafe { wgpu_hal::vulkan::Buffer::from_raw_managed(vk_buffer, vk_memory, 0, size) };
    // SAFETY: `hal_buffer` was produced from this device's hal.
    let dst = unsafe {
        device.create_buffer_from_hal::<wgpu_hal::api::Vulkan>(
            hal_buffer,
            &wgpu::BufferDescriptor {
                label: Some("dmabuf-export"),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )
    };
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc.copy_buffer_to_buffer(src, 0, &dst, 0, size);

    let kept = match sync {
        Some((sem, value)) => {
            // Zero-stall: inject a timeline signal of `value` on the copy submit so
            // the consumer can order its read after the copy on the GPU timeline,
            // and return WITHOUT blocking. `dst` is handed back to the caller, which
            // frees it only once the copy retires (freeing it now, mid-flight, would
            // be a GPU use-after-free).
            // SAFETY: `queue` is this device's live Vulkan queue; `sem` is a timeline
            // semaphore on the device, signalled by the submit below.
            let hal_q = unsafe { queue.as_hal::<wgpu_hal::api::Vulkan>() }.ok_or_else(gpu_err)?;
            hal_q.add_signal_semaphore(sem, Some(value));
            queue.submit([enc.finish()]);
            Some(dst)
        }
        None => {
            queue.submit([enc.finish()]);
            // Wait for the copy so a consumer of the dma-buf sees finished pixels.
            device
                .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
                .map_err(|_| gpu_err())?;
            // Drop the wgpu wrapper: wgpu frees vk_buffer + vk_memory. The exported
            // dma-buf fd remains a valid, independent reference to the memory.
            drop(dst);
            None
        }
    };

    // Hand the fd out (into_raw_fd; the returned i32 is owned by the caller).
    Ok((owned_fd.into_raw_fd(), kept))
}
