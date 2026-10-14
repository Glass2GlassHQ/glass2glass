//! Zero-copy GPU -> DMABUF *export* element (`wgputodmabuf`), the producer half
//! that pairs with the [`DmaBufToWgpu`](crate::dmabufwgpu::DmaBufToWgpu) importer
//! across a process boundary.
//!
//! `WgpuToDmaBuf` consumes a GPU-resident [`MemoryDomain::WgpuBuffer`] frame and
//! emits a [`MemoryDomain::DmaBuf`] one referencing the *same* pixels, so a
//! rendered / decoded GPU frame can leave the process with no CPU copy: feed the
//! emitted dma-buf to a [`DmaBufSink`](crate::localdmabuf::DmaBufSink) (M557) and
//! the peer re-imports it with `DmaBufToWgpu`. This is the export mirror of the
//! import side of design/README.md and the GPU producer named as the M557 follow-up.
//!
//! # How the export works
//!
//! A `wgpu::Buffer` wgpu allocated itself is not exportable (wgpu does not request
//! external-memory flags), so the element allocates its *own* Vulkan buffer backed
//! by `VkExportMemoryAllocateInfo` with the dma-buf handle type, copies the input
//! into it on the GPU (`copy_buffer_to_buffer`), and exports the backing memory as
//! a dma-buf fd with `vkGetMemoryFdKHR`. The input and the exportable buffer must
//! share one `wgpu::Device` for that GPU copy, so the element exports on the
//! producer's device, recovered from the frame's keep-alive. wgpu-hal enables
//! `VK_KHR_external_memory_fd` and `VK_EXT_external_memory_dma_buf` on every
//! Vulkan device whose driver offers them, so a plain wgpu device qualifies. The
//! zero-stall mode below also needs `VK_KHR_external_semaphore_fd`, which only the
//! dma-buf import and export devices ([`WgpuToDmaBuf::gpu`]) request.
//!
//! A buffer carrying a [`PlaneLayout`](g2g_core::meta::PlaneLayout) (a padded
//! import from [`DmaBufToWgpu`](crate::dmabufwgpu::DmaBufToWgpu)) is exported in
//! that layout, its plane-0 stride and offset stamped on the dma-buf.
//!
//! # Lifetime
//!
//! The exportable buffers form a ring of [`EXPORT_RING_SLOTS`], allocated and
//! exported once and copied into in turn, so a consumer that caches imports by
//! dma-buf (as [`DmaBufImporter`](crate::dmabufwgpu::DmaBufImporter) does) binds
//! each one once. Every frame carries its own dup of the slot's fd, an
//! independent reference to the buffer (standard dma-buf refcounting), which
//! stays valid after the element drops the ring. A consumer holding a frame
//! while [`EXPORT_RING_SLOTS`] newer ones are exported sees its pixels
//! overwritten.
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
//! the producer pipelines ahead. Leave the mode
//! off if the dma-buf may be consumed by something that does not honour the
//! semaphore. Cross-device / cross-process timeline share is validated on the RTX
//! 3060 (`dmabuf_timeline_probe`, `m562_dmabuf_semaphore_sync`).
//!
//! Hardware: needs a Vulkan device with `VK_KHR_external_memory_fd` +
//! `VK_EXT_external_memory_dma_buf` *export* support (validated on the RTX 3060 via
//! `dmabuf_export_probe`). CI-excluded like the rest of the GPU stack.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};

use ash::vk;

use g2g_core::log::{short_type_name, Target};
use g2g_core::memory::{
    DomainSet, MemoryDomain, MemoryDomainKind, OwnedDmaBuf, OwnedWgpuBuffer, SyncFd,
};
use g2g_core::meta::PlaneLayout;
use g2g_core::pad_template::{PadTemplate, PadTemplates};
use g2g_core::{
    g2g_error, AsyncElement, Caps, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    HardwareError, OutputSink, PipelinePacket, Rate, RawVideoFormat,
};

use crate::dmabufwgpu::{single_stride_layout, DmaBufWgpuBuffer, DMABUF_FRAME_FORMATS};
use crate::wgpubuffer::{whole_word_size, PlainWgpuBuffer};

fn gpu_err() -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

/// How many exportable buffers the element copies frames into in turn.
pub const EXPORT_RING_SLOTS: usize = 4;

/// One exportable buffer of the ring and the dma-buf fd exported from its
/// memory, which each frame gets a dup of.
#[derive(Debug)]
struct ExportSlot {
    buffer: wgpu::Buffer,
    fd: OwnedFd,
}

fn supported(format: RawVideoFormat) -> bool {
    DMABUF_FRAME_FORMATS.contains(&format)
}

/// Recover the input `wgpu::Buffer` and the device and queue it lives on from a
/// `WgpuBuffer` frame's keep-alive. Known owners: [`PlainWgpuBuffer`] and the
/// importer's [`DmaBufWgpuBuffer`].
fn producer_buffer(
    owned: &OwnedWgpuBuffer,
) -> Option<(&wgpu::Buffer, &wgpu::Device, &wgpu::Queue)> {
    let any = owned.keep_alive().as_any();
    if let Some(p) = any.downcast_ref::<PlainWgpuBuffer>() {
        return Some((p.buffer(), p.device(), p.queue()));
    }
    if let Some(d) = any.downcast_ref::<DmaBufWgpuBuffer>() {
        return Some((d.buffer(), d.device(), d.queue()));
    }
    None
}

fn raw_device(device: &wgpu::Device) -> Option<vk::Device> {
    // SAFETY: reads the device handle only, nothing is created or submitted.
    unsafe { device.as_hal::<wgpu_hal::api::Vulkan>() }.map(|hal| hal.raw_device().handle())
}

/// The first device extension an export on `device` needs that it lacks, or
/// `None` when it has them all. A non-Vulkan device lacks the first one.
fn missing_export_extension(
    device: &wgpu::Device,
    external_semaphore: bool,
) -> Option<&'static core::ffi::CStr> {
    let memory = [
        ash::khr::external_memory_fd::NAME,
        ash::ext::external_memory_dma_buf::NAME,
    ];
    let semaphore: &[&'static core::ffi::CStr] = match external_semaphore {
        true => &[ash::khr::external_semaphore_fd::NAME],
        false => &[],
    };
    // SAFETY: reads the enabled extension list only.
    let Some(hal) = (unsafe { device.as_hal::<wgpu_hal::api::Vulkan>() }) else {
        return Some(memory[0]);
    };
    let enabled = hal.enabled_device_extensions();
    memory
        .iter()
        .chain(semaphore)
        .copied()
        .find(|name| !enabled.contains(name))
}

/// GPU -> DMABUF export element. See the module docs.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::wgpudmabuf::WgpuToDmaBuf;
///
/// let export = WgpuToDmaBuf::new().with_external_semaphore(true);
/// ```
#[derive(Debug)]
pub struct WgpuToDmaBuf {
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    configured: bool,
    /// Pixel format and geometry from the negotiated caps: with the input's plane
    /// layout they give the exported planes ([`single_stride_layout`]).
    format: RawVideoFormat,
    width: u32,
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
    /// The exportable buffers, all `ring_size` bytes, and the slot the next
    /// frame is copied into.
    ring: alloc::vec::Vec<ExportSlot>,
    ring_size: u64,
    next_slot: usize,
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
            width: 0,
            height: 0,
            exported: 0,
            external_semaphore: false,
            semaphore: None,
            sync_fd: None,
            signal_value: 0,
            ring: alloc::vec::Vec::new(),
            ring_size: 0,
            next_slot: 0,
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

    /// Open a device carrying every export extension (the semaphore one too) if
    /// none is adopted yet, and return clones of it and its queue, so a producer
    /// (or a test) can allocate its input `wgpu::Buffer` on it. Frames from any
    /// other export-capable device work as well, as long as one device feeds the
    /// element.
    pub async fn gpu(&mut self) -> Result<(wgpu::Device, wgpu::Queue), G2gError> {
        if self.device.is_none() {
            let (device, queue) = create_export_device().await?;
            self.device = Some(device);
            self.queue = Some(queue);
        }
        Ok((self.device.clone().unwrap(), self.queue.clone().unwrap()))
    }

    /// Wrap a `wgpu::Buffer` (with `COPY_SRC` usage) and the device and queue it
    /// lives on as a `WgpuBuffer` frame domain this element accepts and a download
    /// can read back.
    pub fn wrap_buffer(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: wgpu::Buffer,
        len: usize,
    ) -> OwnedWgpuBuffer {
        crate::wgpubuffer::wrap_buffer(device, queue, buffer, len)
    }

    /// Export on `device` from now on, or fail when it cannot export or another
    /// device was adopted first: the timeline semaphore and the copies still in
    /// flight belong to that one.
    fn adopt_device(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(), G2gError> {
        if let Some(current) = &self.device {
            if raw_device(current) == raw_device(device) {
                return Ok(());
            }
            g2g_error!(
                Target::category(short_type_name::<Self>()),
                "a frame arrived from a second wgpu device, this element exports on the first one only"
            );
            return Err(G2gError::UnsupportedDomain);
        }
        if let Some(extension) = missing_export_extension(device, self.external_semaphore) {
            g2g_error!(
                Target::category(short_type_name::<Self>()),
                "the producer's wgpu device cannot export a dma-buf: {extension:?} is not enabled"
            );
            return Err(G2gError::UnsupportedDomain);
        }
        self.device = Some(device.clone());
        self.queue = Some(queue.clone());
        Ok(())
    }

    /// The plane-0 offset and stride to stamp on the dma-buf and the bytes the
    /// planes span: the input's own layout when it carries one, else tightly packed
    /// planes. A layout whose later planes do not follow plane 0's stride cannot be
    /// described by one dma-buf stride and offset, so it fails.
    fn export_layout(&self, owned: &OwnedWgpuBuffer) -> Result<(u32, u32, u64), G2gError> {
        let input = owned.plane_layout();
        let (offset, stride) = match input.and_then(|layout| layout.plane(0)) {
            Some(plane) => (plane.offset, plane.stride),
            None => (0, 0),
        };
        let to_u32 = |value: usize| u32::try_from(value).map_err(|_| G2gError::CapsMismatch);
        let (layout, size) = single_stride_layout(
            self.format,
            self.width,
            self.height,
            to_u32(offset)?,
            to_u32(stride)?,
        )
        .ok_or(G2gError::CapsMismatch)?;
        let follows_plane_0 = |input: &PlaneLayout| {
            input.count() == layout.count()
                && (0..layout.count()).all(|plane| layout.plane(plane) == input.plane(plane))
        };
        if !input.is_none_or(follows_plane_0) {
            return Err(G2gError::CapsMismatch);
        }
        let first_plane = layout.plane(0).ok_or(G2gError::CapsMismatch)?;
        Ok((
            to_u32(first_plane.offset)?,
            to_u32(first_plane.stride)?,
            size,
        ))
    }

    /// Create the persistent exportable timeline semaphore and export its fd, once.
    /// Requires the export device, from [`gpu`](Self::gpu) or the first frame.
    fn ensure_semaphore(&mut self) -> Result<(), G2gError> {
        if self.semaphore.is_some() {
            return Ok(());
        }
        let device = self.device.as_ref().ok_or_else(gpu_err)?;
        // SAFETY: `device` is a Vulkan device carrying VK_KHR_external_semaphore_fd
        // (added in `create_export_device`, checked in `adopt_device`).
        let (sem, fd) = unsafe { create_export_semaphore(device)? };
        self.semaphore = Some(sem);
        // SAFETY: `fd` is a fresh, owned OPAQUE_FD from the export above.
        self.sync_fd = Some(unsafe { SyncFd::from_raw(fd) });
        Ok(())
    }

    /// The ring slot the next frame of `size` bytes is copied into, allocated
    /// and exported on first use. A new size waits out the copies in flight and
    /// starts a new ring.
    fn next_export_slot(&mut self, size: u64) -> Result<&ExportSlot, G2gError> {
        let device = self.device.as_ref().ok_or_else(gpu_err)?;
        if size != self.ring_size {
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .map_err(|_| gpu_err())?;
            self.ring.clear();
            self.ring_size = size;
            self.next_slot = 0;
        }
        let index = self.next_slot;
        self.next_slot = (index + 1) % EXPORT_RING_SLOTS;
        if index == self.ring.len() {
            // SAFETY: `device` carries the dma-buf export extensions (checked in
            // `adopt_device`).
            let (buffer, fd) = unsafe { create_exportable(device, size)? };
            self.ring.push(ExportSlot { buffer, fd });
        }
        Ok(&self.ring[index])
    }
}

impl Drop for WgpuToDmaBuf {
    fn drop(&mut self) {
        let Some(device) = self.device.as_ref() else {
            return;
        };
        // Any in-flight copy must finish before we free its buffer and destroy the
        // semaphore it signals. A blocking poll is fine here (shutdown, not the hot
        // path). Then the ring drops (freeing its Vulkan handles; the exported
        // dma-buf fds keep the memory alive for consumers) and the timeline
        // semaphore is destroyed.
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        self.ring.clear();
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
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        let set = CapsSet::from_alternatives(DMABUF_FRAME_FORMATS.map(any).to_vec());
        alloc::vec![PadTemplate::sink(set.clone()), PadTemplate::source(set)]
    }
}

impl AsyncElement for WgpuToDmaBuf {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
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
            Caps::RawVideo {
                format,
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                ..
            } => (*format, *w, *h),
            _ => return Err(G2gError::CapsMismatch),
        };
        single_stride_layout(format, w, h, 0, 0).ok_or(G2gError::CapsMismatch)?;
        self.format = format;
        self.width = w;
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
                    let (src, device, queue) =
                        producer_buffer(owned).ok_or(G2gError::UnsupportedDomain)?;
                    self.adopt_device(device, queue)?;

                    let (offset, stride, frame_size) = self.export_layout(owned)?;
                    let size = whole_word_size(frame_size).ok_or(G2gError::CapsMismatch)?;
                    if frame_size == 0 || (owned.len as u64) < frame_size || src.size() < size {
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

                    let slot = self.next_export_slot(size)?;
                    let frame_fd = slot.fd.try_clone().map_err(|_| gpu_err())?;
                    let dst = slot.buffer.clone();
                    let device = self.device.as_ref().unwrap();
                    let queue = self.queue.as_ref().unwrap();
                    // SAFETY: `src` and `dst` are live buffers on `device` of at
                    // least `size` bytes; `sync`, when set, names this device's
                    // timeline semaphore.
                    unsafe { copy_to_export(device, queue, src, &dst, size, sync)? };

                    // SAFETY: `frame_fd` is a dup of the slot's dma-buf fd owned by
                    // this process; OwnedDmaBuf closes it once on drop.
                    let mut dmabuf =
                        unsafe { OwnedDmaBuf::from_raw(frame_fd.into_raw_fd(), stride, offset) };
                    if let (Some(sf), Some((_, value))) = (&self.sync_fd, sync) {
                        dmabuf = dmabuf.with_sync(sf.clone(), value);
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
        let hal = adapter
            .as_hal::<wgpu_hal::api::Vulkan>()
            .ok_or_else(gpu_err)?;
        hal.open_with_callback(
            wgpu::Features::empty(),
            &wgpu::Limits::default(),
            &wgpu::MemoryHints::default(),
            Some(Box::new(
                |args: wgpu_hal::vulkan::CreateDeviceCallbackArgs| {
                    args.extensions.push(ash::khr::external_memory_fd::NAME);
                    args.extensions
                        .push(ash::ext::external_memory_dma_buf::NAME);
                    // For the optional zero-stall timeline-semaphore export path.
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
                label: Some("wgputodmabuf"),
                ..Default::default()
            },
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
        let hal = device
            .as_hal::<wgpu_hal::api::Vulkan>()
            .ok_or_else(gpu_err)?;
        let raw = hal.raw_device();
        let instance = hal.shared_instance().raw_instance();

        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let mut export = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
        let info = vk::SemaphoreCreateInfo::default()
            .push_next(&mut type_info)
            .push_next(&mut export);
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
        (type_bits & (1 << i)) != 0
            && props.memory_types[i as usize]
                .property_flags
                .contains(flags)
    })
}

/// Allocate an exportable `size`-byte Vulkan buffer and export its backing
/// memory as a dma-buf fd. The exported fd is an independent reference to the
/// memory, so it stays valid after the buffer drops.
///
/// # Safety
/// `device` must be a Vulkan-backend wgpu device carrying
/// `VK_KHR_external_memory_fd` + `VK_EXT_external_memory_dma_buf`.
unsafe fn create_exportable(
    device: &wgpu::Device,
    size: u64,
) -> Result<(wgpu::Buffer, OwnedFd), G2gError> {
    // Create the exportable Vulkan buffer + dedicated exported memory, and export
    // its fd, all through the raw device.
    let (vk_buffer, vk_memory, fd) = {
        // SAFETY: caller guarantees a Vulkan export device; the hal guard is held
        // for the whole allocation and the raw handles are handed to wgpu below.
        unsafe {
            let hal = device
                .as_hal::<wgpu_hal::api::Vulkan>()
                .ok_or_else(gpu_err)?;
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
            let Some(mem_type) = find_memory_type(
                &props,
                reqs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ) else {
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

    // Wrap the exportable buffer as wgpu, which takes ownership of the raw
    // handles and frees them on drop.
    // SAFETY: `vk_buffer` is bound to `vk_memory` at offset 0 for `size` bytes and
    // we relinquish the raw handles to wgpu here.
    let hal_buffer =
        unsafe { wgpu_hal::vulkan::Buffer::from_raw_managed(vk_buffer, vk_memory, 0, size) };
    // SAFETY: `hal_buffer` was produced from this device's hal.
    let buffer = unsafe {
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
    Ok((buffer, owned_fd))
}

/// Copy `size` bytes of `src` into the exportable `dst` on the GPU.
///
/// Synchronisation depends on `sync`:
/// - `None`: block on `device.poll(Wait)` until the copy finishes, so a
///   consumer may read the dma-buf immediately.
/// - `Some((sem, value))`: inject a timeline signal of `value` on `sem` into the
///   copy submit and return without waiting. The consumer host-waits `value`
///   on the imported semaphore before reading.
///
/// # Safety
/// `device` must be a Vulkan-backend wgpu device (carrying
/// `VK_KHR_external_semaphore_fd` when `sync` is set), `src` and `dst` live
/// buffers on it of at least `size` bytes, and `sem` (when set) a timeline
/// semaphore on `device`.
unsafe fn copy_to_export(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    size: u64,
    sync: Option<(vk::Semaphore, u64)>,
) -> Result<(), G2gError> {
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc.copy_buffer_to_buffer(src, 0, dst, 0, size);
    match sync {
        Some((sem, value)) => {
            // Zero-stall: inject a timeline signal of `value` on the copy submit so
            // the consumer can order its read after the copy on the GPU timeline,
            // and return WITHOUT blocking.
            // SAFETY: `queue` is this device's live Vulkan queue; `sem` is a timeline
            // semaphore on the device, signalled by the submit below.
            let hal_q = unsafe { queue.as_hal::<wgpu_hal::api::Vulkan>() }.ok_or_else(gpu_err)?;
            hal_q.add_signal_semaphore(sem, Some(value));
            queue.submit([enc.finish()]);
        }
        None => {
            queue.submit([enc.finish()]);
            // Wait for the copy so a consumer of the dma-buf sees finished pixels.
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .map_err(|_| gpu_err())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dmabufwgpu::{create_import_device, DmaBufImporter};
    use crate::testutil::CollectSink;
    use g2g_core::frame::Frame;
    use g2g_core::FrameTiming;
    use std::os::fd::BorrowedFd;
    use std::os::unix::fs::MetadataExt;

    const SIDE: u32 = 4;
    const BYTES_PER_PIXEL: u32 = 4;

    fn inode(dmabuf: &OwnedDmaBuf) -> u64 {
        // SAFETY: the fd is owned by `dmabuf`, alive for this call.
        let fd = unsafe { BorrowedFd::borrow_raw(dmabuf.as_raw()) };
        std::fs::File::from(fd.try_clone_to_owned().unwrap())
            .metadata()
            .unwrap()
            .ino()
    }

    /// The ring hands its slots round, and an importer binds a returning slot
    /// from its cache.
    #[test]
    fn exports_cycle_the_ring_and_reimports_hit_the_cache() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime.block_on(async {
            let mut export = WgpuToDmaBuf::new();
            let Ok((device, queue)) = export.gpu().await else {
                std::eprintln!("skip: no Vulkan export device");
                return;
            };
            let size = (SIDE * SIDE * BYTES_PER_PIXEL) as usize;
            let src = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("export-src"),
                size: size as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let caps = Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(SIDE),
                height: Dim::Fixed(SIDE),
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            };
            export.configure_pipeline(&caps).unwrap();
            let mut sink = CollectSink::default();
            for _ in 0..=EXPORT_RING_SLOTS {
                let domain = MemoryDomain::WgpuBuffer(WgpuToDmaBuf::wrap_buffer(
                    &device,
                    &queue,
                    src.clone(),
                    size,
                ));
                let frame = Frame::new(domain, FrameTiming::default(), 0);
                export
                    .process(PipelinePacket::DataFrame(frame), &mut sink)
                    .await
                    .unwrap();
            }
            let dmabufs: alloc::vec::Vec<&OwnedDmaBuf> = sink
                .packets
                .iter()
                .filter_map(|packet| match packet {
                    PipelinePacket::DataFrame(Frame {
                        domain: MemoryDomain::DmaBuf(dmabuf),
                        ..
                    }) => Some(dmabuf),
                    _ => None,
                })
                .collect();
            let inodes: alloc::vec::Vec<u64> = dmabufs.iter().map(|d| inode(d)).collect();
            assert_eq!(
                inodes[0], inodes[EXPORT_RING_SLOTS],
                "the first slot comes round"
            );
            let mut ring = inodes[..EXPORT_RING_SLOTS].to_vec();
            ring.sort_unstable();
            ring.dedup();
            assert_eq!(ring.len(), EXPORT_RING_SLOTS, "each slot is its own buffer");

            let Ok((import_device, _)) = create_import_device().await else {
                std::eprintln!("skip: no Vulkan import device");
                return;
            };
            let mut importer = DmaBufImporter::new();
            let first = importer
                .import(&import_device, dmabufs[0], size as u64)
                .await;
            let Ok(first) = first else {
                std::eprintln!("skip: dma-buf import unsupported on this driver");
                return;
            };
            let again = importer
                .import(&import_device, dmabufs[EXPORT_RING_SLOTS], size as u64)
                .await
                .unwrap();
            assert!(
                first == again,
                "the returning slot is served from the cache"
            );
        });
    }
}
