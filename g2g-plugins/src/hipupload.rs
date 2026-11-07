use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::sync::Arc;

use g2g_core::log::{short_type_name, Target};
use g2g_core::memory::{
    DomainSet, HipVendor, MemoryDomainKind, OwnedCudaBuffer, OwnedDmaBuf, OwnedHipBuffer,
};
use g2g_core::meta::{Plane, PlaneLayout};
use g2g_core::{
    g2g_error, g2g_warn, AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata,
    G2gError, MemoryDomain, OutputSink, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec,
};

use crate::dmabufimport::DmaBufImportCache;
use crate::dmabufmap::{dmabuf_len, DmaBufReadMap};
use crate::hip::{
    hip_buffer, HipDevice, HipEvent, HipExternalBuffer, HipMemoryPool, HipPinnedMemory, HipStream,
};
use crate::paddedrows::single_stride_layout;
use crate::pixel::RawVideoGeometry;

#[derive(Debug)]
struct Staging {
    memory: HipPinnedMemory,
    copied: HipEvent,
}

impl Drop for Staging {
    fn drop(&mut self) {
        // hipHostFree and cuMemFreeHost do not wait for a copy still reading the memory
        let _ = self.copied.synchronize();
    }
}

#[derive(Debug)]
struct UploadGpu {
    device: Arc<HipDevice>,
    stream: HipStream,
    pool: Option<HipMemoryPool>,
    staging: Option<Staging>,
    dmabuf_imports: DmaBufImportCache<Arc<HipExternalBuffer>>,
    dmabuf_import_failed: bool,
}

#[derive(Debug, Default)]
pub struct HipUpload {
    configured: bool,
    raw_video: Option<RawVideoGeometry>,
    device_id: Option<i32>,
    vendor: Option<HipVendor>,
    gpu: Option<UploadGpu>,
}

impl HipUpload {
    pub fn new() -> Self {
        Self::default()
    }

    fn gpu_for(&mut self, cuda_ordinal: Option<i32>) -> Result<&mut UploadGpu, G2gError> {
        if let Some(cuda_ordinal) = cuda_ordinal {
            let wrong_vendor = self.vendor == Some(HipVendor::Amd);
            let wrong_ordinal = self
                .device_id
                .is_some_and(|ordinal| ordinal != cuda_ordinal);
            let wrong_open_device = self.gpu.as_ref().is_some_and(|gpu| {
                gpu.device.vendor() != HipVendor::Nvidia || gpu.device.ordinal() != cuda_ordinal
            });
            if wrong_vendor || wrong_ordinal || wrong_open_device {
                g2g_error!(
                    Target::category(short_type_name::<Self>()),
                    "a CUDA frame from device {} becomes a Hip frame only on that NVIDIA device, and this element is set to vendor {:?} device {:?}",
                    cuda_ordinal,
                    self.vendor,
                    self.device_id
                );
                return Err(G2gError::UnsupportedDomain);
            }
        }
        let gpu = match self.gpu.take() {
            Some(gpu) => gpu,
            None => {
                let device = match cuda_ordinal {
                    Some(ordinal) => HipDevice::open(HipVendor::Nvidia, ordinal)?,
                    None => open_device(self.vendor, self.device_id.unwrap_or(0))?,
                };
                UploadGpu {
                    stream: HipStream::new(&device)?,
                    device,
                    pool: None,
                    staging: None,
                    dmabuf_imports: DmaBufImportCache::new(),
                    dmabuf_import_failed: false,
                }
            }
        };
        Ok(self.gpu.insert(gpu))
    }
}

fn open_device(vendor: Option<HipVendor>, ordinal: i32) -> Result<Arc<HipDevice>, G2gError> {
    match vendor {
        Some(vendor) => HipDevice::open(vendor, ordinal),
        None => HipDevice::open(HipVendor::Amd, ordinal)
            .or_else(|_| HipDevice::open(HipVendor::Nvidia, ordinal)),
    }
}

fn upload(gpu: &mut UploadGpu, bytes: &[u8]) -> Result<OwnedHipBuffer, G2gError> {
    let len = bytes.len();
    if gpu.pool.as_ref().is_none_or(|pool| pool.size() != len) {
        gpu.pool = Some(HipMemoryPool::new(&gpu.device, len));
    }
    let mut staging = match gpu.staging.take() {
        Some(staging) if staging.memory.size() >= len => staging,
        _ => Staging {
            memory: HipPinnedMemory::allocate(&gpu.device, len)?,
            copied: HipEvent::new(&gpu.device)?,
        },
    };
    staging.copied.synchronize()?;
    staging.memory.as_mut_slice()[..len].copy_from_slice(bytes);
    let memory = gpu
        .pool
        .as_ref()
        .ok_or(G2gError::NotConfigured)?
        .acquire()?;
    gpu.stream.upload_pinned(&memory, &staging.memory, len)?;
    gpu.stream.record(&staging.copied)?;
    gpu.staging = Some(staging);
    let ready = HipEvent::new(&gpu.device)?;
    gpu.stream.record(&ready)?;
    memory.into_buffer(ready, len)
}

fn wrap_cuda(gpu: &UploadGpu, cuda: &OwnedCudaBuffer) -> Result<OwnedHipBuffer, G2gError> {
    let chroma_offset = cuda
        .chroma_ptr
        .checked_sub(cuda.luma_ptr)
        .ok_or(G2gError::UnsupportedDomain)? as usize;
    let chroma_rows = cuda.height.div_ceil(2) as usize;
    let len = chroma_offset + cuda.chroma_pitch as usize * chroma_rows;
    let layout = PlaneLayout::new(&[
        Plane {
            offset: 0,
            stride: cuda.luma_pitch as usize,
        },
        Plane {
            offset: chroma_offset,
            stride: cuda.chroma_pitch as usize,
        },
    ])
    .ok_or(G2gError::UnsupportedDomain)?;
    let ready = HipEvent::new(&gpu.device)?;
    gpu.stream.record(&ready)?;
    // kernels in the primary context read allocations made in other contexts on the same device
    let buffer = hip_buffer(
        &gpu.device,
        cuda.luma_ptr,
        len,
        ready,
        Box::new(cuda.clone()),
    )?;
    Ok(buffer.with_plane_layout(layout))
}

fn upload_dmabuf(
    gpu: &mut UploadGpu,
    dmabuf: &OwnedDmaBuf,
    raw_video: RawVideoGeometry,
) -> Result<OwnedHipBuffer, G2gError> {
    let RawVideoGeometry {
        format,
        width,
        height,
    } = raw_video;
    let tag = Target::category(short_type_name::<HipUpload>());
    if dmabuf.sync_fd().is_some() {
        g2g_error!(
            tag,
            "the dma-buf comes with a GPU semaphore this element cannot wait on, turn off the producer's zero-stall sync"
        );
        return Err(G2gError::UnsupportedDomain);
    }
    let (layout, size) = single_stride_layout(format, width, height, dmabuf.offset, dmabuf.stride)
        .ok_or(G2gError::CapsMismatch)?;
    // hip imports dma-bufs only on amd
    if gpu.device.vendor() == HipVendor::Amd && !gpu.dmabuf_import_failed {
        match import_dmabuf(gpu, dmabuf, size) {
            Ok(buffer) => return Ok(buffer.with_plane_layout(layout)),
            Err(error) => {
                g2g_warn!(
                    tag,
                    "importing a dma-buf into HIP failed ({error:?}), copying frames through the CPU"
                );
                gpu.dmabuf_import_failed = true;
            }
        }
    }
    let map = DmaBufReadMap::read(dmabuf)?;
    let (layout, size) = single_stride_layout(format, width, height, 0, dmabuf.stride)
        .ok_or(G2gError::CapsMismatch)?;
    let bytes = usize::try_from(size)
        .ok()
        .and_then(|size| map.as_slice().get(..size))
        .ok_or(G2gError::CapsMismatch)?;
    Ok(upload(gpu, bytes)?.with_plane_layout(layout))
}

fn import_dmabuf(
    gpu: &mut UploadGpu,
    dmabuf: &OwnedDmaBuf,
    size: u64,
) -> Result<OwnedHipBuffer, G2gError> {
    if u64::try_from(dmabuf_len(dmabuf)?).map_err(|_| G2gError::CapsMismatch)? < size {
        return Err(G2gError::CapsMismatch);
    }
    let device = &gpu.device;
    let imported = gpu.dmabuf_imports.get_or_import(dmabuf, size, || {
        // SAFETY: the fd is a live dma-buf owned by `dmabuf` for this call.
        let fd = unsafe { std::os::fd::BorrowedFd::borrow_raw(dmabuf.as_raw()) }
            .try_clone_to_owned()
            .map_err(|_| G2gError::Hardware(g2g_core::HardwareError::Other))?;
        // SAFETY: `fd` is a dma-buf of at least `size` bytes, checked above, on an AMD device.
        unsafe { HipExternalBuffer::import(fd, size, device.ordinal()) }.map(Arc::new)
    })?;
    let ready = HipEvent::new(device)?;
    gpu.stream.record(&ready)?;
    let len = usize::try_from(size).map_err(|_| G2gError::CapsMismatch)?;
    // the frame keeps the producer from recycling the dma-buf while the import is read
    hip_buffer(
        device,
        imported.device_pointer(),
        len,
        ready,
        Box::new((imported, dmabuf.clone())),
    )
}

static HIP_UPLOAD_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "device-id",
        PropKind::Int,
        "HIP device ordinal, -1 for device 0 or the ordinal of an incoming CUDA frame",
    )
    .with_default("-1")
    .with_range("-1", "2147483647"),
    PropertySpec::new(
        "vendor",
        PropKind::Str,
        "GPU vendor of the HIP device, unknown tries AMD then NVIDIA",
    )
    .with_default("unknown")
    .with_enum_values("unknown | amd | nvidia"),
];

impl AsyncElement for HipUpload {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "HIP upload",
            "Filter/Converter/Video/GPU",
            "Copies a system frame into HIP device memory, imports a linear dma-buf on AMD, or wraps a CUDA frame on NVIDIA without a copy",
            "g2g",
        )
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
            .with(MemoryDomainKind::Cuda)
            .with(MemoryDomainKind::DmaBuf)
            .with(MemoryDomainKind::Hip)
    }

    fn output_memory(&self) -> MemoryDomainKind {
        MemoryDomainKind::Hip
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.raw_video = RawVideoGeometry::of(absolute_caps);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        HIP_UPLOAD_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if self.gpu.is_some() {
            return Err(PropError::ReadOnly);
        }
        match name {
            "device-id" => {
                let device_id = value.as_int().ok_or(PropError::Type)?;
                self.device_id = match device_id {
                    -1 => None,
                    0..=0x7fff_ffff => Some(device_id as i32),
                    _ => return Err(PropError::Value),
                };
            }
            "vendor" => {
                self.vendor = match value.as_str().ok_or(PropError::Type)? {
                    "unknown" => None,
                    "amd" => Some(HipVendor::Amd),
                    "nvidia" => Some(HipVendor::Nvidia),
                    _ => return Err(PropError::Value),
                };
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "device-id" => Some(PropValue::Int(self.device_id.map_or(-1, i64::from))),
            "vendor" => Some(PropValue::Str(
                match self.vendor {
                    None => "unknown",
                    Some(HipVendor::Amd) => "amd",
                    Some(HipVendor::Nvidia) => "nvidia",
                }
                .into(),
            )),
            _ => None,
        }
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
                PipelinePacket::DataFrame(mut frame) => {
                    let uploaded = match &frame.domain {
                        MemoryDomain::Hip(_) => None,
                        MemoryDomain::Cuda(cuda) => {
                            let gpu = self.gpu_for(Some(cuda.device_ordinal))?;
                            Some(wrap_cuda(gpu, cuda)?)
                        }
                        MemoryDomain::DmaBuf(dmabuf) => {
                            let raw_video = self.raw_video.ok_or(G2gError::CapsMismatch)?;
                            let gpu = self.gpu_for(None)?;
                            Some(upload_dmabuf(gpu, dmabuf, raw_video)?)
                        }
                        domain => {
                            let bytes = domain.require_system_slice(short_type_name::<Self>())?;
                            let gpu = self.gpu_for(None)?;
                            Some(upload(gpu, bytes)?)
                        }
                    };
                    if let Some(buffer) = uploaded {
                        frame.domain = MemoryDomain::Hip(buffer);
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                // the runner forwards end-of-stream after `process` returns
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}
