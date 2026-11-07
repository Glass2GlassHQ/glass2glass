use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::sync::Arc;

use g2g_core::log::{short_type_name, Target};
use g2g_core::memory::{DomainSet, HipVendor, MemoryDomainKind, OwnedCudaBuffer, OwnedHipBuffer};
use g2g_core::meta::{Plane, PlaneLayout};
use g2g_core::{
    g2g_error, AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError,
    MemoryDomain, OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::hip::{hip_buffer, HipDevice, HipEvent, HipMemoryPool, HipPinnedMemory, HipStream};

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
}

#[derive(Debug, Default)]
pub struct HipUpload {
    configured: bool,
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
            "Copies a system frame into HIP device memory, or wraps a CUDA frame on NVIDIA without a copy",
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
            .with(MemoryDomainKind::Hip)
    }

    fn output_memory(&self) -> MemoryDomainKind {
        MemoryDomainKind::Hip
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
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
