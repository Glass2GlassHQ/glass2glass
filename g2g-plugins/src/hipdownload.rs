use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::sync::Arc;

use g2g_core::log::{short_type_name, Target};
use g2g_core::memory::{
    CudaKeepAlive, DomainSet, GpuDevice, GpuVendor, MemoryDomainKind, OwnedCudaBuffer,
    OwnedHipBuffer, SystemSlice,
};
use g2g_core::{
    g2g_error, AllocationParams, AsyncElement, Caps, CapsConstraint, ConfigureOutcome,
    ElementMetadata, G2gError, MemoryDomain, OutputSink, PipelinePacket, PropError, PropValue,
    PropertySpec, RawVideoFormat,
};

use crate::hip::{HipDevice, HipEvent, HipPinnedMemory, HipStream};
use crate::hipdeviceid::{HipDeviceChoice, HIP_DEVICE_ID_PROP, HIP_VENDOR_PROP};
use crate::pinnedframes::{PinnedFrames, PinnedOffer};
use crate::pixel::{pack_planes, RawVideoGeometry};
use g2g_core::offload::run_blocking;

const OUTPUT_DOMAINS: DomainSet =
    DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::Cuda);

#[derive(Debug)]
struct DownloadGpu {
    device: Arc<HipDevice>,
    stream: HipStream,
    pinned: Arc<PinnedFrames<HipPinnedMemory>>,
}

static HIP_DOWNLOAD_PROPS: &[PropertySpec] = &[HIP_DEVICE_ID_PROP, HIP_VENDOR_PROP];

#[derive(Debug)]
pub struct HipDownload {
    device: HipDeviceChoice,
    configured: bool,
    out_domain: MemoryDomainKind,
    raw_video: Option<RawVideoGeometry>,
    gpu: Option<DownloadGpu>,
}

impl Default for HipDownload {
    fn default() -> Self {
        Self {
            device: HipDeviceChoice::default(),
            configured: false,
            out_domain: MemoryDomainKind::System,
            raw_video: None,
            gpu: None,
        }
    }
}

impl HipDownload {
    pub fn new() -> Self {
        Self::default()
    }

    fn gpu_for(&mut self, buffer: &OwnedHipBuffer) -> Result<&DownloadGpu, G2gError> {
        if !self.device.holds(buffer) {
            g2g_error!(
                Target::category(short_type_name::<Self>()),
                "runs on {:?}, and a frame arrived on {:?} device {}",
                self.device,
                buffer.vendor,
                buffer.device_ordinal
            );
            return Err(G2gError::UnsupportedDomain);
        }
        let on_device = |device: &HipDevice| {
            device.vendor() == buffer.vendor && device.ordinal() == buffer.device_ordinal
        };
        let gpu = match self.gpu.take() {
            Some(gpu) if on_device(&gpu.device) => gpu,
            _ => {
                let device = HipDevice::open(buffer.vendor, buffer.device_ordinal)?;
                DownloadGpu {
                    stream: HipStream::new(&device)?,
                    pinned: PinnedOffer::new(Arc::clone(&device)).frames,
                    device,
                }
            }
        };
        Ok(self.gpu.insert(gpu))
    }

    async fn download(&mut self, buffer: OwnedHipBuffer) -> Result<MemoryDomain, G2gError> {
        let layout = buffer.plane_layout().copied();
        let gpu = self.gpu_for(&buffer)?;
        let mut output = gpu.pinned.lease(buffer.len)?;
        // SAFETY: the stream is synchronized below, or the blocking wait holds `output` and `buffer` until the copy ends.
        let queued = unsafe { gpu.stream.download_async(&buffer, output.as_mut_slice()) };
        let copied = queued.and_then(|()| {
            let copied = HipEvent::new(&gpu.device)?;
            gpu.stream.record(&copied)?;
            Ok(copied)
        });
        let copied = match copied {
            Ok(copied) => copied,
            Err(error) => {
                // a copy queued before the failure still writes the leased slice
                let _ = gpu.stream.synchronize();
                return Err(error);
            }
        };
        let output = run_blocking(move || {
            let _source = buffer;
            copied.synchronize().map(|()| output)
        })
        .await?;
        let Some(layout) = layout else {
            return Ok(MemoryDomain::System(output));
        };
        let raw_video = self.raw_video.ok_or(G2gError::CapsMismatch)?;
        let packed = pack_planes(
            output.as_slice(),
            raw_video.format,
            raw_video.width as usize,
            raw_video.height as usize,
            &layout,
        )
        .ok_or(G2gError::CapsMismatch)?;
        Ok(MemoryDomain::System(SystemSlice::from_boxed(packed)))
    }

    fn as_cuda(&mut self, buffer: &OwnedHipBuffer) -> Result<MemoryDomain, G2gError> {
        let category = Target::category(short_type_name::<Self>());
        if buffer.vendor != GpuVendor::Nvidia {
            g2g_error!(
                category,
                "only a Hip frame on an NVIDIA device can become a CUDA frame, this one is on {:?}",
                buffer.vendor
            );
            return Err(G2gError::UnsupportedDomain);
        }
        let raw_video = self.raw_video.ok_or(G2gError::CapsMismatch)?;
        let (luma, chroma) = match buffer.plane_layout() {
            Some(layout) => (
                layout.plane(0).ok_or(G2gError::CapsMismatch)?,
                layout.plane(1).ok_or(G2gError::CapsMismatch)?,
            ),
            None => {
                let pitch = match raw_video.format {
                    RawVideoFormat::Nv12 => raw_video.width as usize,
                    RawVideoFormat::P010 => raw_video.width as usize * 2,
                    format => {
                        g2g_error!(
                            category,
                            "a CUDA frame holds NV12 or P010, not {:?}",
                            format
                        );
                        return Err(G2gError::CapsMismatch);
                    }
                };
                (
                    g2g_core::meta::Plane {
                        offset: 0,
                        stride: pitch,
                    },
                    g2g_core::meta::Plane {
                        offset: pitch * raw_video.height as usize,
                        stride: pitch,
                    },
                )
            }
        };
        let device = Arc::clone(&self.gpu_for(buffer)?.device);
        let context = device.cuda_context().ok_or(G2gError::UnsupportedDomain)?;
        let pitch = |stride: usize| u32::try_from(stride).map_err(|_| G2gError::CapsMismatch);
        Ok(MemoryDomain::Cuda(OwnedCudaBuffer::new(
            buffer.device_pointer + luma.offset as u64,
            buffer.device_pointer + chroma.offset as u64,
            pitch(luma.stride)?,
            pitch(chroma.stride)?,
            raw_video.width,
            raw_video.height,
            context,
            buffer.device_ordinal,
            Arc::new(HipFrame {
                frame: buffer.clone(),
                device,
            }),
        )))
    }
}

#[derive(Debug)]
struct HipFrame {
    frame: OwnedHipBuffer,
    device: Arc<HipDevice>,
}

impl CudaKeepAlive for HipFrame {
    fn ready_event(&self) -> Option<u64> {
        // hip events on nvidia are CUevents
        Some(self.frame.ready_event)
    }

    fn wait_until_ready(&self) -> Result<(), G2gError> {
        self.device.wait_until_ready(&self.frame)
    }
}

impl AsyncElement for HipDownload {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "HIP download",
            "Filter/Converter/Video/GPU",
            "Copies a HIP frame to system memory, or hands it to CUDA on NVIDIA without a copy",
            crate::AARON_BOXER,
        )
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::Hip)
    }

    fn output_memory(&self) -> MemoryDomainKind {
        self.out_domain
    }

    fn output_domains(&self) -> DomainSet {
        OUTPUT_DOMAINS
    }

    fn gpu_device(&self) -> Option<GpuDevice> {
        self.device.declared()
    }

    fn properties(&self) -> &'static [PropertySpec] {
        HIP_DOWNLOAD_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let applied = self
            .device
            .set_property(name, &value)
            .ok_or(PropError::Unknown)?;
        self.gpu = None;
        applied
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        self.device.get_property(name)
    }

    fn configure_allocation(&mut self, params: &AllocationParams) {
        let cuda_only = params.accepts.contains(MemoryDomainKind::Cuda)
            && !params.accepts.contains(MemoryDomainKind::System);
        self.out_domain = if cuda_only {
            MemoryDomainKind::Cuda
        } else {
            MemoryDomainKind::System
        };
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.raw_video = RawVideoGeometry::of(absolute_caps);
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
                PipelinePacket::DataFrame(mut frame) => {
                    frame.domain = match frame.domain {
                        MemoryDomain::Hip(buffer) => match self.out_domain {
                            MemoryDomainKind::Cuda => self.as_cuda(&buffer)?,
                            _ => self.download(buffer).await?,
                        },
                        domain => domain,
                    };
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(caps) => {
                    self.raw_video = RawVideoGeometry::of(&caps);
                    out.push(PipelinePacket::CapsChanged(caps)).await?;
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
