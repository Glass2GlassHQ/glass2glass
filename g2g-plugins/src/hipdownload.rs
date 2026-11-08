use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;

use g2g_core::log::{short_type_name, Target};
use g2g_core::memory::{
    CudaKeepAlive, DomainSet, GpuVendor, MemoryDomainKind, OwnedCudaBuffer, OwnedHipBuffer,
    SystemSlice,
};
use g2g_core::{
    g2g_error, AllocationParams, AsyncElement, Caps, CapsConstraint, ConfigureOutcome,
    ElementMetadata, G2gError, MemoryDomain, OutputSink, PipelinePacket, RawVideoFormat,
};

use crate::hip::{HipDevice, HipStream};
use crate::pixel::{tight_frame, RawVideoGeometry};

const OUTPUT_DOMAINS: DomainSet =
    DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::Cuda);

#[derive(Debug)]
pub struct HipDownload {
    configured: bool,
    out_domain: MemoryDomainKind,
    raw_video: Option<RawVideoGeometry>,
    gpu: Option<(Arc<HipDevice>, HipStream)>,
}

impl Default for HipDownload {
    fn default() -> Self {
        Self {
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

    fn stream_for(&mut self, buffer: &OwnedHipBuffer) -> Result<&HipStream, G2gError> {
        let on_device = |device: &HipDevice| {
            device.vendor() == buffer.vendor && device.ordinal() == buffer.device_ordinal
        };
        let gpu = match self.gpu.take() {
            Some(gpu) if on_device(&gpu.0) => gpu,
            _ => {
                let device = HipDevice::open(buffer.vendor, buffer.device_ordinal)?;
                let stream = HipStream::new(&device)?;
                (device, stream)
            }
        };
        Ok(&self.gpu.insert(gpu).1)
    }

    fn download(&mut self, buffer: &OwnedHipBuffer) -> Result<MemoryDomain, G2gError> {
        let mut bytes = vec![0u8; buffer.len];
        self.stream_for(buffer)?
            .download_blocking(buffer, &mut bytes)?;
        let tight = tight_frame(bytes, buffer.plane_layout(), self.raw_video)?;
        Ok(MemoryDomain::System(SystemSlice::from_boxed(
            tight.into_boxed_slice(),
        )))
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
        let stream = self.stream_for(buffer)?;
        let context = stream
            .device()
            .cuda_context()
            .ok_or(G2gError::UnsupportedDomain)?;
        // CUDA consumers read a frame as soon as it arrives, without waiting on an event
        stream.wait_for(buffer)?;
        stream.synchronize()?;
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
                _frame: buffer.clone(),
            }),
        )))
    }
}

#[derive(Debug)]
struct HipFrame {
    _frame: OwnedHipBuffer,
}

impl CudaKeepAlive for HipFrame {}

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
        DomainSet::only(MemoryDomainKind::Hip)
    }

    fn output_memory(&self) -> MemoryDomainKind {
        self.out_domain
    }

    fn output_domains(&self) -> DomainSet {
        OUTPUT_DOMAINS
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
                    if let MemoryDomain::Hip(buffer) = &frame.domain {
                        frame.domain = match self.out_domain {
                            MemoryDomainKind::Cuda => self.as_cuda(buffer)?,
                            _ => self.download(buffer)?,
                        };
                    }
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
