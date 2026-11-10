// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::sync::Arc;
use std::sync::OnceLock;

use g2g_core::log::{short_type_name, Target};
use g2g_core::memory::{
    DomainSet, GpuDevice, GpuVendor, MemoryDomainKind, OwnedCudaBuffer, OwnedDmaBuf, OwnedHipBuffer,
};
use g2g_core::meta::{Plane, PlaneLayout};
use g2g_core::{
    g2g_error, g2g_warn, AllocationParams, AsyncElement, Caps, CapsConstraint, ConfigureOutcome,
    ElementMetadata, Frame, G2gError, MemoryDomain, OutputSink, PipelinePacket, PropError,
    PropValue, PropertySpec, SystemSlice,
};

use crate::dmabuffence::wait_for_producer;
use crate::dmabufimport::DmaBufImportCache;
use crate::dmabufmap::{dmabuf_len, DmaBufReadMap};
use crate::hip::{
    hip_buffer, release_after, CrossDeviceCopier, DeviceStream, DeviceStreamCache, HipDevice,
    HipEvent, HipExternalBuffer, HipMemoryPool, HipPinnedMemory, HipStream,
};
use crate::hipdeviceid::{HipDeviceChoice, HIP_DEVICE_ID_PROP, HIP_VENDOR_PROP};
use crate::paddedrows::single_stride_layout;
use crate::pixel::RawVideoGeometry;

type PinnedFrames = crate::pinnedframes::PinnedFrames<HipPinnedMemory>;
type PinnedOffer = crate::pinnedframes::PinnedOffer<HipPinnedMemory>;

#[derive(Debug)]
struct UploadGpu {
    device: Arc<HipDevice>,
    stream: HipStream,
    pool: Option<HipMemoryPool>,
    dmabuf_imports: DmaBufImportCache<Arc<HipExternalBuffer>>,
    dmabuf_import_failed: bool,
}

#[derive(Debug, Default)]
pub struct HipUpload {
    configured: bool,
    raw_video: Option<RawVideoGeometry>,
    device: HipDeviceChoice,
    pinned: OnceLock<PinnedOffer>,
    gpu: Option<UploadGpu>,
    sources: DeviceStreamCache,
    copier: CrossDeviceCopier,
}

impl HipUpload {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn on_device(device: GpuDevice) -> Self {
        Self {
            device: HipDeviceChoice::of(device),
            ..Self::default()
        }
    }

    fn pinned_offer(&self) -> Result<&PinnedOffer, G2gError> {
        if let Some(offer) = self.pinned.get() {
            return Ok(offer);
        }
        let offer = PinnedOffer::new(self.device.open()?);
        Ok(self.pinned.get_or_init(|| offer))
    }

    fn gpu(&mut self) -> Result<&mut UploadGpu, G2gError> {
        let gpu = match self.gpu.take() {
            Some(gpu) => gpu,
            None => {
                let device = Arc::clone(&self.pinned_offer()?.frames.device);
                UploadGpu {
                    stream: HipStream::new(&device)?,
                    device,
                    pool: None,
                    dmabuf_imports: DmaBufImportCache::new(),
                    dmabuf_import_failed: false,
                }
            }
        };
        Ok(self.gpu.insert(gpu))
    }

    async fn onto_chosen_device(
        &mut self,
        buffer: OwnedHipBuffer,
    ) -> Result<OwnedHipBuffer, G2gError> {
        if !self.device.is_set() {
            return Ok(buffer);
        }
        let device = &self.gpu()?.device;
        let target = GpuDevice {
            vendor: device.vendor(),
            ordinal: device.ordinal(),
        };
        if (buffer.vendor, buffer.device_ordinal) == (target.vendor, target.ordinal) {
            return Ok(buffer);
        }
        self.copier.copy(buffer, target).await
    }
}

fn upload_pinned(gpu: &mut UploadGpu, pinned: SystemSlice) -> Result<OwnedHipBuffer, G2gError> {
    let len = pinned.as_slice().len();
    if gpu.pool.as_ref().is_none_or(|pool| pool.size() != len) {
        gpu.pool = Some(HipMemoryPool::new(&gpu.device, len));
    }
    let memory = gpu
        .pool
        .as_ref()
        .ok_or(G2gError::NotConfigured)?
        .acquire()?;
    // SAFETY: `release_after` below keeps `pinned` alive until the stream passes the copy.
    unsafe { gpu.stream.upload_async(&memory, pinned.as_slice()) }?;
    let copied = HipEvent::new(&gpu.device)?;
    gpu.stream.record(&copied)?;
    release_after(copied, Box::new(pinned));
    let ready = HipEvent::new(&gpu.device)?;
    gpu.stream.record(&ready)?;
    memory.into_buffer(ready, len)
}

fn upload(
    gpu: &mut UploadGpu,
    frames: &Arc<PinnedFrames>,
    bytes: &[u8],
) -> Result<OwnedHipBuffer, G2gError> {
    let mut staging = frames.lease(bytes.len())?;
    staging.as_mut_slice().copy_from_slice(bytes);
    upload_pinned(gpu, staging)
}

fn wrap_cuda(gpu: &DeviceStream, cuda: &OwnedCudaBuffer) -> Result<OwnedHipBuffer, G2gError> {
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
    gpu.stream.wait_for_cuda(cuda)?;
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
    frames: &Arc<PinnedFrames>,
    dmabuf: &OwnedDmaBuf,
    raw_video: RawVideoGeometry,
) -> Result<OwnedHipBuffer, G2gError> {
    let RawVideoGeometry {
        format,
        width,
        height,
    } = raw_video;
    let tag = Target::category(short_type_name::<HipUpload>());
    let (layout, size) = single_stride_layout(format, width, height, dmabuf.offset, dmabuf.stride)
        .ok_or(G2gError::CapsMismatch)?;
    // hip imports dma-bufs only on amd
    if gpu.device.vendor() == GpuVendor::Amd && !gpu.dmabuf_import_failed {
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
    Ok(upload(gpu, frames, bytes)?.with_plane_layout(layout))
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

static HIP_UPLOAD_PROPS: &[PropertySpec] = &[HIP_DEVICE_ID_PROP, HIP_VENDOR_PROP];

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
        DomainSet::only(MemoryDomainKind::System)
            .with(MemoryDomainKind::Cuda)
            .with(MemoryDomainKind::DmaBuf)
            .with(MemoryDomainKind::Hip)
    }

    fn output_memory(&self) -> MemoryDomainKind {
        MemoryDomainKind::Hip
    }

    fn gpu_device(&self) -> Option<GpuDevice> {
        self.device.declared()
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.raw_video = RawVideoGeometry::of(absolute_caps);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        HIP_UPLOAD_PROPS
    }

    fn propose_allocation(&self, _caps: &Caps) -> Option<AllocationParams> {
        let offer = match self.pinned_offer() {
            Ok(offer) => offer,
            Err(error) => {
                g2g_warn!(
                    Target::category(short_type_name::<Self>()),
                    "no pinned host buffers to offer upstream, opening the HIP device failed: {error:?}"
                );
                return None;
            }
        };
        let accepts = self.input_domains();
        Some(AllocationParams {
            domain: accepts.preferred()?,
            accepts,
            host_allocator: Some(offer.allocator.clone()),
            ..AllocationParams::default()
        })
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if self.gpu.is_some() || self.pinned.get().is_some() {
            return Err(PropError::ReadOnly);
        }
        self.device
            .set_property(name, &value)
            .unwrap_or(Err(PropError::Unknown))
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        self.device.get_property(name)
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
                    let Frame {
                        domain,
                        timing,
                        sequence,
                        meta,
                    } = frame;
                    let buffer = match domain {
                        MemoryDomain::Hip(buffer) => self.onto_chosen_device(buffer).await?,
                        MemoryDomain::Cuda(cuda) => {
                            let source = self.sources.on(GpuVendor::Nvidia, cuda.device_ordinal)?;
                            let wrapped = wrap_cuda(source, &cuda)?;
                            self.onto_chosen_device(wrapped).await?
                        }
                        MemoryDomain::DmaBuf(dmabuf) => {
                            wait_for_producer(&dmabuf).await.inspect_err(|_| {
                                g2g_error!(
                                    Target::category(short_type_name::<Self>()),
                                    "the dma-buf carries only a Vulkan timeline semaphore, which HIP cannot wait on"
                                );
                            })?;
                            let raw_video = self.raw_video.ok_or(G2gError::CapsMismatch)?;
                            let frames = Arc::clone(&self.pinned_offer()?.frames);
                            upload_dmabuf(self.gpu()?, &frames, &dmabuf, raw_video)?
                        }
                        MemoryDomain::System(slice)
                            if self.pinned_offer()?.frames.allocated(slice.as_slice()) =>
                        {
                            upload_pinned(self.gpu()?, slice)?
                        }
                        domain => {
                            let frames = Arc::clone(&self.pinned_offer()?.frames);
                            let bytes = domain.require_system_slice(short_type_name::<Self>())?;
                            upload(self.gpu()?, &frames, bytes)?
                        }
                    };
                    out.push(PipelinePacket::DataFrame(Frame {
                        domain: MemoryDomain::Hip(buffer),
                        timing,
                        sequence,
                        meta,
                    }))
                    .await?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use g2g_core::{FrameTiming, PushOutcome, RawVideoFormat};

    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 48;

    #[derive(Default)]
    struct FrameSink {
        frames: Vec<Frame>,
    }

    impl OutputSink for FrameSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            if let Some(PipelinePacket::DataFrame(frame)) = packet.take() {
                self.frames.push(frame);
            }
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    fn upload_and_read_back(upload: &mut HipUpload, source: SystemSlice) -> Vec<u8> {
        let mut sink = FrameSink::default();
        let frame = Frame::new(MemoryDomain::System(source), FrameTiming::default(), 0);
        g2g_core::runtime::block_on(upload.process(PipelinePacket::DataFrame(frame), &mut sink))
            .unwrap();
        let [Frame {
            domain: MemoryDomain::Hip(buffer),
            ..
        }] = <[Frame; 1]>::try_from(sink.frames).unwrap()
        else {
            panic!("hipupload emits one Hip frame");
        };
        let stream = HipStream::new(&upload.pinned.get().unwrap().frames.device).unwrap();
        let mut read_back = alloc::vec![0; buffer.len];
        stream.download_blocking(&buffer, &mut read_back).unwrap();
        read_back
    }

    fn pinned_allocations(upload: &HipUpload) -> usize {
        let frames = &upload.pinned.get().unwrap().frames;
        frames.allocations.lock().unwrap().len()
    }

    #[test]
    fn a_frame_in_offered_pinned_memory_uploads_without_a_staging_copy() {
        let caps = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: g2g_core::Dim::Fixed(WIDTH),
            height: g2g_core::Dim::Fixed(HEIGHT),
            framerate: g2g_core::Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
        };
        let len = RawVideoFormat::Nv12
            .unpadded_frame_bytes(WIDTH, HEIGHT)
            .unwrap() as usize;
        let bytes: Vec<u8> = (0..len).map(|index| (index * 7) as u8).collect();
        for (vendor, name) in [(GpuVendor::Amd, "amd"), (GpuVendor::Nvidia, "nvidia")] {
            if let Err(error) = HipDevice::open(vendor, 0) {
                std::eprintln!("skipping: no {vendor:?} HIP device 0: {error:?}");
                continue;
            }
            let mut upload = HipUpload::new();
            upload
                .set_property("vendor", PropValue::Str(name.into()))
                .unwrap();
            upload.configure_pipeline(&caps).unwrap();
            let offered = upload
                .propose_allocation(&caps)
                .and_then(|params| params.host_allocator)
                .expect("hipupload offers pinned buffers upstream");

            let mut pinned = offered.allocate(len).unwrap();
            pinned.as_mut_slice().copy_from_slice(&bytes);
            assert_eq!(
                upload_and_read_back(&mut upload, pinned),
                bytes,
                "{vendor:?}"
            );
            assert_eq!(
                pinned_allocations(&upload),
                1,
                "{vendor:?}: no staging buffer"
            );
        }
    }
}
