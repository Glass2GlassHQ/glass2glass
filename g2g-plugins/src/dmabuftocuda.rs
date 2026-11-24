// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ffi::c_void;
use core::future::Future;
use core::pin::Pin;

use g2g_core::log::{short_type_name, Target};
use g2g_core::memory::{DomainSet, MemoryDomainKind, OwnedCudaBuffer, OwnedDmaBuf};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    HardwareError, Interlace, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PixelAspect, PropError, PropValue, PropertySpec, Rate, RawVideoFormat,
};
use khronos_egl as egl;

use crate::cuda::{CudaDevice, CudaEvent, CudaStream, FramePool, UploadedFrame};
use crate::cuda_driver_ffi::{self as ffi, check};
use crate::cudadeviceid::{
    get_cuda_device_id, set_cuda_device_id, CUDA_DEVICE_ID_PROP, DEFAULT_CUDA_DEVICE_ID,
};

// EGL_EXT_platform_device
const EGL_PLATFORM_DEVICE_EXT: egl::Enum = 0x313F;
// EGL_EXT_image_dma_buf_import and EGL_EXT_image_dma_buf_import_modifiers
const EGL_LINUX_DMA_BUF_EXT: egl::Enum = 0x3270;
const EGL_LINUX_DRM_FOURCC_EXT: egl::Attrib = 0x3271;
const EGL_DMA_BUF_PLANE0_FD_EXT: egl::Attrib = 0x3272;
const EGL_DMA_BUF_PLANE0_OFFSET_EXT: egl::Attrib = 0x3273;
const EGL_DMA_BUF_PLANE0_PITCH_EXT: egl::Attrib = 0x3274;
const EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT: egl::Attrib = 0x3443;
const EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT: egl::Attrib = 0x3444;
const BYTES_PER_PIXEL: usize = 4;
const PACKED_FORMATS: [RawVideoFormat; 2] = [RawVideoFormat::Rgba8, RawVideoFormat::Bgra8];

fn packed_any() -> CapsSet {
    CapsSet::from_alternatives(
        PACKED_FORMATS
            .map(|format| Caps::RawVideo {
                format,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                interlace: Interlace::Progressive,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
                pixel_aspect: PixelAspect::SQUARE,
            })
            .into(),
    )
}

fn egl_error(what: &str, error: impl core::fmt::Debug) -> G2gError {
    g2g_core::g2g_error!(
        Target::category(short_type_name::<DmaBufToCuda>()),
        "{what}: {error:?}"
    );
    G2gError::Hardware(HardwareError::Other)
}

// bounds the images kept for buffers the producer has since freed
const MAX_REGISTERED_IMAGES: usize = 16;

struct RegisteredImage {
    inode: u64,
    image: egl::Image,
    resource: ffi::CuGraphicsResource,
    array: usize,
}

fn dmabuf_inode(dmabuf: &OwnedDmaBuf) -> Result<u64, G2gError> {
    let mut status = core::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `status` has room for one stat and the fd stays open for the call.
    if unsafe { libc::fstat(dmabuf.as_raw(), status.as_mut_ptr()) } != 0 {
        return Err(egl_error(
            "reading the dma-buf's inode",
            std::io::Error::last_os_error(),
        ));
    }
    // SAFETY: fstat filled `status` when it returned 0.
    Ok(unsafe { status.assume_init() }.st_ino)
}

struct ImportGpu {
    egl: crate::egl_library::Egl,
    display: egl::Display,
    device: Arc<CudaDevice>,
    stream: CudaStream,
    pool: FramePool,
    // linux numbers each dma-buf's inode uniquely, so a buffer the producer reuses keeps its registration
    images: Vec<RegisteredImage>,
}

// SAFETY: EGL display and image handles and CUDA graphics resources are process-wide, the element pushes its CUDA context before each CUDA call and needs no current EGL context.
unsafe impl Send for ImportGpu {}

impl core::fmt::Debug for ImportGpu {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ImportGpu")
            .field("device", &self.device)
            .field("pool", &self.pool)
            .field("registered_images", &self.images.len())
            .finish_non_exhaustive()
    }
}

impl Drop for ImportGpu {
    fn drop(&mut self) {
        for registered in core::mem::take(&mut self.images) {
            self.release(registered);
        }
        let _ = self.egl.terminate(self.display);
    }
}

impl ImportGpu {
    fn open(cuda_device_id: i32, width: u32, height: u32) -> Result<Self, G2gError> {
        let egl = crate::egl_library::load().map_err(|error| egl_error("loading EGL", error))?;
        let egl_device = crate::egldevice::cuda_egl_device(&egl, cuda_device_id)
            .map_err(|error| egl_error("finding the CUDA device's EGL device", error))?;
        // SAFETY: an EGL device handle is the native display EGL_EXT_platform_device takes.
        let display = unsafe {
            egl.get_platform_display(EGL_PLATFORM_DEVICE_EXT, egl_device, &[egl::ATTRIB_NONE])
        }
        .map_err(|error| egl_error("opening the EGL device display", error))?;
        egl.initialize(display)
            .map_err(|error| egl_error("initializing the EGL device display", error))?;
        let device = CudaDevice::open(cuda_device_id)?;
        let stream = CudaStream::new(&device)?;
        let pool = FramePool::new(width as usize * BYTES_PER_PIXEL, height as usize);
        Ok(Self {
            egl,
            display,
            device,
            stream,
            pool,
            images: Vec::new(),
        })
    }

    fn create_image(
        &self,
        dmabuf: &OwnedDmaBuf,
        format: RawVideoFormat,
        width: u32,
        height: u32,
    ) -> Result<egl::Image, G2gError> {
        let fourcc = crate::capsfilter::drm_format(format).ok_or(G2gError::CapsMismatch)?;
        let modifier = dmabuf.modifier();
        let attributes = [
            egl::WIDTH as egl::Attrib,
            width as egl::Attrib,
            egl::HEIGHT as egl::Attrib,
            height as egl::Attrib,
            EGL_LINUX_DRM_FOURCC_EXT,
            fourcc as egl::Attrib,
            EGL_DMA_BUF_PLANE0_FD_EXT,
            dmabuf.as_raw() as egl::Attrib,
            EGL_DMA_BUF_PLANE0_OFFSET_EXT,
            dmabuf.offset as egl::Attrib,
            EGL_DMA_BUF_PLANE0_PITCH_EXT,
            dmabuf.stride as egl::Attrib,
            EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
            (modifier & u64::from(u32::MAX)) as egl::Attrib,
            EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
            (modifier >> u32::BITS) as egl::Attrib,
            egl::ATTRIB_NONE,
        ];
        // SAFETY: EGL_LINUX_DMA_BUF_EXT takes no context and no client buffer, the fd stays open for the call.
        let (no_context, no_buffer) = unsafe {
            (
                egl::Context::from_ptr(egl::NO_CONTEXT),
                egl::ClientBuffer::from_ptr(core::ptr::null_mut()),
            )
        };
        self.egl
            .create_image(
                self.display,
                no_context,
                EGL_LINUX_DMA_BUF_EXT,
                no_buffer,
                &attributes,
            )
            .map_err(|error| {
                egl_error(
                    &alloc::format!("importing a dma-buf with DRM format modifier {modifier:#x}"),
                    error,
                )
            })
    }

    fn register(&self, image: egl::Image) -> Result<(ffi::CuGraphicsResource, usize), G2gError> {
        let _bound = self.device.bind()?;
        let mut resource: ffi::CuGraphicsResource = core::ptr::null_mut();
        // SAFETY: `image` is a live EGLImage of this device's display and the context is current.
        check(unsafe {
            ffi::cu_graphics_egl_register_image(
                &mut resource,
                image.as_ptr().cast::<c_void>(),
                ffi::CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY,
            )
        })?;
        let mut frame = ffi::CuEglFrame::default();
        // SAFETY: `frame` is a valid out pointer and `resource` is a registered EGL image.
        let mapped = check(unsafe {
            ffi::cu_graphics_resource_get_mapped_egl_frame(&mut frame, resource, 0, 0)
        })
        .and_then(|()| {
            if frame.frame_type == ffi::CU_EGL_FRAME_TYPE_ARRAY {
                return Ok(frame.planes[0]);
            }
            Err(egl_error(
                "the imported dma-buf is not a CUDA array",
                frame.frame_type,
            ))
        });
        if mapped.is_err() {
            // SAFETY: the resource was registered above and nothing has used it.
            unsafe { ffi::cu_graphics_unregister_resource(resource) };
        }
        Ok((resource, mapped?))
    }

    fn release(&self, registered: RegisteredImage) {
        let category = Target::category(short_type_name::<DmaBufToCuda>());
        let unregistered = self.device.bind().and_then(|_bound| {
            // SAFETY: the copies that read the resource finished before their frames were pushed.
            check(unsafe { ffi::cu_graphics_unregister_resource(registered.resource) })
        });
        if let Err(error) = unregistered {
            g2g_core::g2g_warn!(
                category,
                "unregistering a dma-buf image from CUDA: {error:?}"
            );
        }
        if let Err(error) = self.egl.destroy_image(self.display, registered.image) {
            g2g_core::g2g_warn!(category, "destroying an imported dma-buf image: {error:?}");
        }
    }

    fn registered_array(
        &mut self,
        dmabuf: &OwnedDmaBuf,
        format: RawVideoFormat,
        width: u32,
        height: u32,
    ) -> Result<usize, G2gError> {
        let inode = dmabuf_inode(dmabuf)?;
        if let Some(registered) = self.images.iter().find(|image| image.inode == inode) {
            return Ok(registered.array);
        }
        let image = self.create_image(dmabuf, format, width, height)?;
        let (resource, array) = match self.register(image) {
            Ok(registered) => registered,
            Err(error) => {
                let _ = self.egl.destroy_image(self.display, image);
                return Err(error);
            }
        };
        if self.images.len() == MAX_REGISTERED_IMAGES {
            let oldest = self.images.remove(0);
            self.release(oldest);
        }
        self.images.push(RegisteredImage {
            inode,
            image,
            resource,
            array,
        });
        Ok(array)
    }

    fn import(
        &mut self,
        dmabuf: &OwnedDmaBuf,
        format: RawVideoFormat,
        width: u32,
        height: u32,
    ) -> Result<OwnedCudaBuffer, G2gError> {
        let array = self.registered_array(dmabuf, format, width, height)?;
        let allocation = self.pool.acquire(&self.device)?;
        let pitch = u32::try_from(allocation.pitch).map_err(|_| G2gError::CapsMismatch)?;
        let copy = ffi::CudaMemcpy2D {
            src_x_in_bytes: 0,
            src_y: 0,
            src_memory_type: ffi::CU_MEMORYTYPE_ARRAY,
            src_host: core::ptr::null(),
            src_device: 0,
            src_array: array as *mut c_void,
            src_pitch: 0,
            dst_x_in_bytes: 0,
            dst_y: 0,
            dst_memory_type: ffi::CU_MEMORYTYPE_DEVICE,
            dst_host: core::ptr::null_mut(),
            dst_device: allocation.pointer,
            dst_array: core::ptr::null_mut(),
            dst_pitch: allocation.pitch,
            width_in_bytes: width as usize * BYTES_PER_PIXEL,
            height: height as usize,
        };
        // SAFETY: the registered array lives until its image is released, and the pool allocation holds pitch * height bytes.
        unsafe { self.stream.copy_2d_async(&copy) }?;
        // producers render into the dma-buf again once its frame is dropped, so the copy finishes first
        self.stream.synchronize()?;
        let ready = CudaEvent::new(&self.device)?;
        self.stream.record(&ready)?;
        Ok(OwnedCudaBuffer::new(
            allocation.pointer,
            0,
            pitch,
            0,
            width,
            height,
            self.device.context,
            self.device.ordinal,
            Arc::new(UploadedFrame::new(allocation, &self.pool, ready)),
        ))
    }
}

#[derive(Debug)]
pub struct DmaBufToCuda {
    cuda_device_id: i32,
    format: RawVideoFormat,
    width: u32,
    height: u32,
    gpu: Option<ImportGpu>,
    imported: u64,
}

impl Default for DmaBufToCuda {
    fn default() -> Self {
        Self {
            cuda_device_id: DEFAULT_CUDA_DEVICE_ID,
            format: RawVideoFormat::Rgba8,
            width: 0,
            height: 0,
            gpu: None,
            imported: 0,
        }
    }
}

impl DmaBufToCuda {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_cuda_device_id(mut self, ordinal: i32) -> Self {
        self.cuda_device_id = ordinal;
        self
    }

    pub fn imported(&self) -> u64 {
        self.imported
    }
}

static DMABUF_TO_CUDA_PROPS: &[PropertySpec] = &[CUDA_DEVICE_ID_PROP];

impl AsyncElement for DmaBufToCuda {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "dma-buf to CUDA",
            "Filter/Converter/Video/GPU",
            "Copies an RGBA dma-buf into CUDA device memory on the same NVIDIA GPU",
            crate::AARON_BOXER,
        )
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        packed_any()
            .alternatives()
            .iter()
            .find_map(|alternative| upstream_caps.intersect(alternative).ok())
            .ok_or(G2gError::CapsMismatch)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(packed_any())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            ..
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !PACKED_FORMATS.contains(format) {
            return Err(G2gError::CapsMismatch);
        }
        self.format = *format;
        self.width = *width;
        self.height = *height;
        // opening the device here fails the configure on a host without an NVIDIA GPU
        self.gpu = Some(ImportGpu::open(self.cuda_device_id, *width, *height)?);
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        DMABUF_TO_CUDA_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "cuda-device-id" => {
                set_cuda_device_id(&mut self.cuda_device_id, self.gpu.is_some(), &value)
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "cuda-device-id" => Some(get_cuda_device_id(self.cuda_device_id)),
            _ => None,
        }
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::DmaBuf)
    }

    fn output_memory(&self) -> MemoryDomainKind {
        MemoryDomainKind::Cuda
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            let PipelinePacket::DataFrame(mut frame) = packet else {
                if !matches!(packet, PipelinePacket::Eos) {
                    out.push(packet).await?;
                }
                return Ok(());
            };
            let MemoryDomain::DmaBuf(dmabuf) = &frame.domain else {
                return Err(G2gError::UnsupportedDomain);
            };
            crate::dmabuffence::wait_for_producer(dmabuf).await?;
            let (format, width, height) = (self.format, self.width, self.height);
            let gpu = self.gpu.as_mut().ok_or(G2gError::NotConfigured)?;
            let buffer = gpu.import(dmabuf, format, width, height)?;
            frame.domain = MemoryDomain::Cuda(buffer);
            self.imported += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
            Ok(())
        })
    }
}

impl PadTemplates for DmaBufToCuda {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(packed_any()),
            PadTemplate::source(packed_any()),
        ])
    }
}
