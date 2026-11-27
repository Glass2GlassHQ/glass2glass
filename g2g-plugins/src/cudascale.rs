// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::ffi::c_void;
use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;

use g2g_core::caps_transform::CapsTransform;
use g2g_core::log::{short_type_name, Target};
use g2g_core::memory::{DomainSet, MemoryDomainKind, OwnedCudaBuffer};
use g2g_core::{
    g2g_error, AsyncElement, Caps, CapsConstraint, CapsPreferences, Colorimetry, ConfigureOutcome,
    Dim, ElementMetadata, Frame, G2gError, Interlace, MemoryDomain, OutputSink, PipelinePacket,
    PixelAspect, Rate, RawVideoFormat,
};

use crate::cuda::{CudaDevice, CudaEvent, CudaStream, FramePool, UploadedFrame};
use crate::cuda_driver_ffi as ffi;
use crate::display_aspect::scaled_video_caps;
use crate::gpurelease::release_after;
use crate::videoscale::input_size_or_any_size;

// cuModuleLoadData reads PTX text up to its terminating NUL
const SCALE_PTX: &str = concat!(include_str!("cudascale.ptx"), "\0");
const BLOCK_SIZE: u32 = 16;
const FORMATS: [RawVideoFormat; 3] = [
    RawVideoFormat::Nv12,
    RawVideoFormat::Rgba8,
    RawVideoFormat::Bgra8,
];
const PACKED_BYTES_PER_PIXEL: u32 = 4;

#[derive(Debug, Clone, Copy)]
struct Plane {
    pointer: u64,
    pitch: u32,
    width: u32,
    height: u32,
}

#[derive(Debug)]
struct ScaleKernel {
    device: Arc<CudaDevice>,
    module: usize,
    function: usize,
}

impl ScaleKernel {
    fn load(device: &Arc<CudaDevice>) -> Result<Self, G2gError> {
        let _bound = device.bind()?;
        let mut module: ffi::CuModule = core::ptr::null_mut();
        // SAFETY: the PTX text ends in a NUL and the device's context is current.
        ffi::check(unsafe { ffi::cu_module_load_data(&mut module, SCALE_PTX.as_ptr().cast()) })?;
        let mut kernel = Self {
            device: Arc::clone(device),
            module: module as usize,
            function: 0,
        };
        let mut function: ffi::CuFunction = core::ptr::null_mut();
        // SAFETY: the module was loaded in the current context and the name is NUL-terminated.
        ffi::check(unsafe {
            ffi::cu_module_get_function(&mut function, module, c"scale_plane".as_ptr())
        })?;
        kernel.function = function as usize;
        Ok(kernel)
    }

    fn scale(
        &self,
        stream: &CudaStream,
        source: Plane,
        destination: Plane,
        channels: u32,
    ) -> Result<(), G2gError> {
        fn argument<T>(value: &T) -> *mut c_void {
            core::ptr::from_ref(value).cast_mut().cast()
        }
        let mut parameters = [
            argument(&source.pointer),
            argument(&source.pitch),
            argument(&source.width),
            argument(&source.height),
            argument(&destination.pointer),
            argument(&destination.pitch),
            argument(&destination.width),
            argument(&destination.height),
            argument(&channels),
        ];
        let grid = [
            destination.width.div_ceil(BLOCK_SIZE),
            destination.height.div_ceil(BLOCK_SIZE),
        ];
        // SAFETY: the parameters match scale_plane's signature, and the caller keeps both planes alive until the stream passes the launch.
        unsafe {
            stream.launch(
                self.function as ffi::CuFunction,
                grid,
                [BLOCK_SIZE, BLOCK_SIZE],
                &mut parameters,
            )
        }
    }
}

impl Drop for ScaleKernel {
    fn drop(&mut self) {
        let Ok(_bound) = self.device.bind() else {
            return;
        };
        // SAFETY: the module was loaded in this context and no launch queued after this uses it.
        unsafe { ffi::cu_module_unload(self.module as ffi::CuModule) };
    }
}

fn planes(format: RawVideoFormat, buffer: &OwnedCudaBuffer) -> Vec<(Plane, u32)> {
    let luma = Plane {
        pointer: buffer.luma_ptr,
        pitch: buffer.luma_pitch,
        width: buffer.width,
        height: buffer.height,
    };
    if format != RawVideoFormat::Nv12 {
        return Vec::from([(luma, PACKED_BYTES_PER_PIXEL)]);
    }
    let chroma = Plane {
        pointer: buffer.chroma_ptr,
        pitch: buffer.chroma_pitch,
        width: buffer.width.div_ceil(2),
        height: buffer.height.div_ceil(2),
    };
    Vec::from([(luma, 1), (chroma, 2)])
}

#[derive(Debug)]
struct ScaleGpu {
    device: Arc<CudaDevice>,
    stream: CudaStream,
    kernel: ScaleKernel,
    pool: Option<FramePool>,
}

impl ScaleGpu {
    fn open(ordinal: i32) -> Result<Self, G2gError> {
        let device = CudaDevice::open(ordinal)?;
        Ok(Self {
            stream: CudaStream::new(&device)?,
            kernel: ScaleKernel::load(&device)?,
            device,
            pool: None,
        })
    }

    fn pool(&mut self, wanted: FramePool) -> &FramePool {
        match self.pool.take() {
            Some(pool) if pool.has_shape_of(&wanted) => self.pool.insert(pool),
            _ => self.pool.insert(wanted),
        }
    }
}

#[derive(Debug, Default)]
pub struct CudaScale {
    input_caps: Option<Caps>,
    output_caps: Option<Caps>,
    last_caps: Option<Caps>,
    gpu: Option<ScaleGpu>,
}

impl CudaScale {
    pub fn new() -> Self {
        Self::default()
    }

    fn output_size(&self) -> Result<(u32, u32), G2gError> {
        match &self.output_caps {
            Some(Caps::RawVideo {
                width: Dim::Fixed(width),
                height: Dim::Fixed(height),
                ..
            }) if *width > 0 && *height > 0 => Ok((*width, *height)),
            _ => Err(G2gError::NotConfigured),
        }
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        let input = self.input_caps.as_ref().ok_or(G2gError::NotConfigured)?;
        let output = self.output_caps.as_ref().ok_or(G2gError::NotConfigured)?;
        Ok(scaled_video_caps(input, output))
    }

    fn gpu_for(&mut self, buffer: &OwnedCudaBuffer) -> Result<&mut ScaleGpu, G2gError> {
        let gpu = match self.gpu.take() {
            Some(gpu) if gpu.device.ordinal == buffer.device_ordinal => gpu,
            _ => ScaleGpu::open(buffer.device_ordinal)?,
        };
        Ok(self.gpu.insert(gpu))
    }

    fn scale(&mut self, input: OwnedCudaBuffer) -> Result<OwnedCudaBuffer, G2gError> {
        let Some(Caps::RawVideo { format, .. }) = &self.input_caps else {
            return Err(G2gError::NotConfigured);
        };
        let format = *format;
        let (width, height) = self.output_size()?;
        let gpu = self.gpu_for(&input)?;
        gpu.stream.wait_for(&input)?;
        let device = Arc::clone(&gpu.device);
        let nv12 = format == RawVideoFormat::Nv12;
        let shape = if nv12 {
            FramePool::nv12(width, height)
        } else {
            FramePool::new((width * PACKED_BYTES_PER_PIXEL) as usize, height as usize)
        };
        let pool = gpu.pool(shape);
        let allocation = pool.acquire(&device)?;
        let pitch = u32::try_from(allocation.pitch).map_err(|_| G2gError::CapsMismatch)?;
        let luma = allocation.pointer;
        let (chroma, chroma_pitch) = if nv12 {
            (luma + u64::from(pitch) * u64::from(height), pitch)
        } else {
            (0, 0)
        };
        let frame = Arc::new(UploadedFrame::new(
            allocation,
            pool,
            CudaEvent::new(&device)?,
        ));
        let output = OwnedCudaBuffer::new(
            luma,
            chroma,
            pitch,
            chroma_pitch,
            width,
            height,
            device.context,
            device.ordinal,
            frame.clone(),
        );
        let pairs = planes(format, &input)
            .into_iter()
            .zip(planes(format, &output));
        for ((source, channels), (destination, _)) in pairs {
            gpu.kernel
                .scale(&gpu.stream, source, destination, channels)?;
        }
        gpu.stream.record(frame.ready())?;
        let read = CudaEvent::new(&device)?;
        gpu.stream.record(&read)?;
        release_after(read, Box::new(input));
        Ok(output)
    }
}

fn any_size(format: RawVideoFormat) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        interlace: Interlace::Any,
        colorimetry: Colorimetry::UNKNOWN,
        pixel_aspect: PixelAspect::Any,
    }
}

impl AsyncElement for CudaScale {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "CUDA video scaler",
            "Filter/Converter/Video/Scaler/Hardware",
            "Resizes NV12, RGBA and BGRA frames in CUDA memory",
            crate::AARON_BOXER,
        )
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        FORMATS
            .iter()
            .find_map(|&format| upstream_caps.intersect(&any_size(format)).ok())
            .ok_or(G2gError::CapsMismatch)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedFields(CapsTransform::RawVideo {
            accept: FORMATS.to_vec(),
            produce: Vec::new(),
            shapes: input_size_or_any_size(),
        })
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::Cuda)
    }

    fn caps_preferences(&self) -> Option<CapsPreferences> {
        Some(
            CapsPreferences::default()
                .with_aspect_preserving_width(core::num::NonZeroU32::MIN, false),
        )
    }

    fn output_memory(&self) -> MemoryDomainKind {
        MemoryDomainKind::Cuda
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::RawVideo { format, .. } = absolute_caps else {
            return Err(G2gError::CapsMismatch);
        };
        if !FORMATS.contains(format) {
            return Err(G2gError::CapsMismatch);
        }
        self.input_caps = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn configure_output(&mut self, output_caps: &Caps) -> Result<(), G2gError> {
        let Caps::RawVideo {
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            ..
        } = output_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if *width == 0 || *height == 0 {
            return Err(G2gError::CapsMismatch);
        }
        self.output_caps = Some(output_caps.clone());
        Ok(())
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::CapsChanged(_) | PipelinePacket::DataFrame(_) => {
                    let output_caps = self.output_caps()?;
                    if self.last_caps.as_ref() != Some(&output_caps) {
                        out.push(PipelinePacket::CapsChanged(output_caps.clone()))
                            .await?;
                        self.last_caps = Some(output_caps);
                    }
                    let PipelinePacket::DataFrame(frame) = packet else {
                        return Ok(());
                    };
                    let Frame {
                        domain,
                        timing,
                        sequence,
                        meta,
                    } = frame;
                    let MemoryDomain::Cuda(buffer) = domain else {
                        g2g_error!(
                            Target::category(short_type_name::<Self>()),
                            "scales Cuda frames, and the link delivered a {:?} frame",
                            domain.kind()
                        );
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let domain = if (buffer.width, buffer.height) == self.output_size()? {
                        MemoryDomain::Cuda(buffer)
                    } else {
                        MemoryDomain::Cuda(self.scale(buffer)?)
                    };
                    out.push(PipelinePacket::DataFrame(Frame {
                        domain,
                        timing,
                        sequence,
                        meta,
                    }))
                    .await?;
                }
                PipelinePacket::Flush => {
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
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
