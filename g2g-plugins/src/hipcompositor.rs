// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::caps_parse::parse_raw_format;
use g2g_core::memory::{DomainSet, GpuDevice, MemoryDomainKind, OwnedHipBuffer};
use g2g_core::meta::Orientation;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, Colorimetry, ConfigureOutcome, Dim, ElementMetadata, Frame,
    FrameTiming, G2gError, MemoryDomain, MultiInputElement, OutputSink, PipelinePacket, PropError,
    PropValue, PropertySpec, Rate, RawVideoFormat,
};

use crate::compositor::{
    configure_compositor_input, paint_order, process_compositor_packet, CompositorBackend,
    CompositorCanvas, CompositorPad, HIP_COMPOSITOR_PROPS,
};
use crate::hip::{
    CrossDeviceCopier, HipDevice, HipEvent, HipKernel, HipKernelModule, HipMemoryPool, HipStream,
    KernelArgument,
};
use crate::hipconvertscale::{
    any_size_formats, converts, narrow_to_converted_format, HipBlender, HipConverter,
};
use crate::hipdeviceid::HipDeviceChoice;

const FILL_SOURCE: &str = r#"
extern "C" __global__ void fill_canvas(float* canvas, unsigned int pixels, float r, float g, float b, float a) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < pixels) {
        canvas[4 * index] = r;
        canvas[4 * index + 1] = g;
        canvas[4 * index + 2] = b;
        canvas[4 * index + 3] = a;
    }
}
"#;

const FILL_BLOCK: u32 = 256;
// four f32 channels per canvas pixel
const CANVAS_PIXEL_BYTES: usize = 16;

fn canvas_caps(width: u32, height: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::RgbaF32,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate: Rate::Any,
        interlace: g2g_core::Interlace::Any,
        colorimetry: Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
    }
}

#[derive(Debug)]
struct CompositeGpu {
    device: Arc<HipDevice>,
    stream: HipStream,
    fill: HipKernel,
    canvas_pool: HipMemoryPool,
    blenders: Vec<Option<HipBlender>>,
    output_caps: Caps,
    output: HipConverter,
    output_pool: HipMemoryPool,
}

impl CompositeGpu {
    fn open(
        device: Arc<HipDevice>,
        inputs: usize,
        output_caps: Caps,
        (width, height): (u32, u32),
    ) -> Result<Self, G2gError> {
        let fill = HipKernelModule::compile(&device, FILL_SOURCE, &[])?.kernel("fill_canvas")?;
        let output = HipConverter::new(
            &device,
            &canvas_caps(width, height),
            &output_caps,
            false,
            Orientation::Identity,
        )?;
        Ok(Self {
            stream: HipStream::new(&device)?,
            fill,
            canvas_pool: HipMemoryPool::new(
                &device,
                width as usize * height as usize * CANVAS_PIXEL_BYTES,
            ),
            blenders: (0..inputs).map(|_| None).collect(),
            output_pool: HipMemoryPool::new(&device, output.output_len()?),
            output,
            output_caps,
            device,
        })
    }
}

#[derive(Debug)]
pub struct HipCompositor {
    canvas: CompositorCanvas<OwnedHipBuffer>,
    device: HipDeviceChoice,
    running_on: Option<GpuDevice>,
    copier: CrossDeviceCopier,
    input_caps: Vec<Option<Caps>>,
    gpu: Option<CompositeGpu>,
}

impl HipCompositor {
    pub fn new(out_w: u32, out_h: u32, pads: Vec<CompositorPad>) -> Self {
        let inputs = pads.len();
        Self {
            canvas: CompositorCanvas::new(out_w, out_h, RawVideoFormat::Rgba8, pads),
            device: HipDeviceChoice::default(),
            running_on: None,
            copier: CrossDeviceCopier::default(),
            input_caps: vec![None; inputs],
            gpu: None,
        }
    }

    pub fn with_format(mut self, format: RawVideoFormat) -> Self {
        assert!(converts(format), "hipcompositor cannot write {format:?}");
        self.canvas.format = format;
        self
    }

    pub fn with_background(mut self, rgba: [u8; 4]) -> Self {
        self.canvas.background = rgba;
        self
    }

    fn running_on(&mut self, first: &OwnedHipBuffer) -> Result<GpuDevice, G2gError> {
        if let Some(device) = self.running_on {
            return Ok(device);
        }
        let device = match self.device.is_set() {
            true => {
                let opened = self.device.open()?;
                GpuDevice {
                    vendor: opened.vendor(),
                    ordinal: opened.ordinal(),
                }
            }
            false => GpuDevice {
                vendor: first.vendor,
                ordinal: first.device_ordinal,
            },
        };
        Ok(*self.running_on.insert(device))
    }

    fn gpu_for(&mut self, base: &OwnedHipBuffer) -> Result<&mut CompositeGpu, G2gError> {
        let output_caps = self.canvas.output();
        let reusable = self.gpu.as_ref().is_some_and(|gpu| {
            gpu.device.vendor() == base.vendor
                && gpu.device.ordinal() == base.device_ordinal
                && gpu.output_caps == output_caps
        });
        if !reusable {
            let device = HipDevice::open(base.vendor, base.device_ordinal)?;
            let size = (self.canvas.out_w, self.canvas.out_h);
            self.gpu = Some(CompositeGpu::open(
                device,
                self.input_caps.len(),
                output_caps,
                size,
            )?);
        }
        self.gpu.as_mut().ok_or(G2gError::NotConfigured)
    }

    fn composite(&mut self, base: &OwnedHipBuffer) -> Result<OwnedHipBuffer, G2gError> {
        self.gpu_for(base)?;
        let Self {
            canvas,
            input_caps,
            gpu,
            ..
        } = self;
        let gpu = gpu.as_mut().ok_or(G2gError::NotConfigured)?;
        let (width, height) = (canvas.out_w, canvas.out_h);
        let pixels = width * height;
        let mut memory = gpu.canvas_pool.acquire()?;
        let [r, g, b, a] = canvas.background.map(|channel| f32::from(channel) / 255.0);
        gpu.stream.launch(
            &gpu.fill,
            [pixels.div_ceil(FILL_BLOCK), 1, 1],
            [FILL_BLOCK, 1, 1],
            &[
                KernelArgument::Write(&mut memory),
                KernelArgument::U32(pixels),
                KernelArgument::F32(r),
                KernelArgument::F32(g),
                KernelArgument::F32(b),
                KernelArgument::F32(a),
            ],
        )?;
        for input in paint_order(&canvas.pads) {
            if canvas.state.geometry(input).is_none() {
                continue;
            }
            let buffer = match input {
                0 => base,
                _ => match canvas.state.latest(input) {
                    Some((_, buffer)) => buffer,
                    None => continue,
                },
            };
            if (buffer.vendor, buffer.device_ordinal) != (gpu.device.vendor(), gpu.device.ordinal())
            {
                return Err(G2gError::UnsupportedDomain);
            }
            let blender = match &mut gpu.blenders[input] {
                Some(blender) => blender,
                slot => {
                    let caps = input_caps[input].as_ref().ok_or(G2gError::NotConfigured)?;
                    slot.insert(HipBlender::new(&gpu.device, caps)?)
                }
            };
            blender.blend(
                &gpu.stream,
                buffer,
                &mut memory,
                (width, height),
                &canvas.pads[input],
            )?;
        }
        let blended = HipEvent::new(&gpu.device)?;
        gpu.stream.record(&blended)?;
        let canvas_buffer = memory.into_buffer(blended, gpu.canvas_pool.size())?;
        let mut output = gpu.output_pool.acquire()?;
        gpu.output
            .convert(&gpu.stream, &canvas_buffer, &mut output)?;
        let ready = HipEvent::new(&gpu.device)?;
        gpu.stream.record(&ready)?;
        output.into_buffer(ready, gpu.output.output_len()?)
    }
}

impl MultiInputElement for HipCompositor {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "HIP compositor",
            "Filter/Editor/Video/Compositor/Hardware",
            "Composites several Hip video inputs onto one timed output canvas on the GPU",
            crate::AARON_BOXER,
        )
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::Hip)
    }

    fn output_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::Hip)
    }

    fn gpu_device(&self) -> Option<GpuDevice> {
        self.device.declared()
    }

    fn input_count(&self) -> usize {
        self.canvas.pads.len()
    }

    fn tick_interval_ns(&self) -> Option<u64> {
        self.canvas.tick_interval_ns()
    }

    fn properties(&self) -> &'static [PropertySpec] {
        HIP_COMPOSITOR_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if let Some(applied) = self.canvas.set_property(name, &value) {
            return applied;
        }
        if let Some(applied) = self.device.set_property(name, &value) {
            self.gpu = None;
            self.running_on = None;
            return applied;
        }
        if name != "format" {
            return Err(PropError::Unknown);
        }
        let format = parse_raw_format(value.as_str().ok_or(PropError::Type)?)
            .filter(|&format| converts(format))
            .ok_or(PropError::Value)?;
        self.canvas.format = format;
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "format" => Some(PropValue::Str(self.canvas.format.gst_name().into())),
            "device-id" | "vendor" => self.device.get_property(name),
            _ => self.canvas.get_property(name),
        }
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        narrow_to_converted_format(upstream_caps)
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::from_alternatives(any_size_formats().collect()))
    }

    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        Ok(CapsConstraint::Produces(CapsSet::one(self.canvas.output())))
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        configure_compositor_input(self, input, absolute_caps)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(self.canvas.output())
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(process_compositor_packet(self, input, packet, out))
    }
}

impl CompositorBackend for HipCompositor {
    type Payload = OwnedHipBuffer;

    fn canvas(&mut self) -> &mut CompositorCanvas<OwnedHipBuffer> {
        &mut self.canvas
    }

    fn accept_input(&mut self, input: usize, caps: &Caps) -> Result<(u32, u32), G2gError> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            ..
        } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !converts(*format) {
            return Err(G2gError::CapsMismatch);
        }
        if self.input_caps[input].as_ref() != Some(caps) {
            self.input_caps[input] = Some(caps.clone());
            if let Some(gpu) = &mut self.gpu {
                gpu.blenders[input] = None;
            }
        }
        Ok((*width, *height))
    }

    async fn payload(&mut self, _input: usize, frame: Frame) -> Result<OwnedHipBuffer, G2gError> {
        let MemoryDomain::Hip(buffer) = frame.domain else {
            return Err(G2gError::UnsupportedDomain);
        };
        let target = self.running_on(&buffer)?;
        if (buffer.vendor, buffer.device_ordinal) == (target.vendor, target.ordinal) {
            return Ok(buffer);
        }
        self.copier.copy(buffer, target).await
    }

    fn compose_frame<'a>(
        &'a mut self,
        base: &'a OwnedHipBuffer,
        timing: FrameTiming,
    ) -> Pin<Box<dyn Future<Output = Result<Frame, G2gError>> + 'a>> {
        let frame = self.composite(base).map(|buffer| {
            Frame::new(
                MemoryDomain::Hip(buffer),
                timing,
                self.canvas.state.next_sequence(),
            )
        });
        Box::pin(core::future::ready(frame))
    }
}
