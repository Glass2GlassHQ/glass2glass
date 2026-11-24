// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::os::fd::{IntoRawFd, OwnedFd};
use std::path::Path;

use g2g_core::memory::OwnedDmaBuf;
use g2g_core::G2gError;
use smithay::backend::allocator::dmabuf::{AsDmabuf, Dmabuf};
use smithay::backend::allocator::gbm::{GbmAllocator, GbmBufferFlags, GbmDevice};
use smithay::backend::allocator::{Allocator, Buffer, Modifier};
use smithay::backend::drm::{DrmDeviceFd, DrmNode};
use smithay::backend::egl::{EGLContext, EGLDisplay};
use smithay::backend::renderer::damage::OutputDamageTracker;
use smithay::backend::renderer::element::surface::WaylandSurfaceRenderElement;
use smithay::backend::renderer::gles::{GlesRenderer, GlesTarget};
use smithay::backend::renderer::{Bind, ExportMem, ImportDma, TextureMapping};
use smithay::desktop::space::render_output;
use smithay::desktop::{Space, Window};
use smithay::output::Output;
use smithay::utils::{DeviceFd, Rectangle};
use smithay::wayland::dmabuf::{DmabufFeedback, DmabufFeedbackBuilder};

use crate::display::{
    hardware_error, DisplaySize, Picture, BACKGROUND, BYTES_PER_PIXEL, OUTPUT_FOURCC,
};

const MAX_OUTPUT_BUFFERS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuOutput {
    DmaBuf,
    System,
}

struct OutputBuffer {
    dmabuf: Dmabuf,
    shared: OwnedDmaBuf,
}

pub(crate) struct GpuRenderer {
    pub(crate) gles: GlesRenderer,
    allocator: GbmAllocator<DrmDeviceFd>,
    node: DrmNode,
    size: DisplaySize,
    output: GpuOutput,
    modifiers: Vec<Modifier>,
    buffers: Vec<OutputBuffer>,
}

impl GpuRenderer {
    pub(crate) fn open(
        render_node: &Path,
        size: DisplaySize,
        output: GpuOutput,
    ) -> Result<Self, G2gError> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(render_node)
            .map_err(|error| hardware_error("opening the render node", error))?;
        let device_fd = DrmDeviceFd::new(DeviceFd::from(OwnedFd::from(file)));
        let node = DrmNode::from_file(&device_fd)
            .map_err(|error| hardware_error("reading the render node", error))?;
        let gbm = GbmDevice::new(device_fd)
            .map_err(|error| hardware_error("opening gbm on the render node", error))?;
        // SAFETY: the gbm device was created above, so no other code holds its handle to get or terminate the same EGL display.
        let display = unsafe { EGLDisplay::new(gbm.clone()) }
            .map_err(|error| hardware_error("creating the EGL display", error))?;
        let context = EGLContext::new(&display)
            .map_err(|error| hardware_error("creating the EGL context", error))?;
        // SAFETY: the context was created on this thread above and is current on no other thread.
        let gles = unsafe { GlesRenderer::new(context) }
            .map_err(|error| hardware_error("creating the GLES renderer", error))?;
        let modifiers = output_modifiers(&gles)?;
        Ok(Self {
            gles,
            allocator: GbmAllocator::new(gbm, GbmBufferFlags::RENDERING),
            node,
            size,
            output,
            modifiers,
            buffers: Vec::new(),
        })
    }

    pub(crate) fn dmabuf_feedback(&self) -> Result<DmabufFeedback, G2gError> {
        DmabufFeedbackBuilder::new(self.node.dev_id(), self.gles.dmabuf_formats())
            .build()
            .map_err(|error| hardware_error("building the dma-buf feedback", error))
    }

    fn free_buffer(&mut self) -> Result<usize, G2gError> {
        if let Some(index) = self
            .buffers
            .iter()
            .position(|buffer| buffer.shared.share_count() == 1)
        {
            return Ok(index);
        }
        if self.buffers.len() == MAX_OUTPUT_BUFFERS {
            return Err(hardware_error(
                "downstream holds every output buffer",
                MAX_OUTPUT_BUFFERS,
            ));
        }
        let buffer = self
            .allocator
            .create_buffer(
                self.size.width,
                self.size.height,
                OUTPUT_FOURCC,
                &self.modifiers,
            )
            .map_err(|error| hardware_error("allocating an output buffer", error))?;
        let dmabuf = buffer
            .export()
            .map_err(|error| hardware_error("exporting an output buffer", error))?;
        let shared = share(&dmabuf)?;
        self.buffers.push(OutputBuffer { dmabuf, shared });
        Ok(self.buffers.len() - 1)
    }

    pub(crate) fn render(
        &mut self,
        output: &Output,
        space: &Space<Window>,
        damage_tracker: &mut OutputDamageTracker,
    ) -> Result<Picture, G2gError> {
        let index = self.free_buffer()?;
        let buffer = &mut self.buffers[index];
        let mut framebuffer = self
            .gles
            .bind(&mut buffer.dmabuf)
            .map_err(|error| hardware_error("binding an output buffer", error))?;
        let result = render_output::<_, WaylandSurfaceRenderElement<GlesRenderer>, _, _>(
            output,
            &mut self.gles,
            &mut framebuffer,
            1.0,
            0,
            [space],
            &[],
            damage_tracker,
            BACKGROUND,
        )
        .map_err(|error| hardware_error("rendering the output", error))?;
        result
            .sync
            .wait()
            .map_err(|error| hardware_error("waiting for the render", error))?;
        match self.output {
            GpuOutput::DmaBuf => Ok(Picture::DmaBuf(buffer.shared.clone())),
            GpuOutput::System => {
                read_framebuffer(&mut self.gles, &framebuffer, self.size).map(Picture::System)
            }
        }
    }
}

fn output_modifiers(gles: &GlesRenderer) -> Result<Vec<Modifier>, G2gError> {
    let renderable: Vec<Modifier> = Bind::<Dmabuf>::supported_formats(gles)
        .ok_or_else(|| hardware_error("listing the renderable formats", OUTPUT_FOURCC))?
        .iter()
        .filter(|format| format.code == OUTPUT_FOURCC && format.modifier != Modifier::Invalid)
        .map(|format| format.modifier)
        .collect();
    if renderable.contains(&Modifier::Linear) {
        return Ok(Vec::from([Modifier::Linear]));
    }
    if renderable.is_empty() {
        return Err(hardware_error("renderable modifiers", OUTPUT_FOURCC));
    }
    Ok(renderable)
}

fn share(dmabuf: &Dmabuf) -> Result<OwnedDmaBuf, G2gError> {
    let modifier = dmabuf.format().modifier;
    if modifier == Modifier::Invalid {
        return Err(hardware_error("output buffer modifier", modifier));
    }
    let ((plane, stride), offset) = dmabuf
        .handles()
        .zip(dmabuf.strides())
        .zip(dmabuf.offsets())
        .next()
        .ok_or_else(|| hardware_error("output buffer planes", dmabuf.num_planes()))?;
    let fd = plane
        .try_clone_to_owned()
        .map_err(|error| hardware_error("duplicating an output buffer", error))?;
    // SAFETY: the descriptor was duplicated above for this buffer alone, and ownership passes to it.
    let shared = unsafe { OwnedDmaBuf::from_raw(fd.into_raw_fd(), stride, offset) };
    Ok(shared.with_modifier(u64::from(modifier)))
}

fn read_framebuffer(
    gles: &mut GlesRenderer,
    framebuffer: &GlesTarget<'_>,
    size: DisplaySize,
) -> Result<Vec<u8>, G2gError> {
    let width = i32::try_from(size.width).map_err(|_| G2gError::CapsMismatch)?;
    let height = i32::try_from(size.height).map_err(|_| G2gError::CapsMismatch)?;
    let mapping = gles
        .copy_framebuffer(
            framebuffer,
            Rectangle::from_size((width, height).into()),
            OUTPUT_FOURCC,
        )
        .map_err(|error| hardware_error("reading back the output", error))?;
    let bytes = gles
        .map_texture(&mapping)
        .map_err(|error| hardware_error("mapping the read back output", error))?;
    // smithay calls a mapping flipped when its first row is the top one
    if mapping.flipped() {
        return Ok(bytes.to_vec());
    }
    let row_bytes = size.width as usize * BYTES_PER_PIXEL;
    Ok(bytes
        .chunks_exact(row_bytes)
        .rev()
        .flatten()
        .copied()
        .collect())
}
