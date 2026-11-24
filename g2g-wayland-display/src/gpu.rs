// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::os::fd::{IntoRawFd, OwnedFd};
use std::path::Path;

use g2g_core::memory::OwnedDmaBuf;
use g2g_core::G2gError;
use g2g_plugins::dmabufmap::DmaBufReadMap;
use smithay::backend::allocator::dmabuf::{AsDmabuf, Dmabuf, DmabufFlags};
use smithay::backend::allocator::gbm::{GbmAllocator, GbmBufferFlags, GbmDevice};
use smithay::backend::allocator::{Allocator, Buffer, Fourcc, Modifier};
use smithay::backend::drm::{DrmDeviceFd, DrmNode};
use smithay::backend::egl::{EGLContext, EGLDisplay};
use smithay::backend::renderer::damage::OutputDamageTracker;
use smithay::backend::renderer::element::surface::WaylandSurfaceRenderElement;
use smithay::backend::renderer::gles::{GlesRenderer, GlesTarget, GlesTexProgram, GlesTexture};
use smithay::backend::renderer::{
    Bind, ExportMem, Frame, ImportDma, Offscreen, Renderer, Texture, TextureMapping,
};
use smithay::desktop::space::render_output;
use smithay::desktop::{Space, Window};
use smithay::output::Output;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::{DeviceFd, Physical, Point, Rectangle, Transform};
use smithay::wayland::dmabuf::{DmabufFeedback, DmabufFeedbackBuilder};

use crate::display::{
    drag_icon_elements, hardware_error, DisplaySize, Picture, SessionOutput, BACKGROUND,
    BYTES_PER_PIXEL, OUTPUT_FOURCC,
};

const MAX_OUTPUT_BUFFERS: usize = 8;
const LUMA_FOURCC: Fourcc = Fourcc::R8;
// drm's gr88 stores R in the first byte, so Cb lands before Cr as NV12 wants
const CHROMA_FOURCC: Fourcc = Fourcc::Gr88;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuOutput {
    DmaBuf,
    System,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuFormat {
    Rgba,
    Nv12,
}

struct Nv12Planes {
    luma: Dmabuf,
    chroma: Dmabuf,
}

struct OutputBuffer<T> {
    targets: T,
    shared: OwnedDmaBuf,
}

struct Nv12Pass {
    composite: GlesTexture,
    luma_program: GlesTexProgram,
    chroma_program: GlesTexProgram,
}

enum OutputPass {
    Rgba {
        modifiers: Vec<Modifier>,
        buffers: Vec<OutputBuffer<Dmabuf>>,
    },
    Nv12 {
        pass: Nv12Pass,
        buffers: Vec<OutputBuffer<Nv12Planes>>,
    },
}

struct OutputTarget {
    output: GpuOutput,
    pass: OutputPass,
}

pub(crate) struct GpuRenderer {
    pub(crate) gles: GlesRenderer,
    allocator: GbmAllocator<DrmDeviceFd>,
    node: DrmNode,
    size: DisplaySize,
    target: Option<OutputTarget>,
}

impl GpuRenderer {
    pub(crate) fn open(render_node: &Path, size: DisplaySize) -> Result<Self, G2gError> {
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
        Ok(Self {
            gles,
            allocator: GbmAllocator::new(gbm, GbmBufferFlags::RENDERING),
            node,
            size,
            target: None,
        })
    }

    pub(crate) fn select_output(&mut self, session: SessionOutput) -> Result<(), G2gError> {
        let pass = match session.format {
            GpuFormat::Rgba => OutputPass::Rgba {
                modifiers: rgba_modifiers(&self.gles)?,
                buffers: Vec::new(),
            },
            GpuFormat::Nv12 => OutputPass::Nv12 {
                pass: nv12_pass(&mut self.gles, self.size)?,
                buffers: Vec::new(),
            },
        };
        self.target = Some(OutputTarget {
            output: session.output,
            pass,
        });
        Ok(())
    }

    pub(crate) fn dmabuf_feedback(&self) -> Result<DmabufFeedback, G2gError> {
        DmabufFeedbackBuilder::new(self.node.dev_id(), self.gles.dmabuf_formats())
            .build()
            .map_err(|error| hardware_error("building the dma-buf feedback", error))
    }

    pub(crate) fn render(
        &mut self,
        output: &Output,
        space: &Space<Window>,
        drag_icon: Option<(&WlSurface, Point<i32, Physical>)>,
        damage_tracker: &mut OutputDamageTracker,
    ) -> Result<Picture, G2gError> {
        let size = self.size;
        let target = self.target.as_mut().ok_or(G2gError::NotConfigured)?;
        match &mut target.pass {
            OutputPass::Rgba { modifiers, buffers } => {
                let index = match reusable(buffers)? {
                    Some(index) => index,
                    None => {
                        let targets = allocate(
                            &mut self.allocator,
                            size.width,
                            size.height,
                            OUTPUT_FOURCC,
                            modifiers,
                        )?;
                        let shared = share(&targets)?;
                        buffers.push(OutputBuffer { targets, shared });
                        buffers.len() - 1
                    }
                };
                let buffer = &mut buffers[index];
                let mut framebuffer = self
                    .gles
                    .bind(&mut buffer.targets)
                    .map_err(|error| hardware_error("binding an output buffer", error))?;
                composite(
                    &mut self.gles,
                    &mut framebuffer,
                    output,
                    space,
                    drag_icon,
                    damage_tracker,
                )?;
                match target.output {
                    GpuOutput::DmaBuf => Ok(Picture::DmaBuf(buffer.shared.clone())),
                    GpuOutput::System => {
                        read_framebuffer(&mut self.gles, &framebuffer, size).map(Picture::System)
                    }
                }
            }
            OutputPass::Nv12 { pass, buffers } => {
                let index = match reusable(buffers)? {
                    Some(index) => index,
                    None => {
                        buffers.push(allocate_nv12(&mut self.allocator, size)?);
                        buffers.len() - 1
                    }
                };
                let buffer = &mut buffers[index];
                {
                    let mut framebuffer = self
                        .gles
                        .bind(&mut pass.composite)
                        .map_err(|error| hardware_error("binding the composite texture", error))?;
                    composite(
                        &mut self.gles,
                        &mut framebuffer,
                        output,
                        space,
                        drag_icon,
                        damage_tracker,
                    )?;
                }
                let planes = &mut buffer.targets;
                let full = (size.width, size.height);
                let half = (size.width / 2, size.height / 2);
                convert(
                    &mut self.gles,
                    &mut planes.luma,
                    &pass.composite,
                    &pass.luma_program,
                    full,
                )?;
                convert(
                    &mut self.gles,
                    &mut planes.chroma,
                    &pass.composite,
                    &pass.chroma_program,
                    half,
                )?;
                match target.output {
                    GpuOutput::DmaBuf => Ok(Picture::DmaBuf(buffer.shared.clone())),
                    GpuOutput::System => read_nv12(&buffer.shared, size).map(Picture::System),
                }
            }
        }
    }
}

fn reusable<T>(buffers: &[OutputBuffer<T>]) -> Result<Option<usize>, G2gError> {
    if let Some(index) = buffers
        .iter()
        .position(|buffer| buffer.shared.share_count() == 1)
    {
        return Ok(Some(index));
    }
    if buffers.len() == MAX_OUTPUT_BUFFERS {
        return Err(hardware_error(
            "downstream holds every output buffer",
            MAX_OUTPUT_BUFFERS,
        ));
    }
    Ok(None)
}

fn allocate_nv12(
    allocator: &mut GbmAllocator<DrmDeviceFd>,
    size: DisplaySize,
) -> Result<OutputBuffer<Nv12Planes>, G2gError> {
    let DisplaySize { width, height } = size;
    // one linear R8 buffer holds both planes, the chroma rows after the luma rows at the same stride
    let whole = allocate(
        allocator,
        width,
        height + height / 2,
        LUMA_FOURCC,
        &[Modifier::Linear],
    )?;
    let shared = share(&whole)?;
    let chroma_offset = shared.offset + shared.stride * height;
    let luma = plane_view(&shared, (width, height), LUMA_FOURCC, shared.offset)?;
    let chroma = plane_view(
        &shared,
        (width / 2, height / 2),
        CHROMA_FOURCC,
        chroma_offset,
    )?;
    Ok(OutputBuffer {
        targets: Nv12Planes { luma, chroma },
        shared,
    })
}

fn composite(
    gles: &mut GlesRenderer,
    framebuffer: &mut GlesTarget<'_>,
    output: &Output,
    space: &Space<Window>,
    drag_icon: Option<(&WlSurface, Point<i32, Physical>)>,
    damage_tracker: &mut OutputDamageTracker,
) -> Result<(), G2gError> {
    let drag_icon = drag_icon_elements(gles, drag_icon);
    let result = render_output::<_, WaylandSurfaceRenderElement<GlesRenderer>, _, _>(
        output,
        gles,
        framebuffer,
        1.0,
        0,
        [space],
        &drag_icon,
        damage_tracker,
        BACKGROUND,
    )
    .map_err(|error| hardware_error("rendering the output", error))?;
    result
        .sync
        .wait()
        .map_err(|error| hardware_error("waiting for the render", error))
}

fn convert(
    gles: &mut GlesRenderer,
    target: &mut Dmabuf,
    composite: &GlesTexture,
    program: &GlesTexProgram,
    (width, height): (u32, u32),
) -> Result<(), G2gError> {
    let source_size = composite.size();
    let target_size = (
        i32::try_from(width).map_err(|_| G2gError::CapsMismatch)?,
        i32::try_from(height).map_err(|_| G2gError::CapsMismatch)?,
    );
    let destination = Rectangle::from_size(target_size.into());
    let mut framebuffer = gles
        .bind(target)
        .map_err(|error| hardware_error("binding an output plane", error))?;
    let mut frame = gles
        .render(&mut framebuffer, target_size.into(), Transform::Normal)
        .map_err(|error| hardware_error("starting a plane pass", error))?;
    frame
        .render_texture_from_to(
            composite,
            Rectangle::from_size(source_size.to_f64()),
            destination,
            &[destination],
            &[],
            Transform::Normal,
            1.0,
            Some(program),
            &[],
        )
        .map_err(|error| hardware_error("drawing an output plane", error))?;
    frame
        .finish()
        .map_err(|error| hardware_error("finishing a plane pass", error))?
        .wait()
        .map_err(|error| hardware_error("waiting for a plane pass", error))
}

fn rgba_modifiers(gles: &GlesRenderer) -> Result<Vec<Modifier>, G2gError> {
    let renderable: Vec<Modifier> = renderable_modifiers(gles, OUTPUT_FOURCC)?;
    if renderable.contains(&Modifier::Linear) {
        return Ok(Vec::from([Modifier::Linear]));
    }
    if renderable.is_empty() {
        return Err(hardware_error("renderable modifiers", OUTPUT_FOURCC));
    }
    Ok(renderable)
}

fn renderable_modifiers(gles: &GlesRenderer, fourcc: Fourcc) -> Result<Vec<Modifier>, G2gError> {
    Ok(Bind::<Dmabuf>::supported_formats(gles)
        .ok_or_else(|| hardware_error("listing the renderable formats", fourcc))?
        .iter()
        .filter(|format| format.code == fourcc && format.modifier != Modifier::Invalid)
        .map(|format| format.modifier)
        .collect())
}

fn nv12_pass(gles: &mut GlesRenderer, size: DisplaySize) -> Result<Nv12Pass, G2gError> {
    for fourcc in [LUMA_FOURCC, CHROMA_FOURCC] {
        if !renderable_modifiers(gles, fourcc)?.contains(&Modifier::Linear) {
            return Err(hardware_error(
                "the render node renders no linear plane",
                fourcc,
            ));
        }
    }
    let composite_size = (
        i32::try_from(size.width).map_err(|_| G2gError::CapsMismatch)?,
        i32::try_from(size.height).map_err(|_| G2gError::CapsMismatch)?,
    );
    let composite =
        Offscreen::<GlesTexture>::create_buffer(gles, OUTPUT_FOURCC, composite_size.into())
            .map_err(|error| hardware_error("allocating the composite texture", error))?;
    let luma_program = gles
        .compile_custom_texture_shader(include_str!("shaders/luma.frag"), &[])
        .map_err(|error| hardware_error("compiling the luma shader", error))?;
    let chroma_program = gles
        .compile_custom_texture_shader(include_str!("shaders/chroma.frag"), &[])
        .map_err(|error| hardware_error("compiling the chroma shader", error))?;
    Ok(Nv12Pass {
        composite,
        luma_program,
        chroma_program,
    })
}

fn allocate(
    allocator: &mut GbmAllocator<DrmDeviceFd>,
    width: u32,
    height: u32,
    fourcc: Fourcc,
    modifiers: &[Modifier],
) -> Result<Dmabuf, G2gError> {
    allocator
        .create_buffer(width, height, fourcc, modifiers)
        .map_err(|error| hardware_error("allocating an output buffer", error))?
        .export()
        .map_err(|error| hardware_error("exporting an output buffer", error))
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

fn plane_view(
    shared: &OwnedDmaBuf,
    (width, height): (u32, u32),
    fourcc: Fourcc,
    offset: u32,
) -> Result<Dmabuf, G2gError> {
    // SAFETY: the fd is a live dma-buf owned by `shared` for this call.
    let fd = unsafe { std::os::fd::BorrowedFd::borrow_raw(shared.as_raw()) }
        .try_clone_to_owned()
        .map_err(|error| hardware_error("duplicating an output buffer", error))?;
    let size = (
        i32::try_from(width).map_err(|_| G2gError::CapsMismatch)?,
        i32::try_from(height).map_err(|_| G2gError::CapsMismatch)?,
    );
    let mut builder = Dmabuf::builder(size, fourcc, Modifier::Linear, DmabufFlags::empty());
    builder.add_plane(fd, offset, shared.stride);
    builder
        .build()
        .ok_or_else(|| hardware_error("building an output plane view", fourcc))
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

fn read_nv12(shared: &OwnedDmaBuf, size: DisplaySize) -> Result<Vec<u8>, G2gError> {
    let mapping = DmaBufReadMap::read(shared)?;
    let bytes = mapping.as_slice();
    let stride = shared.stride as usize;
    let row_bytes = size.width as usize;
    let rows = size.height as usize * 3 / 2;
    let mut picture = Vec::with_capacity(row_bytes * rows);
    for row in 0..rows {
        let start = row * stride;
        let pixels = bytes
            .get(start..start + row_bytes)
            .ok_or_else(|| hardware_error("output buffer length", bytes.len()))?;
        picture.extend_from_slice(pixels);
    }
    Ok(picture)
}
