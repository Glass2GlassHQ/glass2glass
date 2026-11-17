// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Shared GL ES render state for the EGL display sinks.
//!
//! Builds a GL ES 3 program plus the textures for one pixel layout (`GlMode`)
//! and a fullscreen quad, then per frame gets the pixels into those textures and
//! draws. NV12 uses the Y (`R8`) + interleaved UV (`RG8`) pair and a convert
//! shader carrying the caps colorimetry's weights; RGBA uses one `RGBA8` texture
//! and a passthrough shader.
//! Pixels arrive either from CUDA device memory (`upload_and_draw`, the CUDA-GL
//! interop registered lazily on the first frame) or from a system-memory slice
//! (`upload_system_and_draw`).
//!
//! The caller owns the EGL context and the *present* (Wayland `eglSwapBuffers`
//! for `glwindow`, GBM lock + DRM page-flip for
//! `cudakmssink`); everything up to and including the `glDrawArrays` is
//! identical and lives here.
//!
//! Compiled when any of the GL sink features is on (all pull `glow`).

use core::mem::size_of;

use alloc::format;
use alloc::string::{String, ToString};

use glow::HasContext;

use g2g_core::{Colorimetry, G2gError, MemoryDomain, RawVideoFormat};

use crate::display_aspect::{letterbox, ViewportRectangle};
use crate::filled_rectangles::{rectangle_vertices, FilledRectangle};
use crate::videoconvert::display::{display_sink_caps, DisplayInput, NV12_AND_RGBA};
use crate::yuvmatrix::YuvToRgbWeights;

#[cfg(any(feature = "cuda-gl", feature = "cuda-kms"))]
use crate::cuda::{make_context_current, CudaGlInterop};
#[cfg(any(feature = "cuda-gl", feature = "cuda-kms"))]
use g2g_core::memory::OwnedCudaBuffer;

/// GLSL ES 1.00 vertex shader: pass the texcoords through and position a
/// fullscreen quad. Paired with the fragment shaders below.
pub(crate) const VERTEX_SHADER: &str = "\
attribute vec2 a_pos;
attribute vec2 a_uv;
uniform float u_flip_y;
varying vec2 v_uv;
void main() {
    v_uv = a_uv;
    gl_Position = vec4(a_pos.x, a_pos.y * u_flip_y, 0.0, 1.0);
}
";

// desktop GL 3.2 core has no GLSL ES 1.00, these map its keywords onto GLSL 1.50
const DESKTOP_VERTEX_PRELUDE: &str = "#version 150
#define attribute in
#define varying out
";
const DESKTOP_FRAGMENT_PRELUDE: &str = "#version 150
#define varying in
#define texture2D texture
out vec4 g2g_frag_color;
#define gl_FragColor g2g_frag_color
";

/// GLSL ES 1.00 fragment shader: sample the NV12 luma (`R8`) and interleaved
/// chroma (`RG8`) textures and convert YCbCr -> RGB with `colorimetry`'s matrix
/// and range. The sampler pair is verbatim from DESIGN-C3-cuda.md Appendix A;
/// the weights are compiled in, so a mid-stream colorimetry change rebuilds the
/// program.
pub(crate) fn fragment_shader_nv12(colorimetry: Colorimetry) -> String {
    let w = YuvToRgbWeights::new(colorimetry);
    format!(
        "\
precision mediump float;
varying vec2 v_uv;
uniform sampler2D y_tex;
uniform sampler2D uv_tex;
void main() {{
    float y = texture2D(y_tex, v_uv).r;
    vec2  c = texture2D(uv_tex, v_uv).rg;
    y = {luma_gain:?} * (y - {luma_floor:?});
    float cb = c.x - 0.5;
    float cr = c.y - 0.5;
    float r = y + ({red_from_cr:?}) * cr;
    float g = y + ({green_from_cb:?}) * cb + ({green_from_cr:?}) * cr;
    float b = y + ({blue_from_cb:?}) * cb;
    gl_FragColor = vec4(r, g, b, 1.0);
}}
",
        luma_gain = w.luma_gain,
        luma_floor = w.luma_floor,
        red_from_cr = w.red_from_cr,
        green_from_cb = w.green_from_cb,
        green_from_cr = w.green_from_cr,
        blue_from_cb = w.blue_from_cb,
    )
}

const OVERLAY_VERTEX_SHADER: &str = "\
attribute vec2 a_pos;
attribute vec4 a_color;
uniform float u_flip_y;
varying vec4 v_color;
void main() {
    v_color = a_color;
    gl_Position = vec4(a_pos.x, a_pos.y * u_flip_y, 0.0, 1.0);
}
";

const OVERLAY_FRAGMENT_SHADER: &str = "\
precision mediump float;
varying vec4 v_color;
void main() {
    gl_FragColor = v_color;
}
";

/// GLSL ES 1.00 fragment shader for the already-RGBA path: straight texture
/// fetch, no convert.
pub(crate) const FRAGMENT_SHADER_RGBA: &str = "\
precision mediump float;
varying vec2 v_uv;
uniform sampler2D rgba_tex;
void main() {
    gl_FragColor = texture2D(rgba_tex, v_uv);
}
";

/// Pixel layout the state is built for. Fixed at build time from the negotiated
/// caps: it picks the program and the texture set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GlMode {
    Nv12,
    /// Only the system-memory sink negotiates RGBA; the CUDA sinks are NV12-only.
    Rgba,
}

/// The textures of one [`GlMode`], with the sampler uniform locations queried
/// once at link rather than every frame.
enum Textures {
    Nv12 {
        y_tex: glow::Texture,
        uv_tex: glow::Texture,
        y_loc: Option<glow::UniformLocation>,
        uv_loc: Option<glow::UniformLocation>,
    },
    Rgba {
        tex: glow::Texture,
        loc: Option<glow::UniformLocation>,
    },
}

/// GL render state, built once the EGL context is current. Holds the program,
/// its textures, the fullscreen-quad buffer, and (lazily, on the first frame,
/// once the decoder's CUDA context is known) the CUDA-GL interop.
pub(crate) struct GlState {
    gl: glow::Context,
    program: glow::Program,
    /// Vertex attribute locations, queried once at link.
    pos_loc: u32,
    uv_loc: u32,
    flip_y_loc: Option<glow::UniformLocation>,
    textures: Textures,
    vbo: glow::Buffer,
    // core profile contexts draw nothing without a bound vertex array
    vertex_array: glow::VertexArray,
    overlay: OverlayProgram,
    width: u32,
    height: u32,
    /// Registered on the first frame, when `OwnedCudaBuffer::context` is known.
    #[cfg(any(feature = "cuda-gl", feature = "cuda-kms"))]
    interop: Option<CudaGlInterop>,
    /// True once the decoder's CUDA context has been pushed current here.
    #[cfg(any(feature = "cuda-gl", feature = "cuda-kms"))]
    cuda_current: bool,
}

#[derive(Debug)]
struct OverlayProgram {
    program: glow::Program,
    vbo: glow::Buffer,
    pos_loc: u32,
    color_loc: u32,
    flip_y_loc: Option<glow::UniformLocation>,
}

/// # Safety
/// A GL context must be current.
unsafe fn link_for(
    gl: &glow::Context,
    vertex_src: &str,
    fragment_src: &str,
) -> Result<glow::Program, alloc::boxed::Box<dyn std::error::Error>> {
    // SAFETY: the caller guarantees a current GL context.
    unsafe {
        if gl.version().is_embedded {
            return link_program(gl, vertex_src, fragment_src);
        }
        link_program(
            gl,
            &(String::from(DESKTOP_VERTEX_PRELUDE) + vertex_src),
            &(String::from(DESKTOP_FRAGMENT_PRELUDE) + fragment_src),
        )
    }
}

impl GlState {
    /// Compile the shaders for `mode`, link the program, create the
    /// fullscreen-quad buffer and the mode's textures, allocated at the plane
    /// dimensions ready to be written (NV12: luma `R8` full-res + chroma `RG8`
    /// half-res; RGBA: one full-res `RGBA8`).
    ///
    /// # Safety
    /// `gl` must wrap a current GL ES 3 context.
    pub(crate) unsafe fn build(
        gl: glow::Context,
        width: u32,
        height: u32,
        mode: GlMode,
        colorimetry: Colorimetry,
    ) -> Result<GlState, alloc::boxed::Box<dyn std::error::Error>> {
        // SAFETY: the caller guarantees a current GL ES 3 context.
        unsafe {
            let fragment = match mode {
                GlMode::Nv12 => fragment_shader_nv12(colorimetry),
                GlMode::Rgba => String::from(FRAGMENT_SHADER_RGBA),
            };
            let program = link_for(&gl, VERTEX_SHADER, &fragment)?;
            let overlay_program = link_for(&gl, OVERLAY_VERTEX_SHADER, OVERLAY_FRAGMENT_SHADER)?;
            let overlay = OverlayProgram {
                program: overlay_program,
                vbo: gl.create_buffer().map_err(|e| e.to_string())?,
                pos_loc: gl
                    .get_attrib_location(overlay_program, "a_pos")
                    .unwrap_or(0),
                color_loc: gl
                    .get_attrib_location(overlay_program, "a_color")
                    .unwrap_or(1),
                flip_y_loc: gl.get_uniform_location(overlay_program, "u_flip_y"),
            };
            let pos_loc = gl.get_attrib_location(program, "a_pos").unwrap_or(0);
            let uv_loc = gl.get_attrib_location(program, "a_uv").unwrap_or(1);
            let flip_y_loc = gl.get_uniform_location(program, "u_flip_y");
            let vertex_array = gl.create_vertex_array().map_err(|e| e.to_string())?;

            // Fullscreen quad: two triangles, interleaved (x, y, u, v). Flip V so
            // the top row of the frame maps to the top of the window.
            #[rustfmt::skip]
            let verts: [f32; 24] = [
                -1.0, -1.0, 0.0, 1.0,
                 1.0, -1.0, 1.0, 1.0,
                 1.0,  1.0, 1.0, 0.0,
                -1.0, -1.0, 0.0, 1.0,
                 1.0,  1.0, 1.0, 0.0,
                -1.0,  1.0, 0.0, 0.0,
            ];
            let vbo = gl.create_buffer().map_err(|e| e.to_string())?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytemuck_cast(&verts), glow::STATIC_DRAW);

            let textures = match mode {
                GlMode::Nv12 => {
                    let cw = width.div_ceil(2);
                    let ch = height.div_ceil(2);
                    Textures::Nv12 {
                        y_tex: make_texture(&gl, glow::R8 as i32, glow::RED, width, height)?,
                        uv_tex: make_texture(&gl, glow::RG8 as i32, glow::RG, cw, ch)?,
                        y_loc: gl.get_uniform_location(program, "y_tex"),
                        uv_loc: gl.get_uniform_location(program, "uv_tex"),
                    }
                }
                GlMode::Rgba => Textures::Rgba {
                    tex: make_texture(&gl, glow::RGBA8 as i32, glow::RGBA, width, height)?,
                    loc: gl.get_uniform_location(program, "rgba_tex"),
                },
            };

            Ok(GlState {
                gl,
                program,
                pos_loc,
                uv_loc,
                flip_y_loc,
                textures,
                vbo,
                vertex_array,
                overlay,
                width,
                height,
                #[cfg(any(feature = "cuda-gl", feature = "cuda-kms"))]
                interop: None,
                #[cfg(any(feature = "cuda-gl", feature = "cuda-kms"))]
                cuda_current: false,
            })
        }
    }

    /// Upload the decoded NV12 planes into the GL textures via CUDA (lazily making
    /// the decoder's context current and registering the textures on the first
    /// frame), then draw the fullscreen quad through the NV12->RGB shader. The
    /// caller presents (swap / flip) afterwards.
    #[cfg(any(feature = "cuda-gl", feature = "cuda-kms"))]
    pub(crate) fn upload_and_draw(&mut self, buf: &OwnedCudaBuffer) -> Result<(), G2gError> {
        let (y_tex, uv_tex) = match &self.textures {
            Textures::Nv12 { y_tex, uv_tex, .. } => (*y_tex, *uv_tex),
            Textures::Rgba { .. } => return Err(G2gError::CapsMismatch),
        };
        // Lazily make the decoder's CUDA context current on this thread and
        // register the textures with CUDA, now that the context is known.
        if !self.cuda_current {
            // SAFETY: the worker owns this thread; `buf.context` is the ffmpeg CUDA
            // context the frame's pointers are valid in.
            unsafe { make_context_current(buf.context)? };
            self.cuda_current = true;
        }
        if self.interop.is_none() {
            let y = y_tex.0.get();
            let uv = uv_tex.0.get();
            // SAFETY: both textures are live GL_TEXTURE_2D names allocated in
            // `build`; the CUDA context is current (above).
            self.interop = Some(unsafe { CudaGlInterop::register(y, uv)? });
        }

        buf.keep_alive().wait_until_ready()?;
        // SAFETY: textures registered, CUDA context current, planes valid.
        unsafe { self.interop.as_ref().unwrap().upload(buf)? };

        self.draw(self.whole_frame(), UPRIGHT_ON_DEFAULT_FRAMEBUFFER);
        Ok(())
    }

    /// Upload one system-memory frame into the GL textures (`glTexSubImage2D`)
    /// and draw it. `bytes` is the packed frame in the built mode's layout: NV12
    /// = the `width * height` luma plane followed by the interleaved chroma
    /// plane, RGBA = `width * height * 4` bytes.
    #[cfg(feature = "gl-sink")]
    pub(crate) fn upload_system_and_draw(&mut self, bytes: &[u8]) -> Result<(), G2gError> {
        self.upload_system(bytes)?;
        self.draw(self.whole_frame(), UPRIGHT_ON_DEFAULT_FRAMEBUFFER);
        Ok(())
    }

    #[cfg(any(feature = "gl-sink", feature = "cuda-gl", feature = "cuda-kms"))]
    fn whole_frame(&self) -> ViewportRectangle {
        ViewportRectangle::whole(self.width, self.height)
    }

    fn upload_system(&mut self, bytes: &[u8]) -> Result<(), G2gError> {
        let (w, h) = (self.width as usize, self.height as usize);
        // SAFETY: the GL context is current on this thread for the worker's life;
        // every slice passed below is bounds-checked against the texture extent
        // first, so GL never reads past the frame.
        unsafe {
            let gl = &self.gl;
            // Rows are tightly packed; the default 4-byte unpack alignment would
            // mis-stride an R8 / RG8 plane whose row length is not a multiple of 4.
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            match &self.textures {
                Textures::Nv12 { y_tex, uv_tex, .. } => {
                    let nv12 = g2g_core::RawVideoFormat::Nv12;
                    let (uv_at, _, _) =
                        crate::pixel::tight_plane(nv12, 1, w, h).ok_or(G2gError::CapsMismatch)?;
                    let frame_bytes = crate::pixel::frame_byte_size(nv12, self.width, self.height);
                    if bytes.len() < frame_bytes {
                        return Err(G2gError::CapsMismatch);
                    }
                    let (cw, ch) = crate::pixel::chroma_420_size(w, h);
                    sub_image(gl, *y_tex, glow::RED, w, h, &bytes[..uv_at]);
                    sub_image(gl, *uv_tex, glow::RG, cw, ch, &bytes[uv_at..frame_bytes]);
                }
                Textures::Rgba { tex, .. } => {
                    let size = w.saturating_mul(h).saturating_mul(4);
                    if bytes.len() < size {
                        return Err(G2gError::CapsMismatch);
                    }
                    sub_image(gl, *tex, glow::RGBA, w, h, &bytes[..size]);
                }
            }
        }
        Ok(())
    }

    fn draw(&mut self, viewport: ViewportRectangle, flip_y: bool) {
        // SAFETY: the GL context is current on this thread for the worker's life.
        unsafe {
            let gl = &self.gl;
            gl.clear_color(0.0, 0.0, 0.0, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.viewport(
                viewport.x as i32,
                viewport.y as i32,
                viewport.width as i32,
                viewport.height as i32,
            );
            gl.use_program(Some(self.program));
            gl.uniform_1_f32(self.flip_y_loc.as_ref(), if flip_y { 1.0 } else { -1.0 });
            gl.bind_vertex_array(Some(self.vertex_array));

            match &self.textures {
                Textures::Nv12 {
                    y_tex,
                    uv_tex,
                    y_loc,
                    uv_loc,
                } => {
                    gl.active_texture(glow::TEXTURE0);
                    gl.bind_texture(glow::TEXTURE_2D, Some(*y_tex));
                    gl.uniform_1_i32(y_loc.as_ref(), 0);
                    gl.active_texture(glow::TEXTURE1);
                    gl.bind_texture(glow::TEXTURE_2D, Some(*uv_tex));
                    gl.uniform_1_i32(uv_loc.as_ref(), 1);
                }
                Textures::Rgba { tex, loc } => {
                    gl.active_texture(glow::TEXTURE0);
                    gl.bind_texture(glow::TEXTURE_2D, Some(*tex));
                    gl.uniform_1_i32(loc.as_ref(), 0);
                }
            }

            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));
            let pos = self.pos_loc;
            let uv = self.uv_loc;
            let stride = 4 * size_of::<f32>() as i32;
            gl.enable_vertex_attrib_array(pos);
            gl.vertex_attrib_pointer_f32(pos, 2, glow::FLOAT, false, stride, 0);
            gl.enable_vertex_attrib_array(uv);
            gl.vertex_attrib_pointer_f32(
                uv,
                2,
                glow::FLOAT,
                false,
                stride,
                2 * size_of::<f32>() as i32,
            );

            gl.draw_arrays(glow::TRIANGLES, 0, 6);
        }
    }

    fn draw_rectangles(&mut self, vertices: &[f32], (width, height): (u32, u32), flip_y: bool) {
        let overlay = &self.overlay;
        let stride = (crate::filled_rectangles::FLOATS_PER_VERTEX * size_of::<f32>()) as i32;
        let vertex_count = (vertices.len() / crate::filled_rectangles::FLOATS_PER_VERTEX) as i32;
        // SAFETY: the GL context is current on this thread for the worker's life.
        unsafe {
            let gl = &self.gl;
            gl.viewport(0, 0, width as i32, height as i32);
            gl.use_program(Some(overlay.program));
            gl.uniform_1_f32(overlay.flip_y_loc.as_ref(), if flip_y { 1.0 } else { -1.0 });
            gl.bind_vertex_array(Some(self.vertex_array));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(overlay.vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck_cast(vertices),
                glow::STREAM_DRAW,
            );
            gl.enable_vertex_attrib_array(overlay.pos_loc);
            gl.vertex_attrib_pointer_f32(overlay.pos_loc, 2, glow::FLOAT, false, stride, 0);
            gl.enable_vertex_attrib_array(overlay.color_loc);
            gl.vertex_attrib_pointer_f32(
                overlay.color_loc,
                4,
                glow::FLOAT,
                false,
                stride,
                2 * size_of::<f32>() as i32,
            );
            gl.enable(glow::BLEND);
            gl.blend_func_separate(
                glow::SRC_ALPHA,
                glow::ONE_MINUS_SRC_ALPHA,
                glow::ONE,
                glow::ONE_MINUS_SRC_ALPHA,
            );
            gl.draw_arrays(glow::TRIANGLES, 0, vertex_count);
            gl.disable(glow::BLEND);
            gl.disable_vertex_attrib_array(overlay.color_loc);
        }
    }

    #[cfg(any(feature = "cuda-gl", feature = "gl-sink"))]
    pub(crate) fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// The GL context the state renders with, for a caller that needs its own
    /// calls (the headless render test's framebuffer + readback).
    #[cfg(all(test, feature = "gl-sink"))]
    pub(crate) fn gl(&self) -> &glow::Context {
        &self.gl
    }

    /// # Safety
    /// The context the state was built in must be current.
    unsafe fn destroy(self) {
        // SAFETY: the caller guarantees the context these names belong to is current.
        unsafe {
            let gl = &self.gl;
            gl.delete_program(self.program);
            gl.delete_buffer(self.vbo);
            gl.delete_program(self.overlay.program);
            gl.delete_buffer(self.overlay.vbo);
            gl.delete_vertex_array(self.vertex_array);
            match self.textures {
                Textures::Nv12 { y_tex, uv_tex, .. } => {
                    gl.delete_texture(y_tex);
                    gl.delete_texture(uv_tex);
                }
                Textures::Rgba { tex, .. } => gl.delete_texture(tex),
            }
        }
    }
}

// the quad's texture coordinates put the frame's top row at the top of a window's framebuffer
#[cfg(any(feature = "gl-sink", feature = "cuda-gl", feature = "cuda-kms"))]
const UPRIGHT_ON_DEFAULT_FRAMEBUFFER: bool = true;

pub fn accepted_caps() -> g2g_core::CapsSet {
    display_sink_caps(NV12_AND_RGBA)
}

#[derive(Debug)]
pub struct DisplayFrame {
    pub caps: g2g_core::Caps,
    // None when the frame already arrived in a format the renderers draw
    pub converted: Option<alloc::boxed::Box<[u8]>>,
}

pub fn convert_for_display(caps: &g2g_core::Caps, bytes: &[u8]) -> Result<DisplayFrame, G2gError> {
    let input = DisplayInput::from_caps(caps, NV12_AND_RGBA)?;
    let converted = input.convert_frame(bytes)?;
    let g2g_core::Caps::RawVideo {
        width,
        height,
        framerate,
        interlace,
        colorimetry,
        pixel_aspect,
        ..
    } = caps.clone()
    else {
        unreachable!("DisplayInput takes raw video caps only");
    };
    let presented = g2g_core::Caps::RawVideo {
        format: input.presented,
        width,
        height,
        framerate,
        interlace,
        colorimetry,
        pixel_aspect,
    };
    Ok(DisplayFrame {
        caps: presented,
        converted,
    })
}

type GlLoader = alloc::boxed::Box<dyn Fn(&str) -> *const core::ffi::c_void>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlTarget {
    pub framebuffer: u32,
    pub width: u32,
    pub height: u32,
    pub flip_y: bool,
}

pub struct GlVideoRenderer {
    loader: GlLoader,
    input: Option<DisplayInput>,
    state: Option<(DisplayInput, GlState)>,
}

impl core::fmt::Debug for GlVideoRenderer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GlVideoRenderer")
            .field("input", &self.input)
            .finish_non_exhaustive()
    }
}

impl GlVideoRenderer {
    pub fn new(loader: impl Fn(&str) -> *const core::ffi::c_void + 'static) -> Self {
        Self {
            loader: alloc::boxed::Box::new(loader),
            input: None,
            state: None,
        }
    }

    pub fn configure(&mut self, absolute_caps: &g2g_core::Caps) -> Result<(), G2gError> {
        self.input = Some(DisplayInput::from_caps(absolute_caps, NV12_AND_RGBA)?);
        Ok(())
    }

    /// # Safety
    /// The GL context `loader` resolves entry points for must be current, with
    /// `target.framebuffer` a framebuffer of it.
    pub unsafe fn draw(
        &mut self,
        frame: &MemoryDomain,
        target: GlTarget,
    ) -> Result<ViewportRectangle, G2gError> {
        let input = self.input.ok_or(G2gError::NotConfigured)?;
        let bytes = frame.as_system_slice().ok_or(G2gError::UnsupportedDomain)?;
        let converted = input.convert_frame(bytes)?;
        let bytes = converted.as_deref().unwrap_or(bytes);
        if self.state.as_ref().map(|(built_for, _)| *built_for) != Some(input) {
            // SAFETY: the caller guarantees the context is current.
            unsafe { self.destroy_state() };
            let mode = match input.presented {
                RawVideoFormat::Nv12 => GlMode::Nv12,
                _ => GlMode::Rgba,
            };
            // SAFETY: the caller guarantees the context `loader` resolves for is current.
            let state = unsafe {
                let gl = glow::Context::from_loader_function(|name| (self.loader)(name));
                GlState::build(gl, input.width, input.height, mode, input.colorimetry)
            }
            .map_err(|_| G2gError::Hardware(g2g_core::HardwareError::Other))?;
            self.state = Some((input, state));
        }
        let (_, state) = self.state.as_mut().expect("built above");
        state.upload_system(bytes)?;
        let viewport = letterbox(
            input.width,
            input.height,
            input.pixel_aspect,
            target.width,
            target.height,
        );
        // SAFETY: the caller guarantees the framebuffer belongs to the current context.
        unsafe {
            state.gl.bind_framebuffer(
                glow::FRAMEBUFFER,
                core::num::NonZeroU32::new(target.framebuffer).map(glow::NativeFramebuffer),
            );
        }
        state.draw(viewport, target.flip_y);
        Ok(viewport)
    }

    /// # Safety
    /// The GL context `loader` resolves entry points for must be current, with
    /// `target.framebuffer` a framebuffer of it.
    pub unsafe fn draw_rectangles(
        &mut self,
        rectangles: &[FilledRectangle],
        target: GlTarget,
    ) -> Result<(), G2gError> {
        let (_, state) = self.state.as_mut().ok_or(G2gError::NotConfigured)?;
        let vertices = rectangle_vertices(rectangles, (target.width, target.height));
        // SAFETY: the caller guarantees the framebuffer belongs to the current context.
        unsafe {
            state.gl.bind_framebuffer(
                glow::FRAMEBUFFER,
                core::num::NonZeroU32::new(target.framebuffer).map(glow::NativeFramebuffer),
            );
        }
        state.draw_rectangles(&vertices, (target.width, target.height), target.flip_y);
        Ok(())
    }

    /// # Safety
    /// The GL context `loader` resolves entry points for must be current.
    unsafe fn destroy_state(&mut self) {
        if let Some((_, state)) = self.state.take() {
            // SAFETY: the caller guarantees the context is current.
            unsafe { state.destroy() };
        }
    }

    /// # Safety
    /// The GL context `loader` resolves entry points for must be current.
    pub unsafe fn destroy(mut self) {
        // SAFETY: forwarded from the caller.
        unsafe { self.destroy_state() };
    }
}

/// `glTexSubImage2D` a tightly-packed plane over the whole texture.
///
/// # Safety
/// A GL context must be current, `tex` must be a live `GL_TEXTURE_2D` allocated
/// at `w` x `h` in `format`, and `pixels` must hold the whole extent.
unsafe fn sub_image(
    gl: &glow::Context,
    tex: glow::Texture,
    format: u32,
    w: usize,
    h: usize,
    pixels: &[u8],
) {
    // SAFETY: the caller guarantees the context, the texture and the extent.
    unsafe {
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        gl.tex_sub_image_2d(
            glow::TEXTURE_2D,
            0,
            0,
            0,
            w as i32,
            h as i32,
            format,
            glow::UNSIGNED_BYTE,
            glow::PixelUnpackData::Slice(Some(pixels)),
        );
    }
}

/// Allocate a 2D texture with the given internal/source format at `w` x `h`,
/// `LINEAR` filtered and clamped, with no initial pixel data (CUDA or
/// `glTexSubImage2D` writes it).
///
/// # Safety
/// A GL context must be current.
unsafe fn make_texture(
    gl: &glow::Context,
    internal_format: i32,
    format: u32,
    w: u32,
    h: u32,
) -> Result<glow::Texture, alloc::boxed::Box<dyn std::error::Error>> {
    // SAFETY: the caller guarantees a current GL context.
    unsafe {
        let tex = gl.create_texture().map_err(|e| e.to_string())?;
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MIN_FILTER,
            glow::LINEAR as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MAG_FILTER,
            glow::LINEAR as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_S,
            glow::CLAMP_TO_EDGE as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_T,
            glow::CLAMP_TO_EDGE as i32,
        );
        // glow 0.17 `tex_image_2d` takes the pixel source as
        // `PixelUnpackData::Slice(Option<&[u8]>)`; `None` allocates storage
        // without uploading (CUDA / a later sub-image writes the pixels).
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            internal_format,
            w as i32,
            h as i32,
            0,
            format,
            glow::UNSIGNED_BYTE,
            glow::PixelUnpackData::Slice(None),
        );
        Ok(tex)
    }
}

/// Compile + link the vertex and fragment shaders into a program.
///
/// # Safety
/// A GL context must be current.
unsafe fn link_program(
    gl: &glow::Context,
    vertex_src: &str,
    fragment_src: &str,
) -> Result<glow::Program, alloc::boxed::Box<dyn std::error::Error>> {
    // SAFETY: the caller guarantees a current GL context.
    unsafe {
        let program = gl.create_program().map_err(|e| e.to_string())?;
        let shaders = [
            (glow::VERTEX_SHADER, vertex_src),
            (glow::FRAGMENT_SHADER, fragment_src),
        ];
        let mut compiled = alloc::vec::Vec::new();
        for (kind, src) in shaders {
            let shader = gl.create_shader(kind).map_err(|e| e.to_string())?;
            gl.shader_source(shader, src);
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                return Err(gl.get_shader_info_log(shader).into());
            }
            gl.attach_shader(program, shader);
            compiled.push(shader);
        }
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            return Err(gl.get_program_info_log(program).into());
        }
        for shader in compiled {
            gl.detach_shader(program, shader);
            gl.delete_shader(shader);
        }
        Ok(program)
    }
}

/// Reinterpret an `f32` slice as the `&[u8]` GL wants, without pulling in the
/// `bytemuck` crate for one call. The vertex array is `'static`-lifetime local
/// and tightly packed, so the cast is sound.
fn bytemuck_cast(verts: &[f32]) -> &[u8] {
    // SAFETY: `f32` has no padding and any bit pattern is a valid `u8`; the
    // resulting slice covers exactly the same bytes.
    unsafe { core::slice::from_raw_parts(verts.as_ptr() as *const u8, size_of_val(verts)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaders_declare_the_nv12_sampler_pair() {
        // Lock the Appendix A contract the CUDA upload side relies on: a
        // full-res luma sampler and a half-res interleaved chroma sampler.
        let shader = fragment_shader_nv12(Colorimetry::UNKNOWN);
        assert!(shader.contains("uniform sampler2D y_tex"));
        assert!(shader.contains("uniform sampler2D uv_tex"));
        // Vertex shader feeds the fragment shader's texcoord varying.
        assert!(VERTEX_SHADER.contains("varying vec2 v_uv"));
        assert!(shader.contains("varying vec2 v_uv"));
    }

    /// Every weight is written as a float literal, whatever the colorimetry:
    /// GLSL ES 1.00 rejects `y - 0` (an int) where a float belongs, which is
    /// what a full-range stream's zero luma floor would print as.
    #[test]
    fn full_range_weights_stay_float_literals() {
        let shader = fragment_shader_nv12(Colorimetry {
            range: g2g_core::ColorRange::Full,
            ..Colorimetry::BT709
        });
        assert!(shader.contains("(y - 0.0)"), "{shader}");
        assert!(shader.contains("y = 1.0 *"), "{shader}");
        // At the same range, the matrix still reaches the shader.
        assert_ne!(
            fragment_shader_nv12(Colorimetry::BT709),
            fragment_shader_nv12(Colorimetry::BT601)
        );
    }
}

#[cfg(all(test, target_os = "linux", feature = "gl-sink"))]
mod host_target_tests {
    use super::*;
    use crate::headlessgl::{headless_context, HeadlessApi};
    use g2g_core::memory::SystemSlice;
    use g2g_core::{Caps, Dim, Interlace, PixelAspect, Rate};

    const RED: [u8; 4] = [255, 0, 0, 255];
    const BLUE: [u8; 4] = [0, 0, 255, 255];
    const BLACK: [u8; 4] = [0, 0, 0, 255];
    const FRAME_WIDTH: u32 = 4;
    const FRAME_HEIGHT: u32 = 2;
    // the 2:1 frame letterboxes into the middle two rows unscaled, so no texel is filtered
    const TARGET_SIDE: u32 = 4;

    fn red_over_blue() -> MemoryDomain {
        let pixels: alloc::vec::Vec<u8> = [RED; FRAME_WIDTH as usize]
            .into_iter()
            .chain([BLUE; FRAME_WIDTH as usize])
            .flatten()
            .collect();
        MemoryDomain::System(SystemSlice::from_boxed(pixels.into_boxed_slice()))
    }

    fn rgba_caps() -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(FRAME_WIDTH),
            height: Dim::Fixed(FRAME_HEIGHT),
            framerate: Rate::Any,
            interlace: Interlace::Any,
            colorimetry: Colorimetry::UNKNOWN,
            pixel_aspect: PixelAspect::Any,
        }
    }

    // the first column of each row, bottom row first as glReadPixels returns them
    fn draw_into_framebuffer(
        api: HeadlessApi,
        flip_y: bool,
        overlay: &[FilledRectangle],
    ) -> Option<alloc::vec::Vec<[u8; 4]>> {
        let _egl = crate::EGL_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (_headless, gl) = headless_context(api).ok()?;
        let egl = crate::egl_library::load().expect("loaded once already");
        let mut renderer = GlVideoRenderer::new(move |name| match egl.get_proc_address(name) {
            Some(entry) => entry as *const core::ffi::c_void,
            None => core::ptr::null(),
        });
        renderer.configure(&rgba_caps()).unwrap();
        // SAFETY: headless_context made its context current on this thread, the
        // framebuffer and its texture are created in it and sized TARGET_SIDE square.
        unsafe {
            let color = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(color));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA8 as i32,
                TARGET_SIDE as i32,
                TARGET_SIDE as i32,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(None),
            );
            let framebuffer = gl.create_framebuffer().unwrap();
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(color),
                0,
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            let target = GlTarget {
                framebuffer: framebuffer.0.get(),
                width: TARGET_SIDE,
                height: TARGET_SIDE,
                flip_y,
            };
            renderer.draw(&red_over_blue(), target).unwrap();
            renderer.draw_rectangles(overlay, target).unwrap();
            let mut pixels = alloc::vec![0u8; (TARGET_SIDE * TARGET_SIDE * 4) as usize];
            gl.read_pixels(
                0,
                0,
                TARGET_SIDE as i32,
                TARGET_SIDE as i32,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut pixels)),
            );
            renderer.destroy();
            assert_eq!(gl.get_error(), glow::NO_ERROR, "{api:?}");
            Some(
                pixels
                    .chunks((TARGET_SIDE * 4) as usize)
                    .map(|row| row[..4].try_into().unwrap())
                    .collect(),
            )
        }
    }

    #[test]
    fn a_frame_lands_letterboxed_in_a_host_framebuffer_on_gles_and_desktop_core() {
        for api in [HeadlessApi::GlEs3, HeadlessApi::DesktopCore32] {
            let Some(upright) = draw_into_framebuffer(api, true, &[]) else {
                continue;
            };
            assert_eq!(upright, [BLACK, BLUE, RED, BLACK], "{api:?} flip_y");
            let unflipped = draw_into_framebuffer(api, false, &[]).unwrap();
            assert_eq!(unflipped, [BLACK, RED, BLUE, BLACK], "{api:?} no flip_y");
        }
    }

    const GREEN: [u8; 4] = [0, 255, 0, 255];

    #[test]
    fn rectangles_land_on_the_frame_rows_they_name_in_both_orientations() {
        // the frame's top row is the target's second row from the top
        let over_the_red_row = [FilledRectangle {
            x: 0.0,
            y: 1.0,
            width: TARGET_SIDE as f32,
            height: 1.0,
            rgba: GREEN,
        }];
        for api in [HeadlessApi::GlEs3, HeadlessApi::DesktopCore32] {
            let Some(upright) = draw_into_framebuffer(api, true, &over_the_red_row) else {
                continue;
            };
            assert_eq!(upright, [BLACK, BLUE, GREEN, BLACK], "{api:?} flip_y");
            let unflipped = draw_into_framebuffer(api, false, &over_the_red_row).unwrap();
            assert_eq!(unflipped, [BLACK, GREEN, BLUE, BLACK], "{api:?} no flip_y");
        }
    }
}
