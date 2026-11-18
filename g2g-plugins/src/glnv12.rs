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
use crate::overlay_image::OverlayImage;
use crate::videoconvert::display::{display_sink_caps, DisplayInput, NV12_RGBA_AND_P010};
use crate::yuvmatrix::{dither_step, YuvToRgbWeights, DITHER_NOISE_SCALE, DITHER_NOISE_WEIGHTS};

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

/// GLSL ES 1.00 fragment shader: sample the luma (`R8`, or `R16` for P010) and
/// interleaved chroma (`RG8` / `RG16`) textures and convert YCbCr -> RGB with
/// `colorimetry`'s matrix and range. The sampler pair is verbatim from
/// DESIGN-C3-cuda.md Appendix A; the weights are compiled in, so a mid-stream
/// colorimetry change rebuilds the program.
pub(crate) fn fragment_shader_semi_planar(
    colorimetry: Colorimetry,
    format: RawVideoFormat,
) -> String {
    // P010 keeps each 10-bit code in the top bits of a 16-bit word, so it samples as code * 64 / 65535, not the code / 255 the weights take
    let (precision, sample_scale) = match format {
        RawVideoFormat::P010 => ("highp", 65535.0f32 / 65280.0),
        _ => ("mediump", 1.0f32),
    };
    let w = YuvToRgbWeights::new(colorimetry);
    format!(
        "\
precision {precision} float;
varying vec2 v_uv;
uniform sampler2D y_tex;
uniform sampler2D uv_tex;
uniform float u_dither_step;
void main() {{
    float y = texture2D(y_tex, v_uv).r * {sample_scale:?};
    vec2  c = texture2D(uv_tex, v_uv).rg * {sample_scale:?};
    y = {luma_gain:?} * (y - {luma_floor:?});
    float cb = c.x - {chroma_neutral:?};
    float cr = c.y - {chroma_neutral:?};
    float r = y + ({red_from_cr:?}) * cr;
    float g = y + ({green_from_cb:?}) * cb + ({green_from_cr:?}) * cr;
    float b = y + ({blue_from_cb:?}) * cb;
    highp float noise = fract({noise_scale:?} * fract(dot(gl_FragCoord.xy, vec2({noise_x:?}, {noise_y:?}))));
    gl_FragColor = vec4(vec3(r, g, b) + (noise - 0.5) * u_dither_step, 1.0);
}}
",
        luma_gain = w.luma_gain,
        luma_floor = w.luma_floor,
        chroma_neutral = w.chroma_neutral,
        red_from_cr = w.red_from_cr,
        green_from_cb = w.green_from_cb,
        green_from_cr = w.green_from_cr,
        blue_from_cb = w.blue_from_cb,
        noise_scale = DITHER_NOISE_SCALE,
        noise_x = DITHER_NOISE_WEIGHTS[0],
        noise_y = DITHER_NOISE_WEIGHTS[1],
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
    P010,
    /// Only the system-memory sink negotiates RGBA; the CUDA sinks are NV12-only.
    Rgba,
}

/// The textures of one [`GlMode`], with the sampler uniform locations queried
/// once at link rather than every frame.
enum Textures {
    SemiPlanar {
        y_tex: glow::Texture,
        uv_tex: glow::Texture,
        y_loc: Option<glow::UniformLocation>,
        uv_loc: Option<glow::UniformLocation>,
        planes: SemiPlanarPlanes,
    },
    Rgba {
        tex: glow::Texture,
        loc: Option<glow::UniformLocation>,
    },
}

#[derive(Debug, Clone, Copy)]
struct SemiPlanarPlanes {
    format: RawVideoFormat,
    luma_internal: u32,
    chroma_internal: u32,
    sample_type: u32,
}

impl SemiPlanarPlanes {
    fn of(mode: GlMode) -> Option<Self> {
        let (format, luma_internal, chroma_internal, sample_type) = match mode {
            GlMode::Nv12 => (
                RawVideoFormat::Nv12,
                glow::R8,
                glow::RG8,
                glow::UNSIGNED_BYTE,
            ),
            GlMode::P010 => (
                RawVideoFormat::P010,
                glow::R16,
                glow::RG16,
                glow::UNSIGNED_SHORT,
            ),
            GlMode::Rgba => return None,
        };
        Some(Self {
            format,
            luma_internal,
            chroma_internal,
            sample_type,
        })
    }
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
    dither_step_loc: Option<glow::UniformLocation>,
    textures: Textures,
    vbo: glow::Buffer,
    // core profile contexts draw nothing without a bound vertex array
    vertex_array: glow::VertexArray,
    overlay: OverlayProgram,
    image: ImageLayer,
    width: u32,
    height: u32,
    /// Registered on the first frame, when `OwnedCudaBuffer::context` is known.
    #[cfg(any(feature = "cuda-gl", feature = "cuda-kms"))]
    interop: Option<CudaGlInterop>,
    /// True once the decoder's CUDA context has been pushed current here.
    #[cfg(any(feature = "cuda-gl", feature = "cuda-kms"))]
    cuda_current: bool,
}

#[derive(Debug, Clone, Copy)]
struct QuadProgram<'a> {
    pos_loc: u32,
    uv_loc: u32,
    flip_y_loc: Option<&'a glow::UniformLocation>,
}

#[derive(Debug)]
struct ImageLayer {
    program: glow::Program,
    pos_loc: u32,
    uv_loc: u32,
    flip_y_loc: Option<glow::UniformLocation>,
    tex_loc: Option<glow::UniformLocation>,
    uploaded: Option<UploadedImage>,
}

#[derive(Debug)]
struct UploadedImage {
    texture: glow::Texture,
    size: (u32, u32),
    generation: u64,
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
            let fragment = match SemiPlanarPlanes::of(mode) {
                Some(planes) => fragment_shader_semi_planar(colorimetry, planes.format),
                None => String::from(FRAGMENT_SHADER_RGBA),
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
            let image_program = link_for(&gl, VERTEX_SHADER, FRAGMENT_SHADER_RGBA)?;
            let image = ImageLayer {
                program: image_program,
                pos_loc: gl.get_attrib_location(image_program, "a_pos").unwrap_or(0),
                uv_loc: gl.get_attrib_location(image_program, "a_uv").unwrap_or(1),
                flip_y_loc: gl.get_uniform_location(image_program, "u_flip_y"),
                tex_loc: gl.get_uniform_location(image_program, "rgba_tex"),
                uploaded: None,
            };
            let pos_loc = gl.get_attrib_location(program, "a_pos").unwrap_or(0);
            let uv_loc = gl.get_attrib_location(program, "a_uv").unwrap_or(1);
            let flip_y_loc = gl.get_uniform_location(program, "u_flip_y");
            let dither_step_loc = gl.get_uniform_location(program, "u_dither_step");
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

            let textures = match SemiPlanarPlanes::of(mode) {
                Some(planes) => {
                    let cw = width.div_ceil(2);
                    let ch = height.div_ceil(2);
                    let SemiPlanarPlanes {
                        luma_internal,
                        chroma_internal,
                        sample_type,
                        ..
                    } = planes;
                    Textures::SemiPlanar {
                        y_tex: make_texture(
                            &gl,
                            luma_internal,
                            glow::RED,
                            sample_type,
                            width,
                            height,
                        )?,
                        uv_tex: make_texture(&gl, chroma_internal, glow::RG, sample_type, cw, ch)?,
                        y_loc: gl.get_uniform_location(program, "y_tex"),
                        uv_loc: gl.get_uniform_location(program, "uv_tex"),
                        planes,
                    }
                }
                None => Textures::Rgba {
                    tex: make_texture(
                        &gl,
                        glow::RGBA8,
                        glow::RGBA,
                        glow::UNSIGNED_BYTE,
                        width,
                        height,
                    )?,
                    loc: gl.get_uniform_location(program, "rgba_tex"),
                },
            };

            Ok(GlState {
                gl,
                program,
                pos_loc,
                uv_loc,
                flip_y_loc,
                dither_step_loc,
                textures,
                vbo,
                vertex_array,
                overlay,
                image,
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
            Textures::SemiPlanar { y_tex, uv_tex, .. } => (*y_tex, *uv_tex),
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

        self.draw(
            self.whole_frame(),
            UPRIGHT_ON_DEFAULT_FRAMEBUFFER,
            dither_step(None),
        );
        Ok(())
    }

    /// Upload one system-memory frame into the GL textures (`glTexSubImage2D`)
    /// and draw it. `bytes` is the packed frame in the built mode's layout: NV12
    /// = the `width * height` luma plane followed by the interleaved chroma
    /// plane, RGBA = `width * height * 4` bytes.
    #[cfg(feature = "gl-sink")]
    pub(crate) fn upload_system_and_draw(&mut self, bytes: &[u8]) -> Result<(), G2gError> {
        self.upload_system(bytes)?;
        self.draw(
            self.whole_frame(),
            UPRIGHT_ON_DEFAULT_FRAMEBUFFER,
            dither_step(None),
        );
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
                Textures::SemiPlanar {
                    y_tex,
                    uv_tex,
                    planes,
                    ..
                } => {
                    let (uv_at, _, _) = crate::pixel::tight_plane(planes.format, 1, w, h)
                        .ok_or(G2gError::CapsMismatch)?;
                    let frame_bytes =
                        crate::pixel::frame_byte_size(planes.format, self.width, self.height);
                    if bytes.len() < frame_bytes {
                        return Err(G2gError::CapsMismatch);
                    }
                    let (cw, ch) = crate::pixel::chroma_420_size(w, h);
                    let sample = planes.sample_type;
                    sub_image(gl, *y_tex, glow::RED, sample, w, h, &bytes[..uv_at]);
                    sub_image(
                        gl,
                        *uv_tex,
                        glow::RG,
                        sample,
                        cw,
                        ch,
                        &bytes[uv_at..frame_bytes],
                    );
                }
                Textures::Rgba { tex, .. } => {
                    let size = w.saturating_mul(h).saturating_mul(4);
                    if bytes.len() < size {
                        return Err(G2gError::CapsMismatch);
                    }
                    sub_image(
                        gl,
                        *tex,
                        glow::RGBA,
                        glow::UNSIGNED_BYTE,
                        w,
                        h,
                        &bytes[..size],
                    );
                }
            }
        }
        Ok(())
    }

    fn draw(&mut self, viewport: ViewportRectangle, flip_y: bool, dither_step: f32) {
        // SAFETY: the GL context is current on this thread for the worker's life.
        unsafe {
            let gl = &self.gl;
            gl.clear_color(0.0, 0.0, 0.0, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.use_program(Some(self.program));
            gl.uniform_1_f32(self.dither_step_loc.as_ref(), dither_step);

            match &self.textures {
                Textures::SemiPlanar {
                    y_tex,
                    uv_tex,
                    y_loc,
                    uv_loc,
                    ..
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
            let quad = QuadProgram {
                pos_loc: self.pos_loc,
                uv_loc: self.uv_loc,
                flip_y_loc: self.flip_y_loc.as_ref(),
            };
            self.draw_quad(quad, viewport, flip_y);
        }
    }

    /// # Safety
    /// The GL context must be current, with the quad's program in use and its textures bound.
    unsafe fn draw_quad(&self, quad: QuadProgram<'_>, viewport: ViewportRectangle, flip_y: bool) {
        // SAFETY: forwarded from the caller.
        unsafe {
            let gl = &self.gl;
            gl.viewport(
                viewport.x as i32,
                viewport.y as i32,
                viewport.width as i32,
                viewport.height as i32,
            );
            gl.uniform_1_f32(quad.flip_y_loc, if flip_y { 1.0 } else { -1.0 });
            gl.bind_vertex_array(Some(self.vertex_array));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));
            let stride = 4 * size_of::<f32>() as i32;
            gl.enable_vertex_attrib_array(quad.pos_loc);
            gl.vertex_attrib_pointer_f32(quad.pos_loc, 2, glow::FLOAT, false, stride, 0);
            gl.enable_vertex_attrib_array(quad.uv_loc);
            gl.vertex_attrib_pointer_f32(
                quad.uv_loc,
                2,
                glow::FLOAT,
                false,
                stride,
                2 * size_of::<f32>() as i32,
            );
            gl.draw_arrays(glow::TRIANGLES, 0, 6);
        }
    }

    fn draw_image(&mut self, image: &OverlayImage, flip_y: bool) -> Result<(), G2gError> {
        let (width, height) = (image.width as usize, image.height as usize);
        if image.rgba.len() < width.saturating_mul(height).saturating_mul(4) {
            return Err(G2gError::CapsMismatch);
        }
        let stale = self
            .image
            .uploaded
            .as_ref()
            .is_none_or(|uploaded| uploaded.generation != image.generation);
        // SAFETY: the GL context is current on this thread, and the pixels cover the texture extent checked above.
        unsafe {
            let gl = &self.gl;
            if stale {
                let resized = self
                    .image
                    .uploaded
                    .as_ref()
                    .is_none_or(|uploaded| uploaded.size != (image.width, image.height));
                if resized {
                    if let Some(old) = self.image.uploaded.take() {
                        gl.delete_texture(old.texture);
                    }
                    let texture = make_texture(
                        gl,
                        glow::RGBA8,
                        glow::RGBA,
                        glow::UNSIGNED_BYTE,
                        image.width,
                        image.height,
                    )
                    .map_err(|_| G2gError::Hardware(g2g_core::HardwareError::Other))?;
                    self.image.uploaded = Some(UploadedImage {
                        texture,
                        size: (image.width, image.height),
                        generation: image.generation,
                    });
                }
                let uploaded = self.image.uploaded.as_mut().expect("allocated above");
                uploaded.generation = image.generation;
                gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
                sub_image(
                    gl,
                    uploaded.texture,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    width,
                    height,
                    &image.rgba,
                );
            }
            let uploaded = self.image.uploaded.as_ref().expect("uploaded above");
            gl.use_program(Some(self.image.program));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(uploaded.texture));
            gl.uniform_1_i32(self.image.tex_loc.as_ref(), 0);
            gl.enable(glow::BLEND);
            gl.blend_func(glow::ONE, glow::ONE_MINUS_SRC_ALPHA);
            let quad = QuadProgram {
                pos_loc: self.image.pos_loc,
                uv_loc: self.image.uv_loc,
                flip_y_loc: self.image.flip_y_loc.as_ref(),
            };
            self.draw_quad(quad, image.placement, flip_y);
            gl.disable(glow::BLEND);
        }
        Ok(())
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
            gl.delete_program(self.image.program);
            if let Some(uploaded) = self.image.uploaded {
                gl.delete_texture(uploaded.texture);
            }
            gl.delete_vertex_array(self.vertex_array);
            match self.textures {
                Textures::SemiPlanar { y_tex, uv_tex, .. } => {
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
    display_sink_caps(NV12_RGBA_AND_P010)
}

#[derive(Debug)]
pub struct DisplayFrame {
    pub caps: g2g_core::Caps,
    // None when the frame already arrived in a format the renderers draw
    pub converted: Option<alloc::boxed::Box<[u8]>>,
}

pub fn convert_for_display(caps: &g2g_core::Caps, bytes: &[u8]) -> Result<DisplayFrame, G2gError> {
    let input = DisplayInput::from_caps(caps, NV12_RGBA_AND_P010)?;
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
    // mpv's MPV_RENDER_PARAM_DEPTH, the bits each colour channel of the framebuffer holds
    pub bits_per_channel: u32,
}

pub struct GlVideoRenderer {
    loader: GlLoader,
    input: Option<DisplayInput>,
    state: Option<BuiltState>,
}

struct BuiltState {
    configured: DisplayInput,
    drawn: DisplayInput,
    gl: GlState,
}

// GLES 3 has no 16-bit normalized textures without this extension
const NORM16_EXTENSION: &str = "GL_EXT_texture_norm16";

fn has_16_bit_textures(gl: &glow::Context) -> bool {
    !gl.version().is_embedded || gl.supported_extensions().contains(NORM16_EXTENSION)
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
        self.input = Some(DisplayInput::from_caps(absolute_caps, NV12_RGBA_AND_P010)?);
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
        if self.state.as_ref().map(|built| built.configured) != Some(input) {
            // SAFETY: the caller guarantees the context is current.
            unsafe { self.destroy_state() };
            // SAFETY: the caller guarantees the context `loader` resolves for is current.
            let gl = unsafe { glow::Context::from_loader_function(|name| (self.loader)(name)) };
            let drawn = match input.presented {
                RawVideoFormat::P010 if !has_16_bit_textures(&gl) => {
                    input.presented_as(RawVideoFormat::Nv12)
                }
                _ => input,
            };
            let mode = match drawn.presented {
                RawVideoFormat::Nv12 => GlMode::Nv12,
                RawVideoFormat::P010 => GlMode::P010,
                _ => GlMode::Rgba,
            };
            // SAFETY: the caller guarantees the context `loader` resolves for is current.
            let state =
                unsafe { GlState::build(gl, input.width, input.height, mode, input.colorimetry) }
                    .map_err(|_| G2gError::Hardware(g2g_core::HardwareError::Other))?;
            self.state = Some(BuiltState {
                configured: input,
                drawn,
                gl: state,
            });
        }
        let built = self.state.as_mut().expect("built above");
        let converted = built.drawn.convert_frame(bytes)?;
        let bytes = converted.as_deref().unwrap_or(bytes);
        let state = &mut built.gl;
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
        state.draw(
            viewport,
            target.flip_y,
            dither_step(Some(target.bits_per_channel)),
        );
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
        let state = &mut self.state.as_mut().ok_or(G2gError::NotConfigured)?.gl;
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
    /// The GL context `loader` resolves entry points for must be current, with
    /// `target.framebuffer` a framebuffer of it.
    pub unsafe fn draw_image(
        &mut self,
        image: &OverlayImage,
        target: GlTarget,
    ) -> Result<(), G2gError> {
        let state = &mut self.state.as_mut().ok_or(G2gError::NotConfigured)?.gl;
        // SAFETY: the caller guarantees the framebuffer belongs to the current context.
        unsafe {
            state.gl.bind_framebuffer(
                glow::FRAMEBUFFER,
                core::num::NonZeroU32::new(target.framebuffer).map(glow::NativeFramebuffer),
            );
        }
        state.draw_image(image, target.flip_y)
    }

    /// # Safety
    /// The GL context `loader` resolves entry points for must be current.
    unsafe fn destroy_state(&mut self) {
        if let Some(BuiltState { gl: state, .. }) = self.state.take() {
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

const CLIENT_STORAGE_EXTENSION: &str = "GL_APPLE_client_storage";
const UNPACK_CLIENT_STORAGE_APPLE: u32 = 0x85B2;

unsafe fn turn_off_client_storage(gl: &glow::Context) {
    if !gl.supported_extensions().contains(CLIENT_STORAGE_EXTENSION) {
        return;
    }
    // glTexImage2D rejects null pixels while client storage is on, and postkit's grok presenter leaves it on
    // SAFETY: the caller guarantees a current GL context.
    unsafe { gl.pixel_store_i32(UNPACK_CLIENT_STORAGE_APPLE, 0) };
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
    sample_type: u32,
    w: usize,
    h: usize,
    pixels: &[u8],
) {
    // SAFETY: the caller guarantees the context, the texture and the extent.
    unsafe {
        turn_off_client_storage(gl);
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        gl.tex_sub_image_2d(
            glow::TEXTURE_2D,
            0,
            0,
            0,
            w as i32,
            h as i32,
            format,
            sample_type,
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
    internal_format: u32,
    format: u32,
    sample_type: u32,
    w: u32,
    h: u32,
) -> Result<glow::Texture, alloc::boxed::Box<dyn std::error::Error>> {
    // SAFETY: the caller guarantees a current GL context.
    unsafe {
        turn_off_client_storage(gl);
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
            internal_format as i32,
            w as i32,
            h as i32,
            0,
            format,
            sample_type,
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
        let shader = fragment_shader_semi_planar(Colorimetry::UNKNOWN, RawVideoFormat::Nv12);
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
        let shader = fragment_shader_semi_planar(
            Colorimetry {
                range: g2g_core::ColorRange::Full,
                ..Colorimetry::BT709
            },
            RawVideoFormat::Nv12,
        );
        assert!(shader.contains("(y - 0.0)"), "{shader}");
        assert!(shader.contains("y = 1.0 *"), "{shader}");
        // At the same range, the matrix still reaches the shader.
        assert_ne!(
            fragment_shader_semi_planar(Colorimetry::BT709, RawVideoFormat::Nv12),
            fragment_shader_semi_planar(Colorimetry::BT601, RawVideoFormat::Nv12)
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
                bits_per_channel: EIGHT_BITS,
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

    // ten-bit luma codes one apart, which an 8-bit path draws up to two codes off
    const P010_LUMA_CODES: [u16; 8] = [400, 401, 402, 403, 404, 405, 406, 407];
    const P010_NEUTRAL_CHROMA: u16 = 512;
    // P010 stores each 10-bit code in the top bits of its 16-bit word
    const P010_SHIFT: u32 = 6;
    // BT.601 limited range at 10 bits, what untagged caps convert with
    const TEN_BIT_BLACK: f64 = 64.0;
    const TEN_BIT_LUMA_SPAN: f64 = 876.0;
    const TEN_BIT_MAX: u32 = 1023;
    const TEN_BITS: u32 = 10;
    const EIGHT_BITS: u32 = 8;

    fn p010_frame(luma_row: &[u16], rows: u32) -> (Caps, MemoryDomain) {
        let width = luma_row.len();
        let rows = rows as usize;
        let luma = luma_row.iter().copied().cycle().take(width * rows);
        let chroma = core::iter::repeat_n(P010_NEUTRAL_CHROMA, width * rows / 2);
        let bytes: alloc::vec::Vec<u8> = luma
            .chain(chroma)
            .flat_map(|code| (code << P010_SHIFT).to_le_bytes())
            .collect();
        let caps = Caps::RawVideo {
            format: RawVideoFormat::P010,
            width: Dim::Fixed(width as u32),
            height: Dim::Fixed(rows as u32),
            framerate: Rate::Any,
            interlace: Interlace::Any,
            colorimetry: Colorimetry::UNKNOWN,
            pixel_aspect: PixelAspect::Any,
        };
        let frame = MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice()));
        (caps, frame)
    }

    #[derive(Clone, Copy)]
    struct TextureTarget {
        internal_format: u32,
        format: u32,
        sample_type: u32,
        bytes_per_pixel: u32,
        bits_per_channel: u32,
    }

    // the target's pixels, read back in its own format, or None without a desktop GL context
    fn draw_p010_into_texture(
        luma_row: &[u16],
        rows: u32,
        texture: TextureTarget,
    ) -> Option<alloc::vec::Vec<u8>> {
        let _egl = crate::EGL_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (_headless, gl) = headless_context(HeadlessApi::DesktopCore32).ok()?;
        let egl = crate::egl_library::load().expect("loaded once already");
        let mut renderer = GlVideoRenderer::new(move |name| match egl.get_proc_address(name) {
            Some(entry) => entry as *const core::ffi::c_void,
            None => core::ptr::null(),
        });
        let (caps, frame) = p010_frame(luma_row, rows);
        renderer.configure(&caps).unwrap();
        let width = luma_row.len() as u32;
        // SAFETY: headless_context made its context current on this thread, and the
        // framebuffer and its texture are created in it at the frame's size.
        unsafe {
            let color = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(color));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                texture.internal_format as i32,
                width as i32,
                rows as i32,
                0,
                texture.format,
                texture.sample_type,
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
            let target = GlTarget {
                framebuffer: framebuffer.0.get(),
                width,
                height: rows,
                flip_y: true,
                bits_per_channel: texture.bits_per_channel,
            };
            renderer.draw(&frame, target).unwrap();
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
            let mut pixels = alloc::vec![0u8; (width * rows * texture.bytes_per_pixel) as usize];
            gl.read_pixels(
                0,
                0,
                width as i32,
                rows as i32,
                texture.format,
                texture.sample_type,
                glow::PixelPackData::Slice(Some(&mut pixels)),
            );
            renderer.destroy();
            assert_eq!(gl.get_error(), glow::NO_ERROR);
            Some(pixels)
        }
    }

    // a 10-bit luma code in the converted output, BT.601 limited range scaled to max_code
    fn converted_luma(luma: u16, max_code: u32) -> f64 {
        (f64::from(luma) - TEN_BIT_BLACK) / TEN_BIT_LUMA_SPAN * f64::from(max_code)
    }

    #[test]
    fn a_ten_bit_frame_keeps_its_depth_in_a_ten_bit_framebuffer() {
        let ten_bit_target = TextureTarget {
            internal_format: glow::RGB10_A2,
            format: glow::RGBA,
            sample_type: glow::UNSIGNED_INT_2_10_10_10_REV,
            bytes_per_pixel: 4,
            bits_per_channel: TEN_BITS,
        };
        let Some(pixels) = draw_p010_into_texture(&P010_LUMA_CODES, FRAME_HEIGHT, ten_bit_target)
        else {
            return;
        };
        let red_codes: alloc::vec::Vec<u32> = pixels
            .chunks(4)
            .take(P010_LUMA_CODES.len())
            .map(|pixel| u32::from_le_bytes(pixel.try_into().unwrap()) & TEN_BIT_MAX)
            .collect();
        for (column, (&luma, &red)) in P010_LUMA_CODES.iter().zip(&red_codes).enumerate() {
            let expected = converted_luma(luma, TEN_BIT_MAX);
            assert!(
                (f64::from(red) - expected).abs() <= 1.0,
                "column {column}: red {red}, expected {expected:.1} in {red_codes:?}"
            );
        }
    }

    // converts to 97.52 at 8 bits, which rounding alone draws as 98 everywhere
    const BETWEEN_EIGHT_BIT_CODES: u16 = 399;
    const DITHERED_SIDE: usize = 64;
    // rounding alone misses the exact value by up to half a code
    const DITHERED_MEAN_TOLERANCE: f64 = 0.1;
    const EIGHT_BIT_MAX: u32 = 255;

    #[test]
    fn a_ten_bit_frame_dithers_to_an_eight_bit_framebuffer() {
        let eight_bit_target = TextureTarget {
            internal_format: glow::RGBA8,
            format: glow::RGBA,
            sample_type: glow::UNSIGNED_BYTE,
            bytes_per_pixel: 4,
            bits_per_channel: EIGHT_BITS,
        };
        let Some(pixels) = draw_p010_into_texture(
            &[BETWEEN_EIGHT_BIT_CODES; DITHERED_SIDE],
            DITHERED_SIDE as u32,
            eight_bit_target,
        ) else {
            return;
        };
        let reds: alloc::vec::Vec<f64> =
            pixels.chunks(4).map(|pixel| f64::from(pixel[0])).collect();
        let mean = reds.iter().sum::<f64>() / reds.len() as f64;
        let exact = converted_luma(BETWEEN_EIGHT_BIT_CODES, EIGHT_BIT_MAX);
        assert!(
            (mean - exact).abs() < DITHERED_MEAN_TOLERANCE,
            "mean red {mean:.3}, exact {exact:.3}"
        );
    }

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
