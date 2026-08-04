//! Shared GL ES render state for the EGL display sinks.
//!
//! Builds a GL ES 3 program plus the textures for one pixel layout ([`GlMode`])
//! and a fullscreen quad, then per frame gets the pixels into those textures and
//! draws. NV12 uses the Y (`R8`) + interleaved UV (`RG8`) pair and the BT.601
//! convert shader; RGBA uses one `RGBA8` texture and a passthrough shader.
//! Pixels arrive either from CUDA device memory (`upload_and_draw`, the CUDA-GL
//! interop registered lazily on the first frame) or from a system-memory slice
//! (`upload_system_and_draw`).
//!
//! The caller owns the EGL context and the *present* (Wayland `eglSwapBuffers`
//! for [`crate::glwindow`], GBM lock + DRM page-flip for
//! [`crate::cudakmssink`]); everything up to and including the `glDrawArrays` is
//! identical and lives here.
//!
//! Compiled when any of the GL sink features is on (all pull `glow`).

use core::mem::size_of;

use alloc::string::ToString;

use glow::HasContext;

use g2g_core::G2gError;

#[cfg(any(feature = "cuda-gl", feature = "cuda-kms"))]
use crate::cuda::{make_context_current, CudaGlInterop};
#[cfg(any(feature = "cuda-gl", feature = "cuda-kms"))]
use g2g_core::memory::OwnedCudaBuffer;

/// GLSL ES 1.00 vertex shader: pass the texcoords through and position a
/// fullscreen quad. Paired with the fragment shaders below.
pub(crate) const VERTEX_SHADER: &str = "\
attribute vec2 a_pos;
attribute vec2 a_uv;
varying vec2 v_uv;
void main() {
    v_uv = a_uv;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
";

/// GLSL ES 1.00 fragment shader: sample the NV12 luma (`R8`) and interleaved
/// chroma (`RG8`) textures and convert BT.601 limited-range YCbCr -> RGB.
/// Verbatim from DESIGN-C3-cuda.md Appendix A (swap the matrix for BT.709 on
/// HD sources once a colour-metadata field exists on `Caps`).
pub(crate) const FRAGMENT_SHADER_NV12: &str = "\
precision mediump float;
varying vec2 v_uv;
uniform sampler2D y_tex;
uniform sampler2D uv_tex;
void main() {
    float y = texture2D(y_tex, v_uv).r;
    vec2  c = texture2D(uv_tex, v_uv).rg;
    y = 1.1643 * (y - 0.0625);
    float cb = c.x - 0.5;
    float cr = c.y - 0.5;
    float r = y + 1.5958 * cr;
    float g = y - 0.3917 * cb - 0.8129 * cr;
    float b = y + 2.0170 * cb;
    gl_FragColor = vec4(r, g, b, 1.0);
}
";

/// GLSL ES 1.00 fragment shader for the already-RGBA path: straight texture
/// fetch, no convert.
#[cfg(feature = "gl-sink")]
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
    #[cfg(feature = "gl-sink")]
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
    #[cfg(feature = "gl-sink")]
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
    textures: Textures,
    vbo: glow::Buffer,
    width: u32,
    height: u32,
    /// Registered on the first frame, when `OwnedCudaBuffer::context` is known.
    #[cfg(any(feature = "cuda-gl", feature = "cuda-kms"))]
    interop: Option<CudaGlInterop>,
    /// True once the decoder's CUDA context has been pushed current here.
    #[cfg(any(feature = "cuda-gl", feature = "cuda-kms"))]
    cuda_current: bool,
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
    ) -> Result<GlState, alloc::boxed::Box<dyn std::error::Error>> {
        // SAFETY: the caller guarantees a current GL ES 3 context.
        unsafe {
            let fragment = match mode {
                GlMode::Nv12 => FRAGMENT_SHADER_NV12,
                #[cfg(feature = "gl-sink")]
                GlMode::Rgba => FRAGMENT_SHADER_RGBA,
            };
            let program = link_program(&gl, VERTEX_SHADER, fragment)?;
            let pos_loc = gl.get_attrib_location(program, "a_pos").unwrap_or(0);
            let uv_loc = gl.get_attrib_location(program, "a_uv").unwrap_or(1);

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
                #[cfg(feature = "gl-sink")]
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
                textures,
                vbo,
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
            #[cfg(feature = "gl-sink")]
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

        // SAFETY: textures registered, CUDA context current, planes valid.
        unsafe { self.interop.as_ref().unwrap().upload(buf)? };

        self.draw();
        Ok(())
    }

    /// Upload one system-memory frame into the GL textures (`glTexSubImage2D`)
    /// and draw it. `bytes` is the packed frame in the built mode's layout: NV12
    /// = the `width * height` luma plane followed by the interleaved chroma
    /// plane, RGBA = `width * height * 4` bytes.
    #[cfg(feature = "gl-sink")]
    pub(crate) fn upload_system_and_draw(&mut self, bytes: &[u8]) -> Result<(), G2gError> {
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
                    let cw = w.div_ceil(2);
                    let ch = h.div_ceil(2);
                    let y_size = w * h;
                    let uv_size = 2 * cw * ch;
                    if bytes.len() < y_size + uv_size {
                        return Err(G2gError::CapsMismatch);
                    }
                    let (y_plane, rest) = bytes.split_at(y_size);
                    sub_image(gl, *y_tex, glow::RED, w, h, y_plane);
                    sub_image(gl, *uv_tex, glow::RG, cw, ch, &rest[..uv_size]);
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
        self.draw();
        Ok(())
    }

    /// Bind the program + textures and draw the fullscreen quad. The pixels must
    /// already be in the textures.
    fn draw(&mut self) {
        // SAFETY: the GL context is current on this thread for the worker's life.
        unsafe {
            let gl = &self.gl;
            gl.viewport(0, 0, self.width as i32, self.height as i32);
            gl.clear_color(0.0, 0.0, 0.0, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.use_program(Some(self.program));

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
                #[cfg(feature = "gl-sink")]
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

    /// The GL context the state renders with, for a caller that needs its own
    /// calls (the headless render test's framebuffer + readback).
    #[cfg(all(test, feature = "gl-sink"))]
    pub(crate) fn gl(&self) -> &glow::Context {
        &self.gl
    }
}

/// `glTexSubImage2D` a tightly-packed plane over the whole texture.
///
/// # Safety
/// A GL context must be current, `tex` must be a live `GL_TEXTURE_2D` allocated
/// at `w` x `h` in `format`, and `pixels` must hold the whole extent.
#[cfg(feature = "gl-sink")]
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
        assert!(FRAGMENT_SHADER_NV12.contains("uniform sampler2D y_tex"));
        assert!(FRAGMENT_SHADER_NV12.contains("uniform sampler2D uv_tex"));
        // Vertex shader feeds the fragment shader's texcoord varying.
        assert!(VERTEX_SHADER.contains("varying vec2 v_uv"));
        assert!(FRAGMENT_SHADER_NV12.contains("varying vec2 v_uv"));
    }
}
