//! Shared NV12 GL ES render state for the CUDA-GL sinks.
//!
//! Builds a GL ES 3 program with the Y (R8) + interleaved UV (RG8) NV12 textures
//! and a fullscreen quad, registers them with CUDA once, and per frame copies the
//! decoded planes in (CUDA-GL interop) and draws the NV12->RGB shader. The caller
//! owns the EGL context and the *present* (Wayland `eglSwapBuffers` for
//! [`crate::cudaglsink`], GBM lock + DRM page-flip for [`crate::cudakmssink`]);
//! everything up to and including the `glDrawArrays` is identical and lives here.
//!
//! Compiled when either CUDA-GL sink feature is on (both pull `glow`).

use core::mem::size_of;

use alloc::string::ToString;

use glow::HasContext;

use g2g_core::memory::OwnedCudaBuffer;
use g2g_core::G2gError;

use crate::cuda::{make_context_current, CudaGlInterop, FRAGMENT_SHADER_NV12, VERTEX_SHADER};

/// GL render state, built once the EGL context is current. Holds the program,
/// the two NV12 textures, the fullscreen-quad buffer, and (lazily, on the first
/// frame, once the decoder's CUDA context is known) the CUDA-GL interop.
pub(crate) struct GlState {
    gl: glow::Context,
    program: glow::Program,
    /// Program input locations, queried once at link rather than every frame.
    y_tex_loc: Option<glow::UniformLocation>,
    uv_tex_loc: Option<glow::UniformLocation>,
    pos_loc: u32,
    uv_loc: u32,
    y_tex: glow::Texture,
    uv_tex: glow::Texture,
    vbo: glow::Buffer,
    width: u32,
    height: u32,
    /// Registered on the first frame, when `OwnedCudaBuffer::context` is known.
    interop: Option<CudaGlInterop>,
    /// True once the decoder's CUDA context has been pushed current here.
    cuda_current: bool,
}

impl GlState {
    /// Compile the NV12 shaders, link the program, create the fullscreen-quad
    /// buffer and the two NV12 textures (luma `R8` full-res, chroma `RG8`
    /// half-res), allocated at the plane dimensions ready for CUDA to write.
    ///
    /// # Safety
    /// `gl` must wrap a current GL ES 3 context.
    pub(crate) unsafe fn build(
        gl: glow::Context,
        width: u32,
        height: u32,
    ) -> Result<GlState, alloc::boxed::Box<dyn std::error::Error>> {
        // SAFETY: the caller guarantees a current GL ES 3 context.
        unsafe {
            let program = link_program(&gl, VERTEX_SHADER, FRAGMENT_SHADER_NV12)?;
            let y_tex_loc = gl.get_uniform_location(program, "y_tex");
            let uv_tex_loc = gl.get_uniform_location(program, "uv_tex");
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

            let cw = width.div_ceil(2);
            let ch = height.div_ceil(2);
            let y_tex = make_texture(&gl, glow::R8 as i32, glow::RED, width, height)?;
            let uv_tex = make_texture(&gl, glow::RG8 as i32, glow::RG, cw, ch)?;

            Ok(GlState {
                gl,
                program,
                y_tex_loc,
                uv_tex_loc,
                pos_loc,
                uv_loc,
                y_tex,
                uv_tex,
                vbo,
                width,
                height,
                interop: None,
                cuda_current: false,
            })
        }
    }

    /// Upload the decoded NV12 planes into the GL textures via CUDA (lazily making
    /// the decoder's context current and registering the textures on the first
    /// frame), then draw the fullscreen quad through the NV12->RGB shader. The
    /// caller presents (swap / flip) afterwards.
    pub(crate) fn upload_and_draw(&mut self, buf: &OwnedCudaBuffer) -> Result<(), G2gError> {
        // Lazily make the decoder's CUDA context current on this thread and
        // register the textures with CUDA, now that the context is known.
        if !self.cuda_current {
            // SAFETY: the worker owns this thread; `buf.context` is the ffmpeg CUDA
            // context the frame's pointers are valid in.
            unsafe { make_context_current(buf.context)? };
            self.cuda_current = true;
        }
        if self.interop.is_none() {
            let y = self.y_tex.0.get();
            let uv = self.uv_tex.0.get();
            // SAFETY: both textures are live GL_TEXTURE_2D names allocated in
            // `build`; the CUDA context is current (above).
            self.interop = Some(unsafe { CudaGlInterop::register(y, uv)? });
        }

        // SAFETY: textures registered, CUDA context current, planes valid.
        unsafe { self.interop.as_ref().unwrap().upload(buf)? };

        // SAFETY: the GL context is current on this thread for the worker's life.
        unsafe {
            let gl = &self.gl;
            gl.viewport(0, 0, self.width as i32, self.height as i32);
            gl.clear_color(0.0, 0.0, 0.0, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.use_program(Some(self.program));

            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.y_tex));
            gl.uniform_1_i32(self.y_tex_loc.as_ref(), 0);
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.uv_tex));
            gl.uniform_1_i32(self.uv_tex_loc.as_ref(), 1);

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
        Ok(())
    }
}

/// Allocate a 2D texture with the given internal/source format at `w` x `h`,
/// `LINEAR` filtered and clamped, with no initial pixel data (CUDA writes it).
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
        // without uploading (CUDA writes the pixels).
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
