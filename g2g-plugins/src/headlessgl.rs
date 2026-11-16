// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::string::String;

use khronos_egl as egl;

/// Mesa's display-server-free EGL platform (`EGL_MESA_platform_surfaceless`).
const PLATFORM_SURFACELESS_MESA: egl::Enum = 0x31DD;

/// An offscreen EGL context: the surfaceless platform when the EGL client
/// offers it, else the default display, and a pbuffer surface unless the
/// driver supports a surfaceless (`EGL_NO_SURFACE`) make-current.
pub(crate) struct Headless {
    pub(crate) egl: crate::egl_library::Egl,
    display: egl::Display,
    surface: Option<egl::Surface>,
    context: egl::Context,
    /// How the context came up, for the test's evidence line.
    pub(crate) how: String,
}

impl Headless {
    /// Another `glow::Context` over the same current EGL context: each
    /// [`GlState`] owns the one it was built with, and the EGL display does
    /// not survive being torn down and brought up again inside one process,
    /// so a test rendering twice reloads the entry points instead.
    pub(crate) fn gl(&self) -> glow::Context {
        // SAFETY: this thread's current context resolves the GL ES entry
        // points glow asks for by name.
        unsafe {
            glow::Context::from_loader_function(|s| match self.egl.get_proc_address(s) {
                Some(p) => p as *const core::ffi::c_void,
                None => core::ptr::null(),
            })
        }
    }
}

impl Drop for Headless {
    fn drop(&mut self) {
        let _ = self.egl.make_current(self.display, None, None, None);
        if let Some(s) = self.surface {
            let _ = self.egl.destroy_surface(self.display, s);
        }
        let _ = self.egl.destroy_context(self.display, self.context);
        let _ = self.egl.terminate(self.display);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HeadlessApi {
    GlEs3,
    // what GtkGLArea and NSOpenGLView hand an application
    DesktopCore32,
}

/// Bring up an offscreen context of `api`, or return why it was not possible
/// (no EGL device on the host: the test then skips).
pub(crate) fn headless_context(api: HeadlessApi) -> Result<(Headless, glow::Context), String> {
    let egl = crate::egl_library::load()?;
    let client_exts = egl
        .query_string(None, egl::EXTENSIONS)
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();

    let (display, mut how) = if client_exts.contains("EGL_MESA_platform_surfaceless") {
        // SAFETY: the surfaceless platform takes no native display handle,
        // so EGL_DEFAULT_DISPLAY (null) is the value the extension requires.
        let d = unsafe {
            egl.get_platform_display(
                PLATFORM_SURFACELESS_MESA,
                egl::DEFAULT_DISPLAY,
                &[egl::ATTRIB_NONE],
            )
        }
        .map_err(|e| alloc::format!("eglGetPlatformDisplay(surfaceless): {e:?}"))?;
        (d, String::from("EGL_MESA_platform_surfaceless"))
    } else {
        // SAFETY: EGL_DEFAULT_DISPLAY is a valid native display handle for
        // eglGetDisplay on every platform.
        let d = unsafe { egl.get_display(egl::DEFAULT_DISPLAY) }
            .ok_or("eglGetDisplay(EGL_DEFAULT_DISPLAY) failed")?;
        (d, String::from("EGL default display"))
    };

    let version = egl
        .initialize(display)
        .map_err(|e| alloc::format!("eglInitialize: {e:?}"))?;
    let (bound_api, renderable, context_attribs) = match api {
        HeadlessApi::GlEs3 => (
            egl::OPENGL_ES_API,
            egl::OPENGL_ES3_BIT,
            alloc::vec![egl::CONTEXT_MAJOR_VERSION, 3, egl::NONE],
        ),
        HeadlessApi::DesktopCore32 => (
            egl::OPENGL_API,
            egl::OPENGL_BIT,
            alloc::vec![
                egl::CONTEXT_MAJOR_VERSION,
                3,
                egl::CONTEXT_MINOR_VERSION,
                2,
                egl::CONTEXT_OPENGL_PROFILE_MASK,
                egl::CONTEXT_OPENGL_CORE_PROFILE_BIT,
                egl::NONE,
            ],
        ),
    };
    egl.bind_api(bound_api)
        .map_err(|e| alloc::format!("eglBindAPI: {e:?}"))?;
    let vendor = egl
        .query_string(Some(display), egl::VENDOR)
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let display_exts = egl
        .query_string(Some(display), egl::EXTENSIONS)
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    how = alloc::format!("{how}, EGL {}.{} vendor {vendor}", version.0, version.1);

    let config_attribs = [
        egl::SURFACE_TYPE,
        egl::PBUFFER_BIT,
        egl::RENDERABLE_TYPE,
        renderable,
        egl::RED_SIZE,
        8,
        egl::GREEN_SIZE,
        8,
        egl::BLUE_SIZE,
        8,
        egl::ALPHA_SIZE,
        8,
        egl::NONE,
    ];
    let config = egl
        .choose_first_config(display, &config_attribs)
        .map_err(|e| alloc::format!("eglChooseConfig: {e:?}"))?
        .ok_or(alloc::format!("no {api:?} pbuffer config"))?;

    let context = egl
        .create_context(display, config, None, &context_attribs)
        .map_err(|e| alloc::format!("eglCreateContext: {e:?}"))?;

    // The render target is an FBO either way; a surfaceless make-current
    // needs EGL_KHR_surfaceless_context, so fall back to a 1x1 pbuffer.
    let surface = if display_exts.contains("EGL_KHR_surfaceless_context") {
        how.push_str(", surfaceless context");
        None
    } else {
        how.push_str(", pbuffer surface");
        Some(
            egl.create_pbuffer_surface(
                display,
                config,
                &[egl::WIDTH, 1, egl::HEIGHT, 1, egl::NONE],
            )
            .map_err(|e| alloc::format!("eglCreatePbufferSurface: {e:?}"))?,
        )
    };
    egl.make_current(display, surface, surface, Some(context))
        .map_err(|e| alloc::format!("eglMakeCurrent: {e:?}"))?;

    // SAFETY: the context just made current on this thread resolves the GL ES
    // entry points glow asks for by name.
    let gl = unsafe {
        glow::Context::from_loader_function(|s| match egl.get_proc_address(s) {
            Some(p) => p as *const core::ffi::c_void,
            None => core::ptr::null(),
        })
    };

    Ok((
        Headless {
            egl,
            display,
            surface,
            context,
            how,
        },
        gl,
    ))
}
