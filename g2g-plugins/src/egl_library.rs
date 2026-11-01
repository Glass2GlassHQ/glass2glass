// libEGL is loaded when a GL sink starts, so a binary built with the GL sinks
// still starts on a machine without it.

use alloc::string::String;

use khronos_egl as egl;

const LIBEGL: &str = "libEGL.so.1";

pub(crate) type Egl = egl::DynamicInstance<egl::EGL1_5>;

pub(crate) fn load() -> Result<Egl, String> {
    // SAFETY: libEGL runs no initialisation that needs anything from the caller.
    unsafe { Egl::load_required_from_filename(LIBEGL) }
        .map_err(|error| alloc::format!("{LIBEGL} with EGL 1.5 is not installed: {error}"))
}

#[cfg(feature = "gl-sink")]
pub(crate) fn installed() -> Result<(), String> {
    load().map(|_| ())
}
