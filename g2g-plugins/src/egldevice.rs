use alloc::boxed::Box;
use core::ffi::c_void;

use khronos_egl as egl;

pub(crate) const EGL_DEVICE_EXT: egl::Int = 0x322C;
const EGL_CUDA_DEVICE_NV: egl::Int = 0x323A;

type QueryDeviceAttrib = unsafe extern "system" fn(*mut c_void, egl::Int, *mut isize) -> u32;

fn query_device_attrib(
    egl: &egl::Instance<egl::Static>,
) -> Result<QueryDeviceAttrib, Box<dyn std::error::Error>> {
    let entry = egl
        .get_proc_address("eglQueryDeviceAttribEXT")
        .ok_or("EGL has no eglQueryDeviceAttribEXT")?;
    // SAFETY: this is the EGL_EXT_device_query entry point of that name, cast to its C signature.
    Ok(unsafe { core::mem::transmute::<extern "system" fn(), QueryDeviceAttrib>(entry) })
}

// None for an EGL device of a GPU without CUDA
fn cuda_ordinal(query_device_attrib: QueryDeviceAttrib, device: *mut c_void) -> Option<i32> {
    let mut ordinal: isize = -1;
    // SAFETY: `device` is an EGL device handle; a device without EGL_NV_device_cuda
    // returns EGL_FALSE and leaves `ordinal` alone.
    let found = unsafe { query_device_attrib(device, EGL_CUDA_DEVICE_NV, &mut ordinal) };
    if found == 0 {
        return None;
    }
    i32::try_from(ordinal).ok()
}

#[cfg(feature = "cuda-gl")]
const MAX_EGL_DEVICES: usize = 16;

#[cfg(feature = "cuda-gl")]
pub(crate) fn cuda_egl_device(
    egl: &egl::Instance<egl::Static>,
    cuda_device_id: i32,
) -> Result<*mut c_void, Box<dyn std::error::Error>> {
    type QueryDevices = unsafe extern "system" fn(egl::Int, *mut *mut c_void, *mut egl::Int) -> u32;
    let entry = egl
        .get_proc_address("eglQueryDevicesEXT")
        .ok_or("EGL has no eglQueryDevicesEXT")?;
    // SAFETY: this is the EGL_EXT_device_enumeration entry point of that name, cast to its C signature.
    let query_devices =
        unsafe { core::mem::transmute::<extern "system" fn(), QueryDevices>(entry) };
    let query_device_attrib = query_device_attrib(egl)?;
    let mut devices = [core::ptr::null_mut(); MAX_EGL_DEVICES];
    let mut count = 0;
    // SAFETY: `devices` has room for the MAX_EGL_DEVICES handles EGL may write.
    if unsafe {
        query_devices(
            MAX_EGL_DEVICES as egl::Int,
            devices.as_mut_ptr(),
            &mut count,
        )
    } == 0
    {
        return Err("eglQueryDevicesEXT failed".into());
    }
    devices[..count as usize]
        .iter()
        .copied()
        .find(|&device| cuda_ordinal(query_device_attrib, device) == Some(cuda_device_id))
        .ok_or_else(|| alloc::format!("no EGL device is CUDA device {cuda_device_id}").into())
}

#[cfg(feature = "cuda-kms")]
pub(crate) fn display_cuda_ordinal(
    egl: &egl::Instance<egl::Static>,
    display: egl::Display,
) -> Result<Option<i32>, Box<dyn std::error::Error>> {
    type QueryDisplayAttrib = unsafe extern "system" fn(*mut c_void, egl::Int, *mut isize) -> u32;
    let entry = egl
        .get_proc_address("eglQueryDisplayAttribEXT")
        .ok_or("EGL has no eglQueryDisplayAttribEXT")?;
    // SAFETY: this is the EGL_EXT_device_query entry point of that name, cast to its C signature.
    let query_display_attrib =
        unsafe { core::mem::transmute::<extern "system" fn(), QueryDisplayAttrib>(entry) };
    let mut device: isize = 0;
    // SAFETY: `display` is an initialized EGL display.
    if unsafe { query_display_attrib(display.as_ptr(), EGL_DEVICE_EXT, &mut device) } == 0 {
        return Err("eglQueryDisplayAttribEXT(EGL_DEVICE_EXT) failed".into());
    }
    Ok(cuda_ordinal(
        query_device_attrib(egl)?,
        device as *mut c_void,
    ))
}
