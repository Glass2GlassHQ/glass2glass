use alloc::string::String;
use core::ffi::{c_char, c_void};
use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};

use g2g_core::{G2gError, HardwareError};

const LIBAMDHIP: &str = "libamdhip64.so";
const HIP_ERROR_SHARED_OBJECT_INIT_FAILED: i32 = 303;
const HIP_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD: u32 = 1;
const HIP_EXTERNAL_MEMORY_DEDICATED: u32 = 0x1;

type HipExternalMemory = *mut c_void;

// `hipExternalMemoryHandleDesc`, its handle union written as the leading `int fd`
#[repr(C)]
struct HipExternalMemoryHandleDesc {
    type_: u32,
    _pad: u32,
    handle_fd: i32,
    _handle_rest: [u8; 12],
    size: u64,
    flags: u32,
    reserved: [u32; 16],
}

#[repr(C)]
struct HipExternalMemoryBufferDesc {
    offset: u64,
    size: u64,
    flags: u32,
    reserved: [u32; 16],
}

// sizes the C compiler gives the two structs on x86_64
const _: () = assert!(core::mem::size_of::<HipExternalMemoryHandleDesc>() == 104);
const _: () = assert!(core::mem::size_of::<HipExternalMemoryBufferDesc>() == 88);

crate::shared_library::shared_library! {
    struct LibAmdHip = LIBAMDHIP;
    when missing return HIP_ERROR_SHARED_OBJECT_INIT_FAILED;
    fn hip_set_device = "hipSetDevice"(device: i32) -> i32;
    fn hip_import_external_memory = "hipImportExternalMemory"(
        ext_mem_out: *mut HipExternalMemory,
        mem_handle_desc: *const HipExternalMemoryHandleDesc,
    ) -> i32;
    fn hip_external_memory_get_mapped_buffer = "hipExternalMemoryGetMappedBuffer"(
        device_pointer: *mut *mut c_void,
        ext_mem: HipExternalMemory,
        buffer_desc: *const HipExternalMemoryBufferDesc,
    ) -> i32;
    fn hip_destroy_external_memory = "hipDestroyExternalMemory"(ext_mem: HipExternalMemory) -> i32;
    fn hip_free = "hipFree"(pointer: *mut c_void) -> i32;
    fn hip_device_get_pci_bus_id = "hipDeviceGetPCIBusId"(
        pci_bus_id: *mut c_char,
        len: i32,
        device: i32,
    ) -> i32;
}

fn check(code: i32) -> Result<(), G2gError> {
    match code {
        0 => Ok(()),
        HIP_ERROR_SHARED_OBJECT_INIT_FAILED if LibAmdHip::get().is_none() => Err(
            G2gError::Hardware(HardwareError::LibraryNotFound(LIBAMDHIP)),
        ),
        code => Err(G2gError::Hardware(HardwareError::Hip(code))),
    }
}

#[derive(Debug)]
pub struct HipExternalBuffer {
    ext_mem: usize,
    device_pointer: u64,
    device_ordinal: i32,
}

impl HipExternalBuffer {
    /// # Safety
    /// `fd` must be an `OPAQUE_FD` export of a dedicated allocation of at least `size` bytes.
    pub unsafe fn import(fd: OwnedFd, size: u64, device_ordinal: i32) -> Result<Self, G2gError> {
        // SAFETY: HIP owns the fd once the import succeeds, and the import is
        // destroyed when mapping it fails.
        unsafe {
            check(hip_set_device(device_ordinal))?;
            let raw_fd = fd.into_raw_fd();
            let import_desc = HipExternalMemoryHandleDesc {
                type_: HIP_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
                _pad: 0,
                handle_fd: raw_fd,
                _handle_rest: [0; 12],
                size,
                flags: HIP_EXTERNAL_MEMORY_DEDICATED,
                reserved: [0; 16],
            };
            let mut ext_mem: HipExternalMemory = core::ptr::null_mut();
            if let Err(error) = check(hip_import_external_memory(&mut ext_mem, &import_desc)) {
                drop(OwnedFd::from_raw_fd(raw_fd));
                return Err(error);
            }
            let buffer_desc = HipExternalMemoryBufferDesc {
                offset: 0,
                size,
                flags: 0,
                reserved: [0; 16],
            };
            let mut device_pointer: *mut c_void = core::ptr::null_mut();
            let mapped = check(hip_external_memory_get_mapped_buffer(
                &mut device_pointer,
                ext_mem,
                &buffer_desc,
            ));
            if let Err(error) = mapped {
                hip_destroy_external_memory(ext_mem);
                return Err(error);
            }
            Ok(Self {
                ext_mem: ext_mem as usize,
                device_pointer: device_pointer as u64,
                device_ordinal,
            })
        }
    }

    pub fn device_pointer(&self) -> u64 {
        self.device_pointer
    }
}

impl Drop for HipExternalBuffer {
    fn drop(&mut self) {
        // SAFETY: the pointer and the import came from `import` on this device.
        unsafe {
            if hip_set_device(self.device_ordinal) == 0 {
                hip_free(self.device_pointer as *mut c_void);
                hip_destroy_external_memory(self.ext_mem as HipExternalMemory);
            }
        }
    }
}

pub fn hip_pci_bus_id(device_ordinal: i32) -> Result<String, G2gError> {
    // hip writes "domain:bus:device.function", 12 characters plus the terminator
    let mut text = [0u8; 16];
    // SAFETY: `text` is writable for the length passed.
    check(unsafe {
        hip_device_get_pci_bus_id(text.as_mut_ptr().cast(), text.len() as i32, device_ordinal)
    })?;
    let end = text
        .iter()
        .position(|&byte| byte == 0)
        .ok_or(G2gError::Hardware(HardwareError::Other))?;
    core::str::from_utf8(&text[..end])
        .map(String::from)
        .map_err(|_| G2gError::Hardware(HardwareError::Other))
}
