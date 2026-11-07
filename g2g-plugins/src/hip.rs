use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::ffi::{c_char, c_void};
use std::ffi::CString;
use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};
use std::sync::mpsc::{self, Sender};
use std::sync::{Condvar, Mutex, OnceLock, PoisonError};

use g2g_core::memory::{HipKeepAlive, HipVendor, OwnedHipBuffer};
use g2g_core::{g2g_error, G2gError, HardwareError};

use crate::cuda_driver_ffi as cuda;

// rocm-hip-devel ships the unversioned libamdhip64.so, ROCm 7 and 6 runtimes only the sonames
const LIBAMDHIP: &str = "libamdhip64.so.7";
const LIBAMDHIP_ROCM6: &str = "libamdhip64.so.6";
const HIP_ERROR_SHARED_OBJECT_INIT_FAILED: i32 = 303;
const HIP_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD: u32 = 1;
const HIP_EXTERNAL_MEMORY_DEDICATED: u32 = 0x1;
// hipStreamNonBlocking and CU_STREAM_NON_BLOCKING share the value
const STREAM_NON_BLOCKING: u32 = 0x1;
// hipEventDisableTiming and CU_EVENT_DISABLE_TIMING share the value
const EVENT_DISABLE_TIMING: u32 = 0x2;
// hipErrorOutOfMemory and CUDA_ERROR_OUT_OF_MEMORY share the value
const OUT_OF_MEMORY: i32 = 2;

// libhiprtc and libnvrtc come with the HIP and CUDA toolkits, not with the drivers
const LIBHIPRTC: &str = "libhiprtc.so.7";
const LIBHIPRTC_ROCM6: &str = "libhiprtc.so.6";
const LIBNVRTC: &str = "libnvrtc.so.13";
const LIBNVRTC_CUDA12: &str = "libnvrtc.so.12";
// hiprtcResult and nvrtcResult share these values
const RTC_SUCCESS: i32 = 0;
const RTC_ERROR_INVALID_INPUT: i32 = 3;
const RTC_ERROR_INTERNAL_ERROR: i32 = 11;

type HipExternalMemory = *mut c_void;
type HipStreamHandle = *mut c_void;
type HipEventHandle = *mut c_void;
type HipModuleHandle = *mut c_void;
type HipFunctionHandle = *mut c_void;
type RtcProgram = *mut c_void;

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
    struct LibAmdHip = LIBAMDHIP, LIBAMDHIP_ROCM6;
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
    fn hip_malloc = "hipMalloc"(pointer: *mut *mut c_void, size: usize) -> i32;
    fn hip_host_malloc = "hipHostMalloc"(pointer: *mut *mut c_void, size: usize, flags: u32) -> i32;
    fn hip_host_free = "hipHostFree"(pointer: *mut c_void) -> i32;
    fn hip_stream_create_with_flags = "hipStreamCreateWithFlags"(stream: *mut HipStreamHandle, flags: u32) -> i32;
    fn hip_stream_destroy = "hipStreamDestroy"(stream: HipStreamHandle) -> i32;
    fn hip_stream_synchronize = "hipStreamSynchronize"(stream: HipStreamHandle) -> i32;
    fn hip_stream_wait_event = "hipStreamWaitEvent"(stream: HipStreamHandle, event: HipEventHandle, flags: u32) -> i32;
    fn hip_event_create_with_flags = "hipEventCreateWithFlags"(event: *mut HipEventHandle, flags: u32) -> i32;
    fn hip_event_record = "hipEventRecord"(event: HipEventHandle, stream: HipStreamHandle) -> i32;
    fn hip_event_synchronize = "hipEventSynchronize"(event: HipEventHandle) -> i32;
    fn hip_event_destroy = "hipEventDestroy"(event: HipEventHandle) -> i32;
    fn hip_memcpy_htod_async = "hipMemcpyHtoDAsync"(
        destination: *mut c_void,
        source: *const c_void,
        size: usize,
        stream: HipStreamHandle,
    ) -> i32;
    fn hip_memcpy_dtoh_async = "hipMemcpyDtoHAsync"(
        destination: *mut c_void,
        source: *mut c_void,
        size: usize,
        stream: HipStreamHandle,
    ) -> i32;
    fn hip_module_load_data = "hipModuleLoadData"(module: *mut HipModuleHandle, image: *const c_void) -> i32;
    fn hip_module_unload = "hipModuleUnload"(module: HipModuleHandle) -> i32;
    fn hip_module_get_function = "hipModuleGetFunction"(
        function: *mut HipFunctionHandle,
        module: HipModuleHandle,
        name: *const c_char,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn hip_module_launch_kernel = "hipModuleLaunchKernel"(
        function: HipFunctionHandle,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        shared_memory_bytes: u32,
        stream: HipStreamHandle,
        parameters: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
}

crate::shared_library::shared_library! {
    struct LibHipRtc = LIBHIPRTC, LIBHIPRTC_ROCM6;
    when missing return RTC_ERROR_INTERNAL_ERROR;
    fn hiprtc_create_program = "hiprtcCreateProgram"(
        program: *mut RtcProgram,
        source: *const c_char,
        name: *const c_char,
        header_count: i32,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> i32;
    fn hiprtc_compile_program = "hiprtcCompileProgram"(program: RtcProgram, option_count: i32, options: *const *const c_char) -> i32;
    fn hiprtc_destroy_program = "hiprtcDestroyProgram"(program: *mut RtcProgram) -> i32;
    fn hiprtc_get_program_log_size = "hiprtcGetProgramLogSize"(program: RtcProgram, size: *mut usize) -> i32;
    fn hiprtc_get_program_log = "hiprtcGetProgramLog"(program: RtcProgram, log: *mut c_char) -> i32;
    fn hiprtc_get_code_size = "hiprtcGetCodeSize"(program: RtcProgram, size: *mut usize) -> i32;
    fn hiprtc_get_code = "hiprtcGetCode"(program: RtcProgram, code: *mut c_char) -> i32;
}

crate::shared_library::shared_library! {
    struct LibNvrtc = LIBNVRTC, LIBNVRTC_CUDA12;
    when missing return RTC_ERROR_INTERNAL_ERROR;
    fn nvrtc_create_program = "nvrtcCreateProgram"(
        program: *mut RtcProgram,
        source: *const c_char,
        name: *const c_char,
        header_count: i32,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> i32;
    fn nvrtc_compile_program = "nvrtcCompileProgram"(program: RtcProgram, option_count: i32, options: *const *const c_char) -> i32;
    fn nvrtc_destroy_program = "nvrtcDestroyProgram"(program: *mut RtcProgram) -> i32;
    fn nvrtc_get_program_log_size = "nvrtcGetProgramLogSize"(program: RtcProgram, size: *mut usize) -> i32;
    fn nvrtc_get_program_log = "nvrtcGetProgramLog"(program: RtcProgram, log: *mut c_char) -> i32;
    fn nvrtc_get_cubin_size = "nvrtcGetCUBINSize"(program: RtcProgram, size: *mut usize) -> i32;
    fn nvrtc_get_cubin = "nvrtcGetCUBIN"(program: RtcProgram, code: *mut c_char) -> i32;
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

#[derive(Debug)]
enum Backend {
    Amd,
    // the primary context, the one onnxruntime's CUDA provider also uses
    Nvidia {
        device: cuda::CuDevice,
        context: usize,
    },
}

#[derive(Debug)]
pub struct HipDevice {
    ordinal: i32,
    backend: Backend,
}

// pops the CUDA context `HipDevice::bind` pushed
#[derive(Debug)]
struct Bound<'device>(&'device HipDevice);

impl Drop for Bound<'_> {
    fn drop(&mut self) {
        if let Backend::Nvidia { .. } = self.0.backend {
            let mut popped: cuda::CuContext = core::ptr::null_mut();
            // SAFETY: `bind` pushed a context on this thread.
            unsafe { cuda::cu_ctx_pop_current(&mut popped) };
        }
    }
}

impl HipDevice {
    pub fn open(vendor: HipVendor, ordinal: i32) -> Result<Arc<Self>, G2gError> {
        let backend = match vendor {
            HipVendor::Amd => {
                // SAFETY: hipSetDevice validates the ordinal.
                check(unsafe { hip_set_device(ordinal) })?;
                Backend::Amd
            }
            HipVendor::Nvidia => {
                cuda::init_driver()?;
                let mut device: cuda::CuDevice = 0;
                let mut context: cuda::CuContext = core::ptr::null_mut();
                // SAFETY: both out pointers are valid, and cuDeviceGet validates the ordinal.
                unsafe {
                    cuda::check(cuda::cu_device_get(&mut device, ordinal))?;
                    cuda::check(cuda::cu_device_primary_ctx_retain(&mut context, device))?;
                }
                Backend::Nvidia {
                    device,
                    context: context as usize,
                }
            }
        };
        Ok(Arc::new(Self { ordinal, backend }))
    }

    pub fn vendor(&self) -> HipVendor {
        match self.backend {
            Backend::Amd => HipVendor::Amd,
            Backend::Nvidia { .. } => HipVendor::Nvidia,
        }
    }

    pub fn ordinal(&self) -> i32 {
        self.ordinal
    }

    pub(crate) fn cuda_context(&self) -> Option<u64> {
        match self.backend {
            Backend::Amd => None,
            Backend::Nvidia { context, .. } => Some(context as u64),
        }
    }

    fn bind(&self) -> Result<Bound<'_>, G2gError> {
        match self.backend {
            // SAFETY: the ordinal was validated in `open`.
            Backend::Amd => check(unsafe { hip_set_device(self.ordinal) })?,
            Backend::Nvidia { context, .. } => {
                // SAFETY: the context is retained until this device drops.
                cuda::check(unsafe { cuda::cu_ctx_push_current(context as cuda::CuContext) })?
            }
        }
        Ok(Bound(self))
    }

    fn check(&self, code: i32) -> Result<(), G2gError> {
        match self.backend {
            Backend::Amd => check(code),
            Backend::Nvidia { .. } => cuda::check(code),
        }
    }

    fn holds(&self, buffer: &OwnedHipBuffer) -> bool {
        buffer.vendor == self.vendor() && buffer.device_ordinal == self.ordinal
    }

    unsafe fn synchronize_event(&self, event: HipEventHandle) -> Result<(), G2gError> {
        let _bound = self.bind()?;
        // SAFETY: the caller guarantees the event is live and belongs to the bound device.
        let code = unsafe {
            match self.backend {
                Backend::Amd => hip_event_synchronize(event),
                Backend::Nvidia { .. } => cuda::cu_event_synchronize(event),
            }
        };
        self.check(code)
    }

    // returns once the producer's writes into `buffer` have finished
    pub fn wait_until_ready(&self, buffer: &OwnedHipBuffer) -> Result<(), G2gError> {
        if !self.holds(buffer) {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        // SAFETY: the ready event belongs to this device and stays alive while `buffer` does.
        unsafe { self.synchronize_event(buffer.ready_event as HipEventHandle) }
    }
}

impl Drop for HipDevice {
    fn drop(&mut self) {
        if let Backend::Nvidia { device, .. } = self.backend {
            // SAFETY: balances the retain in `open`.
            unsafe { cuda::cu_device_primary_ctx_release(device) };
        }
    }
}

#[derive(Debug)]
pub struct HipEvent {
    device: Arc<HipDevice>,
    handle: usize,
}

impl HipEvent {
    pub fn new(device: &Arc<HipDevice>) -> Result<Self, G2gError> {
        let _bound = device.bind()?;
        let mut handle: HipEventHandle = core::ptr::null_mut();
        // SAFETY: `handle` is a valid out pointer and the device is bound.
        let code = unsafe {
            match device.backend {
                Backend::Amd => hip_event_create_with_flags(&mut handle, EVENT_DISABLE_TIMING),
                Backend::Nvidia { .. } => cuda::cu_event_create(&mut handle, EVENT_DISABLE_TIMING),
            }
        };
        device.check(code)?;
        Ok(Self {
            device: Arc::clone(device),
            handle: handle as usize,
        })
    }

    pub fn synchronize(&self) -> Result<(), G2gError> {
        // SAFETY: the event belongs to the device and lives as long as `self`.
        unsafe { self.device.synchronize_event(self.handle as HipEventHandle) }
    }
}

impl Drop for HipEvent {
    fn drop(&mut self) {
        let Ok(_bound) = self.device.bind() else {
            return;
        };
        let handle = self.handle as HipEventHandle;
        // SAFETY: the event belongs to the bound device and nothing uses it after this.
        unsafe {
            match self.device.backend {
                Backend::Amd => hip_event_destroy(handle),
                Backend::Nvidia { .. } => cuda::cu_event_destroy(handle),
            };
        }
    }
}

type FreeList = Arc<Mutex<Vec<HipMemory>>>;

#[derive(Debug)]
pub struct HipMemory {
    device: Arc<HipDevice>,
    device_pointer: u64,
    size: usize,
    recycle_into: Option<FreeList>,
}

impl HipMemory {
    pub fn allocate(device: &Arc<HipDevice>, size: usize) -> Result<Self, G2gError> {
        let _bound = device.bind()?;
        let mut device_pointer: u64 = 0;
        let mut malloc = || {
            // SAFETY: `device_pointer` is a valid out pointer and the device is bound.
            unsafe {
                match device.backend {
                    Backend::Amd => hip_malloc((&mut device_pointer as *mut u64).cast(), size),
                    Backend::Nvidia { .. } => cuda::cu_mem_alloc(&mut device_pointer, size),
                }
            }
        };
        let mut code = malloc();
        if code == OUT_OF_MEMORY {
            release_queue().wait_until_drained();
            code = malloc();
        }
        device.check(code)?;
        Ok(Self {
            device: Arc::clone(device),
            device_pointer,
            size,
            recycle_into: None,
        })
    }

    pub fn device_pointer(&self) -> u64 {
        self.device_pointer
    }

    pub fn size(&self) -> usize {
        self.size
    }

    // `ready` must be recorded after the last write into this memory
    pub fn into_buffer(self, ready: HipEvent, len: usize) -> Result<OwnedHipBuffer, G2gError> {
        if len > self.size {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        let device = Arc::clone(&self.device);
        let device_pointer = self.device_pointer;
        hip_buffer(&device, device_pointer, len, ready, Box::new(self))
    }
}

pub(crate) fn hip_buffer(
    device: &Arc<HipDevice>,
    device_pointer: u64,
    len: usize,
    ready: HipEvent,
    resources: Box<dyn core::fmt::Debug + Send + Sync>,
) -> Result<OwnedHipBuffer, G2gError> {
    if !Arc::ptr_eq(device, &ready.device) {
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    Ok(OwnedHipBuffer::new(
        device_pointer,
        len,
        device.vendor(),
        device.ordinal,
        ready.handle as u64,
        Arc::new(HipBufferOwner(Some((resources, ready)))),
    ))
}

impl Drop for HipMemory {
    fn drop(&mut self) {
        if let Some(free) = self.recycle_into.take() {
            free.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(HipMemory {
                    device: Arc::clone(&self.device),
                    device_pointer: self.device_pointer,
                    size: self.size,
                    recycle_into: None,
                });
            return;
        }
        let Ok(_bound) = self.device.bind() else {
            return;
        };
        // SAFETY: the allocation came from `allocate` on this device and nothing uses it after this.
        unsafe {
            match self.device.backend {
                Backend::Amd => hip_free(self.device_pointer as *mut c_void),
                Backend::Nvidia { .. } => cuda::cu_mem_free(self.device_pointer),
            };
        }
    }
}

#[derive(Debug)]
pub(crate) struct HipMemoryPool {
    device: Arc<HipDevice>,
    size: usize,
    free: FreeList,
}

impl HipMemoryPool {
    pub(crate) fn new(device: &Arc<HipDevice>, size: usize) -> Self {
        Self {
            device: Arc::clone(device),
            size,
            free: Arc::default(),
        }
    }

    pub(crate) fn size(&self) -> usize {
        self.size
    }

    pub(crate) fn acquire(&self) -> Result<HipMemory, G2gError> {
        let recycled = self
            .free
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop();
        let mut memory = match recycled {
            Some(memory) => memory,
            None => HipMemory::allocate(&self.device, self.size)?,
        };
        memory.recycle_into = Some(Arc::clone(&self.free));
        Ok(memory)
    }
}

#[derive(Debug)]
pub(crate) struct HipPinnedMemory {
    device: Arc<HipDevice>,
    pointer: usize,
    size: usize,
}

impl HipPinnedMemory {
    pub(crate) fn allocate(device: &Arc<HipDevice>, size: usize) -> Result<Self, G2gError> {
        let _bound = device.bind()?;
        let mut pointer: *mut c_void = core::ptr::null_mut();
        // SAFETY: `pointer` is a valid out pointer and the device is bound.
        let code = unsafe {
            match device.backend {
                Backend::Amd => hip_host_malloc(&mut pointer, size, 0),
                Backend::Nvidia { .. } => cuda::cu_mem_alloc_host(&mut pointer, size),
            }
        };
        device.check(code)?;
        Ok(Self {
            device: Arc::clone(device),
            pointer: pointer as usize,
            size,
        })
    }

    pub(crate) fn size(&self) -> usize {
        self.size
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: the allocation is `size` bytes and `&mut self` makes this the only view of it.
        unsafe { core::slice::from_raw_parts_mut(self.pointer as *mut u8, self.size) }
    }
}

impl Drop for HipPinnedMemory {
    fn drop(&mut self) {
        let Ok(_bound) = self.device.bind() else {
            return;
        };
        let pointer = self.pointer as *mut c_void;
        // SAFETY: the allocation came from `allocate` on this device and nothing uses it after this.
        unsafe {
            match self.device.backend {
                Backend::Amd => hip_host_free(pointer),
                Backend::Nvidia { .. } => cuda::cu_mem_free_host(pointer),
            };
        }
    }
}

#[derive(Debug)]
struct HipBufferOwner(Option<(Box<dyn core::fmt::Debug + Send + Sync>, HipEvent)>);

impl HipKeepAlive for HipBufferOwner {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl Drop for HipBufferOwner {
    fn drop(&mut self) {
        if let Some((resources, ready)) = self.0.take() {
            release_after(ready, resources);
        }
    }
}

#[derive(Debug)]
struct Release {
    event: HipEvent,
    resources: Box<dyn core::fmt::Debug + Send>,
}

#[derive(Debug)]
struct ReleaseQueue {
    sender: Mutex<Sender<Release>>,
    pending: Mutex<usize>,
    drained: Condvar,
}

impl ReleaseQueue {
    fn wait_until_drained(&self) {
        let pending = self.pending.lock().unwrap_or_else(PoisonError::into_inner);
        drop(
            self.drained
                .wait_while(pending, |pending| *pending > 0)
                .unwrap_or_else(PoisonError::into_inner),
        );
    }

    fn finish_one(&self) {
        let mut pending = self.pending.lock().unwrap_or_else(PoisonError::into_inner);
        *pending -= 1;
        if *pending == 0 {
            self.drained.notify_all();
        }
    }
}

fn release_queue() -> &'static ReleaseQueue {
    static QUEUE: OnceLock<ReleaseQueue> = OnceLock::new();
    QUEUE.get_or_init(|| {
        let (sender, receiver) = mpsc::channel::<Release>();
        std::thread::Builder::new()
            .name("g2g-hip-release".into())
            .spawn(move || {
                for release in receiver {
                    if let Err(error) = release.event.synchronize() {
                        g2g_error!(
                            g2g_core::log::Target::category("hip"),
                            "waiting for GPU work before freeing {:?} failed: {:?}",
                            release.resources,
                            error
                        );
                    }
                    drop(release);
                    release_queue().finish_one();
                }
            })
            .expect("spawn the HIP release thread");
        ReleaseQueue {
            sender: Mutex::new(sender),
            pending: Mutex::new(0),
            drained: Condvar::new(),
        }
    })
}

// drops `resources` on a background thread once the GPU work before `event` completes
pub fn release_after(event: HipEvent, resources: Box<dyn core::fmt::Debug + Send>) {
    let queue = release_queue();
    *queue.pending.lock().unwrap_or_else(PoisonError::into_inner) += 1;
    let sender = queue.sender.lock().unwrap_or_else(PoisonError::into_inner);
    // the receiver lives in a thread that never exits
    let _ = sender.send(Release { event, resources });
}

struct RuntimeCompiler {
    library: &'static str,
    is_loaded: fn() -> bool,
    create_program: unsafe fn(
        *mut RtcProgram,
        *const c_char,
        *const c_char,
        i32,
        *const *const c_char,
        *const *const c_char,
    ) -> i32,
    compile_program: unsafe fn(RtcProgram, i32, *const *const c_char) -> i32,
    destroy_program: unsafe fn(*mut RtcProgram) -> i32,
    get_program_log_size: unsafe fn(RtcProgram, *mut usize) -> i32,
    get_program_log: unsafe fn(RtcProgram, *mut c_char) -> i32,
    get_code_size: unsafe fn(RtcProgram, *mut usize) -> i32,
    get_code: unsafe fn(RtcProgram, *mut c_char) -> i32,
}

const HIPRTC: RuntimeCompiler = RuntimeCompiler {
    library: LIBHIPRTC,
    is_loaded: || LibHipRtc::get().is_some(),
    create_program: hiprtc_create_program,
    compile_program: hiprtc_compile_program,
    destroy_program: hiprtc_destroy_program,
    get_program_log_size: hiprtc_get_program_log_size,
    get_program_log: hiprtc_get_program_log,
    get_code_size: hiprtc_get_code_size,
    get_code: hiprtc_get_code,
};

const NVRTC: RuntimeCompiler = RuntimeCompiler {
    library: LIBNVRTC,
    is_loaded: || LibNvrtc::get().is_some(),
    create_program: nvrtc_create_program,
    compile_program: nvrtc_compile_program,
    destroy_program: nvrtc_destroy_program,
    get_program_log_size: nvrtc_get_program_log_size,
    get_program_log: nvrtc_get_program_log,
    get_code_size: nvrtc_get_cubin_size,
    get_code: nvrtc_get_cubin,
};

fn rtc_check(code: i32) -> Result<(), G2gError> {
    match code {
        RTC_SUCCESS => Ok(()),
        code => Err(G2gError::Hardware(HardwareError::KernelCompilation(code))),
    }
}

fn c_string(text: &str) -> Result<CString, G2gError> {
    CString::new(text)
        .map_err(|_| G2gError::Hardware(HardwareError::KernelCompilation(RTC_ERROR_INVALID_INPUT)))
}

/// # Safety
/// `program` must be a live program of `compiler`.
unsafe fn program_log(compiler: &RuntimeCompiler, program: RtcProgram) -> Result<String, G2gError> {
    let mut size = 0;
    // SAFETY: the caller passes a live program and `size` is a valid out pointer.
    rtc_check(unsafe { (compiler.get_program_log_size)(program, &mut size) })?;
    let mut log = vec![0u8; size];
    // SAFETY: `log` has the size the compiler reported, terminator included.
    rtc_check(unsafe { (compiler.get_program_log)(program, log.as_mut_ptr().cast()) })?;
    let text = log.split(|&byte| byte == 0).next().unwrap_or_default();
    Ok(String::from_utf8_lossy(text).to_string())
}

/// # Safety
/// `program` must be a live program of `compiler`.
unsafe fn compiled_code(
    compiler: &RuntimeCompiler,
    program: RtcProgram,
    options: &[CString],
) -> Result<Vec<u8>, G2gError> {
    let option_pointers: Vec<*const c_char> =
        options.iter().map(|option| option.as_ptr()).collect();
    // SAFETY: the option strings outlive the call.
    let compiled = unsafe {
        (compiler.compile_program)(
            program,
            option_pointers.len() as i32,
            option_pointers.as_ptr(),
        )
    };
    if compiled != RTC_SUCCESS {
        // SAFETY: the caller passes a live program.
        let log = unsafe { program_log(compiler, program) }?;
        g2g_error!(
            g2g_core::log::Target::category("hip"),
            "{} failed to compile a kernel: {}",
            compiler.library,
            log
        );
        return Err(G2gError::Hardware(HardwareError::KernelCompilation(
            compiled,
        )));
    }
    let mut size = 0;
    // SAFETY: the program compiled and `size` is a valid out pointer.
    rtc_check(unsafe { (compiler.get_code_size)(program, &mut size) })?;
    let mut code = vec![0u8; size];
    // SAFETY: `code` has the size the compiler reported.
    rtc_check(unsafe { (compiler.get_code)(program, code.as_mut_ptr().cast()) })?;
    Ok(code)
}

fn compile_to_code(
    compiler: &RuntimeCompiler,
    source: &str,
    options: &[CString],
) -> Result<Vec<u8>, G2gError> {
    if !(compiler.is_loaded)() {
        return Err(G2gError::Hardware(HardwareError::LibraryNotFound(
            compiler.library,
        )));
    }
    let source = c_string(source)?;
    let mut program: RtcProgram = core::ptr::null_mut();
    // SAFETY: the source and name are NUL-terminated and no headers are passed.
    rtc_check(unsafe {
        (compiler.create_program)(
            &mut program,
            source.as_ptr(),
            c"g2g_kernel".as_ptr(),
            0,
            core::ptr::null(),
            core::ptr::null(),
        )
    })?;
    // SAFETY: the program was created above and is destroyed only after this.
    let code = unsafe { compiled_code(compiler, program, options) };
    // SAFETY: nothing uses the program after this.
    unsafe { (compiler.destroy_program)(&mut program) };
    code
}

#[derive(Debug)]
pub struct HipKernelModule {
    device: Arc<HipDevice>,
    handle: usize,
}

impl HipKernelModule {
    // `source` is C++ that both hiprtc and nvrtc accept, with `extern "C" __global__` kernels
    pub fn compile(
        device: &Arc<HipDevice>,
        source: &str,
        options: &[&str],
    ) -> Result<Arc<Self>, G2gError> {
        let mut options = options
            .iter()
            .map(|option| c_string(option))
            .collect::<Result<Vec<_>, _>>()?;
        let compiler = match device.backend {
            Backend::Amd => &HIPRTC,
            Backend::Nvidia {
                device: cuda_device,
                ..
            } => {
                let mut major = 0;
                let mut minor = 0;
                // SAFETY: both out pointers are valid and the device came from cuDeviceGet.
                unsafe {
                    cuda::check(cuda::cu_device_get_attribute(
                        &mut major,
                        cuda::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                        cuda_device,
                    ))?;
                    cuda::check(cuda::cu_device_get_attribute(
                        &mut minor,
                        cuda::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                        cuda_device,
                    ))?;
                }
                options.push(c_string(&format!("--gpu-architecture=sm_{major}{minor}"))?);
                &NVRTC
            }
        };
        let code = compile_to_code(compiler, source, &options)?;
        let _bound = device.bind()?;
        let mut handle: HipModuleHandle = core::ptr::null_mut();
        // SAFETY: `code` is a code object or cubin for this device and the device is bound.
        let loaded = unsafe {
            match device.backend {
                Backend::Amd => hip_module_load_data(&mut handle, code.as_ptr().cast()),
                Backend::Nvidia { .. } => {
                    cuda::cu_module_load_data(&mut handle, code.as_ptr().cast())
                }
            }
        };
        device.check(loaded)?;
        Ok(Arc::new(Self {
            device: Arc::clone(device),
            handle: handle as usize,
        }))
    }

    pub fn kernel(self: &Arc<Self>, name: &str) -> Result<HipKernel, G2gError> {
        let name = CString::new(name).map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        let _bound = self.device.bind()?;
        let module = self.handle as HipModuleHandle;
        let mut function: HipFunctionHandle = core::ptr::null_mut();
        // SAFETY: the module belongs to the bound device and `name` is NUL-terminated.
        let code = unsafe {
            match self.device.backend {
                Backend::Amd => hip_module_get_function(&mut function, module, name.as_ptr()),
                Backend::Nvidia { .. } => {
                    cuda::cu_module_get_function(&mut function, module, name.as_ptr())
                }
            }
        };
        self.device.check(code)?;
        Ok(HipKernel {
            module: Arc::clone(self),
            handle: function as usize,
        })
    }
}

impl Drop for HipKernelModule {
    fn drop(&mut self) {
        let Ok(_bound) = self.device.bind() else {
            return;
        };
        let module = self.handle as HipModuleHandle;
        // SAFETY: the module belongs to the bound device and every kernel from it holds this module alive.
        unsafe {
            match self.device.backend {
                Backend::Amd => hip_module_unload(module),
                Backend::Nvidia { .. } => cuda::cu_module_unload(module),
            };
        }
    }
}

#[derive(Debug)]
pub struct HipKernel {
    module: Arc<HipKernelModule>,
    handle: usize,
}

#[derive(Debug)]
pub enum KernelArgument<'a> {
    // the kernel starts after the producer's writes into the buffer
    Read(&'a OwnedHipBuffer),
    Write(&'a mut HipMemory),
    U32(u32),
    F32(f32),
}

#[derive(Debug)]
pub struct HipStream {
    device: Arc<HipDevice>,
    handle: usize,
}

impl HipStream {
    pub fn new(device: &Arc<HipDevice>) -> Result<Self, G2gError> {
        let _bound = device.bind()?;
        let mut handle: HipStreamHandle = core::ptr::null_mut();
        // SAFETY: `handle` is a valid out pointer and the device is bound.
        let code = unsafe {
            match device.backend {
                Backend::Amd => hip_stream_create_with_flags(&mut handle, STREAM_NON_BLOCKING),
                Backend::Nvidia { .. } => cuda::cu_stream_create(&mut handle, STREAM_NON_BLOCKING),
            }
        };
        device.check(code)?;
        Ok(Self {
            device: Arc::clone(device),
            handle: handle as usize,
        })
    }

    pub fn device(&self) -> &Arc<HipDevice> {
        &self.device
    }

    // later work on this stream starts after the producer's writes into `buffer`
    pub fn wait_for(&self, buffer: &OwnedHipBuffer) -> Result<(), G2gError> {
        if !self.device.holds(buffer) {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        let _bound = self.device.bind()?;
        let stream = self.handle as HipStreamHandle;
        let event = buffer.ready_event as HipEventHandle;
        // SAFETY: the event belongs to this device and stays alive while `buffer` does.
        let code = unsafe {
            match self.device.backend {
                Backend::Amd => hip_stream_wait_event(stream, event, 0),
                Backend::Nvidia { .. } => cuda::cu_stream_wait_event(stream, event, 0),
            }
        };
        self.device.check(code)
    }

    pub fn record(&self, event: &HipEvent) -> Result<(), G2gError> {
        if !Arc::ptr_eq(&self.device, &event.device) {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        let _bound = self.device.bind()?;
        let stream = self.handle as HipStreamHandle;
        let event = event.handle as HipEventHandle;
        // SAFETY: the stream and the event belong to the bound device.
        let code = unsafe {
            match self.device.backend {
                Backend::Amd => hip_event_record(event, stream),
                Backend::Nvidia { .. } => cuda::cu_event_record(event, stream),
            }
        };
        self.device.check(code)
    }

    pub fn synchronize(&self) -> Result<(), G2gError> {
        let _bound = self.device.bind()?;
        let stream = self.handle as HipStreamHandle;
        // SAFETY: the stream belongs to the bound device.
        let code = unsafe {
            match self.device.backend {
                Backend::Amd => hip_stream_synchronize(stream),
                Backend::Nvidia { .. } => cuda::cu_stream_synchronize(stream),
            }
        };
        self.device.check(code)
    }

    pub fn launch(
        &self,
        kernel: &HipKernel,
        grid: [u32; 3],
        block: [u32; 3],
        arguments: &[KernelArgument],
    ) -> Result<(), G2gError> {
        if !Arc::ptr_eq(&self.device, &kernel.module.device) {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        for argument in arguments {
            match argument {
                KernelArgument::Read(buffer) => self.wait_for(buffer)?,
                KernelArgument::Write(memory) if !Arc::ptr_eq(&self.device, &memory.device) => {
                    return Err(G2gError::Hardware(HardwareError::Other))
                }
                _ => {}
            }
        }
        // hipModuleLaunchKernel and cuLaunchKernel read each argument through a pointer to its value
        let mut parameters: Vec<*mut c_void> = arguments
            .iter()
            .map(|argument| match argument {
                KernelArgument::Read(buffer) => core::ptr::from_ref(&buffer.device_pointer)
                    .cast_mut()
                    .cast(),
                KernelArgument::Write(memory) => core::ptr::from_ref(&memory.device_pointer)
                    .cast_mut()
                    .cast(),
                KernelArgument::U32(value) => core::ptr::from_ref(value).cast_mut().cast(),
                KernelArgument::F32(value) => core::ptr::from_ref(value).cast_mut().cast(),
            })
            .collect();
        let _bound = self.device.bind()?;
        let function = kernel.handle as HipFunctionHandle;
        let stream = self.handle as HipStreamHandle;
        let [grid_x, grid_y, grid_z] = grid;
        let [block_x, block_y, block_z] = block;
        // SAFETY: the kernel and stream belong to the bound device, and the launch copies the argument values before it returns.
        let code = unsafe {
            match self.device.backend {
                Backend::Amd => hip_module_launch_kernel(
                    function,
                    grid_x,
                    grid_y,
                    grid_z,
                    block_x,
                    block_y,
                    block_z,
                    0,
                    stream,
                    parameters.as_mut_ptr(),
                    core::ptr::null_mut(),
                ),
                Backend::Nvidia { .. } => cuda::cu_launch_kernel(
                    function,
                    grid_x,
                    grid_y,
                    grid_z,
                    block_x,
                    block_y,
                    block_z,
                    0,
                    stream,
                    parameters.as_mut_ptr(),
                    core::ptr::null_mut(),
                ),
            }
        };
        self.device.check(code)?;
        // cuMemFree may free a dropped input while this kernel still reads it
        let reads: Vec<OwnedHipBuffer> = arguments
            .iter()
            .filter_map(|argument| match argument {
                KernelArgument::Read(buffer) => Some((*buffer).clone()),
                _ => None,
            })
            .collect();
        if reads.is_empty() {
            return Ok(());
        }
        let done = HipEvent::new(&self.device)?;
        self.record(&done)?;
        release_after(done, Box::new(reads));
        Ok(())
    }

    pub(crate) fn upload_pinned(
        &self,
        memory: &HipMemory,
        staging: &HipPinnedMemory,
        len: usize,
    ) -> Result<(), G2gError> {
        // the GPU reads `staging` after this returns, until the stream reaches the copy
        let same_device =
            Arc::ptr_eq(&self.device, &memory.device) && Arc::ptr_eq(&self.device, &staging.device);
        if len > memory.size || len > staging.size || !same_device {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        let _bound = self.device.bind()?;
        let stream = self.handle as HipStreamHandle;
        let source = staging.pointer as *const c_void;
        // SAFETY: the copy stays inside both allocations, which belong to the bound device.
        let code = unsafe {
            match self.device.backend {
                Backend::Amd => {
                    hip_memcpy_htod_async(memory.device_pointer as *mut c_void, source, len, stream)
                }
                Backend::Nvidia { .. } => {
                    cuda::cu_memcpy_htod_async(memory.device_pointer, source, len, stream)
                }
            }
        };
        self.device.check(code)
    }

    // returns once `bytes` has reached `memory`
    pub fn upload_blocking(&self, memory: &HipMemory, bytes: &[u8]) -> Result<(), G2gError> {
        if bytes.len() > memory.size || !Arc::ptr_eq(&self.device, &memory.device) {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        {
            let _bound = self.device.bind()?;
            let stream = self.handle as HipStreamHandle;
            let source = bytes.as_ptr().cast::<c_void>();
            // SAFETY: the copy stays inside both buffers, and `synchronize` below ends it before `bytes` can go away.
            let code = unsafe {
                match self.device.backend {
                    Backend::Amd => hip_memcpy_htod_async(
                        memory.device_pointer as *mut c_void,
                        source,
                        bytes.len(),
                        stream,
                    ),
                    Backend::Nvidia { .. } => cuda::cu_memcpy_htod_async(
                        memory.device_pointer,
                        source,
                        bytes.len(),
                        stream,
                    ),
                }
            };
            self.device.check(code)?;
        }
        self.synchronize()
    }

    // returns once the first `out.len()` bytes of `buffer` are in `out`
    pub fn download_blocking(
        &self,
        buffer: &OwnedHipBuffer,
        out: &mut [u8],
    ) -> Result<(), G2gError> {
        if out.len() > buffer.len {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        self.wait_for(buffer)?;
        {
            let _bound = self.device.bind()?;
            let stream = self.handle as HipStreamHandle;
            let destination = out.as_mut_ptr().cast::<c_void>();
            // SAFETY: the copy stays inside both buffers, and `synchronize` below ends it before `out` can go away.
            let code = unsafe {
                match self.device.backend {
                    Backend::Amd => hip_memcpy_dtoh_async(
                        destination,
                        buffer.device_pointer as *mut c_void,
                        out.len(),
                        stream,
                    ),
                    Backend::Nvidia { .. } => cuda::cu_memcpy_dtoh_async(
                        destination,
                        buffer.device_pointer,
                        out.len(),
                        stream,
                    ),
                }
            };
            self.device.check(code)?;
        }
        self.synchronize()
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        let Ok(_bound) = self.device.bind() else {
            return;
        };
        let stream = self.handle as HipStreamHandle;
        // SAFETY: the stream belongs to the bound device and nothing uses it after this.
        unsafe {
            match self.device.backend {
                Backend::Amd => hip_stream_destroy(stream),
                Backend::Nvidia { .. } => cuda::cu_stream_destroy(stream),
            };
        }
    }
}
