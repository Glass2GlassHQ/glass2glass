use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use core::ffi::{c_char, c_void};
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

type HipExternalMemory = *mut c_void;
type HipStreamHandle = *mut c_void;
type HipEventHandle = *mut c_void;

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
        let _bound = self.device.bind()?;
        let handle = self.handle as HipEventHandle;
        // SAFETY: the event belongs to the bound device.
        let code = unsafe {
            match self.device.backend {
                Backend::Amd => hip_event_synchronize(handle),
                Backend::Nvidia { .. } => cuda::cu_event_synchronize(handle),
            }
        };
        self.device.check(code)
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

#[derive(Debug)]
pub struct HipMemory {
    device: Arc<HipDevice>,
    device_pointer: u64,
    size: usize,
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
        if len > self.size || !Arc::ptr_eq(&self.device, &ready.device) {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        let buffer = OwnedHipBuffer::new(
            self.device_pointer,
            len,
            self.device.vendor(),
            self.device.ordinal,
            ready.handle as u64,
            Arc::new(HipBufferOwner(Some((self, ready)))),
        );
        Ok(buffer)
    }
}

impl Drop for HipMemory {
    fn drop(&mut self) {
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
struct HipBufferOwner(Option<(HipMemory, HipEvent)>);

impl HipKeepAlive for HipBufferOwner {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl Drop for HipBufferOwner {
    fn drop(&mut self) {
        if let Some((memory, ready)) = self.0.take() {
            release_after(ready, Box::new(memory));
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
