use core::ffi::{c_char, c_int, c_void};

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

const RTLD_NOW: c_int = 2;

// A bare native binary has no binder threadpool, so Codec2, camera and NNAPI callbacks stall.
pub(crate) fn start_binder_threadpool(max_threads: u32) {
    // SAFETY: the dlsym'd symbols have the C signatures declared in <android/binder_process.h>.
    unsafe {
        // The `ABinderProcess_*` symbols are missing from the NDK link stub.
        let library = dlopen(c"libbinder_ndk.so".as_ptr(), RTLD_NOW);
        if library.is_null() {
            return;
        }
        let set_max_threads = dlsym(
            library,
            c"ABinderProcess_setThreadPoolMaxThreadCount".as_ptr(),
        );
        if !set_max_threads.is_null() {
            let set_max_threads: extern "C" fn(u32) -> bool = core::mem::transmute(set_max_threads);
            set_max_threads(max_threads);
        }
        let start = dlsym(library, c"ABinderProcess_startThreadPool".as_ptr());
        if !start.is_null() {
            let start: extern "C" fn() = core::mem::transmute(start);
            start();
        }
    }
}
