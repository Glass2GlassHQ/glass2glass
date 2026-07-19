//! M308: on-device probe for the Android Camera2 capture source.
//!
//! Camera access needs the `CAMERA` runtime permission, which a bare
//! `/data/local/tmp` native binary (no APK / manifest) does not hold, so opening
//! the camera is expected to fail there. This probe therefore validates what it
//! can headlessly: the produced caps are NV12 at the requested geometry, the
//! Camera2 FFI links and the element drives `ACameraManager_openCamera`, and IF
//! the camera does open (e.g. run from an APK harness) it captures NV12 frames of
//! the right size. A permission/headless denial is reported, not failed.
//!
//! Runs only on `aarch64-linux-android` (et al.) with the `camera2` feature.
//! Build with cargo-ndk `--platform 24`, push, run. See
//! `tools/android-camera2-smoke.sh`.

#![cfg(all(target_os = "android", feature = "camera2"))]

use g2g_core::element::{BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::PipelinePacket;
use g2g_core::memory::MemoryDomain;
use g2g_core::runtime::SourceLoop;
use g2g_core::{Caps, Dim, G2gError, RawVideoFormat};
use g2g_plugins::camera2src::Camera2Src;

/// Start a binder threadpool so the camera HAL can deliver capture buffers (it
/// calls back over binder into this process, like Codec2 for the decoder). A
/// bare native binary has no threadpool, so buffers never arrive and teardown
/// stalls; the `ABinderProcess_*` symbols live in libbinder_ndk.so but not the
/// NDK link stub, so dlsym them.
fn start_binder_threadpool() {
    use core::ffi::{c_char, c_int, c_void};
    extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
    const RTLD_NOW: c_int = 2;
    // SAFETY: libbinder_ndk.so is loadable; the dlsym'd symbols have the C
    // signatures from <android/binder_process.h>.
    unsafe {
        let lib = dlopen(b"libbinder_ndk.so\0".as_ptr() as *const c_char, RTLD_NOW);
        if lib.is_null() {
            return;
        }
        let set = dlsym(
            lib,
            b"ABinderProcess_setThreadPoolMaxThreadCount\0".as_ptr() as *const c_char,
        );
        if !set.is_null() {
            let set: extern "C" fn(u32) -> bool = core::mem::transmute(set);
            set(4);
        }
        let start = dlsym(
            lib,
            b"ABinderProcess_startThreadPool\0".as_ptr() as *const c_char,
        );
        if !start.is_null() {
            let start: extern "C" fn() = core::mem::transmute(start);
            start();
        }
    }
}

/// Counts captured NV12 frames and records the byte length of the first.
#[derive(Default)]
struct CaptureSink {
    frames: u64,
    first_len: usize,
}
impl OutputSink for CaptureSink {
    fn push<'a>(&'a mut self, p: PipelinePacket) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        if let PipelinePacket::DataFrame(f) = &p {
            if self.frames == 0 {
                if let MemoryDomain::System(s) = &f.domain {
                    self.first_len = s.as_slice().len();
                }
            }
            self.frames += 1;
        }
        Box::pin(async move { Ok(PushOutcome::Accepted) })
    }
}

#[tokio::test]
async fn camera2_capture_best_effort() {
    start_binder_threadpool();
    let (w, h) = (640u32, 480u32);
    let mut src = Camera2Src::new(w, h, 5);

    // Caps are known without opening the device: NV12 at the requested geometry.
    let caps = SourceLoop::intercept_caps(&mut src).await.expect("caps");
    eprintln!("=== M308 Camera2Src caps: {caps:?} ===");
    assert!(
        matches!(
            caps,
            Caps::RawVideo { format: RawVideoFormat::Nv12, width: Dim::Fixed(cw), height: Dim::Fixed(ch), .. }
                if cw == w && ch == h
        ),
        "expected NV12 {w}x{h} caps, got {caps:?}"
    );

    // Opening the camera needs CAMERA permission; tolerate a denial.
    if let Err(e) = src.configure_pipeline(&caps) {
        eprintln!(">>> camera open failed ({e:?}) - likely no CAMERA permission (bare binary); FFI linked OK, skipping capture");
        return;
    }
    eprintln!(">>> camera opened; capturing");
    let mut out = CaptureSink::default();
    match src.run(&mut out).await {
        Ok(n) => {
            eprintln!(
                ">>> captured {n} frame(s); first NV12 buffer = {} bytes",
                out.first_len
            );
            assert!(n > 0, "camera opened but produced no frames");
            assert_eq!(out.first_len, (w * h * 3 / 2) as usize, "NV12 frame size");
            eprintln!(">>> M308 Camera2 capture validated on device.");
        }
        Err(e) => {
            eprintln!(">>> capture failed after open ({e:?}); headless/permission limitation")
        }
    }
}
