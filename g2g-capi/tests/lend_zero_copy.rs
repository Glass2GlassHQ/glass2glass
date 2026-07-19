//! Zero-copy lend (M234): the application lends a buffer to `appsrc` and the
//! pipeline reads it in place. Proven two ways: the `appsink` callback observes
//! the *same pointer* that was lent (no copy happened), and the free callback
//! fires exactly once after the frame is consumed.

use std::ffi::{c_void, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

use g2g_capi::{
    g2g_appsink_set_callback, g2g_appsrc_end_of_stream, g2g_appsrc_free, g2g_appsrc_new,
    g2g_appsrc_push_lend, g2g_pipeline_free, g2g_pipeline_launch, g2g_pipeline_wait,
};

/// Free callback: counts invocations via the `AtomicUsize` at `user`.
extern "C" fn count_free(user: *mut c_void) {
    // SAFETY: `user` is the &AtomicUsize registered as the lend's free user.
    unsafe { &*(user as *const AtomicUsize) }.fetch_add(1, SeqCst);
}

/// appsink callback: records the data pointer it received via the `AtomicUsize`
/// at `user` (the zero-copy witness).
extern "C" fn record_ptr(data: *const u8, _len: usize, _pts: u64, user: *mut c_void) {
    if !data.is_null() {
        // SAFETY: `user` is the &AtomicUsize registered for the appsink.
        unsafe { &*(user as *const AtomicUsize) }.store(data as usize, SeqCst);
    }
}

#[test]
fn lend_reaches_sink_without_copy_and_frees_once() {
    let free_count = Box::new(AtomicUsize::new(0));
    let seen_ptr = Box::new(AtomicUsize::new(0));
    let free_user = (&*free_count as *const AtomicUsize) as *mut c_void;
    let ptr_user = (&*seen_ptr as *const AtomicUsize) as *mut c_void;

    let cam = CString::new("lendcam").unwrap();
    let out = CString::new("lendout").unwrap();
    // SAFETY: valid C strings; the atomics outlive the pipeline.
    let src = unsafe { g2g_appsrc_new(cam.as_ptr()) };
    assert!(!src.is_null());
    // SAFETY: valid C string; callback + user atomics outlive the pipeline.
    unsafe { g2g_appsink_set_callback(out.as_ptr(), Some(record_ptr), ptr_user) };

    let desc = CString::new(
        "appsrc channel=lendcam caps=video/x-raw,format=RGBA,width=2,height=2,framerate=30/1 \
         ! appsink channel=lendout",
    )
    .unwrap();
    let mut err: *mut c_char = ptr::null_mut();
    // SAFETY: valid C string; err writable.
    let p = unsafe { g2g_pipeline_launch(desc.as_ptr(), &mut err) };
    assert!(!p.is_null());

    // The lent buffer: stays alive (and unmodified) for the whole run; the free
    // callback here only counts, so there is no use-after-free.
    let buf: Vec<u8> = vec![9u8; 16];
    let lent_ptr = buf.as_ptr() as usize;
    // SAFETY: `src` live; `buf` valid until after `wait` (and `count_free`).
    let ok = unsafe {
        g2g_appsrc_push_lend(src, buf.as_ptr(), buf.len(), 0, Some(count_free), free_user)
    };
    assert_eq!(ok, 1, "lend accepted");
    // SAFETY: `src` live.
    unsafe { g2g_appsrc_end_of_stream(src) };

    // SAFETY: `p` live; blocks until EOS, joining the run thread.
    let rc = unsafe { g2g_pipeline_wait(p, ptr::null_mut()) };
    assert_eq!(rc, 0);

    // Zero-copy witness: the sink read the very bytes we lent, in place.
    assert_eq!(
        seen_ptr.load(SeqCst),
        lent_ptr,
        "appsink saw the lent pointer (no copy)"
    );
    // The lend was released exactly once, after consumption.
    assert_eq!(free_count.load(SeqCst), 1, "free fired exactly once");

    // SAFETY: live handles, each freed once.
    unsafe {
        g2g_pipeline_free(p);
        g2g_appsrc_free(src);
    }
    drop(buf);
}
