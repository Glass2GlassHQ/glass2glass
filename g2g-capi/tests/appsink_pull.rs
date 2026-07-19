//! appsink pull mode (M235): the application pulls whole samples out of the
//! pipeline (blocking), and a pulled sample owns its bytes until freed. The
//! second test pushes an `appsrc` lend straight through to a pulled sample and
//! proves the pointer survives end-to-end (zero-copy the whole way).

use std::ffi::{c_void, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

use g2g_capi::{
    g2g_appsink_free, g2g_appsink_new, g2g_appsink_pull, g2g_appsrc_end_of_stream, g2g_appsrc_free,
    g2g_appsrc_new, g2g_appsrc_push, g2g_appsrc_push_lend, g2g_pipeline_free, g2g_pipeline_launch,
    g2g_pipeline_wait, g2g_sample_data, g2g_sample_free, g2g_sample_len, g2g_sample_pts, Sample,
};

const CAPS: &str = "video/x-raw,format=RGBA,width=2,height=2,framerate=30/1";

extern "C" fn count_free(user: *mut c_void) {
    // SAFETY: `user` is the &AtomicUsize registered as the lend's free user.
    unsafe { &*(user as *const AtomicUsize) }.fetch_add(1, SeqCst);
}

#[test]
fn pull_receives_every_frame_then_eos() {
    let cam = CString::new("p1cam").unwrap();
    let out = CString::new("p1out").unwrap();
    // SAFETY: valid C strings.
    let (src, sink) = unsafe { (g2g_appsrc_new(cam.as_ptr()), g2g_appsink_new(out.as_ptr())) };
    assert!(!src.is_null() && !sink.is_null());

    let desc = CString::new(format!(
        "appsrc channel=p1cam caps={CAPS} ! appsink channel=p1out"
    ))
    .unwrap();
    let mut err: *mut c_char = ptr::null_mut();
    // SAFETY: valid C string; err writable.
    let p = unsafe { g2g_pipeline_launch(desc.as_ptr(), &mut err) };
    assert!(!p.is_null());

    for i in 0u8..3 {
        let buf = [i; 16];
        assert_eq!(
            // SAFETY: src live; buf covers its len (copied by push).
            unsafe { g2g_appsrc_push(src, buf.as_ptr(), 16, u64::from(i) * 1_000) },
            1
        );
    }
    // SAFETY: src live.
    unsafe { g2g_appsrc_end_of_stream(src) };

    for i in 0u8..3 {
        let mut smp: *mut Sample = ptr::null_mut();
        // SAFETY: sink live; smp writable.
        let rc = unsafe { g2g_appsink_pull(sink, &mut smp) };
        assert_eq!(rc, 1, "pulled frame {i}");
        assert!(!smp.is_null());
        // SAFETY: smp is a live sample.
        let bytes =
            unsafe { std::slice::from_raw_parts(g2g_sample_data(smp), g2g_sample_len(smp)) };
        assert_eq!(bytes, &[i; 16]);
        // SAFETY: smp live.
        assert_eq!(unsafe { g2g_sample_pts(smp) }, u64::from(i) * 1_000);
        // SAFETY: smp from a pull, freed once.
        unsafe { g2g_sample_free(smp) };
    }

    // The next pull observes end-of-stream.
    let mut smp: *mut Sample = ptr::null_mut();
    assert_eq!(
        // SAFETY: sink live; smp writable.
        unsafe { g2g_appsink_pull(sink, &mut smp) },
        0,
        "stream ended"
    );

    // SAFETY: p live.
    assert_eq!(unsafe { g2g_pipeline_wait(p, ptr::null_mut()) }, 0);
    // SAFETY: live handles, each freed once.
    unsafe {
        g2g_pipeline_free(p);
        g2g_appsrc_free(src);
        g2g_appsink_free(sink);
    }
}

#[test]
fn lend_survives_through_pull_zero_copy() {
    let free_count = Box::new(AtomicUsize::new(0));
    let free_user = (&*free_count as *const AtomicUsize) as *mut c_void;

    let cam = CString::new("p2cam").unwrap();
    let out = CString::new("p2out").unwrap();
    // SAFETY: valid C strings.
    let (src, sink) = unsafe { (g2g_appsrc_new(cam.as_ptr()), g2g_appsink_new(out.as_ptr())) };
    assert!(!src.is_null() && !sink.is_null());

    let desc = CString::new(format!(
        "appsrc channel=p2cam caps={CAPS} ! appsink channel=p2out"
    ))
    .unwrap();
    let mut err: *mut c_char = ptr::null_mut();
    // SAFETY: valid C string; err writable.
    let p = unsafe { g2g_pipeline_launch(desc.as_ptr(), &mut err) };
    assert!(!p.is_null());

    let buf: Vec<u8> = vec![7u8; 16];
    let lent_ptr = buf.as_ptr() as usize;
    // SAFETY: src live; buf valid until after the sample is freed.
    let ok = unsafe { g2g_appsrc_push_lend(src, buf.as_ptr(), 16, 0, Some(count_free), free_user) };
    assert_eq!(ok, 1);
    // SAFETY: src live.
    unsafe { g2g_appsrc_end_of_stream(src) };

    let mut smp: *mut Sample = ptr::null_mut();
    // SAFETY: sink live; smp writable.
    assert_eq!(unsafe { g2g_appsink_pull(sink, &mut smp) }, 1);
    assert!(!smp.is_null());
    // Zero-copy end to end: the pulled sample points at the lent buffer.
    assert_eq!(
        // SAFETY: smp live.
        unsafe { g2g_sample_data(smp) } as usize,
        lent_ptr,
        "pull is zero-copy"
    );
    assert_eq!(
        free_count.load(SeqCst),
        0,
        "lend held while the sample lives"
    );
    // Freeing the sample releases the frame, firing the lend's free callback.
    // SAFETY: smp from a pull, freed once.
    unsafe { g2g_sample_free(smp) };
    assert_eq!(
        free_count.load(SeqCst),
        1,
        "free fired once on sample release"
    );

    // SAFETY: p live.
    assert_eq!(unsafe { g2g_pipeline_wait(p, ptr::null_mut()) }, 0);
    // SAFETY: live handles, each freed once.
    unsafe {
        g2g_pipeline_free(p);
        g2g_appsrc_free(src);
        g2g_appsink_free(sink);
    }
    drop(buf);
}
