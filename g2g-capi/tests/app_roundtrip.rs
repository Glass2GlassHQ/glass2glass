//! appsrc -> appsink round trip through the C ABI: the application pushes
//! buffers in one end and receives them at the other, all by-name through the
//! launch DSL, exactly as a C caller would.

use std::ffi::{c_void, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::Mutex;

use g2g_capi::{
    g2g_appsink_set_callback, g2g_appsrc_end_of_stream, g2g_appsrc_free, g2g_appsrc_new,
    g2g_appsrc_push, g2g_pipeline_free, g2g_pipeline_launch, g2g_pipeline_wait, G2gStats,
};

/// What the appsink callback records, behind a mutex (the callback fires on the
/// pipeline's run thread; the test reads after `g2g_pipeline_wait` joins it).
#[derive(Default)]
struct Recorder {
    frames: Vec<(Vec<u8>, u64)>,
    eos: bool,
}

extern "C" fn on_sample(data: *const u8, len: usize, pts_ns: u64, user: *mut c_void) {
    // SAFETY: `user` is the &Mutex<Recorder> we registered; valid for the run.
    let rec = unsafe { &*(user as *const Mutex<Recorder>) };
    let mut g = rec.lock().unwrap();
    if data.is_null() && len == 0 {
        g.eos = true;
        return;
    }
    // SAFETY: appsink hands us `len` readable bytes for the call's duration.
    let bytes = unsafe { std::slice::from_raw_parts(data, len) }.to_vec();
    g.frames.push((bytes, pts_ns));
}

#[test]
fn appsrc_to_appsink_roundtrip() {
    let recorder = Box::new(Mutex::new(Recorder::default()));
    let user = (&*recorder as *const Mutex<Recorder>) as *mut c_void;

    let cam = CString::new("cam").unwrap();
    let out = CString::new("out").unwrap();
    // SAFETY: valid C strings; callback + user outlive the pipeline.
    let src = unsafe { g2g_appsrc_new(cam.as_ptr()) };
    assert!(!src.is_null());
    // SAFETY: valid C string; callback + user outlive the pipeline.
    unsafe { g2g_appsink_set_callback(out.as_ptr(), Some(on_sample), user) };

    let desc = CString::new(
        "appsrc channel=cam caps=video/x-raw,format=RGBA,width=2,height=2,framerate=30/1 \
         ! appsink channel=out",
    )
    .unwrap();
    let mut err: *mut c_char = ptr::null_mut();
    // SAFETY: valid C string; err writable.
    let p = unsafe { g2g_pipeline_launch(desc.as_ptr(), &mut err) };
    assert!(!p.is_null(), "launch failed: err set = {}", !err.is_null());

    // Push three 2x2 RGBA frames (16 bytes each), then end the stream.
    for i in 0u8..3 {
        let buf = [i; 16];
        // SAFETY: `src` live, `buf` covers its len.
        let ok = unsafe { g2g_appsrc_push(src, buf.as_ptr(), buf.len(), u64::from(i) * 1_000) };
        assert_eq!(ok, 1, "push {i} accepted");
    }
    // SAFETY: `src` live.
    unsafe { g2g_appsrc_end_of_stream(src) };

    let mut stats = G2gStats { frames_emitted: 0, frames_consumed: 0, frames_dropped: 0 };
    // SAFETY: `p` live, `stats` writable. Blocks until EOS, joining the run thread.
    let rc = unsafe { g2g_pipeline_wait(p, &mut stats) };
    assert_eq!(rc, 0, "clean run");
    assert_eq!(stats.frames_consumed, 3);

    let g = recorder.lock().unwrap();
    assert_eq!(g.frames.len(), 3, "appsink saw every pushed frame");
    assert_eq!(g.frames[0].0, vec![0u8; 16], "bytes round-tripped");
    assert_eq!(g.frames[2].0, vec![2u8; 16]);
    assert_eq!(g.frames[1].1, 1_000, "pts carried through");
    assert!(g.eos, "appsink got the EOS marker");
    drop(g);

    // SAFETY: live handles, each freed once.
    unsafe {
        g2g_pipeline_free(p);
        g2g_appsrc_free(src);
    }
}
