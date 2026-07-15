//! Exercises the C ABI through its real entry points (called as Rust fns, the
//! same symbols a C caller links). Mocks nothing: it parses, runs, and drains
//! the bus of an actual pipeline.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use g2g_capi::{
    g2g_pipeline_bus_poll, g2g_pipeline_free, g2g_pipeline_launch, g2g_pipeline_wait,
    g2g_string_free, G2gBusMessage, G2gStats,
};

fn launch(desc: &str, err: &mut *mut c_char) -> *mut g2g_capi::Pipeline {
    let c = CString::new(desc).unwrap();
    // SAFETY: `c` is a valid C string living for the call; `err` is writable.
    unsafe { g2g_pipeline_launch(c.as_ptr(), err) }
}

#[test]
fn launch_runs_to_eos_and_reports_stats() {
    let mut err: *mut c_char = ptr::null_mut();
    let p = launch("videotestsrc num-buffers=3 ! videoconvert ! fakesink", &mut err);
    assert!(!p.is_null(), "launch returned null for a valid pipeline");
    assert!(err.is_null(), "err set on success");

    let mut stats = G2gStats { frames_emitted: 0, frames_consumed: 0, frames_dropped: 0 };
    // SAFETY: `p` is a live handle, `stats` is writable.
    let rc = unsafe { g2g_pipeline_wait(p, &mut stats) };
    assert_eq!(rc, 0, "clean run");
    assert_eq!(stats.frames_consumed, 3, "fakesink consumed every test frame");

    // The bus drains without crashing and terminates (poll returns 0 when empty).
    let mut msg = G2gBusMessage { kind: 0, text: ptr::null(), a: 0, b: 0 };
    let mut drained = 0;
    // SAFETY: `p` live, `msg` writable.
    while unsafe { g2g_pipeline_bus_poll(p, &mut msg) } == 1 {
        drained += 1;
        assert!(drained < 10_000, "bus_poll never drained");
    }

    // SAFETY: `p` is a live handle, freed exactly once here.
    unsafe { g2g_pipeline_free(p) };
}

#[test]
fn parse_error_returns_null_and_sets_message() {
    let mut err: *mut c_char = ptr::null_mut();
    let p = launch("nosuchelement12345 ! fakesink", &mut err);
    assert!(p.is_null(), "bad pipeline must not launch");
    assert!(!err.is_null(), "err message set on parse failure");

    // SAFETY: `err` was set by the library to a valid C string.
    let text = unsafe { CStr::from_ptr(err) }.to_str().unwrap();
    assert!(text.contains("parse error"), "got: {text}");

    // SAFETY: `err` came from the library; freed once.
    unsafe { g2g_string_free(err) };
}

#[test]
fn null_arguments_are_handled() {
    let mut err: *mut c_char = ptr::null_mut();
    // SAFETY: null description is an explicitly supported input.
    let p = unsafe { g2g_pipeline_launch(ptr::null(), &mut err) };
    assert!(p.is_null());
    assert!(!err.is_null());
    // SAFETY: library-owned string.
    unsafe { g2g_string_free(err) };

    // Null handle / string frees are no-ops, not crashes.
    // SAFETY: null is the documented no-op input.
    unsafe {
        g2g_pipeline_free(ptr::null_mut());
        g2g_string_free(ptr::null_mut());
    }
}
