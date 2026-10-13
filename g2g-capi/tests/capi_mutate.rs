use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use g2g_capi::{
    g2g_pipeline_free, g2g_pipeline_get_property, g2g_pipeline_insert_after, g2g_pipeline_launch,
    g2g_pipeline_remove, g2g_pipeline_set_property, g2g_pipeline_wait, g2g_string_free, G2gStats,
    Pipeline,
};

const FRAMES: u64 = 90;
const SOURCE: &str = "src";

fn c(text: &str) -> CString {
    CString::new(text).unwrap()
}

// takes ownership of a library string
fn take(text: *mut c_char) -> Option<String> {
    if text.is_null() {
        return None;
    }
    // SAFETY: a non-null string the library returned, freed once here.
    let owned = unsafe { CStr::from_ptr(text) }
        .to_str()
        .unwrap()
        .to_string();
    // SAFETY: as above.
    unsafe { g2g_string_free(text) };
    Some(owned)
}

fn get_property(p: *mut Pipeline, node: &str, name: &str) -> Option<String> {
    let mut err: *mut c_char = ptr::null_mut();
    let (node, name) = (c(node), c(name));
    // SAFETY: `p` is live, the strings outlive the call, `err` is writable.
    let value = unsafe { g2g_pipeline_get_property(p, node.as_ptr(), name.as_ptr(), &mut err) };
    assert_eq!(take(err), None, "reading {name:?} was refused");
    take(value)
}

#[test]
fn inserts_tunes_and_removes_an_element_while_running() {
    let mut err: *mut c_char = ptr::null_mut();
    let line = c(&format!(
        "videotestsrc name={SOURCE} num-buffers={FRAMES} ! clocksync ! fakesink name=sink"
    ));
    // SAFETY: `line` outlives the call, `err` is writable.
    let p = unsafe { g2g_pipeline_launch(line.as_ptr(), &mut err) };
    assert!(!p.is_null(), "{:?}", take(err));

    let (source, valve) = (c(SOURCE), c("valve drop=false"));
    // SAFETY: `p` is live, the strings outlive the call, `err` is writable.
    let inserted =
        unsafe { g2g_pipeline_insert_after(p, source.as_ptr(), valve.as_ptr(), &mut err) };
    assert_eq!(take(err), None);
    let inserted = take(inserted).expect("the inserted element's name");

    assert_eq!(get_property(p, &inserted, "drop").as_deref(), Some("false"));
    assert_eq!(get_property(p, &inserted, "no-such-property"), None);

    let (node, name, value) = (c(&inserted), c("drop-mode"), c("forward-sticky-events"));
    // SAFETY: `p` is live, the strings outlive the call, `err` is writable.
    let set = unsafe {
        g2g_pipeline_set_property(p, node.as_ptr(), name.as_ptr(), value.as_ptr(), &mut err)
    };
    assert_eq!((set, take(err)), (1, None));
    assert_eq!(
        get_property(p, &inserted, "drop-mode").as_deref(),
        Some("forward-sticky-events")
    );

    // SAFETY: `p` is live, `node` outlives the call, `err` is writable.
    let removed = unsafe { g2g_pipeline_remove(p, node.as_ptr(), &mut err) };
    assert_eq!((removed, take(err)), (1, None));

    let unknown = c("no-such-node");
    // SAFETY: `p` is live, the strings outlive the call, `err` is writable.
    let refused =
        unsafe { g2g_pipeline_insert_after(p, unknown.as_ptr(), valve.as_ptr(), &mut err) };
    assert!(refused.is_null());
    let error = take(err).expect("an error for an unknown node");
    assert!(error.contains("UnknownNode"), "{error}");

    let mut stats = G2gStats {
        frames_emitted: 0,
        frames_consumed: 0,
        frames_dropped: 0,
    };
    // SAFETY: `p` is live, `stats` is writable.
    assert_eq!(unsafe { g2g_pipeline_wait(p, &mut stats) }, 0);
    assert_eq!(stats.frames_consumed, FRAMES, "no frame lost to the edits");
    // SAFETY: `p` is live, freed once here.
    unsafe { g2g_pipeline_free(p) };
}

#[test]
fn a_description_with_a_name_is_refused() {
    let mut err: *mut c_char = ptr::null_mut();
    let line = c(&format!(
        "videotestsrc name={SOURCE} num-buffers=1 ! fakesink"
    ));
    // SAFETY: `line` outlives the call, `err` is writable.
    let p = unsafe { g2g_pipeline_launch(line.as_ptr(), &mut err) };
    assert!(!p.is_null(), "{:?}", take(err));

    let (source, named) = (c(SOURCE), c("identity name=extra"));
    // SAFETY: `p` is live, the strings outlive the call, `err` is writable.
    let refused =
        unsafe { g2g_pipeline_insert_after(p, source.as_ptr(), named.as_ptr(), &mut err) };
    assert!(refused.is_null());
    let error = take(err).expect("an error for name=");
    assert!(error.contains("parse error"), "{error}");

    // SAFETY: `p` is live, freed once here.
    unsafe { g2g_pipeline_free(p) };
}
