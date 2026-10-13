use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use g2g_capi::{
    g2g_pipeline_add_branch, g2g_pipeline_add_input, g2g_pipeline_free, g2g_pipeline_get_property,
    g2g_pipeline_insert_after, g2g_pipeline_insert_chain_after, g2g_pipeline_launch,
    g2g_pipeline_remove, g2g_pipeline_remove_branch, g2g_pipeline_set_property, g2g_pipeline_wait,
    g2g_string_free, g2g_strv_free, G2gStats, Pipeline,
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

// takes ownership of a library name array
fn take_names(names: *mut *mut c_char) -> Option<Vec<String>> {
    if names.is_null() {
        return None;
    }
    let mut owned = Vec::new();
    // SAFETY: a null-terminated array the library returned, read before it is freed.
    while let Some(name) = unsafe { (*names.add(owned.len())).as_ref() } {
        // SAFETY: each entry is a valid C string.
        let name = unsafe { CStr::from_ptr(name) };
        owned.push(name.to_str().unwrap().to_string());
    }
    // SAFETY: as above, freed once here.
    unsafe { g2g_strv_free(names) };
    Some(owned)
}

fn launch(line: &str) -> *mut Pipeline {
    let mut err: *mut c_char = ptr::null_mut();
    let line = c(line);
    // SAFETY: `line` outlives the call, `err` is writable.
    let p = unsafe { g2g_pipeline_launch(line.as_ptr(), &mut err) };
    assert!(!p.is_null(), "{:?}", take(err));
    p
}

fn wait_consumed(p: *mut Pipeline) -> u64 {
    let mut stats = G2gStats {
        frames_emitted: 0,
        frames_consumed: 0,
        frames_dropped: 0,
    };
    // SAFETY: `p` is live, `stats` is writable.
    assert_eq!(unsafe { g2g_pipeline_wait(p, &mut stats) }, 0);
    // SAFETY: `p` is live, freed once here.
    unsafe { g2g_pipeline_free(p) };
    stats.frames_consumed
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

#[test]
fn adds_a_chain_and_a_branch_while_running() {
    let p = launch(&format!(
        "videotestsrc name={SOURCE} num-buffers={FRAMES} ! clocksync ! tee name=t ! fakesink"
    ));
    let mut err: *mut c_char = ptr::null_mut();

    let (source, chain) = (c(SOURCE), c("identity ! valve drop=false"));
    // SAFETY: `p` is live, the strings outlive the call, `err` is writable.
    let names =
        unsafe { g2g_pipeline_insert_chain_after(p, source.as_ptr(), chain.as_ptr(), &mut err) };
    assert_eq!(take(err), None);
    let names = take_names(names).expect("the chain's names");
    assert_eq!(names.len(), 2, "{names:?}");
    assert_eq!(get_property(p, &names[1], "drop").as_deref(), Some("false"));

    let (tee, branch) = (c("t"), c("identity ! fakesink"));
    // SAFETY: `p` is live, the strings outlive the call, `err` is writable.
    let names = unsafe { g2g_pipeline_add_branch(p, tee.as_ptr(), branch.as_ptr(), &mut err) };
    assert_eq!(take(err), None);
    let names = take_names(names).expect("the branch's names");
    assert_eq!(names.len(), 2, "{names:?}");

    let sink = c(&names[1]);
    // SAFETY: `p` is live, `sink` outlives the call, `err` is writable.
    let removed = unsafe { g2g_pipeline_remove_branch(p, sink.as_ptr(), &mut err) };
    assert_eq!((removed, take(err)), (1, None));

    let input = c("videotestsrc");
    // SAFETY: `p` is live, the strings outlive the call, `err` is writable.
    let added = unsafe { g2g_pipeline_add_input(p, tee.as_ptr(), input.as_ptr(), &mut err) };
    assert!(added.is_null());
    let error = take(err).expect("an error for a tee taking an input");
    assert!(error.contains("NotMutable"), "{error}");

    assert_eq!(wait_consumed(p), FRAMES, "no frame lost to the edits");
}

#[test]
fn an_insert_that_changes_the_format_gets_a_converter() {
    let p = launch(&format!(
        "videotestsrc name={SOURCE} num-buffers={FRAMES} ! video/x-raw,format=RGBA ! clocksync ! fakesink"
    ));
    let mut err: *mut c_char = ptr::null_mut();
    let (source, i420) = (c(SOURCE), c("videoconvert format=I420"));
    // SAFETY: `p` is live, the strings outlive the call, `err` is writable.
    let inserted =
        unsafe { g2g_pipeline_insert_after(p, source.as_ptr(), i420.as_ptr(), &mut err) };
    assert_eq!(take(err), None, "the handle spliced no converter");
    assert!(take(inserted).is_some());
    assert_eq!(wait_consumed(p), FRAMES);
}
