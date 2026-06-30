//! C-ABI surface the GStreamer GObject shell (`csrc/gstglass2glass.c`) calls.
//!
//! The shell owns all GObject / GStreamer boilerplate (type registration, pad
//! templates, vmethod thunks); it includes the real GStreamer headers, so the
//! struct layouts the C compiler sees are correct by construction. This module
//! is the *only* g2g-side FFI: opaque handle in, bytes in/out. Everything async
//! lives behind [`BridgeGraph`].
//!
//! Gated by the `gstreamer` feature (which the C shim build also keys off).

use core::ffi::{c_char, c_int, c_void};
use core::ptr;
use core::slice;

use g2g_core::Frame;

use crate::bridge::{frame_bytes, BridgeGraph};

// ---- GStreamer plugin entry points ------------------------------------------
//
// Authored in Rust on purpose: rustc exports only its own `#[no_mangle]` symbols
// from a cdylib and localizes anything pulled from a statically-linked C archive,
// so a C `GST_PLUGIN_DEFINE` descriptor is invisible to GStreamer's loader. The
// loader derives the plugin name from the `libgst<name>.so` filename and calls
// `gst_plugin_<name>_get_desc`; for this plugin the file is `libgstglass2glass.so`
// and the name is `glass2glass`. The heavy GObject machinery (the element type,
// pad templates, vmethods) stays in C and is reached only through the
// `plugin_init` function pointer in the descriptor below.

/// `GstPluginDesc` (gst/gstplugin.h). ABI-stable across GStreamer 1.x: two
/// `gint`, the string/init fields, then `gpointer _gst_reserved[GST_PADDING]`
/// (`GST_PADDING` == 4).
#[repr(C)]
#[derive(Debug)]
struct GstPluginDesc {
    major_version: i32,
    minor_version: i32,
    name: *const c_char,
    description: *const c_char,
    plugin_init: unsafe extern "C" fn(*mut c_void) -> i32,
    version: *const c_char,
    license: *const c_char,
    source: *const c_char,
    package: *const c_char,
    origin: *const c_char,
    release_datetime: *const c_char,
    _gst_reserved: [*mut c_void; 4],
}

// The descriptor holds raw pointers (to static C strings), so it is not `Sync` by
// default; the wrapper certifies the statics it points at are immutable and
// 'static, which they are.
struct PluginDesc(GstPluginDesc);
// SAFETY: every pointer field targets a 'static, immutable NUL-terminated byte
// string literal (or a 'static fn); the struct is never mutated after init.
unsafe impl Sync for PluginDesc {}

impl core::fmt::Debug for PluginDesc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("PluginDesc(..)")
    }
}

extern "C" {
    /// The real `plugin_init` (registers the `glass2glass` element + its GObject
    /// type); defined in `csrc/gstglass2glass.c`, reached via the descriptor.
    fn glass2glass_plugin_init(plugin: *mut c_void) -> i32;

    /// Static-registration entry GStreamer offers for non-dynamic linking.
    fn gst_plugin_register_static(
        major_version: i32,
        minor_version: i32,
        name: *const c_char,
        description: *const c_char,
        init_func: unsafe extern "C" fn(*mut c_void) -> i32,
        version: *const c_char,
        license: *const c_char,
        source: *const c_char,
        package: *const c_char,
        origin: *const c_char,
    ) -> i32;
}

// GStreamer core version this plugin is compiled against (gst/gstversion.h).
const GST_VERSION_MAJOR: i32 = 1;
const GST_VERSION_MINOR: i32 = 26;

// `b"...\0"` literals (not `c"..."`, which is 1.77+; MSRV here is 1.75) give
// 'static NUL-terminated C strings.
const NAME: &[u8] = b"glass2glass\0";
const DESCRIPTION: &[u8] = b"Embed glass2glass sub-graphs in a GStreamer pipeline\0";
const VERSION: &[u8] = b"0.1.0\0";
const LICENSE: &[u8] = b"LGPL\0";
const SOURCE: &[u8] = b"glass2glass\0";
const PACKAGE: &[u8] = b"glass2glass\0";
const ORIGIN: &[u8] = b"https://github.com/Glass2GlassHQ\0";

static DESC: PluginDesc = PluginDesc(GstPluginDesc {
    major_version: GST_VERSION_MAJOR,
    minor_version: GST_VERSION_MINOR,
    name: NAME.as_ptr().cast::<c_char>(),
    description: DESCRIPTION.as_ptr().cast::<c_char>(),
    plugin_init: glass2glass_plugin_init,
    version: VERSION.as_ptr().cast::<c_char>(),
    license: LICENSE.as_ptr().cast::<c_char>(),
    source: SOURCE.as_ptr().cast::<c_char>(),
    package: PACKAGE.as_ptr().cast::<c_char>(),
    origin: ORIGIN.as_ptr().cast::<c_char>(),
    release_datetime: ptr::null(),
    _gst_reserved: [ptr::null_mut(); 4],
});

/// Plugin descriptor accessor: GStreamer's dynamic loader resolves this by name
/// (from the `libgstglass2glass.so` filename) after dlopen.
#[no_mangle]
pub extern "C" fn gst_plugin_glass2glass_get_desc() -> *const c_void {
    core::ptr::addr_of!(DESC.0).cast::<c_void>()
}

/// Static-registration entry point (used when the plugin is linked, not loaded).
#[no_mangle]
pub extern "C" fn gst_plugin_glass2glass_register() {
    // SAFETY: all arguments are 'static C strings / a valid init fn; the call is
    // GStreamer's documented static-plugin registration.
    unsafe {
        gst_plugin_register_static(
            GST_VERSION_MAJOR,
            GST_VERSION_MINOR,
            NAME.as_ptr().cast::<c_char>(),
            DESCRIPTION.as_ptr().cast::<c_char>(),
            glass2glass_plugin_init,
            VERSION.as_ptr().cast::<c_char>(),
            LICENSE.as_ptr().cast::<c_char>(),
            SOURCE.as_ptr().cast::<c_char>(),
            PACKAGE.as_ptr().cast::<c_char>(),
            ORIGIN.as_ptr().cast::<c_char>(),
        );
    }
}

/// Opaque handle the C shell stores per element instance. A boxed
/// [`BridgeGraph`]; created at `set_caps`, destroyed at `stop`.
#[derive(Debug)]
pub struct G2gBridge(BridgeGraph);

/// An output frame lent to C: a borrowed view plus the owning boxed [`Frame`].
/// C copies `data[..len]` into its `GstBuffer`, then calls
/// [`g2g_bridge_out_release`] to drop the frame.
#[repr(C)]
#[derive(Debug)]
pub struct G2gOut {
    data: *const u8,
    len: usize,
    pts_ns: u64,
    /// Type-erased `Box<Frame>` keeping `data` alive until release.
    owner: *mut c_void,
}

/// Turn a serialized `GstCaps` into the form g2g's caps reader and launch DSL
/// accept: `video/x-raw, format=(string)RGBA, width=(int)1280, ...` ->
/// `video/x-raw,format=RGBA,width=1280,...`. Two transforms:
///
/// - drop every `(type)` annotation (a media-caps value never legitimately
///   contains parentheses), and
/// - drop all whitespace, because the launch DSL tokenizes on spaces, so a
///   `caps=` value with the spaces GStreamer inserts after commas would split
///   into separate launch tokens (PORTING.md: no quoted values with spaces).
///
/// Fields g2g does not model (`multiview-mode`, `pixel-aspect-ratio`, ...) are
/// carried through harmlessly; the caps reader ignores unknown fields.
fn normalize_gst_caps(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth = 0u32;
    for ch in s.chars() {
        match ch {
            '(' => depth += 1,
            ')' if depth > 0 => depth -= 1,
            _ if depth > 0 => {}
            _ if ch.is_whitespace() => {}
            _ => out.push(ch),
        }
    }
    out
}

/// Build an embedded sub-graph for `appsrc ! <fragment> ! appsink` with the
/// given input caps. Returns null on a null/invalid argument or a parse error
/// (the shell logs and fails `set_caps`).
///
/// # Safety
/// `fragment` and `caps` must be valid NUL-terminated C strings (or null),
/// borrowed only for this call.
#[no_mangle]
pub unsafe extern "C" fn g2g_bridge_create(
    fragment: *const c_char,
    caps: *const c_char,
) -> *mut G2gBridge {
    // SAFETY: caller contract on the two C strings.
    let (Some(fragment), Some(caps)) =
        (unsafe { opt_str(fragment) }, unsafe { opt_str(caps) })
    else {
        return ptr::null_mut();
    };
    let caps = normalize_gst_caps(caps);
    match BridgeGraph::new(fragment, &caps) {
        Ok(g) => Box::into_raw(Box::new(G2gBridge(g))),
        Err(_) => ptr::null_mut(),
    }
}

/// Push one buffer (copied) into the sub-graph, retrying briefly if the feed is
/// momentarily full. Returns 1 on success, 0 if it stayed full (the graph is
/// wedged) or the handle is null.
///
/// # Safety
/// `bridge` must be a live handle from [`g2g_bridge_create`]; `data` must point
/// to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn g2g_bridge_push_buf(
    bridge: *mut G2gBridge,
    data: *const u8,
    len: usize,
    pts_ns: u64,
) -> c_int {
    // SAFETY: caller contract.
    let Some(bridge) = (unsafe { bridge.as_ref() }) else { return 0 };
    // SAFETY: caller guarantees `len` readable bytes at `data`.
    let bytes = unsafe { slice::from_raw_parts(data, len) };
    // The 1-in / 1-out shell pulls after every push, so the bounded feed only
    // fills under a transient stall; a short bounded retry rides it out without
    // a hard busy-loop.
    for _ in 0..1000 {
        if bridge.0.push(bytes, pts_ns) {
            return 1;
        }
        std::thread::sleep(core::time::Duration::from_micros(100));
    }
    0
}

/// Block until the next processed frame and lend it to C via `*out`. Returns 1
/// with `*out` filled, -1 at end-of-stream (or null handle), -2 if the frame is
/// GPU-resident (the v1 shell handles only system memory; download upstream).
///
/// # Safety
/// `bridge` must be a live handle; `out` must point to writable [`G2gOut`].
#[no_mangle]
pub unsafe extern "C" fn g2g_bridge_pull_buf(bridge: *mut G2gBridge, out: *mut G2gOut) -> c_int {
    // SAFETY: caller contract.
    let Some(bridge) = (unsafe { bridge.as_ref() }) else { return -1 };
    let Some(frame) = bridge.0.pull_blocking() else { return -1 };
    let boxed = Box::new(frame);
    let Some(bytes) = frame_bytes(&boxed) else { return -2 };
    let cout = G2gOut {
        data: bytes.as_ptr(),
        len: bytes.len(),
        pts_ns: boxed.timing.pts_ns,
        owner: Box::into_raw(boxed).cast::<c_void>(),
    };
    // SAFETY: caller contract: `out` is writable.
    unsafe { *out = cout };
    1
}

/// Release a frame lent by [`g2g_bridge_pull_buf`]. Passing a zeroed/owner-null
/// `out` is a no-op.
///
/// # Safety
/// `out` must point to a [`G2gOut`] filled by [`g2g_bridge_pull_buf`] and not
/// already released.
#[no_mangle]
pub unsafe extern "C" fn g2g_bridge_out_release(out: *mut G2gOut) {
    // SAFETY: caller contract.
    let Some(out) = (unsafe { out.as_mut() }) else { return };
    if out.owner.is_null() {
        return;
    }
    // SAFETY: `owner` came from `Box::into_raw(Box<Frame>)` in pull_buf.
    drop(unsafe { Box::from_raw(out.owner.cast::<Frame>()) });
    out.owner = ptr::null_mut();
    out.data = ptr::null();
    out.len = 0;
}

/// Signal EOS, join the run thread, and free the handle. Passing null is a
/// no-op.
///
/// # Safety
/// `bridge` must be a handle from [`g2g_bridge_create`] not already destroyed.
#[no_mangle]
pub unsafe extern "C" fn g2g_bridge_destroy(bridge: *mut G2gBridge) {
    if bridge.is_null() {
        return;
    }
    // SAFETY: caller contract: `bridge` came from `Box::into_raw` and is freed
    // once. Dropping the inner `BridgeGraph` signals EOS and joins its thread.
    drop(unsafe { Box::from_raw(bridge) });
}

/// Borrow a C string as `&str`, or `None` for null / non-UTF-8.
///
/// # Safety
/// `p`, if non-null, must be a valid NUL-terminated C string.
unsafe fn opt_str<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        return None;
    }
    // SAFETY: caller contract on `p`.
    unsafe { core::ffi::CStr::from_ptr(p) }.to_str().ok()
}
