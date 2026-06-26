//! C ABI for glass2glass: the language-neutral waist over the `gst-launch`-style
//! DSL + element registry (PORTING.md §5, DESIGN.md §4.16).
//!
//! A non-Rust caller (C, and through it Python/C#/Swift/...) describes a
//! pipeline as a string, runs it, and watches the pipeline bus, without holding
//! any typed Rust element value. This is the by-string usage model GStreamer
//! apps reach for with `gst_parse_launch`; the typed programmatic builder stays
//! a Rust API.
//!
//! Lifecycle (first slice):
//! 1. [`g2g_pipeline_launch`] parses the description against
//!    [`default_registry`] and starts it on a background thread (the parsed
//!    graph is `'static` and, under the `multi-thread` feature, `Send`).
//! 2. [`g2g_pipeline_bus_poll`] drains one [`BusMessage`] at a time, projected
//!    into a flat [`G2gBusMessage`] (the `gst_bus_pop` analog).
//! 3. [`g2g_pipeline_wait`] blocks for the natural end of stream and yields the
//!    final [`RunStats`]; [`g2g_pipeline_free`] joins and releases.
//!
//! There is no early-cancel channel yet (the known runner gap), so `free` on a
//! still-running pipeline waits for its natural EOS. `appsrc` / `appsink`-style
//! buffer injection and extraction are the planned next slice.

use std::ffi::{c_char, c_void, CStr, CString};
use std::fmt::Write as _;
use std::os::raw::c_int;
use std::ptr;
use std::thread::JoinHandle;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{parse_launch, run_graph_with_bus, RunStats};
use g2g_core::{Bus, BusMessage, G2gError, MemoryDomain, PipelineState};
use g2g_plugins::appsink::{register_appsink_pull, set_appsink_callback, AppSinkPull, Pull, SampleCallback};
use g2g_plugins::appsrc::{register_appsrc, AppSrcFeed};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;

/// Steady-state link depth, matching `g2g-launch` (keeps latency low without
/// starving the source; see DESIGN notes on `link_capacity`).
const LINK_CAPACITY: usize = 4;
/// Bus backlog. Control messages are dropped (not blocked) when full, so a slow
/// poller never stalls the data path; this only bounds how many unread messages
/// are retained.
const BUS_CAPACITY: usize = 64;

/// Opaque pipeline handle returned to the C caller. Never inspect its fields
/// across the ABI; pass the pointer back to the `g2g_pipeline_*` functions.
#[derive(Debug)]
pub struct Pipeline {
    /// Consumer end of the pipeline bus; the producer went to the run thread.
    bus: Bus,
    /// The background run thread, taken once [`g2g_pipeline_wait`] joins it.
    join: Option<JoinHandle<Result<RunStats, G2gError>>>,
    /// Captured run result after the thread is joined.
    result: Option<Result<RunStats, G2gError>>,
    /// Backing store for the most recent bus message's text, so the `text`
    /// pointer handed out by [`g2g_pipeline_bus_poll`] stays valid until the
    /// next poll or [`g2g_pipeline_free`].
    last_text: Option<CString>,
}

/// Discriminant for [`G2gBusMessage::kind`]. Mirrors the common
/// [`BusMessage`] variants; richer ones collapse to [`G2gBusKind::Other`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum G2gBusKind {
    StreamStart = 0,
    Eos = 1,
    Error = 2,
    Warning = 3,
    Info = 4,
    StateChanged = 5,
    Buffering = 6,
    DurationChanged = 7,
    Other = 99,
}

/// A flattened pipeline bus message. `text` is borrowed from the pipeline and
/// valid only until the next [`g2g_pipeline_bus_poll`] / [`g2g_pipeline_free`];
/// copy it if you need to keep it. The `a` / `b` fields are kind-specific:
/// `Buffering` -> `a` = percent; `StateChanged` -> `a` = new state, `b` = old
/// state (0 Null, 1 Ready, 2 Paused, 3 Playing); `DurationChanged` -> `a` = ns.
#[repr(C)]
#[derive(Debug)]
pub struct G2gBusMessage {
    pub kind: c_int,
    pub text: *const c_char,
    pub a: u64,
    pub b: u64,
}

/// Frame counters reported by [`g2g_pipeline_wait`].
#[repr(C)]
#[derive(Debug)]
pub struct G2gStats {
    pub frames_emitted: u64,
    pub frames_consumed: u64,
    pub frames_dropped: u64,
}

/// Parse `description` and start it on a background thread.
///
/// On success returns a non-null [`Pipeline`] handle. On a null/invalid string
/// or a parse error returns null and, when `err_out` is non-null, writes a
/// malloc-equivalent message there (free it with [`g2g_string_free`]).
///
/// # Safety
/// `description` must be a valid NUL-terminated C string. `err_out`, if
/// non-null, must point to writable storage for one `char*`.
#[no_mangle]
pub unsafe extern "C" fn g2g_pipeline_launch(
    description: *const c_char,
    err_out: *mut *mut c_char,
) -> *mut Pipeline {
    if !err_out.is_null() {
        // SAFETY: caller contract: `err_out` points to writable `*mut c_char`.
        unsafe { *err_out = ptr::null_mut() };
    }
    if description.is_null() {
        set_err(err_out, "null pipeline description");
        return ptr::null_mut();
    }
    // SAFETY: caller contract: `description` is a valid NUL-terminated C string,
    // borrowed only for this call.
    let desc = match unsafe { CStr::from_ptr(description) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            set_err(err_out, "pipeline description is not valid UTF-8");
            return ptr::null_mut();
        }
    };

    let reg = default_registry();
    let graph = match parse_launch(&reg, desc) {
        Ok(g) => g,
        Err(e) => {
            set_err(err_out, &format!("parse error: {e}"));
            return ptr::null_mut();
        }
    };
    drop(reg);

    let (bus, bus_handle) = Bus::new(BUS_CAPACITY);
    let spawned = std::thread::Builder::new()
        .name("g2g-capi-run".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_time()
                .build()
                .expect("build tokio runtime");
            let clock = WallClock::new();
            rt.block_on(run_graph_with_bus(graph, &clock, LINK_CAPACITY, &bus_handle))
        });
    // Returning null (not panicking across the FFI boundary) is the documented
    // failure contract.
    let join = match spawned {
        Ok(j) => j,
        Err(e) => {
            set_err(err_out, &format!("failed to spawn run thread: {e}"));
            return ptr::null_mut();
        }
    };

    let p = Box::new(Pipeline { bus, join: Some(join), result: None, last_text: None });
    Box::into_raw(p)
}

/// Drain one bus message into `*out`. Returns 1 if a message was written, 0 if
/// none are pending (or on a null argument).
///
/// # Safety
/// `p` must be a handle from [`g2g_pipeline_launch`] not yet freed; `out` must
/// point to writable [`G2gBusMessage`] storage.
#[no_mangle]
pub unsafe extern "C" fn g2g_pipeline_bus_poll(p: *mut Pipeline, out: *mut G2gBusMessage) -> c_int {
    // SAFETY: caller contract: `p` is a live handle.
    let Some(p) = (unsafe { p.as_mut() }) else { return 0 };
    if out.is_null() {
        return 0;
    }
    let Some(msg) = p.bus.try_recv() else { return 0 };

    let (kind, text, a, b) = project(&msg);
    p.last_text = text.and_then(|t| CString::new(t).ok());
    let cmsg = G2gBusMessage {
        kind: kind as c_int,
        text: p.last_text.as_ref().map_or(ptr::null(), |c| c.as_ptr()),
        a,
        b,
    };
    // SAFETY: caller contract: `out` is writable.
    unsafe { *out = cmsg };
    1
}

/// Returns 1 once the run thread has finished (EOS or error), else 0.
///
/// # Safety
/// `p` must be a live handle from [`g2g_pipeline_launch`].
#[no_mangle]
pub unsafe extern "C" fn g2g_pipeline_is_done(p: *const Pipeline) -> c_int {
    // SAFETY: caller contract: `p` is a live handle.
    let Some(p) = (unsafe { p.as_ref() }) else { return 0 };
    match &p.join {
        Some(j) => c_int::from(j.is_finished()),
        None => 1,
    }
}

/// Block until the run finishes, then write final stats to `*out` (if non-null).
/// Returns 0 on a clean run, -1 if the pipeline errored or `p` is null.
/// Idempotent: subsequent calls return the captured result.
///
/// # Safety
/// `p` must be a live handle; `out`, if non-null, must point to writable
/// [`G2gStats`] storage.
#[no_mangle]
pub unsafe extern "C" fn g2g_pipeline_wait(p: *mut Pipeline, out: *mut G2gStats) -> c_int {
    // SAFETY: caller contract: `p` is a live handle.
    let Some(p) = (unsafe { p.as_mut() }) else { return -1 };
    if let Some(j) = p.join.take() {
        // A panic in the run thread surfaces as a generic shutdown error.
        p.result = Some(j.join().unwrap_or(Err(G2gError::Shutdown)));
    }
    match &p.result {
        Some(Ok(stats)) => {
            if !out.is_null() {
                let c = G2gStats {
                    frames_emitted: stats.frames_emitted,
                    frames_consumed: stats.frames_consumed,
                    frames_dropped: stats.frames_dropped,
                };
                // SAFETY: caller contract: `out` is writable.
                unsafe { *out = c };
            }
            0
        }
        _ => -1,
    }
}

/// Join the run thread (waiting for natural EOS, no early cancel yet) and free
/// the handle. Passing null is a no-op.
///
/// # Safety
/// `p` must be a handle from [`g2g_pipeline_launch`] not already freed.
#[no_mangle]
pub unsafe extern "C" fn g2g_pipeline_free(p: *mut Pipeline) {
    if p.is_null() {
        return;
    }
    // SAFETY: caller contract: `p` came from `Box::into_raw` and is freed once.
    let mut bx = unsafe { Box::from_raw(p) };
    if let Some(j) = bx.join.take() {
        let _ = j.join();
    }
}

/// Free a string returned by this library (e.g. the `err_out` of
/// [`g2g_pipeline_launch`]). Passing null is a no-op.
///
/// # Safety
/// `s` must be a pointer this library returned and not already freed.
#[no_mangle]
pub unsafe extern "C" fn g2g_string_free(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    // SAFETY: caller contract: `s` came from `CString::into_raw` in this crate.
    drop(unsafe { CString::from_raw(s) });
}

// ---- appsrc / appsink (M233) -------------------------------------------------

/// Opaque application-source handle: the push end of an `appsrc channel=<name>`.
/// Create it (and register the named feed) *before* launching the pipeline that
/// contains the matching `appsrc`.
#[derive(Debug)]
pub struct AppSrc {
    feed: AppSrcFeed,
}

/// Register an `appsrc` feed under `channel` (null -> "default") and return its
/// push handle.
///
/// # Safety
/// `channel`, if non-null, must be a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn g2g_appsrc_new(channel: *const c_char) -> *mut AppSrc {
    // SAFETY: caller contract on `channel`.
    let name = unsafe { opt_cstr(channel) }.unwrap_or("default");
    Box::into_raw(Box::new(AppSrc { feed: register_appsrc(name) }))
}

/// Push `len` bytes (copied) with timestamp `pts_ns`. Returns 1 if accepted, 0
/// if the feed is full (retry) or the pipeline is gone / `p` is null.
///
/// # Safety
/// `p` must be a live handle; `data` must point to `len` readable bytes (or be
/// null with `len == 0`).
#[no_mangle]
pub unsafe extern "C" fn g2g_appsrc_push(
    p: *const AppSrc,
    data: *const u8,
    len: usize,
    pts_ns: u64,
) -> c_int {
    // SAFETY: caller contract: `p` is a live handle.
    let Some(p) = (unsafe { p.as_ref() }) else { return 0 };
    if data.is_null() && len != 0 {
        return 0;
    }
    // SAFETY: caller contract: `data` covers `len` bytes (empty when null).
    let bytes = if len == 0 { &[][..] } else { unsafe { core::slice::from_raw_parts(data, len) } };
    c_int::from(p.feed.push(bytes, pts_ns))
}

/// Push a buffer **zero-copy**: the pipeline reads `data[..len]` directly (no
/// copy), and `free(user)` is invoked exactly once when the frame is finally
/// dropped, returning the buffer to the application. A `null` `free` lends the
/// buffer for the whole pipeline lifetime without reclamation (the application
/// guarantees it outlives the run).
///
/// Returns 1 if accepted. Returns 0 if the feed is full or closed (in which case
/// `free` still fires immediately, the lend released), or if `data` is null with
/// `len > 0` (an invalid call: the lend is not taken and `free` is not invoked).
/// A mutating element downstream transparently copies the bytes out first, so
/// the lend stays read-only.
///
/// # Safety
/// `data` must point to `len` bytes that stay valid and unmodified until `free`
/// runs; `free`/`user` must be safe to invoke once from the pipeline's thread.
#[no_mangle]
pub unsafe extern "C" fn g2g_appsrc_push_lend(
    p: *const AppSrc,
    data: *const u8,
    len: usize,
    pts_ns: u64,
    free: Option<unsafe extern "C" fn(*mut c_void)>,
    user: *mut c_void,
) -> c_int {
    // SAFETY: caller contract: `p` is a live handle.
    let Some(p) = (unsafe { p.as_ref() }) else { return 0 };
    if data.is_null() && len != 0 {
        return 0;
    }
    // SAFETY: caller contract: `data[..len]` is valid until `free(user)` runs.
    let slice = unsafe { SystemSlice::from_foreign(data, len, free, user) };
    c_int::from(p.feed.push_slice(slice, pts_ns))
}

/// Signal end-of-stream on the feed: the `appsrc` emits a final EOS. Returns 1
/// if delivered, 0 on a null/closed feed.
///
/// # Safety
/// `p` must be a live handle.
#[no_mangle]
pub unsafe extern "C" fn g2g_appsrc_end_of_stream(p: *const AppSrc) -> c_int {
    // SAFETY: caller contract: `p` is a live handle.
    let Some(p) = (unsafe { p.as_ref() }) else { return 0 };
    c_int::from(p.feed.end_of_stream())
}

/// Free an appsrc handle. Dropping it (without an explicit end-of-stream) also
/// closes the feed, so the source EOSes. Null is a no-op.
///
/// # Safety
/// `p` must be a handle from [`g2g_appsrc_new`], not already freed.
#[no_mangle]
pub unsafe extern "C" fn g2g_appsrc_free(p: *mut AppSrc) {
    if p.is_null() {
        return;
    }
    // SAFETY: caller contract: `p` came from `Box::into_raw`, freed once.
    drop(unsafe { Box::from_raw(p) });
}

/// Register the per-frame callback for `appsink channel=<name>` (null ->
/// "default"). Call before launch. The callback fires on the pipeline's run
/// thread with a borrowed view of each frame's bytes (copy to keep them); EOS is
/// signalled with `data == null, len == 0`. A null `cb` clears nothing and is
/// ignored.
///
/// # Safety
/// `channel`, if non-null, must be a valid C string; `cb`/`user` must remain
/// valid and safe to invoke from another thread until the pipeline finishes.
#[no_mangle]
pub unsafe extern "C" fn g2g_appsink_set_callback(
    channel: *const c_char,
    cb: Option<SampleCallback>,
    user: *mut c_void,
) {
    let Some(cb) = cb else { return };
    // SAFETY: caller contract on `channel`.
    let name = unsafe { opt_cstr(channel) }.unwrap_or("default");
    set_appsink_callback(name, cb, user);
}

/// Opaque application-sink pull handle: the receive end of an
/// `appsink channel=<name>` registered in pull mode.
#[derive(Debug)]
pub struct AppSink {
    pull: AppSinkPull,
}

/// Opaque pulled sample. Owns the frame, so its bytes stay valid (zero-copy,
/// including an `appsrc` foreign lend) until [`g2g_sample_free`].
#[derive(Debug)]
pub struct Sample {
    // Keeps the frame's bytes alive for the sample's lifetime; never read
    // directly (the flat view below points into it).
    _frame: Frame,
    // A materialized copy for strided `SystemView` frames, owned here so its
    // pointer stays valid; `None` for a contiguous `System` frame.
    _materialized: Option<Box<[u8]>>,
    data: *const u8,
    len: usize,
    pts_ns: u64,
}

/// Build a flat sample view over a pulled frame without copying contiguous
/// system memory (a strided view is materialized once).
fn sample_from_frame(frame: Frame) -> Sample {
    let mut materialized: Option<Box<[u8]>> = None;
    let (data, len) = match &frame.domain {
        MemoryDomain::System(s) => {
            let b = s.as_slice();
            (b.as_ptr(), b.len())
        }
        MemoryDomain::SystemView(sv) => {
            let b = sv.materialize();
            let view = (b.as_ptr(), b.len());
            materialized = Some(b);
            view
        }
        // GPU-resident frames need a download the v1 path does not do.
        _ => (ptr::null(), 0),
    };
    let pts_ns = frame.timing.pts_ns;
    Sample { _frame: frame, _materialized: materialized, data, len, pts_ns }
}

/// Register `appsink channel=<name>` (null -> "default") in pull mode and return
/// its handle. Call before launch.
///
/// # Safety
/// `channel`, if non-null, must be a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn g2g_appsink_new(channel: *const c_char) -> *mut AppSink {
    // SAFETY: caller contract on `channel`.
    let name = unsafe { opt_cstr(channel) }.unwrap_or("default");
    Box::into_raw(Box::new(AppSink { pull: register_appsink_pull(name) }))
}

/// Block until the next frame, writing an owned sample to `*out`. Returns 1 with
/// a sample, or 0 once the stream has ended (no sample written). Free the sample
/// with [`g2g_sample_free`].
///
/// # Safety
/// `sink` must be a live handle; `out` must point to writable `*mut Sample`.
#[no_mangle]
pub unsafe extern "C" fn g2g_appsink_pull(sink: *const AppSink, out: *mut *mut Sample) -> c_int {
    // SAFETY: caller contract: `sink` is a live handle.
    let Some(s) = (unsafe { sink.as_ref() }) else { return 0 };
    if out.is_null() {
        return 0;
    }
    match block_on(s.pull.pull()) {
        Some(frame) => {
            // SAFETY: caller contract: `out` is writable.
            unsafe { *out = Box::into_raw(Box::new(sample_from_frame(frame))) };
            1
        }
        None => 0,
    }
}

/// Non-blocking pull. Returns 1 with a sample in `*out`, 0 if none is pending
/// yet, or -1 once the stream has ended (or on a null argument).
///
/// # Safety
/// `sink` must be a live handle; `out` must point to writable `*mut Sample`.
#[no_mangle]
pub unsafe extern "C" fn g2g_appsink_try_pull(sink: *const AppSink, out: *mut *mut Sample) -> c_int {
    // SAFETY: caller contract: `sink` is a live handle.
    let Some(s) = (unsafe { sink.as_ref() }) else { return -1 };
    if out.is_null() {
        return -1;
    }
    match s.pull.try_pull() {
        Pull::Frame(frame) => {
            // SAFETY: caller contract: `out` is writable.
            unsafe { *out = Box::into_raw(Box::new(sample_from_frame(frame))) };
            1
        }
        Pull::Empty => 0,
        Pull::Ended => -1,
    }
}

/// Free an appsink pull handle. Dropping it closes the pull channel. Null is a
/// no-op.
///
/// # Safety
/// `sink` must be a handle from [`g2g_appsink_new`], not already freed.
#[no_mangle]
pub unsafe extern "C" fn g2g_appsink_free(sink: *mut AppSink) {
    if sink.is_null() {
        return;
    }
    // SAFETY: caller contract: `sink` came from `Box::into_raw`, freed once.
    drop(unsafe { Box::from_raw(sink) });
}

/// Pointer to a sample's bytes (valid until [`g2g_sample_free`]), or null.
///
/// # Safety
/// `s` must be a live sample handle.
#[no_mangle]
pub unsafe extern "C" fn g2g_sample_data(s: *const Sample) -> *const u8 {
    // SAFETY: caller contract: `s` is a live handle.
    unsafe { s.as_ref() }.map_or(ptr::null(), |s| s.data)
}

/// Length in bytes of a sample's data.
///
/// # Safety
/// `s` must be a live sample handle.
#[no_mangle]
pub unsafe extern "C" fn g2g_sample_len(s: *const Sample) -> usize {
    // SAFETY: caller contract: `s` is a live handle.
    unsafe { s.as_ref() }.map_or(0, |s| s.len)
}

/// Presentation timestamp (ns) of a sample.
///
/// # Safety
/// `s` must be a live sample handle.
#[no_mangle]
pub unsafe extern "C" fn g2g_sample_pts(s: *const Sample) -> u64 {
    // SAFETY: caller contract: `s` is a live handle.
    unsafe { s.as_ref() }.map_or(0, |s| s.pts_ns)
}

/// Free a sample (releasing its frame; an `appsrc` lend's free callback fires
/// here if this frame carried one). Null is a no-op.
///
/// # Safety
/// `s` must be a sample from a pull call, not already freed.
#[no_mangle]
pub unsafe extern "C" fn g2g_sample_free(s: *mut Sample) {
    if s.is_null() {
        return;
    }
    // SAFETY: caller contract: `s` came from `Box::into_raw`, freed once.
    drop(unsafe { Box::from_raw(s) });
}

/// Minimal park-based executor: drive a future to completion on the calling
/// thread. The runtime channel's recv future registers this waker, and the
/// pipeline's run thread wakes it cross-thread, so a blocking pull works without
/// pulling in a full async runtime.
fn block_on<F: core::future::Future>(fut: F) -> F::Output {
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};

    struct ThreadWaker(std::thread::Thread);
    impl Wake for ThreadWaker {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }

    let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut fut = core::pin::pin!(fut);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => std::thread::park(),
        }
    }
}

/// Borrow a C string as `&str`, or `None` if null / not UTF-8.
///
/// # Safety
/// `p`, if non-null, must be a valid NUL-terminated C string living for the
/// returned borrow.
unsafe fn opt_cstr<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        return None;
    }
    // SAFETY: caller contract: `p` is a valid C string.
    unsafe { CStr::from_ptr(p) }.to_str().ok()
}

/// Project a [`BusMessage`] into the flat C shape: a kind, optional owned text,
/// and two kind-specific numeric fields.
fn project(msg: &BusMessage) -> (G2gBusKind, Option<String>, u64, u64) {
    match msg {
        BusMessage::StreamStart => (G2gBusKind::StreamStart, None, 0, 0),
        BusMessage::Eos => (G2gBusKind::Eos, None, 0, 0),
        BusMessage::Info(s) => (G2gBusKind::Info, Some(s.clone()), 0, 0),
        BusMessage::Error(e) => (G2gBusKind::Error, Some(format_err(e)), 0, 0),
        BusMessage::Warning(e) => (G2gBusKind::Warning, Some(format_err(e)), 0, 0),
        BusMessage::StateChanged { old, new } => {
            (G2gBusKind::StateChanged, None, state_code(*new), state_code(*old))
        }
        BusMessage::Buffering { percent } => (G2gBusKind::Buffering, None, u64::from(*percent), 0),
        BusMessage::DurationChanged { duration_ns } => {
            (G2gBusKind::DurationChanged, None, *duration_ns, 0)
        }
        // Qos / Tag / NegotiationFailed / AsyncDone / Custom: surfaced as Other
        // until the C shape grows fields for them.
        _ => (G2gBusKind::Other, None, 0, 0),
    }
}

/// Stable small-integer code for a [`PipelineState`] (matches the doc on
/// [`G2gBusMessage`]).
fn state_code(s: PipelineState) -> u64 {
    match s {
        PipelineState::Null => 0,
        PipelineState::Ready => 1,
        PipelineState::Paused => 2,
        PipelineState::Playing => 3,
    }
}

/// Render a [`G2gError`] to a human string (the error type is `Debug`, not
/// `Display`-rich, so debug formatting is the faithful projection).
fn format_err(e: &G2gError) -> String {
    let mut s = String::new();
    let _ = write!(s, "{e:?}");
    s
}

/// Write an owned C string to `*err_out` for the caller to free, if `err_out`
/// is non-null.
fn set_err(err_out: *mut *mut c_char, msg: &str) {
    if err_out.is_null() {
        return;
    }
    let c = CString::new(msg).unwrap_or_default();
    // SAFETY: `err_out` non-null per the check; the caller owns the result and
    // frees it with `g2g_string_free`.
    unsafe { *err_out = c.into_raw() };
}
