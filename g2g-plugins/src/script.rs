//! Embedded Rhai scripting (M579): build a [`Graph`] from a script, the
//! logic-carrying sibling of the static [`declarative`](crate::declarative)
//! document. A JSON / YAML file describes a *fixed* graph; a script computes one,
//! so the shape can depend on the environment, a loop, or a parameter:
//!
//! ```rhai
//! // Fan a variable number of RTSP cameras into one compositor.
//! let cams = ["rtsp://a/stream", "rtsp://b/stream", "rtsp://c/stream"];
//! add("compositor", "mix");
//! for i in 0..cams.len() {
//!     let id = "cam" + i;
//!     add("rtspsrc", id);
//!     set(id, "location", cams[i]);
//!     link(id, "mix");                 // several inbounds -> `mix` is a muxer
//! }
//! add("autovideosink", "screen");
//! link("mix", "screen");
//! ```
//!
//! The script drives a small builder API (`add` / `caps` / `set` / `link` /
//! `link_leaky`) that accumulates into the same [`GraphSpec`] the declarative
//! loader uses, then the shared [`build_spec`](crate::declarative::build_spec)
//! turns that into a runnable graph. So a script and a document reach the graph
//! through one builder, one set of role / caps / policy rules.
//!
//! Rhai is pure Rust (no C toolchain, reaches the same wasm / embedded targets
//! the core does), so scripting does not compromise the portability story.
//!
//! This module also hosts the [`scriptelement`](element) runtime transform
//! (M580): an element whose *per-frame* logic is a Rhai `process(frame)`.

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::boxed::Box;
use std::format;
use std::string::{String, ToString};
use std::sync::{Arc, Mutex};
use std::vec::Vec;

use g2g_core::log::LogSource;
use g2g_core::runtime::{GraphNode, Registry};
use g2g_core::{
    g2g_error, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata,
    Frame, G2gError, Graph, HardwareError, MemoryDomain, MultiOutputElement, MultiOutputSink,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec, Rate, RawVideoFormat,
};
use rhai::{Blob, Dynamic, Engine, EvalAltResult, Scope, AST};

use crate::declarative::{build_spec, EdgeSpec, GraphSpec, NodeSpec, ScalarVal, SpecError};

/// Why a graph-building script could not produce a graph.
#[derive(Debug)]
pub enum ScriptError {
    /// The script raised an error (parse or runtime); the message is Rhai's.
    Eval(String),
    /// The script ran, but the graph it described was rejected by the builder.
    Spec(SpecError),
}

impl core::fmt::Display for ScriptError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ScriptError::Eval(m) => write!(f, "script error: {m}"),
            ScriptError::Spec(e) => write!(f, "{e}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ScriptError {}

/// Turn a Rhai scalar into the spec model's [`ScalarVal`], preserving its type so
/// the property system fixes it against the target element's declared kind.
fn scalar_from_dynamic(v: Dynamic) -> ScalarVal {
    if v.is_bool() {
        ScalarVal::Bool(v.as_bool().unwrap_or(false))
    } else if v.is_int() {
        ScalarVal::Int(v.as_int().unwrap_or(0))
    } else if v.is_float() {
        ScalarVal::Float(v.as_float().unwrap_or(0.0))
    } else {
        ScalarVal::Str(v.into_string().unwrap_or_default())
    }
}

/// Register the builder API on `engine`, backed by the shared `spec`. Every
/// function locks the spec, mutates it, and returns; the closures capture an
/// `Arc<Mutex<_>>` (Send + Sync, which the `sync` rhai build requires).
fn register_builder_api(engine: &mut Engine, spec: &Arc<Mutex<GraphSpec>>) {
    // add(element, id): declare a node built from a registered element name.
    let s = spec.clone();
    engine.register_fn(
        "add",
        move |element: &str, id: &str| -> Result<(), Box<EvalAltResult>> {
            let mut g = s.lock().unwrap();
            if g.nodes.iter().any(|n| n.id == id) {
                return Err(format!("duplicate node id '{id}'").into());
            }
            g.nodes.push(NodeSpec {
                id: id.to_string(),
                element: Some(element.to_string()),
                ..NodeSpec::default()
            });
            Ok(())
        },
    );

    // caps(id, "video/x-raw,..."): declare a capsfilter node (the declarative
    // caps shorthand). Its element is derived; the string is validated when the
    // capsfilter parses it at build time.
    let s = spec.clone();
    engine.register_fn(
        "caps",
        move |id: &str, caps: &str| -> Result<(), Box<EvalAltResult>> {
            let mut g = s.lock().unwrap();
            if g.nodes.iter().any(|n| n.id == id) {
                return Err(format!("duplicate node id '{id}'").into());
            }
            g.nodes.push(NodeSpec {
                id: id.to_string(),
                caps: Some(caps.to_string()),
                ..NodeSpec::default()
            });
            Ok(())
        },
    );

    // set(id, key, value): set a property on an already-declared node. The value
    // keeps its Rhai type (int / bool / float / string).
    let s = spec.clone();
    engine.register_fn(
        "set",
        move |id: &str, key: &str, value: Dynamic| -> Result<(), Box<EvalAltResult>> {
            let mut g = s.lock().unwrap();
            let node =
                g.nodes
                    .iter_mut()
                    .find(|n| n.id == id)
                    .ok_or_else(|| -> Box<EvalAltResult> {
                        format!("set: no node '{id}' (declare it with add / caps first)").into()
                    })?;
            node.props
                .insert(key.to_string(), scalar_from_dynamic(value));
            Ok(())
        },
    );

    // link(from, to): connect two nodes with the default (lossless) policy.
    let s = spec.clone();
    engine.register_fn("link", move |from: &str, to: &str| {
        s.lock().unwrap().edges.push(EdgeSpec {
            from: from.to_string(),
            to: to.to_string(),
            ..EdgeSpec::default()
        });
    });

    // link_leaky(from, to, policy): connect with an explicit backpressure policy
    // ("block" / "drop-oldest" / "drop-newest"). Validated at build time.
    let s = spec.clone();
    engine.register_fn("link_leaky", move |from: &str, to: &str, policy: &str| {
        s.lock().unwrap().edges.push(EdgeSpec {
            from: from.to_string(),
            to: to.to_string(),
            policy: Some(policy.to_string()),
            ..EdgeSpec::default()
        });
    });
}

/// Run a Rhai script and build the [`Graph`] it describes, constructing each
/// element by name from `registry`. The script uses the `add` / `caps` / `set` /
/// `link` / `link_leaky` builder API; its return value is ignored (the graph is
/// accumulated as a side effect), so the script body reads as a sequence of
/// build steps.
pub fn build_from_script(registry: &Registry, src: &str) -> Result<Graph<GraphNode>, ScriptError> {
    let spec = Arc::new(Mutex::new(GraphSpec::default()));
    let mut engine = Engine::new();
    register_builder_api(&mut engine, &spec);
    engine
        .run(src)
        .map_err(|e| ScriptError::Eval(e.to_string()))?;
    // The engine dropped its closures at scope end, so the Arc is unique here.
    let spec = Arc::try_unwrap(spec)
        .map(|m| m.into_inner().unwrap())
        .unwrap_or_else(|arc| arc.lock().unwrap().clone());
    build_spec(registry, &spec).map_err(ScriptError::Spec)
}

// ---------------------------------------------------------------------------
// M580: the `scriptelement` runtime transform.
// ---------------------------------------------------------------------------

/// `scriptelement`: a raw-video transform whose per-frame logic is a Rhai
/// `process(frame)` function, the runtime sibling of the graph-building script
/// above (and the pure-Rust cousin of the `pyelement` CPython host). It negotiates
/// as a same-format passthrough (raw video in, the same raw video out), and on
/// each frame hands the script a view of the buffer, then writes back whatever the
/// script returns.
///
/// The `process(frame)` contract: `frame` is a zero-copy handle to the live
/// buffer. Index it to read / write pixels in place, and read its properties:
/// - `frame[i]` / `frame[i] = v`: the byte at `i` (0..=255),
/// - `frame.len`: the buffer length in bytes,
/// - `frame.width`, `frame.height`: the fixed geometry,
/// - `frame.format`: the pixel format name (e.g. `"rgba8"`, `"nv12"`),
/// - `frame.pts`, `frame.sequence`: the presentation timestamp (ns) and index.
///
/// There is no return value: edits land directly in the frame's buffer, and a
/// script that only reads is a pure inspection pass. Example, inverting the red
/// channel of an RGBA frame:
///
/// ```rhai
/// fn process(frame) {
///     let i = 0;
///     while i < frame.len { frame[i] = 255 - frame[i]; i += 4; }
/// }
/// ```
///
/// Zero-copy (M581): the buffer is exposed through a guarded pointer, not copied
/// in and out (a full-buffer copy each way is pure waste for a script that reads
/// metadata or edits a small region, and Rhai clones a *value* argument on entry,
/// so a blob argument could not be copy-free anyway; a custom-type receiver is
/// passed by reference). The handle is valid only for the duration of the call:
/// stored past it (which Rhai's pure functions make hard in the first place), it
/// reads / writes nothing and errors, never a dangling pointer.
///
/// Only [`MemoryDomain::System`] frames are script-addressable; a GPU-resident
/// frame yields [`G2gError::UnsupportedDomain`] (a script cannot touch device
/// memory). The Rhai call runs inline on the pipeline thread (Rhai is synchronous
/// pure Rust, so unlike the GIL there is no worker thread to isolate).
pub struct ScriptElement {
    /// Inline Rhai source (the `script=` property). Takes precedence over
    /// `location` when both are set.
    script: String,
    /// Path to a `.rhai` file (the `location=` property), read at configure time.
    location: String,
    /// Caps accepted on the sink pad. Default RGBA at any geometry / rate; an
    /// upstream `capsfilter` fixes the concrete format the script sees.
    accept: Caps,
    /// The negotiated, fully fixed caps captured at configure time.
    fixed: Option<Caps>,
    configured: bool,
    emitted: u64,
    /// The compiled script + its engine and persistent scope, built at configure.
    /// `Engine`/`AST`/`Scope` are `Send` under rhai's `sync` feature, so the
    /// element satisfies the multi-thread runner's bound.
    engine: Option<Engine>,
    ast: Option<AST>,
    scope: Scope<'static>,
}

impl Default for ScriptElement {
    fn default() -> Self {
        Self::new()
    }
}

impl ScriptElement {
    /// A fresh, unconfigured element. The script is supplied later via the
    /// `script=` / `location=` properties (the launch / registry path), so
    /// construction stays cheap and infallible.
    pub fn new() -> Self {
        Self {
            script: String::new(),
            location: String::new(),
            accept: Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
            },
            fixed: None,
            configured: false,
            emitted: 0,
            engine: None,
            ast: None,
            scope: Scope::new(),
        }
    }

    /// Set the inline Rhai source programmatically.
    pub fn with_script(mut self, src: impl Into<String>) -> Self {
        self.script = src.into();
        self
    }

    /// Override the accepted sink caps (e.g. to script an NV12 element).
    pub fn with_accept(mut self, caps: Caps) -> Self {
        self.accept = caps;
        self
    }

    /// Count of frames pushed downstream. Useful in tests.
    pub fn emitted_count(&self) -> u64 {
        self.emitted
    }

    /// Run the script's `process(frame)` over one System-memory frame, the script
    /// editing the buffer in place through the zero-copy [`FrameBuf`] handle.
    fn run_script(&mut self, frame: &mut Frame) -> Result<(), G2gError> {
        let fixed = self.fixed.as_ref().ok_or(G2gError::NotConfigured)?;
        let (width, height, format) = raw_video_dims(fixed)?;
        // Read the timing before borrowing the buffer, so nothing touches `frame`
        // for the duration of the (raw-pointer-backed) call.
        let pts = frame.timing.pts_ns as i64;
        let sequence = frame.sequence as i64;

        // Zero-copy: hand the script a guarded pointer to the live buffer rather
        // than a copy. Only System memory has CPU bytes to index.
        let buf = match &mut frame.domain {
            MemoryDomain::System(s) => s.as_mut_slice(),
            _ => return Err(G2gError::UnsupportedDomain),
        };
        let guard = Arc::new(FrameGuard::new());
        guard.arm(buf.as_mut_ptr(), buf.len());
        let view = FrameBuf {
            guard: guard.clone(),
            width: width as i64,
            height: height as i64,
            format: format_name(format),
            pts,
            sequence,
        };

        // Borrows of the three element fields are disjoint, so `scope` can be
        // `&mut` while engine / ast are `&`; the block scopes them so `self` is
        // free to log afterwards.
        let called = {
            let engine = self.engine.as_ref().ok_or(G2gError::NotConfigured)?;
            let ast = self.ast.as_ref().ok_or(G2gError::NotConfigured)?;
            engine.call_fn::<Dynamic>(&mut self.scope, ast, "process", (view,))
        };
        // Invalidate the handle the instant the call returns: a `FrameBuf` the
        // script somehow kept now reads / writes nothing (a clean error), never a
        // dangling pointer. `buf` is held borrowed until here, so the pointer the
        // guard exposed has a live provenance for the whole call and nothing else
        // touched the buffer meanwhile.
        guard.disarm();
        // Keep `buf` (and thus the pointer's provenance) alive until here, past
        // the whole call above; a real read, since `drop(&mut _)` is a no-op.
        let _ = buf.len();

        match called {
            Ok(_) => Ok(()),
            Err(e) => {
                g2g_error!(self, "script process() error: {e}");
                Err(G2gError::Hardware(HardwareError::Other))
            }
        }
    }
}

/// Shared validity cell for a [`FrameBuf`]: the live buffer pointer + length
/// while a `process()` call runs, nulled the instant it returns. Because the
/// handle reaches the buffer only through this atomic cell, a `FrameBuf` stored
/// past the call reads a null pointer and errors cleanly instead of dereferencing
/// freed / reused memory, and it needs no `unsafe impl Send`.
#[derive(Debug)]
struct FrameGuard {
    ptr: AtomicPtr<u8>,
    len: AtomicUsize,
}

impl FrameGuard {
    fn new() -> Self {
        Self {
            ptr: AtomicPtr::new(core::ptr::null_mut()),
            len: AtomicUsize::new(0),
        }
    }

    /// Make the buffer live for the handle. Single-threaded with respect to the
    /// `process()` call, so `Relaxed` suffices (the `&mut` borrow provides the
    /// real ordering).
    fn arm(&self, ptr: *mut u8, len: usize) {
        self.len.store(len, Ordering::Relaxed);
        self.ptr.store(ptr, Ordering::Relaxed);
    }

    fn disarm(&self) {
        self.ptr.store(core::ptr::null_mut(), Ordering::Relaxed);
        self.len.store(0, Ordering::Relaxed);
    }
}

/// A zero-copy, script-side handle to the frame's raw bytes (M581). Index it
/// (`frame[i]`) to read / write pixels in the live buffer; read `frame.width` /
/// `.height` / `.format` / `.pts` / `.sequence` / `.len`. Cloning is a shallow
/// `Arc` bump (Rhai's clone-on-argument-entry then costs nothing), and it is
/// `Send + Sync` with no `unsafe impl` because the pointer lives behind the
/// atomic [`FrameGuard`]. Valid only for the duration of the `process()` call.
#[derive(Debug, Clone)]
struct FrameBuf {
    guard: Arc<FrameGuard>,
    width: i64,
    height: i64,
    format: String,
    pts: i64,
    sequence: i64,
}

impl FrameBuf {
    /// The byte at `i`, or a script error if the index is out of range or the
    /// handle has expired (used past the `process()` call). Per-byte scripting is
    /// convenient but interpreted: for a whole-frame transform prefer the bulk
    /// methods ([`invert`](Self::invert) / [`fill`](Self::fill) /
    /// [`apply_lut`](Self::apply_lut)), which loop at native speed.
    fn get(&self, i: i64) -> Result<i64, Box<EvalAltResult>> {
        let (ptr, len) = self.live()?;
        let i = self.bounded(i, len)?;
        // SAFETY: `ptr` is non-null only between `arm` and `disarm`, i.e. during
        // the synchronous `process()` call, where `run_script` holds the frame's
        // buffer borrowed and alive and nothing else touches it; `bounded` kept
        // `i` within the buffer length.
        Ok(unsafe { *ptr.add(i) } as i64)
    }

    /// Write the low 8 bits of `v` to byte `i`, with the same validity rules as
    /// [`get`](Self::get).
    fn set(&mut self, i: i64, v: i64) -> Result<(), Box<EvalAltResult>> {
        let (ptr, len) = self.live()?;
        let i = self.bounded(i, len)?;
        // SAFETY: as in `get`; the buffer is exclusively ours for the call (only
        // one `process()` runs at a time on the owning arm's thread), so this
        // write does not race a read elsewhere.
        unsafe { *ptr.add(i) = (v & 0xff) as u8 };
        Ok(())
    }

    /// Invert every byte (`b -> 255 - b`) at native speed. One native call over
    /// the whole buffer instead of an interpreted per-byte loop.
    fn invert(&mut self) -> Result<(), Box<EvalAltResult>> {
        for b in self.slice_mut()? {
            *b = 255 - *b;
        }
        Ok(())
    }

    /// Set every byte to the low 8 bits of `v`, at native speed.
    fn fill(&mut self, v: i64) -> Result<(), Box<EvalAltResult>> {
        let x = (v & 0xff) as u8;
        for b in self.slice_mut()? {
            *b = x;
        }
        Ok(())
    }

    /// Apply a 256-entry lookup table to every byte (`b -> lut[b]`), at native
    /// speed. The script builds the LUT once (cheap, 256 iterations) and the whole
    /// frame is transformed in one native pass: brightness, contrast, gamma,
    /// threshold, invert, posterize are all a LUT, so this turns most per-pixel
    /// value transforms into O(256) script work + one native sweep.
    fn apply_lut(&mut self, lut: Blob) -> Result<(), Box<EvalAltResult>> {
        if lut.len() != 256 {
            return Err(format!("apply_lut needs a 256-byte LUT, got {}", lut.len()).into());
        }
        for b in self.slice_mut()? {
            *b = lut[*b as usize];
        }
        Ok(())
    }

    /// Load the live `(ptr, len)`, erroring if the handle has expired.
    fn live(&self) -> Result<(*mut u8, usize), Box<EvalAltResult>> {
        let ptr = self.guard.ptr.load(Ordering::Relaxed);
        let len = self.guard.len.load(Ordering::Relaxed);
        if ptr.is_null() {
            return Err("frame buffer expired: do not keep the frame past process()".into());
        }
        Ok((ptr, len))
    }

    /// Validate an index against the buffer length.
    fn bounded(&self, i: i64, len: usize) -> Result<usize, Box<EvalAltResult>> {
        if i < 0 || i as usize >= len {
            return Err(format!("frame index {i} out of range 0..{len}").into());
        }
        Ok(i as usize)
    }

    /// The live buffer as a mutable slice for a native bulk pass. Takes `&mut
    /// self` (the bulk ops are `&mut`) so returning a `&mut` slice is sound.
    fn slice_mut(&mut self) -> Result<&mut [u8], Box<EvalAltResult>> {
        let (ptr, len) = self.live()?;
        // SAFETY: `ptr`/`len` describe the frame's buffer, valid and exclusively
        // ours for the duration of the `process()` call (see `get`); the slice
        // does not escape the native method it is used in.
        Ok(unsafe { core::slice::from_raw_parts_mut(ptr, len) })
    }
}

/// Load a script (inline `script` wins, else the `location` file) and compile it,
/// running the top level once so the script can set up persistent state. `register`
/// installs the element's custom-type API on the engine before compilation. Shared
/// by `scriptelement` and `scriptrouter`; `category` names the element in logs.
fn build_engine(
    category: &'static str,
    script: &str,
    location: &str,
    register: impl FnOnce(&mut Engine),
) -> Result<(Engine, AST, Scope<'static>), G2gError> {
    let log = g2g_core::log::Target::category(category);
    let source = if !script.is_empty() {
        script.to_string()
    } else if !location.is_empty() {
        std::fs::read_to_string(location).map_err(|e| {
            g2g_error!(log, "cannot read script '{location}': {e}");
            G2gError::Hardware(HardwareError::Io(e.raw_os_error().unwrap_or(0)))
        })?
    } else {
        g2g_error!(log, "needs a `script=` inline source or a `location=` file");
        return Err(G2gError::NotConfigured);
    };
    let mut engine = Engine::new();
    register(&mut engine);
    let ast = engine.compile(&source).map_err(|e| {
        g2g_error!(log, "script compile error: {e}");
        G2gError::Hardware(HardwareError::Other)
    })?;
    let mut scope = Scope::new();
    engine.run_ast_with_scope(&mut scope, &ast).map_err(|e| {
        g2g_error!(log, "script init error: {e}");
        G2gError::Hardware(HardwareError::Other)
    })?;
    Ok((engine, ast, scope))
}

/// Register the [`FrameBuf`] custom type and its indexer / property getters on an
/// engine, so a `process(frame)` script can index the live buffer and read its
/// geometry. Called once per compiled `scriptelement`.
fn register_frame_api(engine: &mut Engine) {
    engine.register_type_with_name::<FrameBuf>("FrameBuf");
    engine.register_get("width", |f: &mut FrameBuf| f.width);
    engine.register_get("height", |f: &mut FrameBuf| f.height);
    engine.register_get("pts", |f: &mut FrameBuf| f.pts);
    engine.register_get("sequence", |f: &mut FrameBuf| f.sequence);
    engine.register_get("format", |f: &mut FrameBuf| f.format.clone());
    engine.register_get("len", |f: &mut FrameBuf| {
        f.guard.len.load(Ordering::Relaxed) as i64
    });
    engine.register_indexer_get(|f: &mut FrameBuf, i: i64| f.get(i));
    engine.register_indexer_set(|f: &mut FrameBuf, i: i64, v: i64| f.set(i, v));
    // Native bulk ops: one call transforms the whole buffer at Rust speed, the
    // fast path for whole-frame work (a per-byte Rhai loop is ~1000x slower).
    engine.register_fn("invert", |f: &mut FrameBuf| f.invert());
    engine.register_fn("fill", |f: &mut FrameBuf, v: i64| f.fill(v));
    engine.register_fn("apply_lut", |f: &mut FrameBuf, lut: Blob| f.apply_lut(lut));
}

impl core::fmt::Debug for ScriptElement {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ScriptElement")
            .field("script_len", &self.script.len())
            .field("location", &self.location)
            .field("configured", &self.configured)
            .field("emitted", &self.emitted)
            .finish_non_exhaustive()
    }
}

impl LogSource for ScriptElement {
    fn log_category(&self) -> &'static str {
        "scriptelement"
    }
}

impl AsyncElement for ScriptElement {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Same-format passthrough: output caps equal input (when accepted), so the
    /// solver can derive this element's output edge and steer a mid-stream
    /// `CapsChanged` through it. Mirrors `pyelement`.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let accept = self.accept.clone();
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| {
            match input.intersect(&accept) {
                Ok(_) => CapsSet::one(input.clone()),
                Err(_) => CapsSet::from_alternatives(std::vec::Vec::new()),
            }
        }))
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&self.accept)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        absolute_caps.intersect(&self.accept)?;
        self.fixed = Some(absolute_caps.clone());
        let (engine, ast, scope) = build_engine(
            "scriptelement",
            &self.script,
            &self.location,
            register_frame_api,
        )?;
        self.engine = Some(engine);
        self.ast = Some(ast);
        self.scope = scope;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(mut frame) => {
                    self.run_script(&mut frame)?;
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                // Same-format transform: a change outside the accepted set is a
                // hard error; otherwise forward it so downstream stays in step.
                PipelinePacket::CapsChanged(c) => {
                    c.intersect(&self.accept)?;
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                // The runner forwards the EOS sentinel itself after process(Eos)
                // returns, so an element must NOT re-emit it (that closes the link
                // early and surfaces as Shutdown). Stateless host: nothing to drain.
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Rhai script element",
            "Filter/Effect/Video",
            "Runs a Rhai process(frame) function per frame over a raw-video buffer.",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        SCRIPTELEMENT_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "script" => {
                self.script = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            "location" => {
                self.location = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "script" => Some(PropValue::Str(self.script.clone())),
            "location" => Some(PropValue::Str(self.location.clone())),
            _ => None,
        }
    }
}

impl PadTemplates for ScriptElement {
    /// Advertise the default accepted format (RGBA, any geometry) on both pads;
    /// a same-format transform, so sink and source carry the same set.
    fn pad_templates() -> std::vec::Vec<PadTemplate> {
        let rgba = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        let set = CapsSet::one(rgba);
        std::vec::Vec::from([PadTemplate::sink(set.clone()), PadTemplate::source(set)])
    }
}

/// `ScriptElement`'s settable properties (the runtime / `gst-launch` face).
static SCRIPTELEMENT_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "script",
        PropKind::Str,
        "inline Rhai source (a `process(frame)` function)",
    ),
    PropertySpec::new(
        "location",
        PropKind::Str,
        "path to a .rhai script file (used if `script` is unset)",
    ),
];

/// The fixed geometry + format of a raw-video caps, mirroring the g2g-python host.
fn raw_video_dims(caps: &Caps) -> Result<(u32, u32, RawVideoFormat), G2gError> {
    match caps {
        Caps::RawVideo {
            width,
            height,
            format,
            ..
        } => Ok((dim_fixed(width)?, dim_fixed(height)?, *format)),
        _ => Err(G2gError::CapsMismatch),
    }
}

fn dim_fixed(d: &Dim) -> Result<u32, G2gError> {
    match d {
        Dim::Fixed(n) => Ok(*n),
        _ => Err(G2gError::CapsMismatch),
    }
}

/// The pixel-format name handed to the script (lower-case debug spelling, e.g.
/// `Rgba8` -> `"rgba8"`, `Nv12` -> `"nv12"`).
fn format_name(format: RawVideoFormat) -> String {
    format!("{format:?}").to_lowercase()
}

// ---------------------------------------------------------------------------
// M583: the `scriptrouter` routing demux.
// ---------------------------------------------------------------------------

/// A read-only, script-side handle for a routing decision: `frame.pts` /
/// `.sequence` / `.keyframe` / `.len` and `frame[i]` (peek a byte for
/// content-based routing). Reuses the [`FrameGuard`] validity cell, so bytes are
/// readable only for System memory during the `route()` call and never dangle.
#[derive(Debug, Clone)]
struct RouteFrame {
    guard: Arc<FrameGuard>,
    pts: i64,
    sequence: i64,
    keyframe: bool,
}

impl RouteFrame {
    fn get(&self, i: i64) -> Result<i64, Box<EvalAltResult>> {
        let ptr = self.guard.ptr.load(Ordering::Relaxed);
        let len = self.guard.len.load(Ordering::Relaxed);
        if ptr.is_null() {
            return Err("frame bytes not available (non-System memory, or handle expired)".into());
        }
        if i < 0 || i as usize >= len {
            return Err(format!("frame index {i} out of range 0..{len}").into());
        }
        // SAFETY: `ptr` is non-null only while `route()` runs, where `ScriptRouter`
        // holds the frame alive and borrowed; read-only, and bounds-checked.
        Ok(unsafe { *ptr.add(i as usize) } as i64)
    }
}

/// Register the [`RouteFrame`] custom type + its read-only getters / indexer.
fn register_route_api(engine: &mut Engine) {
    engine.register_type_with_name::<RouteFrame>("RouteFrame");
    engine.register_get("pts", |f: &mut RouteFrame| f.pts);
    engine.register_get("sequence", |f: &mut RouteFrame| f.sequence);
    engine.register_get("keyframe", |f: &mut RouteFrame| f.keyframe);
    engine.register_get("len", |f: &mut RouteFrame| {
        f.guard.len.load(Ordering::Relaxed) as i64
    });
    engine.register_indexer_get(|f: &mut RouteFrame, i: i64| f.get(i));
}

/// `scriptrouter`: a 1-to-N routing demux whose per-frame routing is a Rhai
/// `route(frame)` function. Each `DataFrame` goes to the output port(s) the
/// script returns; control packets (CapsChanged / Flush / Segment) broadcast to
/// every port so all branches stay configured, and the runner broadcasts EOS.
/// The scripted sibling of the built-in `Router`: put an `appsink channel=...`
/// (or anything) on each output pad and the script picks, per buffer, which
/// downstream consumer(s) receive it, e.g.
///
/// ```text
/// whepsrc uri=... ! opusdec ! audioconvert ! scriptrouter name=r \
///     script="fn route(f){ if f.sequence % 2 == 0 { 0 } else { 1 } }" \
///   r.0 ! appsink channel=even   r.1 ! appsink channel=odd
/// ```
///
/// Media-agnostic (routes audio, video, or a byte stream); a non-System frame
/// still routes by metadata, its bytes just are not peekable. `route(frame)`
/// returns:
/// - an output index in `0..outputs`: the frame goes to that one port,
/// - a negative number: the frame is dropped,
/// - an array of indices (e.g. `[0, 2]`): **multicast**, a shared copy of the
///   frame to each listed port (negatives skipped, duplicates collapsed). The
///   copy refcounts the buffer where the memory domain allows and deep-copies
///   owned CPU bytes, so a fan-out to two live consumers is honest about its cost.
pub struct ScriptRouter {
    outputs: usize,
    script: String,
    location: String,
    configured: bool,
    engine: Option<Engine>,
    ast: Option<AST>,
    scope: Scope<'static>,
}

impl ScriptRouter {
    /// A router with `outputs` output ports (the parser derives the count from the
    /// `r.0` / `r.1` / ... pad references). The script is supplied via `script=` /
    /// `location=`.
    pub fn new(outputs: usize) -> Self {
        Self {
            outputs: outputs.max(1),
            script: String::new(),
            location: String::new(),
            configured: false,
            engine: None,
            ast: None,
            scope: Scope::new(),
        }
    }

    /// Set the inline Rhai `route(frame)` source programmatically (the builder
    /// path; the launch / registry path uses the `script=` property instead).
    pub fn with_script(mut self, src: impl Into<String>) -> Self {
        self.script = src.into();
        self
    }

    /// Ask the script which port(s) this frame goes to. `route()` may return a
    /// single integer (one port, or a negative to drop) or an array of integers
    /// (multicast: a shared copy to each, negatives skipped), so the result is the
    /// de-duplicated set of target ports, empty when the frame is dropped.
    fn route(&mut self, frame: &Frame) -> Result<Vec<usize>, G2gError> {
        let pts = frame.timing.pts_ns as i64;
        let sequence = frame.sequence as i64;
        let keyframe = frame.timing.keyframe;
        let guard = Arc::new(FrameGuard::new());
        // Arm the byte-peek pointer only for System memory; other domains route by
        // metadata alone (`frame[i]` then errors, the getters still work).
        let buf: Option<&[u8]> = frame.domain.as_system_slice();
        if let Some(b) = buf {
            guard.arm(b.as_ptr() as *mut u8, b.len());
        }
        let view = RouteFrame {
            guard: guard.clone(),
            pts,
            sequence,
            keyframe,
        };
        let called = {
            let engine = self.engine.as_ref().ok_or(G2gError::NotConfigured)?;
            let ast = self.ast.as_ref().ok_or(G2gError::NotConfigured)?;
            engine.call_fn::<Dynamic>(&mut self.scope, ast, "route", (view,))
        };
        guard.disarm();
        // Keep the buffer borrow (and the peeked pointer's provenance) alive past
        // the call above.
        if let Some(b) = buf {
            let _ = b.len();
        }

        let out = match called {
            Ok(v) => v,
            Err(e) => {
                g2g_error!(self, "script route() error: {e}");
                return Err(G2gError::Hardware(HardwareError::Other));
            }
        };
        self.ports_from(out)
    }

    /// Turn a `route()` return value into the set of target ports. An array
    /// multicasts (each valid, in-range entry, negatives skipped, duplicates
    /// collapsed so a consumer never receives the same frame twice); a bare
    /// integer is one port (or a drop when negative). An out-of-range port or a
    /// non-integer entry is a script bug and fails the frame loudly.
    fn ports_from(&self, out: Dynamic) -> Result<Vec<usize>, G2gError> {
        let mut ports: Vec<usize> = Vec::new();
        let mut push = |i: i64| -> Result<(), G2gError> {
            if i < 0 {
                return Ok(()); // a negative entry drops that route
            }
            let p = i as usize;
            if p >= self.outputs {
                g2g_error!(
                    self,
                    "route() returned port {i}, out of range 0..{}",
                    self.outputs
                );
                return Err(G2gError::Hardware(HardwareError::Other));
            }
            if !ports.contains(&p) {
                ports.push(p);
            }
            Ok(())
        };

        if out.is_array() {
            // is_array() guarded, so into_array cannot fail.
            for v in out.into_array().unwrap_or_default() {
                match v.as_int() {
                    Ok(i) => push(i)?,
                    Err(actual) => {
                        g2g_error!(
                            self,
                            "route() array entry must be an integer (got {actual})"
                        );
                        return Err(G2gError::Hardware(HardwareError::Other));
                    }
                }
            }
        } else {
            match out.as_int() {
                Ok(i) => push(i)?,
                Err(actual) => {
                    g2g_error!(
                        self,
                        "route() must return an integer port or an array of ports (got {actual})"
                    );
                    return Err(G2gError::Hardware(HardwareError::Other));
                }
            }
        }
        Ok(ports)
    }
}

impl core::fmt::Debug for ScriptRouter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ScriptRouter")
            .field("outputs", &self.outputs)
            .field("script_len", &self.script.len())
            .field("location", &self.location)
            .field("configured", &self.configured)
            .finish_non_exhaustive()
    }
}

impl LogSource for ScriptRouter {
    fn log_category(&self) -> &'static str {
        "scriptrouter"
    }
}

impl MultiOutputElement for ScriptRouter {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// Pass-through: every branch carries the input caps (routing the same stream
    /// to different consumers), so it negotiates as a tee. Mirrors `Router`.
    fn caps_constraint_as_input(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (engine, ast, scope) = build_engine(
            "scriptrouter",
            &self.script,
            &self.location,
            register_route_api,
        )?;
        self.engine = Some(engine);
        self.ast = Some(ast);
        self.scope = scope;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let ports = self.outputs;
            match packet {
                PipelinePacket::DataFrame(f) => {
                    let ports = self.route(&f)?;
                    // Multicast: hand every port but the last a shared duplicate
                    // (buffer refcounted where the domain allows, owned CPU bytes
                    // deep-copied), then move the original into the last port so
                    // the single-route case still costs nothing. An empty set is a
                    // drop.
                    if let Some((&last, rest)) = ports.split_last() {
                        for &port in rest {
                            out.push_to(port, PipelinePacket::DataFrame(f.share()))
                                .await?;
                        }
                        out.push_to(last, PipelinePacket::DataFrame(f)).await?;
                    }
                }
                // Control packets broadcast to every branch, so all stay in step.
                PipelinePacket::CapsChanged(c) => {
                    for port in 0..ports {
                        out.push_to(port, PipelinePacket::CapsChanged(c.clone()))
                            .await?;
                    }
                }
                PipelinePacket::Flush => {
                    for port in 0..ports {
                        out.push_to(port, PipelinePacket::Flush).await?;
                    }
                }
                PipelinePacket::Segment(s) => {
                    for port in 0..ports {
                        out.push_to(port, PipelinePacket::Segment(s)).await?;
                    }
                }
                // The runner broadcasts EOS to every port after process() returns.
                PipelinePacket::Eos => {}
                // Future packet kinds (PipelinePacket is non_exhaustive): ignore
                // rather than guess a routing for a variant we do not understand.
                _ => {}
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        SCRIPTROUTER_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "script" => {
                self.script = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            "location" => {
                self.location = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "script" => Some(PropValue::Str(self.script.clone())),
            "location" => Some(PropValue::Str(self.location.clone())),
            _ => None,
        }
    }
}

/// `ScriptRouter`'s settable properties (the runtime / `gst-launch` face).
static SCRIPTROUTER_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "script",
        PropKind::Str,
        "inline Rhai source (a `route(frame) -> port` function)",
    ),
    PropertySpec::new(
        "location",
        PropKind::Str,
        "path to a .rhai script file (used if `script` is unset)",
    ),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::default_registry;

    #[test]
    fn script_builds_a_linear_pipeline() {
        let src = r#"
            add("videotestsrc", "src");
            set("src", "num-buffers", 3);
            add("fakesink", "sink");
            link("src", "sink");
        "#;
        let reg = default_registry();
        let graph = build_from_script(&reg, src).expect("build");
        assert_eq!(graph.edges().len(), 1);
        graph.finish().expect("valid DAG");
    }

    #[test]
    fn a_loop_generates_a_fan_in_muxer() {
        // Three sources into one funnel: the loop-generated fan-in makes
        // `mix` a muxer, so we get 4 edges (3 in + 1 out) after the builder.
        let src = r#"
            add("funnel", "mix");
            for i in 0..3 {
                let id = "src" + i;
                add("videotestsrc", id);
                set(id, "num-buffers", 1);
                link(id, "mix");
            }
            add("fakesink", "sink");
            link("mix", "sink");
        "#;
        let reg = default_registry();
        let graph = build_from_script(&reg, src).expect("build");
        assert_eq!(graph.edges().len(), 4, "3 into mix + mix->sink");
        graph.finish().expect("valid DAG");
    }

    #[test]
    fn caps_and_leaky_link_carry_through() {
        let src = r#"
            add("videotestsrc", "src");
            set("src", "num-buffers", 1);
            caps("cf", "video/x-raw,format=NV12");
            add("fakesink", "sink");
            link("src", "cf");
            link_leaky("cf", "sink", "drop-oldest");
        "#;
        let reg = default_registry();
        let graph = build_from_script(&reg, src).expect("build");
        assert_eq!(graph.edges().len(), 2);
        // The cf->sink edge carries the requested drop-oldest policy.
        assert!(graph
            .edges()
            .iter()
            .any(|e| e.policy == g2g_core::LinkPolicy::DropOldest));
        graph.finish().expect("valid DAG");
    }

    #[test]
    fn a_script_error_is_reported_not_panicked() {
        // `set` on an undeclared node raises a script error the builder surfaces.
        let src = r#" set("ghost", "num-buffers", 1); "#;
        let reg = default_registry();
        assert!(matches!(
            build_from_script(&reg, src),
            Err(ScriptError::Eval(_))
        ));
    }

    #[test]
    fn a_duplicate_id_is_reported() {
        let src = r#"
            add("videotestsrc", "x");
            add("fakesink", "x");
        "#;
        let reg = default_registry();
        assert!(matches!(
            build_from_script(&reg, src),
            Err(ScriptError::Eval(_))
        ));
    }

    // ---- M580: scriptelement runtime transform ----

    use g2g_core::{FrameTiming, SystemSlice};

    fn rgba_frame(w: u32, h: u32) -> Frame {
        let bytes = std::vec![0u8; (w * h * 4) as usize];
        Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            FrameTiming::default(),
            0,
        )
    }

    fn rgba_caps(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(30),
        }
    }

    #[test]
    fn scriptelement_mutates_pixels_in_place() {
        // A process() that writes 255 into the first byte via the zero-copy index.
        let mut el = ScriptElement::new().with_script("fn process(f) { f[0] = 255; }");
        el.configure_pipeline(&rgba_caps(2, 2))
            .expect("configure compiles the script");
        let mut frame = rgba_frame(2, 2);
        el.run_script(&mut frame).expect("run");
        assert_eq!(
            frame
                .domain
                .as_system_slice()
                .expect("expected System memory")[0],
            255,
            "pixel written in place"
        );
    }

    #[test]
    fn scriptelement_reads_back_a_pixel_it_wrote() {
        // Full round-trip through the index: invert then read confirms get + set
        // hit the same live buffer.
        let mut el = ScriptElement::new()
            .with_script("fn process(f) { f[3] = 200; if f[3] != 200 { throw \"readback\"; } }");
        el.configure_pipeline(&rgba_caps(2, 2)).expect("configure");
        let mut frame = rgba_frame(2, 2);
        el.run_script(&mut frame).expect("run");
        assert_eq!(
            frame
                .domain
                .as_system_slice()
                .expect("expected System memory")[3],
            200
        );
    }

    #[test]
    fn scriptelement_sees_geometry_and_can_passthrough() {
        // The script reads width/height (proving the view is populated) and
        // returns () for a pure inspection pass (no write-back).
        let mut el = ScriptElement::new().with_script(
            "fn process(f) { if f.width != 4 || f.height != 2 { throw \"bad dims\"; } }",
        );
        el.configure_pipeline(&rgba_caps(4, 2)).expect("configure");
        let mut frame = rgba_frame(4, 2);
        el.run_script(&mut frame)
            .expect("run: dims match, unit return is passthrough");
    }

    #[test]
    fn scriptelement_bulk_ops_transform_whole_frame() {
        // invert + apply_lut over the whole buffer via native calls (the fast
        // path). LUT here is identity-plus-one on the already-inverted bytes.
        let mut el = ScriptElement::new().with_script(
            "fn process(f) { f.fill(10); f.invert(); let lut = blob(256, 0); let i = 0; \
             while i < 256 { lut[i] = i; i += 1; } lut[245] = 99; f.apply_lut(lut); }",
        );
        el.configure_pipeline(&rgba_caps(2, 2)).expect("configure");
        let mut frame = rgba_frame(2, 2);
        el.run_script(&mut frame).expect("run");
        // fill(10) -> invert -> 245; LUT maps 245 -> 99.
        assert!(frame
            .domain
            .as_system_slice()
            .expect("expected System memory")
            .iter()
            .all(|&b| b == 99));
    }

    #[test]
    fn scriptelement_apply_lut_rejects_wrong_size() {
        let mut el = ScriptElement::new().with_script("fn process(f) { f.apply_lut(blob(4, 0)); }");
        el.configure_pipeline(&rgba_caps(2, 2)).expect("configure");
        let mut frame = rgba_frame(2, 2);
        assert!(
            el.run_script(&mut frame).is_err(),
            "a non-256-byte LUT is rejected"
        );
    }

    #[test]
    fn scriptelement_out_of_range_index_is_an_error() {
        // Indexing past the buffer is a script bug: fail loud, not a bad write.
        let mut el = ScriptElement::new().with_script("fn process(f) { f[999999] = 1; }");
        el.configure_pipeline(&rgba_caps(2, 2)).expect("configure");
        let mut frame = rgba_frame(2, 2);
        assert!(
            el.run_script(&mut frame).is_err(),
            "out-of-range index rejected"
        );
    }

    #[test]
    fn frame_handle_expires_after_the_call() {
        // The safety mechanism directly: a FrameBuf whose guard has been disarmed
        // (as run_script does when the call returns) reads / writes nothing and
        // errors, so a stashed handle can never dereference a stale pointer.
        let mut bytes = std::vec![0u8; 16];
        let guard = Arc::new(FrameGuard::new());
        guard.arm(bytes.as_mut_ptr(), bytes.len());
        let mut view = FrameBuf {
            guard: guard.clone(),
            width: 2,
            height: 2,
            format: "rgba8".into(),
            pts: 0,
            sequence: 0,
        };
        assert!(view.get(0).is_ok(), "live handle reads");
        assert!(view.set(0, 9).is_ok(), "live handle writes");
        assert_eq!(bytes[0], 9);
        guard.disarm();
        assert!(view.get(0).is_err(), "expired handle refuses to read");
        assert!(view.set(0, 1).is_err(), "expired handle refuses to write");
        assert_eq!(bytes[0], 9, "no write landed after expiry");
    }

    #[test]
    fn scriptelement_compile_error_fails_configure() {
        let mut el = ScriptElement::new().with_script("fn process(f) { this is not rhai }");
        assert!(
            el.configure_pipeline(&rgba_caps(2, 2)).is_err(),
            "bad script fails configure"
        );
    }

    #[test]
    fn scriptelement_is_registered_by_name() {
        // The launch registry exposes it, so `scriptelement script=...` parses.
        let reg = default_registry();
        assert!(reg.make_element("scriptelement").is_some());
    }

    // ---- M583: scriptrouter routing demux ----

    fn seq_frame(seq: u64) -> Frame {
        let mut f = rgba_frame(2, 2);
        f.sequence = seq;
        f
    }

    fn configured_router(outputs: usize, script: &str) -> ScriptRouter {
        let mut r = ScriptRouter::new(outputs);
        r.set_property("script", PropValue::Str(script.to_string()))
            .unwrap();
        // The router ignores the caps (pass-through), but configure compiles.
        r.configure_pipeline(&rgba_caps(2, 2))
            .expect("configure compiles the route script");
        r
    }

    #[test]
    fn scriptrouter_routes_by_the_script() {
        let mut r = configured_router(2, "fn route(f) { if f.sequence % 2 == 0 { 0 } else { 1 } }");
        assert_eq!(
            r.route(&seq_frame(0)).unwrap(),
            std::vec![0],
            "even -> port 0"
        );
        assert_eq!(
            r.route(&seq_frame(1)).unwrap(),
            std::vec![1],
            "odd -> port 1"
        );
        assert_eq!(r.route(&seq_frame(2)).unwrap(), std::vec![0]);
    }

    #[test]
    fn scriptrouter_negative_result_drops() {
        let mut r = configured_router(2, "fn route(f) { if f.sequence < 5 { -1 } else { 0 } }");
        assert!(
            r.route(&seq_frame(3)).unwrap().is_empty(),
            "below threshold is dropped"
        );
        assert_eq!(r.route(&seq_frame(9)).unwrap(), std::vec![0]);
    }

    #[test]
    fn scriptrouter_out_of_range_port_is_an_error() {
        let mut r = configured_router(2, "fn route(f) { 5 }");
        assert!(
            r.route(&seq_frame(0)).is_err(),
            "port 5 with 2 outputs is rejected"
        );
    }

    #[test]
    fn scriptrouter_can_peek_bytes_for_content_routing() {
        // Route on the first byte's value (a stand-in for content sniffing).
        let mut r = configured_router(2, "fn route(f) { if f[0] > 100 { 1 } else { 0 } }");
        let mut hi = rgba_frame(2, 2);
        if let MemoryDomain::System(s) = &mut hi.domain {
            s.as_mut_slice()[0] = 200;
        }
        assert_eq!(r.route(&hi).unwrap(), std::vec![1]);
        assert_eq!(
            r.route(&rgba_frame(2, 2)).unwrap(),
            std::vec![0],
            "zero byte -> port 0"
        );
    }

    #[test]
    fn scriptrouter_multicasts_to_an_array_of_ports() {
        // An array return fans one frame to several ports; a bare int is still one.
        let mut r = configured_router(
            3,
            "fn route(f) { if f.sequence == 0 { [0, 2] } else { 1 } }",
        );
        assert_eq!(
            r.route(&seq_frame(0)).unwrap(),
            std::vec![0, 2],
            "array -> both ports"
        );
        assert_eq!(
            r.route(&seq_frame(1)).unwrap(),
            std::vec![1],
            "int -> one port"
        );
    }

    #[test]
    fn scriptrouter_multicast_skips_negatives_and_dedups() {
        // Negative entries drop that route; a repeated port never double-delivers.
        let mut r = configured_router(3, "fn route(f) { [0, -1, 2, 0] }");
        assert_eq!(
            r.route(&seq_frame(0)).unwrap(),
            std::vec![0, 2],
            "negatives skipped, 0 once"
        );
    }

    #[test]
    fn scriptrouter_empty_array_drops() {
        let mut r = configured_router(2, "fn route(f) { [] }");
        assert!(
            r.route(&seq_frame(0)).unwrap().is_empty(),
            "an empty array drops the frame"
        );
    }

    #[test]
    fn scriptrouter_multicast_out_of_range_is_an_error() {
        let mut r = configured_router(2, "fn route(f) { [0, 5] }");
        assert!(
            r.route(&seq_frame(0)).is_err(),
            "an out-of-range array entry is rejected"
        );
    }

    #[test]
    fn scriptrouter_is_registered_as_a_demux() {
        let reg = default_registry();
        assert!(
            reg.is_demux("scriptrouter"),
            "scriptrouter registered as a fan-out demux"
        );
        assert!(reg.make_demux("scriptrouter", 2).is_some());
    }
}
