//! `gstwrap`: host an unported GStreamer element inside a g2g graph.
//!
//! The mirror of `g2g-bridge` (DESIGN.md §7). Where the bridge embeds a g2g
//! sub-graph inside a GStreamer pipeline (adopt one g2g stage in a GStreamer
//! app), `gstwrap` embeds a GStreamer element inside a g2g graph: adopt g2g as
//! the top-level framework now and keep the stages you have not ported yet
//! running as real GStreamer elements. It is the incremental-migration path in
//! the g2g-as-host direction.
//!
//! The element drives `appsrc ! <element> ! appsink` in a real GStreamer
//! pipeline (a small C helper over the gstreamer-1.0 / gstreamer-app-1.0 C API,
//! `csrc/gstwrap_host.c`, built by build.rs). The GStreamer pipeline runs on its
//! own streaming threads; `process` feeds frames into the appsrc and drains the
//! appsink, never owning those threads.
//!
//! v1 is system-memory: input `System` frames are copied into a `GstBuffer` and
//! output samples are copied back out to `System` frames. dma-buf zero-copy
//! through `gstwrap` (import a `GstDmaBufMemory` on both sides) is future work.
//!
//! Properties:
//! - `element` (required): the GStreamer element description, e.g.
//!   `x264enc bitrate=4000` or `videoflip method=horizontal-flip`.
//! - `output-caps` (optional): the caps the hosted element produces, set for a
//!   reformatting element (an encoder, `videoscale`); omit for a caps-preserving
//!   one (`videoflip`, `videobalance`, `gamma`, a proprietary in-place filter).

use core::ffi::{c_char, c_int, c_void};
use core::future::Future;
use core::pin::Pin;
use core::time::Duration;
use core::{ptr, slice};

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use std::ffi::CString;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError,
    HardwareError, OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::capsfilter::parse_caps;
use crate::encoder_base::emit_packets;

// The C-ABI helper (csrc/gstwrap_host.c), linked when the `gstreamer` feature is
// on (build.rs). Drives `appsrc ! <element> ! appsink` and matches sync/async by
// a non-blocking try_pull, exactly as `g2g-bridge` does in the other direction.
extern "C" {
    fn g2g_gstwrap_create(
        element_desc: *const c_char,
        in_caps: *const c_char,
        out_caps: *const c_char,
    ) -> *mut c_void;
    fn g2g_gstwrap_push(w: *mut c_void, data: *const u8, len: usize, pts_ns: u64) -> c_int;
    fn g2g_gstwrap_try_pull(
        w: *mut c_void,
        out_data: *mut *mut u8,
        out_len: *mut usize,
        out_pts: *mut u64,
    ) -> c_int;
    fn g2g_gstwrap_free_buf(p: *mut u8);
    fn g2g_gstwrap_eos(w: *mut c_void);
    fn g2g_gstwrap_free(w: *mut c_void);
}

/// A raw handle to the embedded GStreamer pipeline.
#[derive(Debug, Clone, Copy)]
struct WrapPtr(*mut c_void);

// SAFETY: the pointee is a GStreamer pipeline whose appsrc feed and appsink drain
// are internally thread-safe (MT-safe); this element owns the only handle and
// touches it from a single runner task at a time (never concurrently), so moving
// it between the runtime's worker threads is sound.
unsafe impl Send for WrapPtr {}

/// Hosts an unported GStreamer element inside a g2g graph. See the module docs.
#[derive(Debug)]
pub struct GstWrap {
    /// GStreamer element description, e.g. `"x264enc bitrate=4000"`.
    element: String,
    /// Caps the hosted element produces (gst-launch syntax), for a reformatting
    /// element. `None` means caps/size-preserving (output caps == input caps).
    output_caps: Option<String>,
    /// The running pipeline, `None` until `configure_pipeline`.
    handle: Option<WrapPtr>,
    /// Caps announced downstream (once) before the first output frame: the
    /// declared `output-caps` for a reformatting element, else the input caps.
    announce_caps: Option<Caps>,
    caps_sent: bool,
    emitted: u64,
    configured: bool,
}

impl GstWrap {
    pub fn new() -> Self {
        Self {
            element: String::new(),
            output_caps: None,
            handle: None,
            announce_caps: None,
            caps_sent: false,
            emitted: 0,
            configured: false,
        }
    }
}

impl Default for GstWrap {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for GstWrap {
    fn drop(&mut self) {
        if let Some(p) = self.handle.take() {
            // SAFETY: `p` was returned by `g2g_gstwrap_create` and not yet freed;
            // `free` sets the pipeline to NULL and releases it.
            unsafe { g2g_gstwrap_free(p.0) };
        }
    }
}

static GSTWRAP_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "element",
        PropKind::Str,
        "GStreamer element description to host, e.g. \"x264enc bitrate=4000\"",
    ),
    PropertySpec::new(
        "output-caps",
        PropKind::Str,
        "caps the hosted element produces (gst-launch syntax); set for a reformatting element (encoder, videoscale), omit for a caps-preserving one",
    ),
];

/// Try to drain one processed frame. Returns the C status (`1` = got a frame,
/// `0` = none ready, `-1` = EOS) alongside the frame bytes + PTS on `1`.
fn pull_one(p: WrapPtr) -> (c_int, Option<(Vec<u8>, u64)>) {
    let mut data: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    let mut pts: u64 = 0;
    // SAFETY: the out params are valid local addresses; on `r == 1` the helper
    // sets `data` to a malloc'd block of `len` bytes we then own.
    let r = unsafe { g2g_gstwrap_try_pull(p.0, &mut data, &mut len, &mut pts) };
    if r == 1 {
        // SAFETY: `data` points to `len` initialized bytes allocated by the
        // helper; we copy them out then hand the block back to be freed.
        let v = unsafe { slice::from_raw_parts(data, len) }.to_vec();
        // SAFETY: `data` came from `try_pull` and has not been freed.
        unsafe { g2g_gstwrap_free_buf(data) };
        (r, Some((v, pts)))
    } else {
        (r, None)
    }
}

/// Drain every frame the hosted element has ready right now, without waiting.
/// A latent element (an encoder) may have none yet; that is not an error.
fn drain_ready(p: WrapPtr) -> Vec<(Vec<u8>, u64)> {
    let mut out = Vec::new();
    while let (_, Some(item)) = pull_one(p) {
        out.push(item);
    }
    out
}

/// After EOS, drain the hosted element's flushed frames, waiting for its
/// internal latency. Bounded (~5 s of 1 ms polls) so a stuck element cannot hang
/// the graph; any frames not produced by then are dropped when the pipeline is
/// torn down.
async fn drain_to_eos(p: WrapPtr) -> Vec<(Vec<u8>, u64)> {
    let mut out = Vec::new();
    for _ in 0..5000u32 {
        let (r, item) = pull_one(p);
        match item {
            Some(i) => out.push(i),
            None if r < 0 => break, // EOS: the appsink is drained.
            None => tokio::time::sleep(Duration::from_millis(1)).await, // flushing.
        }
    }
    out
}

impl AsyncElement for GstWrap {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // We hand raw bytes to the hosted element and trust it to accept them, so
        // we accept whatever upstream produces. The output shape is declared via
        // `caps_constraint_as_transform` (output-caps) or equals the input.
        Ok(upstream_caps.clone())
    }

    /// A reformatting wrap (`output-caps` set) produces the declared caps
    /// regardless of input; a preserving wrap couples input == output.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        match self.output_caps.as_deref().and_then(parse_caps) {
            Some(c) => {
                CapsConstraint::DerivedOutput(Box::new(move |_input| CapsSet::one(c.clone())))
            }
            None => CapsConstraint::IdentityAny,
        }
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if self.element.is_empty() {
            return Err(G2gError::NotConfigured);
        }
        // The caps announced downstream before the first frame: the declared
        // output caps for a reformatting element, else the (preserved) input.
        let announce = match self.output_caps.as_deref() {
            Some(s) => parse_caps(s).ok_or(G2gError::CapsMismatch)?,
            None => absolute_caps.clone(),
        };

        // g2g Caps -> GStreamer caps string for the appsrc; `output-caps` (raw
        // gst-launch syntax) is passed through as the appsink filter.
        let in_caps = absolute_caps.to_gst_string();
        let element_c = CString::new(self.element.as_str()).map_err(|_| G2gError::CapsMismatch)?;
        let in_c = CString::new(in_caps).map_err(|_| G2gError::CapsMismatch)?;
        let out_c = match self.output_caps.as_deref() {
            Some(s) => Some(CString::new(s).map_err(|_| G2gError::CapsMismatch)?),
            None => None,
        };
        let out_ptr = out_c.as_ref().map_or(ptr::null(), |c| c.as_ptr());

        // SAFETY: all three are valid NUL-terminated C strings that outlive the
        // call; `create` copies what it needs and returns NULL on any failure.
        let h = unsafe { g2g_gstwrap_create(element_c.as_ptr(), in_c.as_ptr(), out_ptr) };
        if h.is_null() {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        self.handle = Some(WrapPtr(h));
        self.announce_caps = Some(announce);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "GStreamer element host",
            "Bridge/Wrapper",
            "Hosts an unported GStreamer element (appsrc ! <element> ! appsink) inside a g2g graph",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        GSTWRAP_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "element" => {
                self.element = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            "output-caps" => {
                self.output_caps = Some(value.as_str().ok_or(PropError::Type)?.into());
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "element" => Some(PropValue::Str(self.element.clone())),
            "output-caps" => self.output_caps.clone().map(PropValue::Str),
            _ => None,
        }
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
            let p = self.handle.ok_or(G2gError::NotConfigured)?;
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let bytes = slice;
                    // SAFETY: `p` is valid; `bytes` is valid for `bytes.len()`;
                    // `push` copies the bytes into a GstBuffer.
                    let r = unsafe {
                        g2g_gstwrap_push(p.0, bytes.as_ptr(), bytes.len(), frame.timing.pts_ns)
                    };
                    if r != 0 {
                        return Err(G2gError::Hardware(HardwareError::Other));
                    }
                    let packets = drain_ready(p);
                    // `announce_caps` is Some after `configure_pipeline`; cloning
                    // releases the borrow so `emit_packets` can take `&mut self`.
                    let caps = self.announce_caps.clone().ok_or(G2gError::NotConfigured)?;
                    emit_packets(&mut self.caps_sent, &mut self.emitted, packets, &caps, out)
                        .await?;
                }
                PipelinePacket::Eos => {
                    // SAFETY: `p` is valid; signals EOS on the appsrc feed.
                    unsafe { g2g_gstwrap_eos(p.0) };
                    let packets = drain_to_eos(p).await;
                    let caps = self.announce_caps.clone().ok_or(G2gError::NotConfigured)?;
                    emit_packets(&mut self.caps_sent, &mut self.emitted, packets, &caps, out)
                        .await?;
                    // The runner forwards the EOS sentinel after `process(Eos)`.
                }
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}
