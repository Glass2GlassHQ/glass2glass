//! `gstwrap`: host an unported GStreamer element inside a g2g graph.
//!
//! The mirror of `g2g-bridge` (design/README.md). Where the bridge embeds a g2g
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
//! Input `System` frames are copied into a `GstBuffer`. A `DmaBuf` frame is
//! pushed without a copy as a `GstDmaBufMemory` over a dup of its fd, with a
//! `GstVideoMeta` carrying its stride and offset. The appsrc caps stay plain
//! system caps, so an element without dma-buf support maps the memory, which
//! works for a linear buffer.
//!
//! `output-memory` picks the output side. `system` (the default) copies each
//! sample out to a `System` frame. `dmabuf` hands each sample's dma-buf on as a
//! `DmaBuf` frame and fails the stream on a sample in any other memory. In
//! `dmabuf` mode the element also asks its upstream for `DmaBuf` frames.
//!
//! Properties:
//! - `element` (required): the GStreamer element description, e.g.
//!   `x264enc bitrate=4000` or `videoflip method=horizontal-flip`.
//! - `output-caps` (optional): the caps the hosted element produces, set for a
//!   reformatting element (an encoder, `videoscale`); omit for a caps-preserving
//!   one (`videoflip`, `videobalance`, `gamma`, a proprietary in-place filter).
//! - `output-memory` (optional): `system` or `dmabuf`, see above.

use core::ffi::{c_char, c_int, c_uint, c_void};
use core::future::Future;
use core::pin::Pin;
use core::time::Duration;
use core::{ptr, slice};

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use std::ffi::CString;

use g2g_core::log::{short_type_name, Target};
use g2g_core::memory::{DomainSet, MemoryDomain, MemoryDomainKind, OwnedDmaBuf, SystemSlice};
use g2g_core::{
    g2g_error, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata,
    G2gError, HardwareError, OutputSink, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec, RawVideoFormat,
};

use crate::capsfilter::parse_caps;
use crate::encoder_base::emit_packets;

// `GST_VIDEO_MAX_PLANES`, which the C helper asserts at compile time.
const GST_VIDEO_MAX_PLANES: usize = 4;

// `try_pull` / `try_pull_dmabuf` results, mirroring the C helper's defines.
const PULLED: c_int = 1;
const NOT_READY: c_int = 0;
const END_OF_STREAM: c_int = -1;
const NOT_DMABUF: c_int = -2;

// Mirrors `G2gGstWrapDmaBufSample` in the C helper.
#[repr(C)]
#[derive(Debug)]
struct DmaBufSample {
    sample: *mut c_void,
    fd: c_int,
    pts: u64,
    memory_offset: usize,
    memory_size: usize,
    height: c_uint,
    n_planes: c_uint,
    plane_offsets: [usize; GST_VIDEO_MAX_PLANES],
    plane_strides: [c_int; GST_VIDEO_MAX_PLANES],
}

impl DmaBufSample {
    fn empty() -> Self {
        Self {
            sample: ptr::null_mut(),
            fd: -1,
            pts: 0,
            memory_offset: 0,
            memory_size: 0,
            height: 0,
            n_planes: 0,
            plane_offsets: [0; GST_VIDEO_MAX_PLANES],
            plane_strides: [0; GST_VIDEO_MAX_PLANES],
        }
    }
}

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
    fn g2g_gstwrap_push_dmabuf(
        w: *mut c_void,
        fd: c_int,
        data_offset: usize,
        required_end: usize,
        n_planes: c_uint,
        plane_offsets: *const usize,
        plane_strides: *const c_int,
        pts_ns: u64,
        keep_alive: *mut c_void,
        release: unsafe extern "C" fn(*mut c_void),
    ) -> c_int;
    fn g2g_gstwrap_try_pull(
        w: *mut c_void,
        out_data: *mut *mut u8,
        out_len: *mut usize,
        out_pts: *mut u64,
    ) -> c_int;
    fn g2g_gstwrap_try_pull_dmabuf(w: *mut c_void, out: *mut DmaBufSample) -> c_int;
    fn g2g_gstwrap_sample_unref(sample: *mut c_void);
    #[cfg(test)]
    fn g2g_gstwrap_dmabuf_sample_size() -> usize;
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OutputMemory {
    #[default]
    System,
    DmaBuf,
}

impl OutputMemory {
    fn from_nick(nick: &str) -> Option<Self> {
        match nick {
            "system" => Some(Self::System),
            "dmabuf" => Some(Self::DmaBuf),
            _ => None,
        }
    }

    fn nick(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::DmaBuf => "dmabuf",
        }
    }

    fn domain(self) -> MemoryDomainKind {
        match self {
            Self::System => MemoryDomainKind::System,
            Self::DmaBuf => MemoryDomainKind::DmaBuf,
        }
    }
}

// A hosted element's buffer pool reuses the buffer once the sample is released.
#[derive(Debug)]
struct LentSample {
    dmabuf: OwnedDmaBuf,
    sample: *mut c_void,
}

// SAFETY: `sample` is a GstSample reference this struct solely owns, and
// gst_sample_unref is thread-safe.
unsafe impl Send for LentSample {}

impl Drop for LentSample {
    fn drop(&mut self) {
        // SAFETY: `sample` came from `try_pull_dmabuf` and is released once.
        unsafe { g2g_gstwrap_sample_unref(self.sample) };
    }
}

/// Hosts an unported GStreamer element inside a g2g graph. See the module docs.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::gstwrap::GstWrap;
///
/// // gst-launch: ... ! gstwrap element="x264enc bitrate=4000" ! ...
/// let wrap = GstWrap::new();
/// ```
#[derive(Debug)]
pub struct GstWrap {
    /// GStreamer element description, e.g. `"x264enc bitrate=4000"`.
    element: String,
    /// Caps the hosted element produces (gst-launch syntax), for a reformatting
    /// element. `None` means caps/size-preserving (output caps == input caps).
    output_caps: Option<String>,
    output_memory: OutputMemory,
    /// The running pipeline, `None` until `configure_pipeline`.
    handle: Option<WrapPtr>,
    /// Caps announced downstream (once) before the first output frame: the
    /// declared `output-caps` for a reformatting element, else the input caps.
    announce_caps: Option<Caps>,
    input_raw_video: Option<(RawVideoFormat, u32)>,
    output_raw_format: Option<RawVideoFormat>,
    lent_samples: Vec<LentSample>,
    caps_sent: bool,
    emitted: u64,
    configured: bool,
}

impl GstWrap {
    pub fn new() -> Self {
        Self {
            element: String::new(),
            output_caps: None,
            output_memory: OutputMemory::System,
            handle: None,
            announce_caps: None,
            input_raw_video: None,
            output_raw_format: None,
            lent_samples: Vec::new(),
            caps_sent: false,
            emitted: 0,
            configured: false,
        }
    }

    pub fn with_output_memory(mut self, output_memory: OutputMemory) -> Self {
        self.output_memory = output_memory;
        self
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
    PropertySpec::new(
        "output-memory",
        PropKind::Str,
        "memory output frames leave in: system copies each sample out, dmabuf passes each sample's dma-buf on and fails on any other memory",
    )
    .with_enum_values("system | dmabuf")
    .with_default("system"),
];

// g2g's dma-buf layout: planes back to back, chroma stride derived from luma.
#[derive(Debug, PartialEq, Eq)]
struct PlaneLayout {
    count: usize,
    offsets: [u64; GST_VIDEO_MAX_PLANES],
    strides: [u32; GST_VIDEO_MAX_PLANES],
    span: u64,
}

fn plane_layout(format: RawVideoFormat, stride: u32, height: u32) -> Option<PlaneLayout> {
    let count = format.plane_count();
    if count > GST_VIDEO_MAX_PLANES {
        return None;
    }
    let mut layout = PlaneLayout {
        count,
        offsets: [0; GST_VIDEO_MAX_PLANES],
        strides: [0; GST_VIDEO_MAX_PLANES],
        span: 0,
    };
    for plane in 0..count {
        let plane_stride = match format.chroma_shift() {
            Some((horizontal, _)) if plane > 0 => stride >> horizontal,
            _ => stride,
        };
        let rows = format.plane_rows(plane, height)?;
        layout.offsets[plane] = layout.span;
        layout.strides[plane] = plane_stride;
        let plane_bytes = u64::from(plane_stride).checked_mul(u64::from(rows))?;
        layout.span = layout.span.checked_add(plane_bytes)?;
    }
    Some(layout)
}

// Plane offsets count from the fd start.
#[derive(Debug)]
struct PushPlanes {
    data_offset: usize,
    count: c_uint,
    offsets: [usize; GST_VIDEO_MAX_PLANES],
    strides: [c_int; GST_VIDEO_MAX_PLANES],
    required_end: usize,
}

fn push_planes(
    raw_video: Option<(RawVideoFormat, u32)>,
    dmabuf: &OwnedDmaBuf,
) -> Option<PushPlanes> {
    let offset = u64::from(dmabuf.offset);
    let data_offset = usize::try_from(offset).ok()?;
    let mut planes = PushPlanes {
        data_offset,
        count: 0,
        offsets: [0; GST_VIDEO_MAX_PLANES],
        strides: [0; GST_VIDEO_MAX_PLANES],
        required_end: data_offset,
    };
    let Some((format, height)) = raw_video else {
        return Some(planes);
    };
    let layout = plane_layout(format, dmabuf.stride, height)?;
    planes.count = c_uint::try_from(layout.count).ok()?;
    planes.required_end = usize::try_from(offset.checked_add(layout.span)?).ok()?;
    for plane in 0..layout.count {
        planes.offsets[plane] = usize::try_from(offset.checked_add(layout.offsets[plane])?).ok()?;
        planes.strides[plane] = c_int::try_from(layout.strides[plane]).ok()?;
    }
    Some(planes)
}

// `None` unless GStreamer's layout is g2g's dma-buf layout and fits the memory.
fn sample_layout(raw_format: Option<RawVideoFormat>, sample: &DmaBufSample) -> Option<(u32, u32)> {
    let memory_offset = u64::try_from(sample.memory_offset).ok()?;
    let Some(format) = raw_format else {
        return Some((0, u32::try_from(memory_offset).ok()?));
    };
    let stride = u32::try_from(sample.plane_strides[0])
        .ok()
        .filter(|&stride| stride > 0)?;
    let layout = plane_layout(format, stride, sample.height)?;
    if usize::try_from(sample.n_planes).ok()? != layout.count {
        return None;
    }
    let first_offset = u64::try_from(sample.plane_offsets[0]).ok()?;
    for plane in 0..layout.count {
        let expected_offset = first_offset.checked_add(layout.offsets[plane])?;
        let offset_matches = u64::try_from(sample.plane_offsets[plane]).ok()? == expected_offset;
        let stride_matches =
            u32::try_from(sample.plane_strides[plane]).ok()? == layout.strides[plane];
        if !offset_matches || !stride_matches {
            return None;
        }
    }
    if first_offset.checked_add(layout.span)? > u64::try_from(sample.memory_size).ok()? {
        return None;
    }
    let offset = u32::try_from(memory_offset.checked_add(first_offset)?).ok()?;
    Some((stride, offset))
}

fn raw_video_of(caps: &Caps) -> Option<(RawVideoFormat, u32)> {
    match caps {
        Caps::RawVideo {
            format,
            height: Dim::Fixed(height),
            ..
        } => Some((*format, *height)),
        _ => None,
    }
}

unsafe extern "C" fn release_pushed_dmabuf(keep_alive: *mut c_void) {
    // SAFETY: `keep_alive` is the `Box<OwnedDmaBuf>` `push_dmabuf` leaked, and
    // the C helper calls this exactly once.
    drop(unsafe { Box::from_raw(keep_alive.cast::<OwnedDmaBuf>()) });
}

enum Pulled {
    Frame(MemoryDomain, u64),
    NotReady,
    EndOfStream,
}

fn log_target() -> Target<'static> {
    Target::category(short_type_name::<GstWrap>())
}

/// Try to drain one processed frame as system memory, copying its bytes out.
fn pull_system(p: WrapPtr) -> Result<Pulled, G2gError> {
    let mut data: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    let mut pts: u64 = 0;
    // SAFETY: the out params are valid local addresses; on `PULLED` the helper
    // sets `data` to a malloc'd block of `len` bytes we then own.
    let r = unsafe { g2g_gstwrap_try_pull(p.0, &mut data, &mut len, &mut pts) };
    match r {
        PULLED => {
            // SAFETY: `data` points to `len` initialized bytes allocated by the
            // helper; we copy them out then hand the block back to be freed.
            let v = unsafe { slice::from_raw_parts(data, len) }.to_vec();
            // SAFETY: `data` came from `try_pull` and has not been freed.
            unsafe { g2g_gstwrap_free_buf(data) };
            let domain = MemoryDomain::System(SystemSlice::from_boxed(v.into_boxed_slice()));
            Ok(Pulled::Frame(domain, pts))
        }
        NOT_READY => Ok(Pulled::NotReady),
        END_OF_STREAM => Ok(Pulled::EndOfStream),
        _ => {
            g2g_error!(
                log_target(),
                "cannot map or copy out the hosted element's sample"
            );
            Err(G2gError::Hardware(HardwareError::Other))
        }
    }
}

impl GstWrap {
    fn push_dmabuf(&self, p: WrapPtr, dmabuf: &OwnedDmaBuf, pts_ns: u64) -> Result<(), G2gError> {
        let planes = push_planes(self.input_raw_video, dmabuf).ok_or(G2gError::CapsMismatch)?;
        let keep_alive = Box::into_raw(Box::new(dmabuf.clone())).cast::<c_void>();
        // SAFETY: `p` is valid; the plane arrays hold GST_VIDEO_MAX_PLANES
        // entries; the helper dups the fd and takes `keep_alive`, calling
        // `release_pushed_dmabuf` on it exactly once.
        let r = unsafe {
            g2g_gstwrap_push_dmabuf(
                p.0,
                dmabuf.as_raw(),
                planes.data_offset,
                planes.required_end,
                planes.count,
                planes.offsets.as_ptr(),
                planes.strides.as_ptr(),
                pts_ns,
                keep_alive,
                release_pushed_dmabuf,
            )
        };
        if r != 0 {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        Ok(())
    }

    fn pull_dmabuf(&mut self, p: WrapPtr) -> Result<Pulled, G2gError> {
        self.lent_samples
            .retain(|lent| lent.dmabuf.share_count() > 1);
        let mut sample = DmaBufSample::empty();
        // SAFETY: `p` is valid and `sample` is a writable `DmaBufSample`.
        let r = unsafe { g2g_gstwrap_try_pull_dmabuf(p.0, &mut sample) };
        match r {
            PULLED => {}
            NOT_READY => return Ok(Pulled::NotReady),
            END_OF_STREAM => return Ok(Pulled::EndOfStream),
            NOT_DMABUF => return Err(G2gError::UnsupportedDomain),
            _ => {
                g2g_error!(log_target(), "cannot dup the hosted element's dma-buf fd");
                return Err(G2gError::Hardware(HardwareError::Other));
            }
        }
        // SAFETY: on `PULLED` the helper hands over a fresh dup of the sample's
        // fd that nothing else owns.
        let mut dmabuf = unsafe { OwnedDmaBuf::from_raw(sample.fd, 0, 0) };
        let lent = LentSample {
            dmabuf: dmabuf.clone(),
            sample: sample.sample,
        };
        let (stride, offset) =
            sample_layout(self.output_raw_format, &sample).ok_or(G2gError::CapsMismatch)?;
        dmabuf.stride = stride;
        dmabuf.offset = offset;
        self.lent_samples.push(lent);
        Ok(Pulled::Frame(MemoryDomain::DmaBuf(dmabuf), sample.pts))
    }

    fn pull_one(&mut self, p: WrapPtr) -> Result<Pulled, G2gError> {
        match self.output_memory {
            OutputMemory::System => pull_system(p),
            OutputMemory::DmaBuf => self.pull_dmabuf(p),
        }
    }

    /// Drain every frame the hosted element has ready right now, without waiting.
    /// A latent element (an encoder) may have none yet; that is not an error.
    fn drain_ready(&mut self, p: WrapPtr) -> Result<Vec<(MemoryDomain, u64)>, G2gError> {
        let mut out = Vec::new();
        while let Pulled::Frame(domain, pts) = self.pull_one(p)? {
            out.push((domain, pts));
        }
        Ok(out)
    }

    /// After EOS, drain the hosted element's flushed frames, waiting for its
    /// internal latency. Bounded (~5 s of 1 ms polls) so a stuck element cannot hang
    /// the graph; any frames not produced by then are dropped when the pipeline is
    /// torn down.
    async fn drain_to_eos(&mut self, p: WrapPtr) -> Result<Vec<(MemoryDomain, u64)>, G2gError> {
        let mut out = Vec::new();
        for _ in 0..5000u32 {
            match self.pull_one(p)? {
                Pulled::Frame(domain, pts) => out.push((domain, pts)),
                Pulled::EndOfStream => break, // EOS: the appsink is drained.
                Pulled::NotReady => tokio::time::sleep(Duration::from_millis(1)).await, // flushing.
            }
        }
        Ok(out)
    }
}

impl AsyncElement for GstWrap {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// System only unless `output-memory=dmabuf`, so a default graph keeps its download path.
    fn input_domains(&self) -> DomainSet {
        let system = DomainSet::only(MemoryDomainKind::System);
        match self.output_memory {
            OutputMemory::System => system,
            OutputMemory::DmaBuf => system.with(MemoryDomainKind::DmaBuf),
        }
    }

    fn output_memory(&self) -> MemoryDomainKind {
        self.output_memory.domain()
    }

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
        self.input_raw_video = raw_video_of(absolute_caps);
        self.output_raw_format = match &announce {
            Caps::RawVideo { format, .. } => Some(*format),
            _ => None,
        };
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
            "output-memory" => {
                let nick = value.as_str().ok_or(PropError::Type)?;
                self.output_memory = OutputMemory::from_nick(nick).ok_or(PropError::Value)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "element" => Some(PropValue::Str(self.element.clone())),
            "output-caps" => self.output_caps.clone().map(PropValue::Str),
            "output-memory" => Some(PropValue::Str(self.output_memory.nick().into())),
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
                    let pts_ns = frame.timing.pts_ns;
                    if let MemoryDomain::DmaBuf(dmabuf) = &frame.domain {
                        self.push_dmabuf(p, dmabuf, pts_ns)?;
                    } else {
                        let bytes = frame
                            .domain
                            .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                        // SAFETY: `p` is valid; `bytes` is valid for `bytes.len()`;
                        // `push` copies the bytes into a GstBuffer.
                        let r =
                            unsafe { g2g_gstwrap_push(p.0, bytes.as_ptr(), bytes.len(), pts_ns) };
                        if r != 0 {
                            return Err(G2gError::Hardware(HardwareError::Other));
                        }
                    }
                    let frames = self.drain_ready(p)?;
                    // `announce_caps` is Some after `configure_pipeline`; cloning
                    // releases the borrow so `emit_packets` can take `&mut self`.
                    let caps = self.announce_caps.clone().ok_or(G2gError::NotConfigured)?;
                    emit_packets(&mut self.caps_sent, &mut self.emitted, frames, &caps, out)
                        .await?;
                }
                PipelinePacket::Eos => {
                    // SAFETY: `p` is valid; signals EOS on the appsrc feed.
                    unsafe { g2g_gstwrap_eos(p.0) };
                    let frames = self.drain_to_eos(p).await?;
                    let caps = self.announce_caps.clone().ok_or(G2gError::NotConfigured)?;
                    emit_packets(&mut self.caps_sent, &mut self.emitted, frames, &caps, out)
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

#[cfg(test)]
mod tests {
    use super::*;

    const NV12_STRIDE: u32 = 8;
    const NV12_HEIGHT: u32 = 4;
    const LUMA_BYTES: usize = (NV12_STRIDE * NV12_HEIGHT) as usize;
    const FRAME_BYTES: usize = LUMA_BYTES + (NV12_STRIDE * NV12_HEIGHT / 2) as usize;
    const MEMORY_OFFSET: usize = 16;

    // An NV12 sample laid out the way g2g lays out a single-stride dma-buf.
    fn nv12_sample() -> DmaBufSample {
        let mut sample = DmaBufSample::empty();
        sample.memory_offset = MEMORY_OFFSET;
        sample.memory_size = FRAME_BYTES;
        sample.height = NV12_HEIGHT;
        sample.n_planes = 2;
        sample.plane_offsets[1] = LUMA_BYTES;
        sample.plane_strides[0] = NV12_STRIDE as c_int;
        sample.plane_strides[1] = NV12_STRIDE as c_int;
        sample
    }

    #[test]
    fn dmabuf_sample_matches_the_c_struct() {
        // SAFETY: returns a sizeof, no arguments.
        let c_size = unsafe { g2g_gstwrap_dmabuf_sample_size() };
        assert_eq!(c_size, core::mem::size_of::<DmaBufSample>());
    }

    #[test]
    fn plane_layout_spans_the_dmabuf_frame_bytes() {
        const WIDTH: u32 = 6;
        for format in [
            RawVideoFormat::Nv12,
            RawVideoFormat::I420,
            RawVideoFormat::Rgba8,
        ] {
            let stride = format.row_stride(WIDTH).expect("single-stride format");
            let layout = plane_layout(format, stride, NV12_HEIGHT).expect("layout");
            let frame_bytes = format
                .frame_bytes(u64::from(stride), u64::from(NV12_HEIGHT))
                .expect("frame bytes");
            assert_eq!(layout.span, frame_bytes, "{format:?}");
            assert_eq!(layout.count, format.plane_count(), "{format:?}");
        }
    }

    #[test]
    fn accepts_a_single_stride_sample() {
        let layout = sample_layout(Some(RawVideoFormat::Nv12), &nv12_sample());
        assert_eq!(layout, Some((NV12_STRIDE, MEMORY_OFFSET as u32)));
    }

    #[test]
    fn rejects_a_sample_that_overruns_its_memory() {
        let mut sample = nv12_sample();
        sample.memory_size = FRAME_BYTES - 1;
        assert_eq!(sample_layout(Some(RawVideoFormat::Nv12), &sample), None);
    }

    #[test]
    fn rejects_a_plane_offset_that_overflows() {
        let mut sample = nv12_sample();
        sample.plane_offsets[0] = usize::MAX;
        assert_eq!(sample_layout(Some(RawVideoFormat::Nv12), &sample), None);
    }

    #[test]
    fn rejects_a_negative_stride() {
        let mut sample = nv12_sample();
        sample.plane_strides[0] = -(NV12_STRIDE as c_int);
        assert_eq!(sample_layout(Some(RawVideoFormat::Nv12), &sample), None);
    }

    #[test]
    fn rejects_a_padded_chroma_plane() {
        let mut sample = nv12_sample();
        sample.plane_offsets[1] = LUMA_BYTES + MEMORY_OFFSET;
        sample.memory_size = FRAME_BYTES + MEMORY_OFFSET;
        assert_eq!(sample_layout(Some(RawVideoFormat::Nv12), &sample), None);
    }
}
