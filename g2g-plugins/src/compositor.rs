//! Software video compositor (M93): overlays N raw RGBA8 input streams onto one
//! output canvas at configurable positions, z-order, and per-pad alpha, with
//! alpha blending. The `videomixer` / `compositor` analog (picture-in-picture,
//! multi-camera grids, sub-window UIs). Our `mux` is a fan-in *multiplexer*
//! (interleaving encoded tracks); this is a fan-in *mixer* (combining raw
//! pixels into one frame).
//!
//! CPU, `no_std` baseline like the other raw-video transforms
//! (videoconvert/videoscale/...); `WgpuCompositor` (the `wgpu-sink` feature) is
//! the RGBA8 GPU companion for HD / many-input scale.
//! The output format is chosen at construction ([`Compositor::with_format`]):
//! RGBA8 (default, packed source-over with per-pixel alpha) or 8-bit
//! NV12 / I420 / I422 / I444 (mixed plane-by-plane with the scalar per-pad alpha,
//! no RGBA round-trip). Every input must match that format (put a `VideoConvert`
//! upstream otherwise). Geometry per input is whatever each negotiates; the
//! output canvas size and framerate are fixed at construction. For a subsampled
//! YUV format, overlay positions and sizes are aligned down to even so the
//! chroma planes stay on sample boundaries.
//!
//! **Cadence:** input 0 is the timing driver (the background / main stream).
//! One composited output frame is emitted per input-0 frame, overlaying the
//! latest frame cached from every other input. Each overlay updates
//! independently as new frames land, so a live overlay animates at its own rate.
//! To emit at a different constant output rate, put a `VideoRate` downstream
//! (`compositor ! videorate`); the compositor stamps each output frame with
//! input 0's PTS, so videorate resamples the cadence without any compositor-side
//! frame-rate conversion. With
//! [`with_timed_output`](Compositor::with_timed_output) (M875) output no longer
//! stops when input 0 does: on a runner deadline tick the last frame it delivered
//! is re-composited with the current overlays (zero-order-hold), one per empty
//! frame period, so a live overlay keeps animating over a frozen background.
//!
//! **Startup:** inputs start asynchronously and an overlay branch (camera warm-up,
//! extra transforms) can lag the background, in the extreme starting only after a
//! short background has fully drained. So at startup the compositor buffers
//! input-0 frames (bounded by [`PENDING_CAP`]) until every overlay has delivered
//! its first frame, then flushes them composited with the overlays and runs live.
//! Two failure modes are avoided: it must not block the background forever on a
//! slow overlay (so on buffer overflow the oldest input-0 frame is emitted
//! *overlay-less* rather than held or dropped, keeping output flowing and losing
//! no frames), and once primed it must not keep reusing a single stale overlay
//! frame (so live frames composite the latest overlay, not a frozen one).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::paint::blend_px;
use crate::pixel::{frame_byte_size, planar_planes};
use crate::yuvmatrix::YuvRgbMatrix;
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, Colorimetry, ConfigureOutcome, Dim, ElementMetadata,
    FrameTiming, G2gError, InputAggregator, MemoryDomain, MultiInputElement, OutputSink,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat, Segment,
};

/// Placement of one input stream on the output canvas.
#[derive(Debug, Clone, Copy)]
pub struct CompositorPad {
    /// Left edge on the canvas, in pixels. May be negative (clipped at the left).
    pub xpos: i32,
    /// Top edge on the canvas, in pixels. May be negative (clipped at the top).
    pub ypos: i32,
    /// Paint order: lower z-order is painted first (further back). Ties break by
    /// input index, so input 0 is the backmost among equal z-orders.
    pub zorder: u32,
    /// Per-pad alpha 0..=255, multiplied with each pixel's source alpha. 255 is
    /// fully opaque (modulo the source's own alpha channel).
    pub alpha: u8,
    /// On-canvas size `(width, height)` to scale this input to as it composites.
    /// `None` draws the input at its native geometry; `Some` resamples it
    /// (bilinear), so a downscaled camera needs no upstream `VideoScale`.
    pub size: Option<(u32, u32)>,
}

impl CompositorPad {
    /// An opaque pad at `(xpos, ypos)`, z-order 0, drawn at native size.
    pub fn at(xpos: i32, ypos: i32) -> Self {
        Self {
            xpos,
            ypos,
            zorder: 0,
            alpha: 255,
            size: None,
        }
    }

    /// Set the paint order (lower is painted first / further back).
    pub fn with_zorder(mut self, zorder: u32) -> Self {
        self.zorder = zorder;
        self
    }

    /// Set the per-pad alpha (0 transparent, 255 opaque).
    pub fn with_alpha(mut self, alpha: u8) -> Self {
        self.alpha = alpha;
        self
    }

    /// Scale this input to `width` x `height` on the canvas (bilinear), instead
    /// of compositing it at its native geometry.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.size = Some((width, height));
        self
    }

    /// On-canvas size for an input whose native geometry is `sw` x `sh`: the
    /// requested size, with an unset (zero) dimension falling back to native, the
    /// way a `compositor` pad's `width` / `height` of 0 does.
    pub(crate) fn dest_size(&self, sw: u32, sh: u32) -> (u32, u32) {
        match self.size {
            None => (sw, sh),
            Some((w, h)) => (
                match w {
                    0 => sw,
                    w => w,
                },
                match h {
                    0 => sh,
                    h => h,
                },
            ),
        }
    }
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::compositor::{Compositor, CompositorPad};
///
/// let pads = vec![
///     CompositorPad::at(0, 0),
///     CompositorPad::at(960, 540).with_size(320, 180).with_zorder(1),
/// ];
/// let compositor = Compositor::new(1280, 720, pads).with_framerate(30);
/// ```
#[derive(Debug)]
pub struct Compositor {
    out_w: u32,
    out_h: u32,
    /// Output (and required input) pixel format. RGBA8 blends packed with a
    /// per-pixel source alpha; the YUV formats (NV12 / I420 / I422 / I444, 8-bit)
    /// blend per-plane with the scalar per-pad alpha and no RGBA round-trip.
    format: RawVideoFormat,
    framerate_q16: u32,
    /// Per-input placement; `pads.len()` is the input count.
    pads: Vec<CompositorPad>,
    /// Input geometry, cached frames and emit cadence, shared with the GPU
    /// sibling. The CPU element caches each frame's bytes.
    state: CompositorState<Box<[u8]>>,
    /// The canvas fill behind all inputs (RGBA8), default opaque black.
    background: [u8; 4],
    /// Colorimetry input 0 negotiated, which is the space the inputs are mixed
    /// in and so the space the output is in. Drives the YUV background fill and
    /// the output caps.
    colorimetry: Colorimetry,
    /// The output caps last announced downstream, so a colorimetry that firms up
    /// after negotiation is announced once rather than per frame.
    announced_output: Option<Caps>,
    /// The last `Segment` forwarded downstream. Output frames carry input 0's
    /// PTS, so input 0's segment is the one that maps them to running time and a
    /// paced sink needs it: without it the sink paces against the raw PTS base (a
    /// DVD title starting at 2267 s stalls for 37 minutes at zero CPU). A stream
    /// gets more than one: the runner opens every link with a default segment and
    /// the demuxer's real one follows, so a later segment supersedes an earlier
    /// one and must go out. Kept to suppress re-emitting an unchanged one.
    last_segment: Option<Segment>,
}

/// Max input-0 frames buffered during startup before output begins flowing
/// overlay-less (bounds startup memory and latency). Shared with the GPU
/// sibling so both compositors buffer the same startup depth.
pub(crate) const PENDING_CAP: usize = 8;

/// The output pixel format, CPU element only (the GPU one is RGBA8).
const FORMAT_PROP: PropertySpec = PropertySpec::new(
    "format",
    PropKind::Str,
    "output (and required input) pixel format",
)
.with_enum_values("rgba | RGBA | rgba8 | nv12 | NV12 | i420 | I420 | i422 | I422 | i444 | I444")
.with_default("rgba");

/// Both compositors' property table: the canvas knobs, the per-pad placement
/// flattened to `sinkN-*` for the pad indices given, and whatever the element
/// adds. One macro, so the two tables cannot drift apart.
///
/// The pad indices are spelled out because
/// [`properties`](MultiInputElement::properties) is a `&'static` table: gst's
/// request-pad properties (`sink_1::xpos`) have no analog in this launch syntax.
/// Both tables cover pads 0..=7; a graph with more pads places the rest through
/// [`CompositorPad`] at construction.
macro_rules! compositor_props {
    ([$($i:literal)*] $(, $extra:expr)*) => {
        &[
            PropertySpec::new("width", PropKind::Uint, "output canvas width in pixels")
                .with_default("320"),
            PropertySpec::new("height", PropKind::Uint, "output canvas height in pixels")
                .with_default("240"),
            PropertySpec::new(
                "framerate",
                PropKind::Fraction,
                "nominal output framerate, as labelled on the output caps",
            )
            .with_default("30/1"),
            // gst's `compositor` has no element-level geometry (its output caps
            // are negotiated) and spells its fill as a `background` enum
            // (checker / black / white / transparent), so these are g2g names.
            // The packing matches `textoverlay color`.
            PropertySpec::new(
                "background-color",
                PropKind::Uint,
                "canvas fill behind every input, 0xAARRGGBB (4278190080 = opaque black)",
            )
            .with_default("4278190080"),
            PropertySpec::new(
                "timed-output",
                PropKind::Bool,
                "keep emitting at the output framerate while input 0 stalls, holding its last frame",
            )
            .with_default("false"),
            $($extra,)*
            $(
                PropertySpec::new(
                    concat!("sink", $i, "-xpos"),
                    PropKind::Int,
                    concat!("pad ", $i, ": left edge on the canvas in pixels, may be negative"),
                )
                .with_default("0"),
                PropertySpec::new(
                    concat!("sink", $i, "-ypos"),
                    PropKind::Int,
                    concat!("pad ", $i, ": top edge on the canvas in pixels, may be negative"),
                )
                .with_default("0"),
                PropertySpec::new(
                    concat!("sink", $i, "-zorder"),
                    PropKind::Uint,
                    concat!("pad ", $i, ": paint order, lower is painted first"),
                )
                .with_default("0"),
                // gst's compositor pad alpha is a 0.0-1.0 double; this is the
                // 0..=255 byte the blend actually multiplies by.
                PropertySpec::new(
                    concat!("sink", $i, "-alpha"),
                    PropKind::Uint,
                    concat!("pad ", $i, ": per-pad alpha, 0 transparent to 255 opaque"),
                )
                .with_default("255")
                .with_range("0", "255"),
                PropertySpec::new(
                    concat!("sink", $i, "-width"),
                    PropKind::Uint,
                    concat!("pad ", $i, ": on-canvas width to scale to, 0 for native"),
                )
                .with_default("0"),
                PropertySpec::new(
                    concat!("sink", $i, "-height"),
                    PropKind::Uint,
                    concat!("pad ", $i, ": on-canvas height to scale to, 0 for native"),
                )
                .with_default("0"),
            )*
        ]
    };
}

static COMPOSITOR_PROPS: &[PropertySpec] = compositor_props!([0 1 2 3 4 5 6 7], FORMAT_PROP);

/// The same table without the CPU-only `format` (this element is RGBA8).
#[cfg(feature = "wgpu-sink")]
pub(crate) static WGPU_COMPOSITOR_PROPS: &[PropertySpec] = compositor_props!([0 1 2 3 4 5 6 7]);

/// Split a `sinkN-<knob>` property name into the pad index and the knob.
fn split_pad_name(name: &str) -> Option<(usize, &str)> {
    let (index, knob) = name.strip_prefix("sink")?.split_once('-')?;
    Some((index.parse().ok()?, knob))
}

/// Apply a flattened per-pad property. `None` when `name` is not one; `Err` when
/// it names a pad this element does not have (silently ignoring it would leave a
/// launch line thinking its placement applied) or the value is out of range.
pub(crate) fn set_pad_property(
    pads: &mut [CompositorPad],
    name: &str,
    value: &PropValue,
) -> Option<Result<(), PropError>> {
    let (index, knob) = split_pad_name(name)?;
    Some(set_pad_knob(pads, index, knob, value))
}

fn set_pad_knob(
    pads: &mut [CompositorPad],
    index: usize,
    knob: &str,
    value: &PropValue,
) -> Result<(), PropError> {
    let pad = pads.get_mut(index).ok_or(PropError::Value)?;
    let as_pos =
        || i32::try_from(value.as_int().ok_or(PropError::Type)?).map_err(|_| PropError::Value);
    let as_dim =
        || u32::try_from(value.as_uint().ok_or(PropError::Type)?).map_err(|_| PropError::Value);
    match knob {
        "xpos" => pad.xpos = as_pos()?,
        "ypos" => pad.ypos = as_pos()?,
        "zorder" => pad.zorder = as_dim()?,
        "alpha" => {
            pad.alpha = u8::try_from(value.as_uint().ok_or(PropError::Type)?)
                .map_err(|_| PropError::Value)?
        }
        // A zero dimension is "native", so the pair carries whichever of the two
        // the line set (see `CompositorPad::dest_size`).
        "width" | "height" => {
            let (w, h) = pad.size.unwrap_or((0, 0));
            let v = as_dim()?;
            pad.size = Some(match knob {
                "width" => (v, h),
                _ => (w, v),
            });
        }
        _ => return Err(PropError::Unknown),
    }
    Ok(())
}

/// Read back a flattened per-pad property, `None` if `name` is not one (or names
/// a pad this element does not have).
pub(crate) fn pad_property(pads: &[CompositorPad], name: &str) -> Option<PropValue> {
    let (index, knob) = split_pad_name(name)?;
    let pad = pads.get(index)?;
    let (w, h) = pad.size.unwrap_or((0, 0));
    Some(match knob {
        "xpos" => PropValue::Int(pad.xpos as i64),
        "ypos" => PropValue::Int(pad.ypos as i64),
        "zorder" => PropValue::Uint(pad.zorder as u64),
        "alpha" => PropValue::Uint(pad.alpha as u64),
        "width" => PropValue::Uint(w as u64),
        "height" => PropValue::Uint(h as u64),
        _ => return None,
    })
}

/// A canvas dimension property value.
pub(crate) fn dim_property(value: &PropValue) -> Result<u32, PropError> {
    u32::try_from(value.as_uint().ok_or(PropError::Type)?).map_err(|_| PropError::Value)
}

/// A `framerate` property (`fps` or `num/den`) as the Q16 fps both elements store.
pub(crate) fn framerate_property(value: &PropValue) -> Result<u32, PropError> {
    let (num, den) = value.as_fraction().ok_or(PropError::Type)?;
    if num <= 0 || den <= 0 {
        return Err(PropError::Value);
    }
    u32::try_from(num as u64 * 65536 / den as u64).map_err(|_| PropError::Value)
}

/// A packed `0xAARRGGBB` colour property as the `[R, G, B, A]` the blend takes.
pub(crate) fn color_property(value: &PropValue) -> Result<[u8; 4], PropError> {
    let argb =
        u32::try_from(value.as_uint().ok_or(PropError::Type)?).map_err(|_| PropError::Value)?;
    Ok([
        (argb >> 16) as u8,
        (argb >> 8) as u8,
        argb as u8,
        (argb >> 24) as u8,
    ])
}

/// The inverse of [`color_property`], for `get_property`.
pub(crate) fn color_value(rgba: [u8; 4]) -> PropValue {
    let [r, g, b, a] = rgba;
    PropValue::Uint(((a as u64) << 24) | ((r as u64) << 16) | ((g as u64) << 8) | b as u64)
}

/// One output frame period in nanoseconds from a nominal framerate (Q16). Zero
/// for a zero framerate (nothing to pace to).
pub(crate) fn frame_period_ns(framerate_q16: u32) -> u64 {
    match framerate_q16 {
        0 => 0,
        fps => 1_000_000_000u64 * 65536 / fps as u64,
    }
}

/// Paint order: z-order ascending, ties by input index (input 0 backmost).
/// Shared with the wgpu compositor, which uploads its pads in this order.
pub(crate) fn paint_order(pads: &[CompositorPad]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..pads.len()).collect();
    order.sort_by_key(|&i| (pads[i].zorder, i));
    order
}

/// Cap on one overlay input's queued canvases. The overlay branch is unpaced
/// (a subtitle track decodes far faster than the video it annotates), so it
/// races ahead of input 0 and its canvases must be held until they are due.
/// Backpressure alone cannot bound that: the element accepts every packet the
/// runner delivers, so the link never fills and never pushes back. The queue is
/// therefore bounded here. A subtitle stream delivers two canvases per cue (the
/// picture and its clear), so 64 is ~32 cues of lookahead, far beyond any real
/// interleave skew; past it the oldest is dropped, which loses a cue that was
/// already overtaken rather than growing without limit.
const OVERLAY_PENDING_CAP: usize = 64;

/// One overlay input's canvases: those not yet due, and the one in force.
///
/// Overlay inputs advance by *timestamp*, not arrival. An overlay canvas applies
/// from its own pts until a successor becomes due (zero-order hold), which is
/// what makes a cue land on the video frames it belongs to. Taking the
/// latest-arrived instead only looks right when nothing paces: live, the display
/// sink paces input 0 to real time while the subtitle branch runs flat out, so
/// every cue and its clear arrive within the first moments and the last one to
/// land (a clear) is what every visible frame composites with. That is the
/// no-subtitles-on-screen bug.
/// The latest-wins input bookkeeping and emit cadence both compositors run:
/// per-input geometry, the cached frames, startup priming and the output
/// sequence counter. Only the pixel work differs between the CPU and the GPU
/// element. Generic over the cached payload `P`: the CPU element caches the
/// frame's bytes, the GPU one caches either bytes or a texture it binds in
/// place.
#[derive(Debug)]
struct OverlaySlot<P> {
    /// Canvases whose pts is still ahead of the frame being composited, oldest
    /// first.
    pending: alloc::collections::VecDeque<(FrameTiming, P)>,
    /// The canvas in force, held until a successor comes due.
    current: Option<(FrameTiming, P)>,
}

impl<P> Default for OverlaySlot<P> {
    fn default() -> Self {
        Self {
            pending: alloc::collections::VecDeque::new(),
            current: None,
        }
    }
}

impl<P> OverlaySlot<P> {
    /// Whether this overlay has delivered anything yet (priming asks this).
    fn ready(&self) -> bool {
        self.current.is_some() || !self.pending.is_empty()
    }

    /// Promote every queued canvas due at or before `pts`, newest wins.
    fn advance_to(&mut self, pts: u64) {
        while self.pending.front().is_some_and(|(t, _)| t.pts_ns <= pts) {
            self.current = self.pending.pop_front();
        }
    }

    fn push(&mut self, timing: FrameTiming, payload: P) {
        if self.pending.len() >= OVERLAY_PENDING_CAP {
            self.pending.pop_front();
        }
        self.pending.push_back((timing, payload));
    }

    fn clear(&mut self) {
        self.pending.clear();
        self.current = None;
    }
}

#[derive(Debug)]
pub(crate) struct CompositorState<P> {
    /// Per-input frames under the aggregator's latest-wins policy: input 0 (the
    /// timing driver) queues, and one output frame is released per queued item;
    /// every other input holds only its newest frame, read in place by the
    /// compositing pass until a newer one lands. Input 0's queue is empty once
    /// primed, and is the startup buffer until then (bounded to
    /// [`PENDING_CAP`]: on overflow the oldest is emitted overlay-less, so
    /// output keeps flowing and no frame is dropped).
    agg: InputAggregator<(FrameTiming, P)>,
    /// Per-input configured geometry `(width, height)`, set at negotiation.
    inputs: Vec<Option<(u32, u32)>>,
    /// Overlay inputs (1..), each advancing by timestamp against the input-0
    /// frame being composited. Index `i` is input `i + 1`.
    overlays: Vec<OverlaySlot<P>>,
    /// True once every overlay input has delivered at least one frame (or there
    /// are no overlays). Until then the compositor is in startup, buffering
    /// input-0 frames so a late-starting overlay still appears. Latches: an
    /// overlay whose cached frame is later invalidated does not re-open startup.
    primed: bool,
    /// Zero-order-hold output is on: retain each emitted input-0 frame so a
    /// deadline tick can re-composite it while that input stalls. Off by default,
    /// and then nothing is retained at all.
    hold: bool,
    /// The last input-0 frame emitted, with the timestamp it went out on. Dropped
    /// whenever input 0's pixels stop being valid to composite.
    held: Option<(FrameTiming, P)>,
    /// A real (input-driven) frame went out since the last tick, so the next tick
    /// holds off rather than duplicating it.
    emitted_since_tick: bool,
    emitted: u64,
}

impl<P> CompositorState<P> {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            agg: InputAggregator::new(n),
            inputs: vec![None; n],
            overlays: (1..n).map(|_| OverlaySlot::default()).collect(),
            // No overlays (single input) means nothing to wait for: start live.
            primed: n == 1,
            hold: false,
            held: None,
            emitted_since_tick: false,
            emitted: 0,
        }
    }

    /// Retain emitted input-0 frames so a deadline tick can re-composite the last
    /// one (zero-order-hold). Turning it off drops what is retained.
    pub(crate) fn set_hold(&mut self, on: bool) {
        self.hold = on;
        if !on {
            self.held = None;
        }
    }

    /// Whether zero-order-hold output was enabled.
    pub(crate) fn hold_enabled(&self) -> bool {
        self.hold
    }

    /// Take a delivered frame: input 0 queues (each one releases an output),
    /// every other input caches it as its latest. Completes priming once every
    /// overlay has delivered.
    pub(crate) fn ingest(&mut self, input: usize, timing: FrameTiming, payload: P) {
        if input == 0 {
            self.agg.push(0, (timing, payload));
        } else {
            // Overlay: queue by timestamp; it comes into force when an input-0
            // frame at or past its pts is composited.
            self.overlays[input - 1].push(timing, payload);
        }
        if !self.primed && self.overlays.iter().all(OverlaySlot::ready) {
            self.primed = true;
        }
    }

    /// The next input-0 frame to composite, or `None` while output waits. Once
    /// primed that is every queued frame (at the moment of priming this flushes
    /// the startup buffer, in arrival order); during startup it is the oldest
    /// frame once the buffer is over [`PENDING_CAP`], composited overlay-less
    /// rather than dropped so output keeps flowing behind a slow overlay. Call
    /// in a loop: while unprimed one take brings the buffer back to the cap.
    pub(crate) fn take_due(&mut self) -> Option<(FrameTiming, P)> {
        let due = if self.primed || self.agg.queued(0) > PENDING_CAP {
            self.agg.take_round_latest(0)
        } else {
            None
        };
        // Bring the overlays up to this frame's timestamp before it composites,
        // so `latest` reads the canvas in force at that instant.
        if let Some((timing, _)) = &due {
            self.advance_overlays(timing.pts_ns);
        }
        due
    }

    /// Promote each overlay's canvases that are due at `pts`. `take_due` calls
    /// this for the frame it releases; a caller that composites off that path
    /// (a test driving `compose` directly) calls it itself.
    pub(crate) fn advance_overlays(&mut self, pts: u64) {
        for slot in self.overlays.iter_mut() {
            slot.advance_to(pts);
        }
    }

    /// The canvas in force for `input` at the frame being composited: the newest
    /// one whose pts is at or before it, held until a successor comes due. Input
    /// 0 has no such slot (it is the timing driver, taken by `take_due`).
    pub(crate) fn latest(&self, input: usize) -> Option<&(FrameTiming, P)> {
        match input {
            0 => self.agg.latest(0),
            i => self.overlays[i - 1].current.as_ref(),
        }
    }

    /// Retain the input-0 frame just emitted, so a tick can re-composite it, and
    /// note that real output went out (the next tick holds off). With
    /// zero-order-hold off the payload is dropped here, as it always was.
    pub(crate) fn record_emitted(&mut self, timing: FrameTiming, payload: P) {
        self.retain(timing, payload);
        self.emitted_since_tick = true;
    }

    /// Put back the frame a tick re-emitted, at the timestamp it went out on, so
    /// consecutive ticks keep walking forward.
    pub(crate) fn record_held(&mut self, timing: FrameTiming, payload: P) {
        self.retain(timing, payload);
    }

    fn retain(&mut self, timing: FrameTiming, payload: P) {
        if self.hold {
            self.held = Some((timing, payload));
        }
    }

    /// The zero-order-hold frame a deadline tick should emit: the retained
    /// input-0 frame, its clock advanced one `period_ns`. `None` when nothing is
    /// due, which is the whole decision: zero-order-hold off, nothing retained
    /// yet, or real output already went out since the last tick (a tick only says
    /// the period elapsed, so it may fire spuriously). The caller re-composites it
    /// with the overlays as they now stand and hands it back through
    /// [`record_held`](Self::record_held).
    pub(crate) fn take_tick_due(&mut self, period_ns: u64) -> Option<(FrameTiming, P)> {
        let emitted = core::mem::take(&mut self.emitted_since_tick);
        if !self.hold || emitted {
            return None;
        }
        let (mut timing, payload) = self.held.take()?;
        timing.pts_ns = timing.pts_ns.saturating_add(period_ns);
        timing.dts_ns = timing.dts_ns.saturating_add(period_ns);
        // A held frame re-composites at its advanced timestamp, so the overlays
        // move with it: a cue can appear or clear while input 0 stalls.
        self.advance_overlays(timing.pts_ns);
        Some((timing, payload))
    }

    pub(crate) fn geometry(&self, input: usize) -> Option<(u32, u32)> {
        self.inputs[input]
    }

    /// Record `input`'s negotiated geometry, reporting whether it changed. Keeps
    /// the frames a change invalidates: the caller decides what else goes with
    /// them (upload flags, device buffers) and calls [`clear`](Self::clear).
    pub(crate) fn set_geometry(&mut self, input: usize, w: u32, h: u32) -> bool {
        let changed = self.inputs[input] != Some((w, h));
        self.inputs[input] = Some((w, h));
        changed
    }

    /// Drop `input`'s cached / buffered frames.
    pub(crate) fn clear(&mut self, input: usize) {
        self.agg.clear(input);
        if input == 0 {
            self.drop_held();
        } else {
            self.overlays[input - 1].clear();
        }
    }

    /// Forget the retained frame: input 0's pixels are no longer valid to
    /// composite (a new geometry would read them at the wrong dimensions).
    pub(crate) fn drop_held(&mut self) {
        self.held = None;
    }

    /// A flush drops `input`'s frames, and on an overlay re-arms startup so that
    /// overlay is waited for again instead of missing from the next output.
    pub(crate) fn flush(&mut self, input: usize) {
        self.clear(input);
        if input != 0 {
            self.primed = false;
        }
    }

    /// Composited frames emitted so far.
    pub(crate) fn emitted(&self) -> u64 {
        self.emitted
    }

    /// The sequence number for the frame being emitted, advancing the counter.
    pub(crate) fn next_sequence(&mut self) -> u64 {
        let seq = self.emitted;
        self.emitted += 1;
        seq
    }
}

impl Compositor {
    /// A compositor producing an `out_w` x `out_h` RGBA8 canvas at 30 fps, with
    /// one `CompositorPad` per input (input 0 is the timing driver). Panics if
    /// `pads` is empty.
    pub fn new(out_w: u32, out_h: u32, pads: Vec<CompositorPad>) -> Self {
        assert!(!pads.is_empty(), "Compositor needs at least one input");
        let n = pads.len();
        Self {
            out_w,
            out_h,
            format: RawVideoFormat::Rgba8,
            framerate_q16: 30 << 16,
            pads,
            state: CompositorState::new(n),
            background: [0, 0, 0, 255],
            colorimetry: Colorimetry::UNKNOWN,
            announced_output: None,
            last_segment: None,
        }
    }

    /// Set the output framerate in nominal fps (stored Q16). This labels the
    /// output caps; the emit cadence follows input 0's frames. To resample the
    /// output to a different constant rate, put a `VideoRate` downstream
    /// (`compositor ! videorate`), which repeats / drops to the target rate off
    /// the per-frame PTS the compositor stamps.
    pub fn with_framerate(mut self, fps: u32) -> Self {
        self.framerate_q16 = fps << 16;
        self
    }

    /// Set the output (and required input) pixel format. Every input must arrive
    /// in this format (put a `VideoConvert` upstream otherwise), so a YUV mix
    /// composites planar without an RGBA round-trip. Supports RGBA8 (the default)
    /// and 8-bit NV12 / I420 / I422 / I444; panics on any other format.
    pub fn with_format(mut self, format: RawVideoFormat) -> Self {
        let ok = matches!(format, RawVideoFormat::Rgba8)
            || (format.bytes_per_sample() == 1
                && (matches!(format, RawVideoFormat::Nv12) || format.is_planar_yuv()));
        assert!(
            ok,
            "Compositor supports RGBA8 and 8-bit NV12/I420/I422/I444, got {format:?}"
        );
        self.format = format;
        self
    }

    /// Set the RGBA8 background the inputs composite over (default opaque black).
    /// Shows wherever no input covers the canvas.
    pub fn with_background(mut self, rgba: [u8; 4]) -> Self {
        self.background = rgba;
        self
    }

    /// Keep emitting at the output framerate while input 0 stalls: the last
    /// composited input-0 frame is held and re-emitted once per frame period
    /// (zero-order-hold), each time re-composited with the overlays as they stand,
    /// so a live overlay keeps animating over a frozen background. Off by default.
    ///
    /// Needs a pipeline clock that can sleep on a deadline (any
    /// [`AsyncClock`](g2g_core::AsyncClock), which is what the runner turns into the
    /// arm's timer): the element declares the period through
    /// [`tick_interval_ns`](MultiInputElement::tick_interval_ns) and emits on the
    /// [`PipelinePacket::Tick`] it gets back. Against a clock that only tells time
    /// it behaves exactly as it does without this.
    pub fn with_timed_output(mut self) -> Self {
        self.state.set_hold(true);
        self
    }

    /// Number of composited frames emitted so far (one per input-0 frame).
    pub fn emitted(&self) -> u64 {
        self.state.emitted()
    }

    fn output(&self) -> Caps {
        Caps::RawVideo {
            format: self.format,
            width: Dim::Fixed(self.out_w),
            height: Dim::Fixed(self.out_h),
            framerate: Rate::Fixed(self.framerate_q16),
            interlace: g2g_core::Interlace::Any,
            colorimetry: self.colorimetry,
        }
    }

    /// Announce the output caps when the colorimetry input 0 settled on refines
    /// what negotiation fixed. Negotiation runs before any input is configured,
    /// so downstream only ever saw the unknown colorimetry this element starts
    /// with; an untagged input leaves it there and nothing is sent.
    async fn announce_output(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if self.colorimetry == Colorimetry::UNKNOWN {
            return Ok(());
        }
        let caps = self.output();
        if self.announced_output.as_ref() == Some(&caps) {
            return Ok(());
        }
        out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
        self.announced_output = Some(caps);
        Ok(())
    }

    /// The format the compositor accepts on every input, at any geometry.
    fn accepted(&self) -> Caps {
        Caps::RawVideo {
            format: self.format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }
    }

    /// Composite onto a fresh background-filled canvas in z-order. Input 0 uses
    /// `base0` (the frame currently driving output); every other input uses its
    /// latest cached frame. Dispatches to the packed (RGBA) or planar (YUV) path
    /// by the output format.
    fn compose(&self, base0: &[u8]) -> Box<[u8]> {
        match self.format {
            RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8 => self.compose_packed(base0),
            _ => self.compose_planar(base0),
        }
    }

    /// Packed RGBA compositing: source-over with per-pixel source alpha.
    fn compose_packed(&self, base0: &[u8]) -> Box<[u8]> {
        let (cw, ch) = (self.out_w as usize, self.out_h as usize);
        let mut canvas = vec![0u8; cw * ch * 4];
        for px in canvas.as_chunks_mut::<4>().0 {
            px.copy_from_slice(&self.background);
        }
        for i in paint_order(&self.pads) {
            let Some((w, h)) = self.state.geometry(i) else {
                continue;
            };
            let src: &[u8] = if i == 0 {
                base0
            } else {
                match self.state.latest(i) {
                    Some((_, s)) => s,
                    None => continue,
                }
            };
            let pad = self.pads[i];
            let (sw, sh) = (w as usize, h as usize);
            let (dw, dh) = pad.dest_size(w, h);
            let (dw, dh) = (dw as usize, dh as usize);
            if (dw, dh) == (sw, sh) {
                blend_over(
                    &mut canvas,
                    cw,
                    ch,
                    src,
                    sw,
                    sh,
                    pad.xpos,
                    pad.ypos,
                    pad.alpha,
                );
            } else {
                blend_over_scaled(
                    &mut canvas,
                    cw,
                    ch,
                    src,
                    sw,
                    sh,
                    pad.xpos,
                    pad.ypos,
                    dw,
                    dh,
                    pad.alpha,
                );
            }
        }
        canvas.into_boxed_slice()
    }

    /// Planar / semi-planar YUV compositing: each plane (Y, then the two chroma
    /// planes at the format's subsampling) is blended independently with the
    /// scalar per-pad alpha, so a YUV mix needs no RGBA round-trip. Overlay
    /// positions and sizes are aligned to even for the subsampled chroma.
    fn compose_planar(&self, base0: &[u8]) -> Box<[u8]> {
        let (w, h) = (self.out_w as usize, self.out_h as usize);
        let mut canvas = vec![0u8; frame_byte_size(self.format, self.out_w, self.out_h)];

        let dst_chans = channels(self.format, w, h);
        let bg = rgba_to_yuv(self.background, &YuvRgbMatrix::new(self.colorimetry));
        for (chan, &val) in dst_chans.iter().zip(bg.iter()) {
            fill_channel(&mut canvas, chan, val);
        }

        for i in paint_order(&self.pads) {
            let Some((sw, sh)) = self.state.geometry(i) else {
                continue;
            };
            let src: &[u8] = if i == 0 {
                base0
            } else {
                match self.state.latest(i) {
                    Some((_, s)) => s,
                    None => continue,
                }
            };
            let pad = self.pads[i];
            let src_chans = channels(self.format, sw as usize, sh as usize);
            // even-align so a subsampled plane's placement stays on a chroma
            // sample (harmless for the full-res luma plane).
            let (x0, y0) = (pad.xpos & !1, pad.ypos & !1);
            let scaled = pad.size.map(|_| {
                let (dw, dh) = pad.dest_size(sw, sh);
                ((dw & !1) as usize, (dh & !1) as usize)
            });
            for (dc, sc) in dst_chans.iter().zip(src_chans.iter()) {
                let (cx, cy) = (x0 >> dc.hs, y0 >> dc.vs);
                match scaled {
                    None => blend_channel(&mut canvas, dc, src, sc, cx, cy, pad.alpha),
                    Some((dw, dh)) => blend_channel_scaled(
                        &mut canvas,
                        dc,
                        src,
                        sc,
                        cx,
                        cy,
                        dw >> dc.hs,
                        dh >> dc.vs,
                        pad.alpha,
                    ),
                }
            }
        }
        canvas.into_boxed_slice()
    }

    /// Wrap composited `canvas` bytes as the next output frame, advancing the
    /// output sequence counter.
    fn output_frame(&mut self, canvas: Box<[u8]>, timing: FrameTiming) -> Frame {
        Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(canvas)),
            timing,
            sequence: self.state.next_sequence(),
            meta: Default::default(),
        }
    }
}

/// Alpha-blend a `sw` x `sh` RGBA8 source onto a `cw` x `ch` RGBA8 canvas with
/// its top-left at `(x0, y0)` (may be negative), modulating source alpha by
/// `galpha`. Straight "source-over" compositing, integer math; pixels outside
/// the canvas are clipped. The arguments are the canvas + source geometry and
/// the placement: a flat parameter list keeps this inner loop allocation-free.
#[allow(clippy::too_many_arguments)]
fn blend_over(
    canvas: &mut [u8],
    cw: usize,
    ch: usize,
    src: &[u8],
    sw: usize,
    sh: usize,
    x0: i32,
    y0: i32,
    galpha: u8,
) {
    for sy in 0..sh {
        let dy = y0 + sy as i32;
        if dy < 0 || dy as usize >= ch {
            continue;
        }
        for sx in 0..sw {
            let dx = x0 + sx as i32;
            if dx < 0 || dx as usize >= cw {
                continue;
            }
            let s = (sy * sw + sx) * 4;
            let d = (dy as usize * cw + dx as usize) * 4;
            let px = [src[s], src[s + 1], src[s + 2], src[s + 3]];
            blend_px(canvas, d, px, galpha);
        }
    }
}

/// Alpha-blend a `sw` x `sh` RGBA8 source onto the canvas, resampled (bilinear)
/// to a `dw` x `dh` rectangle with its top-left at `(x0, y0)`. Same source-over
/// math as [`blend_over`], with integer fixed-point sampling (no float intrinsics
/// for the `no_std` baseline). Pixels outside the canvas are clipped.
#[allow(clippy::too_many_arguments)]
pub(crate) fn blend_over_scaled(
    canvas: &mut [u8],
    cw: usize,
    ch: usize,
    src: &[u8],
    sw: usize,
    sh: usize,
    x0: i32,
    y0: i32,
    dw: usize,
    dh: usize,
    galpha: u8,
) {
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
        return;
    }
    // Center-aligned source coordinate for a destination index, in Q16 fixed
    // point: ((d + 0.5) * s / dst - 0.5). Clamped into the source extent.
    let map = |d: usize, s: usize, dst: usize, max: i64| -> i64 {
        let q = ((2 * d as i64 + 1) * s as i64 * 32768) / dst as i64 - 32768;
        q.clamp(0, max)
    };
    let max_x = ((sw - 1) as i64) << 16;
    let max_y = ((sh - 1) as i64) << 16;
    for ddy in 0..dh {
        let dy = y0 + ddy as i32;
        if dy < 0 || dy as usize >= ch {
            continue;
        }
        let fy = map(ddy, sh, dh, max_y);
        let y0i = (fy >> 16) as usize;
        let y1i = (y0i + 1).min(sh - 1);
        let ty = ((fy >> 8) & 0xFF) as u32;
        for ddx in 0..dw {
            let dx = x0 + ddx as i32;
            if dx < 0 || dx as usize >= cw {
                continue;
            }
            let fx = map(ddx, sw, dw, max_x);
            let x0i = (fx >> 16) as usize;
            let x1i = (x0i + 1).min(sw - 1);
            let tx = ((fx >> 8) & 0xFF) as u32;
            // Bilinear: interpolate the 2x2 source neighbourhood per channel.
            let i00 = (y0i * sw + x0i) * 4;
            let i01 = (y0i * sw + x1i) * 4;
            let i10 = (y1i * sw + x0i) * 4;
            let i11 = (y1i * sw + x1i) * 4;
            let mut px = [0u8; 4];
            for c in 0..4 {
                let top = src[i00 + c] as u32 * (256 - tx) + src[i01 + c] as u32 * tx;
                let bot = src[i10 + c] as u32 * (256 - tx) + src[i11 + c] as u32 * tx;
                px[c] = ((top * (256 - ty) + bot * ty) >> 16) as u8;
            }
            let d = (dy as usize * cw + dx as usize) * 4;
            blend_px(canvas, d, px, galpha);
        }
    }
}

/// One output/source plane of a YUV frame, addressed uniformly across NV12
/// (interleaved chroma, `stride == 2`) and the fully-planar family (`stride == 1`).
/// `hs` / `vs` are this plane's subsampling shift relative to luma.
#[derive(Debug, Clone, Copy)]
struct Chan {
    /// Byte offset of the plane's first sample.
    base: usize,
    /// Plane dimensions in samples.
    w: usize,
    h: usize,
    /// Bytes between successive samples in a row (2 for NV12 chroma, else 1).
    stride: usize,
    /// Byte offset of this channel within an interleaved sample (V in NV12 = 1).
    off: usize,
    /// Subsampling shift relative to luma (1 = half-resolution, 0 = full).
    hs: u32,
    vs: u32,
}

/// The Y, U, V plane layout of `format` at `w x h`. Handles NV12's interleaved
/// chroma and the fully-planar formats through the shared `pixel` helpers.
fn channels(format: RawVideoFormat, w: usize, h: usize) -> [Chan; 3] {
    match format {
        RawVideoFormat::Nv12 => {
            let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
            [
                Chan {
                    base: 0,
                    w,
                    h,
                    stride: 1,
                    off: 0,
                    hs: 0,
                    vs: 0,
                },
                Chan {
                    base: w * h,
                    w: cw,
                    h: ch,
                    stride: 2,
                    off: 0,
                    hs: 1,
                    vs: 1,
                },
                Chan {
                    base: w * h,
                    w: cw,
                    h: ch,
                    stride: 2,
                    off: 1,
                    hs: 1,
                    vs: 1,
                },
            ]
        }
        _ => {
            let p = planar_planes(format, w, h);
            let (hs, vs) = format.chroma_shift().expect("planar YUV");
            [
                Chan {
                    base: p[0].0,
                    w: p[0].1,
                    h: p[0].2,
                    stride: 1,
                    off: 0,
                    hs: 0,
                    vs: 0,
                },
                Chan {
                    base: p[1].0,
                    w: p[1].1,
                    h: p[1].2,
                    stride: 1,
                    off: 0,
                    hs,
                    vs,
                },
                Chan {
                    base: p[2].0,
                    w: p[2].1,
                    h: p[2].2,
                    stride: 1,
                    off: 0,
                    hs,
                    vs,
                },
            ]
        }
    }
}

/// The RGBA background as `[Y, U, V]` in the output's colorimetry, so the fill
/// matches the samples the inputs carry.
fn rgba_to_yuv(rgba: [u8; 4], matrix: &YuvRgbMatrix) -> [u8; 3] {
    let (y, u, v) = matrix.rgb_to_yuv(rgba[0] as i32, rgba[1] as i32, rgba[2] as i32);
    [y as u8, u as u8, v as u8]
}

/// Fill one plane with a constant sample value.
fn fill_channel(buf: &mut [u8], c: &Chan, val: u8) {
    for y in 0..c.h {
        for x in 0..c.w {
            buf[c.base + (y * c.w + x) * c.stride + c.off] = val;
        }
    }
}

/// Blend one source sample over a destination sample with scalar alpha (0..=255).
#[inline]
fn blend_u8(dst: u8, src: u8, alpha: u8) -> u8 {
    let a = alpha as u32;
    ((src as u32 * a + dst as u32 * (255 - a) + 127) / 255) as u8
}

/// Alpha-blend a source plane onto a destination plane at channel-space
/// `(x0, y0)`, with the scalar per-pad `alpha`. Both planes are the same channel
/// of the same format; samples outside the destination are clipped.
#[allow(clippy::too_many_arguments)]
fn blend_channel(dst: &mut [u8], dc: &Chan, src: &[u8], sc: &Chan, x0: i32, y0: i32, alpha: u8) {
    for sy in 0..sc.h {
        let dy = y0 + sy as i32;
        if dy < 0 || dy as usize >= dc.h {
            continue;
        }
        for sx in 0..sc.w {
            let dx = x0 + sx as i32;
            if dx < 0 || dx as usize >= dc.w {
                continue;
            }
            let s = sc.base + (sy * sc.w + sx) * sc.stride + sc.off;
            let d = dc.base + (dy as usize * dc.w + dx as usize) * dc.stride + dc.off;
            dst[d] = blend_u8(dst[d], src[s], alpha);
        }
    }
}

/// Alpha-blend a source plane onto a destination plane, bilinearly resampled to
/// `dw` x `dh` (channel-space) at `(x0, y0)`. Integer fixed-point sampling, same
/// mapping as the packed [`blend_over_scaled`]; clipped to the destination.
#[allow(clippy::too_many_arguments)]
fn blend_channel_scaled(
    dst: &mut [u8],
    dc: &Chan,
    src: &[u8],
    sc: &Chan,
    x0: i32,
    y0: i32,
    dw: usize,
    dh: usize,
    alpha: u8,
) {
    if sc.w == 0 || sc.h == 0 || dw == 0 || dh == 0 {
        return;
    }
    let map = |d: usize, s: usize, dst_dim: usize, max: i64| -> i64 {
        let q = ((2 * d as i64 + 1) * s as i64 * 32768) / dst_dim as i64 - 32768;
        q.clamp(0, max)
    };
    let max_x = ((sc.w - 1) as i64) << 16;
    let max_y = ((sc.h - 1) as i64) << 16;
    let sample = |xi: usize, yi: usize| src[sc.base + (yi * sc.w + xi) * sc.stride + sc.off] as u32;
    for ddy in 0..dh {
        let dy = y0 + ddy as i32;
        if dy < 0 || dy as usize >= dc.h {
            continue;
        }
        let fy = map(ddy, sc.h, dh, max_y);
        let y0i = (fy >> 16) as usize;
        let y1i = (y0i + 1).min(sc.h - 1);
        let ty = ((fy >> 8) & 0xFF) as u32;
        for ddx in 0..dw {
            let dx = x0 + ddx as i32;
            if dx < 0 || dx as usize >= dc.w {
                continue;
            }
            let fx = map(ddx, sc.w, dw, max_x);
            let x0i = (fx >> 16) as usize;
            let x1i = (x0i + 1).min(sc.w - 1);
            let tx = ((fx >> 8) & 0xFF) as u32;
            let top = sample(x0i, y0i) * (256 - tx) + sample(x1i, y0i) * tx;
            let bot = sample(x0i, y1i) * (256 - tx) + sample(x1i, y1i) * tx;
            let v = ((top * (256 - ty) + bot * ty) >> 16) as u8;
            let d = dc.base + (dy as usize * dc.w + dx as usize) * dc.stride + dc.off;
            dst[d] = blend_u8(dst[d], v, alpha);
        }
    }
}

impl MultiInputElement for Compositor {
    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Video compositor",
            "Filter/Editing/Video",
            "Composites several video inputs onto one timed output canvas",
            "g2g",
        )
    }
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Blits on the CPU, so every pad takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn input_count(&self) -> usize {
        self.pads.len()
    }

    /// Only with timed output on: the arm then ticks once per output frame period,
    /// which is when a zero-order-hold frame may be due.
    fn tick_interval_ns(&self) -> Option<u64> {
        match self.state.hold_enabled() {
            true => Some(frame_period_ns(self.framerate_q16)).filter(|&ns| ns > 0),
            false => None,
        }
    }

    fn properties(&self) -> &'static [PropertySpec] {
        COMPOSITOR_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if let Some(applied) = set_pad_property(&mut self.pads, name, &value) {
            return applied;
        }
        match name {
            "width" => self.out_w = dim_property(&value)?,
            "height" => self.out_h = dim_property(&value)?,
            "framerate" => self.framerate_q16 = framerate_property(&value)?,
            "background-color" => self.background = color_property(&value)?,
            "timed-output" => self.state.set_hold(value.as_bool().ok_or(PropError::Type)?),
            "format" => {
                let format = match value.as_str().ok_or(PropError::Type)? {
                    "rgba" | "RGBA" | "rgba8" => RawVideoFormat::Rgba8,
                    "nv12" | "NV12" => RawVideoFormat::Nv12,
                    "i420" | "I420" => RawVideoFormat::I420,
                    "i422" | "I422" => RawVideoFormat::I422,
                    "i444" | "I444" => RawVideoFormat::I444,
                    _ => return Err(PropError::Value),
                };
                self.format = format;
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if let Some(value) = pad_property(&self.pads, name) {
            return Some(value);
        }
        Some(match name {
            "width" => PropValue::Uint(self.out_w as u64),
            "height" => PropValue::Uint(self.out_h as u64),
            "framerate" => PropValue::Fraction((self.framerate_q16 >> 16) as i32, 1),
            "background-color" => color_value(self.background),
            "timed-output" => PropValue::Bool(self.state.hold_enabled()),
            "format" => PropValue::Str(
                match self.format {
                    RawVideoFormat::Nv12 => "nv12",
                    RawVideoFormat::I420 => "i420",
                    RawVideoFormat::I422 => "i422",
                    RawVideoFormat::I444 => "i444",
                    _ => "rgba",
                }
                .into(),
            ),
            _ => return None,
        })
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&self.accepted())
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(self.accepted()))
    }

    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        Ok(CapsConstraint::Produces(CapsSet::one(self.output())))
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            colorimetry,
            ..
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if *format != self.format {
            return Err(G2gError::CapsMismatch);
        }
        // Inputs are mixed sample for sample, so input 0's colour space is the
        // output's. The overlays are assumed to be in it too.
        if input == 0 {
            self.colorimetry = *colorimetry;
        }
        self.state.set_geometry(input, *w, *h);
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(self.output())
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let (w, h) = self.state.geometry(input).ok_or(G2gError::NotConfigured)?;
                    let src = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let need = frame_byte_size(self.format, w, h);
                    if src.len() < need {
                        return Err(G2gError::CapsMismatch);
                    }
                    self.state.ingest(input, frame.timing, src[..need].into());

                    self.announce_output(out).await?;
                    while let Some((timing, base)) = self.state.take_due() {
                        let canvas = self.compose(&base);
                        let frame = self.output_frame(canvas, timing);
                        self.state.record_emitted(timing, base);
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                    }
                }
                // Zero-order-hold: input 0 has not delivered for a whole output
                // period, so re-composite the frame it last did with the overlays
                // as they now stand, rather than letting output freeze with it.
                PipelinePacket::Tick => {
                    let period = frame_period_ns(self.framerate_q16);
                    if let Some((timing, base)) = self.state.take_tick_due(period) {
                        let canvas = self.compose(&base);
                        let frame = self.output_frame(canvas, timing);
                        self.state.record_held(timing, base);
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                    }
                }
                // A per-input caps refinement updates that input's geometry, and
                // input 0's colorimetry the output's; the refined output caps go
                // out with the next frame.
                PipelinePacket::CapsChanged(Caps::RawVideo {
                    format,
                    width: Dim::Fixed(w),
                    height: Dim::Fixed(h),
                    colorimetry,
                    ..
                }) if format == self.format => {
                    if input == 0 {
                        self.colorimetry = colorimetry;
                    }
                    // A geometry change invalidates that input's queued frames:
                    // compose() would otherwise read the old (smaller) bytes
                    // at the new dims and panic out of bounds. For an overlay
                    // the fresh frame repopulates the cache; for input 0 any
                    // startup-buffered frames are dropped too.
                    if self.state.set_geometry(input, w, h) {
                        self.state.clear(input);
                    }
                }
                // A flush on input 0 clears any buffered startup frames (nothing
                // else is cached); on an overlay it also re-arms startup.
                PipelinePacket::Flush => {
                    if input == 0 {
                        self.last_segment = None;
                    }
                    self.state.flush(input);
                }
                // Per-input Eos is informational; the runner aggregates input
                // ends and emits the single merged Eos.
                PipelinePacket::Eos => {}
                // Only the timing input's segment describes the output, whose
                // frames are stamped from input 0. An overlay's own segment would
                // remap the video, so it is consumed here.
                PipelinePacket::Segment(seg) if input == 0 && self.last_segment != Some(seg) => {
                    self.last_segment = Some(seg);
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Segment(_) => {}
                // future PipelinePacket variants: no-op.
                _ => {}
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::runtime::block_on;
    use g2g_core::PushOutcome;

    /// One output frame period at the default 30 fps.
    const PERIOD_NS: u64 = 1_000_000_000 * 65536 / (30 << 16);

    /// Captures what a compositor emits.
    #[derive(Default)]
    struct FrameSink {
        frames: Vec<Frame>,
    }

    impl OutputSink for FrameSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            core::task::Poll::Ready({
                if let PipelinePacket::DataFrame(frame) = packet {
                    self.frames.push(frame);
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    fn rgba_caps(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(30 << 16),
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }
    }

    fn data(bytes: Vec<u8>, pts_ns: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                ..Default::default()
            },
            0,
        ))
    }

    /// A timed-output 4x4 compositor with a 2x2 overlay pad on top, both inputs
    /// configured.
    fn timed_pair() -> Compositor {
        let mut comp = Compositor::new(
            4,
            4,
            Vec::from([
                CompositorPad::at(0, 0),
                CompositorPad::at(0, 0).with_zorder(1),
            ]),
        )
        .with_timed_output();
        comp.configure_pipeline(0, &rgba_caps(4, 4)).unwrap();
        comp.configure_pipeline(1, &rgba_caps(2, 2)).unwrap();
        comp
    }

    fn solid(w: usize, h: usize, rgba: [u8; 4]) -> Vec<u8> {
        let mut v = Vec::with_capacity(w * h * 4);
        for _ in 0..w * h {
            v.extend_from_slice(&rgba);
        }
        v
    }

    /// Seed an overlay input's cached latest frame, as a delivered frame would.
    /// Deliver one frame on `input` and bring the overlays into force at pts 0,
    /// which is what `take_due` does for the frame it releases. These tests call
    /// `compose` directly, so they do that step themselves.
    fn seed(comp: &mut Compositor, input: usize, bytes: Vec<u8>) {
        comp.state
            .ingest(input, FrameTiming::default(), bytes.into());
        comp.state.advance_overlays(0);
    }

    fn px(buf: &[u8], cw: usize, x: usize, y: usize) -> [u8; 4] {
        let i = (y * cw + x) * 4;
        [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]
    }

    #[test]
    fn opaque_overlay_replaces_destination() {
        // 4x4 red canvas-fill, then a 2x2 opaque blue square at (1,1).
        let mut canvas = solid(4, 4, [255, 0, 0, 255]);
        let blue = solid(2, 2, [0, 0, 255, 255]);
        blend_over(&mut canvas, 4, 4, &blue, 2, 2, 1, 1, 255);
        assert_eq!(
            px(&canvas, 4, 0, 0),
            [255, 0, 0, 255],
            "outside the square stays red"
        );
        assert_eq!(
            px(&canvas, 4, 1, 1),
            [0, 0, 255, 255],
            "square is fully blue"
        );
        assert_eq!(px(&canvas, 4, 2, 2), [0, 0, 255, 255], "square corner blue");
        assert_eq!(
            px(&canvas, 4, 3, 3),
            [255, 0, 0, 255],
            "beyond the square stays red"
        );
    }

    #[test]
    fn half_alpha_blends_halfway() {
        // Blue over red at 50% alpha -> roughly (128, 0, 128).
        let mut canvas = solid(2, 2, [255, 0, 0, 255]);
        let blue = solid(2, 2, [0, 0, 255, 255]);
        blend_over(&mut canvas, 2, 2, &blue, 2, 2, 0, 0, 128);
        let p = px(&canvas, 2, 0, 0);
        assert!((p[0] as i32 - 127).abs() <= 2, "red ~half: {}", p[0]);
        assert_eq!(p[1], 0);
        assert!((p[2] as i32 - 128).abs() <= 2, "blue ~half: {}", p[2]);
        assert_eq!(p[3], 255, "canvas stays opaque");
    }

    #[test]
    fn negative_offset_clips_to_canvas() {
        // A 4x4 green source placed at (-2,-2): only its bottom-right 2x2 lands.
        let mut canvas = solid(4, 4, [0, 0, 0, 255]);
        let green = solid(4, 4, [0, 255, 0, 255]);
        blend_over(&mut canvas, 4, 4, &green, 4, 4, -2, -2, 255);
        assert_eq!(px(&canvas, 4, 0, 0), [0, 255, 0, 255], "top-left now green");
        assert_eq!(
            px(&canvas, 4, 1, 1),
            [0, 255, 0, 255],
            "still in the clipped region"
        );
        assert_eq!(
            px(&canvas, 4, 2, 2),
            [0, 0, 0, 255],
            "beyond the source stays black"
        );
    }

    #[test]
    fn scaled_blend_upsamples_a_solid_source() {
        // A 2x2 blue source scaled into a 4x4 region at (1,1) on a 6x6 red
        // canvas: the whole region is blue (uniform bilinear is exact), the
        // border stays red.
        let mut canvas = solid(6, 6, [255, 0, 0, 255]);
        let blue = solid(2, 2, [0, 0, 255, 255]);
        blend_over_scaled(&mut canvas, 6, 6, &blue, 2, 2, 1, 1, 4, 4, 255);
        assert_eq!(px(&canvas, 6, 0, 0), [255, 0, 0, 255], "border stays red");
        assert_eq!(
            px(&canvas, 6, 1, 1),
            [0, 0, 255, 255],
            "region top-left blue"
        );
        assert_eq!(
            px(&canvas, 6, 4, 4),
            [0, 0, 255, 255],
            "region bottom-right blue"
        );
        assert_eq!(
            px(&canvas, 6, 5, 5),
            [255, 0, 0, 255],
            "beyond the region red"
        );
    }

    #[test]
    fn pad_with_size_downscales_overlay_into_the_inset() {
        // Background 8x8 red; a native 4x4 green overlay scaled down to a 2x2
        // inset at (2,2). The inset is green, everything else red.
        let mut comp = Compositor::new(
            8,
            8,
            Vec::from([
                CompositorPad::at(0, 0),
                CompositorPad::at(2, 2).with_zorder(1).with_size(2, 2),
            ]),
        );
        comp.state.set_geometry(0, 8, 8);
        comp.state.set_geometry(1, 4, 4); // native overlay geometry
        let red = solid(8, 8, [255, 0, 0, 255]);
        seed(&mut comp, 1, solid(4, 4, [0, 255, 0, 255]));
        let out = comp.compose(&red);
        assert_eq!(px(&out, 8, 0, 0), [255, 0, 0, 255], "background red");
        assert_eq!(px(&out, 8, 2, 2), [0, 255, 0, 255], "inset top-left green");
        assert_eq!(
            px(&out, 8, 3, 3),
            [0, 255, 0, 255],
            "inset bottom-right green"
        );
        assert_eq!(
            px(&out, 8, 4, 4),
            [255, 0, 0, 255],
            "beyond the 2x2 inset red"
        );
    }

    #[test]
    fn background_shows_where_no_input_covers() {
        // A 4x4 canvas with a blue background; input 0 is a 2x2 green frame at
        // (0,0), so only the top-left quarter is green, the rest the background.
        let mut comp = Compositor::new(4, 4, Vec::from([CompositorPad::at(0, 0)]))
            .with_background([0, 0, 255, 255]);
        comp.state.set_geometry(0, 2, 2);
        let out = comp.compose(&solid(2, 2, [0, 255, 0, 255]));
        assert_eq!(
            px(&out, 4, 0, 0),
            [0, 255, 0, 255],
            "input 0 paints its 2x2"
        );
        assert_eq!(
            px(&out, 4, 3, 3),
            [0, 0, 255, 255],
            "uncovered area is the background"
        );
        // The default background stays opaque black.
        let mut def = Compositor::new(4, 4, Vec::from([CompositorPad::at(0, 0)]));
        def.state.set_geometry(0, 2, 2);
        let out = def.compose(&solid(2, 2, [0, 255, 0, 255]));
        assert_eq!(
            px(&out, 4, 3, 3),
            [0, 0, 0, 255],
            "default background opaque black"
        );
    }

    #[test]
    fn zorder_paints_higher_last() {
        // Two full-canvas pads at the same position; the higher z-order wins.
        let mut comp = Compositor::new(
            2,
            2,
            Vec::from([
                CompositorPad::at(0, 0).with_zorder(1),
                CompositorPad::at(0, 0).with_zorder(5),
            ]),
        );
        comp.state.set_geometry(0, 2, 2);
        comp.state.set_geometry(1, 2, 2);
        let red = solid(2, 2, [255, 0, 0, 255]);
        seed(&mut comp, 1, solid(2, 2, [0, 0, 255, 255]));
        // input 0 (red) is passed as the base; input 1 (blue) has higher z-order.
        let out = comp.compose(&red);
        assert_eq!(
            px(&out, 2, 0, 0),
            [0, 0, 255, 255],
            "z=5 (blue) painted over z=1 (red)"
        );
    }

    /// A solid YUV frame of the given format at `w x h`, every plane filled flat.
    fn solid_yuv(format: RawVideoFormat, w: usize, h: usize, yuv: [u8; 3]) -> Vec<u8> {
        let mut buf = vec![0u8; frame_byte_size(format, w as u32, h as u32)];
        let chans = channels(format, w, h);
        for (c, &v) in chans.iter().zip(yuv.iter()) {
            fill_channel(&mut buf, c, v);
        }
        buf
    }

    /// Read the Y (c=0), U (c=1), or V (c=2) sample under luma pixel `(x, y)`.
    fn yuv_at(
        buf: &[u8],
        format: RawVideoFormat,
        w: usize,
        h: usize,
        c: usize,
        x: usize,
        y: usize,
    ) -> u8 {
        let ch = channels(format, w, h)[c];
        let (cx, cy) = (x >> ch.hs, y >> ch.vs);
        buf[ch.base + (cy * ch.w + cx) * ch.stride + ch.off]
    }

    #[test]
    fn nv12_opaque_overlay_replaces_all_planes() {
        // 8x8 NV12 base (Y50/U60/V70); a 4x4 opaque overlay (Y200/U100/V150) at
        // (2,2). Y and both chroma planes take the overlay inside, base outside.
        let mut comp = Compositor::new(
            8,
            8,
            Vec::from([
                CompositorPad::at(0, 0),
                CompositorPad::at(2, 2).with_zorder(1),
            ]),
        )
        .with_format(RawVideoFormat::Nv12);
        comp.state.set_geometry(0, 8, 8);
        comp.state.set_geometry(1, 4, 4);
        seed(
            &mut comp,
            1,
            solid_yuv(RawVideoFormat::Nv12, 4, 4, [200, 100, 150]),
        );
        let out = comp.compose(&solid_yuv(RawVideoFormat::Nv12, 8, 8, [50, 60, 70]));

        let f = RawVideoFormat::Nv12;
        assert_eq!(
            yuv_at(&out, f, 8, 8, 0, 0, 0),
            50,
            "base luma outside overlay"
        );
        assert_eq!(yuv_at(&out, f, 8, 8, 0, 2, 2), 200, "overlay luma inside");
        assert_eq!(yuv_at(&out, f, 8, 8, 1, 2, 2), 100, "overlay U inside");
        assert_eq!(yuv_at(&out, f, 8, 8, 2, 2, 2), 150, "overlay V inside");
        assert_eq!(yuv_at(&out, f, 8, 8, 1, 0, 0), 60, "base U outside");
        assert_eq!(yuv_at(&out, f, 8, 8, 2, 6, 6), 70, "base V beyond overlay");
    }

    #[test]
    fn i420_opaque_overlay_replaces_all_planes() {
        // Same as the NV12 case for the fully-planar I420 layout (separate U/V).
        let mut comp = Compositor::new(
            8,
            8,
            Vec::from([
                CompositorPad::at(0, 0),
                CompositorPad::at(2, 2).with_zorder(1),
            ]),
        )
        .with_format(RawVideoFormat::I420);
        comp.state.set_geometry(0, 8, 8);
        comp.state.set_geometry(1, 4, 4);
        seed(
            &mut comp,
            1,
            solid_yuv(RawVideoFormat::I420, 4, 4, [200, 100, 150]),
        );
        let out = comp.compose(&solid_yuv(RawVideoFormat::I420, 8, 8, [50, 60, 70]));

        let f = RawVideoFormat::I420;
        assert_eq!(yuv_at(&out, f, 8, 8, 0, 2, 2), 200, "overlay luma");
        assert_eq!(yuv_at(&out, f, 8, 8, 1, 2, 2), 100, "overlay U");
        assert_eq!(yuv_at(&out, f, 8, 8, 2, 2, 2), 150, "overlay V");
        assert_eq!(yuv_at(&out, f, 8, 8, 0, 0, 0), 50, "base luma outside");
        assert_eq!(yuv_at(&out, f, 8, 8, 2, 6, 6), 70, "base V beyond overlay");
    }

    #[test]
    fn nv12_half_alpha_blends_luma_halfway() {
        // Overlay Y200 over base Y50 at alpha 128 -> ~125.
        let mut comp = Compositor::new(
            4,
            4,
            Vec::from([
                CompositorPad::at(0, 0),
                CompositorPad::at(0, 0).with_zorder(1).with_alpha(128),
            ]),
        )
        .with_format(RawVideoFormat::Nv12);
        comp.state.set_geometry(0, 4, 4);
        comp.state.set_geometry(1, 4, 4);
        seed(
            &mut comp,
            1,
            solid_yuv(RawVideoFormat::Nv12, 4, 4, [200, 128, 128]),
        );
        let out = comp.compose(&solid_yuv(RawVideoFormat::Nv12, 4, 4, [50, 128, 128]));
        let y = yuv_at(&out, RawVideoFormat::Nv12, 4, 4, 0, 0, 0);
        assert!(
            (y as i32 - 125).abs() <= 2,
            "half-blended luma ~125, got {y}"
        );
    }

    /// The background fill is converted with the output's colorimetry, so an
    /// untagged mix gets limited-range black (Y16), the black the sinks decode
    /// as black, rather than the full-range Y0 that used to leak through as
    /// crushed shadows.
    #[test]
    fn planar_background_fills_uncovered_area() {
        let mut comp = Compositor::new(4, 4, Vec::from([CompositorPad::at(0, 0)]))
            .with_format(RawVideoFormat::Nv12);
        comp.state.set_geometry(0, 2, 2);
        let out = comp.compose(&solid_yuv(RawVideoFormat::Nv12, 2, 2, [90, 40, 200]));
        let f = RawVideoFormat::Nv12;
        let black = YuvRgbMatrix::new(Colorimetry::UNKNOWN).rgb_to_yuv(0, 0, 0);
        assert_eq!(yuv_at(&out, f, 4, 4, 0, 0, 0), 90, "input paints its luma");
        assert_eq!(
            yuv_at(&out, f, 4, 4, 0, 3, 3),
            black.0 as u8,
            "uncovered luma is limited-range black"
        );
        assert_eq!(
            yuv_at(&out, f, 4, 4, 1, 3, 3),
            black.1 as u8,
            "uncovered U neutral"
        );
        assert_eq!(
            yuv_at(&out, f, 4, 4, 2, 3, 3),
            black.2 as u8,
            "uncovered V neutral"
        );
    }

    /// A tagged input drives the background convert and the output caps: BT.709
    /// full range moves the black point to 0 and the fill to that matrix, and the
    /// element declares what it produced.
    #[test]
    fn background_and_output_caps_follow_input_colorimetry() {
        let colorimetry = Colorimetry {
            range: g2g_core::ColorRange::Full,
            ..Colorimetry::BT709
        };
        let mut comp = Compositor::new(4, 4, Vec::from([CompositorPad::at(0, 0)]))
            .with_format(RawVideoFormat::Nv12)
            .with_background([255, 0, 0, 255]);
        comp.configure_pipeline(
            0,
            &Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(2),
                height: Dim::Fixed(2),
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
                colorimetry,
            },
        )
        .unwrap();

        let Ok(Caps::RawVideo {
            colorimetry: declared,
            ..
        }) = comp.output_caps()
        else {
            panic!("expected raw-video output caps");
        };
        assert_eq!(declared, colorimetry, "the output declares what it mixed");

        let out = comp.compose(&solid_yuv(RawVideoFormat::Nv12, 2, 2, [90, 40, 200]));
        let f = RawVideoFormat::Nv12;
        let red = YuvRgbMatrix::new(colorimetry).rgb_to_yuv(255, 0, 0);
        assert_eq!(yuv_at(&out, f, 4, 4, 0, 3, 3), red.0 as u8);
        assert_eq!(yuv_at(&out, f, 4, 4, 1, 3, 3), red.1 as u8);
        assert_eq!(yuv_at(&out, f, 4, 4, 2, 3, 3), red.2 as u8);
        assert_ne!(
            red,
            YuvRgbMatrix::new(Colorimetry::UNKNOWN).rgb_to_yuv(255, 0, 0),
            "BT.709 full range differs from the untagged default"
        );
    }

    #[test]
    fn planar_scaled_overlay_downsamples_into_inset() {
        // A native 4x4 solid overlay scaled to a 2x2 inset at (2,2) on an 8x8
        // NV12 base. Uniform source: the inset is exactly the overlay colour.
        let mut comp = Compositor::new(
            8,
            8,
            Vec::from([
                CompositorPad::at(0, 0),
                CompositorPad::at(2, 2).with_zorder(1).with_size(2, 2),
            ]),
        )
        .with_format(RawVideoFormat::Nv12);
        comp.state.set_geometry(0, 8, 8);
        comp.state.set_geometry(1, 4, 4);
        seed(
            &mut comp,
            1,
            solid_yuv(RawVideoFormat::Nv12, 4, 4, [200, 100, 150]),
        );
        let out = comp.compose(&solid_yuv(RawVideoFormat::Nv12, 8, 8, [50, 60, 70]));
        let f = RawVideoFormat::Nv12;
        assert_eq!(
            yuv_at(&out, f, 8, 8, 0, 2, 2),
            200,
            "inset luma is the overlay"
        );
        assert_eq!(
            yuv_at(&out, f, 8, 8, 1, 2, 2),
            100,
            "inset U is the overlay"
        );
        assert_eq!(
            yuv_at(&out, f, 8, 8, 0, 5, 5),
            50,
            "beyond the 2x2 inset is base"
        );
    }

    #[test]
    fn with_format_negotiates_the_chosen_yuv_format() {
        let mut comp = Compositor::new(640, 480, Vec::from([CompositorPad::at(0, 0)]))
            .with_format(RawVideoFormat::Nv12);
        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Fixed(30 << 16),
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        assert!(
            comp.configure_pipeline(0, &nv12).is_ok(),
            "matching NV12 input accepted"
        );
        // The output caps carry the chosen format.
        assert!(matches!(
            comp.output(),
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                ..
            }
        ));
        // A mismatched (RGBA) input is rejected.
        let rgba = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Fixed(30 << 16),
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        assert!(matches!(
            comp.configure_pipeline(0, &rgba),
            Err(G2gError::CapsMismatch)
        ));
    }

    #[test]
    fn timed_output_declares_the_frame_period_only_when_enabled() {
        let plain = Compositor::new(4, 4, Vec::from([CompositorPad::at(0, 0)]));
        assert_eq!(plain.tick_interval_ns(), None, "off by default");
        let timed = Compositor::new(4, 4, Vec::from([CompositorPad::at(0, 0)])).with_timed_output();
        assert_eq!(timed.tick_interval_ns(), Some(PERIOD_NS), "30 fps period");
        let fast = Compositor::new(4, 4, Vec::from([CompositorPad::at(0, 0)]))
            .with_timed_output()
            .with_framerate(60);
        assert_eq!(
            fast.tick_interval_ns(),
            Some(1_000_000_000 * 65536 / (60 << 16)),
            "the period follows the output framerate"
        );
    }

    #[test]
    fn a_tick_holds_the_last_frame_and_walks_its_clock() {
        let mut comp = timed_pair();
        let mut sink = FrameSink::default();
        block_on(async {
            comp.process(1, data(solid(2, 2, [0, 255, 0, 255]), 0), &mut sink)
                .await
                .unwrap();
            comp.process(0, data(solid(4, 4, [255, 0, 0, 255]), 1_000), &mut sink)
                .await
                .unwrap();
            assert_eq!(sink.frames.len(), 1, "the real frame composited");
            // The first tick closes the period that frame arrived in.
            comp.process(0, PipelinePacket::Tick, &mut sink)
                .await
                .unwrap();
            assert_eq!(sink.frames.len(), 1, "a period with input needs no hold");
            // Two empty periods: one held frame each, the clock walking forward.
            comp.process(0, PipelinePacket::Tick, &mut sink)
                .await
                .unwrap();
            comp.process(0, PipelinePacket::Tick, &mut sink)
                .await
                .unwrap();
        });
        assert_eq!(sink.frames.len(), 3, "one held frame per empty period");
        let pts: Vec<u64> = sink.frames.iter().map(|f| f.timing.pts_ns).collect();
        assert_eq!(pts, [1_000, 1_000 + PERIOD_NS, 1_000 + 2 * PERIOD_NS]);
        assert_eq!(
            sink.frames[1].timing.dts_ns,
            1_000 + PERIOD_NS,
            "dts walks with pts"
        );
        let px = |f: &Frame| f.domain.as_system_slice().unwrap().to_vec();
        assert_eq!(
            px(&sink.frames[1]),
            px(&sink.frames[0]),
            "the held frame is the same composite"
        );
        assert_eq!(
            sink.frames[2].sequence, 2,
            "held frames keep the output sequence flowing"
        );
    }

    #[test]
    fn a_held_frame_picks_up_the_current_overlay() {
        // The payoff: the background is stalled, but a live overlay keeps moving.
        let mut comp = timed_pair();
        let mut sink = FrameSink::default();
        block_on(async {
            comp.process(1, data(solid(2, 2, [0, 255, 0, 255]), 0), &mut sink)
                .await
                .unwrap();
            comp.process(0, data(solid(4, 4, [255, 0, 0, 255]), 0), &mut sink)
                .await
                .unwrap();
            comp.process(1, data(solid(2, 2, [0, 0, 255, 255]), 0), &mut sink)
                .await
                .unwrap();
            for _ in 0..2 {
                comp.process(0, PipelinePacket::Tick, &mut sink)
                    .await
                    .unwrap();
            }
        });
        assert_eq!(sink.frames.len(), 2, "an overlay alone emits nothing");
        let at = |f: &Frame, x: usize, y: usize| {
            let b = f.domain.as_system_slice().unwrap().to_vec();
            px(&b, 4, x, y)
        };
        assert_eq!(at(&sink.frames[0], 0, 0), [0, 255, 0, 255], "first overlay");
        assert_eq!(
            at(&sink.frames[1], 0, 0),
            [0, 0, 255, 255],
            "the held frame composites the newer overlay"
        );
        assert_eq!(
            at(&sink.frames[1], 3, 3),
            [255, 0, 0, 255],
            "over the same held background"
        );
    }

    #[test]
    fn ticks_emit_nothing_without_a_frame_to_hold() {
        let mut comp = timed_pair();
        let mut sink = FrameSink::default();
        block_on(async {
            for _ in 0..3 {
                comp.process(0, PipelinePacket::Tick, &mut sink)
                    .await
                    .unwrap();
            }
            // An overlay is not a frame to hold: input 0 drives output.
            comp.process(1, data(solid(2, 2, [0, 255, 0, 255]), 0), &mut sink)
                .await
                .unwrap();
            comp.process(0, PipelinePacket::Tick, &mut sink)
                .await
                .unwrap();
        });
        assert!(sink.frames.is_empty(), "nothing emitted before input 0");
    }

    #[test]
    fn a_flush_on_input_0_drops_the_held_frame() {
        let mut comp = timed_pair();
        let mut sink = FrameSink::default();
        block_on(async {
            comp.process(1, data(solid(2, 2, [0, 255, 0, 255]), 0), &mut sink)
                .await
                .unwrap();
            comp.process(0, data(solid(4, 4, [255, 0, 0, 255]), 0), &mut sink)
                .await
                .unwrap();
            comp.process(0, PipelinePacket::Flush, &mut sink)
                .await
                .unwrap();
            for _ in 0..3 {
                comp.process(0, PipelinePacket::Tick, &mut sink)
                    .await
                    .unwrap();
            }
        });
        assert_eq!(
            sink.frames.len(),
            1,
            "the flushed frame is not held across the discontinuity"
        );
    }

    #[test]
    fn a_geometry_change_drops_the_held_frame() {
        // Holding across it would composite the old, smaller buffer at the new
        // dimensions and read out of bounds.
        let mut comp =
            Compositor::new(4, 4, Vec::from([CompositorPad::at(0, 0)])).with_timed_output();
        comp.configure_pipeline(0, &rgba_caps(2, 2)).unwrap();
        let mut sink = FrameSink::default();
        block_on(async {
            comp.process(0, data(solid(2, 2, [255, 0, 0, 255]), 0), &mut sink)
                .await
                .unwrap();
            comp.process(0, PipelinePacket::CapsChanged(rgba_caps(4, 4)), &mut sink)
                .await
                .unwrap();
            for _ in 0..3 {
                comp.process(0, PipelinePacket::Tick, &mut sink)
                    .await
                    .unwrap();
            }
        });
        assert_eq!(sink.frames.len(), 1, "only the frame at the old geometry");
    }

    #[test]
    fn input_every_period_never_holds() {
        let mut comp = timed_pair();
        let mut sink = FrameSink::default();
        block_on(async {
            comp.process(1, data(solid(2, 2, [0, 255, 0, 255]), 0), &mut sink)
                .await
                .unwrap();
            for i in 0..4u64 {
                comp.process(
                    0,
                    data(solid(4, 4, [255, 0, 0, 255]), i * PERIOD_NS),
                    &mut sink,
                )
                .await
                .unwrap();
                comp.process(0, PipelinePacket::Tick, &mut sink)
                    .await
                    .unwrap();
            }
        });
        assert_eq!(
            sink.frames.len(),
            4,
            "a frame in every period leaves nothing to hold"
        );
    }

    #[test]
    fn a_tick_is_ignored_without_timed_output() {
        let mut comp = Compositor::new(4, 4, Vec::from([CompositorPad::at(0, 0)]));
        comp.configure_pipeline(0, &rgba_caps(2, 2)).unwrap();
        let mut sink = FrameSink::default();
        block_on(async {
            comp.process(0, data(solid(2, 2, [255, 0, 0, 255]), 0), &mut sink)
                .await
                .unwrap();
            for _ in 0..3 {
                comp.process(0, PipelinePacket::Tick, &mut sink)
                    .await
                    .unwrap();
            }
        });
        assert_eq!(sink.frames.len(), 1, "no hold without timed output");
    }

    #[test]
    fn negotiation_narrows_to_rgba_and_fixes_output() {
        let comp =
            Compositor::new(1920, 1080, Vec::from([CompositorPad::at(0, 0)])).with_framerate(60);
        assert_eq!(comp.input_count(), 1);
        // Output is the fixed canvas at the construction framerate.
        let CapsConstraint::Produces(set) = comp.caps_constraint_for_output().unwrap() else {
            panic!("expected Produces");
        };
        assert_eq!(
            set.alternatives(),
            &[Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(1920),
                height: Dim::Fixed(1080),
                framerate: Rate::Fixed(60 << 16),
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN
            }]
        );
        // A non-RGBA input is rejected at configure.
        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Fixed(30 << 16),
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        let mut comp = comp;
        assert!(matches!(
            comp.configure_pipeline(0, &nv12),
            Err(G2gError::CapsMismatch)
        ));
    }
}
