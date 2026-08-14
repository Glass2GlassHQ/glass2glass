//! Deinterlace (`deinterlace`). Removes interlacing combs from a raw video frame,
//! preserving format and geometry. CPU-only `no_std`. Packed RGBA / BGRA,
//! semi-planar NV12 / P010, and the fully-planar I420 / I422 / I444 family at 8,
//! 10 and 12 bits.
//!
//! Three methods (a subset of GStreamer's `deinterlace`):
//! - `yadif` (default): the ffmpeg / GStreamer yadif kernel. Each line of the
//!   discarded field is rebuilt from a spatial edge-directed interpolation
//!   clamped to a temporal window, so static areas keep full vertical detail and
//!   moving ones lose the comb. Needs the previous and next frames, so it runs
//!   one frame behind the input.
//! - `linear`: keep the surviving field's lines, replace each line of the other
//!   field with the average of the lines above and below it. No temporal state.
//! - `blend`: each output line is the average of it and the line below, a soft
//!   vertical blur that suppresses combing without dropping a field. It mixes
//!   the two fields uniformly, so unlike the other two it has no field parity to
//!   flip and `tff` does not change what it produces.
//!
//! The `mode` property (M935) mirrors GStreamer's:
//! `interlaced` (default) always weaves, `auto` weaves only when the incoming
//! caps say `Interlace::Interleaved` (the ffmpeg decoder latches that from the
//! per-picture flag) and passes everything else through untouched, `disabled` is
//! a pure passthrough. The default deviates from GStreamer's `auto` on purpose:
//! most g2g upstreams do not declare interlacing, so a hand-inserted
//! `deinterlace` under `auto` would silently do nothing. `playbin` inserts this
//! element with `auto` on every video branch.
//!
//! `fields` (M1048) selects how many output frames one input frame becomes and
//! `tff` the field order, both named after GStreamer's properties. `fields=all`
//! emits one frame per field and doubles the output framerate; the default
//! `auto` emits one frame per input frame, built from whichever field comes
//! first in time. That default deviates from GStreamer's `all`, which would
//! double the rate of every pipeline that already has this element. `tff=auto`
//! means top-field-first: `Caps::Interlace` carries no field order, so there is
//! nothing to detect, and top-first is ffmpeg's assumption for a stream that
//! declares none.
//!
//! In `auto` mode negotiation is transparent (any raw video is accepted and
//! passed through, including formats the kernels cannot process, e.g. packed
//! YUYV), so inserting the element never narrows what a branch can play; an
//! interleaved stream in an unsupported format stays combed rather than failing.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, Rate, RawVideoFormat,
};

use crate::pixel::{even_dims_required, frame_byte_size, planar_planes};

const FORMATS: [RawVideoFormat; 13] = [
    RawVideoFormat::Rgba8,
    RawVideoFormat::Bgra8,
    RawVideoFormat::Nv12,
    RawVideoFormat::P010,
    RawVideoFormat::I420,
    RawVideoFormat::I420p10,
    RawVideoFormat::I420p12,
    RawVideoFormat::I422,
    RawVideoFormat::I422p10,
    RawVideoFormat::I422p12,
    RawVideoFormat::I444,
    RawVideoFormat::I444p10,
    RawVideoFormat::I444p12,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeinterlaceMethod {
    Yadif,
    Linear,
    Blend,
}

impl DeinterlaceMethod {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "yadif" => Some(Self::Yadif),
            "linear" => Some(Self::Linear),
            "blend" => Some(Self::Blend),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Yadif => "yadif",
            Self::Linear => "linear",
            Self::Blend => "blend",
        }
    }
}

/// When the element weaves vs passes through (M935). See the module docs for
/// why the default is `Interlaced`, not GStreamer's `auto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeinterlaceMode {
    /// Weave only when the incoming caps say `Interlace::Interleaved` (and the
    /// format is one the kernels handle); otherwise forward untouched.
    Auto,
    /// Always weave (the pre-M935 behavior).
    Interlaced,
    /// Never weave; pure passthrough.
    Disabled,
}

impl DeinterlaceMode {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(Self::Auto),
            "interlaced" => Some(Self::Interlaced),
            "disabled" => Some(Self::Disabled),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Interlaced => "interlaced",
            Self::Disabled => "disabled",
        }
    }
}

/// How many output frames one input frame becomes, and which field each is built
/// from (GStreamer's `fields`). See the module docs for why the default is
/// `Auto`, not GStreamer's `All`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeinterlaceFields {
    /// One output frame per field: the output framerate doubles.
    All,
    /// One output frame per input frame, keeping the top field.
    Top,
    /// One output frame per input frame, keeping the bottom field.
    Bottom,
    /// One output frame per input frame, keeping whichever field the field order
    /// puts first in time.
    Auto,
}

impl DeinterlaceFields {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "all" => Some(Self::All),
            "top" => Some(Self::Top),
            "bottom" => Some(Self::Bottom),
            "auto" => Some(Self::Auto),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Top => "top",
            Self::Bottom => "bottom",
            Self::Auto => "auto",
        }
    }
}

/// Which field of an interleaved frame comes first in time (GStreamer's `tff`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldOrder {
    /// Nothing to detect: `Caps::Interlace` carries no field order, so this
    /// reads as top-field-first, ffmpeg's assumption for an undeclared stream.
    Auto,
    TopFirst,
    BottomFirst,
}

impl FieldOrder {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(Self::Auto),
            "tff" => Some(Self::TopFirst),
            "bff" => Some(Self::BottomFirst),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::TopFirst => "tff",
            Self::BottomFirst => "bff",
        }
    }
}

/// What the kernels need to know to produce one output frame, in the same two
/// flags ffmpeg's yadif takes.
#[derive(Debug, Clone, Copy)]
struct FieldPass {
    /// The field whose lines survive; the other field's lines are rebuilt.
    keep_top: bool,
    /// Whether the stream's top field is the earlier one. Only yadif reads it,
    /// to pick which temporal pair brackets the rebuilt field.
    top_field_first: bool,
}

impl FieldPass {
    /// The first row this pass rebuilds; every second row after it follows.
    fn first_rebuilt_row(self) -> usize {
        usize::from(self.keep_top)
    }

    /// ffmpeg's `parity ^ tff`: true takes the temporal pair from (prev, cur),
    /// false from (cur, next). Rebuilding the field that comes second in time
    /// reads forward instead of back.
    fn reads_backward(self) -> bool {
        self.keep_top == self.top_field_first
    }
}

/// One sample as the kernels see it. The 10- and 12-bit formats store each
/// sample in a little-endian 16-bit word, and deinterlacing never converts
/// depth, so the kernels run on the stored word: whether the value sits in the
/// word's low bits (the planar `p10` / `p12` family) or its top ones (P010)
/// never reaches them.
trait Sample {
    /// Largest value the storage holds. Every kernel result is already an
    /// average or a copy of its inputs, so this clamp only bounds the cast.
    const MAX: i32;
    fn load(buf: &[u8], offset: usize) -> i32;
    fn store(buf: &mut [u8], offset: usize, value: i32);
}

#[derive(Debug)]
struct Eight;

impl Sample for Eight {
    const MAX: i32 = u8::MAX as i32;

    #[inline]
    fn load(buf: &[u8], offset: usize) -> i32 {
        buf[offset] as i32
    }

    #[inline]
    fn store(buf: &mut [u8], offset: usize, value: i32) {
        buf[offset] = value as u8;
    }
}

#[derive(Debug)]
struct SixteenLittleEndian;

impl Sample for SixteenLittleEndian {
    const MAX: i32 = u16::MAX as i32;

    #[inline]
    fn load(buf: &[u8], offset: usize) -> i32 {
        u16::from_le_bytes([buf[offset], buf[offset + 1]]) as i32
    }

    #[inline]
    fn store(buf: &mut [u8], offset: usize, value: i32) {
        buf[offset..offset + 2].copy_from_slice(&(value as u16).to_le_bytes());
    }
}

/// One deinterlaceable component of a frame: a 2D grid of samples addressed
/// inside the packed buffer. `step` is the byte distance between horizontally
/// adjacent samples, so an interleaved component (one RGBA channel, one half of
/// an NV12 chroma pair) is filtered on its own samples instead of smearing its
/// neighbour's into the prediction.
#[derive(Debug, Clone, Copy)]
struct Component {
    base: usize,
    /// Bytes per row.
    stride: usize,
    /// Bytes between horizontally adjacent samples.
    step: usize,
    /// Samples per row.
    width: usize,
    rows: usize,
}

impl Component {
    #[inline]
    fn at(&self, y: usize, x: usize) -> usize {
        self.base + y * self.stride + x * self.step
    }
}

/// The component grid of one `w x h` frame in `format`. Only the formats
/// `FORMATS` admits reach here.
fn components(format: RawVideoFormat, w: usize, h: usize) -> Vec<Component> {
    let sample = format.bytes_per_sample();
    match format {
        RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8 => (0..4)
            .map(|c| Component {
                base: c,
                stride: w * 4,
                step: 4,
                width: w,
                rows: h,
            })
            .collect(),
        // Semi-planar: the Cb / Cr pair shares one plane, so each half is its own
        // component with the pair's pitch as its step.
        RawVideoFormat::Nv12 | RawVideoFormat::P010 => {
            let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
            let luma = w * h * sample;
            Vec::from([
                Component {
                    base: 0,
                    stride: w * sample,
                    step: sample,
                    width: w,
                    rows: h,
                },
                Component {
                    base: luma,
                    stride: cw * 2 * sample,
                    step: 2 * sample,
                    width: cw,
                    rows: ch,
                },
                Component {
                    base: luma + sample,
                    stride: cw * 2 * sample,
                    step: 2 * sample,
                    width: cw,
                    rows: ch,
                },
            ])
        }
        // Fully planar (the I420 / I422 / I444 family): one component per plane.
        _ => planar_planes(format, w, h)
            .into_iter()
            .map(|(base, pw, ph)| Component {
                base,
                stride: pw * sample,
                step: sample,
                width: pw,
                rows: ph,
            })
            .collect(),
    }
}

/// A held input frame: yadif predicts from the previous and next frames, so the
/// element keeps its own copy of three of them.
#[derive(Debug)]
struct Held {
    data: Vec<u8>,
    timing: FrameTiming,
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::deinterlace::{Deinterlace, DeinterlaceMethod, DeinterlaceMode};
///
/// let deinterlace = Deinterlace::new()
///     .with_method(DeinterlaceMethod::Yadif)
///     .with_mode(DeinterlaceMode::Auto);
/// ```
#[derive(Debug)]
pub struct Deinterlace {
    method: DeinterlaceMethod,
    mode: DeinterlaceMode,
    fields: DeinterlaceFields,
    field_order: FieldOrder,
    /// Whether the current caps get woven (vs forwarded untouched). Decided at
    /// every (re)configure from `mode` and the incoming `Interlace` field.
    active: bool,
    /// The caps as the upstream declared them, forwarded verbatim in
    /// passthrough and re-stamped `Progressive` when weaving.
    incoming_caps: Option<Caps>,
    input: Option<(RawVideoFormat, u32, u32, Rate)>,
    layout: Vec<Component>,
    /// Whether the configured format stores samples as 16-bit words.
    wide_samples: bool,
    frame_bytes: usize,
    /// Nanoseconds between the two outputs of one input frame under
    /// `fields=all`, from the negotiated framerate. Zero when the caps leave the
    /// rate open, and then the frame's own duration stands in.
    field_step_ns: u64,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
    /// yadif's rolling window, mirroring ffmpeg's: `cur` is the frame being
    /// emitted, `prev` / `next` bracket it in time.
    prev: Option<Held>,
    cur: Option<Held>,
    next: Option<Held>,
}

impl Default for Deinterlace {
    fn default() -> Self {
        Self::new()
    }
}

impl Deinterlace {
    pub fn new() -> Self {
        Self {
            method: DeinterlaceMethod::Yadif,
            mode: DeinterlaceMode::Interlaced,
            fields: DeinterlaceFields::Auto,
            field_order: FieldOrder::Auto,
            active: false,
            incoming_caps: None,
            input: None,
            layout: Vec::new(),
            wide_samples: false,
            frame_bytes: 0,
            field_step_ns: 0,
            configured: false,
            last_caps: None,
            emitted: 0,
            prev: None,
            cur: None,
            next: None,
        }
    }

    pub fn with_method(mut self, method: DeinterlaceMethod) -> Self {
        self.method = method;
        self
    }

    pub fn with_mode(mut self, mode: DeinterlaceMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_fields(mut self, fields: DeinterlaceFields) -> Self {
        self.fields = fields;
        self
    }

    pub fn with_field_order(mut self, field_order: FieldOrder) -> Self {
        self.field_order = field_order;
        self
    }

    fn top_field_first(&self) -> bool {
        self.field_order != FieldOrder::BottomFirst
    }

    /// The output frames one input frame becomes, in emission order, as the
    /// first `count` entries of the returned array.
    fn passes(&self) -> ([FieldPass; 2], usize) {
        let top_field_first = self.top_field_first();
        let pass = |keep_top| FieldPass {
            keep_top,
            top_field_first,
        };
        match self.fields {
            // Earlier field first, so the two outputs stay in presentation order.
            DeinterlaceFields::All => ([pass(top_field_first), pass(!top_field_first)], 2),
            DeinterlaceFields::Auto => ([pass(top_field_first); 2], 1),
            DeinterlaceFields::Top => ([pass(true); 2], 1),
            DeinterlaceFields::Bottom => ([pass(false); 2], 1),
        }
    }

    /// Timing of output `index` built from an input frame timed `base`. Under
    /// `fields=all` each output covers one field's worth of time and the second
    /// one starts half a frame period later.
    fn field_timing(&self, base: FrameTiming, index: usize) -> FrameTiming {
        if self.fields != DeinterlaceFields::All {
            return base;
        }
        let mut timing = base;
        timing.duration_ns = base.duration_ns / 2;
        if index > 0 {
            let step = if self.field_step_ns > 0 {
                self.field_step_ns
            } else {
                base.duration_ns / 2
            };
            timing.pts_ns = base.pts_ns.saturating_add(step);
            timing.dts_ns = base.dts_ns.saturating_add(step);
        }
        timing
    }

    /// The weave path's input contract: a fixed, even geometry in one of the
    /// kernel formats. `None` when the caps are raw video the kernels cannot
    /// process (only an error for the modes that must process).
    fn weavable(caps: &Caps) -> Option<(RawVideoFormat, u32, u32, Rate)> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate,
            ..
        } = caps
        else {
            return None;
        };
        if !FORMATS.contains(format) || *w == 0 || *h == 0 {
            return None;
        }
        // A subsampled format at an odd dimension has no whole chroma grid, so
        // the plane layout below would not describe the buffer it is given.
        let (even_w, even_h) = even_dims_required(*format);
        if (even_w && *w % 2 != 0) || (even_h && *h % 2 != 0) {
            return None;
        }
        Some((*format, *w, *h, framerate.clone()))
    }

    fn reconfigure(&mut self, caps: &Caps) -> Result<(), G2gError> {
        if !matches!(caps, Caps::RawVideo { .. }) {
            return Err(G2gError::CapsMismatch);
        }
        let weavable = Self::weavable(caps);
        // The pre-M935 contract: an explicit always-on deinterlace rejects caps
        // it cannot process, loud.
        if self.mode == DeinterlaceMode::Interlaced && weavable.is_none() {
            return Err(G2gError::CapsMismatch);
        }
        let active = weaves(self.mode, caps);
        if active {
            let (format, w, h, rate) = weavable.ok_or(G2gError::CapsMismatch)?;
            // A geometry change invalidates the held window: its frames are the
            // old size and cannot be combined with the new ones.
            if self.input.as_ref().map(|(f, w, h, _)| (*f, *w, *h)) != Some((format, w, h)) {
                self.prev = None;
                self.cur = None;
                self.next = None;
            }
            self.layout = components(format, w as usize, h as usize);
            self.wide_samples = format.bytes_per_sample() == 2;
            self.frame_bytes = frame_byte_size(format, w, h);
            self.field_step_ns = half_frame_period_ns(&rate);
            self.input = Some((format, w, h, rate));
        } else {
            self.prev = None;
            self.cur = None;
            self.next = None;
            self.input = None;
        }
        if active != self.active {
            g2g_core::g2g_debug!(
                self,
                "{}: {}",
                self.mode.as_str(),
                if active { "weaving" } else { "passthrough" }
            );
        }
        self.active = active;
        self.incoming_caps = Some(caps.clone());
        Ok(())
    }

    fn out_caps(&self) -> Option<Caps> {
        let incoming = self.incoming_caps.as_ref()?;
        if !self.active {
            // Passthrough forwards the upstream declaration verbatim.
            return Some(incoming.clone());
        }
        let mut caps = incoming.clone();
        if let Caps::RawVideo {
            interlace,
            framerate,
            ..
        } = &mut caps
        {
            *interlace = g2g_core::Interlace::Progressive;
            if self.fields == DeinterlaceFields::All {
                *framerate = doubled_rate(framerate);
            }
        }
        Some(caps)
    }

    async fn emit(
        &mut self,
        data: Vec<u8>,
        timing: FrameTiming,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        if let Some(caps) = self.out_caps() {
            if self.last_caps.as_ref() != Some(&caps) {
                out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                self.last_caps = Some(caps);
            }
        }
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            timing,
            sequence: self.emitted,
            meta: Default::default(),
        };
        self.emitted += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }

    /// Run yadif over the held window for one output frame of `cur`.
    fn yadif_current(&self, pass: FieldPass) -> Option<Vec<u8>> {
        let (prev, cur, next) = (self.prev.as_ref()?, self.cur.as_ref()?, self.next.as_ref()?);
        let mut dst = cur.data.clone();
        for c in &self.layout {
            if self.wide_samples {
                yadif_component::<SixteenLittleEndian>(
                    &prev.data, &cur.data, &next.data, &mut dst, *c, pass,
                );
            } else {
                yadif_component::<Eight>(&prev.data, &cur.data, &next.data, &mut dst, *c, pass);
            }
        }
        Some(dst)
    }

    /// Emit every output frame the held yadif window is ready to produce, one
    /// per field under `fields=all`. A window short of `prev` / `next` yields
    /// nothing yet.
    async fn emit_yadif_window(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let Some(base) = self.cur.as_ref().map(|f| f.timing) else {
            return Ok(());
        };
        let (passes, count) = self.passes();
        for (index, pass) in passes[..count].iter().enumerate() {
            let Some(data) = self.yadif_current(*pass) else {
                return Ok(());
            };
            let timing = self.field_timing(base, index);
            self.emit(data, timing, out).await?;
        }
        Ok(())
    }
}

/// Whether `caps` reach the kernels rather than passing through. The `auto`
/// answer is provisional at negotiation time: a stream the decoder only later
/// declares interleaved starts weaving mid-run, and the runtime `CapsChanged`
/// carries the corrected output caps.
fn weaves(mode: DeinterlaceMode, caps: &Caps) -> bool {
    match mode {
        DeinterlaceMode::Disabled => false,
        DeinterlaceMode::Interlaced => Deinterlace::weavable(caps).is_some(),
        // Only a declared-interleaved stream in a format the kernels handle:
        // progressive, undeclared and unsupported all forward untouched.
        DeinterlaceMode::Auto => {
            matches!(
                caps,
                Caps::RawVideo {
                    interlace: g2g_core::Interlace::Interleaved,
                    ..
                }
            ) && Deinterlace::weavable(caps).is_some()
        }
    }
}

/// One output frame per field means twice the frames per second. An open rate
/// stays open: doubling an unconstrained span says nothing new.
fn doubled_rate(rate: &Rate) -> Rate {
    match rate {
        Rate::Fixed(v) => Rate::Fixed(v.saturating_mul(2)),
        Rate::Range { min_q16, max_q16 } => Rate::Range {
            min_q16: min_q16.saturating_mul(2),
            max_q16: max_q16.saturating_mul(2),
        },
        Rate::Any => Rate::Any,
    }
}

/// Nanoseconds per field at a Q16 frames-per-second rate, zero when the rate is
/// not fixed.
fn half_frame_period_ns(rate: &Rate) -> u64 {
    const HALF_SECOND_NS_Q16: u64 = 500_000_000 * 65536;
    match rate {
        Rate::Fixed(q16) if *q16 > 0 => HALF_SECOND_NS_Q16 / *q16 as u64,
        _ => 0,
    }
}

impl g2g_core::log::LogSource for Deinterlace {
    fn log_category(&self) -> &'static str {
        g2g_core::log::short_type_name::<Self>()
    }
}

impl AsyncElement for Deinterlace {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Auto / disabled pass anything raw through, so negotiation must not
        // narrow the branch to the kernel formats.
        if self.mode != DeinterlaceMode::Interlaced {
            return match upstream_caps {
                Caps::RawVideo { .. } => Ok(upstream_caps.clone()),
                _ => Err(G2gError::CapsMismatch),
            };
        }
        for format in FORMATS {
            let candidate = Caps::RawVideo {
                format,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
            };
            if let Ok(narrowed) = upstream_caps.intersect(&candidate) {
                return Ok(narrowed);
            }
        }
        Err(G2gError::CapsMismatch)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let mode = self.mode;
        let doubles = self.fields == DeinterlaceFields::All;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| {
            let Caps::RawVideo { format, .. } = input else {
                return CapsSet::from_alternatives(Vec::new());
            };
            if mode == DeinterlaceMode::Interlaced && !FORMATS.contains(format) {
                return CapsSet::from_alternatives(Vec::new());
            }
            let mut out = input.clone();
            if let Caps::RawVideo {
                interlace,
                framerate,
                ..
            } = &mut out
            {
                // The output never declares interlacing: weaving produces
                // progressive frames, and a passthrough of an unweavable stream
                // is the runtime exception the mid-stream CapsChanged corrects.
                *interlace = g2g_core::Interlace::Progressive;
                // Only a stream that actually reaches the kernels gains frames,
                // and the same CapsChanged corrects an `auto` guess that the
                // decoder later contradicts.
                if doubles && weaves(mode, input) {
                    *framerate = doubled_rate(framerate);
                }
            }
            CapsSet::one(out)
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.reconfigure(absolute_caps)?;
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
                PipelinePacket::DataFrame(frame) => {
                    if !self.active {
                        // Passthrough: forward the frame untouched (any format,
                        // any memory domain), declaring caps first if needed.
                        if let Some(caps) = self.out_caps() {
                            if self.last_caps.as_ref() != Some(&caps) {
                                out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                                self.last_caps = Some(caps);
                            }
                        }
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                        return Ok(());
                    }
                    if self.input.is_none() {
                        return Err(G2gError::NotConfigured);
                    }
                    let src = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let n = self.frame_bytes;
                    if src.len() < n {
                        return Err(G2gError::CapsMismatch);
                    }
                    if self.method == DeinterlaceMethod::Yadif {
                        // ffmpeg's window shift: the first frame stands in for its
                        // own predecessor, so nothing is emitted until a second
                        // one arrives and the element runs one frame behind.
                        self.prev = self.cur.take();
                        self.cur = self.next.take();
                        self.next = Some(Held {
                            data: src[..n].to_vec(),
                            timing: frame.timing,
                        });
                        if self.cur.is_none() {
                            self.cur = self.next.as_ref().map(|f| Held {
                                data: f.data.clone(),
                                timing: f.timing,
                            });
                        }
                        self.emit_yadif_window(out).await?;
                    } else {
                        let base = frame.timing;
                        let (passes, count) = self.passes();
                        for (index, pass) in passes[..count].iter().enumerate() {
                            let mut dst = vec![0u8; n];
                            dst.copy_from_slice(&src[..n]);
                            for c in &self.layout {
                                if self.wide_samples {
                                    blend_component::<SixteenLittleEndian>(
                                        &src[..n],
                                        &mut dst,
                                        *c,
                                        self.method,
                                        *pass,
                                    );
                                } else {
                                    blend_component::<Eight>(
                                        &src[..n],
                                        &mut dst,
                                        *c,
                                        self.method,
                                        *pass,
                                    );
                                }
                            }
                            let timing = self.field_timing(base, index);
                            self.emit(dst, timing, out).await?;
                        }
                    }
                }
                PipelinePacket::CapsChanged(c) => {
                    // The runner calls `configure_pipeline` (input) before
                    // pushing this packet, whose caps are the pre-fixed forward
                    // *output*, not a new input (both sides are `RawVideo`, so
                    // they cannot be told apart by variant; see `VideoConvert`).
                    // Adopting it as input would read our own Progressive-stamped
                    // output as "the stream went progressive" and flip auto mode
                    // back to passthrough right after the decoder declared
                    // interleaved. Forward it and record it for the emit dedup.
                    out.push(PipelinePacket::CapsChanged(c.clone())).await?;
                    self.last_caps = Some(c);
                }
                PipelinePacket::Flush => {
                    self.last_caps = None;
                    self.prev = None;
                    self.cur = None;
                    self.next = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Eos => {
                    // ffmpeg feeds a copy of the last frame so the real one still
                    // gets a `next` to predict from: N in, N out.
                    self.prev = self.cur.take();
                    self.cur = self.next.take();
                    self.next = self.cur.as_ref().map(|f| Held {
                        data: f.data.clone(),
                        timing: f.timing,
                    });
                    self.emit_yadif_window(out).await?;
                    self.prev = None;
                    self.cur = None;
                    self.next = None;
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        DEINTERLACE_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Deinterlace",
            "Filter/Effect/Video/Deinterlace",
            "Deinterlaces video",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "method" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.method = DeinterlaceMethod::from_str(s).ok_or(PropError::Value)?;
            }
            "mode" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.mode = DeinterlaceMode::from_str(s).ok_or(PropError::Value)?;
            }
            "fields" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.fields = DeinterlaceFields::from_str(s).ok_or(PropError::Value)?;
            }
            "tff" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.field_order = FieldOrder::from_str(s).ok_or(PropError::Value)?;
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "method" => Some(PropValue::Str(self.method.as_str().into())),
            "mode" => Some(PropValue::Str(self.mode.as_str().into())),
            "fields" => Some(PropValue::Str(self.fields.as_str().into())),
            "tff" => Some(PropValue::Str(self.field_order.as_str().into())),
            _ => None,
        }
    }
}

static DEINTERLACE_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "method",
        PropKind::Str,
        "deinterlace method: yadif | linear | blend",
    )
    .with_enum_values("yadif | linear | blend"),
    PropertySpec::new(
        "mode",
        PropKind::Str,
        "when to deinterlace: auto (only caps-declared interleaved) | interlaced (always) | disabled",
    )
    .with_enum_values("auto | interlaced | disabled"),
    PropertySpec::new(
        "fields",
        PropKind::Str,
        "fields to output: all (one frame per field, double rate) | top | bottom | auto (the field that comes first in time)",
    )
    .with_enum_values("all | top | bottom | auto"),
    PropertySpec::new(
        "tff",
        PropKind::Str,
        "field order: auto (top field first) | tff | bff",
    )
    .with_enum_values("auto | tff | bff"),
];

impl PadTemplates for Deinterlace {
    fn pad_templates() -> Vec<PadTemplate> {
        let any_geometry = |format| Caps::RawVideo {
            format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
        };
        let set = CapsSet::from_alternatives(FORMATS.map(any_geometry).to_vec());
        Vec::from([PadTemplate::sink(set.clone()), PadTemplate::source(set)])
    }
}

/// The `linear` / `blend` methods over one component. `dst` already holds a copy
/// of `src`, so a row either method leaves alone needs no write.
fn blend_component<S: Sample>(
    src: &[u8],
    dst: &mut [u8],
    c: Component,
    method: DeinterlaceMethod,
    pass: FieldPass,
) {
    let avg = |a: i32, b: i32| ((a + b) / 2).clamp(0, S::MAX);
    match method {
        DeinterlaceMethod::Linear => {
            // The rows of the field this pass discards are rebuilt from their
            // neighbours. A rebuilt row at the very top or bottom has only one
            // neighbour, so it stays as it came in.
            let mut y = if pass.keep_top { 1 } else { 2 };
            while y + 1 < c.rows {
                for x in 0..c.width {
                    let value = avg(S::load(src, c.at(y - 1, x)), S::load(src, c.at(y + 1, x)));
                    S::store(dst, c.at(y, x), value);
                }
                y += 2;
            }
        }
        // A uniform vertical blur over both fields: no field parity to follow, so
        // `pass` does not change the result.
        DeinterlaceMethod::Blend => {
            for y in 0..c.rows.saturating_sub(1) {
                for x in 0..c.width {
                    let value = avg(S::load(src, c.at(y, x)), S::load(src, c.at(y + 1, x)));
                    S::store(dst, c.at(y, x), value);
                }
            }
        }
        DeinterlaceMethod::Yadif => unreachable!("yadif runs on the held window"),
    }
}

/// yadif over one component for one output frame, a port of ffmpeg's
/// `vf_yadif.c` `FILTER` / `CHECK` kernel.
///
/// `pass` names the field whose rows survive (`dst` already holds a copy of
/// `cur`, so those rows need no write) and the field order. Together they pick
/// the temporal pair bracketing the rebuilt field: `(prev, cur)` when it comes
/// first in time, `(cur, next)` when it comes second, which is ffmpeg's
/// `parity ^ tff` reaching its line filter.
///
/// The first and last three columns take the plain `(above + below) / 2` spatial
/// predictor instead of the edge-directed search, exactly as ffmpeg's
/// `filter_edges` does, because the search reads three samples to either side.
fn yadif_component<S: Sample>(
    prev: &[u8],
    cur: &[u8],
    next: &[u8],
    dst: &mut [u8],
    c: Component,
    pass: FieldPass,
) {
    // Two rows are the minimum the row mirroring below is defined for.
    if c.rows < 2 {
        return;
    }
    let (prev2, next2) = if pass.reads_backward() {
        (prev, cur)
    } else {
        (cur, next)
    };
    for y in (pass.first_rebuilt_row()..c.rows).step_by(2) {
        // ffmpeg mirrors at both edges: the row above the first row is the one
        // below it, and the row below the last row is the one above it.
        let up: isize = if y > 0 { -1 } else { 1 };
        let down: isize = if y + 1 < c.rows { 1 } else { -1 };
        let above = (y as isize + up) as usize;
        let below = (y as isize + down) as usize;
        // ffmpeg forces mode 2 on the rows whose second-order neighbours would
        // fall outside the frame, which drops the b / f interval check. That is
        // also what keeps `above2` / `below2` inside the component.
        let second_order = (y != 1 && y + 2 != c.rows).then(|| {
            (
                (y as isize + 2 * up) as usize,
                (y as isize + 2 * down) as usize,
            )
        });
        for x in 0..c.width {
            let s = |buf: &[u8], yy: usize, xx: usize| S::load(buf, c.at(yy, xx));
            let cc = s(cur, above, x);
            let e = s(cur, below, x);
            let (p0, n0) = (s(prev2, y, x), s(next2, y, x));
            let d = (p0 + n0) >> 1;
            let td0 = (p0 - n0).abs();
            let td1 = ((s(prev, above, x) - cc).abs() + (s(prev, below, x) - e).abs()) >> 1;
            let td2 = ((s(next, above, x) - cc).abs() + (s(next, below, x) - e).abs()) >> 1;
            let mut diff = (td0 >> 1).max(td1).max(td2);
            let mut pred = (cc + e) >> 1;

            if x >= 3 && x + 3 < c.width {
                let xi = x as isize;
                let sx = |yy: usize, xx: isize| S::load(cur, c.at(yy, xx as usize));
                let score = |j: isize| -> i32 {
                    (sx(above, xi - 1 + j) - sx(below, xi - 1 - j)).abs()
                        + (sx(above, xi + j) - sx(below, xi - j)).abs()
                        + (sx(above, xi + 1 + j) - sx(below, xi + 1 - j)).abs()
                };
                let interp = |j: isize| (sx(above, xi + j) + sx(below, xi - j)) >> 1;
                let mut best = score(0) - 1;
                // The +/-2 offset is only considered when the +/-1 one improved,
                // matching the nesting of ffmpeg's CHECK macro.
                for dir in [-1isize, 1] {
                    let one = score(dir);
                    if one < best {
                        best = one;
                        pred = interp(dir);
                        let two = score(dir * 2);
                        if two < best {
                            best = two;
                            pred = interp(dir * 2);
                        }
                    }
                }
            }

            if let Some((above2, below2)) = second_order {
                let b = (s(prev2, above2, x) + s(next2, above2, x)) >> 1;
                let f = (s(prev2, below2, x) + s(next2, below2, x)) >> 1;
                let max = (d - e).max(d - cc).max((b - cc).min(f - e));
                let min = (d - e).min(d - cc).min((b - cc).max(f - e));
                diff = diff.max(min).max(-max);
            }

            if pred > d + diff {
                pred = d + diff;
            } else if pred < d - diff {
                pred = d - diff;
            }
            S::store(dst, c.at(y, x), pred.clamp(0, S::MAX));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgba(w: usize, h: usize) -> Component {
        Component {
            base: 0,
            stride: w * 4,
            step: 4,
            width: w,
            rows: h,
        }
    }

    // 1px-wide, 4px-tall RGBA frame with alternating black/white lines (a comb).
    fn comb() -> Vec<u8> {
        let mut v = Vec::new();
        for y in 0..4 {
            let c = if y % 2 == 0 { 0u8 } else { 255u8 };
            v.extend_from_slice(&[c, c, c, 255]);
        }
        v
    }

    const KEEP_TOP: FieldPass = FieldPass {
        keep_top: true,
        top_field_first: true,
    };

    fn run(src: &[u8], w: usize, h: usize, method: DeinterlaceMethod) -> Vec<u8> {
        let mut dst = src.to_vec();
        for c in components(RawVideoFormat::Rgba8, w, h) {
            blend_component::<Eight>(src, &mut dst, c, method, KEEP_TOP);
        }
        dst
    }

    #[test]
    fn linear_interpolates_odd_lines() {
        let dst = run(&comb(), 1, 4, DeinterlaceMethod::Linear);
        // even lines (0,2) stay 0; odd line 1 = avg(0,0)=0; line 3 is last -> passthrough 255.
        assert_eq!(dst[0..4], [0, 0, 0, 255]);
        assert_eq!(dst[4..8], [0, 0, 0, 255]);
        assert_eq!(dst[8..12], [0, 0, 0, 255]);
        assert_eq!(dst[12..16], [255, 255, 255, 255]);
    }

    #[test]
    fn blend_softens_edges() {
        let dst = run(&comb(), 1, 4, DeinterlaceMethod::Blend);
        // line 0 = avg(0,255)=127; the comb is reduced (no full 0/255 jump).
        assert_eq!(dst[0], 127);
        assert_eq!(dst[4], 127);
    }

    /// A static scene: yadif's temporal window agrees with the spatial predictor,
    /// so the kept field passes through and the rebuilt field matches its
    /// neighbours rather than inventing detail.
    #[test]
    fn yadif_static_scene_is_stable() {
        let (w, h) = (8usize, 8usize);
        let mut src = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let v = (x * 16 + y) as u8;
                src[(y * w + x) * 4..][..4].copy_from_slice(&[v, v, v, 255]);
            }
        }
        let mut dst = src.clone();
        yadif_component::<Eight>(&src, &src, &src, &mut dst, rgba(w, h), KEEP_TOP);
        assert_eq!(dst, src, "a frame identical to its neighbours is unchanged");
    }

    /// Every rebuilt row indexes its first- and second-order neighbours through
    /// the same mirroring, so no component shape can walk off either edge.
    #[test]
    fn every_shape_and_parity_stays_inside_the_component() {
        for rows in 2..12usize {
            for keep_top in [true, false] {
                let w = 8usize;
                let src = vec![7u8; w * rows * 4];
                let mut dst = src.clone();
                let pass = FieldPass {
                    keep_top,
                    top_field_first: true,
                };
                yadif_component::<Eight>(&src, &src, &src, &mut dst, rgba(w, rows), pass);
                assert_eq!(dst, src, "flat input, {rows} rows, keep_top {keep_top}");
            }
        }
    }

    #[test]
    fn bottom_field_first_keeps_the_other_field() {
        let mut element = Deinterlace::new();
        assert!(element.top_field_first());
        let (passes, count) = element.passes();
        assert_eq!((passes[0].keep_top, count), (true, 1));

        element.field_order = FieldOrder::BottomFirst;
        let (passes, count) = element.passes();
        assert_eq!((passes[0].keep_top, count), (false, 1));
        assert_eq!(passes[0].first_rebuilt_row(), 0);
        assert!(passes[0].reads_backward(), "the earlier field reads back");

        element.fields = DeinterlaceFields::All;
        let (passes, count) = element.passes();
        assert_eq!(count, 2);
        assert_eq!((passes[0].keep_top, passes[1].keep_top), (false, true));
        assert!(!passes[1].reads_backward(), "the later field reads forward");
    }

    #[test]
    fn all_fields_doubles_the_rate_and_halves_the_step() {
        assert_eq!(doubled_rate(&Rate::Fixed(25 << 16)), Rate::Fixed(50 << 16));
        assert_eq!(doubled_rate(&Rate::Any), Rate::Any);
        assert_eq!(half_frame_period_ns(&Rate::Fixed(25 << 16)), 20_000_000);
        assert_eq!(half_frame_period_ns(&Rate::Any), 0);
    }

    #[test]
    fn method_property_round_trips() {
        let mut d = Deinterlace::new();
        assert_eq!(
            d.get_property("method"),
            Some(PropValue::Str("yadif".into()))
        );
        d.set_property("method", PropValue::Str("blend".into()))
            .unwrap();
        assert_eq!(
            d.get_property("method"),
            Some(PropValue::Str("blend".into()))
        );
        assert_eq!(
            d.set_property("method", PropValue::Str("nope".into()))
                .unwrap_err(),
            PropError::Value
        );
    }
}
