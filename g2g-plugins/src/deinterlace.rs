//! Deinterlace (`deinterlace`). Removes interlacing combs from a raw video frame,
//! preserving format and geometry, one frame out per frame in (single rate: no
//! field doubling). CPU-only `no_std`. Packed RGBA / BGRA and planar I420 / NV12.
//!
//! Three methods (a subset of GStreamer's `deinterlace`):
//! - `yadif` (default): the ffmpeg / GStreamer yadif kernel, single-rate. Each
//!   line of the discarded field is rebuilt from a spatial edge-directed
//!   interpolation clamped to a temporal window, so static areas keep full
//!   vertical detail and moving ones lose the comb. Needs the previous and next
//!   frames, so it runs one frame behind the input.
//! - `linear`: keep the top field's lines, replace each bottom-field line with the
//!   average of the lines above and below it. No temporal state.
//! - `blend`: each output line is the average of it and the line below, a soft
//!   vertical blur that suppresses combing without dropping a field.
//!
//! Field order is assumed top-field-first, matching ffmpeg's default for a stream
//! that declares nothing. The `mode` property (M935) mirrors GStreamer's:
//! `interlaced` (default) always weaves, `auto` weaves only when the incoming
//! caps say `Interlace::Interleaved` (the ffmpeg decoder latches that from the
//! per-picture flag) and passes everything else through untouched, `disabled` is
//! a pure passthrough. The default deviates from GStreamer's `auto` on purpose:
//! most g2g upstreams do not declare interlacing, so a hand-inserted
//! `deinterlace` under `auto` would silently do nothing. `playbin` inserts this
//! element with `auto` on every video branch.
//!
//! In `auto` mode negotiation is transparent (any raw video is accepted and
//! passed through, including formats the kernels cannot process, e.g. 10-bit
//! planar), so inserting the element never narrows what a branch can play; an
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

const FORMATS: [RawVideoFormat; 4] = [
    RawVideoFormat::Rgba8,
    RawVideoFormat::Bgra8,
    RawVideoFormat::Nv12,
    RawVideoFormat::I420,
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

/// The component grid of one `w x h` frame in `format`. Only the four formats
/// `FORMATS` admits reach here.
fn components(format: RawVideoFormat, w: usize, h: usize) -> Vec<Component> {
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
        RawVideoFormat::Nv12 => {
            let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
            let luma = w * h;
            Vec::from([
                Component {
                    base: 0,
                    stride: w,
                    step: 1,
                    width: w,
                    rows: h,
                },
                Component {
                    base: luma,
                    stride: cw * 2,
                    step: 2,
                    width: cw,
                    rows: ch,
                },
                Component {
                    base: luma + 1,
                    stride: cw * 2,
                    step: 2,
                    width: cw,
                    rows: ch,
                },
            ])
        }
        // Fully planar (I420 here): one component per plane.
        _ => planar_planes(format, w, h)
            .into_iter()
            .map(|(base, pw, ph)| Component {
                base,
                stride: pw,
                step: 1,
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

#[derive(Debug)]
pub struct Deinterlace {
    method: DeinterlaceMethod,
    mode: DeinterlaceMode,
    /// Whether the current caps get woven (vs forwarded untouched). Decided at
    /// every (re)configure from `mode` and the incoming `Interlace` field.
    active: bool,
    /// The caps as the upstream declared them, forwarded verbatim in
    /// passthrough and re-stamped `Progressive` when weaving.
    incoming_caps: Option<Caps>,
    input: Option<(RawVideoFormat, u32, u32, Rate)>,
    layout: Vec<Component>,
    frame_bytes: usize,
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
            active: false,
            incoming_caps: None,
            input: None,
            layout: Vec::new(),
            frame_bytes: 0,
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

    /// The weave path's input contract: a fixed, even geometry in one of the
    /// four kernel formats. `None` when the caps are raw video the kernels
    /// cannot process (only an error for the modes that must process).
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
        let Caps::RawVideo { interlace, .. } = caps else {
            return Err(G2gError::CapsMismatch);
        };
        let weavable = Self::weavable(caps);
        let active = match self.mode {
            DeinterlaceMode::Disabled => false,
            // The pre-M935 contract: an explicit always-on deinterlace rejects
            // caps it cannot process, loud.
            DeinterlaceMode::Interlaced => {
                if weavable.is_none() {
                    return Err(G2gError::CapsMismatch);
                }
                true
            }
            // Auto: weave only a declared-interleaved stream in a format the
            // kernels handle; anything else (progressive, undeclared, 10-bit)
            // forwards untouched.
            DeinterlaceMode::Auto => {
                *interlace == g2g_core::Interlace::Interleaved && weavable.is_some()
            }
        };
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
            self.frame_bytes = frame_byte_size(format, w, h);
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
        if let Caps::RawVideo { interlace, .. } = &mut caps {
            *interlace = g2g_core::Interlace::Progressive;
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

    /// Run yadif over the held window and emit the result for `cur`.
    fn yadif_current(&self) -> Option<(Vec<u8>, FrameTiming)> {
        let (prev, cur, next) = (self.prev.as_ref()?, self.cur.as_ref()?, self.next.as_ref()?);
        let mut dst = cur.data.clone();
        for c in &self.layout {
            yadif_component(&prev.data, &cur.data, &next.data, &mut dst, *c);
        }
        Some((dst, cur.timing))
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
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            // The output never declares interlacing: weaving produces
            // progressive frames, and a passthrough of an unweavable stream is
            // the runtime exception the mid-stream CapsChanged corrects.
            Caps::RawVideo { .. } if mode != DeinterlaceMode::Interlaced => {
                let mut out = input.clone();
                if let Caps::RawVideo { interlace, .. } = &mut out {
                    *interlace = g2g_core::Interlace::Progressive;
                }
                CapsSet::one(out)
            }
            Caps::RawVideo { format, .. } if FORMATS.contains(format) => {
                let mut out = input.clone();
                if let Caps::RawVideo { interlace, .. } = &mut out {
                    *interlace = g2g_core::Interlace::Progressive;
                }
                CapsSet::one(out)
            }
            _ => CapsSet::from_alternatives(Vec::new()),
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
                    let Some(src) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
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
                        if let Some((data, timing)) = self.yadif_current() {
                            self.emit(data, timing, out).await?;
                        }
                    } else {
                        let mut dst = vec![0u8; n];
                        dst.copy_from_slice(&src[..n]);
                        for c in &self.layout {
                            blend_component(&src[..n], &mut dst, *c, self.method);
                        }
                        self.emit(dst, frame.timing, out).await?;
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
                    if let Some((data, timing)) = self.yadif_current() {
                        self.emit(data, timing, out).await?;
                    }
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
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "method" => Some(PropValue::Str(self.method.as_str().into())),
            "mode" => Some(PropValue::Str(self.mode.as_str().into())),
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

fn avg(a: u8, b: u8) -> u8 {
    ((a as u16 + b as u16) / 2) as u8
}

/// The `linear` / `blend` methods over one component. `dst` already holds a copy
/// of `src`, so a row either method leaves alone needs no write.
fn blend_component(src: &[u8], dst: &mut [u8], c: Component, method: DeinterlaceMethod) {
    match method {
        DeinterlaceMethod::Linear => {
            // Odd rows (the bottom field) are rebuilt from their neighbours; the
            // last row has no row below, so it stays.
            let mut y = 1;
            while y + 1 < c.rows {
                for x in 0..c.width {
                    dst[c.at(y, x)] = avg(src[c.at(y - 1, x)], src[c.at(y + 1, x)]);
                }
                y += 2;
            }
        }
        DeinterlaceMethod::Blend => {
            for y in 0..c.rows.saturating_sub(1) {
                for x in 0..c.width {
                    dst[c.at(y, x)] = avg(src[c.at(y, x)], src[c.at(y + 1, x)]);
                }
            }
        }
        DeinterlaceMethod::Yadif => unreachable!("yadif runs on the held window"),
    }
}

/// yadif over one component, single-rate and top-field-first, a port of ffmpeg's
/// `vf_yadif.c` `FILTER` / `CHECK` kernel.
///
/// Single-rate is ffmpeg's `mode=0`, which always passes `parity ^ tff == 1` down
/// to the line filter, so the temporal pair is `(prev, cur)`: the two samples of
/// the discarded field that bracket the kept field in time. `dst` already holds a
/// copy of `cur`, so the kept field's rows need no write.
///
/// The first and last three columns take the plain `(above + below) / 2` spatial
/// predictor instead of the edge-directed search, exactly as ffmpeg's
/// `filter_edges` does, because the search reads three samples to either side.
fn yadif_component(prev: &[u8], cur: &[u8], next: &[u8], dst: &mut [u8], c: Component) {
    // Two rows are the minimum the row mirroring below is defined for.
    if c.rows < 2 {
        return;
    }
    // Top field first: only the odd rows are rebuilt, and `y >= 1` is what makes
    // every neighbour index below non-negative.
    for y in (1..c.rows).step_by(2) {
        // ffmpeg mirrors at the bottom edge: prefs is -1 row on the last row.
        let above = y - 1;
        let below = if y + 1 < c.rows { y + 1 } else { y - 1 };
        // ffmpeg forces mode 2 on the rows whose second-order neighbours would
        // fall outside the frame, which drops the b / f interval check. That is
        // also what keeps `above2` / `below2` inside the component.
        let interval = y != 1 && y + 2 != c.rows;
        let (above2, below2) = (y.saturating_sub(2), if below > y { y + 2 } else { y - 2 });
        for x in 0..c.width {
            let s = |buf: &[u8], yy: usize, xx: usize| buf[c.at(yy, xx)] as i32;
            let cc = s(cur, above, x);
            let e = s(cur, below, x);
            // Single-rate yadif reads its temporal pair from prev and cur: the
            // two samples of the discarded field bracketing this field in time.
            let (p0, n0) = (s(prev, y, x), s(cur, y, x));
            let d = (p0 + n0) >> 1;
            let td0 = (p0 - n0).abs();
            let td1 = ((s(prev, above, x) - cc).abs() + (s(prev, below, x) - e).abs()) >> 1;
            let td2 = ((s(next, above, x) - cc).abs() + (s(next, below, x) - e).abs()) >> 1;
            let mut diff = (td0 >> 1).max(td1).max(td2);
            let mut pred = (cc + e) >> 1;

            if x >= 3 && x + 3 < c.width {
                let xi = x as isize;
                let sx = |yy: usize, xx: isize| cur[c.at(yy, xx as usize)] as i32;
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

            if interval {
                let b = (s(prev, above2, x) + s(cur, above2, x)) >> 1;
                let f = (s(prev, below2, x) + s(cur, below2, x)) >> 1;
                let max = (d - e).max(d - cc).max((b - cc).min(f - e));
                let min = (d - e).min(d - cc).min((b - cc).max(f - e));
                diff = diff.max(min).max(-max);
            }

            if pred > d + diff {
                pred = d + diff;
            } else if pred < d - diff {
                pred = d - diff;
            }
            dst[c.at(y, x)] = pred.clamp(0, 255) as u8;
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

    fn run(src: &[u8], w: usize, h: usize, method: DeinterlaceMethod) -> Vec<u8> {
        let mut dst = src.to_vec();
        for c in components(RawVideoFormat::Rgba8, w, h) {
            blend_component(src, &mut dst, c, method);
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
        yadif_component(&src, &src, &src, &mut dst, rgba(w, h));
        assert_eq!(dst, src, "a frame identical to its neighbours is unchanged");
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
