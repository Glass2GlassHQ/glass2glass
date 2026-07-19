//! Shared access-unit parser core for the H.264 and H.265 elementary-stream
//! parsers.
//!
//! `H264Parse` and `H265Parse` do the same job: scan each `DataFrame` for an SPS,
//! refine caps to the coded geometry (suppressing an unchanged re-emit), and
//! optionally re-frame the bitstream to one access unit per `DataFrame` while
//! re-inserting parameter sets on a `config-interval`. They differ only in
//! codec-specific pieces: the NAL classification, the access-unit boundary rules,
//! the SPS geometry parse, and which parameter sets to cache and re-insert.
//!
//! `NalParse<C>` holds the shared machinery; a [`NalCodec`] marker (`H264Codec` /
//! `H265Codec`) supplies the codec-specific hooks. `H264Parse` / `H265Parse` are
//! type aliases over it, so the public surface (constructors, `AsyncElement`,
//! `PadTemplates`) is unchanged.

use core::future::Future;
use core::marker::PhantomData;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropValue, PropertySpec,
    Rate, VideoCodec,
};

use crate::annexb::nal_units;

/// Upper bound on bytes buffered while waiting for an access-unit boundary in
/// re-framing mode. A real stream emits start codes frequently, so this is only a
/// guard against an unbounded accumulator on pathological / non-conforming input:
/// past it, the pending bytes are flushed as one access unit rather than grown
/// without limit. A single intra access unit at 4K stays well under it.
const MAX_REFRAME_BYTES: usize = 16 * 1024 * 1024;

/// Coded-picture geometry recovered from an SPS (post conformance-window
/// cropping), plus the framerate when the codec recovers one.
#[derive(Debug)]
pub struct SpsGeometry {
    pub width: u32,
    pub height: u32,
    /// Framerate as Q16 fixed-point fps (e.g. from the H.264 VUI `timing_info`),
    /// `None` when the SPS carries none or the codec does not recover it.
    pub framerate: Option<u32>,
}

/// Codec-specific hooks for [`NalParse`]. Implemented by zero-sized markers.
pub trait NalCodec: Send + Sync + 'static {
    /// The caps codec tag this parser accepts and emits.
    const CODEC: VideoCodec;
    /// `ElementMetadata` long name.
    const NAME: &'static str;
    /// `ElementMetadata` description.
    const DESCRIPTION: &'static str;
    /// The element's runtime property specs (the `config-interval` knob; its
    /// help text names the codec's parameter sets, so it varies per codec).
    const PROPERTIES: &'static [PropertySpec];
    /// Parameter-set NAL types to cache and re-insert, in prepend order (H.264:
    /// SPS then PPS; H.265: VPS then SPS then PPS).
    const PARAM_SET_TYPES: &'static [u8];
    /// The NAL type whose presence in an access unit means it already carries
    /// config (the SPS): it resets the re-insertion clock and suppresses
    /// prefixing. Must appear in [`PARAM_SET_TYPES`](Self::PARAM_SET_TYPES).
    const SPS_TYPE: u8;

    /// NAL unit type of a NAL body (its header parsed), or `None` if too short.
    fn nal_type(nal: &[u8]) -> Option<u8>;
    /// Start-code offsets in an Annex-B buffer at which a new access unit begins.
    fn au_starts(data: &[u8]) -> Vec<usize>;
    /// Whether an access unit opens a keyframe (IDR / IRAP).
    fn au_is_keyframe(au: &[u8]) -> bool;
    /// Recover SPS geometry (and optional framerate) from an access unit.
    fn extract_sps_info(au: &[u8]) -> Option<SpsGeometry>;
}

/// Access-unit parser generic over a [`NalCodec`]. See the module docs.
pub struct NalParse<C: NalCodec> {
    configured: bool,
    last_emitted_caps: Option<Caps>,
    sps_emitted: u64,
    /// Re-framing mode: re-chunk the input into access-unit-aligned Annex-B
    /// buffers (one coded picture per `DataFrame`) rather than passing buffers
    /// through unchanged. A decoder fed un-aligned units (e.g. one MPEG-TS PES
    /// that is not one access unit) mis-parses slice boundaries; auto-plugged
    /// decode chains use this so the decoder always sees one access unit per
    /// packet, matching GStreamer's `h264parse` / `h265parse`. Off in the default
    /// (caps-refinement-only) construction.
    reframe: bool,
    /// Re-framing accumulator: Annex-B bytes received but not yet emitted as a
    /// complete access unit (the trailing, possibly-incomplete AU is held until
    /// the next AU's start code arrives). Empty outside re-framing mode.
    accum: Vec<u8>,
    /// Timing to stamp the access unit currently at the head of `accum` (captured
    /// when that AU's first byte arrived). Re-framing only.
    au_timing: FrameTiming,
    /// Monotonic sequence number for emitted re-framed access units.
    seq: u64,
    /// Input framing, latched from the first frame (`Some(true)` = Annex-B start
    /// codes, `Some(false)` = length prefixes). Fixed for a stream, so decided
    /// once: a per-frame guess misclassifies a mid-AU Annex-B continuation buffer
    /// (no leading start code) as length-prefixed and mangles it. Re-framing only.
    input_is_annexb: Option<bool>,
    /// Parameter-set re-insertion interval (the gst `config-interval`), in
    /// seconds: `0` = never (default), `-1` = prepend the cached parameter sets to
    /// every keyframe AU, `N > 0` = prepend at the first keyframe once `N` seconds
    /// elapsed. Applied on the re-framing (access-unit-aligned Annex-B) output.
    config_interval: i32,
    /// Cached parameter-set NAL bodies (start codes stripped), one bucket per
    /// [`NalCodec::PARAM_SET_TYPES`] entry in the same order, refreshed as they
    /// flow so a keyframe lacking inline parameter sets can be prefixed.
    cached: Vec<Vec<Vec<u8>>>,
    /// PTS (ns) of the last AU the parameter sets were (re-)inserted before, for
    /// the `config_interval > 0` time-based cadence.
    last_config_pts_ns: Option<u64>,
    _codec: PhantomData<C>,
}

impl<C: NalCodec> Default for NalParse<C> {
    fn default() -> Self {
        Self {
            configured: false,
            last_emitted_caps: None,
            sps_emitted: 0,
            reframe: false,
            accum: Vec::new(),
            au_timing: FrameTiming::default(),
            seq: 0,
            input_is_annexb: None,
            config_interval: 0,
            cached: C::PARAM_SET_TYPES.iter().map(|_| Vec::new()).collect(),
            last_config_pts_ns: None,
            _codec: PhantomData,
        }
    }
}

impl<C: NalCodec> core::fmt::Debug for NalParse<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NalParse")
            .field("codec", &C::CODEC)
            .field("configured", &self.configured)
            .field("reframe", &self.reframe)
            .field("config_interval", &self.config_interval)
            .finish_non_exhaustive()
    }
}

impl<C: NalCodec> NalParse<C> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct the parser in access-unit re-framing mode: it accumulates the
    /// input bitstream and emits one access-unit-aligned Annex-B `DataFrame` per
    /// coded picture (see [`reframe`](Self::reframe)).
    pub fn reframing() -> Self {
        Self {
            reframe: true,
            ..Self::default()
        }
    }

    /// Count of `CapsChanged` packets this element has pushed downstream. Useful
    /// for tests asserting re-emission is suppressed when the SPS geometry is
    /// unchanged.
    pub fn caps_changes_emitted(&self) -> u64 {
        self.sps_emitted
    }

    /// Set the parameter-set re-insertion interval in seconds (`0` off, `-1` every
    /// keyframe, `N` every N seconds); see [`config_interval`](Self::config_interval).
    pub fn with_config_interval(mut self, seconds: i32) -> Self {
        self.config_interval = seconds;
        self
    }

    /// Refresh the cached parameter sets from `au` and, when re-insertion is due
    /// for this access unit, return the AU prefixed with the cached parameter sets
    /// (as Annex-B). `au` must be Annex-B (the re-framing output always is).
    pub(crate) fn apply_config_interval(
        &mut self,
        au: Vec<u8>,
        pts_ns: u64,
        keyframe: bool,
    ) -> Vec<u8> {
        // Collect this AU's parameter sets, bucketed like `PARAM_SET_TYPES`, and
        // note whether it already carries config (leads with / contains an SPS).
        let mut collected: Vec<Vec<Vec<u8>>> =
            C::PARAM_SET_TYPES.iter().map(|_| Vec::new()).collect();
        for nal in nal_units(&au) {
            if let Some(t) = C::nal_type(nal) {
                if let Some(idx) = C::PARAM_SET_TYPES.iter().position(|&x| x == t) {
                    collected[idx].push(nal.to_vec());
                }
            }
        }
        let sps_idx = C::PARAM_SET_TYPES
            .iter()
            .position(|&x| x == C::SPS_TYPE)
            .expect("SPS_TYPE must appear in PARAM_SET_TYPES");
        let has_config = !collected[sps_idx].is_empty();
        for (i, bucket) in collected.into_iter().enumerate() {
            if !bucket.is_empty() {
                self.cached[i] = bucket;
            }
        }

        if self.config_interval == 0 || !keyframe {
            return au;
        }
        // The AU already carries config: nothing to add, but it resets the clock.
        if has_config {
            self.last_config_pts_ns = Some(pts_ns);
            return au;
        }
        let due = if self.config_interval < 0 {
            true // -1: every keyframe
        } else {
            let interval_ns = (self.config_interval as u64).saturating_mul(1_000_000_000);
            match self.last_config_pts_ns {
                None => true,
                Some(last) => pts_ns.saturating_sub(last) >= interval_ns,
            }
        };
        if !due || self.cached[sps_idx].is_empty() {
            return au;
        }
        let mut out = Vec::with_capacity(au.len() + 96);
        for ps in self.cached.iter().flatten() {
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(ps);
        }
        out.extend_from_slice(&au);
        self.last_config_pts_ns = Some(pts_ns);
        out
    }

    /// Refine caps from any SPS in `bytes` (suppressing an unchanged re-emit).
    async fn refine_caps(
        &mut self,
        bytes: &[u8],
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        if let Some(info) = C::extract_sps_info(bytes) {
            let new_caps = Caps::CompressedVideo {
                codec: C::CODEC,
                width: Dim::Fixed(info.width),
                height: Dim::Fixed(info.height),
                framerate: info.framerate.map_or(Rate::Any, Rate::Fixed),
            };
            if self.last_emitted_caps.as_ref() != Some(&new_caps) {
                out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                    .await?;
                self.last_emitted_caps = Some(new_caps);
                self.sps_emitted += 1;
            }
        }
        Ok(())
    }

    /// Re-framing path for one input `DataFrame`: normalize to Annex-B,
    /// accumulate, and emit every access unit whose end is now known (its
    /// successor's start code has arrived). The trailing, possibly-incomplete AU
    /// stays buffered until the next call or `Eos`. Non-`System` domains pass
    /// through unchanged (the byte re-framer only applies to host memory).
    async fn reframe_frame(
        &mut self,
        frame: Frame,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let MemoryDomain::System(slice) = &frame.domain else {
            out.push(PipelinePacket::DataFrame(frame)).await?;
            return Ok(());
        };
        // Normalize to Annex-B: a length-prefixed stream is converted, an Annex-B
        // one is appended as-is. The framing is latched from the first frame, since
        // a mid-AU Annex-B continuation buffer (no leading start code) would
        // otherwise be misread as length-prefixed.
        let bytes = slice.as_slice();
        let is_annexb = *self
            .input_is_annexb
            .get_or_insert_with(|| crate::annexb::is_annex_b(bytes));
        if self.accum.is_empty() {
            self.au_timing = frame.timing;
        }
        if is_annexb {
            self.accum.extend_from_slice(bytes);
        } else {
            self.accum
                .extend_from_slice(&crate::annexb::avcc_to_annexb(bytes));
        }

        // Guard against unbounded growth on non-conforming input: flush what we
        // have as one AU rather than buffering forever.
        if self.accum.len() > MAX_REFRAME_BYTES {
            let au = core::mem::take(&mut self.accum);
            let timing = self.au_timing;
            self.emit_au(au, timing, out).await?;
            return Ok(());
        }

        // Access-unit start offsets in the accumulator. Emit each complete AU
        // (everything before the last start), then retain the trailing AU.
        let starts = C::au_starts(&self.accum);
        if starts.len() < 2 {
            return Ok(()); // at most one (still-open) AU buffered so far
        }
        let frame_timing = frame.timing;
        let tail = starts[starts.len() - 1];
        // Split off the still-open tail, leaving the complete AUs in `done`.
        let done = self.accum[..tail].to_vec();
        self.accum.drain(..tail);
        for w in starts.windows(2) {
            let (lo, hi) = (w[0], w[1]);
            // The head AU carries the timing captured when it began; AUs that both
            // begin and end inside this buffer take this buffer's timing.
            let timing = if lo == 0 {
                self.au_timing
            } else {
                frame_timing
            };
            self.emit_au(done[lo..hi].to_vec(), timing, out).await?;
        }
        // The retained tail began within this buffer (its predecessor ended here).
        self.au_timing = frame_timing;
        Ok(())
    }

    /// Emit one access-unit-aligned Annex-B buffer as a `DataFrame`: refine caps
    /// from any SPS it carries (suppressing an unchanged re-emit) and stamp the
    /// keyframe flag, mirroring the pass-through path's per-frame work.
    async fn emit_au(
        &mut self,
        au: Vec<u8>,
        mut timing: FrameTiming,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        if au.is_empty() {
            return Ok(());
        }
        self.refine_caps(&au, out).await?;
        timing.keyframe = C::au_is_keyframe(&au);
        // Re-insert cached parameter sets before this AU when config-interval calls
        // for it (no-op at interval 0 or when the AU already carries them).
        let au = self.apply_config_interval(au, timing.pts_ns, timing.keyframe);
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
            timing,
            self.seq,
        );
        self.seq += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }

    /// The `CompressedVideo` caps at any geometry that this parser accepts and
    /// emits (it refines geometry mid-stream from the SPS but never changes media
    /// type).
    fn any_caps() -> Caps {
        Caps::CompressedVideo {
            codec: C::CODEC,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }
}

impl<C: NalCodec> AsyncElement for NalParse<C> {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // The parser consumes its codec at any geometry; intersecting against that
        // narrows the proposal and rejects other codecs.
        upstream_caps.intersect(&Self::any_caps())
    }

    /// Pass-through identity over the codec at any geometry. With a fully-native
    /// chain the solver couples input and output links and rejects a mismatched
    /// upstream at negotiation time instead of via the dynamic `intercept_caps`.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(CapsSet::one(Self::any_caps()))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::CompressedVideo { codec, .. } if *codec == C::CODEC => {
                self.configured = true;
                Ok(ConfigureOutcome::Accepted)
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(C::NAME, "Codec/Parser/Video", C::DESCRIPTION, "g2g")
    }

    fn properties(&self) -> &'static [PropertySpec] {
        C::PROPERTIES
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "config-interval" => {
                let secs = value.as_int().ok_or(PropError::Type)?;
                if !(-1..=3600).contains(&secs) {
                    return Err(PropError::Value);
                }
                self.config_interval = secs as i32;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "config-interval" => Some(PropValue::Int(self.config_interval as i64)),
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
            match packet {
                PipelinePacket::DataFrame(mut frame) => {
                    if self.reframe {
                        return self.reframe_frame(frame, out).await;
                    }
                    if let MemoryDomain::System(slice) = &frame.domain {
                        // Surface the keyframe flag for trick-mode / keyframe seek
                        // (the parser is the producer that can detect it).
                        let is_keyframe = C::au_is_keyframe(slice.as_slice());
                        self.refine_caps(slice.as_slice(), out).await?;
                        frame.timing.keyframe = is_keyframe;
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                PipelinePacket::Flush => {
                    // A seek discontinuity: drop the partial AU rather than splice
                    // pre-seek bytes onto the post-seek stream. Reset SPS tracking
                    // so caps re-emit after the seek.
                    self.accum.clear();
                    self.last_emitted_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                // Segment is control: forward unchanged.
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Eos => {
                    // Flush the final buffered access unit at end of stream, else
                    // the last coded picture would never reach the decoder.
                    if self.reframe && !self.accum.is_empty() {
                        let au = core::mem::take(&mut self.accum);
                        let timing = self.au_timing;
                        self.emit_au(au, timing, out).await?;
                    }
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl<C: NalCodec> PadTemplates for NalParse<C> {
    fn pad_templates() -> Vec<PadTemplate> {
        let caps = Self::any_caps();
        Vec::from([
            PadTemplate::sink(CapsSet::one(caps.clone())),
            PadTemplate::source(CapsSet::one(caps)),
        ])
    }
}
