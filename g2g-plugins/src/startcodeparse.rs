//! Shared access-unit parser core for the start-code elementary streams that
//! are not NAL streams: MPEG-1 / MPEG-2 video, MPEG-4 Part 2, and VC-1
//! advanced profile.
//!
//! All three frame the same way: `00 00 01 xx` start codes delimit units, one
//! coded picture plus the headers that lead it is one access unit, and the
//! geometry lives in a sequence header that may sit in an earlier buffer than
//! the picture it describes. `StartCodeParse<C>` holds that machinery
//! (accumulate, split at access-unit boundaries, refine caps, stamp the
//! keyframe flag, re-insert cached configuration headers); a [`StartCodeCodec`]
//! marker supplies the per-codec start-code classification, geometry parse and
//! keyframe rule.
//!
//! `Caps::CompressedVideo` carries no pixel-aspect field, so the aspect ratio
//! these headers signal is surfaced as the read-only `pixel-aspect-ratio`
//! property instead of in caps.

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

/// Upper bound on bytes buffered while waiting for an access-unit boundary. A
/// real stream emits start codes frequently, so this only guards against an
/// unbounded accumulator on non-conforming input: past it, the pending bytes go
/// out as one access unit rather than growing without limit.
const MAX_ACCUM_BYTES: usize = 16 * 1024 * 1024;

/// Widest `config-interval` accepted, in seconds.
const MAX_CONFIG_INTERVAL_SECONDS: i64 = 3600;

/// Name of [`CONFIG_INTERVAL_PROPERTY`], matched in `set_property`.
const CONFIG_INTERVAL_NAME: &str = "config-interval";

/// Name of [`PIXEL_ASPECT_PROPERTY`], matched in `get_property`.
const PIXEL_ASPECT_NAME: &str = "pixel-aspect-ratio";

/// The `config-interval` property, for the codecs whose configuration headers
/// can be re-sent periodically.
pub(crate) const CONFIG_INTERVAL_PROPERTY: PropertySpec = PropertySpec::new(
    CONFIG_INTERVAL_NAME,
    g2g_core::PropKind::Int,
    "configuration-header re-insertion interval in seconds (0 = off, -1 = every keyframe, N = every N s)",
)
.with_range("-1", "3600")
.with_default("0");

/// The read-only pixel aspect ratio recovered from the sequence header, as a
/// `width/height` fraction of one sample. `0/1` until a header has been parsed.
pub(crate) const PIXEL_ASPECT_PROPERTY: PropertySpec = PropertySpec::new(
    PIXEL_ASPECT_NAME,
    g2g_core::PropKind::Fraction,
    "sample aspect ratio read from the sequence header (0/1 = not signalled yet)",
)
.read_only();

/// What one start-code unit does to the access unit being assembled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StartCodeRole {
    /// A header that leads a picture (sequence header, GOP / entry point). It
    /// opens a new access unit once the current one already holds a picture.
    Leads,
    /// The coded picture itself.
    Picture,
    /// Part of the picture in progress (a slice, an extension, user data).
    Continues,
}

/// Coded geometry recovered from a sequence header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoGeometry {
    pub width: u32,
    pub height: u32,
    /// Framerate as Q16 fixed-point fps, `None` when the headers signal none.
    pub framerate: Option<u32>,
    /// One sample's `(width, height)`, `None` when the headers signal none.
    pub pixel_aspect: Option<(u32, u32)>,
}

/// Per-codec hooks for [`StartCodeParse`]. Implemented by zero-sized markers.
pub trait StartCodeCodec: Send + Sync + 'static {
    /// The caps codec tag this parser accepts and emits.
    const CODEC: VideoCodec;
    /// `ElementMetadata` long name.
    const NAME: &'static str;
    /// `ElementMetadata` description.
    const DESCRIPTION: &'static str;
    /// The element's runtime property specs.
    const PROPERTIES: &'static [PropertySpec];

    /// What the unit introduced by start code `code` does to the access unit
    /// being assembled.
    fn start_code_role(code: u8) -> StartCodeRole;

    /// Coded geometry from the sequence header in `au`, `None` when it carries
    /// none or the header does not parse.
    fn geometry(au: &[u8]) -> Option<VideoGeometry>;

    /// Whether `au` opens a decodable resume point.
    fn au_is_keyframe(au: &[u8]) -> bool;

    /// The configuration headers `au` carries, start codes included, ready to
    /// prepend to a later access unit that lacks them. `None` when the codec
    /// has no `config-interval` or the access unit carries no configuration.
    fn config_headers(_au: &[u8]) -> Option<Vec<u8>> {
        None
    }
}

/// The next start code at or after `from`, as `(prefix offset, code offset)`.
///
/// The MPEG-family prefix is exactly `00 00 01`. Zero bytes ahead of it belong
/// to the unit that precedes it, so unlike H.264 Annex-B this must not also
/// match a four-byte `00 00 00 01`: a sequence extension that ends in a zero
/// byte would lose it, and the extension is read to its last bit.
fn next_start_code(data: &[u8], from: usize) -> Option<(usize, usize)> {
    let mut i = from;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            return Some((i, i + 3));
        }
        i += 1;
    }
    None
}

/// Offsets in `data` at which a new access unit begins, per `classify`. The
/// first start code always opens the first access unit; after that a unit that
/// is not a continuation opens one as soon as the access unit in progress holds
/// a picture.
pub(crate) fn au_starts_by(data: &[u8], classify: impl Fn(u8) -> StartCodeRole) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut seen_picture = false;
    let mut search_from = 0;
    while let Some((start_code, payload)) = next_start_code(data, search_from) {
        // Resume past the code byte: a picture start code is 0x00, so a search
        // resuming on it could re-match the prefix it belongs to.
        search_from = payload + 1;
        let Some(&code) = data.get(payload) else {
            break;
        };
        let role = classify(code);
        let opens_au = starts.is_empty() || (seen_picture && role != StartCodeRole::Continues);
        if opens_au {
            starts.push(start_code);
            seen_picture = false;
        }
        if role == StartCodeRole::Picture {
            seen_picture = true;
        }
    }
    starts
}

/// Iterate the start-code units of `data` as `(code, payload)` pairs, the start
/// code prefix stripped and the payload running to the next start code.
pub(crate) fn start_code_units(data: &[u8]) -> impl Iterator<Item = (u8, &'_ [u8])> + '_ {
    let mut search_from = 0;
    core::iter::from_fn(move || {
        let (_, payload) = next_start_code(data, search_from)?;
        search_from = payload + 1;
        let &code = data.get(payload)?;
        let end = next_start_code(data, payload + 1).map_or(data.len(), |(prefix, _)| prefix);
        Some((code, &data[payload + 1..end.max(payload + 1)]))
    })
}

/// Offset of the start-code prefix of the first unit in `data` whose code
/// satisfies `is_wanted`.
pub(crate) fn first_start_code_offset(
    data: &[u8],
    is_wanted: impl Fn(u8) -> bool,
) -> Option<usize> {
    let mut search_from = 0;
    while let Some((prefix, payload)) = next_start_code(data, search_from) {
        search_from = payload + 1;
        if is_wanted(*data.get(payload)?) {
            return Some(prefix);
        }
    }
    None
}

/// Reduce a `(numerator, denominator)` pair, `None` if either side is zero (an
/// aspect ratio with a zero term is no aspect ratio).
pub(crate) fn reduce_ratio(numerator: u32, denominator: u32) -> Option<(u32, u32)> {
    if numerator == 0 || denominator == 0 {
        return None;
    }
    let (mut a, mut b) = (numerator, denominator);
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    Some((numerator / a, denominator / a))
}

/// One sample's `(width, height)` for an aspect-ratio code. Codes 1..=5 are
/// ISO/IEC 14496-2 Table 6-12 (MPEG-4 Part 2); SMPTE 421M (VC-1) keeps those and
/// defines 6..=13 with the same values as ITU-T H.264 Table E-1. Code 0 is
/// unspecified, 14 reserved, and 15 a custom pair coded in the header, so all
/// three read back as no aspect ratio.
const SAMPLE_ASPECT_BY_CODE: [(u32, u32); 16] = [
    (0, 0),
    (1, 1),
    (12, 11),
    (10, 11),
    (16, 11),
    (40, 33),
    (24, 11),
    (20, 11),
    (32, 11),
    (80, 33),
    (18, 11),
    (15, 11),
    (64, 33),
    (160, 99),
    (0, 0),
    (0, 0),
];

/// Look up [`SAMPLE_ASPECT_BY_CODE`], rejecting a code above `highest_defined`
/// (the codecs share the low codes but define different numbers of them).
pub(crate) fn sample_aspect(code: u32, highest_defined: u32) -> Option<(u32, u32)> {
    if code > highest_defined {
        return None;
    }
    let (w, h) = *SAMPLE_ASPECT_BY_CODE.get(code as usize)?;
    reduce_ratio(w, h)
}

/// Access-unit parser for a start-code elementary stream. See the module docs.
pub struct StartCodeParse<C: StartCodeCodec> {
    configured: bool,
    last_emitted_caps: Option<Caps>,
    caps_changes: u64,
    /// Bytes received but not yet emitted as a complete access unit: the
    /// trailing, possibly-incomplete access unit is held until the next one's
    /// start code arrives.
    accum: Vec<u8>,
    /// Timing to stamp the access unit at the head of `accum`, captured when its
    /// first byte arrived.
    au_timing: FrameTiming,
    seq: u64,
    /// Configuration-header re-insertion interval in seconds: `0` off, `-1`
    /// every keyframe, `N` once `N` seconds have elapsed. Only meaningful for a
    /// codec whose [`StartCodeCodec::config_headers`] returns headers.
    config_interval: i32,
    /// Last configuration headers seen, re-inserted before a keyframe that
    /// lacks them.
    cached_config: Vec<u8>,
    /// PTS (ns) of the last access unit the headers were inserted before.
    last_config_pts_ns: Option<u64>,
    /// Sample aspect ratio from the last sequence header, for the read-only
    /// `pixel-aspect-ratio` property (caps carry no field for it).
    pixel_aspect: Option<(u32, u32)>,
    _codec: PhantomData<C>,
}

impl<C: StartCodeCodec> Default for StartCodeParse<C> {
    fn default() -> Self {
        Self {
            configured: false,
            last_emitted_caps: None,
            caps_changes: 0,
            accum: Vec::new(),
            au_timing: FrameTiming::default(),
            seq: 0,
            config_interval: 0,
            cached_config: Vec::new(),
            last_config_pts_ns: None,
            pixel_aspect: None,
            _codec: PhantomData,
        }
    }
}

impl<C: StartCodeCodec> core::fmt::Debug for StartCodeParse<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StartCodeParse")
            .field("codec", &C::CODEC)
            .field("configured", &self.configured)
            .field("config_interval", &self.config_interval)
            .finish_non_exhaustive()
    }
}

impl<C: StartCodeCodec> StartCodeParse<C> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Count of `CapsChanged` packets pushed downstream, for tests asserting
    /// re-emission is suppressed when the geometry is unchanged.
    pub fn caps_changes_emitted(&self) -> u64 {
        self.caps_changes
    }

    /// The sample aspect ratio the last sequence header signalled.
    pub fn pixel_aspect(&self) -> Option<(u32, u32)> {
        self.pixel_aspect
    }

    /// Set the configuration-header re-insertion interval in seconds (`0` off,
    /// `-1` every keyframe, `N` every N seconds).
    pub fn with_config_interval(mut self, seconds: i32) -> Self {
        self.config_interval = seconds;
        self
    }

    /// Whether this codec declares the `config-interval` property, so a direct
    /// `set_property` cannot reach a knob `gst-inspect` does not show.
    fn declares_config_interval() -> bool {
        C::PROPERTIES.iter().any(|s| s.name == CONFIG_INTERVAL_NAME)
    }

    /// Refresh the cached configuration headers from `au` and, when
    /// re-insertion is due, return `au` with them prepended.
    pub(crate) fn apply_config_interval(
        &mut self,
        au: Vec<u8>,
        pts_ns: u64,
        keyframe: bool,
    ) -> Vec<u8> {
        let carries_config = match C::config_headers(&au) {
            Some(headers) => {
                self.cached_config = headers;
                true
            }
            None => false,
        };
        if self.config_interval == 0 || !keyframe {
            return au;
        }
        // Already configured: nothing to add, but it resets the clock.
        if carries_config {
            self.last_config_pts_ns = Some(pts_ns);
            return au;
        }
        let due = if self.config_interval < 0 {
            true
        } else {
            let interval_ns = (self.config_interval as u64).saturating_mul(1_000_000_000);
            match self.last_config_pts_ns {
                None => true,
                Some(last) => pts_ns.saturating_sub(last) >= interval_ns,
            }
        };
        if !due || self.cached_config.is_empty() {
            return au;
        }
        let mut out = Vec::with_capacity(self.cached_config.len() + au.len());
        out.extend_from_slice(&self.cached_config);
        out.extend_from_slice(&au);
        self.last_config_pts_ns = Some(pts_ns);
        out
    }

    /// Refine caps from the sequence header in `bytes`, suppressing an
    /// unchanged re-emit.
    async fn refine_caps(
        &mut self,
        bytes: &[u8],
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let Some(info) = C::geometry(bytes) else {
            return Ok(());
        };
        self.pixel_aspect = info.pixel_aspect;
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
            self.caps_changes += 1;
        }
        Ok(())
    }

    /// Accumulate one input buffer and emit every access unit whose end is now
    /// known. The trailing access unit stays buffered until the next call or
    /// `Eos`. A non-`System` domain passes through unchanged.
    async fn accumulate(&mut self, frame: Frame, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let Some(bytes) = frame.domain.as_system_slice() else {
            out.push(PipelinePacket::DataFrame(frame)).await?;
            return Ok(());
        };
        if self.accum.is_empty() {
            self.au_timing = frame.timing;
        }
        self.accum.extend_from_slice(bytes);

        if self.accum.len() > MAX_ACCUM_BYTES {
            let au = core::mem::take(&mut self.accum);
            let timing = self.au_timing;
            return self.emit_au(au, timing, out).await;
        }

        let starts = au_starts_by(&self.accum, C::start_code_role);
        if starts.len() < 2 {
            return Ok(()); // at most one still-open access unit buffered
        }
        let frame_timing = frame.timing;
        let tail = starts[starts.len() - 1];
        let done = self.accum[..tail].to_vec();
        self.accum.drain(..tail);
        for pair in starts.windows(2) {
            let (lo, hi) = (pair[0], pair[1]);
            // The head access unit carries the timing captured when it began;
            // one that both begins and ends inside this buffer takes this
            // buffer's timing.
            let timing = if lo == 0 {
                self.au_timing
            } else {
                frame_timing
            };
            self.emit_au(done[lo..hi].to_vec(), timing, out).await?;
        }
        self.au_timing = frame_timing;
        Ok(())
    }

    /// Emit one access unit: refine caps from any sequence header it carries,
    /// stamp the keyframe flag, and re-insert configuration headers when the
    /// interval calls for it.
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

    /// The caps this parser accepts and emits: its codec at any geometry.
    fn any_caps() -> Caps {
        Caps::CompressedVideo {
            codec: C::CODEC,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }
}

impl<C: StartCodeCodec> AsyncElement for StartCodeParse<C> {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(C::NAME, "Codec/Parser/Video", C::DESCRIPTION, "g2g")
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::any_caps())
    }

    /// Pass-through identity over the codec at any geometry: the parser refines
    /// geometry mid-stream but never changes media type.
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

    fn properties(&self) -> &'static [PropertySpec] {
        C::PROPERTIES
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            CONFIG_INTERVAL_NAME if Self::declares_config_interval() => {
                let seconds = value.as_int().ok_or(PropError::Type)?;
                if !(-1..=MAX_CONFIG_INTERVAL_SECONDS).contains(&seconds) {
                    return Err(PropError::Value);
                }
                self.config_interval = seconds as i32;
                Ok(())
            }
            PIXEL_ASPECT_NAME => Err(PropError::ReadOnly),
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            CONFIG_INTERVAL_NAME if Self::declares_config_interval() => {
                Some(PropValue::Int(self.config_interval as i64))
            }
            PIXEL_ASPECT_NAME => {
                let (w, h) = self.pixel_aspect.unwrap_or((0, 1));
                Some(PropValue::Fraction(w as i32, h as i32))
            }
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
                PipelinePacket::DataFrame(frame) => self.accumulate(frame, out).await?,
                PipelinePacket::CapsChanged(c) => {
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                PipelinePacket::Flush => {
                    // A seek discontinuity: drop the partial access unit rather
                    // than splice pre-seek bytes onto the post-seek stream.
                    self.accum.clear();
                    self.last_emitted_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Eos => {
                    if !self.accum.is_empty() {
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

impl<C: StartCodeCodec> PadTemplates for StartCodeParse<C> {
    fn pad_templates() -> Vec<PadTemplate> {
        let caps = Self::any_caps();
        Vec::from([
            PadTemplate::sink(CapsSet::one(caps.clone())),
            PadTemplate::source(CapsSet::one(caps)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn reduce_ratio_divides_by_the_common_factor() {
        assert_eq!(reduce_ratio(1920, 1080), Some((16, 9)));
        assert_eq!(reduce_ratio(10000, 6735), Some((2000, 1347)));
        assert_eq!(reduce_ratio(0, 3), None);
        assert_eq!(reduce_ratio(3, 0), None);
    }

    #[test]
    fn sample_aspect_rejects_codes_the_codec_does_not_define() {
        // MPEG-4 Part 2 defines 1..=5; VC-1 goes on to 13.
        assert_eq!(sample_aspect(5, 5), Some((40, 33)));
        assert_eq!(sample_aspect(6, 5), None, "reserved in MPEG-4 Part 2");
        assert_eq!(sample_aspect(6, 13), Some((24, 11)));
        assert_eq!(sample_aspect(0, 13), None, "unspecified");
        assert_eq!(sample_aspect(14, 13), None, "reserved in VC-1");
        assert_eq!(sample_aspect(15, 13), None, "custom, coded in the header");
    }

    #[test]
    fn start_code_units_splits_at_start_codes() {
        let data = vec![0, 0, 1, 0xB3, 0xAA, 0xBB, 0, 0, 1, 0x00, 0xCC];
        let units: Vec<(u8, Vec<u8>)> = start_code_units(&data)
            .map(|(code, payload)| (code, payload.to_vec()))
            .collect();
        assert_eq!(units, vec![(0xB3, vec![0xAA, 0xBB]), (0x00, vec![0xCC])]);
    }

    #[test]
    fn au_starts_groups_leading_headers_with_their_picture() {
        // header, picture, slice | header, picture -> two access units.
        let classify = |code: u8| match code {
            0xB3 => StartCodeRole::Leads,
            0x00 => StartCodeRole::Picture,
            _ => StartCodeRole::Continues,
        };
        let mut data = vec![
            0u8, 0, 1, 0xB3, 0x11, 0, 0, 1, 0x00, 0x22, 0, 0, 1, 0x01, 0x33,
        ];
        let second = data.len();
        data.extend_from_slice(&[0, 0, 1, 0xB3, 0x44, 0, 0, 1, 0x00, 0x55]);
        assert_eq!(au_starts_by(&data, classify), vec![0, second]);
    }
}
