//! Subtitle text overlay (M171): renders timed [`Cue`] text onto an RGBA8 frame
//! at the bottom centre, selecting the active cue by the frame's PTS. The
//! `textoverlay` / `subtitleoverlay` analog and the visible end of an
//! SRT / WebVTT subtitle path.
//!
//! CPU, `no_std` baseline like [`AnalyticsOverlay`](crate::analyticsoverlay): in
//! and out are both RGBA8 at the negotiated geometry (put a `VideoConvert`
//! upstream of a non-RGBA source), the pixels pass through untouched except for
//! the painted text. Cues are held in memory; build them programmatically
//! ([`TextOverlay::from_srt`] / [`from_webvtt`](TextOverlay::from_webvtt)) or, on
//! `std`, set the `location=` property to a `.srt` / `.vtt` file (the
//! `gst-launch` path). Text is drawn with the embedded 8x8 [`bitmapfont`], scaled
//! to the frame height, over a translucent backing box for legibility; the
//! all-caps bitmap font is the baseline, with mixed-case TrueType deferred to a
//! `vello` GPU backend (M172).
//!
//! [`bitmapfont`]: crate::bitmapfont

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    MemoryDomain, MultiInputElement, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec, RawVideoFormat, Rate, TextFormat,
};

use crate::bitmapfont::{glyph, GLYPH_ADVANCE, GLYPH_HEIGHT};
use crate::paint::blend_px;
use crate::subparse::{parse_srt, parse_ssa, parse_ttml, parse_webvtt, Cue, TextAlign};
// Only the std `load_location` path sniffs the format; gate the import to match.
#[cfg(feature = "std")]
use crate::subparse::parse_auto;

/// Renders the active subtitle cue's text onto an RGBA8 frame. Cue selection is
/// by the frame's `pts_ns`; a frame with no covering cue passes through
/// untouched.
#[derive(Debug)]
pub struct TextOverlay {
    width: u32,
    height: u32,
    configured: bool,
    /// Cues in file order. Selection is a linear scan for the first cue covering
    /// the frame PTS (subtitle tracks are small and rarely overlap).
    cues: Vec<Cue>,
    /// Opaque RGBA text colour (default white).
    text_color: [u8; 4],
    /// Translucent RGBA backing-box colour (default ~62% black).
    bg_color: [u8; 4],
    /// The `location=` path, retained for `get_property` round-trips.
    location: Option<String>,
    drawn: u64,
}

impl Default for TextOverlay {
    fn default() -> Self {
        Self::new()
    }
}

impl TextOverlay {
    /// An overlay with no cues, white text on a translucent black box. Geometry
    /// is set at negotiation; cues are added via the builders or `location=`.
    pub fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            configured: false,
            cues: Vec::new(),
            text_color: [0xFF, 0xFF, 0xFF, 0xFF],
            bg_color: [0x00, 0x00, 0x00, 0xA0],
            location: None,
            drawn: 0,
        }
    }

    /// Use a preparsed cue list.
    pub fn with_cues(mut self, cues: Vec<Cue>) -> Self {
        self.cues = cues;
        self
    }

    /// Append one cue to the live list (used by [`TextOverlayN`] as cues arrive on
    /// its text pad). Cues accumulate; selection stays a PTS-covering scan.
    pub fn push_cue(&mut self, cue: Cue) {
        self.cues.push(cue);
    }

    /// Drop all cues (a flush / seek on the text stream).
    pub fn clear_cues(&mut self) {
        self.cues.clear();
    }

    /// Parse SubRip (`.srt`) text into the cue list.
    pub fn from_srt(text: &str) -> Self {
        Self::new().with_cues(parse_srt(text))
    }

    /// Parse WebVTT (`.vtt`) text into the cue list.
    pub fn from_webvtt(text: &str) -> Self {
        Self::new().with_cues(parse_webvtt(text))
    }

    /// Parse SubStation Alpha / ASS (`.ssa` / `.ass`) text into the cue list.
    pub fn from_ssa(text: &str) -> Self {
        Self::new().with_cues(parse_ssa(text))
    }

    /// Parse TTML / DFXP (`.ttml` / `.dfxp`) text into the cue list.
    pub fn from_ttml(text: &str) -> Self {
        Self::new().with_cues(parse_ttml(text))
    }

    /// Set the opaque text colour (alpha forced opaque).
    pub fn with_text_color(mut self, rgb: [u8; 3]) -> Self {
        self.text_color = [rgb[0], rgb[1], rgb[2], 0xFF];
        self
    }

    /// Number of loaded cues.
    pub fn cue_count(&self) -> usize {
        self.cues.len()
    }

    /// Count of frames processed (whether or not a cue was active).
    pub fn drawn_count(&self) -> u64 {
        self.drawn
    }

    /// Every cue covering running time `t_ns`, in cue order. WebVTT (and SRT)
    /// allow overlapping cues to display at once, so all active cues are drawn,
    /// each at its own position rather than only the first.
    fn active(&self, t_ns: u64) -> Vec<&Cue> {
        self.cues.iter().filter(|c| c.covers(t_ns)).collect()
    }

    /// RGBA8 at fixed geometry, the only format this element draws on.
    fn dims(caps: &Caps) -> Option<(u32, u32)> {
        if let Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        } = caps
        {
            Some((*w, *h))
        } else {
            None
        }
    }

    /// Whether `caps` is RGBA8 (geometry may still be unfixed at negotiation).
    fn accepts(caps: &Caps) -> bool {
        matches!(caps, Caps::RawVideo { format: RawVideoFormat::Rgba8, .. })
    }

    /// Integer font scale: one source pixel per `scale` output pixels, derived
    /// from the frame height so text stays readable across resolutions (>= 1).
    fn scale(&self) -> u32 {
        (self.height / 240).max(1)
    }

    /// Draw every cue active at `t_ns` onto the RGBA8 `buf`, each honouring its
    /// WebVTT placement (`position` / `line` / `align`). Cues with an explicit
    /// `line` are placed absolutely; auto-`line` cues stack upward from the
    /// bottom, in cue order, so overlapping subtitles don't collide.
    fn render_active(&self, buf: &mut [u8], t_ns: u64) {
        let w = self.width as i32;
        let h = self.height as i32;
        let scale = self.scale() as i32;
        let cell_w = GLYPH_ADVANCE as i32 * scale;
        let glyph_h = GLYPH_HEIGHT as i32 * scale;
        let line_gap = 2 * scale;
        let line_h = glyph_h + line_gap;
        let margin = 4 * scale;
        let pad = 2 * scale;

        // The bottom edge (above padding) available to the next auto-line cue.
        let mut auto_bottom = h - margin;

        for cue in self.active(t_ns) {
            let lines: Vec<&str> = cue.text.lines().collect();
            if lines.is_empty() {
                continue;
            }
            let block_h = lines.len() as i32 * line_h - line_gap;
            let max_chars = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0) as i32;
            let block_w = max_chars * cell_w;
            let s = cue.settings;

            // Horizontal: `position` (% of width) is the anchor, default centre;
            // `align` decides how the box extends from it.
            let anchor_x = s.position.map(|p| p as i32 * w / 100).unwrap_or(w / 2);
            let block_left = align_left(s.align, anchor_x, block_w).clamp(0, (w - block_w).max(0));

            // Vertical: explicit `line` (% of height) places absolutely; auto
            // stacks from the bottom upward.
            let block_top = match s.line {
                Some(p) => (p as i32 * h / 100).clamp(margin, (h - margin - block_h).max(margin)),
                None => {
                    let top = (auto_bottom - block_h).max(margin);
                    auto_bottom = top - pad - line_gap;
                    top
                }
            };

            // Translucent backing box behind this cue's block.
            self.fill_rect(
                buf,
                block_left - pad,
                block_top - pad,
                block_w + 2 * pad,
                block_h + 2 * pad,
                self.bg_color,
            );

            // Each line, aligned within the block per `align`, then glyphs.
            for (row, line) in lines.iter().enumerate() {
                let line_w = line.chars().count() as i32 * cell_w;
                let x0 = match s.align {
                    TextAlign::Center => block_left + (block_w - line_w) / 2,
                    TextAlign::Start => block_left,
                    TextAlign::End => block_left + (block_w - line_w),
                };
                let y0 = block_top + row as i32 * line_h;
                let mut gx = x0;
                for c in line.chars() {
                    self.blit_glyph(buf, gx, y0, scale, glyph(c));
                    gx += cell_w;
                }
            }
        }
    }

    /// Blit one 8x8 glyph at output `(gx, gy)`, each set bit a `scale` x `scale`
    /// block of the text colour, clipped to the canvas.
    fn blit_glyph(&self, buf: &mut [u8], gx: i32, gy: i32, scale: i32, rows: [u8; 8]) {
        for (ry, bits) in rows.iter().enumerate() {
            if *bits == 0 {
                continue;
            }
            for col in 0..8i32 {
                if bits & (0x80 >> col) != 0 {
                    self.fill_rect(
                        buf,
                        gx + col * scale,
                        gy + ry as i32 * scale,
                        scale,
                        scale,
                        self.text_color,
                    );
                }
            }
        }
    }

    /// Source-over blend a filled rectangle, clipped to the canvas.
    fn fill_rect(&self, buf: &mut [u8], x: i32, y: i32, rw: i32, rh: i32, color: [u8; 4]) {
        let w = self.width as i32;
        let h = self.height as i32;
        for py in y..y + rh {
            if py < 0 || py >= h {
                continue;
            }
            for px in x..x + rw {
                if px < 0 || px >= w {
                    continue;
                }
                blend_px(buf, ((py * w + px) * 4) as usize, color, 255);
            }
        }
    }

    /// Load and parse a subtitle file, replacing the cue list. The format is
    /// chosen by extension (`.vtt` / `.srt` / `.ass` / `.ssa`), else sniffed from
    /// the content. `std`-only: file I/O needs the OS.
    #[cfg(feature = "std")]
    fn load_location(&mut self, path: &str) -> Result<(), PropError> {
        let data = std::fs::read_to_string(path).map_err(|_| PropError::Value)?;
        self.cues = if path.ends_with(".vtt") {
            parse_webvtt(&data)
        } else if path.ends_with(".srt") {
            parse_srt(&data)
        } else if path.ends_with(".ass") || path.ends_with(".ssa") {
            parse_ssa(&data)
        } else if path.ends_with(".ttml") || path.ends_with(".dfxp") {
            parse_ttml(&data)
        } else {
            parse_auto(&data)
        };
        self.location = Some(path.into());
        Ok(())
    }

    /// `no_std` stub: subtitle-file loading requires `std`. The registry / launch
    /// path that sets `location=` is itself `std`-only, so this is unreachable in
    /// practice; it keeps the element compiling on the baseline.
    #[cfg(not(feature = "std"))]
    fn load_location(&mut self, _path: &str) -> Result<(), PropError> {
        Err(PropError::Value)
    }
}

/// Left edge of a `block_w`-wide box whose `align` anchor sits at `anchor`:
/// centred boxes straddle the anchor, start/end boxes hang to its right/left.
fn align_left(align: TextAlign, anchor: i32, block_w: i32) -> i32 {
    match align {
        TextAlign::Center => anchor - block_w / 2,
        TextAlign::Start => anchor,
        TextAlign::End => anchor - block_w,
    }
}

impl PadTemplates for TextOverlay {
    fn pad_templates() -> Vec<PadTemplate> {
        // RGBA8 in and out at any geometry; identity on the pixels apart from the
        // painted text.
        let any = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        let set = CapsSet::one(any);
        Vec::from([PadTemplate::sink(set.clone()), PadTemplate::source(set)])
    }
}

impl AsyncElement for TextOverlay {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if Self::accepts(upstream_caps) {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        // Identity: pixels and geometry pass through; only text is painted.
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| {
            if Self::accepts(input) {
                CapsSet::one(input.clone())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (w, h) = Self::dims(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        self.width = w;
        self.height = h;
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
                    let t_ns = frame.timing.pts_ns;
                    // Draw only when a cue is showing; overlapping cues each get
                    // their own placement (see `render_active`).
                    if self.cues.iter().any(|c| c.covers(t_ns)) {
                        let MemoryDomain::System(slice) = &mut frame.domain else {
                            return Err(G2gError::UnsupportedDomain);
                        };
                        let need = self.width as usize * self.height as usize * 4;
                        let buf = slice.as_mut_slice();
                        if buf.len() < need {
                            return Err(G2gError::CapsMismatch);
                        }
                        self.render_active(&mut buf[..need], t_ns);
                    }
                    self.drawn += 1;
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(caps) => {
                    if let Some((w, h)) = Self::dims(&caps) {
                        self.width = w;
                        self.height = h;
                    }
                    out.push(PipelinePacket::CapsChanged(caps)).await?;
                }
                // The runner's transform arm forwards EOS; don't double it.
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        TEXTOVERLAY_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Text overlay",
            "Filter/Editor/Video",
            "Renders SRT / WebVTT subtitle cues over video by PTS",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" => {
                let path = value.as_str().ok_or(PropError::Type)?;
                self.load_location(path)
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => {
                Some(PropValue::Str(self.location.clone().unwrap_or_default()))
            }
            _ => None,
        }
    }
}

/// Two-input text overlay (M403): a video pad (`RawVideo{Rgba8}`) and a *text
/// stream* pad (`Caps::Text{Utf8}`), painting cues that arrive as a stream onto
/// the video, the `N`-pad sibling of [`TextOverlay`] (which loads cues from a
/// file). The `subtitleoverlay` analog: pair it with [`SubParse`](crate::subparse)
/// to overlay a demuxed / network subtitle track, e.g.
/// `file ! subparse ! textoverlayn.text  videosrc ! textoverlayn.video ! sink`.
///
/// A [`MultiInputElement`] (video + text in, video out) that opts into
/// `input_pts_ordered`, so the runner merges the two pads by PTS: every cue
/// (PTS = its start time) is delivered before the video frame it first covers,
/// giving correct A/V-text alignment. Because [`SubParse`] batch-emits its cues
/// at end of the (small, fast) subtitle stream, the merge buffers video only
/// until that stream ends; incremental cue streaming is a follow-up. The
/// rendering is reused wholesale from [`TextOverlay`] (composition); the text pad
/// only feeds it cues. Cue positioning is the renderer default (bottom-centre):
/// `SubParse` emits plain text, so WebVTT / SSA placement does not survive the
/// `Text{Utf8}` stream (carrying it as text frame-meta is a follow-up).
#[derive(Debug, Default)]
pub struct TextOverlayN {
    /// Owns the cue list + geometry + rendering.
    inner: TextOverlay,
    /// The negotiated video caps, captured at `configure(VIDEO)`; the merged
    /// output (it `output_follows_input` the video pad).
    video_caps: Option<Caps>,
}

impl TextOverlayN {
    /// Input pad indices: video on 0, the text stream on 1.
    const VIDEO: usize = 0;
    const TEXT: usize = 1;

    /// A streamed-subtitle overlay. The output caps follow the video pad
    /// (`output_follows_input`), so no output geometry need be supplied: the
    /// solver derives it from whatever RGBA8 the video source negotiates.
    pub fn new() -> Self {
        Self { inner: TextOverlay::new(), video_caps: None }
    }

    /// Number of cues received on the text pad so far.
    pub fn cue_count(&self) -> usize {
        self.inner.cue_count()
    }

    /// Count of video frames processed.
    pub fn drawn_count(&self) -> u64 {
        self.inner.drawn_count()
    }
}

impl MultiInputElement for TextOverlayN {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        2
    }

    /// Merge the video and text pads by PTS, so a cue lands before the first
    /// video frame it covers (correct subtitle timing).
    fn input_pts_ordered(&self) -> bool {
        true
    }

    /// The merged output is the video pad's stream (identity passthrough with
    /// text painted on), so the solver derives the output caps from pad 0.
    fn output_follows_input(&self) -> Option<usize> {
        Some(Self::VIDEO)
    }

    fn intercept_caps(&self, input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match input {
            Self::VIDEO if TextOverlay::accepts(upstream_caps) => Ok(upstream_caps.clone()),
            Self::TEXT if matches!(upstream_caps, Caps::Text { format: TextFormat::Utf8 }) => {
                Ok(upstream_caps.clone())
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Video pad accepts RGBA8 at any geometry (the output follows it); the text
    /// pad accepts plain UTF-8. `Accepts` both, so the solver narrows each input
    /// edge (unlike a wildcard interleave).
    fn caps_constraint_as_input(&self, input: usize) -> CapsConstraint<'_> {
        match input {
            Self::TEXT => {
                CapsConstraint::Accepts(CapsSet::one(Caps::Text { format: TextFormat::Utf8 }))
            }
            // VIDEO (and any out-of-range pad, defensively): RGBA8, any geometry.
            _ => CapsConstraint::Accepts(CapsSet::one(Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
            })),
        }
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        match input {
            Self::VIDEO => {
                // Reuse the single-input overlay's geometry configuration; capture
                // the caps as the merged output (it follows this pad).
                self.inner.configure_pipeline(absolute_caps)?;
                self.video_caps = Some(absolute_caps.clone());
                Ok(ConfigureOutcome::Accepted)
            }
            Self::TEXT => match absolute_caps {
                Caps::Text { format: TextFormat::Utf8 } => Ok(ConfigureOutcome::Accepted),
                _ => Err(G2gError::CapsMismatch),
            },
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// The merged output is the video stream (RGBA8 at the negotiated geometry).
    /// Negotiation derives the output edge from the video pad (`output_follows_
    /// input`); this is the runtime mirror, valid once the video pad is configured.
    fn output_caps(&self) -> Result<Caps, G2gError> {
        self.video_caps.clone().ok_or(G2gError::NotConfigured)
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match input {
                // Video pad: render the active cues + forward, exactly the
                // single-input overlay's behaviour (it swallows Eos; the runner
                // emits the merged one).
                Self::VIDEO => self.inner.process(packet, out).await,
                // Text pad: turn each timed cue frame into a stored cue. Control
                // packets carry no cue; the text segment / caps don't govern the
                // video output, so they are not forwarded (the video pad's do).
                Self::TEXT => {
                    match packet {
                        PipelinePacket::DataFrame(frame) => {
                            if let MemoryDomain::System(slice) = &frame.domain {
                                let text =
                                    String::from_utf8_lossy(slice.as_slice()).into_owned();
                                let start = frame.timing.pts_ns;
                                let end = start.saturating_add(frame.timing.duration_ns);
                                self.inner.push_cue(Cue {
                                    start_ns: start,
                                    end_ns: end,
                                    text,
                                    settings: crate::subparse::CueSettings::default(),
                                });
                            }
                        }
                        // A flush / seek on the text stream drops pending cues.
                        PipelinePacket::Flush => self.inner.clear_cues(),
                        _ => {}
                    }
                    Ok(())
                }
                _ => Ok(()),
            }
        })
    }
}

/// `TextOverlay`'s settable properties (M171).
static TEXTOVERLAY_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "location",
    PropKind::Str,
    "path to an SRT (.srt) or WebVTT (.vtt) subtitle file; cues render by PTS",
)];

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use g2g_core::frame::Frame;
    use g2g_core::memory::SystemSlice;
    use g2g_core::{FrameTiming, PushOutcome};

    fn rgba_caps(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
        }
    }

    fn black(w: usize, h: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(w * h * 4);
        for _ in 0..w * h {
            v.extend_from_slice(&[0, 0, 0, 255]);
        }
        v
    }

    fn any_nonblack(buf: &[u8], w: usize, h: usize) -> bool {
        (0..w * h).any(|i| buf[i * 4] != 0 || buf[i * 4 + 1] != 0 || buf[i * 4 + 2] != 0)
    }

    fn frame_at(w: u32, h: u32, pts_ns: u64) -> Frame {
        Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(
                black(w as usize, h as usize).into_boxed_slice(),
            )),
            FrameTiming { pts_ns, ..FrameTiming::default() },
            0,
        )
    }

    #[derive(Default)]
    struct PixelSink {
        last: Option<Vec<u8>>,
    }
    impl OutputSink for PixelSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                if let PipelinePacket::DataFrame(frame) = packet {
                    if let MemoryDomain::System(slice) = &frame.domain {
                        self.last = Some(slice.as_slice().to_vec());
                    }
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    #[test]
    fn from_srt_loads_cues() {
        let ov = TextOverlay::from_srt(
            "1\n00:00:01,000 --> 00:00:04,000\nHELLO\n\n2\n00:00:05,000 --> 00:00:06,000\nBYE\n",
        );
        assert_eq!(ov.cue_count(), 2);
        assert_eq!(ov.active(2_000_000_000).iter().map(|c| c.text.as_str()).collect::<Vec<_>>(), ["HELLO"]);
        assert_eq!(ov.active(5_500_000_000).iter().map(|c| c.text.as_str()).collect::<Vec<_>>(), ["BYE"]);
        assert!(ov.active(10_000_000_000).is_empty());
    }

    #[test]
    fn overlapping_cues_are_both_active() {
        // WebVTT allows simultaneous cues: a banner running the whole time plus a
        // line that appears in the middle. Both cover the overlap window, so both
        // are drawn (each at its own placement, see render_active).
        let ov = TextOverlay::from_webvtt(
            "WEBVTT\n\n00:00:00.000 --> 00:00:10.000\nTOP BANNER\n\n00:00:02.000 --> 00:00:04.000\nLOWER LINE\n",
        );
        assert_eq!(ov.cue_count(), 2);
        assert_eq!(ov.active(1_000_000_000).len(), 1, "only the banner early");
        assert_eq!(ov.active(3_000_000_000).len(), 2, "both in the overlap window");
        assert_eq!(ov.active(5_000_000_000).len(), 1, "banner again after the second ends");
    }

    #[tokio::test]
    async fn draws_text_only_while_cue_is_active() {
        let mut ov = TextOverlay::from_srt("1\n00:00:01,000 --> 00:00:02,000\nHELLO\n");
        ov.configure_pipeline(&rgba_caps(160, 64)).unwrap();

        // Before the cue: untouched (all black).
        let mut sink = PixelSink::default();
        ov.process(PipelinePacket::DataFrame(frame_at(160, 64, 0)), &mut sink).await.unwrap();
        let before = sink.last.take().expect("forwarded");
        assert!(!any_nonblack(&before, 160, 64), "no text before the cue starts");

        // During the cue: some white pixels were painted.
        ov.process(
            PipelinePacket::DataFrame(frame_at(160, 64, 1_500_000_000)),
            &mut sink,
        )
        .await
        .unwrap();
        let during = sink.last.take().expect("forwarded");
        assert!(any_nonblack(&during, 160, 64), "text painted during the cue");

        // After the cue: untouched again.
        ov.process(
            PipelinePacket::DataFrame(frame_at(160, 64, 3_000_000_000)),
            &mut sink,
        )
        .await
        .unwrap();
        let after = sink.last.take().expect("forwarded");
        assert!(!any_nonblack(&after, 160, 64), "no text after the cue ends");

        assert_eq!(ov.drawn_count(), 3);
    }

    /// Bounding box (min_x, min_y, max_x, max_y) of pixels brighter than black,
    /// or `None` if the canvas is untouched.
    fn drawn_bounds(buf: &[u8], w: usize, h: usize) -> Option<(usize, usize, usize, usize)> {
        let mut bounds: Option<(usize, usize, usize, usize)> = None;
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                if buf[i] != 0 || buf[i + 1] != 0 || buf[i + 2] != 0 {
                    bounds = Some(match bounds {
                        None => (x, y, x, y),
                        Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
                    });
                }
            }
        }
        bounds
    }

    /// A one-cue overlay (active for all time) of `text` with `settings`, at the
    /// given geometry, configured and ready to `render_active`.
    fn overlay_with(w: u32, h: u32, text: &str, settings: crate::subparse::CueSettings) -> TextOverlay {
        TextOverlay { width: w, height: h, configured: true, ..TextOverlay::new() }.with_cues(vec![Cue {
            start_ns: 0,
            end_ns: u64::MAX,
            text: text.into(),
            settings,
        }])
    }

    #[test]
    fn render_is_clipped_on_a_tiny_canvas_without_panicking() {
        use crate::subparse::CueSettings;
        // A long line on a tiny canvas must not write out of bounds.
        let mut buf = black(32, 16);
        overlay_with(32, 16, "A VERY LONG SUBTITLE LINE", CueSettings::default())
            .render_active(&mut buf, 0);
        assert!(drawn_bounds(&buf, 32, 16).is_some(), "something was drawn");
    }

    #[test]
    fn line_setting_places_the_cue_vertically() {
        use crate::subparse::CueSettings;
        let (w, h) = (160usize, 96usize);

        // line:0% -> top of the frame.
        let mut top_buf = black(w, h);
        overlay_with(w as u32, h as u32, "HI", CueSettings { line: Some(0), ..CueSettings::default() })
            .render_active(&mut top_buf, 0);
        let (_, _, _, top_max_y) = drawn_bounds(&top_buf, w, h).expect("drawn");
        assert!(top_max_y < h / 2, "line:0% lands in the top half ({top_max_y})");

        // Default (auto line) -> bottom of the frame.
        let mut auto_buf = black(w, h);
        overlay_with(w as u32, h as u32, "HI", CueSettings::default()).render_active(&mut auto_buf, 0);
        let (_, auto_min_y, _, _) = drawn_bounds(&auto_buf, w, h).expect("drawn");
        assert!(auto_min_y > h / 2, "auto line stacks at the bottom ({auto_min_y})");
    }

    #[test]
    fn position_and_align_place_the_cue_horizontally() {
        use crate::subparse::{CueSettings, TextAlign};
        let (w, h) = (200usize, 96usize);

        // position:0% align:start -> hugs the left edge.
        let mut left_buf = black(w, h);
        overlay_with(
            w as u32,
            h as u32,
            "HI",
            CueSettings { position: Some(0), align: TextAlign::Start, ..CueSettings::default() },
        )
        .render_active(&mut left_buf, 0);
        let (left_min_x, _, left_max_x, _) = drawn_bounds(&left_buf, w, h).expect("drawn");
        assert!(left_min_x < w / 4, "left-aligned cue starts near the left edge ({left_min_x})");
        assert!(left_max_x < w / 2, "and stays in the left half ({left_max_x})");

        // position:100% align:end -> hugs the right edge.
        let mut right_buf = black(w, h);
        overlay_with(
            w as u32,
            h as u32,
            "HI",
            CueSettings { position: Some(100), align: TextAlign::End, ..CueSettings::default() },
        )
        .render_active(&mut right_buf, 0);
        let (right_min_x, _, right_max_x, _) = drawn_bounds(&right_buf, w, h).expect("drawn");
        assert!(right_max_x > 3 * w / 4, "right-aligned cue ends near the right edge ({right_max_x})");
        assert!(right_min_x > w / 2, "and stays in the right half ({right_min_x})");
    }

    #[test]
    fn intercept_rejects_non_rgba() {
        let ov = TextOverlay::new();
        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(8),
            height: Dim::Fixed(8),
            framerate: Rate::Any,
        };
        assert!(ov.intercept_caps(&nv12).is_err());
        assert!(ov.intercept_caps(&rgba_caps(16, 16)).is_ok());
    }

    // -- TextOverlayN (M403): the two-input video + text-stream overlay. --------

    fn text_cue_frame(pts_ns: u64, duration_ns: u64, text: &str) -> Frame {
        Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(text.as_bytes().to_vec().into_boxed_slice())),
            FrameTiming { pts_ns, duration_ns, ..FrameTiming::default() },
            0,
        )
    }

    #[test]
    fn overlayn_negotiates_video_and_text_pads() {
        use g2g_core::TextFormat;
        let ov = TextOverlayN::new();
        // Pad 0 = video (RGBA8), pad 1 = text (Utf8); each rejects the other's caps.
        assert!(ov.intercept_caps(0, &rgba_caps(16, 16)).is_ok());
        assert!(ov.intercept_caps(0, &Caps::Text { format: TextFormat::Utf8 }).is_err());
        assert!(ov.intercept_caps(1, &Caps::Text { format: TextFormat::Utf8 }).is_ok());
        assert!(ov.intercept_caps(1, &rgba_caps(16, 16)).is_err());
    }

    #[tokio::test]
    async fn overlayn_paints_streamed_cue_onto_video() {
        use g2g_core::TextFormat;
        let mut ov = TextOverlayN::new();
        ov.configure_pipeline(0, &rgba_caps(160, 64)).expect("video pad");
        ov.configure_pipeline(1, &Caps::Text { format: TextFormat::Utf8 }).expect("text pad");
        // Merged output is the video caps.
        assert_eq!(ov.output_caps().unwrap(), rgba_caps(160, 64));

        let mut sink = PixelSink::default();
        // A cue arrives on the text pad first (PTS-merged: it precedes its video).
        ov.process(1, PipelinePacket::DataFrame(text_cue_frame(1_000_000_000, 2_000_000_000, "HELLO")), &mut sink)
            .await
            .unwrap();
        assert_eq!(ov.cue_count(), 1, "cue stored from the text stream");

        // Video frame before the cue window: untouched.
        ov.process(0, PipelinePacket::DataFrame(frame_at(160, 64, 0)), &mut sink).await.unwrap();
        assert!(!any_nonblack(&sink.last.take().unwrap(), 160, 64), "no text before the cue");

        // Video frame inside the window: the streamed cue is painted.
        ov.process(0, PipelinePacket::DataFrame(frame_at(160, 64, 1_500_000_000)), &mut sink)
            .await
            .unwrap();
        assert!(any_nonblack(&sink.last.take().unwrap(), 160, 64), "streamed cue painted on video");

        // Video frame after the window: untouched again.
        ov.process(0, PipelinePacket::DataFrame(frame_at(160, 64, 4_000_000_000)), &mut sink)
            .await
            .unwrap();
        assert!(!any_nonblack(&sink.last.take().unwrap(), 160, 64), "no text after the cue");
        assert_eq!(ov.drawn_count(), 3);
    }

    #[tokio::test]
    async fn overlayn_text_flush_drops_pending_cues() {
        use g2g_core::TextFormat;
        let mut ov = TextOverlayN::new();
        ov.configure_pipeline(0, &rgba_caps(32, 32)).unwrap();
        ov.configure_pipeline(1, &Caps::Text { format: TextFormat::Utf8 }).unwrap();
        let mut sink = PixelSink::default();
        ov.process(1, PipelinePacket::DataFrame(text_cue_frame(0, 1_000_000_000, "X")), &mut sink)
            .await
            .unwrap();
        assert_eq!(ov.cue_count(), 1);
        ov.process(1, PipelinePacket::Flush, &mut sink).await.unwrap();
        assert_eq!(ov.cue_count(), 0, "flush clears pending cues");
    }
}
