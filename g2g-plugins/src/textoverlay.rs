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
//! all-caps ASCII bitmap font is the `no_std` baseline.
//!
//! With the `truetype-overlay` feature (M409) the overlay instead rasterizes
//! glyphs from a loaded `.ttf` / `.otf` / `.ttc` ([`TextOverlay::with_font`] /
//! `font=`), so CJK, accented Latin, and mixed-case text render, horizontal and
//! vertical (`vertical:rl` / `lr`). `ab_glyph` does the parsing / rasterization on
//! the CPU, covering both glyf and CFF/CFF2 outlines; full shaping + bidi
//! (cosmic-text) is a later upgrade.
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
    PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat, TextFormat,
};

use crate::bitmapfont::{glyph, GLYPH_ADVANCE, GLYPH_HEIGHT};
use crate::paint::blend_px;
#[cfg(feature = "truetype-overlay")]
use crate::subparse::WritingMode;
use crate::subparse::{parse_srt, parse_ssa, parse_ttml, parse_webvtt, Cue, TextAlign};

/// A parsed TrueType / OpenType face used by the [`truetype-overlay`](crate)
/// render path. Wraps `ab_glyph` (glyf + CFF/CFF2 outlines) behind a small shim
/// whose `Metrics` mirror the y-up fontdue contract the placement math expects,
/// so switching rasterizer left that math unchanged. Also keeps `TextOverlay`
/// deriving `Debug` (`ab_glyph::FontVec` does not implement it).
#[cfg(feature = "truetype-overlay")]
struct FontFace(ab_glyph::FontVec);

/// One glyph's placement metrics, in the y-up convention fontdue used: `xmin` is
/// the pen-to-left-edge offset, `ymin` the baseline-to-bottom-edge offset
/// (negative below the baseline), `width` / `height` the coverage-bitmap size.
#[cfg(feature = "truetype-overlay")]
struct Metrics {
    advance_width: f32,
    xmin: i32,
    ymin: i32,
    width: usize,
    height: usize,
}

/// Scaled line metrics: `ascent` above the baseline and `new_line_size` the
/// baseline-to-baseline advance (ascent - descent + line gap).
#[cfg(feature = "truetype-overlay")]
struct LineMetrics {
    ascent: f32,
    new_line_size: f32,
}

#[cfg(feature = "truetype-overlay")]
impl FontFace {
    /// Whether this face has a real (non-`.notdef`) glyph for `c`.
    fn has_glyph(&self, c: char) -> bool {
        use ab_glyph::Font;
        self.0.glyph_id(c).0 != 0
    }

    /// Scaled ascent + line advance at `px`.
    fn line_metrics(&self, px: f32) -> LineMetrics {
        use ab_glyph::{Font, ScaleFont};
        let sf = self.0.as_scaled(px);
        LineMetrics {
            ascent: sf.ascent(),
            new_line_size: sf.height() + sf.line_gap(),
        }
    }

    /// Advance width of `c` at `px` (no rasterization); other `Metrics` fields
    /// are unused by the callers that ask only for the advance.
    fn metrics(&self, c: char, px: f32) -> Metrics {
        use ab_glyph::{Font, ScaleFont};
        let id = self.0.glyph_id(c);
        Metrics {
            advance_width: self.0.as_scaled(px).h_advance(id),
            xmin: 0,
            ymin: 0,
            width: 0,
            height: 0,
        }
    }

    /// Rasterize `c` at `px` to a coverage bitmap (one byte per pixel) plus its
    /// placement metrics. A glyph with no outline (space) yields an empty bitmap.
    fn rasterize(&self, c: char, px: f32) -> (Metrics, Vec<u8>) {
        use ab_glyph::{Font, ScaleFont};
        let id = self.0.glyph_id(c);
        let advance_width = self.0.as_scaled(px).h_advance(id);
        let glyph = id.with_scale_and_position(px, ab_glyph::point(0.0, 0.0));
        let Some(outlined) = self.0.outline_glyph(glyph) else {
            return (
                Metrics {
                    advance_width,
                    xmin: 0,
                    ymin: 0,
                    width: 0,
                    height: 0,
                },
                Vec::new(),
            );
        };
        let b = outlined.px_bounds();
        let width = b.width().round() as usize;
        let height = b.height().round() as usize;
        let mut cov = alloc::vec![0u8; width * height];
        outlined.draw(|x, y, c| {
            let (x, y) = (x as usize, y as usize);
            if x < width && y < height {
                cov[y * width + x] = (c * 255.0 + 0.5) as u8;
            }
        });
        // px_bounds is y-down from the baseline; convert to the y-up contract.
        let m = Metrics {
            advance_width,
            xmin: b.min.x.round() as i32,
            ymin: -(b.max.y.round() as i32),
            width,
            height,
        };
        (m, cov)
    }
}

#[cfg(feature = "truetype-overlay")]
impl core::fmt::Debug for FontFace {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("FontFace(..)")
    }
}
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
    /// TrueType / OpenType face fallback chain (the `truetype-overlay` feature).
    /// Glyphs are rasterized from the first face that has the character (so a
    /// Latin primary plus a CJK fallback renders mixed text); empty means the 8x8
    /// ASCII bitmap font is used. `ab_glyph` does no fallback itself, hence the
    /// explicit chain.
    #[cfg(feature = "truetype-overlay")]
    fonts: Vec<FontFace>,
    /// The primary `font=` path, retained for `get_property` round-trips.
    #[cfg(feature = "truetype-overlay")]
    font_path: Option<String>,
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
            #[cfg(feature = "truetype-overlay")]
            fonts: Vec::new(),
            #[cfg(feature = "truetype-overlay")]
            font_path: None,
            drawn: 0,
        }
    }

    /// Append a glyph font from in-memory `.ttf` / `.otf` / `.ttc` bytes to the
    /// fallback chain (the `truetype-overlay` feature). `collection_index` selects
    /// a face in a `.ttc` collection (0 for a plain `.ttf` / `.otf`). The first
    /// font added is the primary; later fonts cover characters the primary lacks
    /// (e.g. a Latin primary + a CJK fallback). Adding any font switches the
    /// render path from the ASCII bitmap to rasterized glyphs. `ab_glyph`
    /// rasterizes both glyf and CFF/CFF2 outlines, so OpenType-CFF fonts (e.g.
    /// Noto Sans CJK OTF) render, not only glyf `.ttf`s.
    #[cfg(feature = "truetype-overlay")]
    pub fn add_font_bytes(&mut self, bytes: &[u8], collection_index: u32) -> Result<(), G2gError> {
        let font = ab_glyph::FontVec::try_from_vec_and_index(bytes.to_vec(), collection_index)
            .map_err(|_| G2gError::CapsMismatch)?;
        self.fonts.push(FontFace(font));
        Ok(())
    }

    /// Builder form of [`add_font_bytes`](Self::add_font_bytes).
    #[cfg(feature = "truetype-overlay")]
    pub fn with_font_bytes(
        mut self,
        bytes: &[u8],
        collection_index: u32,
    ) -> Result<Self, G2gError> {
        self.add_font_bytes(bytes, collection_index)?;
        Ok(self)
    }

    /// Append a glyph font from a `.ttf` / `.ttc` file path to the fallback chain
    /// (`truetype-overlay` + `std`). The first path added is recorded as the
    /// primary `font=`. See [`add_font_bytes`](Self::add_font_bytes).
    #[cfg(all(feature = "truetype-overlay", feature = "std"))]
    pub fn add_font(&mut self, path: &str) -> Result<(), G2gError> {
        let bytes = std::fs::read(path).map_err(|_| G2gError::CapsMismatch)?;
        self.add_font_bytes(&bytes, 0)?;
        if self.font_path.is_none() {
            self.font_path = Some(path.into());
        }
        Ok(())
    }

    /// Builder form of [`add_font`](Self::add_font); chain calls to add fallbacks.
    #[cfg(all(feature = "truetype-overlay", feature = "std"))]
    pub fn with_font(mut self, path: impl AsRef<str>) -> Result<Self, G2gError> {
        self.add_font(path.as_ref())?;
        Ok(self)
    }

    /// The first font in the chain that has a glyph for `c`, else the primary
    /// (which renders the `.notdef` box). Empty chain is unreachable here (the
    /// TTF path only runs with at least one font).
    #[cfg(feature = "truetype-overlay")]
    fn glyph_font(&self, c: char) -> &FontFace {
        for f in &self.fonts {
            if f.has_glyph(c) {
                return f;
            }
        }
        &self.fonts[0]
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
        matches!(
            caps,
            Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                ..
            }
        )
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
            // Per-cue WebVTT `::cue` colours, falling back to the element defaults.
            let fg = s.color.unwrap_or(self.text_color);
            let bg = s.background.unwrap_or(self.bg_color);

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
                bg,
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
                    self.blit_glyph(buf, gx, y0, scale, glyph(c), fg);
                    gx += cell_w;
                }
            }
        }
    }

    /// Blit one 8x8 glyph at output `(gx, gy)`, each set bit a `scale` x `scale`
    /// block of `color`, clipped to the canvas.
    fn blit_glyph(
        &self,
        buf: &mut [u8],
        gx: i32,
        gy: i32,
        scale: i32,
        rows: [u8; 8],
        color: [u8; 4],
    ) {
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
                        color,
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

    /// Subtitle glyph size in pixels for the TrueType path: ~1/20 of the frame
    /// height, with a floor so small frames stay legible.
    #[cfg(feature = "truetype-overlay")]
    fn ttf_px(&self) -> f32 {
        (self.height as f32 / 20.0).max(16.0)
    }

    /// Alpha-blend a rasterized glyph's coverage bitmap (`gw` x `gh`, one byte
    /// per pixel) at output `(x0, y0)` in the text colour, clipped to the canvas.
    #[cfg(feature = "truetype-overlay")]
    fn blit_coverage(
        &self,
        buf: &mut [u8],
        x0: i32,
        y0: i32,
        (gw, gh): (usize, usize),
        cov: &[u8],
        color: [u8; 4],
    ) {
        let w = self.width as i32;
        let h = self.height as i32;
        for ry in 0..gh as i32 {
            let py = y0 + ry;
            if py < 0 || py >= h {
                continue;
            }
            for rx in 0..gw as i32 {
                let px = x0 + rx;
                if px < 0 || px >= w {
                    continue;
                }
                let a = cov[(ry as usize) * gw + rx as usize];
                if a != 0 {
                    blend_px(buf, ((py * w + px) * 4) as usize, color, a);
                }
            }
        }
    }

    /// TrueType render path (the `truetype-overlay` feature): rasterize each
    /// active cue's glyphs from the loaded font. Horizontal cues lay out
    /// left-to-right, top-to-bottom (auto-`line` cues stack from the bottom like
    /// the bitmap path); `vertical:rl` / `lr` cues lay out as top-to-bottom
    /// columns advancing right-to-left / left-to-right, with `align` justifying
    /// each column vertically. Placement (`position` / `line`) mirrors the bitmap
    /// path; metrics and advances come from the font.
    #[cfg(feature = "truetype-overlay")]
    fn render_active_ttf(&self, buf: &mut [u8], t_ns: u64) {
        // Line metrics come from the primary; each glyph is rasterized from the
        // first font in the chain that has it (see `glyph_font`).
        let primary = &self.fonts[0];
        let w = self.width as f32;
        let h = self.height as f32;
        let px = self.ttf_px();
        let lm = primary.line_metrics(px);
        let line_h = lm.new_line_size.max(px);
        let pad = (px * 0.25).max(2.0);
        let margin = px * 0.5;
        let mut auto_bottom = h - margin;

        for cue in self.active(t_ns) {
            let lines: Vec<&str> = cue.text.lines().collect();
            if lines.is_empty() {
                continue;
            }
            let s = cue.settings;
            // Per-cue WebVTT `::cue` colours, falling back to the element defaults.
            let fg = s.color.unwrap_or(self.text_color);
            let bg = s.background.unwrap_or(self.bg_color);

            if matches!(
                s.vertical,
                WritingMode::VerticalRl | WritingMode::VerticalLr
            ) {
                let rl = matches!(s.vertical, WritingMode::VerticalRl);
                let col_w = px * 1.3;
                let cell_h = px * 1.15;
                let n_cols = lines.len();
                let max_len = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0) as f32;
                let block_w = n_cols as f32 * col_w;
                let block_h = max_len * cell_h;
                // `position` anchors the block centre; default hugs the leading
                // edge (right for rl, left for lr). `line` sets the top.
                let block_left = match s.position {
                    Some(p) => p as f32 / 100.0 * w - block_w / 2.0,
                    None if rl => w - block_w - margin,
                    None => margin,
                }
                .clamp(0.0, (w - block_w).max(0.0));
                let block_top = match s.line {
                    Some(p) => {
                        (p as f32 / 100.0 * h).clamp(margin, (h - margin - block_h).max(margin))
                    }
                    None => margin,
                };
                self.fill_rect(
                    buf,
                    (block_left - pad) as i32,
                    (block_top - pad) as i32,
                    (block_w + 2.0 * pad) as i32,
                    (block_h + 2.0 * pad) as i32,
                    bg,
                );
                for (ci, line) in lines.iter().enumerate() {
                    // First logical line is the rightmost column when rl.
                    let col = if rl { n_cols - 1 - ci } else { ci };
                    let col_x = block_left + col as f32 * col_w;
                    let chars: Vec<char> = line.chars().collect();
                    let col_h = chars.len() as f32 * cell_h;
                    let start_y = block_top
                        + match s.align {
                            TextAlign::Start => 0.0,
                            TextAlign::Center => (block_h - col_h) / 2.0,
                            TextAlign::End => block_h - col_h,
                        };
                    for (j, &c) in chars.iter().enumerate() {
                        let (m, cov) = self.glyph_font(c).rasterize(c, px);
                        let gx = col_x + (col_w - m.advance_width) / 2.0 + m.xmin as f32;
                        let baseline = start_y + lm.ascent + j as f32 * cell_h;
                        let gy = baseline - m.ymin as f32 - m.height as f32;
                        self.blit_coverage(
                            buf,
                            gx as i32,
                            gy as i32,
                            (m.width, m.height),
                            &cov,
                            fg,
                        );
                    }
                }
            } else {
                let line_ws: Vec<f32> = lines
                    .iter()
                    .map(|l| {
                        l.chars()
                            .map(|c| self.glyph_font(c).metrics(c, px).advance_width)
                            .sum()
                    })
                    .collect();
                let block_w = line_ws.iter().copied().fold(0.0_f32, f32::max);
                let block_h = lines.len() as f32 * line_h;
                let anchor_x = s.position.map(|p| p as f32 / 100.0 * w).unwrap_or(w / 2.0);
                let block_left =
                    ttf_align_left(s.align, anchor_x, block_w).clamp(0.0, (w - block_w).max(0.0));
                let block_top = match s.line {
                    Some(p) => {
                        (p as f32 / 100.0 * h).clamp(margin, (h - margin - block_h).max(margin))
                    }
                    None => {
                        let t = (auto_bottom - block_h).max(margin);
                        auto_bottom = t - pad - line_h * 0.2;
                        t
                    }
                };
                self.fill_rect(
                    buf,
                    (block_left - pad) as i32,
                    (block_top - pad) as i32,
                    (block_w + 2.0 * pad) as i32,
                    (block_h + 2.0 * pad) as i32,
                    bg,
                );
                for (row, line) in lines.iter().enumerate() {
                    let line_w = line_ws[row];
                    let x0 = match s.align {
                        TextAlign::Center => block_left + (block_w - line_w) / 2.0,
                        TextAlign::Start => block_left,
                        TextAlign::End => block_left + (block_w - line_w),
                    };
                    let baseline = block_top + lm.ascent + row as f32 * line_h;
                    let mut pen = x0;
                    for c in line.chars() {
                        let (m, cov) = self.glyph_font(c).rasterize(c, px);
                        let gx = pen + m.xmin as f32;
                        let gy = baseline - m.ymin as f32 - m.height as f32;
                        self.blit_coverage(
                            buf,
                            gx as i32,
                            gy as i32,
                            (m.width, m.height),
                            &cov,
                            fg,
                        );
                        pen += m.advance_width;
                    }
                }
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

    /// Load the glyph font from a file (`font=` property). Needs both the
    /// `truetype-overlay` feature and `std`; otherwise the build has no font
    /// backend and the call reports an unsupported value.
    #[cfg(all(feature = "truetype-overlay", feature = "std"))]
    fn load_font(&mut self, path: &str) -> Result<(), PropError> {
        // The property sets a single primary font (replacing any chain).
        self.fonts.clear();
        self.font_path = None;
        self.add_font(path).map_err(|_| PropError::Value)
    }

    #[cfg(not(all(feature = "truetype-overlay", feature = "std")))]
    fn load_font(&mut self, _path: &str) -> Result<(), PropError> {
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

/// `f32` form of [`align_left`] for the TrueType render path.
#[cfg(feature = "truetype-overlay")]
fn ttf_align_left(align: TextAlign, anchor: f32, block_w: f32) -> f32 {
    match align {
        TextAlign::Center => anchor - block_w / 2.0,
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
                        // Rasterized font path when one is loaded; else the
                        // ASCII bitmap baseline.
                        #[cfg(feature = "truetype-overlay")]
                        if self.fonts.is_empty() {
                            self.render_active(&mut buf[..need], t_ns);
                        } else {
                            self.render_active_ttf(&mut buf[..need], t_ns);
                        }
                        #[cfg(not(feature = "truetype-overlay"))]
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
            "font" => {
                let path = value.as_str().ok_or(PropError::Type)?;
                self.load_font(path)
            }
            // 0xAARRGGBB packed color, the gst textoverlay convention. The
            // element stores [R, G, B, A].
            "color" => {
                let argb = value.as_uint().ok_or(PropError::Type)? as u32;
                self.text_color = [
                    (argb >> 16) as u8,
                    (argb >> 8) as u8,
                    argb as u8,
                    (argb >> 24) as u8,
                ];
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.location.clone().unwrap_or_default())),
            #[cfg(feature = "truetype-overlay")]
            "font" => Some(PropValue::Str(self.font_path.clone().unwrap_or_default())),
            #[cfg(not(feature = "truetype-overlay"))]
            "font" => Some(PropValue::Str(String::new())),
            "color" => {
                let [r, g, b, a] = self.text_color;
                Some(PropValue::Uint(
                    ((a as u64) << 24) | ((r as u64) << 16) | ((g as u64) << 8) | b as u64,
                ))
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
/// giving correct A/V-text alignment. [`SubParse`] streams each cue as soon as it
/// is fully parsed (M405), so the merge only buffers video up to the next cue's
/// start, not to the end of the subtitle stream. The
/// rendering is reused wholesale from [`TextOverlay`] (composition); the text pad
/// only feeds it cues. Cue positioning (WebVTT / SSA `position` / `line` / `align`)
/// rides the stream as [`TextCueMeta`](crate::subparse::TextCueMeta) frame-meta
/// under the `metadata` feature (M406), so a placed cue renders where it asks; on
/// the ZST baseline (no meta) every cue draws at the renderer default (bottom-centre).
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
        Self {
            inner: TextOverlay::new(),
            video_caps: None,
        }
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

    /// Named request pads (M481): `video`/`video_0` -> the video pad (0),
    /// `text`/`subtitle`/`text_0` -> the text pad (1), so a launch line can wire
    /// `d.video_0 ! ... ! o.video   d.text_0 ! o.text` in either order and the
    /// video still lands on pad 0 (keeping `output_follows_input`/PTS-merge valid).
    fn input_pad_index(
        &self,
        req: &g2g_core::runtime::PadRequest,
        _ordinal: usize,
    ) -> Option<usize> {
        match req.kind {
            g2g_core::runtime::PadKind::Video => Some(Self::VIDEO),
            g2g_core::runtime::PadKind::Text => Some(Self::TEXT),
            _ => None,
        }
    }

    fn intercept_caps(&self, input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match input {
            Self::VIDEO if TextOverlay::accepts(upstream_caps) => Ok(upstream_caps.clone()),
            Self::TEXT
                if matches!(
                    upstream_caps,
                    Caps::Text {
                        format: TextFormat::Utf8
                    }
                ) =>
            {
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
            Self::TEXT => CapsConstraint::Accepts(CapsSet::one(Caps::Text {
                format: TextFormat::Utf8,
            })),
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
                Caps::Text {
                    format: TextFormat::Utf8,
                } => Ok(ConfigureOutcome::Accepted),
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
                            if let Some(slice) = frame.domain.as_system_slice() {
                                let text = String::from_utf8_lossy(slice).into_owned();
                                let start = frame.timing.pts_ns;
                                let end = start.saturating_add(frame.timing.duration_ns);
                                // Recover the cue placement from frame-meta (M406)
                                // if `SubParse` attached it; default otherwise (and
                                // always on the ZST baseline).
                                #[cfg(feature = "metadata")]
                                let settings = frame
                                    .meta
                                    .get::<crate::subparse::TextCueMeta>()
                                    .map(|m| m.settings)
                                    .unwrap_or_default();
                                #[cfg(not(feature = "metadata"))]
                                let settings = crate::subparse::CueSettings::default();
                                self.inner.push_cue(Cue {
                                    start_ns: start,
                                    end_ns: end,
                                    text,
                                    settings,
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
static TEXTOVERLAY_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "location",
        PropKind::Str,
        "path to an SRT (.srt) or WebVTT (.vtt) subtitle file; cues render by PTS",
    ),
    PropertySpec::new(
        "font",
        PropKind::Str,
        "path to a .ttf / .ttc font for glyph rendering (truetype-overlay); \
         needed for CJK / accented text. Without it the 8x8 ASCII bitmap is used",
    ),
    PropertySpec::new(
        "color",
        PropKind::Uint,
        "text color as 0xAARRGGBB (e.g. 4294967295 = opaque white)",
    )
    .with_default("4294967295"),
];

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
            FrameTiming {
                pts_ns,
                ..FrameTiming::default()
            },
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
                    if let Some(slice) = frame.domain.as_system_slice() {
                        self.last = Some(slice.to_vec());
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
        assert_eq!(
            ov.active(2_000_000_000)
                .iter()
                .map(|c| c.text.as_str())
                .collect::<Vec<_>>(),
            ["HELLO"]
        );
        assert_eq!(
            ov.active(5_500_000_000)
                .iter()
                .map(|c| c.text.as_str())
                .collect::<Vec<_>>(),
            ["BYE"]
        );
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
        assert_eq!(
            ov.active(3_000_000_000).len(),
            2,
            "both in the overlap window"
        );
        assert_eq!(
            ov.active(5_000_000_000).len(),
            1,
            "banner again after the second ends"
        );
    }

    #[tokio::test]
    async fn draws_text_only_while_cue_is_active() {
        let mut ov = TextOverlay::from_srt("1\n00:00:01,000 --> 00:00:02,000\nHELLO\n");
        ov.configure_pipeline(&rgba_caps(160, 64)).unwrap();

        // Before the cue: untouched (all black).
        let mut sink = PixelSink::default();
        ov.process(PipelinePacket::DataFrame(frame_at(160, 64, 0)), &mut sink)
            .await
            .unwrap();
        let before = sink.last.take().expect("forwarded");
        assert!(
            !any_nonblack(&before, 160, 64),
            "no text before the cue starts"
        );

        // During the cue: some white pixels were painted.
        ov.process(
            PipelinePacket::DataFrame(frame_at(160, 64, 1_500_000_000)),
            &mut sink,
        )
        .await
        .unwrap();
        let during = sink.last.take().expect("forwarded");
        assert!(
            any_nonblack(&during, 160, 64),
            "text painted during the cue"
        );

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
    fn overlay_with(
        w: u32,
        h: u32,
        text: &str,
        settings: crate::subparse::CueSettings,
    ) -> TextOverlay {
        TextOverlay {
            width: w,
            height: h,
            configured: true,
            ..TextOverlay::new()
        }
        .with_cues(vec![Cue {
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
        overlay_with(
            w as u32,
            h as u32,
            "HI",
            CueSettings {
                line: Some(0),
                ..CueSettings::default()
            },
        )
        .render_active(&mut top_buf, 0);
        let (_, _, _, top_max_y) = drawn_bounds(&top_buf, w, h).expect("drawn");
        assert!(
            top_max_y < h / 2,
            "line:0% lands in the top half ({top_max_y})"
        );

        // Default (auto line) -> bottom of the frame.
        let mut auto_buf = black(w, h);
        overlay_with(w as u32, h as u32, "HI", CueSettings::default())
            .render_active(&mut auto_buf, 0);
        let (_, auto_min_y, _, _) = drawn_bounds(&auto_buf, w, h).expect("drawn");
        assert!(
            auto_min_y > h / 2,
            "auto line stacks at the bottom ({auto_min_y})"
        );
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
            CueSettings {
                position: Some(0),
                align: TextAlign::Start,
                ..CueSettings::default()
            },
        )
        .render_active(&mut left_buf, 0);
        let (left_min_x, _, left_max_x, _) = drawn_bounds(&left_buf, w, h).expect("drawn");
        assert!(
            left_min_x < w / 4,
            "left-aligned cue starts near the left edge ({left_min_x})"
        );
        assert!(
            left_max_x < w / 2,
            "and stays in the left half ({left_max_x})"
        );

        // position:100% align:end -> hugs the right edge.
        let mut right_buf = black(w, h);
        overlay_with(
            w as u32,
            h as u32,
            "HI",
            CueSettings {
                position: Some(100),
                align: TextAlign::End,
                ..CueSettings::default()
            },
        )
        .render_active(&mut right_buf, 0);
        let (right_min_x, _, right_max_x, _) = drawn_bounds(&right_buf, w, h).expect("drawn");
        assert!(
            right_max_x > 3 * w / 4,
            "right-aligned cue ends near the right edge ({right_max_x})"
        );
        assert!(
            right_min_x > w / 2,
            "and stays in the right half ({right_min_x})"
        );
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

    // -- TrueType/OpenType overlay (M409): CJK / vertical rendering via ab_glyph. -

    /// Read the first available CJK-capable system font, or `None` to skip (CI
    /// without CJK fonts). These are the Fedora paths the dev host has.
    #[cfg(feature = "truetype-overlay")]
    fn cjk_font_bytes() -> Option<Vec<u8>> {
        for p in [
            "/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/google-noto-sans-cjk-vf-fonts/NotoSansCJK-VF.ttc",
            "/usr/share/fonts/google-droid-sans-fonts/DroidSansJapanese.ttf",
        ] {
            if let Ok(b) = std::fs::read(p) {
                return Some(b);
            }
        }
        None
    }

    #[cfg(feature = "truetype-overlay")]
    fn cjk_overlay(
        w: u32,
        h: u32,
        text: &str,
        settings: crate::subparse::CueSettings,
    ) -> Option<TextOverlay> {
        let bytes = cjk_font_bytes()?;
        let mut ov = TextOverlay::new()
            .with_font_bytes(&bytes, 0)
            .expect("font parses")
            .with_cues(vec![Cue {
                start_ns: 0,
                end_ns: u64::MAX,
                text: text.into(),
                settings,
            }]);
        ov.width = w;
        ov.height = h;
        ov.configured = true;
        Some(ov)
    }

    #[test]
    #[cfg(feature = "truetype-overlay")]
    fn truetype_renders_cjk_that_the_bitmap_font_cannot() {
        use crate::subparse::CueSettings;
        let (w, h) = (480usize, 160usize);
        // The bitmap path paints nothing for CJK (no glyphs); the TTF path must.
        let bitmap = TextOverlay {
            width: w as u32,
            height: h as u32,
            configured: true,
            ..TextOverlay::new()
        }
        .with_cues(vec![Cue {
            start_ns: 0,
            end_ns: u64::MAX,
            text: "日本語".into(),
            settings: CueSettings::default(),
        }]);
        let mut bbuf = black(w, h);
        bitmap.render_active(&mut bbuf, 0);
        assert!(
            drawn_bounds(&bbuf, w, h).is_none(),
            "bitmap font has no CJK glyphs"
        );

        let Some(ov) = cjk_overlay(w as u32, h as u32, "日本語", CueSettings::default()) else {
            std::eprintln!("skip: no CJK system font found");
            return;
        };
        let mut buf = black(w, h);
        ov.render_active_ttf(&mut buf, 0);
        assert!(
            drawn_bounds(&buf, w, h).is_some(),
            "TTF font renders CJK glyphs"
        );
    }

    #[test]
    #[cfg(feature = "truetype-overlay")]
    fn truetype_vertical_lays_out_in_columns() {
        use crate::subparse::{CueSettings, WritingMode};
        let (w, h) = (320usize, 320usize);
        // vertical:rl with two logical lines -> two columns; both must paint, and
        // the rightmost column (first line) should sit to the right of the second.
        let settings = CueSettings {
            vertical: WritingMode::VerticalRl,
            ..CueSettings::default()
        };
        let Some(ov) = cjk_overlay(w as u32, h as u32, "縦書き\n二列目", settings) else {
            std::eprintln!("skip: no CJK system font found");
            return;
        };
        let mut buf = black(w, h);
        ov.render_active_ttf(&mut buf, 0);
        let bounds = drawn_bounds(&buf, w, h).expect("vertical CJK painted");
        // Taller than one glyph (stacked vertically) and spanning two columns.
        let (x0, y0, x1, y1) = bounds;
        assert!(y1 - y0 > (h / 8), "glyphs stack down the column");
        assert!(x1 - x0 > (w / 12), "two columns span horizontally");
    }

    /// Read the first available OpenType-CFF (`.otf`) system font, or `None` to
    /// skip. `.otf` fonts carry CFF outlines, which the old fontdue backend could
    /// not rasterize (empty glyphs); `ab_glyph` does.
    #[cfg(feature = "truetype-overlay")]
    fn cff_font_bytes() -> Option<Vec<u8>> {
        for p in [
            "/usr/share/fonts/aajohan-comfortaa-fonts/Comfortaa-Regular.otf",
            "/usr/share/fonts/adobe-source-code-pro/SourceCodePro-Regular.otf",
            "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
        ] {
            if let Ok(b) = std::fs::read(p) {
                return Some(b);
            }
        }
        None
    }

    #[test]
    #[cfg(feature = "truetype-overlay")]
    fn opentype_cff_font_renders_glyphs() {
        use crate::subparse::CueSettings;
        let (w, h) = (240usize, 96usize);
        let Some(bytes) = cff_font_bytes() else {
            std::eprintln!("skip: no CFF (.otf) system font found");
            return;
        };
        let mut ov = TextOverlay::new()
            .with_font_bytes(&bytes, 0)
            .expect("CFF font parses")
            .with_cues(vec![Cue {
                start_ns: 0,
                end_ns: u64::MAX,
                text: "Ag".into(),
                settings: CueSettings::default(),
            }]);
        ov.width = w as u32;
        ov.height = h as u32;
        ov.configured = true;
        let mut buf = black(w, h);
        ov.render_active_ttf(&mut buf, 0);
        // fontdue produced empty glyphs for CFF; ab_glyph rasterizes the outlines.
        assert!(
            drawn_bounds(&buf, w, h).is_some(),
            "CFF outlines rasterize to visible glyphs"
        );
    }

    // -- TextOverlayN (M403): the two-input video + text-stream overlay. --------

    fn text_cue_frame(pts_ns: u64, duration_ns: u64, text: &str) -> Frame {
        Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(
                text.as_bytes().to_vec().into_boxed_slice(),
            )),
            FrameTiming {
                pts_ns,
                duration_ns,
                ..FrameTiming::default()
            },
            0,
        )
    }

    #[test]
    fn overlayn_negotiates_video_and_text_pads() {
        use g2g_core::TextFormat;
        let ov = TextOverlayN::new();
        // Pad 0 = video (RGBA8), pad 1 = text (Utf8); each rejects the other's caps.
        assert!(ov.intercept_caps(0, &rgba_caps(16, 16)).is_ok());
        assert!(ov
            .intercept_caps(
                0,
                &Caps::Text {
                    format: TextFormat::Utf8
                }
            )
            .is_err());
        assert!(ov
            .intercept_caps(
                1,
                &Caps::Text {
                    format: TextFormat::Utf8
                }
            )
            .is_ok());
        assert!(ov.intercept_caps(1, &rgba_caps(16, 16)).is_err());
    }

    #[tokio::test]
    async fn overlayn_paints_streamed_cue_onto_video() {
        use g2g_core::TextFormat;
        let mut ov = TextOverlayN::new();
        ov.configure_pipeline(0, &rgba_caps(160, 64))
            .expect("video pad");
        ov.configure_pipeline(
            1,
            &Caps::Text {
                format: TextFormat::Utf8,
            },
        )
        .expect("text pad");
        // Merged output is the video caps.
        assert_eq!(ov.output_caps().unwrap(), rgba_caps(160, 64));

        let mut sink = PixelSink::default();
        // A cue arrives on the text pad first (PTS-merged: it precedes its video).
        ov.process(
            1,
            PipelinePacket::DataFrame(text_cue_frame(1_000_000_000, 2_000_000_000, "HELLO")),
            &mut sink,
        )
        .await
        .unwrap();
        assert_eq!(ov.cue_count(), 1, "cue stored from the text stream");

        // Video frame before the cue window: untouched.
        ov.process(
            0,
            PipelinePacket::DataFrame(frame_at(160, 64, 0)),
            &mut sink,
        )
        .await
        .unwrap();
        assert!(
            !any_nonblack(&sink.last.take().unwrap(), 160, 64),
            "no text before the cue"
        );

        // Video frame inside the window: the streamed cue is painted.
        ov.process(
            0,
            PipelinePacket::DataFrame(frame_at(160, 64, 1_500_000_000)),
            &mut sink,
        )
        .await
        .unwrap();
        assert!(
            any_nonblack(&sink.last.take().unwrap(), 160, 64),
            "streamed cue painted on video"
        );

        // Video frame after the window: untouched again.
        ov.process(
            0,
            PipelinePacket::DataFrame(frame_at(160, 64, 4_000_000_000)),
            &mut sink,
        )
        .await
        .unwrap();
        assert!(
            !any_nonblack(&sink.last.take().unwrap(), 160, 64),
            "no text after the cue"
        );
        assert_eq!(ov.drawn_count(), 3);
    }

    #[cfg(feature = "metadata")]
    #[tokio::test]
    async fn overlayn_honours_streamed_cue_positioning_meta() {
        // A streamed cue carrying TextCueMeta (M406) must render where the meta
        // places it, not at the bottom-centre default: top-left here.
        use crate::subparse::{CueSettings, TextAlign, TextCueMeta};
        use g2g_core::TextFormat;
        let (w, h) = (200u32, 96u32);
        let mut ov = TextOverlayN::new();
        ov.configure_pipeline(0, &rgba_caps(w, h)).unwrap();
        ov.configure_pipeline(
            1,
            &Caps::Text {
                format: TextFormat::Utf8,
            },
        )
        .unwrap();

        let mut frame = text_cue_frame(0, u64::MAX / 2, "HI");
        frame.meta.attach(TextCueMeta {
            settings: CueSettings {
                position: Some(0),
                line: Some(0),
                align: TextAlign::Start,
                ..CueSettings::default()
            },
        });
        let mut sink = PixelSink::default();
        ov.process(1, PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        ov.process(0, PipelinePacket::DataFrame(frame_at(w, h, 0)), &mut sink)
            .await
            .unwrap();
        let painted = sink.last.take().expect("forwarded");
        let (_, _, max_x, max_y) =
            drawn_bounds(&painted, w as usize, h as usize).expect("cue painted");
        assert!(
            max_x < (w / 2) as usize,
            "meta position placed the cue in the left half ({max_x})"
        );
        assert!(
            max_y < (h / 2) as usize,
            "meta line placed the cue in the top half ({max_y})"
        );
    }

    #[tokio::test]
    async fn overlayn_text_flush_drops_pending_cues() {
        use g2g_core::TextFormat;
        let mut ov = TextOverlayN::new();
        ov.configure_pipeline(0, &rgba_caps(32, 32)).unwrap();
        ov.configure_pipeline(
            1,
            &Caps::Text {
                format: TextFormat::Utf8,
            },
        )
        .unwrap();
        let mut sink = PixelSink::default();
        ov.process(
            1,
            PipelinePacket::DataFrame(text_cue_frame(0, 1_000_000_000, "X")),
            &mut sink,
        )
        .await
        .unwrap();
        assert_eq!(ov.cue_count(), 1);
        ov.process(1, PipelinePacket::Flush, &mut sink)
            .await
            .unwrap();
        assert_eq!(ov.cue_count(), 0, "flush clears pending cues");
    }
}
