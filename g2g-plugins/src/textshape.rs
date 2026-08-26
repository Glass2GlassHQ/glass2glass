//! Shaped text layout for [`TextOverlay`](crate::textoverlay::TextOverlay) (M892,
//! the `text-shaping` feature): cosmic-text does harfrust shaping (Arabic
//! joining, Indic reordering, kerning / ligatures), unicode-bidi reordering, and
//! per-codepoint fallback across the system faces fontdb discovers, then swash
//! rasterizes each glyph to a coverage mask the overlay blits.
//!
//! cosmic-text is horizontal-only, so this covers horizontal cues; `vertical:rl`
//! / `lr` cues stay on the overlay's `ab_glyph` column renderer.
//!
//! [`FontSystem`] scans every system font directory on construction (tens to
//! hundreds of milliseconds), so the overlay builds one lazily on the first cue
//! and reuses it, and the [`SwashCache`] keeps rasterized glyphs across frames.

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use cosmic_text::{
    Attrs, Buffer, Family, FontSystem, Metrics, Shaping, Stretch, Style, SwashCache, SwashContent,
    Weight, Wrap,
};

use crate::subparse::FontStretch;

/// Identifies one rasterized glyph (face, glyph index, size, subpixel bin).
pub use cosmic_text::CacheKey as GlyphKey;

/// Identifies a face in the shaper's font database. Every shaped glyph names the
/// face fallback picked for it, so a backend that draws outlines itself (rather
/// than blitting [`TextShaper::image`]) can fetch that face with
/// [`TextShaper::face_data`].
pub type FontId = cosmic_text::fontdb::ID;

/// One positioned, rasterizable glyph. `x` / `y` are the swash pixel origin
/// relative to the block's top-left; the placement offsets in
/// [`GlyphImage`] still apply on top.
#[derive(Debug)]
pub struct ShapedGlyph {
    pub key: GlyphKey,
    pub x: i32,
    pub y: i32,
    /// Byte offset of this glyph's cluster in its logical line. In a
    /// right-to-left run `x` falls as `start` rises.
    pub start: usize,
    /// Glyph index in its font, after shaping (a joined Arabic form differs from
    /// the isolated form of the same character).
    pub glyph_id: u16,
    /// Whether the glyph sits in a right-to-left bidi run.
    pub rtl: bool,
    /// Size this glyph was shaped at, which is the block's `px` unless a
    /// [`StyledSpan`] covered it.
    pub font_size: f32,
    /// Horizontal advance of this glyph's cluster, for a caller measuring the
    /// pixels a byte range covers (a span background fill).
    pub advance: f32,
}

/// One visual line of a laid-out block, in visual (post-bidi) order. `top` and
/// `height` are the line box, relative to the block's top-left.
#[derive(Debug)]
pub struct ShapedLine {
    pub width: f32,
    pub top: f32,
    pub height: f32,
    /// Baseline of this line, relative to the block's top-left. What an
    /// underline is measured down from.
    pub baseline: f32,
    pub glyphs: Vec<ShapedGlyph>,
}

/// A byte range of the text to lay out with its own font attributes rather than
/// the block's. Ranges must be non-overlapping and ascending; the bytes between
/// them take the block's attributes. A `None` field keeps the block's value.
#[derive(Debug, Clone, Copy, Default)]
pub struct StyledSpan<'a> {
    pub start: usize,
    pub end: usize,
    pub font_size: Option<f32>,
    pub weight: Option<u16>,
    pub italic: Option<bool>,
    pub stretch: Option<FontStretch>,
    /// CSS `font-family` name for this run: a generic family (`serif`,
    /// `sans-serif`, `monospace`, `cursive`, `fantasy`) or a family the font
    /// database is queried for by name.
    pub family: Option<&'a str>,
}

impl StyledSpan<'_> {
    /// Whether this span asks for anything at all (an empty one would only
    /// split the shaped run).
    pub fn styles_anything(&self) -> bool {
        self.font_size.is_some()
            || self.weight.is_some()
            || self.italic.is_some()
            || self.stretch.is_some()
            || self.family.is_some()
    }

    /// Whether two spans ask for the same attributes, so an adjacent pair can be
    /// laid out as one run.
    pub fn same_style(&self, other: &Self) -> bool {
        self.font_size == other.font_size
            && self.weight == other.weight
            && self.italic == other.italic
            && self.stretch == other.stretch
            && self.family == other.family
    }
}

/// The CSS generic family names, and the cosmic-text family each selects. Any
/// other name is asked for as a family name.
const GENERIC_FAMILIES: [(&str, Family<'static>); 5] = [
    ("serif", Family::Serif),
    ("sans-serif", Family::SansSerif),
    ("monospace", Family::Monospace),
    ("cursive", Family::Cursive),
    ("fantasy", Family::Fantasy),
];

/// The cosmic-text family a CSS `font-family` name selects.
fn family_of(name: &str) -> Family<'_> {
    GENERIC_FAMILIES
        .iter()
        .find(|(generic, _)| generic.eq_ignore_ascii_case(name))
        .map_or(Family::Name(name), |&(_, family)| family)
}

/// The cosmic-text face width a CSS `font-stretch` keyword selects.
fn stretch_of(stretch: FontStretch) -> Stretch {
    match stretch {
        FontStretch::UltraCondensed => Stretch::UltraCondensed,
        FontStretch::ExtraCondensed => Stretch::ExtraCondensed,
        FontStretch::Condensed => Stretch::Condensed,
        FontStretch::SemiCondensed => Stretch::SemiCondensed,
        FontStretch::Normal => Stretch::Normal,
        FontStretch::SemiExpanded => Stretch::SemiExpanded,
        FontStretch::Expanded => Stretch::Expanded,
        FontStretch::ExtraExpanded => Stretch::ExtraExpanded,
        FontStretch::UltraExpanded => Stretch::UltraExpanded,
    }
}

/// A laid-out text block: `width` is the widest line, `height` the sum of the
/// line heights.
#[derive(Debug)]
pub struct ShapedBlock {
    pub width: f32,
    pub height: f32,
    pub lines: Vec<ShapedLine>,
}

/// A rasterized glyph's coverage (or colour) bitmap plus its placement relative
/// to the glyph origin: `left` rightward, `top` upward from the baseline.
#[derive(Debug)]
pub struct GlyphImage<'a> {
    pub left: i32,
    pub top: i32,
    pub width: usize,
    pub height: usize,
    /// One byte per pixel for a coverage mask, four (RGBA) when `color`.
    pub data: &'a [u8],
    /// A colour bitmap (emoji) rather than an alpha mask.
    pub color: bool,
}

/// Split `text` into the `(slice, attrs)` pairs `set_rich_text` wants: the
/// ranges `styled` names carry their own attributes over `base`, the bytes
/// between them carry `base` itself. A range that runs backwards, is empty, or
/// lands inside a codepoint is skipped rather than shifting the text.
fn styled_attrs<'text, 'attrs>(
    text: &'text str,
    px: f32,
    line_height: f32,
    base: &Attrs<'attrs>,
    styled: &[StyledSpan<'attrs>],
) -> Vec<(&'text str, Attrs<'attrs>)> {
    let mut spans = Vec::new();
    let mut at = 0usize;
    for span in styled {
        let usable = span.start >= at
            && span.end > span.start
            && span.end <= text.len()
            && text.is_char_boundary(span.start)
            && text.is_char_boundary(span.end)
            && span.font_size.is_none_or(|size| size > 0.0)
            && span.styles_anything();
        if !usable {
            continue;
        }
        if at < span.start {
            spans.push((&text[at..span.start], base.clone()));
        }
        let mut attrs = base.clone();
        if let Some(size) = span.font_size {
            // Scale the line height with the size, so a larger span asks for a
            // proportionally taller line box (cosmic-text takes the line's
            // tallest).
            let scale = if px > 0.0 { size / px } else { 1.0 };
            attrs = attrs.metrics(Metrics::new(size, line_height * scale));
        }
        if let Some(weight) = span.weight {
            attrs = attrs.weight(Weight(weight));
        }
        if let Some(italic) = span.italic {
            attrs = attrs.style(if italic { Style::Italic } else { Style::Normal });
        }
        if let Some(stretch) = span.stretch {
            attrs = attrs.stretch(stretch_of(stretch));
        }
        if let Some(family) = span.family {
            attrs = attrs.family(family_of(family));
        }
        spans.push((&text[span.start..span.end], attrs));
        at = span.end;
    }
    if at < text.len() {
        spans.push((&text[at..], base.clone()));
    }
    spans
}

/// Shaping + rasterization state: the discovered font database and the glyph
/// raster cache.
#[derive(Debug)]
pub struct TextShaper {
    fonts: FontSystem,
    cache: SwashCache,
    /// Family name of the first explicitly loaded font, used as the requested
    /// family so an explicit `font=` stays the primary and the system faces only
    /// fill in codepoints it lacks. `None` asks for the generic sans-serif.
    primary: Option<String>,
}

impl TextShaper {
    /// Discover the system fonts and register `explicit` font files (in order)
    /// alongside them.
    pub fn new(explicit: &[Vec<u8>]) -> Self {
        let mut fonts = FontSystem::new();
        let mut primary = None;
        for bytes in explicit {
            let ids = fonts
                .db_mut()
                .load_font_source(cosmic_text::fontdb::Source::Binary(Arc::new(bytes.clone())));
            if primary.is_none() {
                primary = ids
                    .first()
                    .and_then(|id| fonts.db().face(*id))
                    .and_then(|face| face.families.first())
                    .map(|(name, _)| name.clone());
            }
        }
        Self {
            fonts,
            cache: SwashCache::new(),
            primary,
        }
    }

    /// Bytes + collection index of the face the generic sans-serif query
    /// resolves to, so the overlay's `ab_glyph` chain (vertical cues) can share
    /// the system-font discovery done here.
    pub fn default_face(&self) -> Option<(Vec<u8>, u32)> {
        let db = self.fonts.db();
        let query = cosmic_text::fontdb::Query {
            families: &[cosmic_text::fontdb::Family::SansSerif],
            ..cosmic_text::fontdb::Query::default()
        };
        let id = db.query(&query)?;
        let index = db.face(id)?.index;
        db.with_face_data(id, |data, _| (data.to_vec(), index))
    }

    /// Bytes + collection index of a discovered face with a real glyph for `c`,
    /// so the `ab_glyph` chain (vertical cues) can extend its fallback chain for
    /// a script the seeded sans-serif face lacks (CJK on a Latin default). Scans
    /// the database; the caller caches the hit by appending the face to its
    /// chain and remembers misses, so a codepoint is scanned at most once.
    pub fn face_for_char(&self, c: char) -> Option<(Vec<u8>, u32)> {
        use ab_glyph::Font;
        let db = self.fonts.db();
        for info in db.faces() {
            let covered = db.with_face_data(info.id, |data, index| {
                ab_glyph::FontRef::try_from_slice_and_index(data, index)
                    .map(|f| f.glyph_id(c).0 != 0)
                    .unwrap_or(false)
            });
            if covered == Some(true) {
                return db.with_face_data(info.id, |data, index| (data.to_vec(), index));
            }
        }
        None
    }

    /// Bytes + collection index of the face `id` names (the one a shaped glyph
    /// was resolved to), so a GPU backend can draw that glyph's outlines from
    /// the same face this shaper picked. Copies the whole font file, so callers
    /// keep the result rather than asking per frame.
    pub fn face_data(&self, id: FontId) -> Option<(Vec<u8>, u32)> {
        let db = self.fonts.db();
        let index = db.face(id)?.index;
        db.with_face_data(id, |data, _| (data.to_vec(), index))
    }

    /// Shape and lay out `text` at `px` (one visual line per logical line, no
    /// wrapping), optionally at variable-font weight `wght`. Ranges named in
    /// `styled` are laid out with their own size / weight / slant / width
    /// instead, which cosmic-text carries as per-span attributes on the line's
    /// `AttrsList`, so a line mixing them is still one shaped, bidi-reordered
    /// run and takes the tallest span's line height. A span's weight, slant or
    /// width selects a face from the font database, so a family with a real
    /// bold or italic face renders in it; a weight with no such face still
    /// reaches the `wght` variation axis, but a slant with no italic face
    /// renders upright (there is no synthetic oblique here). A span's family
    /// queries the database the same way, generic names included. Line positions are
    /// relative to the block's top-left.
    pub fn layout(
        &mut self,
        text: &str,
        px: f32,
        line_height: f32,
        wght: Option<f32>,
        styled: &[StyledSpan<'_>],
    ) -> ShapedBlock {
        let mut attrs = Attrs::new();
        if let Some(family) = &self.primary {
            attrs = attrs.family(Family::Name(family));
        }
        if let Some(w) = wght {
            attrs = attrs.weight(Weight(w.clamp(1.0, 1000.0) as u16));
        }
        let mut buffer = Buffer::new(&mut self.fonts, Metrics::new(px, line_height));
        // Cue lines are pre-broken by the subtitle format, so no wrapping and no
        // size limit: an over-wide line is clipped at blit time like the
        // ab_glyph path, rather than reflowed.
        buffer.set_wrap(&mut self.fonts, Wrap::None);
        buffer.set_size(&mut self.fonts, None, None);
        if styled.is_empty() {
            buffer.set_text(&mut self.fonts, text, &attrs, Shaping::Advanced, None);
        } else {
            let spans = styled_attrs(text, px, line_height, &attrs, styled);
            buffer.set_rich_text(&mut self.fonts, spans, &attrs, Shaping::Advanced, None);
        }
        buffer.shape_until_scroll(&mut self.fonts, false);

        let mut lines = Vec::new();
        let mut width = 0.0_f32;
        let mut height = 0.0_f32;
        for run in buffer.layout_runs() {
            let glyphs = run
                .glyphs
                .iter()
                .map(|g| {
                    let physical = g.physical((0.0, run.line_y), 1.0);
                    ShapedGlyph {
                        key: physical.cache_key,
                        x: physical.x,
                        y: physical.y,
                        start: g.start,
                        glyph_id: g.glyph_id,
                        rtl: g.level.is_rtl(),
                        font_size: g.font_size,
                        advance: g.w,
                    }
                })
                .collect();
            width = width.max(run.line_w);
            height = height.max(run.line_top + run.line_height);
            lines.push(ShapedLine {
                width: run.line_w,
                top: run.line_top,
                height: run.line_height,
                baseline: run.line_y,
                glyphs,
            });
        }
        ShapedBlock {
            width,
            height,
            lines,
        }
    }

    /// Whether any face was discovered at all, so a test can skip on a host
    /// with no fonts installed.
    #[cfg(test)]
    fn has_faces(&self) -> bool {
        !self.fonts.db().is_empty()
    }

    /// Rasterize (or fetch from the cache) one shaped glyph. `None` for a glyph
    /// with no bitmap (a space) or a subpixel mask, which this path never asks
    /// for.
    pub fn image(&mut self, key: GlyphKey) -> Option<GlyphImage<'_>> {
        let image = self.cache.get_image(&mut self.fonts, key).as_ref()?;
        let color = match image.content {
            SwashContent::Mask => false,
            SwashContent::Color => true,
            SwashContent::SubpixelMask => return None,
        };
        Some(GlyphImage {
            left: image.placement.left,
            top: image.placement.top,
            width: image.placement.width as usize,
            height: image.placement.height as usize,
            data: &image.data,
            color,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A styled span reaches the shaped glyphs: the run it covers is shaped at
    /// its own weight (which the raster key carries, so swash scales the `wght`
    /// axis for it), the bytes outside it at the block's.
    #[test]
    fn a_styled_span_shapes_its_own_run_at_its_own_weight() {
        let mut shaper = TextShaper::new(&[]);
        if !shaper.has_faces() {
            std::eprintln!("no system font; skipping");
            return;
        }
        let px = 32.0;
        let block = shaper.layout(
            "ab",
            px,
            px * 1.25,
            None,
            &[StyledSpan {
                start: 0,
                end: 1,
                weight: Some(700),
                ..StyledSpan::default()
            }],
        );
        let line = block.lines.first().expect("one line");
        assert_eq!(line.glyphs.len(), 2, "one glyph per character");
        assert_eq!(line.glyphs[0].key.font_weight.0, 700);
        assert_eq!(line.glyphs[1].key.font_weight.0, Weight::NORMAL.0);
        // The baseline sits inside the line box, which is what an underline is
        // measured down from.
        assert!(line.baseline > line.top && line.baseline < line.top + line.height);
    }
}
