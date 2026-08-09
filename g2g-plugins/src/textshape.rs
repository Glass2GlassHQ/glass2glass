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
    Attrs, Buffer, Family, FontSystem, Metrics, Shaping, SwashCache, SwashContent, Weight, Wrap,
};

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
}

/// One visual line of a laid-out block, in visual (post-bidi) order.
#[derive(Debug)]
pub struct ShapedLine {
    pub width: f32,
    pub glyphs: Vec<ShapedGlyph>,
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
    /// wrapping), optionally at variable-font weight `wght`. Line positions are
    /// relative to the block's top-left.
    pub fn layout(
        &mut self,
        text: &str,
        px: f32,
        line_height: f32,
        wght: Option<f32>,
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
        buffer.set_text(&mut self.fonts, text, &attrs, Shaping::Advanced, None);
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
                    }
                })
                .collect();
            width = width.max(run.line_w);
            height = height.max(run.line_top + run.line_height);
            lines.push(ShapedLine {
                width: run.line_w,
                glyphs,
            });
        }
        ShapedBlock {
            width,
            height,
            lines,
        }
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
