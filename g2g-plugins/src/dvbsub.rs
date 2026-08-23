//! DVB subtitle (ETSI EN 300 743) decoding: the pure half behind
//! [`DvbSubDec`](crate::dvbsubdec::DvbSubDec).
//!
//! Unlike a VobSub cue, which is one self-contained packet, a DVB subtitle is a
//! *segment stream*: each PES data field (or Matroska block) carries a display
//! set, a run of segments sharing a `page_id`, and the decoder keeps state
//! across them. A display set ends with an end-of-display-set segment.
//!
//! - display definition (0x14): the display geometry, and optionally the window
//!   the regions are placed inside. Absent, the display is 720x576.
//! - page composition (0x10): the page timeout, the page state, and where each
//!   region sits on the display. A page composition listing no region is how a
//!   cue ends.
//! - region composition (0x11): a region's size, bit depth, CLUT and background,
//!   and which objects are drawn into it where.
//! - CLUT definition (0x12): palette entries at 2-, 4- or 8-bit depth, as
//!   Y / Cr / Cb / transparency, converted here to RGBA.
//! - object data (0x13): run-length coded pixels in two interlaced fields, coded
//!   at 2, 4 or 8 bits per pixel with optional map tables lifting a shallower
//!   code into the region's depth.
//!
//! The data field a transport stream carries is prefixed with a data_identifier
//! (0x20) and a subtitle_stream_id; a Matroska block carries the bare segments.
//! [`DvbSubDecoder::feed`] takes either.
//!
//! Every length, count, coordinate and dimension here comes off the wire, so
//! each is range-checked before use and a malformed display set is dropped
//! rather than panicking or allocating on a bogus size.

use alloc::vec;
use alloc::vec::Vec;

/// Largest display or region edge accepted. DVB display sizes are 16-bit fields,
/// so this bounds a crafted geometry well above the 1920x1080 real streams use.
pub const MAX_DISPLAY_DIM: u32 = 4096;

/// Largest region bitmap accepted, in pixels: enough for a full-screen HD region
/// without letting a crafted 65535x65535 region ask for a 4 GB allocation.
pub const MAX_REGION_PIXELS: usize = 2048 * 2048;

/// Display width a stream with no display definition segment runs at
/// (EN 300 743: SD 720x576).
pub const DEFAULT_DISPLAY_WIDTH: u32 = 720;
/// Display height for the same case.
pub const DEFAULT_DISPLAY_HEIGHT: u32 = 576;

/// Leads every segment (EN 300 743 clause 7.2).
const SYNC_BYTE: u8 = 0x0F;
/// Leads a PES data field, before the subtitle_stream_id byte (clause 7.1).
const DATA_IDENTIFIER: u8 = 0x20;

const SEG_PAGE_COMPOSITION: u8 = 0x10;
const SEG_REGION_COMPOSITION: u8 = 0x11;
const SEG_CLUT_DEFINITION: u8 = 0x12;
const SEG_OBJECT_DATA: u8 = 0x13;
const SEG_DISPLAY_DEFINITION: u8 = 0x14;

/// Pixel-data sub-block data types (clause 7.2.5.1).
const PIX_2BIT: u8 = 0x10;
const PIX_4BIT: u8 = 0x11;
const PIX_8BIT: u8 = 0x12;
const MAP_2_TO_4: u8 = 0x20;
const MAP_2_TO_8: u8 = 0x21;
const MAP_4_TO_8: u8 = 0x22;
const END_OF_OBJECT_LINE: u8 = 0xF0;

/// The page ids a subtitle stream's out-of-band configuration names: the page
/// this decoder composes, and the page whose regions / CLUTs / objects it may
/// also share.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageIds {
    pub composition: u16,
    pub ancillary: u16,
}

/// Parse the out-of-band page-id blob: a Matroska `S_DVBSUB` `CodecPrivate`, or
/// the same five bytes synthesized from a PMT `subtitling_descriptor`. The layout
/// is five bytes per substream (composition page id, ancillary page id, then the
/// subtitling type), which is what ffmpeg writes and reads; a four-byte blob
/// without the type byte is accepted too.
///
/// Config and display sets share one pad, so this also has to say which a blob
/// is: a display set always leads with the data identifier or the segment sync
/// byte and is at least a data-field header plus one segment header long, and a
/// blob that looks like one is refused here. Only a multi-substream blob whose
/// first composition page id happens to be `0x0Fxx` or `0x20xx` is ambiguous.
pub fn parse_page_ids(bytes: &[u8]) -> Option<PageIds> {
    if bytes.len() != 4 && (bytes.len() < 5 || !bytes.len().is_multiple_of(5)) {
        return None;
    }
    if bytes.len() >= 8 && matches!(bytes[0], SYNC_BYTE | DATA_IDENTIFIER) {
        return None;
    }
    Some(PageIds {
        composition: u16::from_be_bytes([bytes[0], bytes[1]]),
        ancillary: u16::from_be_bytes([bytes[2], bytes[3]]),
    })
}

/// The five-byte page-id blob for `ids`, the form [`parse_page_ids`] reads. The
/// MPEG-TS demuxer builds one from the `subtitling_descriptor` so both carriages
/// hand the decoder the same configuration.
pub fn page_id_blob(ids: PageIds, subtitling_type: u8) -> [u8; 5] {
    let c = ids.composition.to_be_bytes();
    let a = ids.ancillary.to_be_bytes();
    [c[0], c[1], a[0], a[1], subtitling_type]
}

/// The `subtitling_type` a muxer writes for a stream that names none: EN 300 468
/// 0x10, "DVB subtitles (normal) with no monitor aspect ratio criticality", which
/// is what ffmpeg writes.
pub const DEFAULT_SUBTITLING_TYPE: u8 = 0x10;

/// The composition / ancillary page id a muxer writes for a stream that names
/// none, again ffmpeg's default.
pub const DEFAULT_PAGE_ID: u16 = 1;

/// The segments of a display set, without the PES data-field header ahead of
/// them or the end marker and stuffing behind: the form a Matroska `S_DVBSUB`
/// block carries. Accepts either carriage, so a display set read off a transport
/// stream and one read off a Matroska block both reduce to the same bytes.
///
/// The span is found by walking the segment headers, so a truncated or corrupt
/// display set yields the segments that do hold together rather than panicking.
pub fn segment_span(data: &[u8]) -> &[u8] {
    let start = match data.first() {
        Some(&DATA_IDENTIFIER) if data.len() >= 2 => 2,
        _ => 0,
    };
    let mut end = start;
    while data.get(end) == Some(&SYNC_BYTE) {
        let Some(header) = data.get(end..end + 6) else {
            break;
        };
        let length = u16::from_be_bytes([header[4], header[5]]) as usize;
        let next = end.saturating_add(6).saturating_add(length);
        if next > data.len() {
            break;
        }
        end = next;
    }
    &data[start..end]
}

/// A display set wrapped in the PES data field a transport stream carries it in
/// (EN 300 743 clause 7.1): the data_identifier and subtitle_stream_id ahead of
/// the segments, the end-of-data-field marker behind. Takes either carriage, so
/// re-wrapping a data field that already has the header is a no-op.
pub fn pes_data_field(data: &[u8]) -> Vec<u8> {
    let mut out = vec![DATA_IDENTIFIER, 0x00];
    out.extend_from_slice(segment_span(data));
    out.push(0xFF);
    out
}

/// One composed display set: the page as it should now look.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Page {
    /// `page_time_out`, seconds after which a receiver removes the page even
    /// with no further display set.
    pub timeout_s: u8,
    /// Whether the page composition listed any region. `false` is the stream
    /// saying the cue has ended, and `canvas` is then fully transparent.
    pub visible: bool,
    /// The display, `width * height * 4` RGBA bytes, transparent where no region
    /// paints.
    pub canvas: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// One region's placement on the display, from the page composition segment.
#[derive(Debug, Clone, Copy)]
struct PageRegion {
    id: u8,
    x: u32,
    y: u32,
}

/// An object drawn into a region, from the region composition segment.
#[derive(Debug, Clone, Copy)]
struct RegionObject {
    id: u16,
    x: u32,
    y: u32,
}

/// A region: its own pixel-code bitmap, plus which CLUT resolves those codes.
#[derive(Debug, Clone)]
struct Region {
    id: u8,
    width: u32,
    height: u32,
    /// 2, 4 or 8 bits per pixel.
    depth: u8,
    clut_id: u8,
    /// `width * height` pixel codes, row-major.
    pixels: Vec<u8>,
    objects: Vec<RegionObject>,
}

/// A colour look-up table, held at all three depths because a region names a
/// CLUT by id and reads it at its own depth.
#[derive(Debug, Clone)]
struct Clut {
    id: u8,
    clut2: [u32; 4],
    clut4: [u32; 16],
    clut8: [u32; 256],
}

impl Clut {
    /// The default CLUT of EN 300 743 clause 10, which stands for a region whose
    /// CLUT id no definition segment has filled in.
    fn default_for(id: u8) -> Self {
        let mut clut2 = [0u32; 4];
        clut2[1] = rgba(255, 255, 255, 255);
        clut2[2] = rgba(0, 0, 0, 255);
        clut2[3] = rgba(127, 127, 127, 255);

        let mut clut4 = [0u32; 16];
        for (i, e) in clut4.iter_mut().enumerate().skip(1) {
            let v = if i < 8 { 255 } else { 127 };
            let bit = |m: usize| if i & m != 0 { v } else { 0 };
            *e = rgba(bit(1), bit(2), bit(4), 255);
        }

        let mut clut8 = [0u32; 256];
        for (i, e) in clut8.iter_mut().enumerate().skip(1) {
            let bit = |m: usize, lo: u8, hi: u8| {
                (if i & m != 0 { lo } else { 0 }) + (if i & (m << 4) != 0 { hi } else { 0 })
            };
            let (r, g, b, a) = if i < 8 {
                let one = |m: usize| if i & m != 0 { 255 } else { 0 };
                (one(1), one(2), one(4), 63)
            } else {
                match i & 0x88 {
                    0x00 => (bit(1, 85, 170), bit(2, 85, 170), bit(4, 85, 170), 255),
                    0x08 => (bit(1, 85, 170), bit(2, 85, 170), bit(4, 85, 170), 128),
                    0x80 => (
                        127 + bit(1, 43, 85),
                        127 + bit(2, 43, 85),
                        127 + bit(4, 43, 85),
                        255,
                    ),
                    _ => (bit(1, 43, 85), bit(2, 43, 85), bit(4, 43, 85), 255),
                }
            };
            *e = rgba(r, g, b, a);
        }

        Self {
            id,
            clut2,
            clut4,
            clut8,
        }
    }

    /// The RGBA a pixel code resolves to at `depth` bits.
    fn lookup(&self, depth: u8, code: u8) -> u32 {
        match depth {
            2 => self.clut2[(code & 3) as usize],
            4 => self.clut4[(code & 0x0f) as usize],
            _ => self.clut8[code as usize],
        }
    }
}

/// Pack a colour as `0xAARRGGBB`, the form [`Page::canvas`] unpacks to RGBA bytes.
fn rgba(r: u8, g: u8, b: u8, a: u8) -> u32 {
    ((a as u32) << 24) | ((r as u32) << 16) | ((g as u32) << 8) | b as u32
}

/// BT.601 limited-range Y'CbCr to RGB. DVB subtitles carry no colorimetry and
/// are an SD-era format, so the matrix is always BT.601.
fn ycbcr_to_rgb(y: u8, cr: u8, cb: u8) -> (u8, u8, u8) {
    crate::paint::ycbcr_to_rgb(y, cr, cb, crate::paint::YcbcrMatrix::Bt601)
}

/// One segment of a display set.
#[derive(Debug, Clone, Copy)]
struct Segment<'a> {
    kind: u8,
    page_id: u16,
    body: &'a [u8],
}

/// Split a subtitle data field into its segments, skipping an optional PES
/// data-field header (data_identifier + subtitle_stream_id) and any trailing
/// stuffing / end marker. `None` when a segment header or declared length does
/// not fit what is left, so a truncated display set is dropped whole rather than
/// half applied.
fn segments(data: &[u8]) -> Option<Vec<Segment<'_>>> {
    let mut rest = match data.first() {
        Some(&DATA_IDENTIFIER) => data.get(2..)?,
        _ => data,
    };
    let mut out = Vec::new();
    loop {
        match rest.first() {
            // 0xFF is the end-of-data-field marker, and stuffing after it.
            None | Some(&0xFF) => return Some(out),
            Some(&SYNC_BYTE) => {}
            Some(_) => return None,
        }
        let header = rest.get(..6)?;
        let length = u16::from_be_bytes([header[4], header[5]]) as usize;
        let body = rest.get(6..6 + length)?;
        out.push(Segment {
            kind: header[1],
            page_id: u16::from_be_bytes([header[2], header[3]]),
            body,
        });
        rest = &rest[6 + length..];
    }
}

/// The EN 300 743 decoder: display state built up across display sets.
#[derive(Debug)]
pub struct DvbSubDecoder {
    width: u32,
    height: u32,
    /// Origin of the display window inside the display, when the display
    /// definition segment declared one. Regions are placed relative to it.
    origin_x: u32,
    origin_y: u32,
    /// The composition page this decoder composes. `None` until the first page
    /// composition segment latches one (or [`select_pages`](Self::select_pages)
    /// pins it).
    page: Option<u16>,
    ancillary: Option<u16>,
    regions: Vec<Region>,
    cluts: Vec<Clut>,
    page_regions: Vec<PageRegion>,
}

impl Default for DvbSubDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl DvbSubDecoder {
    pub fn new() -> Self {
        Self {
            width: DEFAULT_DISPLAY_WIDTH,
            height: DEFAULT_DISPLAY_HEIGHT,
            origin_x: 0,
            origin_y: 0,
            page: None,
            ancillary: None,
            regions: Vec::new(),
            cluts: Vec::new(),
            page_regions: Vec::new(),
        }
    }

    /// Compose the given page instead of the first one seen. Segments on any
    /// other page (bar the ancillary one, whose regions / CLUTs / objects are
    /// shared) are ignored.
    pub fn select_pages(&mut self, ids: PageIds) {
        self.page = Some(ids.composition);
        self.ancillary = Some(ids.ancillary).filter(|a| *a != ids.composition);
    }

    /// Set the display geometry to use until a display definition segment names
    /// another.
    pub fn set_display_size(&mut self, width: u32, height: u32) {
        self.width = width.clamp(1, MAX_DISPLAY_DIM);
        self.height = height.clamp(1, MAX_DISPLAY_DIM);
    }

    /// The display geometry the next [`Page`] will be composed at.
    pub fn display_size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// The composition page this decoder is following, once one is known.
    pub fn page_id(&self) -> Option<u16> {
        self.page
    }

    /// Feed one subtitle data field (with or without its PES data-field header).
    /// Returns the composed page when the data field carried a page composition
    /// segment for the selected page, `None` when it carried none or the segment
    /// layer did not hold together.
    pub fn feed(&mut self, data: &[u8]) -> Option<Page> {
        let segments = segments(data)?;
        let mut timeout_s = None;
        for seg in segments {
            if !self.accepts(seg.page_id) {
                continue;
            }
            match seg.kind {
                SEG_DISPLAY_DEFINITION => self.apply_display_definition(seg.body),
                // A page composition on the ancillary page is not this page's.
                SEG_PAGE_COMPOSITION if self.page.is_none() || self.page == Some(seg.page_id) => {
                    if let Some(t) = self.apply_page_composition(seg.body) {
                        self.page = Some(seg.page_id);
                        timeout_s = Some(t);
                    }
                }
                SEG_REGION_COMPOSITION => self.apply_region_composition(seg.body),
                SEG_CLUT_DEFINITION => self.apply_clut_definition(seg.body),
                SEG_OBJECT_DATA => self.apply_object_data(seg.body),
                // end of display set (0x80) and any reserved / future type
                _ => {}
            }
        }
        timeout_s.map(|timeout_s| self.compose(timeout_s))
    }

    /// Whether a segment's `page_id` belongs to this decoder: everything until a
    /// page is latched, then the composition page and its ancillary page.
    fn accepts(&self, page_id: u16) -> bool {
        match self.page {
            None => true,
            Some(p) => page_id == p || Some(page_id) == self.ancillary,
        }
    }

    /// Paint the page's regions onto a transparent display canvas.
    fn compose(&self, timeout_s: u8) -> Page {
        let stride = self.width as usize * 4;
        let mut canvas = vec![0u8; stride * self.height as usize];
        for placed in &self.page_regions {
            let Some(region) = self.regions.iter().find(|r| r.id == placed.id) else {
                continue;
            };
            // A region naming a CLUT no definition segment filled in reads the
            // standard's default one.
            let fallback;
            let clut = match self.cluts.iter().find(|c| c.id == region.clut_id) {
                Some(c) => c,
                None => {
                    fallback = Clut::default_for(region.clut_id);
                    &fallback
                }
            };
            let x0 = placed.x.saturating_add(self.origin_x);
            let y0 = placed.y.saturating_add(self.origin_y);
            for row in 0..region.height {
                let cy = y0.saturating_add(row);
                if cy >= self.height {
                    break;
                }
                for col in 0..region.width {
                    let cx = x0.saturating_add(col);
                    if cx >= self.width {
                        break;
                    }
                    let code = region.pixels[(row * region.width + col) as usize];
                    let px = clut.lookup(region.depth, code);
                    let src = [
                        (px >> 16) as u8,
                        (px >> 8) as u8,
                        px as u8,
                        (px >> 24) as u8,
                    ];
                    if src[3] == 0 {
                        continue;
                    }
                    let at = (cy * self.width + cx) as usize * 4;
                    crate::paint::over_px(&mut canvas, at, src);
                }
            }
        }
        Page {
            timeout_s,
            visible: !self.page_regions.is_empty(),
            canvas,
            width: self.width,
            height: self.height,
        }
    }

    /// Display definition segment (clause 7.2.6): the display geometry, and the
    /// window origin regions are placed against when one is declared. The coded
    /// values are maximum indices, so the size is one more.
    fn apply_display_definition(&mut self, body: &[u8]) {
        let Some(head) = body.get(..5) else {
            return;
        };
        let width = u16::from_be_bytes([head[1], head[2]]) as u32 + 1;
        let height = u16::from_be_bytes([head[3], head[4]]) as u32 + 1;
        if width > MAX_DISPLAY_DIM || height > MAX_DISPLAY_DIM {
            return;
        }
        self.width = width;
        self.height = height;
        self.origin_x = 0;
        self.origin_y = 0;
        if head[0] & 0x08 == 0 {
            return; // no display_window_flag: the window is the whole display
        }
        let Some(window) = body.get(5..13) else {
            return;
        };
        self.origin_x = u16::from_be_bytes([window[0], window[1]]) as u32;
        self.origin_y = u16::from_be_bytes([window[4], window[5]]) as u32;
    }

    /// Page composition segment (clause 7.2.1): the timeout, the page state, and
    /// the region list. Returns the timeout, or `None` when the segment is too
    /// short to be one. A page state of acquisition point or mode change drops
    /// the state carried over from the previous page.
    fn apply_page_composition(&mut self, body: &[u8]) -> Option<u8> {
        let head = body.get(..2)?;
        let timeout_s = head[0];
        let page_state = (head[1] >> 2) & 3;
        if page_state == 1 || page_state == 2 {
            self.regions.clear();
            self.cluts.clear();
        }
        self.page_regions.clear();
        for entry in body[2..].as_chunks::<6>().0 {
            self.page_regions.push(PageRegion {
                id: entry[0],
                x: u16::from_be_bytes([entry[2], entry[3]]) as u32,
                y: u16::from_be_bytes([entry[4], entry[5]]) as u32,
            });
        }
        Some(timeout_s)
    }

    /// Region composition segment (clause 7.2.2): the region's geometry, depth,
    /// CLUT and background, then the objects drawn into it. A geometry change
    /// reallocates (and so clears) the region bitmap.
    fn apply_region_composition(&mut self, body: &[u8]) {
        let Some(head) = body.get(..10) else {
            return;
        };
        let id = head[0];
        let fill = head[1] & 0x08 != 0;
        let width = u16::from_be_bytes([head[2], head[3]]) as u32;
        let height = u16::from_be_bytes([head[4], head[5]]) as u32;
        // `region_depth` codes 2 / 4 / 8 bits as 1 / 2 / 3; anything else is
        // outside the standard and the region is dropped rather than guessed.
        let depth = match (head[6] >> 2) & 7 {
            1 => 2u8,
            2 => 4,
            3 => 8,
            _ => return,
        };
        let clut_id = head[7];
        // The background pixel code sits at the region's own depth: the 8-bit
        // code, then the 4-bit code in the high nibble of the next byte and the
        // 2-bit code in its bits 3..2.
        let background = match depth {
            8 => head[8],
            4 => head[9] >> 4,
            _ => (head[9] >> 2) & 3,
        };
        if width == 0
            || height == 0
            || width > MAX_DISPLAY_DIM
            || height > MAX_DISPLAY_DIM
            || (width as usize).saturating_mul(height as usize) > MAX_REGION_PIXELS
        {
            return;
        }

        let count = width as usize * height as usize;
        let mut objects = Vec::new();
        for entry in body[10..].as_chunks::<6>().0 {
            let x = (u16::from_be_bytes([entry[2], entry[3]]) & 0x0fff) as u32;
            let y = (u16::from_be_bytes([entry[4], entry[5]]) & 0x0fff) as u32;
            if x >= width || y >= height {
                continue; // an object outside its own region draws nothing
            }
            objects.push(RegionObject {
                id: u16::from_be_bytes([entry[0], entry[1]]),
                x,
                y,
            });
        }

        match self.regions.iter_mut().find(|r| r.id == id) {
            Some(region) => {
                if region.pixels.len() != count {
                    region.pixels = vec![background; count];
                } else if fill {
                    region.pixels.fill(background);
                }
                region.width = width;
                region.height = height;
                region.depth = depth;
                region.clut_id = clut_id;
                region.objects = objects;
            }
            None => self.regions.push(Region {
                id,
                width,
                height,
                depth,
                clut_id,
                pixels: vec![background; count],
                objects,
            }),
        }
    }

    /// CLUT definition segment (clause 7.2.3): palette entries as Y / Cr / Cb and
    /// a transparency, at full 8-bit precision or the packed two-byte form.
    fn apply_clut_definition(&mut self, body: &[u8]) {
        let Some(&id) = body.first() else {
            return;
        };
        if !self.cluts.iter().any(|c| c.id == id) {
            self.cluts.push(Clut::default_for(id));
        }
        let Some(clut) = self.cluts.iter_mut().find(|c| c.id == id) else {
            return;
        };
        let Some(mut rest) = body.get(2..) else {
            return;
        };
        while rest.len() >= 2 {
            let entry_id = rest[0];
            let depths = rest[1] & 0xE0;
            let full_range = rest[1] & 1 != 0;
            let (y, cr, cb, transparency) = if full_range {
                let Some(v) = rest.get(2..6) else {
                    return;
                };
                rest = &rest[6..];
                (v[0], v[1], v[2], v[3])
            } else {
                let Some(v) = rest.get(2..4) else {
                    return;
                };
                rest = &rest[4..];
                (
                    v[0] & 0xFC,
                    ((v[0] & 3) << 2 | (v[1] >> 6) & 3) << 4,
                    (v[1] << 2) & 0xF0,
                    (v[1] << 6) & 0xC0,
                )
            };
            // A zero luma is the standard's fully transparent entry, whatever
            // the chroma says.
            let (y, cr, cb, transparency) = if y == 0 {
                (0, 128, 128, 0xFF)
            } else {
                (y, cr, cb, transparency)
            };
            let (r, g, b) = ycbcr_to_rgb(y, cr, cb);
            let colour = rgba(r, g, b, 255 - transparency);
            // The flags say which depths this entry belongs to; the shallowest
            // claimed wins, as in the reference decoder.
            if depths & 0x80 != 0 {
                if let Some(slot) = clut.clut2.get_mut(entry_id as usize) {
                    *slot = colour;
                }
            } else if depths & 0x40 != 0 {
                if let Some(slot) = clut.clut4.get_mut(entry_id as usize) {
                    *slot = colour;
                }
            } else if depths & 0x20 != 0 {
                clut.clut8[entry_id as usize] = colour;
            }
        }
    }

    /// Object data segment (clause 7.2.4): run-length coded pixels for one
    /// object, decoded into every region that placed it. Only the pixel coding
    /// method is decoded; a character-string object carries no bitmap.
    fn apply_object_data(&mut self, body: &[u8]) {
        let Some(head) = body.get(..3) else {
            return;
        };
        let object_id = u16::from_be_bytes([head[0], head[1]]);
        if (head[2] >> 2) & 3 != 0 {
            return; // coding method other than pixels
        }
        let non_modifying = (head[2] >> 1) & 1 != 0;
        let Some(lengths) = body.get(3..7) else {
            return;
        };
        let top_len = u16::from_be_bytes([lengths[0], lengths[1]]) as usize;
        let bottom_len = u16::from_be_bytes([lengths[2], lengths[3]]) as usize;
        let Some(top) = body.get(7..7 + top_len) else {
            return;
        };
        let Some(bottom) = body.get(7 + top_len..7 + top_len + bottom_len) else {
            return;
        };
        // A zero-length bottom field means the object is coded once and both
        // fields read the same data.
        let bottom = if bottom_len == 0 { top } else { bottom };

        for region in &mut self.regions {
            let placements: Vec<RegionObject> = region
                .objects
                .iter()
                .copied()
                .filter(|o| o.id == object_id)
                .collect();
            for at in placements {
                draw_object(region, at, top, 0, non_modifying);
                draw_object(region, at, bottom, 1, non_modifying);
            }
        }
    }
}

/// Decode one field's pixel-data sub-blocks into `region`, starting at the
/// object's position and stepping two rows per coded line.
fn draw_object(
    region: &mut Region,
    at: RegionObject,
    data: &[u8],
    field: u32,
    non_modifying: bool,
) {
    let mut map2to4 = [0x00u8, 0x07, 0x08, 0x0f];
    let mut map2to8 = [0x00u8, 0x77, 0x88, 0xff];
    let mut map4to8 = [
        0x00u8, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
        0xff,
    ];
    let width = region.width;
    let mut x = at.x;
    let mut y = at.y + field;
    let mut rest = data;
    while let Some((&kind, tail)) = rest.split_first() {
        rest = tail;
        if kind == END_OF_OBJECT_LINE {
            x = at.x;
            y += 2;
            continue;
        }
        // A map table is a fixed-size literal, not a coded string.
        let table_len = match kind {
            MAP_2_TO_4 => Some(2),
            MAP_2_TO_8 => Some(4),
            MAP_4_TO_8 => Some(16),
            _ => None,
        };
        if let Some(len) = table_len {
            let Some(table) = rest.get(..len) else {
                return;
            };
            rest = &rest[len..];
            match kind {
                MAP_2_TO_4 => {
                    for (i, slot) in map2to4.iter_mut().enumerate() {
                        *slot = (table[i / 2] >> (4 - 4 * (i % 2))) & 0x0f;
                    }
                }
                MAP_2_TO_8 => map2to8.copy_from_slice(table),
                _ => map4to8.copy_from_slice(table),
            }
            continue;
        }
        if y >= region.height || x >= width {
            return; // a coded line past the region: the object does not fit
        }
        let start = (y * width + x) as usize;
        let end = ((y + 1) * width) as usize;
        let Some(row) = region.pixels.get_mut(start..end) else {
            return;
        };
        let (written, consumed) = match kind {
            PIX_2BIT => {
                let map: Option<&[u8]> = match region.depth {
                    4 => Some(&map2to4),
                    8 => Some(&map2to8),
                    _ => None,
                };
                read_pixel_string(2, rest, row, map, non_modifying)
            }
            PIX_4BIT if region.depth >= 4 => {
                let map: Option<&[u8]> = (region.depth == 8).then_some(&map4to8);
                read_pixel_string(4, rest, row, map, non_modifying)
            }
            PIX_8BIT if region.depth >= 8 => read_pixel_string(8, rest, row, None, non_modifying),
            // a string coded deeper than its region, or a reserved data type
            _ => return,
        };
        x += written as u32;
        rest = &rest[consumed.min(rest.len())..];
    }
}

/// A big-endian bit reader over a coded pixel string.
struct Bits<'a> {
    data: &'a [u8],
    at: usize,
}

impl Bits<'_> {
    fn take(&mut self, n: u32) -> Option<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            let byte = *self.data.get(self.at / 8)?;
            v = (v << 1) | ((byte >> (7 - self.at % 8)) & 1) as u32;
            self.at += 1;
        }
        Some(v)
    }

    /// Bytes consumed, rounding the trailing stuffing bits up to the byte the
    /// next sub-block starts on.
    fn bytes_consumed(&self) -> usize {
        self.at.div_ceil(8)
    }
}

/// Decode one 2-, 4- or 8-bit pixel code string (clause 7.2.5.2) into `row`,
/// which is the remainder of the region row from the current position. Returns
/// the pixels written and the bytes of `data` consumed.
fn read_pixel_string(
    depth: u32,
    data: &[u8],
    row: &mut [u8],
    map: Option<&[u8]>,
    non_modifying: bool,
) -> (usize, usize) {
    let mut bits = Bits { data, at: 0 };
    let mut written = 0usize;
    while let Some((run, code)) = match depth {
        2 => next_2bit(&mut bits),
        4 => next_4bit(&mut bits),
        _ => next_8bit(&mut bits),
    } {
        if run == 0 {
            break; // end of the coded string
        }
        // "non modifying" means code 1 leaves the region's own pixels alone.
        if !(non_modifying && code == 1) {
            let value = map
                .and_then(|m| m.get(code as usize).copied())
                .unwrap_or(code);
            // A run overhanging the row is clipped, not an error: the row bound
            // is the region's, and decoding continues so the byte position of
            // the next sub-block stays right.
            let start = written.min(row.len());
            let end = written.saturating_add(run).min(row.len());
            row[start..end].fill(value);
        }
        written = written.saturating_add(run);
    }
    (written.min(row.len()), bits.bytes_consumed())
}

/// One 2-bit/pixel code string entry: a run length and its pixel code, or `None`
/// when the string is truncated. A zero run is the end-of-string code.
fn next_2bit(bits: &mut Bits) -> Option<(usize, u8)> {
    let first = bits.take(2)?;
    if first != 0 {
        return Some((1, first as u8));
    }
    if bits.take(1)? == 1 {
        let run = bits.take(3)? as usize + 3;
        return Some((run, bits.take(2)? as u8));
    }
    if bits.take(1)? == 1 {
        return Some((1, 0));
    }
    Some(match bits.take(2)? {
        0 => (0, 0),
        1 => (2, 0),
        2 => (bits.take(4)? as usize + 12, bits.take(2)? as u8),
        _ => (bits.take(8)? as usize + 29, bits.take(2)? as u8),
    })
}

/// One 4-bit/pixel code string entry, same contract as [`next_2bit`].
fn next_4bit(bits: &mut Bits) -> Option<(usize, u8)> {
    let first = bits.take(4)?;
    if first != 0 {
        return Some((1, first as u8));
    }
    if bits.take(1)? == 0 {
        // a zero run length here is the end-of-string code
        let run = bits.take(3)? as usize;
        return Some(if run == 0 { (0, 0) } else { (run + 2, 0) });
    }
    if bits.take(1)? == 0 {
        let run = bits.take(2)? as usize + 4;
        return Some((run, bits.take(4)? as u8));
    }
    Some(match bits.take(2)? {
        0 => (1, 0),
        1 => (2, 0),
        2 => (bits.take(4)? as usize + 9, bits.take(4)? as u8),
        _ => (bits.take(8)? as usize + 25, bits.take(4)? as u8),
    })
}

/// One 8-bit/pixel code string entry, same contract as [`next_2bit`].
fn next_8bit(bits: &mut Bits) -> Option<(usize, u8)> {
    let first = bits.take(8)?;
    if first != 0 {
        return Some((1, first as u8));
    }
    if bits.take(1)? == 1 {
        let run = bits.take(7)? as usize;
        return Some((run, bits.take(8)? as u8));
    }
    Some((bits.take(7)? as usize, 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Wrap a segment payload in its 6-byte header.
    fn seg(kind: u8, page_id: u16, body: &[u8]) -> Vec<u8> {
        let mut out = Vec::from([SYNC_BYTE, kind]);
        out.extend_from_slice(&page_id.to_be_bytes());
        out.extend_from_slice(&(body.len() as u16).to_be_bytes());
        out.extend_from_slice(body);
        out
    }

    /// A page composition placing region 0 at (`x`, `y`) with a 30 s timeout.
    fn page_segment(page_id: u16, regions: &[(u8, u16, u16)]) -> Vec<u8> {
        let mut body = Vec::from([30u8, 0x08]); // timeout 30, version 0, state 2
        for (id, x, y) in regions {
            body.push(*id);
            body.push(0xff);
            body.extend_from_slice(&x.to_be_bytes());
            body.extend_from_slice(&y.to_be_bytes());
        }
        seg(SEG_PAGE_COMPOSITION, page_id, &body)
    }

    /// A 4-bit region `w` x `h` holding object 0 at its origin.
    fn region_segment(page_id: u16, w: u16, h: u16) -> Vec<u8> {
        let body = [
            0x00, // region_id
            0x07, // version 0, fill 0
            (w >> 8) as u8,
            w as u8,
            (h >> 8) as u8,
            h as u8,
            0x4b, // level 2, depth 2 (4-bit)
            0x00, // CLUT id
            0x00, // 8-bit background
            0x03, // 4-bit background 0, 2-bit background 0
            0x00,
            0x00, // object id 0
            0x00,
            0x00, // type 0, provider 0, x 0
            0xf0,
            0x00, // y 0
        ];
        seg(SEG_REGION_COMPOSITION, page_id, &body)
    }

    /// A CLUT whose 4-bit entry 1 is opaque and entry 0 transparent.
    fn clut_segment(page_id: u16) -> Vec<u8> {
        clut_segment_t(page_id, 0x00)
    }

    /// As [`clut_segment`], with entry 1 carrying transparency `t` (0 opaque,
    /// 255 clear), so a test can paint a translucent cue.
    fn clut_segment_t(page_id: u16, t: u8) -> Vec<u8> {
        let body = [
            0x00, 0x0f, // CLUT id 0, version 0
            0x00, 0x5f, 0x10, 0x80, 0x80, 0xff, // entry 0, 4-bit, full range, transparent
            0x01, 0x5f, 0x51, 0xf0, 0x5a, t, // entry 1, 4-bit, full range, red
        ];
        seg(SEG_CLUT_DEFINITION, page_id, &body)
    }

    /// A big-endian bit writer, the inverse of [`Bits`].
    #[derive(Default)]
    struct BitW {
        out: Vec<u8>,
        acc: u32,
        nbits: u32,
    }

    impl BitW {
        fn put(&mut self, v: u32, n: u32) {
            for i in (0..n).rev() {
                self.acc = (self.acc << 1) | ((v >> i) & 1);
                self.nbits += 1;
                if self.nbits == 8 {
                    self.out.push(self.acc as u8);
                    self.acc = 0;
                    self.nbits = 0;
                }
            }
        }
        fn finish(mut self) -> Vec<u8> {
            if self.nbits > 0 {
                self.out.push((self.acc << (8 - self.nbits)) as u8);
            }
            self.out
        }
    }

    /// An object whose every coded line is `w` pixels of code 1 (`w` in
    /// 25..=280, the run the 4-bit escape codes), `h` lines tall.
    fn object_segment(page_id: u16, w: usize, h: usize) -> Vec<u8> {
        // a 4-bit string: escape, switch_1 = 1, switch_2 = 1, switch_3 = 3,
        // run_length_25-280, then the pixel code, then the end-of-string escape.
        let mut bw = BitW::default();
        bw.put(0, 4);
        bw.put(1, 1);
        bw.put(1, 1);
        bw.put(3, 2);
        bw.put((w - 25) as u32, 8);
        bw.put(1, 4);
        bw.put(0, 4);
        bw.put(0, 1);
        bw.put(0, 3);
        let line = bw.finish();

        let field = |rows: usize| {
            let mut out = Vec::new();
            for r in 0..rows {
                out.push(PIX_4BIT);
                out.extend_from_slice(&line);
                if r + 1 < rows {
                    out.push(END_OF_OBJECT_LINE);
                }
            }
            out
        };
        let top = field(h.div_ceil(2));
        let bottom = field(h / 2);
        let mut body = Vec::from([0x00u8, 0x00, 0x00]); // object id 0, pixel coding
        body.extend_from_slice(&(top.len() as u16).to_be_bytes());
        body.extend_from_slice(&(bottom.len() as u16).to_be_bytes());
        body.extend_from_slice(&top);
        body.extend_from_slice(&bottom);
        seg(SEG_OBJECT_DATA, page_id, &body)
    }

    /// A whole display set placing a `w` x `h` red block at (`x`, `y`).
    fn display_set(page_id: u16, x: u16, y: u16, w: usize, h: usize) -> Vec<u8> {
        display_set_t(page_id, x, y, w, h, 0x00)
    }

    /// As [`display_set`], with the block's CLUT entry at transparency `t`.
    fn display_set_t(page_id: u16, x: u16, y: u16, w: usize, h: usize, t: u8) -> Vec<u8> {
        let mut out = Vec::from([DATA_IDENTIFIER, 0x00]);
        out.extend_from_slice(&seg(
            SEG_DISPLAY_DEFINITION,
            page_id,
            &[0x00, 0x02, 0xcf, 0x02, 0x3f],
        ));
        out.extend_from_slice(&page_segment(page_id, &[(0, x, y)]));
        out.extend_from_slice(&clut_segment_t(page_id, t));
        out.extend_from_slice(&region_segment(page_id, w as u16, h as u16));
        out.extend_from_slice(&object_segment(page_id, w, h));
        out.extend_from_slice(&seg(0x80, page_id, &[]));
        out.push(0xff);
        out
    }

    fn pixel(page: &Page, x: u32, y: u32) -> [u8; 4] {
        let at = ((y * page.width + x) * 4) as usize;
        page.canvas[at..at + 4].try_into().unwrap()
    }

    #[test]
    fn decodes_a_display_set_to_its_region_rectangle() {
        let mut dec = DvbSubDecoder::new();
        let page = dec.feed(&display_set(1, 100, 40, 200, 60)).unwrap();
        assert_eq!((page.width, page.height), (720, 576));
        assert_eq!(page.timeout_s, 30);
        assert!(page.visible);
        assert_eq!(dec.page_id(), Some(1));
        // the CLUT's entry 1 (Y 0x51, Cr 0xf0, Cb 0x5a) is BT.601 red
        assert_eq!(pixel(&page, 100, 40), [254, 0, 0, 255]);
        assert_eq!(pixel(&page, 299, 99), [254, 0, 0, 255]);
        assert_eq!(pixel(&page, 99, 40), [0, 0, 0, 0]);
        assert_eq!(pixel(&page, 300, 99), [0, 0, 0, 0]);
        assert_eq!(pixel(&page, 100, 100), [0, 0, 0, 0]);
    }

    #[test]
    fn a_translucent_cue_keeps_its_colour_against_the_cleared_canvas() {
        let mut dec = DvbSubDecoder::new();
        // transparency 0x80, so the entry is a half transparent red, not a dark one
        let page = dec.feed(&display_set_t(1, 100, 40, 200, 60, 0x80)).unwrap();
        let px = pixel(&page, 100, 40);
        assert_eq!(px[3], 127, "alpha carries the transparency");
        assert_eq!(
            [px[0], px[1], px[2]],
            [254, 0, 0],
            "the colour is the opaque cue's, not premultiplied down by its alpha"
        );
    }

    #[test]
    fn a_page_composition_with_no_region_clears_the_display() {
        let mut dec = DvbSubDecoder::new();
        dec.feed(&display_set(1, 10, 10, 40, 4)).unwrap();
        let mut clear = Vec::from([DATA_IDENTIFIER, 0x00]);
        clear.extend_from_slice(&page_segment(1, &[]));
        let page = dec.feed(&clear).unwrap();
        assert!(!page.visible);
        assert!(page.canvas.iter().all(|&b| b == 0));
    }

    #[test]
    fn a_display_definition_resizes_the_canvas() {
        let mut dec = DvbSubDecoder::new();
        assert_eq!(dec.display_size(), (720, 576));
        let mut set = Vec::from([DATA_IDENTIFIER, 0x00]);
        // 1920x1080, no display window
        set.extend_from_slice(&seg(
            SEG_DISPLAY_DEFINITION,
            1,
            &[0x00, 0x07, 0x7f, 0x04, 0x37],
        ));
        set.extend_from_slice(&page_segment(1, &[]));
        let page = dec.feed(&set).unwrap();
        assert_eq!((page.width, page.height), (1920, 1080));
        assert_eq!(page.canvas.len(), 1920 * 1080 * 4);
    }

    #[test]
    fn segments_for_another_page_are_ignored() {
        let mut dec = DvbSubDecoder::new();
        dec.select_pages(PageIds {
            composition: 7,
            ancillary: 7,
        });
        assert!(dec.feed(&display_set(1, 10, 10, 40, 4)).is_none());
        let page = dec.feed(&display_set(7, 10, 10, 40, 4)).unwrap();
        assert!(page.visible);
    }

    #[test]
    fn a_bare_segment_stream_decodes_without_the_pes_data_field_header() {
        let with_header = display_set(1, 8, 8, 32, 4);
        let mut dec = DvbSubDecoder::new();
        let bare = dec.feed(&with_header[2..]).unwrap();
        let mut other = DvbSubDecoder::new();
        assert_eq!(other.feed(&with_header).unwrap(), bare);
    }

    #[test]
    fn rejects_a_segment_length_past_the_data_field() {
        let mut set = display_set(1, 10, 10, 40, 4);
        // the display definition segment's length field, blown past the buffer
        set[6..8].copy_from_slice(&0x7fffu16.to_be_bytes());
        assert!(DvbSubDecoder::new().feed(&set).is_none());
    }

    #[test]
    fn rejects_a_bad_sync_byte() {
        let mut set = display_set(1, 10, 10, 40, 4);
        set[2] = 0x0e;
        assert!(DvbSubDecoder::new().feed(&set).is_none());
    }

    #[test]
    fn drops_an_oversized_region_without_allocating_it() {
        let mut set = Vec::from([DATA_IDENTIFIER, 0x00]);
        set.extend_from_slice(&page_segment(1, &[(0, 0, 0)]));
        // 65535x65535 fits the coded fields but not MAX_REGION_PIXELS
        set.extend_from_slice(&region_segment(1, 0xffff, 0xffff));
        let page = DvbSubDecoder::new().feed(&set).unwrap();
        // the page lists the region, but no region was built, so nothing paints
        assert!(page.canvas.iter().all(|&b| b == 0));
    }

    #[test]
    fn drops_an_object_whose_pixel_data_overruns_its_segment() {
        let mut set = Vec::from([DATA_IDENTIFIER, 0x00]);
        set.extend_from_slice(&page_segment(1, &[(0, 0, 0)]));
        set.extend_from_slice(&clut_segment(1));
        set.extend_from_slice(&region_segment(1, 40, 4));
        let mut object = object_segment(1, 40, 4);
        // the top-field length now claims more bytes than the segment holds
        let at = 6 + 3;
        object[at..at + 2].copy_from_slice(&0x7000u16.to_be_bytes());
        set.extend_from_slice(&object);
        let page = DvbSubDecoder::new().feed(&set).unwrap();
        assert!(
            page.canvas.iter().all(|&b| b == 0),
            "no pixels are drawn from a truncated object"
        );
    }

    #[test]
    fn a_region_depth_outside_the_standard_is_dropped() {
        let mut set = Vec::from([DATA_IDENTIFIER, 0x00]);
        set.extend_from_slice(&page_segment(1, &[(0, 0, 0)]));
        let mut region = region_segment(1, 40, 4);
        region[6 + 6] = 0x5c; // region_depth = 7
        set.extend_from_slice(&region);
        let page = DvbSubDecoder::new().feed(&set).unwrap();
        assert!(page.canvas.iter().all(|&b| b == 0));
    }

    #[test]
    fn a_data_field_with_no_page_composition_yields_nothing() {
        let mut set = Vec::from([DATA_IDENTIFIER, 0x00]);
        set.extend_from_slice(&clut_segment(1));
        assert!(DvbSubDecoder::new().feed(&set).is_none());
    }

    #[test]
    fn reads_the_page_id_blob_of_both_lengths() {
        assert_eq!(
            parse_page_ids(&[0x00, 0x01, 0x00, 0x02, 0x10]),
            Some(PageIds {
                composition: 1,
                ancillary: 2
            })
        );
        assert_eq!(
            parse_page_ids(&[0x00, 0x03, 0x00, 0x04]),
            Some(PageIds {
                composition: 3,
                ancillary: 4
            })
        );
        // two substreams, the first one's ids
        assert_eq!(
            parse_page_ids(&[0, 5, 0, 6, 0x10, 0, 7, 0, 8, 0x10]).map(|p| p.composition),
            Some(5)
        );
        assert!(parse_page_ids(&[0x00, 0x01, 0x00]).is_none());
        assert!(parse_page_ids(&[]).is_none());
        assert!(parse_page_ids(&[0; 7]).is_none());
        let ids = PageIds {
            composition: 0x1234,
            ancillary: 0x5678,
        };
        assert_eq!(parse_page_ids(&page_id_blob(ids, 0x10)), Some(ids));
    }

    #[test]
    fn the_2bit_and_8bit_code_strings_round_trip_a_run() {
        // 2-bit, MSB first: escape `00`, switch_1 `1`, run_length_3-10 `101`
        // (so 8), code `10`, then the six-bit end-of-string escape.
        let mut row = [0u8; 16];
        let (written, _) = read_pixel_string(2, &[0b0011_0110, 0b0000_0000], &mut row, None, false);
        assert_eq!(written, 8);
        assert_eq!(&row[..8], &[2u8; 8]);
        // 8-bit: escape, switch_1 = 1, run 6, code 9
        let mut row = [0u8; 16];
        let (written, _) =
            read_pixel_string(8, &[0x00, 0x86, 0x09, 0x00, 0x00], &mut row, None, false);
        assert_eq!(written, 6);
        assert_eq!(&row[..6], &[9u8; 6]);
    }
}
