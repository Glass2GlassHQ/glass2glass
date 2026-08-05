//! Blu-ray Presentation Graphic Stream (PGS / HDMV) decoding: the pure half
//! behind [`PgsDec`](crate::pgsdec::PgsDec).
//!
//! Like a DVB subtitle and unlike a VobSub cue, a PGS subtitle is a *segment
//! stream* with decoder state carried across display sets. A display set is a
//! run of segments ending in an end-of-display-set segment:
//!
//! - presentation composition (0x16): the video geometry, the epoch state, which
//!   palette to read, and where each object is placed. Listing no object is how
//!   a cue ends.
//! - window definition (0x17): the rectangles the objects are drawn inside.
//!   Nothing the composition needs, since every object carries its own position.
//! - palette definition (0x14): CLUT entries as Y / Cr / Cb / alpha, held per
//!   palette id across the epoch so a partial update keeps the untouched entries.
//! - object definition (0x15): the run-length coded bitmap, possibly split over
//!   several fragments when it does not fit one segment.
//!
//! A `.sup` file frames every segment with `PG`, a 90 kHz PTS and a DTS; a
//! Matroska `S_HDMV/PGS` block carries the bare segment headers and puts the PTS
//! on the block. [`PgsDecoder::feed`] takes either.
//!
//! Every length, count, coordinate and dimension here comes off the wire, so
//! each is range-checked before use and a malformed segment is dropped rather
//! than panicking or allocating on a bogus size.

use alloc::vec;
use alloc::vec::Vec;

use crate::paint::{ycbcr_to_rgb, YcbcrMatrix};

/// Largest video edge accepted. The PGS video descriptor is two 16-bit fields,
/// so this bounds a crafted geometry well above the 1920x1080 Blu-ray uses.
pub const MAX_VIDEO_DIM: u32 = 4096;

/// Largest object bitmap accepted, in pixels: enough for a full-screen HD object
/// without letting a crafted 65535x65535 object ask for a 4 GB allocation.
pub const MAX_OBJECT_PIXELS: usize = 2048 * 2048;

/// Video geometry assumed until a presentation composition segment names one
/// (PGS is a Blu-ray format, so HD).
pub const DEFAULT_VIDEO_WIDTH: u32 = 1920;
/// Video height for the same case.
pub const DEFAULT_VIDEO_HEIGHT: u32 = 1080;

/// Blu-ray caps an epoch at 8 palettes and 64 objects, and a display set at 2
/// composition objects (one per window).
const MAX_EPOCH_PALETTES: usize = 8;
const MAX_EPOCH_OBJECTS: usize = 64;
const MAX_OBJECT_REFS: usize = 2;

const SEG_PALETTE: u8 = 0x14;
const SEG_OBJECT: u8 = 0x15;
const SEG_PRESENTATION: u8 = 0x16;
const SEG_WINDOW: u8 = 0x17;
const SEG_END: u8 = 0x80;

/// Leads a `.sup` file's per-segment header.
const SUP_MAGIC: &[u8; 2] = b"PG";

/// `object_cropped_flag` in a composition object's flags byte. The widely-copied
/// Scorpius write-up has this and the forced bit the other way round; a
/// reference decoder reads 0x80 as cropped.
const COMP_CROPPED: u8 = 0x80;
/// `forced_on_flag`, the bit that marks a cue a player shows even with subtitles
/// switched off.
const COMP_FORCED: u8 = 0x40;

/// `sequence_desc` bit marking the first fragment of an object's RLE data. A
/// segment without it appends to the object already in flight.
const SEQ_FIRST: u8 = 0x80;

/// One composed display set: the screen as it should now look.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisplaySet {
    /// Whether anything was painted. `false` is the stream ending the cue (a
    /// presentation composition listing no object), and `canvas` is then fully
    /// transparent.
    pub visible: bool,
    /// Whether any painted object carried the forced flag.
    pub forced: bool,
    /// The 90 kHz PTS of the presentation composition, when the buffer was
    /// `.sup`-framed. `None` for a Matroska block, whose PTS is on the block.
    pub pts_90k: Option<u32>,
    /// The video, `width * height * 4` RGBA bytes, transparent where no object
    /// paints.
    pub canvas: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// A composition object from the presentation segment: an object id, where it is
/// placed, and the sub-rectangle of it to show when the composition crops.
#[derive(Debug, Clone, Copy)]
struct CompObject {
    id: u16,
    x: u32,
    y: u32,
    forced: bool,
    /// `(x, y, w, h)` inside the object bitmap, when `object_cropped_flag` is set.
    crop: Option<(u32, u32, u32, u32)>,
}

/// An epoch palette: 256 RGBA entries, indexed by the codes the RLE carries.
#[derive(Debug, Clone)]
struct Palette {
    id: u8,
    clut: [[u8; 4]; 256],
}

/// An epoch object: its geometry and the RLE bytes gathered so far.
#[derive(Debug, Clone)]
struct Object {
    id: u16,
    width: u32,
    height: u32,
    rle: Vec<u8>,
    /// RLE bytes the first fragment declared and later fragments still owe.
    remaining: usize,
}

/// One segment of a display set.
#[derive(Debug, Clone, Copy)]
struct Segment<'a> {
    kind: u8,
    pts_90k: Option<u32>,
    body: &'a [u8],
}

/// Split a buffer into its segments, accepting both the `.sup` per-segment
/// header (`PG`, PTS, DTS, type, length) and the bare Matroska one (type,
/// length). No segment type is 0x50, so the magic tells the two apart without a
/// mode flag. A header or declared length that does not fit what is left stops
/// the walk, keeping the display sets that did parse, the way a reference
/// decoder does.
fn segments(data: &[u8]) -> Vec<Segment<'_>> {
    let mut rest = data;
    let mut out = Vec::new();
    while !rest.is_empty() {
        let (pts_90k, head_len) = if rest.starts_with(SUP_MAGIC) {
            match rest.get(..13) {
                Some(h) => (Some(u32::from_be_bytes([h[2], h[3], h[4], h[5]])), 13usize),
                None => return out,
            }
        } else {
            (None, 3usize)
        };
        let Some(head) = rest.get(..head_len) else {
            return out;
        };
        let kind = head[head_len - 3];
        let length = u16::from_be_bytes([head[head_len - 2], head[head_len - 1]]) as usize;
        let Some(body) = rest.get(head_len..head_len + length) else {
            return out;
        };
        out.push(Segment {
            kind,
            pts_90k,
            body,
        });
        rest = &rest[head_len + length..];
    }
    out
}

/// The PGS decoder: epoch state (palettes and objects) plus the composition the
/// next end-of-display-set segment will paint.
#[derive(Debug)]
pub struct PgsDecoder {
    width: u32,
    height: u32,
    palettes: Vec<Palette>,
    objects: Vec<Object>,
    comps: Vec<CompObject>,
    palette_id: u8,
    pts_90k: Option<u32>,
    forced_only: bool,
}

impl Default for PgsDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl PgsDecoder {
    pub fn new() -> Self {
        Self {
            width: DEFAULT_VIDEO_WIDTH,
            height: DEFAULT_VIDEO_HEIGHT,
            palettes: Vec::new(),
            objects: Vec::new(),
            comps: Vec::new(),
            palette_id: 0,
            pts_90k: None,
            forced_only: false,
        }
    }

    /// Paint only the objects marked `forced_on_flag`, dropping the ordinary
    /// subtitle track (ffmpeg's `forced_subs_only`).
    pub fn set_forced_only(&mut self, forced_only: bool) {
        self.forced_only = forced_only;
    }

    /// Set the video geometry to compose at until a presentation composition
    /// segment names another.
    pub fn set_video_size(&mut self, width: u32, height: u32) {
        self.width = width.clamp(1, MAX_VIDEO_DIM);
        self.height = height.clamp(1, MAX_VIDEO_DIM);
    }

    /// The video geometry the next [`DisplaySet`] will be composed at.
    pub fn video_size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Feed one buffer, returning the display sets it completed in order. A
    /// Matroska block holds exactly one; a whole `.sup` file holds all of them.
    pub fn feed(&mut self, data: &[u8]) -> Vec<DisplaySet> {
        let mut out = Vec::new();
        for seg in segments(data) {
            match seg.kind {
                SEG_PRESENTATION => self.apply_presentation(seg.body, seg.pts_90k),
                SEG_PALETTE => self.apply_palette(seg.body),
                SEG_OBJECT => self.apply_object(seg.body),
                // A window only names the rectangle an object is drawn inside;
                // the object carries its own position, so nothing to apply.
                SEG_WINDOW => {}
                SEG_END => out.push(self.compose()),
                // Reserved / unknown type: skipped, its length already walked.
                _ => {}
            }
        }
        out
    }

    /// Presentation composition segment: the video descriptor, the epoch state,
    /// the palette to read, and the composition objects.
    fn apply_presentation(&mut self, body: &[u8], pts_90k: Option<u32>) {
        let Some(head) = body.get(..11) else {
            return;
        };
        let width = u16::from_be_bytes([head[0], head[1]]) as u32;
        let height = u16::from_be_bytes([head[2], head[3]]) as u32;
        if width == 0 || height == 0 || width > MAX_VIDEO_DIM || height > MAX_VIDEO_DIM {
            return;
        }
        self.width = width;
        self.height = height;
        // head[4] frame rate and head[5..7] composition number carry nothing the
        // composition needs.
        // Any composition state but "normal" opens a new epoch, so the palettes
        // and objects carried over from the previous one are released.
        if head[7] >> 6 != 0 {
            self.palettes.clear();
            self.objects.clear();
        }
        // head[8] palette_update_flag: the CLUT is persistent per palette id
        // either way, so a partial update lands the same with or without it.
        self.palette_id = head[9];
        self.pts_90k = pts_90k;

        self.comps.clear();
        let mut rest = &body[11..];
        for _ in 0..(head[10] as usize).min(MAX_OBJECT_REFS) {
            let Some(entry) = rest.get(..8) else {
                // Truncated object list: keep the objects that did parse.
                break;
            };
            let id = u16::from_be_bytes([entry[0], entry[1]]);
            let flags = entry[3];
            let x = u16::from_be_bytes([entry[4], entry[5]]) as u32;
            let y = u16::from_be_bytes([entry[6], entry[7]]) as u32;
            let mut used = 8;
            let mut crop = None;
            if flags & COMP_CROPPED != 0 {
                let Some(c) = rest.get(8..16) else {
                    break;
                };
                crop = Some((
                    u16::from_be_bytes([c[0], c[1]]) as u32,
                    u16::from_be_bytes([c[2], c[3]]) as u32,
                    u16::from_be_bytes([c[4], c[5]]) as u32,
                    u16::from_be_bytes([c[6], c[7]]) as u32,
                ));
                used = 16;
            }
            rest = &rest[used..];
            // A placement outside the video is a malformed composition; the
            // object is dropped rather than slid to the origin.
            if x >= self.width || y >= self.height {
                continue;
            }
            self.comps.push(CompObject {
                id,
                x,
                y,
                forced: flags & COMP_FORCED != 0,
                crop,
            });
        }
    }

    /// Palette definition segment: an id, a version, then 5-byte
    /// `entry / Y / Cr / Cb / alpha` entries until the segment runs out. Entries
    /// the segment does not name keep their previous value.
    fn apply_palette(&mut self, body: &[u8]) {
        let Some(head) = body.get(..2) else {
            return;
        };
        let id = head[0];
        let idx = match self.palettes.iter().position(|p| p.id == id) {
            Some(i) => i,
            None => {
                if self.palettes.len() >= MAX_EPOCH_PALETTES {
                    return;
                }
                self.palettes.push(Palette {
                    id,
                    clut: [[0; 4]; 256],
                });
                self.palettes.len() - 1
            }
        };
        // A reference decoder picks the matrix off the video height rather than
        // anything in the stream: BT.709 for HD, BT.601 at SD and below.
        let matrix = if self.height > 576 {
            YcbcrMatrix::Bt709
        } else {
            YcbcrMatrix::Bt601
        };
        for e in body[2..].chunks_exact(5) {
            let (r, g, b) = ycbcr_to_rgb(e[1], e[2], e[3], matrix);
            self.palettes[idx].clut[e[0] as usize] = [r, g, b, e[4]];
        }
    }

    /// Object definition segment: an id, a version, a sequence flag, and either
    /// the first fragment (declared length, geometry, then RLE) or a
    /// continuation carrying nothing but more RLE.
    fn apply_object(&mut self, body: &[u8]) {
        let Some(head) = body.get(..4) else {
            return;
        };
        let id = u16::from_be_bytes([head[0], head[1]]);
        let first = head[3] & SEQ_FIRST != 0;
        let data = &body[4..];
        if data.is_empty() {
            return;
        }
        if !first {
            let Some(obj) = self.objects.iter_mut().find(|o| o.id == id) else {
                return;
            };
            // More RLE than the first fragment declared: the object is corrupt,
            // and appending would decode past its bitmap.
            if data.len() > obj.remaining {
                return;
            }
            obj.remaining -= data.len();
            obj.rle.extend_from_slice(data);
            return;
        }
        let Some(head) = data.get(..7) else {
            return;
        };
        // The declared length counts the two dimension fields as well as the RLE.
        let declared = u32::from_be_bytes([0, head[0], head[1], head[2]]) as usize;
        let Some(rle_len) = declared.checked_sub(4) else {
            return;
        };
        let width = u16::from_be_bytes([head[3], head[4]]) as u32;
        let height = u16::from_be_bytes([head[5], head[6]]) as u32;
        let payload = &data[7..];
        if payload.len() > rle_len {
            return;
        }
        if width == 0 || height == 0 || width > self.width || height > self.height {
            return;
        }
        if (width as usize).saturating_mul(height as usize) > MAX_OBJECT_PIXELS {
            return;
        }
        // Only the bytes that actually arrived are held, so a bogus declared
        // length buys no allocation: the continuations have to carry the rest.
        let obj = Object {
            id,
            width,
            height,
            rle: payload.to_vec(),
            remaining: rle_len - payload.len(),
        };
        match self.objects.iter().position(|o| o.id == id) {
            Some(i) => self.objects[i] = obj,
            None if self.objects.len() < MAX_EPOCH_OBJECTS => self.objects.push(obj),
            None => {}
        }
    }

    /// Paint the composition onto a transparent video-sized canvas.
    fn compose(&self) -> DisplaySet {
        let stride = (self.width as usize) * 4;
        let mut canvas = vec![0u8; stride * self.height as usize];
        let mut painted = false;
        let mut forced = false;
        let palette = self.palettes.iter().find(|p| p.id == self.palette_id);
        for comp in &self.comps {
            if self.forced_only && !comp.forced {
                continue;
            }
            // A composition naming a palette or an object no segment defined is
            // a damaged stream; that object simply does not paint.
            let Some(palette) = palette else {
                continue;
            };
            let Some(obj) = self.objects.iter().find(|o| o.id == comp.id) else {
                continue;
            };
            let Some(pixels) = decode_rle(&obj.rle, obj.width, obj.height) else {
                continue;
            };
            let (sx, sy, cw, ch) = comp.crop.unwrap_or((0, 0, obj.width, obj.height));
            let cw = cw.min(obj.width.saturating_sub(sx));
            let ch = ch.min(obj.height.saturating_sub(sy));
            for row in 0..ch {
                let cy = comp.y + row;
                if cy >= self.height {
                    break;
                }
                for col in 0..cw {
                    let cx = comp.x + col;
                    if cx >= self.width {
                        break;
                    }
                    let code = pixels[((sy + row) * obj.width + sx + col) as usize];
                    let px = palette.clut[code as usize];
                    if px[3] == 0 {
                        continue;
                    }
                    // Composition objects sit in disjoint windows, so each canvas
                    // pixel is written once: a straight copy keeps the palette's
                    // alpha as-is instead of premultiplying it against the
                    // transparent canvas the way a source-over blend would.
                    let at = (cy as usize * self.width as usize + cx as usize) * 4;
                    canvas[at..at + 4].copy_from_slice(&px);
                    painted = true;
                    forced |= comp.forced;
                }
            }
        }
        DisplaySet {
            visible: painted,
            forced,
            pts_90k: self.pts_90k,
            canvas,
            width: self.width,
            height: self.height,
        }
    }
}

/// Decode an object's RLE into `width * height` palette codes, row-major.
///
/// A colour byte stands for one pixel of that colour; a zero byte escapes to a
/// flags byte holding a 6-bit run length, extended to 14 bits by a following
/// byte when 0x40 is set, of colour 0 unless 0x80 names one. A zero run is the
/// end of the line.
///
/// `None` when the data runs out mid-code, a run would overflow the bitmap, or
/// the codes do not cover it: none of those can be rendered, and guessing would
/// shift every following row.
fn decode_rle(rle: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
    let (w, h) = (width as usize, height as usize);
    let total = w.checked_mul(h)?;
    let mut pixels = vec![0u8; total];
    let mut i = 0;
    let mut at = 0;
    let mut line = 0;
    while i < rle.len() && line < h {
        let mut color = rle[i];
        i += 1;
        let mut run = 1usize;
        if color == 0 {
            let flags = *rle.get(i)?;
            i += 1;
            run = (flags & 0x3f) as usize;
            if flags & 0x40 != 0 {
                run = (run << 8) | *rle.get(i)? as usize;
                i += 1;
            }
            color = if flags & 0x80 != 0 {
                let c = *rle.get(i)?;
                i += 1;
                c
            } else {
                0
            };
        }
        if run == 0 {
            // End of line. A line short of `w` codes is padded to the row edge,
            // so a following row still starts where it should.
            line += 1;
            at = line.checked_mul(w)?;
            continue;
        }
        let end = at.checked_add(run)?;
        if end > total {
            return None;
        }
        pixels[at..end].fill(color);
        at = end;
    }
    if at < total {
        return None;
    }
    Some(pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `.sup`-framed segment.
    fn sup(kind: u8, pts_90k: u32, body: &[u8]) -> Vec<u8> {
        let mut out = Vec::from(*SUP_MAGIC);
        out.extend_from_slice(&pts_90k.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        out.push(kind);
        out.extend_from_slice(&(body.len() as u16).to_be_bytes());
        out.extend_from_slice(body);
        out
    }

    /// Build a bare (Matroska) segment.
    fn bare(kind: u8, body: &[u8]) -> Vec<u8> {
        let mut out = Vec::from([kind]);
        out.extend_from_slice(&(body.len() as u16).to_be_bytes());
        out.extend_from_slice(body);
        out
    }

    fn pcs(w: u16, h: u16, palette_id: u8, objects: &[(u16, u8, u16, u16)]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&w.to_be_bytes());
        b.extend_from_slice(&h.to_be_bytes());
        b.push(0x10); // frame rate
        b.extend_from_slice(&0u16.to_be_bytes()); // composition number
        b.push(0x80); // epoch start
        b.push(0x00); // palette update flag
        b.push(palette_id);
        b.push(objects.len() as u8);
        for (id, flags, x, y) in objects {
            b.extend_from_slice(&id.to_be_bytes());
            b.push(0); // window id
            b.push(*flags);
            b.extend_from_slice(&x.to_be_bytes());
            b.extend_from_slice(&y.to_be_bytes());
        }
        b
    }

    fn pds(id: u8, entries: &[(u8, u8, u8, u8, u8)]) -> Vec<u8> {
        let mut b = Vec::from([id, 0]);
        for (i, y, cr, cb, a) in entries {
            b.extend_from_slice(&[*i, *y, *cr, *cb, *a]);
        }
        b
    }

    fn ods(id: u16, w: u16, h: u16, rle: &[u8]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&id.to_be_bytes());
        b.push(0); // version
        b.push(0xC0); // first and last fragment
        let declared = (rle.len() + 4) as u32;
        b.extend_from_slice(&declared.to_be_bytes()[1..]);
        b.extend_from_slice(&w.to_be_bytes());
        b.extend_from_slice(&h.to_be_bytes());
        b.extend_from_slice(rle);
        b
    }

    /// A `w` x `h` solid block of colour 1, one line at a time.
    fn solid_rle(w: usize, h: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for _ in 0..h {
            out.extend_from_slice(&[0x00, 0x80 | w as u8, 0x01]);
            out.extend_from_slice(&[0x00, 0x00]);
        }
        out
    }

    #[test]
    fn a_display_set_paints_its_object_at_the_composition_position() {
        let mut dec = PgsDecoder::new();
        let mut buf = Vec::new();
        buf.extend_from_slice(&bare(SEG_PRESENTATION, &pcs(64, 32, 0, &[(1, 0, 8, 4)])));
        buf.extend_from_slice(&bare(SEG_PALETTE, &pds(0, &[(1, 235, 128, 128, 255)])));
        buf.extend_from_slice(&bare(SEG_OBJECT, &ods(1, 6, 3, &solid_rle(6, 3))));
        buf.extend_from_slice(&bare(SEG_END, &[]));

        let sets = dec.feed(&buf);
        assert_eq!(sets.len(), 1);
        let set = &sets[0];
        assert!(set.visible);
        assert_eq!((set.width, set.height), (64, 32));
        let px = |x: usize, y: usize| &set.canvas[(y * 64 + x) * 4..(y * 64 + x) * 4 + 4];
        // Y=235 Cr=Cb=128 is white at limited range.
        assert_eq!(px(8, 4), [255, 255, 255, 255]);
        assert_eq!(px(13, 6), [255, 255, 255, 255]);
        assert_eq!(px(7, 4), [0, 0, 0, 0]);
        assert_eq!(px(14, 6), [0, 0, 0, 0]);
    }

    #[test]
    fn an_empty_composition_ends_the_cue() {
        let mut dec = PgsDecoder::new();
        let mut buf = Vec::new();
        buf.extend_from_slice(&bare(SEG_PRESENTATION, &pcs(64, 32, 0, &[])));
        buf.extend_from_slice(&bare(SEG_END, &[]));
        let sets = dec.feed(&buf);
        assert_eq!(sets.len(), 1);
        assert!(!sets[0].visible);
        assert!(sets[0].canvas.iter().all(|&b| b == 0));
    }

    #[test]
    fn sup_framing_carries_the_presentation_pts() {
        let mut dec = PgsDecoder::new();
        let mut buf = Vec::new();
        buf.extend_from_slice(&sup(SEG_PRESENTATION, 90_000, &pcs(64, 32, 0, &[])));
        buf.extend_from_slice(&sup(SEG_END, 90_000, &[]));
        buf.extend_from_slice(&sup(SEG_PRESENTATION, 180_000, &pcs(64, 32, 0, &[])));
        buf.extend_from_slice(&sup(SEG_END, 180_000, &[]));
        let sets = dec.feed(&buf);
        assert_eq!(sets.len(), 2);
        assert_eq!(sets[0].pts_90k, Some(90_000));
        assert_eq!(sets[1].pts_90k, Some(180_000));
    }

    #[test]
    fn a_cropped_object_paints_only_its_sub_rectangle() {
        let mut dec = PgsDecoder::new();
        let mut buf = Vec::new();
        let mut comp = pcs(64, 32, 0, &[(1, COMP_CROPPED, 0, 0)]);
        // crop x=2, y=1, w=2, h=1 inside the 6x3 object
        comp.extend_from_slice(&[0, 2, 0, 1, 0, 2, 0, 1]);
        buf.extend_from_slice(&bare(SEG_PRESENTATION, &comp));
        buf.extend_from_slice(&bare(SEG_PALETTE, &pds(0, &[(1, 235, 128, 128, 255)])));
        buf.extend_from_slice(&bare(SEG_OBJECT, &ods(1, 6, 3, &solid_rle(6, 3))));
        buf.extend_from_slice(&bare(SEG_END, &[]));
        let set = dec.feed(&buf).remove(0);
        let opaque = set.canvas.chunks_exact(4).filter(|p| p[3] != 0).count();
        assert_eq!(opaque, 2, "only the 2x1 crop rectangle paints");
    }

    #[test]
    fn forced_only_drops_the_ordinary_track() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&bare(SEG_PRESENTATION, &pcs(64, 32, 0, &[(1, 0, 0, 0)])));
        buf.extend_from_slice(&bare(SEG_PALETTE, &pds(0, &[(1, 235, 128, 128, 255)])));
        buf.extend_from_slice(&bare(SEG_OBJECT, &ods(1, 6, 3, &solid_rle(6, 3))));
        buf.extend_from_slice(&bare(SEG_END, &[]));

        let mut dec = PgsDecoder::new();
        dec.set_forced_only(true);
        assert!(!dec.feed(&buf)[0].visible);

        let mut buf_forced = Vec::new();
        buf_forced.extend_from_slice(&bare(
            SEG_PRESENTATION,
            &pcs(64, 32, 0, &[(1, COMP_FORCED, 0, 0)]),
        ));
        buf_forced.extend_from_slice(&bare(SEG_PALETTE, &pds(0, &[(1, 235, 128, 128, 255)])));
        buf_forced.extend_from_slice(&bare(SEG_OBJECT, &ods(1, 6, 3, &solid_rle(6, 3))));
        buf_forced.extend_from_slice(&bare(SEG_END, &[]));
        let mut dec = PgsDecoder::new();
        dec.set_forced_only(true);
        let set = dec.feed(&buf_forced).remove(0);
        assert!(set.visible && set.forced);
    }

    #[test]
    fn a_fragmented_object_reassembles() {
        let rle = solid_rle(6, 3);
        let (a, b) = rle.split_at(5);
        let mut first = Vec::new();
        first.extend_from_slice(&1u16.to_be_bytes());
        first.push(0);
        first.push(SEQ_FIRST);
        first.extend_from_slice(&((rle.len() + 4) as u32).to_be_bytes()[1..]);
        first.extend_from_slice(&6u16.to_be_bytes());
        first.extend_from_slice(&3u16.to_be_bytes());
        first.extend_from_slice(a);
        let mut cont = Vec::new();
        cont.extend_from_slice(&1u16.to_be_bytes());
        cont.push(0);
        cont.push(0x40); // last fragment only
        cont.extend_from_slice(b);

        let mut buf = Vec::new();
        buf.extend_from_slice(&bare(SEG_PRESENTATION, &pcs(64, 32, 0, &[(1, 0, 0, 0)])));
        buf.extend_from_slice(&bare(SEG_PALETTE, &pds(0, &[(1, 235, 128, 128, 255)])));
        buf.extend_from_slice(&bare(SEG_OBJECT, &first));
        buf.extend_from_slice(&bare(SEG_OBJECT, &cont));
        buf.extend_from_slice(&bare(SEG_END, &[]));
        let set = PgsDecoder::new().feed(&buf).remove(0);
        assert_eq!(
            set.canvas.chunks_exact(4).filter(|p| p[3] != 0).count(),
            18,
            "the 6x3 object paints once both fragments are in"
        );
    }

    #[test]
    fn a_run_past_the_bitmap_is_rejected() {
        // One 40-pixel run into a 6x3 bitmap.
        assert!(decode_rle(&[0x00, 0x80 | 40, 0x01], 6, 3).is_none());
    }

    #[test]
    fn a_truncated_run_is_rejected() {
        // 0x40 promises a second length byte the data does not have.
        assert!(decode_rle(&[0x00, 0x40], 6, 3).is_none());
    }

    #[test]
    fn a_short_object_is_rejected() {
        assert!(decode_rle(&[0x01, 0x00, 0x00], 6, 3).is_none());
    }

    #[test]
    fn a_long_run_spans_fourteen_bits() {
        // 0x40 | 0x80 -> a 14-bit length and an explicit colour.
        let px = decode_rle(&[0x00, 0xC0 | 0x01, 0x2C, 0x07], 300, 1).unwrap();
        assert_eq!(px.len(), 300);
        assert!(px.iter().all(|&c| c == 7));
    }

    #[test]
    fn an_object_larger_than_the_video_is_dropped() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&bare(SEG_PRESENTATION, &pcs(16, 8, 0, &[(1, 0, 0, 0)])));
        buf.extend_from_slice(&bare(SEG_PALETTE, &pds(0, &[(1, 235, 128, 128, 255)])));
        buf.extend_from_slice(&bare(SEG_OBJECT, &ods(1, 32, 32, &solid_rle(32, 32))));
        buf.extend_from_slice(&bare(SEG_END, &[]));
        let set = PgsDecoder::new().feed(&buf).remove(0);
        assert!(!set.visible);
        assert_eq!(set.canvas.len(), 16 * 8 * 4);
    }

    #[test]
    fn a_truncated_segment_stops_the_walk_without_losing_earlier_sets() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&bare(SEG_PRESENTATION, &pcs(64, 32, 0, &[])));
        buf.extend_from_slice(&bare(SEG_END, &[]));
        // A segment header claiming 500 bytes of body with none behind it.
        buf.extend_from_slice(&[SEG_PALETTE, 0x01, 0xF4]);
        assert_eq!(PgsDecoder::new().feed(&buf).len(), 1);
    }
}
