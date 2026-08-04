//! VobSub (DVD subpicture) bitstream parsing: the pure half behind
//! [`VobSubDec`](crate::vobsubdec::VobSubDec).
//!
//! A VobSub cue is one subpicture unit (SPU): a 2-byte packet size, a 2-byte
//! offset to the control sequence, then run-length pixel data and one or more
//! control sequences. The pixels are 2 bits per sample in two interlaced fields
//! (even rows, then odd rows), each field byte-aligned per row; the control
//! sequence carries the display rectangle, the four palette indices and alpha
//! nibbles the 2-bit samples select, the two field data offsets, and the show /
//! hide times relative to the packet's PTS.
//!
//! The 16-entry RGB palette and the display geometry are *not* in the
//! bitstream: they ride out of band in the `.idx` text, which a Matroska
//! `S_VOBSUB` track carries as its `CodecPrivate` ([`parse_idx`]).
//!
//! Every length, offset and coordinate here comes off the wire, so each is
//! range-checked before use and the parse returns `None` rather than panicking
//! or allocating on a bogus size.

use alloc::vec;
use alloc::vec::Vec;

/// Largest display-rectangle edge accepted. The coordinates are 12-bit fields so
/// they cannot exceed this anyway; the constant makes the bound explicit.
pub const MAX_CUE_DIM: u32 = 4096;

/// Largest cue bitmap accepted, in pixels. A DVD subpicture is at most 720x576;
/// this leaves headroom without letting a crafted rectangle ask for a 16 MB
/// allocation.
pub const MAX_CUE_PIXELS: usize = 2048 * 2048;

/// The out-of-band configuration a `.idx` text carries: the subpicture display
/// geometry and the 16-entry palette its cues index into. Either half may be
/// absent from the text, so both are optional.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VobSubConfig {
    /// `size: WxH`, the canvas the display rectangles are placed on.
    pub size: Option<(u32, u32)>,
    /// `palette: rrggbb, ...`, 16 entries as `0x00RRGGBB`.
    pub palette: Option<[u32; 16]>,
}

/// Parse a `.idx` text (a Matroska `S_VOBSUB` track's `CodecPrivate`) into the
/// display geometry and palette it declares. Returns `None` when the bytes are
/// not `.idx` text at all, which is how the decoder tells a config blob from an
/// SPU packet on the same pad.
pub fn parse_idx(bytes: &[u8]) -> Option<VobSubConfig> {
    let text = core::str::from_utf8(bytes).ok()?;
    let mut cfg = VobSubConfig::default();
    // Recognising a key is what makes these bytes `.idx` text; a key whose value
    // does not parse still identifies the blob, it just contributes nothing.
    let mut keyed = false;
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("size:") {
            keyed = true;
            if let Some((w, h)) = rest.trim().split_once('x') {
                if let (Ok(w), Ok(h)) = (w.trim().parse(), h.trim().parse()) {
                    cfg.size = Some((w, h));
                }
            }
        } else if let Some(rest) = line.strip_prefix("palette:") {
            keyed = true;
            let mut entries = [0u32; 16];
            let mut n = 0;
            for field in rest.split(',') {
                let Ok(v) = u32::from_str_radix(field.trim(), 16) else {
                    break;
                };
                let Some(slot) = entries.get_mut(n) else {
                    break;
                };
                *slot = v & 0x00ff_ffff;
                n += 1;
            }
            if n == 16 {
                cfg.palette = Some(entries);
            }
        }
    }
    keyed.then_some(cfg)
}

/// One decoded subpicture cue: when to show it, where, and its palette-indexed
/// bitmap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpuCue {
    /// Show time, nanoseconds after the packet's PTS.
    pub start_ns: u64,
    /// Hide time, nanoseconds after the packet's PTS. `None` when the control
    /// sequence carries no stop-display command, i.e. the cue stands until the
    /// next one replaces it.
    pub stop_ns: Option<u64>,
    /// Left edge of the display rectangle on the subpicture canvas.
    pub x: u32,
    /// Top edge of the display rectangle on the subpicture canvas.
    pub y: u32,
    pub width: u32,
    pub height: u32,
    /// The four palette indices the 2-bit samples select, sample value 0 first.
    pub colormap: [u8; 4],
    /// The four alpha nibbles (0 transparent, 15 opaque), sample value 0 first.
    pub alpha: [u8; 4],
    /// `width * height` sample values (0..=3), row-major.
    pub pixels: Vec<u8>,
}

impl SpuCue {
    /// Paint this cue onto an RGBA8 canvas of `canvas_w` x `canvas_h`, source-over
    /// with the cue's own alpha. Pixels outside the canvas are clipped, so a cue
    /// whose rectangle overhangs the declared display size still renders its
    /// visible part. `palette` is the 16-entry `0x00RRGGBB` table from the `.idx`.
    pub fn paint(&self, palette: &[u32; 16], canvas: &mut [u8], canvas_w: u32, canvas_h: u32) {
        let mut rgba = [[0u8; 4]; 4];
        for (i, px) in rgba.iter_mut().enumerate() {
            let rgb = palette[(self.colormap[i] & 0x0f) as usize];
            // A 4-bit alpha scales to 8 bits by *17 (0 -> 0, 15 -> 255).
            *px = [
                (rgb >> 16) as u8,
                (rgb >> 8) as u8,
                rgb as u8,
                self.alpha[i].min(15) * 17,
            ];
        }
        for row in 0..self.height {
            let cy = self.y + row;
            if cy >= canvas_h {
                break;
            }
            for col in 0..self.width {
                let cx = self.x + col;
                if cx >= canvas_w {
                    break;
                }
                let sample = self.pixels[(row * self.width + col) as usize] & 3;
                let px = rgba[sample as usize];
                if px[3] == 0 {
                    continue;
                }
                let dst = ((cy * canvas_w + cx) * 4) as usize;
                crate::paint::blend_px(canvas, dst, px, 255);
            }
        }
    }
}

/// A control-sequence date is in units of 1024/90000 s (ffmpeg spells the same
/// conversion `(date << 10) / 90` milliseconds).
fn date_to_ns(date: u16) -> u64 {
    date as u64 * 1024 * 1_000_000 / 90
}

fn be16(buf: &[u8], at: usize) -> Option<usize> {
    let hi = *buf.get(at)?;
    let lo = *buf.get(at.checked_add(1)?)?;
    Some(u16::from_be_bytes([hi, lo]) as usize)
}

/// Parse one subpicture unit. Returns `None` for any packet whose sizes, offsets
/// or coordinates do not hold together, so a malformed cue is dropped rather
/// than mis-rendered.
///
/// A cue that omits the colour / contrast commands keeps the identity colormap
/// and an opaque-except-sample-0 alpha, which is what a real stream would set;
/// it must still carry the display area and the two field offsets, without which
/// there is no bitmap to decode.
pub fn parse_spu(data: &[u8]) -> Option<SpuCue> {
    let packet_size = be16(data, 0)?;
    let ctrl_start = be16(data, 2)?;
    // The packet's own size bounds the parse: a claim longer than the buffer is
    // a truncated (or lying) packet, a shorter one just trims trailing padding.
    if packet_size < 4 || packet_size > data.len() {
        return None;
    }
    let packet = &data[..packet_size];
    if ctrl_start < 4 || ctrl_start >= packet_size {
        return None;
    }

    let mut start_date: Option<u16> = None;
    let mut stop_date: Option<u16> = None;
    let mut colormap = [0u8, 1, 2, 3];
    let mut alpha = [0u8, 0x0f, 0x0f, 0x0f];
    let mut area: Option<(u32, u32, u32, u32)> = None;
    let mut field_offsets: Option<(usize, usize)> = None;

    let mut seq = ctrl_start;
    loop {
        let date = be16(packet, seq)? as u16;
        let next = be16(packet, seq + 2)?;
        let mut at = seq + 4;
        loop {
            let cmd = *packet.get(at)?;
            at += 1;
            match cmd {
                // forced start / start display: both show the cue at `date`.
                0x00 | 0x01 => start_date = Some(date),
                0x02 => stop_date = Some(date),
                // SET_COLOR: four palette indices, highest sample value first.
                0x03 => {
                    let a = *packet.get(at)?;
                    let b = *packet.get(at + 1)?;
                    colormap = [b & 0x0f, b >> 4, a & 0x0f, a >> 4];
                    at += 2;
                }
                // SET_CONTR: four alpha nibbles, same order.
                0x04 => {
                    let a = *packet.get(at)?;
                    let b = *packet.get(at + 1)?;
                    alpha = [b & 0x0f, b >> 4, a & 0x0f, a >> 4];
                    at += 2;
                }
                // SET_DAREA: three 12-bit coordinate pairs.
                0x05 => {
                    let c = packet.get(at..at + 6)?;
                    let x1 = ((c[0] as u32) << 4) | (c[1] >> 4) as u32;
                    let x2 = (((c[1] & 0x0f) as u32) << 8) | c[2] as u32;
                    let y1 = ((c[3] as u32) << 4) | (c[4] >> 4) as u32;
                    let y2 = (((c[4] & 0x0f) as u32) << 8) | c[5] as u32;
                    if x2 < x1 || y2 < y1 {
                        return None;
                    }
                    area = Some((x1, y1, x2 - x1 + 1, y2 - y1 + 1));
                    at += 6;
                }
                // SET_DSPXA: byte offsets of the top and bottom field data.
                0x06 => {
                    field_offsets = Some((be16(packet, at)?, be16(packet, at + 2)?));
                    at += 4;
                }
                // CHG_COLCON: a length-prefixed colour-change table this decoder
                // does not apply. Skipped wholesale, and only if it fits.
                0x07 => {
                    let len = be16(packet, at)?;
                    if len < 2 || at.checked_add(len)? > packet_size {
                        return None;
                    }
                    at += len;
                }
                0xff => break,
                _ => return None,
            }
        }
        // The last sequence points at itself; anything not strictly forward is
        // the end (and can never loop).
        if next <= seq || next >= packet_size {
            break;
        }
        seq = next;
    }

    let (x, y, width, height) = area?;
    let (top_off, bottom_off) = field_offsets?;
    if width == 0 || height == 0 || width > MAX_CUE_DIM || height > MAX_CUE_DIM {
        return None;
    }
    let count = (width as usize).checked_mul(height as usize)?;
    if count > MAX_CUE_PIXELS {
        return None;
    }
    // The pixel data sits between the header and the control sequence, so that
    // is the bound on both fields: a field reaching into the control table means
    // the packet is truncated, not that the table is run lengths.
    if top_off < 4 || top_off > ctrl_start || bottom_off < 4 || bottom_off > ctrl_start {
        return None;
    }

    let mut pixels = vec![0u8; count];
    let w = width as usize;
    // Rows alternate between the two fields: even rows come from the top field's
    // stream, odd rows from the bottom field's.
    decode_field(
        packet,
        top_off,
        ctrl_start,
        w,
        height as usize,
        0,
        &mut pixels,
    )?;
    decode_field(
        packet,
        bottom_off,
        ctrl_start,
        w,
        height as usize,
        1,
        &mut pixels,
    )?;

    Some(SpuCue {
        start_ns: date_to_ns(start_date.unwrap_or(0)),
        stop_ns: stop_date.map(date_to_ns),
        x,
        y,
        width,
        height,
        colormap,
        alpha,
        pixels,
    })
}

/// A nibble-granular reader over the packet, the unit the run-length codes are
/// written in.
struct Nibbles<'a> {
    data: &'a [u8],
    /// Position in nibbles from the start of `data`.
    at: usize,
}

impl Nibbles<'_> {
    fn next(&mut self) -> Option<u32> {
        let byte = *self.data.get(self.at / 2)?;
        let v = if self.at % 2 == 0 {
            byte >> 4
        } else {
            byte & 0x0f
        };
        self.at += 1;
        Some(v as u32)
    }

    fn align(&mut self) {
        self.at = (self.at + 1) & !1;
    }
}

/// Decode one interlaced field's run-length data into every `first_row`-th row
/// (stepping by two) of `pixels`. Fails on a truncated stream rather than
/// leaving the rest of the bitmap silently unwritten.
fn decode_field(
    packet: &[u8],
    start: usize,
    end: usize,
    width: usize,
    height: usize,
    first_row: usize,
    pixels: &mut [u8],
) -> Option<()> {
    let mut r = Nibbles {
        data: packet.get(start..end)?,
        at: 0,
    };
    let mut row = first_row;
    while row < height {
        let mut x = 0usize;
        while x < width {
            // 4, 8, 12 or 16 bits, each width signalled by the previous nibbles
            // being small enough that the value could not have ended there.
            let mut v = r.next()?;
            if v < 0x4 {
                v = (v << 4) | r.next()?;
                if v < 0x10 {
                    v = (v << 4) | r.next()?;
                    if v < 0x40 {
                        v = (v << 4) | r.next()?;
                        if v < 0x4 {
                            // a zero run length means "to the end of the line"
                            v |= ((width - x) as u32) << 2;
                        }
                    }
                }
            }
            let run = (v >> 2) as usize;
            if run == 0 {
                return None; // no progress: malformed
            }
            let run = run.min(width - x);
            let base = row * width + x;
            pixels.get_mut(base..base + run)?.fill((v & 3) as u8);
            x += run;
        }
        r.align();
        row += 2;
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Build the SPU an encoder would write for a solid `w` x `h` rectangle of
    /// sample value 1 at (`x`, `y`), showing at once and hiding at `stop_date`.
    fn solid_spu(x: u32, y: u32, w: u32, h: u32, stop_date: u16) -> Vec<u8> {
        // one run per row, 16-bit "rest of line" code, byte aligned per row
        let rows_top = h.div_ceil(2) as usize;
        let rows_bottom = (h / 2) as usize;
        let line = [0x00u8, 0x01]; // 16 bits: run 0 (to end of line), colour 1
        let mut top = Vec::new();
        for _ in 0..rows_top {
            top.extend_from_slice(&line);
        }
        let mut bottom = Vec::new();
        for _ in 0..rows_bottom {
            bottom.extend_from_slice(&line);
        }
        let top_off = 4usize;
        let bottom_off = top_off + top.len();
        let data_end = bottom_off + bottom.len();

        let (x2, y2) = (x + w - 1, y + h - 1);
        let mut show = vec![
            0x03,
            0x32,
            0x10,
            0x04,
            0xff,
            0xf0,
            0x05,
            (x >> 4) as u8,
            (((x & 0xf) << 4) as u8) | ((x2 >> 8) as u8),
            x2 as u8,
            (y >> 4) as u8,
            (((y & 0xf) << 4) as u8) | ((y2 >> 8) as u8),
            y2 as u8,
            0x06,
        ];
        show.extend_from_slice(&(top_off as u16).to_be_bytes());
        show.extend_from_slice(&(bottom_off as u16).to_be_bytes());
        show.extend_from_slice(&[0x01, 0xff]);
        let hide = [0x02u8, 0xff];

        let seq1 = data_end;
        let seq2 = seq1 + 4 + show.len();
        let total = seq2 + 4 + hide.len();

        let mut spu = Vec::new();
        spu.extend_from_slice(&(total as u16).to_be_bytes());
        spu.extend_from_slice(&(seq1 as u16).to_be_bytes());
        spu.extend_from_slice(&top);
        spu.extend_from_slice(&bottom);
        spu.extend_from_slice(&0u16.to_be_bytes());
        spu.extend_from_slice(&(seq2 as u16).to_be_bytes());
        spu.extend_from_slice(&show);
        spu.extend_from_slice(&stop_date.to_be_bytes());
        spu.extend_from_slice(&(seq2 as u16).to_be_bytes());
        spu.extend_from_slice(&hide);
        assert_eq!(spu.len(), total);
        spu
    }

    #[test]
    fn parses_a_solid_rectangle_cue() {
        let cue = parse_spu(&solid_spu(10, 20, 8, 4, 180)).unwrap();
        assert_eq!((cue.x, cue.y, cue.width, cue.height), (10, 20, 8, 4));
        assert_eq!(cue.start_ns, 0);
        assert_eq!(cue.stop_ns, Some(2_048_000_000));
        assert_eq!(cue.colormap, [0, 1, 2, 3]);
        assert_eq!(cue.alpha, [0, 0x0f, 0x0f, 0x0f]);
        assert_eq!(cue.pixels, vec![1u8; 32]);
    }

    #[test]
    fn paints_through_the_idx_palette() {
        let cue = parse_spu(&solid_spu(1, 1, 2, 2, 90)).unwrap();
        let mut palette = [0u32; 16];
        palette[1] = 0x00ff_0000;
        let mut canvas = vec![0u8; 4 * 4 * 4];
        cue.paint(&palette, &mut canvas, 4, 4);
        // the 2x2 block at (1,1) is opaque red, everything else untouched
        let px = |x: usize, y: usize| &canvas[(y * 4 + x) * 4..(y * 4 + x) * 4 + 4];
        assert_eq!(px(1, 1), [255, 0, 0, 255]);
        assert_eq!(px(2, 2), [255, 0, 0, 255]);
        assert_eq!(px(0, 0), [0, 0, 0, 0]);
        assert_eq!(px(3, 3), [0, 0, 0, 0]);
    }

    #[test]
    fn clips_a_rectangle_overhanging_the_canvas() {
        let cue = parse_spu(&solid_spu(2, 2, 4, 4, 90)).unwrap();
        let mut palette = [0u32; 16];
        palette[1] = 0x0000_ff00;
        let mut canvas = vec![0u8; 4 * 4 * 4];
        cue.paint(&palette, &mut canvas, 4, 4);
        let px = |x: usize, y: usize| &canvas[(y * 4 + x) * 4..(y * 4 + x) * 4 + 4];
        assert_eq!(px(3, 3), [0, 255, 0, 255]);
        assert_eq!(px(1, 1), [0, 0, 0, 0]);
    }

    #[test]
    fn parses_an_idx_config() {
        let text = b"# comment\nsize: 720x576\npalette: 000000, ff0000, 00ff00, 0000ff, 04, 05, 06, 07, 08, 09, 0a, 0b, 0c, 0d, 0e, 0f\nlangidx: 0\n";
        let cfg = parse_idx(text).unwrap();
        assert_eq!(cfg.size, Some((720, 576)));
        let pal = cfg.palette.unwrap();
        assert_eq!(&pal[..4], &[0x000000, 0xff0000, 0x00ff00, 0x0000ff]);
        assert_eq!(pal[15], 0x0f);
    }

    #[test]
    fn rejects_a_short_palette_and_non_idx_bytes() {
        // the key is recognised, so it is still `.idx` text: just no usable palette
        assert_eq!(
            parse_idx(b"palette: 000000, ff0000\n").unwrap().palette,
            None
        );
        assert!(parse_idx(b"\x00\x01\x02\x03").is_none());
        assert!(parse_idx(b"nothing to see here\n").is_none());
    }

    /// Byte offset of the `SET_DSPXA` operands in a [`solid_spu`] packet: the
    /// show sequence's commands start four bytes into the sequence, and the two
    /// field offsets follow the fixed colour / contrast / area triple.
    fn dspxa_at(spu: &[u8]) -> usize {
        let ctrl = u16::from_be_bytes([spu[2], spu[3]]) as usize;
        ctrl + 4 + 14
    }

    #[test]
    fn rejects_a_packet_claiming_more_than_it_has() {
        let mut spu = solid_spu(0, 0, 4, 2, 90);
        let len = spu.len();
        spu[0..2].copy_from_slice(&((len + 64) as u16).to_be_bytes());
        assert!(parse_spu(&spu).is_none());
    }

    #[test]
    fn rejects_a_control_offset_outside_the_packet() {
        let mut spu = solid_spu(0, 0, 4, 2, 90);
        let len = spu.len();
        spu[2..4].copy_from_slice(&(len as u16).to_be_bytes());
        assert!(parse_spu(&spu).is_none());
        spu[2..4].copy_from_slice(&1u16.to_be_bytes());
        assert!(parse_spu(&spu).is_none());
    }

    #[test]
    fn rejects_an_inverted_display_rectangle() {
        // SET_DAREA with x1 = 256 and x2 = 0
        let spu = [
            0x00, 0x10, 0x00, 0x04, // packet size 16, control sequence at 4
            0x00, 0x00, 0x00, 0x10, // date 0, next past the packet: one sequence
            0x05, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff,
        ];
        assert!(parse_spu(&spu).is_none());
    }

    #[test]
    fn rejects_an_oversized_rectangle() {
        // 4096x4096 fits the 12-bit coordinates but not MAX_CUE_PIXELS
        let spu = [
            0x00, 0x15, 0x00, 0x04, // packet size 21, control sequence at 4
            0x00, 0x00, 0x00, 0x04, // date 0, next == self: one sequence
            0x05, 0x00, 0x0f, 0xff, 0x00, 0x0f, 0xff, // SET_DAREA 0..4095 both axes
            0x06, 0x00, 0x04, 0x00, 0x04, // SET_DSPXA, both fields at 4
            0xff,
        ];
        assert!(parse_spu(&spu).is_none());
    }

    #[test]
    fn rejects_field_offsets_outside_the_packet() {
        let mut spu = solid_spu(0, 0, 4, 2, 90);
        let at = dspxa_at(&spu);
        spu[at..at + 2].copy_from_slice(&0xfff0u16.to_be_bytes());
        assert!(parse_spu(&spu).is_none());
    }

    #[test]
    fn rejects_truncated_run_length_data() {
        let mut spu = solid_spu(0, 0, 64, 8, 90);
        // point the top field at the last two bytes of the data area: one row of
        // codes for a four-row field
        let ctrl = u16::from_be_bytes([spu[2], spu[3]]) as usize;
        let at = dspxa_at(&spu);
        spu[at..at + 2].copy_from_slice(&((ctrl - 2) as u16).to_be_bytes());
        assert!(parse_spu(&spu).is_none());
    }

    #[test]
    fn rejects_an_unknown_control_command() {
        let mut spu = solid_spu(0, 0, 4, 2, 90);
        let ctrl = u16::from_be_bytes([spu[2], spu[3]]) as usize;
        spu[ctrl + 4] = 0x7e;
        assert!(parse_spu(&spu).is_none());
    }

    #[test]
    fn a_cue_without_a_stop_command_has_no_duration() {
        let mut spu = solid_spu(0, 0, 4, 2, 90);
        // turn the trailing hide sequence's STP_DSP into a second start
        let at = spu.len() - 2;
        spu[at] = 0x01;
        let cue = parse_spu(&spu).unwrap();
        assert_eq!(cue.stop_ns, None);
    }
}
