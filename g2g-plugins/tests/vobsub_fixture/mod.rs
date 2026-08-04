//! A hand-authored VobSub `.idx` / `.sub` pair, shared by the bitmap-subtitle
//! oracles.
//!
//! ffmpeg cannot transcode text subtitles to a bitmap codec ("only possible from
//! text to text or bitmap to bitmap"), so every bitmap-subtitle fixture in this
//! suite starts here: these cues are authored byte by byte, ffmpeg's `vobsub`
//! demuxer reads the pair, and from there ffmpeg muxes them (M899) or transcodes
//! them bitmap-to-bitmap into another subtitle codec (M900).

#![allow(dead_code)]

use std::path::PathBuf;
use std::process::Command;

/// Subpicture display geometry, and therefore the canvas geometry both the
/// reference burn-in and the g2g compositing run at.
pub(crate) const W: u32 = 720;
pub(crate) const H: u32 = 576;

/// The `.idx` palette. Entry 0 is the transparent background; the rest are the
/// distinct colours the two cues index so a wrong colormap cannot pass.
pub(crate) const PALETTE: [u32; 16] = [
    0x000000, 0xff0000, 0x00ff00, 0x0000ff, 0xffff00, 0xff00ff, 0x00ffff, 0xffffff, 0x808080,
    0x404040, 0xc0c0c0, 0x202020, 0x800000, 0x008000, 0x000080, 0x808000,
];

/// The control-sequence stop date every cue uses: 180 units of 1024/90000 s,
/// exactly 2.048 s, so the decoded duration is an exact nanosecond count.
pub(crate) const STOP_DATE: u16 = 180;
pub(crate) const CUE_DURATION_NS: u64 = 2_048_000_000;

const SECTOR: usize = 2048;

/// One cue in the fixture: when it starts, where it sits, and which palette
/// entries its four sample values select.
pub(crate) struct Cue {
    pub(crate) pts_s: f64,
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) w: u32,
    pub(crate) h: u32,
    pub(crate) colormap: [u8; 4],
}

pub(crate) fn cues() -> Vec<Cue> {
    Vec::from([
        Cue {
            pts_s: 1.5,
            x: 100,
            y: 400,
            w: 200,
            h: 60,
            colormap: [0, 1, 2, 3],
        },
        Cue {
            pts_s: 5.5,
            x: 300,
            y: 100,
            w: 120,
            h: 80,
            colormap: [0, 4, 5, 6],
        },
    ])
}

pub(crate) fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
        && Command::new("ffprobe").arg("-version").output().is_ok()
}

// ---- fixture authoring: SPU packets in an MPEG-PS `.sub` ----

/// Nibble-granular writer: run-length codes are 4, 8, 12 or 16 bits wide.
#[derive(Default)]
struct NibbleW {
    nibbles: Vec<u8>,
}

impl NibbleW {
    fn put(&mut self, v: u32, bits: u32) {
        for shift in (0..bits).step_by(4).rev() {
            self.nibbles.push(((v >> shift) & 0xf) as u8);
        }
    }
    fn align(&mut self) {
        if self.nibbles.len() % 2 == 1 {
            self.nibbles.push(0);
        }
    }
    fn bytes(mut self) -> Vec<u8> {
        self.align();
        self.nibbles.chunks(2).map(|c| (c[0] << 4) | c[1]).collect()
    }
}

/// Run-length encode one interlaced field, each row byte-aligned.
fn encode_field(rows: &[Vec<u8>], w: usize) -> Vec<u8> {
    let mut nw = NibbleW::default();
    for row in rows {
        let mut x = 0usize;
        while x < w {
            let c = row[x] as u32;
            let mut run = 1usize;
            while x + run < w && row[x + run] as u32 == c {
                run += 1;
            }
            x += run;
            if x == w {
                // "to the end of the line": a zero run length in a 16-bit code
                nw.put(c, 16);
                continue;
            }
            let mut left = run;
            while left > 0 {
                let n = left.min(0xff);
                let v = ((n as u32) << 2) | c;
                let bits = match n {
                    0..=3 => 4,
                    4..=15 => 8,
                    16..=63 => 12,
                    _ => 16,
                };
                nw.put(v, bits);
                left -= n;
            }
        }
        nw.align();
    }
    nw.bytes()
}

/// A `w` x `h` block of sample value 1 inside a one-pixel border of sample
/// value 3: single-pixel runs and whole-line runs in the same bitmap.
fn bordered_box(w: usize, h: usize) -> Vec<Vec<u8>> {
    (0..h)
        .map(|y| {
            (0..w)
                .map(|x| {
                    if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
                        3
                    } else {
                        1
                    }
                })
                .collect()
        })
        .collect()
}

/// Build the subpicture unit for one cue: header, two interlaced RLE fields,
/// then a show control sequence and a hide one.
fn spu(cue: &Cue) -> Vec<u8> {
    let bitmap = bordered_box(cue.w as usize, cue.h as usize);
    let top: Vec<Vec<u8>> = bitmap.iter().step_by(2).cloned().collect();
    let bottom: Vec<Vec<u8>> = bitmap.iter().skip(1).step_by(2).cloned().collect();
    let top_data = encode_field(&top, cue.w as usize);
    let bottom_data = encode_field(&bottom, cue.w as usize);
    let top_off = 4usize;
    let bottom_off = top_off + top_data.len();
    let data_end = bottom_off + bottom_data.len();

    let cm = cue.colormap;
    // sample value 0 is transparent, the other three opaque
    let alpha = [0u8, 0xf, 0xf, 0xf];
    let (x1, y1) = (cue.x, cue.y);
    let (x2, y2) = (cue.x + cue.w - 1, cue.y + cue.h - 1);
    let mut show = Vec::from([
        0x03,
        (cm[3] << 4) | cm[2],
        (cm[1] << 4) | cm[0],
        0x04,
        (alpha[3] << 4) | alpha[2],
        (alpha[1] << 4) | alpha[0],
        0x05,
        (x1 >> 4) as u8,
        (((x1 & 0xf) << 4) | (x2 >> 8)) as u8,
        x2 as u8,
        (y1 >> 4) as u8,
        (((y1 & 0xf) << 4) | (y2 >> 8)) as u8,
        y2 as u8,
        0x06,
    ]);
    show.extend_from_slice(&(top_off as u16).to_be_bytes());
    show.extend_from_slice(&(bottom_off as u16).to_be_bytes());
    show.extend_from_slice(&[0x01, 0xff]);
    let hide = [0x02u8, 0xff];

    let seq1 = data_end;
    let seq2 = seq1 + 4 + show.len();
    let total = seq2 + 4 + hide.len();

    let mut out = Vec::new();
    out.extend_from_slice(&(total as u16).to_be_bytes());
    out.extend_from_slice(&(seq1 as u16).to_be_bytes());
    out.extend_from_slice(&top_data);
    out.extend_from_slice(&bottom_data);
    out.extend_from_slice(&0u16.to_be_bytes()); // show at once
    out.extend_from_slice(&(seq2 as u16).to_be_bytes());
    out.extend_from_slice(&show);
    out.extend_from_slice(&STOP_DATE.to_be_bytes());
    out.extend_from_slice(&(seq2 as u16).to_be_bytes()); // last sequence: self
    out.extend_from_slice(&hide);
    assert_eq!(out.len(), total);
    out
}

/// A 33-bit MPEG timestamp in the five-byte PES field layout.
fn pts_field(pts90: u64) -> [u8; 5] {
    let p = pts90 & 0x1_ffff_ffff;
    [
        0x20 | ((p >> 29) as u8 & 0x0e) | 1,
        (p >> 22) as u8,
        (p >> 14) as u8 | 1,
        (p >> 7) as u8,
        ((p << 1) as u8) | 1,
    ]
}

/// MPEG-2 program-stream pack header (start code plus the 10-byte SCR / mux-rate
/// bit field). ffmpeg's `vobsub` demuxer seeks to a pack, so each cue starts on
/// one.
fn pack_header(scr90: u64) -> Vec<u8> {
    let s = scr90 & 0x1_ffff_ffff;
    let mut bits: Vec<u8> = Vec::new();
    let mut push = |val: u64, n: u32| {
        for i in (0..n).rev() {
            bits.push(((val >> i) & 1) as u8);
        }
    };
    push(0b01, 2);
    push(s >> 30, 3);
    push(1, 1);
    push(s >> 15, 15);
    push(1, 1);
    push(s, 15);
    push(1, 1);
    push(0, 9); // SCR extension
    push(1, 1);
    push(10_000, 22); // program_mux_rate
    push(1, 1);
    push(1, 1);
    push(0x1f, 5);
    push(0, 3); // pack_stuffing_length
    assert_eq!(bits.len(), 80);
    let mut out = Vec::from([0x00u8, 0x00, 0x01, 0xba]);
    for chunk in bits.chunks(8) {
        out.push(chunk.iter().fold(0u8, |acc, b| (acc << 1) | b));
    }
    out
}

/// One `private_stream_1` PES packet carrying subpicture substream 0x20.
fn pes(spu: &[u8], pts90: u64) -> Vec<u8> {
    let mut body = Vec::from([0x81u8, 0x80, 0x05]);
    body.extend_from_slice(&pts_field(pts90));
    body.push(0x20);
    body.extend_from_slice(spu);
    let mut out = Vec::from([0x00u8, 0x00, 0x01, 0xbd]);
    out.extend_from_slice(&(body.len() as u16).to_be_bytes());
    out.extend_from_slice(&body);
    out
}

/// MPEG padding stream of exactly `n` bytes, so each cue starts on a 2 KiB
/// sector boundary the way a real `.sub` does.
fn padding(n: usize) -> Vec<u8> {
    let mut out = Vec::from([0x00u8, 0x00, 0x01, 0xbe]);
    out.extend_from_slice(&((n - 6) as u16).to_be_bytes());
    out.resize(n, 0xff);
    out
}

/// Author the `.idx` / `.sub` pair for [`cues`].
pub(crate) fn author_vobsub(idx_path: &PathBuf, sub_path: &PathBuf) {
    let mut sub: Vec<u8> = Vec::new();
    let mut index = String::new();
    for cue in cues() {
        let filepos = sub.len();
        let pts90 = (cue.pts_s * 90_000.0) as u64;
        let mut block = pack_header(pts90);
        block.extend_from_slice(&pes(&spu(&cue), pts90));
        let mut rem = (SECTOR - block.len() % SECTOR) % SECTOR;
        if rem > 0 && rem < 6 {
            rem += SECTOR;
        }
        if rem > 0 {
            block.extend_from_slice(&padding(rem));
        }
        sub.extend_from_slice(&block);
        let ms = (cue.pts_s * 1000.0).round() as u64;
        index.push_str(&format!(
            "timestamp: {:02}:{:02}:{:02}:{:03}, filepos: {filepos:09x}\n",
            ms / 3_600_000,
            (ms / 60_000) % 60,
            (ms / 1000) % 60,
            ms % 1000
        ));
    }
    let palette = PALETTE
        .iter()
        .map(|c| format!("{c:06x}"))
        .collect::<Vec<_>>()
        .join(", ");
    let idx = format!(
        "# VobSub index file, v7 (do not modify this line!)\nsize: {W}x{H}\npalette: {palette}\nlangidx: 0\n\nid: en, index: 0\n{index}"
    );
    std::fs::write(idx_path, idx).expect("write .idx");
    std::fs::write(sub_path, sub).expect("write .sub");
}
