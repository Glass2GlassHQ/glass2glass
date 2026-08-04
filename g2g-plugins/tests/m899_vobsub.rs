//! M899: VobSub (DVD subpicture) bitmap subtitles, validated against ffmpeg.
//!
//! ffmpeg cannot transcode text subtitles to a bitmap codec ("only possible from
//! text to text or bitmap to bitmap"), so the fixture starts as a hand-authored
//! `.idx` / `.sub` pair, and ffmpeg is the reference peer on both sides of it:
//! its `vobsub` demuxer reads the pair and its Matroska muxer writes the
//! `S_VOBSUB` track (`CodecPrivate`, block framing and timing all ffmpeg's), and
//! its `dvdsub` decoder renders the same cues through `overlay` for the
//! pixel-for-pixel comparison against g2g's decode composited by `Compositor`.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::path::PathBuf;
use std::process::Command;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, Dim, G2gError, MultiInputElement, OutputSink,
    PushOutcome, Rate, RawVideoFormat, SubPictureFormat,
};
use g2g_plugins::compositor::{Compositor, CompositorPad};
use g2g_plugins::conformance::persist;
use g2g_plugins::mkvdemux::{MkvDemux, MkvStream};
use g2g_plugins::registry::default_registry;
use g2g_plugins::vobsubdec::VobSubDec;

/// Subpicture display geometry, and therefore the canvas geometry both the
/// reference burn-in and the g2g compositing run at.
const W: u32 = 720;
const H: u32 = 576;

/// The `.idx` palette. Entry 0 is the transparent background; the rest are the
/// distinct colours the two cues index so a wrong colormap cannot pass.
const PALETTE: [u32; 16] = [
    0x000000, 0xff0000, 0x00ff00, 0x0000ff, 0xffff00, 0xff00ff, 0x00ffff, 0xffffff, 0x808080,
    0x404040, 0xc0c0c0, 0x202020, 0x800000, 0x008000, 0x000080, 0x808000,
];

/// One cue in the fixture: when it starts, where it sits, and which palette
/// entries its four sample values select.
struct Cue {
    pts_s: f64,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    colormap: [u8; 4],
}

/// The control-sequence stop date every cue uses: 180 units of 1024/90000 s,
/// exactly 2.048 s, so the decoded duration is an exact nanosecond count.
const STOP_DATE: u16 = 180;
const CUE_DURATION_NS: u64 = 2_048_000_000;

fn cues() -> Vec<Cue> {
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

const SECTOR: usize = 2048;

/// Author the `.idx` / `.sub` pair for [`cues`].
fn author_vobsub(idx_path: &PathBuf, sub_path: &PathBuf) {
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

// ---- ffmpeg ----

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
        && Command::new("ffprobe").arg("-version").output().is_ok()
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m899-{}-{name}", std::process::id()))
}

/// Fixture length in seconds, and therefore the reference burn-in's frame count
/// at one frame per second.
const DURATION_S: u32 = 9;

/// Let ffmpeg mux the authored cues into Matroska over an H.264 video track.
/// Everything about the resulting `S_VOBSUB` track (CodecID, `CodecPrivate`,
/// block framing, timestamps) is ffmpeg's.
fn mux_mkv(idx: &PathBuf, out: &PathBuf) {
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args([
            "-f",
            "lavfi",
            "-i",
            &format!("color=c=black:s=320x240:r=10:d={DURATION_S}"),
        ])
        .arg("-i")
        .arg(idx)
        .args(["-map", "0:v", "-map", "1:s"])
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:s", "copy"])
        .arg(out)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg muxed the S_VOBSUB fixture");
}

/// ffmpeg's own burn-in: its `dvdsub` decoder rendered through `overlay` onto
/// black, one RGB frame per second. Blending happens in RGB (not the default
/// YUV) so chroma subsampling cannot smear the reference.
fn reference_burn_in(idx: &PathBuf, raw: &PathBuf) -> Vec<Vec<u8>> {
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args([
            "-f",
            "lavfi",
            "-i",
            &format!("color=c=black:s={W}x{H}:r=1:d={DURATION_S}"),
        ])
        .arg("-i")
        .arg(idx)
        .args([
            "-filter_complex",
            "[0:v]format=rgba[b];[b][1:s]overlay=format=rgb[v]",
        ])
        .args(["-map", "[v]", "-pix_fmt", "rgb24", "-f", "rawvideo"])
        .arg(raw)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg burned the reference frames");
    let bytes = std::fs::read(raw).expect("read reference frames");
    let stride = (W * H * 3) as usize;
    assert_eq!(bytes.len(), stride * DURATION_S as usize);
    bytes.chunks(stride).map(|c| c.to_vec()).collect()
}

// ---- g2g plumbing ----

#[derive(Default)]
struct CaptureSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        self.packets.push(packet);
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

fn data(bytes: Vec<u8>, pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            ..FrameTiming::default()
        },
        0,
    ))
}

fn rgba_caps(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(25 << 16),
    }
}

/// One decoded canvas: its PTS, its declared duration, and its pixels.
struct Canvas {
    pts_ns: u64,
    duration_ns: u64,
    rgba: Vec<u8>,
}

/// Demux the fixture's `S_VOBSUB` track and decode it, both with the real
/// elements: `mkvdemux stream=vobsub ! vobsubdec`.
async fn demux_and_decode(mkv: &[u8]) -> Vec<Canvas> {
    let mut demux = MkvDemux::new().with_stream(MkvStream::VobSub);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        })
        .expect("mkvdemux accepts a Matroska byte stream");
    let mut demuxed = CaptureSink::default();
    for chunk in mkv.chunks(4096) {
        demux
            .process(data(chunk.to_vec(), 0), &mut demuxed)
            .await
            .expect("demux");
    }
    demux
        .process(PipelinePacket::Eos, &mut demuxed)
        .await
        .expect("demux eos");

    let mut dec = VobSubDec::new();
    dec.configure_pipeline(&Caps::SubPicture {
        format: SubPictureFormat::VobSub,
    })
    .expect("vobsubdec accepts a VobSub stream");
    let mut decoded = CaptureSink::default();
    for packet in demuxed.packets {
        dec.process(packet, &mut decoded).await.expect("decode");
    }
    decoded
        .packets
        .into_iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(Canvas {
                pts_ns: f.timing.pts_ns,
                duration_ns: f.timing.duration_ns,
                rgba: f.domain.as_system_slice().expect("system frame").to_vec(),
            }),
            _ => None,
        })
        .collect()
}

/// Bounding box of the non-transparent pixels of an RGBA canvas, and how many
/// there are. `None` for a fully transparent one.
fn opaque_bbox(rgba: &[u8]) -> Option<(u32, u32, u32, u32, usize)> {
    let (mut x0, mut y0, mut x1, mut y1) = (W, H, 0u32, 0u32);
    let mut count = 0usize;
    for y in 0..H {
        for x in 0..W {
            if rgba[((y * W + x) * 4 + 3) as usize] != 0 {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
                count += 1;
            }
        }
    }
    (count > 0).then_some((x0, y0, x1, y1, count))
}

/// Bounding box of the non-black pixels of an RGB frame, and how many.
fn lit_bbox(rgb: &[u8]) -> Option<(u32, u32, u32, u32, usize)> {
    let (mut x0, mut y0, mut x1, mut y1) = (W, H, 0u32, 0u32);
    let mut count = 0usize;
    for y in 0..H {
        for x in 0..W {
            let at = ((y * W + x) * 3) as usize;
            if rgb[at..at + 3] != [0, 0, 0] {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
                count += 1;
            }
        }
    }
    (count > 0).then_some((x0, y0, x1, y1, count))
}

#[test]
fn vobsubdec_and_its_gst_alias_build_from_a_launch_line() {
    let reg = default_registry();
    for name in ["vobsubdec", "dvdsubdec"] {
        assert!(
            reg.make_element(name).is_some(),
            "`{name}` builds from the registry"
        );
    }
    assert!(reg.element_names().contains(&"vobsubdec"));
    let graph = g2g_core::runtime::parse_launch(
        &reg,
        "filesrc location=x.mkv bytestream-format=matroska ! matroskademux stream=vobsub ! vobsubdec width=720 height=480 ! fakesink",
    );
    assert!(graph.is_ok(), "launch line parses: {:?}", graph.err());
}

#[tokio::test]
async fn ffmpeg_muxed_vobsub_cues_demux_and_decode_to_the_authored_rectangles() {
    if !have_ffmpeg() {
        eprintln!("skipping m899 vobsub decode: no ffmpeg on PATH");
        return;
    }
    let (idx, sub, mkv) = (
        temp_path("decode.idx"),
        temp_path("decode.sub"),
        temp_path("decode.mkv"),
    );
    author_vobsub(&idx, &sub);
    mux_mkv(&idx, &mkv);

    let canvases = demux_and_decode(&std::fs::read(&mkv).expect("read mkv")).await;
    // the opening empty canvas, then a shown / cleared pair per cue
    assert_eq!(canvases.len(), 1 + 2 * cues().len(), "canvas count");
    assert!(
        opaque_bbox(&canvases[0].rgba).is_none(),
        "the stream opens on an empty canvas"
    );

    for (i, cue) in cues().iter().enumerate() {
        let shown = &canvases[1 + i * 2];
        let cleared = &canvases[2 + i * 2];
        let pts = (cue.pts_s * 1_000_000_000.0) as u64;
        assert_eq!(shown.pts_ns, pts, "cue {i} shows at its block PTS");
        assert_eq!(
            shown.duration_ns, CUE_DURATION_NS,
            "cue {i} runs to the control sequence's stop date"
        );
        assert_eq!(
            cleared.pts_ns,
            pts + CUE_DURATION_NS,
            "cue {i} clears at its hide time"
        );
        assert!(
            opaque_bbox(&cleared.rgba).is_none(),
            "cue {i}'s clear canvas is fully transparent"
        );
        assert_eq!(
            opaque_bbox(&shown.rgba),
            Some((
                cue.x,
                cue.y,
                cue.x + cue.w - 1,
                cue.y + cue.h - 1,
                (cue.w * cue.h) as usize
            )),
            "cue {i} lands on its authored display rectangle"
        );
    }

    persist::record_evidence(
        "vobsubdec",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("vobsub")
            .detail("g2g demuxes and decodes an ffmpeg-muxed S_VOBSUB track to the authored cue rectangles and times"),
    )
    .expect("record oracle evidence");

    for p in [idx, sub, mkv] {
        let _ = std::fs::remove_file(p);
    }
}

#[tokio::test]
async fn compositing_the_decoded_cues_matches_ffmpegs_dvdsub_burn_in() {
    if !have_ffmpeg() {
        eprintln!("skipping m899 vobsub burn-in: no ffmpeg on PATH");
        return;
    }
    let (idx, sub, mkv, raw) = (
        temp_path("burn.idx"),
        temp_path("burn.sub"),
        temp_path("burn.mkv"),
        temp_path("burn.raw"),
    );
    author_vobsub(&idx, &sub);
    mux_mkv(&idx, &mkv);
    let reference = reference_burn_in(&idx, &raw);

    let canvases = demux_and_decode(&std::fs::read(&mkv).expect("read mkv")).await;

    // The compositor is the consumer the decoder is written for: the subtitle
    // canvases are a sparse overlay input it holds between frames, so the clear
    // canvas is what ends a cue.
    let mut comp = Compositor::new(
        W,
        H,
        Vec::from([
            CompositorPad::at(0, 0),
            CompositorPad::at(0, 0).with_zorder(1),
        ]),
    );
    comp.configure_pipeline(0, &rgba_caps(W, H)).expect("base");
    comp.configure_pipeline(1, &rgba_caps(W, H)).expect("cues");

    let black: Vec<u8> = [0u8, 0, 0, 255]
        .iter()
        .cycle()
        .take((W * H * 4) as usize)
        .copied()
        .collect();
    let mut composed = CaptureSink::default();
    let mut next = 0usize;
    for second in 0..DURATION_S as u64 {
        let now = second * 1_000_000_000;
        // Deliver every canvas whose PTS has arrived, in order, then the base
        // frame that releases one composited output.
        while next < canvases.len() && canvases[next].pts_ns <= now {
            comp.process(1, data(canvases[next].rgba.clone(), now), &mut composed)
                .await
                .expect("overlay input");
            next += 1;
        }
        comp.process(0, data(black.clone(), now), &mut composed)
            .await
            .expect("base input");
    }

    let frames: Vec<Vec<u8>> = composed
        .packets
        .into_iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => {
                Some(f.domain.as_system_slice().expect("system frame").to_vec())
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        frames.len(),
        DURATION_S as usize,
        "one output per base frame"
    );

    let mut lit_frames = 0;
    for (i, (ours, theirs)) in frames.iter().zip(&reference).enumerate() {
        assert_eq!(
            lit_bbox(&rgba_to_rgb(ours)),
            lit_bbox(theirs),
            "frame {i}: g2g's composited subtitle covers the same pixels as ffmpeg's burn-in"
        );
        let mut worst = 0u8;
        for px in 0..(W * H) as usize {
            for c in 0..3 {
                worst = worst.max(ours[px * 4 + c].abs_diff(theirs[px * 3 + c]));
            }
        }
        assert!(
            worst <= 2,
            "frame {i}: worst channel difference vs ffmpeg is {worst}"
        );
        if lit_bbox(theirs).is_some() {
            lit_frames += 1;
        }
    }
    // Two frames per cue sit strictly inside its display window at 1 fps; a run
    // that burned nothing would otherwise compare two black sequences.
    assert_eq!(
        lit_frames,
        2 * cues().len(),
        "ffmpeg burned a subtitle on the expected frames"
    );

    persist::record_evidence(
        "vobsubdec",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("vobsub")
            .detail("compositing g2g's decoded cues matches ffmpeg's dvdsub overlay burn-in pixel for pixel"),
    )
    .expect("record oracle evidence");

    for p in [idx, sub, mkv, raw] {
        let _ = std::fs::remove_file(p);
    }
}

fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4).flat_map(|p| p[..3].to_vec()).collect()
}
