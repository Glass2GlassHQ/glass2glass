//! M925: Blu-ray PGS (HDMV) bitmap subtitles, validated against ffmpeg.
//!
//! ffmpeg has no PGS *encoder*, so the fixture is a hand-authored `.sup` file:
//! two cues, each an epoch of its own, bracketed by the empty presentation
//! composition that ends them. ffmpeg is the reference peer on both sides of it.
//! It reads that `.sup` and remuxes it to Matroska, so the `S_HDMV/PGS` track
//! framing and block timing g2g demuxes are ffmpeg's; and its own `pgssub`
//! decoder renders the same cues to RGBA through `sub2video` (a subtitle stream
//! feeding a video filter), which copies the decoded palette straight into a
//! transparent frame, for a pixel-for-pixel comparison against g2g's canvases.
//!
//! Cropped composition objects are the one thing that comparison cannot reach:
//! ffmpeg parses the crop rectangle and then never applies it (`pgssubdec.c`
//! carries a "TODO: Implement cropping"), so mpv and VLC inherit that and there
//! is no decoder here to compare against. The crop tests below stand on the
//! uncropped path instead. Cropping is a pure selection, so the same object
//! presented cropped must paint exactly the window of what it paints whole, and
//! the whole case is what ffmpeg already pins pixel for pixel. Which placement
//! that window lands at is the part no such test can settle, and it is taken
//! from libbluray's `graphics_controller.c`, which does implement cropping and
//! draws the cropped rectangle at the composition position with the crop offset
//! indexing the object bitmap only.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::path::PathBuf;
use std::process::Command;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, G2gError, OutputSink, PushOutcome, SubPictureFormat,
};
use g2g_plugins::conformance::persist;
use g2g_plugins::mkvdemux::{MkvDemux, MkvStream};
use g2g_plugins::pgsdec::PgsDec;
use g2g_plugins::registry::default_registry;

/// The fixture's video descriptor. Over 576 lines, so a reference decoder reads
/// the palette through BT.709 and not BT.601: the two matrices disagree by tens
/// of levels on these entries, so the pixel comparison pins the choice.
const W: u32 = 1280;
const H: u32 = 720;

/// The fixture palette, `(entry, Y, Cr, Cb, alpha)`, limited range.
const PALETTE: [(u8, u8, u8, u8, u8); 5] = [
    (1, 235, 128, 128, 255), // white
    (2, 63, 102, 240, 255),  // blue
    (3, 173, 26, 20, 255),   // green
    (4, 63, 240, 102, 255),  // red
    (5, 235, 128, 128, 128), // white at half alpha
];

/// What entry 1 and 5 decode to through BT.709 (the same colour, two alphas).
const WHITE: [u8; 4] = [255, 255, 255, 255];
const HALF_WHITE: [u8; 4] = [255, 255, 255, 128];
const BLUE: [u8; 4] = [8, 45, 255, 255];

/// One authored cue: when it shows, where it sits, and how big its object is.
struct Cue {
    pts_90k: u32,
    hide_90k: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

fn cues() -> Vec<Cue> {
    Vec::from([
        Cue {
            pts_90k: 90_000,
            hide_90k: 162_000,
            x: 100,
            y: 500,
            w: 200,
            h: 60,
        },
        Cue {
            pts_90k: 180_000,
            hide_90k: 270_000,
            x: 640,
            y: 40,
            w: 300,
            h: 80,
        },
    ])
}

// ---- fixture authoring ----

/// One `.sup`-framed segment: `PG`, a 90 kHz PTS, a DTS, the type and the length.
fn segment(kind: u8, pts_90k: u32, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::from(*b"PG");
    out.extend_from_slice(&pts_90k.to_be_bytes());
    out.extend_from_slice(&0u32.to_be_bytes());
    out.push(kind);
    out.extend_from_slice(&(body.len() as u16).to_be_bytes());
    out.extend_from_slice(body);
    out
}

/// One composition object reference: which object, where it goes, and the crop
/// rectangle inside the object bitmap when the composition crops it.
struct CompRef {
    id: u16,
    x: u32,
    y: u32,
    crop: Option<(u16, u16, u16, u16)>,
    forced: bool,
}

impl CompRef {
    fn at(id: u16, x: u32, y: u32) -> Self {
        Self {
            id,
            x,
            y,
            crop: None,
            forced: false,
        }
    }

    fn cropped(mut self, x: u16, y: u16, w: u16, h: u16) -> Self {
        self.crop = Some((x, y, w, h));
        self
    }

    fn forced(mut self) -> Self {
        self.forced = true;
        self
    }
}

/// Presentation composition: the video descriptor, the composition descriptor,
/// then one reference per composition object, 8 bytes or 16 when it crops.
fn pcs(objects: &[CompRef], comp_number: u16, state: u8, pts_90k: u32) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&(W as u16).to_be_bytes());
    b.extend_from_slice(&(H as u16).to_be_bytes());
    b.push(0x10); // frame rate, ignored by every decoder
    b.extend_from_slice(&comp_number.to_be_bytes());
    b.push(state);
    b.push(0x00); // palette update flag
    b.push(0x00); // palette id
    b.push(objects.len() as u8);
    for o in objects {
        // 0x80 cropped, 0x40 forced
        let mut flags = 0u8;
        if o.crop.is_some() {
            flags |= 0x80;
        }
        if o.forced {
            flags |= 0x40;
        }
        b.extend_from_slice(&o.id.to_be_bytes());
        b.push(0); // window id
        b.push(flags);
        b.extend_from_slice(&(o.x as u16).to_be_bytes());
        b.extend_from_slice(&(o.y as u16).to_be_bytes());
        if let Some((cx, cy, cw, ch)) = o.crop {
            for v in [cx, cy, cw, ch] {
                b.extend_from_slice(&v.to_be_bytes());
            }
        }
    }
    segment(0x16, pts_90k, &b)
}

/// Window definition: one window, the rectangle the object is drawn inside.
fn wds(cue: &Cue, pts_90k: u32) -> Vec<u8> {
    let mut b = Vec::from([1u8, 0u8]);
    for v in [cue.x, cue.y, cue.w, cue.h] {
        b.extend_from_slice(&(v as u16).to_be_bytes());
    }
    segment(0x17, pts_90k, &b)
}

fn pds(entries: &[(u8, u8, u8, u8, u8)], pts_90k: u32) -> Vec<u8> {
    let mut b = Vec::from([0u8, 0u8]); // palette id, version
    for &(entry, y, cr, cb, a) in entries {
        b.extend_from_slice(&[entry, y, cr, cb, a]);
    }
    segment(0x14, pts_90k, &b)
}

/// Encode one line of `(count, colour)` runs, then the end-of-line code. Short
/// runs, single pixels, 14-bit long runs and colour-0 gaps all appear, so the
/// whole RLE grammar is exercised.
fn rle_line(runs: &[(u32, u8)]) -> Vec<u8> {
    let mut out = Vec::new();
    for &(n, c) in runs {
        match (c, n) {
            (0, n) if n < 64 => out.extend_from_slice(&[0x00, n as u8]),
            (0, n) => out.extend_from_slice(&[0x00, 0x40 | (n >> 8) as u8, n as u8]),
            (c, 1) => out.push(c),
            (c, n) if n < 64 => out.extend_from_slice(&[0x00, 0x80 | n as u8, c]),
            (c, n) => out.extend_from_slice(&[0x00, 0xC0 | (n >> 8) as u8, n as u8, c]),
        }
    }
    out.extend_from_slice(&[0x00, 0x00]);
    out
}

fn bitmap_rle(w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for y in 0..h {
        let runs: Vec<(u32, u8)> = if y == 0 {
            Vec::from([(w, 1)])
        } else if y == 1 {
            Vec::from([(1, 2), (w - 2, 0), (1, 2)])
        } else if y == h - 1 {
            Vec::from([(w, 5)])
        } else if y % 2 == 0 {
            Vec::from([(10, 3), (w - 20, 0), (10, 4)])
        } else {
            Vec::from([(w / 2, 4), (w - w / 2, 3)])
        };
        out.extend_from_slice(&rle_line(&runs));
    }
    out
}

/// Object definition, one fragment: the declared length counts the two dimension
/// fields as well as the RLE.
fn ods(id: u16, w: u32, h: u32, rle: &[u8], pts_90k: u32) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&id.to_be_bytes());
    b.extend_from_slice(&[0x00, 0xC0]); // version, first and last fragment
    b.extend_from_slice(&((rle.len() + 4) as u32).to_be_bytes()[1..]);
    b.extend_from_slice(&(w as u16).to_be_bytes());
    b.extend_from_slice(&(h as u16).to_be_bytes());
    b.extend_from_slice(rle);
    segment(0x15, pts_90k, &b)
}

/// The whole `.sup`: per cue an epoch-start display set, then the empty
/// composition that ends it.
fn author_sup() -> Vec<u8> {
    let mut out = Vec::new();
    for (i, cue) in cues().iter().enumerate() {
        let n = i as u16 * 2;
        out.extend_from_slice(&pcs(&[CompRef::at(1, cue.x, cue.y)], n, 0x80, cue.pts_90k));
        out.extend_from_slice(&wds(cue, cue.pts_90k));
        out.extend_from_slice(&pds(&PALETTE, cue.pts_90k));
        out.extend_from_slice(&ods(
            1,
            cue.w,
            cue.h,
            &bitmap_rle(cue.w, cue.h),
            cue.pts_90k,
        ));
        out.extend_from_slice(&segment(0x80, cue.pts_90k, &[]));

        out.extend_from_slice(&pcs(&[], n + 1, 0x00, cue.hide_90k));
        out.extend_from_slice(&wds(cue, cue.hide_90k));
        out.extend_from_slice(&segment(0x80, cue.hide_90k, &[]));
    }
    out
}

// ---- crop fixture ----

/// The cropping fixture's object. Every pixel of it is a different palette
/// entry, and every entry a different colour, so a crop that drops the offset,
/// swaps the axes or is off by one lands on colours that do not match. 16 x 12
/// keeps every code inside a byte.
const CROP_W: u32 = 16;
const CROP_H: u32 = 12;

/// Palette entry for one object pixel: 1..=192, never 0, since a zero byte in
/// the RLE escapes to a run code instead of standing for a pixel.
fn crop_code(x: u32, y: u32) -> u8 {
    (y * CROP_W + x + 1) as u8
}

/// One entry per code, each a different limited-range luma, so each decodes to
/// its own RGB. Alpha is opaque throughout: every object pixel paints, which is
/// what makes "nothing outside the crop rectangle" a real assertion.
fn crop_palette() -> Vec<(u8, u8, u8, u8, u8)> {
    (0..CROP_W * CROP_H)
        .map(|i| {
            let code = (i + 1) as u8;
            (code, 20 + code, 128, 128, 255)
        })
        .collect()
}

/// The object's RLE: one literal byte per pixel, then the end-of-line code.
fn crop_rle() -> Vec<u8> {
    let mut out = Vec::new();
    for y in 0..CROP_H {
        for x in 0..CROP_W {
            out.push(crop_code(x, y));
        }
        out.extend_from_slice(&[0x00, 0x00]);
    }
    out
}

/// One epoch-start display set placing the crop fixture's object however the
/// references say, with an object definition per distinct id.
fn crop_display_set(objects: &[CompRef]) -> Vec<u8> {
    let pts = 90_000;
    let mut out = pcs(objects, 0, 0x80, pts);
    out.extend_from_slice(&pds(&crop_palette(), pts));
    let rle = crop_rle();
    let mut defined: Vec<u16> = Vec::new();
    for o in objects {
        if defined.contains(&o.id) {
            continue;
        }
        defined.push(o.id);
        out.extend_from_slice(&ods(o.id, CROP_W, CROP_H, &rle, pts));
    }
    out.extend_from_slice(&segment(0x80, pts, &[]));
    out
}

// ---- ffmpeg ----

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m925-{}-{name}", std::process::id()))
}

/// Remux the `.sup` to Matroska, so the `S_HDMV/PGS` track (codec id, block
/// framing, timestamps) is ffmpeg's and not ours.
fn mux_mkv(sup: &PathBuf, mkv: &PathBuf) {
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .arg("-i")
        .arg(sup)
        .args(["-c:s", "copy"])
        .arg(mkv)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg remuxed the .sup to Matroska");
}

/// ffmpeg's own decode: `sub2video` turns the `pgssub` decoder's output into
/// RGBA frames the size of the video, copying the palette entries straight in
/// (no blending), which is exactly the canvas g2g produces. Consecutive
/// duplicates are ffmpeg's frame-rate padding, so only the distinct painted
/// frames come back.
fn reference_canvases(sup: &PathBuf, raw: &PathBuf) -> Vec<Vec<u8>> {
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .arg("-i")
        .arg(sup)
        .args(["-filter_complex", "[0:s:0]copy[v]", "-map", "[v]"])
        .args([
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgba",
            "-f",
            "rawvideo",
        ])
        .arg(raw)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg decoded the PGS fixture");
    let bytes = std::fs::read(raw).expect("read reference frames");
    let stride = (W * H * 4) as usize;
    assert_eq!(bytes.len() % stride, 0, "whole RGBA frames");
    let mut out: Vec<Vec<u8>> = Vec::new();
    for frame in bytes.chunks(stride) {
        if frame.chunks_exact(4).all(|p| p[3] == 0) {
            continue;
        }
        if out.last().map(|p| p.as_slice()) == Some(frame) {
            continue;
        }
        out.push(frame.to_vec());
    }
    out
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

/// One decoded canvas: its PTS and its pixels.
struct Canvas {
    pts_ns: u64,
    rgba: Vec<u8>,
}

/// Demux the fixture's `S_HDMV/PGS` track and decode it, both with the real
/// elements: `mkvdemux stream=pgs ! pgsdec`.
async fn demux_and_decode(mkv: &[u8]) -> Vec<Canvas> {
    let mut demux = MkvDemux::new().with_stream(MkvStream::Pgs);
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

    let mut dec = PgsDec::new();
    dec.configure_pipeline(&Caps::SubPicture {
        format: SubPictureFormat::Pgs,
    })
    .expect("pgsdec accepts a PGS stream");
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

fn px(rgba: &[u8], x: u32, y: u32) -> [u8; 4] {
    let at = ((y * W + x) * 4) as usize;
    rgba[at..at + 4].try_into().expect("rgba pixel")
}

/// Where two canvases first disagree, as `(x, y)`. Canvases are megabytes, so a
/// coordinate is what a failure can actually be read from.
fn first_difference(got: &[u8], want: &[u8]) -> Option<(u32, u32)> {
    assert_eq!(got.len(), want.len(), "canvas size");
    got.chunks_exact(4)
        .zip(want.chunks_exact(4))
        .position(|(a, b)| a != b)
        .map(|i| (i as u32 % W, i as u32 / W))
}

/// Copy a `size` window of one canvas into another at a new position.
fn paste(dst: &mut [u8], src: &[u8], from: (u32, u32), to: (u32, u32), size: (u32, u32)) {
    let n = (size.0 * 4) as usize;
    for row in 0..size.1 {
        let s = (((from.1 + row) * W + from.0) * 4) as usize;
        let d = (((to.1 + row) * W + to.0) * 4) as usize;
        dst[d..d + n].copy_from_slice(&src[s..s + n]);
    }
}

/// Decode one `.sup`-framed display set through the real `pgsdec` element and
/// return the canvas it paints, or `None` when it paints nothing. The element
/// opens on an empty canvas before any display set, which is checked here rather
/// than left for every caller.
async fn crop_canvas(objects: &[CompRef]) -> Option<Vec<u8>> {
    let mut dec = PgsDec::new();
    dec.configure_pipeline(&Caps::SubPicture {
        format: SubPictureFormat::Pgs,
    })
    .expect("pgsdec accepts a PGS stream");
    let mut sink = CaptureSink::default();
    dec.process(data(crop_display_set(objects), 0), &mut sink)
        .await
        .expect("decode");
    let mut canvases: Vec<Vec<u8>> = sink
        .packets
        .into_iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => {
                Some(f.domain.as_system_slice().expect("system frame").to_vec())
            }
            _ => None,
        })
        .collect();
    assert!(
        canvases[0].iter().all(|&b| b == 0),
        "the element opens on an empty canvas"
    );
    assert!(canvases.len() <= 2, "one display set, at most one canvas");
    (canvases.len() == 2).then(|| canvases.remove(1))
}

#[test]
fn pgsdec_builds_from_a_launch_line() {
    let reg = default_registry();
    assert!(reg.make_element("pgsdec").is_some(), "`pgsdec` builds");
    assert!(reg.element_names().contains(&"pgsdec"));
    let graph = g2g_core::runtime::parse_launch(
        &reg,
        "filesrc location=x.mkv bytestream-format=matroska ! matroskademux stream=pgs ! pgsdec width=1280 height=720 forced-subs-only=false ! fakesink",
    );
    assert!(graph.is_ok(), "launch line parses: {:?}", graph.err());
}

#[tokio::test]
async fn ffmpeg_muxed_pgs_display_sets_demux_and_decode_pixel_for_pixel() {
    if !have_ffmpeg() {
        eprintln!("skipping m925 pgs decode: no ffmpeg on PATH");
        return;
    }
    let (sup, mkv, raw) = (
        temp_path("fixture.sup"),
        temp_path("fixture.mkv"),
        temp_path("reference.raw"),
    );
    std::fs::write(&sup, author_sup()).expect("write .sup");
    mux_mkv(&sup, &mkv);

    let canvases = demux_and_decode(&std::fs::read(&mkv).expect("read mkv")).await;
    // the opening empty canvas, then a shown / cleared pair per cue
    assert_eq!(canvases.len(), 1 + 2 * cues().len(), "canvas count");
    assert!(
        opaque_bbox(&canvases[0].rgba).is_none(),
        "the stream opens on an empty canvas"
    );
    assert_eq!(
        canvases[0].rgba.len(),
        (W * H * 4) as usize,
        "the opening canvas is already at the video geometry"
    );

    // The remux normalizes the first cue to zero, so every PTS shifts with it.
    let start_90k = cues()[0].pts_90k as u64;
    let to_ns = |t: u32| (t as u64 - start_90k) * 1_000_000_000 / 90_000;

    for (i, cue) in cues().iter().enumerate() {
        let shown = &canvases[1 + i * 2];
        let cleared = &canvases[2 + i * 2];
        assert_eq!(
            shown.pts_ns,
            to_ns(cue.pts_90k),
            "cue {i} shows at its block PTS"
        );
        assert_eq!(
            cleared.pts_ns,
            to_ns(cue.hide_90k),
            "cue {i} clears when the empty composition arrives"
        );
        assert!(
            opaque_bbox(&cleared.rgba).is_none(),
            "cue {i}'s clear canvas is fully transparent"
        );
        let bbox = opaque_bbox(&shown.rgba).expect("the cue paints something");
        assert_eq!(
            (bbox.0, bbox.1, bbox.2, bbox.3),
            (cue.x, cue.y, cue.x + cue.w - 1, cue.y + cue.h - 1),
            "cue {i} lands on its authored object rectangle"
        );

        // Row 0 is a solid run of entry 1, row 1 a single entry-2 pixel at each
        // edge with a transparent gap between, and the last row entry 5, whose
        // alpha has to pass through unscaled.
        assert_eq!(px(&shown.rgba, cue.x, cue.y), WHITE);
        assert_eq!(px(&shown.rgba, cue.x + cue.w - 1, cue.y), WHITE);
        assert_eq!(px(&shown.rgba, cue.x, cue.y + 1), BLUE);
        assert_eq!(px(&shown.rgba, cue.x + cue.w - 1, cue.y + 1), BLUE);
        assert_eq!(px(&shown.rgba, cue.x + 1, cue.y + 1), [0, 0, 0, 0]);
        assert_eq!(px(&shown.rgba, cue.x + 5, cue.y + cue.h - 1), HALF_WHITE);
        // Just outside the object stays untouched.
        assert_eq!(px(&shown.rgba, cue.x - 1, cue.y), [0, 0, 0, 0]);
    }

    // ffmpeg's own decode of the same bitstream, pixel for pixel.
    let reference = reference_canvases(&sup, &raw);
    assert_eq!(reference.len(), cues().len(), "one painted frame per cue");
    for (i, want) in reference.iter().enumerate() {
        let got = &canvases[1 + i * 2].rgba;
        assert_eq!(got.len(), want.len(), "cue {i} canvas size");
        let differing = got
            .chunks_exact(4)
            .zip(want.chunks_exact(4))
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            differing, 0,
            "cue {i} decodes to ffmpeg's pixels, palette and alpha included"
        );
    }

    persist::record_evidence(
        "pgsdec",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("pgs")
            .detail(
                "g2g demuxes an ffmpeg-muxed S_HDMV/PGS track and decodes it to ffmpeg's own pgssub pixels",
            ),
    )
    .expect("record oracle evidence");

    for p in [sup, mkv, raw] {
        let _ = std::fs::remove_file(p);
    }
}

/// The anchor for everything below: cropping is a pure selection, so a cropped
/// composition object must paint exactly the window of what the same object
/// paints uncropped. The uncropped path is the one the ffmpeg comparison above
/// pins pixel for pixel, so tying the cropped path to it carries that evidence
/// across instead of asserting the crop is right on its own authority.
#[tokio::test]
async fn a_cropped_object_paints_the_uncropped_canvas_windowed() {
    let (ox, oy) = (100u32, 50u32);
    let whole = crop_canvas(&[CompRef::at(1, ox, oy)])
        .await
        .expect("the whole object paints");

    // Without this the oracle would pass under a broken crop: a rectangle of one
    // colour matches any other rectangle of it.
    let mut seen = std::collections::HashSet::new();
    for y in 0..CROP_H {
        for x in 0..CROP_W {
            assert!(
                seen.insert(px(&whole, ox + x, oy + y)),
                "object pixel ({x}, {y}) repeats a colour, so a misread crop could still match"
            );
        }
    }

    // Offset nonzero in both axes: a decoder that takes the top-left corner
    // instead paints entirely different entries.
    let (cx, cy, cw, ch) = (3u32, 2u32, 9u32, 7u32);
    let cropped = crop_canvas(&[
        CompRef::at(1, ox + cx, oy + cy).cropped(cx as u16, cy as u16, cw as u16, ch as u16)
    ])
    .await
    .expect("the cropped object paints");

    let mut want = vec![0u8; whole.len()];
    paste(
        &mut want,
        &whole,
        (ox + cx, oy + cy),
        (ox + cx, oy + cy),
        (cw, ch),
    );
    assert_eq!(
        first_difference(&cropped, &want),
        None,
        "the crop paints the uncropped object's own pixels over that rectangle and nothing else"
    );
}

/// Cropping to the whole object is the identity, which is what says the crop
/// path and the plain path are the same code taking the same pixels.
#[tokio::test]
async fn a_crop_of_the_whole_object_equals_the_uncropped_composition() {
    let (ox, oy) = (100u32, 50u32);
    let whole = crop_canvas(&[CompRef::at(1, ox, oy)])
        .await
        .expect("the whole object paints");
    let full = crop_canvas(&[CompRef::at(1, ox, oy).cropped(0, 0, CROP_W as u16, CROP_H as u16)])
        .await
        .expect("the fully cropped object paints");
    assert_eq!(
        first_difference(&full, &whole),
        None,
        "a crop covering the object paints what no crop paints"
    );
}

/// The crop rectangle is four attacker-controlled u16s indexing a bitmap the
/// stream sized separately, so every degenerate one has to fold through the
/// clamp rather than read past the object. libbluray, reading a disc it trusts,
/// takes the rectangle at its word and walks off the bitmap, so this is g2g
/// being stricter than the reference rather than matching it.
#[tokio::test]
async fn a_degenerate_or_out_of_range_crop_is_clamped_and_reads_nothing_past_the_object() {
    let (ox, oy) = (100u32, 50u32);
    let whole = crop_canvas(&[CompRef::at(1, ox, oy)])
        .await
        .expect("the whole object paints");

    // An empty rectangle selects no pixels, and an origin at or past the object
    // edge selects none either, so neither display set paints.
    for crop in [
        (0, 0, 0, 4),
        (0, 0, 4, 0),
        (0, 0, 0, 0),
        (CROP_W as u16, 0, 4, 4),
        (0, CROP_H as u16, 4, 4),
        (u16::MAX, u16::MAX, u16::MAX, u16::MAX),
    ] {
        assert!(
            crop_canvas(&[CompRef::at(1, ox, oy).cropped(crop.0, crop.1, crop.2, crop.3)])
                .await
                .is_none(),
            "crop {crop:?} selects no pixels, so nothing paints"
        );
    }

    // A rectangle running off the object is clamped to what is there, keeping
    // its offset: the last 6 x 4 corner and not the first.
    let (cx, cy) = (10u32, 8u32);
    let clamped = crop_canvas(&[CompRef::at(1, ox + cx, oy + cy).cropped(
        cx as u16,
        cy as u16,
        u16::MAX,
        u16::MAX,
    )])
    .await
    .expect("the clamped crop still paints its corner");
    let mut want = vec![0u8; whole.len()];
    paste(
        &mut want,
        &whole,
        (ox + cx, oy + cy),
        (ox + cx, oy + cy),
        (CROP_W - cx, CROP_H - cy),
    );
    assert_eq!(
        first_difference(&clamped, &want),
        None,
        "an oversized crop paints the object's remaining corner, offset intact"
    );
}

/// A composition object reference is 8 bytes or 16, so the loop that walks it
/// has to read the flags right: 0x80 is cropped and 0x40 forced. Taking 0x40 for
/// the crop bit (which a widely copied write-up does) eats eight bytes that are
/// the next reference, and the composition falls apart from there.
#[tokio::test]
async fn a_forced_reference_behind_a_cropped_one_still_parses() {
    let (ox, oy) = (100u32, 50u32);
    let (sx, sy) = (600u32, 300u32);
    let whole = crop_canvas(&[CompRef::at(1, ox, oy)])
        .await
        .expect("the whole object paints");

    let (cx, cy, cw, ch) = (3u32, 2u32, 9u32, 7u32);
    let both = crop_canvas(&[
        CompRef::at(1, ox + cx, oy + cy).cropped(cx as u16, cy as u16, cw as u16, ch as u16),
        CompRef::at(2, sx, sy).forced(),
    ])
    .await
    .expect("both composition objects paint");

    let mut want = vec![0u8; whole.len()];
    paste(
        &mut want,
        &whole,
        (ox + cx, oy + cy),
        (ox + cx, oy + cy),
        (cw, ch),
    );
    paste(&mut want, &whole, (ox, oy), (sx, sy), (CROP_W, CROP_H));
    assert_eq!(
        first_difference(&both, &want),
        None,
        "the cropped reference paints its window and the forced one behind it the whole object"
    );
}
