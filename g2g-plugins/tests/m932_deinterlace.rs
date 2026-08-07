//! M932: the CPU `deinterlace` element, and the MPEG-2 sequence extension's
//! `progressive_sequence` flag (which decided the `ps_playbin` insertion until
//! M935 made the insertion universal and the decision the element's `auto` mode).
//!
//! The yadif kernel is checked against ffmpeg's own `yadif` rather than against
//! itself: both filter the same raw interlaced I420 frames, so a mismatch is a
//! porting bug and not a decoder difference. Raw video in means neither side
//! reads a field-order flag, so both assume top-field-first.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::path::PathBuf;
use std::process::Command;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::SystemSlice;
use g2g_core::{AsyncElement, Caps, Dim, G2gError, MemoryDomain, OutputSink, PushOutcome, Rate};
use g2g_core::{PropValue, RawVideoFormat};

use g2g_plugins::deinterlace::{Deinterlace, DeinterlaceMethod};
use g2g_plugins::psdemux::parse_sequence_header;

const W: usize = 320;
const H: usize = 240;
const I420_BYTES: usize = W * H * 3 / 2;

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m932-{}-{name}", std::process::id()))
}

fn ffmpeg(args: &[&str]) {
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .args(args)
        .output()
        .expect("run ffmpeg");
    assert!(
        out.status.success(),
        "ffmpeg {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[derive(Default)]
struct Collect {
    frames: Vec<Vec<u8>>,
    timings: Vec<FrameTiming>,
}

impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        if let PipelinePacket::DataFrame(f) = packet {
            self.frames
                .push(f.domain.as_system_slice().unwrap().to_vec());
            self.timings.push(f.timing);
        }
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

fn caps(format: RawVideoFormat, w: usize, h: usize) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Fixed(w as u32),
        height: Dim::Fixed(h as u32),
        framerate: Rate::Fixed(25 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

/// Push every frame of `input` through a `deinterlace` in `method`, then EOS, and
/// return what came out.
async fn run(
    input: &[Vec<u8>],
    format: RawVideoFormat,
    w: usize,
    h: usize,
    method: DeinterlaceMethod,
) -> Collect {
    let mut el = Deinterlace::new().with_method(method);
    el.configure_pipeline(&caps(format, w, h)).unwrap();
    let mut out = Collect::default();
    for (i, bytes) in input.iter().enumerate() {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.clone().into_boxed_slice())),
            FrameTiming {
                pts_ns: i as u64 * 40_000_000,
                ..Default::default()
            },
            i as u64,
        );
        el.process(PipelinePacket::DataFrame(frame), &mut out)
            .await
            .unwrap();
    }
    el.process(PipelinePacket::Eos, &mut out).await.unwrap();
    out
}

fn split_frames(bytes: &[u8], size: usize) -> Vec<Vec<u8>> {
    bytes.chunks_exact(size).map(|c| c.to_vec()).collect()
}

/// Largest absolute luma difference between two I420 frames, split into the
/// interior and the 3-column border yadif's edge path handles differently.
fn luma_diff(a: &[u8], b: &[u8], w: usize, h: usize) -> (u32, u32) {
    let (mut interior, mut border) = (0u32, 0u32);
    for y in 0..h {
        for x in 0..w {
            let d = (a[y * w + x] as i32 - b[y * w + x] as i32).unsigned_abs();
            if x >= 3 && x + 3 < w {
                interior = interior.max(d);
            } else {
                border = border.max(d);
            }
        }
    }
    (interior, border)
}

// ---- differential: g2g yadif vs ffmpeg yadif ----

#[tokio::test]
async fn yadif_matches_ffmpeg_bit_exactly() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not present: skipping");
        return;
    }
    let il = temp_path("interlaced.yuv");
    let reference = temp_path("ffmpeg-yadif.yuv");
    // testsrc2 at 50 fps interleaved pairwise into 25 interlaced frames: every
    // frame has real motion between its two fields, which is what yadif's
    // temporal branch keys on.
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc2=size={W}x{H}:rate=50:duration=1"),
        "-vf",
        "tinterlace=mode=interleave_top,format=yuv420p",
        "-f",
        "rawvideo",
        il.to_str().unwrap(),
    ]);
    ffmpeg(&[
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-s",
        &format!("{W}x{H}"),
        "-r",
        "25",
        "-i",
        il.to_str().unwrap(),
        "-vf",
        "yadif=0:-1:0",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "rawvideo",
        reference.to_str().unwrap(),
    ]);

    let input = split_frames(&std::fs::read(&il).unwrap(), I420_BYTES);
    let want = split_frames(&std::fs::read(&reference).unwrap(), I420_BYTES);
    assert!(input.len() >= 10, "fixture is {} frames", input.len());
    assert_eq!(
        want.len(),
        input.len(),
        "ffmpeg yadif mode 0 is single rate: N in, N out"
    );

    let got = run(&input, RawVideoFormat::I420, W, H, DeinterlaceMethod::Yadif).await;
    assert_eq!(
        got.frames.len(),
        input.len(),
        "g2g yadif is single rate too"
    );

    for (i, (g, r)) in got.frames.iter().zip(&want).enumerate() {
        let (interior, border) = luma_diff(g, r, W, H);
        let differing = g.iter().zip(r.iter()).filter(|(a, b)| a != b).count();
        assert_eq!(
            (interior, border, differing),
            (0, 0, 0),
            "frame {i} differs from ffmpeg yadif: (max |d| interior, max |d| border, bytes differing)"
        );
    }

    let _ = std::fs::remove_file(&il);
    let _ = std::fs::remove_file(&reference);
}

// ---- combing ----

/// Mean absolute difference between vertically adjacent luma rows: high on a
/// combed frame, low once the comb is gone.
fn comb_energy(frame: &[u8], w: usize, h: usize) -> f64 {
    let mut total = 0u64;
    for y in 0..h - 1 {
        for x in 0..w {
            total +=
                (frame[y * w + x] as i32 - frame[(y + 1) * w + x] as i32).unsigned_abs() as u64;
        }
    }
    total as f64 / ((h - 1) * w) as f64
}

/// A frame whose two fields are horizontally shifted copies of one image: the
/// classic comb. It has to come out materially flatter vertically.
fn combed_frame(shift: usize) -> Vec<u8> {
    let mut f = vec![128u8; I420_BYTES];
    for y in 0..H {
        for x in 0..W {
            // A vertical-edge pattern, so a horizontal field shift produces a
            // large row-to-row difference and nothing else does.
            let sx = if y % 2 == 0 { x } else { x + shift };
            f[y * W + x] = if (sx / 8) % 2 == 0 { 20 } else { 220 };
        }
    }
    f
}

#[tokio::test]
async fn yadif_flattens_a_comb() {
    let frames = vec![combed_frame(4); 5];
    let before = comb_energy(&frames[0], W, H);
    let out = run(
        &frames,
        RawVideoFormat::I420,
        W,
        H,
        DeinterlaceMethod::Yadif,
    )
    .await;
    let after = comb_energy(&out.frames[2], W, H);
    assert!(
        after < before / 4.0,
        "comb energy {before:.1} -> {after:.1}: yadif did not remove the comb"
    );
}

#[tokio::test]
async fn linear_flattens_a_comb() {
    let frames = vec![combed_frame(4)];
    let before = comb_energy(&frames[0], W, H);
    let out = run(
        &frames,
        RawVideoFormat::I420,
        W,
        H,
        DeinterlaceMethod::Linear,
    )
    .await;
    let after = comb_energy(&out.frames[0], W, H);
    assert!(
        after < before / 4.0,
        "comb energy {before:.1} -> {after:.1}"
    );
}

// ---- progressive passthrough ----

/// A still progressive image: neither method may invent detail on it. yadif's
/// temporal clamp collapses to the source value when nothing moves, so it is
/// exactly identity; `linear` interpolates and is only approximately so.
#[tokio::test]
async fn progressive_content_survives_both_methods() {
    let mut still = vec![0u8; I420_BYTES];
    for y in 0..H {
        for x in 0..W {
            // Smooth in both axes, so `linear`'s vertical interpolation is close.
            still[y * W + x] = ((x + y) / 3) as u8;
        }
    }
    for c in still.iter_mut().skip(W * H) {
        *c = 128;
    }
    let frames = vec![still.clone(); 4];

    let yadif = run(
        &frames,
        RawVideoFormat::I420,
        W,
        H,
        DeinterlaceMethod::Yadif,
    )
    .await;
    assert_eq!(yadif.frames.len(), 4);
    for (i, f) in yadif.frames.iter().enumerate() {
        assert_eq!(*f, still, "yadif altered still progressive frame {i}");
    }

    let linear = run(
        &frames,
        RawVideoFormat::I420,
        W,
        H,
        DeinterlaceMethod::Linear,
    )
    .await;
    let (interior, border) = luma_diff(&linear.frames[0], &still, W, H);
    assert!(
        interior <= 1 && border <= 1,
        "linear drifted from a smooth still image: interior {interior}, border {border}"
    );
}

// ---- element surface ----

#[tokio::test]
async fn nv12_chroma_pairs_stay_separate() {
    // A U plane of 0 and a V plane of 255 interleaved: any deinterlace that
    // treated the pair as one component would average them together.
    let (w, h) = (16usize, 8usize);
    let mut f = vec![0u8; w * h * 3 / 2];
    for y in 0..h {
        for x in 0..w {
            f[y * w + x] = if y % 2 == 0 { 0 } else { 255 };
        }
    }
    for i in 0..(w / 2 * h / 2) {
        f[w * h + i * 2] = 0;
        f[w * h + i * 2 + 1] = 255;
    }
    let out = run(
        &vec![f.clone(); 3],
        RawVideoFormat::Nv12,
        w,
        h,
        DeinterlaceMethod::Yadif,
    )
    .await;
    let got = &out.frames[1];
    for i in 0..(w / 2 * h / 2) {
        assert_eq!(got[w * h + i * 2], 0, "U sample {i} picked up V");
        assert_eq!(got[w * h + i * 2 + 1], 255, "V sample {i} picked up U");
    }
}

#[tokio::test]
async fn output_keeps_its_own_frames_timing() {
    let frames = vec![combed_frame(2); 4];
    let out = run(
        &frames,
        RawVideoFormat::I420,
        W,
        H,
        DeinterlaceMethod::Yadif,
    )
    .await;
    let pts: Vec<_> = out.timings.iter().map(|t| t.pts_ns).collect();
    assert_eq!(pts, [0, 40_000_000, 80_000_000, 120_000_000]);
}

#[test]
fn odd_geometry_on_a_subsampled_format_is_refused() {
    let mut el = Deinterlace::new();
    assert!(el
        .configure_pipeline(&caps(RawVideoFormat::I420, 15, 8))
        .is_err());
    assert!(el
        .configure_pipeline(&caps(RawVideoFormat::I420, 16, 8))
        .is_ok());
}

#[test]
fn method_is_a_launch_property() {
    use g2g_core::runtime::parse_launch;
    use g2g_plugins::registry::default_registry;
    for method in ["yadif", "linear", "blend"] {
        parse_launch(
            &default_registry(),
            &format!("videotestsrc num-buffers=1 ! deinterlace method={method} ! fakesink"),
        )
        .unwrap_or_else(|e| panic!("deinterlace method={method}: {e}"));
    }
    let mut el = Deinterlace::new();
    el.set_property("method", PropValue::Str("linear".into()))
        .unwrap();
    assert_eq!(
        el.get_property("method"),
        Some(PropValue::Str("linear".into()))
    );
}

// ---- interlace detection: the MPEG-2 sequence extension ----

/// A minimal MPEG-2 sequence header (no quantiser matrices) followed by a
/// sequence extension declaring `progressive_sequence`.
fn seq_bytes(progressive: bool, extension: bool) -> Vec<u8> {
    let mut v = Vec::from([0x00, 0x00, 0x01, 0xB3]);
    // 720x576, aspect 3, frame_rate_code 3 (25 fps).
    v.extend_from_slice(&[0x2D, 0x02, 0x40, 0x33]);
    // bit_rate_value / marker / vbv / flags, both matrix loads off.
    v.extend_from_slice(&[0x00, 0x00, 0x20, 0x00]);
    if extension {
        v.extend_from_slice(&[0x00, 0x00, 0x01, 0xB5]);
        // id 0001 + profile_and_level 0x48, then progressive_sequence at bit 3.
        v.push(0x14);
        v.push(0x80 | if progressive { 0x08 } else { 0x00 });
        v.extend_from_slice(&[0x00, 0x01]);
    }
    // A picture start code so the unit looks like a real access unit.
    v.extend_from_slice(&[0x00, 0x00, 0x01, 0x00, 0x00, 0x0F, 0xFF, 0xF8]);
    v
}

#[test]
fn sequence_extension_carries_progressive_sequence() {
    let interlaced = parse_sequence_header(&seq_bytes(false, true)).expect("parses");
    assert_eq!((interlaced.width, interlaced.height), (720, 576));
    assert!(!interlaced.progressive);

    let progressive = parse_sequence_header(&seq_bytes(true, true)).expect("parses");
    assert!(progressive.progressive);
}

#[test]
fn a_missing_or_malformed_extension_stays_progressive() {
    // MPEG-1: no extension at all.
    let mpeg1 = parse_sequence_header(&seq_bytes(false, false)).expect("parses");
    assert!(mpeg1.progressive, "MPEG-1 has no interlace signalling");

    // Truncated right after the extension start code.
    let mut cut = seq_bytes(false, true);
    cut.truncate(16 + 1);
    let truncated = parse_sequence_header(&cut).expect("the header itself still parses");
    assert!(truncated.progressive);

    // A B5 extension that is not a sequence extension (identifier 0b0010).
    let mut other = seq_bytes(false, true);
    other[16] = 0x24;
    let ignored = parse_sequence_header(&other).expect("parses");
    assert!(ignored.progressive);

    // Every truncation of an interlaced header parses or declines, never panics.
    let full = seq_bytes(false, true);
    for n in 0..full.len() {
        let _ = parse_sequence_header(&full[..n]);
    }
}

// ---- playbin topology ----

/// Every playbin video branch carries a `deinterlace mode=auto` since M935, so
/// both the interlaced and the progressive DVD-style fixtures build with one in
/// the graph (the auto mode no-ops on the progressive stream at runtime, so the
/// progressive disc no longer pays for weaving, only for the passthrough hop).
/// Needs a real MPEG-2 decoder in the pool, so it runs under the `ffmpeg`
/// feature.
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
#[test]
fn ps_playbin_always_inserts_the_auto_deinterlace() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not present: skipping");
        return;
    }
    let interlaced = temp_path("interlaced.mpg");
    let progressive = temp_path("progressive.mpg");
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=352x288:rate=50:duration=1",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=1",
        "-vf",
        "tinterlace=mode=interleave_top",
        "-c:v",
        "mpeg2video",
        "-flags",
        "+ilme+ildct",
        "-top",
        "1",
        "-c:a",
        "mp2",
        "-f",
        "vob",
        interlaced.to_str().unwrap(),
    ]);
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=352x288:rate=25:duration=1",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=1",
        "-c:v",
        "mpeg2video",
        "-c:a",
        "mp2",
        "-f",
        "vob",
        progressive.to_str().unwrap(),
    ]);

    let dot = graph_dot(&interlaced);
    assert!(
        dot.contains("label=\"Deinterlace\""),
        "interlaced program stream got no deinterlacer: {dot}"
    );
    let dot = graph_dot(&progressive);
    assert!(
        dot.contains("label=\"Deinterlace\""),
        "the auto deinterlacer belongs on every video branch (M935): {dot}"
    );

    let _ = std::fs::remove_file(&interlaced);
    let _ = std::fs::remove_file(&progressive);
}

/// The graph `playbin uri=` builds for a file, rendered to DOT: each node is
/// labelled with its element's log category (the short type name).
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
fn graph_dot(path: &std::path::Path) -> String {
    use g2g_core::runtime::parse_launch;
    use g2g_core::DotAnnotations;
    use g2g_plugins::registry::default_registry;
    let line = format!("playbin uri=file://{}", path.to_str().unwrap());
    let graph = parse_launch(&default_registry(), &line)
        .unwrap_or_else(|e| panic!("playbin builds `{line}`: {e}"));
    graph.to_dot(
        "pipeline",
        |n| graph.element(n).map(|e| e.log_category().to_string()),
        &DotAnnotations::default(),
    )
}
