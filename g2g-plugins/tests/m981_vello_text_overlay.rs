//! M981 - the Vello GPU text overlay, validated on a real GPU. `VelloTextOverlay`
//! draws the same cue as the CPU `TextOverlay`, but as Vello glyph runs into a
//! `MemoryDomain::WgpuTexture`, so a keep-on-GPU pipeline overlays subtitles with
//! no CPU round trip. The tests read the rendered texture back and compare it
//! against the CPU overlay's output for the same cue, font and size.
//!
//! Skips without a wgpu adapter or without a system font to load. Runs for real
//! on the RTX 3060 dev host.

#![cfg(feature = "vello-text-overlay")]

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket,
    PushOutcome, Rate, RawVideoFormat,
};
use g2g_plugins::gpu::{read_rgba_texture, texture_of, GpuContext};
use g2g_plugins::subparse::{Cue, CueSettings};
use g2g_plugins::textoverlay::TextOverlay;
use g2g_plugins::vellooverlay::VelloTextOverlay;

const W: u32 = 320;
const H: u32 = 180;
const FONT_PX: u32 = 32;
/// Uniform frame colour, also the cue's backing-box colour, so every pixel that
/// differs from it is glyph coverage rather than the box.
const BACKDROP: [u8; 4] = [20, 20, 20, 255];
const CUE_START_NS: u64 = 0;
const CUE_END_NS: u64 = 2_000_000_000;

// Parallel per-test device creation intermittently segfaults in the NVIDIA
// driver (the recorded wgpu gotcha), so the GPU tests take one lock for their
// whole body.
static GPU_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// First available Latin system font, or `None` to skip (a host with no fonts).
/// These are the Fedora paths the dev host has.
fn latin_font() -> Option<Vec<u8>> {
    for path in [
        "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
    ] {
        if let Ok(bytes) = std::fs::read(path) {
            return Some(bytes);
        }
    }
    None
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

/// One cue on screen for the first two seconds, its backing box painted in the
/// frame colour so only glyphs change pixels.
fn cue(text: &str) -> Cue {
    Cue {
        start_ns: CUE_START_NS,
        end_ns: CUE_END_NS,
        text: text.into(),
        settings: CueSettings {
            background: Some(BACKDROP),
            ..CueSettings::default()
        },
    }
}

fn backdrop_frame(pts_ns: u64) -> Frame {
    let mut bytes = Vec::with_capacity((W * H * 4) as usize);
    for _ in 0..W * H {
        bytes.extend_from_slice(&BACKDROP);
    }
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            ..FrameTiming::default()
        },
        0,
    )
}

#[derive(Default)]
struct FrameSink {
    last: Option<Frame>,
}
impl OutputSink for FrameSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            if let PipelinePacket::DataFrame(frame) = packet {
                self.last = Some(frame);
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// The CPU overlay's rendering of `text` at `pts_ns`, as RGBA8 bytes.
async fn cpu_render(font: &[u8], text: &str, pts_ns: u64) -> Vec<u8> {
    let mut overlay = TextOverlay::new()
        .with_font_bytes(font, 0)
        .expect("font parses")
        .with_cues(Vec::from([cue(text)]))
        .with_font_size(FONT_PX);
    overlay.configure_pipeline(&caps()).unwrap();
    let mut sink = FrameSink::default();
    overlay
        .process(PipelinePacket::DataFrame(backdrop_frame(pts_ns)), &mut sink)
        .await
        .unwrap();
    let frame = sink.last.expect("frame forwarded");
    frame
        .domain
        .as_system_slice()
        .expect("system memory out")
        .to_vec()
}

/// The GPU overlay's rendering of `text` at `pts_ns`, read back from the
/// `WgpuTexture` it emits.
async fn gpu_render(ctx: &GpuContext, font: &[u8], text: &str, pts_ns: u64) -> Vec<u8> {
    let mut overlay = VelloTextOverlay::new()
        .with_context(ctx.clone())
        .with_font_bytes(font, 0)
        .expect("font parses")
        .with_cues(Vec::from([cue(text)]))
        .with_font_size(FONT_PX);
    overlay.configure_pipeline(&caps()).unwrap();
    let mut sink = FrameSink::default();
    overlay
        .process(PipelinePacket::DataFrame(backdrop_frame(pts_ns)), &mut sink)
        .await
        .unwrap();
    let frame = sink.last.expect("frame forwarded");
    let MemoryDomain::WgpuTexture(owned) = &frame.domain else {
        panic!("output is a GPU texture domain");
    };
    assert_eq!((owned.width, owned.height), (W, H));
    assert_eq!(overlay.drawn_count(), 1);
    read_rgba_texture(ctx, texture_of(owned).expect("texture keep-alive"))
}

/// Which pixels the overlay changed: anything that is no longer the flat
/// backdrop is glyph coverage (the backing box is painted in the backdrop
/// colour).
fn coverage(pixels: &[u8]) -> Vec<bool> {
    pixels
        .as_chunks::<4>()
        .0
        .iter()
        .map(|px| px != &BACKDROP)
        .collect::<Vec<bool>>()
}

/// Bounding box `(left, top, right, bottom)` of the covered pixels, exclusive on
/// the far edges. `None` when nothing was drawn.
fn bounds(mask: &[bool]) -> Option<(u32, u32, u32, u32)> {
    let mut b: Option<(u32, u32, u32, u32)> = None;
    for (i, covered) in mask.iter().enumerate() {
        if !covered {
            continue;
        }
        let (x, y) = (i as u32 % W, i as u32 / W);
        b = Some(match b {
            None => (x, y, x + 1, y + 1),
            Some((l, t, r, bo)) => (l.min(x), t.min(y), r.max(x + 1), bo.max(y + 1)),
        });
    }
    b
}

fn intersection_over_union(a: &[bool], b: &[bool]) -> f32 {
    let inter = a.iter().zip(b).filter(|(x, y)| **x && **y).count();
    let union = a.iter().zip(b).filter(|(x, y)| **x || **y).count();
    if union == 0 {
        return 0.0;
    }
    inter as f32 / union as f32
}

async fn gpu_context() -> Option<GpuContext> {
    match GpuContext::headless().await {
        Ok(ctx) => Some(ctx),
        Err(_) => {
            std::eprintln!("no wgpu adapter; skipping Vello text overlay test");
            None
        }
    }
}

/// The cue renders on the GPU where the overlay says it should: glyph coverage
/// in the bottom-centre band, nothing above it, and the same region the CPU
/// overlay paints (within 3 px per edge, the two rasterizers' hinting spread).
#[tokio::test]
async fn gpu_cue_covers_the_expected_region() {
    let _gpu = GPU_LOCK.lock().await;
    let Some(ctx) = gpu_context().await else {
        return;
    };
    let Some(font) = latin_font() else {
        std::eprintln!("no system font; skipping Vello text overlay test");
        return;
    };

    let gpu = gpu_render(&ctx, &font, "Hello g2g", 1_000_000_000).await;
    let cpu = cpu_render(&font, "Hello g2g", 1_000_000_000).await;
    let gpu_mask = coverage(&gpu);
    let cpu_mask = coverage(&cpu);

    let covered = gpu_mask.iter().filter(|c| **c).count();
    assert!(covered > 200, "GPU drew glyph coverage: {covered} px");

    let (gl, gt, gr, gb) = bounds(&gpu_mask).expect("GPU coverage");
    let (cl, ct, cr, cb) = bounds(&cpu_mask).expect("CPU coverage");

    // Bottom-centre placement, the overlay default for a cue with no `line` /
    // `position`.
    assert!(gt > H / 2, "text sits in the bottom half: top row {gt}");
    assert!(gb <= H, "text stays on the canvas: bottom row {gb}");
    let center = (gl + gr) / 2;
    assert!(
        center.abs_diff(W / 2) <= 4,
        "text is horizontally centred: centre {center}"
    );
    // Nothing outside that band was touched.
    let above = gpu_mask[..(gt as usize * W as usize)].iter().any(|c| *c);
    assert!(!above, "the frame above the cue is untouched");

    for (gpu_edge, cpu_edge, name) in [
        (gl, cl, "left"),
        (gt, ct, "top"),
        (gr, cr, "right"),
        (gb, cb, "bottom"),
    ] {
        assert!(
            gpu_edge.abs_diff(cpu_edge) <= 3,
            "{name} edge matches the CPU overlay: gpu {gpu_edge}, cpu {cpu_edge}"
        );
    }
}

/// The GPU glyphs are the CPU glyphs: the two coverage masks overlap at an
/// intersection-over-union of at least 0.75. Not byte equality, because Vello
/// area-samples unhinted outlines while the CPU path blits swash's hinted
/// rasters, so edge pixels differ by design.
#[tokio::test]
async fn gpu_cue_matches_the_cpu_reference() {
    let _gpu = GPU_LOCK.lock().await;
    let Some(ctx) = gpu_context().await else {
        return;
    };
    let Some(font) = latin_font() else {
        std::eprintln!("no system font; skipping Vello text overlay test");
        return;
    };

    let gpu = gpu_render(&ctx, &font, "Hello g2g", 1_000_000_000).await;
    let cpu = cpu_render(&font, "Hello g2g", 1_000_000_000).await;
    let iou = intersection_over_union(&coverage(&gpu), &coverage(&cpu));
    assert!(iou >= 0.75, "GPU text matches the CPU reference: IoU {iou}");
}

/// A frame with no cue showing comes out of the GPU element byte for byte: the
/// element still renders (it always emits a `WgpuTexture`), so this also pins
/// that the frame image itself survives the Vello pass unchanged.
#[tokio::test]
async fn frame_without_a_cue_is_byte_identical() {
    let _gpu = GPU_LOCK.lock().await;
    let Some(ctx) = gpu_context().await else {
        return;
    };
    let Some(font) = latin_font() else {
        std::eprintln!("no system font; skipping Vello text overlay test");
        return;
    };

    // Past the cue's end time, so nothing is drawn.
    let gpu = gpu_render(&ctx, &font, "Hello g2g", CUE_END_NS + 1).await;
    let expected: Vec<u8> = BACKDROP.repeat((W * H) as usize);
    assert_eq!(gpu, expected, "no-cue frame passes through unchanged");
}

/// The fallback chain is the CPU one: with a Latin-only primary font, the CJK in
/// a cue still renders, because the shaper resolves those codepoints to a
/// discovered system face and the GPU backend draws that face's outlines.
#[tokio::test]
async fn cjk_falls_back_to_a_discovered_face_on_the_gpu() {
    let _gpu = GPU_LOCK.lock().await;
    let Some(ctx) = gpu_context().await else {
        return;
    };
    let Some(font) = latin_font() else {
        std::eprintln!("no system font; skipping Vello text overlay test");
        return;
    };

    let text = "g2g 日本語";
    let gpu = gpu_render(&ctx, &font, text, 1_000_000_000).await;
    let cpu = cpu_render(&font, text, 1_000_000_000).await;
    let (gpu_mask, cpu_mask) = (coverage(&gpu), coverage(&cpu));
    if cpu_mask.iter().filter(|c| **c).count() < 200 {
        std::eprintln!("no CJK-capable system font; skipping fallback assertion");
        return;
    }
    let covered = gpu_mask.iter().filter(|c| **c).count();
    assert!(covered > 200, "GPU drew the fallback glyphs: {covered} px");
    let iou = intersection_over_union(&gpu_mask, &cpu_mask);
    assert!(
        iou >= 0.75,
        "GPU fallback glyphs match the CPU reference: IoU {iou}"
    );
}
