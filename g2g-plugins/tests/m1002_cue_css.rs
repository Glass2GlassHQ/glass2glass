//! M1002 - the WebVTT cue CSS the overlay renders beyond colour: per-span
//! `font-size`, `text-shadow`, and a span-scoped `background-color`. Each test
//! parses a `STYLE` block, renders a frame, and asserts on the pixels.
//!
//! The file runs on either CPU font path: with `truetype-overlay` alone the
//! horizontal cues go through the `ab_glyph` renderer, with `text-shaping` they
//! go through cosmic-text, and the vertical cue is always `ab_glyph`. The Vello
//! tests need a wgpu adapter and skip without one.

#![cfg(feature = "truetype-overlay")]

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket,
    PushOutcome, Rate, RawVideoFormat,
};
use g2g_plugins::subparse::parse_webvtt;
use g2g_plugins::textoverlay::TextOverlay;

const W: u32 = 480;
const H: u32 = 160;
const FONT_PX: u32 = 32;
const SHADOW_OFFSET: i32 = 8;

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
    }
}

/// A one-cue WebVTT document: `style` declarations, then `text` as the cue body.
fn document(style: &str, text: &str) -> String {
    format!("WEBVTT\n\nSTYLE\n{style}\n\n00:00:00.000 --> 00:00:10.000\n{text}\n")
}

fn black_frame() -> Frame {
    let mut bytes = Vec::with_capacity((W * H * 4) as usize);
    for _ in 0..W * H {
        bytes.extend_from_slice(&[0, 0, 0, 255]);
    }
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming::default(),
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
        packet: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::DataFrame(frame)) = packet.take() {
            self.last = Some(frame);
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

/// Render the document's cue over a black frame, as RGBA8 bytes.
async fn render(font: &[u8], vtt: &str) -> Vec<u8> {
    let mut overlay = TextOverlay::new()
        .with_font_bytes(font, 0)
        .expect("font parses")
        .with_cues(parse_webvtt(vtt))
        .with_font_size(FONT_PX);
    overlay.configure_pipeline(&caps()).expect("caps accepted");
    let mut sink = FrameSink::default();
    overlay
        .process(PipelinePacket::DataFrame(black_frame()), &mut sink)
        .await
        .expect("frame rendered");
    sink.last
        .expect("frame forwarded")
        .domain
        .as_system_slice()
        .expect("system memory out")
        .to_vec()
}

/// Whether a pixel is dominated by one channel, which is how a coloured glyph or
/// fill reads once it is alpha-blended onto the black frame.
fn is_red(px: &[u8]) -> bool {
    dominates(px[0], px[1], px[2])
}
fn is_green(px: &[u8]) -> bool {
    dominates(px[1], px[0], px[2])
}
fn is_blue(px: &[u8]) -> bool {
    dominates(px[2], px[0], px[1])
}
fn dominates(channel: u8, other: u8, third: u8) -> bool {
    let (channel, other, third) = (u32::from(channel), u32::from(other), u32::from(third));
    channel > 60 && other * 3 < channel && third * 3 < channel
}
fn is_white(px: &[u8]) -> bool {
    px[0] > 60 && px[1] > 60 && px[2] > 60
}

/// Bounding box `(left, top, right, bottom)` of the pixels `pick` accepts,
/// inclusive on every edge. `None` when it accepted none.
fn bounds(pixels: &[u8], pick: fn(&[u8]) -> bool) -> Option<(u32, u32, u32, u32)> {
    let mut found: Option<(u32, u32, u32, u32)> = None;
    for (i, px) in pixels.chunks_exact(4).enumerate() {
        if !pick(px) {
            continue;
        }
        let (x, y) = (i as u32 % W, i as u32 / W);
        found = Some(match found {
            None => (x, y, x, y),
            Some((l, t, r, b)) => (l.min(x), t.min(y), r.max(x), b.max(y)),
        });
    }
    found
}

fn count(pixels: &[u8], pick: fn(&[u8]) -> bool) -> usize {
    pixels.chunks_exact(4).filter(|px| pick(px)).count()
}

fn height_of(pixels: &[u8], pick: fn(&[u8]) -> bool) -> u32 {
    let (_, top, _, bottom) = bounds(pixels, pick).expect("pixels of this colour");
    bottom - top + 1
}

/// A `::cue(.class)` `font-size` sizes only its own run: the same letter in the
/// sized span covers a taller band than the one outside it. The two runs are
/// told apart by colour, which the overlay already resolves per span.
#[tokio::test]
async fn span_font_size_makes_only_its_own_run_taller() {
    let Some(font) = latin_font() else {
        std::eprintln!("no system font; skipping");
        return;
    };
    let vtt = document(
        "::cue { color: white; background-color: transparent; }\n\
         ::cue(.big) { color: red; font-size: 200%; }",
        "n<c.big>n</c>",
    );
    let pixels = render(&font, &vtt).await;

    let plain = height_of(&pixels, is_white);
    let big = height_of(&pixels, is_red);
    assert!(
        big as f32 >= plain as f32 * 1.4,
        "the 200% span is drawn taller: {big} px vs {plain} px"
    );
    // The sized run is the trailing one, so it also sits to the right.
    let (_, _, plain_right, _) = bounds(&pixels, is_white).expect("plain glyph");
    let (big_left, _, _, _) = bounds(&pixels, is_red).expect("sized glyph");
    assert!(
        big_left > plain_right,
        "the sized run stays in its own span: {big_left} vs {plain_right}"
    );
}

/// A whole-cue `font-size` sizes the cue itself, which the per-span percent then
/// resolves against.
#[tokio::test]
async fn cue_font_size_sizes_the_whole_cue() {
    let Some(font) = latin_font() else {
        std::eprintln!("no system font; skipping");
        return;
    };
    let small = render(
        &font,
        &document(
            "::cue { color: white; background-color: transparent; }",
            "n",
        ),
    )
    .await;
    let large = render(
        &font,
        &document(
            "::cue { color: white; background-color: transparent; font-size: 64px; }",
            "n",
        ),
    )
    .await;
    let (small_h, large_h) = (height_of(&small, is_white), height_of(&large, is_white));
    assert!(
        large_h > small_h,
        "the cue rule resized the text: {large_h} px vs {small_h} px"
    );
}

/// `text-shadow` draws an offset copy under the glyphs: the shadow colour shows
/// on the far side of every glyph, exactly the offset past it, and the glyph
/// itself still paints over the shadow.
#[tokio::test]
async fn text_shadow_paints_offset_under_the_glyphs() {
    let Some(font) = latin_font() else {
        std::eprintln!("no system font; skipping");
        return;
    };
    let vtt = document(
        &format!(
            "::cue {{ color: white; background-color: transparent; \
              text-shadow: {SHADOW_OFFSET}px {SHADOW_OFFSET}px blue; }}"
        ),
        "no",
    );
    let pixels = render(&font, &vtt).await;

    assert!(count(&pixels, is_blue) > 40, "the shadow was drawn");
    assert!(
        count(&pixels, is_white) > 40,
        "the glyphs are drawn over it"
    );
    let (_, white_top, white_right, white_bottom) = bounds(&pixels, is_white).expect("glyphs");
    let (_, blue_top, blue_right, blue_bottom) = bounds(&pixels, is_blue).expect("shadow");
    for (shadow_edge, glyph_edge, name) in [
        (blue_right, white_right, "right"),
        (blue_bottom, white_bottom, "bottom"),
    ] {
        let shift = shadow_edge as i32 - glyph_edge as i32;
        assert!(
            (shift - SHADOW_OFFSET).abs() <= 1,
            "the shadow's {name} edge is the offset past the glyphs: {shift} px"
        );
    }
    assert!(
        blue_top >= white_top,
        "a downward shadow never rises above the glyphs: {blue_top} vs {white_top}"
    );
}

/// A span-scoped `background-color` fills behind that span's glyphs alone,
/// leaving the rest of the cue on the frame it was drawn over.
#[tokio::test]
async fn span_background_fills_behind_its_span_only() {
    let Some(font) = latin_font() else {
        std::eprintln!("no system font; skipping");
        return;
    };
    let vtt = document(
        "::cue { color: white; background-color: transparent; }\n\
         ::cue(.mark) { background-color: rgb(0, 255, 0); }",
        "nn<c.mark>nn</c>",
    );
    let pixels = render(&font, &vtt).await;

    assert!(count(&pixels, is_green) > 200, "the span fill was drawn");
    let (green_left, green_top, green_right, green_bottom) =
        bounds(&pixels, is_green).expect("span fill");
    let (white_left, white_top, white_right, white_bottom) =
        bounds(&pixels, is_white).expect("ink");
    // The fill starts inside the cue (after the unstyled run) and runs to its
    // end.
    assert!(
        green_left > white_left,
        "the fill starts after the unstyled run: {green_left} vs {white_left}"
    );
    assert!(
        green_right + 2 >= white_right,
        "the fill covers its span to the end: {green_right} vs {white_right}"
    );
    // The fill is a line box, so it brackets the glyphs vertically.
    assert!(
        green_top <= white_top && green_bottom >= white_bottom,
        "the fill covers the line box: {green_top}..{green_bottom} vs {white_top}..{white_bottom}"
    );
    // Nothing green in the column strip the first glyph occupies.
    let strip = (white_left as usize)..(white_left as usize + 2);
    for y in 0..H as usize {
        for x in strip.clone() {
            let px = &pixels[(y * W as usize + x) * 4..][..4];
            assert!(
                !is_green(px),
                "the unstyled run keeps its background ({x},{y})"
            );
        }
    }
}

/// The same span fill on the `ab_glyph` column renderer, which a `vertical:rl`
/// cue always goes through: the fill runs down the column behind the span's own
/// characters, below the ones outside it.
#[tokio::test]
async fn span_background_fills_a_vertical_column() {
    let Some(font) = latin_font() else {
        std::eprintln!("no system font; skipping");
        return;
    };
    let vtt = "WEBVTT\n\nSTYLE\n\
         ::cue { color: white; background-color: transparent; }\n\
         ::cue(.mark) { background-color: rgb(0, 255, 0); }\n\n\
         00:00:00.000 --> 00:00:10.000 vertical:rl\nnn<c.mark>nn</c>\n";
    let pixels = render(&font, vtt).await;

    assert!(count(&pixels, is_green) > 200, "the column fill was drawn");
    let (_, green_top, _, _) = bounds(&pixels, is_green).expect("column fill");
    let (_, white_top, _, _) = bounds(&pixels, is_white).expect("ink");
    assert!(
        green_top > white_top,
        "the fill starts below the unstyled characters: {green_top} vs {white_top}"
    );
}

// -- The Vello GPU backend draws the same styling. -----------------------------

#[cfg(feature = "vello-text-overlay")]
mod gpu {
    use super::*;
    use g2g_plugins::gpu::{read_rgba_texture, texture_of, GpuContext};
    use g2g_plugins::vellooverlay::VelloTextOverlay;

    // Parallel per-test device creation intermittently segfaults in the NVIDIA
    // driver (the recorded wgpu gotcha), so the GPU tests take one lock for
    // their whole body.
    static GPU_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    async fn gpu_context() -> Option<GpuContext> {
        match GpuContext::headless().await {
            Ok(ctx) => Some(ctx),
            Err(_) => {
                std::eprintln!("no wgpu adapter; skipping the GPU cue CSS test");
                None
            }
        }
    }

    /// The GPU overlay's rendering of the document, read back from the texture.
    async fn gpu_render(ctx: &GpuContext, font: &[u8], vtt: &str) -> Vec<u8> {
        let mut overlay = VelloTextOverlay::new()
            .with_context(ctx.clone())
            .with_font_bytes(font, 0)
            .expect("font parses")
            .with_cues(parse_webvtt(vtt))
            .with_font_size(FONT_PX);
        overlay.configure_pipeline(&caps()).expect("caps accepted");
        let mut sink = FrameSink::default();
        overlay
            .process(PipelinePacket::DataFrame(black_frame()), &mut sink)
            .await
            .expect("frame rendered");
        let frame = sink.last.expect("frame forwarded");
        let MemoryDomain::WgpuTexture(owned) = &frame.domain else {
            panic!("output is a GPU texture domain");
        };
        read_rgba_texture(ctx, texture_of(owned).expect("texture keep-alive"))
    }

    /// All three properties at once on the GPU, checked against the CPU
    /// rendering of the same cue: the span fill lands in the same place, the
    /// shadow sits the offset past the glyphs, and the sized span is taller.
    #[tokio::test]
    async fn gpu_draws_span_size_shadow_and_background() {
        let _gpu = GPU_LOCK.lock().await;
        let Some(ctx) = gpu_context().await else {
            return;
        };
        let Some(font) = latin_font() else {
            std::eprintln!("no system font; skipping");
            return;
        };
        let vtt = document(
            &format!(
                "::cue {{ color: white; background-color: transparent; \
                  text-shadow: {SHADOW_OFFSET}px {SHADOW_OFFSET}px blue; }}\n\
                 ::cue(.mark) {{ color: red; font-size: 200%; \
                  background-color: rgb(0, 255, 0); }}"
            ),
            "nn<c.mark>nn</c>",
        );
        let gpu = gpu_render(&ctx, &font, &vtt).await;
        let cpu = render(&font, &vtt).await;

        assert!(count(&gpu, is_green) > 200, "the GPU drew the span fill");
        assert!(count(&gpu, is_blue) > 40, "the GPU drew the shadow");
        // The offsets themselves are pinned on the CPU path; here the point is
        // that the GPU backend puts them in the same pixels.
        for (what, pick) in [
            ("span fill", is_green as fn(&[u8]) -> bool),
            ("shadow", is_blue),
        ] {
            let on_gpu = bounds(&gpu, pick).expect("GPU pixels");
            let on_cpu = bounds(&cpu, pick).expect("CPU pixels");
            for (edge, (g, c)) in [
                ("left", (on_gpu.0, on_cpu.0)),
                ("top", (on_gpu.1, on_cpu.1)),
                ("right", (on_gpu.2, on_cpu.2)),
                ("bottom", (on_gpu.3, on_cpu.3)),
            ] {
                assert!(
                    g.abs_diff(c) <= 3,
                    "the {what}'s {edge} edge matches the CPU overlay: {g} vs {c}"
                );
            }
        }

        let (plain, big) = (height_of(&gpu, is_white), height_of(&gpu, is_red));
        assert!(
            big as f32 >= plain as f32 * 1.4,
            "the GPU drew the 200% span taller: {big} px vs {plain} px"
        );
    }
}
