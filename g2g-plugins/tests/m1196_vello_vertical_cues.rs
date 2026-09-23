//! M1196 - vertical cues on the Vello GPU text overlay. A `vertical:rl` / `lr`
//! cue lays out as top-to-bottom columns, the first line rightmost for `rl` and
//! leftmost for `lr`, in the same pixels the CPU overlay puts it in. Each test
//! renders through the real `VelloTextOverlay` and reads the texture back.
//!
//! Skips without a wgpu adapter or without a system font to load. Runs for real
//! on the RTX 3060 dev host.

#![cfg(feature = "vello-text-overlay")]

mod cue_render_common;

use cue_render_common::{
    bounds, gpu_context, gpu_frame, gpu_overlay, gpu_render, is_blue, is_ink, is_red, latin_font,
    read_first, render, GPU_LOCK, H, NO_BOX, W,
};
use g2g_core::{AsyncElement, PropValue};

/// Least intersection-over-union between the GPU and CPU ink masks. Not 1,
/// because Vello and `ab_glyph` antialias the same outline differently, so edge
/// pixels differ by design.
const MIN_CPU_OVERLAP: f32 = 0.75;

/// A two-line cue with the first line red and the second blue, so each column's
/// pixels are told apart by colour. `settings` follows the timing, empty for a
/// horizontal cue.
fn two_line_document(settings: &str) -> String {
    format!(
        "WEBVTT\n\nSTYLE\n{NO_BOX}\n::cue(.first) {{ color: red; }}\n::cue(.second) {{ color: blue; }}\n\n\
         00:00:00.000 --> 00:00:10.000{settings}\n<c.first>ABC</c>\n<c.second>DEF</c>\n"
    )
}

/// A second system face, told apart from [`latin_font`] by its heavier
/// strokes, as its path and bytes. `None` to skip.
fn replacement_font() -> Option<(&'static str, Vec<u8>)> {
    [
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf",
    ]
    .into_iter()
    .find_map(|path| Some((path, read_first(&[path])?)))
}

/// Intersection-over-union of the painted pixels of two frames.
fn ink_overlap(first: &[u8], second: &[u8]) -> f32 {
    let ink = |pixels: &[u8]| -> Vec<bool> {
        pixels
            .as_chunks::<4>()
            .0
            .iter()
            .map(|px| is_ink(px))
            .collect()
    };
    let (first, second) = (ink(first), ink(second));
    let both = first
        .iter()
        .zip(&second)
        .filter(|(a, b)| **a && **b)
        .count();
    let either = first
        .iter()
        .zip(&second)
        .filter(|(a, b)| **a || **b)
        .count();
    assert!(either > 0, "one of the frames has ink");
    both as f32 / either as f32
}

/// `(left, top, right, bottom)` of one line's ink, inclusive.
type Bounds = (u32, u32, u32, u32);

fn width((left, _, right, _): Bounds) -> u32 {
    right - left + 1
}

fn height((_, top, _, bottom): Bounds) -> u32 {
    bottom - top + 1
}

/// Whether a line's ink stands as a column: taller than it is wide.
fn is_column(line: Bounds) -> bool {
    height(line) > width(line)
}

/// The red first line and the blue second line of a [`two_line_document`], as
/// the GPU drew them. `None` to skip on a host without a GPU or a font.
async fn gpu_lines(settings: &str) -> Option<(Bounds, Bounds)> {
    let ctx = gpu_context().await?;
    let Some(font) = latin_font() else {
        std::eprintln!("no system font, skipping");
        return None;
    };
    let pixels = gpu_render(&ctx, &font, &two_line_document(settings)).await;
    let first = bounds(&pixels, is_red).expect("the first line's ink");
    let second = bounds(&pixels, is_blue).expect("the second line's ink");
    Some((first, second))
}

/// `vertical:rl`: each line is a column, the first at the frame's right edge
/// and the second to its left.
#[tokio::test]
async fn rl_columns_advance_right_to_left() {
    let _gpu = GPU_LOCK.lock().await;
    let Some((first, second)) = gpu_lines(" vertical:rl").await else {
        return;
    };
    assert!(is_column(first), "the first line is a column: {first:?}");
    assert!(is_column(second), "the second line is a column: {second:?}");
    assert!(
        first.0 > second.2,
        "the first column is right of the second: {first:?} vs {second:?}"
    );
    assert!(
        first.0 > W * 3 / 4,
        "the block hugs the right edge: first column at {first:?}"
    );
}

/// `vertical:lr`: each line is a column, the first at the frame's left edge
/// and the second to its right.
#[tokio::test]
async fn lr_columns_advance_left_to_right() {
    let _gpu = GPU_LOCK.lock().await;
    let Some((first, second)) = gpu_lines(" vertical:lr").await else {
        return;
    };
    assert!(is_column(first), "the first line is a column: {first:?}");
    assert!(is_column(second), "the second line is a column: {second:?}");
    assert!(
        second.0 > first.2,
        "the second column is right of the first: {first:?} vs {second:?}"
    );
    assert!(
        first.2 < W / 4,
        "the block hugs the left edge: first column at {first:?}"
    );
}

/// The control: with no `vertical:` setting the same cue is two rows stacked up
/// from the bottom edge, so the column checks above tell the modes apart.
#[tokio::test]
async fn horizontal_cue_stays_in_rows() {
    let _gpu = GPU_LOCK.lock().await;
    let Some((first, second)) = gpu_lines("").await else {
        return;
    };
    assert!(!is_column(first), "the first line is a row: {first:?}");
    assert!(!is_column(second), "the second line is a row: {second:?}");
    assert!(
        first.3 < second.1,
        "the first row is above the second: {first:?} vs {second:?}"
    );
    assert!(
        second.3 > H * 3 / 4,
        "the rows stack up from the bottom edge: {second:?}"
    );
}

/// The GPU columns are the CPU columns: both backends read one layout, so the
/// ink masks overlap at [`MIN_CPU_OVERLAP`] or better.
#[tokio::test]
async fn gpu_columns_match_the_cpu_reference() {
    let _gpu = GPU_LOCK.lock().await;
    let Some(ctx) = gpu_context().await else {
        return;
    };
    let Some(font) = latin_font() else {
        std::eprintln!("no system font, skipping");
        return;
    };
    for settings in [" vertical:rl", " vertical:lr"] {
        let vtt = two_line_document(settings);
        let on_gpu = gpu_render(&ctx, &font, &vtt).await;
        let on_cpu = render(Some(&font), &vtt).await;
        let overlap = ink_overlap(&on_gpu, &on_cpu);
        assert!(
            overlap >= MIN_CPU_OVERLAP,
            "{settings}: GPU columns match the CPU reference, IoU {overlap}"
        );
    }
}

/// A new `font=` mid-stream draws the next vertical cue in the new face, not
/// the face the GPU backend cached for the old chain.
#[tokio::test]
async fn font_property_swap_draws_the_new_face() {
    let _gpu = GPU_LOCK.lock().await;
    let Some(ctx) = gpu_context().await else {
        return;
    };
    let (Some(font), Some((replacement_path, replacement))) = (latin_font(), replacement_font())
    else {
        std::eprintln!("no system fonts, skipping");
        return;
    };
    let vtt = two_line_document(" vertical:rl");
    let mut overlay = gpu_overlay(&ctx, &font, &vtt);
    gpu_frame(&ctx, &mut overlay).await;
    overlay
        .set_property("font", PropValue::Str(replacement_path.into()))
        .expect("font accepted");
    let on_gpu = gpu_frame(&ctx, &mut overlay).await;
    let overlap = ink_overlap(&on_gpu, &render(Some(&replacement), &vtt).await);
    assert!(
        overlap >= MIN_CPU_OVERLAP,
        "the swapped face matches its CPU reference, IoU {overlap}"
    );
}
