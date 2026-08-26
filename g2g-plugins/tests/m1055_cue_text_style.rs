//! M1055 - the cue text styling the overlay renders beyond colour and size:
//! `<b>` / `<i>` / `<u>` and the `font-weight` / `font-style` /
//! `text-decoration` / `font-stretch` rules that back them. Each test renders a
//! frame and asserts on the pixels.
//!
//! The file runs on either CPU font path: with `truetype-overlay` alone the
//! horizontal cues go through the `ab_glyph` renderer, with `text-shaping` they
//! go through cosmic-text, and the vertical cue is always `ab_glyph`. Face
//! selection (bold, italic) is the shaper's, so those two tests need
//! `text-shaping` and an installed face to select.

#![cfg(feature = "truetype-overlay")]

mod cue_render_common;

use cue_render_common::{
    bounds, document, is_blue, is_green, is_red, latin_font, render, tallest_column_run,
    widest_row_run, FONT_PX, NO_BOX,
};

/// A cue whose middle run is red and its neighbours green and blue, so each
/// run's pixels are told apart by colour. `mark_style` adds declarations to the
/// middle run's rule, `mark_text` is its markup.
fn marked_document(mark_style: &str, mark_text: &str) -> String {
    document(
        &format!(
            "{NO_BOX}\n\
             ::cue(.before) {{ color: lime; }}\n\
             ::cue(.mark) {{ color: red; {mark_style} }}\n\
             ::cue(.after) {{ color: blue; }}"
        ),
        &format!("<c.before>ab</c> {mark_text} <c.after>ef</c>"),
    )
}

/// A `vertical:rl` cue drawn in red, whose `text` carries the markup under test.
fn vertical_document(text: &str) -> String {
    format!(
        "WEBVTT\n\nSTYLE\n{NO_BOX}\n::cue {{ color: red; }}\n\n\
         00:00:00.000 --> 00:00:10.000 vertical:rl\n{text}\n"
    )
}

/// The underline bar of a [`marked_document`] cue: a red run below the baseline
/// its neighbours share, covering the marked run's width and neither neighbour.
fn assert_bar_under_the_marked_run(pixels: &[u8]) {
    let (_, _, before_right, before_bottom) = bounds(pixels, is_green).expect("the run before");
    let (after_left, _, _, _) = bounds(pixels, is_blue).expect("the run after");
    let (row, left, right) = widest_row_run(pixels, is_red).expect("the marked run's ink");
    assert!(
        row > before_bottom,
        "the bar sits below the baseline the neighbouring run shares: row {row} vs {before_bottom}"
    );
    assert!(
        left > before_right && right < after_left,
        "the bar stays inside its own run: {left}..{right} between {before_right} and {after_left}"
    );
    assert!(
        right - left > FONT_PX / 2,
        "the bar spans the run rather than one glyph stem: {left}..{right}"
    );
}

/// `<u>` underlines its own run, with no stylesheet involved.
#[tokio::test]
async fn inline_underline_tag_draws_a_bar_under_its_run() {
    let Some(font) = latin_font() else {
        std::eprintln!("no system font; skipping");
        return;
    };
    let pixels = render(Some(&font), &marked_document("", "<u><c.mark>cd</c></u>")).await;
    assert_bar_under_the_marked_run(&pixels);
}

/// The same bar through the CSS route: `::cue(.mark) { text-decoration:
/// underline }` on the span the class names.
#[tokio::test]
async fn cue_css_text_decoration_underlines_its_span() {
    let Some(font) = latin_font() else {
        std::eprintln!("no system font; skipping");
        return;
    };
    let pixels = render(
        Some(&font),
        &marked_document("text-decoration: underline;", "<c.mark>cd</c>"),
    )
    .await;
    assert_bar_under_the_marked_run(&pixels);
}

/// A `vertical:rl` cue underlines down the column's right edge, clear of the
/// glyphs the column centres.
#[tokio::test]
async fn vertical_cue_underlines_along_the_column_right_edge() {
    let Some(font) = latin_font() else {
        std::eprintln!("no system font; skipping");
        return;
    };
    let plain = render(Some(&font), &vertical_document("ab")).await;
    let underlined = render(Some(&font), &vertical_document("<u>ab</u>")).await;
    let (_, _, glyph_right, _) = bounds(&plain, is_red).expect("column glyphs");
    let (column, top, bottom) = tallest_column_run(&underlined, is_red).expect("the bar");
    assert!(
        column > glyph_right,
        "the bar sits right of every glyph pixel: column {column} vs {glyph_right}"
    );
    assert!(
        bottom - top > FONT_PX,
        "the bar runs down both characters of the column: {top}..{bottom}"
    );
}

/// `<b>` bolds its own run: with a `wght`-variable face the shaper reaches that
/// axis, so the same word paints more ink than it does unstyled.
#[cfg(feature = "text-shaping")]
#[tokio::test]
async fn bold_span_paints_more_ink_than_the_unstyled_run() {
    use cue_render_common::{ink, variable_font};

    let Some(font) = variable_font() else {
        std::eprintln!("no variable system font; skipping");
        return;
    };
    let style = format!("{NO_BOX}\n::cue {{ color: white; }}");
    let plain = render(Some(&font), &document(&style, "Hamburg")).await;
    let bold = render(Some(&font), &document(&style, "<b>Hamburg</b>")).await;
    assert!(ink(&plain) > 0, "the unstyled run renders");
    assert!(
        ink(&bold) > ink(&plain),
        "the bold run is heavier: {} px vs {} px",
        ink(&bold),
        ink(&plain)
    );
}

/// `font-style: italic` selects an italic face where the default sans-serif
/// family has one. cosmic-text has no synthetic oblique, so with no such face
/// installed the run renders upright and only its ink is asserted.
#[cfg(feature = "text-shaping")]
#[tokio::test]
async fn italic_span_selects_an_italic_face_when_one_is_installed() {
    use cue_render_common::{fc_match_file, ink};

    let style = format!("{NO_BOX}\n::cue {{ color: white; }}");
    let upright = render(None, &document(&style, "Hamburg")).await;
    let italic = render(None, &document(&style, "<i>Hamburg</i>")).await;
    assert!(ink(&italic) > 0, "the italic run renders");

    let sans = fc_match_file("sans-serif");
    let slanted = fc_match_file("sans-serif:style=Italic");
    match (sans, slanted) {
        (Some(sans), Some(slanted)) if sans != slanted => assert_ne!(
            upright, italic,
            "the italic run is drawn from {slanted} rather than {sans}"
        ),
        (Some(sans), _) => std::eprintln!(
            "skip the difference check: fontconfig resolves the italic sans-serif to {sans} too"
        ),
        _ => std::eprintln!("skip the difference check: no fontconfig on this host"),
    }
}

// -- The Vello GPU backend draws the same styling. -----------------------------

#[cfg(feature = "vello-text-overlay")]
mod gpu {
    use super::*;
    use cue_render_common::{black_frame, caps, FrameSink};
    use g2g_core::{AsyncElement, MemoryDomain, PipelinePacket};
    use g2g_plugins::gpu::{read_rgba_texture, texture_of, GpuContext};
    use g2g_plugins::subparse::parse_webvtt;
    use g2g_plugins::vellooverlay::VelloTextOverlay;

    // Parallel per-test device creation intermittently segfaults in the NVIDIA
    // driver (the recorded wgpu gotcha), so the GPU tests take one lock for
    // their whole body.
    static GPU_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    async fn gpu_context() -> Option<GpuContext> {
        match GpuContext::headless().await {
            Ok(ctx) => Some(ctx),
            Err(_) => {
                std::eprintln!("no wgpu adapter; skipping the GPU underline test");
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
            .with_font_size(cue_render_common::FONT_PX);
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

    /// The GPU backend draws the underline bar too, in the pixels the CPU
    /// overlay puts it in.
    #[tokio::test]
    async fn gpu_draws_the_underline_bar() {
        let _gpu = GPU_LOCK.lock().await;
        let Some(ctx) = gpu_context().await else {
            return;
        };
        let Some(font) = latin_font() else {
            std::eprintln!("no system font; skipping");
            return;
        };
        let vtt = marked_document("", "<u><c.mark>cd</c></u>");
        let on_gpu = gpu_render(&ctx, &font, &vtt).await;
        assert_bar_under_the_marked_run(&on_gpu);

        let on_cpu = render(Some(&font), &vtt).await;
        let (gpu_row, gpu_left, gpu_right) = widest_row_run(&on_gpu, is_red).expect("the GPU bar");
        let (cpu_row, cpu_left, cpu_right) = widest_row_run(&on_cpu, is_red).expect("the CPU bar");
        for (edge, (g, c)) in [
            ("row", (gpu_row, cpu_row)),
            ("left", (gpu_left, cpu_left)),
            ("right", (gpu_right, cpu_right)),
        ] {
            assert!(
                g.abs_diff(c) <= 3,
                "the bar's {edge} matches the CPU overlay: {g} vs {c}"
            );
        }
    }
}
