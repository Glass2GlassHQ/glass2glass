//! M1057 - the rest of the cue CSS the overlay renders: `-webkit-text-stroke`,
//! the `::cue(b)` family of type selectors, `<ruby>` annotations, and
//! `font-family`. Each test renders a frame and asserts on the pixels.
//!
//! The file runs on either CPU font path: with `truetype-overlay` alone the
//! cues go through the `ab_glyph` renderer, with `text-shaping` through
//! cosmic-text. Family selection is the shaper's, so that test needs
//! `text-shaping` and a monospace face to select.

#![cfg(feature = "truetype-overlay")]

mod cue_render_common;

use cue_render_common::{bounds, document, is_ink, is_red, latin_font, render, NO_BOX};

/// Stroke width the outline tests ask for, and the extra AA rows a glyph edge
/// can add on each side beyond it, so the dilation is measured with slack but
/// not with a free pass.
const STROKE_PX: u32 = 2;
const EDGE_SLACK: u32 = 3;

/// A white pixel: the glyph fill of the stroke tests, which the red outline
/// never covers.
fn is_white(px: &[u8]) -> bool {
    px[0] > 200 && px[1] > 200 && px[2] > 200
}

/// A cue drawn in white with a red `-webkit-text-stroke` under it.
fn stroked_document(declaration: &str) -> String {
    document(
        &format!("{NO_BOX}\n::cue {{ color: white; {declaration} }}"),
        "Ho",
    )
}

/// The `-webkit-text-stroke` shorthand dilates the glyph mask: the outline
/// colour rings the fill on every side, by the width the rule asked for.
#[tokio::test]
async fn text_stroke_rings_the_fill_on_every_side() {
    let Some(font) = latin_font() else {
        std::eprintln!("no system font; skipping");
        return;
    };
    let pixels = render(
        Some(&font),
        &stroked_document(&format!("-webkit-text-stroke: {STROKE_PX}px red;")),
    )
    .await;
    assert_outline_rings_the_fill(&pixels);
}

/// The longhands reach the same outline as the shorthand, unprefixed spelling
/// included.
#[tokio::test]
async fn text_stroke_longhands_reach_the_same_outline() {
    let Some(font) = latin_font() else {
        std::eprintln!("no system font; skipping");
        return;
    };
    let shorthand = render(
        Some(&font),
        &stroked_document(&format!("-webkit-text-stroke: {STROKE_PX}px red;")),
    )
    .await;
    let longhands = render(
        Some(&font),
        &stroked_document(&format!(
            "text-stroke-width: {STROKE_PX}px; text-stroke-color: red;"
        )),
    )
    .await;
    assert_outline_rings_the_fill(&longhands);
    assert!(
        shorthand == longhands,
        "the longhands paint the shorthand's pixels"
    );
}

/// The outline of a [`stroked_document`] cue: red ink exists, its bounding box
/// grows the white fill's by the stroke width on every side, and it reaches the
/// fill rather than leaving a gap.
fn assert_outline_rings_the_fill(pixels: &[u8]) {
    let (white_left, white_top, white_right, white_bottom) =
        bounds(pixels, is_white).expect("the white fill");
    let (red_left, red_top, red_right, red_bottom) = bounds(pixels, is_red).expect("the outline");
    for (edge, outer, inner) in [
        ("left", white_left - red_left, white_left),
        ("top", white_top - red_top, white_top),
        ("right", red_right - white_right, white_right),
        ("bottom", red_bottom - white_bottom, white_bottom),
    ] {
        assert!(
            outer >= STROKE_PX,
            "the outline clears the fill's {edge} edge ({inner}) by the stroke width: {outer} px"
        );
        assert!(
            outer <= STROKE_PX + EDGE_SLACK,
            "the outline stops within a glyph edge of the stroke width on the {edge}: {outer} px"
        );
    }
    // On every row the fill reaches, the outline flanks it within the stroke
    // width, so it hugs the glyph edge instead of sitting only at the extremes.
    let at = |x: u32, y: u32| &pixels[((y * cue_render_common::W + x) * 4) as usize..][..4];
    let reach = STROKE_PX + EDGE_SLACK;
    for y in white_top..=white_bottom {
        let Some(first) = (white_left..=white_right).find(|&x| is_white(at(x, y))) else {
            continue;
        };
        let last = (white_left..=white_right)
            .rev()
            .find(|&x| is_white(at(x, y)))
            .expect("the row has a white pixel");
        assert!(
            (first.saturating_sub(reach)..first).any(|x| is_red(at(x, y))),
            "the outline flanks the fill's left edge on row {y}: no red in {}..{first}",
            first.saturating_sub(reach)
        );
        assert!(
            (last + 1..=last + reach).any(|x| is_red(at(x, y))),
            "the outline flanks the fill's right edge on row {y}: no red in {}..{}",
            last + 1,
            last + reach
        );
    }
}

/// A `::cue(b)` type selector styles the `<b>` run and nothing else.
#[tokio::test]
async fn type_selector_colours_only_the_tagged_run() {
    let Some(font) = latin_font() else {
        std::eprintln!("no system font; skipping");
        return;
    };
    let pixels = render(
        Some(&font),
        &document(
            &format!("{NO_BOX}\n::cue {{ color: white; }}\n::cue(b) {{ color: red; }}"),
            "plain <b>bold</b>",
        ),
    )
    .await;
    let (white_left, _, white_right, _) = bounds(&pixels, is_white).expect("the untagged run");
    let (red_left, _, red_right, _) = bounds(&pixels, is_red).expect("the tagged run");
    assert!(
        red_left > white_right,
        "the type selector recolours only the `<b>` run: red {red_left}..{red_right} vs white {white_left}..{white_right}"
    );
}

/// A class selector outscores a type selector on the same run, as CSS orders
/// them, whatever the sheet order.
#[tokio::test]
async fn class_selector_beats_a_type_selector_on_the_same_run() {
    let Some(font) = latin_font() else {
        std::eprintln!("no system font; skipping");
        return;
    };
    // The type rule is last in the sheet, so only specificity can keep the
    // class rule's red.
    let pixels = render(
        Some(&font),
        &document(
            &format!("{NO_BOX}\n::cue(.loud) {{ color: red; }}\n::cue(c) {{ color: white; }}"),
            "<c.loud>ab</c>",
        ),
    )
    .await;
    assert!(
        bounds(&pixels, is_red).is_some(),
        "the class rule wins the colour"
    );
    assert!(
        bounds(&pixels, is_white).is_none(),
        "the type rule loses it"
    );
}

/// A compound `::cue(b.loud)` is not supported, so the rule is dropped rather
/// than applied to every `<b>` or every `.loud`.
#[tokio::test]
async fn compound_type_and_class_selector_is_ignored() {
    let Some(font) = latin_font() else {
        std::eprintln!("no system font; skipping");
        return;
    };
    let pixels = render(
        Some(&font),
        &document(
            &format!("{NO_BOX}\n::cue {{ color: white; }}\n::cue(b.loud) {{ color: red; }}"),
            "<b><c.loud>ab</c></b>",
        ),
    )
    .await;
    assert!(bounds(&pixels, is_white).is_some(), "the cue still renders");
    assert!(
        bounds(&pixels, is_red).is_none(),
        "the compound rule is ignored"
    );
}

/// `<ruby>` takes the `rt` out of the line and draws it above the base, smaller,
/// leaving the base on the baseline it would have had on its own.
#[tokio::test]
async fn ruby_annotation_sits_above_the_base_without_moving_it() {
    let Some(font) = latin_font() else {
        std::eprintln!("no system font; skipping");
        return;
    };
    let style = format!("{NO_BOX}\n::cue {{ color: white; }}");
    let plain = render(Some(&font), &document(&style, "base")).await;
    let ruby = render(
        Some(&font),
        &document(&style, "<ruby>base<rt>rt</rt></ruby>"),
    )
    .await;
    let (plain_left, plain_top, plain_right, plain_bottom) =
        bounds(&plain, is_ink).expect("the base ink");
    let (_, ruby_top, _, ruby_bottom) = bounds(&ruby, is_ink).expect("the annotated ink");
    assert_eq!(
        (plain_bottom, plain_left, plain_right),
        bounds(&ruby, is_ink).map(|(l, _, r, b)| (b, l, r)).unwrap(),
        "the base keeps the baseline and the extent it has with no annotation"
    );
    assert!(
        ruby_top < plain_top,
        "the annotation puts ink above the base: {ruby_top} vs {plain_top}"
    );
    let annotation_rows = plain_top - ruby_top;
    let base_rows = plain_bottom - plain_top;
    assert!(
        annotation_rows < base_rows,
        "the annotation is the smaller run: {annotation_rows} rows vs {base_rows}"
    );
    assert!(
        ruby_bottom == plain_bottom,
        "the annotation stays out of the line flow: {ruby_bottom} vs {plain_bottom}"
    );
}

/// A `::cue(rt)` rule styles the annotation alone, leaving the base run's colour.
#[tokio::test]
async fn cue_rt_rule_recolours_the_annotation_alone() {
    let Some(font) = latin_font() else {
        std::eprintln!("no system font; skipping");
        return;
    };
    let pixels = render(
        Some(&font),
        &document(
            &format!("{NO_BOX}\n::cue {{ color: white; }}\n::cue(rt) {{ color: red; }}"),
            "<ruby>base<rt>rt</rt></ruby>",
        ),
    )
    .await;
    let (_, white_top, _, _) = bounds(&pixels, is_white).expect("the base run");
    let (_, _, _, red_bottom) = bounds(&pixels, is_red).expect("the annotation");
    assert!(
        red_bottom <= white_top,
        "only the annotation is red, and it is the ink above the base: {red_bottom} vs {white_top}"
    );
}

/// `font-family: monospace` reaches the shaper's family query, so the same
/// narrow letters take more width than they do in the proportional default.
#[cfg(feature = "text-shaping")]
#[tokio::test]
async fn font_family_monospace_widens_a_narrow_run() {
    use cue_render_common::fc_match_file;

    let (Some(sans), Some(mono)) = (fc_match_file("sans-serif"), fc_match_file("monospace")) else {
        std::eprintln!("skipping font-family: no fontconfig on this host");
        return;
    };
    if sans == mono {
        std::eprintln!("skipping font-family: fontconfig resolves monospace to {sans} too");
        return;
    }
    let width_of = |pixels: &[u8]| {
        let (left, _, right, _) = bounds(pixels, is_ink).expect("the run's ink");
        right - left
    };
    let style = format!("{NO_BOX}\n::cue {{ color: white; }}");
    let proportional = render(None, &document(&style, "iii")).await;
    let monospaced = render(
        None,
        &document(
            &format!("{style}\n::cue {{ font-family: monospace; }}"),
            "iii",
        ),
    )
    .await;
    assert!(
        width_of(&monospaced) > width_of(&proportional),
        "the monospace `i` is the wider one: {} px from {mono} vs {} px from {sans}",
        width_of(&monospaced),
        width_of(&proportional)
    );
}
