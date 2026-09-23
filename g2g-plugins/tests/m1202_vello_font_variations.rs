#![cfg(feature = "vello-text-overlay")]

mod cue_render_common;

use cue_render_common::{
    cpu_frame, cpu_overlay, gpu_context, gpu_frame, gpu_overlay, ink, ink_overlap, read_first,
    GPU_LOCK, MIN_CPU_OVERLAP, NO_BOX,
};
use g2g_core::{AsyncElement, PropValue};

const FONT_VARIATIONS_PROPERTY: &str = "font-variations";
const BOLD_SPEC: &str = "wght=700";

// A default-instance draw at the bold spec paints the same ink as the default.
const MIN_BOLD_INK_GAIN: f32 = 1.2;

// No static bold face covers Cherokee, so the shaper's 700 weight lands on this variable face.
const CHEROKEE_VARIABLE_FONT: &str = "/usr/share/fonts/google-noto-vf/NotoSansCherokee[wght].ttf";
const CHEROKEE_TEXT: &str = "ᏣᎳᎩ";

fn cue_document(settings: &str) -> String {
    format!(
        "WEBVTT\n\nSTYLE\n{NO_BOX}\n\n00:00:00.000 --> 00:00:10.000{settings}\n{CHEROKEE_TEXT}\n"
    )
}

async fn bold_spec_draws_the_bold_instance(settings: &str) {
    let _gpu = GPU_LOCK.lock().await;
    let Some(ctx) = gpu_context().await else {
        return;
    };
    let Some(font) = read_first(&[CHEROKEE_VARIABLE_FONT]) else {
        std::eprintln!("no {CHEROKEE_VARIABLE_FONT}, skipping");
        return;
    };
    let vtt = cue_document(settings);
    let default = gpu_frame(&ctx, &mut gpu_overlay(&ctx, &font, &vtt)).await;

    let mut overlay = gpu_overlay(&ctx, &font, &vtt);
    overlay
        .set_property(FONT_VARIATIONS_PROPERTY, PropValue::Str(BOLD_SPEC.into()))
        .expect("spec accepted");
    assert_eq!(
        overlay.get_property(FONT_VARIATIONS_PROPERTY),
        Some(PropValue::Str(BOLD_SPEC.into()))
    );
    let bold = gpu_frame(&ctx, &mut overlay).await;

    let (bold_ink, default_ink) = (ink(&bold), ink(&default));
    assert!(default_ink > 0, "{settings:?}: the default instance draws");
    assert!(
        bold_ink as f32 > default_ink as f32 * MIN_BOLD_INK_GAIN,
        "{settings:?}: {BOLD_SPEC} paints heavier glyphs than the default ({bold_ink} vs {default_ink})"
    );

    let mut reference = cpu_overlay(Some(&font), &vtt);
    reference
        .set_property(FONT_VARIATIONS_PROPERTY, PropValue::Str(BOLD_SPEC.into()))
        .expect("spec accepted");
    let overlap = ink_overlap(&bold, &cpu_frame(&mut reference).await);
    assert!(
        overlap >= MIN_CPU_OVERLAP,
        "{settings:?}: the bold instance matches the CPU reference, IoU {overlap}"
    );
}

#[tokio::test]
async fn horizontal_cue_draws_the_bold_instance() {
    bold_spec_draws_the_bold_instance("").await;
}

#[tokio::test]
async fn vertical_cue_draws_the_bold_instance() {
    bold_spec_draws_the_bold_instance(" vertical:rl").await;
}
