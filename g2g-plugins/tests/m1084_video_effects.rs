//! M1084: the software video-effect transforms ported from GStreamer.
//!
//! Each test drives the real element through `configure_pipeline` + `process`
//! on a synthetic frame and asserts the pixels that come out, so a wrong
//! transfer function fails here rather than looking plausible on screen.

use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, Dim, Frame, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, PushOutcome, Rate, RawVideoFormat,
};

use g2g_plugins::aspectratiocrop::AspectRatioCrop;
use g2g_plugins::chromahold::ChromaHold;
use g2g_plugins::coloreffects::{ColorEffects, ColorEffectsPreset};
use g2g_plugins::gaussianblur::GaussianBlur;
use g2g_plugins::smooth::Smooth;
use g2g_plugins::videodiff::{VideoDiff, MARK_DARK_LUMA, MARK_LIGHT_LUMA};
use g2g_plugins::videomedian::{MedianSize, VideoMedian};
use g2g_plugins::zebrastripe::{ZebraStripe, STRIPE_LUMA};

#[derive(Default)]
struct CollectSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for CollectSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        self.packets
            .push(packet_slot.take().expect("poll_push without a packet"));
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn raw(format: RawVideoFormat, w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn frame(bytes: Vec<u8>) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns: 0,
            dts_ns: 0,
            duration_ns: 0,
            capture_ns: 0,
            arrival_ns: 0,
            keyframe: false,
        },
        sequence: 0,
        meta: Default::default(),
    }
}

/// Push one frame through a configured element and return everything it emits.
fn push<E: AsyncElement>(element: &mut E, bytes: Vec<u8>) -> Vec<PipelinePacket> {
    let mut sink = CollectSink::default();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    runtime
        .block_on(element.process(PipelinePacket::DataFrame(frame(bytes)), &mut sink))
        .expect("the element processed the frame");
    sink.packets
}

/// Configure an element for `input` and run one frame through it, returning the
/// output pixels.
fn run<E: AsyncElement>(element: &mut E, input: Caps, bytes: Vec<u8>) -> Vec<u8> {
    element.configure_pipeline(&input).expect("input accepted");
    pixels(push(element, bytes))
}

fn pixels(packets: Vec<PipelinePacket>) -> Vec<u8> {
    let PipelinePacket::DataFrame(out) = packets.last().expect("an output packet") else {
        panic!("expected a DataFrame downstream");
    };
    out.domain
        .require_system_slice("test")
        .expect("system memory out")
        .to_vec()
}

fn emitted_caps(packets: &[PipelinePacket]) -> Caps {
    packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::CapsChanged(c) => Some(c.clone()),
            _ => None,
        })
        .expect("the output caps are announced downstream")
}

/// A `w x h` RGBA frame of one colour.
fn rgba_flat(w: usize, h: usize, pixel: [u8; 4]) -> Vec<u8> {
    pixel.repeat(w * h)
}

/// A `w x h` I420 frame: `luma` everywhere, and the two chroma planes at
/// `chroma`, so a filter that strays off the luma plane is caught.
fn i420_flat(w: usize, h: usize, luma: u8, chroma: u8) -> Vec<u8> {
    let mut bytes = vec![luma; w * h];
    bytes.resize(w * h + 2 * (w / 2) * (h / 2), chroma);
    bytes
}

fn i420_luma(bytes: &[u8], w: usize, h: usize) -> &[u8] {
    &bytes[..w * h]
}

fn i420_chroma(bytes: &[u8], w: usize, h: usize) -> &[u8] {
    &bytes[w * h..]
}

// ---------------------------------------------------------------- coloreffects

const COLOREFFECTS_PRESETS: [ColorEffectsPreset; 5] = [
    ColorEffectsPreset::Heat,
    ColorEffectsPreset::Sepia,
    ColorEffectsPreset::XRay,
    ColorEffectsPreset::XPro,
    ColorEffectsPreset::YellowBlue,
];

/// The table entry for level `level`, as the three components it supplies.
fn lut_entry(table: &[u8; 768], level: u8) -> [u8; 3] {
    let at = level as usize * 3;
    [table[at], table[at + 1], table[at + 2]]
}

#[test]
fn coloreffects_maps_a_grey_pixel_to_its_table_entry() {
    // A grey pixel's luma is its own level, and each component indexes the same
    // row, so both mapping styles must land on one whole table entry.
    const GREY_LEVEL: u8 = 137;
    const ALPHA: u8 = 200;
    let (w, h) = (2usize, 2usize);
    for preset in COLOREFFECTS_PRESETS {
        let table = preset
            .table()
            .expect("a preset other than none has a table");
        let expected = lut_entry(table, GREY_LEVEL);
        let mut element = ColorEffects::new().with_preset(preset);
        let out = run(
            &mut element,
            raw(RawVideoFormat::Rgba8, w as u32, h as u32),
            rgba_flat(w, h, [GREY_LEVEL, GREY_LEVEL, GREY_LEVEL, ALPHA]),
        );
        assert_eq!(out.len(), w * h * 4);
        for pixel in out.as_chunks::<4>().0 {
            assert_eq!(&pixel[..3], &expected, "{preset:?}");
            assert_eq!(pixel[3], ALPHA, "{preset:?} kept alpha");
        }
    }
}

#[test]
fn coloreffects_maps_each_component_through_its_own_column() {
    // xpro and yellowblue are the per-component presets: R comes from the red
    // column of the R row, G from the green column of the G row, and so on.
    const PIXEL: [u8; 4] = [200, 100, 50, 255];
    for preset in [ColorEffectsPreset::XPro, ColorEffectsPreset::YellowBlue] {
        assert!(!preset.maps_luma(), "{preset:?} is a per-component preset");
        let table = preset.table().unwrap();
        let expected = [
            lut_entry(table, PIXEL[0])[0],
            lut_entry(table, PIXEL[1])[1],
            lut_entry(table, PIXEL[2])[2],
        ];
        let mut element = ColorEffects::new().with_preset(preset);
        let out = run(
            &mut element,
            raw(RawVideoFormat::Rgba8, 1, 1),
            PIXEL.to_vec(),
        );
        assert_eq!(&out[..3], &expected, "{preset:?}");
    }
}

#[test]
fn coloreffects_luma_presets_emit_a_whole_table_entry() {
    // A luma-mapped preset replaces the colour outright, so whatever comes out
    // has to be one of the table's own rows.
    const PIXEL: [u8; 4] = [200, 100, 50, 255];
    for preset in [
        ColorEffectsPreset::Heat,
        ColorEffectsPreset::Sepia,
        ColorEffectsPreset::XRay,
    ] {
        assert!(preset.maps_luma(), "{preset:?} is a luma-mapped preset");
        let table = preset.table().unwrap();
        let mut element = ColorEffects::new().with_preset(preset);
        let out = run(
            &mut element,
            raw(RawVideoFormat::Rgba8, 1, 1),
            PIXEL.to_vec(),
        );
        assert!(
            (0..=u8::MAX).any(|level| lut_entry(table, level) == out[..3]),
            "{preset:?} produced {:?}, not a table entry",
            &out[..3]
        );
    }
}

#[test]
fn coloreffects_none_passes_the_frame_through() {
    let source = rgba_flat(2, 2, [10, 90, 200, 255]);
    let mut element = ColorEffects::new();
    assert_eq!(element.preset(), ColorEffectsPreset::None);
    let out = run(
        &mut element,
        raw(RawVideoFormat::Rgba8, 2, 2),
        source.clone(),
    );
    assert_eq!(out, source);
}

// ------------------------------------------------------------------ chromahold

/// Full-saturation red, green and blue: hues 0, 120 and 240 degrees.
const RED: [u8; 4] = [255, 0, 0, 255];
const GREEN: [u8; 4] = [0, 255, 0, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];

/// A 3x1 RGBA frame of red, green, blue.
fn primaries() -> Vec<u8> {
    let mut bytes = RED.to_vec();
    bytes.extend_from_slice(&GREEN);
    bytes.extend_from_slice(&BLUE);
    bytes
}

fn is_grey(pixel: &[u8]) -> bool {
    pixel[0] == pixel[1] && pixel[1] == pixel[2]
}

#[test]
fn chromahold_keeps_the_target_hue_and_greys_the_rest() {
    // The default target is red, so red survives byte for byte and the other
    // two primaries, 120 degrees away, lose their colour but keep a brightness
    // between their darkest and lightest component.
    let mut element = ChromaHold::new();
    let out = run(&mut element, raw(RawVideoFormat::Rgba8, 3, 1), primaries());
    assert_eq!(&out[..4], &RED, "the target hue is untouched");
    for (offset, source) in [(4, GREEN), (8, BLUE)] {
        let pixel = &out[offset..offset + 4];
        assert!(is_grey(pixel), "{source:?} became {pixel:?}");
        let (darkest, lightest) = (
            source[..3].iter().min().unwrap(),
            source[..3].iter().max().unwrap(),
        );
        assert!(pixel[0] >= *darkest && pixel[0] <= *lightest);
        assert_eq!(pixel[3], source[3], "alpha survives");
    }
}

#[test]
fn chromahold_tolerance_decides_how_far_the_hue_may_stray() {
    // Green is 120 degrees off the default red target. A tolerance that reaches
    // it keeps it; anything short of that greys it.
    const GREEN_HUE_FROM_RED: u32 = 120;
    for (tolerance, kept) in [
        (GREEN_HUE_FROM_RED - 1, false),
        (GREEN_HUE_FROM_RED, true),
        (GREEN_HUE_FROM_RED + 1, true),
    ] {
        let mut element = ChromaHold::new().with_tolerance(tolerance);
        let out = run(&mut element, raw(RawVideoFormat::Rgba8, 3, 1), primaries());
        assert_eq!(
            !is_grey(&out[4..8]),
            kept,
            "green at tolerance {tolerance} should be kept: {kept}"
        );
        assert_eq!(&out[..4], &RED, "the target itself is always kept");
    }
}

#[test]
fn chromahold_a_grey_target_greys_everything() {
    // A grey target has no hue to hold onto, so nothing can match it.
    let mut element = ChromaHold::new().with_target(128, 128, 128);
    let out = run(&mut element, raw(RawVideoFormat::Rgba8, 3, 1), primaries());
    for pixel in out.as_chunks::<4>().0 {
        assert!(is_grey(pixel), "{pixel:?} kept its colour");
    }
}

// ----------------------------------------------------------------- zebrastripe

#[test]
fn zebrastripe_marks_only_the_samples_above_the_threshold() {
    const CHROMA: u8 = 90;
    let (w, h) = (8usize, 8usize);
    let element = ZebraStripe::new();
    let threshold = element.luma_threshold();

    // Every sample below the threshold: nothing is touched.
    let mut element = ZebraStripe::new();
    let dark = i420_flat(w, h, threshold - 1, CHROMA);
    let out = run(
        &mut element,
        raw(RawVideoFormat::I420, w as u32, h as u32),
        dark.clone(),
    );
    assert_eq!(out, dark, "nothing reaches the threshold");

    // Every sample at the threshold: the stripes appear, and only the stripes.
    let mut element = ZebraStripe::new();
    let bright = i420_flat(w, h, threshold, CHROMA);
    let out = run(
        &mut element,
        raw(RawVideoFormat::I420, w as u32, h as u32),
        bright.clone(),
    );
    let luma = i420_luma(&out, w, h);
    let struck = luma.iter().filter(|&&y| y == STRIPE_LUMA).count();
    assert!(struck > 0, "an overexposed frame is striped");
    assert!(struck < luma.len(), "the stripes leave gaps");
    for &sample in luma {
        assert!(sample == STRIPE_LUMA || sample == threshold);
    }
    assert_eq!(
        i420_chroma(&out, w, h),
        i420_chroma(&bright, w, h),
        "chroma is left alone"
    );
}

#[test]
fn zebrastripe_threshold_property_moves_the_luma_level() {
    // 100 % is video-range white, 0 % video-range black, and the stripe itself
    // is drawn at black.
    assert_eq!(
        ZebraStripe::new().with_threshold(0).luma_threshold(),
        STRIPE_LUMA
    );
    let full = ZebraStripe::new().with_threshold(100).luma_threshold();
    assert!(full > ZebraStripe::new().luma_threshold());
}

// ---------------------------------------------------------------- gaussianblur

/// A `w x h` RGBA frame that is black except for one white pixel at the centre.
fn rgba_impulse(w: usize, h: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; w * h * 4];
    let centre = ((h / 2) * w + w / 2) * 4;
    bytes[centre..centre + 4].copy_from_slice(&[u8::MAX; 4]);
    bytes
}

fn red_at(bytes: &[u8], w: usize, x: usize, y: usize) -> f64 {
    bytes[(y * w + x) * 4] as f64
}

#[test]
fn gaussianblur_impulse_response_follows_the_sigma() {
    // The impulse response of a separable gaussian is the outer product of its
    // taps, so the ratio between the centre and a neighbour `d` away along one
    // axis is exp(-d^2 / 2 sigma^2), whatever the other axis does.
    let (w, h) = (17usize, 17usize);
    let (cx, cy) = (w / 2, h / 2);
    let mut element = GaussianBlur::new();
    let sigma = match element.get_property("sigma") {
        Some(g2g_core::PropValue::Double(v)) => v,
        other => panic!("sigma reads back as a double, got {other:?}"),
    };
    let out = run(
        &mut element,
        raw(RawVideoFormat::Rgba8, w as u32, h as u32),
        rgba_impulse(w, h),
    );

    let centre = red_at(&out, w, cx, cy);
    assert!(centre > 0.0, "the impulse survives the blur");
    for distance in 1..=2 {
        let expected = (-(distance as f64).powi(2) / (2.0 * sigma * sigma)).exp();
        for (x, y) in [
            (cx + distance, cy),
            (cx - distance, cy),
            (cx, cy + distance),
            (cx, cy - distance),
        ] {
            let ratio = red_at(&out, w, x, y) / centre;
            // The output is 8-bit, so a tap of a few levels carries a rounding
            // error of a good fraction of a level.
            let tolerance = 0.5 / red_at(&out, w, x, y).max(1.0) + 0.02;
            assert!(
                (ratio - expected).abs() <= expected * tolerance.max(0.05),
                "tap {distance} at ({x},{y}) gave {ratio}, expected {expected}"
            );
        }
    }
}

#[test]
fn gaussianblur_leaves_a_flat_frame_flat() {
    // The edge windows renormalise, so even the border keeps its level.
    const PIXEL: [u8; 4] = [40, 130, 210, 255];
    let (w, h) = (8usize, 8usize);
    let source = rgba_flat(w, h, PIXEL);
    let mut element = GaussianBlur::new();
    let out = run(
        &mut element,
        raw(RawVideoFormat::Rgba8, w as u32, h as u32),
        source.clone(),
    );
    assert_eq!(out, source);
}

#[test]
fn gaussianblur_negative_sigma_sharpens() {
    // Sharpening concentrates the impulse instead of spreading it, so the
    // centre keeps more of its amplitude than the blur leaves it.
    let (w, h) = (17usize, 17usize);
    let (cx, cy) = (w / 2, h / 2);
    let sigma = 1.2;

    let mut blurred = GaussianBlur::new().with_sigma(sigma);
    let blurred = run(
        &mut blurred,
        raw(RawVideoFormat::Rgba8, w as u32, h as u32),
        rgba_impulse(w, h),
    );
    let mut sharpened = GaussianBlur::new().with_sigma(-sigma);
    let sharpened = run(
        &mut sharpened,
        raw(RawVideoFormat::Rgba8, w as u32, h as u32),
        rgba_impulse(w, h),
    );

    assert!(
        red_at(&sharpened, w, cx, cy) > red_at(&blurred, w, cx, cy),
        "the sharpened centre keeps more amplitude"
    );
    assert!(
        red_at(&sharpened, w, cx + 1, cy) < red_at(&blurred, w, cx + 1, cy),
        "the sharpened surround is pulled down"
    );
}

// ------------------------------------------------------------------ videodiff

#[test]
fn videodiff_marks_nothing_when_the_frame_repeats() {
    const LUMA: u8 = 120;
    const CHROMA: u8 = 90;
    let (w, h) = (8usize, 8usize);
    let source = i420_flat(w, h, LUMA, CHROMA);

    let mut element = VideoDiff::new();
    element
        .configure_pipeline(&raw(RawVideoFormat::I420, w as u32, h as u32))
        .unwrap();
    let first = pixels(push(&mut element, source.clone()));
    assert_eq!(first, source, "the first frame has nothing to compare to");
    let second = pixels(push(&mut element, source.clone()));
    assert_eq!(second, source, "an unchanged frame is unmarked");
}

#[test]
fn videodiff_marks_the_sample_that_changed() {
    const LUMA: u8 = 120;
    const CHROMA: u8 = 90;
    // Well past the default threshold, so the sample counts as moved.
    const MOVED_LUMA: u8 = 220;
    let (w, h) = (8usize, 8usize);
    let (mx, my) = (3usize, 5usize);

    let mut element = VideoDiff::new();
    element
        .configure_pipeline(&raw(RawVideoFormat::I420, w as u32, h as u32))
        .unwrap();
    push(&mut element, i420_flat(w, h, LUMA, CHROMA));

    let mut moved = i420_flat(w, h, LUMA, CHROMA);
    moved[my * w + mx] = MOVED_LUMA;
    let out = pixels(push(&mut element, moved.clone()));

    let marked = out[my * w + mx];
    assert!(
        marked == MARK_DARK_LUMA || marked == MARK_LIGHT_LUMA,
        "the moved sample is marked, got {marked}"
    );
    for (index, &sample) in i420_luma(&out, w, h).iter().enumerate() {
        if index != my * w + mx {
            assert_eq!(sample, LUMA, "still sample {index} was marked");
        }
    }
    assert_eq!(
        i420_chroma(&out, w, h),
        i420_chroma(&moved, w, h),
        "chroma is left alone"
    );
}

#[test]
fn videodiff_ignores_a_change_inside_the_threshold() {
    const LUMA: u8 = 120;
    const CHROMA: u8 = 90;
    let (w, h) = (8usize, 8usize);
    let threshold = match VideoDiff::new().get_property("threshold") {
        Some(g2g_core::PropValue::Int(v)) => v as u8,
        other => panic!("threshold reads back as an int, got {other:?}"),
    };

    let mut element = VideoDiff::new();
    element
        .configure_pipeline(&raw(RawVideoFormat::I420, w as u32, h as u32))
        .unwrap();
    push(&mut element, i420_flat(w, h, LUMA, CHROMA));
    let nudged = i420_flat(w, h, LUMA + threshold, CHROMA);
    let out = pixels(push(&mut element, nudged.clone()));
    assert_eq!(
        out, nudged,
        "a change of exactly the threshold is not motion"
    );
}

// ----------------------------------------------------------------- videomedian

#[test]
fn videomedian_removes_a_lone_speckle() {
    const LUMA: u8 = 60;
    const CHROMA: u8 = 90;
    const SPECKLE: u8 = 230;
    let (w, h) = (8usize, 8usize);
    let (sx, sy) = (4usize, 4usize);

    for size in [MedianSize::Five, MedianSize::Nine] {
        let mut source = i420_flat(w, h, LUMA, CHROMA);
        source[sy * w + sx] = SPECKLE;
        let mut element = VideoMedian::new().with_size(size);
        let out = run(
            &mut element,
            raw(RawVideoFormat::I420, w as u32, h as u32),
            source.clone(),
        );
        assert_eq!(out[sy * w + sx], LUMA, "{size:?} left the speckle");
        assert_eq!(
            i420_chroma(&out, w, h),
            i420_chroma(&source, w, h),
            "{size:?} touched chroma under the default lum-only"
        );
    }
}

#[test]
fn videomedian_filters_chroma_when_asked() {
    const LUMA: u8 = 60;
    const CHROMA: u8 = 90;
    const SPECKLE: u8 = 230;
    let (w, h) = (8usize, 8usize);
    // A speckle in the middle of the U plane, which is (w/2) x (h/2).
    let chroma_speckle = w * h + (h / 4) * (w / 2) + w / 4;

    let mut source = i420_flat(w, h, LUMA, CHROMA);
    source[chroma_speckle] = SPECKLE;

    let mut untouched = VideoMedian::new();
    let out = run(
        &mut untouched,
        raw(RawVideoFormat::I420, w as u32, h as u32),
        source.clone(),
    );
    assert_eq!(out[chroma_speckle], SPECKLE, "lum-only leaves chroma alone");

    let mut filtered = VideoMedian::new().with_luma_only(false);
    let out = run(
        &mut filtered,
        raw(RawVideoFormat::I420, w as u32, h as u32),
        source,
    );
    assert_eq!(out[chroma_speckle], CHROMA, "chroma is filtered too");
}

// --------------------------------------------------------------------- smooth

/// An I420 frame split into a dark left half and a light right half.
fn i420_vertical_step(w: usize, h: usize, dark: u8, light: u8, chroma: u8) -> Vec<u8> {
    let mut bytes: Vec<u8> = (0..w * h)
        .map(|i| if i % w < w / 2 { dark } else { light })
        .collect();
    bytes.resize(w * h + 2 * (w / 2) * (h / 2), chroma);
    bytes
}

#[test]
fn smooth_leaves_a_flat_frame_unchanged() {
    const LUMA: u8 = 77;
    const CHROMA: u8 = 90;
    let (w, h) = (8usize, 8usize);
    let source = i420_flat(w, h, LUMA, CHROMA);
    let mut element = Smooth::new();
    let out = run(
        &mut element,
        raw(RawVideoFormat::I420, w as u32, h as u32),
        source.clone(),
    );
    assert_eq!(out, source);
}

#[test]
fn smooth_bridges_a_step_inside_the_tolerance_but_not_beyond_it() {
    const CHROMA: u8 = 90;
    const DARK: u8 = 100;
    let (w, h) = (8usize, 8usize);
    let tolerance = match Smooth::new().get_property("tolerance") {
        Some(g2g_core::PropValue::Int(v)) => v as u8,
        other => panic!("tolerance reads back as an int, got {other:?}"),
    };

    // A step shorter than the tolerance is averaged across, so the two levels
    // move toward each other at the seam.
    let inside = i420_vertical_step(w, h, DARK, DARK + tolerance - 1, CHROMA);
    let mut element = Smooth::new();
    let out = run(
        &mut element,
        raw(RawVideoFormat::I420, w as u32, h as u32),
        inside.clone(),
    );
    let seam = (h / 2) * w + w / 2;
    assert!(out[seam - 1] > inside[seam - 1], "the dark side is lifted");
    assert!(out[seam] < inside[seam], "the light side is pulled down");

    // A step taller than the tolerance is left alone: no neighbour across it
    // qualifies, so each side averages only its own level.
    let beyond = i420_vertical_step(w, h, DARK, DARK + tolerance, CHROMA);
    let mut element = Smooth::new();
    let out = run(
        &mut element,
        raw(RawVideoFormat::I420, w as u32, h as u32),
        beyond.clone(),
    );
    assert_eq!(out, beyond);
}

#[test]
fn smooth_inactive_passes_the_frame_through() {
    const CHROMA: u8 = 90;
    const DARK: u8 = 100;
    const LIGHT: u8 = 104;
    let (w, h) = (8usize, 8usize);
    let source = i420_vertical_step(w, h, DARK, LIGHT, CHROMA);
    let mut element = Smooth::new().with_active(false);
    let out = run(
        &mut element,
        raw(RawVideoFormat::I420, w as u32, h as u32),
        source.clone(),
    );
    assert_eq!(out, source);
}

// ------------------------------------------------------------ aspectratiocrop

#[test]
fn aspectratiocrop_trims_to_the_target_ratio_and_keeps_the_centre() {
    // 640x480 asked for 16:9 keeps 640 * 9 / 16 = 360 rows out of the middle.
    const WIDTH: u32 = 640;
    const HEIGHT: u32 = 480;
    const ASPECT: (i32, i32) = (16, 9);
    let kept_rows = WIDTH * ASPECT.1 as u32 / ASPECT.0 as u32;
    let dropped_per_edge = (HEIGHT - kept_rows) / 2;

    // Each row carries its own index in the red channel, wrapped to fit a byte,
    // so a crop off the wrong edge shows up immediately.
    let row_marker = |row: u32| (row % (u8::MAX as u32 + 1)) as u8;
    let source: Vec<u8> = (0..HEIGHT)
        .flat_map(|row| [row_marker(row), 0, 0, 255].repeat(WIDTH as usize))
        .collect();

    let mut element = AspectRatioCrop::new().with_aspect_ratio(ASPECT.0, ASPECT.1);
    element
        .configure_pipeline(&raw(RawVideoFormat::Rgba8, WIDTH, HEIGHT))
        .unwrap();
    let packets = push(&mut element, source);
    assert_eq!(
        emitted_caps(&packets),
        raw(RawVideoFormat::Rgba8, WIDTH, kept_rows),
        "the cropped geometry reaches downstream"
    );

    let out = pixels(packets);
    assert_eq!(out.len(), (WIDTH * kept_rows * 4) as usize);
    for row in 0..kept_rows {
        assert_eq!(
            out[(row * WIDTH * 4) as usize],
            row_marker(row + dropped_per_edge),
            "output row {row} came from the wrong source row"
        );
    }
}

#[test]
fn aspectratiocrop_without_a_ratio_passes_the_frame_through() {
    const WIDTH: u32 = 16;
    const HEIGHT: u32 = 8;
    let source = rgba_flat(WIDTH as usize, HEIGHT as usize, [10, 20, 30, 255]);
    let mut element = AspectRatioCrop::new();
    let out = run(
        &mut element,
        raw(RawVideoFormat::Rgba8, WIDTH, HEIGHT),
        source.clone(),
    );
    assert_eq!(out, source);
}

// ------------------------------------------------------------------- registry

/// `registry` is std-gated, so this block only compiles when std is on
/// (`cargo test --workspace`, or `-p g2g-plugins --features std`).
#[cfg(feature = "std")]
mod registry_launch {
    use g2g_plugins::registry::default_registry;

    /// The launch names this milestone registers.
    const EFFECT_NAMES: [&str; 8] = [
        "aspectratiocrop",
        "chromahold",
        "coloreffects",
        "gaussianblur",
        "smooth",
        "videodiff",
        "videomedian",
        "zebrastripe",
    ];

    #[test]
    fn the_registry_builds_every_effect_and_reads_its_knobs_back() {
        // `parse_launch` looks a name up in `properties()` for its kind and then
        // calls `set_property`, so a knob only one half knows about is dead on a
        // launch line.
        let registry = default_registry();
        for name in EFFECT_NAMES {
            let element = registry
                .make_element(name)
                .unwrap_or_else(|| panic!("the registry resolves `{name}`"));
            let specs = element.properties();
            assert!(!specs.is_empty(), "`{name}` exposes no properties");
            for spec in specs {
                assert!(
                    element.get_property(spec.name).is_some(),
                    "`{name}` declares `{}` but cannot read it back",
                    spec.name
                );
            }
            assert!(
                element.metadata().klass.contains("Video"),
                "`{name}` is filed under {}, not a video klass",
                element.metadata().klass
            );
        }
    }
}
