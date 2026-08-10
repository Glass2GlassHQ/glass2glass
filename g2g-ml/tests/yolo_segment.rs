//! M992: a real instance-segmentation model end to end through the g2g element
//! graph, the mask sibling of the M448 detector. A real YOLO `-seg` export runs in
//! `OrtSegmentation` and the emitted `Segmentation` / `Roi` nodes are checked
//! against the picture itself, not against the model: the sample dog is white on
//! grass, so the white pixels are an oracle the model never saw.
//!
//! Three cases: the sample image (mask agrees with the white pixels), a flat frame
//! of the grass colour (nothing found), and the dog shrunk into the top-left
//! quadrant (mask and ROI follow it there).
//!
//! The model is ~11 MB so it is not committed (repo fixtures are KB-scale);
//! `tools/segment-fixture.sh` / `fixtures/segmentation/gen.py` obtain it on demand
//! into a gitignored dir. The tests skip when the fixtures are absent.
//!
//! Run:
//!   tools/segment-fixture.sh
//!   cargo test -p g2g-ml --features "ort analytics" --test yolo_segment -- --nocapture

#![cfg(all(feature = "ort", feature = "analytics"))]

use std::path::PathBuf;

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{AnalyticsMeta, Caps, Dim, G2gError, Rate, RawVideoFormat, Segmentation};
use g2g_ml::ortsegment::OrtSegmentation;

const SIZE: u32 = 640;
/// The prototype grid of a 640-input YOLO `-seg` export: a quarter of the input.
const PROTO: usize = 160;
/// COCO class indices the sample Samoyed can plausibly land on: 15 "cat", 16 "dog".
/// The nano `-seg` export calls it a cat; a larger export calls it a dog. The test
/// is about the mask, so it accepts either animal.
const ANIMAL_LABELS: [u32; 2] = [15, 16];
/// A pixel counts as the dog when it is bright and near-grey (the Samoyed's white
/// coat), which the green grass and the dark foliage both fail.
const WHITE_MIN: i32 = 130;
const WHITE_SPREAD: i32 = 80;

/// Keeps the analytics graph and the pixels of the frame the element forwarded.
#[derive(Default)]
struct MetaSink {
    analytics: Option<AnalyticsMeta>,
    forwarded: Option<Vec<u8>>,
}

impl OutputSink for MetaSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(frame) = &packet {
                if let Some(analytics) = frame.meta.get::<AnalyticsMeta>() {
                    self.analytics = Some(analytics.clone());
                }
                if let Some(bytes) = frame.domain.as_system_slice() {
                    self.forwarded = Some(bytes.to_vec());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/segmentation")
}

fn rgba_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(SIZE),
        height: Dim::Fixed(SIZE),
        framerate: Rate::Any,
        interlace: g2g_core::Interlace::Any,
    }
}

fn frame(bytes: Vec<u8>) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    })
}

/// The configured element plus the sample RGBA frame, or `None` when the
/// gitignored fixtures have not been built.
fn producer_and_input() -> Option<(OrtSegmentation, Vec<u8>)> {
    let dir = fixture_dir();
    let (model_path, input_path) = (dir.join("model.onnx"), dir.join("input_rgba.bin"));
    if !model_path.exists() || !input_path.exists() {
        eprintln!(
            "segmentation fixtures absent ({}); run tools/segment-fixture.sh first. skipping.",
            dir.display()
        );
        return None;
    }
    let input = std::fs::read(&input_path).expect("read input");
    assert_eq!(
        input.len(),
        (SIZE * SIZE * 4) as usize,
        "RGBA8 640x640 fixture"
    );
    let mut element =
        OrtSegmentation::from_file(model_path.to_str().unwrap()).expect("model loads");
    let geometry = element.geometry().expect("model geometry");
    assert_eq!((geometry.input_width, geometry.input_height), (SIZE, SIZE));
    assert_eq!(geometry.classes, 80, "COCO classes");
    assert_eq!(geometry.prototypes, 32, "YOLO -seg mask prototypes");
    assert_eq!(
        (geometry.proto_width, geometry.proto_height),
        (PROTO, PROTO)
    );
    element
        .configure_pipeline(&rgba_caps())
        .expect("configure the producer");
    Some((element, input))
}

async fn segment(element: &mut OrtSegmentation, rgba: Vec<u8>) -> MetaSink {
    let mut sink = MetaSink::default();
    element
        .process(frame(rgba), &mut sink)
        .await
        .expect("segmentation runs");
    sink
}

fn is_white(rgba: &[u8], x: usize, y: usize) -> bool {
    let px = (y * SIZE as usize + x) * 4;
    let (r, g, b) = (rgba[px] as i32, rgba[px + 1] as i32, rgba[px + 2] as i32);
    let low = r.min(g).min(b);
    let high = r.max(g).max(b);
    low > WHITE_MIN && (high - low) < WHITE_SPREAD
}

/// The frame pixel a mask sample covers: the mask spans exactly the instance's box.
fn mask_sample_pixel(segmentation: &Segmentation, i: u32, j: u32) -> (usize, usize) {
    let bbox = segmentation.bbox;
    let mask = &segmentation.mask;
    let fx = bbox.x + (i as f32 + 0.5) / mask.width() as f32 * bbox.w;
    let fy = bbox.y + (j as f32 + 0.5) / mask.height() as f32 * bbox.h;
    let limit = SIZE as f32 - 1.0;
    (
        (fx * SIZE as f32).clamp(0.0, limit) as usize,
        (fy * SIZE as f32).clamp(0.0, limit) as usize,
    )
}

/// The highest-confidence animal instance in the graph.
fn animal(analytics: &AnalyticsMeta) -> Option<&Segmentation> {
    analytics
        .segmentations()
        .filter(|s| ANIMAL_LABELS.contains(&s.label))
        .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
}

#[tokio::test]
async fn the_mask_covers_the_dogs_pixels() {
    let Some((mut element, input)) = producer_and_input() else {
        return;
    };
    let sink = segment(&mut element, input.clone()).await;
    assert_eq!(
        sink.forwarded.as_deref(),
        Some(input.as_slice()),
        "the producer forwards the frame unchanged and only adds metadata"
    );
    let analytics = sink.analytics.expect("frame carries AnalyticsMeta");
    let segmentation = animal(&analytics).expect("an animal instance");
    eprintln!(
        ">> label {} confidence {:.3} bbox {:?} mask {}x{}",
        segmentation.label,
        segmentation.confidence,
        segmentation.bbox,
        segmentation.mask.width(),
        segmentation.mask.height()
    );
    assert!(
        segmentation.confidence > 0.5,
        "confidence {}",
        segmentation.confidence
    );

    // The mask spans the box at prototype resolution, so its sample count follows
    // the box, and the box fills most of this picture.
    let mask = &segmentation.mask;
    assert_eq!(
        mask.width(),
        (segmentation.bbox.w * PROTO as f32).ceil() as u32
    );
    assert!(
        segmentation.bbox.w > 0.6 && segmentation.bbox.h > 0.6,
        "the dog fills the frame: {:?}",
        segmentation.bbox
    );

    // The oracle: the dog is the white region of the picture, which the model never
    // saw as a label. Agreement is measured over the mask's own samples.
    let (mut both, mut either, mut covered) = (0u32, 0u32, 0u32);
    for j in 0..mask.height() {
        for i in 0..mask.width() {
            let (x, y) = mask_sample_pixel(segmentation, i, j);
            let masked = mask.sample(i, j) == Some(u8::MAX);
            let white = is_white(&input, x, y);
            covered += u32::from(masked);
            both += u32::from(masked && white);
            either += u32::from(masked || white);
        }
    }
    let intersection_over_union = both as f32 / either as f32;
    let coverage = covered as f32 / (mask.width() * mask.height()) as f32;
    eprintln!(">> mask/white IoU {intersection_over_union:.3}, box coverage {coverage:.3}");
    assert!(
        intersection_over_union > 0.6,
        "mask should agree with the white dog pixels, IoU {intersection_over_union}"
    );
    // A mask that covered the whole box (or almost none of it) would pass no useful
    // test; a dog leaves grass in the corners of its box.
    assert!(
        (0.3..0.85).contains(&coverage),
        "mask covers {coverage} of its box"
    );

    // The ROI is the mask-tight box: inside the detection box, and smaller.
    let roi = analytics.rois().next().expect("an ROI per instance");
    eprintln!(">> roi {:?}", roi.bbox);
    let (bbox, roi_box) = (segmentation.bbox, roi.bbox);
    assert!(
        roi_box.x >= bbox.x - 1e-4
            && roi_box.y >= bbox.y - 1e-4
            && roi_box.x + roi_box.w <= bbox.x + bbox.w + 1e-4
            && roi_box.y + roi_box.h <= bbox.y + bbox.h + 1e-4,
        "ROI {roi_box:?} must lie inside the detection box {bbox:?}"
    );
    assert!(roi_box.w * roi_box.h > 0.2, "ROI is a real region");
    assert_eq!(roi.label, segmentation.label);
    assert_eq!(
        analytics.relations.len(),
        analytics.segmentations().count(),
        "each segmentation is related to its ROI"
    );
}

#[tokio::test]
async fn a_flat_frame_yields_no_instances() {
    let Some((mut element, _)) = producer_and_input() else {
        return;
    };
    // The grass colour, uniform: no object, so no masks. Proves the graph above is
    // not a fixed shape the element emits for any frame.
    let flat: Vec<u8> = [90u8, 130, 60, 255]
        .iter()
        .copied()
        .cycle()
        .take((SIZE * SIZE * 4) as usize)
        .collect();
    let sink = segment(&mut element, flat).await;
    let analytics = sink.analytics.expect("frame carries AnalyticsMeta");
    eprintln!(">> flat frame nodes: {}", analytics.nodes.len());
    assert_eq!(
        analytics.nodes.len(),
        0,
        "a flat frame has nothing to segment"
    );
}

#[tokio::test]
async fn the_mask_follows_the_dog_into_a_corner() {
    let Some((mut element, input)) = producer_and_input() else {
        return;
    };
    // Halve the picture into the top-left quadrant (nearest-neighbour) over a flat
    // grass background, so the same dog is somewhere else. The mask and ROI have to
    // move with it.
    let mut moved: Vec<u8> = [90u8, 130, 60, 255]
        .iter()
        .copied()
        .cycle()
        .take((SIZE * SIZE * 4) as usize)
        .collect();
    let side = SIZE as usize;
    for y in 0..side / 2 {
        for x in 0..side / 2 {
            let src = ((y * 2) * side + x * 2) * 4;
            let dst = (y * side + x) * 4;
            moved[dst..dst + 4].copy_from_slice(&input[src..src + 4]);
        }
    }
    let sink = segment(&mut element, moved.clone()).await;
    let analytics = sink.analytics.expect("frame carries AnalyticsMeta");
    let segmentation = animal(&analytics).expect("the moved animal is still found");
    eprintln!(
        ">> moved: label {} confidence {:.3} bbox {:?}",
        segmentation.label, segmentation.confidence, segmentation.bbox
    );
    let bbox = segmentation.bbox;
    assert!(
        bbox.x + bbox.w < 0.6 && bbox.y + bbox.h < 0.6,
        "the box must follow the dog into the top-left quadrant: {bbox:?}"
    );
    // Every covered mask sample lands on a white pixel far more often than not,
    // i.e. the mask tracks the moved pixels rather than the previous frame's box.
    let mask = &segmentation.mask;
    let (mut covered, mut white_covered) = (0u32, 0u32);
    for j in 0..mask.height() {
        for i in 0..mask.width() {
            if mask.sample(i, j) == Some(u8::MAX) {
                let (x, y) = mask_sample_pixel(segmentation, i, j);
                covered += 1;
                white_covered += u32::from(is_white(&moved, x, y));
            }
        }
    }
    let precision = white_covered as f32 / covered as f32;
    eprintln!(">> moved mask precision on white pixels {precision:.3}");
    assert!(covered > 0, "the moved dog has a mask");
    assert!(
        precision > 0.75,
        "covered samples should be the dog's white pixels, got {precision}"
    );
    let roi = analytics.rois().find(|r| r.label == segmentation.label);
    let roi = roi.expect("an ROI for the moved instance").bbox;
    assert!(
        roi.x + roi.w < 0.6 && roi.y + roi.h < 0.6,
        "the ROI moved too: {roi:?}"
    );
}
