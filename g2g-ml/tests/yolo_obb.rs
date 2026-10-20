//! M1713: a real YOLO11 oriented-box model (DOTA classes) on a letterboxed
//! wide photo, decoded by `DetectionPostprocess head=obb` and checked against
//! Ultralytics' own predictions for the same ONNX file: the same boxes after
//! probiou suppression, each centre, size and angle where Ultralytics puts it.
//!
//! The model is not committed; `tools/pose-obb-fixture.sh` builds the fixture
//! into a gitignored dir, and the test skips when it is absent.

#![cfg(all(feature = "ort", feature = "analytics"))]

include!("util/letterbox_fixture.rs");

// 15 DOTA classes between the box channels and the angle channel.
const OBB_CHANNELS: u32 = 4 + 15 + 1;
const ANGLE_TOLERANCE: f32 = 0.02;

#[tokio::test]
async fn yolo_obb_matches_ultralytics_on_a_letterboxed_picture() {
    let Some((analytics, reference)) = decode_fixture("obb", "obb", OBB_CHANNELS).await else {
        return;
    };
    let boxes: Vec<_> = analytics.oriented_detections().collect();
    assert_eq!(boxes.len(), reference.rows.len(), "same number of boxes");
    assert!(!boxes.is_empty(), "the boats photo has boats in it");

    let centres: Vec<(f32, f32)> = boxes.iter().map(|o| (o.cx, o.cy)).collect();
    for row in &reference.rows {
        // cx cy w h angle label confidence, as obb/gen.py writes them.
        let decoded = boxes[nearest(&centres, row[0], row[1])];
        assert_within(decoded.cx, row[0], COORDINATE_TOLERANCE, "centre x");
        assert_within(decoded.cy, row[1], COORDINATE_TOLERANCE, "centre y");
        assert_within(decoded.w, row[2], COORDINATE_TOLERANCE, "width");
        assert_within(decoded.h, row[3], COORDINATE_TOLERANCE, "height");
        assert_within(decoded.angle_radians, row[4], ANGLE_TOLERANCE, "angle");
        assert_eq!(decoded.label as f32, row[5], "class");
        assert_within(decoded.confidence, row[6], SCORE_TOLERANCE, "confidence");
    }
}
