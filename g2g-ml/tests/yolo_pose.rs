//! M1713: a real YOLO11 pose model on a letterboxed portrait photo, decoded by
//! `DetectionPostprocess head=pose` and checked against Ultralytics' own
//! predictions for the same ONNX file: the same people, with every box corner
//! and keypoint landing on the picture where Ultralytics puts it.
//!
//! The model is not committed; `tools/pose-obb-fixture.sh` builds the fixture
//! into a gitignored dir, and the test skips when it is absent.

#![cfg(all(feature = "ort", feature = "analytics"))]

include!("util/letterbox_fixture.rs");

// 17 COCO keypoints of three channels after the box and person score.
const KEYPOINTS: usize = 17;
const POSE_CHANNELS: u32 = 5 + 3 * KEYPOINTS as u32;
// A reference row: box corners x1 y1 x2 y2, confidence, then x y visibility
// per keypoint.
const FIRST_KEYPOINT: usize = 5;

#[tokio::test]
async fn yolo_pose_matches_ultralytics_on_a_letterboxed_picture() {
    let Some((analytics, reference)) = decode_fixture("pose", "pose", POSE_CHANNELS).await else {
        return;
    };
    let poses: Vec<_> = analytics.poses().collect();
    assert_eq!(poses.len(), reference.rows.len(), "same number of people");
    assert!(!poses.is_empty(), "the bus photo has people in it");

    let centres: Vec<(f32, f32)> = poses
        .iter()
        .map(|p| (p.bbox.x + p.bbox.w / 2.0, p.bbox.y + p.bbox.h / 2.0))
        .collect();
    for row in &reference.rows {
        let (x1, y1, x2, y2) = (row[0], row[1], row[2], row[3]);
        let pose = poses[nearest(&centres, (x1 + x2) / 2.0, (y1 + y2) / 2.0)];
        assert_within(pose.bbox.x, x1, COORDINATE_TOLERANCE, "box left");
        assert_within(pose.bbox.y, y1, COORDINATE_TOLERANCE, "box top");
        assert_within(
            pose.bbox.x + pose.bbox.w,
            x2,
            COORDINATE_TOLERANCE,
            "box right",
        );
        assert_within(
            pose.bbox.y + pose.bbox.h,
            y2,
            COORDINATE_TOLERANCE,
            "box bottom",
        );
        assert_within(pose.confidence, row[4], SCORE_TOLERANCE, "person score");
        assert_eq!(pose.keypoints.len(), KEYPOINTS);
        let (wanted_keypoints, _) = row[FIRST_KEYPOINT..].as_chunks::<3>();
        for (keypoint, &[x, y, visibility]) in pose.keypoints.iter().zip(wanted_keypoints) {
            assert_within(keypoint.x, x, COORDINATE_TOLERANCE, "keypoint x");
            assert_within(keypoint.y, y, COORDINATE_TOLERANCE, "keypoint y");
            assert_within(
                keypoint.confidence,
                visibility,
                SCORE_TOLERANCE,
                "keypoint visibility",
            );
        }
    }
}
