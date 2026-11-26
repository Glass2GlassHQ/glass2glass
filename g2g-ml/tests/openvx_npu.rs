// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

// Runs on a board with a VeriSilicon NPU through tools/stm32mp2-npu-smoke.sh, which
// names the models, the input pictures and ST's results for them.
#![cfg(all(target_os = "linux", feature = "openvx", feature = "analytics"))]

use g2g_core::element::{AsyncElement, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::{
    AnalyticsMeta, Caps, Colorimetry, Dim, G2gError, Interlace, PixelAspect, PropValue, Rate,
    RawVideoFormat,
};
use g2g_ml::detect::DetectionPostprocess;
use g2g_ml::ovxinfer::OvxInference;

// ST decodes boxes in f16 arithmetic, whose step just below 1.0 is 2^-11
const SSD_BOX_TOLERANCE: f32 = 2.0 / 2048.0;
// class, score, xmin, ymin, xmax, ymax
const ST_DETECTION_FIELDS: usize = 6;

#[derive(Default)]
struct OneFrame {
    frame: Option<Frame>,
}

impl OutputSink for OneFrame {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::DataFrame(frame)) = packet_slot.take() {
            self.frame = Some(frame);
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn env_value(variable: &str) -> String {
    std::env::var(variable).unwrap_or_else(|_| panic!("{variable} is set"))
}

fn threshold_from(variable: &str) -> f32 {
    env_value(variable)
        .parse()
        .unwrap_or_else(|_| panic!("{variable} is a number"))
}

fn top_class(scores: &[f32]) -> usize {
    scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .expect("scores")
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

fn rgb_caps(width: u32, height: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgb8,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate: Rate::Any,
        interlace: Interlace::Any,
        colorimetry: Colorimetry::UNKNOWN,
        pixel_aspect: PixelAspect::Any,
    }
}

fn picture_frame(variable: &str) -> PipelinePacket {
    let picture = std::fs::read(env_value(variable)).expect("read the picture");
    PipelinePacket::DataFrame(Frame::from_system_bytes(picture, FrameTiming::default(), 0))
}

async fn infer(inference: &mut OvxInference, picture: PipelinePacket) -> Frame {
    let (width, height) = inference.input_dims();
    let narrowed = inference
        .intercept_caps(&rgb_caps(width, height))
        .expect("RGB video is accepted");
    inference.configure_pipeline(&narrowed).expect("configure");
    let mut produced = OneFrame::default();
    inference
        .process(picture, &mut produced)
        .await
        .expect("inference runs");
    produced.frame.expect("inference pushed a frame")
}

#[tokio::test]
async fn npu_scores_match_st_runtime_bit_for_bit() {
    if !g2g_plugins::openvx::available() {
        return;
    }
    let mut inference =
        OvxInference::from_file(&env_value("G2G_OPENVX_MODEL")).expect("model loads");
    let scores = infer(&mut inference, picture_frame("G2G_OPENVX_INPUT")).await;
    let scores = f32_values(scores.domain.as_system_slice().expect("a score tensor"));
    let reference =
        f32_values(&std::fs::read(env_value("G2G_OPENVX_REFERENCE")).expect("read the reference"));
    assert_eq!(top_class(&scores), top_class(&reference));
    assert_eq!(scores, reference);
}

#[tokio::test]
async fn npu_ssd_detections_match_st_postprocess() {
    if !g2g_plugins::openvx::available() {
        return;
    }
    let mut inference =
        OvxInference::from_file(&env_value("G2G_OPENVX_SSD_MODEL")).expect("model loads");
    inference
        .set_property("attach-tensor", PropValue::Bool(true))
        .unwrap();
    inference
        .set_property("tensor-name", PropValue::Str(String::from("ssd")))
        .unwrap();
    let (width, height) = inference.input_dims();
    let picture = infer(&mut inference, picture_frame("G2G_OPENVX_SSD_INPUT")).await;

    let mut postprocess = DetectionPostprocess::new(
        threshold_from("G2G_OPENVX_SSD_CONFIDENCE"),
        threshold_from("G2G_OPENVX_SSD_IOU"),
    );
    for (name, value) in [
        ("head", PropValue::Str(String::from("ssd"))),
        ("tensor-name", PropValue::Str(String::from("ssd"))),
    ] {
        postprocess.set_property(name, value).unwrap();
    }
    postprocess
        .configure_pipeline(&rgb_caps(width, height))
        .expect("configure");
    let mut decoded = OneFrame::default();
    postprocess
        .process(PipelinePacket::DataFrame(picture), &mut decoded)
        .await
        .expect("decode runs");
    let decoded = decoded.frame.expect("the picture goes downstream");
    let analytics = decoded
        .meta
        .get::<AnalyticsMeta>()
        .expect("detections attached");
    let mut detections: Vec<_> = analytics.detections().collect();
    detections.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));

    let reference = f32_values(
        &std::fs::read(env_value("G2G_OPENVX_SSD_REFERENCE")).expect("read the reference"),
    );
    let (reference, []) = reference.as_chunks::<ST_DETECTION_FIELDS>() else {
        panic!("the reference holds whole detections");
    };
    assert!(!reference.is_empty(), "ST finds something in the picture");
    assert_eq!(detections.len(), reference.len());
    for (detection, &[class, score, x0, y0, x1, y1]) in detections.iter().zip(reference) {
        // ST's classes count the background as class 0
        assert_eq!(detection.label + 1, class as u32);
        assert_eq!(detection.confidence, score);
        let bbox = detection.bbox;
        let got = [bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h];
        for (got, expected) in got.iter().zip([x0, y0, x1, y1]) {
            assert!(
                (got - expected).abs() <= SSD_BOX_TOLERANCE,
                "box {got:?} against ST's {expected:?}"
            );
        }
    }
}
