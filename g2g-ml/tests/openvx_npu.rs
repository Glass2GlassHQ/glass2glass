// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

// Runs on a board with a VeriSilicon NPU through tools/stm32mp2-npu-smoke.sh, which
// names the model, the input picture and ST's stai_mpu output for it.
#![cfg(all(target_os = "linux", feature = "openvx"))]

use g2g_core::element::{AsyncElement, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, Colorimetry, Dim, G2gError, Interlace, PixelAspect, Rate, RawVideoFormat};
use g2g_ml::ovxinfer::OvxInference;

#[derive(Default)]
struct OneFrame {
    bytes: Option<Vec<u8>>,
}

impl OutputSink for OneFrame {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::DataFrame(frame)) = packet_slot.take() {
            self.bytes = frame.domain.as_system_slice().map(<[u8]>::to_vec);
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn path_from(variable: &str) -> String {
    std::env::var(variable).unwrap_or_else(|_| panic!("{variable} names a file"))
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

#[tokio::test]
async fn npu_scores_match_st_runtime_bit_for_bit() {
    if !g2g_plugins::openvx::available() {
        return;
    }
    let mut inference =
        OvxInference::from_file(&path_from("G2G_OPENVX_MODEL")).expect("model loads");
    let (width, height) = inference.input_dims();
    let video = Caps::RawVideo {
        format: RawVideoFormat::Rgb8,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate: Rate::Any,
        interlace: Interlace::Any,
        colorimetry: Colorimetry::UNKNOWN,
        pixel_aspect: PixelAspect::Any,
    };
    let narrowed = inference
        .intercept_caps(&video)
        .expect("RGB video is accepted");
    inference.configure_pipeline(&narrowed).expect("configure");
    let picture = std::fs::read(path_from("G2G_OPENVX_INPUT")).expect("read the picture");
    let mut scores = OneFrame::default();
    inference
        .process(
            PipelinePacket::DataFrame(Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(picture.into_boxed_slice())),
                timing: FrameTiming::default(),
                sequence: 0,
                meta: Default::default(),
            }),
            &mut scores,
        )
        .await
        .expect("inference runs");
    let scores = f32_values(&scores.bytes.expect("a score tensor"));
    let reference =
        f32_values(&std::fs::read(path_from("G2G_OPENVX_REFERENCE")).expect("read the reference"));
    assert_eq!(top_class(&scores), top_class(&reference));
    assert_eq!(scores, reference);
}
