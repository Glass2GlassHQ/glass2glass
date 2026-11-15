// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::cell::Cell;

use g2g_core::error::G2gError;
use g2g_core::frame::Frame;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{drive_ready, run_source_transform_sink, Chain, StaticSink, StaticTransform};
use g2g_mcu::logmel::MICRO_SPEECH_SETTINGS;
use g2g_mcu::{
    FrameGrabber, GrabberSrc, Int8Quantize, MicroFlowModel, MicroFrontend, TensorInference,
};
use microflow::model;

// rustc runs from this crate's directory, and the macro reads the path as given.
#[model("../../g2g-mcu/tests/fixtures/keyword/speech.tflite")]
struct Speech;

const CLIP: &[u8] = include_bytes!("../../../g2g-mcu/tests/fixtures/keyword/yes.s16le");
const EXPECTED_FEATURES: &[u8; FEATURES] =
    include_bytes!("../../../g2g-mcu/tests/fixtures/keyword/features.i8");
const EXPECTED_SCORE_BYTES: &[u8; KEYWORDS * SCORE_BYTES] =
    include_bytes!("../../../g2g-mcu/tests/fixtures/keyword/scores.f32le");

const CHANNELS: usize = 40;
const FEATURES: usize = MICRO_SPEECH_SETTINGS.frames * CHANNELS;
pub const KEYWORDS: usize = 4;
// speech.tflite's output order.
pub const LABELS: [&str; KEYWORDS] = ["silence", "unknown", "yes", "no"];
const SCORE_BYTES: usize = 4;
pub const CLIP_BYTES: usize = MICRO_SPEECH_SETTINGS.clip_samples() * 2;
const FRONTEND_BYTES: usize = FEATURES * 4;
// One clip, so the frame period only stamps PTS 0.
const CLIP_INTERVAL_NS: u64 = 1_000_000_000;

// speech.tflite's input tensor quantization, as quantization.txt records it.
pub const INPUT_SCALE: f32 = 0.101_715_684;
pub const INPUT_ZERO_POINT: i32 = -128;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeywordRun {
    pub features_digest: u64,
    pub scores: [f32; KEYWORDS],
}

impl KeywordRun {
    pub fn digest(&self) -> u64 {
        let mut digest = self.features_digest;
        for score in self.scores {
            digest = fnv1a(digest, &score.to_bits().to_le_bytes());
        }
        digest
    }
}

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

const fn fnv1a(mut hash: u64, mut bytes: &[u8]) -> u64 {
    while let [first, rest @ ..] = bytes {
        hash = (hash ^ *first as u64).wrapping_mul(FNV_PRIME);
        bytes = rest;
    }
    hash
}

pub const EXPECTED_RUN: KeywordRun = expected_run();

const fn expected_run() -> KeywordRun {
    let mut scores = [0.0f32; KEYWORDS];
    let mut index = 0;
    while index < KEYWORDS {
        let at = index * SCORE_BYTES;
        scores[index] = f32::from_bits(u32::from_le_bytes([
            EXPECTED_SCORE_BYTES[at],
            EXPECTED_SCORE_BYTES[at + 1],
            EXPECTED_SCORE_BYTES[at + 2],
            EXPECTED_SCORE_BYTES[at + 3],
        ]));
        index += 1;
    }
    KeywordRun {
        features_digest: fnv1a(FNV_OFFSET_BASIS, EXPECTED_FEATURES),
        scores,
    }
}

struct ClipGrabber;

impl FrameGrabber for ClipGrabber {
    async fn capture(&mut self, buf: &mut [u8]) -> Result<usize, G2gError> {
        let Some(slot) = buf.get_mut(..CLIP.len()) else {
            return Err(G2gError::CapsMismatch);
        };
        slot.copy_from_slice(CLIP);
        Ok(CLIP.len())
    }
}

// Hashes the int8 features in passing, so a device run can prove them equal to
// the host's without sending them out.
struct FeatureDigest<'a> {
    digest: &'a Cell<u64>,
}

impl StaticTransform for FeatureDigest<'_> {
    async fn process(&mut self, input: Frame) -> Result<Option<Frame>, G2gError> {
        if let Some(features) = input.domain.as_system_slice() {
            self.digest.set(fnv1a(self.digest.get(), features));
        }
        Ok(Some(input))
    }
}

struct ScoreSink<F: FnMut([f32; KEYWORDS])> {
    on_scores: F,
}

impl<F: FnMut([f32; KEYWORDS])> StaticSink for ScoreSink<F> {
    async fn consume(&mut self, frame: Frame) -> Result<(), G2gError> {
        let Some(bytes) = frame.domain.as_system_slice() else {
            return Err(G2gError::UnsupportedDomain);
        };
        let mut scores = [0.0f32; KEYWORDS];
        for (score, chunk) in scores.iter_mut().zip(bytes.chunks_exact(SCORE_BYTES)) {
            let &[b0, b1, b2, b3] = chunk else {
                continue;
            };
            *score = f32::from_le_bytes([b0, b1, b2, b3]);
        }
        (self.on_scores)(scores);
        Ok(())
    }
}

// Returns the digest of every clip's int8 features.
pub async fn run_keyword_graph(
    grabber: impl FrameGrabber,
    clip_limit: Option<u32>,
    clip_interval_ns: u64,
    on_scores: impl FnMut([f32; KEYWORDS]),
) -> Result<u64, G2gError> {
    let clip_ring: StaticLendRing<1, CLIP_BYTES> = StaticLendRing::new();
    let frontend_ring: StaticLendRing<1, FRONTEND_BYTES> = StaticLendRing::new();
    let feature_ring: StaticLendRing<1, FEATURES> = StaticLendRing::new();
    let score_ring: StaticLendRing<1, { KEYWORDS * SCORE_BYTES }> = StaticLendRing::new();
    let digest = Cell::new(FNV_OFFSET_BASIS);

    // SAFETY (every `with_ring` below): the rings are locals of this function
    // and outlive the pipeline run, which drains before returning.
    let frontend = unsafe {
        MicroFrontend::<CHANNELS, 1, FRONTEND_BYTES>::with_ring(
            MICRO_SPEECH_SETTINGS,
            &frontend_ring,
        )
    }?;
    // SAFETY: as above.
    let quantize = unsafe {
        Int8Quantize::with_ring(
            frontend.output_format(),
            INPUT_SCALE,
            INPUT_ZERO_POINT,
            &feature_ring,
        )
    }?;
    // SAFETY: as above.
    let inference = unsafe {
        TensorInference::with_ring(MicroFlowModel::new(Speech::predict_quantized), &score_ring)
    };
    if quantize.output_format() != inference.input_format() {
        return Err(G2gError::CapsMismatch);
    }
    // SAFETY: as above.
    let source = unsafe { GrabberSrc::with_ring(grabber, &clip_ring, clip_interval_ns) };
    let source = match clip_limit {
        Some(clips) => source.with_frame_limit(clips),
        None => source,
    };
    let transform = Chain(
        frontend,
        Chain(
            quantize,
            Chain(FeatureDigest { digest: &digest }, inference),
        ),
    );
    run_source_transform_sink(source, transform, ScoreSink { on_scores }).await?;
    Ok(digest.get())
}

async fn run_keyword_async() -> Option<KeywordRun> {
    let mut scores = [0.0f32; KEYWORDS];
    let features_digest =
        run_keyword_graph(ClipGrabber, Some(1), CLIP_INTERVAL_NS, |clip_scores| {
            scores = clip_scores
        })
        .await
        .ok()?;
    Some(KeywordRun {
        features_digest,
        scores,
    })
}

// One future keeps MicroFlow's operators inlined, so their bounds checks fold away.
pub fn run_keyword() -> Option<KeywordRun> {
    drive_ready(run_keyword_async())?
}
