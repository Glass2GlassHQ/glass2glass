// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::cell::Cell;

use g2g_core::error::G2gError;
use g2g_core::frame::Frame;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{
    drive_ready, run_source_transform_sink, Chain, StaticSink, StaticTransform, TensorDType,
    TensorLayout, TensorShape,
};
use g2g_mcu::logmel::MICRO_SPEECH_SETTINGS;
use g2g_mcu::{FrameGrabber, GrabberSrc, Int8Quantize, MicroFrontend, TensorFormat};

const CLIP: &[u8] = include_bytes!("../../../g2g-mcu/tests/fixtures/keyword/yes.s16le");
const EXPECTED_FEATURES: &[u8; FEATURES] =
    include_bytes!("../../../g2g-mcu/tests/fixtures/keyword/features.i8");

const CHANNELS: usize = 40;
pub const FEATURES: usize = MICRO_SPEECH_SETTINGS.frames * CHANNELS;
pub const FEATURE_FORMAT: TensorFormat = TensorFormat {
    dtype: TensorDType::I8,
    shape: TensorShape::new([1, FEATURES as u32]),
    layout: TensorLayout::Nhwc,
};
pub const CLIP_BYTES: usize = MICRO_SPEECH_SETTINGS.clip_samples() * 2;
const FRONTEND_BYTES: usize = FEATURES * 4;
// One clip, so the frame period only stamps PTS 0.
const CLIP_INTERVAL_NS: u64 = 1_000_000_000;

// speech.tflite's input tensor quantization, as quantization.txt records it.
pub const INPUT_SCALE: f32 = 0.101_715_684;
pub const INPUT_ZERO_POINT: i32 = -128;

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

const fn fnv1a(mut hash: u64, mut bytes: &[u8]) -> u64 {
    while let [first, rest @ ..] = bytes {
        hash = (hash ^ *first as u64).wrapping_mul(FNV_PRIME);
        bytes = rest;
    }
    hash
}

pub const EXPECTED_FEATURES_DIGEST: u64 = fnv1a(FNV_OFFSET_BASIS, EXPECTED_FEATURES);

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

struct Discard;

impl StaticSink for Discard {
    async fn consume(&mut self, _frame: Frame) -> Result<(), G2gError> {
        Ok(())
    }
}

// grabber -> MicroFrontend -> Int8Quantize -> tail -> sink, where tail takes
// FEATURE_FORMAT frames.
pub async fn run_keyword_graph(
    grabber: impl FrameGrabber,
    clip_limit: Option<u32>,
    clip_interval_ns: u64,
    tail: impl StaticTransform,
    sink: impl StaticSink,
) -> Result<(), G2gError> {
    let clip_ring: StaticLendRing<1, CLIP_BYTES> = StaticLendRing::new();
    let frontend_ring: StaticLendRing<1, FRONTEND_BYTES> = StaticLendRing::new();
    let feature_ring: StaticLendRing<1, FEATURES> = StaticLendRing::new();

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
    if quantize.output_format() != FEATURE_FORMAT {
        return Err(G2gError::CapsMismatch);
    }
    // SAFETY: as above.
    let source = unsafe { GrabberSrc::with_ring(grabber, &clip_ring, clip_interval_ns) };
    let source = match clip_limit {
        Some(clips) => source.with_frame_limit(clips),
        None => source,
    };
    run_source_transform_sink(source, Chain(frontend, Chain(quantize, tail)), sink).await
}

async fn run_keyword_async() -> Option<u64> {
    let digest = Cell::new(FNV_OFFSET_BASIS);
    run_keyword_graph(
        ClipGrabber,
        Some(1),
        CLIP_INTERVAL_NS,
        FeatureDigest { digest: &digest },
        Discard,
    )
    .await
    .ok()?;
    Some(digest.get())
}

// Returns the digest of the clip's int8 features.
pub fn run_keyword() -> Option<u64> {
    drive_ready(run_keyword_async())?
}
