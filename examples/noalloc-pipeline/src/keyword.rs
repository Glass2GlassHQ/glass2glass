use core::cell::Cell;

use g2g_core::error::G2gError;
use g2g_core::frame::Frame;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{drive_ready, run_source_transform_sink, Chain, StaticSink, StaticTransform};
use g2g_mcu::{
    FrameGrabber, GrabberSrc, Int8Quantize, LogMelSettings, LogMelSpectrogram, MicroFlowModel,
    TensorInference,
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

const MELS: usize = 40;
const FEATURES: usize = 1960;
pub const KEYWORDS: usize = 4;
const SCORE_BYTES: usize = 4;
const CLIP_BYTES: usize = 15_840 * 2;
const LOG_MEL_BYTES: usize = FEATURES * 4;
// One clip, so the frame period only stamps PTS 0.
const CLIP_INTERVAL_NS: u64 = 1_000_000_000;

// The micro_speech front end: 49 windows of 30 ms every 20 ms at 16 kHz, 40 mel
// bands between 125 Hz and 7.5 kHz.
const SETTINGS: LogMelSettings = LogMelSettings {
    sample_rate: 16_000,
    window_samples: 480,
    hop_samples: 320,
    frames: FEATURES / MELS,
    lowest_frequency: 125.0,
    highest_frequency: 7_500.0,
};
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

struct ScoreSink<'a> {
    scores: &'a Cell<[f32; KEYWORDS]>,
}

impl StaticSink for ScoreSink<'_> {
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
        self.scores.set(scores);
        Ok(())
    }
}

async fn run_keyword_async() -> Option<KeywordRun> {
    let clip_ring: StaticLendRing<1, CLIP_BYTES> = StaticLendRing::new();
    let log_mel_ring: StaticLendRing<1, LOG_MEL_BYTES> = StaticLendRing::new();
    let feature_ring: StaticLendRing<1, FEATURES> = StaticLendRing::new();
    let score_ring: StaticLendRing<1, { KEYWORDS * SCORE_BYTES }> = StaticLendRing::new();
    let digest = Cell::new(FNV_OFFSET_BASIS);
    let scores = Cell::new([0.0f32; KEYWORDS]);

    // SAFETY (every `with_ring` below): the rings are locals of this function
    // and outlive the pipeline run, which drains before returning.
    let log_mel =
        unsafe { LogMelSpectrogram::<MELS, 1, LOG_MEL_BYTES>::with_ring(SETTINGS, &log_mel_ring) }
            .ok()?;
    // SAFETY: as above.
    let quantize = unsafe {
        Int8Quantize::with_ring(
            log_mel.output_format(),
            INPUT_SCALE,
            INPUT_ZERO_POINT,
            &feature_ring,
        )
    }
    .ok()?;
    // SAFETY: as above.
    let inference = unsafe {
        TensorInference::with_ring(MicroFlowModel::new(Speech::predict_quantized), &score_ring)
    };
    if quantize.output_format() != inference.input_format() {
        return None;
    }
    // SAFETY: as above.
    let source = unsafe { GrabberSrc::with_ring(ClipGrabber, &clip_ring, CLIP_INTERVAL_NS) };
    let transform = Chain(
        log_mel,
        Chain(
            quantize,
            Chain(FeatureDigest { digest: &digest }, inference),
        ),
    );
    run_source_transform_sink(
        source.with_frame_limit(1),
        transform,
        ScoreSink { scores: &scores },
    )
    .await
    .ok()?;
    Some(KeywordRun {
        features_digest: digest.get(),
        scores: scores.get(),
    })
}

// One future keeps MicroFlow's operators inlined, so their bounds checks fold away.
pub fn run_keyword() -> Option<KeywordRun> {
    drive_ready(run_keyword_async())?
}
