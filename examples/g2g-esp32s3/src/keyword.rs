// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::cell::Cell;
use core::mem::MaybeUninit;

use esp_hal::dma::DmaRxStreamBuf;
use esp_hal::dma_rx_stream_buffer;
use esp_hal::i2s::master::{I2s, I2sRxDmaTransfer, PdmConfig, PdmRxConfig, PdmSlotMode};
use esp_hal::peripherals::{DMA_CH0, GPIO41, GPIO42, I2S0};
use esp_hal::time::Rate;
use esp_hal::Blocking;
use esp_println::println;
use g2g_core::error::G2gError;
use g2g_core::frame::Frame;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{StaticSink, TensorDType, TensorLayout, TensorShape};
use g2g_mcu::{CTensorModel, FrameGrabber, TensorFormat, TensorInference, TensorModel};
use noalloc_pipeline::keyword::{run_keyword_graph, CLIP_BYTES, FEATURES, FEATURE_FORMAT};

use crate::peripheral;
use crate::tflm::{self, Aligned};

const KEYWORDS: usize = 4;
// speech.tflite's output order.
const LABELS: [&str; KEYWORDS] = ["silence", "unknown", "yes", "no"];
const OUTPUT_FORMAT: TensorFormat = TensorFormat {
    dtype: TensorDType::I8,
    shape: TensorShape::new([1, KEYWORDS as u32]),
    layout: TensorLayout::Nhwc,
};
// speech.tflite's output quantization: probability = (q + 128) / 256.
const OUTPUT_ZERO_POINT: i32 = -128;
const OUTPUT_STEPS: f32 = 256.0;
pub const ARENA_BYTES: usize = 16 * 1024;

static MODEL: &Aligned<[u8]> = &Aligned(*include_bytes!(
    "../../../g2g-mcu/tests/fixtures/keyword/speech.tflite"
));
const FEATURES_FIXTURE: &[u8; FEATURES] =
    include_bytes!("../../../g2g-mcu/tests/fixtures/keyword/features.i8");
// the TFLite 2.15 interpreter's output for features.i8
const TFLITE_SCORES: &[u8; KEYWORDS] =
    include_bytes!("../../../g2g-mcu/tests/fixtures/keyword/tflite_scores.i8");

const SAMPLE_RATE_HZ: u32 = 16_000;
const SAMPLE_BYTES: usize = 2;
const STRIDE_MILLISECONDS: usize = 500;
const STRIDE_BYTES: usize = SAMPLE_RATE_HZ as usize * STRIDE_MILLISECONDS / 1000 * SAMPLE_BYTES;
const STRIDE_NS: u64 = STRIDE_MILLISECONDS as u64 * 1_000_000;
// esp-hal DMA descriptors carry at most 4095 bytes.
const DMA_CHUNK_BYTES: usize = 4000;

const DMA_BYTES: usize = CLIP_BYTES;

pub fn load_model(arena: &'static mut [MaybeUninit<u8>]) -> Result<CTensorModel, G2gError> {
    tflm::load("speech", &MODEL.0, arena, FEATURE_FORMAT, OUTPUT_FORMAT)
}

pub fn check_model(model: &mut CTensorModel) -> bool {
    let mut scores = [0u8; KEYWORDS];
    let result = g2g_core::drive_ready(model.run(FEATURES_FIXTURE, &mut scores));
    let passed = matches!(result, Some(Ok(()))) && &scores == TFLITE_SCORES;
    let verdict = if passed { "OK" } else { "MISMATCH" };
    println!("{verdict} speech: {scores:?} (TFLite interpreter {TFLITE_SCORES:?})");
    passed
}

struct MicWindow<'d, 'a> {
    transfer: I2sRxDmaTransfer<'d, Blocking, DmaRxStreamBuf>,
    window: &'static mut [u8; CLIP_BYTES],
    filled: bool,
    peak: &'a Cell<i16>,
}

impl MicWindow<'_, '_> {
    async fn read_exact(&mut self, from: usize) -> Result<(), G2gError> {
        let mut at = from;
        while at < CLIP_BYTES {
            // esp-hal stops the DMA when every descriptor is full and unread.
            if self.transfer.is_done() {
                return Err(peripheral());
            }
            let popped = self.transfer.pop(&mut self.window[at..]);
            if popped == 0 {
                embassy_futures::yield_now().await;
            }
            at += popped;
        }
        Ok(())
    }
}

impl FrameGrabber for MicWindow<'_, '_> {
    async fn capture(&mut self, buf: &mut [u8]) -> Result<usize, G2gError> {
        if self.filled {
            self.window.copy_within(STRIDE_BYTES.., 0);
            self.read_exact(CLIP_BYTES - STRIDE_BYTES).await?;
        } else {
            self.read_exact(0).await?;
            self.filled = true;
        }
        let Some(slot) = buf.get_mut(..CLIP_BYTES) else {
            return Err(G2gError::CapsMismatch);
        };
        slot.copy_from_slice(self.window.as_slice());
        let peak = self
            .window
            .chunks_exact(SAMPLE_BYTES)
            .map(|sample| i16::from_le_bytes([sample[0], sample[1]]).saturating_abs())
            .max()
            .unwrap_or(0);
        self.peak.set(peak);
        Ok(CLIP_BYTES)
    }
}

struct ScorePrinter<'a> {
    peak: &'a Cell<i16>,
}

impl StaticSink for ScorePrinter<'_> {
    async fn consume(&mut self, frame: Frame) -> Result<(), G2gError> {
        let Some(steps) = frame.domain.as_system_slice() else {
            return Err(G2gError::CapsMismatch);
        };
        let mut scores = [0.0f32; KEYWORDS];
        for (score, &step) in scores.iter_mut().zip(steps) {
            *score = (step as i8 as i32 - OUTPUT_ZERO_POINT) as f32 / OUTPUT_STEPS;
        }
        let mut best = 0;
        for index in 1..KEYWORDS {
            if scores[index] > scores[best] {
                best = index;
            }
        }
        println!(
            "{:<8} silence {:.2} unknown {:.2} yes {:.2} no {:.2} peak {}",
            LABELS[best],
            scores[0],
            scores[1],
            scores[2],
            scores[3],
            self.peak.get()
        );
        Ok(())
    }
}

pub struct Microphone {
    pub i2s0: I2S0<'static>,
    pub dma: DMA_CH0<'static>,
    pub clock: GPIO42<'static>,
    pub data: GPIO41<'static>,
    pub window: &'static mut [u8; CLIP_BYTES],
}

pub async fn listen(model: &mut CTensorModel, microphone: Microphone) -> Result<(), G2gError> {
    let pdm = PdmConfig::rx_only(PdmRxConfig::new_pcm_default(
        Rate::from_hz(SAMPLE_RATE_HZ),
        PdmSlotMode::Mono,
    ));
    let i2s = I2s::new_pdm(microphone.i2s0, microphone.dma, pdm).map_err(|_| peripheral())?;
    let i2s_rx = i2s
        .i2s_rx
        .with_clk(microphone.clock)
        .with_din(microphone.data)
        .build();
    let transfer = i2s_rx
        .read(dma_rx_stream_buffer!(DMA_BYTES, DMA_CHUNK_BYTES))
        .map_err(|_| peripheral())?;

    let peak = Cell::new(0);
    let window = MicWindow {
        transfer,
        window: microphone.window,
        filled: false,
        peak: &peak,
    };
    let score_ring: StaticLendRing<1, KEYWORDS> = StaticLendRing::new();
    // SAFETY: the ring is a local of this function and outlives the graph run.
    let inference = unsafe { TensorInference::with_ring(model, &score_ring) };
    println!("listening: say yes or no");
    run_keyword_graph(
        window,
        None,
        STRIDE_NS,
        inference,
        ScorePrinter { peak: &peak },
    )
    .await
}
