// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::mem::MaybeUninit;
use core::ptr::addr_of;

use esp_hal::time::Instant;
use esp_println::println;
use g2g_core::error::G2gError;
use g2g_core::frame::Frame;
use g2g_core::st2110video::Sampling;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{
    run_source_transform_sink, Chain, StaticSink, StaticTransform, TensorDType, TensorLayout,
    TensorShape,
};
use g2g_mcu::{
    CTensorModel, FrameGrabber, GrabberSrc, Int8Quantize, PacketSender, PixelMapping,
    RawVideoRtpSend, TensorFormat, TensorInference,
};

use crate::network::RTP_PACKET_BYTES;
use crate::tflm;

pub const SIDE: usize = 96;
pub const IMAGE_BYTES: usize = SIDE * SIDE;
const CLASSES: usize = 2;
const PERSON: usize = 1;
pub const ARENA_BYTES: usize = 128 * 1024;

const IMAGE_FORMAT: TensorFormat = TensorFormat {
    dtype: TensorDType::U8,
    shape: TensorShape::new([1, SIDE as u32, SIDE as u32, 1]),
    layout: TensorLayout::Nhwc,
};
const INPUT_FORMAT: TensorFormat = TensorFormat {
    dtype: TensorDType::I8,
    ..IMAGE_FORMAT
};
const OUTPUT_FORMAT: TensorFormat = TensorFormat {
    dtype: TensorDType::I8,
    shape: TensorShape::new([1, CLASSES as u32]),
    layout: TensorLayout::Nhwc,
};
// person_detect.tflite's input quantization.
const INPUT_SCALE: f32 = 1.0 / 127.5;
const INPUT_ZERO_POINT: i32 = -1;
// person_detect.tflite's output quantization: probability = (q + 128) / 256.
const OUTPUT_ZERO_POINT: i32 = -128;
const OUTPUT_STEPS: f32 = 256.0;

const SAMPLE_COUNT: usize = 10;
macro_rules! sample {
    ($index:literal) => {
        include_bytes!(concat!(
            env!("ESP_TFLM_DIR"),
            "/examples/person_detection/static_images/sample_images/image",
            $index
        ))
    };
}
static SAMPLES: [&[u8; IMAGE_BYTES]; SAMPLE_COUNT] = [
    sample!("0"),
    sample!("1"),
    sample!("2"),
    sample!("3"),
    sample!("4"),
    sample!("5"),
    sample!("6"),
    sample!("7"),
    sample!("8"),
    sample!("9"),
];
// tools/esp-tflm-reference.sh, TFLite Micro's reference kernels on the host.
const SAMPLE_SCORES: [[i8; CLASSES]; SAMPLE_COUNT] = [
    [-115, 115],
    [39, -39],
    [-115, 115],
    [-67, 67],
    [-111, 111],
    [100, -100],
    [-97, 97],
    [98, -98],
    [-114, 114],
    [-83, 83],
];

extern "C" {
    static g_person_detect_model_data: u8;
    static g_person_detect_model_data_len: i32;
}

pub fn load_model(arena: &'static mut [MaybeUninit<u8>]) -> Result<CTensorModel, G2gError> {
    // SAFETY: person_detect_model_data.cc defines both symbols, an 8-byte aligned array and its length.
    let model = unsafe {
        core::slice::from_raw_parts(
            addr_of!(g_person_detect_model_data),
            g_person_detect_model_data_len as usize,
        )
    };
    tflm::load("person_detect", model, arena, INPUT_FORMAT, OUTPUT_FORMAT)
}

pub fn person_probability(scores: [i8; CLASSES]) -> f32 {
    (scores[PERSON] as i32 - OUTPUT_ZERO_POINT) as f32 / OUTPUT_STEPS
}

pub struct ScoreSink<F: FnMut([i8; CLASSES])> {
    pub on_scores: F,
}

impl<F: FnMut([i8; CLASSES])> StaticSink for ScoreSink<F> {
    async fn consume(&mut self, frame: Frame) -> Result<(), G2gError> {
        let Some(&[no_person, person]) = frame.domain.as_system_slice() else {
            return Err(G2gError::CapsMismatch);
        };
        (self.on_scores)([no_person as i8, person as i8]);
        Ok(())
    }
}

pub async fn run_person_graph(
    grabber: impl FrameGrabber,
    image_limit: Option<u32>,
    image_interval_ns: u64,
    model: &mut CTensorModel,
    front: impl StaticTransform,
    sink: impl StaticSink,
) -> Result<(), G2gError> {
    let image_ring: StaticLendRing<1, IMAGE_BYTES> = StaticLendRing::new();
    let input_ring: StaticLendRing<1, IMAGE_BYTES> = StaticLendRing::new();
    let score_ring: StaticLendRing<1, CLASSES> = StaticLendRing::new();
    // SAFETY (every `with_ring` below): the rings are locals of this function
    // and outlive the pipeline run, which drains before returning.
    let quantize = unsafe {
        Int8Quantize::with_ring(IMAGE_FORMAT, INPUT_SCALE, INPUT_ZERO_POINT, &input_ring)
    }?
    .with_pixel_mapping(PixelMapping::Int8Range)?;
    // SAFETY: as above.
    let inference = unsafe { TensorInference::with_ring(model, &score_ring) };
    if quantize.output_format() != inference.input_format() {
        return Err(G2gError::CapsMismatch);
    }
    // SAFETY: as above.
    let source = unsafe { GrabberSrc::with_ring(grabber, &image_ring, image_interval_ns) };
    let source = match image_limit {
        Some(images) => source.with_frame_limit(images),
        None => source,
    };
    run_source_transform_sink(source, Chain(front, Chain(quantize, inference)), sink).await
}

struct Unchanged;

impl StaticTransform for Unchanged {
    async fn process(&mut self, input: Frame) -> Result<Option<Frame>, G2gError> {
        Ok(Some(input))
    }
}

struct SampleImages {
    next: usize,
}

impl FrameGrabber for SampleImages {
    async fn capture(&mut self, buf: &mut [u8]) -> Result<usize, G2gError> {
        let (Some(image), Some(slot)) = (SAMPLES.get(self.next), buf.get_mut(..IMAGE_BYTES)) else {
            return Err(G2gError::CapsMismatch);
        };
        slot.copy_from_slice(image.as_slice());
        self.next += 1;
        Ok(IMAGE_BYTES)
    }
}

pub fn check_samples(model: &mut CTensorModel) -> bool {
    let mut matches = 0;
    let mut index = 0;
    let start = Instant::now();
    let sink = ScoreSink {
        on_scores: |scores: [i8; CLASSES]| {
            if Some(&scores) == SAMPLE_SCORES.get(index) {
                matches += 1;
            } else {
                println!("MISMATCH person image{index}: {scores:?}");
            }
            index += 1;
        },
    };
    let result = g2g_core::drive_ready(run_person_graph(
        SampleImages { next: 0 },
        Some(SAMPLE_COUNT as u32),
        1,
        model,
        Unchanged,
        sink,
    ));
    let micros_per_image = start.elapsed().as_micros() / SAMPLE_COUNT as u64;
    let passed = matches!(result, Some(Ok(()))) && matches == SAMPLE_COUNT;
    let verdict = if passed { "OK" } else { "MISMATCH" };
    println!("{verdict} person: {matches} of {SAMPLE_COUNT} sample images equal the reference kernels, {micros_per_image} us per image");
    passed
}

// the frame period measured through person_detect on this board
const CAMERA_FRAME_INTERVAL_NS: u64 = 64_000_000;

const RTP_PAYLOAD_TYPE: u8 = 96;
const RTP_SSRC: u32 = 0x6732_6732;

pub async fn watch(
    model: &mut CTensorModel,
    camera: impl FrameGrabber,
    sender: impl PacketSender,
) -> Result<(), G2gError> {
    let stream = RawVideoRtpSend::<_, RTP_PACKET_BYTES>::new(
        sender,
        Sampling::Gray8,
        SIDE,
        SIDE,
        RTP_PAYLOAD_TYPE,
        RTP_SSRC,
    )
    .ok_or(G2gError::CapsMismatch)?;
    let mut last = Instant::now();
    let sink = ScoreSink {
        on_scores: |scores: [i8; CLASSES]| {
            let now = Instant::now();
            let probability = person_probability(scores);
            let label = if probability > 0.5 {
                "person"
            } else {
                "nobody"
            };
            println!(
                "{label:<7} {probability:.2} {} ms",
                (now - last).as_millis()
            );
            last = now;
        },
    };
    run_person_graph(camera, None, CAMERA_FRAME_INTERVAL_NS, model, stream, sink).await
}
