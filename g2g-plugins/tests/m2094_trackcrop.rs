// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(feature = "std", feature = "analytics"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::meta::{AnalyticsMeta, AnalyticsNode, BBox, Roi};
use g2g_core::runtime::{run_graph, GraphNode, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, Graph, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::trackcrop::TrackCrop;

const INPUT_WIDTH: u32 = 192;
const INPUT_HEIGHT: u32 = 108;
const OUTPUT_WIDTH: u32 = 54;
const OUTPUT_HEIGHT: u32 = 96;
const SQUARE_SIDE: u32 = 8;
const FIRST_SQUARE_X: u32 = 10;
const SQUARE_STEP: u32 = 2;
const FRAMES: u32 = 80;
const DETECTION_STRIDE: u32 = 3;
const FRAME_PERIOD_NS: u64 = 40_000_000;
const SQUARE_REGION_ID: u32 = 1;
const BYTES_PER_PIXEL: usize = 4;
const LIT: u8 = 128;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn rgba() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(INPUT_WIDTH),
        height: Dim::Fixed(INPUT_HEIGHT),
        framerate: Rate::Fixed(25 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
    }
}

fn square_x(index: u32) -> u32 {
    FIRST_SQUARE_X + index * SQUARE_STEP
}

fn square_y() -> u32 {
    (INPUT_HEIGHT - SQUARE_SIDE) / 2
}

fn picture_with_square(index: u32) -> Vec<u8> {
    let mut pixels = vec![0u8; (INPUT_WIDTH * INPUT_HEIGHT) as usize * BYTES_PER_PIXEL];
    for y in square_y()..square_y() + SQUARE_SIDE {
        for x in square_x(index)..square_x(index) + SQUARE_SIDE {
            let at = (y * INPUT_WIDTH + x) as usize * BYTES_PER_PIXEL;
            pixels[at..at + BYTES_PER_PIXEL].fill(u8::MAX);
        }
    }
    pixels
}

fn square_region(index: u32) -> AnalyticsMeta {
    let mut meta = AnalyticsMeta::new();
    meta.push(AnalyticsNode::Roi(Roi {
        bbox: BBox {
            x: square_x(index) as f32 / INPUT_WIDTH as f32,
            y: square_y() as f32 / INPUT_HEIGHT as f32,
            w: SQUARE_SIDE as f32 / INPUT_WIDTH as f32,
            h: SQUARE_SIDE as f32 / INPUT_HEIGHT as f32,
        },
        id: SQUARE_REGION_ID,
        label: 0,
    }));
    meta
}

struct MovingSquareSrc;

impl SourceLoop for MovingSquareSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(rgba()))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for index in 0..FRAMES {
                let mut frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(
                        picture_with_square(index).into_boxed_slice(),
                    )),
                    FrameTiming {
                        pts_ns: u64::from(index) * FRAME_PERIOD_NS,
                        duration_ns: FRAME_PERIOD_NS,
                        ..FrameTiming::default()
                    },
                    u64::from(index),
                );
                if index.is_multiple_of(DETECTION_STRIDE) {
                    frame.meta.attach(square_region(index));
                }
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(u64::from(FRAMES))
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
struct SeenFrame {
    caps: Option<Caps>,
    pixels: Vec<u8>,
}

struct RecordingSink {
    caps: Option<Caps>,
    seen: Arc<Mutex<Vec<SeenFrame>>>,
}

impl AsyncElement for RecordingSink {
    type ProcessFuture<'a>
        = core::future::Ready<Result<(), G2gError>>
    where
        Self: 'a;
    fn intercept_caps(&self, caps: &Caps) -> Result<Caps, G2gError> {
        Ok(caps.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.caps = Some(caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        match packet {
            PipelinePacket::CapsChanged(caps) => self.caps = Some(caps),
            PipelinePacket::DataFrame(frame) => {
                let pixels = frame.domain.as_system_slice().unwrap().to_vec();
                self.seen.lock().unwrap().push(SeenFrame {
                    caps: self.caps.clone(),
                    pixels,
                });
            }
            _ => {}
        }
        core::future::ready(Ok(()))
    }
}

// columns of the output picture with a lit pixel
fn lit_columns(pixels: &[u8]) -> (usize, usize) {
    let columns: Vec<usize> = pixels
        .as_chunks::<BYTES_PER_PIXEL>()
        .0
        .iter()
        .enumerate()
        .filter(|(_, pixel)| pixel[0] >= LIT)
        .map(|(index, _)| index % OUTPUT_WIDTH as usize)
        .collect();
    (
        *columns.iter().min().expect("the square is in the picture"),
        *columns.iter().max().expect("the square is in the picture"),
    )
}

#[tokio::test]
async fn the_crop_follows_a_square_that_crosses_the_frame() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(MovingSquareSrc));
    let crop = graph.add_transform(GraphNode::element(TrackCrop::new(
        OUTPUT_WIDTH,
        OUTPUT_HEIGHT,
    )));
    let sink = graph.add_sink(GraphNode::element(RecordingSink {
        caps: None,
        seen: seen.clone(),
    }));
    graph.link(source, crop).unwrap();
    graph.link(crop, sink).unwrap();

    run_graph(graph, &NullClock, 4).await.expect("graph runs");

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), FRAMES as usize, "every frame arrives");
    let travel = square_x(FRAMES - 1) - square_x(0);
    assert!(
        travel > INPUT_HEIGHT * OUTPUT_WIDTH / OUTPUT_HEIGHT,
        "the square travels farther than the crop is wide"
    );
    for (index, frame) in seen.iter().enumerate() {
        let Some(Caps::RawVideo { width, height, .. }) = &frame.caps else {
            panic!("frame {index} arrives under raw video caps");
        };
        assert_eq!(
            (width, height),
            (&Dim::Fixed(OUTPUT_WIDTH), &Dim::Fixed(OUTPUT_HEIGHT))
        );
        assert_eq!(
            frame.pixels.len(),
            (OUTPUT_WIDTH * OUTPUT_HEIGHT) as usize * BYTES_PER_PIXEL
        );
        let (left, right) = lit_columns(&frame.pixels);
        assert!(
            left > 0 && right < OUTPUT_WIDTH as usize - 1,
            "frame {index}: the whole square is in the crop, columns {left}..={right}"
        );
    }
}
