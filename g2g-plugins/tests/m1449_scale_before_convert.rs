#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;
use std::sync::{Arc, Mutex};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_graph, GraphNode, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    Graph, MemoryDomain, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::capsfilter::CapsFilter;
use g2g_plugins::videoconvert::VideoConvert;
use g2g_plugins::videoscale::VideoScale;

const DECODED_FORMAT: RawVideoFormat = RawVideoFormat::I420;
// what a decoder fixates to before it has parsed the stream geometry
const PLACEHOLDER_SIZE: (u32, u32) = (16, 16);
const DECODED_SIZE: (u32, u32) = (320, 180);
const PINNED_SIZE: (u32, u32) = (100, 60);
const FRAME_COUNT: u64 = 3;
const FRAMERATE_Q16: u32 = 30 << 16;
const LINK_CAPACITY: usize = 4;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn raw(format: RawVideoFormat, (width, height): (u32, u32)) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate: Rate::Fixed(FRAMERATE_Q16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn frame_bytes(format: RawVideoFormat, (width, height): (u32, u32)) -> usize {
    let bytes = format.unpadded_frame_bytes(width, height).unwrap();
    usize::try_from(bytes).unwrap()
}

// Negotiates on a placeholder geometry, then announces the real one before
// its first frame, like ffmpegdec.
struct DecoderLikeSource;

impl SourceLoop for DecoderLikeSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(raw(DECODED_FORMAT, PLACEHOLDER_SIZE)))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::CapsChanged(raw(
                DECODED_FORMAT,
                DECODED_SIZE,
            )))
            .await?;
            let size = frame_bytes(DECODED_FORMAT, DECODED_SIZE);
            for sequence in 0..FRAME_COUNT {
                out.push(PipelinePacket::DataFrame(Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(
                        vec![0u8; size].into_boxed_slice(),
                    )),
                    timing: FrameTiming::default(),
                    sequence,
                    meta: Default::default(),
                }))
                .await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(FRAME_COUNT)
        })
    }
}

#[derive(Debug, Default)]
struct Received {
    caps: Vec<Caps>,
    frame_sizes: Vec<usize>,
}

struct RecordingSink {
    received: Arc<Mutex<Received>>,
}

impl AsyncElement for RecordingSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            let mut received = self.received.lock().unwrap();
            match packet {
                PipelinePacket::CapsChanged(caps) => received.caps.push(caps),
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        panic!("expected a system frame");
                    };
                    received.frame_sizes.push(slice.as_slice().len());
                }
                _ => {}
            }
            Ok(())
        })
    }
}

async fn run_chain(transforms: Vec<GraphNode>, pin: &str) -> Received {
    let received = Arc::new(Mutex::new(Received::default()));
    let mut graph: Graph<GraphNode> = Graph::new();
    let mut previous = graph.add_source(GraphNode::source(DecoderLikeSource));
    let filter = CapsFilter::from_set(CapsSet::from_gst_string(pin).unwrap());
    for transform in transforms.into_iter().chain([GraphNode::element(filter)]) {
        let node = graph.add_transform(transform);
        graph.link(previous, node).unwrap();
        previous = node;
    }
    let sink = graph.add_sink(GraphNode::element(RecordingSink {
        received: Arc::clone(&received),
    }));
    graph.link(previous, sink).unwrap();
    run_graph(graph, &NullClock, LINK_CAPACITY)
        .await
        .unwrap_or_else(|e| panic!("{pin} chain should run: {e:?}"));
    Arc::into_inner(received).unwrap().into_inner().unwrap()
}

fn assert_scaled_to(received: &Received, format: RawVideoFormat) {
    assert_eq!(
        received.caps.last(),
        Some(&raw(format, PINNED_SIZE)),
        "the sink's last caps are the pinned geometry"
    );
    assert_eq!(
        received.frame_sizes,
        vec![frame_bytes(format, PINNED_SIZE); FRAME_COUNT as usize],
        "every frame arrives scaled to the pin"
    );
}

#[tokio::test]
async fn scale_then_convert_scales_to_the_pin_after_a_caps_change() {
    let received = run_chain(
        vec![
            GraphNode::element(VideoScale::new(0, 0)),
            GraphNode::element(VideoConvert::auto()),
        ],
        "video/x-raw,format=BGRA,width=100,height=60",
    )
    .await;
    assert_scaled_to(&received, RawVideoFormat::Bgra8);
}

#[tokio::test]
async fn convert_then_scale_scales_to_the_pin_after_a_caps_change() {
    let received = run_chain(
        vec![
            GraphNode::element(VideoConvert::auto()),
            GraphNode::element(VideoScale::new(0, 0)),
        ],
        "video/x-raw,format=BGRA,width=100,height=60",
    )
    .await;
    assert_scaled_to(&received, RawVideoFormat::Bgra8);
}

#[tokio::test]
async fn scale_alone_scales_to_the_pin_after_a_caps_change() {
    let received = run_chain(
        vec![GraphNode::element(VideoScale::new(0, 0))],
        "video/x-raw,width=100,height=60",
    )
    .await;
    assert_scaled_to(&received, DECODED_FORMAT);
}
