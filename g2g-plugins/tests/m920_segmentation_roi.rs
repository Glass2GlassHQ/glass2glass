//! M920: segmentation and region-of-interest analytics nodes ride a real
//! transform chain. A source attaches an `AnalyticsMeta` holding a `Segmentation`
//! (normalized box plus an owned coverage mask) and a `Roi`; `VideoScale` resizes
//! the frame and declares `Transform::Scale`, so the runner re-attaches the
//! propagated graph to its fresh output; the sink reads both node kinds back.
//!
//! The scale is the point: the nodes are normalized, so they must come out
//! byte-identical at the new geometry, mask included.

#![cfg(all(feature = "std", feature = "metadata"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::meta::{AnalyticsMeta, AnalyticsNode, BBox, Mask, Roi, Segmentation};
use g2g_core::runtime::{run_graph, GraphNode, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, Graph, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::videoscale::VideoScale;

const SRC_W: u32 = 8;
const SRC_H: u32 = 8;
const DST_W: u32 = 4;
const DST_H: u32 = 4;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn rgba(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

fn bbox() -> BBox {
    BBox {
        x: 0.25,
        y: 0.125,
        w: 0.5,
        h: 0.25,
    }
}

/// A 3x2 coverage mask with a padded 4-byte stride.
fn mask() -> Mask {
    Mask::new(3, 2, 4, vec![10, 20, 30, 0, 40, 50, 60, 0]).expect("mask fits its data")
}

fn analytics() -> AnalyticsMeta {
    let mut a = AnalyticsMeta::new();
    a.push(AnalyticsNode::Segmentation(Segmentation {
        bbox: bbox(),
        label: 3,
        confidence: 0.8,
        mask: mask(),
    }));
    a.push(AnalyticsNode::Roi(Roi {
        bbox: bbox(),
        id: 42,
        label: 7,
    }));
    a
}

/// Emits one RGBA frame carrying the segmentation + ROI graph, then EOS.
struct AnalyticsSrc;

impl SourceLoop for AnalyticsSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(rgba(SRC_W, SRC_H)))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let mut f = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(
                    vec![128u8; (SRC_W * SRC_H * 4) as usize].into_boxed_slice(),
                )),
                FrameTiming::default(),
                0,
            );
            f.meta.attach(analytics());
            out.push(PipelinePacket::DataFrame(f)).await?;
            out.push(PipelinePacket::Eos).await?;
            Ok(1)
        })
    }
}

/// One frame as the sink saw it: its analytics graph and its byte size.
type SeenFrame = (Option<AnalyticsMeta>, usize);

/// Records the analytics graph and pixel geometry of every frame it receives.
struct RecSink {
    seen: Arc<Mutex<Vec<SeenFrame>>>,
}

impl AsyncElement for RecSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;
    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let seen = self.seen.clone();
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = &packet {
                let bytes = f.domain.as_system_slice().map(|s| s.len()).unwrap_or(0);
                seen.lock()
                    .unwrap()
                    .push((f.meta.get::<AnalyticsMeta>().cloned(), bytes));
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn segmentation_and_roi_survive_a_real_scale() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(AnalyticsSrc));
    let scale = g.add_transform(GraphNode::element(VideoScale::new(DST_W, DST_H)));
    let sink = g.add_sink(GraphNode::element(RecSink { seen: seen.clone() }));
    g.link(src, scale).unwrap();
    g.link(scale, sink).unwrap();

    run_graph(g, &NullClock, 4).await.expect("graph runs");

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1, "one frame reaches the sink");
    let (meta, bytes) = &seen[0];
    assert_eq!(
        *bytes,
        (DST_W * DST_H * 4) as usize,
        "the frame really was rescaled"
    );

    let meta = meta.as_ref().expect("the analytics graph rode the scale");
    let seg = meta.segmentations().next().expect("segmentation node");
    assert_eq!(seg.bbox, bbox(), "normalized coordinates are unchanged");
    assert_eq!(seg.label, 3);
    assert_eq!(seg.mask, mask(), "the mask bitmap arrived intact");
    // The mask keeps its own grid, not the frame's.
    assert_eq!((seg.mask.width(), seg.mask.height()), (3, 2));
    assert_eq!(seg.mask.sample(2, 1), Some(60));

    let roi = meta.rois().next().expect("roi node");
    assert_eq!((roi.id, roi.label), (42, 7));
    assert_eq!(roi.bbox, bbox());
}
