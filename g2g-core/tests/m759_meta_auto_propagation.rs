//! M759: framework auto-application of the FrameMeta propagation contract. A
//! transform that declares a [`meta_transform`] gets its input frame's metadata
//! propagated and re-attached to its fresh (meta-empty) outputs by the runner,
//! so per-frame metadata survives a linear transform, not just a tee.
//!
//! The mock transform always emits a brand-new frame with an empty meta set, so
//! a green run proves the runner (not the element) carried the metadata across:
//! without auto-application the sink would see nothing.
//!
//! Needs the graph runner (std/runtime) and the real `FrameMetaSet` (metadata).
#![cfg(all(feature = "std", feature = "metadata", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::meta::{AnalyticsMeta, BBox, BlobMeta, ObjectDetection, Transform};
use g2g_core::runtime::{block_on, run_graph, GraphNodeRef, SourceLoop};
use g2g_core::{
    graph::Graph, AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, MemoryDomain, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

/// Per-frame snapshot the sink records: `(analytics label, has_blob)`.
type MetaRecord = (Option<u32>, bool);

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(2),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn frame(seq: u64) -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 2 * 2 * 4]))),
        FrameTiming::default(),
        seq,
    )
}

fn detection(label: u32) -> ObjectDetection {
    ObjectDetection {
        bbox: BBox {
            x: 0.1,
            y: 0.1,
            w: 0.2,
            h: 0.2,
        },
        label,
        confidence: 0.9,
    }
}

/// Source emitting one frame carrying an `AnalyticsMeta` (label 7) and a
/// `BlobMeta`, then EOS.
struct MetaSource;

impl SourceLoop for MetaSource {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(caps()))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let mut f = frame(0);
            let mut a = AnalyticsMeta::new();
            a.add_detection(detection(7));
            f.meta.attach(a);
            let mut b = BlobMeta::new();
            b.push("embed", std::vec![1u8, 2, 3]);
            f.meta.attach(b);
            out.push(PipelinePacket::DataFrame(f)).await?;
            out.push(PipelinePacket::Eos).await?;
            Ok(1)
        })
    }
}

/// A 1-in-1-out transform that always emits a *fresh* frame (empty meta),
/// declaring `decl` as its metadata transform. If `own_label` is set it attaches
/// its own `AnalyticsMeta` to the output, exercising the "never overwrite
/// element-authored meta" rule.
struct FreshTransform {
    decl: Option<Transform>,
    own_label: Option<u32>,
}

impl AsyncElement for FreshTransform {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn meta_transform(&self) -> Option<Transform> {
        self.decl
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let own = self.own_label;
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(input) => {
                    let mut fresh = frame(input.sequence);
                    if let Some(l) = own {
                        let mut a = AnalyticsMeta::new();
                        a.add_detection(detection(l));
                        fresh.meta.attach(a);
                    }
                    out.push(PipelinePacket::DataFrame(fresh)).await?;
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

/// Records `(analytics label, has_blob)` for each `DataFrame` it receives.
struct RecordingSink {
    records: Arc<Mutex<Vec<MetaRecord>>>,
}

impl AsyncElement for RecordingSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let records = self.records.clone();
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = &packet {
                let label = f
                    .meta
                    .get::<AnalyticsMeta>()
                    .and_then(|a| a.detections().next().map(|d| d.label));
                let has_blob = f.meta.get::<BlobMeta>().is_some();
                records.lock().unwrap().push((label, has_blob));
            }
            Ok(())
        })
    }
}

/// Build source -> transform -> sink and run to EOS, returning the sink's
/// recorded per-frame meta snapshot.
fn run_line(decl: Option<Transform>, own_label: Option<u32>) -> Vec<MetaRecord> {
    let records = Arc::new(Mutex::new(Vec::new()));
    let mut g: Graph<GraphNodeRef<'static>> = Graph::new();
    let src = g.add_source(GraphNodeRef::source(MetaSource));
    let xform = g.add_transform(GraphNodeRef::element(FreshTransform { decl, own_label }));
    let sink = g.add_sink(GraphNodeRef::element(RecordingSink {
        records: records.clone(),
    }));
    g.link(src, xform).unwrap();
    g.link(xform, sink).unwrap();
    block_on(run_graph(g, &ZeroClock, 4)).expect("graph runs");
    let out = records.lock().unwrap().clone();
    out
}

#[test]
fn scale_propagates_analytics_and_blob_onto_fresh_output() {
    // A Scale keeps normalized analytics and the opaque blob; the runner
    // re-attaches both to the transform's fresh (meta-empty) output.
    let out = run_line(Some(Transform::Scale), None);
    assert_eq!(out, std::vec![(Some(7), true)]);
}

#[test]
fn encode_drops_analytics_but_keeps_blob() {
    // A re-encode drops pixel-derived analytics; the opaque BlobMeta rides
    // through (its propagate() keeps all).
    let out = run_line(Some(Transform::Encode), None);
    assert_eq!(out, std::vec![(None, true)]);
}

#[test]
fn element_authored_meta_is_not_clobbered() {
    // The transform attaches its OWN analytics (label 99); the output meta is
    // non-empty, so the runner does not overwrite it (and does not add the
    // stashed blob either, since it only fills a fully-empty set).
    let out = run_line(Some(Transform::Copy), Some(99));
    assert_eq!(out, std::vec![(Some(99), false)]);
}

#[test]
fn none_declaration_leaves_output_meta_empty() {
    // No opt-in: the runner does nothing, so the fresh output stays meta-empty.
    let out = run_line(None, None);
    assert_eq!(out, std::vec![(None, false)]);
}
