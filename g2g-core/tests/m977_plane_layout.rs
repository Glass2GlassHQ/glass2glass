//! M977: the `PlaneLayout` meta, which says where a raw frame's rows really sit
//! when they are padded instead of tightly packed.
//!
//! Two properties matter: it round-trips through a frame's meta set, and it
//! never survives an element that writes a new buffer (the runner's
//! `meta_transform` auto-propagation must drop it, or a consumer downstream of a
//! videoscale would read a tight buffer at a stale stride).
//!
//! Needs the graph runner (std/runtime) and the real `FrameMetaSet` (metadata).
#![cfg(all(feature = "std", feature = "metadata", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::meta::{BlobMeta, Plane, PlaneLayout, Transform, MAX_PLANES};
use g2g_core::runtime::{block_on, run_graph, GraphNodeRef, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, Graph, MemoryDomain, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

const WIDTH: u32 = 2;
const HEIGHT: u32 = 2;
/// Padded row pitch of the test frames: 8 tight bytes carried in 16.
const PADDED_STRIDE: usize = 16;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

fn frame() -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(Box::new(
            [0u8; PADDED_STRIDE * HEIGHT as usize],
        ))),
        FrameTiming::default(),
        0,
    )
}

/// Source emitting one padded frame that declares its layout, plus a `BlobMeta`
/// so a green run cannot be a set that lost everything.
struct PaddedSource;

impl SourceLoop for PaddedSource {
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
            let mut f = frame();
            f.meta.attach(PlaneLayout::single(PADDED_STRIDE));
            f.meta.attach(BlobMeta::new());
            out.push(PipelinePacket::DataFrame(f)).await?;
            out.push(PipelinePacket::Eos).await?;
            Ok(1)
        })
    }
}

/// Transform emitting a fresh (tightly-packed, meta-empty) frame and declaring
/// `decl` as what it does to metadata, so the runner applies the propagation
/// contract to the stashed input meta.
struct RewritingTransform {
    decl: Transform,
}

impl AsyncElement for RewritingTransform {
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
        Some(self.decl)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(input) => {
                    out.push(PipelinePacket::DataFrame(frame()))
                        .await
                        .map(|_| ())?;
                    drop(input);
                }
                other => out.push(other).await.map(|_| ())?,
            }
            Ok(())
        })
    }
}

/// What the sink saw on one frame: its layout, and whether the blob rode along.
type MetaRecord = (Option<PlaneLayout>, bool);

/// Records a [`MetaRecord`] per received frame.
struct RecordingSink {
    seen: Arc<Mutex<Vec<MetaRecord>>>,
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
        let seen = self.seen.clone();
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = &packet {
                seen.lock().unwrap().push((
                    f.meta.get::<PlaneLayout>().copied(),
                    f.meta.get::<BlobMeta>().is_some(),
                ));
            }
            Ok(())
        })
    }
}

/// source -> transform (declaring `decl`) -> sink.
fn run_through(decl: Transform) -> Vec<MetaRecord> {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut g: Graph<GraphNodeRef<'static>> = Graph::new();
    let src = g.add_source(GraphNodeRef::source(PaddedSource));
    let xform = g.add_transform(GraphNodeRef::element(RewritingTransform { decl }));
    let sink = g.add_sink(GraphNodeRef::element(RecordingSink { seen: seen.clone() }));
    g.link(src, xform).unwrap();
    g.link(xform, sink).unwrap();
    block_on(run_graph(g, &ZeroClock, 4)).expect("graph runs");
    let out = seen.lock().unwrap().clone();
    out
}

#[test]
fn attach_and_typed_get_round_trip() {
    let mut f = frame();
    f.meta.attach(PlaneLayout::single(PADDED_STRIDE));
    let got = f.meta.get::<PlaneLayout>().expect("layout attached");
    assert_eq!(got.count(), 1);
    assert_eq!(
        got.plane(0),
        Some(Plane {
            offset: 0,
            stride: PADDED_STRIDE
        })
    );
    assert_eq!(got.plane(1), None, "only the described planes exist");
}

#[test]
fn multi_plane_layout_addresses_each_plane() {
    // An NV12-shaped layout: luma at 0, the interleaved chroma plane after it,
    // both padded.
    let layout = PlaneLayout::new(&[
        Plane {
            offset: 0,
            stride: 64,
        },
        Plane {
            offset: 64 * 8,
            stride: 64,
        },
    ])
    .expect("two planes");
    assert_eq!(layout.count(), 2);
    assert_eq!(layout.row_range(0, 2, 48), Some(128..176));
    assert_eq!(layout.row_range(1, 1, 48), Some(576..624));
}

#[test]
fn a_layout_that_does_not_fit_is_rejected() {
    assert_eq!(
        PlaneLayout::new(&[]),
        None,
        "a frame has at least one plane"
    );
    let too_many = [Plane {
        offset: 0,
        stride: 4,
    }; MAX_PLANES + 1];
    assert_eq!(PlaneLayout::new(&too_many), None);
}

#[test]
fn row_arithmetic_is_checked() {
    // Offsets come off a producer, a driver, or a wire: a layout that cannot
    // describe the row asked for returns None instead of a bogus range.
    let layout = PlaneLayout::single(8);
    assert_eq!(layout.row_range(1, 0, 8), None, "no such plane");
    assert_eq!(layout.row_range(0, 0, 9), None, "row wider than the stride");
    assert_eq!(
        layout.row_range(0, usize::MAX, 8),
        None,
        "row * stride overflows"
    );
    let far = PlaneLayout::new(&[Plane {
        offset: usize::MAX - 4,
        stride: 8,
    }])
    .unwrap();
    assert_eq!(far.row_range(0, 0, 8), None, "offset + row bytes overflows");
}

#[test]
fn a_geometry_rewriting_transform_drops_it() {
    // A scale writes a new buffer, so the padded layout describes nothing there.
    let out = run_through(Transform::Scale);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].0, None, "layout dropped across the scale");
    assert!(
        out[0].1,
        "the blob still rode through, so the set was carried"
    );
}

#[test]
fn a_format_only_transform_drops_it_too() {
    // A convert declares `Copy` (its metadata passes through unchanged) and still
    // emits its own tightly-packed frame, so the layout must not ride along.
    let out = run_through(Transform::Copy);
    assert_eq!(out[0].0, None);
    assert!(out[0].1);
}
