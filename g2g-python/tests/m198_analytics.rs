//! M198 step 3: analytics metadata routing (`analytics` feature).
//!
//! Drives the fixture element, which calls `meta.add_object(...)` through the
//! native `g2g.MetaSink`, and asserts the host materialized it into the frame's
//! typed `AnalyticsMeta`. Needs libpython + the `metadata`-enabled core, so the
//! whole file compiles away without the feature.
#![cfg(feature = "analytics")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::memory::SystemSlice;
use g2g_core::{
    AnalyticsMeta, AsyncElement, BlobMeta, Caps, Dim, Frame, FrameTiming, G2gError, MemoryDomain,
    OutputSink, PipelinePacket, PushOutcome, Rate, RawVideoFormat,
};
use g2g_python::PyTransform;

#[derive(Default)]
struct CollectSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for CollectSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        self.packets.push(packet);
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

fn frame_2x1_rgba() -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(vec![0u8; 8].into_boxed_slice())),
        timing: FrameTiming { pts_ns: 0, dts_ns: 0, duration_ns: 0, capture_ns: 0, arrival_ns: 0 , keyframe: false},
        sequence: 0,
        meta: Default::default(),
    }
}

#[test]
fn detection_from_python_lands_in_frame_metadata() {
    std::env::set_var(
        "PYTHONPATH",
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"),
    );

    let mut el = PyTransform::new("echo_element", "EchoTransform");
    let caps = Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(1),
        framerate: Rate::Fixed(30),
    };
    el.configure_pipeline(&caps).unwrap();

    let mut sink = CollectSink::default();
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    rt.block_on(el.process(PipelinePacket::DataFrame(frame_2x1_rgba()), &mut sink))
        .unwrap();

    let PipelinePacket::DataFrame(frame) = &sink.packets[0] else {
        panic!("expected a DataFrame downstream");
    };
    let analytics = frame
        .meta
        .get::<AnalyticsMeta>()
        .expect("Python add_object should have attached an AnalyticsMeta");
    let dets: Vec<_> = analytics.detections().collect();
    assert_eq!(dets.len(), 1, "exactly one detection attached");
    assert_eq!(dets[0].label, 7);
    assert_eq!(dets[0].bbox.x, 1.0);
    assert_eq!(dets[0].bbox.h, 4.0);
    assert!((dets[0].confidence - 0.9).abs() < 1e-6);

    // The opaque blob (FrameIO.append_blob mirror) rode along as a BlobMeta.
    let blobs = frame.meta.get::<BlobMeta>().expect("add_blob should attach a BlobMeta");
    assert_eq!(blobs.len(), 1);
    let blob = blobs.iter().next().unwrap();
    assert_eq!(blob.header, "embedding");
    assert_eq!(blob.payload, vec![1, 2, 3, 4]);
}
