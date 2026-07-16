//! M356: the native `g2g.MetaSink` is a `Sync` pyclass (Mutex-backed), not an
//! `unsendable` `RefCell`, so a hosted element may stage analytics from a thread
//! other than the one the host created the sink on. This is the precondition for
//! the "free-threaded (PEP 703) with no code change" claim in `host.rs`: a
//! parallel-post-processing element reaches the sink off-thread.
//!
//! The fixture `ThreadedTransform` calls `meta.add_object(...)` from a spawned
//! `threading.Thread`. With the old `unsendable` sink that trips pyo3's
//! thread-affinity check (the call raises, the host fails the frame); with the
//! `Sync` sink it succeeds and the detection lands in the frame metadata. The
//! test therefore fails if the sink regresses to thread-affine.
#![cfg(feature = "analytics")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::memory::SystemSlice;
use g2g_core::{
    AnalyticsMeta, AsyncElement, Caps, Dim, Frame, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, PushOutcome, Rate, RawVideoFormat,
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
        timing: FrameTiming {
            pts_ns: 0,
            dts_ns: 0,
            duration_ns: 0,
            capture_ns: 0,
            arrival_ns: 0,
            keyframe: false,
        },
        sequence: 0,
        meta: Default::default(),
    }
}

#[test]
fn metasink_accepts_cross_thread_staging() {
    std::env::set_var("PYTHONPATH", concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"));

    let mut el = PyTransform::new("echo_element", "ThreadedTransform");
    let caps = Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(1),
        framerate: Rate::Fixed(30),
    };
    el.configure_pipeline(&caps).unwrap();

    let mut sink = CollectSink::default();
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    // An unsendable sink makes the fixture's off-thread add_object raise, which
    // the host surfaces as a failed frame; a Sync sink lets process() succeed.
    rt.block_on(el.process(PipelinePacket::DataFrame(frame_2x1_rgba()), &mut sink))
        .expect("off-thread add_object must succeed with a Sync MetaSink");

    let PipelinePacket::DataFrame(frame) = &sink.packets[0] else {
        panic!("expected a DataFrame downstream");
    };
    let analytics = frame
        .meta
        .get::<AnalyticsMeta>()
        .expect("the off-thread add_object should have attached an AnalyticsMeta");
    let dets: Vec<_> = analytics.detections().collect();
    assert_eq!(dets.len(), 1, "exactly one detection staged from the worker thread");
    assert_eq!(dets[0].label, 11);
}
