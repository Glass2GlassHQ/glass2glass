//! M198 step 4b: `PyAggregator`, a batched Python element hosted as a g2g muxer.
//!
//! Negotiation is always testable; the partial-round collection runs on the
//! default (no-interpreter) build; the live batch + metadata path runs under
//! `analytics`.

use g2g_core::{Caps, Dim, MultiInputElement, Rate, RawVideoFormat};
use g2g_python::PyAggregator;

// The sink and its imports are only used by the collection / batch tests
// (default + analytics builds), not by the python-only negotiation test.
#[cfg(any(not(feature = "python"), feature = "analytics"))]
use core::future::Future;
#[cfg(any(not(feature = "python"), feature = "analytics"))]
use core::pin::Pin;
#[cfg(any(not(feature = "python"), feature = "analytics"))]
use g2g_core::{G2gError, OutputSink, PipelinePacket, PushOutcome};

#[cfg(any(not(feature = "python"), feature = "analytics"))]
#[derive(Default)]
struct CollectSink {
    packets: Vec<PipelinePacket>,
}

#[cfg(any(not(feature = "python"), feature = "analytics"))]
impl OutputSink for CollectSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        self.packets.push(packet);
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

fn rgba_2x1() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(1),
        framerate: Rate::Fixed(30),
    }
}

#[test]
fn negotiates_each_input_against_accept() {
    let agg = PyAggregator::new("echo_element", "EchoTransform", 2);
    assert_eq!(agg.input_count(), 2);
    let caps = rgba_2x1();
    assert_eq!(agg.intercept_caps(0, &caps).unwrap(), caps);
    assert_eq!(agg.intercept_caps(1, &caps).unwrap(), caps);
}

// Collection without an interpreter: an incomplete round emits nothing. (Under
// `python` configure spawns a worker, so this is the no-interpreter build only.)
#[cfg(not(feature = "python"))]
#[test]
fn incomplete_round_emits_nothing() {
    use g2g_core::memory::SystemSlice;
    use g2g_core::{Frame, FrameTiming, MemoryDomain};

    let mut agg = PyAggregator::new("echo_element", "EchoTransform", 2);
    agg.configure_pipeline(0, &rgba_2x1()).unwrap();
    agg.configure_pipeline(1, &rgba_2x1()).unwrap();

    let frame = Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(vec![1u8; 8].into_boxed_slice())),
        timing: FrameTiming { pts_ns: 0, dts_ns: 0, duration_ns: 0, capture_ns: 0, arrival_ns: 0 , keyframe: false},
        sequence: 0,
        meta: Default::default(),
    };
    let mut sink = CollectSink::default();
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    // Only input 0 has a frame: the round is incomplete, so nothing is emitted
    // and the (interpreter-less) batch call is never reached.
    rt.block_on(agg.process(0, PipelinePacket::DataFrame(frame), &mut sink)).unwrap();
    assert!(sink.packets.is_empty());
    assert_eq!(agg.emitted_count(), 0);
}

#[cfg(feature = "analytics")]
#[test]
fn batches_two_inputs_into_one_frame_with_metadata() {
    use g2g_core::memory::SystemSlice;
    use g2g_core::{AnalyticsMeta, Frame, FrameTiming, MemoryDomain};

    std::env::set_var(
        "PYTHONPATH",
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"),
    );

    let mut agg = PyAggregator::new("echo_element", "EchoTransform", 2);
    agg.configure_pipeline(0, &rgba_2x1()).unwrap();
    agg.configure_pipeline(1, &rgba_2x1()).unwrap();

    let frame = |first: u8| Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed({
            let mut b = vec![0u8; 8];
            b[0] = first;
            b.into_boxed_slice()
        })),
        timing: FrameTiming { pts_ns: 0, dts_ns: 0, duration_ns: 0, capture_ns: 0, arrival_ns: 0 , keyframe: false},
        sequence: 0,
        meta: Default::default(),
    };

    let mut sink = CollectSink::default();
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    // First input: no complete round yet.
    rt.block_on(agg.process(0, PipelinePacket::DataFrame(frame(10)), &mut sink)).unwrap();
    assert!(sink.packets.is_empty());
    // Second input completes the round -> one Python batch call -> one anchor.
    rt.block_on(agg.process(1, PipelinePacket::DataFrame(frame(20)), &mut sink)).unwrap();

    assert_eq!(sink.packets.len(), 1);
    assert_eq!(agg.emitted_count(), 1);
    let PipelinePacket::DataFrame(out) = &sink.packets[0] else {
        panic!("expected a DataFrame");
    };
    let MemoryDomain::System(slice) = &out.domain else { panic!("expected System memory") };
    // Anchor byte 0 = sum of the batch's first bytes (10 + 20).
    assert_eq!(slice.as_slice()[0], 30);
    // The batch attached one detection whose label is the batch size (2).
    let analytics = out.meta.get::<AnalyticsMeta>().expect("aggregate metadata attached");
    let dets: Vec<_> = analytics.detections().collect();
    assert_eq!(dets.len(), 1);
    assert_eq!(dets[0].label, 2);
}
