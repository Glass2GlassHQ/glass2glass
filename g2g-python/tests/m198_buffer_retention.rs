//! The zero-copy frame buffer lent to a hosted element must not outlive the
//! `g2g_process` call: a retained `memoryview` / numpy view holds a raw pointer
//! that dangles once the frame is freed downstream. The host counts outstanding
//! buffer exports and fails the frame loud instead of risking a use-after-free.
#![cfg(feature = "python")]

use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, Dim, Frame, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, PushOutcome, Rate, RawVideoFormat,
};
use g2g_python::PyTransform;

#[derive(Default)]
struct CollectSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for CollectSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");

        self.packets.push(packet);
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
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

fn run_one_frame(class: &str) -> (Result<(), G2gError>, CollectSink) {
    std::env::set_var(
        "PYTHONPATH",
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"),
    );

    let mut el = PyTransform::new("echo_element", class);
    let caps = Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(1),
        framerate: Rate::Fixed(30),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    };
    el.configure_pipeline(&caps).unwrap();

    let mut sink = CollectSink::default();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let result = rt.block_on(el.process(PipelinePacket::DataFrame(frame_2x1_rgba()), &mut sink));
    (result, sink)
}

#[test]
fn retained_buffer_view_is_rejected_not_use_after_free() {
    let (result, sink) = run_one_frame("RetainingTransform");

    // The element retained the buffer past the call; the host must fail the
    // frame rather than forward it (which would later dangle).
    assert!(
        result.is_err(),
        "retained buffer view must fail the frame, got {result:?}"
    );
    assert!(
        sink.packets.is_empty(),
        "no frame should be forwarded after a retention violation"
    );
}

#[test]
fn a_view_held_only_by_a_garbage_cycle_is_not_retained() {
    let (result, sink) = run_one_frame("CycleHoldingTransform");

    assert!(
        result.is_ok(),
        "a view only a dead cycle holds must not fail the frame, got {result:?}"
    );
    assert_eq!(sink.packets.len(), 1);
}
