//! The zero-copy frame buffer lent to a hosted element must not outlive the
//! `g2g_process` call: a retained `memoryview` / numpy view holds a raw pointer
//! that dangles once the frame is freed downstream. The host counts outstanding
//! buffer exports and fails the frame loud instead of risking a use-after-free.
#![cfg(feature = "python")]

use core::future::Future;
use core::pin::Pin;

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
fn retained_buffer_view_is_rejected_not_use_after_free() {
    std::env::set_var("PYTHONPATH", concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"));

    let mut el = PyTransform::new("echo_element", "RetainingTransform");
    let caps = Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(1),
        framerate: Rate::Fixed(30),
    };
    el.configure_pipeline(&caps).unwrap();

    let mut sink = CollectSink::default();
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let result = rt.block_on(el.process(PipelinePacket::DataFrame(frame_2x1_rgba()), &mut sink));

    // The element retained the buffer past the call; the host must fail the
    // frame rather than forward it (which would later dangle).
    assert!(result.is_err(), "retained buffer view must fail the frame, got {result:?}");
    assert!(sink.packets.is_empty(), "no frame should be forwarded after a retention violation");
}
