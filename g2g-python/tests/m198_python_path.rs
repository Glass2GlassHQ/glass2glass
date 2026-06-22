//! M198 step 2: the live per-frame path, embedded CPython (`python` feature).
//!
//! Drives `PyTransform` against a stdlib-only hosted element fixture and proves
//! the zero-copy buffer-protocol contract: Python writes into the frame's own
//! System memory in place, and that write is observable on the frame that flows
//! downstream. Needs libpython at build + run time, so the whole file compiles
//! away without the feature.
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

fn frame_2x1_rgba(first: u8) -> Frame {
    // 2x1 RGBA = 8 bytes; only the first byte's value matters to the assertion.
    let mut bytes = vec![0u8; 8];
    bytes[0] = first;
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming { pts_ns: 0, dts_ns: 0, duration_ns: 0, capture_ns: 0, arrival_ns: 0 },
        sequence: 0,
        meta: Default::default(),
    }
}

#[test]
fn python_writes_into_frame_memory_in_place() {
    // Put the fixture element on the interpreter's import path. Set before the
    // first GIL acquisition so it is on sys.path at interpreter init.
    std::env::set_var(
        "PYTHONPATH",
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"),
    );

    let mut el = PyTransform::new("echo_element", "EchoTransform").with_draw_label(true);
    let caps = Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(1),
        framerate: Rate::Fixed(30),
    };
    // Instantiates the Python class under the GIL.
    el.configure_pipeline(&caps).unwrap();

    let mut sink = CollectSink::default();
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    rt.block_on(el.process(PipelinePacket::DataFrame(frame_2x1_rgba(10)), &mut sink))
        .unwrap();

    // Exactly one frame flowed, and the first byte was incremented (10 -> 11)
    // by Python writing into the Rust buffer through the buffer protocol.
    assert_eq!(sink.packets.len(), 1);
    let PipelinePacket::DataFrame(frame) = &sink.packets[0] else {
        panic!("expected a DataFrame downstream");
    };
    let MemoryDomain::System(slice) = &frame.domain else {
        panic!("expected System memory");
    };
    assert_eq!(slice.as_slice()[0], 11, "in-place write did not reach Rust");
}
