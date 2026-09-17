//! M1178: `pyelement only-on=` gates the Python call (`analytics` feature).
//!
//! A frame that does not carry the named blob goes downstream untouched, with no
//! call into the hosted class at all. The fixture counts its calls and reports
//! the count as a blob, so a skipped frame is visible as a frame with no count
//! on it. Needs libpython + the `metadata`-enabled core.
#![cfg(feature = "analytics")]

use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, BlobMeta, Caps, Dim, Frame, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, PropValue, PushOutcome, Rate, RawVideoFormat,
};
use g2g_python::PyTransform;

const WIDTH: u32 = 2;
const HEIGHT: u32 = 1;
/// The blob the element is gated on, and one it is not.
const GATE: &str = "alert";
const OTHER_HEADER: &str = "embedding";
/// Blob the fixture reports its call count under.
const CALLS_HEADER: &str = "calls";

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

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(30),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

/// A frame carrying one blob tagged `header`.
fn frame_carrying(header: &str, sequence: u64) -> Frame {
    let bytes = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
    let mut frame = Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence,
        meta: Default::default(),
    };
    let mut blobs = BlobMeta::new();
    blobs.push(header, Vec::from(b"x".as_slice()));
    frame.meta.attach(blobs);
    frame
}

#[test]
fn only_on_skips_the_python_call_for_a_frame_without_the_blob() {
    std::env::set_var(
        "PYTHONPATH",
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"),
    );

    let mut element = PyTransform::new("m1178_element", "CountingTransform");
    element
        .set_property("only-on", PropValue::Str(String::from(GATE)))
        .expect("only-on is a declared property");
    assert_eq!(
        element.get_property("only-on"),
        Some(PropValue::Str(String::from(GATE)))
    );
    assert!(
        element
            .properties()
            .iter()
            .any(|spec| spec.name == "only-on"),
        "a launch line looks the name up in properties() before setting it"
    );
    element.configure_pipeline(&caps()).unwrap();

    let mut sink = CollectSink::default();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    // First the frame the gate rejects, then the one it admits, through the one
    // instance: the fixture's counter spans both.
    for (index, header) in [OTHER_HEADER, GATE].into_iter().enumerate() {
        runtime
            .block_on(element.process(
                PipelinePacket::DataFrame(frame_carrying(header, index as u64)),
                &mut sink,
            ))
            .unwrap();
    }

    assert_eq!(sink.packets.len(), 2, "both frames go downstream");
    assert_eq!(element.emitted_count(), 2);

    let PipelinePacket::DataFrame(skipped) = &sink.packets[0] else {
        panic!("expected a DataFrame downstream");
    };
    let skipped_blobs = skipped.meta.get::<BlobMeta>().expect("its own blob");
    assert!(
        skipped_blobs.get(CALLS_HEADER).is_none(),
        "the hosted class was never called for a frame without {GATE}"
    );
    assert!(
        skipped_blobs.get(OTHER_HEADER).is_some(),
        "the frame is forwarded with its metadata untouched"
    );

    let PipelinePacket::DataFrame(ran) = &sink.packets[1] else {
        panic!("expected a DataFrame downstream");
    };
    let ran_blobs = ran.meta.get::<BlobMeta>().expect("blobs on the way out");
    assert_eq!(
        ran_blobs
            .get(CALLS_HEADER)
            .map(|blob| blob.payload.as_slice()),
        Some([1u8].as_slice()),
        "exactly one call reached the hosted class, on the frame carrying {GATE}"
    );
}
