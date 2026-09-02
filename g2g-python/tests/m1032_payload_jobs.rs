//! M1032: a hosted aggregator reads a non-video payload and emits a buffer of a
//! different size.
//!
//! `output-caps=` (M1031) let the element declare audio in / text out, but the
//! host could not carry it: every job needed raw-video geometry, and the frames
//! that came back were the ones Python was handed, mutated in place. A
//! transcript is neither the shape nor the size of the audio it came from, so a
//! payload job routes to `g2g_process_payload` and the element returns its
//! output through `meta.emit`.
#![cfg(feature = "analytics")]

use g2g_core::memory::SystemSlice;
use g2g_core::{
    AudioFormat, BlobMeta, Caps, Frame, FrameTiming, G2gError, MemoryDomain, MultiInputElement,
    OutputSink, PipelinePacket, PropValue, PushOutcome,
};
use g2g_python::PyAggregator;

const AUDIO_BYTES: usize = 64;

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

fn audio_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: 1,
        sample_rate: 16_000,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

/// A buffer of PCM, byte-filled so a passthrough is distinguishable from an
/// emitted payload.
fn audio_frame() -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(
            vec![0xABu8; AUDIO_BYTES].into_boxed_slice(),
        )),
        FrameTiming {
            pts_ns: 1_000,
            duration_ns: 500,
            ..Default::default()
        },
        7,
    )
}

/// Every test configures the element, which spawns a worker that imports the
/// fixture, so the path has to be set before any of them run.
fn fixtures_on_path() {
    static SET: std::sync::Once = std::sync::Once::new();
    SET.call_once(|| {
        std::env::set_var(
            "PYTHONPATH",
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"),
        );
    });
}

fn push_one(el: &mut PyAggregator, frame: Frame) -> Result<CollectSink, G2gError> {
    let mut sink = CollectSink::default();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    rt.block_on(el.process(0, PipelinePacket::DataFrame(frame), &mut sink))?;
    Ok(sink)
}

fn transcriber() -> PyAggregator {
    fixtures_on_path();
    let mut el = PyAggregator::new("echo_element", "AudioTranscriber", 1);
    el.set_property(
        "input-caps",
        PropValue::Str("audio/x-raw,format=S16LE,rate=16000,channels=1".into()),
    )
    .unwrap();
    el.set_property(
        "output-caps",
        PropValue::Str("text/x-raw,format=utf8".into()),
    )
    .unwrap();
    el.configure_pipeline(0, &audio_caps()).unwrap();
    el
}

/// The frame that reaches the sink is the one the element emitted, not the audio
/// it read: different bytes, different length.
#[test]
fn an_emitted_payload_replaces_the_frame_it_was_read_from() {
    let mut el = transcriber();
    let sink = push_one(&mut el, audio_frame()).unwrap();

    let PipelinePacket::DataFrame(frame) = &sink.packets[0] else {
        panic!("expected a DataFrame downstream");
    };
    let MemoryDomain::System(bytes) = &frame.domain else {
        panic!("the emitted payload is System memory");
    };
    let expected = format!("heard {AUDIO_BYTES} bytes");
    assert_eq!(bytes.as_slice(), expected.as_bytes());
    assert_ne!(bytes.as_slice().len(), AUDIO_BYTES);
    assert!(
        !bytes.as_slice().contains(&0xAB),
        "none of the input audio survived into the output"
    );
}

/// The emitted frame stands in for the anchor, so it keeps its place in the
/// stream: the same timing, the same number, and the metadata the call produced.
/// One input emitting several buffers numbers them on from there, since a sink
/// reads a repeated number as a stream fault.
#[test]
fn the_emitted_frame_keeps_the_anchor_timing_and_metadata() {
    let mut el = transcriber();
    let sink = push_one(&mut el, audio_frame()).unwrap();

    let PipelinePacket::DataFrame(frame) = &sink.packets[0] else {
        panic!("expected a DataFrame downstream");
    };
    assert_eq!(frame.timing.pts_ns, 1_000);
    assert_eq!(frame.timing.duration_ns, 500);
    assert_eq!(
        frame.sequence, 7,
        "the emitted buffer takes the place of the one it was made from"
    );

    let blobs = frame.meta.get::<BlobMeta>().expect("the caps blob");
    let caps = blobs.iter().find(|b| b.header == "caps").unwrap();
    assert_eq!(
        caps.payload,
        audio_caps().to_gst_string().into_bytes(),
        "the payload hook receives the negotiated input caps"
    );
}

/// An element that defines only the picture hooks cannot take an audio batch:
/// `g2g_process_batch` would be handed a buffer it has no geometry for, so the
/// host refuses the frame instead.
#[test]
fn an_element_without_the_payload_hook_is_refused() {
    fixtures_on_path();
    let mut el = PyAggregator::new("echo_element", "EchoTransform", 1);
    el.set_property(
        "input-caps",
        PropValue::Str("audio/x-raw,format=S16LE,rate=16000,channels=1".into()),
    )
    .unwrap();
    el.configure_pipeline(0, &audio_caps()).unwrap();

    assert_eq!(
        push_one(&mut el, audio_frame()).err(),
        Some(G2gError::UnsupportedDomain)
    );
}
