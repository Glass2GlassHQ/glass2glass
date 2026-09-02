//! M1033: a hosted 1-in-1-out element changes media type.
//!
//! The single-chain gst-python-ml families (transcribe, translate, llm,
//! separate, tts) are `filesrc ! decodebin ! audioconvert ! pyml_x ! sink`, so
//! they host on `pyelement` (`PyTransform`), not on the fan-in `pyaggregator`.
//! `PyTransform` gets the same `input-caps` / `output-caps` split M1031 gave the
//! aggregator, declares the boundary its output caps make it, and pushes the
//! payload the hosted element emitted (M1032) rather than the buffer it read.
#![cfg(feature = "analytics")]

use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, BlobMeta, Caps, CapsConstraint, Dim, Frame, FrameTiming, G2gError,
    Interlace, MemoryDomain, OutputSink, PipelinePacket, PropValue, PushOutcome, Rate,
    RawVideoFormat, TextFormat,
};
use g2g_python::PyTransform;

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

fn text_caps() -> Caps {
    Caps::Text {
        format: TextFormat::Utf8,
    }
}

fn rgba_2x1() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(1),
        framerate: Rate::Fixed(30),
        interlace: Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

/// Timing the emitted frame is expected to inherit from the frame it replaces.
fn anchor_timing() -> FrameTiming {
    FrameTiming {
        pts_ns: 2_000,
        dts_ns: 1_500,
        duration_ns: 500,
        ..Default::default()
    }
}

fn payload_frame(bytes: Vec<u8>) -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        anchor_timing(),
        3,
    )
}

fn audio_frame() -> Frame {
    payload_frame(vec![0xABu8; AUDIO_BYTES])
}

fn fixtures_on_path() {
    static SET: std::sync::Once = std::sync::Once::new();
    SET.call_once(|| {
        std::env::set_var(
            "PYTHONPATH",
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"),
        );
    });
}

/// The transcription shape: audio on the sink pad, text on the source pad.
fn transcriber() -> PyTransform {
    fixtures_on_path();
    let mut el = PyTransform::new("echo_element", "AudioTranscriber");
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
    el
}

/// A text-to-speech element hosting `class`: text on the sink pad, generated
/// audio on the source pad.
fn text_to_speech(class: &str) -> PyTransform {
    fixtures_on_path();
    let mut el = PyTransform::new("echo_element", class);
    el.set_property(
        "input-caps",
        PropValue::Str("text/x-raw,format=utf8".into()),
    )
    .unwrap();
    el.set_property(
        "output-caps",
        PropValue::Str("audio/x-raw,format=S16LE,rate=16000,channels=1".into()),
    )
    .unwrap();
    el
}

/// The text-to-speech shape, the transcriber run backwards: text on the sink
/// pad, generated audio on the source pad.
fn synthesizer() -> PyTransform {
    text_to_speech("SpeechSynthesizer")
}

/// The streaming shape: one text buffer in, several audio buffers out.
fn chunking_synthesizer() -> PyTransform {
    text_to_speech("ChunkedSynthesizer")
}

/// The streaming shape with a schedule: `chunks` buffers, each `chunk_ns` long,
/// the first presented at the anchor's PTS.
fn streaming_synthesizer(chunks: u64, chunk_ns: u64) -> PyTransform {
    let mut el = text_to_speech("StreamingSynthesizer");
    el.set_property("chunks", PropValue::Uint(chunks)).unwrap();
    el.set_property("chunk-duration", PropValue::Uint(chunk_ns))
        .unwrap();
    el.set_property("first-pts", PropValue::Uint(anchor_timing().pts_ns))
        .unwrap();
    el
}

fn push(el: &mut PyTransform, packet: PipelinePacket) -> CollectSink {
    let mut sink = CollectSink::default();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    rt.block_on(el.process(packet, &mut sink)).unwrap();
    sink
}

fn pushed_frame(sink: &CollectSink) -> &Frame {
    let PipelinePacket::DataFrame(frame) = &sink.packets[0] else {
        panic!("expected a DataFrame downstream");
    };
    frame
}

fn pushed_frames(sink: &CollectSink) -> Vec<&Frame> {
    sink.packets
        .iter()
        .map(|packet| {
            let PipelinePacket::DataFrame(frame) = packet else {
                panic!("expected a DataFrame downstream");
            };
            frame
        })
        .collect()
}

fn payload_of(frame: &Frame) -> &[u8] {
    let MemoryDomain::System(bytes) = &frame.domain else {
        panic!("an emitted payload is System memory");
    };
    bytes.as_slice()
}

fn derived_output(el: &PyTransform, input: &Caps) -> Vec<Caps> {
    let CapsConstraint::DerivedOutput(derive) = el.caps_constraint_as_transform() else {
        panic!("a same-format PyTransform declares a DerivedOutput constraint");
    };
    derive(input).alternatives().to_vec()
}

/// The `(accepted, produced)` pairs a format-boundary element states outright.
fn mapped_pairs(el: &PyTransform) -> Vec<(Vec<Caps>, Vec<Caps>)> {
    let CapsConstraint::Mapping(pairs) = el.caps_constraint_as_transform() else {
        panic!("a format-boundary PyTransform declares a Mapping constraint");
    };
    pairs
        .into_iter()
        .map(|(input, output)| {
            (
                input.alternatives().to_vec(),
                output.alternatives().to_vec(),
            )
        })
        .collect()
}

/// The whole point: what reaches the sink is the text the element emitted, not
/// the audio it was handed.
#[test]
fn a_payload_transform_pushes_the_emitted_text_downstream() {
    let mut el = transcriber();
    el.configure_pipeline(&audio_caps()).unwrap();
    let sink = push(&mut el, PipelinePacket::DataFrame(audio_frame()));

    let frame = pushed_frame(&sink);
    assert_eq!(
        payload_of(frame),
        format!("heard {AUDIO_BYTES} bytes").as_bytes()
    );
    assert_ne!(payload_of(frame).len(), AUDIO_BYTES);
    assert_eq!(
        frame.timing,
        anchor_timing(),
        "a one-argument emit inherits the whole timing of the frame it replaced"
    );
    assert_eq!(
        frame.sequence, 3,
        "an emitted buffer keeps the place of the one it replaced"
    );
    assert_eq!(el.emitted_count(), 1);
}

/// Streaming speech is chunked and the separation family emits one buffer per
/// stem, so one input can produce several outputs and every one has to travel
/// on. The host used to hold a single emitted payload, which dropped all but
/// the first.
#[test]
fn every_emitted_buffer_reaches_the_sink() {
    const TEXT: &[u8] = b"say ";

    let mut el = chunking_synthesizer();
    el.configure_pipeline(&text_caps()).unwrap();
    let sink = push(
        &mut el,
        PipelinePacket::DataFrame(payload_frame(TEXT.to_vec())),
    );

    let frames: Vec<&Frame> = sink
        .packets
        .iter()
        .map(|packet| {
            let PipelinePacket::DataFrame(frame) = packet else {
                panic!("expected a DataFrame downstream");
            };
            frame
        })
        .collect();
    assert_eq!(
        frames.iter().map(|f| payload_of(f)).collect::<Vec<_>>(),
        [b"say 0", b"say 1", b"say 2"],
        "in the order the element emitted them"
    );
    assert!(
        frames.iter().all(|f| f.timing == anchor_timing()),
        "each inherits the timing of the buffer they were generated from"
    );
    assert_eq!(
        frames.iter().map(|f| f.sequence).collect::<Vec<_>>(),
        [3, 4, 5],
        "numbered on from the buffer they replace, and never repeating"
    );
    assert_eq!(el.emitted_count(), 3);
}

/// Whatever upstream attached describes the stream, not the one buffer this
/// element replaced, so it reaches the sink on every buffer sent in its place.
#[test]
fn every_emitted_buffer_carries_the_metadata_it_arrived_with() {
    const TEXT: &[u8] = b"say ";
    const UPSTREAM: &str = "upstream";

    let mut el = chunking_synthesizer();
    el.configure_pipeline(&text_caps()).unwrap();

    let mut frame = payload_frame(TEXT.to_vec());
    let mut blobs = BlobMeta::new();
    blobs.push(UPSTREAM.to_string(), vec![1, 2, 3]);
    frame.meta.attach(blobs);

    let sink = push(&mut el, PipelinePacket::DataFrame(frame));

    let carried: Vec<bool> = sink
        .packets
        .iter()
        .map(|packet| {
            let PipelinePacket::DataFrame(frame) = packet else {
                panic!("expected a DataFrame downstream");
            };
            frame
                .meta
                .get::<BlobMeta>()
                .is_some_and(|blobs| blobs.iter().any(|blob| blob.header == UPSTREAM))
        })
        .collect();
    assert_eq!(carried, [true, true, true]);
}

/// Generated audio runs for as long as its samples, not for as long as the text
/// buffer it came from, so the element states the duration and the host takes it
/// over the anchor's.
#[test]
fn an_emitted_duration_replaces_the_anchors() {
    const TEXT: &[u8] = b"hello";
    // What the fixture generates: 100 samples per character, S16 mono at 16 kHz.
    const SAMPLES: u64 = TEXT.len() as u64 * 100;
    const SPEECH_NS: u64 = SAMPLES * 1_000_000_000 / 16_000;

    let mut el = synthesizer();
    el.configure_pipeline(&text_caps()).unwrap();
    let sink = push(
        &mut el,
        PipelinePacket::DataFrame(payload_frame(TEXT.to_vec())),
    );

    let frame = pushed_frame(&sink);
    assert_eq!(payload_of(frame).len() as u64, SAMPLES * 2);
    assert_eq!(frame.timing.duration_ns, SPEECH_NS);
    assert_ne!(frame.timing.duration_ns, anchor_timing().duration_ns);
    assert_eq!(
        frame.timing.pts_ns,
        anchor_timing().pts_ns,
        "the speech is presented when the text it was generated from arrived"
    );
    assert_eq!(frame.timing.dts_ns, anchor_timing().dts_ns);
    assert_eq!(frame.sequence, 3);
}

/// Streaming speech plays chunk after chunk, so each emitted buffer states the
/// time it starts at. Without that they all inherit the anchor's PTS and the
/// whole utterance stacks at the instant the text arrived.
#[test]
fn each_emitted_buffer_can_carry_its_own_pts() {
    const TEXT: &[u8] = b"say ";
    const CHUNKS: u64 = 4;
    const CHUNK_NS: u64 = 250_000_000;

    let mut el = streaming_synthesizer(CHUNKS, CHUNK_NS);
    el.configure_pipeline(&text_caps()).unwrap();
    let sink = push(
        &mut el,
        PipelinePacket::DataFrame(payload_frame(TEXT.to_vec())),
    );

    let frames = pushed_frames(&sink);
    let expected: Vec<u64> = (0..CHUNKS)
        .map(|chunk| anchor_timing().pts_ns + chunk * CHUNK_NS)
        .collect();
    assert_eq!(
        frames.iter().map(|f| f.timing.pts_ns).collect::<Vec<_>>(),
        expected,
        "one chunk after another, not all at the anchor's PTS"
    );
    assert!(
        frames.iter().all(|f| f.timing.dts_ns == f.timing.pts_ns),
        "an emitted buffer is never reordered"
    );
    assert!(
        frames.iter().all(|f| f.timing.duration_ns == CHUNK_NS),
        "each chunk runs for the length it states"
    );
    assert_eq!(el.emitted_count(), CHUNKS);
}

/// An element with nothing to say about when its buffer is presented emits it
/// with no presentation time at all, and a sink presents it on arrival.
#[test]
fn an_emitted_buffer_can_have_no_pts() {
    const TEXT: &[u8] = b"say ";

    let mut el = text_to_speech("UnstampedSynthesizer");
    el.configure_pipeline(&text_caps()).unwrap();
    let sink = push(
        &mut el,
        PipelinePacket::DataFrame(payload_frame(TEXT.to_vec())),
    );

    let frame = pushed_frame(&sink);
    assert_eq!(frame.timing.pts_ns, FrameTiming::PTS_NONE);
    assert!(frame.timing.pts().is_none());
    assert_ne!(frame.timing.pts_ns, anchor_timing().pts_ns);
}

/// Negotiation has to agree with what the element actually pushes, or a text
/// sink downstream is offered audio.
#[test]
fn the_declared_output_caps_are_the_ones_the_element_advertises() {
    let el = transcriber();
    assert_eq!(el.intercept_caps(&audio_caps()).unwrap(), audio_caps());
    assert_eq!(el.propose_output_caps(&audio_caps()), text_caps());
    assert!(
        el.is_format_boundary(),
        "audio in, text out is a format boundary"
    );
}

/// The accepted input is stated, not only derived from whatever arrives.
///
/// An upstream parser advertises its whole range with the rate and channel count
/// left open (`wavparse`), so with nothing narrowing the link it fixates on its
/// own and the element then refuses what it picked.
#[test]
fn a_boundary_element_states_the_input_it_accepts() {
    assert_eq!(
        mapped_pairs(&transcriber()),
        [(vec![audio_caps()], vec![text_caps()])]
    );
}

/// A mid-stream caps change announces this element's output side, not the input
/// it just accepted.
#[test]
fn a_caps_change_is_forwarded_as_the_declared_output_caps() {
    let mut el = transcriber();
    el.configure_pipeline(&audio_caps()).unwrap();
    let sink = push(&mut el, PipelinePacket::CapsChanged(audio_caps()));

    assert!(matches!(
        &sink.packets[0],
        PipelinePacket::CapsChanged(c) if *c == text_caps()
    ));
}

/// Without `output-caps=` a `pyelement` is still the same-format transform every
/// hosted detector and overlay depends on.
#[test]
fn without_output_caps_the_element_stays_a_passthrough() {
    let el = PyTransform::new("echo_element", "EchoTransform");
    assert_eq!(derived_output(&el, &rgba_2x1()), [rgba_2x1()]);
    assert_eq!(el.propose_output_caps(&rgba_2x1()), rgba_2x1());
    assert!(!el.is_format_boundary());
    assert_eq!(el.get_property("output-caps"), None);
}

/// The two caps properties round trip through the `gst-launch` face, and a
/// description naming a set rather than one caps is refused.
#[test]
fn the_caps_properties_round_trip() {
    let el = transcriber();
    assert_eq!(
        el.get_property("input-caps"),
        Some(PropValue::Str(audio_caps().to_gst_string()))
    );
    assert_eq!(
        el.get_property("output-caps"),
        Some(PropValue::Str(text_caps().to_gst_string()))
    );

    let mut el = PyTransform::new("echo_element", "EchoTransform");
    assert_eq!(
        el.set_property("output-caps", PropValue::Str("video/x-raw".into())),
        Err(g2g_core::PropError::Value),
        "format-less video/x-raw names every raw format at any geometry"
    );
}
