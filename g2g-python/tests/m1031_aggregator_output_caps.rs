//! M1031: a hosted aggregator declares its own output caps and receives its
//! element properties.
//!
//! The gst-python-ml aggregator families read one media type and emit another
//! (audio in / text out for transcription, text in / text out for translation
//! and the LLM stages). `PyAggregator` used to accept one caps on every pad and
//! produce that same caps, and it forwarded no properties at all, so those
//! elements could not be hosted. `input-caps=` / `output-caps=` split the two
//! sides, and the declared properties reach the Python instance the way
//! `PyTransform`'s do (M321).
#![cfg(feature = "analytics")]

use g2g_core::memory::SystemSlice;
use g2g_core::property::{PropError, PropKind};
use g2g_core::{
    AudioFormat, BlobMeta, Caps, Dim, Frame, FrameTiming, G2gError, Interlace, MemoryDomain,
    MultiInputElement, OutputSink, PipelinePacket, PropValue, PushOutcome, Rate, RawVideoFormat,
    TextFormat,
};
use g2g_python::PyAggregator;

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

/// Every test here configures the element, and configuring spawns a worker that
/// imports the fixture, so the path has to be set before any of them run. Tests
/// in one binary share a process and run on parallel threads, hence the `Once`.
fn fixtures_on_path() {
    static SET: std::sync::Once = std::sync::Once::new();
    SET.call_once(|| {
        std::env::set_var(
            "PYTHONPATH",
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"),
        );
    });
}

/// Without `output-caps=` the aggregator still emits what it read, the shape
/// every batched detector depends on.
#[test]
fn the_output_is_the_negotiated_input_caps_by_default() {
    fixtures_on_path();
    let mut el = PyAggregator::new("echo_element", "EchoTransform", 2);
    el.configure_pipeline(0, &rgba_2x1()).unwrap();
    assert_eq!(el.output_caps().unwrap(), rgba_2x1());
}

/// `output-caps=text/x-raw,format=utf8` is what a transcription element needs:
/// audio on every input pad, text downstream.
#[test]
fn output_caps_declares_a_different_media_type_than_the_input() {
    fixtures_on_path();
    let mut el = PyAggregator::new("echo_element", "EchoTransform", 1);
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

    let audio = Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: 1,
        sample_rate: 16_000,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    };
    // The input pad now accepts audio, which the RGBA-only default refused.
    assert_eq!(el.intercept_caps(0, &audio).unwrap(), audio);
    el.configure_pipeline(0, &audio).unwrap();
    assert_eq!(
        el.output_caps().unwrap(),
        Caps::Text {
            format: TextFormat::Utf8
        },
        "the declared output caps win over the negotiated input caps"
    );
}

/// A description naming a set rather than one caps has no single answer, so it
/// is refused at property-set time instead of resolving to an arbitrary member.
#[test]
fn an_unfixed_caps_description_is_refused() {
    let mut el = PyAggregator::new("echo_element", "EchoTransform", 1);
    assert_eq!(
        el.set_property("output-caps", PropValue::Str("video/x-raw".into())),
        Err(PropError::Value),
        "format-less video/x-raw names every raw format at any geometry"
    );
    assert_eq!(
        el.set_property("output-caps", PropValue::Str("nonsuch/type".into())),
        Err(PropError::Value)
    );
}

/// The declared element properties reach the hosted Python instance, so a
/// gst-python-ml aggregator gets its `model-name` / `device` / `language` the
/// way a hosted transform does.
#[test]
fn element_properties_reach_the_hosted_aggregator() {
    fixtures_on_path();

    let mut el = PyAggregator::new("echo_element", "PropEcho", 1);
    el.set_property("model-name", PropValue::Str("whisper-small".into()))
        .unwrap();
    el.set_property("language", PropValue::Str("ko".into()))
        .unwrap();
    el.configure_pipeline(0, &rgba_2x1()).unwrap();

    let mut sink = CollectSink::default();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    rt.block_on(el.process(0, PipelinePacket::DataFrame(frame_2x1_rgba()), &mut sink))
        .unwrap();

    let PipelinePacket::DataFrame(frame) = &sink.packets[0] else {
        panic!("expected a DataFrame downstream");
    };
    let blobs = frame
        .meta
        .get::<BlobMeta>()
        .expect("PropEcho should attach blobs");
    let by_header = |h: &str| {
        blobs
            .iter()
            .find(|b| b.header == h)
            .map(|b| b.payload.clone())
            .unwrap_or_default()
    };
    assert_eq!(by_header("model_name"), b"whisper-small");
    assert_eq!(by_header("language"), b"ko");
}

/// `parse_launch` reads a property's kind out of `properties()` and then sets it
/// by name, so a declared spec with no `set_property` arm is a launch line that
/// fails on a property the element advertises.
#[test]
fn every_declared_property_is_settable() {
    fixtures_on_path();
    let mut el = PyAggregator::new("echo_element", "EchoTransform", 1);
    for spec in el.properties() {
        let value = match spec.kind {
            PropKind::Bool => PropValue::Bool(true),
            PropKind::Int => PropValue::Int(1),
            PropKind::Uint => PropValue::Uint(1),
            PropKind::Double => PropValue::Double(1.0),
            PropKind::Fraction => PropValue::Fraction(30, 1),
            // The two caps properties are the ones that validate their value.
            PropKind::Str if spec.name.ends_with("-caps") => {
                PropValue::Str("text/x-raw,format=utf8".into())
            }
            PropKind::Str => PropValue::Str("x".into()),
            _ => continue,
        };
        assert_ne!(
            el.set_property(spec.name, value),
            Err(PropError::Unknown),
            "declared property '{}' has no set_property arm",
            spec.name
        );
    }
}
