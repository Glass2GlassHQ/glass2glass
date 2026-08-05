//! M921: the blob-header registry decodes what the Python element host really
//! produces. `PropEcho` tags its results `model_name` / `device`, and
//! `EchoTransform` tags an `embedding`; `g2g_core::decode_blob` turns those
//! opaque payloads into typed values without the consumer knowing which element
//! wrote them.
//!
//! Needs libpython + the `metadata`-enabled core, so the file compiles away
//! without the `analytics` feature.
#![cfg(feature = "analytics")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::memory::SystemSlice;
use g2g_core::{
    decode_blob, AsyncElement, BlobMeta, Caps, DecodedBlob, Dim, Frame, FrameTiming, G2gError,
    MemoryDomain, OutputSink, PipelinePacket, PropValue, PushOutcome, Rate, RawVideoFormat,
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
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    }
}

fn caps_2x1() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(1),
        framerate: Rate::Fixed(30),
    }
}

/// Run one frame through the named fixture class and return its blobs.
fn blobs_from(class: &str, props: &[(&str, PropValue)]) -> BlobMeta {
    std::env::set_var(
        "PYTHONPATH",
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"),
    );
    let mut el = PyTransform::new("echo_element", class);
    for (name, value) in props {
        el.set_property(name, value.clone()).unwrap();
    }
    el.configure_pipeline(&caps_2x1()).unwrap();

    let mut sink = CollectSink::default();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    rt.block_on(el.process(PipelinePacket::DataFrame(frame_2x1_rgba()), &mut sink))
        .unwrap();

    let PipelinePacket::DataFrame(frame) = &sink.packets[0] else {
        panic!("expected a DataFrame downstream");
    };
    frame
        .meta
        .get::<BlobMeta>()
        .expect("the fixture attaches blobs")
        .clone()
}

#[test]
fn registry_decodes_the_python_hosts_text_blobs() {
    let blobs = blobs_from(
        "PropEcho",
        &[
            ("model-name", PropValue::Str("yolo11m.onnx".into())),
            ("device", PropValue::Str("cuda:0".into())),
            ("batch-size", PropValue::Int(4)),
        ],
    );
    assert_eq!(
        decode_blob(blobs.get("model_name").expect("model_name blob")),
        Some(DecodedBlob::Text("yolo11m.onnx".into()))
    );
    assert_eq!(
        decode_blob(blobs.get("device").expect("device blob")),
        Some(DecodedBlob::Text("cuda:0".into()))
    );
}

#[test]
fn registry_decodes_the_python_hosts_embedding_blob() {
    let blobs = blobs_from("EchoTransform", &[]);
    let embedding = blobs.get("embedding").expect("embedding blob");
    // The fixture writes bytes 1..4, one little-endian f32.
    let Some(DecodedBlob::Embedding(values)) = decode_blob(embedding) else {
        panic!("the embedding header must decode to an f32 vector");
    };
    assert_eq!(values.len(), embedding.payload.len() / 4);
    assert_eq!(values[0], f32::from_le_bytes([1, 2, 3, 4]));
}
