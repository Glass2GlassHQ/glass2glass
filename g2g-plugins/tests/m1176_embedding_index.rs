#![cfg(feature = "embedding-index")]
//! M1176: the embedding index sink. The rows it writes are read back with
//! rusqlite in the element's own unit tests (the integration crate cannot reach
//! the optional dependency); here the element is driven through the whole launch
//! path, storing both payload shapes and refusing a mixed index.

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{BlobMeta, ByteStreamEncoding, Caps, G2gError, OutputSink, PushOutcome};
use g2g_plugins::embeddingsink::EmbeddingSink;

/// The stream the rows are recorded under, and the model that produced them.
const SOURCE_ID: &str = "north-gate";
const MODEL_NAME: &str = "clip-vit-b32";
const OTHER_MODEL: &str = "dinov2";
/// The vector each frame carries, and the pts it is stored at.
const VECTOR: &[f32] = &[0.5, -0.25, 0.125];
const FIRST_PTS_NS: u64 = 1_500_000_000;
const SECOND_PTS_NS: u64 = 2_500_000_000;

struct NullSink;

impl OutputSink for NullSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        packet_slot.take();
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

/// The bare payload shape: the vector's little-endian bytes on their own.
fn bare_payload() -> Vec<u8> {
    VECTOR
        .iter()
        .flat_map(|component| component.to_le_bytes())
        .collect()
}

/// gst-python-ml's framed shape: a little-endian header length, the JSON header
/// naming the model, then the vector.
fn framed_payload(model_name: &str) -> Vec<u8> {
    let header = format!("{{\"model_name\": \"{model_name}\"}}");
    let mut payload = Vec::from((header.len() as u32).to_le_bytes());
    payload.extend_from_slice(header.as_bytes());
    payload.extend_from_slice(&bare_payload());
    payload
}

fn embedded_frame(pts_ns: u64, payload: Vec<u8>) -> Frame {
    let mut frame = Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(Vec::new().into_boxed_slice())),
        timing: FrameTiming {
            pts_ns,
            ..FrameTiming::default()
        },
        sequence: pts_ns,
        meta: Default::default(),
    };
    let mut blobs = BlobMeta::new();
    blobs.push("GST-EMBEDDING:", payload);
    frame.meta.attach(blobs);
    frame
}

fn index_path(name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(name);
    let _ = std::fs::remove_file(&path);
    path
}

fn byte_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::Raw,
    }
}

#[tokio::test]
async fn stores_both_payload_shapes_and_refuses_a_second_model() {
    let path = index_path("m1176_embeddings.db");
    let mut sink = EmbeddingSink::new()
        .with_location(path.to_string_lossy().into_owned())
        .with_source_id(SOURCE_ID)
        .with_model_name(MODEL_NAME);
    sink.configure_pipeline(&byte_caps())
        .expect("the index opens");
    let mut out = NullSink;

    // The framed payload names the model; the bare one leans on `model-name`.
    sink.process(
        PipelinePacket::DataFrame(embedded_frame(FIRST_PTS_NS, framed_payload(MODEL_NAME))),
        &mut out,
    )
    .await
    .expect("the framed vector is stored");
    sink.process(
        PipelinePacket::DataFrame(embedded_frame(SECOND_PTS_NS, bare_payload())),
        &mut out,
    )
    .await
    .expect("the bare vector is stored");
    assert_eq!(sink.rows_written(), 2);

    // A frame with no embedding is not a row.
    sink.process(
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(Vec::new().into_boxed_slice())),
            timing: FrameTiming {
                pts_ns: SECOND_PTS_NS,
                ..FrameTiming::default()
            },
            sequence: 0,
            meta: Default::default(),
        }),
        &mut out,
    )
    .await
    .expect("a frame without an embedding passes");
    assert_eq!(sink.rows_written(), 2);

    let mixed = sink
        .process(
            PipelinePacket::DataFrame(embedded_frame(SECOND_PTS_NS, framed_payload(OTHER_MODEL))),
            &mut out,
        )
        .await;
    assert_eq!(
        mixed,
        Err(G2gError::InputRefused),
        "one index holds one model"
    );
    assert!(std::fs::metadata(&path).expect("the index file").len() > 0);
    let _ = std::fs::remove_file(&path);
}

/// The sink takes its whole configuration from a launch line, which is how a
/// `gst-launch`-style pipeline reaches it.
#[test]
fn takes_its_properties_from_a_launch_line() {
    let path = index_path("m1176_embeddings_launch.db");
    let line = format!(
        "fakesrc ! embeddingsink location={} source-id={SOURCE_ID} model-name={MODEL_NAME}",
        path.to_string_lossy()
    );
    g2g_core::runtime::parse_launch(&g2g_plugins::registry::default_registry(), &line)
        .expect("embeddingsink is registered and takes these properties");
}
