//! Embedding sink (M1176): stores the embedding vector of each frame in a
//! sqlite index, the gst-python-ml `pyml_embeddingsink` analog. The schema is the
//! one `embedding_index.py` creates, so a file either side writes is searchable
//! by the other's `search_video`.
//!
//! The `embedding` blob arrives in one of two shapes: gst-python-ml's framed
//! form, a four-byte little-endian header length then a JSON header (which may
//! name the model) then the vector's little-endian `f32` bytes, or the bare
//! vector on its own. The framed form is recognized by a header length that fits
//! the payload and parses as JSON; anything else is read as a bare vector.
//!
//! One index holds one model: a second model name is refused rather than mixed
//! in, since a search compares vectors across the whole table. A frame whose blob
//! names no model needs the `model-name` property, and is an error without it.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::{
    g2g_error, AsyncElement, BlobMeta, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata,
    G2gError, HardwareError, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec,
};

/// The blob header an embedding arrives under.
const EMBEDDING_BLOB: &str = "embedding";
/// The framed payload's header length prefix, in bytes.
const HEADER_LENGTH_BYTES: usize = 4;
/// The header key naming the model that produced the vector.
const HEADER_MODEL_KEY: &str = "model_name";
const BYTES_PER_COMPONENT: usize = 4;
const NS_PER_SECOND: f64 = 1_000_000_000.0;

/// The index schema, byte for byte the one `embedding_index.py` creates.
const CREATE_TABLE: &str = "CREATE TABLE IF NOT EXISTS embeddings (
    source_id TEXT NOT NULL,
    pts REAL NOT NULL,
    model_name TEXT NOT NULL,
    vector BLOB NOT NULL
)";
const INSERT_ROW: &str =
    "INSERT INTO embeddings (source_id, pts, model_name, vector) VALUES (?, ?, ?, ?)";
const SELECT_MODEL_NAME: &str = "SELECT model_name FROM embeddings LIMIT 1";

/// The model name a payload declares, and its vector.
#[derive(Debug, PartialEq)]
struct Embedding {
    model_name: Option<String>,
    vector: Vec<f32>,
}

/// Read one blob payload. The framed form's header is JSON; everything else is
/// the bare vector. A trailing partial component is ignored rather than trusted.
fn decode_embedding(payload: &[u8]) -> Embedding {
    let framed = payload
        .get(..HEADER_LENGTH_BYTES)
        .and_then(|prefix| prefix.try_into().ok())
        .map(u32::from_le_bytes)
        .and_then(|length| {
            let end = HEADER_LENGTH_BYTES.checked_add(length as usize)?;
            let header = payload.get(HEADER_LENGTH_BYTES..end)?;
            let header: serde_json::Value = serde_json::from_slice(header).ok()?;
            Some((header, end))
        });
    let (model_name, body) = match framed {
        Some((header, end)) => (
            header
                .get(HEADER_MODEL_KEY)
                .and_then(serde_json::Value::as_str)
                .map(String::from),
            &payload[end..],
        ),
        None => (None, payload),
    };
    Embedding {
        model_name: model_name.filter(|name| !name.is_empty()),
        vector: body
            .as_chunks::<BYTES_PER_COMPONENT>()
            .0
            .iter()
            .map(|component| f32::from_le_bytes(*component))
            .collect(),
    }
}

/// Stores each frame's embedding vector in a searchable sqlite index.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::embeddingsink::EmbeddingSink;
///
/// // gst-launch equivalent: embeddingsink location=index.db source-id=cam1
/// let sink = EmbeddingSink::new().with_location("index.db");
/// assert_eq!(sink.rows_written(), 0);
/// ```
#[derive(Debug)]
pub struct EmbeddingSink {
    location: String,
    source_id: String,
    model_name: String,
    index: Option<rusqlite::Connection>,
    rows_written: u64,
    log_name: LogName,
}

impl Default for EmbeddingSink {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingSink {
    pub fn new() -> Self {
        Self {
            location: String::new(),
            source_id: String::new(),
            model_name: String::new(),
            index: None,
            rows_written: 0,
            log_name: LogName::new(),
        }
    }

    /// The sqlite file the index is kept in (the `location` property).
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = location.into();
        self
    }

    /// The name a search result reports for this stream (the `source-id`
    /// property).
    pub fn with_source_id(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = source_id.into();
        self
    }

    /// The model recorded when the blob header names none (the `model-name`
    /// property).
    pub fn with_model_name(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = model_name.into();
        self
    }

    /// Rows stored so far.
    pub fn rows_written(&self) -> u64 {
        self.rows_written
    }

    fn sqlite_failed(&self, doing: &str, error: rusqlite::Error) -> G2gError {
        g2g_error!(self, "cannot {doing} in {}: {error}", self.location);
        G2gError::Hardware(HardwareError::Other)
    }

    /// Store one vector, refusing a model the index does not already hold.
    fn store(&mut self, pts_ns: u64, embedding: Embedding) -> Result<(), G2gError> {
        let index = self.index.as_ref().ok_or(G2gError::NotConfigured)?;
        let model_name = embedding
            .model_name
            .unwrap_or_else(|| self.model_name.clone());
        if model_name.is_empty() {
            g2g_error!(
                self,
                "the embedding blob names no model, set model-name on this sink"
            );
            return Err(G2gError::NotConfigured);
        }
        let held: Option<String> = match index.query_row(SELECT_MODEL_NAME, [], |row| row.get(0)) {
            Ok(held) => Some(held),
            Err(rusqlite::Error::QueryReturnedNoRows) => None,
            Err(error) => return Err(self.sqlite_failed("read the index's model", error)),
        };
        if held.as_deref().is_some_and(|held| held != model_name) {
            g2g_error!(
                self,
                "the index holds {} embeddings, it cannot also hold {model_name}",
                held.unwrap_or_default()
            );
            return Err(G2gError::InputRefused);
        }
        let vector: Vec<u8> = embedding
            .vector
            .iter()
            .flat_map(|component| component.to_le_bytes())
            .collect();
        index
            .execute(
                INSERT_ROW,
                rusqlite::params![
                    self.source_id,
                    pts_ns as f64 / NS_PER_SECOND,
                    model_name,
                    vector
                ],
            )
            .map_err(|error| self.sqlite_failed("store an embedding", error))?;
        self.rows_written += 1;
        Ok(())
    }
}

impl AsyncElement for EmbeddingSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Embedding sink",
            "Sink",
            "Stores the embedding vector of each frame in a searchable sqlite index",
            "g2g",
        )
    }

    /// The vector rides beside the frame as metadata, so the frame's own bytes
    /// are never read: any domain will do.
    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if self.index.is_some() {
            return Ok(ConfigureOutcome::Accepted);
        }
        if self.location.is_empty() {
            g2g_error!(self, "set location to the sqlite file the index goes in");
            return Err(G2gError::NotConfigured);
        }
        let index = rusqlite::Connection::open(&self.location)
            .map_err(|error| self.sqlite_failed("open the index", error))?;
        index
            .execute(CREATE_TABLE, [])
            .map_err(|error| self.sqlite_failed("create the index", error))?;
        self.index = Some(index);
        Ok(ConfigureOutcome::Accepted)
    }

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if self.index.is_none() {
                return Err(G2gError::NotConfigured);
            }
            let PipelinePacket::DataFrame(frame) = packet else {
                return Ok(());
            };
            let Some(pts_ns) = frame.timing.pts() else {
                return Ok(());
            };
            let Some(payload) = frame
                .meta
                .get::<BlobMeta>()
                .and_then(|blobs| blobs.get(EMBEDDING_BLOB))
                .map(|blob| blob.payload.clone())
            else {
                return Ok(());
            };
            self.store(pts_ns, decode_embedding(&payload))
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        EMBEDDINGSINK_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let text = value.as_str().ok_or(PropError::Type)?.to_string();
        match name {
            "location" => self.location = text,
            "source-id" => self.source_id = text,
            "model-name" => self.model_name = text,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.location.clone())),
            "source-id" => Some(PropValue::Str(self.source_id.clone())),
            "model-name" => Some(PropValue::Str(self.model_name.clone())),
            _ => None,
        }
    }
}

impl LogSource for EmbeddingSink {
    fn log_category(&self) -> &'static str {
        short_type_name::<Self>()
    }
    fn log_instance(&self) -> Option<&str> {
        self.log_name.instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        self.log_name.category()
    }
}

impl PadTemplates for EmbeddingSink {
    /// Wildcard sink, matching the `AcceptsAny` constraint.
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink_any()])
    }
}

/// `EmbeddingSink`'s settable properties, named as the Python element's.
static EMBEDDINGSINK_PROPS: &[PropertySpec] = &[
    PropertySpec::new("location", PropKind::Str, "sqlite file holding the index").with_default(""),
    PropertySpec::new(
        "source-id",
        PropKind::Str,
        "name a search result reports for this stream",
    )
    .with_default(""),
    PropertySpec::new(
        "model-name",
        PropKind::Str,
        "model recorded when the blob header names none",
    )
    .with_default(""),
];

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn decodes_both_payload_shapes() {
        let vector = [1.5f32, -2.25, 3.0];
        let bare: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            decode_embedding(&bare),
            Embedding {
                model_name: None,
                vector: Vec::from(vector),
            }
        );
        let header = format!("{{\"{HEADER_MODEL_KEY}\": \"clip\"}}");
        let mut framed = Vec::from((header.len() as u32).to_le_bytes());
        framed.extend_from_slice(header.as_bytes());
        framed.extend_from_slice(&bare);
        assert_eq!(
            decode_embedding(&framed),
            Embedding {
                model_name: Some(String::from("clip")),
                vector: Vec::from(vector),
            }
        );
    }

    /// A header length past the end of the payload must read as a bare vector,
    /// not index out of bounds.
    #[test]
    fn a_bogus_header_length_is_not_framed() {
        let mut payload = Vec::from(u32::MAX.to_le_bytes());
        payload.extend_from_slice(&1.0f32.to_le_bytes());
        assert_eq!(decode_embedding(&payload).model_name, None);
    }

    /// The row a stored vector becomes, read back through the same schema
    /// `search_video` queries. The integration test cannot do this: rusqlite is
    /// an optional dependency of this crate alone.
    #[test]
    fn a_stored_vector_reads_back_as_the_row_a_search_finds() {
        let source_id = "north-gate";
        let model_name = "clip-vit-b32";
        let vector = [0.5f32, -0.25, 0.125];
        let pts_ns = 1_500_000_000u64;
        let path = std::env::temp_dir().join("g2g_embeddingsink_row.db");
        let _ = std::fs::remove_file(&path);

        let mut sink = EmbeddingSink::new()
            .with_location(path.to_string_lossy().into_owned())
            .with_source_id(source_id);
        sink.index = Some(rusqlite::Connection::open(&path).expect("the index opens"));
        sink.index
            .as_ref()
            .expect("just opened")
            .execute(CREATE_TABLE, [])
            .expect("the schema is created");
        sink.store(
            pts_ns,
            Embedding {
                model_name: Some(String::from(model_name)),
                vector: Vec::from(vector),
            },
        )
        .expect("the vector is stored");

        let index = rusqlite::Connection::open(&path).expect("the index reopens");
        let (stored_source, stored_pts, stored_model, stored_vector): (
            String,
            f64,
            String,
            Vec<u8>,
        ) = index
            .query_row(
                "SELECT source_id, pts, model_name, vector FROM embeddings",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .expect("one row");
        assert_eq!(stored_source, source_id);
        assert_eq!(stored_pts, pts_ns as f64 / NS_PER_SECOND);
        assert_eq!(stored_model, model_name);
        assert_eq!(
            decode_embedding(&stored_vector).vector,
            Vec::from(vector),
            "the blob holds the little-endian f32 vector"
        );
        let _ = std::fs::remove_file(&path);
    }
}
