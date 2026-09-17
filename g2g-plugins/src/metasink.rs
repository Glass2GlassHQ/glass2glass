//! Metadata sink (M1176): writes one JSON line per frame holding the frame's
//! presentation time and whatever analytics it carries, the gst-python-ml
//! `pyml_metasink` analog. The line goes to a file (appended) or to stdout, and
//! onto the bus as [`BusMessage::MetadataRecord`] so a tool watching the run
//! sees each record without reading the file back.
//!
//! The record is the interchange format [`MetaReplay`](crate::metareplay::MetaReplay)
//! reads: the same keys, in the same units, as the Python element writes, so a
//! file either one produces is readable by the other.
//!
//! - `pts`: seconds, omitted on a frame with no presentation time.
//! - `text`: the frame's bytes as UTF-8, for a `Caps::Text` input. Nothing else
//!   is written for text.
//! - `detections`: one object per [`ObjectDetection`] with the class name (the
//!   label id as a decimal string without a name table), the box in pixels, and
//!   the confidence.
//! - one key per [`BlobMeta`] blob whose payload is JSON, under the blob's
//!   canonical header. An embedding's f32 bytes are not JSON, so they are
//!   skipped.
//!
//! A record holding nothing but `pts` is not written: a pooled frame carrying an
//! emptied relation meta would otherwise write a line per frame saying nothing.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};

use serde_json::{Map, Value};

use g2g_core::log::{io_err, path_io_err, short_type_name, LogName, LogSource};
use g2g_core::{
    AnalyticsMeta, AsyncElement, BlobMeta, BusHandle, BusMessage, Caps, CapsConstraint,
    ConfigureOutcome, Dim, ElementMetadata, Frame, G2gError, HardwareError, ObjectDetection,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec,
};

/// The record key holding the frame time in seconds.
pub(crate) const PTS_KEY: &str = "pts";
/// The record key holding a text frame's payload.
pub(crate) const TEXT_KEY: &str = "text";
/// The record key holding the detection array.
pub(crate) const DETECTIONS_KEY: &str = "detections";
/// The detection keys: a class name, the box in pixels, the confidence.
pub(crate) const LABEL_KEY: &str = "label";
pub(crate) const X_KEY: &str = "x";
pub(crate) const Y_KEY: &str = "y";
pub(crate) const WIDTH_KEY: &str = "w";
pub(crate) const HEIGHT_KEY: &str = "h";
pub(crate) const SCORE_KEY: &str = "score";

/// The keys a record carries that are not a blob, so everything else in it is
/// one (the `metareplay` split).
pub(crate) const NON_BLOB_KEYS: &[&str] = &[PTS_KEY, TEXT_KEY, DETECTIONS_KEY];

pub(crate) const NS_PER_SECOND: f64 = 1_000_000_000.0;

/// An empty `location`, i.e. write to stdout.
const STDOUT_LOCATION: &str = "";

/// The pixel geometry a normalized box is rendered against, when the caps fix
/// one. `None` for caps without both dimensions (text, audio, an unfixed size).
pub(crate) fn caps_dimensions(caps: &Caps) -> Option<(u32, u32)> {
    let (width, height) = match caps {
        Caps::RawVideo { width, height, .. } => (width, height),
        Caps::CompressedVideo { width, height, .. } => (width, height),
        _ => return None,
    };
    match (width, height) {
        (Dim::Fixed(width), Dim::Fixed(height)) => Some((*width, *height)),
        _ => None,
    }
}

/// One detection as the record writes it: the class name (or the label id in
/// decimal where the producer published no name table) and the box in pixels of
/// a `width` x `height` frame.
pub(crate) fn detection_json(
    detection: &ObjectDetection,
    class_name: Option<&str>,
    width: u32,
    height: u32,
) -> Value {
    let pixels = |normalized: f32, span: u32| (normalized * span as f32).round() as i64;
    let mut object = Map::new();
    object.insert(
        LABEL_KEY.to_string(),
        Value::String(class_name.map_or_else(|| detection.label.to_string(), String::from)),
    );
    object.insert(X_KEY.to_string(), pixels(detection.bbox.x, width).into());
    object.insert(Y_KEY.to_string(), pixels(detection.bbox.y, height).into());
    object.insert(
        WIDTH_KEY.to_string(),
        pixels(detection.bbox.w, width).into(),
    );
    object.insert(
        HEIGHT_KEY.to_string(),
        pixels(detection.bbox.h, height).into(),
    );
    object.insert(
        SCORE_KEY.to_string(),
        Value::from(f64::from(detection.confidence)),
    );
    Value::Object(object)
}

/// Writes the analytics metadata, blobs and text of each frame as one JSON line.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::metasink::MetaSink;
///
/// // gst-launch equivalent: metasink location=records.jsonl
/// let sink = MetaSink::new().with_location("records.jsonl");
/// assert_eq!(sink.records_written(), 0);
/// ```
#[derive(Debug)]
pub struct MetaSink {
    location: String,
    writer: Option<BufWriter<std::fs::File>>,
    /// The input is timed text, so the record carries its bytes instead of
    /// detections.
    text_input: bool,
    width: u32,
    height: u32,
    bus: Option<BusHandle>,
    records_written: u64,
    configured: bool,
    log_name: LogName,
}

impl Default for MetaSink {
    fn default() -> Self {
        Self::new()
    }
}

impl MetaSink {
    pub fn new() -> Self {
        Self {
            location: String::from(STDOUT_LOCATION),
            writer: None,
            text_input: false,
            width: 0,
            height: 0,
            bus: None,
            records_written: 0,
            configured: false,
            log_name: LogName::new(),
        }
    }

    /// Append the lines to this file instead of stdout (the `location`
    /// property).
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = location.into();
        self
    }

    /// Records written so far. A frame whose record said nothing but its time
    /// does not count.
    pub fn records_written(&self) -> u64 {
        self.records_written
    }

    /// The record for one frame, or `None` when it holds nothing but a time.
    fn record_for(&self, frame: &Frame) -> Option<Map<String, Value>> {
        let mut record = Map::new();
        if let Some(pts_ns) = frame.timing.pts() {
            record.insert(
                PTS_KEY.to_string(),
                Value::from(pts_ns as f64 / NS_PER_SECOND),
            );
        }
        if self.text_input {
            let bytes = frame.domain.as_system_slice()?;
            record.insert(
                TEXT_KEY.to_string(),
                Value::String(String::from_utf8_lossy(bytes).into_owned()),
            );
            return Some(record);
        }
        if let Some(analytics) = frame.meta.get::<AnalyticsMeta>() {
            let detections: Vec<Value> = analytics
                .detections()
                .map(|detection| {
                    detection_json(
                        detection,
                        analytics.class_name(detection.label),
                        self.width,
                        self.height,
                    )
                })
                .collect();
            if !detections.is_empty() {
                record.insert(DETECTIONS_KEY.to_string(), Value::Array(detections));
            }
        }
        if let Some(blobs) = frame.meta.get::<BlobMeta>() {
            for blob in blobs.iter() {
                if let Ok(value) = serde_json::from_slice::<Value>(&blob.payload) {
                    record.insert(blob.header.clone(), value);
                }
            }
        }
        (record.len() > 1).then_some(record)
    }

    fn write_line(&mut self, line: &str) -> Result<(), G2gError> {
        match &mut self.writer {
            Some(file) => {
                writeln!(file, "{line}").map_err(io_err)?;
                file.flush().map_err(io_err)
            }
            None => {
                let mut out = std::io::stdout().lock();
                writeln!(out, "{line}").map_err(io_err)?;
                out.flush().map_err(io_err)
            }
        }
    }
}

impl AsyncElement for MetaSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Metadata sink",
            "Sink",
            "Writes the analytics metadata, blobs and text of each frame as one JSON line",
            "g2g",
        )
    }

    /// Reads the text payload out of host memory; the metadata itself travels
    /// beside the frame in any domain.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// Metadata rides on any media type, so the sink takes whatever arrives.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.text_input = matches!(absolute_caps, Caps::Text { .. });
        if let Some((width, height)) = caps_dimensions(absolute_caps) {
            self.width = width;
            self.height = height;
        }
        // Opened once: a mid-stream caps change re-enters here and must keep
        // appending to the same file.
        if self.writer.is_none() && self.location != STDOUT_LOCATION {
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.location)
                .map_err(|e| {
                    path_io_err(
                        self.log_category(),
                        "open",
                        std::path::Path::new(&self.location),
                        e,
                    )
                })?;
            self.writer = Some(BufWriter::new(file));
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn set_bus(&mut self, bus: BusHandle) {
        self.bus = Some(bus);
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
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let Some(record) = self.record_for(&frame) else {
                        return Ok(());
                    };
                    let line = serde_json::to_string(&record)
                        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                    self.write_line(&line)?;
                    self.records_written += 1;
                    if let Some(bus) = &self.bus {
                        bus.try_post(BusMessage::MetadataRecord {
                            element: self
                                .log_name
                                .instance()
                                .unwrap_or_else(|| short_type_name::<Self>())
                                .to_string(),
                            record: line,
                        });
                    }
                }
                PipelinePacket::CapsChanged(caps) => {
                    self.configure_pipeline(&caps)?;
                }
                PipelinePacket::Eos => {
                    if let Some(file) = &mut self.writer {
                        file.flush().map_err(io_err)?;
                    }
                }
                _ => {}
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        METASINK_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" => {
                self.location = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.location.clone())),
            _ => None,
        }
    }
}

impl LogSource for MetaSink {
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

impl PadTemplates for MetaSink {
    /// Wildcard sink, matching the `AcceptsAny` constraint.
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink_any()])
    }
}

/// `MetaSink`'s settable properties: where the lines go.
static METASINK_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "location",
    PropKind::Str,
    "file the JSON lines are appended to, stdout when empty",
)
.with_default(STDOUT_LOCATION)];
