//! Metadata replay (M1176): reattaches the records a
//! [`MetaSink`](crate::metasink::MetaSink) wrote to the frames at the same
//! timestamps, the gst-python-ml `pyml_metareplay` analog. Pixels pass through
//! untouched; only metadata is added, so a detector run once can be replayed onto
//! the video as often as wanted.
//!
//! A frame takes the record whose time is closest to its own, within half a frame
//! duration (20 ms when the duration is unknown). `detections` come back as an
//! [`AnalyticsMeta`] with the pixel boxes normalized against the negotiated
//! geometry and a class-name table built from the labels in the file, in order of
//! first appearance. Every other key except `pts` and `text` comes back as a
//! [`BlobMeta`] blob under that key, holding the value's compact JSON.
//!
//! A line that is not a JSON object, or carries no `pts`, is skipped: a truncated
//! tail must not stop the replay. A file that cannot be read fails negotiation.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use serde_json::{Map, Value};

use g2g_core::log::{path_io_err, short_type_name, LogName, LogSource};
use g2g_core::{
    AnalyticsMeta, AsyncElement, BBox, BlobMeta, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, G2gError, ObjectDetection, OutputSink, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec,
};

use crate::metasink::{
    caps_dimensions, DETECTIONS_KEY, HEIGHT_KEY, LABEL_KEY, NON_BLOB_KEYS, PTS_KEY, SCORE_KEY,
    WIDTH_KEY, X_KEY, Y_KEY,
};

const NS_PER_MILLISECOND: f64 = 1_000_000.0;
const MILLISECONDS_PER_SECOND: f64 = 1000.0;
/// How far a frame may sit from a record when its duration is unknown, the same
/// window the Python element allows.
const UNKNOWN_DURATION_TOLERANCE_MS: f64 = 20.0;

/// The records of one file: each one keyed by its time in whole milliseconds,
/// plus the class-name table the labels in them index.
#[derive(Debug, Default)]
struct Records {
    by_millisecond: BTreeMap<i64, Map<String, Value>>,
    /// Times in order, so the record nearest a frame is a binary search.
    times: Vec<i64>,
    class_names: Vec<String>,
    label_ids: BTreeMap<String, u32>,
}

impl Records {
    fn read(text: &str) -> Self {
        let mut records = Records::default();
        for line in text.lines() {
            let Ok(Value::Object(record)) = serde_json::from_str::<Value>(line.trim()) else {
                continue;
            };
            let Some(seconds) = record.get(PTS_KEY).and_then(Value::as_f64) else {
                continue;
            };
            records.intern_labels(&record);
            let millisecond = (seconds * MILLISECONDS_PER_SECOND).round() as i64;
            records.by_millisecond.insert(millisecond, record);
        }
        records.times = records.by_millisecond.keys().copied().collect();
        records
    }

    /// Give every label name an id, in the order the file first mentions it.
    fn intern_labels(&mut self, record: &Map<String, Value>) {
        let Some(detections) = record.get(DETECTIONS_KEY).and_then(Value::as_array) else {
            return;
        };
        for detection in detections {
            let Some(name) = detection.get(LABEL_KEY).and_then(Value::as_str) else {
                continue;
            };
            if !self.label_ids.contains_key(name) {
                self.label_ids
                    .insert(name.to_string(), self.class_names.len() as u32);
                self.class_names.push(name.to_string());
            }
        }
    }

    /// The record nearest `pts_ns`, within half a frame duration.
    fn nearest(&self, pts_ns: u64, duration_ns: u64) -> Option<&Map<String, Value>> {
        if self.times.is_empty() {
            return None;
        }
        let millisecond = (pts_ns as f64 / NS_PER_MILLISECOND).round() as i64;
        let tolerance = if duration_ns == 0 {
            UNKNOWN_DURATION_TOLERANCE_MS
        } else {
            duration_ns as f64 / NS_PER_MILLISECOND / 2.0
        };
        let after = self.times.partition_point(|time| *time < millisecond);
        let first = after.saturating_sub(1);
        let last = (after + 1).min(self.times.len());
        let closest = self.times[first..last]
            .iter()
            .copied()
            .min_by_key(|time| (time - millisecond).abs())?;
        ((closest - millisecond).abs() as f64 <= tolerance)
            .then(|| self.by_millisecond.get(&closest))
            .flatten()
    }

    /// The detections of one record, with the pixel boxes normalized against a
    /// `width` x `height` frame. A malformed entry is skipped.
    fn detections(
        &self,
        record: &Map<String, Value>,
        width: u32,
        height: u32,
    ) -> Vec<ObjectDetection> {
        let Some(entries) = record.get(DETECTIONS_KEY).and_then(Value::as_array) else {
            return Vec::new();
        };
        entries
            .iter()
            .filter_map(|entry| self.detection(entry, width, height))
            .collect()
    }

    fn detection(&self, entry: &Value, width: u32, height: u32) -> Option<ObjectDetection> {
        let name = entry.get(LABEL_KEY)?.as_str()?;
        let label = *self.label_ids.get(name)?;
        let normalized = |key: &str, span: u32| -> Option<f32> {
            let pixels = entry.get(key)?.as_f64()?;
            (span > 0).then(|| (pixels / f64::from(span)) as f32)
        };
        Some(ObjectDetection {
            bbox: BBox {
                x: normalized(X_KEY, width)?,
                y: normalized(Y_KEY, height)?,
                w: normalized(WIDTH_KEY, width)?,
                h: normalized(HEIGHT_KEY, height)?,
            },
            label,
            confidence: entry.get(SCORE_KEY)?.as_f64()? as f32,
        })
    }
}

/// Reattaches a `metasink` file's detections and blobs to the frames they were
/// recorded from.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::metareplay::MetaReplay;
///
/// // gst-launch equivalent: metareplay location=records.jsonl
/// let replay = MetaReplay::new().with_location("records.jsonl");
/// ```
#[derive(Debug)]
pub struct MetaReplay {
    location: String,
    records: Records,
    width: u32,
    height: u32,
    configured: bool,
    log_name: LogName,
}

impl Default for MetaReplay {
    fn default() -> Self {
        Self::new()
    }
}

impl MetaReplay {
    pub fn new() -> Self {
        Self {
            location: String::new(),
            records: Records::default(),
            width: 0,
            height: 0,
            configured: false,
            log_name: LogName::new(),
        }
    }

    /// Read the records from this JSON lines file (the `location` property).
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = location.into();
        self
    }

    /// How many records the file held.
    pub fn records(&self) -> usize {
        self.records.by_millisecond.len()
    }

    fn accepts(caps: &Caps) -> bool {
        matches!(caps, Caps::RawVideo { .. }) && caps_dimensions(caps).is_some()
    }

    /// Attach one record's detections and blobs to a frame.
    fn attach(&self, frame: &mut g2g_core::Frame, record: &Map<String, Value>) {
        let detections = self.records.detections(record, self.width, self.height);
        if !detections.is_empty() {
            let mut analytics = AnalyticsMeta::new();
            analytics.set_class_names(self.records.class_names.iter().map(String::as_str));
            for detection in detections {
                analytics.add_detection(detection);
            }
            frame.meta.attach(analytics);
        }
        let blobs: Vec<(&String, Vec<u8>)> = record
            .iter()
            .filter(|(key, _)| !NON_BLOB_KEYS.contains(&key.as_str()))
            .map(|(key, value)| (key, value.to_string().into_bytes()))
            .collect();
        if blobs.is_empty() {
            return;
        }
        if frame.meta.get::<BlobMeta>().is_none() {
            frame.meta.attach(BlobMeta::new());
        }
        let carried = frame.meta.get_mut::<BlobMeta>().expect("just attached");
        for (key, payload) in blobs {
            carried.push(key.clone(), payload);
        }
    }
}

impl AsyncElement for MetaReplay {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Metadata replay",
            "Filter/Analytics",
            "Reattaches the detections and blobs a metasink wrote at the same timestamps",
            "g2g",
        )
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if Self::accepts(upstream_caps) {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| {
            if Self::accepts(input) {
                CapsSet::one(input.clone())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (width, height) = caps_dimensions(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        self.width = width;
        self.height = height;
        if !self.configured {
            let text = std::fs::read_to_string(&self.location).map_err(|e| {
                path_io_err(
                    self.log_category(),
                    "read",
                    std::path::Path::new(&self.location),
                    e,
                )
            })?;
            self.records = Records::read(&text);
        }
        self.configured = true;
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
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(mut frame) => {
                    if let Some(pts_ns) = frame.timing.pts() {
                        if let Some(record) = self.records.nearest(pts_ns, frame.timing.duration_ns)
                        {
                            // The borrow of `self.records` ends with the clone, so
                            // `attach` can take `&self` for the class-name table.
                            let record = record.clone();
                            self.attach(&mut frame, &record);
                        }
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(caps) => {
                    if let Some((width, height)) = caps_dimensions(&caps) {
                        self.width = width;
                        self.height = height;
                    }
                    out.push(PipelinePacket::CapsChanged(caps)).await?;
                }
                // The runner's transform arm forwards EOS; don't double it.
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        METAREPLAY_PROPS
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

impl LogSource for MetaReplay {
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

/// `MetaReplay`'s settable properties: the file the records come from.
static METAREPLAY_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "location",
    PropKind::Str,
    "JSON lines file a metasink wrote",
)
.with_default("")];
