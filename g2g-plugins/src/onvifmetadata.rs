//! ONVIF analytics metadata (M1151, `onvif` feature): the scene description a
//! camera streams on its `application/vnd.onvif.metadata` RTSP track, turned
//! into the same [`AnalyticsMeta`] graph a detector writes.
//!
//! One `Caps::OnvifMetadata` frame is one complete `tt:MetadataStream` XML
//! document, as [`RtspSrcN`](crate::rtspsrcn::RtspSrcN) hands it over
//! (concatenated across the RTP marker bit and gzip-inflated already).
//! [`OnvifMetadataParse`] splits it into one output frame per `tt:Frame`,
//! carrying that frame's objects as detections plus the `UtcTime` it names as
//! [`WallClockMeta`]. [`OnvifMetadataCombiner`] then merges those onto the
//! video frames they describe:
//!
//! ```text
//! rtspsrcn onvif-metadata=true name=s
//!   s. ! h264parse ! avdec ! videoconvert ! onvifmetadatacombiner name=c ! analyticsoverlay ! autovideosink
//!   s. ! onvifmetadataparse ! c.
//! ```
//!
//! The two pads are matched on wall clock, not on RTP time: the ONVIF Streaming
//! Specification gives the metadata track's RTP timestamps no meaning, and says
//! a `tt:Frame`'s `UtcTime` is what names the picture it describes. `RtspSrcN`
//! puts the video's sender wall clock on the frames from the RTCP sender
//! reports, so both sides carry the same clock. Without it (a server that sends
//! no sender report, or a synthetic graph) the two fall back to PTS on the play
//! timeline.
//!
//! Coordinates: after the `tt:Transformation` stack is applied a point is in the
//! ONVIF normalized frame system, `x` and `y` in `[-1, 1]` with the origin at
//! the image centre and `y` pointing **up**. A [`BBox`] is `[0, 1]` from the
//! top-left with `y` pointing down, so both axes are remapped.
//!
//! Every count, length and coordinate here comes off the wire from a camera:
//! the document is parsed with bounds on how many frames and objects it may
//! describe, non-finite numbers are refused, and a malformed document yields no
//! output rather than an error or a panic.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::meta::{
    AnalyticsMeta, AnalyticsNode, BBox, ObjectDetection, RelationKind, Tracking, WallClockMeta,
};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError,
    MultiInputElement, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec,
};

use crate::xmlutil::days_from_civil;

/// The ONVIF schema namespace every scene-description element lives in.
/// Elements are matched by namespace and local name, never by the `tt:` prefix
/// a particular camera happens to bind to it.
const ONVIF_SCHEMA_NS: &str = "http://www.onvif.org/ver10/schema";

/// `tt:Frame` elements one document may describe before the rest are dropped.
pub const MAX_FRAMES_PER_DOCUMENT: usize = 256;
/// `tt:Object` elements one document may describe before the rest are dropped.
pub const MAX_OBJECTS_PER_DOCUMENT: usize = 1024;
/// Deepest element nesting a document may have: the schema nests about eight
/// deep, and the XML parser recurses per level, so a deeper one is refused
/// before it can exhaust the stack.
pub const MAX_ELEMENT_DEPTH: usize = 64;

/// Label id for an object the camera gave no class. Past the end of any
/// `class_names` table, so `AnalyticsMeta::class_name` reports no name for it.
pub const UNCLASSIFIED_LABEL: u32 = u32::MAX;

/// Confidence of a detection whose object carries no class descriptor. ONVIF
/// publishes a likelihood per class, not per detection, so an object with no
/// class has nothing to be uncertain about.
const UNCLASSIFIED_CONFIDENCE: f32 = 1.0;

const NANOS_PER_SEC: i64 = 1_000_000_000;
const SECS_PER_DAY: i64 = 86_400;

/// One `tt:Frame` of a metadata document: the instant it names and the objects
/// it describes.
#[derive(Debug, Clone, PartialEq)]
pub struct OnvifMetadataFrame {
    /// The `UtcTime` attribute as nanoseconds since the Unix epoch, or `None`
    /// when it is missing or unreadable.
    pub unix_nanos: Option<i64>,
    /// One detection per `tt:Object` that carried a usable bounding box, each
    /// related to a `Tracking` node holding its `ObjectId`.
    pub analytics: AnalyticsMeta,
}

// ---- document parsing ----

/// Split a payload into the XML documents it holds and parse every `tt:Frame`
/// in each. A metadata payload may carry several concatenated
/// `<?xml ...?><tt:MetadataStream>` roots, which no XML parser accepts as one
/// document, so the payload is cut at each declaration first. A chunk that does
/// not parse contributes nothing.
pub fn parse_metadata_documents(payload: &[u8]) -> Vec<OnvifMetadataFrame> {
    let Ok(text) = core::str::from_utf8(payload) else {
        return Vec::new();
    };
    let mut frames = Vec::new();
    for chunk in split_xml_documents(text) {
        parse_one_document(chunk, &mut frames);
        if frames.len() >= MAX_FRAMES_PER_DOCUMENT {
            break;
        }
    }
    frames
}

/// The XML declarations in `text` as the document boundaries they mark. A
/// payload with no declaration at all is one chunk.
fn split_xml_documents(text: &str) -> Vec<&str> {
    const DECLARATION: &str = "<?xml";
    // A declaration inside CDATA or text is content, so only one that follows
    // whitespace or a closing tag marks a boundary.
    let mut starts: Vec<usize> = text
        .match_indices(DECLARATION)
        .map(|(i, _)| i)
        .filter(|&i| {
            i == 0 || text.as_bytes()[i - 1] == b'>' || text.as_bytes()[i - 1].is_ascii_whitespace()
        })
        .collect();
    if starts.first() != Some(&0) {
        starts.insert(0, 0);
    }
    let mut chunks = Vec::with_capacity(starts.len());
    for (i, &start) in starts.iter().enumerate() {
        let end = starts.get(i + 1).copied().unwrap_or(text.len());
        let chunk = &text[start..end];
        if !chunk.trim().is_empty() {
            chunks.push(chunk);
        }
    }
    chunks
}

fn parse_one_document(text: &str, out: &mut Vec<OnvifMetadataFrame>) {
    if nests_deeper_than(text, MAX_ELEMENT_DEPTH) {
        return;
    }
    let Ok(doc) = roxmltree::Document::parse(text) else {
        return;
    };
    // A `tt:Frame` is looked for anywhere in the tree rather than under a fixed
    // root: the streaming and analytics specifications spell the root two ways
    // (`MetadataStream` and `MetaDataStream`) and show `tt:VideoAnalytics`
    // standing alone as well.
    let mut objects_left = MAX_OBJECTS_PER_DOCUMENT;
    for node in doc.descendants() {
        if out.len() >= MAX_FRAMES_PER_DOCUMENT {
            return;
        }
        if !is_onvif(node, "Frame") {
            continue;
        }
        out.push(parse_frame(node, &mut objects_left));
    }
}

/// A coordinate system: a point `p` in it maps to `p * scale + translate` in the
/// ONVIF normalized frame system.
#[derive(Debug, Clone, Copy, PartialEq)]
struct CoordinateSystem {
    translate: (f32, f32),
    scale: (f32, f32),
}

impl CoordinateSystem {
    /// The frame node's starting system: the normalized one itself.
    const NORMALIZED: Self = Self {
        translate: (0.0, 0.0),
        scale: (1.0, 1.0),
    };

    /// This system with a `tt:Transformation`'s own translation `v` and scaling
    /// `u` applied on top: `t' = v * s + t`, `s' = u * s`.
    fn compose(self, translate: (f32, f32), scale: (f32, f32)) -> Self {
        Self {
            translate: (
                translate.0 * self.scale.0 + self.translate.0,
                translate.1 * self.scale.1 + self.translate.1,
            ),
            scale: (scale.0 * self.scale.0, scale.1 * self.scale.1),
        }
    }

    /// A point in this system, in the normalized frame system.
    fn map(self, x: f32, y: f32) -> (f32, f32) {
        (
            x * self.scale.0 + self.translate.0,
            y * self.scale.1 + self.translate.1,
        )
    }
}

fn parse_frame(frame: roxmltree::Node, objects_left: &mut usize) -> OnvifMetadataFrame {
    let unix_nanos = frame.attribute("UtcTime").and_then(parse_utc_time);
    let system = transformation_of(frame, CoordinateSystem::NORMALIZED);

    let mut analytics = AnalyticsMeta::new();
    let mut class_names: Vec<String> = Vec::new();
    // ObjectId -> detection node index, so a `Parent` attribute can be wired to
    // the object it names when that object is in this same frame.
    let mut detection_of_object: Vec<(u64, usize)> = Vec::new();
    let mut parents: Vec<(usize, u64)> = Vec::new();

    for object in frame.children().filter(|n| is_onvif(*n, "Object")) {
        if *objects_left == 0 {
            break;
        }
        *objects_left -= 1;
        let Some(object_id) = attribute_u64(object, "ObjectId") else {
            continue;
        };
        let appearance = child(object, "Appearance");
        let object_system = appearance
            .map(|a| transformation_of(a, system))
            .unwrap_or(system);
        let Some(bbox) = appearance
            .and_then(|a| child(a, "Shape"))
            .and_then(|s| child(s, "BoundingBox"))
            .and_then(|b| bounding_box(b, object_system))
        else {
            continue;
        };
        let (label, confidence) = match appearance
            .and_then(|a| child(a, "Class"))
            .and_then(best_class)
        {
            Some((name, likelihood)) => (intern_class(&mut class_names, name), likelihood),
            None => (UNCLASSIFIED_LABEL, UNCLASSIFIED_CONFIDENCE),
        };

        let detection = analytics.add_detection(ObjectDetection {
            bbox,
            label,
            confidence,
        });
        let tracking = analytics.push(AnalyticsNode::Tracking(Tracking { object_id }));
        analytics.relate(detection, tracking, RelationKind::Tracks);
        detection_of_object.push((object_id, detection));
        match attribute_u64(object, "Parent") {
            Some(parent) if parent != object_id => parents.push((detection, parent)),
            _ => {}
        }
    }

    // Wired after the pass: a parent may be listed after the child that names it.
    for (child_detection, parent_id) in parents {
        if let Some((_, parent_detection)) =
            detection_of_object.iter().find(|(id, _)| *id == parent_id)
        {
            analytics.relate(*parent_detection, child_detection, RelationKind::Contains);
        }
    }
    if !class_names.is_empty() {
        analytics.set_class_names(class_names);
    }

    OnvifMetadataFrame {
        unix_nanos,
        analytics,
    }
}

/// `node`'s coordinate system: `parent` with the node's own direct
/// `tt:Transformation` child composed on top, or `parent` unchanged when it has
/// none.
fn transformation_of(node: roxmltree::Node, parent: CoordinateSystem) -> CoordinateSystem {
    let Some(transformation) = child(node, "Transformation") else {
        return parent;
    };
    let translate = child(transformation, "Translate")
        .and_then(|n| vector(n, "x", "y"))
        .unwrap_or((0.0, 0.0));
    let scale = child(transformation, "Scale")
        .and_then(|n| vector(n, "x", "y"))
        .unwrap_or((1.0, 1.0));
    parent.compose(translate, scale)
}

/// A `tt:BoundingBox` as a normalized [`BBox`]. All four edges are required and
/// finite; the box is clipped to the picture, and one that ends up with a
/// negative or non-finite extent is refused so a bogus document loses that
/// object rather than the whole frame.
fn bounding_box(node: roxmltree::Node, system: CoordinateSystem) -> Option<BBox> {
    let left = attribute_f32(node, "left")?;
    let top = attribute_f32(node, "top")?;
    let right = attribute_f32(node, "right")?;
    let bottom = attribute_f32(node, "bottom")?;
    let (qx0, qy0) = system.map(left, top);
    let (qx1, qy1) = system.map(right, bottom);
    // Normalized ONVIF (origin centre, y up) to normalized g2g (origin
    // top-left, y down).
    let x0 = ((qx0 + 1.0) / 2.0).clamp(0.0, 1.0);
    let x1 = ((qx1 + 1.0) / 2.0).clamp(0.0, 1.0);
    let y0 = ((1.0 - qy0) / 2.0).clamp(0.0, 1.0);
    let y1 = ((1.0 - qy1) / 2.0).clamp(0.0, 1.0);
    let (w, h) = (x1 - x0, y1 - y0);
    if !w.is_finite() || !h.is_finite() || w < 0.0 || h < 0.0 {
        return None;
    }
    Some(BBox { x: x0, y: y0, w, h })
}

/// The likeliest class a `tt:Class` names, in either encoding: the current
/// `tt:Type Likelihood="..."` list, and the legacy
/// `tt:ClassCandidate/{tt:Type, tt:Likelihood}` pairs. A missing likelihood
/// means certain.
fn best_class<'a>(class: roxmltree::Node<'a, '_>) -> Option<(&'a str, f32)> {
    let direct = class
        .children()
        .filter(|n| is_onvif(*n, "Type"))
        .filter_map(|n| Some((n.text()?.trim(), likelihood_attribute(n)?)));
    let legacy = class
        .children()
        .filter(|n| is_onvif(*n, "ClassCandidate"))
        .filter_map(|candidate| {
            let name = child(candidate, "Type")?.text()?.trim();
            let likelihood = match child(candidate, "Likelihood") {
                Some(n) => parse_likelihood(n.text()?.trim())?,
                None => 1.0,
            };
            Some((name, likelihood))
        });
    direct
        .chain(legacy)
        .filter(|(name, _)| !name.is_empty())
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
}

fn likelihood_attribute(node: roxmltree::Node) -> Option<f32> {
    match node.attribute("Likelihood") {
        Some(text) => parse_likelihood(text),
        None => Some(1.0),
    }
}

/// A likelihood in `[0, 1]`. Out-of-range values are clamped, non-numeric ones
/// refused (the candidate is then skipped rather than ranked at zero).
fn parse_likelihood(text: &str) -> Option<f32> {
    let value: f32 = text.trim().parse().ok()?;
    value.is_finite().then(|| value.clamp(0.0, 1.0))
}

/// The label id for a class name in this document's table, appending it when it
/// is new.
fn intern_class(names: &mut Vec<String>, name: &str) -> u32 {
    if let Some(index) = names.iter().position(|n| n == name) {
        return index as u32;
    }
    names.push(name.to_string());
    (names.len() - 1) as u32
}

fn is_onvif(node: roxmltree::Node, local: &str) -> bool {
    node.is_element()
        && node.tag_name().namespace() == Some(ONVIF_SCHEMA_NS)
        && node.tag_name().name() == local
}

fn child<'a, 'i>(node: roxmltree::Node<'a, 'i>, local: &str) -> Option<roxmltree::Node<'a, 'i>> {
    node.children().find(|n| is_onvif(*n, local))
}

fn attribute_f32(node: roxmltree::Node, name: &str) -> Option<f32> {
    let value: f32 = node.attribute(name)?.trim().parse().ok()?;
    value.is_finite().then_some(value)
}

/// Whether any element in `text` sits more than `limit` levels deep, counted
/// off the raw tags so the walk is flat. Declarations, comments and CDATA
/// openers are not levels; a self-closing tag opens none.
fn nests_deeper_than(text: &str, limit: usize) -> bool {
    let bytes = text.as_bytes();
    let mut depth = 0usize;
    let mut i = 0;
    while let Some(open) = bytes[i..].iter().position(|&b| b == b'<') {
        let start = i + open;
        let Some(close) = bytes[start..].iter().position(|&b| b == b'>') else {
            return false;
        };
        let end = start + close;
        match bytes.get(start + 1) {
            Some(b'/') => depth = depth.saturating_sub(1),
            Some(b'?') | Some(b'!') => {}
            _ if bytes[end - 1] == b'/' => {}
            _ => {
                depth += 1;
                if depth > limit {
                    return true;
                }
            }
        }
        i = end + 1;
    }
    false
}

fn attribute_u64(node: roxmltree::Node, name: &str) -> Option<u64> {
    node.attribute(name)?.trim().parse().ok()
}

fn vector(node: roxmltree::Node, x: &str, y: &str) -> Option<(f32, f32)> {
    Some((attribute_f32(node, x)?, attribute_f32(node, y)?))
}

/// An `xs:dateTime` as nanoseconds since the Unix epoch: `CCYY-MM-DDThh:mm:ss`
/// with optional fractional seconds and an optional `Z` or `±hh:mm` zone. No
/// zone means UTC, which is the only thing ONVIF puts in a metadata stream.
pub fn parse_utc_time(text: &str) -> Option<i64> {
    let text = text.trim();
    let (date, rest) = text.split_once('T')?;
    let mut ymd = date.splitn(3, '-');
    let year: i64 = ymd.next()?.parse().ok()?;
    let month: u32 = ymd.next()?.parse().ok()?;
    let day: u32 = ymd.next()?.parse().ok()?;
    if ymd.next().is_some() || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }

    // The zone designator is a trailing `Z`, or a sign that is not the first
    // character (the time itself never starts with one).
    let (time, zone_secs) = match rest.strip_suffix('Z') {
        Some(t) => (t, 0),
        None => match rest.rfind(['+', '-']).filter(|&i| i > 0) {
            Some(i) => (&rest[..i], parse_zone_offset_secs(&rest[i..])?),
            None => (rest, 0),
        },
    };
    // Everything past the second colon is the seconds field, so a third colon
    // lands inside it and fails to parse rather than being ignored.
    let mut hms = time.splitn(3, ':');
    let hour: i64 = hms.next()?.parse().ok()?;
    let minute: i64 = hms.next()?.parse().ok()?;
    let seconds_field = hms.next()?;
    let (seconds_text, fraction_text) =
        seconds_field.split_once('.').unwrap_or((seconds_field, ""));
    let second: i64 = seconds_text.parse().ok()?;
    if !(0..24).contains(&hour) || !(0..60).contains(&minute) {
        return None;
    }
    // A leap second (`:60`) is legal in xs:dateTime and counts as the second
    // after it here, which is what a POSIX epoch can represent.
    if !(0..=60).contains(&second) {
        return None;
    }

    let days = days_from_civil(year, month, day);
    let secs = days
        .checked_mul(SECS_PER_DAY)?
        .checked_add(hour * 3_600 + minute * 60 + second)?
        .checked_sub(zone_secs)?;
    secs.checked_mul(NANOS_PER_SEC)?
        .checked_add(parse_fraction_nanos(fraction_text)?)
}

/// Fractional seconds (the digits after the decimal point) as nanoseconds,
/// padded or truncated to nine digits. An empty fraction is zero.
fn parse_fraction_nanos(digits: &str) -> Option<i64> {
    const NANO_DIGITS: usize = 9;
    if digits.is_empty() {
        return Some(0);
    }
    if !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let mut nanos: i64 = 0;
    for i in 0..NANO_DIGITS {
        let digit = digits.as_bytes().get(i).map(|b| i64::from(b - b'0'));
        nanos = nanos * 10 + digit.unwrap_or(0);
    }
    Some(nanos)
}

/// A `+hh:mm` / `-hh:mm` zone designator as seconds east of UTC.
fn parse_zone_offset_secs(text: &str) -> Option<i64> {
    let sign = if text.starts_with('-') { -1 } else { 1 };
    let (hours, minutes) = text.get(1..)?.split_once(':')?;
    let hours: i64 = hours.parse().ok()?;
    let minutes: i64 = minutes.parse().ok()?;
    if hours > 23 || minutes > 59 {
        return None;
    }
    Some(sign * (hours * 3_600 + minutes * 60))
}

// ---- onvifmetadataparse ----

/// Split an ONVIF metadata document into one frame per `tt:Frame`, each
/// carrying that frame's objects as [`AnalyticsMeta`] and its `UtcTime` as
/// [`WallClockMeta`]. The payload is passed through unchanged (the same buffer,
/// shared rather than copied), so a downstream branch can still record or
/// forward the original XML.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::onvifmetadata::OnvifMetadataParse;
///
/// let parse = OnvifMetadataParse::new();
/// ```
#[derive(Debug, Default)]
pub struct OnvifMetadataParse {
    configured: bool,
    emitted: u64,
    log_name: LogName,
}

impl OnvifMetadataParse {
    pub fn new() -> Self {
        Self::default()
    }
}

impl AsyncElement for OnvifMetadataParse {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "ONVIF metadata parser",
            "Parser/Metadata",
            "Splits an ONVIF tt:MetadataStream document into per-frame analytics metadata",
            "g2g",
        )
    }

    /// Reads the XML out of host memory.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Caps::OnvifMetadata)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::OnvifMetadata => CapsSet::one(Caps::OnvifMetadata),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(absolute_caps, Caps::OnvifMetadata) {
            return Err(G2gError::CapsMismatch);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
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
                    let Some(payload) = frame.domain.as_system_slice() else {
                        return Ok(());
                    };
                    let parsed = parse_metadata_documents(payload);
                    if parsed.is_empty() {
                        g2g_core::g2g_warn!(
                            self,
                            "dropping a {}-byte document with no readable tt:Frame",
                            payload.len(),
                        );
                        return Ok(());
                    }
                    // One refcount bump per output instead of a copy of the
                    // document per frame it describes.
                    frame.domain.make_shareable();
                    for parsed_frame in parsed {
                        let mut out_frame =
                            Frame::new(frame.domain.share(), frame.timing, self.emitted);
                        self.emitted += 1;
                        out_frame.meta.attach(parsed_frame.analytics);
                        if let Some(unix_nanos) = parsed_frame.unix_nanos {
                            out_frame.meta.attach(WallClockMeta { unix_nanos });
                        }
                        out.push(PipelinePacket::DataFrame(out_frame)).await?;
                    }
                }
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        &[]
    }

    fn set_property(&mut self, _name: &str, _value: PropValue) -> Result<(), PropError> {
        Err(PropError::Unknown)
    }

    fn get_property(&self, _name: &str) -> Option<PropValue> {
        None
    }

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }
}

impl LogSource for OnvifMetadataParse {
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

impl PadTemplates for OnvifMetadataParse {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(Caps::OnvifMetadata)),
            PadTemplate::source(CapsSet::one(Caps::OnvifMetadata)),
        ])
    }
}

// ---- onvifmetadatacombiner ----

/// How long a video frame waits for the metadata that describes it, and how far
/// behind the video a metadata frame may be before it is dropped. Both default
/// to a fifth of a second, enough for a camera's analytics delay without a
/// visible lag in the picture.
const DEFAULT_LATENCY_NS: u64 = 200_000_000;
const DEFAULT_MAX_LATENESS_NS: u64 = 200_000_000;

static COMBINER_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "latency",
        PropKind::Uint,
        "how long a video frame waits for its metadata before going out, nanoseconds",
    )
    .with_default("200000000"),
    PropertySpec::new(
        "max-lateness",
        PropKind::Uint,
        "how far behind the video a metadata frame may arrive before it is dropped, nanoseconds",
    )
    .with_default("200000000"),
];

/// Where an item sits on both clocks the two pads might share.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Instant {
    wall_nanos: Option<i64>,
    pts_ns: u64,
}

impl Instant {
    fn of(frame: &Frame) -> Self {
        Self {
            wall_nanos: frame.meta.get::<WallClockMeta>().map(|w| w.unix_nanos),
            pts_ns: frame.timing.pts_ns,
        }
    }

    /// This instant on the axis the two pads agreed on.
    fn on(self, wall_clock: bool) -> i64 {
        match (wall_clock, self.wall_nanos) {
            (true, Some(wall)) => wall,
            _ => self.pts_ns as i64,
        }
    }
}

/// A video frame held while its metadata is still expected.
#[derive(Debug)]
struct HeldVideo {
    frame: Frame,
    at: Instant,
}

/// Attach an ONVIF analytics stream to the video frames it describes, the
/// `onvifmetadataoverlay` half that does the matching (the drawing is
/// `analyticsoverlay` downstream). Input 0 is the video, whose caps and pixels
/// pass through untouched; input 1 is `onvifmetadataparse` output.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::onvifmetadata::OnvifMetadataCombiner;
///
/// let combiner = OnvifMetadataCombiner::new();
/// ```
#[derive(Debug)]
pub struct OnvifMetadataCombiner {
    video_caps: Option<Caps>,
    /// Video frames waiting for metadata, oldest first.
    held: VecDeque<HeldVideo>,
    /// Metadata not yet matched to a video frame, oldest first.
    pending: VecDeque<(Instant, AnalyticsMeta)>,
    /// The newest instant seen on either pad, the element's idea of now.
    newest: Option<Instant>,
    /// The newest instant seen on the video pad, which is what `max-lateness`
    /// measures a metadata frame against.
    newest_video: Option<Instant>,
    /// Whether a frame on that pad has ever carried a wall clock. The pads are
    /// matched on wall clock once both have: the video's arrives with the first
    /// RTCP sender report, which can be seconds after its first frame.
    video_has_wall_clock: bool,
    metadata_has_wall_clock: bool,
    latency_ns: u64,
    max_lateness_ns: u64,
    log_name: LogName,
}

impl Default for OnvifMetadataCombiner {
    fn default() -> Self {
        Self {
            video_caps: None,
            held: VecDeque::new(),
            pending: VecDeque::new(),
            newest: None,
            newest_video: None,
            video_has_wall_clock: false,
            metadata_has_wall_clock: false,
            latency_ns: DEFAULT_LATENCY_NS,
            max_lateness_ns: DEFAULT_MAX_LATENESS_NS,
            log_name: LogName::default(),
        }
    }
}

impl OnvifMetadataCombiner {
    /// Input pad indices: video on 0, the parsed metadata stream on 1.
    pub const VIDEO: usize = 0;
    pub const METADATA: usize = 1;

    pub fn new() -> Self {
        Self::default()
    }

    /// The axis both pads are compared on. Until both have carried a wall
    /// clock the PTS timeline stands in, which is what a silent metadata pad
    /// leaves in place anyway. A frame without a wall clock is always read on
    /// PTS, so the video frames from before the first sender report go out
    /// plain once the axis switches.
    fn axis(&self) -> bool {
        self.video_has_wall_clock && self.metadata_has_wall_clock
    }

    fn note(&mut self, at: Instant, video: bool) {
        let axis = self.axis();
        for newest in [
            Some(&mut self.newest),
            video.then_some(&mut self.newest_video),
        ]
        .into_iter()
        .flatten()
        {
            let newer = match newest {
                Some(seen) => at.on(axis) > seen.on(axis),
                None => true,
            };
            if newer {
                *newest = Some(at);
            }
        }
    }

    /// Drop metadata that arrived further behind the video than `max-lateness`
    /// allows: the frames it described are already gone.
    fn drop_late_metadata(&mut self) {
        let axis = self.axis();
        let Some(newest_video) = self.newest_video.map(|n| n.on(axis)) else {
            return;
        };
        let cutoff = newest_video.saturating_sub(self.max_lateness_ns as i64);
        while self
            .pending
            .front()
            .is_some_and(|(at, _)| at.on(axis) < cutoff)
        {
            self.pending.pop_front();
        }
    }

    /// The end of the front held frame's window: its own duration when it has
    /// one, else the start of the frame after it, else unknown.
    fn front_window_end(&self, axis: bool) -> Option<i64> {
        let front = self.held.front()?;
        let start = front.at.on(axis);
        if front.frame.timing.duration_ns > 0 {
            return Some(start.saturating_add(front.frame.timing.duration_ns as i64));
        }
        self.held.get(1).map(|next| next.at.on(axis))
    }

    /// Release every held video frame whose wait is up, each carrying the
    /// metadata that falls in its window. `flush` releases them all, which is
    /// what an EOS on either pad calls for.
    async fn release(&mut self, out: &mut dyn OutputSink, flush: bool) -> Result<(), G2gError> {
        let axis = self.axis();
        loop {
            let Some(front) = self.held.front() else {
                return Ok(());
            };
            // A frame from before the first sender report has no place on the
            // wall clock, so nothing can be matched to it.
            if axis && front.at.wall_nanos.is_none() {
                let held = self.held.pop_front().expect("the front frame is present");
                out.push(PipelinePacket::DataFrame(held.frame)).await?;
                continue;
            }
            let start = front.at.on(axis);
            let newest = self.newest.map(|n| n.on(axis)).unwrap_or(start);
            if !flush && newest.saturating_sub(start) < self.latency_ns as i64 {
                return Ok(());
            }
            // With no frame after it and no duration of its own, the window
            // closes where the wait does.
            let end = self
                .front_window_end(axis)
                .unwrap_or_else(|| start.saturating_add(self.latency_ns as i64));
            let mut held = self.held.pop_front().expect("the front frame is present");
            while self
                .pending
                .front()
                .is_some_and(|(at, _)| at.on(axis) < end)
            {
                let (at, analytics) = self.pending.pop_front().expect("front is present");
                if at.on(axis) < start {
                    continue;
                }
                append_analytics(&mut held.frame, &analytics);
            }
            out.push(PipelinePacket::DataFrame(held.frame)).await?;
        }
    }
}

/// Merge one metadata frame's analytics onto a video frame, keeping whatever a
/// detector upstream already wrote. Node indices are offset so the incoming
/// relations still point at the incoming nodes, and the two class-name tables
/// are concatenated with the incoming labels shifted past the existing ones.
fn append_analytics(frame: &mut Frame, incoming: &AnalyticsMeta) {
    let Some(existing) = frame.meta.get_mut::<AnalyticsMeta>() else {
        frame.meta.attach(incoming.clone());
        return;
    };
    let node_offset = existing.nodes.len();
    // A producer that labelled nodes without publishing a table keeps its ids
    // meaningless: the incoming labels start past the highest id it used, and
    // no table is built, since one reaching them could run to u32::MAX names.
    let label_offset = match &existing.class_names {
        Some(names) => names.len() as u32,
        None => highest_label(existing).map_or(0, |label| label.saturating_add(1)),
    };
    let table_lines_up = existing.class_names.is_some() || label_offset == 0;
    if let (true, Some(incoming_names)) = (table_lines_up, &incoming.class_names) {
        let mut names: Vec<String> = existing
            .class_names
            .as_ref()
            .map(|names| names.iter().map(|n| n.to_string()).collect())
            .unwrap_or_default();
        names.extend(incoming_names.iter().map(|n| n.to_string()));
        existing.set_class_names(names);
    }

    for node in &incoming.nodes {
        let mut node = node.clone();
        shift_label(&mut node, label_offset);
        existing.nodes.push(node);
    }
    for relation in &incoming.relations {
        existing.relations.push(g2g_core::meta::Relation {
            from: relation.from + node_offset,
            to: relation.to + node_offset,
            kind: relation.kind,
        });
    }
}

/// The largest class label any node carries, `None` when none do.
fn highest_label(meta: &AnalyticsMeta) -> Option<u32> {
    meta.nodes
        .iter()
        .filter_map(|node| match node {
            AnalyticsNode::Detection(d) => Some(d.label),
            AnalyticsNode::Classification(c) => Some(c.label),
            AnalyticsNode::Segmentation(s) => Some(s.label),
            AnalyticsNode::Roi(r) => Some(r.label),
            AnalyticsNode::Tracking(_) => None,
        })
        // An unclassified object's id names nothing and must not stretch the table.
        .filter(|label| *label != UNCLASSIFIED_LABEL)
        .max()
}

fn shift_label(node: &mut AnalyticsNode, offset: u32) {
    let label = match node {
        AnalyticsNode::Detection(d) => &mut d.label,
        AnalyticsNode::Classification(c) => &mut c.label,
        AnalyticsNode::Segmentation(s) => &mut s.label,
        AnalyticsNode::Roi(r) => &mut r.label,
        AnalyticsNode::Tracking(_) => return,
    };
    if *label != UNCLASSIFIED_LABEL {
        *label = label.saturating_add(offset);
    }
}

impl MultiInputElement for OnvifMetadataCombiner {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        2
    }

    /// The two pads are merged by PTS, so the metadata for a picture reaches the
    /// element around the picture itself rather than a whole queue later.
    fn input_pts_ordered(&self) -> bool {
        true
    }

    fn output_follows_input(&self) -> Option<usize> {
        Some(Self::VIDEO)
    }

    /// Named request pads: `video` -> the video pad, anything text-shaped ->
    /// the metadata pad, so a launch line can wire the branches in either order.
    fn input_pad_index(
        &self,
        req: &g2g_core::runtime::PadRequest,
        _ordinal: usize,
    ) -> Option<usize> {
        match req.kind {
            g2g_core::runtime::PadKind::Video => Some(Self::VIDEO),
            g2g_core::runtime::PadKind::Text => Some(Self::METADATA),
            _ => None,
        }
    }

    fn intercept_caps(&self, input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match input {
            // The pixels are never touched, so the video pad takes any caps.
            Self::VIDEO => Ok(upstream_caps.clone()),
            Self::METADATA => upstream_caps.intersect(&Caps::OnvifMetadata),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn caps_constraint_as_input(&self, input: usize) -> CapsConstraint<'_> {
        match input {
            Self::METADATA => CapsConstraint::Accepts(CapsSet::one(Caps::OnvifMetadata)),
            _ => CapsConstraint::AcceptsAny,
        }
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        match (input, absolute_caps) {
            (Self::VIDEO, _) => {
                self.video_caps = Some(absolute_caps.clone());
                Ok(ConfigureOutcome::Accepted)
            }
            (Self::METADATA, Caps::OnvifMetadata) => Ok(ConfigureOutcome::Accepted),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        self.video_caps.clone().ok_or(G2gError::NotConfigured)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "ONVIF metadata combiner",
            "Filter/Video/Metadata",
            "Attaches an ONVIF analytics stream to the video frames it describes",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        COMBINER_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match (name, value) {
            ("latency", PropValue::Uint(v)) => {
                self.latency_ns = v;
                Ok(())
            }
            ("max-lateness", PropValue::Uint(v)) => {
                self.max_lateness_ns = v;
                Ok(())
            }
            ("latency" | "max-lateness", _) => Err(PropError::Type),
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "latency" => Some(PropValue::Uint(self.latency_ns)),
            "max-lateness" => Some(PropValue::Uint(self.max_lateness_ns)),
            _ => None,
        }
    }

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match (input, packet) {
                (Self::VIDEO, PipelinePacket::DataFrame(frame)) => {
                    let at = Instant::of(&frame);
                    self.video_has_wall_clock |= at.wall_nanos.is_some();
                    self.note(at, true);
                    self.held.push_back(HeldVideo { frame, at });
                    self.drop_late_metadata();
                    self.release(out, false).await
                }
                (Self::METADATA, PipelinePacket::DataFrame(frame)) => {
                    let at = Instant::of(&frame);
                    self.metadata_has_wall_clock |= at.wall_nanos.is_some();
                    self.note(at, false);
                    if let Some(analytics) = frame.meta.get::<AnalyticsMeta>() {
                        self.pending.push_back((at, analytics.clone()));
                    }
                    self.drop_late_metadata();
                    self.release(out, false).await
                }
                // The runner aggregates the per-pad Eos, so what is left is
                // flushed here and the Eos itself is not forwarded.
                (_, PipelinePacket::Eos) => self.release(out, true).await,
                // A caps change (or a flush) describes the frames after it, so
                // the ones still waiting go out in front of it.
                (Self::VIDEO, other) => {
                    self.release(out, true).await?;
                    out.push(other).await.map(|_| ())
                }
                // A caps change on the metadata pad says nothing about the
                // output, which follows the video.
                (_, _) => Ok(()),
            }
        })
    }
}

impl LogSource for OnvifMetadataCombiner {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// ONVIF Analytics Specification 21.12, section 5.1.2.2 (pages 11-12): the
    /// frame system of a 640x480 picture whose origin is its lower-left corner.
    const PICTURE_640X480: CoordinateSystem = CoordinateSystem {
        translate: (-1.0, -1.0),
        scale: (0.003125, 0.00416667),
    };
    /// Enough slack for the f32 arithmetic, far tighter than a pixel of a
    /// 640x480 picture (1/640 = 0.0016).
    const TOLERANCE: f32 = 1e-5;

    fn close(got: f32, want: f32) -> bool {
        (got - want).abs() <= TOLERANCE
    }

    #[test]
    fn a_frame_transformation_maps_pixels_to_the_normalized_system() {
        // The lower-left corner of the picture is the bottom-left of the
        // normalized system, and the upper-right corner is its top-right.
        let (x, y) = PICTURE_640X480.map(0.0, 0.0);
        assert!(close(x, -1.0) && close(y, -1.0), "({x}, {y})");
        let (x, y) = PICTURE_640X480.map(640.0, 480.0);
        assert!(close(x, 1.0) && close(y, 1.0), "({x}, {y})");
    }

    #[test]
    fn an_appearance_transformation_composes_on_the_frame_one() {
        // An object-centric system twice as coarse, shifted 100 units right in
        // the frame's own units.
        let object = PICTURE_640X480.compose((100.0, 0.0), (2.0, 2.0));
        let (x, y) = object.map(10.0, 0.0);
        let (direct_x, direct_y) = PICTURE_640X480.map(100.0 + 10.0 * 2.0, 0.0);
        assert!(close(x, direct_x) && close(y, direct_y), "({x}, {y})");
    }

    #[test]
    fn utc_time_reads_fractional_seconds_and_zones() {
        // The instant the analytics specification's first example frame names
        // (section 5.1.3.1, page 13), then the same instant written three other
        // legal ways.
        const SPEC_EXAMPLE_NANOS: i64 = 1_223_641_497_321_000_000;
        assert_eq!(
            parse_utc_time("2008-10-10T12:24:57.321"),
            Some(SPEC_EXAMPLE_NANOS)
        );
        assert_eq!(
            parse_utc_time("2008-10-10T12:24:57.321Z"),
            Some(SPEC_EXAMPLE_NANOS)
        );
        assert_eq!(
            parse_utc_time("2008-10-10T14:24:57.321+02:00"),
            Some(SPEC_EXAMPLE_NANOS)
        );
        assert_eq!(
            parse_utc_time("2008-10-10T12:24:57.321000000"),
            Some(SPEC_EXAMPLE_NANOS)
        );
        // Whole seconds, and a fraction finer than a nanosecond (truncated).
        assert_eq!(
            parse_utc_time("2008-10-10T12:24:57"),
            Some(SPEC_EXAMPLE_NANOS - 321_000_000)
        );
        assert_eq!(
            parse_utc_time("2008-10-10T12:24:57.3210000009"),
            Some(SPEC_EXAMPLE_NANOS)
        );
    }

    #[test]
    fn a_malformed_utc_time_is_refused() {
        for text in [
            "",
            "2008-10-10",
            "2008-13-10T12:24:57",
            "2008-10-10T25:24:57",
            "2008-10-10T12:61:57",
            "2008-10-10T12:24:57.abc",
            "not a time",
        ] {
            assert_eq!(parse_utc_time(text), None, "{text:?} must not parse");
        }
    }

    #[test]
    fn a_likelihood_is_clamped_and_a_missing_one_means_certain() {
        assert_eq!(parse_likelihood("0.8"), Some(0.8));
        assert_eq!(parse_likelihood(" 1.5 "), Some(1.0));
        assert_eq!(parse_likelihood("-0.2"), Some(0.0));
        assert_eq!(parse_likelihood("NaN"), None);
        assert_eq!(parse_likelihood("high"), None);
    }

    #[test]
    fn a_payload_with_two_declarations_splits_into_two_documents() {
        let text = "<?xml version=\"1.0\"?><a/>\n<?xml version=\"1.0\"?><b/>";
        let chunks = split_xml_documents(text);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].contains("<a/>") && chunks[1].contains("<b/>"));
        // One document is one chunk, declaration or not.
        assert_eq!(split_xml_documents("<a/>").len(), 1);
        assert_eq!(split_xml_documents("<?xml version=\"1.0\"?><a/>").len(), 1);
    }
}
