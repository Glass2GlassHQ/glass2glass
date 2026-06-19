//! `gst-launch`-style text pipeline parser (M106): turn
//! `"videotestsrc num-buffers=3 ! videoflip method=rotate-180 ! fakesink"` into a
//! runnable [`Graph`], the front door that makes g2g usable without hand-writing
//! Rust for every pipeline.
//!
//! Built on the M104 property system and the M105 by-name registry: each `!`
//! separated stage is `element-name key=value ...`; the parser constructs the
//! element by name from the [`Registry`], looks up each property's
//! [`PropKind`](crate::PropKind) to parse its textual value, and applies it. The
//! first stage is the source, the last is the sink, the middle are transforms,
//! linked in order. The result drops straight onto
//! [`run_graph`](crate::runtime::run_graph).
//!
//! Scope (v1): a single linear chain; `key=value` with no spaces in the value
//! (double quotes around a value are stripped). Branching (`tee`/named pads) and
//! caps-filter string syntax are follow-ups.

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::element::DynAsyncElement;
use crate::graph::{Graph, GraphError};
use crate::property::PropValue;
use crate::runtime::autoplug::Registry;
use crate::runtime::{DynSourceLoop, GraphNode, GraphNodeRef};

/// Why [`parse_launch`] could not build a graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// The pipeline string was empty or all whitespace.
    Empty,
    /// A stage between `!` separators had no element name.
    EmptyStage,
    /// Fewer than two stages: a runnable pipeline needs at least a source and a
    /// sink.
    TooFewStages,
    /// The first stage names no registered source.
    UnknownSource(String),
    /// A non-first stage names no registered transform / sink.
    UnknownElement(String),
    /// A property token had no `=` (expected `key=value`).
    MalformedProperty { element: String, token: String },
    /// The element has no property of that name.
    UnknownProperty { element: String, key: String },
    /// The value did not parse for the property's kind, or was rejected.
    BadValue { element: String, key: String, value: String },
    /// Linking two stages into the graph failed.
    Graph(GraphError),
}

impl From<GraphError> for ParseError {
    fn from(e: GraphError) -> Self {
        ParseError::Graph(e)
    }
}

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ParseError::Empty => f.write_str("empty pipeline"),
            ParseError::EmptyStage => f.write_str("empty stage between '!' separators"),
            ParseError::TooFewStages => f.write_str("pipeline needs at least a source and a sink"),
            ParseError::UnknownSource(n) => write!(f, "unknown source element: {n}"),
            ParseError::UnknownElement(n) => write!(f, "unknown element: {n}"),
            ParseError::MalformedProperty { element, token } => {
                write!(f, "{element}: malformed property '{token}' (expected key=value)")
            }
            ParseError::UnknownProperty { element, key } => {
                write!(f, "{element}: no property named '{key}'")
            }
            ParseError::BadValue { element, key, value } => {
                write!(f, "{element}: invalid value '{value}' for property '{key}'")
            }
            ParseError::Graph(e) => write!(f, "graph link error: {e:?}"),
        }
    }
}

/// One parsed stage: the element name and its `key=value` properties, all owned
/// so error messages can name them.
struct Stage {
    name: String,
    props: Vec<(String, String)>,
}

/// Split a `gst-launch` pipeline string into stages, each an element name plus
/// its parsed `key=value` properties.
fn parse_stages(pipeline: &str) -> Result<Vec<Stage>, ParseError> {
    let trimmed = pipeline.trim();
    if trimmed.is_empty() {
        return Err(ParseError::Empty);
    }
    let mut stages = Vec::new();
    for raw in trimmed.split('!') {
        let mut tokens = raw.split_whitespace();
        let name = tokens.next().ok_or(ParseError::EmptyStage)?.to_string();
        if name.is_empty() {
            return Err(ParseError::EmptyStage);
        }
        // A bare caps description (`video/x-raw,format=nv12,width=320`, a single
        // token with a media-type `/`) is the gst-launch shorthand for a
        // capsfilter; rewrite it to that element with the whole token as `caps`.
        if name.contains('/') {
            stages.push(Stage { name: "capsfilter".to_string(), props: alloc::vec![("caps".to_string(), name)] });
            continue;
        }
        let mut props = Vec::new();
        for tok in tokens {
            let (key, value) = tok.split_once('=').ok_or_else(|| ParseError::MalformedProperty {
                element: name.clone(),
                token: tok.to_string(),
            })?;
            // Strip a single layer of surrounding double quotes from the value.
            let value = value.strip_prefix('"').and_then(|v| v.strip_suffix('"')).unwrap_or(value);
            props.push((key.to_string(), value.to_string()));
        }
        stages.push(Stage { name, props });
    }
    Ok(stages)
}

/// Apply parsed `key=value` props to a source, parsing each value for its
/// declared [`PropKind`](crate::PropKind).
fn apply_source_props(
    el: &mut Box<dyn DynSourceLoop>,
    name: &str,
    props: &[(String, String)],
) -> Result<(), ParseError> {
    for (key, value) in props {
        let kind = el
            .properties()
            .iter()
            .find(|s| s.name == key)
            .ok_or_else(|| ParseError::UnknownProperty { element: name.into(), key: key.clone() })?
            .kind;
        let parsed = PropValue::parse(kind, value).map_err(|_| ParseError::BadValue {
            element: name.into(),
            key: key.clone(),
            value: value.clone(),
        })?;
        el.set_property(key, parsed).map_err(|_| ParseError::BadValue {
            element: name.into(),
            key: key.clone(),
            value: value.clone(),
        })?;
    }
    Ok(())
}

/// Apply parsed `key=value` props to a transform / sink element.
fn apply_element_props(
    el: &mut Box<dyn DynAsyncElement>,
    name: &str,
    props: &[(String, String)],
) -> Result<(), ParseError> {
    for (key, value) in props {
        let kind = el
            .properties()
            .iter()
            .find(|s| s.name == key)
            .ok_or_else(|| ParseError::UnknownProperty { element: name.into(), key: key.clone() })?
            .kind;
        let parsed = PropValue::parse(kind, value).map_err(|_| ParseError::BadValue {
            element: name.into(),
            key: key.clone(),
            value: value.clone(),
        })?;
        el.set_property(key, parsed).map_err(|_| ParseError::BadValue {
            element: name.into(),
            key: key.clone(),
            value: value.clone(),
        })?;
    }
    Ok(())
}

/// Parse a `gst-launch`-style pipeline string into a runnable [`Graph`], building
/// each stage by name from `registry` and applying its `key=value` properties.
///
/// The first stage is the source, the last the sink, the middle transforms,
/// linked in order. The graph is ready for
/// [`run_graph`](crate::runtime::run_graph).
///
/// ```text
/// videotestsrc num-buffers=3 ! videoflip method=rotate-180 ! fakesink
/// ```
pub fn parse_launch(registry: &Registry, pipeline: &str) -> Result<Graph<GraphNode>, ParseError> {
    let stages = parse_stages(pipeline)?;
    if stages.len() < 2 {
        return Err(ParseError::TooFewStages);
    }

    let mut graph: Graph<GraphNode> = Graph::new();

    // First stage: the source.
    let head = &stages[0];
    let mut source = registry
        .make_source(&head.name)
        .ok_or_else(|| ParseError::UnknownSource(head.name.clone()))?;
    apply_source_props(&mut source, &head.name, &head.props)?;
    let mut prev = graph.add_source(GraphNodeRef::Source(source));

    // Interior stages: transforms.
    let last = stages.len() - 1;
    for stage in &stages[1..last] {
        let mut el = registry
            .make_element(&stage.name)
            .ok_or_else(|| ParseError::UnknownElement(stage.name.clone()))?;
        apply_element_props(&mut el, &stage.name, &stage.props)?;
        let node = graph.add_transform(GraphNodeRef::Element(el));
        graph.link(prev, node)?;
        prev = node;
    }

    // Last stage: the sink.
    let tail = &stages[last];
    let mut sink = registry
        .make_element(&tail.name)
        .ok_or_else(|| ParseError::UnknownElement(tail.name.clone()))?;
    apply_element_props(&mut sink, &tail.name, &tail.props)?;
    let sink_node = graph.add_sink(GraphNodeRef::Element(sink));
    graph.link(prev, sink_node)?;

    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_stages_splits_names_and_props() {
        let stages =
            parse_stages("videotestsrc num-buffers=3 pattern=snow ! videoflip method=rotate-180 ! fakesink")
                .unwrap();
        assert_eq!(stages.len(), 3);
        assert_eq!(stages[0].name, "videotestsrc");
        assert_eq!(
            stages[0].props,
            [("num-buffers".to_string(), "3".to_string()), ("pattern".into(), "snow".into())]
        );
        assert_eq!(stages[1].name, "videoflip");
        assert_eq!(stages[2].name, "fakesink");
        assert!(stages[2].props.is_empty());
    }

    #[test]
    fn parse_stages_strips_quoted_values() {
        // A double-quoted value (no spaces) has its quotes stripped. Values with
        // spaces are a known v1 gap (the whitespace tokenizer would split them).
        let stages = parse_stages("filesrc location=\"file.mp4\" ! fakesink").unwrap();
        assert_eq!(stages[0].props[0], ("location".to_string(), "file.mp4".to_string()));
    }

    #[test]
    fn empty_and_too_few_stages_error() {
        let reg = Registry::new();
        assert!(matches!(parse_launch(&reg, "   "), Err(ParseError::Empty)));
        assert!(matches!(parse_launch(&reg, "videotestsrc"), Err(ParseError::TooFewStages)));
    }

    #[test]
    fn malformed_property_is_reported() {
        let stages = parse_stages("videotestsrc bogus ! fakesink");
        assert!(matches!(stages, Err(ParseError::MalformedProperty { .. })));
    }

    #[test]
    fn caps_description_becomes_capsfilter() {
        // A bare `media/type,...` stage is the inline caps-filter shorthand.
        let stages =
            parse_stages("videotestsrc ! video/x-raw,format=nv12,width=320 ! fakesink").unwrap();
        assert_eq!(stages.len(), 3);
        assert_eq!(stages[1].name, "capsfilter");
        assert_eq!(
            stages[1].props,
            [("caps".to_string(), "video/x-raw,format=nv12,width=320".to_string())]
        );
    }
}
