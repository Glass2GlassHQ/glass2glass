//! Declarative graph format (M578): build a runnable [`Graph`] from a JSON or
//! YAML document, the structured sibling of the `gst-launch` text parser
//! ([`parse_launch`]). A launch string is the ergonomic one-liner; a declarative
//! document is the version-controllable, tool-generated, comment-carrying form a
//! config file or an orchestrator emits.
//!
//! The document is a list of `nodes` and `edges`:
//!
//! ```yaml
//! nodes:
//!   - { id: src,  element: videotestsrc, props: { num-buffers: 30 } }
//!   - { id: cf,   caps: "video/x-raw,format=NV12" }   # a capsfilter shorthand
//!   - { id: sink, element: autovideosink }
//! edges:
//!   - { from: src, to: cf }
//!   - { from: cf,  to: sink }
//! ```
//!
//! Roles follow connectivity exactly as in the launch parser: a node with no
//! inbound edge is a source, one with no outbound edge a sink, one with several
//! inbound edges a muxer (built from the registry's `MuxerFactory`), and a node
//! feeding several consumers gets an implicit `tee` spliced onto its output (the
//! same auto-tee, M473). An explicit `element: tee` node is honored too, and a
//! registered demuxer fans out on its own pads. Per-edge backpressure is a
//! `policy` string (`block` / `drop-oldest` / `drop-newest`, the [`LinkPolicy`]
//! names) with an optional `capacity` (the per-edge channel depth).
//!
//! Property values are typed by the target element's [`PropertySpec`] table and
//! parsed with [`PropValue::parse`], the *same* path the launch parser uses, so a
//! `num-buffers: 30` in JSON and `num-buffers=30` in a launch string mean exactly
//! the same thing (no second, divergent JSON-number coercion). A top-level
//! `pipeline:` string is an escape hatch that just defers to [`parse_launch`].
//!
//! This is also the shared spec model the Rhai builder ([`crate::script`], M579)
//! emits into, so a script and a static document reach the graph through one
//! builder.

use std::collections::BTreeMap;
use std::string::{String, ToString};
use std::vec::Vec;

use g2g_core::runtime::{parse_launch, GraphNode, GraphNodeRef, ParseError, Registry};
use g2g_core::{
    Graph, GraphError, LinkPolicy, NodeId, PadId, PropError, PropValue, PropertySpec,
};
use serde::{Deserialize, Serialize};

/// A format-agnostic scalar property value. Deserializes the same from JSON and
/// YAML (both hand us bool / integer / float / string), then converts to the
/// textual form [`PropValue::parse`] consumes, so the type is fixed by the target
/// element's [`PropertySpec`], not by how the document happened to spell it.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ScalarVal {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

impl ScalarVal {
    /// Render as the textual value the property system parses. A string is taken
    /// verbatim (so a `"30/1"` fraction or a `"nv12"` enum survives); the others
    /// print their literal form.
    fn to_prop_text(&self) -> String {
        match self {
            ScalarVal::Bool(b) => b.to_string(),
            ScalarVal::Int(i) => i.to_string(),
            ScalarVal::Float(f) => f.to_string(),
            ScalarVal::Str(s) => s.clone(),
        }
    }
}

/// One node: a `id` handle, the `element` to build (or a `caps` shorthand that
/// builds a `capsfilter`), and its `props`.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct NodeSpec {
    /// Unique handle referenced by [`EdgeSpec::from`] / [`EdgeSpec::to`].
    pub id: String,
    /// Registered element name. Optional only when `caps` is set (then it
    /// defaults to `capsfilter`).
    #[serde(default)]
    pub element: Option<String>,
    /// `key=value` properties, typed by the element's [`PropertySpec`] table.
    #[serde(default)]
    pub props: BTreeMap<String, ScalarVal>,
    /// Caps-filter shorthand: a `video/x-raw,...` string. Sets the `caps`
    /// property of a `capsfilter` (the node's `element` must be absent or
    /// `capsfilter`), the declarative analog of the launch bare-caps token.
    #[serde(default)]
    pub caps: Option<String>,
}

/// One directed edge from node `from`'s output to node `to`'s input, with an
/// optional backpressure `policy` / `capacity` and explicit pad indices (needed
/// only to override the default positional assignment on a tee/demux/muxer).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct EdgeSpec {
    pub from: String,
    pub to: String,
    /// Output pad index on `from` (defaults to the next free tee/demux pad, else 0).
    #[serde(default)]
    pub from_pad: Option<u8>,
    /// Input pad index on `to` (defaults to the next free muxer pad, else 0).
    #[serde(default)]
    pub to_pad: Option<u8>,
    /// `block` (default) / `drop-oldest` / `drop-newest`, the [`LinkPolicy`] names.
    #[serde(default)]
    pub policy: Option<String>,
    /// Per-edge channel depth; `None` uses the runner's graph-wide capacity.
    #[serde(default)]
    pub capacity: Option<usize>,
}

/// A whole graph document: either a `pipeline` launch-string escape hatch, or a
/// `nodes` + `edges` description.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct GraphSpec {
    /// A `gst-launch` string. When set, the whole document is parsed by
    /// [`parse_launch`] and `nodes` / `edges` are ignored.
    #[serde(default)]
    pub pipeline: Option<String>,
    #[serde(default)]
    pub nodes: Vec<NodeSpec>,
    #[serde(default)]
    pub edges: Vec<EdgeSpec>,
}

/// Why a document could not be turned into a [`Graph`].
#[derive(Debug)]
pub enum SpecError {
    /// The document deserialized but was empty (no `pipeline`, no `nodes`).
    Empty,
    /// The `pipeline:` escape hatch failed to parse.
    Parse(ParseError),
    /// Two nodes share the same `id`.
    DuplicateId(String),
    /// An edge references an `id` no node declares.
    UnknownReference(String),
    /// A node has neither an `element` nor a `caps` shorthand.
    MissingElement(String),
    /// A `caps` shorthand on a node whose `element` is not `capsfilter`.
    CapsOnNonCapsfilter(String),
    /// A tee node carries properties (the structural tee has none).
    TeeWithProperties(String),
    /// A source-position element (no inbound edge) names no registered source.
    UnknownSource(String),
    /// A transform / sink / demux names no registered element.
    UnknownElement(String),
    /// A node with several inbound edges names no registered muxer.
    NotAMuxer(String),
    /// The element has no property of that name.
    UnknownProperty { node: String, key: String },
    /// A value did not parse for the property's kind, or was rejected.
    BadValue { node: String, key: String, value: String },
    /// An edge `policy` was not one of the [`LinkPolicy`] names.
    BadPolicy(String),
    /// Linking the nodes into the graph failed.
    Graph(GraphError),
    /// The JSON / YAML did not deserialize (the message is the parser's).
    Deserialize(String),
}

impl From<GraphError> for SpecError {
    fn from(e: GraphError) -> Self {
        SpecError::Graph(e)
    }
}

impl core::fmt::Display for SpecError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SpecError::Empty => write!(f, "empty graph document (no `pipeline` or `nodes`)"),
            SpecError::Parse(e) => write!(f, "pipeline parse error: {e}"),
            SpecError::DuplicateId(id) => write!(f, "duplicate node id '{id}'"),
            SpecError::UnknownReference(id) => write!(f, "edge references unknown node '{id}'"),
            SpecError::MissingElement(id) => {
                write!(f, "node '{id}' has neither `element` nor `caps`")
            }
            SpecError::CapsOnNonCapsfilter(id) => {
                write!(f, "node '{id}': `caps` is only valid on a capsfilter")
            }
            SpecError::TeeWithProperties(id) => write!(f, "tee node '{id}' takes no properties"),
            SpecError::UnknownSource(n) => write!(f, "unknown source element '{n}'"),
            SpecError::UnknownElement(n) => write!(f, "unknown element '{n}'"),
            SpecError::NotAMuxer(n) => {
                write!(f, "'{n}' has several inputs but is not a registered muxer")
            }
            SpecError::UnknownProperty { node, key } => {
                write!(f, "node '{node}': no property '{key}'")
            }
            SpecError::BadValue { node, key, value } => {
                write!(f, "node '{node}': bad value '{value}' for '{key}'")
            }
            SpecError::BadPolicy(p) => {
                write!(f, "bad edge policy '{p}' (want block / drop-oldest / drop-newest)")
            }
            SpecError::Graph(e) => write!(f, "graph error: {e:?}"),
            SpecError::Deserialize(m) => write!(f, "deserialize error: {m}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SpecError {}

/// The element name a node resolves to: its `element`, or `capsfilter` when only
/// a `caps` shorthand is given.
fn resolved_name(node: &NodeSpec) -> Result<String, SpecError> {
    match (&node.element, &node.caps) {
        (Some(e), Some(_)) if e != "capsfilter" => {
            Err(SpecError::CapsOnNonCapsfilter(node.id.clone()))
        }
        (Some(e), _) => Ok(e.clone()),
        (None, Some(_)) => Ok("capsfilter".to_string()),
        (None, None) => Err(SpecError::MissingElement(node.id.clone())),
    }
}

/// The effective `key=value` list for a node: its `props` plus the `caps`
/// shorthand folded in as a `caps` property (last, so it wins over an explicit
/// `props.caps`).
fn effective_props(node: &NodeSpec) -> Vec<(String, ScalarVal)> {
    let mut kv: Vec<(String, ScalarVal)> =
        node.props.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    if let Some(caps) = &node.caps {
        kv.retain(|(k, _)| k != "caps");
        kv.push(("caps".to_string(), ScalarVal::Str(caps.clone())));
    }
    kv
}

/// Apply a node's typed properties to a freshly-built element, looking each key
/// up in the element's [`PropertySpec`] table for its [`PropKind`] and parsing
/// the textual value exactly as the launch parser does. `specs` is `&'static`, so
/// reading it does not borrow the element the `set` closure mutates.
fn apply_props(
    specs: &'static [PropertySpec],
    node_id: &str,
    kv: &[(String, ScalarVal)],
    mut set: impl FnMut(&str, PropValue) -> Result<(), PropError>,
) -> Result<(), SpecError> {
    for (key, val) in kv {
        let kind = specs
            .iter()
            .find(|s| s.name == key)
            .ok_or_else(|| SpecError::UnknownProperty { node: node_id.into(), key: key.clone() })?
            .kind;
        let text = val.to_prop_text();
        let parsed = PropValue::parse(kind, &text).map_err(|_| SpecError::BadValue {
            node: node_id.into(),
            key: key.clone(),
            value: text.clone(),
        })?;
        set(key, parsed).map_err(|_| SpecError::BadValue {
            node: node_id.into(),
            key: key.clone(),
            value: text,
        })?;
    }
    Ok(())
}

/// Map an edge `policy` string to a [`LinkPolicy`]. Accepts the g2g names
/// (`block` / `drop-oldest` / `drop-newest`) plus the gst `leaky=` nicks
/// (`upstream` -> `DropNewest`, `downstream` -> `DropOldest`), so a launch habit
/// carries over. Absent / empty is the lossless `Block` default.
fn parse_policy(policy: &Option<String>) -> Result<LinkPolicy, SpecError> {
    match policy.as_deref().map(str::trim).unwrap_or("") {
        "" | "block" => Ok(LinkPolicy::Block),
        "drop-oldest" | "downstream" => Ok(LinkPolicy::DropOldest),
        "drop-newest" | "upstream" => Ok(LinkPolicy::DropNewest),
        other => Err(SpecError::BadPolicy(other.to_string())),
    }
}

/// Build a runnable [`Graph`] from an already-deserialized [`GraphSpec`], using
/// `registry` to construct each element by name. This is the shared builder the
/// JSON / YAML front-ends and the Rhai script builder all reach the graph
/// through. Roles follow link degree, mirroring [`parse_launch`].
pub fn build_spec(registry: &Registry, spec: &GraphSpec) -> Result<Graph<GraphNode>, SpecError> {
    if let Some(pipeline) = &spec.pipeline {
        return parse_launch(registry, pipeline).map_err(SpecError::Parse);
    }
    if spec.nodes.is_empty() {
        return Err(SpecError::Empty);
    }

    // id -> node index, rejecting duplicates.
    let mut index: BTreeMap<&str, usize> = BTreeMap::new();
    for (i, node) in spec.nodes.iter().enumerate() {
        if index.insert(node.id.as_str(), i).is_some() {
            return Err(SpecError::DuplicateId(node.id.clone()));
        }
    }

    // Resolve edge endpoints to node indices, and tally in/out degree.
    let n = spec.nodes.len();
    let mut in_deg = std::vec![0usize; n];
    let mut out_deg = std::vec![0usize; n];
    let mut edge_idx: Vec<(usize, usize)> = Vec::with_capacity(spec.edges.len());
    for edge in &spec.edges {
        let s = *index
            .get(edge.from.as_str())
            .ok_or_else(|| SpecError::UnknownReference(edge.from.clone()))?;
        let d = *index
            .get(edge.to.as_str())
            .ok_or_else(|| SpecError::UnknownReference(edge.to.clone()))?;
        out_deg[s] += 1;
        in_deg[d] += 1;
        edge_idx.push((s, d));
    }

    // Construct nodes in declaration order so `node_of[i]` lines up with index i.
    let mut graph: Graph<GraphNode> = Graph::new();
    let mut node_of: Vec<NodeId> = Vec::with_capacity(n);
    let mut names: Vec<String> = Vec::with_capacity(n);
    for (i, node) in spec.nodes.iter().enumerate() {
        let name = resolved_name(node)?;
        let kv = effective_props(node);
        let nid = if name == "tee" {
            if !kv.is_empty() {
                return Err(SpecError::TeeWithProperties(node.id.clone()));
            }
            graph.add_tee(out_deg[i] as u8).node()
        } else if in_deg[i] == 0 {
            let mut src = registry
                .make_source(&name)
                .ok_or_else(|| SpecError::UnknownSource(name.clone()))?;
            let specs = src.properties();
            apply_props(specs, &node.id, &kv, |k, v| src.set_property(k, v))?;
            graph.add_source(GraphNodeRef::Source(src))
        } else if in_deg[i] > 1 {
            let mut mux = registry
                .make_muxer(&name, in_deg[i])
                .ok_or_else(|| SpecError::NotAMuxer(name.clone()))?;
            let specs = mux.properties();
            apply_props(specs, &node.id, &kv, |k, v| mux.set_property(k, v))?;
            graph.add_muxer(GraphNodeRef::Muxer(mux), in_deg[i] as u8).node()
        } else if registry.is_demux(&name) {
            let mut demux = registry
                .make_demux(&name, out_deg[i])
                .ok_or_else(|| SpecError::UnknownElement(name.clone()))?;
            let specs = demux.properties();
            apply_props(specs, &node.id, &kv, |k, v| demux.set_property(k, v))?;
            graph.add_demux(GraphNodeRef::Demux(demux), out_deg[i] as u8).node()
        } else if out_deg[i] == 0 {
            let mut el = registry
                .make_element(&name)
                .ok_or_else(|| SpecError::UnknownElement(name.clone()))?;
            let specs = el.properties();
            apply_props(specs, &node.id, &kv, |k, v| el.set_property(k, v))?;
            graph.add_sink(GraphNodeRef::Element(el))
        } else {
            let mut el = registry
                .make_element(&name)
                .ok_or_else(|| SpecError::UnknownElement(name.clone()))?;
            let specs = el.properties();
            apply_props(specs, &node.id, &kv, |k, v| el.set_property(k, v))?;
            graph.add_transform(GraphNodeRef::Element(el))
        };
        node_of.push(nid);
        names.push(name);
    }

    // Auto-tee (M473): a plain node feeding several consumers gets an implicit
    // tee on its output; its consumers then source from the tee's pads.
    let mut implicit_tee: Vec<Option<NodeId>> = std::vec![None; n];
    for i in 0..n {
        if out_deg[i] > 1 && names[i] != "tee" && !registry.is_demux(&names[i]) {
            let tee = graph.add_tee(out_deg[i] as u8).node();
            graph.link_with(PadId::from(node_of[i]), PadId::from(tee), LinkPolicy::Block)?;
            implicit_tee[i] = Some(tee);
        }
    }

    // Wire edges. Fan-out pads (tee / demux / implicit tee) and muxer input pads
    // are assigned in edge order unless an explicit `from_pad` / `to_pad` overrides.
    let mut tee_next = std::vec![0u8; n];
    let mut mux_next = std::vec![0u8; n];
    for (edge, &(s, d)) in spec.edges.iter().zip(edge_idx.iter()) {
        let src = if let Some(tee) = implicit_tee[s] {
            let index = edge.from_pad.unwrap_or_else(|| next(&mut tee_next[s]));
            PadId { node: tee, index }
        } else if names[s] == "tee" || registry.is_demux(&names[s]) {
            let index = edge.from_pad.unwrap_or_else(|| next(&mut tee_next[s]));
            PadId { node: node_of[s], index }
        } else {
            PadId { node: node_of[s], index: edge.from_pad.unwrap_or(0) }
        };
        let dst = if in_deg[d] > 1 {
            let index = edge.to_pad.unwrap_or_else(|| next(&mut mux_next[d]));
            PadId { node: node_of[d], index }
        } else {
            PadId { node: node_of[d], index: edge.to_pad.unwrap_or(0) }
        };
        graph.link_full(src, dst, parse_policy(&edge.policy)?, edge.capacity)?;
    }

    Ok(graph)
}

/// Post-increment a running pad counter, returning the pre-increment value.
fn next(counter: &mut u8) -> u8 {
    let v = *counter;
    *counter += 1;
    v
}

/// Build a graph from a JSON document.
pub fn from_json(registry: &Registry, json: &str) -> Result<Graph<GraphNode>, SpecError> {
    let spec: GraphSpec =
        serde_json::from_str(json).map_err(|e| SpecError::Deserialize(e.to_string()))?;
    build_spec(registry, &spec)
}

/// Build a graph from a YAML document.
#[cfg(feature = "declarative-yaml")]
pub fn from_yaml(registry: &Registry, yaml: &str) -> Result<Graph<GraphNode>, SpecError> {
    let spec: GraphSpec =
        serde_yaml::from_str(yaml).map_err(|e| SpecError::Deserialize(e.to_string()))?;
    build_spec(registry, &spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::default_registry;

    #[test]
    fn json_linear_pipeline_builds_and_validates() {
        let json = r#"{
            "nodes": [
                { "id": "src",  "element": "videotestsrc", "props": { "num-buffers": 3 } },
                { "id": "sink", "element": "fakesink" }
            ],
            "edges": [ { "from": "src", "to": "sink" } ]
        }"#;
        let reg = default_registry();
        let graph = from_json(&reg, json).expect("build");
        // Two real nodes, one edge, and it passes the runner's validation.
        assert_eq!(graph.edges().len(), 1);
        graph.finish().expect("valid DAG");
    }

    #[test]
    fn caps_shorthand_becomes_a_capsfilter() {
        let json = r#"{
            "nodes": [
                { "id": "src", "element": "videotestsrc", "props": { "num-buffers": 1 } },
                { "id": "cf",  "caps": "video/x-raw,format=NV12" },
                { "id": "sink","element": "fakesink" }
            ],
            "edges": [
                { "from": "src", "to": "cf" },
                { "from": "cf",  "to": "sink" }
            ]
        }"#;
        let reg = default_registry();
        let graph = from_json(&reg, json).expect("build");
        assert_eq!(graph.edges().len(), 2, "src->cf->sink");
        graph.finish().expect("valid DAG");
    }

    #[test]
    fn auto_tee_splices_on_fan_out() {
        // One source feeding two sinks: the builder must splice an implicit tee,
        // yielding src->tee plus two tee->sink edges (3 total).
        let json = r#"{
            "nodes": [
                { "id": "src", "element": "videotestsrc", "props": { "num-buffers": 1 } },
                { "id": "a",   "element": "fakesink" },
                { "id": "b",   "element": "fakesink" }
            ],
            "edges": [
                { "from": "src", "to": "a" },
                { "from": "src", "to": "b" }
            ]
        }"#;
        let reg = default_registry();
        let graph = from_json(&reg, json).expect("build");
        assert_eq!(graph.edges().len(), 3, "src->tee + two tee->sink");
        graph.finish().expect("valid DAG");
    }

    #[test]
    fn pipeline_escape_hatch_defers_to_parse_launch() {
        let json = r#"{ "pipeline": "videotestsrc num-buffers=2 ! fakesink" }"#;
        let reg = default_registry();
        let graph = from_json(&reg, json).expect("build");
        assert_eq!(graph.edges().len(), 1);
        graph.finish().expect("valid DAG");
    }

    #[test]
    fn per_edge_policy_and_capacity_apply() {
        let json = r#"{
            "nodes": [
                { "id": "src",  "element": "videotestsrc", "props": { "num-buffers": 1 } },
                { "id": "sink", "element": "fakesink" }
            ],
            "edges": [ { "from": "src", "to": "sink", "policy": "drop-oldest", "capacity": 2 } ]
        }"#;
        let reg = default_registry();
        let graph = from_json(&reg, json).expect("build");
        let e = graph.edges()[0];
        assert_eq!(e.policy, LinkPolicy::DropOldest);
        assert_eq!(e.capacity, Some(2));
    }

    #[test]
    fn unknown_element_is_reported() {
        let json = r#"{
            "nodes": [
                { "id": "src",  "element": "nosuchsrc" },
                { "id": "sink", "element": "fakesink" }
            ],
            "edges": [ { "from": "src", "to": "sink" } ]
        }"#;
        let reg = default_registry();
        assert!(matches!(from_json(&reg, json), Err(SpecError::UnknownSource(_))));
    }

    #[test]
    fn bad_property_value_is_reported() {
        let json = r#"{
            "nodes": [
                { "id": "src",  "element": "videotestsrc", "props": { "num-buffers": "lots" } },
                { "id": "sink", "element": "fakesink" }
            ],
            "edges": [ { "from": "src", "to": "sink" } ]
        }"#;
        let reg = default_registry();
        assert!(matches!(from_json(&reg, json), Err(SpecError::BadValue { .. })));
    }
}
