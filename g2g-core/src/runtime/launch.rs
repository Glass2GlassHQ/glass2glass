//! `gst-launch`-style text pipeline parser (M106, M117, M118): turn
//! `"videotestsrc num-buffers=3 ! videoflip method=rotate-180 ! fakesink"` into a
//! runnable [`Graph`], the front door that makes g2g usable without hand-writing
//! Rust for every pipeline.
//!
//! Built on the M104 property system and the M105 by-name registry: each `!`
//! separated node is `element-name key=value ...`; the parser constructs the
//! element by name from the [`Registry`], looks up each property's
//! [`PropKind`](crate::PropKind) to parse its textual value, and applies it.
//! Roles follow connectivity: an element with no incoming link is a source, one
//! with no outgoing link a sink, the rest transforms (so a linear chain is still
//! source -> transforms -> sink).
//!
//! Branching (M118): `tee name=t` fans one output to many. A `tee` is the
//! structural fan-out node (no element), its output width derived from how many
//! branches reference it; a branch is a `t.` pad reference that starts a chain (a
//! head ref, linking *from* the named element) or, right after a `!`, ends one (a
//! tail ref, linking *into* it). So
//! `videotestsrc ! tee name=t ! fakesink   t. ! fakesink` broadcasts each frame
//! to two sinks. The caps shorthand (`! video/x-raw,format=nv12,... !`, M117) is
//! a bare media-type node rewritten to a `capsfilter`.
//!
//! Fan-in (M122): an element with several inbound links is a muxer, built from
//! the registry's [`MuxerFactory`](crate::runtime::MuxerFactory) with that input
//! count (so its `input_count` matches the node's pads). Each feeding chain ends
//! with a `m.` tail ref, so
//! `src1 ! m.   src2 ! m.   funnel name=m ! fakesink` joins two streams. Feeding
//! chains come first (a new chain can only begin after a `!` / ref / caps
//! boundary, so a chain starting with a bare element name would be read as the
//! previous element's property); the muxer chain is last. A muxer has one output
//! pad, so it must feed a downstream consumer.
//!
//! Scope: `key=value` with no spaces in the value (double quotes around a value
//! are stripped). Muxer `key=value` properties are not applied (the in-tree
//! muxers have none; `name=` is still the handle). A pad-name suffix on a
//! reference (`t.src_0`) is accepted but ignored (pads are positional).

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::caps::Caps;
use crate::element::DynAsyncElement;
use crate::graph::{Graph, GraphError, NodeId, PadId};
use crate::link::LinkPolicy;
use crate::property::PropValue;
use crate::runtime::autoplug::{is_raw_audio, is_raw_video, Registry, UriError};
use crate::runtime::{DynSourceLoop, GraphNode, GraphNodeRef};

/// Why [`parse_launch`] could not build a graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// The pipeline string was empty or all whitespace.
    Empty,
    /// A node between `!` separators had no element name.
    EmptyStage,
    /// Fewer than two elements: a runnable pipeline needs at least a source and a
    /// sink.
    TooFewStages,
    /// A source-position element names no registered source.
    UnknownSource(String),
    /// A transform / sink-position element names no registered element.
    UnknownElement(String),
    /// A property token had no `=` (expected `key=value`).
    MalformedProperty { element: String, token: String },
    /// The element has no property of that name.
    UnknownProperty { element: String, key: String },
    /// The value did not parse for the property's kind, or was rejected.
    BadValue { element: String, key: String, value: String },
    /// A `name.` reference names no element declared with that `name=`.
    UnknownReference(String),
    /// Two elements share the same `name=` handle.
    DuplicateName(String),
    /// A non-`tee` element's output links more than once: a pad peers with one
    /// other pad, so an explicit `tee` must express the fan-out.
    FanOutWithoutTee(String),
    /// More than one link feeds an element's input, but it names no registered
    /// muxer: fan-in needs a [`MuxerFactory`](crate::runtime::MuxerFactory).
    NotAMuxer(String),
    /// A muxer (an element with several inputs) has no outgoing link; its single
    /// output pad must feed a downstream consumer.
    MuxerWithoutOutput(String),
    /// A `queue` / `queue2` sits anywhere but a 1-in/1-out position. It is not an
    /// element in g2g (it collapses into the edge's backpressure policy), so it
    /// cannot be a source, a sink, or a fan-out / fan-in node.
    QueueRole(String),
    /// A `decodebin` has no upstream element to take its input caps from (it was
    /// the first element, or followed a bare `name.` reference). decodebin
    /// auto-plugs from its predecessor's declared caps, so it needs one.
    DecodebinNoUpstream,
    /// `decodebin` found no chain of registered decoders / parsers from its input
    /// caps to raw video or audio (the input caps are quoted). Either no decoder
    /// feature is compiled in, or the input is a container that needs a demuxer
    /// (auto-plugging through fan-out demuxers is not yet supported).
    NoDecodeChain(String),
    /// A `uridecodebin` / `playbin` was not at the head of its chain. It provides
    /// the source, so it must start the pipeline.
    UriSourceNotAtHead(String),
    /// A `uridecodebin` / `playbin` had no `uri=` property.
    MissingUri(String),
    /// The `uri=` could not be turned into a source (bad URI, or no handler
    /// registered for its scheme). The message quotes the URI and reason.
    Uri(String),
    /// Linking two nodes into the graph failed.
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
            ParseError::EmptyStage => f.write_str("empty node between '!' separators"),
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
            ParseError::UnknownReference(n) => write!(f, "reference to undeclared element name: {n}"),
            ParseError::DuplicateName(n) => write!(f, "duplicate element name: {n}"),
            ParseError::FanOutWithoutTee(n) => {
                write!(f, "{n}: output fans out to more than one consumer; insert a 'tee' to branch")
            }
            ParseError::NotAMuxer(n) => {
                write!(f, "{n}: more than one input links here, but it is not a registered muxer")
            }
            ParseError::MuxerWithoutOutput(n) => {
                write!(f, "{n}: muxer has no outgoing link; its output must feed a consumer")
            }
            ParseError::QueueRole(n) => {
                write!(f, "{n}: a queue must sit between two elements (1-in/1-out); it maps to an edge policy, not a source/sink/branch")
            }
            ParseError::DecodebinNoUpstream => {
                write!(f, "decodebin has no upstream element to decode; it must follow a source or element with declared caps")
            }
            ParseError::NoDecodeChain(caps) => {
                write!(f, "decodebin: no decoder chain from {caps} to raw (no decoder feature compiled in, or a container that needs a demuxer)")
            }
            ParseError::UriSourceNotAtHead(n) => {
                write!(f, "{n}: provides the source, so it must start the pipeline (be the first element)")
            }
            ParseError::MissingUri(n) => write!(f, "{n}: missing required 'uri=' property"),
            ParseError::Uri(msg) => write!(f, "uri error: {msg}"),
            ParseError::Graph(e) => write!(f, "graph link error: {e:?}"),
        }
    }
}

/// One parsed element: factory name plus its `key=value` properties (all owned so
/// errors can name them), and the optional `name=` handle that pad references
/// resolve against. `name` is special-cased here, never applied as a property.
struct ElementSpec {
    name: String,
    props: Vec<(String, String)>,
    instance: Option<String>,
}

/// An item in a chain: an element to build, a `t.` reference to a named element
/// declared elsewhere (the branching / link-by-name syntax), or a node already
/// constructed by a macro expansion (`uridecodebin` / `playbin`, M196), spliced
/// in directly rather than built by name.
enum Item {
    Element(ElementSpec),
    Ref(String),
    Prebuilt(PrebuiltNode),
}

/// A node a macro expansion built ahead of the structural pass: a source
/// constructed from a `uri=` scheme handler, or a decoder the auto-plug search
/// instantiated. Spliced into the graph as-is (it has no name to build by).
enum PrebuiltNode {
    Source(Box<dyn DynSourceLoop>),
    Element(Box<dyn DynAsyncElement>),
}

/// A run of items linked left-to-right by `!`. Branches are separate chains
/// joined through named references.
type Chain = Vec<Item>;

/// A caps description node (`video/x-raw,format=nv12,...`): a media type whose
/// `/` precedes any `=` field. A property value's `/` (a path or a fraction)
/// comes after its `=`, so it is not mistaken for caps.
fn is_caps_token(tok: &str) -> bool {
    match (tok.find('/'), tok.find('=')) {
        (Some(slash), Some(eq)) => slash < eq,
        (Some(_), None) => true,
        _ => false,
    }
}

/// A pad reference (`t.` or `t.src_0`): a name, a `.`, and no `=` / `/`. Returns
/// the referenced element name; the pad suffix is ignored (pads are positional).
fn as_ref_name(tok: &str) -> Option<&str> {
    if tok.contains('=') || tok.contains('/') || !tok.contains('.') {
        return None;
    }
    let name = tok.split('.').next().unwrap_or("");
    (!name.is_empty()).then_some(name)
}

/// Consume an element's `key=value` properties from the token stream, stopping at
/// a `!`, a caps node, or a pad reference (the next node begins). A bare token
/// with no `=` is a malformed property (the gst typo case), reported by name.
fn consume_element<'a, I: Iterator<Item = &'a str>>(
    name: &str,
    tokens: &mut core::iter::Peekable<I>,
) -> Result<ElementSpec, ParseError> {
    let mut spec = ElementSpec { name: name.to_string(), props: Vec::new(), instance: None };
    while let Some(&tok) = tokens.peek() {
        if tok == "!" || is_caps_token(tok) || as_ref_name(tok).is_some() {
            break;
        }
        let (key, value) = tok.split_once('=').ok_or_else(|| ParseError::MalformedProperty {
            element: name.to_string(),
            token: tok.to_string(),
        })?;
        tokens.next();
        // Strip a single layer of surrounding double quotes from the value.
        let value = value.strip_prefix('"').and_then(|v| v.strip_suffix('"')).unwrap_or(value);
        if key == "name" {
            spec.instance = Some(value.to_string());
        } else {
            spec.props.push((key.to_string(), value.to_string()));
        }
    }
    Ok(spec)
}

/// Split a `gst-launch` pipeline string into chains: runs of nodes linked by `!`,
/// with branches expressed as separate chains joined through `name=` / `t.`.
fn parse_chains(pipeline: &str) -> Result<Vec<Chain>, ParseError> {
    let trimmed = pipeline.trim();
    if trimmed.is_empty() {
        return Err(ParseError::Empty);
    }
    // Make every `!` a standalone token regardless of surrounding spaces, then
    // split on whitespace. (A value containing spaces is the documented v1 gap.)
    let spaced = trimmed.replace('!', " ! ");
    let mut tokens = spaced.split_whitespace().peekable();

    #[derive(Clone, Copy)]
    enum St {
        Start,
        AfterBang,
        AfterNode,
    }

    let mut chains: Vec<Chain> = Vec::new();
    let mut cur: Chain = Vec::new();
    let mut st = St::Start;

    loop {
        match st {
            St::Start | St::AfterBang => {
                let after_bang = matches!(st, St::AfterBang);
                let Some(tok) = tokens.next() else {
                    if after_bang {
                        return Err(ParseError::EmptyStage); // trailing `!`
                    }
                    break;
                };
                if tok == "!" {
                    return Err(ParseError::EmptyStage); // leading or doubled `!`
                }
                if is_caps_token(tok) {
                    cur.push(Item::Element(ElementSpec {
                        name: "capsfilter".to_string(),
                        props: alloc::vec![("caps".to_string(), tok.to_string())],
                        instance: None,
                    }));
                    st = St::AfterNode;
                } else if let Some(name) = as_ref_name(tok) {
                    cur.push(Item::Ref(name.to_string()));
                    if after_bang {
                        // Tail ref (`! t.`): links the upstream node into the
                        // named element and ends the chain.
                        chains.push(core::mem::take(&mut cur));
                        st = St::Start;
                    } else {
                        // Head ref (`t. ! ...`): feeds the chain from it.
                        st = St::AfterNode;
                    }
                } else {
                    cur.push(Item::Element(consume_element(tok, &mut tokens)?));
                    st = St::AfterNode;
                }
            }
            St::AfterNode => match tokens.peek() {
                Some(&"!") => {
                    tokens.next();
                    st = St::AfterBang;
                }
                Some(_) => {
                    // A node not joined by `!`: the current chain ends here and a
                    // new one starts at this token (reprocessed as a head).
                    chains.push(core::mem::take(&mut cur));
                    st = St::Start;
                }
                None => break,
            },
        }
    }

    if !cur.is_empty() {
        chains.push(cur);
    }
    Ok(chains)
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

/// Apply `key=value` properties to a muxer node, the same as
/// [`apply_element_props`] but over the [`DynMultiInputElement`] surface (M199:
/// muxers gained a property surface, so `pyaggregator module=... class=...` and
/// the like parse). `name=` is already consumed as the node handle.
fn apply_muxer_props(
    mux: &mut Box<dyn crate::runtime::DynMultiInputElement>,
    name: &str,
    props: &[(String, String)],
) -> Result<(), ParseError> {
    for (key, value) in props {
        let kind = mux
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
        mux.set_property(key, parsed).map_err(|_| ParseError::BadValue {
            element: name.into(),
            key: key.clone(),
            value: value.clone(),
        })?;
    }
    Ok(())
}

/// Build the runnable [`Graph`] from parsed chains: flatten elements, resolve
/// `t.` references into directed links, derive each element's role (and any tee's
/// fan-out width) from its link degree, then construct and wire the nodes.
/// `decodebin` / `decodebin3`: not an element, but a macro that expands, at
/// parse time, into the chain of decoders / parsers the auto-plug search finds
/// from its upstream caps down to raw video or audio (M193).
fn is_decodebin(name: &str) -> bool {
    matches!(name, "decodebin" | "decodebin3")
}

/// Depth bound for the decodebin auto-plug search: a parse + decode (+ a spare
/// hop) is 2-3, so this leaves headroom without letting an unsatisfiable target
/// wander.
const DECODEBIN_MAX_DEPTH: usize = 6;

/// Expand every `decodebin` node into the decoder chain the registry auto-plugs
/// from its predecessor's declared caps down to raw (video or audio). An empty
/// chain (the input is already raw) drops the node entirely, so its predecessor
/// links straight to its consumer. The predecessor is the element immediately
/// before the `decodebin` in the same chain; a `decodebin` with no upstream
/// element (chain head, or after a bare `name.` reference) is a loud error,
/// since it has nothing to take its input caps from.
fn expand_decodebin(registry: &Registry, chains: Vec<Chain>) -> Result<Vec<Chain>, ParseError> {
    let mut out = Vec::with_capacity(chains.len());
    for chain in chains {
        let mut new_chain: Chain = Vec::with_capacity(chain.len());
        // The element (name + props) whose output caps feed the next decodebin:
        // the most recent real element. A `Ref` clears it (its caps live in
        // another chain). Props matter because they can re-type the output (a
        // `filesrc`'s `bytestream-format` selects the container).
        let mut upstream: Option<(String, Vec<(String, String)>)> = None;
        for item in chain {
            match item {
                Item::Element(spec) if is_decodebin(&spec.name) => {
                    let (pred, props) = upstream.as_ref().ok_or(ParseError::DecodebinNoUpstream)?;
                    let caps = resolve_upstream_caps(registry, pred, props)?;
                    let target = |c: &Caps| is_raw_video(c) || is_raw_audio(c);
                    let names = registry
                        .autoplug_names(&caps, &target, DECODEBIN_MAX_DEPTH)
                        .ok_or_else(|| ParseError::NoDecodeChain(alloc::format!("{caps:?}")))?;
                    for name in names {
                        new_chain.push(Item::Element(ElementSpec {
                            name: name.to_string(),
                            props: Vec::new(),
                            instance: None,
                        }));
                        upstream = Some((name.to_string(), Vec::new()));
                    }
                }
                Item::Element(spec) => {
                    upstream = Some((spec.name.clone(), spec.props.clone()));
                    new_chain.push(Item::Element(spec));
                }
                Item::Ref(name) => {
                    upstream = None;
                    new_chain.push(Item::Ref(name));
                }
                // A pre-built node (from `uridecodebin` / `playbin`) carries no
                // name to take declared caps from, so a `decodebin` cannot follow
                // it. Clear the upstream; if a decodebin does follow, it reports
                // the missing-upstream error.
                prebuilt @ Item::Prebuilt(_) => {
                    upstream = None;
                    new_chain.push(prebuilt);
                }
            }
        }
        out.push(new_chain);
    }
    Ok(out)
}

/// The caps a `decodebin` predecessor produces, used as the auto-plug input
/// (M195). For a registered source, build it and apply its properties so a
/// property that re-types the output (a `filesrc`'s `bytestream-format`) is
/// reflected via [`SourceLoop::configured_output_caps`]; fall back to the
/// registry's declared caps (a fixed source, or a transform's source-pad
/// template). `bytestream-format=auto` returns `None` from `configured_output_caps`
/// (the container is only known after a run-time header sniff), so it too falls
/// back to the declared default.
fn resolve_upstream_caps(
    registry: &Registry,
    name: &str,
    props: &[(String, String)],
) -> Result<Caps, ParseError> {
    if let Some(mut src) = registry.make_source(name) {
        apply_source_props(&mut src, name, props)?;
        if let Some(caps) = src.configured_output_caps() {
            return Ok(caps);
        }
    }
    registry.declared_output_caps(name).ok_or(ParseError::DecodebinNoUpstream)
}

/// `uridecodebin` / `playbin`: a source-providing macro. `uridecodebin uri=X`
/// builds the source from the URI scheme handler and auto-plugs the decode chain
/// to raw; `playbin uri=X` is that plus an auto sink (`autovideosink`, or the
/// `video-sink=` override), i.e. a complete pipeline.
fn is_uri_source(name: &str) -> bool {
    matches!(name, "uridecodebin" | "uridecodebin3" | "playbin" | "playbin3")
}

/// The value of a spec property by key, if present.
fn prop<'a>(spec: &'a ElementSpec, key: &str) -> Option<&'a str> {
    spec.props.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}

/// Expand every `uridecodebin` / `playbin` (a source position element) into the
/// `uri=` scheme handler's source plus the auto-plugged decode chain, as
/// pre-built nodes spliced straight into the chain. `playbin` additionally
/// appends an auto sink so the line is a complete pipeline. The element must head
/// its chain (it provides the source).
fn expand_uri_sources(registry: &Registry, chains: Vec<Chain>) -> Result<Vec<Chain>, ParseError> {
    let mut out = Vec::with_capacity(chains.len());
    for chain in chains {
        let mut new_chain: Chain = Vec::with_capacity(chain.len());
        for (i, item) in chain.into_iter().enumerate() {
            let spec = match item {
                Item::Element(spec) if is_uri_source(&spec.name) => spec,
                other => {
                    new_chain.push(other);
                    continue;
                }
            };
            if i != 0 {
                return Err(ParseError::UriSourceNotAtHead(spec.name));
            }
            let is_playbin = spec.name.starts_with("playbin");
            let uri = prop(&spec, "uri").ok_or_else(|| ParseError::MissingUri(spec.name.clone()))?;
            let (source, caps) = registry
                .build_uri_source(uri)
                .map_err(|e: UriError| ParseError::Uri(alloc::format!("{uri}: {e:?}")))?;
            let target = |c: &Caps| is_raw_video(c) || is_raw_audio(c);
            let decoders = registry
                .autoplug(&caps, &target, DECODEBIN_MAX_DEPTH)
                .ok_or_else(|| ParseError::NoDecodeChain(alloc::format!("{caps:?}")))?;
            new_chain.push(Item::Prebuilt(PrebuiltNode::Source(source)));
            for dec in decoders {
                new_chain.push(Item::Prebuilt(PrebuiltNode::Element(dec)));
            }
            if is_playbin {
                let sink = prop(&spec, "video-sink").unwrap_or("autovideosink").to_string();
                new_chain.push(Item::Element(ElementSpec { name: sink, props: Vec::new(), instance: None }));
            }
        }
        out.push(new_chain);
    }
    Ok(out)
}

fn build_graph(registry: &Registry, chains: Vec<Chain>) -> Result<Graph<GraphNode>, ParseError> {
    // Expand the source-providing (uridecodebin / playbin) and mid-chain
    // (decodebin) macros into concrete nodes before the structural build, so the
    // rest of the builder sees only real elements and pre-built nodes.
    let chains = expand_uri_sources(registry, chains)?;
    let chains = expand_decodebin(registry, chains)?;

    // A chain endpoint after flattening: a concrete element index, or a still
    // unresolved reference by name.
    enum Endpoint {
        Element(usize),
        Ref(String),
    }

    let mut specs: Vec<ElementSpec> = Vec::new();
    // Parallel to `specs`: the pre-built node for that index (a `uridecodebin` /
    // `playbin` source or decoder), or `None` for a normal name-built element.
    // The placeholder spec for a pre-built node carries a benign name so the
    // structural closures (`is_queue` / `is_tee`) never match it, and node
    // construction uses the pre-built node instead of looking the name up.
    let mut prebuilt: Vec<Option<PrebuiltNode>> = Vec::new();
    let mut names: Vec<(String, usize)> = Vec::new();
    let mut chain_eps: Vec<Vec<Endpoint>> = Vec::with_capacity(chains.len());

    for chain in chains {
        let mut eps = Vec::with_capacity(chain.len());
        for item in chain {
            match item {
                Item::Element(spec) => {
                    let ei = specs.len();
                    if let Some(inst) = &spec.instance {
                        if names.iter().any(|(n, _)| n == inst) {
                            return Err(ParseError::DuplicateName(inst.clone()));
                        }
                        names.push((inst.clone(), ei));
                    }
                    specs.push(spec);
                    prebuilt.push(None);
                    eps.push(Endpoint::Element(ei));
                }
                Item::Prebuilt(node) => {
                    let ei = specs.len();
                    let name = match node {
                        PrebuiltNode::Source(_) => "uridecodebin",
                        PrebuiltNode::Element(_) => "(decoder)",
                    };
                    specs.push(ElementSpec { name: name.to_string(), props: Vec::new(), instance: None });
                    prebuilt.push(Some(node));
                    eps.push(Endpoint::Element(ei));
                }
                Item::Ref(name) => eps.push(Endpoint::Ref(name)),
            }
        }
        chain_eps.push(eps);
    }

    if specs.len() < 2 {
        return Err(ParseError::TooFewStages);
    }

    // Resolve references and collect the directed links (by element index).
    let mut raw_links: Vec<(usize, usize)> = Vec::new();
    for eps in &chain_eps {
        let mut idxs: Vec<usize> = Vec::with_capacity(eps.len());
        for ep in eps {
            idxs.push(match ep {
                Endpoint::Element(ei) => *ei,
                Endpoint::Ref(name) => names
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, i)| *i)
                    .ok_or_else(|| ParseError::UnknownReference(name.clone()))?,
            });
        }
        for w in idxs.windows(2) {
            raw_links.push((w[0], w[1]));
        }
    }

    // M190: `queue` / `queue2` is not an element in g2g. Per the design,
    // per-edge `LinkPolicy` (Block / DropOldest / DropNewest) is the leaky-queue
    // analog, so a queue node collapses into the backpressure policy of the edge
    // it sits on rather than becoming a buffering element + extra hop. Validate
    // each queue is 1-in/1-out, read its `leaky=`, then contract it out of the
    // link list, walking chains of queues to the first real consumer and keeping
    // the downstream-most leaky policy.
    let is_queue = |ei: usize| matches!(specs[ei].name.as_str(), "queue" | "queue2");
    let mut raw_in = alloc::vec![0usize; specs.len()];
    let mut raw_out = alloc::vec![0usize; specs.len()];
    for &(s, d) in &raw_links {
        raw_out[s] += 1;
        raw_in[d] += 1;
    }
    let mut queue_succ: Vec<Option<usize>> = alloc::vec![None; specs.len()];
    let mut queue_policy = alloc::vec![LinkPolicy::Block; specs.len()];
    for ei in 0..specs.len() {
        if is_queue(ei) {
            if raw_in[ei] != 1 || raw_out[ei] != 1 {
                return Err(ParseError::QueueRole(specs[ei].name.clone()));
            }
            queue_policy[ei] = queue_leaky_policy(&specs[ei]);
            queue_succ[ei] = raw_links.iter().find(|(s, _)| *s == ei).map(|(_, d)| *d);
        }
    }
    // Each edge whose source is a real element walks through any run of queues to
    // its terminal consumer, carrying the accumulated policy; edges out of a queue
    // are consumed by that walk (skipped here).
    let mut links: Vec<(usize, usize, LinkPolicy)> = Vec::new();
    for &(s, d) in &raw_links {
        if is_queue(s) {
            continue;
        }
        let mut cur = d;
        let mut policy = LinkPolicy::Block;
        while is_queue(cur) {
            if queue_policy[cur] != LinkPolicy::Block {
                policy = queue_policy[cur];
            }
            cur = queue_succ[cur].expect("queue validated 1-out above");
        }
        links.push((s, cur, policy));
    }

    // Link degree per element fixes its role and any tee's output width. Computed
    // over the contracted links, so queue indices drop to degree 0 and are skipped
    // as nodes below.
    let mut in_deg = alloc::vec![0usize; specs.len()];
    let mut out_deg = alloc::vec![0usize; specs.len()];
    for &(s, d, _) in &links {
        out_deg[s] += 1;
        in_deg[d] += 1;
    }

    let is_tee = |ei: usize| specs[ei].name == "tee";
    // A non-tee node with several inbound links is a muxer (built from the
    // registry with that input count); a tee has a single input pad.
    let is_muxer = |ei: usize| !is_tee(ei) && in_deg[ei] > 1;
    for ei in 0..specs.len() {
        if !is_tee(ei) && out_deg[ei] > 1 {
            return Err(ParseError::FanOutWithoutTee(specs[ei].name.clone()));
        }
        if is_muxer(ei) && out_deg[ei] == 0 {
            return Err(ParseError::MuxerWithoutOutput(specs[ei].name.clone()));
        }
        if is_tee(ei) && !specs[ei].props.is_empty() {
            // The structural tee carries no element, so it has no properties.
            return Err(ParseError::UnknownProperty {
                element: "tee".to_string(),
                key: specs[ei].props[0].0.clone(),
            });
        }
    }

    // Construct nodes in element-index order so `node_of[ei]` lines up. Queue
    // indices were contracted into edge policies above, so they get no node
    // (`None`); they never appear as an endpoint in the contracted links.
    let mut graph: Graph<GraphNode> = Graph::new();
    let mut node_of: Vec<Option<NodeId>> = Vec::with_capacity(specs.len());
    for ei in 0..specs.len() {
        if is_queue(ei) {
            node_of.push(None);
            continue;
        }
        // A pre-built node (uridecodebin / playbin source or decoder) is spliced
        // in directly; its role still follows link degree (a source has no input,
        // a terminal decoder no output).
        if let Some(node) = prebuilt[ei].take() {
            let nid = match node {
                PrebuiltNode::Source(src) => graph.add_source(GraphNodeRef::Source(src)),
                PrebuiltNode::Element(el) if out_deg[ei] == 0 => {
                    graph.add_sink(GraphNodeRef::Element(el))
                }
                PrebuiltNode::Element(el) => graph.add_transform(GraphNodeRef::Element(el)),
            };
            node_of.push(Some(nid));
            continue;
        }
        let spec = &specs[ei];
        let node = if is_tee(ei) {
            graph.add_tee(out_deg[ei] as u8).node()
        } else if in_deg[ei] == 0 {
            let mut src = registry
                .make_source(&spec.name)
                .ok_or_else(|| ParseError::UnknownSource(spec.name.clone()))?;
            apply_source_props(&mut src, &spec.name, &spec.props)?;
            graph.add_source(GraphNodeRef::Source(src))
        } else if is_muxer(ei) {
            let mut mux = registry
                .make_muxer(&spec.name, in_deg[ei])
                .ok_or_else(|| ParseError::NotAMuxer(spec.name.clone()))?;
            apply_muxer_props(&mut mux, &spec.name, &spec.props)?;
            graph.add_muxer(GraphNodeRef::Muxer(mux), in_deg[ei] as u8).node()
        } else if out_deg[ei] == 0 {
            let mut el = registry
                .make_element(&spec.name)
                .ok_or_else(|| ParseError::UnknownElement(spec.name.clone()))?;
            apply_element_props(&mut el, &spec.name, &spec.props)?;
            graph.add_sink(GraphNodeRef::Element(el))
        } else {
            let mut el = registry
                .make_element(&spec.name)
                .ok_or_else(|| ParseError::UnknownElement(spec.name.clone()))?;
            apply_element_props(&mut el, &spec.name, &spec.props)?;
            graph.add_transform(GraphNodeRef::Element(el))
        };
        node_of.push(Some(node));
    }

    // Wire edges. Each tee branch takes a distinct output pad (0..n) and each
    // muxer input a distinct input pad (0..n); every other output and input is
    // pad 0. A queue's `leaky=` rides along as the edge's `LinkPolicy`.
    let mut tee_next = alloc::vec![0u8; specs.len()];
    let mut mux_next = alloc::vec![0u8; specs.len()];
    for &(s, d, policy) in &links {
        let node_s = node_of[s].expect("contracted link source is a real node");
        let node_d = node_of[d].expect("contracted link destination is a real node");
        let src = if is_tee(s) {
            let index = tee_next[s];
            tee_next[s] += 1;
            PadId { node: node_s, index }
        } else {
            PadId::from(node_s)
        };
        let dst = if is_muxer(d) {
            let index = mux_next[d];
            mux_next[d] += 1;
            PadId { node: node_d, index }
        } else {
            PadId::from(node_d)
        };
        graph.link_with(src, dst, policy)?;
    }

    Ok(graph)
}

/// Map a `queue` / `queue2` node's `leaky=` property to the edge backpressure
/// policy it stands in for (M190). gst accepts the enum by value or nick:
/// `0`/`no` (lossless, the default), `1`/`upstream` (drop the newest incoming
/// buffer), `2`/`downstream` (drop the oldest queued buffer). Other queue
/// properties (the `max-size-*` / `min-threshold-*` bounds, `silent`, ...) are
/// accepted but not modeled: g2g has no per-edge capacity (`link_capacity` is a
/// single pipeline-wide knob), so they are ignored for paste compatibility
/// rather than rejected.
fn queue_leaky_policy(spec: &ElementSpec) -> LinkPolicy {
    for (k, v) in &spec.props {
        if k == "leaky" {
            return match v.as_str() {
                "1" | "upstream" => LinkPolicy::DropNewest,
                "2" | "downstream" => LinkPolicy::DropOldest,
                // "0" / "no" and anything unrecognized: lossless block.
                _ => LinkPolicy::Block,
            };
        }
    }
    LinkPolicy::Block
}

/// Parse a `gst-launch`-style pipeline string into a runnable [`Graph`], building
/// each element by name from `registry`, applying its `key=value` properties, and
/// linking the chains (including `tee` branches) into the DAG. Roles follow
/// connectivity. The result drops straight onto
/// [`run_graph`](crate::runtime::run_graph).
///
/// ```text
/// videotestsrc num-buffers=3 ! tee name=t ! fakesink   t. ! videoflip ! fakesink
/// ```
pub fn parse_launch(registry: &Registry, pipeline: &str) -> Result<Graph<GraphNode>, ParseError> {
    let chains = parse_chains(pipeline)?;
    build_graph(registry, chains)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item_names(chain: &Chain) -> Vec<&str> {
        chain
            .iter()
            .map(|i| match i {
                Item::Element(s) => s.name.as_str(),
                Item::Ref(n) => n.as_str(),
                Item::Prebuilt(_) => "(prebuilt)",
            })
            .collect()
    }

    #[test]
    fn parse_chains_splits_names_and_props() {
        let chains = parse_chains(
            "videotestsrc num-buffers=3 pattern=snow ! videoflip method=rotate-180 ! fakesink",
        )
        .unwrap();
        assert_eq!(chains.len(), 1);
        assert_eq!(item_names(&chains[0]), ["videotestsrc", "videoflip", "fakesink"]);
        let Item::Element(src) = &chains[0][0] else { panic!("first is an element") };
        assert_eq!(
            src.props,
            [("num-buffers".to_string(), "3".to_string()), ("pattern".into(), "snow".into())]
        );
        let Item::Element(sink) = &chains[0][2] else { panic!("last is an element") };
        assert!(sink.props.is_empty());
    }

    #[test]
    fn parse_chains_strips_quoted_values() {
        // A double-quoted value (no spaces) has its quotes stripped. Values with
        // spaces are a known v1 gap (the whitespace tokenizer would split them).
        let chains = parse_chains("filesrc location=\"file.mp4\" ! fakesink").unwrap();
        let Item::Element(src) = &chains[0][0] else { panic!("element") };
        assert_eq!(src.props[0], ("location".to_string(), "file.mp4".to_string()));
    }

    #[test]
    fn caps_description_becomes_capsfilter() {
        // A bare `media/type,...` node is the inline caps-filter shorthand.
        let chains =
            parse_chains("videotestsrc ! video/x-raw,format=nv12,width=320 ! fakesink").unwrap();
        assert_eq!(item_names(&chains[0]), ["videotestsrc", "capsfilter", "fakesink"]);
        let Item::Element(caps) = &chains[0][1] else { panic!("element") };
        assert_eq!(
            caps.props,
            [("caps".to_string(), "video/x-raw,format=nv12,width=320".to_string())]
        );
    }

    #[test]
    fn tee_branch_parses_into_two_chains() {
        // `name=` is the instance handle (not a property); `t.` opens the branch.
        let chains =
            parse_chains("videotestsrc ! tee name=t ! fakesink t. ! videoflip ! fakesink").unwrap();
        assert_eq!(chains.len(), 2);
        assert_eq!(item_names(&chains[0]), ["videotestsrc", "tee", "fakesink"]);
        assert_eq!(item_names(&chains[1]), ["t", "videoflip", "fakesink"]);
        let Item::Element(tee) = &chains[0][1] else { panic!("element") };
        assert_eq!(tee.instance.as_deref(), Some("t"));
        assert!(tee.props.is_empty(), "name= is the handle, not a property");
        assert!(matches!(&chains[1][0], Item::Ref(n) if n == "t"));
    }

    #[test]
    fn empty_and_too_few_stages_error() {
        let reg = Registry::new();
        assert!(matches!(parse_launch(&reg, "   "), Err(ParseError::Empty)));
        assert!(matches!(parse_launch(&reg, "videotestsrc"), Err(ParseError::TooFewStages)));
    }

    #[test]
    fn malformed_property_is_reported() {
        assert!(matches!(
            parse_chains("videotestsrc bogus ! fakesink"),
            Err(ParseError::MalformedProperty { .. })
        ));
    }

    #[test]
    fn unknown_reference_is_reported() {
        // The degree / reference checks precede registry construction, so an
        // empty registry still surfaces them.
        let reg = Registry::new();
        let err =
            parse_launch(&reg, "videotestsrc ! tee name=t ! fakesink nope. ! fakesink").unwrap_err();
        assert_eq!(err, ParseError::UnknownReference("nope".to_string()));
    }

    #[test]
    fn duplicate_name_is_reported() {
        let reg = Registry::new();
        let err =
            parse_launch(&reg, "videotestsrc name=x ! videoflip name=x ! fakesink").unwrap_err();
        assert_eq!(err, ParseError::DuplicateName("x".to_string()));
    }

    #[test]
    fn fan_out_without_tee_is_reported() {
        let reg = Registry::new();
        // `s` is not a tee, yet it feeds the inline sink and the `s.` branch.
        let err =
            parse_launch(&reg, "videotestsrc name=s ! fakesink s. ! fakesink").unwrap_err();
        assert_eq!(err, ParseError::FanOutWithoutTee("videotestsrc".to_string()));
    }
}
