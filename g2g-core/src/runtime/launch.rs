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
use crate::runtime::autoplug::{
    is_raw_audio, is_raw_video, PadKind, PadRequest, Registry, UriError,
};
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
    BadValue {
        element: String,
        key: String,
        value: String,
    },
    /// A `name.` reference names no element declared with that `name=`.
    UnknownReference(String),
    /// Two elements share the same `name=` handle.
    DuplicateName(String),
    /// More than one link feeds an element's input, but it names no registered
    /// muxer: fan-in needs a [`MuxerFactory`](crate::runtime::MuxerFactory).
    NotAMuxer(String),
    /// A muxer (an element with several inputs) has no outgoing link; its single
    /// output pad must feed a downstream consumer.
    MuxerWithoutOutput(String),
    /// A named input-pad reference (`mux.foo_0`) names a request pad this muxer
    /// does not define (M481): the element's `input_pad_index` scheme declined it.
    UnknownInputPad(String),
    /// Two input-pad references resolve to the same muxer input index (M481), e.g.
    /// `mux.video_0` named twice, or a named pad colliding with a positional one.
    DuplicateInputPad(String),
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
                write!(
                    f,
                    "{element}: malformed property '{token}' (expected key=value)"
                )
            }
            ParseError::UnknownProperty { element, key } => {
                write!(f, "{element}: no property named '{key}'")
            }
            ParseError::BadValue {
                element,
                key,
                value,
            } => {
                write!(f, "{element}: invalid value '{value}' for property '{key}'")?;
                // Inline caps become a `capsfilter` (key `caps`); a gst dev often
                // reaches for range / list / feature syntax g2g's launch parser
                // does not accept, and the bare "invalid value" hides why.
                if element == "capsfilter" && key == "caps" && value.contains(['[', '{', '(']) {
                    write!(
                        f,
                        " (a launch caps filter takes fixed fields, ranges [min,max], \
                         and lists {{a,b}} with numeric / known values; caps features \
                         like (memory:...) are not supported)"
                    )?;
                }
                Ok(())
            }
            ParseError::UnknownReference(n) => {
                write!(f, "reference to undeclared element name: {n}")
            }
            ParseError::DuplicateName(n) => write!(f, "duplicate element name: {n}"),
            ParseError::NotAMuxer(n) => {
                write!(
                    f,
                    "{n}: more than one input links here, but it is not a registered muxer"
                )
            }
            ParseError::MuxerWithoutOutput(n) => {
                write!(
                    f,
                    "{n}: muxer has no outgoing link; its output must feed a consumer"
                )
            }
            ParseError::UnknownInputPad(n) => {
                write!(
                    f,
                    "{n}: no such input request pad (this element defines no pad by that name)"
                )
            }
            ParseError::DuplicateInputPad(n) => {
                write!(f, "{n}: two inputs resolve to the same request pad index")
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
    /// A `t.` / `d.video_0` reference to a named element. `pad` is the suffix after
    /// the dot (`""` for a bare `t.`, `"video_0"` for `d.video_0`), used by the
    /// explicit-demux fan-out (M476) to select which stream a branch reads; ignored
    /// for a tee (positional).
    Ref {
        name: String,
        pad: String,
    },
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

/// A pad reference (`t.`, `t.src_0`, `d.video_0`): a name, a `.`, and no `=` / `/`.
/// Returns the referenced element name and the pad suffix after the first dot
/// (`""` for a bare `t.`). The suffix drives the explicit-demux fan-out (M476);
/// a tee ignores it (positional).
fn split_pad_ref(tok: &str) -> Option<(&str, &str)> {
    if tok.contains('=') || tok.contains('/') || !tok.contains('.') {
        return None;
    }
    let (name, pad) = tok.split_once('.').unwrap_or((tok, ""));
    (!name.is_empty()).then_some((name, pad))
}

/// The referenced element name of a pad reference (the suffix dropped); the
/// token-boundary test in [`consume_element`].
fn as_ref_name(tok: &str) -> Option<&str> {
    split_pad_ref(tok).map(|(name, _)| name)
}

/// Parse a demux output-pad suffix into a [`PadRequest`] (M476): `"video_0"` ->
/// `{ Video, 0 }`, `"audio_1"` -> `{ Audio, 1 }`, `"text_0"` / `"subtitle_0"` ->
/// `{ Text, 0 }`, `"src_2"` -> `{ Any, 2 }`. A bare `d.` (empty suffix) or an
/// unrecognized prefix is `{ Any, ordinal }`, i.e. positional by reference order.
fn parse_pad_request(pad: &str, ordinal: usize) -> PadRequest {
    let (prefix, index) = match pad.rsplit_once('_') {
        Some((p, n)) => (p, n.parse::<usize>().ok()),
        None => (pad, None),
    };
    let kind = match prefix {
        "video" => PadKind::Video,
        "audio" => PadKind::Audio,
        "text" | "subtitle" => PadKind::Text,
        _ => PadKind::Any,
    };
    // `src_N` (output) / `sink_N` (input) and unrecognized prefixes select the Nth
    // stream / pad; a bare `d.` has no index, so it takes the positional ordinal.
    let index = match kind {
        PadKind::Any if prefix == "src" || prefix == "sink" || prefix.is_empty() => {
            index.unwrap_or(ordinal)
        }
        PadKind::Any => ordinal,
        _ => index.unwrap_or(0),
    };
    PadRequest { kind, index }
}

/// Consume an element's `key=value` properties from the token stream, stopping at
/// a `!`, a caps node, or a pad reference (the next node begins). A bare token
/// with no `=` is a malformed property (the gst typo case), reported by name.
fn consume_element<'a, I: Iterator<Item = &'a str>>(
    name: &str,
    tokens: &mut core::iter::Peekable<I>,
) -> Result<ElementSpec, ParseError> {
    let mut spec = ElementSpec {
        name: name.to_string(),
        props: Vec::new(),
        instance: None,
    };
    while let Some(&tok) = tokens.peek() {
        if tok == "!" || is_caps_token(tok) || as_ref_name(tok).is_some() {
            break;
        }
        let (key, value) = tok
            .split_once('=')
            .ok_or_else(|| ParseError::MalformedProperty {
                element: name.to_string(),
                token: tok.to_string(),
            })?;
        tokens.next();
        // Strip a single layer of surrounding quotes (double or single) from the value.
        let value = strip_quotes(value);
        if key == "name" {
            spec.instance = Some(value.to_string());
        } else {
            spec.props.push((key.to_string(), value.to_string()));
        }
    }
    Ok(spec)
}

/// Strip a single matching pair of surrounding quotes (double or single) from a
/// property value. gst-launch accepts both `location="a b"` and `location='a b'`.
fn strip_quotes(v: &str) -> &str {
    let b = v.as_bytes();
    // Quotes are ASCII, so byte-slicing at these bounds stays on char boundaries.
    if b.len() >= 2 && (b[0] == b'"' || b[0] == b'\'') && b[b.len() - 1] == b[0] {
        &v[1..v.len() - 1]
    } else {
        v
    }
}

/// Split a pipeline string into tokens, honoring quoted property values and
/// `#` comments. Outside quotes, whitespace separates tokens and `!` is a
/// standalone token; inside a `"..."` or `'...'` region both are literal, so a
/// value may contain spaces (and even `!`), e.g. `element="x264enc bitrate=4000"`
/// or `location='/my file.ts'`. A `#` outside quotes starts a comment that runs
/// to end of line (a pasted multi-line pipeline may carry them). The surrounding
/// quotes are kept on the token; [`consume_element`] strips them from the value.
/// An unterminated quote runs to end of input (best-effort; the property parse
/// then reports any resulting malformed token).
fn tokenize(s: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    // The open quote char (`"` or `'`), or `None` outside a quoted region.
    let mut quote: Option<char> = None;
    let mut in_comment = false;
    for c in s.chars() {
        if in_comment {
            // A comment runs to end of line; the newline (whitespace) ends it.
            if c == '\n' {
                in_comment = false;
            }
            continue;
        }
        match c {
            '"' | '\'' if quote.is_none() => {
                quote = Some(c);
                cur.push(c);
            }
            _ if Some(c) == quote => {
                quote = None;
                cur.push(c);
            }
            // A `#` only starts a comment at a token boundary (nothing buffered);
            // mid-token it is literal, so a URI fragment (`uri=...#closed-captions=cc1`,
            // `#t=10`) is preserved.
            '#' if quote.is_none() && cur.is_empty() => {
                in_comment = true;
            }
            '!' if quote.is_none() => {
                if !cur.is_empty() {
                    tokens.push(core::mem::take(&mut cur));
                }
                tokens.push("!".to_string());
            }
            c if c.is_whitespace() && quote.is_none() => {
                if !cur.is_empty() {
                    tokens.push(core::mem::take(&mut cur));
                }
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }
    tokens
}

/// Split a `gst-launch` pipeline string into chains: runs of nodes linked by `!`,
/// with branches expressed as separate chains joined through `name=` / `t.`.
fn parse_chains(pipeline: &str) -> Result<Vec<Chain>, ParseError> {
    let trimmed = pipeline.trim();
    if trimmed.is_empty() {
        return Err(ParseError::Empty);
    }
    // Tokenize quote-aware so a `!` (a standalone token) and whitespace inside a
    // quoted value are literal, letting a property value carry spaces.
    let toks = tokenize(trimmed);
    let mut tokens = toks.iter().map(String::as_str).peekable();

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
                } else if let Some((name, pad)) = split_pad_ref(tok) {
                    cur.push(Item::Ref {
                        name: name.to_string(),
                        pad: pad.to_string(),
                    });
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
/// Apply launch-line properties to a terminal fan-out source (M727), the
/// fanout-source analog of [`apply_source_props`].
fn apply_fanout_src_props(
    src: &mut alloc::boxed::Box<dyn crate::fanout::DynMultiOutputSource>,
    name: &str,
    props: &[(String, String)],
) -> Result<(), ParseError> {
    for (key, raw) in props {
        if key == "name" {
            continue;
        }
        let spec = src
            .properties()
            .iter()
            .find(|p| p.name == key)
            .ok_or_else(|| ParseError::UnknownProperty {
                element: name.to_string(),
                key: key.clone(),
            })?;
        let value = crate::property::PropValue::parse(spec.kind, raw).map_err(|_| {
            ParseError::BadValue {
                element: name.to_string(),
                key: key.clone(),
                value: raw.clone(),
            }
        })?;
        src.set_property(key, value)
            .map_err(|_| ParseError::BadValue {
                element: name.to_string(),
                key: key.clone(),
                value: raw.clone(),
            })?;
    }
    Ok(())
}

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
            .ok_or_else(|| ParseError::UnknownProperty {
                element: name.into(),
                key: key.clone(),
            })?
            .kind;
        let parsed = PropValue::parse(kind, value).map_err(|_| ParseError::BadValue {
            element: name.into(),
            key: key.clone(),
            value: value.clone(),
        })?;
        el.set_property(key, parsed)
            .map_err(|_| ParseError::BadValue {
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
            .ok_or_else(|| ParseError::UnknownProperty {
                element: name.into(),
                key: key.clone(),
            })?
            .kind;
        let parsed = PropValue::parse(kind, value).map_err(|_| ParseError::BadValue {
            element: name.into(),
            key: key.clone(),
            value: value.clone(),
        })?;
        el.set_property(key, parsed)
            .map_err(|_| ParseError::BadValue {
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
            .ok_or_else(|| ParseError::UnknownProperty {
                element: name.into(),
                key: key.clone(),
            })?
            .kind;
        let parsed = PropValue::parse(kind, value).map_err(|_| ParseError::BadValue {
            element: name.into(),
            key: key.clone(),
            value: value.clone(),
        })?;
        mux.set_property(key, parsed)
            .map_err(|_| ParseError::BadValue {
                element: name.into(),
                key: key.clone(),
                value: value.clone(),
            })?;
    }
    Ok(())
}

fn apply_demux_props(
    demux: &mut Box<dyn crate::runtime::DynMultiOutputElement>,
    name: &str,
    props: &[(String, String)],
) -> Result<(), ParseError> {
    for (key, value) in props {
        let kind = demux
            .properties()
            .iter()
            .find(|s| s.name == key)
            .ok_or_else(|| ParseError::UnknownProperty {
                element: name.into(),
                key: key.clone(),
            })?
            .kind;
        let parsed = PropValue::parse(kind, value).map_err(|_| ParseError::BadValue {
            element: name.into(),
            key: key.clone(),
            value: value.clone(),
        })?;
        demux
            .set_property(key, parsed)
            .map_err(|_| ParseError::BadValue {
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
/// `decodebin`: not an element, but a macro that expands, at parse time, into the
/// chain of decoders / parsers the auto-plug search finds from its upstream caps
/// down to raw video or audio (M193).
fn is_decodebin(name: &str) -> bool {
    matches!(name, "decodebin")
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
    // Names referenced as `name.` somewhere: a `decodebin name=d` with such refs is
    // a FAN-OUT node (M482), not the inline linear case, so it is left unexpanded
    // here and handled by the decodebin-select path in `build_graph` (which probes
    // the file, demuxes, and decodes each requested port). Only the unreferenced
    // inline `... ! decodebin ! ...` expands to a linear decode chain below.
    let mut referenced: alloc::collections::BTreeSet<String> = alloc::collections::BTreeSet::new();
    for chain in &chains {
        for item in chain {
            if let Item::Ref { name, .. } = item {
                referenced.insert(name.clone());
            }
        }
    }
    let is_fanout_decodebin = |spec: &ElementSpec| {
        is_decodebin(&spec.name)
            && spec
                .instance
                .as_deref()
                .map(|n| referenced.contains(n))
                .unwrap_or(false)
    };

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
                // A fan-out `decodebin name=d` is left for `build_graph`'s
                // decodebin-select path; it is not linearly expandable.
                Item::Element(spec) if is_fanout_decodebin(&spec) => {
                    upstream = Some((spec.name.clone(), spec.props.clone()));
                    new_chain.push(Item::Element(spec));
                }
                Item::Element(spec) if is_decodebin(&spec.name) => {
                    let (pred, props) = upstream.as_ref().ok_or(ParseError::DecodebinNoUpstream)?;
                    let caps = resolve_upstream_caps(registry, pred, props)?;
                    // M746: a single-stream demux fixes its output pad before parsing
                    // any byte, so it defaults to a video port; on an audio-only
                    // container the default auto-plug would pick a video decoder and
                    // fail "no caps overlap". If a primary-stream hook sniffs the file
                    // and names the real (audio) stream, plug that demux with its
                    // stream selection and auto-plug the decoder from the elementary
                    // caps instead. A hook declines a container with a video track (the
                    // default video path is right) or one it does not parse.
                    let location = props
                        .iter()
                        .find(|(k, _)| k == "location")
                        .map(|(_, v)| v.as_str());
                    let chain_input =
                        match location.and_then(|loc| registry.primary_stream(loc, &caps)) {
                            Some(primary) => {
                                new_chain.push(Item::Element(ElementSpec {
                                    name: primary.demux.to_string(),
                                    props: primary.props.clone(),
                                    instance: None,
                                }));
                                upstream = Some((primary.demux.to_string(), primary.props));
                                primary.caps
                            }
                            None => caps,
                        };
                    let target = |c: &Caps| is_raw_video(c) || is_raw_audio(c);
                    let mut names = registry
                        .autoplug_names(&chain_input, &target, DECODEBIN_MAX_DEPTH)
                        .ok_or_else(|| {
                            ParseError::NoDecodeChain(alloc::format!("{chain_input:?}"))
                        })?;
                    // M421/M676: prepend the re-framing parser ahead of a real
                    // decode of an elementary stream, like the boxed `decodebin`
                    // splice (the caps-identity parser is invisible to the
                    // shortest-chain search, so it never appears in `names`).
                    if let Some(parser) = registry.parser_name(&chain_input) {
                        if !names.is_empty() {
                            names.insert(0, parser);
                        }
                    }
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
                Item::Ref { name, pad } => {
                    upstream = None;
                    new_chain.push(Item::Ref { name, pad });
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
/// reflected via [`SourceLoop::probe_output_caps`]; fall back to the registry's
/// declared caps (a fixed source, or a transform's source-pad template).
/// `probe_output_caps` also sniffs a `bytestream-format=auto` source's header at
/// parse time (M480), so `decodebin` picks the demuxer from the real content even
/// when the file extension is wrong; only an unreadable / unrecognized file falls
/// back to the declared default.
fn resolve_upstream_caps(
    registry: &Registry,
    name: &str,
    props: &[(String, String)],
) -> Result<Caps, ParseError> {
    if let Some(mut src) = registry.make_source(name) {
        apply_source_props(&mut src, name, props)?;
        // `probe_output_caps` may sniff the header (a `bytestream-format=auto`
        // source), so `decodebin` picks the demuxer from the real content, not a
        // mislabeled extension; it falls back to the no-I/O caps otherwise.
        if let Some(caps) = src.probe_output_caps() {
            return Ok(caps);
        }
    }
    registry
        .declared_output_caps(name)
        .ok_or(ParseError::DecodebinNoUpstream)
}

/// `uridecodebin` / `playbin`: a source-providing macro. `uridecodebin uri=X`
/// builds the source from the URI scheme handler and auto-plugs the decode chain
/// to raw; `playbin uri=X` is that plus an auto sink (`autovideosink`, or the
/// `video-sink=` override), i.e. a complete pipeline.
fn is_uri_source(name: &str) -> bool {
    matches!(name, "uridecodebin" | "playbin")
}

/// The value of a spec property by key, if present.
fn prop<'a>(spec: &'a ElementSpec, key: &str) -> Option<&'a str> {
    spec.props
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
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
            let uri =
                prop(&spec, "uri").ok_or_else(|| ParseError::MissingUri(spec.name.clone()))?;
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
                let sink = prop(&spec, "video-sink")
                    .unwrap_or("autovideosink")
                    .to_string();
                new_chain.push(Item::Element(ElementSpec {
                    name: sink,
                    props: Vec::new(),
                    instance: None,
                }));
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
        Ref { name: String, pad: String },
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
                    specs.push(ElementSpec {
                        name: name.to_string(),
                        props: Vec::new(),
                        instance: None,
                    });
                    prebuilt.push(Some(node));
                    eps.push(Endpoint::Element(ei));
                }
                Item::Ref { name, pad } => eps.push(Endpoint::Ref { name, pad }),
            }
        }
        chain_eps.push(eps);
    }

    if specs.len() < 2 {
        return Err(ParseError::TooFewStages);
    }

    // Resolve references, collect the directed links (by element index), and
    // record each demux output-pad request (M476): a head-ref `d.video_0` that
    // sources a link contributes a `PadRequest` to `demux_pads[d]`, in link (port)
    // order, so a demux-select hook can map port i to the requested stream.
    let mut raw_links: Vec<(usize, usize)> = Vec::new();
    let mut demux_pads: Vec<Vec<PadRequest>> = alloc::vec![Vec::new(); specs.len()];
    // The DESTINATION pad request per raw link (M481): a named input-pad ref
    // (`... ! mux.audio_0`) carries a request; a bare `mux.` or an inline consumer
    // carries `None` (positional). The transpose of `demux_pads` (output side).
    let mut raw_dest_req: Vec<Option<PadRequest>> = Vec::new();
    for eps in &chain_eps {
        let mut idxs: Vec<usize> = Vec::with_capacity(eps.len());
        // The pad suffix of each endpoint that is a reference (`None` for a
        // concrete element), parallel to `idxs`.
        let mut pads: Vec<Option<&str>> = Vec::with_capacity(eps.len());
        for ep in eps {
            match ep {
                Endpoint::Element(ei) => {
                    idxs.push(*ei);
                    pads.push(None);
                }
                Endpoint::Ref { name, pad } => {
                    let i = names
                        .iter()
                        .find(|(n, _)| n == name)
                        .map(|(_, i)| *i)
                        .ok_or_else(|| ParseError::UnknownReference(name.clone()))?;
                    idxs.push(i);
                    pads.push(Some(pad.as_str()));
                }
            }
        }
        for w in 0..idxs.len().saturating_sub(1) {
            let (s, d) = (idxs[w], idxs[w + 1]);
            raw_links.push((s, d));
            // Record the source's output-pad request in port order: a pad-ref
            // source (`d.video_0`) carries a named request; an inline output
            // (`d ! x`) takes the positional Nth-forwardable-stream request. Only
            // consulted for an explicit-demux fan-out node (M476); harmless noise
            // for a normal element or a tee.
            let ordinal = demux_pads[s].len();
            let req = match pads[w] {
                Some(pad) => parse_pad_request(pad, ordinal),
                None => PadRequest {
                    kind: PadKind::Any,
                    index: ordinal,
                },
            };
            demux_pads[s].push(req);
            // The destination's input-pad request: a named ref (`mux.audio_0`)
            // parses; a bare `mux.` (empty suffix) or an inline consumer is `None`
            // (positional, resolved by the sequential input counter below).
            raw_dest_req.push(match pads[w + 1] {
                Some(pad) if !pad.is_empty() => Some(parse_pad_request(pad, 0)),
                _ => None,
            });
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
    let mut queue_capacity: Vec<Option<usize>> = alloc::vec![None; specs.len()];
    for ei in 0..specs.len() {
        if is_queue(ei) {
            if raw_in[ei] != 1 || raw_out[ei] != 1 {
                return Err(ParseError::QueueRole(specs[ei].name.clone()));
            }
            queue_policy[ei] = queue_leaky_policy(&specs[ei]);
            queue_capacity[ei] = queue_capacity_of(&specs[ei]);
            queue_succ[ei] = raw_links.iter().find(|(s, _)| *s == ei).map(|(_, d)| *d);
        }
    }
    // Each edge whose source is a real element walks through any run of queues to
    // its terminal consumer, carrying the accumulated policy; edges out of a queue
    // are consumed by that walk (skipped here).
    let mut links: Vec<(usize, usize, LinkPolicy, Option<usize>)> = Vec::new();
    // The destination input-pad request per contracted link (M481), aligned with
    // `links`; taken from the raw link that lands on the terminal consumer (so a
    // `... ! queue ! mux.audio_0` keeps its named pad through the queue contraction).
    let mut link_dest_req: Vec<Option<PadRequest>> = Vec::new();
    for (li, &(s, d)) in raw_links.iter().enumerate() {
        if is_queue(s) {
            continue;
        }
        let (mut cur, mut src_li) = (d, li);
        let mut policy = LinkPolicy::Block;
        let mut capacity: Option<usize> = None;
        while is_queue(cur) {
            if queue_policy[cur] != LinkPolicy::Block {
                policy = queue_policy[cur];
            }
            // A run of queues carries the last explicit depth (rare to chain them).
            if let Some(c) = queue_capacity[cur] {
                capacity = Some(c);
            }
            let next = queue_succ[cur].expect("queue validated 1-out above");
            // The named-pad suffix lives on the raw link that enters the terminal
            // consumer, i.e. `(cur -> next)`; find it for the request.
            src_li = raw_links
                .iter()
                .position(|&(a, b)| a == cur && b == next)
                .unwrap_or(src_li);
            cur = next;
        }
        links.push((s, cur, policy, capacity));
        link_dest_req.push(raw_dest_req[src_li].clone());
    }

    // Link degree per element fixes its role and any tee's output width. Computed
    // over the contracted links, so queue indices drop to degree 0 and are skipped
    // as nodes below.
    let mut in_deg = alloc::vec![0usize; specs.len()];
    let mut out_deg = alloc::vec![0usize; specs.len()];
    for &(s, d, _, _) in &links {
        out_deg[s] += 1;
        in_deg[d] += 1;
    }

    let is_tee = |ei: usize| specs[ei].name == "tee";
    // A non-tee node with several inbound links is a muxer (built from the
    // registry with that input count); a tee has a single input pad.
    let is_muxer = |ei: usize| !is_tee(ei) && in_deg[ei] > 1;
    // A node registered as a demuxer with several outbound links is a fan-out
    // demux (M210): the transpose of a muxer. A registered name with one output
    // falls back to its single-output launch element (e.g. `tsdemux`), the way a
    // one-input muxer name falls back to its single-input element.
    let is_demux = |ei: usize| !is_tee(ei) && out_deg[ei] > 1 && registry.is_demux(&specs[ei].name);
    // Explicit-demux fan-out (M476): a non-tee, non-registered-demux element that
    // fans out to several pads and is fed by a file source is built by a registered
    // demux-select hook, which probes the file (`location=`) and returns a
    // multi-output demuxer with one port per pad request (in reference order). This
    // is how `matroskademux` / `tsdemux` / `qtdemux` in a launch line split a file
    // into its elementary streams, honoring `d.video_0` / `d.audio_0` selection.
    let mut demux_select_node: Vec<Option<Box<dyn crate::runtime::DynMultiOutputElement>>> =
        (0..specs.len()).map(|_| None).collect();
    if !registry.demux_select_hooks().is_empty() {
        for ei in 0..specs.len() {
            if is_tee(ei) || is_demux(ei) || out_deg[ei] <= 1 || prebuilt[ei].is_some() {
                continue;
            }
            // The upstream file location (the source linking into this demux).
            let Some(location) = links
                .iter()
                .find(|(_, d, _, _)| *d == ei)
                .and_then(|(s, _, _, _)| prop(&specs[*s], "location"))
            else {
                continue;
            };
            for hook in registry.demux_select_hooks() {
                if let Some(demux) = hook(&specs[ei].name, location, &demux_pads[ei]) {
                    demux_select_node[ei] = Some(demux);
                    break;
                }
            }
        }
    }
    // `decodebin name=d` fan-out (M482): a `decodebin` node left unexpanded (it has
    // named refs) with a file source upstream probes the file, builds the
    // multi-output demuxer (stored like a demux-select node so it fans out on its
    // own pads), and records each port's elementary caps so the wiring below splices
    // a decoder onto every port (the decode-per-port that makes it `decodebin`, not a
    // bare demuxer). Declining hooks leave it unbuilt (a loud error at node build).
    let mut decode_fanout_caps: Vec<Option<Vec<Caps>>> = (0..specs.len()).map(|_| None).collect();
    if !registry.decodebin_select_hooks().is_empty() {
        for ei in 0..specs.len() {
            if !is_decodebin(&specs[ei].name) || out_deg[ei] == 0 || prebuilt[ei].is_some() {
                continue;
            }
            let Some(location) = links
                .iter()
                .find(|(_, d, _, _)| *d == ei)
                .and_then(|(s, _, _, _)| prop(&specs[*s], "location"))
            else {
                continue;
            };
            for hook in registry.decodebin_select_hooks() {
                if let Some((demux, caps)) = hook(location, &demux_pads[ei]) {
                    demux_select_node[ei] = Some(demux);
                    decode_fanout_caps[ei] = Some(caps);
                    break;
                }
            }
        }
    }
    let is_select: Vec<bool> = demux_select_node.iter().map(|d| d.is_some()).collect();
    // Auto-tee (M473): a non-tee, non-demux node whose single output fans out to
    // several consumers gets an implicit `tee` spliced in below, so a gst-launch
    // line that omits the explicit tee still builds. `tee`, registered demuxers,
    // and explicit-demux fan-out nodes fan out on their own pads and are left alone.
    let needs_tee = |ei: usize| {
        !is_tee(ei)
            && !is_demux(ei)
            && !is_select[ei]
            && !registry.is_fanout_src(&specs[ei].name)
            && out_deg[ei] > 1
    };
    // A fan-in element with no output is checked at construction below: a
    // terminal session (`is_terminal`) legally ends the graph (M713), a
    // merging muxer without a downstream stays `MuxerWithoutOutput`.
    for (ei, spec) in specs.iter().enumerate() {
        if is_tee(ei) && !spec.props.is_empty() {
            // The structural tee carries no element, so it has no properties.
            return Err(ParseError::UnknownProperty {
                element: "tee".to_string(),
                key: spec.props[0].0.clone(),
            });
        }
    }

    // Construct nodes in element-index order so `node_of[ei]` lines up. Queue
    // indices were contracted into edge policies above, so they get no node
    // (`None`); they never appear as an endpoint in the contracted links.
    let mut graph: Graph<GraphNode> = Graph::new();
    let mut node_of: Vec<Option<NodeId>> = Vec::with_capacity(specs.len());
    // The resolved muxer input-pad index per contracted link (M481), aligned with
    // `links`; filled at muxer construction from the element's `input_pad_index`
    // scheme. `None` for a non-muxer link or an unnamed ref (sequential fallback).
    let mut mux_pad_of_link: Vec<Option<u8>> = alloc::vec![None; links.len()];
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
        } else if in_deg[ei] == 0 && registry.is_fanout_src(&spec.name) {
            // Terminal fan-out source (M727): 0 inputs, one output per named
            // pad reference (`s. ! ...`). The element's intrinsic port count
            // must match the linked outputs.
            let mut src = registry
                .make_fanout_src(&spec.name, out_deg[ei])
                .ok_or_else(|| ParseError::UnknownElement(spec.name.clone()))?;
            if src.output_count() != out_deg[ei] {
                return Err(ParseError::UnknownInputPad(spec.name.clone()));
            }
            apply_fanout_src_props(&mut src, &spec.name, &spec.props)?;
            graph
                .add_fanout_src(GraphNodeRef::FanoutSource(src), out_deg[ei] as u8)
                .node()
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
            // Resolve named input-pad refs (M481) to concrete indices via the
            // muxer's own scheme, so `... ! mux.audio_0  ... ! mux.video_0` routes
            // by name regardless of order. Named refs claim their index first;
            // bare refs fill the remaining slots in link order (the historical
            // positional behavior).
            let n = in_deg[ei];
            let incoming: Vec<usize> = links
                .iter()
                .enumerate()
                .filter(|(_, (_, d, _, _))| *d == ei)
                .map(|(k, _)| k)
                .collect();
            let mut used = alloc::vec![false; n];
            for (ord, &k) in incoming.iter().enumerate() {
                let Some(req) = &link_dest_req[k] else {
                    continue;
                };
                let idx = mux
                    .input_pad_index(req, ord)
                    .filter(|&i| i < n)
                    .ok_or_else(|| ParseError::UnknownInputPad(spec.name.clone()))?;
                if core::mem::replace(&mut used[idx], true) {
                    return Err(ParseError::DuplicateInputPad(spec.name.clone()));
                }
                mux_pad_of_link[k] = Some(idx as u8);
            }
            for &k in &incoming {
                if link_dest_req[k].is_none() {
                    let idx = used
                        .iter()
                        .position(|u| !u)
                        .expect("in_deg matches link count");
                    used[idx] = true;
                    mux_pad_of_link[k] = Some(idx as u8);
                }
            }
            if out_deg[ei] == 0 {
                // Nothing downstream: legal only for a terminal fan-in session
                // (M713), whose element consumes its inputs with no merged
                // output. A merging muxer here would silently drop its output.
                if !mux.is_terminal() {
                    return Err(ParseError::MuxerWithoutOutput(spec.name.clone()));
                }
                graph
                    .add_fanin_sink(GraphNodeRef::Muxer(mux), in_deg[ei] as u8)
                    .node()
            } else {
                graph
                    .add_muxer(GraphNodeRef::Muxer(mux), in_deg[ei] as u8)
                    .node()
            }
        } else if is_select[ei] {
            // M476: a demux-select hook already built the multi-output demuxer
            // (probing the upstream file); splice it in with one port per pad.
            let demux = demux_select_node[ei]
                .take()
                .expect("select demux built above");
            graph
                .add_demux(GraphNodeRef::Demux(demux), out_deg[ei] as u8)
                .node()
        } else if is_demux(ei) {
            let mut demux = registry
                .make_demux(&spec.name, out_deg[ei])
                .ok_or_else(|| ParseError::UnknownElement(spec.name.clone()))?;
            apply_demux_props(&mut demux, &spec.name, &spec.props)?;
            graph
                .add_demux(GraphNodeRef::Demux(demux), out_deg[ei] as u8)
                .node()
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

    // Auto-tee (M473): for each fan-out node that is not itself a tee/demux, splice
    // an implicit `tee` onto its output. The node's single output pad feeds the
    // tee (a plain blocking link); the tee's `out_deg` pads feed the consumers,
    // which the edge loop below sources from the tee instead of the node.
    let mut implicit_tee: Vec<Option<NodeId>> = alloc::vec![None; specs.len()];
    for ei in 0..specs.len() {
        if needs_tee(ei) {
            if let Some(src_node) = node_of[ei] {
                let tee = graph.add_tee(out_deg[ei] as u8).node();
                graph.link_with(PadId::from(src_node), PadId::from(tee), LinkPolicy::Block)?;
                implicit_tee[ei] = Some(tee);
            }
        }
    }

    // Wire edges. Each tee or demux branch takes a distinct output pad (0..n) and
    // each muxer input a distinct input pad (0..n); every other output and input
    // is pad 0. A queue's `leaky=` rides along as the edge's `LinkPolicy`.
    let mut tee_next = alloc::vec![0u8; specs.len()];
    for (k, &(s, d, policy, capacity)) in links.iter().enumerate() {
        let node_s = node_of[s].expect("contracted link source is a real node");
        let node_d = node_of[d].expect("contracted link destination is a real node");
        let src = if let Some(tee) = implicit_tee[s] {
            // The fan-out node's consumers source from its spliced-in tee's pads.
            let index = tee_next[s];
            tee_next[s] += 1;
            PadId { node: tee, index }
        } else if is_tee(s) || is_demux(s) || is_select[s] || registry.is_fanout_src(&specs[s].name)
        {
            let index = tee_next[s];
            tee_next[s] += 1;
            PadId {
                node: node_s,
                index,
            }
        } else {
            PadId::from(node_s)
        };
        let dst = if is_muxer(d) {
            // The input index was resolved at construction (named pads via the
            // muxer's scheme, bare refs sequentially); every muxer link has one.
            let index = mux_pad_of_link[k].expect("muxer link assigned an input pad");
            PadId {
                node: node_d,
                index,
            }
        } else {
            PadId::from(node_d)
        };
        // `decodebin` fan-out (M482): splice a decoder between the demux port and the
        // branch consumer instead of a bare link, so each `d.video_0` / `d.audio_0`
        // branch receives DECODED (raw) frames. A text port carries `Text{Utf8}`
        // already, so it links straight through (no codec).
        if let Some(caps) = &decode_fanout_caps[s] {
            let port = src.index as usize;
            let kind = demux_pads[s]
                .get(port)
                .map(|r| r.kind)
                .unwrap_or(PadKind::Any);
            if !matches!(kind, PadKind::Text) {
                let target: &dyn Fn(&Caps) -> bool = match kind {
                    PadKind::Audio => &is_raw_audio,
                    _ => &is_raw_video,
                };
                registry
                    .decodebin(
                        &mut graph,
                        src,
                        dst,
                        &caps[port],
                        target,
                        DECODEBIN_MAX_DEPTH,
                    )
                    .map_err(|_| ParseError::NoDecodeChain(alloc::format!("{:?}", caps[port])))?;
                continue;
            }
        }
        graph.link_full(src, dst, policy, capacity)?;
    }

    Ok(graph)
}

/// Map a `queue` / `queue2` node's `leaky=` property to the edge backpressure
/// policy it stands in for (M190). gst accepts the enum by value or nick:
/// `0`/`no` (lossless, the default), `1`/`upstream` (drop the newest incoming
/// buffer), `2`/`downstream` (drop the oldest queued buffer). The other buffering
/// bounds (`max-size-bytes` / `max-size-time`, `min-threshold-*`, `silent`) are
/// accepted but not modeled (g2g's link depth is a buffer count), so they are
/// ignored for paste compatibility rather than rejected; `max-size-buffers` maps
/// to the edge depth, see [`queue_capacity_of`].
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

/// A `queue max-size-buffers=N` sets the depth of the edge it contracts to (the
/// gst per-queue buffer bound), overriding the runner's graph-wide `link_capacity`
/// for just that link. `0` in gst means "unbounded"; g2g has no unbounded channel,
/// so a `0` is ignored (falls back to the default depth). An unparseable value is
/// ignored too (paste compatibility).
fn queue_capacity_of(spec: &ElementSpec) -> Option<usize> {
    spec.props
        .iter()
        .find(|(k, _)| k == "max-size-buffers")
        .and_then(|(_, v)| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
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
    // playbin uri=X auto-fan-out (M382): a lone `playbin uri=` probes the
    // container via the registered hook and auto-builds source -> demux ->
    // per-stream decode -> auto sinks (multi-stream). Without a hook (or if it
    // declines, e.g. a non-Matroska file), fall through to build_graph, which
    // expands `playbin` to the single-stream pipeline (M196).
    if let Some(uri) = lone_playbin_uri(&chains) {
        // Try each registered hook (one per container type) until one handles the
        // URI; a hook returns Ok(None) to decline a container it does not parse.
        for hook in registry.playbin_hooks() {
            if let Some(graph) = hook(registry, uri)? {
                return Ok(graph);
            }
        }
    }
    build_graph(registry, chains)
}

/// The `uri=` of a pipeline that is a single bare `playbin uri=X` element (and
/// nothing else), the M382 multi-stream auto-fan-out trigger. `None` for any
/// other shape: a `playbin` mid-pipeline, alongside other elements, or without a
/// `uri=` is left to the normal builder (the M196 single-stream expansion).
fn lone_playbin_uri(chains: &[Chain]) -> Option<&str> {
    let [chain] = chains else { return None };
    let [Item::Element(spec)] = chain.as_slice() else {
        return None;
    };
    if spec.name != "playbin" {
        return None;
    }
    prop(spec, "uri")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item_names(chain: &Chain) -> Vec<&str> {
        chain
            .iter()
            .map(|i| match i {
                Item::Element(s) => s.name.as_str(),
                Item::Ref { name, .. } => name.as_str(),
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
        assert_eq!(
            item_names(&chains[0]),
            ["videotestsrc", "videoflip", "fakesink"]
        );
        let Item::Element(src) = &chains[0][0] else {
            panic!("first is an element")
        };
        assert_eq!(
            src.props,
            [
                ("num-buffers".to_string(), "3".to_string()),
                ("pattern".into(), "snow".into())
            ]
        );
        let Item::Element(sink) = &chains[0][2] else {
            panic!("last is an element")
        };
        assert!(sink.props.is_empty());
    }

    #[test]
    fn parse_chains_strips_quoted_values() {
        // A double-quoted value has its quotes stripped.
        let chains = parse_chains("filesrc location=\"file.mp4\" ! fakesink").unwrap();
        let Item::Element(src) = &chains[0][0] else {
            panic!("element")
        };
        assert_eq!(
            src.props[0],
            ("location".to_string(), "file.mp4".to_string())
        );
    }

    #[test]
    fn parse_chains_keeps_spaces_in_quoted_values() {
        // The quote-aware tokenizer keeps spaces inside a value, so a nested
        // element description (the `gstwrap` case) survives as one property.
        let chains = parse_chains("gstwrap element=\"x264enc bitrate=4000\" ! fakesink").unwrap();
        assert_eq!(item_names(&chains[0]), ["gstwrap", "fakesink"]);
        let Item::Element(w) = &chains[0][0] else {
            panic!("element")
        };
        assert_eq!(
            w.props[0],
            ("element".to_string(), "x264enc bitrate=4000".to_string())
        );
    }

    #[test]
    fn parse_chains_keeps_bang_inside_quotes() {
        // A `!` inside a quoted value is literal, not a stage separator: one
        // element with one property, not two chained nodes.
        let chains = parse_chains("gstwrap element=\"a ! b\" ! fakesink").unwrap();
        assert_eq!(item_names(&chains[0]), ["gstwrap", "fakesink"]);
        let Item::Element(w) = &chains[0][0] else {
            panic!("element")
        };
        assert_eq!(w.props[0], ("element".to_string(), "a ! b".to_string()));
    }

    #[test]
    fn tokenize_treats_quoted_region_as_one_token() {
        assert_eq!(
            tokenize("gstwrap element=\"x y\" ! sink"),
            ["gstwrap", "element=\"x y\"", "!", "sink"]
        );
    }

    #[test]
    fn tokenize_treats_single_quoted_region_as_one_token() {
        assert_eq!(
            tokenize("filesink location='/my file.ts' ! sink"),
            ["filesink", "location='/my file.ts'", "!", "sink"]
        );
    }

    #[test]
    fn single_quoted_value_is_unquoted() {
        let chains = parse_chains("videotestsrc ! identity note='a b c' ! fakesink").unwrap();
        let Item::Element(id) = &chains[0][1] else {
            panic!("element")
        };
        assert_eq!(id.props, [("note".to_string(), "a b c".to_string())]);
    }

    #[test]
    fn hash_starts_a_comment_to_end_of_line() {
        // Trailing comment on one line, and a comment mid-pipeline across lines.
        let chains = parse_chains("videotestsrc ! fakesink # trailing note").unwrap();
        assert_eq!(item_names(&chains[0]), ["videotestsrc", "fakesink"]);
        let chains = parse_chains("videotestsrc  # the source\n  ! fakesink").unwrap();
        assert_eq!(item_names(&chains[0]), ["videotestsrc", "fakesink"]);
    }

    #[test]
    fn hash_inside_a_value_is_literal_not_a_comment() {
        // A URI fragment (`#closed-captions=cc1`, `#t=10`) must survive; `#` is a
        // comment only at a token boundary.
        assert_eq!(
            tokenize("uridecodebin uri=file:///v.mp4#closed-captions=cc1 ! sink"),
            [
                "uridecodebin",
                "uri=file:///v.mp4#closed-captions=cc1",
                "!",
                "sink"
            ]
        );
    }

    #[test]
    fn caps_range_value_error_carries_a_syntax_hint() {
        let e = ParseError::BadValue {
            element: "capsfilter".to_string(),
            key: "caps".to_string(),
            value: "video/x-raw,width=[1,1920]".to_string(),
        };
        let msg = alloc::format!("{e}");
        assert!(msg.contains("ranges"), "caps range hint present: {msg}");
        // A plain element property error keeps the bare message (no caps hint).
        let plain = ParseError::BadValue {
            element: "videobox".to_string(),
            key: "top".to_string(),
            value: "abc".to_string(),
        };
        assert!(!alloc::format!("{plain}").contains("ranges"));
    }

    #[test]
    fn caps_description_becomes_capsfilter() {
        // A bare `media/type,...` node is the inline caps-filter shorthand.
        let chains =
            parse_chains("videotestsrc ! video/x-raw,format=nv12,width=320 ! fakesink").unwrap();
        assert_eq!(
            item_names(&chains[0]),
            ["videotestsrc", "capsfilter", "fakesink"]
        );
        let Item::Element(caps) = &chains[0][1] else {
            panic!("element")
        };
        assert_eq!(
            caps.props,
            [(
                "caps".to_string(),
                "video/x-raw,format=nv12,width=320".to_string()
            )]
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
        let Item::Element(tee) = &chains[0][1] else {
            panic!("element")
        };
        assert_eq!(tee.instance.as_deref(), Some("t"));
        assert!(tee.props.is_empty(), "name= is the handle, not a property");
        assert!(matches!(&chains[1][0], Item::Ref { name, .. } if name == "t"));
    }

    #[test]
    fn empty_and_too_few_stages_error() {
        let reg = Registry::new();
        assert!(matches!(parse_launch(&reg, "   "), Err(ParseError::Empty)));
        assert!(matches!(
            parse_launch(&reg, "videotestsrc"),
            Err(ParseError::TooFewStages)
        ));
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
        let err = parse_launch(
            &reg,
            "videotestsrc ! tee name=t ! fakesink nope. ! fakesink",
        )
        .unwrap_err();
        assert_eq!(err, ParseError::UnknownReference("nope".to_string()));
    }

    #[test]
    fn duplicate_name_is_reported() {
        let reg = Registry::new();
        let err =
            parse_launch(&reg, "videotestsrc name=x ! videoflip name=x ! fakesink").unwrap_err();
        assert_eq!(err, ParseError::DuplicateName("x".to_string()));
    }

    // Auto-tee (M473): fan-out without an explicit `tee` no longer errors; the
    // parser splices one in. Covered end to end (build + run + topology) in
    // g2g-plugins/tests/m118_launch_branching.rs, where real elements exist.
}
